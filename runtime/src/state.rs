//! Hybrid 7:1 Recurrent + Anchor GQA State Manager (`schemas/terra_format.md` Sections 4.2 & 4.3).
//!
//! - `GatedLinearRecurrence` layers (`layer_kinds[l] == 1`) maintain a strictly fixed-size $O(1)$
//!   state tensor `S` of shape `[H, d, d]` (`Vec<f32>`) whose byte footprint never grows with sequence length.
//! - `AnchorGQA` layers (`layer_kinds[l] == 0`) maintain a contiguous KV cache `[seq_len, H_kv, d]`
//!   exclusively for those anchor layers.

use crate::format::LayerKind;
use crate::kernels::softmax_in_place;
use half::f16;

#[derive(Debug, Clone)]
pub struct RecurrentLayerState {
    /// Row-major `[H, d, d]` recurrent state matrix $S_t$.
    pub state: Vec<f32>,
    /// Pre-allocated fixed-size rollback snapshot `[H, d, d]` (zero runtime allocation).
    backup: Vec<f32>,
    /// True when `state` has been mutated in the current uncommitted token step.
    dirty: bool,
}

impl RecurrentLayerState {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        let len = num_heads * head_dim * head_dim;
        let mut state = Vec::with_capacity(len);
        state.resize(len, 0.0f32);
        let mut backup = Vec::with_capacity(len);
        backup.resize(len, 0.0f32);
        Self {
            state,
            backup,
            dirty: false,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.backup.fill(0.0);
        self.dirty = false;
    }

    /// Logical `[H, d, d]` recurrent state size in bytes.
    #[inline]
    pub fn state_bytes(&self) -> usize {
        self.state.capacity() * std::mem::size_of::<f32>()
    }

    /// Total physical heap bytes owned by this recurrent layer (`state` + rollback `backup`).
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        (self.state.capacity() + self.backup.capacity()) * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Clone)]
pub struct AnchorKvCache {
    /// Contiguous `[seq_len, H_kv, d]` RoPE-rotated key cache stored as IEEE-754 `f16` bits (`u16`).
    pub k_cache: Vec<u16>,
    /// Contiguous `[seq_len, H_kv, d]` value cache stored as IEEE-754 `f16` bits (`u16`).
    pub v_cache: Vec<u16>,
    pub seq_len: usize,
    pub max_tokens: usize,
}

impl AnchorKvCache {
    pub fn new(max_tokens: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let stride = num_kv_heads * head_dim;
        let cap = max_tokens * stride;
        let mut k_cache = Vec::new();
        k_cache.reserve_exact(cap);
        let mut v_cache = Vec::new();
        v_cache.reserve_exact(cap);
        Self {
            k_cache,
            v_cache,
            seq_len: 0,
            max_tokens,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.k_cache.clear();
        self.v_cache.clear();
        self.seq_len = 0;
    }

    #[inline]
    pub fn live_bytes(&self) -> usize {
        (self.k_cache.len() + self.v_cache.len()) * std::mem::size_of::<u16>()
    }

    #[inline]
    pub fn capacity_bytes(&self) -> usize {
        (self.k_cache.capacity() + self.v_cache.capacity()) * std::mem::size_of::<u16>()
    }
}

#[derive(Debug, Clone)]
pub enum LayerState {
    AnchorGqa(AnchorKvCache),
    Recurrent(RecurrentLayerState),
}

#[derive(Debug, Clone)]
pub struct HybridStateManager {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_kv_tokens: usize,
    rope_inv_freq: Vec<f32>,
    layers: Vec<LayerState>,
    scores_scratch: Vec<f32>,
    seq_len: usize,
}

impl HybridStateManager {
    pub fn new(
        layer_kinds: &[LayerKind],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        max_kv_tokens: usize,
    ) -> Self {
        assert!(num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0);
        assert!(head_dim > 0 && head_dim % 2 == 0);

        let half_d = head_dim / 2;
        let rope_inv_freq: Vec<f32> = (0..half_d)
            .map(|j| rope_theta.powf(-(2.0f32 * (j as f32)) / (head_dim as f32)))
            .collect();

        let layers = layer_kinds
            .iter()
            .map(|&kind| match kind {
                LayerKind::AnchorGqa => LayerState::AnchorGqa(AnchorKvCache::new(
                    max_kv_tokens,
                    num_kv_heads,
                    head_dim,
                )),
                LayerKind::GatedLinearRecurrence => {
                    LayerState::Recurrent(RecurrentLayerState::new(num_heads, head_dim))
                }
            })
            .collect();

        let mut scores_scratch = Vec::new();
        scores_scratch.reserve_exact(max_kv_tokens);

        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            max_kv_tokens,
            rope_inv_freq,
            layers,
            scores_scratch,
            seq_len: 0,
        }
    }

    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    #[inline]
    pub fn max_kv_tokens(&self) -> usize {
        self.max_kv_tokens
    }

    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn num_recurrent_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l, LayerState::Recurrent(_)))
            .count()
    }

    pub fn num_anchor_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l, LayerState::AnchorGqa(_)))
            .count()
    }

    pub fn layer_state(&self, layer_idx: usize) -> &LayerState {
        &self.layers[layer_idx]
    }

    /// Exact fixed logical bytes used by all $O(1)$ recurrent state tensors `[H, d, d]`.
    pub fn recurrent_state_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::Recurrent(r) => r.state_bytes(),
                LayerState::AnchorGqa(_) => 0,
            })
            .sum()
    }

    /// Total heap bytes used by recurrent layers including transactional rollback buffers.
    pub fn recurrent_allocated_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::Recurrent(r) => r.memory_bytes(),
                LayerState::AnchorGqa(_) => 0,
            })
            .sum()
    }

    /// Live payload bytes stored in anchor-layer KV caches (`2 * seq_len * H_kv * d * 2` per anchor layer).
    pub fn anchor_kv_live_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::AnchorGqa(kv) => kv.live_bytes(),
                LayerState::Recurrent(_) => 0,
            })
            .sum()
    }

    /// Allocated capacity bytes of anchor-layer KV caches.
    pub fn anchor_kv_capacity_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::AnchorGqa(kv) => kv.capacity_bytes(),
                LayerState::Recurrent(_) => 0,
            })
            .sum()
    }

    /// Total heap bytes currently owned by `HybridStateManager` (recurrent states + rollback buffers + anchor KV + RoPE/scratch).
    pub fn memory_bytes(&self) -> usize {
        self.recurrent_allocated_bytes()
            + self.anchor_kv_capacity_bytes()
            + (self.rope_inv_freq.capacity() + self.scores_scratch.capacity())
                * std::mem::size_of::<f32>()
    }

    /// Projected RAM budget reservation for `target_seq_len` tokens (used when sizing `ExpertLruPool`).
    pub fn projected_bytes_for_seq_len(&self, target_seq_len: usize) -> usize {
        let anchor_count = self.num_anchor_layers();
        let kv_per_anchor =
            2 * target_seq_len * self.num_kv_heads * self.head_dim * std::mem::size_of::<u16>();
        self.recurrent_allocated_bytes()
            + anchor_count * kv_per_anchor
            + self.rope_inv_freq.len() * std::mem::size_of::<f32>()
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            match layer {
                LayerState::AnchorGqa(kv) => kv.reset(),
                LayerState::Recurrent(rec) => rec.reset(),
            }
        }
        self.seq_len = 0;
    }

    /// Commits a completed forward token step: clears recurrent dirty flags and advances `seq_len`.
    #[inline]
    pub fn commit_token(&mut self) {
        let next_seq = self.seq_len + 1;
        for layer in &mut self.layers {
            match layer {
                LayerState::AnchorGqa(kv) => {
                    debug_assert_eq!(kv.seq_len, next_seq);
                }
                LayerState::Recurrent(rec) => {
                    rec.dirty = false;
                }
            }
        }
        self.seq_len = next_seq;
    }

    #[inline]
    pub fn advance_token(&mut self) {
        self.commit_token();
    }

    /// Restores every anchor KV cache and mutated recurrent state to the exact pre-token snapshot.
    pub fn rollback_token(&mut self) {
        let target_seq = self.seq_len;
        let kv_stride = self.num_kv_heads * self.head_dim;
        let target_len = target_seq * kv_stride;
        for layer in &mut self.layers {
            match layer {
                LayerState::AnchorGqa(kv) => {
                    if kv.seq_len > target_seq {
                        kv.k_cache.truncate(target_len);
                        kv.v_cache.truncate(target_len);
                        kv.seq_len = target_seq;
                    }
                }
                LayerState::Recurrent(rec) => {
                    if rec.dirty {
                        rec.state.copy_from_slice(&rec.backup);
                        rec.dirty = false;
                    }
                }
            }
        }
    }

    /// Applies half-split RoPE (`schemas/terra_format.md` Section 4.2) at token position `pos`
    /// to `heads_flat` of shape `[count_heads, d]`.
    pub fn apply_half_split_rope_in_place(
        &self,
        heads_flat: &mut [f32],
        count_heads: usize,
        pos: usize,
    ) {
        let d = self.head_dim;
        let half = d / 2;
        assert_eq!(heads_flat.len(), count_heads * d);
        let pos_f = pos as f32;

        for j in 0..half {
            let theta = pos_f * self.rope_inv_freq[j];
            let (sin_t, cos_t) = theta.sin_cos();
            for h in 0..count_heads {
                let base = h * d;
                let z0 = heads_flat[base + j];
                let z1 = heads_flat[base + half + j];
                heads_flat[base + j] = z0 * cos_t - z1 * sin_t;
                heads_flat[base + half + j] = z1 * cos_t + z0 * sin_t;
            }
        }
    }

    /// Executes Section 4.2 Anchor GQA step for `layer_idx` at current token position `self.seq_len`:
    /// applies half-split RoPE to `q_flat` and `k_flat`, appends `(k_rope, v_flat)` to the anchor KV cache,
    /// and writes concatenated causal attention head outputs `[H * d]` to `out_attn`.
    pub fn step_anchor_gqa(
        &mut self,
        layer_idx: usize,
        q_flat: &mut [f32],
        k_flat: &mut [f32],
        v_flat: &[f32],
        out_attn: &mut [f32],
    ) {
        let h_count = self.num_heads;
        let h_kv_count = self.num_kv_heads;
        let d = self.head_dim;
        assert_eq!(q_flat.len(), h_count * d);
        assert_eq!(k_flat.len(), h_kv_count * d);
        assert_eq!(v_flat.len(), h_kv_count * d);
        assert_eq!(out_attn.len(), h_count * d);

        let pos = self.seq_len;
        self.apply_half_split_rope_in_place(q_flat, h_count, pos);
        self.apply_half_split_rope_in_place(k_flat, h_kv_count, pos);

        let kv = match &mut self.layers[layer_idx] {
            LayerState::AnchorGqa(kv) => kv,
            LayerState::Recurrent(_) => {
                panic!("step_anchor_gqa called on recurrent layer {layer_idx}")
            }
        };
        assert_eq!(
            kv.seq_len, pos,
            "anchor layer {layer_idx} seq_len ({}) out of sync with manager seq_len ({pos})",
            kv.seq_len
        );

        assert!(
            pos < kv.max_tokens,
            "anchor layer {layer_idx} token position {pos} exceeds reserved max_tokens ({})",
            kv.max_tokens
        );
        let cap_before = kv.k_cache.capacity();
        kv.k_cache.extend(k_flat.iter().map(|&v| f16::from_f32(v).to_bits()));
        kv.v_cache.extend(v_flat.iter().map(|&v| f16::from_f32(v).to_bits()));
        debug_assert_eq!(
            kv.k_cache.capacity(),
            cap_before,
            "AnchorKvCache unexpectedly reallocated beyond reserved RAM budget"
        );
        kv.seq_len = pos + 1;

        let total_t = kv.seq_len;
        if self.scores_scratch.len() < total_t {
            self.scores_scratch.resize(total_t, 0.0);
        }
        let scores = &mut self.scores_scratch[..total_t];

        let kv_stride = h_kv_count * d;
        let group_size = h_count / h_kv_count;
        let scale = 1.0f64 / (d as f64).sqrt();

        for h in 0..h_count {
            let h_kv = h / group_size;
            let q_h = &q_flat[h * d..(h + 1) * d];

            for tau in 0..total_t {
                let k_off = tau * kv_stride + h_kv * d;
                let k_tau = &kv.k_cache[k_off..k_off + d];
                let mut dot = 0.0f64;
                for j in 0..d {
                    dot += (q_h[j] as f64) * (f16::from_bits(k_tau[j]).to_f32() as f64);
                }
                scores[tau] = (dot * scale) as f32;
            }

            softmax_in_place(scores);

            let out_h = &mut out_attn[h * d..(h + 1) * d];
            out_h.fill(0.0);
            for tau in 0..total_t {
                let p = scores[tau];
                let v_off = tau * kv_stride + h_kv * d;
                let v_tau = &kv.v_cache[v_off..v_off + d];
                for j in 0..d {
                    out_h[j] += p * f16::from_bits(v_tau[j]).to_f32();
                }
            }
        }
    }

    /// Executes Section 4.3 Gated Linear Recurrence update for `layer_idx`:
    /// `S_t = alpha_t * S_{t-1} + k_hat_t * v_t^T`, `o_t = S_t^T * q_hat_t` (no RoPE, $O(1)$ memory).
    pub fn step_recurrent(
        &mut self,
        layer_idx: usize,
        q_flat: &[f32],
        k_flat: &[f32],
        v_flat: &[f32],
        alpha: &[f32],
        out_rec: &mut [f32],
    ) {
        let h_count = self.num_heads;
        let h_kv_count = self.num_kv_heads;
        let d = self.head_dim;
        assert_eq!(q_flat.len(), h_count * d);
        assert_eq!(k_flat.len(), h_kv_count * d);
        assert_eq!(v_flat.len(), h_kv_count * d);
        assert_eq!(alpha.len(), h_count);
        assert_eq!(out_rec.len(), h_count * d);

        let rec = match &mut self.layers[layer_idx] {
            LayerState::Recurrent(r) => r,
            LayerState::AnchorGqa(_) => {
                panic!("step_recurrent called on anchor GQA layer {layer_idx}")
            }
        };

        if !rec.dirty {
            rec.backup.copy_from_slice(&rec.state);
            rec.dirty = true;
        }

        let group_size = h_count / h_kv_count;
        let inv_sqrt_d = 1.0f32 / (d as f32).sqrt();
        let head_state_len = d * d;

        for h in 0..h_count {
            let h_kv = h / group_size;
            let a_h = alpha[h];
            let q_h = &q_flat[h * d..(h + 1) * d];
            let k_h = &k_flat[h_kv * d..(h_kv + 1) * d];
            let v_h = &v_flat[h_kv * d..(h_kv + 1) * d];
            let s_mat = &mut rec.state[h * head_state_len..(h + 1) * head_state_len];
            let out_h = &mut out_rec[h * d..(h + 1) * d];
            out_h.fill(0.0);

            for i in 0..d {
                let k_hat_i = k_h[i] * inv_sqrt_d;
                let q_hat_i = q_h[i] * inv_sqrt_d;
                let s_row = &mut s_mat[i * d..(i + 1) * d];
                for j in 0..d {
                    let updated = a_h * s_row[j] + k_hat_i * v_h[j];
                    s_row[j] = updated;
                    out_h[j] += updated * q_hat_i;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recurrent_state_strictly_o1_over_2048_tokens_and_anchor_only_kv() {
        // 16-layer 7:1 hybrid schedule: layers 0..6, 8..14 Recurrent; layers 7, 15 AnchorGQA
        let mut kinds = vec![LayerKind::GatedLinearRecurrence; 16];
        kinds[7] = LayerKind::AnchorGqa;
        kinds[15] = LayerKind::AnchorGqa;

        let h = 4;
        let h_kv = 2;
        let d = 16;
        let mut mgr = HybridStateManager::new(&kinds, h, h_kv, d, 10000.0, 2048);
        let total_mem_at_0 = mgr.memory_bytes();

        assert_eq!(mgr.num_recurrent_layers(), 14);
        assert_eq!(mgr.num_anchor_layers(), 2);

        let expected_rec_bytes = 14 * h * d * d * std::mem::size_of::<f32>();
        assert_eq!(mgr.recurrent_state_bytes(), expected_rec_bytes);

        let q_init: Vec<f32> = (0..(h * d)).map(|i| ((i as f32) * 0.03).sin()).collect();
        let k_init: Vec<f32> = (0..(h_kv * d)).map(|i| ((i as f32) * 0.05).cos()).collect();
        let v_init: Vec<f32> = (0..(h_kv * d)).map(|i| ((i as f32) * 0.07).sin()).collect();
        let alpha = vec![0.92f32; h];
        let mut out = vec![0.0f32; h * d];

        // Step 1 token
        let mut q = q_init.clone();
        let mut k = k_init.clone();
        for l in 0..16 {
            if kinds[l] == LayerKind::GatedLinearRecurrence {
                mgr.step_recurrent(l, &q, &k, &v_init, &alpha, &mut out);
            } else {
                q.copy_from_slice(&q_init);
                k.copy_from_slice(&k_init);
                mgr.step_anchor_gqa(l, &mut q, &mut k, &v_init, &mut out);
            }
        }
        mgr.advance_token();

        let rec_bytes_at_1 = mgr.recurrent_state_bytes();
        assert_eq!(rec_bytes_at_1, expected_rec_bytes);
        assert_eq!(
            mgr.anchor_kv_live_bytes(),
            2 * (2 * 1 * h_kv * d * std::mem::size_of::<u16>())
        );

        // Run up to 2,048 tokens!
        for _step in 1..2048 {
            for l in 0..16 {
                if kinds[l] == LayerKind::GatedLinearRecurrence {
                    mgr.step_recurrent(l, &q_init, &k_init, &v_init, &alpha, &mut out);
                } else {
                    q.copy_from_slice(&q_init);
                    k.copy_from_slice(&k_init);
                    mgr.step_anchor_gqa(l, &mut q, &mut k, &v_init, &mut out);
                }
            }
            mgr.advance_token();
        }

        assert_eq!(mgr.seq_len(), 2048);
        // Recurrent state is strictly O(1): byte-identical size at token 2048 as at token 1!
        assert_eq!(mgr.recurrent_state_bytes(), rec_bytes_at_1);
        // Total allocated state memory (`reserve_exact`) never grew by a single byte across 2048 tokens!
        assert_eq!(mgr.memory_bytes(), total_mem_at_0);
        // Anchor KV cache tracks ONLY the 2 anchor layers (not 16 layers -> 8x smaller KV footprint!).
        assert_eq!(
            mgr.anchor_kv_live_bytes(),
            2 * (2 * 2048 * h_kv * d * std::mem::size_of::<u16>())
        );
        for l in 0..16 {
            match (kinds[l], mgr.layer_state(l)) {
                (LayerKind::GatedLinearRecurrence, LayerState::Recurrent(r)) => {
                    assert_eq!(r.state.len(), h * d * d);
                }
                (LayerKind::AnchorGqa, LayerState::AnchorGqa(kv)) => {
                    assert_eq!(kv.seq_len, 2048);
                }
                _ => panic!("layer {l} state kind mismatch"),
            }
        }
    }
}

//! `TerraEngine`: Full Micro-TERRA Forward Pass (`schemas/terra_format.md` Section 4),
//! Greedy Autoregressive Decode / Benchmark Harness, and `golden_ref.json` Parity Verifier.

use std::path::Path;
use std::time::Instant;

use half::f16;
use serde::{Deserialize, Serialize};

use crate::format::{CoreFile, CoreHeader, ExpertsDirectory, FormatError, LayerKind, TensorDtype};
use crate::kernels::ternary_simd::{
    active_simd_backend, ternary_gemv, ternary_gemv_prequantized, QuantizedActivation,
};
use crate::kernels::{
    f16_gemv, f32_gemv, rms_norm, rms_norm_in_place, sigmoid, silu, softmax_in_place,
    top_k_with_ties,
};
use crate::nvme_pool::{ExpertLruPool, PoolSizingConfig, PrefetchMetrics};
use crate::state::HybridStateManager;
use crate::RuntimeError;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Strict RAM budget in megabytes (`--ram-budget-mb`), subtracting resident `core.bin` + state.
    pub ram_budget_mb: Option<usize>,
    /// Optional explicit override for cached expert slots (primarily for deterministic cache tests).
    pub max_cached_experts: Option<usize>,
    /// Enables 3-layer lookahead router prefetching (`Layer L -> L + lookahead_offset`).
    pub enable_lookahead_prefetch: bool,
    /// Hard maximum sequence length reserved up-front (`reserve_exact`, zero `Vec` doubling) for anchor KV caches.
    /// `0` (the default) reserves the model's full `header.max_position_embeddings`.
    pub kv_reserve_tokens: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            ram_budget_mb: None,
            max_cached_experts: None,
            enable_lookahead_prefetch: true,
            kv_reserve_tokens: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LinearHandle {
    entry_idx: usize,
    dtype: TensorDtype,
    rows: usize,
    cols: usize,
}

impl LinearHandle {
    fn resolve(
        core: &CoreFile,
        name: &str,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<Self, FormatError> {
        let idx = core.require_entry_index(name)?;
        let e = &core.entries[idx];
        if e.ndim != 2
            || e.shape[0] as usize != expected_rows
            || e.shape[1] as usize != expected_cols
        {
            return Err(FormatError::InvalidMetadata(format!(
                "tensor `{name}` expected shape [{expected_rows}, {expected_cols}], got {:?}",
                e.dims()
            )));
        }
        match e.dtype {
            TensorDtype::Ternary2Bit | TensorDtype::F32 | TensorDtype::F16 => Ok(Self {
                entry_idx: idx,
                dtype: e.dtype,
                rows: expected_rows,
                cols: expected_cols,
            }),
            other => Err(FormatError::InvalidMetadata(format!(
                "unsupported linear projection dtype {other:?} for `{name}`"
            ))),
        }
    }

    fn apply(
        &self,
        core: &CoreFile,
        x: &[f32],
        rms_prescale: bool,
        rms_eps: f32,
        out: &mut [f32],
    ) {
        match self.dtype {
            TensorDtype::Ternary2Bit => {
                let tv = core.ternary_by_index(self.entry_idx);
                ternary_gemv(
                    tv.packed,
                    &tv.scales,
                    x,
                    self.rows,
                    self.cols,
                    rms_prescale,
                    rms_eps,
                    out,
                );
            }
            TensorDtype::F32 => {
                let w = core.f32_by_index(self.entry_idx);
                f32_gemv(&w, x, self.rows, self.cols, out);
            }
            TensorDtype::F16 => {
                let bits = core.f16_bits_by_index(self.entry_idx);
                f16_gemv(&bits, x, self.rows, self.cols, out);
            }
            TensorDtype::I8 => unreachable!("validated in resolve"),
        }
    }

    fn apply_with_optional_qx(
        &self,
        core: &CoreFile,
        x: &[f32],
        qx: Option<&QuantizedActivation>,
        rms_prescale: bool,
        rms_eps: f32,
        out: &mut [f32],
    ) {
        if self.dtype == TensorDtype::Ternary2Bit {
            if let Some(q) = qx {
                let tv = core.ternary_by_index(self.entry_idx);
                ternary_gemv_prequantized(tv.packed, &tv.scales, q, self.rows, self.cols, out);
                return;
            }
        }
        self.apply(core, x, rms_prescale, rms_eps, out);
    }
}

#[derive(Debug, Clone, Copy)]
struct HintHandles {
    table_idx: usize,
    num_buckets: usize,
    hint_dim: usize,
    proj: LinearHandle,
}

#[derive(Debug, Clone)]
struct LayerHandles {
    kind: LayerKind,
    input_ln_idx: usize,
    post_ln_idx: usize,
    q_proj: LinearHandle,
    k_proj: LinearHandle,
    v_proj: LinearHandle,
    o_proj: LinearHandle,
    q_norm_idx: usize,
    k_norm_idx: usize,
    decay_proj: Option<(LinearHandle, usize)>,
    moe_gate: LinearHandle,
    lookahead_router: Option<LinearHandle>,
}

#[derive(Debug, Clone)]
pub struct ForwardStepOutput {
    pub logits: Vec<f32>,
    pub layer0_post_moe: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBreakdown {
    pub core_resident_bytes: usize,
    pub recurrent_state_bytes: usize,
    pub anchor_kv_live_bytes: usize,
    pub expert_pool_bytes: usize,
    pub total_resident_bytes: usize,
    pub max_cached_experts: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub simd_backend: String,
    pub io_backend: String,
    pub direct_io_active: bool,
    pub lookahead_enabled: bool,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub decode_steps: usize,
    pub output_token_ids: Vec<u32>,
    pub ttft_ms: f64,
    pub decode_ms: f64,
    pub decode_tokens_per_sec: f64,
    pub memory: MemoryBreakdown,
    pub prefetch_metrics: PrefetchMetrics,
    pub cache_hit_rate_pct: f64,
}

pub struct TerraEngine {
    core: CoreFile,
    embed_idx: usize,
    embed_dtype: TensorDtype,
    final_norm_idx: usize,
    lm_head: LinearHandle,
    hint_handles: Option<HintHandles>,
    layers: Vec<LayerHandles>,
    state: HybridStateManager,
    nvme_pool: ExpertLruPool,
    config: EngineConfig,
}

impl TerraEngine {
    pub fn open_dir(dir: &Path, config: EngineConfig) -> Result<Self, RuntimeError> {
        Self::open(&dir.join("core.bin"), &dir.join("experts.bin"), config)
    }

    pub fn open(
        core_path: &Path,
        experts_path: &Path,
        config: EngineConfig,
    ) -> Result<Self, RuntimeError> {
        let core = CoreFile::open(core_path)?;
        let experts_dir = ExpertsDirectory::open(experts_path)?;
        experts_dir.validate_matches_core(&core.header)?;

        let hdr = &core.header;
        let vocab = hdr.vocab_size as usize;
        let d = hdr.hidden_size as usize;
        let h = hdr.num_attention_heads as usize;
        let h_kv = hdr.num_kv_heads as usize;
        let hd = hdr.head_dim as usize;
        let num_layers = hdr.num_layers as usize;
        let num_experts = hdr.num_experts as usize;

        // Validate embeddings
        let embed_idx = core.require_entry_index("model.embed_tokens.weight")?;
        let embed_entry = &core.entries[embed_idx];
        if embed_entry.ndim != 2
            || embed_entry.shape[0] as usize != vocab
            || embed_entry.shape[1] as usize != d
            || !matches!(embed_entry.dtype, TensorDtype::F32 | TensorDtype::F16)
        {
            return Err(FormatError::InvalidMetadata(format!(
                "invalid model.embed_tokens.weight metadata: dtype={:?}, dims={:?}",
                embed_entry.dtype,
                embed_entry.dims()
            ))
            .into());
        }
        let embed_dtype = embed_entry.dtype;

        let final_norm_idx = Self::require_f32_vec(&core, "model.norm.weight", d)?;

        let lm_head = if core.entry_index("lm_head.weight").is_some() {
            LinearHandle::resolve(&core, "lm_head.weight", vocab, d)?
        } else if hdr.tie_word_embeddings() {
            LinearHandle {
                entry_idx: embed_idx,
                dtype: embed_dtype,
                rows: vocab,
                cols: d,
            }
        } else {
            return Err(FormatError::MissingTensor("lm_head.weight".into()).into());
        };

        let hint_handles = if hdr.has_hint_embedding()
            && core.entry_index("model.hint_embedding.table.weight").is_some()
        {
            let table_idx = core.require_entry_index("model.hint_embedding.table.weight")?;
            let te = &core.entries[table_idx];
            if te.dtype != TensorDtype::F32 || te.ndim != 2 {
                return Err(FormatError::InvalidMetadata(
                    "model.hint_embedding.table.weight must be 2D F32".into(),
                )
                .into());
            }
            let num_buckets = te.shape[0] as usize;
            if num_buckets < 4 || num_buckets % 4 != 0 {
                return Err(FormatError::InvalidMetadata(format!(
                    "model.hint_embedding.table.weight num_buckets ({num_buckets}) must be >= 4 and divisible by 4"
                ))
                .into());
            }
            let hint_dim = te.shape[1] as usize;
            let proj = LinearHandle::resolve(
                &core,
                "model.hint_embedding.proj.weight",
                d,
                4 * hint_dim,
            )?;
            Some(HintHandles {
                table_idx,
                num_buckets,
                hint_dim,
                proj,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let kind = hdr.layer_kinds[l];
            let input_ln_idx =
                Self::require_f32_vec(&core, &format!("layers.{l}.input_layernorm.weight"), d)?;
            let post_ln_idx = Self::require_f32_vec(
                &core,
                &format!("layers.{l}.post_attention_layernorm.weight"),
                d,
            )?;

            let q_proj = LinearHandle::resolve(
                &core,
                &format!("layers.{l}.self_attn.q_proj.weight"),
                h * hd,
                d,
            )?;
            let k_proj = LinearHandle::resolve(
                &core,
                &format!("layers.{l}.self_attn.k_proj.weight"),
                h_kv * hd,
                d,
            )?;
            let v_proj = LinearHandle::resolve(
                &core,
                &format!("layers.{l}.self_attn.v_proj.weight"),
                h_kv * hd,
                d,
            )?;
            let o_proj = LinearHandle::resolve(
                &core,
                &format!("layers.{l}.self_attn.o_proj.weight"),
                d,
                h * hd,
            )?;

            let q_norm_idx =
                Self::require_f32_vec(&core, &format!("layers.{l}.self_attn.q_norm.weight"), h * hd)?;
            let k_norm_idx = Self::require_f32_vec(
                &core,
                &format!("layers.{l}.self_attn.k_norm.weight"),
                h_kv * hd,
            )?;

            let decay_proj = if kind == LayerKind::GatedLinearRecurrence {
                let w = LinearHandle::resolve(
                    &core,
                    &format!("layers.{l}.self_attn.decay_proj.weight"),
                    h,
                    d,
                )?;
                let b_idx = Self::require_f32_vec(
                    &core,
                    &format!("layers.{l}.self_attn.decay_proj.bias"),
                    h,
                )?;
                Some((w, b_idx))
            } else {
                None
            };

            let moe_gate = LinearHandle::resolve(
                &core,
                &format!("layers.{l}.mlp.gate.weight"),
                num_experts,
                d,
            )?;

            let look_name = format!("layers.{l}.lookahead_router.weight");
            let lookahead_router = if core.entry_index(&look_name).is_some() {
                Some(LinearHandle::resolve(&core, &look_name, num_experts, d)?)
            } else {
                None
            };

            layers.push(LayerHandles {
                kind,
                input_ln_idx,
                post_ln_idx,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_idx,
                k_norm_idx,
                decay_proj,
                moe_gate,
                lookahead_router,
            });
        }

        let max_model_pos = (hdr.max_position_embeddings as usize).max(1);
        let reserve_tokens = if config.kv_reserve_tokens == 0 {
            max_model_pos
        } else {
            config.kv_reserve_tokens.min(max_model_pos)
        };
        let state = HybridStateManager::new(
            &hdr.layer_kinds,
            h,
            h_kv,
            hd,
            hdr.rope_theta,
            reserve_tokens,
        );

        let sizing = PoolSizingConfig {
            ram_budget_bytes: config.ram_budget_mb.map(|mb| mb * 1024 * 1024),
            max_cached_experts: config.max_cached_experts,
            core_resident_bytes: core.memory_bytes(),
            state_reserved_bytes: state.memory_bytes(),
        };
        let nvme_pool = ExpertLruPool::open(experts_path, hdr.top_k as usize, sizing)?;

        Ok(Self {
            core,
            embed_idx,
            embed_dtype,
            final_norm_idx,
            lm_head,
            hint_handles,
            layers,
            state,
            nvme_pool,
            config,
        })
    }

    fn require_f32_vec(
        core: &CoreFile,
        name: &str,
        expected_len: usize,
    ) -> Result<usize, FormatError> {
        let idx = core.require_entry_index(name)?;
        let e = &core.entries[idx];
        if e.dtype != TensorDtype::F32 || e.ndim != 1 || e.shape[0] as usize != expected_len {
            return Err(FormatError::InvalidMetadata(format!(
                "tensor `{name}` expected 1D F32 [{expected_len}], got dtype={:?}, dims={:?}",
                e.dtype,
                e.dims()
            )));
        }
        Ok(idx)
    }

    #[inline]
    pub fn header(&self) -> &CoreHeader {
        &self.core.header
    }

    #[inline]
    pub fn state(&self) -> &HybridStateManager {
        &self.state
    }

    #[inline]
    pub fn nvme_pool(&self) -> &ExpertLruPool {
        &self.nvme_pool
    }

    pub fn reset_state(&mut self) {
        self.state.reset();
        self.nvme_pool.reset_metrics();
    }

    pub fn memory_breakdown(&self) -> MemoryBreakdown {
        let core_b = self.core.memory_bytes();
        let rec_b = self.state.recurrent_state_bytes();
        let kv_b = self.state.anchor_kv_live_bytes();
        let pool_b = self.nvme_pool.pool_buffer_bytes();
        MemoryBreakdown {
            core_resident_bytes: core_b,
            recurrent_state_bytes: rec_b,
            anchor_kv_live_bytes: kv_b,
            expert_pool_bytes: pool_b,
            total_resident_bytes: core_b + self.state.memory_bytes() + pool_b,
            max_cached_experts: self.nvme_pool.max_cached_experts(),
        }
    }

    /// Executes a single-token forward step across all layers, returning `[vocab_size]` logits.
    pub fn forward_token(&mut self, token_id: u32) -> Result<Vec<f32>, RuntimeError> {
        let out = self.forward_token_with_hints(token_id, None, false)?;
        Ok(out.logits)
    }

    /// Executes Section 4 of `schemas/terra_format.md` for one token (+ optional `[4]` edge hint tuple).
    pub fn forward_token_with_hints(
        &mut self,
        token_id: u32,
        hint: Option<[i32; 4]>,
        capture_layer0: bool,
    ) -> Result<ForwardStepOutput, RuntimeError> {
        let vocab = self.core.header.vocab_size as usize;
        let tok = token_id as usize;
        if tok >= vocab {
            return Err(RuntimeError::TokenOutOfRange {
                token_id,
                vocab_size: self.core.header.vocab_size,
            });
        }
        if self.state.seq_len() >= self.state.max_kv_tokens() {
            return Err(RuntimeError::ContextLengthExceeded {
                seq_len: self.state.seq_len() + 1,
                max_seq_len: self.state.max_kv_tokens(),
            });
        }

        let d = self.core.header.hidden_size as usize;
        let inter = self.core.header.intermediate_size as usize;
        let h_heads = self.core.header.num_attention_heads as usize;
        let h_kv_heads = self.core.header.num_kv_heads as usize;
        let hd = self.core.header.head_dim as usize;
        let num_layers = self.core.header.num_layers as usize;
        let num_experts = self.core.header.num_experts as usize;
        let top_k = self.core.header.top_k as usize;
        let lookahead_offset = self.core.header.lookahead_offset as usize;
        let eps = self.core.header.rms_norm_eps;
        let rms_prescale = self.core.header.has_rms_prescale();
        let norm_topk = self.core.header.norm_topk_prob();

        // 1. Token Embedding (+ optional Section 2.3 HintEmbedding)
        let mut h = vec![0.0f32; d];
        match self.embed_dtype {
            TensorDtype::F32 => {
                let table = self.core.f32_by_index(self.embed_idx);
                h.copy_from_slice(&table[tok * d..(tok + 1) * d]);
            }
            TensorDtype::F16 => {
                let table = self.core.f16_bits_by_index(self.embed_idx);
                let row = &table[tok * d..(tok + 1) * d];
                for i in 0..d {
                    h[i] = f16::from_bits(row[i]).to_f32();
                }
            }
            _ => unreachable!(),
        }

        if let (Some(hh), Some(hint_ids)) = (self.hint_handles, hint) {
            if hint_ids[0] != 0 {
                let table = self.core.f32_by_index(hh.table_idx);
                let mut hint_flat = vec![0.0f32; 4 * hh.hint_dim];
                let slot_buckets = hh.num_buckets / 4;
                for (slot, &raw_id) in hint_ids.iter().enumerate() {
                    let bucket = slot * slot_buckets
                        + ((raw_id as isize).rem_euclid(slot_buckets as isize) as usize);
                    let src = &table[bucket * hh.hint_dim..(bucket + 1) * hh.hint_dim];
                    hint_flat[slot * hh.hint_dim..(slot + 1) * hh.hint_dim].copy_from_slice(src);
                }
                let mut hint_delta = vec![0.0f32; d];
                hh.proj
                    .apply(&self.core, &hint_flat, rms_prescale, eps, &mut hint_delta);
                for i in 0..d {
                    h[i] += hint_delta[i];
                }
            }
        }

        // Reusable per-layer working buffers
        let mut u_t = vec![0.0f32; d];
        let mut q_flat = vec![0.0f32; h_heads * hd];
        let mut k_flat = vec![0.0f32; h_kv_heads * hd];
        let mut v_flat = vec![0.0f32; h_kv_heads * hd];
        let mut attn_out = vec![0.0f32; h_heads * hd];
        let mut o_proj_out = vec![0.0f32; d];
        let mut alpha = vec![0.0f32; h_heads];

        let mut look_logits = vec![0.0f32; num_experts];
        let mut m_t = vec![0.0f32; d];
        let mut router_probs = vec![0.0f32; num_experts];
        let mut gate_buf = vec![0.0f32; inter];
        let mut up_buf = vec![0.0f32; inter];
        let mut swiglu_buf = vec![0.0f32; inter];
        let mut down_buf = vec![0.0f32; d];
        let mut moe_accum = vec![0.0f32; d];

        let mut layer0_post_moe = None;

        let step_res: Result<(), RuntimeError> = (|| {
            for l in 0..num_layers {
                let lw = &self.layers[l];

                // Step A: Pre-attention RMSNorm
                {
                    let in_ln = self.core.f32_by_index(lw.input_ln_idx);
                    rms_norm(&h, &in_ln, eps, &mut u_t);
                }

                // Pre-quantize `u_t` once if any of Q/K/V is Ternary2Bit
                let u_qx = if lw.q_proj.dtype == TensorDtype::Ternary2Bit
                    || lw.k_proj.dtype == TensorDtype::Ternary2Bit
                    || lw.v_proj.dtype == TensorDtype::Ternary2Bit
                {
                    Some(QuantizedActivation::quantize(&u_t, rms_prescale, eps))
                } else {
                    None
                };

                lw.q_proj.apply_with_optional_qx(
                    &self.core,
                    &u_t,
                    u_qx.as_ref(),
                    rms_prescale,
                    eps,
                    &mut q_flat,
                );
                lw.k_proj.apply_with_optional_qx(
                    &self.core,
                    &u_t,
                    u_qx.as_ref(),
                    rms_prescale,
                    eps,
                    &mut k_flat,
                );
                lw.v_proj.apply_with_optional_qx(
                    &self.core,
                    &u_t,
                    u_qx.as_ref(),
                    rms_prescale,
                    eps,
                    &mut v_flat,
                );

                // OLMoE full-vector QK-Norm BEFORE head reshape (Section 4.1)
                {
                    let q_norm_w = self.core.f32_by_index(lw.q_norm_idx);
                    rms_norm_in_place(&mut q_flat, &q_norm_w, eps);
                    let k_norm_w = self.core.f32_by_index(lw.k_norm_idx);
                    rms_norm_in_place(&mut k_flat, &k_norm_w, eps);
                }

                // Step B: Hybrid Sequence Mixer (AnchorGQA vs O(1) GatedLinearRecurrence)
                match lw.kind {
                    LayerKind::AnchorGqa => {
                        self.state
                            .step_anchor_gqa(l, &mut q_flat, &mut k_flat, &v_flat, &mut attn_out);
                    }
                    LayerKind::GatedLinearRecurrence => {
                        let (decay_w, decay_b_idx) =
                            lw.decay_proj.expect("validated recurrent decay_proj");
                        decay_w.apply(&self.core, &u_t, rms_prescale, eps, &mut alpha);
                        let decay_b = self.core.f32_by_index(decay_b_idx);
                        for head_i in 0..h_heads {
                            alpha[head_i] = sigmoid(alpha[head_i] + decay_b[head_i]);
                        }
                        self.state
                            .step_recurrent(l, &q_flat, &k_flat, &v_flat, &alpha, &mut attn_out);
                    }
                }

                lw.o_proj
                    .apply(&self.core, &attn_out, rms_prescale, eps, &mut o_proj_out);
                for i in 0..d {
                    h[i] += o_proj_out[i];
                }
                // `h` now holds residual `h'_t`

                // Step C: 3-Layer Lookahead Router Prefetch (`Layer l -> l + lookahead_offset`)
                // Speculative prefetch errors are intentionally ignored here; any needed expert
                // is re-read on demand at `l + lookahead_offset`.
                if self.config.enable_lookahead_prefetch && l + lookahead_offset < num_layers {
                    if let Some(look_w) = lw.lookahead_router {
                        look_w.apply(&self.core, &h, rms_prescale, eps, &mut look_logits);
                        let pred_topk = top_k_with_ties(&look_logits, top_k);
                        let pred_ids: Vec<usize> = pred_topk.iter().map(|&(eid, _)| eid).collect();
                        let _ = self
                            .nvme_pool
                            .prefetch_layer_experts(l + lookahead_offset, &pred_ids);
                    }
                }

                // Step D: Post-Attention RMSNorm + Ground-Truth MoE Routing
                {
                    let post_ln = self.core.f32_by_index(lw.post_ln_idx);
                    rms_norm(&h, &post_ln, eps, &mut m_t);
                }

                lw.moe_gate
                    .apply(&self.core, &m_t, rms_prescale, eps, &mut router_probs);
                softmax_in_place(&mut router_probs);

                let mut topk = top_k_with_ties(&router_probs, top_k);
                if norm_topk {
                    let sum_p: f32 = topk.iter().map(|&(_, p)| p).sum();
                    if sum_p > 0.0 {
                        let inv = 1.0f32 / sum_p;
                        for (_, p) in &mut topk {
                            *p *= inv;
                        }
                    }
                }

                let active_ids: Vec<usize> = topk.iter().map(|&(eid, _)| eid).collect();
                self.nvme_pool.acquire_layer_experts(l, &active_ids)?;

                // Pre-quantize `m_t` ONCE and reuse across all `top_k` experts' `gate_proj` and `up_proj`!
                let m_qx = QuantizedActivation::quantize(&m_t, rms_prescale, eps);
                moe_accum.fill(0.0);

                let expert_res: Result<(), RuntimeError> = (|| {
                    for &(expert_id, weight_p) in &topk {
                        let view = self.nvme_pool.expert_view(l, expert_id)?;
                        ternary_gemv_prequantized(
                            view.gate_packed,
                            view.gate_scales,
                            &m_qx,
                            inter,
                            d,
                            &mut gate_buf,
                        );
                        ternary_gemv_prequantized(
                            view.up_packed,
                            view.up_scales,
                            &m_qx,
                            inter,
                            d,
                            &mut up_buf,
                        );
                        for j in 0..inter {
                            swiglu_buf[j] = silu(gate_buf[j]) * up_buf[j];
                        }
                        ternary_gemv(
                            view.down_packed,
                            view.down_scales,
                            &swiglu_buf,
                            d,
                            inter,
                            rms_prescale,
                            eps,
                            &mut down_buf,
                        );
                        for i in 0..d {
                            moe_accum[i] += weight_p * down_buf[i];
                        }
                    }
                    Ok(())
                })();

                self.nvme_pool.release_layer_experts(l, &active_ids);
                expert_res?;

                for i in 0..d {
                    h[i] += moe_accum[i];
                }

                if l == 0 && capture_layer0 {
                    layer0_post_moe = Some(h.clone());
                }
            }
            Ok(())
        })();

        if let Err(err) = step_res {
            self.state.rollback_token();
            return Err(err);
        }
        self.state.commit_token();

        // Final RMSNorm + LM Head
        let mut h_norm = vec![0.0f32; d];
        {
            let final_ln = self.core.f32_by_index(self.final_norm_idx);
            rms_norm(&h, &final_ln, eps, &mut h_norm);
        }

        let mut logits = vec![0.0f32; vocab];
        self.lm_head
            .apply(&self.core, &h_norm, rms_prescale, eps, &mut logits);

        Ok(ForwardStepOutput {
            logits,
            layer0_post_moe,
        })
    }

    /// Runs prefill over `prompt_tokens` and greedy decode for `max_new_tokens`, collecting benchmark telemetry.
    pub fn generate_greedy(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
    ) -> Result<BenchmarkReport, RuntimeError> {
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::EmptyPrompt);
        }

        let prefill_start = Instant::now();
        let mut last_logits = Vec::new();
        for &tok in prompt_tokens {
            last_logits = self.forward_token(tok)?;
        }
        let ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let mut output_token_ids = Vec::with_capacity(max_new_tokens);
        if max_new_tokens > 0 {
            // First generated token is sampled directly from the final prefill logits (accounted in `ttft_ms`).
            let first_tok = argmax_token(&last_logits);
            output_token_ids.push(first_tok);
        }

        let decode_steps = max_new_tokens.saturating_sub(1);
        let decode_start = Instant::now();
        if decode_steps > 0 {
            let mut next_tok = output_token_ids[0];
            for _ in 0..decode_steps {
                let logits = self.forward_token(next_tok)?;
                next_tok = argmax_token(&logits);
                output_token_ids.push(next_tok);
            }
        }

        let decode_elapsed = decode_start.elapsed();
        let decode_ms = decode_elapsed.as_secs_f64() * 1000.0;
        let decode_tokens_per_sec = if decode_steps > 0 && decode_elapsed.as_secs_f64() > 0.0 {
            (decode_steps as f64) / decode_elapsed.as_secs_f64()
        } else {
            0.0
        };

        let metrics = self.nvme_pool.metrics().clone();
        let hit_rate_pct = metrics.effective_hit_rate() * 100.0;

        Ok(BenchmarkReport {
            simd_backend: active_simd_backend().to_string(),
            io_backend: self.nvme_pool.backend_name().to_string(),
            direct_io_active: self.nvme_pool.direct_io_active(),
            lookahead_enabled: self.config.enable_lookahead_prefetch,
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: output_token_ids.len(),
            decode_steps,
            output_token_ids,
            ttft_ms,
            decode_ms,
            decode_tokens_per_sec,
            memory: self.memory_breakdown(),
            prefetch_metrics: metrics,
            cache_hit_rate_pct: hit_rate_pct,
        })
    }
}

#[inline]
pub fn argmax_token(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx as u32
}

// ============================================================================
// Golden Reference (`golden_ref.json`) Numerical Parity Verifier
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GoldenLayer0 {
    LastStep(Vec<f32>),
    PerStep(Vec<Vec<f32>>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenReference {
    #[serde(alias = "token_ids")]
    pub prompt_token_ids: Vec<u32>,
    #[serde(default, alias = "hints")]
    pub hint_matrix: Option<Vec<[i32; 4]>>,
    #[serde(default)]
    pub step_logits: Option<Vec<Vec<f32>>>,
    #[serde(default, alias = "last_logits")]
    pub prefill_last_logits: Option<Vec<f32>>,
    #[serde(default)]
    pub layer0_post_moe: Option<GoldenLayer0>,
    #[serde(default)]
    pub layer0_post_moe_steps: Option<Vec<Vec<f32>>>,
}

/// Loads `core.bin`, `experts.bin`, and `golden_ref.json` from `dir` and verifies that
/// Rust forward-pass outputs match PyTorch reference outputs within `tol` (e.g., `1e-2`).
/// Returns the observed maximum absolute difference across all checked tensors.
pub fn verify_golden_dir(dir: &Path, tol: f32) -> Result<f32, RuntimeError> {
    verify_golden_dir_with_config(dir, tol, EngineConfig::default())
}

pub fn verify_golden_dir_with_config(
    dir: &Path,
    tol: f32,
    config: EngineConfig,
) -> Result<f32, RuntimeError> {
    let golden_path = dir.join("golden_ref.json");
    let raw_json = std::fs::read_to_string(&golden_path).map_err(FormatError::Io)?;
    let golden: GoldenReference = serde_json::from_str(&raw_json)?;

    if golden.prompt_token_ids.is_empty() {
        return Err(RuntimeError::EmptyPrompt);
    }
    if golden.step_logits.is_none() && golden.prefill_last_logits.is_none() {
        return Err(RuntimeError::MissingGoldenTargets);
    }

    let mut engine = TerraEngine::open_dir(dir, config)?;
    let t_count = golden.prompt_token_ids.len();

    let mut actual_step_logits = Vec::with_capacity(t_count);
    let mut actual_layer0_steps = Vec::with_capacity(t_count);

    for (pos, &tok) in golden.prompt_token_ids.iter().enumerate() {
        let hint = golden
            .hint_matrix
            .as_ref()
            .and_then(|hm| hm.get(pos).copied());
        let step_out = engine.forward_token_with_hints(tok, hint, true)?;
        actual_step_logits.push(step_out.logits);
        actual_layer0_steps.push(step_out.layer0_post_moe.expect("layer0 captured"));
    }

    let mut max_diff = 0.0f32;
    let mut max_loc = String::from("none");
    let mut checked_elements = 0usize;

    let mut check_vec = |label: &str, expected: &[f32], actual: &[f32]| -> Result<(), RuntimeError> {
        if expected.len() != actual.len() {
            return Err(RuntimeError::GoldenShapeMismatch {
                field: label.to_string(),
                expected: expected.len(),
                actual: actual.len(),
            });
        }
        checked_elements += expected.len();
        for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (e - a).abs();
            if diff.is_nan() || diff > max_diff {
                max_diff = diff;
                max_loc = format!("{label}[{i}] (expected={e}, actual={a})");
            }
        }
        Ok(())
    };

    if let Some(ref_steps) = &golden.step_logits {
        if ref_steps.len() != t_count {
            return Err(RuntimeError::GoldenShapeMismatch {
                field: "step_logits.len()".into(),
                expected: t_count,
                actual: ref_steps.len(),
            });
        }
        for t in 0..t_count {
            check_vec(&format!("step_logits[{t}]"), &ref_steps[t], &actual_step_logits[t])?;
        }
    }

    if let Some(ref_last) = &golden.prefill_last_logits {
        check_vec(
            "prefill_last_logits",
            ref_last,
            &actual_step_logits[t_count - 1],
        )?;
    }

    if let Some(ref_l0_steps) = &golden.layer0_post_moe_steps {
        if ref_l0_steps.len() != t_count {
            return Err(RuntimeError::GoldenShapeMismatch {
                field: "layer0_post_moe_steps.len()".into(),
                expected: t_count,
                actual: ref_l0_steps.len(),
            });
        }
        for t in 0..t_count {
            check_vec(
                &format!("layer0_post_moe_steps[{t}]"),
                &ref_l0_steps[t],
                &actual_layer0_steps[t],
            )?;
        }
    }

    if let Some(ref_l0) = &golden.layer0_post_moe {
        match ref_l0 {
            GoldenLayer0::LastStep(vec) => {
                check_vec(
                    "layer0_post_moe",
                    vec,
                    &actual_layer0_steps[t_count - 1],
                )?;
            }
            GoldenLayer0::PerStep(steps) => {
                if steps.len() != t_count {
                    return Err(RuntimeError::GoldenShapeMismatch {
                        field: "layer0_post_moe.len()".into(),
                        expected: t_count,
                        actual: steps.len(),
                    });
                }
                for t in 0..t_count {
                    check_vec(
                        &format!("layer0_post_moe[{t}]"),
                        &steps[t],
                        &actual_layer0_steps[t],
                    )?;
                }
            }
        }
    }

    if checked_elements == 0 {
        return Err(RuntimeError::MissingGoldenTargets);
    }
    if max_diff.is_nan() || max_diff > tol {
        return Err(RuntimeError::GoldenParityExceeded {
            max_abs_diff: max_diff,
            tolerance: tol,
            detail: max_loc,
        });
    }

    Ok(max_diff)
}

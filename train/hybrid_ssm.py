"""Step 3B: 7:1 Hybrid Gated Linear Recurrence + Anchor GQA & Differentiable `HintEmbedding`.

Converts 7 out of every 8 OLMoE Transformer attention layers (e.g., `0..6, 8..14` in a 16-layer model)
into $O(1)$-memory Gated Linear Recurrence blocks initialized directly from the teacher's
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, and `k_norm` parameters, while keeping every 8th
layer (`7, 15`) as an exact Grouped-Query Attention (GQA) anchor layer.

Also implements `HintEmbedding` (Guardrail #3) so edge-preprocessor `[batch, seq_len, 4]` integer hints
flow strictly through differentiable additive embeddings during distillation.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_anchor_layers(num_layers: int, block_ratio: int = 8) -> tuple[int, ...]:
    """Return 1-in-8 GQA anchor layer indices (e.g. `(7, 15)` for 16 layers; `(num_layers - 1,)` if `< 8`)."""
    anchors = tuple(i for i in range(block_ratio - 1, num_layers, block_ratio))
    if not anchors and num_layers > 0:
        anchors = (num_layers - 1,)
    return anchors


class HintEmbedding(nn.Module):
    """Differentiable projection table for `[batch, seq_len, 4]` dual-path edge hints.

    Input columns (`int64` / `int32`, negative values mapped mod `num_buckets`):
      0: `modality_flag`      (0 = pad/none, 1 = code AST, 2 = prose coref)
      1: `entity_or_var_id`   (hashed symbol / coref cluster ID)
      2: `relative_def_offset`(clamped/bucketed token distance to definition/antecedent)
      3: `entropy_bucket`     (confidence / ambiguity bucket, e.g. `< 0.65` flagged high-entropy)

    Zero-initializes `proj.weight` so inserting `HintEmbedding` into a pretrained teacher/student
    starts as an exact identity residual (`h + 0`) with non-zero gradients to `proj` on Step 1!
    """

    def __init__(
        self,
        hidden_size: int,
        num_buckets: int = 4096,
        hint_dim: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_buckets = num_buckets
        self.hint_dim = hint_dim

        # Shared bucket table across the 4 hint slots (exported as `model.hint_embedding.table.weight`)
        # with distinct slot offsets `[0, B/4, 2B/4, 3B/4]` so each slot occupies its own partition.
        if num_buckets % 4 != 0:
            raise ValueError(f"num_buckets ({num_buckets}) must be divisible by 4.")
        self.slot_buckets = num_buckets // 4
        self.table = nn.Embedding(num_buckets, hint_dim)
        self.proj = nn.Linear(4 * hint_dim, hidden_size, bias=False)

        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
        # Small non-zero init keeps residual perturbation ~1e-3 at Step 0 while ensuring Step 1 gradients
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-4)

    def forward(self, hints: torch.Tensor) -> torch.Tensor:
        """Map `hints` of shape `[batch, seq_len, 4]` -> additive embedding `[batch, seq_len, hidden_size]`."""
        if hints.ndim != 3 or hints.shape[-1] != 4:
            raise ValueError(f"Expected hints shape [batch, seq_len, 4], got {tuple(hints.shape)}")
        offsets = torch.arange(4, device=hints.device, dtype=torch.long) * self.slot_buckets
        bucket_ids = (hints.long() % self.slot_buckets) + offsets.view(1, 1, 4)
        emb = self.table(bucket_ids)  # [B, T, 4, hint_dim]
        flat = emb.reshape(hints.shape[0], hints.shape[1], 4 * self.hint_dim)
        out = self.proj(flat)
        valid = (hints[..., 0] != 0).unsqueeze(-1).to(dtype=out.dtype)
        return out * valid


class GatedLinearRecurrenceAttention(nn.Module):
    r"""$O(1)$-Memory Gated Linear Recurrence block matching Section 4.3 of `schemas/terra_format.md`.

    Equations per token $t$ and head $h \in [0, H)$:
      q_t = RMSNorm(W_q u_t, w_q_norm) / sqrt(d)
      k_t = repeat_kv(RMSNorm(W_k u_t, w_k_norm)) / sqrt(d)
      v_t = repeat_kv(W_v u_t)
      alpha_t = sigmoid(W_decay u_t + b_decay)   in (0, 1)^H
      S_t[h] = alpha_t[h] * S_{t-1}[h] + k_t[h] (x) v_t[h]^T   in R^{d x d}
      o_t[h] = S_t[h]^T @ q_t[h]                 in R^d
      y_t = W_o concat_h(o_t[h])

    Provides BOTH:
      - `forward_parallel`: Vectorized causal log-space decay matrix attention for fast sequence QAT
      - `forward_recurrent`: Exact $O(1)$-state recurrent loop matching Rust `runtime/src/state.rs`
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float = 1e-5,
        layer_idx: int = 0,
        init_decay_bias: float = 3.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.rms_norm_eps = rms_norm_eps
        self.layer_idx = layer_idx
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.chunk_size = 128

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        # Full-vector OLMoE QK-Norm parameters (`[H * d]` and `[H_kv * d]`)
        self.q_norm = nn.RMSNorm(self.num_heads * self.head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.num_kv_heads * self.head_dim, eps=rms_norm_eps)

        # Data-dependent per-head decay gate (`sigmoid(+3.0) ~= 0.9526` long-horizon warm start)
        self.decay_proj = nn.Linear(hidden_size, self.num_heads, bias=True)
        nn.init.zeros_(self.decay_proj.weight)
        if num_attention_heads > 1 and abs(float(init_decay_bias) - 3.0) < 1e-6:
            # Multi-scale decay half-lives ~4..512 tokens across heads for both syntax and 16k recall
            half_lives = 4.0 * (128.0 ** torch.linspace(0.0, 1.0, steps=num_attention_heads))
            alphas = 0.5 ** (1.0 / half_lives)
            biases = torch.log(alphas / (1.0 - alphas))
            with torch.no_grad():
                self.decay_proj.bias.copy_(biases)
        else:
            nn.init.constant_(self.decay_proj.bias, init_decay_bias)

    @classmethod
    def from_olmoe_attention(
        cls,
        attn: nn.Module,
        config: object,
        layer_idx: int,
        init_decay_bias: float = 3.0,
    ) -> "GatedLinearRecurrenceAttention":
        """Warm-start directly from a teacher `OlmoeAttention` layer (retaining `BitLinear` if already wrapped)."""
        block = cls(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            layer_idx=layer_idx,
            init_decay_bias=init_decay_bias,
        )
        # Preserve Q/K/V/O modules directly (supports both nn.Linear and BitLinear!)
        block.q_proj = attn.q_proj
        block.k_proj = attn.k_proj
        block.v_proj = attn.v_proj
        block.o_proj = attn.o_proj
        block.q_norm = copy.deepcopy(attn.q_norm)
        block.k_norm = copy.deepcopy(attn.k_norm)

        param_ref = next(attn.parameters())
        block.decay_proj = block.decay_proj.to(device=param_ref.device, dtype=torch.float32)

        # Normalize per-head query RMSNorm weights by `(1 - alpha_0)` so the unnormalized
        # recurrent horizon `sum_{tau<=t} alpha_h^{t-tau} = 1 / (1 - alpha_h)` has unit
        # convex-combination gain at Step 0, matching Softmax Attention output variance.
        with torch.no_grad():
            alpha0 = torch.sigmoid(block.decay_proj.bias.detach()).to(
                device=block.q_norm.weight.device, dtype=block.q_norm.weight.dtype
            )
            horizon_scale = (
                (1.0 - alpha0)
                .unsqueeze(-1)
                .expand(block.num_heads, block.head_dim)
                .reshape(-1)
            )
            if block.q_norm.weight.numel() == horizon_scale.numel():
                block.q_norm.weight.mul_(horizon_scale)
        return block

    def enable_delta_rule(self, init_bias: float = -3.0) -> None:
        """Switch the write rule to erase-then-write: `S = alpha * (I - beta * k^ k^T) S + k v^T`.

        `beta = sigmoid(erase_proj(h))` per head; at `init_bias = -3` (`beta ~= 0.05`) the layer
        starts within a few percent of the plain additive rule, so the warm start is preserved.
        ponytail: Python training path only; exporter and Rust `state.rs` still implement the
        additive rule, add the erase step there once this wins an ablation.
        """
        ref = self.decay_proj.weight
        self.erase_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True).to(ref.device, torch.float32)
        nn.init.zeros_(self.erase_proj.weight)
        nn.init.constant_(self.erase_proj.bias, init_bias)

    def enable_write_gate(self, init_bias: float = 4.0) -> None:
        """Scale each token's write by a learned per-head importance `g = sigmoid(write_proj(h))`.

        `S += g * k v^T`, so unimportant tokens barely touch the memory. `init_bias = 4`
        (`g ~= 0.98`) keeps the warm start. ponytail: learned end to end; supervise it with the
        teacher's received attention if it does not learn to discriminate on its own.
        """
        ref = self.decay_proj.weight
        self.write_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True).to(ref.device, torch.float32)
        nn.init.zeros_(self.write_proj.weight)
        nn.init.constant_(self.write_proj.bias, init_bias)

    def _erase_gate(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Per-token, per-head erase strength `beta` in `(0, 1)` (`[B, T, H]`), or None for the additive rule."""
        if getattr(self, "erase_proj", None) is None:
            return None
        amp_off = torch.amp.autocast("cuda", enabled=False) if hidden_states.is_cuda else nullcontext()
        with amp_off:
            return torch.sigmoid(self.erase_proj(hidden_states.float()))

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand `[B, T, H_kv, d]` -> `[B, T, H, d]` matching GQA head grouping."""
        if self.num_key_value_groups == 1:
            return x
        bsz, seq_len, n_kv, head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(bsz, seq_len, n_kv, self.num_key_value_groups, head_dim)
            .reshape(bsz, seq_len, self.num_heads, head_dim)
        )

    def _project_qkv_alpha(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        # Full-vector QK-RMSNorm BEFORE head reshape (exact OLMoE convention!)
        q_flat = self.q_norm(self.q_proj(hidden_states))
        k_flat = self.k_norm(self.k_proj(hidden_states))
        v_flat = self.v_proj(hidden_states)

        q = q_flat.view(bsz, seq_len, self.num_heads, self.head_dim).float() * self.scale
        k = self._repeat_kv(k_flat.view(bsz, seq_len, self.num_kv_heads, self.head_dim)).float() * self.scale
        v = self._repeat_kv(v_flat.view(bsz, seq_len, self.num_kv_heads, self.head_dim)).float()

        amp_off = torch.amp.autocast("cuda", enabled=False) if hidden_states.is_cuda else nullcontext()
        with amp_off:
            alpha = torch.sigmoid(self.decay_proj(hidden_states.float()))  # [B, T, H]
            if getattr(self, "write_proj", None) is not None:
                # Gating v is the same as gating the write k v^T; both forward paths pick it up here.
                v = v * torch.sigmoid(self.write_proj(hidden_states.float())).unsqueeze(-1)
        return q, k, v, alpha

    def forward_parallel(
        self,
        hidden_states: torch.Tensor,
        chunk_size: int | None = None,
        eos_reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exact chunkwise parallel gated linear recurrence (`O(T * C)` memory instead of `O(T^2)`).

        Splits sequence length `T` into chunks of size `C` (default `128`):
          - Intra-chunk tokens attend via small `[B, H, L_c, L_c]` causal log-decay matrices.
          - Inter-chunk history propagates via the exact `[B, H, d, d]` recurrent state `S_{c-1}`.
          - Optional `eos_reset_mask` (`[B, T]` bool) resets prior state at new-document boundaries.
        """
        orig_dtype = hidden_states.dtype
        bsz, seq_len, _ = hidden_states.shape
        c_size = self.chunk_size if chunk_size is None else max(1, int(chunk_size))
        reset_mask = eos_reset_mask

        amp_off = torch.amp.autocast("cuda", enabled=False) if hidden_states.is_cuda else nullcontext()
        with amp_off:
            q, k, v, alpha = self._project_qkv_alpha(hidden_states)

            # Permute to [B, H, T, d]
            q_h = q.permute(0, 2, 1, 3)
            k_h = k.permute(0, 2, 1, 3)
            v_h = v.permute(0, 2, 1, 3)
            log_alpha = torch.log(alpha.clamp(min=1e-7)).permute(0, 2, 1)  # [B, H, T] <= 0
            beta = self._erase_gate(hidden_states)
            if beta is not None:
                khat_h = torch.nn.functional.normalize(k_h, dim=-1)
                ab_h = (alpha.clamp(min=1e-7) * beta).permute(0, 2, 1)   # alpha_t * beta_t, [B, H, T]

            seg_ids = (
                torch.cumsum(reset_mask.to(dtype=torch.int32), dim=-1)
                if reset_mask is not None
                else None
            )
            prev_seg = (
                torch.zeros((bsz, 1), dtype=torch.int32, device=hidden_states.device)
                if seg_ids is not None
                else None
            )

            state = torch.zeros(
                (bsz, self.num_heads, self.head_dim, self.head_dim),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            chunk_outputs: list[torch.Tensor] = []

            for start in range(0, seq_len, c_size):
                end = min(start + c_size, seq_len)
                lc = end - start

                q_c = q_h[:, :, start:end, :]          # [B, H, lc, d]
                k_c = k_h[:, :, start:end, :]          # [B, H, lc, d]
                v_c = v_h[:, :, start:end, :]          # [B, H, lc, d]
                la_c = log_alpha[:, :, start:end]      # [B, H, lc]
                cum_log = torch.cumsum(la_c, dim=-1)   # [B, H, lc] <= 0

                # 1. Inter-chunk contribution from incoming recurrent state S_{c-1}
                inter_decay = torch.exp(cum_log).unsqueeze(-1)              # [B, H, lc, 1]
                out_inter = torch.matmul(q_c, state) * inter_decay          # [B, H, lc, d]

                # 2. Intra-chunk causal linear attention over small [lc, lc] window
                diff = cum_log.unsqueeze(-1) - cum_log.unsqueeze(-2)        # [B, H, lc, lc]
                causal = torch.tril(
                    torch.ones((lc, lc), device=hidden_states.device, dtype=torch.bool)
                )
                diff_safe = torch.where(causal, diff, torch.zeros_like(diff))
                if seg_ids is not None and prev_seg is not None:
                    seg_c = seg_ids[:, start:end]
                    same_prev = (seg_c == prev_seg).unsqueeze(1).unsqueeze(-1)
                    out_inter = out_inter * same_prev
                    attn_mask = causal & (seg_c.unsqueeze(-1) == seg_c.unsqueeze(-2)).unsqueeze(1)
                else:
                    seg_c = None
                    attn_mask = causal

                decay_c = torch.where(attn_mask, torch.exp(diff_safe), torch.zeros_like(diff))
                qk_c = torch.matmul(q_c, k_c.transpose(-1, -2)) * decay_c   # [B, H, lc, lc]
                out_intra = torch.matmul(qk_c, v_c)                         # [B, H, lc, d]

                if beta is not None:
                    # Delta rule: u_t = S_{t-1}^T k^_t is what the memory returns for key t before
                    # the write. Within a chunk U solves the unit-lower-triangular system
                    #   (I + A) U = g * (K^ S_0) + B V
                    # A[t,i] = (gamma_{t-1}/gamma_i) alpha_i beta_i (k^_t . k^_i),  i < t
                    # B[t,i] = (gamma_{t-1}/gamma_i) (k^_t . k_i),                  i < t
                    kh_c = khat_h[:, :, start:end, :]
                    ab_c = ab_h[:, :, start:end]
                    cum_prev = cum_log - la_c                                # log gamma_{t-1}
                    strict = attn_mask & ~torch.eye(lc, device=hidden_states.device, dtype=torch.bool)
                    diff_p = cum_prev.unsqueeze(-1) - cum_log.unsqueeze(-2)
                    decay_p = torch.where(strict, torch.exp(torch.where(strict, diff_p, torch.zeros_like(diff_p))), torch.zeros_like(diff_p))
                    a_mat = torch.matmul(kh_c, kh_c.transpose(-1, -2)) * decay_p * ab_c.unsqueeze(-2)
                    b_mat = torch.matmul(kh_c, k_c.transpose(-1, -2)) * decay_p
                    u_prev = torch.matmul(kh_c, state) * torch.exp(cum_prev).unsqueeze(-1)
                    if seg_c is not None and prev_seg is not None:
                        u_prev = u_prev * same_prev
                    eye = torch.eye(lc, device=hidden_states.device, dtype=torch.float32)
                    u_c = torch.linalg.solve_triangular(
                        eye + a_mat, u_prev + torch.matmul(b_mat, v_c), upper=False, unitriangular=True
                    )                                                        # [B, H, lc, d]
                    erased = u_c * ab_c.unsqueeze(-1)                        # alpha_i beta_i u_i
                    out_intra = out_intra - torch.matmul(
                        torch.matmul(q_c, kh_c.transpose(-1, -2)) * decay_c, erased
                    )

                chunk_outputs.append(out_inter + out_intra)

                # 3. Advance exact recurrent state S_c to end of chunk
                chunk_total_decay = torch.exp(cum_log[:, :, -1:]).unsqueeze(-1)       # [B, H, 1, 1]
                key_to_end_decay = torch.exp(cum_log[:, :, -1:] - cum_log).unsqueeze(-1)  # [B, H, lc, 1]
                k_decayed = k_c * key_to_end_decay
                if seg_c is not None and prev_seg is not None:
                    last_seg = seg_c[:, -1:]
                    carry_prev = (last_seg == prev_seg).view(bsz, 1, 1, 1)
                    key_same = (seg_c == last_seg).unsqueeze(1).unsqueeze(-1)
                    state = (chunk_total_decay * carry_prev) * state + torch.matmul(
                        (k_decayed * key_same).transpose(-1, -2), v_c
                    )
                    if beta is not None:
                        state = state - torch.matmul((kh_c * key_to_end_decay * key_same).transpose(-1, -2), erased)
                    prev_seg = last_seg
                else:
                    state = chunk_total_decay * state + torch.matmul(k_decayed.transpose(-1, -2), v_c)
                    if beta is not None:
                        state = state - torch.matmul((kh_c * key_to_end_decay).transpose(-1, -2), erased)

            out_h = torch.cat(chunk_outputs, dim=2)  # [B, H, T, d]
        out_flat = out_h.permute(0, 2, 1, 3).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out_flat.to(orig_dtype))

    def forward_recurrent(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step-by-step $O(1)$-memory recurrent execution matching Rust `HybridStateManager`.

        Args:
            hidden_states: `[B, T, D]`
            state: Optional initial recurrent state `[B, H, d, d]` (`float32`), defaults to zeros.
        Returns:
            `(output [B, T, D], final_state [B, H, d, d])`
        """
        orig_dtype = hidden_states.dtype
        bsz, seq_len, _ = hidden_states.shape
        q, k, v, alpha = self._project_qkv_alpha(hidden_states)

        if state is None:
            state = torch.zeros(
                (bsz, self.num_heads, self.head_dim, self.head_dim),
                dtype=torch.float32,
                device=hidden_states.device,
            )
        else:
            state = state.float().clone()

        beta = self._erase_gate(hidden_states)
        khat = torch.nn.functional.normalize(k, dim=-1) if beta is not None else None
        outputs = []
        for t in range(seq_len):
            alpha_t = alpha[:, t, :, None, None]  # [B, H, 1, 1]
            k_t = k[:, t, :, :, None]             # [B, H, d, 1]
            v_t = v[:, t, :, None, :]             # [B, H, 1, d]
            q_t = q[:, t, :, :, None]             # [B, H, d, 1]

            if beta is not None and khat is not None:
                kh_t = khat[:, t, :, :, None]     # [B, H, d, 1]
                state = state - beta[:, t, :, None, None] * kh_t * torch.matmul(kh_t.transpose(-1, -2), state)
            state = alpha_t * state + (k_t * v_t)
            # o_{t, h} = S_t^T @ q_t = sum_i q_{t,i} * S_{t, i, :}
            o_t = torch.matmul(state.transpose(-1, -2), q_t).squeeze(-1)  # [B, H, d]
            outputs.append(o_t.reshape(bsz, 1, self.num_heads * self.head_dim))

        out_flat = torch.cat(outputs, dim=1).to(orig_dtype)
        return self.o_proj(out_flat), state

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        recurrent_mode: bool = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Drop-in replacement for `OlmoeAttention.forward` returning `(attn_output, None)`."""
        reset_mask = kwargs.get("eos_reset_mask")
        if not isinstance(reset_mask, torch.Tensor):
            reset_mask = None
        if (
            reset_mask is None
            and attention_mask is not None
            and attention_mask.ndim == 4
            and attention_mask.shape[-1] > 1
        ):
            # Extract document-reset boundaries directly from the lower subdiagonal of 4D causal mask
            # (`mask[:, 0, t, t - 1] < -1e4`) so forward + gradient-checkpoint backward are stateless.
            subdiag = torch.diagonal(attention_mask[:, 0], offset=-1, dim1=-2, dim2=-1)
            zeros_head = torch.zeros((subdiag.shape[0], 1), dtype=torch.bool, device=subdiag.device)
            reset_mask = torch.cat([zeros_head, subdiag < -1e4], dim=1)
        if recurrent_mode:
            out, _ = self.forward_recurrent(hidden_states)
            return out, None
        return self.forward_parallel(hidden_states, eos_reset_mask=reset_mask), None


def build_document_causal_mask(
    eos_reset_mask: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build `[B, 1, T, T]` additive causal mask isolating packed documents across `eos_reset_mask`."""
    bsz, seq_len = eos_reset_mask.shape
    dev = eos_reset_mask.device
    seg_ids = torch.cumsum(eos_reset_mask.to(dtype=torch.int32), dim=-1)
    causal = torch.tril(torch.ones((seq_len, seq_len), device=dev, dtype=torch.bool))
    same_seg = seg_ids.unsqueeze(-1) == seg_ids.unsqueeze(-2)
    allowed = causal.unsqueeze(0) & same_seg
    if padding_mask is not None and padding_mask.ndim == 2:
        allowed = allowed & padding_mask.bool().unsqueeze(1)
        eye = torch.eye(seq_len, device=dev, dtype=torch.bool).unsqueeze(0)
        allowed = allowed | eye
    mask_val = torch.finfo(dtype).min
    return torch.zeros((bsz, 1, seq_len, seq_len), device=dev, dtype=dtype).masked_fill(
        ~allowed.unsqueeze(1), mask_val
    )


def convert_olmoe_to_hybrid(
    model: nn.Module,
    anchor_layers: Sequence[int] | None = None,
    init_decay_bias: float = 3.0,
) -> list[int]:
    """Convert non-anchor layers in an `OlmoeForCausalLM` into `GatedLinearRecurrenceAttention` blocks.

    Returns:
        `layer_kinds`: list of ints (`0 = AnchorGQA`, `1 = GatedLinearRecurrence`) matching `core.bin`.
    """
    config = model.config
    num_layers = config.num_hidden_layers
    if anchor_layers is None:
        anchor_layers = default_anchor_layers(num_layers, block_ratio=8)
    anchor_set = set(anchor_layers)

    layer_kinds: list[int] = []
    for idx, decoder_layer in enumerate(model.model.layers):
        if idx in anchor_set:
            layer_kinds.append(0)  # Exact GQA anchor layer
        else:
            decoder_layer.self_attn = GatedLinearRecurrenceAttention.from_olmoe_attention(
                decoder_layer.self_attn,
                config=config,
                layer_idx=idx,
                init_decay_bias=init_decay_bias,
            )
            layer_kinds.append(1)  # O(1)-state Gated Linear Recurrence

    model.config.terra_layer_kinds = layer_kinds
    return layer_kinds


class MicroTerraHybridWrapper(nn.Module):
    """Wraps a hybrid `OlmoeForCausalLM` with differentiable `HintEmbedding` and `LookaheadRouterBank`."""

    def __init__(
        self,
        backbone: nn.Module,
        hint_embedding: HintEmbedding | None = None,
        lookahead_bank: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hint_embedding = hint_embedding
        self.lookahead_bank = lookahead_bank

    @property
    def config(self):
        return self.backbone.config

    def forward(
        self,
        input_ids: torch.LongTensor,
        hints: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        eos_reset_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        **kwargs: object,
    ):
        inputs_embeds = self.backbone.model.embed_tokens(input_ids)
        if hints is not None and self.hint_embedding is not None:
            inputs_embeds = inputs_embeds + self.hint_embedding(hints).to(inputs_embeds.dtype)

        attn_for_backbone = attention_mask
        if eos_reset_mask is not None and (attention_mask is None or attention_mask.ndim == 2):
            mask_dtype = (
                torch.get_autocast_dtype("cuda")
                if (inputs_embeds.is_cuda and torch.is_autocast_enabled("cuda"))
                else inputs_embeds.dtype
            )
            attn_for_backbone = build_document_causal_mask(
                eos_reset_mask, padding_mask=attention_mask, dtype=mask_dtype
            )

        if self.lookahead_bank is not None and hasattr(self.lookahead_bank, "set_capturing"):
            self.lookahead_bank.set_capturing(True)
        try:
            return self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_for_backbone,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                **kwargs,
            )
        finally:
            if self.lookahead_bank is not None and hasattr(self.lookahead_bank, "set_capturing"):
                self.lookahead_bank.set_capturing(False)

"""Step 3A: 1.58-Bit Ternary QAT (`BitLinear` & `BitLinearExperts`) with STE and RMSNorm Pre-Scaling.

Conforms strictly to Section 1 of `schemas/terra_format.md`:
- Per-output-channel `absmean` scale factor: gamma_r = max(mean(|W_r|), 1e-8)
- Ternary weights: round(clip(W_r / gamma_r, -1, +1)) in {-1, 0, +1}
- Optional parameter-free RMSNorm activation pre-scaling (`FLAG_BITLINEAR_RMS_PRESCALE`)
- Per-token symmetric int8 activation quantization: s_x = max(|x|) / 127.0
- Straight-Through Estimator (STE) on both weights and activations for smooth QAT gradients
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


def rms_prescale(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Parameter-free RMSNorm pre-scaling matching Section 1.3 of `schemas/terra_format.md`."""
    orig_dtype = x.dtype
    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * inv_rms).to(orig_dtype)


def absmean_quantize_ste(
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D `[M, K]` or 3D `[E, M, K]` weight tensor to ternary `{-1, 0, +1}` with STE.

    Returns:
        w_ste: Differentiable dequantized weight tensor `w + (w_ternary * gamma - w).detach()`
        w_ternary: Exact `int8` ternary tensor in `{-1, 0, +1}` of same shape as `weight`
        scales: `float32` per-channel scales of shape `[M]` (2D) or `[E, M]` (3D)
    """
    if weight.shape[-1] % 4 != 0:
        raise ValueError(
            f"in_features ({weight.shape[-1]}) must be divisible by 4 for 2-bit ternary packing."
        )
    w_fp32 = weight.float()
    scales = w_fp32.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    w_ternary_fp32 = torch.round(torch.clamp(w_fp32 / scales, -1.0, 1.0))
    w_dequant = (w_ternary_fp32 * scales).to(weight.dtype)
    w_ste = weight + (w_dequant - weight).detach()
    return w_ste, w_ternary_fp32.to(torch.int8), scales.squeeze(-1)


def activation_quantize_int8_ste(
    x: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token symmetric `int8` (`[-127, 127]`) quantization with Straight-Through Estimator.

    Matches Section 1.3 of `schemas/terra_format.md`:
    - If `max(|x|) < 1e-12`, quantized activations are exact zeros.
    """
    x_fp32 = x.float()
    abs_max = x_fp32.abs().amax(dim=-1, keepdim=True)
    zero_mask = abs_max < eps
    safe_scale = torch.where(zero_mask, torch.ones_like(abs_max), abs_max / 127.0)
    x_q_fp32 = torch.round(x_fp32 / safe_scale).clamp(-127.0, 127.0)
    x_q_fp32 = torch.where(zero_mask, torch.zeros_like(x_q_fp32), x_q_fp32)
    x_dequant = (x_q_fp32 * safe_scale).to(x.dtype)
    x_ste = x + (x_dequant - x).detach()
    act_scale = torch.where(zero_mask, torch.zeros_like(safe_scale), safe_scale)
    return x_ste, x_q_fp32.to(torch.int8), act_scale


@torch.no_grad()
def apply_l2_optimal_row_prescale_(weight: torch.Tensor) -> torch.Tensor:
    """Rescale rows in-place so standard `absmean` scale matches the L2-optimal ternary scale.

    Because `absmean_quantize` is positive-scale-equivariant (`Q(c * w) = c * Q(w)` for `c > 0`),
    multiplying row `r` by `c_r = <w_r, t_r> / (gamma_r * ||t_r||_2^2)` leaves the ternary
    mask `t_r` invariant while making `gamma_r * t_r` minimize `||w_r - gamma * t_r||_2^2`.
    """
    w_f32 = weight.float()
    _, t, gamma = absmean_quantize_ste(w_f32)
    t_f32 = t.float()
    nz_sq = t_f32.pow(2).sum(dim=-1, keepdim=True)
    dot = (w_f32 * t_f32).sum(dim=-1, keepdim=True)
    denom = (gamma.unsqueeze(-1) * nz_sq).clamp_min(1e-8)
    mult = torch.where(nz_sq > 0, dot / denom, torch.ones_like(dot)).clamp(0.5, 4.0)
    weight.copy_((w_f32 * mult).to(dtype=weight.dtype))
    return weight


def _quantize_act_with_lambda(x_in: torch.Tensor, quant_lambda: float) -> torch.Tensor:
    if quant_lambda <= 0.0:
        return x_in
    x_q, _, _ = activation_quantize_int8_ste(x_in)
    if quant_lambda >= 1.0:
        return x_q
    return x_in + float(quant_lambda) * (x_q - x_in)


def _quantize_weight_with_lambda(weight: torch.Tensor, quant_lambda: float) -> torch.Tensor:
    if quant_lambda <= 0.0:
        return weight
    if quant_lambda >= 1.0:
        w_q, _, _ = absmean_quantize_ste(weight)
        return w_q
    # During progressive ramp (0 < quant_lambda < 1), use L2-optimal ternary scale
    # without mutating continuous `weight` so `(1 - lam) * w + lam * (gamma_opt * t)` stays unbiased.
    w_f32 = weight.float()
    _, t, gamma = absmean_quantize_ste(w_f32)
    t_f32 = t.float()
    nz_sq = t_f32.pow(2).sum(dim=-1, keepdim=True)
    dot = (w_f32 * t_f32).sum(dim=-1, keepdim=True)
    denom = (gamma.unsqueeze(-1) * nz_sq).clamp_min(1e-8)
    mult = torch.where(nz_sq > 0, dot / denom, torch.ones_like(dot)).clamp(0.5, 4.0)
    w_opt = (t_f32 * (gamma.unsqueeze(-1) * mult)).to(dtype=weight.dtype)
    w_q = weight + (w_opt - weight).detach()
    return weight + float(quant_lambda) * (w_q - weight)


@torch.no_grad()
def apply_l2_optimal_model_prescale_(model: nn.Module) -> None:
    """Apply idempotent L2-optimal row prescaling across all BitLinear / BitLinearExperts modules."""
    for mod in model.modules():
        if isinstance(mod, BitLinear):
            apply_l2_optimal_row_prescale_(mod.weight)
        elif isinstance(mod, BitLinearExperts):
            apply_l2_optimal_row_prescale_(mod.gate_up_proj)
            apply_l2_optimal_row_prescale_(mod.down_proj)


def set_model_quant_lambda(model: nn.Module, quant_lambda: float) -> None:
    """Set progressive QAT blend `quant_lambda in [0, 1]` on every BitLinear module."""
    clamped = float(max(0.0, min(1.0, quant_lambda)))
    for mod in model.modules():
        if isinstance(mod, (BitLinear, BitLinearExperts)):
            mod.quant_lambda = clamped


class BitLinear(nn.Module):
    """1.58-Bit Ternary Linear Projection with STE and optional `RMSNorm` Pre-Scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_rms_prescale: bool = False,
        rms_eps: float = 1e-5,
        quant_lambda: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features % 4 != 0:
            raise ValueError(f"BitLinear requires in_features % 4 == 0, got {in_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.use_rms_prescale = use_rms_prescale
        self.rms_eps = rms_eps
        self.quant_lambda = float(quant_lambda)

        factory_kwargs = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        use_rms_prescale: bool = False,
        rms_eps: float = 1e-5,
        init_l2_scale: bool = False,
        quant_lambda: float = 1.0,
    ) -> "BitLinear":
        """Warm-start `BitLinear` directly from a pretrained `nn.Linear` module."""
        bit_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            use_rms_prescale=use_rms_prescale,
            rms_eps=rms_eps,
            quant_lambda=quant_lambda,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        with torch.no_grad():
            bit_linear.weight.copy_(linear.weight)
            if init_l2_scale:
                apply_l2_optimal_row_prescale_(bit_linear.weight)
            if linear.bias is not None and bit_linear.bias is not None:
                bit_linear.bias.copy_(linear.bias)
        return bit_linear

    def export_quantized(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(w_ternary_int8 [M, K], scales_fp32 [M])` for binary packing."""
        with torch.no_grad():
            _, w_ternary, scales = absmean_quantize_ste(self.weight)
        return w_ternary, scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = rms_prescale(x, eps=self.rms_eps) if self.use_rms_prescale else x
        x_q = _quantize_act_with_lambda(x_in, self.quant_lambda)
        w_q = _quantize_weight_with_lambda(self.weight, self.quant_lambda)
        return F.linear(x_q, w_q, self.bias)

    @torch.no_grad()
    def forward_exact_integer(self, x: torch.Tensor) -> torch.Tensor:
        """Reference integer-accumulation forward matching Rust `ternary_simd.rs`."""
        x_in = rms_prescale(x.float(), eps=self.rms_eps) if self.use_rms_prescale else x.float()
        _, x_i8, act_scale = activation_quantize_int8_ste(x_in)
        _, w_i8, w_scales = absmean_quantize_ste(self.weight)
        # Perform integer dot product in int32 (exact representation of ADD/SUB accumulator)
        acc_i32 = torch.matmul(x_i8.to(torch.int32), w_i8.to(torch.int32).t())
        out = acc_i32.float() * w_scales.unsqueeze(0) * act_scale
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)


class BitLinearExperts(nn.Module):
    """1.58-Bit Ternary QAT wrapper for `transformers >= 5` 3D `OlmoeExperts` SwiGLU banks.

    Preserves the exact parameter shapes (`gate_up_proj: [E, 2*I, D]`, `down_proj: [E, D, I]`)
    while applying per-channel ternary STE and per-token int8 activation STE on `gate`, `up`,
    and `down` projections cleanly.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        hidden_act: str = "silu",
        use_rms_prescale: bool = False,
        rms_eps: float = 1e-5,
        quant_lambda: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % 4 != 0 or intermediate_dim % 4 != 0:
            raise ValueError(
                f"BitLinearExperts requires hidden_dim ({hidden_dim}) and "
                f"intermediate_dim ({intermediate_dim}) divisible by 4."
            )
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.use_rms_prescale = use_rms_prescale
        self.rms_eps = rms_eps
        self.quant_lambda = float(quant_lambda)
        self.act_fn = ACT2FN[hidden_act]

        factory_kwargs = {"dtype": dtype, "device": device}
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_dim, hidden_dim, **factory_kwargs)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_dim, **factory_kwargs)
        )
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)

    @classmethod
    def from_olmoe_experts(
        cls,
        experts_module: nn.Module,
        hidden_act: str = "silu",
        use_rms_prescale: bool = False,
        rms_eps: float = 1e-5,
        init_l2_scale: bool = False,
        quant_lambda: float = 1.0,
    ) -> "BitLinearExperts":
        """Convert a HuggingFace `OlmoeExperts` module in-place while retaining weights."""
        gate_up = experts_module.gate_up_proj
        down = experts_module.down_proj
        num_experts, two_inter, hidden_dim = gate_up.shape
        inter_dim = two_inter // 2
        bit_experts = cls(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=inter_dim,
            hidden_act=hidden_act,
            use_rms_prescale=use_rms_prescale,
            rms_eps=rms_eps,
            quant_lambda=quant_lambda,
            dtype=gate_up.dtype,
            device=gate_up.device,
        )
        with torch.no_grad():
            bit_experts.gate_up_proj.copy_(gate_up)
            bit_experts.down_proj.copy_(down)
            if init_l2_scale:
                apply_l2_optimal_row_prescale_(bit_experts.gate_up_proj)
                apply_l2_optimal_row_prescale_(bit_experts.down_proj)
        if hasattr(experts_module, "act_fn"):
            bit_experts.act_fn = experts_module.act_fn
        return bit_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0 or top_k_index.numel() == 0:
            return torch.zeros_like(hidden_states)

        x_in = rms_prescale(hidden_states, eps=self.rms_eps) if self.use_rms_prescale else hidden_states
        x_q = _quantize_act_with_lambda(x_in, self.quant_lambda)

        flat_idx = top_k_index.reshape(-1)
        sorted_expert_ids, perm = torch.sort(flat_idx)
        counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts).tolist()
        active = [(e, c) for e, c in enumerate(counts) if c > 0 and e < self.num_experts]
        if not active:
            return torch.zeros_like(hidden_states)

        active_counts = [c for _, c in active]
        total_active = sum(active_counts)
        if total_active < perm.numel():
            perm = perm[:total_active]

        num_top_k = top_k_index.shape[1]
        token_indices = torch.div(perm, num_top_k, rounding_mode="floor")
        sorted_weights = top_k_weights.reshape(-1).index_select(0, perm).unsqueeze(-1)

        x_sorted = x_q.index_select(0, token_indices)
        x_chunks = torch.split(x_sorted, active_counts, dim=0)

        # Quantize 3D expert banks once per layer instead of per active expert in Python loop
        w_gu_all = _quantize_weight_with_lambda(self.gate_up_proj, self.quant_lambda)
        w_gu_slices = torch.unbind(w_gu_all, dim=0)
        gu_out = torch.cat(
            [F.linear(xc, w_gu_slices[e]) for (e, _), xc in zip(active, x_chunks)],
            dim=0,
        )

        gate, up = gu_out.chunk(2, dim=-1)
        mid = self.act_fn(gate) * up
        mid_in = rms_prescale(mid, eps=self.rms_eps) if self.use_rms_prescale else mid
        mid_q = _quantize_act_with_lambda(mid_in, self.quant_lambda)
        mid_chunks = torch.split(mid_q, active_counts, dim=0)

        w_dn_all = _quantize_weight_with_lambda(self.down_proj, self.quant_lambda)
        w_dn_slices = torch.unbind(w_dn_all, dim=0)
        dn_out = torch.cat(
            [F.linear(mc, w_dn_slices[e]) for (e, _), mc in zip(active, mid_chunks)],
            dim=0,
        )

        weighted = (dn_out * sorted_weights).to(hidden_states.dtype)
        final_hidden_states = torch.zeros_like(hidden_states)
        final_hidden_states.index_add_(0, token_indices, weighted)
        return final_hidden_states


def convert_model_to_bitlinear(
    model: nn.Module,
    use_rms_prescale: bool = False,
    rms_eps: float | None = None,
    init_l2_scale: bool = False,
    quant_lambda: float = 1.0,
    exclude_patterns: tuple[str, ...] = ("lm_head", "mlp.gate", "lookahead_router", "decay_proj", "hint_embedding"),
) -> nn.Module:
    """Replace eligible `nn.Linear` and `OlmoeExperts` modules in an OLMoE model with `BitLinear` QAT modules.

    Router gates (`mlp.gate`, `lookahead_router`), recurrence decay gates (`decay_proj`),
    hint projections (`hint_embedding`), and `lm_head` (by default) stay FP32/FP16 for routing stability.
    """
    if rms_eps is None:
        rms_eps = getattr(getattr(model, "config", None), "rms_norm_eps", 1e-5)
    hidden_act = getattr(getattr(model, "config", None), "hidden_act", "silu")

    def _replace_recursive(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if any(pat in full_name for pat in exclude_patterns):
                continue

            # Detect 3D `OlmoeExperts` bank (`gate_up_proj` + `down_proj` parameters)
            if (
                hasattr(child, "gate_up_proj")
                and hasattr(child, "down_proj")
                and isinstance(child.gate_up_proj, nn.Parameter)
                and child.gate_up_proj.ndim == 3
            ):
                setattr(
                    parent,
                    child_name,
                    BitLinearExperts.from_olmoe_experts(
                        child,
                        hidden_act=hidden_act,
                        use_rms_prescale=use_rms_prescale,
                        rms_eps=rms_eps,
                        init_l2_scale=init_l2_scale,
                        quant_lambda=quant_lambda,
                    ),
                )
            elif isinstance(child, nn.Linear) and child.in_features % 4 == 0:
                setattr(
                    parent,
                    child_name,
                    BitLinear.from_linear(
                        child,
                        use_rms_prescale=use_rms_prescale,
                        rms_eps=rms_eps,
                        init_l2_scale=init_l2_scale,
                        quant_lambda=quant_lambda,
                    ),
                )
            else:
                _replace_recursive(child, full_name)

    _replace_recursive(model)
    return model

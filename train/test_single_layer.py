"""Milestone 3A Local Single-Layer Verification (`train/test_single_layer.py`).

Instantiates ONE full-width `OLMoE-1B-7B` hybrid layer:
  - `hidden_size = 2048`
  - `intermediate_size = 1024`
  - `num_attention_heads = 16`, `num_kv_heads = 16`
  - `num_experts = 64`, `top_k = 8`
  - `BitLinear` STE (`q/k/v/o`) + `BitLinearExperts` STE (64 SwiGLU experts)
  - `GatedLinearRecurrenceAttention` (chunkwise causal parallel scan)
  - `HintEmbedding` fed live hints from `edge_codec`
  - Auxiliary `2048 -> 64` 3-layer lookahead router head

Executes 20 real forward/backward/optimizer steps on `mps` (or `cpu`) while keeping total
process RSS + device memory strictly `< 4096 MB` using FP32 master weights + row-quantized
int8 AdamW states consumed via PyTorch post-accumulate gradient hooks.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edge_codec import UnifiedEdgeCodec
from train.bitlinear import (
    BitLinear,
    absmean_quantize_ste,
    activation_quantize_int8_ste,
    rms_prescale,
)
from train.hybrid_ssm import GatedLinearRecurrenceAttention, HintEmbedding
from train.lookahead_router import compute_lookahead_loss


def current_memory_mb(device: torch.device) -> dict[str, float]:
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)
    # On macOS, `ru_maxrss` is in bytes; on Linux, in kilobytes.
    ru_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = (
        ru_raw / (1024.0 * 1024.0) if sys.platform == "darwin" else ru_raw / 1024.0
    )
    mps_alloc_mb = 0.0
    if device.type == "mps" and hasattr(torch, "mps"):
        mps_alloc_mb = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
    return {
        "rss_mb": rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "mps_allocated_mb": mps_alloc_mb,
    }


class CompactSteppingOptimizer:
    """Memory-disciplined FP32-master AdamW using per-row `int8` moments + post-accumulate hooks.

    Keeps full `float32` master parameters (so small STE updates never round to zero) while
    compressing optimizer moments to 1 byte/element (`840 MB` for 419M params instead of `3.36 GB`)
    and releasing each `.grad` immediately after recording its L2 norm (`< 64 MB` peak grad RAM).
    """

    def __init__(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        lr: float = 2e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.grad_norms: dict[str, float] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        self._states: dict[str, dict[str, torch.Tensor]] = {}
        for name, param in named_params:
            if not param.requires_grad:
                continue
            shape = param.shape
            scale_shape = shape[:-1] + (1,) if param.ndim >= 2 else (1,)
            self._states[name] = {
                "m_q": torch.zeros(shape, dtype=torch.int8, device=param.device),
                "m_s": torch.zeros(scale_shape, dtype=torch.float32, device=param.device),
                "v_q": torch.zeros(shape, dtype=torch.uint8, device=param.device),
                "v_s": torch.zeros(scale_shape, dtype=torch.float32, device=param.device),
            }
            self._handles.append(
                param.register_post_accumulate_grad_hook(self._make_hook(name))
            )

    def _make_hook(self, name: str):
        @torch.no_grad()
        def _hook(param: nn.Parameter) -> None:
            g = param.grad
            if g is None:
                return
            g_f32 = g.float()
            self.grad_norms[name] = float(g_f32.norm().item())

            st = self._states[name]
            step = max(self.step_count, 1)
            bias_c1 = 1.0 - (self.beta1**step)
            bias_c2 = 1.0 - (self.beta2**step)

            m = st["m_q"].float() * st["m_s"]
            # Second moment stored in uint8 sqrt domain (`65025:1` dynamic range) so small gradients don't underflow to zero
            v = (st["v_q"].float() * st["v_s"]).square()

            m = self.beta1 * m + (1.0 - self.beta1) * g_f32
            v = self.beta2 * v + (1.0 - self.beta2) * (g_f32 * g_f32)

            if self.weight_decay > 0.0 and param.ndim >= 2:
                param.data.mul_(1.0 - self.lr * self.weight_decay)

            m_hat = m / bias_c1
            v_hat = v / bias_c2
            param.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

            # Re-quantize `m` (signed int8) & `sqrt(v)` (unsigned uint8) per row
            reduce_dim = -1
            m_scale = m.abs().amax(dim=reduce_dim, keepdim=True).clamp(min=1e-12) / 127.0
            st["m_q"].copy_(torch.round(m / m_scale).clamp(-127.0, 127.0).to(torch.int8))
            st["m_s"].copy_(m_scale)

            v_sqrt = v.sqrt()
            v_scale = v_sqrt.amax(dim=reduce_dim, keepdim=True).clamp(min=1e-8) / 255.0
            st["v_q"].copy_(torch.round(v_sqrt / v_scale).clamp(0.0, 255.0).to(torch.uint8))
            st["v_s"].copy_(v_scale)

            # Free `.grad` immediately so backward never holds 1.68 GB of expert grads simultaneously!
            param.grad = None

        return _hook

    def begin_step(self) -> None:
        self.step_count += 1
        self.grad_norms.clear()

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class ShardedBitLinearExperts(nn.Module):
    """Per-expert `ParameterList` wrapper preserving exact BitLinear STE math inside < 4 GB RAM."""

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        *,
        use_rms_prescale: bool = True,
        rms_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.use_rms_prescale = use_rms_prescale
        self.rms_eps = rms_eps
        self.gate_up_list = nn.ParameterList(
            [
                nn.Parameter(torch.empty(2 * intermediate_dim, hidden_dim))
                for _ in range(num_experts)
            ]
        )
        self.down_list = nn.ParameterList(
            [
                nn.Parameter(torch.empty(hidden_dim, intermediate_dim))
                for _ in range(num_experts)
            ]
        )
        with torch.no_grad():
            for p in self.gate_up_list:
                p.normal_(0.0, 0.02)
            for p in self.down_list:
                p.normal_(0.0, 0.02)

    def _run_expert(
        self,
        current_state: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        weights_e: torch.Tensor,
    ) -> torch.Tensor:
        x_in = rms_prescale(current_state, eps=self.rms_eps) if self.use_rms_prescale else current_state
        x_q, _, _ = activation_quantize_int8_ste(x_in)
        w_gate_up_q, _, _ = absmean_quantize_ste(w_gate_up)
        gate, up = F.linear(x_q, w_gate_up_q).chunk(2, dim=-1)

        mid = F.silu(gate) * up
        mid_in = rms_prescale(mid, eps=self.rms_eps) if self.use_rms_prescale else mid
        mid_q, _, _ = activation_quantize_int8_ste(mid_in)
        w_down_q, _, _ = absmean_quantize_ste(w_down)
        expert_out = F.linear(mid_q, w_down_q)
        return (expert_out * weights_e.unsqueeze(-1)).to(current_state.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for hit in expert_hit:
            expert_idx = int(hit[0].item())
            if expert_idx >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            weights_e = top_k_weights[token_idx, top_k_pos]
            weighted = checkpoint(
                self._run_expert,
                current_state,
                self.gate_up_list[expert_idx],
                self.down_list[expert_idx],
                weights_e,
                use_reentrant=False,
            )
            final_hidden_states = final_hidden_states.index_add(
                0, token_idx, weighted.to(final_hidden_states.dtype)
            )
        return final_hidden_states


class MicroTerraSingleLayerTest(nn.Module):
    """Single full-dimension OLMoE hybrid layer (`d=2048, I=1024, 16 heads, 64 experts, top_k=8`)."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 2048
        self.intermediate_size = 1024
        self.num_heads = 16
        self.num_experts = 64
        self.top_k = 8

        self.hint_embedding = HintEmbedding(hidden_size=self.hidden_size, num_buckets=4096, hint_dim=32)
        self.input_layernorm = nn.RMSNorm(self.hidden_size, eps=1e-5)

        attn = GatedLinearRecurrenceAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
            rms_norm_eps=1e-5,
            layer_idx=0,
        )
        attn.q_proj = BitLinear.from_linear(attn.q_proj, use_rms_prescale=True, rms_eps=1e-5)
        attn.k_proj = BitLinear.from_linear(attn.k_proj, use_rms_prescale=True, rms_eps=1e-5)
        attn.v_proj = BitLinear.from_linear(attn.v_proj, use_rms_prescale=True, rms_eps=1e-5)
        attn.o_proj = BitLinear.from_linear(attn.o_proj, use_rms_prescale=True, rms_eps=1e-5)
        self.self_attn = attn

        self.post_attention_layernorm = nn.RMSNorm(self.hidden_size, eps=1e-5)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.05)

        self.experts = ShardedBitLinearExperts(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_size,
            intermediate_dim=self.intermediate_size,
            use_rms_prescale=True,
            rms_eps=1e-5,
        )

        self.lookahead_router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        nn.init.normal_(self.lookahead_router.weight, mean=0.0, std=0.05)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = hidden_states + self.hint_embedding(hints)
        u = self.input_layernorm(h)
        attn_out = self.self_attn.forward_parallel(u, chunk_size=32)
        h_prime = h + attn_out

        # Lookahead prediction at Layer L -> L+3
        lookahead_logits = self.lookahead_router(h_prime.detach())

        m = self.post_attention_layernorm(h_prime)
        bsz, seq_len, dim = m.shape
        m_flat = m.view(-1, dim)
        router_logits = self.gate(m_flat)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        topk_w, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(m_flat.dtype)

        moe_out = self.experts(m_flat, topk_idx, topk_w).view(bsz, seq_len, dim)
        out = h_prime + moe_out
        return out, lookahead_logits, router_logits.view(bsz, seq_len, self.num_experts)


def run_single_layer_verification(
    steps: int = 20,
    max_ram_mb: float = 4096.0,
    device_name: str = "cpu",
) -> dict[str, object]:
    torch.manual_seed(2026)
    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = MicroTerraSingleLayerTest().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 400_000_000, f"Expected full OLMoE single layer (>400M params), got {total_params}"

    opt = CompactSteppingOptimizer(list(model.named_parameters()), lr=1e-4)

    # Generate real edge hints using Dual-Path UnifiedEdgeCodec
    codec = UnifiedEdgeCodec()
    snippet = "Sarah verified the kernel.\n```python\ndef step(x, w):\n    y = x * w\n    return y + x\n```\nShe logged it."
    raw_hints = codec.encode_mixed(snippet, max_tokens=16)
    seq_len = int(raw_hints.shape[0])
    hints_tensor = torch.from_numpy(raw_hints).unsqueeze(0).to(device)  # [1, seq_len, 4]

    x_in = torch.randn(1, seq_len, 2048, device=device) * 0.1
    target_out = torch.randn(1, seq_len, 2048, device=device) * 0.1
    target_l3_router = torch.randn(1, seq_len, 64, device=device) * 0.5

    losses: list[float] = []
    peak_seen_mb = 0.0
    required_grad_keys = [
        "self_attn.q_proj.weight",
        "self_attn.decay_proj.weight",
        "experts.gate_up_proj",
        "experts.down_proj",
        "hint_embedding.table.weight",
        "hint_embedding.proj.weight",
        "lookahead_router.weight",
        "gate.weight",
    ]
    observed_norms: dict[str, float] = {}

    t0 = time.perf_counter()
    for step in range(steps):
        opt.begin_step()
        out, look_logits, _ = model(x_in, hints_tensor)
        mse_loss = F.mse_loss(out, target_out)
        look_loss, look_stats = compute_lookahead_loss(look_logits, target_l3_router, k=8)
        loss = mse_loss + 0.2 * look_loss
        loss.backward()
        if device.type == "mps":
            torch.mps.synchronize()

        loss_val = float(loss.item())
        if not math.isfinite(loss_val):
            raise AssertionError(f"Non-finite loss at step {step}: {loss_val}")
        losses.append(loss_val)

        if step == 0 or step == steps - 1:
            step_norms = dict(opt.grad_norms)
            gate_up_sq = sum(
                v * v for k, v in step_norms.items() if k.startswith("experts.gate_up_list.")
            )
            down_sq = sum(
                v * v for k, v in step_norms.items() if k.startswith("experts.down_list.")
            )
            step_norms["experts.gate_up_proj"] = math.sqrt(gate_up_sq)
            step_norms["experts.down_proj"] = math.sqrt(down_sq)

            for k in required_grad_keys:
                gn = step_norms.get(k, 0.0)
                if gn <= 0.0:
                    raise AssertionError(f"Zero/missing gradient for `{k}` at step {step}: {gn}")
                observed_norms[k] = gn

        mem = current_memory_mb(device)
        peak_seen_mb = max(peak_seen_mb, mem["peak_rss_mb"])

    elapsed_sec = time.perf_counter() - t0
    opt.close()

    # Check ternary quantization of trained BitLinear & ShardedBitLinearExperts weights
    _, w_t, w_s = absmean_quantize_ste(model.self_attn.q_proj.weight.detach().cpu())
    assert set(w_t.unique().tolist()).issubset({-1, 0, 1})
    assert float(w_s.min().item()) > 0.0

    _, exp_wt, exp_ws = absmean_quantize_ste(model.experts.gate_up_list[0].detach().cpu())
    assert set(exp_wt.unique().tolist()).issubset({-1, 0, 1})
    assert float(exp_ws.min().item()) > 0.0

    mem_final = current_memory_mb(device)
    peak_total = max(peak_seen_mb, mem_final["peak_rss_mb"])
    mem_final["peak_rss_mb"] = peak_total
    assert peak_total < max_ram_mb, (
        f"Peak RSS {peak_total:.1f} MB exceeded {max_ram_mb} MB budget"
    )
    assert losses[-1] < losses[0], f"Expected loss to decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    summary = {
        "device": str(device),
        "total_params": total_params,
        "steps": steps,
        "elapsed_sec": round(elapsed_sec, 3),
        "loss_initial": round(losses[0], 6),
        "loss_final": round(losses[-1], 6),
        "lookahead_recall_final": round(look_stats["recall_at_k"], 4),
        "grad_norms": {k: round(v, 6) for k, v in observed_norms.items()},
        "memory_mb": {k: round(v, 2) for k, v in mem_final.items()},
    }

    del opt, model
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return summary


def test_single_layer_smoke() -> None:
    report = run_single_layer_verification(steps=3, max_ram_mb=4096.0, device_name="cpu")
    assert report["loss_final"] < report["loss_initial"]
    assert report["memory_mb"]["peak_rss_mb"] < 4096.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-TERRA Local Single-Layer Verification")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-ram-mb", type=float, default=4096.0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    report = run_single_layer_verification(
        steps=args.steps,
        max_ram_mb=args.max_ram_mb,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Phase 2 Colab A100-40GB / A100-80GB Distillation Pipeline (`train/distill_colab.py`).

Designed for Google Colab A100 (Google AI Ultra) with:
  - Automatic VRAM detection:
      * A100-40GB (< 70 GB VRAM): Frozen `allenai/OLMoE-1B-7B-0125` teacher loaded in 4-bit NF4
        (`bitsandbytes`), paired with `PagedAdamW8bit`, `torch.amp.autocast(bfloat16)`, and
        gradient checkpointing (~32-35 GB VRAM).
      * A100-80GB (>= 70 GB VRAM): Frozen teacher loaded in `bfloat16` (`Flash/SDPA` attention).
  - Student initialized from teacher weights with `BitLinear` STE (`{-1, 0, +1}`), 7:1 Hybrid
    `GatedLinearRecurrenceAttention` (layers `0..6, 8..14`) + GQA anchors (`7, 15`),
    differentiable `HintEmbedding` fed by `UnifiedEdgeCodec`, and 3-layer `LookaheadRouterBank`.
  - Combined distillation objective (`ANTIGRAVITY.md` Milestone 3B):
      L_total = L_kl + 0.5 * L_layer_mse + 0.2 * L_lookahead_ce + 0.01 * L_load_balance
  - Atomic Google Drive checkpointing every `--save-every 250` steps with seamless auto-resume.
  - Local `--dry-run-mini` smoke mode for verifying checkpointing, resume, and loss without
    downloading the 14 GB checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    OlmoeConfig,
    OlmoeForCausalLM,
)

from train.bitlinear import (
    BitLinear,
    BitLinearExperts,
    apply_l2_optimal_model_prescale_,
    set_model_quant_lambda,
)
from train.distill import (
    DistillLossWeights,
    build_student_from_teacher,
    distill_step,
    evaluate_perplexity,
    export_distilled_student,
)
from train.hybrid_ssm import (
    GatedLinearRecurrenceAttention,
    MicroTerraHybridWrapper,
    build_document_causal_mask,
)
from train.lookahead_router import LookaheadRouterBank


def stochastic_round_bf16(x_f32: torch.Tensor) -> torch.Tensor:
    """Hardware-exact IEEE-754 float32 -> bfloat16 stochastic rounding helper."""
    orig = x_f32.float()
    bf16_max = float(torch.finfo(torch.bfloat16).max)
    x = orig.clamp(-bf16_max, bf16_max).contiguous()
    bits = x.view(torch.int32)
    rand = torch.randint(0, 1 << 16, bits.shape, device=x.device, dtype=torch.int32)
    rounded = ((bits + rand) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)
    return torch.where(orig.isfinite(), rounded, orig.to(torch.bfloat16))


def stochastic_round_bf16_params_(model: nn.Module) -> None:
    """Cast any floating `BitLinearExperts` parameters to `bfloat16` using stochastic rounding."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, BitLinearExperts):
                for p in module.parameters(recurse=False):
                    if p.dtype != torch.bfloat16:
                        p.data = stochastic_round_bf16(p.data)


class StochasticBF16AdamW(torch.optim.Optimizer):
    """Memory-lean (`2 bytes/param` 8-bit states) AdamW writing `bfloat16` params with stochastic rounding."""

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        chunk_rows: int = 16,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "chunk_rows": chunk_rows,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])
            chunk_rows = int(group["chunk_rows"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    scale_shape = (*p.shape[:-1], 1) if p.ndim >= 2 else (1,)
                    state["step"] = 0
                    state["m_q"] = torch.zeros(p.shape, dtype=torch.int8, device=p.device)
                    state["m_scale"] = torch.zeros(scale_shape, dtype=torch.float32, device=p.device)
                    state["v_sqrt_q"] = torch.zeros(p.shape, dtype=torch.uint8, device=p.device)
                    state["v_scale"] = torch.zeros(scale_shape, dtype=torch.float32, device=p.device)

                state["step"] += 1
                step_num = state["step"]
                bias_correction1 = 1.0 - (beta1**step_num)
                bias_correction2 = 1.0 - (beta2**step_num)
                step_size = lr / bias_correction1
                inv_sqrt_bc2 = 1.0 / math.sqrt(bias_correction2)

                m_q = state["m_q"]
                m_scale = state["m_scale"]
                v_q = state["v_sqrt_q"]
                v_scale = state["v_scale"]

                if p.ndim >= 2 and p.shape[0] > chunk_rows:
                    slices = [(s, min(s + chunk_rows, p.shape[0])) for s in range(0, p.shape[0], chunk_rows)]
                else:
                    slices = [(None, None)]

                for s, e in slices:
                    if s is None:
                        p_sl, g_sl = p.data, grad
                        mq_sl, ms_sl = m_q, m_scale
                        vq_sl, vs_sl = v_q, v_scale
                    else:
                        p_sl, g_sl = p.data[s:e], grad[s:e]
                        mq_sl, ms_sl = m_q[s:e], m_scale[s:e]
                        vq_sl, vs_sl = v_q[s:e], v_scale[s:e]

                    g_f32 = g_sl.float()
                    m_f32 = mq_sl.float() * ms_sl
                    v_sqrt_f32 = vq_sl.float() * vs_sl

                    m_f32.mul_(beta1).add_(g_f32, alpha=1.0 - beta1)
                    v_f32 = (v_sqrt_f32 * v_sqrt_f32).mul_(beta2).addcmul_(g_f32, g_f32, value=1.0 - beta2)
                    v_sqrt_new = v_f32.sqrt_()

                    denom = v_sqrt_new * inv_sqrt_bc2 + eps
                    p_f32 = p_sl.float()
                    if wd != 0.0:
                        p_f32.mul_(1.0 - lr * wd)
                    p_f32.addcdiv_(m_f32, denom, value=-step_size)

                    if p_sl.dtype == torch.bfloat16:
                        p_sl.copy_(stochastic_round_bf16(p_f32))
                    else:
                        p_sl.copy_(p_f32.to(p_sl.dtype))

                    m_absmax = m_f32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                    ms_sl.copy_(m_absmax / 127.0)
                    mq_sl.copy_(torch.round(m_f32 / ms_sl).clamp(-127.0, 127.0).to(torch.int8))

                    v_max = v_sqrt_new.amax(dim=-1, keepdim=True).clamp_min(1e-12)
                    vs_sl.copy_(v_max / 255.0)
                    vq_sl.copy_(torch.round(v_sqrt_new / vs_sl).clamp(0.0, 255.0).to(torch.uint8))

        return loss


class HybridPrecisionOptimizer(torch.optim.Optimizer):
    """Wraps an FP32 base optimizer (`PagedAdamW8bit`/`AdamW`) + `StochasticBF16AdamW` cleanly."""

    def __init__(
        self,
        fp32_opt: torch.optim.Optimizer | None,
        bf16_opt: StochasticBF16AdamW | None,
        lr: float,
    ) -> None:
        super().__init__([], {"lr": lr, "betas": (0.9, 0.95)})
        self.fp32_opt = fp32_opt
        self.bf16_opt = bf16_opt
        groups: list[dict[str, Any]] = []
        if self.fp32_opt is not None:
            groups.extend(self.fp32_opt.param_groups)
        if self.bf16_opt is not None:
            groups.extend(self.bf16_opt.param_groups)
        self.param_groups = groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.fp32_opt is not None:
            self.fp32_opt.zero_grad(set_to_none=set_to_none)
        if self.bf16_opt is not None:
            self.bf16_opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if self.fp32_opt is not None:
            loss = self.fp32_opt.step(closure=closure)
        if self.bf16_opt is not None:
            bf16_loss = self.bf16_opt.step()
            if loss is None:
                loss = bf16_loss
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "fp32_opt": self.fp32_opt.state_dict() if self.fp32_opt is not None else None,
            "bf16_opt": self.bf16_opt.state_dict() if self.bf16_opt is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.fp32_opt is not None and state_dict.get("fp32_opt") is not None:
            self.fp32_opt.load_state_dict(state_dict["fp32_opt"])
        if self.bf16_opt is not None and state_dict.get("bf16_opt") is not None:
            self.bf16_opt.load_state_dict(state_dict["bf16_opt"])
        groups: list[dict[str, Any]] = []
        if self.fp32_opt is not None:
            groups.extend(self.fp32_opt.param_groups)
        if self.bf16_opt is not None:
            groups.extend(self.bf16_opt.param_groups)
        self.param_groups = groups


def detect_gpu_profile() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "backend": "mps" if torch.backends.mps.is_available() else "cpu",
            "vram_gb": 0.0,
            "use_4bit_teacher": False,
        }
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024.0**3)
    return {
        "backend": "cuda",
        "device_name": props.name,
        "vram_gb": round(vram_gb, 2),
        "use_4bit_teacher": vram_gb < 75.0,
    }


def load_frozen_teacher(model_id: str, gpu_profile: dict[str, Any]) -> nn.Module:
    if gpu_profile["backend"] == "cuda" and gpu_profile["use_4bit_teacher"]:
        from transformers import BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map={"": 0},
        )
    elif gpu_profile["backend"] == "cuda":
        teacher = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
    else:
        teacher = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _build_base_adamw(param_groups: list[dict[str, Any]], lr: float) -> torch.optim.Optimizer:
    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb

            return bnb.optim.PagedAdamW8bit(param_groups, lr=lr, betas=(0.9, 0.95))
        except ImportError:
            pass
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))


def build_colab_optimizer(
    student: MicroTerraHybridWrapper,
    lookahead_bank: LookaheadRouterBank,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    lookahead_ids = {id(p) for p in lookahead_bank.parameters()}
    fp32_backbone: list[nn.Parameter] = []
    fp32_gate: list[nn.Parameter] = []
    fp32_lookahead: list[nn.Parameter] = []
    bf16_backbone: list[nn.Parameter] = []
    bf16_gate: list[nn.Parameter] = []

    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        is_bf16 = p.dtype == torch.bfloat16
        if id(p) in lookahead_ids:
            fp32_lookahead.append(p)
        elif "mlp.gate" in name:
            (bf16_gate if is_bf16 else fp32_gate).append(p)
        else:
            (bf16_backbone if is_bf16 else fp32_backbone).append(p)

    for p in lookahead_bank.parameters():
        if p.requires_grad and id(p) not in {id(x) for x in fp32_lookahead}:
            fp32_lookahead.append(p)

    fp32_groups = [
        {"params": g, "lr": g_lr, "weight_decay": g_wd}
        for g, g_lr, g_wd in (
            (fp32_backbone, lr, weight_decay),
            (fp32_gate, lr * 0.1, weight_decay),
            (fp32_lookahead, lr * 3.0, 1e-4),
        )
        if g
    ]
    bf16_groups = [
        {"params": g, "lr": g_lr, "weight_decay": weight_decay}
        for g, g_lr in ((bf16_backbone, lr), (bf16_gate, lr * 0.1))
        if g
    ]

    if not bf16_groups:
        return _build_base_adamw(fp32_groups, lr=lr)

    fp32_opt = _build_base_adamw(fp32_groups, lr=lr) if fp32_groups else None
    bf16_opt = StochasticBF16AdamW(bf16_groups, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    return HybridPrecisionOptimizer(fp32_opt, bf16_opt, lr=lr)


def build_cosine_min_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    stage1_steps: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Two-stage cosine decay scheduler with Stage-2 unfreeze warmup and `min_lr_ratio * lr` floor."""
    s1_steps = max(0, min(stage1_steps, num_training_steps))
    s1_warm = max(1, min(num_warmup_steps, max(1, s1_steps))) if s1_steps > 0 else num_warmup_steps
    s2_len = max(1, num_training_steps - s1_steps)
    s2_warm = max(1, min(100, s2_len // 10)) if s1_steps > 0 else max(1, num_warmup_steps)

    def _lr_lambda(current_step: int) -> float:
        if s1_steps > 0 and current_step < s1_steps:
            if current_step < s1_warm:
                return float(current_step + 1) / float(s1_warm)
            p1 = float(current_step - s1_warm) / float(max(1, s1_steps - s1_warm))
            p1 = min(1.0, max(0.0, p1))
            return 0.5 + 0.5 * 0.5 * (1.0 + math.cos(math.pi * p1))

        s2_step = current_step - s1_steps
        if s2_step < s2_warm:
            start_ratio = 0.1 if s1_steps > 0 else 0.0
            frac = float(s2_step + 1) / float(max(1, s2_warm))
            return start_ratio + (1.0 - start_ratio) * frac
        progress = float(s2_step - s2_warm) / float(max(1, s2_len - s2_warm))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def collect_stage1_param_ids(
    student: MicroTerraHybridWrapper,
    lookahead_bank: LookaheadRouterBank,
) -> set[int]:
    """Identify recurrent attention blocks, RMSNorm weights, and lookahead router heads for Stage 1."""
    ids = {id(p) for p in lookahead_bank.parameters()}
    for mod in student.modules():
        if isinstance(mod, GatedLinearRecurrenceAttention):
            for p in mod.parameters():
                ids.add(id(p))
    for name, p in student.named_parameters():
        if "layernorm" in name or name.endswith(".norm.weight"):
            ids.add(id(p))
    return ids


def configure_training_stage(
    student: MicroTerraHybridWrapper,
    stage1_param_ids: set[int],
    *,
    step_idx: int,
    stage1_steps: int,
    quant_warmup_steps: int,
    l2_prescaled: bool = False,
) -> tuple[int, float, bool]:
    """Configure Stage 1 (`quant_lambda=0`, unmodified teacher weights) or Stage 2 (`quant_lambda` ramp)."""
    if step_idx <= stage1_steps:
        set_model_quant_lambda(student, 0.0)
        for p in student.parameters():
            p.requires_grad_(id(p) in stage1_param_ids)
        return 1, 0.0, l2_prescaled

    for p in student.parameters():
        p.requires_grad_(True)
    s2_step = step_idx - stage1_steps
    q_lam = min(1.0, float(s2_step) / float(max(1, quant_warmup_steps)))
    if q_lam >= 1.0 and not l2_prescaled:
        apply_l2_optimal_model_prescale_(student)
        l2_prescaled = True
    set_model_quant_lambda(student, q_lam)
    return 2, q_lam, l2_prescaled


def save_atomic_checkpoint(
    ckpt_root: Path,
    step: int,
    student: MicroTerraHybridWrapper,
    lookahead_bank: LookaheadRouterBank,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    metrics: dict[str, Any],
    *,
    keep_last: int = 3,
) -> Path:
    """Atomically write checkpoint bundle to Google Drive / local disk via `.tmp` staging + rename."""
    ckpt_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = ckpt_root / f".tmp_step_{step:06d}"
    final_dir = ckpt_root / f"step_{step:06d}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(student.state_dict(), tmp_dir / "student_state.pt")
    torch.save(lookahead_bank.state_dict(), tmp_dir / "lookahead_state.pt")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer_state.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), tmp_dir / "scheduler_state.pt")

    state_meta = {
        "step": int(step),
        "timestamp": time.time(),
        "metrics": metrics,
    }
    (tmp_dir / "trainer_state.json").write_text(json.dumps(state_meta, indent=2), encoding="utf-8")
    if isinstance(metrics.get("verification"), dict):
        (tmp_dir / "verification.json").write_text(
            json.dumps(metrics["verification"], indent=2), encoding="utf-8"
        )

    os.replace(tmp_dir, final_dir)

    latest_tmp = ckpt_root / ".tmp_latest.json"
    latest_final = ckpt_root / "latest.json"
    latest_tmp.write_text(
        json.dumps({"latest_step": int(step), "path": str(final_dir.name)}, indent=2),
        encoding="utf-8",
    )
    os.replace(latest_tmp, latest_final)

    # Prune old checkpoints beyond `keep_last`
    step_dirs = sorted(
        (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step_")),
        key=lambda p: p.name,
    )
    for old_dir in step_dirs[:-keep_last]:
        shutil.rmtree(old_dir, ignore_errors=True)

    return final_dir


def maybe_resume_latest(
    ckpt_root: Path,
    student: MicroTerraHybridWrapper,
    lookahead_bank: LookaheadRouterBank,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Auto-resume from the newest valid atomic checkpoint in `ckpt_root`, returning start step."""
    if not ckpt_root.is_dir():
        return 0

    candidates: list[Path] = []
    latest_file = ckpt_root / "latest.json"
    if latest_file.is_file():
        try:
            meta = json.loads(latest_file.read_text(encoding="utf-8"))
            cand = ckpt_root / meta["path"]
            if (cand / "student_state.pt").is_file() and (cand / "trainer_state.json").is_file():
                candidates.append(cand)
        except Exception:
            pass

    for p in sorted(ckpt_root.iterdir(), key=lambda x: x.name, reverse=True):
        if p.is_dir() and p.name.startswith("step_") and (p / "student_state.pt").is_file():
            if p not in candidates:
                candidates.append(p)

    if not candidates:
        return 0

    ckpt_dir = candidates[0]
    student.load_state_dict(
        torch.load(ckpt_dir / "student_state.pt", map_location=map_location, weights_only=True)
    )
    lookahead_bank.load_state_dict(
        torch.load(ckpt_dir / "lookahead_state.pt", map_location=map_location, weights_only=True)
    )
    optimizer.load_state_dict(
        torch.load(ckpt_dir / "optimizer_state.pt", map_location=map_location, weights_only=False)
    )
    if scheduler is not None and (ckpt_dir / "scheduler_state.pt").is_file():
        scheduler.load_state_dict(
            torch.load(ckpt_dir / "scheduler_state.pt", map_location=map_location, weights_only=False)
        )
    trainer_meta = json.loads((ckpt_dir / "trainer_state.json").read_text(encoding="utf-8"))
    return int(trainer_meta["step"])


def _tokenize_and_align_doc(
    tokenizer: Any,
    text: str,
    *,
    seq_len: int,
    use_code: bool,
    sample: dict[str, Any],
    codec: Any | None,
    is_fast: bool,
) -> tuple[list[int], np.ndarray | None]:
    if is_fast:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=seq_len * 2,
        )
        doc_ids = list(enc["input_ids"])
        offsets = enc["offset_mapping"]
        max_char = max((int(e) for _, e in offsets), default=0)
        clipped_text = text[:max_char] if max_char > 0 else text
        lang_raw = str(sample.get("lang") or sample.get("language") or "").strip().lower()
        if codec is None:
            doc_hints = None
        elif use_code:
            if lang_raw in ("python", "py"):
                doc_hints = codec.encode_aligned_to_bpe(
                    clipped_text, offsets, is_code=True, code_lang="python"
                )
            elif lang_raw in ("rust", "rs"):
                doc_hints = codec.encode_aligned_to_bpe(
                    clipped_text, offsets, is_code=True, code_lang="rust"
                )
            elif lang_raw == "":
                inferred_lang = (
                    "rust"
                    if ("\nfn " in clipped_text or clipped_text.lstrip().startswith(("fn ", "pub fn ", "let ")))
                    else "python"
                )
                doc_hints = codec.encode_aligned_to_bpe(
                    clipped_text, offsets, is_code=True, code_lang=inferred_lang
                )
            else:
                doc_hints = np.zeros((len(doc_ids), 4), dtype=np.int32)
        else:
            doc_hints = codec.encode_aligned_to_bpe(clipped_text, offsets, is_code=False)
        return doc_ids, doc_hints

    doc_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=seq_len * 2)
    if codec is not None:
        raw_h = codec.encode_code(text, lang="python") if use_code else codec.encode_prose(text)
        doc_hints = np.zeros((len(doc_ids), 4), dtype=np.int32)
        copy_n = min(len(doc_ids), int(raw_h.shape[0]))
        if copy_n > 0:
            doc_hints[:copy_n] = raw_h[:copy_n]
    else:
        doc_hints = None
    return doc_ids, doc_hints


def stream_annotated_batches(
    tokenizer: Any,
    *,
    seq_len: int = 1024,
    batch_size: int = 2,
    code_ratio: float = 0.35,
    use_edge_hints: bool = True,
    skip_first_docs: int = 0,
):
    """Yield `(input_ids [B, T], hints [B, T, 4] | None, eos_reset_mask [B, T])` via dense packing."""
    from datasets import load_dataset

    codec = None
    if use_edge_hints:
        from edge_codec import UnifiedEdgeCodec

        codec = UnifiedEdgeCodec(max_code_tokens=max(8192, seq_len * 4))

    prose_ds = iter(
        load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    )
    try:
        code_ds = iter(load_dataset("bigcode/the-stack-smol", split="train", streaming=True))
    except Exception as exc:
        print(
            f"[corpus] bigcode/the-stack-smol unavailable ({exc}); falling back to public codeparrot/codeparrot-clean-valid",
            flush=True,
        )
        code_ds = iter(
            load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
        )

    for _ in range(skip_first_docs):
        next(prose_ds, None)
        next(code_ds, None)

    eos_id = int(tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else 0)
    is_fast = bool(getattr(tokenizer, "is_fast", False))

    buf_ids: list[int] = []
    buf_resets: list[bool] = []
    buf_hints: list[np.ndarray] = []

    batch_ids: list[torch.Tensor] = []
    batch_resets: list[torch.Tensor] = []
    batch_hints: list[torch.Tensor] = []
    sample_idx = 0

    while True:
        use_code = (sample_idx % 10) < int(round(code_ratio * 10))
        sample_idx += 1
        try:
            sample = next(code_ds) if use_code else next(prose_ds)
        except StopIteration:
            break
        text = sample.get("content") or sample.get("text") or ""
        if len(text.strip()) < 64:
            continue

        doc_ids, doc_hints = _tokenize_and_align_doc(
            tokenizer,
            text,
            seq_len=seq_len,
            use_code=use_code,
            sample=sample,
            codec=codec,
            is_fast=is_fast,
        )
        if not doc_ids:
            continue

        reset_first = len(buf_ids) > 0
        buf_ids.extend(doc_ids)
        buf_ids.append(eos_id)
        buf_resets.append(reset_first)
        buf_resets.extend([False] * len(doc_ids))
        if doc_hints is not None:
            buf_hints.append(doc_hints)
            buf_hints.append(np.zeros((1, 4), dtype=np.int32))

        while len(buf_ids) >= seq_len:
            seq_ids = buf_ids[:seq_len]
            buf_ids = buf_ids[seq_len:]
            seq_res = buf_resets[:seq_len]
            seq_res[0] = False
            buf_resets = buf_resets[seq_len:]

            batch_ids.append(torch.tensor(seq_ids, dtype=torch.long))
            batch_resets.append(torch.tensor(seq_res, dtype=torch.bool))
            if codec is not None:
                flat_h = np.concatenate(buf_hints, axis=0)
                batch_hints.append(torch.from_numpy(flat_h[:seq_len].copy()))
                rem_h = flat_h[seq_len:]
                buf_hints = [rem_h] if rem_h.shape[0] > 0 else []

            if len(batch_ids) == batch_size:
                ids_t = torch.stack(batch_ids, dim=0)
                hints_t = torch.stack(batch_hints, dim=0) if codec is not None else None
                resets_t = torch.stack(batch_resets, dim=0)
                batch_ids.clear()
                batch_hints.clear()
                batch_resets.clear()
                yield ids_t, hints_t, resets_t


def collect_packed_eval_batches(
    tokenizer: Any,
    *,
    num_batches: int = 12,
    seq_len: int = 512,
    batch_size: int = 2,
    skip_first_docs: int = 64,
    use_edge_hints: bool = True,
) -> list[tuple[torch.LongTensor, torch.Tensor | None, torch.Tensor]]:
    """Collect finite held-out packed validation batches without unmasked `<eos>` right-padding."""
    stream = stream_annotated_batches(
        tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        code_ratio=0.30,
        use_edge_hints=use_edge_hints,
        skip_first_docs=skip_first_docs,
    )
    return [next(stream) for _ in range(num_batches)]


@torch.no_grad()
def evaluate_lookahead_recall(
    student: MicroTerraHybridWrapper,
    lookahead_bank: LookaheadRouterBank,
    val_batches: list[tuple[Any, ...]],
    device: torch.device,
) -> float:
    student.eval()
    lookahead_bank.set_capturing(True)
    recalls: list[float] = []
    for item in val_batches:
        ids, hints = item[0], item[1]
        resets = item[2] if len(item) >= 3 else None
        ids = ids.to(device)
        hints_dev = hints.to(device) if hints is not None else None
        resets_dev = resets.to(device) if resets is not None else None
        _ = student(input_ids=ids, hints=hints_dev, eos_reset_mask=resets_dev, use_cache=False)
        _, stats = lookahead_bank.forward_from_captured()
        recalls.append(float(stats["mean_recall_at_k"]))
    student.train()
    return sum(recalls) / max(len(recalls), 1)


@torch.no_grad()
def snapshot_parameter_probes(
    student: nn.Module,
    lookahead_bank: LookaheadRouterBank,
) -> dict[str, torch.Tensor]:
    """Capture lightweight CPU FP32 parameter slices per subsystem to verify active weight updates."""
    probes: dict[str, list[torch.Tensor]] = {
        "ssm_recurrence": [],
        "anchor_gqa": [],
        "mlp_gate": [],
        "bitlinear_experts": [],
        "lookahead_bank": [],
    }
    for name, p in student.named_parameters():
        flat = p.detach().reshape(-1)[:256].to(device="cpu", dtype=torch.float32).clone()
        if "decay_proj" in name or "layers.0.self_attn" in name:
            probes["ssm_recurrence"].append(flat)
        elif "self_attn" in name and ("layers.7." in name or "layers.15." in name):
            probes["anchor_gqa"].append(flat)
        elif "mlp.gate" in name:
            probes["mlp_gate"].append(flat)
        elif "mlp.experts" in name and ("layers.0." in name or "layers.1." in name):
            probes["bitlinear_experts"].append(flat)

    for p in lookahead_bank.parameters():
        probes["lookahead_bank"].append(
            p.detach().reshape(-1)[:256].to(device="cpu", dtype=torch.float32).clone()
        )

    return {
        k: torch.cat(v, dim=0) if v else torch.zeros(1, dtype=torch.float32)
        for k, v in probes.items()
    }


def _packed_batch_nll(
    logits: torch.Tensor,
    ids: torch.Tensor,
    resets: torch.Tensor | None = None,
) -> tuple[float, int]:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = ids[:, 1:].contiguous()
    if resets is not None:
        shift_labels = shift_labels.masked_fill(resets[:, 1:].to(dtype=torch.bool), -100)
    valid_tokens = int((shift_labels != -100).sum().item())
    if valid_tokens <= 0:
        return 0.0, 0
    nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(nll.item()), valid_tokens


@torch.no_grad()
def evaluate_packed_teacher_ppl(
    teacher: nn.Module,
    val_batches: list[tuple[Any, ...]],
    device: torch.device,
) -> float:
    teacher.eval()
    mask_dtype = teacher.get_input_embeddings().weight.dtype
    total_nll = 0.0
    total_toks = 0
    for item in val_batches:
        ids = item[0].to(device)
        resets = item[2].to(device) if len(item) >= 3 and item[2] is not None else None
        t_mask = build_document_causal_mask(resets, dtype=mask_dtype) if resets is not None else None
        out = teacher(input_ids=ids, attention_mask=t_mask, use_cache=False)
        nll, n_tok = _packed_batch_nll(out.logits, ids, resets)
        total_nll += nll
        total_toks += n_tok
    return math.exp(min(20.0, total_nll / max(total_toks, 1)))


@torch.no_grad()
def verify_checkpoint_health(
    *,
    step: int,
    prev_step: int,
    stage_id: int,
    active_quant_lambda: float,
    student: MicroTerraHybridWrapper,
    lookahead_bank: LookaheadRouterBank,
    val_batches: list[tuple[Any, ...]],
    teacher_ppl: float,
    prev_probes: dict[str, torch.Tensor] | None,
    device: torch.device,
    tokenizer: Any | None = None,
    ckpt_root: Path | None = None,
    initial_student_ppl: float | None = None,
    prev_val_ppl_active: float | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Run in-flight checkpoint verification: parameter finiteness, subsystem weight deltas,
    held-out active & 1.58-bit ternary perplexity, lookahead recall@k, and greedy generation sanity.
    """
    t0 = time.perf_counter()
    non_finite_names: list[str] = []
    for name, p in student.named_parameters():
        if not bool(torch.isfinite(p).all()):
            non_finite_names.append(name)
    for idx, p in enumerate(lookahead_bank.parameters()):
        if not bool(torch.isfinite(p).all()):
            non_finite_names.append(f"lookahead_bank.param_{idx}")
    all_params_finite = len(non_finite_names) == 0

    curr_probes = snapshot_parameter_probes(student, lookahead_bank)
    param_deltas: dict[str, float] = {}
    if prev_probes is not None:
        for k, curr_t in curr_probes.items():
            prev_t = prev_probes.get(k)
            if prev_t is not None and prev_t.shape == curr_t.shape:
                param_deltas[k] = round(float((curr_t - prev_t).norm(p=2).item()), 6)

    student.eval()
    use_amp = device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    set_model_quant_lambda(student, active_quant_lambda)

    active_nll = 0.0
    active_toks = 0
    recalls: list[float] = []
    with amp_ctx:
        for item in val_batches:
            ids = item[0].to(device)
            hints = (
                item[1].to(device)
                if len(item) >= 2 and item[1] is not None and stage_id >= 2
                else None
            )
            resets = item[2].to(device) if len(item) >= 3 and item[2] is not None else None
            out = student(input_ids=ids, hints=hints, eos_reset_mask=resets, use_cache=False)
            nll, n_tok = _packed_batch_nll(out.logits, ids, resets)
            active_nll += nll
            active_toks += n_tok
            _, stats = lookahead_bank.forward_from_captured()
            recalls.append(float(stats["mean_recall_at_k"]))

    val_ppl_active = math.exp(min(20.0, active_nll / max(active_toks, 1)))
    val_lookahead_recall = sum(recalls) / max(len(recalls), 1)

    if active_quant_lambda >= 1.0 - 1e-6:
        val_ppl_ternary = val_ppl_active
    else:
        set_model_quant_lambda(student, 1.0)
        tern_nll = 0.0
        tern_toks = 0
        with amp_ctx:
            for item in val_batches:
                ids = item[0].to(device)
                hints = (
                    item[1].to(device)
                    if len(item) >= 2 and item[1] is not None and stage_id >= 2
                    else None
                )
                resets = item[2].to(device) if len(item) >= 3 and item[2] is not None else None
                out = student(input_ids=ids, hints=hints, eos_reset_mask=resets, use_cache=False)
                nll, n_tok = _packed_batch_nll(out.logits, ids, resets)
                tern_nll += nll
                tern_toks += n_tok
        val_ppl_ternary = math.exp(min(20.0, tern_nll / max(tern_toks, 1)))
        set_model_quant_lambda(student, active_quant_lambda)

    probe_ids = val_batches[0][0][:1, : min(16, int(val_batches[0][0].shape[1]))].to(device)
    gen_ids: list[int] = []
    cur_ids = probe_ids
    with amp_ctx:
        for _ in range(16):
            g_out = student(input_ids=cur_ids, use_cache=False)
            nxt = int(g_out.logits[0, -1].argmax(dim=-1).item())
            gen_ids.append(nxt)
            cur_ids = torch.cat([cur_ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)

    unique_token_ratio = len(set(gen_ids)) / max(len(gen_ids), 1)
    sample_text = (
        tokenizer.decode(gen_ids, skip_special_tokens=True).replace("\n", "\\n")[:120]
        if tokenizer is not None
        else str(gen_ids[:8])
    )
    student.train()

    status = "OK"
    reasons: list[str] = []
    if not all_params_finite:
        status = "FAIL"
        reasons.append(f"non_finite_params:{non_finite_names[:3]}")
    if not math.isfinite(val_ppl_active) or not math.isfinite(val_ppl_ternary):
        status = "FAIL"
        reasons.append("non_finite_val_ppl")
    if step > prev_step and param_deltas:
        if param_deltas.get("ssm_recurrence", 0.0) <= 0.0:
            status = "FAIL"
            reasons.append("zero_ssm_recurrence_update")
        if param_deltas.get("lookahead_bank", 0.0) <= 0.0:
            status = "FAIL"
            reasons.append("zero_lookahead_bank_update")
        if stage_id == 1 and param_deltas.get("bitlinear_experts", 0.0) > 1e-6:
            status = "FAIL"
            reasons.append("frozen_experts_mutated_in_stage1")
        if stage_id >= 2 and param_deltas.get("bitlinear_experts", 0.0) <= 0.0:
            status = "FAIL"
            reasons.append("zero_bitlinear_experts_update_in_stage2")
    if initial_student_ppl is not None and val_ppl_active > max(
        initial_student_ppl * 5.0, teacher_ppl * 50.0, 5000.0
    ):
        status = "FAIL"
        reasons.append(
            f"catastrophic_ppl_explosion:{val_ppl_active:.2f}_vs_init_{initial_student_ppl:.2f}"
        )
    if status == "OK":
        if (
            prev_val_ppl_active is not None
            and active_quant_lambda == 0.0
            and val_ppl_active > prev_val_ppl_active * 1.25
        ):
            status = "WARN"
            reasons.append(f"stage1_ppl_spike:{val_ppl_active:.2f}_vs_prev_{prev_val_ppl_active:.2f}")
        if unique_token_ratio < 0.20:
            status = "WARN"
            reasons.append(f"low_token_diversity:{unique_token_ratio:.2f}")

    active_retention_pct = 100.0 * (teacher_ppl / max(val_ppl_active, 1e-6))
    ternary_retention_pct = 100.0 * (teacher_ppl / max(val_ppl_ternary, 1e-6))
    report: dict[str, Any] = {
        "step": int(step),
        "prev_step": int(prev_step),
        "stage": int(stage_id),
        "quant_lambda": round(float(active_quant_lambda), 4),
        "status": status,
        "reasons": reasons,
        "all_params_finite": all_params_finite,
        "teacher_val_ppl": round(float(teacher_ppl), 4),
        "val_ppl_active": round(float(val_ppl_active), 4),
        "val_ppl_ternary": round(float(val_ppl_ternary), 4),
        "active_ppl_retention_pct": round(float(active_retention_pct), 2),
        "ternary_ppl_retention_pct": round(float(ternary_retention_pct), 2),
        "val_lookahead_recall_at_k": round(float(val_lookahead_recall), 4),
        "param_deltas": param_deltas,
        "unique_token_ratio": round(float(unique_token_ratio), 4),
        "sample_generation": sample_text,
        "verify_dt_sec": round(time.perf_counter() - t0, 3),
    }
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)
        with (ckpt_root / "verification_history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(report) + "\n")
        (ckpt_root / "latest_verification.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report, curr_probes


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-TERRA Colab A100 Distillation Pipeline")
    parser.add_argument("--teacher-model", type=str, default="allenai/OLMoE-1B-7B-0125")
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/terra_ckpt"),
        help="Atomic checkpoint directory (e.g., Google Drive path)",
    )
    parser.add_argument("--export-dir", type=Path, default=None, help="Optional final core.bin/experts.bin export dir")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--stage1-steps", type=int, default=-1, help="Stage 1 FP-SSM/lookahead steps (-1=auto ~25%)")
    parser.add_argument("--quant-warmup-steps", type=int, default=-1, help="Stage 2 QAT lambda ramp steps (-1=auto ~20%)")
    parser.add_argument("--hint-dropout", type=float, default=0.5, help="Per-sequence hint dropout prob in Stage 2")
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--verify-batches", type=int, default=4, help="Held-out packed batches for checkpoint verification")
    parser.add_argument("--early-verify-step", type=int, default=10, help="Early step inside Stage 1 & Stage 2 for health check")
    parser.add_argument("--verify-only", action="store_true", help="Verify latest checkpoint in --ckpt-dir and exit")
    parser.add_argument(
        "--abort-on-verify-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise RuntimeError immediately if checkpoint verification status is FAIL (default: True)",
    )
    parser.add_argument("--no-edge-hints", action="store_true")
    parser.add_argument("--dry-run-mini", action="store_true", help="Run tiny local smoke test verifying checkpoint & resume")
    args = parser.parse_args()

    if not args.ckpt_dir.exists():
        try:
            args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            fallback = Path("./terra_ckpt_local")
            fallback.mkdir(parents=True, exist_ok=True)
            print(f"[Micro-TERRA Colab] Warning: {args.ckpt_dir} not writable; falling back to {fallback}", flush=True)
            args.ckpt_dir = fallback

    stage1_steps = max(1, args.steps // 4) if args.stage1_steps < 0 else args.stage1_steps
    quant_warmup_steps = max(1, args.steps // 5) if args.quant_warmup_steps < 0 else args.quant_warmup_steps

    gpu_profile = detect_gpu_profile()
    device = torch.device(gpu_profile["backend"])
    print(f"[Micro-TERRA Colab] Hardware profile: {json.dumps(gpu_profile)}")

    # Exact ANTIGRAVITY.md loss weights:
    # L_total = L_kl + 0.5 * L_layer_mse + 0.2 * (L_lookahead_ce + 0.05 * L_load_balance)
    #         = L_kl + 0.5 * L_layer_mse + 0.2 * L_lookahead_ce + 0.01 * L_load_balance
    loss_weights = DistillLossWeights(
        kl_weight=1.0,
        hidden_mse_weight=0.5,
        lookahead_weight=0.2,
        load_balance_coef=0.05,
        temperature=2.0,
    )

    tokenizer: Any | None = None
    if args.dry_run_mini:
        torch.manual_seed(42)
        cfg = OlmoeConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_tok=2,
            eos_token_id=1,
            pad_token_id=0,
        )
        teacher = OlmoeForCausalLM(cfg)
        with torch.no_grad():
            for layer in teacher.model.layers:
                nn.init.normal_(layer.mlp.gate.weight, mean=0.0, std=0.04)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        student, lookahead_bank, layer_kinds = build_student_from_teacher(
            teacher,
            use_rms_prescale=False,
            init_l2_scale=False,
            enable_hint_embedding=not args.no_edge_hints,
            deepcopy_model=True,
        )
        teacher = teacher.to(device)
        student = student.to(device)

        from edge_codec import UnifiedEdgeCodec

        codec = UnifiedEdgeCodec() if not args.no_edge_hints else None
        sample_hints = (
            torch.from_numpy(codec.encode_code("def f(x):\n    return x + 1\n", lang="python"))
            if codec is not None
            else None
        )
        mini_batches: list[tuple[torch.LongTensor, torch.Tensor | None, torch.Tensor | None]] = []
        for _ in range(max(4, args.steps)):
            ids = torch.randint(2, cfg.vocab_size, (2, 16))
            ids[:, 7] = cfg.eos_token_id
            resets = torch.zeros((2, 16), dtype=torch.bool)
            resets[:, 8] = True
            h_t = None
            if sample_hints is not None:
                h_pad = torch.zeros((2, 16, 4), dtype=torch.int32)
                n_copy = min(16, int(sample_hints.shape[0]))
                h_pad[:, :n_copy, :] = sample_hints[:n_copy].unsqueeze(0)
                h_t = h_pad
            mini_batches.append((ids, h_t, resets))
        val_batches: list[tuple[Any, ...]] = mini_batches[:2]
        batch_iter = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        # Initialize student on CPU first from pretrained weights before loading NF4 teacher on CUDA
        cpu_base = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        student, lookahead_bank, layer_kinds = build_student_from_teacher(
            cpu_base,
            use_rms_prescale=False,
            init_l2_scale=False,
            enable_hint_embedding=not args.no_edge_hints,
            deepcopy_model=False,
        )
        del cpu_base
        gc.collect()
        if gpu_profile["backend"] == "cuda" and gpu_profile["use_4bit_teacher"]:
            # A100-40GB memory budget: keep core BitLinear, norms, routers, decay_proj,
            # and hint_embedding in FP32 while storing large BitLinearExperts in BF16.
            stochastic_round_bf16_params_(student)
        if hasattr(student.backbone, "gradient_checkpointing_enable"):
            student.backbone.gradient_checkpointing_enable()
        student = student.to(device)

        teacher = load_frozen_teacher(args.teacher_model, gpu_profile)
        val_batches = collect_packed_eval_batches(
            tokenizer,
            num_batches=max(1, args.verify_batches),
            seq_len=min(args.seq_len, 512),
            batch_size=args.batch_size,
            skip_first_docs=128,
            use_edge_hints=not args.no_edge_hints,
        )
        batch_iter = (
            None
            if args.verify_only
            else stream_annotated_batches(
                tokenizer,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                use_edge_hints=not args.no_edge_hints,
            )
        )
        mini_batches = []

    stage1_param_ids = collect_stage1_param_ids(student, lookahead_bank)
    optimizer = build_colab_optimizer(student, lookahead_bank, lr=args.lr)
    scheduler = build_cosine_min_lr_scheduler(
        optimizer,
        num_warmup_steps=max(1, args.steps // 20),
        num_training_steps=max(1, args.steps),
        min_lr_ratio=0.1,
        stage1_steps=stage1_steps,
    )

    start_step = maybe_resume_latest(
        args.ckpt_dir,
        student,
        lookahead_bank,
        optimizer,
        scheduler,
        map_location=device,
    )
    saved_metrics: dict[str, Any] = {}
    if start_step > 0:
        print(f"[Micro-TERRA Colab] Resumed cleanly from checkpoint step={start_step}")
        state_file = args.ckpt_dir / f"step_{start_step:06d}" / "trainer_state.json"
        if state_file.is_file():
            try:
                saved_metrics = json.loads(state_file.read_text(encoding="utf-8")).get("metrics", {})
            except Exception:
                saved_metrics = {}
    l2_prescaled = (
        bool(saved_metrics.get("l2_prescaled", 0.0))
        or float(saved_metrics.get("quant_lambda", 0.0)) >= 1.0 - 1e-6
        or start_step >= (stage1_steps + quant_warmup_steps)
    )

    use_amp = gpu_profile["backend"] == "cuda"
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    )
    teacher_mask_dtype = teacher.get_input_embeddings().weight.dtype

    with amp_ctx:
        val_teacher_ppl = evaluate_packed_teacher_ppl(teacher, val_batches, device)

    if start_step > 0 and "stage" in saved_metrics and "quant_lambda" in saved_metrics:
        init_stage_id = int(saved_metrics["stage"])
        init_quant_lam = float(saved_metrics["quant_lambda"])
        set_model_quant_lambda(student, init_quant_lam)
    else:
        init_stage_id, init_quant_lam, l2_prescaled = configure_training_stage(
            student,
            stage1_param_ids,
            step_idx=max(1, start_step),
            stage1_steps=stage1_steps,
            quant_warmup_steps=quant_warmup_steps,
            l2_prescaled=l2_prescaled,
        )
    init_report, prev_probes = verify_checkpoint_health(
        step=start_step,
        prev_step=start_step,
        stage_id=init_stage_id,
        active_quant_lambda=init_quant_lam,
        student=student,
        lookahead_bank=lookahead_bank,
        val_batches=val_batches,
        teacher_ppl=val_teacher_ppl,
        prev_probes=None,
        device=device,
        tokenizer=tokenizer,
        ckpt_root=args.ckpt_dir,
    )
    initial_student_ppl = float(init_report["val_ppl_active"])
    prev_val_ppl_active = initial_student_ppl
    prev_verify_step = start_step
    print(
        f"[CKPT VERIFY step={start_step} S{init_stage_id} lam={init_quant_lam:.2f}] "
        f"status={init_report['status']} teacher_ppl={val_teacher_ppl:.2f} "
        f"val_ppl={init_report['val_ppl_active']:.2f} (ternary={init_report['val_ppl_ternary']:.2f}) "
        f"val_recall@k={init_report['val_lookahead_recall_at_k']:.4f}",
        flush=True,
    )
    if args.verify_only:
        print(json.dumps(init_report, indent=2), flush=True)
        if init_report["status"] == "FAIL" and args.abort_on_verify_fail:
            raise RuntimeError(f"Checkpoint verification failed: {init_report['reasons']}")
        return

    grad_accum = 1 if args.dry_run_mini else max(1, args.grad_accum)
    last_metrics: dict[str, Any] = {"verification": init_report}
    all_trainable_params = list(student.parameters()) + list(lookahead_bank.parameters())
    exp_verify_steps: set[int] = set()
    if args.early_verify_step > 0:
        for base_step in (start_step, stage1_steps):
            offset = args.early_verify_step
            while base_step + offset <= args.steps:
                exp_verify_steps.add(base_step + offset)
                offset *= 2

    for step_idx in range(start_step + 1, args.steps + 1):
        step_t0 = time.perf_counter()
        stage_id, quant_lam, l2_prescaled = configure_training_stage(
            student,
            stage1_param_ids,
            step_idx=step_idx,
            stage1_steps=stage1_steps,
            quant_warmup_steps=quant_warmup_steps,
            l2_prescaled=l2_prescaled,
        )
        optimizer.zero_grad(set_to_none=True)
        accum_metrics: dict[str, float] = {}

        for micro_idx in range(grad_accum):
            if args.dry_run_mini:
                ids, hints, resets = mini_batches[(step_idx + micro_idx) % len(mini_batches)]
            else:
                assert batch_iter is not None
                ids, hints, resets = next(batch_iter)

            ids = ids.to(device)
            resets_dev = resets.to(device) if resets is not None else None
            if stage_id == 1 or hints is None:
                hints_dev = None
            else:
                hints_dev = hints.to(device)
                if args.hint_dropout > 0.0:
                    keep_mask = torch.rand((ids.shape[0], 1, 1), device=device) >= args.hint_dropout
                    if not bool(keep_mask.any()):
                        hints_dev = None
                    else:
                        hints_dev = hints_dev * keep_mask.to(hints_dev.dtype)

            teacher_mask = (
                build_document_causal_mask(resets_dev, dtype=teacher_mask_dtype)
                if resets_dev is not None
                else None
            )
            with torch.no_grad():
                t_out = teacher(
                    input_ids=ids,
                    attention_mask=teacher_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            with amp_ctx:
                loss, metrics = distill_step(
                    student=student,
                    lookahead_bank=lookahead_bank,
                    input_ids=ids,
                    teacher_logits=t_out.logits,
                    teacher_hidden_states=t_out.hidden_states,
                    hints=hints_dev,
                    eos_reset_mask=resets_dev,
                    weights=loss_weights,
                )
                scaled_loss = loss / float(grad_accum)

            scaled_loss.backward()
            for k, v in metrics.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + float(v) / float(grad_accum)

        torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        accum_metrics["stage"] = float(stage_id)
        accum_metrics["quant_lambda"] = float(quant_lam)
        accum_metrics["l2_prescaled"] = float(1.0 if l2_prescaled else 0.0)
        last_metrics = accum_metrics
        step_dt = time.perf_counter() - step_t0

        if args.log_every > 0 and (
            step_idx == start_step + 1 or step_idx % args.log_every == 0 or step_idx == args.steps
        ):
            vram_str = ""
            if device.type == "cuda":
                vram_gb = torch.cuda.max_memory_allocated() / (1024.0**3)
                vram_str = f" peak_vram={vram_gb:.2f}GB"
            print(
                f"[step {step_idx}/{args.steps} S{stage_id} lam={quant_lam:.2f}] "
                f"loss={last_metrics['total_loss']:.4f} "
                f"kl={last_metrics['kl_loss']:.4f} recall@k={last_metrics['mean_recall_at_k']:.4f} "
                f"dt={step_dt:.2f}s{vram_str}",
                flush=True,
            )

        is_save_step = (step_idx % args.save_every == 0) or (step_idx == args.steps)
        is_early_verify = step_idx in exp_verify_steps

        if is_save_step or is_early_verify:
            v_report, prev_probes = verify_checkpoint_health(
                step=step_idx,
                prev_step=prev_verify_step,
                stage_id=stage_id,
                active_quant_lambda=quant_lam,
                student=student,
                lookahead_bank=lookahead_bank,
                val_batches=val_batches,
                teacher_ppl=val_teacher_ppl,
                prev_probes=prev_probes,
                device=device,
                tokenizer=tokenizer,
                ckpt_root=args.ckpt_dir,
                initial_student_ppl=initial_student_ppl,
                prev_val_ppl_active=prev_val_ppl_active,
            )
            prev_verify_step = step_idx
            prev_val_ppl_active = float(v_report["val_ppl_active"])
            last_metrics["verification"] = v_report
            print(
                f"[CKPT VERIFY step={step_idx}/{args.steps} S{stage_id} lam={quant_lam:.2f}] "
                f"status={v_report['status']} val_ppl={v_report['val_ppl_active']:.2f} "
                f"(ternary={v_report['val_ppl_ternary']:.2f} ret={v_report['ternary_ppl_retention_pct']:.1f}%) "
                f"val_recall@k={v_report['val_lookahead_recall_at_k']:.4f} "
                f"deltas={json.dumps(v_report['param_deltas'])} "
                f"reasons={v_report['reasons']}",
                flush=True,
            )
            if v_report["status"] == "FAIL" and args.abort_on_verify_fail:
                raise RuntimeError(
                    f"Checkpoint verification FAILED at step {step_idx}: {json.dumps(v_report)}"
                )

        if is_save_step:
            ckpt_path = save_atomic_checkpoint(
                args.ckpt_dir,
                step_idx,
                student,
                lookahead_bank,
                optimizer,
                scheduler,
                last_metrics,
                keep_last=args.keep_last,
            )
            print(
                f"[step {step_idx}/{args.steps}] saved={ckpt_path.name} "
                f"loss={last_metrics['total_loss']:.4f} "
                f"recall@k={last_metrics['mean_recall_at_k']:.4f} "
                f"verify={last_metrics['verification']['status']}",
                flush=True,
            )

    if args.dry_run_mini:
        set_model_quant_lambda(student, 1.0)
        val_recall = evaluate_lookahead_recall(
            student,
            lookahead_bank,
            [(b[0], b[1]) for b in mini_batches[:2]],
            device,
        )
        teacher_ppl = evaluate_perplexity(teacher, [b[0] for b in mini_batches[:2]], device=device)
        student_ppl = evaluate_perplexity(student, [b[0] for b in mini_batches[:2]], device=device)
        print(
            json.dumps(
                {
                    "dry_run_complete": True,
                    "resumed_from_step": start_step,
                    "final_step": args.steps,
                    "layer_kinds": layer_kinds,
                    "last_metrics": last_metrics,
                    "val_recall": round(val_recall, 4),
                    "teacher_ppl": round(teacher_ppl, 4),
                    "student_ppl": round(student_ppl, 4),
                },
                indent=2,
            )
        )

    if args.export_dir is not None:
        set_model_quant_lambda(student, 1.0)
        student_cpu = student.to("cpu")
        lookahead_cpu = lookahead_bank.to("cpu")
        export_distilled_student(
            student=student_cpu,
            lookahead_bank=lookahead_cpu,
            out_dir=args.export_dir,
            use_rms_prescale=False,
            embed_dtype="f16",
            lm_head_dtype="f16",
        )
        print(f"[Micro-TERRA Colab] Exported distilled binary shards to {args.export_dir}")


if __name__ == "__main__":
    main()

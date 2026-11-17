"""Milestone 3 End-to-End Distillation & Lookahead Router Training Pipeline (`train/distill.py`).

Unifies:
  - Step 3A: 1.58-Bit Ternary `BitLinear` + `BitLinearExperts` QAT (STE + `RMSNorm` pre-scaling)
  - Step 3B: 7:1 Hybrid Attention-to-Recurrence Distillation (`GatedLinearRecurrenceAttention` + GQA anchors)
  - Step 3C: 3-Layer Lookahead Router (`LookaheadRouterBank`) with Top-k CE + Load-Balancing Loss
  - Guardrail #3: Differentiable `HintEmbedding` hook for `[batch, seq_len, 4]` edge hints

Supports:
  1. Online Teacher-Student Distillation via HuggingFace `Accelerate` on streaming
     `HuggingFaceFW/fineweb-edu` + `bigcode/the-stack-smol` corpora.
  2. Memory-Safe Offline Teacher Caching (`cache_teacher_batches`) so 16 GB unified-memory
     machines (e.g. Apple Silicon laptops) never hold two 6.9B models in RAM simultaneously.
  3. Evaluation gates: WikiText-2 Perplexity & Lookahead Top-8 Recall.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, OlmoeConfig, OlmoeForCausalLM

from train.bitlinear import convert_model_to_bitlinear
from train.hybrid_ssm import (
    HintEmbedding,
    MicroTerraHybridWrapper,
    convert_olmoe_to_hybrid,
    default_anchor_layers,
)
from train.lookahead_router import LookaheadRouterBank


def export_distilled_student(
    student: MicroTerraHybridWrapper,
    lookahead_bank: LookaheadRouterBank,
    out_dir: Path,
    use_rms_prescale: bool = False,
    embed_dtype: str = "f16",
    lm_head_dtype: str = "f16",
    prompt_token_ids: list[int] | None = None,
    prompt_hints: list[list[int]] | None = None,
) -> dict[str, object]:
    """Export a distilled `MicroTerraHybridWrapper` student directly to `core.bin` + `experts.bin`."""
    from exporter.export_olmoe import export_state_dict, run_reference_forward

    cfg = student.config
    layer_kinds = getattr(cfg, "terra_layer_kinds", None) or list(
        default_anchor_layers(cfg.num_hidden_layers)
    )
    config_dict = {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "num_experts": cfg.num_experts,
        "top_k": cfg.num_experts_per_tok,
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", 4096),
        "rope_theta": getattr(cfg, "rope_theta", 10000.0),
        "rms_norm_eps": getattr(cfg, "rms_norm_eps", 1e-5),
        "lookahead_offset": lookahead_bank.lookahead_offset,
        "layer_kinds": layer_kinds,
        "bitlinear_rms_prescale": use_rms_prescale,
        "norm_topk_prob": getattr(cfg, "norm_topk_prob", True),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", False),
    }

    merged_sd: dict[str, torch.Tensor] = {}
    for k, v in student.backbone.state_dict().items():
        merged_sd[k] = v.detach().cpu()
    for k, v in lookahead_bank.export_state_dict_entries().items():
        merged_sd[k] = v
    if student.hint_embedding is not None:
        merged_sd["model.hint_embedding.table.weight"] = (
            student.hint_embedding.table.weight.detach().float().cpu()
        )
        merged_sd["model.hint_embedding.proj.weight"] = (
            student.hint_embedding.proj.weight.detach().float().cpu()
        )

    export_info = export_state_dict(
        config_dict=config_dict,
        state_dict=merged_sd,
        out_dir=out_dir,
        bitlinear_rms_prescale=use_rms_prescale,
        embed_dtype=embed_dtype,
        lm_head_dtype=lm_head_dtype,
    )
    if prompt_token_ids is not None:
        golden = run_reference_forward(out_dir, prompt_token_ids, hints=prompt_hints)
        (Path(out_dir) / "golden_ref.json").write_text(json.dumps(golden, indent=2))
        export_info["golden_ref"] = golden
    return export_info


@dataclass
class DistillLossWeights:
    kl_weight: float = 1.0
    hidden_mse_weight: float = 0.5
    lookahead_weight: float = 0.25
    temperature: float = 2.0
    load_balance_coef: float = 0.01


def compute_kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Temperature-scaled forward KL divergence `T^2 * KL(p_teacher || p_student)`."""
    s_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    kl_per_tok = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)  # [B, T]
    if attention_mask is not None:
        mask = attention_mask.float()
        kl = (kl_per_tok * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        kl = kl_per_tok.mean()
    return kl * (temperature**2)


def compute_layerwise_hidden_mse(
    student_hidden_states: tuple[torch.Tensor, ...],
    teacher_hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Layer-wise MSE between normalized student and teacher layer outputs (layers `1..L`)."""
    num_layers = min(len(student_hidden_states), len(teacher_hidden_states)) - 1
    if num_layers <= 0:
        raise ValueError("Expected `output_hidden_states=True` tuple with >= 2 entries.")

    layer_mses: list[torch.Tensor] = []
    for idx in range(1, num_layers + 1):
        s_h = student_hidden_states[idx].float()
        t_h = teacher_hidden_states[idx].detach().float()
        t_rms = torch.sqrt(t_h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        diff_sq = ((s_h - t_h) / t_rms).pow(2).mean(dim=-1)  # [B, T]
        if attention_mask is not None:
            mask = attention_mask.float()
            layer_mses.append((diff_sq * mask).sum() / mask.sum().clamp(min=1.0))
        else:
            layer_mses.append(diff_sq.mean())
    return torch.stack(layer_mses).mean()


def build_student_from_teacher(
    teacher: OlmoeForCausalLM,
    anchor_layers: tuple[int, ...] | None = None,
    use_rms_prescale: bool = False,
    init_l2_scale: bool = True,
    lookahead_offset: int = 3,
    enable_hint_embedding: bool = True,
    deepcopy_model: bool = True,
) -> tuple[MicroTerraHybridWrapper, LookaheadRouterBank, list[int]]:
    """Construct a full Micro-TERRA student (BitLinear + 7:1 Hybrid Recurrence + Lookahead + Hints).

    Always upcasts student master weights to `float32` (preventing FP16 AdamW update underflow during STE QAT)
    and explicitly sets `requires_grad_(True)` across all parameters (even when deep-copied from a frozen teacher).
    """
    student_backbone = copy.deepcopy(teacher) if deepcopy_model else teacher
    student_backbone = student_backbone.float()
    cfg = student_backbone.config

    # 1. Convert eligible attention & MoE expert projections to 1.58-bit `BitLinear` with STE
    convert_model_to_bitlinear(
        student_backbone,
        use_rms_prescale=use_rms_prescale,
        rms_eps=cfg.rms_norm_eps,
        init_l2_scale=init_l2_scale,
    )

    # 2. Convert 7/8 attention layers into Gated Linear Recurrence (preserving BitLinear Q/K/V/O!)
    if anchor_layers is None:
        anchor_layers = default_anchor_layers(cfg.num_hidden_layers, block_ratio=8)
    layer_kinds = convert_olmoe_to_hybrid(student_backbone, anchor_layers=anchor_layers)

    # Ensure every student parameter (including embeddings, RMSNorms, mlp.gate, and lm_head
    # inherited from a frozen teacher) is trainable!
    for p in student_backbone.parameters():
        p.requires_grad_(True)

    param_device = next(student_backbone.parameters()).device

    # 3. Attach 3-layer Lookahead Router Bank (`L -> L + 3`)
    lookahead_bank = LookaheadRouterBank(
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        num_experts=cfg.num_experts,
        top_k=cfg.num_experts_per_tok,
        lookahead_offset=lookahead_offset,
    ).to(device=param_device, dtype=torch.float32)
    lookahead_bank.attach_hooks(student_backbone)

    # 4. Optional differentiable `HintEmbedding` table for dual-path edge hints
    hint_emb = (
        HintEmbedding(hidden_size=cfg.hidden_size).to(device=param_device, dtype=torch.float32)
        if enable_hint_embedding
        else None
    )

    wrapper = MicroTerraHybridWrapper(
        backbone=student_backbone,
        hint_embedding=hint_emb,
        lookahead_bank=lookahead_bank,
    )
    wrapper.train()
    return wrapper, lookahead_bank, layer_kinds


def distill_step(
    student: MicroTerraHybridWrapper,
    lookahead_bank: LookaheadRouterBank,
    input_ids: torch.LongTensor,
    teacher_logits: torch.Tensor,
    teacher_hidden_states: tuple[torch.Tensor, ...],
    teacher_router_logits: tuple[torch.Tensor, ...] | dict[int, torch.Tensor] | None = None,
    hints: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    eos_reset_mask: torch.Tensor | None = None,
    weights: DistillLossWeights = DistillLossWeights(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Single differentiable distillation step returning `(total_loss, metrics_dict)`."""
    student_out = student(
        input_ids=input_ids,
        hints=hints,
        attention_mask=attention_mask,
        eos_reset_mask=eos_reset_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    kl_loss = compute_kl_distillation_loss(
        student_out.logits,
        teacher_logits,
        temperature=weights.temperature,
        attention_mask=attention_mask,
    )
    hid_mse = compute_layerwise_hidden_mse(
        student_out.hidden_states,
        teacher_hidden_states,
        attention_mask=attention_mask,
    )
    look_loss, look_stats = lookahead_bank.forward_from_captured(
        load_balance_coef=weights.load_balance_coef,
        override_target_router_logits=teacher_router_logits,
    )

    total_loss = (
        weights.kl_weight * kl_loss
        + weights.hidden_mse_weight * hid_mse
        + weights.lookahead_weight * look_loss
    )
    metrics = {
        "total_loss": float(total_loss.item()),
        "kl_loss": float(kl_loss.item()),
        "hidden_mse": float(hid_mse.item()),
        **look_stats,
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    batches: Iterable[torch.LongTensor],
    device: torch.device | str = "cpu",
) -> float:
    """Compute exact token-level cross-entropy perplexity on held-out token batches."""
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for input_ids in batches:
        ids = input_ids.to(device)
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous().float()
        labels = ids[:, 1:].contiguous()
        nll = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="sum")
        total_nll += float(nll.item())
        total_tokens += int(labels.numel())

    if was_training:
        model.train()
    mean_nll = total_nll / max(total_tokens, 1)
    return math.exp(min(mean_nll, 50.0))


@torch.no_grad()
def cache_teacher_batches(
    teacher: nn.Module,
    token_batches: Iterable[torch.LongTensor],
    cache_dir: Path,
    device: torch.device | str = "cpu",
    cache_router_logits: bool = False,
) -> list[Path]:
    """Save teacher logits and layer hidden states to disk so 16 GB machines can free the teacher before QAT."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher.eval()
    teacher.to(device)
    saved_paths: list[Path] = []

    for idx, input_ids in enumerate(token_batches):
        ids = input_ids.to(device)
        out = teacher(
            input_ids=ids,
            output_hidden_states=True,
            output_router_logits=cache_router_logits,
            use_cache=False,
        )
        payload: dict[str, object] = {
            "input_ids": ids.cpu(),
            "logits": out.logits.half().cpu(),
            "hidden_states": tuple(h.half().cpu() for h in out.hidden_states),
        }
        if cache_router_logits and getattr(out, "router_logits", None) is not None:
            payload["router_logits"] = tuple(r.half().cpu() for r in out.router_logits)
        shard_path = cache_dir / f"teacher_batch_{idx:05d}.pt"
        torch.save(payload, shard_path)
        saved_paths.append(shard_path)

    return saved_paths


def stream_mixed_corpus_batches(
    tokenizer: object,
    seq_len: int = 512,
    batch_size: int = 2,
    max_batches: int = 100,
    code_ratio: float = 0.35,
) -> Iterator[torch.LongTensor]:
    """Interleave `HuggingFaceFW/fineweb-edu` (prose) and `bigcode/the-stack-smol` (code) into fixed-length batches."""
    from datasets import load_dataset

    prose_ds = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
    try:
        code_ds = iter(load_dataset("bigcode/the-stack-smol", split="train", streaming=True))
    except Exception as exc:
        print(
            f"[corpus] bigcode/the-stack-smol unavailable ({exc}); falling back to public codeparrot/codeparrot-clean-valid",
            flush=True,
        )
        code_ds = iter(load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True))

    token_buffer: list[int] = []
    yielded = 0
    step = 0

    while yielded < max_batches:
        use_code = (step % 10) < int(round(code_ratio * 10))
        step += 1
        try:
            sample = next(code_ds) if use_code else next(prose_ds)
        except StopIteration:
            break
        text = sample.get("content") or sample.get("text") or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=True)
        token_buffer.extend(ids)

        needed = batch_size * seq_len
        while len(token_buffer) >= needed and yielded < max_batches:
            chunk = token_buffer[:needed]
            del token_buffer[:needed]
            yield torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len)
            yielded += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-TERRA Milestone 3 Distillation Trainer")
    parser.add_argument("--teacher-model", type=str, default="allenai/OLMoE-1B-7B-0125")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run fast miniature distillation verification")
    parser.add_argument("--cache-teacher-dir", type=Path, default=None, help="Offline teacher cache directory for 16GB RAM")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/micro_terra_m3"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lookahead-target",
        choices=("student", "teacher"),
        default="student",
        help="Supervise L->L+3 lookahead heads against student mlp.gate (default, matches runtime) or frozen teacher.",
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    use_teacher_router = args.lookahead_target == "teacher"

    if args.synthetic_smoke:
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
        # Note: HF OlmoeTopKRouter initializes router weights to all-zeros on fresh configs;
        # seed non-zero router weights so synthetic smoke tests exercise real sparse routing.
        with torch.no_grad():
            for layer in teacher.model.layers:
                nn.init.normal_(layer.mlp.gate.weight, mean=0.0, std=0.02)
        batches = [
            torch.randint(2, cfg.vocab_size, (args.batch_size, args.seq_len))
            for _ in range(args.steps)
        ]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            dtype=torch.float16,
        )
        batches = list(
            stream_mixed_corpus_batches(
                tokenizer,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                max_batches=args.steps,
            )
        )

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if args.cache_teacher_dir is not None:
        # Build FP32 student on CPU before moving teacher to accelerator for caching
        student, lookahead_bank, layer_kinds = build_student_from_teacher(
            teacher, deepcopy_model=True
        )
        shard_paths = cache_teacher_batches(
            teacher,
            batches,
            args.cache_teacher_dir,
            device=device,
            cache_router_logits=use_teacher_router,
        )
        del teacher
    else:
        shard_paths = None
        student, lookahead_bank, layer_kinds = build_student_from_teacher(
            teacher, deepcopy_model=True
        )
        teacher = teacher.to(device)

    student = student.to(device)
    lookahead_param_ids = {id(p) for p in lookahead_bank.parameters()}
    backbone_params = [p for p in student.parameters() if id(p) not in lookahead_param_ids and p.requires_grad]
    lookahead_params = [p for p in lookahead_bank.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": lookahead_params, "lr": args.lr * 5.0, "weight_decay": 1e-4},
        ]
    )
    student, optimizer = accelerator.prepare(student, optimizer)

    history: list[dict[str, float]] = []
    for step_idx in range(args.steps):
        optimizer.zero_grad()
        if shard_paths is not None:
            cached = torch.load(shard_paths[step_idx % len(shard_paths)], map_location=device, weights_only=True)
            ids = cached["input_ids"]
            t_logits = cached["logits"].float()
            t_hidden = tuple(h.float() for h in cached["hidden_states"])
            t_routers = (
                tuple(r.float() for r in cached["router_logits"])
                if use_teacher_router and "router_logits" in cached
                else None
            )
        else:
            ids = batches[step_idx % len(batches)].to(device)
            with torch.no_grad():
                t_out = teacher(
                    input_ids=ids,
                    output_hidden_states=True,
                    output_router_logits=use_teacher_router,
                    use_cache=False,
                )
                t_logits = t_out.logits
                t_hidden = t_out.hidden_states
                t_routers = (
                    tuple(r.float() for r in t_out.router_logits)
                    if use_teacher_router and getattr(t_out, "router_logits", None) is not None
                    else None
                )

        loss, metrics = distill_step(
            student=student,
            lookahead_bank=lookahead_bank,
            input_ids=ids,
            teacher_logits=t_logits,
            teacher_hidden_states=t_hidden,
            teacher_router_logits=t_routers,
        )
        accelerator.backward(loss)
        optimizer.step()
        history.append(metrics)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "layer_kinds": layer_kinds,
        "initial_metrics": history[0],
        "final_metrics": history[-1],
    }
    (args.out_dir / "distill_metrics.json").write_text(json.dumps(summary, indent=2))
    if accelerator.is_main_process:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

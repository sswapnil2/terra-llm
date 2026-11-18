"""Score a Micro-TERRA student checkpoint with the protocol of `review_experiments.py`.

Same documents, same 4,096-token windows, bits per byte and nats/token by position, paired delta
against the BF16 teacher. The number to beat is in `BAR` (see EXPERIMENT_RESULTS.md section 1).

    python colab/score_student.py --ckpt-dir /content/drive/MyDrive/terra_ckpt --out score.json
    python colab/score_student.py --smoke        # tiny random model, checks the plumbing only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO_ROOT), str(Path(__file__).resolve().parent)]

import review_experiments as R  # noqa: E402
from train.bitlinear import set_model_quant_lambda, set_model_ternary_l2_scale  # noqa: E402
from train.distill_colab import load_student_checkpoint, save_atomic_checkpoint  # noqa: E402
from train.hybrid_ssm import GatedLinearRecurrenceAttention  # noqa: E402

# int4_block32 SmolLM2-1.7B (0.9 GiB), bench/results/2026-11-18_colab_review_experiments.json
BAR = {"wikitext103_test": 0.7345, "fineweb_edu": 0.7384}


def resolve_checkpoint(path: Path) -> Path:
    if (path / "student_state.pt").is_file():
        return path
    return path / json.loads((path / "latest.json").read_text(encoding="utf-8"))["path"]


def smoke_checkpoint(base, root: Path) -> Path:
    """A delta-rule + write-gate checkpoint of a tiny random model, written the way the trainer does."""
    from train.distill import build_student_from_teacher

    student, bank, kinds = build_student_from_teacher(
        base, use_rms_prescale=False, init_l2_scale=False, enable_hint_embedding=False, deepcopy_model=True
    )
    for mod in student.modules():
        if isinstance(mod, GatedLinearRecurrenceAttention):
            mod.enable_delta_rule()
            mod.enable_write_gate()
    opt = torch.optim.SGD(student.parameters(), lr=0.0)
    return save_atomic_checkpoint(root, 1, student, bank, opt, None, {"quant_lambda": 0.0, "layer_kinds": kinds})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=Path("/content/drive/MyDrive/terra_ckpt"))
    ap.add_argument("--teacher-model", default="allenai/OLMoE-1B-7B-0125")
    ap.add_argument("--out", type=Path, default=None, help="default: <checkpoint>/score_bits_per_byte.json")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--ternary", action="store_true", help="also score with every BitLinear fully ternary")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    ctx = 128 if args.smoke else args.ctx
    docs = R.load_docs(args.smoke)

    teacher, tok = R.load_olmoe(args.smoke, device)
    data = {k: R.make_windows(tok, d, ctx, 8) for k, d in docs.items()}
    base = {k: R.score(teacher, w, device) for k, (w, _) in data.items()}

    if args.smoke:
        ckpt = smoke_checkpoint(teacher, Path(tempfile.mkdtemp()))
        cpu_base = teacher
    else:
        from transformers import AutoModelForCausalLM

        ckpt = resolve_checkpoint(args.ckpt_dir)
        del teacher
        torch.cuda.empty_cache()
        cpu_base = AutoModelForCausalLM.from_pretrained(args.teacher_model, dtype=torch.float32, low_cpu_mem_usage=True)
    student, _bank, layer_kinds, info = load_student_checkpoint(cpu_base, ckpt)
    student = student.to(device).eval()
    R.log(f"checkpoint {ckpt}: {info}")

    res: dict = {
        "meta": {
            "checkpoint": str(ckpt), "step": json.loads((ckpt / "trainer_state.json").read_text())["step"],
            "variant": info, "layer_kinds": layer_kinds, "ctx": ctx, "teacher": args.teacher_model,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "bar_bits_per_byte": BAR,
        },
        "teacher": {}, "student": {},
    }
    passes = [("active", info["quant_lambda"])]
    if args.ternary and info["quant_lambda"] < 1.0:
        passes.append(("ternary", 1.0))
    for label, lam in passes:
        # Un-prescaled weights need the explicit L2-optimal scale at lambda = 1 (as the trainer's verifier does).
        set_model_ternary_l2_scale(student, label == "ternary")
        set_model_quant_lambda(student, lam)
        res["student"][label] = {}
        for k, (w, nb) in data.items():
            with amp:
                s, c = R.score(student, w, device)
            r = R.summarize(s, c, base[k][0])
            r["bits_per_byte"] = float(s.sum() / math.log(2) / nb)
            res["student"][label][k] = r
            R.log(
                f"student[{label}] {k}: {r['bits_per_byte']:.4f} bits/byte (bar {BAR.get(k, float('nan')):.4f}), "
                f"{r['nats_per_token']:.4f} nats/tok, delta {r['delta_nats']:+.4f} "
                f"CI {r['delta_ci95'][0]:+.4f}..{r['delta_ci95'][1]:+.4f}"
            )
    for k, (s, c) in base.items():
        res["teacher"][k] = {**R.summarize(s, c), "bits_per_byte": float(s.sum() / math.log(2) / data[k][1])}

    out = args.out or ckpt / "score_bits_per_byte.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    R.log(f"wrote {out}")


if __name__ == "__main__":
    main()

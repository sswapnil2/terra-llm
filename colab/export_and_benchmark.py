"""Self-bootstrapping one-shot final export, evaluation & benchmark runner (`colab/export_and_benchmark.py`).

In a single command on any Colab VM:
  1. Bootstraps missing Python packages (`human-eval`, `datasets`, `onnxruntime`, etc.) and Rust `--release`
     binaries (`edge_codec`, `runtime/target/release/terra-cli`) if not yet built.
  2. Loads the newest checkpoint (`step_002250`) from `/content/drive/MyDrive/terra_ckpt` and exports
     distilled `core.bin` + 4KB-aligned `experts.bin` + `golden_ref.json` to `/tmp/terra_olmoe`.
  3. Evaluates Held-Out Lookahead Top-8 Recall (`12` packed batches), WikiText-2 Perplexity (`24 x 512` windows),
     HumanEval `pass@1` (`20` problems, teacher vs student with BPE-aligned AST hints), and 16k NIAH recall.
  4. Runs `terra-cli --verify-golden` and `bench/run_benchmarks.py` (`--ram-budget-mb 2048` + 16k stream check).
  5. Archives `bench_results.json`, `eval_gates.json`, `golden_ref.json`, and `core.bin` to
     `/content/drive/MyDrive/terra_ckpt/final_release/` and prints both complete JSON summaries.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _bootstrap_deps_and_rust(repo_root: Path) -> Path:
    """Ensure all Python dependencies and Rust `--release` binaries are installed/built in one pass."""
    required_mods = ("human_eval", "datasets", "transformers", "safetensors", "accelerate", "onnxruntime", "psutil")
    if any(importlib.util.find_spec(m) is None for m in required_mods):
        print("[final 1/7] Installing Python dependencies...", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--upgrade",
                "bitsandbytes>=0.43.0",
                "transformers>=4.46.0",
                "datasets>=2.20.0",
                "accelerate>=0.34.0",
                "safetensors>=0.4.5",
                "onnxruntime>=1.19.0",
                "psutil>=6.0.0",
                "human-eval",
            ],
            check=True,
        )

    cargo_bin = Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"
    if shutil.which("cargo", path=os.environ["PATH"]) is None:
        print("[final 1/7] Installing Rust toolchain via rustup...", flush=True)
        subprocess.run(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
            shell=True,
            env=os.environ,
            check=True,
        )

    terra_cli = repo_root / "runtime" / "target" / "release" / "terra-cli"
    edge_release = repo_root / "edge_codec" / "target" / "release"
    has_edge_lib = any(
        (edge_release / name).is_file()
        for name in ("libedge_codec.so", "libedge_codec.dylib", "edge_codec.dll")
    )
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-D warnings"
    if not has_edge_lib:
        print("[final 1/7] Building edge_codec --release...", flush=True)
        subprocess.run(["cargo", "build", "--release"], cwd=str(repo_root / "edge_codec"), env=env, check=True)
    if not terra_cli.is_file():
        print("[final 1/7] Building runtime (terra-cli) --release...", flush=True)
        subprocess.run(["cargo", "build", "--release"], cwd=str(repo_root / "runtime"), env=env, check=True)
    return terra_cli


def _find_latest_checkpoint(ckpt_root: Path) -> Path:
    candidates: list[Path] = []
    latest_file = ckpt_root / "latest.json"
    if latest_file.is_file():
        meta = json.loads(latest_file.read_text(encoding="utf-8"))
        cand = ckpt_root / meta["path"]
        if (cand / "student_state.pt").is_file() and (cand / "lookahead_state.pt").is_file():
            candidates.append(cand)
    for p in sorted(ckpt_root.iterdir(), key=lambda x: x.name, reverse=True):
        if (
            p.is_dir()
            and p.name.startswith("step_")
            and (p / "student_state.pt").is_file()
            and (p / "lookahead_state.pt").is_file()
            and p not in candidates
        ):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No valid checkpoint directory found in {ckpt_root}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot export, evaluation & benchmark runner")
    parser.add_argument("--teacher-model", type=str, default="allenai/OLMoE-1B-7B-0125")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/content/drive/MyDrive/terra_ckpt"))
    parser.add_argument("--export-dir", type=Path, default=Path("/tmp/terra_olmoe"))
    parser.add_argument("--ram-budget-mb", type=int, default=2048)
    parser.add_argument("--humaneval-problems", type=int, default=20)
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)
    terra_cli = _bootstrap_deps_and_rust(_REPO_ROOT)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from colab.run_eval_only import _eval_humaneval_pass1
    from train.distill import evaluate_perplexity, export_distilled_student
    from train.distill_colab import (
            collect_packed_eval_batches,
        detect_gpu_profile,
        evaluate_lookahead_recall,
        load_frozen_teacher,
    load_student_checkpoint,
    )

    args.export_dir.mkdir(parents=True, exist_ok=True)
    release_dir = args.ckpt_dir / "final_release"
    release_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = _find_latest_checkpoint(args.ckpt_dir)
    trainer_meta = json.loads((ckpt_path / "trainer_state.json").read_text(encoding="utf-8"))
    ckpt_step = int(trainer_meta["step"])
    print(f"[final 2/7] Loaded checkpoint: {ckpt_path} (step={ckpt_step})", flush=True)

    gpu_profile = detect_gpu_profile()
    device = torch.device(gpu_profile["backend"])
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

    print("[final 2/7] Reconstructing student architecture & loading weights...", flush=True)
    cpu_base = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    student, lookahead_bank, layer_kinds, ckpt_info = load_student_checkpoint(cpu_base, ckpt_path)
    del cpu_base
    gc.collect()
    if ckpt_info["delta_rule"] or ckpt_info["write_gate"]:
        # The exporter and Rust `state.rs` implement the additive write rule only; exporting would
        # silently drop erase_proj / write_proj and ship a different model.
        raise SystemExit(f"Cannot export this checkpoint, the runtime has no delta rule / write gate: {ckpt_info}")

    print("[final 2/7] Exporting distilled 1.58-bit core.bin + 4KB-aligned experts.bin + golden_ref.json...", flush=True)
    export_info = export_distilled_student(
        student=student,
        lookahead_bank=lookahead_bank,
        out_dir=args.export_dir,
        use_rms_prescale=False,
        embed_dtype="f16",
        lm_head_dtype="f16",
        prompt_token_ids=[1, 42, 17, 99],
        prompt_hints=[[1, 5, -2, 0], [0, 9, 3, 2], [2, 18, -1, 3], [1, 5, -3, 0]],
    )
    print(
        f"[final 2/7] Exported shards to {args.export_dir}: "
        f"core={export_info.get('core_bytes')} B, experts={export_info.get('experts_bytes')} B",
        flush=True,
    )

    print("[final 3/7] Evaluating Held-Out Lookahead Top-8 Recall (12 packed batches)...", flush=True)
    student = student.to(device=device)
    lookahead_bank = lookahead_bank.to(device=device)
    student.eval()
    teacher = load_frozen_teacher(args.teacher_model, gpu_profile)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    val_batches = collect_packed_eval_batches(tokenizer, num_batches=12, seq_len=512, batch_size=2)
    with amp_ctx:
        lookahead_recall = evaluate_lookahead_recall(student, lookahead_bank, val_batches, device)
    print(f"[final 3/7] lookahead_top8_recall={lookahead_recall:.4f}", flush=True)

    print("[final 4/7] Evaluating WikiText-2 Perplexity (24 x 512 test windows)...", flush=True)
    wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wt2_text = "\n\n".join(x["text"] for x in wt2 if x["text"].strip())
    wt2_ids = tokenizer.encode(wt2_text, add_special_tokens=False)[: 24 * 512]
    wt2_batches = [
        torch.tensor(wt2_ids[i : i + 512], dtype=torch.long).unsqueeze(0)
        for i in range(0, len(wt2_ids) - 512 + 1, 512)
    ]
    with amp_ctx:
        teacher_ppl = evaluate_perplexity(teacher, wt2_batches, device=device)
        student_ppl = evaluate_perplexity(student, wt2_batches, device=device)
    ppl_retention_pct = 100.0 * (teacher_ppl / max(student_ppl, 1e-6))
    print(
        f"[final 4/7] teacher_wikitext2_ppl={teacher_ppl:.3f}, "
        f"student_wikitext2_ppl={student_ppl:.3f} (retention={ppl_retention_pct:.2f}%)",
        flush=True,
    )

    print(f"[final 5/7] Evaluating HumanEval pass@1 ({args.humaneval_problems} problems)...", flush=True)
    t_pass1, t_passed, n_probs = _eval_humaneval_pass1(
        teacher,
        tokenizer,
        device,
        is_student=False,
        amp_ctx=amp_ctx,
        max_problems=args.humaneval_problems,
        max_new_tokens=64,
    )
    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    s_pass1, s_passed, _ = _eval_humaneval_pass1(
        student,
        tokenizer,
        device,
        is_student=True,
        amp_ctx=amp_ctx,
        max_problems=args.humaneval_problems,
        max_new_tokens=64,
    )
    humaneval_retention_pct = 100.0 * (s_pass1 / max(t_pass1, 1e-6)) if t_pass1 > 0.0 else 100.0

    print("[final 6/7] Running 16,000-Token Needle-in-a-Haystack Passkey Probe...", flush=True)
    passkey = "7391042"
    filler_ids = tokenizer.encode(
        "The coastal fog drifted past the granite lighthouse at dawn. ", add_special_tokens=False
    )
    needle_ids = tokenizer.encode(
        f"Important archival note: the secret launch passkey is {passkey}. ", add_special_tokens=False
    )
    query_ids = tokenizer.encode(
        "Question: What is the secret launch passkey? Answer: The secret launch passkey is",
        add_special_tokens=False,
    )
    pre_len = 4096
    post_len = 16000 - pre_len - len(needle_ids) - len(query_ids)
    niah_ids = (
        (filler_ids * ((pre_len // len(filler_ids)) + 1))[:pre_len]
        + needle_ids
        + (filler_ids * ((post_len // len(filler_ids)) + 1))[:post_len]
        + query_ids
    )
    inp = torch.tensor([niah_ids], dtype=torch.long, device=device)
    with torch.no_grad(), amp_ctx:
        generated = []
        for _ in range(16):
            out = student(input_ids=inp, use_cache=False)
            nxt = int(out.logits[0, -1].argmax().item())
            generated.append(nxt)
            inp = torch.cat([inp, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    niah_answer = tokenizer.decode(generated, skip_special_tokens=True)

    del student, lookahead_bank, val_batches, wt2_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[final 7/7] Running Rust terra-cli --verify-golden & Linux io_uring NVMe benchmarks...", flush=True)
    verify_proc = subprocess.run(
        [
            str(terra_cli),
            "--model-dir",
            str(args.export_dir),
            "--ram-budget-mb",
            str(args.ram_budget_mb),
            "--verify-golden",
            "--golden-tol",
            "5e-2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    verify_out = verify_proc.stdout.strip()
    print(f"[final 7/7] {verify_out}", flush=True)

    eval_summary = {
        "checkpoint_dir": str(ckpt_path),
        "checkpoint_step": ckpt_step,
        "verify_golden": verify_out,
        "layer_kinds": layer_kinds,
        "lookahead_top8_recall": round(lookahead_recall, 4),
        "teacher_wikitext2_ppl": round(teacher_ppl, 3),
        "student_wikitext2_ppl": round(student_ppl, 3),
        "wikitext2_ppl_retention_pct": round(ppl_retention_pct, 2),
        "humaneval_problems_tested": n_probs,
        "teacher_humaneval_pass1": round(t_pass1, 4),
        "teacher_humaneval_passed": t_passed,
        "student_humaneval_pass1": round(s_pass1, 4),
        "student_humaneval_passed": s_passed,
        "humaneval_retention_pct": round(humaneval_retention_pct, 2),
        "niah_16k_passkey_expected": passkey,
        "niah_16k_generated": niah_answer,
        "niah_16k_exact_match": passkey in niah_answer,
    }
    eval_path = args.export_dir / "eval_gates.json"
    eval_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    bench_path = args.export_dir / "bench_results.json"
    subprocess.run(
        [
            sys.executable,
            "bench/run_benchmarks.py",
            "--model-dir",
            str(args.export_dir),
            "--terra-cli",
            str(terra_cli),
            "--ram-budget-mb",
            str(args.ram_budget_mb),
            "--max-seq-len",
            "16384",
            "--run-16k-check",
            "--out-json",
            str(bench_path),
        ],
        check=True,
    )

    for src in (
        bench_path,
        eval_path,
        args.export_dir / "golden_ref.json",
        args.export_dir / "core.bin",
    ):
        if src.is_file():
            dst = release_dir / src.name
            shutil.copy2(src, dst)
            print(f"[final] Archived {src.name} ({src.stat().st_size:,} B) -> {dst}", flush=True)

    print("EVAL_GATES_JSON_START", flush=True)
    print(eval_path.read_text(encoding="utf-8"), flush=True)
    print("EVAL_GATES_JSON_END", flush=True)
    print("BENCH_RESULTS_JSON_START", flush=True)
    print(bench_path.read_text(encoding="utf-8"), flush=True)
    print("BENCH_RESULTS_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

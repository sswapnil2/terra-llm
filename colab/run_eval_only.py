"""Post-training Colab evaluation runner (`colab/run_eval_only.py`).

Runs all remaining Milestone 3B & Milestone 5 evaluation gates on the distilled
`step_002500` checkpoint and exported `/content/terra_exported` binary shards:
  1. Rust `terra-cli --verify-golden` parity check on exported real shards
  2. Held-out Lookahead `L -> L+3` Top-8 Recall (`eval_lookahead_recall`)
  3. WikiText-2 Perplexity retention (`wikitext-2-raw-v1`, test split vs BF16 teacher)
  4. HumanEval `pass@1` retention on 20 problems (`human_eval`)
  5. 16,000-token Needle-in-a-Haystack passkey recall (`use_cache=False`)
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from datasets import load_dataset
from human_eval.data import read_problems
from human_eval.execution import check_correctness
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edge_codec import UnifiedEdgeCodec
from exporter.export_olmoe import run_reference_forward
from train.distill import build_student_from_teacher, evaluate_perplexity
from train.distill_colab import (
    collect_packed_eval_batches,
    detect_gpu_profile,
    evaluate_lookahead_recall,
    load_frozen_teacher,
)


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


def _resolve_export_dir(ckpt_dir: Path) -> Path:
    for cand in (Path("/content/terra_exported"), Path("/tmp/terra_olmoe"), ckpt_dir / "exported"):
        if (cand / "core.bin").is_file() and (cand / "experts.bin").is_file():
            return cand
    raise FileNotFoundError("Could not find exported `core.bin` + `experts.bin` directory.")


def _eval_humaneval_pass1(
    model_obj: torch.nn.Module,
    tokenizer: object,
    device: torch.device,
    *,
    is_student: bool,
    amp_ctx: object,
    max_problems: int = 20,
    max_new_tokens: int = 64,
) -> tuple[float, int, int]:
    codec = UnifiedEdgeCodec(max_code_tokens=2048) if is_student else None
    problems = list(read_problems().items())[:max_problems]
    passed = 0
    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop_markers = ("\ndef ", "\nclass ", "\nif __name__", "\n# ")

    for idx, (_, prob) in enumerate(problems, start=1):
        prompt = prob["prompt"]
        if getattr(tokenizer, "is_fast", False):
            enc = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            hints_t = None
            if is_student and codec is not None:
                raw_h = codec.encode_aligned_to_bpe(prompt, enc["offset_mapping"][0].tolist(), is_code=True)
                hints_t = torch.from_numpy(raw_h).unsqueeze(0).to(device)
        else:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            ids = enc["input_ids"]
            hints_t = None
            if is_student and codec is not None:
                raw_h = codec.encode_code(prompt, lang="python")
                h_pad = torch.zeros((1, ids.shape[1], 4), dtype=torch.int32, device=device)
                n_copy = min(h_pad.shape[1], int(raw_h.shape[0]))
                if n_copy > 0:
                    h_pad[0, :n_copy] = torch.from_numpy(raw_h[:n_copy]).to(device)
                hints_t = h_pad

        gen_tokens: list[int] = []
        with torch.no_grad(), amp_ctx:
            past_kv = None
            for _ in range(max_new_tokens):
                if is_student:
                    out = model_obj(input_ids=ids, hints=hints_t, use_cache=False)
                    nxt = int(out.logits[0, -1].argmax(dim=-1).item())
                    ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
                    if hints_t is not None:
                        hints_t = torch.cat(
                            [hints_t, torch.zeros((1, 1, 4), dtype=torch.int32, device=device)], dim=1
                        )
                else:
                    step_ids = ids if past_kv is None else torch.tensor([[gen_tokens[-1]]], device=device)
                    out = model_obj(input_ids=step_ids, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    nxt = int(out.logits[0, -1].argmax(dim=-1).item())

                gen_tokens.append(nxt)
                if eos_id is not None and nxt == eos_id:
                    break
                comp_so_far = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                if any(marker in comp_so_far for marker in stop_markers):
                    break

        completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        for marker in stop_markers:
            if marker in completion:
                completion = completion.split(marker, 1)[0]
        res = check_correctness(prob, completion, timeout=3.0, completion_id=idx)
        if res.get("passed"):
            passed += 1
        if idx % 5 == 0 or idx == len(problems):
            tag = "student" if is_student else "teacher"
            print(f"[humaneval:{tag}] {idx}/{len(problems)} passed={passed}", flush=True)

    return passed / max(len(problems), 1), passed, len(problems)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Micro-TERRA post-training Colab evaluation gates")
    parser.add_argument("--teacher-model", type=str, default="allenai/OLMoE-1B-7B-0125")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/content/drive/MyDrive/terra_ckpt"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--humaneval-problems", type=int, default=20)
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)
    export_dir = _resolve_export_dir(args.ckpt_dir)
    terra_cli = _REPO_ROOT / "runtime" / "target" / "release" / "terra-cli"

    print(f"[eval] 1. Running run_reference_forward + terra-cli --verify-golden on {export_dir}...", flush=True)
    golden = run_reference_forward(
        export_dir,
        [1, 42, 17, 99],
        hints=[[1, 5, -2, 0], [0, 9, 3, 2], [2, 18, -1, 3], [1, 5, -3, 0]],
    )
    golden_path = export_dir / "golden_ref.json"
    golden_path.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    verify_proc = subprocess.run(
        [
            str(terra_cli),
            "--model-dir",
            str(export_dir),
            "--ram-budget-mb",
            "2048",
            "--verify-golden",
            "--golden-tol",
            "1e-2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    verify_out = verify_proc.stdout.strip()
    print(f"[eval] {verify_out}", flush=True)

    ckpt_path = _find_latest_checkpoint(args.ckpt_dir)
    ckpt_step = int(json.loads((ckpt_path / "trainer_state.json").read_text(encoding="utf-8"))["step"])
    print(f"[eval] 2. Loading checkpoint step={ckpt_step} from {ckpt_path}...", flush=True)

    gpu_profile = detect_gpu_profile()
    device = torch.device(gpu_profile["backend"])
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

    cpu_base = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    student, lookahead_bank, layer_kinds = build_student_from_teacher(
        cpu_base,
        use_rms_prescale=False,
        init_l2_scale=False,
        enable_hint_embedding=True,
        deepcopy_model=False,
    )
    del cpu_base
    gc.collect()

    student.load_state_dict(
        torch.load(ckpt_path / "student_state.pt", map_location="cpu", weights_only=True)
    )
    lookahead_bank.load_state_dict(
        torch.load(ckpt_path / "lookahead_state.pt", map_location="cpu", weights_only=True)
    )
    student = student.to(device=device)
    lookahead_bank = lookahead_bank.to(device=device)
    student.eval()

    print("[eval] 3. Evaluating held-out Lookahead L->L+3 Top-8 Recall...", flush=True)
    val_batches = collect_packed_eval_batches(tokenizer, num_batches=12, seq_len=512, batch_size=2)
    with amp_ctx:
        lookahead_recall = evaluate_lookahead_recall(student, lookahead_bank, val_batches, device)
    print(f"[eval] lookahead_top8_recall={lookahead_recall:.4f}", flush=True)

    print("[eval] 4. Loading BF16 Teacher & evaluating WikiText-2 Perplexity...", flush=True)
    teacher = load_frozen_teacher(args.teacher_model, gpu_profile)
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
        f"[eval] teacher_ppl={teacher_ppl:.3f}, student_ppl={student_ppl:.3f}, retention={ppl_retention_pct:.2f}%",
        flush=True,
    )

    print(f"[eval] 5. Evaluating HumanEval pass@1 ({args.humaneval_problems} problems)...", flush=True)
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

    print("[eval] 6. Running 16k Needle-in-a-Haystack passkey probe (use_cache=False)...", flush=True)
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
    niah_tokens: list[int] = []
    with torch.no_grad(), amp_ctx:
        for _ in range(8):
            out = student(input_ids=inp, use_cache=False)
            nxt = int(out.logits[0, -1].argmax(dim=-1).item())
            niah_tokens.append(nxt)
            inp = torch.cat([inp, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    niah_answer = tokenizer.decode(niah_tokens, skip_special_tokens=True)

    summary = {
        "checkpoint_step": ckpt_step,
        "export_dir": str(export_dir),
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
        "niah_16k_context_len": len(niah_ids),
        "niah_16k_passkey_expected": passkey,
        "niah_16k_generated": niah_answer,
        "niah_16k_exact_match": passkey in niah_answer,
    }

    eval_out_path = args.ckpt_dir / "eval_gates.json"
    eval_out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    drive_export_dir = args.ckpt_dir / "exported"
    if drive_export_dir.is_dir():
        shutil.copy2(golden_path, drive_export_dir / "golden_ref.json")

    print("EVAL_GATES_JSON_START", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("EVAL_GATES_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

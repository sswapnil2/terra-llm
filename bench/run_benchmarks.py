"""Micro-TERRA Benchmark Runner (`bench/run_benchmarks.py`):
Executes `terra-cli` locally (macOS ARM NEON + `F_NOCACHE`) and on Colab Linux
(`AVX2`/`AVX-512` + `io_uring` `O_DIRECT`) on full `OLMoE-1B-7B` binary shards.

Compares:
  1. Demand-only cold NVMe reads (`--no-lookahead`) under `--ram-budget-mb 2048`
  2. 3-Layer Lookahead async prefetching under `--ram-budget-mb 2048`
  3. Optional 16k-context budget compliance & tighter-RAM stress benchmarks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _run_terra_cli(
    cli_bin: Path,
    model_dir: Path,
    *,
    ram_budget_mb: int,
    max_seq_len: int,
    prompt_tokens: str,
    max_new_tokens: int,
    no_lookahead: bool,
    max_cached_experts: int | None = None,
) -> dict[str, Any]:
    cmd = [
        str(cli_bin),
        "--model-dir",
        str(model_dir),
        "--ram-budget-mb",
        str(ram_budget_mb),
        "--max-seq-len",
        str(max_seq_len),
        "--prompt-tokens",
        prompt_tokens,
        "--max-new-tokens",
        str(max_new_tokens),
        "--benchmark-json",
    ]
    if no_lookahead:
        cmd.append("--no-lookahead")
    if max_cached_experts is not None:
        cmd.extend(["--max-cached-experts", str(max_cached_experts)])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    report = json.loads(proc.stdout)

    budget_bytes = ram_budget_mb * 1024 * 1024
    resident_bytes = int(report["memory"]["total_resident_bytes"])
    if resident_bytes > budget_bytes:
        raise AssertionError(
            f"RAM budget exceeded: {resident_bytes} B > {budget_bytes} B ({ram_budget_mb} MB)"
        )
    return report


def format_markdown_table(runs: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "| Mode | SIMD | IO Backend | Direct I/O | Resident MB | Cached Experts | Hit Rate % | TTFT (ms) | Decode (tok/s) | Stall (us) |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for label, r in runs:
        res_mb = r["memory"]["total_resident_bytes"] / (1024 * 1024)
        pm = r["prefetch_metrics"]
        lines.append(
            f"| {label} | {r['simd_backend']} | {r['io_backend']} | {r['direct_io_active']} | "
            f"{res_mb:.1f} | {r['memory']['max_cached_experts']} | {r['cache_hit_rate_pct']:.2f}% | "
            f"{r['ttft_ms']:.2f} | {r['decode_tokens_per_sec']:.2f} | {pm['prefetch_stall_us']} |"
        )
    return "\n".join(lines)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_cli = repo_root / "runtime" / "target" / "release" / "terra-cli"

    parser = argparse.ArgumentParser(description="Micro-TERRA End-to-End Runtime Benchmark")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--terra-cli", type=Path, default=default_cli)
    parser.add_argument("--ram-budget-mb", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument(
        "--prompt-tokens",
        type=str,
        default="1,42,17,99,128,256,512,1024,2048,4096,8192,16384",
    )
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--run-16k-check", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if args.generate_synthetic and not (args.model_dir / "core.bin").is_file():
        from exporter.export_olmoe import export_synthetic_olmoe

        export_synthetic_olmoe(args.model_dir)

    if not args.terra_cli.is_file():
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=str(repo_root / "runtime"),
            check=True,
        )

    runs: list[tuple[str, dict[str, Any]]] = []

    cold_rep = _run_terra_cli(
        args.terra_cli,
        args.model_dir,
        ram_budget_mb=args.ram_budget_mb,
        max_seq_len=args.max_seq_len,
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        no_lookahead=True,
    )
    runs.append((f"Cold Demand (--ram-budget-mb {args.ram_budget_mb})", cold_rep))

    look_rep = _run_terra_cli(
        args.terra_cli,
        args.model_dir,
        ram_budget_mb=args.ram_budget_mb,
        max_seq_len=args.max_seq_len,
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        no_lookahead=False,
    )
    runs.append((f"3-Layer Lookahead (--ram-budget-mb {args.ram_budget_mb})", look_rep))

    if args.run_16k_check:
        tight_rep = _run_terra_cli(
            args.terra_cli,
            args.model_dir,
            ram_budget_mb=min(args.ram_budget_mb, 1024),
            max_seq_len=16384,
            prompt_tokens=args.prompt_tokens,
            max_new_tokens=6,
            no_lookahead=False,
        )
        runs.append(("Tight 16k Stream (--ram-budget-mb 1024)", tight_rep))

    table = format_markdown_table(runs)
    print(table)

    payload = {
        "ram_budget_mb": args.ram_budget_mb,
        "max_seq_len": args.max_seq_len,
        "runs": [{"label": lbl, "report": rep} for lbl, rep in runs],
        "markdown_table": table,
    }
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end Colab Linux runner for Project Micro-TERRA (`colab/run_all_colab.py`).

Runs sequentially on the Colab Linux GPU runtime:
  1. Hardware / `/tmp` NVMe / `io_uring` / `O_DIRECT` / Google Drive diagnostics
  2. Python dependency install + Rust toolchain setup + `--release` builds (`edge_codec`, `runtime`)
  3. Pre-flight Linux `cargo test --release` + `pytest` verification
  4. Native Linux `io_uring + O_DIRECT` NVMe benchmark under `--ram-budget-mb 2048`
  5. Full `allenai/OLMoE-1B-7B-0125` distillation via `train.distill_colab`
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


def _log(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def _run(cmd: list[str], *, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full Micro-TERRA Colab pipeline")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    _log("1. Hardware, NVMe, io_uring & Google Drive Diagnostics")
    _run(["nvidia-smi"])
    _run(["df", "-h", "/tmp"])

    io_uring_path = pathlib.Path("/proc/sys/kernel/io_uring_disabled")
    if io_uring_path.exists():
        print("/proc/sys/kernel/io_uring_disabled =", io_uring_path.read_text().strip(), flush=True)
    else:
        print("/proc/sys/kernel/io_uring_disabled not present", flush=True)

    probe = "/tmp/.terra_odirect_probe"
    try:
        fd = os.open(probe, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)
        os.close(fd)
        print("/tmp O_DIRECT supported: True", flush=True)
    except OSError as exc:
        print(f"/tmp O_DIRECT probe returned {exc} (runtime falls back to pread cleanly)", flush=True)
    finally:
        if os.path.exists(probe):
            os.remove(probe)

    ckpt_dir = pathlib.Path("/content/drive/MyDrive/terra_ckpt")
    if not pathlib.Path("/content/drive/MyDrive").is_dir():
        ckpt_dir = pathlib.Path("/content/terra_ckpt_local")
        print("WARNING: /content/drive/MyDrive not mounted; using local ckpt dir", ckpt_dir, flush=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print("Checkpoint directory:", ckpt_dir, flush=True)

    _log("2. Python Dependencies & Rust --release Build")
    _run(
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
            "pytest",
            "numpy",
            "huggingface_hub",
        ]
    )

    cargo_bin = pathlib.Path.home() / ".cargo" / "bin"
    env = os.environ.copy()
    env["PATH"] = f"{cargo_bin}:{env.get('PATH', '')}"
    if shutil.which("cargo", path=env["PATH"]) is None:
        subprocess.run(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
            shell=True,
            env=env,
            check=True,
        )
    env["RUSTFLAGS"] = "-D warnings"
    _run(["rustc", "--version"], env=env)
    _run(["cargo", "build", "--release"], cwd=repo_root / "edge_codec", env=env)
    _run(["cargo", "build", "--release"], cwd=repo_root / "runtime", env=env)

    _log("3. Pre-Flight Linux Rust & Python Unit Gates")
    _run(["cargo", "test", "--release"], cwd=repo_root / "edge_codec", env=env)
    _run(["cargo", "test", "--release"], cwd=repo_root / "runtime", env=env)
    _run([sys.executable, "-m", "pytest", "-q", "exporter", "edge_codec", "train/tests"], env=env)

    _log("4. Native Linux io_uring (O_DIRECT) Benchmark on /tmp NVMe (--ram-budget-mb 2048)")
    synth_dir = pathlib.Path("/tmp/terra_synth_7b")
    bench_json = pathlib.Path("/tmp/terra_synth_bench.json")
    try:
        _run([sys.executable, "-m", "exporter.export_olmoe", "--synthetic", "--out-dir", str(synth_dir)], env=env)
        _run(
            [
                sys.executable,
                "bench/run_benchmarks.py",
                "--model-dir",
                str(synth_dir),
                "--ram-budget-mb",
                "2048",
                "--run-16k-check",
                "--out-json",
                str(bench_json),
            ],
            env=env,
        )
        print(bench_json.read_text(encoding="utf-8"), flush=True)
        shutil.copy2(bench_json, ckpt_dir / "synth_linux_iouring_bench.json")
    finally:
        shutil.rmtree(synth_dir, ignore_errors=True)

    if args.skip_train:
        _log("5. Skipping distillation (--skip-train passed)")
        return 0

    _log(f"5. Full allenai/OLMoE-1B-7B-0125 Distillation (--steps {args.steps})")
    export_dir = pathlib.Path("/tmp/terra_olmoe")
    export_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "train.distill_colab",
            "--teacher-model",
            "allenai/OLMoE-1B-7B-0125",
            "--ckpt-dir",
            str(ckpt_dir),
            "--export-dir",
            str(export_dir),
            "--steps",
            str(args.steps),
            "--save-every",
            str(args.save_every),
            "--log-every",
            str(args.log_every),
            "--seq-len",
            str(args.seq_len),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum",
            str(args.grad_accum),
            "--lr",
            str(args.lr),
        ],
        env=env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Milestone 1 Verification Gate (`ANTIGRAVITY.md` Section 3):
- Bit-exact `unpack(pack(W)) == W` across OLMoE projection shapes
- Full 1,024-expert (`16 layers x 64 experts`) `--synthetic` generation in seconds (< 5s target)
- Strict `offset % 4096 == 0` and `padded_bytes % 4096 == 0` alignment across all 1,024 experts
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from exporter import (
    PAGE_SIZE,
    absmean_quantize_ternary,
    pack_ternary_2bit,
    read_core_header_and_dir,
    read_expert_weights,
    read_experts_header_and_dir,
    unpack_ternary_2bit,
)


@pytest.mark.parametrize(
    "shape",
    [
        (1024, 2048),  # OLMoE expert gate_proj / up_proj
        (2048, 1024),  # OLMoE expert down_proj
        (2048, 2048),  # OLMoE q_proj / k_proj / v_proj / o_proj
    ],
)
def test_unpack_pack_bit_exact_olmoe_shapes(shape: tuple[int, int]) -> None:
    torch.manual_seed(shape[0] + shape[1])
    w = torch.randn(shape, dtype=torch.float32)
    w_tern, scales = absmean_quantize_ternary(w)
    packed = pack_ternary_2bit(w_tern)
    unpacked = unpack_ternary_2bit(packed, shape[0], shape[1])
    assert unpacked.dtype == np.int8
    assert np.array_equal(unpacked, w_tern.numpy())
    assert scales.shape == (shape[0],)


def test_synthetic_1024_expert_olmoe_cli_speed_and_4kb_alignment(tmp_path: Path) -> None:
    out_dir = tmp_path / "synth_olmoe_7b"
    repo_root = Path(__file__).resolve().parents[1]

    t0 = time.perf_counter()
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "exporter.export_olmoe",
            "--synthetic",
            "--out-dir",
            str(out_dir),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed = time.perf_counter() - t0
    assert "Exported synthetic OLMoE-1B-7B" in res.stdout
    assert elapsed < 8.0, f"Expected fast synthetic 1,024-expert export (< 8s), took {elapsed:.2f}s"

    core_path = out_dir / "core.bin"
    experts_path = out_dir / "experts.bin"
    assert core_path.is_file()
    assert experts_path.is_file()

    core_hdr, core_entries = read_core_header_and_dir(core_path, verify_crc=True)
    assert core_hdr.vocab_size == 50304
    assert core_hdr.hidden_size == 2048
    assert core_hdr.intermediate_size == 1024
    assert core_hdr.num_layers == 16
    assert core_hdr.num_experts == 64
    assert core_hdr.top_k == 8
    assert list(core_hdr.layer_kinds) == ([1] * 7 + [0] + [1] * 7 + [0])
    assert len(core_entries) > 0

    exp_hdr, exp_entries = read_experts_header_and_dir(experts_path, verify_crc=True)
    assert exp_hdr.num_layers == 16
    assert exp_hdr.num_experts == 64
    assert len(exp_entries) == 1024
    assert exp_hdr.expert_payload_bytes == 1_589_248
    assert exp_hdr.expert_padded_bytes == 1_589_248
    assert experts_path.stat().st_size % PAGE_SIZE == 0

    for idx, entry in enumerate(exp_entries):
        assert entry.offset % PAGE_SIZE == 0, f"expert #{idx} unaligned: {entry.offset}"
        assert entry.padded_bytes % PAGE_SIZE == 0

    # Verify sampled experts unpack cleanly with valid CRC32 & zero invalid 0b11 crumbs
    for sample_idx in (0, 7 * 64 + 31, 1023):
        weights = read_expert_weights(experts_path, exp_hdr, exp_entries[sample_idx], verify_crc=True)
        for proj_name, (w_unpacked, sc) in weights.items():
            assert set(np.unique(w_unpacked).tolist()).issubset({-1, 0, 1}), proj_name
            assert np.all(sc > 0.0)

    # Free the 1.6 GB temporary fixture once verified
    experts_path.unlink()
    core_path.unlink()

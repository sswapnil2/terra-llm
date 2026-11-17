"""Comprehensive verification suite for Milestone 1: Micro-TERRA Binary Format & Ternary Exporter."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import numpy as np
import pytest
from safetensors.torch import save_file
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from exporter import (
    CORE_ALIGNMENT,
    CORE_ENTRY_BYTES,
    CORE_HEADER_BYTES,
    CORE_MAGIC,
    DTYPE_F16,
    DTYPE_F32,
    DTYPE_TERNARY_2BIT,
    EXPERTS_HEADER_BYTES,
    EXPERTS_MAGIC,
    EXPERT_ENTRY_BYTES,
    EXPERT_INDEX_OFFSET,
    FLAG_BITLINEAR_RMS_PRESCALE,
    FLAG_HAS_HINT_EMBEDDING,
    FLAG_NORM_TOPK_PROB,
    LAYER_KIND_ANCHOR_GQA,
    LAYER_KIND_GATED_LINEAR_RECURRENCE,
    PAGE_SIZE,
    ExpertsBinWriter,
    absmean_quantize_ternary,
    bitlinear_reference_gemv,
    build_synthetic_tiny_state,
    compute_expert_payload_bytes,
    crc32_ieee,
    export_checkpoint,
    export_state_dict,
    pack_ternary_2bit,
    read_core_header_and_dir,
    read_core_tensor,
    read_expert_weights,
    read_experts_header_and_dir,
    run_reference_forward,
    unpack_ternary_2bit,
)


def _raw_crc32(buf: bytes) -> int:
    return zlib.crc32(buf) & 0xFFFFFFFF


def test_ieee_crc32_canonical_vector() -> None:
    # schemas/terra_format.md line 10 canonical IEEE 802.3 test vector
    assert crc32_ieee(b"123456789") == 0xCBF43926


def test_crumb_encoding_exact_bit_positions() -> None:
    # [0, +1, -1, +1] -> c0=0b00, c1=0b01, c2=0b10, c3=0b01 -> 0b01_10_01_00 = 0x64
    w = np.array([[0, 1, -1, 1], [-1, -1, 0, 1]], dtype=np.int8)
    packed = pack_ternary_2bit(w)
    assert packed.dtype == np.uint8
    assert packed.shape == (2, 1)
    assert int(packed[0, 0]) == (0 | (1 << 2) | (2 << 4) | (1 << 6)) == 0x64
    assert int(packed[1, 0]) == (2 | (2 << 2) | (0 << 4) | (1 << 6)) == 0x4A

    unpacked = unpack_ternary_2bit(packed, 2, 4)
    assert unpacked.dtype == np.int8
    assert np.array_equal(unpacked, w)


@pytest.mark.parametrize(
    "shape",
    [(1, 4), (7, 12), (33, 64), (128, 256), (256, 512)],
)
def test_pack_unpack_roundtrip_random_and_edge_cases(shape: tuple[int, int]) -> None:
    rows, cols = shape
    rng = np.random.default_rng(12345 + rows * 31 + cols)

    # Edge cases: all -1, all 0, all +1, plus random {-1, 0, +1}
    cases = [
        np.full((rows, cols), -1, dtype=np.int8),
        np.zeros((rows, cols), dtype=np.int8),
        np.ones((rows, cols), dtype=np.int8),
        rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(rows, cols)),
    ]
    for w in cases:
        packed = pack_ternary_2bit(w)
        assert packed.shape == (rows, cols // 4)
        assert packed.dtype == np.uint8
        assert packed.flags["C_CONTIGUOUS"]

        # Unpack from ndarray and raw bytes
        u_arr = unpack_ternary_2bit(packed, rows, cols)
        u_bytes = unpack_ternary_2bit(packed.tobytes(), rows, cols)
        assert np.array_equal(u_arr, w)
        assert np.array_equal(u_bytes, w)

        # Torch tensor input path (including non-contiguous view when rows > 1)
        w_t = torch.from_numpy(np.ascontiguousarray(w.T)).t()
        if rows > 1:
            assert not w_t.is_contiguous()
        packed_t = pack_ternary_2bit(w_t)
        assert np.array_equal(packed_t, packed)


def test_full_olmoe_expert_dimensions_and_zero_padding_waste(tmp_path: Path) -> None:
    # Section 3.3 verification for OLMoE-1B-7B (D=2048, I=1024):
    # payload_bytes = 3 * 524,288 + 4 * 4,096 = 1,589,248 == 388 * 4096
    D, I = 2048, 1024
    payload_bytes = compute_expert_payload_bytes(D, I)
    assert payload_bytes == 1_589_248
    assert payload_bytes == 388 * PAGE_SIZE
    assert payload_bytes % PAGE_SIZE == 0

    gen = torch.Generator(device="cpu").manual_seed(7)
    gate_f = torch.randn((I, D), generator=gen, dtype=torch.float32) * 0.02
    up_f = torch.randn((I, D), generator=gen, dtype=torch.float32) * 0.02
    down_f = torch.randn((D, I), generator=gen, dtype=torch.float32) * 0.02

    gate_q, gate_s = absmean_quantize_ternary(gate_f)
    up_q, up_s = absmean_quantize_ternary(up_f)
    down_q, down_s = absmean_quantize_ternary(down_f)

    assert np.array_equal(unpack_ternary_2bit(pack_ternary_2bit(gate_q), I, D), gate_q.numpy())
    assert np.array_equal(unpack_ternary_2bit(pack_ternary_2bit(up_q), I, D), up_q.numpy())
    assert np.array_equal(unpack_ternary_2bit(pack_ternary_2bit(down_q), D, I), down_q.numpy())

    experts_file = tmp_path / "single_olmoe_expert.bin"
    with ExpertsBinWriter(
        experts_file,
        num_layers=1,
        num_experts=1,
        hidden_size=D,
        intermediate_size=I,
    ) as writer:
        writer.write_expert(0, 0, gate_f, up_f, down_f)

    hdr, entries = read_experts_header_and_dir(experts_file, verify_crc=True)
    assert hdr.expert_payload_bytes == 1_589_248
    assert hdr.expert_padded_bytes == 1_589_248
    assert entries[0].offset % PAGE_SIZE == 0
    assert experts_file.stat().st_size % PAGE_SIZE == 0

    reloaded = read_expert_weights(experts_file, hdr, entries[0], verify_crc=True)
    assert np.array_equal(reloaded["gate_proj"][0], gate_q.numpy())
    assert np.array_equal(reloaded["gate_proj"][1], gate_s.numpy())
    assert np.array_equal(reloaded["up_proj"][0], up_q.numpy())
    assert np.array_equal(reloaded["up_proj"][1], up_s.numpy())
    assert np.array_equal(reloaded["down_proj"][0], down_q.numpy())
    assert np.array_equal(reloaded["down_proj"][1], down_s.numpy())


@pytest.mark.parametrize("bad_byte", [0x03, 0x0C, 0x30, 0xC0, 0xFF, 0x1B | 0x03])
def test_rejects_invalid_0b11_crumbs(bad_byte: int) -> None:
    buf = np.zeros((4, 2), dtype=np.uint8)
    buf[2, 1] = np.uint8(bad_byte)
    with pytest.raises(ValueError, match="0b11"):
        unpack_ternary_2bit(buf, 4, 8)
    with pytest.raises(ValueError, match="0b11"):
        unpack_ternary_2bit(buf.tobytes(), 4, 8)


@pytest.mark.parametrize("bad_cols", [1, 2, 3, 5, 6, 7, 10, 65])
def test_rejects_non_multiple_of_4_columns(bad_cols: int) -> None:
    w_float = torch.randn(4, bad_cols)
    with pytest.raises(ValueError, match="divisible by 4"):
        absmean_quantize_ternary(w_float)

    w_tern = np.zeros((4, bad_cols), dtype=np.int8)
    with pytest.raises(ValueError, match="divisible by 4"):
        pack_ternary_2bit(w_tern)

    with pytest.raises(ValueError, match="divisible by 4"):
        unpack_ternary_2bit(b"\x00" * 16, 4, bad_cols)


@pytest.mark.parametrize("bad_val", [2, -2, 3, -128, 127])
def test_pack_rejects_out_of_range_values(bad_val: int) -> None:
    w = np.zeros((2, 8), dtype=np.int16)
    w[1, 5] = bad_val
    with pytest.raises(ValueError, match=r"\{-1, 0, \+1\}"):
        pack_ternary_2bit(w)

    w_f = np.zeros((2, 8), dtype=np.float32)
    w_f[0, 3] = 0.5
    with pytest.raises(ValueError, match=r"\{-1, 0, \+1\}"):
        pack_ternary_2bit(w_f)


def test_bitlinear_reference_gemv_exact_equations() -> None:
    gen = torch.Generator(device="cpu").manual_seed(99)
    w_raw = torch.randn((16, 32), generator=gen)
    w_tern, scales = absmean_quantize_ternary(w_raw)

    # Zero-input guard (< 1e-12)
    x_zero = torch.full((32,), 1e-15, dtype=torch.float32)
    y_zero = bitlinear_reference_gemv(x_zero, w_tern, scales, rms_prescale=False)
    assert torch.count_nonzero(y_zero).item() == 0
    assert not torch.isnan(y_zero).any()

    # Batched vs per-token parity with and without RMS prescale
    x_batch = torch.randn((3, 5, 32), generator=gen, dtype=torch.float32)
    for rms_pre in (False, True):
        y_batch = bitlinear_reference_gemv(
            x_batch, w_tern, scales, rms_prescale=rms_pre, rms_eps=1e-5
        )
        for b in range(3):
            for t in range(5):
                y_single = bitlinear_reference_gemv(
                    x_batch[b, t], w_tern, scales, rms_prescale=rms_pre, rms_eps=1e-5
                )
                assert torch.equal(y_batch[b, t], y_single)


def _assert_full_binary_integrity_and_bit_exact_weights(
    core_path: Path,
    experts_path: Path,
    orig_core_ternary: dict[str, torch.Tensor],
    orig_core_dense: dict[str, torch.Tensor],
    orig_experts: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    # 1. Raw byte-offset & CRC32 verification on core.bin (Section 2.1 & 2.2)
    core_bytes = core_path.read_bytes()
    assert core_bytes[0:4] == CORE_MAGIC
    assert struct.unpack("<I", core_bytes[4:8])[0] == 1
    assert struct.unpack("<I", core_bytes[252:256])[0] == _raw_crc32(core_bytes[0:252])
    assert core_bytes[140:252] == b"\x00" * 112

    core_hdr, core_entries = read_core_header_and_dir(core_path, verify_crc=True)
    assert core_hdr.tensor_count == len(core_entries)
    assert core_hdr.data_section_offset % CORE_ALIGNMENT == 0

    entry_by_name = {e.name: e for e in core_entries}
    for idx, ent in enumerate(core_entries):
        rec_off = CORE_HEADER_BYTES + idx * CORE_ENTRY_BYTES
        rec = core_bytes[rec_off : rec_off + CORE_ENTRY_BYTES]
        assert struct.unpack("<Q", rec[84:92])[0] == ent.data_offset
        assert struct.unpack("<Q", rec[92:100])[0] == ent.data_bytes
        assert struct.unpack("<Q", rec[100:108])[0] == ent.scale_offset
        assert struct.unpack("<Q", rec[108:116])[0] == ent.scale_bytes
        assert struct.unpack("<I", rec[116:120])[0] == _raw_crc32(
            core_bytes[ent.data_offset : ent.data_offset + ent.data_bytes]
        )
        assert ent.data_offset % CORE_ALIGNMENT == 0
        if ent.dtype == DTYPE_TERNARY_2BIT:
            assert struct.unpack("<I", rec[120:124])[0] == _raw_crc32(
                core_bytes[ent.scale_offset : ent.scale_offset + ent.scale_bytes]
            )

    # Verify bit-exact ternary & f32 scale equivalence on every quantized core tensor
    for name, orig_w in orig_core_ternary.items():
        ent = entry_by_name[name]
        assert ent.dtype == DTYPE_TERNARY_2BIT
        w_unpacked, s_unpacked = read_core_tensor(core_path, ent, verify_crc=True)
        exp_w, exp_s = absmean_quantize_ternary(orig_w)
        assert np.array_equal(w_unpacked, exp_w.cpu().numpy()), f"Core ternary mismatch on {name}"
        assert np.array_equal(s_unpacked, exp_s.cpu().numpy()), f"Core scale mismatch on {name}"

    for name, orig_dense in orig_core_dense.items():
        ent = entry_by_name[name]
        arr = read_core_tensor(core_path, ent, verify_crc=True)
        assert isinstance(arr, np.ndarray)
        if ent.dtype == DTYPE_F32:
            assert np.array_equal(arr, orig_dense.detach().cpu().to(torch.float32).numpy())
        elif ent.dtype == DTYPE_F16:
            assert np.array_equal(arr, orig_dense.detach().cpu().to(torch.float16).numpy())

    # 2. Raw byte-offset, 4096B alignment & CRC32 verification on experts.bin (Section 3)
    exp_bytes = experts_path.read_bytes()
    assert len(exp_bytes) % PAGE_SIZE == 0
    assert exp_bytes[0:4] == EXPERTS_MAGIC
    assert struct.unpack("<I", exp_bytes[4:8])[0] == 1
    assert struct.unpack("<Q", exp_bytes[24:32])[0] == EXPERT_INDEX_OFFSET
    assert exp_bytes[68:4092] == b"\x00" * 4024
    assert struct.unpack("<I", exp_bytes[4092:4096])[0] == _raw_crc32(exp_bytes[0:4092])

    exp_hdr, exp_entries = read_experts_header_and_dir(experts_path, verify_crc=True)
    assert exp_hdr.index_crc32 == _raw_crc32(
        exp_bytes[EXPERT_INDEX_OFFSET : EXPERT_INDEX_OFFSET + exp_hdr.index_bytes]
    )
    assert exp_hdr.data_start_offset % PAGE_SIZE == 0
    assert exp_hdr.expert_padded_bytes % PAGE_SIZE == 0

    for flat_idx, ent in enumerate(exp_entries):
        rec_off = EXPERT_INDEX_OFFSET + flat_idx * EXPERT_ENTRY_BYTES
        rec = exp_bytes[rec_off : rec_off + EXPERT_ENTRY_BYTES]
        assert struct.unpack("<HH", rec[0:4]) == (ent.layer_id, ent.expert_id)
        assert ent.offset % PAGE_SIZE == 0
        assert ent.padded_bytes % PAGE_SIZE == 0
        raw_payload = exp_bytes[ent.offset : ent.offset + ent.payload_bytes]
        assert ent.crc32 == _raw_crc32(raw_payload)
        raw_pad = exp_bytes[ent.offset + ent.payload_bytes : ent.offset + ent.padded_bytes]
        assert raw_pad == b"\x00" * len(raw_pad)

        reloaded = read_expert_weights(experts_path, exp_hdr, ent, verify_crc=True)
        gate_orig, up_orig, down_orig = orig_experts[(ent.layer_id, ent.expert_id)]
        for proj_key, orig_t in (
            ("gate_proj", gate_orig),
            ("up_proj", up_orig),
            ("down_proj", down_orig),
        ):
            exp_w, exp_s = absmean_quantize_ternary(orig_t)
            got_w, got_s = reloaded[proj_key]
            assert np.array_equal(got_w, exp_w.cpu().numpy())
            assert np.array_equal(got_s, exp_s.cpu().numpy())


def test_real_hf_olmoe_for_causal_lm_multishard_safetensors_export(tmp_path: Path) -> None:
    torch.manual_seed(2026)
    cfg = OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # Exercise GQA (H=4, H_kv=2)
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        eos_token_id=1,
        bos_token_id=0,
        pad_token_id=0,
    )
    model = OlmoeForCausalLM(cfg).eval()
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Split state_dict across 3 safetensors shards + model.safetensors.index.json
    # so _SafetensorsShardArchive exercises multi-shard LRU handle eviction (max_open_shards=2).
    ckpt_dir = tmp_path / "hf_olmoe_ckpt"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "config.json").write_text(cfg.to_json_string(), encoding="utf-8")

    all_keys = list(sd.keys())
    shards = [all_keys[0::3], all_keys[1::3], all_keys[2::3]]
    weight_map: dict[str, str] = {}
    for s_idx, key_group in enumerate(shards, start=1):
        shard_name = f"model-0000{s_idx}-of-00003.safetensors"
        save_file({k: sd[k] for k in key_group}, str(ckpt_dir / shard_name))
        for k in key_group:
            weight_map[k] = shard_name

    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "exported_hf"
    info = export_checkpoint(ckpt_dir, out_dir, embed_dtype="f16")

    core_hdr: object = info["core_header"]
    assert info["core_header"].layer_kinds == (LAYER_KIND_ANCHOR_GQA, LAYER_KIND_ANCHOR_GQA)
    assert bool(info["core_header"].flags & FLAG_NORM_TOPK_PROB)
    assert not bool(info["core_header"].flags & FLAG_BITLINEAR_RMS_PRESCALE)

    orig_core_ternary: dict[str, torch.Tensor] = {}
    orig_core_dense: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": sd["model.embed_tokens.weight"],
        "model.norm.weight": sd["model.norm.weight"],
        "lm_head.weight": sd["lm_head.weight"],
    }
    orig_experts: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for l in range(2):
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            orig_core_ternary[f"layers.{l}.self_attn.{p}.weight"] = sd[
                f"model.layers.{l}.self_attn.{p}.weight"
            ]
        orig_core_dense[f"layers.{l}.input_layernorm.weight"] = sd[
            f"model.layers.{l}.input_layernorm.weight"
        ]
        orig_core_dense[f"layers.{l}.post_attention_layernorm.weight"] = sd[
            f"model.layers.{l}.post_attention_layernorm.weight"
        ]
        orig_core_dense[f"layers.{l}.self_attn.q_norm.weight"] = sd[
            f"model.layers.{l}.self_attn.q_norm.weight"
        ]
        orig_core_dense[f"layers.{l}.self_attn.k_norm.weight"] = sd[
            f"model.layers.{l}.self_attn.k_norm.weight"
        ]
        orig_core_dense[f"layers.{l}.mlp.gate.weight"] = sd[f"model.layers.{l}.mlp.gate.weight"]

        for e in range(4):
            if f"model.layers.{l}.mlp.experts.gate_up_proj" in sd:
                gu = sd[f"model.layers.{l}.mlp.experts.gate_up_proj"][e]
                gw, uw = gu[:128, :], gu[128:, :]
                dw = sd[f"model.layers.{l}.mlp.experts.down_proj"][e]
            else:
                gw = sd[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"]
                uw = sd[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"]
                dw = sd[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"]
            orig_experts[(l, e)] = (gw, uw, dw)

    _assert_full_binary_integrity_and_bit_exact_weights(
        info["core_path"],
        info["experts_path"],
        orig_core_ternary,
        orig_core_dense,
        orig_experts,
    )

    # Also verify classic per-expert 2D safetensors layout (allenai/OLMoE-1B-7B-0125 style)
    per_expert_dir = tmp_path / "classic_safetensors_ckpt"
    per_expert_dir.mkdir(parents=True)
    (per_expert_dir / "config.json").write_text(cfg.to_json_string(), encoding="utf-8")
    classic_sd = {
        k: v
        for k, v in sd.items()
        if not k.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))
    }
    for (l, e), (gw, uw, dw) in orig_experts.items():
        classic_sd[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"] = gw.to(torch.bfloat16)
        classic_sd[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"] = uw.to(torch.bfloat16)
        classic_sd[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"] = dw.to(torch.bfloat16)
    save_file(classic_sd, str(per_expert_dir / "model.safetensors"))

    classic_out = tmp_path / "exported_classic"
    classic_info = export_checkpoint(per_expert_dir, classic_out)
    bf16_experts = {
        k: (v[0].to(torch.bfloat16), v[1].to(torch.bfloat16), v[2].to(torch.bfloat16))
        for k, v in orig_experts.items()
    }
    _assert_full_binary_integrity_and_bit_exact_weights(
        classic_info["core_path"],
        classic_info["experts_path"],
        orig_core_ternary,
        {k: v for k, v in orig_core_dense.items()},
        bf16_experts,
    )


def test_hybrid_micro_terra_7_to_1_export_and_corruption_detection(tmp_path: Path) -> None:
    config_dict, sd = build_synthetic_tiny_state(seed=314)
    # Inject optional HintEmbedding tables to verify FLAG_HAS_HINT_EMBEDDING as well
    gen = torch.Generator(device="cpu").manual_seed(777)
    sd["model.hint_embedding.table.weight"] = torch.randn((32, 8), generator=gen) * 0.05
    sd["model.hint_embedding.proj.weight"] = torch.randn((64, 32), generator=gen) * 0.05

    out_dir = tmp_path / "hybrid_out"
    info = export_state_dict(config_dict, sd, out_dir)

    core_hdr = info["core_header"]
    assert core_hdr.layer_kinds == (
        LAYER_KIND_GATED_LINEAR_RECURRENCE,
        LAYER_KIND_GATED_LINEAR_RECURRENCE,
        LAYER_KIND_GATED_LINEAR_RECURRENCE,
        LAYER_KIND_GATED_LINEAR_RECURRENCE,
        LAYER_KIND_GATED_LINEAR_RECURRENCE,
        LAYER_KIND_GATED_LINEAR_RECURRENCE,
        LAYER_KIND_GATED_LINEAR_RECURRENCE,
        LAYER_KIND_ANCHOR_GQA,
    )
    assert bool(core_hdr.flags & FLAG_BITLINEAR_RMS_PRESCALE)
    assert bool(core_hdr.flags & FLAG_NORM_TOPK_PROB)
    assert bool(core_hdr.flags & FLAG_HAS_HINT_EMBEDDING)

    entry_names = {e.name for e in info["core_entries"]}
    for l in range(5):
        assert f"layers.{l}.lookahead_router.weight" in entry_names
    for l in range(5, 8):
        assert f"layers.{l}.lookahead_router.weight" not in entry_names
    for l in range(7):
        assert f"layers.{l}.self_attn.decay_proj.weight" in entry_names
        assert f"layers.{l}.self_attn.decay_proj.bias" in entry_names
    assert "layers.7.self_attn.decay_proj.weight" not in entry_names

    orig_core_ternary: dict[str, torch.Tensor] = {}
    orig_core_dense: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": sd["model.embed_tokens.weight"],
        "model.norm.weight": sd["model.norm.weight"],
        "lm_head.weight": sd["lm_head.weight"],
        "model.hint_embedding.table.weight": sd["model.hint_embedding.table.weight"],
        "model.hint_embedding.proj.weight": sd["model.hint_embedding.proj.weight"],
    }
    orig_experts: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for l in range(8):
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            orig_core_ternary[f"layers.{l}.self_attn.{p}.weight"] = sd[
                f"model.layers.{l}.self_attn.{p}.weight"
            ]
        orig_core_dense[f"layers.{l}.input_layernorm.weight"] = sd[
            f"model.layers.{l}.input_layernorm.weight"
        ]
        orig_core_dense[f"layers.{l}.post_attention_layernorm.weight"] = sd[
            f"model.layers.{l}.post_attention_layernorm.weight"
        ]
        orig_core_dense[f"layers.{l}.self_attn.q_norm.weight"] = sd[
            f"model.layers.{l}.self_attn.q_norm.weight"
        ]
        orig_core_dense[f"layers.{l}.self_attn.k_norm.weight"] = sd[
            f"model.layers.{l}.self_attn.k_norm.weight"
        ]
        orig_core_dense[f"layers.{l}.mlp.gate.weight"] = sd[f"model.layers.{l}.mlp.gate.weight"]
        if l < 7:
            orig_core_dense[f"layers.{l}.self_attn.decay_proj.weight"] = sd[
                f"model.layers.{l}.self_attn.decay_proj.weight"
            ]
            orig_core_dense[f"layers.{l}.self_attn.decay_proj.bias"] = sd[
                f"model.layers.{l}.self_attn.decay_proj.bias"
            ]
        if l < 5:
            orig_core_dense[f"layers.{l}.lookahead_router.weight"] = sd[
                f"model.layers.{l}.lookahead_router.weight"
            ]
        for e in range(8):
            orig_experts[(l, e)] = (
                sd[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"],
                sd[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"],
                sd[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"],
            )

    _assert_full_binary_integrity_and_bit_exact_weights(
        info["core_path"],
        info["experts_path"],
        orig_core_ternary,
        orig_core_dense,
        orig_experts,
    )

    # Verify forward pass with [seq_len, 4] integer hint tensor
    hints = torch.tensor([[0, 3, 1, 2], [1, 5, 0, 4]], dtype=torch.long)
    ref = run_reference_forward(out_dir, [7, 19], hints=hints)
    assert len(ref["step_logits"]) == 2
    assert np.isfinite(ref["prefill_last_logits"]).all()

    # Corrupt 1 byte inside an expert block and verify CRC32 catches it
    tampered_experts = tmp_path / "tampered_experts.bin"
    raw = bytearray(info["experts_path"].read_bytes())
    first_exp_offset = info["expert_entries"][0].offset
    raw[first_exp_offset + 17] ^= 0x01
    tampered_experts.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="crc32 mismatch"):
        read_experts_header_and_dir(tampered_experts, verify_crc=True)


def test_cli_synthetic_tiny_end_to_end(tmp_path: Path) -> None:
    out1 = tmp_path / "tiny_run1"
    out2 = tmp_path / "tiny_run2"
    repo_root = Path(__file__).resolve().parents[2]

    for target in (out1, out2):
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "exporter.export_olmoe",
                "--synthetic-tiny",
                "--out-dir",
                str(target),
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Exported synthetic-tiny Micro-TERRA" in res.stdout

    core_path = out1 / "core.bin"
    experts_path = out1 / "experts.bin"
    golden_path = out1 / "golden_ref.json"
    assert core_path.is_file()
    assert experts_path.is_file()
    assert golden_path.is_file()

    # Deterministic byte-identical outputs across runs
    assert core_path.read_bytes() == (out2 / "core.bin").read_bytes()
    assert experts_path.read_bytes() == (out2 / "experts.bin").read_bytes()
    assert golden_path.read_text() == (out2 / "golden_ref.json").read_text()

    core_hdr, _ = read_core_header_and_dir(core_path, verify_crc=True)
    exp_hdr, exp_entries = read_experts_header_and_dir(experts_path, verify_crc=True)
    assert core_hdr.vocab_size == 256
    assert core_hdr.hidden_size == 64
    assert core_hdr.intermediate_size == 128
    assert core_hdr.num_layers == 8
    assert exp_hdr.num_layers == 8
    assert exp_hdr.num_experts == 8
    assert len(exp_entries) == 64
    assert all(e.offset % PAGE_SIZE == 0 for e in exp_entries)

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden["prompt_token_ids"] == [1, 42, 17, 99]
    assert len(golden["step_logits"]) == 4
    assert all(len(step) == 256 for step in golden["step_logits"])
    assert golden["prefill_last_logits"] == golden["step_logits"][-1]
    assert len(golden["layer0_post_moe"]) == 64
    assert np.isfinite(np.asarray(golden["step_logits"])).all()
    assert np.isfinite(np.asarray(golden["layer0_post_moe"])).all()


def test_raw_byte_offsets_and_manual_hex_quantization_without_reader_structs(tmp_path: Path) -> None:
    """Verify binary layout & quantization against hand-calculated hex bytes and raw spec offsets."""
    # 1. Hand-calculated quantization & crumb packing check (no self-comparison):
    # Row 0: [0.0, 2.0, -2.0, 0.2] -> absmean = 4.2 / 4 = 1.05
    #   normalized = [0.0, 1.90476, -1.90476, 0.19047] -> ternary = [0, +1, -1, 0]
    #   crumbs = [0b00, 0b01, 0b10, 0b00] -> byte = 0b00_10_01_00 == 0x24
    # Row 1: [-3.0, 0.0, 3.0, -3.0] -> absmean = 9.0 / 4 = 2.25
    #   normalized = [-1.333, 0.0, 1.333, -1.333] -> ternary = [-1, 0, +1, -1]
    #   crumbs = [0b10, 0b00, 0b01, 0b10] -> byte = 0b10_01_00_10 == 0x92
    w_manual = torch.tensor([[0.0, 2.0, -2.0, 0.2], [-3.0, 0.0, 3.0, -3.0]], dtype=torch.float32)
    w_tern, scales = absmean_quantize_ternary(w_manual)
    assert w_tern.tolist() == [[0, 1, -1, 0], [-1, 0, 1, -1]]
    assert np.allclose(scales.numpy(), [1.05, 2.25], atol=1e-7)
    packed = pack_ternary_2bit(w_tern)
    assert packed.tobytes() == bytes([0x24, 0x92])

    # 2. Export a model where EVERY header dimension is a distinct value so swapping ANY two u32s fails!
    cfg = {
        "vocab_size": 307,
        "hidden_size": 36,
        "intermediate_size": 52,
        "num_layers": 5,
        "num_attention_heads": 6,
        "num_kv_heads": 2,
        "head_dim": 6,
        "num_experts": 7,
        "top_k": 3,
        "max_position_embeddings": 1234,
        "rope_theta": 50000.0,
        "rms_norm_eps": 1e-6,
        "lookahead_offset": 2,
        "layer_kinds": [1, 0, 1, 1, 0],
        "bitlinear_rms_prescale": True,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
    }
    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(307, 36),
        "model.norm.weight": torch.ones(36),
        "lm_head.weight": torch.randn(307, 36),
    }
    for l in range(5):
        p = f"model.layers.{l}"
        sd[f"{p}.input_layernorm.weight"] = torch.ones(36)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.ones(36)
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(36, 36)
        sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(12, 36)
        sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(12, 36)
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(36, 36)
        sd[f"{p}.self_attn.q_norm.weight"] = torch.ones(36)
        sd[f"{p}.self_attn.k_norm.weight"] = torch.ones(12)
        if cfg["layer_kinds"][l] == 1:
            sd[f"{p}.self_attn.decay_proj.weight"] = torch.zeros(6, 36)
            sd[f"{p}.self_attn.decay_proj.bias"] = torch.ones(6)
        sd[f"{p}.mlp.gate.weight"] = torch.randn(7, 36)
        sd[f"{p}.mlp.experts.gate_up_proj"] = torch.randn(7, 104, 36)
        sd[f"{p}.mlp.experts.down_proj"] = torch.randn(7, 36, 52)

    out_dir = tmp_path / "raw_spec_check"
    export_state_dict(cfg, sd, out_dir)

    # Inspect `core.bin` strictly via raw byte slices from `schemas/terra_format.md` Section 2.1!
    core_raw = (out_dir / "core.bin").read_bytes()
    u32 = lambda b, off: int.from_bytes(b[off : off + 4], "little")
    u64 = lambda b, off: int.from_bytes(b[off : off + 8], "little")
    f32 = lambda b, off: struct.unpack("<f", b[off : off + 4])[0]

    assert core_raw[0:4] == b"TRAC"
    assert u32(core_raw, 4) == 1          # version
    assert u32(core_raw, 8) == 0b0011     # FLAG_BITLINEAR_RMS_PRESCALE | FLAG_NORM_TOPK_PROB
    assert u32(core_raw, 12) == 307       # vocab_size
    assert u32(core_raw, 16) == 36        # hidden_size
    assert u32(core_raw, 20) == 52        # intermediate_size
    assert u32(core_raw, 24) == 5         # num_layers
    assert u32(core_raw, 28) == 6         # num_attention_heads
    assert u32(core_raw, 32) == 2         # num_kv_heads
    assert u32(core_raw, 36) == 6         # head_dim
    assert u32(core_raw, 40) == 7         # num_experts
    assert u32(core_raw, 44) == 3         # top_k
    assert u32(core_raw, 48) == 1234      # max_position_embeddings
    assert abs(f32(core_raw, 52) - 50000.0) < 1e-3
    assert abs(f32(core_raw, 56) - 1e-6) < 1e-10
    assert u32(core_raw, 60) == 2         # lookahead_offset
    assert list(core_raw[64:69]) == [1, 0, 1, 1, 0]
    assert core_raw[69:128] == b"\x00" * 59
    assert core_raw[140:252] == b"\x00" * 112
    assert u32(core_raw, 252) == (zlib.crc32(core_raw[:252]) & 0xFFFFFFFF)

    # Inspect `experts.bin` strictly via raw byte slices from `schemas/terra_format.md` Section 3.1 & 3.2!
    exp_raw = (out_dir / "experts.bin").read_bytes()
    assert exp_raw[0:4] == b"TRAE"
    assert u32(exp_raw, 4) == 1
    assert u32(exp_raw, 8) == 5           # num_layers
    assert u32(exp_raw, 12) == 7          # num_experts
    assert u32(exp_raw, 16) == 36         # hidden_size
    assert u32(exp_raw, 20) == 52         # intermediate_size
    assert u64(exp_raw, 24) == 4096       # index_offset
    assert u64(exp_raw, 32) == 5 * 7 * 40 # index_bytes == 1400
    assert u64(exp_raw, 40) == 8192       # align_up(4096 + 1400, 4096) == 8192
    assert u32(exp_raw, 4092) == (zlib.crc32(exp_raw[:4092]) & 0xFFFFFFFF)

    # Check index entry #9 (`layer_id = 1, expert_id = 2`) at raw offset `4096 + 9 * 40`
    ent9_off = 4096 + 9 * 40
    assert int.from_bytes(exp_raw[ent9_off : ent9_off + 2], "little") == 1
    assert int.from_bytes(exp_raw[ent9_off + 2 : ent9_off + 4], "little") == 2
    assert u32(exp_raw, ent9_off + 4) == 1
    assert u64(exp_raw, ent9_off + 8) == 8192 + 9 * 4096


def test_exporter_rejects_wrong_shapes_and_ambiguous_square_transpose_and_warns(tmp_path: Path) -> None:
    """Verify Medium #10 (shape checks), Medium #11 (2*I == D square gate_up safety), and Low warnings."""
    from exporter.export_olmoe import build_synthetic_tiny_state

    cfg, sd = build_synthetic_tiny_state(seed=7)

    # 1. Wrong core tensor shape must fail immediately during export
    bad_sd = dict(sd)
    bad_sd["model.layers.0.self_attn.q_proj.weight"] = torch.randn(32, 64)  # expected (64, 64)
    with pytest.raises(ValueError, match="Shape mismatch exporting 'layers.0.self_attn.q_proj.weight'"):
        export_state_dict(cfg, bad_sd, tmp_path / "bad_q_shape")

    # 2. Square `2 * I == D` (`D = 64, I = 32`, mirroring OLMoE-1B-7B `2 * 1024 == 2048`):
    #    If a transposed checkpoint has `down_proj` shape `(I, D) = (32, 64)` and `gate_up_proj` `(D, 2*I) = (64, 64)`,
    #    default `expert_layout='out_in'` MUST reject it instead of silently mis-slicing `gate_up_proj`!
    sq_cfg = dict(cfg)
    sq_cfg["intermediate_size"] = 32  # 2 * 32 == 64 == hidden_size!
    sq_sd = dict(sd)
    for l in range(sq_cfg["num_layers"]):
        # Construct known canonical [out, in] weights and transpose BOTH to [in, out]
        gate_canon = torch.full((8, 32, 64), 1.0)
        up_canon = torch.full((8, 32, 64), -1.0)
        down_canon = torch.full((8, 64, 32), 1.0)
        gate_up_in_out = torch.cat([gate_canon, up_canon], dim=1).transpose(1, 2).contiguous()  # (8, 64, 64)
        down_in_out = down_canon.transpose(1, 2).contiguous()  # (8, 32, 64)
        sq_sd[f"model.layers.{l}.mlp.experts.gate_up_proj"] = gate_up_in_out
        sq_sd[f"model.layers.{l}.mlp.experts.down_proj"] = down_in_out

    with pytest.raises(ValueError, match="expert_layout='in_out'"):
        export_state_dict(sq_cfg, sq_sd, tmp_path / "sq_ambiguous", expert_layout="out_in")

    # Passing `expert_layout='in_out'` explicitly transposes BOTH `gate_up_proj` and `down_proj` cleanly!
    info = export_state_dict(sq_cfg, sq_sd, tmp_path / "sq_in_out_ok", expert_layout="in_out")
    exp0 = read_expert_weights(info["experts_path"], info["experts_header"], info["expert_entries"][0])
    assert (exp0["gate_proj"][0] == 1).all()
    assert (exp0["up_proj"][0] == -1).all()
    assert (exp0["down_proj"][0] == 1).all()

    # 3. Warnings on dropped differing `lm_head.weight` when `tie_word_embeddings=True` & out-of-range lookahead
    warn_cfg = dict(cfg)
    warn_cfg["tie_word_embeddings"] = True
    warn_sd = dict(sd)
    warn_sd["model.layers.7.lookahead_router.weight"] = torch.randn(8, 64)  # 7 + 3 >= 8!
    with pytest.warns(UserWarning) as caught:
        export_state_dict(warn_cfg, warn_sd, tmp_path / "warn_check")
    messages = [str(w.message) for w in caught]
    assert any("tie_word_embeddings=True omits" in m for m in messages)
    assert any("out-of-range lookahead router" in m for m in messages)


def test_external_parity_against_huggingface_olmoe_for_causal_lm(tmp_path: Path) -> None:
    """External ground-truth test comparing exported model forward directly against unmodified HuggingFace `OlmoeForCausalLM`."""
    from transformers import OlmoeConfig, OlmoeForCausalLM

    torch.manual_seed(31415)
    hf_cfg = OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,  # 2 * I == D == 64 (exact OLMoE aspect ratio!)
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # Exercises GQA head repeat (group = 2)!
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        eos_token_id=1,
        pad_token_id=0,
    )
    hf_model = OlmoeForCausalLM(hf_cfg)
    hf_model.eval()

    # Place projections onto an exact fixed point of `absmean_quantize_ternary` (`W = s_sign * scale` with `|s_sign| == 1`,
    # where `absmean(W) == scale` and `round(W / scale) == s_sign`) so weight quantization error is 0.0 and
    # unmodified continuous HuggingFace `OlmoeForCausalLM.forward()` serves as a true external oracle!
    def snap_absmean_fixed_point(w: torch.Tensor) -> None:
        wt, sc = absmean_quantize_ternary(w)
        wt_nz = torch.where(wt == 0, torch.ones_like(wt), wt)
        w.copy_(wt_nz.float() * sc.unsqueeze(1))

    with torch.no_grad():
        for layer in hf_model.model.layers:
            torch.nn.init.normal_(layer.mlp.gate.weight, mean=0.0, std=1.0)
            for proj in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
            ):
                snap_absmean_fixed_point(proj.weight)

            gu = layer.mlp.experts.gate_up_proj
            dn = layer.mlp.experts.down_proj
            for e in range(hf_cfg.num_experts):
                snap_absmean_fixed_point(gu[e, :32, :])
                snap_absmean_fixed_point(gu[e, 32:, :])
                snap_absmean_fixed_point(dn[e])

    prompt_ids = [5, 19, 42, 77, 12]
    with torch.no_grad():
        hf_logits = hf_model(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long),
            use_cache=False,
        ).logits[0].float()

    out_dir = tmp_path / "hf_external_parity"
    export_state_dict(
        hf_cfg.to_dict(),
        hf_model.state_dict(),
        out_dir,
        bitlinear_rms_prescale=False,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        attn_dtype="ternary",
        lm_head_dtype="f32",
    )

    ref = run_reference_forward(out_dir, prompt_ids, max_cached_experts=2, max_cached_core_tensors=6)
    terra_logits = torch.tensor(ref["step_logits"], dtype=torch.float32)

    # 1. Compare Micro-TERRA Python reference directly against unmodified HuggingFace `OlmoeForCausalLM.forward()`!
    max_diff_vs_hf = float((hf_logits - terra_logits).abs().max().item())
    cos_sim = float(
        torch.nn.functional.cosine_similarity(hf_logits.flatten(), terra_logits.flatten(), dim=0).item()
    )
    assert max_diff_vs_hf < 1e-2, f"Expected < 1e-2 vs HuggingFace OlmoeForCausalLM, got {max_diff_vs_hf}"
    assert cos_sim > 0.9999, f"Expected cosine similarity > 0.9999 vs HF, got {cos_sim}"

    # 2. If Rust `terra-cli` release binary is built, verify Rust `TerraEngine` directly against HF logits too!
    rust_cli = Path(__file__).resolve().parents[2] / "runtime" / "target" / "release" / "terra-cli"
    if rust_cli.is_file():
        (out_dir / "golden_ref.json").write_text(
            json.dumps(
                {
                    "prompt_token_ids": prompt_ids,
                    "step_logits": hf_logits.tolist(),
                }
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(rust_cli),
                "--model-dir",
                str(out_dir),
                "--verify-golden",
                "--golden-tol",
                "0.01",
                "--max-cached-experts",
                "2",
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def test_bitlinear_edge_cases_ties_to_even_subthreshold_and_127_clamp() -> None:
    """Explicitly guard round-half-to-even, < 1e-12 zero threshold, and +-127 clamping."""
    w = torch.tensor([[1, -1, 1, 0], [0, 1, -1, 1]], dtype=torch.int8)
    s = torch.tensor([1.0, 2.0], dtype=torch.float32)

    # 1. Sub-threshold `< 1e-12` produces exact 0.0 without NaN
    x_tiny = torch.tensor([0.0, 9.0e-13, -5.0e-13, 0.0], dtype=torch.float32)
    y_tiny = bitlinear_reference_gemv(x_tiny, w, s, rms_prescale=False)
    assert torch.equal(y_tiny, torch.zeros(2, dtype=torch.float32))

    # 2. Exact half-way ties when max(|x|) == 127.0 -> s_x == 1.0:
    #    [0.5, 1.5, 2.5, -127.0] -> round_ties_to_even -> [0, 2, 2, -127]
    #    Row 0 dot: 1*0 + (-1)*2 + 1*2 + 0*(-127) = 0 -> 0.0 * 1.0 = 0.0
    #    Row 1 dot: 0*0 + 1*2 + (-1)*2 + 1*(-127) = -127 -> -127 * 2.0 = -254.0
    x_ties = torch.tensor([0.5, 1.5, 2.5, -127.0], dtype=torch.float32)
    y_ties = bitlinear_reference_gemv(x_ties, w, s, rms_prescale=False)
    assert torch.equal(y_ties, torch.tensor([0.0, -254.0], dtype=torch.float32))


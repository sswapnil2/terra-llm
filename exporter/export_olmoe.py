"""Micro-TERRA checkpoint exporter, binary writers/readers, and golden reference runner.

Conforms strictly to Sections 2, 3, and 4 of ``schemas/terra_format.md``:
- ``core.bin``: ``TRAC`` v1 header (256B), ``CoreTensorEntry`` directory (128B each),
  and 64-byte-aligned tensor & scale payloads protected by IEEE CRC-32.
- ``experts.bin``: ``TRAE`` v1 Page 0 header (4096B), ``ExpertIndexEntry`` directory (40B each),
  zero-padded to 4096B page boundary, followed by 4096B-aligned ``TernarySwiGLU`` expert blocks.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence, Union
import warnings

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F

from exporter.ternary_pack import (
    absmean_quantize_ternary,
    align_up,
    bitlinear_reference_gemv,
    crc32_ieee,
    pack_ternary_2bit,
    unpack_ternary_2bit,
)

# ---------------------------------------------------------------------------
# Binary Format Constants & Structs (schemas/terra_format.md Sections 2 & 3)
# ---------------------------------------------------------------------------

CORE_MAGIC = b"TRAC"
EXPERTS_MAGIC = b"TRAE"
FORMAT_VERSION = 1

CORE_HEADER_BYTES = 256
CORE_ENTRY_BYTES = 128
CORE_ALIGNMENT = 64

PAGE_SIZE = 4096
EXPERTS_HEADER_BYTES = 4096
EXPERT_INDEX_OFFSET = 4096
EXPERT_ENTRY_BYTES = 40
EXPERT_FLAG_TERNARY_SWIGLU = 1

# Header flags in core.bin (offset 8..12)
FLAG_BITLINEAR_RMS_PRESCALE = 1 << 0
FLAG_NORM_TOPK_PROB = 1 << 1
FLAG_TIE_WORD_EMBEDDINGS = 1 << 2
FLAG_HAS_HINT_EMBEDDING = 1 << 3

# Per-layer kind enum in core.bin (offset 64..128)
LAYER_KIND_ANCHOR_GQA = 0
LAYER_KIND_GATED_LINEAR_RECURRENCE = 1

# CoreTensorEntry dtype enum (rel offset 64..65)
DTYPE_TERNARY_2BIT = 0
DTYPE_F32 = 1
DTYPE_F16 = 2
DTYPE_I8 = 3

# Section 2.1: bytes 0..252 prefix + bytes 252..256 header_crc32
CORE_HEADER_PREFIX_STRUCT = struct.Struct("<4s12IffI64sIQ112s")
CORE_CRC_STRUCT = struct.Struct("<I")
assert CORE_HEADER_PREFIX_STRUCT.size + CORE_CRC_STRUCT.size == CORE_HEADER_BYTES

# Section 2.2: 128-byte CoreTensorEntry
CORE_TENSOR_ENTRY_STRUCT = struct.Struct("<64sBBH4IQQQQIII")
assert CORE_TENSOR_ENTRY_STRUCT.size == CORE_ENTRY_BYTES

# Section 3.1: bytes 0..4092 prefix + bytes 4092..4096 header_crc32
EXPERTS_HEADER_PREFIX_STRUCT = struct.Struct("<4s5I3QI2Q4024s")
assert EXPERTS_HEADER_PREFIX_STRUCT.size + CORE_CRC_STRUCT.size == EXPERTS_HEADER_BYTES

# Section 3.2: 40-byte ExpertIndexEntry
EXPERT_INDEX_ENTRY_STRUCT = struct.Struct("<HHIQQQII")
assert EXPERT_INDEX_ENTRY_STRUCT.size == EXPERT_ENTRY_BYTES


# ---------------------------------------------------------------------------
# Dataclasses for Headers, Directory Entries, and Model Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerraModelConfig:
    """Normalized model metadata serialized into core.bin and experts.bin."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    lookahead_offset: int = 3
    layer_kinds: tuple[int, ...] = ()
    flags: int = 0

    def __post_init__(self) -> None:
        if not (1 <= self.num_layers <= 64):
            raise ValueError(f"num_layers must be in 1..=64, got {self.num_layers}")
        if self.hidden_size <= 0 or self.hidden_size % 4 != 0:
            raise ValueError(
                f"hidden_size must be positive and divisible by 4, got {self.hidden_size}"
            )
        if self.intermediate_size <= 0 or self.intermediate_size % 4 != 0:
            raise ValueError(
                f"intermediate_size must be positive and divisible by 4, got {self.intermediate_size}"
            )
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be positive and even, got {self.head_dim}")
        if self.num_attention_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Attention head counts must be positive")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.num_experts <= 0 or self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError(
                f"Invalid expert routing config: num_experts={self.num_experts}, top_k={self.top_k}"
            )
        if len(self.layer_kinds) == 0:
            object.__setattr__(
                self, "layer_kinds", tuple(LAYER_KIND_ANCHOR_GQA for _ in range(self.num_layers))
            )
        elif len(self.layer_kinds) != self.num_layers:
            raise ValueError(
                f"layer_kinds length ({len(self.layer_kinds)}) != num_layers ({self.num_layers})"
            )
        for lk in self.layer_kinds:
            if lk not in (LAYER_KIND_ANCHOR_GQA, LAYER_KIND_GATED_LINEAR_RECURRENCE):
                raise ValueError(f"Invalid layer_kind value: {lk}")


@dataclass(frozen=True)
class CoreHeader:
    magic: bytes
    version: int
    flags: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    lookahead_offset: int
    layer_kinds: tuple[int, ...]
    tensor_count: int
    data_section_offset: int
    header_crc32: int


@dataclass(frozen=True)
class CoreTensorEntry:
    name: str
    dtype: int
    ndim: int
    flags: int
    shape: tuple[int, int, int, int]
    data_offset: int
    data_bytes: int
    scale_offset: int
    scale_bytes: int
    data_crc32: int
    scale_crc32: int
    reserved: int = 0

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return self.shape[: self.ndim]


@dataclass(frozen=True)
class ExpertsHeader:
    magic: bytes
    version: int
    num_layers: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    index_offset: int
    index_bytes: int
    data_start_offset: int
    index_crc32: int
    expert_payload_bytes: int
    expert_padded_bytes: int
    header_crc32: int


@dataclass(frozen=True)
class ExpertIndexEntry:
    layer_id: int
    expert_id: int
    flags: int
    offset: int
    payload_bytes: int
    padded_bytes: int
    crc32: int
    reserved: int = 0


def compute_expert_payload_bytes(hidden_size: int, intermediate_size: int) -> int:
    """Exact unpadded byte size per expert block (Section 3.3)."""
    if hidden_size % 4 != 0 or intermediate_size % 4 != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) and intermediate_size ({intermediate_size}) "
            "must be divisible by 4"
        )
    return (3 * hidden_size * intermediate_size) // 4 + 4 * (2 * intermediate_size + hidden_size)


# ---------------------------------------------------------------------------
# Binary Writers: CoreBinWriter & ExpertsBinWriter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepackedTernary:
    """Pre-packed 2-bit ternary matrix + per-row f32 scales for fast streaming writes."""

    rows: int
    cols: int
    packed: np.ndarray
    scales: np.ndarray


TensorProvider = Union[
    torch.Tensor,
    np.ndarray,
    PrepackedTernary,
    Callable[[], Union[torch.Tensor, np.ndarray, PrepackedTernary]],
]


class CoreBinWriter:
    """Streaming writer for ``core.bin`` conforming to Section 2 of ``schemas/terra_format.md``."""

    def __init__(self, config: TerraModelConfig) -> None:
        self.config = config
        self._pending: list[tuple[str, int, TensorProvider]] = []
        self._seen_names: set[str] = set()

    def add_tensor(
        self,
        name: str,
        tensor_or_getter: TensorProvider,
        dtype: int,
    ) -> None:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) == 0 or len(name_bytes) >= 64:
            raise ValueError(
                f"Tensor name {name!r} UTF-8 byte length must be in 1..=63, got {len(name_bytes)}"
            )
        if name in self._seen_names:
            raise ValueError(f"Duplicate tensor registered in CoreBinWriter: {name!r}")
        if dtype not in (DTYPE_TERNARY_2BIT, DTYPE_F32, DTYPE_F16, DTYPE_I8):
            raise ValueError(f"Unsupported CoreTensorEntry dtype enum: {dtype}")
        self._seen_names.add(name)
        self._pending.append((name, dtype, tensor_or_getter))

    def add_prepacked_ternary(
        self,
        name: str,
        provider: Union[PrepackedTernary, Callable[[], PrepackedTernary]],
    ) -> None:
        """Register an already-packed 2-bit ternary matrix + f32 scale vector."""
        self.add_tensor(name, provider, DTYPE_TERNARY_2BIT)

    def write(self, out_path: Union[str, Path]) -> tuple[CoreHeader, list[CoreTensorEntry]]:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tensor_count = len(self._pending)
        dir_end = CORE_HEADER_BYTES + tensor_count * CORE_ENTRY_BYTES
        data_section_offset = align_up(dir_end, CORE_ALIGNMENT)

        entries: list[CoreTensorEntry] = []

        with path.open("wb+") as f:
            # Reserve header + directory + alignment padding with zeros.
            f.write(b"\x00" * data_section_offset)
            cur_offset = data_section_offset

            for name, dtype, provider in self._pending:
                raw_tensor = provider() if callable(provider) else provider
                tensor: torch.Tensor | None = None

                # Pad current file cursor to 64-byte boundary.
                aligned_data_offset = align_up(cur_offset, CORE_ALIGNMENT)
                if aligned_data_offset > cur_offset:
                    f.write(b"\x00" * (aligned_data_offset - cur_offset))
                    cur_offset = aligned_data_offset

                if isinstance(raw_tensor, PrepackedTernary):
                    if dtype != DTYPE_TERNARY_2BIT:
                        raise ValueError(
                            f"PrepackedTernary for {name!r} requires DTYPE_TERNARY_2BIT"
                        )
                    ndim = 2
                    shape_4 = (int(raw_tensor.rows), int(raw_tensor.cols), 0, 0)
                    packed_arr = np.ascontiguousarray(raw_tensor.packed, dtype=np.uint8)
                    if packed_arr.shape != (raw_tensor.rows, raw_tensor.cols // 4):
                        raise ValueError(
                            f"Prepacked shape mismatch for {name!r}: {packed_arr.shape}"
                        )
                    if np.any((packed_arr & (packed_arr >> 1)) & 0x55):
                        raise ValueError(f"Invalid 0b11 crumb in prepacked tensor {name!r}")
                    scale_arr = np.ascontiguousarray(raw_tensor.scales, dtype="<f4")
                    if scale_arr.shape != (raw_tensor.rows,):
                        raise ValueError(
                            f"Prepacked scales shape mismatch for {name!r}: {scale_arr.shape}"
                        )

                    data_buf = packed_arr.tobytes(order="C")
                    scale_buf = scale_arr.tobytes(order="C")

                    data_offset = cur_offset
                    data_bytes = len(data_buf)
                    data_crc = crc32_ieee(data_buf)
                    f.write(data_buf)
                    cur_offset += data_bytes

                    scale_offset = align_up(cur_offset, CORE_ALIGNMENT)
                    if scale_offset > cur_offset:
                        f.write(b"\x00" * (scale_offset - cur_offset))
                        cur_offset = scale_offset

                    scale_bytes = len(scale_buf)
                    scale_crc = crc32_ieee(scale_buf)
                    f.write(scale_buf)
                    cur_offset += scale_bytes
                else:
                    if isinstance(raw_tensor, np.ndarray):
                        tensor = torch.from_numpy(raw_tensor)
                    elif isinstance(raw_tensor, torch.Tensor):
                        tensor = raw_tensor.detach().cpu()
                    else:
                        raise TypeError(
                            f"Expected torch.Tensor, np.ndarray, or PrepackedTernary for {name!r}, "
                            f"got {type(raw_tensor).__name__}"
                        )

                    ndim = tensor.ndim
                    if not (1 <= ndim <= 4):
                        raise ValueError(f"Tensor {name!r} rank must be 1..=4, got {ndim}")
                    shape_4 = tuple(int(s) for s in tensor.shape) + (0,) * (4 - ndim)

                    if dtype == DTYPE_TERNARY_2BIT:
                        if ndim != 2:
                            raise ValueError(
                                f"Ternary2Bit tensor {name!r} must be 2D, got shape {tuple(tensor.shape)}"
                            )
                        w_tern, scales = absmean_quantize_ternary(tensor)
                        packed = pack_ternary_2bit(w_tern)
                        data_buf = packed.tobytes(order="C")
                        scale_arr = np.ascontiguousarray(scales.cpu().numpy(), dtype="<f4")
                        scale_buf = scale_arr.tobytes(order="C")

                        data_offset = cur_offset
                        data_bytes = len(data_buf)
                        data_crc = crc32_ieee(data_buf)
                        f.write(data_buf)
                        cur_offset += data_bytes

                        # Align scale vector to 64 bytes as well for aligned SIMD loads.
                        scale_offset = align_up(cur_offset, CORE_ALIGNMENT)
                        if scale_offset > cur_offset:
                            f.write(b"\x00" * (scale_offset - cur_offset))
                            cur_offset = scale_offset

                        scale_bytes = len(scale_buf)
                        scale_crc = crc32_ieee(scale_buf)
                        f.write(scale_buf)
                        cur_offset += scale_bytes
                    else:
                        if dtype == DTYPE_F32:
                            arr = np.ascontiguousarray(
                                tensor.to(dtype=torch.float32).numpy(), dtype="<f4"
                            )
                        elif dtype == DTYPE_F16:
                            arr = np.ascontiguousarray(
                                tensor.to(dtype=torch.float16).numpy(), dtype="<f2"
                            )
                        elif dtype == DTYPE_I8:
                            arr = np.ascontiguousarray(
                                tensor.to(dtype=torch.int8).numpy(), dtype=np.int8
                            )
                        else:
                            raise AssertionError(f"Unhandled dtype {dtype}")

                        data_buf = arr.tobytes(order="C")
                        data_offset = cur_offset
                        data_bytes = len(data_buf)
                        data_crc = crc32_ieee(data_buf)
                        f.write(data_buf)
                        cur_offset += data_bytes

                        scale_offset = 0
                        scale_bytes = 0
                        scale_crc = 0

                entries.append(
                    CoreTensorEntry(
                        name=name,
                        dtype=dtype,
                        ndim=ndim,
                        flags=0,
                        shape=shape_4,  # type: ignore[arg-type]
                        data_offset=data_offset,
                        data_bytes=data_bytes,
                        scale_offset=scale_offset,
                        scale_bytes=scale_bytes,
                        data_crc32=data_crc,
                        scale_crc32=scale_crc,
                        reserved=0,
                    )
                )

                # Release temporary tensor immediately so streaming peak RAM stays minimal.
                del raw_tensor, tensor

            # Pad trailing file tail to 64-byte alignment for clean mmap alignment.
            end_aligned = align_up(cur_offset, CORE_ALIGNMENT)
            if end_aligned > cur_offset:
                f.write(b"\x00" * (end_aligned - cur_offset))

            # Serialize directory at offset 256.
            f.seek(CORE_HEADER_BYTES)
            for entry in entries:
                name_padded = entry.name.encode("utf-8").ljust(64, b"\x00")
                raw_entry = CORE_TENSOR_ENTRY_STRUCT.pack(
                    name_padded,
                    entry.dtype,
                    entry.ndim,
                    entry.flags,
                    entry.shape[0],
                    entry.shape[1],
                    entry.shape[2],
                    entry.shape[3],
                    entry.data_offset,
                    entry.data_bytes,
                    entry.scale_offset,
                    entry.scale_bytes,
                    entry.data_crc32,
                    entry.scale_crc32,
                    entry.reserved,
                )
                f.write(raw_entry)

            # Serialize 256-byte header at offset 0.
            cfg = self.config
            layer_kinds_buf = bytes(cfg.layer_kinds) + b"\x00" * (64 - len(cfg.layer_kinds))
            header_prefix = CORE_HEADER_PREFIX_STRUCT.pack(
                CORE_MAGIC,
                FORMAT_VERSION,
                cfg.flags,
                cfg.vocab_size,
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.num_layers,
                cfg.num_attention_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.num_experts,
                cfg.top_k,
                cfg.max_position_embeddings,
                float(cfg.rope_theta),
                float(cfg.rms_norm_eps),
                cfg.lookahead_offset,
                layer_kinds_buf,
                tensor_count,
                data_section_offset,
                b"\x00" * 112,
            )
            header_crc = crc32_ieee(header_prefix)
            f.seek(0)
            f.write(header_prefix)
            f.write(CORE_CRC_STRUCT.pack(header_crc))
            f.flush()

        header = CoreHeader(
            magic=CORE_MAGIC,
            version=FORMAT_VERSION,
            flags=cfg.flags,
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_layers=cfg.num_layers,
            num_attention_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            max_position_embeddings=cfg.max_position_embeddings,
            rope_theta=float(cfg.rope_theta),
            rms_norm_eps=float(cfg.rms_norm_eps),
            lookahead_offset=cfg.lookahead_offset,
            layer_kinds=cfg.layer_kinds,
            tensor_count=tensor_count,
            data_section_offset=data_section_offset,
            header_crc32=header_crc,
        )
        return header, entries


class ExpertsBinWriter:
    """4KB-page-aligned streaming writer for ``experts.bin`` (Section 3 of ``schemas/terra_format.md``)."""

    def __init__(
        self,
        out_path: Union[str, Path],
        *,
        num_layers: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        if num_layers <= 0 or num_experts <= 0:
            raise ValueError("num_layers and num_experts must be positive")
        self.path = Path(out_path)
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)

        self.index_offset = EXPERT_INDEX_OFFSET
        self.index_bytes = self.num_layers * self.num_experts * EXPERT_ENTRY_BYTES
        self.data_start_offset = align_up(self.index_offset + self.index_bytes, PAGE_SIZE)
        self.expert_payload_bytes = compute_expert_payload_bytes(
            self.hidden_size, self.intermediate_size
        )
        self.expert_padded_bytes = align_up(self.expert_payload_bytes, PAGE_SIZE)
        self._pad_tail = b"\x00" * (self.expert_padded_bytes - self.expert_payload_bytes)

        self._entries: dict[tuple[int, int], ExpertIndexEntry] = {}
        self._fp: Any = None
        self._finalized = False

    def __enter__(self) -> "ExpertsBinWriter":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None and not self._finalized:
            self.finalize()
        elif self._fp is not None:
            self._fp.close()
            self._fp = None

    def open(self) -> None:
        if self._fp is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("wb+")
        # Pre-fill Page 0 header + index + alignment padding up to data_start_offset with 0x00.
        self._fp.write(b"\x00" * self.data_start_offset)

    def write_expert(
        self,
        layer_id: int,
        expert_id: int,
        gate_weight: Union[torch.Tensor, np.ndarray],
        up_weight: Union[torch.Tensor, np.ndarray],
        down_weight: Union[torch.Tensor, np.ndarray],
    ) -> ExpertIndexEntry:
        """Quantize, 2-bit pack, 4KB-pad, and write a single ``(layer_id, expert_id)`` block."""
        if self._fp is None:
            self.open()
        if self._finalized:
            raise RuntimeError("ExpertsBinWriter is already finalized")
        if not (0 <= layer_id < self.num_layers):
            raise ValueError(f"layer_id {layer_id} out of range 0..{self.num_layers}")
        if not (0 <= expert_id < self.num_experts):
            raise ValueError(f"expert_id {expert_id} out of range 0..{self.num_experts}")

        D = self.hidden_size
        I = self.intermediate_size

        gate_t = torch.as_tensor(gate_weight)
        up_t = torch.as_tensor(up_weight)
        down_t = torch.as_tensor(down_weight)

        if tuple(gate_t.shape) != (I, D):
            raise ValueError(
                f"Layer {layer_id} expert {expert_id} gate_proj shape {tuple(gate_t.shape)} != ({I}, {D})"
            )
        if tuple(up_t.shape) != (I, D):
            raise ValueError(
                f"Layer {layer_id} expert {expert_id} up_proj shape {tuple(up_t.shape)} != ({I}, {D})"
            )
        if tuple(down_t.shape) != (D, I):
            raise ValueError(
                f"Layer {layer_id} expert {expert_id} down_proj shape {tuple(down_t.shape)} != ({D}, {I})"
            )

        gate_q, gate_s = absmean_quantize_ternary(gate_t)
        up_q, up_s = absmean_quantize_ternary(up_t)
        down_q, down_s = absmean_quantize_ternary(down_t)

        gate_packed = pack_ternary_2bit(gate_q)
        up_packed = pack_ternary_2bit(up_q)
        down_packed = pack_ternary_2bit(down_q)

        # Section 3.3 exact sub-buffer order:
        # 1. gate_proj.packed, 2. up_proj.packed, 3. down_proj.packed,
        # 4. gate_proj.scales, 5. up_proj.scales, 6. down_proj.scales
        payload = b"".join(
            (
                gate_packed.tobytes(order="C"),
                up_packed.tobytes(order="C"),
                down_packed.tobytes(order="C"),
                np.ascontiguousarray(gate_s.cpu().numpy(), dtype="<f4").tobytes(order="C"),
                np.ascontiguousarray(up_s.cpu().numpy(), dtype="<f4").tobytes(order="C"),
                np.ascontiguousarray(down_s.cpu().numpy(), dtype="<f4").tobytes(order="C"),
            )
        )
        if len(payload) != self.expert_payload_bytes:
            raise AssertionError(
                f"Internal payload size mismatch: {len(payload)} != {self.expert_payload_bytes}"
            )

        crc = crc32_ieee(payload)
        flat_idx = layer_id * self.num_experts + expert_id
        offset = self.data_start_offset + flat_idx * self.expert_padded_bytes
        assert offset % PAGE_SIZE == 0

        self._fp.seek(offset)
        self._fp.write(payload)
        if self._pad_tail:
            self._fp.write(self._pad_tail)

        entry = ExpertIndexEntry(
            layer_id=layer_id,
            expert_id=expert_id,
            flags=EXPERT_FLAG_TERNARY_SWIGLU,
            offset=offset,
            payload_bytes=self.expert_payload_bytes,
            padded_bytes=self.expert_padded_bytes,
            crc32=crc,
            reserved=0,
        )
        self._entries[(layer_id, expert_id)] = entry
        return entry

    def write_prepacked_expert(
        self,
        layer_id: int,
        expert_id: int,
        gate_packed_bytes: bytes | bytearray | memoryview | np.ndarray,
        up_packed_bytes: bytes | bytearray | memoryview | np.ndarray,
        down_packed_bytes: bytes | bytearray | memoryview | np.ndarray,
        gate_scales: np.ndarray,
        up_scales: np.ndarray,
        down_scales: np.ndarray,
        *,
        validate_crumbs: bool = True,
    ) -> ExpertIndexEntry:
        """Write an already-2bit-packed ternary expert directly to its 4KB-aligned slot."""
        if self._fp is None:
            self.open()
        if self._finalized:
            raise RuntimeError("ExpertsBinWriter is already finalized")
        if not (0 <= layer_id < self.num_layers):
            raise ValueError(f"layer_id {layer_id} out of range 0..{self.num_layers}")
        if not (0 <= expert_id < self.num_experts):
            raise ValueError(f"expert_id {expert_id} out of range 0..{self.num_experts}")

        D = self.hidden_size
        I = self.intermediate_size
        expected_gu_len = I * (D // 4)
        expected_dn_len = D * (I // 4)

        def _as_raw(buf: Any, exp_len: int, label: str) -> bytes:
            arr = np.frombuffer(buf, dtype=np.uint8) if not isinstance(buf, np.ndarray) else buf.reshape(-1).view(np.uint8)
            if arr.size != exp_len:
                raise ValueError(f"{label} byte length {arr.size} != {exp_len}")
            if validate_crumbs and np.any((arr & (arr >> 1)) & 0x55):
                raise ValueError(f"Invalid 0b11 crumb in prepacked {label}")
            return arr.tobytes()

        g_b = _as_raw(gate_packed_bytes, expected_gu_len, "gate_packed")
        u_b = _as_raw(up_packed_bytes, expected_gu_len, "up_packed")
        d_b = _as_raw(down_packed_bytes, expected_dn_len, "down_packed")
        gs_b = np.ascontiguousarray(gate_scales, dtype="<f4").tobytes()
        us_b = np.ascontiguousarray(up_scales, dtype="<f4").tobytes()
        ds_b = np.ascontiguousarray(down_scales, dtype="<f4").tobytes()

        payload = b"".join((g_b, u_b, d_b, gs_b, us_b, ds_b))
        if len(payload) != self.expert_payload_bytes:
            raise AssertionError(
                f"Internal payload size mismatch: {len(payload)} != {self.expert_payload_bytes}"
            )

        crc = crc32_ieee(payload)
        flat_idx = layer_id * self.num_experts + expert_id
        offset = self.data_start_offset + flat_idx * self.expert_padded_bytes
        assert offset % PAGE_SIZE == 0

        self._fp.seek(offset)
        self._fp.write(payload)
        if self._pad_tail:
            self._fp.write(self._pad_tail)

        entry = ExpertIndexEntry(
            layer_id=layer_id,
            expert_id=expert_id,
            flags=EXPERT_FLAG_TERNARY_SWIGLU,
            offset=offset,
            payload_bytes=self.expert_payload_bytes,
            padded_bytes=self.expert_padded_bytes,
            crc32=crc,
            reserved=0,
        )
        self._entries[(layer_id, expert_id)] = entry
        return entry

    def finalize(self) -> tuple[ExpertsHeader, list[ExpertIndexEntry]]:
        if self._fp is None:
            raise RuntimeError("ExpertsBinWriter was never opened")
        if self._finalized:
            raise RuntimeError("ExpertsBinWriter already finalized")

        expected_total = self.num_layers * self.num_experts
        if len(self._entries) != expected_total:
            missing = [
                (l, e)
                for l in range(self.num_layers)
                for e in range(self.num_experts)
                if (l, e) not in self._entries
            ]
            raise ValueError(
                f"Cannot finalize experts.bin: {len(missing)} experts missing "
                f"(first missing: {missing[:5]})"
            )

        ordered_entries: list[ExpertIndexEntry] = []
        index_chunks: list[bytes] = []
        for l in range(self.num_layers):
            for e in range(self.num_experts):
                ent = self._entries[(l, e)]
                ordered_entries.append(ent)
                index_chunks.append(
                    EXPERT_INDEX_ENTRY_STRUCT.pack(
                        ent.layer_id,
                        ent.expert_id,
                        ent.flags,
                        ent.offset,
                        ent.payload_bytes,
                        ent.padded_bytes,
                        ent.crc32,
                        ent.reserved,
                    )
                )

        index_buf = b"".join(index_chunks)
        assert len(index_buf) == self.index_bytes
        index_crc = crc32_ieee(index_buf)

        self._fp.seek(self.index_offset)
        self._fp.write(index_buf)
        # Ensure zero padding between index_offset + index_bytes and data_start_offset.
        pad_after_index = self.data_start_offset - (self.index_offset + self.index_bytes)
        if pad_after_index > 0:
            self._fp.write(b"\x00" * pad_after_index)

        header_prefix = EXPERTS_HEADER_PREFIX_STRUCT.pack(
            EXPERTS_MAGIC,
            FORMAT_VERSION,
            self.num_layers,
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
            self.index_offset,
            self.index_bytes,
            self.data_start_offset,
            index_crc,
            self.expert_payload_bytes,
            self.expert_padded_bytes,
            b"\x00" * 4024,
        )
        header_crc = crc32_ieee(header_prefix)
        self._fp.seek(0)
        self._fp.write(header_prefix)
        self._fp.write(CORE_CRC_STRUCT.pack(header_crc))
        self._fp.flush()

        self._fp.seek(0, 2)
        final_size = self._fp.tell()
        expected_file_size = self.data_start_offset + expected_total * self.expert_padded_bytes
        if final_size != expected_file_size or final_size % PAGE_SIZE != 0:
            raise AssertionError(
                f"experts.bin final size {final_size} != expected {expected_file_size}"
            )

        self._fp.close()
        self._fp = None
        self._finalized = True

        header = ExpertsHeader(
            magic=EXPERTS_MAGIC,
            version=FORMAT_VERSION,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            index_offset=self.index_offset,
            index_bytes=self.index_bytes,
            data_start_offset=self.data_start_offset,
            index_crc32=index_crc,
            expert_payload_bytes=self.expert_payload_bytes,
            expert_padded_bytes=self.expert_padded_bytes,
            header_crc32=header_crc,
        )
        return header, ordered_entries


# ---------------------------------------------------------------------------
# Binary Readers & Validators (Sections 2 & 3)
# ---------------------------------------------------------------------------


def read_core_header_and_dir(
    path: Union[str, Path],
    *,
    verify_crc: bool = True,
) -> tuple[CoreHeader, list[CoreTensorEntry]]:
    """Read and strictly validate ``core.bin`` header, tensor directory, alignments, and CRCs."""
    p = Path(path)
    file_size = p.stat().st_size
    if file_size < CORE_HEADER_BYTES:
        raise ValueError(f"core.bin too small ({file_size} bytes < {CORE_HEADER_BYTES})")

    with p.open("rb") as f:
        raw_header = f.read(CORE_HEADER_BYTES)
        prefix = raw_header[:252]
        (stored_header_crc,) = CORE_CRC_STRUCT.unpack(raw_header[252:256])
        calc_header_crc = crc32_ieee(prefix)
        if stored_header_crc != calc_header_crc:
            raise ValueError(
                f"core.bin header_crc32 mismatch: stored=0x{stored_header_crc:08X}, "
                f"computed=0x{calc_header_crc:08X}"
            )

        (
            magic,
            version,
            flags,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            num_experts,
            top_k,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            lookahead_offset,
            layer_kinds_raw,
            tensor_count,
            data_section_offset,
            reserved,
        ) = CORE_HEADER_PREFIX_STRUCT.unpack(prefix)

        if magic != CORE_MAGIC:
            raise ValueError(f"Invalid core.bin magic: {magic!r} != {CORE_MAGIC!r}")
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported core.bin version: {version}")
        if reserved != b"\x00" * 112:
            raise ValueError("core.bin header reserved bytes (140..252) must be zero-filled")
        if not (1 <= num_layers <= 64):
            raise ValueError(f"Invalid num_layers in core.bin: {num_layers}")
        if layer_kinds_raw[num_layers:] != b"\x00" * (64 - num_layers):
            raise ValueError("Unused trailing layer_kinds bytes must be zero")

        layer_kinds = tuple(int(b) for b in layer_kinds_raw[:num_layers])
        for lk in layer_kinds:
            if lk not in (LAYER_KIND_ANCHOR_GQA, LAYER_KIND_GATED_LINEAR_RECURRENCE):
                raise ValueError(f"Invalid layer_kind {lk} in core.bin")

        dir_end = CORE_HEADER_BYTES + tensor_count * CORE_ENTRY_BYTES
        if data_section_offset < dir_end or data_section_offset % CORE_ALIGNMENT != 0:
            raise ValueError(
                f"Invalid data_section_offset={data_section_offset} (dir_end={dir_end})"
            )
        if file_size < data_section_offset:
            raise ValueError("core.bin truncated before data_section_offset")

        raw_dir = f.read(tensor_count * CORE_ENTRY_BYTES)
        pad_between = f.read(data_section_offset - dir_end)
        if pad_between != b"\x00" * len(pad_between):
            raise ValueError("Padding between core.bin directory and data section must be zero")

        entries: list[CoreTensorEntry] = []
        for idx in range(tensor_count):
            chunk = raw_dir[idx * CORE_ENTRY_BYTES : (idx + 1) * CORE_ENTRY_BYTES]
            (
                raw_name,
                dtype,
                ndim,
                entry_flags,
                d0,
                d1,
                d2,
                d3,
                data_offset,
                data_bytes,
                scale_offset,
                scale_bytes,
                data_crc32,
                scale_crc32,
                entry_reserved,
            ) = CORE_TENSOR_ENTRY_STRUCT.unpack(chunk)

            if b"\x00" not in raw_name:
                raise ValueError(f"CoreTensorEntry #{idx} name is not null-terminated")
            nul_pos = raw_name.index(b"\x00")
            if raw_name[nul_pos:] != b"\x00" * (64 - nul_pos):
                raise ValueError(f"CoreTensorEntry #{idx} has non-zero bytes after NUL in name")
            name = raw_name[:nul_pos].decode("utf-8")

            if entry_flags != 0 or entry_reserved != 0:
                raise ValueError(f"CoreTensorEntry {name!r} reserved/flags must be 0")
            if data_offset < data_section_offset or data_offset % CORE_ALIGNMENT != 0:
                raise ValueError(
                    f"CoreTensorEntry {name!r} data_offset={data_offset} not 64B-aligned"
                )
            if data_offset + data_bytes > file_size:
                raise ValueError(f"CoreTensorEntry {name!r} data buffer exceeds file size")

            if dtype == DTYPE_TERNARY_2BIT:
                expected_data = d0 * (d1 // 4)
                expected_scale = d0 * 4
                if ndim != 2 or d1 % 4 != 0:
                    raise ValueError(f"Invalid Ternary2Bit shape for {name!r}: {(d0, d1, d2, d3)}")
                if data_bytes != expected_data or scale_bytes != expected_scale:
                    raise ValueError(
                        f"Ternary2Bit byte count mismatch on {name!r}: "
                        f"data={data_bytes} (exp {expected_data}), "
                        f"scale={scale_bytes} (exp {expected_scale})"
                    )
                if scale_offset < data_section_offset or scale_offset + scale_bytes > file_size:
                    raise ValueError(f"CoreTensorEntry {name!r} scale buffer out of bounds")
            else:
                if scale_offset != 0 or scale_bytes != 0 or scale_crc32 != 0:
                    raise ValueError(
                        f"Non-ternary tensor {name!r} must have zero scale_offset/bytes/crc32"
                    )

            if verify_crc:
                f.seek(data_offset)
                calc_data_crc = crc32_ieee(f.read(data_bytes))
                if calc_data_crc != data_crc32:
                    raise ValueError(
                        f"core.bin data_crc32 mismatch for {name!r}: "
                        f"stored=0x{data_crc32:08X}, computed=0x{calc_data_crc:08X}"
                    )
                if dtype == DTYPE_TERNARY_2BIT:
                    f.seek(scale_offset)
                    calc_scale_crc = crc32_ieee(f.read(scale_bytes))
                    if calc_scale_crc != scale_crc32:
                        raise ValueError(
                            f"core.bin scale_crc32 mismatch for {name!r}: "
                            f"stored=0x{scale_crc32:08X}, computed=0x{calc_scale_crc:08X}"
                        )

            entries.append(
                CoreTensorEntry(
                    name=name,
                    dtype=dtype,
                    ndim=ndim,
                    flags=entry_flags,
                    shape=(d0, d1, d2, d3),
                    data_offset=data_offset,
                    data_bytes=data_bytes,
                    scale_offset=scale_offset,
                    scale_bytes=scale_bytes,
                    data_crc32=data_crc32,
                    scale_crc32=scale_crc32,
                    reserved=entry_reserved,
                )
            )

    header = CoreHeader(
        magic=magic,
        version=version,
        flags=flags,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        top_k=top_k,
        max_position_embeddings=max_position_embeddings,
        rope_theta=float(rope_theta),
        rms_norm_eps=float(rms_norm_eps),
        lookahead_offset=lookahead_offset,
        layer_kinds=layer_kinds,
        tensor_count=tensor_count,
        data_section_offset=data_section_offset,
        header_crc32=stored_header_crc,
    )
    return header, entries


def read_core_tensor(
    path: Union[str, Path],
    entry: CoreTensorEntry,
    *,
    verify_crc: bool = True,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Read and unpack a single tensor from ``core.bin``.

    Returns ``(w_ternary_int8, scales_f32)`` if ``entry.dtype == DTYPE_TERNARY_2BIT``,
    otherwise returns a C-contiguous NumPy array with ``entry.logical_shape``.
    """
    with Path(path).open("rb") as f:
        f.seek(entry.data_offset)
        data_buf = f.read(entry.data_bytes)
        if verify_crc and crc32_ieee(data_buf) != entry.data_crc32:
            raise ValueError(f"CRC32 mismatch on core tensor {entry.name!r}")

        if entry.dtype == DTYPE_TERNARY_2BIT:
            rows, cols = entry.shape[0], entry.shape[1]
            w_tern = unpack_ternary_2bit(data_buf, rows, cols)
            f.seek(entry.scale_offset)
            scale_buf = f.read(entry.scale_bytes)
            if verify_crc and crc32_ieee(scale_buf) != entry.scale_crc32:
                raise ValueError(f"Scale CRC32 mismatch on core tensor {entry.name!r}")
            scales = np.frombuffer(scale_buf, dtype="<f4").astype(np.float32, copy=True)
            return w_tern, scales

        if entry.dtype == DTYPE_F32:
            arr = np.frombuffer(data_buf, dtype="<f4").astype(np.float32, copy=True)
        elif entry.dtype == DTYPE_F16:
            arr = np.frombuffer(data_buf, dtype="<f2").astype(np.float16, copy=True)
        elif entry.dtype == DTYPE_I8:
            arr = np.frombuffer(data_buf, dtype=np.int8).copy()
        else:
            raise ValueError(f"Unknown CoreTensorEntry dtype: {entry.dtype}")

        return arr.reshape(entry.logical_shape)


def read_experts_header_and_dir(
    path: Union[str, Path],
    *,
    verify_crc: bool = True,
) -> tuple[ExpertsHeader, list[ExpertIndexEntry]]:
    """Read and strictly validate ``experts.bin`` Page 0 header, 4KB alignments, and CRCs."""
    p = Path(path)
    file_size = p.stat().st_size
    if file_size < EXPERTS_HEADER_BYTES:
        raise ValueError(f"experts.bin too small ({file_size} bytes < {EXPERTS_HEADER_BYTES})")
    if file_size % PAGE_SIZE != 0:
        raise ValueError(
            f"experts.bin file_size ({file_size}) is not a multiple of {PAGE_SIZE}"
        )

    with p.open("rb") as f:
        raw_header = f.read(EXPERTS_HEADER_BYTES)
        prefix = raw_header[:4092]
        (stored_header_crc,) = CORE_CRC_STRUCT.unpack(raw_header[4092:4096])
        calc_header_crc = crc32_ieee(prefix)
        if stored_header_crc != calc_header_crc:
            raise ValueError(
                f"experts.bin header_crc32 mismatch: stored=0x{stored_header_crc:08X}, "
                f"computed=0x{calc_header_crc:08X}"
            )

        (
            magic,
            version,
            num_layers,
            num_experts,
            hidden_size,
            intermediate_size,
            index_offset,
            index_bytes,
            data_start_offset,
            index_crc32,
            expert_payload_bytes,
            expert_padded_bytes,
            reserved,
        ) = EXPERTS_HEADER_PREFIX_STRUCT.unpack(prefix)

        if magic != EXPERTS_MAGIC:
            raise ValueError(f"Invalid experts.bin magic: {magic!r} != {EXPERTS_MAGIC!r}")
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported experts.bin version: {version}")
        if reserved != b"\x00" * 4024:
            raise ValueError("experts.bin header reserved bytes (68..4092) must be zero-filled")
        if index_offset != EXPERT_INDEX_OFFSET:
            raise ValueError(f"experts.bin index_offset must be 4096, got {index_offset}")

        total_experts = num_layers * num_experts
        expected_index_bytes = total_experts * EXPERT_ENTRY_BYTES
        if index_bytes != expected_index_bytes:
            raise ValueError(
                f"experts.bin index_bytes={index_bytes} != expected {expected_index_bytes}"
            )
        expected_data_start = align_up(EXPERT_INDEX_OFFSET + expected_index_bytes, PAGE_SIZE)
        if data_start_offset != expected_data_start or data_start_offset % PAGE_SIZE != 0:
            raise ValueError(
                f"experts.bin data_start_offset={data_start_offset} != expected {expected_data_start}"
            )

        expected_payload = compute_expert_payload_bytes(hidden_size, intermediate_size)
        expected_padded = align_up(expected_payload, PAGE_SIZE)
        if expert_payload_bytes != expected_payload or expert_padded_bytes != expected_padded:
            raise ValueError(
                f"Expert block size mismatch: payload={expert_payload_bytes} (exp {expected_payload}), "
                f"padded={expert_padded_bytes} (exp {expected_padded})"
            )

        expected_file_size = data_start_offset + total_experts * expert_padded_bytes
        if file_size != expected_file_size:
            raise ValueError(
                f"experts.bin file_size={file_size} != expected {expected_file_size}"
            )

        f.seek(index_offset)
        raw_index = f.read(index_bytes)
        calc_index_crc = crc32_ieee(raw_index)
        if calc_index_crc != index_crc32:
            raise ValueError(
                f"experts.bin index_crc32 mismatch: stored=0x{index_crc32:08X}, "
                f"computed=0x{calc_index_crc:08X}"
            )

        index_pad = f.read(data_start_offset - (index_offset + index_bytes))
        if index_pad != b"\x00" * len(index_pad):
            raise ValueError(
                "Bytes between 4096 + index_bytes and data_start_offset must be 0x00"
            )

        entries: list[ExpertIndexEntry] = []
        for flat_idx in range(total_experts):
            exp_layer = flat_idx // num_experts
            exp_id = flat_idx % num_experts
            chunk = raw_index[
                flat_idx * EXPERT_ENTRY_BYTES : (flat_idx + 1) * EXPERT_ENTRY_BYTES
            ]
            (
                layer_id,
                expert_id,
                flags,
                offset,
                payload_bytes,
                padded_bytes,
                crc32_val,
                ent_reserved,
            ) = EXPERT_INDEX_ENTRY_STRUCT.unpack(chunk)

            if (layer_id, expert_id) != (exp_layer, exp_id):
                raise ValueError(
                    f"ExpertIndexEntry #{flat_idx} out of row-major order: "
                    f"got ({layer_id}, {expert_id}), expected ({exp_layer}, {exp_id})"
                )
            if flags != EXPERT_FLAG_TERNARY_SWIGLU:
                raise ValueError(f"Unsupported expert flags={flags} at ({layer_id}, {expert_id})")
            if ent_reserved != 0:
                raise ValueError(f"Non-zero reserved field in ExpertIndexEntry ({layer_id}, {expert_id})")
            if offset % PAGE_SIZE != 0 or padded_bytes % PAGE_SIZE != 0:
                raise ValueError(
                    f"Expert ({layer_id}, {expert_id}) violates 4KB alignment: "
                    f"offset={offset}, padded_bytes={padded_bytes}"
                )
            expected_offset = data_start_offset + flat_idx * expert_padded_bytes
            if offset != expected_offset:
                raise ValueError(
                    f"Expert ({layer_id}, {expert_id}) offset={offset} != expected {expected_offset}"
                )
            if payload_bytes != expert_payload_bytes or padded_bytes != expert_padded_bytes:
                raise ValueError(f"Expert ({layer_id}, {expert_id}) payload/padded size mismatch")

            if verify_crc:
                f.seek(offset)
                block = f.read(padded_bytes)
                payload = block[:payload_bytes]
                padding = block[payload_bytes:]
                calc_crc = crc32_ieee(payload)
                if calc_crc != crc32_val:
                    raise ValueError(
                        f"Expert ({layer_id}, {expert_id}) crc32 mismatch: "
                        f"stored=0x{crc32_val:08X}, computed=0x{calc_crc:08X}"
                    )
                if padding != b"\x00" * len(padding):
                    raise ValueError(
                        f"Expert ({layer_id}, {expert_id}) trailing padding is not zero-filled"
                    )

            entries.append(
                ExpertIndexEntry(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    flags=flags,
                    offset=offset,
                    payload_bytes=payload_bytes,
                    padded_bytes=padded_bytes,
                    crc32=crc32_val,
                    reserved=ent_reserved,
                )
            )

    header = ExpertsHeader(
        magic=magic,
        version=version,
        num_layers=num_layers,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        index_offset=index_offset,
        index_bytes=index_bytes,
        data_start_offset=data_start_offset,
        index_crc32=index_crc32,
        expert_payload_bytes=expert_payload_bytes,
        expert_padded_bytes=expert_padded_bytes,
        header_crc32=stored_header_crc,
    )
    return header, entries


def read_expert_weights(
    path: Union[str, Path],
    header: ExpertsHeader,
    entry: ExpertIndexEntry,
    *,
    verify_crc: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read and unpack ``gate_proj``, ``up_proj``, and ``down_proj`` for one expert."""
    D = header.hidden_size
    I = header.intermediate_size
    gate_pack_len = I * (D // 4)
    up_pack_len = I * (D // 4)
    down_pack_len = D * (I // 4)
    gate_scale_len = I * 4
    up_scale_len = I * 4
    down_scale_len = D * 4

    with Path(path).open("rb") as f:
        f.seek(entry.offset)
        block = f.read(entry.padded_bytes)

    payload = block[: entry.payload_bytes]
    padding = block[entry.payload_bytes :]
    if verify_crc:
        if crc32_ieee(payload) != entry.crc32:
            raise ValueError(
                f"CRC32 mismatch reading expert ({entry.layer_id}, {entry.expert_id})"
            )
        if padding != b"\x00" * len(padding):
            raise ValueError(
                f"Non-zero padding in expert ({entry.layer_id}, {entry.expert_id})"
            )

    pos = 0
    gate_packed = payload[pos : pos + gate_pack_len]
    pos += gate_pack_len
    up_packed = payload[pos : pos + up_pack_len]
    pos += up_pack_len
    down_packed = payload[pos : pos + down_pack_len]
    pos += down_pack_len
    gate_scales = np.frombuffer(payload[pos : pos + gate_scale_len], dtype="<f4").astype(
        np.float32, copy=True
    )
    pos += gate_scale_len
    up_scales = np.frombuffer(payload[pos : pos + up_scale_len], dtype="<f4").astype(
        np.float32, copy=True
    )
    pos += up_scale_len
    down_scales = np.frombuffer(payload[pos : pos + down_scale_len], dtype="<f4").astype(
        np.float32, copy=True
    )
    pos += down_scale_len
    assert pos == entry.payload_bytes

    return {
        "gate_proj": (unpack_ternary_2bit(gate_packed, I, D), gate_scales),
        "up_proj": (unpack_ternary_2bit(up_packed, I, D), up_scales),
        "down_proj": (unpack_ternary_2bit(down_packed, D, I), down_scales),
    }


# ---------------------------------------------------------------------------
# Streaming Tensor Sources (Safetensors Shards & In-Memory State Dict)
# ---------------------------------------------------------------------------


class _TensorArchive:
    """Abstract read-only interface over either safetensors shards or a state_dict."""

    def keys(self) -> Iterable[str]:
        raise NotImplementedError

    def has(self, key: str) -> bool:
        raise NotImplementedError

    def get(self, key: str) -> torch.Tensor:
        raise NotImplementedError

    def get_slice_row(self, key: str, idx: int) -> torch.Tensor:
        return self.get(key)[idx]

    def close(self) -> None:
        pass


class _StateDictArchive(_TensorArchive):
    def __init__(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self._sd = state_dict

    def keys(self) -> Iterable[str]:
        return self._sd.keys()

    def has(self, key: str) -> bool:
        return key in self._sd

    def get(self, key: str) -> torch.Tensor:
        return self._sd[key].detach().cpu()

    def get_slice_row(self, key: str, idx: int) -> torch.Tensor:
        return self._sd[key][idx].detach().cpu()


class _SafetensorsShardArchive(_TensorArchive):
    """Lazily streams tensors from single or multi-shard safetensors with a tiny handle LRU."""

    def __init__(self, model_dir: Path, max_open_shards: int = 2) -> None:
        self.model_dir = model_dir
        self.max_open_shards = max(1, max_open_shards)
        self._key_to_shard: dict[str, Path] = {}
        self._handles: OrderedDict[Path, Any] = OrderedDict()

        index_file = model_dir / "model.safetensors.index.json"
        if index_file.is_file():
            index_data = json.loads(index_file.read_text(encoding="utf-8"))
            weight_map: dict[str, str] = index_data.get("weight_map", {})
            for tensor_name, rel_shard in weight_map.items():
                shard_path = model_dir / rel_shard
                if not shard_path.is_file():
                    raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
                self._key_to_shard[tensor_name] = shard_path
        else:
            shard_files = sorted(model_dir.glob("*.safetensors"))
            if not shard_files:
                raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
            for shard_path in shard_files:
                with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        self._key_to_shard[k] = shard_path

    def _open_shard(self, shard_path: Path) -> Any:
        if shard_path in self._handles:
            self._handles.move_to_end(shard_path)
            return self._handles[shard_path]
        while len(self._handles) >= self.max_open_shards:
            _, (old_ctx, _) = self._handles.popitem(last=False)
            old_ctx.__exit__(None, None, None)
        ctx = safe_open(str(shard_path), framework="pt", device="cpu")
        handle = ctx.__enter__()
        # Keep context manager paired for clean __exit__ on eviction/close.
        self._handles[shard_path] = (ctx, handle)
        return (ctx, handle)

    def keys(self) -> Iterable[str]:
        return self._key_to_shard.keys()

    def has(self, key: str) -> bool:
        return key in self._key_to_shard

    def get(self, key: str) -> torch.Tensor:
        shard_path = self._key_to_shard[key]
        _, handle = self._open_shard(shard_path)
        return handle.get_tensor(key)

    def get_slice_row(self, key: str, idx: int) -> torch.Tensor:
        shard_path = self._key_to_shard[key]
        _, handle = self._open_shard(shard_path)
        return handle.get_slice(key)[idx]

    def close(self) -> None:
        while self._handles:
            _, (ctx, _) = self._handles.popitem(last=False)
            ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Checkpoint Normalization & Core/Experts Export Logic
# ---------------------------------------------------------------------------


def _first_present(archive: _TensorArchive, candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if archive.has(c):
            return c
    return None


def _parse_dtype_name(dtype_str: str, *, allow_ternary: bool = False) -> int:
    norm = dtype_str.strip().lower()
    if norm in ("f32", "float32", "fp32"):
        return DTYPE_F32
    if norm in ("f16", "float16", "fp16", "half"):
        return DTYPE_F16
    if allow_ternary and norm in ("ternary", "ternary2bit", "2bit"):
        return DTYPE_TERNARY_2BIT
    raise ValueError(f"Unsupported dtype option: {dtype_str!r}")


def _extract_rope_theta(config_dict: Mapping[str, Any]) -> float:
    if config_dict.get("rope_theta") is not None:
        return float(config_dict["rope_theta"])
    rope_params = config_dict.get("rope_parameters")
    if isinstance(rope_params, Mapping) and rope_params.get("rope_theta") is not None:
        return float(rope_params["rope_theta"])
    rope_scaling = config_dict.get("rope_scaling")
    if isinstance(rope_scaling, Mapping) and rope_scaling.get("rope_theta") is not None:
        return float(rope_scaling["rope_theta"])
    return 10000.0


def _build_terra_config(
    config_dict: Mapping[str, Any],
    archive: _TensorArchive,
    *,
    bitlinear_rms_prescale: bool | None = None,
    norm_topk_prob: bool | None = None,
    tie_word_embeddings: bool | None = None,
) -> TerraModelConfig:
    vocab_size = int(config_dict["vocab_size"])
    hidden_size = int(config_dict["hidden_size"])
    intermediate_size = int(config_dict["intermediate_size"])
    num_layers_raw = config_dict.get("num_layers", config_dict.get("num_hidden_layers"))
    if num_layers_raw is None:
        raise ValueError("config_dict missing num_layers / num_hidden_layers")
    num_layers = int(num_layers_raw)

    num_attention_heads = int(config_dict["num_attention_heads"])
    num_kv_heads = int(
        config_dict.get(
            "num_kv_heads",
            config_dict.get("num_key_value_heads", num_attention_heads),
        )
    )
    head_dim = int(config_dict.get("head_dim") or (hidden_size // num_attention_heads))

    num_experts_raw = config_dict.get("num_experts", config_dict.get("num_local_experts"))
    if num_experts_raw is None:
        raise ValueError("config_dict missing num_experts / num_local_experts")
    num_experts = int(num_experts_raw)

    top_k_raw = config_dict.get("top_k", config_dict.get("num_experts_per_tok"))
    if top_k_raw is None:
        raise ValueError("config_dict missing top_k / num_experts_per_tok")
    top_k = int(top_k_raw)

    max_pos = int(config_dict.get("max_position_embeddings", 4096))
    rope_theta = _extract_rope_theta(config_dict)
    rms_norm_eps = float(config_dict.get("rms_norm_eps", 1e-5))
    lookahead_offset = int(config_dict.get("lookahead_offset", 3))

    raw_layer_kinds = config_dict.get("layer_kinds")
    if raw_layer_kinds is not None:
        parsed_kinds: list[int] = []
        for item in raw_layer_kinds:
            if isinstance(item, str):
                low = item.strip().lower()
                if low in ("0", "anchorgqa", "anchor_gqa", "gqa", "attention"):
                    parsed_kinds.append(LAYER_KIND_ANCHOR_GQA)
                elif low in (
                    "1",
                    "gatedlinearrecurrence",
                    "gated_linear_recurrence",
                    "recurrence",
                    "ssm",
                ):
                    parsed_kinds.append(LAYER_KIND_GATED_LINEAR_RECURRENCE)
                else:
                    raise ValueError(f"Unrecognized layer_kind string: {item!r}")
            else:
                parsed_kinds.append(int(item))
        layer_kinds = tuple(parsed_kinds)
    else:
        detected: list[int] = []
        for l in range(num_layers):
            decay_key = _first_present(
                archive,
                (
                    f"model.layers.{l}.self_attn.decay_proj.weight",
                    f"layers.{l}.self_attn.decay_proj.weight",
                    f"model.layers.{l}.recurrence.decay_proj.weight",
                    f"layers.{l}.recurrence.decay_proj.weight",
                ),
            )
            detected.append(
                LAYER_KIND_GATED_LINEAR_RECURRENCE
                if decay_key is not None
                else LAYER_KIND_ANCHOR_GQA
            )
        layer_kinds = tuple(detected)

    rms_pre = (
        bool(bitlinear_rms_prescale)
        if bitlinear_rms_prescale is not None
        else bool(config_dict.get("bitlinear_rms_prescale", False))
    )
    norm_topk = (
        bool(norm_topk_prob)
        if norm_topk_prob is not None
        else bool(config_dict.get("norm_topk_prob", False))
    )
    tie_emb = (
        bool(tie_word_embeddings)
        if tie_word_embeddings is not None
        else bool(config_dict.get("tie_word_embeddings", False))
    )

    hint_table_key = _first_present(
        archive,
        ("model.hint_embedding.table.weight", "hint_embedding.table.weight"),
    )
    hint_proj_key = _first_present(
        archive,
        ("model.hint_embedding.proj.weight", "hint_embedding.proj.weight"),
    )
    if (hint_table_key is None) != (hint_proj_key is None):
        raise ValueError(
            "Both hint_embedding.table.weight and hint_embedding.proj.weight must be present together"
        )
    has_hint = hint_table_key is not None and hint_proj_key is not None

    flags = 0
    if rms_pre:
        flags |= FLAG_BITLINEAR_RMS_PRESCALE
    if norm_topk:
        flags |= FLAG_NORM_TOPK_PROB
    if tie_emb:
        flags |= FLAG_TIE_WORD_EMBEDDINGS
    if has_hint:
        flags |= FLAG_HAS_HINT_EMBEDDING

    return TerraModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        top_k=top_k,
        max_position_embeddings=max_pos,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        lookahead_offset=lookahead_offset,
        layer_kinds=layer_kinds,
        flags=flags,
    )


def _export_from_archive(
    config_dict: Mapping[str, Any],
    archive: _TensorArchive,
    out_dir: Union[str, Path],
    *,
    bitlinear_rms_prescale: bool | None = None,
    norm_topk_prob: bool | None = None,
    tie_word_embeddings: bool | None = None,
    embed_dtype: str = "f32",
    lm_head_dtype: str = "f32",
    attn_dtype: str = "ternary",
    expert_layout: str = "out_in",
) -> dict[str, Any]:
    if expert_layout not in ("out_in", "in_out"):
        raise ValueError(
            f"expert_layout must be 'out_in' ([out, in], HF default) or 'in_out' ([in, out]), got {expert_layout!r}"
        )
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_terra_config(
        config_dict,
        archive,
        bitlinear_rms_prescale=bitlinear_rms_prescale,
        norm_topk_prob=norm_topk_prob,
        tie_word_embeddings=tie_word_embeddings,
    )

    emb_dt = _parse_dtype_name(embed_dtype, allow_ternary=False)
    lm_dt = _parse_dtype_name(lm_head_dtype, allow_ternary=True)
    attn_dt = _parse_dtype_name(attn_dtype, allow_ternary=True)

    core_writer = CoreBinWriter(cfg)

    def req_key(canonical: str, candidates: Sequence[str]) -> str:
        found = _first_present(archive, candidates)
        if found is None:
            raise KeyError(
                f"Required tensor for {canonical!r} not found (tried {list(candidates)})"
            )
        return found

    def add_checked_core_tensor(
        canonical: str,
        archive_key: str,
        expected_shape: tuple[int, ...],
        dtype: int,
    ) -> None:
        def _load(k: str = archive_key, exp: tuple[int, ...] = expected_shape, name: str = canonical) -> torch.Tensor:
            t = archive.get(k)
            if tuple(t.shape) != exp:
                raise ValueError(
                    f"Shape mismatch exporting {name!r} (from {k!r}): expected {exp}, got {tuple(t.shape)}"
                )
            return t

        core_writer.add_tensor(canonical, _load, dtype)

    D = cfg.hidden_size
    I = cfg.intermediate_size
    H = cfg.num_attention_heads
    H_kv = cfg.num_kv_heads
    d = cfg.head_dim
    E = cfg.num_experts

    # 1. Global embeddings & final norm
    emb_key = req_key(
        "model.embed_tokens.weight",
        ("model.embed_tokens.weight", "embed_tokens.weight"),
    )
    add_checked_core_tensor("model.embed_tokens.weight", emb_key, (cfg.vocab_size, D), emb_dt)

    norm_key = req_key("model.norm.weight", ("model.norm.weight", "norm.weight"))
    add_checked_core_tensor("model.norm.weight", norm_key, (D,), DTYPE_F32)

    if not (cfg.flags & FLAG_TIE_WORD_EMBEDDINGS):
        lm_key = req_key("lm_head.weight", ("lm_head.weight", "model.lm_head.weight"))
        add_checked_core_tensor("lm_head.weight", lm_key, (cfg.vocab_size, D), lm_dt)
    else:
        unused_lm = _first_present(archive, ("lm_head.weight", "model.lm_head.weight"))
        if unused_lm is not None:
            lm_t = archive.get(unused_lm)
            emb_t = archive.get(emb_key)
            if lm_t.shape != emb_t.shape or not torch.equal(lm_t, emb_t):
                warnings.warn(
                    f"tie_word_embeddings=True omits {unused_lm!r}, which differs from {emb_key!r}; "
                    "pass tie_word_embeddings=False if lm_head is untied.",
                    UserWarning,
                    stacklevel=2,
                )
            del lm_t, emb_t

    attn_expected_shapes = {
        "q_proj": (H * d, D),
        "k_proj": (H_kv * d, D),
        "v_proj": (H_kv * d, D),
        "o_proj": (D, H * d),
    }
    norm_expected_shapes = {
        "q_norm": (H * d,),
        "k_norm": (H_kv * d,),
    }

    # 2. Per-layer norms, attention/recurrence weights, ground-truth & lookahead routers
    for l in range(cfg.num_layers):
        in_ln = req_key(
            f"layers.{l}.input_layernorm.weight",
            (
                f"model.layers.{l}.input_layernorm.weight",
                f"layers.{l}.input_layernorm.weight",
            ),
        )
        add_checked_core_tensor(f"layers.{l}.input_layernorm.weight", in_ln, (D,), DTYPE_F32)

        post_ln = req_key(
            f"layers.{l}.post_attention_layernorm.weight",
            (
                f"model.layers.{l}.post_attention_layernorm.weight",
                f"layers.{l}.post_attention_layernorm.weight",
            ),
        )
        add_checked_core_tensor(
            f"layers.{l}.post_attention_layernorm.weight",
            post_ln,
            (D,),
            DTYPE_F32,
        )

        for proj_name, exp_shape in attn_expected_shapes.items():
            pkey = req_key(
                f"layers.{l}.self_attn.{proj_name}.weight",
                (
                    f"model.layers.{l}.self_attn.{proj_name}.weight",
                    f"layers.{l}.self_attn.{proj_name}.weight",
                    f"model.layers.{l}.recurrence.{proj_name}.weight",
                    f"layers.{l}.recurrence.{proj_name}.weight",
                ),
            )
            add_checked_core_tensor(
                f"layers.{l}.self_attn.{proj_name}.weight",
                pkey,
                exp_shape,
                attn_dt,
            )

        for norm_name, exp_shape in norm_expected_shapes.items():
            nkey = req_key(
                f"layers.{l}.self_attn.{norm_name}.weight",
                (
                    f"model.layers.{l}.self_attn.{norm_name}.weight",
                    f"layers.{l}.self_attn.{norm_name}.weight",
                    f"model.layers.{l}.recurrence.{norm_name}.weight",
                    f"layers.{l}.recurrence.{norm_name}.weight",
                ),
            )
            add_checked_core_tensor(
                f"layers.{l}.self_attn.{norm_name}.weight",
                nkey,
                exp_shape,
                DTYPE_F32,
            )

        if cfg.layer_kinds[l] == LAYER_KIND_GATED_LINEAR_RECURRENCE:
            dw_key = req_key(
                f"layers.{l}.self_attn.decay_proj.weight",
                (
                    f"model.layers.{l}.self_attn.decay_proj.weight",
                    f"layers.{l}.self_attn.decay_proj.weight",
                    f"model.layers.{l}.recurrence.decay_proj.weight",
                    f"layers.{l}.recurrence.decay_proj.weight",
                ),
            )
            db_key = req_key(
                f"layers.{l}.self_attn.decay_proj.bias",
                (
                    f"model.layers.{l}.self_attn.decay_proj.bias",
                    f"layers.{l}.self_attn.decay_proj.bias",
                    f"model.layers.{l}.recurrence.decay_proj.bias",
                    f"layers.{l}.recurrence.decay_proj.bias",
                ),
            )
            add_checked_core_tensor(
                f"layers.{l}.self_attn.decay_proj.weight",
                dw_key,
                (H, D),
                DTYPE_F32,
            )
            add_checked_core_tensor(
                f"layers.{l}.self_attn.decay_proj.bias",
                db_key,
                (H,),
                DTYPE_F32,
            )

        gate_key = req_key(
            f"layers.{l}.mlp.gate.weight",
            (
                f"model.layers.{l}.mlp.gate.weight",
                f"layers.{l}.mlp.gate.weight",
            ),
        )
        add_checked_core_tensor(f"layers.{l}.mlp.gate.weight", gate_key, (E, D), DTYPE_F32)

        look_key = _first_present(
            archive,
            (
                f"model.layers.{l}.lookahead_router.weight",
                f"layers.{l}.lookahead_router.weight",
                f"model.layers.{l}.mlp.lookahead_router.weight",
                f"layers.{l}.mlp.lookahead_router.weight",
                f"model.lookahead_routers.{l}.weight",
                f"lookahead_routers.{l}.weight",
            ),
        )
        if look_key is not None:
            if l + cfg.lookahead_offset < cfg.num_layers:
                add_checked_core_tensor(
                    f"layers.{l}.lookahead_router.weight",
                    look_key,
                    (E, D),
                    DTYPE_F32,
                )
            else:
                warnings.warn(
                    f"Skipping out-of-range lookahead router {look_key!r} at layer {l} "
                    f"(layer + lookahead_offset={l + cfg.lookahead_offset} >= num_layers={cfg.num_layers})",
                    UserWarning,
                    stacklevel=2,
                )

    # 3. Optional HintEmbedding tables (Section 2.3)
    if cfg.flags & FLAG_HAS_HINT_EMBEDDING:
        ht_key = req_key(
            "model.hint_embedding.table.weight",
            ("model.hint_embedding.table.weight", "hint_embedding.table.weight"),
        )
        hp_key = req_key(
            "model.hint_embedding.proj.weight",
            ("model.hint_embedding.proj.weight", "hint_embedding.proj.weight"),
        )
        ht_sample = archive.get(ht_key)
        if (
            ht_sample.ndim != 2
            or ht_sample.shape[0] < 4
            or ht_sample.shape[0] % 4 != 0
            or ht_sample.shape[1] <= 0
        ):
            raise ValueError(
                f"Invalid hint_embedding.table.weight shape {tuple(ht_sample.shape)}; "
                "expected 2D [num_buckets, hint_dim] with num_buckets >= 4 and divisible by 4"
            )
        num_buckets, hint_dim = int(ht_sample.shape[0]), int(ht_sample.shape[1])
        del ht_sample
        add_checked_core_tensor(
            "model.hint_embedding.table.weight",
            ht_key,
            (num_buckets, hint_dim),
            DTYPE_F32,
        )
        add_checked_core_tensor(
            "model.hint_embedding.proj.weight",
            hp_key,
            (D, 4 * hint_dim),
            DTYPE_F32,
        )

    core_path = target_dir / "core.bin"
    core_header, core_entries = core_writer.write(core_path)

    # 4. Stream all (layer_id, expert_id) blocks into experts.bin
    experts_path = target_dir / "experts.bin"

    with ExpertsBinWriter(
        experts_path,
        num_layers=cfg.num_layers,
        num_experts=cfg.num_experts,
        hidden_size=D,
        intermediate_size=I,
    ) as exp_writer:
        for l in range(cfg.num_layers):
            fused_gate_up_key = _first_present(
                archive,
                (
                    f"model.layers.{l}.mlp.experts.gate_up_proj",
                    f"model.layers.{l}.mlp.experts.gate_up_proj.weight",
                    f"layers.{l}.mlp.experts.gate_up_proj",
                    f"layers.{l}.mlp.experts.gate_up_proj.weight",
                ),
            )
            fused_down_key = _first_present(
                archive,
                (
                    f"model.layers.{l}.mlp.experts.down_proj",
                    f"model.layers.{l}.mlp.experts.down_proj.weight",
                    f"layers.{l}.mlp.experts.down_proj",
                    f"layers.{l}.mlp.experts.down_proj.weight",
                ),
            )
            batched_gate_key = _first_present(
                archive,
                (
                    f"model.layers.{l}.mlp.experts.gate_proj",
                    f"model.layers.{l}.mlp.experts.gate_proj.weight",
                    f"layers.{l}.mlp.experts.gate_proj",
                    f"layers.{l}.mlp.experts.gate_proj.weight",
                ),
            )
            batched_up_key = _first_present(
                archive,
                (
                    f"model.layers.{l}.mlp.experts.up_proj",
                    f"model.layers.{l}.mlp.experts.up_proj.weight",
                    f"layers.{l}.mlp.experts.up_proj",
                    f"layers.{l}.mlp.experts.up_proj.weight",
                ),
            )

            for e in range(cfg.num_experts):
                if fused_gate_up_key is not None and fused_down_key is not None:
                    gate_up_e = archive.get_slice_row(fused_gate_up_key, e)
                    down_e = archive.get_slice_row(fused_down_key, e)
                    if expert_layout == "out_in":
                        # Standard HuggingFace OlmoeExperts layout: gate_up_proj [E, 2*I, D], down_proj [E, D, I].
                        # Note: In OLMoE-1B-7B, 2*I == D == 2048, so auto-guessing transpose per matrix is unsafe!
                        if tuple(gate_up_e.shape) != (2 * I, D) or tuple(down_e.shape) != (D, I):
                            raise ValueError(
                                f"Fused expert shape mismatch at layer {l} expert {e} for expert_layout='out_in': "
                                f"got gate_up={tuple(gate_up_e.shape)} (expected ({2 * I}, {D})), "
                                f"down={tuple(down_e.shape)} (expected ({D}, {I})). "
                                f"If exporting a transposed [in, out] checkpoint, pass expert_layout='in_out'."
                            )
                        gate_w = gate_up_e[:I, :]
                        up_w = gate_up_e[I:, :]
                        down_w = down_e
                    else:
                        if tuple(gate_up_e.shape) != (D, 2 * I) or tuple(down_e.shape) != (I, D):
                            raise ValueError(
                                f"Fused expert shape mismatch at layer {l} expert {e} for expert_layout='in_out': "
                                f"got gate_up={tuple(gate_up_e.shape)} (expected ({D}, {2 * I})), "
                                f"down={tuple(down_e.shape)} (expected ({I}, {D}))."
                            )
                        gate_up_t = gate_up_e.t().contiguous()
                        gate_w = gate_up_t[:I, :]
                        up_w = gate_up_t[I:, :]
                        down_w = down_e.t().contiguous()
                elif (
                    batched_gate_key is not None
                    and batched_up_key is not None
                    and fused_down_key is not None
                ):
                    gate_w = archive.get_slice_row(batched_gate_key, e)
                    up_w = archive.get_slice_row(batched_up_key, e)
                    down_w = archive.get_slice_row(fused_down_key, e)
                else:
                    gk = req_key(
                        f"layers.{l}.mlp.experts.{e}.gate_proj.weight",
                        (
                            f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight",
                            f"layers.{l}.mlp.experts.{e}.gate_proj.weight",
                        ),
                    )
                    uk = req_key(
                        f"layers.{l}.mlp.experts.{e}.up_proj.weight",
                        (
                            f"model.layers.{l}.mlp.experts.{e}.up_proj.weight",
                            f"layers.{l}.mlp.experts.{e}.up_proj.weight",
                        ),
                    )
                    dk = req_key(
                        f"layers.{l}.mlp.experts.{e}.down_proj.weight",
                        (
                            f"model.layers.{l}.mlp.experts.{e}.down_proj.weight",
                            f"layers.{l}.mlp.experts.{e}.down_proj.weight",
                        ),
                    )
                    gate_w = archive.get(gk)
                    up_w = archive.get(uk)
                    down_w = archive.get(dk)

                exp_writer.write_expert(l, e, gate_w, up_w, down_w)
                del gate_w, up_w, down_w

        experts_header, expert_entries = exp_writer.finalize()

    return {
        "core_path": core_path,
        "experts_path": experts_path,
        "core_header": core_header,
        "core_entries": core_entries,
        "experts_header": experts_header,
        "expert_entries": expert_entries,
    }


def export_state_dict(
    config_dict: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    out_dir: Union[str, Path],
    *,
    bitlinear_rms_prescale: bool | None = None,
    norm_topk_prob: bool | None = None,
    tie_word_embeddings: bool | None = None,
    embed_dtype: str = "f32",
    lm_head_dtype: str = "f32",
    attn_dtype: str = "ternary",
    expert_layout: str = "out_in",
) -> dict[str, Any]:
    """Export an in-memory PyTorch ``state_dict`` and ``config_dict`` to ``core.bin`` + ``experts.bin``."""
    archive = _StateDictArchive(state_dict)
    try:
        return _export_from_archive(
            config_dict,
            archive,
            out_dir,
            bitlinear_rms_prescale=bitlinear_rms_prescale,
            norm_topk_prob=norm_topk_prob,
            tie_word_embeddings=tie_word_embeddings,
            embed_dtype=embed_dtype,
            lm_head_dtype=lm_head_dtype,
            attn_dtype=attn_dtype,
            expert_layout=expert_layout,
        )
    finally:
        archive.close()


def export_checkpoint(
    model_id_or_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    revision: str | None = None,
    bitlinear_rms_prescale: bool | None = None,
    norm_topk_prob: bool | None = None,
    tie_word_embeddings: bool | None = None,
    embed_dtype: str = "f32",
    lm_head_dtype: str = "f32",
    attn_dtype: str = "ternary",
    expert_layout: str = "out_in",
) -> dict[str, Any]:
    """Stream a local safetensors directory or remote HuggingFace repo into ``core.bin`` + ``experts.bin``."""
    candidate_path = Path(model_id_or_path)
    if candidate_path.exists():
        local_dir = candidate_path
    else:
        from huggingface_hub import snapshot_download

        local_dir = Path(
            snapshot_download(
                repo_id=str(model_id_or_path),
                revision=revision,
                allow_patterns=["*.json", "*.safetensors"],
            )
        )

    config_path = local_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in checkpoint directory: {local_dir}")
    config_dict = json.loads(config_path.read_text(encoding="utf-8"))

    archive = _SafetensorsShardArchive(local_dir)
    try:
        return _export_from_archive(
            config_dict,
            archive,
            out_dir,
            bitlinear_rms_prescale=bitlinear_rms_prescale,
            norm_topk_prob=norm_topk_prob,
            tie_word_embeddings=tie_word_embeddings,
            embed_dtype=embed_dtype,
            lm_head_dtype=lm_head_dtype,
            attn_dtype=attn_dtype,
            expert_layout=expert_layout,
        )
    finally:
        archive.close()


# ---------------------------------------------------------------------------
# Exact Section 4 Reference Forward Pass (PyTorch <-> Rust Golden Parity)
# ---------------------------------------------------------------------------


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f64 = x.to(dtype=torch.float64)
    inv_rms = torch.rsqrt(torch.mean(x_f64 * x_f64, dim=-1, keepdim=True) + float(eps))
    return (x_f64 * inv_rms).to(dtype=torch.float32) * weight.to(dtype=torch.float32)


def _apply_half_split_rope(
    z: torch.Tensor,
    position: int,
    rope_theta: float,
) -> torch.Tensor:
    """Section 4.2 half-split RoPE on ``[num_heads, head_dim]`` at scalar position ``p``."""
    head_dim = z.shape[-1]
    half = head_dim // 2
    j = torch.arange(half, dtype=torch.float32, device=z.device)
    theta = float(position) * (float(rope_theta) ** (-2.0 * j / float(head_dim)))
    cos_t = torch.cos(theta).unsqueeze(0)
    sin_t = torch.sin(theta).unsqueeze(0)
    z_first = z[:, :half]
    z_second = z[:, half:]
    return torch.cat(
        (z_first * cos_t - z_second * sin_t, z_second * cos_t + z_first * sin_t),
        dim=-1,
    )


def run_reference_forward(
    model_dir: Union[str, Path],
    prompt_token_ids: Sequence[int],
    hints: torch.Tensor | None = None,
    *,
    max_cached_experts: int = 16,
    max_cached_core_tensors: int = 32,
) -> dict[str, Any]:
    """Execute the exact Section 4 forward pass from ``core.bin`` and ``experts.bin`` with bounded LRU caches."""
    if len(prompt_token_ids) == 0:
        raise ValueError("prompt_token_ids must be non-empty")

    base = Path(model_dir)
    core_path = base / "core.bin"
    experts_path = base / "experts.bin"

    core_hdr, core_entries = read_core_header_and_dir(core_path, verify_crc=True)
    exp_hdr, exp_entries = read_experts_header_and_dir(experts_path, verify_crc=True)

    core_entry_by_name = {ent.name: ent for ent in core_entries}
    core_lru: OrderedDict[str, Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = OrderedDict()
    core_cap = max(4, int(max_cached_core_tensors))

    def get_core_item(name: str) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if name in core_lru:
            core_lru.move_to_end(name)
            return core_lru[name]
        ent = core_entry_by_name[name]
        loaded = read_core_tensor(core_path, ent, verify_crc=True)
        if ent.dtype == DTYPE_TERNARY_2BIT:
            assert isinstance(loaded, tuple)
            w_np, s_np = loaded
            val: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] = (
                torch.from_numpy(w_np),
                torch.from_numpy(s_np),
            )
        else:
            assert isinstance(loaded, np.ndarray)
            val = torch.from_numpy(loaded).to(dtype=torch.float32)
        while len(core_lru) >= core_cap:
            core_lru.popitem(last=False)
        core_lru[name] = val
        return val

    exp_entry_map = {(ent.layer_id, ent.expert_id): ent for ent in exp_entries}
    expert_cache: OrderedDict[tuple[int, int], dict[str, tuple[torch.Tensor, torch.Tensor]]] = (
        OrderedDict()
    )
    exp_cap = max(1, int(max_cached_experts))

    def get_expert(l: int, e: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        key = (l, e)
        if key in expert_cache:
            expert_cache.move_to_end(key)
            return expert_cache[key]
        raw = read_expert_weights(
            experts_path, exp_hdr, exp_entry_map[key], verify_crc=True
        )
        loaded_exp = {
            k: (torch.from_numpy(v[0]), torch.from_numpy(v[1])) for k, v in raw.items()
        }
        while len(expert_cache) >= exp_cap:
            expert_cache.popitem(last=False)
        expert_cache[key] = loaded_exp
        return loaded_exp

    rms_prescale = bool(core_hdr.flags & FLAG_BITLINEAR_RMS_PRESCALE)
    norm_topk = bool(core_hdr.flags & FLAG_NORM_TOPK_PROB)
    tie_emb = bool(core_hdr.flags & FLAG_TIE_WORD_EMBEDDINGS)
    has_hint = bool(core_hdr.flags & FLAG_HAS_HINT_EMBEDDING)
    eps = float(core_hdr.rms_norm_eps)

    def apply_core_linear(name: str, x: torch.Tensor) -> torch.Tensor:
        item = get_core_item(name)
        if isinstance(item, tuple):
            w_t, s_t = item
            return bitlinear_reference_gemv(
                x, w_t, s_t, rms_prescale=rms_prescale, rms_eps=eps
            )
        return torch.matmul(x.to(dtype=torch.float32), item.t())

    def dense_param(name: str) -> torch.Tensor:
        item = get_core_item(name)
        if isinstance(item, tuple):
            raise TypeError(f"Expected dense tensor for {name!r}, found Ternary2Bit")
        return item

    H = core_hdr.num_attention_heads
    H_kv = core_hdr.num_kv_heads
    d = core_hdr.head_dim
    gqa_groups = H // H_kv
    inv_sqrt_d = 1.0 / math.sqrt(float(d))

    # Per-layer stateful buffers: AnchorGQA KV cache or GatedLinearRecurrence S_t [H, d, d].
    kv_caches: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    rec_states: dict[int, torch.Tensor] = {}
    for l in range(core_hdr.num_layers):
        if core_hdr.layer_kinds[l] == LAYER_KIND_ANCHOR_GQA:
            kv_caches[l] = ([], [])
        else:
            rec_states[l] = torch.zeros((H, d, d), dtype=torch.float32)

    embed_table = dense_param("model.embed_tokens.weight")
    final_norm_w = dense_param("model.norm.weight")

    step_logits: list[list[float]] = []
    layer0_post_moe_steps: list[list[float]] = []

    for pos, tok_id in enumerate(prompt_token_ids):
        if not (0 <= int(tok_id) < core_hdr.vocab_size):
            raise ValueError(f"Token ID {tok_id} out of range 0..{core_hdr.vocab_size}")
        h_t = embed_table[int(tok_id)].clone()

        if has_hint and hints is not None and int(hints[pos, 0].item()) != 0:
            hint_table = dense_param("model.hint_embedding.table.weight")
            hint_proj = dense_param("model.hint_embedding.proj.weight")
            slot_buckets = hint_table.shape[0] // 4
            slot_offsets = torch.arange(4, dtype=torch.long) * slot_buckets
            raw_ids = hints[pos].to(dtype=torch.long)
            bucket_ids = torch.remainder(raw_ids, slot_buckets) + slot_offsets
            hint_flat = hint_table[bucket_ids].reshape(-1)
            h_t = h_t + torch.matmul(hint_flat, hint_proj.t())

        for l in range(core_hdr.num_layers):
            u_t = _rms_norm(h_t, dense_param(f"layers.{l}.input_layernorm.weight"), eps)

            q_flat = _rms_norm(
                apply_core_linear(f"layers.{l}.self_attn.q_proj.weight", u_t),
                dense_param(f"layers.{l}.self_attn.q_norm.weight"),
                eps,
            )
            k_flat = _rms_norm(
                apply_core_linear(f"layers.{l}.self_attn.k_proj.weight", u_t),
                dense_param(f"layers.{l}.self_attn.k_norm.weight"),
                eps,
            )
            v_flat = apply_core_linear(f"layers.{l}.self_attn.v_proj.weight", u_t)

            q_t = q_flat.reshape(H, d)
            k_t = k_flat.reshape(H_kv, d)
            v_t = v_flat.reshape(H_kv, d)

            if core_hdr.layer_kinds[l] == LAYER_KIND_ANCHOR_GQA:
                q_rope = _apply_half_split_rope(q_t, pos, core_hdr.rope_theta)
                k_rope = _apply_half_split_rope(k_t, pos, core_hdr.rope_theta)
                k_list, v_list = kv_caches[l]
                k_list.append(k_rope.to(dtype=torch.float16).to(dtype=torch.float32))
                v_list.append(v_t.to(dtype=torch.float16).to(dtype=torch.float32))

                # k_seq, v_seq: [pos + 1, H_kv, d] -> repeat GQA groups -> [pos + 1, H, d]
                k_seq = torch.stack(k_list, dim=0).repeat_interleave(gqa_groups, dim=1)
                v_seq = torch.stack(v_list, dim=0).repeat_interleave(gqa_groups, dim=1)

                # Causal scores across [0 .. pos] for each head h in 0..H
                scores = torch.einsum("hd,thd->ht", q_rope, k_seq) * inv_sqrt_d
                attn_probs = F.softmax(scores, dim=-1)
                o_heads = torch.einsum("ht,thd->hd", attn_probs, v_seq)
            else:
                k_rep = k_t.repeat_interleave(gqa_groups, dim=0)
                v_rep = v_t.repeat_interleave(gqa_groups, dim=0)
                q_hat = q_t * inv_sqrt_d
                k_hat = k_rep * inv_sqrt_d

                decay_w = dense_param(f"layers.{l}.self_attn.decay_proj.weight")
                decay_b = dense_param(f"layers.{l}.self_attn.decay_proj.bias")
                alpha_t = torch.sigmoid(torch.matmul(u_t, decay_w.t()) + decay_b)  # [H]

                state = rec_states[l]
                # S_t^(h) = alpha_{t,h} * S_{t-1}^(h) + k_hat_{t,h} (x) v_{t,h}^T
                outer_kv = torch.einsum("hi,hj->hij", k_hat, v_rep)
                state = alpha_t.view(H, 1, 1) * state + outer_kv
                rec_states[l] = state
                # o_{t,h} = (S_t^(h))^T q_hat_{t,h}
                o_heads = torch.einsum("hij,hi->hj", state, q_hat)

            o_flat = o_heads.reshape(H * d)
            h_prime = h_t + apply_core_linear(f"layers.{l}.self_attn.o_proj.weight", o_flat)

            # Section 4.4 Sparse SwiGLU MoE Block
            m_t = _rms_norm(
                h_prime, dense_param(f"layers.{l}.post_attention_layernorm.weight"), eps
            )
            router_logits = torch.matmul(
                m_t, dense_param(f"layers.{l}.mlp.gate.weight").t()
            )
            router_probs = F.softmax(router_logits, dim=-1)

            # Deterministic tie-breaking: lowest expert_id wins on exact probability ties.
            ranked_experts = sorted(
                range(core_hdr.num_experts),
                key=lambda eid: (-float(router_probs[eid].item()), eid),
            )[: core_hdr.top_k]
            top_weights = router_probs[ranked_experts]
            if norm_topk:
                top_weights = top_weights / top_weights.sum()

            moe_acc = torch.zeros_like(h_prime)
            for idx_pos, eid in enumerate(ranked_experts):
                exp_w = get_expert(l, eid)
                gate_w, gate_s = exp_w["gate_proj"]
                up_w, up_s = exp_w["up_proj"]
                down_w, down_s = exp_w["down_proj"]

                gate_act = bitlinear_reference_gemv(
                    m_t, gate_w, gate_s, rms_prescale=rms_prescale, rms_eps=eps
                )
                up_act = bitlinear_reference_gemv(
                    m_t, up_w, up_s, rms_prescale=rms_prescale, rms_eps=eps
                )
                hidden_act = F.silu(gate_act) * up_act
                exp_out = bitlinear_reference_gemv(
                    hidden_act, down_w, down_s, rms_prescale=rms_prescale, rms_eps=eps
                )
                moe_acc = moe_acc + top_weights[idx_pos] * exp_out

            h_t = h_prime + moe_acc
            if l == 0:
                layer0_post_moe_steps.append([float(v) for v in h_t.tolist()])

        h_norm = _rms_norm(h_t, final_norm_w, eps)
        if tie_emb:
            logits = torch.matmul(h_norm, embed_table.t())
        else:
            logits = apply_core_linear("lm_head.weight", h_norm)
        step_logits.append([float(v) for v in logits.tolist()])

    return {
        "prompt_token_ids": [int(t) for t in prompt_token_ids],
        "step_logits": step_logits,
        "prefill_last_logits": step_logits[-1],
        "layer0_post_moe": layer0_post_moe_steps[-1],
        "layer0_post_moe_steps": layer0_post_moe_steps,
    }


# ---------------------------------------------------------------------------
# Deterministic Synthetic Tiny Hybrid Micro-TERRA Builder & Golden Exporter
# ---------------------------------------------------------------------------


def build_synthetic_tiny_state(
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Construct a deterministic 8-layer (7:1 recurrent:GQA) miniature Micro-TERRA model."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    vocab_size = 256
    hidden_size = 64
    intermediate_size = 128
    num_layers = 8
    num_attention_heads = 4
    num_kv_heads = 4
    head_dim = 16
    num_experts = 8
    top_k = 2
    lookahead_offset = 3
    layer_kinds = [1, 1, 1, 1, 1, 1, 1, 0]

    config_dict: dict[str, Any] = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_experts": num_experts,
        "top_k": top_k,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "lookahead_offset": lookahead_offset,
        "layer_kinds": layer_kinds,
        "bitlinear_rms_prescale": True,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
    }

    def rand_w(rows: int, cols: int, std: float = 0.05) -> torch.Tensor:
        return torch.randn((rows, cols), generator=gen, dtype=torch.float32) * std

    def rand_norm(dim: int) -> torch.Tensor:
        return 1.0 + 0.05 * torch.randn((dim,), generator=gen, dtype=torch.float32)

    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": rand_w(vocab_size, hidden_size, std=0.04),
        "model.norm.weight": rand_norm(hidden_size),
        "lm_head.weight": rand_w(vocab_size, hidden_size, std=0.04),
    }

    for l in range(num_layers):
        prefix = f"model.layers.{l}"
        sd[f"{prefix}.input_layernorm.weight"] = rand_norm(hidden_size)
        sd[f"{prefix}.post_attention_layernorm.weight"] = rand_norm(hidden_size)

        sd[f"{prefix}.self_attn.q_proj.weight"] = rand_w(
            num_attention_heads * head_dim, hidden_size
        )
        sd[f"{prefix}.self_attn.k_proj.weight"] = rand_w(num_kv_heads * head_dim, hidden_size)
        sd[f"{prefix}.self_attn.v_proj.weight"] = rand_w(num_kv_heads * head_dim, hidden_size)
        sd[f"{prefix}.self_attn.o_proj.weight"] = rand_w(
            hidden_size, num_attention_heads * head_dim
        )
        sd[f"{prefix}.self_attn.q_norm.weight"] = rand_norm(num_attention_heads * head_dim)
        sd[f"{prefix}.self_attn.k_norm.weight"] = rand_norm(num_kv_heads * head_dim)

        if layer_kinds[l] == LAYER_KIND_GATED_LINEAR_RECURRENCE:
            sd[f"{prefix}.self_attn.decay_proj.weight"] = rand_w(
                num_attention_heads, hidden_size, std=0.05
            )
            sd[f"{prefix}.self_attn.decay_proj.bias"] = -0.8 + 0.2 * torch.randn(
                (num_attention_heads,), generator=gen, dtype=torch.float32
            )

        sd[f"{prefix}.mlp.gate.weight"] = rand_w(num_experts, hidden_size, std=0.05)
        if l + lookahead_offset < num_layers:
            sd[f"{prefix}.lookahead_router.weight"] = rand_w(
                num_experts, hidden_size, std=0.05
            )

        for e in range(num_experts):
            ep = f"{prefix}.mlp.experts.{e}"
            sd[f"{ep}.gate_proj.weight"] = rand_w(intermediate_size, hidden_size)
            sd[f"{ep}.up_proj.weight"] = rand_w(intermediate_size, hidden_size)
            sd[f"{ep}.down_proj.weight"] = rand_w(hidden_size, intermediate_size)

    return config_dict, sd


def export_synthetic_tiny(
    out_dir: Union[str, Path],
    *,
    seed: int = 42,
    prompt_token_ids: Sequence[int] = (1, 42, 17, 99),
) -> dict[str, Any]:
    """Export deterministic synthetic tiny Micro-TERRA artifacts + ``golden_ref.json``."""
    target = Path(out_dir)
    config_dict, state_dict = build_synthetic_tiny_state(seed=seed)
    export_info = export_state_dict(config_dict, state_dict, target)
    golden = run_reference_forward(target, prompt_token_ids)
    golden_path = target / "golden_ref.json"
    golden_path.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    export_info["golden_ref_path"] = golden_path
    export_info["golden_ref"] = golden
    return export_info


def _build_valid_ternary_byte_lut() -> np.ndarray:
    """Return all 81 valid 2-bit packed bytes whose 4 crumbs lie strictly in `{0b00, 0b01, 0b10}`."""
    crumbs = (0, 1, 2)
    lut = [
        c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
        for c0 in crumbs
        for c1 in crumbs
        for c2 in crumbs
        for c3 in crumbs
    ]
    return np.asarray(lut, dtype=np.uint8)


def export_synthetic_olmoe(
    out_dir: Union[str, Path],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate full `OLMoE-1B-7B-0125`-shaped (`16` layers x `64` experts = `1,024` experts)
    4KB-aligned ternary binary shards (`core.bin` + `experts.bin`) in seconds with minimal RAM.
    """
    t_start = time.perf_counter()
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    byte_lut = _build_valid_ternary_byte_lut()

    vocab_size = 50304
    hidden_size = 2048
    intermediate_size = 1024
    num_layers = 16
    num_heads = 16
    num_kv_heads = 16
    head_dim = 128
    num_experts = 64
    top_k = 8
    lookahead_offset = 3
    layer_kinds = tuple([LAYER_KIND_GATED_LINEAR_RECURRENCE] * 7 + [LAYER_KIND_ANCHOR_GQA] + [LAYER_KIND_GATED_LINEAR_RECURRENCE] * 7 + [LAYER_KIND_ANCHOR_GQA])

    cfg = TerraModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        top_k=top_k,
        max_position_embeddings=16384,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        lookahead_offset=lookahead_offset,
        layer_kinds=layer_kinds,
        flags=FLAG_BITLINEAR_RMS_PRESCALE | FLAG_NORM_TOPK_PROB,
    )

    def _packed_ternary_provider(rows: int, cols: int, local_seed: int) -> Callable[[], PrepackedTernary]:
        def _build() -> PrepackedTernary:
            r = np.random.default_rng(local_seed)
            # Sample random indices into the 81-entry valid-ternary byte LUT (zero 0b11 crumbs!)
            idx = r.integers(0, 81, size=(rows, cols // 4), dtype=np.uint8)
            packed = byte_lut[idx]
            scales = r.uniform(0.015, 0.035, size=(rows,)).astype(np.float32)
            return PrepackedTernary(rows=rows, cols=cols, packed=packed, scales=scales)
        return _build

    def _f32_norm_provider(dim: int, local_seed: int) -> Callable[[], np.ndarray]:
        def _build() -> np.ndarray:
            r = np.random.default_rng(local_seed)
            return (1.0 + r.normal(0.0, 0.02, size=(dim,))).astype(np.float32)
        return _build

    def _f32_mat_provider(rows: int, cols: int, std: float, local_seed: int) -> Callable[[], np.ndarray]:
        def _build() -> np.ndarray:
            r = np.random.default_rng(local_seed)
            return r.normal(0.0, std, size=(rows, cols)).astype(np.float32)
        return _build

    core_writer = CoreBinWriter(cfg)
    core_writer.add_tensor(
        "model.embed_tokens.weight",
        lambda: np.random.default_rng(seed + 1).normal(0.0, 0.02, size=(vocab_size, hidden_size)).astype(np.float16),
        DTYPE_F16,
    )
    core_writer.add_tensor("model.norm.weight", _f32_norm_provider(hidden_size, seed + 2), DTYPE_F32)
    core_writer.add_prepacked_ternary(
        "lm_head.weight",
        _packed_ternary_provider(vocab_size, hidden_size, seed + 3),
    )

    for l in range(num_layers):
        l_seed = seed + 100 * (l + 1)
        core_writer.add_tensor(
            f"layers.{l}.input_layernorm.weight",
            _f32_norm_provider(hidden_size, l_seed + 1),
            DTYPE_F32,
        )
        core_writer.add_tensor(
            f"layers.{l}.post_attention_layernorm.weight",
            _f32_norm_provider(hidden_size, l_seed + 2),
            DTYPE_F32,
        )
        for p_idx, proj in enumerate(("q_proj", "k_proj", "v_proj", "o_proj")):
            core_writer.add_prepacked_ternary(
                f"layers.{l}.self_attn.{proj}.weight",
                _packed_ternary_provider(hidden_size, hidden_size, l_seed + 10 + p_idx),
            )
        core_writer.add_tensor(
            f"layers.{l}.self_attn.q_norm.weight",
            _f32_norm_provider(hidden_size, l_seed + 20),
            DTYPE_F32,
        )
        core_writer.add_tensor(
            f"layers.{l}.self_attn.k_norm.weight",
            _f32_norm_provider(hidden_size, l_seed + 21),
            DTYPE_F32,
        )
        if layer_kinds[l] == LAYER_KIND_GATED_LINEAR_RECURRENCE:
            core_writer.add_tensor(
                f"layers.{l}.self_attn.decay_proj.weight",
                _f32_mat_provider(num_heads, hidden_size, 0.02, l_seed + 30),
                DTYPE_F32,
            )
            core_writer.add_tensor(
                f"layers.{l}.self_attn.decay_proj.bias",
                lambda ls=l_seed: np.full((num_heads,), 2.8 + (ls % 7) * 0.05, dtype=np.float32),
                DTYPE_F32,
            )
        core_writer.add_tensor(
            f"layers.{l}.mlp.gate.weight",
            _f32_mat_provider(num_experts, hidden_size, 0.05, l_seed + 40),
            DTYPE_F32,
        )
        if l + lookahead_offset < num_layers:
            core_writer.add_tensor(
                f"layers.{l}.lookahead_router.weight",
                _f32_mat_provider(num_experts, hidden_size, 0.05, l_seed + 50),
                DTYPE_F32,
            )

    core_path = target_dir / "core.bin"
    core_header, core_entries = core_writer.write(core_path)

    # Pre-generate 16 unique packed slabs (2 * 524,288 bytes each = 16 MB total RAM)
    # so writing 1,024 full OLMoE experts (1.627 GB) takes ~1-3s with zero per-expert float generation!
    proj_bytes = intermediate_size * (hidden_size // 4)  # 524,288 B == D * (I // 4)
    bank_count = 16
    bank_slabs = [
        byte_lut[rng.integers(0, 81, size=2 * proj_bytes, dtype=np.uint8)].tobytes()
        for _ in range(bank_count)
    ]
    base_scale_i = rng.uniform(0.015, 0.035, size=(intermediate_size,)).astype(np.float32)
    base_scale_d = rng.uniform(0.015, 0.035, size=(hidden_size,)).astype(np.float32)

    experts_path = target_dir / "experts.bin"
    with ExpertsBinWriter(
        experts_path,
        num_layers=num_layers,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ) as exp_writer:
        for l in range(num_layers):
            for e in range(num_experts):
                flat = l * num_experts + e
                slab_g = memoryview(bank_slabs[(flat * 3) % bank_count])
                slab_u = memoryview(bank_slabs[(flat * 3 + 1) % bank_count])
                slab_d = memoryview(bank_slabs[(flat * 3 + 2) % bank_count])
                off_g = ((flat * 4099) % proj_bytes)
                off_u = ((flat * 8191) % proj_bytes)
                off_d = ((flat * 12289) % proj_bytes)
                scale_mult = np.float32(0.85 + 0.30 * ((flat % 29) / 28.0))
                exp_writer.write_prepacked_expert(
                    l,
                    e,
                    slab_g[off_g : off_g + proj_bytes],
                    slab_u[off_u : off_u + proj_bytes],
                    slab_d[off_d : off_d + proj_bytes],
                    base_scale_i * scale_mult,
                    base_scale_i * (np.float32(1.1) - np.float32(0.2) * (e / 63.0)),
                    base_scale_d * scale_mult,
                    validate_crumbs=False,  # Guaranteed by construction from 81-entry `byte_lut`
                )
        experts_header, expert_entries = exp_writer.finalize()

    elapsed_sec = time.perf_counter() - t_start
    return {
        "core_path": core_path,
        "experts_path": experts_path,
        "core_header": core_header,
        "core_entries": core_entries,
        "experts_header": experts_header,
        "expert_entries": expert_entries,
        "elapsed_sec": elapsed_sec,
    }


# ---------------------------------------------------------------------------
# Command-Line Entrypoint
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export OLMoE / Micro-TERRA checkpoints to core.bin + 4KB-aligned experts.bin"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--model-id-or-path",
        "--checkpoint",
        dest="model_id_or_path",
        type=str,
        help="HuggingFace repo ID (e.g. allenai/OLMoE-1B-7B-0125) or local safetensors directory",
    )
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate full OLMoE-1B-7B-shaped (16 layers x 64 experts = 1,024 experts) synthetic .bin shards",
    )
    src.add_argument(
        "--synthetic-tiny",
        action="store_true",
        help="Export a deterministic 8-layer 7:1 hybrid miniature model + golden_ref.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination directory for core.bin and experts.bin",
    )
    parser.add_argument("--revision", type=str, default=None, help="Optional HF git revision")
    parser.add_argument(
        "--rms-prescale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force or disable FLAG_BITLINEAR_RMS_PRESCALE bit on exported core.bin",
    )
    parser.add_argument(
        "--embed-dtype",
        choices=("f32", "f16"),
        default="f32",
        help="Storage dtype for model.embed_tokens.weight (default: f32)",
    )
    parser.add_argument(
        "--lm-head-dtype",
        choices=("f32", "f16", "ternary"),
        default="f32",
        help="Storage dtype for lm_head.weight (default: f32)",
    )
    parser.add_argument(
        "--attn-dtype",
        choices=("ternary", "f32"),
        default="ternary",
        help="Storage dtype for self_attn.{q,k,v,o}_proj.weight (default: ternary)",
    )
    parser.add_argument(
        "--expert-layout",
        choices=("out_in", "in_out"),
        default="out_in",
        help="Fused expert weight layout: 'out_in' (HF [out, in] default) or 'in_out' (transposed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed for --synthetic / --synthetic-tiny (default: 42)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.synthetic:
        info = export_synthetic_olmoe(args.out_dir, seed=args.seed)
        print(
            f"Exported synthetic OLMoE-1B-7B Micro-TERRA to {args.out_dir} in {info['elapsed_sec']:.2f}s: "
            f"core.bin ({info['core_path'].stat().st_size} B, {len(info['core_entries'])} tensors), "
            f"experts.bin ({info['experts_path'].stat().st_size} B, {len(info['expert_entries'])} experts)"
        )
        return 0

    if args.synthetic_tiny:
        info = export_synthetic_tiny(args.out_dir, seed=args.seed)
        print(
            f"Exported synthetic-tiny Micro-TERRA to {args.out_dir}: "
            f"core.bin ({info['core_path'].stat().st_size} B), "
            f"experts.bin ({info['experts_path'].stat().st_size} B), "
            f"golden_ref.json ({info['golden_ref_path'].stat().st_size} B)"
        )
        return 0

    info = export_checkpoint(
        args.model_id_or_path,
        args.out_dir,
        revision=args.revision,
        bitlinear_rms_prescale=True if args.rms_prescale else None,
        embed_dtype=args.embed_dtype,
        lm_head_dtype=args.lm_head_dtype,
        attn_dtype=args.attn_dtype,
        expert_layout=args.expert_layout,
    )
    print(
        f"Exported {args.model_id_or_path} to {args.out_dir}: "
        f"core.bin ({info['core_path'].stat().st_size} B, {len(info['core_entries'])} tensors), "
        f"experts.bin ({info['experts_path'].stat().st_size} B, {len(info['expert_entries'])} experts)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

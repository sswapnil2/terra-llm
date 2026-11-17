"""1.58-bit ternary weight quantization, 2-bit crumb packing, and BitLinear GEMV reference.

Conforms strictly to Section 1 of ``schemas/terra_format.md``:
- ``0  -> 0b00`` (0)
- ``+1 -> 0b01`` (1)
- ``-1 -> 0b10`` (2)
- ``0b11`` (3) is reserved/invalid and rejected on unpack.
"""

from __future__ import annotations

import zlib
from typing import Union

import numpy as np
import torch

_DECODE_LUT = np.array([0, 1, -1, 0], dtype=np.int8)


def align_up(value: int, alignment: int) -> int:
    """Round non-negative integer ``value`` up to the nearest multiple of ``alignment``."""
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of two, got {alignment}")
    return (value + alignment - 1) & ~(alignment - 1)


def crc32_ieee(data: Union[bytes, bytearray, memoryview, np.ndarray]) -> int:
    """Compute standard IEEE 802.3 unsigned 32-bit CRC-32."""
    if isinstance(data, np.ndarray):
        buf = memoryview(np.ascontiguousarray(data)).cast("B")
    elif isinstance(data, memoryview):
        buf = data.cast("B")
    else:
        buf = data
    return zlib.crc32(buf) & 0xFFFFFFFF


def absmean_quantize_ternary(
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight tensor ``[M, K]`` to ternary ``{-1, 0, +1}`` via absmean scaling.

    Implements Section 1.1 of ``schemas/terra_format.md``:
        gamma_r = max((1 / K) * sum_k |W_{r,k}|, eps)
        W_tilde_{r,k} = round_ties_to_even(clip(W_{r,k} / gamma_r, -1.0, +1.0))

    Args:
        weight: 2D tensor of shape ``[M, K]`` where ``K % 4 == 0``.
        eps: Minimum scale clamp (default ``1e-8``).

    Returns:
        ``(w_ternary, scales)`` where ``w_ternary`` is ``torch.int8`` of shape ``[M, K]``
        with values in ``{-1, 0, +1}`` and ``scales`` is ``torch.float32`` of shape ``[M]``.
    """
    if isinstance(weight, np.ndarray):
        weight = torch.from_numpy(weight)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(weight).__name__}")
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor [M, K], got shape {tuple(weight.shape)}")

    rows, cols = weight.shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Weight dimensions must be positive, got {(rows, cols)}")
    if cols % 4 != 0:
        raise ValueError(f"in_features K must be divisible by 4, got K={cols}")

    w_f32 = weight.detach().to(dtype=torch.float32)
    scales = w_f32.abs().mean(dim=1).clamp(min=float(eps)).contiguous()
    w_ternary = (
        torch.round(torch.clamp(w_f32 / scales.unsqueeze(1), -1.0, 1.0))
        .to(dtype=torch.int8)
        .contiguous()
    )
    return w_ternary, scales


def pack_ternary_2bit(w_ternary: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Pack a 2D ternary array ``[M, K]`` (values in ``{-1, 0, +1}``) into ``uint8`` ``[M, K // 4]``.

    Implements Section 1.2 of ``schemas/terra_format.md``:
        packed[r, b] = code(W[r, 4b]) | (code(W[r, 4b+1]) << 2)
                     | (code(W[r, 4b+2]) << 4) | (code(W[r, 4b+3]) << 6)
    where ``code(0) = 0b00``, ``code(+1) = 0b01``, ``code(-1) = 0b10``.
    """
    if isinstance(w_ternary, torch.Tensor):
        arr = w_ternary.detach().cpu().numpy()
    elif isinstance(w_ternary, np.ndarray):
        arr = w_ternary
    else:
        arr = np.asarray(w_ternary)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D ternary array [M, K], got shape {arr.shape}")

    rows, cols = arr.shape
    if cols % 4 != 0:
        raise ValueError(f"Column count K must be divisible by 4, got K={cols}")

    if arr.dtype == np.bool_:
        raise ValueError("Boolean arrays are not valid ternary weight tensors")

    valid_mask = (arr == -1) | (arr == 0) | (arr == 1)
    if not bool(np.all(valid_mask)):
        bad_vals = arr[~valid_mask]
        sample = bad_vals.flat[0] if bad_vals.size > 0 else None
        raise ValueError(
            f"Ternary tensor contains values outside {{-1, 0, +1}} (e.g. {sample!r})"
        )

    w_i8 = np.ascontiguousarray(arr, dtype=np.int8)
    # Branchless code map on {-1, 0, +1}: 0 -> 0, +1 -> 1, -1 -> 2 (0b10)
    codes = np.where(w_i8 < 0, np.uint8(2), w_i8.astype(np.uint8))
    codes_4 = codes.reshape(rows, cols // 4, 4)
    packed = (
        codes_4[:, :, 0]
        | (codes_4[:, :, 1] << np.uint8(2))
        | (codes_4[:, :, 2] << np.uint8(4))
        | (codes_4[:, :, 3] << np.uint8(6))
    )
    return np.ascontiguousarray(packed, dtype=np.uint8)


def unpack_ternary_2bit(
    packed: Union[np.ndarray, bytes, bytearray, memoryview, torch.Tensor],
    rows: int,
    cols: int,
) -> np.ndarray:
    """Unpack ``uint8`` ``[rows, cols // 4]`` into ``int8`` ``[rows, cols]`` in ``{-1, 0, +1}``.

    Raises ``ValueError`` if ``cols % 4 != 0``, buffer size mismatches, or any 2-bit crumb
    equals ``0b11`` (reserved/invalid per Section 1.2 of ``schemas/terra_format.md``).
    """
    if rows < 0 or cols < 0:
        raise ValueError(f"rows and cols must be non-negative, got rows={rows}, cols={cols}")
    if cols % 4 != 0:
        raise ValueError(f"cols must be divisible by 4, got cols={cols}")

    expected_bytes = rows * (cols // 4)

    if isinstance(packed, torch.Tensor):
        raw = packed.detach().cpu().numpy()
        if raw.dtype != np.uint8:
            raise ValueError(f"Expected uint8 tensor for packed weights, got {raw.dtype}")
        if raw.ndim == 2 and raw.shape != (rows, cols // 4):
            raise ValueError(
                f"Packed shape {raw.shape} does not match expected {(rows, cols // 4)}"
            )
        if raw.size != expected_bytes:
            raise ValueError(
                f"Packed element count {raw.size} != expected {expected_bytes}"
            )
        arr = np.ascontiguousarray(raw, dtype=np.uint8).reshape(rows, cols // 4)
    elif isinstance(packed, (bytes, bytearray, memoryview)):
        buf = memoryview(packed).cast("B")
        if len(buf) != expected_bytes:
            raise ValueError(
                f"Packed byte length {len(buf)} != expected {expected_bytes} "
                f"for shape ({rows}, {cols})"
            )
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols // 4)
    elif isinstance(packed, np.ndarray):
        if packed.dtype != np.uint8:
            raise ValueError(f"Expected uint8 ndarray, got dtype {packed.dtype}")
        if packed.ndim == 2 and packed.shape != (rows, cols // 4):
            raise ValueError(
                f"Packed shape {packed.shape} does not match expected {(rows, cols // 4)}"
            )
        if packed.size != expected_bytes:
            raise ValueError(
                f"Packed size {packed.size} != expected {expected_bytes} "
                f"for shape ({rows}, {cols})"
            )
        arr = np.ascontiguousarray(packed, dtype=np.uint8).reshape(rows, cols // 4)
    else:
        raise TypeError(f"Unsupported packed buffer type: {type(packed).__name__}")

    # SWAR check: in any 2-bit lane (b1, b0), b1 & b0 == 1 iff crumb == 0b11.
    if arr.size > 0 and bool(np.any(((arr & (arr >> np.uint8(1))) & np.uint8(0x55)) != 0)):
        raise ValueError("Invalid reserved ternary crumb 0b11 encountered while unpacking")

    crumbs = np.empty((rows, cols // 4, 4), dtype=np.uint8)
    crumbs[:, :, 0] = arr & np.uint8(0x03)
    crumbs[:, :, 1] = (arr >> np.uint8(2)) & np.uint8(0x03)
    crumbs[:, :, 2] = (arr >> np.uint8(4)) & np.uint8(0x03)
    crumbs[:, :, 3] = (arr >> np.uint8(6)) & np.uint8(0x03)

    unpacked = _DECODE_LUT[crumbs].reshape(rows, cols)
    return np.ascontiguousarray(unpacked, dtype=np.int8)


def bitlinear_reference_gemv(
    x: torch.Tensor,
    w_ternary: torch.Tensor,
    scales: torch.Tensor,
    rms_prescale: bool = False,
    rms_eps: float = 1e-5,
) -> torch.Tensor:
    """Reference implementation of Section 1.3 BitLinear dynamic int8 GEMV / GEMM.

    Supports both 1D ``x`` of shape ``[K]`` (returns ``[M]``) and batched ``x``
    of shape ``[..., K]`` (returns ``[..., M]``) in ``torch.float32``.
    """
    if isinstance(w_ternary, np.ndarray):
        w_ternary = torch.from_numpy(w_ternary)
    if isinstance(scales, np.ndarray):
        scales = torch.from_numpy(scales)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if w_ternary.ndim != 2:
        raise ValueError(f"w_ternary must be 2D [M, K], got {tuple(w_ternary.shape)}")
    rows, cols = w_ternary.shape
    if x.ndim < 1 or x.shape[-1] != cols:
        raise ValueError(
            f"Input x shape {tuple(x.shape)} incompatible with w_ternary {tuple(w_ternary.shape)}"
        )
    if scales.ndim != 1 or scales.shape[0] != rows:
        raise ValueError(
            f"scales shape {tuple(scales.shape)} must match out_features M={rows}"
        )

    device = x.device
    x_f32 = x.to(dtype=torch.float32)

    if rms_prescale:
        x_f64 = x_f32.to(dtype=torch.float64)
        inv_rms = torch.rsqrt(torch.mean(x_f64 * x_f64, dim=-1, keepdim=True) + float(rms_eps))
        x_hat = (x_f64 * inv_rms).to(dtype=torch.float32)
    else:
        x_hat = x_f32

    max_abs = x_hat.abs().amax(dim=-1, keepdim=True)
    zero_token = max_abs < 1e-12
    s_x = max_abs / 127.0
    safe_s_x = torch.where(zero_token, torch.ones_like(s_x), s_x)

    x_q = torch.clamp(torch.round(x_hat / safe_s_x), -127.0, 127.0).to(dtype=torch.int32)
    x_q = torch.where(zero_token, torch.zeros_like(x_q), x_q)

    w_i32 = w_ternary.to(device=device, dtype=torch.int32)
    acc = torch.matmul(x_q, w_i32.t())  # exact integer accumulation

    scales_f32 = scales.to(device=device, dtype=torch.float32)
    eff_s_x = torch.where(zero_token, torch.zeros_like(s_x), s_x)
    y = acc.to(dtype=torch.float32) * scales_f32 * eff_s_x
    return torch.where(zero_token, torch.zeros_like(y), y)

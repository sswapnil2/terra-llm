//! Zero-FP-Multiply 1.58-bit Ternary GEMV Kernels (`NEON`, `AVX2`, `AVX-512`, and Scalar).
//!
//! Implements Section 1.3 of `schemas/terra_format.md`:
//! - Optional `RMSNorm` activation pre-scaling (`FLAG_BITLINEAR_RMS_PRESCALE`)
//! - Dynamic symmetric `int8` quantization (`round_ties_even`, clamped to `[-127, 127]`)
//! - Amortized 4-way crumb lane deinterleaving (`lane0..lane3`)
//! - Byte-shuffle LUT masking + pure integer `ADD`/`SUB` accumulation
//! - Scalar floating-point epilogue strictly outside the inner loop

const POS_MASK_SCALAR: [i32; 4] = [0, -1, 0, 0];
const NEG_MASK_SCALAR: [i32; 4] = [0, 0, -1, 0];

/// Pre-quantized and 4-way lane-deinterleaved input vector $x \in \mathbb{R}^K$.
/// Can be reused across multiple projections sharing the same input (e.g., `q/k/v` or `gate/up` across `top_k` experts).
#[derive(Debug, Clone)]
pub struct QuantizedActivation {
    pub cols: usize,
    pub act_scale: f32,
    pub is_zero: bool,
    pub lane0: Vec<i8>,
    pub lane1: Vec<i8>,
    pub lane2: Vec<i8>,
    pub lane3: Vec<i8>,
}

impl QuantizedActivation {
    /// Quantizes $x \in \mathbb{R}^K$ (`K % 4 == 0`) per Section 1.3 of `schemas/terra_format.md`.
    pub fn quantize(x: &[f32], rms_prescale: bool, rms_eps: f32) -> Self {
        let cols = x.len();
        assert!(
            cols > 0 && cols % 4 == 0,
            "BitLinear input length ({cols}) must be > 0 and divisible by 4"
        );

        let inv_rms: f64 = if rms_prescale {
            let sum_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let mean_sq = sum_sq / (cols as f64);
            1.0f64 / (mean_sq + rms_eps as f64).sqrt()
        } else {
            1.0f64
        };

        let mut max_abs = 0.0f32;
        for &v in x {
            let scaled = (((v as f64) * inv_rms) as f32).abs();
            if scaled > max_abs {
                max_abs = scaled;
            }
        }

        if max_abs < 1e-12 {
            return Self {
                cols,
                act_scale: 0.0,
                is_zero: true,
                lane0: Vec::new(),
                lane1: Vec::new(),
                lane2: Vec::new(),
                lane3: Vec::new(),
            };
        }

        let act_scale = max_abs / 127.0f32;
        let num_bytes = cols / 4;
        let mut lane0 = vec![0i8; num_bytes];
        let mut lane1 = vec![0i8; num_bytes];
        let mut lane2 = vec![0i8; num_bytes];
        let mut lane3 = vec![0i8; num_bytes];

        for b in 0..num_bytes {
            let base = 4 * b;
            let q0 = ((((x[base] as f64) * inv_rms) as f32) / act_scale)
                .round_ties_even()
                .clamp(-127.0, 127.0) as i8;
            let q1 = ((((x[base + 1] as f64) * inv_rms) as f32) / act_scale)
                .round_ties_even()
                .clamp(-127.0, 127.0) as i8;
            let q2 = ((((x[base + 2] as f64) * inv_rms) as f32) / act_scale)
                .round_ties_even()
                .clamp(-127.0, 127.0) as i8;
            let q3 = ((((x[base + 3] as f64) * inv_rms) as f32) / act_scale)
                .round_ties_even()
                .clamp(-127.0, 127.0) as i8;
            lane0[b] = q0;
            lane1[b] = q1;
            lane2[b] = q2;
            lane3[b] = q3;
        }

        Self {
            cols,
            act_scale,
            is_zero: false,
            lane0,
            lane1,
            lane2,
            lane3,
        }
    }
}

/// Branchless integer-only scalar helper over byte range `start_byte..end_byte` of a single row.
/// Contains ZERO floating-point operations and ZERO integer multiplications.
#[inline]
fn scalar_row_range_acc(
    row_packed: &[u8],
    l0: &[i8],
    l1: &[i8],
    l2: &[i8],
    l3: &[i8],
    start_byte: usize,
    end_byte: usize,
) -> i32 {
    let mut acc: i32 = 0;
    for b in start_byte..end_byte {
        let p = row_packed[b] as usize;
        let c0 = p & 0x03;
        let c1 = (p >> 2) & 0x03;
        let c2 = (p >> 4) & 0x03;
        let c3 = (p >> 6) & 0x03;

        let x0 = l0[b] as i32;
        let x1 = l1[b] as i32;
        let x2 = l2[b] as i32;
        let x3 = l3[b] as i32;

        acc += (x0 & POS_MASK_SCALAR[c0]) - (x0 & NEG_MASK_SCALAR[c0]);
        acc += (x1 & POS_MASK_SCALAR[c1]) - (x1 & NEG_MASK_SCALAR[c1]);
        acc += (x2 & POS_MASK_SCALAR[c2]) - (x2 & NEG_MASK_SCALAR[c2]);
        acc += (x3 & POS_MASK_SCALAR[c3]) - (x3 & NEG_MASK_SCALAR[c3]);
    }
    acc
}

/// Portable branchless LUT scalar ternary GEMV on pre-quantized activations.
pub fn ternary_gemv_scalar_prequantized(
    packed: &[u8],
    weight_scales: &[f32],
    qx: &QuantizedActivation,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(cols, qx.cols);
    assert_eq!(cols % 4, 0);
    let row_bytes = cols / 4;
    assert_eq!(packed.len(), rows * row_bytes);
    assert_eq!(weight_scales.len(), rows);
    assert_eq!(out.len(), rows);

    if qx.is_zero {
        out.fill(0.0);
        return;
    }

    let act_scale = qx.act_scale;
    for r in 0..rows {
        let row_packed = &packed[r * row_bytes..(r + 1) * row_bytes];
        let acc_i32 = scalar_row_range_acc(
            row_packed,
            &qx.lane0,
            &qx.lane1,
            &qx.lane2,
            &qx.lane3,
            0,
            row_bytes,
        );
        // Scalar epilogue strictly outside the integer inner loop.
        out[r] = (acc_i32 as f32) * weight_scales[r] * act_scale;
    }
}

/// Portable scalar ternary GEMV (`[rows, cols] x [cols] -> [rows]`).
pub fn ternary_gemv_scalar(
    packed: &[u8],
    weight_scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    rms_prescale: bool,
    rms_eps: f32,
    out: &mut [f32],
) {
    let qx = QuantizedActivation::quantize(x, rms_prescale, rms_eps);
    ternary_gemv_scalar_prequantized(packed, weight_scales, &qx, rows, cols, out);
}

/// Apple Silicon / ARM64 NEON ternary GEMV kernel using `vqtbl1q_u8` byte-shuffle LUT masks
/// and pure integer `ADD`/`SUB` accumulation (`vaddl_s8`, `vaddl_high_s8`, `vaddq_s16`, `vsubq_s16`,
/// flushing `i16` into `int32x4_t` via `vpadalq_s16` every 32 chunks).
#[cfg(target_arch = "aarch64")]
pub fn ternary_gemv_neon_prequantized(
    packed: &[u8],
    weight_scales: &[f32],
    qx: &QuantizedActivation,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;

    assert_eq!(cols, qx.cols);
    assert_eq!(cols % 4, 0);
    let row_bytes = cols / 4;
    assert_eq!(packed.len(), rows * row_bytes);
    assert_eq!(weight_scales.len(), rows);
    assert_eq!(out.len(), rows);

    if qx.is_zero {
        out.fill(0.0);
        return;
    }

    const POS_MASK_LUT: [u8; 16] = [
        0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];
    const NEG_MASK_LUT: [u8; 16] = [
        0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];

    let num_simd_chunks = row_bytes / 16;
    let simd_bytes = num_simd_chunks * 16;
    let act_scale = qx.act_scale;

    let l0_ptr = qx.lane0.as_ptr();
    let l1_ptr = qx.lane1.as_ptr();
    let l2_ptr = qx.lane2.as_ptr();
    let l3_ptr = qx.lane3.as_ptr();

    // SAFETY: NEON is mandatory on all `aarch64` targets; bounds are checked above.
    unsafe {
        let pos_lut = vld1q_u8(POS_MASK_LUT.as_ptr());
        let neg_lut = vld1q_u8(NEG_MASK_LUT.as_ptr());
        let mask_03 = vdupq_n_u8(0x03);

        for r in 0..rows {
            let row_slice = &packed[r * row_bytes..(r + 1) * row_bytes];
            let row_ptr = row_slice.as_ptr();

            let mut acc32 = vdupq_n_s32(0);
            let mut acc16_lo = vdupq_n_s16(0);
            let mut acc16_hi = vdupq_n_s16(0);
            let mut chunks_since_flush = 0usize;

            for chunk_idx in 0..num_simd_chunks {
                let off = chunk_idx * 16;
                let p = vld1q_u8(row_ptr.add(off));

                let idx0 = vandq_u8(p, mask_03);
                let idx1 = vandq_u8(vshrq_n_u8::<2>(p), mask_03);
                let idx2 = vandq_u8(vshrq_n_u8::<4>(p), mask_03);
                let idx3 = vshrq_n_u8::<6>(p);

                let x0 = vld1q_s8(l0_ptr.add(off));
                let x1 = vld1q_s8(l1_ptr.add(off));
                let x2 = vld1q_s8(l2_ptr.add(off));
                let x3 = vld1q_s8(l3_ptr.add(off));

                let pos0 = vandq_s8(x0, vreinterpretq_s8_u8(vqtbl1q_u8(pos_lut, idx0)));
                let neg0 = vandq_s8(x0, vreinterpretq_s8_u8(vqtbl1q_u8(neg_lut, idx0)));

                let pos1 = vandq_s8(x1, vreinterpretq_s8_u8(vqtbl1q_u8(pos_lut, idx1)));
                let neg1 = vandq_s8(x1, vreinterpretq_s8_u8(vqtbl1q_u8(neg_lut, idx1)));

                let pos2 = vandq_s8(x2, vreinterpretq_s8_u8(vqtbl1q_u8(pos_lut, idx2)));
                let neg2 = vandq_s8(x2, vreinterpretq_s8_u8(vqtbl1q_u8(neg_lut, idx2)));

                let pos3 = vandq_s8(x3, vreinterpretq_s8_u8(vqtbl1q_u8(pos_lut, idx3)));
                let neg3 = vandq_s8(x3, vreinterpretq_s8_u8(vqtbl1q_u8(neg_lut, idx3)));

                let pos_lo = vaddq_s16(
                    vaddl_s8(vget_low_s8(pos0), vget_low_s8(pos1)),
                    vaddl_s8(vget_low_s8(pos2), vget_low_s8(pos3)),
                );
                let neg_lo = vaddq_s16(
                    vaddl_s8(vget_low_s8(neg0), vget_low_s8(neg1)),
                    vaddl_s8(vget_low_s8(neg2), vget_low_s8(neg3)),
                );
                acc16_lo = vaddq_s16(acc16_lo, vsubq_s16(pos_lo, neg_lo));

                let pos_hi = vaddq_s16(vaddl_high_s8(pos0, pos1), vaddl_high_s8(pos2, pos3));
                let neg_hi = vaddq_s16(vaddl_high_s8(neg0, neg1), vaddl_high_s8(neg2, neg3));
                acc16_hi = vaddq_s16(acc16_hi, vsubq_s16(pos_hi, neg_hi));

                chunks_since_flush += 1;
                if chunks_since_flush == 32 {
                    acc32 = vpadalq_s16(acc32, acc16_lo);
                    acc32 = vpadalq_s16(acc32, acc16_hi);
                    acc16_lo = vdupq_n_s16(0);
                    acc16_hi = vdupq_n_s16(0);
                    chunks_since_flush = 0;
                }
            }

            if chunks_since_flush > 0 {
                acc32 = vpadalq_s16(acc32, acc16_lo);
                acc32 = vpadalq_s16(acc32, acc16_hi);
            }

            let mut acc_i32 = vaddvq_s32(acc32);
            if simd_bytes < row_bytes {
                acc_i32 += scalar_row_range_acc(
                    row_slice,
                    &qx.lane0,
                    &qx.lane1,
                    &qx.lane2,
                    &qx.lane3,
                    simd_bytes,
                    row_bytes,
                );
            }

            // Scalar epilogue strictly outside the inner loop.
            out[r] = (acc_i32 as f32) * weight_scales[r] * act_scale;
        }
    }
}

/// Direct NEON ternary GEMV (`[rows, cols] x [cols] -> [rows]`).
#[cfg(target_arch = "aarch64")]
pub fn ternary_gemv_neon(
    packed: &[u8],
    weight_scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    rms_prescale: bool,
    rms_eps: f32,
    out: &mut [f32],
) {
    let qx = QuantizedActivation::quantize(x, rms_prescale, rms_eps);
    ternary_gemv_neon_prequantized(packed, weight_scales, &qx, rows, cols, out);
}

/// x86_64 AVX2 ternary GEMV kernel (32 packed bytes = 128 ternary weights per iteration)
/// using `_mm256_shuffle_epi8` byte-shuffle LUT masks + integer `_mm256_add_epi16`/`_mm256_sub_epi16`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn ternary_gemv_avx2_prequantized(
    packed: &[u8],
    weight_scales: &[f32],
    qx: &QuantizedActivation,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    assert_eq!(cols, qx.cols);
    assert_eq!(cols % 4, 0);
    let row_bytes = cols / 4;
    assert_eq!(packed.len(), rows * row_bytes);
    assert_eq!(weight_scales.len(), rows);
    assert_eq!(out.len(), rows);

    if qx.is_zero {
        out.fill(0.0);
        return;
    }

    let pos_lut = _mm256_setr_epi8(
        0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    );
    let neg_lut = _mm256_setr_epi8(
        0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    );
    let mask_03 = _mm256_set1_epi8(0x03);

    let num_chunks = row_bytes / 32;
    let simd_bytes = num_chunks * 32;
    let act_scale = qx.act_scale;

    let l0_ptr = qx.lane0.as_ptr();
    let l1_ptr = qx.lane1.as_ptr();
    let l2_ptr = qx.lane2.as_ptr();
    let l3_ptr = qx.lane3.as_ptr();

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn widen_sum_4_i8_avx2(
        a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
    ) -> (__m256i, __m256i) {
        let ab_lo = _mm256_add_epi16(
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a)),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b)),
        );
        let cd_lo = _mm256_add_epi16(
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(c)),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(d)),
        );
        let ab_hi = _mm256_add_epi16(
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(a)),
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(b)),
        );
        let cd_hi = _mm256_add_epi16(
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(c)),
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(d)),
        );
        (_mm256_add_epi16(ab_lo, cd_lo), _mm256_add_epi16(ab_hi, cd_hi))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn flush_i16_to_i32_avx2(acc32: __m256i, acc16: __m256i) -> __m256i {
        let lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc16));
        let hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(acc16));
        _mm256_add_epi32(acc32, _mm256_add_epi32(lo, hi))
    }

    for r in 0..rows {
        let row_slice = &packed[r * row_bytes..(r + 1) * row_bytes];
        let row_ptr = row_slice.as_ptr();

        let mut acc32 = _mm256_setzero_si256();
        let mut acc16_lo = _mm256_setzero_si256();
        let mut acc16_hi = _mm256_setzero_si256();
        let mut chunks_since_flush = 0usize;

        for chunk_idx in 0..num_chunks {
            let off = chunk_idx * 32;
            let p = _mm256_loadu_si256(row_ptr.add(off) as *const __m256i);

            let idx0 = _mm256_and_si256(p, mask_03);
            let idx1 = _mm256_and_si256(_mm256_srli_epi16(p, 2), mask_03);
            let idx2 = _mm256_and_si256(_mm256_srli_epi16(p, 4), mask_03);
            let idx3 = _mm256_and_si256(_mm256_srli_epi16(p, 6), mask_03);

            let x0 = _mm256_loadu_si256(l0_ptr.add(off) as *const __m256i);
            let x1 = _mm256_loadu_si256(l1_ptr.add(off) as *const __m256i);
            let x2 = _mm256_loadu_si256(l2_ptr.add(off) as *const __m256i);
            let x3 = _mm256_loadu_si256(l3_ptr.add(off) as *const __m256i);

            let pos0 = _mm256_and_si256(x0, _mm256_shuffle_epi8(pos_lut, idx0));
            let neg0 = _mm256_and_si256(x0, _mm256_shuffle_epi8(neg_lut, idx0));
            let pos1 = _mm256_and_si256(x1, _mm256_shuffle_epi8(pos_lut, idx1));
            let neg1 = _mm256_and_si256(x1, _mm256_shuffle_epi8(neg_lut, idx1));
            let pos2 = _mm256_and_si256(x2, _mm256_shuffle_epi8(pos_lut, idx2));
            let neg2 = _mm256_and_si256(x2, _mm256_shuffle_epi8(neg_lut, idx2));
            let pos3 = _mm256_and_si256(x3, _mm256_shuffle_epi8(pos_lut, idx3));
            let neg3 = _mm256_and_si256(x3, _mm256_shuffle_epi8(neg_lut, idx3));

            let (pos_lo, pos_hi) = widen_sum_4_i8_avx2(pos0, pos1, pos2, pos3);
            let (neg_lo, neg_hi) = widen_sum_4_i8_avx2(neg0, neg1, neg2, neg3);

            acc16_lo = _mm256_add_epi16(acc16_lo, _mm256_sub_epi16(pos_lo, neg_lo));
            acc16_hi = _mm256_add_epi16(acc16_hi, _mm256_sub_epi16(pos_hi, neg_hi));

            chunks_since_flush += 1;
            if chunks_since_flush == 32 {
                acc32 = flush_i16_to_i32_avx2(acc32, acc16_lo);
                acc32 = flush_i16_to_i32_avx2(acc32, acc16_hi);
                acc16_lo = _mm256_setzero_si256();
                acc16_hi = _mm256_setzero_si256();
                chunks_since_flush = 0;
            }
        }

        if chunks_since_flush > 0 {
            acc32 = flush_i16_to_i32_avx2(acc32, acc16_lo);
            acc32 = flush_i16_to_i32_avx2(acc32, acc16_hi);
        }

        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc32);
        let mut acc_i32: i32 = lanes.iter().sum();

        if simd_bytes < row_bytes {
            acc_i32 += scalar_row_range_acc(
                row_slice,
                &qx.lane0,
                &qx.lane1,
                &qx.lane2,
                &qx.lane3,
                simd_bytes,
                row_bytes,
            );
        }

        out[r] = (acc_i32 as f32) * weight_scales[r] * act_scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn ternary_gemv_avx2(
    packed: &[u8],
    weight_scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    rms_prescale: bool,
    rms_eps: f32,
    out: &mut [f32],
) {
    let qx = QuantizedActivation::quantize(x, rms_prescale, rms_eps);
    ternary_gemv_avx2_prequantized(packed, weight_scales, &qx, rows, cols, out);
}

/// x86_64 AVX-512 ternary GEMV kernel (64 packed bytes = 256 ternary weights per iteration)
/// using `_mm512_shuffle_epi8` + integer `_mm512_add_epi16`/`_mm512_sub_epi16`.
/// Enabled automatically when compiled with `rustc >= 1.89` where `avx512f`/`avx512bw` are stable.
#[cfg(all(target_arch = "x86_64", terra_has_stable_avx512))]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn ternary_gemv_avx512_prequantized(
    packed: &[u8],
    weight_scales: &[f32],
    qx: &QuantizedActivation,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    assert_eq!(cols, qx.cols);
    assert_eq!(cols % 4, 0);
    let row_bytes = cols / 4;
    assert_eq!(packed.len(), rows * row_bytes);
    assert_eq!(weight_scales.len(), rows);
    assert_eq!(out.len(), rows);

    if qx.is_zero {
        out.fill(0.0);
        return;
    }

    let pos_128 = _mm_setr_epi8(0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let neg_128 = _mm_setr_epi8(0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let pos_lut = _mm512_broadcast_i32x4(pos_128);
    let neg_lut = _mm512_broadcast_i32x4(neg_128);
    let mask_03 = _mm512_set1_epi8(0x03);

    let num_chunks = row_bytes / 64;
    let simd_bytes = num_chunks * 64;
    let act_scale = qx.act_scale;

    let l0_ptr = qx.lane0.as_ptr();
    let l1_ptr = qx.lane1.as_ptr();
    let l2_ptr = qx.lane2.as_ptr();
    let l3_ptr = qx.lane3.as_ptr();

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn widen_sum_4_i8_avx512(
        a: __m512i,
        b: __m512i,
        c: __m512i,
        d: __m512i,
    ) -> (__m512i, __m512i) {
        let ab_lo = _mm512_add_epi16(
            _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a)),
            _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b)),
        );
        let cd_lo = _mm512_add_epi16(
            _mm512_cvtepi8_epi16(_mm512_castsi512_si256(c)),
            _mm512_cvtepi8_epi16(_mm512_castsi512_si256(d)),
        );
        let ab_hi = _mm512_add_epi16(
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(a)),
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(b)),
        );
        let cd_hi = _mm512_add_epi16(
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(c)),
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(d)),
        );
        (_mm512_add_epi16(ab_lo, cd_lo), _mm512_add_epi16(ab_hi, cd_hi))
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn flush_i16_to_i32_avx512(acc32: __m512i, acc16: __m512i) -> __m512i {
        let lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(acc16));
        let hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(acc16));
        _mm512_add_epi32(acc32, _mm512_add_epi32(lo, hi))
    }

    for r in 0..rows {
        let row_slice = &packed[r * row_bytes..(r + 1) * row_bytes];
        let row_ptr = row_slice.as_ptr();

        let mut acc32 = _mm512_setzero_si512();
        let mut acc16_lo = _mm512_setzero_si512();
        let mut acc16_hi = _mm512_setzero_si512();
        let mut chunks_since_flush = 0usize;

        for chunk_idx in 0..num_chunks {
            let off = chunk_idx * 64;
            let p = _mm512_loadu_si512(row_ptr.add(off) as *const _);

            let idx0 = _mm512_and_si512(p, mask_03);
            let idx1 = _mm512_and_si512(_mm512_srli_epi16(p, 2), mask_03);
            let idx2 = _mm512_and_si512(_mm512_srli_epi16(p, 4), mask_03);
            let idx3 = _mm512_and_si512(_mm512_srli_epi16(p, 6), mask_03);

            let x0 = _mm512_loadu_si512(l0_ptr.add(off) as *const _);
            let x1 = _mm512_loadu_si512(l1_ptr.add(off) as *const _);
            let x2 = _mm512_loadu_si512(l2_ptr.add(off) as *const _);
            let x3 = _mm512_loadu_si512(l3_ptr.add(off) as *const _);

            let pos0 = _mm512_and_si512(x0, _mm512_shuffle_epi8(pos_lut, idx0));
            let neg0 = _mm512_and_si512(x0, _mm512_shuffle_epi8(neg_lut, idx0));
            let pos1 = _mm512_and_si512(x1, _mm512_shuffle_epi8(pos_lut, idx1));
            let neg1 = _mm512_and_si512(x1, _mm512_shuffle_epi8(neg_lut, idx1));
            let pos2 = _mm512_and_si512(x2, _mm512_shuffle_epi8(pos_lut, idx2));
            let neg2 = _mm512_and_si512(x2, _mm512_shuffle_epi8(neg_lut, idx2));
            let pos3 = _mm512_and_si512(x3, _mm512_shuffle_epi8(pos_lut, idx3));
            let neg3 = _mm512_and_si512(x3, _mm512_shuffle_epi8(neg_lut, idx3));

            let (pos_lo, pos_hi) = widen_sum_4_i8_avx512(pos0, pos1, pos2, pos3);
            let (neg_lo, neg_hi) = widen_sum_4_i8_avx512(neg0, neg1, neg2, neg3);

            acc16_lo = _mm512_add_epi16(acc16_lo, _mm512_sub_epi16(pos_lo, neg_lo));
            acc16_hi = _mm512_add_epi16(acc16_hi, _mm512_sub_epi16(pos_hi, neg_hi));

            chunks_since_flush += 1;
            if chunks_since_flush == 32 {
                acc32 = flush_i16_to_i32_avx512(acc32, acc16_lo);
                acc32 = flush_i16_to_i32_avx512(acc32, acc16_hi);
                acc16_lo = _mm512_setzero_si512();
                acc16_hi = _mm512_setzero_si512();
                chunks_since_flush = 0;
            }
        }

        if chunks_since_flush > 0 {
            acc32 = flush_i16_to_i32_avx512(acc32, acc16_lo);
            acc32 = flush_i16_to_i32_avx512(acc32, acc16_hi);
        }

        let mut acc_i32 = _mm512_reduce_add_epi32(acc32);
        if simd_bytes < row_bytes {
            acc_i32 += scalar_row_range_acc(
                row_slice,
                &qx.lane0,
                &qx.lane1,
                &qx.lane2,
                &qx.lane3,
                simd_bytes,
                row_bytes,
            );
        }

        out[r] = (acc_i32 as f32) * weight_scales[r] * act_scale;
    }
}

#[cfg(all(target_arch = "x86_64", terra_has_stable_avx512))]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn ternary_gemv_avx512(
    packed: &[u8],
    weight_scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    rms_prescale: bool,
    rms_eps: f32,
    out: &mut [f32],
) {
    let qx = QuantizedActivation::quantize(x, rms_prescale, rms_eps);
    ternary_gemv_avx512_prequantized(packed, weight_scales, &qx, rows, cols, out);
}

/// Returns the active SIMD kernel name on the current CPU.
pub fn active_simd_backend() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(terra_has_stable_avx512)]
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
            {
                return "avx512";
            }
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
        return "scalar";
    }
    #[allow(unreachable_code)]
    "scalar"
}

/// Hardware-dispatched ternary GEMV on pre-quantized activations.
#[inline]
pub fn ternary_gemv_prequantized(
    packed: &[u8],
    weight_scales: &[f32],
    qx: &QuantizedActivation,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        ternary_gemv_neon_prequantized(packed, weight_scales, qx, rows, cols, out);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(terra_has_stable_avx512)]
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
            {
                // SAFETY: feature detected above.
                unsafe {
                    ternary_gemv_avx512_prequantized(packed, weight_scales, qx, rows, cols, out);
                }
                return;
            }
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: feature detected above.
            unsafe {
                ternary_gemv_avx2_prequantized(packed, weight_scales, qx, rows, cols, out);
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    ternary_gemv_scalar_prequantized(packed, weight_scales, qx, rows, cols, out);
}

/// Row-partitioned [`ternary_gemv_prequantized`] for large matrices. Rows are independent, so the
/// output is bit-identical to the single-threaded kernel for any thread count.
pub fn ternary_gemv_prequantized_par(
    packed: &[u8],
    weight_scales: &[f32],
    qx: &QuantizedActivation,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    let threads = crate::kernels::num_threads();
    // Below ~4M weights (0.4 ms of work) the thread spawns cost more than they save.
    if threads <= 1 || rows * cols < (1 << 22) {
        ternary_gemv_prequantized(packed, weight_scales, qx, rows, cols, out);
        return;
    }
    assert_eq!(packed.len(), rows * cols / 4);
    assert_eq!(weight_scales.len(), rows);
    assert_eq!(out.len(), rows);
    let bytes_per_row = cols / 4;
    let rows_per = rows.div_ceil(threads);
    let mut blocks: Vec<&mut [f32]> = out.chunks_mut(rows_per).collect();
    crate::kernels::par_each_mut(&mut blocks, |b, block| {
        let r0 = b * rows_per;
        let n = block.len();
        ternary_gemv_prequantized(
            &packed[r0 * bytes_per_row..(r0 + n) * bytes_per_row],
            &weight_scales[r0..r0 + n],
            qx,
            n,
            cols,
            block,
        );
    });
}

/// Hardware-dispatched ternary GEMV (`[rows, cols] x [cols] -> [rows]`).
#[inline]
pub fn ternary_gemv(
    packed: &[u8],
    weight_scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    rms_prescale: bool,
    rms_eps: f32,
    out: &mut [f32],
) {
    let qx = QuantizedActivation::quantize(x, rms_prescale, rms_eps);
    ternary_gemv_prequantized(packed, weight_scales, &qx, rows, cols, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::quantize_pack_ternary_absmean;

    fn det_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 40) as f32) / ((1u32 << 24) as f32);
                (u * 2.0 - 1.0) * scale
            })
            .collect()
    }

    #[test]
    fn test_row_partitioned_kernel_bit_identical_to_serial() {
        // 2048 x 2048 crosses the threading threshold; 2047 rows leaves a ragged last block.
        for &rows in &[2048usize, 2047] {
            let cols = 2048;
            let w = det_vec(rows * cols, 11, 0.02);
            let (packed, scales) = quantize_pack_ternary_absmean(&w, rows, cols).unwrap();
            let x = det_vec(cols, 12, 1.5);
            let qx = QuantizedActivation::quantize(&x, false, 1e-5);
            let mut want = vec![0.0f32; rows];
            ternary_gemv_prequantized(&packed, &scales, &qx, rows, cols, &mut want);
            for threads in [1usize, 3, 10] {
                crate::kernels::set_num_threads(threads);
                let mut got = vec![0.0f32; rows];
                ternary_gemv_prequantized_par(&packed, &scales, &qx, rows, cols, &mut got);
                assert!(
                    got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "rows={rows} threads={threads}"
                );
            }
            crate::kernels::set_num_threads(0);
        }
    }

    #[test]
    fn test_simd_vs_scalar_bit_exact_across_shapes_and_edge_cases() {
        let shapes = [
            (4, 4),
            (7, 12),
            (19, 68),
            (64, 132),
            (128, 2080), // > 32 SIMD chunks -> exercises i16 -> i32 periodic flush!
            (1024, 2048),
            (2048, 1024),
        ];

        for &(rows, cols) in &shapes {
            let w = det_vec(rows * cols, (rows * 1000 + cols) as u64, 1.5);
            let (packed, scales) = quantize_pack_ternary_absmean(&w, rows, cols).unwrap();
            let x = det_vec(cols, (rows * 777 + cols) as u64, 3.0);

            for rms_prescale in [false, true] {
                let mut y_scalar = vec![0.0f32; rows];
                let mut y_dispatch = vec![0.0f32; rows];

                ternary_gemv_scalar(
                    &packed,
                    &scales,
                    &x,
                    rows,
                    cols,
                    rms_prescale,
                    1e-5,
                    &mut y_scalar,
                );
                ternary_gemv(
                    &packed,
                    &scales,
                    &x,
                    rows,
                    cols,
                    rms_prescale,
                    1e-5,
                    &mut y_dispatch,
                );

                for r in 0..rows {
                    assert_eq!(
                        y_scalar[r].to_bits(),
                        y_dispatch[r].to_bits(),
                        "bit mismatch at shape ({rows},{cols}) row {r} (rms={rms_prescale}): scalar={} vs dispatch={}",
                        y_scalar[r],
                        y_dispatch[r]
                    );
                }

                #[cfg(target_arch = "aarch64")]
                {
                    let mut y_neon = vec![0.0f32; rows];
                    ternary_gemv_neon(
                        &packed,
                        &scales,
                        &x,
                        rows,
                        cols,
                        rms_prescale,
                        1e-5,
                        &mut y_neon,
                    );
                    for r in 0..rows {
                        assert_eq!(
                            y_scalar[r].to_bits(),
                            y_neon[r].to_bits(),
                            "NEON bit mismatch at ({rows},{cols}) row {r}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_zero_and_extreme_saturation_no_i16_overflow() {
        // K = 8192 = 2048 bytes = 128 NEON chunks (4x flush cycles!), all +1 weights and all +127 activations
        let rows = 8;
        let cols = 8192;
        let packed_all_pos = vec![0b01_01_01_01u8; rows * (cols / 4)];
        let packed_all_neg = vec![0b10_10_10_10u8; rows * (cols / 4)];
        let scales = vec![0.5f32; rows];
        let x_ones = vec![1.0f32; cols];
        let x_zeros = vec![0.0f32; cols];

        // 1. Zero input -> exact 0.0
        let mut out_zero = vec![123.0f32; rows];
        ternary_gemv(
            &packed_all_pos,
            &scales,
            &x_zeros,
            rows,
            cols,
            false,
            1e-5,
            &mut out_zero,
        );
        assert!(out_zero.iter().all(|&v| v == 0.0));

        // 2. All +1 weights * all +1.0 inputs (quantized to +127, scale = 1/127):
        // acc_i32 = 8192 * 127 = 1,040,384 (far exceeds i16::MAX = 32,767!)
        // y = 1_040_384 * 0.5 * (1.0 / 127.0) = 4096.0
        let mut out_pos_simd = vec![0.0f32; rows];
        let mut out_pos_scalar = vec![0.0f32; rows];
        ternary_gemv(
            &packed_all_pos,
            &scales,
            &x_ones,
            rows,
            cols,
            false,
            1e-5,
            &mut out_pos_simd,
        );
        ternary_gemv_scalar(
            &packed_all_pos,
            &scales,
            &x_ones,
            rows,
            cols,
            false,
            1e-5,
            &mut out_pos_scalar,
        );
        for r in 0..rows {
            assert_eq!(out_pos_simd[r].to_bits(), out_pos_scalar[r].to_bits());
            assert!((out_pos_simd[r] - 4096.0).abs() < 1e-3, "got {}", out_pos_simd[r]);
        }

        // 3. All -1 weights * all +1.0 inputs -> -4096.0
        let mut out_neg_simd = vec![0.0f32; rows];
        let mut out_neg_scalar = vec![0.0f32; rows];
        ternary_gemv(
            &packed_all_neg,
            &scales,
            &x_ones,
            rows,
            cols,
            false,
            1e-5,
            &mut out_neg_simd,
        );
        ternary_gemv_scalar(
            &packed_all_neg,
            &scales,
            &x_ones,
            rows,
            cols,
            false,
            1e-5,
            &mut out_neg_scalar,
        );
        for r in 0..rows {
            assert_eq!(out_neg_simd[r].to_bits(), out_neg_scalar[r].to_bits());
            assert!((out_neg_simd[r] + 4096.0).abs() < 1e-3, "got {}", out_neg_simd[r]);
        }
    }

    #[test]
    fn test_round_ties_to_even_subthreshold_zero_and_clamp_127_edges() {
        // 1. Sub-threshold `< 1e-12` activation vector must set `is_zero = true` and emit exact 0.0 without NaN
        let sub_thresh = [0.0f32, 9.5e-13, -8.0e-13, 1.0e-15, 0.0, 0.0, 0.0, 0.0];
        let q_zero = QuantizedActivation::quantize(&sub_thresh, false, 1e-5);
        assert!(q_zero.is_zero);
        assert_eq!(q_zero.act_scale, 0.0);

        let packed = vec![0b01_10_01_00u8; 4 * 2];
        let scales = vec![1.25f32; 4];
        let mut out = vec![99.0f32; 4];
        ternary_gemv_prequantized(&packed, &scales, &q_zero, 4, 8, &mut out);
        assert!(out.iter().all(|&v| v.to_bits() == 0.0f32.to_bits()));

        // 2. Exact IEEE round-half-to-even & +-127 boundary:
        // Set max_abs = 127.0 so act_scale = 127.0 / 127.0 == 1.0 exactly!
        // Then normalized values equal raw inputs:
        //   0.5 -> 0 (tie to even)
        //   1.5 -> 2 (tie to even)
        //   2.5 -> 2 (tie to even)
        //   3.5 -> 4 (tie to even)
        //  -0.5 -> 0 (tie to even)
        //  -2.5 -> -2 (tie to even)
        //  127.0 -> 127 (upper clamp)
        // -127.0 -> -127 (lower clamp)
        let x_ties = [0.5f32, 1.5, 2.5, 3.5, -0.5, -2.5, 127.0, -127.0];
        let q_ties = QuantizedActivation::quantize(&x_ties, false, 1e-5);
        assert!(!q_ties.is_zero);
        assert_eq!(q_ties.act_scale, 1.0);
        assert_eq!(q_ties.lane0, vec![0i8, 0]);
        assert_eq!(q_ties.lane1, vec![2i8, -2]);
        assert_eq!(q_ties.lane2, vec![2i8, 127]);
        assert_eq!(q_ties.lane3, vec![4i8, -127]);
    }
}

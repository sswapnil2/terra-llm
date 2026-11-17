//! Compute kernels for Micro-TERRA inference (`ternary_simd` + norm / activation / router ops).

pub mod ternary_simd;

use half::f16;

/// Applies Section 4.1 `RMSNorm`: `out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`.
#[inline]
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), out.len());
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let inv_rms = 1.0f64 / (sum_sq / n + eps as f64).sqrt();
    for i in 0..x.len() {
        out[i] = (((x[i] as f64) * inv_rms) as f32) * weight[i];
    }
}

/// Applies `RMSNorm` in place.
#[inline]
pub fn rms_norm_in_place(x: &mut [f32], weight: &[f32], eps: f32) {
    assert_eq!(x.len(), weight.len());
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let inv_rms = 1.0f64 / (sum_sq / n + eps as f64).sqrt();
    for (v, &w) in x.iter_mut().zip(weight.iter()) {
        *v = (((*v as f64) * inv_rms) as f32) * w;
    }
}

/// SiLU (`z / (1 + exp(-z))`).
#[inline]
pub fn silu(z: f32) -> f32 {
    z / (1.0f32 + (-z).exp())
}

/// Sigmoid (`1 / (1 + exp(-z))`).
#[inline]
pub fn sigmoid(z: f32) -> f32 {
    1.0f32 / (1.0f32 + (-z).exp())
}

/// Numerically stable softmax in `f32`.
pub fn softmax_in_place(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0f32 / sum;
        for v in logits.iter_mut() {
            *v *= inv;
        }
    }
}

/// Selects top-`k` `(index, value)` pairs sorted descending by value, breaking ties by lowest index.
pub fn top_k_with_ties(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    assert!(k <= values.len());
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(k);
    indexed
}

/// Dense `F32` matrix-vector multiply (`[rows, cols] x [cols] -> [rows]`), used for small routers / decay heads.
pub fn f32_gemv(weights: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    assert_eq!(weights.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    for r in 0..rows {
        let row = &weights[r * cols..(r + 1) * cols];
        let mut acc = 0.0f64;
        for c in 0..cols {
            acc += (row[c] as f64) * (x[c] as f64);
        }
        out[r] = acc as f32;
    }
}

/// Dense `F16` (stored as raw IEEE-754 `u16` bits) matrix-vector multiply (`[rows, cols] x [cols] -> [rows]`).
pub fn f16_gemv(weight_bits: &[u16], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    assert_eq!(weight_bits.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    for r in 0..rows {
        let row = &weight_bits[r * cols..(r + 1) * cols];
        let mut acc = 0.0f64;
        for c in 0..cols {
            acc += (f16::from_bits(row[c]).to_f32() as f64) * (x[c] as f64);
        }
        out[r] = acc as f32;
    }
}

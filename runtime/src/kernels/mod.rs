//! Compute kernels for Micro-TERRA inference (`ternary_simd` + norm / activation / router ops).

pub mod ternary_simd;

use std::sync::atomic::{AtomicUsize, Ordering};

use half::f16;
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};

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

// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Sets the compute thread count (`0` = one per available core). Process-wide.
pub fn set_num_threads(n: usize) {
    NUM_THREADS.store(n, Ordering::Relaxed);
}

/// Compute threads used by [`par_each_mut`].
pub fn num_threads() -> usize {
    match NUM_THREADS.load(Ordering::Relaxed) {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    }
}

/// Runs `f(i, &mut items[i])` for every item across up to [`num_threads`] scoped threads.
/// Each item is touched by exactly one thread, so results are independent of the thread count.
// ponytail: spawns scoped threads per call (~10-20 us each); swap for a persistent pool if a
// profile ever shows spawn cost mattering (it is <2% of a token at OLMoE shapes).
pub fn par_each_mut<T: Send>(items: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
    let n = num_threads().min(items.len());
    if n <= 1 {
        for (i, it) in items.iter_mut().enumerate() {
            f(i, it);
        }
        return;
    }
    let per = items.len().div_ceil(n);
    let f = &f;
    std::thread::scope(|s| {
        let mut chunks = items.chunks_mut(per).enumerate();
        let first = chunks.next();
        for (c, chunk) in chunks {
            s.spawn(move || {
                for (j, it) in chunk.iter_mut().enumerate() {
                    f(c * per + j, it);
                }
            });
        }
        if let Some((_, chunk)) = first {
            for (j, it) in chunk.iter_mut().enumerate() {
                f(j, it);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// F16 kernels
// ---------------------------------------------------------------------------

/// Decodes raw IEEE-754 `f16` bits to `f32` (exact; hardware-converted where the CPU supports it).
#[inline]
pub fn f16_bits_to_f32(bits: &[u16], out: &mut [f32]) {
    bits.reinterpret_cast::<f16>().convert_to_f32_slice(out);
}

/// `f32` dot product over 16 independent accumulators (auto-vectorizes; pairwise-style rounding).
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (xa, xb) in ca.zip(cb) {
        for i in 0..16 {
            acc[i] += xa[i] * xb[i];
        }
    }
    let mut tail = 0.0f32;
    for (&x, &y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    let mut w = 16;
    while w > 1 {
        w /= 2;
        for i in 0..w {
            acc[i] += acc[i + w];
        }
    }
    acc[0] + tail
}

/// Reference `F16` GEMV: software decode, `f64` accumulation. Kept as the oracle for [`f16_gemv`].
pub fn f16_gemv_reference(weight_bits: &[u16], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
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

/// Dense `F16` (stored as raw IEEE-754 `u16` bits) matrix-vector multiply (`[rows, cols] x [cols] -> [rows]`).
/// Rows are decoded to `f32` exactly, accumulated in `f32`, and partitioned across threads.
pub fn f16_gemv(weight_bits: &[u16], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    assert_eq!(weight_bits.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    if rows == 0 {
        return;
    }
    // Below ~1M weights the thread spawn costs more than it saves.
    let blocks = if rows * cols < (1 << 20) { 1 } else { num_threads() };
    let rows_per = rows.div_ceil(blocks);
    let mut row_blocks: Vec<&mut [f32]> = out.chunks_mut(rows_per).collect();
    par_each_mut(&mut row_blocks, |b, block| {
        let mut wrow = vec![0.0f32; cols];
        for (j, o) in block.iter_mut().enumerate() {
            let r = b * rows_per + j;
            f16_bits_to_f32(&weight_bits[r * cols..(r + 1) * cols], &mut wrow);
            *o = dot_f32(&wrow, x);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn test_f16_gemv_matches_f64_reference_across_shapes_and_threads() {
        // Shapes cover: remainder columns (not a multiple of 16), single row, and the threaded path.
        for &(rows, cols) in &[(1usize, 1usize), (3, 17), (7, 128), (700, 2048)] {
            let mut seed = (rows * 31 + cols) as u64;
            let mut bits: Vec<u16> = (0..rows * cols)
                .map(|_| f16::from_f32(lcg(&mut seed) * 0.05).to_bits())
                .collect();
            // Edge values: signed zero, subnormal, largest finite.
            bits[0] = f16::from_f32(-0.0).to_bits();
            if bits.len() > 2 {
                bits[1] = 0x0001;
                bits[2] = f16::MAX.to_bits();
            }
            let x: Vec<f32> = (0..cols).map(|_| lcg(&mut seed) * 3.0).collect();
            let mut want = vec![0.0f32; rows];
            f16_gemv_reference(&bits, &x, rows, cols, &mut want);

            let mut prev: Option<Vec<f32>> = None;
            for threads in [1usize, 2, 5] {
                set_num_threads(threads);
                let mut got = vec![0.0f32; rows];
                f16_gemv(&bits, &x, rows, cols, &mut got);
                for r in 0..rows {
                    // |x| <= 3, |w| mostly <= 0.05: f32 accumulation error is far below this bound.
                    let tol = 1e-4f32 * (1.0 + want[r].abs());
                    assert!(
                        (got[r] - want[r]).abs() <= tol,
                        "{rows}x{cols} t={threads} row {r}: {} vs {}",
                        got[r],
                        want[r]
                    );
                }
                // Thread count must not change a single bit.
                if let Some(p) = &prev {
                    assert!(p.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()));
                }
                prev = Some(got);
            }
            set_num_threads(0);
        }
    }

    /// Microbench at the real OLMoE output-head shape. Run with:
    /// `cargo test --release bench_f16_gemv_lm_head_shape -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn bench_f16_gemv_lm_head_shape() {
        let (rows, cols) = (50304usize, 2048usize);
        let mut seed = 7u64;
        let bits: Vec<u16> = (0..rows * cols)
            .map(|_| f16::from_f32(lcg(&mut seed) * 0.05).to_bits())
            .collect();
        let x: Vec<f32> = (0..cols).map(|_| lcg(&mut seed)).collect();
        let time = |f: &dyn Fn(&mut [f32])| {
            let mut out = vec![0.0f32; rows];
            f(&mut out); // warm-up
            let t = std::time::Instant::now();
            for _ in 0..5 {
                f(&mut out);
            }
            (t.elapsed().as_secs_f64() * 1000.0 / 5.0, out)
        };
        let (ref_ms, want) = time(&|o| f16_gemv_reference(&bits, &x, rows, cols, o));
        println!("f16_gemv {rows}x{cols}: reference (scalar, f64) {ref_ms:.2} ms");
        for threads in [1usize, 2, 4, 6, 8, 0] {
            set_num_threads(threads);
            let (ms, got) = time(&|o| f16_gemv(&bits, &x, rows, cols, o));
            let max_diff = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!(
                "f16_gemv {rows}x{cols}: threads={} {ms:.2} ms, max|diff| vs reference {max_diff:.3e}",
                num_threads()
            );
        }
        set_num_threads(0);
    }

    #[test]
    fn test_par_each_mut_visits_every_index_once() {
        for n in [0usize, 1, 2, 7, 64] {
            let mut v = vec![usize::MAX; n];
            par_each_mut(&mut v, |i, it| *it = i);
            assert!(v.iter().enumerate().all(|(i, &x)| i == x));
        }
    }
}

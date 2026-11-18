# Micro-TERRA — Implementation Report for `ARCHITECTURE_REVIEW.md`

Date: 2026-11-18. Base commit: `251f0b2` (branch `clean-val-anchor-ablation`). Changes are uncommitted in the working tree.
This report is written by the implementer. **Nothing here is independently verified.**

Evidence labels: **[MEASURED]** = run in this session on the machine below; **[CODE]** = by reading source; **[ESTIMATE]** = reasoning only.

**Measurement conditions (read before using any number).** Apple M2 Pro (6P+4E), 16 GB, macOS (Darwin 25.4), rustc 1.85.1 `--release`, NEON backend, `F_NOCACHE + pread` pool. The machine was under heavy unrelated load for the whole session (load average 23–40, another project's build running). Throughput figures are therefore best-of-3 and noisy; individual repeats varied by up to 10×. Microbenchmarks and all correctness results are unaffected. Model: the repo's full-size synthetic export (`--synthetic`, seed 42: random weights, ternary `lm_head`, F32 embeddings, 317 MB core, 1,024 experts). No trained checkpoint was available, so **no quality number (perplexity, recall on real routing, HumanEval) was measured**.

---

## 1. Summary

Scope decision: I implemented every finding that is a verified defect and can be validated on this machine without a trained checkpoint, and I did not start any change that alters the research architecture, the binary format, or training semantics for existing checkpoints. Those are listed in §9 with a proposal each and need an owner decision.

| Result | Before | After | Basis |
|---|---|---|---|
| f16 output head, 50,304 × 2,048 | 98.3 ms (scalar, f64) | 18.3 ms (1 thread), 5.2 ms (10 threads) | [MEASURED] microbench |
| Decode tok/s, all experts resident, 201 tokens | 14.5 (best of 3) | 16.0 @1 thread, 38.6 @4–6 threads (best of 3) | [MEASURED], noisy |
| Anchor attention per layer @4,096 ctx | ~24 ms [review's ESTIMATE] | 4.1 ms (1 thread), 3.1 ms (10 threads) | [MEASURED] microbench |
| "Lookahead" credit, random router, 2 GB cold start | 428 prefetched hits (hit rate 80.3 → 88.7%) | 57 same-token hits; recall@8 12.45%, precision 12.7% (chance = 12.5%) | [MEASURED] |
| 8-bit Adam update on a zero-gradient step after a row spike | 194× the FP32 AdamW update | ≤ 1.76× | [MEASURED] |
| Tests | 15 Rust, 59 Python (58 pass + 1 timing flake) | 21 Rust (+2 ignored benches), 63 Python, all pass | [MEASURED] |

The 15–30 tok/s target is met on this laptop for the synthetic model with experts resident and a short context. That is not a claim about the real export (f16 head, real routing, 4k context), edge-class hardware, or model quality.

## 2. Architecture changes

None to the model, the binary format, or the training objective. Runtime execution changed in three ways:

1. **Threading** (`kernels::par_each_mut`, std scoped threads, no new dependency). The top-k experts of a layer run one per thread; large GEMVs (≥ 4M weights ternary, ≥ 1M weights f16) are row-partitioned; anchor attention is partitioned by head above a context threshold. Every unit of work is owned by one thread and sums are taken in router order, so results do not depend on thread count. `terra-cli --threads N` (0 = all cores).
2. **Prefetch accounting** now has a token epoch (`ExpertLruPool::begin_token`). Stale prefetches lose their eviction protection after the token that issued them. Cache policy affects timing only, never values.
3. **Anchor attention** accumulates the q·k dot product in f32 (16 lanes) instead of sequential f64. This is the only change that alters numerics (§4).

## 3. Files modified

| File | Change |
|---|---|
| `runtime/src/kernels/mod.rs` | `par_each_mut`, `set_num_threads`, `f16_bits_to_f32`, `dot_f32`, threaded `f16_gemv`; old kernel kept as `f16_gemv_reference` (test oracle) |
| `runtime/src/kernels/ternary_simd.rs` | `ternary_gemv_prequantized_par` |
| `runtime/src/engine.rs` | parallel expert loop; lookahead recall counters; context-clamp warning; new `BenchmarkReport` fields (`lookahead_recall_at_k_pct`, `prefetch_precision_pct`, `prefetch_wasted_bytes`, `peak_rss_bytes`, `threads`, `kv_reserved_tokens`) |
| `runtime/src/nvme_pool.rs` | `PrefetchMetrics` reclassified (`prefetched_hits` same-token only, `stale_prefetch_hits`, `prefetch_late`, `prefetch_bytes_read`, `prefetch_useful_bytes`, `validate_us`); warning when unbuffered I/O is unavailable |
| `runtime/src/state.rs` | f32 SIMD-friendly, head-parallel anchor attention; removed shared score scratch |
| `runtime/src/bin/terra_cli.rs` | `--threads`; prints new metrics |
| `runtime/src/lib.rs` | 2 new integration tests; 1 existing test adapted to the new counters |
| `runtime/examples/dump_logits.rs` | new: dumps logits for a fixed token sequence (cross-build equivalence check) |
| `train/bitlinear.py` | explicit L2-scale option at λ=1; model prescale made idempotent; docstring corrected |
| `train/distill_colab.py` | verifier scores ternary with the L2-optimal scale before the prescale; optimizer zero-code fix; prescale record restored on resume |
| `train/tests/test_review_fixes.py` | new: 4 tests |
| `bench/run_benchmarks.py` | reports/asserts OS peak RSS; new metric columns; defaults 200 new tokens, model's own context; 16k run relabelled as a reservation check |
| `ARCHITECTURE.md` | wrong figures corrected in place, unverified ones marked |
| `bench/results/2026-11-18_m2pro_synthetic_raw.jsonl` | raw benchmark reports for every run quoted here |

`uv.lock` was already modified before this work and was not touched.

## 4. Mathematical / numerical changes

- **Anchor dot product, f64 → f32.** Verified to be the *only* numerical difference from `251f0b2`: with the f64 dot temporarily restored, logits of the full-size model were bit-identical to the baseline build over 24 tokens (so threading, the expert loop and the pool changes are bit-exact) [MEASURED]. With the f32 dot, 22 of 24 tokens differ by ≤ 9.5e-7 and 2 tokens by 2.6e-2 / 2.7e-2 (logit std 0.95); argmax agreed on 24/24. The larger differences are a 1-ulp change crossing an int8 activation rounding boundary downstream, which the quantized architecture amplifies by construction. Against an independent f64 implementation of softmax attention the new anchor path is within 2e-5 over 300 tokens (test). Trade-off accepted: the f64 variant measured 6.2–8.8 ms per anchor layer at 4k context vs 3.1–4.1 ms. If bit-exactness with the old build matters more than ~5 ms/token at 4k, it is a one-line revert.
- **f16 GEMV** decodes exactly (hardware where available) and accumulates in f32: max |diff| vs the f64 reference 8.0e-7 at the output-head shape [MEASURED].
- **Ternary validation (finding 14).** At λ=1 on un-prescaled weights the verifier used γ = absmean; it now uses γ* = ⟨w,t⟩/‖t‖², the scale the ramp blends toward and the one the model has after the prescale. Test shows the scored function equals the post-prescale function (rtol 1e-4) and that the old one was a different function (output norm ratio 1.15–1.5).
- **8-bit Adam (finding 16).** `v_sqrt` code 0 was decoded as exactly 0, so a non-zero momentum could be divided by ε alone. It is now decoded as half a quantization step (the top of the interval code 0 represents).
- **Prescale (finding 17).** `apply_l2_optimal_row_prescale_` is not idempotent (the rescaled row is not its own fixed point); the docstring said otherwise. The model-level function now records per module that it ran and skips on repeat.

## 5. Tests added

Rust: `test_f16_gemv_matches_f64_reference_across_shapes_and_threads`, `test_par_each_mut_visits_every_index_once`, `test_row_partitioned_kernel_bit_identical_to_serial`, `test_anchor_attention_matches_f64_reference`, `test_prefetch_credit_requires_same_token_use`, `test_lookahead_credit_is_bounded_by_prediction_recall`; ignored benches `bench_f16_gemv_lm_head_shape`, `bench_anchor_attention`.
Python: `test_ternary_validation_is_continuous_through_the_prescale_step`, `test_model_prescale_is_idempotent_and_row_prescale_is_not`, `test_8bit_adam_never_divides_momentum_by_eps_alone`, `test_8bit_adam_tracks_fp32_adamw_on_a_small_regression`.

## 6. Tests executed and actual results

| Command | Result |
|---|---|
| Baseline `cargo test --release` | 15 passed |
| Baseline `uv run pytest -q` | 58 passed, **1 failed**: `exporter/test_packing.py::test_synthetic_1024_expert_olmoe_cli_speed_and_4kb_alignment` (asserts export wall time < 8 s; ran concurrently with the Rust suite). Pre-existing timing flake, passed on the final run |
| Final `cargo test --release` | 21 passed, 0 failed, 2 ignored (benches) — includes PyTorch-exporter golden parity at 1e-2 and the engine bit-identity tests |
| Final `uv run pytest -q` | 63 passed |

Not run: x86 AVX2/AVX-512 paths, Linux `io_uring`/`O_DIRECT`, any training, any real-checkpoint evaluation.

## 7. Performance before/after [MEASURED, conditions above]

Decode, 8 prompt + 201 new tokens, 2,048 MiB budget, all 1,024 experts cached, no lookahead, context ≤ 210:

| Build | tok/s, 3 repeats | Best |
|---|---|---|
| Baseline `251f0b2` (1 thread) | 4.5 / 6.3 / 14.5 | 14.5 |
| New, 1 thread | 8.0 / 4.4 / 16.0 | 16.0 |
| New, 2 threads | 16.8 / 6.6 / 19.9 | 19.9 |
| New, 4 threads | 38.6 / 3.6 / 32.0 | 38.6 |
| New, 6 threads | 38.6 / 21.7 / 9.2 | 38.6 |
| New, 8 threads | 32.7 / 29.1 / 14.3 | 32.7 |
| New, 10 threads | 29.7 / 18.0 / 12.5 | 29.7 |

The spread is machine contention, not the code; these need re-running on an idle machine before anyone quotes them. The fall-off above 6 threads is consistent with 6 performance cores but is not established by this data.

Lookahead with a router that has no predictive power (synthetic model):

| Run | Lookahead | tok/s | Cache hit % | Recall@8 % | Prefetch precision % | MiB read / token |
|---|---|---|---|---|---|---|
| 2 GB, 8+33 tokens (cold) | on | 19.6 | 88.8 | 12.5 | 12.7 | 39 |
| 2 GB, 8+33 tokens (cold) | off | 17.1 | 80.1 | — | — | 39 |
| 1 GB (452 slots), 8+101 tokens | on | 10.2 | 44.9 | 12.3 | 12.3 | 199 |
| 1 GB (452 slots), 8+101 tokens | off | 12.0 | 42.8 | — | — | 111 |

The cache hit rate still rises with a random router (cache warming), exactly as the review found; the difference is that the report no longer attributes it to prediction. At 1 GB the random router read 79% more data and wasted 8.7 GiB of speculative reads over 109 tokens for no speedup.

## 8. Memory before/after [MEASURED]

Self-accounted resident bytes are unchanged (1,982 MiB at the 2,048 MiB budget with this 317 MB synthetic core). OS peak RSS, newly reported: 1,895–1,899 MiB at the 2,048 budget and 1,025–1,026 MiB at the 1,024 budget. **The 1 GB runs exceed their budget by 1–2 MiB of RSS**, which is the out-of-budget scratch/stack/allocator memory the review predicted; `bench/run_benchmarks.py` now fails on this. Some runs showed lower RSS (1,531–1,762 MiB) because macOS was compressing memory under pressure, so macOS RSS can under-read. Threads add stacks outside the budget (not separately measured). No cgroup- or `ulimit`-enforced run was done (needs Linux).

## 9. Status of every review finding

| # | Finding | Status | Notes |
|---|---|---|---|
| 1 | No result artifacts; contaminated perplexities | **BLOCKED** | Needs the checkpoint and a GPU re-run on the fixed split. Doc figures now marked unverified |
| 2 | No dense 4-bit baseline at equal RAM | **PARTIALLY IMPLEMENTED** | Baselines measured (`EXPERIMENT_RESULTS.md` §1): the bar is 0.735 / 0.738 bits/byte. The student itself has not been scored against it |
| 3 | Throughput: single-threaded, slow f16 head | **IMPLEMENTED** | §7. Real-export end-to-end tok/s not measured (no real export here) |
| 4 | Hit-rate metric credits random prefetches | **IMPLEMENTED** | Random router scores 12.45% recall / 12.7% precision vs 12.5% chance |
| 5 | NVMe pillar moot at 2 GB | **NOT IMPLEMENTED** | Deleting a pillar is an owner decision. Now confirmed on real routing traces: 0.15% miss rate at 2,048 MiB (`EXPERIMENT_RESULTS.md` §4) |
| 6 | 1 GB is bandwidth-bound | **PARTIALLY IMPLEMENTED** | Re-measured with honest metrics (synthetic). Trace-driven simulator built and run on 512k tokens of teacher routing (`EXPERIMENT_RESULTS.md` §4): 91 MiB/token at 1,024 MiB. It lives in the Colab script, not in the runtime |
| 7 | Context capped at 4,096; KV figure wrong | **PARTIALLY IMPLEMENTED** | Doc corrected; clamp now warns and `kv_reserved_tokens` is reported; 16k bench relabelled. Context target itself undecided |
| 8 | Recurrence is the largest quality loss | **PARTIALLY IMPLEMENTED** | Sliding-window control run (`EXPERIMENT_RESULTS.md` §3). The trained student's recurrence loss still needs a checkpoint |
| 9 | (1−α) normalization attenuates long heads | **NOT IMPLEMENTED** | Changing the init changes what Stage 1 trains; documented as a known limitation in `ARCHITECTURE.md`. Proposal: per-head output gain initialised to √((1+α)/(1−α)) behind a flag, validated by per-head output RMS at step 0 |
| 10 | Ternary vs INT4 error | **PARTIALLY IMPLEMENTED** | Untrained int4 / int3 / ternary reference and per-layer sweep measured (`EXPERIMENT_RESULTS.md` §2). No QAT comparison |
| 11 | Hints not dereferenceable; unused in Rust generation | **NOT IMPLEMENTED** | Ablation needs a checkpoint. Prose "ONNX model" description corrected in the doc |
| 12 | Lookahead recall below gate; doc describes a non-existent MLP | **PARTIALLY IMPLEMENTED** | Doc corrected; runtime now reports cache-independent recall@k. Popularity, previous-token, token-id and linear-probe baselines measured on the teacher (`EXPERIMENT_RESULTS.md` §4); the student's own router is not evaluated |
| 13 | Eval harness underpowered | **NOT IMPLEMENTED** | Cannot be validated without a model; should be done together with #1 |
| 14 | Ternary validation scale mismatch during ramp | **IMPLEMENTED** | Curve continuity itself can only be confirmed on a real run |
| 15 | NF4 teacher | **PARTIALLY IMPLEMENTED** | BF16 teacher reference measured (`EXPERIMENT_RESULTS.md` §1, §5). NF4 loss not measured; training still uses NF4 |
| 16 | 8-bit Adam divide-by-ε | **IMPLEMENTED** | Upgraded from "unverified risk" to reproduced bug, then fixed. See limitation L5 |
| 17 | Prescale documented idempotent | **IMPLEMENTED** | |
| 18 | Layers 0–2 never prefetched | **NOT IMPLEMENTED** | Depends on #5. A token-id table reaches 39–56% recall on these layers (`EXPERIMENT_RESULTS.md` §4) |
| 19 | 5-trit packing | **NOT IMPLEMENTED** | Binary-format and kernel change; needs approval |
| 20 | `core.bin` size figure | **IMPLEMENTED** | Doc |
| 21 | No `mlock`; silent `O_DIRECT` fallback; self-reported RAM | **PARTIALLY IMPLEMENTED** | Fallback warns; peak RSS reported and asserted; doc no longer says "pinned". No `mmap`/`mlock`, no cgroup test |
| 22 | No batched prefill | **NOT IMPLEMENTED** | TTFT benefits from threading only |
| 23 | Per-token allocations, f64 norms, full sort | **NOT IMPLEMENTED** | P3; the threaded paths add small per-call allocations |
| 24 | Abs-max activation scaling on SwiGLU output | **NOT IMPLEMENTED** | Needs activations from a real model |
| 25 | Hint offset bucket aliasing | **NOT IMPLEMENTED** | Fix changes the meaning of trained hint embeddings and must change in Python and Rust together |

## 10. Known limitations

- **L1.** All throughput numbers were taken on a contended machine and on synthetic weights with a ternary head and short context.
- **L2.** Threads are spawned per parallel region (scoped threads). Fine at these shapes; a persistent pool would cut overhead further. Marked in code.
- **L3.** The anchor f32 change makes logits differ from the old build by up to ~3e-2 on some tokens (§4).
- **L4.** `test_lookahead_credit_is_bounded_by_prediction_recall` checks a bound, not chance level: the small Rust fixture deliberately correlates its lookahead weights with the routers. The chance-level result is from the full-size export and is not an automated test (1.9 GB fixture).
- **L5.** The 8-bit optimizer still deviates from FP32 AdamW in the opposite direction: an element whose momentum is < 1/254 of its row maximum is stored as zero, so on a zero-gradient step it does not move at all where AdamW would [MEASURED]. Bounded and conservative, but it is not AdamW. No two-seed or FP32-reference training comparison was run.
- **L6.** The prescale record is a Python attribute, not in `state_dict`; resume restores it from saved metrics.
- **L7.** New runtime code was compiled and tested on aarch64/macOS only.

## 11. Remaining risks

The review's central risks are untouched by this work: no evidence the model retains useful quality, no equal-RAM dense baseline, recurrence capacity and normalization, and an eval harness too small to detect differences. Making the runtime fast does not address any of them, and the review's own ordering puts findings 1, 2 and 13 first.

## 12. Reproduction and verifier handoff

```bash
cd runtime && cargo test --release
```
```bash
uv run pytest -q
```
```bash
cd runtime && cargo test --release bench_f16_gemv_lm_head_shape -- --ignored --nocapture --test-threads=1
```
```bash
cd runtime && cargo test --release bench_anchor_attention -- --ignored --nocapture --test-threads=1
```
```bash
uv run python -m exporter.export_olmoe --synthetic --out-dir /tmp/terra_syn
```
```bash
runtime/target/release/terra-cli --model-dir /tmp/terra_syn --ram-budget-mb 2048 --max-seq-len 512 --prompt-tokens 1,42,17,99,128,256,512,1024 --max-new-tokens 201 --no-lookahead --threads 6 --benchmark-json
```

Chance-level lookahead check: same command with `--max-new-tokens 33` and without `--no-lookahead`; expect `lookahead_recall_at_k_pct` and `prefetch_precision_pct` near 12.5.

Cross-build equivalence: build `runtime/examples/dump_logits.rs` at `251f0b2` and at this tree (`cargo build --release --example dump_logits`), run each as `dump_logits <model_dir> <out.bin> 24`, and compare the raw f32 files.

Suggested checks for the verifier, in order of value: (1) re-run §7 on an idle machine, and on Linux/x86; (2) run the real export with an f16 head end to end; (3) run a 1 GB budget under a cgroup limit; (4) confirm on a real training run that `val_ppl_ternary` is continuous across the prescale step; (5) review the anchor f32 trade-off in §4.

# Project Micro-TERRA — Independent Architecture & Implementation Review

Reviewed: `ARCHITECTURE.pdf` (identical in content to `ARCHITECTURE.md`), `ANTIGRAVITY.md`, and the repository at `251f0b2` (branch `clean-val-anchor-ablation`).
Method: Pass 1 read the design and listed its scientific weaknesses. Pass 2 traced each one into the code, ran both test suites, and measured the runtime on a full-size synthetic model.

**Evidence labels used throughout**

| Label | Meaning |
|---|---|
| **[CODE]** | Verified by reading the source |
| **[MEASURED]** | I ran it in this review (Apple M2 Pro, 16 GB, macOS, other apps running) |
| **[MATH]** | Follows from a derivation given here |
| **[DOC-ONLY]** | Stated in the design doc; no artifact in the repo supports it |
| **[ESTIMATE]** | My reasoning, not measured |

**What I did not do.** I did not run training, and I do not have the distilled checkpoint, so every quality number (perplexity, recall, HumanEval) is unverified by me. I did not run on Linux, so `io_uring`/`O_DIRECT` behaviour is reviewed from code only. The synthetic model has random weights, so its expert routing is close to uniform — the worst case for cache locality. Throughput on compute is representative; cache hit rates are not.

---

## 1. Executive Summary

Micro-TERRA is five research projects stacked on one another, each with its own unproven quality cost, and the repository contains no result artifact for any of them. Three findings change the picture more than the rest:

1. **At the headline 2 GB budget the model almost entirely fits in RAM, so the NVMe pillar has nothing to do.** 1,024 experts at 2 bits/weight are 1,552 MiB. With an f16 core (~472 MiB by my arithmetic) roughly 95% of experts are resident at a 4k context. On the synthetic model all 1,024 were cached with zero evictions **[MEASURED]**. Lookahead routing, `io_uring`, and the LRU pool only matter below about 1.5 GB.
2. **The measured throughput is 3–10× below the claim.** Best case here is 9.3 tok/s (single thread, every expert resident, ternary output head). The real export uses an f16 output head that costs 201 ms per token by itself **[MEASURED]**, capping decode under 5 tok/s. The runtime has no threading **[CODE]**. The doc's 30.5–31.2 tok/s is **[DOC-ONLY]**.
3. **The evidence for model quality is not trustworthy as written.** Commit `e404655` states that validation "was measured on trained-on data"; the doc's perplexities (7.37, 9.98) reference the step-2250 run that predates that fix. The doc's cache-hit improvement (80% → 89.4%) reproduces almost exactly with a *random* lookahead router (80.3% → 88.7%) **[MEASURED]**, because the metric counts any prefetched expert that is eventually used as a hit.

The engineering is careful where it exists: the binary format, the SIMD kernel, the buffer-conservation logic, and the test suites (15 Rust + 58 Python tests pass **[MEASURED]**) are solid. The problem is that careful engineering was spent on infrastructure before the central scientific question was answered: *is a ternary, recurrence-converted 7B MoE better than a 4-bit 1.5–2B dense model in the same 2 GB?* Nothing in the repo addresses it.

**Overall assessment: promising but requires significant redesign.** Details and a simpler alternative are in sections 11–13.

---

## 2. Architecture Understanding

| Pillar | Doc claim | What the code does |
|---|---|---|
| Ternary | 1.58-bit weights, 2-bit packed, int8 activations, L2-optimal row scale | Matches. `train/bitlinear.py`, `runtime/src/kernels/ternary_simd.rs` **[CODE]** |
| NVMe MoE | Experts on disk, 3-layer lookahead **MLP 2048→256→64**, LRU pool | Lookahead is a **single bias-free linear layer 2048→64** (`train/lookahead_router.py:131`). Pool and readers match **[CODE]** |
| Hybrid attention | 14 recurrence layers + 2 softmax anchors (L7, L15) | Matches. Delta-rule and write-gate variants exist in Python only; the exporter and `state.rs` implement the additive rule **[CODE]** |
| Edge preprocessor | tree-sitter scopes + "quantized ONNX coreference" | Code path is real. The prose path is a hand-written rule scorer with hard-coded weights serialized as ONNX (`edge_codec/prose_parser.py:115`), not a trained model **[CODE]** |
| Progressive QAT | Stage 1 continuous alignment, Stage 2 λ ramp, BF16 stochastic optimizer | Matches `train/distill_colab.py` **[CODE]** |

OLMoE facts the design depends on, as encoded in the repo's own full-size fixture (`exporter/export_olmoe.py:2472`): 16 layers, 64 experts, top-8, hidden 2048, intermediate 1024, 16 heads, **16 KV heads** (so "GQA" is ordinary multi-head attention here — there is no key/value sharing), head dim 128, untied `lm_head`.

**Documentation/code discrepancies [CODE]**

| Doc statement | Reality |
|---|---|
| Lookahead: 2-layer MLP, 13 predictors, "~1.4 MB" | Linear 2048→64; 13 × 131,072 params = 6.8 MB in F32. The MLP as described would be ~28 MB. 1.4 MB matches neither |
| `core.bin` "~868 MB with f16 embeddings" | 868 MiB is what F32 embeddings + F32 `lm_head` produce (2 × 393 + 64 + 15). With f16 both, it is ~472 MiB **[MATH]** |
| Anchor KV "~64 MB total at 16,384 tokens" | 2 layers × 2 (K,V) × T × 2048 × 2 bytes. That is 64 MiB at **4,096** tokens and **256 MiB at 16,384** **[MATH]**, confirmed by pool sizing **[MEASURED]** |
| "128 synchronous stalls × 50–100 µs → <1–2 tok/s" | 128 × 100 µs = 12.8 ms per token, which would allow ~78 tok/s. Section 3.1 of the same doc says reads take 3–5 ms. The two numbers contradict |
| RAM "never exceeds `--ram-budget-mb` by a single page" | Budget covers core + state + pool only. Per-token `Vec` allocations, thread stacks, and allocator overhead are outside it. In practice RSS stayed under budget in my runs **[MEASURED]**, but it is not a guarantee |
| Verifier "guaranteeing monotonic, verifiable progress" | It reports metrics and aborts on NaN or a 5× blow-up. It guarantees nothing about progress |
| Core is "pinned in RAM" | Loaded with `std::fs::read` into the heap (`format.rs:672`). No `mlock`. It is ordinary swappable/compressible memory |

---

## 3. What Is Good

- **Binary format and integrity checks.** 4 KB alignment, per-expert CRC32, invalid-crumb trap, header CRCs, strict shape validation on load. This is production-grade work.
- **The ternary kernel is correct and reasonably fast.** I checked the overflow bound: each i16 lane accumulates at most 4 × 127 per chunk and is flushed every 32 chunks (16,256 < 32,767), and the i32 row total is at most 2048 × 127. Measured 9.95 G weights/s on one M2 Pro core **[MEASURED]**.
- **L2-optimal row scale.** The derivation is right, and the 1.25–1.35 ratio is right: I measured γ*/absmean = 1.338 on Gaussian rows, cutting relative reconstruction MSE from 0.263 to 0.212 **[MEASURED]**. Applying it as a one-off row rescale that preserves the sign pattern is a neat trick.
- **Chunkwise-parallel recurrence with document resets.** Exact, memory-bounded, and consistent between the parallel and step-by-step paths (covered by tests).
- **Pre-quantizing activations once** and reusing them across Q/K/V and across the 8 experts' gate/up projections.
- **Buffer ownership in the I/O pool.** In-flight buffers are never freed under a pending read; error paths return buffers; a conservation invariant is asserted on every poll.
- **Self-correction.** The validation leak was found and fixed (`e404655`), and the current branch is running exactly the right ablations (`--anchor-every`, delta rule, write gate). That is the correct instinct.
- **Hint dropout (0.5)** in Stage 2 keeps the model usable when hints are absent.

---

## 4. What Is Bad

Ordered by how much each one threatens the project.

### 4.1 No reproducible evidence for any headline number
The repo has no `eval_gates.json`, `bench_results.json`, training log, or `verification_history.jsonl`. Every figure in the doc is **[DOC-ONLY]**. Two of them are actively undermined:

- **Perplexity.** `e404655`: "Validation previously read docs 128+ of the same streams training read from doc 0, so val_ppl was measured on trained-on data." The doc's 7.37 / 9.98 come from the step-2250 run that `colab/export_and_benchmark.py` already referenced before that commit. Unless the author re-ran after the fix, those numbers are contaminated.
- **Cache hit rate.** See 4.4.

### 4.2 Throughput claim is not met by this code
- Single-threaded: `Cargo.toml` has no threading dependency and `forward_token` runs the 8 experts serially **[CODE]**.
- `f16_gemv` (`kernels/mod.rs`) converts each weight from f16 in software and accumulates in f64, one element at a time. On the 50,304 × 2,048 output head that is **200.8 ms per call**, versus 10.3 ms for the ternary kernel at the same shape **[MEASURED]**. The Colab export path sets `lm_head_dtype="f16"`.
- Anchor attention is a scalar f64 loop over the whole KV cache: 3.2 / 5.5 / 12.1 ms per anchor layer at 512 / 1,024 / 2,048 tokens **[MEASURED]**. Linear in context, so ~24 ms per layer at 4k and ~95 ms at 16k **[ESTIMATE]**.

### 4.3 Context is hard-capped at 4,096 for the real model
`engine.rs:391` reserves `min(requested, header.max_position_embeddings)` and returns `ContextLengthExceeded` past it. The real export copies `max_position_embeddings` from the teacher config (exporter default 4096); only the synthetic fixture sets 16,384. So `--max-seq-len 16384` on a real export silently becomes 4,096, and the "Tight 16k Stream" benchmark generates **6 tokens** — it tests the reservation, not a 16k context. Training uses 1,024-token sequences and validation 512. No part of the system has been exercised at long context.

### 4.4 The lookahead "hit rate" metric rewards random guesses
`prefetched_hits` increments whenever an acquired expert still carries the `prefetched_unused` flag, no matter how many tokens ago it was prefetched (`nvme_pool.rs:911`). Prefetched slots are also protected from prefetch-driven eviction. So any speculative read that is *eventually* used counts as a hit.

The synthetic fixture's lookahead weights are independent random matrices (seed +50 vs +40 for the true router), so its predictions are chance. Yet:

| Run (synthetic, 2 GB) | Hit rate without lookahead | With (random) lookahead |
|---|---|---|
| 8 prompt + 33 new tokens | 80.3% | 88.7% |
| `bench/run_benchmarks.py` defaults (12 + 12) | 67.4% | 79.7% |
| Doc claim | "80%" | "89.4%" |

**[MEASURED]** The improvement is cache warming during cold start, not prediction. It vanishes in steady state (97.8% at 200 tokens either way, because everything is resident).

### 4.5 Under real memory pressure, storage *bandwidth* is the limit, and lookahead cannot help
At a 1 GB budget on the synthetic model (415 slots, uniform routing):

| | tok/s | Hit rate | Bytes read / token |
|---|---|---|---|
| Lookahead on | 4.16 | 40.7% | 209 MB |
| Lookahead off | 4.21 | 36.3% | 130 MB |

**[MEASURED]** Lookahead read 61% more data for no speedup. Prefetching hides latency; it does not create bandwidth. Each token needs 128 distinct experts × 1.52 MiB = 194 MiB of weights. At miss rate *m* and *R* tok/s the drive must sustain *m* × 194 MiB × *R*: at 30 tok/s and 50% misses that is 2.9 GB/s of sustained random 1.5 MB reads, plus CRC and validation on the main thread. Real routing will be more skewed than uniform, so this is the pessimistic end — but the doc offers no routing trace to say how skewed.

### 4.6 The recurrence is the largest quality loss, by the author's own numbers
Teacher 4.61 → Stage 1 (full-precision weights, recurrence only) 7.37 is +0.47 nats/token. Ternarization then adds +0.30 (7.37 → 9.98). **Replacing attention costs more than ternarizing the whole model**, and it is the part with the weakest theoretical footing (section 6.3–6.4).

### 4.7 The symbolic hint is a pointer the model cannot dereference
The hint for a token is `[modality, hash(name) mod 1023, relative_def_offset, confidence]`. The symbol hash restates information already in the token. The offset says *where* the definition is, but 14 of 16 layers are recurrences with no positional addressing; only the two anchors could use a relative position, and nothing biases their attention toward it. The definition's *content* is never delivered. Additionally `generate_greedy` calls `forward_token(tok)` with no hint (`engine.rs:788`), so the Rust runtime never uses hints during generation.

### 4.8 Evaluation harness is too small to detect anything
- In-training validation: `--verify-batches 4` × batch × 512 tokens — a few thousand tokens.
- WikiText-2: 24 × 512 = 12k tokens, each window scored from an empty context.
- HumanEval: 20 problems, `max_new_tokens=64` (most reference solutions are longer).
- Needle test: one trial, one depth, one key, at a length 4× beyond the teacher's position range.
- The project's own gates (`ANTIGRAVITY.md`: recall ≥ 80%, ≥ 90% of teacher perplexity and HumanEval) are not met by the doc's own numbers (65.7% recall, 46% perplexity "retention").

---

## 5. What Can Be Improved

Specific changes, each with the experiment that would validate it.

| # | Change | Why | Validation |
|---|---|---|---|
| 1 | Vectorize `f16_gemv` (hardware f16→f32 convert, f32 accumulate, 4-way unrolled), or store `lm_head` as int8 with per-row scale | 201 ms → roughly 10–15 ms per token | Microbench at 50,304 × 2,048; logit max-abs-diff vs current |
| 2 | Thread the expert loop (8 experts are independent) and row-partition large GEMVs | 9.3 tok/s single-thread ceiling | tok/s vs thread count at fixed context |
| 3 | Replace `prefetched_hits` with "prefetch completed and was used *at its target layer in the same token*"; also log wasted prefetch bytes | Current metric scores random guesses as hits | Random-weight lookahead must score ≈12.5% |
| 4 | Record real routing traces (layer, token, top-8) on 100k+ tokens of held-out text; run a trace-driven cache simulator comparing LRU, static popularity pinning, previous-token-same-layer, learned lookahead, and Belady's optimal | Decides whether Pillar 2 is needed at all, for a few minutes of compute | Hit rate and bytes/token per policy per budget (0.75 / 1 / 1.5 / 2 GB) |
| 5 | Predict layers 0–2 experts from the token id (a 50,304 × 3 × 8 table) | Predictors only exist for targets 3–15; layers 0–2 (19% of acquisitions) are never prefetched **[CODE]** | Recall of the table on the traces from #4 |
| 6 | Evaluate `val_ppl_ternary` during the ramp with the L2-optimal scale | During the ramp the verifier flips λ to 1.0, which uses the plain absmean scale on un-prescaled weights (`bitlinear.py:102`), ~25% too small. The reported 11,425 → 89 sequence measures a quantizer the training is not targeting | Curve should be continuous through the prescale step |
| 7 | Normalize recurrence heads by √((1−α)/(1+α))⁻¹-style variance matching, or better, learn a per-head output gain initialized from measured output RMS vs the teacher's | Current (1−α) factor attenuates long-horizon heads ~38× (section 6.3) | Per-head output RMS at step 0 vs teacher attention output RMS |
| 8 | Pack 5 trits per byte (1.6 bits/weight) for experts | 2-bit packing wastes 26%; saves ~310 MiB, enough to hold *everything* in 2 GB | Kernel throughput with a 256-entry unpack LUT vs current |
| 9 | Use a BF16 (or FP32) teacher for at least the evaluation and one training run | On a 40 GB GPU the teacher is NF4 4-bit, so the student is distilled toward, and scored against, a degraded teacher | Teacher perplexity NF4 vs BF16 on the same held-out set |
| 10 | Make the hint carry content: add an attention bias on anchor layers toward `t + relative_def_offset`, or drop the subsystem | Section 4.7 | Hints on/off ablation on a variable-tracking task |
| 11 | `mlock` the core or stop calling it pinned; `mmap` instead of `fs::read` to halve load-time peak | Honest memory semantics | RSS under a 2 GB cgroup |

---

## 6. Mathematical and Scientific Audit

### 6.1 "1.58-bit" **[MATH][MEASURED]**
log₂3 = 1.585 bits is the information ceiling of a ternary symbol. With the absmean threshold on Gaussian weights the zero fraction is 31%, giving an entropy of 1.583 bits — so the name is fair as information content. Storage is 2.0 bits/weight, 26% more. Row scales add a negligible ~0.008 bits/weight.

### 6.2 Ternary error vs alternatives **[MEASURED on Gaussian rows]**

| Quantizer (per-row scale) | Relative weight MSE |
|---|---|
| Ternary, absmean scale | 0.263 |
| Ternary, L2-optimal scale | 0.212 |
| 2-bit uniform (4 levels), clip-optimal | 0.213 |
| 3-bit uniform | 0.047 |
| 4-bit uniform | 0.013 |
| 8-bit uniform | 0.00009 |

Ternary discards 21% of each row's energy; INT4 discards 1.3% — a 16× gap. Post-training INT4 of a pretrained model needs no retraining; ternary needs QAT to *relearn* the function. The doc's "why not INT4" section argues only compression and multiply-free arithmetic. On CPUs with SDOT/VNNI an int8 multiply-accumulate is a single instruction, whereas this kernel spends about 8 table lookups, 8 ANDs, and widening adds per 64 weights **[CODE]**. Ternary's real advantage is 2× less memory traffic than INT4, not cheaper arithmetic **[ESTIMATE]**. Whether that is worth a 16× larger quantization error is the question the project has not asked.

The training budget makes it sharper: 2,250 steps × 16 sequences × 1,024 tokens ≈ 37M tokens to ternarize 6.4B expert weights. Published ternary results (BitNet b1.58 and successors) train from scratch or for orders of magnitude longer **[ESTIMATE from literature recall]**.

Activation quantization uses per-token abs-max scaling to int8, including on the SwiGLU product `silu(gate) * up`, which is heavy-tailed: one outlier sets the scale and crushes the rest into a few levels. Unmeasured; worth a histogram.

### 6.3 EMA-horizon normalization is dimensionally wrong **[MATH][MEASURED]**
The doc reasons that Σ α^τ = 1/(1−α), so multiplying by (1−α) restores unit gain. That holds only if the summed terms are *coherent* (same sign, as softmax weights are). Linear-attention match scores q·k are signed and largely incoherent across unrelated tokens, so the sum grows as √(Σ α^{2τ}) = 1/√(1−α²). After the (1−α) factor the readout gain is √((1−α)/(1+α)):

| Head half-life | Analytic gain | Simulated |
|---|---|---|
| 4 tokens | 0.294 | 0.334 |
| 20 | 0.131 | 0.108 |
| 102 | 0.058 | 0.063 |
| 512 | 0.026 | 0.027 |

The long-horizon heads — the ones meant to carry long-range information — start ~38× quieter than a softmax head and ~11× quieter than the short heads. The fix took Step-0 perplexity from 668,613 to 1,125, which shows it removed an explosion, not that it matched the teacher (teacher is ~4.6). Consequence: gradient signal to long heads is weak, and the optimizer is likely to leave them unused. Experiment: log per-head output RMS and per-head gradient norm through Stage 1.

### 6.4 State capacity **[MATH][MEASURED]**
One head's state is a 128 × 128 matrix: a linear associative memory of rank ≤ 128. Storing N random key→value pairs and reading one back gives cosine similarity ≈ 1/√(1 + (N−1)/d):

| Pairs stored | Readout cosine |
|---|---|
| 16 | 0.95 |
| 128 | 0.72 |
| 512 | 0.44 |
| 2,048 | 0.22 |
| 16,384 | 0.07 |

Exact retrieval degrades past a few dozen items per head and is noise by a few thousand. Decay limits the effective N to the horizon, which is precisely the trade: remember long *or* remember precisely. This is why the design needs anchors, and why two anchors carry the entire burden of exact long-range recall. The delta rule on this branch is the textbook remedy (it removes interference for repeated keys) and is the right thing to be testing.

The linear readout also has no softmax sharpening: every stored key contributes in proportion to q·k, so there is no winner-take-all. Softmax attention's selectivity is what the teacher's Q/K projections were trained for; reusing those projections in a linear memory changes their meaning.

### 6.5 Two anchors
Layer 15 is the last layer: whatever it retrieves has one MoE block and the output head to act on it. Layer 7 is the only mid-network exact-retrieval point. No ablation in the repo justifies 7:1 over 3:1 or 1:1; the `--anchor-every` flag was added one day ago. **Unsupported**, pending that ablation.

### 6.6 Things that are correct
- STE formulation (`w + (q(w) − w).detach()`), λ-blend continuity through the prescale step, document-boundary masking on both recurrence and anchors, RoPE half-split, QK-norm placement, top-k renormalization.
- `stochastic_round_bf16` rounds magnitude bits with the right probability for both signs.

### 6.7 Things that are risky but unverified
- **`StochasticBF16AdamW` stores both Adam moments as 8-bit linear codes per row.** An element whose √v rounds to code 0 while its momentum does not gets divided by ε = 1e-8 on a step where its gradient is zero — which happens for every expert not routed in that batch. No test compares this optimizer against FP32 AdamW. (bitsandbytes uses non-linear block-wise codes for this reason.)
- **`apply_l2_optimal_row_prescale_` is documented as idempotent and is not.** A second application multiplies rows by 1.338 again **[MEASURED]**. It is guarded by an `l2_prescaled` flag that is re-derived from step counts on resume, so it is currently safe, but one refactor away from a silent 34% weight-scale error.

---

## 7. Source-Code and Implementation Audit

**Test status [MEASURED]:** `cargo test --release` 15/15 pass; `pytest` 58/58 pass.

| Area | Finding | Location | Status |
|---|---|---|---|
| Runtime | No multithreading in compute path | `runtime/Cargo.toml`, `engine.rs:700` | Verified |
| Runtime | Scalar f16 GEMV with f64 accumulator: 201 ms on output head | `kernels/mod.rs` `f16_gemv` | Verified, measured |
| Runtime | Anchor attention scalar f64 over full cache, software f16 decode per element | `state.rs:401-427` | Verified, measured |
| Runtime | Context capped at `max_position_embeddings` | `engine.rs:391-396` | Verified |
| Runtime | `generate_greedy` never passes hints | `engine.rs:788`, `:804` | Verified |
| Runtime | Per-token heap allocations (~20 `Vec`s incl. 50k-float logits, 4 lane vectors per quantization) | `engine.rs:522-574`, `ternary_simd.rs:65` | Verified, minor |
| Runtime | Core loaded by `fs::read`, not `mmap`/`mlock` | `format.rs:672` | Verified |
| Runtime | Delta rule / write gate not implemented in exporter or runtime | `hybrid_ssm.py:190` (acknowledged) | Verified |
| I/O | `prefetched_hits` counts eventual use | `nvme_pool.rs:911` | Verified, measured |
| I/O | `O_DIRECT` silently falls back to buffered I/O on `EINVAL` | `nvme_pool.rs:106` | Verified; only visible via `direct_io_active` in the report |
| I/O | CRC32 + crumb scan over 1.5 MB on the inference thread for every load; time spent in non-blocking polls is excluded from `prefetch_stall_us` | `nvme_pool.rs:1213-1240`, `:889` | Verified |
| I/O | Layers 0–2 have no predictor | `lookahead_router.py:129`, `engine.rs:660` | Verified |
| I/O | Eviction victim search is a linear scan of all 1,024 slots | `nvme_pool.rs:1044` | Verified, negligible at this size |
| Training | Validation previously overlapped training | commit `e404655` | Verified (fixed) |
| Training | Ternary validation during ramp uses mismatched scale | `distill_colab.py:935`, `bitlinear.py:102` | Verified |
| Training | NF4 teacher on < 75 GB VRAM | `distill_colab.py:262` | Verified |
| Training | `init_l2_scale=True` default in `build_student_from_teacher` rescales weights while λ could be 0; Colab path overrides to `False` | `distill.py:164` | Verified, latent |
| Training | Health check "code generation" is 16 greedy tokens scored by unique-token ratio | `distill_colab.py:954-964` | Verified |
| Benchmark | Default run is 12 prompt + 12 generated tokens of arbitrary ids; RAM check uses the engine's self-reported bytes, not RSS | `bench/run_benchmarks.py:91-95`, `:55` | Verified |
| Benchmark | Golden-parity tolerance loosened from 1e-2 to 5e-2 in the final script | `colab/export_and_benchmark.py:290` | Verified |
| Edge codec | "ONNX coreference" is hand-set weights over 8 hand-built features | `prose_parser.py:115-147` | Verified |
| Edge codec | Offsets are taken modulo 1,024 buckets, so −2 and 1,022 alias | `hybrid_ssm.py:73` | Verified, minor |

I did not find memory-safety defects in the `unsafe` SIMD or I/O code in the time spent; the bounds assertions precede every raw-pointer loop.

---

## 8. Memory and Performance Audit

### 8.1 Memory model (real OLMoE shapes, f16 embeddings and head, 4,096-token context)

| Component | Size (MiB) | Class | Basis |
|---|---|---|---|
| **On-disk total** | ~2,024 | storage | 1,552 experts + ~472 core |
| Token embedding (f16) | 196.5 | resident, pageable | 50,304 × 2,048 × 2 |
| `lm_head` (f16) | 196.5 | resident, pageable | same |
| Attention Q/K/V/O, ternary | 64 + scales | resident, pageable | 16 × 4 × 2,048² / 4 |
| Routers + lookahead + norms + hints | ~16 | resident, pageable | F32 |
| Recurrent state | 14 | resident | 14 × 16 × 128² × 4 |
| Recurrent rollback buffers | 14 | resident | `state.rs:17` — doubles the doc's figure |
| Anchor KV | 64 @4k / 128 @8k / **256 @16k** | reserved up-front | 2 × 2 × T × 2,048 × 2 |
| Expert pool | remainder | resident, page-aligned | 1.516 MiB per slot |
| Per-token scratch, logits, thread stacks, allocator | ~5–20 | **outside the budget** | **[ESTIMATE]** |
| Truly pinned (`mlock`) | **0** | — | no `mlock` anywhere |

Pool capacity that results:

| Budget | Context | Pool (MiB) | Expert slots | Share of 1,024 |
|---|---|---|---|---|
| 2,048 | 4k | ~1,484 | ~979 | 96% |
| 2,048 | 16k | ~1,292 | ~852 | 83% |
| 1,024 | 4k | ~460 | ~303 | 30% |

**[MATH]**, cross-checked: the engine's own sizing gave 964 slots at 2 GB / 16k with the 303 MiB synthetic core, exactly matching this formula **[MEASURED]**.

**Is 2 GB feasible?** Yes — for memory. Measured max RSS on the synthetic model was 1.2–1.6 GB at a 2,048 MiB budget and ~0.9 GB at 1,024 MiB. The irony is that feasibility comes from 2-bit packing alone; the streaming machinery is barely exercised. Peak transient RAM at load is roughly core size (one full read into the heap). No test enforces the limit externally (cgroup / `ulimit`), which is the only measurement that should count.

### 8.2 Performance

Measured on M2 Pro, single thread, NEON, `F_NOCACHE` + `pread` pool, synthetic weights:

| Scenario | Result |
|---|---|
| Steady state, all experts resident, ternary head, short context | **9.3 tok/s** (tokens 101–200: 10.8 s) |
| Cold start, first 32 decode steps, 2 GB | 2.6 tok/s |
| 1 GB budget, 100 decode steps | 4.2 tok/s, with or without lookahead |
| Ternary kernel | 9.95 G weights/s per core |
| f16 output head | 200.8 ms per token |
| Anchor attention | ~6 ms per layer per 1,000 tokens of context |

Per-token work is ~1.17 G weight operations (805 M expert, 268 M attention projections, 103 M head), which at the measured kernel rate predicts ~8.5 tok/s — consistent with the 9.3 observed. Cold-start figures are noisy: resident memory was below allocated memory, which suggests macOS was compressing the pool under system memory pressure.

Three scenarios for the **real export** at 2 GB, 4k context **[ESTIMATE built from the measurements above]**:

| | Assumptions | tok/s |
|---|---|---|
| As implemented | 1 thread, f16 scalar head (+200 ms), scalar anchors (+~50 ms at 4k) | **~3** |
| Realistic after fixes 1–2 in §5 | 4–6 performance cores, SIMD head and anchors, experts resident | **15–25** on a current laptop CPU |
| Optimistic | 8+ cores with AVX-512, memory bandwidth not limiting (≈290 MB of packed weights per token → ~9 GB/s at 30 tok/s) | 30+ |
| Worst case | 1 GB budget, poor routing locality, SATA/eMMC storage | **< 2** |

Time-to-first-token: prefill is one token at a time through the same path, so a 1,000-token prompt costs 1,000 decode steps (~100 s as implemented). There is no batched prefill. Energy was not measured.

So 15–30 tok/s is **plausible on laptop-class hardware after real engineering work, undemonstrated today, and not realistic on what is usually meant by "edge"** (4 small cores, eMMC or SATA).

---

## 9. Training and Evaluation Audit

| Question | Assessment |
|---|---|
| Teacher/student compatibility | Good: student is the teacher's own weights, same tokenizer, same residual stream. Weakness: NF4 teacher on 40 GB GPUs |
| Loss | KL (T=2) + layer-wise normalized hidden MSE + lookahead CE. Reasonable. No ground-truth CE term, so the student inherits teacher errors, which is fine for distillation |
| Stage 1 | Trains recurrence blocks, norms, lookahead with everything else frozen. Sound design |
| Stage 2 | Unfreezes all ~7B parameters at λ ramp start with a 100-step warmup. 900 pure-ternary steps is very short |
| Optimizer state | 8-bit Adam moments, untested against a reference (§6.7) |
| Data | FineWeb-edu + the-stack-smol, 35% code, 1,024-token packed sequences. No sequence longer than 1,024 is ever seen, yet the recurrence is meant to carry state far beyond that |
| Validation | Now disjoint by construction (`EVAL_RESERVED_DOCS`, with a test). Still tiny: a few thousand tokens at 512 context |
| Convergence | Not established. A falling curve on a contaminated, few-thousand-token set over 2,250 steps shows the optimizer is working, nothing more |
| Reproducibility | Poor. No seeds for streaming order beyond dataset order, no logged configs/results in the repo, Colab-only, checkpoints on a private Drive. The dry-run mode is a good smoke test but proves nothing about quality |

The reported perplexity ratio, taken at face value, is 9.98 / 4.61 = 2.16× the teacher, i.e. +0.77 nats/token. That is a very large loss. For scale: it is the kind of gap that separates model size classes, which is exactly why baseline 6 in the next section is mandatory.

### Benchmarking plan

Systems (all scored on identical data and harness):

1. OLMoE-1B-7B FP16 (BF16 teacher, not NF4)
2. OLMoE ternary-only (softmax attention kept)
3. OLMoE hybrid recurrence, full-precision weights
4. Hybrid + ternary
5. Full Micro-TERRA (4 + hints + lookahead runtime)
6. Dense small models at 4-bit in `llama.cpp` sized to the same 2 GB (1.5–2B class), plus OLMoE itself at INT4 **without** the RAM cap as a reference point
7. **OLMoE with sliding-window attention on the 14 non-anchor layers, no training** — the zero-cost control for pillar 3

| Capability | Dataset / protocol | Minimum size |
|---|---|---|
| Language modelling | FineWeb-edu held-out + WikiText-103 test, whole documents, sliding context | ≥ 250k tokens; report nats/token with bootstrap CI |
| **Loss vs position** | Same, bucketed by token position (0–256, …, 2k–4k), student minus teacher | The key diagnostic for recurrence memory |
| Knowledge / reasoning | The OLMoE paper's suite (ARC-C/E, HellaSwag, PIQA, WinoGrande, MMLU, BoolQ) so numbers are comparable to published ones | full sets |
| Code | HumanEval (all 164) + MBPP, `max_new_tokens ≥ 512`, hints on and off | full sets |
| Instruction following | Only meaningful from the instruct checkpoint; IFEval if that variant is distilled | full set |
| Long-context retrieval | Needle grid: lengths {512, 1k, 2k, 4k} × depths {10, 30, 50, 70, 90}% × 20 random keys; plus variable-tracking in code | 400 trials |
| Routing | Recall@8 and @16 vs popularity and previous-token baselines; trace-driven cache simulation incl. Belady | ≥ 100k tokens of traces |
| Runtime | tok/s at 256 / 1k / 4k context, TTFT for 1k prompt, RSS under a 2 GB cgroup, `fio`-characterized device, RAPL / `powermetrics` energy | 3 runs each, ≥ 500 generated tokens |

Minimum evidence before accepting each pillar:

| Pillar | Accept only if |
|---|---|
| Ternary | System 2 beats system 6 (dense 4-bit at equal RAM) on perplexity and the reasoning suite |
| NVMe + lookahead | Trace simulation shows learned lookahead beating popularity pinning + LRU by a margin that changes tok/s at a budget someone actually wants |
| Recurrence | System 3 beats system 7 on loss-vs-position and the needle grid |
| Hints | Hints-on beats hints-off on code tasks with CI excluding zero |
| QAT recipe | Same result from two seeds and from an FP32-Adam reference run on a subset |

---

## 10. Critical Risks and Failure Modes

1. **Stacked degradation with no attribution.** Three lossy transforms (recurrence, ternary weights, int8 activations) are trained in one pipeline and evaluated only end-to-end. If the result is poor, the project cannot say which pillar to drop.
2. **Losing to the trivial baseline.** A 1.5–2B dense model at 4-bit fits the same RAM, needs no training, no custom runtime, and already runs at tens of tok/s on laptops **[ESTIMATE]**. If system 5 does not beat it, the whole stack is negative value. This is the single biggest technical risk.
3. **Long context fails silently.** Training never exceeds 1,024 tokens; the recurrence's capacity is ~dozens of exact items per head; anchors run at positions the teacher never saw beyond 4,096. Expect loss to rise with position.
4. **Storage-bound regimes collapse.** Below ~1.5 GB the system needs GB/s of sustained random reads. On anything short of a good NVMe drive throughput falls to low single digits, and flash wear from hundreds of MB read per token is at least worth a thought on embedded storage (reads are benign for endurance, but thermal throttling of the drive is not).
5. **Train/inference mismatch on hints.** Trained with hints 50% of the time on full documents; at generation time hints must come from re-parsing incomplete code every token, which the runtime does not do.
6. **Optimizer artefacts.** If the 8-bit Adam state misbehaves, it will look like "ternary QAT is hard" rather than a bug.

---

## 11. Recommended Alternative Architecture

Optimize for the real objective — useful capability in 2 GB at practical CPU speed — and remove everything that does not buy quality.

**Step 0 (one week, no new code): measure the baseline.** Run system 6. If a stock 4-bit ~1.5–2B model beats the ternary hybrid, stop and reconsider the premise.

**If OLMoE is kept, proposed "TERRA-Lite":**

| Element | Choice | Rationale |
|---|---|---|
| Expert storage | Hold **all** experts in RAM. 5-trits-per-byte packing brings experts to ~1,242 MiB; or keep 2-bit and shrink the core | Removes the lookahead bank, `io_uring`, LRU pool, and their failure modes |
| Core | int8 embeddings and int8 `lm_head` (~196 MiB together vs 393) | Frees ~200 MiB and fixes the 201 ms head |
| Loading | `mmap` the file, popularity-ordered, `madvise(WILLNEED)` | The OS page cache becomes a free, evictable, zero-copy second tier for < 2 GB budgets. The doc treats page cache as an enemy; it is reclaimable memory and the kernel manages it better than a userspace LRU |
| Sequence mixing | **Sliding-window softmax attention (W = 512–1,024) on the 14 non-anchor layers**, full attention on 2–4 anchors, int8 KV | Bounded memory like the recurrence (14 × 512 × 8 KiB ≈ 56 MiB at f16, half at int8), exact teacher behaviour inside the window, **zero training**. Then distill only to recover beyond-window behaviour |
| If recurrence is still wanted | Delta rule, unit-norm keys, variance-matched per-head gain, 3:1 ratio, training sequences ≥ 4k | Addresses §6.3–6.4 |
| Quantization | Start from INT4 experts as the quality reference; move to ternary only on layers where a per-layer sensitivity sweep shows it is cheap | Mixed precision instead of all-or-nothing |
| Hints | Remove until an ablation shows a gain | Section 4.7 |

Budget check for TERRA-Lite at 4k context: experts 1,242 (1.6-bit) + core ~275 + KV ~120 + state/scratch ~40 ≈ **1,680 MiB**, all resident, no storage dependence **[MATH]**.

Trade-offs:

| | Micro-TERRA as designed | TERRA-Lite | Dense 4-bit small model |
|---|---|---|---|
| Quality risk | Highest (three lossy transforms) | Medium (ternary only; attention exact within window) | Lowest, known |
| RAM at 4k context | 2 GB with streaming | ~1.7 GB, all resident | 1.1–1.9 GB |
| Latency variance | High (storage-dependent tail) | Low | Low |
| Engineering | Custom runtime, I/O stack, parser, optimizer | Custom kernel + simple loader | None |
| Below 1 GB | Possible in principle, bandwidth-bound | `mmap` paging, same bound | Pick a smaller model |

---

## 12. Prioritized Findings and Action Plan

| # | Finding | Component | Sev | Kind | Evidence | Action |
|---|---|---|---|---|---|---|
| 1 | No result artifacts; headline perplexities predate the validation-leak fix | Evidence | **P0** | Verified | commit `e404655`; repo search | Re-run eval on the fixed split; commit `eval_gates.json` and logs |
| 2 | No comparison against a dense 4-bit model at equal RAM | Premise | **P0** | Research risk | Absent from code and doc | Run benchmark system 6 before further investment |
| 3 | Throughput claim unmet: single-threaded, 9.3 tok/s best case; f16 head costs 201 ms/token | Runtime | **P0** | Verified bug | Measured | §5 items 1–2 |
| 4 | Hit-rate metric scores random prefetches as hits; doc's 80→89% reproduces with a random router | NVMe | **P1** | Verified bug | Measured (80.3→88.7%) | §5 item 3 |
| 5 | At 2 GB ~96% of experts are resident; streaming pillar is moot | NVMe | **P1** | Verified | Measured + arithmetic | Decide the target budget; if 2 GB, delete the pillar |
| 6 | At 1 GB, bandwidth-bound; lookahead gives no speedup, +61% reads | NVMe | **P1** | Measured (synthetic, macOS) | 4.16 vs 4.21 tok/s | Trace-driven simulation, §5 item 4 |
| 7 | Context capped at 4,096; 16k KV is 256 MiB, not 64 | Runtime / doc | **P1** | Verified | `engine.rs:391`; arithmetic | Fix the doc; decide the real context target |
| 8 | Recurrence costs more quality than ternarization; capacity ~dozens of items per head | Recurrence | **P1** | Research risk | Doc numbers; simulation | Sliding-window control (system 7); finish the anchor/delta ablations |
| 9 | (1−α) normalization attenuates long heads ~38× | Recurrence | **P1** | Math error | Derivation + simulation | §5 item 7 |
| 10 | Ternary has 16× the weight error of INT4; 37M-token QAT budget | Quantization | **P1** | Research risk | Measured MSE | INT4 reference run; per-layer sensitivity sweep |
| 11 | Hints are an undereferenceable pointer; unused in Rust generation; prose "ONNX model" is hand-coded rules | Edge codec | **P1** | Verified | `engine.rs:788`, `prose_parser.py:115` | Ablate; remove if no gain |
| 12 | Lookahead recall 65.7% < own 80% gate; no popularity baseline; doc describes an MLP that does not exist | Lookahead | **P1** | Verified | `lookahead_router.py:131` | Baselines; fix doc |
| 13 | Eval harness underpowered (20 HumanEval @ 64 tokens, 1 needle trial, ~4k-token validation) | Evaluation | **P1** | Verified | `export_and_benchmark.py` | §9 plan |
| 14 | Ternary validation during ramp uses a mismatched scale | Training | P2 | Verified bug | `bitlinear.py:102` | §5 item 6 |
| 15 | NF4 teacher used for both distillation and the retention metric | Training | P2 | Verified | `distill_colab.py:262` | BF16 teacher |
| 16 | 8-bit Adam state can divide by ε for unrouted experts | Training | P2 | Unverified risk | Code reading | Compare against FP32 AdamW on a small model |
| 17 | Prescale documented idempotent but compounds ×1.338 | Training | P2 | Verified | Measured | Fix docstring; assert single application |
| 18 | Layers 0–2 never prefetched | NVMe | P2 | Verified | `engine.rs:660` | Token-id table |
| 19 | 2-bit packing wastes 26% (≈310 MiB of experts) | Format | P2 | Math | 2.0 vs 1.585 bits | 5-trit packing |
| 20 | `core.bin` 868 MB figure is the F32 size; f16 is ~472 MiB | Doc | P2 | Math | Arithmetic | Fix doc |
| 21 | Core not `mlock`ed; `O_DIRECT` fallback is silent; RAM check is self-reported | Runtime | P2 | Verified | `format.rs:672`, `nvme_pool.rs:106` | cgroup-enforced test |
| 22 | No batched prefill; TTFT scales as prompt length × decode cost | Runtime | P2 | Verified | `engine.rs:787` | Batched prefill |
| 23 | Per-token allocations, f64 scalar norms, full sort for top-k | Runtime | P3 | Verified | `engine.rs:522` | Scratch arena |
| 24 | Abs-max int8 activation scaling on heavy-tailed SwiGLU output | Quantization | P3 | Unverified risk | Code reading | Histogram; percentile clipping |
| 25 | Hint offset buckets alias; 1,023-bucket symbol hash | Edge codec | P3 | Verified | `hybrid_ssm.py:73` | Signed bucketing |

**Order of work**

1. Findings 1, 2, 13 — establish honest numbers and the baseline. (Days.)
2. Finding 6's trace simulation and the system-7 control — both cheap, both can delete a pillar. (Days.)
3. Findings 3, 4, 7 — make the runtime's numbers real. (1–2 weeks.)
4. Only then: recurrence fixes (8, 9) and quantization strategy (10), guided by the ablations.

---

## 13. Final Research Verdict

| # | Question | Answer | Confidence | Basis |
|---|---|---|---|---|
| 1 | Scientifically sound? | **Partly.** Each pillar is a legitimate idea; the combination is unvalidated and the supporting evidence does not hold up | High | §4.1, §4.4 |
| 2 | Mathematical foundation correct? | **Mostly.** Ternary scaling, STE, chunked scan are right. The EMA-horizon normalization is wrong, and capacity limits are not acknowledged | High | §6 |
| 3 | Retains useful language capability? | **Unknown, leaning doubtful as built.** The only numbers show a 2.2× perplexity ratio on a contaminated split | Medium | §9 |
| 4 | 2 GB feasible? | **Yes** for memory — because 2-bit packing nearly fits the whole model | High | Measured RSS, §8.1 |
| 5 | 15–30+ tok/s realistic? | **Not today (≈3 as exported, 9.3 best case). Plausible on a multi-core laptop after threading and SIMD fixes. Not on low-end edge hardware** | High for today; medium for the projection | §8.2 |
| 6 | Retain | Binary format and integrity checks; ternary kernel; L2-optimal scaling; staged distillation skeleton; document-isolation masking; the ablation flags | High | §3 |
| 7 | Redesign | Sequence mixer (window attention control first; delta rule if recurrence stays); evaluation harness; output head and threading; hit-rate accounting; quantization as mixed precision | High | §5, §11 |
| 8 | Remove | NVMe lookahead stack **if the target stays 2 GB**; the symbolic hint subsystem pending an ablation; the custom 8-bit optimizer unless it is validated | Medium-high | §4.5, §4.7 |
| 9 | Biggest technical risk | That the finished system is worse than an off-the-shelf 4-bit small dense model in the same RAM | High that it is the biggest risk; outcome unknown | §10 |
| 10 | Approve further investment? | **Conditionally.** Fund ~2 weeks for steps 1–2 of the action plan. Do not fund further runtime or training work until the baseline comparison and the two cheap controls are in | High | — |

**Overall: promising but requires significant redesign.** The work shows real engineering skill and one genuinely useful result — a full 7B MoE can be made to fit in 2 GB by ternary packing alone. What it lacks is the experiment that says whether that model is worth running.

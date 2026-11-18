# Micro-TERRA — zero-training control experiments (2026-11-18)

Companion to `IMPLEMENTATION_REPORT.md`. These runs answer the questions in `ARCHITECTURE_REVIEW.md`
that needed a GPU and the real teacher but **no trained student**. Nothing here was trained; no Micro-TERRA
checkpoint was evaluated. Everything is self-reported and not independently verified.

## Conditions

| | |
|---|---|
| Hardware | Colab "G4": NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB. A100 and H100 were both requested and unavailable |
| Software | torch 2.14.1+cu130, transformers 5.23.0, seed 1234 |
| Teacher | `allenai/OLMoE-1B-7B-0125` in **BF16** (not NF4) |
| Data | wikitext-103 test: 60 documents, 101 windows, 285,512 scored tokens, 1.29 MB. fineweb-edu `sample-10BT`: first 60 documents ≥ 9,000 chars, 81 windows, 226,335 tokens, 1.03 MB |
| Protocol | Whole documents cut into 4,096-token windows, each starting with BOS/EOS; every text token scored. Deltas are paired per window, 95% CI from 2,000 bootstrap resamples |
| Scripts | `colab/review_experiments.py` (sha256 `881eccbd3a35…`), `colab/review_followup.py` (`7c11369e2ef2…`) |
| Raw results | `bench/results/2026-11-18_colab_review_experiments.json`, `bench/results/2026-11-18_colab_review_followup.json` |
| Run time | 8 min + 13 min |

fineweb-edu `sample-10BT` is training-distribution text for this project, and wikitext may be in any of these
models' pretraining data. Treat cross-model comparisons as indicative, not as a clean benchmark.

## 1. What the student has to beat (findings 2, 15)

Bits per byte, lower is better (comparable across tokenizers). [MEASURED]

| Model | Precision | Weights | wikitext | fineweb-edu |
|---|---|---|---|---|
| OLMoE-1B-7B (teacher) | BF16 | ~13 GiB | 0.588 | 0.653 |
| OLMoE-1B-7B | int4 experts only | ~3.6 GiB est. | 0.598 | 0.662 |
| OLMoE-1B-7B | int4 every Linear | ~3.6 GiB est. | 0.606 | 0.669 |
| OLMoE-1B-7B | int4 experts + 512-token window on 14 layers | ~3.6 GiB est. | 0.659 | 0.708 |
| SmolLM2-1.7B | int4 | 0.90 GiB est. | 0.735 | 0.738 |
| Qwen2.5-1.5B | int4 | 0.81 GiB est. | 0.745 | 0.762 |
| SmolLM2-1.7B | BF16 | | 0.689 | 0.694 |
| Qwen2.5-1.5B | BF16 | | 0.715 | 0.734 |

"int4" is a Q4_0-style simulation (blocks of 32, 4.5 bits/weight), applied in PyTorch, not a real GGUF run.

- **The bar is 0.735 / 0.738 bits/byte**: an int4 1.7B dense model that fits in 0.9 GiB with no paging,
  no recurrence and no distillation.
- In teacher tokens that leaves the student a total budget of about **+0.46 nats/token on wikitext and
  +0.27 on fineweb-edu** over the BF16 teacher. Any student worse than that loses to a model that already exists.
  [DERIVED from the table]
- OLMoE at int4 clears the bar by a wide margin (0.606 vs 0.735) but weighs ~3.6 GiB, so at 2 GiB it has to page (§4).
- Qwen2.5-3B is excluded: it scored worse than Qwen2.5-1.5B on wikitext in both BF16 (0.761) and FP32 (0.806),
  with no single exploding window. I did not find the cause, so I do not trust those numbers.

## 2. Ternary without training is not a starting point (finding 10)

Expert weights only, attention untouched. Delta in nats/token vs the BF16 teacher. [MEASURED]

| Expert quantisation | Relative weight MSE | wikitext | fineweb-edu |
|---|---|---|---|
| int4, block 32 | 0.010 | +0.031 (0.025..0.038) | +0.028 (0.020..0.039) |
| int3, block 32 | 0.052 | +0.173 (0.141..0.213) | +0.158 (0.113..0.213) |
| ternary, L2-optimal row scale | 0.216 | **+5.83** (5.67..6.01) | **+5.09** (4.94..5.24) |

Ternary rows use 24 windows (70k / 66k tokens); the others use all windows.

Ternarising one layer at a time (12 windows, 35k tokens per dataset) costs +0.026 to +0.162 on wikitext;
the last layer is the most sensitive (+0.16), layer 1 and layers 13–14 next (+0.08). The 16 single-layer
costs sum to +0.93 (wikitext) and +0.66 (fineweb-edu), against +5.83 and +5.09 for all layers at once:
the errors compound roughly sixfold.

- QAT has to recover more than 5 nats and land inside the +0.27..0.46 budget of §1 *together with* the
  recurrence loss. int4 experts start at +0.03 with no training at all.
- If ternary is kept, the per-layer numbers suggest leaving layer 15 (and possibly 1, 13, 14) at a higher
  precision. [HYPOTHESIS: untested in combination]

## 3. How much context the 14 non-anchor layers need (finding 8)

Attention in the 14 non-anchor layers restricted to the last W tokens; layers 7 and 15 keep full attention.
No training. Delta in nats/token. [MEASURED]

| W | wikitext | fineweb-edu |
|---|---|---|
| 64 | +1.027 | +0.741 |
| 128 | +0.637 | +0.457 |
| 256 | +0.372 | +0.280 |
| 512 | +0.189 | +0.149 |
| 1024 | +0.068 | +0.054 |
| 2048 | +0.013 | +0.010 |

Anchor layers, at W = 256 (wikitext / fineweb-edu): none +0.391 / +0.285, two (7, 15) +0.372 / +0.280,
four (3, 7, 11, 15) +0.380 / +0.300. At W = 512: +0.197 / +0.148, +0.189 / +0.149, +0.194 / +0.160.
The confidence intervals overlap in every case.

- **Full attention in two or four layers buys nothing measurable** when the other layers are windowed,
  zero-shot. This does not show that trained anchors are useless, but there is no free benefit to inherit
  from the teacher.
- This control overstates the information lost. The teacher itself gains only 0.05–0.07 nats between
  positions 256–512 and 2,048–4,096 (1.845 → 1.795 wikitext, 2.077 → 2.004 fineweb-edu), yet a 256-token
  window costs +0.27 already at positions 256–512. Most of the penalty is the pretrained weights being run
  out of distribution, which training can in principle remove. [INFERENCE]
- So on these documents long-range context is worth roughly 0.05–0.07 nats. A student limited to ~512
  tokens of effective memory is not capped by missing context; whatever the recurrence loses beyond that is
  approximation error, not information that was never available. [INFERENCE] Tasks that need exact recall
  over long distances are not measured by average NLL on this text.

## 4. Routing and the NVMe pillar (findings 5, 6, 12, 18)

512,029 tokens × 16 layers × top-8 of 64 experts. [MEASURED]

**Routing is close to uniform.** Entropy 5.87 of 6.00 bits; the 8 most-used experts per layer take 22% of
acquisitions, the top 32 take 66%.

**Predicting the experts of layer L+3** (recall@8, chance 12.5%):

| Predictor | Recall@8 |
|---|---|
| Same experts as this token at layer L | 13.2% |
| Static popularity | 18.2% (20–29% on the probe split) |
| Same experts as the previous token | 37.1% |
| Token-id table, layers 0 / 1 / 2 (5.2% unseen tokens) | 39.0% / 50.0% / 56.0% |
| Linear probe on the residual, 0→3 / 4→7 / 8→11 / 12→15 | **72.2% / 81.4% / 83.5% / 85.3%** |

The probe is a single linear layer trained on 81,100 tokens and tested on 20,275 held-out tokens of the
BF16 teacher. A ternary student's residual stream may be less predictable.

**Trace-driven cache simulation** (120,000 scored tokens after a 30,000-token warm-up for popularity;
core + KV + state taken as 564 MiB; miss rate, and MiB read per token):

| Budget | Experts | Slots of 1,024 | LRU | Pin top half + LRU | Belady (optimal) |
|---|---|---|---|---|---|
| 2,048 MiB | ternary 1.52 MiB | 978 | 0.15%, 0.3 | 0.16% | 0.05% |
| 1,536 MiB | ternary | 641 | 14.9%, 29 | 15.6% | 4.9%, 9.6 |
| 1,024 MiB | ternary | 303 | 46.8%, 91 | 47.3% | 22.5%, 44 |
| 768 MiB | ternary | 134 | 63.4%, 123 | 85.9% | 43.9%, 85 |
| 2,048 MiB | int3 2.63 MiB | 565 | 20.7%, 70 | 21.4% | 7.4%, 25 |
| 3,072 MiB | int3 | 955 | 0.35%, 1.2 | 0.39% | 0.11% |
| 2,048 MiB | int4 3.38 MiB | 439 | 31.7%, 137 | 32.3% | 13.1%, 57 |
| 3,072 MiB | int4 | 743 | 8.1%, 35 | 8.9% | 2.5%, 11 |

- **Finding 5 confirmed.** With ternary experts at 2,048 MiB, 978 of 1,024 experts are resident and the miss
  rate is 0.15%. Paging, lookahead and prefetch contribute nothing at the headline budget.
- **Finding 6 confirmed.** At 1,024 MiB the cache reads 91 MiB per token under LRU and 44 MiB even with a
  perfect oracle. At 2 GB/s sequential that is ~48 ms and ~23 ms per token of I/O alone. [ESTIMATE: the
  drive speed is assumed, not measured]
- **Pinning popular experts never helps**; it is equal or worse than plain LRU at every size, as near-uniform
  routing predicts.
- **Finding 12.** A linear lookahead is feasible on the teacher: 72–85% recall against 37% for the best
  trivial baseline. The optimal policy misses 1.4–3.3× less than LRU, which is the headroom a good
  predictor could claim through eviction, and it only matters below ~1.5 GiB.
- **Finding 18.** A token-id lookup predicts layers 0–2 at 39–56% recall for the cost of a table. Only worth
  building if a sub-2 GiB target is kept.
- **The int4 alternative.** int4 experts cost +0.03 nats with no training, but at 2,048 MiB they read
  137 MiB per token (57 with an oracle): I/O-bound to roughly 14 tok/s at 2 GB/s before any compute.
  [ESTIMATE] At 3,072 MiB it drops to 35 MiB per token.

## 5. Candidate teachers (user request)

BF16, 4,096-token context, same documents, bits per byte. [MEASURED]

| Model | Params | wikitext | fineweb-edu |
|---|---|---|---|
| OLMoE-1B-7B (current teacher) | 6.9B | 0.588 | 0.653 |
| Qwen2.5-7B | 7.6B | 0.617 | 0.646 |
| phi-4 | 14.7B | 0.579 | 0.641 |
| Qwen2.5-14B | 14.8B | 0.529 | 0.574 |
| Qwen2.5-32B | 32.8B | 0.510 | 0.550 |

- Qwen2.5-14B is 10–12% better than OLMoE and Qwen2.5-32B 13–16%; both fit a 80–98 GB card in BF16.
  Qwen2.5-7B and phi-4 are not meaningfully better than the current teacher.
- A different teacher has a different tokenizer and no experts, so it cannot supply the logit or
  hidden-state targets the current pipeline distils from. It could only be used to generate training text
  or through a token-alignment scheme that does not exist in this repo.
- The teacher is not the bottleneck: the available gain is 0.06–0.10 bits/byte, while the student's
  open problems (§2, §3) are measured in whole nats. I recommend keeping OLMoE and switching it from NF4 to
  **BF16**, which now fits easily. The NF4 teacher's own loss was not measured here.

## 6. What this changes

1. The project's quality claim is now falsifiable: the student must reach **< 0.735 bits/byte on wikitext
   and < 0.738 on fineweb-edu** under this protocol, or a 0.9 GiB off-the-shelf model is better.
2. At 2 GiB the NVMe machinery is dead weight with ternary experts. It earns its place only for a
   ≤ 1.5 GiB target or for int3/int4 experts.
3. The cheapest route to a 7B-quality model is not ternary: int4 experts lose 0.03 nats untrained. Its cost
   is I/O, 137 MiB per token at 2 GiB. Ternary's cost is 5+ nats that QAT must win back.
4. Anchor layers are not justified by these controls, and average NLL on this text does not need more than
   ~512–1,024 tokens of memory.

## 7. Not done

- No Micro-TERRA checkpoint was evaluated (finding 1 stays blocked). The student's actual bits/byte under
  this protocol is the single most useful missing number.
- No QAT recovery experiment; §2 measures the starting point only.
- The window control is an out-of-distribution stress test, not a trained hybrid.
- int4 results are simulated quantisation in PyTorch; no llama.cpp throughput or RSS was measured.
- Cache simulation counts reads; it does not model prefetch timing, the lookahead predictor, or real drive latency.
- The Qwen2.5-3B anomaly is unexplained.
- One seed, one document sample.

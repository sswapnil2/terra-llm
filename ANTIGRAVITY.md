# ANTIGRAVITY.md — Project Micro-TERRA Execution Blueprint

## 1. Objective & Hardware Split
Build and validate **Micro-TERRA**, a `MatMul`-free, NVMe-tiered LLM inference and distillation system combining:
1. **1.58-Bit Ternary Weights (`{-1, 0, +1}`):** Replaces floating-point `MatMul` with integer `ADD`/`SUB` via `BitLinear` layers.
2. **7:1 Hybrid Recurrent + Anchor Attention:** Distills 7 of every 8 Transformer attention layers into $O(1)$-memory gated linear recurrence, keeping 1 Grouped-Query Attention (GQA) layer per 8-layer block strictly for exact variable/needle recall.
3. **Dual-Path Edge Pre-Processor:** Uses deterministic `tree-sitter` AST/scope pointers for code and lightweight coreference span clustering for prose, injected as non-destructive hint embeddings.
4. **3-Layer Lookahead MoE Routing:** Predicts Layer $L+3$ top-$k$ experts at Layer $L$ to prefetch cold expert blocks from NVMe asynchronously.
5. **Bare-Metal Rust Runtime:** 4KB-aligned binary shards (`core.bin`, `experts.bin`) executed via SIMD intrinsics + Linux `io_uring` (`O_DIRECT`).

### Execution Split (MacBook 16 GB Local vs. Google AI Ultra Colab A100)
- **Local Workstation (Apple Silicon MacBook, 16 GB Unified Memory):**
  - Schema definition, synthetic binary fixture generation, `tree-sitter` + coreference edge parser, Rust `ARM NEON` SIMD kernels, macOS `pread` async pool fallback, and **single-layer PyTorch unit/gradient tests** (`< 4 GB` RAM).
- **Cloud Training & Linux I/O Runtime (Google Colab A100-40GB / A100-80GB via Google AI Ultra):**
  - Full `allenai/OLMoE-1B-7B-0125` distillation using 4-bit NF4 frozen teacher + `PagedAdamW8bit` + gradient checkpointing (~32–35 GB VRAM footprint), background execution enabled, checkpointing every 250 steps to `/content/drive/MyDrive/terra_ckpt`, followed by native Linux `io_uring` + `O_DIRECT` benchmarks on `/tmp` NVMe.

---

## 2. Non-Negotiable Engineering Guardrails
1. **Strict Memory Isolation Locally:** Never load the full `OLMoE-1B-7B-0125` checkpoint in `BF16` or run multi-layer backpropagation on the local 16 GB MacBook. All local PyTorch verification scripts must isolate **1 single layer** (`MicroTerraSingleLayerTest`) or use `--synthetic` random tensors.
2. **Strict 4KB Block Alignment:** Every exported expert weight tensor in `experts.bin` must be padded and aligned to 4096-byte boundaries (`#[repr(align(4096))]`) for `O_DIRECT` DMA compatibility.
3. **Cross-Platform Rust Compilation:** `runtime/` must compile and pass `cargo test` on both `aarch64-apple-darwin` (using `std::arch::aarch64` NEON + async `pread` pool) and `x86_64-unknown-linux-gnu` (using `AVX2`/`AVX-512` + native `io_uring` `O_DIRECT`).
4. **End-to-End Differentiability:** Edge parser hints (`[modality, entity_or_var_id, relative_def_offset, entropy_bucket]`) must be projected via a learnable `HintEmbedding` layer—never discrete string replacement—so gradients flow cleanly during distillation.

---

## 3. Execution Milestones

### Milestone 1: Binary Contract & Synthetic/Real Exporter (`schemas/`, `exporter/`) — *Run Locally*
- **Deliverables:**
  - `schemas/terra_format.md`: Byte-level spec for `core.bin` (pinned recurrent/anchor weights + routers) and `experts.bin` (4KB-aligned 2-bit packed ternary blocks + `f32` channel scales + CRC32 table).
  - `exporter/export_olmoe.py`: Supports `--synthetic` (generates valid OLMoE-shaped `core.bin` + `experts.bin` in `< 5s` locally) and `--checkpoint allenai/OLMoE-1B-7B-0125` (exports trained Colab weights).
- **Verification Gate:** `pytest exporter/test_packing.py` verifies `unpack(pack(W)) == W` bit-exactness and `offset % 4096 == 0` for all 1,024 experts.

### Milestone 2: Dual-Path Edge Pre-Processor (`edge_codec/`) — *Run Locally*
- **Deliverables:**
  - `edge_codec/code_parser.rs`: `tree-sitter` (Python & Rust) scope/definition resolver emitting exact relative token offsets.
  - `edge_codec/prose_parser.py`: Lightweight ONNX coreference/span tagger emitting cluster IDs and entropy flags (`confidence < 0.65` marked high-entropy).
- **Verification Gate:** `< 2 ms` latency per 512-token mixed code/prose snippet on MacBook CPU; 100% exact local variable binding on a 25-file Python test suite.

### Milestone 3A: Single-Layer Local Verification Suite (`train/test_single_layer.py`) — *Run Locally (< 4 GB RAM)*
- **Deliverables:**
  - Standalone script instantiating **1 layer** matching OLMoE dimensions (`d_model=2048`, `64 experts`, `top_k=8`) with `BitLinear` STE (`{-1, 0, +1}`), gated linear recurrence, `HintEmbedding`, and an auxiliary 3-layer lookahead router head.
  - Runs 20 synthetic training steps on `mps` or `cpu`, verifying non-zero gradients across STE weights, recurrent gates, and lookahead router heads while tracking peak RSS memory (`< 4096 MB`).
- **Verification Gate:** `python train/test_single_layer.py` exits `0` with verified gradient norms and peak RAM `< 4 GB`.

### Milestone 3B: Colab A100 Full Distillation Pipeline (`train/distill_colab.py` & `colab/train_micro_terra.ipynb`) — *Run on Google Colab A100*
- **Deliverables:**
  - **Memory-Optimized Colab Config (`A100-40GB` & `A100-80GB` auto-detect):**
    - Loads frozen teacher `allenai/OLMoE-1B-7B-0125` in **4-bit NF4** (`bitsandbytes`) on 40GB A100 (or `BF16` if `>= 75 GB` VRAM detected).
    - Student initialized from teacher weights with `BitLinear` STE projections, layers `0..6, 8..14` converted to gated linear recurrence, and layers `7, 15` preserved as exact GQA anchor layers.
    - Trains with `PagedAdamW8bit`, `torch.amp.autocast(dtype=torch.bfloat16)`, gradient checkpointing, and combined loss: `L_total = L_kl_logits + 0.5 * L_layer_mse + 0.2 * L_lookahead_ce + 0.01 * L_load_balance`.
  - **Resilience:** Mounts Google Drive (`/content/drive/MyDrive/terra_ckpt`) and saves atomic checkpoints + optimizer state every 250 steps for uninterrupted Colab background execution.
- **Verification Gate:** Lookahead router achieves `>= 80%` top-8 recall at $L \to L+3$; distilled model retains `>= 90%` of teacher WikiText-2 perplexity and HumanEval pass@1.

### Milestone 4: Bare-Metal Rust SIMD + Async NVMe Runtime (`runtime/`) — *Build & Test Locally, Benchmark on Colab Linux*
- **Deliverables:**
  - `runtime/src/kernels/ternary_simd.rs`: `ARM NEON` (`vqtbl1q_u8` + `vaddq_s16`) and `x86 AVX2/AVX-512` (`_mm256_shuffle_epi8`) 2-bit packed ternary dot-product kernels with zero floating-point multiplications inside the inner loop.
  - `runtime/src/nvme_pool.rs`: LRU hot-expert RAM pool (`--ram-budget-mb`) + 3-layer lookahead async prefetcher (`io_uring` + `O_DIRECT` on Linux; bounded async `pread` worker pool on macOS).
- **Verification Gate:** `cargo test --release` passes numerical parity against PyTorch reference (`atol=1e-2`) on MacBook using synthetic fixtures.

### Milestone 5: End-to-End Linux `io_uring` Benchmarks & Walkthrough (`bench/`) — *Run on Colab A100 VM*
- **Deliverables:**
  - Notebook cells in `colab/train_micro_terra.ipynb` that export the distilled checkpoint to `/tmp/experts.bin`, compile `runtime/` in `--release` mode on Linux, and benchmark:
    1. Synchronous cold NVMe reads (no lookahead) vs. 3-layer `io_uring` lookahead prefetching under a strict 2 GB RAM cap.
    2. Decode tokens/sec, NVMe cache hit rate (%), lookahead stall latency (ms), and 16k-token Needle-in-a-Haystack recall.
  - Exports final `core.bin`, `experts.bin`, and `WALKTHROUGH.md` back to Google Drive.

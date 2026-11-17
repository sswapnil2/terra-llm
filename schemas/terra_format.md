# Micro-TERRA Binary Format & Mathematical Contract (`v1`)

This specification is the **canonical contract** shared across:
- **`exporter/`** (Python checkpoint converter & ternary bit-packer)
- **`train/`** (PyTorch `BitLinear` QAT, 7:1 Hybrid Gated Linear Recurrence, and 3-Layer Lookahead Router)
- **`runtime/`** (Bare-metal Rust SIMD + `io_uring` / `F_NOCACHE` inference engine)
- **`edge_codec/`** (Dual-path AST + coreference `[seq_len, 4]` hint generator)

All multi-byte integers (`u16`, `u32`, `u64`, `i32`) and IEEE-754 floating-point values (`f16`, `f32`) are serialized in **little-endian** byte order.
All `crc32` checksums use standard **IEEE 802.3 CRC-32** (`zlib.crc32(buf) & 0xFFFFFFFF` in Python; `crc32fast::hash(buf)` in Rust; test vector `b"123456789"` $\rightarrow$ `0xCBF43926`).

---

## 1. 1.58-Bit Ternary Weight Packing & `BitLinear` Contract

### 1.1 `absmean` Ternary Weight Quantization

Given a 2D weight tensor $W \in \mathbb{R}^{M \times K}$ (`out_features = M`, `in_features = K`, where **`K % 4 == 0`** is mandatory):

1. **Per-output-channel scale factor** ($\gamma \in \mathbb{R}^M$, stored as little-endian `f32`):
   $$\gamma_r = \max\!\left(\frac{1}{K} \sum_{k=0}^{K-1} |W_{r, k}|,\; 10^{-8}\right), \quad r \in [0, M)$$

2. **Ternary weight value** ($\tilde{W}_{r,k} \in \{-1, 0, +1\}$, using IEEE round-half-to-even, i.e., `torch.round` / `np.rint`):
   $$\tilde{W}_{r, k} = \operatorname{round}\!\left(\operatorname{clip}\!\left(\frac{W_{r, k}}{\gamma_r},\; -1.0,\; +1.0\right)\right)$$

### 1.2 2-Bit Crumb Encoding & Byte Layout

Each ternary value maps to a 2-bit unsigned code:

| Weight $\tilde{W}$ | Binary Crumb | Decimal Code | Notes |
| :---: | :---: | :---: | :--- |
| `0` | `0b00` | `0` | Zero weight (skip in ADD/SUB) |
| `+1` | `0b01` | `1` | Positive accumulate (`+x`) |
| `-1` | `0b10` | `2` | Negative accumulate (`-x`) |
| *Invalid* | `0b11` | `3` | Reserved; unpackers MUST return an error if encountered |

Four consecutive weights along the `in_features` ($K$) axis pack into a single `u8` in little-endian crumb order:

```text
Bit position:   7   6 | 5   4 | 3   2 | 1   0
Crumb lane:     w[4b+3] w[4b+2] w[4b+1] w[4b+0]
```

$$\text{packed}[r, b] = \operatorname{code}(\tilde{W}_{r, 4b}) \;\mid\; (\operatorname{code}(\tilde{W}_{r, 4b+1}) \ll 2) \;\mid\; (\operatorname{code}(\tilde{W}_{r, 4b+2}) \ll 4) \;\mid\; (\operatorname{code}(\tilde{W}_{r, 4b+3}) \ll 6)$$

- Packed tensor shape: `[M, K / 4]` of `u8` in row-major (`C`-contiguous) layout (`M * (K / 4)` bytes).
- Scale vector shape: `[M]` of `f32` (`M * 4` bytes).

### 1.3 `BitLinear` Dynamic `int8` Activation Quantization & GEMV

Given an input vector $x \in \mathbb{R}^K$:

1. **Optional `RMSNorm` Pre-Scaling** (controlled by `FLAG_BITLINEAR_RMS_PRESCALE` in `core.bin`):
   - If `FLAG_BITLINEAR_RMS_PRESCALE` (`bit 0`) is `1`:
     $$\hat{x} = \frac{x}{\sqrt{\frac{1}{K}\sum_{k=0}^{K-1} x_k^2 + \varepsilon_{\text{rms}}}}$$
   - If `FLAG_BITLINEAR_RMS_PRESCALE` is `0` (direct PTQ export of unmodified `OLMoE` checkpoints):
     $$\hat{x} = x$$

2. **Per-Token Symmetric `int8` Quantization**:
   $$s_x = \frac{\max_{0 \le k < K} |\hat{x}_k|}{127.0}$$
   - If $\max_k |\hat{x}_k| < 10^{-12}$, the output vector $y \in \mathbb{R}^M$ is all zeros.
   - Otherwise:
     $$x_{q, k} = \operatorname{clamp}\!\left(\operatorname{round\_ties\_to\_even}\!\left(\frac{\hat{x}_k}{s_x}\right),\; -127,\; 127\right) \in \mathbb{Z}$$

3. **Zero-FP-Multiply Integer Inner Loop + Scalar Epilogue**:
   $$\text{acc}_r = \sum_{k=0}^{K-1} \tilde{W}_{r, k} \, x_{q, k} \in \mathbb{Z} \quad (\text{computed strictly via integer ADD/SUB}), \qquad y_r = \operatorname{f32}(\text{acc}_r) \cdot \gamma_r \cdot s_x$$

---

## 2. `core.bin` Layout — Resident Pinned Model Image

`core.bin` holds model metadata, embeddings, `RMSNorm` weights, attention / gated-linear-recurrence weights, ground-truth MoE routers, lookahead routers, optional `HintEmbedding` tables, and `lm_head`.

### 2.1 File Header (`256` Bytes at Offset `0`)

| Byte Offset | Type | Field Name | Description |
| :---: | :---: | :--- | :--- |
| `0..4` | `[u8; 4]` | `magic` | ASCII `b"TRAC"` (`0x54, 0x52, 0x41, 0x43`) |
| `4..8` | `u32` | `version` | Format version = `1` |
| `8..12` | `u32` | `flags` | Bitfield (`bit 0`: `FLAG_BITLINEAR_RMS_PRESCALE`, `bit 1`: `FLAG_NORM_TOPK_PROB`, `bit 2`: `FLAG_TIE_WORD_EMBEDDINGS`, `bit 3`: `FLAG_HAS_HINT_EMBEDDING`) |
| `12..16` | `u32` | `vocab_size` | e.g., `50304` for `OLMoE-1B-7B` |
| `16..20` | `u32` | `hidden_size` | $D$, e.g., `2048` (`% 4 == 0`) |
| `20..24` | `u32` | `intermediate_size` | $I$, e.g., `1024` (`% 4 == 0`) |
| `24..28` | `u32` | `num_layers` | $L$, e.g., `16` (max `64`) |
| `28..32` | `u32` | `num_attention_heads` | $H$, e.g., `16` |
| `32..36` | `u32` | `num_kv_heads` | $H_{kv}$, e.g., `16` |
| `36..40` | `u32` | `head_dim` | $d = D / H$, e.g., `128` |
| `40..44` | `u32` | `num_experts` | $E$, e.g., `64` |
| `44..48` | `u32` | `top_k` | Active experts per token, e.g., `8` |
| `48..52` | `u32` | `max_position_embeddings` | e.g., `4096` or `16384` |
| `52..56` | `f32` | `rope_theta` | e.g., `10000.0` |
| `56..60` | `f32` | `rms_norm_eps` | e.g., `1e-5` |
| `60..64` | `u32` | `lookahead_offset` | Default `3` (Layer $L$ predicts Layer $L + 3$) |
| `64..128` | `[u8; 64]` | `layer_kinds` | Per-layer enum for `0..num_layers`: `0 = AnchorGQA`, `1 = GatedLinearRecurrence` |
| `128..132` | `u32` | `tensor_count` | Number of `CoreTensorEntry` records following at offset `256` |
| `132..140` | `u64` | `data_section_offset` | Byte offset where tensor payloads begin (aligned to `64` bytes) |
| `140..252` | `[u8; 112]` | `reserved` | Zero-filled (`0x00`) |
| `252..256` | `u32` | `header_crc32` | IEEE CRC32 computed over bytes `0..252` |

### 2.2 Tensor Directory (`tensor_count * 128` Bytes Starting at Offset `256`)

Each `CoreTensorEntry` is exactly `128` bytes:

| Rel Offset | Type | Field Name | Description |
| :---: | :---: | :--- | :--- |
| `0..64` | `[u8; 64]` | `name` | Null-padded UTF-8 string (e.g., `b"layers.0.self_attn.q_proj.weight\0..."`) |
| `64..65` | `u8` | `dtype` | `0 = Ternary2Bit`, `1 = F32`, `2 = F16`, `3 = I8` |
| `65..66` | `u8` | `ndim` | Rank (`1..=4`) |
| `66..68` | `u16` | `flags` | Reserved (`0`) |
| `68..84` | `[u32; 4]` | `shape` | Logical unpacked dimensions `[d0, d1, d2, d3]` (unused trailing dims = `1` or `0`; for 2D `Ternary2Bit`, `[M, K, 0, 0]`) |
| `84..92` | `u64` | `data_offset` | Absolute byte offset in `core.bin` (`% 64 == 0`) |
| `92..100` | `u64` | `data_bytes` | Byte length of primary buffer (`M * (K / 4)` for `Ternary2Bit`) |
| `100..108` | `u64` | `scale_offset` | Absolute byte offset of `f32` scale vector (`0` if `dtype != Ternary2Bit`) |
| `108..116` | `u64` | `scale_bytes` | Byte length of scale vector (`M * 4` for `Ternary2Bit`, else `0`) |
| `116..120` | `u32` | `data_crc32` | IEEE CRC32 over `[data_offset .. data_offset + data_bytes]` |
| `120..124` | `u32` | `scale_crc32` | IEEE CRC32 over `[scale_offset .. scale_offset + scale_bytes]` (`0` if none) |
| `124..128` | `u32` | `reserved` | Zero-filled (`0`) |

### 2.3 Canonical Tensor Naming in `core.bin`

| Tensor Name | Shape | Allowed `dtype` | Required? |
| :--- | :--- | :--- | :---: |
| `model.embed_tokens.weight` | `[vocab_size, D]` | `F16` or `F32` | Yes |
| `model.norm.weight` | `[D]` | `F32` | Yes |
| `lm_head.weight` | `[vocab_size, D]` | `F16`, `F32`, or `Ternary2Bit` | Unless tied |
| `layers.{l}.input_layernorm.weight` | `[D]` | `F32` | Yes |
| `layers.{l}.post_attention_layernorm.weight` | `[D]` | `F32` | Yes |
| `layers.{l}.self_attn.q_proj.weight` | `[H * d, D]` | `Ternary2Bit` or `F32` | Yes |
| `layers.{l}.self_attn.k_proj.weight` | `[H_kv * d, D]` | `Ternary2Bit` or `F32` | Yes |
| `layers.{l}.self_attn.v_proj.weight` | `[H_kv * d, D]` | `Ternary2Bit` or `F32` | Yes |
| `layers.{l}.self_attn.o_proj.weight` | `[D, H * d]` | `Ternary2Bit` or `F32` | Yes |
| `layers.{l}.self_attn.q_norm.weight` | `[H * d]` | `F32` | Yes |
| `layers.{l}.self_attn.k_norm.weight` | `[H_kv * d]` | `F32` | Yes |
| `layers.{l}.self_attn.decay_proj.weight` | `[H, D]` | `F32` | When `layer_kinds[l] == 1` |
| `layers.{l}.self_attn.decay_proj.bias` | `[H]` | `F32` | When `layer_kinds[l] == 1` |
| `layers.{l}.mlp.gate.weight` | `[E, D]` | `F32` | Yes |
| `layers.{l}.lookahead_router.weight` | `[E, D]` | `F32` or `F16` | Optional ($l < L - \Delta$) |
| `model.hint_embedding.table.weight` | `[num_buckets, D_hint]` | `F32` | Optional (Bit 3) |
| `model.hint_embedding.proj.weight` | `[D, 4 * D_hint]` | `F32` | Optional (Bit 3) |

> **4-Slot Partitioned `HintEmbedding` Indexing:**
> `num_buckets` MUST be $\ge 4$ and divisible by `4`. The embedding table is partitioned into 4 disjoint quarters of size `slot_buckets = num_buckets / 4`. When `modality_flag == 0` (`hint[0] == 0`), `HintEmbedding` is suppressed (`delta = 0`, zero residual perturbation for padding, `<eos>`, or unannotated tokens). Otherwise (`hint[0] != 0`), slot `s in {0, 1, 2, 3}` reads row:
> $$\text{bucket}(s, \text{raw\_id}) = s \cdot \text{slot\_buckets} + (\text{raw\_id} \bmod \text{slot\_buckets})$$
> (using Euclidean modulo $\in [0, \text{slot\_buckets})$ for signed negative offsets such as `relative_def_offset < 0`).


---

## 3. `experts.bin` Layout — Strict 4KB-Aligned Direct-DMA Shard

Every expert block in `experts.bin` MUST start on a **4096-byte page boundary** (`offset % 4096 == 0`) and have a padded byte length that is an **exact multiple of 4096 bytes** (`padded_bytes % 4096 == 0`) for unbuffered `O_DIRECT` (Linux `io_uring`) and `F_NOCACHE` (macOS `pread`) DMA transfers.

### 3.1 Page 0 Header (`4096` Bytes at Offset `0`)

| Byte Offset | Type | Field Name | Description |
| :---: | :---: | :--- | :--- |
| `0..4` | `[u8; 4]` | `magic` | ASCII `b"TRAE"` (`0x54, 0x52, 0x41, 0x45`) |
| `4..8` | `u32` | `version` | Format version = `1` |
| `8..12` | `u32` | `num_layers` | $L$, e.g., `16` |
| `12..16` | `u32` | `num_experts` | $E$, e.g., `64` |
| `16..20` | `u32` | `hidden_size` | $D$, e.g., `2048` |
| `20..24` | `u32` | `intermediate_size` | $I$, e.g., `1024` |
| `24..32` | `u64` | `index_offset` | Always `4096` |
| `32..40` | `u64` | `index_bytes` | `num_layers * num_experts * 40` |
| `40..48` | `u64` | `data_start_offset` | `align_up(4096 + index_bytes, 4096)` |
| `48..52` | `u32` | `index_crc32` | IEEE CRC32 over `[index_offset .. index_offset + index_bytes]` |
| `52..60` | `u64` | `expert_payload_bytes` | Exact unpadded byte size per expert block |
| `60..68` | `u64` | `expert_padded_bytes` | `align_up(expert_payload_bytes, 4096)` |
| `68..4092` | `[u8; 4024]` | `reserved` | Zero-filled (`0x00`) |
| `4092..4096` | `u32` | `header_crc32` | IEEE CRC32 over bytes `0..4092` |

### 3.2 Expert Directory (`num_layers * num_experts` Entries at Offset `4096`)

Entries are ordered row-major by `flat_idx = layer_id * num_experts + expert_id`. Each `ExpertIndexEntry` is **`40` bytes**:

| Rel Offset | Type | Field Name | Description |
| :---: | :---: | :--- | :--- |
| `0..2` | `u16` | `layer_id` | Layer index `0 .. num_layers - 1` |
| `2..4` | `u16` | `expert_id` | Expert index `0 .. num_experts - 1` |
| `4..8` | `u32` | `flags` | `1 = TernarySwiGLU` |
| `8..16` | `u64` | `offset` | File byte offset (`offset % 4096 == 0`) |
| `16..24` | `u64` | `payload_bytes` | Exact unpadded payload length in bytes |
| `24..32` | `u64` | `padded_bytes` | `align_up(payload_bytes, 4096)` (`padded_bytes % 4096 == 0`) |
| `32..36` | `u32` | `crc32` | IEEE CRC32 over `[offset .. offset + payload_bytes]` |
| `36..40` | `u32` | `reserved` | Zero-filled (`0`) |

Bytes between `4096 + index_bytes` and `data_start_offset` MUST be `0x00`.

### 3.3 Internal Layout of Each 4KB-Aligned Expert Block

For hidden size $D$ and intermediate size $I$, each expert block at `offset` packs six contiguous sub-buffers in exact order, followed by zero-padding to `padded_bytes`:

1. **`gate_proj.packed`**: `I * (D / 4)` bytes (`u8`, shape `[I, D / 4]`)
2. **`up_proj.packed`**: `I * (D / 4)` bytes (`u8`, shape `[I, D / 4]`)
3. **`down_proj.packed`**: `D * (I / 4)` bytes (`u8`, shape `[D, I / 4]`)
4. **`gate_proj.scales`**: `I * 4` bytes (`f32`, shape `[I]`)
5. **`up_proj.scales`**: `I * 4` bytes (`f32`, shape `[I]`)
6. **`down_proj.scales`**: `D * 4` bytes (`f32`, shape `[D]`)
7. **`zero_padding`**: `padded_bytes - payload_bytes` bytes of `0x00`

$$\text{payload\_bytes} = \frac{3 \cdot D \cdot I}{4} + 4 \cdot (2 I + D)$$

> **Size Verification for `OLMoE-1B-7B` ($D = 2048, I = 1024$):**
> $\text{payload\_bytes} = 3 \times 524,288 + 4 \times 4,096 = 1,572,864 + 16,384 = 1,589,248\text{ bytes} = 388 \times 4096\text{ bytes}$ (0 bytes of padding waste!).

---

## 4. Forward-Pass Mathematical Reference (PyTorch $\leftrightarrow$ Rust Parity)

To guarantee `< 1e-2` end-to-end parity between PyTorch (`train/`) and Rust (`runtime/`), both implementations execute the exact equations below:

### 4.1 RMSNorm & OLMoE QK-Norm
$$\operatorname{RMSNorm}(x, w, \varepsilon) = \frac{x}{\sqrt{\frac{1}{N}\sum_{i=0}^{N-1} x_i^2 + \varepsilon}} \odot w$$
- In OLMoE, `q_norm` (`shape = [H * d]`) and `k_norm` (`shape = [H_kv * d]`) are applied to the **full projected flat vectors** $q_{\text{flat}} = W_q u$ and $k_{\text{flat}} = W_k u$ **before** reshaping into `[H, d]` / `[H_kv, d]` heads!

### 4.2 Anchor GQA Layer (`layer_kinds[l] == 0`, e.g., Layers `7, 15`)
1. $u_t = \operatorname{RMSNorm}(h_t, w_{\text{in\_ln}}, \varepsilon)$
2. $q_t = \operatorname{reshape}_{[H, d]}(\operatorname{RMSNorm}(W_q u_t, w_{q\_\text{norm}}, \varepsilon))$, $k_t = \operatorname{reshape}_{[H_{kv}, d]}(\operatorname{RMSNorm}(W_k u_t, w_{k\_\text{norm}}, \varepsilon))$, $v_t = \operatorname{reshape}_{[H_{kv}, d]}(W_v u_t)$.
3. **Half-split RoPE** at token position $p \in \mathbb{N}_0$:
   For frequency $\theta_j = p \cdot \text{rope\_theta}^{-2j / d}$ ($j \in [0, d/2)$):
   $$\operatorname{RoPE}(z)_j = z_j \cos\theta_j - z_{j + d/2} \sin\theta_j, \qquad \operatorname{RoPE}(z)_{j + d/2} = z_{j + d/2} \cos\theta_j + z_j \sin\theta_j$$
4. Append $(k_t^{\text{rope}}, v_t)$ to anchor layer's KV cache stored as IEEE-754 `f16` (promoted to `f32` for dot products and weighted accumulation); compute causal scaled dot-product attention ($1/\sqrt{d}$) with GQA group size $g = H / H_{kv}$; project concatenated heads via $W_o$ and add residual: $h'_t = h_t + W_o o_t$.

### 4.3 Gated Linear Recurrence Layer (`layer_kinds[l] == 1`, e.g., Layers `0..6, 8..14`)
1. Compute $u_t$, $q_t \in \mathbb{R}^{H \times d}$ (post full-vector `q_norm`), $k_t \in \mathbb{R}^{H_{kv} \times d}$ (post full-vector `k_norm`), and $v_t \in \mathbb{R}^{H_{kv} \times d}$ identically to steps 1–2 above (no RoPE; decay gate supplies positional recency). Repeat $k_t, v_t$ across GQA groups to $[H, d]$.
2. Scale queries and keys by $d^{-1/2}$: $\hat{q}_t = q_t / \sqrt{d}$, $\hat{k}_t = k_t / \sqrt{d}$.
3. Compute per-head decay gate:
   $$\alpha_t = \sigma(W_{\text{decay}} u_t + b_{\text{decay}}) \in (0, 1)^H$$
4. Update fixed-size $O(1)$ recurrent state $S_t \in \mathbb{R}^{H \times d \times d}$ ($S_{-1} = 0$):
   $$S_t^{(h)} = \alpha_{t, h} \, S_{t-1}^{(h)} + \hat{k}_{t, h} \, v_{t, h}^\top, \qquad o_{t, h} = \left(S_t^{(h)}\right)^\top \hat{q}_{t, h}$$
5. Project flattened $o_t \in \mathbb{R}^{H \cdot d}$ via $W_o$ and add residual: $h'_t = h_t + W_o o_t$.

### 4.4 3-Layer Lookahead Prefetch + Sparse SwiGLU MoE Block
1. **Lookahead Trigger (if `lookahead_router` present at Layer $l$ and $l + \Delta < L$):**
   Compute $z_{\text{look}} = W_{\text{look}}^{(l)} h'_t \in \mathbb{R}^E$, take top-$k$ indices, and enqueue non-blocking async read SQEs for cold experts in Layer $l + \Delta$.
2. **MoE Routing & Execution at Layer $l$:**
   - $m_t = \operatorname{RMSNorm}(h'_t, w_{\text{post\_ln}}, \varepsilon)$
   - $g_t = \operatorname{softmax}(W_{\text{gate}}^{(l)} m_t) \in \mathbb{R}^E$ (in `f32`)
   - Select top-$k$ expert indices $I_k$ and weights $p_k$ (if ` tie `, lowest `expert_id` wins). If `FLAG_NORM_TOPK_PROB` is set, renormalize $p_k \leftarrow p_k / \sum_{j=1}^k p_{k,j}$.
   - Reap any in-flight CQEs for $(l, e \in I_k)$, then compute each active expert's SwiGLU output:
     $$\operatorname{Expert}_{l,e}(m_t) = W_{\text{down}}^{(l,e)}\!\left(\operatorname{SiLU}\!\left(W_{\text{gate}}^{(l,e)} m_t\right) \odot \left(W_{\text{up}}^{(l,e)} m_t\right)\right)$$
     where $\operatorname{SiLU}(z) = \frac{z}{1 + e^{-z}}$.
   - Combine and add residual:
     $$h^{\text{out}}_t = h'_t + \sum_{j=1}^{k} p_{k, j} \, \operatorname{Expert}_{l, I_{k,j}}(m_t)$$

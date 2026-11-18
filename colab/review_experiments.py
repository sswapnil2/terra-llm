"""No-checkpoint experiments for ARCHITECTURE_REVIEW.md (findings 2, 5, 6, 8, 10, 12, 18).

Needs only public models. Everything is zero-training except a linear lookahead probe.

  A. Teacher loss vs position on whole documents (the reference for everything else).
  B. System 7: sliding-window softmax attention on the non-anchor layers, no training.
  C. Post-training quantization of the MoE experts: 4-bit blockwise, 3-bit, ternary (L2-optimal),
     per-layer ternary sensitivity, and window + 4-bit combined.
  D. Real routing traces: popularity skew, lookahead baselines vs a trained linear probe,
     token-id table for layers 0-2, trace-driven cache simulation (LRU / pinned+LRU / Belady).
  E. Dense small models at 4-bit in the same 2 GB, compared in bits per UTF-8 byte.

Run:  python -u colab/review_experiments.py --out /content/review_results.json
Smoke: python colab/review_experiments.py --smoke   (tiny random model, CPU, seconds)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

BUCKETS = [0, 256, 512, 1024, 2048, 4096]
SEED = 1234


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- data


def load_docs(smoke: bool) -> dict[str, list[str]]:
    if smoke:
        rng = np.random.default_rng(0)
        words = ["alpha", "beta", "gamma", "delta", "x", "y", "the", "of", "="]
        return {"smoke": [" ".join(rng.choice(words, 300)) for _ in range(6)]}
    from datasets import load_dataset

    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    docs, cur = [], []
    for line in wt["text"]:
        if re.match(r"^ = [^=].* = \n?$", line) and cur:
            docs.append("".join(cur))
            cur = []
        cur.append(line)
    docs.append("".join(cur))
    wiki = [d for d in docs if len(d) > 2000]

    fw, stream = [], load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for i, row in enumerate(stream):
        if len(row["text"]) >= 9000:
            fw.append(row["text"])
        if len(fw) >= 60 or i > 60000:
            break
    return {"wikitext103_test": wiki, "fineweb_edu": fw}


def make_windows(tok, docs: list[str], ctx: int, max_windows_per_doc: int) -> tuple[list[torch.Tensor], int]:
    """Whole documents cut into windows of `ctx` tokens, each starting with a BOS/EOS start token
    so that every text token is scored. Returns (windows, utf8_bytes_scored)."""
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    windows, n_bytes = [], 0
    for d in docs:
        ids = tok(d, add_special_tokens=False)["input_ids"][: (ctx - 1) * max_windows_per_doc]
        n_bytes += len(tok.decode(ids).encode("utf-8"))
        for s in range(0, len(ids), ctx - 1):
            chunk = ids[s : s + ctx - 1]
            if len(chunk) >= 16:
                windows.append(torch.tensor([start] + chunk, dtype=torch.long))
    return windows, n_bytes


@torch.no_grad()
def score(model, windows: list[torch.Tensor], device) -> tuple[np.ndarray, np.ndarray]:
    """Per-window NLL sums (nats) and token counts per position bucket: arrays [n_windows, n_buckets]."""
    nb = len(BUCKETS) - 1
    sums, cnts = np.zeros((len(windows), nb)), np.zeros((len(windows), nb))
    edges = torch.tensor(BUCKETS[1:-1], device=device)
    for i, w in enumerate(windows):
        ids = w.to(device)[None]
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        nll = F.cross_entropy(logits, ids[0, 1:], reduction="none")
        b = torch.bucketize(torch.arange(1, ids.shape[1], device=device), edges, right=True)
        sums[i] = torch.zeros(nb, device=device, dtype=torch.float64).scatter_add_(0, b, nll.double()).cpu().numpy()
        cnts[i] = torch.bincount(b, minlength=nb).cpu().numpy()
    return sums, cnts


def summarize(sums, cnts, base=None, n_boot: int = 2000) -> dict:
    """nats/token overall and per bucket; if `base` (sums of the reference run on the same windows)
    is given, also the paired delta with a bootstrap 95% CI over windows."""
    out = {
        "tokens": int(cnts.sum()),
        "nats_per_token": float(sums.sum() / cnts.sum()),
        "by_position": {
            f"{BUCKETS[j]}-{BUCKETS[j + 1]}": (float(sums[:, j].sum() / cnts[:, j].sum()) if cnts[:, j].sum() else None)
            for j in range(cnts.shape[1])
        },
    }
    out["ppl"] = math.exp(out["nats_per_token"])
    if base is not None:
        d, n = (sums - base).sum(1), cnts.sum(1)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boots = d[idx].sum(1) / n[idx].sum(1)
        out["delta_nats"] = float(d.sum() / n.sum())
        out["delta_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        out["delta_by_position"] = {
            f"{BUCKETS[j]}-{BUCKETS[j + 1]}": (
                float((sums[:, j] - base[:, j]).sum() / cnts[:, j].sum()) if cnts[:, j].sum() else None
            )
            for j in range(cnts.shape[1])
        }
    return out


# --------------------------------------------------------------------------- B: window attention


class WindowAttention:
    """Limits chosen layers to a causal window of W tokens by replacing their attention mask."""

    def __init__(self, layers):
        self.win: dict[int, int] = {}
        self._cache: dict[tuple, torch.Tensor] = {}
        for i, layer in enumerate(layers):
            layer.self_attn.register_forward_pre_hook(self._make(i), with_kwargs=True)

    def _make(self, idx: int):
        def hook(_mod, args, kwargs):
            w = self.win.get(idx)
            if w is None:
                return None
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            key = (hs.shape[1], w, hs.dtype, hs.device)
            if key not in self._cache:
                self._cache.clear()
                i = torch.arange(hs.shape[1], device=hs.device)
                dist = i[:, None] - i[None, :]
                mask = torch.zeros(hs.shape[1], hs.shape[1], dtype=hs.dtype, device=hs.device)
                mask.masked_fill_((dist < 0) | (dist >= w), torch.finfo(hs.dtype).min)
                self._cache[key] = mask[None, None]
            kwargs["attention_mask"] = self._cache[key]
            return args, kwargs

        return hook


# --------------------------------------------------------------------------- C: quantizers


def quant_block(w: torch.Tensor, bits: int, block: int = 32) -> torch.Tensor:
    """Symmetric uniform quantization with one abs-max scale per `block` weights along the last
    dim (llama.cpp Q4_0-style: bits + 16/block bits per weight)."""
    shp, x = w.shape, w.float()
    if shp[-1] % block:
        block = shp[-1]
    x = x.reshape(-1, shp[-1] // block, block)
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    return (torch.round(x / s).clamp(-qmax - 1, qmax) * s).reshape(shp).to(w.dtype)


def quant_ternary(w: torch.Tensor) -> torch.Tensor:
    """absmean ternary pattern with the L2-optimal per-row scale (the project's inference form)."""
    x = w.float()
    g = x.abs().mean(-1, keepdim=True).clamp_min(1e-12)
    t = torch.round((x / g).clamp(-1, 1))
    gamma = (x * t).sum(-1, keepdim=True) / t.pow(2).sum(-1, keepdim=True).clamp_min(1.0)
    return (gamma * t).to(w.dtype)


def rel_mse(w, q) -> float:
    return float(((w.float() - q.float()) ** 2).sum() / (w.float() ** 2).sum())


class ExpertQuant:
    """Quantizes MoE expert weights in place, keeping the originals on the CPU."""

    def __init__(self, layers):
        self.layers = layers
        self.orig = [
            (l.mlp.experts.gate_up_proj.detach().cpu().clone(), l.mlp.experts.down_proj.detach().cpu().clone())
            for l in layers
        ]

    @torch.no_grad()
    def apply(self, fn, only: set[int] | None = None) -> float:
        errs = []
        for i, layer in enumerate(self.layers):
            ex = layer.mlp.experts
            for p, o in ((ex.gate_up_proj, self.orig[i][0]), (ex.down_proj, self.orig[i][1])):
                if only is not None and i not in only:
                    p.copy_(o.to(p.device))
                    continue
                for s in range(0, o.shape[0], 8):
                    src = o[s : s + 8].to(p.device)
                    q = fn(src)
                    if s == 0:
                        errs.append(rel_mse(src, q))
                    p[s : s + 8].copy_(q)
        return float(np.mean(errs)) if errs else 0.0

    def restore(self) -> None:
        self.apply(lambda w: w, only=set())


# --------------------------------------------------------------------------- D: routing


class Tracer:
    """Records top-k expert ids per (token, layer), and the post-attention residual at `h_layers`."""

    def __init__(self, layers, h_layers: list[int], max_h_tokens: int):
        self.on, self.idx, self.h = False, [[] for _ in layers], {l: [] for l in h_layers}
        self.max_h, self.n_h = max_h_tokens, 0
        for i, layer in enumerate(layers):
            layer.mlp.gate.register_forward_hook(self._gate(i))
            if i in self.h:
                layer.post_attention_layernorm.register_forward_pre_hook(self._h(i))

    def _gate(self, i):
        def hook(_m, _a, out):
            if self.on:
                self.idx[i].append(out[2].to(torch.uint8).cpu())

        return hook

    def _h(self, i):
        def hook(_m, args):
            if self.on and self.n_h < self.max_h:
                self.h[i].append(args[0][0].to(torch.float16).cpu())
                if i == max(self.h):
                    self.n_h += args[0].shape[1]

        return hook

    def traces(self) -> np.ndarray:  # [tokens, layers, k]
        return torch.stack([torch.cat(x) for x in self.idx], 1).numpy()


def recall_at_k(pred: np.ndarray, true: np.ndarray, n_exp: int) -> float:
    """Mean |pred ∩ true| / k for id arrays [N, k]."""
    hit = np.zeros((len(true), n_exp), bool)
    np.put_along_axis(hit, true.astype(np.int64), True, 1)
    return float(np.take_along_axis(hit, pred.astype(np.int64), 1).mean())


def routing_analysis(tr: np.ndarray, tok_ids: np.ndarray, h: dict[int, torch.Tensor], n_exp: int, off: int, device) -> dict:
    n, n_layers, k = tr.shape
    split = int(n * 0.8)
    out: dict = {"tokens": int(n), "chance_recall": k / n_exp}

    counts = np.stack([np.bincount(tr[:split, l].ravel(), minlength=n_exp) for l in range(n_layers)])
    share = np.sort(counts, 1)[:, ::-1].cumsum(1) / counts.sum(1, keepdims=True)
    out["share_of_acquisitions_from_top_experts_mean_over_layers"] = {
        f"top{m}": float(share[:, m - 1].mean()) for m in (8, 16, 32) if m <= n_exp
    }
    p = counts / counts.sum(1, keepdims=True)
    out["routing_entropy_bits_mean"] = float(-(p * np.log2(p.clip(1e-12))).sum(1).mean())
    out["routing_entropy_bits_uniform"] = math.log2(n_exp)

    pop = np.argsort(-counts, 1)[:, :k]  # static popularity top-k per layer, fit on the first 80%
    te = slice(split, n)
    out["recall_static_popularity"] = float(
        np.mean([recall_at_k(np.broadcast_to(pop[l], tr[te, l].shape), tr[te, l], n_exp) for l in range(n_layers)])
    )
    out["recall_previous_token_same_layer"] = float(
        np.mean([recall_at_k(tr[split - 1 : n - 1, l], tr[te, l], n_exp) for l in range(n_layers)])
    )
    out["recall_same_token_layer_minus_offset"] = float(
        np.mean([recall_at_k(tr[te, l - off], tr[te, l], n_exp) for l in range(off, n_layers)])
    )

    # Token-id table for the layers the lookahead cannot reach (finding 18).
    tab = {}
    for l in range(min(off, n_layers)):
        votes: dict[int, np.ndarray] = {}
        for t, row in zip(tok_ids[:split], tr[:split, l]):
            votes.setdefault(int(t), np.zeros(n_exp, np.int32))[row] += 1
        pred = np.stack([np.argsort(-votes[int(t)])[:k] if int(t) in votes else pop[l] for t in tok_ids[te]])
        tab[f"layer{l}"] = recall_at_k(pred, tr[te, l], n_exp)
    out["recall_token_id_table_layers_below_offset"] = tab
    out["token_id_table_unseen_token_rate"] = float(np.mean(~np.isin(tok_ids[te], np.unique(tok_ids[:split]))))

    # Learned linear lookahead (same form as train/lookahead_router.py: bias-free linear), trained here.
    probe = {}
    for l, hl in h.items():
        m = hl.shape[0]
        sp = int(m * 0.8)
        y = torch.zeros(m, n_exp)
        y.scatter_(1, torch.from_numpy(tr[:m, l + off].astype(np.int64)), 1.0)
        x = hl.float()
        x = x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        lin = torch.nn.Linear(x.shape[1], n_exp, bias=False).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(SEED)
        for _ in range(30):
            for b in torch.randperm(sp, generator=g).split(2048):
                loss = F.binary_cross_entropy_with_logits(lin(x[b].to(device)), y[b].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred = lin(x[sp:].to(device)).topk(k, -1).indices.cpu().numpy()
        probe[f"layer{l}_to_{l + off}"] = {
            "learned_linear": recall_at_k(pred, tr[sp:m, l + off], n_exp),
            "static_popularity": recall_at_k(np.broadcast_to(pop[l + off], pred.shape), tr[sp:m, l + off], n_exp),
            "train_tokens": sp,
            "test_tokens": m - sp,
        }
    out["lookahead_probe"] = probe
    return out


def cache_sim(tr: np.ndarray, n_exp: int, slots_list: list[int], expert_mib: float) -> dict:
    """Trace-driven cache simulation over (layer, expert) accesses in execution order."""
    try:
        from numba import njit
    except ImportError:  # smoke test without numba

        def njit(f):
            return f

    n, n_layers, k = tr.shape
    acc = (tr.astype(np.int64) + (np.arange(n_layers) * n_exp)[None, :, None]).reshape(-1)
    total = n_layers * n_exp
    split = int(n * 0.2) * n_layers * k  # popularity is fit on the first 20%, policies scored on the rest
    freq = np.bincount(acc[:split], minlength=total)
    order = np.argsort(-freq)
    ev = acc[split:]

    @njit
    def lru(ev, total, cap, pinned):
        last = np.full(total, -1, np.int64)
        res = pinned.copy()
        used, miss = int(pinned.sum()), 0
        for t in range(len(ev)):
            e = ev[t]
            if not res[e]:
                miss += 1
                if used >= cap:
                    v, best = -1, 1 << 62
                    for j in range(total):
                        if res[j] and not pinned[j] and last[j] < best:
                            best, v = last[j], j
                    if v < 0:
                        continue  # everything pinned: read and discard
                    res[v] = False
                    used -= 1
                res[e] = True
                used += 1
            last[e] = t
        return miss

    @njit
    def belady(ev, total, cap):
        nxt = np.empty(len(ev), np.int64)
        seen = np.full(total, 1 << 62, np.int64)
        for t in range(len(ev) - 1, -1, -1):
            nxt[t] = seen[ev[t]]
            seen[ev[t]] = t
        nu = np.full(total, -1, np.int64)
        res = np.zeros(total, np.bool_)
        used, miss = 0, 0
        for t in range(len(ev)):
            e = ev[t]
            if not res[e]:
                miss += 1
                if used >= cap:
                    v, far = -1, -1
                    for j in range(total):
                        if res[j] and nu[j] > far:
                            far, v = nu[j], j
                    res[v] = False
                    used -= 1
                res[e] = True
                used += 1
            nu[e] = nxt[t]
        return miss

    tokens = len(ev) / (n_layers * k)
    out = {"scored_tokens": int(tokens), "accesses_per_token": n_layers * k, "note": "cold start at the beginning of the scored segment"}
    for cap in slots_list:
        cap = min(cap, total)
        none = np.zeros(total, np.bool_)
        half = np.zeros(total, np.bool_)
        half[order[: cap // 2]] = True
        allp = np.zeros(total, np.bool_)
        allp[order[:cap]] = True
        r = {
            "lru": lru(ev, total, cap, none),
            "pin_top_half_plus_lru": lru(ev, total, cap, half),
            "static_pin_only": int((~allp[ev]).sum()),
            "belady_optimal": belady(ev, total, cap),
        }
        out[f"slots_{cap}"] = {
            p: {"miss_rate": m / len(ev), "mib_read_per_token": m * expert_mib / tokens} for p, m in r.items()
        }
    return out


# --------------------------------------------------------------------------- main


def load_olmoe(smoke: bool, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if smoke:
        from transformers import OlmoeConfig, OlmoeForCausalLM

        class Tok:
            bos_token_id, eos_token_id = None, 1

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [2 + (hash(w) % 60) for w in text.split()]}

            def decode(self, ids):
                return " ".join(str(i) for i in ids)

        torch.manual_seed(0)
        cfg = OlmoeConfig(
            vocab_size=64, hidden_size=64, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
            num_key_value_heads=4, num_experts=8, num_local_experts=8, num_experts_per_tok=2, max_position_embeddings=512,
        )
        return OlmoeForCausalLM(cfg).eval(), Tok()
    name = "allenai/OLMoE-1B-7B-0125"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="review_results.json")
    ap.add_argument("--skip-dense", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = 128 if args.smoke else 4096
    res: dict = {
        "meta": {
            "seed": SEED, "ctx": ctx, "torch": torch.__version__, "buckets": BUCKETS,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    import transformers

    res["meta"]["transformers"] = transformers.__version__

    def save():
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)

    docs = load_docs(args.smoke)
    model, tok = load_olmoe(args.smoke, device)
    layers = model.model.layers
    n_layers, n_exp, top_k = len(layers), model.config.num_experts, model.config.num_experts_per_tok
    anchors = {n_layers // 2 - 1, n_layers - 1}
    off = 3
    data = {}
    for name, d in docs.items():
        w, nb = make_windows(tok, d, ctx, 8)
        data[name] = (w, nb)
        log(f"{name}: {len(d)} docs, {len(w)} windows, {sum(len(x) for x in w)} tokens, {nb} bytes")
    res["meta"]["data"] = {k: {"windows": len(v[0]), "tokens": int(sum(len(x) for x in v[0])), "bytes": v[1]} for k, v in data.items()}

    win = WindowAttention(layers)
    h_layers = [l for l in (0, 4, 8, 12) if l + off < n_layers]
    tracer = Tracer(layers, h_layers, 16 if args.smoke else 100_000)

    # ---- A: baseline (also records routing traces)
    tracer.on = True
    base = {}
    for name, (w, nb) in data.items():
        base[name] = score(model, w, device)
    tracer.on = False
    res["A_teacher"] = {}
    for name, (s, c) in base.items():
        r = summarize(s, c)
        r["bits_per_byte"] = float(s.sum() / math.log(2) / data[name][1])
        res["A_teacher"][name] = r
        log(f"A teacher {name}: {r['nats_per_token']:.4f} nats/tok, ppl {r['ppl']:.3f}, {r['bits_per_byte']:.4f} bits/byte")
    save()

    def variant(section: str, label: str, subset: int | None = None) -> None:
        res.setdefault(section, {})[label] = {}
        for name, (w, _) in data.items():
            ww = w[:subset] if subset else w
            s, c = score(model, ww, device)
            b = base[name][0][: len(ww)]
            r = summarize(s, c, b)
            res[section][label][name] = r
            log(f"{section} {label} {name}: {r['nats_per_token']:.4f} (delta {r['delta_nats']:+.4f} CI {r['delta_ci95'][0]:+.4f}..{r['delta_ci95'][1]:+.4f})")
        save()

    # ---- B: sliding-window attention, zero training
    non_anchor = [l for l in range(n_layers) if l not in anchors]
    widths = [16, 32] if args.smoke else [64, 128, 256, 512, 1024, 2048]
    for wd in widths:
        win.win = {l: wd for l in non_anchor}
        variant("B_window", f"W{wd}_on_{len(non_anchor)}_layers_anchors_{sorted(anchors)}")
    for wd in widths[2:4] if not args.smoke else widths[:1]:
        win.win = {l: wd for l in range(n_layers)}
        variant("B_window", f"W{wd}_on_all_{n_layers}_layers_no_anchor")
        every4 = {l for l in range(n_layers) if l % 4 == 3}
        win.win = {l: wd for l in range(n_layers) if l not in every4}
        variant("B_window", f"W{wd}_anchors_every_4_{sorted(every4)}")
    win.win = {}

    # ---- D: routing (before weights are touched)
    tr = tracer.traces()
    tok_ids = torch.cat([w for (ws, _) in data.values() for w in ws]).numpy()
    assert len(tok_ids) == len(tr), (len(tok_ids), len(tr))
    h = {l: torch.cat(v)[: tracer.n_h] for l, v in tracer.h.items()}
    tracer.h, tracer.idx = {}, []
    res["D_routing"] = routing_analysis(tr, tok_ids, h, n_exp, off, device)
    del h
    log("D routing: " + json.dumps({k: v for k, v in res["D_routing"].items() if not isinstance(v, dict)}))
    log("D probe: " + json.dumps(res["D_routing"]["lookahead_probe"]))
    # Slots for 0.75 / 1 / 1.5 / 2 GiB with an f16 core (472) + 4k KV (64) + state (28), 1.516 MiB per expert.
    slots = [4, 8] if args.smoke else [int((b - 564) / 1.516) for b in (768, 1024, 1536, 2048)]
    res["D_cache_sim"] = cache_sim(tr[: 4000 if args.smoke else 150_000], n_exp, slots, 1.516)
    log("D cache sim: " + json.dumps(res["D_cache_sim"]))
    save()

    # ---- C: expert quantization, zero training
    eq = ExpertQuant(layers)
    for label, fn in (
        ("int4_block32", lambda w: quant_block(w, 4)),
        ("int3_block32", lambda w: quant_block(w, 3)),
        ("ternary_l2", quant_ternary),
    ):
        err = eq.apply(fn)
        variant("C_expert_ptq", label, subset=None if label.startswith("int") else 24)
        res["C_expert_ptq"][label]["rel_weight_mse"] = err
    sub = 4 if args.smoke else 12
    for l in range(n_layers):
        eq.apply(quant_ternary, only={l})
        variant("C_ternary_one_layer", f"layer{l}", subset=sub)
    if not args.smoke:
        eq.apply(lambda w: quant_block(w, 4))
        win.win = {l: 512 for l in non_anchor}
        variant("C_combined", "W512_on_14_layers_plus_int4_experts")
        win.win = {}
    eq.restore()
    save()

    # ---- E: dense small models at 4-bit in the same RAM (bits per byte; tokenizers differ)
    if not args.smoke and not args.skip_dense:
        del model, eq, layers, win, tracer
        torch.cuda.empty_cache()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        res["E_dense"] = {}
        for name in ("Qwen/Qwen2.5-1.5B", "HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen2.5-3B"):
            try:
                tk = AutoTokenizer.from_pretrained(name)
                m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
                n_params = sum(p.numel() for p in m.parameters())
                dctx = min(ctx, getattr(m.config, "max_position_embeddings", ctx))
                wins = {k: make_windows(tk, d, dctx, 8) for k, d in docs.items()}
                entry = {"params": n_params, "ctx": dctx, "est_q4_0_gib": n_params * 4.5 / 8 / 2**30}
                for prec in ("bf16", "int4_block32"):
                    if prec != "bf16":
                        with torch.no_grad():
                            for mod_name, mod in m.named_modules():
                                if isinstance(mod, torch.nn.Linear) and "lm_head" not in mod_name:
                                    mod.weight.copy_(quant_block(mod.weight, 4))
                    entry[prec] = {}
                    for k, (w, nb) in wins.items():
                        s, c = score(m, w, device)
                        entry[prec][k] = {"bits_per_byte": float(s.sum() / math.log(2) / nb), "nats_per_token": float(s.sum() / c.sum()), "tokens": int(c.sum())}
                        log(f"E {name} {prec} {k}: {entry[prec][k]['bits_per_byte']:.4f} bits/byte")
                res["E_dense"][name] = entry
                del m
                torch.cuda.empty_cache()
            except Exception as exc:  # one unavailable model must not lose the rest of the run
                res["E_dense"][name] = {"error": repr(exc)}
                log(f"E {name} FAILED: {exc!r}")
            save()

    res["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    print("=====RESULTS_JSON_BEGIN=====")
    print(json.dumps(res))
    print("=====RESULTS_JSON_END=====", flush=True)


if __name__ == "__main__":
    main()

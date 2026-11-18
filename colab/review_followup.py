"""Follow-up to review_experiments.py (run in the same directory, zero training):

F  candidate teachers: bits/byte of larger dense models (BF16, ctx 4096) on the same documents,
   plus an FP32 re-check of Qwen2.5-3B whose first-run numbers looked anomalous.
G  OLMoE with *every* Linear at int4_block32 (experts + attention + router), the like-for-like
   comparison against the int4 dense baselines of section E.
H  cache simulation re-sized for int4 / int3 experts (3.375 / 2.625 MiB each) on the same routing traces.

    python review_followup.py --out review_followup.json
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time

import numpy as np
import torch

import review_experiments as R

TEACHERS = ["Qwen/Qwen2.5-7B", "microsoft/phi-4", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B"]


def bpb(model, wins, device) -> dict:
    out = {}
    for k, (w, nb) in wins.items():
        s, c = R.score(model, w, device)
        per_win = s.sum(1) / c.sum(1)
        out[k] = {
            "bits_per_byte": float(s.sum() / math.log(2) / nb),
            "nats_per_token": float(s.sum() / c.sum()),
            "tokens": int(c.sum()),
            # an exploding window (numerical trouble) shows up here, not in the mean
            "window_nats_median": float(np.median(per_win)),
            "window_nats_max": float(per_win.max()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="review_followup.json")
    args = ap.parse_args()
    device = torch.device("cuda")
    ctx = 4096
    res: dict = {"meta": {"ctx": ctx, "gpu": torch.cuda.get_device_name(0), "started": time.strftime("%Y-%m-%d %H:%M:%S")}}

    def save():
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)

    docs = R.load_docs(False)

    # ---- G + H: OLMoE
    model, tok = R.load_olmoe(False, device)
    layers = model.model.layers
    n_exp = model.config.num_experts
    data = {k: R.make_windows(tok, d, ctx, 8) for k, d in docs.items()}
    tracer = R.Tracer(layers, [], 0)
    tracer.on = True
    res["G_olmoe"] = {"bf16": bpb(model, data, device)}
    tracer.on = False
    tr = tracer.traces()[:150_000]
    R.log("G olmoe bf16: " + json.dumps(res["G_olmoe"]["bf16"]))

    res["H_cache_sim"] = {}
    for label, mib in (("int4_block32", 3.375), ("int3_block32", 2.625)):
        slots = [int((b - 564) / mib) for b in (2048, 3072)]
        res["H_cache_sim"][label] = {"expert_mib": mib, "sim": R.cache_sim(tr, n_exp, slots, mib)}
        R.log(f"H {label}: " + json.dumps(res["H_cache_sim"][label]))
    save()

    eq = R.ExpertQuant(layers)
    eq.apply(lambda w: R.quant_block(w, 4))
    res["G_olmoe"]["int4_experts_only"] = bpb(model, data, device)
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
                mod.weight.copy_(R.quant_block(mod.weight, 4))
    res["G_olmoe"]["int4_all_linear"] = bpb(model, data, device)
    for k in ("int4_experts_only", "int4_all_linear"):
        R.log(f"G olmoe {k}: " + json.dumps(res["G_olmoe"][k]))
    save()
    del model, eq, layers, tracer
    torch.cuda.empty_cache()

    # ---- F: candidate teachers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    res["F_teachers"] = {}
    for name, dtype in [("Qwen/Qwen2.5-3B", torch.float32)] + [(n, torch.bfloat16) for n in TEACHERS]:
        key = f"{name}:{str(dtype).split('.')[-1]}"
        try:
            tk = AutoTokenizer.from_pretrained(name)
            m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
            wins = {k: R.make_windows(tk, d, ctx, 8) for k, d in docs.items()}
            res["F_teachers"][key] = {"params": sum(p.numel() for p in m.parameters()), **bpb(m, wins, device)}
            R.log(f"F {key}: " + json.dumps(res["F_teachers"][key]))
            del m
            torch.cuda.empty_cache()
        except Exception as exc:  # one unavailable model must not lose the rest of the run
            res["F_teachers"][key] = {"error": repr(exc)}
            R.log(f"F {key} FAILED: {exc!r}")
        shutil.rmtree(f"/root/.cache/huggingface/hub/models--{name.replace('/', '--')}", ignore_errors=True)
        save()

    res["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()


if __name__ == "__main__":
    main()

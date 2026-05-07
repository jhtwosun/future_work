"""
EXP-step5: step-branching with NEW info-theoretic triggers.

Goal: test whether the new info-theoretic per-step signals
(top1_margin_min, entropy_max, tempered_kl_max, kl_to_uniform_min)
make better step-branching triggers than lp_min.

Procedure:
1. Greedy decode 200 MATH-500 with logprobs=20.
2. For each trajectory, compute per-step features.
3. For each candidate trigger, find the WORST step under that signal.
4. K=4 resample at that step at T=0.7, majority-vote aggregate.
5. Compare accuracy across triggers + against lp_min baseline.

Output: SX_step_new_triggers.json with all triggers' accuracy.
"""

import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, normalize, equal_strict

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_QUESTIONS = int(os.environ.get("NQ", "200"))
TOPK = 20
K = 4
T = 0.7
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / "SX_step_new_triggers.json"

PROMPT_BASE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
)


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def topk_to_distribution(topk_dict):
    if not topk_dict:
        return None
    entries = sorted(topk_dict.values(), key=lambda v: -v.logprob)
    lps = np.array([e.logprob for e in entries], dtype=float)
    probs = np.exp(lps - lps.max())
    probs = probs / probs.sum()
    return {"probs": probs.tolist(), "lps": lps.tolist()}


def js_divergence(p, q):
    p = np.asarray(p); q = np.asarray(q)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        if mask.sum() == 0: return 0.0
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())
    return 0.5 * (_kl(p, m) + _kl(q, m))


def step_entropy(distribs):
    if not distribs: return float("nan")
    ents = []
    for d in distribs:
        if d is None: continue
        p = np.array(d["probs"])
        p = p[p > 0]
        if len(p) > 0:
            ents.append(-float((p * np.log(p)).sum()))
    return float(np.mean(ents)) if ents else float("nan")


def step_top1_margin(distribs):
    if not distribs: return float("nan")
    margins = []
    for d in distribs:
        if d is None or len(d["lps"]) < 2: continue
        margins.append(d["lps"][0] - d["lps"][1])
    return float(np.mean(margins)) if margins else float("nan")


def step_tempered_kl(distribs, T1=1.0, T2=2.0):
    if not distribs: return float("nan")
    js_list = []
    for d in distribs:
        if d is None: continue
        lps = np.asarray(d["lps"])
        p1 = np.exp(lps / T1 - (lps / T1).max()); p1 = p1 / p1.sum()
        p2 = np.exp(lps / T2 - (lps / T2).max()); p2 = p2 / p2.sum()
        js_list.append(js_divergence(p1, p2))
    return float(np.mean(js_list)) if js_list else float("nan")


def step_kl_to_uniform(distribs):
    if not distribs: return float("nan")
    vals = []
    for d in distribs:
        if d is None: continue
        p = np.array(d["probs"])
        p = p[p > 0]
        if len(p) > 0:
            H = -float((p * np.log(p)).sum())
            log_k = math.log(len(p))
            vals.append(log_k - H)
    return float(np.mean(vals)) if vals else float("nan")


def step_lp_mean(token_lps_seg):
    valid = [lp for lp in token_lps_seg if not (lp != lp)]
    return float(np.mean(valid)) if valid else float("nan")


def main():
    print("Loading MATH-500...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    use = ds.select(range(min(N_QUESTIONS, len(ds))))
    questions = [r["problem"] for r in use]
    golds = [r["answer"] for r in use]
    print(f"  {len(questions)} problems", flush=True)

    print(f"Loading {MODEL}...", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2560,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tok = llm.get_tokenizer()
    base_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_BASE.format(question=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]

    # Greedy with TOP-K logprobs
    print("=== Greedy (logprobs=20) ===", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1536, logprobs=TOPK, seed=SEED)
    t0 = time.time()
    outs = llm.generate(base_prompts, sp)
    print(f"  generated in {time.time()-t0:.1f}s", flush=True)

    greedy_records = []
    for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
        o = out.outputs[0]
        text = o.text
        tok_ids = list(o.token_ids)
        tok_strs = [tok.decode([tid]) for tid in tok_ids]
        chosen_lps = []
        distribs = []
        for step_lp in o.logprobs or []:
            if step_lp is None:
                chosen_lps.append(float("nan")); distribs.append(None); continue
            entries = sorted(step_lp.items(), key=lambda kv: -kv[1].logprob)
            chosen_lps.append(entries[0][1].logprob)
            distribs.append(topk_to_distribution(step_lp))
        boundaries = find_step_boundaries(tok_strs)
        n = len(tok_ids)
        bdy = boundaries + [n]

        # Per-step features
        step_features = []
        for sa, sb in zip(bdy[:-1], bdy[1:]):
            sd = distribs[sa:sb]
            sl = chosen_lps[sa:sb]
            step_features.append({
                "lp_mean": step_lp_mean(sl),
                "entropy_mean": step_entropy(sd),
                "top1_margin_mean": step_top1_margin(sd),
                "tempered_kl_mean": step_tempered_kl(sd),
                "kl_uniform_mean": step_kl_to_uniform(sd),
            })

        pred = extract_pred(text)
        ok = int(equal_strict(pred, gold))
        greedy_records.append({
            "id": i, "gold": gold, "pred": pred, "correct": ok,
            "tok_ids": tok_ids, "boundaries": boundaries, "step_features": step_features,
            "text": text, "n_tokens": n,
        })
    n_greedy = sum(r["correct"] for r in greedy_records)
    print(f"  Greedy: {n_greedy}/{len(questions)} = {n_greedy/len(questions):.3f}", flush=True)

    # Define triggers
    # Each trigger maps step_features -> worst-step idx (the one to branch from).
    # Convention: lower = worse (we want to branch where signal indicates problem).
    # For lp_mean / top1_margin_mean / kl_uniform_mean: low = uncertain, branch there (argmin).
    # For entropy_mean / tempered_kl_mean: high = uncertain, branch there (argmax).
    triggers = {
        "lp_min": ("lp_mean", "argmin"),
        "top1_margin_min": ("top1_margin_mean", "argmin"),
        "entropy_max": ("entropy_mean", "argmax"),
        "tempered_kl_max": ("tempered_kl_mean", "argmax"),
        "kl_uniform_min": ("kl_uniform_mean", "argmin"),
    }

    summary = {"greedy_acc": n_greedy / len(questions), "n": len(questions),
                "K": K, "T": T, "model": MODEL}
    summary["triggers"] = {}

    # For each trigger, build branch prompts and resample.
    for trig_name, (feat_key, op) in triggers.items():
        print(f"\n=== Trigger: {trig_name} (feat={feat_key}, op={op}) ===", flush=True)
        branch_prompts = []
        ids = []
        trigger_steps = []
        for r in greedy_records:
            feats = [s.get(feat_key, float("nan")) for s in r["step_features"]]
            valid = [(i, f) for i, f in enumerate(feats) if not (f != f)]
            if len(valid) < 1: continue
            if op == "argmin":
                idx = min(valid, key=lambda kv: kv[1])[0]
            else:
                idx = max(valid, key=lambda kv: kv[1])[0]
            if idx >= len(r["boundaries"]): continue
            bdy_pos = r["boundaries"][idx]
            prefix_text = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
            branch_prompts.append(base_prompts[r["id"]] + prefix_text)
            ids.append(r["id"])
            trigger_steps.append(idx)
        if not branch_prompts:
            continue
        sp_b = SamplingParams(n=K, temperature=T, top_p=0.95, max_tokens=1024, seed=SEED + 100)
        t0 = time.time()
        outs_b = llm.generate(branch_prompts, sp_b)
        dur = time.time() - t0

        n_correct = 0
        for ii, out_b in zip(ids, outs_b):
            r = greedy_records[ii]
            alt_preds = [extract_pred(c.text) for c in out_b.outputs]
            preds_clean = [p for p in alt_preds if p is not None]
            combined = [r["pred"]] + preds_clean
            counter = Counter([str(normalize(p)) for p in combined if p is not None])
            top, _ = counter.most_common(1)[0] if counter else (None, 0)
            n_correct += int(equal_strict(top, r["gold"]))

        acc = n_correct / len(ids)
        delta = acc - (n_greedy / len(questions))
        summary["triggers"][trig_name] = {
            "n": len(ids), "acc": acc, "delta_vs_greedy": delta,
            "wallclock_sec": dur,
            "mean_trigger_step_idx": float(np.mean(trigger_steps)) if trigger_steps else float("nan"),
        }
        print(f"  acc={acc:.3f}  Δ greedy={delta:+.3f}  wall={dur:.1f}s  mean_step={np.mean(trigger_steps):.1f}", flush=True)

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Final ===")
    print(f"  greedy: {n_greedy/len(questions):.3f}")
    for tn in sorted(summary["triggers"].keys(), key=lambda k: -summary["triggers"][k]["acc"]):
        td = summary["triggers"][tn]
        print(f"  {tn:20s} acc={td['acc']:.3f}  Δ={td['delta_vs_greedy']:+.3f}  n={td['n']}  meanStep={td.get('mean_trigger_step_idx', 'nan'):.1f}")
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

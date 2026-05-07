"""
EXP-PARETO: gate_frac × K combination Pareto frontier.

Test all combinations of (gate_score, gate_frac, K) to map cost-accuracy frontier:
- gate_score: entropy_mean, lp_min, combined (union of 3 signals)
- gate_frac: 0.25, 0.50, 0.75 (flag bottom 25/50/75%)
- K: 2, 4, 8

Plus baselines:
- always_K (no gate, K=2/4/8)
- vanilla (no rejection)

Total: 3 × 3 × 3 + 4 = 31 strategies per (model, dataset).

Env vars: MODEL, TAG, DATASET, NQ
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
from robust_eval import extract_pred, equal_strict, normalize

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
TAG = os.environ.get("TAG", "qwen25_7b")
DATASET = os.environ.get("DATASET", "math500")
N_PER_DS = int(os.environ.get("NQ", "200"))
SEED = 0
TOPK = 20

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / f"SX_pareto_{TAG}_{DATASET}.json"

PROMPT_MATH = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\nProblem: {question}\n\n"
)

EQUATION_RE = re.compile(r"([^\s=]+)\s*=\s*([^\s=]+)")


def _lp_of(e):
    if hasattr(e, "logprob"):
        return e.logprob
    if isinstance(e, (tuple, list)) and len(e) > 0:
        return float(e[0])
    return float(e) if isinstance(e, (int, float)) else float("nan")


def load_dataset_uniform(name, n):
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["problem"], "gold": r["answer"]} for r in ds]
    elif name == "aime":
        ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["Question"], "gold": str(r["Answer"])} for r in ds]
    elif name == "olympiad":
        ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["question"], "gold": str(r["final_answer"][0]) if r["final_answer"] else ""} for r in ds]
    raise ValueError(name)


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def step_lp_means(chosen_lps, boundaries, n):
    bdy = boundaries + [n]
    out = []
    for sa, sb in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in chosen_lps[sa:sb] if not (lp != lp)]
        out.append(float(np.mean(seg)) if seg else float("nan"))
    return out


def majority_vote(preds_list):
    votes = [str(normalize(p)) for p in preds_list if p is not None]
    if not votes: return None
    return Counter(votes).most_common(1)[0][0]


def count_arith_violations(text):
    n_v = 0
    for m in EQUATION_RE.finditer(text or ""):
        lhs, rhs = m.group(1).strip(), m.group(2).strip()
        try:
            if re.match(r"^[\d\s\.\+\-\*\/\(\)]+$", lhs) and re.match(r"^[\d\s\.\+\-\*\/\(\)]+$", rhs):
                lv = eval(lhs, {"__builtins__": {}}, {})
                rv = eval(rhs, {"__builtins__": {}}, {})
                if abs(float(lv) - float(rv)) >= 1e-6:
                    n_v += 1
        except Exception:
            pass
    return n_v


def main():
    print(f"=== Pareto: {MODEL} on {DATASET} ===", flush=True)
    t0 = time.time()
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=2560, tensor_parallel_size=1, seed=SEED, trust_remote_code=True)
    tok = llm.get_tokenizer()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    ds = load_dataset_uniform(DATASET, N_PER_DS)
    base_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_MATH.format(question=r["q"])}],
            tokenize=False, add_generation_prompt=True,
        )
        for r in ds
    ]

    # Phase 1: greedy
    print(f"\n[Phase 1] greedy decode...", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1536, logprobs=TOPK, seed=SEED)
    t0 = time.time()
    outs = llm.generate(base_prompts, sp)
    print(f"  generated in {time.time()-t0:.1f}s", flush=True)

    greedy_records = []
    for i, (rec, out) in enumerate(zip(ds, outs)):
        o = out.outputs[0]
        text = o.text
        tok_ids = list(o.token_ids)
        tok_strs = [tok.decode([tid]) for tid in tok_ids]
        chosen_lps = []; distribs = []
        for step_lp in o.logprobs or []:
            if step_lp is None:
                chosen_lps.append(float("nan")); distribs.append(None); continue
            try:
                items = list(step_lp.items())
                items.sort(key=lambda kv: -_lp_of(kv[1]))
                if items:
                    chosen_lps.append(_lp_of(items[0][1]))
                    entries_lps = np.array([_lp_of(it[1]) for it in items], dtype=float)
                    probs = np.exp(entries_lps - entries_lps.max()); probs = probs / probs.sum()
                    distribs.append({"probs": probs.tolist(), "lps": entries_lps.tolist()})
                else:
                    chosen_lps.append(float("nan")); distribs.append(None)
            except Exception:
                chosen_lps.append(float("nan")); distribs.append(None)
        boundaries = find_step_boundaries(tok_strs)
        n = len(tok_ids)
        step_lps = step_lp_means(chosen_lps, boundaries, n)
        valid_lps = [v for v in step_lps if not (v != v)]
        lp_min_score = float(np.min(valid_lps)) if valid_lps else float("inf")
        ents = []
        for d in distribs:
            if d is None: continue
            p = np.array(d["probs"]); p = p[p > 0]
            if len(p) > 0:
                ents.append(-float((p * np.log(p)).sum()))
        entropy_mean = float(np.mean(ents)) if ents else float("nan")
        valid_lp_pairs = [(j, v) for j, v in enumerate(step_lps) if not (v != v)]
        worst_lp_step = min(valid_lp_pairs, key=lambda kv: kv[1])[0] if valid_lp_pairs else 0
        n_arith = count_arith_violations(text)
        pred = extract_pred(text)
        ok = int(equal_strict(pred, rec["gold"]))
        greedy_records.append({
            "id": i, "gold": rec["gold"], "pred": pred, "correct": ok,
            "tok_ids": tok_ids, "boundaries": boundaries,
            "worst_lp_step": worst_lp_step,
            "lp_min_score": lp_min_score, "entropy_mean": entropy_mean,
            "n_arith": n_arith,
        })
    n_greedy = sum(r["correct"] for r in greedy_records)
    vanilla = n_greedy / len(greedy_records)
    print(f"  vanilla: {n_greedy}/{len(greedy_records)} = {vanilla:.3f}", flush=True)

    summary = {"model": MODEL, "tag": TAG, "dataset": DATASET, "n": len(greedy_records),
                "vanilla_acc": vanilla, "results": {}}

    # Pre-compute thresholds for each gate score × gate_frac
    lp_mins = sorted([r["lp_min_score"] for r in greedy_records if r["lp_min_score"] != float("inf")])
    entropy_means = sorted([r["entropy_mean"] for r in greedy_records if not (r["entropy_mean"] != r["entropy_mean"])])
    n_total = len(greedy_records)

    def get_flagged(score_name, gate_frac):
        if score_name == "ent":
            t = float(np.quantile(entropy_means, 1 - gate_frac))
            return [r["id"] for r in greedy_records
                    if not (r["entropy_mean"] != r["entropy_mean"]) and r["entropy_mean"] > t]
        elif score_name == "lp":
            t = float(np.quantile(lp_mins, gate_frac))
            return [r["id"] for r in greedy_records if r["lp_min_score"] < t]
        elif score_name == "combined":
            t_ent = float(np.quantile(entropy_means, 1 - gate_frac))
            t_lp = float(np.quantile(lp_mins, gate_frac))
            return [r["id"] for r in greedy_records
                    if (not (r["entropy_mean"] != r["entropy_mean"]) and r["entropy_mean"] > t_ent)
                    or r["lp_min_score"] < t_lp
                    or r["n_arith"] > 0]
        return []

    def run_combo(name, K, T, flagged_ids):
        prompts = []; meta_ids = []
        for r in greedy_records:
            if r["id"] in flagged_ids and r["worst_lp_step"] < len(r["boundaries"]):
                bdy_pos = r["boundaries"][r["worst_lp_step"]]
                prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
                prompts.append(base_prompts[r["id"]] + prefix)
                meta_ids.append(r["id"])
        if not prompts:
            summary["results"][name] = {"acc": vanilla, "delta": 0, "n": n_total,
                                          "flagged_n": 0, "avg_compute": 1.0}
            return
        sp_b = SamplingParams(n=K, temperature=T, top_p=0.95, max_tokens=1024, seed=SEED + 100)
        t0 = time.time()
        outs_b = llm.generate(prompts, sp_b)
        flagged_results = {}
        for ii, ob in zip(meta_ids, outs_b):
            r = greedy_records[ii]
            alts = [extract_pred(c.text) for c in ob.outputs]
            chosen = majority_vote([r["pred"]] + alts)
            flagged_results[ii] = chosen
        n_correct = 0
        for r in greedy_records:
            pred = flagged_results.get(r["id"], r["pred"])
            n_correct += int(equal_strict(pred, r["gold"]))
        acc = n_correct / n_total
        avg_cost = 1 + K * (len(flagged_ids) / n_total)
        summary["results"][name] = {
            "acc": acc, "delta": acc - vanilla, "n": n_total,
            "flagged_n": len(flagged_ids), "avg_compute": avg_cost
        }

    # Strategy combinations
    for gate_score in ["ent", "lp", "combined"]:
        for gate_frac in [0.25, 0.50, 0.75]:
            for K in [2, 4, 8]:
                flagged = get_flagged(gate_score, gate_frac)
                name = f"gate_{gate_score}_f{gate_frac}_K{K}"
                run_combo(name, K, 0.7, flagged)

    # Always_K baselines
    all_ids = [r["id"] for r in greedy_records]
    for K in [2, 4, 8]:
        run_combo(f"always_K{K}", K, 0.7, all_ids)

    # Save
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done. Wrote {OUT} ===", flush=True)
    print(f"\n=== Summary {TAG} {DATASET} (vanilla={vanilla:.3f}) ===")
    sorted_methods = sorted(summary["results"].items(), key=lambda x: -x[1]["acc"])
    for k, v in sorted_methods[:10]:
        print(f"  {k:25s} acc={v['acc']:.3f}  Δ={v['delta']:+.3f}  cost={v['avg_compute']:.2f}×")


if __name__ == "__main__":
    main()

"""
EXP-STEP-REJECTION T5 COMBINED: 2-stage pipeline combining CP + step rejection.

Strategy: at test time, compute trajectory CP score (e.g., entropy_mean).
- If score above threshold → accept greedy (high confidence).
- If score below threshold → apply K=4 step rejection to try to recover.

Hypothesis: spend compute (K=4) only on "uncertain" traces where it helps most.

Variants:
1. **t5_entropy_gate_K4**: gate on entropy_mean trajectory; K=4 maj on flagged.
2. **t5_lp_min_gate_K4**: gate on lp_min; K=4 maj on flagged.
3. **t5_prov_match_gate_K4**: gate on prov_match_latest_rate (running, last step value).
4. **t5_combined_gate**: union of any score < its respective threshold → K=4.

Compute cost: for traces NOT flagged, just greedy (1×). For flagged, K=4 (5×). Avg ~1+ε for small flagged-frac.

Env vars: MODEL, TAG, DATASET, NQ, GATE (default 0.5 = flag bottom 50% by score)
"""

import json
import math
import os
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
GATE_FRAC = float(os.environ.get("GATE", "0.5"))
SEED = 0
TOPK = 20

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / f"SX_step_rej_t5_{TAG}_{DATASET}.json"

PROMPT_MATH = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\nProblem: {question}\n\n"
)


def _lp_of(e):
    if hasattr(e, "logprob"):
        return e.logprob
    if isinstance(e, (tuple, list)) and len(e) > 0:
        return float(e[0])
    if isinstance(e, (int, float)):
        return float(e)
    return float("nan")


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
    if not votes:
        return None
    counter = Counter(votes)
    return counter.most_common(1)[0][0]


def main():
    print(f"=== T5 Combined CP+Rejection: {MODEL} on {DATASET} (gate={GATE_FRAC}) ===", flush=True)
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
        # Trajectory scores
        valid_lps = [v for v in step_lps if not (v != v)]
        lp_min_score = float(np.min(valid_lps)) if valid_lps else float("inf")
        # entropy_mean
        ents = []
        for d in distribs:
            if d is None: continue
            p = np.array(d["probs"]); p = p[p > 0]
            if len(p) > 0:
                ents.append(-float((p * np.log(p)).sum()))
        entropy_mean = float(np.mean(ents)) if ents else float("nan")
        # Worst step under lp_min
        valid_lp_pairs = [(j, v) for j, v in enumerate(step_lps) if not (v != v)]
        worst_lp_step = min(valid_lp_pairs, key=lambda kv: kv[1])[0] if valid_lp_pairs else 0
        pred = extract_pred(text)
        ok = int(equal_strict(pred, rec["gold"]))
        greedy_records.append({
            "id": i, "gold": rec["gold"], "pred": pred, "correct": ok,
            "tok_ids": tok_ids, "boundaries": boundaries, "step_lps": step_lps,
            "worst_lp_step": worst_lp_step,
            "lp_min_score": lp_min_score, "entropy_mean": entropy_mean,
            "text": text,
        })
    n_greedy = sum(r["correct"] for r in greedy_records)
    vanilla = n_greedy / len(greedy_records)
    print(f"  vanilla: {n_greedy}/{len(greedy_records)} = {vanilla:.3f}", flush=True)

    # Compute thresholds for each gate score (use median split)
    lp_mins = [r["lp_min_score"] for r in greedy_records if r["lp_min_score"] != float("inf")]
    entropy_means = [r["entropy_mean"] for r in greedy_records if not (r["entropy_mean"] != r["entropy_mean"])]
    lp_threshold = float(np.quantile(lp_mins, GATE_FRAC)) if lp_mins else float("-inf")
    ent_threshold = float(np.quantile(entropy_means, 1 - GATE_FRAC)) if entropy_means else float("inf")

    # Determine which traces are flagged (low confidence)
    flagged_lp = [r["id"] for r in greedy_records if r["lp_min_score"] < lp_threshold]
    flagged_ent = [r["id"] for r in greedy_records
                   if not (r["entropy_mean"] != r["entropy_mean"]) and r["entropy_mean"] > ent_threshold]
    flagged_either = sorted(set(flagged_lp) | set(flagged_ent))

    print(f"  flagged (lp_min): {len(flagged_lp)}/{len(greedy_records)} ({len(flagged_lp)/len(greedy_records):.2f})", flush=True)
    print(f"  flagged (ent_mean): {len(flagged_ent)}/{len(greedy_records)} ({len(flagged_ent)/len(greedy_records):.2f})", flush=True)
    print(f"  flagged (either): {len(flagged_either)}/{len(greedy_records)} ({len(flagged_either)/len(greedy_records):.2f})", flush=True)

    summary = {"model": MODEL, "tag": TAG, "dataset": DATASET, "n": len(greedy_records),
                "vanilla_acc": vanilla, "gate_frac": GATE_FRAC,
                "lp_threshold": lp_threshold, "ent_threshold": ent_threshold,
                "results": {}}

    # For each flagging strategy, run K=4 majority on flagged traces only
    for flag_name, flagged_ids in [("lp_min", flagged_lp), ("ent_mean", flagged_ent), ("either", flagged_either)]:
        if not flagged_ids:
            continue
        prompts = []; meta_ids = []
        for r in greedy_records:
            if r["id"] in flagged_ids and r["worst_lp_step"] < len(r["boundaries"]):
                bdy_pos = r["boundaries"][r["worst_lp_step"]]
                prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
                prompts.append(base_prompts[r["id"]] + prefix)
                meta_ids.append(r["id"])
        if not prompts: continue
        sp_b = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 100)
        t0 = time.time()
        outs_b = llm.generate(prompts, sp_b)
        # Map results
        flagged_results = {}
        for ii, ob in zip(meta_ids, outs_b):
            r = greedy_records[ii]
            alts = [extract_pred(c.text) for c in ob.outputs]
            chosen = majority_vote([r["pred"]] + alts)
            flagged_results[ii] = chosen
        # Compute final acc: flagged use majority, others use greedy
        n_correct = 0
        for r in greedy_records:
            if r["id"] in flagged_results:
                pred = flagged_results[r["id"]]
            else:
                pred = r["pred"]
            n_correct += int(equal_strict(pred, r["gold"]))
        acc = n_correct / len(greedy_records)
        # avg compute: 1 (greedy) + 4 (alts) for flagged, 1 for unflagged
        avg_cost = 1 + 4 * (len(flagged_ids) / len(greedy_records))
        summary["results"][f"gate_{flag_name}_K4"] = {
            "acc": acc, "delta": acc - vanilla, "n": len(greedy_records),
            "flagged_n": len(flagged_ids), "avg_compute": avg_cost
        }
        print(f"  gate_{flag_name}_K4 acc={acc:.3f}  Δ={acc-vanilla:+.3f}  flagged={len(flagged_ids)}  avg_cost={avg_cost:.2f}× ({time.time()-t0:.1f}s)", flush=True)

    # Compare to "always K=4 majority"
    print(f"\n[Comparison: always K=4 majority]", flush=True)
    prompts_all = []; ids_all = []
    for r in greedy_records:
        if r["worst_lp_step"] < len(r["boundaries"]):
            bdy_pos = r["boundaries"][r["worst_lp_step"]]
            prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
            prompts_all.append(base_prompts[r["id"]] + prefix)
            ids_all.append(r["id"])
    sp_all = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 200)
    t0 = time.time()
    outs_all = llm.generate(prompts_all, sp_all)
    n_all = 0
    for ii, oa in zip(ids_all, outs_all):
        r = greedy_records[ii]
        alts = [extract_pred(c.text) for c in oa.outputs]
        chosen = majority_vote([r["pred"]] + alts)
        n_all += int(equal_strict(chosen, r["gold"]))
    acc_all = n_all / len(ids_all)
    summary["results"]["always_K4_baseline"] = {"acc": acc_all, "delta": acc_all - vanilla,
                                                  "n": len(ids_all), "avg_compute": 5.0}
    print(f"  always_K4 acc={acc_all:.3f}  Δ={acc_all-vanilla:+.3f}  avg_cost=5.0× ({time.time()-t0:.1f}s)", flush=True)

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done. Wrote {OUT} ===", flush=True)
    print(f"\n=== Summary {TAG} {DATASET} (vanilla={vanilla:.3f}) ===")
    for method, res in summary["results"].items():
        print(f"  {method:25s} acc={res['acc']:.3f}  Δ={res['delta']:+.3f}  cost={res.get('avg_compute', 1):.2f}×")


if __name__ == "__main__":
    main()

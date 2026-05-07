"""
EXP-STEP-REJECTION T3 ONLINE: online hard per-step rejection (multi-step rejection in single trace).

After Tier 1 (K4_majority wins), Tier 2 (K4 variants), Tier 3 explores ONLINE rejection at MULTIPLE bad steps:

1. **online_K1_threshold**: at EVERY step boundary, if lp < threshold, regen once T=0.7. Continue.
2. **online_BoN_lp_select**: at every bad step, regen K=2, pick higher lp. Continue.
3. **multi_worst_K2_robust**: TOP-3 worst steps, K=2 each, vote across all alts + greedy.
4. **prov_anchor_K2**: K=2 alts with provisional-answer hint at worst step.
5. **K4_all_pred_vote** (Tier 1 baseline reproduced): K=4 + greedy majority for comparison.

Threshold for online methods: derived from per-trace lp_mean distribution. Use median - 0.5*std.

Env vars: MODEL, TAG, DATASET, NQ
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
SEED = 0
TOPK = 20

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / f"SX_step_rej_t3_{TAG}_{DATASET}.json"

PROMPT_MATH = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\nProblem: {question}\n\n"
)

PROV_ANSWER_PROMPT = "\n\nBased on the reasoning so far, the most likely answer is: "


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
    print(f"=== T3 Online step rejection: {MODEL} on {DATASET} ===", flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2560,
        tensor_parallel_size=1,
        seed=SEED,
        trust_remote_code=True,
    )
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

    # Phase 1: greedy with logprobs=20 (collects step lps + per-step boundaries)
    print(f"\n[Phase 1] greedy decode...", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1536, logprobs=TOPK, seed=SEED)
    t0 = time.time()
    outs = llm.generate(base_prompts, sp)
    print(f"  greedy done in {time.time()-t0:.1f}s", flush=True)

    greedy_records = []
    all_lp_means = []
    for i, (rec, out) in enumerate(zip(ds, outs)):
        o = out.outputs[0]
        text = o.text
        tok_ids = list(o.token_ids)
        tok_strs = [tok.decode([tid]) for tid in tok_ids]
        chosen_lps = []
        for step_lp in o.logprobs or []:
            if step_lp is None:
                chosen_lps.append(float("nan")); continue
            entries = sorted(step_lp.items(), key=lambda kv: -kv[1].logprob)
            chosen_lps.append(entries[0][1].logprob)
        boundaries = find_step_boundaries(tok_strs)
        n = len(tok_ids)
        step_lps = step_lp_means(chosen_lps, boundaries, n)
        valid_lps = [(j, v) for j, v in enumerate(step_lps) if not (v != v)]
        sorted_lps = sorted(valid_lps, key=lambda kv: kv[1])
        top3_worst = [s for s, _ in sorted_lps[:3]]
        for _, v in valid_lps:
            all_lp_means.append(v)
        pred = extract_pred(text)
        ok = int(equal_strict(pred, rec["gold"]))
        greedy_records.append({
            "id": i, "gold": rec["gold"], "pred": pred, "correct": ok,
            "tok_ids": tok_ids, "boundaries": boundaries, "step_lps": step_lps,
            "top3_worst": top3_worst,
            "text": text,
        })
    n_greedy = sum(r["correct"] for r in greedy_records)
    vanilla = n_greedy / len(greedy_records)
    print(f"  vanilla: {n_greedy}/{len(greedy_records)} = {vanilla:.3f}", flush=True)

    # Threshold for "bad step": 25th percentile of all step-lp_means
    threshold = float(np.quantile(all_lp_means, 0.25)) if all_lp_means else float("-inf")
    print(f"  bad-step threshold (25th percentile lp): {threshold:.3f}", flush=True)

    # Identify ALL bad steps per record
    for r in greedy_records:
        r["bad_steps"] = [j for j, v in enumerate(r["step_lps"])
                          if not (v != v) and v < threshold]
        # Cap to top-3 worst to bound cost
        sorted_bad = sorted([(j, r["step_lps"][j]) for j in r["bad_steps"]], key=lambda kv: kv[1])
        r["bad_steps_capped"] = [j for j, _ in sorted_bad[:3]]

    summary = {"model": MODEL, "tag": TAG, "dataset": DATASET, "n": len(greedy_records),
                "vanilla_acc": vanilla, "threshold": threshold,
                "results": {}}

    # === Strategy 1: online_K1 — regen at EVERY bad step with T=0.7, single alt
    # Approximate by: at each capped bad step, regen K=1 from the prefix; collect alt preds
    print(f"\n[Strategy T3-1] online_K1 (regen K=1 at top-3 worst steps)...", flush=True)
    online_prompts = []; online_meta = []
    for r in greedy_records:
        for step_idx in r["bad_steps_capped"]:
            if step_idx < len(r["boundaries"]):
                bdy_pos = r["boundaries"][step_idx]
                prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
                online_prompts.append(base_prompts[r["id"]] + prefix)
                online_meta.append((r["id"], step_idx))
    sp_o1 = SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 100)
    t0 = time.time()
    outs_o1 = llm.generate(online_prompts, sp_o1)
    online_results_K1 = {}
    for (qid, step_idx), oo in zip(online_meta, outs_o1):
        for c in oo.outputs:
            pred = extract_pred(c.text)
            if qid not in online_results_K1: online_results_K1[qid] = []
            online_results_K1[qid].append(pred)
    n_o1 = 0
    for r in greedy_records:
        alts = online_results_K1.get(r["id"], [])
        chosen = majority_vote([r["pred"]] + alts)
        n_o1 += int(equal_strict(chosen, r["gold"]))
    acc_o1 = n_o1 / len(greedy_records)
    summary["results"]["online_K1_top3_worst"] = {"acc": acc_o1, "delta": acc_o1 - vanilla,
                                                    "n": len(greedy_records),
                                                    "avg_alts": len(online_meta) / max(len(greedy_records), 1)}
    print(f"  online_K1_top3_worst acc={acc_o1:.3f}  Δ={acc_o1-vanilla:+.3f}  avg_alts={len(online_meta)/len(greedy_records):.1f} ({time.time()-t0:.1f}s)", flush=True)

    # === Strategy 2: online_K2 with lp-select (regen K=2 at every bad step, pick higher lp)
    print(f"\n[Strategy T3-2] online_K2_BoN (regen K=2 at top-3 worst, pick by lp)...", flush=True)
    sp_o2 = SamplingParams(n=2, temperature=0.7, top_p=0.95, max_tokens=1024, logprobs=1, seed=SEED + 200)
    t0 = time.time()
    outs_o2 = llm.generate(online_prompts, sp_o2)
    online_results_K2 = {}
    for (qid, step_idx), oo in zip(online_meta, outs_o2):
        # Pick higher-lp alt
        best_pred = None; best_lp = float("-inf")
        for c in oo.outputs:
            lps = []
            for step_lp in c.logprobs or []:
                if step_lp is None or not step_lp: continue
                e = next(iter(step_lp.values()))
                if hasattr(e, "logprob"):
                    lps.append(e.logprob)
            mean_lp = float(np.mean([lp for lp in lps if not (lp != lp)])) if lps else float("-inf")
            if mean_lp > best_lp:
                best_lp = mean_lp; best_pred = extract_pred(c.text)
        if qid not in online_results_K2: online_results_K2[qid] = []
        online_results_K2[qid].append(best_pred)
    n_o2 = 0
    for r in greedy_records:
        alts = online_results_K2.get(r["id"], [])
        chosen = majority_vote([r["pred"]] + alts)
        n_o2 += int(equal_strict(chosen, r["gold"]))
    acc_o2 = n_o2 / len(greedy_records)
    summary["results"]["online_K2_BoN_top3_worst"] = {"acc": acc_o2, "delta": acc_o2 - vanilla,
                                                       "n": len(greedy_records)}
    print(f"  online_K2_BoN acc={acc_o2:.3f}  Δ={acc_o2-vanilla:+.3f} ({time.time()-t0:.1f}s)", flush=True)

    # === Strategy 3: K4_majority (Tier 1 baseline reproduced for comparison)
    print(f"\n[Strategy T3-3] K4_majority (Tier 1 baseline) at single worst step...", flush=True)
    k4_prompts = []; k4_ids = []
    for r in greedy_records:
        if r["top3_worst"]:
            ws = r["top3_worst"][0]
            if ws < len(r["boundaries"]):
                bdy_pos = r["boundaries"][ws]
                prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
                k4_prompts.append(base_prompts[r["id"]] + prefix)
                k4_ids.append(r["id"])
    sp_k4 = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 400)
    t0 = time.time()
    outs_k4 = llm.generate(k4_prompts, sp_k4)
    n_k4 = 0
    for ii, ok4 in zip(k4_ids, outs_k4):
        r = greedy_records[ii]
        alts = [extract_pred(c.text) for c in ok4.outputs]
        chosen = majority_vote([r["pred"]] + alts)
        n_k4 += int(equal_strict(chosen, r["gold"]))
    acc_k4 = n_k4 / len(k4_ids) if k4_ids else 0
    summary["results"]["K4_majority_baseline"] = {"acc": acc_k4, "delta": acc_k4 - vanilla, "n": len(k4_ids)}
    print(f"  K4_majority_baseline acc={acc_k4:.3f}  Δ={acc_k4-vanilla:+.3f} ({time.time()-t0:.1f}s)", flush=True)

    # === Strategy 4: prov_anchor_K2 (K=2 with provisional answer hint)
    # First, get provisional answer at each worst step
    print(f"\n[Strategy T3-4] prov_anchor_K2 (provisional answer hint)...", flush=True)
    prov_prompts = []; prov_meta = []
    for r in greedy_records:
        if r["top3_worst"]:
            ws = r["top3_worst"][0]
            if ws < len(r["boundaries"]):
                bdy_pos = r["boundaries"][ws]
                prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
                prov_prompts.append(base_prompts[r["id"]] + prefix + PROV_ANSWER_PROMPT)
                prov_meta.append(r["id"])
    sp_prov = SamplingParams(temperature=0.0, max_tokens=16, seed=SEED + 500)
    t0 = time.time()
    outs_prov = llm.generate(prov_prompts, sp_prov)
    prov_answers = {}
    for qid, op in zip(prov_meta, outs_prov):
        prov_pred = extract_pred(op.outputs[0].text)
        prov_answers[qid] = prov_pred
    print(f"  Got {len(prov_answers)} provisional answers ({time.time()-t0:.1f}s)", flush=True)
    # Now regen K=2 with hint
    hint_prompts = []; hint_ids = []
    for r in greedy_records:
        if r["id"] in prov_answers and r["top3_worst"]:
            prov = prov_answers[r["id"]]
            ws = r["top3_worst"][0]
            if ws < len(r["boundaries"]):
                bdy_pos = r["boundaries"][ws]
                prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
                hint = ""
                if prov is not None and len(str(prov)) < 50:
                    hint = f"\n\nHypothesis: the answer is approximately {prov}. Continue reasoning to verify:\n\n"
                hint_prompts.append(base_prompts[r["id"]] + prefix + hint)
                hint_ids.append(r["id"])
    sp_hint = SamplingParams(n=2, temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 600)
    if hint_prompts:
        t0 = time.time()
        outs_hint = llm.generate(hint_prompts, sp_hint)
        n_hint = 0
        for ii, oh in zip(hint_ids, outs_hint):
            r = greedy_records[ii]
            alts = [extract_pred(c.text) for c in oh.outputs]
            chosen = majority_vote([r["pred"]] + alts)
            n_hint += int(equal_strict(chosen, r["gold"]))
        acc_hint = n_hint / len(hint_ids)
        summary["results"]["prov_anchor_K2"] = {"acc": acc_hint, "delta": acc_hint - vanilla, "n": len(hint_ids)}
        print(f"  prov_anchor_K2 acc={acc_hint:.3f}  Δ={acc_hint-vanilla:+.3f} ({time.time()-t0:.1f}s)", flush=True)

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done. Wrote {OUT} ===", flush=True)
    print(f"\n=== Summary {TAG} {DATASET} (vanilla={vanilla:.3f}) ===")
    for method, res in summary["results"].items():
        print(f"  {method:35s} acc={res['acc']:.3f}  Δ={res['delta']:+.3f}")


if __name__ == "__main__":
    main()

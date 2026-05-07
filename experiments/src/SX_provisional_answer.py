"""
EXP-PROVISIONAL: at each step boundary, ask model "what's your best answer
based on reasoning so far?" and track:
- provisional_answer_t (what the model would commit to at step t)
- flip_count, first_match_step, last_N_stable, final_match_rate

Also computes trajectory-diff scores (FREE) at each prefix.

Saves all per-step features to JSONL for downstream per-step CP analysis.

Env vars:
  MODEL, TAG, DATASET (single, default math500), NQ
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, equal_strict, normalize

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
TAG = os.environ.get("TAG", "qwen25_7b")
DATASET = os.environ.get("DATASET", "math500")
N_PER_DS = int(os.environ.get("NQ", "100"))
TOPK = 20
SEED = 0
PROV_ANSWER_TOKS = 16
MAX_STEPS = 30  # cap per-step queries

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / f"SX_prov_{TAG}_{DATASET}.json"
OUT_TRACES = OUTDIR / f"SX_prov_{TAG}_{DATASET}_per_step.jsonl"

PROMPT_MATH = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\nProblem: {question}\n\n"
)

PROV_ANSWER_PROMPT = (
    "\n\nBased on the reasoning so far, the most likely answer is: "
)


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


def topk_dist(topk_dict):
    if not topk_dict: return None
    entries = sorted(topk_dict.values(), key=lambda v: -v.logprob)
    lps = np.array([e.logprob for e in entries], dtype=float)
    probs = np.exp(lps - lps.max()); probs = probs / probs.sum()
    return {"probs": probs.tolist(), "lps": lps.tolist()}


def step_basic(distribs, token_lps_seg):
    if not distribs:
        return {"lp": float("nan"), "ent": float("nan"), "marg": float("nan")}
    ents, margs = [], []
    for d in distribs:
        if d is None: continue
        p = np.array(d["probs"]); p = p[p > 0]
        if len(p) > 0:
            ents.append(-float((p * np.log(p)).sum()))
        if len(d["lps"]) >= 2:
            margs.append(d["lps"][0] - d["lps"][1])
    valid_lp = [lp for lp in token_lps_seg if not (lp != lp)]
    return {
        "lp": float(np.mean(valid_lp)) if valid_lp else float("nan"),
        "ent": float(np.mean(ents)) if ents else float("nan"),
        "marg": float(np.mean(margs)) if margs else float("nan"),
    }


def main():
    print(f"=== Provisional answer: {MODEL} on {DATASET} ===", flush=True)
    print(f"  N={N_PER_DS}, max_steps_query={MAX_STEPS}", flush=True)

    print(f"Loading {MODEL}...", flush=True)
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

    # Phase 1: greedy with logprobs=20
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
            entries = sorted(step_lp.items(), key=lambda kv: -kv[1].logprob)
            chosen_lps.append(entries[0][1].logprob)
            distribs.append(topk_dist(step_lp))
        boundaries = find_step_boundaries(tok_strs)
        n = len(tok_ids)
        bdy = boundaries + [n]
        per_step_basic = []
        for sa, sb in zip(bdy[:-1], bdy[1:]):
            per_step_basic.append(step_basic(distribs[sa:sb], chosen_lps[sa:sb]))
        pred = extract_pred(text)
        ok = int(equal_strict(pred, rec["gold"]))
        greedy_records.append({
            "id": i, "question": rec["q"], "gold": rec["gold"], "text": text, "pred": pred, "correct": ok,
            "tok_ids": tok_ids, "boundaries": boundaries,
            "per_step_basic": per_step_basic,
            "n_steps": len(per_step_basic),
        })

    n_correct = sum(r["correct"] for r in greedy_records)
    print(f"  vanilla = {n_correct}/{len(greedy_records)} = {n_correct/len(greedy_records):.3f}", flush=True)

    # Phase 2: at each step boundary, query provisional answer
    # Build batch of prov-answer prompts: (q_id, step_idx, prompt)
    print(f"\n[Phase 2] provisional answer queries (batched)...", flush=True)
    batched_prompts = []
    batch_meta = []  # (q_id, step_idx)
    for r in greedy_records:
        if r["n_steps"] == 0: continue
        bdy_full = r["boundaries"] + [len(r["tok_ids"])]
        # Limit number of step queries
        n_query = min(r["n_steps"], MAX_STEPS)
        # Spread across steps: query at positions {1, 2, ..., n_query} OR stride if n_steps > MAX_STEPS
        if r["n_steps"] <= MAX_STEPS:
            indices = list(range(1, r["n_steps"] + 1))
        else:
            stride = r["n_steps"] / MAX_STEPS
            indices = [int(round(i * stride)) for i in range(1, MAX_STEPS + 1)]
            indices = sorted(set(min(idx, r["n_steps"]) for idx in indices))
        for step_t in indices:
            # prefix = tokens up to end of step step_t
            if step_t < len(bdy_full):
                end_pos = bdy_full[step_t]
            else:
                end_pos = len(r["tok_ids"])
            prefix_text = tok.decode(r["tok_ids"][:end_pos], skip_special_tokens=True)
            # Build prompt: original chat template + prefix + provisional cue
            prompt = base_prompts[r["id"]] + prefix_text + PROV_ANSWER_PROMPT
            batched_prompts.append(prompt)
            batch_meta.append((r["id"], step_t))

    print(f"  total prov queries: {len(batched_prompts)}", flush=True)
    sp_p = SamplingParams(temperature=0.0, max_tokens=PROV_ANSWER_TOKS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs_p = llm.generate(batched_prompts, sp_p)
    print(f"  prov queries done in {time.time()-t0:.1f}s", flush=True)

    # Parse provisional answers
    prov_per_q = {r["id"]: {} for r in greedy_records}
    for (qid, step_t), out_p in zip(batch_meta, outs_p):
        prov_text = out_p.outputs[0].text
        prov_pred = extract_pred(prov_text) or extract_pred(prov_text + " ")
        prov_per_q[qid][step_t] = {
            "step": step_t,
            "prov_text": prov_text[:100],
            "prov_pred": prov_pred,
        }

    # Phase 3: compute per-step features (trajectory-diff, prov-answer features)
    print(f"\n[Phase 3] computing per-step features...", flush=True)
    summary_rows = []
    with OUT_TRACES.open("w") as fw:
        for r in greedy_records:
            qid = r["id"]
            prov_steps = prov_per_q.get(qid, {})
            sorted_step_indices = sorted(prov_steps.keys())
            # Provisional answer features
            prov_preds = [prov_steps[s]["prov_pred"] for s in sorted_step_indices]
            final_pred = r["pred"]
            # Match with final pred
            match_with_final = [int(equal_strict(p, final_pred)) for p in prov_preds]
            # Match with gold
            match_with_gold = [int(equal_strict(p, r["gold"])) for p in prov_preds]
            # Flips between consecutive
            n_flips = 0
            for i in range(1, len(prov_preds)):
                if prov_preds[i-1] is None or prov_preds[i] is None:
                    continue
                if not equal_strict(prov_preds[i-1], prov_preds[i]):
                    n_flips += 1
            # First step where prov matches final
            first_match_step = next((i for i, m in enumerate(match_with_final) if m == 1), -1)
            # Last 3 steps stable?
            last_N = 3
            last_stable = 1
            if len(prov_preds) < last_N:
                last_stable = 0
            else:
                last_set = set(str(normalize(p)) for p in prov_preds[-last_N:] if p is not None)
                last_stable = int(len(last_set) == 1)
            final_match_rate = float(np.mean(match_with_final)) if match_with_final else 0.0
            gold_match_rate = float(np.mean(match_with_gold)) if match_with_gold else 0.0

            # Trajectory-diff scores (FREE) — running aggregates and their drifts
            sfeats = r["per_step_basic"]
            # Running mean/min/max
            running_lp = []
            running_ent = []
            running_marg = []
            lp_mins = []
            for j, s in enumerate(sfeats):
                if not (s["lp"] != s["lp"]):
                    running_lp.append(s["lp"])
                if not (s["ent"] != s["ent"]):
                    running_ent.append(s["ent"])
                if not (s["marg"] != s["marg"]):
                    running_marg.append(s["marg"])
            # Per-step prefix-mean drift = how much running mean changed at each step
            prefix_lp_means = []
            prefix_ent_means = []
            prefix_marg_means = []
            for j in range(len(sfeats)):
                lp_part = [s["lp"] for s in sfeats[:j+1] if not (s["lp"] != s["lp"])]
                ent_part = [s["ent"] for s in sfeats[:j+1] if not (s["ent"] != s["ent"])]
                marg_part = [s["marg"] for s in sfeats[:j+1] if not (s["marg"] != s["marg"])]
                prefix_lp_means.append(float(np.mean(lp_part)) if lp_part else float("nan"))
                prefix_ent_means.append(float(np.mean(ent_part)) if ent_part else float("nan"))
                prefix_marg_means.append(float(np.mean(marg_part)) if marg_part else float("nan"))

            # Drifts (changes between consecutive prefix means)
            lp_drifts = []
            ent_drifts = []
            marg_drifts = []
            for j in range(1, len(prefix_lp_means)):
                if not (prefix_lp_means[j] != prefix_lp_means[j]) and not (prefix_lp_means[j-1] != prefix_lp_means[j-1]):
                    lp_drifts.append(prefix_lp_means[j] - prefix_lp_means[j-1])
                if not (prefix_ent_means[j] != prefix_ent_means[j]) and not (prefix_ent_means[j-1] != prefix_ent_means[j-1]):
                    ent_drifts.append(prefix_ent_means[j] - prefix_ent_means[j-1])
                if not (prefix_marg_means[j] != prefix_marg_means[j]) and not (prefix_marg_means[j-1] != prefix_marg_means[j-1]):
                    marg_drifts.append(prefix_marg_means[j] - prefix_marg_means[j-1])

            # Aggregate drifts
            traj_diff_scores = {
                "prefix_lp_drift_max_abs": float(np.max(np.abs(lp_drifts))) if lp_drifts else float("nan"),
                "prefix_lp_drift_pos_sum": float(sum(d for d in lp_drifts if d > 0)) if lp_drifts else float("nan"),
                "prefix_lp_drift_neg_sum": float(sum(d for d in lp_drifts if d < 0)) if lp_drifts else float("nan"),
                "prefix_ent_drift_max_abs": float(np.max(np.abs(ent_drifts))) if ent_drifts else float("nan"),
                "prefix_ent_drift_pos_sum": float(sum(d for d in ent_drifts if d > 0)) if ent_drifts else float("nan"),
                "prefix_marg_drift_max_abs": float(np.max(np.abs(marg_drifts))) if marg_drifts else float("nan"),
                # Monotonicity: fraction of consecutive lp pairs where lp goes up
                "lp_monotone_up_frac": float(np.mean([d > 0 for d in lp_drifts])) if lp_drifts else float("nan"),
                "lp_pref_var": float(np.var(prefix_lp_means)) if len(prefix_lp_means) >= 2 else float("nan"),
            }

            row = {
                "id": qid, "gold": r["gold"], "pred": r["pred"], "correct": r["correct"],
                "n_steps": r["n_steps"], "n_prov_queries": len(prov_preds),
                # Prov features
                "prov_n_flips": n_flips,
                "prov_final_match_rate": final_match_rate,
                "prov_gold_match_rate": gold_match_rate,
                "prov_first_match_step": first_match_step,
                "prov_last_3_stable": last_stable,
                **traj_diff_scores,
                "per_step_prov": [
                    {"step": s, "prov_pred": prov_steps[s]["prov_pred"],
                     "match_final": int(equal_strict(prov_steps[s]["prov_pred"], final_pred)),
                     "match_gold": int(equal_strict(prov_steps[s]["prov_pred"], r["gold"]))}
                    for s in sorted_step_indices
                ],
                "per_step_basic": sfeats,
                "prefix_lp_means": prefix_lp_means,
                "prefix_ent_means": prefix_ent_means,
                "prefix_marg_means": prefix_marg_means,
            }
            summary_rows.append(row)
            fw.write(json.dumps(row) + "\n")

    # Phase 4: Spearman + bootstrap CP
    print(f"\n[Phase 4] Spearman correlations with greedy_correct...", flush=True)
    from scipy.stats import spearmanr
    correct = np.array([r["correct"] for r in summary_rows])
    score_keys = [
        "prov_n_flips", "prov_final_match_rate", "prov_gold_match_rate",
        "prov_first_match_step", "prov_last_3_stable",
        "prefix_lp_drift_max_abs", "prefix_lp_drift_pos_sum", "prefix_lp_drift_neg_sum",
        "prefix_ent_drift_max_abs", "prefix_ent_drift_pos_sum",
        "prefix_marg_drift_max_abs", "lp_monotone_up_frac", "lp_pref_var",
    ]
    summary = {"model": MODEL, "tag": TAG, "dataset": DATASET, "n": len(summary_rows),
                "vanilla_acc": float(correct.mean())}
    summary["scores"] = {}

    def cp_eval_with_ci(scores, correct, alpha, n_boot=300, n_seeds=5):
        s = np.array(scores, dtype=float); c = np.array(correct, dtype=int)
        valid = ~np.isnan(s)
        s = s[valid]; c = c[valid]
        if len(s) < 20: return None
        rng = np.random.default_rng(0)
        boot_acc = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(s), size=len(s))
            sb = s[idx]; cb = c[idx]
            accs = []
            for _ in range(n_seeds):
                perm = rng.permutation(len(sb))
                nc = len(sb) // 2
                ci, ti = perm[:nc], perm[nc:]
                cal_corr = sb[ci][cb[ci] == 1]
                if len(cal_corr) < 5: continue
                n_c = len(cal_corr)
                ql = max(0.0, min(1.0, math.floor(alpha * (n_c + 1)) / n_c))
                q = float(np.quantile(cal_corr, ql))
                kept = sb[ti] >= q
                if kept.sum():
                    accs.append(float(cb[ti][kept].mean()))
            if accs:
                boot_acc.append(np.mean(accs))
        if not boot_acc: return None
        return {"kept_acc": float(np.mean(boot_acc)),
                "ci95": [float(np.quantile(boot_acc, 0.025)), float(np.quantile(boot_acc, 0.975))]}

    for sk in score_keys:
        scores = np.array([r.get(sk, float("nan")) for r in summary_rows], dtype=float)
        valid = ~np.isnan(scores)
        if valid.sum() < 20: continue
        rho, p = spearmanr(scores[valid], correct[valid])
        scores_dir = scores * (1 if rho >= 0 else -1)
        cp03 = cp_eval_with_ci(scores_dir, correct, 0.30)
        if cp03 is None: continue
        summary["scores"][sk] = {"rho": float(rho), "p": float(p), "kept_acc_alpha_0.30": cp03}
        print(f"  {sk:30s} ρ={rho:+.3f} p={p:.3g} kept@0.3={cp03['kept_acc']:.3f} CI=[{cp03['ci95'][0]:.3f},{cp03['ci95'][1]:.3f}]", flush=True)

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done. Wrote {OUT} (and {OUT_TRACES}) ===", flush=True)


if __name__ == "__main__":
    main()

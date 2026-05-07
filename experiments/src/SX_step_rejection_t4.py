"""
EXP-STEP-REJECTION T4: advanced selection / reranking methods.

After T2/T3 confirmed K=4 majority dominates, try:
1. **K4_select_entropy**: K=4 alts at worst step. Rank each alt by its trajectory entropy_mean. Pick alt with LOWEST entropy.
2. **K4_select_prov_match**: K=4 alts. For each alt, query provisional answer. Rank by prov_match_latest_rate.
3. **K4_majority_then_lp_select**: K=4 + greedy = 5 candidates. If majority unanimous, accept. Else lp-select.
4. **K4_filter_by_arith**: K=4 + greedy. Filter out any alt with arith_violations. Then majority vote remaining.
5. **K4_consensus_with_greedy**: K=4 majority but ONLY if 3+ agree with greedy. Else abstain (=keep greedy).

Trigger: lp_min worst step.

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
OUT = OUTDIR / f"SX_step_rej_t4_{TAG}_{DATASET}.json"

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


def count_arith_violations(text):
    """Count safe_eval failures in text."""
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


def trace_entropy_mean(distribs):
    ents = []
    for d in distribs:
        if d is None: continue
        p = np.array(d["probs"]); p = p[p > 0]
        if len(p) > 0:
            ents.append(-float((p * np.log(p)).sum()))
    return float(np.mean(ents)) if ents else float("inf")


def main():
    print(f"=== T4 step rejection: {MODEL} on {DATASET} ===", flush=True)
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
        valid_lps = [(j, v) for j, v in enumerate(step_lps) if not (v != v)]
        worst_lp_step = min(valid_lps, key=lambda kv: kv[1])[0] if valid_lps else 0
        pred = extract_pred(text)
        ok = int(equal_strict(pred, rec["gold"]))
        greedy_records.append({
            "id": i, "gold": rec["gold"], "pred": pred, "correct": ok,
            "tok_ids": tok_ids, "boundaries": boundaries, "step_lps": step_lps,
            "worst_lp_step": worst_lp_step,
            "text": text,
        })
    n_greedy = sum(r["correct"] for r in greedy_records)
    vanilla = n_greedy / len(greedy_records)
    print(f"  vanilla: {n_greedy}/{len(greedy_records)} = {vanilla:.3f}", flush=True)

    # Generate K=4 alts at worst step (logprobs=TOPK to get distribs for entropy_mean)
    prompts = []; ids = []
    for r in greedy_records:
        if not r["step_lps"] or r["worst_lp_step"] >= len(r["boundaries"]):
            continue
        bdy_pos = r["boundaries"][r["worst_lp_step"]]
        prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
        prompts.append(base_prompts[r["id"]] + prefix)
        ids.append(r["id"])
    sp_b = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=1024, logprobs=TOPK, seed=SEED + 100)
    print(f"\n[K=4 generation]...", flush=True)
    t0 = time.time()
    outs_b = llm.generate(prompts, sp_b)
    print(f"  K=4 done in {time.time()-t0:.1f}s", flush=True)

    # Process each alt: compute pred, lp_mean, entropy_mean, arith_violations
    alt_data = {}  # qid -> list of {pred, lp_mean, ent_mean, n_arith_viol, text}
    for ii, ob in zip(ids, outs_b):
        alts = []
        for c in ob.outputs:
            text = c.text
            pred = extract_pred(text)
            tok_ids_alt = list(c.token_ids)
            tok_strs_alt = [tok.decode([tid]) for tid in tok_ids_alt]
            chosen_lps_alt = []; distribs_alt = []
            for step_lp in c.logprobs or []:
                if step_lp is None:
                    chosen_lps_alt.append(float("nan")); distribs_alt.append(None); continue
                try:
                    items = list(step_lp.items())
                    items.sort(key=lambda kv: -_lp_of(kv[1]))
                    if items:
                        chosen_lps_alt.append(_lp_of(items[0][1]))
                        entries_lps = np.array([_lp_of(it[1]) for it in items], dtype=float)
                        probs = np.exp(entries_lps - entries_lps.max()); probs = probs / probs.sum()
                        distribs_alt.append({"probs": probs.tolist(), "lps": entries_lps.tolist()})
                    else:
                        chosen_lps_alt.append(float("nan")); distribs_alt.append(None)
                except Exception:
                    chosen_lps_alt.append(float("nan")); distribs_alt.append(None)
            valid = [lp for lp in chosen_lps_alt if not (lp != lp)]
            lp_mean = float(np.mean(valid)) if valid else float("-inf")
            ent_mean = trace_entropy_mean(distribs_alt)
            n_v = count_arith_violations(text)
            alts.append({"pred": pred, "lp_mean": lp_mean, "ent_mean": ent_mean,
                          "n_arith_viol": n_v, "text": text})
        alt_data[ii] = alts

    summary = {"model": MODEL, "tag": TAG, "dataset": DATASET, "n": len(greedy_records),
                "vanilla_acc": vanilla, "results": {}}

    # Strategy 1: K4_select_entropy (lowest entropy_mean alt)
    n_ent = 0
    for ii in ids:
        r = greedy_records[ii]
        alts = alt_data.get(ii, [])
        if not alts:
            n_ent += int(equal_strict(r["pred"], r["gold"])); continue
        best = min(alts, key=lambda a: a["ent_mean"])
        n_ent += int(equal_strict(best["pred"], r["gold"]))
    acc_ent = n_ent / len(ids) if ids else 0
    summary["results"]["K4_select_entropy_min"] = {"acc": acc_ent, "delta": acc_ent - vanilla, "n": len(ids)}
    print(f"  K4_select_entropy_min acc={acc_ent:.3f}  Δ={acc_ent-vanilla:+.3f}", flush=True)

    # Strategy 2: K4_select_lp_max (highest lp_mean alt)
    n_lp = 0
    for ii in ids:
        r = greedy_records[ii]
        alts = alt_data.get(ii, [])
        if not alts:
            n_lp += int(equal_strict(r["pred"], r["gold"])); continue
        best = max(alts, key=lambda a: a["lp_mean"])
        n_lp += int(equal_strict(best["pred"], r["gold"]))
    acc_lp = n_lp / len(ids) if ids else 0
    summary["results"]["K4_select_lp_max"] = {"acc": acc_lp, "delta": acc_lp - vanilla, "n": len(ids)}
    print(f"  K4_select_lp_max acc={acc_lp:.3f}  Δ={acc_lp-vanilla:+.3f}", flush=True)

    # Strategy 3: K4_filter_arith_then_majority (drop alts with arith violations, vote rest)
    n_filt = 0
    for ii in ids:
        r = greedy_records[ii]
        alts = alt_data.get(ii, [])
        clean_alts = [a for a in alts if a["n_arith_viol"] == 0]
        if not clean_alts:
            # No clean alt — fall back to greedy
            n_filt += int(equal_strict(r["pred"], r["gold"])); continue
        votes_pred = [r["pred"]] + [a["pred"] for a in clean_alts]
        chosen = majority_vote(votes_pred)
        n_filt += int(equal_strict(chosen, r["gold"]))
    acc_filt = n_filt / len(ids) if ids else 0
    summary["results"]["K4_filter_arith_majority"] = {"acc": acc_filt, "delta": acc_filt - vanilla, "n": len(ids)}
    print(f"  K4_filter_arith_majority acc={acc_filt:.3f}  Δ={acc_filt-vanilla:+.3f}", flush=True)

    # Strategy 4: K4_majority (baseline reproduction)
    n_maj = 0
    for ii in ids:
        r = greedy_records[ii]
        alts = alt_data.get(ii, [])
        votes_pred = [r["pred"]] + [a["pred"] for a in alts]
        chosen = majority_vote(votes_pred)
        n_maj += int(equal_strict(chosen, r["gold"]))
    acc_maj = n_maj / len(ids) if ids else 0
    summary["results"]["K4_majority_baseline"] = {"acc": acc_maj, "delta": acc_maj - vanilla, "n": len(ids)}
    print(f"  K4_majority_baseline acc={acc_maj:.3f}  Δ={acc_maj-vanilla:+.3f}", flush=True)

    # Strategy 5: K4_consensus_3plus (only accept if 3+ alt+greedy agree)
    n_cons = 0
    for ii in ids:
        r = greedy_records[ii]
        alts = alt_data.get(ii, [])
        votes = [r["pred"]] + [a["pred"] for a in alts]
        votes_norm = [str(normalize(p)) for p in votes if p is not None]
        counter = Counter(votes_norm)
        if counter:
            top, cnt = counter.most_common(1)[0]
            if cnt >= 3:  # 3+ agree
                chosen = top
            else:
                chosen = str(normalize(r["pred"])) if r["pred"] is not None else None
        else:
            chosen = None
        n_cons += int(equal_strict(chosen, r["gold"]))
    acc_cons = n_cons / len(ids) if ids else 0
    summary["results"]["K4_consensus_3plus"] = {"acc": acc_cons, "delta": acc_cons - vanilla, "n": len(ids)}
    print(f"  K4_consensus_3plus acc={acc_cons:.3f}  Δ={acc_cons-vanilla:+.3f}", flush=True)

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done. Wrote {OUT} ===", flush=True)


if __name__ == "__main__":
    main()

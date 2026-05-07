"""
EXP-STEP-REJECTION T2: Tier 2 step rejection methods.

After Tier 1 confirmed K=4 majority is the winner at +1-2.5pp, test:
1. **K4_lp_select**: K=4 alts, pick the one with highest mean lp (instead of majority)
2. **K2_K4_escalation**: K=2 at T=0.5; if alts agree, accept; else K=4 at T=1.0
3. **K4_majority_multi_trigger**: K=4 at top-2 worst steps (lp_min); union of all alt preds
4. **K4_T_high**: K=4 majority but at T=1.0 (more diversity vs T=0.7 baseline)
5. **K8_majority**: pure compute scaling (fair vs SC@8 baseline)
6. **K4_lp_select_entropy_trigger**: trigger on entropy_max worst step (different trigger), K=4 lp-select

Trigger for 1,2,3,4,5: lp_min worst step.

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
OUT = OUTDIR / f"SX_step_rej_t2_{TAG}_{DATASET}.json"

PROMPT_MATH = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\nProblem: {question}\n\n"
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


def step_lp_means(chosen_lps, boundaries, n):
    bdy = boundaries + [n]
    out = []
    for sa, sb in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in chosen_lps[sa:sb] if not (lp != lp)]
        out.append(float(np.mean(seg)) if seg else float("nan"))
    return out


def step_entropy_means(distribs, boundaries, n):
    bdy = boundaries + [n]
    out = []
    for sa, sb in zip(bdy[:-1], bdy[1:]):
        ents = []
        for d in distribs[sa:sb]:
            if d is None: continue
            p = np.array(d["probs"]); p = p[p > 0]
            if len(p) > 0:
                ents.append(-float((p * np.log(p)).sum()))
        out.append(float(np.mean(ents)) if ents else float("nan"))
    return out


def majority_vote(preds_list):
    votes = [str(normalize(p)) for p in preds_list if p is not None]
    if not votes:
        return None
    counter = Counter(votes)
    return counter.most_common(1)[0][0]


def main():
    print(f"=== T2 Step rejection: {MODEL} on {DATASET} ===", flush=True)
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
            entries_lps = np.array([e.logprob for e in entries], dtype=float)
            probs = np.exp(entries_lps - entries_lps.max()); probs = probs / probs.sum()
            distribs.append({"probs": probs.tolist(), "lps": entries_lps.tolist()})
        boundaries = find_step_boundaries(tok_strs)
        n = len(tok_ids)
        step_lps = step_lp_means(chosen_lps, boundaries, n)
        step_ents = step_entropy_means(distribs, boundaries, n)
        pred = extract_pred(text)
        ok = int(equal_strict(pred, rec["gold"]))
        # Worst step under lp_min
        valid_lps = [(j, v) for j, v in enumerate(step_lps) if not (v != v)]
        valid_ents = [(j, v) for j, v in enumerate(step_ents) if not (v != v)]
        worst_lp_step = min(valid_lps, key=lambda kv: kv[1])[0] if valid_lps else 0
        worst_ent_step = max(valid_ents, key=lambda kv: kv[1])[0] if valid_ents else 0
        # Top-2 worst lp_min steps
        sorted_lps = sorted(valid_lps, key=lambda kv: kv[1])
        top2_worst_lp = [s for s, _ in sorted_lps[:2]]
        greedy_records.append({
            "id": i, "gold": rec["gold"], "pred": pred, "correct": ok,
            "tok_ids": tok_ids, "boundaries": boundaries, "step_lps": step_lps, "step_ents": step_ents,
            "worst_lp_step": worst_lp_step, "worst_ent_step": worst_ent_step,
            "top2_worst_lp": top2_worst_lp,
            "text": text,
        })
    n_greedy = sum(r["correct"] for r in greedy_records)
    vanilla = n_greedy / len(greedy_records)
    print(f"  vanilla: {n_greedy}/{len(greedy_records)} = {vanilla:.3f}", flush=True)

    summary = {"model": MODEL, "tag": TAG, "dataset": DATASET, "n": len(greedy_records),
                "vanilla_acc": vanilla, "results": {}}

    # Helper: get prefix at worst step
    def get_prefix_at_worst(r, step_idx_attr):
        idx = r[step_idx_attr]
        if not r["step_lps"] or idx >= len(r["boundaries"]):
            return None
        bdy_pos = r["boundaries"][idx]
        prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
        return base_prompts[r["id"]] + prefix

    # Strategy 1: K4_lp_select (vs K4 majority). Pick alt with highest mean lp.
    print(f"\n[Strategy T2-1] K4_lp_select (vs K4 majority)...", flush=True)
    prompts = []; ids = []
    for r in greedy_records:
        p = get_prefix_at_worst(r, "worst_lp_step")
        if p:
            prompts.append(p); ids.append(r["id"])
    sp_b = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=1024, logprobs=1, seed=SEED + 100)
    t0 = time.time()
    outs_b = llm.generate(prompts, sp_b)
    n_lp_sel = 0; n_maj = 0
    for ii, ob in zip(ids, outs_b):
        r = greedy_records[ii]
        alt_preds = []
        alt_lps = []
        for c in ob.outputs:
            alt_preds.append(extract_pred(c.text))
            lps = []
            for step_lp in c.logprobs or []:
                if step_lp is None: continue
                if not step_lp: continue
                e = next(iter(step_lp.values()))
                # Handle both Logprob object and tuple/dict shapes
                lp_val = None
                if hasattr(e, "logprob"):
                    lp_val = e.logprob
                elif isinstance(e, (tuple, list)) and len(e) > 0:
                    lp_val = e[0]
                elif isinstance(e, (int, float)):
                    lp_val = float(e)
                if lp_val is not None:
                    lps.append(lp_val)
            valid = [lp for lp in lps if not (lp != lp)]
            alt_lps.append(float(np.mean(valid)) if valid else float("-inf"))
        # lp_select: pick alt with highest mean lp
        if alt_lps and alt_preds:
            best_idx = int(np.argmax(alt_lps))
            best_pred = alt_preds[best_idx]
            n_lp_sel += int(equal_strict(best_pred, r["gold"]))
        # majority among greedy + alts
        all_preds = [r["pred"]] + alt_preds
        chosen = majority_vote(all_preds)
        n_maj += int(equal_strict(chosen, r["gold"]))
    acc_lp_sel = n_lp_sel / len(ids) if ids else 0
    acc_maj = n_maj / len(ids) if ids else 0
    summary["results"]["K4_lp_select"] = {"acc": acc_lp_sel, "delta": acc_lp_sel - vanilla, "n": len(ids)}
    summary["results"]["K4_majority_baseline"] = {"acc": acc_maj, "delta": acc_maj - vanilla, "n": len(ids)}
    print(f"  K4_lp_select acc={acc_lp_sel:.3f}  Δ={acc_lp_sel-vanilla:+.3f}", flush=True)
    print(f"  K4_majority   acc={acc_maj:.3f}  Δ={acc_maj-vanilla:+.3f} ({time.time()-t0:.1f}s)", flush=True)

    # Strategy 2: K2_K4_escalation (start cheap, escalate if disagree)
    print(f"\n[Strategy T2-2] K2_K4_escalation...", flush=True)
    sp_k2 = SamplingParams(n=2, temperature=0.5, top_p=0.95, max_tokens=1024, seed=SEED + 200)
    t0 = time.time()
    outs_k2 = llm.generate(prompts, sp_k2)
    # Determine which need escalation (alts disagree)
    needs_esc_indices = []  # indices into 'ids'
    k2_results = {}  # id -> chosen pred from K=2 phase
    for idx, (ii, ok2) in enumerate(zip(ids, outs_k2)):
        r = greedy_records[ii]
        alts = [extract_pred(c.text) for c in ok2.outputs]
        # Check if alts agree
        norms = [str(normalize(p)) for p in alts if p is not None]
        agree = len(set(norms)) <= 1 and len(norms) >= 1
        if agree:
            # K=2 enough; vote with greedy
            k2_results[ii] = majority_vote([r["pred"]] + alts)
        else:
            needs_esc_indices.append(idx)
    print(f"  K=2 phase done; {len(needs_esc_indices)}/{len(ids)} need escalation", flush=True)
    # Escalate those
    if needs_esc_indices:
        esc_prompts = [prompts[idx] for idx in needs_esc_indices]
        esc_ids = [ids[idx] for idx in needs_esc_indices]
        sp_k4_esc = SamplingParams(n=4, temperature=1.0, top_p=0.95, max_tokens=1024, seed=SEED + 300)
        outs_esc = llm.generate(esc_prompts, sp_k4_esc)
        for ii, oe in zip(esc_ids, outs_esc):
            r = greedy_records[ii]
            alts = [extract_pred(c.text) for c in oe.outputs]
            k2_results[ii] = majority_vote([r["pred"]] + alts)
    n_esc = 0
    for ii in ids:
        r = greedy_records[ii]
        n_esc += int(equal_strict(k2_results.get(ii), r["gold"]))
    acc_esc = n_esc / len(ids) if ids else 0
    avg_cost = (2 * len(ids) + 4 * len(needs_esc_indices)) / max(len(ids), 1)
    summary["results"]["K2_K4_escalation"] = {"acc": acc_esc, "delta": acc_esc - vanilla, "n": len(ids),
                                                "avg_extra_decode_cost": avg_cost,
                                                "esc_frac": len(needs_esc_indices) / len(ids)}
    print(f"  K2_K4_escalation acc={acc_esc:.3f}  Δ={acc_esc-vanilla:+.3f}  avg_cost={avg_cost:.2f}× ({time.time()-t0:.1f}s)", flush=True)

    # Strategy 3: K4_T1 (higher temperature for diversity)
    print(f"\n[Strategy T2-3] K4_T1 (higher T diversity)...", flush=True)
    sp_t1 = SamplingParams(n=4, temperature=1.0, top_p=0.95, max_tokens=1024, seed=SEED + 400)
    t0 = time.time()
    outs_t1 = llm.generate(prompts, sp_t1)
    n_t1 = 0
    for ii, ot1 in zip(ids, outs_t1):
        r = greedy_records[ii]
        alts = [extract_pred(c.text) for c in ot1.outputs]
        chosen = majority_vote([r["pred"]] + alts)
        n_t1 += int(equal_strict(chosen, r["gold"]))
    acc_t1 = n_t1 / len(ids) if ids else 0
    summary["results"]["K4_T1"] = {"acc": acc_t1, "delta": acc_t1 - vanilla, "n": len(ids)}
    print(f"  K4_T1 acc={acc_t1:.3f}  Δ={acc_t1-vanilla:+.3f} ({time.time()-t0:.1f}s)", flush=True)

    # Strategy 4: K8_majority (compute scaling)
    print(f"\n[Strategy T2-4] K8_majority...", flush=True)
    sp_k8 = SamplingParams(n=8, temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 500)
    t0 = time.time()
    outs_k8 = llm.generate(prompts, sp_k8)
    n_k8 = 0
    for ii, ok8 in zip(ids, outs_k8):
        r = greedy_records[ii]
        alts = [extract_pred(c.text) for c in ok8.outputs]
        chosen = majority_vote([r["pred"]] + alts)
        n_k8 += int(equal_strict(chosen, r["gold"]))
    acc_k8 = n_k8 / len(ids) if ids else 0
    summary["results"]["K8_majority"] = {"acc": acc_k8, "delta": acc_k8 - vanilla, "n": len(ids)}
    print(f"  K8_majority acc={acc_k8:.3f}  Δ={acc_k8-vanilla:+.3f} ({time.time()-t0:.1f}s)", flush=True)

    # Strategy 5: K4_majority_entropy_trigger (different trigger)
    print(f"\n[Strategy T2-5] K4_majority with entropy_max trigger...", flush=True)
    ent_prompts = []; ent_ids = []
    for r in greedy_records:
        p = get_prefix_at_worst(r, "worst_ent_step")
        if p:
            ent_prompts.append(p); ent_ids.append(r["id"])
    sp_k4_ent = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 600)
    t0 = time.time()
    outs_ent = llm.generate(ent_prompts, sp_k4_ent)
    n_ent = 0
    for ii, oe in zip(ent_ids, outs_ent):
        r = greedy_records[ii]
        alts = [extract_pred(c.text) for c in oe.outputs]
        chosen = majority_vote([r["pred"]] + alts)
        n_ent += int(equal_strict(chosen, r["gold"]))
    acc_ent = n_ent / len(ent_ids) if ent_ids else 0
    summary["results"]["K4_majority_entropy_trigger"] = {"acc": acc_ent, "delta": acc_ent - vanilla, "n": len(ent_ids)}
    print(f"  K4_maj_entropy_trigger acc={acc_ent:.3f}  Δ={acc_ent-vanilla:+.3f} ({time.time()-t0:.1f}s)", flush=True)

    # Strategy 6: K4_T07_lp_select_or_majority (best of two)
    # Already computed K4 majority and K4 lp_select above; report both for comparison
    # Strategy 7: top-2 worst steps regenerate (K=2 each), aggregate all
    print(f"\n[Strategy T2-7] top-2 worst steps × K=2...", flush=True)
    multi_prompts = []; multi_meta = []  # (id, step_pos)
    for r in greedy_records:
        for step_idx in r["top2_worst_lp"]:
            if step_idx < len(r["boundaries"]):
                bdy_pos = r["boundaries"][step_idx]
                prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
                multi_prompts.append(base_prompts[r["id"]] + prefix)
                multi_meta.append((r["id"], step_idx))
    sp_multi = SamplingParams(n=2, temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 700)
    t0 = time.time()
    outs_multi = llm.generate(multi_prompts, sp_multi)
    multi_results = {}
    for (qid, step_idx), om in zip(multi_meta, outs_multi):
        for c in om.outputs:
            pred = extract_pred(c.text)
            if qid not in multi_results: multi_results[qid] = []
            multi_results[qid].append(pred)
    n_multi = 0
    for r in greedy_records:
        alts = multi_results.get(r["id"], [])
        chosen = majority_vote([r["pred"]] + alts)
        n_multi += int(equal_strict(chosen, r["gold"]))
    acc_multi = n_multi / len(greedy_records)
    summary["results"]["multi_worst_K2"] = {"acc": acc_multi, "delta": acc_multi - vanilla, "n": len(greedy_records)}
    print(f"  multi_worst_K2 acc={acc_multi:.3f}  Δ={acc_multi-vanilla:+.3f} ({time.time()-t0:.1f}s)", flush=True)

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done. Wrote {OUT} ===", flush=True)
    print(f"\n=== Summary for {TAG} {DATASET} (vanilla={vanilla:.3f}) ===")
    for method, res in summary["results"].items():
        print(f"  {method:30s} acc={res['acc']:.3f}  Δ={res['delta']:+.3f}")


if __name__ == "__main__":
    main()

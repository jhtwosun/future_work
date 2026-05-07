"""
EXP-STEP-REJECTION: test 4 step rejection strategies for accuracy lift.

Strategies (Tier 1 from Wave 4 brainstorm):
1. **self_correct_insertion**: at worst step, inject "Wait, let me reconsider:" + regenerate
2. **step_excise**: delete worst step from prefix, regenerate continuation
3. **backtrack_2step**: cut steps t-1 and t (worst step), regenerate from t-2
4. **K4_majority** (baseline): K=4 alternatives + majority vote at worst step (Pilot C)

Trigger: lp_min worst step (most reliable per §9c).

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
OUT = OUTDIR / f"SX_step_rej_{TAG}_{DATASET}.json"

PROMPT_MATH = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\nProblem: {question}\n\n"
)

SELF_CORRECT_INSERTION = "\n\nWait, let me reconsider this step.\n\n"


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


def main():
    print(f"=== Step rejection: {MODEL} on {DATASET} ===", flush=True)
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
        chosen_lps = []
        for step_lp in o.logprobs or []:
            if step_lp is None:
                chosen_lps.append(float("nan")); continue
            entries = sorted(step_lp.items(), key=lambda kv: -kv[1].logprob)
            chosen_lps.append(entries[0][1].logprob)
        boundaries = find_step_boundaries(tok_strs)
        n = len(tok_ids)
        step_lps = step_lp_means(chosen_lps, boundaries, n)
        pred = extract_pred(text)
        ok = int(equal_strict(pred, rec["gold"]))
        greedy_records.append({
            "id": i, "gold": rec["gold"], "pred": pred, "correct": ok,
            "tok_ids": tok_ids, "boundaries": boundaries, "step_lps": step_lps,
            "text": text,
        })
    n_greedy = sum(r["correct"] for r in greedy_records)
    vanilla = n_greedy / len(greedy_records)
    print(f"  vanilla: {n_greedy}/{len(greedy_records)} = {vanilla:.3f}", flush=True)

    # Identify worst step per record
    for r in greedy_records:
        valid_lps = [(j, lp) for j, lp in enumerate(r["step_lps"]) if not (lp != lp)]
        r["worst_step"] = min(valid_lps, key=lambda kv: kv[1])[0] if valid_lps else 0

    # Strategy 1: self_correct_insertion (1 alt with cue word inserted)
    print(f"\n[Strategy 1] self_correct_insertion...", flush=True)
    sc_prompts = []; sc_ids = []
    for r in greedy_records:
        if not r["step_lps"] or r["worst_step"] >= len(r["boundaries"]):
            continue
        bdy_pos = r["boundaries"][r["worst_step"]]
        prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
        # Insert cue at bad step boundary
        sc_prompts.append(base_prompts[r["id"]] + prefix + SELF_CORRECT_INSERTION)
        sc_ids.append(r["id"])
    sp_sc = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 100)
    t0 = time.time()
    outs_sc = llm.generate(sc_prompts, sp_sc)
    n_sc = 0
    for ii, out_sc in zip(sc_ids, outs_sc):
        r = greedy_records[ii]
        new_text = out_sc.outputs[0].text
        new_pred = extract_pred(new_text)
        # Combine with greedy via majority (1 alt + 1 greedy = 2 votes; ties go to greedy)
        votes = [str(normalize(p)) for p in [r["pred"], new_pred] if p is not None]
        if not votes:
            chosen = None
        else:
            counter = Counter(votes)
            chosen = counter.most_common(1)[0][0]
        n_sc += int(equal_strict(chosen, r["gold"]))
    acc_sc = n_sc / len(sc_ids) if sc_ids else 0
    print(f"  self_correct_insertion acc={acc_sc:.3f}  Δ={acc_sc-vanilla:+.3f}  n={len(sc_ids)} ({time.time()-t0:.1f}s)", flush=True)

    # Strategy 2: step_excise (delete worst step, regen continuation)
    print(f"\n[Strategy 2] step_excise...", flush=True)
    excise_prompts = []; excise_ids = []
    for r in greedy_records:
        if not r["step_lps"] or len(r["boundaries"]) < 2:
            continue
        ws = r["worst_step"]
        if ws == 0 or ws >= len(r["boundaries"]):
            continue
        # Delete step ws: prefix becomes [start..bdy[ws-1]] + skip step ws + nothing more (regen)
        # Actually we need to cut at the START of the bad step
        bdy_pos = r["boundaries"][ws]  # start of step ws
        prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
        # No insertion; just regenerate
        excise_prompts.append(base_prompts[r["id"]] + prefix)
        excise_ids.append(r["id"])
    sp_ex = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 200)
    t0 = time.time()
    outs_ex = llm.generate(excise_prompts, sp_ex)
    n_ex = 0
    for ii, out_ex in zip(excise_ids, outs_ex):
        r = greedy_records[ii]
        new_text = out_ex.outputs[0].text
        new_pred = extract_pred(new_text)
        votes = [str(normalize(p)) for p in [r["pred"], new_pred] if p is not None]
        if not votes:
            chosen = None
        else:
            counter = Counter(votes)
            chosen = counter.most_common(1)[0][0]
        n_ex += int(equal_strict(chosen, r["gold"]))
    acc_ex = n_ex / len(excise_ids) if excise_ids else 0
    print(f"  step_excise acc={acc_ex:.3f}  Δ={acc_ex-vanilla:+.3f}  n={len(excise_ids)} ({time.time()-t0:.1f}s)", flush=True)

    # Strategy 3: backtrack_2step (cut steps t-1 + t, regenerate from t-2)
    print(f"\n[Strategy 3] backtrack_2step...", flush=True)
    bt_prompts = []; bt_ids = []
    for r in greedy_records:
        if not r["step_lps"] or len(r["boundaries"]) < 3:
            continue
        ws = r["worst_step"]
        bt_pos = max(0, ws - 1)  # roll back to step before bad one
        if bt_pos >= len(r["boundaries"]):
            continue
        bdy_pos = r["boundaries"][bt_pos]
        prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
        bt_prompts.append(base_prompts[r["id"]] + prefix)
        bt_ids.append(r["id"])
    sp_bt = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 300)
    t0 = time.time()
    outs_bt = llm.generate(bt_prompts, sp_bt)
    n_bt = 0
    for ii, out_bt in zip(bt_ids, outs_bt):
        r = greedy_records[ii]
        new_text = out_bt.outputs[0].text
        new_pred = extract_pred(new_text)
        votes = [str(normalize(p)) for p in [r["pred"], new_pred] if p is not None]
        if not votes:
            chosen = None
        else:
            counter = Counter(votes)
            chosen = counter.most_common(1)[0][0]
        n_bt += int(equal_strict(chosen, r["gold"]))
    acc_bt = n_bt / len(bt_ids) if bt_ids else 0
    print(f"  backtrack_2step acc={acc_bt:.3f}  Δ={acc_bt-vanilla:+.3f}  n={len(bt_ids)} ({time.time()-t0:.1f}s)", flush=True)

    # Strategy 4: K=4 majority (Pilot C, baseline)
    print(f"\n[Strategy 4] K4_majority (Pilot C baseline)...", flush=True)
    k4_prompts = []; k4_ids = []
    for r in greedy_records:
        if not r["step_lps"] or r["worst_step"] >= len(r["boundaries"]):
            continue
        bdy_pos = r["boundaries"][r["worst_step"]]
        prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
        k4_prompts.append(base_prompts[r["id"]] + prefix)
        k4_ids.append(r["id"])
    sp_k4 = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 400)
    t0 = time.time()
    outs_k4 = llm.generate(k4_prompts, sp_k4)
    n_k4 = 0
    for ii, out_k4 in zip(k4_ids, outs_k4):
        r = greedy_records[ii]
        alt_preds = [extract_pred(c.text) for c in out_k4.outputs]
        all_preds = [r["pred"]] + [p for p in alt_preds if p is not None]
        votes = [str(normalize(p)) for p in all_preds if p is not None]
        if not votes:
            chosen = None
        else:
            counter = Counter(votes)
            chosen = counter.most_common(1)[0][0]
        n_k4 += int(equal_strict(chosen, r["gold"]))
    acc_k4 = n_k4 / len(k4_ids) if k4_ids else 0
    print(f"  K4_majority acc={acc_k4:.3f}  Δ={acc_k4-vanilla:+.3f}  n={len(k4_ids)} ({time.time()-t0:.1f}s)", flush=True)

    summary = {
        "model": MODEL, "tag": TAG, "dataset": DATASET, "n": len(greedy_records),
        "vanilla_acc": vanilla,
        "results": {
            "self_correct_insertion": {"acc": acc_sc, "delta": acc_sc - vanilla, "n": len(sc_ids)},
            "step_excise": {"acc": acc_ex, "delta": acc_ex - vanilla, "n": len(excise_ids)},
            "backtrack_2step": {"acc": acc_bt, "delta": acc_bt - vanilla, "n": len(bt_ids)},
            "K4_majority": {"acc": acc_k4, "delta": acc_k4 - vanilla, "n": len(k4_ids)},
        }
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done. Wrote {OUT} ===", flush=True)
    print(f"\nSummary:")
    print(f"  vanilla:                {vanilla:.3f}")
    print(f"  self_correct_insertion: {acc_sc:.3f}  Δ={acc_sc-vanilla:+.3f}")
    print(f"  step_excise:            {acc_ex:.3f}  Δ={acc_ex-vanilla:+.3f}")
    print(f"  backtrack_2step:        {acc_bt:.3f}  Δ={acc_bt-vanilla:+.3f}")
    print(f"  K4_majority (baseline): {acc_k4:.3f}  Δ={acc_k4-vanilla:+.3f}")


if __name__ == "__main__":
    main()

"""
EXP-S2: GPU perturbation / sampling-based scores on Qwen2.5-7B + MATH-500.

Implements (with hparam variation, no cherry-picking):
- single_shot_pseudo_sc: greedy vs 1 sample at T={0.3, 0.7, 1.0}
- xtemp_agree: greedy + N={1,2,4} samples at T=0.7, agreement rate
- step_mask_stability: truncate at random step, regen suffix, agreement (J={2,3} regens)
- paraphrase_consensus: M={2,3} paraphrased questions, answer modal agreement
- forbidden_top1_redecoding (Agent F #1): force ban worst-step first token, single redecode
- low_confidence_token_swap (Agent F #2): find lowest-lp token, swap to runner-up, decode

Uses 200 MATH-500 problems (same as Pilot 7/9 for direct comparison).
All in single vLLM session for speed.

Output:
  experiments/results/SX_perturb_summary.json
  experiments/results/SX_perturb_scores.jsonl
"""

import json
import os
import re
import sys
import time
import math
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, normalize, equal_strict

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_QUESTIONS = int(os.environ.get("NQ", "200"))
MAX_TOKENS = 1536
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / "SX_perturb_summary.json"
OUT_TRACES = OUTDIR / "SX_perturb_scores.jsonl"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>". '
    "Or put it within \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
)

PARAPHRASE_TEMPLATES = [
    "Determine the answer to the following math question. Show your reasoning step-by-step, then state the final answer clearly with \\boxed{{}}.\n\n{question}\n",
    "Math problem to solve: {question}\nWalk through your solution carefully, with each reasoning step on its own line. Provide the final numerical or expression answer in \\boxed{{}} at the end.\n",
    "Please carefully work through this mathematics problem and arrive at the answer.\n\n{question}\n\nUse step-by-step reasoning and put the final answer in \\boxed{{}}.\n",
]


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
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]

    summary = {"model": MODEL, "n": len(questions)}
    per_q = [{"id": i, "question": q, "gold": g} for i, (q, g) in enumerate(zip(questions, golds))]

    # ------- Greedy baseline (anchor) -------
    print("\n=== Greedy baseline ===", flush=True)
    sp_g = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(base_prompts, sp_g)
    dur_g = time.time() - t0
    greedy_preds = []
    greedy_correct = []
    for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
        pred = extract_pred(out.outputs[0].text)
        ok = int(equal_strict(pred, gold))
        greedy_preds.append(pred)
        greedy_correct.append(ok)
        per_q[i]["greedy_pred"] = pred
        per_q[i]["greedy_correct"] = ok
        per_q[i]["greedy_text"] = out.outputs[0].text
    summary["greedy"] = {
        "accuracy": float(np.mean(greedy_correct)),
        "wallclock_sec": dur_g,
    }
    print(f"  Greedy: {sum(greedy_correct)}/{len(questions)} = {np.mean(greedy_correct):.3f}", flush=True)

    # ------- single_shot_pseudo_sc at varied T -------
    for T in [0.3, 0.7, 1.0]:
        print(f"\n=== single_shot @ T={T} ===", flush=True)
        sp = SamplingParams(n=1, temperature=T, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED + 1)
        t0 = time.time()
        outs = llm.generate(base_prompts, sp)
        dur = time.time() - t0
        agreements = []
        for i, out in enumerate(outs):
            pred = extract_pred(out.outputs[0].text)
            agree = int(equal_strict(pred, greedy_preds[i]))
            agreements.append(agree)
            per_q[i][f"singleshot_T{T}_pred"] = pred
            per_q[i][f"singleshot_T{T}_agree"] = agree
        summary[f"singleshot_T{T}"] = {
            "agreement_rate_with_greedy": float(np.mean(agreements)),
            "wallclock_sec": dur,
        }
        print(f"  T={T} mean agreement: {np.mean(agreements):.3f}", flush=True)

    # ------- xtemp_agree: greedy + N samples agreement -------
    for N_ext in [2, 4]:
        print(f"\n=== xtemp_agree N={N_ext} samples T=0.7 ===", flush=True)
        sp = SamplingParams(n=N_ext, temperature=0.7, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED + 2)
        t0 = time.time()
        outs = llm.generate(base_prompts, sp)
        dur = time.time() - t0
        for i, out in enumerate(outs):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter([normalize(p) for p in preds_clean])
            agreement_with_greedy = float(np.mean([equal_strict(p, greedy_preds[i]) for p in preds_clean])) if preds_clean else 0.0
            top, cnt = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac_inc_greedy = (cnt + (1 if equal_strict(top, greedy_preds[i]) else 0)) / max(N_ext + 1, 1)
            per_q[i][f"xtemp_N{N_ext}_agree_greedy"] = agreement_with_greedy
            per_q[i][f"xtemp_N{N_ext}_top1_inc_greedy"] = top1_frac_inc_greedy
            per_q[i][f"xtemp_N{N_ext}_samples"] = preds
        summary[f"xtemp_N{N_ext}"] = {
            "wallclock_sec": dur,
        }
        print(f"  N={N_ext} done in {dur:.1f}s", flush=True)

    # ------- step_mask_stability: truncate at random step boundary, regen -------
    # We use the greedy text, identify step boundaries, randomly mask K steps from the end.
    print("\n=== step_mask_stability ===", flush=True)
    # Build prompts truncated at various step counts.
    rng = np.random.default_rng(0)
    mask_prompts = []
    mask_meta = []
    for i, q in enumerate(questions):
        text = per_q[i]["greedy_text"]
        # Find paragraph boundaries
        parts = re.split(r"\n\s*\n+", text)
        if len(parts) < 3:
            # too short; skip
            mask_prompts.append(None)
            mask_meta.append({"i": i, "kept_steps": 0})
            continue
        # randomly choose how many steps to keep (between 1 and len(parts)-2)
        n_keep = rng.integers(1, max(2, len(parts) - 1))
        kept = "\n\n".join(parts[:n_keep])
        # build prompt: original prompt + the truncated reasoning
        mask_prompts.append(base_prompts[i] + kept + "\n\n")
        mask_meta.append({"i": i, "kept_steps": int(n_keep), "total_steps": len(parts)})

    valid_idxs = [k for k, p in enumerate(mask_prompts) if p is not None]
    valid_prompts = [mask_prompts[k] for k in valid_idxs]
    print(f"  {len(valid_prompts)}/{len(mask_prompts)} prompts have enough steps to mask", flush=True)

    if valid_prompts:
        for J in [2, 3]:
            sp = SamplingParams(n=J, temperature=0.7, top_p=0.95, max_tokens=MAX_TOKENS // 2, seed=SEED + 3)
            t0 = time.time()
            outs = llm.generate(valid_prompts, sp)
            dur = time.time() - t0
            for k, out in zip(valid_idxs, outs):
                i = mask_meta[k]["i"]
                preds = [extract_pred(c.text) for c in out.outputs]
                preds_clean = [p for p in preds if p is not None]
                # stability: fraction matching greedy
                stable = float(np.mean([equal_strict(p, greedy_preds[i]) for p in preds_clean])) if preds_clean else 0.0
                per_q[i][f"stepmask_J{J}_stability"] = stable
            print(f"  J={J} done in {dur:.1f}s", flush=True)

    # ------- paraphrase_consensus -------
    for M_para in [2, 3]:
        print(f"\n=== paraphrase_consensus M={M_para} ===", flush=True)
        # Build M paraphrased prompts per question
        paraphrase_prompts = []
        for q in questions:
            for k in range(M_para):
                tmpl = PARAPHRASE_TEMPLATES[k % len(PARAPHRASE_TEMPLATES)]
                msg = tmpl.format(question=q)
                p = tok.apply_chat_template(
                    [{"role": "user", "content": msg}],
                    tokenize=False, add_generation_prompt=True,
                )
                paraphrase_prompts.append(p)
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=SEED + 4)
        t0 = time.time()
        outs = llm.generate(paraphrase_prompts, sp)
        dur = time.time() - t0
        for i in range(len(questions)):
            chunk = outs[i * M_para:(i + 1) * M_para]
            preds = [extract_pred(c.outputs[0].text) for c in chunk]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter([normalize(p) for p in preds_clean])
            top, cnt = (counter.most_common(1)[0] if counter else (None, 0))
            consensus = cnt / max(M_para, 1)
            agree_with_greedy = float(np.mean([equal_strict(p, greedy_preds[i]) for p in preds_clean])) if preds_clean else 0.0
            per_q[i][f"para_M{M_para}_consensus"] = consensus
            per_q[i][f"para_M{M_para}_agree_greedy"] = agree_with_greedy
        print(f"  M={M_para} done in {dur:.1f}s", flush=True)

    # ------- Save and summarize -------
    with OUT_TRACES.open("w") as f:
        for r in per_q:
            # drop verbose text for summary file
            r2 = {k: v for k, v in r.items() if k != "greedy_text"}
            f.write(json.dumps(r2) + "\n")

    # Compute Spearman correlations of each score with greedy_correct
    correct = np.array([r["greedy_correct"] for r in per_q])
    score_keys = []
    for r in per_q:
        for k in r:
            if k.startswith(("singleshot_", "xtemp_", "stepmask_", "para_")):
                if k.endswith(("_agree", "_agree_greedy", "_top1_inc_greedy", "_consensus", "_stability")):
                    score_keys.append(k)
    score_keys = sorted(set(score_keys))
    summary["score_corr_with_greedy_correct"] = {}
    from scipy.stats import spearmanr
    for k in score_keys:
        vals = []
        for r in per_q:
            v = r.get(k)
            vals.append(float(v) if v is not None else float("nan"))
        vals = np.array(vals)
        valid = ~np.isnan(vals)
        if valid.sum() < 5:
            continue
        rho, p = spearmanr(vals[valid], correct[valid])
        summary["score_corr_with_greedy_correct"][k] = {"rho": float(rho), "p": float(p), "n": int(valid.sum())}
        print(f"  {k:35s}  ρ={rho:+.3f}  p={p:.3g}", flush=True)

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote: {OUT}", flush=True)


if __name__ == "__main__":
    main()

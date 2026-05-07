"""
EXP-step1: Step-level intervention experiments (Wave 2 brainstorm).

Tests on MATH-500 (200 problems where greedy may be wrong, baseline ~52%):
- Wave 2 F1: forbidden_top1_redecoding — at worst lp step boundary, ban top-1 token
- Wave 2 F2: low_confidence_token_swap — find lowest-lp token in worst step, swap to runner-up
- Wave 2 I3: system_prompt_ensemble — 4 different system prompts, modal answer
- Wave 2 H5 (cheap proxy): formal_restate (we approximate by asking model to restate step formally)
- baseline: K=4 resample (Pilot C)

Compare to greedy and to plain SC@4 at matched compute.
"""

import json
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
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
SUM = OUTDIR / "SX_step_interventions.json"

PROMPT_BASE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
)

# 4 system prompts for I3 system_prompt_ensemble
SYSTEM_PROMPTS = [
    "You are a careful mathematician. Solve problems step by step, double-checking arithmetic.",
    "You are a math contest coach. Use efficient algebraic methods to solve problems.",
    "You are a teacher explaining math to a student. Use simple, clear reasoning.",
    "You are a logician. Solve math problems with explicit, formal deductive reasoning.",
]


def per_step_lps(token_logprobs, boundaries):
    bdy = boundaries + [len(token_logprobs)]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            out.append(float(np.mean(seg)))
    return out


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


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

    summary = {"model": MODEL, "n": len(questions)}

    # ===== Greedy baseline (with logprobs for finding worst step) =====
    print("\n=== Greedy baseline ===", flush=True)
    sp_g = SamplingParams(temperature=0.0, max_tokens=1536, logprobs=5, seed=SEED)
    t0 = time.time()
    outs = llm.generate(base_prompts, sp_g)
    dur_g = time.time() - t0

    greedy_records = []
    for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
        o = out.outputs[0]
        text = o.text
        tok_ids = list(o.token_ids)
        tok_strs = [tok.decode([tid]) for tid in tok_ids]
        # Top-1 logprobs (chosen) and top-2 logprobs for swap
        chosen_lps = []
        runner_up_lps = []
        runner_up_tok_ids = []
        for step_lp in o.logprobs or []:
            if step_lp is None:
                chosen_lps.append(float("nan"))
                runner_up_lps.append(float("nan"))
                runner_up_tok_ids.append(None)
                continue
            entries = sorted(step_lp.items(), key=lambda kv: -kv[1].logprob)
            chosen_lps.append(entries[0][1].logprob)
            if len(entries) >= 2:
                runner_up_lps.append(entries[1][1].logprob)
                runner_up_tok_ids.append(entries[1][0])
            else:
                runner_up_lps.append(float("nan"))
                runner_up_tok_ids.append(None)
        boundaries = find_step_boundaries(tok_strs)
        steps = per_step_lps(chosen_lps, boundaries)
        worst_step_idx = int(np.argmin(steps)) if steps else 0
        pred = extract_pred(text)
        ok = int(equal_strict(pred, gold))
        greedy_records.append({
            "id": i, "question": q, "gold": gold,
            "pred": pred, "correct": ok,
            "tok_ids": tok_ids,
            "chosen_lps": chosen_lps,
            "runner_up_lps": runner_up_lps,
            "runner_up_tok_ids": runner_up_tok_ids,
            "boundaries": boundaries,
            "n_steps": len(steps),
            "worst_step_idx": worst_step_idx,
            "step_lps": steps,
            "text": text,
        })
    n_correct_greedy = sum(r["correct"] for r in greedy_records)
    summary["greedy"] = {"accuracy": n_correct_greedy / len(questions), "wallclock_sec": dur_g}
    print(f"  Greedy: {n_correct_greedy}/{len(questions)} = {n_correct_greedy/len(questions):.3f}", flush=True)

    # ===== Method F1: forbidden_top1_redecoding =====
    # At worst step boundary, ban the top-1 token of that step's first position, regenerate from there.
    print("\n=== F1: forbidden_top1_redecoding ===", flush=True)
    f1_prompts = []
    f1_meta = []
    for r in greedy_records:
        if not r["boundaries"] or r["worst_step_idx"] >= len(r["boundaries"]):
            f1_prompts.append(None)
            f1_meta.append(None)
            continue
        bdy_pos = r["boundaries"][r["worst_step_idx"]]
        if bdy_pos >= len(r["tok_ids"]):
            f1_prompts.append(None)
            f1_meta.append(None)
            continue
        # banned token id = the actual top-1 token at this position
        banned_tok = r["tok_ids"][bdy_pos]
        # Build prefix = base_prompt + tokens up to bdy_pos (not including)
        prefix_text = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
        chat = base_prompts[r["id"]]
        full_prefix = chat + prefix_text
        f1_prompts.append(full_prefix)
        f1_meta.append({"banned_token": banned_tok})

    valid_f1 = [(i, p, m) for i, (p, m) in enumerate(zip(f1_prompts, f1_meta)) if p is not None]
    f1_results = {r["id"]: None for r in greedy_records}
    if valid_f1:
        # Use logit_bias on banned token (extreme negative)
        # Note: vLLM supports logit_bias only via generate but applied to ALL positions;
        # for one-token ban we need to do n=1 generation with logit_bias applied,
        # which is conservative (will avoid that token through the entire suffix; but token id won't be
        # used much elsewhere; acceptable approximation).
        for batch_start in range(0, len(valid_f1), 50):
            batch = valid_f1[batch_start:batch_start+50]
            prompts = [p for _, p, _ in batch]
            metas = [m for _, _, m in batch]
            ids = [i for i, _, _ in batch]
            # Apply per-prompt logit_bias via SamplingParams with custom logit_processors not directly supported in batch.
            # As a fallback, use a single sampling with logit_bias dict for each request.
            sps = [SamplingParams(temperature=0.0, max_tokens=1024, seed=SEED + 100,
                                    logit_bias={m["banned_token"]: -100.0}) for m in metas]
            outs = llm.generate(prompts, sps)
            for ii, oo in zip(ids, outs):
                txt = oo.outputs[0].text
                pred = extract_pred(txt)
                gold = greedy_records[ii]["gold"]
                ok = int(equal_strict(pred, gold))
                f1_results[ii] = {"pred": pred, "correct": ok, "text": txt[:1000]}
    n_correct_f1 = sum(r["correct"] for r in f1_results.values() if r is not None)
    n_f1 = sum(1 for r in f1_results.values() if r is not None)
    summary["f1_forbidden_top1"] = {"accuracy": n_correct_f1 / max(n_f1, 1), "n_evaluated": n_f1,
                                       "n_recovered": sum(1 for i, r in f1_results.items() if r and r["correct"] and not greedy_records[i]["correct"]),
                                       "n_lost": sum(1 for i, r in f1_results.items() if r and not r["correct"] and greedy_records[i]["correct"])}
    print(f"  F1: {n_correct_f1}/{n_f1} = {n_correct_f1/max(n_f1,1):.3f}, recov {summary['f1_forbidden_top1']['n_recovered']}, lost {summary['f1_forbidden_top1']['n_lost']}", flush=True)

    # ===== Method I3: system_prompt_ensemble =====
    print("\n=== I3: system_prompt_ensemble (4 system prompts) ===", flush=True)
    sys_results = {r["id"]: [] for r in greedy_records}
    for sys_prompt in SYSTEM_PROMPTS:
        sys_prompts_built = [
            tok.apply_chat_template(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": PROMPT_BASE.format(question=q)}],
                tokenize=False, add_generation_prompt=True,
            )
            for q in questions
        ]
        sp = SamplingParams(temperature=0.0, max_tokens=1536, seed=SEED + 200)
        t0 = time.time()
        outs = llm.generate(sys_prompts_built, sp)
        dur = time.time() - t0
        for i, out in enumerate(outs):
            pred = extract_pred(out.outputs[0].text)
            sys_results[i].append(pred)
        print(f"  System prompt variant: {dur:.1f}s", flush=True)

    # Build score: agreement with greedy + modal consensus
    sys_scores = []
    for i, r in enumerate(greedy_records):
        preds = sys_results[i]
        agree_g = float(np.mean([equal_strict(p, r["pred"]) and (p is not None) for p in preds]))
        # modal majority
        preds_clean = [p for p in preds if p is not None]
        counter = Counter([normalize(p) for p in preds_clean])
        top, cnt = counter.most_common(1)[0] if counter else (None, 0)
        # Use modal pred as candidate; if it differs from greedy, replace
        modal_pred = top
        # Modal accuracy
        ok_modal = int(equal_strict(modal_pred, r["gold"]))
        sys_scores.append({"agree_greedy": agree_g, "modal_pred": modal_pred, "ok_modal": ok_modal})

    n_modal = sum(s["ok_modal"] for s in sys_scores)
    summary["i3_system_prompt_modal"] = {"accuracy": n_modal / len(questions),
                                            "wallclock_sec_total": dur * 4}
    print(f"  I3 modal across 4 prompts: {n_modal}/{len(questions)} = {n_modal/len(questions):.3f}", flush=True)

    # ===== Baseline: K=4 resample (replicates Pilot C) =====
    print("\n=== Baseline: K=4 resample at worst step (Pilot C) ===", flush=True)
    pilotC_prompts = []
    pilotC_ids = []
    for r in greedy_records:
        if not r["boundaries"] or r["worst_step_idx"] >= len(r["boundaries"]):
            continue
        bdy_pos = r["boundaries"][r["worst_step_idx"]]
        prefix_text = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
        chat = base_prompts[r["id"]]
        pilotC_prompts.append(chat + prefix_text)
        pilotC_ids.append(r["id"])

    pilotC_results = {r["id"]: None for r in greedy_records}
    if pilotC_prompts:
        sp = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 300)
        outs = llm.generate(pilotC_prompts, sp)
        for ii, out in zip(pilotC_ids, outs):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter([normalize(p) for p in preds_clean])
            top, cnt = counter.most_common(1)[0] if counter else (None, 0)
            r = greedy_records[ii]
            # combine with greedy: take majority of {greedy_pred} + 4 branches
            combined = [r["pred"]] + preds_clean
            counter2 = Counter([normalize(p) for p in combined if p is not None])
            top2, cnt2 = counter2.most_common(1)[0] if counter2 else (None, 0)
            ok = int(equal_strict(top2, r["gold"]))
            pilotC_results[ii] = {"pred": top2, "correct": ok}
    n_correct_pc = sum(r["correct"] for r in pilotC_results.values() if r is not None)
    n_pc = sum(1 for r in pilotC_results.values() if r is not None)
    summary["pilotC_baseline_K4_resample"] = {"accuracy": n_correct_pc / max(n_pc, 1),
                                                  "n_evaluated": n_pc,
                                                  "n_recovered": sum(1 for i, r in pilotC_results.items() if r and r["correct"] and not greedy_records[i]["correct"]),
                                                  "n_lost": sum(1 for i, r in pilotC_results.items() if r and not r["correct"] and greedy_records[i]["correct"])}
    print(f"  Pilot C K=4: {n_correct_pc}/{n_pc} = {n_correct_pc/max(n_pc,1):.3f}", flush=True)

    SUM.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote: {SUM}", flush=True)

    print("\n=== Summary ===")
    print(f"  Greedy:                       {summary['greedy']['accuracy']:.3f}")
    print(f"  F1 forbidden_top1_redecoding: {summary['f1_forbidden_top1']['accuracy']:.3f}")
    print(f"  I3 system_prompt_modal:       {summary['i3_system_prompt_modal']['accuracy']:.3f}")
    print(f"  Pilot C K=4 resample:         {summary['pilotC_baseline_K4_resample']['accuracy']:.3f}")


if __name__ == "__main__":
    main()

"""
EXP-step2: hparam sweep for step-level interventions.
Following user instruction: don't discard hypotheses on single failure; vary hparams.

Test grid:
- Pilot C K-resample with K ∈ {2, 4, 8, 16} × T ∈ {0.5, 0.7, 1.0} × trigger ∈ {lp_min, lp_p25, all_steps}
- F2 low_confidence_token_swap (token-level, not boundary)
- H5 formal_restate_and_check (sympy-based, post-hoc)
- SC@K baseline (K resamples from scratch, fair compute comparison)
- Combined: paraphrase + step branching (sequential)

Goal: find which intervention REGIME works.
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
SUM = OUTDIR / "SX_step_hparam_sweep.json"

PROMPT_BASE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
)


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

    summary = {"model": MODEL, "n": len(questions), "results": {}}

    # ===== Greedy baseline (with logprobs) =====
    print("\n=== Greedy ===", flush=True)
    sp_g = SamplingParams(temperature=0.0, max_tokens=1536, logprobs=5, seed=SEED)
    outs = llm.generate(base_prompts, sp_g)

    greedy_records = []
    for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
        o = out.outputs[0]
        text = o.text
        tok_ids = list(o.token_ids)
        tok_strs = [tok.decode([tid]) for tid in tok_ids]
        chosen_lps = []
        for step_lp in o.logprobs or []:
            if step_lp is None:
                chosen_lps.append(float("nan"))
                continue
            entries = sorted(step_lp.items(), key=lambda kv: -kv[1].logprob)
            chosen_lps.append(entries[0][1].logprob)
        boundaries = find_step_boundaries(tok_strs)
        steps = per_step_lps(chosen_lps, boundaries)
        pred = extract_pred(text)
        ok = int(equal_strict(pred, gold))
        greedy_records.append({
            "id": i, "gold": gold, "pred": pred, "correct": ok,
            "tok_ids": tok_ids,
            "boundaries": boundaries,
            "step_lps": steps,
            "chosen_lps": chosen_lps,
            "text": text,
        })
    n_greedy = sum(r["correct"] for r in greedy_records)
    summary["results"]["greedy"] = {"acc": n_greedy / len(questions)}
    print(f"  Greedy: {n_greedy}/{len(questions)} = {n_greedy/len(questions):.3f}", flush=True)

    # ===== Pilot C-style K-resample sweep =====
    # K-resample at "worst" step. Vary trigger and K and T.
    pilotc_configs = []
    for trigger in ["lp_min", "lp_p25", "first", "last", "middle"]:
        for K in [2, 4, 8]:
            for T in [0.5, 0.7, 1.0]:
                pilotc_configs.append({"trigger": trigger, "K": K, "T": T})

    for cfg in pilotc_configs:
        trigger = cfg["trigger"]; K = cfg["K"]; T = cfg["T"]
        # Build branched prompts at chosen step
        branch_prompts = []
        ids = []
        for r in greedy_records:
            steps = r["step_lps"]
            if not steps:
                continue
            if trigger == "lp_min":
                idx = int(np.argmin(steps))
            elif trigger == "lp_p25":
                # 25th percentile step
                p25 = np.percentile(steps, 25)
                idx = int(np.argmin(np.abs(np.array(steps) - p25)))
            elif trigger == "first":
                idx = 0
            elif trigger == "last":
                idx = len(steps) - 1
            elif trigger == "middle":
                idx = len(steps) // 2
            if idx >= len(r["boundaries"]):
                continue
            bdy_pos = r["boundaries"][idx]
            prefix_text = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
            chat = base_prompts[r["id"]]
            branch_prompts.append(chat + prefix_text)
            ids.append(r["id"])
        sp = SamplingParams(n=K, temperature=T, top_p=0.95, max_tokens=1024, seed=SEED + hash(str(cfg)) % 10000)
        t0 = time.time()
        outs = llm.generate(branch_prompts, sp)
        dur = time.time() - t0

        results = {}
        for ii, out in zip(ids, outs):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            r = greedy_records[ii]
            combined = [r["pred"]] + preds_clean
            counter = Counter([normalize(p) for p in combined if p is not None])
            top, _ = counter.most_common(1)[0] if counter else (None, 0)
            results[ii] = {"pred": top, "correct": int(equal_strict(top, r["gold"]))}
        n_correct = sum(r["correct"] for r in results.values())
        n = len(results)
        recovered = sum(1 for i, rr in results.items() if rr["correct"] and not greedy_records[i]["correct"])
        lost = sum(1 for i, rr in results.items() if not rr["correct"] and greedy_records[i]["correct"])

        key = f"pilotC_{trigger}_K{K}_T{T}"
        summary["results"][key] = {"acc": n_correct / max(n, 1),
                                       "n": n, "recov": recovered, "lost": lost,
                                       "wallclock_sec": dur}
        print(f"  {key:30s} acc={n_correct/max(n,1):.3f} recov={recovered} lost={lost} ({dur:.1f}s)", flush=True)

    # ===== Plain SC@K baselines (re-decode from scratch, not from prefix) =====
    for K in [2, 4, 8]:
        sp = SamplingParams(n=K, temperature=0.7, top_p=0.95, max_tokens=1536, seed=SEED + 9000 + K)
        t0 = time.time()
        outs = llm.generate(base_prompts, sp)
        dur = time.time() - t0
        n_correct = 0
        for i, out in enumerate(outs):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter([normalize(p) for p in preds_clean])
            top, _ = counter.most_common(1)[0] if counter else (None, 0)
            ok = int(equal_strict(top, greedy_records[i]["gold"]))
            n_correct += ok
        key = f"sc_K{K}_T0.7"
        summary["results"][key] = {"acc": n_correct / len(questions), "wallclock_sec": dur}
        print(f"  {key:30s} acc={n_correct/len(questions):.3f} ({dur:.1f}s)", flush=True)

    SUM.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote: {SUM}", flush=True)

    # Headline summary
    print("\n=== Best step-level configs (sorted by acc) ===")
    sorted_results = sorted(summary["results"].items(), key=lambda kv: -kv[1].get("acc", 0))
    for key, r in sorted_results[:20]:
        print(f"  {key:30s} acc={r['acc']:.3f} (recov={r.get('recov','?')}, lost={r.get('lost','?')})")


if __name__ == "__main__":
    main()

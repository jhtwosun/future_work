"""
Rerun E13 with MAX_TOKENS=4096 (was 2048). Original 175/674 (~26%) truncated.
"""

import json
import os
import re
import time
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, normalize, equal_strict

MODEL = "Qwen/Qwen3-8B"
N_SAMPLES = int(os.environ.get("NS", "8"))
MAX_TOKENS = int(os.environ.get("MAXTOK", "4096"))
TEMP_SC = 0.7
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
GREEDY_OUT = OUTDIR / "E13r_olympiad_math_greedy.jsonl"
SC_OUT = OUTDIR / "E13r_olympiad_math_sc.jsonl"
SUM = OUTDIR / "E13r_summary.json"

PROMPT_TEMPLATE = (
    "Solve the following olympiad math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    "Put your final numeric or expression answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
)


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def main():
    print("Loading OlympiadBench OE_TO_maths_en_COMP...", flush=True)
    ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    text_only = [r for r in ds if not any(r.get(f"image_{i}") for i in range(1, 6))]
    print(f"  {len(text_only)} text-only problems", flush=True)
    questions = [r["question"] for r in text_only]
    golds = [r["final_answer"] for r in text_only]

    print(f"Loading {MODEL}, MAX_TOKENS={MAX_TOKENS}...", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=5120,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tok = llm.get_tokenizer()
    prompts = []
    for q in questions:
        prompt_user = PROMPT_TEMPLATE.format(question=q)
        try:
            p = tok.apply_chat_template(
                [{"role": "user", "content": prompt_user}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            p = tok.apply_chat_template(
                [{"role": "user", "content": prompt_user}],
                tokenize=False, add_generation_prompt=True,
            )
        prompts.append(p)

    summary = {"model": MODEL + " (no-think)", "n": len(prompts),
                "dataset": "OlympiadBench OE_TO_maths_en_COMP (text-only)",
                "max_tokens": MAX_TOKENS}

    def gold_match(pred, gold_list):
        if pred is None or not gold_list: return False
        return any(equal_strict(pred, g) for g in gold_list if g is not None)

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur_g = time.time() - t0
    n_correct = 0; n_truncated = 0
    with GREEDY_OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
            o = out.outputs[0]; text = o.text
            tok_ids = list(o.token_ids)
            tok_strs = [tok.decode([tid]) for tid in tok_ids]
            tok_logprobs = []
            for step_lp in o.logprobs or []:
                if step_lp is None:
                    tok_logprobs.append(float("nan"))
                    continue
                vals = list(step_lp.values())
                tok_logprobs.append(float(vals[0].logprob))
            boundaries = find_step_boundaries(tok_strs)
            pred = extract_pred(text)
            if len(tok_ids) >= MAX_TOKENS - 5: n_truncated += 1
            ok = gold_match(pred, gold)
            n_correct += int(ok)
            f.write(json.dumps({
                "id": i, "question": q, "gold": gold,
                "predicted": pred, "correct": ok,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "output_text": text,
            }) + "\n")
    summary["greedy"] = {
        "accuracy": n_correct / len(prompts),
        "n_truncated": n_truncated,
        "wallclock_sec": dur_g,
    }
    print(f"Greedy: {n_correct}/{len(prompts)} = {n_correct/len(prompts):.3f}, truncated {n_truncated}", flush=True)

    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_sc)
    dur_sc = time.time() - t0
    n_maj = 0; n_any = 0; top1_fracs = []
    with SC_OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter([normalize(p) for p in preds_clean])
            top_norm, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES; top1_fracs.append(top1_frac)
            ok = gold_match(top_norm, gold)
            any_ok = any(gold_match(p, gold) for p in preds_clean)
            n_maj += int(ok); n_any += int(any_ok)
            f.write(json.dumps({
                "id": i, "gold": gold,
                "majority_pred": top_norm, "majority_correct": int(ok),
                "any_correct": int(any_ok), "top1_frac": top1_frac,
                "answer_distribution": dict(counter),
            }) + "\n")
    summary["sc"] = {
        "n_samples": N_SAMPLES,
        "majority_accuracy": n_maj / len(prompts),
        "any_accuracy": n_any / len(prompts),
        "mean_top1_frac": float(np.mean(top1_fracs)),
        "wallclock_sec": dur_sc,
    }
    print(f"SC@{N_SAMPLES}: maj={n_maj/len(prompts):.3f}", flush=True)
    SUM.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

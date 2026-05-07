"""
Experiment E16: Qwen2.5-Math-7B-Instruct on MATH-500 (greedy + SC@8).

Math-specialized 7B from the Qwen2.5-Math family. Published ~83% on
MATH-500 — should anchor a SOTA-near baseline. With CP+SC, target 95%+.
"""

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
N_SAMPLES = int(os.environ.get("NS", "8"))
MAX_TOKENS = 2048
TEMP_SC = 0.7
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
GREEDY_OUT = OUTDIR / "E16_qwen_math_7b_greedy.jsonl"
SC_OUT = OUTDIR / "E16_qwen_math_7b_sc.jsonl"
SUM = OUTDIR / "E16_summary.json"

# Qwen2.5-Math uses CoT prompt format with \boxed{} for final answer
PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "{question}"
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
    print("Loading MATH-500...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    questions = [r["problem"] for r in ds]
    golds = [r["answer"] for r in ds]
    print(f"  {len(questions)} problems", flush=True)

    print(f"Loading {MODEL}...", flush=True)
    t_load = time.time()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        tensor_parallel_size=1,
        seed=SEED,
    )
    print(f"  loaded in {time.time()-t_load:.1f}s", flush=True)
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]
    summary = {"model": MODEL, "n": len(prompts), "dataset": "MATH-500"}

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur_g = time.time() - t0
    n_tokens_all = []; n_truncated = 0
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
            n_tokens_all.append(len(tok_ids))
            if len(tok_ids) >= MAX_TOKENS - 5:
                n_truncated += 1
            f.write(json.dumps({
                "id": i, "question": q, "gold": gold,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "output_text": text,
            }) + "\n")
    summary["greedy"] = {"wallclock_sec": dur_g, "n_truncated": n_truncated,
                          "mean_tokens": float(np.mean(n_tokens_all)),
                          "median_tokens": float(np.median(n_tokens_all))}
    print(f"Greedy done in {dur_g:.1f}s, mean tokens={np.mean(n_tokens_all):.0f}, truncated={n_truncated}", flush=True)

    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_sc)
    dur_sc = time.time() - t0
    with SC_OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
            samples = [c.text for c in out.outputs]
            f.write(json.dumps({
                "id": i, "question": q, "gold": gold,
                "samples": samples,
            }) + "\n")
    summary["sc"] = {"n_samples": N_SAMPLES, "wallclock_sec": dur_sc}
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""
Experiment E10: Mixtral-8x7B-Instruct-v0.1 on MATH-500 (TP=2).

Classic MoE 46B (8 experts × 7B). Cross-family MoE comparison.
Uses tensor parallelism across 2 H100s.
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

MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
N_SAMPLES = int(os.environ.get("NS", "8"))
MAX_TOKENS = 1536
TEMP_SC = 0.7
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUTDIR.mkdir(parents=True, exist_ok=True)
GREEDY_OUT = OUTDIR / "E10_mixtral_greedy.jsonl"
SC_OUT = OUTDIR / "E10_mixtral_sc.jsonl"
SUM = OUTDIR / "E10_summary.json"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'Put your final answer within \\boxed{{}}.\n\n'
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
    print("Loading MATH-500...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    questions = [r["problem"] for r in ds]
    golds = [r["answer"] for r in ds]

    print(f"Loading {MODEL} (Mixtral 46B MoE, TP=2)...", flush=True)
    t_load = time.time()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2560,
        tensor_parallel_size=2,
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
            f.write(json.dumps({
                "id": i, "question": q, "gold": gold,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "output_text": text,
            }) + "\n")
    summary["greedy"] = {"wallclock_sec": dur_g}

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

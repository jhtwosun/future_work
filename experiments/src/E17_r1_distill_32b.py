"""
Experiment E17: DeepSeek-R1-Distill-Qwen-32B on MATH-500.

Larger R1-Distill (~64GB bf16). Long-CoT reasoning model.
Greedy on full 500, SC@4 on 200 (long-CoT is expensive).
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

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
N_SAMPLES = int(os.environ.get("NS", "4"))
N_SC = int(os.environ.get("N_SC_PROBLEMS", "200"))
MAX_TOKENS = int(os.environ.get("MAXTOK", "8192"))
TEMP_SC = 0.7
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
GREEDY_OUT = OUTDIR / "E17_r1_distill_32b_greedy.jsonl"
SC_OUT = OUTDIR / "E17_r1_distill_32b_sc.jsonl"
SUM = OUTDIR / "E17_summary.json"

PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
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

    print(f"Loading {MODEL} (32B, long-CoT)...", flush=True)
    t_load = time.time()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        max_model_len=12288,
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

    print("=== Greedy with logprobs ===", flush=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur_g = time.time() - t0
    n_tokens_all = []; n_truncated = 0
    n_steps_all = []
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
            n_steps_all.append(len(boundaries))
            if len(tok_ids) >= MAX_TOKENS - 5: n_truncated += 1
            f.write(json.dumps({
                "id": i, "question": q, "gold": gold,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "output_text": text,
            }) + "\n")
    summary["greedy"] = {
        "wallclock_sec": dur_g,
        "mean_tokens": float(np.mean(n_tokens_all)),
        "median_tokens": float(np.median(n_tokens_all)),
        "max_tokens": int(np.max(n_tokens_all)),
        "n_truncated": n_truncated,
        "mean_steps": float(np.mean(n_steps_all)),
    }
    print(f"Greedy: {dur_g:.1f}s, mean tokens={np.mean(n_tokens_all):.0f}, mean steps={np.mean(n_steps_all):.0f}, truncated={n_truncated}", flush=True)

    # SC on subset to fit time budget
    print(f"=== SC@{N_SAMPLES} on first {N_SC} problems ===", flush=True)
    sub_prompts = prompts[:N_SC]
    sub_questions = questions[:N_SC]
    sub_golds = golds[:N_SC]
    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(sub_prompts, sp_sc)
    dur_sc = time.time() - t0
    with SC_OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(sub_questions, sub_golds, outs)):
            samples = [c.text for c in out.outputs]
            f.write(json.dumps({
                "id": i, "question": q, "gold": gold,
                "samples": samples,
            }) + "\n")
    summary["sc"] = {"n_samples": N_SAMPLES, "n_problems": N_SC, "wallclock_sec": dur_sc}
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

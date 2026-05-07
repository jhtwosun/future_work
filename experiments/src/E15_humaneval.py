"""
Experiment E15: HumanEval (164 problems) on Qwen3-8B (no-think).
Code generation with test execution. Different output modality.
"""

import json
import os
import re
import time
import signal
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-8B"
N_SAMPLES = int(os.environ.get("NS", "8"))
MAX_TOKENS = 1024
TEMP_SC = 0.7
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
GREEDY_OUT = OUTDIR / "E15_humaneval_greedy.jsonl"
SC_OUT = OUTDIR / "E15_humaneval_sc.jsonl"
SUM = OUTDIR / "E15_summary.json"

PROMPT_TEMPLATE = (
    "Complete the following Python function. Reason step by step about the implementation. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    "After all reasoning, output ONLY the complete function definition (def ...) wrapped in a "
    "```python``` code block.\n\n"
    "{prompt}"
)


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


CODE_RE = re.compile(r"```python\n(.+?)```", re.DOTALL)
DEF_RE = re.compile(r"(?:^|\n)(def\s+\w+\([^)]*\)[^\n]*:[\s\S]*?)(?=\n(?:def|class|\Z|```))", re.MULTILINE)


def extract_code(text, entry_point):
    """Extract the function with name `entry_point` from generated text."""
    # Try ```python ... ``` first
    blocks = CODE_RE.findall(text)
    for b in reversed(blocks):
        if f"def {entry_point}" in b:
            return b.strip()
    # Fallback: any code block
    if blocks:
        return blocks[-1].strip()
    # Fallback: search for def directly
    matches = DEF_RE.findall(text)
    for m in matches:
        if f"def {entry_point}" in m:
            return m.strip()
    if matches:
        return matches[-1].strip()
    return None


def run_test(code, test_code, entry_point, timeout=10):
    """Run code + test in subprocess. Returns True if all tests pass."""
    if code is None:
        return False
    full = code + "\n" + test_code + f"\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full)
            fname = f.name
        try:
            r = subprocess.run(["python3", fname], timeout=timeout,
                                capture_output=True, text=True)
            return r.returncode == 0
        finally:
            os.unlink(fname)
    except (subprocess.TimeoutExpired, Exception):
        return False


def main():
    print("Loading HumanEval...", flush=True)
    ds = load_dataset("openai/openai_humaneval", split="test")
    print(f"  {len(ds)} problems", flush=True)

    print(f"Loading {MODEL}...", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tok = llm.get_tokenizer()
    prompts = []
    for r in ds:
        prompt_user = PROMPT_TEMPLATE.format(prompt=r["prompt"])
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

    summary = {"model": MODEL + " (no-think)", "n": len(prompts), "dataset": "HumanEval"}

    # Greedy
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur_g = time.time() - t0
    n_correct = 0
    with GREEDY_OUT.open("w") as f:
        for i, (r, out) in enumerate(zip(ds, outs)):
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
            code = extract_code(text, r["entry_point"])
            ok = run_test(code, r["test"], r["entry_point"])
            n_correct += int(ok)
            f.write(json.dumps({
                "id": i, "task_id": r["task_id"], "entry_point": r["entry_point"],
                "predicted_code": code, "correct": ok,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "output_text": text,
            }) + "\n")
    summary["greedy"] = {"accuracy": n_correct / len(prompts), "wallclock_sec": dur_g}
    print(f"HumanEval greedy pass@1: {n_correct}/{len(prompts)} = {n_correct/len(prompts):.3f}", flush=True)

    # SC@8: pass@8 (any sample passes) + majority code by AST string
    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_sc)
    dur_sc = time.time() - t0
    n_any = 0; n_maj = 0; top1_fracs = []
    with SC_OUT.open("w") as f:
        for i, (r, out) in enumerate(zip(ds, outs)):
            samples = [c.text for c in out.outputs]
            codes = [extract_code(s, r["entry_point"]) for s in samples]
            results = [run_test(c, r["test"], r["entry_point"]) for c in codes]
            any_ok = any(results)
            n_any += int(any_ok)
            # Majority: most-common code (by exact string)
            valid_codes = [c for c in codes if c is not None]
            counter = Counter(valid_codes)
            top, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES; top1_fracs.append(top1_frac)
            maj_ok = run_test(top, r["test"], r["entry_point"]) if top else False
            n_maj += int(maj_ok)
            f.write(json.dumps({
                "id": i, "task_id": r["task_id"],
                "any_correct": int(any_ok), "majority_correct": int(maj_ok),
                "individual_correct": [int(b) for b in results],
                "top1_frac": top1_frac,
            }) + "\n")
    summary["sc"] = {
        "n_samples": N_SAMPLES,
        "pass_at_n": n_any / len(prompts),
        "majority_correct": n_maj / len(prompts),
        "mean_top1_frac": float(np.mean(top1_fracs)),
        "wallclock_sec": dur_sc,
    }
    print(f"HumanEval pass@{N_SAMPLES}: {n_any/len(prompts):.3f}, majority: {n_maj/len(prompts):.3f}", flush=True)
    SUM.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

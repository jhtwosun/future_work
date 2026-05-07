"""
Experiment E14: HumanEval (164 Python coding problems).

Different modality: code generation. Tests if CoT-CP generalizes
beyond math to code. Greedy + SC@8. Pass@1 evaluation via
sandboxed test execution.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_SAMPLES = int(os.environ.get("NS", "8"))
MAX_TOKENS = 1024
TEMP_SC = 0.7
SEED = 0
EXEC_TIMEOUT = 10  # seconds per test execution

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUTDIR.mkdir(parents=True, exist_ok=True)
GREEDY_OUT = OUTDIR / "E14_humaneval_greedy.jsonl"
SC_OUT = OUTDIR / "E14_humaneval_sc.jsonl"
SUM = OUTDIR / "E14_summary.json"

PROMPT_TEMPLATE = (
    "Complete the following Python function. Reason step by step about your approach in a comment, "
    "then provide the implementation. Wrap the final implementation in a Python code block (```python ... ```).\n\n"
    "{prompt}\n"
)
CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
CODE_BLOCK_GENERIC = re.compile(r"```\n?(.*?)```", re.DOTALL)


def extract_code(text):
    matches = list(CODE_BLOCK.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    matches = list(CODE_BLOCK_GENERIC.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    # Fallback: look for "def ..." block
    m = re.search(r"(def \w+.*?)(?=\n```|\Z)", text, re.DOTALL)
    if m: return m.group(1).strip()
    return None


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def run_test(prompt: str, code: str, test: str, entry_point: str) -> tuple[bool, str]:
    """Execute code + tests in subprocess. Return (passed, error_msg)."""
    if not code:
        return False, "no code"
    # Build the test program
    program = f"""
import sys
{prompt}
{code}
{test}
check({entry_point})
print("OK")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, timeout=EXEC_TIMEOUT
        )
        if result.returncode == 0 and "OK" in result.stdout:
            return True, ""
        return False, (result.stderr or result.stdout)[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:200]


def main():
    print("Loading HumanEval...", flush=True)
    ds = load_dataset("openai/openai_humaneval", split="test")
    print(f"  {len(ds)} problems", flush=True)
    prompts_field = [r["prompt"] for r in ds]
    tests = [r["test"] for r in ds]
    entry_points = [r["entry_point"] for r in ds]

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
    chat_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=p)}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts_field
    ]
    summary = {"model": MODEL, "n": len(chat_prompts), "dataset": "HumanEval"}

    print("=== Greedy ===", flush=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(chat_prompts, sp)
    dur_g = time.time() - t0
    n_pass = 0
    with GREEDY_OUT.open("w") as f:
        for i, (p, test, ep, out) in enumerate(zip(prompts_field, tests, entry_points, outs)):
            o = out.outputs[0]; text = o.text
            code = extract_code(text)
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
            passed, err = run_test(p, code, test, ep)
            n_pass += int(passed)
            f.write(json.dumps({
                "id": i, "prompt": p[:500], "entry_point": ep,
                "extracted_code": code,
                "passed": passed, "error": err if not passed else "",
                "correct": int(passed),
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "output_text": text,
            }) + "\n")
    summary["greedy"] = {"pass_rate": n_pass / len(chat_prompts), "wallclock_sec": dur_g}
    print(f"Greedy pass@1: {n_pass}/{len(chat_prompts)} = {n_pass/len(chat_prompts):.3f}", flush=True)

    print(f"=== SC@{N_SAMPLES} ===", flush=True)
    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(chat_prompts, sp_sc)
    dur_sc = time.time() - t0
    n_maj_pass = 0; n_any_pass = 0; top1_fracs = []
    with SC_OUT.open("w") as f:
        for i, (p, test, ep, out) in enumerate(zip(prompts_field, tests, entry_points, outs)):
            samples = [c.text for c in out.outputs]
            codes = [extract_code(s) for s in samples]
            results = [run_test(p, c, test, ep) for c in codes]
            pass_flags = [r[0] for r in results]
            # Majority by code (string equality after strip)
            valid_codes = [c for c in codes if c]
            counter = Counter(valid_codes)
            top_code, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES; top1_fracs.append(top1_frac)
            # Majority pass = pass if the most-common code passes
            top_idx = next((j for j, c in enumerate(codes) if c == top_code), 0) if top_code else 0
            maj_pass = pass_flags[top_idx] if top_code else False
            any_pass = any(pass_flags)
            n_maj_pass += int(maj_pass); n_any_pass += int(any_pass)
            f.write(json.dumps({
                "id": i, "entry_point": ep,
                "majority_code": top_code,
                "majority_passed": int(maj_pass),
                "any_passed": int(any_pass),
                "top1_frac": top1_frac,
                "n_unique_codes": len(counter),
                "individual_passed": [int(p) for p in pass_flags],
            }) + "\n")
    summary["sc"] = {"n_samples": N_SAMPLES,
                       "majority_pass_rate": n_maj_pass / len(chat_prompts),
                       "any_pass_rate": n_any_pass / len(chat_prompts),
                       "mean_top1_frac": float(np.mean(top1_fracs)),
                       "wallclock_sec": dur_sc}
    print(f"SC@{N_SAMPLES}: maj_pass={n_maj_pass/len(chat_prompts):.3f}  any_pass={n_any_pass/len(chat_prompts):.3f}", flush=True)
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

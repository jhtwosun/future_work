"""
Experiment E2: Full AIME 1983-2024 (all 933 problems) with greedy + SC@8.

Pilot I had 300 problems. This run takes the full 933 to give a clean
OOD result with proper bootstrap CIs.

Output:
  results/E2_aime_greedy_traces.jsonl
  results/E2_aime_sc8_traces.jsonl
  results/E2_summary.json
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

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_SAMPLES = int(os.environ.get("NS", "8"))
MAX_TOKENS = 2048
TEMP_SC = 0.7
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUTDIR.mkdir(parents=True, exist_ok=True)
GREEDY_OUT = OUTDIR / "E2_aime_greedy_traces.jsonl"
SC_OUT = OUTDIR / "E2_aime_sc8_traces.jsonl"
SUM = OUTDIR / "E2_summary.json"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number>". '
    "AIME answers are integers from 0 to 999.\n\n"
    "Problem: {question}\n\n"
)
PRED_RE = re.compile(r"(?i)answer\s*[:=]\s*([^\n]+)")
NUM_RE = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")


def normalize(s):
    if s is None: return None
    s = str(s).strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.rstrip(".,;:")
    s = re.sub(r"\s+", "", s)
    m = re.match(r"\\boxed\{(.+)\}$", s)
    if m: s = m.group(1)
    return s


def extract_pred(text):
    m = PRED_RE.search(text)
    if m: return normalize(m.group(1))
    nums = NUM_RE.findall(text)
    return normalize(nums[-1]) if nums else None


def equal(a, b):
    if a is None or b is None: return False
    if a == b: return True
    try:
        af = float(a); bf = float(b)
        return abs(af - bf) < 1e-4
    except Exception:
        return False


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def main():
    print("Loading AIME 1983-2024...", flush=True)
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    questions = [r["Question"] for r in ds]
    golds = [normalize(r["Answer"]) for r in ds]
    years = [r["Year"] for r in ds]
    print(f"  {len(questions)} problems (years {min(years)}-{max(years)})", flush=True)

    print(f"Loading {MODEL}...", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]

    summary = {"model": MODEL, "n": len(prompts), "dataset": "AIME 1983-2024",
                "year_range": [min(years), max(years)]}

    # Greedy
    print("=== Greedy with logprobs ===", flush=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur_g = time.time() - t0
    n_correct = 0
    n_parse_fail = 0
    with GREEDY_OUT.open("w") as f:
        for i, (q, gold, year, out) in enumerate(zip(questions, golds, years, outs)):
            o = out.outputs[0]
            text = o.text
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
            if pred is None: n_parse_fail += 1
            ok = equal(pred, gold)
            n_correct += int(ok)
            f.write(json.dumps({
                "id": i, "year": year, "question": q, "gold": gold,
                "predicted": pred, "correct": ok,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "output_text": text,
            }) + "\n")
    summary["greedy"] = {
        "accuracy": n_correct / len(prompts),
        "parse_failures": n_parse_fail,
        "wallclock_sec": dur_g,
    }
    print(f"AIME greedy: {n_correct}/{len(prompts)} = {n_correct/len(prompts):.3f}", flush=True)

    # SC@N
    print(f"=== SC@{N_SAMPLES} ===", flush=True)
    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_sc)
    dur_sc = time.time() - t0
    n_maj = 0; n_any = 0
    top1_fracs = []
    with SC_OUT.open("w") as f:
        for i, (q, gold, year, out) in enumerate(zip(questions, golds, years, outs)):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter(preds_clean)
            top, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES
            top1_fracs.append(top1_frac)
            ok = equal(top, gold)
            any_ok = any(equal(p, gold) for p in preds_clean)
            n_maj += int(ok); n_any += int(any_ok)
            f.write(json.dumps({
                "id": i, "year": year,
                "majority_correct": int(ok), "any_correct": int(any_ok),
                "top1_frac": top1_frac, "majority_pred": top, "gold": gold,
                "answer_distribution": dict(counter),
            }) + "\n")
    summary["sc"] = {
        "n_samples": N_SAMPLES,
        "majority_accuracy": n_maj / len(prompts),
        "oracle_any": n_any / len(prompts),
        "mean_top1_frac": float(np.mean(top1_fracs)),
        "wallclock_sec": dur_sc,
    }
    print(f"AIME SC@{N_SAMPLES}: maj={n_maj}/{len(prompts)} ({n_maj/len(prompts):.3f}) any={n_any/len(prompts):.3f}", flush=True)
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""
Experiment E3: Full GSM8K (all 1319 test problems) greedy + SC@8.

Pilot 2 had 300 problems, Pilot 4 had 150 SC@8. This extends to full
1319-question test set for clean comparison numbers.
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
MAX_TOKENS = 1024
TEMP_SC = 0.7
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUTDIR.mkdir(parents=True, exist_ok=True)
GREEDY_OUT = OUTDIR / "E3_gsm8k_greedy_traces.jsonl"
SC_OUT = OUTDIR / "E3_gsm8k_sc8_traces.jsonl"
SUM = OUTDIR / "E3_summary.json"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number>".\n\n'
    "Problem: {question}\n\n"
)
GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
PRED_RE = re.compile(r"(?i)answer\s*[:=]\s*(-?[\d,]+(?:\.\d+)?)")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_gold(answer_field):
    m = GOLD_RE.search(answer_field)
    return m.group(1).replace(",", "") if m else None


def extract_pred(text):
    m = PRED_RE.search(text)
    if m: return m.group(1).replace(",", "")
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def numeric_equal(a, b):
    if a is None or b is None: return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
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
    print("Loading GSM8K test...", flush=True)
    ds = load_dataset("gsm8k", "main", split="test")
    questions = [r["question"] for r in ds]
    golds = [parse_gold(r["answer"]) for r in ds]
    print(f"  {len(questions)} problems", flush=True)

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
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]

    summary = {"model": MODEL, "n": len(prompts), "dataset": "GSM8K"}

    # Greedy
    print("=== Greedy with logprobs ===", flush=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur_g = time.time() - t0
    n_correct = 0
    with GREEDY_OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
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
            ok = numeric_equal(pred, gold)
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
        "wallclock_sec": dur_g,
    }
    print(f"Greedy: {n_correct}/{len(prompts)} = {n_correct/len(prompts):.3f}", flush=True)

    # SC@8
    print(f"=== SC@{N_SAMPLES} ===", flush=True)
    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_sc)
    dur_sc = time.time() - t0
    n_maj = 0; n_any = 0
    top1_fracs = []
    with SC_OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter(preds_clean)
            top, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES
            top1_fracs.append(top1_frac)
            ok = numeric_equal(top, gold)
            any_ok = any(numeric_equal(p, gold) for p in preds_clean)
            n_maj += int(ok); n_any += int(any_ok)
            f.write(json.dumps({
                "id": i, "question": q, "gold": gold,
                "majority_pred": top, "majority_correct": int(ok),
                "any_correct": int(any_ok), "top1_frac": top1_frac,
                "answer_distribution": dict(counter),
            }) + "\n")
    summary["sc"] = {
        "n_samples": N_SAMPLES,
        "majority_accuracy": n_maj / len(prompts),
        "oracle_any": n_any / len(prompts),
        "mean_top1_frac": float(np.mean(top1_fracs)),
        "wallclock_sec": dur_sc,
    }
    print(f"SC@{N_SAMPLES}: maj={n_maj}/{len(prompts)} ({n_maj/len(prompts):.3f})", flush=True)
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""
Pilot 4: Self-consistency baseline (N=8) on a subset of GSM8K.

Goal: anchor the SC baseline accuracy and record the per-question top-1
fraction (which is a candidate prompt-level conformity score S-SC).

Output: results/pilot4_sc_traces.jsonl, results/pilot4_summary.json
"""

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_QUESTIONS = int(os.environ.get("N_QUESTIONS_SC", "150"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
MAX_TOKENS = 1024
TEMP = 0.7
SEED = 0
OUT = Path("/home/nvidia/future/pilots/cot_cp/results/pilot4_sc_traces.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number>".\n\n'
    "Problem: {question}\n\n"
)

GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
PRED_RE = re.compile(r"(?i)answer\s*[:=]\s*(-?[\d,]+(?:\.\d+)?)")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_gold(answer_field: str) -> str | None:
    m = GOLD_RE.search(answer_field)
    return m.group(1).replace(",", "") if m else None


def parse_pred(text: str) -> str | None:
    m = PRED_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def numeric_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def main() -> None:
    ds = load_dataset("gsm8k", "main", split="test")
    questions = ds.select(range(min(N_QUESTIONS, len(ds))))
    golds = [parse_gold(q["answer"]) for q in questions]
    print(f"Loaded {len(questions)} GSM8K questions, N_SAMPLES={N_SAMPLES}")

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tokenizer = llm.get_tokenizer()

    sp = SamplingParams(
        n=N_SAMPLES,
        temperature=TEMP,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        seed=SEED,
    )
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q["question"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    dur = time.time() - t0
    print(f"Generation took {dur:.1f}s ({dur/len(prompts):.2f}s/q for n={N_SAMPLES})")

    n_correct_majority = 0
    n_correct_any = 0
    sc_top1_fracs, mc_correct_corr_records = [], []
    with OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outputs)):
            preds = [parse_pred(c.text) for c in out.outputs]
            preds = [p for p in preds if p is not None]
            counter = Counter(preds)
            if counter:
                top, top_count = counter.most_common(1)[0]
            else:
                top, top_count = None, 0
            top1_frac = top_count / N_SAMPLES if N_SAMPLES else 0.0
            sc_top1_fracs.append(top1_frac)
            ok = numeric_equal(top, gold)
            n_correct_majority += int(ok)
            any_ok = any(numeric_equal(p, gold) for p in preds)
            n_correct_any += int(any_ok)
            mc_correct_corr_records.append({
                "id": i,
                "majority_correct": int(ok),
                "any_correct": int(any_ok),
                "top1_frac": top1_frac,
                "answer_distribution": dict(counter),
                "gold": gold,
                "majority_pred": top,
                "n_parsed": len(preds),
            })
            f.write(json.dumps({
                "id": i,
                "question": q["question"],
                "gold": gold,
                "majority_pred": top,
                "majority_correct": int(ok),
                "any_correct": int(any_ok),
                "top1_frac": top1_frac,
                "answer_distribution": dict(counter),
                "samples": [c.text for c in out.outputs],
            }) + "\n")

    summary = {
        "model": MODEL,
        "n": len(prompts),
        "n_samples_per_q": N_SAMPLES,
        "temperature": TEMP,
        "majority_accuracy": n_correct_majority / len(prompts),
        "oracle_any_correct": n_correct_any / len(prompts),
        "mean_sc_top1_frac": sum(sc_top1_fracs) / len(sc_top1_fracs),
        "wallclock_sec": dur,
        "sec_per_question": dur / len(prompts),
    }
    (OUT.parent / "pilot4_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Pilot 9: Self-consistency N=8 on MATH-500 to see SC lift in mid-accuracy regime."""

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N = int(os.environ.get("N", "200"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
MAX_TOKENS = 1536
TEMP = 0.7
SEED = 0
OUT = Path("/home/nvidia/future/pilots/cot_cp/results/pilot9_math500_sc_traces.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>".\n\n'
    "Problem: {question}\n\n"
)
PRED_RE = re.compile(r"(?i)answer\s*[:=]\s*([^\n]+)")
NUM_RE = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")


def normalize(s):
    if s is None:
        return None
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.rstrip(".,;:")
    s = re.sub(r"\s+", "", s)
    m = re.match(r"\\boxed\{(.+)\}$", s)
    if m:
        s = m.group(1)
    return s


def extract_pred(text):
    m = PRED_RE.search(text)
    if m:
        return normalize(m.group(1))
    nums = NUM_RE.findall(text)
    return normalize(nums[-1]) if nums else None


def equal(a, b):
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        af = float(a) if "/" not in a else float(a.split("/")[0]) / float(a.split("/")[1])
        bf = float(b) if "/" not in b else float(b.split("/")[0]) / float(b.split("/")[1])
        return abs(af - bf) < 1e-4
    except Exception:
        pass
    return False


def main():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    use = ds.select(range(min(N, len(ds))))
    questions = [r["problem"] for r in use]
    golds = [normalize(r["answer"]) for r in use]
    print(f"{len(questions)} MATH-500 problems, N_SAMPLES={N_SAMPLES}")

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2560,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tokenizer = llm.get_tokenizer()

    sp = SamplingParams(n=N_SAMPLES, temperature=TEMP, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur = time.time() - t0
    print(f"done in {dur:.1f}s ({dur/len(prompts):.2f}s/q)")

    n_maj_correct = 0
    n_any_correct = 0
    top1_fracs = []
    with OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter(preds_clean)
            top, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES
            top1_fracs.append(top1_frac)
            ok = equal(top, gold)
            any_ok = any(equal(p, gold) for p in preds_clean)
            n_maj_correct += int(ok)
            n_any_correct += int(any_ok)
            f.write(json.dumps({
                "id": i,
                "question": q,
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
        "dataset": "MATH-500",
        "n": len(prompts),
        "n_samples": N_SAMPLES,
        "majority_accuracy": n_maj_correct / len(prompts),
        "oracle_any_correct": n_any_correct / len(prompts),
        "mean_top1_frac": sum(top1_fracs) / len(top1_fracs),
        "wallclock_sec": dur,
    }
    (OUT.parent / "pilot9_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

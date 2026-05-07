"""
Pilot G2: small-scale R1-Distill SC@4 to fill the R1 SC row.

100 questions × N=4 samples instead of 200 × 8 to fit within the
time budget. Uses MATH-500 problems aligned with Pilot 7 / E ids
for direct comparison.
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

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
N = int(os.environ.get("N", "100"))
N_SAMPLES = int(os.environ.get("NS", "4"))
MAX_TOKENS = int(os.environ.get("MAXTOK", "4096"))
TEMP = 0.7
SEED = 0

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")
OUT = RESULTS / "pilotG2_r1_sc_small.jsonl"
SUM = RESULTS / "pilotG2_summary.json"

PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "{question}"
)
PRED_BOX = re.compile(r"\\boxed\{([^{}]+)\}")
PRED_ANS = re.compile(r"(?i)answer\s*[:=]\s*([^\n]+)")
NUM_RE = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")


def normalize(s):
    if s is None:
        return None
    s = str(s).strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.rstrip(".,;:")
    s = re.sub(r"\s+", "", s)
    m = re.match(r"\\boxed\{(.+)\}$", s)
    if m:
        s = m.group(1)
    return s


def extract_pred(text):
    matches = list(PRED_BOX.finditer(text))
    if matches:
        return normalize(matches[-1].group(1))
    m = PRED_ANS.search(text)
    if m:
        return normalize(m.group(1))
    nums = NUM_RE.findall(text)
    return normalize(nums[-1]) if nums else None


def equal(a, b):
    if a is None or b is None: return False
    if a == b: return True
    try:
        af = float(a) if "/" not in a else float(a.split("/")[0]) / float(a.split("/")[1])
        bf = float(b) if "/" not in b else float(b.split("/")[0]) / float(b.split("/")[1])
        return abs(af - bf) < 1e-4
    except Exception:
        return False


def main():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    use = ds.select(range(min(N, len(ds))))
    questions = [r["problem"] for r in use]
    golds = [normalize(r["answer"]) for r in use]
    print(f"{len(questions)} MATH-500 problems × N={N_SAMPLES}", flush=True)

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
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

    sp = SamplingParams(n=N_SAMPLES, temperature=TEMP, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur = time.time() - t0
    print(f"done in {dur:.1f}s ({dur/len(prompts):.2f}s/q for n={N_SAMPLES})", flush=True)

    n_maj = 0; n_any = 0
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
            n_maj += int(ok); n_any += int(any_ok)
            f.write(json.dumps({
                "id": i, "question": q, "gold": gold,
                "majority_pred": top, "majority_correct": int(ok),
                "any_correct": int(any_ok), "top1_frac": top1_frac,
                "answer_distribution": dict(counter),
            }) + "\n")

    summary = {
        "model": MODEL, "dataset": "MATH-500",
        "n": len(prompts), "n_samples": N_SAMPLES,
        "majority_accuracy": n_maj / len(prompts),
        "oracle_any": n_any / len(prompts),
        "mean_top1_frac": float(np.mean(top1_fracs)),
        "wallclock_sec": dur,
    }
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

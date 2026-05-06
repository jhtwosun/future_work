"""
Pilot 7: MATH-500 with Qwen2.5-7B-Instruct + step log-probs.

Motivation: GSM8K saturates at 90% accuracy — too little variance to see
whether step-level log-prob meaningfully predicts final correctness.
MATH-500 is harder (~60-70% expected accuracy), so the calibration
signal should have more headroom.

Reuses the trace format of Pilot 2; downstream Pilot 3 analysis script
can be re-pointed at this file.

Output: results/pilot7_math500_traces.jsonl
"""

import json
import os
import re
import time
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N = int(os.environ.get("N_MATH500", "500"))
MAX_TOKENS = 1536
SEED = 0
OUT = Path("/home/nvidia/future/pilots/cot_cp/results/pilot7_math500_traces.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>".\n\n'
    "Problem: {question}\n\n"
)

# MATH-500 has more complex answer formats; use a permissive matcher
PRED_RE = re.compile(r"(?i)answer\s*[:=]\s*([^\n]+)")
NUM_RE = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")


def normalize(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    # strip $...$ wrappers
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # strip trailing punctuation
    s = s.rstrip(".,;:")
    # collapse whitespace
    s = re.sub(r"\s+", "", s)
    # remove \boxed{}
    m = re.match(r"\\boxed\{(.+)\}$", s)
    if m:
        s = m.group(1)
    # remove leading + and trailing zeros after decimal
    return s


def extract_pred(text: str) -> str | None:
    m = PRED_RE.search(text)
    if m:
        return normalize(m.group(1))
    # fallback: last number
    nums = NUM_RE.findall(text)
    return normalize(nums[-1]) if nums else None


def equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    # try numeric
    try:
        af = float(eval(a)) if "/" not in a else float(a.split("/")[0]) / float(a.split("/")[1])
        bf = float(eval(b)) if "/" not in b else float(b.split("/")[0]) / float(b.split("/")[1])
        return abs(af - bf) < 1e-4
    except Exception:
        pass
    return False


def find_step_boundaries(token_strs: list[str]) -> list[int]:
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def main() -> None:
    print("Loading MATH-500...")
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception as e:
        print(f"HuggingFaceH4/MATH-500 failed: {e}")
        ds = load_dataset("hendrycks/competition_math", split="test")
    print(f"  {len(ds)} problems; columns={ds.column_names}")
    use = ds.select(range(min(N, len(ds))))

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2560,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tokenizer = llm.get_tokenizer()

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
        logprobs=1,
        seed=SEED,
    )

    questions = [r["problem"] if "problem" in r else r["question"] for r in use]
    answers = [r.get("answer") if "answer" in r else r.get("solution") for r in use]
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
    print(f"Done in {dur:.1f}s ({dur/len(prompts):.2f}s/q)")

    n_correct = 0
    n_parse_fail = 0
    with OUT.open("w") as f:
        for i, (q, a, out) in enumerate(zip(questions, answers, outs)):
            o = out.outputs[0]
            text = o.text
            tok_ids = list(o.token_ids)
            tok_strs = [tokenizer.decode([tid]) for tid in tok_ids]
            tok_logprobs = []
            for step_lp in o.logprobs or []:
                if step_lp is None:
                    tok_logprobs.append(float("nan"))
                    continue
                vals = list(step_lp.values())
                tok_logprobs.append(float(vals[0].logprob))
            boundaries = find_step_boundaries(tok_strs)
            pred = extract_pred(text)
            gold = normalize(a) if isinstance(a, str) else None
            ok = equal(pred, gold)
            if pred is None:
                n_parse_fail += 1
            n_correct += int(ok)
            f.write(json.dumps({
                "id": i,
                "question": q,
                "gold_raw": a,
                "gold_norm": gold,
                "predicted": pred,
                "correct": ok,
                "output_text": text,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "n_steps": len(boundaries),
            }) + "\n")

    summary = {
        "model": MODEL,
        "dataset": "MATH-500",
        "n": len(prompts),
        "accuracy": n_correct / len(prompts),
        "parse_failures": n_parse_fail,
        "wallclock_sec": dur,
        "sec_per_question": dur / len(prompts),
    }
    (OUT.parent / "pilot7_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

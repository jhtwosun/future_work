"""
Pilot 2: GSM8K vanilla CoT baseline with Qwen2.5-7B-Instruct.

Captures per-token logprobs so downstream Pilot 3 can compute step-level
S-LP (mean log-prob per step) and entropy without re-running the model.

Output: results/pilot2_gsm8k_traces.jsonl
  - one record per question
  - fields: id, question, gold_answer, gold_value, prompt, output_text,
            tokens, token_logprobs, step_boundaries, predicted_value, correct
"""

import json
import os
import re
import time
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "300"))
MAX_TOKENS = 1024
SEED = 0
OUT = Path("/home/nvidia/future/pilots/cot_cp/results/pilot2_gsm8k_traces.jsonl")
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
    if m:
        return m.group(1).replace(",", "")
    return None


def parse_pred(text: str) -> str | None:
    m = PRED_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def numeric_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def build_chat_prompt(tokenizer, question: str) -> str:
    user_content = PROMPT_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def find_step_boundaries(token_strs: list[str]) -> list[int]:
    """A step boundary = the index *after* a token that contains a newline
    sequence creating a blank line ('\\n\\n'). We mark the position of the
    NEXT step's first token. Boundary 0 is the implicit start of step 1.
    """
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        # detect '\n\n' just produced (works even if newlines come as separate tokens)
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def main() -> None:
    print(f"Loading GSM8K test set...")
    ds = load_dataset("gsm8k", "main", split="test")
    questions = ds.select(range(min(N_QUESTIONS, len(ds))))
    print(f"  using {len(questions)} questions")

    print(f"Loading vLLM model: {MODEL}")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
        logprobs=1,  # capture log-prob of chosen token
        stop=None,
        seed=SEED,
    )

    # Build prompts
    prompts = [build_chat_prompt(tokenizer, q["question"]) for q in questions]
    golds = [parse_gold(q["answer"]) for q in questions]

    print(f"Generating {len(prompts)} responses...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    dur = time.time() - t0
    print(f"  generation took {dur:.1f}s ({dur/len(prompts):.2f}s/q)")

    n_correct = 0
    n_parse_fail = 0
    with OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outputs)):
            o = out.outputs[0]
            text = o.text
            tok_ids = list(o.token_ids)
            tok_strs = [tokenizer.decode([tid]) for tid in tok_ids]

            # vLLM logprobs structure: list[dict[token_id -> Logprob]]
            tok_logprobs: list[float] = []
            for step_lp in o.logprobs or []:
                if step_lp is None:
                    tok_logprobs.append(float("nan"))
                    continue
                # logprob of *chosen* token is at the chosen id
                # find the entry whose rank == 1 ; vLLM also provides chosen
                vals = list(step_lp.values())
                # the "rank" attr is 1 for the chosen token when logprobs=1
                chosen = vals[0]
                tok_logprobs.append(float(chosen.logprob))

            boundaries = find_step_boundaries(tok_strs)
            pred = parse_pred(text)
            if pred is None:
                n_parse_fail += 1
            ok = numeric_equal(pred, gold)
            n_correct += int(ok)

            rec = {
                "id": i,
                "question": q["question"],
                "gold_answer_raw": q["answer"],
                "gold_value": gold,
                "predicted_value": pred,
                "correct": ok,
                "output_text": text,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "n_steps": len(boundaries),
                "tokens_decoded_preview": tok_strs[:20],
            }
            f.write(json.dumps(rec) + "\n")

    acc = n_correct / len(prompts)
    print(f"Accuracy: {n_correct}/{len(prompts)} = {acc:.3f}")
    print(f"Parse failures: {n_parse_fail}")
    print(f"Output: {OUT}")
    summary = {
        "model": MODEL,
        "n": len(prompts),
        "accuracy": acc,
        "parse_failures": n_parse_fail,
        "wallclock_sec": dur,
        "sec_per_question": dur / len(prompts),
    }
    (OUT.parent / "pilot2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

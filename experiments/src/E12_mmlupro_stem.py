"""
Experiment E12: MMLU-Pro STEM subset (math + physics + chemistry + CS)
on Qwen3-8B (no-think). ~4200 MCQ problems.

Tests CoT-CP across domains beyond pure math.
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

MODEL = "Qwen/Qwen3-8B"
N_SAMPLES = int(os.environ.get("NS", "8"))
MAX_TOKENS = 1536
TEMP_SC = 0.7
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
GREEDY_OUT = OUTDIR / "E12_mmlupro_stem_greedy.jsonl"
SC_OUT = OUTDIR / "E12_mmlupro_stem_sc.jsonl"
SUM = OUTDIR / "E12_summary.json"

PROMPT_TEMPLATE = (
    "Answer the following multiple-choice question. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, put your final answer letter (A-J) within \\boxed{{}}.\n\n'
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
)


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def extract_letter(text):
    """Extract A-J letter from \\boxed{} or 'Answer: X' patterns."""
    boxed = re.findall(r"\\boxed\{([A-J])\}", text)
    if boxed:
        return boxed[-1]
    boxed_full = re.findall(r"\\boxed\{(?:Option |Answer )?\(?([A-J])\)?[^}]*\}", text)
    if boxed_full:
        return boxed_full[-1]
    ans = re.findall(r"(?i)\bAnswer\s*[:\-=]?\s*\(?([A-J])\)?\b", text)
    if ans:
        return ans[-1]
    final = re.findall(r"(?i)\b(?:final\s+answer|the\s+answer\s+is)\s*[:\-=]?\s*\(?([A-J])\)?\b", text)
    if final:
        return final[-1]
    # Last bare letter at end
    tail = text[-200:]
    letters = re.findall(r"\b([A-J])\b", tail)
    if letters:
        return letters[-1]
    return None


def main():
    print("Loading MMLU-Pro test...", flush=True)
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    # STEM subset
    stem_categories = {"math", "physics", "chemistry", "computer science"}
    stem = [r for r in ds if r["category"] in stem_categories]
    print(f"  STEM subset: {len(stem)} problems", flush=True)
    cats = Counter(r["category"] for r in stem)
    print(f"  by category: {dict(cats)}", flush=True)

    print(f"Loading {MODEL}...", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2560,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tok = llm.get_tokenizer()

    questions = []
    golds = []
    cats_per_q = []
    for r in stem:
        # MMLU-Pro options is a list of 10 strings (some may be N/A)
        options = r["options"]
        opt_str = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
        # answer is letter A-J in MMLU-Pro
        questions.append(r["question"])
        golds.append(r["answer"])
        cats_per_q.append(r["category"])

    prompts = []
    for r in stem:
        opt_str = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(r["options"]))
        prompt_user = PROMPT_TEMPLATE.format(question=r["question"], options=opt_str)
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

    summary = {"model": MODEL + " (no-think)", "n": len(prompts),
                "dataset": "MMLU-Pro STEM",
                "by_category": dict(cats)}

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur_g = time.time() - t0
    n_correct = 0; n_parse_fail = 0
    n_truncated = 0
    cat_correct = Counter(); cat_total = Counter()
    with GREEDY_OUT.open("w") as f:
        for i, (q, gold, cat, out) in enumerate(zip(questions, golds, cats_per_q, outs)):
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
            pred = extract_letter(text)
            if pred is None: n_parse_fail += 1
            if len(tok_ids) >= MAX_TOKENS - 5: n_truncated += 1
            ok = (pred == gold)
            n_correct += int(ok)
            cat_total[cat] += 1
            if ok: cat_correct[cat] += 1
            f.write(json.dumps({
                "id": i, "category": cat, "question": q, "gold": gold,
                "predicted": pred, "correct": ok,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "output_text": text,
            }) + "\n")
    summary["greedy"] = {
        "accuracy": n_correct / len(prompts),
        "by_category": {c: cat_correct[c] / cat_total[c] for c in cat_total},
        "parse_failures": n_parse_fail,
        "n_truncated": n_truncated,
        "wallclock_sec": dur_g,
    }
    print(f"Greedy: {n_correct}/{len(prompts)} = {n_correct/len(prompts):.3f}, truncated {n_truncated}", flush=True)
    print(f"  by category: {dict(summary['greedy']['by_category'])}", flush=True)

    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_sc)
    dur_sc = time.time() - t0
    n_maj = 0; n_any = 0; top1_fracs = []
    cat_maj = Counter()
    with SC_OUT.open("w") as f:
        for i, (q, gold, cat, out) in enumerate(zip(questions, golds, cats_per_q, outs)):
            preds = [extract_letter(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter(preds_clean)
            top, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES; top1_fracs.append(top1_frac)
            ok = (top == gold); any_ok = any(p == gold for p in preds_clean)
            n_maj += int(ok); n_any += int(any_ok)
            if ok: cat_maj[cat] += 1
            f.write(json.dumps({
                "id": i, "category": cat, "gold": gold,
                "majority_pred": top, "majority_correct": int(ok),
                "any_correct": int(any_ok), "top1_frac": top1_frac,
                "answer_distribution": dict(counter),
            }) + "\n")
    summary["sc"] = {
        "n_samples": N_SAMPLES,
        "majority_accuracy": n_maj / len(prompts),
        "any_accuracy": n_any / len(prompts),
        "by_category": {c: cat_maj[c] / cat_total[c] for c in cat_total},
        "mean_top1_frac": float(np.mean(top1_fracs)),
        "wallclock_sec": dur_sc,
    }
    print(f"SC@{N_SAMPLES}: maj={n_maj/len(prompts):.3f}", flush=True)
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

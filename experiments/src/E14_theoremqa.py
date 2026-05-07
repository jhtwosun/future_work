"""
Experiment E14: TheoremQA (800 problems) on Qwen3-8B (no-think).
Theorem-based math reasoning.

Answer types: float, integer, list of float, list of integer, option, bool.
"""

import json
import os
import re
import time
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, normalize, equal_strict, to_number

MODEL = "Qwen/Qwen3-8B"
N_SAMPLES = int(os.environ.get("NS", "8"))
MAX_TOKENS = 1536
TEMP_SC = 0.7
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
GREEDY_OUT = OUTDIR / "E14_theoremqa_greedy.jsonl"
SC_OUT = OUTDIR / "E14_theoremqa_sc.jsonl"
SUM = OUTDIR / "E14_summary.json"

PROMPT_TEMPLATE = (
    "Solve the following theorem-based math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
)


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def thm_equal(pred, gold, gold_type):
    """Compare based on answer type."""
    if pred is None or gold is None:
        return False
    if gold_type in ("integer", "float"):
        pf = to_number(normalize(pred))
        try:
            gf = float(gold) if isinstance(gold, (int, float, str)) else None
        except (ValueError, TypeError):
            gf = None
        if pf is not None and gf is not None:
            return abs(pf - gf) < 1e-3 * max(1.0, abs(gf))
        return False
    if gold_type == "bool":
        gold_str = str(gold).lower()
        pred_str = str(pred).lower()
        return ("true" in pred_str) == ("true" in gold_str) or ("yes" in pred_str) == ("yes" in gold_str) or ("false" in pred_str) == ("false" in gold_str)
    if gold_type == "option":
        # answer is e.g., "(a)" or "a"
        m = re.search(r"\b([a-d])\b", str(pred).lower())
        gm = re.search(r"\b([a-d])\b", str(gold).lower())
        return m is not None and gm is not None and m.group(1) == gm.group(1)
    if "list" in gold_type:
        # parse both as list of numbers
        def to_list(x):
            if isinstance(x, list): return x
            s = str(x)
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            return [float(n) for n in nums]
        try:
            pl = to_list(pred); gl = to_list(gold)
            if len(pl) != len(gl): return False
            return all(abs(a - b) < 1e-3 * max(1.0, abs(b)) for a, b in zip(pl, gl))
        except Exception:
            return False
    # Fallback: string compare
    return equal_strict(str(pred), str(gold))


def main():
    print("Loading TheoremQA...", flush=True)
    ds = load_dataset("TIGER-Lab/TheoremQA", split="test")
    questions = [r["Question"] for r in ds]
    golds = [r["Answer"] for r in ds]
    types = [r["Answer_type"] for r in ds]
    has_pic = [r.get("Picture") is not None for r in ds]
    print(f"  {len(ds)} problems; {sum(has_pic)} with images (will skip)", flush=True)
    # Keep only text-only
    keep = [i for i in range(len(ds)) if not has_pic[i]]
    questions = [questions[i] for i in keep]
    golds = [golds[i] for i in keep]
    types = [types[i] for i in keep]
    print(f"  text-only: {len(questions)}", flush=True)

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
    prompts = []
    for q in questions:
        prompt_user = PROMPT_TEMPLATE.format(question=q)
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

    summary = {"model": MODEL + " (no-think)", "n": len(prompts), "dataset": "TheoremQA"}
    type_counter = Counter(types)
    summary["by_answer_type"] = dict(type_counter)

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur_g = time.time() - t0
    n_correct = 0; n_truncated = 0
    type_correct = Counter()
    with GREEDY_OUT.open("w") as f:
        for i, (q, gold, gtyp, out) in enumerate(zip(questions, golds, types, outs)):
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
            pred = extract_pred(text)
            if len(tok_ids) >= MAX_TOKENS - 5: n_truncated += 1
            ok = thm_equal(pred, gold, gtyp)
            n_correct += int(ok)
            if ok: type_correct[gtyp] += 1
            f.write(json.dumps({
                "id": i, "question": q, "gold": gold, "answer_type": gtyp,
                "predicted": pred, "correct": ok,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "output_text": text,
            }) + "\n")
    summary["greedy"] = {
        "accuracy": n_correct / len(prompts),
        "n_truncated": n_truncated,
        "by_type": {t: type_correct[t] / type_counter[t] for t in type_counter if type_counter[t]},
        "wallclock_sec": dur_g,
    }
    print(f"Greedy: {n_correct}/{len(prompts)} = {n_correct/len(prompts):.3f}", flush=True)

    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_sc)
    dur_sc = time.time() - t0
    n_maj = 0; n_any = 0; top1_fracs = []
    with SC_OUT.open("w") as f:
        for i, (q, gold, gtyp, out) in enumerate(zip(questions, golds, types, outs)):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter([normalize(p) for p in preds_clean])
            top_norm, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES; top1_fracs.append(top1_frac)
            ok = thm_equal(top_norm, gold, gtyp)
            any_ok = any(thm_equal(p, gold, gtyp) for p in preds_clean)
            n_maj += int(ok); n_any += int(any_ok)
            f.write(json.dumps({
                "id": i, "gold": gold, "answer_type": gtyp,
                "majority_pred": top_norm, "majority_correct": int(ok),
                "any_correct": int(any_ok), "top1_frac": top1_frac,
                "answer_distribution": dict(counter),
            }) + "\n")
    summary["sc"] = {
        "n_samples": N_SAMPLES,
        "majority_accuracy": n_maj / len(prompts),
        "any_accuracy": n_any / len(prompts),
        "mean_top1_frac": float(np.mean(top1_fracs)),
        "wallclock_sec": dur_sc,
    }
    print(f"SC@{N_SAMPLES}: maj={n_maj/len(prompts):.3f}", flush=True)
    SUM.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

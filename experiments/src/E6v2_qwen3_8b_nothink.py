"""
Experiment E6 v2: Qwen3-8B with thinking mode DISABLED.

The original E6 ran with default chat template which enables Qwen3's
"thinking" mode. 375/500 traces hit max_tokens=1536 mid-think and
never produced a final answer, giving an artificially low 37% accuracy.

This version sets enable_thinking=False on apply_chat_template, which
gives a comparable inference cost to Qwen2.5-7B (no internal thinking).

Uses boxed-aware extractor to handle Qwen3's default \\boxed{} answer style.
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
OUTDIR.mkdir(parents=True, exist_ok=True)
GREEDY_OUT = OUTDIR / "E6v2_qwen3_8b_nothink_greedy.jsonl"
SC_OUT = OUTDIR / "E6v2_qwen3_8b_nothink_sc.jsonl"
SUM = OUTDIR / "E6v2_summary.json"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, put your final answer within \\boxed{{}}.\n\n'
    "Problem: {question}\n\n"
)
PRED_BOX = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
PRED_ANS = re.compile(r"(?i)answer\s*[:=]\s*([^\n]+)")
NUM_RE = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")


def normalize(s):
    if s is None: return None
    s = str(s).strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.rstrip(".,;:")
    s = re.sub(r"\s+", "", s)
    while True:
        m = re.match(r"\\boxed\{(.+)\}$", s)
        if m: s = m.group(1)
        else: break
    return s


def extract_pred(text):
    matches = list(PRED_BOX.finditer(text))
    if matches: return normalize(matches[-1].group(1))
    m = PRED_ANS.search(text)
    if m: return normalize(m.group(1))
    nums = NUM_RE.findall(text)
    return normalize(nums[-1]) if nums else None


def equal(a, b):
    if a is None or b is None: return False
    a_n = normalize(a); b_n = normalize(b)
    if a_n == b_n: return True
    try:
        af = float(a_n) if "/" not in a_n else float(a_n.split("/")[0]) / float(a_n.split("/")[1])
        bf = float(b_n) if "/" not in b_n else float(b_n.split("/")[0]) / float(b_n.split("/")[1])
        return abs(af - bf) < 1e-4
    except Exception:
        pass
    def latex_strip(x):
        return re.sub(r"\\(left|right|cdot|,|;|!|:|quad|qquad)", "", x or "")
    return latex_strip(a_n) == latex_strip(b_n)


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def main():
    print("Loading MATH-500...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    questions = [r["problem"] for r in ds]
    golds = [normalize(r["answer"]) for r in ds]

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
    # Disable Qwen3 thinking mode
    prompts = []
    for q in questions:
        try:
            p = tok.apply_chat_template(
                [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # tokenizer may not support enable_thinking; fall back
            p = tok.apply_chat_template(
                [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
                tokenize=False, add_generation_prompt=True,
            )
        prompts.append(p)

    summary = {"model": MODEL + " (no-think)", "n": len(prompts), "dataset": "MATH-500"}

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur_g = time.time() - t0
    n_correct = 0; n_parse_fail = 0
    n_truncated = 0
    with GREEDY_OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
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
            if pred is None: n_parse_fail += 1
            if len(tok_ids) >= MAX_TOKENS - 5: n_truncated += 1
            ok = equal(pred, gold); n_correct += int(ok)
            f.write(json.dumps({
                "id": i, "question": q, "gold": gold,
                "predicted": pred, "correct": ok,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "output_text": text,
            }) + "\n")
    summary["greedy"] = {"accuracy": n_correct / len(prompts),
                          "parse_failures": n_parse_fail,
                          "n_truncated": n_truncated,
                          "wallclock_sec": dur_g}
    print(f"Greedy: {n_correct}/{len(prompts)} = {n_correct/len(prompts):.3f}, truncated {n_truncated}", flush=True)

    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_sc)
    dur_sc = time.time() - t0
    n_maj = 0; n_any = 0; top1_fracs = []
    with SC_OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter(preds_clean)
            top, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES; top1_fracs.append(top1_frac)
            ok = equal(top, gold); any_ok = any(equal(p, gold) for p in preds_clean)
            n_maj += int(ok); n_any += int(any_ok)
            f.write(json.dumps({
                "id": i, "question": q, "gold": gold,
                "majority_pred": top, "majority_correct": int(ok),
                "any_correct": int(any_ok), "top1_frac": top1_frac,
                "answer_distribution": dict(counter),
            }) + "\n")
    summary["sc"] = {"n_samples": N_SAMPLES,
                       "majority_accuracy": n_maj / len(prompts),
                       "oracle_any": n_any / len(prompts),
                       "mean_top1_frac": float(np.mean(top1_fracs)),
                       "wallclock_sec": dur_sc}
    print(f"SC@{N_SAMPLES}: maj={n_maj/len(prompts):.3f}", flush=True)
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

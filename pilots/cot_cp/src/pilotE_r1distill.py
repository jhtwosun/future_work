"""
Pilot E: DeepSeek-R1-Distill-Qwen-7B on MATH-500.

Long-CoT reasoning model — many more steps per problem.
Test whether step-level calibration is more or less informative
in the long-CoT regime.

Strategy: greedy + SC@8 on MATH-500 200 problems, with logprobs.
Uses larger max_tokens (4096) to accommodate long reasoning.

If model is missing locally, vLLM will download (~14 GiB).
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
N_QUESTIONS = int(os.environ.get("NQ", "200"))
N_SAMPLES = int(os.environ.get("NS", "8"))
MAX_TOKENS = int(os.environ.get("MAXTOK", "4096"))
SEED = 0

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")
GREEDY_OUT = RESULTS / "pilotE_r1_greedy_traces.jsonl"
SC_OUT = RESULTS / "pilotE_r1_sc_traces.jsonl"
SUM = RESULTS / "pilotE_summary.json"

# R1-distill expects pure problem prompt, no special chat (it has its own template)
PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "{question}"
)
PRED_RE_BOX = re.compile(r"\\boxed\{([^{}]+)\}")
PRED_RE_ANS = re.compile(r"(?i)answer\s*[:=]\s*([^\n]+)")
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
    matches = list(PRED_RE_BOX.finditer(text))
    if matches:
        return normalize(matches[-1].group(1))
    m = PRED_RE_ANS.search(text)
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


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def main():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    use = ds.select(range(min(N_QUESTIONS, len(ds))))
    questions = [r["problem"] for r in use]
    golds = [normalize(r["answer"]) for r in use]
    print(f"{len(questions)} MATH-500 problems on R1-Distill", flush=True)

    print(f"Loading model {MODEL} (may download)...", flush=True)
    t_load = time.time()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        tensor_parallel_size=1,
        seed=SEED,
        download_dir=None,
    )
    print(f"  loaded in {time.time()-t_load:.1f}s", flush=True)
    tok = llm.get_tokenizer()

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    summary = {"model": MODEL, "n_questions": len(prompts)}

    # ============== Greedy with logprobs ===============
    print("=== Greedy with logprobs ===", flush=True)
    sp_greedy = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED,
    )
    t0 = time.time()
    outs = llm.generate(prompts, sp_greedy)
    dur = time.time() - t0

    n_correct = 0
    n_parse_fail = 0
    n_steps_list = []
    n_tokens_list = []
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
            ok = equal(pred, gold)
            if pred is None:
                n_parse_fail += 1
            n_correct += int(ok)
            n_steps_list.append(len(boundaries))
            n_tokens_list.append(len(tok_ids))
            f.write(json.dumps({
                "id": i,
                "question": q,
                "gold": gold,
                "predicted": pred,
                "correct": ok,
                "output_text": text,
                "n_tokens": len(tok_ids),
                "token_logprobs": tok_logprobs,
                "step_boundaries": boundaries,
                "n_steps": len(boundaries),
            }) + "\n")
    summary["greedy"] = {
        "accuracy": n_correct / len(prompts),
        "parse_failures": n_parse_fail,
        "wallclock_sec": dur,
        "mean_steps": float(np.mean(n_steps_list)),
        "median_steps": float(np.median(n_steps_list)),
        "max_steps": int(np.max(n_steps_list)),
        "mean_tokens": float(np.mean(n_tokens_list)),
        "median_tokens": float(np.median(n_tokens_list)),
    }
    print(f"  greedy acc={summary['greedy']['accuracy']:.3f}  steps mean={summary['greedy']['mean_steps']:.1f} median={summary['greedy']['median_steps']:.1f}  tok mean={summary['greedy']['mean_tokens']:.0f}", flush=True)

    # ============== SC@N_SAMPLES ===============
    print(f"=== SC@{N_SAMPLES} ===", flush=True)
    sp_sc = SamplingParams(
        n=N_SAMPLES, temperature=0.7, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED,
    )
    t0 = time.time()
    outs = llm.generate(prompts, sp_sc)
    dur = time.time() - t0
    n_maj = 0
    n_any = 0
    top1s = []
    with SC_OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter(preds_clean)
            top, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES
            top1s.append(top1_frac)
            ok = equal(top, gold)
            any_ok = any(equal(p, gold) for p in preds_clean)
            n_maj += int(ok)
            n_any += int(any_ok)
            f.write(json.dumps({
                "id": i,
                "majority_correct": int(ok),
                "any_correct": int(any_ok),
                "top1_frac": top1_frac,
                "majority_pred": top,
                "gold": gold,
                "answer_distribution": dict(counter),
            }) + "\n")
    summary["sc"] = {
        "n_samples": N_SAMPLES,
        "majority_accuracy": n_maj / len(prompts),
        "oracle_any": n_any / len(prompts),
        "mean_top1_frac": float(np.mean(top1s)),
        "wallclock_sec": dur,
    }
    print(f"  sc maj={summary['sc']['majority_accuracy']:.3f}  any={summary['sc']['oracle_any']:.3f}  top1={summary['sc']['mean_top1_frac']:.3f}", flush=True)

    SUM.write_text(json.dumps(summary, indent=2))
    print(f"Wrote: {SUM}", flush=True)


if __name__ == "__main__":
    main()

"""
Pilot B: OOD coverage test — calibrate on MATH-500, evaluate on AIME-2024.

This tests whether the conformal quantile q_hat learned on MATH-500 traces
still gives the promised coverage when applied to AIME-2024 (harder, OOD).

Steps:
  1. Run AIME-2024 (30 problems) with greedy + SC@8, mirroring pilot7/pilot9.
  2. Use MATH-500 traces (pilot7) as calibration; compute q_hat.
  3. Apply q_hat to AIME and report empirical coverage and selective accuracy.
"""

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_SAMPLES = 8
MAX_TOKENS = 2048
TEMP_SC = 0.7
SEED = 0

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")
GREEDY_OUT = RESULTS / "pilotB_aime_greedy.jsonl"
SC_OUT = RESULTS / "pilotB_aime_sc.jsonl"
SUM = RESULTS / "pilotB_summary.json"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number>".\n\n'
    "Problem: {question}\n\n"
)
PRED_RE = re.compile(r"(?i)answer\s*[:=]\s*([^\n]+)")
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
        af = float(a)
        bf = float(b)
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


def trajectory_lp_scores(token_logprobs, boundaries):
    bdy = boundaries + [len(token_logprobs)]
    steps = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            steps.append(np.mean(seg))
    if not steps:
        return None
    return {
        "lp_mean": float(np.mean(steps)),
        "lp_min":  float(np.min(steps)),
        "lp_median": float(np.median(steps)),
        "lp_tok": float(np.mean([x for x in token_logprobs if not (x != x)])),
        "n_steps": len(steps),
    }


def main():
    print("Loading AIME-2024...", flush=True)
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    questions = [r["Problem"] for r in ds]
    golds = [normalize(r["Answer"]) for r in ds]
    print(f"  {len(questions)} AIME problems", flush=True)

    print(f"Loading {MODEL}...", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    summary = {"model": MODEL, "n_aime": len(prompts)}

    # Greedy with logprobs
    sp_g = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_g)
    dur_g = time.time() - t0
    n_correct = 0
    aime_records = []
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
            n_correct += int(ok)
            scores = trajectory_lp_scores(tok_logprobs, boundaries)
            rec = {
                "id": i, "question": q, "gold": gold,
                "predicted": pred, "correct": ok,
                "n_tokens": len(tok_ids),
                "boundaries_n": len(boundaries),
                **(scores or {}),
            }
            aime_records.append(rec)
            f.write(json.dumps({**rec, "output_text": text,
                                  "token_logprobs": tok_logprobs,
                                  "step_boundaries": boundaries}) + "\n")
    summary["aime_greedy"] = {
        "accuracy": n_correct / len(prompts),
        "wallclock_sec": dur_g,
    }
    print(f"AIME greedy: {n_correct}/{len(prompts)} = {n_correct/len(prompts):.3f}", flush=True)

    # SC@8
    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_sc)
    dur_sc = time.time() - t0
    n_maj = 0
    n_any = 0
    sc_records = []
    with SC_OUT.open("w") as f:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter(preds_clean)
            top, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES
            ok = equal(top, gold)
            any_ok = any(equal(p, gold) for p in preds_clean)
            n_maj += int(ok)
            n_any += int(any_ok)
            rec = {
                "id": i,
                "majority_correct": int(ok),
                "any_correct": int(any_ok),
                "top1_frac": top1_frac,
                "majority_pred": top,
                "gold": gold,
                "answer_distribution": dict(counter),
            }
            sc_records.append(rec)
            f.write(json.dumps(rec) + "\n")
    summary["aime_sc"] = {
        "majority_accuracy": n_maj / len(prompts),
        "any_accuracy": n_any / len(prompts),
        "mean_top1_frac": float(np.mean([r["top1_frac"] for r in sc_records])),
        "wallclock_sec": dur_sc,
    }
    print(f"AIME SC@8: maj={n_maj}/{len(prompts)} any={n_any}/{len(prompts)}", flush=True)

    # === OOD coverage: calibrate on MATH-500, evaluate on AIME ===
    math_lp_path = RESULTS / "pilot7_math500_traces.jsonl"
    math_sc_path = RESULTS / "pilot9_math500_sc_traces.jsonl"
    math_traces = [json.loads(l) for l in math_lp_path.read_text().splitlines() if l.strip()]
    math_sc_traces = [json.loads(l) for l in math_sc_path.read_text().splitlines() if l.strip()]

    def _math_lp(rs):
        out = []
        for r in rs:
            sc = trajectory_lp_scores(r["token_logprobs"], r["step_boundaries"])
            if sc:
                out.append({**sc, "correct": int(bool(r["correct"]))})
        return out

    math_lp_all = _math_lp(math_traces)
    math_sc_all = [{"sc_top1": float(r["top1_frac"]), "correct": int(r["majority_correct"])}
                    for r in math_sc_traces]

    def cp_eval(cal_records, te_records, score_key, alphas):
        cal_scores = np.array([r[score_key] for r in cal_records])
        cal_correct = np.array([r["correct"] for r in cal_records])
        te_scores = np.array([r[score_key] for r in te_records])
        te_correct = np.array([r["correct"] for r in te_records])
        rows = []
        for alpha in alphas:
            cal_corr_scores = cal_scores[cal_correct == 1]
            n = len(cal_corr_scores)
            ql = max(0.0, min(1.0, math.floor(alpha * (n + 1)) / n))
            q = float(np.quantile(cal_corr_scores, ql))
            kept = te_scores >= q
            cov = float(((kept) & (te_correct == 1)).sum() / max((te_correct == 1).sum(), 1))
            acc = float(te_correct[kept].mean()) if kept.sum() else float("nan")
            rows.append({
                "alpha": alpha, "target": 1 - alpha,
                "empirical_coverage": cov,
                "kept_acc": acc,
                "kept_frac": float(kept.mean()),
                "q_hat": q,
                "n_test": int(len(te_scores)),
            })
        return rows

    aime_lp_records = []
    for r in aime_records:
        if "lp_mean" in r and r["lp_mean"] is not None:
            aime_lp_records.append({**r, "correct": int(bool(r["correct"]))})

    aime_sc_records_all = [{"sc_top1": float(r["top1_frac"]), "correct": int(r["majority_correct"])}
                            for r in sc_records]

    summary["ood_results"] = {}
    alphas = [0.05, 0.1, 0.2, 0.3, 0.5]
    for sk in ["lp_mean", "lp_min", "lp_median"]:
        if not all(sk in r for r in aime_lp_records):
            continue
        out_rows = cp_eval(math_lp_all, aime_lp_records, sk, alphas)
        summary["ood_results"][f"calMATH-->testAIME LP/{sk}"] = out_rows
        for r in out_rows:
            print(f"[OOD LP/{sk} α={r['alpha']:.2f}] target={r['target']:.2f} cov={r['empirical_coverage']:.3f} keepacc={r['kept_acc']:.3f} keep%={r['kept_frac']:.2f} qhat={r['q_hat']:.3f}", flush=True)

    out_rows = cp_eval(math_sc_all, aime_sc_records_all, "sc_top1", alphas)
    summary["ood_results"]["calMATH-->testAIME SC"] = out_rows
    for r in out_rows:
        print(f"[OOD SC α={r['alpha']:.2f}] target={r['target']:.2f} cov={r['empirical_coverage']:.3f} keepacc={r['kept_acc']:.3f} keep%={r['kept_frac']:.2f} qhat={r['q_hat']:.3f}", flush=True)

    SUM.write_text(json.dumps(summary, indent=2))
    print(f"Wrote: {SUM}")


if __name__ == "__main__":
    main()

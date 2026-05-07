"""
Pilot I: AIME 1983-2024 (~933 problems) as a meaningful OOD test for
MATH-calibrated CP. Replaces the under-powered Pilot B (n=30).

Run Qwen2.5-7B-Instruct greedy + SC@8 on AIME 1983-2024, then test
empirical OOD coverage with both vanilla and weighted CP, and produce
proper bootstrap CIs.

Output: pilotI_aime_extended_traces.jsonl (greedy), pilotI_aime_sc.jsonl,
pilotI_summary.json
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
N_SAMPLES = int(os.environ.get("NS", "8"))
N_QUESTIONS = int(os.environ.get("NQ", "300"))  # subset for speed
MAX_TOKENS = 2048
TEMP_SC = 0.7
SEED = 0

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")
GREEDY_OUT = RESULTS / "pilotI_aime_extended_traces.jsonl"
SC_OUT = RESULTS / "pilotI_aime_sc.jsonl"
SUM = RESULTS / "pilotI_summary.json"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number>". '
    "AIME answers are integers from 0 to 999.\n\n"
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
        af = float(a); bf = float(b)
        return abs(af - bf) < 1e-4
    except Exception:
        return False


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def step_lp(token_logprobs, boundaries):
    bdy = boundaries + [len(token_logprobs)]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            out.append(np.mean(seg))
    return out


def split_cp_simple(scores, correct, alpha):
    s = np.array(scores); c = np.array(correct)
    cal_correct = s[c == 1]
    if len(cal_correct) < 5:
        return None
    n = len(cal_correct)
    ql = max(0.0, min(1.0, math.floor(alpha * (n + 1)) / n))
    return float(np.quantile(cal_correct, ql))


def main():
    print("Loading AIME 1983-2024...", flush=True)
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    # Sort by year so we can subset newest if needed, but use random subset for diversity
    rng = np.random.default_rng(0)
    indices = rng.permutation(len(ds))[:N_QUESTIONS]
    use = ds.select(sorted(indices.tolist()))
    questions = [r["Question"] for r in use]
    golds = [normalize(r["Answer"]) for r in use]
    years = [r["Year"] for r in use]
    print(f"  {len(questions)} AIME problems sampled (years {min(years)}-{max(years)})", flush=True)

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
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]

    summary = {"model": MODEL, "n_aime": len(prompts), "year_range": [min(years), max(years)]}

    # Greedy with logprobs
    sp_g = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=1, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_g)
    dur_g = time.time() - t0
    n_correct = 0
    n_parse_fail = 0
    aime_records = []
    with GREEDY_OUT.open("w") as f:
        for i, (q, gold, year, out) in enumerate(zip(questions, golds, years, outs)):
            o = out.outputs[0]
            text = o.text
            tok_ids = list(o.token_ids)
            tok_strs = [tok.decode([tid]) for tid in tok_ids]
            tok_logprobs = []
            for step_lp_dict in o.logprobs or []:
                if step_lp_dict is None:
                    tok_logprobs.append(float("nan"))
                    continue
                vals = list(step_lp_dict.values())
                tok_logprobs.append(float(vals[0].logprob))
            boundaries = find_step_boundaries(tok_strs)
            pred = extract_pred(text)
            if pred is None:
                n_parse_fail += 1
            ok = equal(pred, gold)
            n_correct += int(ok)
            ss = step_lp(tok_logprobs, boundaries)
            scores = {} if not ss else {
                "lp_mean": float(np.mean(ss)),
                "lp_min": float(np.min(ss)),
                "lp_median": float(np.median(ss)),
            }
            rec = {"id": i, "year": year, "question": q, "gold": gold,
                    "predicted": pred, "correct": ok,
                    "n_tokens": len(tok_ids),
                    "boundaries_n": len(boundaries),
                    **scores}
            aime_records.append(rec)
            f.write(json.dumps({**rec, "output_text": text,
                                  "token_logprobs": tok_logprobs,
                                  "step_boundaries": boundaries}) + "\n")
    summary["aime_greedy"] = {
        "accuracy": n_correct / len(prompts),
        "parse_failures": n_parse_fail,
        "wallclock_sec": dur_g,
    }
    print(f"AIME greedy: {n_correct}/{len(prompts)} = {n_correct/len(prompts):.3f} (parse fail {n_parse_fail})", flush=True)

    # SC@8
    sp_sc = SamplingParams(n=N_SAMPLES, temperature=TEMP_SC, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, sp_sc)
    dur_sc = time.time() - t0
    n_maj = 0; n_any = 0
    sc_records = []
    with SC_OUT.open("w") as f:
        for i, (q, gold, year, out) in enumerate(zip(questions, golds, years, outs)):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter(preds_clean)
            top, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N_SAMPLES
            ok = equal(top, gold)
            any_ok = any(equal(p, gold) for p in preds_clean)
            n_maj += int(ok)
            n_any += int(any_ok)
            rec = {"id": i, "year": year,
                    "majority_correct": int(ok),
                    "any_correct": int(any_ok),
                    "top1_frac": top1_frac,
                    "majority_pred": top, "gold": gold,
                    "answer_distribution": dict(counter)}
            sc_records.append(rec)
            f.write(json.dumps(rec) + "\n")
    summary["aime_sc"] = {
        "majority_accuracy": n_maj / len(prompts),
        "any_accuracy": n_any / len(prompts),
        "mean_top1_frac": float(np.mean([r["top1_frac"] for r in sc_records])),
        "wallclock_sec": dur_sc,
    }
    print(f"AIME SC@{N_SAMPLES}: maj={n_maj}/{len(prompts)} ({n_maj/len(prompts):.3f}) any={n_any}/{len(prompts)} ({n_any/len(prompts):.3f})", flush=True)

    # OOD CP: calibrate on MATH-500, evaluate on AIME
    p7 = [json.loads(l) for l in (RESULTS / "pilot7_math500_traces.jsonl").read_text().splitlines() if l.strip()]
    p9 = [json.loads(l) for l in (RESULTS / "pilot9_math500_sc_traces.jsonl").read_text().splitlines() if l.strip()]
    cal_lp = []
    for r in p7:
        ss = step_lp(r["token_logprobs"], r["step_boundaries"])
        if not ss:
            continue
        cal_lp.append({
            "lp_mean": float(np.mean(ss)),
            "lp_min": float(np.min(ss)),
            "lp_median": float(np.median(ss)),
            "correct": int(bool(r["correct"])),
        })
    cal_sc = [{"sc_top1": float(r["top1_frac"]),
                "correct": int(r["majority_correct"])} for r in p9]

    summary["ood_cp"] = []
    for sk in ["lp_mean", "lp_min", "lp_median"]:
        cal_scores = np.array([r[sk] for r in cal_lp])
        cal_correct = np.array([r["correct"] for r in cal_lp])
        te_scores = np.array([r[sk] for r in aime_records if sk in r])
        te_correct = np.array([int(r["correct"]) for r in aime_records if sk in r])
        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
            q = split_cp_simple(cal_scores.tolist(), cal_correct.tolist(), alpha)
            if q is None:
                continue
            kept = te_scores >= q
            n_corr = (te_correct == 1).sum()
            cov = float(((kept) & (te_correct == 1)).sum() / max(n_corr, 1))
            acc = float(te_correct[kept].mean()) if kept.sum() else float("nan")
            keep = float(kept.mean())
            row = {"score": sk, "alpha": alpha, "target": 1 - alpha,
                    "n_test": int(len(te_scores)),
                    "vanilla": {"q": q, "cov": cov, "kept_acc": acc, "kept_frac": keep}}

            # weighted CP
            from scipy.stats import gaussian_kde
            cal_corr_scores = cal_scores[cal_correct == 1]
            kde_cal = gaussian_kde(cal_scores, bw_method="silverman")
            kde_te = gaussian_kde(te_scores, bw_method="silverman")
            weights = (kde_te(cal_corr_scores) / np.maximum(kde_cal(cal_corr_scores), 1e-9)).flatten()
            weights = np.clip(weights, 1e-6, 1e6)
            order = np.argsort(cal_corr_scores)
            s_sorted = cal_corr_scores[order]
            w_sorted = weights[order]
            cum = np.cumsum(w_sorted / w_sorted.sum())
            idx = min(np.searchsorted(cum, alpha), len(s_sorted) - 1)
            qw = float(s_sorted[idx])
            keptw = te_scores >= qw
            covw = float(((keptw) & (te_correct == 1)).sum() / max(n_corr, 1))
            accw = float(te_correct[keptw].mean()) if keptw.sum() else float("nan")
            keepw = float(keptw.mean())
            row["weighted"] = {"q": qw, "cov": covw, "kept_acc": accw, "kept_frac": keepw}

            # bootstrap CIs (1000 boot)
            rng = np.random.default_rng(0)
            boot_v_cov, boot_w_cov = [], []
            for _ in range(1000):
                idx_b = rng.integers(0, len(te_scores), size=len(te_scores))
                ts = te_scores[idx_b]; tc = te_correct[idx_b]
                n_corr_b = (tc == 1).sum()
                if n_corr_b == 0:
                    continue
                kv = ts >= q; kw = ts >= qw
                boot_v_cov.append(float(((kv) & (tc == 1)).sum() / n_corr_b))
                boot_w_cov.append(float(((kw) & (tc == 1)).sum() / n_corr_b))
            row["vanilla"]["cov_ci95"] = [float(np.quantile(boot_v_cov, 0.025)),
                                            float(np.quantile(boot_v_cov, 0.975))]
            row["weighted"]["cov_ci95"] = [float(np.quantile(boot_w_cov, 0.025)),
                                             float(np.quantile(boot_w_cov, 0.975))]

            summary["ood_cp"].append(row)
            print(f"[OOD {sk:9s} α={alpha:.2f}]  vanilla cov={cov:.3f} CI=[{row['vanilla']['cov_ci95'][0]:.3f},{row['vanilla']['cov_ci95'][1]:.3f}]  weighted cov={covw:.3f} CI=[{row['weighted']['cov_ci95'][0]:.3f},{row['weighted']['cov_ci95'][1]:.3f}]", flush=True)

    # SC OOD
    cal_sc_scores = np.array([r["sc_top1"] for r in cal_sc])
    cal_sc_correct = np.array([r["correct"] for r in cal_sc])
    te_sc_scores = np.array([r["top1_frac"] for r in sc_records])
    te_sc_correct = np.array([r["majority_correct"] for r in sc_records])
    summary["ood_cp_sc"] = []
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
        q = split_cp_simple(cal_sc_scores.tolist(), cal_sc_correct.tolist(), alpha)
        kept = te_sc_scores >= q
        n_corr = (te_sc_correct == 1).sum()
        cov = float(((kept) & (te_sc_correct == 1)).sum() / max(n_corr, 1))
        acc = float(te_sc_correct[kept].mean()) if kept.sum() else float("nan")
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(1000):
            idx_b = rng.integers(0, len(te_sc_scores), size=len(te_sc_scores))
            ts = te_sc_scores[idx_b]; tc = te_sc_correct[idx_b]
            n_corr_b = (tc == 1).sum()
            if n_corr_b == 0: continue
            kv = ts >= q
            boots.append(float(((kv) & (tc == 1)).sum() / n_corr_b))
        ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        summary["ood_cp_sc"].append({"alpha": alpha, "target": 1-alpha,
                                       "q": q, "cov": cov, "kept_acc": acc,
                                       "kept_frac": float(kept.mean()),
                                       "cov_ci95": ci})
        print(f"[OOD sc_top1 α={alpha:.2f}]  cov={cov:.3f} CI=[{ci[0]:.3f},{ci[1]:.3f}]  acc={acc:.3f}  keep%={kept.mean():.2f}", flush=True)

    SUM.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {SUM}", flush=True)


if __name__ == "__main__":
    main()

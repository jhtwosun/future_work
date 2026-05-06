"""Pilot E-analysis: reliability and CP simulation on R1-Distill traces.

Mirrors pilot10 structure but for the long-CoT model. The interesting
question is whether step-level log-prob is more informative for R1-Distill
(which has many more reasoning steps) than for Qwen2.5-7B-Instruct.
"""

import json
import math
from pathlib import Path

import numpy as np

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")


def step_lp(token_logprobs, boundaries):
    bdy = boundaries + [len(token_logprobs)]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            out.append(np.mean(seg))
    return out


def split_cp(scores, correct, alpha, n_seeds=200, cal_frac=0.5):
    rng = np.random.default_rng(0)
    accs, fracs, covs = [], [], []
    for _ in range(n_seeds):
        idx = rng.permutation(len(scores))
        nc = int(cal_frac * len(scores))
        ci, ti = idx[:nc], idx[nc:]
        cal_corr = scores[ci][correct[ci] == 1]
        if len(cal_corr) < 5:
            continue
        n = len(cal_corr)
        ql = max(0.0, min(1.0, math.floor(alpha * (n + 1)) / n))
        q = float(np.quantile(cal_corr, ql))
        kept = scores[ti] >= q
        n_corr = (correct[ti] == 1).sum()
        if n_corr == 0:
            continue
        cov = float((kept & (correct[ti] == 1)).sum() / n_corr)
        acc = float(correct[ti][kept].mean()) if kept.sum() else float("nan")
        fr  = float(kept.mean())
        accs.append(acc); fracs.append(fr); covs.append(cov)
    return {
        "kept_acc_mean":  float(np.mean(accs))  if accs  else float("nan"),
        "kept_frac_mean": float(np.mean(fracs)) if fracs else float("nan"),
        "coverage_mean":  float(np.mean(covs))  if covs  else float("nan"),
    }


def main():
    greedy_path = RESULTS / "pilotE_r1_greedy_traces.jsonl"
    sc_path = RESULTS / "pilotE_r1_sc_traces.jsonl"
    if not greedy_path.exists():
        raise SystemExit(f"missing {greedy_path}")

    rows = [json.loads(l) for l in greedy_path.read_text().splitlines() if l.strip()]
    print(f"R1-Distill greedy traces: {len(rows)}")

    per = []
    for r in rows:
        steps = step_lp(r["token_logprobs"], r["step_boundaries"])
        if not steps:
            continue
        per.append({
            "id": r["id"],
            "lp_mean":   float(np.mean(steps)),
            "lp_min":    float(np.min(steps)),
            "lp_median": float(np.median(steps)),
            "lp_tok":    float(np.mean([x for x in r["token_logprobs"] if not (x != x)])),
            "n_steps":   len(steps),
            "correct":   int(bool(r["correct"])),
        })
    scores = np.array([r["lp_mean"] for r in per])
    min_scores = np.array([r["lp_min"] for r in per])
    correct = np.array([r["correct"] for r in per])

    from scipy.stats import spearmanr, pointbiserialr
    rho_mean, p_mean = spearmanr(scores, correct)
    rho_min, p_min = spearmanr(min_scores, correct)
    rpb_mean, ppb_mean = pointbiserialr(scores, correct)
    rpb_min, ppb_min = pointbiserialr(min_scores, correct)
    n_steps_arr = np.array([r["n_steps"] for r in per])

    summary = {
        "model": "DeepSeek-R1-Distill-Qwen-7B",
        "n": len(per),
        "accuracy": float(correct.mean()),
        "n_steps_per_traj": {
            "mean": float(n_steps_arr.mean()),
            "median": float(np.median(n_steps_arr)),
            "max": int(n_steps_arr.max()),
        },
        "spearman_lp_mean": {"rho": float(rho_mean), "p": float(p_mean)},
        "spearman_lp_min":  {"rho": float(rho_min),  "p": float(p_min)},
        "pointbiserial_lp_mean": {"r": float(rpb_mean), "p": float(ppb_mean)},
        "pointbiserial_lp_min":  {"r": float(rpb_min),  "p": float(ppb_min)},
    }

    # CP simulation on greedy LP scores
    summary["cp_lp"] = []
    for sk in ["lp_mean", "lp_min", "lp_median", "lp_tok"]:
        s = np.array([r[sk] for r in per])
        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
            r = split_cp(s, correct, alpha)
            summary["cp_lp"].append({"score": sk, "alpha": alpha, **r})
            print(f"[R1-LP/{sk:9s} α={alpha:.2f}] cov={r['coverage_mean']:.3f} keepacc={r['kept_acc_mean']:.3f} keep%={r['kept_frac_mean']:.2f}")

    # SC reliability if available
    if sc_path.exists():
        sc_rows = [json.loads(l) for l in sc_path.read_text().splitlines() if l.strip()]
        sc_recs = [{"sc_top1": float(r["top1_frac"]),
                     "correct": int(r["majority_correct"])} for r in sc_rows]
        scs = np.array([r["sc_top1"] for r in sc_recs])
        scc = np.array([r["correct"] for r in sc_recs])
        summary["sc_majority_accuracy"] = float(scc.mean())
        summary["sc_mean_top1"] = float(scs.mean())
        summary["cp_sc"] = []
        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
            r = split_cp(scs, scc, alpha)
            summary["cp_sc"].append({"alpha": alpha, **r})
            print(f"[R1-SC                  α={alpha:.2f}] cov={r['coverage_mean']:.3f} keepacc={r['kept_acc_mean']:.3f} keep%={r['kept_frac_mean']:.2f}")

    out = RESULTS / "pilotE_analysis.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:1500])


if __name__ == "__main__":
    main()

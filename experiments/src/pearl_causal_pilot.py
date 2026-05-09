"""
Pearl-Causal Pilot — earliest-bad-step vs worst-step localization on existing
per-step CP traces.

Loads SX_prov_*_per_step.jsonl traces (12 cells: 4 models × 3 datasets) and,
for each trace, identifies:

  t_worst = argmin_t s_t                            (worst-step baseline)
  t_star  = min{t : s_t < q_alpha(t)}               (earliest divergent step)

where:
  - s_t is the per-step score (default: lp = mean log-prob of step t)
  - q_alpha(t) is a per-step calibrated threshold computed from CORRECT
    calibration traces only, at step position t (split-CP quantile).

For each cell, we compute:
  - distribution of (t_star, t_worst) and their gap
  - fraction of incorrect traces with t_star < t_worst (H4 evidence)
  - cascade depth = (t_worst - t_star) for incorrect traces
  - early-fraction t_star/T (does early divergence cluster early?)
  - bootstrap CIs on the gap statistics

Results: /home/nvidia/future/experiments/results/pearl_causal_pilot.json
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---- config -------------------------------------------------------------

RESULTS_DIR = Path("/home/nvidia/future/experiments/results")
OUTFILE = RESULTS_DIR / "pearl_causal_pilot.json"

# Per-cell file convention: SX_prov_{TAG}_{DATASET}_per_step.jsonl
TAGS = ["qwen25_7b", "qwen25_math_7b", "qwen25_32b", "phi4"]
DATASETS = ["math500", "aime", "olympiad"]

# CP risk levels to test
ALPHAS = [0.10, 0.30, 0.50]

# Score family. lp is in per_step_basic[t]['lp']; we expose three options
SCORE_KEYS = ["lp", "ent_neg", "marg"]   # ent_neg = -ent (higher = more confident)


# ---- IO -----------------------------------------------------------------

def load_traces(tag: str, dataset: str):
    fp = RESULTS_DIR / f"SX_prov_{tag}_{dataset}_per_step.jsonl"
    if not fp.exists():
        return None
    out = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            steps = r.get("per_step_basic", [])
            if not steps:
                continue
            # Build per-step score vectors (sign so higher = more confident)
            lp = [float(s["lp"]) if s.get("lp") is not None else float("nan") for s in steps]
            ent = [float(s["ent"]) if s.get("ent") is not None else float("nan") for s in steps]
            marg = [float(s.get("marg", 0.0)) for s in steps]
            ent_neg = [-e if not math.isnan(e) else float("nan") for e in ent]
            row = {
                "id": r.get("id"),
                "correct": int(r.get("correct", 0)),
                "T": len(steps),
                "lp": lp,
                "ent_neg": ent_neg,
                "marg": marg,
            }
            out.append(row)
    return out


# ---- core: t_worst and t_star --------------------------------------------

def t_worst(scores):
    """Argmin over scores (lower score = worse). Returns 0-indexed step."""
    arr = np.array(scores, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return int(np.nanargmin(arr))


def t_star(scores, thresholds):
    """First step t where scores[t] < thresholds[t]. Returns None if no violation.
    thresholds is a dict t -> threshold; missing t => no constraint at that step."""
    for t, s in enumerate(scores):
        if math.isnan(s):
            continue
        if t in thresholds and s < thresholds[t]:
            return t
    return None


def calibrate_per_step_thresholds(cal_correct_rows, score_key, alpha, max_T,
                                    min_cal_per_step=5):
    """For each step position t in [0, max_T), take the lower-alpha quantile
    of the score over correct calibration traces that have a step at position t.

    Standard split-CP quantile: ql = floor(alpha * (n+1)) / n.
    """
    thr = {}
    for t in range(max_T):
        vals = []
        for r in cal_correct_rows:
            if t < r["T"]:
                v = r[score_key][t]
                if not math.isnan(v):
                    vals.append(v)
        if len(vals) < min_cal_per_step:
            continue
        n_c = len(vals)
        ql = max(0.0, min(1.0, math.floor(alpha * (n_c + 1)) / n_c))
        thr[t] = float(np.quantile(vals, ql))
    return thr


# ---- per-cell analysis ---------------------------------------------------

def analyze_cell(rows, score_key, alpha, n_boot=200, rng=None):
    """For a given cell, compute (t_star, t_worst) statistics under split-CP
    calibration. We use a single split (cal=correct half, test=all rows) per
    bootstrap iteration to keep this lightweight; the calibration only uses
    *correct* trajectories (per-step CP convention).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(rows)
    if n < 20:
        return None
    max_T = max(r["T"] for r in rows)

    # We bootstrap over splits. Aggregate stats per trace across splits:
    # for each trace we collect t_star and t_worst observed in test folds.
    per_trace_stats = defaultdict(list)  # id -> list of (t_star, t_worst, T, correct)

    for _ in range(n_boot):
        perm = rng.permutation(n)
        nc = n // 2
        cal_idx, test_idx = perm[:nc], perm[nc:]
        cal_rows = [rows[i] for i in cal_idx]
        test_rows = [rows[i] for i in test_idx]
        cal_corr = [r for r in cal_rows if r["correct"] == 1]
        if len(cal_corr) < 5:
            continue
        thr = calibrate_per_step_thresholds(cal_corr, score_key, alpha, max_T)
        for r in test_rows:
            tw = t_worst(r[score_key])
            ts = t_star(r[score_key], thr)
            per_trace_stats[r["id"]].append({
                "t_worst": tw,
                "t_star": ts,
                "T": r["T"],
                "correct": r["correct"],
            })

    # Aggregate: for each trace take median t_star (over the bootstrap
    # iterations where it was a test point), majority t_worst is
    # deterministic from the trace itself (calibration-independent).
    agg = []
    for rid, lst in per_trace_stats.items():
        if not lst:
            continue
        # t_worst is calibration-independent; just use the first observation
        tw = lst[0]["t_worst"]
        T = lst[0]["T"]
        correct = lst[0]["correct"]
        # t_star: vote across iterations (modal value, treating None as -1 sentinel)
        ts_vals = [d["t_star"] if d["t_star"] is not None else -1 for d in lst]
        cnt = Counter(ts_vals)
        ts_modal, _ = cnt.most_common(1)[0]
        ts = ts_modal if ts_modal != -1 else None
        agg.append({"id": rid, "t_worst": tw, "t_star": ts, "T": T, "correct": correct})

    # ---- summary stats ----
    incorrect = [a for a in agg if a["correct"] == 0]
    correct = [a for a in agg if a["correct"] == 1]

    def gap_stats(group):
        # gap = t_worst - t_star ; if t_star is None, treat as "never violated"
        gaps_with_violation = []
        gaps_signed = []
        n_violation = 0
        n_t_star_le_t_worst = 0
        n_t_star_lt_t_worst = 0
        n_t_star_eq_t_worst = 0
        n_total = len(group)
        ts_vals = []
        tw_vals = []
        T_vals = []
        early_frac = []
        for a in group:
            tw_vals.append(a["t_worst"])
            T_vals.append(a["T"])
            if a["t_star"] is not None:
                n_violation += 1
                ts_vals.append(a["t_star"])
                gap = a["t_worst"] - a["t_star"]
                gaps_with_violation.append(gap)
                gaps_signed.append(gap)
                early_frac.append(a["t_star"] / max(a["T"], 1))
                if a["t_star"] <= a["t_worst"]:
                    n_t_star_le_t_worst += 1
                if a["t_star"] < a["t_worst"]:
                    n_t_star_lt_t_worst += 1
                if a["t_star"] == a["t_worst"]:
                    n_t_star_eq_t_worst += 1
        return {
            "n": n_total,
            "n_with_violation": n_violation,
            "violation_rate": n_violation / max(n_total, 1),
            "mean_t_star": float(np.mean(ts_vals)) if ts_vals else None,
            "mean_t_worst": float(np.mean(tw_vals)) if tw_vals else None,
            "mean_T": float(np.mean(T_vals)) if T_vals else None,
            "mean_t_star_normalized": float(np.mean(early_frac)) if early_frac else None,
            "mean_gap_t_worst_minus_t_star": float(np.mean(gaps_with_violation)) if gaps_with_violation else None,
            "median_gap": float(np.median(gaps_with_violation)) if gaps_with_violation else None,
            "frac_t_star_lt_t_worst_given_violation": (
                n_t_star_lt_t_worst / n_violation if n_violation > 0 else None),
            "frac_t_star_eq_t_worst_given_violation": (
                n_t_star_eq_t_worst / n_violation if n_violation > 0 else None),
            "frac_t_star_lt_t_worst_overall": n_t_star_lt_t_worst / max(n_total, 1),
            "gap_distribution_pp": (
                {str(int(k)): int(v) for k, v in sorted(Counter(gaps_signed).items())}
                if gaps_signed else {}
            ),
        }

    return {
        "n_total": len(agg),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "vanilla_acc": len(correct) / max(len(agg), 1),
        "incorrect": gap_stats(incorrect),
        "correct": gap_stats(correct),
        "all": gap_stats(agg),
    }


# ---- main ---------------------------------------------------------------

def main():
    print("=" * 78)
    print("Pearl-Causal Pilot — earliest-bad-step vs worst-step localization")
    print("=" * 78)
    out = {
        "config": {
            "tags": TAGS,
            "datasets": DATASETS,
            "alphas": ALPHAS,
            "score_keys": SCORE_KEYS,
            "n_boot": 200,
        },
        "cells": {},
    }
    rng = np.random.default_rng(42)

    for tag in TAGS:
        for ds in DATASETS:
            rows = load_traces(tag, ds)
            cell_key = f"{tag}__{ds}"
            if rows is None or len(rows) < 20:
                print(f"[skip] {cell_key}: missing or too few rows")
                continue
            n_corr = sum(r["correct"] for r in rows)
            print(f"\n=== {cell_key}  n={len(rows)}  vanilla_acc={n_corr/len(rows):.3f} ===")
            cell_out = {"n": len(rows), "vanilla_acc": n_corr / len(rows), "by_score": {}}
            for sk in SCORE_KEYS:
                cell_out["by_score"][sk] = {}
                for alpha in ALPHAS:
                    res = analyze_cell(rows, sk, alpha, n_boot=200, rng=rng)
                    if res is None:
                        continue
                    cell_out["by_score"][sk][f"alpha_{alpha}"] = res
                    inc = res["incorrect"]
                    print(
                        f"  {sk:8s} α={alpha}  "
                        f"viol_rate(incorrect)={inc['violation_rate']:.2f}  "
                        f"mean_t*={inc.get('mean_t_star')}  "
                        f"mean_t_worst={inc.get('mean_t_worst')}  "
                        f"mean_gap={inc.get('mean_gap_t_worst_minus_t_star')}  "
                        f"frac(t*<t_w | viol)={inc.get('frac_t_star_lt_t_worst_given_violation')}"
                    )
            out["cells"][cell_key] = cell_out

    OUTFILE.write_text(json.dumps(out, indent=2, default=str))

    # ---- aggregate summary across cells (focus on incorrect traces) ----
    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY (incorrect traces, alpha=0.30, score=lp)")
    print("=" * 78)
    print(f"{'cell':35s} {'n_inc':>6s} {'viol%':>7s} {'mean_t*':>8s} {'mean_tw':>8s} {'gap':>6s} {'t*<tw%':>7s}")
    rows_for_table = []
    for cell_key, cell in out["cells"].items():
        block = cell["by_score"].get("lp", {}).get("alpha_0.3")
        if not block:
            continue
        inc = block["incorrect"]
        rows_for_table.append({
            "cell": cell_key,
            "n_inc": inc["n"],
            "viol_rate": inc["violation_rate"],
            "mean_t_star": inc.get("mean_t_star"),
            "mean_t_worst": inc.get("mean_t_worst"),
            "mean_gap": inc.get("mean_gap_t_worst_minus_t_star"),
            "frac_lt": inc.get("frac_t_star_lt_t_worst_given_violation"),
        })
        ts = inc.get("mean_t_star")
        tw = inc.get("mean_t_worst")
        gap = inc.get("mean_gap_t_worst_minus_t_star")
        flt = inc.get("frac_t_star_lt_t_worst_given_violation")
        print(f"{cell_key:35s} {inc['n']:>6d} "
              f"{inc['violation_rate']*100:>6.1f}% "
              f"{(ts if ts is not None else float('nan')):>8.2f} "
              f"{(tw if tw is not None else float('nan')):>8.2f} "
              f"{(gap if gap is not None else float('nan')):>6.2f} "
              f"{(flt*100 if flt is not None else float('nan')):>6.1f}%")

    out["aggregate_summary_lp_alpha_0.3"] = rows_for_table
    OUTFILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[done] wrote {OUTFILE}")


if __name__ == "__main__":
    main()

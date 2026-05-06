"""
Pilot F: weighted CP on MATH→AIME OOD shift.

Vanilla split-CP assumes calibration ⊥ test exchangeability. Pilot B
showed that's broken under MATH→AIME shift (coverage off by 2-3x at
high alpha). Tibshirani-Foygel-Barber-Candès-Ramdas 2019 fix: weight each
calibration nonconformity by an estimated likelihood ratio
  w_i = p_test(x_i) / p_cal(x_i)
and use the weighted (1-α)-quantile.

Here we estimate the density ratio using a simple kernel density estimate
on the score distribution itself (one-dimensional). This is the simplest
weighted-CP variant we can run on existing data.

Reads:
  results/pilot7_math500_traces.jsonl    (cal source: MATH lp scores)
  results/pilotB_aime_greedy.jsonl       (test target: AIME lp scores)
Writes:
  results/pilotF_weighted_cp.json
  results/pilotF_density_ratio.png
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")


def step_lp(token_logprobs, boundaries):
    bdy = boundaries + [len(token_logprobs)]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            out.append(np.mean(seg))
    return out


def trajectory_scores(rows):
    out = []
    for r in rows:
        ss = step_lp(r["token_logprobs"], r["step_boundaries"])
        if not ss:
            continue
        out.append({
            "lp_mean":   float(np.mean(ss)),
            "lp_min":    float(np.min(ss)),
            "lp_median": float(np.median(ss)),
            "correct":   int(bool(r["correct"])),
        })
    return out


def estimate_density_ratio(cal_scores, test_scores, bandwidth=None):
    """1-D Gaussian KDE on each, return w(x) = p_test(x) / p_cal(x)."""
    from scipy.stats import gaussian_kde
    if bandwidth is None:
        # Silverman's rule
        bw_cal = 1.06 * np.std(cal_scores) * len(cal_scores) ** (-1/5)
        bw_test = 1.06 * np.std(test_scores) * len(test_scores) ** (-1/5)
    kde_cal = gaussian_kde(cal_scores, bw_method="silverman")
    kde_test = gaussian_kde(test_scores, bw_method="silverman")
    def ratio(x):
        return kde_test(x) / np.maximum(kde_cal(x), 1e-9)
    return ratio, kde_cal, kde_test


def vanilla_split_cp_quantile(cal_scores_correct, alpha):
    n = len(cal_scores_correct)
    ql = max(0.0, min(1.0, math.floor(alpha * (n + 1)) / n))
    return float(np.quantile(cal_scores_correct, ql))


def weighted_split_cp_quantile(cal_scores_correct, weights, alpha):
    """Weighted quantile of cal scores. weights normalized to sum to 1."""
    order = np.argsort(cal_scores_correct)
    s = cal_scores_correct[order]
    w = weights[order]
    w_norm = w / w.sum()
    cum = np.cumsum(w_norm)
    # find smallest s_i with cum_i >= alpha (weighted alpha-quantile from below)
    idx = np.searchsorted(cum, alpha)
    idx = min(idx, len(s) - 1)
    return float(s[idx])


def main():
    p7 = [json.loads(l) for l in (RESULTS / "pilot7_math500_traces.jsonl").read_text().splitlines() if l.strip()]
    pB = [json.loads(l) for l in (RESULTS / "pilotB_aime_greedy.jsonl").read_text().splitlines() if l.strip()]
    cal_recs = trajectory_scores(p7)
    te_recs = trajectory_scores(pB)
    print(f"MATH cal: {len(cal_recs)}  AIME test: {len(te_recs)}")

    summary = {"score_results": []}

    for sk in ["lp_mean", "lp_min", "lp_median"]:
        cal_scores = np.array([r[sk] for r in cal_recs])
        cal_correct = np.array([r["correct"] for r in cal_recs])
        te_scores = np.array([r[sk] for r in te_recs])
        te_correct = np.array([r["correct"] for r in te_recs])

        # Density ratio of test vs cal score distributions (over ALL trajectories)
        ratio_fn, kde_cal, kde_te = estimate_density_ratio(cal_scores, te_scores)
        # Weights for cal-correct points
        cal_corr_scores = cal_scores[cal_correct == 1]
        weights = ratio_fn(cal_corr_scores)
        weights = np.array(weights).flatten()
        # Numerical stability
        weights = np.clip(weights, 1e-6, 1e6)

        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
            q_vanilla = vanilla_split_cp_quantile(cal_corr_scores, alpha)
            q_weighted = weighted_split_cp_quantile(cal_corr_scores, weights, alpha)

            # Vanilla on test
            kept_v = te_scores >= q_vanilla
            n_corr_te = (te_correct == 1).sum()
            cov_v = float(((kept_v) & (te_correct == 1)).sum() / max(n_corr_te, 1))
            acc_v = float(te_correct[kept_v].mean()) if kept_v.sum() else float("nan")
            fr_v = float(kept_v.mean())

            # Weighted on test
            kept_w = te_scores >= q_weighted
            cov_w = float(((kept_w) & (te_correct == 1)).sum() / max(n_corr_te, 1))
            acc_w = float(te_correct[kept_w].mean()) if kept_w.sum() else float("nan")
            fr_w = float(kept_w.mean())

            summary["score_results"].append({
                "score": sk, "alpha": alpha, "target": 1 - alpha,
                "n_test": int(len(te_scores)),
                "vanilla":  {"q": q_vanilla,  "cov": cov_v, "kept_acc": acc_v, "kept_frac": fr_v},
                "weighted": {"q": q_weighted, "cov": cov_w, "kept_acc": acc_w, "kept_frac": fr_w},
            })
            print(f"[OOD {sk:9s} α={alpha:.2f}] target={1-alpha:.2f}  "
                  f"vanilla cov={cov_v:.2f} acc={acc_v:.2f} keep%={fr_v:.2f}  |  "
                  f"weighted cov={cov_w:.2f} acc={acc_w:.2f} keep%={fr_w:.2f}")

    # Plot density ratio for lp_mean
    cal_scores = np.array([r["lp_mean"] for r in cal_recs])
    te_scores = np.array([r["lp_mean"] for r in te_recs])
    ratio_fn, kde_cal, kde_te = estimate_density_ratio(cal_scores, te_scores)
    grid = np.linspace(min(cal_scores.min(), te_scores.min()),
                        max(cal_scores.max(), te_scores.max()), 200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(grid, kde_cal(grid), label="MATH-500 cal density", color="C0")
    axes[0].plot(grid, kde_te(grid), label="AIME test density",  color="C3")
    axes[0].set_xlabel("trajectory lp_mean score")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[0].set_title("Score densities: cal vs test")
    axes[1].plot(grid, ratio_fn(grid), color="C2")
    axes[1].set_xlabel("trajectory lp_mean score")
    axes[1].set_ylabel("p_test / p_cal")
    axes[1].set_title("Density ratio")
    axes[1].axhline(1, ls="--", c="gray")
    fig.tight_layout()
    fig.savefig(RESULTS / "pilotF_density_ratio.png", dpi=120)
    plt.close(fig)

    out = RESULTS / "pilotF_weighted_cp.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote: {out}  +  pilotF_density_ratio.png")


if __name__ == "__main__":
    main()

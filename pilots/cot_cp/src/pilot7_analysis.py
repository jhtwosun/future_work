"""Pilot 7-analysis: re-run pilot3-style reliability on MATH-500 traces."""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACES = Path("/home/nvidia/future/pilots/cot_cp/results/pilot7_math500_traces.jsonl")
RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")


def step_lp(token_logprobs, boundaries):
    n = len(token_logprobs)
    bdy = boundaries + [n]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            out.append(seg)
    return out


def main() -> None:
    rows = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} MATH-500 traces")

    per_traj = []
    n_steps_all = []
    for r in rows:
        steps = step_lp(r["token_logprobs"], r["step_boundaries"])
        if not steps:
            continue
        step_means = [float(np.mean(s)) for s in steps]
        per_traj.append({
            "id": r["id"],
            "correct": int(bool(r["correct"])),
            "lp_mean": float(np.mean(step_means)),
            "lp_min": float(np.min(step_means)),
            "lp_median": float(np.median(step_means)),
            "n_steps": len(step_means),
        })
        n_steps_all.append(len(step_means))

    valid = per_traj
    scores = np.array([p["lp_mean"] for p in valid])
    min_scores = np.array([p["lp_min"] for p in valid])
    correct = np.array([p["correct"] for p in valid])

    from scipy.stats import spearmanr, pointbiserialr
    rho_mean, p_mean = spearmanr(scores, correct)
    rho_min, p_min = spearmanr(min_scores, correct)
    rpb_mean, ppb_mean = pointbiserialr(scores, correct)
    rpb_min, ppb_min = pointbiserialr(min_scores, correct)

    # reliability bins (deciles)
    def reliab(arr, label):
        edges = np.quantile(arr, np.linspace(0, 1, 11))
        edges[0] = -np.inf
        edges[-1] = np.inf
        bins = []
        for a, b in zip(edges[:-1], edges[1:]):
            mask = (arr >= a) & (arr < b)
            if mask.sum() == 0:
                continue
            bins.append({
                "score_mean": float(arr[mask].mean()),
                "acc": float(correct[mask].mean()),
                "count": int(mask.sum()),
            })
        return bins

    rd_mean = reliab(scores, "lp_mean")
    rd_min = reliab(min_scores, "lp_min")

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, bins, t in [(axes[0], rd_mean, "MATH-500: lp_mean"),
                          (axes[1], rd_min, "MATH-500: lp_min (worst)")]:
        ax.plot([b["score_mean"] for b in bins], [b["acc"] for b in bins], "o-")
        for b in bins:
            ax.annotate(f"n={b['count']}", (b["score_mean"], b["acc"]),
                        fontsize=7, textcoords="offset points", xytext=(4, 4))
        ax.grid(True, alpha=0.3)
        ax.set_title(t)
        ax.set_xlabel("trajectory log-prob score")
        ax.set_ylabel("P(correct)")
        ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(RESULTS / "pilot7_reliability.png", dpi=120)
    plt.close(fig)

    # histogram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, arr, t in [(axes[0], scores, "lp_mean"), (axes[1], min_scores, "lp_min")]:
        bins = np.linspace(arr.min(), arr.max(), 30)
        ax.hist(arr[correct == 1], bins=bins, alpha=0.6, label="correct", color="C2")
        ax.hist(arr[correct == 0], bins=bins, alpha=0.6, label="incorrect", color="C3")
        ax.set_title(f"MATH-500: {t}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "pilot7_score_hist.png", dpi=120)
    plt.close(fig)

    out = {
        "n": len(per_traj),
        "accuracy": float(correct.mean()),
        "n_steps_per_traj": {
            "mean": float(np.mean(n_steps_all)),
            "median": float(np.median(n_steps_all)),
            "max": int(np.max(n_steps_all)),
        },
        "spearman": {
            "lp_mean": {"rho": float(rho_mean), "p": float(p_mean)},
            "lp_min":  {"rho": float(rho_min),  "p": float(p_min)},
        },
        "pointbiserial": {
            "lp_mean": {"r": float(rpb_mean), "p": float(ppb_mean)},
            "lp_min":  {"r": float(rpb_min),  "p": float(ppb_min)},
        },
        "reliability_lp_mean": rd_mean,
        "reliability_lp_min":  rd_min,
    }
    (RESULTS / "pilot7_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

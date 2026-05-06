"""
Pilot 3: Step-level confidence vs correctness reliability analysis.

Reads pilot2 traces, computes per-step S-LP (mean log-prob) and entropy
proxies, then asks two questions:

  Q1. Distribution: how do per-step scores look across the corpus?
  Q2. Weak-supervision calibration: do *trajectories with higher mean
      step S-LP* tend to produce correct final answers?
      (We do not have step-level gold labels here; we use final-answer
      correctness as a weak label per trajectory.)

Outputs:
  results/pilot3_step_stats.json     - aggregate statistics
  results/pilot3_per_traj.jsonl      - per-trajectory rolled-up scores
  results/pilot3_reliability.png     - reliability diagram (binned)
  results/pilot3_score_hist.png      - score distribution by correctness
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACES = Path("/home/nvidia/future/pilots/cot_cp/results/pilot2_gsm8k_traces.jsonl")
RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")
RESULTS.mkdir(parents=True, exist_ok=True)


def step_logprobs(token_logprobs: list[float], boundaries: list[int]) -> list[list[float]]:
    """Slice token-level log-probs into per-step lists."""
    n = len(token_logprobs)
    bdy = boundaries + [n]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]  # drop NaN
        if seg:
            out.append(seg)
    return out


def trajectory_scores(steps: list[list[float]]) -> dict:
    """Roll a list of per-step log-probs into trajectory-level summary."""
    if not steps:
        return {
            "mean_logprob": float("nan"),
            "min_step_mean_logprob": float("nan"),
            "n_steps": 0,
        }
    step_means = [float(np.mean(s)) for s in steps]
    return {
        "mean_logprob": float(np.mean(step_means)),
        "min_step_mean_logprob": float(np.min(step_means)),
        "max_step_mean_logprob": float(np.max(step_means)),
        "median_step_mean_logprob": float(np.median(step_means)),
        "n_steps": len(steps),
        "n_tokens": int(sum(len(s) for s in steps)),
    }


def reliability_diagram(scores: np.ndarray, correct: np.ndarray, out_path: Path,
                         title: str, n_bins: int = 10) -> dict:
    """Compute and plot calibration: bin trajectories by score, plot accuracy per bin."""
    if len(scores) < n_bins:
        n_bins = max(2, len(scores) // 5)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(scores, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf

    bin_acc, bin_score, bin_count = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (scores >= a) & (scores < b)
        if mask.sum() == 0:
            continue
        bin_acc.append(float(correct[mask].mean()))
        bin_score.append(float(scores[mask].mean()))
        bin_count.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(bin_score, bin_acc, "o-", label="empirical accuracy")
    ax.set_xlabel("Trajectory score (mean step log-prob)")
    ax.set_ylabel("P(correct final answer)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    for x, y, c in zip(bin_score, bin_acc, bin_count):
        ax.annotate(f"n={c}", (x, y), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    # Spearman-style correlation between score and correctness
    from scipy.stats import spearmanr, pointbiserialr
    if len(scores) >= 5:
        rho, p = spearmanr(scores, correct)
        rpb, ppb = pointbiserialr(scores, correct)
    else:
        rho = p = rpb = ppb = float("nan")

    return {
        "n_bins_used": len(bin_acc),
        "bin_score": bin_score,
        "bin_acc": bin_acc,
        "bin_count": bin_count,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "pointbiserial_r": float(rpb),
        "pointbiserial_p": float(ppb),
    }


def score_histogram(scores: np.ndarray, correct: np.ndarray, out_path: Path,
                     title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(scores.min(), scores.max(), 30)
    ax.hist(scores[correct == 1], bins=bins, alpha=0.55, label="correct", color="C2")
    ax.hist(scores[correct == 0], bins=bins, alpha=0.55, label="incorrect", color="C3")
    ax.set_xlabel("Trajectory score (mean step log-prob)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    if not TRACES.exists():
        raise SystemExit(f"missing traces: {TRACES}")

    rows = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} traces")

    per_traj = []
    n_steps_all, step_lens, step_means_flat = [], [], []
    for r in rows:
        steps = step_logprobs(r["token_logprobs"], r["step_boundaries"])
        score = trajectory_scores(steps)
        per_traj.append({
            "id": r["id"],
            "correct": int(r["correct"]),
            "predicted": r["predicted_value"],
            "gold": r["gold_value"],
            **score,
        })
        n_steps_all.append(score["n_steps"])
        step_lens.extend([len(s) for s in steps])
        step_means_flat.extend([float(np.mean(s)) for s in steps])

    with (RESULTS / "pilot3_per_traj.jsonl").open("w") as f:
        for p in per_traj:
            f.write(json.dumps(p) + "\n")

    valid = [p for p in per_traj if p["n_steps"] > 0 and not math.isnan(p["mean_logprob"])]
    scores = np.array([p["mean_logprob"] for p in valid])
    min_scores = np.array([p["min_step_mean_logprob"] for p in valid])
    correct = np.array([p["correct"] for p in valid])

    aggregate = {
        "n_traj": len(rows),
        "n_traj_valid": int(len(valid)),
        "accuracy": float(correct.mean()) if len(correct) else float("nan"),
        "n_steps_per_traj": {
            "mean": float(np.mean(n_steps_all)) if n_steps_all else 0,
            "median": float(np.median(n_steps_all)) if n_steps_all else 0,
            "min": int(np.min(n_steps_all)) if n_steps_all else 0,
            "max": int(np.max(n_steps_all)) if n_steps_all else 0,
        },
        "tokens_per_step": {
            "mean": float(np.mean(step_lens)) if step_lens else 0,
            "median": float(np.median(step_lens)) if step_lens else 0,
        },
        "step_mean_logprob_distribution": {
            "mean": float(np.mean(step_means_flat)) if step_means_flat else 0,
            "median": float(np.median(step_means_flat)) if step_means_flat else 0,
            "p10": float(np.percentile(step_means_flat, 10)) if step_means_flat else 0,
            "p90": float(np.percentile(step_means_flat, 90)) if step_means_flat else 0,
        },
    }

    rd_mean = reliability_diagram(
        scores, correct,
        RESULTS / "pilot3_reliability_mean.png",
        title="Reliability: trajectory mean step log-prob vs final-answer correctness",
    )
    rd_min = reliability_diagram(
        min_scores, correct,
        RESULTS / "pilot3_reliability_min.png",
        title="Reliability: WORST step mean log-prob vs correctness",
    )

    score_histogram(
        scores, correct,
        RESULTS / "pilot3_score_hist_mean.png",
        title="Mean step log-prob distribution by correctness",
    )
    score_histogram(
        min_scores, correct,
        RESULTS / "pilot3_score_hist_min.png",
        title="Min step log-prob (worst-step) distribution by correctness",
    )

    aggregate["reliability_mean_score"] = rd_mean
    aggregate["reliability_min_score"] = rd_min

    out = RESULTS / "pilot3_step_stats.json"
    out.write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()

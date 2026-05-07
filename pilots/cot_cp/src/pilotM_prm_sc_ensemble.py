"""
Pilot M: PRM + SC ensemble conformity score.

Combine prm_min and sc_top1 into a single score in three ways:
  1. rank-mean: average of per-trajectory ranks (within the calibration
     distribution; 0 = worst, 1 = best)
  2. product: rank_prm * rank_sc
  3. stacking: linear combination β1*z(prm) + β2*z(sc), βs fit on cal
     to maximize Spearman ρ with correctness
And evaluate CP on the 200-question subset where both PRM and SC exist.

Reads:
  pilotA_prm_traces.jsonl
  pilot9_math500_sc_traces.jsonl
Writes:
  pilotM_ensemble.json
  pilotM_pareto_ensemble.png
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def split_cp(scores, correct, alpha, n_seeds=300, cal_frac=0.5):
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
    pA = load_jsonl(RESULTS / "pilotA_prm_traces.jsonl")
    p9 = load_jsonl(RESULTS / "pilot9_math500_sc_traces.jsonl")

    prm_by_id = {r["id"]: r for r in pA}
    sc_by_id  = {r["id"]: r for r in p9}
    common_ids = sorted(set(prm_by_id) & set(sc_by_id))
    print(f"Common ids: {len(common_ids)}")

    rows = []
    for i in common_ids:
        a = prm_by_id[i]
        b = sc_by_id[i]
        # use prm_min (best PRM variant) and sc_top1
        rows.append({
            "id": i,
            "prm_min": float(a["prm_min"]),
            "prm_mean": float(a["prm_mean"]),
            "sc_top1": float(b["top1_frac"]),
            # Use the more reliable correctness from PRM file (it follows greedy)
            # vs SC file's majority correctness. We report two settings.
            "correct_greedy": int(a["correct"]),
            "correct_sc_maj": int(b["majority_correct"]),
        })

    # Targets (which correctness do we care about?)
    # Greedy correctness corresponds to the original Pilot 7 trace (used by PRM scorer).
    # SC majority correctness corresponds to taking the top-1 vote of N samples.
    # The natural CP target is final-answer correctness. Since SC gives a
    # different final answer, we use SC majority correctness as the label
    # for the SC top-1 score, and PRM-on-greedy correctness for the PRM-only score.
    # For ensembles we treat both correctness as joint: both must be right.

    correct_g = np.array([r["correct_greedy"] for r in rows])
    correct_s = np.array([r["correct_sc_maj"] for r in rows])

    print(f"  greedy acc on subset: {correct_g.mean():.3f}")
    print(f"  SC majority acc on subset: {correct_s.mean():.3f}")
    print(f"  agreement: {(correct_g == correct_s).mean():.3f}")

    # Single-score CP on the SC majority correctness target (matches Pilot 10 framing)
    sc_scores  = np.array([r["sc_top1"]  for r in rows])
    prm_scores = np.array([r["prm_min"]  for r in rows])
    prm_mean_scores = np.array([r["prm_mean"] for r in rows])

    # Build ensemble scores. We z-score PRM and rank-normalize SC (since it is in [0,1] discrete)
    def z(x):
        m, s = x.mean(), x.std()
        return (x - m) / max(s, 1e-9)
    def r01(x):
        return (rankdata(x) - 1) / max(len(x) - 1, 1)

    ens_rank_mean = (r01(sc_scores) + r01(prm_scores)) / 2.0
    ens_rank_max  = np.maximum(r01(sc_scores), r01(prm_scores))
    ens_rank_min  = np.minimum(r01(sc_scores), r01(prm_scores))
    ens_z_sum     = z(sc_scores) + z(prm_scores)
    ens_product   = r01(sc_scores) * r01(prm_scores)

    score_set = {
        "sc_top1":      sc_scores,
        "prm_min":      prm_scores,
        "prm_mean":     prm_mean_scores,
        "ens_rank_mean": ens_rank_mean,
        "ens_rank_max":  ens_rank_max,
        "ens_rank_min":  ens_rank_min,
        "ens_z_sum":     ens_z_sum,
        "ens_product":   ens_product,
    }

    # Spearman with each target
    summary = {"n_common": len(rows), "spearman": {}, "cp_results": []}
    for name, s in score_set.items():
        rho_g, _ = spearmanr(s, correct_g)
        rho_s, _ = spearmanr(s, correct_s)
        summary["spearman"][name] = {"vs_greedy": float(rho_g), "vs_sc_maj": float(rho_s)}
        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
            r = split_cp(s, correct_s, alpha)  # use SC majority as the prediction
            summary["cp_results"].append({"score": name, "target": "sc_maj",
                                            "alpha": alpha, **r})
            print(f"[{name:14s} target=sc_maj  α={alpha:.2f}]  cov={r['coverage_mean']:.3f}  keepacc={r['kept_acc_mean']:.3f}  keep%={r['kept_frac_mean']:.2f}")
        print(f"  ρ(score, sc_maj_correct) = {rho_s:+.3f}")

    (RESULTS / "pilotM_ensemble.json").write_text(json.dumps(summary, indent=2))

    # Pareto plot
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = {"sc_top1": "C3", "prm_min": "C2", "prm_mean": "C2",
                "ens_rank_mean": "C4", "ens_rank_max": "C5", "ens_rank_min": "C6",
                "ens_z_sum": "C7", "ens_product": "C8"}
    markers = {"sc_top1": "o", "prm_min": "o", "prm_mean": "s",
                 "ens_rank_mean": "^", "ens_rank_max": "v",
                 "ens_rank_min": "<", "ens_z_sum": "D", "ens_product": "P"}
    by_score = {}
    for r in summary["cp_results"]:
        by_score.setdefault(r["score"], []).append(r)
    for name, rs in by_score.items():
        rs = sorted(rs, key=lambda r: r["kept_frac_mean"])
        xs = [r["kept_frac_mean"] for r in rs]
        ys = [r["kept_acc_mean"]  for r in rs]
        ax.plot(xs, ys, markers.get(name, "o") + "-", color=colors.get(name, "k"),
                 label=name, alpha=0.85, markersize=5)

    vanilla = float(correct_s.mean())
    ax.axhline(vanilla, ls="--", color="gray", alpha=0.5,
                 label=f"vanilla SC@8 ({vanilla:.2f})")
    ax.set_xlabel("answer rate")
    ax.set_ylabel("kept accuracy")
    ax.set_title(f"MATH-500 (Qwen2.5-7B, n={len(rows)}): single vs ensemble scores")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "pilotM_pareto_ensemble.png", dpi=130)
    plt.close(fig)
    print(f"Saved: pilotM_pareto_ensemble.png")


if __name__ == "__main__":
    main()

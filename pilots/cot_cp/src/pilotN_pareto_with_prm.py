"""
Pilot N: re-render the headline Pareto figure with PRM-min as a third
score family alongside SC and lp variants on MATH-500.

Reads:
  pilot7_math500_traces.jsonl   (lp scores)
  pilot9_math500_sc_traces.jsonl (sc top1)
  pilotA_prm_traces.jsonl       (prm scores)

Writes:
  pilotN_pareto_with_prm.png
  pilotN_summary.json
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


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
    p7 = load_jsonl(RESULTS / "pilot7_math500_traces.jsonl")
    p9 = load_jsonl(RESULTS / "pilot9_math500_sc_traces.jsonl")
    pA = load_jsonl(RESULTS / "pilotA_prm_traces.jsonl")

    # LP records on MATH-500
    lp_recs = []
    for r in p7:
        ss = step_lp(r["token_logprobs"], r["step_boundaries"])
        if not ss:
            continue
        lp_recs.append({
            "id": r["id"],
            "lp_mean":   float(np.mean(ss)),
            "lp_min":    float(np.min(ss)),
            "lp_median": float(np.median(ss)),
            "correct":   int(bool(r["correct"])),
        })

    sc_recs = [{"id": r["id"], "sc_top1": float(r["top1_frac"]),
                 "correct": int(r["majority_correct"])} for r in p9]

    prm_recs = [{"id": r["id"], "prm_min": float(r["prm_min"]),
                  "prm_mean": float(r["prm_mean"]),
                  "correct": int(r["correct"])} for r in pA]

    print(f"LP: {len(lp_recs)}, SC: {len(sc_recs)}, PRM: {len(prm_recs)}")

    alphas = np.linspace(0.05, 0.7, 14)
    rows = []
    series = {
        "lp_min":   ([r["lp_min"]   for r in lp_recs],  [r["correct"] for r in lp_recs]),
        "lp_mean":  ([r["lp_mean"]  for r in lp_recs],  [r["correct"] for r in lp_recs]),
        "prm_min":  ([r["prm_min"]  for r in prm_recs], [r["correct"] for r in prm_recs]),
        "prm_mean": ([r["prm_mean"] for r in prm_recs], [r["correct"] for r in prm_recs]),
        "sc_top1":  ([r["sc_top1"]  for r in sc_recs],  [r["correct"] for r in sc_recs]),
    }
    for name, (s, c) in series.items():
        s = np.array(s); c = np.array(c)
        for a in alphas:
            r = split_cp(s, c, float(a))
            rows.append({"score": name, "alpha": float(a), **r})

    out = {"alphas": list(alphas), "rows": rows,
            "vanilla_acc_lp_set":  float(np.mean([r["correct"] for r in lp_recs])),
            "vanilla_acc_sc_set":  float(np.mean([r["correct"] for r in sc_recs])),
            "vanilla_acc_prm_set": float(np.mean([r["correct"] for r in prm_recs]))}
    (RESULTS / "pilotN_summary.json").write_text(json.dumps(out, indent=2))

    # Pareto plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    color_for = {
        "lp_min":   ("C0", "o", "lp_min  (free)"),
        "lp_mean":  ("C0", "s", "lp_mean (free)"),
        "prm_min":  ("C2", "o", "prm_min (Qwen2.5-Math-PRM-7B, 2× cost)"),
        "prm_mean": ("C2", "s", "prm_mean (2× cost)"),
        "sc_top1":  ("C3", "o", "sc_top1 (SC@8, 8× cost)"),
    }
    for name, (color, marker, label) in color_for.items():
        rs = [r for r in rows if r["score"] == name]
        rs.sort(key=lambda r: r["kept_frac_mean"])
        xs = [r["kept_frac_mean"] for r in rs]
        ys = [r["kept_acc_mean"]  for r in rs]
        ax.plot(xs, ys, marker + "-", color=color, label=label, alpha=0.85, markersize=6)

    # Vanilla baselines
    ax.axhline(out["vanilla_acc_lp_set"],  ls="--", color="gray", alpha=0.5,
                 label=f"vanilla MATH-500 ({out['vanilla_acc_lp_set']:.2f})")
    ax.set_xlabel("answer rate (kept fraction)")
    ax.set_ylabel("accuracy among kept")
    ax.set_title("MATH-500 (Qwen2.5-7B-Instruct): selective accuracy by score family")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(0.45, 1.0)
    fig.tight_layout()
    fig.savefig(RESULTS / "pilotN_pareto_with_prm.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {RESULTS / 'pilotN_pareto_with_prm.png'}")


if __name__ == "__main__":
    main()

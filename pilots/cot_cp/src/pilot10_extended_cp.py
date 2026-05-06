"""
Pilot 10: extended CP simulation = pilot 8 + MATH-500 SC top1_frac.

Goal: produce a single Pareto-style table with rows
  (dataset, score family, alpha) -> (kept%, kept_acc, empirical_coverage).
"""

import json
import math
from pathlib import Path

import numpy as np

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def build_lp_scores(traces, correct_field="correct"):
    out = []
    for r in traces:
        bdy = r["step_boundaries"] + [len(r["token_logprobs"])]
        steps = []
        for a, b in zip(bdy[:-1], bdy[1:]):
            seg = [lp for lp in r["token_logprobs"][a:b] if not (lp != lp)]
            if seg:
                steps.append(np.mean(seg))
        if not steps:
            continue
        out.append({
            "id": r["id"],
            "lp_mean":   float(np.mean(steps)),
            "lp_min":    float(np.min(steps)),
            "lp_median": float(np.median(steps)),
            "lp_tok":    float(np.mean([x for x in r["token_logprobs"] if not (x != x)])),
            "n_steps":   len(steps),
            "correct":   int(bool(r[correct_field])),
        })
    return out


def split_cp(records, score_key, alpha, n_seeds=200, cal_frac=0.5):
    s = np.array([r[score_key] for r in records])
    c = np.array([r["correct"] for r in records])
    rng = np.random.default_rng(0)
    covs, accs, fracs, qhs = [], [], [], []
    for seed in range(n_seeds):
        idx = rng.permutation(len(s))
        n_cal = int(cal_frac * len(s))
        ci, ti = idx[:n_cal], idx[n_cal:]
        cal_corr_scores = s[ci][c[ci] == 1]
        if len(cal_corr_scores) < 5:
            continue
        n = len(cal_corr_scores)
        q_level = math.floor(alpha * (n + 1)) / n
        q_level = max(0.0, min(1.0, q_level))
        q = float(np.quantile(cal_corr_scores, q_level))
        kept = s[ti] >= q
        n_correct_te = (c[ti] == 1).sum()
        if n_correct_te == 0:
            continue
        cov = float((kept & (c[ti] == 1)).sum() / n_correct_te)
        acc = float(c[ti][kept].mean()) if kept.sum() else float("nan")
        fr  = float(kept.mean())
        covs.append(cov); accs.append(acc); fracs.append(fr); qhs.append(q)
    return {
        "score": score_key, "alpha": alpha, "n_seeds": len(covs),
        "target_coverage": 1 - alpha,
        "empirical_coverage_mean": float(np.mean(covs)) if covs else float("nan"),
        "empirical_coverage_std":  float(np.std(covs))  if covs else float("nan"),
        "kept_acc_mean":  float(np.mean(accs))  if accs  else float("nan"),
        "kept_frac_mean": float(np.mean(fracs)) if fracs else float("nan"),
        "q_hat_mean":     float(np.mean(qhs))   if qhs   else float("nan"),
    }


def main():
    # GSM8K
    p2 = load_jsonl(RESULTS / "pilot2_gsm8k_traces.jsonl")
    gsm_lp = build_lp_scores(p2, "correct")
    p4 = load_jsonl(RESULTS / "pilot4_sc_traces.jsonl")
    gsm_sc = [{"id": r["id"], "sc_top1": float(r["top1_frac"]),
                "correct": int(r["majority_correct"])} for r in p4]
    # MATH-500
    p7 = load_jsonl(RESULTS / "pilot7_math500_traces.jsonl")
    math_lp = build_lp_scores(p7, "correct")
    p9 = load_jsonl(RESULTS / "pilot9_math500_sc_traces.jsonl")
    math_sc = [{"id": r["id"], "sc_top1": float(r["top1_frac"]),
                 "correct": int(r["majority_correct"])} for r in p9]

    print(f"GSM8K LP={len(gsm_lp)} acc={np.mean([r['correct'] for r in gsm_lp]):.3f}")
    print(f"GSM8K SC={len(gsm_sc)} acc={np.mean([r['correct'] for r in gsm_sc]):.3f}")
    print(f"MATH  LP={len(math_lp)} acc={np.mean([r['correct'] for r in math_lp]):.3f}")
    print(f"MATH  SC={len(math_sc)} acc={np.mean([r['correct'] for r in math_sc]):.3f}")

    rows = []
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for ds_name, recs, score_keys in [
        ("GSM8K-LP", gsm_lp, ["lp_mean", "lp_min", "lp_median", "lp_tok"]),
        ("GSM8K-SC", gsm_sc, ["sc_top1"]),
        ("MATH500-LP", math_lp, ["lp_mean", "lp_min", "lp_median", "lp_tok"]),
        ("MATH500-SC", math_sc, ["sc_top1"]),
    ]:
        for sk in score_keys:
            for alpha in alphas:
                r = split_cp(recs, sk, alpha)
                row = {"dataset": ds_name, **r}
                rows.append(row)
                print(f"[{ds_name:11s} {sk:9s} α={alpha:.2f}] cov={r['empirical_coverage_mean']:.3f}±{r['empirical_coverage_std']:.3f} keepacc={r['kept_acc_mean']:.3f} keep%={r['kept_frac_mean']:.2f}")

    out = RESULTS / "pilot10_extended_cp.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote: {out}")

    # also produce a Pareto plot per dataset
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ds_prefix, vanilla in [
        (axes[0], "GSM8K", 0.903),
        (axes[1], "MATH500", 0.516),
    ]:
        for score in ["lp_mean", "lp_min", "lp_median", "lp_tok", "sc_top1"]:
            xs, ys = [], []
            for r in rows:
                if not r["dataset"].startswith(ds_prefix):
                    continue
                if r["score"] != score:
                    continue
                xs.append(r["kept_frac_mean"])
                ys.append(r["kept_acc_mean"])
            if xs:
                order = np.argsort(xs)
                ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", label=score)
        ax.axhline(vanilla, ls="--", c="gray", label=f"vanilla ({vanilla:.2f})")
        ax.set_xlabel("kept fraction (answer rate)")
        ax.set_ylabel("accuracy among kept")
        ax.set_title(f"Selective accuracy: {ds_prefix}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "pilot10_pareto.png", dpi=120)
    plt.close(fig)
    print("Saved: pilot10_pareto.png")


if __name__ == "__main__":
    main()

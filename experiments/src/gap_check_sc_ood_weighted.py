"""
Gap-check: does weighted CP fix sc_top1 OOD under-coverage on AIME-extended?

Pilot J / Pilot I showed:
- Vanilla CP on sc_top1: AIME-extended n=300, target 0.5 → empirical 0.27
- Weighted CP on lp_min: target 0.5 → empirical 0.50 (fixed!)
- BUT we never tested weighted CP on sc_top1 directly.

The catch: sc_top1 is discrete (∈ {0/8, 1/8, ..., 8/8} = 9 levels for SC@8).
Density-ratio KDE may not work well; we'll use empirical density ratio
(probability mass) instead.

Reads:
  experiments/results/E1_math500_sc8_traces.jsonl  (cal: MATH SC)
  experiments/results/E2_aime_sc8_traces.jsonl     (test: AIME SC, n=933)
Writes:
  experiments/results/gap_sc_ood_weighted.json
  experiments/results/gap_sc_ood_weighted.png
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import equal_strict, normalize, extract_pred

EXP = Path("/home/nvidia/future/experiments/results")


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def sc_records_robust(rows):
    """Re-evaluate correctness using robust extractor on majority pred."""
    out = []
    for r in rows:
        ok = equal_strict(r.get("majority_pred"), r.get("gold"))
        out.append({"sc_top1": float(r.get("top1_frac", 0.0)),
                     "correct": int(ok)})
    return out


def vanilla_cp_quantile(cal_correct_scores, alpha):
    n = len(cal_correct_scores)
    if n < 5: return None
    ql = max(0.0, min(1.0, math.floor(alpha * (n + 1)) / n))
    return float(np.quantile(cal_correct_scores, ql))


def empirical_pmf(scores, levels):
    """Empirical PMF over discrete `levels`."""
    counter = Counter(scores)
    return {l: counter[l] / max(len(scores), 1) for l in levels}


def weighted_cp_quantile_discrete(cal_correct_scores, weights, alpha):
    """Weighted lower alpha-quantile using empirical density-ratio weights."""
    order = np.argsort(cal_correct_scores)
    s = np.array(cal_correct_scores)[order]
    w = np.array(weights)[order]
    w_norm = w / max(w.sum(), 1e-9)
    cum = np.cumsum(w_norm)
    idx = min(np.searchsorted(cum, alpha), len(s) - 1)
    return float(s[idx])


def evaluate_coverage(scores, correct, q):
    scores = np.asarray(scores); correct = np.asarray(correct)
    kept = scores >= q
    n_corr = (correct == 1).sum()
    if n_corr == 0:
        return None
    cov = float(((kept) & (correct == 1)).sum() / n_corr)
    acc = float(correct[kept].mean()) if kept.sum() else float("nan")
    return {"coverage": cov, "kept_acc": acc, "kept_frac": float(kept.mean())}


def bootstrap_ci(scores, correct, q, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(scores)
    boot_cov = []
    scores = np.asarray(scores); correct = np.asarray(correct)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = scores[idx]; c = correct[idx]
        nc = (c == 1).sum()
        if nc == 0: continue
        kept = s >= q
        boot_cov.append(((kept) & (c == 1)).sum() / nc)
    if not boot_cov: return [float("nan"), float("nan")]
    return [float(np.quantile(boot_cov, 0.025)), float(np.quantile(boot_cov, 0.975))]


def main():
    cal_path = EXP / "E1_math500_sc8_traces.jsonl"
    test_path = EXP / "E2_aime_sc8_traces.jsonl"
    if not cal_path.exists() or not test_path.exists():
        raise SystemExit("missing E1 or E2 SC traces")

    cal_recs = sc_records_robust(load_jsonl(cal_path))
    test_recs = sc_records_robust(load_jsonl(test_path))
    print(f"Cal (MATH-500 SC@8): n={len(cal_recs)}, acc={np.mean([r['correct'] for r in cal_recs]):.3f}")
    print(f"Test (AIME SC@8): n={len(test_recs)}, acc={np.mean([r['correct'] for r in test_recs]):.3f}")

    cal_scores = [r["sc_top1"] for r in cal_recs]
    cal_correct = [r["correct"] for r in cal_recs]
    test_scores = [r["sc_top1"] for r in test_recs]
    test_correct = [r["correct"] for r in test_recs]

    # discrete score levels for SC@8
    levels = [k / 8 for k in range(9)]  # 0/8, 1/8, ..., 8/8
    pmf_cal = empirical_pmf(cal_scores, levels)
    pmf_test = empirical_pmf(test_scores, levels)
    print(f"Cal PMF: {pmf_cal}")
    print(f"Test PMF: {pmf_test}")

    # weights for cal-correct points: w(x) = p_test(x) / p_cal(x) at x = score level
    cal_correct_scores = [s for s, c in zip(cal_scores, cal_correct) if c == 1]
    weights = []
    for s in cal_correct_scores:
        p_t = pmf_test.get(s, 1e-6)
        p_c = pmf_cal.get(s, 1e-6)
        w = p_t / max(p_c, 1e-9)
        # clip extremes
        w = float(np.clip(w, 1e-3, 1e3))
        weights.append(w)

    summary = {"alphas": [], "vanilla_cov": [], "vanilla_ci": [],
                "weighted_cov": [], "weighted_ci": [],
                "vanilla_kept_acc": [], "weighted_kept_acc": [],
                "vanilla_kept_frac": [], "weighted_kept_frac": [],
                "q_vanilla": [], "q_weighted": []}
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        target = 1 - alpha
        q_v = vanilla_cp_quantile(cal_correct_scores, alpha)
        q_w = weighted_cp_quantile_discrete(cal_correct_scores, weights, alpha)

        ev_v = evaluate_coverage(test_scores, test_correct, q_v)
        ev_w = evaluate_coverage(test_scores, test_correct, q_w)
        ci_v = bootstrap_ci(test_scores, test_correct, q_v)
        ci_w = bootstrap_ci(test_scores, test_correct, q_w)

        summary["alphas"].append(alpha)
        summary["vanilla_cov"].append(ev_v["coverage"])
        summary["vanilla_ci"].append(ci_v)
        summary["weighted_cov"].append(ev_w["coverage"])
        summary["weighted_ci"].append(ci_w)
        summary["vanilla_kept_acc"].append(ev_v["kept_acc"])
        summary["weighted_kept_acc"].append(ev_w["kept_acc"])
        summary["vanilla_kept_frac"].append(ev_v["kept_frac"])
        summary["weighted_kept_frac"].append(ev_w["kept_frac"])
        summary["q_vanilla"].append(q_v)
        summary["q_weighted"].append(q_w)

        print(f"[α={alpha:.2f} target={target:.2f}]"
                f"  van cov={ev_v['coverage']:.3f} CI=[{ci_v[0]:.3f},{ci_v[1]:.3f}] q={q_v:.3f}"
                f"  wgt cov={ev_w['coverage']:.3f} CI=[{ci_w[0]:.3f},{ci_w[1]:.3f}] q={q_w:.3f}"
                f"  van keep%={ev_v['kept_frac']:.2f}  wgt keep%={ev_w['kept_frac']:.2f}")

    out_json = EXP / "gap_sc_ood_weighted.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(summary["alphas"], [1 - a for a in summary["alphas"]], "k--",
              label="target $1-\\alpha$")
    van_cov = np.array(summary["vanilla_cov"])
    wgt_cov = np.array(summary["weighted_cov"])
    van_ci = np.array(summary["vanilla_ci"])
    wgt_ci = np.array(summary["weighted_ci"])
    ax1.errorbar(summary["alphas"], van_cov,
                  yerr=[van_cov - van_ci[:, 0], van_ci[:, 1] - van_cov],
                  fmt="o-", color="C3", label="vanilla CP", capsize=3)
    ax1.errorbar(summary["alphas"], wgt_cov,
                  yerr=[wgt_cov - wgt_ci[:, 0], wgt_ci[:, 1] - wgt_cov],
                  fmt="s-", color="C2", label="weighted CP", capsize=3)
    ax1.set_xlabel("$\\alpha$")
    ax1.set_ylabel("empirical coverage")
    ax1.set_title("AIME-extended (n=933) OOD coverage: vanilla vs weighted")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # PMF visualization
    levels_arr = np.arange(len(levels))
    cal_pmf_arr = [pmf_cal[l] for l in levels]
    test_pmf_arr = [pmf_test[l] for l in levels]
    width = 0.35
    ax2.bar(levels_arr - width/2, cal_pmf_arr, width, label="MATH-500 cal", color="C0")
    ax2.bar(levels_arr + width/2, test_pmf_arr, width, label="AIME test", color="C3")
    ax2.set_xticks(levels_arr)
    ax2.set_xticklabels([f"{int(l*8)}/8" for l in levels])
    ax2.set_xlabel("sc_top1 level")
    ax2.set_ylabel("empirical PMF")
    ax2.set_title("sc_top1 distribution: cal vs test (discrete)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(EXP / "gap_sc_ood_weighted.png", dpi=130)
    plt.close(fig)
    print(f"Wrote: {out_json}, gap_sc_ood_weighted.png")


if __name__ == "__main__":
    main()

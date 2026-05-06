"""
Pilot J: bootstrap 95% CIs on coverage and kept_acc for headline numbers.

For each (dataset, score, alpha) operating point that we want in the
paper, repeatedly resample the test set with replacement and recompute
coverage and kept_acc. This gives proper CIs that the previous
n_seeds=200 random splits did not (those varied cal/test splits but did
not bootstrap within a fixed test set).

Operating points:
  - GSM8K + lp_min, α=0.1
  - GSM8K + sc_top1, α=0.1
  - MATH-500 + lp_min, α=0.5
  - MATH-500 + sc_top1, α=0.3
  - MATH-500 + sc_top1, α=0.5
  - R1-Distill MATH-500 + lp_mean, α=0.5
  - AIME OOD + lp_min, α=0.5  (vanilla and weighted)
"""

import json
import math
from pathlib import Path

import numpy as np

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


def lp_records(rows):
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


def sc_records(rows, correct_field="majority_correct"):
    return [{"sc_top1": float(r["top1_frac"]),
              "correct": int(r[correct_field])} for r in rows]


def bootstrap_cp(records, score_key, alpha, n_boot=500, n_split=10, cal_frac=0.5, seed=0):
    """For each bootstrap resample of records, do n_split random cal/test splits
    and average. Then take quantiles of those averages across bootstraps."""
    rng = np.random.default_rng(seed)
    rs = np.array([r[score_key] for r in records])
    cs = np.array([r["correct"] for r in records])
    n = len(rs)
    boot_acc, boot_keep, boot_cov = [], [], []
    for _ in range(n_boot):
        # bootstrap sample
        idx_b = rng.integers(0, n, size=n)
        rs_b = rs[idx_b]
        cs_b = cs[idx_b]
        accs, keeps, covs = [], [], []
        for _ in range(n_split):
            perm = rng.permutation(n)
            n_cal = int(cal_frac * n)
            ci, ti = perm[:n_cal], perm[n_cal:]
            cal_correct_scores = rs_b[ci][cs_b[ci] == 1]
            if len(cal_correct_scores) < 5:
                continue
            n_c = len(cal_correct_scores)
            ql = max(0.0, min(1.0, math.floor(alpha * (n_c + 1)) / n_c))
            q = float(np.quantile(cal_correct_scores, ql))
            kept = rs_b[ti] >= q
            n_corr = (cs_b[ti] == 1).sum()
            if n_corr == 0:
                continue
            cov = float(((kept) & (cs_b[ti] == 1)).sum() / n_corr)
            acc = float(cs_b[ti][kept].mean()) if kept.sum() else float("nan")
            keep = float(kept.mean())
            accs.append(acc); keeps.append(keep); covs.append(cov)
        if accs:
            boot_acc.append(np.mean(accs))
            boot_keep.append(np.mean(keeps))
            boot_cov.append(np.mean(covs))
    if not boot_acc:
        return None
    boot_acc = np.array(boot_acc)
    boot_keep = np.array(boot_keep)
    boot_cov = np.array(boot_cov)
    return {
        "n_boot": len(boot_acc),
        "kept_acc":  {"mean": float(boot_acc.mean()),
                       "ci95": [float(np.quantile(boot_acc, 0.025)), float(np.quantile(boot_acc, 0.975))]},
        "kept_frac": {"mean": float(boot_keep.mean()),
                       "ci95": [float(np.quantile(boot_keep, 0.025)), float(np.quantile(boot_keep, 0.975))]},
        "coverage":  {"mean": float(boot_cov.mean()),
                       "ci95": [float(np.quantile(boot_cov, 0.025)), float(np.quantile(boot_cov, 0.975))]},
    }


def bootstrap_ood(cal_records, te_records, score_key, alpha, n_boot=500, seed=0,
                   weighted=False):
    """Bootstrap test set; cal stays fixed."""
    rng = np.random.default_rng(seed)
    cal_scores = np.array([r[score_key] for r in cal_records])
    cal_correct = np.array([r["correct"] for r in cal_records])
    te_scores = np.array([r[score_key] for r in te_records])
    te_correct = np.array([r["correct"] for r in te_records])
    cal_corr_scores = cal_scores[cal_correct == 1]

    if weighted:
        from scipy.stats import gaussian_kde
        kde_cal = gaussian_kde(cal_scores, bw_method="silverman")
        kde_te = gaussian_kde(te_scores, bw_method="silverman")
        weights = (kde_te(cal_corr_scores) / np.maximum(kde_cal(cal_corr_scores), 1e-9)).flatten()
        weights = np.clip(weights, 1e-6, 1e6)
        order = np.argsort(cal_corr_scores)
        s = cal_corr_scores[order]
        w = weights[order]
        w_norm = w / w.sum()
        cum = np.cumsum(w_norm)
        idx = min(np.searchsorted(cum, alpha), len(s) - 1)
        q = float(s[idx])
    else:
        n_c = len(cal_corr_scores)
        ql = max(0.0, min(1.0, math.floor(alpha * (n_c + 1)) / n_c))
        q = float(np.quantile(cal_corr_scores, ql))

    n = len(te_scores)
    boot_cov, boot_acc, boot_keep = [], [], []
    for _ in range(n_boot):
        idx_b = rng.integers(0, n, size=n)
        ts = te_scores[idx_b]
        tc = te_correct[idx_b]
        kept = ts >= q
        n_corr = (tc == 1).sum()
        if n_corr == 0:
            continue
        cov = float(((kept) & (tc == 1)).sum() / n_corr)
        acc = float(tc[kept].mean()) if kept.sum() else float("nan")
        keep = float(kept.mean())
        boot_cov.append(cov); boot_acc.append(acc); boot_keep.append(keep)
    if not boot_cov:
        return None
    return {
        "q_hat": q,
        "n_boot": len(boot_cov),
        "kept_acc":  {"mean": float(np.nanmean(boot_acc)),
                       "ci95": [float(np.nanquantile(boot_acc, 0.025)),
                                float(np.nanquantile(boot_acc, 0.975))]},
        "kept_frac": {"mean": float(np.mean(boot_keep)),
                       "ci95": [float(np.quantile(boot_keep, 0.025)),
                                float(np.quantile(boot_keep, 0.975))]},
        "coverage":  {"mean": float(np.mean(boot_cov)),
                       "ci95": [float(np.quantile(boot_cov, 0.025)),
                                float(np.quantile(boot_cov, 0.975))]},
    }


def main():
    p2 = load_jsonl(RESULTS / "pilot2_gsm8k_traces.jsonl")
    p4 = load_jsonl(RESULTS / "pilot4_sc_traces.jsonl")
    p7 = load_jsonl(RESULTS / "pilot7_math500_traces.jsonl")
    p9 = load_jsonl(RESULTS / "pilot9_math500_sc_traces.jsonl")
    pE = load_jsonl(RESULTS / "pilotE_r1_greedy_traces.jsonl")
    pBg = load_jsonl(RESULTS / "pilotB_aime_greedy.jsonl")

    gsm_lp = lp_records(p2)
    gsm_sc = sc_records(p4)
    math_lp = lp_records(p7)
    math_sc = sc_records(p9)
    r1_lp = lp_records(pE)
    aime_lp = lp_records(pBg)

    print(f"sizes: GSM-LP={len(gsm_lp)}, GSM-SC={len(gsm_sc)}, MATH-LP={len(math_lp)}, MATH-SC={len(math_sc)}, R1-LP={len(r1_lp)}, AIME-LP={len(aime_lp)}")

    out = {"operating_points": []}

    points = [
        ("GSM8K", gsm_lp, "lp_min", 0.1),
        ("GSM8K", gsm_sc, "sc_top1", 0.1),
        ("MATH-500", math_lp, "lp_min", 0.5),
        ("MATH-500", math_sc, "sc_top1", 0.3),
        ("MATH-500", math_sc, "sc_top1", 0.5),
        ("MATH-500-R1", r1_lp, "lp_mean", 0.5),
        ("MATH-500-R1", r1_lp, "lp_min", 0.5),
    ]
    for name, recs, sk, a in points:
        b = bootstrap_cp(recs, sk, a)
        if b is None:
            continue
        b["dataset"] = name; b["score"] = sk; b["alpha"] = a; b["n_recs"] = len(recs)
        out["operating_points"].append(b)
        print(f"[{name:14s} {sk:9s} α={a:.2f}] cov={b['coverage']['mean']:.3f} CI=[{b['coverage']['ci95'][0]:.3f},{b['coverage']['ci95'][1]:.3f}]  "
              f"kept_acc={b['kept_acc']['mean']:.3f} CI=[{b['kept_acc']['ci95'][0]:.3f},{b['kept_acc']['ci95'][1]:.3f}]  "
              f"keep%={b['kept_frac']['mean']:.2f}")

    # OOD points
    ood_points = [
        ("AIME OOD vanilla", math_lp, aime_lp, "lp_min", 0.5, False),
        ("AIME OOD weighted", math_lp, aime_lp, "lp_min", 0.5, True),
        ("AIME OOD vanilla mean", math_lp, aime_lp, "lp_mean", 0.5, False),
        ("AIME OOD weighted mean", math_lp, aime_lp, "lp_mean", 0.5, True),
    ]
    out["ood_points"] = []
    for name, cal, te, sk, a, w in ood_points:
        b = bootstrap_ood(cal, te, sk, a, weighted=w)
        if b is None:
            continue
        b["name"] = name; b["score"] = sk; b["alpha"] = a; b["weighted"] = w
        out["ood_points"].append(b)
        print(f"[{name:25s} {sk:8s} α={a:.2f}] cov={b['coverage']['mean']:.3f} CI=[{b['coverage']['ci95'][0]:.3f},{b['coverage']['ci95'][1]:.3f}]  "
              f"kept_acc={b['kept_acc']['mean']:.3f} CI=[{b['kept_acc']['ci95'][0]:.3f},{b['kept_acc']['ci95'][1]:.3f}]")

    out_path = RESULTS / "pilotJ_bootstrap_ci.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

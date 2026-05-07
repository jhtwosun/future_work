"""
EXP-S2-followup: CP simulation on the perturbation scores from SX_perturb.
Use scores from SX_perturb_scores.jsonl + greedy_correct as label.
Compute kept_acc with bootstrap CIs across α.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

EXP = Path("/home/nvidia/future/experiments/results")
sys.path.insert(0, str(Path(__file__).parent))


def split_cp_with_ci(scores, correct, alpha, n_boot=300, n_seeds_inner=10, cal_frac=0.5):
    rng = np.random.default_rng(0)
    n = len(scores)
    boot_acc, boot_cov, boot_keep = [], [], []
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    valid_mask = ~np.isnan(scores)
    if valid_mask.sum() < 20:
        return None
    scores = scores[valid_mask]; correct = correct[valid_mask]
    n = len(scores)
    for _ in range(n_boot):
        idx_b = rng.integers(0, n, size=n)
        rs_b = scores[idx_b]; cs_b = correct[idx_b]
        accs, covs, keeps = [], [], []
        for _ in range(n_seeds_inner):
            perm = rng.permutation(n)
            nc = int(cal_frac * n)
            ci, ti = perm[:nc], perm[nc:]
            cal_corr = rs_b[ci][cs_b[ci] == 1]
            if len(cal_corr) < 5:
                continue
            n_c = len(cal_corr)
            ql = max(0.0, min(1.0, math.floor(alpha * (n_c + 1)) / n_c))
            q = float(np.quantile(cal_corr, ql))
            kept = rs_b[ti] >= q
            n_corr = (cs_b[ti] == 1).sum()
            if n_corr == 0:
                continue
            covs.append(float((kept & (cs_b[ti] == 1)).sum() / n_corr))
            keeps.append(float(kept.mean()))
            if kept.sum():
                accs.append(float(cs_b[ti][kept].mean()))
        if accs:
            boot_acc.append(np.mean(accs))
            boot_cov.append(np.mean(covs))
            boot_keep.append(np.mean(keeps))
    if not boot_acc:
        return None
    return {
        "kept_acc":  {"mean": float(np.mean(boot_acc)),
                       "ci95": [float(np.quantile(boot_acc, 0.025)), float(np.quantile(boot_acc, 0.975))]},
        "coverage":  {"mean": float(np.mean(boot_cov)),
                       "ci95": [float(np.quantile(boot_cov, 0.025)), float(np.quantile(boot_cov, 0.975))]},
        "kept_frac": {"mean": float(np.mean(boot_keep)),
                       "ci95": [float(np.quantile(boot_keep, 0.025)), float(np.quantile(boot_keep, 0.975))]},
    }


def main():
    rows = [json.loads(l) for l in (EXP / "SX_perturb_scores.jsonl").read_text().splitlines() if l.strip()]
    correct = np.array([r["greedy_correct"] for r in rows])

    score_specs = []
    for k in rows[0]:
        if k.startswith(("singleshot_", "xtemp_", "stepmask_", "para_")):
            if k.endswith(("_agree", "_agree_greedy", "_top1_inc_greedy", "_consensus", "_stability")):
                score_specs.append(k)
    score_specs = sorted(set(score_specs))
    print(f"Score families: {len(score_specs)}")

    out_rows = []
    for spec in score_specs:
        scores = np.array([float(r.get(spec, np.nan)) for r in rows])
        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
            cp = split_cp_with_ci(scores, correct, alpha)
            if cp is None:
                continue
            out_rows.append({
                "score": spec,
                "alpha": alpha,
                "n": int(len(rows)),
                "vanilla_acc": float(correct.mean()),
                **cp,
            })
            print(f"[{spec:35s} α={alpha:.2f}]  cov={cp['coverage']['mean']:.3f}  kept_acc={cp['kept_acc']['mean']:.3f}±{(cp['kept_acc']['ci95'][1]-cp['kept_acc']['ci95'][0])/2:.3f}  keep%={cp['kept_frac']['mean']:.2f}")

    out_path = EXP / "SX_perturb_cp_results.json"
    out_path.write_text(json.dumps(out_rows, indent=2))

    # Headline: best score per α
    print("\n=== Best perturb score per α ===")
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
        relevant = [r for r in out_rows if abs(r["alpha"] - alpha) < 0.001]
        relevant.sort(key=lambda r: -r["kept_acc"]["mean"])
        print(f"\nα={alpha}:")
        for r in relevant[:5]:
            ci = r["kept_acc"]["ci95"]
            print(f"  {r['score']:35s} kept_acc={r['kept_acc']['mean']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] keep%={r['kept_frac']['mean']:.2f}")


if __name__ == "__main__":
    main()

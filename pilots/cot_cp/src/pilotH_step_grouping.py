"""
Pilot H: R1-Distill step grouping.

R1-Distill traces have ~60 steps per trajectory, which makes lp_min
dominated by single-step noise (Pilot E showed lp_min becomes weaker
than lp_mean for R1). Test whether grouping every K consecutive steps
into one "super-step" recovers lp_min as a useful score.

Reads: results/pilotE_r1_greedy_traces.jsonl
Writes: results/pilotH_grouped_analysis.json
"""

import json
import math
from pathlib import Path
from itertools import zip_longest

import numpy as np

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")
TRACES = RESULTS / "pilotE_r1_greedy_traces.jsonl"


def step_lp_lists(token_logprobs, boundaries):
    bdy = boundaries + [len(token_logprobs)]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        out.append(seg)
    return out


def group_into_super_steps(per_step_token_lps, K):
    """Concat K consecutive step token lists into one super-step token list."""
    grouped = []
    for i in range(0, len(per_step_token_lps), K):
        chunk = []
        for s in per_step_token_lps[i:i+K]:
            chunk.extend(s)
        if chunk:
            grouped.append(chunk)
    return grouped


def super_step_means(grouped):
    return [float(np.mean(s)) for s in grouped if s]


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
    rows = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} R1 traces")

    out = {"model": "DeepSeek-R1-Distill-Qwen-7B", "K_values": [], "stats": []}

    for K in [1, 2, 5, 10, 20]:
        records = []
        for r in rows:
            token_lps_per_step = step_lp_lists(r["token_logprobs"], r["step_boundaries"])
            grouped = group_into_super_steps(token_lps_per_step, K)
            ssm = super_step_means(grouped)
            if not ssm:
                continue
            records.append({
                "id": r["id"],
                "n_groups": len(ssm),
                "lp_mean":   float(np.mean(ssm)),
                "lp_min":    float(np.min(ssm)),
                "lp_median": float(np.median(ssm)),
                "correct":   int(bool(r["correct"])),
            })
        if not records:
            continue

        from scipy.stats import spearmanr
        for sk in ["lp_mean", "lp_min", "lp_median"]:
            scores = np.array([r[sk] for r in records])
            correct = np.array([r["correct"] for r in records])
            rho, p = spearmanr(scores, correct)
            cp_03 = split_cp(scores, correct, 0.3)
            cp_05 = split_cp(scores, correct, 0.5)
            entry = {
                "K": K, "score": sk, "n_records": len(records),
                "mean_groups": float(np.mean([r["n_groups"] for r in records])),
                "spearman_rho": float(rho), "spearman_p": float(p),
                "cp_alpha_0.3": cp_03,
                "cp_alpha_0.5": cp_05,
            }
            out["stats"].append(entry)
            print(f"[K={K:2d}  {sk:9s}] groups~{entry['mean_groups']:.1f}  ρ={rho:+.3f}  CP@0.3 acc={cp_03['kept_acc_mean']:.3f}/{cp_03['kept_frac_mean']:.2f}  CP@0.5 acc={cp_05['kept_acc_mean']:.3f}/{cp_05['kept_frac_mean']:.2f}")
        out["K_values"].append(K)

    out_path = RESULTS / "pilotH_grouped_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

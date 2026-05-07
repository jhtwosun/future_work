"""
EXP-PER-STEP-CP: implement per-step CP variants from saved provisional-answer traces.

Approaches:
A. Pure per-step CP (online early-abort): at each step t, abort if score_t < q_α(t).
B. Per-step CP with coverage guarantee: α budget split across step positions.
   - Bonferroni: α_t = α / T_max
   - Hochberg: BH-style ordered correction
   - Sequential e-process (Vovk-style)

Online-computable scores (no knowledge of final answer):
- `running_lp_min` (min over steps 1..t)
- `running_lp_drawdown` (peak-to-current over 1..t)
- `running_entropy_mean` (mean entropy 1..t)
- `running_prov_majority_count` (count of steps where prov == prov_latest)
- `running_prov_n_distinct` (distinct provisional answers)
- `running_prov_late_stability` (last K=3 provisionals all agree)

Loads from SX_prov_*.json + SX_prov_*_per_step.jsonl.

Env vars:
  TAG (e.g. qwen25_7b), DATASET (math500/aime/olympiad)
"""

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import normalize, equal_strict

TAG = os.environ.get("TAG", "qwen25_7b")
DATASET = os.environ.get("DATASET", "math500")

OUTDIR = Path("/home/nvidia/future/experiments/results")
SUM = OUTDIR / f"SX_per_step_cp_{TAG}_{DATASET}.json"

ALPHAS = [0.10, 0.30, 0.50]


def load_data():
    f = OUTDIR / f"SX_prov_{TAG}_{DATASET}_per_step.jsonl"
    if not f.exists():
        raise FileNotFoundError(f)
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def compute_running_scores(row):
    """For each step t in 1..n_steps, compute online scores using prefix info only."""
    sfeats = row["per_step_basic"]
    prov_steps = row.get("per_step_prov", [])
    n_steps = len(sfeats)
    if n_steps == 0:
        return []
    out = []
    running_lp_min = float("inf")
    running_lp_max = float("-inf")
    running_ent_sum = 0.0; running_ent_count = 0
    running_lp_sum = 0.0; running_lp_count = 0
    prov_dict = {p["step"]: p for p in prov_steps}
    seen_prov = []
    for t in range(n_steps):
        s = sfeats[t]
        # Update running stats
        if not (s["lp"] != s["lp"]):
            running_lp_min = min(running_lp_min, s["lp"])
            running_lp_max = max(running_lp_max, s["lp"])
            running_lp_sum += s["lp"]; running_lp_count += 1
        if not (s["ent"] != s["ent"]):
            running_ent_sum += s["ent"]; running_ent_count += 1
        # prov score at this step (if queried)
        prov_record = prov_dict.get(t + 1)  # step boundaries are 1-indexed
        if prov_record:
            seen_prov.append(prov_record["prov_pred"])

        # Online scores
        feats = {
            "running_lp_min": running_lp_min if running_lp_min != float("inf") else float("nan"),
            "running_lp_drawdown": (running_lp_max - running_lp_min) if running_lp_max > -1e9 else float("nan"),
            "running_entropy_mean": running_ent_sum / running_ent_count if running_ent_count > 0 else float("nan"),
            "running_lp_mean": running_lp_sum / running_lp_count if running_lp_count > 0 else float("nan"),
        }
        # prov-based features
        if seen_prov:
            counter = Counter([str(normalize(p)) for p in seen_prov if p is not None])
            if counter:
                modal, modal_count = counter.most_common(1)[0]
                feats["prov_modal_count"] = modal_count
                feats["prov_modal_frac"] = modal_count / len(seen_prov)
                feats["prov_n_distinct"] = len(counter)
            else:
                feats["prov_modal_count"] = 0
                feats["prov_modal_frac"] = 0.0
                feats["prov_n_distinct"] = 0
            # last 3 stability
            last_k = seen_prov[-3:]
            if len(last_k) >= 3:
                last_set = set(str(normalize(p)) for p in last_k if p is not None)
                feats["prov_last3_stable"] = int(len(last_set) == 1)
            else:
                feats["prov_last3_stable"] = 0
            # n flips so far
            n_flips = 0
            for i in range(1, len(seen_prov)):
                if seen_prov[i-1] is not None and seen_prov[i] is not None and not equal_strict(seen_prov[i-1], seen_prov[i]):
                    n_flips += 1
            feats["prov_n_flips"] = n_flips
            # match rate vs LATEST prov (proxy for "does early reasoning agree with current commitment")
            if seen_prov[-1] is not None:
                matches = sum(1 for p in seen_prov if p is not None and equal_strict(p, seen_prov[-1]))
                feats["prov_match_latest_rate"] = matches / len(seen_prov)
            else:
                feats["prov_match_latest_rate"] = 0.0
        else:
            feats.update({
                "prov_modal_count": 0, "prov_modal_frac": 0.0, "prov_n_distinct": 0,
                "prov_last3_stable": 0, "prov_n_flips": 0, "prov_match_latest_rate": 0.0,
            })

        out.append(feats)
    return out


def per_step_cp_approach_A(rows, score_key, alpha, n_boot=200, n_seeds=3):
    """Approach A: pure per-step early-abort.
    For each step position t, calibrate threshold q_α(t) from cal correct trajectories at that step.
    At test, abort first time score_t < q_α(t).
    For trajectories shorter than max step, treat as 'never aborted'.
    """
    rng = np.random.default_rng(0)
    boot_kept_acc = []
    boot_avg_steps = []
    boot_n_kept = []

    # max step length
    max_T = max(len(r["online_scores"]) for r in rows)
    n = len(rows)

    for boot_iter in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rows = [rows[i] for i in idx]
        for _ in range(n_seeds):
            perm = rng.permutation(len(boot_rows))
            nc = len(boot_rows) // 2
            cal_idx, test_idx = perm[:nc], perm[nc:]
            cal_rows = [boot_rows[i] for i in cal_idx]
            test_rows = [boot_rows[i] for i in test_idx]
            cal_corr_rows = [r for r in cal_rows if r["correct"] == 1]
            if len(cal_corr_rows) < 5: continue

            # Calibrate threshold per step position
            thresholds = {}
            for t in range(max_T):
                vals = [r["online_scores"][t][score_key] for r in cal_corr_rows
                        if t < len(r["online_scores"]) and not np.isnan(r["online_scores"][t][score_key])]
                if len(vals) < 5: continue
                n_c = len(vals)
                ql = max(0.0, min(1.0, math.floor(alpha * (n_c + 1)) / n_c))
                thresholds[t] = float(np.quantile(vals, ql))

            # Test
            kept_acc_steps = []
            n_kept = 0
            for r in test_rows:
                # Try to keep up to longest step; abort if any score_t < threshold_t
                aborted = False
                steps_used = len(r["online_scores"])
                for t, feats in enumerate(r["online_scores"]):
                    if t in thresholds and not np.isnan(feats[score_key]):
                        if feats[score_key] < thresholds[t]:
                            aborted = True
                            steps_used = t
                            break
                if not aborted:
                    n_kept += 1
                    kept_acc_steps.append(r["correct"])
            if n_kept > 0:
                boot_kept_acc.append(float(np.mean(kept_acc_steps)))
                boot_avg_steps.append(float(np.mean([
                    len(r["online_scores"]) if not check_aborted(r, thresholds, score_key) else
                    abort_step(r, thresholds, score_key)
                    for r in test_rows
                ])))
                boot_n_kept.append(n_kept / len(test_rows))
    if not boot_kept_acc:
        return None
    return {
        "kept_acc": float(np.mean(boot_kept_acc)),
        "kept_acc_ci": [float(np.quantile(boot_kept_acc, 0.025)), float(np.quantile(boot_kept_acc, 0.975))],
        "avg_steps_used": float(np.mean(boot_avg_steps)),
        "kept_frac": float(np.mean(boot_n_kept)),
    }


def check_aborted(r, thresholds, score_key):
    for t, feats in enumerate(r["online_scores"]):
        if t in thresholds and not np.isnan(feats[score_key]):
            if feats[score_key] < thresholds[t]:
                return True
    return False


def abort_step(r, thresholds, score_key):
    for t, feats in enumerate(r["online_scores"]):
        if t in thresholds and not np.isnan(feats[score_key]):
            if feats[score_key] < thresholds[t]:
                return t
    return len(r["online_scores"])


def per_step_cp_bonferroni(rows, score_key, alpha, n_boot=200, n_seeds=3):
    """Approach B (Bonferroni): allocate α/T_max per step, calibrate threshold per step.
    Coverage guarantee: P(all correct test traces survive all steps) ≥ 1 - α.
    """
    max_T = max(len(r["online_scores"]) for r in rows)
    if max_T == 0: return None
    alpha_per_step = alpha / max_T
    return per_step_cp_approach_A(rows, score_key, alpha_per_step, n_boot=n_boot, n_seeds=n_seeds)


def per_step_cp_geometric(rows, score_key, alpha, n_boot=200, n_seeds=3):
    """Approach C (geometric / weighted): allocate α with weights ∝ 1/(t+1) (more budget at early steps).
    Σα_t = α. Less conservative than Bonferroni for typical data where early steps need more headroom.
    """
    max_T = max(len(r["online_scores"]) for r in rows)
    if max_T == 0: return None
    weights = [1.0 / (t + 1) for t in range(max_T)]
    Z = sum(weights)
    alpha_per_step = [alpha * w / Z for w in weights]
    rng = np.random.default_rng(0)
    n = len(rows)
    boot_kept_acc = []
    boot_avg_steps = []
    boot_n_kept = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rows = [rows[i] for i in idx]
        for _ in range(n_seeds):
            perm = rng.permutation(len(boot_rows))
            nc = len(boot_rows) // 2
            cal_idx, test_idx = perm[:nc], perm[nc:]
            cal_rows = [boot_rows[i] for i in cal_idx]
            test_rows = [boot_rows[i] for i in test_idx]
            cal_corr_rows = [r for r in cal_rows if r["correct"] == 1]
            if len(cal_corr_rows) < 5: continue
            thresholds = {}
            for t in range(max_T):
                vals = [r["online_scores"][t][score_key] for r in cal_corr_rows
                        if t < len(r["online_scores"]) and not np.isnan(r["online_scores"][t][score_key])]
                if len(vals) < 5: continue
                n_c = len(vals)
                a_t = alpha_per_step[t]
                ql = max(0.0, min(1.0, math.floor(a_t * (n_c + 1)) / n_c))
                thresholds[t] = float(np.quantile(vals, ql))
            kept_pts = []; n_kept = 0; steps_used_list = []
            for r in test_rows:
                aborted = False; steps_used = len(r["online_scores"])
                for t, feats in enumerate(r["online_scores"]):
                    if t in thresholds and not np.isnan(feats[score_key]):
                        if feats[score_key] < thresholds[t]:
                            aborted = True; steps_used = t; break
                if not aborted:
                    n_kept += 1
                    kept_pts.append(r["correct"])
                steps_used_list.append(steps_used)
            if n_kept > 0:
                boot_kept_acc.append(float(np.mean(kept_pts)))
                boot_avg_steps.append(float(np.mean(steps_used_list)))
                boot_n_kept.append(n_kept / len(test_rows))
    if not boot_kept_acc: return None
    return {
        "kept_acc": float(np.mean(boot_kept_acc)),
        "kept_acc_ci": [float(np.quantile(boot_kept_acc, 0.025)), float(np.quantile(boot_kept_acc, 0.975))],
        "avg_steps_used": float(np.mean(boot_avg_steps)),
        "kept_frac": float(np.mean(boot_n_kept)),
    }


def trajectory_cp_baseline(rows, score_key_traj, alpha, n_boot=200, n_seeds=3):
    """Compare against trajectory-level CP using a single aggregate score.
    score_key_traj is taken from the LAST step's online score (i.e., after seeing full trace).
    """
    rng = np.random.default_rng(0)
    n = len(rows)
    boot_acc = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rows = [rows[i] for i in idx]
        for _ in range(n_seeds):
            perm = rng.permutation(len(boot_rows))
            nc = len(boot_rows) // 2
            cal_idx, test_idx = perm[:nc], perm[nc:]
            cal_rows = [boot_rows[i] for i in cal_idx]
            test_rows = [boot_rows[i] for i in test_idx]
            cal_corr_vals = []
            for r in cal_rows:
                if r["correct"] == 1 and r["online_scores"]:
                    v = r["online_scores"][-1][score_key_traj]
                    if not np.isnan(v):
                        cal_corr_vals.append(v)
            if len(cal_corr_vals) < 5: continue
            n_c = len(cal_corr_vals)
            ql = max(0.0, min(1.0, math.floor(alpha * (n_c + 1)) / n_c))
            q = float(np.quantile(cal_corr_vals, ql))
            kept_acc_pts = []
            for r in test_rows:
                if not r["online_scores"]:
                    continue
                v = r["online_scores"][-1][score_key_traj]
                if np.isnan(v): continue
                if v >= q:
                    kept_acc_pts.append(r["correct"])
            if kept_acc_pts:
                boot_acc.append(float(np.mean(kept_acc_pts)))
    if not boot_acc: return None
    return {"kept_acc": float(np.mean(boot_acc)),
            "kept_acc_ci": [float(np.quantile(boot_acc, 0.025)), float(np.quantile(boot_acc, 0.975))]}


def main():
    print(f"=== Per-step CP analysis: {TAG} {DATASET} ===", flush=True)
    rows = load_data()
    print(f"Loaded {len(rows)} traces", flush=True)
    n_correct = sum(r["correct"] for r in rows)
    print(f"Vanilla acc: {n_correct/len(rows):.3f}", flush=True)

    # Compute online scores per step
    for r in rows:
        r["online_scores"] = compute_running_scores(r)
    print(f"Online scores computed", flush=True)

    score_keys = [
        "running_lp_min",
        "running_lp_drawdown",
        "running_entropy_mean",
        "running_lp_mean",
        "prov_modal_frac",
        "prov_n_distinct",
        "prov_last3_stable",
        "prov_n_flips",
        "prov_match_latest_rate",
    ]

    # Determine sign convention by Spearman with greedy correctness on FINAL step value
    print("\n=== Score sign convention (Spearman on final-step value) ===", flush=True)
    from scipy.stats import spearmanr
    correct = np.array([r["correct"] for r in rows])
    score_signs = {}
    for sk in score_keys:
        finals = np.array([r["online_scores"][-1][sk] if r["online_scores"] else float("nan") for r in rows], dtype=float)
        valid = ~np.isnan(finals)
        if valid.sum() < 20:
            score_signs[sk] = +1
            continue
        rho, _ = spearmanr(finals[valid], correct[valid])
        score_signs[sk] = +1 if rho >= 0 else -1
        print(f"  {sk:30s} ρ_final={rho:+.3f} sign={score_signs[sk]:+d}", flush=True)
        # Apply sign
        for r in rows:
            for feats in r["online_scores"]:
                if not np.isnan(feats[sk]):
                    feats[sk] = feats[sk] * score_signs[sk]

    # Run all 3 approaches
    summary = {"tag": TAG, "dataset": DATASET, "n": len(rows),
                "vanilla_acc": float(n_correct/len(rows)),
                "score_signs": score_signs,
                "results": {}}

    for alpha in ALPHAS:
        print(f"\n=== Alpha = {alpha} ===", flush=True)
        for sk in score_keys:
            if sk not in score_signs: continue

            # Trajectory-level baseline
            t_res = trajectory_cp_baseline(rows, sk, alpha)
            # Approach A
            a_res = per_step_cp_approach_A(rows, sk, alpha)
            # Approach B (Bonferroni)
            b_res = per_step_cp_bonferroni(rows, sk, alpha)

            summary["results"][f"{sk}_alpha_{alpha}"] = {
                "trajectory_cp": t_res,
                "approach_A_per_step": a_res,
                "approach_B_bonferroni": b_res,
            }
            line = f"  {sk:25s} α={alpha} "
            if t_res:
                line += f"traj_cp_kept={t_res['kept_acc']:.3f} "
            if a_res:
                line += f"A_kept={a_res['kept_acc']:.3f} (avg_steps={a_res['avg_steps_used']:.1f}, kept_frac={a_res['kept_frac']:.2f}) "
            if b_res:
                line += f"B_bonf_kept={b_res['kept_acc']:.3f} (kept_frac={b_res['kept_frac']:.2f})"
            print(line, flush=True)

    SUM.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done. Wrote {SUM} ===", flush=True)


if __name__ == "__main__":
    main()

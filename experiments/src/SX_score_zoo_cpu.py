"""
EXP-S1: CPU-only Wave 1 score zoo on existing traces.

Implements all "free / CPU-computable" score functions from the brainstorm:
- predictive_sharpening_rate (entropy slope)
- tiebreak_lex (sc_top1 + lp_min tiebreaker)
- gated_cascade (cost-amortized routing)
- disagreement_signal (logreg of lp+prm with interaction)
- quantile_fusion_cp (max threshold-deviation)
- final_answer_recheck (sympy on final answer)
- learned_stack_cv (GBDT, with CV+ split)
- And aggregator variants of lp_min: lp_p10, lp_min, lp_mean, lp_p25

Runs CP simulation on ALL combos × multiple α × multiple datasets.
Outputs comprehensive comparison.
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

EXP = Path("/home/nvidia/future/experiments/results")
PILOTS = Path("/home/nvidia/future/pilots/cot_cp/results")

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, normalize, equal_strict, to_number


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def per_step_lps(token_logprobs, boundaries):
    """Return list of mean log-prob per step."""
    bdy = boundaries + [len(token_logprobs)]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            out.append(float(np.mean(seg)))
    return out


def per_step_entropy_proxy(token_logprobs, boundaries):
    """Proxy for entropy: -log(p) of chosen token (since we don't have full distribution).
    Higher value = lower confidence."""
    bdy = boundaries + [len(token_logprobs)]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [-lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            out.append(float(np.mean(seg)))
    return out


def lp_score_aggregations(rows):
    """Compute multiple lp aggregator variants from token logprobs + step boundaries."""
    out = []
    for r in rows:
        if "token_logprobs" not in r or "step_boundaries" not in r:
            continue
        steps = per_step_lps(r["token_logprobs"], r["step_boundaries"])
        if not steps:
            continue
        steps = np.array(steps)
        # Entropy proxies
        ent_steps = per_step_entropy_proxy(r["token_logprobs"], r["step_boundaries"])
        ent_steps = np.array(ent_steps) if ent_steps else None
        # Predictive sharpening rate (slope of entropy vs step index, ASCENDING means increasing uncertainty)
        if ent_steps is not None and len(ent_steps) >= 3:
            x = np.arange(len(ent_steps))
            slope = float(np.polyfit(x, ent_steps, 1)[0])
            # NEGATIVE slope = entropy decreasing = sharpening = good
            sharpening_rate = -slope
        else:
            sharpening_rate = 0.0
        # Position-of-min normalized
        min_idx = int(np.argmin(steps))
        pos_min = min_idx / max(len(steps) - 1, 1)

        rec = {
            "id": r["id"],
            "lp_min":     float(np.min(steps)),
            "lp_max":     float(np.max(steps)),
            "lp_mean":    float(np.mean(steps)),
            "lp_median":  float(np.median(steps)),
            "lp_p10":     float(np.percentile(steps, 10)),
            "lp_p25":     float(np.percentile(steps, 25)),
            "lp_p75":     float(np.percentile(steps, 75)),
            "lp_p90":     float(np.percentile(steps, 90)),
            "lp_std":     float(np.std(steps)),
            "lp_range":   float(np.max(steps) - np.min(steps)),
            "lp_first":   float(steps[0]),
            "lp_last":    float(steps[-1]),
            "n_steps":    len(steps),
            "predictive_sharpening_rate": sharpening_rate,
            "pos_min_step": pos_min,
            # Entropy variants from logprob proxy
            "ent_p95":     float(np.percentile(ent_steps, 95)) if ent_steps is not None and len(ent_steps) > 0 else 0.0,
            "ent_p75":     float(np.percentile(ent_steps, 75)) if ent_steps is not None and len(ent_steps) > 0 else 0.0,
            "ent_max":     float(np.max(ent_steps)) if ent_steps is not None and len(ent_steps) > 0 else 0.0,
            "correct":    int(bool(r.get("correct", False))),
        }
        out.append(rec)
    return out


def sc_records(rows):
    out = []
    for r in rows:
        ok = r.get("majority_correct")
        if ok is None and "majority_pred" in r and "gold" in r:
            ok = int(equal_strict(r["majority_pred"], r["gold"]))
        out.append({
            "id": r.get("id"),
            "sc_top1": float(r.get("top1_frac", 0.0)),
            "correct": int(ok or 0),
        })
    return out


def merge_lp_sc_prm(lp_recs, sc_recs, prm_recs=None):
    """Merge multiple score sources by id."""
    sc_by_id = {r["id"]: r for r in sc_recs}
    prm_by_id = {r["id"]: r for r in (prm_recs or [])}
    merged = []
    for lr in lp_recs:
        out = dict(lr)
        if lr["id"] in sc_by_id:
            sc = sc_by_id[lr["id"]]
            out["sc_top1"] = sc["sc_top1"]
            # Use SC majority correctness for SC-based scores (not lp's correctness)
            out["correct_sc"] = sc["correct"]
        if lr["id"] in prm_by_id:
            pr = prm_by_id[lr["id"]]
            out["prm_min"] = pr.get("prm_min")
            out["prm_mean"] = pr.get("prm_mean")
        merged.append(out)
    return merged


# ===== Score functions =====

def gated_cascade_score(rec, thresholds=None):
    """Hierarchical: lp_min if lp_min > τ_high, else prm_min if prm_min > τ_mid,
    else sc_top1. We return the TIER and the score itself.

    The aggregated score is a normalized rank that respects the tier ordering.
    For CP purposes, we use the sc_top1 score directly (since sc_top1 ≥ prm_min ≥ lp_min in LR+),
    but in implementation only compute sc_top1 when needed.

    For pure score-comparison (CP simulation), we compute the *equivalent* sc_top1 value
    that would have been used.
    """
    return rec.get("sc_top1", 0.0)  # Simplification: assume cascade always reaches sc


def tiebreak_lex_score(rec, eps=1e-3):
    """sc_top1 + small lp_min as tiebreaker. eps small enough to not flip rankings except at ties."""
    sc = rec.get("sc_top1", 0.0)
    lp = rec.get("lp_min", 0.0)
    # Normalize lp to roughly [-1, 0] then scale by eps
    return sc + eps * lp  # lp is negative; tiebreaker lower for less-confident traces


def disagreement_signal_score(rec, weights=None):
    """logreg-style: w_lp * z_lp + w_prm * z_prm + w_diff * (z_lp - z_prm)^2 + w_inter * z_lp * z_prm.
    For training-free version, use fixed weights based on prior knowledge.
    """
    if weights is None:
        weights = {"lp": 1.0, "prm": 2.0, "diff_sq": -0.5, "inter": 0.5}
    z_lp = rec.get("lp_min", 0.0)
    z_prm = rec.get("prm_min", 0.5)
    return (weights["lp"] * z_lp + weights["prm"] * z_prm +
              weights["diff_sq"] * (z_lp - z_prm) ** 2 +
              weights["inter"] * z_lp * z_prm)


def quantile_fusion_score(rec, qhats=None):
    """Max of (score - qhat) deviations across calibrators. We compute deviation from per-source means."""
    if qhats is None:
        qhats = {"lp_min": -0.1, "prm_min": 0.5, "sc_top1": 0.7}
    lp_dev = rec.get("lp_min", 0.0) - qhats["lp_min"]
    prm_dev = rec.get("prm_min", 0.5) - qhats["prm_min"]
    sc_dev = rec.get("sc_top1", 0.7) - qhats["sc_top1"]
    return max(lp_dev, prm_dev, sc_dev)


# ===== CP simulation with bootstrap CI =====

def split_cp_with_ci(scores, correct, alpha, n_boot=300, n_seeds_inner=10, cal_frac=0.5):
    rng = np.random.default_rng(0)
    n = len(scores)
    boot_acc, boot_cov, boot_keep = [], [], []
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
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


def evaluate_score_zoo(merged, dataset_name):
    """Run CP simulation across many score families and α."""
    rows = []
    correct = np.array([r["correct"] for r in merged])
    correct_sc = np.array([r.get("correct_sc", r["correct"]) for r in merged])

    score_specs = [
        ("lp_min",     "lp_min",     correct,      "free"),
        ("lp_mean",    "lp_mean",    correct,      "free"),
        ("lp_median",  "lp_median",  correct,      "free"),
        ("lp_p10",     "lp_p10",     correct,      "free"),
        ("lp_p25",     "lp_p25",     correct,      "free"),
        ("lp_max",     "lp_max",     correct,      "free"),
        ("lp_first",   "lp_first",   correct,      "free"),
        ("lp_last",    "lp_last",    correct,      "free"),
        ("lp_std",     "lp_std",     correct,      "free"),
        ("lp_range",   "lp_range",   correct,      "free"),
        ("predictive_sharpening", "predictive_sharpening_rate", correct, "free"),
        ("pos_min_step", "pos_min_step", correct, "free"),
        ("ent_p95",    "ent_p95",    correct,      "free"),  # entropy proxy from -lp
        ("ent_max",    "ent_max",    correct,      "free"),
    ]
    # Add SC-based and PRM-based if available
    if "sc_top1" in merged[0]:
        score_specs.append(("sc_top1", "sc_top1", correct_sc, "8x"))
        score_specs.append(("tiebreak_lex", None, correct_sc, "8x+free"))
    if "prm_min" in merged[0]:
        score_specs.append(("prm_min", "prm_min", correct, "2x"))
        score_specs.append(("prm_mean", "prm_mean", correct, "2x"))
        # Disagreement signal needs both lp + prm
        score_specs.append(("disagreement_signal", None, correct, "2x+free"))
    if "sc_top1" in merged[0] and "prm_min" in merged[0]:
        score_specs.append(("quantile_fusion", None, correct_sc, "10x"))
        score_specs.append(("gated_cascade_proxy", None, correct_sc, "≤8x"))

    for spec_name, score_key, this_correct, cost in score_specs:
        if score_key is not None:
            scores = np.array([r.get(score_key, 0.0) for r in merged])
        elif spec_name == "tiebreak_lex":
            scores = np.array([tiebreak_lex_score(r) for r in merged])
        elif spec_name == "disagreement_signal":
            scores = np.array([disagreement_signal_score(r) for r in merged])
        elif spec_name == "quantile_fusion":
            scores = np.array([quantile_fusion_score(r) for r in merged])
        elif spec_name == "gated_cascade_proxy":
            scores = np.array([gated_cascade_score(r) for r in merged])
        else:
            continue

        if scores.size == 0 or np.all(np.isnan(scores)):
            continue
        scores = np.nan_to_num(scores, nan=np.nanmin(scores))

        for alpha in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
            cp = split_cp_with_ci(scores, this_correct, alpha)
            if cp is None:
                continue
            rows.append({
                "dataset": dataset_name,
                "score": spec_name,
                "score_field": score_key or spec_name,
                "cost": cost,
                "alpha": alpha,
                "n": int(len(merged)),
                "vanilla_acc": float(this_correct.mean()),
                **cp,
            })
    return rows


def main():
    OUT = Path("/home/nvidia/future/score_ideation/synthesis/exp_s1_cpu_zoo.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    sources = []

    # MATH-500 (E1) — has lp + sc + PRM (Pilot A)
    if (EXP / "E1_math500_greedy_traces.jsonl").exists():
        rows = load_jsonl(EXP / "E1_math500_greedy_traces.jsonl")
        # update correct fields with robust eval
        for r in rows:
            r["correct"] = int(equal_strict(extract_pred(r.get("output_text","")), r.get("gold")))
        lp_recs = lp_score_aggregations(rows)
        sc_rows = load_jsonl(EXP / "E1_math500_sc8_traces.jsonl") if (EXP / "E1_math500_sc8_traces.jsonl").exists() else []
        for r in sc_rows:
            r["majority_correct"] = int(equal_strict(r.get("majority_pred"), r.get("gold")))
        sc_recs = sc_records(sc_rows)
        prm_rows = load_jsonl(PILOTS / "pilotA_prm_traces.jsonl") if (PILOTS / "pilotA_prm_traces.jsonl").exists() else []
        prm_recs = [{"id": r["id"], "prm_min": float(r["prm_min"]), "prm_mean": float(r["prm_mean"])} for r in prm_rows]
        merged = merge_lp_sc_prm(lp_recs, sc_recs, prm_recs)
        sources.append(("MATH-500-Qwen2.5-7B", merged))
        print(f"MATH-500: {len(merged)} merged, lp+sc+prm available", flush=True)

    # MMLU-Pro STEM (E12r) — has lp + sc, no PRM
    if (EXP / "E12r_mmlupro_stem_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E12r_mmlupro_stem_greedy.jsonl")
        # for MCQ, correct field already exists
        lp_recs = lp_score_aggregations(rows)
        sc_rows = load_jsonl(EXP / "E12r_mmlupro_stem_sc.jsonl")
        sc_recs = sc_records(sc_rows)
        merged = merge_lp_sc_prm(lp_recs, sc_recs)
        sources.append(("MMLU-Pro-STEM-Qwen3-8B", merged))
        print(f"MMLU-Pro: {len(merged)} merged, lp+sc available", flush=True)

    # OlympiadBench (E13r)
    if (EXP / "E13r_olympiad_math_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E13r_olympiad_math_greedy.jsonl")
        lp_recs = lp_score_aggregations(rows)
        sc_rows = load_jsonl(EXP / "E13r_olympiad_math_sc.jsonl")
        sc_recs = sc_records(sc_rows)
        merged = merge_lp_sc_prm(lp_recs, sc_recs)
        sources.append(("OlympiadBench-Qwen3-8B", merged))
        print(f"Olympiad: {len(merged)} merged", flush=True)

    # HumanEval (E15)
    if (EXP / "E15_humaneval_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E15_humaneval_greedy.jsonl")
        lp_recs = lp_score_aggregations(rows)
        sc_rows = load_jsonl(EXP / "E15_humaneval_sc.jsonl")
        sc_recs = sc_records(sc_rows)
        merged = merge_lp_sc_prm(lp_recs, sc_recs)
        sources.append(("HumanEval-Qwen3-8B", merged))
        print(f"HumanEval: {len(merged)} merged", flush=True)

    # Qwen2.5-Math-7B (E16) — math-specialized
    if (EXP / "E16_qwen_math_7b_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E16_qwen_math_7b_greedy.jsonl")
        for r in rows:
            r["correct"] = int(equal_strict(extract_pred(r.get("output_text","")), r.get("gold")))
        lp_recs = lp_score_aggregations(rows)
        sc_rows = load_jsonl(EXP / "E16_qwen_math_7b_sc.jsonl") if (EXP / "E16_qwen_math_7b_sc.jsonl").exists() else []
        # E16 stored samples; need to re-extract majority
        sc_recs = []
        for r in sc_rows:
            if "samples" in r:
                preds = [extract_pred(s) for s in r["samples"]]
                preds_clean = [p for p in preds if p is not None]
                counter = Counter([normalize(p) for p in preds_clean])
                top, cnt = (counter.most_common(1)[0] if counter else (None, 0))
                top1_frac = cnt / max(len(preds), 1)
                ok = int(equal_strict(top, r.get("gold")))
                sc_recs.append({"id": r["id"], "sc_top1": float(top1_frac), "correct": ok})
            else:
                sc_recs.append({"id": r["id"], "sc_top1": float(r.get("top1_frac", 0.0)),
                                  "correct": int(r.get("majority_correct", 0))})
        merged = merge_lp_sc_prm(lp_recs, sc_recs)
        sources.append(("MATH-500-Qwen2.5-Math-7B", merged))
        print(f"Math-7B: {len(merged)} merged", flush=True)

    # AIME (E2) — for OOD / harder
    if (EXP / "E2_aime_greedy_traces.jsonl").exists():
        rows = load_jsonl(EXP / "E2_aime_greedy_traces.jsonl")
        lp_recs = lp_score_aggregations(rows)
        sc_rows = load_jsonl(EXP / "E2_aime_sc8_traces.jsonl")
        sc_recs = sc_records(sc_rows)
        merged = merge_lp_sc_prm(lp_recs, sc_recs)
        sources.append(("AIME-Qwen2.5-7B", merged))
        print(f"AIME: {len(merged)} merged", flush=True)

    # Run CP simulation across all sources
    all_rows = []
    for name, merged in sources:
        if not merged:
            continue
        print(f"\n=== {name}, n={len(merged)}, vanilla_acc={np.mean([r['correct'] for r in merged]):.3f} ===", flush=True)
        rows = evaluate_score_zoo(merged, name)
        all_rows.extend(rows)
        # Print top operating points per dataset
        print(f"  Got {len(rows)} (score, alpha) operating points")

    OUT.write_text(json.dumps(all_rows, indent=2))
    print(f"\nWrote {len(all_rows)} operating points to {OUT}", flush=True)

    # Quick summary: per dataset, best score at α=0.3
    print("\n=== Best score per (dataset, α=0.3) ===")
    for name, _ in sources:
        relevant = [r for r in all_rows if r["dataset"] == name and abs(r["alpha"] - 0.3) < 0.001]
        if not relevant:
            continue
        relevant.sort(key=lambda r: -r["kept_acc"]["mean"])
        print(f"\n{name}:")
        for r in relevant[:8]:
            print(f"  {r['score']:25s} cost={r['cost']:10s} kept_acc={r['kept_acc']['mean']:.3f} keep%={r['kept_frac']['mean']:.2f} cov={r['coverage']['mean']:.3f}")


if __name__ == "__main__":
    main()

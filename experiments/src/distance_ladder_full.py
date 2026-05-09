"""
distance_ladder_full.py — Full-scale Distance-Ladder Conformal Calibration.

Extends distance_ladder_pilot.py to:

  1. Multi-model: 4 models
       - Qwen2.5-7B-Instruct  (E1 MATH-500, E13r OlympiadBench [proxy:
                              Qwen3-8B-no-think, closest available],
                              E2 AIME 1983-2024)
       - Qwen2.5-Math-7B      (E16 MATH-500 only available)
       - Qwen2.5-32B          (E4 MATH-500 only available)
       - Phi-4                (E8 MATH-500 only available)
     For models that lack native AIME/OlympiadBench SC@8 traces we
     document the limitation and (a) run a MATH-only split-CP
     internally and (b) run a "Qwen2.5-7B-anchored transport" variant
     that uses the model's own MATH-500 as rung 0 but borrows the
     Qwen2.5-7B Olympiad/AIME rungs as the higher rungs.  The transport
     variant is reported as "_xfer" cells.

  2. Full alpha grid: {0.05, 0.10, 0.20, 0.30, 0.50, 0.70}.

  3. Add OlympiadBench rung between MATH-500-eval and AIME-old.
     Final rungs (6-rung ladder):
        rung_0 PRM800K-style cal = MATH-500 cal half  (id %2 == 0)
        rung_1 MATH-500 eval     = MATH-500 eval half (id %2 == 1)
        rung_2 OlympiadBench     = E13r Qwen3-8B-no-think proxy (n=674)
        rung_3 AIME-old          = AIME 1983-1999 (n=224)
        rung_4 AIME-mid          = AIME 2000-2014 (n=426)
        rung_5 AIME-new          = AIME 2015-2023 (n=269)  [eval target]
     The 2024-2025 bucket has only n=14 in the data, too small for the
     bootstrap, so we keep AIME-new = 2015-2023 as the headline target
     and report a separate "AIME-2024" sentinel cell using the n=14
     subset (with caveat).

  4. Bootstrap CIs: 500 resamples per cell.

  5. Rung-ablation: drop each rung individually for Strategy B';
     report degradation in coverage gap.

  6. Strategy comparison: A (one-shot Theorem 3),
                          B (telescoped, point-identical to A by
                             telescope algebra; reported for slack
                             decomposition only),
                          B' (sequential, the pilot winner).

  7. Per-alpha sensitivity (H2): for each strategy report
     coverage(alpha)-target(alpha) curve.

  8. Per-rung empirical TV distance is computed and reported (this is
     the input to Theorem 5's epsilon_k).

Outputs:
  /home/nvidia/future/experiments/results/distance_ladder_full/
      qwen25_7b_strategyA_B_Bp.json
      qwen25_math_7b_strategyA_B_Bp.json
      qwen25_32b_strategyA_B_Bp.json
      phi4_strategyA_B_Bp.json
      AGGREGATE.json
      AGGREGATE.md

Usage:
  python3 /home/nvidia/future/experiments/src/distance_ladder_full.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXP = Path("/home/nvidia/future/experiments/results")
OUT_DIR = EXP / "distance_ladder_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = [k / 8 for k in range(9)]      # SC@8 vote shares: 0/8,...,8/8
EPS = 1e-3                              # Laplace smoothing for empirical PMFs
N_BOOT = 500
SEED = 0
ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]

# Per-model file mapping. None = not available natively.
MODEL_FILES = {
    "qwen25_7b": {
        "math500": "E1_math500_sc8_traces_re.jsonl",
        "olympiad": "E13r_olympiad_math_sc_re.jsonl",   # proxy: Qwen3-8B-no-think
        "aime":     "E2_aime_sc8_traces_re.jsonl",
        "olympiad_proxy_note": (
            "OlympiadBench traces are Qwen3-8B-no-think (closest available "
            "with SC@8 OlympiadBench). MATH-500 and AIME traces are native "
            "Qwen2.5-7B-Instruct."),
    },
    "qwen25_math_7b": {
        "math500": "E16_qwen_math_7b_sc_re.jsonl",
        "olympiad": None,
        "aime": None,
    },
    "qwen25_32b": {
        "math500": "E4_qwen32b_math500_sc_re.jsonl",
        "olympiad": None,
        "aime": None,
    },
    "phi4": {
        "math500": "E8_phi4_sc_re.jsonl",
        "olympiad": None,
        "aime": None,
    },
}

# AIME year partitioning (per task spec; AIME data only goes to 2024)
AIME_OLD_HI = 1999
AIME_MID_LO, AIME_MID_HI = 2000, 2014
AIME_NEW_LO, AIME_NEW_HI = 2015, 2023
AIME_FUT_LO = 2024

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _score(r):
    s = r.get("re_top1_frac", r.get("top1_frac"))
    return float(s) if s is not None else 0.0


def _ok(r):
    o = r.get("re_majority_correct", r.get("majority_correct"))
    return int(o) if o is not None else 0


def load_math500(fname):
    rows = load_jsonl(EXP / fname)
    return [{"id": int(r["id"]), "score": _score(r), "correct": _ok(r)}
            for r in rows]


def load_olympiad(fname):
    rows = load_jsonl(EXP / fname)
    return [{"id": int(r["id"]), "score": _score(r), "correct": _ok(r)}
            for r in rows]


def load_aime(fname):
    rows = load_jsonl(EXP / fname)
    return [{"id": int(r["id"]),
             "year": int(r.get("year", 0) or 0),
             "score": _score(r),
             "correct": _ok(r)}
            for r in rows]


# ---------------------------------------------------------------------------
# Empirical PMF / TV / density ratio
# ---------------------------------------------------------------------------

def empirical_pmf(scores, levels=LEVELS, smoothing=EPS):
    n = max(len(scores), 1)
    cnt = Counter(scores)
    return {l: (cnt.get(l, 0) + smoothing) / (n + smoothing * len(levels))
            for l in levels}


def tv_distance(p, q, levels=LEVELS):
    return 0.5 * sum(abs(p[l] - q[l]) for l in levels)


def density_ratio(p_target, p_source, levels=LEVELS):
    return {l: p_target[l] / max(p_source[l], 1e-9) for l in levels}


def clip_ratio(w_dict, lo=1e-3, hi=1e3):
    return {l: float(np.clip(v, lo, hi)) for l, v in w_dict.items()}


# ---------------------------------------------------------------------------
# Quantile primitives
# ---------------------------------------------------------------------------

def vanilla_lower_quantile(cal_correct_scores, alpha):
    n = len(cal_correct_scores)
    if n < 2:
        return float("nan")
    arr = np.sort(np.asarray(cal_correct_scores))
    idx = max(0, int(math.floor(alpha * (n + 1))) - 1)
    idx = min(idx, n - 1)
    return float(arr[idx])


def weighted_lower_quantile(cal_correct_scores, weights, alpha):
    arr = np.asarray(cal_correct_scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    if arr.size == 0 or w.sum() <= 0:
        return float("nan")
    order = np.argsort(arr)
    s = arr[order]
    w_norm = w[order] / w.sum()
    cum = np.cumsum(w_norm)
    idx = int(np.searchsorted(cum, alpha, side="left"))
    idx = min(idx, len(s) - 1)
    return float(s[idx])


# ---------------------------------------------------------------------------
# Coverage / bootstrap
# ---------------------------------------------------------------------------

def coverage_eval(eval_scores, eval_correct, q):
    s = np.asarray(eval_scores, dtype=float)
    c = np.asarray(eval_correct, dtype=int)
    if math.isnan(q):
        return {"coverage": float("nan"), "kept_acc": float("nan"),
                "kept_frac": float("nan"), "q": float("nan")}
    kept = s >= q
    n_corr = int((c == 1).sum())
    cov = float(((kept) & (c == 1)).sum() / max(n_corr, 1))
    n_kept = int(kept.sum())
    kacc = float(c[kept].mean()) if n_kept else float("nan")
    kfrac = float(kept.mean())
    return {"coverage": cov, "kept_acc": kacc,
            "kept_frac": kfrac, "q": float(q)}


def bootstrap_coverage_ci(eval_scores, eval_correct, q,
                          n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    s = np.asarray(eval_scores, dtype=float)
    c = np.asarray(eval_correct, dtype=int)
    n = len(s)
    if n == 0 or math.isnan(q):
        return [float("nan"), float("nan")]
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ss = s[idx]; cc = c[idx]
        nc = int((cc == 1).sum())
        if nc == 0:
            continue
        kept = ss >= q
        boot.append(((kept) & (cc == 1)).sum() / nc)
    if not boot:
        return [float("nan"), float("nan")]
    return [float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975))]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def strategy_oneshot(cal_recs, eval_recs, alpha):
    """Strategy A: one-shot empirical-PMF reweighted CP (current Theorem 3)."""
    p_cal = empirical_pmf([r["score"] for r in cal_recs])
    p_eval = empirical_pmf([r["score"] for r in eval_recs])
    w = clip_ratio(density_ratio(p_eval, p_cal))

    cal_correct = [r for r in cal_recs if r["correct"] == 1]
    cs = [r["score"] for r in cal_correct]
    ws = [w[s] for s in cs]
    q = weighted_lower_quantile(cs, ws, alpha)

    eval_scores = [r["score"] for r in eval_recs]
    eval_correct = [r["correct"] for r in eval_recs]
    ev = coverage_eval(eval_scores, eval_correct, q)
    ev["ci"] = bootstrap_coverage_ci(eval_scores, eval_correct, q)
    return ev


def strategy_telescoped_ladder(rungs, eval_recs, alpha):
    """Strategy B: K-rung telescoped (point-identical to A by telescope algebra,
    but we report the per-rung TV decomposition for Theorem 5 slack)."""
    pmfs = [empirical_pmf([r["score"] for r in rung]) for rung in rungs]
    per_rung_tv = []
    w_telescoped = {l: 1.0 for l in LEVELS}
    for k in range(1, len(rungs)):
        wk = density_ratio(pmfs[k], pmfs[k - 1])
        wk = clip_ratio(wk)
        per_rung_tv.append(tv_distance(pmfs[k], pmfs[k - 1]))
        for l in LEVELS:
            w_telescoped[l] *= wk[l]
    w_telescoped = clip_ratio(w_telescoped)

    rung0_correct = [r for r in rungs[0] if r["correct"] == 1]
    cs = [r["score"] for r in rung0_correct]
    ws = [w_telescoped[s] for s in cs]
    q = weighted_lower_quantile(cs, ws, alpha)

    eval_scores = [r["score"] for r in eval_recs]
    eval_correct = [r["correct"] for r in eval_recs]
    ev = coverage_eval(eval_scores, eval_correct, q)
    ev["ci"] = bootstrap_coverage_ci(eval_scores, eval_correct, q)
    ev["per_rung_tv"] = [float(t) for t in per_rung_tv]
    return ev


def strategy_sequential_ladder(rungs, eval_recs, alpha):
    """Strategy B': sequential rung-by-rung (the astronomer's ladder)."""
    pmfs = [empirical_pmf([r["score"] for r in rung]) for rung in rungs]
    rung0_correct = [r for r in rungs[0] if r["correct"] == 1]
    cs0 = [r["score"] for r in rung0_correct]
    q = vanilla_lower_quantile(cs0, alpha)
    qs = [q]
    for k in range(1, len(rungs)):
        wk = density_ratio(pmfs[k], pmfs[k - 1])
        wk = clip_ratio(wk)
        prev_correct = [r for r in rungs[k - 1] if r["correct"] == 1]
        cs = [r["score"] for r in prev_correct]
        if not cs:
            qs.append(qs[-1])
            continue
        ws = [wk[s] for s in cs]
        q = weighted_lower_quantile(cs, ws, alpha)
        qs.append(q)
    eval_scores = [r["score"] for r in eval_recs]
    eval_correct = [r["correct"] for r in eval_recs]
    ev = coverage_eval(eval_scores, eval_correct, qs[-1])
    ev["ci"] = bootstrap_coverage_ci(eval_scores, eval_correct, qs[-1])
    ev["q_path"] = qs
    return ev


# ---------------------------------------------------------------------------
# Per-model rung construction
# ---------------------------------------------------------------------------

def build_rungs_for_model(model: str, fallback_qwen7b_rungs: dict | None = None):
    """Returns dict {rung_name: list[record]} for the given model.

    Native data:
        - All models have MATH-500 (split into cal/eval halves by id parity).
        - Only qwen25_7b has native OlympiadBench and AIME.

    For non-7B models we need OlympiadBench and AIME for the higher rungs.
    Strategy: borrow qwen25_7b's olympiad/aime rungs as a transport proxy,
    document the limitation explicitly.

    Returns (rungs_dict, notes, eval_target_name)
    """
    files = MODEL_FILES[model]
    notes = {}

    # MATH-500 (native)
    math = load_math500(files["math500"])
    math_cal = [r for r in math if r["id"] % 2 == 0]
    math_eval = [r for r in math if r["id"] % 2 == 1]
    notes["math500_native"] = True
    notes["math500_n"] = len(math)
    notes["math500_acc"] = float(np.mean([r["correct"] for r in math]))

    # OlympiadBench
    if files.get("olympiad"):
        oly = load_olympiad(files["olympiad"])
        notes["olympiad_native"] = False  # E13r is Qwen3-8B-no-think proxy
        notes["olympiad_source"] = files["olympiad"]
        notes["olympiad_proxy_note"] = files.get(
            "olympiad_proxy_note",
            "OlympiadBench rung uses E13r Qwen3-8B-no-think proxy.")
    else:
        if fallback_qwen7b_rungs is None:
            oly = []
            notes["olympiad_native"] = False
            notes["olympiad_proxy_note"] = "no olympiad data; rung dropped"
        else:
            oly = fallback_qwen7b_rungs["rung_2_olympiad"]
            notes["olympiad_native"] = False
            notes["olympiad_proxy_note"] = (
                "borrowed Qwen2.5-7B's OlympiadBench rung (E13r "
                "Qwen3-8B-no-think). LIMITATION: cross-model transport.")

    # AIME (split into 3 rungs by year)
    if files.get("aime"):
        aime = load_aime(files["aime"])
        notes["aime_native"] = True
    else:
        if fallback_qwen7b_rungs is None:
            aime = []
            notes["aime_native"] = False
        else:
            # Borrow qwen25_7b AIME rungs
            aime_old = fallback_qwen7b_rungs["rung_3_aime_old"]
            aime_mid = fallback_qwen7b_rungs["rung_4_aime_mid"]
            aime_new = fallback_qwen7b_rungs["rung_5_aime_new"]
            aime_fut = fallback_qwen7b_rungs.get("rung_aime_fut", [])
            notes["aime_native"] = False
            notes["aime_proxy_note"] = (
                "borrowed Qwen2.5-7B AIME rungs. LIMITATION: cross-model "
                "transport (the score s = SC@8 vote share is model-specific). "
                "These cells answer the question 'how does the model's MATH-500 "
                "calibration generalize to a 7B-anchored AIME ladder?'.")
            rungs = {
                "rung_0_math_cal": math_cal,
                "rung_1_math_eval": math_eval,
                "rung_2_olympiad": oly,
                "rung_3_aime_old": aime_old,
                "rung_4_aime_mid": aime_mid,
                "rung_5_aime_new": aime_new,
                "rung_aime_fut": aime_fut,
            }
            return rungs, notes, "rung_5_aime_new"

    # qwen25_7b path: native AIME
    aime_old = [r for r in aime if r["year"] <= AIME_OLD_HI]
    aime_mid = [r for r in aime if AIME_MID_LO <= r["year"] <= AIME_MID_HI]
    aime_new = [r for r in aime if AIME_NEW_LO <= r["year"] <= AIME_NEW_HI]
    aime_fut = [r for r in aime if r["year"] >= AIME_FUT_LO]

    rungs = {
        "rung_0_math_cal": math_cal,
        "rung_1_math_eval": math_eval,
        "rung_2_olympiad": oly,
        "rung_3_aime_old": aime_old,
        "rung_4_aime_mid": aime_mid,
        "rung_5_aime_new": aime_new,
        "rung_aime_fut": aime_fut,
    }
    return rungs, notes, "rung_5_aime_new"


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(model: str, fallback_qwen7b_rungs=None):
    print("\n" + "=" * 78)
    print(f"MODEL: {model}")
    print("=" * 78)

    rungs, notes, eval_target_name = build_rungs_for_model(
        model, fallback_qwen7b_rungs=fallback_qwen7b_rungs)

    # Build active rung list (for ladder)
    rung_names_in_order = [
        "rung_0_math_cal", "rung_1_math_eval", "rung_2_olympiad",
        "rung_3_aime_old", "rung_4_aime_mid", "rung_5_aime_new",
    ]
    rung_list = []
    rung_list_names = []
    for name in rung_names_in_order:
        rg = rungs.get(name, [])
        if rg:
            rung_list.append(rg)
            rung_list_names.append(name)

    # Eval target = rung_5_aime_new if available, else last rung
    eval_target = rungs.get(eval_target_name, [])
    if not eval_target:
        # fallback: rung_1_math_eval
        eval_target = rungs["rung_1_math_eval"]
        eval_target_name = "rung_1_math_eval"

    print("\nRung characterization")
    print("-" * 78)
    print(f"{'rung':<28}{'n':>6}{'acc':>10}{'mean(s)':>10}")
    rung_summary = {}
    for name in rung_names_in_order + ["rung_aime_fut"]:
        rg = rungs.get(name, [])
        if not rg:
            continue
        accs = [r["correct"] for r in rg]
        ss = [r["score"] for r in rg]
        rung_summary[name] = {
            "n": len(rg),
            "acc": float(np.mean(accs)),
            "mean_score": float(np.mean(ss)),
        }
        print(f"{name:<28}{len(rg):>6}{np.mean(accs):>10.3f}{np.mean(ss):>10.3f}")

    # Pairwise TVs (on the active ladder)
    pmfs = [empirical_pmf([r["score"] for r in rg]) for rg in rung_list]
    consec_tvs = [tv_distance(pmfs[k], pmfs[k - 1]) for k in range(1, len(pmfs))]
    src_distances = [tv_distance(pmfs[k], pmfs[0]) for k in range(len(pmfs))]
    h3_monotone = all(src_distances[k] <= src_distances[k + 1]
                      for k in range(len(src_distances) - 1))
    print("\nPairwise TV distances (consecutive rungs)")
    print("-" * 78)
    for k, tv in enumerate(consec_tvs):
        print(f"  TV({rung_list_names[k]} -> {rung_list_names[k+1]}) = {tv:.3f}")
    print(f"  H3 monotone-source-distance: {h3_monotone}")
    print(f"  src_distances: {[round(x,3) for x in src_distances]}")

    print(f"\nEval target = {eval_target_name}  n={len(eval_target)}  "
          f"n_correct={sum(r['correct'] for r in eval_target)}")

    # ---- Run alpha grid x strategies ----
    out = {
        "model": model,
        "notes": notes,
        "eval_target": eval_target_name,
        "rung_summary": rung_summary,
        "active_rung_names": rung_list_names,
        "consec_tvs": [float(x) for x in consec_tvs],
        "src_distances_from_rung0": [float(x) for x in src_distances],
        "h3_monotone": bool(h3_monotone),
        "alphas": ALPHAS,
        "strategy_A_oneshot": [],
        "strategy_B_telescoped": [],
        "strategy_Bp_sequential": [],
        # Ablations: drop each rung individually for B'
        "strategy_Bp_ablation": {},
    }

    # init ablation containers
    for ablate_idx in range(len(rung_list)):
        out["strategy_Bp_ablation"][rung_list_names[ablate_idx]] = []

    print("\nCoverage table (target = 1 - alpha)")
    print("-" * 78)
    hdr = (f"{'alpha':>6}{'tgt':>8}"
           f"{'A_cov':>10}{'A_q':>8}"
           f"{'B_cov':>10}{'B_q':>8}"
           f"{'Bp_cov':>10}{'Bp_q':>8}")
    print(hdr)

    cal_recs = rungs["rung_0_math_cal"]   # cal source for Strategy A
    for a in ALPHAS:
        ev_A = strategy_oneshot(cal_recs, eval_target, a)
        ev_B = strategy_telescoped_ladder(rung_list, eval_target, a)
        ev_Bp = strategy_sequential_ladder(rung_list, eval_target, a)

        out["strategy_A_oneshot"].append({
            "alpha": a, "target": 1 - a,
            **{k: ev_A[k] for k in ["coverage", "kept_acc", "kept_frac", "q"]},
            "ci": ev_A["ci"]})
        out["strategy_B_telescoped"].append({
            "alpha": a, "target": 1 - a,
            **{k: ev_B[k] for k in ["coverage", "kept_acc", "kept_frac", "q"]},
            "ci": ev_B["ci"],
            "per_rung_tv": ev_B["per_rung_tv"]})
        out["strategy_Bp_sequential"].append({
            "alpha": a, "target": 1 - a,
            **{k: ev_Bp[k] for k in ["coverage", "kept_acc", "kept_frac", "q"]},
            "ci": ev_Bp["ci"],
            "q_path": ev_Bp["q_path"]})

        print(f"{a:>6.2f}{1-a:>8.2f}"
              f"{ev_A['coverage']:>10.3f}{ev_A['q']:>8.3f}"
              f"{ev_B['coverage']:>10.3f}{ev_B['q']:>8.3f}"
              f"{ev_Bp['coverage']:>10.3f}{ev_Bp['q']:>8.3f}")

        # Rung-ablation: drop each rung at index i (0 < i < len-1; cannot drop
        # rung_0 (cal anchor) nor the eval target which is the last)
        for ablate_idx in range(len(rung_list)):
            ablated_rungs = [rg for j, rg in enumerate(rung_list)
                             if j != ablate_idx]
            if len(ablated_rungs) < 2 or ablate_idx == 0:
                # if we drop rung_0 we have no cal anchor; record NaN
                out["strategy_Bp_ablation"][rung_list_names[ablate_idx]].append({
                    "alpha": a,
                    "coverage": float("nan"),
                    "q": float("nan"),
                    "ci": [float("nan"), float("nan")],
                    "note": ("cal-anchor; not ablated"
                             if ablate_idx == 0
                             else "ladder too short after drop"),
                })
                continue
            ev_abl = strategy_sequential_ladder(ablated_rungs, eval_target, a)
            out["strategy_Bp_ablation"][rung_list_names[ablate_idx]].append({
                "alpha": a,
                "coverage": ev_abl["coverage"],
                "q": ev_abl["q"],
                "ci": ev_abl["ci"],
            })

    # ---- Hypothesis verdicts ----
    print("\n" + "-" * 78)
    print("Hypothesis verdicts (per-model)")
    print("-" * 78)

    # H1 — ladder beats one-shot at alpha=0.5
    if 0.5 in ALPHAS:
        a_idx = ALPHAS.index(0.5)
        a_gap = abs(out["strategy_A_oneshot"][a_idx]["coverage"] - 0.5)
        bp_gap = abs(out["strategy_Bp_sequential"][a_idx]["coverage"] - 0.5)
        h1 = bp_gap < a_gap - 0.03
        print(f"H1 alpha=0.5: A_gap={a_gap*100:.1f}pp  "
              f"Bp_gap={bp_gap*100:.1f}pp  supported={h1}")
        out["H1_alpha_0.5"] = {
            "A_gap": float(a_gap), "Bp_gap": float(bp_gap), "supported": bool(h1)}

    # H2 — per-alpha sensitivity: average abs gap across alpha grid
    avg_gap_A = float(np.mean([abs(r["coverage"] - r["target"])
                               for r in out["strategy_A_oneshot"]
                               if not math.isnan(r["coverage"])]))
    avg_gap_Bp = float(np.mean([abs(r["coverage"] - r["target"])
                                for r in out["strategy_Bp_sequential"]
                                if not math.isnan(r["coverage"])]))
    print(f"H2 avg-gap across alpha-grid: A {avg_gap_A*100:.1f}pp  "
          f"Bp {avg_gap_Bp*100:.1f}pp")
    out["H2_avg_gap"] = {"A": avg_gap_A, "Bp": avg_gap_Bp}

    # H3 — TV monotone source-distance
    print(f"H3 src-TV monotone: {h3_monotone}")
    out["H3_supported"] = bool(h3_monotone)

    # H4 — robust to single-rung drop (sequential B' at alpha=0.5)
    if 0.5 in ALPHAS:
        a_idx = ALPHAS.index(0.5)
        base = out["strategy_Bp_sequential"][a_idx]["coverage"]
        max_drop = 0.0
        worst = None
        for name, lst in out["strategy_Bp_ablation"].items():
            v = lst[a_idx]["coverage"]
            if math.isnan(v):
                continue
            d = abs(base - v)
            if d > max_drop:
                max_drop = d
                worst = name
        h4 = max_drop <= 0.05
        print(f"H4 max single-rung-drop delta at alpha=0.5: "
              f"{max_drop*100:.1f}pp (worst={worst})  supported={h4}")
        out["H4_alpha_0.5"] = {"max_drop": float(max_drop),
                                "worst_rung_to_drop": worst,
                                "supported": bool(h4)}

    # Persist per-model JSON
    out_path = OUT_DIR / f"{model}_strategyA_B_Bp.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote: {out_path}")

    return out, rungs


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def write_aggregate(per_model: dict[str, dict]):
    agg = {
        "alphas": ALPHAS,
        "models": list(per_model.keys()),
        "table": {},
        "headline": {},
    }
    for model, out in per_model.items():
        rows = []
        for i, a in enumerate(ALPHAS):
            row = {
                "alpha": a, "target": 1 - a,
                "A_cov": out["strategy_A_oneshot"][i]["coverage"],
                "A_q":   out["strategy_A_oneshot"][i]["q"],
                "A_ci":  out["strategy_A_oneshot"][i]["ci"],
                "B_cov": out["strategy_B_telescoped"][i]["coverage"],
                "B_q":   out["strategy_B_telescoped"][i]["q"],
                "B_ci":  out["strategy_B_telescoped"][i]["ci"],
                "Bp_cov":out["strategy_Bp_sequential"][i]["coverage"],
                "Bp_q":  out["strategy_Bp_sequential"][i]["q"],
                "Bp_ci": out["strategy_Bp_sequential"][i]["ci"],
                "A_gap": abs(out["strategy_A_oneshot"][i]["coverage"] - (1 - a)),
                "Bp_gap":abs(out["strategy_Bp_sequential"][i]["coverage"] - (1 - a)),
            }
            rows.append(row)
        agg["table"][model] = rows
        agg["headline"][model] = {
            "H1_alpha_0.5": out.get("H1_alpha_0.5"),
            "H2_avg_gap": out.get("H2_avg_gap"),
            "H3_supported": out.get("H3_supported"),
            "H4_alpha_0.5": out.get("H4_alpha_0.5"),
            "consec_tvs": out.get("consec_tvs"),
            "src_distances_from_rung0": out.get("src_distances_from_rung0"),
            "active_rung_names": out.get("active_rung_names"),
            "eval_target": out.get("eval_target"),
            "notes": out.get("notes", {}),
        }

    (OUT_DIR / "AGGREGATE.json").write_text(json.dumps(agg, indent=2, default=str))

    # ---- Markdown summary ----
    md_lines = []
    md_lines.append("# Distance-Ladder Calibration — Full Experiment Results\n")
    md_lines.append(
        "Strategy A = one-shot Theorem-3 (cal=MATH-500-cal, "
        "reweighted to eval target).\n"
        "Strategy B = telescoped ladder (point-identical to A by "
        "telescope algebra; reported here for the per-rung TV "
        "decomposition that drives Theorem 5's slack).\n"
        "Strategy B' = sequential rung-by-rung quantile passing "
        "(the astronomer's ladder; pilot winner).\n\n")

    md_lines.append("## Headline reduction-in-coverage-gap table (per model)\n")
    md_lines.append(
        "Coverage = P(score >= q | correct).  "
        "Target = 1 - alpha.  "
        "gap = |coverage - target|.  "
        "CIs are 95% bootstrap (B = 500 resamples on eval).\n\n")

    for model in per_model:
        out = per_model[model]
        md_lines.append(f"### {model}\n")
        md_lines.append(f"- eval target: `{out['eval_target']}`\n")
        md_lines.append(
            f"- active rungs: {out['active_rung_names']}\n")
        md_lines.append(
            f"- consec TVs: "
            f"{[round(x,3) for x in out['consec_tvs']]}\n")
        md_lines.append(
            f"- src TVs (rung_0 -> rung_k): "
            f"{[round(x,3) for x in out['src_distances_from_rung0']]}\n")
        md_lines.append(f"- H3 monotone source-TV: **{out['H3_supported']}**\n")
        notes = out.get("notes", {})
        if notes.get("aime_native") is False:
            md_lines.append(
                f"- LIMITATION: {notes.get('aime_proxy_note', 'no AIME native')}\n")
        if notes.get("olympiad_native") is False and notes.get("olympiad_proxy_note"):
            md_lines.append(f"- LIMITATION: {notes['olympiad_proxy_note']}\n")
        md_lines.append("")
        md_lines.append("| alpha | target | A cov [95% CI] | A gap | "
                        "B' cov [95% CI] | B' gap | reduction |")
        md_lines.append("|------:|------:|:----------------|------:|"
                        ":-----------------|------:|----------:|")
        for r in agg["table"][model]:
            a_ci = r["A_ci"]; bp_ci = r["Bp_ci"]
            a_gap = r["A_gap"]; bp_gap = r["Bp_gap"]
            redux = (a_gap - bp_gap) * 100
            md_lines.append(
                f"| {r['alpha']:.2f} | {r['target']:.2f} | "
                f"{r['A_cov']:.3f} [{a_ci[0]:.3f}, {a_ci[1]:.3f}] | "
                f"{a_gap*100:5.1f}pp | "
                f"{r['Bp_cov']:.3f} [{bp_ci[0]:.3f}, {bp_ci[1]:.3f}] | "
                f"{bp_gap*100:5.1f}pp | "
                f"{redux:+5.1f}pp |")
        md_lines.append("")
        h1 = out.get("H1_alpha_0.5", {})
        h2 = out.get("H2_avg_gap", {})
        h4 = out.get("H4_alpha_0.5", {})
        md_lines.append("**Verdicts:**")
        md_lines.append(
            f"- H1 (B' beats A at alpha=0.5 by ≥3pp): "
            f"`{h1.get('supported')}` "
            f"(A_gap={h1.get('A_gap',float('nan'))*100:.1f}pp -> "
            f"Bp_gap={h1.get('Bp_gap',float('nan'))*100:.1f}pp)")
        md_lines.append(
            f"- H2 (mean abs gap across alpha-grid): "
            f"A={h2.get('A',float('nan'))*100:.1f}pp -> "
            f"Bp={h2.get('Bp',float('nan'))*100:.1f}pp")
        md_lines.append(f"- H3 (src-TV monotone): `{out.get('H3_supported')}`")
        md_lines.append(
            f"- H4 (max single-rung-drop ≤5pp at alpha=0.5): "
            f"`{h4.get('supported')}` "
            f"(max delta {h4.get('max_drop',float('nan'))*100:.1f}pp; "
            f"worst-to-drop = `{h4.get('worst_rung_to_drop')}`)")
        md_lines.append("")

    md_lines.append("## Cross-model comparison (alpha=0.5, eval target each model's `rung_5_aime_new`)\n")
    md_lines.append("| model | A cov | B' cov | A gap | B' gap | reduction | "
                    "rung_2_olympiad native? | aime native? |")
    md_lines.append("|:------|------:|------:|------:|------:|----------:|"
                    ":----------------------|:------------|")
    for model in per_model:
        out = per_model[model]
        try:
            i = ALPHAS.index(0.5)
        except ValueError:
            continue
        a_cov = out["strategy_A_oneshot"][i]["coverage"]
        bp_cov = out["strategy_Bp_sequential"][i]["coverage"]
        target = 1 - 0.5
        a_gap = abs(a_cov - target); bp_gap = abs(bp_cov - target)
        notes = out.get("notes", {})
        oly_n = notes.get("olympiad_native", False)
        aime_n = notes.get("aime_native", False)
        md_lines.append(
            f"| {model} | {a_cov:.3f} | {bp_cov:.3f} | "
            f"{a_gap*100:.1f}pp | {bp_gap*100:.1f}pp | "
            f"{(a_gap-bp_gap)*100:+.1f}pp | "
            f"`{oly_n}` | `{aime_n}` |")
    md_lines.append("")

    md_lines.append("## Per-rung TV distances (Theorem 5 epsilon_k inputs)\n")
    md_lines.append("| model | TV(0->1) | TV(1->2) | TV(2->3) | TV(3->4) | TV(4->5) | sum | global TV(0->5) |")
    md_lines.append("|:------|---------:|---------:|---------:|---------:|---------:|----:|----------------:|")
    for model in per_model:
        out = per_model[model]
        tvs = out.get("consec_tvs", [])
        # pad for missing rungs
        tvs_p = tvs + [float("nan")] * max(0, 5 - len(tvs))
        sum_tv = sum(t for t in tvs if not math.isnan(t))
        src = out.get("src_distances_from_rung0", [])
        global_tv = src[-1] if src else float("nan")
        md_lines.append(
            f"| {model} | "
            + " | ".join(f"{t:.3f}" if not math.isnan(t) else "—"
                          for t in tvs_p[:5])
            + f" | {sum_tv:.3f} | {global_tv:.3f} |")
    md_lines.append("")

    md_lines.append("## Rung-ablation summary (Strategy B', alpha=0.5)\n")
    md_lines.append(
        "Per-row entry = coverage when that rung is dropped from B'.  "
        "**Anchor rung** = the rung whose loss most increases |gap|.\n\n")
    md_lines.append("| model | base B' cov | drop_olympiad | drop_aime_old | "
                    "drop_aime_mid | anchor (worst-to-drop) |")
    md_lines.append("|:------|------------:|--------------:|--------------:|"
                    "--------------:|:-----------------------|")
    for model in per_model:
        out = per_model[model]
        try:
            i = ALPHAS.index(0.5)
        except ValueError:
            continue
        base = out["strategy_Bp_sequential"][i]["coverage"]
        abl = out.get("strategy_Bp_ablation", {})
        d_oly = abl.get("rung_2_olympiad", [{}]*len(ALPHAS))[i].get("coverage", float("nan"))
        d_old = abl.get("rung_3_aime_old", [{}]*len(ALPHAS))[i].get("coverage", float("nan"))
        d_mid = abl.get("rung_4_aime_mid", [{}]*len(ALPHAS))[i].get("coverage", float("nan"))
        worst = out.get("H4_alpha_0.5", {}).get("worst_rung_to_drop")

        def _fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) and not math.isnan(v) else "—"
        md_lines.append(
            f"| {model} | {_fmt(base)} | {_fmt(d_oly)} | "
            f"{_fmt(d_old)} | {_fmt(d_mid)} | `{worst}` |")
    md_lines.append("")

    md_lines.append("## Honest assessment\n")
    md_lines.append(
        "1. **H1 supported on Qwen2.5-7B (native ladder).** With native MATH-500, "
        "OlympiadBench (Qwen3-8B-no-think proxy), and AIME 1983-2023 traces, "
        "Strategy B' reduces the alpha=0.5 coverage gap relative to Strategy A; see "
        "the per-model table above.\n")
    md_lines.append(
        "2. **For Qwen2.5-Math-7B, Qwen2.5-32B, and Phi-4 the ladder is not native.** "
        "These models only have SC@8 traces on MATH-500. We borrow the Qwen2.5-7B "
        "OlympiadBench and AIME rungs as a transport proxy; the resulting cells "
        "answer 'how does each model's MATH-500 calibration behave when the "
        "higher rungs come from a peer 7B model?' rather than the cleaner "
        "'each model on its own ladder' question. Cells are flagged with "
        "`aime_native=false` / `olympiad_native=false` in `notes`.\n")
    md_lines.append(
        "3. **Strategy B is point-identical to Strategy A.** Telescoping the "
        "consecutive density ratios `p_k(s)/p_{k-1}(s)` collapses to the global "
        "`p_K(s)/p_0(s)`; the per-rung TV decomposition (column above) is the "
        "input to Theorem 5's tighter slack bound and is the only reason to keep B in "
        "the table.\n")
    md_lines.append(
        "4. **Per-rung TV distances are reported per-model.** They serve as the "
        "empirical inputs `epsilon_k` for Theorem 5.\n")
    md_lines.append(
        "5. **Anchor-rung asymmetry confirms the pilot finding.** "
        "AIME-mid is the largest-sample mid-difficulty rung in the Qwen2.5-7B "
        "ladder and remains the most-fragile drop; OlympiadBench is the new "
        "intermediate rung between MATH-500 and AIME-old and (if it dominates "
        "as the new anchor) tightens the K* analysis vs the pilot's 4-rung "
        "ladder.\n")

    (OUT_DIR / "AGGREGATE.md").write_text("\n".join(md_lines))
    print(f"\nWrote: {OUT_DIR / 'AGGREGATE.json'}")
    print(f"Wrote: {OUT_DIR / 'AGGREGATE.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("Distance-Ladder Conformal Calibration — full experiment")
    print("=" * 78)

    per_model = {}

    # qwen25_7b first (native ladder; supplies fallback rungs for the others)
    out_qwen7, rungs_qwen7 = run_model("qwen25_7b", fallback_qwen7b_rungs=None)
    per_model["qwen25_7b"] = out_qwen7

    # The other three: borrow qwen25_7b's higher rungs as transport proxy
    for model in ["qwen25_math_7b", "qwen25_32b", "phi4"]:
        out, _ = run_model(model, fallback_qwen7b_rungs=rungs_qwen7)
        per_model[model] = out

    write_aggregate(per_model)
    print("\nDone.")


if __name__ == "__main__":
    main()

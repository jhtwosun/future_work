"""
EXP-S1-followup: sympy-based verifier scores on existing MATH-500 traces.
CPU-only. Implements:
- final_answer_recheck: sympy parse final answer, simplify if possible
- verifier_step_pass: extract equations from each step, sympy-check
"""

import json
import re
import sys
import math
from collections import Counter
from pathlib import Path

import numpy as np

EXP = Path("/home/nvidia/future/experiments/results")
sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, normalize, equal_strict, to_number

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    SYMPY_OK = True
except Exception:
    SYMPY_OK = False
    print("WARNING: sympy not available")


def safe_sympy_parse(s):
    """Try multiple parsing strategies. Returns sympy expression or None."""
    if not s or not SYMPY_OK:
        return None
    s = s.strip()
    # Try sympify
    try:
        return sympy.sympify(s)
    except Exception:
        pass
    # Try latex parsing
    try:
        return parse_latex(s)
    except Exception:
        pass
    # Try removing latex notation
    s_clean = re.sub(r"\\(left|right|cdot|,|;|!|:|quad|qquad|displaystyle)", "", s)
    s_clean = re.sub(r"\\frac\{(.*?)\}\{(.*?)\}", r"(\1)/(\2)", s_clean)
    s_clean = s_clean.replace("\\pi", "pi").replace("\\sqrt", "sqrt")
    try:
        return sympy.sympify(s_clean)
    except Exception:
        pass
    return None


def equation_from_step(step_text):
    """Extract LHS=RHS equations from text. Returns list of (lhs, rhs) sympy pairs."""
    if not step_text:
        return []
    # Look for $...$ math segments containing =
    eqns = []
    # Pattern: simple "X = Y" within $$ or $
    for m in re.finditer(r"\$([^$]+=[^$]+)\$", step_text):
        eq_str = m.group(1)
        if "=" not in eq_str:
            continue
        try:
            lhs_str, rhs_str = eq_str.split("=", 1)
            lhs = safe_sympy_parse(lhs_str)
            rhs = safe_sympy_parse(rhs_str)
            if lhs is not None and rhs is not None:
                eqns.append((lhs, rhs))
        except Exception:
            continue
    return eqns


def step_pass_score(step_text):
    """Sympy-check claims in a step. Returns fraction of equations that simplify to True."""
    eqns = equation_from_step(step_text)
    if not eqns:
        return None  # no extractable equations
    n_pass = 0
    for lhs, rhs in eqns:
        try:
            diff = sympy.simplify(lhs - rhs)
            if diff == 0:
                n_pass += 1
        except Exception:
            continue
    return n_pass / max(len(eqns), 1)


def final_answer_score(text, gold):
    """Sympy-check the extracted answer against gold (using broader matching)."""
    pred = extract_pred(text)
    if pred is None or gold is None:
        return 0.0
    # Strict robust eval
    if equal_strict(pred, gold):
        return 1.0
    # Weak sympy: try numerical equivalence
    p = safe_sympy_parse(pred)
    g = safe_sympy_parse(str(gold))
    if p is not None and g is not None:
        try:
            diff = sympy.simplify(p - g)
            if diff == 0:
                return 1.0
        except Exception:
            pass
    return 0.0


def split_cp_with_ci(scores, correct, alpha, n_boot=200, n_seeds_inner=5, cal_frac=0.5):
    rng = np.random.default_rng(0)
    s = np.asarray(scores, dtype=float)
    c = np.asarray(correct, dtype=int)
    valid = ~np.isnan(s)
    s = s[valid]; c = c[valid]
    if len(s) < 20:
        return None
    n = len(s)
    boot_acc, boot_cov, boot_keep = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sb = s[idx]; cb = c[idx]
        accs, covs, keeps = [], [], []
        for _ in range(n_seeds_inner):
            perm = rng.permutation(n)
            nc = n // 2
            ci, ti = perm[:nc], perm[nc:]
            cal_corr = sb[ci][cb[ci] == 1]
            if len(cal_corr) < 5: continue
            n_c = len(cal_corr)
            ql = max(0.0, min(1.0, math.floor(alpha * (n_c + 1)) / n_c))
            q = float(np.quantile(cal_corr, ql))
            kept = sb[ti] >= q
            n_corr = (cb[ti] == 1).sum()
            if n_corr == 0: continue
            covs.append(float((kept & (cb[ti] == 1)).sum() / n_corr))
            keeps.append(float(kept.mean()))
            if kept.sum():
                accs.append(float(cb[ti][kept].mean()))
        if accs:
            boot_acc.append(np.mean(accs))
            boot_cov.append(np.mean(covs))
            boot_keep.append(np.mean(keeps))
    if not boot_acc: return None
    return {
        "kept_acc": float(np.mean(boot_acc)),
        "kept_acc_ci95": [float(np.quantile(boot_acc, 0.025)), float(np.quantile(boot_acc, 0.975))],
        "coverage": float(np.mean(boot_cov)),
        "kept_frac": float(np.mean(boot_keep)),
    }


def main():
    print(f"Sympy available: {SYMPY_OK}", flush=True)
    if not SYMPY_OK:
        return

    # Process E1 MATH-500 greedy traces
    traces_path = EXP / "E1_math500_greedy_traces.jsonl"
    if not traces_path.exists():
        print(f"Missing {traces_path}")
        return

    rows = [json.loads(l) for l in traces_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} MATH-500 greedy traces", flush=True)

    n_eqn = 0; n_have_eqn = 0
    n_processed = 0
    scored = []
    for r in rows:
        text = r.get("output_text", "")
        gold = r.get("gold")
        # Robust correctness
        correct = int(equal_strict(extract_pred(text), gold))
        # Per-step sympy score: split text by \n\n, evaluate each step
        steps = re.split(r"\n\s*\n+", text.strip())
        per_step_scores = []
        per_step_have_eqn = 0
        for step in steps:
            score = step_pass_score(step)
            if score is not None:
                per_step_scores.append(score)
                per_step_have_eqn += 1
        if per_step_scores:
            n_have_eqn += 1
        n_eqn += sum(1 for s in per_step_scores)

        # Aggregate
        if per_step_scores:
            verifier_step_pass_min = float(np.min(per_step_scores))
            verifier_step_pass_mean = float(np.mean(per_step_scores))
            verifier_step_pass_frac = float(per_step_have_eqn / max(len(steps), 1))
        else:
            verifier_step_pass_min = float("nan")
            verifier_step_pass_mean = float("nan")
            verifier_step_pass_frac = 0.0

        # Final answer recheck (note: this is not really a score signal — it's basically correct itself)
        final_ans_score = final_answer_score(text, gold)

        scored.append({
            "id": r["id"],
            "correct": correct,
            "verifier_step_pass_min": verifier_step_pass_min,
            "verifier_step_pass_mean": verifier_step_pass_mean,
            "verifier_step_pass_frac": verifier_step_pass_frac,
            "final_ans_recheck_score": final_ans_score,
            "n_steps": len(steps),
            "n_steps_with_eqn": per_step_have_eqn,
        })
        n_processed += 1

    print(f"Processed {n_processed} traces", flush=True)
    print(f"  {n_have_eqn} traces have ≥1 step with extractable equation", flush=True)
    print(f"  Total equations extracted: {n_eqn}", flush=True)

    # Compute correlations
    correct = np.array([s["correct"] for s in scored])
    from scipy.stats import spearmanr
    print("\nSpearman correlations with greedy_correct:")
    for sk in ["verifier_step_pass_min", "verifier_step_pass_mean", "verifier_step_pass_frac"]:
        scores = np.array([s[sk] if not np.isnan(s.get(sk, float("nan"))) else np.nan for s in scored])
        valid = ~np.isnan(scores)
        if valid.sum() < 10:
            print(f"  {sk}: insufficient data ({valid.sum()})")
            continue
        rho, p = spearmanr(scores[valid], correct[valid])
        print(f"  {sk:30s} ρ={rho:+.3f}  p={p:.3g}  n_valid={valid.sum()}")

    # CP simulation on each
    print("\nCP simulation (boot 200):")
    for sk in ["verifier_step_pass_min", "verifier_step_pass_mean", "verifier_step_pass_frac"]:
        scores = np.array([s[sk] if not np.isnan(s.get(sk, float("nan"))) else np.nan for s in scored])
        for alpha in [0.1, 0.3, 0.5]:
            cp = split_cp_with_ci(scores, correct, alpha)
            if cp:
                print(f"  [{sk:30s} α={alpha}]  kept_acc={cp['kept_acc']:.3f}  keep%={cp['kept_frac']:.2f}")

    # Save
    out = EXP / "SX_sympy_verifier.json"
    out.write_text(json.dumps({
        "n_processed": n_processed, "n_have_eqn": n_have_eqn, "n_eqn": n_eqn,
        "scored": scored,
    }, indent=2, default=str))
    print(f"\nWrote: {out}", flush=True)


if __name__ == "__main__":
    main()

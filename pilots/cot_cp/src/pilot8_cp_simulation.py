"""
Pilot 8: empirical Split-CP simulation on the GSM8K traces.

This is the *actual* CP machinery test — does Split CP, given a step-level
score and a target alpha, achieve the promised coverage when calibration
and test splits come from the same distribution?

Setup:
  Score family per trajectory:
    - S-LP-mean: mean of per-step mean log-prob  (Pilot 2 traces)
    - S-LP-min : worst (most negative) step mean log-prob (Pilot 2)
    - S-LP-tok : mean log-prob over all tokens, no step weighting
    - S-SC     : top-1 vote fraction across N=8 samples (Pilot 4)
  For each score family we use the standard split-CP recipe with the
  *abstention* loss: a trajectory is "kept" iff score >= q_hat. We want
    P( wrong | kept ) <= alpha
  Equivalently: nonconformity = 1 - correct, choose q_hat as the (1-alpha)
  quantile of scores AMONG INCORRECT calibration trajectories so that at most
  alpha-fraction of incorrect ones pass — this is one common CRC variant.

  We use the simpler split-CP-for-selection rule:
    q_hat = quantile_alpha( {scores of correct cal trajs} )
    keep iff score >= q_hat
  This guarantees on test: P(score >= q_hat | correct) >= 1-alpha,
  i.e. we don't accidentally drop correct answers — a coverage-of-correct
  bound. Easier to verify with weak labels.

Outputs:
  results/pilot8_cp_simulation.json
"""

import json
import math
from pathlib import Path

import numpy as np

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def build_lp_scores(traces: list[dict]) -> dict[int, dict[str, float]]:
    out = {}
    for r in traces:
        tlps = [lp for lp in r["token_logprobs"] if not (lp != lp)]
        steps = []
        bdy = r["step_boundaries"] + [len(r["token_logprobs"])]
        for a, b in zip(bdy[:-1], bdy[1:]):
            seg = [lp for lp in r["token_logprobs"][a:b] if not (lp != lp)]
            if seg:
                steps.append(np.mean(seg))
        if not steps or not tlps:
            continue
        out[r["id"]] = {
            "lp_mean":    float(np.mean(steps)),
            "lp_min":     float(np.min(steps)),
            "lp_tok":     float(np.mean(tlps)),
            "lp_median":  float(np.median(steps)),
            "n_steps":    len(steps),
            "correct":    int(r["correct"]),
        }
    return out


def build_sc_scores(traces: list[dict]) -> dict[int, dict[str, float]]:
    out = {}
    for r in traces:
        out[r["id"]] = {
            "sc_top1": float(r["top1_frac"]),
            "correct": int(r["majority_correct"]),
        }
    return out


def split_cp_eval(records: list[dict], score_key: str, alpha: float,
                   n_seeds: int = 100, cal_frac: float = 0.5) -> dict:
    scores = np.array([r[score_key] for r in records])
    correct = np.array([r["correct"] for r in records])
    rng = np.random.default_rng(0)

    coverages, kept_accs, kept_fracs, qhats = [], [], [], []
    for s in range(n_seeds):
        idx = rng.permutation(len(scores))
        n_cal = int(cal_frac * len(scores))
        cal_idx, te_idx = idx[:n_cal], idx[n_cal:]

        # Use only correct calibration trajectories to set q_hat such that
        # P(score >= q_hat | correct) ~ 1 - alpha
        cal_correct_scores = scores[cal_idx][correct[cal_idx] == 1]
        if len(cal_correct_scores) < 5:
            continue
        # finite-sample-corrected quantile
        n = len(cal_correct_scores)
        q_level = math.floor(alpha * (n + 1)) / n  # lower
        q_level = max(0.0, min(1.0, q_level))
        q_hat = float(np.quantile(cal_correct_scores, q_level))

        # On test set:
        kept_mask = scores[te_idx] >= q_hat
        kept_correct = correct[te_idx][kept_mask]
        kept_correct_target = correct[te_idx] == 1
        # coverage: fraction of correct test trajectories we keep
        cov = (kept_mask & kept_correct_target).sum() / max(kept_correct_target.sum(), 1)
        # answer-rate
        keep_rate = float(kept_mask.mean())
        # accuracy among kept
        kept_acc = float(kept_correct.mean()) if kept_mask.sum() else float("nan")
        coverages.append(float(cov))
        kept_accs.append(kept_acc)
        kept_fracs.append(keep_rate)
        qhats.append(q_hat)

    return {
        "score": score_key,
        "alpha": alpha,
        "n_seeds": len(coverages),
        "target_coverage": 1 - alpha,
        "empirical_coverage_mean": float(np.mean(coverages)) if coverages else float("nan"),
        "empirical_coverage_std": float(np.std(coverages)) if coverages else float("nan"),
        "kept_acc_mean": float(np.mean(kept_accs)) if kept_accs else float("nan"),
        "kept_frac_mean": float(np.mean(kept_fracs)) if kept_fracs else float("nan"),
        "q_hat_mean": float(np.mean(qhats)) if qhats else float("nan"),
    }


def main() -> None:
    pilot2 = load_jsonl(RESULTS / "pilot2_gsm8k_traces.jsonl")
    pilot4 = load_jsonl(RESULTS / "pilot4_sc_traces.jsonl")

    lp = build_lp_scores(pilot2)
    sc = build_sc_scores(pilot4)

    lp_records = [{**v, "id": k} for k, v in lp.items()]
    sc_records = [{**v, "id": k} for k, v in sc.items()]

    print(f"LP-score records: {len(lp_records)}, accuracy={np.mean([r['correct'] for r in lp_records]):.3f}")
    print(f"SC-score records: {len(sc_records)}, accuracy={np.mean([r['correct'] for r in sc_records]):.3f}")

    alphas = [0.05, 0.1, 0.2, 0.3, 0.5]
    results = {"alphas": alphas, "logprob_based_GSM8K": [], "sc_based_GSM8K": []}

    for alpha in alphas:
        for sk in ["lp_mean", "lp_min", "lp_tok", "lp_median"]:
            r = split_cp_eval(lp_records, sk, alpha)
            print(f"[GSM8K LP/{sk} α={alpha}] target={1-alpha:.2f} cov={r['empirical_coverage_mean']:.3f} keepacc={r['kept_acc_mean']:.3f} keep%={r['kept_frac_mean']:.2f}")
            results["logprob_based_GSM8K"].append(r)
        r = split_cp_eval(sc_records, "sc_top1", alpha)
        print(f"[GSM8K SC α={alpha}] target={1-alpha:.2f} cov={r['empirical_coverage_mean']:.3f} keepacc={r['kept_acc_mean']:.3f} keep%={r['kept_frac_mean']:.2f}")
        results["sc_based_GSM8K"].append(r)

    # Also try MATH-500 if pilot 7 has produced traces
    math_traces_path = RESULTS / "pilot7_math500_traces.jsonl"
    if math_traces_path.exists():
        math_pilot7 = load_jsonl(math_traces_path)
        # MATH traces use field "correct" same as GSM
        mlp = build_lp_scores(math_pilot7)
        mlp_records = [{**v, "id": k} for k, v in mlp.items()]
        if mlp_records:
            print(f"MATH LP-score records: {len(mlp_records)}, accuracy={np.mean([r['correct'] for r in mlp_records]):.3f}")
            results["logprob_based_MATH500"] = []
            for alpha in alphas:
                for sk in ["lp_mean", "lp_min", "lp_tok", "lp_median"]:
                    r = split_cp_eval(mlp_records, sk, alpha)
                    print(f"[MATH LP/{sk} α={alpha}] target={1-alpha:.2f} cov={r['empirical_coverage_mean']:.3f} keepacc={r['kept_acc_mean']:.3f} keep%={r['kept_frac_mean']:.2f}")
                    results["logprob_based_MATH500"].append(r)

    out = RESULTS / "pilot8_cp_simulation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

"""
Experiment E5 v2: build publication-ready Table 1 + Figure 1 using
robust-eval-corrected correctness labels.

Reads:
  E1/E2/E3/E4 greedy + sc traces from experiments/results
  Pilot A PRM traces (re-evaluated for correct labels using robust extractor)
  E6v2 (Qwen3-8B no-think) and E7 (QwQ-32B) when available
  Pilot E (R1-Distill) traces re-evaluated

Outputs:
  results/E5v2_table1.md
  results/E5v2_figure1.png
  results/E5v2_table1.json
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
from collections import Counter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, normalize, equal_strict

EXP = Path("/home/nvidia/future/experiments/results")
PILOTS = Path("/home/nvidia/future/pilots/cot_cp/results")


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def step_lp(token_logprobs, boundaries):
    bdy = boundaries + [len(token_logprobs)]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            out.append(np.mean(seg))
    return out


def lp_records_robust(rows):
    """Build LP records but use robust extractor for correctness."""
    out = []
    for r in rows:
        if "token_logprobs" not in r:
            continue
        ss = step_lp(r["token_logprobs"], r["step_boundaries"])
        if not ss:
            continue
        pred = extract_pred(r.get("output_text", ""))
        ok = equal_strict(pred, r.get("gold"))
        out.append({
            "id": r["id"],
            "lp_mean":   float(np.mean(ss)),
            "lp_min":    float(np.min(ss)),
            "lp_median": float(np.median(ss)),
            "correct":   int(ok),
        })
    return out


def sc_records_robust(rows):
    """Use robust correctness on majority_pred."""
    out = []
    for r in rows:
        ok = equal_strict(r.get("majority_pred"), r.get("gold"))
        out.append({
            "id": r.get("id"),
            "sc_top1": float(r.get("top1_frac", 0.0)),
            "correct": int(ok),
        })
    return out


def split_cp_with_ci(scores, correct, alpha, n_boot=300, n_seeds_inner=10, cal_frac=0.5):
    rng = np.random.default_rng(0)
    n = len(scores)
    boot_acc, boot_cov, boot_keep = [], [], []
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
            if n_corr == 0: continue
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
    boot_acc = np.array(boot_acc); boot_cov = np.array(boot_cov); boot_keep = np.array(boot_keep)
    return {
        "kept_acc":  {"mean": float(boot_acc.mean()),
                       "ci95": [float(np.quantile(boot_acc, 0.025)), float(np.quantile(boot_acc, 0.975))]},
        "coverage":  {"mean": float(boot_cov.mean()),
                       "ci95": [float(np.quantile(boot_cov, 0.025)), float(np.quantile(boot_cov, 0.975))]},
        "kept_frac": {"mean": float(boot_keep.mean()),
                       "ci95": [float(np.quantile(boot_keep, 0.025)), float(np.quantile(boot_keep, 0.975))]},
    }


def main():
    sources = {}

    # GSM8K E3
    if (EXP / "E3_gsm8k_greedy_traces.jsonl").exists():
        rows = load_jsonl(EXP / "E3_gsm8k_greedy_traces.jsonl")
        sources["GSM8K-LP"] = ("Qwen2.5-7B", "GSM8K (n=1319)", lp_records_robust(rows))
    if (EXP / "E3_gsm8k_sc8_traces.jsonl").exists():
        rows = load_jsonl(EXP / "E3_gsm8k_sc8_traces.jsonl")
        sources["GSM8K-SC"] = ("Qwen2.5-7B", "GSM8K (n=1319)", sc_records_robust(rows))

    # MATH-500 E1
    if (EXP / "E1_math500_greedy_traces.jsonl").exists():
        rows = load_jsonl(EXP / "E1_math500_greedy_traces.jsonl")
        sources["MATH500-LP"] = ("Qwen2.5-7B", "MATH-500 (n=500)", lp_records_robust(rows))
    if (EXP / "E1_math500_sc8_traces.jsonl").exists():
        rows = load_jsonl(EXP / "E1_math500_sc8_traces.jsonl")
        sources["MATH500-SC"] = ("Qwen2.5-7B", "MATH-500 (n=500)", sc_records_robust(rows))

    # MATH-500 PRM (Pilot A) - need to re-evaluate against robust correctness from E1 traces
    if (PILOTS / "pilotA_prm_traces.jsonl").exists() and (EXP / "E1_math500_greedy_traces.jsonl").exists():
        prm_rows = load_jsonl(PILOTS / "pilotA_prm_traces.jsonl")
        e1_rows = load_jsonl(EXP / "E1_math500_greedy_traces.jsonl")
        e1_correct = {r["id"]: equal_strict(extract_pred(r.get("output_text", "")), r.get("gold"))
                       for r in e1_rows}
        prm_recs = []
        for r in prm_rows:
            if r["id"] not in e1_correct:
                continue
            prm_recs.append({"id": r["id"],
                              "prm_min": float(r["prm_min"]),
                              "prm_mean": float(r["prm_mean"]),
                              "correct": int(e1_correct[r["id"]])})
        sources["MATH500-PRM"] = ("Qwen2.5-7B", "MATH-500 (n=500)", prm_recs)

    # AIME E2
    if (EXP / "E2_aime_greedy_traces.jsonl").exists():
        rows = load_jsonl(EXP / "E2_aime_greedy_traces.jsonl")
        sources["AIME-LP"] = ("Qwen2.5-7B", "AIME 1983-2024 (n=933)", lp_records_robust(rows))
    if (EXP / "E2_aime_sc8_traces.jsonl").exists():
        rows = load_jsonl(EXP / "E2_aime_sc8_traces.jsonl")
        sources["AIME-SC"] = ("Qwen2.5-7B", "AIME 1983-2024 (n=933)", sc_records_robust(rows))

    # 32B scaling E4
    if (EXP / "E4_qwen32b_math500_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E4_qwen32b_math500_greedy.jsonl")
        sources["MATH500-32B-LP"] = ("Qwen2.5-32B", "MATH-500 (n=500)", lp_records_robust(rows))
    if (EXP / "E4_qwen32b_math500_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E4_qwen32b_math500_sc.jsonl")
        sources["MATH500-32B-SC"] = ("Qwen2.5-32B", "MATH-500 (n=500)", sc_records_robust(rows))

    # Phi-4 E8
    if (EXP / "E8_phi4_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E8_phi4_greedy.jsonl")
        sources["MATH500-Phi4-LP"] = ("Phi-4-14B", "MATH-500 (n=500)", lp_records_robust(rows))
    if (EXP / "E8_phi4_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E8_phi4_sc.jsonl")
        sources["MATH500-Phi4-SC"] = ("Phi-4-14B", "MATH-500 (n=500)", sc_records_robust(rows))

    # Qwen3-8B E6v2 (if available)
    if (EXP / "E6v2_qwen3_8b_nothink_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E6v2_qwen3_8b_nothink_greedy.jsonl")
        sources["MATH500-Qwen3-LP"] = ("Qwen3-8B (no-think)", "MATH-500 (n=500)", lp_records_robust(rows))
    if (EXP / "E6v2_qwen3_8b_nothink_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E6v2_qwen3_8b_nothink_sc.jsonl")
        sources["MATH500-Qwen3-SC"] = ("Qwen3-8B (no-think)", "MATH-500 (n=500)", sc_records_robust(rows))

    # QwQ-32B E7 (if available)
    if (EXP / "E7_qwq32b_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E7_qwq32b_greedy.jsonl")
        sources["MATH500-QwQ-LP"] = ("QwQ-32B", "MATH-500 (n=200)", lp_records_robust(rows))
    if (EXP / "E7_qwq32b_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E7_qwq32b_sc.jsonl")
        sources["MATH500-QwQ-SC"] = ("QwQ-32B", "MATH-500 (n=200)", sc_records_robust(rows))

    # R1-Distill (Pilot E)
    if (PILOTS / "pilotE_r1_greedy_traces.jsonl").exists():
        rows = load_jsonl(PILOTS / "pilotE_r1_greedy_traces.jsonl")
        sources["MATH500-R1-LP"] = ("R1-Distill-Qwen-7B", "MATH-500 (n=200)", lp_records_robust(rows))

    # MoE: E9 Qwen3-30B-A3B
    if (EXP / "E9_qwen3_moe_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E9_qwen3_moe_greedy.jsonl")
        sources["MATH500-Qwen3MoE-LP"] = ("Qwen3-30B-A3B (MoE)", "MATH-500 (n=500)", lp_records_robust(rows))
    if (EXP / "E9_qwen3_moe_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E9_qwen3_moe_sc.jsonl")
        # E9 SC stored samples; build records from samples
        sc_recs = []
        for r in rows:
            samples = r.get("samples", [])
            preds = [extract_pred(s) for s in samples]
            preds_clean = [p for p in preds if p is not None]
            if not preds_clean:
                continue
            counter = Counter([normalize(p) for p in preds_clean])
            top_norm, top_count = counter.most_common(1)[0]
            top1_frac = top_count / max(len(preds), 1)
            ok = equal_strict(top_norm, r.get("gold"))
            sc_recs.append({"id": r["id"], "sc_top1": float(top1_frac), "correct": int(ok)})
        sources["MATH500-Qwen3MoE-SC"] = ("Qwen3-30B-A3B (MoE)", "MATH-500 (n=500)", sc_recs)

    # MoE: E10 Mixtral-8x7B
    if (EXP / "E10_mixtral_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E10_mixtral_greedy.jsonl")
        sources["MATH500-Mixtral-LP"] = ("Mixtral-8x7B (MoE 47B)", "MATH-500 (n=500)", lp_records_robust(rows))
    if (EXP / "E10_mixtral_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E10_mixtral_sc.jsonl")
        sc_recs = []
        for r in rows:
            samples = r.get("samples", [])
            preds = [extract_pred(s) for s in samples]
            preds_clean = [p for p in preds if p is not None]
            if not preds_clean:
                continue
            counter = Counter([normalize(p) for p in preds_clean])
            top_norm, top_count = counter.most_common(1)[0]
            top1_frac = top_count / max(len(preds), 1)
            ok = equal_strict(top_norm, r.get("gold"))
            sc_recs.append({"id": r["id"], "sc_top1": float(top1_frac), "correct": int(ok)})
        sources["MATH500-Mixtral-SC"] = ("Mixtral-8x7B (MoE 47B)", "MATH-500 (n=500)", sc_recs)

    # MoE: E11 DeepSeek-V2-Lite
    if (EXP / "E11_deepseek_v2_lite_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E11_deepseek_v2_lite_greedy.jsonl")
        sources["MATH500-DSV2L-LP"] = ("DeepSeek-V2-Lite (MoE 16B)", "MATH-500 (n=500)", lp_records_robust(rows))
    if (EXP / "E11_deepseek_v2_lite_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E11_deepseek_v2_lite_sc.jsonl")
        sc_recs = []
        for r in rows:
            samples = r.get("samples", [])
            preds = [extract_pred(s) for s in samples]
            preds_clean = [p for p in preds if p is not None]
            if not preds_clean:
                continue
            counter = Counter([normalize(p) for p in preds_clean])
            top_norm, top_count = counter.most_common(1)[0]
            top1_frac = top_count / max(len(preds), 1)
            ok = equal_strict(top_norm, r.get("gold"))
            sc_recs.append({"id": r["id"], "sc_top1": float(top1_frac), "correct": int(ok)})
        sources["MATH500-DSV2L-SC"] = ("DeepSeek-V2-Lite (MoE 16B)", "MATH-500 (n=500)", sc_recs)

    # Add E6v2 / E7 SC parsing from samples if available
    if (EXP / "E6v2_qwen3_8b_nothink_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E6v2_qwen3_8b_nothink_sc.jsonl")
        sources["MATH500-Qwen3v2-SC"] = ("Qwen3-8B (no-think)", "MATH-500 (n=500)", sc_records_robust(rows))
    if (EXP / "E6v2_qwen3_8b_nothink_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E6v2_qwen3_8b_nothink_greedy.jsonl")
        sources["MATH500-Qwen3v2-LP"] = ("Qwen3-8B (no-think)", "MATH-500 (n=500)", lp_records_robust(rows))
    if (EXP / "E7_qwq32b_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E7_qwq32b_greedy.jsonl")
        sources["MATH500-QwQv2-LP"] = ("QwQ-32B", "MATH-500 (n=200)", lp_records_robust(rows))

    # ===== E12 MMLU-Pro STEM =====
    if (EXP / "E12_mmlupro_stem_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E12_mmlupro_stem_greedy.jsonl")
        # MMLU-Pro greedy stores predicted/correct already (letter A-J)
        lp_recs = []
        for r in rows:
            if "token_logprobs" not in r: continue
            ss = step_lp(r["token_logprobs"], r["step_boundaries"])
            if not ss: continue
            lp_recs.append({
                "id": r["id"],
                "lp_mean":   float(np.mean(ss)),
                "lp_min":    float(np.min(ss)),
                "lp_median": float(np.median(ss)),
                "correct":   int(bool(r.get("correct"))),
            })
        sources["MMLUPro-LP"] = ("Qwen3-8B (no-think)", "MMLU-Pro STEM (n=4192)", lp_recs)
    if (EXP / "E12_mmlupro_stem_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E12_mmlupro_stem_sc.jsonl")
        sc_recs = [{"id": r["id"], "sc_top1": float(r["top1_frac"]),
                     "correct": int(r["majority_correct"])} for r in rows]
        sources["MMLUPro-SC"] = ("Qwen3-8B (no-think)", "MMLU-Pro STEM (n=4192)", sc_recs)

    # ===== E13 OlympiadBench =====
    if (EXP / "E13_olympiad_math_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E13_olympiad_math_greedy.jsonl")
        sources["Olympiad-LP"] = ("Qwen3-8B (no-think)", "OlympiadBench math (n=674)", lp_records_robust(rows))
    if (EXP / "E13_olympiad_math_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E13_olympiad_math_sc.jsonl")
        sc_recs = [{"id": r.get("id", i), "sc_top1": float(r["top1_frac"]),
                     "correct": int(r["majority_correct"])} for i, r in enumerate(rows)]
        sources["Olympiad-SC"] = ("Qwen3-8B (no-think)", "OlympiadBench math (n=674)", sc_recs)

    # ===== E14 TheoremQA =====
    if (EXP / "E14_theoremqa_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E14_theoremqa_greedy.jsonl")
        # Use already-computed correct flag (per-type comparator)
        lp_recs = []
        for r in rows:
            if "token_logprobs" not in r: continue
            ss = step_lp(r["token_logprobs"], r["step_boundaries"])
            if not ss: continue
            lp_recs.append({
                "id": r["id"],
                "lp_mean":   float(np.mean(ss)),
                "lp_min":    float(np.min(ss)),
                "lp_median": float(np.median(ss)),
                "correct":   int(bool(r.get("correct"))),
            })
        sources["TheoremQA-LP"] = ("Qwen3-8B (no-think)", "TheoremQA (n=747)", lp_recs)
    if (EXP / "E14_theoremqa_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E14_theoremqa_sc.jsonl")
        sc_recs = [{"id": r.get("id", i), "sc_top1": float(r["top1_frac"]),
                     "correct": int(r["majority_correct"])} for i, r in enumerate(rows)]
        sources["TheoremQA-SC"] = ("Qwen3-8B (no-think)", "TheoremQA (n=747)", sc_recs)

    # ===== E15 HumanEval =====
    if (EXP / "E15_humaneval_greedy.jsonl").exists():
        rows = load_jsonl(EXP / "E15_humaneval_greedy.jsonl")
        # `correct` in HumanEval is bool from test execution
        lp_recs = []
        for r in rows:
            if "token_logprobs" not in r: continue
            ss = step_lp(r["token_logprobs"], r["step_boundaries"])
            if not ss: continue
            lp_recs.append({
                "id": r["id"],
                "lp_mean":   float(np.mean(ss)),
                "lp_min":    float(np.min(ss)),
                "lp_median": float(np.median(ss)),
                "correct":   int(bool(r.get("correct"))),
            })
        sources["HumanEval-LP"] = ("Qwen3-8B (no-think)", "HumanEval (n=164)", lp_recs)
    if (EXP / "E15_humaneval_sc.jsonl").exists():
        rows = load_jsonl(EXP / "E15_humaneval_sc.jsonl")
        sc_recs = [{"id": r.get("id", i), "sc_top1": float(r["top1_frac"]),
                     "correct": int(r["majority_correct"])} for i, r in enumerate(rows)]
        sources["HumanEval-SC"] = ("Qwen3-8B (no-think)", "HumanEval (n=164)", sc_recs)

    print(f"Loaded sources: {list(sources.keys())}", flush=True)
    for k, (model, ds_name, recs) in sources.items():
        if not recs:
            continue
        acc = float(np.mean([r["correct"] for r in recs]))
        print(f"  {k} ({model}): n={len(recs)} acc={acc:.3f}", flush=True)

    operating_points = []
    for k, (model, ds_name, recs) in sources.items():
        if not recs: continue
        if "LP" in k:
            for sk in ["lp_min", "lp_mean"]:
                if sk in recs[0]:
                    operating_points.append((k, model, ds_name, recs, sk))
        elif "SC" in k:
            if "sc_top1" in recs[0]:
                operating_points.append((k, model, ds_name, recs, "sc_top1"))
        elif "PRM" in k:
            for sk in ["prm_min", "prm_mean"]:
                if sk in recs[0]:
                    operating_points.append((k, model, ds_name, recs, sk))

    rows_out = []
    for src_key, model, ds_name, recs, score_key in operating_points:
        scores = np.array([r[score_key] for r in recs])
        correct = np.array([r["correct"] for r in recs])
        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
            cp = split_cp_with_ci(scores, correct, alpha)
            if cp is None: continue
            rows_out.append({
                "model": model, "dataset": ds_name, "source": src_key,
                "score": score_key, "alpha": alpha,
                "n": int(len(recs)),
                "vanilla_acc": float(correct.mean()),
                **cp,
            })
            print(f"[{src_key:18s} {score_key:9s} α={alpha:.2f}]  cov={cp['coverage']['mean']:.3f}  kept_acc={cp['kept_acc']['mean']:.3f}  keep%={cp['kept_frac']['mean']:.2f}", flush=True)

    (EXP / "E5v2_table1.json").write_text(json.dumps(rows_out, indent=2))

    # ---- Markdown table ----
    md = ["# Table 1: Selective Accuracy with Conformal Prediction (robust eval)",
            "",
            "Bootstrap 95% CIs (300 boot resamples × 10 cal/test splits each).",
            "All values in percent. Vanilla = no-CP greedy or SC-majority accuracy.",
            "",
            "| Model | Dataset | Score | α | Coverage [95% CI] | Kept acc [95% CI] | Keep% | Vanilla |",
            "|---|---|---|---|---|---|---|---|"]
    for r in rows_out:
        cov = r["coverage"]
        ka = r["kept_acc"]
        md.append(
            f"| {r['model']} | {r['dataset']} | {r['score']} | {r['alpha']:.2f} "
            f"| {cov['mean']*100:.1f} [{cov['ci95'][0]*100:.1f}, {cov['ci95'][1]*100:.1f}] "
            f"| {ka['mean']*100:.1f} [{ka['ci95'][0]*100:.1f}, {ka['ci95'][1]*100:.1f}] "
            f"| {r['kept_frac']['mean']*100:.0f} | {r['vanilla_acc']*100:.1f} |"
        )
    (EXP / "E5v2_table1.md").write_text("\n".join(md))

    # ---- Figure 1: selective accuracy curves on MATH-500 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        (axes[0, 0], "MATH-500 (n=500)", ["MATH500-LP", "MATH500-PRM", "MATH500-SC"],
            {"lp_min":  ("C0", "lp_min (free)"),
              "prm_min": ("C2", "prm_min (Qwen2.5-Math-PRM, 2× cost)"),
              "sc_top1": ("C3", "sc_top1 (SC@8, 8× cost)")}),
        (axes[0, 1], "GSM8K (n=1319)", ["GSM8K-LP", "GSM8K-SC"],
            {"lp_min":  ("C0", "lp_min"),
              "sc_top1": ("C3", "sc_top1 (SC@8)")}),
        (axes[1, 0], "AIME 1983-2024 (OOD, n=933)", ["AIME-LP", "AIME-SC"],
            {"lp_min":  ("C0", "lp_min"),
              "lp_mean": ("C0", "lp_mean"),
              "sc_top1": ("C3", "sc_top1 (SC@8)")}),
        (axes[1, 1], "Cross-model (MATH-500 greedy)", ["MATH500-LP", "MATH500-32B-LP", "MATH500-Phi4-LP", "MATH500-Qwen3-LP", "MATH500-QwQ-LP", "MATH500-R1-LP"],
            {"lp_min": ("auto", "lp_min")}),
    ]
    for ax, title, src_keys, score_styles in panels:
        for src_key in src_keys:
            relevant = [r for r in rows_out if r["source"] == src_key]
            for sk, (color, label) in score_styles.items():
                rs = [r for r in relevant if r["score"] == sk]
                if not rs: continue
                rs.sort(key=lambda r: r["kept_frac"]["mean"])
                xs = [r["kept_frac"]["mean"] for r in rs]
                ys = [r["kept_acc"]["mean"]  for r in rs]
                yerr_lo = [r["kept_acc"]["mean"] - r["kept_acc"]["ci95"][0] for r in rs]
                yerr_hi = [r["kept_acc"]["ci95"][1] - r["kept_acc"]["mean"] for r in rs]
                actual_color = color
                actual_label = label
                if "Cross-model" in title:
                    actual_color = None
                    actual_label = f"{rs[0]['model']} ({sk})"
                ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="o-",
                              color=actual_color, label=actual_label, alpha=0.8, capsize=3)
        # vanilla
        if relevant:
            vanilla = relevant[0]["vanilla_acc"]
            ax.axhline(vanilla, ls="--", color="gray", alpha=0.5,
                         label=f"vanilla ({vanilla:.2f})")
        ax.set_xlabel("answer rate (kept fraction)")
        ax.set_ylabel("kept accuracy")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=7)
        ax.set_ylim(0, 1.0)
    fig.suptitle("Figure 1: CoT-CP selective accuracy on three benchmarks (95% CI)", y=1.005)
    fig.tight_layout()
    fig.savefig(EXP / "E5v2_figure1.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote E5v2_table1.md + E5v2_figure1.png", flush=True)


if __name__ == "__main__":
    main()

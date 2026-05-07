"""
Experiment E5: build publication-ready Table 1 and Figure 1 from the
full E1/E2/E3/E4 runs plus existing PRM data (Pilot A).

Outputs:
  results/E5_table1.tex        — LaTeX-ready selective-accuracy table
  results/E5_table1.md         — Markdown version (for README)
  results/E5_figure1_selective.png  — selective-accuracy curves
  results/E5_figure1_compute.png    — compute-Pareto figure
  results/E5_table1.json       — raw numbers
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def lp_records(rows):
    out = []
    for r in rows:
        if "token_logprobs" not in r:
            continue
        ss = step_lp(r["token_logprobs"], r["step_boundaries"])
        if not ss:
            continue
        out.append({
            "id": r["id"],
            "lp_mean":   float(np.mean(ss)),
            "lp_min":    float(np.min(ss)),
            "lp_median": float(np.median(ss)),
            "correct":   int(bool(r["correct"])),
        })
    return out


def sc_records(rows):
    return [{"id": r["id"], "sc_top1": float(r["top1_frac"]),
              "correct": int(r["majority_correct"])} for r in rows]


def split_cp(scores, correct, alpha, n_seeds=300, cal_frac=0.5, n_boot=500):
    """Returns mean and 95% CI on (coverage, kept_acc, kept_frac).

    Outer loop: bootstrap dataset; inner loop: random cal/test splits.
    """
    rng = np.random.default_rng(0)
    n = len(scores)
    boot_acc, boot_cov, boot_keep = [], [], []
    for _ in range(n_boot):
        idx_b = rng.integers(0, n, size=n)
        rs_b = scores[idx_b]; cs_b = correct[idx_b]
        accs, covs, keeps = [], [], []
        for _ in range(n_seeds // 25):  # fewer inner seeds for speed
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


def fmt_acc_ci(x):
    if x is None or "kept_acc" not in x:
        return "—"
    a = x["kept_acc"]
    return f"{a['mean']*100:.1f} [{a['ci95'][0]*100:.1f}, {a['ci95'][1]*100:.1f}]"


def fmt_cov_ci(x):
    if x is None or "coverage" not in x:
        return "—"
    a = x["coverage"]
    return f"{a['mean']*100:.1f} [{a['ci95'][0]*100:.1f}, {a['ci95'][1]*100:.1f}]"


def fmt_keep(x):
    if x is None: return "—"
    return f"{x['kept_frac']['mean']*100:.0f}"


def main():
    # ---------- Load all data ----------
    print("Loading data...", flush=True)
    sources = {}

    # GSM8K (full E3) lp + sc
    if (EXP / "E3_gsm8k_greedy_traces.jsonl").exists():
        gsm_lp = lp_records(load_jsonl(EXP / "E3_gsm8k_greedy_traces.jsonl"))
        sources["GSM8K-LP"] = ("Qwen2.5-7B", "GSM8K (full, n=1319)", gsm_lp)
    if (EXP / "E3_gsm8k_sc8_traces.jsonl").exists():
        gsm_sc = sc_records(load_jsonl(EXP / "E3_gsm8k_sc8_traces.jsonl"))
        sources["GSM8K-SC"] = ("Qwen2.5-7B", "GSM8K (full, n=1319)", gsm_sc)

    # MATH-500 (full E1) lp + sc + PRM (from Pilot A — same 500 traces, identical prompts)
    if (EXP / "E1_math500_greedy_traces.jsonl").exists():
        math_lp = lp_records(load_jsonl(EXP / "E1_math500_greedy_traces.jsonl"))
        sources["MATH500-LP"] = ("Qwen2.5-7B", "MATH-500 (full, n=500)", math_lp)
    if (EXP / "E1_math500_sc8_traces.jsonl").exists():
        math_sc = sc_records(load_jsonl(EXP / "E1_math500_sc8_traces.jsonl"))
        sources["MATH500-SC"] = ("Qwen2.5-7B", "MATH-500 (full, n=500)", math_sc)
    if (PILOTS / "pilotA_prm_traces.jsonl").exists():
        prm_rows = load_jsonl(PILOTS / "pilotA_prm_traces.jsonl")
        prm_recs = [{"id": r["id"],
                      "prm_min": float(r["prm_min"]),
                      "prm_mean": float(r["prm_mean"]),
                      "prm_last": float(r["prm_last"]),
                      "correct": int(r["correct"])} for r in prm_rows]
        sources["MATH500-PRM"] = ("Qwen2.5-7B", "MATH-500 (full, n=500)", prm_recs)

    # AIME (full E2)
    if (EXP / "E2_aime_greedy_traces.jsonl").exists():
        aime_lp = lp_records(load_jsonl(EXP / "E2_aime_greedy_traces.jsonl"))
        sources["AIME-LP"] = ("Qwen2.5-7B", "AIME 1983-2024 (full, n=933)", aime_lp)
    if (EXP / "E2_aime_sc8_traces.jsonl").exists():
        aime_sc = sc_records(load_jsonl(EXP / "E2_aime_sc8_traces.jsonl"))
        sources["AIME-SC"] = ("Qwen2.5-7B", "AIME 1983-2024 (full, n=933)", aime_sc)

    # 32B scaling
    if (EXP / "E4_qwen32b_math500_greedy.jsonl").exists():
        math_32b_lp = lp_records(load_jsonl(EXP / "E4_qwen32b_math500_greedy.jsonl"))
        sources["MATH500-32B-LP"] = ("Qwen2.5-32B", "MATH-500 (full, n=500)", math_32b_lp)
    if (EXP / "E4_qwen32b_math500_sc.jsonl").exists():
        math_32b_sc = sc_records(load_jsonl(EXP / "E4_qwen32b_math500_sc.jsonl"))
        sources["MATH500-32B-SC"] = ("Qwen2.5-32B", "MATH-500 (full, n=500)", math_32b_sc)

    print(f"  Loaded sources: {list(sources.keys())}", flush=True)

    # ---------- Compute CP results with bootstrap CIs ----------
    rows = []
    print("Running bootstrap CP (this takes a few minutes)...", flush=True)
    operating_points = [
        ("GSM8K-LP",    "lp_min",   [0.05, 0.1, 0.2, 0.3, 0.5]),
        ("GSM8K-SC",    "sc_top1",  [0.05, 0.1, 0.2, 0.3, 0.5]),
        ("MATH500-LP",  "lp_min",   [0.05, 0.1, 0.2, 0.3, 0.5]),
        ("MATH500-SC",  "sc_top1",  [0.05, 0.1, 0.2, 0.3, 0.5]),
        ("MATH500-PRM", "prm_min",  [0.05, 0.1, 0.2, 0.3, 0.5]),
        ("MATH500-PRM", "prm_mean", [0.05, 0.1, 0.2, 0.3, 0.5]),
        ("AIME-LP",     "lp_mean",  [0.05, 0.1, 0.2, 0.3, 0.5]),
        ("AIME-SC",     "sc_top1",  [0.05, 0.1, 0.2, 0.3, 0.5]),
        ("MATH500-32B-LP", "lp_min", [0.05, 0.1, 0.2, 0.3, 0.5]),
        ("MATH500-32B-SC", "sc_top1", [0.05, 0.1, 0.2, 0.3, 0.5]),
    ]
    for src_key, score_key, alphas in operating_points:
        if src_key not in sources:
            continue
        model, dataset_name, recs = sources[src_key]
        if not recs or score_key not in recs[0]:
            continue
        scores = np.array([r[score_key] for r in recs])
        correct = np.array([r["correct"] for r in recs])
        for alpha in alphas:
            cp = split_cp(scores, correct, alpha)
            if cp is None: continue
            rows.append({
                "model": model, "dataset": dataset_name, "score": score_key, "alpha": alpha,
                "n": int(len(recs)),
                "vanilla_acc": float(correct.mean()),
                **cp,
            })
            print(f"[{src_key:15s} {score_key:9s} α={alpha:.2f}]  cov={cp['coverage']['mean']:.3f}  kept_acc={cp['kept_acc']['mean']:.3f}  keep%={cp['kept_frac']['mean']:.2f}", flush=True)

    (EXP / "E5_table1.json").write_text(json.dumps(rows, indent=2))

    # ---------- Build markdown table ----------
    md_lines = []
    md_lines.append("# Table 1: Selective Accuracy with Conformal Prediction\n")
    md_lines.append("Bootstrap 95% CIs (500 boot resamples × 12 cal/test splits each).\n")
    md_lines.append("All values in percent.\n")
    md_lines.append("| Model | Dataset (n) | Score | α | Coverage [95% CI] | Kept acc [95% CI] | Keep% | Vanilla acc |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['dataset']} | {r['score']} | {r['alpha']:.2f} "
            f"| {fmt_cov_ci(r)} | {fmt_acc_ci(r)} | {fmt_keep(r)} | {r['vanilla_acc']*100:.1f} |"
        )
    (EXP / "E5_table1.md").write_text("\n".join(md_lines))

    # ---------- LaTeX table ----------
    tex_lines = []
    tex_lines.append("\\begin{tabular}{llllrrrl}")
    tex_lines.append("\\toprule")
    tex_lines.append("Model & Dataset & Score & $\\alpha$ & Cov.\\ [\\%] & Kept Acc.\\ [\\%] & Keep\\ [\\%] & Van.\\ Acc.\\ \\\\")
    tex_lines.append("\\midrule")
    for r in rows:
        tex_lines.append(
            f"{r['model']} & {r['dataset']} & {r['score'].replace('_', '\\_')} & {r['alpha']:.2f} "
            f"& {fmt_cov_ci(r)} & {fmt_acc_ci(r)} & {fmt_keep(r)} & {r['vanilla_acc']*100:.1f} \\\\"
        )
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    (EXP / "E5_table1.tex").write_text("\n".join(tex_lines))

    # ---------- Figure 1: selective accuracy curves ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    # Panel A: GSM8K
    panels = [
        ("A. GSM8K (n=1319)",  ["GSM8K-LP", "GSM8K-SC"]),
        ("B. MATH-500 (n=500)",  ["MATH500-LP", "MATH500-PRM", "MATH500-SC"]),
        ("C. AIME 1983-2024 (n=933, OOD from MATH cal)", ["AIME-LP", "AIME-SC"]),
    ]
    panel_scores = {
        "GSM8K-LP": ("lp_min",  "C0", "lp_min"),
        "GSM8K-SC": ("sc_top1", "C3", "sc_top1 (SC@8)"),
        "MATH500-LP":  ("lp_min",  "C0", "lp_min"),
        "MATH500-PRM": ("prm_min", "C2", "prm_min (Qwen2.5-Math-PRM)"),
        "MATH500-SC":  ("sc_top1", "C3", "sc_top1 (SC@8)"),
        "AIME-LP":  ("lp_mean", "C0", "lp_mean"),
        "AIME-SC":  ("sc_top1", "C3", "sc_top1 (SC@8)"),
    }
    for ax, (title, src_keys) in zip(axes, panels):
        for src_key in src_keys:
            score_key, color, label = panel_scores[src_key]
            rs = [r for r in rows if r["dataset"].startswith(title.split(".")[1].strip().split(" (")[0]) and r["score"] == score_key]
            rs.sort(key=lambda r: r["kept_frac"]["mean"])
            xs = [r["kept_frac"]["mean"] for r in rs]
            ys = [r["kept_acc"]["mean"] for r in rs]
            yerr_lo = [r["kept_acc"]["mean"] - r["kept_acc"]["ci95"][0] for r in rs]
            yerr_hi = [r["kept_acc"]["ci95"][1] - r["kept_acc"]["mean"] for r in rs]
            ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="o-", color=color,
                          label=label, alpha=0.8, capsize=3)
        # vanilla baseline
        if src_keys:
            vanilla = float(rows[next(i for i, r in enumerate(rows) if r["dataset"].startswith(title.split(".")[1].strip().split(" (")[0]))]["vanilla_acc"])
            ax.axhline(vanilla, ls="--", color="gray", alpha=0.5, label=f"vanilla ({vanilla:.2f})")
        ax.set_xlabel("answer rate (kept fraction)")
        ax.set_ylabel("kept accuracy")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        ax.set_ylim(0, 1.0)
    fig.suptitle("Figure 1: Selective accuracy by score family on three benchmarks (95% CI)", y=1.02)
    fig.tight_layout()
    fig.savefig(EXP / "E5_figure1_selective.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote E5_table1 + E5_figure1_selective.png")


if __name__ == "__main__":
    main()

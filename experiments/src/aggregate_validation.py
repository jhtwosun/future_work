"""Aggregate SX_validation_*.json into a unified cross-model cross-dataset table."""

import json
from pathlib import Path

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / "SX_validation_summary.md"

def main():
    # Find all validation JSONs
    base = ["qwen25_7b", "qwen25_math_7b", "qwen25_32b", "phi4"]
    aime_redo = ["qwen25_7b_aime_redo", "qwen25_math_7b_aime_redo"]

    runs = {}
    for tag in base:
        f = OUTDIR / f"SX_validation_{tag}.json"
        if f.exists():
            runs[tag] = json.load(open(f))
    # Merge in AIME redo if present (overwrite the small AIME with the larger one)
    for tag in aime_redo:
        f = OUTDIR / f"SX_validation_{tag}.json"
        if not f.exists(): continue
        d = json.load(open(f))
        base_tag = tag.replace("_aime_redo", "")
        if base_tag in runs and "aime" in d.get("datasets", {}):
            runs[base_tag]["datasets"]["aime"] = d["datasets"]["aime"]

    score_keys = [
        "lp_mean_min", "top1_margin_mean", "top1_margin_min",
        "entropy_max", "entropy_mean",
        "tempered_kl_max", "tempered_kl_mean",
        "kl_uniform_min", "kl_uniform_mean",
    ]
    datasets = ["math500", "aime", "olympiad", "mmlu_pro"]  # skip humaneval (placeholder)

    lines = []
    lines.append("# Validation matrix — info-theoretic scores cross-model cross-dataset\n")
    lines.append("kept_acc at α=0.30 (bootstrap mean over 300×5 splits, 95% CI in [low, high]).\n")
    lines.append("Vanilla = greedy accuracy. n is per-dataset sample size.\n\n")

    # Per-dataset summary table: model × score
    for ds in datasets:
        lines.append(f"## Dataset: {ds}\n")
        # Header
        header = ["model", "n", "vanilla"] + score_keys
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for tag, run in runs.items():
            r = run["datasets"].get(ds, {})
            if "vanilla_acc" not in r: continue
            row = [tag, str(r["n"]), f"{r['vanilla_acc']:.3f}"]
            for sk in score_keys:
                if sk in r and r[sk].get("kept_acc_alpha_0.30"):
                    cp = r[sk]["kept_acc_alpha_0.30"]
                    rho = r[sk]["rho"]
                    row.append(f"{cp['kept_acc']:.3f}")
                else:
                    row.append("--")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Cross-dataset summary: best score per (model, dataset)
    lines.append("## Best score per (model, dataset)\n")
    lines.append("| model | dataset | n | vanilla | best score | best kept@0.3 | lp_min kept@0.3 | Δ vs lp_min |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for tag, run in runs.items():
        for ds in datasets:
            r = run["datasets"].get(ds, {})
            if "vanilla_acc" not in r: continue
            results_with_kept = []
            for sk in score_keys:
                if sk in r and r[sk].get("kept_acc_alpha_0.30"):
                    results_with_kept.append((sk, r[sk]["kept_acc_alpha_0.30"]["kept_acc"]))
            if not results_with_kept: continue
            best_sk, best_acc = max(results_with_kept, key=lambda x: x[1])
            lp_acc = r["lp_mean_min"]["kept_acc_alpha_0.30"]["kept_acc"] if (
                "lp_mean_min" in r and r["lp_mean_min"].get("kept_acc_alpha_0.30")
            ) else None
            lp_str = f"{lp_acc:.3f}" if lp_acc is not None else "N/A"
            delta_str = f"{best_acc - lp_acc:+.3f}" if lp_acc is not None else "N/A"
            lines.append(f"| {tag} | {ds} | {r['n']} | {r['vanilla_acc']:.3f} | {best_sk} | {best_acc:.3f} | {lp_str} | {delta_str} |")

    # Aggregate winner across model/dataset
    lines.append("\n## Score win-rate (counts of being top-1 / top-3)\n")
    winners = {sk: {"top1": 0, "top3": 0} for sk in score_keys}
    total_cells = 0
    for tag, run in runs.items():
        for ds in datasets:
            r = run["datasets"].get(ds, {})
            if "vanilla_acc" not in r: continue
            results_with_kept = []
            for sk in score_keys:
                if sk in r and r[sk].get("kept_acc_alpha_0.30"):
                    results_with_kept.append((sk, r[sk]["kept_acc_alpha_0.30"]["kept_acc"]))
            if len(results_with_kept) < 3: continue
            ranked = sorted(results_with_kept, key=lambda x: -x[1])
            winners[ranked[0][0]]["top1"] += 1
            for sk, _ in ranked[:3]:
                winners[sk]["top3"] += 1
            total_cells += 1

    lines.append(f"\nTotal (model, dataset) cells: {total_cells}\n")
    lines.append("| Score | Top-1 wins | Top-3 wins |")
    lines.append("|---|---|---|")
    for sk in score_keys:
        lines.append(f"| {sk} | {winners[sk]['top1']} | {winners[sk]['top3']} |")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

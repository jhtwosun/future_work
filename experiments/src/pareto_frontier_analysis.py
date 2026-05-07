"""Identify Pareto-optimal points on cost-accuracy frontier."""

import json
from pathlib import Path
import numpy as np

OUTDIR = Path('/home/nvidia/future/experiments/results')


def main():
    # Aggregate Pareto results
    all_data = {}
    for tag in ['qwen25_7b', 'qwen25_math_7b', 'qwen25_32b', 'phi4']:
        for ds in ['math500', 'aime', 'olympiad']:
            f = OUTDIR / f'SX_pareto_{tag}_{ds}.json'
            if f.exists():
                all_data[(tag, ds)] = json.load(open(f))

    # Per (model, ds), compute Pareto frontier
    print("=== Pareto-optimal (kept after dominance filter) per (model, ds) ===\n")
    for (tag, ds), d in sorted(all_data.items()):
        # Collect (method, cost, acc) tuples
        points = []
        for m, v in d['results'].items():
            points.append((m, v.get('avg_compute', 1), v['acc']))
        # vanilla as point at cost=1, acc=vanilla_acc
        points.append(('vanilla', 1.0, d['vanilla_acc']))
        # Sort by cost, then by acc desc
        points.sort(key=lambda x: (x[1], -x[2]))
        # Pareto frontier: keep point if no later point has both lower cost AND higher acc
        # Standard: a point is Pareto-dominated if exists another with cost <= and acc >=
        pareto = []
        max_acc_so_far = -1
        for m, c, a in points:
            if a > max_acc_so_far:
                pareto.append((m, c, a))
                max_acc_so_far = a
        print(f"\n{tag} {ds}:")
        print(f"{'method':35s} {'cost':>6s} {'acc':>7s} {'lift':>7s}")
        for m, c, a in pareto:
            lift = a - d['vanilla_acc']
            print(f"  {m:35s} {c:6.2f}× {a:7.3f} {lift:+7.3f}")

    # Aggregate Pareto: average over cells
    print("\n\n=== Aggregate Pareto (avg over 12 cells) ===")
    methods = set()
    for d in all_data.values():
        methods.update(d['results'].keys())

    avg_data = []
    for m in methods:
        deltas = []; costs = []
        for d in all_data.values():
            if m in d['results']:
                deltas.append(d['results'][m]['delta'])
                costs.append(d['results'][m].get('avg_compute', 1))
        if deltas:
            avg_data.append((m, np.mean(costs), np.mean(deltas)))
    avg_data.append(('vanilla', 1.0, 0.0))
    avg_data.sort(key=lambda x: (x[1], -x[2]))

    pareto = []
    max_d = -100
    for m, c, d_val in avg_data:
        if d_val > max_d:
            pareto.append((m, c, d_val))
            max_d = d_val
    print(f"{'method':35s} {'avg_cost':>8s} {'avg_Δ':>8s}")
    for m, c, d_val in pareto:
        print(f"  {m:35s} {c:8.2f}× {d_val:+.4f}")

    # Save Pareto data for plotting
    pareto_data = {
        "per_cell": {f"{tag}_{ds}": [
            {"method": m, "cost": c, "acc": a, "delta": a - all_data[(tag, ds)]['vanilla_acc']}
            for m, v in all_data[(tag, ds)]['results'].items()
            for c, a in [(v.get('avg_compute', 1), v['acc'])]
        ] + [{"method": "vanilla", "cost": 1.0, "acc": all_data[(tag, ds)]['vanilla_acc'], "delta": 0.0}]
            for (tag, ds) in all_data
        },
        "aggregate": [{"method": m, "cost": c, "delta": d_val} for m, c, d_val in avg_data],
        "aggregate_pareto": [{"method": m, "cost": c, "delta": d_val} for m, c, d_val in pareto],
    }
    out = OUTDIR / "PARETO_FRONTIER.json"
    out.write_text(json.dumps(pareto_data, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

"""Aggregate ALL step rejection results (Tier 1, T2, T3, T4) into a unified table."""

import json
from pathlib import Path

OUTDIR = Path('/home/nvidia/future/experiments/results')

TAGS = ['qwen25_7b', 'qwen25_math_7b', 'qwen25_32b', 'phi4']
DSETS = ['math500', 'aime', 'olympiad']


def load_results(prefix):
    out = {}
    for tag in TAGS:
        for ds in DSETS:
            f = OUTDIR / f'{prefix}{tag}_{ds}.json'
            if f.exists():
                out[(tag, ds)] = json.load(open(f))
    return out


def main():
    t1 = load_results('SX_step_rej_')
    t2 = load_results('SX_step_rej_t2_')
    t3 = load_results('SX_step_rej_t3_')
    t4 = load_results('SX_step_rej_t4_')

    cells = sorted(set(list(t1.keys()) + list(t2.keys()) + list(t3.keys()) + list(t4.keys())))

    print(f"Total cells with at least one result: {len(cells)}")
    print(f"  T1: {len(t1)}, T2: {len(t2)}, T3: {len(t3)}, T4: {len(t4)}")

    # Build full method × cell matrix
    print("\n=== Best method per cell across all tiers ===")
    print(f"{'tag':20s} {'ds':10s} {'vanilla':>8s} {'best_method':40s} {'acc':>7s} {'Δ':>6s}")
    print('-' * 100)
    method_wins = {}
    for tag, ds in cells:
        all_methods = []
        vanilla = 0
        for tier_data in [t1.get((tag, ds), {}), t2.get((tag, ds), {}),
                          t3.get((tag, ds), {}), t4.get((tag, ds), {})]:
            if not tier_data: continue
            vanilla = tier_data.get('vanilla_acc', vanilla)
            for method, res in tier_data.get('results', {}).items():
                all_methods.append((method, res['acc'], res['delta']))
        if not all_methods: continue
        best = max(all_methods, key=lambda x: x[1])
        method_wins[best[0]] = method_wins.get(best[0], 0) + 1
        print(f"{tag:20s} {ds:10s} {vanilla:8.3f} {best[0][:40]:40s} {best[1]:7.3f} {best[2]:+6.3f}")

    print("\n=== Method Top-1 win count ===")
    for m, c in sorted(method_wins.items(), key=lambda x: -x[1]):
        print(f"  {m:40s} {c}")

    # Detailed per-cell table
    print("\n=== Detailed per-cell results (top 5 methods per cell) ===")
    for tag, ds in cells:
        all_methods = []
        vanilla = 0
        for tier_name, tier_data in [('T1', t1.get((tag, ds), {})), ('T2', t2.get((tag, ds), {})),
                                       ('T3', t3.get((tag, ds), {})), ('T4', t4.get((tag, ds), {}))]:
            if not tier_data: continue
            vanilla = tier_data.get('vanilla_acc', vanilla)
            for method, res in tier_data.get('results', {}).items():
                all_methods.append((tier_name, method, res['acc'], res['delta']))
        if not all_methods: continue
        all_methods.sort(key=lambda x: -x[2])
        print(f"\n--- {tag} {ds} (vanilla={vanilla:.3f}) ---")
        for tier_name, method, acc, delta in all_methods[:5]:
            print(f"  [{tier_name}] {method:40s} acc={acc:.3f}  Δ={delta:+.3f}")


if __name__ == "__main__":
    main()

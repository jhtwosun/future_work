"""Aggregate step rejection results (Tier 1, Tier 2, Tier 3) into a unified table."""

import json
from pathlib import Path

OUTDIR = Path('/home/nvidia/future/experiments/results')
OUT = OUTDIR / 'STEP_REJECTION_SUMMARY.md'

TAGS = ['qwen25_7b', 'qwen25_math_7b', 'qwen25_32b', 'phi4']
DSETS = ['math500', 'aime', 'olympiad']


def load_results(prefix):
    """Load all matching files."""
    out = {}
    for tag in TAGS:
        for ds in DSETS:
            f = OUTDIR / f'{prefix}{tag}_{ds}.json'
            if f.exists():
                out[(tag, ds)] = json.load(open(f))
    return out


def main():
    t1 = load_results('SX_step_rej_')  # Tier 1 + T2 t3 different name
    t2 = load_results('SX_step_rej_t2_')
    t3 = load_results('SX_step_rej_t3_')

    print("=== Tier 1 (4 methods) ===")
    for (tag, ds), data in sorted(t1.items()):
        print(f"\n{tag} {ds} (vanilla={data['vanilla_acc']:.3f}):")
        for method, res in data['results'].items():
            print(f"  {method:30s} acc={res['acc']:.3f}  Δ={res['delta']:+.3f}")

    print("\n=== Tier 2 (6+ methods) ===")
    for (tag, ds), data in sorted(t2.items()):
        print(f"\n{tag} {ds} (vanilla={data['vanilla_acc']:.3f}):")
        for method, res in data['results'].items():
            print(f"  {method:30s} acc={res['acc']:.3f}  Δ={res['delta']:+.3f}")

    print("\n=== Tier 3 (online multi-step) ===")
    for (tag, ds), data in sorted(t3.items()):
        print(f"\n{tag} {ds} (vanilla={data['vanilla_acc']:.3f}):")
        for method, res in data['results'].items():
            print(f"  {method:30s} acc={res['acc']:.3f}  Δ={res['delta']:+.3f}")

    # Unified table: best method per (tag, ds) across all tiers
    print("\n\n=== Best method per cell (all tiers) ===")
    print(f"{'tag':20s} {'ds':10s} {'vanilla':>8s} {'best_method':35s} {'kept_acc':>8s} {'Δ':>6s}")
    print('-' * 100)
    cells = sorted(set(list(t1.keys()) + list(t2.keys()) + list(t3.keys())))
    for tag, ds in cells:
        best = (None, -1, 0)
        for tier_data in [t1.get((tag, ds), {}), t2.get((tag, ds), {}), t3.get((tag, ds), {})]:
            if not tier_data: continue
            for method, res in tier_data.get('results', {}).items():
                if res['acc'] > best[1]:
                    best = (method, res['acc'], res['delta'])
        vanilla = (t1.get((tag, ds)) or t2.get((tag, ds)) or t3.get((tag, ds), {})).get('vanilla_acc', 0)
        print(f"{tag:20s} {ds:10s} {vanilla:8.3f} {best[0] or '?':35s} {best[1]:8.3f} {best[2]:+6.3f}")

    # Win-rate per method
    print("\n\n=== Method Top-1 win rate ===")
    method_wins = {}
    for tag, ds in cells:
        all_methods = []
        for tier_data in [t1.get((tag, ds), {}), t2.get((tag, ds), {}), t3.get((tag, ds), {})]:
            if not tier_data: continue
            for method, res in tier_data.get('results', {}).items():
                all_methods.append((method, res['acc']))
        if not all_methods: continue
        winner = max(all_methods, key=lambda x: x[1])
        method_wins[winner[0]] = method_wins.get(winner[0], 0) + 1
    for m, c in sorted(method_wins.items(), key=lambda x: -x[1]):
        print(f"  {m:35s} {c}")


if __name__ == "__main__":
    main()

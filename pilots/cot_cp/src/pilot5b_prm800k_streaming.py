"""
Pilot 5b: PRM800K rating distribution via streaming (avoid full 1M-row download).

Schema observed in Birchlabs/openai-prm800k-stepwise-critic:
  instruction, responses[list], next_response, answer, is_human_response,
  is_solution, is_preferred_response, rating

`rating` is the per-step quality label. We stream up to N records and
report the rating distribution + token statistics.
"""

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

OUT = Path("/home/nvidia/future/pilots/cot_cp/results/pilot5b_prm800k_streaming.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
N = 100_000  # cap


def main() -> None:
    ds = load_dataset(
        "Birchlabs/openai-prm800k-stepwise-critic",
        split="train",
        streaming=True,
    )
    rating_counter: Counter = Counter()
    bool_counters = {
        "is_solution": Counter(),
        "is_human_response": Counter(),
        "is_preferred_response": Counter(),
    }
    n_seen = 0
    n_with_rating = 0
    samples_per_rating: dict[str, list] = {}
    response_lens = []
    for r in ds:
        n_seen += 1
        rating = r.get("rating")
        rating_counter[str(rating)] += 1
        if rating is not None:
            n_with_rating += 1
        for k in bool_counters:
            bool_counters[k][str(r.get(k))] += 1
        nr = r.get("next_response")
        if isinstance(nr, str):
            response_lens.append(len(nr))
        # collect 2 samples per rating
        rk = str(rating)
        samples_per_rating.setdefault(rk, [])
        if len(samples_per_rating[rk]) < 2:
            samples_per_rating[rk].append({
                "instruction": r.get("instruction", "")[:200],
                "context_steps": len(r.get("responses") or []),
                "next_response": nr[:300] if isinstance(nr, str) else nr,
                "rating": rating,
                "is_solution": r.get("is_solution"),
                "is_preferred_response": r.get("is_preferred_response"),
            })
        if n_seen >= N:
            break

    import numpy as np
    summary = {
        "n_records_streamed": n_seen,
        "n_with_rating": n_with_rating,
        "rating_counter": dict(rating_counter.most_common()),
        "is_solution_counter": dict(bool_counters["is_solution"].most_common()),
        "is_human_response_counter": dict(bool_counters["is_human_response"].most_common()),
        "is_preferred_response_counter": dict(bool_counters["is_preferred_response"].most_common()),
        "next_response_length_stats": {
            "n": len(response_lens),
            "mean_chars": float(np.mean(response_lens)) if response_lens else 0.0,
            "median_chars": float(np.median(response_lens)) if response_lens else 0.0,
            "p90_chars": float(np.percentile(response_lens, 90)) if response_lens else 0.0,
        },
        "samples_per_rating": samples_per_rating,
    }
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

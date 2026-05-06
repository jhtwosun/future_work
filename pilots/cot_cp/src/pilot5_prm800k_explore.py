"""
Pilot 5: PRM800K dataset exploration.

The user's research_plan calls PRM800K the calibration source for CoT-CP.
This script:
  - tries the canonical mirror first (Birchlabs/openai-prm800k-stepwise-critic)
  - falls back to direct GitHub download of phase2 jsonl
  - reports schema, label distribution, step length, and saves a small sample

Output:
  results/pilot5_prm800k_summary.json
  results/pilot5_prm800k_sample.jsonl
"""

import json
import os
import sys
from pathlib import Path

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")
RESULTS.mkdir(parents=True, exist_ok=True)
SAMPLE_OUT = RESULTS / "pilot5_prm800k_sample.jsonl"
SUMMARY_OUT = RESULTS / "pilot5_prm800k_summary.json"


def try_huggingface() -> dict | None:
    try:
        from datasets import load_dataset
    except Exception as e:
        return {"error": f"datasets import failed: {e}"}
    candidates = [
        ("Birchlabs/openai-prm800k-stepwise-critic", None, "train"),
        ("Birchlabs/openai-prm800k-stepwise-critic", None, "test"),
        ("HuggingFaceH4/prm800k", None, "train"),
        ("openai/prm800k", None, "train"),
    ]
    for repo, cfg, split in candidates:
        try:
            print(f"[hf] try {repo} cfg={cfg} split={split}", flush=True)
            ds = load_dataset(repo, cfg, split=split, streaming=False)
            print(f"  ok: {len(ds)} rows; columns={ds.column_names}", flush=True)
            return {
                "source": f"hf:{repo}:{split}",
                "n_rows": len(ds),
                "columns": list(ds.column_names),
                "first": ds[0],
                "_dataset_handle_split": split,
                "_dataset_handle_repo": repo,
            }
        except Exception as e:
            print(f"  fail: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return None


def explore_hf(meta: dict) -> dict:
    from datasets import load_dataset
    ds = load_dataset(meta["_dataset_handle_repo"], split=meta["_dataset_handle_split"])
    cols = ds.column_names
    sample_n = min(5, len(ds))
    samples = []
    for i in range(sample_n):
        samples.append(ds[i])
    SAMPLE_OUT.write_text("\n".join(json.dumps(s, default=str) for s in samples))

    # Try to summarize step labels if a recognizable column exists
    label_summary = {}
    for col in cols:
        if "rating" in col.lower() or "label" in col.lower():
            try:
                vals = ds[col]
                # flatten if list of lists
                flat = []
                for v in vals:
                    if isinstance(v, list):
                        flat.extend(v)
                    elif v is not None:
                        flat.append(v)
                from collections import Counter
                c = Counter(flat)
                label_summary[col] = dict(c.most_common(20))
            except Exception:
                pass
    return {
        "n_rows": len(ds),
        "columns": cols,
        "label_columns_summary": label_summary,
    }


def try_github_download() -> dict | None:
    """Download official PRM800K phase 2 jsonl from openai/prm800k repo."""
    import urllib.request
    base = "https://raw.githubusercontent.com/openai/prm800k/main/prm800k/data"
    files = ["phase2_train.jsonl", "phase2_test.jsonl"]
    out = RESULTS / "prm800k_raw"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for f in files:
        url = f"{base}/{f}"
        target = out / f
        if target.exists() and target.stat().st_size > 1000:
            print(f"[gh] cached: {f} ({target.stat().st_size} B)", flush=True)
        else:
            print(f"[gh] downloading {url}", flush=True)
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    target.write_bytes(r.read())
                print(f"  ok: {target.stat().st_size} B", flush=True)
            except Exception as e:
                print(f"  fail: {e}", flush=True)
                continue
        # peek first line
        with target.open() as fh:
            first = json.loads(fh.readline())
            n_lines = sum(1 for _ in fh) + 1
        summary[f] = {"path": str(target), "n_lines": n_lines, "first_keys": list(first.keys())}
    return summary or None


def summarize_prm800k_jsonl(path: str) -> dict:
    """Walk a phase2 jsonl and aggregate label statistics."""
    from collections import Counter
    label_counts = Counter()
    n_records = 0
    n_steps_per_problem = []
    sample_rec = None
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            n_records += 1
            if sample_rec is None:
                sample_rec = r
            # PRM800K phase2 schema: r["label"]["steps"] is list of step objects
            label = r.get("label", {})
            steps = label.get("steps", []) if isinstance(label, dict) else []
            n_steps_per_problem.append(len(steps))
            for s in steps:
                if not isinstance(s, dict):
                    continue
                comps = s.get("completions") or []
                # each completion has "rating" int or None
                for c in comps:
                    if isinstance(c, dict):
                        r_ = c.get("rating")
                        label_counts[str(r_)] += 1
                # human-chosen completion
                chosen = s.get("chosen_completion")
                if chosen is not None:
                    label_counts[f"chosen_idx_present"] += 1
            if n_records >= 20000:  # cap for speed
                break
    import numpy as np
    return {
        "n_records_scanned": n_records,
        "rating_counter": dict(label_counts.most_common()),
        "steps_per_problem": {
            "mean": float(np.mean(n_steps_per_problem)) if n_steps_per_problem else 0,
            "median": float(np.median(n_steps_per_problem)) if n_steps_per_problem else 0,
            "max": int(np.max(n_steps_per_problem)) if n_steps_per_problem else 0,
        },
        "first_record_keys": list(sample_rec.keys()) if sample_rec else [],
    }


def main() -> None:
    summary = {"sources_tried": []}

    hf_meta = try_huggingface()
    if hf_meta and "error" not in hf_meta:
        try:
            details = explore_hf(hf_meta)
            summary["huggingface"] = {**hf_meta, **details}
            summary["sources_tried"].append("huggingface_ok")
        except Exception as e:
            summary["huggingface_explore_error"] = str(e)

    gh = try_github_download()
    if gh:
        summary["github"] = gh
        summary["sources_tried"].append("github_ok")
        # summarize phase2_train if available
        train = gh.get("phase2_train.jsonl")
        if train:
            try:
                summary["github_train_summary"] = summarize_prm800k_jsonl(train["path"])
            except Exception as e:
                summary["github_train_summary_error"] = str(e)

    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()

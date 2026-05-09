"""
PEARL FULL EXPERIMENT — earliest-bad-step (t*) vs worst-step (t_worst) K=4
majority intervention.

This is the H1 empirical test for the Pearl Causal Step Intervention paper.

For each (model, dataset, score_family, alpha) cell:
  1. Re-decode greedy with logprobs to recover (token_ids, boundaries, step_lps, step_ents, step_margs).
  2. Compute t_worst and t* under per-step split-CP calibration.
  3. Generate K=4 alternatives at every (trace, step_loc) where step_loc ∈ {t_worst, t*}
     for *any* (score_family, alpha). De-dup the (trace, step_loc) pairs to avoid
     wasted forward passes — many alphas/scores produce the same step.
  4. For every (score_family, alpha) compute kept_acc with K=4 majority at t* vs t_worst.
  5. Cascade-depth stratified breakdown (gap = t_worst - t* in {1, 2-4, ≥5}).
  6. Bootstrap CIs (500 resamples × 10 cal/test splits) on lift_earliest_vs_worst.

Outputs:
  /home/nvidia/future/experiments/results/pearl_full/{tag}_{dataset}.json (per cell, all scores/alphas)
  /home/nvidia/future/experiments/results/pearl_full/AGGREGATE.json
  /home/nvidia/future/experiments/results/pearl_full/AGGREGATE.md

Env vars (optional):
  MODEL, TAG, DATASET, NQ — to run a single cell. If unset, iterates over the full grid.
  CELL_ONLY — semicolon-separated list of "tag__dataset" to restrict (e.g. "qwen25_7b__math500;phi4__aime")
  ALPHAS — comma-separated alphas (default "0.1,0.3,0.5")
  SCORES — comma-separated score keys (default "lp,ent_neg,marg")
  N_BOOT — bootstrap iterations (default 500)
  N_SPLITS — calibration splits per bootstrap (default 10)
  SKIP_GENERATION — if "1", skip vllm and only re-aggregate from cached per-cell data.
"""

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

# vLLM imports are deferred so SKIP_GENERATION=1 paths don't require GPU.
sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, equal_strict, normalize  # noqa: E402

# ---------------- config ----------------

OUTDIR = Path("/home/nvidia/future/experiments/results/pearl_full")
OUTDIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUTDIR / "_cache"
CACHE_DIR.mkdir(exist_ok=True)

PROV_DIR = Path("/home/nvidia/future/experiments/results")

DEFAULT_GRID = [
    ("Qwen/Qwen2.5-7B-Instruct",      "qwen25_7b"),
    ("Qwen/Qwen2.5-Math-7B-Instruct", "qwen25_math_7b"),
    ("Qwen/Qwen2.5-32B-Instruct",     "qwen25_32b"),
    ("microsoft/phi-4",               "phi4"),
]
DEFAULT_DATASETS = ["math500", "aime", "olympiad"]
DEFAULT_ALPHAS   = [0.1, 0.3, 0.5]
DEFAULT_SCORES   = ["lp", "ent_neg", "marg"]

ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "0.1,0.3,0.5").split(",")]
SCORES = os.environ.get("SCORES", "lp,ent_neg,marg").split(",")
N_BOOT = int(os.environ.get("N_BOOT", "500"))
N_SPLITS = int(os.environ.get("N_SPLITS", "10"))
NQ = int(os.environ.get("NQ", "200"))
SEED = 0
TOPK = 20
SKIP_GENERATION = os.environ.get("SKIP_GENERATION", "0") == "1"
MAX_MODEL_LEN = 2560
MAX_TOKENS = 1024

PROMPT_MATH = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\nProblem: {question}\n\n"
)

# ---------------- dataset loaders ----------------

def load_dataset_uniform(name, n):
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["problem"], "gold": r["answer"]} for r in ds]
    if name == "aime":
        ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["Question"], "gold": str(r["Answer"])} for r in ds]
    if name == "olympiad":
        ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["question"], "gold": str(r["final_answer"][0]) if r["final_answer"] else ""} for r in ds]
    raise ValueError(name)

# ---------------- step helpers ----------------

def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries

def _lp_of(e):
    if hasattr(e, "logprob"):
        return e.logprob
    if isinstance(e, (tuple, list)) and len(e) > 0:
        return float(e[0])
    if isinstance(e, (int, float)):
        return float(e)
    return float("nan")

def step_lp_means(chosen_lps, boundaries, n):
    bdy = boundaries + [n]
    out = []
    for sa, sb in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in chosen_lps[sa:sb] if not (lp != lp)]
        out.append(float(np.mean(seg)) if seg else float("nan"))
    return out

def step_ent_means(distribs, boundaries, n):
    bdy = boundaries + [n]
    out = []
    for sa, sb in zip(bdy[:-1], bdy[1:]):
        ents = []
        for d in distribs[sa:sb]:
            if d is None:
                continue
            p = np.array(d["probs"])
            p = p[p > 0]
            if len(p) > 0:
                ents.append(-float((p * np.log(p)).sum()))
        out.append(float(np.mean(ents)) if ents else float("nan"))
    return out

def step_marg_means(distribs, boundaries, n):
    """Margin = top1_lp - top2_lp at each token, averaged over a step."""
    bdy = boundaries + [n]
    out = []
    for sa, sb in zip(bdy[:-1], bdy[1:]):
        margs = []
        for d in distribs[sa:sb]:
            if d is None:
                continue
            lps = d.get("lps")
            if lps is None or len(lps) < 2:
                continue
            margs.append(float(lps[0] - lps[1]))
        out.append(float(np.mean(margs)) if margs else float("nan"))
    return out

def majority_vote(preds_list):
    votes = [str(normalize(p)) for p in preds_list if p is not None]
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]

# ---------------- t_worst & t_star (split-CP) ----------------

def argmin_t(scores):
    arr = np.array(scores, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return int(np.nanargmin(arr))

def calibrate_per_step_thresholds(cal_correct_rows, score_key, alpha, max_T,
                                  min_cal_per_step=5):
    """For each step position t, lower-alpha quantile of correct calibration scores."""
    thr = {}
    for t in range(max_T):
        vals = []
        for r in cal_correct_rows:
            if t < len(r[score_key]):
                v = r[score_key][t]
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals.append(v)
        if len(vals) < min_cal_per_step:
            continue
        n_c = len(vals)
        ql = max(0.0, min(1.0, math.floor(alpha * (n_c + 1)) / n_c))
        thr[t] = float(np.quantile(vals, ql))
    return thr

def t_star_for(scores, thresholds):
    """First step t where scores[t] < thresholds[t]."""
    for t, s in enumerate(scores):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        if t in thresholds and s < thresholds[t]:
            return t
    return None

# ---------------- vLLM Phase: greedy + K=4 alts ----------------

def run_greedy_phase(llm, tok, base_prompts):
    """Run greedy with logprobs=20 and return per-trace token data + step scores."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=TOPK, seed=SEED)
    outs = llm.generate(base_prompts, sp)
    records = []
    for i, out in enumerate(outs):
        o = out.outputs[0]
        text = o.text
        tok_ids = list(o.token_ids)
        tok_strs = [tok.decode([tid]) for tid in tok_ids]
        chosen_lps = []
        distribs = []
        for step_lp in (o.logprobs or []):
            if step_lp is None:
                chosen_lps.append(float("nan"))
                distribs.append(None)
                continue
            try:
                items = list(step_lp.items())
                items.sort(key=lambda kv: -_lp_of(kv[1]))
                if items:
                    chosen_lps.append(_lp_of(items[0][1]))
                    entries_lps = np.array([_lp_of(it[1]) for it in items], dtype=float)
                    probs = np.exp(entries_lps - entries_lps.max())
                    probs = probs / probs.sum()
                    distribs.append({"probs": probs.tolist(), "lps": entries_lps.tolist()})
                else:
                    chosen_lps.append(float("nan"))
                    distribs.append(None)
            except Exception:
                chosen_lps.append(float("nan"))
                distribs.append(None)
        boundaries = find_step_boundaries(tok_strs)
        n = len(tok_ids)
        s_lp   = step_lp_means(chosen_lps, boundaries, n)
        s_ent  = step_ent_means(distribs,  boundaries, n)
        s_marg = step_marg_means(distribs, boundaries, n)
        records.append({
            "id": i,
            "tok_ids": tok_ids,
            "boundaries": boundaries,
            "text": text,
            "lp":      s_lp,
            "ent_neg": [-e if (e is not None and not (isinstance(e, float) and math.isnan(e))) else float("nan") for e in s_ent],
            "marg":    s_marg,
            "T": len(s_lp),
        })
    return records

def get_prefix_at_step(rec, step_idx, base_prompt, tok):
    if step_idx is None:
        return None
    if not rec["boundaries"]:
        return None
    if step_idx >= len(rec["boundaries"]):
        return None
    bdy_pos = rec["boundaries"][step_idx]
    prefix = tok.decode(rec["tok_ids"][:bdy_pos], skip_special_tokens=True)
    return base_prompt + prefix

def run_K4_alts_for_locs(llm, tok, locs, base_prompts, greedy_records):
    """locs: list of (trace_id, step_idx). Returns dict (trace_id, step_idx) -> [pred1..pred4]."""
    from vllm import SamplingParams
    prompts = []
    keys = []
    for (tid, sidx) in locs:
        rec = greedy_records[tid]
        p = get_prefix_at_step(rec, sidx, base_prompts[tid], tok)
        if p is None:
            continue
        prompts.append(p)
        keys.append((tid, sidx))
    if not prompts:
        return {}
    sp = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED + 100)
    outs = llm.generate(prompts, sp)
    out_map = {}
    for k, ob in zip(keys, outs):
        out_map[k] = [extract_pred(c.text) for c in ob.outputs]
    return out_map

# ---------------- bootstrap analysis ----------------

def bootstrap_lift(rows, score_key, alpha, alts_map, n_boot=500, n_splits=10, rng=None):
    """For a single (score, alpha), bootstrap over (cal/test) splits and resamples.

    Returns kept_accs and lift summary.
    rows: list of dicts with keys id, correct, T, lp, ent_neg, marg, pred (greedy), gold
    alts_map: dict (trace_id, step_idx) -> [4 preds]
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(rows)
    if n < 20:
        return None
    max_T = max(r["T"] for r in rows)

    # Per-trace, deterministic t_worst (calibration-independent):
    for r in rows:
        r["_t_worst"] = argmin_t(r[score_key])

    # Collect per-split kept_acc at t_worst, t*, and the cascade-stratified versions
    K4_worst_accs = []
    K4_earliest_accs = []
    vanilla_accs = []
    cascade_strata = {"gap_1": [], "gap_2_4": [], "gap_5p": [], "gap_negzero": []}
    frac_changed_worst = []
    frac_changed_earliest = []

    # outer bootstrap × splits
    for b in range(n_boot):
        # one cal/test split per iter
        perm = rng.permutation(n)
        nc = n // 2
        cal_idx, test_idx = perm[:nc], perm[nc:]
        cal_rows = [rows[i] for i in cal_idx]
        test_rows = [rows[i] for i in test_idx]
        cal_corr = [r for r in cal_rows if r["correct"] == 1]
        if len(cal_corr) < 5:
            continue
        thr = calibrate_per_step_thresholds(cal_corr, score_key, alpha, max_T)

        # also resample the test-set (bootstrap CI)
        boot_idx = rng.integers(0, len(test_rows), size=len(test_rows))
        boot_rows = [test_rows[i] for i in boot_idx]

        n_van = 0
        n_w = 0
        n_e = 0
        n_chg_w = 0
        n_chg_e = 0
        cascade_buckets = defaultdict(list)  # gap_bucket -> list of (kept_correct_e - kept_correct_w)

        for r in boot_rows:
            tw = r["_t_worst"]
            ts = t_star_for(r[score_key], thr)
            gold = r["gold"]
            greedy_pred = r["pred"]

            # vanilla
            van_correct = int(equal_strict(greedy_pred, gold))
            n_van += van_correct

            # K=4 majority at t_worst (vote of [greedy, alt1..alt4])
            alts_w = alts_map.get((r["id"], tw)) if tw is not None else None
            if alts_w:
                kept_w = majority_vote([greedy_pred] + alts_w)
                changed_w = int(str(normalize(kept_w)) != str(normalize(greedy_pred)))
            else:
                kept_w = greedy_pred
                changed_w = 0
            corr_w = int(equal_strict(kept_w, gold))
            n_w += corr_w
            n_chg_w += changed_w

            # K=4 majority at t*
            if ts is not None:
                alts_e = alts_map.get((r["id"], ts))
                if alts_e:
                    kept_e = majority_vote([greedy_pred] + alts_e)
                    changed_e = int(str(normalize(kept_e)) != str(normalize(greedy_pred)))
                else:
                    # No K=4 cached for this loc — skip (treat as no intervention)
                    kept_e = greedy_pred
                    changed_e = 0
            else:
                # No violation under this (score, alpha): no intervention, keep greedy
                kept_e = greedy_pred
                changed_e = 0
            corr_e = int(equal_strict(kept_e, gold))
            n_e += corr_e
            n_chg_e += changed_e

            # cascade-depth bucket (only when both t_worst and t* exist and t* ≤ t_worst)
            if tw is not None and ts is not None:
                gap = tw - ts
                if gap <= 0:
                    bucket = "gap_negzero"
                elif gap == 1:
                    bucket = "gap_1"
                elif 2 <= gap <= 4:
                    bucket = "gap_2_4"
                else:
                    bucket = "gap_5p"
                cascade_buckets[bucket].append(corr_e - corr_w)

        denom = max(len(boot_rows), 1)
        vanilla_accs.append(n_van / denom)
        K4_worst_accs.append(n_w / denom)
        K4_earliest_accs.append(n_e / denom)
        frac_changed_worst.append(n_chg_w / denom)
        frac_changed_earliest.append(n_chg_e / denom)
        for bk in cascade_strata:
            if cascade_buckets.get(bk):
                cascade_strata[bk].append(float(np.mean(cascade_buckets[bk])))

    if not K4_worst_accs:
        return None

    def _ci(xs, lo=2.5, hi=97.5):
        if not xs:
            return [None, None, None]
        a = np.array(xs)
        return [float(np.mean(a)), float(np.percentile(a, lo)), float(np.percentile(a, hi))]

    lift = np.array(K4_earliest_accs) - np.array(K4_worst_accs)
    return {
        "n_boot": len(K4_worst_accs),
        "vanilla_acc":       _ci(vanilla_accs),
        "K4_worst_acc":      _ci(K4_worst_accs),
        "K4_earliest_acc":   _ci(K4_earliest_accs),
        "lift_earliest_vs_worst": _ci(lift.tolist()),
        "frac_changed_worst":    _ci(frac_changed_worst),
        "frac_changed_earliest": _ci(frac_changed_earliest),
        "cascade_lift_by_gap": {
            bk: _ci(cascade_strata[bk]) for bk in cascade_strata
        },
    }

# ---------------- per-cell driver ----------------

def run_cell(model_str, tag, dataset, llm=None, tok=None):
    """Run greedy + K=4 alts (if needed), then aggregate. Returns cell_result dict."""
    cell_key = f"{tag}__{dataset}"
    out_path = OUTDIR / f"{tag}_{dataset}.json"
    cache_path = CACHE_DIR / f"greedy_alts_{tag}_{dataset}.npz"
    print(f"\n{'='*78}\n[CELL] {cell_key}\n{'='*78}", flush=True)

    # Load dataset
    ds_rows = load_dataset_uniform(dataset, NQ)
    base_prompts = None  # built lazily after we have a tokenizer

    if cache_path.exists():
        print(f"  loading cached greedy+alts from {cache_path.name}", flush=True)
        cache = np.load(cache_path, allow_pickle=True)
        greedy_records = cache["greedy_records"].tolist()
        alts_map_serialized = cache["alts_map"].item()
        alts_map = {tuple(k): v for k, v in alts_map_serialized.items()}
        base_prompts_list = cache["base_prompts"].tolist()
    else:
        if SKIP_GENERATION:
            print(f"  [SKIP_GENERATION=1] no cache and skipping; cell skipped.")
            return None
        # Build prompts using the model's tokenizer
        assert llm is not None and tok is not None, "llm/tok required for generation"
        base_prompts_list = [
            tok.apply_chat_template(
                [{"role": "user", "content": PROMPT_MATH.format(question=r["q"])}],
                tokenize=False, add_generation_prompt=True,
            )
            for r in ds_rows
        ]
        # ---- Phase 1: greedy ----
        print(f"  [phase 1/3] greedy decode (n={len(ds_rows)})...", flush=True)
        t0 = time.time()
        greedy_records = run_greedy_phase(llm, tok, base_prompts_list)
        print(f"    done in {time.time()-t0:.1f}s", flush=True)
        # Score greedy
        for rec, gd in zip(greedy_records, ds_rows):
            pred = extract_pred(rec["text"])
            rec["pred"]    = pred
            rec["gold"]    = gd["gold"]
            rec["correct"] = int(equal_strict(pred, gd["gold"]))

        # ---- Phase 2: identify all unique (trace, step_loc) pairs ----
        # For each (score, alpha), the calibration depends on the cal/test split.
        # To avoid burning compute on every possible step, we do ONE deterministic
        # full-data calibration (cal = ALL correct rows) to pre-compute the union of
        # candidate step locations per (score, alpha). Then within bootstrap we use
        # split-CP for thresholds; if the split-CP t* differs from the full-data t*
        # we'll fall back to "no alts cached → keep greedy" (logged as a miss).
        rows_for_cp = [
            {"id": r["id"], "correct": r["correct"], "T": r["T"],
             "lp": r["lp"], "ent_neg": r["ent_neg"], "marg": r["marg"]}
            for r in greedy_records
        ]
        max_T = max(r["T"] for r in rows_for_cp)
        cal_corr_full = [r for r in rows_for_cp if r["correct"] == 1]

        # Collect locs to generate K=4 for
        locs_set = set()
        for r in greedy_records:
            for sk in SCORES:
                tw = argmin_t(r[sk])
                if tw is not None:
                    locs_set.add((r["id"], tw))
        # For t*, use split-CP-aligned thresholds for *all* alphas at multiple seeds.
        rng_locs = np.random.default_rng(7)
        for sk in SCORES:
            # full-data threshold (median fallback location)
            thr_full = calibrate_per_step_thresholds(cal_corr_full, sk, max(ALPHAS) + 0.05, max_T)
            # also threshold sweep across alphas
            for alpha in ALPHAS:
                # 5 different cal/test splits — collect the union of t* candidates
                for seed_off in range(5):
                    perm = rng_locs.permutation(len(rows_for_cp))
                    nc = len(rows_for_cp) // 2
                    cal_idx = perm[:nc]
                    cal_rows = [rows_for_cp[i] for i in cal_idx]
                    cal_c = [r for r in cal_rows if r["correct"] == 1]
                    if len(cal_c) < 5:
                        continue
                    thr = calibrate_per_step_thresholds(cal_c, sk, alpha, max_T)
                    for r in rows_for_cp:
                        ts = t_star_for(r[sk], thr)
                        if ts is not None:
                            locs_set.add((r["id"], ts))
                # also include the full-data thresholds
                for r in rows_for_cp:
                    ts = t_star_for(r[sk], thr_full)
                    if ts is not None:
                        locs_set.add((r["id"], ts))

        locs = sorted(locs_set)
        print(f"  [phase 2/3] unique (trace, step_loc) pairs to generate K=4 for: {len(locs)}", flush=True)

        # ---- Phase 3: K=4 alts at every loc ----
        t0 = time.time()
        alts_map = run_K4_alts_for_locs(llm, tok, locs, base_prompts_list, greedy_records)
        print(f"    K=4 alts generated in {time.time()-t0:.1f}s ({len(alts_map)} locs)", flush=True)

        # Cache
        # Strip heavy fields (keep only what aggregate needs)
        slim_records = [{
            "id": r["id"],
            "pred": r["pred"],
            "gold": r["gold"],
            "correct": r["correct"],
            "T": r["T"],
            "lp": r["lp"],
            "ent_neg": r["ent_neg"],
            "marg": r["marg"],
        } for r in greedy_records]
        np.savez_compressed(
            cache_path,
            greedy_records=np.array(slim_records, dtype=object),
            alts_map=np.array({tuple(k): v for k, v in alts_map.items()}, dtype=object),
            base_prompts=np.array(base_prompts_list, dtype=object),
        )
        # For aggregation use slim records
        greedy_records = slim_records
        print(f"    cached → {cache_path.name}", flush=True)

    # ---------------- aggregate stats per (score, alpha) ----------------
    rows = greedy_records  # has id, pred, gold, correct, T, lp, ent_neg, marg
    cell_result = {
        "model": model_str,
        "tag": tag,
        "dataset": dataset,
        "n": len(rows),
        "vanilla_acc_pointwise": float(np.mean([r["correct"] for r in rows])),
        "n_alts_locs": len(alts_map),
        "by_score_alpha": {},
    }
    rng = np.random.default_rng(42)
    for sk in SCORES:
        for alpha in ALPHAS:
            res = bootstrap_lift(rows, sk, alpha, alts_map, n_boot=N_BOOT, rng=rng)
            cell_result["by_score_alpha"][f"{sk}__a{alpha}"] = res
            if res is None:
                continue
            print(
                f"  [agg] score={sk:8s} α={alpha}  "
                f"vanilla={res['vanilla_acc'][0]:.3f}  "
                f"K4_worst={res['K4_worst_acc'][0]:.3f}  "
                f"K4_earliest={res['K4_earliest_acc'][0]:.3f}  "
                f"Δ_lift={res['lift_earliest_vs_worst'][0]:+.4f} "
                f"[{res['lift_earliest_vs_worst'][1]:+.4f},{res['lift_earliest_vs_worst'][2]:+.4f}]",
                flush=True,
            )

    out_path.write_text(json.dumps(cell_result, indent=2, default=str))
    print(f"  wrote {out_path}")
    return cell_result

# ---------------- aggregation across cells ----------------

def aggregate_all():
    cells = {}
    for fp in sorted(OUTDIR.glob("*.json")):
        if fp.name.startswith("AGGREGATE"):
            continue
        try:
            cells[fp.stem] = json.loads(fp.read_text())
        except Exception as e:
            print(f"  [warn] could not read {fp}: {e}")
    agg = {"cells": cells}

    # Build summary table for the canonical setting (score=lp, α=0.30)
    table_rows = []
    for cell_key, cell in cells.items():
        block = cell.get("by_score_alpha", {}).get("lp__a0.3")
        if not block:
            continue
        table_rows.append({
            "cell": cell_key,
            "n": cell["n"],
            "vanilla":      block["vanilla_acc"][0],
            "K4_worst":     block["K4_worst_acc"][0],
            "K4_earliest":  block["K4_earliest_acc"][0],
            "lift":         block["lift_earliest_vs_worst"][0],
            "lift_lo":      block["lift_earliest_vs_worst"][1],
            "lift_hi":      block["lift_earliest_vs_worst"][2],
            "frac_changed_worst":    block["frac_changed_worst"][0],
            "frac_changed_earliest": block["frac_changed_earliest"][0],
            "cascade_gap1":   block["cascade_lift_by_gap"]["gap_1"][0],
            "cascade_gap2_4": block["cascade_lift_by_gap"]["gap_2_4"][0],
            "cascade_gap5p":  block["cascade_lift_by_gap"]["gap_5p"][0],
        })
    agg["summary_lp_a0.3"] = table_rows

    (OUTDIR / "AGGREGATE.json").write_text(json.dumps(agg, indent=2, default=str))

    # Markdown
    lines = []
    lines.append("# Pearl Causal Step Intervention — Full Experiment\n")
    lines.append("**Setup:** K=4 majority re-roll at t* (earliest-bad-step) vs t_worst (worst-step baseline). "
                 "Bootstrap CIs: 500 resamples × 1 cal/test split per resample (CP calibration on correct half).\n")
    lines.append("## Headline table (score=lp, α=0.30)\n")
    lines.append("| cell | n | vanilla | K4_worst | K4_earliest | Δlift | 95% CI | %changed_worst | %changed_earliest |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in table_rows:
        lines.append(
            f"| {r['cell']} | {r['n']} | {r['vanilla']:.3f} | {r['K4_worst']:.3f} | {r['K4_earliest']:.3f} | "
            f"{r['lift']:+.4f} | [{r['lift_lo']:+.4f},{r['lift_hi']:+.4f}] | "
            f"{r['frac_changed_worst']*100:.1f}% | {r['frac_changed_earliest']*100:.1f}% |"
        )
    lines.append("\n## Cascade-depth-stratified lift (Δ = K4_earliest − K4_worst per stratum)\n")
    lines.append("| cell | gap=1 | gap=2..4 | gap≥5 |")
    lines.append("|---|---|---|---|")
    for r in table_rows:
        def fmt(x):
            return "n/a" if x is None else f"{x:+.4f}"
        lines.append(f"| {r['cell']} | {fmt(r['cascade_gap1'])} | {fmt(r['cascade_gap2_4'])} | {fmt(r['cascade_gap5p'])} |")

    # Aggregate average lift
    lifts = [r["lift"] for r in table_rows]
    if lifts:
        lines.append(f"\n## Aggregate (mean across {len(lifts)} cells)\n")
        lines.append(f"- mean(K4_earliest − K4_worst) = **{np.mean(lifts):+.4f}**")
        lines.append(f"- median = {np.median(lifts):+.4f}")
        lines.append(f"- # cells with lift > 0: {sum(1 for x in lifts if x > 0)}/{len(lifts)}")
        lines.append(f"- # cells with CI excluding 0: {sum(1 for r in table_rows if r['lift_lo'] > 0 or r['lift_hi'] < 0)}/{len(table_rows)}")
    (OUTDIR / "AGGREGATE.md").write_text("\n".join(lines) + "\n")
    print(f"\n[done] wrote AGGREGATE.{{json,md}} ({len(cells)} cells)")
    return agg

# ---------------- main ----------------

def main():
    cell_only = os.environ.get("CELL_ONLY", "").strip()
    cell_filter = set(cell_only.split(";")) if cell_only else None

    grid = []
    if "MODEL" in os.environ and "TAG" in os.environ and "DATASET" in os.environ:
        grid.append((os.environ["MODEL"], os.environ["TAG"], os.environ["DATASET"]))
    else:
        for model, tag in DEFAULT_GRID:
            for ds in DEFAULT_DATASETS:
                grid.append((model, tag, ds))

    # Group by model so we only initialize the LLM once per model
    by_model = defaultdict(list)
    for m, t, ds in grid:
        ck = f"{t}__{ds}"
        if cell_filter and ck not in cell_filter:
            continue
        by_model[(m, t)].append(ds)

    if not by_model:
        print("Nothing to run.")
        aggregate_all()
        return

    if SKIP_GENERATION:
        # Just re-aggregate from caches
        for (m, t), datasets in by_model.items():
            for ds in datasets:
                run_cell(m, t, ds, llm=None, tok=None)
        aggregate_all()
        return

    from vllm import LLM  # type: ignore

    for (model_str, tag), datasets in by_model.items():
        # Decide tensor parallel size by model
        tp = 2 if "32" in model_str else 1
        # Phi-4 is 14B, fits on one H100 NVL
        print(f"\n[model] loading {model_str} (tp={tp})...", flush=True)
        t0 = time.time()
        llm = LLM(
            model=model_str,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=tp,
            seed=SEED,
            trust_remote_code=True,
        )
        tok = llm.get_tokenizer()
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        for ds in datasets:
            try:
                run_cell(model_str, tag, ds, llm=llm, tok=tok)
            except Exception as e:
                print(f"  [ERROR] cell {tag}__{ds} failed: {e}", flush=True)
                import traceback; traceback.print_exc()
        # Free LLM
        del llm
        try:
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    aggregate_all()


if __name__ == "__main__":
    main()

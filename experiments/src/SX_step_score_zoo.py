"""
EXP-STEP-ZOO: 12 free step-level scoring features from Wave 3 brainstorm.

Computes per-step features online from greedy decoding with logprobs=20:
- F1 step_lp_zscore        (running z-score)
- F2 step_lp_drawdown      (peak-to-current drawdown)
- F3 step_lp_rank_quantile (online rank)
- F4 step_entropy_growth_ratio
- F5 step_local_lp_jump    (max down-jump)
- F6 step_neighbor_js      (JS between adjacent step distribs)
- F7 step_arith_violations (sympy/safe_eval check on LHS=RHS)
- F8 step_backtrack_marker (count of wait/actually/etc)
- F9 step_repetition_4gram (max 4gram overlap with prev step)
- F10 step_numeric_density
- F11 step_token_curvature (second diff)
- F12 step_branch_marker

Each step-level feature is aggregated to trajectory level via 4 aggregators:
  max, mean, last, count_high (count above 0.5 threshold for normalized features)

Cross-model + cross-dataset validation matrix.

Env vars:
  MODEL, TAG, DATASETS, NQ
"""

import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, equal_strict

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
TAG = os.environ.get("TAG", "qwen25_7b")
DATASETS = os.environ.get("DATASETS", "math500,aime,olympiad").split(",")
N_PER_DS = int(os.environ.get("NQ", "200"))
TOPK = 20
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / f"SX_step_zoo_{TAG}.json"

PROMPT_MATH = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\nProblem: {question}\n\n"
)

BACKTRACK_RE = re.compile(r"\b(wait|actually|hmm|but(?!\s+also)|no wait|let me try|let me reconsider|on second thought|reconsider)\b", re.IGNORECASE)
BRANCH_RE = re.compile(r"\b(case [\d\w]+|alternatively|however|on the other hand|either way|in either case)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+\.?\d*")
OPERATOR_RE = re.compile(r"[+\-*/=<>≤≥≠]")
EQUATION_RE = re.compile(r"([^\s=]+)\s*=\s*([^\s=]+)")


def load_dataset_uniform(name, n):
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["problem"], "gold": r["answer"]} for r in ds]
    elif name == "aime":
        ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["Question"], "gold": str(r["Answer"])} for r in ds]
    elif name == "olympiad":
        ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["question"], "gold": str(r["final_answer"][0]) if r["final_answer"] else ""} for r in ds]
    raise ValueError(name)


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def topk_dist(topk_dict):
    if not topk_dict: return None
    entries = sorted(topk_dict.values(), key=lambda v: -v.logprob)
    lps = np.array([e.logprob for e in entries], dtype=float)
    probs = np.exp(lps - lps.max()); probs = probs / probs.sum()
    return {"probs": probs.tolist(), "lps": lps.tolist(), "ids": [e.decoded_token if hasattr(e, "decoded_token") else None for e in entries]}


def js_div(p, q):
    p = np.asarray(p); q = np.asarray(q)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float((a[mask] * np.log(a[mask] / b[mask])).sum()) if mask.sum() else 0.0
    return 0.5 * (_kl(p, m) + _kl(q, m))


def step_basic(distribs, token_lps_seg):
    """Compute lp_mean, entropy_mean, top1_margin_mean for one step."""
    if not distribs:
        return {"lp": float("nan"), "ent": float("nan"), "marg": float("nan")}
    ents, margs = [], []
    for d in distribs:
        if d is None: continue
        p = np.array(d["probs"]); p = p[p > 0]
        if len(p) > 0:
            ents.append(-float((p * np.log(p)).sum()))
        if len(d["lps"]) >= 2:
            margs.append(d["lps"][0] - d["lps"][1])
    valid_lp = [lp for lp in token_lps_seg if not (lp != lp)]
    return {
        "lp": float(np.mean(valid_lp)) if valid_lp else float("nan"),
        "ent": float(np.mean(ents)) if ents else float("nan"),
        "marg": float(np.mean(margs)) if margs else float("nan"),
    }


# ============ FREE step-level features (operate per-step from prefix info) ============

class WelfordOnline:
    """Running mean/var via Welford."""
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
    def var(self):
        return self.M2 / max(self.n - 1, 1)
    def std(self):
        v = self.var()
        return math.sqrt(v) if v > 0 else 1e-9


def compute_step_features(per_step_basic, per_step_distribs, per_step_text):
    """Compute all 12 step-level scores per step. Returns list of dicts (one per step)."""
    n_steps = len(per_step_basic)
    out = []
    welford_lp = WelfordOnline()
    welford_ent = WelfordOnline()
    welford_len = WelfordOnline()
    lp_history = []
    ent_history = []
    running_max_lp = float("-inf")
    prev_lp = None; prev_ent = None
    prev_step_distribs = None
    prev_step_text = ""
    for t, (basic, distribs, text) in enumerate(zip(per_step_basic, per_step_distribs, per_step_text)):
        feats = {}
        # ---- F1 step_lp_zscore (signed, lower = more anomalous low) ----
        if welford_lp.n >= 2:
            feats["lp_zscore"] = (basic["lp"] - welford_lp.mean) / welford_lp.std()
        else:
            feats["lp_zscore"] = 0.0
        # ---- F2 step_lp_drawdown ----
        feats["lp_drawdown"] = max(0.0, running_max_lp - basic["lp"]) if not (basic["lp"] != basic["lp"]) and running_max_lp > -1e9 else 0.0
        # ---- F3 step_lp_rank_quantile (online rank) ----
        if lp_history:
            rank = sum(1 for h in lp_history if basic["lp"] >= h)
            feats["lp_rank_q"] = rank / len(lp_history)
        else:
            feats["lp_rank_q"] = 0.5
        # ---- F4 step_entropy_growth_ratio ----
        if ent_history and basic["ent"] > 0:
            geo_mean = math.exp(np.mean([math.log(max(e, 1e-9)) for e in ent_history]))
            feats["ent_growth"] = math.log(basic["ent"] / max(geo_mean, 1e-9))
        else:
            feats["ent_growth"] = 0.0
        # ---- F5 step_local_lp_jump (down-jump only) ----
        if prev_lp is not None and not (basic["lp"] != basic["lp"]) and not (prev_lp != prev_lp):
            feats["lp_jump"] = max(0.0, prev_lp - basic["lp"])  # positive when current is worse
        else:
            feats["lp_jump"] = 0.0
        # ---- F6 step_neighbor_js (JS of token distribs between adj steps) ----
        if prev_step_distribs and distribs:
            # For each token position match, compute mean JS; here we average across all tokens
            try:
                # Take the FIRST distrib of each step (most stable), or AVG token JS
                # We just compare the mean distribution across each step
                cur_avg = np.zeros(TOPK)
                cur_n = 0
                for d in distribs:
                    if d is None: continue
                    p = np.zeros(TOPK)
                    p[:len(d["probs"])] = d["probs"][:TOPK]
                    cur_avg += p; cur_n += 1
                cur_avg = cur_avg / max(cur_n, 1)
                prev_avg = np.zeros(TOPK)
                prev_n = 0
                for d in prev_step_distribs:
                    if d is None: continue
                    p = np.zeros(TOPK)
                    p[:len(d["probs"])] = d["probs"][:TOPK]
                    prev_avg += p; prev_n += 1
                prev_avg = prev_avg / max(prev_n, 1)
                if cur_avg.sum() > 0 and prev_avg.sum() > 0:
                    feats["neighbor_js"] = js_div(cur_avg / cur_avg.sum(), prev_avg / prev_avg.sum())
                else:
                    feats["neighbor_js"] = 0.0
            except Exception:
                feats["neighbor_js"] = 0.0
        else:
            feats["neighbor_js"] = 0.0
        # ---- F7 step_arith_violations (regex equation + safe_eval) ----
        n_eq = 0; n_pass = 0
        for m in EQUATION_RE.finditer(text):
            lhs, rhs = m.group(1).strip(), m.group(2).strip()
            n_eq += 1
            try:
                # Tighten safe_eval: only allow digits, operators, and parens
                if re.match(r"^[\d\s\.\+\-\*\/\(\)]+$", lhs) and re.match(r"^[\d\s\.\+\-\*\/\(\)]+$", rhs):
                    lv = eval(lhs, {"__builtins__": {}}, {})
                    rv = eval(rhs, {"__builtins__": {}}, {})
                    if abs(float(lv) - float(rv)) < 1e-6:
                        n_pass += 1
            except Exception:
                pass
        feats["arith_violations"] = (n_eq - n_pass)  # raw violation count
        # ---- F8 step_backtrack_marker ----
        feats["backtrack"] = len(BACKTRACK_RE.findall(text))
        # ---- F9 step_repetition_4gram (vs prev step text) ----
        if prev_step_text:
            cur_tokens = re.findall(r"\w+", text.lower())
            prev_tokens = re.findall(r"\w+", prev_step_text.lower())
            if len(cur_tokens) >= 4 and len(prev_tokens) >= 4:
                cur_4grams = set(tuple(cur_tokens[i:i+4]) for i in range(len(cur_tokens) - 3))
                prev_4grams = set(tuple(prev_tokens[i:i+4]) for i in range(len(prev_tokens) - 3))
                if cur_4grams:
                    feats["rep_4gram"] = len(cur_4grams & prev_4grams) / len(cur_4grams)
                else:
                    feats["rep_4gram"] = 0.0
            else:
                feats["rep_4gram"] = 0.0
        else:
            feats["rep_4gram"] = 0.0
        # ---- F10 step_numeric_density ----
        n_tok = max(len(text.split()), 1)
        n_num = len(NUMBER_RE.findall(text))
        n_op = len(OPERATOR_RE.findall(text))
        feats["num_density"] = (n_num + n_op) / n_tok
        # ---- F11 step_token_curvature (second diff of margin) ----
        if t >= 2:
            m_t, m_tm1, m_tm2 = per_step_basic[t]["marg"], per_step_basic[t-1]["marg"], per_step_basic[t-2]["marg"]
            if not any(math.isnan(x) for x in [m_t, m_tm1, m_tm2]):
                feats["token_curvature"] = m_t - 2 * m_tm1 + m_tm2
            else:
                feats["token_curvature"] = 0.0
        else:
            feats["token_curvature"] = 0.0
        # ---- F12 step_branch_marker ----
        feats["branch"] = len(BRANCH_RE.findall(text))

        out.append(feats)

        # Update running state
        if not (basic["lp"] != basic["lp"]):
            welford_lp.update(basic["lp"])
            lp_history.append(basic["lp"])
            running_max_lp = max(running_max_lp, basic["lp"])
            prev_lp = basic["lp"]
        if not (basic["ent"] != basic["ent"]):
            welford_ent.update(basic["ent"])
            ent_history.append(basic["ent"])
            prev_ent = basic["ent"]
        prev_step_distribs = distribs
        prev_step_text = text
    return out


def trajectory_aggregate(step_feats):
    """Aggregate per-step features to trajectory-level scores via max/mean/last/count_high."""
    out = {}
    keys = ["lp_zscore", "lp_drawdown", "lp_rank_q", "ent_growth", "lp_jump",
            "neighbor_js", "arith_violations", "backtrack", "rep_4gram",
            "num_density", "token_curvature", "branch"]
    for k in keys:
        vals = [s.get(k, 0.0) for s in step_feats]
        if not vals:
            for ag in ["max", "min", "mean", "last", "sum"]:
                out[f"{k}_{ag}"] = float("nan")
            continue
        out[f"{k}_max"] = float(np.max(vals))
        out[f"{k}_min"] = float(np.min(vals))
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_last"] = float(vals[-1]) if vals else 0.0
        out[f"{k}_sum"] = float(np.sum(vals))
    return out


def cp_eval_with_ci(scores, correct, alpha, n_boot=300, n_seeds=5):
    s = np.array(scores, dtype=float); c = np.array(correct, dtype=int)
    valid = ~np.isnan(s)
    s = s[valid]; c = c[valid]
    if len(s) < 20: return None
    rng = np.random.default_rng(0)
    boot_acc = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), size=len(s))
        sb = s[idx]; cb = c[idx]
        accs = []
        for _ in range(n_seeds):
            perm = rng.permutation(len(sb))
            nc = len(sb) // 2
            ci, ti = perm[:nc], perm[nc:]
            cal_corr = sb[ci][cb[ci] == 1]
            if len(cal_corr) < 5: continue
            n_c = len(cal_corr)
            ql = max(0.0, min(1.0, math.floor(alpha * (n_c + 1)) / n_c))
            q = float(np.quantile(cal_corr, ql))
            kept = sb[ti] >= q
            if kept.sum():
                accs.append(float(cb[ti][kept].mean()))
        if accs:
            boot_acc.append(np.mean(accs))
    if not boot_acc: return None
    return {"kept_acc": float(np.mean(boot_acc)),
            "ci95": [float(np.quantile(boot_acc, 0.025)), float(np.quantile(boot_acc, 0.975))]}


def main():
    print(f"=== Step-zoo: {MODEL} on {DATASETS} ===", flush=True)
    print(f"Loading {MODEL}...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2560,
        tensor_parallel_size=1,
        seed=SEED,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    all_results = {"model": MODEL, "tag": TAG, "datasets": {}}

    for ds_name in DATASETS:
        ds_name = ds_name.strip()
        if not ds_name: continue
        print(f"\n--- Dataset: {ds_name} ---", flush=True)
        try:
            ds = load_dataset_uniform(ds_name, N_PER_DS)
        except Exception as e:
            print(f"  load failed: {e}", flush=True); continue

        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": PROMPT_MATH.format(question=r["q"])}],
                tokenize=False, add_generation_prompt=True,
            )
            for r in ds
        ]
        max_tokens = 1536 if ds_name == "math500" else 2048
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=TOPK, seed=SEED)
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        print(f"  generated {len(outs)} in {time.time()-t0:.1f}s", flush=True)

        rows = []
        for i, (rec, out) in enumerate(zip(ds, outs)):
            o = out.outputs[0]
            text = o.text
            tok_ids = list(o.token_ids)
            tok_strs = [tok.decode([tid]) for tid in tok_ids]
            chosen_lps = []; distribs = []
            for step_lp in o.logprobs or []:
                if step_lp is None:
                    chosen_lps.append(float("nan")); distribs.append(None); continue
                entries = sorted(step_lp.items(), key=lambda kv: -kv[1].logprob)
                chosen_lps.append(entries[0][1].logprob)
                distribs.append(topk_dist(step_lp))
            boundaries = find_step_boundaries(tok_strs)
            n = len(tok_ids)
            bdy = boundaries + [n]
            per_step_basic = []; per_step_distribs = []; per_step_text = []
            for sa, sb in zip(bdy[:-1], bdy[1:]):
                per_step_basic.append(step_basic(distribs[sa:sb], chosen_lps[sa:sb]))
                per_step_distribs.append(distribs[sa:sb])
                per_step_text.append(tok.decode(tok_ids[sa:sb], skip_special_tokens=True))

            step_feats = compute_step_features(per_step_basic, per_step_distribs, per_step_text)
            traj = trajectory_aggregate(step_feats)
            # Also compute lp_min baseline + entropy_mean + top1_margin_mean for comparison
            lp_means = [b["lp"] for b in per_step_basic if not (b["lp"] != b["lp"])]
            ent_means = [b["ent"] for b in per_step_basic if not (b["ent"] != b["ent"])]
            marg_means = [b["marg"] for b in per_step_basic if not (b["marg"] != b["marg"])]
            traj["lp_min"] = float(np.min(lp_means)) if lp_means else float("nan")
            traj["entropy_mean"] = float(np.mean(ent_means)) if ent_means else float("nan")
            traj["top1_margin_mean"] = float(np.mean(marg_means)) if marg_means else float("nan")

            pred = extract_pred(text)
            ok = int(equal_strict(pred, rec["gold"]))
            traj["correct"] = ok; traj["n_steps"] = len(step_feats)
            rows.append(traj)

        n_correct = sum(r["correct"] for r in rows)
        vanilla = n_correct / len(rows) if rows else 0
        print(f"  vanilla = {vanilla:.3f} ({n_correct}/{len(rows)})", flush=True)

        # Spearman + CP for all features
        from scipy.stats import spearmanr
        correct = np.array([r["correct"] for r in rows])
        score_keys = ["lp_min", "entropy_mean", "top1_margin_mean"]  # baselines
        # All step-level aggregated features
        for fkey in ["lp_zscore", "lp_drawdown", "lp_rank_q", "ent_growth", "lp_jump",
                     "neighbor_js", "arith_violations", "backtrack", "rep_4gram",
                     "num_density", "token_curvature", "branch"]:
            for ag in ["max", "min", "mean", "last", "sum"]:
                score_keys.append(f"{fkey}_{ag}")

        ds_results = {"vanilla_acc": vanilla, "n": len(rows)}
        for sk in score_keys:
            scores = np.array([r.get(sk, float("nan")) for r in rows], dtype=float)
            valid = ~np.isnan(scores)
            if valid.sum() < 20: continue
            rho, p = spearmanr(scores[valid], correct[valid])
            scores_dir = scores * (1 if rho >= 0 else -1)
            cp03 = cp_eval_with_ci(scores_dir, correct, 0.30)
            if cp03 is None: continue
            ds_results[sk] = {"rho": float(rho), "p": float(p), "kept_acc_alpha_0.30": cp03}

        # Print top 10 by kept_acc
        ranked = sorted(
            [(sk, ds_results[sk]["kept_acc_alpha_0.30"]["kept_acc"], ds_results[sk]["rho"])
             for sk in score_keys if sk in ds_results],
            key=lambda x: -x[1]
        )
        for sk, kept, rho in ranked[:12]:
            print(f"  {sk:30s} kept@0.3={kept:.3f}  ρ={rho:+.3f}", flush=True)

        all_results["datasets"][ds_name] = ds_results
        OUT.write_text(json.dumps(all_results, indent=2))

    print(f"\n=== Done. Wrote {OUT} ===", flush=True)


if __name__ == "__main__":
    main()

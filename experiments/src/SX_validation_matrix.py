"""
EXP-VALIDATION: cross-dataset, cross-model validation of new info-theoretic scores.

Goal: confirm that `top1_margin_mean`, `tempered_kl_max`, `entropy_max`, etc.
generalize across datasets and models, with bootstrap CIs.

Loads ONE model, runs through 5 datasets with logprobs=20 each.

Env vars:
  MODEL: HF model ID
  GPU: which GPU (CUDA_VISIBLE_DEVICES set externally)
  TAG: short tag for output files
  DATASETS: comma-sep list, default 'math500,aime,olympiad,mmlu_pro,humaneval'

Output: SX_validation_<TAG>.json with per-dataset scores + CIs.
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, equal_strict, normalize

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
TAG = os.environ.get("TAG", "qwen25_7b")
DATASETS = os.environ.get("DATASETS", "math500,aime,olympiad,mmlu_pro,humaneval").split(",")
N_PER_DS = int(os.environ.get("NQ", "200"))
TOPK = 20
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / f"SX_validation_{TAG}.json"

PROMPT_MATH = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\nProblem: {question}\n\n"
)

PROMPT_MMLU = (
    "Answer the following multiple choice question. Reason step by step in clearly numbered steps "
    'separated by blank lines. End with a line: "Answer: <letter>".\n\n'
    "Question: {question}\n\nOptions:\n{options}\n\n"
)

PROMPT_HUMANEVAL = (
    "Complete the following Python function. Reason about the problem step by step in commented "
    "blocks separated by blank lines, then write the implementation.\n\n```python\n{prompt}\n```\n\n"
)


def load_dataset_uniform(name, n):
    """Return list of dicts with 'question' and 'gold' fields, normalized across datasets."""
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["problem"], "gold": r["answer"], "type": "math"} for r in ds]
    elif name == "aime":
        # Try larger AIME archive first; fall back to AIME 2024
        for spec in [
            ("agentlans/aime-1983-2024", None, "train", "Problem", "Answer"),
            ("di-zhang-fdu/AIME_1983_2024", None, "train", "Question", "Answer"),
            ("AI-MO/aimo-validation-aime", None, "train", "problem", "answer"),
            ("Maxwell-Jia/AIME_2024", None, "train", "Problem", "Answer"),
        ]:
            try:
                if spec[1] is None:
                    ds = load_dataset(spec[0], split=spec[2])
                else:
                    ds = load_dataset(spec[0], spec[1], split=spec[2])
                ds = ds.select(range(min(n, len(ds))))
                qkey, akey = spec[3], spec[4]
                rows = [{"q": r[qkey], "gold": str(r[akey]), "type": "math"} for r in ds]
                print(f"  AIME loaded from {spec[0]}: {len(rows)} problems", flush=True)
                return rows
            except Exception as e:
                print(f"  AIME try {spec[0]} failed: {e}", flush=True)
                continue
        return []
    elif name == "olympiad":
        ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["question"], "gold": str(r["final_answer"][0]) if r["final_answer"] else "", "type": "math"} for r in ds]
    elif name == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        # Filter STEM only
        stem_categories = {"math", "physics", "chemistry", "biology", "computer science", "engineering"}
        ds = ds.filter(lambda r: r["category"] in stem_categories)
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["question"], "options": r["options"], "gold": r["answer"], "type": "mmlu"} for r in ds]
    elif name == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["prompt"], "gold": r["canonical_solution"], "test": r["test"], "type": "code"} for r in ds]
    else:
        raise ValueError(f"Unknown dataset {name}")


def make_prompt(rec, tok):
    if rec["type"] == "math":
        text = PROMPT_MATH.format(question=rec["q"])
    elif rec["type"] == "mmlu":
        opts = "\n".join([f"  {chr(65+i)}. {o}" for i, o in enumerate(rec["options"])])
        text = PROMPT_MMLU.format(question=rec["q"], options=opts)
    elif rec["type"] == "code":
        text = PROMPT_HUMANEVAL.format(prompt=rec["q"])
    else:
        raise ValueError(rec["type"])
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def topk_to_distribution(topk_dict):
    if not topk_dict: return None
    entries = sorted(topk_dict.values(), key=lambda v: -v.logprob)
    lps = np.array([e.logprob for e in entries], dtype=float)
    probs = np.exp(lps - lps.max()); probs = probs / probs.sum()
    return {"probs": probs.tolist(), "lps": lps.tolist()}


def js_divergence(p, q):
    p = np.asarray(p); q = np.asarray(q)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float((a[mask] * np.log(a[mask] / b[mask])).sum()) if mask.sum() else 0.0
    return 0.5 * (_kl(p, m) + _kl(q, m))


def step_features(distribs, token_lps_seg):
    if not distribs:
        return {k: float("nan") for k in ["entropy", "top1_margin", "tempered_kl", "kl_uniform", "lp_mean"]}
    ents, margs, kls, kus = [], [], [], []
    for d in distribs:
        if d is None: continue
        p = np.array(d["probs"]); p = p[p > 0]
        if len(p) > 0:
            H = -float((p * np.log(p)).sum())
            ents.append(H)
            kus.append(math.log(len(p)) - H)
        if len(d["lps"]) >= 2:
            margs.append(d["lps"][0] - d["lps"][1])
        lps = np.asarray(d["lps"])
        p1 = np.exp(lps / 1.0 - (lps / 1.0).max()); p1 = p1 / p1.sum()
        p2 = np.exp(lps / 2.0 - (lps / 2.0).max()); p2 = p2 / p2.sum()
        kls.append(js_divergence(p1, p2))
    valid_lp = [lp for lp in token_lps_seg if not (lp != lp)]
    return {
        "entropy": float(np.mean(ents)) if ents else float("nan"),
        "top1_margin": float(np.mean(margs)) if margs else float("nan"),
        "tempered_kl": float(np.mean(kls)) if kls else float("nan"),
        "kl_uniform": float(np.mean(kus)) if kus else float("nan"),
        "lp_mean": float(np.mean(valid_lp)) if valid_lp else float("nan"),
    }


def trajectory_aggregates(step_features_list):
    """Aggregate per-step features into trajectory-level scores."""
    out = {}
    for k in ["entropy", "top1_margin", "tempered_kl", "kl_uniform", "lp_mean"]:
        vals = [s[k] for s in step_features_list if not (s[k] != s[k])]
        if not vals:
            out[f"{k}_min"] = float("nan"); out[f"{k}_max"] = float("nan"); out[f"{k}_mean"] = float("nan")
            continue
        out[f"{k}_min"] = float(np.min(vals))
        out[f"{k}_max"] = float(np.max(vals))
        out[f"{k}_mean"] = float(np.mean(vals))
    return out


def cp_eval_with_ci(scores, correct, alpha, n_boot=300, n_seeds=5):
    """Bootstrap CP kept_acc with 95% CI."""
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
    print(f"=== Validation matrix run ===", flush=True)
    print(f"  MODEL: {MODEL}", flush=True)
    print(f"  TAG:   {TAG}", flush=True)
    print(f"  DATASETS: {DATASETS}", flush=True)
    print(f"  N_PER_DS: {N_PER_DS}, TOPK: {TOPK}", flush=True)

    print(f"\nLoading {MODEL}...", flush=True)
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
            print(f"  load failed: {e}", flush=True)
            all_results["datasets"][ds_name] = {"error": str(e)}
            continue
        if not ds:
            continue
        prompts = [make_prompt(r, tok) for r in ds]
        max_tokens = 2048 if ds_name in ("aime", "olympiad") else 1536
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=TOPK, seed=SEED)
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        gen_time = time.time() - t0
        print(f"  generated {len(outs)} in {gen_time:.1f}s", flush=True)

        rows = []
        for i, (rec, out) in enumerate(zip(ds, outs)):
            o = out.outputs[0]
            text = o.text
            tok_ids = list(o.token_ids)
            tok_strs = [tok.decode([tid]) for tid in tok_ids]
            chosen_lps = []
            distribs = []
            for step_lp in o.logprobs or []:
                if step_lp is None:
                    chosen_lps.append(float("nan")); distribs.append(None); continue
                entries = sorted(step_lp.items(), key=lambda kv: -kv[1].logprob)
                chosen_lps.append(entries[0][1].logprob)
                distribs.append(topk_to_distribution(step_lp))
            boundaries = find_step_boundaries(tok_strs)
            n = len(tok_ids)
            bdy = boundaries + [n]
            sfeats = []
            for sa, sb in zip(bdy[:-1], bdy[1:]):
                sfeats.append(step_features(distribs[sa:sb], chosen_lps[sa:sb]))
            traj = trajectory_aggregates(sfeats)

            pred = extract_pred(text)
            if rec["type"] == "code":
                # crude code correctness: substring match of canonical solution
                # Actually use AST matching is hard; for now use exact text equality on the function body.
                # We just record the raw text — correctness for HumanEval will be by exec test, but skipped here.
                ok = 0  # placeholder
            elif rec["type"] == "mmlu":
                # answer is letter
                gold_letter = chr(65 + rec["gold"]) if isinstance(rec["gold"], int) else str(rec["gold"]).strip()
                pred_letter = (pred or "").strip().upper()[:1]
                ok = int(pred_letter == gold_letter[:1].upper())
            else:
                ok = int(equal_strict(pred, rec["gold"]))
            rows.append({**traj, "correct": ok, "n_tokens": n, "n_steps": len(sfeats)})

        n_correct = sum(r["correct"] for r in rows)
        vanilla_acc = n_correct / len(rows) if rows else 0
        print(f"  vanilla_acc = {vanilla_acc:.3f} ({n_correct}/{len(rows)})", flush=True)

        # Spearman correlations
        from scipy.stats import spearmanr
        correct = np.array([r["correct"] for r in rows])
        score_keys = [
            "lp_mean_min",  # baseline lp_min equivalent
            "top1_margin_mean", "top1_margin_min",
            "entropy_max", "entropy_mean",
            "tempered_kl_max", "tempered_kl_mean",
            "kl_uniform_min", "kl_uniform_mean",
        ]
        ds_results = {"vanilla_acc": vanilla_acc, "n": len(rows), "gen_time_sec": gen_time}
        for sk in score_keys:
            scores = np.array([r.get(sk, float("nan")) for r in rows], dtype=float)
            valid = ~np.isnan(scores)
            if valid.sum() < 20:
                continue
            rho, p = spearmanr(scores[valid], correct[valid])
            # CP: rho-correct sign
            scores_dir = scores * (1 if rho >= 0 else -1)
            cp03 = cp_eval_with_ci(scores_dir, correct, 0.30)
            cp05 = cp_eval_with_ci(scores_dir, correct, 0.50)
            ds_results[sk] = {
                "rho": float(rho), "p": float(p),
                "kept_acc_alpha_0.30": cp03,
                "kept_acc_alpha_0.50": cp05,
            }
            print(f"  {sk:20s} ρ={rho:+.3f} p={p:.3g} kept@0.3={cp03['kept_acc']:.3f} CI=[{cp03['ci95'][0]:.3f},{cp03['ci95'][1]:.3f}]" if cp03 else
                  f"  {sk:20s} ρ={rho:+.3f} p={p:.3g} (CP: insufficient)", flush=True)

        all_results["datasets"][ds_name] = ds_results
        # Save incrementally
        OUT.write_text(json.dumps(all_results, indent=2))

    print(f"\n=== Done. Wrote {OUT} ===", flush=True)


if __name__ == "__main__":
    main()

"""
EXP-S4: Information-theoretic scores via vLLM logprobs=20.

Implements Wave 1 Agent A's:
- step_entropy_p95: 95-percentile across steps of mean token entropy
- step_entropy_mean: average step entropy
- step_entropy_max: max step entropy
- predictive_sharpening_rate: OLS slope of entropy vs step idx (already from chosen-token proxy in S1; here recomputed with full top-k)
- tempered_kl_divergence: JS between low-T (T=1.0 softmax) and high-T (T=2.0 softmax) of TOP-20 logprobs at each token
- top1_top2_gap: average margin between top-1 and top-2 token logprob (informative proxy)
- step_perplexity: per-step exp(-mean log p)
- step_kl_to_uniform: KL of top-k distribution to uniform (concentration measure)

Uses MATH-500 200 problems on Qwen2.5-7B-Instruct with logprobs=20.

Output: SX_info_theoretic.json + per-trace JSONL with all info-theoretic scores.
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

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_QUESTIONS = int(os.environ.get("NQ", "200"))
TOPK = 20
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / "SX_info_theoretic.json"
OUT_TRACES = OUTDIR / "SX_info_theoretic_traces.jsonl"

PROMPT_BASE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
)


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def topk_to_distribution(topk_dict, n_tokens_proxy=20):
    """Given vLLM's logprobs dict {token_id: Logprob} for one position, return:
    - probs: list of probabilities (top-k)
    - tokens: list of token ids
    The remaining (vocab - k) probability mass is treated as if uniformly distributed
    in residual_mass. We mostly need entropy/KL within the top-k tail.
    """
    if not topk_dict:
        return None
    entries = sorted(topk_dict.values(), key=lambda v: -v.logprob)
    lps = np.array([e.logprob for e in entries], dtype=float)
    # Convert to probabilities (softmax over top-k slice; this is a proxy)
    probs = np.exp(lps - lps.max())
    probs = probs / probs.sum()
    return {"probs": probs.tolist(), "lps": lps.tolist(), "n": len(probs)}


def step_entropy(distribs):
    """Mean Shannon entropy across tokens within a step (using top-k slice)."""
    if not distribs:
        return float("nan")
    ents = []
    for d in distribs:
        if d is None: continue
        p = np.array(d["probs"])
        p = p[p > 0]
        if len(p) > 0:
            ents.append(-float((p * np.log(p)).sum()))
    return float(np.mean(ents)) if ents else float("nan")


def step_top1_margin(distribs):
    """Mean margin (logprob_top1 - logprob_top2) across tokens in a step."""
    if not distribs:
        return float("nan")
    margins = []
    for d in distribs:
        if d is None or len(d["lps"]) < 2: continue
        margins.append(d["lps"][0] - d["lps"][1])
    return float(np.mean(margins)) if margins else float("nan")


def step_perplexity(token_logprobs_chosen):
    """exp(-mean log p of chosen tokens) within a step."""
    valid = [lp for lp in token_logprobs_chosen if not (lp != lp)]
    if not valid:
        return float("nan")
    return float(np.exp(-np.mean(valid)))


def js_divergence(p, q):
    """JS divergence between two probability vectors (same length)."""
    p = np.asarray(p); q = np.asarray(q)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        if mask.sum() == 0: return 0.0
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())
    return 0.5 * (_kl(p, m) + _kl(q, m))


def step_tempered_kl(distribs, T1=1.0, T2=2.0):
    """JS divergence between two temperature-rescaled distributions over the same top-k logprobs.
    Per-step mean of token-level JS.
    """
    if not distribs: return float("nan")
    js_list = []
    for d in distribs:
        if d is None: continue
        lps = np.asarray(d["lps"])
        # T1 softmax (sharper since T1=1, then T2 softer)
        p1 = np.exp(lps / T1 - (lps / T1).max())
        p1 = p1 / p1.sum()
        p2 = np.exp(lps / T2 - (lps / T2).max())
        p2 = p2 / p2.sum()
        js_list.append(js_divergence(p1, p2))
    return float(np.mean(js_list)) if js_list else float("nan")


def step_top1_concentration(distribs):
    """Top-1 probability mass relative to top-k slice."""
    if not distribs: return float("nan")
    vals = []
    for d in distribs:
        if d is None: continue
        if len(d["probs"]) > 0:
            vals.append(d["probs"][0])
    return float(np.mean(vals)) if vals else float("nan")


def step_kl_to_uniform(distribs):
    """KL(p, uniform) per token (i.e., log k - H). Higher = more concentrated."""
    if not distribs: return float("nan")
    vals = []
    for d in distribs:
        if d is None: continue
        p = np.array(d["probs"])
        p = p[p > 0]
        if len(p) > 0:
            H = -float((p * np.log(p)).sum())
            log_k = math.log(len(p))
            vals.append(log_k - H)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    print("Loading MATH-500...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    use = ds.select(range(min(N_QUESTIONS, len(ds))))
    questions = [r["problem"] for r in use]
    golds = [r["answer"] for r in use]
    print(f"  {len(questions)} problems", flush=True)

    print(f"Loading {MODEL} with logprobs={TOPK}...", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2560,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tok = llm.get_tokenizer()
    base_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_BASE.format(question=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]

    # Greedy with TOP-K logprobs
    print("=== Greedy (logprobs=20) ===", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1536, logprobs=TOPK, seed=SEED)
    t0 = time.time()
    outs = llm.generate(base_prompts, sp)
    dur = time.time() - t0
    print(f"  Generation took {dur:.1f}s", flush=True)

    summary_rows = []
    with OUT_TRACES.open("w") as fw:
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
            o = out.outputs[0]
            text = o.text
            tok_ids = list(o.token_ids)
            tok_strs = [tok.decode([tid]) for tid in tok_ids]
            # Per-token chosen logprob (the actual greedy token)
            chosen_lps = []
            distribs = []  # per token: top-k distribution
            for step_lp in o.logprobs or []:
                if step_lp is None:
                    chosen_lps.append(float("nan"))
                    distribs.append(None)
                    continue
                entries = sorted(step_lp.items(), key=lambda kv: -kv[1].logprob)
                chosen_lps.append(entries[0][1].logprob)
                distribs.append(topk_to_distribution(step_lp))
            boundaries = find_step_boundaries(tok_strs)
            n = len(tok_ids)

            # Per-step compute info-theoretic features
            bdy = boundaries + [n]
            step_features = []
            for sa, sb in zip(bdy[:-1], bdy[1:]):
                step_distribs = distribs[sa:sb]
                step_lps_seg = chosen_lps[sa:sb]
                step_features.append({
                    "entropy": step_entropy(step_distribs),
                    "perplexity": step_perplexity(step_lps_seg),
                    "top1_margin": step_top1_margin(step_distribs),
                    "tempered_kl": step_tempered_kl(step_distribs, T1=1.0, T2=2.0),
                    "top1_conc": step_top1_concentration(step_distribs),
                    "kl_to_uniform": step_kl_to_uniform(step_distribs),
                })

            # Aggregate per-trajectory
            ents = [s["entropy"] for s in step_features if not (s["entropy"] != s["entropy"])]
            kls = [s["tempered_kl"] for s in step_features if not (s["tempered_kl"] != s["tempered_kl"])]
            margs = [s["top1_margin"] for s in step_features if not (s["top1_margin"] != s["top1_margin"])]
            kl_uniforms = [s["kl_to_uniform"] for s in step_features if not (s["kl_to_uniform"] != s["kl_to_uniform"])]

            entropy_p95 = float(np.percentile(ents, 95)) if ents else float("nan")
            entropy_mean = float(np.mean(ents)) if ents else float("nan")
            entropy_max = float(np.max(ents)) if ents else float("nan")
            entropy_min = float(np.min(ents)) if ents else float("nan")
            # Predictive sharpening rate (slope of entropy across step idx; negative slope = decreasing entropy = sharpening)
            sharpening_rate = 0.0
            if len(ents) >= 3:
                x = np.arange(len(ents))
                slope = float(np.polyfit(x, ents, 1)[0])
                sharpening_rate = -slope  # negate so larger = more sharpening
            tempered_kl_max = float(np.max(kls)) if kls else float("nan")
            tempered_kl_mean = float(np.mean(kls)) if kls else float("nan")
            top1_margin_min = float(np.min(margs)) if margs else float("nan")
            top1_margin_mean = float(np.mean(margs)) if margs else float("nan")
            kl_uniform_min = float(np.min(kl_uniforms)) if kl_uniforms else float("nan")
            kl_uniform_mean = float(np.mean(kl_uniforms)) if kl_uniforms else float("nan")

            pred = extract_pred(text)
            ok = int(equal_strict(pred, gold))
            rec = {
                "id": i, "gold": gold, "pred": pred, "correct": ok,
                "n_tokens": n, "n_steps": len(step_features),
                "entropy_p95": entropy_p95,
                "entropy_mean": entropy_mean,
                "entropy_max": entropy_max,
                "entropy_min": entropy_min,
                "sharpening_rate": sharpening_rate,
                "tempered_kl_max": tempered_kl_max,
                "tempered_kl_mean": tempered_kl_mean,
                "top1_margin_min": top1_margin_min,
                "top1_margin_mean": top1_margin_mean,
                "kl_uniform_min": kl_uniform_min,
                "kl_uniform_mean": kl_uniform_mean,
            }
            summary_rows.append(rec)
            fw.write(json.dumps(rec) + "\n")

    print(f"\nProcessed {len(summary_rows)} traces", flush=True)
    correct = np.array([r["correct"] for r in summary_rows])
    print(f"Vanilla acc: {correct.mean():.3f}", flush=True)

    # Spearman correlations
    from scipy.stats import spearmanr
    score_keys = [
        "entropy_p95", "entropy_mean", "entropy_max", "entropy_min",
        "sharpening_rate",
        "tempered_kl_max", "tempered_kl_mean",
        "top1_margin_min", "top1_margin_mean",
        "kl_uniform_min", "kl_uniform_mean",
    ]
    print("\nSpearman correlations:")
    corrs = {}
    for sk in score_keys:
        scores = np.array([r[sk] if not np.isnan(r[sk]) else np.nan for r in summary_rows])
        valid = ~np.isnan(scores)
        if valid.sum() < 10:
            continue
        rho, p = spearmanr(scores[valid], correct[valid])
        corrs[sk] = {"rho": float(rho), "p": float(p), "n": int(valid.sum())}
        # NOTE: for entropy / perplexity / kl_to_uniform: HIGHER usually = LESS confident.
        # We want score to predict correctness POSITIVELY. So negate appropriately when reporting.
        print(f"  {sk:30s} ρ={rho:+.3f}  p={p:.3g}  n={valid.sum()}")

    # CP simulation
    def cp_eval(scores, correct, alpha):
        s = np.array(scores, dtype=float); c = np.array(correct, dtype=int)
        valid = ~np.isnan(s)
        s = s[valid]; c = c[valid]
        if len(s) < 10: return None
        rng = np.random.default_rng(0)
        boot_acc = []
        for _ in range(300):
            idx = rng.integers(0, len(s), size=len(s))
            sb = s[idx]; cb = c[idx]
            for _ in range(5):
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
                    boot_acc.append(float(cb[ti][kept].mean()))
                break
        if not boot_acc: return None
        return {"mean": float(np.mean(boot_acc)),
                "ci95": [float(np.quantile(boot_acc, 0.025)), float(np.quantile(boot_acc, 0.975))]}

    print("\nCP kept_acc at α=0.30:")
    cp_results = {}
    for sk in score_keys:
        scores = np.array([r[sk] for r in summary_rows], dtype=float)
        # If a score is INVERSELY related to correctness (e.g., entropy is higher when wrong),
        # negate to make 'higher = better'.
        rho = corrs.get(sk, {}).get("rho", 0)
        if rho < 0:
            scores = -scores
        cp = cp_eval(scores, correct, 0.30)
        if cp:
            cp_results[sk] = cp
            print(f"  {sk:30s} (rho-corrected) kept_acc={cp['mean']:.3f}  CI=[{cp['ci95'][0]:.3f},{cp['ci95'][1]:.3f}]")

    summary = {
        "model": MODEL, "n": len(summary_rows),
        "vanilla_acc": float(correct.mean()),
        "spearman": corrs,
        "cp_alpha_0.30": cp_results,
        "wallclock_sec": dur,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote: {OUT}", flush=True)


if __name__ == "__main__":
    main()

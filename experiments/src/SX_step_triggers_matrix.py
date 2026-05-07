"""
EXP-VALIDATION-STEP: step-branching with new info-theoretic triggers,
cross-model cross-dataset.

For each (model, dataset, trigger) cell:
1. Greedy decode with logprobs=20
2. Compute per-step features (lp_mean, top1_margin_mean, entropy_mean, tempered_kl_mean, kl_uniform_mean)
3. For each trigger, find worst step under that signal
4. K=4 resample at T=0.7, majority vote
5. Compare accuracy across triggers + against greedy

Env vars:
  MODEL, TAG, DATASETS (comma sep), NQ, K (default 4), T (default 0.7)
"""

import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, equal_strict, normalize

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
TAG = os.environ.get("TAG", "qwen25_7b")
DATASETS = os.environ.get("DATASETS", "math500,aime,olympiad").split(",")
N_PER_DS = int(os.environ.get("NQ", "200"))
K = int(os.environ.get("K", "4"))
T = float(os.environ.get("T", "0.7"))
TOPK = 20
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / f"SX_step_triggers_{TAG}.json"

PROMPT_MATH = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\nProblem: {question}\n\n"
)


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
    else:
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
    return {"probs": probs.tolist(), "lps": lps.tolist()}


def js_div(p, q):
    p = np.asarray(p); q = np.asarray(q)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float((a[mask] * np.log(a[mask] / b[mask])).sum()) if mask.sum() else 0.0
    return 0.5 * (_kl(p, m) + _kl(q, m))


def step_features(distribs, token_lps_seg):
    if not distribs:
        return {k: float("nan") for k in ["lp", "ent", "marg", "tkl", "klu"]}
    ents, margs, kls, klus = [], [], [], []
    for d in distribs:
        if d is None: continue
        p = np.array(d["probs"]); p = p[p > 0]
        if len(p) > 0:
            H = -float((p * np.log(p)).sum())
            ents.append(H)
            klus.append(math.log(len(p)) - H)
        if len(d["lps"]) >= 2:
            margs.append(d["lps"][0] - d["lps"][1])
        lps = np.asarray(d["lps"])
        p1 = np.exp(lps / 1.0 - lps.max()); p1 = p1 / p1.sum()
        p2 = np.exp(lps / 2.0 - (lps / 2.0).max()); p2 = p2 / p2.sum()
        kls.append(js_div(p1, p2))
    valid_lp = [lp for lp in token_lps_seg if not (lp != lp)]
    return {
        "lp": float(np.mean(valid_lp)) if valid_lp else float("nan"),
        "ent": float(np.mean(ents)) if ents else float("nan"),
        "marg": float(np.mean(margs)) if margs else float("nan"),
        "tkl": float(np.mean(kls)) if kls else float("nan"),
        "klu": float(np.mean(klus)) if klus else float("nan"),
    }


def main():
    print(f"=== Step triggers: {MODEL} on {DATASETS} ===", flush=True)
    print(f"  K={K}, T={T}, n_per_ds={N_PER_DS}", flush=True)

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

    all_results = {"model": MODEL, "tag": TAG, "K": K, "T": T, "datasets": {}}

    triggers = {
        "lp_min": ("lp", "argmin"),
        "top1_margin_min": ("marg", "argmin"),
        "entropy_max": ("ent", "argmax"),
        "tempered_kl_max": ("tkl", "argmax"),
        "kl_uniform_min": ("klu", "argmin"),
    }

    for ds_name in DATASETS:
        ds_name = ds_name.strip()
        if not ds_name: continue
        print(f"\n--- Dataset: {ds_name} ---", flush=True)
        try:
            ds = load_dataset_uniform(ds_name, N_PER_DS)
        except Exception as e:
            print(f"  load failed: {e}", flush=True)
            continue
        prompts_chat = [
            tok.apply_chat_template(
                [{"role": "user", "content": PROMPT_MATH.format(question=r["q"])}],
                tokenize=False, add_generation_prompt=True,
            )
            for r in ds
        ]
        max_tokens = 1536 if ds_name == "math500" else 2048

        # Greedy with logprobs=20
        print(f"  greedy decode...", flush=True)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=TOPK, seed=SEED)
        t0 = time.time()
        outs = llm.generate(prompts_chat, sp)
        gen_time = time.time() - t0

        greedy = []
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
            sfeats = []
            for sa, sb in zip(bdy[:-1], bdy[1:]):
                sfeats.append(step_features(distribs[sa:sb], chosen_lps[sa:sb]))
            pred = extract_pred(text)
            ok = int(equal_strict(pred, rec["gold"]))
            greedy.append({
                "id": i, "tok_ids": tok_ids, "boundaries": boundaries,
                "step_features": sfeats, "text": text, "pred": pred, "gold": rec["gold"], "correct": ok,
            })
        n_correct = sum(r["correct"] for r in greedy)
        vanilla = n_correct / len(greedy) if greedy else 0
        print(f"  greedy: {n_correct}/{len(greedy)} = {vanilla:.3f} in {gen_time:.0f}s", flush=True)

        ds_res = {"vanilla_acc": vanilla, "n": len(greedy), "greedy_time": gen_time, "triggers": {}}

        for trig_name, (feat_key, op) in triggers.items():
            branch_prompts = []; ids = []
            for r in greedy:
                feats = [s.get(feat_key, float("nan")) for s in r["step_features"]]
                valid = [(j, f) for j, f in enumerate(feats) if not (f != f)]
                if not valid: continue
                if op == "argmin":
                    idx = min(valid, key=lambda kv: kv[1])[0]
                else:
                    idx = max(valid, key=lambda kv: kv[1])[0]
                if idx >= len(r["boundaries"]): continue
                bdy_pos = r["boundaries"][idx]
                prefix = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
                branch_prompts.append(prompts_chat[r["id"]] + prefix)
                ids.append(r["id"])
            if not branch_prompts:
                continue
            sp_b = SamplingParams(n=K, temperature=T, top_p=0.95, max_tokens=1024, seed=SEED + 100)
            t0 = time.time()
            outs_b = llm.generate(branch_prompts, sp_b)
            dur = time.time() - t0
            n_correct_b = 0
            for ii, ob in zip(ids, outs_b):
                r = greedy[ii]
                alt_preds = [extract_pred(c.text) for c in ob.outputs]
                preds_clean = [p for p in alt_preds if p is not None]
                combined = [r["pred"]] + preds_clean
                counter = Counter([str(normalize(p)) for p in combined if p is not None])
                top, _ = counter.most_common(1)[0] if counter else (None, 0)
                n_correct_b += int(equal_strict(top, r["gold"]))
            acc = n_correct_b / len(ids)
            ds_res["triggers"][trig_name] = {
                "acc": acc, "delta": acc - vanilla, "n": len(ids), "wallclock": dur,
            }
            print(f"  {trig_name:18s} acc={acc:.3f}  Δ={acc-vanilla:+.3f}  n={len(ids)}  wall={dur:.0f}s", flush=True)

        all_results["datasets"][ds_name] = ds_res
        OUT.write_text(json.dumps(all_results, indent=2))

    print(f"\n=== Done. Wrote {OUT} ===", flush=True)


if __name__ == "__main__":
    main()

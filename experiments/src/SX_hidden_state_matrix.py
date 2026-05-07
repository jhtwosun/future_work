"""
EXP-VALIDATION-HS: hidden_state cross-model cross-dataset validation.

Loads ONE model via HF transformers, runs over multiple datasets with
output_hidden_states=True, computes hidden_drift_*, hidden_norm_*,
layer_disagreement_* per trajectory, then bootstrap CP.

Env vars:
  MODEL: HF model ID
  TAG:   short tag for output files
  DATASETS: comma-sep list, default 'math500,olympiad'
  NQ:    per-dataset sample size (default 100)
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, equal_strict, normalize

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
TAG = os.environ.get("TAG", "qwen25_7b")
DATASETS = os.environ.get("DATASETS", "math500,olympiad").split(",")
N_PER_DS = int(os.environ.get("NQ", "100"))
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / f"SX_hidden_validation_{TAG}.json"

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


def load_dataset_uniform(name, n):
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["problem"], "gold": r["answer"], "type": "math"} for r in ds]
    elif name == "aime":
        ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["Question"], "gold": str(r["Answer"]), "type": "math"} for r in ds]
    elif name == "olympiad":
        ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["question"], "gold": str(r["final_answer"][0]) if r["final_answer"] else "", "type": "math"} for r in ds]
    elif name == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        stem = {"math", "physics", "chemistry", "biology", "computer science", "engineering"}
        ds = ds.filter(lambda r: r["category"] in stem)
        ds = ds.select(range(min(n, len(ds))))
        return [{"q": r["question"], "options": r["options"], "gold": r["answer"], "type": "mmlu"} for r in ds]
    else:
        raise ValueError(name)


def make_prompt_text(rec):
    if rec["type"] == "math":
        return PROMPT_MATH.format(question=rec["q"])
    elif rec["type"] == "mmlu":
        opts = "\n".join([f"  {chr(65+i)}. {o}" for i, o in enumerate(rec["options"])])
        return PROMPT_MMLU.format(question=rec["q"], options=opts)
    raise ValueError(rec["type"])


def find_step_boundaries(decoded_tokens):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(decoded_tokens):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(decoded_tokens):
            boundaries.append(i + 1)
    return boundaries


def cos_dist_sequence(states):
    states_n = states / (states.norm(dim=-1, keepdim=True) + 1e-9)
    if states_n.shape[0] < 2:
        return []
    diffs = []
    for t in range(1, states_n.shape[0]):
        sim = float((states_n[t] * states_n[t-1]).sum())
        diffs.append(1.0 - sim)
    return diffs


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
    print(f"=== HS validation: {MODEL} on {DATASETS} ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s, {model.config.num_hidden_layers} layers", flush=True)

    all_results = {"model": MODEL, "tag": TAG, "datasets": {}}
    score_keys = [
        "hidden_drift_max_last", "hidden_drift_mean_last",
        "hidden_drift_max_penult", "hidden_drift_mean_penult",
        "hidden_norm_var", "hidden_norm_range",
    ]

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
        rows = []
        n_correct = 0
        t_ds = time.time()
        for qi, rec in enumerate(ds):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": make_prompt_text(rec)}],
                tokenize=False, add_generation_prompt=True,
            )
            input_ids = tok.encode(chat, return_tensors="pt").to(model.device)
            input_len = input_ids.shape[1]
            with torch.inference_mode():
                gen = model.generate(
                    input_ids,
                    max_new_tokens=1024,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    pad_token_id=tok.eos_token_id,
                )
            generated_ids = gen.sequences[0, input_len:].cpu().tolist()
            n_gen = len(generated_ids)
            if n_gen == 0:
                continue
            try:
                last_hidden = []; penult_hidden = []
                for step_hs in gen.hidden_states:
                    h_last = step_hs[-1][:, -1, :].squeeze(0).float().cpu()
                    last_hidden.append(h_last)
                    h_pen = (step_hs[-2] if len(step_hs) >= 2 else step_hs[-1])[:, -1, :].squeeze(0).float().cpu()
                    penult_hidden.append(h_pen)
                last_hidden = torch.stack(last_hidden, dim=0)
                penult_hidden = torch.stack(penult_hidden, dim=0)
            except Exception as e:
                continue

            decoded_tokens = [tok.decode([tid]) for tid in generated_ids]
            boundaries = find_step_boundaries(decoded_tokens)
            bdy = boundaries + [n_gen]
            step_last = []; step_penult = []
            for sa, sb in zip(bdy[:-1], bdy[1:]):
                if sb - 1 < n_gen:
                    step_last.append(last_hidden[sb - 1])
                    step_penult.append(penult_hidden[sb - 1])
            if len(step_last) < 2:
                continue
            step_last = torch.stack(step_last)
            step_penult = torch.stack(step_penult)
            drifts_last = cos_dist_sequence(step_last)
            drifts_penult = cos_dist_sequence(step_penult)
            norms_last = step_last.norm(dim=-1).cpu().numpy()

            text = tok.decode(generated_ids, skip_special_tokens=True)
            pred = extract_pred(text)
            if rec["type"] == "mmlu":
                gold_letter = chr(65 + rec["gold"]) if isinstance(rec["gold"], int) else str(rec["gold"]).strip()[:1]
                ok = int((pred or "").strip().upper()[:1] == gold_letter[:1].upper())
            else:
                ok = int(equal_strict(pred, rec["gold"]))
            n_correct += ok

            rows.append({
                "id": qi, "correct": ok, "n_steps": len(step_last),
                "hidden_drift_max_last": float(np.max(drifts_last)) if drifts_last else float("nan"),
                "hidden_drift_mean_last": float(np.mean(drifts_last)) if drifts_last else float("nan"),
                "hidden_drift_max_penult": float(np.max(drifts_penult)) if drifts_penult else float("nan"),
                "hidden_drift_mean_penult": float(np.mean(drifts_penult)) if drifts_penult else float("nan"),
                "hidden_norm_var": float(norms_last.var()),
                "hidden_norm_range": float(norms_last.max() - norms_last.min()),
            })
            if (qi + 1) % 20 == 0:
                elapsed = time.time() - t_ds
                eta = elapsed / (qi + 1) * (len(ds) - qi - 1)
                print(f"  {qi+1}/{len(ds)} acc={n_correct/(len(rows) or 1):.3f} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        gen_time = time.time() - t_ds
        vanilla_acc = n_correct / len(rows) if rows else 0
        print(f"\n  {ds_name}: vanilla={vanilla_acc:.3f} ({n_correct}/{len(rows)}) in {gen_time:.0f}s", flush=True)

        # Spearman + CP
        from scipy.stats import spearmanr
        correct = np.array([r["correct"] for r in rows])
        ds_results = {"vanilla_acc": vanilla_acc, "n": len(rows), "gen_time_sec": gen_time}
        for sk in score_keys:
            scores = np.array([r.get(sk, float("nan")) for r in rows], dtype=float)
            valid = ~np.isnan(scores)
            if valid.sum() < 20: continue
            rho, p = spearmanr(scores[valid], correct[valid])
            scores_dir = scores * (1 if rho >= 0 else -1)
            cp03 = cp_eval_with_ci(scores_dir, correct, 0.30)
            ds_results[sk] = {"rho": float(rho), "p": float(p), "kept_acc_alpha_0.30": cp03}
            if cp03:
                print(f"  {sk:30s} ρ={rho:+.3f} p={p:.3g} kept@0.3={cp03['kept_acc']:.3f} CI=[{cp03['ci95'][0]:.3f},{cp03['ci95'][1]:.3f}]", flush=True)
        all_results["datasets"][ds_name] = ds_results
        OUT.write_text(json.dumps(all_results, indent=2))

    print(f"\n=== Done. Wrote {OUT} ===", flush=True)


if __name__ == "__main__":
    main()

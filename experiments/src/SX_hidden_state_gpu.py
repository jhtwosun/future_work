"""
EXP-S3: Hidden state scores via HF transformers + output_hidden_states=True.

Implements Wave 1 Agent B's:
- hidden_drift_max: max step-to-step cosine distance on penultimate residual
- hidden_drift_mean: mean step-to-step cosine distance
- hidden_norm_var: variance of last-hidden-state norms across steps
- layer_consistency_disagreement: JS divergence between Logit-Lens distributions across last 4 layers (cheap proxy: cos similarity of last 4 layers' final-token logits projection)
- per_step_hidden_norm: trajectory-level features

Uses HF AutoModelForCausalLM, NOT vLLM (because we need hidden states).
Uses GPU 0 OR 1 (CUDA_VISIBLE_DEVICES env var).

Setup: greedy decode 200 MATH-500 problems with output_hidden_states. Re-extract
boundary positions (using \\n\\n as step delimiter on the decoded text).

Output: SX_hidden_state.json + per-trace JSONL.
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
from robust_eval import extract_pred, equal_strict

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_QUESTIONS = int(os.environ.get("NQ", "100"))  # smaller because HF is slower
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
OUT = OUTDIR / "SX_hidden_state.json"
OUT_TRACES = OUTDIR / "SX_hidden_state_traces.jsonl"

PROMPT_BASE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
)


def find_step_boundaries(decoded_tokens):
    """Find positions where '\\n\\n' creates step boundaries.
    Returns list of token indices where a new step begins."""
    boundaries = [0]
    accum = ""
    for i, t in enumerate(decoded_tokens):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(decoded_tokens):
            boundaries.append(i + 1)
    return boundaries


def main():
    print("Loading MATH-500...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    use = ds.select(range(min(N_QUESTIONS, len(ds))))
    questions = [r["problem"] for r in use]
    golds = [r["answer"] for r in use]
    print(f"  {len(questions)} problems", flush=True)

    print(f"Loading {MODEL} via HF transformers...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  model has {model.config.num_hidden_layers} layers", flush=True)
    n_layers = model.config.num_hidden_layers

    summary_rows = []
    n_correct = 0
    n_processed = 0
    t_start = time.time()

    with OUT_TRACES.open("w") as fw:
        for qi, (q, gold) in enumerate(zip(questions, golds)):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": PROMPT_BASE.format(question=q)}],
                tokenize=False, add_generation_prompt=True,
            )
            input_ids = tok.encode(chat, return_tensors="pt").to(model.device)
            input_len = input_ids.shape[1]

            # Greedy decode (one token at a time, manually, capturing hidden states at each step)
            # Using model.generate with output_hidden_states=True returns hidden states per generated token
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

            # gen.hidden_states is a tuple of length n_gen, each a tuple of (n_layers+1) tensors
            # Each tensor has shape (1, current_seq_len_so_far, hidden_dim) for that step.
            # The first call has the full prompt; subsequent calls have only 1 new token.
            # We want the last hidden state for each generated token: gen.hidden_states[step][-1][:, -1, :]
            try:
                # last_hidden[t] has shape (1, hidden_dim) — the residual for the t-th generated token at the last layer
                last_hidden = []
                for step_hs in gen.hidden_states:
                    # step_hs[-1] is last layer; take last token's hidden state
                    h = step_hs[-1][:, -1, :].squeeze(0).float().cpu()
                    last_hidden.append(h)
                last_hidden = torch.stack(last_hidden, dim=0)  # (n_gen, hidden_dim)

                # Also get penultimate layer for hidden_drift_max
                penult_hidden = []
                for step_hs in gen.hidden_states:
                    if len(step_hs) >= 2:
                        h = step_hs[-2][:, -1, :].squeeze(0).float().cpu()
                    else:
                        h = step_hs[-1][:, -1, :].squeeze(0).float().cpu()
                    penult_hidden.append(h)
                penult_hidden = torch.stack(penult_hidden, dim=0)
            except Exception as e:
                print(f"  q{qi}: hidden state extraction failed: {e}", flush=True)
                continue

            # Decode tokens to find step boundaries
            decoded_tokens = [tok.decode([tid]) for tid in generated_ids]
            boundaries = find_step_boundaries(decoded_tokens)

            # Compute per-step features
            # Use the LAST token in each step as the step's representative hidden state
            bdy = boundaries + [n_gen]
            step_last_hidden = []  # last layer
            step_penult_hidden = []  # penultimate
            for sa, sb in zip(bdy[:-1], bdy[1:]):
                if sb - 1 < n_gen:
                    step_last_hidden.append(last_hidden[sb - 1])
                    step_penult_hidden.append(penult_hidden[sb - 1])

            if len(step_last_hidden) < 2:
                continue
            step_last_hidden = torch.stack(step_last_hidden)
            step_penult_hidden = torch.stack(step_penult_hidden)

            # hidden_drift: cosine distance between consecutive step end-hidden states
            def cos_dist_sequence(states):
                # states: (T, D)
                states_n = states / (states.norm(dim=-1, keepdim=True) + 1e-9)
                if states_n.shape[0] < 2:
                    return []
                diffs = []
                for t in range(1, states_n.shape[0]):
                    sim = float((states_n[t] * states_n[t-1]).sum())
                    diffs.append(1.0 - sim)
                return diffs

            drifts_last = cos_dist_sequence(step_last_hidden)
            drifts_penult = cos_dist_sequence(step_penult_hidden)

            hidden_drift_max_last = float(np.max(drifts_last)) if drifts_last else float("nan")
            hidden_drift_mean_last = float(np.mean(drifts_last)) if drifts_last else float("nan")
            hidden_drift_max_penult = float(np.max(drifts_penult)) if drifts_penult else float("nan")
            hidden_drift_mean_penult = float(np.mean(drifts_penult)) if drifts_penult else float("nan")

            # Hidden norm variance across steps
            norms_last = step_last_hidden.norm(dim=-1).cpu().numpy()
            hidden_norm_var = float(norms_last.var())
            hidden_norm_max = float(norms_last.max())
            hidden_norm_min = float(norms_last.min())
            hidden_norm_range = hidden_norm_max - hidden_norm_min

            # Layer consistency: get logit-lens projections at last 4 layers for the LAST generated token
            # gen.hidden_states[-1] = (n_layers+1, ...) for the last step
            last_step_hs = gen.hidden_states[-1]  # tuple of (n_layers+1)
            lm_head = model.get_output_embeddings()  # final LM head
            with torch.inference_mode():
                last_4_layer_logits = []
                for layer_idx in [-1, -2, -3, -4]:
                    if abs(layer_idx) > len(last_step_hs):
                        break
                    h = last_step_hs[layer_idx][:, -1, :].float()  # (1, hidden)
                    logits = lm_head(h.to(model.dtype).to(model.device)).float().cpu()  # (1, vocab)
                    last_4_layer_logits.append(logits.squeeze(0))
            # JS divergence between last layer's distribution and each of the 3 earlier layers
            layer_disagreement_max = float("nan")
            layer_disagreement_mean = float("nan")
            if len(last_4_layer_logits) >= 2:
                # softmax over top-100 to keep memory low
                def topk_dist(logits, k=100):
                    vals, idx = torch.topk(logits, k=min(k, logits.shape[-1]))
                    p = torch.softmax(vals, dim=-1)
                    return p.numpy(), idx.numpy()
                p_last, idx_last = topk_dist(last_4_layer_logits[0])
                disagrees = []
                for layer_logits in last_4_layer_logits[1:]:
                    p_l, idx_l = topk_dist(layer_logits)
                    # Project both onto union of token indices for proper comparison
                    union = np.union1d(idx_last, idx_l)
                    # Build aligned distributions
                    p_a = np.zeros(len(union))
                    p_b = np.zeros(len(union))
                    for j, t_id in enumerate(union):
                        if t_id in idx_last:
                            p_a[j] = p_last[np.where(idx_last == t_id)[0][0]]
                        if t_id in idx_l:
                            p_b[j] = p_l[np.where(idx_l == t_id)[0][0]]
                    p_a = p_a / (p_a.sum() + 1e-9)
                    p_b = p_b / (p_b.sum() + 1e-9)
                    m = 0.5 * (p_a + p_b)
                    def kl(p, q):
                        mask = (p > 0) & (q > 0)
                        return float((p[mask] * np.log(p[mask] / q[mask])).sum()) if mask.sum() > 0 else 0.0
                    js = 0.5 * (kl(p_a, m) + kl(p_b, m))
                    disagrees.append(js)
                layer_disagreement_max = float(np.max(disagrees)) if disagrees else float("nan")
                layer_disagreement_mean = float(np.mean(disagrees)) if disagrees else float("nan")

            # Decode and check correctness
            text = tok.decode(generated_ids, skip_special_tokens=True)
            pred = extract_pred(text)
            ok = int(equal_strict(pred, gold))
            n_correct += ok
            n_processed += 1

            rec = {
                "id": qi, "gold": gold, "pred": pred, "correct": ok,
                "n_tokens": n_gen,
                "n_steps": len(boundaries),
                "hidden_drift_max_last": hidden_drift_max_last,
                "hidden_drift_mean_last": hidden_drift_mean_last,
                "hidden_drift_max_penult": hidden_drift_max_penult,
                "hidden_drift_mean_penult": hidden_drift_mean_penult,
                "hidden_norm_var": hidden_norm_var,
                "hidden_norm_max": hidden_norm_max,
                "hidden_norm_min": hidden_norm_min,
                "hidden_norm_range": hidden_norm_range,
                "layer_disagreement_max": layer_disagreement_max,
                "layer_disagreement_mean": layer_disagreement_mean,
            }
            summary_rows.append(rec)
            fw.write(json.dumps(rec) + "\n")
            if (qi + 1) % 20 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (qi + 1) * (len(questions) - qi - 1)
                print(f"  {qi+1}/{len(questions)} done, acc so far={n_correct/n_processed:.3f}, elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"\nTotal processed: {n_processed}, correct: {n_correct} ({n_correct/n_processed:.3f})", flush=True)
    correct = np.array([r["correct"] for r in summary_rows])
    from scipy.stats import spearmanr

    print("\nSpearman correlations:")
    score_keys = [
        "hidden_drift_max_last", "hidden_drift_mean_last",
        "hidden_drift_max_penult", "hidden_drift_mean_penult",
        "hidden_norm_var", "hidden_norm_range",
        "layer_disagreement_max", "layer_disagreement_mean",
    ]
    corrs = {}
    for sk in score_keys:
        scores = np.array([r[sk] if not np.isnan(r[sk]) else np.nan for r in summary_rows])
        valid = ~np.isnan(scores)
        if valid.sum() < 10:
            continue
        rho, p = spearmanr(scores[valid], correct[valid])
        corrs[sk] = {"rho": float(rho), "p": float(p), "n": int(valid.sum())}
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
        rho = corrs.get(sk, {}).get("rho", 0)
        if rho < 0:
            scores = -scores
        cp = cp_eval(scores, correct, 0.30)
        if cp:
            cp_results[sk] = cp
            print(f"  {sk:30s} kept_acc={cp['mean']:.3f}  CI=[{cp['ci95'][0]:.3f},{cp['ci95'][1]:.3f}]")

    summary = {"model": MODEL, "n": len(summary_rows),
                "vanilla_acc": float(correct.mean()),
                "spearman": corrs,
                "cp_alpha_0.30": cp_results}
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote: {OUT}", flush=True)


if __name__ == "__main__":
    main()

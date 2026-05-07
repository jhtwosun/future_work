"""
EXP-step4: PRM-guided step branching aggregation.

Hypothesis: at worst step, K alternatives have varying quality. Majority vote
gives equal weight to each, but using PRM to score and select the best
should be stronger.

Test variants:
- K=4 + majority vote (= existing PilotC)
- K=4 + PRM-selected (highest PRM-min over alternatives)
- K=4 + lp-selected (highest mean lp over alternatives)
- K=4 + best-of-N criterion: keep alt only if its PRM > original step's PRM

Also test K=8 for richer comparison.

Uses Qwen2.5-Math-PRM-7B for scoring.
"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from robust_eval import extract_pred, normalize, equal_strict

LM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
N_QUESTIONS = int(os.environ.get("NQ", "200"))
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
SUM = OUTDIR / "SX_prm_guided_branching.json"

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


def per_step_lps(token_logprobs, boundaries):
    bdy = boundaries + [len(token_logprobs)]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            out.append(float(np.mean(seg)))
    return out


def make_step_rewards(logits, token_masks):
    probs = F.softmax(logits, dim=-1)
    probs = probs * token_masks.unsqueeze(-1)
    out = []
    for i in range(probs.size(0)):
        sample = probs[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        out.append(positive_probs.cpu().tolist())
    return out


def score_with_prm(prm_model, prm_tok, question, full_text, sep_id, max_steps=50):
    """Get PRM step rewards for a (question, reasoning text) pair."""
    steps = re.split(r"\n\s*\n+", full_text.strip())
    if not steps: return []
    if len(steps) > max_steps:
        half = max_steps // 2
        steps = steps[:half] + steps[-half:]
    SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
    assistant_str = "<extra_0>".join(steps) + "<extra_0>"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_str},
    ]
    conv = prm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = prm_tok.encode(conv, return_tensors="pt").to(prm_model.device)
    if input_ids.shape[1] > 4096:
        input_ids = input_ids[:, -4096:]
    with torch.inference_mode():
        outputs = prm_model(input_ids=input_ids, use_cache=False)
    logits = outputs[0]
    mask = (input_ids == sep_id)
    rewards = make_step_rewards(logits, mask)[0]
    if len(rewards) > len(steps):
        rewards = rewards[:len(steps)]
    elif len(rewards) < len(steps):
        m = float(np.mean(rewards)) if rewards else 0.5
        rewards = rewards + [m] * (len(steps) - len(rewards))
    return rewards


def main():
    print("Loading MATH-500...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    use = ds.select(range(min(N_QUESTIONS, len(ds))))
    questions = [r["problem"] for r in use]
    golds = [r["answer"] for r in use]

    # Load LM (vLLM, GPU 0)
    print(f"Loading {LM_MODEL}...", flush=True)
    llm = LLM(
        model=LM_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.50,  # leave room for PRM
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

    # Greedy
    print("\n=== Greedy ===", flush=True)
    sp_g = SamplingParams(temperature=0.0, max_tokens=1536, logprobs=1, seed=SEED)
    outs = llm.generate(base_prompts, sp_g)
    greedy_records = []
    for i, out in enumerate(outs):
        o = out.outputs[0]
        text = o.text
        tok_ids = list(o.token_ids)
        tok_strs = [tok.decode([tid]) for tid in tok_ids]
        chosen_lps = []
        for step_lp in o.logprobs or []:
            if step_lp is None:
                chosen_lps.append(float("nan")); continue
            entries = sorted(step_lp.items(), key=lambda kv: -kv[1].logprob)
            chosen_lps.append(entries[0][1].logprob)
        boundaries = find_step_boundaries(tok_strs)
        steps = per_step_lps(chosen_lps, boundaries)
        pred = extract_pred(text)
        ok = int(equal_strict(pred, golds[i]))
        greedy_records.append({
            "id": i, "gold": golds[i], "pred": pred, "correct": ok,
            "tok_ids": tok_ids, "boundaries": boundaries, "step_lps": steps,
            "text": text,
        })
    n_greedy = sum(r["correct"] for r in greedy_records)
    print(f"  Greedy: {n_greedy}/{len(questions)} = {n_greedy/len(questions):.3f}", flush=True)

    # K-resample at lp_min worst step (K=4 and K=8)
    summary = {"greedy_acc": n_greedy / len(questions)}
    for K in [4, 8]:
        print(f"\n=== K={K} resample at lp_min worst step ===", flush=True)
        branch_prompts = []
        ids = []
        for r in greedy_records:
            if not r["step_lps"]: continue
            idx = int(np.argmin(r["step_lps"]))
            if idx >= len(r["boundaries"]): continue
            bdy_pos = r["boundaries"][idx]
            prefix_text = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
            branch_prompts.append(base_prompts[r["id"]] + prefix_text)
            ids.append(r["id"])
        sp = SamplingParams(n=K, temperature=0.7, top_p=0.95, max_tokens=1024, seed=SEED + 100 + K)
        t0 = time.time()
        outs = llm.generate(branch_prompts, sp)
        dur = time.time() - t0

        # Store all alternative texts for later PRM scoring
        alt_texts = {ii: [c.text for c in out.outputs] for ii, out in zip(ids, outs)}
        alt_preds = {ii: [extract_pred(c.text) for c in out.outputs] for ii, out in zip(ids, outs)}
        # Also keep the FULL trace = original prefix + alt continuation for PRM scoring
        alt_full = {ii: [tok.decode(greedy_records[ii]["tok_ids"][:greedy_records[ii]["boundaries"][int(np.argmin(greedy_records[ii]["step_lps"]))]], skip_special_tokens=True) + c.text for c in out.outputs] for ii, out in zip(ids, outs)}

        # Aggregator 1: majority vote (existing)
        n_majority = 0
        for ii in ids:
            r = greedy_records[ii]
            preds_clean = [p for p in alt_preds[ii] if p is not None]
            combined = [r["pred"]] + preds_clean
            counter = Counter([str(normalize(p)) for p in combined if p is not None])
            top, _ = counter.most_common(1)[0] if counter else (None, 0)
            n_majority += int(equal_strict(top, r["gold"]))

        summary[f"pilotC_K{K}_majority"] = n_majority / max(len(ids), 1)

        # Save alt data for PRM scoring
        summary[f"K{K}_n_alt"] = len(ids)
        np.save(OUTDIR / f"SX_alt_K{K}_data.npy",
                  {"ids": ids, "alt_texts": alt_texts, "alt_preds": alt_preds, "alt_full": alt_full})

        print(f"  K={K} majority acc: {n_majority/max(len(ids),1):.3f} ({dur:.1f}s)", flush=True)

    # Save greedy_records data for PRM phase
    greedy_data = []
    for r in greedy_records:
        greedy_data.append({
            "id": r["id"], "gold": r["gold"], "pred": r["pred"], "correct": r["correct"],
            "text": r["text"], "step_lps": r["step_lps"], "boundaries": r["boundaries"],
        })
    (OUTDIR / "SX_greedy_data.json").write_text(json.dumps(greedy_data, indent=2))
    SUM.write_text(json.dumps(summary, indent=2))

    print(f"\nSaved greedy + alt data. Now releasing LM and loading PRM for selection...", flush=True)
    # Release LM memory
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Load PRM
    print(f"Loading PRM {PRM_MODEL}...", flush=True)
    prm_tok = AutoTokenizer.from_pretrained(PRM_MODEL, trust_remote_code=True)
    prm_model = AutoModel.from_pretrained(
        PRM_MODEL,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    sep_id = prm_tok.encode("<extra_0>")[0]
    print(f"  PRM loaded. step_sep_id={sep_id}", flush=True)

    # PRM-guided aggregation
    for K in [4, 8]:
        data = np.load(OUTDIR / f"SX_alt_K{K}_data.npy", allow_pickle=True).item()
        ids = data["ids"]
        alt_full = data["alt_full"]
        alt_preds = data["alt_preds"]

        n_prm_selected = 0
        for i, ii in enumerate(ids):
            r = greedy_records[ii]
            # Score the original full trace + each K alternative
            orig_rewards = score_with_prm(prm_model, prm_tok, r["text"][:200], r["text"], sep_id)
            orig_prm_min = float(np.min(orig_rewards)) if orig_rewards else 0.5

            alt_prm_mins = []
            for k in range(K):
                alt_text = alt_full[ii][k]
                rew = score_with_prm(prm_model, prm_tok, r["text"][:200], alt_text, sep_id)
                alt_prm_mins.append(float(np.min(rew)) if rew else 0.5)

            # Pick the one with highest PRM-min (including original)
            all_prm_mins = [orig_prm_min] + alt_prm_mins
            all_preds = [r["pred"]] + alt_preds[ii]
            best_idx = int(np.argmax(all_prm_mins))
            best_pred = all_preds[best_idx]
            ok = int(equal_strict(best_pred, r["gold"]))
            n_prm_selected += ok
            if (i + 1) % 50 == 0:
                print(f"  K={K} PRM-sel {i+1}/{len(ids)}: running acc={n_prm_selected/(i+1):.3f}", flush=True)

        summary[f"pilotC_K{K}_prm_selected"] = n_prm_selected / max(len(ids), 1)
        print(f"\nK={K} PRM-selected acc: {n_prm_selected/max(len(ids),1):.3f}", flush=True)

    SUM.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")


if __name__ == "__main__":
    main()

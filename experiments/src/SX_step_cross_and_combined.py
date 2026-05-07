"""
EXP-step3: Cross-dataset replication + combined paraphrase + step-branching.

Goals:
1. Replicate pilotC_lp_min_K4_T0.7 (+4.5pp on MATH-500) on:
   - AIME 1983-2024 (200)
   - OlympiadBench (200)
   - MMLU-Pro STEM (300)
2. Combined method: para_M3 modal vote + (if low consensus) → step-branching K=4
3. Compare to: greedy, pure step branching, pure paraphrase, SC@4

This addresses: cross-dataset generalization + interaction effects.
"""

import json
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
from robust_eval import extract_pred, normalize, equal_strict

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 0

OUTDIR = Path("/home/nvidia/future/experiments/results")
SUM = OUTDIR / "SX_step_cross_combined.json"

PROMPT_BASE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>" '
    "or in \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
)
PARAPHRASES = [
    "Determine the answer to the following math question. Show your reasoning step-by-step, then state the final answer clearly with \\boxed{{}}.\n\n{question}\n",
    "Math problem to solve: {question}\nWalk through your solution carefully, with each reasoning step on its own line. Provide the final numerical or expression answer in \\boxed{{}} at the end.\n",
    "Please carefully work through this mathematics problem and arrive at the answer.\n\n{question}\n\nUse step-by-step reasoning and put the final answer in \\boxed{{}}.\n",
]
PROMPT_MMLU = (
    "Answer the following multiple-choice question. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, put your final answer letter (A-J) within \\boxed{{}}.\n\n'
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
)


def per_step_lps(token_logprobs, boundaries):
    bdy = boundaries + [len(token_logprobs)]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            out.append(float(np.mean(seg)))
    return out


def find_step_boundaries(token_strs):
    boundaries = [0]
    accum = ""
    for i, t in enumerate(token_strs):
        accum += t
        if accum.endswith("\n\n") and i + 1 < len(token_strs):
            boundaries.append(i + 1)
    return boundaries


def extract_letter(text):
    if not text: return None
    boxed = re.findall(r"\\boxed\{([A-J])\}", text)
    if boxed: return boxed[-1]
    boxed_full = re.findall(r"\\boxed\{(?:Option |Answer )?\(?([A-J])\)?[^}]*\}", text)
    if boxed_full: return boxed_full[-1]
    ans = re.findall(r"(?i)\bAnswer\s*[:\-=]?\s*\(?([A-J])\)?\b", text)
    if ans: return ans[-1]
    final = re.findall(r"(?i)\b(?:final\s+answer|the\s+answer\s+is)\s*[:\-=]?\s*\(?([A-J])\)?\b", text)
    if final: return final[-1]
    return None


def is_correct(pred, gold, mcq=False):
    if mcq: return pred == gold
    return equal_strict(pred, gold)


def run_dataset(llm, tok, name, questions, golds, prompt_base, paraphrases, max_tok=1536, mcq=False, options=None):
    print(f"\n=== {name} (n={len(questions)}) ===", flush=True)
    extract_fn = extract_letter if mcq else extract_pred
    base_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": prompt_base.format(question=q, options=opt) if mcq else prompt_base.format(question=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        for q, opt in zip(questions, options or [None]*len(questions))
    ]

    # Greedy with logprobs
    sp = SamplingParams(temperature=0.0, max_tokens=max_tok, logprobs=5, seed=SEED)
    outs = llm.generate(base_prompts, sp)
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
        pred = extract_fn(text)
        ok = int(is_correct(pred, golds[i], mcq))
        greedy_records.append({
            "id": i, "gold": golds[i], "pred": pred, "correct": ok,
            "tok_ids": tok_ids, "boundaries": boundaries, "step_lps": steps,
            "text": text,
        })
    greedy_acc = float(np.mean([r["correct"] for r in greedy_records]))
    print(f"  Greedy: {sum(r['correct'] for r in greedy_records)}/{len(questions)} = {greedy_acc:.3f}", flush=True)

    # Pilot C lp_min K=4 T=0.7 (best config)
    branch_prompts = []
    ids = []
    for r in greedy_records:
        if not r["step_lps"] or not r["boundaries"]:
            continue
        idx = int(np.argmin(r["step_lps"]))
        if idx >= len(r["boundaries"]):
            continue
        bdy_pos = r["boundaries"][idx]
        prefix_text = tok.decode(r["tok_ids"][:bdy_pos], skip_special_tokens=True)
        branch_prompts.append(base_prompts[r["id"]] + prefix_text)
        ids.append(r["id"])
    sp = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=max_tok, seed=SEED + 42)
    outs = llm.generate(branch_prompts, sp)
    pilotC_results = {r["id"]: None for r in greedy_records}
    for ii, out in zip(ids, outs):
        preds = [extract_fn(c.text) for c in out.outputs]
        preds_clean = [p for p in preds if p is not None]
        r = greedy_records[ii]
        combined = [r["pred"]] + preds_clean
        counter = Counter([str(normalize(p) if not mcq else p) for p in combined if p is not None])
        top, _ = counter.most_common(1)[0] if counter else (None, 0)
        pilotC_results[ii] = {"pred": top, "correct": int(is_correct(top, r["gold"], mcq))}
    pc_acc = np.mean([rr["correct"] for rr in pilotC_results.values() if rr is not None])
    print(f"  Pilot C lp_min K4 T0.7: acc={pc_acc:.3f}", flush=True)

    # Paraphrase M=3 (consensus)
    para_prompts = []
    for i, q in enumerate(questions):
        for k in range(3):
            tmpl = paraphrases[k % len(paraphrases)]
            msg = tmpl.format(question=q, options=options[i] if mcq else "")
            p = tok.apply_chat_template(
                [{"role": "user", "content": msg}],
                tokenize=False, add_generation_prompt=True,
            )
            para_prompts.append(p)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tok, seed=SEED + 4)
    outs = llm.generate(para_prompts, sp)
    para_modal = []
    para_consensus = []
    for i in range(len(questions)):
        chunk = outs[i*3:(i+1)*3]
        preds = [extract_fn(c.outputs[0].text) for c in chunk]
        preds_clean = [p for p in preds if p is not None]
        # Combined with greedy
        combined = [greedy_records[i]["pred"]] + preds_clean
        counter = Counter([str(normalize(p) if not mcq else p) for p in combined if p is not None])
        top, cnt = counter.most_common(1)[0] if counter else (None, 0)
        para_modal.append({"pred": top, "correct": int(is_correct(top, golds[i], mcq))})
        para_consensus.append(cnt / max(4, 1))  # 4 = greedy + 3 paraphrases
    para_acc = np.mean([p["correct"] for p in para_modal])
    print(f"  Para M3 modal: acc={para_acc:.3f}", flush=True)

    # Combined: if para_consensus < 0.75, do step-branch K=4 to break tie
    combined_results = []
    n_branched = 0
    for i in range(len(questions)):
        cons = para_consensus[i]
        if cons >= 0.75:
            # high agreement: keep paraphrase modal answer
            combined_results.append(para_modal[i])
        else:
            # low agreement: use step-branching result
            n_branched += 1
            if pilotC_results[i] is not None:
                combined_results.append(pilotC_results[i])
            else:
                combined_results.append({"pred": greedy_records[i]["pred"], "correct": greedy_records[i]["correct"]})
    comb_acc = np.mean([r["correct"] for r in combined_results])
    print(f"  Combined (para→pilotC): acc={comb_acc:.3f}, n_branched={n_branched}/{len(questions)}", flush=True)

    return {
        "name": name, "n": len(questions),
        "greedy_acc": greedy_acc,
        "pilotC_lp_min_K4_T0.7_acc": pc_acc,
        "para_M3_modal_acc": para_acc,
        "combined_para_pilotC_acc": comb_acc,
        "n_branched_in_combined": n_branched,
    }


def main():
    print(f"Loading {MODEL}...", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2560,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tok = llm.get_tokenizer()

    all_results = []

    # MATH-500 (full to confirm previous)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    questions = [r["problem"] for r in ds][:200]
    golds = [r["answer"] for r in ds][:200]
    all_results.append(run_dataset(llm, tok, "MATH-500-200", questions, golds, PROMPT_BASE, PARAPHRASES))

    # AIME
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    rng = np.random.default_rng(0)
    indices = sorted(rng.permutation(len(ds))[:200].tolist())
    questions = [ds[i]["Question"] for i in indices]
    golds = [str(ds[i]["Answer"]).strip() for i in indices]
    all_results.append(run_dataset(llm, tok, "AIME-200", questions, golds, PROMPT_BASE, PARAPHRASES, max_tok=2048))

    # OlympiadBench
    ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    text_only = [r for r in ds if not any(r.get(f"image_{i}") for i in range(1, 6))][:200]
    questions = [r["question"] for r in text_only]
    golds = [r["final_answer"][0] if isinstance(r["final_answer"], list) and r["final_answer"] else r["final_answer"] for r in text_only]
    all_results.append(run_dataset(llm, tok, "Olympiad-200", questions, golds, PROMPT_BASE, PARAPHRASES, max_tok=2048))

    # MMLU-Pro STEM
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    stem = [r for r in ds if r["category"] in {"math", "physics", "chemistry"}][:200]
    questions = [r["question"] for r in stem]
    golds = [r["answer"] for r in stem]
    options = ["\n".join(f"{chr(ord('A')+j)}. {o}" for j, o in enumerate(r["options"])) for r in stem]
    all_results.append(run_dataset(llm, tok, "MMLU-Pro-200", questions, golds, PROMPT_MMLU,
                                       [PROMPT_MMLU]*3,  # use base for paraphrase too (mcq)
                                       max_tok=1536, mcq=True, options=options))

    SUM.write_text(json.dumps(all_results, indent=2))
    print(f"\n=== Final summary ===")
    for r in all_results:
        delta_pc = r['pilotC_lp_min_K4_T0.7_acc'] - r['greedy_acc']
        delta_para = r['para_M3_modal_acc'] - r['greedy_acc']
        delta_comb = r['combined_para_pilotC_acc'] - r['greedy_acc']
        print(f"  {r['name']:18s} greedy={r['greedy_acc']:.3f}  pilotC={r['pilotC_lp_min_K4_T0.7_acc']:.3f} ({delta_pc:+.3f})  para={r['para_M3_modal_acc']:.3f} ({delta_para:+.3f})  combined={r['combined_para_pilotC_acc']:.3f} ({delta_comb:+.3f})")


if __name__ == "__main__":
    main()

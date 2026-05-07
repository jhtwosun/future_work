"""
EXP-S2-extended: Test paraphrase_consensus + xtemp_agree on multiple datasets.
GOAL: replicate the strong para_M3 finding (ρ=0.70, kept_acc 92%) on more datasets.

Datasets:
- MATH-500 (full 500)
- AIME 1983-2024 (200 subset)
- OlympiadBench (200 subset)
- MMLU-Pro STEM (300 subset)

For each: greedy + para_M3 (M=3 paraphrases at T=0.0) + xtemp_N2 (N=2 at T=0.7).

This is the "diverse experimentation" phase — not cherry-picking single best, evaluating multiple.
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
OUT = OUTDIR / "SX_para_cross_dataset.jsonl"
SUM = OUTDIR / "SX_para_cross_summary.json"

PROMPT_BASE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>". '
    "Or put it within \\boxed{{}}.\n\n"
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
PARA_MMLU = [
    "Solve this multiple-choice problem.\n\nQ: {question}\nOptions:\n{options}\n\nThink step by step. Final answer in \\boxed{{}}.",
    "Carefully analyze this question and pick the best answer.\n\n{question}\n\nChoices:\n{options}\n\nReasoning step by step. Boxed final letter at end.",
    "Multiple choice question:\n\n{question}\n\n{options}\n\nReason carefully and provide your answer letter inside \\boxed{{}}.",
]


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
    tail = text[-200:]
    letters = re.findall(r"\b([A-J])\b", tail)
    if letters: return letters[-1]
    return None


def is_correct(pred, gold, mcq=False):
    if mcq:
        return pred == gold
    return equal_strict(pred, gold)


def run_dataset(llm, tok, dataset_name, questions, golds, prompt_base, paraphrases, max_tok=1536, mcq=False, options=None):
    print(f"\n=== {dataset_name}, n={len(questions)} ===", flush=True)
    base_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": prompt_base.format(question=q, options=opt) if mcq else prompt_base.format(question=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        for q, opt in zip(questions, options or [None]*len(questions))
    ]
    extract_fn = extract_letter if mcq else extract_pred

    # Greedy baseline
    sp = SamplingParams(temperature=0.0, max_tokens=max_tok, seed=SEED)
    t0 = time.time()
    outs = llm.generate(base_prompts, sp)
    dur_g = time.time() - t0
    greedy_preds = [extract_fn(o.outputs[0].text) for o in outs]
    greedy_correct = [int(is_correct(p, g, mcq)) for p, g in zip(greedy_preds, golds)]
    print(f"  Greedy: {sum(greedy_correct)}/{len(questions)} = {np.mean(greedy_correct):.3f} ({dur_g:.1f}s)", flush=True)

    # Paraphrase M=3
    para_prompts = []
    for i, q in enumerate(questions):
        for k in range(3):
            tmpl = paraphrases[k % len(paraphrases)]
            msg = tmpl.format(question=q, options=options[i] if mcq and options else "")
            p = tok.apply_chat_template(
                [{"role": "user", "content": msg}],
                tokenize=False, add_generation_prompt=True,
            )
            para_prompts.append(p)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tok, seed=SEED + 4)
    t0 = time.time()
    outs = llm.generate(para_prompts, sp)
    dur_p = time.time() - t0

    para_results = []
    for i in range(len(questions)):
        chunk = outs[i*3:(i+1)*3]
        preds = [extract_fn(c.outputs[0].text) for c in chunk]
        # Score = how many paraphrases agree with greedy
        agree_greedy = float(np.mean([is_correct(p, greedy_preds[i], mcq) and (p is not None) for p in preds]))
        # consensus = modal frequency
        preds_clean = [p for p in preds if p is not None]
        counter = Counter([str(p) for p in preds_clean])
        top, cnt = (counter.most_common(1)[0] if counter else (None, 0))
        consensus = cnt / 3
        para_results.append({
            "agree_greedy": agree_greedy,
            "consensus": consensus,
            "preds": preds,
        })
    print(f"  Para M=3: done in {dur_p:.1f}s", flush=True)

    # xtemp N=2 at T=0.7
    sp = SamplingParams(n=2, temperature=0.7, top_p=0.95, max_tokens=max_tok, seed=SEED + 5)
    t0 = time.time()
    outs = llm.generate(base_prompts, sp)
    dur_x = time.time() - t0

    xtemp_results = []
    for i, out in enumerate(outs):
        preds = [extract_fn(c.text) for c in out.outputs]
        agree_greedy = float(np.mean([is_correct(p, greedy_preds[i], mcq) and (p is not None) for p in preds]))
        preds_clean = [p for p in preds if p is not None]
        counter = Counter([str(p) for p in preds_clean])
        top, cnt = (counter.most_common(1)[0] if counter else (None, 0))
        top1_inc_greedy = (cnt + (1 if str(top) == str(greedy_preds[i]) else 0)) / 3
        xtemp_results.append({
            "agree_greedy": agree_greedy,
            "top1_inc_greedy": top1_inc_greedy,
        })
    print(f"  xtemp N=2: done in {dur_x:.1f}s", flush=True)

    return {
        "dataset": dataset_name,
        "n": len(questions),
        "greedy_acc": float(np.mean(greedy_correct)),
        "greedy_correct": greedy_correct,
        "greedy_wallclock": dur_g,
        "para_results": para_results,
        "para_wallclock": dur_p,
        "xtemp_results": xtemp_results,
        "xtemp_wallclock": dur_x,
    }


def cp_eval(scores, correct, alpha):
    import math as _m
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
            ql = max(0.0, min(1.0, _m.floor(alpha * (n_c + 1)) / n_c))
            q = float(np.quantile(cal_corr, ql))
            kept = sb[ti] >= q
            if kept.sum():
                boot_acc.append(float(cb[ti][kept].mean()))
            break
    if not boot_acc: return None
    return {
        "mean": float(np.mean(boot_acc)),
        "ci95": [float(np.quantile(boot_acc, 0.025)), float(np.quantile(boot_acc, 0.975))],
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

    # MATH-500 full
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    questions = [r["problem"] for r in ds][:500]
    golds = [r["answer"] for r in ds][:500]
    res = run_dataset(llm, tok, "MATH-500-full", questions, golds, PROMPT_BASE, PARAPHRASES)
    all_results.append(res)

    # AIME 200
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    rng = np.random.default_rng(0)
    indices = sorted(rng.permutation(len(ds))[:200].tolist())
    questions = [ds[i]["Question"] for i in indices]
    golds = [str(ds[i]["Answer"]).strip() for i in indices]
    res = run_dataset(llm, tok, "AIME-200", questions, golds, PROMPT_BASE, PARAPHRASES, max_tok=2048)
    all_results.append(res)

    # OlympiadBench math 200
    ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    text_only = [r for r in ds if not any(r.get(f"image_{i}") for i in range(1, 6))][:200]
    questions = [r["question"] for r in text_only]
    golds = [r["final_answer"][0] if isinstance(r["final_answer"], list) and r["final_answer"] else r["final_answer"] for r in text_only]
    res = run_dataset(llm, tok, "Olympiad-200", questions, golds, PROMPT_BASE, PARAPHRASES, max_tok=2048)
    all_results.append(res)

    # MMLU-Pro 300 STEM
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    stem = [r for r in ds if r["category"] in {"math", "physics", "chemistry"}][:300]
    questions = [r["question"] for r in stem]
    golds = [r["answer"] for r in stem]
    options = ["\n".join(f"{chr(ord('A')+j)}. {o}" for j, o in enumerate(r["options"])) for r in stem]
    res = run_dataset(llm, tok, "MMLU-Pro-300", questions, golds, PROMPT_MMLU, PARA_MMLU,
                       max_tok=1536, mcq=True, options=options)
    all_results.append(res)

    # Save raw
    with OUT.open("w") as f:
        for r in all_results:
            # strip per-question raw arrays; keep aggregates
            f.write(json.dumps({k: v for k, v in r.items()}) + "\n")

    # CP simulation per dataset, per score, per α
    summary = {"datasets": []}
    for r in all_results:
        name = r["dataset"]
        correct = r["greedy_correct"]
        para_agree = [pr["agree_greedy"] for pr in r["para_results"]]
        para_consensus = [pr["consensus"] for pr in r["para_results"]]
        xtemp_agree = [xr["agree_greedy"] for xr in r["xtemp_results"]]
        xtemp_top1 = [xr["top1_inc_greedy"] for xr in r["xtemp_results"]]

        ds_summary = {
            "dataset": name,
            "n": r["n"],
            "vanilla_acc": r["greedy_acc"],
            "scores": {},
        }
        for score_name, scores in [
            ("para_M3_agree_greedy", para_agree),
            ("para_M3_consensus", para_consensus),
            ("xtemp_N2_agree_greedy", xtemp_agree),
            ("xtemp_N2_top1_inc_greedy", xtemp_top1),
        ]:
            ds_summary["scores"][score_name] = {}
            for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
                cp = cp_eval(scores, correct, alpha)
                if cp:
                    ds_summary["scores"][score_name][f"alpha_{alpha}"] = cp
                    print(f"[{name:18s} {score_name:25s} α={alpha:.2f}] kept_acc={cp['mean']:.3f} CI=[{cp['ci95'][0]:.3f},{cp['ci95'][1]:.3f}]", flush=True)
        summary["datasets"].append(ds_summary)

    SUM.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote: {SUM}")


if __name__ == "__main__":
    main()

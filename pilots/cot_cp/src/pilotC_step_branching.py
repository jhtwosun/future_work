"""
Pilot C: step-level branching prototype.

Hypothesis: re-generating from the *worst step* (smallest mean log-prob)
with K alternatives at higher temperature can recover wrong answers,
without paying the cost of full N-sample SC on every question.

Procedure:
  1. Take pilot7 MATH-500 traces (greedy + step log-probs).
  2. For each trace:
     a. Find step t* whose mean log-prob is min (= the suspected weak step).
     b. Build a continuation prompt = original_prompt + tokens[: boundary[t*]].
     c. Sample K=4 alternative continuations at temp=0.7, max_tokens=remaining budget.
     d. Combine: original answer + K alternatives → 5 candidates total.
     e. Take majority vote among the 5 candidates.
  3. Compare:
     - vanilla greedy accuracy (51.6%)
     - branching at WORST step (random branch baseline)
     - branching only when lp_min < q_hat (CP-triggered)
  4. Report compute (extra forward passes used).

Output: results/pilotC_branching.json
"""

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_QUESTIONS = int(os.environ.get("NQ", "200"))
K_BRANCH = int(os.environ.get("K_BRANCH", "4"))
TEMP_BRANCH = 0.7
MAX_TOKENS = 1536
SEED = 0

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")
TRACES = RESULTS / "pilot7_math500_traces.jsonl"
OUT = RESULTS / "pilotC_branching_traces.jsonl"
SUM = RESULTS / "pilotC_summary.json"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step. "
    "Each reasoning step should be on its own line, separated by a blank line. "
    'After all reasoning, write the final answer on its own line as "Answer: <number-or-expression>".\n\n'
    "Problem: {question}\n\n"
)
PRED_RE = re.compile(r"(?i)answer\s*[:=]\s*([^\n]+)")
NUM_RE = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")


def normalize(s):
    if s is None:
        return None
    s = str(s).strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.rstrip(".,;:")
    s = re.sub(r"\s+", "", s)
    m = re.match(r"\\boxed\{(.+)\}$", s)
    if m:
        s = m.group(1)
    return s


def extract_pred(text):
    m = PRED_RE.search(text)
    if m:
        return normalize(m.group(1))
    nums = NUM_RE.findall(text)
    return normalize(nums[-1]) if nums else None


def equal(a, b):
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        af = float(a) if "/" not in a else float(a.split("/")[0]) / float(a.split("/")[1])
        bf = float(b) if "/" not in b else float(b.split("/")[0]) / float(b.split("/")[1])
        return abs(af - bf) < 1e-4
    except Exception:
        pass
    return False


def step_lp_scores(token_logprobs, boundaries):
    bdy = boundaries + [len(token_logprobs)]
    means = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        if seg:
            means.append(float(np.mean(seg)))
        else:
            means.append(float("nan"))
    return means


def main():
    print("Loading pilot7 traces and MATH-500...", flush=True)
    traces = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()][:N_QUESTIONS]
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    use = ds.select(range(min(N_QUESTIONS, len(ds))))
    questions = [r["problem"] for r in use]
    golds = [normalize(r["answer"]) for r in use]
    print(f"  {len(traces)} traces, {len(questions)} questions", flush=True)

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

    # Compute q_hat from CP-style calibration on the FIRST HALF (cal),
    # apply branching policy on the SECOND HALF (test).
    n_cal = len(traces) // 2
    cal_traces = traces[:n_cal]
    test_traces = traces[n_cal:]
    cal_questions = questions[:n_cal]
    test_questions = questions[n_cal:]
    cal_golds = golds[:n_cal]
    test_golds = golds[n_cal:]

    # Calibrate using lp_min trajectory score for "is this trace likely correct"
    cal_lp_min = []
    cal_correct = []
    for r in cal_traces:
        steps = step_lp_scores(r["token_logprobs"], r["step_boundaries"])
        steps = [s for s in steps if not (s != s)]
        if not steps:
            continue
        cal_lp_min.append(min(steps))
        cal_correct.append(int(bool(r["correct"])))
    cal_lp_min = np.array(cal_lp_min)
    cal_correct = np.array(cal_correct)
    alpha = 0.3
    cal_correct_scores = cal_lp_min[cal_correct == 1]
    n = len(cal_correct_scores)
    ql = max(0.0, min(1.0, math.floor(alpha * (n + 1)) / n))
    q_hat = float(np.quantile(cal_correct_scores, ql))
    print(f"q_hat (lp_min, α={alpha}) = {q_hat:.4f}", flush=True)

    # Test loop: for each test trace, decide whether to branch
    # If we branch: build continuation prompt, sample K alternatives.
    # Build all branch prompts in a batch for vLLM efficiency.
    branch_prompts = []
    branch_meta = []  # (test_idx, original_text, original_pred, original_correct, gold)
    for i, (r, q, gold) in enumerate(zip(test_traces, test_questions, test_golds)):
        steps = step_lp_scores(r["token_logprobs"], r["step_boundaries"])
        steps_clean = [(j, s) for j, s in enumerate(steps) if not (s != s)]
        if not steps_clean:
            continue
        worst_idx, worst_score = min(steps_clean, key=lambda x: x[1])
        original_pred = r["predicted_value"] if "predicted_value" in r else r.get("predicted")
        original_correct = bool(r["correct"])

        # Find the token boundary at start of worst step
        bdy = r["step_boundaries"]
        if worst_idx >= len(bdy):
            continue
        prefix_len = bdy[worst_idx]

        # Reconstruct prefix text from tokens[:prefix_len]
        # We need the original output_text up to prefix_len tokens.
        full_ids = r.get("token_ids")  # may not be present; fallback: use text
        # Use first prefix_len characters from output_text approximated by token-decoded prefix
        # Simpler: re-tokenize output_text and slice
        out_text = r.get("output_text", "")
        ids = tok.encode(out_text, add_special_tokens=False)
        prefix_ids = ids[:prefix_len]
        prefix_text = tok.decode(prefix_ids, skip_special_tokens=True) if prefix_ids else ""

        # Construct continuation prompt
        chat = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        cont_prompt = chat + prefix_text  # vLLM will continue from here

        will_branch_cp  = (worst_score < q_hat)
        will_branch_all = True  # baseline: always branch at worst step
        meta = {
            "test_idx": i,
            "id": r["id"],
            "worst_step_idx": worst_idx,
            "worst_step_lp": worst_score,
            "q_hat": q_hat,
            "below_qhat": will_branch_cp,
            "original_pred": original_pred,
            "original_correct": int(original_correct),
            "gold": gold,
            "prefix_len_tokens": prefix_len,
            "prompt_len": len(cont_prompt),
        }
        branch_meta.append(meta)
        branch_prompts.append(cont_prompt)

    print(f"  total branchable test traces: {len(branch_prompts)}", flush=True)
    print(f"  CP-triggered (below q_hat): {sum(1 for m in branch_meta if m['below_qhat'])}", flush=True)

    sp = SamplingParams(
        n=K_BRANCH, temperature=TEMP_BRANCH, top_p=0.95,
        max_tokens=MAX_TOKENS, seed=SEED,
    )
    t0 = time.time()
    outs = llm.generate(branch_prompts, sp)
    dur = time.time() - t0
    print(f"branch generation took {dur:.1f}s", flush=True)

    n_test = len(branch_meta)
    n_cp_branch = 0
    n_recovered_cp = 0
    n_lost_cp = 0
    n_recovered_all = 0
    n_lost_all = 0
    rows = []
    with OUT.open("w") as f:
        for meta, out in zip(branch_meta, outs):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter([meta["original_pred"]] + preds_clean)  # combine
            top, top_count = counter.most_common(1)[0]
            new_correct = equal(top, meta["gold"])
            counter_branch_only = Counter(preds_clean)
            top_branch, _ = (counter_branch_only.most_common(1)[0]
                              if counter_branch_only else (None, 0))
            new_correct_branch_only = equal(top_branch, meta["gold"])

            if not meta["original_correct"] and new_correct:
                n_recovered_all += 1
            if meta["original_correct"] and not new_correct:
                n_lost_all += 1

            if meta["below_qhat"]:
                n_cp_branch += 1
                if not meta["original_correct"] and new_correct:
                    n_recovered_cp += 1
                if meta["original_correct"] and not new_correct:
                    n_lost_cp += 1

            rec = {**meta,
                   "branched_majority_pred": top,
                   "branched_majority_correct": int(new_correct),
                   "branched_only_majority_pred": top_branch,
                   "branched_only_correct": int(new_correct_branch_only),
                   "n_branches": K_BRANCH,
                   "branch_predictions": preds,
                   }
            rows.append(rec)
            f.write(json.dumps(rec) + "\n")

    n_orig_correct = sum(m["original_correct"] for m in branch_meta)
    # Final accuracy under three policies on the test split:
    # (i)   vanilla greedy (no branching)
    # (ii)  always branch at worst step, take majority of {original} + K branches
    # (iii) only branch when lp_min < q_hat, else keep original
    n_always = sum(int(equal(r["branched_majority_pred"], r["gold"])) for r in rows)
    n_cp = sum(
        int(equal(r["branched_majority_pred"] if r["below_qhat"] else r["original_pred"], r["gold"]))
        for r in rows
    )

    summary = {
        "model": MODEL,
        "alpha": alpha,
        "q_hat_lp_min": q_hat,
        "n_cal": n_cal,
        "n_test": n_test,
        "vanilla_acc": n_orig_correct / n_test if n_test else 0.0,
        "always_branch_acc": n_always / n_test if n_test else 0.0,
        "cp_branch_acc": n_cp / n_test if n_test else 0.0,
        "n_cp_triggered": n_cp_branch,
        "cp_trigger_rate": n_cp_branch / n_test if n_test else 0.0,
        "always_recovered": n_recovered_all,
        "always_lost": n_lost_all,
        "cp_recovered": n_recovered_cp,
        "cp_lost": n_lost_cp,
        "K_branch": K_BRANCH,
        "branch_wallclock_sec": dur,
        "compute_overhead_total_branches": n_test * K_BRANCH,
        "compute_overhead_cp_branches": n_cp_branch * K_BRANCH,
    }
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

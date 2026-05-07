"""
Pilot L: rewrite-style step branching.

Pilot C/K showed that simple K-resample at temperature 0.7 from the
worst-step boundary does not yield net accuracy gains. The conjecture
in OBSERVATIONS was that the bottleneck is the *intervention*: the model
keeps producing similar continuations because the prefix anchors it.

This pilot tests the alternative intervention: prepend an explicit
"reconsideration" cue at the worst-step boundary, e.g.:
    "<prefix> Wait, let me reconsider this step — I might have made
     an error. Let me try a different approach.\\n\\n"
and then sample K continuations.

Setup mirrors Pilot C exactly so results are directly comparable.
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

REWRITE_CUE = (
    "Hmm, wait. Let me reconsider this step — I might have made an error. "
    "Let me try a different approach.\n\n"
)

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")
TRACES = RESULTS / "pilot7_math500_traces.jsonl"
OUT = RESULTS / "pilotL_rewrite_branching.jsonl"
SUM = RESULTS / "pilotL_summary.json"

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
    if a is None or b is None: return False
    if a == b: return True
    try:
        af = float(a) if "/" not in a else float(a.split("/")[0]) / float(a.split("/")[1])
        bf = float(b) if "/" not in b else float(b.split("/")[0]) / float(b.split("/")[1])
        return abs(af - bf) < 1e-4
    except Exception:
        return False


def step_lp_means(token_logprobs, boundaries):
    bdy = boundaries + [len(token_logprobs)]
    out = []
    for a, b in zip(bdy[:-1], bdy[1:]):
        seg = [lp for lp in token_logprobs[a:b] if not (lp != lp)]
        out.append(float(np.mean(seg)) if seg else float("nan"))
    return out


def main():
    traces = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()][:N_QUESTIONS]
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    use = ds.select(range(min(N_QUESTIONS, len(ds))))
    questions = [r["problem"] for r in use]
    golds = [normalize(r["answer"]) for r in use]
    print(f"{len(traces)} traces, {len(questions)} questions", flush=True)

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

    n_cal = len(traces) // 2
    cal_traces = traces[:n_cal]
    test_traces = traces[n_cal:]
    test_questions = questions[n_cal:]
    test_golds = golds[n_cal:]

    # CP threshold on lp_min, α=0.3 (same as Pilot C)
    cal_lp_min = []
    cal_correct = []
    for r in cal_traces:
        steps = step_lp_means(r["token_logprobs"], r["step_boundaries"])
        steps = [s for s in steps if not (s != s)]
        if not steps:
            continue
        cal_lp_min.append(min(steps))
        cal_correct.append(int(bool(r["correct"])))
    cal_lp_min = np.array(cal_lp_min); cal_correct = np.array(cal_correct)
    alpha = 0.3
    cal_corr = cal_lp_min[cal_correct == 1]
    n = len(cal_corr)
    ql = max(0.0, min(1.0, math.floor(alpha * (n + 1)) / n))
    q_hat = float(np.quantile(cal_corr, ql))
    print(f"q_hat (lp_min, α={alpha}) = {q_hat:.4f}", flush=True)

    # Build prompts: continuation from worst step boundary + REWRITE_CUE
    branch_prompts = []
    branch_meta = []
    for i, (r, q, gold) in enumerate(zip(test_traces, test_questions, test_golds)):
        steps = step_lp_means(r["token_logprobs"], r["step_boundaries"])
        steps_clean = [(j, s) for j, s in enumerate(steps) if not (s != s)]
        if not steps_clean:
            continue
        worst_idx, worst_score = min(steps_clean, key=lambda x: x[1])
        original_pred = r.get("predicted_value") or r.get("predicted")
        original_correct = bool(r["correct"])

        bdy = r["step_boundaries"]
        if worst_idx >= len(bdy):
            continue
        prefix_len = bdy[worst_idx]
        out_text = r.get("output_text", "")
        ids = tok.encode(out_text, add_special_tokens=False)
        prefix_ids = ids[:prefix_len]
        prefix_text = tok.decode(prefix_ids, skip_special_tokens=True) if prefix_ids else ""

        chat = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        # Two prompts per question: one with cue, one without (control = Pilot C-style)
        cont_with_cue = chat + prefix_text + REWRITE_CUE
        cont_without_cue = chat + prefix_text

        meta = {
            "test_idx": i, "id": r["id"],
            "worst_step_idx": worst_idx, "worst_step_lp": worst_score,
            "q_hat": q_hat, "below_qhat": worst_score < q_hat,
            "original_pred": original_pred,
            "original_correct": int(original_correct),
            "gold": gold,
        }
        branch_meta.append(meta)
        branch_prompts.append(cont_with_cue)
        branch_meta.append({**meta, "_variant": "no_cue"})
        branch_prompts.append(cont_without_cue)

    print(f"  generation requests: {len(branch_prompts)} (each with K={K_BRANCH})", flush=True)
    sp = SamplingParams(n=K_BRANCH, temperature=TEMP_BRANCH, top_p=0.95,
                         max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(branch_prompts, sp)
    dur = time.time() - t0
    print(f"branch generation took {dur:.1f}s", flush=True)

    # Pair the with_cue / without_cue results back
    # Each pair: meta (no _variant) for with_cue, then meta with _variant=no_cue
    n_pairs = len(branch_meta) // 2
    rows = []
    n_orig_correct = 0
    n_cue_correct = 0
    n_nocue_correct = 0
    n_recovered_cue = 0
    n_lost_cue = 0
    n_recovered_nocue = 0
    n_lost_nocue = 0
    n_below = 0
    n_recovered_cp_cue = 0
    n_lost_cp_cue = 0
    with OUT.open("w") as f:
        for k in range(n_pairs):
            m_with = branch_meta[2*k]
            m_no   = branch_meta[2*k + 1]
            o_with = outs[2*k]
            o_no   = outs[2*k + 1]
            preds_with = [extract_pred(c.text) for c in o_with.outputs]
            preds_no   = [extract_pred(c.text) for c in o_no.outputs]
            preds_with_clean = [p for p in preds_with if p is not None]
            preds_no_clean   = [p for p in preds_no if p is not None]

            # Combined majority: original + branched
            counter_with = Counter([m_with["original_pred"]] + preds_with_clean)
            counter_no   = Counter([m_with["original_pred"]] + preds_no_clean)
            top_with = counter_with.most_common(1)[0][0] if counter_with else None
            top_no   = counter_no.most_common(1)[0][0]   if counter_no   else None

            ok_with = equal(top_with, m_with["gold"])
            ok_no   = equal(top_no,   m_with["gold"])
            ok_orig = bool(m_with["original_correct"])
            n_orig_correct += int(ok_orig)
            n_cue_correct += int(ok_with)
            n_nocue_correct += int(ok_no)
            if not ok_orig and ok_with: n_recovered_cue += 1
            if ok_orig and not ok_with: n_lost_cue += 1
            if not ok_orig and ok_no: n_recovered_nocue += 1
            if ok_orig and not ok_no: n_lost_nocue += 1
            if m_with["below_qhat"]:
                n_below += 1
                if not ok_orig and ok_with: n_recovered_cp_cue += 1
                if ok_orig and not ok_with: n_lost_cp_cue += 1

            rec = {
                **m_with,
                "branch_with_cue_pred":   top_with,
                "branch_with_cue_correct": int(ok_with),
                "branch_no_cue_pred":      top_no,
                "branch_no_cue_correct":   int(ok_no),
                "n_branches": K_BRANCH,
            }
            rows.append(rec)
            f.write(json.dumps(rec) + "\n")

    n_test = n_pairs
    # CP-triggered policies
    n_cp_with = sum(int(equal(r["branch_with_cue_pred"] if r["below_qhat"] else r["original_pred"], r["gold"])) for r in rows)
    n_cp_no   = sum(int(equal(r["branch_no_cue_pred"]  if r["below_qhat"] else r["original_pred"], r["gold"])) for r in rows)

    summary = {
        "model": MODEL, "alpha": alpha, "q_hat_lp_min": q_hat,
        "n_test": n_test, "K_branch": K_BRANCH,
        "vanilla_acc":            n_orig_correct / n_test,
        "always_branch_with_cue": n_cue_correct / n_test,
        "always_branch_no_cue":   n_nocue_correct / n_test,  # Pilot C replication
        "cp_branch_with_cue":     n_cp_with / n_test,
        "cp_branch_no_cue":       n_cp_no / n_test,
        "recovered_with_cue":     n_recovered_cue, "lost_with_cue": n_lost_cue,
        "recovered_no_cue":       n_recovered_nocue, "lost_no_cue": n_lost_nocue,
        "cp_recovered_with_cue":  n_recovered_cp_cue, "cp_lost_with_cue": n_lost_cp_cue,
        "n_cp_triggered":         n_below,
        "branch_wallclock_sec":   dur,
    }
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

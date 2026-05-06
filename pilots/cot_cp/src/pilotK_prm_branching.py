"""
Pilot K: PRM-based step branching.

Same protocol as Pilot C, but uses Qwen2.5-Math-PRM-7B step rewards
(from Pilot A) to choose where to branch.

Hypothesis: a PRM trained for step quality is a stronger
step-error localizer than lp_min. So worst-PRM-step branching
should recover more incorrect trajectories than worst-lp_min-step.
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
PRM_RECS = RESULTS / "pilotA_prm_traces.jsonl"
OUT = RESULTS / "pilotK_prm_branching.jsonl"
SUM = RESULTS / "pilotK_summary.json"

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


def main():
    traces = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()][:N_QUESTIONS]
    prm_records = {json.loads(l)["id"]: json.loads(l) for l in PRM_RECS.read_text().splitlines() if l.strip()}

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    use = ds.select(range(min(N_QUESTIONS, len(ds))))
    questions = [r["problem"] for r in use]
    golds = [normalize(r["answer"]) for r in use]
    print(f"{len(traces)} traces, {len(prm_records)} PRM records, {len(questions)} questions", flush=True)

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

    # 50/50 cal/test split. Calibrate q_hat on PRM_min trajectory score.
    n_cal = len(traces) // 2
    cal_traces = traces[:n_cal]
    test_traces = traces[n_cal:]
    cal_questions = questions[:n_cal]
    test_questions = questions[n_cal:]
    cal_golds = golds[:n_cal]
    test_golds = golds[n_cal:]

    cal_prm_min = []
    cal_correct = []
    for r in cal_traces:
        rec = prm_records.get(r["id"])
        if rec is None:
            continue
        cal_prm_min.append(rec["prm_min"])
        cal_correct.append(int(bool(r["correct"])))
    cal_prm_min = np.array(cal_prm_min)
    cal_correct = np.array(cal_correct)
    alpha = 0.3
    cal_corr = cal_prm_min[cal_correct == 1]
    n = len(cal_corr)
    ql = max(0.0, min(1.0, math.floor(alpha * (n + 1)) / n))
    q_hat = float(np.quantile(cal_corr, ql))
    print(f"q_hat (prm_min, α={alpha}) = {q_hat:.4f}", flush=True)

    branch_prompts = []
    branch_meta = []
    for i, (r, q, gold) in enumerate(zip(test_traces, test_questions, test_golds)):
        rec = prm_records.get(r["id"])
        if rec is None or not rec.get("prm_step_rewards"):
            continue
        rewards = rec["prm_step_rewards"]
        worst_idx, worst_score = min(enumerate(rewards), key=lambda x: x[1])
        original_pred = r.get("predicted_value") or r.get("predicted")
        original_correct = bool(r["correct"])

        # Find boundary at start of worst step in the ORIGINAL token sequence.
        # Pilot 7 already has token boundaries. Use those mapped to step index.
        bdy = r["step_boundaries"]
        # The PRM may have re-tokenized differently (split on \n\s*\n+ chars).
        # We assume the PRM step index maps approximately to the same step index.
        if worst_idx >= len(bdy):
            # PRM has more steps than vLLM's \n\n splitter detected — fall back to last
            worst_idx = len(bdy) - 1
        prefix_len = bdy[worst_idx]

        out_text = r.get("output_text", "")
        ids = tok.encode(out_text, add_special_tokens=False)
        prefix_ids = ids[:prefix_len]
        prefix_text = tok.decode(prefix_ids, skip_special_tokens=True) if prefix_ids else ""
        chat = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        cont_prompt = chat + prefix_text

        will_branch_cp = (worst_score < q_hat)
        meta = {
            "test_idx": i, "id": r["id"],
            "worst_step_idx": worst_idx, "worst_step_prm": worst_score,
            "q_hat": q_hat, "below_qhat": will_branch_cp,
            "original_pred": original_pred,
            "original_correct": int(original_correct),
            "gold": gold,
            "n_steps_prm": len(rewards),
        }
        branch_meta.append(meta)
        branch_prompts.append(cont_prompt)

    print(f"  branchable test traces: {len(branch_prompts)}", flush=True)
    print(f"  CP-triggered (below q_hat): {sum(1 for m in branch_meta if m['below_qhat'])}", flush=True)

    sp = SamplingParams(n=K_BRANCH, temperature=TEMP_BRANCH, top_p=0.95,
                         max_tokens=MAX_TOKENS, seed=SEED)
    t0 = time.time()
    outs = llm.generate(branch_prompts, sp)
    dur = time.time() - t0
    print(f"branch generation took {dur:.1f}s", flush=True)

    n_test = len(branch_meta)
    n_cp_branch = 0
    n_recovered_cp, n_lost_cp = 0, 0
    n_recovered_all, n_lost_all = 0, 0
    rows = []
    with OUT.open("w") as f:
        for meta, out in zip(branch_meta, outs):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter([meta["original_pred"]] + preds_clean)
            top, top_count = counter.most_common(1)[0]
            new_correct = equal(top, meta["gold"])

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
            rec = {**meta, "branched_majority_pred": top,
                   "branched_majority_correct": int(new_correct),
                   "n_branches": K_BRANCH, "branch_predictions": preds}
            rows.append(rec)
            f.write(json.dumps(rec) + "\n")

    n_orig_correct = sum(m["original_correct"] for m in branch_meta)
    n_always = sum(int(equal(r["branched_majority_pred"], r["gold"])) for r in rows)
    n_cp = sum(
        int(equal(r["branched_majority_pred"] if r["below_qhat"] else r["original_pred"], r["gold"]))
        for r in rows
    )

    summary = {
        "model": MODEL, "alpha": alpha, "q_hat_prm_min": q_hat,
        "n_cal": n_cal, "n_test": n_test,
        "vanilla_acc": n_orig_correct / n_test if n_test else 0.0,
        "always_branch_acc": n_always / n_test if n_test else 0.0,
        "cp_branch_acc": n_cp / n_test if n_test else 0.0,
        "n_cp_triggered": n_cp_branch,
        "cp_trigger_rate": n_cp_branch / n_test if n_test else 0.0,
        "always_recovered": n_recovered_all, "always_lost": n_lost_all,
        "cp_recovered": n_recovered_cp, "cp_lost": n_lost_cp,
        "K_branch": K_BRANCH, "branch_wallclock_sec": dur,
    }
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

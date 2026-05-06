"""
Pilot D: compute-matched self-consistency sweep on MATH-500.

For each N in [1, 4, 8, 16, 32]:
  - run SC@N on the same 200 MATH-500 questions
  - record per-question top1_frac, majority pred, correctness
  - plot vanilla SC accuracy vs N (compute)
  - apply CP filter using top1_frac as score; report kept_acc at α=0.3
"""

import json
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
N_LIST = [1, 4, 8, 16, 32]
MAX_TOKENS = 1536
TEMP = 0.7
SEED = 0

RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")
OUT = RESULTS / "pilotD_compute_matched_sc.jsonl"
SUM = RESULTS / "pilotD_summary.json"

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
    s = s.strip()
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


def split_cp(scores, correct, alpha, n_seeds=200, cal_frac=0.5):
    import math
    rng = np.random.default_rng(0)
    accs, fracs, covs = [], [], []
    for _ in range(n_seeds):
        idx = rng.permutation(len(scores))
        nc = int(cal_frac * len(scores))
        ci, ti = idx[:nc], idx[nc:]
        cal_correct = scores[ci][correct[ci] == 1]
        if len(cal_correct) < 5:
            continue
        n = len(cal_correct)
        ql = max(0.0, min(1.0, math.floor(alpha * (n + 1)) / n))
        q = float(np.quantile(cal_correct, ql))
        kept = scores[ti] >= q
        n_corr = (correct[ti] == 1).sum()
        if n_corr == 0:
            continue
        cov = float((kept & (correct[ti] == 1)).sum() / n_corr)
        acc = float(correct[ti][kept].mean()) if kept.sum() else float("nan")
        fr = float(kept.mean())
        covs.append(cov); accs.append(acc); fracs.append(fr)
    return {
        "kept_acc_mean": float(np.mean(accs)) if accs else float("nan"),
        "kept_frac_mean": float(np.mean(fracs)) if fracs else float("nan"),
        "coverage_mean": float(np.mean(covs)) if covs else float("nan"),
    }


def main():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    use = ds.select(range(min(N_QUESTIONS, len(ds))))
    questions = [r["problem"] for r in use]
    golds = [normalize(r["answer"]) for r in use]
    print(f"{len(questions)} MATH-500 problems; N_LIST={N_LIST}")

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2560,
        tensor_parallel_size=1,
        seed=SEED,
    )
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    rows = []
    summary = []
    f = OUT.open("w")
    for N in N_LIST:
        sp = SamplingParams(n=N, temperature=TEMP if N > 1 else 0.0, top_p=0.95 if N > 1 else 1.0,
                             max_tokens=MAX_TOKENS, seed=SEED)
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        dur = time.time() - t0
        n_correct = 0
        n_any = 0
        top1_fracs = []
        correctness = []
        for i, (q, gold, out) in enumerate(zip(questions, golds, outs)):
            preds = [extract_pred(c.text) for c in out.outputs]
            preds_clean = [p for p in preds if p is not None]
            counter = Counter(preds_clean)
            top, top_count = (counter.most_common(1)[0] if counter else (None, 0))
            top1_frac = top_count / N
            top1_fracs.append(top1_frac)
            ok = equal(top, gold)
            correctness.append(int(ok))
            any_ok = any(equal(p, gold) for p in preds_clean)
            n_correct += int(ok)
            n_any += int(any_ok)
            f.write(json.dumps({
                "N": N,
                "id": i,
                "majority_correct": int(ok),
                "any_correct": int(any_ok),
                "top1_frac": top1_frac,
                "answer_distribution": dict(counter),
            }) + "\n")

        scores = np.array(top1_fracs)
        correct = np.array(correctness)
        cp_03 = split_cp(scores, correct, 0.3)
        cp_05 = split_cp(scores, correct, 0.5)
        out_row = {
            "N": N,
            "n_questions": len(prompts),
            "majority_accuracy": n_correct / len(prompts),
            "oracle_any_correct": n_any / len(prompts),
            "mean_top1_frac": float(np.mean(top1_fracs)),
            "wallclock_sec": dur,
            "tokens_per_q": dur,  # filled below
            "cp_alpha_0.3": cp_03,
            "cp_alpha_0.5": cp_05,
        }
        summary.append(out_row)
        print(f"[N={N:2d}] majAcc={out_row['majority_accuracy']:.3f}  any={out_row['oracle_any_correct']:.3f}  "
              f"top1={out_row['mean_top1_frac']:.3f}  CP@0.3 acc={cp_03['kept_acc_mean']:.3f} keep%={cp_03['kept_frac_mean']:.2f}  "
              f"CP@0.5 acc={cp_05['kept_acc_mean']:.3f} keep%={cp_05['kept_frac_mean']:.2f}  "
              f"wall={dur:.1f}s")
    f.close()
    SUM.write_text(json.dumps(summary, indent=2))
    print(f"Wrote: {SUM}")

    # Pareto plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = [r["N"] for r in summary]
    vanilla = [r["majority_accuracy"] for r in summary]
    cp_03_acc = [r["cp_alpha_0.3"]["kept_acc_mean"] for r in summary]
    cp_03_frac = [r["cp_alpha_0.3"]["kept_frac_mean"] for r in summary]
    cp_05_acc = [r["cp_alpha_0.5"]["kept_acc_mean"] for r in summary]
    cp_05_frac = [r["cp_alpha_0.5"]["kept_frac_mean"] for r in summary]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Ns, vanilla, "o-", label="vanilla SC@N (full coverage)")
    ax.plot(Ns, cp_03_acc, "s-", label="CP+SC@N α=0.3")
    ax.plot(Ns, cp_05_acc, "^-", label="CP+SC@N α=0.5")
    for n, fr03, fr05 in zip(Ns, cp_03_frac, cp_05_frac):
        ax.annotate(f"keep={fr03:.2f}", (n, cp_03_acc[Ns.index(n)]),
                    fontsize=7, textcoords="offset points", xytext=(4, 4), color="C1")
    ax.set_xscale("log")
    ax.set_xlabel("N (samples per question, log)")
    ax.set_ylabel("accuracy")
    ax.set_title("MATH-500: vanilla SC vs CP-filtered SC by compute")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "pilotD_compute_matched.png", dpi=120)
    plt.close(fig)
    print("Saved: pilotD_compute_matched.png")


if __name__ == "__main__":
    main()

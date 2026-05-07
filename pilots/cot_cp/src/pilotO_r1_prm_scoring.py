"""
Pilot O: PRM scoring on DeepSeek-R1-Distill-Qwen-7B greedy traces.

Tests whether the Qwen2.5-Math-PRM-7B signal generalizes across
generator models — specifically to long-CoT R1-Distill traces with
~60 reasoning steps.

Reads:
  pilotE_r1_greedy_traces.jsonl
Writes:
  pilotO_r1_prm_traces.jsonl
  pilotO_r1_prm_summary.json
"""

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

PRM_REPO = "Qwen/Qwen2.5-Math-PRM-7B"
RESULTS = Path("/home/nvidia/future/pilots/cot_cp/results")
TRACES_IN = RESULTS / "pilotE_r1_greedy_traces.jsonl"
TRACES_OUT = RESULTS / "pilotO_r1_prm_traces.jsonl"
SUM_OUT = RESULTS / "pilotO_r1_prm_summary.json"

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

# R1-Distill outputs are usually wrapped in <think> ... </think>; we drop the
# tag-only segments and split on \n\n boundaries.
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def make_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor) -> list[list[float]]:
    probs = F.softmax(logits, dim=-1)
    probs = probs * token_masks.unsqueeze(-1)
    out = []
    for i in range(probs.size(0)):
        sample = probs[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        out.append(positive_probs.cpu().tolist())
    return out


def split_steps(text: str) -> list[str]:
    # Strip any <think> wrapper but keep its content
    pieces = []
    last_end = 0
    for m in THINK_RE.finditer(text):
        pieces.append(text[last_end:m.start()])
        pieces.append(m.group(1))
        last_end = m.end()
    pieces.append(text[last_end:])
    cleaned = "\n\n".join(p.strip() for p in pieces if p.strip())
    parts = re.split(r"\n\s*\n+", cleaned.strip())
    steps = [p.strip() for p in parts if p.strip()]
    return steps


def split_cp(scores, correct, alpha, n_seeds=200, cal_frac=0.5):
    rng = np.random.default_rng(0)
    accs, fracs, covs = [], [], []
    for _ in range(n_seeds):
        idx = rng.permutation(len(scores))
        nc = int(cal_frac * len(scores))
        ci, ti = idx[:nc], idx[nc:]
        cal_corr = scores[ci][correct[ci] == 1]
        if len(cal_corr) < 5:
            continue
        n = len(cal_corr)
        ql = max(0.0, min(1.0, math.floor(alpha * (n + 1)) / n))
        q = float(np.quantile(cal_corr, ql))
        kept = scores[ti] >= q
        n_corr = (correct[ti] == 1).sum()
        if n_corr == 0:
            continue
        cov = float((kept & (correct[ti] == 1)).sum() / n_corr)
        acc = float(correct[ti][kept].mean()) if kept.sum() else float("nan")
        fr  = float(kept.mean())
        accs.append(acc); fracs.append(fr); covs.append(cov)
    return {
        "kept_acc_mean":  float(np.mean(accs))  if accs  else float("nan"),
        "kept_frac_mean": float(np.mean(fracs)) if fracs else float("nan"),
        "coverage_mean":  float(np.mean(covs))  if covs  else float("nan"),
    }


def main():
    rows = [json.loads(l) for l in TRACES_IN.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} R1 greedy traces", flush=True)

    print(f"Loading PRM {PRM_REPO} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(PRM_REPO, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        PRM_REPO,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    sep_id = tok.encode("<extra_0>")[0]

    out_records = []
    n_truncated = 0
    with TRACES_OUT.open("w") as f:
        for i, r in enumerate(rows):
            question = r["question"]
            steps = split_steps(r["output_text"])
            if not steps:
                continue
            # PRM has 4096-token context; limit number of steps to fit budget
            # Approximate tokens: ~25 per step + question + system overhead
            MAX_STEPS = 80
            if len(steps) > MAX_STEPS:
                # keep first half (early reasoning) + last half (final answer area)
                half = MAX_STEPS // 2
                steps = steps[:half] + steps[-half:]
                n_truncated += 1

            assistant_str = "<extra_0>".join(steps) + "<extra_0>"
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_str},
            ]
            conv = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            input_ids = tok.encode(conv, return_tensors="pt").to(model.device)
            if input_ids.shape[1] > 4096:
                input_ids = input_ids[:, -4096:]
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs[0]
            mask = (input_ids == sep_id)
            step_rewards = make_step_rewards(logits, mask)[0]
            if len(step_rewards) > len(steps):
                step_rewards = step_rewards[:len(steps)]
            elif len(step_rewards) < len(steps):
                m = float(np.mean(step_rewards)) if step_rewards else 0.5
                step_rewards = step_rewards + [m] * (len(steps) - len(step_rewards))

            rec = {
                "id": r["id"],
                "correct": int(bool(r["correct"])),
                "n_steps_used": len(steps),
                "prm_step_rewards": step_rewards,
                "prm_mean":   float(np.mean(step_rewards)),
                "prm_min":    float(np.min(step_rewards)),
                "prm_median": float(np.median(step_rewards)),
                "prm_last":   float(step_rewards[-1]),
                "prm_first":  float(step_rewards[0]),
            }
            out_records.append(rec)
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(rows)} done", flush=True)

    print(f"Total scored: {len(out_records)}, truncated: {n_truncated}", flush=True)

    summary = {"prm_repo": PRM_REPO, "generator": "DeepSeek-R1-Distill-Qwen-7B",
                "n": len(out_records), "n_truncated": n_truncated,
                "accuracy": float(np.mean([r["correct"] for r in out_records])),
                "cp_results": []}
    correct = np.array([r["correct"] for r in out_records])
    from scipy.stats import spearmanr, pointbiserialr
    for sk in ["prm_mean", "prm_min", "prm_median", "prm_last", "prm_first"]:
        scores = np.array([r[sk] for r in out_records])
        rho, p = spearmanr(scores, correct)
        rpb, ppb = pointbiserialr(scores, correct)
        summary[f"corr_{sk}"] = {
            "spearman_rho": float(rho), "spearman_p": float(p),
            "pointbiserial_r": float(rpb), "pointbiserial_p": float(ppb),
        }
        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
            cp = split_cp(scores, correct, alpha)
            summary["cp_results"].append({"score": sk, "alpha": alpha, **cp})
            print(f"[R1-PRM/{sk:10s} α={alpha:.2f}] cov={cp['coverage_mean']:.3f} keepacc={cp['kept_acc_mean']:.3f} keep%={cp['kept_frac_mean']:.2f}")
        print(f"  ρ(R1-PRM/{sk}) = {rho:+.3f}")
    SUM_OUT.write_text(json.dumps(summary, indent=2))
    print(f"Wrote: {SUM_OUT}", flush=True)


if __name__ == "__main__":
    main()

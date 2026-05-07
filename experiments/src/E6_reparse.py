"""Re-parse E6 (Qwen3-8B) traces using boxed-aware extractor."""
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np

RESULTS = Path("/home/nvidia/future/experiments/results")
GREEDY = RESULTS / "E6_qwen3_8b_greedy.jsonl"
SC = RESULTS / "E6_qwen3_8b_sc.jsonl"
GREEDY_OUT = RESULTS / "E6_qwen3_8b_greedy_reparsed.jsonl"
SC_OUT = RESULTS / "E6_qwen3_8b_sc_reparsed.jsonl"
SUM = RESULTS / "E6_summary_reparsed.json"

PRED_BOX = re.compile(r"\\boxed\{([^{}]+)\}")
PRED_BOX_NESTED = re.compile(r"\\boxed\{(.+?)\}\s*$", re.DOTALL)
PRED_ANS = re.compile(r"(?i)answer\s*[:=]\s*([^\n]+)")
NUM_RE = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")


def normalize(s):
    if s is None: return None
    s = str(s).strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.rstrip(".,;:")
    s = re.sub(r"\s+", "", s)
    while True:
        m = re.match(r"\\boxed\{(.+)\}$", s)
        if m: s = m.group(1)
        else: break
    return s


def extract_pred(text):
    # find ALL \boxed{} (handles nested somewhat) and take the last one
    matches = list(re.finditer(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text))
    if matches:
        return normalize(matches[-1].group(1))
    m = PRED_ANS.search(text)
    if m: return normalize(m.group(1))
    nums = NUM_RE.findall(text)
    return normalize(nums[-1]) if nums else None


def equal(a, b):
    if a is None or b is None: return False
    a_n = normalize(a); b_n = normalize(b)
    if a_n == b_n: return True
    try:
        af = float(a_n) if "/" not in a_n else float(a_n.split("/")[0]) / float(a_n.split("/")[1])
        bf = float(b_n) if "/" not in b_n else float(b_n.split("/")[0]) / float(b_n.split("/")[1])
        return abs(af - bf) < 1e-4
    except Exception:
        pass
    # fall back to substring after stripping latex helpers
    def latex_strip(x):
        return re.sub(r"\\(left|right|,|;|!|:)", "", x or "")
    return latex_strip(a_n) == latex_strip(b_n)


def main():
    # Greedy
    n_correct = 0
    rows = [json.loads(l) for l in GREEDY.read_text().splitlines() if l.strip()]
    with GREEDY_OUT.open("w") as f:
        for r in rows:
            pred = extract_pred(r["output_text"])
            ok = equal(pred, r["gold"])
            r["predicted"] = pred
            r["correct"] = ok
            n_correct += int(ok)
            f.write(json.dumps(r) + "\n")
    greedy_acc = n_correct / len(rows)
    print(f"E6 reparsed greedy: {n_correct}/{len(rows)} = {greedy_acc:.3f}")

    # SC@8 — re-derive majority from samples (output_text isn't stored for SC; need to re-derive from saved samples in original SC traces)
    # SC traces in E6 don't have full sample text — only majority_pred.
    # The original extractor was wrong, so the sample-level preds and majority_pred are also wrong.
    # We can only fix this by re-running SC, but the user said no pilot/re-run.
    # Best we can do: re-classify majority_correct using the new equal() comparator.
    sc_rows = [json.loads(l) for l in SC.read_text().splitlines() if l.strip()]
    n_maj = 0
    n_any = 0
    re_count_fixed = 0
    for r in sc_rows:
        # Re-evaluate majority pred against gold using new equal()
        old_correct = int(r["majority_correct"])
        new_correct = int(equal(r["majority_pred"], r["gold"]))
        if new_correct != old_correct:
            re_count_fixed += 1
            r["majority_correct"] = new_correct
        n_maj += new_correct
        # any: re-evaluate against gold using new equal() over the answer_distribution keys
        any_ok = any(equal(p, r["gold"]) for p in (r.get("answer_distribution") or {}).keys())
        r["any_correct"] = int(any_ok)
        n_any += int(any_ok)
    with SC_OUT.open("w") as f:
        for r in sc_rows:
            f.write(json.dumps(r) + "\n")
    sc_acc = n_maj / len(sc_rows)
    any_acc = n_any / len(sc_rows)
    print(f"E6 reparsed SC@8: maj {n_maj}/{len(sc_rows)} = {sc_acc:.3f}, any={any_acc:.3f}, fixed={re_count_fixed}")

    summary = {
        "model": "Qwen/Qwen3-8B (reparsed)",
        "n": len(rows), "dataset": "MATH-500",
        "greedy": {"accuracy": greedy_acc, "n": len(rows)},
        "sc": {"n_samples": 8, "majority_accuracy": sc_acc, "any_accuracy": any_acc, "n": len(sc_rows)},
        "n_re_classified_sc": re_count_fixed,
        "note": "SC is approximate: only majority_pred could be re-evaluated against gold (sample texts not stored)."
    }
    SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

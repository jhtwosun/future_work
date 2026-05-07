"""
Robust answer evaluator. Re-extracts predictions and re-classifies
correctness for all greedy + SC traces produced by E1-E8 / Pilot 7,9
using a unified extractor that handles:
  - "Answer: X" format (Qwen2.5)
  - \\boxed{X} format (R1, Qwen3, QwQ, Phi-4)
  - LaTeX inline math \\(X\\) and \\[X\\]
  - LaTeX wrappers \\left( \\right), \\cdot, \\,, \\quad, \\frac
  - Numerical comparison via float / fraction parsing
  - LaTeX-fraction → simple fraction conversion

Outputs *_eval.json next to each summary.
"""

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np

EXP = Path("/home/nvidia/future/experiments/results")
PILOTS = Path("/home/nvidia/future/pilots/cot_cp/results")


# --- Extractors ---

PRED_BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})+)\}")
PRED_ANS = re.compile(r"(?i)\banswer\s*[:=]\s*([^\n]+)")
PRED_FINAL = re.compile(r"(?i)\bfinal\s+answer\s*[:=]?\s*([^\n]+)")
NUM_RE = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")


def extract_pred(text: str) -> str | None:
    """Try multiple strategies. Prefer \\boxed{} (most reliable for math models),
    then 'Final answer:', then 'Answer:' (LAST match to skip mid-trace mentions),
    then last number."""
    if not text:
        return None
    matches = list(PRED_BOX.finditer(text))
    if matches:
        return matches[-1].group(1)
    final_matches = list(PRED_FINAL.finditer(text))
    if final_matches:
        return final_matches[-1].group(1)
    ans_matches = list(PRED_ANS.finditer(text))
    if ans_matches:
        return ans_matches[-1].group(1)
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def normalize(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    # LaTeX inline math wrappers
    s = re.sub(r"^\\\((.+?)\\\)$", r"\1", s)
    s = re.sub(r"^\\\[(.+?)\\\]$", r"\1", s)
    # Plain $...$ wrapper
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1]
    # Leading currency-style $ (e.g., "$125")
    if s.startswith("$") and not s.endswith("$"):
        s = s[1:]
    s = s.strip()
    # Trailing punctuation
    s = s.rstrip(".,;:")
    # Trailing units like "dollars", "cents", "%"
    s = re.sub(r"\s*(dollars|cents|%|degrees|cm|km|kg|g|m|ft|in|mph)$", "", s, flags=re.I)
    # Strip outer parens / brackets if balanced
    while len(s) > 2 and s[0] in "([{" and s[-1] in ")]}" and s.count(s[0]) == 1:
        s = s[1:-1].strip()
    # Strip whitespace
    s = re.sub(r"\s+", "", s)
    # Strip outer \boxed{...}
    while True:
        m = re.match(r"^\\boxed\{(.+)\}$", s)
        if m:
            s = m.group(1)
        else:
            break
    # Strip \text{...}
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    # Strip \mathrm{...}
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    # Strip \left and \right
    s = re.sub(r"\\(left|right)", "", s)
    # Strip \! \, \; \: \quad \qquad
    s = re.sub(r"\\(!|,|;|:|quad|qquad)", "", s)
    # Strip \cdot
    s = re.sub(r"\\cdot", "*", s)
    # Strip multiple inner LaTeX inline wrappers
    s = re.sub(r"\\\((.+?)\\\)", r"\1", s)
    # Final whitespace strip
    s = re.sub(r"\s+", "", s)
    return s


def to_number(s: str) -> float | None:
    """Try to parse s as a number (int, float, simple fraction, leading-number)."""
    if s is None:
        return None
    s = s.strip()
    # Strip leading $
    if s.startswith("$"):
        s = s[1:]
    # Plain number
    try:
        return float(s)
    except ValueError:
        pass
    # Simple fraction a/b
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    # \frac{a}{b}
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    # Plain integer with sign
    m = re.fullmatch(r"-?\d+", s)
    if m:
        return float(s)
    # Extract first number from string (e.g. "$125", "18 dollars")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def equal_strict(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    a_n = normalize(a)
    b_n = normalize(b)
    if a_n is None or b_n is None:
        return False
    if a_n == b_n:
        return True
    # Numeric comparison
    af = to_number(a_n)
    bf = to_number(b_n)
    if af is not None and bf is not None:
        return abs(af - bf) < 1e-4
    # Compare further with extra normalizations
    def aggressive(x):
        x = re.sub(r"[\\(){}\[\]]", "", x)
        x = re.sub(r"\\frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", x)
        return x
    return aggressive(a_n) == aggressive(b_n)


def reparse_greedy(path: Path):
    """Re-evaluate a greedy traces jsonl. Returns (n, n_correct, list_of_correctness)."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n_correct = 0
    correctness = []
    for r in rows:
        pred = extract_pred(r.get("output_text", ""))
        gold = r.get("gold")
        ok = equal_strict(pred, gold)
        n_correct += int(ok)
        correctness.append({"id": r.get("id"), "predicted_re": pred,
                              "gold": gold, "correct_re": int(ok)})
    return len(rows), n_correct, correctness


def reparse_sc(path: Path, has_samples=False):
    """Re-evaluate SC traces. If samples are stored, redo majority; else use stored majority_pred."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n_maj = 0; n_any = 0
    fixed_majority = 0
    top1_fracs = []
    new_rows = []
    for r in rows:
        gold = r.get("gold")
        if "samples" in r and isinstance(r["samples"], list):
            # Re-extract from sample texts
            preds = [extract_pred(s) for s in r["samples"]]
            preds_clean = [p for p in preds if p is not None]
            if preds_clean:
                counter = Counter([normalize(p) for p in preds_clean])
                # find raw key matching the most-common normalized
                # for simplicity, just use most-common raw
                raw_counter = Counter(preds_clean)
                top, top_count = raw_counter.most_common(1)[0]
                top1_frac = top_count / max(len(preds), 1)
            else:
                top, top1_frac = None, 0.0
            ok = equal_strict(top, gold)
            any_ok = any(equal_strict(p, gold) for p in preds_clean)
            if int(ok) != r.get("majority_correct", 0):
                fixed_majority += 1
            r["re_majority_pred"] = top
            r["re_majority_correct"] = int(ok)
            r["re_any_correct"] = int(any_ok)
            r["re_top1_frac"] = top1_frac
            top1_fracs.append(top1_frac)
            n_maj += int(ok)
            n_any += int(any_ok)
        else:
            # No sample texts; only re-evaluate stored majority and answer_distribution
            top = r.get("majority_pred")
            ok = equal_strict(top, gold)
            any_ok = any(equal_strict(p, gold) for p in (r.get("answer_distribution") or {}).keys())
            if int(ok) != r.get("majority_correct", 0):
                fixed_majority += 1
            r["re_majority_pred"] = top
            r["re_majority_correct"] = int(ok)
            r["re_any_correct"] = int(any_ok)
            r["re_top1_frac"] = r.get("top1_frac")
            if r.get("top1_frac") is not None:
                top1_fracs.append(r["top1_frac"])
            n_maj += int(ok)
            n_any += int(any_ok)
        new_rows.append(r)
    return new_rows, n_maj, n_any, fixed_majority


def main():
    targets = [
        ("E1_math500_greedy",  EXP / "E1_math500_greedy_traces.jsonl",        EXP / "E1_math500_sc8_traces.jsonl"),
        ("E2_aime_greedy",     EXP / "E2_aime_greedy_traces.jsonl",            EXP / "E2_aime_sc8_traces.jsonl"),
        ("E3_gsm8k_greedy",    EXP / "E3_gsm8k_greedy_traces.jsonl",           EXP / "E3_gsm8k_sc8_traces.jsonl"),
        ("E4_qwen32b",         EXP / "E4_qwen32b_math500_greedy.jsonl",        EXP / "E4_qwen32b_math500_sc.jsonl"),
        ("E6_qwen3_8b",        EXP / "E6_qwen3_8b_greedy.jsonl",               EXP / "E6_qwen3_8b_sc.jsonl"),
        ("E6v2_qwen3_8b_nothink", EXP / "E6v2_qwen3_8b_nothink_greedy.jsonl",  EXP / "E6v2_qwen3_8b_nothink_sc.jsonl"),
        ("E7_qwq32b_greedy",   EXP / "E7_qwq32b_greedy.jsonl",                 EXP / "E7_qwq32b_sc.jsonl"),
        ("E8_phi4",            EXP / "E8_phi4_greedy.jsonl",                   EXP / "E8_phi4_sc.jsonl"),
        ("E9_qwen3_moe",       EXP / "E9_qwen3_moe_greedy.jsonl",              EXP / "E9_qwen3_moe_sc.jsonl"),
        ("E10_mixtral",        EXP / "E10_mixtral_greedy.jsonl",               EXP / "E10_mixtral_sc.jsonl"),
        ("E11_deepseek_v2_lite", EXP / "E11_deepseek_v2_lite_greedy.jsonl",    EXP / "E11_deepseek_v2_lite_sc.jsonl"),
        ("E16_qwen_math_7b",   EXP / "E16_qwen_math_7b_greedy.jsonl",          EXP / "E16_qwen_math_7b_sc.jsonl"),
        ("E17_r1_distill_32b", EXP / "E17_r1_distill_32b_greedy.jsonl",        EXP / "E17_r1_distill_32b_sc.jsonl"),
        ("E18_r1_distill_llama_70b", EXP / "E18_r1_distill_llama_70b_greedy.jsonl", EXP / "E18_r1_distill_llama_70b_sc.jsonl"),
        ("E6v3_qwen3_8b_long",  EXP / "E6v3_qwen3_8b_nothink_greedy.jsonl",        EXP / "E6v3_qwen3_8b_nothink_sc.jsonl"),
        ("E14r_theoremqa_long", EXP / "E14r_theoremqa_greedy.jsonl",                EXP / "E14r_theoremqa_sc.jsonl"),
        ("E13r_olympiad_long",  EXP / "E13r_olympiad_math_greedy.jsonl",            EXP / "E13r_olympiad_math_sc.jsonl"),
        ("E12r_mmlupro_long",   EXP / "E12r_mmlupro_stem_greedy.jsonl",             EXP / "E12r_mmlupro_stem_sc.jsonl"),
    ]
    summary = {}
    for name, gpath, scpath in targets:
        if not gpath.exists():
            continue
        n, ng, _ = reparse_greedy(gpath)
        srows, nmaj, nany, fixed = reparse_sc(scpath, has_samples=False) if scpath.exists() else (None, 0, 0, 0)
        summary[name] = {
            "n": n,
            "greedy_acc_orig": json.loads([l for l in gpath.read_text().splitlines() if l.strip()][0]).get("correct"),
            "greedy_acc_re":  ng / n if n else None,
            "sc_acc_re":      nmaj / n if n and scpath.exists() else None,
            "any_acc_re":     nany / n if n and scpath.exists() else None,
            "n_majority_fixed": fixed,
        }
        print(f"[{name}] greedy={ng}/{n}={ng/n:.3f}  sc_maj={nmaj/n if n else 'NA':.3f}  any={nany/n if n else 'NA':.3f}  fixed={fixed}")
        # Save reparsed sc
        if scpath.exists():
            out = scpath.parent / (scpath.stem + "_re.jsonl")
            with out.open("w") as f:
                for r in srows:
                    f.write(json.dumps(r) + "\n")

    out_summary = EXP / "robust_eval_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"Wrote: {out_summary}")


if __name__ == "__main__":
    main()

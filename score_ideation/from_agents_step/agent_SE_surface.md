# Agent SE — Step-Level Surface / Structural / Textual Features for CoT-CP

**Author:** Agent SE
**Date:** 2026-05-07
**Scope:** Step-level cheap features for CoT-CP — fired at every step boundary, no extra forward passes, no hidden states, no logprobs, no trained ML.

## Design Premises

A *step* is a single chain-of-thought atom — typically a sentence, line, or numbered/bulleted item terminated by `\n`, `\n\n`, `;`, or a numbered prefix (`Step 3:`, `3.`, `(iii)`). All features below assume the boundary detector has already segmented the trace into `step[t]` for `t = 1..T`. We build a *running statistics state* `S` that is updated incrementally as steps stream in, so each feature is `O(|step[t]|)` work amortized.

Features below are designed to be:

- **Streaming-compatible**: usable as a per-step abstain signal *before* the trajectory finishes, so CoT-CP can abort/branch cheaply.
- **Aggregable**: each per-step scalar can be combined into a trajectory-level CP score (max, mean, last, AUC) for the standard split-CP wrapper at the end.
- **Interpretable**: a positive signal points at a hypothesis ("the model hedged", "an equation is unbalanced"), giving a human-readable abstain reason.

I draw on prior surface-feature traditions: lateral hedging from NLI (Vincze et al.), NER-confidence cascades, regex-based fact extraction in QA pipelines, lexical-coherence baselines from MT QE (Specia et al.), and "wait/actually" backtracking as a self-correction marker observed in DeepSeek-R1 traces.

---

## Feature 1: `step_arith_imbalance`

### Definition
For every `=` in `step[t]`, parse a left-hand side (LHS) and right-hand side (RHS) substring, attempt a Python-`eval`-style numerical evaluation under safe rules (digits, `+ - * / ** ( )`, parentheses), and check `|LHS - RHS| / max(1, |LHS|) <= eps`. The feature is the *fraction of `=` occurrences that fail to balance* in the step.

```
step_arith_imbalance = num_unbalanced_equals / max(1, num_equals)
```

If `num_equals == 0`, the feature is `NaN` and is dropped from CP aggregation (use mean-impute).

### Cost
**FREE.** Pure regex + safe `eval` over short substrings. No tokenizer, no model. For a step with `k` equals signs, work is `O(k * avg_expr_len)` which is bounded by the step's character count.

### Hypothesis
Models that drop an arithmetic step or transcribe a digit wrong frequently *write the wrong RHS*; the LHS is usually a faithful copy of the prior subexpression. So a single unbalanced `=` is a strong local correctness signal — much more reliable than the model's own self-report. This is also the basis of "verifier-by-evaluation" tricks in PRMs but at zero cost.

### Pseudocode
```python
import re, ast, operator as op

EQ = re.compile(r"([0-9\.\+\-\*\/\(\)\s]+)=([0-9\.\+\-\*\/\(\)\s]+)")

def safe_eval(expr):
    try:
        node = ast.parse(expr, mode="eval")
        return _eval(node.body)
    except Exception:
        return None

def step_arith_imbalance(step, eps=1e-3):
    matches = EQ.findall(step)
    if not matches: return float("nan")
    bad = 0
    for lhs, rhs in matches:
        l, r = safe_eval(lhs), safe_eval(rhs)
        if l is None or r is None: continue
        if abs(l - r) > eps * max(1.0, abs(l)):
            bad += 1
    return bad / len(matches)
```

### CP Usage
**Per-step abstain.** This is one of the few step-level signals strong enough to *abort early*: if a step has `step_arith_imbalance > 0`, the trajectory is almost certainly wrong (assuming the model meant the equation literally). Calibrate a per-step threshold via split CP on a math benchmark; aggregate to trajectory by max for a safety net.

### Failure Modes
- Models writing definitional `=` (`let x = 5y`, `f(x) = ...`) trigger false positives.
  - **Mitigation:** if either side fails to `safe_eval`, skip that match — definitions naturally fail to evaluate.
- Floating-point inequality (`pi = 3.14`) — eps tolerant evaluation handles this.
- Non-Latin-script math (Chinese/Arabic numerals) — language-specific extension needed.

---

## Feature 2: `step_hedge_ratio`

### Definition
A weighted count of hedging tokens per step, normalized by step token length:

```
step_hedge_ratio = sum_{w in step} hedge_weight(w) / token_count(step)
```

Hedge lexicon (English, with weights):
- weight 1.0: `maybe`, `perhaps`, `possibly`, `might`, `i think`, `i guess`, `not sure`
- weight 0.7: `seems`, `appears`, `likely`, `probably`, `presumably`, `roughly`, `about`, `around`
- weight 0.5: `usually`, `often`, `sometimes`, `tends to`
- weight 1.5 (negative-hedge / explicit doubt): `i don't know`, `unclear`, `confusing`, `not certain`

### Cost
**FREE.** A single pass with an Aho-Corasick automaton or a precompiled regex disjunction over `step[t]`. Even on long steps, microseconds.

### Hypothesis
Hedging in CoT correlates inversely with correctness in factual and arithmetic reasoning — verified in TruthfulQA and SciQ analyses. When a model writes "I think this is around 7," it is *self-reporting* uncertainty in a way that is weak per-token but useful per-step. By normalizing to step length we avoid penalizing long correct steps.

### Pseudocode
```python
HEDGES = {"maybe":1.0, "perhaps":1.0, "possibly":1.0, "might":1.0,
          "seems":0.7, "appears":0.7, "likely":0.7, "probably":0.7,
          "roughly":0.7, "about":0.7, "around":0.7,
          "usually":0.5, "often":0.5, "sometimes":0.5,
          "i think":1.0, "i guess":1.0, "not sure":1.0,
          "i don't know":1.5, "unclear":1.5}

def step_hedge_ratio(step):
    s = step.lower()
    toks = s.split()
    if not toks: return 0.0
    score = 0.0
    for h, w in HEDGES.items():
        score += w * s.count(h)
    return score / len(toks)
```

### CP Usage
**Trajectory-level aggregation.** Per-step thresholding is too noisy (one "maybe" doesn't doom a trace). Aggregate via `mean(hedge_ratio)` and `max(hedge_ratio)` over the trajectory for the CP score. Useful as a *combination feature* with logprob-based agents.

### Failure Modes
- Style-tuned models (e.g., "thoughtful" personas) hedge more without being wrong.
  - **Mitigation:** baseline-subtract using the model's own per-corpus mean.
- Multilingual: lexicon must be ported per language. Detect language at trace start and switch.
- "I think" inside a counterfactual ("if I think X, then ...") is a false positive — accept this noise.

---

## Feature 3: `step_backtrack_marker`

### Definition
Binary or count feature: how many *self-correction markers* appear in the step? Lexicon (case-insensitive, regex with word boundaries):

```
\b(wait|actually|hmm|hold on|on second thought|i made a mistake|
let me reconsider|let me redo|scratch that|never mind|that's wrong|
correction|oops|i was wrong)\b
```

Two variants:
- `step_backtrack_count`: integer count
- `step_backtrack_first`: position (in tokens) of the first marker, normalized by step length (early backtracks more meaningful)

### Cost
**FREE.** One regex scan per step.

### Hypothesis
Backtracking is a double-edged signal: in *strong* reasoners (R1, o1) it correlates with *eventual correctness* (the model self-corrects). In *weaker* reasoners it correlates with *thrashing* and ultimate wrongness. The sign of the correlation is therefore model-dependent — calibrate per model on the held-out CP set. Either way it is informative; the magnitude of the slope varies.

### Pseudocode
```python
import re
BT = re.compile(r"\b(wait|actually|hmm|hold on|on second thought|"
                r"i made a mistake|let me reconsider|let me redo|"
                r"scratch that|never mind|that's wrong|correction|"
                r"oops|i was wrong)\b", re.IGNORECASE)

def step_backtrack_count(step):
    return len(BT.findall(step))

def step_backtrack_first(step):
    m = BT.search(step)
    if not m: return 1.0
    return m.start() / max(1, len(step))
```

### CP Usage
**Both.** Per-step: a single `wait` is mild — don't abort. Trajectory: `sum(backtrack_count) / num_steps` becomes a trace-level "thrash score". Combine with `step_arith_imbalance` — if the model backtracks *and* still leaves arithmetic broken, abort.

### Failure Modes
- Quoted speech containing "wait" produces false positives.
- Models trained on synthetic data may overuse "actually" stylistically without semantic backtrack.
- Cross-lingual: trivial port via translated lexicon.

---

## Feature 4: `step_length_zscore`

### Definition
Running z-score of step token length against the trajectory's running mean and std:

```
mu_t   = mean(len(step[1..t-1]))
sig_t  = std(len(step[1..t-1])) + 1e-3
z_t    = (len(step[t]) - mu_t) / sig_t
step_length_zscore = z_t              # signed
step_length_outlier = |z_t|           # magnitude
```

Optionally also a *prior-based* version using a precomputed corpus mean/std (avoids cold-start at t=2,3).

### Cost
**FREE.** Welford's online algorithm: O(1) update per step. Total state: 3 floats (count, mean, M2).

### Hypothesis
Step lengths in correct CoT are roughly stationary (similar work per step). A sudden very-short step ("So the answer is 42.") near the *middle* of a trace, or a sudden very-long step in an otherwise terse trace, indicates the model either skipped reasoning or padded uncertainty. Both correlate with errors. The *signed* z is interpretable: negative z mid-trace = skipped step; large positive z late = rambling justification.

### Pseudocode
```python
class LenStat:
    def __init__(self):
        self.n, self.mu, self.M2 = 0, 0.0, 0.0
    def update(self, x):
        self.n += 1
        d = x - self.mu
        self.mu += d / self.n
        self.M2 += d * (x - self.mu)
    def std(self):
        return (self.M2 / max(1, self.n - 1)) ** 0.5

def step_length_zscore(step, stat):
    L = len(step.split())
    if stat.n < 2:
        z = 0.0
    else:
        z = (L - stat.mu) / max(1e-3, stat.std())
    stat.update(L)
    return z
```

### CP Usage
**Per-step abstain** for `|z| > 3` only (extreme outliers). Trajectory-level: `max(|z_t|)` and `mean(|z_t|)` as features fed into the trajectory CP score.

### Failure Modes
- First two steps always have `z=0` — prior helps.
- Multi-modal step types in the same trace (math + prose) inflate variance, depressing z. Acceptable.

---

## Feature 5: `step_repetition_ngram`

### Definition
Maximum 4-gram overlap between `step[t]` and `step[t-1]` and (separately) the trace prefix `step[1..t-1]`:

```
ngram_overlap_prev   = |4grams(step[t]) ∩ 4grams(step[t-1])| / |4grams(step[t])|
ngram_overlap_global = |4grams(step[t]) ∩ 4grams(step[1..t-1])| / |4grams(step[t])|
longest_copy_len     = length of longest exact substring of step[t] that appears in step[t-1]
```

The cheapest variant uses 4-gram set-intersection over hashed n-grams.

### Cost
**FREE.** Maintain a rolling set of seen 4-grams; per-step update is O(|step|). For `longest_copy_len`, a suffix-automaton over `step[t-1]` runs in O(|step[t-1]| + |step[t]|).

### Hypothesis
Two distinct failure modes manifest as repetition:
1. **Degenerate looping** — model outputs near-duplicate steps because it's stuck. Strong negative signal.
2. **Restating premises** — useful repetition of the problem. Weak/neutral signal.

The discriminator is `ngram_overlap_global` (which catches degeneracy) vs. just `ngram_overlap_prev` (which can be benign). The pair separates them.

### Pseudocode
```python
def kgrams(text, k=4):
    toks = text.split()
    return {tuple(toks[i:i+k]) for i in range(max(0, len(toks)-k+1))}

class NgramState:
    def __init__(self): self.seen = set(); self.prev = set()

def step_repetition(step, state):
    cur = kgrams(step)
    if not cur: return 0.0, 0.0
    prev_overlap = len(cur & state.prev) / len(cur)
    glob_overlap = len(cur & state.seen) / len(cur)
    state.seen |= cur
    state.prev = cur
    return prev_overlap, glob_overlap
```

### CP Usage
**Per-step abstain** if `ngram_overlap_global > 0.8` and step is not the first — strong degeneracy. Trajectory-level: `max(ngram_overlap_global)` is a direct dropout-into-loop detector.

### Failure Modes
- Math traces legitimately repeat formulas — set k=5 or 6 for math.
- Code traces repeat function names, inflating overlap. Tokenize on identifiers.

---

## Feature 6: `step_numeric_density`

### Definition
A vector of cheap counts normalized by step length:

```
num_count        = #digit-runs in step
op_count         = #(+, -, *, /, =, <, >, ^) in step
unique_num_ratio = #unique numbers / #total numbers
new_num_count    = #numbers in step[t] not seen in step[1..t-1]
```

Plus a derived signal:

```
step_num_drift = new_num_count / max(1, num_count)
```

i.e., what fraction of numbers in this step are *fresh* versus carried-forward.

### Cost
**FREE.** One regex pass `\b\d+(?:\.\d+)?\b`, plus an integer set lookup against numbers seen so far.

### Hypothesis
In multi-step arithmetic, numbers should *flow*: most numbers in step[t] are derived from step[t-1]. A step that introduces many *new* numbers without setup signals a hallucinated quantity. Conversely, a step with zero numbers in a numeric problem is a "talking-around-the-problem" step. The `new_num_count` feature is essentially a poor-man's variable-tracking signal at near-zero cost.

### Pseudocode
```python
import re
NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
OPS = set("+-*/=<>^")

class NumState:
    def __init__(self): self.seen = set()

def step_numeric(step, state):
    nums = NUM.findall(step)
    ops = sum(1 for c in step if c in OPS)
    new = [n for n in nums if n not in state.seen]
    state.seen.update(nums)
    L = max(1, len(step.split()))
    return {"num_density": len(nums)/L,
            "op_density":  ops/L,
            "new_num_ratio": len(new)/max(1,len(nums)) if nums else 0.0,
            "uniq_num_ratio": len(set(nums))/max(1,len(nums)) if nums else 0.0}
```

### CP Usage
**Trajectory-level.** No single step is damning; aggregate across the trace. Pair with `step_arith_imbalance`: a step that adds many new numbers AND fails an equation balance is highly suspect (combine multiplicatively).

### Failure Modes
- Word-form numbers ("seventeen") are missed. Add a word-to-digit pass for English if needed.
- Date strings, version numbers inflate counts. Heuristic filter for `\d{4}` followed by `-` etc.

---

## Feature 7: `step_branch_marker`

### Definition
Detect whether the model is *opening a case split* or *enumerating alternatives* — a structural marker of internal uncertainty:

```
\b(case (one|two|1|2|i|ii)|either|or alternatively|on one hand|
on the other hand|alternatively|option (a|b|1|2)|
let's consider|first possibility|second possibility|
suppose (that)?|assuming|if .* then.*else|either way)\b
```

Two scalars:
- `step_branch_open`: 1 if step opens a branch, 0 otherwise.
- `step_branch_unresolved`: running count of opened-but-not-closed branches (closures = `therefore`, `thus`, `in conclusion`, `hence`, `so the answer`, `combining both`).

### Cost
**FREE.** Two regex scans per step + integer counter.

### Hypothesis
A step that opens a case split increases the *future* probability of error in two ways: (a) more reasoning branches = more places to fail; (b) opening unprompted suggests the model didn't see a unique deterministic path. `step_branch_unresolved > 0` near the trace's end is a critical red flag — branches must be closed for the answer to be valid.

### Pseudocode
```python
import re
OPEN = re.compile(r"\b(case [12i]+|case (one|two)|either|alternatively|"
                  r"on (one|the other) hand|let's consider|"
                  r"suppose( that)?|assuming|first possibility|"
                  r"second possibility)\b", re.IGNORECASE)
CLOSE = re.compile(r"\b(therefore|thus|hence|in conclusion|"
                   r"so the answer|combining both|either way)\b",
                   re.IGNORECASE)

class BranchState:
    def __init__(self): self.open = 0

def step_branch(step, state):
    o = len(OPEN.findall(step))
    c = len(CLOSE.findall(step))
    state.open = max(0, state.open + o - c)
    return {"branch_open": int(o > 0),
            "branch_unresolved": state.open}
```

### CP Usage
**Both.** Per-step: if a step *contains the final answer marker* AND `branch_unresolved > 0`, abstain — the model concluded without resolving its own case split. Trajectory: max(`branch_unresolved`) over the trace.

### Failure Modes
- Quoted/hypothetical content ("if I had a dollar...") triggers false opens.
- Some closure markers ("therefore") are colloquial filler. Tune lexicon per model corpus.

---

## Feature 8 (bonus): `step_unigram_surprise`

### Definition
Step-level lexical-surprise approximation using a *precomputed* unigram model `P(w)` over a domain corpus (e.g., MATH train, GSM8K train):

```
step_unigram_surprise = -mean_{w in step} log P(w)
```

Optional: also `step_oov_ratio = #(w not in vocab) / |step|`.

### Cost
**FREE.** A single hash lookup per token. The unigram model is a `dict[str, float]` precomputed offline; loading is one-time cost (~MB).

### Hypothesis
Steps that drift away from the domain's lexical distribution (high surprise, many OOV) are more likely to be hallucinations or off-topic ramblings. This is a poor man's domain-adaptation signal that requires no model forward pass — directly inspired by KenLM / SRILM-style features used in MT QE for two decades.

### Pseudocode
```python
import math
def step_unigram_surprise(step, unigram_logp, default=-15.0):
    toks = step.lower().split()
    if not toks: return 0.0
    s = sum(unigram_logp.get(t, default) for t in toks) / len(toks)
    return -s
```

### CP Usage
**Trajectory-level**, mean over steps. Combine with `step_length_zscore` — steps that are both long and high-surprise are particularly red-flagged.

### Failure Modes
- Domain-shift between calibration set and test inputs causes spurious high surprise. Update the unigram model per domain.
- Code/symbol tokens dominate the tail of P(w). Consider character-level fallback.

---

## Combination & CP Wrapper Notes

For CoT-CP we recommend a two-tier integration:

1. **Per-step gating layer.** Use `step_arith_imbalance`, `step_repetition_ngram (global)`, and `step_branch_unresolved-at-conclusion` as *hard* per-step abstain triggers — each with its own split-CP-calibrated threshold. These three are the highest-precision signals and justify early termination.

2. **Trajectory CP score.** Combine the remaining features (hedge ratio, backtrack count, length z-scores, numeric density, unigram surprise) into a vector that is *aggregated* over the trace via `[mean, max, last]` pooling. Concatenate with the trajectory-level score from logprob agents and run split CP on the aggregate.

This gives CoT-CP both the *speed* benefit of step-level abort (saving compute) and the *coverage* guarantee of trajectory-level CP, while keeping every feature O(token count) per step with no model invocations.

## Interpretability Hooks

Each feature exposes a clear *failure-mode hypothesis* that can be surfaced to users when CP abstains:

| Feature | Abstain reason given to user |
|---|---|
| `step_arith_imbalance` | "Arithmetic step did not balance: `12 + 7 = 18`." |
| `step_hedge_ratio` | "Model hedged repeatedly in this step." |
| `step_backtrack_marker` | "Model self-corrected mid-trace." |
| `step_length_zscore` | "Step length is anomalous vs. trajectory norm." |
| `step_repetition_ngram` | "Step repeats a previous step verbatim." |
| `step_numeric_density` | "Many new numbers introduced without derivation." |
| `step_branch_marker` | "Trajectory ended with unresolved case split." |
| `step_unigram_surprise` | "Step's vocabulary drifts from domain." |

This is critical for selective-prediction UX: a CP system that abstains *with a reason* is markedly more trustworthy than a black-box reject.

## Final Remarks

All eight features above (seven primary + one bonus) cost no forward passes, no logprobs, no hidden states, and no trained components. Each can be swapped in or out independently, allowing CoT-CP to compose them with the heavier feature families produced by sibling agents. Most are language-dependent only via small lexicons that can be ported in under an hour per language, and the `step_arith_imbalance` and `step_repetition_ngram` features are language-agnostic outright. We expect the strongest individual signal to be `step_arith_imbalance` on math benchmarks and `step_repetition_ngram (global)` on long-horizon reasoning (where degeneracy dominates failures), with the lexical features `step_hedge_ratio` and `step_backtrack_marker` providing useful late-stage trajectory aggregations.

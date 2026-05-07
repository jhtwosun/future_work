# Agent C (external-verifier angle): Five new score functions for CoT-CP

**Author identifier:** Agent C — external-verifier / symbolic-feedback angle. Companion to Agent A (intrinsic confidence) and Agent B (PRM/process).

The three baselines (`lp_min`, `prm_min`, `sc_top1`) all answer one question: *what does the model think about its trajectory?* They share one failure mode: when the model is **confidently wrong** (systematic algebraic slip, off-by-one, hallucinated theorem), all three return arbitrarily high scores. External verifiers are powerful here: a sympy `equals` call or a Python `assert` is correlated with truth in a way the model's own logits are not.

All compute costs are multiples of one greedy decode of the trajectory being scored. Verifiers themselves are essentially free (sympy ~1 ms, Python `exec` ~10 ms, embedding lookup ~1 ms); the dominant cost is whatever extra **decode** the score needs.

---

## C1. `verifier_step_pass` — claim-by-claim symbolic verification

**What is verified:** every individual intermediate equation / numeric claim emitted in the chain of thought (NOT just the final answer).

**Verifier type:** sympy (`sympify`, `simplify`, `Eq(a,b)`, `equals`) for math; Python `ast` + `exec` in a sandbox for code; regex extraction first.

**Mathematical definition.** Let the trajectory have steps `s_1..s_T`. An extractor yields checkable claims `C_t` per step (equations `LHS = RHS` or numeric assertions). For each claim, `v(c) = 1` iff sympy proves `LHS - RHS == 0` (or numerically close within tol). Let `N = sum_t |C_t|`. Define

```
verifier_step_pass = (1/N) * sum_{t,j} v(c_{t,j})    if N > 0
                   = lp_min(traj)                    if N == 0  (fallback)
```

A weighted variant uses weights `w_t` proportional to position-from-end (later steps matter more) or a PRM weight.

**Compute cost.** ~1.0–1.05× — no extra decoding; sympy work is tens of ms per trajectory.

**Intuition.** Catches *local* arithmetic / algebraic mistakes the policy flows through confidently. The classic "model wrote `7 * 8 = 54`, then continued correctly from 54" is invisible to lp_min (high logprob), often invisible to PRM (PRM evaluates plausibility, not truth), and survives SC if the slip is systematic. Sympy catches it instantly.

**Implementation sketch.**
```python
import re, sympy as sp
EQ_RE = re.compile(r'([^\n=]{1,80})=([^\n=]{1,80})')
def verify_step(text):
    claims, hits = 0, 0
    for lhs, rhs in EQ_RE.findall(text):
        try:
            L, R = sp.sympify(lhs), sp.sympify(rhs)
            claims += 1
            if sp.simplify(L - R) == 0: hits += 1
        except Exception: pass
    return claims, hits
```
Code variant: extract `# expect:` comments or final `print(...)` and `exec` in a 1-second subprocess.

**Hypothesized Pareto position.** ~1.05× — expected to dominate `lp_min` on math (strictly more info, free) and complement `prm_min`: lower variance on arithmetic, higher on prose-heavy reasoning where claims aren't extractable.

**Coverage.** *Catches:* arithmetic slips, sign errors, factoring mistakes, runtime exceptions in code. *Misses:* strategy-level errors where every equation is locally correct but the wrong sub-problem is being solved; prose steps with no extractable equation.

**Datasets:** GSM8K (very strong), MATH-500 (strong), AIME (strong, extraction harder), OlympiadBench (medium, prose-heavy), TheoremQA (medium), HumanEval (AST + exec variant), MMLU-Pro (weak).

---

## C2. `final_answer_recheck` — cheap reverse-direction sanity check

**What is verified:** the *final answer* by plugging it back into the original problem statement (a "substitute and check" pass).

**Verifier type:** sympy for math equations / inequalities; Python execution against held-out test inputs for code; multiple-choice elimination for MCQ.

**Mathematical definition.** Let `a*` be the extracted final answer and `Q` a structured form of the question. Define `R(Q, a*) in [0, 1]`:

- Math equation: `R = 1` iff `simplify(Q.subs(x, a*)) == 0`.
- Math word problem: parse stated constraints into sympy relations; score = fraction satisfied.
- Code: `R = (# property-test passes) / (# tests)` over a Hypothesis-style fuzzer on the signature.
- MCQ: `R = 1` iff every other option fails an extracted constraint, else `1 - (# other options also passing) / (K-1)`.

```
final_answer_recheck = R(Q, a*)
```

Combine with a tiebreaker: `0.5 * R + 0.5 * lp_mean` keeps ranking signal when `R` is binary.

**Compute cost.** 1.0× (no extra decode). ~10 property tests with a 200 ms timeout is essentially free.

**Intuition.** The textbook "check your answer" step — orthogonal to forward generation. lp_min, prm_min and sc_top1 all share a forward bias; substitution is symmetric and cannot be fooled by it.

**Implementation sketch.**
```python
def recheck_math(question_eq, ans):
    x = sp.Symbol('x'); eq = sp.sympify(question_eq)
    return int(sp.simplify(eq.subs(x, sp.sympify(ans))) == 0)
def recheck_code(fn_src, sig, n=10):
    g = {}; exec(fn_src, g); fn = g[sig.name]
    inputs = [sample_input(sig) for _ in range(n)]
    return sum(safe_call(fn, x) is not _ERROR for x in inputs) / n
```

**Hypothesized Pareto position.** ~1.0× — same compute as `lp_min`, expected higher accuracy on tasks where substitution is well-defined (equations, code with a runnable signature). Complements C1: C1 checks the journey, C2 the destination.

**Coverage.** *Catches:* extraneous roots, off-by-one, wrong sign, wrong unit, code that compiles but fails on basic inputs, MCQ where the chosen option violates a stated constraint. *Misses:* open-ended correctness criteria (proofs, "explain why"); unparseable `Q`.

**Datasets:** GSM8K (strong with constraint extraction), MATH-500 (strong on equations/inequalities, weaker on proofs), AIME (strong — integer answers substitute cleanly), OlympiadBench (mixed), TheoremQA (medium), MMLU-Pro (limited), HumanEval (very strong via property-test fuzzing).

---

## C3. `sc_verifier_pass` — self-consistency *gated by* a verifier

**What is verified:** K independent samples are each run through an external verifier; the score is the verifier-pass fraction, not the answer-agreement fraction.

**Verifier type:** any of the above (sympy, Python exec, retrieval). Agnostic.

**Mathematical definition.** Sample `K` trajectories `tau_1, ..., tau_K` (temperature > 0, including the trajectory being scored as one of them). For each, run a verifier `V(tau_k) in {0, 1}` (final-answer check, e.g. C2-style). Let `a*` be the answer of the trajectory we are scoring. Define

```
sc_verifier_pass = ( #{k : V(tau_k) = 1 AND answer(tau_k) == a*} ) / K
```

This is *agreement on a verified answer*: a unanimous majority that all fail the verifier scores 0.

**Compute cost.** K× decode, same as `sc_top1` (K=8 → 8×); verifier overhead negligible. Same compute, more signal.

**Intuition.** `sc_top1` is high whenever K samples agree, even if they share a bias. `sc_verifier_pass` requires agreement *and* external validation, down-weighting the exact failure mode where SC is most overconfident.

**Implementation sketch.**
```python
verified = [(answer(t), V(t)) for t in samples]
score = sum(a == a_star and ok for a, ok in verified) / K
```

**Hypothesized Pareto position.** Same x-axis as `sc_top1` (≈4×–8×), strictly dominant where the verifier is reliable. Bends the Pareto frontier downward at the SC compute point — especially on AIME / MATH-500 / HumanEval.

**Coverage.** *Catches:* systematic forward-pass biases (where SC is most dangerous); low-confidence-but-correct trajectories. *Misses:* tasks with incomplete verifiers (proofs); inherits SC variance for small K.

**Datasets:** GSM8K, MATH-500, AIME, OlympiadBench (final-answer parts), HumanEval (verifier = test suite). Limited: MMLU-Pro, TheoremQA proofs.

---

## C4. `backward_consistency` — reverse-direction trajectory check

**What is verified:** that the *question can be reconstructed* from the answer plus the reasoning tail. A correct answer makes the question recoverable; a hallucinated one doesn't.

**Verifier type:** tiny LLM (1B–3B local model) as backward decoder + sentence embedding for similarity.

**Mathematical definition.** Given question `Q`, final answer `a*`, last `m` steps `S_end`, prompt the small model to reconstruct: `Q_hat = SmallLM("Answer is a*, reasoning was S_end, what was the question?")`. With sentence embedding `phi`, define `backward_consistency = cos(phi(Q), phi(Q_hat))`.

A symbolic variant for math: ask the small model to recover stated parameters from `a*` and check via sympy.

**Compute cost.** One small-model decode + embedding pair. 1B at ~1/10 base cost → ~1.10–1.20× total.

**Intuition.** Catches hallucinated answers that look right forward but bear no relation to the question. lp/PRM/SC only ever read forward; cycle-consistency (cf. backtranslation, "ask-to-yourself") is the natural fix.

**Implementation sketch.**
```python
prompt = f"Answer: {a_star}\nReasoning tail: {tail}\nReconstruct the question."
Q_hat = small_lm.generate(prompt, max_new_tokens=128)
score = util.cos_sim(embed(Q), embed(Q_hat)).item()
```
For numeric tasks replace cosine with sympy equality on extracted parameters.

**Hypothesized Pareto position.** ~1.15× — slots **between `lp_min` (1×) and `prm_min` (2×)**, the gap the prompt highlights. Comparable accuracy to `prm_min` on final-answer tasks at half the cost; weaker than `prm_min` on stepwise quality.

**Coverage.** *Catches:* hallucinated answers with plausible-but-unrelated reasoning; answer-anchored generation. *Misses:* mutually-consistent-but-both-wrong cases (mild misreading); small-LM failures.

**Datasets:** GSM8K, MATH-500 (numeric variant strong), MMLU-Pro (language variant — strong, options are short), TheoremQA, AIME, OlympiadBench. HumanEval variant: reconstruct the docstring from the function body, compare embeddings.

---

## C5. `cross_reform_agreement` — cross-question reformulation consistency

**What is verified:** that the answer is invariant under semantically-preserving paraphrases of the question. A robust solver should produce the same answer to "Alice has 12 apples, gives Bob 1/3 …" and "If Alice gives Bob a third of her 12 apples …". A confidently-wrong solver typically isn't.

**Verifier type:** small LLM for reformulation, sympy / Python exec for answer equivalence checking.

**Mathematical definition.** Generate `M` reformulations `Q_1..Q_M = Reform(Q)`; run the same policy greedily on each to get `a_1..a_M`. Define `eq(a_i, a*)` via sympy for math, canonicalised string match for MCQ, behavioural equivalence on a fuzzed test set for code.

```
cross_reform_agreement = (1/M) * sum_i eq(a_i, a*)
```

**Compute cost.** `M` reformulations (~0.1× each from a small LM) + `M` base-model decodes ≈ `M × 1.1`. M=3 → ~3.3×. Same regime as small-K SC, but variation is *across the question*, an orthogonal source of randomness.

**Intuition.** SC marginalises over decoding noise; cross-reformulation marginalises over *surface form*. Many real failure modes are surface-form-locked: the model latches onto a phrasing. SC, PRM and lp share blindness here because they all read one prompt. This is the only score in the list that probes *understanding* rather than execution.

**Implementation sketch.**
```python
reforms = small_lm.generate(f"Reword without changing meaning: {Q}", n=M)
answers = [base_lm.greedy(r) for r in reforms]
score = sum(sympy_equiv(a, a_star) for a in answers) / M
```

**Hypothesized Pareto position.** ~3–4× — right of `prm_min`, left of large-K SC. Should dominate `sc_top1` at matched compute on surface-form-sensitive tasks (MMLU-Pro, OlympiadBench).

**Coverage.** *Catches:* surface-form-locked errors, phrasing-dependent comprehension failures, numeric-vs-symbolic mode confusion. *Misses:* errors that survive paraphrase (genuine domain misunderstanding, systematic algebra slips); reformulator drift.

**Datasets:** GSM8K (medium), MATH-500 (low — questions already minimal), AIME / OlympiadBench (medium-high), MMLU-Pro (high — surface form matters most), TheoremQA (medium), HumanEval (paraphrase the docstring; check resulting function still passes hidden tests).

---

## Summary table

| Name | Verifier | Compute | Best datasets | Catches |
|---|---|---|---|---|
| `verifier_step_pass` (C1) | sympy / exec on every step | 1.05× | GSM8K, MATH-500, AIME, HumanEval | Per-step arithmetic / runtime errors |
| `final_answer_recheck` (C2) | sympy substitute / fuzz tests | 1.0× | GSM8K, MATH-500, AIME, HumanEval | Final-answer slip, extraneous roots, MCQ constraint violations |
| `sc_verifier_pass` (C3) | per-sample verifier + agreement | K× (≈4–8×) | GSM8K, MATH-500, AIME, HumanEval | SC's confidently-wrong consensus |
| `backward_consistency` (C4) | tiny LM + embedding / sympy | 1.15× | GSM8K, MATH-500, MMLU-Pro, TheoremQA | Hallucinated answers, answer-anchored reasoning |
| `cross_reform_agreement` (C5) | small LM reformulation + verifier | 3–4× | MMLU-Pro, OlympiadBench, GSM8K, HumanEval | Surface-form lock, comprehension brittleness |

## Cross-cutting design notes

- **Fallbacks.** Each score needs a deterministic fallback when the verifier input is undefined (no equations, no runnable code, no parseable MCQ). Cleanest convention: fall back to `lp_min`, preserving CP exchangeability.
- **Calibration.** C2/C3 are often binary and produce ties, hurting CP threshold resolution. Always add a continuous tiebreaker: `score = R + eps * lp_mean`.
- **Timeouts.** Wrap all sympy calls in a 250 ms timeout (`signal.alarm` or thread-pool).
- **Soundness, not completeness.** Verifiers needn't catch every error; they need to be sound on positives — when they say PASS they're usually right. CP absorbs the rest as residual risk.
- **Pareto summary.** C2 strictly dominates `lp_min` on math/code (same compute, more signal). C4 fills the lp/prm gap. C1 should dominate `prm_min` on arithmetic at half the cost. C3 bends the SC point downward. C5 is most dataset-sensitive — a clear win on MMLU-Pro, possibly a wash on MATH-500.

The five span three axes orthogonal to lp/PRM/SC — *symbolic truth* (C1, C2, C3), *backward direction* (C4), and *surface-form invariance* (C5) — which is exactly where intrinsic-confidence scores tend to be miscalibrated.

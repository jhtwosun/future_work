# Agent SD: Step-Level Self-Prompting / Per-Step Verifier Methods for CoT-CP

**Author:** Agent SD
**Scope:** Cheap step-level scores `s_t = f(question, step_{1..t})` that fire at every step boundary. Each method below adds one short forward pass (or one `prompt_logprobs` call in vLLM) per step, never requires a full trace, and never trains a verifier.

**Notation.** Let `Q` = question, `R_{<t}` = reasoning prefix through step `t-1`, `r_t` = the step just emitted, and `S_t = R_{<t} ++ r_t` the running prefix. We append a short *probe* `p` and inspect logits/logprobs of one or two answer tokens. The "yes" bias of LLMs is well-known; we therefore prefer **logit differences** `Δ = logit("yes") - logit("no")` over absolute probabilities, and we recommend **affine calibration** of `Δ` on a held-out set before passing to split-CP.

For CP, all step scores are aggregated into a trajectory score
`S(Q, trace) = ψ(s_1, ..., s_T)` with `ψ ∈ {min, mean, max-clip, last-k-mean}` chosen on the calibration split. Split conformal then yields a question-level coverage guarantee.

---

## Method 1. `step_counterfactual_swap` — "Would a wrong step fool you?"

**Idea.** Self-eval prompts ("is this step correct?") have a "yes" bias. Calibrate it *per step* by asking the same question after a **deliberate corruption** of `r_t` and using the *contrast* as the score. Cheap because we only need 2 short forward passes.

**Probe template (clean):**
```
{Q}
{R_<t}
{r_t}
Q: Is the LAST step internally consistent with the previous steps and the problem? Answer yes or no.
A:
```

**Probe template (corrupted) — same prompt but `r_t` is replaced by `r̃_t`,
which we form by negating the first numeric/relational token in `r_t`
(e.g. `=` → `≠`, `>` → `<`, flip sign of one number).**

**Score formula:**
`s_t = (logit_clean("yes") - logit_clean("no")) - (logit_corrupt("yes") - logit_corrupt("no"))`
A model that *uniformly* says "yes" cancels out; only contrastive signal survives.

**Cost.** 2 short forward passes per step. With vLLM `prompt_logprobs=5`, each pass is the prefix length only, no generation.

**Hypothesis.** Step-eval prompts are biased; the *gap* between clean-step yes-prob and corrupt-step yes-prob measures whether the model is *sensitive to the actual content* of `r_t`. Sharper gap ⇒ model has signal ⇒ step is more likely correct (or at least more meaningful).

**Pseudocode.**
```python
def step_counterfactual_swap(model, Q, R_prev, r_t):
    r_corrupt = negate_first_relational_or_number(r_t)
    p = "\nQ: Is the LAST step internally consistent...? Answer yes or no.\nA:"
    clean   = model.logits(Q + R_prev + r_t      + p)
    corrupt = model.logits(Q + R_prev + r_corrupt + p)
    yt, nt  = tok("yes"), tok("no")
    d_clean   = clean[yt]   - clean[nt]
    d_corrupt = corrupt[yt] - corrupt[nt]
    return d_clean - d_corrupt
```

**CP usage.** Per-trajectory aggregate `ψ = mean(s_t)`. Use sign + magnitude:
small/negative aggregates flag low-confidence traces. Conformal threshold
`q̂_α` is computed on calibration `S(Q,·)`; new trajectories are accepted iff
`S ≤ q̂_α`.

**Failure modes.** (i) Corruption operator may flip a token that does not change semantics ("the" → "no"); use a *targeted* operator that touches arithmetic/logical tokens. (ii) On purely textual steps with no numbers/relations, the contrast collapses to ~0; in that case the score should default to a fall-back metric (e.g., Method 3).

---

## Method 2. `step_next_token_surprise` — entropy spike under the model's own continuation

**Idea.** After step `t`, ask the model to start step `t+1` with a single-token *cue* drawn from a fixed vocabulary (`So`, `Therefore`, `Thus`, `Then`). Read the **predictive entropy** over the next token *given* that cue. A confident prefix should yield low-entropy continuations; a buggy step often produces flailing, high-entropy distributions.

**Probe template:**
```
{Q}
{R_<t}
{r_t}
Therefore,
```

**Score formula:** `s_t = H(p(· | prefix ++ "Therefore,"))` (Shannon entropy in nats over top-K logprobs, K=20). Higher = worse. Optionally use the **logprob of the realized next token** when continuing the trace, which is free (already computed during generation).

**Cost.** 1 short forward pass at each step boundary, OR effectively 0 if we reuse the logprobs the model emitted while continuing the trace. No generation, no probe-specific tokens beyond the cue.

**Hypothesis.** Internal coherence reveals itself as low-entropy continuations. A wrong arithmetic step destabilizes the joint distribution because the next valid sentence becomes ambiguous (the model "doesn't know what to say next").

**Pseudocode.**
```python
def step_next_token_surprise(model, Q, R_prev, r_t):
    prefix = Q + R_prev + r_t + "\nTherefore,"
    logprobs = model.logprobs(prefix, top_k=20)
    p = softmax(np.array(list(logprobs.values())))
    H = -(p * np.log(p + 1e-12)).sum()
    return H
```

**CP usage.** Per-step score aggregated by `ψ = max(s_t)` — a single high-entropy step poisons the trace. Calibrate `q̂_α` on `max`-aggregated scores.

**Failure modes.** (i) Entropy is naturally higher near step starts that branch into multi-line subroutines (e.g., enumerations), causing false positives; mitigate with **position-conditioned z-scoring** computed on the calibration set. (ii) A confidently-wrong step (model is sure of its mistake) yields low entropy and slips through.

---

## Method 3. `step_inverse_question_logprob` — can the question be reconstructed?

**Idea.** A genuine reasoning step preserves the *information content* of the original question. If `Q` is irrecoverable from `S_t` alone, the chain has either drifted off-topic or compressed away essential constraints. Score = the conditional logprob of `Q` given `S_t` (no generation).

**Probe template:**
```
{R_<t}
{r_t}
The original question this reasoning was answering was:
{Q}
```
We compute logprob of just the `{Q}` continuation — vLLM's `prompt_logprobs` returns per-token logprobs over the whole prompt in one pass.

**Score formula:** `s_t = -(1/|Q|) Σ_i log p(Q_i | R_<t ++ r_t ++ "...:")`
Length-normalized negative logprob; smaller = better.

**Cost.** 1 forward pass, prefix-length only. With KV-cache reuse across steps, the marginal cost per step is ≈ `|r_t| + |Q|` tokens, well under 2× compute.

**Hypothesis.** Correct steps are *bidirectional* — you can deduce the question from a partial solution because the partial solution is "about" the question. Hallucinated or off-task steps lose this property.

**Pseudocode.**
```python
def step_inverse_question_logprob(model, Q, R_prev, r_t):
    cue = "\nThe original question this reasoning was answering was:\n"
    full = R_prev + r_t + cue + Q
    per_tok = model.prompt_logprobs(full)
    Q_lp = per_tok[-len_tokens(Q):]
    return -np.mean(Q_lp)   # smaller = better
```

**CP usage.** Aggregate as `ψ = mean(s_t)` — mean conditional NLL on `Q`. Conformal nonconformity score = the aggregate; trajectories whose mean NLL exceeds `q̂_α` are flagged.

**Failure modes.** (i) Long questions dominate; length-normalize. (ii) Models with strong question-prior copying (memorized benchmark) score artificially well; mitigate by using a *paraphrase* of `Q` instead of the literal text.

---

## Method 4. `step_redundancy_drop` — does the step add information?

**Idea.** A *useful* step changes the model's belief about the answer. We measure the KL divergence between the next-step distribution **with** vs **without** `r_t`. If `r_t` doesn't move the distribution, it is filler; if it moves it *a lot*, it might be hallucinated detour. We score the **absolute log-ratio at the end-of-answer token** rather than full KL.

**Probe template:**
```
{Q}
{R_<t}
{r_t}
Final answer:
```
and the same prompt with `r_t` removed.

**Score formula:**
`s_t = | log p(answer_first_token | with r_t) - log p(answer_first_token | without r_t) |`
We threshold from *both* sides: too-small ⇒ redundant; too-large ⇒ disruptive.

**Cost.** 2 short prefix forward passes per step; with KV cache only the suffix differs, so marginal cost ≈ 1× step compute.

**Hypothesis.** Useful steps push the answer-token distribution *moderately* — they incrementally narrow the answer. A flat distribution after a step ⇒ the step did nothing. A wildly different distribution ⇒ likely a hallucinated jump.

**Pseudocode.**
```python
def step_redundancy_drop(model, Q, R_prev, r_t):
    cue = "\nFinal answer:"
    with_r    = model.logprobs(Q + R_prev + r_t + cue, top_k=1)
    without_r = model.logprobs(Q + R_prev       + cue, top_k=1)
    return abs(top1(with_r) - top1(without_r))
```

**CP usage.** Two-sided. Convert to a conformal score by a learned monotone transform `g: ℝ → ℝ_+` fit on calibration data so that `g(s_t)` is small for correct traces. Aggregate `ψ = mean(g(s_t))`.

**Failure modes.** (i) Steps that are *correct rephrasings* (no info gain) get penalized; mitigate by allowing a band `[ε_lo, ε_hi]` learned on calibration. (ii) Numerical instability for tiny logprobs near eps.

---

## Method 5. `step_dual_polarity_yesno` — bias-cancelling self-eval

**Idea.** Vanilla `step_self_yesno` is contaminated by the "yes" prior. Ask the *same* question in two polarities and average the contradictions away.

**Probe templates (run both):**
```
P+:  "{Q}\n{R_<t}\n{r_t}\nQ: Is this step CORRECT? Answer yes or no.\nA:"
P-:  "{Q}\n{R_<t}\n{r_t}\nQ: Does this step contain a MISTAKE? Answer yes or no.\nA:"
```

**Score formula:**
`Δ+ = logit_P+("yes") - logit_P+("no")`
`Δ− = logit_P-("yes") - logit_P-("no")`
`s_t = (Δ+ - Δ−) / 2`
A model with pure "yes" bias gives `Δ+ = Δ−`, yielding `s_t = 0`. Real signal survives because the *truth-conditional* answer flips polarity but the bias does not.

**Cost.** 2 short forward passes per step; both prefixes share the long shared prefix `Q + R_<t + r_t`, so KV-cache reuse makes the second pass marginal-cost ≈ probe length only (~20 tokens).

**Hypothesis.** Bias is polarity-invariant; truth signal is polarity-anti-invariant. Their algebraic difference isolates the latter.

**Pseudocode.**
```python
def step_dual_polarity_yesno(model, Q, R_prev, r_t):
    base = Q + R_prev + r_t
    p_pos = "\nQ: Is this step CORRECT? Answer yes or no.\nA:"
    p_neg = "\nQ: Does this step contain a MISTAKE? Answer yes or no.\nA:"
    L_pos = model.logits(base + p_pos)
    L_neg = model.logits(base + p_neg)
    yt, nt = tok("yes"), tok("no")
    d_pos = L_pos[yt] - L_pos[nt]
    d_neg = L_neg[yt] - L_neg[nt]
    return 0.5 * (d_pos - d_neg)   # higher ⇒ more confident step is correct
```

**CP usage.** Aggregate `ψ = -mean(s_t)` (so smaller is "better" trace). Calibrate threshold via split-CP on the calibration set's correct/incorrect labels (or ground-truth-free residuals if labels unavailable).

**Failure modes.** (i) Models can be consistently biased *toward* mistake-detection ("Does this step contain a mistake?" → almost always yes) — both Δ values shift, but the *difference* still works as long as the bias is symmetric. (ii) Tokenizer quirks: ensure `tok("yes")` and `tok("Yes")` are both summed.

---

## Method 6. `step_committee_vote_logprobs` — single-pass internal committee

**Idea.** Append *several* short verifier questions in a *single* prompt as a numbered list, and ask the model to answer each with one letter. With `prompt_logprobs`, we read *each* letter's logprob simultaneously — one forward pass yields a multi-question committee.

**Probe template (≤30 tokens):**
```
{Q}
{R_<t}
{r_t}
Q1 step uses correct facts? (Y/N)
Q2 step has arithmetic error? (Y/N)
Q3 step is on-topic? (Y/N)
Q4 final answer derivable now? (Y/N)
A1
```
We then read logprobs of `Y/N` after each `A_i` cue. (In practice we structure as 4 micro-prompts in a single forward pass, separated by `\nA{i}:`.)

**Score formula:** logit-difference for each question:
`s_t^{(i)} = logit("Y") - logit("N") at position of A_i`
Combine: `s_t = w_1 s_t^{(1)} + w_2 (-s_t^{(2)}) + w_3 s_t^{(3)} + w_4 s_t^{(4)}`
with `w_i ≥ 0` learned by logistic regression on a tiny calibration subset (this is *not* a trained verifier — it's a 4-dimensional linear head on frozen features, fit with a closed-form on ≤200 examples).

**Cost.** 1 forward pass total (≈30 extra tokens of probe). All 4 question scores extracted from a single `prompt_logprobs` call.

**Hypothesis.** Different aspects of step quality (factuality, arithmetic, topicality, completeness) carry independent signal. Cheap committee ≈ cheap ensemble. Linear combination removes the need for a trained model.

**Pseudocode.**
```python
def step_committee_vote(model, Q, R_prev, r_t, w):
    base = Q + R_prev + r_t + "\n"
    probe = ("Q1 step uses correct facts? (Y/N)\n"
             "Q2 step has arithmetic error? (Y/N)\n"
             "Q3 step is on-topic? (Y/N)\n"
             "Q4 final answer derivable now? (Y/N)\n"
             "A1: . A2: . A3: . A4: .")
    plp = model.prompt_logprobs(base + probe)
    s = []
    for pos in find_dot_positions(probe):
        ly, ln = plp[pos]["Y"], plp[pos]["N"]
        s.append(ly - ln)
    s[1] = -s[1]   # Q2 is negatively phrased
    return float(np.dot(w, s))
```

**CP usage.** Per-trajectory aggregate `ψ = mean(s_t)`. Optionally, the committee's *disagreement variance* `Var_i s_t^{(i)}` is itself a useful nonconformity feature — high disagreement ⇒ ambiguous step ⇒ flag for inclusion in the prediction set.

**Failure modes.** (i) The 4 questions may be highly correlated in the model's head, collapsing to one effective question; mitigate with question-bank rotation. (ii) Tokenization: `Y/N` may merge with following punctuation; insert a space and `tok("Y")` precisely.

---

## Method 7. `step_symbolic_recast_validity` — does symbolic re-casting match?

**Idea.** Ask the model to re-emit `r_t` as a single line of *symbols only* (no prose), then check whether the symbolic line is **syntactically valid** and **numerically self-consistent** via a tiny external evaluator (Python `eval` on the equation string with sympy / `ast.parse`). The model itself does the natural→symbol translation cheaply; an external symbolic check provides a near-zero-cost ground truth on that one line.

**Probe template:**
```
{r_t}
Restate the above as one math expression with no prose:
```
Generate ≤20 tokens (greedy). Then post-process the generation string `e_t`.

**Score formula:**
- `valid = 1` iff `ast.parse(e_t)` succeeds AND, if it is an equality, `lhs == rhs` after sympy simplification.
- `s_t = -log p(e_t | prefix)` if valid, `+∞` otherwise.
A second component: cross-check that variables in `e_t` are a subset of variables seen in `R_<t}` (no symbol invention).

**Cost.** A short *generation* (≤20 tokens) — slightly more than the other methods but bounded; alternative cheap variant: instead of generating, force-decode a *templated* expression skeleton and score its logprob (turns it into a 1-pass scoring method again).

**Hypothesis.** A buggy reasoning step often cannot be re-cast as a single coherent equation; the model either invents undefined variables, produces a syntactic mess, or yields an equation that simplifies to false. Symbolic algebra gives us a near-free, *external*, verifier on the one-line summary of each step.

**Pseudocode.**
```python
import ast, sympy as sp

def step_symbolic_recast(model, Q, R_prev, r_t, allowed_vars):
    cue = "\nRestate the above as one math expression with no prose:\n"
    expr = model.generate(R_prev + r_t + cue, max_tokens=20, greedy=True)
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return float("inf")
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if not used.issubset(allowed_vars):
        return float("inf")
    # consistency: equality must simplify to True or to a tautology
    if "=" in expr or "==" in expr:
        try:
            lhs, rhs = expr.split("=")[-2:]
            ok = sp.simplify(sp.sympify(lhs) - sp.sympify(rhs)) == 0
            if not ok: return 5.0  # contradiction
        except Exception:
            return 3.0
    return 0.0   # passes all symbolic checks
```

**CP usage.** Per-step *binary-ish* score (0 / mid / inf). Aggregate via `ψ = max(s_t)` — a single contradicting step is fatal. Conformalize on the resulting trajectory score.

**Failure modes.** (i) Many steps are not naturally one-line equations (qualitative reasoning, set construction); for those, allow a "skip" tag returned by the model and treat the step as unscored. (ii) Greedy generation may hallucinate irrelevant equations; the `allowed_vars` whitelist mitigates symbol invention.

---

## Cross-cutting design notes

**Bias correction.** Methods 1, 5, 6 cancel the "yes" prior via contrast (corruption, dual polarity, weighted combination). Methods 2, 4 sidestep yes/no entirely by reading distributional features. Methods 3, 7 ground in external signals (question text, symbolic checker).

**Cost.** All except #7 are one or two prefix-only forward passes per step. With vLLM KV-cache reuse, marginal cost is `O(probe_length)` ≤30 tokens, well under the ≤2× budget.

**CP layering.** Each method yields per-step `s_t`; aggregate `ψ` is method-specific (mean / max). Split-CP treats `S(Q,trace) = ψ({s_t})` as the nonconformity score. For early-exit, conformalize each step's `s_t` separately using position-indexed calibrations.

**Ensembling.** Methods 1 + 5 + 6 share logit-difference machinery and can run as one batched call with three probes on a shared base. A linear head over the resulting feature vector, fit on a small calibration set, yields a stronger nonconformity score at the cost of method 6 alone.

**Common failure mode.** Confidently-wrong models give confidently-wrong self-evaluations. Contrastive (1, 5), distributional (2, 4), and grounded (3, 7) families attack this differently; a robust deployment combines at least one contrastive + one grounded method.

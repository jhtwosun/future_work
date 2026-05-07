# Agent SA — Step-Level Streaming Scores for CoT-CP

**Author:** Agent SA
**Date:** 2026-05-07
**Scope:** New step-level scoring methods that can be computed *during* generation, support mid-trace abstention/early termination, and integrate cleanly with a Conformal Prediction wrapper.

The goal is a per-step real-valued score `s_t = S(question, s_{0:t})` such that **higher = more likely correct**, computable from the prefix alone, cheap (≤2× per-step compute, ideally free), and usable as a calibrated CP gate that decides at each step boundary whether to (a) keep going, (b) switch to a fallback, or (c) abstain.

Below are 7 candidate methods. They are ordered roughly by cost (free → modest).

---

## 1. `step_token_curvature` — Online curvature of token-level confidence

### Definition
Let `c_i = lp_top1(i) − lp_top2(i)` be the per-token margin at position `i` (already produced by the sampler at no extra cost). For step `t` covering token range `[a_t, b_t]`, define the smoothed first and second differences of `c_i` *within and across* the previous K steps. Specifically:

- `mu_t = mean_{i in step t}(c_i)`
- `sigma_t = std_{i in step t}(c_i)`
- `delta_t = mu_t − mu_{t-1}` (drift in mean confidence)
- `curv_t = mu_t − 2*mu_{t-1} + mu_{t-2}` (second difference)
- Score: `s_t = mu_t − lambda1 * |delta_t| − lambda2 * max(0, −curv_t) − lambda3 * sigma_t`

(`lambda` tuned by CP calibration; the negative-curvature penalty fires only when confidence is *accelerating downward*.)

### Cost
**FREE.** Reuses logprobs already in the sampler buffer.

### Hypothesis
A step that is suddenly less confident than recent steps (large negative `delta`) or whose confidence is *accelerating* downward (negative `curv`) is the model "noticing" something off — typically a contradiction or a hard branch. Trajectory-level `lp_min` works because the *minimum* step lp is a strong predictor; here we catch the descent earlier by reading the derivative, before the minimum is reached.

### Pseudocode
```python
def step_token_curvature(prefix_buffer, t, K=2, lam=(0.5, 1.0, 0.3)):
    margins = [lp_top1[i] - lp_top2[i] for i in step_range(t)]
    mu = mean(margins); sigma = std(margins)
    mu_prev = ema_mu[t-1]; mu_prev2 = ema_mu[t-2] if t>=2 else mu_prev
    delta = mu - mu_prev
    curv  = mu - 2*mu_prev + mu_prev2
    s = mu - lam[0]*abs(delta) - lam[1]*max(0, -curv) - lam[2]*sigma
    ema_mu[t] = mu
    return s
```

### CP usage
Per-step nonconformity `α_t = −s_t`. On the calibration set, compute, for each *trace position index* `t`, the empirical quantile of `α_t` among traces that ended up correct at the same step index. At inference, abstain at step `t` if `α_t > Q̂_{1−ε}(t)`. Because the marginal step-position distribution shifts, calibrate per-step (or with monotone smoothing in `t`) and use the *cumulative* CP rule (any step crossing triggers abstention).

### Failure modes
- Steps that are intrinsically short (1–3 tokens, e.g. equation labels) have noisy `mu_t`; mitigate with a min-length floor.
- Stylistic discontinuities (e.g. switching from English to LaTeX) cause spurious negative curvature.
- For models with very flat top-k distributions (heavy entropy regularization), margins compress and dynamic range collapses.

---

## 2. `self_check_yesno` — Self-prompted on-track probe at step boundary

### Definition
At each step boundary (e.g. after `\n\n` or `Step N:` token, or whenever the parser declares a step), insert a *side-prompt* into a parallel sequence:

```
[prefix s_{0:t}]
Question: Given my work so far, am I on track to solve the original problem? Answer with a single token YES or NO.
```

Score: `s_t = lp("YES") − lp("NO")` from the same model, computed with KV-cache reuse on the prefix.

### Cost
**Cheap, ~free with KV reuse.** A single forward pass over a short suffix (~30 tokens). With paged KV-cache, this is ≪ 1× the cost of generating the next step.

### Hypothesis
The model already encodes a posterior over its own correctness; we just need a probe that surfaces it. Calibration (Kadavath et al. style) shows P(IK) is nontrivially well-calibrated on math/QA. The CP layer absorbs miscalibration globally, so we only need *monotonicity* with correctness, which holds empirically.

### Pseudocode
```python
def self_check_yesno(model, prefix_kv, step_text):
    probe = "\n[Self-check] Given my work so far, am I on track? Answer YES or NO.\nAnswer:"
    logits = model.forward_from_kv(prefix_kv, probe)[-1]
    lp_yes = logsoftmax(logits)[YES_id]
    lp_no  = logsoftmax(logits)[NO_id]
    return lp_yes - lp_no
```

### CP usage
This is a monotone scalar; use it directly as the per-step CP statistic. It also pairs well with **two-stage CP**: first use a free score (e.g. `step_token_curvature`) to gate which steps trigger the probe, then use the probe to make a tighter abstain decision. This keeps amortized cost near zero.

### Failure modes
- Over-confidence: instruction-tuned models bias toward "YES."  Recalibrate with the symmetric prompt and average (debias).
- Long prefixes erode the probe's salience; rotate the prompt to the *end* and prepend a short summary token.
- Adversarial/role-played traces ("I am pretending to fail...") break the probe — out of scope for math.

---

## 3. `lookahead_branch_disagreement` — One-step lookahead under temperature jitter

### Definition
At step boundary `t`, sample `m` short continuations (one step or `L≈40` tokens) from the same prefix at two temperatures `T_lo` and `T_hi` (or with two different top-p values). Define:

- For each continuation `j`, extract a content fingerprint `f_j` (cheap: numeric token sequence, or hash of the first N content tokens after stripping markup).
- `agree_t = max_freq(f_j) / m` (mode share)
- Score: `s_t = agree_t + β * mean_logprob(continuations)`

Crucially, `m` is *small* (m=4) and the continuation length is short — total cost ≈ 0.5–1× of generating one step, **not** of a full re-roll.

### Cost
**Cheap, ~1× per step.** Bounded by `m × L_step` tokens. Configure to ≤ the cost of the step itself.

### Hypothesis
Self-consistency works at the trajectory level because correct reasoning has a wider basin of attraction. The same effect should appear *locally*: at well-grounded steps, low-temperature lookahead variants converge; at ambiguous steps, they diverge. Rather than waiting for the end-of-trace SC, we measure *step-local* divergence as a leading indicator.

### Pseudocode
```python
def lookahead_branch_disagreement(model, prefix_kv, m=4, L=40, T=(0.2, 0.9)):
    conts = []
    for j in range(m):
        T_j = T[j % 2]
        c = model.sample_from_kv(prefix_kv, max_tokens=L, temperature=T_j)
        conts.append((fingerprint(c.tokens), c.mean_lp))
    fps = [c[0] for c in conts]
    mode_share = max(Counter(fps).values()) / m
    mean_lp = mean([c[1] for c in conts])
    return mode_share + 0.1 * mean_lp
```

### CP usage
Use as the per-step nonconformity `α_t = −s_t`. Strong fit for **sequential CP with stopping** (Bates et al. style): if the conformal `p`-value at any step falls below `ε`, abstain immediately. Because the score is bounded `[0,1]`-ish, plain split-CP with one quantile per step index works.

### Failure modes
- Steps with intrinsically multi-modal continuations (e.g. "next we could try X *or* Y") get penalized even when both are valid.
- Heavy tail in continuation length; cap at `L` strictly.
- For very low-temperature sampling, all four samples collapse and the score saturates — pair with a small jitter floor.

---

## 4. `prefix_consistency_cosine` — Online consistency between current step embedding and earlier "claim" embeddings

### Definition
Maintain a running pool of *claim embeddings* `E = {e_1, ..., e_{t-1}}`, where each `e_i` is the mean-pooled hidden state (last layer, *no probe*) of an automatically extracted "claim span" within step `i` (e.g. anything matching `=`, `therefore`, `so`, or numeric assertions). At step `t`:

- Extract claim spans in step `t`, get embedding `e_t`.
- Compute `consistency_t = max_{i<t} cos(e_t, e_i) − γ * max_{i<t} cos(e_t, neg(e_i))` where `neg(e_i)` is the embedding of the *negation-augmented* span (precomputed cheaply by prepending "It is not the case that ").
- Score: `s_t = consistency_t`.

### Cost
**FREE for the embeddings** (already produced during forward pass; we just mean-pool a short span). The negation embeddings cost one extra short forward per claim, but can be batched and computed lazily *only when* the step's free signals are borderline (gating, see §2).

### Hypothesis
A correct CoT trace has high *internal* semantic agreement: each new claim restates, refines, or builds on prior claims rather than contradicting them. Wrong traces drift: the model commits early, then produces claims inconsistent with that commitment. We surface that as embedding contradiction. This is **not** a probe — we use raw cosines without learned weights, so no pre-training is needed.

### Pseudocode
```python
def prefix_consistency_cosine(hidden_layer_states, claim_pool, t, gamma=0.5):
    spans = extract_claim_spans(step_text[t])
    if not spans: return last_score  # carry forward
    e_t = mean_pool(hidden_layer_states, spans)
    pos = max(cos(e_t, e) for e in claim_pool['pos'])
    neg = max(cos(e_t, e) for e in claim_pool['neg']) if claim_pool['neg'] else 0
    claim_pool['pos'].append(e_t)
    return pos - gamma * neg
```

### CP usage
Use directly as `s_t`. Particularly powerful as a **veto** in a two-score CP rule: keep going only if `s_token_curvature_t > τ_1` AND `s_consistency_t > τ_2` where both `τ` are CP-calibrated (Bonferroni-corrected to preserve coverage).

### Failure modes
- Claim extraction is noisy on free-form prose.
- Cosine geometry on raw hidden states is direction-sensitive and may not separate semantic agreement from surface similarity ("the answer is 7" vs "the answer is 8" are very close).
- The user excluded *probe-based* hidden methods; this one is probe-free (pure cosine), which we believe satisfies the constraint.

---

## 5. `step_answer_anchor_drift` — Provisional-answer drift score

### Definition
At each step boundary, ask the model (via a side-prompt with KV reuse) for its *current best guess* at the final answer:

```
[prefix]
Tentatively, my best guess at the final answer right now is: <ANS>
```

Read off the top-k logits at the `<ANS>` position, extract a normalized *answer distribution* `p_t(a)` (over short numeric / multiple-choice tokens). Then:

- `entropy_t = H(p_t)`
- `drift_t = TV(p_t, p_{t-1})` (total variation)
- `agree_t = max_a p_t(a)`
- Score: `s_t = agree_t − α * drift_t − β * entropy_t`

### Cost
**Cheap.** One forward pass over ~20 tokens with KV reuse, per step. Cheaper than method 3.

### Hypothesis
A "correct" trace converges: as steps accumulate, the implied answer distribution sharpens and *stops moving*. A wrong trace either thrashes between answers or stays flat. The score thus combines a **commitment** signal (`agree_t`) with a **stability** signal (`drift_t`). At end-of-trace this is essentially `sc_top1` *for free* — no need for 8 full re-rolls.

### Pseudocode
```python
def step_answer_anchor_drift(model, prefix_kv, prev_p, alpha=1.0, beta=0.5):
    probe = "\nTentatively my best guess at the final answer right now is:"
    logits = model.forward_from_kv(prefix_kv, probe)[-1]
    p = softmax_over_answer_tokens(logits)  # restricted vocab
    drift = 0.5 * sum(abs(p[a] - prev_p.get(a, 0)) for a in p)
    s = max(p.values()) - alpha*drift - beta*entropy(p)
    return s, p
```

### CP usage
Use `s_t` directly. Especially elegant: at any step, the *current* `argmax p_t` is a candidate prediction — CP can output a calibrated *current set* of plausible answers and stop early when the set shrinks below a target size. This converts CoT-CP from a binary keep/abstain into an **anytime selective predictor**.

### Failure modes
- Open-ended answer spaces (essays, code) — restrict to math/MCQA.
- Models that refuse to commit early ("I need to think more") — prefix the probe with "If you had to guess right now, even if uncertain, ..."
- Correlated drift across all problems (calibration drift) — handled by per-position CP quantiles.

---

## 6. `step_attribution_focus` — Attention-mass alignment with the question

### Definition
At each step, compute the **fraction of attention mass** in the last few layers (averaged over heads) that is allocated, from tokens in step `t`, back to tokens in the original question vs. recent self-generated tokens vs. early-CoT tokens. Define:

- `q_share_t = mean_{i in step t} sum_{j in question} attn(i, j)`
- `recent_share_t = mean_{i in step t} sum_{j in step t-1} attn(i, j)`
- `early_share_t = mean_{i in step t} sum_{j in steps 0..t-3} attn(i, j)`
- Score: `s_t = α * q_share_t + β * recent_share_t − γ * (1 − q_share_t − recent_share_t − early_share_t)`

### Cost
**FREE.** Attention weights are computed during the forward pass; we just aggregate.  Implementation requires hooking the attention module (most inference frameworks expose this); negligible runtime cost.

### Hypothesis
Wrong steps are often *confabulations*: the model riffs off its own recent text without grounding back in the original problem. Anchoring attention back on the question (or on early problem-restatement tokens) is a structural correlate of "still solving the right problem." This generalizes the `hidden_drift_max_penult` intuition into a directly-interpretable, free signal.

### Pseudocode
```python
def step_attribution_focus(attn_weights_step, question_idx, recent_idx, early_idx):
    A = attn_weights_step.mean(axis='heads').mean(axis='layers_top4')
    q_share = A[:, question_idx].sum(axis=-1).mean()
    rec = A[:, recent_idx].sum(axis=-1).mean()
    ear = A[:, early_idx].sum(axis=-1).mean()
    return 1.0*q_share + 0.5*rec - 0.5*(1 - q_share - rec - ear)
```

### CP usage
Free per-step nonconformity. Combine with `step_token_curvature` (also free) as a multi-dimensional score, and use Mahalanobis distance to a per-position calibration set as the conformal statistic — this gives near-zero amortized overhead.

### Failure modes
- Heavily quantized / KV-compressed deployments where attention is approximate or unavailable.
- Architectures with global attention sinks (first BOS token absorbs most mass) — normalize by removing sinks.
- Domains where the right behavior *is* to stop attending to the question (e.g. long deductive chains where the question is entirely encoded in a transformed state).

---

## 7. `step_arithmetic_witness` — Online verifier for checkable substeps

### Definition
A lightweight programmatic verifier runs at each step boundary, scanning the new step text for *checkable claims*: arithmetic identities, unit consistency, monotonicity claims, citation of named theorems with checkable hypotheses, etc. Each detected claim becomes a witness:

- `w_i ∈ {+1, −1, 0}` (verified, falsified, undecidable)
- `s_t = (#verified − #falsified) / max(1, #checkable) + λ * sigmoid(cumulative_witnesses)`

Examples of checkable claims (regex + tiny eval):
- `a op b = c` for op ∈ {+, −, *, /, ^} on numeric tokens → eval and compare.
- "X > Y" with numeric X, Y → check.
- "Therefore Z is even/odd" with Z numeric → check.
- "Let n = ..." subsequently used with a different value → contradiction.

### Cost
**FREE.** Pure CPU regex + arithmetic, parallel to model generation.

### Hypothesis
Most CoT errors propagate from a *checkable* arithmetic / logical slip. A single witness of falsification is near-deterministic evidence the trace will be wrong; one verified witness is mild positive evidence that the model is being careful. This is the only score in the list that can produce a *hard* abstain signal (`w_i = −1`), which is gold for CP.

### Pseudocode
```python
def step_arithmetic_witness(step_text, var_env):
    witnesses = []
    for claim in extract_arithmetic_claims(step_text):
        v = eval_claim(claim, var_env)
        witnesses.append(v)
    update_variable_bindings(var_env, step_text)
    pos = sum(1 for w in witnesses if w == +1)
    neg = sum(1 for w in witnesses if w == -1)
    n   = max(1, len([w for w in witnesses if w != 0]))
    return (pos - neg) / n + 0.1 * tanh(running_witness_sum)
```

### CP usage
The `−1` witness gives a hard veto: emit immediate abstain regardless of CP threshold (this preserves coverage trivially since false-positive vetoes are bounded by the verifier's precision, which is near-1 for arithmetic). For everything else, use `s_t` as a standard nonconformity. Combine additively with `self_check_yesno` (§2) and `step_token_curvature` (§1): a *learned* linear combination calibrated as a single CP score.

### Failure modes
- Domains with no checkable claims (literary QA, ethics).
- Symbolic algebra where the verifier needs a CAS — implement as optional plugin (SymPy at <10 ms per claim).
- The verifier itself can be wrong (regex parses a citation as an equation); set verifier confidence threshold high to keep false-`−1` rate low.

---

## Cross-cutting design notes

### Combining scores under CP
The seven scores split into three natural tiers:

- **Tier-0 (always-on, free):** §1, §6, §7.  Compute every step.
- **Tier-1 (cheap probes, gated by Tier-0):** §2, §5.  Trigger only when Tier-0 score is borderline.
- **Tier-2 (lookahead, gated by Tier-1):** §3, §4.  Trigger only when probes disagree.

This **cascade** keeps amortized cost near 1× while giving the CP layer rich signal precisely on hard examples.

### Per-step calibration
For sequential CP, calibrate a separate threshold per step *index* `t` (or per *normalized* step position `t / T_pred`). Use isotonic smoothing across `t` to avoid overfitting at rare large indices. The cumulative-coverage rule (Cauchois et al., Angelopoulos et al.) maintains the global ε guarantee: emit abstain at the first step where the running CP test rejects.

### Anytime predictor
Method §5 turns CoT-CP into an *anytime* system: at any step, we have a calibrated set. Methods §1, §6, §7 provide gating; §5 provides the actual current prediction. This is a stronger product than end-of-trace CP and worth a section in the paper.

### Empirical priors (from the existing trajectory results)
The fact that `lp_min` (a *minimum* over steps) wins suggests that at the trajectory level, **a single bad step is decisive**. This justifies sequential CP with a *threshold-crossing* rule (any step crossing → abstain), and predicts that §1 and §7 should be the strongest individual streaming scores — both are sensitive to the worst step. §3 and §5 should be the strongest as *added* signal on top of the free scores, since they capture orthogonal information (counterfactual continuations and answer commitment, respectively). I'd recommend prioritizing §1 + §5 + §7 for first experiments — together they cover token-level dynamics, semantic commitment, and hard logic checks at near-zero cost.

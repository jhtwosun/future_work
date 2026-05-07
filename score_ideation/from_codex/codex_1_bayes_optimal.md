# Codex 1 (Bayes-optimal angle)

## Bayes-optimal target

Let `R` denote the full reasoning trajectory (tokens, intermediate steps, tool calls, and final answer), and let `Y ∈ {0,1}` indicate whether the trajectory is correct. For selective prediction with fixed coverage, the Bayes-optimal score is any monotone transform of the posterior odds

```math
s^*(R) \;=\; \log \frac{P(Y=1 \mid R)}{P(Y=0 \mid R)}
        \;=\; \log \frac{P(R \mid Y=1)}{P(R \mid Y=0)} + \log \frac{P(Y=1)}{P(Y=0)}.
```

So the mathematically correct object is a likelihood ratio, or equivalently posterior odds. This is stronger than `lp_min`, `prm_min`, or `sc_top1` because it uses all available evidence in `R`, not just the weakest step, a single reward head, or a majority vote over samples.

The practical difficulty is that `P(R | Y=1)` is not directly available from a standard LM. If we can estimate the posterior `P(Y=1 | R)` with calibration, Bayes’ rule gives the optimal ranking immediately. If we cannot, we need proxies for the likelihood ratio. The best approximations should therefore preserve one of three structures:

1. Direct posterior estimation.
2. Evidence accumulation over trajectory blocks.
3. Description-length or information-gain proxies for the same likelihood ratio.

Below are four score families that are theoretically tied to the Bayes-optimal target rather than just heuristic confidence.

---

## 1. `calibrated_verifier_odds`

**Bayesian motivation.**  
This is the closest practical estimator of the true posterior odds. Train a verifier `q_φ(R)` to approximate `P(Y=1 | R)`. If calibrated, then

```math
s(R) = \log \frac{q_\phi(R)}{1-q_\phi(R)} - \log \frac{\pi}{1-\pi},
```

where `π = P(Y=1)` is the class prior, estimates `log P(R|Y=1) / P(R|Y=0)`.

**Concrete formula.**

```math
q_\phi(R) \approx P(Y=1 \mid R), \qquad
s(R) = \operatorname{logit}(q_\phi(R)) - \operatorname{logit}(\pi).
```

Use a sequence-level verifier over the full trajectory, not a minimum over steps.

**Compute cost.**  
About `1x` to `2x` greedy decode, depending on whether the verifier is a lightweight head or a separate forward pass.

**Why it may dominate.**  
If `q_φ` is calibrated, this is asymptotically Bayes-optimal. It also strictly dominates `prm_min` in principle because it aggregates all evidence instead of taking the worst step. Compared with `sc_top1`, it avoids Monte Carlo variance and uses a graded posterior rather than a binary vote.

**Implementation sketch.**

1. Collect `(R, Y)` pairs from reasoning traces.
2. Train a sequence verifier on the full trace, possibly with a scalar head over the last hidden state.
3. Calibrate with temperature scaling or isotonic regression.
4. Use the calibrated log-odds as the conformity score.

---

## 2. `jackknife_evidence_odds`

**Bayesian motivation.**  
If the trajectory contains multiple approximately independent evidence blocks, then the full log Bayes factor should decompose additively across blocks. Leave-one-out scores estimate how much each block contributes to the posterior odds, which is a jackknife approximation to the evidence under perturbation.

**Concrete formula.**  
Split `R` into `B` blocks `R_1, ..., R_B`. Let `R_{-b}` be the trajectory with block `b` removed. Define

```math
s(R) = \frac{1}{B} \sum_{b=1}^B \operatorname{logit}\!\big(q_\phi(R_{-b})\big).
```

A stronger variant uses the full-minus-LOO gap:

```math
s(R) = \operatorname{logit}(q_\phi(R)) - \frac{1}{B}\sum_{b=1}^B \operatorname{logit}(q_\phi(R_{-b})).
```

Large positive gaps mean the posterior is robust to deleting any single block, which is what a genuinely supported correct trajectory should look like.

**Compute cost.**  
About `Bx` verifier passes. With `B=4` blocks, the cost is roughly `4x` greedy decode, still below `sc_top1` with `N=8`.

**Why it may dominate.**  
`lp_min` and `prm_min` can be brittle to local noise. Jackknife evidence is explicitly stability-aware: it rewards trajectories whose correctness is supported redundantly across steps. That makes it theoretically closer to posterior concentration under evidence accumulation.

**Implementation sketch.**

1. Chunk the reasoning trace into semantic blocks.
2. Score the full trace and each leave-one-out variant with the same verifier.
3. Aggregate the log-odds gaps.
4. Optional: fit a linear correction from jackknife gaps to empirical correctness.

---

## 3. `mutual_information_gain`

**Bayesian motivation.**  
Correct reasoning should reduce uncertainty about the answer. That suggests measuring the information gain from observing `R`:

```math
I(A;R) = H(A) - H(A \mid R),
```

where `A` is the final answer variable. In a near-deterministic task, sharper answer posteriors correspond to higher correctness probability. This is a posterior-concentration view of the Bayes factor.

**Concrete formula.**

Estimate the answer posterior from `K` candidate answers or `K` self-consistency samples:

```math
\hat p(a \mid R) \propto \exp(\ell(a; R)),
```

then define

```math
s(R) = H[\hat p(A)] - H[\hat p(A \mid R)].
```

If the answer prior is hard to estimate, a usable proxy is the normalized entropy drop

```math
s(R) = -H[\hat p(A \mid R)].
```

**Compute cost.**  
Typically `2x` to `4x` greedy decode if you estimate a small answer posterior from top candidates.

**Why it may dominate.**  
`sc_top1` throws away posterior shape and keeps only the top vote fraction. Mutual information uses the entire posterior mass, so it is strictly more informative whenever the answer distribution is non-degenerate.

**Implementation sketch.**

1. Use the base LM to produce a small candidate answer set.
2. Re-score each answer under the trajectory-conditioned prompt.
3. Convert scores to a normalized posterior.
4. Use entropy drop or mutual information as the conformity score.

---

## 4. `mdl_compression_gap`

**Bayesian motivation.**  
By Shannon coding and MDL, log likelihood ratios equal code-length gaps. A trajectory that is better compressed under a correct-answer model than under a null model has stronger evidence for correctness.

**Concrete formula.**

Let `L_0(R)` be the code length under an unconditional trajectory model, and `L_1(R)` the code length under a correctness- or answer-conditioned model. Then

```math
s(R) = L_0(R) - L_1(R).
```

This approximates

```math
\log \frac{P(R \mid Y=1)}{P(R \mid Y=0)}.
```

If a learned conditional generator for rationales exists, `L_1` can condition on the predicted answer, while `L_0` is a generic rationale LM.

**Compute cost.**  
About `2x` greedy decode if both code lengths are evaluated with one forward pass each.

**Why it may dominate.**  
This is a principled universal proxy for Bayes factors. Compared with `lp_min`, it uses total description length rather than the weakest token. Compared with `prm_min`, it is model-based rather than a single scalar head.

**Implementation sketch.**

1. Train or reuse a generic trajectory LM and an answer-conditioned trajectory LM.
2. Compute token-level NLL for both models on the same trace.
3. Use the code-length difference as the score.
4. Optionally normalize by trace length to avoid a trivial preference for short traces.

---

## 5. `contrastive_answer_margin`

**Bayesian motivation.**  
If the posterior over answers is sharply peaked, correctness is often separated from the runner-up by a large posterior margin. This is a low-cost approximation to the Bayes factor between the best answer hypothesis and its nearest competitor.

**Concrete formula.**

Let `a_1` and `a_2` be the top two answer hypotheses under the posterior induced by the trajectory. Then

```math
s(R) = \log P(a_1 \mid R) - \log P(a_2 \mid R).
```

If a correctness verifier is available, replace answer probability with `P(Y=1 | R, a)`.

**Compute cost.**  
About `1x` to `3x`, depending on how many candidate answers must be scored.

**Why it may dominate.**  
Margins are more statistically efficient than top-1 counts. This is a cheaper alternative to `sc_top1` that still encodes uncertainty and is often much less brittle on near-tie examples.

**Implementation sketch.**

1. Generate a small candidate answer set from the model.
2. Score candidates conditionally on the full trajectory.
3. Take the log-probability margin between the best and second-best candidate.

---

## Bottom line

If we knew the true posterior `P(Y=1 | R)`, the optimal conformity score would simply be posterior odds. The practical design goal is therefore not to invent unrelated heuristics, but to build estimators of that odds ratio with progressively less modeling bias:

1. `calibrated_verifier_odds` for direct posterior estimation.
2. `jackknife_evidence_odds` for stability-aware evidence aggregation.
3. `mutual_information_gain` for posterior concentration.
4. `mdl_compression_gap` for a universal code-length proxy.
5. `contrastive_answer_margin` for a cheap posterior-margin approximation.

Among these, the first is the cleanest Bayes-optimal approximation, while the others are better thought of as structured relaxations of the same likelihood-ratio target.

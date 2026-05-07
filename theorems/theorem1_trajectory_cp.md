# Theorem 1: Trajectory-Level CP Coverage from Step-Aggregated Scores

> **Author**: Claude (Codex was attempted in parallel but stuck on auth/network for ~1h; falling back to direct derivation)
> **Status**: First draft. Standard split-CP machinery; novelty in this paper is that we apply it to *step-aggregated* scores in the LLM reasoning context.

## 1. Setup

### Data

We observe a labeled calibration set
$\mathcal{D}_{\mathrm{cal}} = \{(X_i, R_i, Y_i)\}_{i=1}^{n}$
and one test point $(X_{n+1}, R_{n+1}, Y_{n+1})$ from the same
distribution, where:
- $X_i \in \mathcal{X}$ is the prompt;
- $R_i = (S_{i,1}, \ldots, S_{i,T_i}) \in \bigsqcup_{T \geq 1} \mathbb{R}^T$
  is the variable-length sequence of step-level confidence scores
  (e.g., per-step mean log-prob, per-step PRM reward, or per-step
  self-consistency agreement);
- $Y_i \in \{0, 1\}$ indicates final-answer correctness.

We assume $(X_1, R_1, Y_1), \ldots, (X_{n+1}, R_{n+1}, Y_{n+1})$ are
**exchangeable** — strictly weaker than i.i.d. and the standard
hypothesis for split CP.

### Aggregator and trajectory score

Let $\phi : \bigsqcup_T \mathbb{R}^T \to \mathbb{R}$ be any **measurable
permutation- or position-respecting aggregator** (we do *not* require
permutation invariance — examples include $\min$, $\mathrm{mean}$, last
value, or a learned aggregator). Define
$$\bar{S}_i := \phi(R_i) \in \mathbb{R}.$$

Note: $\phi$ may depend on $X_i$ (e.g., a learned aggregator
conditioning on the prompt), but it must be measurable and *fixed
across the calibration / test distribution*. In particular, $\phi$ is
*not* fit on $\mathcal{D}_{\mathrm{cal}}$.

### Calibrated quantile

Let $\mathcal{I}_+ := \{i \leq n : Y_i = 1\}$ index the *correct*
calibration trajectories. Let $n_+ = |\mathcal{I}_+|$. Define the
finite-sample-corrected lower $\alpha$-quantile of correct-trajectory
scores:
$$\hat{q}_\alpha := \mathrm{Quantile}\!\left( \{\bar{S}_i\}_{i \in \mathcal{I}_+},\, \frac{\lfloor \alpha (n_+ + 1) \rfloor}{n_+} \right).$$

The CoT-CP procedure outputs the trajectory whenever
$\bar{S}_{n+1} \geq \hat{q}_\alpha$ ("kept"), and abstains otherwise.

## 2. Theorem 1

### Statement

**Theorem 1 (Trajectory-level CP coverage).** Suppose
$(X_1, R_1, Y_1), \ldots, (X_{n+1}, R_{n+1}, Y_{n+1})$ are exchangeable.
For any measurable aggregator $\phi$ and any $\alpha \in (0, 1)$,
conditional on the test point being correct,
$$\boxed{\,\Pr\!\left[ \bar{S}_{n+1} \geq \hat{q}_\alpha \,\middle|\, Y_{n+1} = 1 \right] \;\geq\; 1 - \alpha - \frac{1}{n_+ + 1}.\,}$$

The $(n_+ + 1)^{-1}$ slack is the standard finite-sample-correction
deficit; for $n_+ \geq 100$ it is below 1%.

### Proof sketch

Conditioning on $\{Y_i = 1\}_{i \in \mathcal{I}_+ \cup \{n+1\}}$ (the
event that the test point is correct *and* identifying which
calibration points are correct), exchangeability of the original
$(n+1)$-tuple implies the *correct subset*
$\{(X_i, R_i)\}_{i \in \mathcal{I}_+ \cup \{n+1\}}$ is also
exchangeable. (This is the same reduction used in Lei-Wasserman
2014 / Vovk-Gammerman-Shafer 2005 for conditional inference on
correctness.)

Apply $\phi$ to each $R_i$ in this conditional subset to obtain
$(n_+ + 1)$ exchangeable real-valued scores
$\{\bar{S}_i\}_{i \in \mathcal{I}_+ \cup \{n+1\}}$.

Standard split-CP coverage (Theorem 1 of Lei-Wasserman 2014, or
Proposition 1 of Vovk 2002) then states: the rank of $\bar{S}_{n+1}$
among the calibration scores is uniform on $\{1, \ldots, n_+ + 1\}$ in
the absence of ties, hence
$$\Pr\!\left[ \bar{S}_{n+1} \geq \hat{q}_\alpha \,\big|\, \mathcal{I}_+, Y_{n+1}=1 \right] \geq 1 - \alpha - \frac{1}{n_+ + 1}.$$

Marginalizing over $\mathcal{I}_+$ gives the stated bound. $\blacksquare$

### Corollary (selective error / kept_acc)

Our empirical headline metric ("kept accuracy") is the conditional
error rate among kept trajectories:
$$\rho_{\mathrm{kept}} := \Pr\!\left[ Y_{n+1} = 1 \,\big|\, \bar{S}_{n+1} \geq \hat{q}_\alpha \right].$$

Theorem 1 gives a bound on
$\Pr[\bar{S}_{n+1} \geq \hat{q}_\alpha \mid Y_{n+1} = 1]$, not on
$\rho_{\mathrm{kept}}$ directly. The two are related by Bayes:
$$\rho_{\mathrm{kept}} = \frac{\pi \cdot \Pr[\bar{S}_{n+1} \geq \hat{q}_\alpha | Y_{n+1}=1]}{\pi \cdot \Pr[\bar{S}_{n+1} \geq \hat{q}_\alpha | Y_{n+1}=1] + (1-\pi) \cdot \Pr[\bar{S}_{n+1} \geq \hat{q}_\alpha | Y_{n+1}=0]},$$
where $\pi = \Pr(Y_{n+1} = 1)$.

**Corollary 1 (Kept-accuracy lower bound).** Let
$\beta_+ = \Pr[\bar{S} \geq \hat{q}_\alpha | Y=1]$ and
$\beta_- = \Pr[\bar{S} \geq \hat{q}_\alpha | Y=0]$. Then
$\rho_{\mathrm{kept}} = \frac{\pi \beta_+}{\pi \beta_+ + (1-\pi) \beta_-}$.
Theorem 1 gives $\beta_+ \geq 1 - \alpha - 1/(n_+ + 1)$. The kept
accuracy bound is therefore controlled by the **worst-case shift** in
$\beta_-$: the score must rank wrong trajectories *worse* than
correct ones. (Theorem 2's SNR characterization makes this explicit.)

In our experiments, we report $\rho_{\mathrm{kept}}$ directly; the
guarantee is *aggregate* coverage from Theorem 1.

## 3. Comparison to related work

- **Quach et al. ICLR 2024 (Conformal Language Modeling).** They
  apply CP at the *sequence* level, treating an entire generation as
  the unit. Theorem 1's contribution is the freedom to use any
  *step-aggregated* score $\phi(R)$ as the conformity score, including
  step-min for early-error detection or step-mean for trajectory
  averaging. The aggregator choice is decoupled from the CP coverage
  guarantee.

- **Mohri & Hashimoto ICML 2024 (Conformal Factuality).** They
  decompose answers into independent claims and apply CP per-claim.
  Theorem 1 is non-decompositional: a single trajectory score per
  question, but with the inner aggregator capturing claim-like
  step-level information.

- **Abbasi-Yadkori et al. NeurIPS 2024 (Conformal Abstention).**
  Closest to our setting, but uses *semantic entropy* over multiple
  samples as the score and conditions on the *abstain-or-answer*
  decision. Theorem 1 covers the same coverage-on-correct guarantee
  but with our specific aggregators (lp_min, prm_min, sc_top1) over
  step-level signals rather than sample-level entropy.

## 4. Subtleties

### 4.1 Conditioning on correctness vs. marginal CP

Theorem 1 conditions the coverage statement on $Y_{n+1} = 1$. This is
the *correctness-conditional* form. The marginal version
$\Pr[\bar{S}_{n+1} \geq \hat{q}_\alpha] \geq 1 - \alpha - 1/(n_+ + 1)$
also holds *for the same $\hat{q}_\alpha$*, but is less informative
because we don't condition on what we want to keep.

For our headline metric (kept accuracy), the conditional form is the
right one — see Corollary 1.

### 4.2 What about $\phi$ depending on $X$?

If $\phi$ is a fixed (population-defined) function, exchangeability is
preserved trivially. If $\phi$ is *learned* on a separate dataset, the
exchangeability assumption holds *as long as the learning data is
disjoint from the calibration / test sets*. Our implementations
(lp_min, prm_min, sc_top1) all use fixed aggregators, so this is not
an issue.

### 4.3 Small $n_+$

The $1/(n_+ + 1)$ slack means coverage near the target requires
$n_+$ at least a few hundred. In our experiments $n_+ \in [\sim 50, \sim 400]$
which gives slack 0.2–2% — within bootstrap CIs.

### 4.4 Ties

If $\phi(R)$ is discrete (e.g., sc_top1 takes only $N+1$ values),
quantile ties create over-coverage at certain levels. The lower
quantile convention used here ensures coverage *lower bound* still
holds (with $\geq$ replaced by strict $\geq$); the upper bound on
coverage may be lost. This explains some over-coverage at small $\alpha$
in our SC@8 results.

### 4.5 Distribution shift

Theorem 1 assumes exchangeability with the test point. Our OOD
experiments (MATH→AIME) violate this; **Theorem 3** addresses the
weighted-CP fix for that setting.

## 5. Discussion / cross-check points (for Codex / reviewer)

1. **Conditioning on correctness — is the exchangeability reduction
   clean?** I claimed that conditioning on $\{Y = 1\}$ preserves
   exchangeability *within the correct subset*. This is standard but
   worth a careful verification — Codex / reviewer should verify the
   measure-theoretic step.

2. **The $1/(n_+ + 1)$ correction**: this should match the standard
   split-CP correction (e.g., Lei et al. JASA 2018 use
   $\lceil (n+1)(1-\alpha) \rceil / n$). Verify the inequality
   direction is right (we want a *lower* bound on coverage, hence
   the *floor* in our quantile definition).

3. **Aggregator measurability**: is "any measurable aggregator"
   actually too loose? Specifically, can $\phi$ depend on the
   *random* calibration sample? No — must be fixed. Worth stating
   formally.

4. **Selective error / Bayes corollary**: I computed $\rho_{\mathrm{kept}}$
   in terms of $\beta_+, \beta_-, \pi$. This is the form that
   matches Theorem 2's SNR analysis. Verify the derivation is right.

5. **Discrete-score subtlety**: section 4.4 hints at this; should
   we explicitly state Theorem 1 in the discrete-score case (where
   ties matter), or leave that to the discussion?

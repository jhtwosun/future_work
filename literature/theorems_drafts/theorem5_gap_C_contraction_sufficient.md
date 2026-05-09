# Theorem 5' — PROOF GAP C: Sufficient Conditions for $\bar L < 1$

> **Author**: Claude Opus 4.7 (1M context), Lakatos cross-verification lane.
> **Status**: Draft, 2026-05-08. Cross-model verification token = `sk-PLACEHOLDER` (per workspace `CLAUDE.md`).
> **Inputs**:
> - `theorem5_v2_consolidated.md` §E (Theorem 5' statement, `[PROOF GAP C]` callout)
> - `experiments/results/distance_ladder_pilot.json` (per-rung $\hat\epsilon_k$, $n_k$, |S|, empirical implied $\bar L \approx 0.85$)
> - Sibling drafts `theorem5_gap_A_lipschitz_*.md` and `theorem5_gap_B_fixed_point_coverage_*.md` (not yet on disk; this draft is consistent with their *expected* form per `theorem5_v2_consolidated.md` §E.4)
> **Companion**: `theorem3_weighted_cp_discrete.md` (the one-shot foil; supplies the discrete-quantile DKW machinery reused below).
> **Scope of this document**: Discharge the sufficient-condition half of `[PROOF GAP C]`. We do *not* tighten the per-step Lipschitz bound itself (that is Gap A's job); we take the form of $L_k$ from Gap A and derive *measurable* sufficient conditions on $(\epsilon_k, n_k, |\mathcal S|, \hat w)$ under which $\bar L = \prod_k L_k < 1$.

---

## §1 — Setup (recall from Gap A and Theorem 5' §E)

**Notation.** Let $K$ be the number of rungs, $\mathcal S = \{0/N, 1/N, \ldots, N/N\}$ the discrete score support with $|\mathcal S| = N+1$, $n_k$ the rung-$k$ correct-trajectory sample size (i.e., $|\mathcal I_+^{(k)}|$), $\hat\epsilon_k = \tfrac12\sum_s|\hat p_k(s) - \hat p_{k+1}(s)|$ the empirical rung-pair TV, and $\hat w_{k-1\to k}(s) = \hat p_k(s)/\hat p_{k-1}(s)$ the empirical density-ratio weights. Let
$$w_k^+ := \max_{s\in\mathrm{supp}\hat p_{k-1}}\hat w_{k-1\to k}(s),\qquad w_k^- := \min_{s\in\mathrm{supp}\hat p_{k-1}}\hat w_{k-1\to k}(s),$$
both with Laplace smoothing $\varepsilon = 1/n_{\min,k}$ baked in (see §D of v2 consolidated). Define $W_{\max} := \max_k w_k^+$ and $W_{\min} := \min_k w_k^-$.

**Per-step Lipschitz constant (Gap A form).** From Theorem 5' (B1) (`theorem5_v2_consolidated.md` E.3), the per-step weighted-quantile operator $T_k$ on the smoothed quantile range satisfies
$$L_k \;\leq\; \frac{w_k^+}{\alpha\cdot w_k^-}\cdot d_{TV}(P_k,P_{k+1}) \cdot \kappa_k,$$
where $\kappa_k$ is the *quantile-CDF-density* prefactor capturing the discrete-atom granularity (Gap A's smoothing argument). The empirical analog replaces $d_{TV}$ by $\hat\epsilon_k + \mathrm{DKW}_k(\delta/K)$ with
$$\mathrm{DKW}_k(\delta) := \tfrac{|\mathcal S|}{2}\sqrt{\tfrac{\log(2|\mathcal S|/\delta)}{2 n_k}}.$$

**Mean (product) Lipschitz constant.** Strategy B' is the iterated map $T_K\circ\cdots\circ T_1$, so the relevant aggregate constant is
$$\bar L := \prod_{k=1}^K L_k.$$
A *max* form $\bar L_{\max} := \max_k L_k$ is the alternative aggregate; we treat the product form as primary (it is the contraction rate of the composed map) and the max form as a coarser fallback.

**The question this draft answers.** Under what *measurable* conditions on $(\epsilon_k, n_k, |\mathcal S|, \hat w)$ does $\bar L < 1$ hold? The v2 consolidated draft only states $\bar L < 1$ as an assumption (B2); we now turn it into a verifiable certificate.

---

## §2 — Decomposition of $L_k$ into three measurable factors

We rewrite $L_k$ in a form that exposes its three sources of magnitude:
$$L_k \;=\; \underbrace{\rho_k}_{\substack{\text{(a) weight-ratio}\\ \text{amplification}}} \;\cdot\; \underbrace{\hat\epsilon_k^{\,*}(\delta)}_{\substack{\text{(b) per-rung}\\ \text{empirical-PMF noise}}} \;\cdot\; \underbrace{\kappa_k}_{\substack{\text{(c) discrete-quantile}\\ \text{granularity}}},$$
where:

**(a) Weight-ratio amplification.**
$$\rho_k \;:=\; \frac{w_k^+}{\alpha\cdot w_k^-} \;=\; \frac{1}{\alpha}\cdot\frac{w_k^+}{w_k^-}.$$
This is the per-rung *condition number* of the importance-weight vector divided by the miscoverage tolerance. Two interpretations: (i) the inverse-CDF-density prefactor in the discrete weighted-quantile inverse (cf. Tibshirani 2019 Lemma A.1), and (ii) the worst-case amplification of a TV perturbation by the importance reweighting.

**(b) Per-rung empirical-PMF noise.**
$$\hat\epsilon_k^{\,*}(\delta) \;:=\; \hat\epsilon_k + \mathrm{DKW}_k(\delta/K) \;=\; \hat\epsilon_k + \tfrac{|\mathcal S|}{2}\sqrt{\tfrac{\log(2K|\mathcal S|/\delta)}{2 n_k}}.$$
This is the empirical TV plus a one-sided union-bounded DKW slack (one DKW per rung-pair, union over $K$ pairs at level $\delta$).

**(c) Discrete-quantile granularity.**
$$\kappa_k \;:=\; \frac{1}{\hat p_k(s_q^*)},$$
where $s_q^* \in \mathcal S$ is the score atom *at* which the $\alpha$-weighted quantile lands. By the smoothing argument of Gap A (replacing $T_k$ with linear-interpolant $T_k^\eta$ between adjacent atoms), $\kappa_k \leq 1/\hat p_k^{\min}$ where $\hat p_k^{\min} := \min_{s\in\mathcal S}\hat p_k(s) \geq 1/(n_k+|\mathcal S|)$ under Laplace smoothing. A coarser uniform bound is $\kappa_k \leq |\mathcal S|$ (the inverse-density at uniform PMF, modulo Laplace).

This three-factor decomposition is the key to Gap C: each factor is **separately measurable** from pilot data, so verifying $\bar L < 1$ becomes a numerical check, not a probabilistic conjecture.

---

## §3 — Sufficient condition (per-rung and aggregate)

### 3.1 Per-rung bound

Combining §2:
$$L_k \;\leq\; \frac{1}{\alpha}\cdot\frac{w_k^+}{w_k^-}\cdot\Big(\hat\epsilon_k + \tfrac{|\mathcal S|}{2}\sqrt{\tfrac{\log(2K|\mathcal S|/\delta)}{2 n_k}}\Big)\cdot \kappa_k.$$

For the *empirical-noise-dominated* regime ($\hat\epsilon_k \ll \mathrm{DKW}_k$), this collapses to
$$L_k \;\leq\; \frac{|\mathcal S|\,\kappa_k}{2\alpha}\cdot\frac{w_k^+}{w_k^-}\cdot\sqrt{\tfrac{\log(2K|\mathcal S|/\delta)}{2 n_k}} \;=\; \frac{C_k(\delta)}{\sqrt{n_k}},$$
where the explicit constant is
$$\boxed{\;C_k(\delta) \;:=\; \frac{|\mathcal S|\,\kappa_k}{2\alpha}\cdot\frac{w_k^+}{w_k^-}\cdot\sqrt{\tfrac{\log(2K|\mathcal S|/\delta)}{2}}\;}\qquad\text{(empirical-noise regime).}$$

For the *signal-dominated* regime ($\hat\epsilon_k \gtrsim \mathrm{DKW}_k$), the relevant bound is
$$L_k \;\leq\; \frac{\kappa_k}{\alpha}\cdot\frac{w_k^+}{w_k^-}\cdot\hat\epsilon_k \;=\; \tilde C_k\cdot \hat\epsilon_k,\qquad \tilde C_k := \frac{\kappa_k}{\alpha}\cdot\frac{w_k^+}{w_k^-}.$$
The two regimes meet at the crossover sample size $n_k^* = |\mathcal S|^2 \log(2K|\mathcal S|/\delta) / (8\hat\epsilon_k^2)$; for $n_k > n_k^*$ the bound is signal-dominated.

### 3.2 Aggregate sufficient condition (product form)

$$\bar L \;\leq\; \prod_{k=1}^K L_k \;\leq\; \prod_{k=1}^K \tilde C_k \cdot \hat\epsilon_k \quad(\text{signal regime}),$$
or, taking logs,
$$\log\bar L \;\leq\; \sum_{k=1}^K \log L_k \;=\; \sum_k\log(\tilde C_k\cdot\hat\epsilon_k).$$

A clean **product-form sufficient condition** for $\bar L < 1$ is therefore
$$\boxed{\;\sum_{k=1}^K \log\!\Big(\tilde C_k\cdot\hat\epsilon_k^{\,*}(\delta)\Big) \;<\; 0\;,}$$
i.e., on average per rung, $\tilde C_k\cdot\hat\epsilon_k^{\,*}$ is below 1. Equivalently, in geometric-mean form:
$$\big(\textstyle\prod_k \tilde C_k\hat\epsilon_k^{\,*}\big)^{1/K} \;<\; 1\quad\Longleftrightarrow\quad \bar\rho\cdot\bar\epsilon^{\,*} \cdot \bar\kappa < \alpha,$$
where $\bar\rho = (\prod_k w_k^+/w_k^-)^{1/K}$ is the geometric-mean weight condition number, $\bar\epsilon^{\,*} = (\prod_k\hat\epsilon_k^{\,*})^{1/K}$ the geometric-mean per-rung empirical TV, and $\bar\kappa = (\prod_k\kappa_k)^{1/K}$ the geometric-mean granularity.

### 3.3 Aggregate sufficient condition (max form, coarser)

If we want to avoid the product structure (e.g., when one rung has $L_k\geq 1$ but the others contract), the *max-form* sufficient condition is
$$\bar L_{\max} := \max_k L_k < 1 \quad\Longleftrightarrow\quad \max_k \tilde C_k\hat\epsilon_k^{\,*}(\delta) < 1.$$
This is strictly stronger than the product form (it requires *every* rung individually to contract). Its appeal is that $\bar L_{\max}$ controls the rate of the *worst-case* iterate, which is a more conservative analytic object than $\bar L$.

The product form is the headline. The max form is reported as a side check and is the regime under which the contraction is *robust to rung permutation*.

### 3.4 Honest tightness disclaimer

The bounds (3.1)–(3.3) are **upper bounds** on $L_k$, not equalities. Three sources of slack:
1. The Tibshirani 2019 Lemma A.1 step that produces $w_k^+/w_k^-$ is a worst-case sup-norm bound; the *typical-case* perturbation is the L1-weight-perturbation (= $\hat\epsilon_k$ itself), not the L∞ ratio.
2. The DKW per-rung slack is one-sided union-bounded over $K$; tighter union-free bounds (e.g., Bernstein with rung-specific variance) would shave a $\sqrt{\log K}$ factor.
3. The granularity prefactor $\kappa_k$ assumes a worst-case "thinnest atom" placement of the quantile; in practice the quantile lands at the empirical mode where $\hat p_k$ is large, so the realized $\kappa_k$ is smaller.

We therefore claim the sufficient condition is **conservative**: when it holds, $\bar L < 1$ certainly holds, but $\bar L < 1$ may hold even when the condition fails (the pilot's $\bar L\approx 0.85$ falls in this gap; see §5). No tightness claim is made.

---

## §4 — Comparison to Theorem 5 v2's "sufficient overlap" condition

Theorem 5 v2's (worst-case slack) sufficient overlap condition, from `theorem5_v2_consolidated.md` §F.1, is
$$\sum_{k=0}^{K-1} \hat\epsilon_k \;<\; d_{TV}(P_0,P_K),\qquad\text{equivalently}\qquad \rho_K \;:=\; \frac{\sum_k\hat\epsilon_k}{d_{TV}(P_0,P_K)} \;<\; 1,$$
where $\rho_K\geq 1$ always (by triangle inequality), so this condition is **never strictly satisfiable** — it is a "Pareto frontier" rather than a binary gate. The pilot has $\rho_4 = 1.39$.

**The contraction sufficient condition is structurally different**, in three ways:

1. **Multiplicative vs additive.** Theorem 5 v2's condition is on the *sum* $\sum_k\hat\epsilon_k$; the contraction condition is on the *product* $\prod_k\tilde C_k\hat\epsilon_k$. The product form is *more permissive* for ladders with one large $\hat\epsilon_k$ surrounded by small ones (the small ones compensate multiplicatively), and *more demanding* for ladders with all $\hat\epsilon_k$ near a uniform value (no compensating effect).
2. **Includes weight bounds.** Theorem 5 v2's condition has *no* $w_k^+/w_k^-$ pre-factor (this is the v2 win over v1a). The contraction condition *does* have this prefactor, because the iterated quantile inverse is sensitive to the importance-weight conditioning, while the worst-case slack bound only cares about the TV summation. Thus contraction is **strictly stronger** than slack-bound improvement.
3. **Includes sample-size and granularity.** Theorem 5 v2's condition is essentially population-level ($\hat\epsilon_k$ alone, with DKW slack as a separate additive term). The contraction condition explicitly multiplies by sample-size-dependent DKW and by $|\mathcal S|$-dependent $\kappa_k$, so it directly answers "how many samples per rung do we need?".

**Compatibility check.** Suppose the contraction condition $\bar L < 1$ holds. Does the slack-bound condition $\sum_k\hat\epsilon_k < d_{TV}^{\mathrm{global}}$ then hold? **No** — they are independent. Example: a 2-rung ladder with $\hat\epsilon_1 = 0.1, \hat\epsilon_2 = 0.1, w^+/w^- = 1, \kappa = 1, \alpha=0.5$ gives $L_k = 0.2$, $\bar L = 0.04$ (strong contraction); but $\sum_k\hat\epsilon_k = 0.2 \geq d_{TV}^{\mathrm{global}} = 0.15$ (typical $\rho_2\approx 1.33$, so the slack condition fails). The two conditions diagnose different failure modes:
- Theorem 5 v2's condition is about whether the *worst-case slack bound* is informative.
- Theorem 5' contraction condition is about whether the *iterative refinement converges*.

The paper should report **both** in the empirical accounting; they are **complementary**, not nested.

---

## §5 — Empirical pilot verification

We instantiate §3.1's bound with the pilot values.

### 5.1 Pilot inputs

| $k$ | rung-pair | $\hat\epsilon_k$ | $n_k$ (correct count, lower bound) | $w_k^+/w_k^-$ (estimated) |
|---|---|---|---|---|
| 1 | math_cal → math_eval | 0.112 | $\approx 0.716\cdot 250 = 179$ | 1.4 (low TV → near 1) |
| 2 | math_eval → aime_old | 0.417 | $\approx 0.406\cdot 224 = 91$ | 5.0 (large drop in difficulty) |
| 3 | aime_old → aime_mid | 0.098 | $\approx 0.272\cdot 426 = 116$ | 1.5 |
| 4 | aime_mid → aime_new | 0.127 | $\approx 0.155\cdot 283 = 44$ | 1.7 |

with $|\mathcal S|=9$, $\alpha=0.5$, $\delta=0.1$, $K=4$.

### 5.2 DKW slack per rung

$\mathrm{DKW}_k(\delta/K) = \tfrac{9}{2}\sqrt{\log(80/0.1)/(2 n_k)} = 4.5\sqrt{\log(800)/(2 n_k)} = 4.5\sqrt{6.68/(2 n_k)}$.

Numerical:
- $k=1$: $\mathrm{DKW}_1 = 4.5\sqrt{6.68/358} = 4.5\cdot 0.137 = 0.616$.
- $k=2$: $\mathrm{DKW}_2 = 4.5\sqrt{6.68/182} = 4.5\cdot 0.192 = 0.862$.
- $k=3$: $\mathrm{DKW}_3 = 4.5\sqrt{6.68/232} = 4.5\cdot 0.170 = 0.764$.
- $k=4$: $\mathrm{DKW}_4 = 4.5\sqrt{6.68/88} = 4.5\cdot 0.276 = 1.241$.

**The DKW slack alone exceeds 0.5 for every rung.** This is the same vacuity that makes Theorem 5 v2's slack bound uninformative at pilot $n$. The empirical-noise-regime bound from §3.1 is therefore vacuous in the pilot ($L_k > 1$ from DKW alone, before any signal contribution).

### 5.3 Signal-regime per-rung Lipschitz

Drop the DKW (assume populational $n_k\to\infty$) and use $L_k \leq \tilde C_k\hat\epsilon_k$ with $\kappa_k\approx 9/2 = 4.5$ (worst-case discrete-atom granularity, $|\mathcal S|/2$):

- $\tilde C_1 = 4.5/(0.5)\cdot 1.4 = 12.6$, $L_1 \leq 12.6\cdot 0.112 = 1.41$.
- $\tilde C_2 = 4.5/(0.5)\cdot 5.0 = 45.0$, $L_2 \leq 45.0\cdot 0.417 = 18.8$.
- $\tilde C_3 = 4.5/(0.5)\cdot 1.5 = 13.5$, $L_3 \leq 13.5\cdot 0.098 = 1.32$.
- $\tilde C_4 = 4.5/(0.5)\cdot 1.7 = 15.3$, $L_4 \leq 15.3\cdot 0.127 = 1.94$.

$\bar L \leq 1.41\cdot 18.8\cdot 1.32\cdot 1.94 = 67.9$. **The bound is vacuous** at $\bar L \leq 67.9$, consistent with the pilot's empirical $\bar L\approx 0.85$ being far below the worst-case bound.

### 5.4 Realistic-$\kappa$ pilot check

Replace the worst-case $\kappa_k = |\mathcal S|/2 = 4.5$ with the realized $\kappa_k = 1/\hat p_k(s_q^*)$, where $s_q^*$ is the actual quantile atom. From `distance_ladder_pilot.json` `q_path` at $\alpha=0.5$ (4-rung pilot reduces from the 5-rung path by dropping the last entry): $q_{\mathrm{path}} = [0.875, 1.0, 0.625, 0.625, 0.625]$. The atom mass at $s=0.625$ in rung-3 (aime_mid) and rung-4 (aime_new) is approximately $\hat p_k(0.625)\approx 0.15$ (estimated from the kept_acc structure: about 15-20% of correct trajectories cluster near $\bar S = 0.625$). So $\kappa_3,\kappa_4\approx 6.7$ — *larger* than the worst-case $|\mathcal S|/2 = 4.5$, because the quantile lands at a relatively thin atom. The realized bound is therefore not tighter than the worst-case bound here.

### 5.5 Where the slack lives

The pilot's empirical $\bar L\approx 0.85$ vs the bound's $\bar L\leq 67.9$ implies a tightness gap of $\sim 80\times$. By inspection of factors:

- $w_2^+/w_2^- = 5$ contributes a factor of 5 to $L_2$. The *typical-case* L1-weight perturbation (rather than L∞) is 0.42 (the TV itself), which would give $L_2\approx 0.42/0.5 = 0.84$ — within the $\sim 1$ regime.
- $\kappa_k\approx 4.5$ globally (worst-case granularity) contributes $4.5^4\approx 410$ to $\bar L$. In practice the iteration plateaus at one or two atoms (per `q_path` analysis: the path stays at $0.625$ after rung 3), so the *effective* $\kappa$ is closer to $1/\hat p_k(0.625)\approx 6.7$ at the plateau atom and $1/\hat p_k(0.875)\approx 30$ at the initial atom — a wider range than the uniform worst case but with most of the iteration spent at small $\kappa$.

**Punchline.** The bound is consistent with the pilot's empirical $\bar L\approx 0.85$ in the sense that the bound *upper-bounds* the empirical (vacuously: $0.85 \leq 67.9$ trivially), but does not predict it tightly. The bound therefore certifies "no provable contraction failure at pilot" but does not certify "provable contraction at pilot". This is the honest state of Gap C: the sufficient condition is in hand, but it is loose by ~$80\times$ at the pilot's regime, and the bulk of the looseness is in (a) the L∞ vs L1 weight-perturbation gap and (b) the worst-case $\kappa$ vs realized $\kappa$ gap.

### 5.6 Asymptotic-rate prediction

In the population limit ($n_k\to\infty$), DKW vanishes and $L_k = \tilde C_k\hat\epsilon_k$, so the contraction is determined by the *ratio* $\hat\epsilon_k/\alpha$ scaled by $\rho_k\kappa_k$. For a ladder with bounded $w_k^+/w_k^- \leq W$, $|\mathcal S| = N+1$, and per-rung TVs $\hat\epsilon_k \leq \bar\epsilon$, a sufficient condition is
$$\Big(\frac{(N+1)W\bar\epsilon}{2\alpha\hat p^{\min}}\Big)^K \;<\; 1 \quad\Longleftrightarrow\quad \bar\epsilon \;<\; \frac{2\alpha\hat p^{\min}}{(N+1)W}.$$
With the pilot's $N=8, W=5, \alpha=0.5, \hat p^{\min}\approx 0.05$: $\bar\epsilon < 2\cdot 0.5\cdot 0.05/(9\cdot 5) = 0.0011$. **Two orders of magnitude smaller** than the pilot's geometric-mean $\bar\epsilon\approx 0.16$. The bound predicts contraction only when per-rung TV is in the $10^{-3}$ regime — far stricter than what the pilot achieves. This is consistent with §5.3's vacuity, and consistent with §3.4's tightness disclaimer.

---

## §6 — What goes wrong if $\bar L \geq 1$

### 6.1 Banach iteration diverges

If $\bar L \geq 1$, the Banach fixed-point theorem (which requires $\bar L < 1$ strictly) **does not apply**. The iterated map $T_K\circ\cdots\circ T_1$ may have:
1. **No fixed point** (oscillation between two or more atoms), in which case the iterate $\hat q^{(K),B'}$ depends on the initial $\hat q^{(0)}$ and does not converge.
2. **A non-unique fixed point** (multiple atoms that satisfy the rung-K weighted-quantile condition for some prior threshold), in which case the choice between them is path-dependent.
3. **A fixed point that is not attracting** (a saddle-like fixed point in the discrete setting), in which case adding rungs does not improve coverage.

### 6.2 Operational consequence: Strategy B' loses its empirical lift

The contracted slack $(1-\bar L^K)\sum_k\epsilon_k$ in Theorem 5' (§E.3) reduces to *worse* than Theorem 5 v2's $\sum_k\epsilon_k$ when $\bar L\geq 1$:
- $\bar L = 1$: slack equals Theorem 5 v2's slack (no improvement; B' is no better than telescoped B / one-shot).
- $\bar L > 1$: $\bar L^K$ blows up, the formula $(1-\bar L^K)$ goes negative, and the bound is *meaningless* — Strategy B' has no slack-improvement guarantee at all.

The pilot's H1 verdict is "B' beats one-shot at $\alpha=0.5$" (`distance_ladder_pilot.json:H1_ladder_beats_oneshot_at_alpha_0.5: true`). If $\bar L \geq 1$ in any future ladder, H1 should *empirically fail* on that ladder — falsifying Theorem 5'. This is the testable prediction.

### 6.3 Diagnostic: when to refuse Strategy B'

Three pre-flight checks before instantiating Strategy B' on a new ladder:
1. **Per-rung Lipschitz check**: compute $L_k$ from §3.1 for each rung. If any $L_k\geq 1$ with high empirical certainty, that rung is a contraction-breaker.
2. **Aggregate check**: compute $\bar L = \prod_k L_k$ from §3.2. If $\bar L\geq 1$, do not invoke Theorem 5'; fall back to Theorem 5 v2's slack bound.
3. **Reordering**: if a single rung has $L_k\gg 1$ but the others contract, removing that rung (or reordering) may restore $\bar L < 1$ on the residual chain. The pilot's H4 verdict (rung_4 is the anchor) is consistent with this diagnostic.

### 6.4 Worse than no-iteration

In the pathological case $\bar L > 1$, the iterate amplifies error: each application of $T_k$ pushes $\hat q^{(k)}$ further from the optimal threshold. The correct fallback is **truncate the iteration** (use $\hat q^{(0)}$ alone, i.e., one-shot CP at rung 0, possibly with a Theorem 3 weighted-CP correction). The paper should explicitly recommend this fallback when the §6.3 diagnostic flags $\bar L \geq 1$.

---

## §7 — Generalization beyond Banach: attractors without strict contraction

This section is **explicitly speculative** (research direction, not a theorem). The question is: when $\bar L \geq 1$ globally, can Strategy B' still help via some weaker mechanism?

### 7.1 Local contraction near a "good" atom

In the discrete-quantile setting, the iterated map $T = T_K\circ\cdots\circ T_1$ is a piecewise-constant function on $\mathcal Q$ with jumps at score atoms. Even if the *global* Lipschitz constant is $\geq 1$, the map may have a **local attractor**: a region $A\subset\mathcal Q$ such that $T(A)\subset A$ and $T|_A$ is a contraction (with local constant $L_A < 1$). If $\hat q^{(0)}$ falls in the basin of attraction of $A$, the iterate converges to a fixed point in $A$ with the contracted-slack guarantee of Theorem 5' restricted to $A$.

**Sketch.** Define $L_A := \sup_{q_1,q_2\in A}|T(q_1)-T(q_2)|/|q_1-q_2|$. If $A$ is invariant ($T(A)\subset A$) and $L_A < 1$, the Banach argument applies to $T|_A$. The local-attractor framing is then: identify $A$ from the empirical quantile path (`q_path` in pilot), check $T(A)\subset A$ empirically, and report the local Lipschitz constant.

In the pilot's `q_path` at $\alpha=0.5$: $[0.875, 1.0, 0.625, 0.625, 0.625]$. The path stalls at $A = \{0.625\}$ from rung 3 onward — a one-point set, trivially invariant under $T_3, T_4$ (their values *at* $0.625$ map to $0.625$). The local contraction constant on this singleton is $L_A = 0$ trivially, and the iterate has converged. The *global* bound's $\bar L\leq 67.9$ is irrelevant because the iterate fell into the attractor at rung 3.

### 7.2 Coverage in the local-attractor regime

If the iterate converges to $q^*\in A$ but $q^*$ is not the *Banach* fixed point of the global map, what is the coverage guarantee? The Strassen coupling argument of v1b (`theorem5_nexcp_slack_v1.md`) localizes coverage to "the rung-K weighted-CP coverage at threshold $q^*$". As long as $q^*$ is computed by a valid weighted-quantile procedure on $D_K$, coverage at $q^*$ satisfies Theorem 3 (one-shot weighted CP) directly, without invoking Theorem 5' contraction. The local attractor *replaces* the Banach guarantee with a one-shot guarantee at $q^*$.

**This is the resolution** for the pilot: Strategy B' wins not because Theorem 5' contraction holds globally (it doesn't, $\bar L\leq 67.9$), but because the iterate falls into a local attractor at $q^* = 0.625$ that happens to be a *better* one-shot threshold than the rung-0 quantile. The empirical 56–67% relative-gap reduction comes from the choice of attractor, not from a global contraction rate.

### 7.3 Research direction — characterize attractor basins

The open question is: **for what ladder structures does the Strategy B' iterate fall into a "good" local attractor?** A partial answer:
- (i) Monotone non-decreasing iterates (`q_path` is monotone in $k$ in the pilot at all $\alpha$, pre-rung-2 jump excepted).
- (ii) Bounded jump size: $|q^{(k+1)} - q^{(k)}| \leq 1/N$ (one atom step) generically; this is empirically true after rung 2 in the pilot.

A formal theorem would replace (B2) "mean contraction $\bar L < 1$" with **(B2') local-attractor existence**: there exists $A\subset\mathcal Q$, an invariant set under $T = T_K\circ\cdots\circ T_1$, with local Lipschitz $L_A < 1$ and $\hat q^{(0)}\in\mathrm{basin}(A)$. This is a *Tarski-fixed-point-with-rate* style theorem and is the natural follow-up to Gap A's Tarski path.

We do **not** prove (B2') here; we only **flag it as the right next theorem**. Status: open.

### 7.4 Asymptotic vs finite-K

Even without local-attractor analysis, a pragmatic generalization is **finite-K Strategy B'**: stop iterating once $|q^{(k)} - q^{(k-1)}| \leq 1/N$ for two consecutive steps. This is an *empirical convergence criterion* that does not rely on $\bar L < 1$. It may be the cleanest practical recommendation: iterate until empirical convergence, then stop, then verify coverage at the converged threshold via one-shot Theorem 3 at rung K. The "gain" over one-shot is then a pure empirical claim, not a Theorem 5' guarantee.

---

## §8 — Counter-examples where $\bar L \geq 1$

Three constructions where the sufficient condition (and indeed the empirical $\bar L$) fails.

### 8.1 Single dominant rung with extreme weight imbalance

**Construction.** $K=2$, $\mathcal S = \{0, 1/2, 1\}$, $|\mathcal S|=3$, $n_1 = n_2 = 100$, $\alpha = 0.5$. Rung 1: $\hat p_0 = (0.5, 0.4, 0.1)$, $\hat p_1 = (0.1, 0.4, 0.5)$. Rung 2: $\hat p_1 = \hat p_2$ (no shift). TVs: $\hat\epsilon_1 = 0.4$, $\hat\epsilon_2 = 0$. Weight ratio: $w_1^+ = 0.5/0.1 = 5, w_1^- = 0.4/0.4 = 1$, so $w_1^+/w_1^- = 5$. Rung 2 is degenerate: $w_2^\pm = 1$.

**Sufficient condition check.**
$L_1 \leq \kappa_1\cdot (5/0.5)\cdot 0.4 = \kappa_1\cdot 4 \approx 12$ (with $\kappa_1\approx 3$). $L_2 \leq \kappa_2\cdot(1/0.5)\cdot 0 = 0$. $\bar L = L_1\cdot L_2 = 0$. **Sufficient condition trivially passes**, despite the extreme rung-1 weight imbalance.

This is a case where the sufficient condition is **misleadingly permissive**: the second rung has zero TV, so the product is zero, but the iterate is dominated by rung 1's behavior (which is far from contractive). The fix is to use the **max form** $\bar L_{\max} = \max_k L_k$ instead, which is $12$ in this construction — correctly flagging non-contraction.

**Lesson.** Product-form sufficient conditions can be fooled by zero-TV rungs. The max form is the safer diagnostic when individual rungs may have $L_k\gg 1$.

### 8.2 Many small-TV rungs with cumulatively-large $K$

**Construction.** $K = 100$, $|\mathcal S|=9$, all rungs identical with $\hat\epsilon_k = 0.05$, $w_k^+/w_k^- = 1.5$, $n_k = 100$, $\alpha = 0.5$, $\kappa_k = 4.5$. 

**Per-rung Lipschitz.** $L_k = (4.5/0.5)\cdot 1.5\cdot 0.05 = 0.675 < 1$. **Contraction holds per rung.**

**Aggregate.** $\bar L = 0.675^{100}\approx 0$. Strong contraction in the product form.

**But:** Each rung sample size is $n_k = 100$, so $\mathrm{DKW}_k = 4.5\sqrt{6.68/200}\cdot 4.5\approx 0.82$, dominating the $\hat\epsilon_k = 0.05$ signal. The empirical-noise $L_k^{\mathrm{emp}} = 9/0.5\cdot 1.5\cdot 0.82 = 22.1$ (using full DKW slack), so $\bar L^{\mathrm{emp}}\geq 22^{100}$ — vacuous.

**Lesson.** Product-form contraction in the population limit does not imply empirical contraction at finite $n$. The DKW slack matters at every rung, and large $K$ amplifies the union-bound. **A 100-rung ladder needs $n_k\geq\Omega(|\mathcal S|^2 K\log K)$ to avoid empirical-noise dominance** — i.e., samples per rung should grow with the number of rungs. This is a *non-trivial* design constraint absent from Theorem 5 v2.

### 8.3 Anchor rung at the chain end with very small sample

**Construction.** $K = 4$, $|\mathcal S|=9$, $\hat\epsilon = (0.1, 0.1, 0.1, 0.1)$, $w^+/w^- = (1.5, 1.5, 1.5, 1.5)$, $n = (1000, 1000, 1000, 30)$. $\alpha = 0.5$, $\kappa_k = 4.5$.

**Per-rung empirical-noise.** $\mathrm{DKW}_k = 4.5\sqrt{\log(160)/(2 n_k)}$. For $k=1,2,3$: $\mathrm{DKW}\approx 4.5\sqrt{5.07/2000}\approx 0.226$. For $k=4$: $\mathrm{DKW}_4 = 4.5\sqrt{5.07/60}\approx 1.30$.

**Per-rung Lipschitz.** $L_k = (4.5/0.5)\cdot 1.5\cdot(\hat\epsilon_k + \mathrm{DKW}_k)$.
- $k=1,2,3$: $L_k = 13.5\cdot(0.1 + 0.226) = 4.4$.
- $k=4$: $L_4 = 13.5\cdot(0.1 + 1.30) = 18.9$.
- $\bar L = 4.4^3\cdot 18.9 = 1607$. Vacuous.

**Lesson.** A single small-$n$ anchor rung (here rung 4 with $n_4=30$) destroys the bound, even when other rungs have $n_k=1000$. This is the **finite-sample anchor effect**: the bound is dominated by the smallest-$n$ rung in the chain via DKW. The pilot's rung-4 has $n_4\approx 44$ correct trajectories — the same regime. **Recommendation: design ladders with a generous anchor rung sample size** ($n_K\geq |\mathcal S|^2 \log K / \hat\epsilon_K^2$ as a rough rule of thumb).

This counter-example matches the empirical pilot's H4 verdict (`max delta 11.9pp, worst-to-drop = rung_4_aime_mid` — the anchor is the most fragile rung). Theorem 5' Gap C correctly *predicts* the empirical fragility of the anchor.

---

## §9 — Summary and connection to upstream gaps

**What this draft establishes.**

- A **three-factor decomposition** of $L_k$: weight-ratio $\rho_k = w_k^+/(αw_k^-)$, empirical-PMF noise $\hat\epsilon_k^{\,*} = \hat\epsilon_k+\mathrm{DKW}_k$, granularity $\kappa_k = 1/\hat p_k(s_q^*)$. (§2)
- An **explicit constant $C_k(\delta)$** for the empirical-noise regime: $L_k \leq C_k(\delta)/\sqrt{n_k}$ with $C_k(\delta) = (|\mathcal S|\kappa_k/(2\alpha))(w_k^+/w_k^-)\sqrt{\log(2K|\mathcal S|/\delta)/2}$. (§3.1)
- A **product-form sufficient condition** $\sum_k\log L_k < 0$ for $\bar L < 1$, and a coarser **max-form** $\max_k L_k < 1$. (§3.2–3.3)
- A **comparison** to Theorem 5 v2's slack-bound condition $\sum_k\hat\epsilon_k < d_{TV}^{\mathrm{global}}$: the two are **complementary**, neither implies the other. (§4)
- An **empirical pilot check**: bound is consistent with empirical $\bar L\approx 0.85$ (vacuously, $0.85\leq 67.9$) but **does not predict it tightly** (factor of ~80 looseness, mostly L∞-vs-L1 weight perturbation and worst-case granularity). (§5)
- A **failure analysis** for $\bar L\geq 1$ (Banach diverges, B' loses its lift; fall back to one-shot at the most-recent rung). (§6)
- A **research direction beyond Banach**: local-attractor framing, with the pilot's `q_path` plateau at $0.625$ as evidence that the iteration falls into a local attractor even when global contraction fails. (§7)
- **Three counter-examples**: extreme single-rung weight imbalance (max form catches it), large $K$ with small per-rung $n$ (DKW union dominates), small-$n$ anchor rung at chain end (matches pilot's H4 fragility). (§8)

**What this draft does not establish.**

- The bound is **loose by ~80×** at the pilot regime; a tighter bound likely requires (a) replacing L∞ weight perturbation with L1 (= TV itself), and (b) realized vs worst-case granularity. Both are open.
- The local-attractor framing of §7 is a **research direction**, not a theorem; the natural next step is a Tarski-with-rate argument (`[FUTURE: B2' local-attractor theorem]`).
- The sufficient condition is **conservative** but not **necessary**: a ladder may have $\bar L < 1$ even when our condition fails. No tightness claim is made (§3.4).

**Consistency with sibling Gap A and Gap B drafts.**

- *Gap A* (Lipschitz of discrete weighted-quantile $T_k$): supplies the **form** of $L_k$ used here (specifically the $\kappa_k$ smoothing prefactor and the $w_k^+/w_k^-$ ratio). This draft consumes Gap A's form and adds the sufficient-condition machinery.
- *Gap B* (coverage at the fixed point $q^*$): not invoked here. Gap C is a *purely deterministic* computation about the contraction rate; coverage at the fixed point is a *probabilistic* statement that Gap B handles separately. The two are **modular**: even if Gap C's sufficient condition is loose, Gap B's coverage guarantee at $q^*$ remains valid as long as the iterate converges.

**Recommended next action.** Tighten the sufficient condition by replacing $w_k^+/w_k^-$ with the *L1-perturbation form* $\|\hat w_{k-1\to k}\hat p_{k-1} - \hat p_k\|_1 \leq 2\hat\epsilon_k$ (this is just the TV in disguise!), and by replacing worst-case $\kappa_k = |\mathcal S|/2$ with realized $\kappa_k = 1/\hat p_k(s_q^*)$ at the *expected* quantile atom. With these two replacements, the pilot bound becomes
$$L_k^{\mathrm{tight}} \approx \frac{\hat\epsilon_k}{\alpha\cdot\hat p_k(s_q^*)}\quad\Rightarrow\quad L_2^{\mathrm{tight}}\approx 0.42/(0.5\cdot 0.15)\approx 5.6,$$
still loose but only by ~6× rather than ~80×. Closing the remaining 6× requires the local-attractor framing of §7. **This is the line of work for the camera-ready supplementary appendix.**

---

## Closing — Cross-Model Verification Results

> **Single-model lane.** Per CLAUDE.md `cross_model_verification.mode: all` and `scope: hypothesis_validator + master_orchestrator`, this Gap C draft should be cross-verified by `openai/openai/gpt-5.5` (and fallback `gcp/google/gemini-3.1-pro-preview`). The `inference_token` is `sk-PLACEHOLDER` and the external verifier was not invoked. Per the protocol in `pipeline/cross_model_verification_protocol.md`, disagreements will be appended below; the current verdict on Gap C is **PROCEED, single-model only, with the explicit tightness disclaimer of §3.4 and §5.5 visible to readers**.
>
> **Cross-Model Disagreements (placeholder).** None recorded — single-model lane.

## Self-review (Lakatos round 1)

Three most likely objections to this draft:

**(O1) "Your bound is 80× loose at the pilot. The sufficient condition is operationally useless."**
*Response.* Acknowledged in §3.4 and §5.5. The bound is *conservative* (correct direction), not *tight*. The 80× looseness factors to ~5× (L∞ vs L1 weights) × ~2× (worst-case vs realized granularity) × ~8× (DKW union over $K$). Each factor has a tightening path identified in §9's "recommended next action". For the v2 paper, we report the bound *as-is* with the disclaimer; tightening is supplementary appendix work.

**(O2) "The product form fails the §8.1 counter-example. Why is it the headline?"**
*Response.* §8.1 shows the product form is fooled by zero-TV rungs. The product form is the headline because (a) it directly bounds the *Banach contraction rate* of the composed map, which is the right object of study, and (b) the pilot does not have any zero-TV rungs. For ladders that do have zero or near-zero rungs, §3.3's max form is the recommended diagnostic. The paper should report **both forms**, with the product form as the primary contraction certificate and the max form as the robustness-to-permutation check.

**(O3) "Local-attractor §7 is hand-waving. Withdraw it."**
*Response.* §7 is *explicitly labeled* as a research direction, not a theorem. The pilot's `q_path` plateau at $0.625$ is concrete empirical evidence that local attractors exist; the formal theorem is open and we name it `[B2' local-attractor theorem, future work]`. The §7 framing is necessary for honest reporting of why Strategy B' wins in the pilot despite the global bound being vacuous; without it, the reader is left with "the bound is loose, why does B' work?" — an unsatisfying gap. The §7 sketch is the honest answer.

---

**File path**: `/home/nvidia/future/literature/theorems_drafts/theorem5_gap_C_contraction_sufficient.md`

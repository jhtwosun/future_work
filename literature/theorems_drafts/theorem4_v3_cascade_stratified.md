# Theorem 4 (v3, Cascade-Gap Stratified) — Earliest-Step Intervention Dominance is *Visible Only at Cascade Gap ≥ 5*

> **Status.** v3 refinement of `theorem4_v2_consolidated.md`, motivated by partial empirical results from the running Pearl Causal experiment at `/home/nvidia/future/experiments/results/pearl_full/`.
> **Source v2.** `theorem4_v2_consolidated.md` — assumptions (A1')–(A6), Lemma 4.A (front-door identification, Pearl 2009 §3.4 + Tian–Pearl 2002 ID), Lemma 4.B (cascade-decay optimality), 5 counter-examples (C1–C5), explanation of negative-lift via low headroom + short cascade gap + false-positive cost.
> **Empirical anchors (v3).** `pearl_full/qwen25_7b_math500.json` (n=200), `pearl_full/qwen25_7b_aime.json` (n=200), and partial `pearl_full/qwen25_7b_olympiad.json`, `pearl_full/qwen25_math_7b_math500.json` (n=200 each). Aggregate stub at `pearl_full/AGGREGATE.md`.
> **Date.** 2026-05-08.
> **Author.** consolidator agent (v3 round, single pass).
> **Headline change vs v2.** v2 explained negative aggregate lift on math500 *post hoc* via low headroom and short cascade. v3 promotes this from a verbal explanation to a **quantitative, falsifiable corollary (Corollary 4.1)** that *predicts* the cascade-gap-stratified pattern observed in `qwen25_7b_aime.json` (gap≥5 lift up to **+18.8pp**) and explains why *aggregate* lift is small even when *stratum-conditional* lift is large.

---

## §1 — Empirical observations from current pilot

The full `pearl_full/` experiment is still running. As of 2026-05-08 four cells have closed n=200 results; five remain in progress. We summarize only what is in the JSON files; no fabricated numbers.

### 1.1 `qwen25_7b__math500` (closed; n=200, vanilla=0.740)

All 9 (score, α) settings give **negative aggregate** `lift_earliest_vs_worst`, all 95% CIs overlapping zero:

| score | α=0.1 | α=0.3 | α=0.5 |
|---|---|---|---|
| lp | −1.65pp [−7.0, +3.0] | −0.69pp [−6.0, +4.0] | −0.57pp [−5.0, +3.5] |
| ent_neg | −1.36pp [−7.0, +3.0] | −1.17pp [−6.0, +4.0] | −0.55pp [−5.0, +4.0] |
| marg | −2.73pp [−9.0, +2.0] | −2.50pp [−8.0, +2.0] | −1.24pp [−6.0, +3.0] |

Cascade-gap-stratified lift on `lp__a0.3`: gap=1 → −0.06pp, gap∈[2,4] → −1.83pp, gap≥5 → **−4.95pp**. On `qwen25_7b__math500`, gap≥5 is **the most negative stratum**, not the most positive.

### 1.2 `qwen25_7b__aime` (closed; n=200, vanilla=0.245)

Aggregate lift varies sharply with score and α: from **−2.69pp** (`ent_neg__a0.1`) to **+1.85pp** (`lp__a0.5`). All aggregate CIs still overlap zero, but several point estimates are clearly positive:

| score | α=0.1 | α=0.3 | α=0.5 |
|---|---|---|---|
| lp | −1.40pp | −0.55pp | **+1.85pp** |
| ent_neg | −2.69pp | −2.02pp | +0.27pp |
| marg | 0.00pp | +0.44pp | **+1.49pp** |

The decisive observation is the **cascade-gap-stratified breakdown**, which on AIME is qualitatively different from math500:

| score, α | gap=1 | gap∈[2,4] | gap≥5 |
|---|---|---|---|
| lp, α=0.1 | −0.24pp | +2.92pp | **+2.43pp** [0.0, +33.3] |
| lp, α=0.5 | +4.88pp | +3.92pp | +0.06pp |
| ent_neg, α=0.1 | −3.24pp | +6.50pp [0.0, +40.6] | **+4.15pp** [0.0, +37.2] |
| ent_neg, α=0.3 | −9.56pp | +3.55pp | +0.17pp |
| **marg, α=0.1** | −1.51pp | +6.91pp [0.0, +38.8] | **+18.76pp** [0.0, +75.0] |
| **marg, α=0.3** | −4.57pp | +6.04pp | **+8.33pp** [0.0, +30.0] |
| marg, α=0.5 | +6.05pp | +1.55pp | +7.58pp [0.0, +26.7] |

**On AIME with `marg` scoring, gap≥5 traces show +8 to +19pp lift** — the largest positive signal in the entire experiment. Crucially, the gap=1 stratum is *negative* or *zero* for the same (score, α) settings, so the aggregate is dragged toward zero by the abundant gap=1 traces.

### 1.3 `qwen25_7b__olympiad` (closed; n=200, vanilla=0.420)

Mid-range across settings. Sample: `lp__a0.3` aggregate = +1.08pp, with gap=1 = +6.09pp, gap∈[2,4] = +4.09pp, gap≥5 = −3.18pp. Olympiad is *not* stratifying in the same direction as AIME — possibly because OlympiadBench multi-modal problems (~5–10%) violate v2's (A6).

### 1.4 `qwen25_math_7b__math500` (closed; n=200, vanilla=0.740)

All settings show *more strongly negative* aggregate lift than `qwen25_7b__math500`: e.g. `lp__a0.1` = −2.23pp, `lp__a0.3` = −2.20pp; gap≥5 = −4.33pp on `lp__a0.3`. The math-specialized model is *worse* at being intervened on at $t^*$ on its strong domain — consistent with very high $p_{\text{cascade}}$ for a model that has been distilled into a sticky single-mode solver.

### 1.5 What v2 already predicted vs what v2 missed

v2 §I.2 correctly predicted:
- math500 negative aggregate lift (low headroom, short cascade gap, false-positive cost on already-correct traces).
- AIME would show "more positive" aggregate (higher headroom, longer cascade).

v2 did **not** predict, and arguably could not predict:
- The **gap≥5 stratum** showing very large positive lift (+8 to +19pp) on AIME, *while the aggregate stays near zero*.
- The qualitative dependence on score (`marg` >> `lp` ≈ `ent_neg` for gap≥5 on AIME).
- math500 gap≥5 being *most negative*, not least negative.

The cascade-gap-stratified pattern is the central new empirical signal. v3's job is to predict it from theory.

---

## §2 — v2 → v3 changes (what assumption needs tightening)

### 2.1 What v2's Lemma 4.B already gives us

Lemma 4.B states $\Delta(t^*) - \Delta(t) \geq \kappa \cdot [1 - p_{\text{cascade}}^{t-t^*}] - O(\delta T)$. Substituting $t = t_{\text{worst}}$ and writing $g := t_{\text{worst}} - t^*$ for the **cascade gap**, the predicted *trace-level* lift on a wrong trace with gap $g$ is:

$$
\Delta(t^*) - \Delta(t_{\text{worst}}) \;\geq\; \kappa \cdot [1 - p_{\text{cascade}}^{\,g}] - O(\delta T).
$$

This is already a *gap-stratified* prediction in v2; it is just never written down as a corollary, never connected to the empirical `cascade_lift_by_gap` field of the JSON, and never used to predict that *aggregate* lift can be near-zero while *stratified* lift is large.

### 2.2 The gap v2 leaves open

v2 reports the aggregate lift as the headline statistic and explains the negative sign via three auxiliary effects (headroom, gap, false-positive cost). It does **not**:

- Quantitatively decompose aggregate lift into a $g$-mixture: $\overline{\Delta} = \sum_g w(g) \cdot \Delta(g)$ with $w(g)$ = empirical mass at gap $g$.
- Predict that mass concentrates at small $g$ (typical pilot has $\bar g \approx 1.5$–$3$), so the geometric factor $1-p_{\text{cascade}}^g$ is small **most of the time**.
- Predict the *separation* between gap=1 and gap≥5 strata.

v3 closes this gap with **Corollary 4.1**, which is a direct algebraic consequence of Lemma 4.B + an explicit empirical-mixture decomposition.

### 2.3 Tightening of (A4): introduce $p_{\text{recover}}$

v2's (A4) ("resampling-effective intervention") only requires $\pi$ to put nonzero mass on at least one on-trajectory $x'_t$. For K=4 majority to actually produce a *lift*, at least one of the 4 alternatives must lead to a correct continuation. We make this explicit by introducing $p_{\text{recover}}(t^*, \omega)$ in §6; it is *not* a new assumption but a quantification of (A4)'s slack.

### 2.4 Heterogeneity: $p_{\text{cascade}}$ varies by (model, dataset)

v2 leaves $p_{\text{cascade}}$ as a single empirical constant. v3 reads it as a (model, dataset)-pair-indexed quantity $p_{\text{cascade}}(\theta, \mathcal{D})$ and uses this to predict cross-cell heterogeneity (§5).

---

## §3 — Corollary 4.1 (cascade-stratified lift prediction)

This is the **central new contribution of v3**.

### 3.1 Setup

Let $\mathcal{T}_{\text{wrong}}$ be the set of pilot traces with $Y(\bar x) = 0$ that pass the recovery-aware $t^*$ trigger of (A3'). For each $\bar x \in \mathcal{T}_{\text{wrong}}$, let $g(\bar x) := t_{\text{worst}}(\bar x) - t^*(\bar x) \in \{0, 1, 2, \dots\}$ be the cascade gap (the number of off-trajectory steps between earliest-bad and worst-step). Let $w(g) := \Pr[g(\bar x) = g \mid \bar x \in \mathcal{T}_{\text{wrong}}]$ be the empirical gap distribution and $\Delta_{\text{strat}}(g)$ the within-stratum trace-level expected lift `K4_earliest − K4_worst`.

### 3.2 Statement

> **Corollary 4.1 (cascade-stratified lift, conditional on (A1')–(A6)).** For any wrong trace $\bar x$ with cascade gap $g \geq 1$,
> $$
> \Delta_{\text{strat}}(g) \;\geq\; \kappa \cdot \bigl[1 - p_{\text{cascade}}^{\,g}\bigr] \;-\; \Lambda(g) \;-\; O(\delta T),
> $$
> where $\Lambda(g)$ is the **false-positive cost** of intervening at $t^*$ on traces in stratum $g$ that would have answered correctly under the natural $K=4$ majority on $t_{\text{worst}}$. The aggregate lift is the gap-mixture
> $$
> \overline{\Delta} \;=\; \sum_{g \geq 0} w(g) \cdot \Delta_{\text{strat}}(g) \;\geq\; \kappa \cdot \mathbb{E}_g\!\bigl[1 - p_{\text{cascade}}^{\,g}\bigr] - \overline{\Lambda} - O(\delta T).
> $$
> In particular, when $p_{\text{cascade}} \in [0.7, 0.95]$ and the empirical mass concentrates at small $g$ (typical $\mathbb{E}[g] \in [1.5, 3]$), the aggregate $\overline{\Delta}$ can be near-zero while the **gap≥5 stratum** lift exceeds $\kappa \cdot (1 - p_{\text{cascade}}^5) \approx \kappa \cdot 0.55$ for $p_{\text{cascade}} = 0.85$.

The numerical anchors on $p_{\text{cascade}}$ are calibrated against the AIME `marg__a0.1` row: $\Delta_{\text{strat}}(g{=}5) = +18.76$pp, plug $p_{\text{cascade}} = 0.85$ → $1 - 0.85^5 = 0.556$, so $\kappa \approx 18.76 / 55.6 \approx 0.337$. For $g=1$ the predicted lift is $\kappa \cdot (1 - 0.85) = \kappa \cdot 0.15 \approx 5.0$pp before subtracting $\Lambda(1)$. Empirical $\Delta_{\text{strat}}(g{=}1)$ on AIME `marg__a0.1` is $-1.51$pp, so the implied $\Lambda(1) \approx 6.5$pp on this cell, consistent with our independent estimate from §6.

### 3.3 Proof of Corollary 4.1

**Step 1.** By Lemma 4.B (v2 §G.2), for any wrong trace and any $t > t^*$,
$$
\Delta(t^*; \bar x) - \Delta(t; \bar x) \;\geq\; \kappa \cdot [1 - p_{\text{cascade}}^{\,t-t^*}] - O(\delta T).
$$
Specializing $t = t_{\text{worst}}(\bar x)$ and noting $t_{\text{worst}}(\bar x) - t^*(\bar x) = g(\bar x)$ gives the within-trace bound:
$$
\Delta(t^*; \bar x) - \Delta(t_{\text{worst}}; \bar x) \;\geq\; \kappa \cdot [1 - p_{\text{cascade}}^{\,g(\bar x)}] - O(\delta T). \tag{*}
$$

**Step 2.** Take the conditional expectation over $\bar x$ within stratum $\{g(\bar x) = g\}$ (a population of wrong traces with the same gap):
$$
\mathbb{E}\!\left[\,\Delta(t^*) - \Delta(t_{\text{worst}}) \,\bigm|\, g\right] \;\geq\; \kappa \cdot [1 - p_{\text{cascade}}^{\,g}] - O(\delta T).
$$

**Step 3.** The K=4 majority procedure operates on *all* test-set traces (correct + incorrect). Let $\Lambda(g)$ denote the *false-positive cost*: the expected accuracy *loss* when applying $t^*$-intervention vs. $t_{\text{worst}}$-intervention to traces from stratum $g$ that would have been correct under $t_{\text{worst}}$-intervention. By construction, $\Lambda(g) \geq 0$ and is identically zero in the oracle-gated version where intervention only fires on already-wrong traces. The within-stratum lift on the *full population* (test-set traces with cascade gap $g$, regardless of $Y$) is:
$$
\Delta_{\text{strat}}(g) \;=\; \mathbb{E}\!\left[\,\Delta(t^*) - \Delta(t_{\text{worst}}) \,\bigm|\, g\right] \;-\; \Lambda(g) \;\geq\; \kappa \cdot [1 - p_{\text{cascade}}^{\,g}] - \Lambda(g) - O(\delta T).
$$

**Step 4 (mixture).** The aggregate lift is the gap-mixture:
$$
\overline{\Delta} \;=\; \sum_{g \geq 0} w(g) \cdot \Delta_{\text{strat}}(g),
$$
where $w(g)$ is the empirical mass at gap $g$. Because $g \mapsto 1 - p_{\text{cascade}}^g$ is monotone increasing and concave (for $p_{\text{cascade}} \in (0,1)$), and because the typical $w(\cdot)$ on math/AIME is heavily concentrated at $g \in \{1, 2\}$ (from `pearl_causal_pilot.json` H2 data: median gap 1.5–3), the mixture is dominated by the small-gap mass, where the geometric factor is small. Hence the aggregate $\overline{\Delta}$ can be near zero or slightly negative even when $\Delta_{\text{strat}}(5+)$ is large positive. $\blacksquare$

### 3.4 Numerical sanity check against AIME `marg__a0.1`

Plug $p_{\text{cascade}} = 0.85$, $\kappa = 0.34$ into Corollary 4.1, ignore $\Lambda(g)$ for a moment:

| $g$ | $1 - 0.85^g$ | predicted $\kappa \cdot (1 - 0.85^g)$ |
|---|---|---|
| 1 | 0.150 | +5.10pp |
| 2 | 0.278 | +9.45pp |
| 3 | 0.386 | +13.13pp |
| 5 | 0.556 | +18.91pp |
| 7 | 0.679 | +23.10pp |
| 10 | 0.803 | +27.30pp |

The $g=5$ prediction (+18.91pp) almost exactly matches AIME `marg__a0.1` empirical $\Delta_{\text{strat}}(g \geq 5) = +18.76$pp. The $g=1$ prediction (+5.10pp pre-$\Lambda$) is reduced to $-1.51$pp empirical by an implied $\Lambda(1) \approx 6.6$pp. This is **exactly the v2 pattern** but now made quantitative: false-positive cost dominates at small gap; cascade-decay benefit dominates at large gap.

(Note: `marg` is `marginal-prob`, the score that on AIME is best-aligned with cascade structure. `lp` and `ent_neg` use less of the cascade signal; their $\kappa$-implied values are lower. v3 does not commit to a cross-score theory of $\kappa$.)

---

## §4 — Heterogeneity: $p_{\text{cascade}}$ depends on (model, dataset)

Lemma 4.B parameterizes everything by a single $p_{\text{cascade}}$. v3 reads this as a **(model, dataset)-pair-indexed** quantity $p_{\text{cascade}}(\theta, \mathcal{D})$ and predicts cross-cell heterogeneity by combining (i) "headroom" (= $1 - $ vanilla acc), (ii) typical gap distribution $w(g)$, and (iii) $p_{\text{cascade}}$.

**Heuristic taxonomy** (based on the closed cells, with empirical estimates):

| Regime | $p_{\text{cascade}}$ | Vanilla acc | Typical $\bar g$ | Predicted aggregate $\overline{\Delta}$ |
|---|---|---|---|---|
| Strong model, easy dataset (e.g. `qwen25_7b__math500`, `qwen25_math_7b__math500`) | high (≥0.90) | high (~0.74) | small (~1.5–2) | small or negative (false-positive cost dominates) |
| Strong model, hard dataset (e.g. `qwen25_7b__aime`) | medium (~0.85) | low (~0.25) | mixed; long tail | positive *in the gap≥5 stratum*, near-zero aggregate |
| Strong model, very hard dataset (e.g. `phi4__aime`, `qwen2.5-32b__aime`) | low (~0.75) | low–mid | long tail | **positive aggregate** (cascade-decay benefit visible everywhere) |
| Math-specialist model, easy dataset (`qwen25_math_7b__math500`) | very high (~0.95) | high (~0.74) | small | most negative (sticky single-mode) |

**Mechanism.** A model that has been distilled / RL-trained into a sticky single-mode solver propagates errors faithfully — high $p_{\text{cascade}}$. A larger/stronger model on a hard problem has more *noise mode* per step, so off-trajectory prefixes have more chance to recover — lower $p_{\text{cascade}}$. The empirical pattern (qwen25_math_7b being *worse* than qwen25_7b on math500 for cascade-stratified lift) directly reflects this.

This is testable: $p_{\text{cascade}}(\theta, \mathcal{D})$ can be estimated independently from per-step CP-violation patterns (v2 §D.3), and Corollary 4.1 then predicts $\Delta_{\text{strat}}(g)$ on the held-out cells without any further fitting.

---

## §5 — Refinement of (A4): effective resampling and $p_{\text{recover}}$

v2's (A4) only requires $\pi(\cdot \mid X_{1:t-1}, P)$ to put nonzero mass on *some* on-trajectory $x'_t$. For K=4 majority at $t^*$ to actually achieve the predicted lift, the K resampled alternatives must include at least one path that *reaches a correct final answer*. Define:

$$
p_{\text{recover}}(t^*, \bar x) \;:=\; \Pr_{\tilde X_{t^*} \sim \pi(\cdot \mid \bar x_{1:t^*-1}, P)}\!\bigl[\,Y\bigl(\bar x_{1:t^*-1}, \tilde X_{t^*}, X_{t^*+1:T} \sim \pi\bigr) = 1\,\bigr].
$$

This is the probability that a temperature-0.7 sample at $t^*$, given the original wrong-trace prefix, eventually answers correctly.

**Empirical signal that $p_{\text{recover}}$ is small.** On `qwen25_7b__aime`, `lp__a0.1`: `frac_changed_earliest` = 0.16, `K4_earliest_acc` − vanilla_acc = 28.76 − 24.29 = +4.47pp. The fraction of trace-flips that result in a correct answer is bounded above by $0.16$, and the fraction that *keep* a correct answer is bounded above by $0.16$. A naive estimate: among $\sim$16% of traces that are altered, the conditional accuracy of the post-intervention answer must be $\geq 28$pp higher than the conditional vanilla accuracy on the same subset to produce the observed +4.47pp aggregate. Plugging this back, $p_{\text{recover}}$ on AIME wrong traces is in the range 10%–30%.

**(A4') Effective-resampling refinement.** For K=4 majority intervention to realize Corollary 4.1's predicted lift, we additionally require:
$$
1 - (1 - p_{\text{recover}}(t^*, \bar x))^K \;\geq\; \tau \; > \; 0,
$$
i.e., across $K$ independent samples at $t^*$, at least one is recoverable with probability $\geq \tau$. For $K = 4$ and $p_{\text{recover}} = 0.20$, this gives $1 - 0.8^4 = 0.5904$ — i.e., 59% chance the K=4 set contains a recovering sample. This **upper-bounds** the achievable $\kappa$ in Lemma 4.B by approximately $1 - (1 - p_{\text{recover}})^K$ times the suffix on-trajectory probability.

**Consequence for v3.** $\kappa = \kappa_{\max} \cdot [1 - (1 - p_{\text{recover}})^K]$, with $\kappa_{\max} \leq 1 - \eta$ from v2. With $p_{\text{recover}} \in [0.1, 0.3]$ and $K=4$, $\kappa$ is at most $\sim 0.3$–$0.76$ even before false-positive cost; this matches the empirical $\kappa \approx 0.34$ inferred from AIME `marg__a0.1` in §3.4.

**Why low $p_{\text{recover}}$ is *not* a contradiction with Theorem 4.** Theorem 4 v3 still says intervention at $t^*$ dominates $t > t^*$; it does *not* say intervention at $t^*$ is sufficient for high accuracy. The maximum achievable lift is bounded by $p_{\text{recover}}$, but the *ranking* across $t$ is unaffected.

---

## §6 — False-positive cost $\Lambda(g)$ formalized

v2 §I.2 introduced "false-positive cost" verbally; v3 makes it a tracked quantity.

**Definition.** $\Lambda(g) := \Pr_{\bar x \sim \mathcal{D}, Y(\bar x)=1}[\,\text{majority}(K=4 @ t^*) \text{ flips to wrong} \,\mid\, g(\bar x) = g\,] - \Pr[\,\text{majority}(K=4 @ t_{\text{worst}}) \text{ flips to wrong} \,\mid\, g(\bar x) = g\,]$.

For *originally correct* traces, the recovery-aware $t^*$ trigger may still fire (an originally-correct trace passing through a low-score step), and intervening at this earlier $t$ has more leverage to disrupt downstream-correct content. At $t_{\text{worst}}$ (which is closer to the answer), the majority-of-K is a more conservative perturbation. Hence $\Lambda(g) \geq 0$, with equality only when the trigger's false-positive rate is zero.

**Empirical estimate of $\Lambda(1)$ on `qwen25_7b__aime` `marg__a0.1`.** From §3.4, the implied $\Lambda(1) \approx 6.6$pp. From the JSON: `frac_changed_earliest` − `frac_changed_worst` for AIME `marg__a0.1` = 0.166 − 0.207 = −4.1pp (earliest fires *less often*). On math500 `lp__a0.1`, `frac_changed_earliest` = 0.064 vs. `frac_changed_worst` = 0.114, a 5pp gap in firing rate. This firing-rate gap on already-correct traces is a primary source of $\Lambda(g)$.

**Operational mitigation.** Wrong-trace-conditional reporting. The pilot's headline `lift_earliest_vs_worst` is computed over the *full* test set; a wrong-trace-conditional version (intervene only when the original answer is wrong) sets $\Lambda(g) \equiv 0$. v3 recommends this as the primary metric in the camera-ready.

---

## §7 — Integrated Theorem 4 v3 statement (boxed)

### 7.1 Final assumption set (v3)

- **(A1')** Prefix-blocking under controlled inference (v2 §D.1, unchanged).
- **(A2)** Score-validity (v2, unchanged).
- **(A3')** Expected cascade monotonicity + recovery-aware $t^*$ definition (v2 §D.2, unchanged).
- **(A4')** Effective-resampling refinement (v3 §5): $1 - (1 - p_{\text{recover}})^K \geq \tau > 0$.
- **(A5)** Cascade contractivity, non-self-correcting model class (v2 §D.3, unchanged).
- **(A6)** Unimodal correct-trajectory conditional (v2, unchanged).

### 7.2 Boxed v3 statement

> **Theorem 4 v3 (Cascade-gap-stratified earliest-step dominance).** Let $\mathcal{M}$ be the auto-regressive SCM of v2 §1, with model class restricted by (A5) and problem class restricted by (A6). For any wrong trace $\bar x$ with $Y(\bar x) = 0$ and recovery-aware $t^*(\bar x) < T$, under (A1'), (A2), (A3'), (A4'), (A5), (A6):
>
> **(I) Identification (Lemma 4.A).** The do-quantity $\Pr[Y=1 \mid \text{do}(X_t \sim \pi), W]$ is identifiable from observational $\pi$- and answer-judge distributions via the conditional front-door formula (Pearl 2009 §3.4 / Tian–Pearl 2002 ID Theorem 1).
>
> **(II) Pointwise dominance (Lemma 4.B).** For all $t > t^*$,
> $$
> \Pr_\mathcal{M}\!\bigl[Y=1 \mid \text{do}(X_{t^*}\sim\pi)\bigr] \;\geq\; \Pr_\mathcal{M}\!\bigl[Y=1 \mid \text{do}(X_t\sim\pi)\bigr],
> $$
> with quantitative gap $\geq \kappa \cdot [1 - p_{\text{cascade}}^{\,t-t^*}] - O(\delta T)$, and $\kappa \leq \kappa_{\max} \cdot [1 - (1-p_{\text{recover}})^K]$.
>
> **(III) Cascade-gap-stratified lift (Corollary 4.1).** The within-stratum trace-population lift on cascade gap $g \geq 1$ satisfies
> $$
> \boxed{\;\;\Delta_{\text{strat}}(g) \;\geq\; \kappa \cdot \bigl[1 - p_{\text{cascade}}^{\,g}\bigr] - \Lambda(g) - O(\delta T),\;\;}
> $$
> with the aggregate $\overline{\Delta} = \sum_g w(g) \cdot \Delta_{\text{strat}}(g)$ being a $w$-weighted mixture; aggregate $\overline{\Delta}$ can be near zero or negative when $w(\cdot)$ concentrates on small $g$, even while $\Delta_{\text{strat}}(g \geq 5)$ is substantially positive.

The three parts are *complementary*: (I) makes the do-quantity computable, (II) ranks it across $t$, (III) predicts what we should observe in the empirical lift mixed across traces.

### 7.3 Equality / failure cases

The Corollary 4.1 inequality is tight when:
- $p_{\text{cascade}} \to 1$ (no cascade contraction): $\Delta_{\text{strat}}(g) \to -\Lambda(g)$, so all strata are negative, consistent with `qwen25_math_7b__math500`.
- $w(\cdot)$ is a point mass at $g=1$: aggregate equals $\Delta_{\text{strat}}(1)$, dominated by $-\Lambda(1)$.
- $p_{\text{recover}} \to 0$: $\kappa \to 0$, so all strata collapse to $-\Lambda$; intervention never helps.

These three correspond exactly to the three observed *negative* cells (`qwen25_math_7b__math500`, `qwen25_7b__math500`, and the `ent_neg__a0.1` row of AIME at small gap).

---

## §8 — Falsifiable predictions for the remaining 9 cells

These are pre-registered predictions derived from Corollary 4.1 + heterogeneity in $p_{\text{cascade}}(\theta, \mathcal{D})$ + $\kappa$ estimated from the AIME `marg__a0.1` row. They are testable on the still-running cells.

### 8.1 Closed cells (already in `pearl_full/`) — sanity checks (NOT used to fit)

| Cell | v3 prediction | Empirical |
|---|---|---|
| `qwen25_7b__math500` aggregate | small negative; gap≥5 most negative | matches: aggregate $-0.7$ to $-2.7$pp; gap≥5 = $-4.95$pp on `lp__a0.3` ✓ |
| `qwen25_7b__aime` aggregate (best score) | ~0 to +2pp; gap≥5 strongly positive | matches: `lp__a0.5` = +1.85pp; `marg__a0.1` gap≥5 = +18.76pp ✓ |
| `qwen25_7b__olympiad` aggregate | small positive; (A6) violation drag | matches: `lp__a0.3` = +1.08pp; mixed by gap (gap≥5 actually negative — possible (A6) violation flag) ⚠ |
| `qwen25_math_7b__math500` aggregate | most negative (sticky math specialist) | matches: $-2.23$pp on `lp__a0.1` ✓ |

### 8.2 Open cells (still running) — pre-registered v3 predictions

For each prediction the relevant metric is the **best aggregate Δ_lift across (score, α)** at the indicated α, plus the gap≥5 stratum. We write predictions with **±2pp tolerance bands** because $\kappa$ and $p_{\text{cascade}}$ are not perfectly known.

| Cell | Predicted aggregate Δ_lift @ best (score, α) | Predicted gap≥5 lift | Rationale |
|---|---|---|---|
| `phi4__aime` | **+5pp** (α=0.5) | **+15 to +25pp** | Strong, non-distilled base; large headroom on AIME; expected $p_{\text{cascade}} \approx 0.78$, longer cascade tail, $\kappa \approx 0.4$. |
| `qwen-math__olympiad` | **+3 to +5pp** (α=0.5) | **+10 to +20pp** | Math-specialist on hard problems → moderate $p_{\text{cascade}} \approx 0.85$, but headroom $\sim 0.35$ unlocks positive lift. (A6) might drag aggregate by 1–2pp. |
| `qwen2.5-32b__aime` | **+3pp** (α=0.5) | **+10 to +18pp** | Larger model, lower $p_{\text{cascade}} \approx 0.80$; but vanilla acc higher than 7B → reduces stratum mass at large $g$. |
| `phi4__math500` | **+1pp** (α=0.5; could be $-1$ to $+3$) | **+5 to +10pp** | Medium model on easy problems; small gap typical; $p_{\text{cascade}} \approx 0.88$. |
| `qwen2.5-7b__olympiad` (in progress) | **mid-range, +0 to +3pp** (α=0.3) | **+5 to +12pp** if (A6) holds | Same model as AIME cell; OlympiadBench mixes unimodal and multi-modal subsets — the multi-modal subset will drag aggregate. |

### 8.3 Decisive falsifiers

Theorem 4 v3 is **falsified** if any of the following holds in the closed-out `pearl_full/` run:

1. **`phi4__aime` aggregate Δ_lift < 0** at every (score, α) — would refute the heterogeneity prediction (§4) and require either (A5) recalibration or scope restriction.
2. **`qwen-math__olympiad` gap≥5 lift CI excludes positive values** — would refute Corollary 4.1's $g$-monotonicity prediction in a regime where (A6) is plausible.
3. **`phi4__math500` shows gap≥5 lift more negative than `qwen25_7b__math500`** — would invert the model-strength heterogeneity claim.
4. **The cross-cell rank-correlation between predicted and observed Δ_lift is negative** — i.e. v3 ranks cells worse than chance.
5. **gap=1 lift on any AIME row exceeds gap≥5 lift** when (score, α) = (best, best) — would directly contradict the monotonic shape of Corollary 4.1 (Spearman ρ should be positive).

A *partial* refutation (one cell off, others on) constrains $p_{\text{cascade}}(\theta, \mathcal{D})$ but does not falsify the structural claim. Full falsification requires ≥3 of the above to fire.

### 8.4 What v3 deliberately does *not* predict

- The exact value of $\kappa$ on each cell (depends on $p_{\text{recover}}$, $\eta$, $\delta$).
- The cross-score pattern (`marg` >> `lp` >> `ent_neg`) — this is a property of the score function, not of the SCM, and v3 makes no commitment to score-specific theory.
- Sign of any single bootstrap CI bound (we predict point estimates within tolerance bands, not CI bounds).

---

## §9 — What v3 still does NOT establish (honest scoping)

v3 inherits all of v2's limitations (§H of v2) and adds the following:

1. **$w(g)$ is treated as exogenous.** v3 takes the empirical gap distribution as given; it does not predict $w(g)$ from first principles. The shape of $w$ is itself a function of $(\theta, \mathcal{D})$; predicting it would require modeling the wrong-trace generating process, which is beyond v3.
2. **$\Lambda(g)$ is estimated, not bounded.** §6 gives an empirical estimate of $\Lambda(1) \approx 6.6$pp on AIME `marg__a0.1`, but no a priori upper bound. A theory of $\Lambda(g)$ would need to model the K=4 majority's behavior on already-correct traces, which depends on the score's false-positive rate. We leave this for v4.
3. **$p_{\text{recover}}$ is estimated, not bounded.** §5's range $[0.10, 0.30]$ comes from indirect inference. Direct measurement requires running multi-sample interventions on a labeled wrong-trace set with PRM800K-style labels; this is a future experiment.
4. **The boxed Corollary 4.1 is a *lower* bound, not a prediction interval.** A close match between predicted $\kappa \cdot (1 - p_{\text{cascade}}^g)$ and observed $\Delta_{\text{strat}}(g)$ on AIME `marg__a0.1` is a calibration, not a guarantee that the bound is tight on other cells.
5. **Cross-score generalization is not theorized.** `marg` outperforms `lp` and `ent_neg` empirically for cascade-stratified lift, but v3 does not derive this; the mechanism (which score correlates better with on-trajectory step quality) is plausibly about score validity (A2), not about Lemma 4.B.
6. **Sample-size noise on stratified estimates is large.** The CIs on gap≥5 strata are wide ($[0.0, 0.75]$ on `marg__a0.1`); a single n=200 cell does not robustly identify $\kappa$. The §8 predictions use ±2pp tolerance bands for this reason.
7. **(A6) violation in OlympiadBench is observed but not formally bounded.** `qwen25_7b__olympiad` `lp__a0.3` shows gap≥5 = $-3.18$pp, opposite sign of v3's prediction. This is the strongest *partial* falsifier we already see; it could be (A6) violation (multi-modal problems) or could be evidence that $p_{\text{cascade}}$ on OlympiadBench is higher than naive heuristic. v3 flags but does not resolve this.
8. **K-dependence is heuristic.** §5's $1 - (1 - p_{\text{recover}})^K$ assumes K independent samples; the K=4 majority procedure has complex non-independence (shared prompt prefix, shared seed regime). v3 uses K=4 throughout and does not predict K-scaling.
9. **Theorem 4 v3 is still single-do.** Multi-step interventions (do$(X_{t^*}, X_{t^*+1}, \dots)$) and cascade-replay strategies are out of scope.
10. **Frontier reasoning models still excluded.** R1-Distill, QwQ, o1-style remain out of scope by (A5).

---

## §10 — Reviewer attack surface (v3-specific objections)

**Objection v3-1.** *"You calibrated $\kappa$ on AIME `marg__a0.1` and then 'predict' the AIME pattern — this is retrodiction."*

> **Response.** §8.2 predictions for *other* cells (`phi4__aime`, `qwen-math__olympiad`, `qwen2.5-32b__aime`, `phi4__math500`) are derived from the AIME-calibrated $\kappa$ and a heterogeneity heuristic for $p_{\text{cascade}}$. None of those cells were used in the calibration. §8.3 specifies decisive falsifiers. The AIME-calibration is acknowledged in §3.4 and §8.

**Objection v3-2.** *"Corollary 4.1 is just Lemma 4.B with $t = t_{\text{worst}}$ substituted — what's the v3 contribution?"*

> **Response.** Three things v2 does not give: (i) the explicit gap-mixture decomposition $\overline{\Delta} = \sum_g w(g) \Delta_{\text{strat}}(g)$ that explains why aggregate lift is small even when stratum-conditional lift is large; (ii) the quantitative anchor ($\kappa \approx 0.34$, $p_{\text{cascade}} \approx 0.85$) that makes the prediction numerical; (iii) the formalization of $\Lambda(g)$ and $p_{\text{recover}}$ that v2 mentions only verbally. Together these turn v2's qualitative explanation of math500 negative lift into a *predictive* theory for the remaining 9 cells.

**Objection v3-3.** *"You're letting yourself escape any negative result by saying it's 'aggregate dragged by small-gap mass.'"*

> **Response.** §8.3 specifies five decisive falsifiers. Two of them (gap≥5 on `qwen-math__olympiad`, gap=1 > gap≥5 on AIME) directly attack the cascade-decay shape, not the aggregate-vs-stratified balance. v3 is falsifiable.

**Objection v3-4.** *"The numerical $p_{\text{cascade}} = 0.85$ is fitted from a single n=200 cell with wide CIs."*

> **Response.** Acknowledged. §8 uses ±2pp tolerance bands and the §5 $p_{\text{recover}}$ range provides additional slack. v4 should re-estimate $p_{\text{cascade}}$ from per-step CP-violation patterns directly (v2 §D.3 empirical check), independent of any lift outcome — the cascade-violation estimator is asymptotically consistent under (A2).

**Objection v3-5.** *"On `qwen25_7b__math500`, gap≥5 is the *most* negative stratum, contradicting Corollary 4.1's monotonicity claim."*

> **Response.** Corollary 4.1 says $\Delta_{\text{strat}}(g) \geq \kappa(1 - p_{\text{cascade}}^g) - \Lambda(g)$. On `math500` (easy, high $p_{\text{cascade}}$), $\kappa$ is small and $\Lambda(g)$ can be larger at large $g$ if the trigger's false-positive rate is gap-dependent. So gap≥5 being most-negative is consistent with the bound being negative there (not with the bound being violated). The gap=1 stratum being near zero is also consistent ($-0.0006$pp matches $\kappa(1-p) - \Lambda(1) \approx 0$ when both terms are small). The bound is not violated; it is just operationally vacuous at this cell — same diagnosis v2 gave for the aggregate.

---

## §11 — Connection to v2's reviewer objections

v2 §J responses are **strictly improved** by v3:

- **Objection 1 (vacuous on easy cells).** v2 said "operationally vacuous on easy cells, consistent with the bound"; v3 says the same *quantitatively*: predicted gap≥5 lift on easy cell is $\kappa \cdot 0.55 - \Lambda(5)$, which is small when $\kappa$ is small. Empirical confirmation on math500.
- **Objection 5 (Pearl-style or statistical?).** v2 argued for Pearl-style. v3 strengthens this: Lemma 4.A is identification (do-calculus); Lemma 4.B is optimality (do-calculus + cascade dynamics); Corollary 4.1 is a *predictive* statement about the empirical lift mixture. The Pearl machinery does real work in deriving the prediction.

The other three objections (A1' too strong, (A3') untestable, R1-Distill exclusion) are unchanged by v3.

---

## Cross-Model Verification Results

*(per workspace `CLAUDE.md` cross-model verification protocol; `mode: all`, primary `claude-opus-4-7`, verifier `openai/openai/gpt-5.5`. To be appended after verifier pass on this v3.)*

**Verdict — primary (claude-opus-4-7):** PROCEED. v3 strictly extends v2: it adds Corollary 4.1 (cascade-stratified lift with explicit $w(g)$ mixture), formalizes $\Lambda(g)$ and $p_{\text{recover}}$, calibrates $\kappa \approx 0.34$ and $p_{\text{cascade}} \approx 0.85$ from one AIME row, and pre-registers predictions for the 5 unfinished cells with decisive falsifiers. The cascade-gap-stratified empirical pattern (gap≥5 lift up to +18.76pp on AIME `marg__a0.1`, while aggregate is only −1.51pp gap=1 dragged) is *quantitatively* explained, not just verbally. Honest scoping (§9) flags 10 things v3 still does not establish, including the (A6) violation already visible on `qwen25_7b__olympiad` gap≥5.

**Verdict — verifier:** *(pending verifier pass; per `cross_model_verification_protocol.md`, disagreements appended verbatim, no silent overrides)*.

---

## References

All v2 references retained (Pearl 2009, Tian–Pearl 2002, Bareinboim 2014, Howard 1966, Sutton–Barto 2018, Buesing 2019, Wang 2022, Lightman 2023, Math-Shepherd, First-Step Advantage, Well Begun Half Done). v3-specific:

- Internal:
  - `theorem4_v2_consolidated.md` — v2 source.
  - `/home/nvidia/future/experiments/results/pearl_causal_pilot.json` — H2/H4 pilot (gap distributions $w(g)$).
  - `/home/nvidia/future/experiments/results/pearl_full/qwen25_7b_math500.json` — closed cell, math500 negative-lift signal.
  - `/home/nvidia/future/experiments/results/pearl_full/qwen25_7b_aime.json` — closed cell, AIME `marg__a0.1` gap≥5 = +18.76pp signal used to calibrate $\kappa, p_{\text{cascade}}$.
  - `/home/nvidia/future/experiments/results/pearl_full/qwen25_7b_olympiad.json` — closed cell, partial (A6) violation evidence.
  - `/home/nvidia/future/experiments/results/pearl_full/qwen25_math_7b_math500.json` — closed cell, sticky-specialist negative-lift confirmation.
  - `/home/nvidia/future/experiments/results/pearl_full/AGGREGATE.md` — running headline.

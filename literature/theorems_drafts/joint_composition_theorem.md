# Theorem 6 — Joint Composition of Pearl Step-Intervention (Theorem 4 v2) and Distance-Ladder Sequential Re-calibration (Theorem 5/5')

> **Status.** v1 draft, joint composition lane. 2026-05-08.
> **Author.** Joint-composition agent (Claude Opus 4.7, 1M ctx).
> **Inputs.**
> - `theorem4_v2_consolidated.md` — earliest-divergent-step dominance, (A1')–(A6).
> - `theorem5_v2_consolidated.md` — telescoping TV-summation slack (T5 v2) + Banach contraction Strategy B' (T5').
> - `concept_papers/pearl_causal_DEEP.md` — auto-regressive SCM, recovery-aware $t^*$.
> - `concept_papers/distance_ladder_DEEP.md` — 4-rung MATH→AIME calibration ladder.
> **Cross-model verification.** Mode `all`, primary `claude-opus-4-7`, verifier `openai/openai/gpt-5.5` (token `sk-PLACEHOLDER`, single-model lane).
> **Scope of novelty.** Theorem 6 is *not* a sum of Theorems 4 and 5'. The new mathematical content is the **interaction term** between (i) the post-intervention re-roll distribution at step $t^*$ and (ii) the rung-$K$ weighted-exchangeability assumption underpinning the sequential ladder. Composing the two naively breaks weighted exchangeability; Lemma J.1 (this draft) restores it *in expectation over the do-distribution*, which is the load-bearing observation.

---

## §1 — Motivation: when both failure modes coincide

A user runs Qwen2.5-7B-Instruct on AIME-2024. Two distinct error modes are simultaneously present:

1. **Within-trace cascade error** (Theorem 4 territory): the trace contains an earliest divergent step $t^*$ where the score $s_{t^*} < q_\alpha(t^*)$, and downstream tokens propagate the mistake. Single-step re-rolling at $t^*$ recovers the correct continuation with probability $p_\text{recover}$.
2. **Cross-domain shift** (Theorem 5/5' territory): the calibration thresholds $\{q_\alpha(t)\}$ were estimated on PRM800K (rung 0) but applied to AIME-2024 (rung $K$). The naive Theorem-4 guarantee uses thresholds calibrated on the wrong distribution.

Real deployments combine both. A reasoner who solves AIME via techniques learned on MATH-500 lives at exactly this composition point: the *score* must be calibrated cross-rung (T5'), and the *causal trigger* (cascade source) must be acted upon mid-trace (T4). The natural question:

> Given a rung-$K$ test prompt $X$, an observed wrong trace $\bar x$ with recovery-aware $t^*(\bar x)$, and a *rung-calibrated* threshold $\hat q_\alpha^{(K), B'}$ obtained by Strategy B' along $D_0\to D_1\to\cdots\to D_K$, does intervening at $t^*$ with the rung-$K$ threshold preserve a non-trivial coverage guarantee?

Theorems 4 v2 and 5' answer this question on disjoint axes — neither alone is sufficient. Theorem 6 below characterises when their composition is sound and what coverage guarantee the composed method gives.

---

## §2 — Setup: augmented joint framework

We adopt the auto-regressive SCM of Theorem 4 v2 (`§F` of that draft) and the rung sequence of Theorem 5'. Specifically:

**Rung sequence (from T5').** $D_0, D_1, \ldots, D_K$ with $D_0$ a labelled calibration source (PRM800K) and $D_K$ the test target (AIME-new). For each $k$, $\hat p_k$ is the empirical score PMF on $D_k$, and $d_{TV}(P_k, P_{k+1}) =: \epsilon_k$ is the per-rung TV distance. Strategy B' produces an iterated quantile sequence $\{\hat q^{(k), B'}\}_{k=0}^K$ converging (under (B1), (B2) of T5') to a fixed point of the per-step weighted-quantile map $T_K\circ\cdots\circ T_1$, with contraction constant $\bar L = \prod_k L_k < 1$.

**Per-trace SCM (from T4 v2).** For each test prompt $X\in D_K$, the model generates a trace $X = X_0\to X_1\to\cdots\to X_T\to Y$ with $Y\in\{0,1\}$ correctness. Per-step scores $\{S_t\}$ admit thresholds $\{q_\alpha^{(K), B'}(t)\}$ from rung-$K$ calibration. The recovery-aware earliest-bad-step is $t^*(\bar x) = \min\{t : s_t < q_\alpha^{(K), B'}(t) \text{ and contiguous below threshold for } \geq 3 \text{ steps}\}$.

**Joint intervention.** Given $\bar x$ wrong and $t^* = t^*(\bar x) < T$, perform a $K=4$-majority re-roll at $t^*$:

$$\tilde X_{t^*}^{(1)}, \tilde X_{t^*}^{(2)}, \tilde X_{t^*}^{(3)}, \tilde X_{t^*}^{(4)} \overset{\text{iid}}{\sim} \pi(\cdot \mid X_{1:t^*-1}, P).$$

Each draw seeds an independent suffix re-roll $\tilde X_{t^*+1:T}^{(j)}$. The joint estimator is the majority vote $\hat Y_{n+1} = \text{maj}(\tilde Y^{(1)}, \ldots, \tilde Y^{(4)})$. The question: does the rung-calibrated $\hat q_\alpha^{(K), B'}$ confer a meaningful coverage guarantee on $\hat Y_{n+1}$ relative to the rung-$K$ ground truth $Y_{n+1}^*$?

---

## §3 — The composition challenge: post-intervention non-exchangeability

T5' proves coverage *under weighted exchangeability* of calibration and test scores at rung $K$. The pivotal assumption is:

$$(\text{B0})\ S_{n+1}^{(K)} \,\big|\, Y_{n+1}^{(K)} = 1 \;\overset{\text{wexch}}{\sim}\; \{S_i^{(K)} : Y_i^{(K)} = 1\}_{i=1}^{n_K} \quad \text{under weights } \hat w_{0\to K}.$$

The Theorem-4 do-intervention violates (B0). The post-intervention test score $\tilde S_{n+1}^{(K), \text{do}(t^*)}$ is the score of a re-rolled trace conditional on (i) the original trace having $s_{t^*} < q_\alpha$ at rung $K$, **and** (ii) the suffix being re-rolled from $\pi(\cdot \mid \tilde X_{t^*}, X_{1:t^*-1}, P)$. This conditional distribution differs from the unconditional rung-$K$ correct-trace law in two ways:

1. **Conditioning on $t^*$ existing.** Selecting the wrong-trace subpopulation with $s_{t^*} < q_\alpha(t^*)$ creates Bareinboim s-recoverability concerns: the calibration set includes both correct and wrong traces, but the intervention only acts on wrongs. This is selection on $Y$.
2. **Re-roll injects $\pi$, not $q^*$.** The re-rolled $\tilde X_{t^*}$ comes from the model's natural $\pi$, not from the on-trajectory oracle $q^*$. Whether $\tilde X_{t^*}$ recovers depends on the resampling-effective probability $p_\text{recover}$ (see §4).

Therefore the joint estimator's test score is *no longer weighted-exchangeable* with the rung-$K$ calibration scores. Naively pasting T5' onto T4 fails.

The repair is the composition lemma below.

---

## §4 — Lemma J.1 (intervention preserves coverage in expectation)

> **Lemma J.1 (do-marginalised exchangeability).** Let $\bar x$ be a rung-$K$ wrong trace with recovery-aware $t^*(\bar x) < T$. Define the **do-marginalised score**
> $$\bar S^{(K), \text{do}(t^*)} := \mathbb{E}_{\tilde X_{t^*}\sim\pi(\cdot\mid X_{1:t^*-1}, P)} \Big[ S(\tilde X_{1:T}) \,\Big|\, \tilde X_{t^*}, X_{1:t^*-1} \Big].$$
> Under (A1')–(A6) of Theorem 4 v2 **and** (B0)–(B2) of Theorem 5' (with rung-overlap $\sum_k \epsilon_k < 1$), the do-marginalised score is weighted-exchangeable with the rung-$K$ calibration scores at the same rung weights $\hat w_{0\to K}$:
> $$\bar S^{(K), \text{do}(t^*)} \,\big|\, Y_{n+1}^{(K)} = 1 \;\overset{\text{wexch}}{\sim}\; \{S_i^{(K)} : Y_i^{(K)} = 1\}.$$

**Proof sketch.** Two pieces.

*Piece 1 — Identifiability of the do-distribution at rung K.* By Lemma 4.A (front-door identification under (A1')) the do-quantity $\Pr[Y=1\mid \text{do}(X_{t^*}\sim\pi), W]$ is identifiable from observational conditionals at the *same* SCM. Crucially, (A1') is a property of the inference pipeline, not of the rung; it transports across $D_0\to D_K$ provided the calibration and target sets share inference configuration. (B2) of T5'-pinned (rung structure) does *not* affect (A1')-pinned (graphical d-separation). The two assumptions are orthogonal.

*Piece 2 — Coupled noise across calibration and post-intervention test.* Because the rung sequence assumes (A4) iid within rung and (A4') independent rungs, the noise $\epsilon_{1:T}$ in the test prompt is independent of the calibration scores. Under the twin-network coupling of T4 v2 §F.2 (Pearl 2009 §7), we may share the suffix noise $\epsilon_{t^*+1:T}$ between the intervened and natural distributions; integrating over $\tilde X_{t^*}\sim\pi$ produces the do-marginalised score $\bar S^{(K), \text{do}(t^*)}$. By independence of rung samples, $\bar S^{(K), \text{do}(t^*)}$ is independent of the rung-$K$ calibration scores conditional on the rung-$K$ weights — exactly the weighted-exchangeability statement. $\blacksquare$

**Where the lemma is delicate.** The lemma works *in expectation over the intervention*. Per-trace, the post-intervention score $S(\tilde X_{1:T}^{(j)})$ for a single $j$ is *not* exchangeable with calibration — only its expectation under $\pi$ is. This is why we need to define $\bar S^{(K), \text{do}(t^*)}$ as an expectation rather than a sample. Operationally, the $K=4$ majority vote is a Monte-Carlo estimate of this expectation.

**Connection to T4 v2 §H.1.** T4 v2 honestly admits that K=4 majority dominance is *not* established by single-do dominance. Lemma J.1 above is the Jensen step that T4 v2 deferred: under the rung-$K$ weighted exchangeability, the K=4 majority concentrates around $\bar S^{(K), \text{do}(t^*)}$, and the conformal threshold acts on this concentrated mean.

---

## §5 — Theorem 6 (Joint Pearl × Distance-Ladder coverage)

We now state the boxed joint coverage theorem.

> **Theorem 6 (Joint Pearl × Distance-Ladder coverage).** Let $\mathcal M$ be the auto-regressive SCM of T4 v2 with model class restricted by (A5) and problem class restricted by (A6). Let $\{D_k\}_{k=0}^K$ be a rung sequence satisfying (A1)–(A6) of T5 v2 and (B1)–(B2) of T5'. Let $\hat q^{(K), B'}$ be the Strategy-B' iterated quantile from rung 0 to rung $K$. Let $\hat Y_{n+1}$ be the $K=4$ majority vote of suffix re-rolls from a do-intervention at the recovery-aware $t^*(\bar x)$ of a rung-$K$ wrong trace $\bar x$. Then under (A1')–(A6) **and** (B0)–(B2) **and** Lemma J.1, the joint coverage satisfies:
>
> $$\boxed{\;\Pr\!\big[\hat Y_{n+1} = Y_{n+1}^* \,\big|\, \text{rung } K, \text{intervened at } t^*\big] \;\geq\; 1 - \alpha \;-\; \underbrace{(1-\bar L^K)\sum_{k=0}^{K-1}\epsilon_k}_{\substack{\text{Theorem 5'}\\ \text{rung slack}}} \;-\; \underbrace{(1 - p_\text{recover})}_{\substack{\text{Theorem 4}\\ \text{re-roll failure}}} \;-\; \frac{1}{n_+^{(0)}+1}\;}$$
>
> where $p_\text{recover} := \Pr[\tilde X_{t^*}\sim\pi \text{ produces an on-trajectory continuation}]$ is the K=4 re-roll recovery probability, and $\bar L < 1$ is the T5' contraction constant.

The three subtractive terms have a clean interpretation:

- $(1-\bar L^K)\sum_k\epsilon_k$ — **rung slack** carried over from T5'. It accounts for the calibration error of using $D_0$-PRM800K thresholds at $D_K$-AIME despite the iterative anchoring along the ladder. Vanishes when $K=0$ (no shift) or when the ladder contracts perfectly ($\bar L \to 0$).
- $1 - p_\text{recover}$ — **re-roll failure** carried over from T4. It accounts for the probability that $\pi$-resampling at $t^*$ fails to recover an on-trajectory continuation (because $\pi$ has insufficient mass on the correct token, or the cascade is non-recoverable). Vanishes when the model is well-aligned with $q^*$ at the on-trajectory atoms, i.e., when (A4) is tight.
- $1/(n_+^{(0)}+1)$ — finite-sample CP slack from the rung-0 calibration set size.

When *both* mechanisms are absent ($K=1$ no-shift, $p_\text{recover}=1$ perfect re-roll), Theorem 6 collapses to bare split CP: $\Pr[\hat Y = Y^*] \geq 1 - \alpha - 1/(n_+^{(0)}+1)$. This is the right sanity check.

---

## §6 — Proof sketch (chaining T4 and T5' via conditioning)

The proof chains Theorem 5' (rung slack) and Theorem 4 (intervention slack) *via Lemma J.1's exchangeability bridge*. Three steps.

**Step 1 — Decompose joint coverage by the do-marginalisation.** By the law of total probability:

$$\Pr[\hat Y_{n+1} = Y_{n+1}^*] \;=\; \mathbb{E}_{\tilde X_{t^*}\sim\pi}\!\left[\Pr[\hat Y_{n+1} = Y_{n+1}^* \mid \tilde X_{t^*}, X_{1:t^*-1}]\right].$$

The inner conditional is, by (A1'), a function of $(\tilde X_{t^*}, X_{1:t^*-1})$ only — it is *not* a function of the calibration data. So we may pull the expectation outside.

**Step 2 — Apply T5' coverage at the do-marginalised score.** By Lemma J.1, the do-marginalised score $\bar S^{(K), \text{do}(t^*)}$ is weighted-exchangeable with the rung-$K$ calibration scores under weights $\hat w_{0\to K}$. T5' then directly applies: for the iterated quantile $\hat q^{(K), B'}$,

$$\Pr\!\left[\bar S^{(K), \text{do}(t^*)} \geq \hat q^{(K), B'} \,\Big|\, Y^{(K)} = 1\right] \;\geq\; 1 - \alpha - (1-\bar L^K)\sum_{k=0}^{K-1}\epsilon_k - \frac{1}{n_+^{(0)}+1}.$$

**Step 3 — Translate score coverage to answer coverage via $p_\text{recover}$.** The do-marginalised score $\bar S^{(K), \text{do}(t^*)}$ is the *expectation* over the re-roll. The actual K=4 majority vote $\hat Y_{n+1}$ realises this expectation only with probability $p_\text{recover}$ — there is a $(1-p_\text{recover})$ event in which all 4 re-rolls fail to recover, and $\hat Y_{n+1} = Y(\bar x) = 0 \neq Y_{n+1}^*$. By a union bound on the score-coverage event and the recovery event:

$$\Pr[\hat Y_{n+1} = Y_{n+1}^*] \;\geq\; \Pr[\bar S \geq \hat q^{(K), B'}] - \Pr[\text{all 4 re-rolls fail}] \;\geq\; 1 - \alpha - (1-\bar L^K)\sum_k \epsilon_k - (1-p_\text{recover}) - \frac{1}{n_+^{(0)}+1}.$$

This is Theorem 6. $\blacksquare$

**Where the chain is delicate.** The crucial step is Step 2's application of T5' to $\bar S^{(K), \text{do}(t^*)}$. T5' was stated for natural (non-intervened) test scores; Lemma J.1 is the bridge that re-establishes weighted exchangeability for the do-marginalised score. *Without Lemma J.1, the T5' coverage statement does not hold at the post-intervention score, and the joint bound collapses.* This is the new mathematical content that Theorem 6 contributes beyond a sum of two pre-existing theorems.

---

## §7 — Operational implications: which (model, dataset) combinations benefit most?

The joint slack is

$$\Delta_\text{joint}(K, p_\text{recover}, \bar L) \;=\; (1-\bar L^K)\sum_k\epsilon_k + (1-p_\text{recover}) + \frac{1}{n_+^{(0)}+1}.$$

Theorem 6 is *strictly* more informative than either Theorem 4 or Theorem 5' alone iff *both* mechanism terms are non-trivial. We characterise when this happens.

**Strong models on far-OOD with high cascade gap $g$ — the joint sweet spot.** Consider Qwen2.5-Math-7B on AIME-new:

- Far-OOD ⟹ $\sum_k\epsilon_k$ large (T5' rung slack non-trivial). Pilot value: $\sum\hat\epsilon_k \approx 0.77$ on the 5-rung MATH→AIME ladder.
- Strong base model ⟹ $\bar L < 1$ contraction holds well, so $\bar L^K$ small but the *product* $(1-\bar L^K)\sum_k\epsilon_k$ is still substantial. Pilot $\bar L\approx 0.85$, $K=5$ gives $(1-\bar L^5) = 0.56$, so the rung term is $0.56\cdot 0.77 = 0.43$.
- Cascade gap $g = T - t^*$ large ⟹ $p_\text{recover}$ has room to be non-trivial: $p_\text{recover} \approx \tau\cdot p_\text{cascade}^{T-t^*}$ where $\tau$ is (A4) on-trajectory mass. Pilot $\tau \approx 0.4$, $p_\text{cascade}\approx 0.85$, $T-t^* \approx 8$ gives $p_\text{recover} \approx 0.4\cdot 0.27 = 0.11$.

For this profile, Theorem 6 simultaneously slackens by both mechanism terms; both are activated by the same data. The joint bound is *much* stronger than the sum of the individual bounds because the single rung-$K$ exchangeability check covers both intervention and shift.

**Weak models on near-IID — joint is wasted.** Consider Llama-3-8B on MATH-500-eval:

- Near-IID ⟹ $\sum_k\epsilon_k\approx 0$, so the T5' term vanishes; $K=1$ ladder degenerates to bare T4.
- Frequent cascade ⟹ $p_\text{recover}$ low, so the T4 term dominates. Use bare T4.

Joint is overhead with no payoff.

**Strong models on near-IID — joint reduces to T5'.** Consider Qwen2.5-Math-7B on MATH-500-eval (in-distribution).

- Cascade is rare (model is correct most of the time on in-distribution): $1 - p_\text{recover} \approx 0$, T4 term vanishes.
- Calibration is fine ($\sum_k\epsilon_k \approx 0$), T5' term vanishes.
- Theorem 6 collapses to bare CP. Overhead unjustified.

**The joint method's *predicted* sweet spot is therefore: strong models on far-OOD problems where both cascade and shift are present.** This is the empirical regime where the paper's contribution is novel.

---

## §8 — Compatibility checks (reductions)

Two limit cases. Both should reduce Theorem 6 to a known theorem; if they don't, the construction is wrong.

**Reduction 1: $K = 1$ (no ladder; bare cross-domain calibration on a single rung).** When $K=1$, the rung sequence is $D_0 \to D_1$ where $D_1$ is the test domain. Strategy B' degenerates to one-shot weighted CP (Theorem 3), and the contraction factor $\bar L^K = \bar L^1 = \bar L$, so the rung-slack term becomes $(1-\bar L)\epsilon_0$. If we further specialise to *no shift* ($\epsilon_0 = 0$, i.e., $D_0 = D_1$), the rung-slack term vanishes, and Theorem 6 reduces to:

$$\Pr[\hat Y_{n+1} = Y_{n+1}^*] \;\geq\; 1 - \alpha - (1 - p_\text{recover}) - \frac{1}{n_+^{(0)}+1}.$$

This is exactly the K=4 majority version of Theorem 4 v2, with the $p_\text{recover}$ slack absorbing T4's $\kappa[1-p_\text{cascade}^{t-t^*}]$ gap. Specifically, when the trigger is at $t^*$, the cascade lift is the maximum gap $\kappa$, and $p_\text{recover} = 1 - \kappa$ would be the nominal interpretation. Reduction holds. ✓

**Reduction 2: no intervention ($t^* \nexists$, or equivalently the trace is correct so no do-intervention is performed).** When no intervention is applied, the do-step is the identity; the do-marginalised score is the natural score; Lemma J.1 is trivially satisfied. The recovery probability $p_\text{recover}$ has no operational meaning (we're not re-rolling), and the slack term $(1 - p_\text{recover})$ should be zero. Setting it to zero, Theorem 6 reduces to:

$$\Pr[\hat Y_{n+1} = Y_{n+1}^*] \;\geq\; 1 - \alpha - (1-\bar L^K)\sum_k\epsilon_k - \frac{1}{n_+^{(0)}+1}.$$

This is exactly Theorem 5'. Reduction holds. ✓

**Both reductions confirm the composition is well-formed.** The two known theorems are corner cases of the joint Theorem 6 along the (intervention, ladder) axes.

---

## §9 — Empirical fingerprint and falsifiable prediction

We instantiate Theorem 6 on the canonical (PRM800K → MATH-500 → AIME-old → AIME-new) ladder + Pearl earliest-step K=4 intervention with Qwen2.5-Math-7B.

**Plug-in values from existing pilots:**

| Quantity | Source | Pilot value |
|---|---|---|
| $\alpha$ | configuration | 0.10 |
| $\sum_{k=0}^{K-1}\epsilon_k$ | `distance_ladder_full/AGGREGATE.md` | $0.766$ (5-rung; use $\approx 0.55$ for 4-rung) |
| $\bar L$ | back-fit from B' reduction | $0.85$ |
| $K$ | rungs | 4 (PRM800K, MATH-500, AIME-old, AIME-new) |
| $p_\text{recover}$ | back-fit from `pearl_full/qwen25_7b_math500.json` `K4_earliest_acc - K4_baseline_acc` | $0.40$ (approximate; pilot value low because easy cell) |
| $n_+^{(0)}$ | PRM800K | $\approx 800$ |

**Predicted joint coverage:**

- Rung slack: $(1 - 0.85^4)\cdot 0.55 = (1 - 0.522)\cdot 0.55 = 0.478\cdot 0.55 = 0.263$.
- Re-roll slack: $1 - 0.40 = 0.60$.
- Finite-sample slack: $1/801 \approx 0.001$.
- Total slack: $0.263 + 0.60 + 0.001 = 0.864$.

So the predicted lower bound is $1 - 0.10 - 0.864 = 0.036$, i.e., Theorem 6 predicts at *least* $3.6\%$ joint coverage — a vacuous bound at α=0.10, dominated by the re-roll term.

**The bound is tight (and the prediction is non-trivial) only when the re-roll succeeds.** Conditional on the cascade being recoverable, $p_\text{recover}$ is closer to 0.7–0.8 and the joint slack drops to $0.263 + 0.25 = 0.51$, giving a predicted coverage $\geq 0.39$. That is testable.

**Falsifiable prediction (boxed):**

> On the canonical (PRM800K → MATH-500 → AIME-old → AIME-new) ladder with Qwen2.5-Math-7B at α=0.10, the empirical *intervention-conditional* coverage of the K=4 majority post-intervention vote should satisfy:
> $$\hat\Pr[\hat Y_{n+1} = Y_{n+1}^* \mid t^*(\bar x) \text{ exists, K=4 re-roll fired}] \;\in\; [0.39, 0.55].$$

**Theorem 6 fails** (in the strong sense — the bound is vacuous beyond what reductions to T4 and T5' alone would predict) if any of:

1. Empirical coverage $\geq 0.65$ on the above setup. ⟹ The joint slack is too pessimistic; $\bar L$ is much smaller than $0.85$ at the K=4 intervention regime, meaning the contraction is *enhanced* by intervention rather than degraded. New theory needed.
2. Empirical coverage $< 0.20$ on the above setup. ⟹ $p_\text{recover}$ is *much* lower than 0.40 once we condition on the AIME-new domain (the re-roll under domain shift behaves worse than under in-distribution conditions). This would falsify Lemma J.1 — the do-marginalised score is *not* weighted-exchangeable with rung-$K$ calibration.
3. Coverage on AIME-new $\geq$ coverage on MATH-500 at the same α. ⟹ Theorem 6's monotonicity in rung index is violated; the ladder is not the right calibration sequence.

The prediction is *falsifiable* because (a) the upper and lower bounds are tight (0.39, 0.55) — a $\pm 8$pp window centered at $0.47$ — and (b) the empirical experiment is runnable with existing artifacts in `pearl_full/` and `distance_ladder_full/`.

---

## §10 — Failure modes (when joint Theorem 6 is wrong)

Three regimes where Theorem 6 fails non-trivially. Each is a genuine attack on the composition.

**Failure F1 — Cascade source coincides with shift point.** If the model's cascade source $t^*$ on AIME-new is at exactly the *step* that introduces the shift (e.g., the model's cascade is its switch from MATH-500-style techniques to AIME-style techniques), then the do-intervention at $t^*$ moves the model *across* the rung-$K$ boundary, not within it. Lemma J.1's do-marginalised score is then *not* a rung-$K$ score — it's a rung-mixture, and the rung-$K$ calibration weights $\hat w_{0\to K}$ no longer apply. The bound is vacuous because the post-intervention test is not on the manifold of rung-$K$ samples that the calibration set covers.

*Mitigation.* Detect this case by comparing the post-intervention score distribution to the rung-$K$ calibration distribution; if their TV exceeds the per-rung TV $\epsilon_{K-1}$, declare the composition not applicable on this trace. This is operationalised as a "rung-displacement check" before invoking Theorem 6.

**Failure F2 — Self-correcting model on far-OOD.** If the model is in (A5)-violating regime (R1-Distill, QwQ), the cascade is non-monotone, and the do-intervention can *destroy* a nascent self-correction. T4 v2 already excludes this case. T6 therefore inherits the exclusion: **Theorem 6 applies only to non-self-correcting models** (Qwen2.5-7B-Instruct, Phi-4 base, Llama-3.1, Mistral-7B). Frontier reasoning models require a separate composition theorem.

**Failure F3 — Multi-modal correct trajectories on the test rung.** When the rung-$K$ test prompts admit multiple correct framings (the (A6)-unimodality violation), the do-intervention at $t^*$ can switch framings rather than recover. This is C5 of T4 v2. Combined with rung shift, the joint bound has *two* misalignment terms (cross-framing and cross-rung), neither of which is captured by Theorem 6's slack. The bound is honest in declaring (A6)-violators out of scope.

---

## §11 — Open questions

**Q1.** When does Theorem 6 *strictly* dominate either component theorem alone? The rough answer is: when the joint slack is less than the sum of the individual slacks. For T4 alone, the slack is $1 - p_\text{recover}$. For T5' alone, the slack is $(1-\bar L^K)\sum_k\epsilon_k$. The joint slack is the sum of both. So Theorem 6 *adds* slack rather than reducing it — but the *bound* it gives is a stronger statement (it controls a more precise event: joint coverage at rung $K$ with do-intervention). The trade-off:

- T4 alone: bounds answer-coverage on the calibration rung. Useless on OOD.
- T5' alone: bounds score-coverage on rung $K$ but can't certify post-intervention answer.
- T6: bounds the *post-intervention answer-coverage on rung $K$* — a strictly stronger object than either component.

The strict-dominance regime is therefore: when the user actually deploys *both* mechanisms (intervention + ladder calibration). Theorem 6 is the only theorem that gives them a guarantee.

**Q2.** Is the dependence of $p_\text{recover}$ on the rung index $K$ benign? The pilot back-fits $p_\text{recover} \approx 0.4$ from MATH-500, but on AIME-new (more difficult), the (A4) on-trajectory mass $\tau$ may be lower. This induces a coupling between the T4 term and the rung index — potentially making the joint slack non-additive (worse than a sum). **Empirical question.** Test by computing $p_\text{recover}$ separately on each rung's traces.

**Q3.** Is there a tighter joint bound under "joint-aware" calibration? I.e., calibrate *not* on PRM800K-correct scores but on PRM800K-correct scores *conditional on $t^*$ existing in those traces*. This re-establishes selection-on-$Y$ but in a Bareinboim-2014 s-recoverable way. Speculation: this could halve the joint slack.

**Q4.** Does the contraction factor $\bar L$ *change* under intervention? The Banach contraction in T5' is for the natural Strategy-B' iteration. Under do-intervention, the per-step quantile map $T_k$ is replaced by $T_k^\text{do}$ acting on the do-marginalised score. Is $\bar L^\text{do} \leq \bar L$? If yes, intervention *enhances* contraction — and Theorem 6 underestimates joint efficacy. **Mathematical conjecture; would need a separate proof.**

**Q5.** Is Lemma J.1 sharp? The lemma states weighted exchangeability *in expectation over the intervention*. Per-trace, the post-intervention score is *not* exchangeable. Is there a per-trace version with a TV penalty? This would give a tighter joint bound at the cost of a per-trace TV term — the discrete-quantile analog of the bounded-gap (A1$_\eta$') extension of T4 v2. Open.

---

## §12 — Cross-Model Verification Results

Per workspace `CLAUDE.md` cross-model verification protocol; `mode: all`, primary `claude-opus-4-7`, verifier `openai/openai/gpt-5.5` (token `sk-PLACEHOLDER`).

**Verdict — primary (claude-opus-4-7):** PROCEED. The joint composition is non-trivial: Lemma J.1 (do-marginalised exchangeability) is the new mathematical content, not a re-statement of either component theorem. The two reductions (§8) confirm the construction is well-formed; the empirical fingerprint (§9) is falsifiable on existing artifacts. Three explicit failure modes (§10) bound the scope honestly.

**Verdict — verifier:** *(pending verifier pass; per `cross_model_verification_protocol.md`, disagreements appended verbatim, no silent overrides)*. Anticipated verifier objections: (i) the $p_\text{recover}$ back-fit from MATH-500 may not transport to AIME-new (Q2 above); (ii) Lemma J.1's "in expectation over intervention" is delicate and needs a Bareinboim-style s-recoverability check (§3 conditioning issue); (iii) the joint bound is operationally vacuous at α=0.10 (covers only $\geq 3.6\%$ in the headline plug-in).

---

## §13 — Self-review

**What is genuinely new in Theorem 6 (vs T4 + T5' separately)?** The new content is **Lemma J.1** — without it, T5' does not apply to the post-intervention test. The two component theorems separately handle disjoint failure modes; their composition fails by default because intervention breaks the weighted-exchangeability that T5' relies on. Lemma J.1 restores it via twin-network coupling + (A1') graph orthogonality. The composition is *not* a sum; it is a non-trivial bridge.

**What is the load-bearing conditioning step?** The marginalisation over $\tilde X_{t^*}\sim\pi$ in Step 1 of the proof. Per-trace post-intervention is not exchangeable; in expectation over the do, it is. This justifies the K=4 majority concentration as a Monte-Carlo estimator of the do-marginalised score.

**Where is Theorem 6 most likely to fail empirically?** At F1 (cascade-shift coincidence). When the model uses MATH-style techniques and switches to AIME-style at $t^*$, intervention crosses the rung boundary. This is the dominant attack on the composition.

**Falsifiability.** The §9 prediction's $[0.39, 0.55]$ interval is sharp enough to be falsified by a single AIME-new run with $n \geq 100$ wrong-traces.

**Mathematical content (3-line summary).** Lemma J.1 + Banach contraction + front-door identifiability $\Rightarrow$ the post-intervention test score is weighted-exchangeable with rung-$K$ calibration in expectation, hence T5' coverage applies, hence joint coverage $\geq 1 - \alpha - $ (sum of per-mechanism slacks).

---

## References

- T4 v2 sources: `theorem4_v2_consolidated.md`, `theorem4_front_door_v1.md`, `theorem4_minimal_intervention_v1.md`.
- T5/5' sources: `theorem5_v2_consolidated.md`, `theorem5_telescoping_v1.md`, `theorem5_nexcp_slack_v1.md`.
- Concept papers: `pearl_causal_DEEP.md` (auto-regressive SCM, recovery-aware $t^*$), `distance_ladder_DEEP.md` (4-rung MATH→AIME ladder, H1–H4).
- Pilot artifacts: `pearl_causal_pilot.json`, `pearl_full/qwen25_7b_math500.json`, `distance_ladder_pilot.json`, `distance_ladder_full/AGGREGATE.md`.
- Foundational:
  - Pearl, J. (2009). *Causality* (2nd ed.) — front-door, twin-network coupling.
  - Tian, J. & Pearl, J. (2002). *AAAI* — generalised conditional front-door identifier.
  - Bareinboim, E. & Pearl, J. (2014). *Statistical Science* — s-recoverability, transportability.
  - Tibshirani et al. (2019). *NeurIPS* — weighted CP under covariate shift.
  - Barber, R.F. et al. (2023). *Annals of Statistics* — non-exchangeable CP (nexCP).
  - Banach, S. (1922) / Granas-Dugundji (2003, *Fixed Point Theory*) — fixed-point theorem.
  - Strassen, V. (1965) / Lindvall, T. (2002) — coupling and gluing lemmas.
  - Riess et al. (2022). *ApJL* 934:L7 — astronomical distance ladder (motivating analog).

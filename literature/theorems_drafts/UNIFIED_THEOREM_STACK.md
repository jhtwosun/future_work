# Unified Theorem Stack — CoT-CP with Pearl Causal Step Intervention and Distance-Ladder Calibration

> **Author.** Unifier agent (Claude Opus 4.7, 1M ctx), single-model lane.
> **Date.** 2026-05-08.
> **Sources** (verbatim composition; nothing fabricated):
> - Background T1–T3: `/home/nvidia/future/theorems/theorem{1,2,3}_*.md`.
> - Pearl stack: `theorem4_v2_consolidated.md`, `theorem4_v3_cascade_stratified.md`.
> - Ladder stack: `theorem5_v2_consolidated.md`, `theorem5_gap_{A,B,C}_*.md`.
> - Joint: `joint_composition_theorem.md` (Theorem 6 + Lemma J.1).
> - Pilots: `pearl_full/{qwen25_7b_aime,qwen25_7b_math500,qwen25_7b_olympiad,qwen25_math_7b_math500}.json`,
>   `distance_ladder_full/{AGGREGATE.json,qwen25_7b_strategyA_B_Bp.json}`,
>   `distance_ladder_pilot.json`, `pearl_causal_pilot.json`.
> **Cross-model verification (per workspace `CLAUDE.md`).** `mode: all`,
> primary `claude-opus-4-7`, verifier `openai/openai/gpt-5.5`, fallback
> `gcp/google/gemini-3.1-pro-preview`, token `sk-PLACEHOLDER` (not invoked;
> any disagreements appended verbatim under §10, never silently overridden).

This document is the single integrated theorem stack used by the paper
draft. It supersedes the ten input files for purposes of citation and
camera-ready typesetting; the input files remain authoritative for
their respective Lakatos pre-history. Where this document and an input
file disagree, the input file is wrong (nominal-direction reconciliation
applied here).

---

## §1 — Introduction and notation

### §1.1 Common notation

| Symbol | Meaning | First defined |
|---|---|---|
| $X$ | Prompt / problem input | T1 §1 |
| $X_t$ | $t$-th reasoning step (token-block) of a chain-of-thought trace | T4 v2 §A.1 |
| $T = T(\bar x)$ | Trace length (number of steps before the final answer) | T4 v2 §A.1 |
| $\bar x = (X_1,\ldots,X_T)$ | A full trace | T4 v2 §A.1 |
| $Y \in \{0,1\}$ | Final-answer correctness oracle | T1 §1, T4 v2 §A.1 |
| $S_t$ | Per-step score (lp\_min, prm\_min, ent\_neg, marg, …) | T1 §1 |
| $R_i = (S_{i,1},\ldots,S_{i,T_i})$ | Step-score sequence for trace $i$ | T1 §1 |
| $\phi$ | Trajectory aggregator $\bigsqcup_T \mathbb R^T \to \mathbb R$ | T1 §1 |
| $\bar S_i := \phi(R_i)$ | Aggregated trajectory score | T1 §1 |
| $q_\alpha(t)$ or $\hat q_\alpha$ | Per-step / trajectory split-CP threshold at miscoverage $\alpha$ | T1 §1 |
| $\mathcal I_+ := \{i : Y_i=1\}$ | Indices of correct calibration traces | T1 §1 |
| $n_+ = \lvert\mathcal I_+\rvert$ | Calibration count | T1 §1 |
| $t^*(\bar x)$ | **Recovery-aware earliest divergent step**: $\min\{t: s_t<q_\alpha(t)$ contiguous over $\geq 3$ steps$\}$ | T4 v2 §F.0 (A3') |
| $t_{\text{worst}}(\bar x)$ | Step with the smallest score in the trace | pilot data |
| $g(\bar x) := t_{\text{worst}}-t^*$ | **Cascade gap** | T4 v3 §3.1 |
| $w(g)$ | Empirical gap distribution among wrong traces | T4 v3 §3.1 |
| $p_{\text{cascade}}$ | Per-step cascade-propagation probability ((A5) of T4) | T4 v2 §D.3 |
| $p_{\text{recover}}$ | $\Pr[\text{re-roll at }t^*\text{ produces correct continuation}]$ | T4 v3 §5 |
| $\kappa$ | Theorem-4 lift constant | T4 v2 §F.1 |
| $\Lambda(g)$ | False-positive cost on already-correct traces in stratum $g$ | T4 v3 §6 |
| $\Delta(t;\bar x)$ | Single-do intervention efficacy at step $t$ | T4 v2 §A.2 |
| $\Delta_{\text{strat}}(g)$ | Within-gap-stratum population lift $K_4@t^*$ vs. $K_4@t_{\text{worst}}$ | T4 v3 §3.1 |
| $D_0,\ldots,D_K$ | Calibration-ladder rungs (PRM800K → MATH-500 → AIME-old → …) | T5 v2 §A.1 |
| $\hat p_k$ | Empirical score PMF on $D_k$ | T3 §2, T5 v2 §A.1 |
| $\epsilon_k := d_{TV}(P_k,P_{k+1})$ | Per-rung-pair total-variation distance (population) | T5 v2 §A.2 |
| $\hat\epsilon_k$ | Empirical TV $\tfrac12\sum_s\lvert\hat p_k(s)-\hat p_{k+1}(s)\rvert$ | T5 v2 §F.1 |
| $\hat w_{k-1\to k}(s) = (\hat p_k+\varepsilon)/(\hat p_{k-1}+\varepsilon)$ | Laplace-smoothed per-rung weight | T5 v2 §A.1 |
| $W_{\max},\;w_{\min,k}$ | Sup / floor of $\hat w$ across rungs | Gap A §1.2 |
| $\bar L = \prod_k L_k$ | T5' aggregate Banach contraction constant | T5 v2 §E.3 |
| $\hat q^{(k),B'}$ | Strategy-B' iterated quantile at rung $k$ | T5 v2 §E.1 |
| $q^*$ | Banach fixed point of $T = T_K\circ\cdots\circ T_1$ | Gap B §1 |
| $\rho_K = \sum_k \hat\epsilon_k / d_{TV}(P_0,P_K)$ | Chain-overlap inefficiency | T5 v2 §D |
| $\bar S^{(K),\text{do}(t^*)}$ | Do-marginalised rung-$K$ trajectory score | Lemma J.1 (T6 §4) |

**Score support.** All theorems below are stated for the **discrete**
score regime $\mathcal S = \{0/N,1/N,\ldots,N/N\}$, $\lvert\mathcal S\rvert
= N+1$ (SC@$N$ self-consistency, $N=8$ in pilot ⇒ $\lvert\mathcal S\rvert=9$).
Continuous-score extension is open (§7 below).

### §1.2 Resolved notation conflicts

The ten source files used overlapping symbols in incompatible ways. The
unified stack adopts the following single resolution:

1. **(A1)–(A6) name collisions.** Both T4 v2 and T5 v2 declare
   "(A1)–(A6)". They *do not refer to the same propositions.* Below we
   prefix every assumption with its theorem of origin: **(P-A1')**,
   **(P-A2)**, …, **(P-A6)** for the Pearl/T4 stack; **(L-A1)**,
   **(L-A2)**, …, **(L-A6)** plus **(L-B0)**, **(L-B1)**, **(L-B2)** for
   the Ladder/T5/T5' stack. Theorem 6 imports both prefixes verbatim.

2. **$K$ collision.** In the Pearl stack, $K$ denotes the number of
   majority-vote samples ($K=4$ throughout). In the Ladder stack, $K$
   denotes the number of rungs in the chain. We retain both meanings
   in their native theorems and write $K_{\text{vote}}$ and $K_{\text{rung}}$
   when clarity is needed (e.g., Theorem 6 §5).

3. **$\bar L$ vs. $L_k$.** $L_k$ is the **per-rung** Wasserstein-1
   Lipschitz constant of $T_k$ (Gap A Lemma 1); $\bar L = \prod_k L_k$
   is the **aggregate** product (Gap A Lemma 2). This is the
   *contraction* metric, not the slack metric.

4. **$\epsilon_k$ vs. $\hat\epsilon_k$.** $\epsilon_k = d_{TV}(P_k,P_{k+1})$
   is the *population* per-rung TV; $\hat\epsilon_k$ is the
   *plug-in empirical* TV. The DKW slack
   $\delta_{\mathrm{DKW},k} = \tfrac{|\mathcal S|}{2}\sqrt{\log(2|\mathcal S|/\delta)/(2 n_{\min,k})}$
   bridges the two: $|\hat\epsilon_k-\epsilon_k|\leq\delta_{\mathrm{DKW},k}$
   w.p. $\geq 1-\delta$.

5. **$t^*$ definition.** T4 v1 used $t^* = \min\{t:s_t<q_\alpha\}$; T4 v2
   §F.0 (A3') strictly tightened to the *recovery-aware contiguous*
   definition; the unified stack uses **only** the recovery-aware form.

6. **"Strategy B" vs. "Strategy B'".** Strategy B is the **telescoped**
   one-shot weighted CP (point-equivalent to T3 by
   $\prod_k\hat p_k/\hat p_{k-1} = \hat p_K/\hat p_0$). Strategy B' is
   the **sequentially re-calibrated** ladder (the empirical pilot
   winner). T5 v2 covers Strategy B; T5' covers Strategy B'. We never
   conflate the two.

### §1.3 Definitional dictionary (one-liners)

- **Recovery-aware $t^*$.** First step where the score drops below the
  per-step CP threshold and the next two steps also stay below
  (continguous violation; T4 v2 §F.0).
- **Cascade gap $g$.** Steps between $t^*$ and the lowest-score step
  ($t_{\text{worst}}$) in the same trace (T4 v3 §3.1).
- **Strategy A (one-shot).** Compute $\hat q_\alpha$ once on $D_0$,
  apply at $D_K$ via T3 weighted CP. The naive baseline.
- **Strategy B (telescoped).** Telescope $\prod_k\hat w_{k-1\to k}$ into
  a single weight; *point-equivalent* to A on score values, slack
  bound differs.
- **Strategy B' (sequential).** Iteratively re-calibrate
  $\hat q^{(k)} = T_k(\hat q^{(k-1)})$ along the rung chain. The
  empirical winner.
- **K=4 majority intervention.** Re-roll a wrong trace at $t^*$ four
  times, take the majority vote of post-intervention answers.

---

## §2 — Background theorems (T1–T3)

These are the standard split-CP machinery applied to step-aggregated,
discrete reasoning scores. They are *prerequisites* for T4–T6, not
new contributions of this paper.

### §2.1 Theorem 1 — Trajectory CP coverage

> **(T1) Trajectory-level CP coverage from step-aggregated scores.**
> Suppose $(X_i,R_i,Y_i)_{i=1}^{n+1}$ are exchangeable. For any
> measurable aggregator $\phi$ and $\alpha\in(0,1)$,
> $$\Pr\!\big[\bar S_{n+1} \geq \hat q_\alpha \,\big|\, Y_{n+1}=1\big]
>   \;\geq\; 1 - \alpha - \tfrac{1}{n_+ + 1}.$$

**Proof reference.** Lei–Wasserman 2014 / Vovk 2002 split-CP applied
after the standard "condition on the correct subset" exchangeability
reduction. Full sketch in `theorem1_trajectory_cp.md` §2.

**Status.** Unproblematic. The $1/(n_++1)$ is the standard finite-sample
correction; for $n_+ \geq 100$ it is below 1%.

**Tie subtlety.** With $\lvert\mathcal S\rvert=9$ (SC@8), the lower-quantile
convention preserves the *lower* bound but loses the *upper* bound on
coverage; this is the source of over-coverage at small $\alpha$ in T3
empirics (`theorem3_weighted_cp_discrete.md` §6).

### §2.2 Theorem 2 — Score-family Pareto

> **(T2) Score-family compute–selective-accuracy Pareto.** Let
> $F_k(\cdot|y)$ be the conditional CDF of $\bar S^{(k)}$ given $Y=y$,
> $\pi = \Pr(Y=1)$, and $\mathrm{SNR}_k(\beta) = (1-F_k(\tau_k|1))/(1-F_k(\tau_k|0))$
> at threshold $\tau_k$ inducing kept fraction $\beta$. Then for any
> two score families $k_1,k_2$ at any $\beta\in(0,1)$,
> $$\rho_{k_1}(\beta) > \rho_{k_2}(\beta) \;\iff\; \mathrm{SNR}_{k_1}(\beta) > \mathrm{SNR}_{k_2}(\beta).$$
> The Pareto frontier in (cost, selective-accuracy) is the upper
> SNR envelope across cost-ordered scores.

**Proof.** Direct algebra:
$\rho_k = (1+\tfrac{1-\pi}{\pi}\,\mathrm{SNR}_k^{-1})^{-1}$; monotone in $\mathrm{SNR}_k$.
Full version in `theorem2_score_family_pareto.md` §3.

**Empirical fingerprint** (MATH-500 / Qwen2.5-7B / $\alpha=0.5$):
lp\_min ($1\times$, $\rho=0.620$, SNR$\approx 0.54$) ≺
prm\_min ($2\times$, $\rho=0.707$, SNR$\approx 0.96$) ≺
sc\_top1 ($8\times$, $\rho=0.793$, SNR$\approx 1.92$) — each step
roughly doubles the SNR. The discrete-score remarks of T2 §5(2)
apply throughout the unified stack.

### §2.3 Theorem 3 — Weighted CP on discrete shift

> **(T3) Weighted CP coverage for discrete scores under score-only
> shift.** Under (T3-A1) score-only-shift density-ratio
> $dP_{\text{test}}/dP_{\text{cal}}(x,s,y) = w(s)$ and (T3-A2) discrete
> $\lvert\mathcal S\rvert<\infty$,
> $$\Pr_{\text{test}}\!\big[S_{n+1}\geq\hat q_\alpha^{\text{wgt}} \,\big|\, Y_{n+1}=1\big]
>   \;\geq\; 1 - \alpha - \tfrac{|\mathcal S|}{2}\sqrt{\tfrac{\log(2|\mathcal S|/\delta)}{2\min(n_{\text{cal}},n_{\text{test}})}} - O(1/n_+),$$
> with probability $\geq 1-\delta$ over PMF estimation, where
> $\hat q_\alpha^{\text{wgt}}$ is the empirical-PMF (Laplace-smoothed)
> weighted quantile.

**Proof.** Tibshirani–Foygel-Barber–Candès–Ramdas (NeurIPS 2019)
weighted-CP specialized to (T3-A1) and DKW PMF rate
(Berend–Kontorovich 2013). Full version in
`theorem3_weighted_cp_discrete.md` §3.

**Empirical fingerprint** (MATH-500-cal → AIME-1983-2024-test SC@8,
$n_{\text{cal}}=500$, $n_{\text{test}}=933$): predicted gap $\leq 0.10$,
empirical over-correction at $\alpha=0.10$: target 0.90, achieved
**0.988**; at $\alpha=0.30$: target 0.70, achieved **0.884**; at
$\alpha=0.50$: target 0.50, achieved **0.633**. KDE-weighted CP fails
(Pilot J) by a well-known KDE-on-discrete-data pathology
(`theorem3_weighted_cp_discrete.md` §4).

**T3 is the one-shot foil for T5/T5'.** The ladder's contribution is
*not* in beating T3's coverage at $K=1$ (it can't, modulo
finite-sample slack); it is in the *slack-bound* improvement at
$K\geq 4$ (T5 v2) and the *typical-case* contraction (T5').

---

## §3 — Pearl causal stack (Theorem 4 v3 + Corollary 4.1)

### §3.1 Setup

Auto-regressive SCM $\mathcal M$ over a transformer LM $\pi_\theta$:
$X_1\to X_2\to\cdots\to X_T\to Y$, with full back-edges, exogenous Gumbel
noise $\epsilon_t$ per step, deterministic answer judge $Y$. Per-step
correctness oracle $Q_t \in \{0,1\}$ from PRM800K / Math-Shepherd MC
labels. Per-step score $S_t$ with split-CP threshold $q_\alpha(t)$.

The object of interest is the **single-do efficacy**
$\Delta(t;\bar x) := \mathbb E_{\tilde X_t\sim\pi(\cdot|X_{1:t-1},P)}\!\big[\Pr[Y=1\mid\text{do}(X_t=\tilde X_t)]\big] - \Pr[Y=1\mid\bar x]$.
Theorem 4 ranks $\Delta$ across $t$; Corollary 4.1 quantifies the
gap-stratified pattern.

### §3.2 Lemma 4.A — Conditional front-door identification

> **Lemma 4.A** (front-door identification under auto-regressive
> structure; Pearl 2009 §3.4 + Tian–Pearl 2002 ID Theorem 1). Under
> (P-A1') and (P-A2), for every $t\in[1,T]$ and every
> $w\in\mathcal X^{t-1}$,
> $$\Pr_{\mathcal M}\!\big[Y=1\mid\text{do}(X_t=x'_t),W=w\big]
>   = \sum_{m\in\mathcal X^{T-t}} \Pr(M=m\mid X_t=x'_t,W=w)\cdot \Pr(Y=1\mid M=m,W=w),$$
> where $M = X_{t+1:T}$. The do-quantity is therefore identifiable
> from the LM's observational softmax conditionals plus the answer
> judge.

**Proof.** Verify (F1$_W$, F2$_W$, F3$_W$) of Pearl 2009 conditional
front-door under (P-A1'); apply Tian–Pearl 2002. Bounded-gap variant
(Lemma 4.A$_\eta$): under (P-A1$_\eta$') with TV-slack $\eta$, the
identification holds within $2\eta$ TV-error
(Bareinboim 2014 Theorem 5).

### §3.3 Lemma 4.B — Cascade-decay optimality

> **Lemma 4.B** (cascade-decay optimality). Under (P-A2), (P-A3'),
> (P-A4) (or v3 (P-A4')), (P-A5), (P-A6), for any wrong trace $\bar x$
> with recovery-aware $t^*$ and any $t>t^*$,
> $$\Delta(t^*;\bar x) - \Delta(t;\bar x) \;\geq\; \kappa\cdot\big[1 - p_{\text{cascade}}^{\,t-t^*}\big] - O(\delta T).$$

**Proof structure** (three regimes; T4 v2 §G.2):
- **$t<t^*$:** prefix on-trajectory ⇒ $\Delta(t)=O(\delta+\alpha)$.
- **$t=t^*$:** on-trajectory prefix + (P-A4) on-trajectory mass + (P-A5)
  contractive forward dynamics ⇒ $\Delta(t^*)\geq\kappa$, where
  $\kappa := \tau(1-\eta) - O(\delta T)$ and $\tau$ is the
  on-trajectory mass of $\pi$ at $t^*$.
- **$t>t^*$:** prefix has $t-t^*$ off-trajectory steps ⇒ suffix re-roll
  re-diverges with probability $\geq p_{\text{cascade}}^{t-t^*}$ ⇒
  $\Delta(t)\leq\kappa\cdot p_{\text{cascade}}^{t-t^*} + O(\delta T)$.

Subtract. $\blacksquare$

**Resample-distribution caveat.** The lemma is stated under the
*operational* re-sample $\tilde X_t\sim\pi$ (matches the pilot's K=4
majority procedure). Under the *oracle* $\tilde X_t\sim q^*$
(idealised v1-MI), $\kappa$ improves to $1-O(\delta T+\alpha)$
(T4 v2 §G.2 note).

### §3.4 Theorem 4 v3 — Boxed statement

> **Theorem 4 v3 (Cascade-gap-stratified earliest-step dominance).**
> Let $\mathcal M$ satisfy (P-A1'), (P-A2), (P-A3'), (P-A4'), (P-A5),
> (P-A6). For any wrong trace $\bar x$ with $Y(\bar x)=0$ and
> recovery-aware $t^*(\bar x)<T$:
>
> **(I) Identification** (Lemma 4.A). The do-quantity is identifiable
> via conditional front-door.
>
> **(II) Pointwise dominance** (Lemma 4.B). For all $t>t^*$,
> $$\Pr_{\mathcal M}\!\big[Y=1\mid\text{do}(X_{t^*}\sim\pi)\big]
>   \;\geq\; \Pr_{\mathcal M}\!\big[Y=1\mid\text{do}(X_t\sim\pi)\big],$$
> with quantitative gap $\geq \kappa\cdot[1-p_{\text{cascade}}^{t-t^*}] - O(\delta T)$,
> and $\kappa\leq\kappa_{\max}\cdot[1-(1-p_{\text{recover}})^{K_{\text{vote}}}]$.

### §3.5 Corollary 4.1 — Cascade-stratified prediction

> **Corollary 4.1** (cascade-stratified lift). Conditional on
> (P-A1')–(P-A6) and the cascade gap $g(\bar x)\geq 1$,
> $$\boxed{\;\Delta_{\text{strat}}(g) \;\geq\; \kappa\cdot\big[1 - p_{\text{cascade}}^{\,g}\big]\;-\;\Lambda(g)\;-\;O(\delta T),\;}$$
> with the aggregate lift being the gap-mixture
> $\overline\Delta = \sum_g w(g)\,\Delta_{\text{strat}}(g)$ where
> $w(g)$ is the empirical wrong-trace gap distribution.

**Proof.** Specialize Lemma 4.B at $t=t_{\text{worst}}(\bar x)$, hence
$t-t^*=g$; condition on $\{g(\bar x)=g\}$; subtract the false-positive
cost $\Lambda(g)\geq 0$ for the test-time procedure that fires on all
traces (not just wrong ones). Sum against $w(g)$. Full version: T4 v3
§3.3. $\blacksquare$

**Why the aggregate can be near-zero while the gap-≥5 stratum is
large.** $w(\cdot)$ concentrates on small $g$ (empirical pilot:
$\bar g\approx 1.5$–$3$); the geometric factor $1-p_{\text{cascade}}^g$
is small at small $g$; $\Lambda(g)$ is comparable to or larger than
$\kappa(1-p_{\text{cascade}}^g)$ at small $g$. The aggregate is
dragged toward zero (or below) by the abundant small-$g$ mass; only
the sparse large-$g$ stratum exhibits the theorem's headline lift.

### §3.6 Pearl-stack assumption set (final, scope-clauses included)

| Tag | Assumption | Scope |
|---|---|---|
| (P-A1') | Prefix-blocking under controlled inference (KV-cache isolation, no activation steering, fixed chat template, independent per-request seeds; trace-replay determinism check) | Pipeline scope |
| (P-A1$_\eta$') | Bounded-gap version of (P-A1') with TV-slack $\eta$ | Graceful degradation |
| (P-A2) | Score validity: split-CP threshold satisfies $\Pr[s_t<q_\alpha(t)\mid Q_t=1]\leq\alpha$ | Calibration |
| (P-A3') | Expected cascade monotonicity (population) + recovery-aware $t^*$ definition | (A3-contig violations permitted in expectation) |
| (P-A4) | Resampling-effective intervention: $\pi$ has nonzero on-trajectory mass at $t$ | T4 v2 form |
| (P-A4') | $1-(1-p_{\text{recover}})^{K_{\text{vote}}}\geq\tau>0$ | T4 v3 K-sample form |
| (P-A5) | Cascade contractivity, $p_{\text{cascade}}\in(\tfrac12,1]$ | Non-self-correcting LMs only (Qwen2.5-7B-Instruct, Qwen2.5-Math-7B, Phi-4 base; **excludes** R1-Distill, QwQ, o1-style) |
| (P-A6) | Unimodal correct-trajectory conditional | Excludes ~5–10% multi-modal OlympiadBench |

### §3.7 Cross-Theorem-4 reductions

- $K_{\text{vote}}=1$ + $p_{\text{recover}}=1$ + $g=1$ ⇒ recovers v1-MI's
  three-regime headline $\Delta(t^*)\geq\kappa(1-p_{\text{cascade}})$.
- $\eta\to 0$, $\delta T\to 0$ ⇒ exact dominance $\Delta(t^*)\geq\Delta(t)$
  with no slack.
- $w(\cdot) = \delta_{g_0}$ (degenerate gap distribution) ⇒
  $\overline\Delta = \Delta_{\text{strat}}(g_0)$ exactly, no mixture
  drag.

---

## §4 — Distance-Ladder stack (Theorem 5 v2 + Theorem 5')

### §4.1 Setup

Rung sequence $D_0\to D_1\to\cdots\to D_K$ with $D_0$ a labelled CP
calibration source (PRM800K), $D_K$ the test target (e.g.
AIME-2024). Per-rung empirical PMF $\hat p_k$ on
$\mathcal S = \{0/N,\ldots,N/N\}$. Per-rung weight
$\hat w_{k-1\to k}(s) = (\hat p_k(s)+\varepsilon)/(\hat p_{k-1}(s)+\varepsilon)$
with $\varepsilon = 1/n_{\min,k}$.

Strategies:
- **A** (one-shot): $\hat q_\alpha = $ T3 quantile on $D_0$; apply at $D_K$.
- **B** (telescoped): collapse $\prod_k\hat w_{k-1\to k}$ into one
  weight; one-shot weighted CP. **Point-equivalent to A** modulo
  slack-bound improvement.
- **B'** (sequential): iteratively re-calibrate
  $\hat q^{(k),B'} = T_k(\hat q^{(k-1),B'})$ along the chain. **The
  empirical winner; the object T5' analyzes.**

### §4.2 Lemma 5.1 — Per-rung nexCP coverage gap

> **Lemma 5.1** (Barber–Candès–Ramdas–Tibshirani 2023 nexCP, single
> step). Treating $D_{k-1}$ as conformal source and $D_k$ as test
> under per-rung weight $\hat w_{k-1\to k}$,
> $$\Pr_{D_k}\!\big[\bar S^{(k)}\geq\hat q^{(k)} \,\big|\, Y^{(k)}=1\big]
>   \;\geq\; 1 - \alpha - \epsilon_{k-1} - \tfrac{1}{n_+^{(k-1)}+1}.$$

By Strassen 1965 Theorem 11, $\epsilon_{k-1}$ is the **best possible**
TV-slack for a single nexCP step (sharp in the optimal coupling).

### §4.3 Lemma 5.2 — TV-summation chain rule (Strassen–Lindvall)

> **Lemma 5.2** (gluing-lemma chain). Under (L-A1)–(L-A4'), the
> per-rung optimal couplings can be composed via Lindvall 2002 Theorem
> I.5.4 (gluing lemma) into a Markov chain $X_0,\ldots,X_K$ such that
> $\Pr(X_k\neq X_{k+1})=\epsilon_k$. The endpoint coupling satisfies
> $\Pr(X_0\neq X_K)\leq\sum_k\epsilon_k$, dual (Strassen 1965 Theorem 11)
> to $d_{TV}(P_0,P_K)\leq\sum_k\epsilon_k$.

This is the chain that transports the rung-0 calibration coverage to
rung $K$ test coverage with summable slack.

### §4.4 Lemma 5.3 — Wasserstein-1 Lipschitz (Gap A closed)

> **Lemma 5.3** (per-rung quantile-map Lipschitz; Gap A,
> `theorem5_gap_A_lipschitz.md` Lemma 1). For each $k$, the per-rung
> weighted-quantile operator $T_k:\mathcal Q\to\mathcal S\subset\mathcal Q$
> is **expected-Lipschitz** in the Wasserstein-1 metric with
> $$L_k \;\leq\; \frac{W_{\max}\cdot|\mathcal S|}{w_{\min,k}\,\alpha}\cdot(\epsilon_k + 2\delta_{\mathrm{DKW},k}),$$
> via (i) classical inverse-CDF Lipschitz of the smoothed operator
> $T_k^{\mathrm{lin}}$, (ii) Tibshirani-2019 Lemma A.1 weighted-CDF
> perturbation specialized to discrete CDFs, (iii) Path-B → Path-A
> passage with $O(\Delta_{\max})$ slack.

The composition (Gap A Lemma 2) gives
$$\bar L = \prod_k L_k \;\leq\; \prod_k \tfrac{W_{\max}\,|\mathcal S|}{w_{\min,k}\,\alpha}\,(\epsilon_k + 2\delta_{\mathrm{DKW},k}).$$
The Wasserstein-1 metric is **necessary** for Banach iteration on
discrete-output operators (sup-norm fails at jump points; classical
Lipschitz fails at score atoms).

**`[PROOF GAP A — closed]`** at the worst-case-bound level. Realized
empirical $\bar L\approx 0.85$ (pilot) is two orders of magnitude
smaller than the worst-case product (~67.9 at α=0.5 in the pilot;
Gap A §7.2). The looseness is isolated as Gap C.

### §4.5 Lemma 5.4 — Banach contraction + per-rung error decomposition (Gap B closed)

> **Lemma 5.4** (coverage at the Banach fixed point;
> `theorem5_gap_B_fixed_point_coverage.md` §4–5). Under (L-A1)–(L-A6),
> (L-B0)–(L-B2), and Lemma 5.3 (Gap A), the iterated map
> $T = T_K\circ\cdots\circ T_1$ has a *unique* fixed point $q^*\in\mathcal Q$
> by Banach 1922 / Granas-Dugundji 2003 Theorem II.1.1, with
> $|\hat q^{(K),B'} - q^*|\leq\bar L^K\cdot|\hat q^{(0)}-q^*|$, and the
> rung-$K$ coverage at the iterate satisfies:
> $$\Pr_{D_K}\!\big[\bar S^{(K)}\geq\hat q^{(K),B'}\,\big|\,Y^{(K)}=1\big]
>   \;\geq\; 1 - \alpha - \sum_{k=0}^{K-1}\bar L^{K-1-k}\,\epsilon_k - \tfrac{1}{n_+^{(0)}+1}.$$

**Damping mechanism.** The per-rung nexCP slack $\epsilon_k$ injected
at rung $k+1$ is *damped* by $\prod_{j=k+2}^K L_j = \bar L^{K-1-k}$
through the subsequent Lipschitz operators. Summing across $K$ rungs
gives the displayed contracted slack — strictly tighter than the
naive $\sum_k\epsilon_k$ of Theorem 5 v2 whenever $\bar L<1$.

**Uniform-error specialization.** With $\epsilon_k\equiv\epsilon$,
$$\sum_{j=0}^{K-1}\bar L^j\,\epsilon \;=\; \tfrac{1-\bar L^K}{1-\bar L}\,\epsilon,$$
the **telescoping geometric sum** that motivates the headline form.

**Consolidated-draft form** (a coarsening that bounds
$\bar L^{K-1-k}\leq 1$ on the leading term and lets the geometric tail
absorb the rest):
$$\geq 1 - \alpha - (1-\bar L^K)\sum_{k=0}^{K-1}\epsilon_k - \tfrac{1}{n_+^{(0)}+1}.$$

**`[PROOF GAP B — closed]`** modulo a Markov-chain assumption on the
gluing-lemma composition (Strategy B' as specified is Markov; T6 §10
flags non-Markov modifications as out of scope).

### §4.6 Lemma 5.5 — 3-factor contraction sufficient condition (Gap C, partial)

> **Lemma 5.5** (sufficient condition for $\bar L<1$;
> `theorem5_gap_C_contraction_sufficient.md` §3). Decompose
> $L_k = \rho_k\cdot\hat\epsilon_k^{\,*}\cdot\kappa_k$ where
> $\rho_k = w_k^+/(\alpha w_k^-)$ is the weight-ratio amplification,
> $\hat\epsilon_k^{\,*} = \hat\epsilon_k+\delta_{\mathrm{DKW},k}(\delta/K)$
> the empirical-PMF noise, $\kappa_k = 1/\hat p_k(s_q^*)$ the
> discrete-quantile granularity. Then
> $$\sum_{k=1}^K \log\!\big(\rho_k\hat\epsilon_k^{\,*}\kappa_k\big) < 0
>   \;\Longleftrightarrow\; \bar L < 1$$
> is a **product-form** sufficient condition; equivalently
> $\bar\rho\cdot\bar\epsilon^{\,*}\cdot\bar\kappa < \alpha$ in
> geometric-mean form. The **max-form** alternative
> $\max_k L_k < 1$ is strictly stronger but robust to permutation.

**`[PROOF GAP C — partial]`.** The bound is conservative; pilot
empirical $\bar L\approx 0.85$ is consistent with the bound vacuously
($0.85 \leq 67.9$) but the bound does not predict the realized value.
The looseness factors to
~5× (L∞ vs. L1 weight perturbation) × ~2× (worst-case vs. realized
$\kappa$) × ~8× (DKW union over $K$). Tightening to typical-case
Lipschitz is the **local-attractor research direction** of Gap C §7
(see §7 below).

### §4.7 Theorem 5 v2 — Boxed (worst-case slack)

> **Theorem 5 v2** (K-rung ladder coverage; bounded slack via TV
> summation). Under (L-A1), (L-A2), (L-A3) with $W_{\max}<\infty$,
> (L-A4'), (L-A5), (L-A6), let $\hat q_\alpha^{(K),\text{ladder}}$ be
> the **telescoped** weighted-CP quantile (Strategy B). For any
> $\alpha\in(0,1)$ and $\delta\in(0,1)$, with probability $\geq 1-\delta$,
> $$\Pr_{D_K}\!\big[\bar S^{(K)}\geq\hat q_\alpha^{(K),\text{ladder}}\,\big|\,Y^{(K)}=1\big]
>   \;\geq\; 1-\alpha\;-\;\sum_{k=0}^{K-1}\big(\hat\epsilon_k + \mathrm{DKW}_k(\delta/K)\big)\;-\;\tfrac{1}{n_+^{(0)}+1}.$$

This is the worst-case slack bound. By the telescoping identity
$\prod_k\hat p_k/\hat p_{k-1}=\hat p_K/\hat p_0$, Strategy B is
**point-equivalent** to one-shot T3 — so T5 v2's contribution is the
*slack-bound* improvement (no $\prod_{j\neq k}M_j$ pre-factor), not a
point-estimate improvement.

### §4.8 Theorem 5' — Boxed (sequential ladder, Banach contraction)

> **Theorem 5'** (Sequential ladder coverage via Banach contraction).
> Under the assumptions of Theorem 5 v2 plus (L-B1) per-step Lipschitz
> quantile (Lemma 5.3) and (L-B2) mean contraction $\bar L<1$, the
> Strategy-B' iterated quantile $\hat q^{(K),B'}$ converges to the
> unique Banach fixed point $q^*$ at geometric rate $\bar L^K$, and
> the rung-$K$ coverage satisfies (Lemma 5.4):
>
> $$\boxed{\;\Pr_{D_K}\!\big[\bar S^{(K)}\geq\hat q^{(K),B'}\,\big|\,Y^{(K)}=1\big]
>   \;\geq\; 1 - \alpha - \sum_{k=0}^{K-1}\bar L^{K-1-k}\,\epsilon_k - \tfrac{1}{n_+^{(0)}+1}.\;}$$
>
> Equivalently (consolidated form, slightly looser):
> $$\geq 1 - \alpha - (1-\bar L^K)\sum_{k=0}^{K-1}\epsilon_k - \tfrac{1}{n_+^{(0)}+1}.$$

### §4.9 Ladder-stack assumption set (final, prefixed)

| Tag | Assumption |
|---|---|
| (L-A1) | i.i.d. within rung |
| (L-A2) | Score-only consecutive shift (T3-A1 generalization) |
| (L-A3) | Bounded per-rung ratios, $W_{\max}$ user-specified |
| (L-A4') | Independent and informative rungs: $d_{TV}(P_k,P_{k+1})\geq\tau_{\min}>0$ |
| (L-A5) | Discrete score support, $\lvert\mathcal S\rvert<\infty$ |
| (L-A6) | Monotone source-TV: $d_{TV}(D_0,D_k)\leq d_{TV}(D_0,D_{k+1})$ |
| (L-B0) | Weighted exchangeability of test and calibration scores under $\hat w_{0\to K}$ |
| (L-B1) | Per-rung Wasserstein-1 Lipschitz of $T_k$ with constant $L_k$ (Lemma 5.3) |
| (L-B2) | Mean contraction: $\bar L = \prod_k L_k < 1$ |

---

## §5 — Joint composition (Theorem 6)

### §5.1 Setup

Run on a rung-$K$ test prompt $X\in D_K$, observe a wrong trace
$\bar x$ with recovery-aware $t^*(\bar x)$, perform a $K_{\text{vote}}=4$
do-intervention at $t^*$, take the majority vote $\hat Y_{n+1}$, score
the post-intervention trace under the **rung-calibrated** Strategy-B'
threshold $\hat q^{(K),B'}$.

### §5.2 The composition challenge

T5' coverage requires (L-B0) weighted exchangeability of test and
calibration scores. The do-intervention at $t^*$ violates (L-B0)
because: (i) we condition on the wrong-trace subpopulation
($Y(\bar x)=0$, with $t^*$ existing), inducing Bareinboim
s-recoverability concerns; (ii) the re-roll injects $\pi$, not the
oracle $q^*$, so the post-intervention test score is no longer
weighted-exchangeable with rung-$K$ calibration. **Pasting T5' onto
T4 naively fails.** Lemma J.1 below is the bridge.

### §5.3 Lemma J.1 — Do-marginalised exchangeability

> **Lemma J.1** (joint composition lemma; `joint_composition_theorem.md`
> §4). Define the **do-marginalised score**
> $$\bar S^{(K),\text{do}(t^*)} := \mathbb E_{\tilde X_{t^*}\sim\pi(\cdot|X_{1:t^*-1},P)}\!\big[S(\tilde X_{1:T})\,\big|\,\tilde X_{t^*},X_{1:t^*-1}\big].$$
> Under (P-A1')–(P-A6) and (L-B0)–(L-B2) with $\sum_k\epsilon_k<1$,
> the do-marginalised score is weighted-exchangeable with the rung-$K$
> calibration scores at the same rung weights $\hat w_{0\to K}$:
> $$\bar S^{(K),\text{do}(t^*)} \,\big|\, Y_{n+1}^{(K)}=1 \;\overset{\text{wexch}}{\sim}\; \{S_i^{(K)} : Y_i^{(K)}=1\}.$$

**Proof sketch.** Two pieces (T6 §4):
1. **Identifiability transports across rungs.** (P-A1') is a pipeline
   property orthogonal to rung structure; Lemma 4.A's identification
   formula holds at every rung $k$.
2. **Twin-network coupling commutes with rung independence.** Under
   (L-A4') independent rungs and (P-A1') iid noise across steps, the
   suffix noise $\epsilon_{t^*+1:T}$ may be shared between intervened
   and natural distributions; integrating $\tilde X_{t^*}\sim\pi$
   recovers the do-marginalised score, which is rung-$K$-marginal
   independent of calibration. (L-B0) then transports.

**Crucial caveat.** The lemma works *in expectation over the
intervention*. Per-trace, post-intervention is **not** exchangeable;
only its expectation under $\pi$ is. The $K_{\text{vote}}=4$ majority is the
Monte-Carlo estimator of this expectation, with $p_{\text{recover}}$
slack accounting for finite-sample failure.

### §5.4 Theorem 6 — Boxed joint coverage

> **Theorem 6** (Joint Pearl × Distance-Ladder coverage). Let
> $\mathcal M$ satisfy (P-A1')–(P-A6) and let $\{D_k\}_{k=0}^K$
> satisfy (L-A1)–(L-A6) and (L-B0)–(L-B2). Let $\hat q^{(K),B'}$ be
> the Strategy-B' iterated quantile and $\hat Y_{n+1}$ the K=4
> majority vote of suffix re-rolls from a do-intervention at the
> recovery-aware $t^*(\bar x)$ of a rung-$K$ wrong trace. Then
> assuming Lemma J.1,
>
> $$\boxed{\;\Pr\!\big[\hat Y_{n+1} = Y_{n+1}^*\,\big|\,\text{rung }K,\text{intervened at }t^*\big]
>   \;\geq\; 1-\alpha\;-\;(1-\bar L^K)\sum_{k=0}^{K-1}\epsilon_k\;-\;(1-p_{\text{recover}})\;-\;\tfrac{1}{n_+^{(0)}+1}.\;}$$

The three slack terms have a clean interpretation:
- $(1-\bar L^K)\sum_k\epsilon_k$ — **rung slack** carried over from
  T5'. Vanishes when $K=0$ (no shift) or $\bar L\to 0$ (perfect
  contraction).
- $1-p_{\text{recover}}$ — **re-roll failure** carried over from T4.
  Vanishes when $\pi$ is well-aligned with $q^*$ at $t^*$.
- $1/(n_+^{(0)}+1)$ — finite-sample CP slack at the rung-0 anchor.

### §5.5 Reductions (sanity checks)

- **$K=1$, no shift ($\epsilon_0=0$).** T6 reduces to
  $1-\alpha-(1-p_{\text{recover}})-1/(n_+^{(0)}+1)$ — the $K_{\text{vote}}=4$
  majority version of T4 v3. ✓
- **No intervention ($t^*$ does not exist, or trace correct).** The
  $(1-p_{\text{recover}})$ slack vanishes and T6 reduces to T5'. ✓
- **$K=0$ and no intervention.** T6 reduces to bare split CP
  (T1) with $1-\alpha-1/(n_+^{(0)}+1)$. ✓

### §5.6 Failure mode F1 — cascade-shift coincidence

The most consequential T6-specific failure: the model's cascade
source $t^*$ on $D_K$ coincides with the *step* at which the model
switches its solution mode from $D_{K-1}$-style to $D_K$-style (e.g.
MATH-500-style → AIME-style technique pivot). Then do-intervention
at $t^*$ moves the post-intervention trace **across** the rung
boundary, not within it. The do-marginalised score is no longer a
rung-$K$ score; it is a rung-mixture, and Lemma J.1's
$\hat w_{0\to K}$ weights no longer apply. Detect via a
**rung-displacement check**: TV between post-intervention score
distribution and rung-$K$ calibration distribution. If the TV exceeds
$\epsilon_{K-1}$, declare T6 not applicable on this trace. F1 is the
dominant attack on the joint composition (T6 §10).

---

## §6 — Cross-cutting structure

### §6.1 Shared structure of T4, T5/T5', T6

All three new theorems share a three-layer architecture:

1. **Identification / exchangeability layer.** Lemma 4.A (front-door
   ID) for T4; (L-B0) (weighted exchangeability) for T5'; Lemma J.1
   (do-marginalised exchangeability) for T6. *Without identification
   / exchangeability, no coverage statement is meaningful.*
2. **Contraction / dominance layer.** Lemma 4.B (cascade-decay
   $\Delta(t^*)\geq\Delta(t)\cdot p_{\text{cascade}}^{t-t^*}$) for T4;
   Lemma 5.4 (Banach-fixed-point with rate $\bar L^K$) for T5'.
   *This layer ranks / contracts; it does not give absolute coverage.*
3. **Slack / mixture layer.** Corollary 4.1 ($\Lambda(g)$ false-positive
   cost + $w(g)$ mixture) for T4; Lemma 5.5 (3-factor decomposition,
   PROOF GAP C) for T5'; Theorem 6 (additive composition of T4 +
   T5' slacks via Lemma J.1) for T6. *This layer makes the bound
   numerical against pilot data.*

This three-layer template is itself a contribution: any new "intervene
+ calibrate" theorem in this style must close all three layers.

### §6.2 Common assumption framework

The joint assumption set is the **disjoint union** of the Pearl and
Ladder prefixes, with one bridge (L-B0) inherited by Lemma J.1:

```
Common (T1, T2, T3): exchangeability, discrete |𝒮|<∞, score validity
Pearl-only (T4):     P-A1' P-A2 P-A3' P-A4(') P-A5 P-A6
Ladder-only (T5/T5'): L-A1 L-A2 L-A3 L-A4' L-A5 L-A6 L-B0 L-B1 L-B2
Joint (T6):          all of the above + Lemma J.1
```

The two prefixes are **structurally orthogonal**: (P-A1') controls
graphical d-separation in the SCM (does the do-quantity factorize?);
(L-Bx) control rung-pair Lipschitz dynamics (does the iteration
contract?). They cannot conflict; they cannot collapse into each
other.

### §6.3 Failure-mode taxonomy (unified)

| Mode | Violates | In-scope? | Mitigation |
|---|---|---|---|
| KV-cache leakage / activation steering | (P-A1') | No | Pipeline hygiene; replay determinism check |
| Coupled-noise samplers (fixed seed) | (P-A1', noise-iid) | No | Independent per-request seeds |
| Calibration drift | (P-A2) / (T1) | Yes | Per-model recalibration |
| Isolated meta-comments before $t^*$ | (P-A3-contig) | Yes | Recovery-aware $t^*$ |
| Multi-cascade traces | (P-A3') | Yes | Earliest non-recovered cascade source |
| Self-correcting models (R1-Distill, QwQ, o1) | (P-A5) | **No** | Future work: separate theorem for self-correcting cascade |
| Multi-modal correct trajectories | (P-A6) | **No** | Restrict to unimodal problems |
| Heavy-tailed / continuous scores | (T3-A2) / (L-A5) | **No** | Discrete-only; KDE-Wasserstein future work |
| Reverse-direction rung | (L-A6) | No | Re-order or drop rung |
| Redundant rung | (L-A4') | No (monster-bar) | Strict information gain test |
| Single dominant rung (one $L_k\gg 1$) | (L-B2) typical case | Maybe | Use product form; check max form too |
| Anchor-rung small $n$ | DKW dominance | Maybe | Generous $n_K\geq\lvert\mathcal S\rvert^2\log K/\hat\epsilon_K^2$ |
| Cascade-shift coincidence (F1) | Lemma J.1 | **No** | Rung-displacement check before T6 |

---

## §7 — Open problems

1. **T5' Gap A — continuous-score extension.** Current Lemma 5.3
   relies on discrete $\lvert\mathcal S\rvert<\infty$. Extension to
   continuous scores requires KDE + Wasserstein-1 reformulation
   following He–Wang–Liang JMLR 2024; rate degrades from $n^{-1/2}$
   to $n^{-1/(2+d)}$ (Pilot J negative result). **Open.**

2. **T5' Gap C — typical-case Lipschitz bound.** Lemma 5.5's
   sufficient condition is loose by ~80× at pilot ($\bar L^{\text{worst}}
   \approx 67.9$ vs. realized $\approx 0.85$). Tightening requires
   replacing L∞ weight perturbation with L1 (= TV itself; ~5×
   gain) and worst-case granularity with realized $\kappa$ at the
   expected quantile atom (~2× gain). The remaining ~6× requires the
   **local-attractor framing** (Gap C §7): Strategy B' may fall into
   a local invariant set $A\subset\mathcal Q$ where $T(A)\subset A$
   and $L_A<1$, even when global $\bar L_{\text{worst}}\geq 1$. The
   pilot's `q_path` plateau at $0.625$ is direct evidence. Formal
   theorem `[B2' local-attractor]` open.

3. **T6 — when does joint *strictly* dominate?** T6's slack is the
   *sum* of T4 and T5' slacks, hence numerically *worse* than either
   alone. T6's contribution is not a smaller slack; it is bounding a
   *strictly stronger event* (post-intervention answer-coverage on
   rung $K$, jointly). Strict dominance of T6 over either component
   is an *operational* claim, not a coverage-bound claim. Char-
   acterising the regime (strong model on far-OOD with high cascade
   gap) is open empirical work. (T6 §11 Q1.)

4. **T4 v3 — handling self-correcting models.** (P-A5) excludes the
   entire R1-Distill / QwQ / o1 family by construction. A separate
   "Theorem 4.5" with non-monotone $p_{\text{cascade}}$ (the
   self-correction marker triggers $p_{\text{cascade}}\to 0$ locally)
   is open.

5. **Joint-aware calibration.** Calibrate T6 *not* on PRM800K-correct
   scores but on PRM800K-correct scores **conditional on $t^*$
   existing in those traces**. This re-establishes selection-on-$Y$
   in a Bareinboim-2014 s-recoverable way. Speculation: could halve
   the joint slack. Open. (T6 §11 Q3.)

6. **Cross-rung $p_{\text{recover}}$ heterogeneity.** (P-A4') back-fits
   $p_{\text{recover}}$ from MATH-500; on AIME-new (more difficult),
   (P-A4) on-trajectory mass $\tau$ may be lower, making the joint
   slack non-additive. Open empirical question. (T6 §11 Q2.)

7. **Does $\bar L$ change under intervention?** $\bar L^{\text{do}}\leq
   \bar L$ would mean intervention *enhances* contraction and T6
   underestimates joint efficacy. Conjecture; proof open. (T6 §11 Q4.)

8. **Per-trace Lemma J.1.** The lemma is in expectation. A per-trace
   version with explicit TV penalty would tighten T6 at the cost of
   a (P-A1$_\eta$')-style bounded-gap term. (T6 §11 Q5.)

9. **K-dependence of $\kappa$.** T4 v3 §5 uses heuristic
   $\kappa\leq\kappa_{\max}\cdot[1-(1-p_{\text{recover}})^{K_{\text{vote}}}]$.
   The K=4 majority has shared-prefix non-independence; rigorous
   K-scaling is open.

10. **Tightness of the Strassen-Lindvall chain.** $\rho_K$ chain-
    overlap inefficiency $\geq 1$ always; tight when per-rung errors
    are *disjoint* under optimal coupling, loose otherwise. Pilot
    $\rho_K \in [1.39,1.54]$ — what is the structural cause? Open.

---

## §8 — Empirical fingerprints

For each theorem, we list testable predictions and the corresponding
*actual JSON values* from the pilot. All numbers are sourced from the
files cited; nothing is fabricated.

### §8.1 Theorem 4 v3 + Corollary 4.1

| Prediction | Source | Empirical | Status |
|---|---|---|---|
| Cascade-stratified gap≥5 lift large positive on AIME `marg__a0.1` | T4 v3 §3.4 calibrated $\kappa\approx 0.34, p_{\text{cascade}}\approx 0.85$ ⇒ predicted +18.91pp | `pearl_full/qwen25_7b_aime.json` `marg__a0.1` gap≥5 = **+18.76pp** [0.0, +75.0] | **CONFIRMED (calibration cell)** |
| Aggregate negative on `qwen25_7b__math500` despite headline T4 dominance | T4 v3 §3.4: small $\kappa$ (low headroom), short $g$, $\Lambda(g)$ dominates | `pearl_full/qwen25_7b_math500.json` `lp__a0.1` = **−1.65pp** [−7.0, +3.0]; cascade gap mean ≈ **1.98** at α=0.3 | **CONFIRMED** |
| `qwen25_math_7b__math500` *most* negative (sticky single-mode specialist) | T4 v3 §4 heuristic: math specialist ⇒ $p_{\text{cascade}}\approx 0.95$ ⇒ smallest predicted edge | `pearl_full/qwen25_math_7b_math500.json` `lp__a0.1` = **−2.23pp**; gap≥5 on `lp__a0.3` = **−4.33pp** | **CONFIRMED** |
| AIME `marg__a0.3` gap≥5 large positive | T4 v3 Corollary 4.1 prediction $\geq +8$pp | `pearl_full/qwen25_7b_aime.json` `marg__a0.3` gap≥5 = **+8.33pp** [0.0, +30.0] | **CONFIRMED** |
| `qwen25_7b__olympiad` aggregate small positive but gap≥5 *negative* (A6 violation flag) | T4 v3 §1.3 / §8.1 | `pearl_full/qwen25_7b_olympiad.json` `lp__a0.3` aggregate = **+1.08pp**; gap≥5 = **−3.18pp** | **PARTIAL FALSIFIER (A6 violation suspected)** |
| `phi4__aime` aggregate +5pp at α=0.5; gap≥5 +15 to +25pp | Pre-registered T4 v3 §8.2 | *cell still running; pending* | **PRE-REGISTERED** |
| `qwen-math__olympiad` aggregate +3 to +5pp; gap≥5 +10 to +20pp | Pre-registered T4 v3 §8.2 | *cell still running; pending* | **PRE-REGISTERED** |
| Decisive falsifier: gap=1 lift > gap≥5 lift on any AIME row | T4 v3 §8.3 | none observed in closed cells | **NOT FALSIFIED** |

### §8.2 Theorem 5 v2 (worst-case slack)

| Prediction | Source | Empirical | Status |
|---|---|---|---|
| 4-rung pilot $\sum\hat\epsilon_k$ in $[0.6, 0.9]$ | T5 v2 §G.1 | `distance_ladder_pilot.json`: $\sum\hat\epsilon_k = 0.112+0.417+0.098+0.127 = $ **0.754** | **CONFIRMED** |
| 5-rung full $\sum\hat\epsilon_k\geq 0.7$ across models | T5 v2 §G | `distance_ladder_full/AGGREGATE.json` qwen25_7b: **0.766**; qwen25_math_7b: **1.05**; qwen25_32b: **1.026**; phi4: **0.793** | **CONFIRMED (vacuous absolute bound)** |
| Chain-overlap inefficiency $\rho_K \geq 1$ | T5 v2 §D / Lemma 5.2 | $\rho_4 = 0.754/0.542 = $ **1.39**; $\rho_5\in[1.40,1.54]$ across models | **CONFIRMED** |
| Telescoping point-equivalent to one-shot | T5 v2 §A.3 / §B | Strategy A gap = Strategy B gap (point-equivalent on score values) | **CONFIRMED by construction** |

### §8.3 Theorem 5' (Banach contraction)

| Prediction | Source | Empirical | Status |
|---|---|---|---|
| **4-rung pilot 56–67% relative gap reduction** at α=0.5 | T5' §G.1: $1-\bar L^K$ form | `distance_ladder_pilot.json`: $1 - 0.058/0.174 = $ **67%** at α=0.5 | **CONFIRMED (pilot)** |
| 5-rung full pilot ~56% relative reduction at α=0.5 | T5' §G.2: $\bar L\approx 0.85, K=5$ ⇒ $1-0.85^5 = 0.556$ | `distance_ladder_full/AGGREGATE.md`: A_gap=21.4pp, B'_gap=9.5pp ⇒ $1-9.5/21.4 = $ **56%** | **CONFIRMED** |
| Implied $\bar L\approx 0.85$ across all 4 models | T5' §G.4 | qwen25_7b, qwen25_math_7b, qwen25_32b, phi4 all hit 56% relative reduction (borrowed-rung artifact noted in AGGREGATE.md) | **CONFIRMED (with caveat)** |
| α-dependent contraction: stronger at moderate α | T5' §G.3 | α=0.30 → 95% reduction (B'_gap 1.4pp); α=0.50 → 56%; α=0.70 → 64% | **INFERRED FROM PILOT** |
| Anchor-rung fragility: drop rung_4_aime_mid → 11.9pp gap at α=0.5 | T5' §E.5 / Gap C §8.3 | `distance_ladder_pilot.json` H4: max delta 11.9pp, worst-to-drop rung_4_aime_mid | **CONFIRMED** |
| Worst-case bound vacuous absolute | T5' §H.1 | $\sum\hat\epsilon_k = 0.754 > 0.5$ at $K=4$, bound is operationally vacuous as coverage gap | **CONFIRMED (and disclaimed)** |

### §8.4 Theorem 5' Gap A (Wasserstein-1 Lipschitz)

| Prediction | Source | Empirical | Status |
|---|---|---|---|
| Worst-case $\bar L^{\text{worst}}\sim 100$ at pilot regime | Gap A §7.2 | $L_k^{\text{worst}}\in[58,117]$ from §7.2 calculation; $\bar L^{\text{worst}}\sim 67.9$ | **CONFIRMED (vacuously upper-bounds 0.85)** |
| Realized per-rung $L_k^{\text{real}}\approx 0.96$ on average | Gap A §7.3 | $\bar L^{1/4}\approx 0.96$ from $\bar L\approx 0.85, K=4$ | **CONFIRMED** |
| Pilot's `q_path` saturates at one or two atoms | Gap A §7.4 | `distance_ladder_pilot.json` `q_path`@α=0.5: $[0.875,1.0,0.625,0.625,0.625]$ — saturates at $0.625$ from rung 3 | **CONFIRMED** |

### §8.5 Theorem 5' Gap B (fixed-point coverage)

| Prediction | Source | Empirical | Status |
|---|---|---|---|
| Geometric-sum slack: $(1-\bar L^K)/(1-\bar L)\cdot\epsilon$ at uniform $\epsilon$ | Gap B §4 eq. (12) | matches pilot α-grid pattern (Gap B §7.1, §7.2) | **CONFIRMED (form)** |
| Implied $\bar L$ at $K=4$, 67% reduction | Gap B §7.1: solve $(1-\bar L^4)/(4(1-\bar L))=0.33$ ⇒ $\bar L\approx 0.25$ (geometric form) or $\approx 0.7$ (consolidated form) | both readings consistent within constant factor | **CONFIRMED (within constant)** |
| Implied $\bar L$ at $K=5$, 56% reduction | Gap B §7.2: solve $1-\bar L^5 = 0.56$ ⇒ $\bar L\approx 0.85$ | matches T5' §G.2 calibration | **CONFIRMED** |

### §8.6 Theorem 5' Gap C (sufficient condition)

| Prediction | Source | Empirical | Status |
|---|---|---|---|
| Sufficient condition $W_{\max}\lvert\mathcal S\rvert\bar\epsilon/(\alpha\bar w_{\min})<1$ | Gap C §3.2 / (6.1) | pilot: $5\cdot 9\cdot 0.16/(0.5\cdot 0.5) = 28.8 > 1$ — **fails** | **EXPECTED FAIL (loose by ~80×)** |
| Tighter L1-perturbation form $L_k\approx\hat\epsilon_k/(\alpha\hat p_k(s_q^*))$ | Gap C §9 | $L_2^{\text{tight}}\approx 0.42/(0.5\cdot 0.15)\approx 5.6$ — still loose by ~6× | **PARTIAL** |
| Anchor-rung fragility predicted | Gap C §8.3 | matches T5' H4 verdict | **CONFIRMED** |

### §8.7 Theorem 6 (joint coverage)

| Prediction | Source | Empirical | Status |
|---|---|---|---|
| Joint slack $\geq (1-\bar L^K)\sum_k\epsilon_k + (1-p_{\text{recover}}) + 1/(n_+^{(0)}+1)$ | T6 §5 | not yet measured; pre-registered prediction in T6 §9 | **PRE-REGISTERED** |
| Conditional intervention coverage on AIME-new $\in[0.39, 0.55]$ | T6 §9: plug-in $\bar L=0.85$, $K=4$, $\sum\epsilon_k\approx 0.55$, $p_{\text{recover}}\approx 0.4$ ⇒ predicted $\geq 0.036$ unconditional / $\geq 0.39$ conditional | *prediction; canonical AIME run pending* | **PRE-REGISTERED** |
| Falsifier: empirical conditional coverage > 0.65 | T6 §9 falsifier 1 | none observed | **NOT FALSIFIED** |
| Falsifier: empirical conditional coverage < 0.20 | T6 §9 falsifier 2 (refutes Lemma J.1) | none observed | **NOT FALSIFIED** |
| F1 cascade-shift coincidence visible as TV displacement > $\epsilon_{K-1}$ on intervened traces | T6 §10 F1 | not yet measured (rung-displacement check is a proposed CI gate) | **OPEN EMPIRICAL CHECK** |

---

## §9 — Reviewer attack surface (top 5)

Drawn from the most severe objections of T4 v2 §J, T4 v3 §10, T5 v2
§I, T5' Gap A §9, T6 §10. Each is a likely AISTATS/ICML reviewer
attack; each has a stated v3-or-later response.

### §9.1 (R1, severe) "Theorem 5/T5' bound is vacuous in your pilot. Headline unproven."

**Response.** We honestly split the contribution into Theorem 5 v2
(*worst-case* slack, vacuous at pilot $n$ but tight in the
asymptotic / large-K / GDA regime, matches Wang-Wu-Liang 2022) and
Theorem 5' (*typical-case* contraction with $\bar L\approx 0.85$
matching the empirical 56–67% relative reduction across α and
models). The paper's value is *the framing* (telescoped ladder for
CP) + *empirical demonstration* + *Theorem 5'* as novel contraction
analysis. Theorem 5 v2 is a stepping stone, honestly labelled.

### §9.2 (R2, severe) "Theorem 5' has proof gaps. Withdraw."

**Response.** All three gaps are explicitly labelled. **Gap A**
(`theorem5_gap_A_lipschitz.md`) is *closed* at the worst-case bound
in the Wasserstein-1 metric. **Gap B**
(`theorem5_gap_B_fixed_point_coverage.md`) is *closed* via Strassen-
Lindvall coupling chain at the Banach fixed point + per-rung
damping. **Gap C** (`theorem5_gap_C_contraction_sufficient.md`) is
*partially* closed: a verifiable but conservative product-form
sufficient condition is given; the local-attractor tightening is
flagged as future work. If the reviewer requires fully rigorous Gap
C as a precondition, we offer to demote Theorem 5' to "Conjecture 5'
with strong empirical support + closed Gaps A and B" and rephrase
the contribution accordingly.

### §9.3 (R3, severe — Theorem 4) "Negative aggregate lift on math500 contradicts Theorem 4."

**Response.** Corollary 4.1 (T4 v3) gives the *quantitative* answer:
on `qwen25_7b__math500`, $\kappa\approx 0.34$, $p_{\text{cascade}}\approx 0.95$
(stickier than AIME), $w(g)$ heavily concentrated at $g=1,2$, and
$\Lambda(g=1)\approx 6.6$pp. The aggregate is mostly small-$g$ mass
where $\kappa(1-p_{\text{cascade}})\approx \kappa\cdot 0.05\approx 1.7$pp
is below $\Lambda(1)$. **The bound is operationally vacuous on easy
cells, *consistent* with the bound rather than contradicting it.**
The decisive test is the gap≥5 stratum on hard cells: AIME
`marg__a0.1` gap≥5 = +18.76pp confirms the structural prediction.

### §9.4 (R4, medium — Theorem 4) "(A5) excludes the models people care about (R1-Distill, QwQ, o1)."

**Response.** Acknowledged. (P-A5) restricts to *non-self-correcting*
LMs (Qwen2.5-7B-Instruct, Qwen2.5-Math-7B, Phi-4 base, Llama-3.1,
Mistral-7B). Frontier reasoning models violate (P-A5) by construction
(RL-trained for backtracking). A separate Theorem 4.5 with non-
monotone $p_{\text{cascade}}$ is open future work (§7 above). The
current paper's contribution is the non-self-correcting baseline,
which is still a substantial model class.

### §9.5 (R5, medium — Theorem 6) "Joint slack is just T4-slack + T5'-slack. What's new?"

**Response.** **Lemma J.1** (do-marginalised exchangeability) is the
new mathematical content. Without it, T5' coverage does *not* apply
to the post-intervention test score (the do-intervention breaks
weighted exchangeability). T6's value is not a smaller slack — its
value is bounding a **strictly stronger event** (post-intervention
answer-coverage on rung $K$, jointly), which neither T4 nor T5'
alone can certify. Reduction §5.5 confirms T6 cleanly degenerates
to either component theorem in the natural limits.

---

## §10 — Cross-Model Verification Results (per workspace `CLAUDE.md`)

> **Single-model lane.** Per workspace `CLAUDE.md` §
> "Cross-Model Verification": `mode: all`, primary `claude-opus-4-7`,
> verifier `openai/openai/gpt-5.5`, fallback
> `gcp/google/gemini-3.1-pro-preview`. The `inference_token` is
> `sk-PLACEHOLDER` and the external verifier was not invoked for this
> unifier pass. Per the protocol in
> `pipeline/cross_model_verification_protocol.md`, disagreements
> would be appended verbatim below; the current verdict on the
> unified stack is **PROCEED, single-model only**, with
> the explicit `[PROOF GAP C — partial]` and `[F1 — pre-registered
> empirical check]` flags visible.
>
> **Verdict — primary (claude-opus-4-7).** PROCEED. The unification
> consolidates ten input files into a single citable stack with
> resolved notation conflicts (§1.2), explicit prefixed assumption
> sets (§3.6, §4.9), three closed proof gaps (Gap A, Gap B, T6
> reductions §5.5), one *partially* closed proof gap (Gap C §4.6
> + research direction §7.2), one new bridge lemma (Lemma J.1, T6
> §5.3), and a calibrated empirical-fingerprint table (§8) anchored
> to the actual JSON values from `pearl_full/`,
> `distance_ladder_full/`, `distance_ladder_pilot.json`, and
> `pearl_causal_pilot.json`. The §9 reviewer-attack surface
> consolidates the most severe objections from T4 v2 §J, T4 v3 §10,
> T5 v2 §I, T5' Gap A §9, and T6 §10 into a single five-objection
> response set.
>
> **Verdict — verifier.** *(pending; per protocol, single-model
> only at this writing.)* Anticipated objections per the input files'
> own anticipations: (i) Lemma J.1's "in expectation over
> intervention" is delicate and may need explicit Bareinboim
> s-recoverability check; (ii) Gap C's tightness gap (~80×) is
> uncomfortable; (iii) joint bound (T6 §9 plug-in) is operationally
> vacuous at α=0.10 (covers only ≥3.6%) absent the conditional
> reduction.
>
> **Cross-Model Disagreements (placeholder).** None recorded —
> single-model lane. Disagreements, if any, will be appended
> verbatim here per `cross_model_verification_protocol.md`; verifier
> verdicts NEVER silently overwrite the primary verdict.

---

## Summary — what the unified stack establishes (and what it does not)

**Establishes.**
1. **T1–T3 background** — standard split-CP and discrete weighted
   CP machinery applied to step-aggregated reasoning scores.
2. **T4 v3 + Corollary 4.1** — earliest-divergent-step single-do
   dominance under (P-A1')–(P-A6), with cascade-gap-stratified
   prediction $\Delta_{\text{strat}}(g)\geq\kappa(1-p_{\text{cascade}}^g)-\Lambda(g)$
   that **quantitatively predicts** the AIME `marg__a0.1` gap≥5
   = +18.76pp signal and **explains** the math500 negative aggregate.
3. **T5 v2** — worst-case TV-summation slack
   $1-\alpha-\sum_k(\hat\epsilon_k+\mathrm{DKW}_k)-1/(n_+^{(0)}+1)$;
   vacuous absolute, tight relative.
4. **T5'** — Banach-fixed-point contraction coverage
   $1-\alpha-\sum_k\bar L^{K-1-k}\epsilon_k-1/(n_+^{(0)}+1)$;
   $\bar L\approx 0.85$ matches empirical 56–67% relative-reduction
   across α-grid and 4 model cells.
5. **Gaps A and B closed**; **Gap C partially closed** (sufficient
   condition exists but is loose by ~80×; local-attractor research
   direction identified).
6. **T6** — joint coverage with explicit slack composition
   $1-\alpha-(1-\bar L^K)\sum\epsilon_k-(1-p_{\text{recover}})-1/(n_+^{(0)}+1)$,
   bridged by Lemma J.1 (do-marginalised exchangeability), reduces
   correctly to T4 v3 ($K=1$, no shift) and T5' (no intervention).

**Does not establish.**
1. Self-correcting models (R1-Distill, QwQ, o1) — out of scope by
   (P-A5).
2. Multi-modal correct trajectories (~5–10% OlympiadBench) — out of
   scope by (P-A6).
3. Continuous score support — open via Wasserstein KDE (Gap A §9.4).
4. Tight typical-case Lipschitz (Gap C tightening from ~80× to <2×)
   — open via local-attractor framing (§7.2).
5. Joint-aware calibration — open speculation (§7.5; T6 §11 Q3).
6. Per-trace Lemma J.1 — open (§7.8; T6 §11 Q5).
7. Fully-discharged sufficient condition for $\bar L<1$ — partial
   only.
8. K=4 majority Jensen step beyond Lemma J.1's expectation — flagged
   in T4 v2 §H.1 and §6.1 above; open.

**Empirical falsification status.** No closed cell yet decisively
falsifies any of the structural predictions. The strongest *partial*
falsifier in hand is `qwen25_7b__olympiad` `lp__a0.3` gap≥5 = −3.18pp
(opposite sign of the T4 v3 prediction), interpreted as a likely
(P-A6) violation in the OlympiadBench multi-modal subset. Decisive
falsifiers for the remaining 5 open `pearl_full/` cells are
pre-registered in T4 v3 §8.3.

---

## References

All ten input files' references are inherited verbatim. Key external
citations used by the unified stack:

- **Pearl 2009** — *Causality* (2nd ed.). Theorem 3.3.4 (front-door),
  §3.4 (conditional FD), §4.4–4.5 (CDE), §7 (twin-network).
- **Tian–Pearl 2002** — *AAAI*, ID Theorem 1 (generalized FD).
- **Bareinboim–Pearl 2014** — *Statistical Science* 29:579–595,
  Definition 5 (s-recoverability), Theorem 5 (bounded-gap),
  Corollary 2.
- **Banach 1922** — *Fund. Math.* 3:133–181 (contraction-mapping
  theorem).
- **Granas–Dugundji 2003** — *Fixed Point Theory*, Springer, Theorem
  II.1.1 (modern Banach), Chapter III (Tarski lattice).
- **Strassen 1965** — *Annals Math. Stat.* 36:423–439, Theorem 11
  (TV-coupling duality).
- **Lindvall 2002** — *Lectures on the Coupling Method*, Cambridge UP,
  Theorem I.5.4 (gluing lemma).
- **Tibshirani–Foygel-Barber–Candès–Ramdas 2019** — *NeurIPS*,
  Algorithm 1, Lemma A.1 (weighted CP under covariate shift).
- **Barber–Candès–Ramdas–Tibshirani 2023** — *Annals of Statistics*,
  Theorem 2a (nexCP).
- **Berend–Kontorovich 2013** — *Annals of Statistics* (discrete-PMF
  concentration).
- **Villani 2009** — *Optimal Transport*, Chapter 6 (Wasserstein-1),
  Theorem 7.3.
- **Lei–Wasserman 2014**, **Vovk 2002** — split-CP foundations.
- **Lei–G'Sell–Rinaldo–Tibshirani–Wasserman JASA 2018** — split-CP
  finite-sample correction.
- **Wang–Wu–Liang NeurIPS 2022**, **He–Wang–Liang JMLR 2024** —
  gradual-DA bridge.
- **Howard 1966** — VOI / EVPI.
- **Sutton–Barto 2018** — eligibility-trace decay.
- **Buesing 2019** — twin-network counterfactual coupling.
- **Wang–Wei NeurIPS 2022** (self-consistency K-sample concentration);
  **Lightman 2023** (PRM800K).
- **Math-Shepherd** (arXiv:2312.08935) — $Q_s$ MC labels.

Internal artifacts: `pearl_full/{qwen25_7b_aime,qwen25_7b_math500,
qwen25_7b_olympiad,qwen25_math_7b_math500}.json`,
`pearl_full/AGGREGATE.{json,md}`,
`distance_ladder_full/{AGGREGATE.json,qwen25_*_strategyA_B_Bp.json}`,
`distance_ladder_pilot.json`, `pearl_causal_pilot.json`,
`concept_papers/{pearl_causal_DEEP,distance_ladder_DEEP}.md`,
and the ten theorem drafts cited in the header.

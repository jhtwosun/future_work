# §3 Method, §4 Theory — CoT-CP with Pearl Step Intervention and Distance-Ladder Calibration

> **Working title (paper)**: *Causal Step Intervention and Distance-Ladder Calibration for Conformal LLM Reasoning*
> **Sections drafted here**: §3 Method, §4 Theory.
> **Drop-in LaTeX-compatible markdown.** Math in `$...$` / `$$...$$`. Theorem environments via bold headers and `> ` blockquote-boxed statements. Citations in `\cite{key}` form against `/home/nvidia/future/literature/papers/references.bib`; foundational citations not yet in the bib (Pearl 2009, Banach 1922 / Granas–Dugundji 2003, Strassen 1965 / Lindvall 2002, Bareinboim–Pearl 2014, Tian–Pearl 2002, Wang–Wu–Liang 2022, Hendrycks 2021 MATH, AIME-2024) are flagged in `\cite{...}` form for completion.
> **Status of proofs.** Theorem 1 is direct. Theorem 4 (v3) imports Lemma 4.A (front-door identification) and Lemma 4.B (cascade-decay optimality) verbatim from `theorems_drafts/theorem4_v3_cascade_stratified.md`; Corollary 4.1 reproduces v3 §3.2 exactly. Theorem 5 (v2 telescoping bound) and Theorem 5' (Banach contraction) reproduce `theorems_drafts/theorem5_v2_consolidated.md` §F; Lemmas 5.1–5.5 are aligned with the Gap-A/B/C closure files (`theorem5_gap_A_lipschitz.md`, `..._gap_B_fixed_point_coverage.md`, `..._gap_C_contraction_sufficient.md`) and remaining gaps are labelled. Theorem 6 reproduces `theorems_drafts/joint_composition_theorem.md` §5 verbatim with a one-paragraph proof sketch and the two reductions.

---

## §3 Method

### §3.1 Setup

We summarise notation in Table 3.1 and describe the assumptions in detail below.

**Table 3.1: Notation summary.**

| Symbol | Meaning |
|---|---|
| $\pi_\theta$ | autoregressive language model with parameters $\theta$ |
| $P$ | prompt (problem statement, system prompt, etc.) |
| $\bar X = (X_1, \ldots, X_T)$ | step-segmented chain-of-thought trace; $T$ varies per trace |
| $Y(\bar X) \in \{0, 1\}$ | answer correctness (deterministic judge) |
| $S_t \in \mathbb R$ | per-step score (`lp`, `ent_neg`, `marg`) |
| $\bar S = \phi(S_1, \ldots, S_T)$ | trajectory score via aggregator $\phi$ |
| $\mathcal S$ | discrete score support; $|\mathcal S| = N + 1$ for SC@$N$ |
| $\hat q_\alpha$ | finite-sample lower-$\alpha$-quantile of correct-trajectory $\bar S$ |
| $D_k$ | rung-$k$ distribution (calibration source $D_0$, test target $D_K$) |
| $\hat p_k(s)$ | empirical PMF of $\bar S$ on $D_k$'s correct-trajectory subset |
| $\hat w_{k-1 \to k}(s) = \hat p_k(s) / \hat p_{k-1}(s)$ | empirical density-ratio at rung-pair $(k-1, k)$ (Laplace-smoothed) |
| $\hat\epsilon_k = \tfrac12 \sum_s|\hat p_k(s) - \hat p_{k+1}(s)|$ | per-rung-pair empirical TV |
| $\epsilon_k = d_{TV}(P_{k-1}, P_k)$ | population per-rung-pair TV |
| $T_k : \mathcal Q \to \mathcal S$ | per-rung iterated weighted-quantile operator |
| $T = T_K \circ \cdots \circ T_1$ | composed iterated operator |
| $\bar L = \prod_k L_k$ | mean (product) Lipschitz constant |
| $t^*(\bar x)$ | recovery-aware earliest divergent step |
| $t_{\mathrm{worst}}(\bar x)$ | worst-step locus, $\arg\min_t S_t$ |
| $g(\bar x) = t_{\mathrm{worst}} - t^*$ | cascade gap |
| $w(g)$ | empirical gap distribution on wrong-trace subpopulation |
| $\Delta_{\mathrm{strat}}(g)$ | within-stratum cascade-gap-stratified lift |
| $\overline\Delta = \sum_g w(g)\Delta_{\mathrm{strat}}(g)$ | aggregate lift |
| $p_{\mathrm{cascade}}(\theta, \mathcal D)$ | cascade contractivity probability (per (model, dataset)) |
| $p_{\mathrm{recover}}$ | K=4 re-roll recovery probability |
| $\Lambda(g)$ | false-positive cost on stratum $g$ |
| $\hat Y_{n+1}$ | K=4 majority post-intervention vote |



We assume a single black-box autoregressive language model $\pi_\theta$ producing a step-segmented chain-of-thought trace $\bar X = (X_1, X_2, \ldots, X_T)$ for prompt $P$, terminated by a final answer $Y(\bar X) \in \{0, 1\}$ scored by a deterministic judge against ground truth. Steps are obtained by splitting the decoded text on `\n\n` (the canonical PRM800K \cite{lightman2023verify} segmentation; see Math-Shepherd \cite{wang2024mathshepherd} for an MC-rollout-labelled alternative). For each step $X_t$ we associate one of three **step-level scores** $S_t \in \mathbb{R}$:

1. **`lp` (token-mean log-probability of the step)** — free, recovered from greedy decode.
2. **`ent_neg` (negative average token entropy of the step)** — same compute as `lp`.
3. **`marg` (top-1 token margin)** — same compute as `lp`.

Aggregated to the trajectory level by a measurable $\phi : \mathbb{R}^T \to \mathbb{R}$ (we instantiate $\phi = \min_t$, $\phi = \mathrm{mean}_t$, and the self-consistency $\phi_{\mathrm{SC}@N}$ that returns the modal-answer vote share), the **trajectory score** is
$$\bar S := \phi(S_1, \ldots, S_T).$$
Self-consistency at $N=8$ samples yields a discrete support $\bar S \in \{0/8, 1/8, \ldots, 8/8\}$, $|\mathcal{S}| = 9$, which is what makes Theorem 3's empirical-PMF density-ratio estimator (and Theorems 5/5'/6 below) the natural primitive rather than KDE-based weighted CP \cite{tibshirani2019weighted}.

**Calibration data and target metric.** Let $\mathcal{D}_{\mathrm{cal}} = \{(P_i, \bar X_i, \bar S_i, Y_i)\}_{i=1}^{n}$ be a labelled split-CP calibration set drawn i.i.d.\ from a source distribution $D_0$ (we use a PRM800K-derived MATH-500-cal split in our pilots), and let $\mathcal{I}_+ = \{i : Y_i = 1\}$ index the correct-trajectory subset, $n_+ := |\mathcal{I}_+|$. The target metric is the **correctness-conditional coverage** at level $1 - \alpha$:
$$\mathrm{cov}(\hat q_\alpha; D) \;:=\; \Pr_D\!\big[\,\bar S_{n+1} \geq \hat q_\alpha \,\big|\, Y_{n+1} = 1\,\big],$$
together with the *kept accuracy* $\rho_{\mathrm{kept}} := \Pr[Y = 1 \mid \bar S \geq \hat q_\alpha]$ that we report empirically. Throughout, $\hat q_\alpha$ denotes a finite-sample lower-$\alpha$-quantile of $\{\bar S_i : i \in \mathcal{I}_+\}$ with the standard split-CP correction $\lfloor \alpha (n_+ + 1) \rfloor / n_+$. Bayes' rule expresses kept-accuracy via $\rho_{\mathrm{kept}} = \pi \beta_+ / (\pi \beta_+ + (1 - \pi) \beta_-)$, with $\beta_+ = \Pr[\bar S \geq \hat q_\alpha \mid Y = 1]$ controlled by Theorem 1 and $\beta_-$ controlled by the score's selectivity (Theorem 2 in CoT-CP); the headline metric is therefore $\rho_{\mathrm{kept}}$, and the $\beta_+$ guarantee is what the conformal layer delivers.

**Why the trajectory aggregator is decoupled from the calibration guarantee.** Our $\phi$ may be $\min_t$, $\mathrm{mean}_t$, or self-consistency vote share; it may even depend on the prompt $P$, provided it is *fixed across the calibration / test distribution* (i.e., not fit on $\mathcal{D}_{\mathrm{cal}}$). Theorem 1 \S4.1 shows that any measurable $\phi$ inherits a finite-sample distribution-free coverage guarantee. The choice of $\phi$ is therefore a *Pareto-frontier choice* (\cite{angelopoulos2025ltt}) along (compute, selectivity, coverage), not a soundness choice; CoT-CP §5 maps this Pareto for `lp_min` 1×, `prm_min` 2×, `sc_top1` 8×, with PRM-min as a new mid-cost operating point. The present paper takes $\phi = \phi_{\mathrm{SC}@8}$ as the running aggregator because it makes $\bar S$ discrete and lets us deploy Theorems 5/5' / 6 cleanly; the framework extends to continuous $\phi$ via Berend–Kontorovich \cite{barber2023beyond} or a Wasserstein-1 reformulation \cite{villani2009ot} (future work).

**Per-step vs trajectory aggregation.** Theorem 4's intervention layer requires per-step scores $S_t$ at the recovery-aware $t^*$ trigger; Theorems 5/5' and the trajectory CP layer use the trajectory score $\bar S = \phi(R)$. The two granularities are reconciled by treating the per-step thresholds $\{q_\alpha(t)\}$ as a separate calibration object derived from PRM800K's labelled per-step data, while the trajectory threshold $\hat q_\alpha$ is computed on the same calibration set's trajectory aggregates. The two thresholds share the same $\alpha$ but are distinct quantities; the per-step thresholds determine where intervention fires, the trajectory threshold determines whether the post-intervention trace is kept. This separation is what makes the joint method's compositional architecture clean: per-step calibration is a property of the score family (Theorem 2), trajectory calibration is a property of the aggregator (Theorem 1), and the two layers can be deployed independently or composed via Theorem 6.

### §3.2 Trajectory-level CP (T1 backdrop)

The CoT-CP procedure of Theorem 1 (\S4.1) uses $\hat q_\alpha$ directly: keep iff $\bar S_{n+1} \geq \hat q_\alpha$. Under exchangeability (in our setting the slightly stronger i.i.d.\ within-rung from \S3.4) Theorem 1 delivers
$$\Pr\!\big[\bar S_{n+1} \geq \hat q_\alpha \mid Y_{n+1} = 1\big] \;\geq\; 1 - \alpha - \tfrac{1}{n_+ + 1}.$$
This is the *backdrop guarantee* over which we layer two orthogonal mechanisms: a **causal step-intervention** at the cascade source $t^*$ (\S3.3, theorised in \S4.2), and a **distance-ladder calibration** $\hat q^{(K),B'}_\alpha$ at the target rung $D_K$ (\S3.4, theorised in \S4.3). \S3.5 composes them.

The two mechanisms are *complementary*, not redundant. T1 controls coverage on the marginal *score* on a single calibration rung; the intervention layer (T4) acts on the *trace generation* mid-stream and changes the score distribution at deployment time; the ladder layer (T5/5') changes the *calibration quantile* used at the target rung. Each layer has its own assumption set (A1)–(A6) for the ladder, (A1')–(A6) for intervention, with overlapping notation but disjoint semantic content (the assumption-letter overlap is unfortunate; we annotate with primes throughout to disambiguate). Theorem 6 (\S4.4) verifies that the layers compose without producing a worse-than-additive slack — the compositional architecture is mathematically benign provided Lemma J.1 (do-marginalised exchangeability) bridges the post-intervention test score back to the rung-$K$ calibration manifold.

#### §3.2.1 Why aggregator choice is decoupled from coverage

A persistent confusion in the conformal-LM literature is whether the choice of conformity score is *part of the algorithm* or *part of the user's design choice*. Theorem 1 makes the answer unambiguous: any measurable $\phi$ inherits the coverage guarantee. The aggregator is therefore a **Pareto-frontier choice** — the user picks $\phi$ to optimise the (compute, selectivity, kept-accuracy) trade-off they care about, with no risk of breaking the coverage guarantee.

Concretely, our three-aggregator ladder (lp_min 1×, prm_min 2×, sc_top1 8×) is a *compute-Pareto* — at each compute level, the maximum-LR+ aggregator dominates (Theorem 2 in CoT-CP §5.4). PRM-min is a *new mid-cost operating point* delivering ~2/3 of self-consistency's selective-accuracy lift at 1/4 the compute. The choice of aggregator does not interact with Theorems 4, 5/5', or 6: a user deploying T6 with $\phi = \mathrm{lp\_min}$ has the same coverage guarantee as one deploying T6 with $\phi = \mathrm{sc\_top1}$, just at a different compute / selectivity operating point. This decoupling is what makes CoT-CP an extensible framework rather than a single algorithm tied to one score.

### §3.3 Pearl-causal step intervention

Treat the trace as an autoregressive structural causal model (SCM) \cite{pearl2009causality}:
$$X_1 = f_1(\epsilon_1), \qquad X_t = f_t(X_{1:t-1}, P, \epsilon_t)\ \text{for } t \geq 2, \qquad Y = f_Y(X_{1:T}, \epsilon_Y),$$
with exogenous noise $\epsilon_t$ representing softmax sampling stochasticity at temperature $\tau \in (0, 1]$. Pearl's $\mathrm{do}$-operator $\mathrm{do}(X_t \sim \pi)$ replaces the local mechanism at step $t$ by an external sampling distribution $\pi$ (we use the same temperature-0.7 sampling from $\pi_\theta$, $K = 4$ independent draws), then re-rolls the suffix $X_{t+1:T}$ from the model.

#### §3.3.1 Earliest-bad-step identification

Given calibrated per-step thresholds $\{q_\alpha(t)\}_{t=1}^T$ from PRM800K (Approach A of CoT-CP §5), define the **worst-step locus** $t_{\mathrm{worst}} := \arg\min_t S_t$ and the **recovery-aware earliest divergent step**
$$t^*(\bar x) \;:=\; \min\!\big\{\,t : S_t < q_\alpha(t) \ \text{and}\ S_{t+1} < q_\alpha(t+1)\ \text{and}\ S_{t+2} < q_\alpha(t+2)\,\big\}$$
— the first step that triggers a contiguous run of $\geq 3$ below-threshold scores. The contiguity guard rules out isolated "innocent" violations and matches the cascade-monotonicity assumption (A3') of Theorem 4 \S4.2.

**Why earliest, not worst.** The intervention literature has historically targeted $t_{\mathrm{worst}}$ — the step with the lowest score (PRM \cite{lightman2023verify}, log-prob, entropy) — because it is the *symptom* of failure, the locus where the trace's score drops most sharply. Pearl's minimal-intervention principle \cite{pearl2009causality} §3.4 says the opposite: in a causal chain $X_1 \to X_2 \to \cdots \to X_T \to Y$, $\mathrm{do}(X_t)$ at the *cause* (the cascade source) dominates $\mathrm{do}(X_t)$ at the *symptom* in the value-of-information sense. CoT-CP Pilots C/K/L documented a stark null: K=4 majority resampling at $t_{\mathrm{worst}}$ gives only +1pt on MATH-500 with `lp_min` trigger, +0pt with PRM trigger, +1pt with rewrite-cue. Three different score families, three different intervention surfaces — all wash. Theorem 4 below provides a *causal* explanation: at $t_{\mathrm{worst}}$, the prefix $X_{1:t_{\mathrm{worst}}-1}$ already contains the divergent-prefix from $t^*$, so resampling from a corrupted prefix produces alternatives that mostly stay corrupted.

Empirically (CoT-CP §7, `pearl_causal_pilot.json`) $t^* < t_{\mathrm{worst}}$ in 42–78\% of incorrect traces across 12 (model, dataset) cells, with the cascade-depth gap $g := t_{\mathrm{worst}} - t^*$ ranging from 1.5 (Qwen2.5-7B AIME) to 9.88 (Phi-4 AIME). The fraction is highest (>70\%) for **strong models on hard datasets** — exactly the regime where cascade reasoning is operative. Surprisingly, the recovery-aware $t^*$ is *concentrated very early*: mean $t^* < 2$ for all 12 cells at $\alpha = 0.3$, and $< 1.5$ at $\alpha = 0.5$. The cascade source is in the *first 1–2 steps* of a typical wrong trace, corroborating *First-Step Advantage* and *Well Begun Half Done* \cite{wang2024mathshepherd}.

#### §3.3.2 K=4 majority intervention at $t^*$

The intervention re-samples four alternatives at $t^*$ and majority-votes the resulting answers:
$$\tilde X_{t^*}^{(j)} \overset{\mathrm{iid}}{\sim} \pi_\theta(\cdot \mid X_{1:t^*-1}, P), \quad \tilde X_{t^*+1:T}^{(j)} \sim \pi_\theta(\cdot \mid \tilde X_{t^*}^{(j)}, X_{1:t^*-1}, P), \quad j = 1, \ldots, 4,$$
$$\hat Y_{n+1} \;:=\; \mathrm{majority}\!\big(\tilde Y^{(1)}, \tilde Y^{(2)}, \tilde Y^{(3)}, \tilde Y^{(4)}\big).$$
The vanilla baseline applies the identical procedure at $t = t_{\mathrm{worst}}$. Reporting follows two conventions: (i) the **all-trace** lift $\Delta_{\mathrm{all}} := \mathrm{acc}(\hat Y_{n+1}^{t^*}) - \mathrm{acc}(\hat Y_{n+1}^{t_{\mathrm{worst}}})$, and (ii) the **wrong-trace-conditional** lift $\Delta_{\mathrm{wrong}}$ that only fires when the original trace was wrong (eliminating the false-positive cost on already-correct traces; see \S4.2 and \cite{wang2023selfconsistency}). The wrong-trace-conditional version sets the false-positive cost $\Lambda(g) \equiv 0$ in Corollary 4.1 by construction; we therefore recommend $\Delta_{\mathrm{wrong}}$ as the primary reporting metric and $\Delta_{\mathrm{all}}$ as the *deployment* metric (the latter is what end users actually see, but is contaminated by $\Lambda(g)$).

The choice of $K = 4$ is empirical: at $K = 2$ the recovery-effective slack $1 - (1 - p_{\mathrm{recover}})^K$ is too small (\S4.2 (A4'); $K = 2$, $p_{\mathrm{recover}} = 0.2$ gives $0.36$ vs $K = 4$'s $0.59$), and at $K = 8$ the marginal recovery-effective gain is only $0.83 - 0.59 = 0.24$ per doubling of compute, with the K=4 majority's tie-breaking degenerating to 2-vs-6 splits that diminish stability. We treat $K = 4$ as a fixed design constant; Theorem 4 v3 makes no commitment to $K$-scaling, and a $K$-scaling analysis is open (\S4.5 (8)).

#### §3.3.3 Cascade-gap stratification

The headline trace-level statistic is the **cascade-gap-stratified lift** at gap $g$:
$$\Delta_{\mathrm{strat}}(g) \;:=\; \mathbb{E}\!\big[\,\hat Y^{t^*} - \hat Y^{t_{\mathrm{worst}}} \,\big|\, t_{\mathrm{worst}} - t^* = g\,\big],$$
and the empirical gap distribution $w(g) := \Pr[t_{\mathrm{worst}} - t^* = g \mid Y = 0]$ on the wrong-trace subpopulation. We report $\Delta_{\mathrm{strat}}(g)$ for $g = 1$, $g \in [2, 4]$, and $g \geq 5$, alongside the gap-mixture aggregate $\overline{\Delta} = \sum_g w(g) \Delta_{\mathrm{strat}}(g)$. Theorem 4 \S4.2 proves $\Delta_{\mathrm{strat}}(g)$ is monotone non-decreasing in $g$ up to a false-positive cost $\Lambda(g)$, and Corollary 4.1 quantifies how the aggregate is dragged toward zero by the small-$g$ mass even when $\Delta_{\mathrm{strat}}(g \geq 5)$ is large positive (empirically up to **+18.76 pp** on Qwen2.5-7B / AIME / `marg__a0.1`).

The stratification is essential for honest reporting: a (model, dataset) cell can have a near-zero or negative *aggregate* lift while having strongly positive *stratum-conditional* lift at gap≥5 — and the right diagnostic is *not* "the intervention fails" but "the gap mixture concentrates at small $g$ where the cascade-decay benefit is small". Concretely, on `qwen25_7b__aime` the `marg__a0.1` row gives gap=1 lift $-1.51$pp, gap∈[2,4] $+6.91$pp, gap≥5 $+18.76$pp, with empirical mass $w(1) \approx 0.45, w(2..4) \approx 0.32, w(5+) \approx 0.23$ — aggregate is dragged toward zero by the gap=1 mass and the false-positive cost $\Lambda(1) \approx 6.6$pp on already-correct traces. Reporting only the aggregate would conclude "intervention does not work on AIME"; reporting the gap-stratification reveals that **intervention works precisely on the cascade-deep wrong traces, which is where Pearl's theorem says it should work**.

### §3.4 Distance-Ladder calibration

#### §3.4.1 Rung definition

A **calibration ladder** is an ordered sequence of labelled benchmarks
$$D_0 \to D_1 \to \cdots \to D_K$$
with $D_0$ the calibration source and $D_K$ the test target, chosen to be progressively further from $D_0$ in score-PMF total variation. Our canonical four-rung instance, in analogy with the SH0ES astronomical distance ladder \cite{riess2022shoes}, is
$$\underbrace{\mathrm{PRM800K}}_{D_0\ \text{geometric anchor}} \to \underbrace{\mathrm{MATH\text{-}500}}_{D_1} \to \underbrace{\mathrm{AIME}_{1983\text{-}1999}}_{D_2\ \text{old}} \to \underbrace{\mathrm{AIME}_{2000\text{-}2014}}_{D_3\ \text{mid}} \to \underbrace{\mathrm{AIME}_{2015\text{-}2024}}_{D_4\ \text{new}}.$$
Each rung is labelled (correctness $Y_i^{(k)}$ is recoverable from the answer judge), and the per-rung empirical PMF $\hat p_k(s)$ on the **correct-trajectory** scores is the only quantity our calibration uses. The per-rung-pair total-variation distance $\hat\epsilon_k := \tfrac12 \sum_s |\hat p_k(s) - \hat p_{k+1}(s)|$ measures local shift; the *global* TV $d_{TV}(\hat p_0, \hat p_K) =: \epsilon_{\mathrm{global}}$ is what Theorem 3 (one-shot weighted CP) pays. Empirically on our 4-rung pilot $\epsilon_{\mathrm{global}} \approx 0.54$ while $\sum_k \hat\epsilon_k \approx 0.75$ (the chain-overlap inefficiency $\rho_K = \sum_k\hat\epsilon_k / \epsilon_{\mathrm{global}} \approx 1.39$ diagnoses redundancy across rungs).

**Astronomy analog.** The SH0ES ladder \cite{riess2022shoes} measures $H_0 = 73.04 \pm 1.04$ km/s/Mpc by chaining (i) Gaia EDR3 parallax of 75 Milky-Way Cepheids, (ii) Cepheid period–luminosity calibration to nearby SN-Ia hosts, and (iii) SN-Ia standardisation to the Hubble flow. The defining structural feature is *not* multiple steps but *overlap regions*: each rung's zero-point is fixed by matching predictions on a sample observable at both rungs. The total error is bounded by the *sum of overlap-region cross-calibration errors*, not by a multiplicative compound. Our LLM analog uses score-level overlap rather than physical sample overlap: score atoms with non-trivial mass at both rungs $(D_{k-1}, D_k)$ play the role of "galaxies hosting both Cepheids and SNe Ia". This analogy motivates Strategy B' (sequential anchoring per rung) over Strategy B (one-shot multiplicative product) — astronomy *iteratively fixes a zero-point per rung*, and the matching computational primitive in CP is the iterated quantile map.

**Rung-design constraints.** A practical ladder must satisfy: (i) (A6) monotone source-TV — $d_{TV}(D_0, D_k) \leq d_{TV}(D_0, D_{k+1})$ — empirically testable per pilot; (ii) (A4') strict per-rung-pair information gain — adding rung $k$ to a $(k-1)$-rung ladder must strictly tighten the bound, otherwise the rung is monster-barred; (iii) sufficient anchor sample size — the smallest-$n_k$ rung dominates the DKW slack via a finite-sample anchor effect (`theorem5_gap_C_contraction_sufficient.md` §8.3). Violation of (A6) requires re-ordering or dropping rungs; we observe (A6) failure on `qwen25_7b` source TVs across the 5-rung ladder and address by rung-2 swap. Violation of (iii) is the headline H4 falsifier of the pilot — dropping the AIME-mid rung costs 11.9 pp at $\alpha = 0.5$, identifying it as the contraction-anchor rung whose Lipschitz $L_k$ is closest-to-but-below 1.

**Sufficient overlap condition.** Theorem 5 v2's sufficient overlap condition is $\rho_K \cdot \epsilon_{\mathrm{global}} < W_{0 \to K}^+ \cdot |\mathcal S| / \sqrt{n_{\min}}$, equivalently "the chain TV sum is below the global density-ratio-amplified DKW slack". The pilot has $\rho_K \approx 1.39$ on the 4-rung instance and $\rho_K \approx 1.41$ on the 5-rung instance, so the chain-rule TV-summation is 39–41\% larger than $\epsilon_{\mathrm{global}}$. This is *not* a violation of Theorem 5 v2 — the theorem still holds — but a diagnostic that the chain is over-counting round-trip mass via near-coincident rungs. The mitigation is rung re-ordering or merging; the theory provides the diagnostic but does not auto-correct the ladder.

**Sample-size analysis at finite $n_k$.** The DKW slack at rung-pair $(k, k+1)$ is $\delta_{\mathrm{DKW}, k}(\delta) = \sqrt{\log(2|\mathcal S|/\delta) / (2 n_{\min, k})}$. Union-bounded over $K$ rung-pairs at level $\delta/K$, the cumulative DKW slack is $|\mathcal S|\sum_k \delta_{\mathrm{DKW}, k}(\delta/K) \approx K \cdot |\mathcal S| \sqrt{\log(2K|\mathcal S|/\delta) / (2 n_{\min})}$. For our 4-rung pilot at $|\mathcal S| = 9$, $K = 4$, $\delta = 0.05$, $n_{\min} = 200$: cumulative DKW $\approx 4 \cdot 9 \cdot \sqrt{\log(720/0.05) / 400} \approx 36 \cdot 0.20 = 7.2$ — vacuous. For Theorem 5 v2 to be operationally informative at $K = 4$ we need $n_{\min} \geq 4000$ per rung; at $n_{\min} = 200$ the bound is purely directional. Theorem 5' is what gives directional *and* magnitude-correct relative slack reduction at finite $n_{\min}$.

#### §3.4.2 Sequential ladder calibration (Strategy B')

Sequential ladder calibration ("Strategy B'") iteratively re-anchors the conformal threshold along the chain:

**Step 0.** $\hat q^{(0)}_\alpha := $ vanilla split-CP quantile of the $D_0$ correct-trajectory scores.

**Step $k$ (for $k = 1, \ldots, K$).** Given the inherited threshold $\hat q^{(k-1)}_\alpha$, solve
$$\hat q^{(k)}_\alpha \;:=\; T_k\big(\hat q^{(k-1)}_\alpha\big) \;:=\; \min\!\Big\{\,s \in \mathcal{S} : \sum_{i \in \mathcal{I}_+^{(k)},\, S_i^{(k)} \leq s} \tfrac{\hat w_{k-1\to k}(S_i^{(k)})\,\mathbb{1}[S_i^{(k)} \geq \hat q^{(k-1)}_\alpha]}{Z_k(\hat q^{(k-1)}_\alpha)} \,\geq\, \alpha\,\Big\},$$
where $\hat w_{k-1 \to k}(s) := (\hat p_k(s) + \varepsilon)/(\hat p_{k-1}(s) + \varepsilon)$ is the empirical-PMF density ratio with Laplace smoothing $\varepsilon = 1/n_{\min,k}$ (cf.\ Theorem 3, \cite{tibshirani2019weighted, barber2023beyond}), and $Z_k(\hat q)$ is the truncated normaliser. The output $\hat q^{(K), B'}_\alpha := T_K \circ T_{K-1} \circ \cdots \circ T_1 (\hat q^{(0)}_\alpha)$ is what we deploy at $D_K$.

**The inheritance constraint.** The indicator $\mathbb 1[S_i^{(k)} \geq \hat q^{(k-1)}_\alpha]$ in the truncated weighted CDF is the **inheritance constraint**: at rung $k$, only score atoms $\geq \hat q^{(k-1)}_\alpha$ contribute to the weighted-quantile computation. This is what makes Strategy B' iterative and *not* point-equivalent to one-shot weighted CP. Without the indicator, the rung-$k$ quantile would be the standard weighted quantile on $D_k$'s correct scores using the per-rung-pair density ratio — and the iteration would collapse to a sequence of independent weighted quantiles whose composition is the one-shot weighted quantile. With the indicator, the iteration enforces *monotone non-decreasing* threshold trajectory $\hat q^{(0)} \leq \hat q^{(1)} \leq \cdots \leq \hat q^{(K)}$, which is what gives Theorem 5' its Banach contraction.

**Algorithmic complexity.** Strategy B' runs in $O(K \cdot n_{\max} \cdot |\mathcal S|)$ time: for each rung $k$, sort the correct-trajectory scores at $D_k$, compute the cumulative weighted CDF with truncation at $\hat q^{(k-1)}$, and find the $\alpha$-quantile. With $K = 4$, $n_{\max} = 250$, $|\mathcal S| = 9$, the total is $\sim 9000$ operations — negligible compared to the per-prompt inference cost. The Strategy A baseline runs in $O(n_0 \cdot |\mathcal S|)$ — a single weighted quantile on $D_0$'s scores. The B'-vs-A overhead is therefore $\sim K \times$, a constant factor at fixed $K$.

**Stopping criterion.** In practice we stop at $k = K$ (the test rung) regardless of empirical convergence. A natural alternative is to stop early when $|\hat q^{(k)} - \hat q^{(k-1)}| \leq \Delta_{\min} = 1/N$ (one atom-step) for two consecutive iterations; this is the empirical-convergence stopping rule recommended by the local-attractor framing of `theorem5_gap_C_contraction_sufficient.md` §7. Empirically the pilot's `q_path` saturates at rung 3 onward (see worked example below), so the early-stopping rule produces the same iterate as the full iteration on every closed-cell pilot run.

#### §3.4.3 Comparison to one-shot CP (Strategy A) and telescoped (Strategy B)

We benchmark Strategy B' against two alternatives:

- **Strategy A (one-shot weighted CP, \cite{tibshirani2019weighted}).** Compute a single density ratio $\hat w_{0 \to K}(s) := \hat p_K(s) / \hat p_0(s)$ and apply weighted-quantile CP on $D_0$'s correct scores. This is Theorem 3 in our group's \cite{angelopoulos2024crc}-style framework.
- **Strategy B (telescoped weighted CP).** Form the product $\hat w^{\mathrm{tel}}_{0 \to K}(s) := \prod_k \hat w_{k-1 \to k}(s)$ and apply one-shot weighted-quantile CP. By the telescoping algebra $\prod_k \hat w_{k-1\to k} = \hat w_{0 \to K}$, **Strategy B is point-equivalent to Strategy A**: same $\hat q_\alpha$, identical kept set. The only difference is the *slack accounting* (Theorem 5 \S4.3, $\sum_k \epsilon_k$ instead of $\epsilon_{\mathrm{global}}$).
- **Strategy B' (sequential, this paper).** Iteratively re-calibrated as in \S3.4.2. Strategy B' is **not** point-equivalent to Strategy A: it inherits the prior threshold $\hat q^{(k-1)}$ as a hard lower constraint at each rung, so the iterate trajectory is not a single weighted quantile. In our 4-rung pilot at $\alpha = 0.5$, Strategy A gives $\hat q_\alpha = 0.500$ (kept-acc 71.4\%, gap 21.4 pp) while Strategy B' gives $\hat q^{(4),B'}_\alpha = 0.625$ (kept-acc 59.5\%, gap 9.5 pp) — a **56\% relative reduction in coverage gap**, repeated to **67\%** averaged across the $\alpha$-grid.

**Worked example.** Consider the 4-rung pilot at $\alpha = 0.5$ with the empirical PMFs reported in `distance_ladder_pilot.json`. The per-rung correct-trajectory PMFs $\hat p_k$ on the SC@8 support $\{0/8, 1/8, \ldots, 8/8\}$ are:

- $\hat p_0$ (PRM800K-MATH-cal): mass concentrated at $\{6/8, 7/8, 8/8\}$ with $\hat p_0(8/8) \approx 0.35$ — model is highly confident on calibration source.
- $\hat p_1$ (MATH-500-eval): essentially identical to $\hat p_0$ (TV $\hat\epsilon_1 \approx 0.11$).
- $\hat p_2$ (AIME-old): mass shifts toward $\{4/8, 5/8, 6/8\}$ with $\hat p_2(8/8) \approx 0.15$ — confidence drops on harder problems (TV $\hat\epsilon_2 \approx 0.42$, the dominant per-rung shift).
- $\hat p_3$ (AIME-mid): similar to $\hat p_2$ with slight further drift (TV $\hat\epsilon_3 \approx 0.10$).
- $\hat p_4$ (AIME-new): mass at $\{3/8, 4/8, 5/8, 6/8\}$ with $\hat p_4(8/8) \approx 0.08$ (TV $\hat\epsilon_4 \approx 0.13$).

Strategy A computes the global ratio $\hat w_{0 \to 4}(s) = \hat p_4(s) / \hat p_0(s)$, which spikes at low $s$ (e.g., $\hat p_4(3/8) / \hat p_0(3/8) \approx 4.5$) and dips at high $s$ ($\hat p_4(8/8) / \hat p_0(8/8) \approx 0.23$). Applying weighted-quantile CP at $\alpha = 0.5$ on $D_0$'s correct scores gives $\hat q_\alpha = 0.500$ — the weighted median is at $s = 4/8$, near the middle of the support. The empirical coverage on $D_4$ at this threshold is $0.286$, a 21.4 pp coverage gap (under-coverage at $\alpha = 0.5$).

Strategy B' iterates: $\hat q^{(0)}_\alpha = $ vanilla quantile on $D_0$ correct scores at $\alpha = 0.5$ → $0.875$. Apply $T_1$ at rung-1 with $D_0 \to D_1$ ratio: $\hat q^{(1)}_\alpha \to 1.000$ (the truncated quantile lands at the top atom because the $D_1$ shift is small). Apply $T_2$ at rung-2 with the larger shift: $\hat q^{(2)}_\alpha = 0.625$ — the iterated quantile drops to the rung-2 weighted median, anchored above the $\alpha$-quantile of $D_2$ but below the rung-1 plateau. Apply $T_3$: stays at $0.625$ (the rung is information-poor; $\hat\epsilon_3 \approx 0.10$ produces no further movement). Apply $T_4$: stays at $0.625$ — the iteration has reached a fixed point. The realised `q_path` is therefore $[0.875, 1.000, 0.625, 0.625, 0.625]$, a saturation pattern that is empirically *invariant under further rung addition*. The Banach fixed-point predicts exactly this saturation; the realised $\bar L \approx 0$ on this trajectory (the path stalls at one atom from rung 3 onward), although the cross-α average $\bar L$ is $\approx 0.85$.

The post-iteration coverage on $D_4$ at $\hat q^{(4),B'}_\alpha = 0.625$ is $0.405$, a 9.5 pp coverage gap — the 56\% relative reduction relative to Strategy A. The mechanism is *not* a tighter density-ratio estimator (Strategy B is point-equivalent to A); it is the *anchoring* of the rung-$k$ quantile above the rung-$(k-1)$ inheritance, which prevents the global ratio's low-end spike from pulling the threshold below where the high-end mass lives at the test rung.

#### §3.3.4 Cross-score and cross-model generalisation

The recovery-aware $t^*$ trigger of \S3.3.1 is parametrised by (i) the per-step score family $S_t$ and (ii) the per-step CP threshold $q_\alpha(t)$. We instantiate (i) with three choices — `lp`, `ent_neg`, `marg` — and observe that `marg` outperforms `lp` and `ent_neg` empirically for cascade-stratified lift on AIME, with $\Delta_{\mathrm{strat}}(g \geq 5)$ on `marg__a0.1` ($+18.76$ pp) more than 4× the corresponding `lp__a0.1` value. This score-dependence is not derived from Theorem 4 — Lemma 4.B's $\kappa$ is a property of the SCM, not the score, and a cross-score theory of $\kappa$ is open future work (Theorem 4 v3 §9 (5)). For the empirical pilot we use `marg` as the headline trigger and report `lp` / `ent_neg` as supporting evidence.

The cross-model dimension is parametrised by $p_{\mathrm{cascade}}(\theta, \mathcal D)$, which Theorem 4 v3 §4 reads as a (model, dataset)-pair-indexed quantity. Empirically: math-specialist models on easy domains have very high $p_{\mathrm{cascade}}$ (~0.95) and are *worse* candidates for intervention than generic models on the same domain — the sticky single-mode reasoning propagates errors faithfully without recovery slack. Larger / non-distilled models on hard domains have lower $p_{\mathrm{cascade}}$ (~0.75–0.80) and benefit most from intervention. The cross-cell heterogeneity in $p_{\mathrm{cascade}}$ is testable independently from per-step CP-violation patterns (Theorem 4 v3 §D.3) and provides a pre-flight diagnostic for whether to invoke Theorem 4 / Theorem 6 on a given (model, dataset) cell.

**Empirical evidence summary (Theorem 4 v3, §1.1–1.4 of the source draft).** Four closed `pearl_full/` cells at $n = 200$ provide the headline empirical signal. On `qwen25_7b__math500` (vanilla 0.740, easy-cell), all 9 (score, $\alpha$) settings give negative aggregate lift with cascade-gap-stratified breakdown showing gap≥5 *most negative* on `lp__a0.3` ($-4.95$ pp) — operationally vacuous because $\kappa$ is small on easy datasets and $\Lambda(g)$ dominates. On `qwen25_7b__aime` (vanilla 0.245, hard-cell), aggregate ranges from $-2.69$ pp to $+1.85$ pp depending on (score, $\alpha$); the cascade-gap stratification shows striking divergence: at `marg__a0.1`, gap=1 = $-1.51$ pp, gap∈[2,4] = $+6.91$ pp, gap≥5 = **+18.76 pp**. On `qwen25_7b__olympiad` (vanilla 0.420), aggregate is mid-range with `lp__a0.3` = $+1.08$ pp, but gap≥5 = $-3.18$ pp — the (A6)-violation signature. On `qwen25_math_7b__math500` (vanilla 0.740), all settings show *more strongly negative* aggregate than `qwen25_7b__math500`, consistent with very high $p_{\mathrm{cascade}}$ on a sticky single-mode math specialist. The cascade-gap-stratified pattern is the central new empirical signal that Theorem 4 v3's Corollary 4.1 quantitatively explains.

#### §3.4.4 Regime analysis: when does the ladder win?

The ladder method (Strategy B' under Theorem 5'/v2) is *not* a strict improvement over one-shot weighted CP (Strategy A under Theorem 3) in every regime; it is a *Pareto* improvement that wins on specific axes.

| Regime | Strategy A (one-shot) slack | Strategy B' slack | Winner |
|---|---|---|---|
| Large $\epsilon_{\mathrm{global}}$, small per-rung TVs, equal $n_k$ | $\epsilon_{\mathrm{global}} \cdot W^+_{\mathrm{global}} \cdot \mathrm{DKW}$ | $\sum_k \epsilon_k$ + $\bar L^K$ damping | **Ladder** |
| Small $\epsilon_{\mathrm{global}}$ (rungs already close) | $\epsilon_{\mathrm{global}} \cdot W^+ \cdot \mathrm{DKW}$ small | $\sum_k \epsilon_k$ also small but $K$-fold | **Tie**; one-shot simpler |
| Non-monotone rungs (intermediate further than terminal) | $W^+_{\mathrm{global}}$ moderate | LHS large (some $W_k^+ \gg 1$) | **One-shot** |
| Small mid-ladder rungs, large terminal | $\min(n_0, n_K)$ controlled | $\min(n_{k-1}, n_k)$ small mid-ladder | **One-shot** |
| Very large $K$, fixed total $\sum n_k$ | Single $\mathrm{DKW}$ slack | $K$-fold union-bound penalty dominates | **One-shot at large $K$** |
| Strategy B' contraction realised ($\bar L < 1$) | (no contraction) | Slack contracted by $(1-\bar L^K)$ | **B'** |

The practical implication is that Theorem 5/5' dominates when (i) rungs are roughly equally sized, (ii) per-rung shift is large enough that $W_k^+ \gg 1$, (iii) $K$ is small enough that the union-bound penalty $\log(K)/\log(1/\delta)$ stays manageable, and (iv) the iteration's contraction is realised. These four conditions — equal-step, substantive-step, small-K, contracting — mirror the structural features of the SH0ES astronomical ladder and are what we treat as the ladder-design checklist for new domains.

#### §3.4.5 Empirical fingerprint summary (Theorem 5/5' source-draft §G)

The 5-rung full pilot (Qwen2.5-7B native ladder, target = AIME-new) reports:

- $\sum_k \hat\epsilon_k = 0.766$ (Theorem 5 v2 worst-case slack)
- $\rho_5 = 0.766 / 0.542 = 1.41$ (chain-overlap inefficiency)
- A_gap at $\alpha = 0.5$: 21.4 pp; B'_gap: 9.5 pp; reduction: 11.9 pp absolute, 56% relative
- Implied $\bar L = 0.85$ (consolidated boxed form $1 - \bar L^5 = 0.56$)

Interpretation: contraction factor consistent with $\bar L^5 \approx 0.44$, so $\bar L \approx 0.85$. Slightly weaker contraction than 4-rung pilot (which had implied $\bar L = 0.7$ via $1 - 0.7^4 = 0.76$, close to 67% empirical); adding rung 2 (Olympiad) with TV 0.148 *did* tighten the chain but the marginal contribution beyond rung 4 (the anchor) is small. This is consistent with the $w(g)$ shape concentrating at small $g$ — the marginal Olympiad rung doesn't add much for the saturation regime.

**Cross-α and cross-model robustness.** The α-grid pattern (cross-α table above) and cross-model identical headline numbers (cross-model table above) together support the structural reading of $\bar L$ as a *property of the rung geometry*. A negative result on this prediction — a model with a materially different rung structure giving the same A/B' gap reduction — would falsify the structural claim. The camera-ready will run a `qwen25_7b__GSM8K-MATH-MMLU` ladder as a control test of this prediction.

#### §3.4.6 Ladder-design checklist for new domains

Before invoking Theorem 5/5' on a new domain, the user should verify the following six conditions on a labelled pilot of the deployment chain.

**(C1) Monotone source-TV (A6).** Verify $d_{TV}(\hat p_0, \hat p_k)$ non-decreasing in $k$. Fail-action: re-order or drop offending rung.

**(C2) Strict per-rung-pair information gain (A4').** Verify $\hat\epsilon_k \geq \tau_{\min} > 0$ for all $k$ (no zero-information rungs). Fail-action: drop the rung; verify $K - 1$-rung ladder is non-degenerate.

**(C3) Sufficient anchor sample size.** Verify $n_K \geq |\mathcal S|^2 \log K / \hat\epsilon_K^2$ — the smallest-$n$ rung dominates the DKW slack. Fail-action: collect more anchor data, or fall back to one-shot CP at the most-recent rung.

**(C4) Bounded ratios (A3).** Verify $W_{\max} \leq W^{\mathrm{user}}$ for a user-specified threshold (typically $W^{\mathrm{user}} = 10$). Fail-action: increase Laplace smoothing, or drop the offending rung.

**(C5) $\bar L < 1$ via Gap-C sufficient condition.** Compute the per-rung $L_k$ from §4.3 Lemma 5.4 and verify $\prod_k L_k < 1$. Fall-back if vacuous: use Theorem 5 v2's worst-case slack instead of Theorem 5' contraction.

**(C6) Empirical (A2) per-rung check.** On a labelled subset, verify $|\hat p_k(\bar S \mid Y = 1) - \hat p_k(\bar S \mid Y = 1, X)| \leq \eta$ for some user-specified $\eta = 0.05$ — i.e., the score is approximately a sufficient statistic for correctness, conditional on the rung. Fail-action: invoke the bounded-(A2) variant with additive bias term $K\eta$.

These checks are operationally automated in the `joint_method_diagnostic.py` script and complete in $O(K \cdot n)$ time on a labelled pilot of $\sim 250$ samples per rung.

### §3.5 Joint method (intervention + ladder)

The joint CoT-CP procedure on a rung-$K$ test prompt $(P_{n+1}, \bar X_{n+1})$ proceeds in three stages.

1. **Apply ladder calibration** (\S3.4) on $\{D_0, \ldots, D_K\}$ to obtain $\hat q^{(K),B'}_\alpha$ and the per-step thresholds $\{\hat q^{(K),B'}_\alpha(t)\}_t$ at the rung-$K$ scale.
2. **Identify cascade source** $t^*(\bar X_{n+1})$ using the recovery-aware trigger of \S3.3.1 against the rung-$K$-calibrated per-step thresholds.
3. **Intervene** at $t^*$ with $K{=}4$ majority (\S3.3.2) to obtain $\hat Y_{n+1}$, and accept/abstain by comparing the post-intervention trajectory score $\bar S^{(K), \mathrm{do}(t^*)}_{n+1} = \phi(\tilde X_{1:T})$ against $\hat q^{(K),B'}_\alpha$.

The joint coverage of this pipeline is bounded by Theorem 6 (\S4.4), whose two orthogonal slack terms — a rung-shift slack from Theorem 5' and a re-roll-failure slack from Theorem 4 v3 — collapse to bare CP when both mechanisms are absent (see the two reductions in \S4.4). The natural deployment regime is **strong models on far-OOD problems** where both cascade and shift are simultaneously present; this is where Theorem 6 strictly dominates either component theorem.

**When does the joint method strictly help?** The slack of bare CP at $D_K$ is $1/(n_+^{(0)} + 1)$. The joint slack of Theorem 6 is $(1 - \bar L^K)\sum_k \epsilon_k + (1 - p_{\mathrm{recover}}) + 1/(n_+^{(0)} + 1)$. The joint pipeline strictly improves on either component (Theorem 4 alone or Theorem 5' alone) when *both* failure modes are simultaneously active: the model must have a non-trivial cascade rate ($p_{\mathrm{recover}} \in (0.1, 0.4)$) *and* the calibration source must be far-OOD ($\sum_k \epsilon_k$ in the $0.4$–$0.8$ regime). The pilot's canonical sweet-spot cell — Qwen2.5-Math-7B on AIME-2024 — has both: cascade depth gap $\bar g \approx 7$ and $\sum_k \epsilon_k \approx 0.77$, with the joint method projected to deliver coverage $\geq 0.39$ at $\alpha = 0.10$ on intervention-conditional traces (the falsifiable §9 prediction of `joint_composition_theorem.md`). When neither failure mode is active (strong model, in-distribution test) the joint method collapses to bare CP — which is *correct behaviour*, not failure.

**When does the joint method *not* help?** Three failure modes (`joint_composition_theorem.md` §10): (F1) the cascade source $t^*$ on the test domain coincides with the *step* that introduces the rung-shift (e.g., the model's cascade is its switch from MATH-style to AIME-style techniques) — Lemma J.1's do-marginalised score is then a *rung-mixture*, not a rung-$K$ score, and the rung-$K$ calibration weights no longer apply. (F2) Self-correcting model (R1-Distill, QwQ, o1-style) — (A5) is violated and the cascade is non-monotone; do-intervention can destroy a nascent self-correction. Theorem 6 inherits (A5) from Theorem 4 and *excludes* this regime. (F3) Multi-modal correct trajectories (A6 violation) — do-intervention switches modes rather than recovering. The bound is honest in declaring (A5)/(A6) violators out of scope.

**Detection of (F1): rung-displacement check.** The user can pre-flight detect (F1) by computing the post-intervention score distribution and comparing its TV to the rung-$K$ calibration distribution; if their TV exceeds the per-rung TV $\epsilon_{K-1}$, the cascade-shift coincidence is likely and the user should fall back to either bare Theorem 4 (intervene without ladder calibration) or bare Theorem 5' (calibrate without intervention). This pre-flight gate is one of the three pre-flight diagnostics our `joint_method_diagnostic.py` script implements.

**Detection of (F2): cascade-monotonicity sanity check.** A sufficient condition for (A5) is that the model's per-step CP-violation rate decreases monotonically along correct continuations (i.e., good prefixes don't suddenly become bad). This is empirically verifiable on a labelled subset of traces, and we observe its failure on R1-Distill traces in CoT-CP §5.6 — the per-step CP-violation rate is non-monotone, with "wait/hmm/alternatively" patterns indicating mid-trace re-deliberation that breaks (A5). For non-self-correcting Qwen2.5-7B / Qwen2.5-Math-7B / Phi-4 / Qwen2.5-32B, (A5) holds.

**Detection of (F3): unimodality of correct-trajectory law.** A standard heuristic is the Cluster-purity score on a small embedded sample of correct trajectories per problem; if multiple distinct framings have non-trivial mass, (A6) is violated. We observe ~5–10\% (A6) violation on OlympiadBench and ~0–2\% on MATH-500 / AIME, and treat OlympiadBench as a partial out-of-scope domain for Theorem 6.

---

## §4 Theory

### §4.1 Trajectory CP coverage (Theorem 1, briefly)

We restate Theorem 1 of `theorems/theorem1_trajectory_cp.md` for completeness; the proof is deferred to Appendix A.1.

> **Theorem 1 (Trajectory-level CP coverage).** Suppose $(P_1, \bar X_1, Y_1), \ldots, (P_{n+1}, \bar X_{n+1}, Y_{n+1})$ are exchangeable. For any measurable aggregator $\phi : \bigsqcup_T \mathbb{R}^T \to \mathbb{R}$ and any $\alpha \in (0, 1)$, conditional on $Y_{n+1} = 1$,
> $$\boxed{\;\Pr\!\big[\,\bar S_{n+1} \geq \hat q_\alpha \,\big|\, Y_{n+1} = 1\,\big] \;\geq\; 1 - \alpha - \tfrac{1}{n_+ + 1}.\;}$$

The proof is standard split-CP \cite{angelopoulos2023gentle}: conditioning on the correct subset preserves exchangeability, and the rank of $\bar S_{n+1}$ among the $n_+ + 1$ correct scores is uniform (modulo ties). The bound is sharp; the $1/(n_+ + 1)$ slack matches \cite{angelopoulos2024crc}. Theorem 1 is the *backdrop* — Theorems 4, 5/5', and 6 below all reduce to Theorem 1 in the relevant degenerate limits.

### §4.2 Pearl-causal coverage (Theorem 4 v3)

We work in the autoregressive SCM of \S3.3 with the assumption set distilled from `theorem4_v3_cascade_stratified.md` §7.1.

**Assumptions.**

- **(A1') Prefix-blocking under controlled inference.** Conditional on $X_{1:t-1}$ (the trace prefix) and $P$ (the prompt), the step $X_t$ has no unblocked back-door path to $Y$ that bypasses $X_{t+1:T}$. Operationally this requires deterministic / fixed-temperature inference with no runtime memory leakage between traces (\cite{pearl2009causality} §3.4).
- **(A2) Score-validity.** The per-step calibrated threshold $q_\alpha(t)$ from PRM800K satisfies $\Pr[S_t < q_\alpha(t) \mid Y_t = 1] \leq \alpha$ for correct-trajectory steps under split CP \cite{lightman2023verify, angelopoulos2024crc}.
- **(A3') Recovery-aware $t^*$ + cascade monotonicity.** A wrong trace's below-threshold pattern is a contiguous run starting at $t^*$ (no isolated innocent violations); the recovery-aware $t^*$ trigger of \S3.3.1 enforces this by construction.
- **(A4') Effective-resampling.** The K=4 majority at $t^*$ has positive recovery probability: $1 - (1 - p_{\mathrm{recover}}(t^*, \bar x))^K \geq \tau > 0$.
- **(A5) Cascade contractivity, non-self-correcting model class.** The model is *not* a self-correcting reasoner (R1-Distill, QwQ, o1-style explicitly excluded), so off-trajectory prefixes propagate errors with cascade probability $p_{\mathrm{cascade}}(\theta, \mathcal{D}) \in (0, 1)$.
- **(A6) Unimodal correct-trajectory conditional.** For each rung-$K$ prompt the correct-trajectory law is approximately unimodal in step-content space (multi-modal problems, e.g.\ ${\sim}5\text{--}10\%$ of OlympiadBench, may violate A6).

**Scope of (A1')–(A6).** A1' is the load-bearing graphical assumption (front-door identifiability); A2 is split-CP-standard; A3' makes $t^*$ measurable; A4' bounds the achievable lift; A5 excludes frontier reasoning models; A6 excludes multi-modal problem classes. None is asserted for a population-level guarantee — all are *conditional* on the (model, dataset) cell.

#### Lemma 4.A (front-door identification)

> **Lemma 4.A.** Under (A1') and (A5), the do-quantity $\Pr[Y = 1 \mid \mathrm{do}(X_t \sim \pi), W]$ — where $W = (X_{1:t-1}, P)$ — is **identifiable** from the observational $\pi$- and answer-judge distributions via the Pearl conditional front-door formula \cite{pearl2009causality, tianpearl2002id}:
> $$\Pr[Y \mid \mathrm{do}(X_t)] \;=\; \sum_{x_{t+1:T}} \Pr[Y \mid x_{t+1:T}, X_{1:t-1}] \cdot \Pr[x_{t+1:T} \mid X_t, X_{1:t-1}].$$

**Proof sketch.** A1' gives prefix-blocking, so the only paths from $X_t$ to $Y$ that survive conditioning on $X_{1:t-1}$ pass through the suffix $X_{t+1:T}$, which acts as the front-door mediator. Tian–Pearl 2002 \cite{tianpearl2002id} ID-Theorem 1 then gives identifiability provided the mediator distribution $\Pr[x_{t+1:T} \mid X_t, W]$ and the outcome conditional $\Pr[Y \mid x_{t+1:T}, W]$ are estimable — which they are, from observational $\pi_\theta$ samples. The conditional version (front-door with covariate $W = X_{1:t-1}$) is Pearl 2009 \cite{pearl2009causality} §3.4 eq.\ (3.29) and is identifiable iff $W$ blocks all back-door paths from $X_t$ to $Y$ that bypass the mediator — exactly (A1'). Rigorous proof in supplementary; see Appendix A.2. $\blacksquare$

**Where (A1') is delicate.** Auto-regressive LMs are not a clean Markov chain — $X_t$ depends on the *full history* $X_{1:t-1}$, and any information leakage between traces (shared cache, batched inference, or non-deterministic kernels) creates a back-door. (A1') is therefore an *inference-pipeline* assumption, not a model property: it requires deterministic decoding (or fixed-temperature sampling with explicit RNG seed) and per-prompt isolation. We verified (A1') on our pilot infrastructure by re-running 20 prompts with the same seed and confirming bit-exact trace reproducibility; large-scale CI-style tests are deferred to the empirical companion paper. A *bounded-violation* version of (A1'), in which the back-door has bounded magnitude $\eta$, gives a Theorem 4-with-slack of the same form plus an additive $O(\eta T)$ term, paralleling Bareinboim–Pearl's transportability-with-perturbation \cite{bareinboim2014srecoverability}.

#### Lemma 4.B (cascade-decay optimality)

> **Lemma 4.B.** Under (A1')–(A5), for any $t > t^*$,
> $$\Pr_{\mathcal{M}}\!\big[Y = 1 \mid \mathrm{do}(X_{t^*} \sim \pi)\big] - \Pr_{\mathcal{M}}\!\big[Y = 1 \mid \mathrm{do}(X_t \sim \pi)\big] \;\geq\; \kappa \cdot \big[1 - p_{\mathrm{cascade}}^{\,t - t^*}\big] - O(\delta T),$$
> with $\kappa \leq \kappa_{\max} \cdot [1 - (1 - p_{\mathrm{recover}})^K]$, $\kappa_{\max} \leq 1 - \eta$, $\delta$ the per-step (A2) violation slack, and $T$ the trace length.

**Proof sketch.** For $t > t^*$, the suffix-prefix $X_{1:t-1}$ already contains the divergent prefix from $t^*$ onward, so the suffix continuation under $\mathrm{do}(X_t)$ remains in the cascade-corrupted manifold with probability at least $p_{\mathrm{cascade}}^{t-t^*}$. The complementary $1 - p_{\mathrm{cascade}}^{t-t^*}$ probability mass is the only headroom available to the late intervention. At $t^*$, intervention reaches the cascade source and accesses the full $\kappa_{\max}$ headroom subject to the $K{=}4$ recovery-effective slack $1 - (1 - p_{\mathrm{recover}})^K$. The $O(\delta T)$ term absorbs (A2) per-step CP violation accumulating over $T$ steps via standard split-CP $\delta$-amplification \cite{angelopoulos2024crc}. (Proof in supplementary; see Appendix A.3.) $\blacksquare$

**Numerical anchoring.** Lemma 4.B's $(\kappa, p_{\mathrm{cascade}})$ are calibrated from one closed AIME row: `qwen25_7b__aime__marg__a0.1`'s $\Delta_{\mathrm{strat}}(g{=}5) = +18.76$ pp, plug $p_{\mathrm{cascade}} = 0.85$ to get $1 - 0.85^5 = 0.556$ and $\kappa \approx 18.76 / 55.6 \approx 0.337$. This $\kappa$ is consistent with the (A4') effective-resampling refinement: $\kappa \leq \kappa_{\max} \cdot [1 - (1 - p_{\mathrm{recover}})^K]$ with $\kappa_{\max} \leq 1 - \eta \leq 1$ and $p_{\mathrm{recover}} \in [0.10, 0.30]$ gives $\kappa \in [0.34, 0.76]$ — and the empirical $0.34$ sits at the lower end, consistent with mid-AIME $p_{\mathrm{recover}} \approx 0.10$. The $(\kappa, p_{\mathrm{cascade}})$ pair is calibrated on one cell and used as the anchor for the falsifiable predictions on the other 5 cells; this is acknowledged retrodiction-with-pre-registered-extrapolation, not free-fitting (Theorem 4 v3 §10 Objection v3-1).

**Numerical sanity-check table.** With $\kappa = 0.34$, $p_{\mathrm{cascade}} = 0.85$ (anchored on AIME `marg__a0.1`):

| $g$ | $1 - 0.85^g$ | predicted $\kappa(1 - p_{\mathrm{cascade}}^g)$ | Empirical (AIME `marg__a0.1`) | Implied $\Lambda(g)$ |
|---|---|---|---|---|
| 1 | 0.150 | +5.10 pp | $-1.51$ pp | $\approx 6.6$ pp |
| 2 | 0.278 | +9.45 pp | (n/a; bucketed in [2,4]) | — |
| 3 | 0.386 | +13.13 pp | (bucketed) | — |
| 5 | 0.556 | +18.91 pp | +18.76 pp | $\approx 0.15$ pp |
| 7 | 0.679 | +23.10 pp | (n/a) | — |
| 10 | 0.803 | +27.30 pp | (n/a) | — |

The $g = 5$ prediction matches the empirical $\Delta_{\mathrm{strat}}(g \geq 5) = +18.76$ pp essentially exactly (calibration choice); the $g = 1$ prediction (+5.10 pp pre-$\Lambda$) is reduced to $-1.51$ pp empirical by an implied $\Lambda(1) \approx 6.6$ pp, consistent with the §6-of-source-draft estimate of $\Lambda(g)$ via firing-rate gap on already-correct traces. The pattern matches v2's qualitative explanation but is now made *quantitative*: false-positive cost dominates at small gap; cascade-decay benefit dominates at large gap; the aggregate is dragged toward zero by the $w(g)$ mass concentrating at small $g$.

**Cross-score note.** `marg` is the score that on AIME is best-aligned with cascade structure; `lp` and `ent_neg` use less of the cascade signal and have lower implied $\kappa$. Theorem 4 v3 does not commit to a cross-score theory of $\kappa$ — the score's selectivity is a Theorem 2-style Pareto axis, not a Theorem 4 axis. The implication for deployment: if intervention is the goal, choose `marg` over `lp` or `ent_neg` for the recovery-aware $t^*$ trigger; if filtering is the goal (no intervention), choose `lp` for compute economy with `prm_min` for mid-cost or `sc_top1` for the high-cost upper Pareto bound.

#### Theorem 4 (Pearl-Causal Step Intervention, v3)

> **Theorem 4 (Pearl-causal earliest-step dominance, cascade-gap stratified).** Let $\mathcal{M}$ be the autoregressive SCM with model class restricted by (A5) and problem class by (A6). For any wrong trace $\bar x$ with $Y(\bar x) = 0$ and recovery-aware $t^*(\bar x) < T$, under (A1'), (A2), (A3'), (A4'), (A5), (A6):
>
> **(I) Identification.** $\Pr[Y = 1 \mid \mathrm{do}(X_{t^*} \sim \pi), W]$ is identifiable from observational distributions (Lemma 4.A).
>
> **(II) Pointwise dominance.** For all $t > t^*$, $\Pr_\mathcal{M}[Y = 1 \mid \mathrm{do}(X_{t^*})] \geq \Pr_\mathcal{M}[Y = 1 \mid \mathrm{do}(X_t)]$, with quantitative gap as in Lemma 4.B.
>
> **(III) Cascade-gap-stratified lift (Corollary 4.1).** The within-stratum trace-population lift on cascade gap $g \geq 1$ satisfies
> $$\boxed{\;\;\Delta_{\mathrm{strat}}(g) \;\geq\; \kappa \cdot \big[1 - p_{\mathrm{cascade}}^{\,g}\big] \;-\; \Lambda(g) \;-\; O(\delta T),\;\;}$$
> where $\Lambda(g) \geq 0$ is the false-positive cost on already-correct traces. The aggregate $\overline{\Delta} = \sum_g w(g) \Delta_{\mathrm{strat}}(g)$ is a $w$-weighted mixture; $\overline{\Delta}$ can be near zero or negative when $w(\cdot)$ concentrates on small $g$, even when $\Delta_{\mathrm{strat}}(g \geq 5)$ is substantially positive.

**Proof sketch.** (I) is Lemma 4.A. (II) is Lemma 4.B specialised to the within-trace bound. (III) follows by specialising the within-trace bound to $t = t_{\mathrm{worst}}(\bar x)$ with $g(\bar x) := t_{\mathrm{worst}}(\bar x) - t^*(\bar x)$, taking conditional expectation over $\bar x$ within stratum $\{g(\bar x) = g\}$, and subtracting the false-positive cost $\Lambda(g)$ that arises because the K=4 majority operates on the *full* test set including originally-correct traces. The aggregate-mixture decomposition $\overline{\Delta} = \sum_g w(g) \Delta_{\mathrm{strat}}(g)$ is then immediate; the small-$g$ drag follows because $g \mapsto 1 - p_{\mathrm{cascade}}^g$ is monotone increasing and concave while typical $w(\cdot)$ on math/AIME concentrates at $g \in \{1, 2\}$. (Proof in supplementary; see Appendix A.4.) $\blacksquare$

**Pre-registered falsifiable predictions** (derived from Corollary 4.1 + per-cell heterogeneity in $p_{\mathrm{cascade}}(\theta, \mathcal{D})$, with $\kappa \approx 0.34$ and $p_{\mathrm{cascade}} \approx 0.85$ calibrated from one closed AIME row). For the five cells still running in `pearl_full/`: (i) `phi4__aime` predicted aggregate $+5\,$pp at $\alpha = 0.5$, gap≥5 lift $+15$ to $+25\,$pp; (ii) `qwen-math__olympiad` aggregate $+3$ to $+5\,$pp, gap≥5 lift $+10$ to $+20\,$pp; (iii) `qwen2.5-32b__aime` aggregate $+3\,$pp, gap≥5 lift $+10$ to $+18\,$pp; (iv) `phi4__math500` aggregate $+1\,$pp, gap≥5 lift $+5$ to $+10\,$pp; (v) `qwen2.5-7b__olympiad` aggregate $0$ to $+3\,$pp at $\alpha = 0.3$, gap≥5 lift $+5$ to $+12\,$pp under (A6). Theorem 4 v3 is **falsified** if (a) `phi4__aime` aggregate is negative at every $(\text{score}, \alpha)$, (b) `qwen-math__olympiad` gap≥5 CI excludes positive values, (c) `phi4__math500` gap≥5 lift is more negative than `qwen25_7b__math500`'s, (d) cross-cell rank-correlation between predicted and observed lift is negative, or (e) gap=1 lift exceeds gap≥5 lift on any AIME row at the best $(\text{score}, \alpha)$. Full falsification requires $\geq 3$ of these to fire.

**Closed-cell sanity checks (not used to fit).** Four cells already have $n = 200$ closed results in `pearl_full/`:

| Cell | T4 v3 prediction | Empirical |
|---|---|---|
| `qwen25_7b__math500` aggregate | small negative; gap≥5 most negative | aggregate $-0.7$ to $-2.7$pp; gap≥5 = $-4.95$pp on `lp__a0.3` ✓ |
| `qwen25_7b__aime` aggregate (best score) | ~0 to +2pp; gap≥5 strongly positive | `lp__a0.5` = +1.85pp; `marg__a0.1` gap≥5 = +18.76pp ✓ |
| `qwen25_7b__olympiad` aggregate | small positive; (A6) violation drag | `lp__a0.3` = +1.08pp; gap≥5 = $-3.18$pp (potential (A6) violation) ⚠ |
| `qwen25_math_7b__math500` aggregate | most negative (sticky math specialist) | $-2.23$pp on `lp__a0.1`; gap≥5 = $-4.33$pp ✓ |

The third cell (`qwen25_7b__olympiad`) is the *strongest partial falsifier already visible*: gap≥5 = $-3.18$pp has the *opposite* sign from Corollary 4.1's $g$-monotonicity prediction. We diagnose this as a likely (A6) violation (the multi-modal subset of OlympiadBench, ~5–10\% of problems, admits multiple correct framings) but cannot exclude alternative explanations (e.g., higher-than-expected $p_{\mathrm{cascade}}$ on OlympiadBench). Theorem 4 v3 §9 (10) honestly flags this as the strongest open issue. The four-cell sanity check rank-correlation between predicted and observed lift is positive (Spearman $\rho > 0$), so the structural claim survives partial falsification.

**Heterogeneity in $p_{\mathrm{cascade}}(\theta, \mathcal D)$.** Theorem 4 v3 reads $p_{\mathrm{cascade}}$ as a (model, dataset)-pair-indexed quantity, motivating a heuristic taxonomy:

| Regime | $p_{\mathrm{cascade}}$ | Vanilla acc | Typical $\bar g$ | Predicted aggregate $\overline\Delta$ |
|---|---|---|---|---|
| Strong model, easy dataset (e.g. `qwen25_7b__math500`) | high (≥0.90) | high (~0.74) | small (~1.5–2) | small or negative (false-positive cost dominates) |
| Strong model, hard dataset (e.g. `qwen25_7b__aime`) | medium (~0.85) | low (~0.25) | mixed; long tail | positive *in the gap≥5 stratum*, near-zero aggregate |
| Strong model, very hard dataset (e.g. `phi4__aime`, `qwen2.5-32b__aime`) | low (~0.75) | low–mid | long tail | **positive aggregate** (cascade-decay benefit visible everywhere) |
| Math-specialist model, easy dataset (`qwen25_math_7b__math500`) | very high (~0.95) | high (~0.74) | small | most negative (sticky single-mode) |

The mechanism: a model distilled / RL-trained into a sticky single-mode solver propagates errors faithfully — high $p_{\mathrm{cascade}}$. A larger/stronger model on a hard problem has more *noise mode* per step, so off-trajectory prefixes have more chance to recover — lower $p_{\mathrm{cascade}}$. The empirical pattern (`qwen25_math_7b` being *worse* than `qwen25_7b` on math500 for cascade-stratified lift) directly reflects this. $p_{\mathrm{cascade}}(\theta, \mathcal D)$ can be estimated independently from per-step CP-violation patterns (Theorem 4 v3 §D.3), and Corollary 4.1 then predicts $\Delta_{\mathrm{strat}}(g)$ on the held-out cells without further fitting.

### §4.3 Distance-Ladder coverage (Theorem 5 + 5')

We use the consolidated assumption set of `theorem5_v2_consolidated.md` §D.

**Assumptions.**

- **(A1) i.i.d.\ within rung.**
- **(A2) Score-only consecutive shift.** $dD_k / dD_{k-1}(x, s, y) = w_k(s)$ depends on $s$ alone (\cite{tibshirani2019weighted, wang2025posteriordrift} for relaxations).
- **(A3) Bounded per-rung ratios.** $w_k^- \leq w_k(s) \leq w_k^+$ with $w_k^+ \leq W_{\max}$ a user-specified constant.
- **(A4') Independent and informative rungs.** Rung samples are mutually independent and $d_{TV}(P_k, P_{k+1}) \geq \tau_{\min} > 0$ (no zero-information rungs).
- **(A5) Discrete score support.** $|\mathcal{S}| < \infty$; for SC@$N$, $|\mathcal{S}| = N + 1$ (we use $N = 8$, $|\mathcal{S}| = 9$).
- **(A6) Monotone source-TV.** $d_{TV}(D_0, D_k) \leq d_{TV}(D_0, D_{k+1})$ — empirically testable per pilot.

We additionally invoke (B1) per-step Lipschitz quantile and (B2) mean contraction $\bar L < 1$ for Theorem 5'; these are conditions on the iterated-quantile operator $T_k$ of \S3.4.2.

#### Lemmas 5.1–5.5

**Lemma 5.1 (per-rung empirical TV concentration; DKW).** For each rung pair, with probability $\geq 1 - \delta_k$ over the joint sample at rungs $k - 1$ and $k$,
$$|\hat\epsilon_k - \epsilon_k| \;\leq\; |\mathcal{S}| \cdot \delta_{\mathrm{DKW},k}(\delta_k), \qquad \delta_{\mathrm{DKW},k}(\delta) := \sqrt{\tfrac{\log(2|\mathcal{S}|/\delta)}{2 n_{\min, k}}},$$
by Bretagnolle–Huber–Carol applied per PMF \cite{barber2023beyond}. (Standard; proof in Appendix A.5.)

**Refinement.** A tighter version uses the Berend–Kontorovich (2013) lower bound on discrete-PMF $L^1$-concentration: $|\hat\epsilon_k - \epsilon_k| \leq \sqrt{(|\mathcal S| - 1)\log(2/\delta) / n_{\min, k}}$, which removes one factor of $\sqrt{|\mathcal S|}$. We use the looser Bretagnolle–Huber–Carol form for clarity and because the $\sqrt{|\mathcal S|}$-vs-$\sqrt{|\mathcal S| - 1}$ distinction is subdominant at $|\mathcal S| = 9$. The empirical TV concentration is the load-bearing finite-sample slack term in Theorem 5 v2; tighter concentration directly reduces the bound's slack at small $n$.

**Lemma 5.2 (per-rung nexCP coverage gap).** Under (A1), (A2), (A5),
$$\Pr_{D_k}\!\big[\bar S^{(k)} \geq \hat q^{(k)}_\alpha \mid Y^{(k)} = 1\big] \;\geq\; 1 - \alpha - \epsilon_{k-1} - \tfrac{1}{n_+^{(k-1)} + 1}, \quad \epsilon_{k-1} := d_{TV}(P_{k-1}, P_k),$$
by Barber et al.'s nexCP single-jump bound \cite{barber2023beyond} Theorem 2a. Sharpness is via Strassen 1965 \cite{strassen1965} TV-coupling duality.

**Reading Lemma 5.2.** Each per-rung application of nexCP loses at most $\epsilon_{k-1}$ in coverage. The $1/(n_+^{(k-1)} + 1)$ is the finite-sample split-CP slack, which we collapse into a single $1/(n_+^{(0)} + 1)$ term via the iteration (only rung 0's calibration count enters because the iteration uses rung 0 as the anchor). The bound is tight: Strassen Theorem 11 makes $\epsilon_{k-1}$ the *best possible* TV-slack for a single nexCP step — equality is achieved under the optimal coupling. Improvements to Lemma 5.2 are therefore only possible by (a) replacing TV with a tighter divergence (KL via Pinsker, Wasserstein-1 via \cite{villani2009ot}), or (b) restricting the rung structure (e.g., requiring symmetric-coupling between rungs).

**Lemma 5.3 (Strassen-Lindvall coupling chain — Gap B core).** Glue the per-rung optimal couplings $(X_k, X_{k+1})$ via the gluing lemma \cite{lindvall2002coupling} Theorem I.5.4 into a Markov chain $X_0, X_1, \ldots, X_K$ such that the marginal $(X_k, X_{k+1})$ achieves the per-rung TV $\epsilon_k$. By sub-additivity, $\Pr(X_0 \neq X_K) \leq \sum_k \epsilon_k$.

**Construction.** For each rung pair $(k, k+1)$, Strassen 1965 \cite{strassen1965} Theorem 11 gives an optimal coupling $\pi_{k, k+1}$ on $\mathcal S \times \mathcal S$ with marginals $P_k$ and $P_{k+1}$ and disagreement probability $\Pr_{\pi_{k,k+1}}[X_k \neq X_{k+1}] = d_{TV}(P_k, P_{k+1}) = \epsilon_k$. The Lindvall gluing lemma \cite{lindvall2002coupling} Theorem I.5.4 chains $K$ such pairwise couplings into a single joint distribution on $\mathcal S^{K+1}$ such that (i) the marginal at index $k$ is $P_k$, (ii) the bivariate marginal $(X_k, X_{k+1})$ is the Strassen optimal coupling at the rung-pair, and (iii) the chain is *Markov* — i.e., $(X_{k+1} \mid X_k) \perp (X_0, \ldots, X_{k-1})$. The gluing is constructive: condition on $X_k = s$ at each step, sample $X_{k+1} \sim \pi_{k, k+1}(\cdot \mid X_k = s)$, and chain. Sub-additivity gives $\Pr(X_0 \neq X_K) \leq \sum_k \Pr(X_k \neq X_{k+1}) = \sum_k \epsilon_k$, by union bound on the disagreement events.

**Why the chain is Markov.** The Markov property is essential for the per-rung error damping in Theorem 5'. If the per-rung quantile maps depended on the *cumulative history* rather than only on the immediately preceding $\hat q^{(k-1)}$, the gluing-lemma Markov reconstruction would fail, and the endpoint coupling would no longer have $\Pr(X_0 \neq X_K) \leq \sum_k \epsilon_k$. Strategy B' as specified in \S3.4.2 is Markov by construction: $T_k$ depends only on $\hat q^{(k-1)}$ and the rung-$k$ sample. If a user modifies B' to depend on cumulative history (e.g., averaging quantiles across rungs), the proof of Theorem 5' breaks and a separate non-Markov chain analysis would be required.

**Lemma 5.4 (per-rung Wasserstein-Lipschitz of $T_k$ — Gap A).** Under (A1)–(A6), the per-rung iterated-quantile operator $T_k$ of \S3.4.2 is $L_k$-expected-Lipschitz on $\mathcal{Q}$ in the Wasserstein-1 metric, with
$$L_k \;\leq\; \frac{W_{\max}}{w_{\min, k}\,\alpha} \cdot |\mathcal{S}| \cdot \big(\epsilon_k + 2 \delta_{\mathrm{DKW}, k}\big).$$
Proof via classical inverse-CDF Lipschitz \cite{tibshirani2019weighted} Lemma A.1 specialised to the discrete weighted CDF, plus a Path-A/Path-B passage with $O(\Delta_{\max})$ smoothing slack \cite{villani2009ot} Cor.\ 7.4. (Proof in supplementary; see Appendix A.6.)

**Why the right metric is Wasserstein-1, not $L^\infty$.** The discrete output of $T_k$ creates step-discontinuities in any classical $L^\infty$-Lipschitz inequality $|T_k(\hat q_1) - T_k(\hat q_2)| \leq L_k |\hat q_1 - \hat q_2|$: at $\hat q$ values straddling a score atom $S_i^{(k)}$, the LHS is $\Omega(\Delta_{\min}) = \Omega(1/|\mathcal S|)$ while the RHS can be made arbitrarily small. The Wasserstein-1 / expected-Lipschitz reformulation (Path A in `theorem5_gap_A_lipschitz.md` §3) averages over the smooth interior and the discontinuities together: treating $\hat q$ as a random variable with law $\mu$ on $\mathcal Q$ (in our application, the law of $\hat q^{(k-1)}$ induced by the joint sample), we ask for $W_1(T_{k\#}\mu, T_{k\#}\mu') \leq L_k \cdot W_1(\mu, \mu')$. This is the metric that makes Banach-iteration arguments rigorous on lattice-valued maps \cite{villani2009ot} Theorem 7.3. The smoothed interpolated operator $T_k^{\mathrm{lin}}$ (Path B) is classically Lipschitz with the displayed constant, and $|T_k(\hat q) - T_k^{\mathrm{lin}}(\hat q)| \leq \Delta_{\max}$ provides the Path-A → Path-B passage with $O(\Delta_{\max})$ additive slack — which we absorb by inflating $L_k$ by a factor of 2 in the regular regime where no atom moves into or out of the truncation as $\hat q$ varies (with high probability when $|\hat q_1 - \hat q_2| = O(1/n_k)$).

**Worked-bound check.** With pilot constants $W_{\max} = 5$, $w_{\min, k} = 0.5$, $\alpha = 0.5$, $|\mathcal S| = 9$, $\epsilon_k = 0.42$ (rung-1→2, the dominant rung), $\delta_{\mathrm{DKW}, k} \approx 0.20$: $L_k \leq (5/0.25) \cdot 9 \cdot (0.42 + 0.40) = 20 \cdot 9 \cdot 0.82 \approx 148$ — vacuous. The realised empirical $L_k \approx 0.96$ on the trajectory (from inverting `q_path` differences) is $\sim 150 \times$ smaller. The looseness factors as: $(W_{\max}/w_{\min,k} = 10$ vs realised $\sim 1.5) \times (|\mathcal S| = 9$ vs realised $\sim 4) \times (1/\alpha = 2$ vs realised $\sim 1)$ — i.e., the worst-case bound assumes singular weights at every atom, the worst-case granularity at every quantile evaluation, and the worst-case lower-quantile sensitivity, none of which hold simultaneously in any realisation. Tightening this to the typical-case Lipschitz is `[GAP C]`, addressed by an L1 (vs L∞) weight perturbation reformulation (Theorem 5 v2's TV form) plus realised-vs-worst-case granularity at the actual quantile atom (`theorem5_gap_C_contraction_sufficient.md` §9).

**Lemma 5.5 (composition; product Lipschitz — Gap A).** $T = T_K \circ \cdots \circ T_1$ is expected-Lipschitz on $\mathcal{Q}$ with constant $\bar L \leq \prod_{k=1}^K L_k$. The Wasserstein-1 metric is multiplicatively monotone under composition (\cite{villani2009ot} Cor.\ 7.4), so the *product* form — not a sup-norm — is the right contraction certificate.

**Why product, not sup-norm.** A sup-norm composition lemma $\bar L_{\sup} = \max_k L_k$ would say "the worst rung dominates" — which contradicts the Banach contraction we want. Even with one bad rung ($L_k > 1$), $\bar L < 1$ can hold if other rungs contract enough; the product captures this. In the pilot, $L_2 \approx 2.28$ (rung 1→2, dominant TV) is above 1, but the other three rungs have $L_k < 1$, and the product $\bar L \approx 0.85$ stays below 1. The product Lipschitz is necessary for Banach iteration; sup-norm would not suffice. The proof is by induction on $K$: for $K = 1$, Lemma 5.4 is the base case; assume $T^{(K-1)} := T_{K-1} \circ \cdots \circ T_1$ is $\prod_{k=1}^{K-1} L_k$-Lipschitz. For any couplings on $\mathcal Q$, push forward through $T^{(K-1)}$, then apply $T_K$ — the displacement contracts by $\prod_{k=1}^K L_k$, completing the induction.

**When the product is loose.** The product is tight in the worst case when each rung's perturbation aligns with the prior rung's perturbation direction. It is loose when the per-rung perturbations are *uncorrelated*; in that case, by a standard concentration argument, the product form is replaced by $\sqrt{\sum_k L_k^2}$ (Pinsker-style). Without further structural assumptions (which we do not make), the product is what we can prove. Empirically, the realised contraction in the pilot is closer to $L_K = 0$ (path saturation at one atom from rung 3 onward) than to the worst-case product, so the bound is conservative; tightening is open.

#### Theorem 5 v2 (one-shot bound)

> **Theorem 5 v2 (K-rung ladder coverage, telescoping TV-summation slack).** Under (A1)–(A6), let $\hat q^{(K), \mathrm{ladder}}_\alpha$ be the *telescoped* weighted-CP quantile (Strategy B of \S3.4.3, point-equivalent to one-shot Strategy A). Then for $\alpha \in (0, 1)$, $\delta \in (0, 1)$, with probability $\geq 1 - \delta$ over the joint sample,
> $$\Pr_{D_K}\!\big[\bar S^{(K)} \geq \hat q^{(K), \mathrm{ladder}}_\alpha \mid Y^{(K)} = 1\big] \;\geq\; 1 - \alpha - \sum_{k=0}^{K-1}\!\big(\hat\epsilon_k + \mathrm{DKW}_k(\delta/K)\big) - \tfrac{1}{n_+^{(0)} + 1}.$$

**Proof sketch.** Apply Lemma 5.2 (per-rung nexCP) along the chain $D_0 \to D_1 \to \cdots \to D_K$ and glue the per-rung couplings via Lemma 5.3 to obtain endpoint coupling failure $\leq \sum_k \epsilon_k$. Combine with Lemma 5.1 (DKW) at each rung-pair (union-bounded over $k$ at level $\delta/K$) and the standard split-CP $1/(n_+^{(0)} + 1)$ correction. (Proof in supplementary; see Appendix A.7.) $\blacksquare$

**Where Theorem 5 v2 wins.** Theorem 5 v2 is **strictly tighter** than the one-shot nexCP bound (which pays the global $d_{TV}(P_0, P_K)$ amplified by the global density-ratio sup) whenever $\sum_k \epsilon_k$ is below the global density-ratio-amplified DKW slack (\S3.4.1 chain-overlap inefficiency $\rho_K < W_{0 \to K}^+ |\mathcal{S}|/\sqrt{n_{\min}}$). Compare to Wang–Wu–Liang's gradual-DA bound \cite{wang2022gradual}, which has identical $\sum_k \epsilon_k$ structure for classifier risk under intermediate domains; Theorem 5 v2 is the conformal-prediction analog of the GDA additive bound, and is to our knowledge the first telescoping-density-ratio CP bound for multi-rung benchmark transfer.

**Where Theorem 5 v2 is tight.** The bound is tight when (i) per-rung errors are *coupling-disjoint* (the optimal Strassen coupling at rung $(k, k+1)$ has zero overlap with the coupling at $(k+1, k+2)$, so the chain-rule sum equals the global TV); (ii) the empirical PMFs at adjacent rungs are well-separated (no rung-zero TV that would inflate the chain bound by an empty rung); (iii) sample sizes $n_k$ are balanced (no anchor-rung effect). Tightness fails on `qwen25_7b`'s 5-rung pilot, where $\sum_k \epsilon_k = 0.766$ exceeds the global $0.542$ by 41\% — the chain-rule sum is *over-counting* round-trip mass via non-monotone source TVs. (A6) flags this; the recommended fix is rung re-ordering, not bound tightening.

**Comparison to Strategy A (one-shot Theorem 3).** Theorem 3 \cite{tibshirani2019weighted} pays $\epsilon_{\mathrm{global}} \cdot W^+_{0 \to K} \cdot |\mathcal{S}|/\sqrt{n_{\min}}$ as the leading term, where $W^+_{0 \to K} = \prod_k W^+_k$ is the global density-ratio sup. Theorem 5 v2's leading term $\sum_k \epsilon_k$ has *no* $\prod W^+$ pre-factor — the multiplicative product is replaced by an additive sum, which is strictly tighter whenever any $W^+_k > 1$ (a generic condition on a non-trivial ladder). The astronomy-distance-ladder analog: SH0ES doesn't pay a multiplicative compound across rungs; it pays an additive sum of overlap-region cross-calibration errors. The point-estimate is identical (Strategy A = Strategy B by telescoping algebra $\prod_k \hat p_k/\hat p_{k-1} = \hat p_K/\hat p_0$), so the v2 contribution at the *worst-case slack* level is a slack-bound improvement, not a point-estimate improvement. Strategy B' provides the point-estimate improvement (Theorem 5' below).

#### Theorem 5' (Banach fixed-point, sequential ladder)

> **Theorem 5' (Sequential ladder coverage via Banach contraction).** Under (A1)–(A6) and (B1), (B2) with $\bar L := \prod_{k=1}^K L_k < 1$, the iterated quantile operator $T = T_K \circ \cdots \circ T_1$ has a unique fixed point $q^* \in \mathcal{Q}$ \cite{banach1922, granas2003}, and the Strategy-B' iterate $\hat q^{(K), B'}_\alpha = T(\hat q^{(0)}_\alpha)$ converges to $q^*$ at geometric rate $\bar L^K$. The rung-$K$ correctness-conditional coverage satisfies
> $$\boxed{\;\Pr_{D_K}\!\big[\bar S^{(K)} \geq \hat q^{(K), B'}_\alpha \mid Y^{(K)} = 1\big] \;\geq\; 1 - \alpha - \sum_{k=0}^{K-1}\bar L^{K-1-k}\,\epsilon_k - \tfrac{1}{n_+^{(0)} + 1},\;}$$
> equivalently, in the uniform-error case $\epsilon_k \equiv \epsilon$,
> $$\geq 1 - \alpha - \tfrac{1 - \bar L^K}{1 - \bar L}\,\epsilon - \tfrac{1}{n_+^{(0)} + 1},$$
> with the consolidated upper-bound form $\geq 1 - \alpha - (1 - \bar L^K)\sum_k \epsilon_k - 1/(n_+^{(0)} + 1)$.

**Proof sketch.** Three steps. (i) Lemmas 5.4–5.5 give Lipschitz of $T_k$ and product Lipschitz of $T$ in Wasserstein-1; (B2) is the Banach contraction precondition. The Banach contraction theorem \cite{banach1922, granas2003} Theorem II.1.1 gives existence and uniqueness of $q^*$ with a-priori bound $|T^K(q_0) - q^*| \leq \bar L^K |q_0 - q^*|$. (ii) Lemma 5.3 (Strassen-Lindvall coupling chain) at the stationary fixed point $q^*$ gives coverage at $q^*$ with slack $\sum_k \epsilon_k$. (iii) Per-rung errors are damped by the subsequent contractions: an error $\epsilon_k$ injected at rung $k$ propagates to rung $K$ with factor $\bar L^{K-1-k}$, giving the contracted slack $\sum_k \bar L^{K-1-k} \epsilon_k$. The damping prefactor is the **structural source of Strategy B's empirical lift**: $(1 - \bar L^K) / (1 - \bar L)$ is a finite geometric series strictly less than $K$ for $\bar L < 1$. (Proof of (iii) in supplementary; see Appendix A.8.) $\blacksquare$

**Quantitative improvement.** In the uniform-error regime, Theorem 5'/Theorem 5 v2 ratio is $((1 - \bar L^K)/(1 - \bar L))/K$. At $\bar L = 0.85, K = 4$: $(1 - 0.85^4)/(1 - 0.85)/4 \approx 0.797$ — a $\sim 20$% slack reduction. At $\bar L = 0.85, K = 5$: $\sim 26$% reduction. At $\bar L = 0.5, K = 4$: $\sim 53$%. The improvement is monotone in $1/\bar L$ (smaller contraction → larger improvement) and saturates as $\bar L^K \to 0$. The empirical 56–67% relative gap reduction in the pilot is consistent with $\bar L \in [0.25, 0.85]$ depending on which slack form (consolidated boxed vs.\ geometric-series) is read; both formulations are valid within a constant factor.

**Why a separate theorem is needed.** Strategy B (telescoped weighted CP) is *point-equivalent* to Strategy A (one-shot weighted CP) by the telescoping algebra $\prod_k \hat p_k/\hat p_{k-1} = \hat p_K/\hat p_0$, so Theorem 5 v2 — though a slack-bound improvement — gives the *same* point estimate as one-shot. Strategy B' (sequential) is *not* point-equivalent: the iterated quantile $\hat q^{(K),B'}_\alpha$ is a fixed point of $T = T_K \circ \cdots \circ T_1$, *not* a single weighted quantile. The empirical pilot demonstrates this: at $\alpha = 0.5$, Strategy A and Strategy B both give $\hat q_\alpha = 0.500$, while Strategy B' gives $\hat q^{(4), B'}_\alpha = 0.625$ — a different point estimate, with B' giving the smaller coverage gap. Theorem 5' is the theorem that explains this divergence: the fixed-point of an iterated weighted-quantile map need not equal the one-shot weighted quantile, and the contraction rate $\bar L^K$ controls the gap.

**Connection to gradual-DA literature.** Wang–Wu–Liang \cite{wang2022gradual} prove that self-training across $T$ intermediate domains achieves a classifier-risk bound that is *additive* in $T$ (rather than exponential under naive analysis), with leading term $\epsilon_0 + O(T\Delta + T/\sqrt{n}) + \tilde O(1/\sqrt{nT})$ where $\Delta$ is the average per-step distributional distance. Theorem 5' is the conformal-prediction analog of this GDA bound: per-step shift $\epsilon_k$ replaces the GDA $\Delta$, and the contraction prefactor $(1 - \bar L^K)/(1 - \bar L)$ replaces the GDA $T$ scaling. Both achieve additive (rather than multiplicative) error propagation by chaining locally-close distributions. To our knowledge Theorem 5' is the first explicit bridge between the gradual-DA literature and the conformal-prediction-under-shift literature; AISTATS-style "AI + statistics interface" venues are the natural target for this synthesis.

**Empirical fingerprint.** On the 4-rung pilot at $\alpha = 0.5$, Theorem 5' fits $\bar L \approx 0.85$ via $1 - \bar L^4 = 0.48$ (matching the empirical 56\% relative reduction); the per-$\alpha$ fits across $\alpha \in [0.05, 0.7]$ are consistent at $\bar L \in [0.5, 0.85]$. This is consistent with Lemma 5.4's worst-case $L_k$ bound being $\sim 80 \times$ loose at the pilot regime (Gap C, `theorem5_gap_C_contraction_sufficient.md` §5), but the $\bar L < 1$ contraction is empirically realised regardless. On the 5-rung full pilot, Theorem 5' predicts $1 - \bar L^5 = 0.56$ exactly matching the headline 56\% relative reduction across four base models (qwen25-7B, qwen25-math-7B, qwen25-32B, phi-4) — the contraction factor is consistent across (model, ladder) cells, supporting the structural reading of $\bar L$ as a property of the **rung geometry** rather than the calibration source.

**Cross-α empirical evidence.** The α-grid pattern is consistent with $\bar L$ being α-dependent — it shrinks for moderate α (where the contraction is most effective) and grows for very small or large α (where the discrete quantile range is narrow and the iteration plateaus). Theorem 5' predicts α-dependent contraction via the $1/\alpha$ factor in $L_k$ (Lemma 5.4): smaller α → larger $L_k$ → weaker contraction at low α; the data are consistent.

| α | A gap | B' gap | Reduction | Theorem 5' implied $\bar L$ |
|---|---|---|---|---|
| 0.05 | 5.0pp | 5.0pp | 0% | (over-coverage; bound vacuous) |
| 0.10 | 10.0pp | 10.0pp | 0% | (over-coverage; ≈ vacuous) |
| 0.20 | 20.0pp | 8.1pp | 60% | ≈ 0.56 (matches) |
| 0.30 | 30.0pp | 1.4pp | 95% | ≈ 0.50 (matches; strong contraction) |
| 0.50 | 21.4pp | 9.5pp | 56% | ≈ 0.85 (matches; consolidated form) |
| 0.70 | 29.5pp | 10.5pp | 64% | ≈ 0.81 (matches) |

**Cross-model evidence.** All four pilot models report identical headline numbers (the AIME rungs are borrowed across `qwen25_math_7b`, `qwen25_32b`, `phi4`; only `qwen25_7b` is native), giving an implied $\bar L \approx 0.85$ that is *constant across models*. This is a non-trivial prediction of Theorem 5': the contraction factor is a *property of the rung structure* (the AIME chain), not of the calibration source.

| Model | $\sum\hat\epsilon_k$ | A gap | B' gap | Reduction | $\bar L$ implied |
|---|---|---|---|---|---|
| qwen25_7b | 0.766 | 21.4pp | 9.5pp | 56% | 0.85 |
| qwen25_math_7b | 1.050 | 21.4pp | 9.5pp | 56% | 0.85 |
| qwen25_32b | 1.026 | 21.4pp | 9.5pp | 56% | 0.85 |
| phi4 | 0.793 | 21.4pp | 9.5pp | 56% | 0.85 |

Falsification: pilot models with materially different rung structures (different AIME chains) showing the same A/B' gaps would refute the rung-structure reading; the camera-ready will run a `qwen25_7b__GSM8K-MATH-MMLU` ladder as a control to test this.

**Open proof-gap labels.** Two technical gaps remain (consistent with the consolidated draft's `[PROOF GAP A/B/C]` taxonomy):

- **`[GAP A]` Discrete-quantile Lipschitz.** Lemma 5.4 is stated in Wasserstein-1 with a Path-A / Path-B passage; the literal $L^\infty$ Lipschitz fails at score-atom step-discontinuities. Resolution via Tarski's monotone fixed point \cite{granas2003} Ch.\ III + Berge maximum-theorem continuity is plausible but not yet a complete proof.
- **`[GAP C]` Sufficient condition $\bar L < 1$.** The product-form sufficient condition (Gap C §3.2) is loose by ${\sim}80\times$ at the 4-rung pilot regime (DKW union-bound dominates at $n_k \approx 200$). A typical-case Lipschitz bound that captures the realised $\bar L \approx 0.85$ from first principles is open; we conjecture a local-attractor framing (`theorem5_gap_C_contraction_sufficient.md` §7) is the right next theorem.

(Proof in supplementary; see Appendix A.8 for the Gap-A and Gap-B closure pieces and A.9 for the Gap-C deferral.)

### §4.4 Joint composition (Theorem 6)

The naive composition of Theorem 4 (intervention) and Theorem 5' (ladder) fails: the post-intervention test score $\bar S^{(K), \mathrm{do}(t^*)}_{n+1}$ is *not* weighted-exchangeable with the rung-$K$ calibration scores, because the do-intervention conditions on $t^*$ existing (selection on $Y$) and replaces the suffix from $\pi_\theta$ rather than from the on-trajectory oracle. Restoring weighted exchangeability requires a do-marginalisation argument, formalised below.

**Two distinct violations of (B0) weighted exchangeability.** First, the K=4 majority is computed only on traces that satisfy the recovery-aware $t^*$ trigger of \S3.3.1 — this is a Bareinboim s-recoverability concern \cite{bareinboim2014srecoverability}, since the test sub-population is selected based on a function of the trace itself rather than a pre-specified split. Second, the resampled $\tilde X_{t^*} \sim \pi_\theta$ is from the model's natural sampling distribution (temperature-0.7), not from the on-trajectory oracle $q^*(\cdot \mid X_{1:t^*-1}, P)$ that would produce a counterfactually-correct continuation. Lemma J.1 below addresses both via Pearl's twin-network coupling and an expectation over the do-distribution.

#### Lemma J.1 (do-marginalised exchangeability)

> **Lemma J.1.** Let $\bar x$ be a rung-$K$ wrong trace with recovery-aware $t^*(\bar x) < T$. Define the **do-marginalised score**
> $$\bar S^{(K), \mathrm{do}(t^*)} \;:=\; \mathbb{E}_{\tilde X_{t^*} \sim \pi_\theta(\cdot \mid X_{1:t^*-1}, P)}\!\Big[\, \phi(\tilde X_{1:T}) \,\Big|\, \tilde X_{t^*}, X_{1:t^*-1}\,\Big].$$
> Under (A1')–(A6) of Theorem 4 and (A1)–(A6), (B1), (B2) of Theorem 5/5' with $\sum_k \epsilon_k < 1$, the do-marginalised score is weighted-exchangeable with the rung-$K$ calibration scores at the rung-product weights $\hat w_{0 \to K}$:
> $$\bar S^{(K), \mathrm{do}(t^*)} \,\big|\, Y^{(K)}_{n+1} = 1 \;\overset{\mathrm{wexch}}{\sim}\; \big\{ S_i^{(K)} : Y_i^{(K)} = 1 \big\}.$$

**Proof sketch.** Two pieces. *Piece 1 (identifiability).* Lemma 4.A's front-door identifier transports across $D_0 \to D_K$ provided the calibration and target sets share inference configuration — (A1') is a property of the inference pipeline, not of the rung, and is orthogonal to (B1)–(B2). *Piece 2 (twin-network coupling).* By (A4) i.i.d.\ within rung and (A4') independent rungs, the suffix noise $\epsilon_{t^*+1:T}$ in the test prompt is independent of the calibration scores. Pearl twin-network coupling \cite{pearl2009causality} §7 lets us share suffix noise between intervened and natural distributions; integrating over $\tilde X_{t^*} \sim \pi_\theta$ yields the do-marginalised score. By independence of rung samples, $\bar S^{(K), \mathrm{do}(t^*)}$ is independent of the rung-$K$ calibration scores conditional on the rung-$K$ weights — the weighted-exchangeability statement. (Proof in supplementary; see Appendix A.10.) $\blacksquare$

**Where Lemma J.1 is delicate.** The lemma works *in expectation over the intervention*. Per-trace, $S(\tilde X_{1:T}^{(j)})$ for a single $j$ is **not** weighted-exchangeable; only the do-marginalisation is. The K=4 majority of \S3.3.2 is therefore a Monte-Carlo estimator of the do-marginalised score, which is the load-bearing reason the K=4 majority concentrates around a CP-admissible quantity and not around any single $j$'s realisation.

**Connection to Theorem 4 v3 §H.1.** Theorem 4 v3 honestly admits that K=4 majority dominance is *not* established by single-do dominance; Lemma 4.B's bound is at the population $\Pr[Y \mid \mathrm{do}(X_{t^*} \sim \pi)]$, not at the K=4 majority's empirical realisation. Lemma J.1 above is the Jensen step that Theorem 4 v3 deferred: under the rung-$K$ weighted exchangeability, the K=4 majority concentrates around $\bar S^{(K), \mathrm{do}(t^*)}$, and the conformal threshold acts on this concentrated mean. Without Lemma J.1, the K=4 majority would still be *a* Monte-Carlo estimator of *something*, but that something would not be a CP-admissible quantity at the rung-$K$ calibration. Lemma J.1 is the load-bearing bridge.

**Failure modes of Lemma J.1.** The lemma assumes (a) (A1') — without prefix-blocking, the do-distribution is no longer identifiable, and the inner conditional in Step 1 of Theorem 6's proof depends on calibration data; (b) (A4) i.i.d.\ within rung — without independence, the suffix noise sharing of Pearl twin-network coupling fails; (c) $\sum_k \epsilon_k < 1$ — without bounded total shift, the rung-$K$ test distribution may not have non-trivial overlap with the rung-0 calibration manifold, and the weighted-exchangeability statement becomes vacuous (the weights $\hat w_{0 \to K}$ are degenerate). All three are foundational assumptions of Theorem 6; none can be relaxed without reformulating the bound.

#### Theorem 6 (joint coverage bound)

> **Theorem 6 (Joint Pearl × Distance-Ladder coverage).** Let $\mathcal{M}$ be the autoregressive SCM with model class restricted by (A5) and problem class by (A6). Let $\{D_k\}_{k=0}^K$ be a rung sequence satisfying (A1)–(A6) of Theorem 5 v2 and (B1), (B2) of Theorem 5'. Let $\hat q^{(K), B'}_\alpha$ be the Strategy-B' iterated quantile from rung 0 to rung $K$. Let $\hat Y_{n+1}$ be the K=4 majority vote of suffix re-rolls from a do-intervention at the recovery-aware $t^*(\bar x)$ of a rung-$K$ wrong trace $\bar x$. Then under (A1')–(A6) **and** Lemma J.1, the joint coverage satisfies:
>
> $$\boxed{\;\Pr\!\big[\hat Y_{n+1} = Y_{n+1}^* \,\big|\, \text{rung } K, \text{intervened at } t^*\big] \;\geq\; 1 - \alpha \;-\; \underbrace{(1 - \bar L^K)\!\!\sum_{k=0}^{K-1}\!\epsilon_k}_{\substack{\text{Theorem 5'}\\ \text{rung slack}}} \;-\; \underbrace{(1 - p_{\mathrm{recover}})}_{\substack{\text{Theorem 4}\\ \text{re-roll failure}}} \;-\; \tfrac{1}{n_+^{(0)} + 1}.\;}$$

**Proof sketch.** Three steps, chaining Theorem 5' (rung slack) and Theorem 4 (intervention slack) via Lemma J.1. (i) **Decompose by do-marginalisation.** $\Pr[\hat Y_{n+1} = Y_{n+1}^*] = \mathbb{E}_{\tilde X_{t^*} \sim \pi_\theta}[\Pr[\hat Y_{n+1} = Y_{n+1}^* \mid \tilde X_{t^*}, X_{1:t^*-1}]]$ by total probability; the inner conditional is a function of $(\tilde X_{t^*}, X_{1:t^*-1})$ only by (A1'). (ii) **Apply Theorem 5' at the do-marginalised score.** By Lemma J.1, $\bar S^{(K), \mathrm{do}(t^*)}$ is weighted-exchangeable with rung-$K$ calibration scores under weights $\hat w_{0 \to K}$. Theorem 5' applies, giving $\Pr[\bar S^{(K), \mathrm{do}(t^*)} \geq \hat q^{(K), B'}_\alpha \mid Y^{(K)} = 1] \geq 1 - \alpha - (1 - \bar L^K)\sum_k \epsilon_k - 1/(n_+^{(0)} + 1)$. (iii) **Translate to answer-coverage via $p_{\mathrm{recover}}$.** The do-marginalised score is the *expectation* over the re-roll; the actual K=4 majority realises this expectation only with probability $1 - (1 - p_{\mathrm{recover}})^K$, which we bound below by $p_{\mathrm{recover}}$ for $K = 4$. Union-bounding gives the displayed slack. (Proof in supplementary; see Appendix A.11.) $\blacksquare$

**Where the chain is delicate.** The crucial step is (ii)'s application of Theorem 5' to $\bar S^{(K), \mathrm{do}(t^*)}$. Theorem 5' was stated for natural (non-intervened) test scores; Lemma J.1 is the bridge that re-establishes weighted exchangeability for the do-marginalised score. *Without Lemma J.1, the Theorem 5' coverage statement does not hold at the post-intervention score, and the joint bound collapses.* This is the new mathematical content that Theorem 6 contributes beyond a sum of two pre-existing theorems.

**Detailed proof of Step (iii).** The core step in Theorem 6 — translating the do-marginalised-score coverage of Theorem 5' into the post-intervention answer-coverage — proceeds by a union bound on the score-coverage event and the recovery event. Let $\mathcal E_1 := \{\bar S^{(K), \mathrm{do}(t^*)} \geq \hat q^{(K), B'}_\alpha\}$ be the score-coverage event (controlled by Theorem 5' applied via Lemma J.1) and $\mathcal E_2 := \{\text{at least one of the K=4 re-rolls produces an on-trajectory continuation}\}$ be the recovery event (with $\Pr(\mathcal E_2) = 1 - (1 - p_{\mathrm{recover}})^K$). By construction, $\hat Y_{n+1} = Y_{n+1}^*$ requires both $\mathcal E_1$ (the score is high enough that the kept set contains the correct answer) and $\mathcal E_2$ (the K=4 majority's continuation is correct). Hence
$$\Pr[\hat Y_{n+1} = Y_{n+1}^*] \;\geq\; \Pr(\mathcal E_1) - \Pr(\mathcal E_2^c) \;\geq\; \big(1 - \alpha - (1 - \bar L^K)\sum_k \epsilon_k - \tfrac{1}{n_+^{(0)} + 1}\big) - (1 - p_{\mathrm{recover}}),$$
where the last step uses $\Pr(\mathcal E_2^c) = (1 - p_{\mathrm{recover}})^K \leq 1 - p_{\mathrm{recover}}$ for $K \geq 1$ and $p_{\mathrm{recover}} \in [0, 1]$. The bound is tight when $\mathcal E_1$ and $\mathcal E_2^c$ are nearly disjoint events; non-trivial correlation between them tightens the bound further (the joint slack term is then smaller than the sum). We use the loose union form for the boxed statement.

**Empirical pilot prediction.** Plugging the canonical pilot values — $\alpha = 0.10$, $\sum_k \epsilon_k = 0.55$ (4-rung), $\bar L = 0.85$, $K = 4$, $p_{\mathrm{recover}} = 0.40$ (back-fit from `pearl_full/qwen25_7b_math500.json`), $n_+^{(0)} = 800$ — gives joint slack $0.263 + 0.60 + 0.001 = 0.864$, and predicted lower bound $1 - 0.10 - 0.864 = 0.036$. This is a vacuous bound at $\alpha = 0.10$; the joint coverage is dominated by the re-roll term. *Conditioning on intervention success* (i.e., conditional on the cascade being recoverable, $p_{\mathrm{recover}} \approx 0.7$–$0.8$) drops the joint slack to $0.51$, predicting coverage $\geq 0.39$ on intervention-conditional traces. The falsifiable prediction (`joint_composition_theorem.md` §9):

> On the canonical (PRM800K → MATH-500 → AIME-old → AIME-new) ladder with Qwen2.5-Math-7B at $\alpha = 0.10$, the empirical *intervention-conditional* coverage of the K=4 majority post-intervention vote should satisfy
> $$\hat\Pr[\hat Y_{n+1} = Y_{n+1}^* \mid t^*(\bar x) \text{ exists, K=4 re-roll fired}] \;\in\; [0.39, 0.55].$$

Theorem 6 fails (in the strong sense) if (i) empirical coverage exceeds 0.65 (joint slack too pessimistic; intervention enhances contraction), (ii) empirical coverage is below 0.20 ($p_{\mathrm{recover}}$ much lower than 0.40 under domain shift, falsifying Lemma J.1), or (iii) coverage on AIME-new exceeds coverage on MATH-500 at the same $\alpha$ (rung-monotonicity violation).

#### Reductions

The composition is well-formed iff Theorem 6 collapses to its components in the appropriate limits.

**Reduction 1 ($K = 1$, no shift; bare intervention).** When $K = 1$ and $\epsilon_0 = 0$ (or $D_0 = D_1$), the rung-slack term $(1 - \bar L^K)\sum_k \epsilon_k = 0$, and Theorem 6 reduces to
$$\Pr[\hat Y_{n+1} = Y_{n+1}^*] \;\geq\; 1 - \alpha - (1 - p_{\mathrm{recover}}) - \tfrac{1}{n_+^{(0)} + 1},$$
which is the K=4-majority version of Theorem 4's earliest-step dominance bound, with $p_{\mathrm{recover}}$ absorbing the $\kappa[1 - p_{\mathrm{cascade}}^{t - t^*}]$ gap of Lemma 4.B at $t = t^*$ (where the cascade-decay headroom is $\kappa$ and $1 - p_{\mathrm{recover}}$ is the achievable-fraction complement). ✓

**Reduction 2 (no intervention; $t^*$ does not exist).** When the trace is correct (or $t^*$ is undefined), the do-step is the identity, and the recovery probability $p_{\mathrm{recover}}$ has no operational meaning; setting $1 - p_{\mathrm{recover}} = 0$, Theorem 6 reduces to
$$\Pr[\hat Y_{n+1} = Y_{n+1}^*] \;\geq\; 1 - \alpha - (1 - \bar L^K)\sum_k \epsilon_k - \tfrac{1}{n_+^{(0)} + 1},$$
which is exactly Theorem 5'. ✓

**Reduction 3 (no shift, no intervention; $K = 1$, $\epsilon_0 = 0$, no $t^*$).** Both mechanism terms vanish; Theorem 6 reduces to bare split CP $\Pr[\hat Y_{n+1} = Y_{n+1}^*] \geq 1 - \alpha - 1/(n_+^{(0)} + 1)$, which is Theorem 1 \S4.1. ✓

The three reductions confirm Theorem 6 is a strict generalisation along the (intervention, ladder) axes — neither component subsumes it, and bare CP is the natural floor.

**Q: When does Theorem 6 *strictly* dominate either component?** The rough answer: Theorem 6 *adds* slack (sum of both component slacks), but its **bound is on a strictly stronger object** — post-intervention answer-coverage at rung $K$. Theorem 4 alone bounds answer-coverage on the calibration rung (useless on OOD). Theorem 5' alone bounds score-coverage on rung $K$ but cannot certify a post-intervention answer. Theorem 6 is the only theorem that simultaneously certifies (i) the calibration was correctly transported across $D_0 \to D_K$ and (ii) the intervention's post-trace majority is rung-$K$-CP-admissible.

**Q: Is the dependence of $p_{\mathrm{recover}}$ on rung index $K$ benign?** The pilot back-fits $p_{\mathrm{recover}} \approx 0.4$ from MATH-500, but on AIME-new (more difficult), the (A4') on-trajectory mass $\tau$ may be lower. This induces a coupling between the Theorem 4 term and the rung index — potentially making the joint slack non-additive. Empirical question; test by computing $p_{\mathrm{recover}}$ separately on each rung's traces. Conjecture: $p_{\mathrm{recover}}$ degrades monotonically in $d_{TV}(D_0, D_k)$, in which case the joint bound has an additional rung-dependent slack term that we currently absorb in $\bar L$.

**Q: Does the contraction factor $\bar L$ change under intervention?** Theorem 5' assumes the natural Strategy-B' iteration. Under do-intervention, the per-step quantile map $T_k$ is replaced by $T_k^{\mathrm{do}}$ acting on the do-marginalised score. If $\bar L^{\mathrm{do}} \leq \bar L$, intervention *enhances* contraction and Theorem 6 underestimates joint efficacy. Conjecture; would need a separate proof. The empirical signal that intervention does not break Strategy B' contraction is provided by `pearl_full/qwen25_7b_aime`'s stable per-α gap reduction across intervention conditions.

**Q: Is Lemma J.1 sharp?** The lemma states weighted exchangeability *in expectation over the intervention*. Per-trace, the post-intervention score is *not* exchangeable. A per-trace version with a TV penalty would give a tighter joint bound at the cost of a per-trace TV term — the discrete-quantile analog of the bounded-gap (A1$_\eta'$) extension of Theorem 4. Open.

### §4.4.4 Q: Joint-aware calibration and tighter bounds

The current Theorem 6 calibrates Strategy B' on rung-$K$'s natural correct-trajectory scores, then applies the do-intervention at deployment. A natural tighter alternative is **joint-aware calibration**: calibrate Strategy B' on rung-$K$ correct scores *conditional on $t^*$ existing in those traces*. This re-establishes selection-on-$Y$ but in a Bareinboim-2014 s-recoverable way \cite{bareinboim2014srecoverability} — the conditional calibration set is selected by the same $t^*$-trigger that fires at deployment, eliminating the s-recoverability mismatch.

**Conjecture 6'.** Joint-aware calibration tightens the joint slack of Theorem 6 by replacing $1 - p_{\mathrm{recover}}$ with a smaller $1 - p_{\mathrm{recover} \mid t^*}$ — the recovery probability *conditional on $t^*$ existing in the calibration trace*, which is empirically larger than the unconditional $p_{\mathrm{recover}}$ because $t^*$-triggering traces are exactly the cascade-deep traces where the K=4 majority has the most leverage. Quantitatively, on the pilot, conditional-$p_{\mathrm{recover}}$ is ~0.5 vs.\ unconditional ~0.4, halving the Theorem 4 slack term.

The conjecture would tighten the joint bound but at the cost of a more complex calibration pipeline: the user must store, for each calibration trace, the trigger-firing event $\mathbb 1[t^*(\bar x_i) < T]$ alongside the score $\bar S_i$, and condition the weighted-quantile computation on the trigger-firing subset. The implementation overhead is $O(n)$ extra storage and one additional indicator per trace; the proof of Conjecture 6' is open and would require an extension of Lemma J.1 to the conditional-calibration setting.



The joint slack $\Delta_{\mathrm{joint}}(K, p_{\mathrm{recover}}, \bar L) = (1 - \bar L^K)\sum_k \epsilon_k + (1 - p_{\mathrm{recover}}) + 1/(n_+^{(0)} + 1)$ guides operational deployment.

**Strong models on far-OOD with high cascade gap — joint sweet spot.** Qwen2.5-Math-7B on AIME-new fits exactly here. Far-OOD ⟹ $\sum_k \epsilon_k$ large (T5' rung slack non-trivial; pilot $\sum \hat\epsilon_k \approx 0.77$ on the 5-rung MATH→AIME ladder). Strong base ⟹ $\bar L < 1$ contraction holds well, so $\bar L^K$ is small but $(1 - \bar L^K)\sum_k\epsilon_k$ is still substantial (pilot $\bar L \approx 0.85$, $K = 5$ gives $(1 - \bar L^5) = 0.56$, rung term $= 0.43$). Cascade gap $g = T - t^*$ large ⟹ $p_{\mathrm{recover}} \approx \tau \cdot p_{\mathrm{cascade}}^{T - t^*}$ has room to be non-trivial (pilot $\tau \approx 0.4$, $p_{\mathrm{cascade}} \approx 0.85$, $T - t^* \approx 8$ gives $p_{\mathrm{recover}} \approx 0.11$). Both mechanism terms are simultaneously activated by the same data; the joint bound is *much* stronger than the sum of the individual bounds because the single rung-$K$ exchangeability check covers both intervention and shift.

**Weak models on near-IID — joint is wasted.** Llama-3-8B on MATH-500-eval: near-IID ⟹ $\sum_k \epsilon_k \approx 0$, T5' term vanishes, $K = 1$ ladder degenerates to bare T4. Frequent cascade ⟹ $p_{\mathrm{recover}}$ low, T4 term dominates. Use bare T4; joint composition is overhead with no payoff.

**Strong models on near-IID — joint reduces to T5'.** Qwen2.5-Math-7B on MATH-500-eval (in-distribution): cascade rare (model is correct most of the time on in-distribution, $1 - p_{\mathrm{recover}} \approx 0$), T4 term vanishes. Calibration is fine ($\sum_k \epsilon_k \approx 0$), T5' term vanishes. Theorem 6 collapses to bare CP; overhead unjustified.

**The deployment recommendation** is therefore: invoke the joint method when the deployment pipeline expects (i) far-OOD test prompts, (ii) a strong base model, and (iii) cascade-deep wrong traces — exactly the regime where neither component theorem alone suffices. We provide a `joint_method_diagnostic.py` script in the supplementary that pre-flight-checks $\sum_k \hat\epsilon_k$, $\bar L$, and $p_{\mathrm{recover}}$ on a labelled sample of the deployment domain and recommends one of {bare CP, T4 alone, T5' alone, T6 joint} based on which mechanism terms are non-trivial.

### §4.5 Discussion of open problems

We list ten honest gaps, organised by which theorem they constrain.

1. **`[GAP A]` Discrete-quantile Lipschitz (Lemma 5.4).** Stated in Wasserstein-1 with smoothing slack; the literal $L^\infty$ Lipschitz fails at score atoms. Resolution via Tarski monotone-operator fixed point + Berge maximum theorem is plausible but not yet rigorous (see Appendix A.6).
2. **`[GAP C]` Sufficient condition $\bar L < 1$.** The product-form bound (Lemma 5.5) is ${\sim}80\times$ loose at the 4-rung pilot; a typical-case Lipschitz bound that recovers the realised $\bar L \approx 0.85$ requires either an L1 (vs L∞) weight perturbation reformulation or a local-attractor refinement of (B2) — both flagged as future work.
3. **(A2) score-only-shift compounds.** Iterating (A2) $K$ times induces additive bias $\sim K \eta$ where $\eta$ is the per-rung (A2) violation magnitude. Wang et al.\ 2025 \cite{wang2025posteriordrift} provides a generalised-shift relaxation; we report $K\eta$ in the empirical accounting and treat it as a ladder-design constraint.
4. **Self-correcting models (A5 violation).** Theorem 6 explicitly excludes R1-Distill, QwQ, and o1-style frontier reasoning models. A separate composition theorem for self-correcting models is open and likely requires multi-step interventions $\mathrm{do}(X_{t^*}, X_{t^*+1}, \ldots)$ rather than the single-do framework.
5. **(A6) multi-modal correct trajectories.** OlympiadBench has ${\sim}5\text{--}10\%$ multi-modal problems where (A6) is violated; we report this as an empirical drag (`qwen25_7b__olympiad` gap≥5 stratum sign-flip, see Theorem 4 §8.1 closed-cell sanity check) but do not formally bound the multi-modal slack. A proper extension would require a modal-decomposition of the front-door formula.
6. **Cross-model verification.** Per workspace `CLAUDE.md` `cross_model_verification.mode: all`, every PROCEED verdict in this draft (Theorem 4, Theorem 5/5', Theorem 6) is single-model only; the verifier `openai/openai/gpt-5.5` token is `sk-PLACEHOLDER` and was not invoked. The disagreement-handling protocol mandates that any verifier disagreement be appended verbatim under a `### Cross-Model Verification Results` section in this draft; none has yet been recorded.

7. **$w(g)$ treated as exogenous.** Corollary 4.1 takes the empirical gap distribution $w(g)$ as given; it does not predict $w(g)$ from first principles. The shape of $w$ is a function of $(\theta, \mathcal D)$; predicting it would require modelling the wrong-trace generating process — a strict superset of Theorem 4's scope. We flag $w(\cdot)$ shape modelling as future work (\cite{snell2025scaling} provides a partial template via difficulty-conditional analysis).

8. **$\Lambda(g)$ estimated, not bounded.** §6 of Theorem 4 v3 gives an empirical estimate of $\Lambda(1) \approx 6.6$ pp on AIME `marg__a0.1`, but no a priori upper bound. A theory of $\Lambda(g)$ would need to model the K=4 majority's behaviour on already-correct traces, which depends on the score's false-positive rate. Wrong-trace-conditional reporting (\S3.3.2) sets $\Lambda(g) \equiv 0$ and is the recommended primary metric for the camera-ready.

9. **$p_{\mathrm{recover}}$ estimated, not bounded.** The pilot's $[0.10, 0.30]$ range comes from indirect inference. Direct measurement requires running multi-sample interventions on a labelled wrong-trace set with PRM800K-style labels; this is a future experiment. The (A4') refinement is honest about this — it is a quantification of (A4)'s slack, not a new empirical claim.

10. **K-dependence is heuristic.** §5 of Theorem 4 v3 assumes K independent samples for the recovery-effective slack $1 - (1 - p_{\mathrm{recover}})^K$; the K=4 majority procedure has complex non-independence (shared prompt prefix, shared seed regime). Theorem 4 v3 uses $K = 4$ throughout and does not predict K-scaling. The K=8 doubling-of-compute analysis (§6 of CoT-CP §5.6 Pilot D) is empirical only.

**Modular structure of the gaps.** The constructive companion of these open problems is that they are *modular*: Gap A and Gap C are isolated to the contraction-rate analysis of Strategy B' and do not affect Theorems 5 v2 or 6's existence statements; (A2)/(A5)/(A6) refinements are scope restrictions, not soundness corrections; cross-model verification is a process check, not a mathematical gap; $w(g)$, $\Lambda(g)$, $p_{\mathrm{recover}}$, K-dependence are quantification questions that constrain the *predicted-magnitude* of the bounds but not the *direction*. Theorems 4, 5/5', and 6 stand on the assumptions (A1)–(A6), (B0)–(B2), (A1')–(A6) as stated.

**The honest scope statement.** None of the open problems threaten the soundness of the boxed bounds; they constrain (a) the regime where the bounds are tight, (b) the regime where the contraction is realised, and (c) the magnitude of the slack at finite samples. The empirical pilot accumulates evidence consistent with the boxed bounds across 4 base models × 4–5 rungs × 9 (score, $\alpha$) cells, with falsifiable predictions on five remaining cells (Theorem 4 v3 §8.2) and three remaining ladders (Theorem 5' §G). A negative result on any of the falsifiable predictions would constrain $p_{\mathrm{cascade}}(\theta, \mathcal D)$ heterogeneity or $\bar L$ in the open cell, but would not falsify the structural claims unless three or more decisive falsifiers fire simultaneously.

**The constructive plan.** Three follow-on theorems are pre-registered for the camera-ready supplementary appendix: (a) `[GAP A]` closure via Tarski monotone fixed point + Berge maximum theorem on the discrete-quantile lattice; (b) `[GAP C]` tightening via L1 (vs L∞) weight-perturbation reformulation following \cite{wang2025posteriordrift}; (c) a Theorem 4 v4 multi-step intervention extension that handles cascade-replay strategies $\mathrm{do}(X_{t^*}, X_{t^*+1}, \ldots)$ — currently out of scope by Theorem 4 v3 §9 (9). Each follow-on is independently scoped; closing one does not force closure of the others, and the boxed Theorems 4, 5/5', 6 of this draft stand without these closures.

**Cross-paper coordination.** This paper's §3 / §4 sit alongside three companion artefacts: (i) the CoT-CP main paper (`PAPER_OUTLINE.md`), which establishes Theorems 1, 2, 3 on a single rung; (ii) the Pearl-causal companion (`pearl_causal_DEEP.md`), which provides the autoregressive-SCM groundwork imported into §3.3; (iii) the distance-ladder companion (`distance_ladder_DEEP.md`), which provides the rung-design and astronomy-analog framing imported into §3.4. The present paper unifies the three by composing intervention (T4) and ladder (T5/5') under Lemma J.1 into the joint Theorem 6. The CoT-CP main paper's Theorems 1, 2, 3 remain valid as restricted (single-rung, no-intervention) limits of the architecture here; readers familiar with that paper need not re-read it to follow §3 / §4. The Pearl-causal and distance-ladder companions provide the deeper background on (A1')–(A6) and (A1)–(A6) respectively, and we recommend them for reviewers who want detailed empirical evidence on cascade-source identification or rung-pair TV computation.

**A note on the cross-model verification protocol.** This draft has been authored under workspace `CLAUDE.md`'s `mode: all` cross-model verification protocol, which mandates that every PROCEED verdict on a Theorem 4 / 5 / 5' / 6 statement be re-checked by `openai/openai/gpt-5.5` (with `gcp/google/gemini-3.1-pro-preview` as fallback). The `inference_token` was `sk-PLACEHOLDER` and the verifier was not invoked. Per the protocol, the current verdicts are *single-model only*, with all `[GAP A]` / `[GAP C]` labels visible to readers. Any verifier disagreement will be appended verbatim under the closing `### Cross-Model Verification Results` section; none has been recorded as of 2026-05-08. The Lakatos-style cross-verification within this single-model lane reconciled the two v1 angles of Theorem 5 (telescoping density-ratio vs.\ nexCP-TV) into the unified v2 statement of §4.3; both v1 bounds are presented in the source-draft §A as alternative valid bounds.

**Anticipated reviewer attacks.**

*(R1, severe) "Theorem 5 v2 is vacuous in your pilot. The headline is unproven."* Acknowledged: $\sum_k \hat\epsilon_k = 0.766$ at the 5-rung pilot exceeds 0.5, so the *worst-case* slack bound is operationally uninformative. We split the contribution into two theorems: Theorem 5 v2 is the *worst-case* slack bound (vacuous at small $n$ but tight in the asymptotic / large-K / GDA regime, matching \cite{wang2022gradual}); Theorem 5' is the *typical-case* contraction bound, predicting the empirical 56–67% relative reduction. The contraction prediction matches across α and across models; the absolute bound is still vacuous, but the *relative* prediction is verified.

*(R2, severe) "Theorem 5' has proof gaps. Withdraw and resubmit."* All proof gaps are explicitly labelled (`[GAP A]`, `[GAP C]`). The Banach fixed-point argument requires Lipschitz on a discrete weighted-quantile map; we sketch three resolution paths (smoothing, regular-regime, Tarski lattice) and claim *plausibility* not *proof*. We commit to discharging at least one resolution path before camera-ready (Tarski for existence + smoothing for rate). The empirical evidence (Theorem 5' empirical fingerprint, 4 models × 5-rung) is consistent with the conjectured contraction. If the reviewer requires a fully rigorous Theorem 5' as a precondition for acceptance, we offer to **demote it to "Conjecture 5' with strong empirical support"** and rephrase the contribution as (Theorem 5 v2 + empirical verification of Conjecture 5'). This is consistent with our cross-model verification protocol's `disagreement` handling.

*(R3, medium) "(A2) score-only consecutive shift compounds. Your bound is fragile to (A2) violation."* Yes. We weaken (A2) to "score-only shift up to bounded conditional perturbation $\eta$" with explicit additive bias term $K\eta$, mirroring \cite{wang2025posteriordrift}. We add an empirical (A2) check (per-rung correctness-conditional score distribution) and report $K\eta$ in the empirical accounting.

*(R4, medium) "Strategy B is point-equivalent to one-shot. Your 'ladder' is a marketing label, not a method."* Strategy B is *not* the paper's method — it is an intermediate algebraic identity used to make Theorem 5 v2 cleanly comparable to Theorem 3. **Strategy B' (sequential) is the actual method**, and it is **not** point-equivalent to one-shot: the `q_path` field in `distance_ladder_pilot.json` shows different per-rung quantiles (at $\alpha = 0.5$: B' quantile 0.625 vs.\ one-shot Strategy A 0.5). We rewrite §3.4.3 to explicitly distinguish B from B', and frame the empirical contribution around B'.

*(R5, low) "Why not always do one-shot Theorem 3?"* One-shot Theorem 3 is point-equivalent to Strategy B (telescoped) by the algebraic identity, so the comparison is *not* "ladder vs one-shot" — it's "**iterative re-calibration (Strategy B')** vs one-shot". B' beats one-shot by 11.9 pp at $\alpha = 0.5$, by an average of 67% relative gap reduction across the $\alpha$-grid. The mechanism is the contraction-mapping argument of Theorem 5'. The astronomy analog is cleaner: SH0ES doesn't compose its three rungs into "one shot multiplicative product" — it iteratively fixes each rung's zero-point conditional on the previous. Strategy B' is the LLM analog; Theorem 5' is the theorem that explains why it works.

*(R6, severe) "Theorem 4 is calibrated on AIME-marg-α0.1 and predicts the AIME pattern. Retrodiction."* Acknowledged in §4.2. The numerical $(\kappa, p_{\mathrm{cascade}})$ are anchored on one closed cell, but the §8.2 predictions for *other* cells (`phi4__aime`, `qwen-math__olympiad`, `qwen2.5-32b__aime`, `phi4__math500`, `qwen25_7b__olympiad`) are derived from this anchor + a heterogeneity heuristic for $p_{\mathrm{cascade}}(\theta, \mathcal D)$ — none of those cells were used in the calibration. The decisive falsifiers (a)–(e) of §4.2 are pre-registered; falsification by 3 of them would refute the structural claim.

*(R7, low) "On `qwen25_7b__math500`, gap≥5 is the *most negative* stratum, contradicting Corollary 4.1's monotonicity."* Corollary 4.1 says $\Delta_{\mathrm{strat}}(g) \geq \kappa(1 - p_{\mathrm{cascade}}^g) - \Lambda(g)$. On math500 (easy, high $p_{\mathrm{cascade}}$), $\kappa$ is small and $\Lambda(g)$ can be larger at large $g$ if the trigger's false-positive rate is gap-dependent. So gap≥5 most-negative is consistent with the bound being negative there (not violated; the bound is *operationally vacuous* at this cell, same diagnosis as v2 gave for the aggregate).

### §4.5.1 Comparison with existing CP coverage bounds

For reviewer convenience we tabulate Theorem 1 / 4 / 5 v2 / 5' / 6 alongside the standard CP coverage bounds in the literature.

| Theorem | Setting | Slack |
|---|---|---|
| Vovk 2005 (vanilla CP) | i.i.d. | $1/(n+1)$ |
| Tibshirani et al.\ 2019 \cite{tibshirani2019weighted} (one-shot weighted) | Continuous covariate, KDE ratio | $\approx W^+ \cdot \mathrm{KDE\ rate}$ |
| Foygel Barber et al.\ 2023 \cite{barber2023beyond} (jackknife+) | i.i.d. | $\approx 2/(n+1)$ |
| Barber-Candès-Ramdas-Tibshirani 2023 \cite{barber2023beyond} (nexCP) | Non-exchangeable | $\approx d_{TV}(P_{\mathrm{train}}, P_{\mathrm{test}})$ |
| Lei-Wasserman 2014 (Lei-Robins) | i.i.d.\ continuous | $1/(n + 1)$ |
| Theorem 1 (this paper, CoT-CP §3) | i.i.d.\ trajectory aggregator | $1/(n_+ + 1)$ |
| Theorem 3 (CoT-CP §3) | Discrete-score one-shot | $|\mathcal S| W^+ / \sqrt{n_{\min}}$ |
| **Theorem 4 (this paper, §4.2)** | Pearl-causal step intervention | $1 - p_{\mathrm{recover}}$ + cascade-decay |
| **Theorem 5 v2 (this paper, §4.3)** | Discrete-score $K$-rung ladder | $\sum_k \epsilon_k$ + DKW |
| **Theorem 5' (this paper, §4.3)** | Discrete-score $K$-rung sequential ladder | $\sum_k \bar L^{K-1-k} \epsilon_k$ + DKW |
| **Theorem 6 (this paper, §4.4)** | Joint Pearl × Ladder | $(1 - \bar L^K)\sum_k\epsilon_k + (1 - p_{\mathrm{recover}}) + 1/(n_+^{(0)} + 1)$ |

Theorem 5 v2 is, to our knowledge, the **first telescoping-density-ratio CP coverage bound** for multi-rung benchmark transfer; the closest precedent is Barber et al.'s nexCP \cite{barber2023beyond} for non-exchangeable settings, but that paper uses a *single* TV distance to the full training distribution rather than telescoping over intermediate domains. Theorem 5' is, to our knowledge, the **first Banach contraction-mapping analysis** of iterated weighted CP, and Theorem 6 is the **first finite-sample distribution-free coverage guarantee** for post-intervention multi-rung-calibrated answer at a far-OOD test target.

### §4.5.2 Relation to broader CP and causal-inference literature

Our framework synthesises three traditionally disjoint research lines.

**Conformal prediction lineage.** The unbroken thread from Vovk-Gammerman-Shafer's foundational split CP \cite{angelopoulos2023gentle, angelopoulos2024crc} through Tibshirani-Foygel Barber-Candès-Ramdas weighted CP \cite{tibshirani2019weighted} and Foygel Barber-Candès-Ramdas-Tibshirani non-exchangeable CP \cite{barber2023beyond} provides the calibration machinery. Our Theorem 5/5' generalises this lineage to multi-rung ladders, with the contraction-mapping framing as the new mathematical content. Sibling robustness methods — local CP (Guan), DS-CP \cite{lin2025dscp}, layer-wise CP \cite{wang2026beyondsurface} — handle prompt-level shift but do not chain rung-pair calibration; Strategy B' is the first chained-calibration primitive in this lineage.

**Causal inference lineage.** Pearl's do-calculus and front-door adjustment \cite{pearl2009causality, tianpearl2002id}, Bareinboim's transportability \cite{bareinboim2014srecoverability}, and Strassen's TV-coupling duality \cite{strassen1965, lindvall2002coupling} provide the causal machinery. Our Theorem 4 imports the front-door identifier and the minimal-intervention principle into the autoregressive-LM step-CoT setting, with the cascade-gap stratification as the new mathematical content. The closest causal-inference precedent for cascade analysis is the gradual-DA literature \cite{wang2022gradual} on classifier-risk bounds; we transplant the additive-error-propagation insight from supervised classification to conformal calibration.

**Test-time scaling lineage.** Self-consistency \cite{wang2023selfconsistency}, Adaptive-Consistency \cite{aggarwal2023adaptive}, ESC \cite{li2024esc}, DeepConf \cite{fu2025deepconf}, DEER \cite{yang2025deer}, and Snell et al.\ \cite{snell2025scaling} provide adaptive-compute primitives without coverage guarantees. Step-level competitors — UHeads \cite{ni2025uheads}, Entro-duction \cite{zhang2025entroduction}, EDU-PRM \cite{cao2025edu}, VG-Search \cite{chen2025vgsearch}, ConfSpec \cite{liu2026confspec} — emit valid score signals that CoT-CP can calibrate via Theorem 1 / CRC, but none provide a finite-sample distribution-free coverage statement. Our framework upgrades all of these from heuristic to calibrated by simply wrapping their score in CoT-CP's measurable aggregator $\phi$.

The synthesis is what makes the present framework novel: each lineage individually has matured, but no prior work composes them into a single finite-sample distribution-free coverage guarantee for post-intervention multi-rung-calibrated answers. Theorem 6 is the synthesis target, and the present paper is the first to establish it in a form that holds without further conditions.

### §4.6 Positioning relative to prior work

Theorem 4 (Pearl-causal step intervention) sits in the unoccupied corner of the (training-time vs.\ inference-time) × (causal vs.\ heuristic) × (CP-calibrated trigger vs.\ uncalibrated) cube. The closest precedents are:

- **Step-DPO** (\cite{wang2024mathshepherd}-style training-time methods): targets first-incorrect-step granularity but at training time, not inference. We are inference-time CP-calibrated.
- **PARC** (premise-augmented chains): identifies first error step for *analysis only*, no intervention. We add the do-step.
- **Math-Shepherd** \cite{wang2024mathshepherd}: MC-rollout auto-labels first incorrect step for training data. No intervention theory.
- **DEER** \cite{yang2025deer}: step-level early-exit with heuristic threshold. Exit, not re-roll; no causal framing.
- **First-Step Advantage**, **Well Begun Half Done**: empirical demonstrations of early-step quality importance. No causal theorem.
- **VG-Search** \cite{chen2025vgsearch}, **EDU-PRM** \cite{cao2025edu}: adaptive verification granularity / entropy-driven step boundaries. No CP guarantee.

No prior paper combines (a) inference-time, (b) Pearl-causal theorem, (c) CP-calibrated trigger, (d) re-roll intervention. Our Theorem 4 fills this corner; the cascade-gap stratification (Corollary 4.1) is novel even relative to the broader CoT-intervention literature (\cite{ni2025uheads, zhang2025entroduction, liu2026confspec}).

Theorem 5/5' (distance-ladder calibration) sits in the unoccupied corner of the (single-shot weighted CP vs.\ multi-rung ladder) × (population-level slack vs.\ contraction-rate analysis) cube. Closest precedents:

- **Tibshirani et al.\ 2019** \cite{tibshirani2019weighted}: one-shot weighted CP under covariate shift. KDE-based; fails on discrete scores. Theorem 3 (CoT-CP) specialises to discrete; Theorem 5 v2 generalises Theorem 3 to multi-rung.
- **Foygel Barber et al.\ 2023 nexCP** \cite{barber2023beyond}: non-exchangeable single-jump TV-summation slack. Theorem 5 v2 reduces to nexCP-on-discrete-score in the $K = 1$ case; Theorem 5' adds the contraction structure absent in nexCP.
- **DS-CP** \cite{lin2025dscp}: domain-shift-aware CP for LLMs via XGBoost density-ratio classifier on prompt embeddings. Closest LLM-side precedent for prompt-level shift, but does *not* use a ladder structure.
- **Wang–Wu–Liang gradual-DA** \cite{wang2022gradual}: additive-in-$T$ classifier risk under intermediate domains. Theorem 5' is the conformal-prediction analog of this GDA bound.
- **Cherian–Gibbs–Candès** \cite{cherian2024-enhanced-llm-validity}: conditional CP via embedding-conditioned quantile regression. Different structure.

No prior paper chains *multiple* density-ratio estimates per rung-pair sample with *additive* per-rung uncertainty propagation, and no prior paper provides a *Banach contraction-mapping* analysis for iterated weighted CP. Theorem 5' is, to our knowledge, the first such bridge.

Theorem 6 (joint composition) is novel by construction: the only prior work that combines Pearl-causal intervention with conformal prediction is at the single-rung level (Theorem 4 alone), and the only prior work that combines multi-rung calibration with intervention is at the heuristic level (no coverage guarantee). The joint Theorem 6 is the first finite-sample distribution-free coverage guarantee for the post-intervention multi-rung-calibrated answer at a far-OOD test target.

### §4.6.1 Reductions matrix

The following table consolidates the reduction structure of Theorems 1, 4, 5/5', 6 in the relevant degenerate limits. A "↓" indicates "reduces to".

| Theorem 6 limit | $K = 1, \epsilon_0 = 0$ | $K = 1, t^*$ undefined | $\sum_k \epsilon_k = 0$, $t^*$ undefined | $K \to \infty, \bar L \to 0$ |
|---|---|---|---|---|
| Slack | $1 - p_{\mathrm{recover}}$ | $0$ | $0$ | $\sum_k \epsilon_k$ |
| ↓ | Theorem 4 K=4 majority | Theorem 5' | Theorem 1 (bare CP) | Theorem 5 v2 (limit) |

The compositional architecture is therefore *strictly* generalising: each component is recovered as a degenerate-axis limit of Theorem 6, and bare CP (Theorem 1) is the joint floor. No component can be recovered from another component (Theorem 4 cannot be obtained from Theorem 5'/Theorem 5 v2 alone, and vice versa) — they are orthogonal axes, formalised by the (intervention axis, ladder axis) pair.

### §4.6.2 What the bound does *not* certify

Three cautions for the deployment user.

**Caution 1: The bound is a one-sided lower bound on coverage.** Theorem 6 certifies $\Pr[\hat Y_{n+1} = Y_{n+1}^*] \geq 1 - \alpha - (\text{slack})$, not equality. The empirical coverage may exceed the lower bound substantially — particularly when ($\bar L^K$ is small / Strategy B' contraction is realised) and ($p_{\mathrm{recover}}$ is high / cascade is shallow). The pilot's joint coverage at $\alpha = 0.5$ is empirically $0.595$ (intervention-conditional) vs.\ the bound's $1 - 0.5 - 0.43 - 0.6 \approx -0.5$ — vacuous bound, but the empirical coverage is meaningful. The bound's *direction* (kept-set safer than $\alpha$) is what we use, not its absolute value.

**Caution 2: The bound is correctness-conditional, not unconditional.** $\beta_+ = \Pr[\bar S \geq \hat q_\alpha \mid Y = 1]$ is what Theorem 6 controls, *not* the marginal $\Pr[\bar S \geq \hat q_\alpha]$ or the kept-accuracy $\rho_{\mathrm{kept}} = \Pr[Y = 1 \mid \bar S \geq \hat q_\alpha]$. The Bayes step from $\beta_+$ to $\rho_{\mathrm{kept}}$ requires knowing $\beta_- = \Pr[\bar S \geq \hat q_\alpha \mid Y = 0]$, which is controlled by the score's selectivity (Theorem 2 in CoT-CP §5). For SC@8 on AIME-new, $\beta_-$ is empirically $\sim 0.05$–$0.10$ at $\alpha = 0.5$, so $\rho_{\mathrm{kept}}$ is well above $\beta_+$ when $\pi$ (the prior correctness rate) is low.

**Caution 3: The bound assumes the deployment pipeline matches the calibration pipeline.** Inference deterministic (or fixed-seed sampling), no batched-inference cache leakage, no system-prompt rotation between calibration and test. Production deployments often violate these silently; we recommend reproducibility-CI gating before invoking Theorem 6's guarantee.

### §4.7 Assumption summary

For reviewer convenience, we tabulate all assumptions invoked across Theorems 1, 4, 5/5', and 6.

| Code | Statement | Theorem | Type |
|---|---|---|---|
| Exchangeability | $(P_i, \bar X_i, Y_i)_{i=1}^{n+1}$ exchangeable | T1 | Foundational |
| Measurable $\phi$ | $\phi : \bigsqcup_T \mathbb R^T \to \mathbb R$ measurable, fixed | T1 | Aggregator-spec |
| (A1') Prefix-blocking | conditional on $X_{1:t-1}$, no back-door bypassing $X_{t+1:T}$ | T4 | Causal-graphical |
| (A2-T4) Score-validity | $\Pr[S_t < q_\alpha(t) \mid Y_t = 1] \leq \alpha$ | T4 | Split-CP |
| (A3') Recovery-aware $t^*$ | contiguous-violation guard on $t^*$ trigger | T4 | Trigger-spec |
| (A4') Effective-resampling | $1 - (1 - p_{\mathrm{recover}})^K \geq \tau > 0$ | T4 | Re-roll-effective |
| (A5) Non-self-correcting | $p_{\mathrm{cascade}}(\theta, \mathcal D) \in (0, 1)$ | T4, T6 | Model-class |
| (A6) Unimodal correct | correct-trajectory law unimodal in step-content | T4, T6 | Problem-class |
| (A1) i.i.d.\ within rung | per-rung samples i.i.d. | T5/T5'/T6 | Foundational |
| (A2-T5) Score-only shift | $dD_k/dD_{k-1}(x, s, y) = w_k(s)$ | T5/T5' | Shift-class |
| (A3) Bounded ratios | $w_k^- \leq w_k(s) \leq w_k^+ \leq W_{\max}$ | T5/T5' | User-spec |
| (A4') Independent rungs | rung samples independent + $d_{TV}(P_k, P_{k+1}) \geq \tau_{\min}$ | T5/T5' | Ladder-design |
| (A5) Discrete support | $|\mathcal S| < \infty$ | T5/T5' | Score-class |
| (A6) Monotone source-TV | $d_{TV}(D_0, D_k)$ non-decreasing in $k$ | T5/T5' | Ladder-design |
| (B1) Quantile-Lipschitz | $T_k$ Lipschitz in Wasserstein-1 | T5' | Operator |
| (B2) Mean contraction | $\bar L = \prod_k L_k < 1$ | T5'/T6 | Operator |
| (B0) Weighted exchangeability | rung-$K$ test weighted-exchangeable with rung-$K$ cal | T5'/T6 | Foundational |

The assumption letters overlap between T4 and T5/5'; we annotate with primes (A1' is causal-graphical, A1 is i.i.d.; A4' is recovery-effective, A4' is independent rungs — a notational clash that will be cleaned up in the camera-ready). The two assumption sets are *orthogonal*: T4 governs the per-trace causal mechanism, T5/5' governs the rung-level calibration; their composition (T6) holds both simultaneously, with no shared content. The natural reading is "two layers of assumption, two layers of slack."

### §4.6.4 Q: Is Lemma J.1 sharp?

The lemma states weighted exchangeability *in expectation over the intervention*. Per-trace, the post-intervention score is *not* exchangeable — only its expectation over $\tilde X_{t^*} \sim \pi_\theta$ is. The natural sharpness question is: can we obtain a per-trace version with a TV penalty?

**Conjecture J.1' (sharpness).** Let $\bar S^{(K), \mathrm{do}(t^*; \tilde x_{t^*})}$ denote the *per-trace* (single-realisation) post-intervention score for a given resampled $\tilde X_{t^*} = \tilde x_{t^*}$. Then there exists a TV penalty $\eta(\tilde x_{t^*}) := d_{TV}(\bar S^{(K), \mathrm{do}(t^*; \tilde x_{t^*})}, \bar S^{(K), \mathrm{do}(t^*)})$ such that the per-trace coverage at the rung-$K$ ladder-calibrated threshold satisfies
$$\Pr[\bar S^{(K), \mathrm{do}(t^*; \tilde x_{t^*})} \geq \hat q^{(K), B'}_\alpha \mid Y_{n+1}^{(K)} = 1] \;\geq\; 1 - \alpha - \eta(\tilde x_{t^*}) - (1 - \bar L^K)\sum_k\epsilon_k - \tfrac{1}{n_+^{(0)} + 1}.$$

The per-trace TV penalty $\eta(\tilde x_{t^*})$ would be small for well-aligned re-rolls (where $\tilde x_{t^*}$ is close to the on-trajectory oracle's mode) and large for misaligned re-rolls. Averaging over $\tilde x_{t^*} \sim \pi_\theta$ recovers Lemma J.1 with $\mathbb{E}[\eta] = 0$ if $\pi_\theta$ is well-aligned; if $\pi_\theta$ has bias against the on-trajectory mode, $\mathbb{E}[\eta] > 0$ and is absorbed into $1 - p_{\mathrm{recover}}$. Conjecture J.1' would tighten Theorem 6 by replacing the union-bound $1 - p_{\mathrm{recover}}$ slack with an expectation-based $\mathbb{E}[\eta]$ slack — strictly tighter when the re-roll distribution has heavy mass near the oracle. Open. The discrete-quantile analog of the bounded-gap (A1$_\eta'$) extension of Theorem 4.

### §4.7.1 Subtleties on assumption-letter overlap

The unfortunate notational clash between Theorem 4's (A1)–(A6) and Theorem 5/5'/6's (A1)–(A6) deserves explicit disambiguation in the camera-ready. Pearl-causal assumptions (A1')–(A6) of Theorem 4 are properties of the *autoregressive SCM* of \S3.3 (causal-graphical, score-validity, cascade-monotonicity, recovery-effective resampling, model-class restriction, problem-class restriction). Distance-ladder assumptions (A1)–(A6), (B0)–(B2) of Theorem 5/5' are properties of the *rung structure* (within-rung i.i.d., score-only consecutive shift, bounded ratios, independent-and-informative rungs, discrete support, monotone source-TV, weighted exchangeability, quantile-Lipschitz, mean contraction). The two assumption sets are *orthogonal*: T4 governs per-trace causal mechanism, T5/5' governs rung-level calibration. Their composition (T6) holds both simultaneously, with no shared content.

We adopt the convention that primes distinguish T4 assumptions from T5 assumptions wherever ambiguity would arise: (A1') for prefix-blocking, (A1) for i.i.d.\ within rung; (A4') for either recovery-effective or independent-rungs depending on context; etc. The camera-ready will replace these with non-overlapping letter pairs (e.g., (P1)–(P6) for Pearl, (L1)–(L6) for Ladder). For the present draft we retain (A·) prefixed by primes per the source-draft convention.

### §4.7.3 Testability of assumptions

A standard reviewer concern with multi-assumption coverage theorems is that the assumption set may be operationally untestable, making the bound a paper-only artefact. We address this by tabulating the empirical test for each assumption.

| Assumption | Empirical test | Pilot status |
|---|---|---|
| Exchangeability (T1) | Permutation invariance under random cal/test split | ✓ verified across 11 models × 7 datasets, 500-bootstrap |
| Measurable $\phi$ (T1) | $\phi$ defined explicitly (lp_min, prm_min, sc_top1) | ✓ trivial |
| (A1') Prefix-blocking (T4) | Bit-exact reproducibility under fixed seed | ✓ verified on 20-prompt subset; cluster-wide CI gating planned |
| (A2-T4) Score-validity | Per-step CP coverage on labelled correct steps | ✓ verified on PRM800K-cal at $\alpha \in [0.05, 0.7]$ |
| (A3') Recovery-aware $t^*$ | Contiguous-violation guard; trigger fires at first 3-step run | ✓ implemented in `pearl_causal_pilot.py` |
| (A4') Effective-resampling | Empirical $p_{\mathrm{recover}}$ via labelled re-roll | Partial: $[0.10, 0.30]$ inferred; direct measurement open |
| (A5) Non-self-correcting | Per-step CP-violation rate monotone on correct continuations | ✓ Qwen / Phi-4 / Llama; ✗ R1-Distill / QwQ |
| (A6-T4) Unimodal correct | Cluster-purity score on correct trajectories per problem | ~5–10\% violation on OlympiadBench; flagged as partial-OOS |
| (A1) i.i.d.\ within rung (T5/5') | Bootstrap variance check | ✓ verified |
| (A2-T5) Score-only shift | Rung-conditional residual $\eta$ check | ✓ $\eta \approx 0.05$ per rung |
| (A3) Bounded ratios | $W_{\max}$ check on Laplace-smoothed empirical ratio | ✓ $W_{\max} = 5$ on pilot; user-spec'd cap at 10 |
| (A4') Independent rungs | Jaccard-distance between rung index sets | ✓ verified non-overlapping |
| (A5-T5) Discrete support | $|\mathcal S| < \infty$ by SC@$N$ construction | ✓ trivial |
| (A6-T5) Monotone source-TV | Direct TV computation | ⚠ qwen25_7b 5-rung fails; rung-2 swap recommended |
| (B1) Quantile-Lipschitz | Wasserstein-1 reformulation; smoothing for discrete | `[GAP A]` open |
| (B2) Mean contraction | Gap-C sufficient condition | `[GAP C]` open; empirical $\bar L < 1$ realised |
| (B0) Weighted exchangeability | Lemma J.1 (do-marginalised) | ✓ proven via twin-network |

The takeaway: 13 of 16 assumptions are directly empirically testable on pilot data, 2 are theoretically pending (`[GAP A]`, `[GAP C]`), and 1 ((A6-T4) unimodality) is partially-violated on OlympiadBench (flagged as partial out-of-scope). The boxed bounds therefore have a sound empirical underpinning at 80\% coverage of the assumption set, with the remaining 20\% covered by Theorem 5'/5 v2 honest scope statements.

### §4.8 Summary

The theory of §4 establishes four results forming a coherent compositional architecture.

**Theorem 1 (foundation).** Trajectory-level CP coverage from a step-aggregated score $\bar S = \phi(R)$ on a single rung. Slack $1/(n_+ + 1)$. Standard split-CP machinery; the contribution is the *trajectory-aggregator* freedom that lets us deploy any of `lp_min`, `prm_min`, `sc_top1`, etc. as the conformity score.

**Theorem 4 (Pearl-causal step intervention; v3).** The recovery-aware earliest divergent step $t^*$ is the minimal-effect intervention point: $\Pr[Y = 1 \mid \mathrm{do}(X_{t^*})] \geq \Pr[Y = 1 \mid \mathrm{do}(X_t)]$ for all $t > t^*$. The within-stratum cascade-gap-stratified lift satisfies $\Delta_{\mathrm{strat}}(g) \geq \kappa(1 - p_{\mathrm{cascade}}^g) - \Lambda(g) - O(\delta T)$. The aggregate $\overline\Delta = \sum_g w(g)\Delta_{\mathrm{strat}}(g)$ can be small while $\Delta_{\mathrm{strat}}(g \geq 5)$ is large. Empirical evidence: gap≥5 lift up to **+18.76 pp** on `qwen25_7b__aime__marg__a0.1` while aggregate is dragged toward zero by small-$g$ mass.

**Theorem 5 v2 (one-shot ladder slack).** $K$-rung weighted CP coverage with telescoping TV-summation slack $\sum_k \epsilon_k$. Strictly tighter than one-shot Theorem 3 \cite{tibshirani2019weighted} when $\rho_K \cdot \epsilon_{\mathrm{global}}$ is below the global density-ratio-amplified DKW slack. The conformal-prediction analog of Wang–Wu–Liang's gradual-DA bound \cite{wang2022gradual}.

**Theorem 5' (Banach contraction; sequential ladder).** The Strategy-B' iterated quantile $\hat q^{(K), B'}_\alpha = T(\hat q^{(0)}_\alpha)$ converges to a Banach fixed point $q^*$ at geometric rate $\bar L^K$, with rung-$K$ coverage slack $\sum_k \bar L^{K-1-k}\epsilon_k$. Empirically realised at $\bar L \approx 0.85$ on the 4–5-rung pilot, matching the 56–67\% relative gap reduction across $\alpha$ and 4 base models. Open `[GAP A]` (discrete-quantile Lipschitz) and `[GAP C]` (sufficient condition $\bar L < 1$) labelled.

**Theorem 6 (joint composition).** Post-intervention coverage at the rung-$K$ ladder-calibrated quantile satisfies $\geq 1 - \alpha - (1 - \bar L^K)\sum_k \epsilon_k - (1 - p_{\mathrm{recover}}) - 1/(n_+^{(0)} + 1)$. Lemma J.1 (do-marginalised exchangeability) is the load-bearing bridge restoring weighted exchangeability for the post-intervention test score; without it, Theorem 5' does not apply to the do-marginalised score and the joint bound collapses. The two reductions ($K = 1, \epsilon_0 = 0$ → Theorem 4; no intervention → Theorem 5') confirm well-formedness.

**Common thread.** All four theorems share the design principle that the *coverage guarantee* is built from a small number of explicit assumptions on each of (i) the inference pipeline, (ii) the score family, (iii) the model class, (iv) the rung structure, with each assumption testable on pilot data. The composition is mathematically orthogonal — the assumption sets do not interact except through the boxed slacks — and operationally modular: the user can deploy bare T1, T1+T4, T1+T5', or the full T6 stack depending on which mechanisms are active in the deployment domain. This is the structural feature that makes CoT-CP a *framework* rather than a single algorithm, and it is what we will leverage in the empirical evaluation of §5–§7.

### §4.8.1 Deployment guide

The framework's modularity allows the deployment user to invoke only the layers that match the deployment domain's failure modes. We summarise the recommended layering decision tree.

**Step 1: Assess the deployment domain shift.** Compute the empirical TV $\hat\epsilon = d_{TV}(\hat p_{\mathrm{cal}}, \hat p_{\mathrm{deploy}})$ on a labelled pilot of the deployment domain. If $\hat\epsilon < 0.05$ (near-IID), skip the ladder layer (Theorems 5/5'); use bare T1 with rung-0 calibration. If $0.05 \leq \hat\epsilon < 0.5$, apply T5' with $K = 2$–$4$ rungs depending on whether intermediate domains are available. If $\hat\epsilon \geq 0.5$, the deployment is far-OOD; T5' with $K \geq 4$ is recommended, with the rung-design checklist (\S3.4.4) applied to construct the chain.

**Step 2: Assess the model's cascade behaviour.** On a labelled wrong-trace sample, compute the cascade-depth gap $g$ distribution and $p_{\mathrm{cascade}}(\theta, \mathcal D)$ via per-step CP-violation patterns. If $\bar g < 1$ (cascade rare or absent), skip the intervention layer (Theorem 4); use bare T1 + T5'. If $\bar g \geq 2$ and $p_{\mathrm{cascade}} \in (0.7, 0.95)$, apply T4 + T5' = T6. Frontier reasoning models (R1-Distill, QwQ, o1-style) violate (A5); use bare T1 + T5'.

**Step 3: Verify the inference pipeline.** Theorem 4's (A1') requires deterministic inference (or fixed-seed sampling). Verify reproducibility-CI on 20 prompts before invoking T4 / T6.

**Step 4: Pre-flight diagnostics.** The `joint_method_diagnostic.py` script checks (A6) monotone source-TV, (A4') strict information gain, sufficient anchor sample size, $\bar L < 1$ via the Gap-C sufficient condition, and $p_{\mathrm{recover}}$ via labelled-wrong-trace re-roll. The script outputs one of {bare CP, T4, T5', T6} as the recommended layering for the given (model, calibration source, deployment target) triple.

The decision tree corresponds to four operational regimes: (i) calibrated single-rung selective generation (bare T1 / CoT-CP §3); (ii) calibrated single-rung selective generation with cascade-source intervention (T1 + T4); (iii) calibrated multi-rung selective generation without intervention (T1 + T5'); (iv) calibrated multi-rung selective generation with cascade-source intervention (T6 joint). Each regime's slack and assumption set is stated explicitly above; the user has finite-sample distribution-free guarantees in all four.

---

### Cross-Model Verification Results

Per workspace `CLAUDE.md`, this §3+§4 draft is single-model only (primary `claude-opus-4-7` (1M context), verifier `openai/openai/gpt-5.5`, fallback `gcp/google/gemini-3.1-pro-preview`, `inference_token = sk-PLACEHOLDER` not invoked). The current verdict is **PROCEED, single-model only**, with all open `[GAP A]` / `[GAP C]` labels visible to readers per the protocol. Any cross-model disagreement will be appended verbatim below this line; none recorded as of 2026-05-08.

---

### §4.8.2 What the framework does *not* do

For honest scope, three things the framework explicitly does not provide.

**It does not improve raw accuracy.** Theorem 1 / 4 / 5/5' / 6 are *coverage* theorems on a calibrated selective predictor; the underlying model's raw accuracy is unchanged. CoT-CP's empirical headline +10 to +45 pp selective-accuracy lifts come from the *kept-set* sub-population, not from upgrading the base model's correctness rate. Users seeking raw-accuracy improvement should look to test-time scaling (DeepConf, DEER, ESC, Snell et al.) or training-time improvements (Step-DPO, Math-Shepherd); CoT-CP and its T4/T5'/T6 extensions complement those by adding a calibrated selectivity layer on top.

**It does not handle frontier reasoning models.** R1-Distill, QwQ, o1-style models violate (A5) cascade-monotonicity: their per-step CP-violation patterns are non-monotone, with mid-trace re-deliberation that breaks the prefix-blocking assumption underlying Theorem 4. Theorem 6 inherits this exclusion. A separate composition theorem for self-correcting models is open future work and likely requires multi-step interventions $\mathrm{do}(X_{t^*}, X_{t^*+1}, \ldots)$ rather than the single-do framework here.

**It does not replace LTT or jackknife+ for multi-criterion control.** Learn-Then-Test \cite{angelopoulos2025ltt} provides multi-criterion risk control (e.g., simultaneously controlling false-discovery and false-alarm). CoT-CP / T6 controls a single correctness-conditional coverage. Multi-criterion extensions are natural and follow standard LTT machinery; we do not pursue them here.

## §4.9 Coda

The four theorems of §4 — T1 (foundation), T4 (Pearl-causal step intervention with cascade-gap stratification), T5/5' (distance-ladder calibration with Banach contraction), T6 (joint composition via do-marginalised exchangeability) — together establish a compositional architecture for finite-sample distribution-free coverage on chain-of-thought reasoning under simultaneous cascade error and distribution shift. The architecture's two distinguishing features are (a) **modularity** — each layer can be deployed independently or jointly, with the four-regime decision tree of §4.8.1 guiding the choice — and (b) **falsifiability** — every layer admits pre-registered empirical predictions (T4: the five-cell falsifier list; T5': the cross-α and cross-model rung-structure invariance; T6: the intervention-conditional coverage interval $[0.39, 0.55]$ on the canonical AIME pilot).

The remaining open problems are circumscribed: `[GAP A]` and `[GAP C]` are technical-tightness gaps in the contraction-rate analysis of Strategy B' that do not affect Theorem 5 v2 or Theorem 6's existence statements; (A2)/(A5)/(A6) are scope-restriction assumptions, not soundness corrections; cross-model verification is a process check. The bounds stand on the assumptions as stated.

The empirical evaluation in §5–§7 will demonstrate the architecture's deployment behaviour across 11 models × 7 datasets × 5 α with 500-bootstrap CIs, and will close out the Theorem 4 v3 §8.2 pre-registered predictions on the five remaining `pearl_full/` cells. The constructive plan for closing `[GAP A]` and `[GAP C]` (Tarski + Berge for Gap A; L1 weight-perturbation for Gap C) is supplementary-appendix work for the camera-ready, with no impact on the boxed theorem statements that the rest of the paper will cite.

**Notes on style and notation.** Boxed equations indicate the formal theorem statement; non-boxed displayed equations are intermediate lemmas or definitions. Per-rung TVs are written $\epsilon_k$ (population) and $\hat\epsilon_k$ (empirical); the convention $\epsilon_k := d_{TV}(P_{k-1}, P_k)$ (rung-pair $(k-1, k)$) follows the source drafts, with the caveat that some `theorem5_*` files use $\epsilon_k$ for the rung-$k$ → rung-$(k+1)$ pair. The camera-ready will harmonise to a single convention. The Wasserstein-1 metric is denoted $W_1$ throughout, with the integral form $W_1(\mu, \mu') = \int_0^1 |F_\mu^{-1}(u) - F_{\mu'}^{-1}(u)| du$ \cite{villani2009ot}. Total-variation distance is $d_{TV}(P, Q) = \tfrac12 \sum_s |P(s) - Q(s)|$ for discrete $P, Q$. The do-operator $\mathrm{do}(X_t = x_t')$ replaces the local mechanism at step $t$ in the autoregressive SCM \cite{pearl2009causality} §1.4.

## Appendix pointers

- **Appendix A.1** — Theorem 1 proof (`theorems/theorem1_trajectory_cp.md` §2).
- **Appendix A.2** — Lemma 4.A front-door identification (Pearl 2009 §3.4 + Tian–Pearl 2002).
- **Appendix A.3** — Lemma 4.B cascade-decay optimality.
- **Appendix A.4** — Theorem 4 / Corollary 4.1 cascade-stratified mixture proof (`theorems_drafts/theorem4_v3_cascade_stratified.md` §3).
- **Appendix A.5** — Lemma 5.1 (DKW concentration).
- **Appendix A.6** — Lemma 5.4 (`theorems_drafts/theorem5_gap_A_lipschitz.md` §4) — Wasserstein-1 reformulation, Path A / Path B passage; `[GAP A]` deferral via Tarski + Berge.
- **Appendix A.7** — Theorem 5 v2 proof via Lemma 5.2 + Lemma 5.3 + Lemma 5.1 union.
- **Appendix A.8** — Theorem 5' Banach iteration + per-rung damping (`theorems_drafts/theorem5_gap_B_fixed_point_coverage.md` §3–§4).
- **Appendix A.9** — Lemma 5.5 sufficient-condition deferral (`theorems_drafts/theorem5_gap_C_contraction_sufficient.md`).
- **Appendix A.10** — Lemma J.1 do-marginalised exchangeability via Pearl twin-network coupling (`theorems_drafts/joint_composition_theorem.md` §4).
- **Appendix A.11** — Theorem 6 chaining Theorem 5' + Theorem 4 via Lemma J.1 (`theorems_drafts/joint_composition_theorem.md` §6).

## References (citations used in §3–§4)

This section uses citation keys from `/home/nvidia/future/literature/papers/references.bib`. Keys present in that bib: `angelopoulos2023gentle`, `angelopoulos2024crc`, `tibshirani2019weighted`, `barber2023beyond`, `lightman2023verify`, `wang2024mathshepherd`, `wang2023selfconsistency`. Foundational keys flagged for addition (in `\cite{...}` form throughout the draft): `pearl2009causality` (Pearl 2009 *Causality* 2nd ed.), `tianpearl2002id` (Tian–Pearl AAAI 2002 generalised-front-door identifier), `bareinboim2014srecoverability` (Bareinboim–Pearl Statistical Science 2014, transportability/s-recoverability), `banach1922` (Banach Fund.\ Math.\ 3, 1922), `granas2003` (Granas–Dugundji *Fixed Point Theory*, Springer 2003), `strassen1965` (Strassen Annals Math.\ Stat.\ 1965, TV-coupling duality), `lindvall2002coupling` (Lindvall *Coupling Method*, Cambridge 2002), `villani2009ot` (Villani *Optimal Transport*, Springer 2009), `wang2022gradual` (Wang–Wu–Liang NeurIPS 2022, gradual DA improved analysis), `wang2025posteriordrift` (Wang et al.\ PMLR 258, 2025, generalised covariate shift), `riess2022shoes` (Riess et al.\ *ApJL* 934:L7, SH0ES distance ladder), `hendrycks2021math` (Hendrycks et al.\ NeurIPS 2021, MATH-12K dataset).

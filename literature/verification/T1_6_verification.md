# T1.6 — Local CP via SBERT + KNN — Full V1–V6 Verification

> **Verifier**: Claude Opus 4.7 (1M context), single-model lane
> **Date**: 2026-05-08
> **Proposal**: For each test prompt, retrieve top-K (K=30–60 per HGJ correction) nearest calibration prompts via SBERT embedding + FAISS, compute the local CP quantile from those. Provides per-prompt conditional coverage. Targeted at MATH→AIME OOD.
> **HGJ priority**: #1 (⭐⭐⭐, "OOD 1순위")
> **Token budget for execution**: 6–10 GPU hours (HGJ corrected)

---

## V1 — Prior art deep search

### V1.1 Methodology

10 web queries fired in two parallel waves:

1. `"local conformal prediction" LLM language model 2024 2025`
2. `"localized conformal prediction" KNN nearest neighbor LLM`
3. `prompt embedding KNN conformal prediction language model calibration`
4. `conditional coverage conformal prediction LLM evaluation 2025`
5. `Ulmer Zerva Martins non-exchangeable conformal prediction NLP token level`
6. `Guan 2023 localized conformal prediction LCP`
7. `SBERT sentence embedding FAISS conformal prediction OOD calibration`
8. `"local CP" "chain of thought" reasoning conformal 2026`
9. `Foygel Barber jackknife+ neighborhood conformal LLM 2024`
10. `retrieval augmented conformal prediction calibration LLM 2025`

Plus 2 follow-ups: `trajectory-level local conformal prediction reasoning chain LLM 2025 2026` and `"weighted conformal" "self-consistency" prompt embedding distribution shift math 2025`.

### V1.2 Closest 5 prior-art entries (deep-read)

#### A. **Guan 2023 — Localized Conformal Prediction** (Biometrika 110(1), 33–50; arXiv 2106.08460)
- **What it is**: General LCP framework. Reweights *every* calibration point by a localizer kernel `H(x_test, x_i)` (e.g., `exp(-d/h)`) before computing the conformal quantile.
- **Guarantees**:
  - Theorem 1: marginal coverage, finite-sample, distribution-free.
  - Theorem 3: *asymptotic* conditional coverage as n → ∞.
  - Theorem 4: approximate conditional coverage with explicit slack term ε(X_test).
- **Bandwidth selection**: `J(h) = avg PI length + λ × variability` via grid search; **no K-NN truncation discussed** — full kernel weighting only.
- **NOT applied to LLMs / NLP** in the paper.
- **Verdict for T1.6**: This is the *primary citation*. K-NN truncation (our SBERT+FAISS top-K) is a *finite computational shortcut* to LCP's full kernel weighting, equivalent to an indicator kernel `H(x,x_i) = 1[i ∈ top-K(x)]`.

#### B. **Han, Tang, Ghosh, Liu 2022 — Split Localized Conformal Prediction** (arXiv 2206.13092)
- **What it is**: KDE-based modified non-conformity score in the *split-CP* framework, much closer in spirit to our setting than Guan's transductive LCP.
- **Improvement over Guan**: simple, efficient, scales to high dimensions; preserves split-CP structure.
- **Empirical claim**: better conditional coverage at competitive average coverage.
- **NOT applied to LLMs / NLP**.
- **Verdict for T1.6**: A strictly *closer* methodological precedent than Guan for what we'd implement. Should be cited *alongside* Guan and Foygel-Barber. Was missing from HGJ feedback citation list.

#### C. **Lu et al. 2025 — Stable Localized Conformal Prediction via Transduction (SLCP)** (arXiv 2605.01452)
- **What it is**: Combines transductive stabilization with localized CP. Adds "set stability" metric quantifying sensitivity to calibration data — which is *exactly* the K-choice sensitivity HGJ flagged.
- **Direct relevance to our K=30–60 worry**: SLCP literally targets the small-effective-K instability that the HGJ review pointed out (`effective K_+ ≈ 50` at K=60). Provides theoretical handle and empirical evidence.
- **NOT applied to LLMs/NLP**.
- **Verdict**: Concurrent (2025–26 vintage) — must cite. Possibly even **adopt** SLCP's stabilization on top of vanilla local CP if pilot variance is high.

#### D. **Ulmer-Zerva-Martins 2024 — Non-Exchangeable Conformal Language Generation with Nearest Neighbors** (Findings of EACL 2024, arXiv 2402.00707)
- **What it is**: KNN-based non-exchangeable CP for **token-level** generation (NMT, language modeling).
- **Granularity**: Confirmed token-level — supplies "token-level, calibrated prediction sets equipped with statistical guarantees." KNN is over a kNN-LM-style token-context datastore (not a prompt-embedding store).
- **What it is NOT**: not trajectory-level, not prompt-similarity over reasoning prompts, not OOD-shift focused, not on math reasoning correctness.
- **Verdict**: This is the **only** previously-published "local CP for LLMs" baseline at *any* granularity. Critical to differentiate from. Our T1.6 is the **trajectory-level** analog that they explicitly do not target.

#### E. **Lin et al. 2025 — DS-CP: Domain-Shift-Aware CP for LLMs** (arXiv 2510.05566)
- **What it is**: Prompt-level *weighted* CP (not local-quantile) that uses all-MiniLM-L6-v2 embeddings + XGBoost density-ratio classifier to reweight the *entire* calibration set by proximity. Evaluated on MMLU subject-shift.
- **Granularity**: prompt-level, multiple-choice answer prediction.
- **NOT a local-CP / KNN**: it is a continuous reweighting of the full calibration set, *not* a top-K local quantile. No FAISS, no KNN truncation, no per-test local-quantile.
- **Score function**: LAC and APS softmax-style scores — *not* trajectory-aggregated step scores.
- **Verdict**: Parallel/complementary, not the same method. T1.6 differs along **three** axes: (i) granularity (trajectory vs. prompt MC), (ii) score (sc_top1/lp_min/prm_min vs. softmax LAC/APS), (iii) mechanism (top-K local quantile vs. full reweighting).

### V1.3 Other relevant 2025–26 hits (not deep-read)

| Paper | Why it matters | Relevance |
|---|---|---|
| Cherian-Gibbs-Candès 2024 NeurIPS (2406.09714) | conditional CP via embedding-conditioned quantile regression for LLMs | parallel — not KNN, but same conditional-coverage motivation |
| Wu et al. 2025 — Thought Calibration (2505.18404) | per-prompt early-stopping for reasoning LLMs via probes | does NOT use local CP; uses hidden-state probes — *different* mechanism |
| Liu et al. 2025 — Stable Localized CP (2605.01452) | already covered in C above | — |
| 2026 paper — Beyond Surface Statistics (2604.16217) | hidden-state CP under shift; not KNN | parallel, not competing |
| 2026 papers on RAG-CP (Conformal-RAG) | retrieval embedding for *factuality*, not *coverage adaptation* | parallel |

### V1.4 Bottom line on prior art

**No published paper does prompt-similarity local CP at trajectory level for math/CoT reasoning under domain shift.** The closest prior art:

- **Method side** (KNN/local CP): Guan 2023, Han et al. 2022, Lu et al. 2025 — all in regression / generic settings, none on LLM trajectories.
- **LLM side** (CP for LLMs): Ulmer-Zerva-Martins 2024 (token-level KNN-CP), Lin et al. 2025 DS-CP (prompt-level full reweighting).

T1.6 sits in the **empty cell**: trajectory-level + KNN-local + LLM + math reasoning OOD.

---

## V2 — Academic value

### V2.1 Is "Local CP for trajectory CoT" a meaningful contribution?

**Strong answer: yes, but the value is the *combination* with our existing artifacts, not the primitive itself.**

The primitive (KNN + local quantile) is textbook (Guan 2023; Han 2022). A paper that says only "we applied local CP to LLM trajectories" is weak — a competent reviewer will write "this is local CP applied to a new domain; methodologically incremental."

The contribution becomes **paper-worthy** under three frames:

1. **Frame A — empirical comparison** (weak, TMLR-acceptable): T1.6 is a §6 *Robustness* ablation row alongside Theorem 3 (PMF-weighted CP). The headline is "we tested two complementary OOD adaptation strategies; here is when each wins." This is a 1-figure / 1-table contribution.

2. **Frame B — methodological combination** (medium, TMLR-strong / NeurIPS-borderline): Compose local CP *with* Theorem 3's discrete-PMF weighting — i.e., do KNN local quantile, but *within the local neighborhood* apply PMF reweighting. This gives a strictly more general theorem than either alone (see V4).

3. **Frame C — diagnostic** (strong, no separate paper but high reviewer-defensibility): Use local CP as an empirical *diagnostic* of whether shift is "score-only" (A1 of Theorem 3) — if local CP and PMF-weighted CP give similar coverage, A1 holds; if they diverge, A1 fails. This is a free additional contribution that strengthens Theorem 3's rigor.

### V2.2 Comparison vs Ulmer-Zerva-Martins 2024 (EACL Findings)

The paper's contribution is *token-level KNN-CP for generation*. The trajectory-level extension is **NOT a trivial port**:

| Axis | Ulmer et al. 2024 | T1.6 (proposed) |
|---|---|---|
| Granularity | per-token | per-trajectory |
| Datastore | token-context vectors (kNN-LM style) | prompt embeddings (SBERT) |
| Quantile target | token-level prediction set coverage | final-answer correctness coverage |
| Task | NMT, LM | math reasoning, OOD |
| Aggregator $\phi$ | none (raw token prob) | non-trivial (`min_t` over PRM/lp/sc steps) |
| Theorem | non-exch CP at token level | trajectory-level local CP composed with score aggregator |

A trajectory-level local CP with a measurable aggregator $\phi$ over step scores is a non-trivial extension — it requires the aggregator to be measurable and the i.i.d. condition (Theorem 1's correction) to extend through the local-quantile estimator.

**Verdict**: extension is paper-worthy *as an ablation row* but NOT as the headline contribution. Headline contribution remains the trajectory-CP framework + Theorem 3 PMF correction; T1.6 strengthens robustness story.

### V2.3 Venue norms

| Venue | Headline-worthy? | Ablation-worthy? |
|---|---|---|
| TMLR | No | **Yes — required §6 row** |
| NeurIPS / ICLR (D&B) | No (alone) | Yes |
| NeurIPS / ICLR (main) | Possibly with Frame B (composition theorem) | Yes |
| EACL / ACL / EMNLP | Possibly as a standalone short paper | — |

**Recommendation**: position T1.6 as Frame A+C in TMLR submission; reserve Frame B (composition theorem) as future work or push toward a stronger venue if proven.

---

## V3 — Feasibility & performance

### V3.1 Compute budget

HGJ corrected: **6–10 GPU hours** (3-reviewer revision from initial 3–5 hr).

Decomposition:
- SBERT embedding of cal prompts (n_cal ≤ 4192 max): <5 min on H100.
- FAISS index build: trivial (<1 min).
- Per-test KNN search: O(log n) — negligible in inference loop.
- **Main cost** = re-running selective accuracy + bootstrap CI (500 × 10 splits) for each (α, K) cell across the cross-dataset transfer matrix. With ~12 cells × 30–45 min/cell empirical = 6–9 hr matches HGJ corrected estimate.

**Feasibility verdict**: ✅ realistic, no blocking risk.

### V3.2 Performance prediction

HGJ Idea 2.1 prediction table (re-verified from `HGJ_experiment_ideas.md` line 96):

| Method | empirical coverage @ α=0.5 | source |
|---|---|---|
| Vanilla CP | 0.187 | METHOD_AND_RESULTS §2.6 |
| Theorem 3 (PMF) | 0.633 (overshoot by +13pp) | METHOD_AND_RESULTS §2.6 |
| **Local CP** (predicted) | **0.45–0.55** (target near) | HGJ Idea 2.1 |
| Local CP + PMF combined | **0.48–0.52** (best) | HGJ Idea 2.1 |

**Plausibility check**:
- Local CP is *less* assumption-laden than Theorem 3 (relaxes A1 score-only-shift to local exchangeability). Should reduce *bias* (overshoot) at cost of higher *variance* (small effective K).
- 0.45–0.55 prediction is consistent with this: moves the 0.633 overshoot toward 0.5 target while remaining close to nominal.
- The combined estimate (0.48–0.52) is *optimistic* — it assumes stochastic-dominance composition, which is not generally guaranteed. Realistic outcome: 0.45–0.60 with possibly higher variance. Worth caveating in writing.

### V3.3 Risk: K-choice sensitivity

HGJ review noted: at K=60 with n_+ ≈ 200–375, effective `K_+ ≈ 50`, giving discreteness slack `1/(K_+ + 1) ≈ 2%`. Real risk:

- **K too small**: high quantile variance, possibly worse than Theorem 3.
- **K too large**: approaches global vanilla CP, loses locality.
- **Optimal K** is dataset-dependent (problem density in embedding space).

**Mitigation**:
1. **Sweep K ∈ {30, 45, 60, 90}** as primary ablation. ~2× compute, but gives the K-sensitivity figure reviewers will ask for anyway.
2. **Adopt SLCP stabilization** (Lu et al. 2025) if pilot shows high variance — keeps marginal coverage, reduces set-size variability.
3. **Sanity bookends**: K=1 (overfit, cov ≪ target), K=n_cal (recovers vanilla CP, cov = 0.187). Both should fail gracefully.

### V3.4 Embedding choice ablation

HGJ Idea 2.1 already plans 3-way comparison:
- **SBERT prompt** (e.g., `all-MiniLM-L6-v2`) — same as DS-CP. Free, fast.
- **LLM last hidden state** — already computed during decode; richer signal but high-dim.
- **MathBERT** — math-domain-tuned; possibly best for math OOD but extra dependency.

**Recommendation**: SBERT primary (matches DS-CP, comparable apples-to-apples), MathBERT as sanity ablation (1 extra cell), LLM hidden state as exploratory.

---

## V4 — Incremental vs structural

### V4.1 Incremental view

Local CP is textbook (Guan 2023). KNN is textbook. SBERT is textbook. FAISS is textbook. Applying these to LLM trajectories is **engineering integration**, not new methodology.

Reviewer will write: *"The proposed method is a direct application of localized conformal prediction (Guan, 2023) using off-the-shelf sentence embeddings and KNN retrieval. While the empirical evaluation on math reasoning is useful, the methodological contribution is incremental."*

### V4.2 Structural view — the composition theorem

Where T1.6 *could* become structural: **composing local CP with Theorem 3's discrete-PMF weighted CP**.

Sketch (NOT yet proven; flagged as plausible structural contribution):

> **Conjecture (Local Empirical-PMF Weighted CP)**: Suppose Assumption A1 (score-only shift) holds *locally* in an SBERT-embedding neighborhood of the test prompt — i.e., within neighborhood `N_K(x_test)`, the conditional structure (X, Y | S) is invariant. Then, applying empirical-PMF weighted CP *restricted to N_K(x_test)*, with K → ∞ at rate K/n → 0, achieves marginal coverage with combined slack
> $$\text{gap} \leq |\mathcal{S}|/2 \cdot \sqrt{\log(2|\mathcal{S}|/\delta) / 2K} + O(1/K_+) + \epsilon_{\text{loc}}(K, n)$$
> where $\epsilon_{\text{loc}}$ vanishes as the embedding-induced metric concentrates the conditional distribution.

This is **stronger** than either Theorem 3 alone (which assumes A1 globally — likely violated for MATH→AIME) **or** local CP alone (which makes no use of the discrete-score structure).

If the conjecture proves out, T1.6 is **structural**: it's a new theorem, not a new domain application.

### V4.3 Honest verdict

- **As proposed by HGJ Idea 2.1 (run local CP and report coverage table)**: incremental. Acceptable as TMLR §6 Robustness ablation, not headline.
- **As Frame B (composition theorem)**: structural, but the theorem is unproved and might fail. Risk-adjusted value: high upside, ~50% chance of working out cleanly.
- **As Frame C (diagnostic for A1)**: free side benefit, low risk, modest value.

**Honest recommendation**: execute T1.6 with HGJ specs (K∈{30,45,60,90}, full shift matrix, SBERT primary + MathBERT ablation) as a §6 Robustness ablation. **Separately** prototype the composition theorem on paper — if it proves out in a 1–2 day theory pass, it becomes a 4th theorem; if not, drop it without loss. Net cost: 1–2 days extra; net upside: significantly stronger paper.

---

## V5 — Reviewer objection + response

### Objection 1: "This is just local CP applied to a new domain."

**Response**: Granted that the *primitive* is from Guan 2023 and Han et al. 2022. The contributions specific to the LLM trajectory setting are: (i) a measurable aggregator $\phi$ over step-scores (`lp_min`, `prm_min`, `sc_top1`), making the local quantile defined over an aggregated discrete score (not a continuous regression score); (ii) the SBERT-embedding choice is empirically motivated by the parallel DS-CP (Lin et al. 2025) prompt-embedding result on MMLU; (iii) the comparison with empirical-PMF weighted CP (our Theorem 3) gives a *taxonomy* of OOD adaptation strategies for discrete-score CP-for-LLM, not just a single method. We position the result as "OOD coverage taxonomy," not "new local-CP method."

### Objection 2: "Why not just use Ulmer-Zerva-Martins 2024 EACL Findings?"

**Response**: They operate at token level using a kNN-LM token-context datastore; we operate at trajectory level using a prompt-embedding datastore. The two settings have non-overlapping aggregator structure: their score is per-token, ours is `min_t` (or analog) over step scores; their datastore is per-token-context, ours is per-prompt. The token-level method *cannot* directly answer the question "is the final answer correct with probability 1−α" without an additional aggregator — which is a separate research question. We cite them as the closest prior local-CP-for-LLM precedent.

### Objection 3: "K=30–60 is too small; you'll get massive variance."

**Response**: Acknowledged in V3.3. We sweep K ∈ {30, 45, 60, 90} and report the K-sensitivity curve as a primary figure. We additionally adopt the SLCP stabilization (Lu, Foygel-Barber, et al. 2025) as a secondary row for variance reduction. The K=30 lower bound is justified by `n_+ ≈ 200–375` per Foygel-Barber's effective-sample-size analysis (jackknife+ paper).

### Objection 4: "Why not just use DS-CP (Lin et al. 2025) — same idea?"

**Response**: DS-CP applies *full-set reweighting* via a learned XGBoost density-ratio classifier; T1.6 applies *top-K local quantile*. The two are mathematically distinct: DS-CP retains all calibration points with continuous weights; T1.6 truncates to a hard neighborhood. DS-CP is *complementary* — we cite it as the canonical embedding-similarity OOD baseline and report DS-CP as an additional row in our Robustness table. If anything, our method shows that for *trajectory-level discrete scores*, the top-K local quantile is a more transparent and theoretically clean choice (Guan 2023 LCP framework directly applies; DS-CP requires nontrivial density-ratio assumptions).

### Objection 5: "Conditional coverage is impossible without distributional assumptions (Lei-Wasserman 2014)."

**Response**: Acknowledged. We claim *approximate* conditional coverage (Theorem 4 of Guan 2023), not exact. The `local exchangeability` assumption is named explicitly. Our experimental coverage tables report marginal coverage; the conditional-coverage angle is a *secondary* claim documented in §6.

---

## V6 — Final verdict

### V6.1 Execute or skip?

**Execute**. T1.6 is the highest-value OOD ablation in HGJ's plan and survives all 5 verification lenses.

### V6.2 Specs (consolidated)

| Item | Spec | Source |
|---|---|---|
| Method | top-K local quantile via SBERT embedding + FAISS | HGJ 2.1 |
| K sweep | {30, 45, 60, 90} | HGJ review (correction from K=100) |
| Embedding primary | `all-MiniLM-L6-v2` (matches DS-CP) | this verification |
| Embedding ablation | MathBERT, LLM last hidden state | HGJ 2.1 |
| Calibration set | MATH-500 sc_top1 (n_+ ≈ 200–375) | METHOD_AND_RESULTS §2.6 |
| Test set | AIME-2024 (primary), full shift matrix (extension) | HGJ review |
| Comparison rows | vanilla CP, Theorem 3 PMF, DS-CP (full-reweight), local CP (ours), local CP + SLCP stabilization | this verification |
| Compute | 6–10 GPU hours | HGJ corrected |

### V6.3 Citations to add

Mandatory:
1. **Guan 2023** (Biometrika 110(1)) — primary local CP framework, Theorem 4 approximate conditional coverage.
2. **Han, Tang, Ghosh, Liu 2022** (arXiv 2206.13092) — Split Localized CP, closest split-CP precedent.
3. **Foygel Barber, Candès, Ramdas, Tibshirani 2023** — jackknife+ effective-sample-size analysis (already in HGJ correction list).
4. **Lei & Wasserman 2014** — impossibility result for exact conditional coverage (already in HGJ correction list).
5. **Tibshirani, Foygel-Barber, Candès, Ramdas 2019** — weighted CP (already cited as Theorem 3 backbone, in our 30-paper).
6. **Ulmer, Zerva, Martins 2024** (Findings of EACL) — *only* prior LLM local-CP, token-level. Already in our 30-paper §2.3.
7. **Lin et al. 2025 (DS-CP)** — full-reweighting prompt embedding shift CP. Already in our 30-paper.

Discretionary (cite if Frame B composition theorem proves out, or if pilot shows variance issues):
8. **Lu, Foygel-Barber, et al. 2025 (SLCP)** (arXiv 2605.01452) — Stable Localized CP via Transduction.

### V6.4 Frame to use

**Primary**: Frame A (empirical comparison row in §6 Robustness).
**Augmentation**: Frame C (diagnostic for Theorem 3's A1 assumption — free side benefit).
**Stretch goal**: Frame B (composition theorem — 1–2 day theory pass; abandon if not clean).

### V6.5 Honest verdict on contribution magnitude

For a TMLR paper as currently scoped: T1.6 is **2nd most important OOD result** after Theorem 3 itself, and **first most defensible reviewer pushback prevention**. Its primary value is *not* claiming a new method, but completing a *taxonomy* of OOD adaptation strategies that demonstrates we understand the assumption-cost trade-off.

For ICLR/NeurIPS main: T1.6 alone is insufficient as a paper; with Frame B composition theorem proven, it becomes a co-headline contribution.

### V6.6 Cross-Model Verification Results

> Single-model verification only (per CLAUDE.md `cross_model_verification.mode: all`, but inference token is `sk-PLACEHOLDER` and external verifier is not invoked in this lane). This verification is single-model (Claude Opus 4.7); reader should treat verdict as *primary verdict only*. Cross-model verifier should be invoked separately on the final §6 draft when the paper is written.

### V6.7 Recommendation

**PROCEED** with T1.6 execution per V6.2 specs. Schedule for HGJ Week 2 plan as currently specified (`HGJ_review_feedback.md` Week 2). Allocate 6–10 GPU hours plus 1–2 days for Frame B theory prototyping. Accept that the contribution is *robustness completeness*, not headline novelty.

---

## Sources

- Guan 2023, Localized Conformal Prediction (Biometrika): https://academic.oup.com/biomet/article/110/1/33/6647831 ; arXiv 2106.08460
- Han, Tang, Ghosh, Liu 2022, Split Localized Conformal Prediction: https://arxiv.org/abs/2206.13092
- Lu, Foygel-Barber et al. 2025, Stable Localized CP via Transduction: https://arxiv.org/abs/2605.01452
- Ulmer, Zerva, Martins 2024, Non-Exchangeable Conformal Language Generation with Nearest Neighbors: https://aclanthology.org/2024.findings-eacl.129/ ; arXiv 2402.00707
- Lin et al. 2025, DS-CP (Domain-Shift-Aware CP for LLMs): https://arxiv.org/html/2510.05566
- Tibshirani, Foygel-Barber, Candès, Ramdas 2019, Conformal Prediction Under Covariate Shift: https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf
- Wu et al. 2025, Thought Calibration: https://arxiv.org/abs/2505.18404
- Cherian, Gibbs, Candès 2024, Enhanced LLM Validity: https://arxiv.org/abs/2406.09714
- Internal: /home/nvidia/future/METHOD_AND_RESULTS.md §2.6
- Internal: /home/nvidia/future/theorems/theorem3_weighted_cp_discrete.md
- Internal: /home/nvidia/future/HGJ_experiment_ideas.md Idea 2.1
- Internal: /home/nvidia/future/HGJ_review_feedback.md

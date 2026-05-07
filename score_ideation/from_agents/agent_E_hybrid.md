# Agent E (hybrid / ensemble angle) — New Score Functions for CoT-CP

**Author:** Agent E (hybrid / ensemble angle)
**Scope:** Five new score-function proposals that combine `lp_min`, `prm_min`, `sc_top1` (and a small set of cheap auxiliary signals) in non-trivial ways. Each proposal explicitly addresses the empirical failure of naive rank-mean averaging, makes a compute-efficiency case, and discusses finite-sample coverage.

## Why naive averaging failed (working hypothesis)

A rank-mean of `prm_min` and `sc_top1` is essentially a *uniform stacked* score. It implicitly assumes both signals carry equal information and that informativeness is uniform across the score range. Empirically, `sc_top1` saturates the "easy" mass — when 8 chains agree, correctness is nearly certain — so the marginal signal of `prm_min` only matters in the *contested* region. Uniform averaging dilutes `sc_top1` where it is decisive and injects `prm_min` noise where `prm_min` is least calibrated. All proposals below *condition* the combination on regime instead of averaging globally.

**Coverage note.** Split-conformal (Theorem 1) requires only that calibration and test scores be exchangeable under the same fixed measurable scoring function S. Non-monotone/learned aggregators preserve coverage as long as S is fixed and trained on a fold disjoint from calibration.

---

## 1. `gated_cascade` — cost-amortized hierarchical CP

**Components:** `lp_min` (1×) → `prm_min` (2×) → `sc_top1` (8×).

**Combination method:** Hierarchical cascaded gating with two thresholds learned on a held-out fold; final score is a *single fixed function* of the three signals.

**Compute cost (expected):** 1× + p1·2× + p1·p2·8×. With p1 ≈ 0.4, p2 ≈ 0.5 → ~3.4× vs 8× for `sc_top1`.

**Mathematical definition.**
Define fixed thresholds τ_lp, τ_prm (learned once on a training fold). Define
S_GC(x) = w_1 · σ(τ_lp − lp_min(x))   (gate 1: 0 if very confident lp)
       + w_2 · 1[lp_min(x) ≤ τ_lp] · σ(τ_prm − prm_min(x))  (gate 2)
       + w_3 · 1[lp_min(x) ≤ τ_lp ∧ prm_min(x) ≤ τ_prm] · (1 − sc_top1(x))
with w_1 + w_2 + w_3 = 1, σ logistic. Crucially, signals are computed *only* in their region; for queries that pass gate 1 the score equals w_1·σ(τ_lp − lp_min), still a valid score, just based on the cheap signal alone for that subset.

**Why it should beat sc_top1.** sc_top1 spends 8× compute even on queries where lp_min already gives a near-perfect signal. Cascading concentrates expensive compute on the contested mid-region where sc_top1 actually adds value. AUROC vs correctness is preserved while *compute-normalized* AUROC improves sharply.

**Why it dodges naive averaging.** Each component dominates *only* in its assigned regime; saturated `sc_top1` is never summed with noisy `prm_min`.

**Implementation.** Trained: τ_lp, τ_prm by maximizing Spearman ρ(S_GC, 1[correct]) under a budget constraint on ~500 examples; w_i via isotonic regression. Freeze before calibration.

**Coverage.** S_GC is a fixed measurable function once τ, w are frozen → split-CP exchangeability holds → Theorem 1 preserved. Non-monotonicity is irrelevant.

**Literature.** Cost-aware split CP, related to Cherubin et al.'s cascaded classifiers, Angelopoulos et al.'s learn-then-test, and Fisch et al.'s cascaded conformal variant.

---

## 2. `disagreement_signal` — disagreement-aware fusion

**Components:** `lp_min`, `prm_min` (and optionally `sc_top1`).

**Combination method:** Treat *disagreement* between calibrators as an explicit feature. Use a 3-input logistic model fit on training fold.

**Compute cost:** 1× + 2× = 3× (sc_top1 optional → +8×).

**Mathematical definition.**
Let z_lp = Φ(lp_min), z_prm = Φ(prm_min) be empirical-CDF rank-transforms (computed on training fold, frozen). Define the disagreement feature
d(x) = (z_lp(x) − z_prm(x))^2.
Score:
S_DS(x) = β_0 + β_1·z_lp(x) + β_2·z_prm(x) + β_3·d(x) + β_4·(z_lp · z_prm),
sign-flipped so larger ↔ less likely correct.

**Why it should beat sc_top1.** When lp and prm agree, low residual uncertainty → strong signal. When they disagree, the disagreement itself flags a likely failure mode (e.g., locally fluent text with semantic step-error). Naive averaging cancels this; the d(x) term *amplifies* it.

**Why it dodges naive averaging.** β_3·d(x) is the explicit antidote — contested cases are pushed up regardless of their average level.

**Implementation.** L2-regularized logistic regression on a 1k held-out fold, frozen pre-calibration. Training-free fallback: β = (0, 1, 1, 2, 0).

**Coverage.** Fixed measurable function of (lp_min, prm_min) → split-CP holds. Score is *non-monotone* in lp_min when β_3 > 0; this does not violate finite-sample CP, which is distribution-free and requires no monotonicity.

**Literature.** Cousin to Romano et al.'s CQR (learned residual on a base predictor) and "agreement-based selective prediction" (Lakshminarayanan et al.).

---

## 3. `quantile_fusion_cp` — per-α weighted-quantile fusion

**Components:** `lp_min`, `prm_min`, `sc_top1`.

**Combination method:** Compute three separate split-CP cutoffs, then fuse at the *cutoff level* rather than the score level. The fused decision is "keep iff at least k of 3 cutoffs accept" with k chosen by training-fold selection.

**Compute cost:** 1× + 2× + 8× = 11× (worst case; can be reduced via cascading with proposal 1).

**Mathematical definition.**
Let q_α^{(j)} be the (1−α)-quantile of calibration scores S_j (j ∈ {lp, prm, sc}). Per-α majority fusion:
A_k(x; α) = 1[ Σ_{j} 1[S_j(x) ≤ q_α^{(j)}] ≥ k ].
For a single conformal score we use the rank-quantile distance:
S_QF(x; α) = max_j w_j(α) · (S_j(x) − q_α^{(j)}) ,
with w_j(α) ≥ 0 fit per α-grid on training fold.

**Why it should beat sc_top1.** Each S_j calibrates a different region; per-α fusion picks the best subset of calibrators at that operating point. At small α, sc_top1 ties dominate; lp_min quantile breaks them. At loose α, prm_min and lp_min provide higher resolution.

**Why it dodges naive averaging.** Fusion is at the *threshold* level per α, never at the score scale.

**Implementation.** w_j(α) fit on training fold maximizing acceptance at fixed empirical coverage. Frozen pre-calibration.

**Coverage.** S_QF as a single score yields exact (1−α) coverage by standard split-CP. The *vote* form A_k requires either Bonferroni (≥ 1 − k·α/3) or joint calibration. Recommend S_QF form for clean Theorem-1 coverage.

**Literature.** Weighted-aggregation CP (Solari & Djordjilović); per-α weighting is a learn-then-test protocol (Angelopoulos et al. 2021).

---

## 4. `tiebreak_lex` — discreteness-aware lexicographic score

**Components:** `sc_top1` (primary), `lp_min` (tiebreaker), optionally `prm_min` (secondary tiebreaker).

**Combination method:** Lexicographic / continuous tiebreaking. No averaging.

**Compute cost:** 1× + 8× = 9× (sc + lp; prm optional → 11×).

**Mathematical definition.**
With N = 8 SC samples, sc_top1 ∈ {1/8, 2/8, …, 8/8}, only 8 distinct values → many ties at the CP cutoff. Define
S_TL(x) = (1 − sc_top1(x)) + ε · (1 − Φ_lp(lp_min(x))),
with ε ∈ (0, 1/8) so that the lp_min contribution is strictly smaller than one sc_top1 step. Φ_lp is the training-fold empirical CDF of lp_min, mapping it to [0,1]. Score is increasing in "badness".

**Why it should beat sc_top1 alone.** With only 8 distinct levels, the CP cutoff at most α cannot control coverage tightly — it must over- or under-include ties. Adding a continuous tiebreaker with ε < 1 step *strictly refines* the order while preserving the ordering of distinct levels. Acceptance set size shrinks in expectation with no loss in correctness fidelity. Pareto improvement over `sc_top1`.

**Why it dodges naive averaging.** ε < 1/8 guarantees the secondary signal cannot override the primary — deterministic ordering refinement within ties only.

**Implementation.** Training-free (ε = 1/16, Φ_lp = identity-rescaled). Optionally fit ε on training fold.

**Coverage.** Deterministic measurable function of (sc_top1, lp_min) → split-CP holds. Strict refinement makes the (1−α) quantile well-defined; Theorem 1 applies directly.

**Literature.** Equivalent to randomized-tiebreak CP (Romano et al. 2020 Appendix; Sadinle et al. 2019), but with a *meaningful* tiebreaker.

---

## 5. `learned_stack_cv` — cross-validation+ stacked learner

**Components:** `lp_min`, `prm_min`, `sc_top1`, plus two cheap auxiliary signals shipped with the LLM at inference time at near-zero marginal cost: token-level *attention entropy* H_attn (mean entropy of last-layer attention over reasoning tokens) and *answer-span perplexity ratio* r_ans = PPL(answer | CoT) / PPL(answer | nothing).

**Combination method:** Gradient-boosted decision tree (depth ≤ 4, ≤ 100 trees) trained via cross-validation+ (CV+, Barber et al. 2021), so the score function is *not* fit on the calibration set.

**Compute cost:** 1× + 2× + 8× + ~0 + ~0 ≈ 11× (auxiliaries are read-out from the SC forward passes already done).

**Mathematical definition.**
Let f̂ : ℝ^5 → ℝ be the trained GBDT. For a test point we use
S_LS(x) = − f̂(lp_min, prm_min, sc_top1, H_attn, r_ans).
Under CV+, calibration is performed using *held-out-fold* predictions f̂_{-k}, ensuring exchangeability between training and calibration scores.

**Why it should beat sc_top1.** A non-linear, non-monotone learner can carve regions like "high sc_top1 *but* high attention entropy on the answer span = hallucinated agreement" that no single score or linear combo captures. The cheap auxiliaries break sc_top1 saturation in the high-agreement region — its dominant failure mode.

**Why it dodges naive averaging.** GBDT fits explicit *interactions*, e.g., a leaf for "sc_top1=1 ∧ H_attn>τ" can override the base rate. Linear stacking cannot represent this.

**Implementation.** 1–5k training fold disjoint from calibration; CV+ (5-fold) or fully separate training set; aggressive regularization (depth ≤ 4, early stopping); freeze f̂ before calibration.

**Coverage.** (a) **Separate training set:** fixed S_LS → split-CP gives exact (1−α) (Theorem 1). (b) **CV+ on calibration data:** Barber et al. (2021) give ≥ 1 − 2α (constant doubles but coverage preserved). Recommend (a) when a training fold is affordable. Non-monotonicity is fine — CP needs only measurability and exchangeability.

**Literature.** Direct instance of stacked CP / learned scoring with CV+ (Barber et al. 2021; Lei & Wasserman); related to Romano et al.'s deep-score CP and Fisch et al.'s learned-filter CP.

---

## Summary table

| Name | Components | Cost (×LP) | Trained? | Coverage |
|---|---|---|---|---|
| 1. gated_cascade | lp, prm, sc | ~3.4 | yes (2 thresholds) | exact 1−α |
| 2. disagreement_signal | lp, prm | 3 | yes (5-param logistic) | exact 1−α |
| 3. quantile_fusion_cp | lp, prm, sc | 11 | yes (per-α weights) | exact 1−α (S_QF form) |
| 4. tiebreak_lex | sc, lp | 9 | no | exact 1−α |
| 5. learned_stack_cv | lp, prm, sc, +2 aux | 11 | yes (GBDT, CV+) | 1−α (split) or 1−2α (CV+) |

## How each addresses "naive averaging fails"

1. **Gated cascade** — never averages; routes by regime.
2. **Disagreement signal** — explicitly amplifies the disagreement that averaging cancels.
3. **Quantile fusion** — fuses at thresholds per α, not at score scale.
4. **Tiebreak lex** — strict refinement only within ties; cannot dilute.
5. **Learned stack** — non-linear interactions capture regime-specific dominance that linear averaging cannot.

All proposals preserve marginal coverage via the standard split-CP argument (or CV+ with constant-doubled bound for proposal 5 in its in-sample variant); none requires monotonicity in any single signal.

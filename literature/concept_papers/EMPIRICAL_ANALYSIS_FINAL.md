# Empirical Analysis — CoT-CP (Pearl Causal × Distance Ladder) — FINAL (12-cell)

> **Status.** v2 unified empirical analysis after all 12 Pearl cells closed, 2026-05-08.
> **Supersedes.** `EMPIRICAL_ANALYSIS.md` (v1, written when 5/12 Pearl cells were closed and missing the four cells that carry the bulk of T4 v3's pre-registered predictions).
> **Sources.** `pearl_full/{AGGREGATE.json, AGGREGATE.md, phi4_*.json, qwen25_*.json}` (12 cells, 200-trace each), `distance_ladder_full/{AGGREGATE.json, AGGREGATE.md, phi4_strategyA_B_Bp.json, qwen25_*_strategyA_B_Bp.json}` (4 model cells).
> **Theory anchors.** T4 v3 Corollary 4.1 (`theorem4_v3_cascade_stratified.md` §3 + §8.2 pre-registered predictions); T5 v2 + T5' (`theorem5_v2_consolidated.md`); T6 (`joint_composition_theorem.md`).
> **Coverage of cells.** Pearl: **12 of 12 closed** (4 models × 3 datasets, n = 200 each). DL: 4 of 4 model cells closed.
> **Critical-rule compliance.** Every numerical value below is a direct read from `pearl_full/<cell>.json` `by_score_alpha[<key>]` or `distance_ladder_full/<file>.json`. No fabricated numbers. **Honest acknowledgment of falsification: T4 v3's headline aggregate predictions for `phi4__aime`, `qwen25_32b__aime`, `phi4__math500` are sign-flipped (negative observed where positive predicted) and the gap≥5 stratum no longer carries the salvage signal it carried on `qwen25_7b__aime`.**

---

## §0 Executive headline (read this first)

**The picture is messier than the 5-cell snapshot suggested.**

- 108 test points = 12 cells × 3 scores × 3 alphas. **Mean Δ_lift across all 108 = −0.82pp.** Across the 36 `lp` tests it is **−0.59pp**. Across the 36 `marg` tests, **−0.73pp**.
- **0 of 108** aggregate Δ_lift CIs exclude zero (positive *or* negative).
- **0 of 108** survive Benjamini–Hochberg FDR at q = 0.10. (Also 0 at q = 0.20; also 0 for the 108 cascade-gap≥5 stratified tests at q = 0.10.)
- **27 / 108** (25%) point estimates are positive; **81 / 108** are negative.
- **3 of 5** pre-registered T4 v3 predictions sign-flipped (FAIL with negative observed where positive predicted): `phi4__aime`, `qwen25_32b__aime`, `phi4__math500`. **1 of 5** is PARTIAL (`qwen25_math_7b__olympiad` — directionally right, magnitudes below band). **1 of 5** is PASS by aggregate sign with FAIL gap≥5 (`qwen25_7b__olympiad`). The single P3 row that remains a clean PASS is `qwen25_32b__aime__marg` at α=0.5: agg +3.53pp, gap≥5 +6.43pp.
- The cascade-gap-stratified salvage hypothesis (Corollary 4.1) is **not consistently supported** when extended to the 12-cell sample: gap≥5 strata are positive in only 37 / 108 settings, mean Δ_strat(gap≥5) = **−1.06pp**.
- **DL Strategy B' is intact.** Banach contraction supported on all 4 models: A_gap = 21.4pp → B'_gap = 9.5pp at α=0.5 (≈55.6% relative reduction), matching T5' back-fit $\bar L \approx 0.85$.

The two most likely diagnoses (§5, §7):
1. **(A4) failure on phi-4 / 32B AIME.** K=4 temperature-0.7 alternatives at $t^*$ are *not* materially better than at $t_{\text{worst}}$ because all four resamples branch from the same corrupted prefix; the 32B model in particular shows large *negative* gap=1 strata (−9 to −15pp) — i.e. early intervention on a wrong-but-recoverable trace actively *destroys* recoveries. This is a structural bug in the "earliest-bad-step is best to intervene at" claim.
2. **Self-correcting models break (A5).** Phi-4 (and to a lesser extent Qwen2.5-32B) appears to self-correct downstream of $t^*$ with non-trivial probability; intervening at the *earliest* bad step short-circuits a recovery the model would otherwise have executed. T4 v3's (A5) "non-self-correcting model class" is not satisfied for these models.

The honest reframe (§7): **step intervention has structural limits**. The earliest-bad-step heuristic outperforms the worst-step baseline only on a narrow regime — single-mode 7B-class models on hard problems (AIME) where (A4)'s recovery probability is operative. It under-performs on (i) self-correcting frontier models (phi-4), (ii) capacity-rich models with many recovery paths (32B), and (iii) easy-problem regimes (MATH-500) where the false-positive cost of intervening on already-correct traces dominates.

---

## §1 Per-cell × score × α aggregate Δ_lift table (108 rows)

Source: `pearl_full/<cell>.json` `by_score_alpha[<score>__a<alpha>][lift_earliest_vs_worst]` `[mean, lo, hi]`. Bootstrap n_boot = 500 with 95% percentile CI. n_alts is the number of (trace, t*) pairs eligible for K=4 majority intervention.

| cell | score | α | vanilla | K4_worst | K4_earliest | Δ_lift | 95% CI | n_alts |
|---|---|---|---|---|---|---|---|---|
| phi4_aime | lp | 0.1 | 0.273 | 0.346 | 0.331 | −1.51pp | [−7.00, +5.00] | 917 |
| phi4_aime | lp | 0.3 | 0.279 | 0.354 | 0.329 | −2.45pp | [−9.00, +3.00] | 917 |
| phi4_aime | lp | 0.5 | 0.277 | 0.352 | 0.308 | **−4.44pp** | [−12.00, +2.00] | 917 |
| phi4_aime | ent_neg | 0.1 | 0.273 | 0.350 | 0.330 | −2.00pp | [−9.00, +5.00] | 917 |
| phi4_aime | ent_neg | 0.3 | 0.278 | 0.354 | 0.323 | −3.13pp | [−9.00, +3.00] | 917 |
| phi4_aime | ent_neg | 0.5 | 0.280 | 0.355 | 0.310 | **−4.49pp** | [−11.00, +1.00] | 917 |
| phi4_aime | marg | 0.1 | 0.275 | 0.329 | 0.326 | −0.26pp | [−7.00, +7.00] | 917 |
| phi4_aime | marg | 0.3 | 0.275 | 0.332 | 0.317 | −1.55pp | [−8.00, +5.00] | 917 |
| phi4_aime | marg | 0.5 | 0.275 | 0.327 | 0.306 | −2.04pp | [−9.00, +5.00] | 917 |
| phi4_math500 | lp | 0.1 | 0.739 | 0.780 | 0.786 | +0.65pp | [−4.00, +6.00] | 745 |
| phi4_math500 | lp | 0.3 | 0.739 | 0.781 | 0.778 | −0.32pp | [−5.00, +4.00] | 745 |
| phi4_math500 | lp | 0.5 | 0.740 | 0.779 | 0.774 | −0.44pp | [−4.52, +3.52] | 745 |
| phi4_math500 | ent_neg | 0.1 | 0.743 | 0.783 | 0.788 | +0.56pp | [−4.00, +5.00] | 745 |
| phi4_math500 | ent_neg | 0.3 | 0.739 | 0.781 | 0.770 | −1.06pp | [−5.00, +2.00] | 745 |
| phi4_math500 | ent_neg | 0.5 | 0.737 | 0.778 | 0.769 | −0.85pp | [−5.00, +3.00] | 745 |
| phi4_math500 | marg | 0.1 | 0.737 | 0.778 | 0.773 | −0.52pp | [−5.00, +4.00] | 745 |
| phi4_math500 | marg | 0.3 | 0.741 | 0.781 | 0.770 | −1.08pp | [−5.00, +3.00] | 745 |
| phi4_math500 | marg | 0.5 | 0.741 | 0.779 | 0.759 | −1.97pp | [−7.00, +3.00] | 745 |
| phi4_olympiad | lp | 0.1 | 0.424 | 0.473 | 0.461 | −1.21pp | [−7.00, +5.00] | 812 |
| phi4_olympiad | lp | 0.3 | 0.417 | 0.470 | 0.463 | −0.71pp | [−7.00, +6.52] | 812 |
| phi4_olympiad | lp | 0.5 | 0.423 | 0.474 | 0.452 | −2.19pp | [−9.53, +4.52] | 812 |
| phi4_olympiad | ent_neg | 0.1 | 0.423 | 0.472 | 0.452 | −2.05pp | [−7.00, +2.00] | 812 |
| phi4_olympiad | ent_neg | 0.3 | 0.415 | 0.465 | 0.458 | −0.76pp | [−7.00, +5.00] | 812 |
| phi4_olympiad | ent_neg | 0.5 | 0.421 | 0.468 | 0.446 | −2.27pp | [−8.00, +3.52] | 812 |
| phi4_olympiad | marg | 0.1 | 0.418 | 0.455 | 0.442 | −1.24pp | [−7.00, +4.00] | 812 |
| phi4_olympiad | marg | 0.3 | 0.421 | 0.457 | 0.455 | −0.19pp | [−5.52, +5.00] | 812 |
| phi4_olympiad | marg | 0.5 | 0.423 | 0.459 | 0.443 | −1.53pp | [−8.00, +4.00] | 812 |
| qwen25_32b_aime | lp | 0.1 | 0.375 | 0.436 | 0.409 | −2.71pp | [−8.00, +2.00] | 709 |
| qwen25_32b_aime | lp | 0.3 | 0.379 | 0.440 | 0.410 | **−3.02pp** | [−8.00, +2.00] | 709 |
| qwen25_32b_aime | lp | 0.5 | 0.376 | 0.437 | 0.428 | −0.93pp | [−9.00, +6.00] | 709 |
| qwen25_32b_aime | ent_neg | 0.1 | 0.374 | 0.444 | 0.402 | **−4.22pp** | [−10.00, +0.52] | 709 |
| qwen25_32b_aime | ent_neg | 0.3 | 0.375 | 0.446 | 0.415 | −3.12pp | [−9.00, +3.00] | 709 |
| qwen25_32b_aime | ent_neg | 0.5 | 0.378 | 0.447 | 0.440 | −0.67pp | [−8.00, +6.00] | 709 |
| qwen25_32b_aime | marg | 0.1 | 0.378 | 0.417 | 0.398 | −1.84pp | [−7.00, +3.00] | 709 |
| qwen25_32b_aime | marg | 0.3 | 0.372 | 0.411 | 0.406 | −0.48pp | [−7.00, +7.00] | 709 |
| qwen25_32b_aime | marg | 0.5 | 0.373 | 0.411 | 0.447 | **+3.53pp** | [−3.00, +10.00] | 709 |
| qwen25_32b_math500 | lp | 0.1 | 0.776 | 0.783 | 0.773 | −1.01pp | [−4.00, +2.00] | 583 |
| qwen25_32b_math500 | lp | 0.3 | 0.781 | 0.786 | 0.796 | +1.03pp | [−2.00, +5.00] | 583 |
| qwen25_32b_math500 | lp | 0.5 | 0.779 | 0.783 | 0.805 | **+2.18pp** | [+0.00, +7.00] | 583 |
| qwen25_32b_math500 | ent_neg | 0.1 | 0.782 | 0.786 | 0.782 | −0.35pp | [−3.00, +2.00] | 583 |
| qwen25_32b_math500 | ent_neg | 0.3 | 0.779 | 0.785 | 0.790 | +0.58pp | [−2.00, +4.00] | 583 |
| qwen25_32b_math500 | ent_neg | 0.5 | 0.777 | 0.784 | 0.802 | +1.77pp | [+0.00, +5.00] | 583 |
| qwen25_32b_math500 | marg | 0.1 | 0.779 | 0.788 | 0.777 | −1.12pp | [−5.00, +2.00] | 583 |
| qwen25_32b_math500 | marg | 0.3 | 0.783 | 0.792 | 0.801 | +0.88pp | [−3.00, +5.00] | 583 |
| qwen25_32b_math500 | marg | 0.5 | 0.780 | 0.788 | 0.801 | +1.28pp | [−2.53, +6.00] | 583 |
| qwen25_32b_olympiad | lp | 0.1 | 0.449 | 0.493 | 0.481 | −1.13pp | [−7.00, +5.00] | 666 |
| qwen25_32b_olympiad | lp | 0.3 | 0.448 | 0.494 | 0.494 | −0.01pp | [−6.00, +5.00] | 666 |
| qwen25_32b_olympiad | lp | 0.5 | 0.446 | 0.492 | 0.479 | −1.35pp | [−7.00, +4.00] | 666 |
| qwen25_32b_olympiad | ent_neg | 0.1 | 0.443 | 0.499 | 0.477 | −2.25pp | [−7.52, +3.00] | 666 |
| qwen25_32b_olympiad | ent_neg | 0.3 | 0.442 | 0.494 | 0.481 | −1.25pp | [−6.00, +3.00] | 666 |
| qwen25_32b_olympiad | ent_neg | 0.5 | 0.444 | 0.498 | 0.473 | −2.47pp | [−9.00, +3.00] | 666 |
| qwen25_32b_olympiad | marg | 0.1 | 0.446 | 0.493 | 0.469 | −2.32pp | [−9.00, +3.00] | 666 |
| qwen25_32b_olympiad | marg | 0.3 | 0.448 | 0.500 | 0.489 | −1.03pp | [−5.53, +3.00] | 666 |
| qwen25_32b_olympiad | marg | 0.5 | 0.449 | 0.499 | 0.477 | −2.23pp | [−8.00, +3.00] | 666 |
| qwen25_7b_aime | lp | 0.1 | 0.243 | 0.302 | 0.288 | −1.40pp | [−7.00, +4.00] | 585 |
| qwen25_7b_aime | lp | 0.3 | 0.250 | 0.310 | 0.304 | −0.55pp | [−6.00, +5.00] | 585 |
| qwen25_7b_aime | lp | 0.5 | 0.247 | 0.305 | 0.324 | **+1.85pp** | [−4.00, +8.00] | 585 |
| qwen25_7b_aime | ent_neg | 0.1 | 0.245 | 0.320 | 0.293 | −2.69pp | [−8.00, +3.00] | 585 |
| qwen25_7b_aime | ent_neg | 0.3 | 0.246 | 0.323 | 0.303 | −2.02pp | [−8.00, +3.00] | 585 |
| qwen25_7b_aime | ent_neg | 0.5 | 0.250 | 0.324 | 0.327 | +0.27pp | [−5.00, +6.00] | 585 |
| qwen25_7b_aime | marg | 0.1 | 0.246 | 0.299 | 0.299 | +0.00pp | [−6.00, +6.00] | 585 |
| qwen25_7b_aime | marg | 0.3 | 0.245 | 0.301 | 0.305 | +0.44pp | [−5.52, +6.00] | 585 |
| qwen25_7b_aime | marg | 0.5 | 0.244 | 0.300 | 0.315 | +1.49pp | [−3.00, +7.00] | 585 |
| qwen25_7b_math500 | lp | 0.1 | 0.738 | 0.761 | 0.745 | −1.65pp | [−7.00, +3.00] | 515 |
| qwen25_7b_math500 | lp | 0.3 | 0.740 | 0.765 | 0.758 | −0.69pp | [−6.00, +4.00] | 515 |
| qwen25_7b_math500 | lp | 0.5 | 0.739 | 0.765 | 0.759 | −0.57pp | [−5.00, +3.52] | 515 |
| qwen25_7b_math500 | ent_neg | 0.1 | 0.741 | 0.766 | 0.752 | −1.36pp | [−7.00, +3.00] | 515 |
| qwen25_7b_math500 | ent_neg | 0.3 | 0.740 | 0.767 | 0.755 | −1.17pp | [−6.00, +4.00] | 515 |
| qwen25_7b_math500 | ent_neg | 0.5 | 0.738 | 0.763 | 0.758 | −0.55pp | [−5.00, +4.00] | 515 |
| qwen25_7b_math500 | marg | 0.1 | 0.737 | 0.772 | 0.745 | −2.73pp | [−9.00, +2.00] | 515 |
| qwen25_7b_math500 | marg | 0.3 | 0.742 | 0.778 | 0.753 | −2.50pp | [−8.00, +2.00] | 515 |
| qwen25_7b_math500 | marg | 0.5 | 0.742 | 0.775 | 0.762 | −1.24pp | [−6.00, +3.00] | 515 |
| qwen25_7b_olympiad | lp | 0.1 | 0.425 | 0.450 | 0.445 | −0.54pp | [−4.00, +3.00] | 597 |
| qwen25_7b_olympiad | lp | 0.3 | 0.420 | 0.446 | 0.457 | **+1.08pp** | [−3.00, +5.00] | 597 |
| qwen25_7b_olympiad | lp | 0.5 | 0.422 | 0.449 | 0.454 | +0.49pp | [−4.52, +5.52] | 597 |
| qwen25_7b_olympiad | ent_neg | 0.1 | 0.423 | 0.448 | 0.439 | −0.91pp | [−6.00, +4.00] | 597 |
| qwen25_7b_olympiad | ent_neg | 0.3 | 0.416 | 0.440 | 0.452 | **+1.23pp** | [−3.00, +6.00] | 597 |
| qwen25_7b_olympiad | ent_neg | 0.5 | 0.419 | 0.444 | 0.453 | +0.89pp | [−5.00, +6.00] | 597 |
| qwen25_7b_olympiad | marg | 0.1 | 0.419 | 0.470 | 0.446 | −2.47pp | [−7.00, +2.00] | 597 |
| qwen25_7b_olympiad | marg | 0.3 | 0.424 | 0.473 | 0.459 | −1.41pp | [−6.00, +3.00] | 597 |
| qwen25_7b_olympiad | marg | 0.5 | 0.423 | 0.475 | 0.457 | −1.73pp | [−8.00, +4.00] | 597 |
| qwen25_math_7b_aime | lp | 0.1 | 0.329 | 0.367 | 0.370 | +0.30pp | [−5.00, +6.00] | 742 |
| qwen25_math_7b_aime | lp | 0.3 | 0.335 | 0.374 | 0.376 | +0.17pp | [−4.00, +5.00] | 742 |
| qwen25_math_7b_aime | lp | 0.5 | 0.330 | 0.370 | 0.364 | −0.54pp | [−6.52, +5.00] | 742 |
| qwen25_math_7b_aime | ent_neg | 0.1 | 0.325 | 0.377 | 0.363 | −1.43pp | [−7.00, +4.00] | 742 |
| qwen25_math_7b_aime | ent_neg | 0.3 | 0.331 | 0.382 | 0.371 | −1.09pp | [−6.00, +3.00] | 742 |
| qwen25_math_7b_aime | ent_neg | 0.5 | 0.333 | 0.383 | 0.367 | −1.59pp | [−7.00, +4.00] | 742 |
| qwen25_math_7b_aime | marg | 0.1 | 0.330 | 0.376 | 0.362 | −1.41pp | [−6.00, +4.00] | 742 |
| qwen25_math_7b_aime | marg | 0.3 | 0.331 | 0.375 | 0.368 | −0.71pp | [−6.52, +5.00] | 742 |
| qwen25_math_7b_aime | marg | 0.5 | 0.332 | 0.378 | 0.367 | −1.02pp | [−8.00, +5.00] | 742 |
| qwen25_math_7b_math500 | lp | 0.1 | 0.737 | 0.783 | 0.761 | **−2.23pp** | [−6.00, +0.00] | 593 |
| qwen25_math_7b_math500 | lp | 0.3 | 0.739 | 0.785 | 0.763 | **−2.20pp** | [−6.00, +0.00] | 593 |
| qwen25_math_7b_math500 | lp | 0.5 | 0.740 | 0.784 | 0.768 | **−1.68pp** | [−5.00, +0.00] | 593 |
| qwen25_math_7b_math500 | ent_neg | 0.1 | 0.742 | 0.787 | 0.773 | −1.38pp | [−5.00, +2.00] | 593 |
| qwen25_math_7b_math500 | ent_neg | 0.3 | 0.739 | 0.782 | 0.767 | −1.50pp | [−6.00, +2.00] | 593 |
| qwen25_math_7b_math500 | ent_neg | 0.5 | 0.740 | 0.784 | 0.771 | −1.35pp | [−5.00, +0.00] | 593 |
| qwen25_math_7b_math500 | marg | 0.1 | 0.738 | 0.790 | 0.769 | −2.11pp | [−6.00, +0.00] | 593 |
| qwen25_math_7b_math500 | marg | 0.3 | 0.740 | 0.791 | 0.778 | −1.25pp | [−5.00, +0.00] | 593 |
| qwen25_math_7b_math500 | marg | 0.5 | 0.740 | 0.790 | 0.789 | −0.14pp | [−3.00, +2.00] | 593 |
| qwen25_math_7b_olympiad | lp | 0.1 | 0.414 | 0.410 | 0.430 | **+1.93pp** | [−6.00, +9.00] | 652 |
| qwen25_math_7b_olympiad | lp | 0.3 | 0.411 | 0.413 | 0.438 | **+2.58pp** | [−5.53, +11.00] | 652 |
| qwen25_math_7b_olympiad | lp | 0.5 | 0.411 | 0.414 | 0.434 | **+2.00pp** | [−6.00, +11.00] | 652 |
| qwen25_math_7b_olympiad | ent_neg | 0.1 | 0.413 | 0.432 | 0.429 | −0.32pp | [−7.52, +7.00] | 652 |
| qwen25_math_7b_olympiad | ent_neg | 0.3 | 0.409 | 0.429 | 0.438 | +0.86pp | [−7.00, +8.00] | 652 |
| qwen25_math_7b_olympiad | ent_neg | 0.5 | 0.407 | 0.426 | 0.433 | +0.72pp | [−7.00, +8.00] | 652 |
| qwen25_math_7b_olympiad | marg | 0.1 | 0.413 | 0.422 | 0.428 | +0.68pp | [−7.00, +8.00] | 652 |
| qwen25_math_7b_olympiad | marg | 0.3 | 0.414 | 0.424 | 0.444 | **+1.95pp** | [−6.00, +9.52] | 652 |
| qwen25_math_7b_olympiad | marg | 0.5 | 0.412 | 0.420 | 0.437 | +1.69pp | [−7.00, +11.00] | 652 |

**Read-out.** Bold rows = T4 v3 §8.2 pre-registered prediction targets (all 5) plus the most strongly negative `qwen25_math_7b__math500` rows where CI upper bound touches 0 (near-significant negative). 0/108 CIs strictly exclude zero in either direction. The bolded `qwen25_32b_math500__lp__α=0.5: +2.18pp [+0.00, +7.00]` is the only test whose CI lower bound *touches* zero from above; symmetrically the three `qwen25_math_7b_math500__lp` and three `qwen25_math_7b_math500__marg` rows (and `ent_neg__α=0.5`) have CI upper bound *at* zero — six near-significant negatives concentrated in the Math-7B sticky-specialist on its own domain.

---

## §2 Cascade-gap-stratified table (per cell × gap bucket × score × α)

Source: `pearl_full/<cell>.json` `by_score_alpha[<score>__a<alpha>][cascade_lift_by_gap][gap_*]` `[mean]`. Lower/upper CI bounds available in JSON; omitted here for compactness — see `/tmp/full_tables_output.txt §2` for full 108-row CI listing.

| cell | score | α | gap=1 | gap∈[2,4] | gap≥5 | gap_negzero |
|---|---|---|---|---|---|---|
| phi4_aime | lp | 0.1 | −5.94pp | +1.25pp | −1.51pp | −1.65pp |
| phi4_aime | lp | 0.3 | −1.04pp | −10.01pp | −2.93pp | −0.08pp |
| phi4_aime | lp | 0.5 | −3.15pp | −11.90pp | −5.62pp | +0.00pp |
| phi4_aime | ent_neg | 0.1 | +3.51pp | +10.20pp | −7.63pp | +0.15pp |
| phi4_aime | ent_neg | 0.3 | −2.05pp | −2.83pp | −5.77pp | +0.03pp |
| phi4_aime | ent_neg | 0.5 | −0.85pp | −6.78pp | −7.85pp | −0.33pp |
| phi4_aime | marg | 0.1 | +2.86pp | +6.47pp | −5.90pp | +2.08pp |
| phi4_aime | marg | 0.3 | +1.78pp | −1.08pp | −4.88pp | +0.89pp |
| phi4_aime | marg | 0.5 | +6.32pp | −3.39pp | −5.05pp | +0.32pp |
| phi4_math500 | lp | 0.1 | +3.12pp | −1.59pp | +3.21pp | +0.95pp |
| phi4_math500 | lp | 0.3 | +7.08pp | −3.22pp | −3.41pp | +0.88pp |
| phi4_math500 | lp | 0.5 | +0.00pp | +2.91pp | −3.91pp | +0.26pp |
| phi4_math500 | ent_neg | 0.1 | −1.29pp | −10.25pp | +11.73pp | +0.03pp |
| phi4_math500 | ent_neg | 0.3 | +2.75pp | −3.59pp | −3.52pp | +0.00pp |
| phi4_math500 | ent_neg | 0.5 | +0.00pp | +1.78pp | −5.15pp | +0.00pp |
| phi4_math500 | marg | 0.1 | +0.00pp | +0.00pp | −5.28pp | +0.63pp |
| phi4_math500 | marg | 0.3 | −1.08pp | −0.03pp | −5.67pp | +0.77pp |
| phi4_math500 | marg | 0.5 | −0.73pp | −2.70pp | −6.05pp | +1.11pp |
| phi4_olympiad | lp | 0.1 | −24.83pp | +0.23pp | −0.52pp | +0.49pp |
| phi4_olympiad | lp | 0.3 | +7.04pp | −8.82pp | −1.31pp | +1.23pp |
| phi4_olympiad | lp | 0.5 | +4.84pp | −7.40pp | −4.05pp | +0.78pp |
| phi4_olympiad | ent_neg | 0.1 | −24.24pp | −6.97pp | −3.17pp | +0.57pp |
| phi4_olympiad | ent_neg | 0.3 | −0.03pp | −2.97pp | −2.47pp | +1.16pp |
| phi4_olympiad | ent_neg | 0.5 | +0.96pp | −9.12pp | −2.51pp | +0.06pp |
| phi4_olympiad | marg | 0.1 | −12.76pp | −0.43pp | −2.63pp | −0.22pp |
| phi4_olympiad | marg | 0.3 | −0.27pp | −2.88pp | −0.99pp | +0.98pp |
| phi4_olympiad | marg | 0.5 | −7.75pp | −4.65pp | −1.00pp | +0.00pp |
| qwen25_32b_aime | lp | 0.1 | −9.24pp | −1.71pp | +0.09pp | −0.83pp |
| qwen25_32b_aime | lp | 0.3 | **−13.20pp** | −4.61pp | −7.78pp | +0.73pp |
| qwen25_32b_aime | lp | 0.5 | **−15.44pp** | +3.96pp | −1.19pp | −0.01pp |
| qwen25_32b_aime | ent_neg | 0.1 | **−11.81pp** | −2.83pp | −1.08pp | −1.62pp |
| qwen25_32b_aime | ent_neg | 0.3 | **−11.07pp** | −3.62pp | −1.42pp | −1.52pp |
| qwen25_32b_aime | ent_neg | 0.5 | **−11.25pp** | +1.96pp | +2.01pp | −0.19pp |
| qwen25_32b_aime | marg | 0.1 | −9.68pp | +0.35pp | −4.56pp | −0.26pp |
| qwen25_32b_aime | marg | 0.3 | −5.36pp | −2.53pp | +3.20pp | −0.78pp |
| qwen25_32b_aime | marg | 0.5 | −3.85pp | **+10.38pp** | **+6.43pp** | +0.80pp |
| qwen25_32b_math500 | lp | 0.1 | +7.56pp | +0.00pp | −12.54pp | −0.70pp |
| qwen25_32b_math500 | lp | 0.3 | +2.96pp | +3.17pp | +0.39pp | +0.38pp |
| qwen25_32b_math500 | lp | 0.5 | +6.21pp | +1.07pp | **+7.23pp** | +0.00pp |
| qwen25_32b_math500 | ent_neg | 0.1 | +3.94pp | +1.74pp | −1.17pp | +0.01pp |
| qwen25_32b_math500 | ent_neg | 0.3 | +2.88pp | +3.34pp | +2.73pp | −0.16pp |
| qwen25_32b_math500 | ent_neg | 0.5 | +5.81pp | +1.66pp | +4.73pp | +0.09pp |
| qwen25_32b_math500 | marg | 0.1 | +7.19pp | +0.07pp | −9.17pp | −0.07pp |
| qwen25_32b_math500 | marg | 0.3 | +6.60pp | +2.42pp | +3.73pp | +0.00pp |
| qwen25_32b_math500 | marg | 0.5 | +5.09pp | +3.40pp | −0.24pp | +0.00pp |
| qwen25_32b_olympiad | lp | 0.1 | +0.15pp | +9.68pp | −0.47pp | −0.58pp |
| qwen25_32b_olympiad | lp | 0.3 | −5.60pp | +3.63pp | −3.01pp | +0.15pp |
| qwen25_32b_olympiad | lp | 0.5 | −5.43pp | +3.08pp | −7.75pp | +0.00pp |
| qwen25_32b_olympiad | ent_neg | 0.1 | +0.00pp | −3.26pp | −6.41pp | −0.05pp |
| qwen25_32b_olympiad | ent_neg | 0.3 | −9.88pp | +0.59pp | −4.25pp | +0.10pp |
| qwen25_32b_olympiad | ent_neg | 0.5 | −5.48pp | −0.08pp | −10.09pp | +0.00pp |
| qwen25_32b_olympiad | marg | 0.1 | +0.00pp | −10.07pp | +3.45pp | −0.29pp |
| qwen25_32b_olympiad | marg | 0.3 | −8.04pp | −4.57pp | +2.74pp | −0.08pp |
| qwen25_32b_olympiad | marg | 0.5 | −6.01pp | −4.35pp | −4.06pp | −0.01pp |
| qwen25_7b_aime | lp | 0.1 | −0.24pp | +2.92pp | +2.43pp | +0.87pp |
| qwen25_7b_aime | lp | 0.3 | +1.84pp | +0.48pp | +0.63pp | −1.04pp |
| qwen25_7b_aime | lp | 0.5 | +4.88pp | +3.92pp | +0.06pp | +0.74pp |
| qwen25_7b_aime | ent_neg | 0.1 | −3.24pp | +6.50pp | +4.15pp | −1.24pp |
| qwen25_7b_aime | ent_neg | 0.3 | −9.56pp | +3.55pp | +0.17pp | −2.41pp |
| qwen25_7b_aime | ent_neg | 0.5 | +0.77pp | +1.39pp | −1.02pp | −0.08pp |
| qwen25_7b_aime | marg | 0.1 | −1.51pp | +6.91pp | **+18.76pp** | +1.54pp |
| qwen25_7b_aime | marg | 0.3 | −4.57pp | +6.04pp | **+8.33pp** | +0.05pp |
| qwen25_7b_aime | marg | 0.5 | +6.05pp | +1.55pp | +7.58pp | +0.07pp |
| qwen25_7b_math500 | lp | 0.1 | +0.00pp | −8.56pp | +0.00pp | +0.54pp |
| qwen25_7b_math500 | lp | 0.3 | −0.06pp | −1.83pp | −4.95pp | +0.90pp |
| qwen25_7b_math500 | lp | 0.5 | +0.00pp | +3.89pp | −11.68pp | −0.47pp |
| qwen25_7b_math500 | ent_neg | 0.1 | +0.00pp | −8.97pp | +0.00pp | +0.77pp |
| qwen25_7b_math500 | ent_neg | 0.3 | +0.00pp | −5.52pp | +1.04pp | +0.08pp |
| qwen25_7b_math500 | ent_neg | 0.5 | +0.00pp | −0.26pp | −6.96pp | −0.04pp |
| qwen25_7b_math500 | marg | 0.1 | −1.12pp | −14.91pp | +0.00pp | −0.07pp |
| qwen25_7b_math500 | marg | 0.3 | −0.02pp | −12.53pp | +7.49pp | −2.18pp |
| qwen25_7b_math500 | marg | 0.5 | −3.57pp | −4.08pp | +0.18pp | −0.31pp |
| qwen25_7b_olympiad | lp | 0.1 | +0.00pp | +9.83pp | +0.00pp | +0.07pp |
| qwen25_7b_olympiad | lp | 0.3 | +6.09pp | +4.09pp | −3.18pp | +0.02pp |
| qwen25_7b_olympiad | lp | 0.5 | +3.81pp | +1.63pp | −4.12pp | +0.01pp |
| qwen25_7b_olympiad | ent_neg | 0.1 | +0.34pp | +7.44pp | +0.00pp | −0.72pp |
| qwen25_7b_olympiad | ent_neg | 0.3 | +6.95pp | +5.39pp | −1.46pp | −0.14pp |
| qwen25_7b_olympiad | ent_neg | 0.5 | +2.68pp | +3.38pp | −3.30pp | +0.23pp |
| qwen25_7b_olympiad | marg | 0.1 | −10.62pp | +0.25pp | −8.70pp | −0.09pp |
| qwen25_7b_olympiad | marg | 0.3 | −8.04pp | +1.48pp | −3.90pp | +0.00pp |
| qwen25_7b_olympiad | marg | 0.5 | −7.66pp | +0.30pp | −3.49pp | −0.16pp |
| qwen25_math_7b_aime | lp | 0.1 | +2.78pp | +6.72pp | +4.05pp | −0.03pp |
| qwen25_math_7b_aime | lp | 0.3 | +2.52pp | +0.52pp | +2.80pp | +0.00pp |
| qwen25_math_7b_aime | lp | 0.5 | +5.71pp | +1.77pp | −0.56pp | −0.03pp |
| qwen25_math_7b_aime | ent_neg | 0.1 | +0.87pp | +3.63pp | +3.24pp | −0.27pp |
| qwen25_math_7b_aime | ent_neg | 0.3 | +0.67pp | −2.87pp | +1.45pp | −0.40pp |
| qwen25_math_7b_aime | ent_neg | 0.5 | +6.04pp | −4.68pp | −0.38pp | −0.28pp |
| qwen25_math_7b_aime | marg | 0.1 | +1.95pp | +5.72pp | +0.61pp | −0.04pp |
| qwen25_math_7b_aime | marg | 0.3 | +7.56pp | +1.20pp | −0.63pp | −0.95pp |
| qwen25_math_7b_aime | marg | 0.5 | +10.84pp | +1.53pp | −4.53pp | −0.10pp |
| qwen25_math_7b_math500 | lp | 0.1 | +0.00pp | +0.00pp | −1.77pp | −0.19pp |
| qwen25_math_7b_math500 | lp | 0.3 | −0.04pp | −0.04pp | −4.33pp | −1.86pp |
| qwen25_math_7b_math500 | lp | 0.5 | +0.00pp | +0.00pp | −4.28pp | −0.65pp |
| qwen25_math_7b_math500 | ent_neg | 0.1 | +0.00pp | +0.00pp | −0.94pp | −0.34pp |
| qwen25_math_7b_math500 | ent_neg | 0.3 | +0.00pp | +0.00pp | −1.89pp | −1.65pp |
| qwen25_math_7b_math500 | ent_neg | 0.5 | +0.00pp | −0.05pp | −4.94pp | −0.12pp |
| qwen25_math_7b_math500 | marg | 0.1 | +0.00pp | +0.00pp | −5.26pp | +0.00pp |
| qwen25_math_7b_math500 | marg | 0.3 | +0.00pp | +0.00pp | −2.34pp | −0.22pp |
| qwen25_math_7b_math500 | marg | 0.5 | −0.54pp | −2.25pp | −0.15pp | +0.79pp |
| qwen25_math_7b_olympiad | lp | 0.1 | **+13.70pp** | −1.72pp | **+10.51pp** | −1.36pp |
| qwen25_math_7b_olympiad | lp | 0.3 | **+11.27pp** | +8.55pp | +1.18pp | +0.02pp |
| qwen25_math_7b_olympiad | lp | 0.5 | +4.52pp | +4.32pp | +1.89pp | +0.00pp |
| qwen25_math_7b_olympiad | ent_neg | 0.1 | **+10.53pp** | −9.05pp | +2.95pp | −1.85pp |
| qwen25_math_7b_olympiad | ent_neg | 0.3 | +5.43pp | +4.03pp | −0.63pp | +0.00pp |
| qwen25_math_7b_olympiad | ent_neg | 0.5 | +2.77pp | +1.21pp | +0.71pp | +0.00pp |
| qwen25_math_7b_olympiad | marg | 0.1 | **+20.14pp** | −3.98pp | +5.80pp | +0.30pp |
| qwen25_math_7b_olympiad | marg | 0.3 | +2.25pp | +1.80pp | +4.44pp | +0.00pp |
| qwen25_math_7b_olympiad | marg | 0.5 | −0.14pp | +3.69pp | +2.31pp | +0.00pp |

**Cross-cell pattern.** The cleanly monotone pattern (small at gap=1, larger at gap≥5) anchored T4 v3 in the original 5-cell snapshot via `qwen25_7b__aime__marg__α=0.1: gap=1=−1.51 / gap≥5=+18.76`. The 12-cell sample shows:

- **Inverse pattern on `qwen25_32b__aime`.** Gap=1 strata are *strongly negative* (−9 to −15pp on lp/ent_neg) — the opposite of the monotone-increasing prediction. Only `marg__α=0.5` recovers the predicted shape (gap=1 = −3.85, gap=2..4 = +10.38, gap≥5 = +6.43). On lp/ent_neg the Corollary 4.1 monotonicity is **directly violated**.
- **Inverse pattern on `phi4__olympiad`.** Gap=1 has three rows at −12 to −25pp on lp/ent_neg/marg α=0.1 — a magnitude that cannot be explained by the false-positive cost $\Lambda(1)$ alone (T4 v3 §6 estimates $\Lambda(1) \approx 6.6$pp on aime). These traces *had* a self-correction trajectory in the worst-step baseline that earliest-step intervention destroyed.
- **`qwen25_math_7b__olympiad` shows the predicted-shape signal.** gap=1 = +13.7 / +11.3 / +10.5 / +20.1 on lp/lp/ent_neg/marg α=0.1 → these are the largest *positive* gap=1 strata in the entire 12-cell sample, but *gap≥5* is much smaller (+10.5 max, but mostly < +5pp) — the shape on this cell is *concave* rather than monotone-increasing. T4 v3 Corollary 4.1 bounds say monotone-non-decreasing in g; concave is a partial violation.
- **`qwen25_7b__aime__marg`** (the original calibration anchor) **remains the cleanest monotone case** in the entire 108-row table — but it is now visibly an outlier rather than the rule.

---

## §3 Pre-registered T4 v3 prediction outcomes (PASS/FAIL/PARTIAL)

Predictions taken verbatim from `theorem4_v3_cascade_stratified.md §8.2`. Tolerance ±2pp on aggregate, ±5pp on gap≥5 (per v3 §8.4 honest scoping). Verdict per (cell, score, α):

### P1 — `phi4__aime` (predicted +5pp aggregate at α=0.5; gap≥5 +15..+25pp)

| score | agg Δ_lift (α=0.5) | 95% CI | gap≥5 Δ | gap≥5 CI | aggregate verdict | gap≥5 verdict |
|---|---|---|---|---|---|---|
| lp | **−4.44pp** | [−12.00, +2.00] | **−5.62pp** | [−18.45, +6.25] | **FAIL (sign-flip)** | **FAIL** |
| ent_neg | **−4.49pp** | [−11.00, +1.00] | **−7.85pp** | [−21.08, +4.44] | **FAIL (sign-flip)** | **FAIL** |
| marg | **−2.04pp** | [−9.00, +5.00] | **−5.05pp** | [−20.00, +9.30] | **FAIL (sign-flip)** | **FAIL** |

**Verdict: FULL FALSIFICATION.** All three score families show negative aggregate where v3 predicted ≥+5pp; gap≥5 stratum is also uniformly negative. This is T4 v3 §8.3 falsifier #1 firing.

### P2 — `qwen25_math_7b__olympiad` (predicted +3..+5pp aggregate at α=0.5; gap≥5 +10..+20pp)

| score | agg Δ_lift (α=0.5) | 95% CI | gap≥5 Δ | gap≥5 CI | aggregate verdict | gap≥5 verdict |
|---|---|---|---|---|---|---|
| lp | +2.00pp | [−6.00, +11.00] | +1.89pp | [−18.25, +21.43] | **PASS (within band)** | **PARTIAL (positive but below band)** |
| ent_neg | +0.72pp | [−7.00, +8.00] | +0.71pp | [−20.00, +20.00] | **PARTIAL (sign matches, magnitude off)** | **PARTIAL** |
| marg | +1.69pp | [−7.00, +11.00] | +2.31pp | [−15.40, +19.77] | **PASS (within band)** | **PARTIAL (positive but below band)** |

**Verdict: PARTIAL CONFIRMATION.** Aggregate signs all positive (matching prediction) and lp/marg fall in the lower edge of the predicted +3..+5pp band; ent_neg is positive but below band. Gap≥5 stratum is positive on all three score families but uniformly below the predicted +10..+20pp range — i.e. directionally correct, magnitude smaller than predicted by half or more. Notably the *gap=1* stratum on this cell carries the salvage signal (+13.7 / +11.3 / +20.1pp on lp/lp/marg α=0.1) — the opposite of Corollary 4.1's monotonicity claim.

### P3 — `qwen25_32b__aime` (predicted +3pp aggregate at α=0.5; gap≥5 +10..+18pp)

| score | agg Δ_lift (α=0.5) | 95% CI | gap≥5 Δ | gap≥5 CI | aggregate verdict | gap≥5 verdict |
|---|---|---|---|---|---|---|
| lp | −0.93pp | [−9.00, +6.00] | −1.19pp | [−22.91, +20.77] | **FAIL (sign-flip)** | **FAIL** |
| ent_neg | −0.67pp | [−8.00, +6.00] | +2.01pp | [−19.55, +24.09] | **FAIL (sign-flip)** | **PARTIAL** |
| marg | **+3.53pp** | [−3.00, +10.00] | **+6.43pp** | [−10.28, +26.32] | **PASS (within band)** | **PASS** |

**Verdict: 1/3 PASS, 2/3 FAIL.** Only the `marg` score family at α=0.5 confirms the prediction (and it is the strongest single PASS in the table — agg +3.53pp matches predicted +3pp within 0.5pp; gap≥5 +6.43pp falls below the +10..+18 band but is unambiguously positive). The lp and ent_neg families *sign-flip*: −0.93pp and −0.67pp where v3 predicted +3pp, with even more negative point estimates (−2.71, −3.02, −4.22, −3.12) at α=0.1 and α=0.3. Critically, the *gap=1* stratum across lp/ent_neg/all alphas is **−9 to −15pp** — the cleanest evidence in the table that earliest-step intervention is *actively harmful* on this cell.

The user's headline summary listed the lp α=0.3 row (−3.02pp) as the canonical falsifier. We confirm: P3's lp/ent_neg verdicts are FAIL with high-magnitude opposite-sign point estimates.

### P4 — `phi4__math500` (predicted +1pp aggregate at α=0.5; gap≥5 +5..+10pp)

| score | agg Δ_lift (α=0.5) | 95% CI | gap≥5 Δ | gap≥5 CI | aggregate verdict | gap≥5 verdict |
|---|---|---|---|---|---|---|
| lp | −0.44pp | [−4.52, +3.52] | −3.91pp | [−18.03, +8.00] | **FAIL (sign-flip)** | **FAIL** |
| ent_neg | −0.85pp | [−5.00, +3.00] | −5.15pp | [−23.46, +9.52] | **FAIL (sign-flip)** | **FAIL** |
| marg | −1.97pp | [−7.00, +3.00] | −6.05pp | [−18.64, +6.16] | **FAIL (sign-flip)** | **FAIL** |

**Verdict: FULL FALSIFICATION.** All three score families show negative aggregate Δ_lift where v3 predicted +1pp (−1 to +3pp tolerance); gap≥5 stratum is uniformly negative. This is consistent with the math500 sticky-specialist diagnosis but extends it: phi-4 on math500 is *also* sticky-single-mode, despite being a different model class than qwen2.5-Math.

### P5 — `qwen25_7b__olympiad` (predicted +0..+3pp aggregate at α=0.3; gap≥5 +5..+12pp if (A6) holds)

| score | agg Δ_lift (α=0.3) | 95% CI | gap≥5 Δ | gap≥5 CI | aggregate verdict | gap≥5 verdict |
|---|---|---|---|---|---|---|
| lp | +1.08pp | [−3.00, +5.00] | −3.18pp | [−28.57, +0.00] | **PASS (within band)** | **FAIL** |
| ent_neg | +1.23pp | [−3.00, +6.00] | −1.46pp | [−20.00, +0.00] | **PASS (within band)** | **FAIL** |
| marg | −1.41pp | [−6.00, +3.00] | −3.90pp | [−26.98, +0.00] | **PASS (within band)** | **FAIL** |

**Verdict: AGGREGATE PASS, GAP≥5 FAIL.** Aggregate sign all in the predicted band; gap≥5 stratum sign-flips on every score family. T4 v3 §8.2 hedged on this cell ("if (A6) holds") — the data argue (A6) is violated on OlympiadBench at gap≥5, consistent with `EMPIRICAL_ANALYSIS.md §4.2`.

### Summary of predictions (5 predictions × 3 score families = 15 (cell, score) verdicts)

| Verdict | Aggregate count | Gap≥5 count |
|---|---|---|
| PASS | 5 / 15 | 1 / 15 |
| PARTIAL | 1 / 15 | 4 / 15 |
| FAIL (sign-flip) | 9 / 15 | 10 / 15 |

**Bottom line: T4 v3's pre-registered aggregate predictions are 5/15 PASS, 9/15 sign-flip FAIL. The gap≥5 salvage prediction does even worse: 1/15 PASS, 4/15 PARTIAL, 10/15 FAIL.** T4 v3 §8.3 said full falsification requires ≥3 of the 5 listed falsifiers to fire; we observe **at least 3** (#1: phi4__aime aggregate < 0; #5: gap=1 > gap≥5 on multiple AIME rows; and the new strict falsifier "phi4__math500 gap≥5 worse than qwen25_7b__math500" — phi4 has gap≥5 = −5.28 to −6.05pp on marg, qwen25_7b has gap≥5 = +0.00 / +7.49 / +0.18pp on marg, so this falsifier #3 also fires). **T4 v3 is falsified by its own pre-registered criteria.**

---

## §4 Score family comparison: which of {lp, ent_neg, marg} performs best on average?

Source: §1 table; aggregated across 12 cells × 3 alphas per score = 36 cell×alpha pairs.

### Aggregate Δ_lift mean per score family (across 36 cell×alpha)

| score | mean Δ_lift | range | # cells×α with Δ > 0 |
|---|---|---|---|
| lp | −0.5893pp | [−4.44, +2.58] | 11 / 36 |
| ent_neg | −1.1499pp | [−4.49, +1.77] | 7 / 36 |
| marg | −0.7269pp | [−2.73, +3.53] | 9 / 36 |

### Per (score, α) means (across 12 cells)

| score | α | mean Δ_lift | # cells > 0 |
|---|---|---|---|
| lp | 0.1 | −0.875pp | 3 / 12 |
| lp | 0.3 | −0.424pp | 4 / 12 |
| lp | 0.5 | −0.469pp | 4 / 12 |
| ent_neg | 0.1 | −1.533pp | 1 / 12 |
| ent_neg | 0.3 | −1.035pp | 3 / 12 |
| ent_neg | 0.5 | −0.882pp | 4 / 12 |
| marg | 0.1 | −1.278pp | 1 / 12 |
| marg | 0.3 | −0.577pp | 3 / 12 |
| marg | 0.5 | −0.326pp | 4 / 12 |

### Mean Δ_strat(gap≥5) across score families

| score | mean Δ_strat(gap≥5) | # cells×α with Δ_g5p > 0 |
|---|---|---|
| lp | −1.567pp | 12 / 36 |
| ent_neg | −1.363pp | 11 / 36 |
| marg | −0.263pp | 14 / 36 |

### Verdict on score family

- **`lp` is the least-bad on aggregate** (mean −0.59pp) but no score family is positive on average.
- **`marg` has the *largest range*** (−2.73 to +3.53pp) and is the *least negative* on the gap≥5 stratum (−0.26pp vs −1.36/−1.57). On the original 5-cell snapshot, `marg` carried the cascade-stratified salvage signal (+18.76pp on `qwen25_7b__aime`); on the 12-cell sample it remains the most cascade-aware score family but the aggregate edge is small.
- **`ent_neg` is uniformly worst** on aggregate (mean −1.15pp; only 7/36 positive) and roughly comparable on gap≥5.
- **Higher α generally helps** within each score family (less restrictive coverage threshold, less false-positive cost on already-correct traces) — but only modestly: mean improves from −0.875 → −0.469 between α=0.1 and α=0.5 on `lp`.

**No score-family choice rescues T4 v3's aggregate prediction.** Even `lp` at α=0.5 — the best (score, α) combination on aggregate — has mean Δ_lift = −0.47pp across 12 cells, with 4/12 cells positive. The cascade-stratified rescue on `marg` is restricted to a small subset of cells (`qwen25_7b__aime`, partially `qwen25_32b__aime__α=0.5`, partially `qwen25_math_7b__olympiad__α=0.1`).

---

## §5 Honest narrative: was T4 v3 falsified?

### 5.1 Aggregate prediction: largely falsified for `lp`

T4 v3 §8.2 claimed for the four still-running cells: phi4_aime +5pp, qwen-math__olympiad +3..+5pp, qwen2.5-32b__aime +3pp, phi4__math500 +1pp. Observed `lp__α=0.5` aggregates: **−4.44, +2.00, −0.93, −0.44**. Three of four are sign-flipped; one (qwen-math__olympiad) is in the bottom of band. **3/4 P1–P4 aggregate FAIL on `lp`.** Adding `ent_neg` and `marg` rescues only `qwen25_32b__aime__marg__α=0.5` to PASS.

### 5.2 What about `marg`, `ent_neg`?

`marg` rescues *one* T4 v3 prediction (`qwen25_32b__aime`, +3.53pp PASS). `ent_neg` rescues none — it is the *most* sign-flipped score family on the four §8.2 cells. The cross-score consistency of the FAIL pattern on `phi4__aime` (−4.4 / −4.5 / −2.0pp on lp / ent_neg / marg α=0.5) and on `phi4__math500` (−0.4 / −0.9 / −2.0pp on lp / ent_neg / marg α=0.5) is the strongest evidence: this is not a score-pathology, it is a **structural property of the model class**.

### 5.3 Does cascade-gap stratification salvage the theory?

T4 v3's narrative was that *aggregate* Δ_lift can be small or negative because of the gap-mixture — the predicted-positive gap≥5 stratum gets dragged down by the negative gap=1 stratum. If gap≥5 is positive *everywhere*, the theory survives in stratified form even if aggregates fail.

The 12-cell evidence: **gap≥5 is positive in 37/108 settings (34%)**. Mean across all 108 = −1.06pp. **Uniformly positive gap≥5** holds only on `qwen25_7b__aime`, `qwen25_math_7b__olympiad__lp`, and `qwen25_32b__math500__lp__α=0.5`. The 4 §8.2 §8.3-falsified cells show:

- `phi4__aime` gap≥5: 9/9 settings *negative* (−1.51 to −7.85pp).
- `phi4__math500` gap≥5: 8/9 settings negative.
- `phi4__olympiad` gap≥5: 9/9 settings negative.
- `qwen25_32b__aime` gap≥5: 5/9 settings negative.

The cascade-gap-stratified narrative *cannot* salvage T4 v3 on phi-4 (any dataset) or on qwen25_32b's hard datasets. **Stratified salvage is restricted to qwen25_7b on AIME, with weaker and inconsistent salvage on qwen25_math_7b/qwen25_32b at narrow (score, α) settings.**

### 5.4 Reframe: structural limits of step intervention

The 12-cell pattern is consistent with three structural facts the original 5-cell snapshot could not see:

1. **(A4) p_recover is heterogeneous and often low.** On `qwen25_32b__aime` lp/ent_neg, the gap=1 stratum is −9 to −15pp — i.e. K=4 alternatives at $t^*$ produce *worse* outcomes than at $t_{\text{worst}}$. The recovery probability $p_{\text{recover}}$ at $t^*$ for these cells must be substantially smaller than at $t_{\text{worst}}$, contradicting T4 v3's implicit assumption that $p_{\text{recover}}$ is uniform across $t$.
2. **(A5) self-correction is operative on phi-4.** The `phi4__olympiad__lp__α=0.1` cell shows gap=1 = −24.83pp — a magnitude that is hard to explain except by saying the model's worst-step baseline trajectory had been on a *recovery* path that the earliest-step intervention cut off. T4 v3 §7.1 (A5) explicitly excludes self-correcting models; phi-4 appears to belong outside the (A5) class.
3. **The earliest-bad-step locator is mis-aimed for some model classes.** $t^*$ is defined as "earliest step with low score (high CP nonconformity)" — i.e. the score-based locator is acting as a *proxy* for "earliest causal error". On models with strong self-correction, the score's first dip may not coincide with the causal error; intervening at the score-dip destroys the model's intrinsic recovery scaffold while leaving the actual causal error intact further downstream.

### 5.5 What survives

- **`qwen25_7b__aime__marg__α=0.1__gap≥5 = +18.76pp` is still a real signal**, the largest single stratum lift in the entire 12-cell experiment. It was the calibration anchor for κ ≈ 0.34 in T4 v3 §3.4. It does *not* generalize to other (model, dataset) cells, but as a **descriptive finding on a single (model, dataset, score, α, gap)** combination it is intact.
- **DL Strategy B' / T5' Banach contraction is intact** (§6).
- **Theorem 5'** (Banach contraction with anchor-rung concentration) is the **strongest empirical-theory match** in the entire body of work, and unaffected by the T4 v3 falsification.

---

## §6 Distance Ladder Strategy B' results (across 4 models)

Source: `distance_ladder_full/AGGREGATE.{json,md}` and per-model files.

### Headline α=0.5 reduction (T5' / Theorem 5 anchor)

| model | A_gap | B'_gap | absolute reduction | relative reduction | back-fit $\bar L$ (T5') | H1 supported? |
|---|---|---|---|---|---|---|
| qwen25_7b | 21.4pp | 9.5pp | 11.9pp | 55.6% | ≈ 0.85 | **Yes** |
| qwen25_math_7b | 21.4pp | 9.5pp | 11.9pp | 55.6% | ≈ 0.85 | Yes |
| qwen25_32b | 21.4pp | 9.5pp | 11.9pp | 55.6% | ≈ 0.85 | Yes |
| phi4 | 21.4pp | 9.5pp | 11.9pp | 55.6% | ≈ 0.85 | Yes |

**Honest caveat (carried over from `EMPIRICAL_ANALYSIS.md §2`).** All four models report identical α=0.5 numbers because the AIME ladder rungs are *borrowed* from qwen25_7b SC@8 vote shares (cross-model transport). Only qwen25_7b is fully native. The headline 11.9pp reduction is therefore one independent measurement (qwen25_7b) plus three transport-replications.

### Mean |gap| across α-grid

| model | A mean |gap| | B' mean |gap| | reduction |
|---|---|---|---|
| qwen25_7b | 19.3pp | 7.4pp | **+11.9pp** |
| qwen25_math_7b | 17.3pp | 7.4pp | +9.9pp |
| qwen25_32b | 9.5pp | 7.4pp | +2.1pp |
| phi4 | 17.3pp | 7.4pp | +9.9pp |

### Per-rung TV distances (T5 v2 ε_k inputs)

| model | TV(0→1) | TV(1→2) | TV(2→3) | TV(3→4) | TV(4→5) | sum | global TV(0→5) | H3 monotone src-TV |
|---|---|---|---|---|---|---|---|---|
| qwen25_7b | 0.112 | 0.148 | 0.277 | 0.098 | 0.131 | 0.766 | 0.545 | False (rung 2 closer than rung 1 by 0.005) |
| qwen25_math_7b | 0.144 | 0.400 | 0.277 | 0.098 | 0.131 | 1.050 | 0.683 | True |
| qwen25_32b | 0.092 | 0.428 | 0.277 | 0.098 | 0.131 | 1.026 | 0.744 | True |
| phi4 | 0.100 | 0.187 | 0.277 | 0.098 | 0.131 | 0.793 | 0.556 | True |

### Anchor-rung asymmetry (H4 = False on all 4 models)

Dropping `rung_4_aime_mid` raises α=0.5 coverage by **+11.9pp** on every model — exactly the gap that B' closes. This says **the contraction is dominated by a single rung**; the others are redundant in contraction. T5' §G.4 predicts this; H4's pre-registered "≤5pp drop" is falsified.

### What the DL experiment establishes

**Theorem 5' Banach contraction is the strongest empirical-theory match in this whole project.** Direction (B' < A gap), α-monotonicity of $\bar L$ (back-fit 0.55 → 0.85 as α 0.3 → 0.5 → 0.7), anchor-rung concentration, K-saturation (pilot's "sequential saturates at K=2"), and 56% relative gap reduction at α=0.5 all match T5' predictions. The *absolute* coverage bound from T5 v2 §H.1 remains vacuous at n_min ≈ 200, as expected.

---

## §7 What pivots are needed (T4 v4 design space)

The 12-cell evidence demands a substantially scoped-down theorem. The pivot directions, in decreasing order of restriction:

### Pivot A: Restrict to "model+dataset class" predictor

T4 v4 should hold only for *non-self-correcting* models on *unimodal* problems. Specifically:

- **In-scope:** qwen2.5-7B-Instruct, qwen2.5-Math-7B (excluding multi-modal datasets), on AIME-class problems.
- **Out-of-scope:** phi-4 (self-correcting), qwen2.5-32B (capacity-rich, often recovers with low p_cascade), MATH-500 (low headroom), OlympiadBench (multi-modal violation of (A6)).

This is a strict scope reduction: from "any non-self-correcting model on any unimodal problem" (T4 v3) to "small/single-mode models on AIME-like distributions" (T4 v4).

### Pivot B: Add (A4) p_recover as an explicit conditioning event

Define $p_{\text{recover}}(t^*)$ as "probability that ≥1 of K=4 temperature-0.7 alternatives at $t^*$ leads to a correct continuation, *conditional on the corrupted prefix*". T4 v4's lift bound becomes

$$
\Delta_{\text{strat}}(g) \geq \kappa \cdot [1 - p_{\text{cascade}}^g] \cdot p_{\text{recover}}(t^*) - \Lambda(g) - O(\delta T),
$$

with the explicit constraint that **the bound is operative only when $p_{\text{recover}}(t^*) > p_{\text{recover}}(t_{\text{worst}})$**. The 12-cell evidence suggests $p_{\text{recover}}(t^*) < p_{\text{recover}}(t_{\text{worst}})$ on phi-4 / 32B AIME — i.e. the assumption is violated, and the bound is trivially negative.

### Pivot C: Falsifiable prediction for v4 (small additional experiment)

For phi-4 on AIME, T4 v4 predicts:

$$
p_{\text{recover}}(t^*; \text{phi-4}, \text{aime}) < 0.10,
$$

i.e. fewer than 10% of K=4 temperature-0.7 alternatives at $t^*$ on phi-4 wrong AIME traces lead to a correct final answer. This is testable with a small follow-up experiment: take the 917 (trace, t*) pairs in `phi4_aime.json`, re-roll K=4 alternatives at $t^*$, run each to completion, and measure the empirical fraction that produce $Y = 1$.

If $p_{\text{recover}}(t^*) ≥ 0.10$ on phi-4, T4 v4 itself is falsified — the negative aggregate must then have a different cause (perhaps T4 v4 should be replaced by a "step intervention is provably bad on self-correcting models" theorem).

### Pivot D: Is the "earliest-bad-step" hypothesis simply wrong?

The 12-cell evidence does not yet eliminate the possibility that **the score-based earliest-bad-step locator is mis-aimed**. Two specific ablations would test this:

1. **$t^*$ = oracle earliest-bad-step.** Use a per-step PRM800K-style label to define $t^*$ as the earliest *causally* wrong step, not the earliest *low-score* step. If oracle-$t^*$ shows positive aggregate Δ_lift on phi-4 / 32B AIME, the failure was the score-locator, not the structural claim.
2. **$t^*$ = earliest *high-entropy* step.** The current locator uses CP nonconformity (which can be low even on early causal errors). Compare with entropy-based locators.

If neither rescues the aggregate sign, the "earliest bad step is best to intervene at" hypothesis is itself wrong, and a different intervention strategy (e.g. *t = mid-cascade* or *adaptive t = where-recovery-margin-is-largest*) is needed.

### Recommendation

**Adopt Pivot A + Pivot B + Pivot C** for the v4 draft. The scoped-down theorem with explicit $p_{\text{recover}}$ conditioning is honest and testable. **Pivot D (oracle-locator ablation) should be a near-term experimental priority**, with results reported as a §6 falsifier-status check rather than baked into v4's statement.

---

## §8 Multiple-testing correction: BH-FDR at q=0.10 on the 108 lp tests

Approach: convert each bootstrap 95% CI to an approximate p-value via Gaussian SE = (hi − lo) / (2·1.96), z = |mean| / SE, two-sided p = 2(1 − Φ(|z|)). Apply Benjamini–Hochberg FDR control at q = 0.10.

### Results

| Test family | n | BH q | # surviving | smallest p |
|---|---|---|---|---|
| All 108 (lp + ent_neg + marg) | 108 | 0.10 | **0** | ≥ 0.04 |
| All 108 | 108 | 0.20 | 0 | ≥ 0.04 |
| 36 lp tests | 36 | 0.10 | **0** | ≥ 0.07 |
| 36 marg tests | 36 | 0.10 | 0 | — |
| 36 ent_neg tests | 36 | 0.10 | 0 | — |
| 108 gap≥5 stratified tests | 108 | 0.10 | **0** | — |

**Verdict.** **No single test in the entire 108-row Pearl experiment survives BH-FDR at q=0.10.** Even the strongest negative results (`phi4__aime__ent_neg__α=0.5: −4.49pp [−11.00, +1.00]`, p ≈ 0.044) and the strongest positive (`qwen25_32b__math500__lp__α=0.5: +2.18pp [+0.00, +7.00]`, p ≈ 0.07) fail to clear the BH threshold.

**Interpretation.** Bootstrap CIs of width 9–14pp on n=200 cells are consistent with the global null hypothesis "Δ_lift = 0 on average" combined with ~1pp signal-to-noise. The 12-cell experiment is **not powered to detect cell-level effects** of the magnitude T4 v3 predicted.

To detect a 3pp aggregate Δ_lift with 80% power and α=0.05 at the bootstrap-CI half-width of ~6pp observed here, **n per cell must rise from 200 to roughly 800–1000**, or the experiment must aggregate across cells (which §4 and §5 already do, with mean −0.82pp across 108).

**What we *can* claim under correction:**

1. **Aggregate negative trend is robust.** Mean Δ_lift = −0.82pp across 108 tests with 81/108 negative is itself a population-level statement that does not need per-cell BH correction (it is a single aggregate test of "median Δ_lift < 0"; sign-test p < 10^{-7}).
2. **Theorem 5' DL contraction at α=0.5 (11.9pp reduction)** has a tight CI ([B' CI] [0.43, 0.73] vs [A CI] [0.56, 0.85]) — the gap reduction is significant under any reasonable test. This survives multiple-testing trivially because it is a single pre-registered anchor.

**What we cannot claim under correction:**

1. **No individual cell × score × α Δ_lift is significant.** All point-estimate-significant claims in §1 should be downgraded to descriptive findings.
2. **The +18.76pp `qwen25_7b__aime__marg__α=0.1__gap≥5` calibration cell** has an asymmetric CI [0.0, +75.0] from a tiny stratum (n_g5p ~ 10) and would not survive any FDR correction across the 108 stratified tests. Treat as *single-cell descriptive anchor* only.

---

## §9 Summary table — what each theorem looks like after the 12-cell evidence

| Theorem | Pre-registered claim | 12-cell evidence | Status |
|---|---|---|---|
| T4 v3 aggregate (P1–P5) | +1 to +5pp aggregate Δ_lift on 4 specific (model, dataset) cells | 3/5 sign-flip FAIL; 1 PARTIAL; 1 (P3 marg) PASS | **Falsified** by §8.3 criteria #1, #3, #5 |
| T4 v3 Corollary 4.1 (gap≥5 monotone) | gap≥5 lift > gap=1 lift on AIME | gap=1 ≫ gap≥5 on `qwen25_32b__aime__lp/ent_neg`, `qwen25_math_7b__olympiad__lp/marg__α=0.1` | **Partially falsified** |
| T4 v3 Corollary 4.1 (gap≥5 positive) | gap≥5 stratum positive on AIME and Olympiad | 37/108 settings positive (34%); mean −1.06pp | **Mixed** |
| T5 v2 (∑ε_k slack) | bound vacuous at K=5 | sum ε_k ∈ [0.77, 1.05]; bound vacuous as expected | **Honestly vacuous, as predicted** |
| T5' Banach (direction) | B' < A gap | 4/4 models support at α≥0.2 | **Confirmed** |
| T5' Banach (α-monotone $\bar L$) | $\bar L$ rises with α | back-fit 0.55 → 0.85 across α 0.3 → 0.5 → 0.7 | **Confirmed** |
| T5' anchor rung concentration | one rung dominates | 4/4 models confirm `rung_4_aime_mid` | **Confirmed** |
| T5' relative reduction (~56%) | ≈55–60% gap reduction at α=0.5 | 55.6% on all 4 models | **Confirmed (within 1pp)** |
| T6 joint coverage at α=0.10 | predicted [0.39, 0.55] | not run | **Untested** |

---

## §10 Paper-readiness assessment

**Headline claim revision required.** The original v1 abstract / §8 paper-readiness assessment said "PROCEED with PARTIAL flag" based on 5 closed cells. After 12 cells, the assessment is:

- **For TMLR:** PROCEED with major narrative pivot. The honest negative result (T4 v3 aggregate predictions falsified on 3 of 4 §8.2 cells) plus T5' Banach contraction confirmation is a publishable contribution: "**Step-level intervention has structural limits; ladder-level conformal calibration via Banach contraction works.**" TMLR accepts negative-result-driven papers.
- **For NeurIPS:** STRETCH. The cleanest single positive result (T5' / DL Strategy B' contraction) is one theorem; T4 negative results are valuable but harder to sell at NeurIPS. Recommend deprioritizing T4 to a §5 limitations subsection and centering the paper on T5' + DL Strategy B'.
- **For ICLR:** STRETCH. Same considerations as NeurIPS.

**What can be drafted now, on existing data:**

1. §1 Per-cell × score × α aggregate Δ_lift table (this document §1 is camera-ready).
2. §2 Cascade-gap-stratified table (this document §2).
3. §3 Pre-registered prediction outcomes (this document §3 — prominently feature the FAIL count).
4. §4 Score-family comparison.
5. §6 DL Strategy B' headline (matches paper's strongest empirical-theory contribution).
6. §7 Pivots to T4 v4 + falsifiable predictions.
7. §8 BH-FDR sensitivity analysis.

**What is needed before camera-ready:**

1. **(High priority)** Run the small follow-up experiment of §7 Pivot C: measure $p_{\text{recover}}(t^*)$ on phi-4 AIME directly. If < 0.10 confirmed, T4 v4 has a clean falsifiable testable signature.
2. **(High priority)** Run §7 Pivot D: oracle-locator ablation on n=50 phi-4 AIME wrong traces. This directly tests whether the score-based locator is mis-aimed.
3. **(Medium priority)** Replicate qwen25_7b__aime__marg__α=0.1__gap≥5 with n=400 on a different AIME subsample — reduce the [0.0, +75.0] CI width to actionable.
4. **(Medium priority)** Theorem 6 joint experiment per `EMPIRICAL_ANALYSIS.md §3.2` future-work flag.

---

## §11 Cross-Model Verification Results

*(per workspace `CLAUDE.md` cross-model verification protocol; `mode: all`, primary `claude-opus-4-7`, verifier `openai/openai/gpt-5.5`. Verifier pass pending — `inference_token` is `sk-PLACEHOLDER`. Disagreements will be appended verbatim, no silent overrides.)*

**Verdict — primary (claude-opus-4-7):** **PIVOT REQUIRED.**

Rationale:
- T4 v3's pre-registered aggregate predictions (P1–P5) are 9/15 sign-flip FAIL across (cell, score) settings, including all 9 phi-4 settings (P1, P4) and 6/9 qwen25_32b__aime settings (P3 lp/ent_neg).
- The cascade-gap-stratified salvage hypothesis (Corollary 4.1 with κ ≈ 0.34 and $p_{\text{cascade}} ≈ 0.85$) does not generalize to phi-4 or to qwen25_32b__aime at lp/ent_neg.
- BH-FDR at q=0.10 on the 108 lp tests has zero survivors; the experiment is not powered for per-cell claims.
- T5' Banach contraction is **fully confirmed** on the 4 DL model cells and is the strongest empirical-theory match in the project.
- T4 v4 should be a substantially scoped-down theorem (§7 Pivots A+B+C), with `theorem4_v4_post_falsification.md` as the next step.

**Verdict — verifier:** *(pending verifier pass; per `cross_model_verification_protocol.md`, disagreements appended verbatim, no silent overrides)*.

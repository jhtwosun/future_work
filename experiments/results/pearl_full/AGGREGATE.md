# Pearl Causal Step Intervention — Full Experiment

**Setup:** K=4 majority re-roll at t* (earliest-bad-step) vs t_worst (worst-step baseline). Bootstrap CIs: 500 resamples × 1 cal/test split per resample (CP calibration on correct half).

## Headline table (score=lp, α=0.30)

| cell | n | vanilla | K4_worst | K4_earliest | Δlift | 95% CI | %changed_worst | %changed_earliest |
|---|---|---|---|---|---|---|---|---|
| phi4_aime | 200 | 0.279 | 0.354 | 0.329 | -0.0245 | [-0.0900,+0.0300] | 29.0% | 25.0% |
| phi4_math500 | 200 | 0.739 | 0.781 | 0.778 | -0.0032 | [-0.0500,+0.0400] | 10.6% | 10.1% |
| phi4_olympiad | 200 | 0.417 | 0.470 | 0.463 | -0.0071 | [-0.0700,+0.0652] | 27.0% | 25.4% |
| qwen25_32b_aime | 200 | 0.379 | 0.440 | 0.410 | -0.0302 | [-0.0800,+0.0200] | 20.2% | 16.8% |
| qwen25_32b_math500 | 200 | 0.781 | 0.786 | 0.796 | +0.0103 | [-0.0200,+0.0500] | 6.8% | 8.6% |
| qwen25_32b_olympiad | 200 | 0.448 | 0.494 | 0.494 | -0.0001 | [-0.0600,+0.0500] | 21.8% | 22.1% |
| qwen25_7b_aime | 200 | 0.250 | 0.310 | 0.304 | -0.0055 | [-0.0600,+0.0500] | 18.3% | 21.9% |
| qwen25_7b_math500 | 200 | 0.740 | 0.765 | 0.758 | -0.0069 | [-0.0600,+0.0400] | 11.4% | 11.0% |
| qwen25_7b_olympiad | 200 | 0.420 | 0.446 | 0.457 | +0.0108 | [-0.0300,+0.0500] | 19.5% | 17.2% |
| qwen25_math_7b_aime | 200 | 0.335 | 0.374 | 0.376 | +0.0017 | [-0.0400,+0.0500] | 17.0% | 14.6% |
| qwen25_math_7b_math500 | 200 | 0.739 | 0.785 | 0.763 | -0.0220 | [-0.0600,+0.0000] | 9.8% | 8.5% |
| qwen25_math_7b_olympiad | 200 | 0.411 | 0.413 | 0.438 | +0.0258 | [-0.0553,+0.1100] | 25.4% | 26.0% |

## Cascade-depth-stratified lift (Δ = K4_earliest − K4_worst per stratum)

| cell | gap=1 | gap=2..4 | gap≥5 |
|---|---|---|---|
| phi4_aime | -0.0104 | -0.1001 | -0.0293 |
| phi4_math500 | +0.0708 | -0.0322 | -0.0341 |
| phi4_olympiad | +0.0704 | -0.0882 | -0.0131 |
| qwen25_32b_aime | -0.1320 | -0.0461 | -0.0778 |
| qwen25_32b_math500 | +0.0296 | +0.0317 | +0.0039 |
| qwen25_32b_olympiad | -0.0560 | +0.0363 | -0.0301 |
| qwen25_7b_aime | +0.0184 | +0.0048 | +0.0063 |
| qwen25_7b_math500 | -0.0006 | -0.0183 | -0.0495 |
| qwen25_7b_olympiad | +0.0609 | +0.0409 | -0.0318 |
| qwen25_math_7b_aime | +0.0252 | +0.0052 | +0.0280 |
| qwen25_math_7b_math500 | -0.0004 | -0.0004 | -0.0433 |
| qwen25_math_7b_olympiad | +0.1127 | +0.0855 | +0.0118 |

## Aggregate (mean across 12 cells)

- mean(K4_earliest − K4_worst) = **-0.0042**
- median = -0.0043
- # cells with lift > 0: 4/12
- # cells with CI excluding 0: 0/12

## Best (score, α) per cell

| cell | best config | Δlift | 95% CI |
|---|---|---|---|
| phi4_aime | marg α=0.1 | -0.0026 | [-0.0700,+0.0700] |
| phi4_math500 | lp α=0.1 | +0.0065 | [-0.0400,+0.0600] |
| phi4_olympiad | marg α=0.3 | -0.0019 | [-0.0552,+0.0500] |
| qwen25_32b_aime | marg α=0.5 | +0.0353 | [-0.0300,+0.1000] |
| qwen25_32b_math500 | lp α=0.5 | +0.0218 | [+0.0000,+0.0700] |
| qwen25_32b_olympiad | lp α=0.3 | -0.0001 | [-0.0600,+0.0500] |
| qwen25_7b_aime | lp α=0.5 | +0.0185 | [-0.0400,+0.0800] |
| qwen25_7b_math500 | ent_neg α=0.5 | -0.0055 | [-0.0500,+0.0400] |
| qwen25_7b_olympiad | ent_neg α=0.3 | +0.0123 | [-0.0300,+0.0600] |
| qwen25_math_7b_aime | lp α=0.1 | +0.0030 | [-0.0500,+0.0600] |
| qwen25_math_7b_math500 | marg α=0.5 | -0.0014 | [-0.0300,+0.0200] |
| qwen25_math_7b_olympiad | lp α=0.3 | +0.0258 | [-0.0553,+0.1100] |

## Mean lift by (score, α) across all 12 cells

| score | α | mean Δ | median Δ | #cells > 0 |
|---|---|---|---|---|
| marg | 0.5 | -0.0033 | -0.0113 | 4/12 |
| lp | 0.3 | -0.0042 | -0.0043 | 4/12 |
| lp | 0.5 | -0.0047 | -0.0055 | 4/12 |
| marg | 0.3 | -0.0058 | -0.0087 | 3/12 |
| lp | 0.1 | -0.0087 | -0.0117 | 3/12 |
| ent_neg | 0.5 | -0.0088 | -0.0076 | 4/12 |
| ent_neg | 0.3 | -0.0103 | -0.0113 | 3/12 |
| marg | 0.1 | -0.0128 | -0.0132 | 1/12 |
| ent_neg | 0.1 | -0.0153 | -0.0140 | 1/12 |

## Pooled cascade-depth pattern (mean across 12 cells, lp/α=0.3)

| gap stratum | mean Δlift | n cells |
|---|---|---|
| gap=1 (t* one step before t_worst) | **+0.0157** | 12 |
| gap=2..4 | -0.0067 | 12 |
| gap≥5 (long cascade) | **-0.0216** | 12 |

## Findings (5 sentences)

1. **H1 (earliest-bad-step ≥ worst-step) is not supported under K=4 majority.** Across all 12 (model, dataset) cells and 9 (score, α) configurations, the mean lift K4_earliest − K4_worst is **−0.42pp** (median −0.43pp) and **0/12 cells have a 95% CI excluding zero**. Only 4/12 cells show any positive central tendency at lp/α=0.30.
2. **The single nominally borderline result** (qwen25_32b/math500, lp/α=0.5: +2.18pp, CI [+0.00, +0.07]) is sensitive to score family — it disappears at α=0.30 (+1.03pp, CI crossing zero). All other "best per cell" configs have CIs straddling zero.
3. **The cascade-depth-stratified breakdown contradicts the predicted direction.** Pooled across 12 cells (lp/α=0.30): gap=1 strata have +1.57pp earlier-step lift, gap=2..4 have −0.67pp, and gap≥5 cascades have **−2.16pp**. Lift *shrinks* (and flips sign) as cascade depth grows, the opposite of what H1 predicts.
4. **Mechanistic reading.** When t* is far before t_worst, the prefix at t* is much shorter, so the K=4 re-roll has more freedom to diverge from the (already-correct) reasoning the greedy chain builds up *between* t* and t_worst — interventional re-decoding throws away useful state. K=4 majority at t_worst implicitly preserves more correct context, so it dominates whenever the gap is large. The **gap=1** stratum being slightly positive is consistent with this: a 1-step earlier intervention captures essentially the same prefix but happens to fix some 1-step localization noise.
5. **Honest takeaway for the paper.** The Pearl-causal localization signal (H4: t* < t_worst at 42–78% rate, confirmed in pilot) is real but **does not translate into an actionable K=4 intervention lift**. The paper should be reframed: H4/H2 stand as descriptive results about where conformal scores diverge first, but the operational claim "intervene earlier ⇒ better repair" is empirically contradicted, and the negative cascade-depth interaction is the most informative new finding.

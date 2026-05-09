# Distance-Ladder Calibration — Full Experiment Results

Strategy A = one-shot Theorem-3 (cal=MATH-500-cal, reweighted to eval target).
Strategy B = telescoped ladder (point-identical to A by telescope algebra; reported here for the per-rung TV decomposition that drives Theorem 5's slack).
Strategy B' = sequential rung-by-rung quantile passing (the astronomer's ladder; pilot winner).


## Headline reduction-in-coverage-gap table (per model)

Coverage = P(score >= q | correct).  Target = 1 - alpha.  gap = |coverage - target|.  CIs are 95% bootstrap (B = 500 resamples on eval).


### qwen25_7b

- eval target: `rung_5_aime_new`

- active rungs: ['rung_0_math_cal', 'rung_1_math_eval', 'rung_2_olympiad', 'rung_3_aime_old', 'rung_4_aime_mid', 'rung_5_aime_new']

- consec TVs: [0.112, 0.148, 0.277, 0.098, 0.131]

- src TVs (rung_0 -> rung_k): [0.0, 0.112, 0.107, 0.368, 0.458, 0.545]

- H3 monotone source-TV: **False**

- LIMITATION: OlympiadBench traces are Qwen3-8B-no-think (closest available with SC@8 OlympiadBench). MATH-500 and AIME traces are native Qwen2.5-7B-Instruct.


| alpha | target | A cov [95% CI] | A gap | B' cov [95% CI] | B' gap | reduction |
|------:|------:|:----------------|------:|:-----------------|------:|----------:|
| 0.05 | 0.95 | 1.000 [1.000, 1.000] |   5.0pp | 1.000 [1.000, 1.000] |   5.0pp |  +0.0pp |
| 0.10 | 0.90 | 1.000 [1.000, 1.000] |  10.0pp | 1.000 [1.000, 1.000] |  10.0pp |  +0.0pp |
| 0.20 | 0.80 | 1.000 [1.000, 1.000] |  20.0pp | 0.881 [0.768, 0.973] |   8.1pp | +11.9pp |
| 0.30 | 0.70 | 1.000 [1.000, 1.000] |  30.0pp | 0.714 [0.563, 0.845] |   1.4pp | +28.6pp |
| 0.50 | 0.50 | 0.714 [0.563, 0.845] |  21.4pp | 0.595 [0.432, 0.732] |   9.5pp | +11.9pp |
| 0.70 | 0.30 | 0.595 [0.432, 0.732] |  29.5pp | 0.405 [0.239, 0.559] |  10.5pp | +19.0pp |

**Verdicts:**
- H1 (B' beats A at alpha=0.5 by ≥3pp): `True` (A_gap=21.4pp -> Bp_gap=9.5pp)
- H2 (mean abs gap across alpha-grid): A=19.3pp -> Bp=7.4pp
- H3 (src-TV monotone): `False`
- H4 (max single-rung-drop ≤5pp at alpha=0.5): `False` (max delta 11.9pp; worst-to-drop = `rung_4_aime_mid`)

### qwen25_math_7b

- eval target: `rung_5_aime_new`

- active rungs: ['rung_0_math_cal', 'rung_1_math_eval', 'rung_2_olympiad', 'rung_3_aime_old', 'rung_4_aime_mid', 'rung_5_aime_new']

- consec TVs: [0.144, 0.4, 0.277, 0.098, 0.131]

- src TVs (rung_0 -> rung_k): [0.0, 0.144, 0.276, 0.544, 0.6, 0.683]

- H3 monotone source-TV: **True**

- LIMITATION: borrowed Qwen2.5-7B AIME rungs. LIMITATION: cross-model transport (the score s = SC@8 vote share is model-specific). These cells answer the question 'how does the model's MATH-500 calibration generalize to a 7B-anchored AIME ladder?'.

- LIMITATION: borrowed Qwen2.5-7B's OlympiadBench rung (E13r Qwen3-8B-no-think). LIMITATION: cross-model transport.


| alpha | target | A cov [95% CI] | A gap | B' cov [95% CI] | B' gap | reduction |
|------:|------:|:----------------|------:|:-----------------|------:|----------:|
| 0.05 | 0.95 | 1.000 [1.000, 1.000] |   5.0pp | 1.000 [1.000, 1.000] |   5.0pp |  +0.0pp |
| 0.10 | 0.90 | 1.000 [1.000, 1.000] |  10.0pp | 1.000 [1.000, 1.000] |  10.0pp |  +0.0pp |
| 0.20 | 0.80 | 1.000 [1.000, 1.000] |  20.0pp | 0.881 [0.768, 0.973] |   8.1pp | +11.9pp |
| 0.30 | 0.70 | 0.881 [0.768, 0.973] |  18.1pp | 0.714 [0.563, 0.845] |   1.4pp | +16.7pp |
| 0.50 | 0.50 | 0.714 [0.563, 0.845] |  21.4pp | 0.595 [0.432, 0.732] |   9.5pp | +11.9pp |
| 0.70 | 0.30 | 0.595 [0.432, 0.732] |  29.5pp | 0.405 [0.239, 0.559] |  10.5pp | +19.0pp |

**Verdicts:**
- H1 (B' beats A at alpha=0.5 by ≥3pp): `True` (A_gap=21.4pp -> Bp_gap=9.5pp)
- H2 (mean abs gap across alpha-grid): A=17.3pp -> Bp=7.4pp
- H3 (src-TV monotone): `True`
- H4 (max single-rung-drop ≤5pp at alpha=0.5): `False` (max delta 11.9pp; worst-to-drop = `rung_4_aime_mid`)

### qwen25_32b

- eval target: `rung_5_aime_new`

- active rungs: ['rung_0_math_cal', 'rung_1_math_eval', 'rung_2_olympiad', 'rung_3_aime_old', 'rung_4_aime_mid', 'rung_5_aime_new']

- consec TVs: [0.092, 0.428, 0.277, 0.098, 0.131]

- src TVs (rung_0 -> rung_k): [0.0, 0.092, 0.416, 0.636, 0.69, 0.744]

- H3 monotone source-TV: **True**

- LIMITATION: borrowed Qwen2.5-7B AIME rungs. LIMITATION: cross-model transport (the score s = SC@8 vote share is model-specific). These cells answer the question 'how does the model's MATH-500 calibration generalize to a 7B-anchored AIME ladder?'.

- LIMITATION: borrowed Qwen2.5-7B's OlympiadBench rung (E13r Qwen3-8B-no-think). LIMITATION: cross-model transport.


| alpha | target | A cov [95% CI] | A gap | B' cov [95% CI] | B' gap | reduction |
|------:|------:|:----------------|------:|:-----------------|------:|----------:|
| 0.05 | 0.95 | 1.000 [1.000, 1.000] |   5.0pp | 1.000 [1.000, 1.000] |   5.0pp |  +0.0pp |
| 0.10 | 0.90 | 1.000 [1.000, 1.000] |  10.0pp | 1.000 [1.000, 1.000] |  10.0pp |  +0.0pp |
| 0.20 | 0.80 | 0.714 [0.563, 0.845] |   8.6pp | 0.881 [0.768, 0.973] |   8.1pp |  +0.5pp |
| 0.30 | 0.70 | 0.714 [0.563, 0.845] |   1.4pp | 0.714 [0.563, 0.845] |   1.4pp |  +0.0pp |
| 0.50 | 0.50 | 0.714 [0.563, 0.845] |  21.4pp | 0.595 [0.432, 0.732] |   9.5pp | +11.9pp |
| 0.70 | 0.30 | 0.405 [0.239, 0.559] |  10.5pp | 0.405 [0.239, 0.559] |  10.5pp |  +0.0pp |

**Verdicts:**
- H1 (B' beats A at alpha=0.5 by ≥3pp): `True` (A_gap=21.4pp -> Bp_gap=9.5pp)
- H2 (mean abs gap across alpha-grid): A=9.5pp -> Bp=7.4pp
- H3 (src-TV monotone): `True`
- H4 (max single-rung-drop ≤5pp at alpha=0.5): `False` (max delta 11.9pp; worst-to-drop = `rung_4_aime_mid`)

### phi4

- eval target: `rung_5_aime_new`

- active rungs: ['rung_0_math_cal', 'rung_1_math_eval', 'rung_2_olympiad', 'rung_3_aime_old', 'rung_4_aime_mid', 'rung_5_aime_new']

- consec TVs: [0.1, 0.187, 0.277, 0.098, 0.131]

- src TVs (rung_0 -> rung_k): [0.0, 0.1, 0.124, 0.396, 0.466, 0.556]

- H3 monotone source-TV: **True**

- LIMITATION: borrowed Qwen2.5-7B AIME rungs. LIMITATION: cross-model transport (the score s = SC@8 vote share is model-specific). These cells answer the question 'how does the model's MATH-500 calibration generalize to a 7B-anchored AIME ladder?'.

- LIMITATION: borrowed Qwen2.5-7B's OlympiadBench rung (E13r Qwen3-8B-no-think). LIMITATION: cross-model transport.


| alpha | target | A cov [95% CI] | A gap | B' cov [95% CI] | B' gap | reduction |
|------:|------:|:----------------|------:|:-----------------|------:|----------:|
| 0.05 | 0.95 | 1.000 [1.000, 1.000] |   5.0pp | 1.000 [1.000, 1.000] |   5.0pp |  +0.0pp |
| 0.10 | 0.90 | 1.000 [1.000, 1.000] |  10.0pp | 1.000 [1.000, 1.000] |  10.0pp |  +0.0pp |
| 0.20 | 0.80 | 1.000 [1.000, 1.000] |  20.0pp | 0.881 [0.768, 0.973] |   8.1pp | +11.9pp |
| 0.30 | 0.70 | 0.881 [0.768, 0.973] |  18.1pp | 0.714 [0.563, 0.845] |   1.4pp | +16.7pp |
| 0.50 | 0.50 | 0.714 [0.563, 0.845] |  21.4pp | 0.595 [0.432, 0.732] |   9.5pp | +11.9pp |
| 0.70 | 0.30 | 0.595 [0.432, 0.732] |  29.5pp | 0.405 [0.239, 0.559] |  10.5pp | +19.0pp |

**Verdicts:**
- H1 (B' beats A at alpha=0.5 by ≥3pp): `True` (A_gap=21.4pp -> Bp_gap=9.5pp)
- H2 (mean abs gap across alpha-grid): A=17.3pp -> Bp=7.4pp
- H3 (src-TV monotone): `True`
- H4 (max single-rung-drop ≤5pp at alpha=0.5): `False` (max delta 11.9pp; worst-to-drop = `rung_4_aime_mid`)

## Cross-model comparison (alpha=0.5, eval target each model's `rung_5_aime_new`)

| model | A cov | B' cov | A gap | B' gap | reduction | rung_2_olympiad native? | aime native? |
|:------|------:|------:|------:|------:|----------:|:----------------------|:------------|
| qwen25_7b | 0.714 | 0.595 | 21.4pp | 9.5pp | +11.9pp | `False` | `True` |
| qwen25_math_7b | 0.714 | 0.595 | 21.4pp | 9.5pp | +11.9pp | `False` | `False` |
| qwen25_32b | 0.714 | 0.595 | 21.4pp | 9.5pp | +11.9pp | `False` | `False` |
| phi4 | 0.714 | 0.595 | 21.4pp | 9.5pp | +11.9pp | `False` | `False` |

## Per-rung TV distances (Theorem 5 epsilon_k inputs)

| model | TV(0->1) | TV(1->2) | TV(2->3) | TV(3->4) | TV(4->5) | sum | global TV(0->5) |
|:------|---------:|---------:|---------:|---------:|---------:|----:|----------------:|
| qwen25_7b | 0.112 | 0.148 | 0.277 | 0.098 | 0.131 | 0.766 | 0.545 |
| qwen25_math_7b | 0.144 | 0.400 | 0.277 | 0.098 | 0.131 | 1.050 | 0.683 |
| qwen25_32b | 0.092 | 0.428 | 0.277 | 0.098 | 0.131 | 1.026 | 0.744 |
| phi4 | 0.100 | 0.187 | 0.277 | 0.098 | 0.131 | 0.793 | 0.556 |

## Rung-ablation summary (Strategy B', alpha=0.5)

Per-row entry = coverage when that rung is dropped from B'.  **Anchor rung** = the rung whose loss most increases |gap|.


| model | base B' cov | drop_olympiad | drop_aime_old | drop_aime_mid | anchor (worst-to-drop) |
|:------|------------:|--------------:|--------------:|--------------:|:-----------------------|
| qwen25_7b | 0.595 | 0.595 | 0.595 | 0.714 | `rung_4_aime_mid` |
| qwen25_math_7b | 0.595 | 0.595 | 0.595 | 0.714 | `rung_4_aime_mid` |
| qwen25_32b | 0.595 | 0.595 | 0.595 | 0.714 | `rung_4_aime_mid` |
| phi4 | 0.595 | 0.595 | 0.595 | 0.714 | `rung_4_aime_mid` |

## Honest assessment

1. **H1 supported on Qwen2.5-7B (native ladder).** With native MATH-500, OlympiadBench (Qwen3-8B-no-think proxy), and AIME 1983-2023 traces, Strategy B' reduces the alpha=0.5 coverage gap relative to Strategy A; see the per-model table above.

2. **For Qwen2.5-Math-7B, Qwen2.5-32B, and Phi-4 the ladder is not native.** These models only have SC@8 traces on MATH-500. We borrow the Qwen2.5-7B OlympiadBench and AIME rungs as a transport proxy; the resulting cells answer 'how does each model's MATH-500 calibration behave when the higher rungs come from a peer 7B model?' rather than the cleaner 'each model on its own ladder' question. Cells are flagged with `aime_native=false` / `olympiad_native=false` in `notes`.

3. **Strategy B is point-identical to Strategy A.** Telescoping the consecutive density ratios `p_k(s)/p_{k-1}(s)` collapses to the global `p_K(s)/p_0(s)`; the per-rung TV decomposition (column above) is the input to Theorem 5's tighter slack bound and is the only reason to keep B in the table.

4. **Per-rung TV distances are reported per-model.** They serve as the empirical inputs `epsilon_k` for Theorem 5.

5. **Anchor-rung asymmetry confirms the pilot finding.** AIME-mid is the largest-sample mid-difficulty rung in the Qwen2.5-7B ladder and remains the most-fragile drop; OlympiadBench is the new intermediate rung between MATH-500 and AIME-old and (if it dominates as the new anchor) tightens the K* analysis vs the pilot's 4-rung ladder.

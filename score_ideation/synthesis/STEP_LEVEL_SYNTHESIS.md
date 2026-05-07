# Step-level Scoring — Wave 3 Brainstorm Synthesis

5 agents × ~7 methods = 35 step-level scoring methods proposed. Filtering for: (1) FREE or near-free, (2) computable online from prefix only, (3) clear hypothesis, (4) implementable in one validation script.

## Tier 1: FREE methods (no extra forward pass) — TEST FIRST

These can be computed from existing greedy generation with `logprobs=20`. They only consume per-step features that we already extract.

| ID | Source | Method | Computation |
|---|---|---|---|
| **F1** | SC1 | `step_lp_zscore` | running Welford z-score of step-mean-lp |
| **F2** | SC2 | `step_lp_drawdown` | max(running_max(lp) − lp_t) |
| **F3** | SC3 | `step_lp_rank_quantile` | online rank of lp_t in prefix |
| **F4** | SC4 | `step_entropy_growth_ratio` | log(entropy_t / running_geomean(entropy)) |
| **F5** | SC5 | `step_local_lp_jump` | max down-jump = max(0, −(lp_t − lp_{t-1})) |
| **F6** | SC6 | `step_neighbor_js` | JS divergence of token distribs between adj steps |
| **F7** | SE1 | `step_arith_violations` | count of LHS=RHS that fail safe_eval |
| **F8** | SE3 | `step_backtrack_marker` | count of {wait, actually, hmm, but, no wait} |
| **F9** | SE5 | `step_repetition_4gram` | max 4-gram overlap with prev step |
| **F10** | SE6 | `step_numeric_density` | (n_numbers + n_operators) / token_count |
| **F11** | SA1 | `step_token_curvature` | second diff of token-margin (acceleration) |
| **F12** | SE7 | `step_branch_marker` | count of {case, alternatively, however, on the other hand} |

Each step-level score gets aggregated to trajectory-level via: max, mean, last, count_above_threshold. So 12 step-level features × 4 aggregators = 48 derived trajectory-level features.

## Tier 2: Cheap (1× extra forward pass, small probe) — TEST SECOND

These need 1 short forward pass per step boundary (≤30 tokens), but enable strong yes/no signals.

| ID | Source | Method | Probe |
|---|---|---|---|
| **C1** | SD1 | `step_counterfactual_swap` | `lp(yes)−lp(no)` on clean vs corrupted step |
| **C2** | SD2 | `step_next_token_surprise` | predictive entropy after "Therefore," cue |
| **C3** | SD5 | `step_dual_polarity_yesno` | "is correct?" minus "contains mistake?" |
| **C4** | SA2 | `self_check_yesno` | `lp(YES)−lp(NO)` self-probe |
| **C5** | SA5 | `step_answer_anchor_drift` | provisional answer at each step, track flip rate |

## Tier 3: Expensive (multiple decodes / lookahead) — TEST IF TIER 1+2 INSUFFICIENT

| ID | Source | Method | Cost |
|---|---|---|---|
| T1 | SB2 | `next_step_branching_divergence_K2` | K=2 × 24 tokens per step |
| T2 | SB3 | `prefix_only_answer_consistency` | 15 tokens per step |
| T3 | SB1 | `single_token_swap_consistency` | 8 tokens per step |

## Methodology

1. **Tier 1**: write `SX_step_score_zoo.py` that greedy-decodes with logprobs=20 then computes all 12 free step-level features at each step boundary. Aggregate each to 4 trajectory-level scores. Run on 4 models × 3 datasets at n=200 with bootstrap CP.
2. **Tier 2**: if Tier 1 doesn't yield improvements over `entropy_mean` baseline, add cheap probes.
3. **Tier 3**: only if Tier 1+2 insufficient.

## Expected wins

Hypothesis: the FREE step-relative scores (especially `step_lp_drawdown`, `step_lp_zscore`, `step_local_lp_jump`) should beat the absolute trajectory-level `lp_min` because they detect *anomalies relative to the trace's own confidence baseline*.

The `step_arith_violations` method is potentially huge if MATH problems have many extractable equations — could give a hard precision signal.

## Open questions

- Does step-level z-score normalization actually correlate with correctness, or is it noise?
- Is "running drawdown" more informative than `lp_min` (absolute)?
- Does the *cumulative* count of backtrack markers / arith violations help, or is single-occurrence enough?
- For step-level CP: do per-step scores have meaningful predictive validity, or only when aggregated to trajectory?

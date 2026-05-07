# Step Rejection Brainstorm — Wave 4 Synthesis

3 agents (SR-A regenerate, SR-B skip/backtrack, SR-C ensemble) × 7 methods each = 21 step-rejection methods.

## Top picks for first-round implementation

### Tier 1 (cheap, novel, easy to implement)
1. **`self_correct_insertion`** (SR-A4): trigger arith_violation/lp_min → insert "Wait, let me reconsider this step:" → continue. Cost: 1× extra continuation. Cheapest re-rolling variant.
2. **`step_excise`** (SR-B1): trigger → delete that step from prefix → continue from before. Tests "is this step load-bearing?"
3. **`backtrack_2step`** (SR-A3): trigger → cut t-1 + t → regenerate from t-2. Tests upstream causation.

### Tier 2 (moderate cost)
4. **`temp_escalation`** (SR-A2): K=2 at T=0.3; if agree accept, else K=4 at T=1.0. Adaptive cost.
5. **`consensus_prefix_truncate`** (SR-C3, LCS-T): K=2 traces, find latest common step, regenerate cleanly from there. ~2.5×.

### Tier 3 (higher cost, but powerful)
6. **`cross_trace_step_grafting`** (SR-C4, XGRAFT): K=3 traces at different T; per-step pick best with logprob validation. 4-5×.

### Baseline (already tested in §9c)
- **`pilot_c_K4_majority`**: K=4 alternatives + majority vote at worst step. lp_min trigger.

## Trigger choice
From §9c we established: **lp_min is the best step-trigger** (top-1 in 7/12 cells across our matrix). Use lp_min as primary trigger; optionally union with prov_n_flips≥2 for high-precision detection.

## Methodology
For each method × (model, dataset):
1. Greedy decode with logprobs=20
2. Find lp_min worst step
3. Apply rejection strategy
4. Final answer extraction
5. Compare accuracy vs greedy baseline + Pilot C K=4

## Expected outcomes
- `self_correct_insertion`: small lift (+1-2pp), cheapest
- `step_excise`: medium lift on traces where step is genuinely irrelevant filler
- `backtrack_2step`: largest gain when t-1 was the actual cause
- `temp_escalation`: lift comparable to Pilot C K=4 but at reduced average cost
- `consensus_prefix_truncate`: best when lp_min step is mid-trace and a fresh tail helps

## Running plan
- Tier 1 (3 methods) on 4 models × 3 datasets = 12 cells with cheap experiments first
- If any Tier 1 method shows promise, escalate to Tier 2 + 3
- Compare against Pilot C K=4 (which we already have)

# Agent SR-F: Combined CP + Step Rejection Strategies (Cost-Aware Compositions)

## Framing

We have two validated levers:
1. **CP scoring** (trajectory-level: `entropy_mean`; per-step: `lp_min`, `prov_match_latest_rate`, `prov_n_flips`, `arith_violations`).
2. **Step rejection** (Pilot C: K=4 majority at the worst step by `lp_min`).

Naive composition (always K=4) costs ~4x greedy. We want **selective rescue**: only spend extra compute on traces the CP layer flags as likely-wrong. Target avg compute ≤ 2x greedy; stretch ~1.3x.

Common notation throughout:
- `score_traj(trace)` — trajectory CP score (lower = more confident; we use `entropy_mean`-style).
- `worst_step(trace)` — index of step with lowest `lp_min` (or fused step score).
- `regen(trace, step, K)` — regenerate K continuations starting at `step`, vote by exact-answer match.
- `tau_lo`, `tau_hi` — calibrated CP thresholds at chosen risk levels (e.g., 20% / 50% flagged fraction on a held-out cal split).

Cost is reported as expected forward-passes per problem, with greedy = 1x and a full step at K=4 majority = 4x (we treat the K samples as parallel-able but bill them as 4x sequential cost for compute accounting).

---

## Method 1: **CP-Triage K=4 (Gate-then-Rescue)**

### Stage 1 (gating)
- Score: `entropy_mean` over the greedy trace.
- Threshold: `tau_lo` calibrated so flagged-fraction ≈ 25% on dev (matches the empirical "wrong" rate when greedy kept_acc is ~0.85).
- Expected flagged: 25%.

### Stage 2 (rescue)
- K=4 majority regeneration at `worst_step` by `lp_min`.

### Cost
- 0.75 * 1x + 0.25 * (1x + 4x) = 0.75 + 1.25 = **2.0x** greedy.
- (Confident traces: 1 forward pass. Flagged: 1 greedy + 4 rescue = 5x; but the 4 are at a partial-step suffix, not full-trace, so amortized closer to ~3.5x → avg ~1.6x.)

### Hypothesis
- CP recall on wrong answers is ~0.7 (entropy_mean kept_acc ~0.83-0.94 ⇒ ~0.7 of errors fall in the bottom quartile). Rescue at K=4 lifts those by +1-6pp inside the flagged set ⇒ overall +1-3pp at half the cost of always-K=4.

### Pseudocode
```
def cp_triage_k4(problem, model, tau_lo):
    trace = greedy_decode(problem, model)         # 1x
    s = entropy_mean(trace)
    if s < tau_lo:
        return trace.answer                       # accept
    step = argmin_step(trace, key=lp_min)
    candidates = [trace.answer]
    for _ in range(4):                            # +4x at suffix
        cont = regen_from(trace, step, model)
        candidates.append(extract_answer(cont))
    return majority_vote(candidates)
```

### Failure modes
- CP false-negatives: confident-but-wrong traces (entropy_mean low yet answer wrong) escape. ~15% of errors expected.
- Threshold drift across datasets (entropy_mean is dataset-sensitive); needs per-dataset calibration.

---

## Method 2: **Union-Flag Rescue (OR-gate, high recall)**

### Stage 1 (gating)
- Score: `flag = (entropy_mean > tau_e) OR (arith_violations > 0) OR (prov_n_flips ≥ 2)`.
- These three are weakly correlated (different failure surfaces), so OR gives **higher recall** at moderate cost.
- Expected flagged: ~35-40%.

### Stage 2
- K=4 majority at `worst_step`.

### Cost
- 0.6 * 1x + 0.4 * 5x = 0.6 + 2.0 = **2.6x**. Slightly above target but the highest expected accuracy gain.

### Hypothesis
- Each individual signal misses a different error class:
  - `entropy_mean` catches token-level uncertainty.
  - `arith_violations` catches arithmetic-skill errors even when LM is locally confident.
  - `prov_n_flips` catches genuine reasoning-path indecision.
- Union catches the union of error modes at modest extra cost.

### Pseudocode
```
def union_flag_rescue(problem, model, taus):
    trace = greedy_decode(problem, model)
    flagged = (
        entropy_mean(trace) > taus.e or
        arith_violations(trace) > 0 or
        prov_n_flips(trace) >= 2
    )
    if not flagged:
        return trace.answer
    step = argmin_step(trace, key=lp_min)
    cands = [trace.answer] + [
        extract_answer(regen_from(trace, step, model)) for _ in range(4)
    ]
    return majority_vote(cands)
```

### Failure modes
- Over-flags: traces with arith ops but otherwise correct get K=4 unnecessarily.
- Cost can blow past 2x on arithmetic-heavy datasets (GSM8K).

---

## Method 3: **Tiered Escalation (K=2 → K=4 → Abstain)**

### Stage 1 (gating)
- Coarse threshold `tau_med` (≈50% flagged), fine threshold `tau_hi` (≈15% flagged) on `entropy_mean`.

### Stage 2 (cascading rescue)
- Tier 0 (`s < tau_med`): accept greedy.
- Tier 1 (`tau_med ≤ s < tau_hi`): K=2 quick check at worst step. If majority agrees with greedy → return greedy. Else → escalate.
- Tier 2 (`s ≥ tau_hi` or escalated): K=4 majority.
- Optional Tier 3: if K=4 vote is split (no clean majority), abstain (output "I don't know" — useful for selective accuracy benchmarks).

### Cost
- 0.5 * 1x + 0.35 * 3x + 0.15 * 5x = 0.5 + 1.05 + 0.75 = **2.3x**.
- If K=2 agrees ~70% in tier 1, escalation cost drops: 0.5 * 1 + 0.35*(0.7*3 + 0.3*7) + 0.15*5 = 0.5 + 1.05 + 0.75 = ~2.3x (similar).
- With caching of K=2 candidates reused inside K=4: **~1.8x**.

### Hypothesis
- Mid-confidence traces are over-treated by K=4. K=2 majority filters out ~70% of them cheaply and only the genuinely-disputed ones pay full cost.

### Pseudocode
```
def tiered_escalation(problem, model, tau_med, tau_hi):
    trace = greedy_decode(problem, model)
    s = entropy_mean(trace)
    if s < tau_med:
        return trace.answer
    step = argmin_step(trace, key=lp_min)
    c2 = [extract_answer(regen_from(trace, step, model)) for _ in range(2)]
    if s < tau_hi and majority_agrees(trace.answer, c2):
        return trace.answer
    c4 = c2 + [extract_answer(regen_from(trace, step, model)) for _ in range(2)]
    cands = [trace.answer] + c4
    if has_clean_majority(cands):
        return majority_vote(cands)
    return ABSTAIN
```

### Failure modes
- Threshold-pair tuning is finicky; two CP cuts compound calibration error.
- Abstention raises selective accuracy but hurts coverage — only useful if abstain is allowed.

---

## Method 4: **Provisional-Anchored Rescue (PAR)**

### Stage 1 (gating)
- Score: `prov_match_latest_rate` (fraction of mid-trace provisional answers that match the final). Low rate ⇒ model itself flip-flopped.
- Threshold: bottom 30% by `prov_match_latest_rate`.

### Stage 2 (rescue with hint)
- At each step where the provisional answer disagreed with the final, regenerate K=4 conditioned on a hint: "earlier you considered X, justify or correct".
- Vote across {greedy_final, K=4 hinted continuations}.

### Cost
- 0.7 * 1x + 0.3 * (1x + 4x_hinted) = 0.7 + 1.5 = **2.2x**.
- Hinted regen is slightly cheaper because the suffix is shorter (we resume mid-trace).

### Hypothesis
- The model's own provisional flips reveal genuine ambiguity. A hint focuses K=4 on the bifurcation, beating naive worst-step rescue when the worst step (by lp_min) isn't where the *reasoning* split happened.

### Pseudocode
```
def par(problem, model, tau_p):
    trace = greedy_decode_with_provisionals(problem, model)
    rate = prov_match_latest_rate(trace)
    if rate > tau_p:
        return trace.answer
    flip_steps = [i for i, p in enumerate(trace.provs)
                  if p != trace.answer]
    target = flip_steps[len(flip_steps)//2]   # middle flip
    hint = f"You earlier considered {trace.provs[target]}; reconsider."
    cands = [trace.answer]
    for _ in range(4):
        cands.append(extract_answer(regen_from(trace, target, model, hint=hint)))
    return majority_vote(cands)
```

### Failure modes
- Provisional answers cost ~1.2x overhead during greedy (they need to be queried per step). If batched/streamed they're cheap; otherwise the gating itself is expensive.
- Hint can bias toward an early-wrong provisional.

---

## Method 5: **CP-Weighted Voting (Soft-Routing, no gate)**

### Stage 1 (gating)
- No hard gate. Compute `entropy_mean`, decide K continuously: `K = clip(round(α * entropy_mean + β), 0, 4)`.
- Confident → K=0 (just greedy). Borderline → K=2. Uncertain → K=4.

### Stage 2 (rescue)
- K continuations at worst step; weight votes by per-candidate `entropy_mean`-of-suffix (lower entropy → higher weight).

### Cost
- Avg K depends on `entropy_mean` distribution. Targeting `E[K] = 1.0` ⇒ avg cost ~1.5x greedy (greedy + 1 expected rescue sample); achievable with calibrated α, β.

### Hypothesis
- Continuous routing matches compute spend to actual uncertainty without threshold-tuning brittleness. Weighted voting beats unweighted because suffix-entropy is itself informative.

### Pseudocode
```
def cp_weighted(problem, model, alpha, beta):
    trace = greedy_decode(problem, model)
    s = entropy_mean(trace)
    K = max(0, min(4, round(alpha * s + beta)))
    if K == 0:
        return trace.answer
    step = argmin_step(trace, key=lp_min)
    samples = [(trace.answer, 1.0)]
    for _ in range(K):
        cont = regen_from(trace, step, model)
        w = 1.0 / (entropy_of_suffix(cont) + 1e-6)
        samples.append((extract_answer(cont), w))
    return weighted_majority(samples)
```

### Failure modes
- Continuous tuning of α, β requires more cal data than two thresholds.
- Suffix-entropy as weight can over-favor "confidently wrong" rephrasings.

---

## Method 6: **Cheap-First Disagree-Rescue (Anti-Gate)**

### Stage 1 (cheap probe, not a CP gate)
- Run greedy + 1 cheap regeneration (K=1, temp=0.8) at the lowest-`lp_min` step.
- If they **agree** → return immediately (cost 2x).

### Stage 2 (CP-conditioned escalation on disagreement)
- If they disagree, compute `entropy_mean`. If high → K=4 full rescue (cost 5x). If low → trust greedy (cost 2x).

### Cost
- Disagreement rate empirically ~25% when greedy is wrong. Avg = 1*1 + 1*1 + 0.25*P(high|disagree)*4. With P~0.6: 2 + 0.6*0.25*4 = 2 + 0.6 = **~2.6x**.
- Tighter version with K=1 instead of K=4 on escalation: ~2.25x.

### Hypothesis
- A single cheap sample is a strong correctness signal — when it agrees with greedy, error rate drops sharply. CP is then used only to disambiguate the remaining disagreement, where it's most informative.
- Inverts the usual gate-first logic: empirically, agreement with one alt sample is cheaper to compute than a global trajectory CP score.

### Pseudocode
```
def cheap_first_disagree(problem, model, tau_e):
    trace = greedy_decode(problem, model)
    step = argmin_step(trace, key=lp_min)
    alt = extract_answer(regen_from(trace, step, model, temp=0.8))
    if alt == trace.answer:
        return trace.answer
    if entropy_mean(trace) < tau_e:
        return trace.answer        # disagree but greedy looks confident
    cands = [trace.answer, alt] + [
        extract_answer(regen_from(trace, step, model)) for _ in range(3)
    ]
    return majority_vote(cands)
```

### Failure modes
- High-temp probe can disagree spuriously on confident-correct traces, triggering needless rescue.
- The "trust greedy on low-entropy disagreement" branch is a known soft-spot — could miss confident-wrong traces.

---

## Method 7: **Two-Layer CP (Trajectory Gate + Per-Step Gate, Local Rescue)**

### Stage 1a (trajectory gate)
- `entropy_mean > tau_lo` ⇒ flag for inspection. Else accept.

### Stage 1b (per-step gate, only on flagged)
- For flagged traces, compute per-step CP score `lp_min` over the trace. If **the worst step is genuinely an outlier** (per-step CP says: `lp_min < tau_step`), proceed to rescue. Else (no clearly bad step) → still accept greedy (the trace is "uniformly mediocre" — rescue probably won't help).

### Stage 2 (rescue)
- K=4 majority **only on the offending step**.

### Cost
- 0.75 * 1x + 0.25 * (0.6 * 5x + 0.4 * 1x) = 0.75 + 0.75 + 0.10 = **1.6x**.
- The double-gate kills false alarms cheaply (per-step CP is already computed during decoding).

### Hypothesis
- Pilot C confirmed K=4-at-worst-step works *only when there's a localized weak step*. Traces flagged by trajectory CP but with no localized outlier are likely "globally hard" — K=4 at a fake worst step won't help and just burns compute. Filter them out.

### Pseudocode
```
def two_layer_cp(problem, model, tau_traj, tau_step):
    trace = greedy_decode(problem, model)
    if entropy_mean(trace) < tau_traj:
        return trace.answer
    step_scores = [lp_min(s) for s in trace.steps]
    if min(step_scores) > tau_step:
        return trace.answer       # no localized weak step → don't rescue
    step = argmin(step_scores)
    cands = [trace.answer] + [
        extract_answer(regen_from(trace, step, model)) for _ in range(4)
    ]
    return majority_vote(cands)
```

### Failure modes
- The "uniformly mediocre" abstain-from-rescue branch sacrifices recall on globally-confused traces. Net accuracy impact depends on whether those traces are rescuable — pilot data suggests not, but worth verifying.
- Two thresholds compound calibration error.

---

## Summary Table

| # | Method | Stage 1 Score | Flag % | Stage 2 | Avg Cost | Expected Δacc |
|---|---|---|---|---|---|---|
| 1 | CP-Triage K=4 | entropy_mean | 25% | K=4 worst | 1.6-2.0x | +1-3pp |
| 2 | Union-Flag | entropy ∨ arith ∨ flips | 40% | K=4 worst | 2.6x | +2-5pp |
| 3 | Tiered Escalation | entropy (2 thresholds) | 50/15% | K=2→K=4→abstain | 1.8-2.3x | +2-4pp + selective |
| 4 | Provisional-Anchored | prov_match_rate | 30% | K=4 hinted at flip | 2.2x | +1-4pp |
| 5 | CP-Weighted Voting | entropy_mean (soft) | 100% (variable K) | weighted K=0..4 | 1.5x | +1-2pp |
| 6 | Cheap-First Disagree | K=1 probe + entropy | 25% disagree | K=4 if escalated | 2.25-2.6x | +2-4pp |
| 7 | Two-Layer CP | entropy_mean + lp_min | ~15% (after both) | K=4 worst step | **1.6x** | +1-3pp |

## Recommendations for Implementation Order

1. **Method 7 (Two-Layer CP)** — best cost/accuracy: filters false-alarm flags using per-step CP, lands ~1.6x while still capturing the rescuable cases. Cleanest extension of validated Pilot C.
2. **Method 1 (CP-Triage)** — simplest baseline; one threshold; easy to ablate.
3. **Method 3 (Tiered Escalation)** — best when abstention is allowed (selective-accuracy regime).
4. **Method 6 (Cheap-First)** — surprising baseline that may dominate in practice; one-sample agreement is a strong signal.
5. **Method 2 (Union-Flag)** — accuracy-max at the cost ceiling; reserve for compute-rich scenarios.
6. **Method 4 (PAR)** — needs provisional plumbing already; pairs well with Approach A's online decoding.
7. **Method 5 (Soft-Routing)** — most principled but hardest to calibrate; treat as research direction.

## Key Open Questions for Empirical Validation

- **Calibration transfer**: do `tau_lo` thresholds calibrated on GSM8K transfer to MATH? Probably not — need per-dataset cal splits.
- **Localized vs global weak steps**: what fraction of trajectory-CP-flagged traces have a localized lp_min outlier? This single number determines whether Method 7 dominates Method 1.
- **K=2 sufficiency**: in Method 3, what fraction of mid-confidence traces are correctly resolved by K=2 alone? If >70%, Tiered is the winner.
- **Provisional-answer overhead**: does provisional-answer querying actually add 1.2x or can it be made nearly free with logit-only readout? Determines viability of Method 4.
- **Worst-step localization noise**: when entropy_mean is high but `lp_min` floor isn't an outlier, K=4 at "worst" step is shooting in the dark. Method 7 explicitly handles this; others don't.

## Failure-Mode Themes Across Methods

1. **Confident-wrong traces**: Methods 1, 4, 5, 7 all rely on a CP score correlating with correctness. The ~10-15% of traces where the model is calibrated-wrong evade every gate. Only Method 6 (cheap-first probe) catches some of these via the disagreement signal.
2. **Threshold brittleness**: any method with two thresholds (3, 7) compounds calibration error. A held-out cal split per dataset is mandatory.
3. **Cost variance**: average cost ≤2x doesn't bound worst-case cost. For latency-sensitive deployments, prefer Method 7 (bounded by 5x) over Method 3 (bounded by 7x with abstain logic).
4. **Vote-tie ambiguity**: K=4 + greedy = 5 candidates; clean majority isn't guaranteed. All methods need a tiebreak rule (e.g., highest individual log-prob, or escalate to K=8). This affects ~5-10% of rescued traces.

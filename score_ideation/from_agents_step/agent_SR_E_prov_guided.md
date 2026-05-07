# Agent SR-E: Provisional-Answer-Guided Regeneration for CoT-CP

## Framing

The provisional-answer signal is the strongest handle we have on chain-of-thought correctness. Gold-leakage variants give ρ=+0.59-0.69, kept_acc=0.94. Free variants (`prov_match_latest_rate`, `prov_final_match_rate`) sit at kept_acc 0.79-0.87 — informative but noisy.

Key idea: **even when the provisional signal is too noisy for rejection, it is useful as a regeneration *hint*.** Provisional queries are 5-10 tokens, already in pipeline. Below: 7 methods with trigger, policy, cost, hypothesis, pseudocode, failure modes.

---

## Method E1: Modal-Anchored Soft-Hint Regeneration (MASH-R)

**Name.** Modal-Anchored Soft-Hint Regeneration (MASH-R).

**Trigger.** Worst step (lowest token-confidence or scoring-feature outlier) AND `prov_last_3_stable=1` (last 3 step-boundary provisionals agree). Stability is the safety check: we only inject when the model's running guess has converged.

**Provisional info used.** `modal_provisional` = mode of provisional answers across all completed step boundaries. Tie-break by latest.

**Regeneration policy.** Truncate at start of worst step. Resample step continuation with a *soft* hint inserted as a parenthetical in the system prefix or as a bracketed thought:

```
[The reasoning so far suggests the answer is approximately {modal_prov}.
Continue the next step. If your work disagrees, override; if it agrees, proceed cleanly.]
```

We use "approximately" and an explicit override permission so the hint is not a hard pin. K=2 alts; pick by joint score (token-conf of new step + alignment with modal).

**Cost.** 2 short regenerations + 1 cheap provisional query at end of step = ~2.1 forward passes' worth of tokens (regen is partial, not full trace). If we then complete the trace with greedy from the chosen alt, add 1 completion. Total ~3 passes.

**Hypothesis.** When `prov_last_3_stable=1`, modal is right ~85% of the time on math500 (extrapolating from gold-match rates). A soft hint shifts the regeneration distribution toward the correct manifold without locking the model into a wrong answer when stability is misleading.

**Pseudocode.**
```python
def mash_r(trace, scores, llm, sp_full, sp_short):
    worst = argmin(scores.step_conf)
    provs = trace.step_provisionals  # list of "X" strings
    if not last_k_stable(provs, k=3):
        return trace  # bail; signal too noisy
    modal = mode(provs)
    prefix = trace.tokens_up_to_step(worst)
    hint = (f"[The reasoning so far suggests the answer is approximately "
            f"{modal}. Continue the next step. If your work disagrees, "
            f"override; if it agrees, proceed cleanly.]\n")
    seed_prompt = prefix + hint
    alts = llm.generate([seed_prompt]*2, sp_short, n=1, temperature=0.7)
    best_alt = max(alts, key=lambda a: a.step_conf + 0.3*matches(a.prov, modal))
    completed = llm.generate(prefix + best_alt.text, sp_full)
    return completed
```

**Failure modes.**
1. *Modal is wrong but stable.* The model can be confidently mistaken — e.g., a misread units cue. The hint then locks in error. Mitigated only partially by "if your work disagrees, override"; in practice models follow the hint.
2. *Hint contaminates greedy completion.* Even when the regenerated step disagrees, the hint text remains in context and biases the rest. Mitigation: strip the hint from KV cache before completion (replace with "Continue:").
3. *Approximate-match failures.* "approximately 42" causes the model to round and produce 40. Mitigation: only inject when `modal` is integer/exact form, not for floats.

---

## Method E2: Mode-Filtered Resampling (MFR)

**Name.** Mode-Filtered Resampling (MFR).

**Trigger.** Worst step OR step with `prov_drift=1` (provisional changed at this boundary vs prior).

**Provisional info used.** `modal_provisional` from steps preceding the bad step (do not include any after, to avoid circularity).

**Regeneration policy.** Sample K=4 alts at worst step at temperature 0.7. For each alt, complete trace, extract final answer. **Filter**: keep only alts whose final answer matches `modal_prov_pre`. If none match, escalate temp to 1.0 and resample K=4. If still none, fall back to vanilla K=4 majority (Pilot C).

**Cost.** 4 regen + 4 short completions; in escalation case, +4 more. Average ~6 passes; worst-case ~12.

**Hypothesis.** Filtering acts as a Bayesian posterior: P(answer | trace, modal_pre). The modal-pre is a strong prior (kept_acc ~0.86), so filtering should beat unfiltered majority on cases where the bad step is due to a sampling slip rather than a deeper conceptual error.

**Pseudocode.**
```python
def mfr(trace, scores, llm, sp_full):
    worst = argmin(scores.step_conf)
    pre_provs = trace.step_provisionals[:worst]
    modal_pre = mode(pre_provs) if pre_provs else None
    if modal_pre is None:
        return pilot_c_majority(trace, llm, K=4)
    alts = resample_at_step(trace, worst, llm, K=4, T=0.7)
    matches = [a for a in alts if a.final_answer == modal_pre]
    if not matches:
        alts = resample_at_step(trace, worst, llm, K=4, T=1.0)
        matches = [a for a in alts if a.final_answer == modal_pre]
    if not matches:
        return majority_vote(alts + [trace])
    return max(matches, key=lambda a: a.mean_step_conf)
```

**Failure modes.**
1. *Modal-pre is the symptom of the bug.* If the bug is upstream of `worst`, modal_pre carries the bug, and filtering picks the wrong-but-consistent alt. Mitigation: also require alt's `prov_last_3` to be stable (a wrong-but-consistent alt often has flapping provisionals).
2. *Escalation thrash.* When the model has truly zero probability mass on modal_pre, both K=4 rounds fail, and we waste 8 passes before falling back. Mitigation: cap escalation, and short-circuit if first K=4 has 0 matches and all final answers agree on a different value (model has flipped its mind).

---

## Method E3: Iterative Provisional-Feedback Refinement (IPFR)

**Name.** Iterative Provisional-Feedback Refinement (IPFR).

**Trigger.** Worst-step is in bottom decile of step-conf. Bounded retry loop.

**Provisional info used.** Online provisional probe after each regenerated step; compare to running modal.

**Regeneration policy.** At worst step, regen ONE step (not full trace). Probe provisional. If new_prov == running_modal, accept and continue. Else regen step again. Up to N=3 retries per step. After acceptance, complete trace greedily.

**Cost.** Up to 3 step-regens + 3 provisional probes (5-10 tok each) + 1 completion = ~4 passes. Median expected ~2 passes (most steps accept first or second try).

**Hypothesis.** A single bad step is rarely a single bad token; it's a bad branch. Resampling at higher temperature *with rejection-on-prov-mismatch* is essentially Metropolis-Hastings using prov as the acceptance probe. Step-local feedback localizes corrections without re-doing the whole trace.

**Pseudocode.**
```python
def ipfr(trace, scores, llm, sp_step, sp_prov, sp_full, N=3):
    worst = argmin(scores.step_conf)
    pre_provs = trace.step_provisionals[:worst]
    if not pre_provs:
        return trace
    running_modal = mode(pre_provs)
    prefix = trace.tokens_up_to_step(worst)
    accepted = None
    for attempt in range(N):
        T = 0.7 + 0.1*attempt
        cand_step = llm.generate(prefix, sp_step, T=T).text
        new_prov = llm.generate(prefix + cand_step + PROV_PROBE,
                                sp_prov).answer
        if new_prov == running_modal:
            accepted = cand_step
            break
    if accepted is None:
        accepted = cand_step  # last try
    completed = llm.generate(prefix + accepted, sp_full)
    return completed
```

**Failure modes.**
1. *Provisional probe is itself stochastic.* Same step can yield different provisionals across queries. Mitigation: query prov twice with greedy decoding; if disagree, treat as no-match.
2. *N=3 always rejects.* When the true answer ≠ modal, every regen is rejected. Last-try fallback ensures we don't crash but defeats the point. Mitigation: track "all N rejected" as a feature and feed it back to the score (low confidence in any regen = signal to abstain rather than answer).
3. *Higher T destabilizes more than it diversifies.* Step quality may drop as T climbs. Mitigation: keep T ≤ 1.0; prefer prompt variation over T-escalation.

---

## Method E4: Early-Commit Cross-Validation (ECCV)

**Name.** Early-Commit Cross-Validation (ECCV).

**Trigger.** Always run as a post-trace check. Specifically, when `prov_at_step_3_or_earlier` is well-formed (the model committed by step 3).

**Provisional info used.** `prov_early` = provisional answer at the earliest step where the model produced a parseable guess (typically step 1-3). And the final answer of the trace.

**Regeneration policy.** If `final_answer == prov_early`, accept (the model committed early and stayed consistent — strong signal). If `final_answer != prov_early`, the trace either corrected an early miscommitment OR drifted away from a correct early instinct. Regenerate the trace with explicit framing:

```
[Earlier draft work suggested the answer might be {prov_early}.
The full solution gave {final_answer}. Re-solve carefully. State your final
answer explicitly.]
```

K=2 alternates from scratch (full prompt + injected note). Take majority across {original, alt1, alt2}.

**Cost.** 2 full regens. Heavy. Reserve for ambiguous cases.

**Hypothesis.** For arithmetic-style problems, the model's early instinct is often correct; later steps either confirm or rationalize away. For multi-step proofs, the early instinct is often wrong and refinement helps. The disagreement is itself a high-info signal — both regimes benefit from a second look.

**Pseudocode.**
```python
def eccv(trace, llm, sp_full):
    prov_early = first_parseable_prov(trace.step_provisionals)
    final = trace.final_answer
    if prov_early is None or prov_early == final:
        return trace  # consistent or no early commit
    note = (f"[Earlier draft work suggested the answer might be "
            f"{prov_early}. The full solution gave {final}. "
            f"Re-solve carefully. State your final answer explicitly.]\n")
    alts = []
    for _ in range(2):
        alt = llm.generate(trace.question + "\n" + note,
                           sp_full, T=0.7)
        alts.append(alt)
    candidates = [trace, *alts]
    return majority_by_final_answer(candidates)
```

**Failure modes.**
1. *Self-fulfilling bias.* The note exposes both candidates, and the model often defaults to the "more careful" trace which is whichever one we put last in the note. Mitigation: randomize order of `prov_early` and `final` in the note.
2. *Cost.* 2 full regens is expensive on long traces; only worth it when disagreement actually signals something. Mitigation: gate on whether step-confidence variance is high (high variance + early/late disagreement = run; low variance + disagreement = trust the trace).
3. *Cumulative bias toward early commit.* On problems where the model habitually miscommits then self-corrects, this method anchors back to wrong. Mitigation: track per-domain calibration and disable when prov_early kept_acc < 0.7 in train split.

---

## Method E5: Multi-Trace Provisional-Confidence Ensemble (MTPCE)

**Name.** Multi-Trace Provisional-Confidence Ensemble (MTPCE).

**Trigger.** Always (this is a generation-time strategy, not a rejection trigger).

**Provisional info used.** Per-trace `prov_match_latest_rate` (online, free) — measures how often the model's running guess equals its current best guess at step t.

**Regeneration policy.** Generate K=3 traces from the same question with T=0.7, distinct seeds. For each, compute (a) `prov_match_latest_rate` and (b) `prov_last_3_stable`. Score each trace as `S = prov_match_latest_rate + 0.5 * prov_last_3_stable`. Pick the trace with highest S as canonical. If top-2 tied, fall back to majority of final answers across all 3.

**Cost.** 3 full traces + 3*N_steps cheap provisional probes (already in pipeline). Roughly 3x baseline.

**Hypothesis.** A trace that "knew where it was going" — i.e., its running guess was stable and matched itself across steps — is both more confident and (per the data) more likely correct. We're not picking by final answer alone; we're picking by *self-consistency of running answer*.

**Pseudocode.**
```python
def mtpce(question, llm, sp_full, sp_prov, K=3):
    traces = []
    for seed in range(K):
        t = llm.generate(question, sp_full, seed=seed, T=0.7)
        t.provs = probe_provs_per_step(t, llm, sp_prov)
        t.match_latest = prov_match_latest_rate(t.provs)
        t.stable3 = int(last_3_agree(t.provs))
        t.score = t.match_latest + 0.5 * t.stable3
        traces.append(t)
    top = max(traces, key=lambda t: t.score)
    runner = sorted(traces, key=lambda t: -t.score)[1]
    if abs(top.score - runner.score) < 0.05:
        return majority_by_final_answer(traces)
    return top
```

**Failure modes.**
1. *All 3 traces share the same delusion.* Self-consistency rewards confidently wrong reasoning. Mitigation: when all 3 final answers agree but problem is in a known-hard category (e.g., AIME-style), still run a verifier pass.
2. *High-variance correct trace gets dropped.* A trace that legitimately self-corrects (prov flips mid-trace) has lower `match_latest_rate` but is right. Mitigation: also compute `prov_correction_quality` — does each prov change come with high-confidence local steps? Reward "decisive" flips, penalize flapping.
3. *Cost prohibitive for cheap problems.* Running 3 traces for GSM8K is wasteful. Mitigation: gate on first-trace score; if `match_latest_rate > 0.95` and `stable3=1`, accept first trace and skip 2 and 3 (adaptive K).

---

## Method E6: Stability-Conditioned Aggression (SCA)

**Name.** Stability-Conditioned Aggression (SCA).

**Trigger.** Two-branch policy. Compute `prov_last_3_stable` over the trace's tail.

**Provisional info used.** `prov_last_3_stable` (the last 3 step-boundary provisionals all equal). Boolean.

**Regeneration policy.**
- Branch A (`stable3=1`): trust the trace. **No regeneration**, even if a step has low confidence. Stable provisionals are a strong correctness proxy (kept_acc up to 0.94).
- Branch B (`stable3=0` AND there exists a worst-step in bottom decile): **aggressive K=4 majority** at worst step (Pilot C), with additional K=4 at the *second*-worst step too if the second-worst is also bottom-quartile.

**Cost.** Branch A: 0. Branch B: 4-8 passes. Average depends on stable3 rate; on math500 likely ~30-50% in Branch A, so amortized 2-4 passes.

**Hypothesis.** This is the cheapest method and exploits the stability binary as a circuit-breaker. On stable traces, we save compute and avoid unnecessary regeneration noise. On unstable ones, we spend more because the signal indicates real uncertainty. This decouples regeneration intensity from a single threshold.

**Pseudocode.**
```python
def sca(trace, scores, llm, sp_full):
    if last_3_agree(trace.step_provisionals):
        return trace  # Branch A: trust
    # Branch B
    sorted_steps = sorted(range(len(scores.step_conf)),
                          key=lambda i: scores.step_conf[i])
    worst, second = sorted_steps[0], sorted_steps[1]
    alts1 = resample_at_step(trace, worst, llm, K=4, T=0.7)
    cands = [trace] + alts1
    if scores.step_conf[second] < quartile_threshold(scores.step_conf, 0.25):
        alts2 = resample_at_step(trace, second, llm, K=4, T=0.7)
        cands += alts2
    return majority_by_final_answer(cands)
```

**Failure modes.**
1. *Stable but wrong (the dreaded case).* The model is confidently mistaken — Branch A gives the wrong answer with no recourse. This is exactly the kept_acc=0.94 ceiling: 6% of the time we miss. Mitigation: layer this with a calibrated abstention threshold.
2. *Branch B over-aggressive on long traces.* 4+4=8 regens on a trace with many low-conf steps. Mitigation: cap total at 8 and require steps ≥ 3 apart to prevent cascading.
3. *Stable3 is a coarse binary.* Could improve with `stable_k` for k∈{2,3,4,5}, but this adds tuning surface.

---

## Method E7: Hint-Verify Two-Pass (HVTP)

**Name.** Hint-Verify Two-Pass (HVTP).

**Trigger.** Worst step's local-region (worst step ± 1) has step-conf in bottom 15%, AND `modal_prov` exists.

**Provisional info used.** `modal_provisional` over all step boundaries; also a *fresh* provisional probe after the regenerated step.

**Regeneration policy.** Two-pass. Pass 1 (Hypothesis-Test): regen worst step with hint:

```
Hypothesis: the answer is X. Verify this by working through the next step
carefully. If the hypothesis is consistent with valid reasoning, continue
toward it; if not, identify why and adjust.
```

Pass 2 (Verify): probe a fresh provisional. If new_prov == modal, accept. If new_prov != modal, regenerate WITHOUT the hint (vanilla resample at the same step) and probe again. The two-pass design separates "hint helped" from "hint biased" cases.

**Cost.** 1 hinted regen + 1 prov probe + (conditional) 1 vanilla regen + 1 prov probe + 1 completion = 3-5 passes.

**Hypothesis.** A correctly-aimed hint speeds convergence to right answer; a wrongly-aimed hint produces a regen step that doesn't actually verify (the model writes plausible verification but the new prov drifts). The two-pass acts as a soft validity check on the hint itself.

**Pseudocode.**
```python
def hvtp(trace, scores, llm, sp_step, sp_prov, sp_full):
    worst = argmin(scores.step_conf)
    region = scores.step_conf[max(0,worst-1):worst+2]
    if percentile(scores.step_conf, region.mean()) > 0.15:
        return trace
    modal = mode(trace.step_provisionals)
    if modal is None: return trace
    prefix = trace.tokens_up_to_step(worst)
    hint = (f"Hypothesis: the answer is {modal}. Verify this by working "
            f"through the next step carefully. If the hypothesis is "
            f"consistent with valid reasoning, continue toward it; if "
            f"not, identify why and adjust.\n")
    cand = llm.generate(prefix + hint, sp_step, T=0.5).text
    p = llm.generate(prefix + cand + PROV_PROBE, sp_prov).answer
    if p == modal:
        return llm.generate(prefix + cand, sp_full)
    # hint failed validity check; vanilla resample
    cand2 = llm.generate(prefix, sp_step, T=0.7).text
    p2 = llm.generate(prefix + cand2 + PROV_PROBE, sp_prov).answer
    final_prefix = (prefix + cand) if p2 != modal else (prefix + cand2)
    return llm.generate(final_prefix, sp_full)
```

**Failure modes.**
1. *Hinted step writes fake verification.* The model may produce a plausible-looking verification that simply restates the hint without checking it; new prov matches modal trivially. Mitigation: require the regenerated step to contain a numeric or symbolic computation (regex check); reject pure restatements.
2. *Two-pass when hint is right.* When modal is correct, the verify-pass is wasted compute (always passes). Mitigation: skip verify when scoring features at the regenerated step are very high.
3. *Modal exists but is uniformly wrong.* Hint always passes verify (model self-confirms). This is the same kept_acc ceiling. Mitigation: if `prov_match_latest_rate` is suspiciously high (>0.99) and step-confidence is mediocre, treat as a sycophantic-loop pattern and disable hint.

---

## Cross-Cutting Notes

**Hint placement.** Bracketed inline notes rather than system-prompt edits — they get treated as "scratch" rather than "instruction."

**Hint contamination cleanup.** For E1/E7, after regenerated step is accepted, do a second completion with hint stripped (replace with "Continue:") and KV invalidated for the bracketed span.

**Probe template.** `PROV_PROBE = "\n\n(Quick check — current best guess at the final answer? Reply in a box.)\n"`. ~5-10 output tokens, ~30 input (cached).

**Ranking by expected gain on math500.** SCA (E6) > MASH-R (E1) > MFR (E2) > MTPCE (E5) > HVTP (E7) > IPFR (E3) > ECCV (E4). Run E6 first as a binary stable3 switch; then layer E1 inside Branch B; then try E2 as alternate Branch B; compare to Pilot C.

**Honest caveat.** ρ=0.59-0.69 numbers use gold. Free variants are weaker (kept_acc 0.79-0.86). "modal_provisional ≈ gold" bet is right ~80% of the time, so gains are bounded above by ~0.86 even with perfect filtering; methods must clear the resampling-variance noise floor.

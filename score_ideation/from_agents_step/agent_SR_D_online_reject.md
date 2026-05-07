# Agent SR-D: Online Hard Per-Step Rejection Methods for CoT-CP

## Framing: Online Per-Step Rejection vs. Worst-Step Branching

The K=4-majority-at-worst-step result (+1-2.5pp) says there is signal in step-level scores, but only at the tail. Single-alt replacement at the worst step is flat likely because (a) replacing one step among several bad ones doesn't fix the rest, and (b) worst-step mis-locates the true error when there are multiple weak steps.

**Online per-step rejection inverts the geometry.** Instead of "find the most-suspicious step in a finished trace, re-roll it," we say: "as we generate left-to-right, every time the score for the just-finished step crosses a threshold, regenerate THAT step, continue." Three consequences:

1. **Many-bad-steps fix:** a trace with 3 weak steps gets all three repaired locally rather than only the worst.
2. **KV is tail-truncated, not re-warmed.** Rejecting step *t* reuses KV up to start-of-step-*t* and only re-decodes from there.
3. **Online compute scaling.** Easy traces cost 1x; hard traces pay more, exactly when marginal value of compute is highest.

**Risks unique to online rejection** that every method must address:
- **Rejection cascades:** regenerated step *t'* may also be below threshold. Need a max-retry per step.
- **Score drift after regen:** thresholds calibrated on unconditional-generation distributions are biased once we condition on "previous step regenerated." Calibration must use the same regen procedure.
- **Threshold inflation under selection:** rejecting below quantile q and accepting regen is a min-of-K trick that biases scores upward. Conformal calibration must use *post-selection* scores.
- **KV invalidation:** rewinds need per-step KV checkpoints. vLLM prefix caching makes this clean if we store `(token_offset, step_idx)` pairs.

Now: 7 methods, each targeting a different point in the design space.

---

## Method 1: TRIPWIRE — single threshold, single regen

### Name
`tripwire_lpmin_T07`

### Trigger
`lp_min < tau_lp`, where `tau_lp` is the 20th-percentile lp_min on a labeled cal set, computed per model. Captures the "stuck on a hard token" failure mode.

### Per-step regeneration policy
- K = 1 alternative
- T = 0.7 (slightly above the base 0.0-0.3 we typically use, to break out of the bad mode)
- Cue: none (just resample from the step boundary with the existing prefix)
- Max retries per step: 1 (no rejection cascade — accept regen even if still bad)

### Cost
- Avg: ~1.3-1.5x (assuming ~20% of steps trip and tail-of-trace regen averages ~1.5 steps decoded)
- Worst case: ~3x for traces where many steps trip and each has a long tail

### Hypothesis
When lp_min is genuinely low, any random alternative is roughly as good as the original (often better, since the original was stuck). A single regen at moderate-high T should capture most of K=4's gain at a quarter the cost, *because we apply it to every bad step*. The many-bad-steps fix should compound.

### Pseudocode
```python
def tripwire_generate(prompt, tau_lp, model, max_steps=32):
    kv = model.warm(prompt)                          # vLLM prefix cache
    text = prompt
    step_starts = [len(text)]
    for s in range(max_steps):
        step, lp_min = model.decode_step(kv, T=0.0, stop="\n\n")
        if lp_min < tau_lp:
            # rewind: drop tokens of `step` from KV; re-decode at higher T
            kv = model.rewind_to(kv, step_starts[-1])
            step, lp_min = model.decode_step(kv, T=0.7, stop="\n\n")
        text += step
        kv = model.append(kv, step)                  # new KV state
        step_starts.append(len(text))
        if model.eos_seen(): break
    return text
```

### Failure modes
- If lp_min anti-correlates with quality (over-confidence on wrong tokens), we regen good steps. Validate sign in cal data.
- T=0.7 may overshoot fluency, producing hallucinations rather than fixes.
- 20th-percentile threshold means 20% trigger rate per step; on a 10-step trace ~2 regens average with high variance.

---

## Method 2: BEST-OF-2 — small-K rescue with confidence pick

### Name
`bo2_step_rescue`

### Trigger
Composite: `(entropy_max > tau_H) OR (prov_n_flips >= 2)`. OR captures two failure modes: local token uncertainty (entropy_max) and provisional-answer instability.

### Per-step regeneration policy
- K = 2 alternatives at T = 0.7
- Pick: argmax of `(mean_logprob - alpha * entropy_max)` of the new step
- No cue
- Max retries per step: 1 (we only do one round of K=2)

### Cost
- Avg: ~1.6-2.0x
- Worst case: ~4x (if every step trips, K=2 doubles cost on tripped steps)

### Hypothesis
K=2 with a confidence-based pick captures most of the K=4-majority signal because majority is dominated by confidence-of-best-sample, not vote agreement. The picker `(lp_mean - alpha * H_max)` leverages two validated scores at once. Cost is half of K=4 applied to *each* tripped step — should beat K=4-at-worst on multi-error traces.

### Pseudocode
```python
def bo2_step_rescue(prompt, tau_H, model, alpha=0.3):
    kv = model.warm(prompt)
    text = prompt
    while not done(text):
        ckpt = kv.snapshot()
        step0, scores0 = model.decode_step(kv, T=0.0)
        prov = model.query_provisional(kv, step0)        # one extra short decode
        n_flips = count_flips(prov_history + [prov])
        trip = (scores0.H_max > tau_H) or (n_flips >= 2)
        if trip:
            cand = [(step0, scores0)]
            for _ in range(2):
                kv2 = ckpt.clone()
                s, sc = model.decode_step(kv2, T=0.7)
                cand.append((s, sc))
            best = max(cand, key=lambda x: x[1].lp_mean - alpha*x[1].H_max)
            step, scores = best
            kv = ckpt; kv = model.append(kv, step)
        else:
            kv = model.append(kv, step0)
            step = step0
        text += step
        prov_history.append(query_provisional(kv))
    return text
```

### Failure modes
- Provisional-answer query adds ~1 short decode per step. Queried every step, baseline rises to ~1.2x even with no trips.
- alpha needs joint tuning with tau_H.
- prov_n_flips needs a sliding window; early in the trace there's nothing to flip against.

---

## Method 3: META-CUE — explicit self-correction prompt

### Name
`metacue_redo`

### Trigger
`lp_min < tau_lp` AND `prov_match_latest_rate < 0.5`. Fires only when local logprob *and* global provisional-answer agree the step is bad — high precision.

### Per-step regeneration policy
- After rejection, inject a short cue into the context before regen: `"\n\nWait — that last step seems off. Let me reconsider:\n\n"`
- K = 1 at T = 0.4
- Max retries per step: 1
- The cue itself is *not* kept in the final output (it's stripped post-hoc)

### Cost
- Avg: ~1.2x (high-precision trigger means low rate)
- Worst case: ~2.5x

### Hypothesis
The flat `self_correct_insertion` baseline was likely a *trigger-precision* failure (worst-step often isn't the real error site). With a high-precision AND trigger that fires mostly on real errors, the cue should act as a strong attention signal and unlock real correction.

### Pseudocode
```python
CUE = "\n\nWait — that last step seems off. Let me reconsider:\n\n"

def metacue_redo(prompt, tau_lp, model):
    kv = model.warm(prompt); text = prompt; ckpt_stack = []
    while not done(text):
        ckpt_stack.append(kv.snapshot())
        step, sc = model.decode_step(kv, T=0.0)
        prov = model.query_provisional(kv, step)
        rate = prov_match_latest_rate(prov_history + [prov])
        if sc.lp_min < tau_lp and rate < 0.5:
            kv = ckpt_stack[-1].clone()
            kv = model.append(kv, CUE)
            step2, sc2 = model.decode_step(kv, T=0.4)
            kv = ckpt_stack[-1].clone()
            kv = model.append(kv, step2)        # cue dropped from final
            step = step2
        else:
            kv = model.append(kv, step)
        text += step
    return strip_cues(text)
```

### Failure modes
- Cue may bias the model toward over-correcting (changing a correct step to a wrong one).
- Stripping the cue from text but keeping it in KV may leave the model in a hedge-y mode for the rest of the trace.
- High-precision trigger may be too precise and miss moderately-bad steps.

---

## Method 4: ESCALATION — tiered regen budget

### Name
`escalate_K2_to_K4`

### Trigger
Two-tier:
- Tier-1 trigger: `lp_min < tau_lp_loose` (e.g. 30th percentile) → K=2 at T=0.7
- Tier-2 trigger: K=2 alts disagree (their provisional-answer projections diverge) → escalate to K=4 at T=1.0, majority pick

### Per-step regeneration policy
- Tier-1: K=2, T=0.7, pick by lp_mean
- Tier-2: K=4, T=1.0, pick by majority over their projected provisional answers
- Max retries: bounded by tier (no further escalation past tier-2)
- After tier-2, accept majority even if low-confidence

### Cost
- Avg: ~1.5-1.8x (most trips resolve at tier-1)
- Worst case: ~5-6x (rare cases where every step escalates to tier-2)

### Hypothesis
The closest online analogue to K=4-at-worst-step, but adaptive. Most low-lp_min steps have a clear winner among 2 alts; only diverging cases need K=4. Should dominate K=4-at-worst because it can fix multiple errors and tier-2 fires only where K=4 majority is most informative.

### Pseudocode
```python
def escalate_step(kv, ckpt, model):
    a1 = model.decode_step(ckpt.clone(), T=0.7)
    a2 = model.decode_step(ckpt.clone(), T=0.7)
    p1 = model.query_provisional(ckpt.clone(), a1.text)
    p2 = model.query_provisional(ckpt.clone(), a2.text)
    if p1 == p2:
        return max([a1,a2], key=lambda x: x.lp_mean)
    # tier-2
    extra = [model.decode_step(ckpt.clone(), T=1.0) for _ in range(4)]
    projs = [model.query_provisional(ckpt.clone(), e.text) for e in extra]
    winner_proj = majority(projs + [p1, p2])
    pool = [(a1,p1),(a2,p2)] + list(zip(extra, projs))
    candidates = [x for x,p in pool if p == winner_proj]
    return max(candidates, key=lambda x: x.lp_mean)

def escalate_generate(prompt, tau_loose, model):
    kv = model.warm(prompt); text = prompt
    while not done(text):
        ckpt = kv.snapshot()
        step, sc = model.decode_step(kv, T=0.0)
        if sc.lp_min < tau_loose:
            best = escalate_step(kv, ckpt, model)
            kv = ckpt; kv = model.append(kv, best.text)
            text += best.text
        else:
            kv = model.append(kv, step); text += step
    return text
```

### Failure modes
- Provisional-answer query is the bottleneck — called on every alt at every tripped step.
- Tier-2 cost concentrates on long-context problems, the regime where conformal calibration is hardest.
- Majority over projected provisional answers collapses if the answer space is continuous; needs fuzzy match.

---

## Method 5: GIVE-UP — abort-step-and-skip on persistent failure

### Name
`giveup_skip_step`

### Trigger
Standard `lp_min < tau_lp`. If a regenerated step is *also* below threshold, abort the step and emit `[skipped]\n\n` rather than continuing to regen.

### Per-step regeneration policy
- K=1 at T=0.7 first attempt
- If still bad, K=1 at T=1.0 second attempt
- If still bad, emit `[skipped]\n\n` and continue
- After 3 cumulative skips in a single trace, abort the whole trace (escalate to abstain — model returns "no answer")

### Cost
- Avg: ~1.4x
- Worst case: ~3x (capped by skip mechanism — can't infinitely regen)

### Hypothesis
Some steps are unrecoverable; regen is wasted, and continuing with a known-bad step poisons downstream. Better to skip or abort. Conformal-aware: produces an explicit abstain signal the conformal layer can use to set quantiles.

### Pseudocode
```python
def giveup_skip(prompt, tau_lp, model, max_skips=3):
    kv = model.warm(prompt); text = prompt; skips = 0
    while not done(text):
        ckpt = kv.snapshot()
        step, sc = model.decode_step(kv, T=0.0)
        if sc.lp_min < tau_lp:
            kv2 = ckpt.clone()
            step2, sc2 = model.decode_step(kv2, T=0.7)
            if sc2.lp_min < tau_lp:
                kv3 = ckpt.clone()
                step3, sc3 = model.decode_step(kv3, T=1.0)
                if sc3.lp_min < tau_lp:
                    skips += 1
                    if skips >= 3: return ABSTAIN
                    step = "[skipped]\n\n"
                    kv = ckpt; kv = model.append(kv, step)
                else:
                    kv = ckpt; kv = model.append(kv, step3); step = step3
            else:
                kv = ckpt; kv = model.append(kv, step2); step = step2
        else:
            kv = model.append(kv, step)
        text += step
    return text
```

### Failure modes
- "[skipped]" placeholders may confuse the downstream model; try natural cues like "(I'll come back to this)".
- Abstain rate may exceed the conformal coverage budget.
- 3-skips-then-abort is brittle to step length — long steps get equal weight to short ones.

---

## Method 6: GRANULAR — sentence-level + paragraph-level dual rejection

### Name
`dual_granularity_reject`

### Trigger
Two rejection checks at two granularities:
- **Sentence-level (fine):** at each sentence boundary inside a step, check `lp_min_sent < tau_sent`. If trip, regen *just this sentence* with K=1 at T=0.5.
- **Paragraph-level (coarse):** at end of step, check `lp_min_step < tau_step` (overall). If trip *despite* sentence-level rescues, do a full paragraph regen with K=2 at T=0.7.

### Per-step regeneration policy
- Sentence: K=1, T=0.5, max 1 retry per sentence
- Paragraph: K=2, T=0.7, picks by lp_mean
- Sentence-level rescues are cheap (1 sentence ~ 20 tokens vs step ~ 80 tokens)

### Cost
- Avg: ~1.5-1.8x (most trips are sentence-local)
- Worst case: ~4x

### Hypothesis
Errors often live at sentence granularity (a bad clause, an arithmetic slip), but paragraph rejection rolls back the whole step. Finer granularity is cheaper per fix and more surgical; combining both catches sentence slips and step-level coherence breaks.

### Pseudocode
```python
def dual_gran(prompt, tau_sent, tau_step, model):
    kv = model.warm(prompt); text = prompt
    while not done(text):
        step_ckpt = kv.snapshot()
        step_text = ""
        while not step_done(step_text):
            sent_ckpt = kv.snapshot()
            sent, sc_s = model.decode_sentence(kv, T=0.0, stop=[".","\n\n"])
            if sc_s.lp_min < tau_sent:
                kv = sent_ckpt.clone()
                sent2, sc_s2 = model.decode_sentence(kv, T=0.5)
                kv = sent_ckpt; kv = model.append(kv, sent2); sent = sent2
            else:
                kv = model.append(kv, sent)
            step_text += sent
        # paragraph-level check on the assembled step
        sc_step = score_step(step_text, model)
        if sc_step.lp_min < tau_step:
            kv = step_ckpt.clone()
            cand = [model.decode_step(step_ckpt.clone(), T=0.7) for _ in range(2)]
            best = max(cand, key=lambda x: x.lp_mean)
            kv = step_ckpt; kv = model.append(kv, best.text); step_text = best.text
        text += step_text
    return text
```

### Failure modes
- Sentence detection is finicky for code/math (decimal points break splitting).
- Two thresholds doubles calibration cost.
- Sentence rescue may be locally-better but globally-incoherent; paragraph check is the backstop but adds overhead.

---

## Method 7: BUDGET-ADAPTIVE — promising-trace amplification

### Name
`budget_amplify_promising`

### Trigger
- Standard `lp_min < tau_lp` per step.
- *Per-trace* budget: total regen budget B = 4 step-equivalents.
- **Adaptive allocation:** budget remaining is preferentially spent on traces that look "promising" — defined as `prov_match_latest_rate >= 0.7` so far AND running mean lp_mean of accepted steps in top quartile.
- A non-promising trace that exhausts B early is allowed to continue *without* further rejection (cheap accept).

### Per-step regeneration policy
- Promising trace, budget remaining: K=2 at T=0.7, pick by lp_mean
- Non-promising trace: K=1 at T=0.7
- Promising trace, budget exhausted: no rejection (accept all)

### Cost
- Avg: ~1.5-2x (capped by B)
- Worst case: ~5x (only promising traces get the full budget)

### Hypothesis
Allocate compute where it most helps. A consistent, confident trace with one bad step is likely to be saved by fixing it; a wandering trace won't. This makes regen budget a function of running quality estimate — the same signal the conformal layer reads.

### Pseudocode
```python
def budget_amplify(prompt, tau_lp, model, B=4):
    kv = model.warm(prompt); text = prompt; budget = B
    accepted_lps = []; prov_history = []
    while not done(text):
        ckpt = kv.snapshot()
        step, sc = model.decode_step(kv, T=0.0)
        prov = model.query_provisional(kv, step)
        match_rate = prov_match_latest_rate(prov_history + [prov])
        promising = (match_rate >= 0.7 and
                     mean_or_zero(accepted_lps) >= top_quartile_threshold)
        if sc.lp_min < tau_lp and budget > 0:
            if promising:
                cands = [model.decode_step(ckpt.clone(), T=0.7) for _ in range(2)]
                best = max(cands + [step_obj(step,sc)], key=lambda x: x.lp_mean)
                budget -= 2
            else:
                best = model.decode_step(ckpt.clone(), T=0.7)
                budget -= 1
            kv = ckpt; kv = model.append(kv, best.text); step = best.text; sc = best.sc
        else:
            kv = model.append(kv, step)
        text += step; accepted_lps.append(sc.lp_mean); prov_history.append(prov)
    return text
```

### Failure modes
- "Promising" is self-fulfilling — early-correct-looking traces get amplified, but a trace with one early bad step poisoning the rest gets the least help.
- Global B=4 doesn't adapt to trace length; long traces are under-budget.
- match_rate and lp_mean coupling can correlate with the conformal score, causing circular calibration.

---

## Cross-method considerations and recommended ordering

Ranked by likely signal-to-cost:

1. **TRIPWIRE (M1):** simplest; tells us whether online rejection adds anything over worst-step. Baseline.
2. **BEST-OF-2 (M2):** most likely to dominate K=4-at-worst-step at half cost. Strong default.
3. **ESCALATION (M4):** the "smart K=4" — highest accuracy when budget allows.
4. **META-CUE (M3):** orthogonal — tests whether explicit cueing helps when trigger precision is high.
5. **GRANULAR (M6):** orthogonal — tests granularity hypothesis. Pairs with any of M1-M4.
6. **GIVE-UP (M5):** conformal-aware variant — produces explicit abstains.
7. **BUDGET-ADAPTIVE (M7):** speculative; requires a good "promising" detector.

**Calibration:** label ~500 cal traces with step-level correctness, find each score's threshold maximizing step-level F1, then re-run the rejection pipeline to set the conformal quantile on *post-selection* scores — otherwise coverage is biased by selection.

**vLLM notes:** with `enable_prefix_caching=True`, rewinding to a step boundary is effectively free; the cost is the new tokens decoded after rewind (~80 tokens per regen), matching the cost analyses above.

**Key uncertainty:** all of this hinges on meaningful step boundaries. If the model's "steps" are sometimes one sentence and sometimes a whole derivation, per-step thresholds are ill-defined. Sanity-check by histogramming step-token-lengths on cal data; if bimodal, prefer Method 6's sentence granularity as primary rather than complementary.

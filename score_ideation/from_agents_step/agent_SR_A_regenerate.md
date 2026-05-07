# Agent SR-A — Step Regeneration / Replacement Methods for CoT-CP

**Goal:** propose 5–7 concrete methods that detect a "bad" reasoning step using
the score family we already validated (entropy_max, top1_margin_min, lp_drawdown,
arith_violations, prov_n_flips, prov_entropy, etc.) and **act** on it to recover
the trace, with the explicit aim of improving final-answer accuracy (not merely
selective accuracy / coverage). Cost budget: ≤ 4× extra forward passes per question.

Below, "step t" denotes a single reasoning step (a delimited line / sentence in
the CoT). `s_i(t)` denotes score `i` evaluated on step t. `q_i^*` denotes the
calibrated threshold for score `i` (chosen on a held-out calibration split,
e.g. the 90th percentile of "good" steps in correct traces).

A baseline is assumed: Pilot C (K=4, T=0.7, majority-vote at the lp_min step)
gives +1.5–5pp on its own. We try to beat it.

---

## Method 1 — `intersection_trigger_branch_and_prm`

**Trigger.** Step t is "bad" iff **at least 2 of 3 independent score families**
flag it simultaneously:
1. *Internal logit signal:* `lp_drawdown(t) > q_lp` **or** `entropy_max(t) > q_H`
2. *Symbolic signal:* `arith_violations(t) ≥ 1`
3. *Provisional-answer signal:* `prov_n_flips(t) ≥ 2` **or** `prov_entropy(t) > q_pe`

The conjunction is what makes this different from a UNION trigger: it treats the
trigger as a **high-precision detector**, so every replacement attempt is
"earned" and we don't waste budget on borderline steps.

**Replacement action.** From step t, sample K=4 alternative continuations at
T=0.7. Score each completed branch with the PRM; pick argmax PRM. If PRM not
available, fall back to majority vote on extracted answers.

**Cost.** Up to 4 extra forwards on triggered questions. On non-triggered
questions, 0 extra. Empirically, intersection triggers on 15–25% of questions,
so amortized ≈ 0.6–1.0× extra.

**Hypothesis.** Pilot C's lp_min trigger already triggers on ~100% of questions
(it always picks *some* worst step). The waste is in cases where the worst step
is actually fine. Intersection only acts when multiple signals corroborate, so
each replacement has higher P(replacement helps | triggered).

**Pseudocode.**
```python
def intersection_branch(prompt, traj, scores, K=4, T=0.7):
    bad = []
    for t in range(len(traj)):
        votes = 0
        if scores.lp_drawdown[t] > Q_LP or scores.entropy_max[t] > Q_H: votes += 1
        if scores.arith_violations[t] >= 1: votes += 1
        if scores.prov_n_flips[t] >= 2 or scores.prov_entropy[t] > Q_PE: votes += 1
        if votes >= 2: bad.append(t)
    if not bad: return traj.final_answer
    t_star = bad[0]               # earliest is best (compounding)
    prefix = traj.text_up_to(t_star)
    branches = vllm.sample(prefix, n=K, temperature=T, max_tokens=512)
    if HAVE_PRM:
        return branches[argmax([prm.score(b) for b in branches])].answer
    return majority_vote([b.answer for b in branches])
```

**Failure modes.** (a) If all three signals are correlated (e.g. arith errors
inflate logit entropy), we just recover lp_min. Mitigated by *partial-correlation
audit* on calibration set. (b) Earliest bad step could be a recoverable hiccup
where the model self-corrects on its own; we'd inject noise. Mitigated by also
requiring the step to be in the front 60% of the chain.

**CP integration.** Trigger and replacement happen *before* the calibrated
abstention score is computed. Coverage guarantee is preserved as long as we
re-calibrate the abstention quantile on a held-out set with the replacement
policy active (i.e. treat the entire pipeline as a black-box predictor when
calibrating). No theoretical loss.

---

## Method 2 — `temperature_escalation_consensus`

**Trigger.** `lp_min(t)` (the existing winner) **or** `entropy_max(t) > q_H`.

**Replacement action.** Two-stage:
- Stage 1: sample K=2 at low temperature **T=0.3**. If both branches produce the
  *same* extracted answer, accept it. (Cheap, high-precision: agreement at low
  T is strong evidence.)
- Stage 2: only if stage 1 disagrees, sample K=4 more at **T=1.0** (high
  diversity). Take majority vote over the 6 total branches, breaking ties by PRM
  if available, else by lowest mean lp_drawdown.

The intuition: when the model is *confidently* wrong (low entropy on a wrong
token), low-T resampling won't help — but the very fact that low-T branches
disagree is itself a useful signal that this step is genuinely uncertain, and we
should escalate to high-T.

**Cost.** 2 extra forwards in the easy case (~50–60% of triggers), 6 in the
hard case. Expected ≈ 3.5×.

**Hypothesis.** Most "bad" lp_min steps are actually fine and just have a noisy
local logit dip. Two cheap low-T samples cheaply confirm. The expensive K=4 T=1.0
fan-out is reserved for genuinely ambiguous steps where diversity actually
matters.

**Pseudocode.**
```python
def temp_escalation(prompt, traj, scores):
    t_star = argmax(scores.lp_drawdown)
    if scores.lp_drawdown[t_star] < Q_LP and scores.entropy_max[t_star] < Q_H:
        return traj.final_answer
    prefix = traj.text_up_to(t_star)
    cheap = vllm.sample(prefix, n=2, temperature=0.3, max_tokens=512)
    if cheap[0].answer == cheap[1].answer:
        return cheap[0].answer                     # stage-1 accept
    extra = vllm.sample(prefix, n=4, temperature=1.0, max_tokens=512)
    pool  = cheap + extra
    if HAVE_PRM:
        return pool[argmax([prm.score(b) for b in pool])].answer
    return majority_vote([b.answer for b in pool])
```

**Failure modes.** (a) Low-T branches collude on a *systematically* wrong path
(model is confidently wrong). Stage-1 will accept the wrong answer. Mitigated by
also requiring stage-1 agreement to match the *original* trace's answer — if
they all three agree, accept; else escalate. (b) Stage-2 becomes expensive on
hard problems (where it's most needed) — acceptable.

**CP integration.** Same as Method 1: re-calibrate end-to-end. Note that
escalation may inflate latency variance, which can matter for streaming-CP
variants.

---

## Method 3 — `backtrack_two_steps`

**Trigger.** `prov_n_flips(t) ≥ 2` AND step t is not the first step. Rationale:
multiple flips of the provisional answer means the trace is committing to the
wrong direction *somewhere around* t, not necessarily *at* t. The actual mistake
is often at t-1 (the premise that produced the bad inference at t).

**Replacement action.** Cut steps {t-1, t} from the context. Regenerate from the
end of step t-2 with K=2 at T=0.7. Take majority vote over the K branches'
final answers (or PRM if available).

**Cost.** 2 extra forwards per triggered question (~30% of questions trigger
based on prov flips), so ≈ 0.6× amortized.

**Hypothesis.** "Bad step" detectors often detect the *symptom*, not the *cause*.
Backtracking 1 extra step gives the model a chance to take a different premise,
which can cascade into a fundamentally different (correct) chain. This is the
main thing single-step replacement cannot do.

**Pseudocode.**
```python
def backtrack_two(prompt, traj, scores, K=2, T=0.7):
    flips = scores.prov_n_flips
    cands = [t for t in range(2, len(traj)) if flips[t] >= 2]
    if not cands: return traj.final_answer
    t_star = cands[0]
    prefix = traj.text_up_to(t_star - 1)           # cut steps t-1 AND t
    branches = vllm.sample(
        prefix, n=K, temperature=T, max_tokens=1024,
        stop=traj.step_separator                   # let it write multi-step
    )
    answers = [b.answer for b in branches] + [traj.final_answer]
    return majority_vote(answers)                  # incumbent gets a vote too
```

**Failure modes.** (a) Step t-1 was actually correct and t was the only bad step
— we throw away good reasoning. Partially mitigated by *including the original
final answer in the vote* so a strong original gets a third vote. (b) On short
chains (≤3 steps), we backtrack to the question itself and effectively
re-roll the whole problem; this is fine but degenerates to a self-consistency
baseline.

**CP integration.** Re-calibrate. The fact that backtracking can shorten the
chain means step-level scores need to be aggregated post-hoc on the *new*
trajectory. Recompute the abstention score on the post-replacement trace.

---

## Method 4 — `self_correct_insertion`

**Trigger.** `arith_violations(t) ≥ 1` (sympy/regex-checkable arithmetic flagged
as wrong). High-precision symbolic trigger.

**Replacement action.** Do **not** delete step t. Instead, append a templated
self-correction marker after step t and let the model continue:
```
<step t verbatim>
Wait — I should double-check this calculation. Let me redo it carefully.
```
Then continue greedy-decoding (or K=2 at T=0.5) until end-of-trace. Take the
new final answer.

**Cost.** 1–2 extra forwards per triggered question. Trigger rate ≈ 10–20%.
Amortized ≈ 0.2×.

**Hypothesis.** When the trigger is symbolic and high-precision (arithmetic
violation detectable by sympy), the model knows *how* to fix it; it just needs
permission. Insertion preserves all good reasoning before t and gives the model
a structured second-attempt context. Empirically, "Wait, let me reconsider"
prompts work well in self-refinement papers (e.g. Madaan et al. 2023).

**Pseudocode.**
```python
def self_correct_insert(prompt, traj, scores, T=0.5, K=2):
    av = scores.arith_violations
    cands = [t for t in range(len(traj)) if av[t] >= 1]
    if not cands: return traj.final_answer
    t_star = cands[0]
    insertion = "\nWait — I should double-check this calculation. Let me redo it.\n"
    new_prefix = traj.text_up_to(t_star + 1) + insertion
    branches = vllm.sample(new_prefix, n=K, temperature=T, max_tokens=512)
    if HAVE_PRM:
        return branches[argmax([prm.score(b) for b in branches])].answer
    return majority_vote([b.answer for b in branches] + [traj.final_answer])
```

**Failure modes.** (a) Sympy false positive (model wrote in a notation sympy
can't parse). Mitigated by requiring the violation to also coincide with
`top1_margin_min(t) < q_m` (model also low-confidence). (b) Self-correct
insertion can spiral into infinite "actually wait" loops — bound max_tokens.

**CP integration.** Insertion changes the trace text but not the calibration
contract: the score functions are recomputed on the post-insertion trace.
Coverage holds with re-calibration.

---

## Method 5 — `provisional_anchor_regeneration`

**Trigger.** `prov_entropy(t) > q_pe` *and* the provisional answer at step t
disagrees with the dominant provisional answer over steps `[1..t-1]`.
Interpretation: the trace was heading toward answer A, then at step t something
diverted it toward answer B, with high uncertainty.

**Replacement action.** Find the *modal* provisional answer M over the prefix
`[1..t-1]`. Regenerate step t and onward, with the prompt augmented by a soft
hint:
```
(Reasoning toward an answer of approximately {M}.)
```
inserted *before* step t. Sample K=2 at T=0.6, take majority vote, but **only
accept the new answer if its PRM/log-prob is higher than the original**;
otherwise keep original.

**Cost.** 2 extra forwards on triggered questions (~25%), so ≈ 0.5×.

**Hypothesis.** The prefix already encoded a (probably correct) trajectory
toward M; the divergence at t is more likely a slip than a genuine
reconsideration. Anchoring on M nudges the regeneration to recover the natural
continuation. This is essentially a "self-conditioned" hint with no external
information leaked.

**Pseudocode.**
```python
def prov_anchor_regen(prompt, traj, scores, K=2, T=0.6):
    pa = scores.provisional_answers     # one per step
    pe = scores.prov_entropy
    for t in range(2, len(traj)):
        if pe[t] <= Q_PE: continue
        modal = mode(pa[:t])
        if pa[t] == modal: continue
        hint = f"(Reasoning toward an answer of approximately {modal}.)\n"
        prefix = traj.text_up_to(t) + hint
        branches = vllm.sample(prefix, n=K, temperature=T, max_tokens=512)
        cand = majority_vote([b.answer for b in branches])
        if HAVE_PRM and prm.score_full(prefix + cand) > prm.score_full(traj.text):
            return cand
        elif lp_mean(branches) > lp_mean(traj.tail_from(t)):
            return cand
        return traj.final_answer
    return traj.final_answer
```

**Failure modes.** (a) Modal prefix answer is itself wrong — anchoring locks in
the wrong region. Important: this is bounded by the original prov-answer
distribution; we never anchor outside what the model already proposed.
(b) "Hint" can leak as explicit constraint and the model just outputs M without
real reasoning. The PRM/log-prob acceptance gate guards against degenerate copy.

**CP integration.** Crucial: the hint is constructed entirely from the model's
own provisional outputs, so we are not injecting external information that
breaks i.i.d. assumptions. Re-calibrate as with the others.

---

## Method 6 — `multishot_rescore_with_budget`

**Trigger.** Initial trigger is `lp_min` (the proven winner). What's new is
that after one regeneration we **re-score the new step** and may regenerate
again, up to N=3 attempts, with a *shared* budget of 4 forward passes total.

**Replacement action.** Iterative refinement with a budget pool:
- Try 1: K=2 at T=0.5. Re-score regenerated step. If score drops below
  threshold, accept.
- Try 2 (if needed): K=2 at T=0.8 from same prefix.
- Try 3 (if needed): K=2 at T=1.0 with self-correct insertion (Method 4 fallback).
After exhausting budget, take the *best-scored* regenerated branch (lowest
lp_drawdown on its worst step).

**Cost.** Hard-capped at 4 forwards (so always 4× when triggered, 0× otherwise).

**Hypothesis.** A single regeneration can fail to land on a good step. Multi-shot
with re-scoring uses the score itself as a stopping criterion — we only spend
the next chunk of budget if the current attempt is still bad. This concentrates
compute where it matters. Importantly, cap at 4× keeps total cost tractable.

**Pseudocode.**
```python
def multishot_rescore(prompt, traj, scores, budget=4):
    t_star = argmax(scores.lp_drawdown)
    if scores.lp_drawdown[t_star] < Q_LP: return traj.final_answer
    prefix = traj.text_up_to(t_star)
    pool, spent = [], 0
    plan = [(2, 0.5, ""), (2, 0.8, ""),
            (2, 1.0, "\nLet me reconsider this step.\n")]
    for K, T, ins in plan:
        if spent + K > budget: break
        branches = vllm.sample(prefix + ins, n=K, temperature=T, max_tokens=512)
        spent += K
        for b in branches:
            new_score = score_step(b.first_step)
            pool.append((new_score, b))
            if new_score < Q_LP_GOOD:
                return b.answer                    # early accept
    pool.sort(key=lambda x: x[0])
    best = pool[0][1]
    return majority_vote([best.answer, traj.final_answer])
```

**Failure modes.** (a) Score is a poor predictor of *recoverability* — we keep
re-rolling and never improve. Bounded by budget. (b) Early-accept threshold
`Q_LP_GOOD` is hard to set; tune on calibration. (c) For some questions, no
regeneration is good; we fall back to majority vote with the original.

**CP integration.** Re-calibrate. Variable cost is fine because CP guarantees
are about marginal coverage, not latency.

---

## Method 7 — `skip_then_resume_with_audit`

**Trigger.** **Intersection** of: `entropy_max(t) > q_H` AND
`top1_margin_min(t) < q_m` AND step t is *not load-bearing*. "Not load-bearing"
is approximated by: the deltas in provisional answer between t-1 and t+1 in the
*original trace* are zero — i.e. step t didn't change the running answer.

**Replacement action.** Literally **delete** step t from the context and
regenerate step t+1 onward (one continuation, K=1 at T=0.0 / greedy).

**Cost.** 1 extra forward per triggered question (~5–10% of questions trigger
the strict intersection), amortized ≈ 0.1×.

**Hypothesis.** Some steps are noise — high entropy filler that doesn't actually
contribute to the answer. Removing them gives the model a cleaner context and
may avoid the "garden path" effect of having to consume a noisy step.
This is the cheapest method by far and is the right choice when the model
*recovered* on its own (provisional answer same before and after).

**Pseudocode.**
```python
def skip_resume(prompt, traj, scores):
    H, M, pa = scores.entropy_max, scores.top1_margin_min, scores.provisional_answers
    for t in range(1, len(traj) - 1):
        if H[t] <= Q_H or M[t] >= Q_M: continue
        if pa[t-1] != pa[t+1]: continue            # step IS load-bearing
        new_text = traj.text_without_step(t)
        cont = vllm.sample(new_text, n=1, temperature=0.0, max_tokens=512)[0]
        # audit: only accept if new lp_drawdown profile is strictly better
        new_scores = score_trace(new_text + cont.text)
        if max(new_scores.lp_drawdown) < max(scores.lp_drawdown):
            return cont.answer
    return traj.final_answer
```

**Failure modes.** (a) Misjudging "load-bearing" — the provisional-answer
heuristic is imperfect; deleted step might have set up a variable used later.
Mitigated by audit gate (only accept if max-drawdown of new trace decreases).
(b) Greedy resampling can still produce the same broken chain. Could fall back
to K=2 at T=0.3 if greedy reproduces the deleted step verbatim.

**CP integration.** Skip reduces the trace length and changes the score
distribution. Critical to recompute the calibrated abstention score on the
*post-skip* trace and re-calibrate. Sequential application of skip + other
methods is fine, but each must be in the calibration pipeline.

---

## Cost / Effect Summary

| Method | Trigger precision | Amortized cost | Hypothesized lift over Pilot C |
|---|---|---|---|
| 1. intersection_trigger_branch_and_prm | high | 0.6–1.0× | +1–3pp |
| 2. temperature_escalation_consensus | medium | 3–4× | +2–4pp |
| 3. backtrack_two_steps | medium | 0.6× | +1–2pp |
| 4. self_correct_insertion | high (symbolic) | 0.2× | +0.5–1.5pp |
| 5. provisional_anchor_regeneration | medium | 0.5× | +1–2pp |
| 6. multishot_rescore_with_budget | varies | 4× (cap) | +2–4pp |
| 7. skip_then_resume_with_audit | very high | 0.1× | +0.2–0.8pp |

## Recommended Combo for Initial Run

If we can pick only one experiment to run first, pick **Method 1** (intersection
trigger + PRM-select), because (a) it reuses Pilot C's K=4 T=0.7 sampling code,
(b) the only new component is an AND-of-three-thresholds detector that is cheap
to ablate, and (c) the failure modes are well understood.

If we have budget for two experiments, add **Method 2** (temperature escalation),
because it directly tests whether Pilot C's T=0.7 single-shot is leaving lift on
the table, and the early-stop-at-low-T optimization keeps cost bounded.

For an aggressive, accuracy-first sweep with budget, run **Method 6** (multishot
rescore) — it stress-tests the score's usefulness as a stopping criterion, which
is itself a useful diagnostic for the whole CoT-CP score validation effort.

## CP Coverage Notes (general)

All seven methods compose with CoT-CP's marginal coverage guarantee under the
following discipline:
1. Treat the entire regeneration policy as part of the predictor.
2. Re-calibrate the abstention quantile on a held-out split that has the policy
   active.
3. Compute per-step scores on the *final, post-replacement* trace, not the
   original.
4. Do not let the abstention score "see" the original trace's scores — that
   would create a leak between the prediction and calibration sets.

If we want stronger guarantees (e.g. conditional on "regeneration triggered"),
we would need stratified CP with the trigger as the stratification variable,
which is a known extension and not blocked by any of the above.

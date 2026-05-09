# T1.4 — Earliest-bad-step re-roll with Five-Whys / CPM justification

> **Verification date**: 2026-05-08
> **Verifier**: Claude Opus 4.7 (1M context), single-model pass.
>   Cross-model verification (per CLAUDE.md cross_model_verification block, mode = `all`)
>   should be triggered by master_orchestrator at G1 / G4 review, since this is a
>   PROCEED-on-marginal-Δ candidate (HGJ predicts +0 to +5pp lift).
> **Proposal**: Replace K=4 re-roll target in Pilots C/K/L from "worst step"
>   (current method, lp_min / prm_min argmin) to **earliest divergent step**
>   (first step where the running PRM / lp / sc-derived score crosses below
>   a calibrated threshold). Justified on Five-Whys (Toyota root-cause; Ohno)
>   and CPM critical-path-method grounds: the *earliest* threshold violation
>   is the upstream cause; the *worst* is often the downstream symptom of a
>   prior cascade.
> **Internal anchors**:
>   - `/home/nvidia/future/METHOD_AND_RESULTS.md` §2.8 — Pilot C/K/L null result.
>   - `/home/nvidia/future/HGJ_experiment_ideas.md` Idea 1.1 — paper value HIGH, 4-6 GPU hr.
>   - `/home/nvidia/future/HGJ_review_feedback.md` — empirical reviewer flags effort
>     under-estimate (1-2 hr → 4-6 hr), theoretical reviewer flags trigger-calibration
>     issue (per-step abort threshold ≠ re-roll trigger; coverage interpretation
>     breaks unless separately calibrated).
>   - `/home/nvidia/future/literature/02_cot_reasoning_step_verification.md` §B (PRMs),
>     §E8 (Step-DPO), §F (recent step-level adaptive computation), §G (HIGH-threat).

---

## V1 — Prior art deep search

### V1.1 Queries executed

1. "first incorrect step" + re-roll + LLM (2024-26)
2. earliest-error + step-level + reasoning + correction + inference time
3. PARC premise-augmented chains + first-error
4. ProcessBench-aware step-level error correction + replan (2025)
5. "good prefix" + process reward + first incorrect step + truncate
6. Step-level rejection sampling / tree search + resample first low-confidence step
7. "Importance of Starting Right" + multi-step math + prefix
8. Causal intervention + Pearl do-calculus + first error step + cascade
9. "earliest divergent step" / "minimal intervention point" + reasoning chain + re-sample
10. Five-Whys / Toyota root-cause + reasoning chain + critical-path-method
11. PedCoT (IJCAI 2024) — pedagogical mistake-finding
12. "When the Chain Breaks" (2603.21286) — interactive diagnosis

### V1.2 Closest prior art (read in detail)

**Already in our litreview** (02_cot_reasoning_step_verification.md):
- **B4 ProcessBench (Zheng+ 2412.06559, ACL 2025)** — canonical *earliest erroneous step localization* benchmark. Existing PRMs collapse on Olympiad subsets. **Sets the evaluation target if T1.4 reports a localization metric**.
- **B7 Lessons of Developing PRMs (Zhang+ 2501.07301)** — best-current-practice PRM, MC-bias aware. Score source for our trigger.
- **D5 UHeads (2511.06209)** — tiny <10M-param uncertainty heads on frozen LLM internals; step-level UQ. Direct competitor for the *trigger signal*, not the *intervention*.
- **E8 Step-DPO (Lai+ 2406.18629)** — DPO at *first-incorrect-step granularity*, but **training-side**. Justifies first-error granularity philosophically; not an inference baseline.
- **F1 DEER (2504.15895)** — step-level early-exit on confidence at "switch points". Heuristic threshold; symmetric in spirit but action is *exit*, not *re-roll*.
- **F5 ConfSpec (2602.18447)** — confidence-gated step-level speculative reasoning. Uncalibrated.
- **F9 VG-Search (2505.11730)** — adaptive verification granularity. Heuristic.

**New finds from V1 search** (not yet in our 30-paper bibliography):

- **PARC — Premise-Augmented Reasoning Chains (Mukherjee+ 2502.02362, ICML 2025)**.
  Restructures linear CoT into a DAG of premise links; introduces *accumulation
  errors* (locally correct, globally wrong because premise was wrong). Step-by-step
  verification under premises improves error-id accuracy by **6-16 pp absolute**.
  Releases PERL dataset. **Most directly aligned with our T1.4 motivation**: the
  HGJ example "step 2 wrong substitution → step 4 stuck (worst)" *is* an
  accumulation error in PARC's taxonomy. PARC localizes the root step; T1.4
  *intervenes* on it. **PARC ≈ trigger; T1.4 ≈ trigger + re-roll action**.

- **PedCoT (Peng+ 2405.06705, IJCAI 2024)** — pedagogical chain-of-thought for
  mistake finding. Two-stage interaction (solve → critique-step-by-step) with
  Bloom-cognitive-model prompts. Evaluated on PRM800K + BIG-Bench Mistake
  step labels. Zero-shot mistake-locator. **Trigger-side competitor**.

- **"When the Chain Breaks" (2603.21286, Mar 2026)** — ReasonDiag, an
  interactive visualization for diagnosing CoT errors via fact-check + symbolic
  validation. Arc diagram for error-propagation. **Tooling, not method**;
  reinforces "earliest-step + propagation" framing.

- **Save the Good Prefix / VPPO (2601.18984, Jan 2026)** — uses PRMs **only as
  first-error detectors** in RL; rewards the prefix up to the first incorrect
  step. *Closest training-side analog* to T1.4's inference-side intervention.
  Their hypothesis is identical: "first incorrect step is the
  causal locus of the error". **Strong external validation** that the
  earliest-bad-step framing is becoming standard in the field, but operates at
  *training-time on RL rewards*, not at *inference-time on re-roll branching*.

- **Truncated Step-Level Sampling (Slate, 2602.23440, 2026)** — step-level
  advantage estimation via *truncated exploration*: isolate variation to a
  single action with provable variance guarantees. **Theoretical kin** but again
  RL-side.

- **"Well Begun, Half Done" / Beginning Lock-in Effect (2512.15274, Dec 2025)**
  — empirically shows early-stage thoughts **substantially constrain** subsequent
  trajectory. Direct empirical support for our cascade-source intuition.

- **First-Step Advantage / QuestCoT (2311.07945)** — guiding the *first* step
  matters most; +24 pt on GSM8K-7B. Same intuition, applied to step 1 only,
  not arbitrary "earliest threshold violation".

- **Less-is-More / Minimal Test-Time Intervention (2510.13940)** — high-entropy
  tokens flagged as "critical steps"; selective intervention there. **Closest
  inference-time analog**: "earliest divergent" ≈ "first high-entropy token".
  Our T1.4 differs in (i) score is a calibrated trigger, not raw entropy;
  (ii) action is a K=4 re-roll, not a token-stabilization edit.

- **Stepwise Correction (STEPCO, OpenReview 2025, YAhTj2VgBw)** — iterative
  verify-then-revise; 1 iteration beats Best-of-10 by +0.6 avg. Verifier-side,
  full-trace-revision granularity. Less close to T1.4 than the above.

- **S3c-Math (2409.01524)** — spontaneous step-level self-correction. Training-side.

### V1.3 Novelty pocket for T1.4

After V1, the **clean novelty claim** for T1.4 is:

> "First inference-time, **CP-calibrated** re-roll trigger that targets the
> *earliest* threshold violation (not the worst step), with the trigger
> threshold inheriting trajectory-level coverage guarantees from CoT-CP."

Key contrast vectors:

| Axis | T1.4 (earliest-bad-step re-roll) | Closest prior |
|---|---|---|
| Granularity | step-level | Step-DPO, PARC, ProcessBench, VPPO |
| Action | K=4 re-roll branch from earliest violation | DEER (exit), VPPO (RL reward shape), MTI (token stabilize), STEPCO (full revise) |
| Trigger | calibrated CP threshold (per-step or trajectory) | DEER heuristic conf, MTI raw entropy, PRM hard threshold |
| Time | inference | Step-DPO, VPPO, PRIME, AlphaMath = training; PARC, PedCoT = analysis only |
| Guarantee | inherits trajectory-CP coverage 1−α−1/(n+ +1) | none of the above |

**The novelty is not the "first error" framing** (that exists in PARC, Step-DPO,
ProcessBench, VPPO, "When the Chain Breaks") **— it is the combination of
earliest-violation-targeting + calibrated threshold + inference-time K=4 re-roll +
coverage proof**. Each piece exists individually; the package does not.

---

## V2 — Academic value

### V2.1 Theorem-grade contribution?

The HGJ-feedback theoretical reviewer suggested three formal-framing options:

1. **Causal (Pearl SCM)**: "Earliest bad step ≈ minimal intervention point in the
   trajectory SCM". The cleanest formalization: model the trace as a sequence of
   structural equations $S_t = f_t(S_{<t}, U_t)$ with exogenous noise $U_t$.
   *Worst* step = step with largest reconstruction loss; *earliest* step = first
   $t$ where $do(S_t \leftarrow s'_t)$ for resampled $s'_t$ would change all
   downstream steps in distribution. The *minimal* intervention property is
   topological (earliest in the partial order of premise dependencies as in
   PARC), which is genuinely a *theorem-shaped* claim.

2. **MDP**: depth-D = finite-horizon MDP; cite MCTS regret bound and argue that
   re-roll at earliest violation matches a UCB-1 expansion at the *root of the
   subtree where uncertainty first crosses threshold*. Less clean — re-roll is
   not a tree-search policy in the usual sense.

3. **CP**: re-roll breaks exchangeability of the calibration set with the test
   trajectory (the test trajectory is *no longer i.i.d.* from the original
   distribution after re-roll). Requires conditional weighted-CP wrapping (à la
   Theorem 3 / Tibshirani-Foygel-Barber 2019). **Necessary plumbing** if we
   want to keep the coverage guarantee end-to-end.

**Honest assessment**: option 1 (Pearl-causal) is the *only* one that makes T1.4
read as a theorem contribution rather than an ablation. To write that theorem
honestly we need:

- A formal definition of "trajectory SCM" with premise dependencies (PARC
  already supplies the DAG; we'd cite-and-extend).
- A propostion: *earliest threshold violation in topological order = minimal
  intervention point that decouples downstream cascade in distribution.*
  Provable under a faithfulness assumption on the score $S_t$.
- A second proposition: under (i) score-monotonicity + (ii) per-step CP
  calibration, the earliest-violation re-roll is **at least weakly Pareto**
  over worst-step re-roll in selective accuracy.

That second proposition is the **paper-worthy** piece — it would be a small
formal result that **explains why our Pilot C/K/L worst-step re-roll fails**
(the worst step is dominated in posterior reconstruction-loss by upstream
errors, so re-rolling it does not decouple the cascade), and **predicts** when
earliest-violation should beat it (the gap is monotone in the cascade depth,
i.e., in the number of downstream steps after the first violation).

### V2.2 Pearl-causal framing — paper-worthy?

**Yes, with caveats**. The framing is not novel as a *concept* (Pearl-causal
LLMs is its own subfield, e.g., 2407.08029, 2410.16676, CLadder-NeurIPS-2023,
the Awesome-Causal-LLM list). But the *application* — "earliest threshold
violation = minimal-intervention point in a trajectory SCM" — is, to our V1
search, **not yet published**. PARC supplies the DAG; VPPO supplies the
training-side use of "good prefix"; the inference-side causal-intervention
framing for K=4 re-roll appears unclaimed.

A *referee* will accept this as a contribution **only if the theorem is
non-trivial and the empirical ablation supports it**. The theorem above, if
clean, is non-trivial. Empirical support requires earliest-violation to beat
worst-step *or* at least exhibit the predicted cascade-depth monotonicity even
when both are flat (which would already be a paper-valuable diagnostic — the
HGJ "negative result has paper value" framing).

### V2.3 Comparison to Step-DPO

Step-DPO uses first-incorrect-step granularity for *training* (DPO loss). The
similarity is **superficial**: Step-DPO needs ground-truth labels of the
first-incorrect-step (collected via PRM800K-style annotation or MC rollouts);
T1.4 detects the first-violation **online from a calibrated score** with no
ground-truth labels at inference. The two are **complementary** — a Step-DPO-
trained model that produces a trajectory whose first-violation re-roll
(T1.4) targets the same step the trainer optimized would be the strongest
end-to-end pipeline.

**For the paper**, Step-DPO is a §Related-Work cite, not a competitor.

### V2.4 Verdict on academic value

- **Standalone ablation**: medium value; reads as "different worst-step formula
  in K=4 majority". Reviewer-magnet only if results are decisive.
- **+ Pearl-causal theorem**: HIGH value; reads as a small theorem +
  empirical-confirmation paper that *explains* the Pilot C/K/L null and
  predicts when step-level branching helps. This is the framing the paper
  should use.
- **+ Empirical results**: depends on outcome. Even null results have value
  via the predicted cascade-depth diagnostic.

---

## V3 — Feasibility & performance

### V3.1 Cost

- **HGJ estimate**: 1-2 GPU hr.
- **Empirical-reviewer correction (HGJ_review_feedback.md)**: 4-6 GPU hr.
- **Implementation-planner notes**: NEW file `experiments/src/SX_earliest_bad_step.py`,
  reuses `SX_step_rejection.py` (boundary detection + K=4 sampling) and
  `SX_per_step_cp.py` (per-step quantile). 3-5 min wall-clock per cell × 12
  cells = ~1 GPU hr *raw compute*; multi-cell evaluation + bootstrap CI
  inflates to 4-6 hr.

**Verifier estimate**: 4-6 GPU hr is the right number. Risk of overrun:

1. Trigger-threshold calibration is a separate sweep (HGJ-feedback flag #2):
   per-step CP threshold reuse from §V is wrong because operating goal (abort vs
   re-roll-trigger) differs. **+1-2 GPU hr** for separate trigger calibration on
   a held-out slice. Total: **5-8 GPU hr realistic**.
2. If we add the cascade-depth diagnostic figure (V2.4 prediction), one more
   sweep: **+1 GPU hr**. Total: **6-9 GPU hr**.

### V3.2 Predicted lift

HGJ predicts **"+0 to +2-5pp"**. Verifier independent prediction:

- **Lower bound (0pp)**: if the cascade-depth distribution in MATH-500 is
  narrow (i.e., earliest violation ≈ worst step on most traces), then
  earliest-violation collapses to worst-step and we recover Pilot C/K/L.
- **Upper bound (+2-5pp)**: if cascade-depth ≥ 2 on a non-trivial fraction
  (HGJ informal estimate ~30% of error traces have step-2-causes-step-4
  pattern), and re-rolling the earliest step has *at least* the same
  recoverability as worst-step (no reason to expect worse — same K=4, same
  prefix-conditional sampling), then we get a strict improvement on the
  cascade subset, weighted into a +2-5pp overall lift.
- **Realistic median**: **+1-2pp**. Tight CI overlap with worst-step is
  expected. The negative result is paper-valuable per HGJ framing.

### V3.3 Risk factors

1. **Trigger-calibration confound** (HGJ_review_feedback.md #2): per-step CP
   threshold's coverage interpretation breaks when reused as a re-roll
   trigger. **Mitigation**: separately calibrate trigger threshold using a
   held-out re-roll outcome label (binary: did the re-roll improve? from
   majority-of-K=4 voting). **Required**, not optional.

2. **Re-roll exchangeability break** (V2.1 option 3): re-rolling the *earliest*
   step changes the prefix more aggressively than re-rolling the worst step
   (which is usually deeper in the trace). The post-re-roll distribution
   diverges further from calibration. **Mitigation**: weighted-CP wrapper as
   in our Theorem 3, or honest disclaimer + empirical coverage check.

3. **Threshold-too-permissive failure mode**: if threshold is set too high,
   *every* step triggers re-roll → re-roll target = step 1 always → degenerates
   to "best of K=4 from-scratch" which is just SC@4. Sanity check needed.

4. **Threshold-too-strict failure mode**: re-roll never triggers → degenerates
   to greedy. Sanity check needed.

5. **K=4 ceiling**: HGJ Tier-3 (parallel branching from original) suggests a
   "step branching ceiling". If true, no choice of trigger-step beats
   trajectory CP. **The most likely outcome**.

### V3.4 Reuse / infrastructure

Excellent. From `HGJ_review_feedback.md` implementation-planner section:

- Reuse `SX_step_rejection.py` (boundary detection, K=4 sampling).
- Reuse `SX_per_step_cp.py` (per-step quantile machinery).
- Reuse `gap_check_sc_ood_weighted.py` (PMF baseline if we want the OOD
  trigger-coverage figure).
- Reuse E1/E2 traces (already on disk).
- Reuse `robust_eval.py` (extractor + bootstrap CI).
- Sanity tests (alpha=0 → kept_acc == vanilla; T_max=1 → trajectory CP match)
  are inherited.

**Infrastructure risk**: LOW.

---

## V4 — Incremental vs structural

### V4.1 Incremental case

T1.4 = swap `argmin(s_t)` for `argmin{t : s_t < hat{q}}` in the existing
K=4 worst-step re-roll loop. One line of pseudocode change. The HGJ "Idea 1.1"
framing — "different worst-step formula" — is exactly this read. From a
reviewer-skeptic angle: this is **a different argmin in a 3-line resampling
loop, evaluated on the same datasets, with the same K=4 majority vote, plus a
trigger-threshold calibration**. Plausible to read as a single ablation row.

### V4.2 Structural case

T1.4 = the **first inference-time procedure** in the literature (per V1) that
combines:
- step-level intervention (intervene, not just detect)
- earliest-error targeting (causal-locus, not symptom-locus)
- calibrated trigger threshold (CP, not heuristic)
- K=4 re-roll action (re-sample, not exit / not stabilize / not RL-reward)
- inheriting trajectory-level coverage guarantee

…wrapped in a Pearl-causal "minimal intervention point" theorem that explains
the Pilot C/K/L null result and predicts cascade-depth-monotonic lift. This is
**a small but coherent theoretical contribution + an empirical confirmation +
a diagnostic figure**.

### V4.3 Honest verdict

**T1.4 alone is incremental.** A K=4 ablation row, +1pp expected, will not
move a TMLR-let-alone-NeurIPS bar.

**T1.4 + Pearl-causal-minimal-intervention theorem + cascade-depth diagnostic
figure** is structural at the *paper-section* level (a §4 subsection or §5
diagnostic sub-experiment), not at the *paper-novelty* level. The paper's
main contribution remains CoT-CP / trajectory-level CP coverage; T1.4 is a
**supporting structural piece** that:

1. Gives a formal explanation for why step-level branching null-results
   (Pilot C/K/L) happen — the "step branching ceiling" hypothesis becomes a
   provable corollary under faithfulness, instead of an empirical tautology.
2. Provides a positive-or-negative-still-valuable result: if earliest beats
   worst, the theorem is confirmed; if not, the cascade-depth-monotonicity
   diagnostic still distinguishes "branching is fundamentally weak" from
   "branching needs the right step".
3. Reviewer-armor: the obvious "why worst step?" objection has a calibrated
   answer with a citation to PARC + Step-DPO + VPPO + Five-Whys / CPM.

**Bottom line**: incremental as an ablation; structural as a §4-or-§5
sub-contribution if the Pearl-causal theorem is written. Without the theorem,
**skip** in favor of higher-priority work (Local CP 2.1, Qwen3-32B 3.1).

---

## V5 — Reviewer objection + response

### V5.1 Anticipated objections

**O1 (Reviewer A — empirical skeptic)**:
> "This is a different argmin inside the same K=4 loop. You report
> (Pilot C, K, L) ≈ +0-2pp; T1.4 will be in the same noise band. Why is
> this a contribution?"

**Response**: T1.4's contribution is *not* the empirical Δ alone — it is the
formal characterization of when step-level branching can / cannot help, given
by the Pearl-causal "minimal intervention point" theorem (V2.1 option 1). The
empirical result either confirms the theorem (earliest > worst on cascade-deep
traces) or supplies the negative-result evidence that even the optimally-
targeted step-level intervention does not beat trajectory CP — both are
paper-valuable. The Pilot C/K/L results, by contrast, leave open whether
"better worst-step targeting" was the missing ingredient; T1.4 closes that gap.

**O2 (Reviewer B — theoretical reviewer)**:
> "Re-rolling at the earliest step changes the trace more than re-rolling at
> the worst step, so calibration drifts. Your coverage proof breaks."

**Response**: Acknowledged. We wrap T1.4 in conditional weighted-CP (Tibshirani
2019, Foygel-Barber 2023) the same way we wrap Theorem 3's PMF-shift case.
The coverage statement becomes 1−α−1/(n+ +1) − slack, with slack bounded by the
DKW-rate of the PMF-of-trigger-step. We additionally calibrate the trigger
threshold separately (held-out re-roll-improvement label) so that the operating
goal matches the calibration target.

**O3 (Reviewer C — methodology purist)**:
> "Five-Whys / CPM is engineering folklore. Cite Pearl, not Toyota."

**Response**: We move Five-Whys / CPM to a single §1 paragraph as
*motivation-by-analogy*; the formal grounding is Pearl's SCM and PARC's premise
DAG. Five-Whys gives the reader a fast intuition; the theorem gives the
guarantee.

**O4 (Reviewer D — adversarial novelty-checker)**:
> "PARC, Step-DPO, VPPO, ProcessBench, 'When the Chain Breaks' all use
> first-error / earliest-incorrect-step framing. What is novel?"

**Response**: V1.3 novelty pocket: *first inference-time CP-calibrated K=4
re-roll trigger* targeting the earliest threshold-violation. PARC = analysis;
Step-DPO / VPPO = training; ProcessBench = benchmark; "When the Chain
Breaks" = visualization. None propose nor implement an inference-time re-roll
intervention with calibrated coverage guarantees on the post-re-roll trace.

**O5 (Reviewer E — sample-size skeptic)**:
> "Your effect-size CIs overlap with worst-step. Bootstrap n=500 is fine but
> the cascade-depth-≥2 subset is small."

**Response**: We pre-register the cascade-depth-stratified analysis (depth=1
trivial subset vs depth≥2 cascade subset) and report stratum-conditional CIs.
The theorem's prediction is *zero gap on depth=1 traces* (correct expectation)
and *positive gap on depth≥2* — exactly where the cascade subset's bootstrap CI
should narrow.

### V5.2 Worst-case rebuttal scenario

If empirically earliest = worst (degenerate case where cascade-depth distribution
is heavily mass on 1), the paper-saving move is:

> "Pilot C/K/L + T1.4 jointly establish that under our trigger-score family
> (lp_min, prm_min, sc_top1), step-level branching is at its ceiling. This is
> the negative result that motivates trajectory-level CP as the right
> primitive — exactly the paper's main thesis."

This is intellectually honest and converts the null into a load-bearing data
point.

---

## V6 — Final verdict

### V6.1 GO / NO-GO matrix

| Path | Cost | Paper value | Verdict |
|---|---|---|---|
| **A. T1.4 ablation row only** (no theorem, no diagnostic) | 4-6 GPU hr | LOW (incremental row) | **NO-GO** alone — only worth doing if it's free riding on Path B |
| **B. T1.4 + Pearl-causal theorem + cascade-depth diagnostic** | 6-9 GPU hr + 1-2 days theorem-writing | MEDIUM-HIGH (§4 sub-contribution + reviewer-armor) | **GO** — *if* the theorem is provably clean (V2.1 option 1) and TMLR is the target |
| **C. T1.4 + theorem + Local CP (T2.1) + Qwen3-32B (T3.1)** | aggregate 13-23 hr | HIGH | **GO** — Path B as part of the bundled Week-1 plan in HGJ_review_feedback.md |
| **D. Skip T1.4 entirely** | 0 | (negative — ceding reviewer-armor for "why worst step?" objection) | **NO-GO** — leaves the most-obvious-objection unanswered |

### V6.2 Recommended verdict

**GO on Path B (= Path C with bundling).**

Rationale:
1. The HGJ-review-feedback Week-1 plan already budgets 4-6 hr for T1.4 alongside
   3-4 hr Qwen3-32B and 6-10 hr Local CP. The bundled execution amortizes setup.
2. The Pearl-causal "minimal intervention point" theorem is the *only* angle
   that makes T1.4 paper-grade rather than ablation-row; we should write the
   theorem on a 1-2-day pass *before* spending 4-6 GPU hr to ensure the
   empirical result has a principled home.
3. The cascade-depth-stratified diagnostic figure is the safety net: even a
   null result yields a load-bearing figure that explains the Pilot C/K/L
   null and reinforces the trajectory-CP-is-the-right-primitive thesis.
4. Reviewer-armor against the "why worst step?" objection is non-negotiable for
   any peer-review venue.

### V6.3 Required pre-conditions

Before executing the GPU sweep:

1. ✅ Trigger threshold separately calibrated (held-out re-roll-improvement
   label), per HGJ_review_feedback.md flag #2.
2. ✅ Pearl-causal "minimal intervention point" theorem drafted (V2.1 option 1)
   with the cascade-depth-monotonicity prediction explicit, even if proof
   only sketched.
3. ✅ Sanity tests pre-registered: alpha=0 → kept_acc == vanilla;
   T_max=1 → trajectory-CP match; threshold-too-permissive →
   step-1-always; threshold-too-strict → never-triggers.
4. ✅ Cascade-depth stratification specified (depth=1 vs depth≥2 buckets)
   *before* the sweep, to avoid post-hoc subset selection.

### V6.4 Confidence calibration (per CLAUDE.md researcher-persona requirement)

- **High confidence** (>0.85):
  - V1 prior-art coverage; the novelty pocket is real.
  - Path A (ablation alone) is not paper-grade.
  - HGJ effort estimate (1-2 hr) is too optimistic; 4-6 hr realistic, 5-8
    with separate trigger calibration.

- **Medium confidence** (0.55-0.85):
  - The Pearl-causal theorem is "clean enough" to draft in 1-2 days. *I have
    not actually drafted the proof*; this is a sketch-grade prediction.
  - Earliest will beat worst by +1-2pp median (uncertainty: cascade-depth
    distribution in MATH-500 is unknown to me without fetching the trace
    statistics).

- **Low confidence** (<0.55):
  - Whether the theorem alone makes T1.4 venue-worthy *as a §4 subsection*
    vs a §6-related-work footnote depends on TMLR/NeurIPS taste; reviewer
    Path A pushback ("incremental ablation") is plausible.
  - Whether re-roll-coverage slack from the weighted-CP wrapper is acceptable
    empirically; we have no data on this yet.

### V6.5 Cross-model verification trigger

Per CLAUDE.md `cross_model_verification.mode = all` and the per-agent trigger
table, this verification should be re-checked by the verifier model
(`openai/openai/gpt-5.5` or fallback `gcp/google/gemini-3.1-pro-preview`)
because:

- Master-orchestrator G1 / G4 gate: **PROCEED on marginal Δ** is the predicted
  outcome; this is exactly the trigger condition.
- Hypothesis-curator: T1.4 re-uses the "step-level intervention" axis that
  Pilot C/K/L pivoted away from after their null result; Idea 1.1 is a
  re-engagement on a previously-pivoted axis. Per the protocol, this is a
  cross-model trigger.

Any disagreement with the verifier model goes into a `### Cross-Model
Verification Results` section appended to **this** file, not silently
overwritten.

---

## Appendix A — One-line summary

> **T1.4 GO conditional on writing the Pearl-causal "minimal intervention
> point" theorem first; the ablation alone is not paper-grade, but the
> theorem + cascade-depth diagnostic + empirical sweep together earn a §4
> sub-contribution and provide reviewer-armor against the otherwise-fatal
> 'why worst step?' objection. Bundle with Local CP (T2.1) and
> Qwen3-32B-thinking (T3.1) for Week-1 execution per HGJ_review_feedback.md.**

## Appendix B — New citations to add to bibliography

(Not yet in our 30-paper §E8 / B-tier list as of 2026-05-08:)

1. PARC — Mukherjee et al., 2502.02362, ICML 2025.
2. PedCoT — Peng et al., 2405.06705, IJCAI 2024.
3. When the Chain Breaks (ReasonDiag) — 2603.21286, Mar 2026.
4. Save the Good Prefix (VPPO) — 2601.18984, Jan 2026.
5. Truncated Step-Level Sampling (Slate) — 2602.23440, 2026.
6. Well Begun Half Done / BLE — 2512.15274, Dec 2025.
7. First-Step Advantage / QuestCoT — 2311.07945.
8. Less is More / Minimal Test-Time Intervention — 2510.13940.

These eight reinforce both the "earliest matters" intuition (V2.3, V4.2) and
the "first error step" framing (V1.2). They also collectively establish that
T1.4's *combination* (inference + calibrated + re-roll + coverage) is the
unclaimed pocket.

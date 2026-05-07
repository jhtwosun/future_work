# Wave 2 — Agent H (structured-rewrite angle)

**Mandate.** Pilots C/K/L closed the door on temperature-only K-resample at the worst step (lp_min, PRM-min, and "Hmm wait" cue all delivered 0 to +2pt — within noise). The Korean observations file §6 is explicit: *"the bottleneck is not the score, it is the intervention class itself."* This brief proposes five **structured rewrite** interventions that replace stochastic re-rolls with explicit, instruction-driven transformations of the worst step, each engineered to inject *new information* that temperature cannot produce while preserving trajectory-level CP compatibility.

**Framing.** K-resample at T=0.7 stays inside the same conditional p(step | prefix); raising T samples surface variants of the same wrong reasoning. A structured rewrite *shifts the conditioning* — different tools, different role, different language, different granularity, or an external oracle — and draws from a fundamentally different distribution.

---

## Method 1 — `method_switch_rewrite`

**Template.**
> "Below is a partial solution. Step *i* uses **{detected_method}**. Re-derive **only step *i*** using a different method from this menu: {geometry, calculus, combinatorics, induction, generating-functions, complex-numbers, vectors, modular-arithmetic}. State your choice on the first line, then write the new step. Do not change earlier steps."

**Granularity.** Per-step (worst step by lp_min or PRM-min).

**Trigger.** Worst-step PRM-min < α-calibrated cutoff *and* a 1-shot method-tagger LM returns a confident label. If method tag is "unknown", fall through to Method 4.

**Aggregation.** Generate M=3 method-switched rewrites + 1 same-method baseline; greedy-complete each to final answer; PRM rescore the four trajectories; pick argmax. **Replace** original only if rewrite trajectory's PRM beats original by margin τ.

**Compute.** ≈ 3–5× greedy (tag + 3 rewrites + 4 completions + PRM rescores).

**Why this beats K-resample.** K-resample leaves the prefix's induced method bias untouched: if step 4 commits to "let x = …", every resample also substitutes. Method-switch *forces* a regime change by writing the method choice into the prompt. New samples come from p(step | prefix, method=geometry) — a different conditional, not a hotter one. The information injected is the inductive bias of an alternative branch of mathematics.

**Implementation.** (a) Method tagger LM call returning a single label. (b) Sample 3 rewrites at T=0.4 (instruction provides diversity, not temperature). (c) Greedy-complete from each rewrite. (d) PRM-min over each full trajectory. (e) Trajectory-CP threshold on the selected one.

**Risks.** Over-correction when worst-step is a PRM false positive. Method refusal: "use geometry" on pure number theory produces nonsense — guard with a tagger applicability check. Prompt sensitivity: menu phrasing matters; pilot ≥3 wordings.

**Coverage.** Preserved. The rewriter is a deterministic-given-seed function of (problem, prefix); trajectory exchangeability holds. Calibration must be re-run under the new pipeline.

---

## Method 2 — `verify_then_correct`

**Template.** Stage A:
> "Step *i*: <step>. In one line, either confirm it (e.g. plug in numbers, check units, derive again) or point to the wrong substring and explain the error in one sentence."

Stage B (only if A returns "wrong"):
> "You identified <error>. Rewrite step *i* correcting it. Output only the new step."

**Granularity.** Per-step, with Stage A free to verify equation-by-equation inside the step.

**Trigger.** Worst-step PRM-min < cutoff. Stage B fires only on "wrong" — early exit otherwise.

**Aggregation.** Replace if Stage B fires *and* corrected trajectory PRM ≥ original; revert otherwise. No averaging — verification is binary.

**Compute.** ≈ 1.3× greedy average (Stage A cheap; Stage B fires on ~30–50% of triggers per Pilot K base rate). Worst case 2.5×.

**Why this beats K-resample.** K-resample assumes perturbation surfaces the right answer. Verification forces the model into a *different cognitive role* — a critic with a binary commitment — breaking the auto-regressive momentum that locked the wrong step in. Self-Refine and Reflexion show critique-as-prompt produces qualitatively different outputs than re-sampling.

**Implementation.** Two LM calls per triggered step. Constrain Stage A first token to {"correct", "wrong"} via logit bias; on "wrong" extract reason, feed into Stage B, greedy-complete.

**Risks.** Sycophantic refusal — model always says "correct". Guard with periodic adversarial-framing Stage A and majority vote. Prompt sensitivity moderate; verification prompts are well-studied and stable.

**Coverage.** Preserved (same argument as Method 1).

---

## Method 3 — `substep_decomposition`

**Template.**
> "Step *i* below is dense. Rewrite it as **exactly 3 sub-steps**, each one short sentence with its equation. Sub-step 3 must end at the same intermediate result as the original. Do not skip arithmetic."

**Granularity.** Per-step, expanded into 3 micro-steps; trajectory grows from n to n+2.

**Trigger.** Worst-step PRM-min < cutoff *and* worst-step token length > median × 1.5 — only decompose dense steps where there is room to expose hidden computation.

**Aggregation.** **Replace** worst step with the 3 sub-steps inline. Re-score with per-step PRM. Accept if min(new sub-step PRM) ≥ original step PRM; else revert.

**Compute.** ≈ 1.5× greedy (one rewrite + PRM over 2 extra steps).

**Why this beats K-resample.** Decomposition is a *measurement intervention*, not a generation intervention — it exposes hidden intermediate computations to the PRM grader. A dense step where a sign error hides mid-line gets a single low score under the original segmentation; after decomposition, the PRM sees the actual error. K-resample cannot do this because it never changes the granularity at which the score operates. Bonus: writing out 3 substeps forces the model to externalise reasoning that was implicit, sometimes self-fixing via constraint-of-thought.

**Implementation.** (a) Detect long dense steps (token count + has-equation regex). (b) Single LM call, stop tokens enforce exactly 3 newline-separated sentences. (c) Splice into trajectory list. (d) Re-PRM. (e) Continue greedy from last sub-step.

**Risks.** Decomposition can hallucinate intermediates that weren't actually computed (post-hoc reverse-engineered justification). Mitigate by also asking for "value at end of sub-step 2" and checking it matches original's intermediate. Prompt sensitivity low.

**Coverage.** Preserved, with one caveat: trajectory-length distribution shifts, so length-sensitive scores (lp_mean, lp_tok) need re-normalisation. lp_min and PRM-min are length-robust and slot in unchanged.

---

## Method 4 — `adversarial_critique_then_redo`

**Template.** Stage A:
> "Step *i* contains an error. Find it. Name the wrong operation, state the correct one, explain why in one sentence. If you genuinely cannot find an error, output `NO_ERROR`."

Stage B:
> "Given the critique above, rewrite step *i* correctly. Same notation. Output only the new step."

**Granularity.** Per-step.

**Trigger.** Worst-step PRM-min < cutoff. Differs from Method 2 in *presupposing* an error — pushes the model into adversarial mode rather than neutral verifier. Use Method 4 when PRM signal is strong (bottom 10%); Method 2 when weak (bottom 25%).

**Aggregation.** NO_ERROR → keep original. Else replace if rewrite trajectory PRM ≥ original; revert otherwise.

**Compute.** ≈ 1.5–2× greedy (NO_ERROR escapes ~20%).

**Why this beats K-resample.** Adversarial framing exploits the known asymmetry that LLMs are better at *finding* errors than at avoiding them at generation time (cf. critic-vs-generator scaling literature). The presupposition pushes the model out of the generation-time prior into the critique-time prior, with different failure modes. K-resample stays in the generation prior. The rewrite produces information of a *different epistemic type* (a critique), which then conditions the redo.

**Implementation.** Same plumbing as Method 2 with adversarial system prompt and NO_ERROR escape token. Logit-bias slightly toward NO_ERROR to counter the framing's false-positive bias.

**Risks.** Catastrophic failure mode: hallucinated critique → "fix" turns correct step into wrong step. This is the dominant Pilot L mechanism (3/0 → 4/2 with cue). Guards: (a) NO_ERROR escape, (b) PRM-monotonicity revert, (c) optional second adversarial pass on the rewrite. Prompt sensitivity high — pilot ≥5 phrasings.

**Coverage.** Preserved.

---

## Method 5 — `formal_restate_and_check`

**Template.**
> "Restate step *i* purely as a SymPy expression. Output a single Python block that (1) defines symbols, (2) writes the equation as `expr = ...`, (3) calls `sp.simplify` or `sp.solve`, (4) prints the result. The output must equal the intermediate value claimed in step *i*. If the step cannot be expressed, output `NOT_FORMALISABLE`."

Then *execute* the SymPy and compare to the step's claimed value.

**Granularity.** Per-equation (within a step). One step may produce multiple snippets.

**Trigger.** Worst-step PRM-min < cutoff *and* step contains ≥1 equation (regex). Skip pure-narrative steps.

**Aggregation.** Three branches: (1) SymPy executes and matches → step verified, mark high-confidence, *no rewrite*. (2) SymPy disagrees → trigger Method 2 or 4 with the SymPy-derived value as a hint. (3) NOT_FORMALISABLE or syntax error → fall through to Method 1.

**Compute.** ≈ 1.2–2× greedy. SymPy execution is near-free; cost is the LM rewrite plus fallback when formalisation fails.

**Why this beats K-resample.** This is the strongest information injection of the five: an *external symbolic verifier* brings ground truth the model itself does not have. K-resample is purely model-internal; this method imports a separate computational system. The formal-language rewrite is the *bridge* enabling the import. It is the only one of the five that can definitively *prove* the worst step right or wrong, rather than re-sampling around it. For arithmetic-heavy MATH-500 steps, near-perfect; for steps depending on geometric intuition or unstated lemmas, NOT_FORMALISABLE rate will be high (~50–60% expected).

**Implementation.** (a) Per-step equation regex. (b) LM call with template + 2-shot. (c) Sandbox SymPy with 5s timeout. (d) `sp.simplify(lhs - rhs) == 0` for equality. (e) Branch.

**Risks.** Translation error — LM mis-symbolises and "verifies" a wrong formal version of a correct step (spurious mismatch). Mitigate with inverse-translation sanity check ("here is SymPy, write it back as math; does it match?"). Refusal rare, handled by NOT_FORMALISABLE. Prompt sensitivity low for arithmetic, moderate for symbolic.

**Coverage.** Preserved, with the cleanest CP story of the five: when SymPy verifies, the step is *deterministically* high-confidence and folds into a hybrid score (`prm_or_sympy_min`) that strictly dominates PRM-min on the formalisable subset.

---

## Cross-cutting analysis

**How each creates information K-resample cannot.** Method 1: changes *method-prior conditioning*. Method 2: forces *role switch* (verifier ≠ generator). Method 3: changes *granularity at which the score operates* (measurement, not generation, intervention). Method 4: exploits *generator-vs-critic asymmetry*. Method 5: imports an *external symbolic oracle*. None is temperature-reachable. K-resample explores variance within the same conditional distribution; these methods change the distribution.

**Catastrophic failure profile.** Method 4 most dangerous (hallucinated critiques) — needs strongest guards. Method 1 fails softly when the menu doesn't fit the problem (tagger prefilter mitigates). Method 5 fails by NOT_FORMALISABLE — clean fallback, not wrong answer. Methods 2 and 3 are safest: Method 2 has early-exit on "correct"; Method 3 only exposes existing reasoning to PRM and cannot inject false content.

**Coverage preservation across the suite.** All five preserve trajectory-level CP because each rewriter is a black-box function returning a final-answer trajectory; (problem, trajectory) exchangeability holds when the same pipeline runs on calibration and test. The single gotcha: length-dependent scores (lp_mean, lp_tok) need re-normalisation under Method 3. lp_min and PRM-min are length-robust.

**Recommended pilot order.** Method 5 first (cleanest, strongest signal, cheapest fallback); Method 3 next (low-risk measurement); then Method 1 (highest information-injection rewrite); then Method 2; then Method 4 last (highest variance, strongest guards). Reuse the Pilot K trace pool (200 MATH-500 traces with PRM-min worst-step labels) for head-to-head evaluation in roughly 4 hours of H100 time.

**Headline hypothesis.** If any of the five works, it will be Method 5 on the formalisable subset (~40% of MATH-500 steps), with expected lift +5 to +10pt over PRM-min trajectory CP — SymPy turns probability into proof. Methods 1 and 3 are next most promising; Methods 2 and 4 are likely net-zero unless paired with strong reversion guards. The crucial test: any of the five must beat Pilot K's net-zero baseline on the same trace pool to be worth scaling.

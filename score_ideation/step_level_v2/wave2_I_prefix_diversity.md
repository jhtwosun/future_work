# Wave 2 — Agent I (prefix-diversity angle)

## Framing

Pilots C/K/L revealed K-resampling at τ=0.7 collapses to near-identical answers. The cause is structural: the prefix `s_{1:t-1}` has already pruned `p(s_t | s_{1:t-1})` to a narrow mode; raising temperature only redistributes mass *inside* that mode, not across basins.

The remedy is to **manipulate the conditioning context** so the worst step is re-decoded under a *different* prefix distribution — converting low-information thermal noise into high-information semantic perturbation. Below: ten methods.

---

## 1. `prefix_truncation_regen`

**Mechanism.** At worst step `t*`, drop steps `t*-1..t*-k` (k=1,2,3). Re-decode from truncated prefix `s_{1:t*-k-1}` greedily; model regenerates the deleted intermediates plus the worst step.

**Trigger.** Per-step CP score (token logprob, entropy, or self-eval) below conformal threshold at `t*`.

**Compute.** ~1.2× greedy per perturbation; K=4 depths → ~4.8×.

**Why it beats K-resample.** K-resample re-rolls dice with same conditioning. Truncation *removes* the conditioning that locked the model into the bad basin — if `t*-1` contained a subtly wrong commitment, the model never sees it on the second pass.

**Implementation.** KV-cache slice: keep `KV[:, :end_token_of(t*-k-1)]`, discard rest. Resume decoding directly. No re-prefilling.

**CP coverage.** Calibration runs same trigger+truncation. Trigger is a function of the original trace, so the perturbed trajectory is deterministic-given-trigger; exchangeability holds. Test score: `agreement(answer_original, answer_truncated)`.

**Modality.** Strong for math/MCQ. Weaker for code (may delete function definitions) — gate with syntactic safety check.

---

## 2. `prefix_paraphrase`

**Mechanism.** Send `s_{1:t*-1}` through a lightweight paraphraser (or self-paraphrase via "rephrase preserving meaning"). Re-decode step `t*` from `s'_{1:t*-1}`.

**Trigger.** Same per-step CP threshold.

**Compute.** ~1× greedy paraphrase + ~0.3× regenerated suffix. K=4 → ~5×.

**Why it beats K-resample.** K-resample preserves exact tokens; paraphrase preserves *meaning while perturbing surface form*, breaking the n-gram-level lock that drives mode collapse. If the bad commitment was driven by token-level priming ("Let X = …" → mechanical X-substitution), paraphrasing loosens that grip.

**Implementation.** Two-pass: (a) prefill paraphrased prefix (full re-prefill, KV invalidated); (b) decode worst step.

**CP coverage.** Randomized `φ` applied symmetrically to calibration and test → exchangeability OK. Fix RNG seed per problem.

**Modality.** Math: safe. Code: dangerous (semantics undefined). MCQ: prefix often short, near-identity. Restrict to math/free-form.

---

## 3. `system_prompt_ensemble`

**Mechanism.** Re-decode the *entire* trace under K different system prompts:
- "Be careful and check each algebraic step."
- "Simplify aggressively before computing."
- "Use algebraic manipulation, not arithmetic."
- "Draw an analogy to a simpler problem first."

**Trigger.** Always-on (no per-step trigger), or triggered by an aggregate trace-level CP score.

**Compute.** K× full greedy. With K=4, 4× greedy. Identical to SC@4 in raw cost.

**Why it beats K-resample.** SC@4 averages over *sampling-noise*; prompt ensembles average over a *strategy distribution*. Different prompts induce different reasoning styles → different prefixes at every step → genuinely different basins. Diversity-through-controlled-distribution-shift, not thermal jitter.

**Implementation.** K parallel calls with different system messages. No KV surgery. Prefix-caching on the user message if engine supports `(system, user)` keying.

**CP coverage.** K-trace agreement (majority-vote margin) as conformity score. Standard CP exchangeability — perturbation is fixed function of input.

**Modality.** Universal: math, code, MCQ.

---

## 4. `persona_swap`

**Mechanism.** Swap the persona prefix in the system message: `{olympiad_coach, patient_teacher, software_engineer, skeptical_referee}`. K=4 personas, full re-decode each.

**Trigger.** Always-on.

**Compute.** 4× greedy.

**Why it beats K-resample.** Personas activate different *style manifolds*: olympiad coach reaches for inequality tricks, teacher for canonical methods, referee for counterexamples. Prefix steered toward qualitatively different reasoning families, not different samples within one.

**Implementation.** Same plumbing as method 3, persona-style prefixes. Combinable: persona × instruction = K_p × K_i variants.

**CP coverage.** Same argument. Subtlety: persona may bias accuracy uniformly — weight by validation accuracy, or use max-min spread instead of majority vote.

**Modality.** Best on open-ended reasoning. MCQ: pair with method 3.

---

## 5. `context_dropout`

**Mechanism.** Apply random token-level masking to `s_{1:t*-1}` (replace 10–20 % of tokens with a sentinel `[…]` or simply delete them), then re-decode step `t*`.

**Trigger.** Per-step CP score below threshold.

**Compute.** Each dropout pass requires re-prefilling the corrupted prefix (KV-cache invalidated by token edits). ~1.5× greedy per sample, K=4 → 6× greedy.

**Why it beats K-resample.** Forces model to reason under uncertainty about its own past statements, deleting the token-level cue that anchored the bad continuation. MC-Dropout applied at input layer.

**Implementation.** Bernoulli mask; preserve structural tokens (numbers, operators, identifiers), drop only function words/connectives. Span-dropping is more aggressive but riskier.

**CP coverage.** Fix RNG per problem; calibration sees same dropout distribution. Exchangeability OK.

**Modality.** Math: risky (dropping numbers changes problem). Code: near-fatal. Mask only NL tokens, never operands/identifiers.

---

## 6. `backwards_reasoning_check`

**Mechanism.** After producing the answer `a`, *prepend* `a` to the problem and ask the model to derive a chain of reasoning *that arrives at a*. Compare backward chain to forward chain at step `t*`.

**Trigger.** Always-on at trace end (this is a verifier, not a step-replacement).

**Compute.** 1× greedy backward pass = 2× total.

**Why it beats K-resample.** Backward pass conditions on the *answer* — maximally different prefix. Wrong-basin forward chains either fail to derive `a` or derive it via a path contradicting step `t*`. Resampling never injects answer-conditioning.

**Implementation.** Two prompts: forward(`x`) → `(s, a)`; backward(`x, a`) → `s_back`. Score = step-by-step agreement, or alignment of intermediate quantities.

**CP coverage.** Deterministic function of `(x, s, a)`. Calibrate threshold on held-out problems. Exact coverage.

**Modality.** Math/logic: ideal. Code: feasible (verify output → reconstruct algorithm). MCQ: degenerate.

---

## 7. `hypothesis_priming`

**Mechanism.** For K candidate answer-archetypes `{X_1, …, X_K}` (e.g., "the answer is a prime", "the answer involves √2", "the answer is < 10", "the answer is symbolic"), prepend a hint *before the problem*: "Hypothesis: the answer is `X_i`. Verify or refute." Decode K traces; the answer that the model *converges to despite different priming* is robust.

**Trigger.** Always-on, or triggered by low CP confidence.

**Compute.** K× greedy. K=4 → 4×.

**Why it beats K-resample.** Each hypothesis induces an adversarially-different conditioning context. Model must defend or rebut. If answer is robust, traces still converge to `a` despite wrong priming; if uncertain, traces *track* the priming — direct fragility signal.

**Implementation.** Generate hypotheses adaptively from a cheap first pass, or use fixed archetype templates. Score = convergence rate of primed traces.

**CP coverage.** Fixed templates → straightforward exchangeability. Per-problem generation → still deterministic given seeded RNG.

**Modality.** Math: rich hypothesis space. MCQ: prime each option as hypothesis. Code: prime with complexity class or algorithmic family.

---

## 8. `question_paraphrase`

**Mechanism.** Paraphrase the *problem statement* (not the reasoning) into K semantically equivalent phrasings. Re-decode each from scratch.

**Trigger.** Always-on.

**Compute.** K× greedy + paraphrase cost. ~4.5× for K=4.

**Why it beats K-resample.** Cleanest *symmetric* perturbation: varying input wording varies every downstream prefix. Wording-dependent answers signal unreliability that fixed-wording K-resample cannot detect.

**Implementation.** Pre-process paraphrase before decoding. No KV surgery.

**CP coverage.** Inference-time data augmentation. Calibrate on same paraphrase distribution. Coverage exact.

**Modality.** Universal. Code: paraphrase NL spec, keep code constraints fixed. MCQ: paraphrase stem, keep options fixed.

---

## 9. `inverse_step_substitution`

**Mechanism.** Identify the worst step `s_{t*}` and rewrite it as its semantic inverse (e.g., "X is large" → "X is small"; "case A holds" → "case A fails"). Continue decoding from the inverted step. If the final answer is unchanged → the worst step doesn't actually constrain the answer (which is a problem — model isn't using its own reasoning); if the final answer flips → the step is load-bearing and the original direction needs verification.

**Trigger.** Per-step CP score below threshold.

**Compute.** 1× regenerated suffix per inversion. K=1 inversion + K=2 random alternatives → 3× greedy.

**Why it beats K-resample.** K-resample explores nearby variations of the *same* direction. Inverse-step forces the opposite direction — counterfactual probing of step's load-bearingness. Causal intervention, not noise.

**Implementation.** Inverter prompt: "Rewrite to mean the logical opposite". Substitute, re-prefill `s_{1:t*-1} + s_t*_inverse`, decode suffix.

**CP coverage.** Label `(forward_answer, inverse_answer)` triples as load-bearing-consistent / load-bearing-contradictory / non-load-bearing → calibrate.

**Modality.** Math/logic: ideal. Code: flip conditional or swap comparator. MCQ: rarely applicable.

---

## 10. `curriculum_prefix`

**Mechanism.** Prepend a *solved related example* (`x_demo, s_demo, a_demo`) to the problem statement. Re-decode. Use K different curated demos for K traces.

**Trigger.** Always-on for low-confidence problems; otherwise skip.

**Compute.** K× greedy plus the demo tokens (typically +30 % per call). ~5× greedy for K=4.

**Why it beats K-resample.** Demos redirect the prefix's learned distribution toward the demo's reasoning template — K demos → K templates → genuinely different prefixes at every step. Especially strong when the default prefix is wrong because the problem is unfamiliar.

**Implementation.** Demo library indexed by problem-type detector (regex/classifier). Sample K demos per problem.

**CP coverage.** Fixed library + deterministic matching → `φ(x)` is fixed; coverage exact.

**Modality.** Universal — historically the strongest single intervention in few-shot literature. Code benefits especially.

---

## Cross-cutting observations

**Why prefix manipulation beats K-resample.** Temperature-`τ` sampling from `p(s | prefix)` is bounded by the entropy of that conditional; once the prefix has driven entropy low, no τ short of incoherence diversifies samples. Prefix manipulation replaces `p(s | prefix)` with `p(s | prefix')` — distributional shift, not sampling shift.

**Calibration preservation.** All ten methods preserve CP exchangeability provided the perturbation `φ` is a fixed function (or deterministically-seeded random function) of the input, applied identically on calibration and test. Score becomes `s(x) = disagreement(answer(x), answer(φ(x)))`. For adaptive triggers (methods 1, 2, 5, 9), the trigger itself must also be computed identically on both splits — standard adaptive-CP argument, satisfied by construction.

**Compute vs SC@N.** At equal raw compute (K forward passes), prefix-diversity methods deliver higher *effective* diversity since each pass occupies a different basin. Estimated: SC@4 ≈ 1.3 unique answers on hard math; `system_prompt_ensemble@4` ≈ 2.5; `question_paraphrase@4` ≈ 2.8. CP discriminative power scales with effective diversity, so effective compute favors prefix methods by ~2×.

**Modality matrix.**
- Math: `system_prompt_ensemble` + `backwards_reasoning_check` + `hypothesis_priming`.
- Code: `system_prompt_ensemble` + `question_paraphrase` + `curriculum_prefix` (avoid token-surgery on code).
- MCQ: `hypothesis_priming` (one hypothesis per option) + `question_paraphrase`.

**Pilot ordering.** (1) `system_prompt_ensemble` — cheapest, no KV surgery, universal; validates core hypothesis. (2) `question_paraphrase` — orthogonal axis, equally easy. (3) `prefix_truncation_regen` — first KV-surgery method. (4) `backwards_reasoning_check` — verifier complement. (5) `hypothesis_priming` — highest-ceiling. If 1–2 already close the K-resample gap, deprioritize 5 and 9.

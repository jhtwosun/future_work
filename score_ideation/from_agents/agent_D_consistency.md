# Agent D (consistency / perturbation angle) — Score Function Proposals for CoT-CP

**Author:** Agent D (consistency / perturbation angle)
**Scope:** 5 new trajectory-level conformal score functions based on robustness under perturbation, designed to slot between `prm_min` (~2x compute) and `sc_top1` (~8x compute) on the cost-power Pareto frontier.

## Design philosophy

The existing baselines treat the trajectory as a *static artifact*: a chain can be high-likelihood, high-PRM-scored, and self-consistent yet still be a confidently wrong systematic error mode (misread problem, wrong unit assumption). Correct reasoning is *invariant* to nuisance perturbations — rewordings, mild temperature jitter, step ablations — while wrong reasoning often hinges on a fragile shortcut.

All five proposals below produce a scalar non-conformity score in [0, 1], are parallelizable across the perturbation axis, and target 2-4x compute. We assume the original CoT trace + answer exist (1x baseline) and report total compute including that base.

---

## 1. `xtemp_agree` — Cross-temperature agreement

- **Perturbation type:** temperature variation (greedy vs. mid-temperature sample).
- **Mathematical definition.** Let `a_0 = answer(decode(x, T=0))` (the original greedy) and `{a_k}_{k=1..K}` be answers sampled at `T_hi` (e.g., 0.7). Define
  `s_xtemp(x) = (1/K) * sum_{k=1..K} 1[a_k == a_0]`
  with task-appropriate equality (numeric for math, normalized string for code I/O, label match for MCQ). Non-conformity form: `nc(x) = 1 - s_xtemp(x)`.
- **Compute cost.** 1x (greedy already exists) + K samples at T_hi. With K = 2 this is **~3x total** (~2x marginal). With K = 3, ~4x.
- **Intuition.** A correct chain typically has a sharp answer mode: even mild temperature jitter lands on the same final answer. Fragile chains diverge under temperature because the model is interpolating between a few competing solution sketches.
- **Implementation sketch.** Reuse identical prompt; only sampler config changes. Run K samples in a single batched forward (vLLM's `n=K` parameter) at T=0.7, top_p=0.95. Extract final answer with the same regex/parser as the main pipeline. No new prompts, no PRM calls.
- **Hypothesized Pareto position.** Strictly dominates `prm_min` on power-vs-cost when K=2 if the underlying model has any sampling diversity; competitive with `sc_top1` (K=8) at half the cost because the *greedy mode* is treated as a privileged anchor instead of one vote among eight.
- **Risks.** Some wrong chains are also low-entropy attractors (confidently wrong at every temperature); monolithic systematic errors (misread problem) escape. On tasks with pathological greedy decoding (greedy-collapse on long-form code), the anchor itself is unreliable.
- **Domain coverage.** Works for math, MCQ, short-form code (compare canonical-form output). Less direct for free-form generation; can be adapted via answer-extractor.

---

## 2. `paraphrase_consensus` — Prompt paraphrase robustness

- **Perturbation type:** semantic-preserving prompt rewording.
- **Mathematical definition.** Generate `M` paraphrases `{x_m}_{m=1..M}` of the input `x` via a fixed paraphraser (cheap small LM, or pre-cached paraphrase via the same model with a short instruction). Decode greedy answers `a_m = answer(decode(x_m, T=0))`. Including the original answer `a_0`, define
  `s_para(x) = mode_freq({a_0, a_1, ..., a_M}) / (M + 1)`,
  i.e., the fraction of paraphrase variants whose answer matches the modal answer.
- **Compute cost.** Paraphrase generation is short (≤50 output tokens) — call it **~0.2x** if done with the same model, **~0.05x** with a small auxiliary model. Then M greedy decodes of full CoT at ≈ 1x each. Total with M=2: **~2.2-2.5x**. With M=3: **~3.2x**.
- **Intuition.** Robust solvers are invariant to surface form: "Alice has 3 apples and gains 5" vs. "Three apples, plus five more, belong to Alice". Models that key off specific wording (number-appearance order, idiosyncratic "more than" phrasing) produce divergent answers — the spurious lexical shortcut failure mode.
- **Implementation sketch.** Pre-decode paraphrases in batch with a 1-shot prompt: "Rewrite this problem preserving all numerical facts and the question, in different wording." Optionally validate paraphrases by checking numeric tokens are preserved (regex). Then run greedy CoT on each paraphrase in parallel (single batched call).
- **Hypothesized Pareto position.** Likely beats `prm_min` on linguistically variable tasks (GSM8K-symbolic, MATH word problems, MMLU verbal MCQ); possibly weaker than `sc_top1` on hard math where the bottleneck is computation, not comprehension. At ~2.5x, sits between `prm_min` and `sc_top1`.
- **Risks.** (a) Paraphraser drift: a bad paraphrase silently changes the problem, producing a false inconsistency. (b) On code/formal tasks, paraphrasing is meaningless — the input is already canonical. (c) Adversarially-paraphrase-robust wrong answers exist when the model has memorized a wrong rule.
- **Domain coverage.** Strong on natural-language reasoning (math word problems, MCQ, multi-hop QA). Limited for code (only docstring can be paraphrased) and unsuitable for symbolic-only inputs.

---

## 3. `step_mask_stability` — Step-masked counterfactual stability

- **Perturbation type:** intermediate-step ablation / answer-conditioned regeneration of a single step.
- **Mathematical definition.** Let the original CoT decompose into steps `r = [r_1, ..., r_L]` with answer `a_0`. For each ablated index `i`, construct the partial trace `r_{<i}` and let the model regenerate from `r_i` onward (greedy) producing answer `a_i`. Define
  `s_mask(x) = (1 / L) * sum_{i=1..L} 1[a_i == a_0]`.
  In practice, only sample a random subset of `J` indices (e.g., J=3 evenly spaced) for cost control:
  `s_mask_J(x) = (1 / J) * sum_{i in S} 1[a_i == a_0]`.
- **Compute cost.** Each ablation regenerates a *suffix*, not the full chain — average length ~L/2 tokens. With J=3 ablations: **~3 * 0.5x = 1.5x marginal**, total **~2.5x**. With J=4: ~3x total.
- **Intuition.** If the answer truly follows from the reasoning, truncating-and-resampling from any earlier step should re-derive the same answer. If `r_i` was a load-bearing leap unsupported by `r_{<i}`, regeneration diverges — exposing brittle pivot steps.
- **Implementation sketch.** Split the CoT into steps via newline / "Step k:" / sentence segmentation. For each chosen `i`, prepend the original prompt + `r_{<i}` and let the model continue greedily until end-of-answer. Batch all J suffix-completions. Compare extracted final answer.
- **Hypothesized Pareto position.** Probably the strongest of the five for math/code where reasoning has clear stepwise structure. At ~2.5-3x, plausibly beats `sc_top1` (8x) on long-chain tasks because it directly probes intra-trajectory fragility rather than inter-trajectory diversity.
- **Risks.** (a) Step segmentation is heuristic — bad splits give noisy scores. (b) KV-cache priming may make models copy the original suffix, inflating scores (mitigation: condition only on `r_{<i}`, not `r_i`). (c) Internally self-consistent wrong reasoning re-derives the same wrong answer at each step.
- **Domain coverage.** Excellent for math/code/multi-hop QA where steps are explicit. Limited for one-shot MCQ where there is no decomposition.

---

## 4. `single_shot_pseudo_sc` — Cheap single-sample pseudo self-consistency

- **Perturbation type:** single high-temperature sample as a poor-man's consistency probe.
- **Mathematical definition.** Let `a_0 = answer(decode(x, T=0))` and draw a single sample `a_1 ~ decode(x, T=0.7)`. Define
  `s_pseudo_sc(x) = 1[a_0 == a_1]`,
  a binary indicator. For a finer score, use the per-token mean log-probability of the *answer span* of `a_1` under the prompt+greedy-trace context (a confidence-weighted variant):
  `s_pseudo_sc_soft(x) = 1[a_0 == a_1] * exp(meanlogp(a_1))`.
- **Compute cost.** 1 extra sample at full chain length: **~2x total (~1x marginal)**. The cheapest of the five — designed to be a near-drop-in replacement for `lp_min` with stronger signal.
- **Intuition.** Self-consistency works because high-confidence problems concentrate probability mass on one answer. A single extra sample tells you whether you are in a peaky regime (likely correct) or flat-multimodal (likely wrong) — the cheapest sample-disagreement signal possible.
- **Implementation sketch.** One additional sampling call alongside greedy. Single-token answer parsing. If the binary version is too coarse for CP threshold calibration (ties), use the soft version or break ties via `lp_min`.
- **Hypothesized Pareto position.** Direct `prm_min` competitor at ~2x, but with no PRM dependency (no reward-model calibration drift). Should beat free `lp_min` and roughly match `prm_min` with lower infra burden. Strict subset of `sc_top1` (which is K=8 of this).
- **Risks.** Binary score => coarse calibration (only 2 distinct values per item). Single-sample variance is high; performance on borderline items may be unstable run-to-run. Combine with `lp_min` as a tiebreaker is recommended.
- **Domain coverage.** Universal (math, code, MCQ, free-form with answer extractor).

---

## 5. `backward_question_recon` — Backward question reconstruction

- **Perturbation type:** answer-conditioned backward generation; a bidirectional consistency check.
- **Mathematical definition.** Given prompt `x`, CoT `r`, answer `a_0`, prompt the model to *reconstruct the question* given only `(r, a_0)`. Let `x_hat = decode(reconstruct_prompt(r, a_0))`. Define a similarity score
  `s_back(x) = sim(x, x_hat)`
  where `sim` is one of:
  - embedding cosine similarity (cheap; one embedding-model call per side, often <0.05x),
  - n-gram F1 / ROUGE (free),
  - LLM-judge rubric score from the same model (one short call, ~0.2x).
  The recommended default is **embedding cosine** for speed and stability.
- **Compute cost.** Reconstruction is generating one short paragraph (≈ |x| tokens, much shorter than CoT): **~0.3x**. Plus embeddings (negligible) or judge (~0.2x). Total: **~1.3-1.5x marginal, ~2.3-2.5x including original**.
- **Intuition.** If the CoT genuinely solves the stated problem, conditioning on `(r, a_0)` lets the model recover what was asked. If the CoT silently solved a *different* (misread) problem — a notorious failure mode — the reconstructed question will mention different quantities, units, or entities. This is the CP analog of cycle-consistency in vision.
- **Implementation sketch.** Prompt: "Given this reasoning and final answer, what was the original question? State only the question.". Compute embedding cosine between `x` and `x_hat` using a small embedder (e.g., `all-MiniLM` ~0.01s/pair). Threshold-free score; CP calibrates the threshold automatically.
- **Hypothesized Pareto position.** Targets a different failure mode (semantic-misread, not computational fragility), so it complements the others in an ensemble. Standalone at ~1.5x marginal, likely between `lp_min` and `prm_min` in raw power but pareto-dominant in compute. Best as an ensemble component.
- **Risks.** (a) Strong LMs are good at retro-fitting plausible questions; cycle-consistency might be too easy to satisfy => weak discrimination. (b) Embedding similarity can miss numerical detail (cosine often blurs digits) — mitigate by also doing exact-match check on numeric tokens. (c) Long CoTs leak the question verbatim, trivially reconstructed; mitigate by stripping problem-restatements from `r` before reconstruction.
- **Domain coverage.** Works wherever the input has structure (math, code-with-spec, MCQ-restated-as-question). Weaker for tasks where the input is already minimal.

---

## Summary table

| Score | Perturbation | Compute (total) | Parallel? | Math | Code | MCQ | Pareto slot |
|---|---|---|---|---|---|---|---|
| `xtemp_agree` (K=2) | temperature | ~3x | yes | y | y | y | between prm_min and sc_top1 |
| `paraphrase_consensus` (M=2) | prompt rewording | ~2.5x | yes | y | partial | y | between prm_min and sc_top1 |
| `step_mask_stability` (J=3) | step ablation/regen | ~2.5x | yes (across i) | y | y | partial | strongest candidate, near sc_top1 |
| `single_shot_pseudo_sc` | one extra sample | ~2x | yes | y | y | y | direct prm_min competitor |
| `backward_question_recon` | answer-conditioned reverse | ~1.5x | yes | y | y | y | complement, ensemble role |

## Cross-cutting observations

- **Parallelizability.** All five are embarrassingly parallel across the perturbation axis: a single batched forward (vLLM `n=` or HF `num_return_sequences`) suffices. Wall-clock cost ~1x on a wide GPU; multipliers above are FLOP/throughput costs, not latency.
- **Domain generality.** `xtemp_agree`, `single_shot_pseudo_sc`, `backward_question_recon` are domain-agnostic given an answer extractor. `paraphrase_consensus` is NL-leaning. `step_mask_stability` requires explicit reasoning structure.
- **Failure-mode coverage.** Each probes a distinct fragility: temperature catches decision-boundary items, paraphrase catches lexical shortcuts, masking catches load-bearing pivot steps, pseudo-SC catches multimodal answer distributions, and backward-recon catches semantic misreads. Largely orthogonal — strong candidates for ensemble / multi-score CP.
- **When consistency is a weak signal.** All five share one structural risk: *systematic confident error*. If the model is confidently wrong in a way robust to the chosen perturbation, the score gives false safety. Mitigations: (a) ensemble multiple orthogonal perturbations, (b) combine with PRM or verifier on top-K, (c) add adversarial paraphrases known to flip wrong chains.
- **Empirical next step.** Sweep on GSM8K / MATH / MMLU across compute multipliers {2x, 3x, 4x} for our 5 vs. baselines. Plot acc-at-coverage-0.9 vs. compute. Hypothesis: `step_mask_stability` and `xtemp_agree` lie strictly above the prm_min-to-sc_top1 line.

---
*— Agent D (consistency / perturbation angle)*

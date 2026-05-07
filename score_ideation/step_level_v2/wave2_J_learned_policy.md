# Wave 2 — Agent J (learned-policy angle)

**Context.** Pilots C / K / L (K-resample) collapsed: hand-crafted "branch when entropy > τ" misfired both ways — wasted compute on stable steps, missed pivot steps that look locally confident. *Which* step to intervene on, and *which* intervention, is a learnable function of state. Eight **learned step-policy** methods follow. Shared constraint: the policy must be **frozen before CP calibration** so (cal, test) remain exchangeable — the only configuration preserving marginal coverage.

---

## 1. `step_rl_ppo_policy`

1. **Name.** `step_rl_ppo_policy`
2. **Architecture.** 2-layer MLP head (h=512) on frozen base-LM hidden at end-of-step. Action head + value head.
3. **Action space.** {continue, branch_k=2, branch_k=4, rewrite, verify, abstain}. No continuous params.
4. **Training signal.** Online PPO. Reward = +1 correct, -ε per branch, -λ abstain. Episode = one trajectory. GAE over steps.
5. **Compute.** ~10k rollouts, 4 GPUs, ≈1 day. Inference: <0.1ms/step.
6. **Why beats heuristics.** Heuristics use one statistic; PPO integrates the full hidden, which encodes problem-specific epistemic uncertainty. The value head gives a principled per-step stopping signal entropy never gives.
7. **Sketch.** Stepwise env (step = `\n\n`). MLP reads last hidden, samples action; env resamples k, picks via PRM. PPO with KL-to-uniform-continue prior prevents "always branch" collapse.
8. **CP guarantee.** Holds iff frozen at calibration. Train on A, calibrate on B, deploy on C.
- **OOD.** PPO can overfit to MATH-style. Mitigation: mixed-domain training (MATH + GSM8K + ARC + LogiQA), KL-to-prior. RL on hidden states generalizes better than on tokens because hiddens are domain-abstracted.
- **Calibration.** None if frozen. Online retraining breaks CP — set `requires_grad_(False)`.
- **Failure.** Low-data (<5k trajectories) → high PPO variance. Severe shift → abstain-rate spikes, harming conditional efficiency.

---

## 2. `imitation_from_human_corrections`

1. **Name.** `imitation_from_human_corrections`
2. **Architecture.** Small encoder-decoder transformer (50M params) initialized from a distilled base. Encoder reads partial reasoning; decoder outputs an action token + (for rewrite) a corrected step.
3. **Action space.** {continue, rewrite(text), branch, verify, abstain}. Rewrite is structured: action head emits a discrete category, generator head emits text only when category=rewrite.
4. **Training signal.** Supervised behavioral cloning on human-corrected reasoning traces (ProcessBench, PRM800K human edits, MathStackExchange edit histories). Loss = CE on action + token-CE on rewrite text.
5. **Compute.** Training: ~20 GPU-hours on 50k corrected traces. Inference: encoder pass per step (~5ms); decoder only on intervention steps (~20ms).
6. **Why beats heuristics.** Humans correct *semantically* — they spot algebraic errors, missing cases, off-by-one — none of which trigger entropy spikes. Imitation captures the *type* of error worth fixing, not just the location.
7. **Sketch.** Curate (prefix, gold_action, optional_rewrite) triples from PRM800K-style edit data. Train with teacher forcing. At inference, if action != continue, execute. Use DAgger-style rollouts to fix exposure bias.
8. **CP guarantee.** Frozen policy preserves exchangeability. Caveat: rewrites change solution length — use length-normalized nonconformity score.
- **OOD.** BC generalizes *worse* than RL because it mimics surface form. Mitigation: diverse domains; mixture-of-corrections per problem.
- **Calibration.** Frozen-policy rule. Note: human edits have selection bias (only fixable errors corrected) — calibration must match deployment distribution, *not* the edit set.
- **Failure.** OOD error types (e.g., physics when trained on algebra) → policy confidently rewrites into a *different* wrong answer.

---

## 3. `verifier_distilled_policy`

1. **Name.** `verifier_distilled_policy`
2. **Architecture.** 1-layer MLP (input=hidden_state ⊕ last-step-PRM-score, output=intervene_logit + which-intervention-logits). Tiny: ~500k params.
3. **Action space.** Binary first stage (intervene? yes/no), then categorical {branch, rewrite, verify, abstain}.
4. **Training signal.** Distillation. At training time we have an expensive PRM (Math-Shepherd, RLHFlow, or our own 8B PRM); we run it densely on training trajectories and label each step with "PRM said this step is bad → intervention helped." Train MLP to predict the *expensive PRM's intervention-utility judgement* from the cheap hidden state alone.
5. **Compute.** Training: PRM forwards dominate (~$1k for 100k trajectories); MLP training is minutes. Inference: PRM never runs — only the MLP, ~0.05ms/step.
6. **Why beats heuristics.** Heuristics approximate PRM with one scalar; distillation targets the PRM's full decision function. Distilled classifiers typically recover ≥90% PRM AUC at 1/1000 inference cost.
7. **Sketch.** (a) Generate trajectories. (b) PRM-score every step, label intervention_useful. (c) Train MLP with focal loss (positives are rare). (d) Deploy MLP-only.
8. **CP guarantee.** Clean — offline distillation, frozen MLP. PRM never appears at cal/test, so the inference chain is identical between cal and test.
- **OOD.** Inherits PRM OOD profile *plus* distillation gap. Cheap features correlate with PRM only in-distribution.
- **Calibration.** Frozen MLP → CP-clean. Nonconformity score should not call PRM — use MLP-confidence + final-answer logprob.
- **Failure.** New problem types where hidden-state geometry shifts (code vs. math). Brittle outside training manifold.

---

## 4. `contextual_bandit_intervention`

1. **Name.** `contextual_bandit_intervention`
2. **Architecture.** LinUCB or NeuralUCB. Context = [hidden_state_pca_64, step_index, running_logprob, prev_intervention]. Arms = {none, k2_resample, k4_resample, rewrite, verify, abstain}.
3. **Action space.** Discrete arm selection per step, with UCB confidence widths exposed for budget allocation.
4. **Training signal.** Bandit feedback — each pulled arm yields reward (final correctness − arm_cost). Trained via Thompson sampling or LinUCB updates over historical trajectories; can run offline (counterfactual via inverse propensity) or online.
5. **Compute.** Training: closed-form. Inference: O(d²) where d=64. <0.1ms.
6. **Why beats heuristics.** Heuristics commit to one intervention; bandits learn *which intervention fits which step type*. UCB explicitly trades exploration vs. exploitation.
7. **Sketch.** Per-arm Σ⁻¹, θ. Pull arm = argmax θᵀx + α√(xᵀΣ⁻¹x). Reward observed at episode end; exponential-decay credit assignment over steps.
8. **CP guarantee.** **Required:** freeze (Σ, θ) before calibration. Online updates break exchangeability. Tradeoff: lose adaptivity for coverage.
- **OOD.** Linear assumption breaks on nonlinear reward surfaces. NeuralUCB helps but the network must also be frozen.
- **Calibration.** Freeze. Resisting "just keep learning" is the single most common failure mode.
- **Failure.** Rare-positive arms (abstain) under-explored. Initialize with heuristic-baseline priors.

---

## 5. `ucb_budget_allocator`

1. **Name.** `ucb_budget_allocator`
2. **Architecture.** Per-step uncertainty estimator (lightweight head on hidden state, predicts variance of correctness) + global budget B. UCB scoring across all steps in the trajectory; allocate the top-B by uncertainty for verification.
3. **Action space.** Continuous-flavored: each step gets allocation a_i ∈ [0, B], Σa_i = B. Practically discretized to {0, 1, k} verifications.
4. **Training signal.** Uncertainty head trained via heteroscedastic regression on (hidden_state → final_correctness). UCB params tuned on validation.
5. **Compute.** Training: few GPU-hours. Inference: forward pass per step + streaming top-B heap.
6. **Why beats heuristics.** Heuristics are per-step independent; budget allocation is *global*. 10 steps with budget 2 → spend on the 2 most uncertain, not every step over a fixed τ.
7. **Sketch.** Per step compute uncertainty(h_t), maintain top-B heap. At end (or two-pass), verify top-B. Verification fail → re-derive. KV-cache management is the engineering hazard.
8. **CP guarantee.** Frozen head → CP holds. Budget B is a fixed hyperparameter. Two-pass version: verification verdict must enter nonconformity score consistently between cal and test.
- **OOD.** Variance heads are miscalibrated OOD. Use 5-head ensembles sharing hidden state.
- **Calibration.** Allocator deterministic given hiddens; CP-friendly. Use same B at cal and test.
- **Failure.** Trajectories shorter than B (budget wasted on noise) or much longer than training (top-B saturates). Make B a function of trajectory length, fixed.

---

## 6. `e2e_differentiable_selector`

1. **Name.** `e2e_differentiable_selector`
2. **Architecture.** Gumbel-softmax intervention selector + soft-relaxed CP loss. Selector: 2-layer MLP on hidden state → temperature-annealed soft action distribution.
3. **Action space.** Categorical with continuous relaxation (Gumbel-softmax) — at training time soft-mixed interventions; at inference, hard argmax.
4. **Training signal.** End-to-end gradient on a *validation-set CP efficiency surrogate*: minimize expected prediction-set size at target coverage 1−α. Differentiable through Gumbel + a smooth conformal quantile estimator (e.g., Stutz et al. ConfTr).
5. **Compute.** ~3-4 GPU-days (backprop through conformal procedure). Inference: same as PPO method.
6. **Why beats heuristics.** Directly optimizes the metric we ship (CP set size). Heuristics optimize nothing; PPO optimizes correctness; this optimizes efficiency.
7. **Sketch.** ConfTr-style: smooth quantile, loss = E[set size] + λ·1{coverage<1−α}·penalty. Gumbel over interventions, anneal τ 1.0 → 0.1.
8. **CP guarantee.** Riskiest of the eight. **Strict A/B/C split** — never overlap.
- **OOD.** Overfits to validation distribution shape. Use multi-distribution validation.
- **Calibration.** Marginal coverage holds with frozen selector + clean split. Conditional coverage may suffer (selector concentrates on easy-to-shrink regions).
- **Failure.** Mode collapse to one intervention. Mitigation: entropy regularization on action distribution.

---

## 7. `auxiliary_step_lm`

1. **Name.** `auxiliary_step_lm`
2. **Architecture.** Small LM (~150M params, e.g., distilled Qwen-0.5B) running in parallel. Reads current reasoning prefix, outputs either next step OR a special token `<STUCK>` / `<BRANCH>` / `<VERIFY>`.
3. **Action space.** Token-level: ordinary next-step text *or* special control tokens. Effectively a categorical over {step_text_distribution, control_tokens}.
4. **Training signal.** Two-stage: (a) supervised on (prefix → next_step) from base LM trajectories that ended correctly. (b) RL fine-tune where emitting a control token at the right step earns reward (when control prevents an error).
5. **Compute.** Training ~50 GPU-hours. Inference: ~10ms/step, pipelinable behind base LM latency.
6. **Why beats heuristics.** This policy *understands* the reasoning. A small LM can recognize "I just made an arithmetic claim without justification" — heuristics can't.
7. **Sketch.** Train aux LM to predict next step from prefix. Disagreement with base-LM next-step distribution = branch candidate. Control tokens emerge from RL phase rewarded for catching errors before base commits.
8. **CP guarantee.** Frozen aux-LM is part of the deterministic-given-randomness pipeline; CP-clean.
- **OOD.** Best of all methods: language understanding generalizes better than statistics.
- **Calibration.** Frozen → fine. Disagreement rate shifts with domain — calibrate on representative mixture.
- **Failure.** Aux-LM is weaker than base LM. Must catch errors a stronger model makes — only possible if training data exhibits those failure modes.

---

## 8. `active_step_uncertainty_selection`

1. **Name.** `active_step_uncertainty_selection`
2. **Architecture.** Bayesian-ish ensemble of step-quality classifiers (MC-dropout or 5-head ensemble on shared hidden state). Output: per-step (mean_quality, epistemic_var, aleatoric_var).
3. **Action space.** Per step: verify if epistemic_var > τ; otherwise continue. Optional second action level: branch if mean_quality < τ' AND epistemic_var > τ.
4. **Training signal.** Supervised on (step, was_correct_in_hindsight) labels derived from rollout success. Disagreement → epistemic; consistent uncertainty → aleatoric.
5. **Compute.** Few GPU-hours. Inference: 5 MLPs/step (~0.5ms).
6. **Why beats heuristics.** Active learning: spend compute where the model is uncertain about its own uncertainty, not where surface statistics peak. Epistemic-vs-aleatoric decomposition is impossible for heuristics.
7. **Sketch.** Train ensemble. Per-step epistemic = head disagreement. Verify when epistemic > τ, calibrated on val to hit per-trajectory budget.
8. **CP guarantee.** Frozen ensemble + fixed τ → CP-clean.
- **OOD.** Epistemic uncertainty *is* an OOD signal by design. Ensemble disagrees more on OOD → triggers more verification. This is the method's biggest asset.
- **Calibration.** Frozen → fine. Use verification outcome as a feature in nonconformity (score = -logprob + α·any_verification_failed) to maintain power.
- **Failure.** Joint miscalibration under severe shift (all heads agree on a wrong answer). Per-step quality labels are expensive.

---

## Cross-cutting: shipping learned policies under CP

The frozen-policy protocol is non-negotiable:

1. **Split A:** train the policy.
2. **Freeze:** `requires_grad_(False)` on every learned component (including BN running stats, dropout masks, ensemble seeds).
3. **Split B (calibration):** base-LM + frozen-policy → nonconformity scores → (1−α)-quantile.
4. **Split C (test):** identical pipeline. Marginal coverage holds because (B, C) are exchangeable under one frozen process.

The bug that silently broke pilot K: leaking policy-training data into calibration, or letting the policy adapt online. A subtler bug: conditioning the policy on test-time *batch* statistics (e.g., "intervene if this trajectory's logprob is in the batch bottom-10%") — that creates batch-dependent behavior and destroys i.i.d.

## Recommended pilot ordering

1. **`verifier_distilled_policy`** first: cheapest training, biggest expected lift, well-understood failures.
2. **`active_step_uncertainty_selection`** second: OOD-aware verification with low engineering risk.
3. **`step_rl_ppo_policy`** third: highest ceiling but biggest variance; pilot only after distillation establishes that intervention helps at all.
4. **`e2e_differentiable_selector`** last: largest CP risk but largest alignment with the shipped metric.

`imitation_from_human_corrections` and `auxiliary_step_lm` are deferred — they need data we don't yet have (human edit traces, diverse parallel-reasoning corpora). Flag as data-collection priorities for Wave 3.

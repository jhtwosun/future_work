# Score Brainstorm Synthesis (Wave 1 — Trajectory-level)

> 5 Subagents (A-E) + 2 Codex (1, 2) → 27 score function proposals
> Date: 2026-05-07

## All 27 proposed scores, organized by compute class

### Class 1×: Free (no extra forward, computable from existing greedy traces)

| Score | Source | Mechanism | CPU only? |
|---|---|---|---|
| `step_entropy_p95` | A | 95-percentile of per-step token entropy | ✓ if we have logprobs=k |
| `predictive_sharpening_rate` | A | OLS slope of step entropy vs step index | ✓ from existing per-step lp |
| `final_answer_recheck` | C | substitute-and-check (sympy on extracted ans) | ✓ |
| `verifier_step_pass` | C | sympy/exec on each step's claims | ✓ |

### Class ~1.05–1.15× (single forward + light overhead)

| Score | Source | Mechanism |
|---|---|---|
| `tempered_kl_divergence` | A | JS divergence between low-T and high-T softmax (same logits) |
| `hidden_drift_max` | B | max step-to-step cosine distance on penultimate residuals |
| `attn_entropy_focus` | B | top-5% retrieval-head attention entropy |
| `wrongness_probe_max` | B | linear probe on hidden states for "I'll be wrong" |
| `layer_consistency_disagreement` | B | logit/tuned-lens JS across last 4 layers |
| `mutual_info_step_to_answer` | A | offline-trained probe on (h_t, final_answer) PMI |
| `backward_consistency` | C | tiny LM + embedding cosine for reverse reconstruction |

### Class 1.5–2.5× (one extra sample)

| Score | Source | Mechanism |
|---|---|---|
| `single_shot_pseudo_sc` | D, Codex2 | greedy vs T=0.7 single sample agreement |
| `lookahead_branching_entropy` | A | KV-cache reused short rollouts at K fork points |
| `next_step_disagreement` | B | step-boundary M short rollouts, terminal hidden Cov |
| `paraphrase_consensus` | D, Codex2 | M=2 paraphrase agreement |

### Class 2.5–4× (multi-sample / multi-prompt)

| Score | Source | Mechanism |
|---|---|---|
| `xtemp_agree` | D | greedy vs K=2 stochastic samples agreement |
| `step_mask_stability` | D | random step truncate + suffix regen, stability rate |
| `cross_reform_agreement` | C | M=3 question paraphrase, answer invariance |
| `sc_verifier_pass` | C | verifier-gated SC (K samples, fraction passing verifier) |

### Class 8× (fully expensive — existing sc_top1 baseline)

| Score | Source | Mechanism |
|---|---|---|
| `sc_top1` | (existing) | 8 SC samples top-1 fraction |

### Hybrid / Ensemble (cost = sum of components)

| Score | Source | Mechanism |
|---|---|---|
| `gated_cascade` | E | lp→prm→sc hierarchical routing |
| `disagreement_signal` | E | logreg with (z_lp − z_prm)² interaction |
| `quantile_fusion_cp` | E | weighted-quantile-deviation max across calibrators |
| `tiebreak_lex` | E | sc_top1 + lp_min as tiebreaker |
| `learned_stack_cv` | E | GBDT over (lp, prm, sc, attn_ent, span_PPL) |

### Bayesian / Theoretical (Codex 1)

| Score | Source | Theoretical motivation |
|---|---|---|
| `posterior_logit_probe` | Codex1 | direct probe estimating P(Y=1 \| R) |
| `mdl_bonus` | Codex1 | description length proxy for likelihood ratio |
| `evidence_accumulation` | Codex1 | per-step block-evidence sum |
| `fisher_info_proxy` | Codex1 | curvature of log p around chosen tokens |

---

## Top picks for immediate experimentation

Based on (i) compute efficiency (ii) implementability (iii) coverage of distinct mechanisms:

### Tier A (CPU-only, fast, high coverage)
1. **`tiebreak_lex`** — sc_top1 with lp_min tiebreaker. Pareto improvement over sc_top1, free.
2. **`gated_cascade`** — lp→prm→sc routing. ~3.4× compute vs sc_top1 8×. Major saving.
3. **`final_answer_recheck`** — sympy on final answer. ~1.0× compute.
4. **`disagreement_signal`** — logreg (lp, prm) + interaction. Extends naive ensemble.
5. **`predictive_sharpening_rate`** — entropy slope. New independent signal.

### Tier B (1 extra forward, info-theoretic)
6. **`tempered_kl_divergence`** — JS(low-T, high-T softmax). 1.1×. New independent signal.
7. **`hidden_drift_max`** — penultimate cosine. 1.05×. Most portable across model families.
8. **`wrongness_probe_max`** — SAPLMA-style probe. 1.02× inference, requires offline training.

### Tier C (2-3× compute, perturbation)
9. **`single_shot_pseudo_sc`** — greedy vs 1 sample. 2× compute, simple.
10. **`xtemp_agree`** — 2-sample agreement at varied T. 3× compute.
11. **`step_mask_stability`** — random step mask + regen. 2.5× compute, may dominate prm_min.

### Tier D (verifier — domain-specific)
12. **`verifier_step_pass`** — math verifier. ~1.05×, math-only but can be powerful.

---

## Mechanism diversity map

Each tier above represents a different *mechanism*:

| Mechanism family | Tier-A representative | Tier-B representative | Tier-C representative |
|---|---|---|---|
| Likelihood-based | `predictive_sharpening_rate` | `tempered_kl_divergence` | (sc_top1) |
| Internal-state | — | `hidden_drift_max`, `wrongness_probe_max` | `next_step_disagreement` |
| Verifier-based | `final_answer_recheck` | `backward_consistency` | `verifier_step_pass`, `sc_verifier_pass` |
| Perturbation | — | `single_shot_pseudo_sc` | `xtemp_agree`, `step_mask_stability`, `paraphrase_consensus` |
| Hybrid | `tiebreak_lex`, `gated_cascade`, `disagreement_signal` | `quantile_fusion`, `learned_stack_cv` | — |

This gives **broad coverage** of the score-space. Experiments should cover at least one representative from each family and each tier.

---

## Experimental priorities (no cherry-picking — implement everything cheap)

**Phase 1 (CPU, immediate)**: All Tier A + selected Tier B (probes if hidden states available).

**Phase 2 (GPU, single new forward pass)**: Tier B + Tier C single-shot.

**Phase 3 (GPU, extensive)**: Tier C multi-sample, hparam sweeps.

**Phase 4 (GPU, hidden states)**: Tier B internal-state via transformers (not vLLM).

For all: **bootstrap CIs**, multiple α, multiple datasets (MATH-500 primary; GSM8K saturation check; OlympiadBench mid-difficulty; MMLU-Pro cross-domain).

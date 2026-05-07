# CoT-CP Paper Outline (Draft)

> **Working title**: *Calibrated Selective Generation for LLM Reasoning via Conformal Prediction: A Score-Family Pareto Across Models, Domains, and Architectures*
>
> Drafted while Codex works on Theorem 1 in parallel. To be merged after we cross-verify each other's theorem statements.

## Target venue

- **Primary**: TMLR (8-page expedited, empirical-heavy fits well)
- **Secondary**: NeurIPS Datasets & Benchmarks (showcasing broad evaluation)
- **Stretch**: ICLR / NeurIPS main (if Theorem 3 lands as a methodological contribution)

## Section structure

### 1. Abstract (~150 words)

CoT-CP wraps any step-level confidence signal — log-prob (lp_min),
process reward model output (prm_min), or self-consistency top-1
fraction (sc_top1) — into a calibrated selective predictor for LLM
reasoning. Coverage is finite-sample distribution-free under
exchangeability (Theorem 1). The three score families form a
**compute-Pareto ladder** (lp 1×, PRM 2×, SC 8×) — Theorem 2
characterizes when the cheaper score wins. Empirically, CoT-CP
delivers **+10 to +36 pp selective accuracy** across 11 models × 7
benchmarks (math, code, MCQ, theorem). For OOD shift (MATH→AIME),
**weighted CP with empirical-PMF density-ratio recovers target
coverage** for discrete scores (Theorem 3) — a key methodological
correction over standard KDE-based weighted CP.

### 2. Introduction (~1.5 pages)

- Motivation: LLM reasoning is unreliable; users need calibrated
  selective predictors.
- Existing test-time compute literature (DeepConf, DEER, ESC, PAVs,
  Snell+) gives heuristic thresholds without coverage guarantees.
- Existing CP-for-LLM work (Quach 2024, Mohri-Hashimoto 2024,
  Abbasi-Yadkori 2024) controls factuality / abstention but not
  reasoning-step confidence aggregation.
- We close the gap: trajectory-level CP wrapper that converts any
  step-level confidence into a calibrated predictor with bootstrap-
  validated coverage.
- **Contributions**:
  (i) Theorem 1: trajectory-level CP from arbitrary step aggregator
  (ii) Theorem 2: score-family Pareto via SNR ordering
  (iii) Theorem 3: weighted CP for discrete scores (OOD robustness)
  (iv) Empirical: 11 models × 7 datasets × 5 α × 500 bootstrap CIs
  (v) Negative result: step-level branching does NOT pay
  (vi) New mid-cost score family: **PRM-min** (Qwen2.5-Math-PRM-7B)

### 3. Background and related work (~1 page)

- Split CP & CRC (Vovk-Gammerman-Shafer 2005; Angelopoulos+ 2022)
- Conformal LM (Quach+ ICLR 2024) — sequence-level, our work step-aggregated
- Conformal factuality (Mohri-Hashimoto ICML 2024) — claim-level
- Conformal abstention (Abbasi-Yadkori NeurIPS 2024) — closest, but
  abstention vs selective acceptance
- Test-time scaling: DeepConf, DEER, Self-Consistency, PRM-based search
- Process reward models: PRM800K (Lightman+), Math-Shepherd, Qwen2.5-Math-PRM
- Weighted CP & beyond exchangeability: Tibshirani+ 2019, Barber+ 2022

### 4. Method (~1.5 pages)

#### 4.1 Setup
- Calibration / test data, exchangeability
- Step-level confidence $S_{i,t}$, trajectory aggregator $\phi$, score $\bar{S}$
- Three concrete score families (lp_min, prm_min, sc_top1) with cost ratios

#### 4.2 Theorem 1 (foundation)
Statement; proof in appendix; corollary on selective error.

#### 4.3 Theorem 2 (Pareto)
Selective accuracy = monotone in SNR; cost-Pareto = upper SNR envelope.

#### 4.4 Theorem 3 (weighted CP for discrete scores)
Empirical-PMF density-ratio estimator + score-only-shift assumption.

#### 4.5 Step-level branching: an explored-and-rejected alternative
Brief discussion: K-resample from worst-step prefix doesn't work
across lp/PRM/rewrite-cue scores. Bottleneck is intervention class,
not score quality. (Detailed null result in appendix.)

### 5. Experiments (~3 pages — main empirical content)

#### 5.1 Setup
- 11 models: Qwen2.5-7B/32B, Qwen3-8B (no-think) / Qwen3-30B-A3B-MoE,
  Qwen2.5-Math-7B-Instruct, QwQ-32B, R1-Distill-7B/32B/Llama-70B,
  Phi-4, Mixtral-8x7B, DeepSeek-V2-Lite-MoE
- 7 datasets: GSM8K (1319), MATH-500 (500), AIME 1983-2024 (933),
  MMLU-Pro STEM (4192), OlympiadBench math (674), TheoremQA (747),
  HumanEval (164)
- Robust answer extractor (latex-aware)
- Bootstrap 95% CIs (500 boot × 10 cal/test splits)

#### 5.2 In-distribution coverage validation
Empirical coverage matches target $1-\alpha$ across all 5 score families
× 11 models × 7 datasets. (Headline figure: per-α calibration curve.)

#### 5.3 Score-family Pareto (Figure 1)
Selective accuracy vs answer rate, three score families overlaid for
MATH-500 + Qwen2.5-7B. PRM-min sits between lp_min and sc_top1.
Cross-model consistency.

#### 5.4 Cross-domain generalization
7 datasets × Qwen3-8B-no-think: MMLU-Pro 78%, OlympiadBench 58%,
HumanEval 82%, TheoremQA 59% with SC@8; CP+SC pushes all to 80%+
on at least 50% of test problems.

#### 5.5 Cross-architecture: dense vs MoE vs reasoning-distilled
Qwen3-30B-A3B (3B active MoE) matches Qwen2.5-32B dense; CoT-CP works
identically on both architectures.

#### 5.6 OOD robustness: MATH→AIME (Theorem 3 in action)
Vanilla CP under-covers on AIME; KDE-weighted CP fails for
discrete sc_top1 (Pilot J initial finding); empirical-PMF-weighted
CP recovers target coverage (key Table).

#### 5.7 Compute-Pareto frontier
Figure showing: at fixed compute, the optimal score family. Confirms
PRM-min as new mid-cost operating point.

### 6. Discussion (~0.5 page)

- Limitations: large vanilla-accuracy gap to SOTA frontier models; we
  don't compete on raw accuracy but on calibration.
- The bottleneck of step-level intervention (negative result)
  motivates future work on better repair primitives.
- Honest framing: most theorems are clean specializations of existing
  CP machinery; the contribution is the empirical map and the
  empirical-PMF discrete-shift result.

### 7. Conclusion (~0.25 page)

- CoT-CP = trajectory-level CP wrapper, not a new score
- PRM-min as new mid-cost operating point
- Coverage-validated Pareto frontier across models / domains / architectures
- Open question: can step-level interventions ever pay?

---

## Appendices

### A. Full proofs (Theorems 1, 2, 3)

### B. Robust answer extractor implementation details

### C. Bootstrap CI methodology

### D. Step-level branching null result (Pilots C/K/L details)

### E. Per-dataset full results table (the 256-row table)

### F. Reproducibility checklist

---

## Empirical results inventory (already in hand)

| Section | Source | Status |
|---|---|---|
| §5.1 setup | E5v2_table1.md (256 rows) | ✓ Ready |
| §5.2 coverage | All E1-E18 + Pilot 8/J | ✓ Ready |
| §5.3 Pareto figure | E5v2_figure1.png + pilotN_pareto_with_prm.png | ✓ Ready |
| §5.4 cross-domain | E12-E15 results | ✓ Ready (with truncation rerun) |
| §5.5 cross-arch | E4 (32B), E9 (Qwen3-MoE), E10 (Mixtral), E11 (DSV2L), E18 (70B) | ✓ Ready |
| §5.6 OOD weighted CP | gap_sc_ood_weighted.json + Pilot F | ✓ Ready (just produced) |
| §5.7 compute Pareto | Pilot D (compute_matched.png) | ✓ Ready |
| Step branching negative | Pilots C/K/L | ✓ Ready (appendix) |

---

## Critical-path remaining work

1. **Theorem 1 from Codex** ← waiting now
2. **Cross-verify Theorems 1, 2, 3 between Claude and Codex** ← next
3. **Truncation rerun (E12r, E13r, E14r, E6v3)** to clean up §5.4 numbers
4. **Polish figures** (E5v2_figure1 rough, needs publication-grade)
5. **Write the actual paper** (~3-4 weeks)

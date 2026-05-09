# Papers — INDEX

> 30 reference papers for the **CoT-CP** project (Conformal Prediction for LLM Chain-of-Thought).
> Each row links to a per-paper note (`{arxiv_id}.md`) with verbatim abstract, method summary, reported results, and explicit positioning vs CoT-CP.
> Last build: 2026-05-08 from 6 parallel WebFetch passes.

---

## Quick navigation

- [Tier 1 — Foundations (5)](#tier-1--cp-foundations) — must-cite, low threat
- [Tier 2 — CP-for-LLMs core (5)](#tier-2--cp-for-llms-core) — direct precedents, mostly HIGH threat
- [Tier 3 — CP-for-LLMs recent 2024-26 (5)](#tier-3--cp-for-llms-recent) — newest CP variants, mixed threat
- [Tier 4 — HIGH-threat test-time-scaling (5)](#tier-4--test-time-scaling-high-threat) — empirical competitors using uncalibrated thresholds
- [Tier 5 — PRMs + step-level infrastructure (5)](#tier-5--prms--step-level-infrastructure) — calibration data, candidate scores, benchmark
- [Tier 6 — Recent step-level competitors (5)](#tier-6--recent-step-level-competitors) — 2025-26 step-level adaptive compute
- [Threat ranking](#threat-ranking) — read this first
- [Mandatory baselines](#mandatory-experimental-baselines) — papers we must reproduce

---

## Tier 1 — CP foundations

Must-cite. Provide the proof machinery our Theorems 1-3 inherit.

| arxiv | title | cite key | role | note |
|---|---|---|---|---|
| [2107.07511](2107.07511.md) | A Gentle Introduction to CP and Distribution-Free UQ | `angelopoulos2023gentle` | tutorial, split-CP backbone | reading-order-1 |
| [2208.02814](2208.02814.md) | Conformal Risk Control | `angelopoulos2024crc` | **Theorem 1 backbone** (CRC for monotone losses) | core dependency |
| [1904.06019](1904.06019.md) | CP Under Covariate Shift | `tibshirani2019weighted` | **Theorem 3 backbone** (weighted CP) | discrete-shift specialization |
| [2202.13415](2202.13415.md) | CP Beyond Exchangeability (nexCP) | `barber2023beyond` | non-exchangeable framework | OOD section |
| [2110.01052](2110.01052.md) | Learn Then Test (LTT) | `angelopoulos2025ltt` | risk control via multiple testing | CRC complement |

## Tier 2 — CP for LLMs core

Direct precedents in the CP-for-LLM literature. **All five must be explicit baselines or comparison anchors** in §2 / §5.

| arxiv | title | cite key | threat | role |
|---|---|---|---|---|
| [2306.10193](quach2024-conformal-language-modeling.md) | Conformal Language Modeling (Quach 2024, ICLR) | `quach2024-conformal-language-modeling` | medium | sequence-level set-of-generations CP |
| [2402.10978](2402.10978.md) | Language Models with Conformal Factuality (Mohri-Hashimoto, ICML 2024) | `mohri2024-conformal-factuality` | **HIGH** | claim-decomposition baseline |
| [2406.09714](2406.09714.md) | LLM Validity via Enhanced CP (Cherian-Gibbs-Candès, NeurIPS 2024) | `cherian2024-enhanced-llm-validity` | medium | conditional CP + diff-through-CP |
| [2405.01563](2405.01563.md) | Mitigating LLM Hallucinations via Conformal Abstention (Abbasi-Yadkori, ICLR 2024) | `abbasi-yadkori2024-conformal-abstention` | **HIGH** | **closest direct precedent** — their score ≈ our `sc_top1` |
| [2505.17126](2505.17126.md) | Coherent Factuality (Rubin-Toles, ICLR 2025) | `rubin-toles2025-coherent-factuality` | **HIGH** | first to break exchangeable-claim via deducibility graphs |

## Tier 3 — CP for LLMs recent

Newer 2024-26 CP variants. Two are the closest threats; three are useful baselines/composables.

| arxiv | title | cite key | threat | role |
|---|---|---|---|---|
| [2509.04733](2509.04733.md) | CoVeR: Conformal Calibration for Versatile Next-Token Prediction | `chen2025cover` | **HIGH** | closest step-level CP — token-cluster granularity |
| [2604.20098](2604.20098.md) | Differentiable Conformal Training for LLM Reasoning Factuality | `hittesdorf2026dcf` | **HIGH** | training-time CP via differentiable scorer |
| [2407.00499](2407.00499.md) | ConU (Wang+ EMNLP-Findings 2024) | `wang2024conu` | medium | sequence-level black-box CP baseline |
| [2510.05566](2510.05566.md) | Domain-Shift-Aware CP for LLMs | `lin2025dscp` | medium | composable, prompt-embedding weighting |
| [2604.16217](2604.16217.md) | Beyond Surface Statistics (LI scores) | `wang2026beyondsurface` | medium | white-box hidden-state-based CP |

## Tier 4 — Test-time scaling HIGH threat

Empirical competitors using uncalibrated thresholds. **All five must be reproduced as baselines** on overlapping benchmarks (MATH-500, AIME).

| arxiv | title | cite key | threat | concrete heuristic threshold |
|---|---|---|---|---|
| [2508.15260](2508.15260.md) | Deep Think with Confidence (DeepConf) — Meta 2025 | `fu2025deepconf` | **HIGH** | η-percentile of N_init=16 warm-up traces (per prompt) |
| [2504.15895](2504.15895.md) | Dynamic Early Exit in Reasoning Models (DEER) | `yang2025deer` | **HIGH** | λ ≈ 0.95 swept on eval set (no held-out cal) |
| [2305.11860](2305.11860.md) | Adaptive-Consistency (Aggarwal+, EMNLP 2023) | `aggarwal2023adaptive` | **HIGH** | Beta-Binomial posterior > 0.95 (over majority, not correctness) |
| [2401.10480](2401.10480.md) | ESC: Early-Stopping Self-Consistency (Li+, ICLR 2024) | `li2024esc` | **HIGH** | window w=8 (MATH) / w=5 (others); entropy-zero stop |
| [2408.03314](2408.03314.md) | Scaling Test-Time Compute Optimally (Snell+, ICLR 2025) | `snell2025scaling` | **HIGH** | difficulty bins from 2048 oracle samples per question |

## Tier 5 — PRMs + step-level infrastructure

Calibration data sources, candidate score functions, and the standard step-error benchmark.

| arxiv | title | cite key | role |
|---|---|---|---|
| [2305.20050](2305.20050.md) | Let's Verify Step by Step (Lightman+, ICLR 2024) | `lightman2023verify` | **PRM800K = our calibration data** (800K labels / 75K solutions / 12K problems) |
| [2312.08935](2312.08935.md) | Math-Shepherd (Wang+, ACL 2024) | `wang2024mathshepherd` | auto-labeled PRM via MC rollouts (HE/SE) — alternative score |
| [2410.08146](2410.08146.md) | Rewarding Progress / PAVs (Setlur+, Google+CMU 2024) | `setlur2024rewarding` | **HIGH threat (G5)** — step-reward as advantage under prover policy |
| [2412.06559](2412.06559.md) | ProcessBench (Zheng+, Qwen 2024) | `zheng2024processbench` | step-error eval; reveals PRM OOD failure (47.9 → 23.8 F1) |
| [2203.11171](2203.11171.md) | Self-Consistency (Wang+, ICLR 2023) | `wang2023selfconsistency` | original SC; vote-share is one of our scores |

## Tier 6 — Recent step-level competitors

2025-26 step-level adaptive compute papers. All HIGH threats.

| arxiv | title | cite key | threat | wedge for CoT-CP |
|---|---|---|---|---|
| [2511.06209](2511.06209.md) | UHeads / ReProbe (Ni+, ACL 2026) | `ni2025uheads` | **HIGH** | wrap their probe scores with CRC quantile |
| [2503.15848](2503.15848.md) | Entro-duction (Zhang+, ACL 2025) | `zhang2025entroduction` | **HIGH** | entropy-variance idea worth borrowing as one component of $s_t$ |
| [2503.22233](2503.22233.md) | EDU-PRM (Cao+ 2025) | `cao2025edu` | **HIGH** | entropy anchors threaten "split-by-newline"; need ablation |
| [2505.11730](2505.11730.md) | VG-Search (Chen+, NeurIPS 2025) | `chen2025vgsearch` | **HIGH** | **closest competitor on granularity axis; open-source code → mandatory baseline** |
| [2602.18447](2602.18447.md) | ConfSpec (Liu+He, 2026) | `liu2026confspec` | **HIGH** | small-draft calibrated-within-competence threatens "calibration is novel" pitch |

---

## Threat ranking

Read in this order before writing §2 (Related Work):

| rank | paper | why it's the worst threat |
|---|---|---|
| 1 | `chen2025cover` (CoVeR) | step-level CP for autoregressive reasoning. Closest published prior art. |
| 2 | `hittesdorf2026dcf` (Differentiable CT) | claim-graph CP, 141% retention gain. Training-time differentiable scorer. |
| 3 | `rubin-toles2025-coherent-factuality` | first to break exchangeable-claim assumption with deducibility graphs |
| 4 | `chen2025vgsearch` (VG-Search) | identical question — verification granularity — answered with adaptive heuristic + open-source code |
| 5 | `fu2025deepconf` | span-confidence filtering on AIME-25 99.9% / -84.7% tokens |
| 6 | `abbasi-yadkori2024-conformal-abstention` | their semantic-entropy score ≈ our `sc_top1` rung at trajectory level (must label explicitly) |
| 7 | `mohri2024-conformal-factuality` | foundational claim-CP; canonical anchor |
| 8 | `setlur2024rewarding` (PAVs) | step-level adaptive compute via prover-policy advantage |
| 9 | `yang2025deer` (DEER) | step-level early exit driven by trial-answer confidence |
| 10 | `cao2025edu` (EDU-PRM) | entropy-driven step segmentation threatens our segmentation choice |

## CoT-CP defensible niche (synthesized from positioning notes)

1. **Semantic-step granularity** (vs CoVeR's token-cluster, vs Mohri-Hashimoto's atomic-claim, vs Quach's whole-sequence)
2. **Inference-time post-hoc on closed-source APIs** (vs DCF's training-time, vs UHeads's white-box probe)
3. **Selective accuracy on the final answer of multi-step chains** (vs claim-coverage, vs decoding-trajectory coverage)
4. **Distribution-free coverage statement P(correct) ≥ 1−α via CRC** (vs all heuristic-threshold competitors)
5. **Empirical-PMF weighted CP for discrete scores under shift** (Theorem 3 — original methodological correction)

## Mandatory experimental baselines

The following must be reproduced on at least MATH-500 and AIME:

- `fu2025deepconf` (DeepConf) — span-confidence filtering
- `yang2025deer` (DEER) — step-level early exit
- `aggarwal2023adaptive` + `li2024esc` — adaptive-consistency / ESC
- `snell2025scaling` — compute-optimal best-of-N + revision + PRM-beam
- `chen2025vgsearch` (VG-Search) — adaptive verification granularity (open code)
- `setlur2024rewarding` (PAVs) — process advantage verifier
- `wang2023selfconsistency` (SC) — vote-majority baseline
- `mohri2024-conformal-factuality` — claim-decomposition CP
- `abbasi-yadkori2024-conformal-abstention` — sequence-level conformal abstention
- `chen2025cover` (CoVeR) — token-cluster step CP

Composable, not strictly competitive (can stack with CoT-CP):
- `ni2025uheads` (UHeads) — wrap their probe scores
- `cao2025edu` (EDU-PRM) — wrap their step-segmented scores
- `lin2025dscp` (DS-CP) — apply on top of our step-level CP for shift

## Files generated
- `INDEX.md` (this file)
- `positioning_matrix.md` — column-axis cross-comparison
- `references.bib` — BibTeX starter for the LaTeX paper

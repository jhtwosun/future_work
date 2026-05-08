# Literature Review 01: Conformal Prediction for Large Language Models

**Compiled:** 2026-05-05
**Last updated:** 2026-05-08 — added Thought Calibration (EMNLP 2025), CP Beyond the Seen (NeurIPS 2025), Paraphrase-Robust CP (ICLR 2026 submission), CP+ASP Scaffolds, Unsupervised Conformal Prediction (UCP), CP: A Data Perspective survey. See §2.17–2.22 and updated §8 gap analysis.
**Scope:** Foundational CP + 2023-2026 papers on CP applied to LLMs/VLMs.
**For projects:** CoT-CP (step-level CP for chain-of-thought), CoT-Tool (calibrated tool routing), CG-VL (PMI-based conformity score for VLM hallucination).

Tags used in "Relevance":
- **CoT-CP** = step-level CP for reasoning
- **CoT-Tool** = calibrated tool/agent routing
- **CG-VL** = VLM hallucination via conformity score
- **Common** = relevant to all three (foundational machinery)

---

## 1. Foundational Conformal Prediction (Brief)

These are background; the user already knows them.

1. **Vovk, Gammerman, Shafer (2005).** *Algorithmic Learning in a Random World.* Springer. The original CP textbook; defines transductive and inductive (split) CP under exchangeability.
2. **Lei, G'Sell, Rinaldo, Tibshirani, Wasserman (2018).** *Distribution-Free Predictive Inference for Regression.* JASA. Established split (inductive) conformal prediction in modern form.
3. **Angelopoulos & Bates (2021/2023).** *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* Foundations and Trends in ML. https://arxiv.org/abs/2107.07511 - The standard tutorial; will be cited throughout.
4. **Romano, Sesia, Candes (2020).** *Classification with Valid and Adaptive Coverage* (APS). NeurIPS. Adaptive prediction sets with conditional coverage; the de facto MCQ baseline.
5. **Angelopoulos, Bates, Fisch, Lei, Schuster (2022/ICLR 2024).** *Conformal Risk Control.* https://arxiv.org/abs/2208.02814 - Extends split CP to control any monotone bounded loss in expectation. **Backbone for all three planned projects** (CRC = the user's working machinery).
6. **Tibshirani, Barber, Candes, Ramdas (NeurIPS 2019).** *Conformal Prediction Under Covariate Shift.* https://arxiv.org/abs/1904.06019 - Weighted CP via likelihood-ratio reweighting; the foundation for all distribution-shift work.
7. **Barber, Candes, Ramdas, Tibshirani (Annals of Stat. 2023).** *Conformal Prediction Beyond Exchangeability* (nexCP). https://arxiv.org/abs/2202.13415 - General framework for non-exchangeable data via fixed weights; provides the *coverage gap* bound used by many follow-ups.
8. **Angelopoulos, Bates, Candes, Jordan, Lei (2021/AOAS 2025).** *Learn Then Test (LTT).* https://arxiv.org/abs/2110.01052 - Reframes risk control as multiple hypothesis testing; enables non-monotone losses (CRC's complement).
9. **Gibbs, Cherian, Candes (2023).** *Conformal Prediction with Conditional Guarantees.* The conditional conformal procedure, used as a building block for adaptive LLM-CP work below.

---

## 2. CP for LLM Sequence / Text Generation (deep)

### 2.1 Quach et al. — Conformal Language Modeling (CLM)
- **Title:** Conformal Language Modeling
- **Authors:** Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi Jaakkola, Regina Barzilay
- **Venue:** ICLR 2024
- **Link:** https://openreview.net/forum?id=pzUhfQ74c5
- **TL;DR:** Adapts split CP to *open-ended generation*: samples K candidate sequences, uses a stopping rule + admission function to construct a prediction set guaranteed to contain at least one acceptable response with prob >= 1-alpha. Conformity score is a per-sample quality estimate from the LM itself (likelihood + self-eval). Applied to QA, summarization, and radiology reports.
- **Relevance:** **Common** — the seminal LLM-CP paper. CoT-CP must compare directly; CG-VL borrows the "set-of-generations" framing.
- **Gap:** Treats response as an opaque sequence; no step-level decomposition, no explicit dependency on retrieval evidence, no multimodal grounding.

### 2.2 Kumar et al. — Conformal Prediction with LLMs for Multi-Choice QA
- **Title:** Conformal Prediction with Large Language Models for Multi-Choice Question Answering
- **Authors:** Bhawesh Kumar, Charlie Lu, Gauri Gupta, Anil Palepu, et al.
- **Venue:** arXiv 2305.18404 (ICML 2023 workshop; widely cited)
- **Link:** https://arxiv.org/abs/2305.18404
- **TL;DR:** Applies split CP to MCQ logits (LLaMA-13B on MMLU-style benchmarks); the conformity score is the softmaxed answer-token probability. Shows CP set sizes correlate strongly with accuracy, enabling selective classification.
- **Relevance:** **CoT-Tool** — direct precedent for routing decisions phrased as MCQ over tools; **Common** baseline.
- **Gap:** Logit-only conformity; treats each option independently; no multi-step reasoning or tool-call cost.

### 2.3 Ulmer, Zerva, Martins — Non-Exchangeable Conformal Language Generation
- **Title:** Non-Exchangeable Conformal Language Generation with Nearest Neighbors
- **Authors:** Dennis Ulmer, Chrysoula Zerva, Andre Martins
- **Venue:** EACL Findings 2024
- **Link:** https://arxiv.org/abs/2402.00707
- **TL;DR:** Token-level CP for autoregressive generation under nexCP; uses kNN over decoder hidden states + distance-based weights to form per-token prediction sets without IID assumptions. Demonstrates on MT and LM tasks.
- **Relevance:** **CoT-CP** — closest existing work to per-step / per-token calibration; the user's step-level CoT-CP must position against this.
- **Gap:** Token-level, not semantic-step-level; no notion of reasoning structure; no risk control over the full chain.

### 2.4 Wang et al. — ConU
- **Title:** ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees
- **Authors:** Zhiyuan Wang, Jinhao Duan, Lu Cheng, et al.
- **Venue:** EMNLP Findings 2024
- **Link:** https://arxiv.org/abs/2407.00499
- **TL;DR:** Black-box CP for open-ended NLG. Conformity score = self-consistency frequency (cluster sampled answers semantically; score by cluster size). Validated across 7 LLMs and 4 free-form QA datasets, including medical.
- **Relevance:** **CoT-Tool** — self-consistency-as-conformity is a strong baseline for tool-decision confidence.
- **Gap:** Treats output as atomic; no step-level calibration; no tool-cost trade-off.

### 2.5 Wang et al. — SConU
- **Title:** SConU: Selective Conformal Uncertainty in Large Language Models
- **Authors:** Zhiyuan Wang, Qingni Wang, Yue Zhang, Tianlong Chen, et al.
- **Venue:** ACL 2025 (Long)
- **Link:** https://arxiv.org/abs/2504.14154
- **TL;DR:** Adds a significance test using two conformal p-values to detect calibration-set outliers (exchangeability violations) at deployment time; manages miscoverage across single- and cross-domain QA.
- **Relevance:** **Common** — practical fix when calibration set != deployment data; relevant for CoT-Tool deployment in unseen tool ecosystems.
- **Gap:** Detects outliers but does not adapt the conformity score; not multimodal.

### 2.6 Su et al. — Conformal Information Pursuit
- **Title:** Conformal Information Pursuit for Interactively Guiding Large Language Models
- **Authors:** Kwan Ho Ryan Chan, Yuyan Ge, Edgar Dobriban, Hamed Hassani, Rene Vidal
- **Venue:** arXiv 2507.03279 (2025)
- **Link:** https://arxiv.org/html/2507.03279v2
- **TL;DR:** Combines CP with information-theoretic question selection so an LLM agent asks the most informative follow-up; CP set size acts as the uncertainty signal driving exploration.
- **Relevance:** **CoT-Tool** — analogous to "ask for help" routing; CoT-Tool can adopt this as a baseline for clarification tools.
- **Gap:** Single-turn information gain; no formal cost-aware tool routing.

### 2.7 Vishwakarma et al. — Prune 'n Predict (CROQ + CP-OPT)
- **Title:** Prune 'n Predict: Optimizing LLM Decision-making with Conformal Prediction
- **Authors:** Harit Vishwakarma, Alan Mishler, Thomas Cook, Niccolo Dalmasso, Natraj Raman, Sumitra Ganesh
- **Venue:** ICML 2025 (also ICLR 2025 workshop)
- **Link:** https://arxiv.org/abs/2501.00555
- **TL;DR:** CROQ revises a multiple-choice question by *removing* options outside the CP set and re-asking — analogous to test-taking elimination. CP-OPT learns conformity scores end-to-end to minimize expected set size. Tested on MMLU, ToolAlpaca, TruthfulQA.
- **Relevance:** **CoT-Tool** — DIRECT precedent: ToolAlpaca routing via CP. **Threat-level: high** for the tool-routing project.
- **Gap:** Static MCQ-style routing; no dynamic agent loops; no cost / latency budget per tool.

### 2.8 Wang et al. — TECP
- **Title:** TECP: Token-Entropy Conformal Prediction for LLMs
- **Authors:** Wang et al.
- **Venue:** Mathematics (MDPI) 2025
- **Link:** https://www.mdpi.com/2227-7390/13/20/3351
- **TL;DR:** Conformity score = token-entropy of sampled generations; integrates with split CP to give finite-sample coverage on open-ended generation.
- **Relevance:** **CoT-CP** — token-entropy is a candidate per-step score; useful baseline.
- **Gap:** Episodic (full-sequence) entropy; no stepwise structure.

### 2.9 Han et al. — Multi-LLM Adaptive Conformal Inference (MACI)
- **Title:** Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses
- **Authors:** Han et al. (Yonsei MLAI)
- **Venue:** arXiv 2602.01285 (2026) — note: posted under future-arXiv id; actual ACL/NeurIPS submission status pending
- **Link:** https://arxiv.org/abs/2602.01285
- **TL;DR:** Reformulates CP for LLM factuality as multiplicative filtering on claim scores from an *ensemble* of LLMs; group-conditional calibration preserves validity while ensemble scoring boosts retention.
- **Relevance:** **CoT-Tool** (ensemble of tool-LMs); **CG-VL** (could be ensembled VLMs).
- **Gap:** No reasoning structure, no retrieval / vision grounding.

### 2.10 Zhao et al. — Domain-Shift-Aware CP for LLMs
- **Title:** Domain-Shift-Aware Conformal Prediction for Large Language Models
- **Authors:** Zhao et al.
- **Venue:** arXiv 2510.05566 (2025)
- **Link:** https://arxiv.org/abs/2510.05566
- **TL;DR:** Embeds prompts in a sentence-embedding space and applies non-exchangeable CP by reweighting calibration samples by proximity to the test prompt; restores coverage under domain shift.
- **Relevance:** **CoT-Tool** (different tool ecosystems = covariate shift); **Common**.
- **Gap:** Sentence-level proximity is a shallow proxy; no per-step or multimodal extension.

### 2.11 Wang et al. — Online Reasoning Calibration
- **Title:** Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning
- **Authors:** anonymous
- **Venue:** arXiv 2604.01170 (preprint)
- **Link:** https://arxiv.org/abs/2604.01170
- **TL;DR:** Test-time training for the conformity score on the fly during reasoning, generalizing CP to OOD reasoning tasks.
- **Relevance:** **CoT-CP** — test-time-trained scores are a competitor to fixed step-level scores.
- **Gap:** Requires gradient updates at inference; CoT-CP can position as zero-update.

### 2.12 Differentiable Conformal Training for LLM Reasoning Factuality
- **Title:** Differentiable Conformal Training for LLM Reasoning Factuality
- **Authors:** anonymous
- **Venue:** arXiv 2604.20098 (preprint)
- **Link:** https://arxiv.org/abs/2604.20098
- **TL;DR:** Differentiates through the CP procedure to *train* the LLM to output well-calibrated reasoning; applies specifically to math/logic chains.
- **Relevance:** **CoT-CP** — strongly overlapping; threat-level: high. The user's CoT-CP must differ either by being post-hoc / step-level or by adding human-in-the-loop tools.
- **Gap:** Requires retraining; expensive; the user's split-CP-at-inference framing is cheaper.

### 2.13 CoVeR — Conformal Calibration for Versatile Next-Token Prediction
- **Title:** CoVeR: Conformal Calibration for Versatile and Reliable Autoregressive Next-Token Prediction
- **Authors:** anonymous
- **Venue:** arXiv 2509.04733 (2025)
- **Link:** https://arxiv.org/abs/2509.04733
- **TL;DR:** Models conformal scores of token-sequences sharing a step-prefix as an empirical distribution; clusters tokens with similar score distributions for step-level calibration.
- **Relevance:** **CoT-CP** — CLOSEST competitor to the planned step-level CoT-CP; threat-level: very high.
- **Gap:** Token-cluster level, not semantic-step level (sentences / proof steps); CoT-CP could differ by using semantic decomposition (akin to Mohri-Hashimoto subclaims) at each *reasoning step*.

### 2.14 Beyond Surface Statistics — Internal-Representation CP
- **Title:** Beyond Surface Statistics: Robust Conformal Prediction for LLMs via Internal Representations
- **Link:** https://arxiv.org/html/2604.16217
- **TL;DR:** Uses LLM hidden states (not logits) as conformity-score features for robustness under shift.
- **Relevance:** **CoT-CP, CoT-Tool**.
- **Gap:** Closed-source/API LLMs cannot use internal states.

### 2.15 SAFER — Risk-Constrained Sample-then-Filter
- **Title:** SAFER: Risk-Constrained Sample-then-Filter in Large Language Models
- **Link:** https://arxiv.org/html/2510.10193v3
- **TL;DR:** Two-stage: sample multiple LLM outputs, filter via CP-calibrated risk; controls the *expected* hallucination rate.
- **Relevance:** **CG-VL** (VLM analog), **CoT-Tool** (filter bad tool calls).

### 2.16 CP for NLP Survey (TACL 2024)
- **Title:** Conformal Prediction for Natural Language Processing: A Survey
- **Authors:** Margarida Campos, Antonio Farinhas, Chrysoula Zerva, Mario Figueiredo, Andre Martins
- **Venue:** TACL 2024
- **Link:** https://aclanthology.org/2024.tacl-1.82/
- **TL;DR:** Comprehensive survey of CP-for-NLP; categorizes by task and exchangeability assumption. Mandatory citation.

### 2.17 Stewart et al. — Thought Calibration ⚠️ HIGHEST-THREAT 2025 ADDITION
- **Title:** Thought Calibration: Efficient and Confident Test-Time Scaling
- **Authors:** Stewart et al.
- **Venue:** EMNLP 2025 (also arXiv 2505.18404)
- **Links:** https://aclanthology.org/2025.emnlp-main.722.pdf , https://arxiv.org/pdf/2505.18404
- **TL;DR:** Applies CP to DeepSeek-R1 reasoning trajectories to decide *when to stop thinking*. Achieves up to 60% reduction in thinking tokens at full performance on in-distribution; 20% reduction on DeepSeek-distilled Qwen-2.5 32B over math/science. Explicitly frames stopping rule via finite-sample coverage.
- **Relevance:** **CoT-CP** — *single highest-threat paper* for CoT-CP. Same problem (calibrated test-time scaling), same era (May–Sep 2025), overlapping benchmarks (R1 / R1-distill family on math reasoning).
- **Differentiation from CoT-CP:** (i) Thought Calibration is *online stopping-rule* CP — decides when to terminate a single reasoning trace, while CoT-CP is *post-hoc selective filtering* over completed trajectories; (ii) CoT-CP introduces score-family Pareto (Theorem 2 — lp/PRM/SC ladder) which Thought Calibration does not; (iii) CoT-CP introduces empirical-PMF weighted CP for discrete OOD shift (Theorem 3) which Thought Calibration does not address; (iv) CoT-CP covers 11 models × 7 datasets vs Thought Calibration's narrower R1-family scope. Must explicitly cite and benchmark against in §3 and §5.
- **Gap that remains:** No multi-score Pareto, no OOD coverage analysis, no PRM as a mid-cost operating point.

### 2.18 Conformal Prediction Beyond the Seen — Missing Mass for Generative Models
- **Title:** Conformal Prediction Beyond the Seen: A Missing Mass Perspective for Uncertainty Quantification in Generative Models
- **Venue:** NeurIPS 2025
- **Link:** https://neurips.cc/virtual/2025/poster/118606
- **TL;DR:** Brings the *missing-mass* (Good-Turing) perspective to CP for generative models. Bounds the probability of unseen outputs in the prediction set, addressing a gap that standard exchangeability-based CP cannot handle for open-ended generation.
- **Relevance:** **Common** — important theoretical addition for any open-ended generation CP. Particularly relevant for CoT-CP where "novel reasoning traces" at test time may not be represented in calibration.
- **Gap:** Missing-mass framing assumes a fixed atomic output space; reasoning traces are compositional / unbounded — not directly applicable but a useful framing.

### 2.19 Paraphrase-Robust Conformal Prediction
- **Title:** Paraphrase-Robust Conformal Prediction for Reliable LLM Uncertainty Quantification
- **Venue:** ICLR 2026 submission (under review)
- **Link:** https://openreview.net/forum?id=Uf04r8gDn7
- **TL;DR:** Builds CP framework that is robust to paraphrase-induced score variance: introduces a paraphrase-aware nonconformity score that ensures valid coverage across semantically equivalent prompts.
- **Relevance:** **CoT-CP** — directly relevant to our `SX_paraphrase_cross_dataset` experiment. Should be cited as motivation for paraphrase-robustness experiments and benchmarked.
- **Gap:** Sentence-level paraphrase, not multi-step reasoning paraphrase; no PRM/SC integration.

### 2.20 An Empirical Study of CP in LLM with ASP Scaffolds
- **Title:** An Empirical Study of Conformal Prediction in LLM with ASP Scaffolds for Robust Reasoning
- **Venue:** arXiv 2503.05439 (March 2025)
- **Link:** https://arxiv.org/html/2503.05439v1
- **TL;DR:** Combines Answer Set Programming (ASP) scaffolds with Conformal Language Modeling to provide statistical guarantees on multi-step reasoning correctness. Evaluates on logical reasoning benchmarks.
- **Relevance:** **CoT-CP** — alternative architecture (symbolic scaffold + CP). Does not compete directly because it requires ASP encoding, but worth citing as evidence that "CP for multi-step reasoning" is an active area.
- **Gap:** Restricted to formally encodable reasoning; no support for free-form math/science CoT.

### 2.21 Unsupervised Conformal Prediction (UCP)
- **Title:** Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty
- **Venue:** arXiv 2509.23002 (September 2025)
- **Link:** https://arxiv.org/html/2509.23002
- **TL;DR:** Operates without labels: uses bootstrap calibration + conformal alignment to reconcile heterogeneous modalities. Provides distribution-free finite-sample guarantees and reports strong gains in hallucination detection / factuality.
- **Relevance:** **CG-VL** (label-free calibration is essential for VLM hallucination where ground-truth grounding is hard); **CoT-Tool** (tool-trigger calibration without labels).
- **Gap:** Does not handle reasoning structure; complementary not competing for CoT-CP.

### 2.22 CP: A Data Perspective (ACM Computing Surveys)
- **Title:** Conformal Prediction: A Data Perspective
- **Venue:** ACM Computing Surveys 2025
- **Link:** https://dl.acm.org/doi/10.1145/3736575
- **TL;DR:** Companion survey to Campos+ TACL 2024 with stronger emphasis on data-centric considerations (calibration set design, distribution shift, label noise). Mandatory secondary survey citation.

---

## 3. CP for Hallucination / Factuality (deep)

### 3.1 Mohri & Hashimoto — Conformal Factuality
- **Title:** Language Models with Conformal Factuality Guarantees
- **Authors:** Christopher Mohri, Tatsunori Hashimoto
- **Venue:** ICML 2024
- **Link:** https://arxiv.org/abs/2402.10978
- **TL;DR:** Decomposes an LM response into atomic subclaims, scores each (e.g., self-eval probability of correctness), and runs split CP to remove subclaims below a calibrated threshold. Formalizes correctness as set-coverage over an entailment back-off, giving a *high-probability factuality* guarantee. Foundational paper.
- **Relevance:** **CG-VL** (DIRECT analog: replace "claim correctness probability" with PMI grounding score), **CoT-CP** (subclaims -> reasoning steps).
- **Gap:** Single-modal; treats every claim as i.i.d. and exchangeable; ignores claim *dependencies* (which CG-VL with image grounding and CoT-CP with logical chain need).

### 3.2 Cherian, Gibbs, Candes — Enhanced LLM Validity
- **Title:** Large Language Model Validity via Enhanced Conformal Prediction Methods
- **Authors:** John J. Cherian, Isaac Gibbs, Emmanuel J. Candes
- **Venue:** NeurIPS 2024
- **Link:** https://arxiv.org/abs/2406.09714
- **TL;DR:** Two extensions to Mohri-Hashimoto: (i) generalizes Gibbs-Candes conditional conformal to issue *adaptively weaker* guarantees per topic to preserve utility; (ii) uses differentiation-through-CP to optimize the scoring function. Demos on biographies and medical QA.
- **Relevance:** **CG-VL** (per-image conditional guarantee: hallucinations cluster by visual category); **CoT-CP** (per-question-type adaptive guarantee).
- **Gap:** Designed for atomic-claim filtering; coherence and modality not handled.

### 3.3 Yadkori et al. — Mitigating LLM Hallucinations via Conformal Abstention
- **Title:** Mitigating LLM Hallucinations via Conformal Abstention
- **Authors:** Yasin Abbasi-Yadkori, Ilja Kuzborskij, David Stutz, Andras Gyorgy, Adam Fisch, Arnaud Doucet, Iuliya Beloshapka, Wei-Hung Weng, Yao-Yuan Yang, Csaba Szepesvari, Ali Eslami, Arnaud Doucet
- **Venue:** ICLR 2024 / arXiv 2405.01563
- **Link:** https://arxiv.org/abs/2405.01563
- **TL;DR:** Use the LLM to self-evaluate similarity among sampled responses; CP calibrates an abstention threshold so that hallucination rate is bounded with prob 1-alpha. Two finite-sample guarantees: participation-rate bound + conditional correctness bound.
- **Relevance:** **CoT-Tool** (abstention -> "ask user / use a different tool"), **CG-VL** (abstain on un-groundable image questions).
- **Gap:** Treats response monolithically; no per-step / per-claim breakdown.

### 3.4 Adaptive Conformal Prediction for LLM Factuality
- **Title:** Adaptive Conformal Prediction for Improving Factuality of Generations by Large Language Models
- **Link:** https://arxiv.org/html/2604.13991
- **TL;DR:** Prompt-adaptive conformal score transformation for long-form factuality; achieves both marginal and improved conditional coverage.
- **Relevance:** **CG-VL, CoT-CP**.

### 3.5 Geometry-Calibrated Conformal Abstention
- **Title:** Geometry-Calibrated Conformal Abstention for Language Models
- **Link:** https://arxiv.org/html/2604.27914
- **TL;DR:** Uses geometric structure of response embeddings (clusters of semantically similar generations) as the conformity signal for abstention.
- **Relevance:** **CoT-Tool, CG-VL** (geometric structure of vision-language embeddings is a natural conformity score).

### 3.6 SelectLLM — Calibrating LLMs for Selective Prediction
- **Title:** SelectLLM: Calibrating LLMs for Selective Prediction: Balancing Coverage and Risk
- **Venue:** OpenReview 2025
- **Link:** https://openreview.net/forum?id=JJPAy8mvrQ
- **TL;DR:** End-to-end fine-tuned selective predictor that integrates CP-style risk constraints during training.
- **Relevance:** **CoT-Tool** baseline.

---

## 4. CP for RAG / Retrieval (full)

### 4.1 Li et al. — TRAQ
- **Title:** TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction
- **Authors:** Shuo Li, Sangdon Park, Insup Lee, Osbert Bastani
- **Venue:** NAACL 2024
- **Link:** https://arxiv.org/abs/2307.04642
- **TL;DR:** First end-to-end statistical correctness guarantee for RAG: CP on the *semantic* level guarantees the prediction set contains a semantically-correct answer; Bayesian optimization minimizes set size. Reduces sets by 16.2% over baseline CP.
- **Relevance:** **CoT-Tool** (retrieval is one tool; TRAQ is the prototype for tool-conformity), **CG-VL** baseline if VLM uses retrieved captions.
- **Gap:** Single-step retrieval; no multi-tool routing; no vision modality.

### 4.2 Conformal-RAG
- **Title:** Conformal-RAG: Conformal Retrieval-Augmented Generation with Statistical Guarantees on Sub-Claims
- **Venue:** SIGIR 2025 (also discussed as 2025 work)
- **Link:** https://arxiv.org/abs/2506.20978 (related: Response Quality Assessment for RAG via Conditional Conformal Factuality)
- **TL;DR:** CP framework specialized to RAG that uses retrieved evidence as the *context* for the conformity scoring function; group-conditional coverage across sub-domains; retains 60% more high-quality sub-claims than vanilla Mohri-Hashimoto-on-RAG.
- **Relevance:** **CoT-Tool** (RAG as a tool); **CG-VL** (image-as-evidence is the visual analog of retrieved-document-as-evidence).
- **Gap:** Text-only retrieval; no multimodal; no chain-of-thought.

### 4.3 Conformal Factuality for RAG — Robustness Audit
- **Title:** Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights
- **Link:** https://arxiv.org/abs/2603.16817
- **TL;DR:** Empirical/theoretical audit showing failure modes of conformal-factuality on RAG (claim independence violated by retrieval); proposes new metrics.
- **Relevance:** **Common** — important caveat to cite; supports the user's argument that step-level / coherence-aware CP is needed.

### 4.4 Principled Context Engineering for RAG via CP
- **Link:** https://link.springer.com/chapter/10.1007/978-3-032-21300-6_45
- **TL;DR:** Uses CP to certify which retrieved chunks should be included in the LLM's context; statistical guarantee on long-context degradation.
- **Relevance:** **CoT-Tool** (retrieval-as-tool); **Common**.

### 4.5 CONFLARE
- **Title:** CONFLARE: CONFormal LArge language model REtrieval
- **TL;DR:** CP on retrieval scores to certify a sufficient retrieval set.
- **Relevance:** **CoT-Tool**.

---

## 5. CP under Distribution Shift / Weighted CP (LLM-relevant)

### 5.1 Tibshirani et al. (NeurIPS 2019) — Weighted CP under Covariate Shift (foundational; section 1)

### 5.2 Barber et al. (2022/2023) — Conformal Prediction Beyond Exchangeability (foundational; section 1)

### 5.3 Domain-Shift-Aware CP for LLMs (see 2.10)

### 5.4 Robust Conformal Prediction via LLM Internal Representations (see 2.14)

### 5.5 Conformal Prediction under Levy-Prokhorov Distribution Shifts
- **Link:** https://arxiv.org/html/2502.14105v2
- **TL;DR:** Robustness guarantees for CP under bounded local + global perturbations via Levy-Prokhorov metric.
- **Relevance:** **Common** theory; relevant for CG-VL where image perturbations are local.

### 5.6 Multi-Source Conformal Inference Under Distribution Shift
- **Venue:** ICML 2024
- **TL;DR:** CP that pools multiple shifted calibration sources; relevant when the user has tool-specific calibration data.
- **Relevance:** **CoT-Tool**.

---

## 6. CP for VLM / Multimodal (full — sparse but important)

### 6.1 Pinto et al. — Are Foundation Models for Computer Vision Good Conformal Predictors?
- **Title:** Are Foundation Models for Computer Vision Good Conformal Predictors?
- **Authors:** Leo Fillioux, Julio Silva-Rodriguez, Ismail Ben Ayed, Paul-Henry Cournede, Maria Vakalopoulou, Stergios Christodoulidis, Jose Dolz
- **Venue:** arXiv 2412.06082 (Dec 2024) - workshop paper
- **Link:** https://arxiv.org/abs/2412.06082
- **TL;DR:** Empirical study of CP on 17 vision foundation models (incl. CLIP, DINO) across multiple datasets/CP methods (LAC, APS, RAPS). Finds vision transformers conformalize well; few-shot adaptation improves conformal efficiency; *standard temperature calibration hurts adaptive CP*. Establishes that APS does not violate marginal coverage.
- **Relevance:** **CG-VL** — DIRECT empirical baseline for CP on VLMs; the user's PMI score must be benchmarked against APS/LAC/RAPS on CLIP-style models.
- **Gap:** Closed-set classification only — no open-ended generation, no PMI/grounding, no hallucination-specific definition of correctness.

### 6.2 Conformal Prediction for Zero-Shot Models (Conf-OT)
- **Title:** Conformal Prediction for Zero-Shot Models
- **Authors:** Julio Silva-Rodriguez, Ismail Ben Ayed, Jose Dolz
- **Venue:** CVPR 2025
- **Link:** https://arxiv.org/abs/2505.24693
- **TL;DR:** Conf-OT is a transductive transfer-learning conformal procedure over the union of calibration + query sets; up to 20% set-size improvement on zero-shot CLIP, 15x faster than transductive baselines.
- **Relevance:** **CG-VL** — strong baseline for VLM zero-shot CP.
- **Gap:** Classification only; no generation hallucination.

### 6.3 Conformal Predictions for Human Action Recognition with VLMs
- **Title:** Conformal Predictions for Human Action Recognition with Vision-Language Models
- **Venue:** arXiv 2502.06631 (2025)
- **Link:** https://arxiv.org/abs/2502.06631
- **TL;DR:** CP for VLM-based HAR (Kinetics400, UCF101, HMDB51). Tunes softmax temperature to control conformal-set distribution without extra calibration data.
- **Relevance:** **CG-VL** — VLM CP precedent on a discriminative video task.
- **Gap:** Not generative; not about hallucination; but shows VLM CP works.

### 6.4 Conformal Alignment (Gui, Jin, Ren)
- **Title:** Conformal Alignment: Knowing When to Trust Foundation Models with Guarantees
- **Authors:** Yu Gui, Ying Jin, Zhimei Ren
- **Venue:** NeurIPS 2024
- **Link:** https://arxiv.org/abs/2405.10301
- **TL;DR:** Trains an alignment predictor on reference data with ground-truth alignment status; selects new units with predicted scores above a data-dependent threshold so that on average a prescribed fraction of selected units truly align (FDR-style guarantee). Demonstrated on QA + radiology report generation.
- **Relevance:** **CG-VL** (radiology = VLM-style image-conditioned generation), **CoT-Tool** (alignment = correct-tool-use).
- **Gap:** Selection-FDR rather than per-claim coverage; no PMI / grounding-based score.

### 6.5 CP Survey for Multimodal Foundation Models
- **Title:** Conformal Prediction in the Age of Multimodal Foundation Models: A Survey
- **Link:** https://link.springer.com/chapter/10.1007/978-3-032-15120-9_13
- **TL;DR:** Recent survey - mandatory citation for CG-VL.

### 6.6 Grounding Language with Vision: Conditional Mutual Information Decoding (NeurIPS 2025)
- **Title:** Grounding Language with Vision: A Conditional Mutual Information Calibrated Decoding Strategy for Reducing Hallucinations in LVLMs
- **Link:** https://arxiv.org/abs/2505.19678
- **TL;DR:** C-PMI bi-level optimization of decoding to maximize conditional mutual information between generated tokens and image, mitigating hallucinations.
- **Relevance:** **CG-VL** — closest non-CP method using PMI for VLM hallucination. *Threat-level: high* — uses the same conformity-score idea (PMI/CMI) but for decoding, not for CP-style prediction sets / risk control. CG-VL must position as: PMI-as-conformity-score with formal CRC guarantee, not just decoding heuristic.
- **Gap:** Heuristic decoding, NO statistical guarantee; no calibration set; no CP.

### 6.7 Multi-Modal Hallucination Control by Visual Information Grounding (M3ID)
- **Title:** Multi-Modal Hallucination Control by Visual Information Grounding
- **Venue:** CVPR 2024
- **Link:** https://arxiv.org/abs/2403.14003
- **TL;DR:** M3ID amplifies mutual information between image and generated tokens during decoding.
- **Relevance:** **CG-VL** — mutual-information-as-grounding precedent (decoding-time, not CP).

### 6.8 SafePath — CP for LLM-Based Autonomous Navigation
- **Title:** SafePath: Conformal Prediction for Safe LLM-Based Autonomous Navigation
- **Link:** https://arxiv.org/html/2505.09427v1
- **TL;DR:** CP-certified path generation in autonomous driving via LLM planner.
- **Relevance:** **CoT-Tool** (planner-as-tool, formal guarantees).

### 6.9 KnowNo — Robots that Ask for Help
- **Title:** Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners
- **Authors:** Allen Z. Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh, Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake Varley, Zhenjia Xu, Dorsa Sadigh, Andy Zeng, Anirudha Majumdar
- **Venue:** CoRL 2023
- **Link:** https://arxiv.org/abs/2307.01928
- **TL;DR:** Robot LLM planner formulated as MCQ; CP gives a prediction set of plausible actions; if not a singleton, the robot asks for human help. Provides task-completion guarantees while minimizing help.
- **Relevance:** **CoT-Tool** — DIRECT precedent for "if CP set is not a singleton, escalate" routing logic; threat-level: high. CoT-Tool must extend to multi-step agents and cost-aware choice among many tools, not just MCQ-over-skills.

### 6.10 Conformal Constrained Policy Optimization for Cost-Effective LLM Agents
- **Link:** https://arxiv.org/html/2511.11828v2
- **TL;DR:** Selects among LLM agents to satisfy CP coverage while minimizing average cost.
- **Relevance:** **CoT-Tool** — closest competitor for *cost-aware* tool routing. Threat-level: very high.
- **Gap:** Treats agents as black-box LLMs; no per-step or per-tool conformity decomposition.

---

## 7. Adjacent — LLM Calibration / Abstention without CP (brief baselines)

1. **Kuhn, Gal, Farquhar (ICLR 2023).** *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* https://arxiv.org/abs/2302.09664 - Cluster sampled outputs by entailment; entropy over clusters. Backbone uncertainty signal.
2. **Farquhar, Kossen, Kuhn, Gal (Nature 2024).** *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature.
3. **Nikitin, Kossen, Gal, Marttinen (NeurIPS 2024).** *Kernel Language Entropy.* https://arxiv.org/abs/2405.20003 - Generalizes semantic entropy via kernels.
4. **Lin, Trivedi, Sun (TMLR 2024).** *Generating with Confidence: Uncertainty Quantification for Black-box LLMs.* https://arxiv.org/abs/2305.19187 - Practical black-box scores: similarity, p(True), self-eval.
5. **Tian, Mitchell, et al. (EMNLP 2023).** *Just Ask for Calibration.* Shows verbalized confidence is poor but improvable.
6. **Xiong et al. (ICLR 2024).** *Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation.* https://github.com/MiaoXiong2320/llm-uncertainty
7. **Kossen et al. (2024).** *Semantic Entropy Probes.* Cheap probe-based hallucination detection. https://arxiv.org/abs/2406.15927
8. **Survey: Shorinwa et al. (2024).** *A Survey on Uncertainty Quantification of Large Language Models: Taxonomy, Open Research Challenges, and Future Directions.* https://arxiv.org/abs/2412.05563 - Mandatory citation.
9. **Survey: Geng et al. (March 2025).** *Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey.* https://arxiv.org/abs/2503.15850
10. **Survey: ACL 2025.** *A Survey of Uncertainty Estimation Methods on Large Language Models.* https://aclanthology.org/2025.findings-acl.1101/
11. **Know Your Limits: A Survey of Abstention in LLMs (TACL 2025).** https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00754/131566 - Abstention taxonomy.
12. **Trust or Escalate (ICLR 2025).** LLM-judge with statistical trust thresholds; relevant to CoT-Tool escalation.

---

## 8. Gap Analysis — What the User's 3 Projects Can Newly Contribute

The CP-for-LLM landscape splits into three roughly parallel streams: (i) **set-of-generations** CP (Quach 2024, MACI), (ii) **claim-decomposition** CP (Mohri-Hashimoto 2024, Cherian-Gibbs-Candes 2024, Conformal-RAG 2025), and (iii) **action / abstention** CP (KnowNo 2023, Conformal Abstention 2024, Prune 'n Predict 2025, Conformal Constrained Policy 2025). Stream (ii) provides the strongest factuality guarantees but assumes claims are exchangeable and structurally independent — an assumption Conformal Language Model Reasoning with Coherent Factuality (Rubin-Toles et al., ICLR 2025) is the *first* paper to break, by introducing deducibility graphs. CoVeR (2025) and Differentiable Conformal Training (2026) target step-level conformity but at the token-cluster / training-time level rather than the semantic-step / inference-time level.

**CoT-CP's contribution** is to combine (a) Mohri-Hashimoto-style semantic decomposition into reasoning *steps* (not arbitrary claims) with (b) coherence-aware CRC across the chain (extending Rubin-Toles 2025) and (c) inference-time, training-free calibration (unlike Differentiable Conformal Training). The closest threats are CoVeR (step-level but token-cluster, not semantic) and Conformal Language Model Reasoning with Coherent Factuality (semantic but not specifically focused on CoT exploration / sampling).

**CoT-Tool's contribution** is cost-aware multi-tool routing with CRC. KnowNo (2023) and Prune 'n Predict / CROQ (2025) handle tool selection as MCQ; Conformal Constrained Policy Optimization (2025) adds cost. None combine *per-tool* conformity scores, *cost budget* as a CRC loss, AND *retrieval-as-tool* in a multi-step agent loop. This is the open white space.

**CG-VL's contribution** is the first VLM-hallucination CP that uses **PMI (or conditional MI) as the conformity score** with a formal CRC guarantee. The PMI/MI signal is well-validated as a *decoding heuristic* (M3ID, CVPR 2024; C-PMI Grounding, NeurIPS 2025) but has *never* been used as a conformity score with finite-sample coverage. Existing VLM-CP work (Pinto 2024, Conf-OT CVPR 2025, HAR-CP 2025) only handles closed-set classification, and Conformal Alignment (NeurIPS 2024) handles selection-FDR for radiology but uses a learned alignment predictor rather than an information-theoretic grounding score. The gap is concrete and defensible.

**Common theoretical contribution**: all three projects can extend CRC machinery in unified ways — non-exchangeable CP for distribution-shifted tool ecosystems / domains (citing Barber 2023, Domain-Shift-Aware CP 2025), and conditional CP for per-prompt / per-image guarantees (Cherian-Gibbs-Candes 2024).

**Note on threats (updated 2026-05-08):** the highest-overlap papers, in order, are now:
(1) **Thought Calibration** (Stewart et al., EMNLP 2025, §2.17) — *new highest threat to CoT-CP*: same problem (calibrated test-time stopping), same era, R1-distill overlap. CoT-CP must differentiate via score-family Pareto (Theorem 2) + discrete weighted CP (Theorem 3) + 11×7 model-dataset matrix.
(2) **Conformal Language Model Reasoning with Coherent Factuality** (Rubin-Toles et al., ICLR 2025) — graphical reasoning-CP; threatens CoT-CP theoretical neighbor.
(3) **CoVeR** (2509.04733, 2025) — token-cluster step CP; threatens CoT-CP.
(4) **Differentiable Conformal Training** (2604.20098, 2026 preprint) — train-time CP; CoT-CP differentiates as inference-time / training-free.
(5) **Conformal Constrained Policy Optimization** (2511.11828, 2025) — threatens CoT-Tool.
(6) **Prune 'n Predict / CROQ** (ICML 2025) — threatens CoT-Tool.
(7) **C-PMI Grounding Decoding** (NeurIPS 2025) — threatens CG-VL conceptually, non-CP.

Watch list: Paraphrase-Robust CP (ICLR 2026 submission, §2.19) for the cross-dataset paraphrase angle; UCP (§2.21) if label-free calibration becomes a deployment requirement.

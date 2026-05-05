# Literature Review — CoT-CP Application-Side Baselines & Adjacent Work

**Project**: CoT-CP (Conformal Chain-of-Thought) — step-level conformity scoring + Conformal Risk Control over reasoning traces, calibrated on PRM800K, evaluated on GSM8K / MATH-500 / AIME / GPQA-Diamond.
**Scope of this file**: application-side competitors (NOT CP-for-LLM theory; that is covered separately). Top-tier venues 2022–2026 with heavy 2024-2026 emphasis.
**Notation**: Threat = how close this work is to "step-level adaptive compute with calibrated guarantee". Differentiator = what CoT-CP brings that this paper does not.

---

## A. Test-Time Scaling / Inference Compute

### A1. OpenAI o1 System Card
- Authors: OpenAI (El-Kishky, Wong, McAleese, et al.)
- Venue/Year: arXiv 2412.16720 (Dec 2024)
- Link: https://arxiv.org/abs/2412.16720
- TL;DR: Reports the o1 series trained with large-scale RL to "think before responding" via long chain-of-thought. Establishes that test-time reasoning can be scaled by spending more tokens on internal deliberation; describes safety-via-deliberative-alignment.
- Threat: LOW. System paper, no CP, no statistical guarantees.
- Differentiator: CoT-CP gives a finite-sample, distribution-free correctness guarantee on the *output* of an o1-style chain.

### A2. OpenAI o3 / o4-mini System Card
- Authors: OpenAI
- Venue/Year: OpenAI tech report (Apr 2025)
- Link: https://openai.com/index/o3-o4-mini-system-card/
- TL;DR: Successor reasoning models with tool use and "thinking with images". Claims 20% fewer major errors than o1; SOTA on AIME 2024/2025.
- Threat: LOW. Black-box system; no public step-level instrumentation.
- Differentiator: Open, calibrated step gating that is model-agnostic.

### A3. DeepSeek-R1: Incentivizing Reasoning Capability via RL
- Authors: DeepSeek-AI (Guo, Yang, Zhang, et al.)
- Venue/Year: arXiv 2501.12948 (Jan 2025)
- Link: https://arxiv.org/abs/2501.12948
- TL;DR: Pure-RL ("R1-Zero") and multi-stage RL ("R1") training elicits emergent reasoning patterns (self-reflection, verification, dynamic strategy). Pass@1 71.0 on AIME, 86.7 with majority voting; matches OpenAI-o1.
- Threat: LOW. Training-side method; complements rather than competes with CP at inference.
- Differentiator: CoT-CP works on top of R1 to give per-query risk control without retraining.

### A4. Kimi k1.5: Scaling RL with LLMs
- Authors: Kimi Team (Moonshot AI)
- Venue/Year: arXiv 2501.12599 (Jan 2025)
- Link: https://arxiv.org/abs/2501.12599
- TL;DR: Multi-modal RL recipe explicitly *avoiding* MCTS, value functions, and PRMs. 77.5 AIME, 96.2 MATH-500. Long-CoT → short-CoT distillation.
- Threat: LOW. Argues PRMs are unnecessary; CoT-CP can still wrap their model and add guarantees.
- Differentiator: Calibrated control; their short-CoT is heuristic, not statistically sized.

### A5. Qwen3 Technical Report (incl. QwQ-32B reasoning lineage)
- Authors: Qwen Team, Alibaba
- Venue/Year: arXiv 2505.09388 (May 2025)
- Link: https://arxiv.org/abs/2505.09388
- TL;DR: Unifies thinking and non-thinking modes; QwQ-32B / Qwen3-Thinking are open SOTA reasoning models; the planned base for CoT-CP experiments.
- Threat: LOW. Base model.
- Differentiator: N/A — CoT-CP uses these as backbones.

### A6. Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Parameters
- Authors: Snell, Lee, Xu, Kumar
- Venue/Year: arXiv 2408.03314 (Aug 2024); ICLR 2025
- Link: https://arxiv.org/abs/2408.03314
- TL;DR: Compute-optimal test-time-compute strategy that adaptively chooses between revision and verifier search per-prompt. Shows compute can be 4× more efficient than uniform best-of-N.
- Threat: **HIGH**. Closest spiritual ancestor — adaptive compute conditional on prompt difficulty. But adaptation is heuristic (oracle-difficulty proxy), not calibrated.
- Differentiator: CoT-CP is *step-level* not prompt-level, and gives a CRC guarantee on the final-answer error rate.

### A7. Large Language Monkeys: Scaling Inference Compute with Repeated Sampling
- Authors: Brown, Juravsky, Ehrlich, et al. (Stanford Scaling Intelligence)
- Venue/Year: arXiv 2407.21787 (Jul 2024)
- Link: https://arxiv.org/abs/2407.21787
- TL;DR: Coverage scales log-linearly with #samples up to 10^4. Demonstrates monkey-typewriter style BoN with verifiers; SWE-bench-Lite 15.9 → 56% with 250 samples.
- Threat: MED. Pure parallel scaling baseline that CoT-CP must beat in compute-vs-accuracy.
- Differentiator: CoT-CP allocates samples *adaptively where uncertain*, not flat N.

### A8. Reasoning on a Budget: Survey of Adaptive/Controllable Test-Time Compute in LLMs
- Authors: Chen, et al.
- Venue/Year: arXiv 2507.02076 (2025)
- Link: https://arxiv.org/abs/2507.02076
- TL;DR: Survey grouping budget-aware reasoning into prompt-level, step-level, and token-level adaptive compute methods.
- Threat: LOW (survey). Useful for positioning.
- Differentiator: Survey explicitly notes the lack of statistical-guarantee approaches — gap CoT-CP fills.

### A9. Trust but Verify! A Survey on Verification Design for Test-Time Scaling
- Authors: Venktesh, et al.
- Venue/Year: arXiv 2508.16665 (Aug 2025)
- Link: https://arxiv.org/abs/2508.16665
- TL;DR: Taxonomy of generative vs. discriminative verifiers, prompt-based vs. fine-tuned, process vs. outcome.
- Threat: LOW. Survey.
- Differentiator: Identifies that no surveyed verifier provides distribution-free coverage — exact CoT-CP positioning.

---

## B. Process / Step-Level Verification

### B1. Let's Verify Step by Step (PRM800K)
- Authors: Lightman, Kosaraju, Burda, et al. (OpenAI)
- Venue/Year: arXiv 2305.20050; ICLR 2024
- Link: https://arxiv.org/abs/2305.20050
- TL;DR: Process supervision (PRM) > outcome supervision on MATH; releases PRM800K (800K step-level labels, 75K traces, 12K problems). 78% on MATH subset.
- Threat: MED. The dataset CoT-CP calibrates on; its score function is a candidate $s_t$.
- Differentiator: CoT-CP calibrates the PRM scores into distribution-free quantiles, addressing PRM mis-calibration.

### B2. Math-Shepherd: Verify and Reinforce LLMs Step-by-Step Without Human Annotations
- Authors: Wang, Li, Shao, et al.
- Venue/Year: arXiv 2312.08935; ACL 2024
- Link: https://arxiv.org/abs/2312.08935
- TL;DR: Auto-labels step-level rewards via MC rollouts of completion success; uses for both verification re-ranking and step-PPO. Mistral-7B GSM8K 77.9 → 89.1.
- Threat: MED. Provides one of the candidate score functions; an automatic alternative to PRM800K.
- Differentiator: Math-Shepherd scores are point estimates with no coverage guarantee; CoT-CP wraps them.

### B3. Improve Mathematical Reasoning by Automated Process Supervision (OmegaPRM)
- Authors: Luo, Wang, Wei, et al. (Google)
- Venue/Year: arXiv 2406.06592 (Jun 2024)
- Link: https://arxiv.org/abs/2406.06592
- TL;DR: Divide-and-conquer MCTS for collecting >1.5M PRM labels without humans. PRM + weighted SC raises Gemini-Pro on MATH-500 from 51 → 69.4.
- Threat: MED. Another auto-labeled PRM source for calibration.
- Differentiator: CoT-CP turns these labels into a calibrated quantile $\hat{q}$ rather than just supervised reward.

### B4. ProcessBench: Identifying Process Errors in Mathematical Reasoning
- Authors: Zheng, Zhang, Bai, et al. (Qwen)
- Venue/Year: arXiv 2412.06559 (Dec 2024)
- Link: https://arxiv.org/abs/2412.06559
- TL;DR: 3,400 expert-annotated competition/Olympiad problems; models must locate the earliest erroneous step. Reveals existing PRMs do not generalize beyond GSM8K/MATH.
- Threat: MED. Critical evaluation benchmark for any step-verification work — CoT-CP must report on it.
- Differentiator: CoT-CP can use *ProcessBench-style* errors as the loss in CRC.

### B5. PRMBench: Fine-Grained Benchmark for Process-Level Reward Models
- Authors: Song, et al.
- Venue/Year: arXiv 2501.03124 (Jan 2025)
- Link: https://arxiv.org/abs/2501.03124
- TL;DR: 6,216 problems, 83K step labels probing simplicity/soundness/sensitivity of PRMs.
- Threat: MED. Evaluation companion to ProcessBench.
- Differentiator: Same as B4.

### B6. Rewarding Progress: Scaling Automated Process Verifiers (PAVs)
- Authors: Setlur, Nagpal, Fisch, et al. (Google + CMU)
- Venue/Year: arXiv 2410.08146 (Oct 2024)
- Link: https://arxiv.org/abs/2410.08146
- TL;DR: Defines step reward as *progress under a separate prover policy*; PAVs give 8% accuracy + 1.5–5× compute efficiency over ORMs in test-time search.
- Threat: **HIGH**. Adaptive compute via PRM at the step level. But still no calibrated guarantee.
- Differentiator: CoT-CP's quantile gating gives "P(answer correct) ≥ 1−α" instead of just better expected accuracy.

### B7. The Lessons of Developing PRMs in Mathematical Reasoning
- Authors: Zhang, Zheng, Bai, et al. (Qwen)
- Venue/Year: arXiv 2501.07301 (Jan 2025)
- Link: https://arxiv.org/abs/2501.07301
- TL;DR: Identifies MC-estimation bias for PRM training; consensus filtering with LLM-as-a-judge fixes it. Releases SOTA open PRM.
- Threat: MED. Best-current-practice PRM that CoT-CP should use as score baseline.
- Differentiator: Calibration layer on top.

### B8. Free Process Rewards Without Process Labels (Implicit PRM)
- Authors: Yuan, Liu, et al. (PRIME-RL)
- Venue/Year: arXiv 2412.01981 (Dec 2024)
- Link: https://arxiv.org/abs/2412.01981
- TL;DR: An ORM trained as log-likelihood ratio is implicitly a PRM at every step. No step labels needed.
- Threat: MED. Provides another candidate $s_t$ that's cheap to compute.
- Differentiator: Implicit PRM is uncalibrated; CoT-CP calibrates it.

### B9. PRIME: Process Reinforcement through Implicit Rewards
- Authors: Cui, Yuan, et al.
- Venue/Year: arXiv 2502.01456 (Feb 2025)
- Link: https://arxiv.org/abs/2502.01456
- TL;DR: Online PRM updates from policy rollouts + outcome labels via implicit rewards.
- Threat: LOW. Training-side use of implicit PRM.
- Differentiator: Inference-time CP rather than training-time RL.

### B10. Process Reward Model with Q-Value Rankings (PQM)
- Authors: Li, et al.
- Venue/Year: arXiv 2410.11287 (Oct 2024)
- Link: https://arxiv.org/abs/2410.11287
- TL;DR: Re-frames PRM as Q-value ranking in an MDP using a comparative loss.
- Threat: MED. Yet another candidate score function.
- Differentiator: CoT-CP converts ranks → calibrated quantiles.

### B11. Generative Verifiers (GenRM)
- Authors: Zhang, Hosseini, Bansal, et al.
- Venue/Year: arXiv 2408.15240 (Aug 2024)
- Link: https://arxiv.org/abs/2408.15240
- TL;DR: Train verifier with next-token prediction jointly with solving; 73 → 93.4 on GSM8K.
- Threat: MED. Top verifier for BoN; uses test-time compute.
- Differentiator: Their re-ranking is heuristic; CoT-CP can call GenRM only at uncertain steps.

### B12. CriticGPT: LLM Critics Help Catch LLM Bugs
- Authors: McAleese, Pokorny, et al. (OpenAI)
- Venue/Year: arXiv 2407.00215 (Jun 2024)
- Link: https://arxiv.org/abs/2407.00215
- TL;DR: GPT-4-based critic catches more inserted bugs than human reviewers; preferred 80% of the time.
- Threat: LOW. Critic for code, not stepwise math.
- Differentiator: CoT-CP could use CriticGPT-style critic only when $s_t < \hat{q}$.

### B13. AlphaMath Almost Zero: Process Supervision Without Process
- Authors: Chen, Liu, et al.
- Venue/Year: arXiv 2405.03553 (May 2024)
- Link: https://arxiv.org/abs/2405.03553
- TL;DR: MCTS with a value model trained jointly with the LLM produces step-level supervision and step-beam-search at inference.
- Threat: MED. Step-level beam search is exactly the "branch on uncertainty" the user proposes — but uses a value model not a calibrated quantile.
- Differentiator: Value-based vs. quantile-based gating; CoT-CP gives coverage proof.

### B14. ReST-MCTS*: LLM Self-Training via Process-Reward-Guided Tree Search
- Authors: Zhang, Zhoubian, et al. (Tsinghua)
- Venue/Year: NeurIPS 2024 (arXiv 2406.03816)
- Link: https://arxiv.org/abs/2406.03816
- TL;DR: Tree-search RL infers process rewards from oracle final answers; uses traces for self-training.
- Threat: MED. Tree search with PRM is a CoT-CP competitor for "branch when uncertain".
- Differentiator: CoT-CP triggers branching by calibrated $\hat{q}$ instead of MCTS UCB heuristic.

### B15. Process Reward Models That Think
- Authors: Khalifa, et al.
- Venue/Year: arXiv 2504.16828 (Apr 2025)
- Link: https://arxiv.org/abs/2504.16828
- TL;DR: PRM that produces a CoT critique before scoring; better OOD generalization.
- Threat: MED. Strongest current PRM.
- Differentiator: Calibration on top.

### B16. AlphaProof / Olympiad-Level Formal Reasoning with RL
- Authors: DeepMind AlphaProof team
- Venue/Year: Nature (Nov 2025)
- Link: https://www.nature.com/articles/s41586-025-09833-y
- TL;DR: Lean-based formal prover + AlphaZero-style RL; IMO-2024 silver-medal-equivalent.
- Threat: LOW. Formal proof, not natural-language CoT.
- Differentiator: CoT-CP works on natural language; complementary.

---

## C. Self-Consistency and Ensembling

### C1. Self-Consistency Improves CoT (the original)
- Authors: Wang, Wei, Schuurmans, et al. (Google)
- Venue/Year: arXiv 2203.11171; ICLR 2023
- Link: https://arxiv.org/abs/2203.11171
- TL;DR: Sample N reasoning chains, marginalize via majority vote on final answer. +17.9 GSM8K.
- Threat: MED. Direct baseline; one candidate $s_t$ (vote-share at step t).
- Differentiator: CoT-CP gives statistical guarantee; SC has none and uses fixed N.

### C2. Universal Self-Consistency (USC)
- Authors: Chen, Aksitov, et al.
- Venue/Year: arXiv 2311.17311 (Nov 2023)
- Link: https://arxiv.org/abs/2311.17311
- TL;DR: Use the LLM itself to pick most-consistent answer in free-form domains.
- Threat: MED. Generalizes SC to non-math; relevant baseline.
- Differentiator: Same calibration gap as C1.

### C3. Adaptive-Consistency: Let's Sample Step by Step
- Authors: Aggarwal, Madaan, Yang, Mausam
- Venue/Year: EMNLP 2023 (arXiv 2305.11860)
- Link: https://arxiv.org/abs/2305.11860
- TL;DR: Stop sampling when posterior over majority answer is concentrated; 7.9× cheaper than SC.
- Threat: **HIGH**. Adaptive sample-count via stopping rule (Beta-binomial). The closest non-CP analog.
- Differentiator: Adaptive-Consistency is *prompt-level* and uses Bayesian posterior heuristics; CoT-CP is *step-level* and gives a frequentist coverage guarantee.

### C4. ESC: Escape Sky-High Cost — Early-Stopping Self-Consistency
- Authors: Li, et al.
- Venue/Year: arXiv 2401.10480 (Jan 2024)
- Link: https://arxiv.org/abs/2401.10480
- TL;DR: Stop SC sampling when answer entropy in a sliding window is zero; 67–84% sample reduction.
- Threat: **HIGH**. Sister work to C3.
- Differentiator: Same — heuristic stopping rule, no statistical guarantee, prompt-level only.

### C5. Soft Self-Consistency for Language Model Agents
- Authors: Wang, et al.
- Venue/Year: arXiv 2402.13212 (Feb 2024)
- Link: https://arxiv.org/abs/2402.13212
- TL;DR: Replace majority vote with continuous likelihood-weighted aggregation.
- Threat: MED. Uses log-prob as candidate $s_t$.
- Differentiator: No calibration.

### C6. Scalable Best-of-N via Self-Certainty
- Authors: Kang, et al.
- Venue/Year: arXiv 2502.18581 (Feb 2025)
- Link: https://arxiv.org/abs/2502.18581
- TL;DR: KL-divergence-from-uniform of model output as a self-certainty score for BoN selection.
- Threat: MED. Closest single-pass confidence proxy.
- Differentiator: Self-certainty is uncalibrated; CoT-CP turns it into a quantile.

### C7. Bridging Internal Probability and Self-Consistency
- Authors: Zhao, et al.
- Venue/Year: arXiv 2502.00511 (Feb 2025)
- Link: https://arxiv.org/abs/2502.00511
- TL;DR: Hybrid score combining model log-prob with consistency frequency for efficient SC.
- Threat: MED. Hybrid candidate $s_t$.
- Differentiator: Same — heuristic blend.

### C8. Confidence Improves Self-Consistency in LLMs
- Authors: Taubenfeld, et al.
- Venue/Year: arXiv 2502.06233 (Feb 2025)
- Link: https://arxiv.org/abs/2502.06233
- TL;DR: Use logit confidence to weight votes in SC.
- Threat: MED.
- Differentiator: Same.

---

## D. Step-Level Uncertainty / Confidence

### D1. Semantic Uncertainty: Linguistic Invariances for UQ in NLG
- Authors: Kuhn, Gal, Farquhar
- Venue/Year: ICLR 2023 (arXiv 2302.09664)
- Link: https://arxiv.org/abs/2302.09664
- TL;DR: Cluster sampled outputs by NLI-induced meaning; entropy over clusters detects hallucinations. (Nature 2024 follow-up scales it.)
- Threat: MED. The canonical sequence-level UQ baseline.
- Differentiator: Semantic entropy is uncalibrated; sequence-level not step-level.

### D2. Reasoning Models Better Express Their Confidence
- Authors: Yoon, Kim, et al.
- Venue/Year: arXiv 2505.14489 (May 2025)
- Link: https://arxiv.org/abs/2505.14489
- TL;DR: o1/R1-style models become *better calibrated* as CoT unfolds because backtracking gives confidence-update signal.
- Threat: MED. Suggests confidence inside CoT is meaningful — supports CoT-CP's premise.
- Differentiator: They observe calibration; CoT-CP enforces it via CRC.

### D3. CoT-UQ: Response-wise UQ via Chain-of-Thought
- Authors: Zhang, et al.
- Venue/Year: ACL Findings 2025 (arXiv 2502.17214)
- Link: https://arxiv.org/abs/2502.17214
- TL;DR: Extract uncertainty signals from CoT tokens to score the whole response.
- Threat: MED.
- Differentiator: Response-level scalar; CoT-CP gates per-step.

### D4. Uncertainty-Aware Step-Wise Verification with Generative Reward Models
- Authors: anon (arXiv 2502.11250, Feb 2025)
- Link: https://arxiv.org/abs/2502.11250
- TL;DR: CoT-based entropy of a generative verifier as a step-level UQ for verification.
- Threat: **HIGH**. Step-level + uncertainty + verification = the user's exact territory.
- Differentiator: Uses verifier entropy heuristically; CoT-CP applies CRC to give coverage.

### D5. Efficient Verification of LLM Reasoning Steps (UHeads)
- Authors: anon (arXiv 2511.06209, Nov 2025)
- Link: https://arxiv.org/abs/2511.06209
- TL;DR: Train tiny "uncertainty heads" (<10M params) on top of frozen LLM internal states for step verification.
- Threat: **HIGH**. Step-level UQ trained on auto-labels — direct competitor.
- Differentiator: UHead score is uncalibrated; CoT-CP wraps it with quantile threshold.

### D6. Entropy-Based Exploration Conduction for Multi-Step Reasoning (Entro-duction)
- Authors: anon (arXiv 2503.15848, Mar 2025)
- Link: https://arxiv.org/abs/2503.15848
- TL;DR: Adjust exploration depth using output-entropy and entropy-variance across consecutive steps.
- Threat: **HIGH**. Entropy-driven step branching.
- Differentiator: Threshold is tuned empirically per dataset, not calibrated.

### D7. More Bang for the Buck: PRM with Entropy-Driven Uncertainty (EDU-PRM)
- Authors: anon (arXiv 2503.22233, 2025)
- Link: https://arxiv.org/abs/2503.22233
- TL;DR: Identifies high-entropy tokens as "uncertainty anchors" to partition CoT into steps and branch there.
- Threat: **HIGH**. Step segmentation via entropy + branch selection.
- Differentiator: Heuristic anchor selection; no risk guarantee.

### D8. Making Slow Thinking Faster: Compressing CoT via Step Entropy
- Authors: anon (arXiv 2508.03346, 2025)
- Link: https://arxiv.org/abs/2508.03346
- TL;DR: Step-entropy metric as informational contribution; prune low-entropy steps.
- Threat: MED.
- Differentiator: Compression vs. CP gating.

### D9. MUR: Momentum Uncertainty-Guided Reasoning
- Authors: anon (OpenReview 2025)
- Link: https://openreview.net/forum?id=5fUVJ2cHid
- TL;DR: Momentum-style accumulation of token uncertainty triggers reasoning gates.
- Threat: MED.
- Differentiator: Same.

---

## E. CoT Decoding / Branching Strategies

### E1. Tree of Thoughts
- Authors: Yao, Yu, Zhao, et al.
- Venue/Year: NeurIPS 2023 (arXiv 2305.10601)
- Link: https://arxiv.org/abs/2305.10601
- TL;DR: BFS/DFS over thought-states with self-evaluation; classic deliberate-search baseline.
- Threat: MED. The "branch" half of CoT-CP.
- Differentiator: ToT branches always; CoT-CP branches conditionally with CP guarantee.

### E2. Graph of Thoughts
- Authors: Besta, Blach, et al.
- Venue/Year: AAAI 2024 (arXiv 2308.09687)
- Link: https://arxiv.org/abs/2308.09687
- TL;DR: General DAG over thoughts; +62% on sorting vs ToT, −31% cost.
- Threat: LOW.
- Differentiator: Same.

### E3. Self-Refine: Iterative Refinement with Self-Feedback
- Authors: Madaan, Tandon, et al.
- Venue/Year: NeurIPS 2023 (arXiv 2303.17651)
- Link: https://arxiv.org/abs/2303.17651
- TL;DR: Single LLM critiques and refines its own output iteratively; +20% across 7 tasks.
- Threat: MED. Refinement = a backtrack action CoT-CP can trigger.
- Differentiator: Self-Refine refines unconditionally; CoT-CP refines only when $s_t < \hat{q}$.

### E4. Chain-of-Verification (CoVe)
- Authors: Dhuliawala, Komeili, Xu, et al. (Meta)
- Venue/Year: ACL Findings 2024 (arXiv 2309.11495)
- Link: https://arxiv.org/abs/2309.11495
- TL;DR: Draft → plan verification questions → answer them independently → re-generate.
- Threat: MED. Heuristic verification scaffold.
- Differentiator: CoT-CP's verifier-trigger is calibrated.

### E5. CRITIC: Tool-Interactive Critiquing
- Authors: Gou, Shao, et al.
- Venue/Year: ICLR 2024 (arXiv 2305.11738)
- Link: https://arxiv.org/abs/2305.11738
- TL;DR: LLM critiques output and uses tools to verify; iterative correction.
- Threat: MED.
- Differentiator: Same.

### E6. Chain-of-Thought Reasoning Without Prompting (CoT-decoding)
- Authors: Wang, Zhou (Google DeepMind)
- Venue/Year: NeurIPS 2024 (arXiv 2402.10200)
- Link: https://arxiv.org/abs/2402.10200
- TL;DR: Look at top-k decoding paths; CoT paths correlate with higher answer-confidence margin.
- Threat: MED. Their answer-margin is a candidate $s_t$.
- Differentiator: CoT-CP turns the margin into a calibrated threshold.

### E7. Chain of Preference Optimization
- Authors: Zhang, et al.
- Venue/Year: NeurIPS 2024
- Link: https://proceedings.neurips.cc/paper_files/paper/2024/file/00d80722b756de0166523a87805dd00f-Paper-Conference.pdf
- TL;DR: Distill ToT preferences into a single CoT model via DPO at the step level.
- Threat: LOW.
- Differentiator: Training-side method.

### E8. Step-DPO: Step-Wise Preference Optimization for Long-Chain Reasoning
- Authors: Lai, Tian, et al.
- Venue/Year: arXiv 2406.18629 (Jun 2024)
- Link: https://arxiv.org/abs/2406.18629
- TL;DR: DPO at the first-incorrect-step granularity. Qwen2-72B → 70.8 MATH, 94.0 GSM8K.
- Threat: LOW. Training method.
- Differentiator: Inference-time CP.

### E9. Zero-Shot Verification-Guided CoT
- Authors: Ray Chowdhury, et al.
- Venue/Year: arXiv 2501.13122 (Jan 2025)
- Link: https://arxiv.org/abs/2501.13122
- TL;DR: Combine generation prob and verification prompt prob to score steps.
- Threat: MED.
- Differentiator: No calibration.

---

## F. Recent Step-Level Adaptive Computation (most-threatening)

### F1. Dynamic Early Exit in Reasoning Models (DEER)
- Authors: anon (arXiv 2504.15895, Apr 2025)
- Link: https://arxiv.org/abs/2504.15895
- TL;DR: At each "switch point" generates a trial answer + confidence; exits CoT early if confident. 19–80% token reduction, +0.3–5% accuracy.
- Threat: **HIGH**. Direct competitor — step-level adaptive exit driven by confidence.
- Differentiator: DEER's exit threshold is heuristic and per-model tuned; CoT-CP's $\hat{q}$ comes from CRC and yields a sequence-level guarantee.

### F2. Deep Think with Confidence (DeepConf)
- Authors: Fu, Zhao, et al. (Meta)
- Venue/Year: arXiv 2508.15260 (Aug 2025)
- Link: https://arxiv.org/abs/2508.15260
- TL;DR: Group-confidence (averaged token-confidence over span) filters low-quality CoT traces during/after generation. AIME-25 99.9%, −84.7% tokens.
- Threat: **HIGH**. The single closest paper to CoT-CP — span-level confidence gating.
- Differentiator: DeepConf threshold is empirical; CoT-CP delivers $P(\text{correct}) \geq 1-\alpha$ on any new prompt under exchangeability.

### F3. Speculative Chain-of-Thought (SCoT)
- Authors: anon (arXiv 2504.19095, Apr 2025)
- Link: https://arxiv.org/abs/2504.19095
- TL;DR: Small-model drafts CoT, large-model verifies at thought-level. 2.92× speedup.
- Threat: **HIGH**. Step-level speculative decoding for reasoning.
- Differentiator: Acceptance is heuristic; CoT-CP could replace it with CP-acceptance.

### F4. SpecReason: Inference-Time Speculative Reasoning
- Authors: anon (arXiv 2504.07891, Apr 2025)
- Link: https://arxiv.org/abs/2504.07891
- TL;DR: Lightweight model speculates intermediate steps; base model corrects. 1.4–3× faster, +0.4–9% accuracy.
- Threat: **HIGH**.
- Differentiator: Same.

### F5. ConfSpec: Confidence-Gated Step-Level Speculative Reasoning
- Authors: anon (arXiv 2602.18447, 2025)
- Link: https://arxiv.org/abs/2602.18447
- TL;DR: Confidence gates whether to accept speculated step.
- Threat: **HIGH**. Literally "branch on confidence at the step level".
- Differentiator: Threshold uncalibrated.

### F6. Step-Level Verifier-Guided Hybrid Test-Time Scaling
- Authors: anon (arXiv 2507.15512, Jul 2025)
- Link: https://arxiv.org/abs/2507.15512
- TL;DR: Hybrid of BoN + step-PRM-verifier; allocate compute to uncertain steps.
- Threat: **HIGH**.
- Differentiator: Verifier threshold heuristic.

### F7. Anytime Verified Agents (AVA): Adaptive Compute Allocation
- Authors: anon (OpenReview 2025)
- Link: https://openreview.net/forum?id=JMDCMf7mlF
- TL;DR: Calibrated UQ + value-of-information search expansion + selective verification cascades w/ early exit, under user budget.
- Threat: **HIGH**. Calibrated UQ for adaptive verification — sounds very close.
- Differentiator: AVA is per-agent-tool-call (not step-of-CoT) and does not invoke CRC for sequence-level correctness.

### F8. FlexiVe: Flexible Verification with Dynamic Allocation
- Authors: anon (2025)
- TL;DR: Verifier with fast/slow modes governed by Flexible Allocation of Verification Budget.
- Threat: MED.
- Differentiator: No CP layer.

### F9. Rethinking Optimal Verification Granularity (VG-Search)
- Authors: anon (arXiv 2505.11730, May 2025)
- Link: https://arxiv.org/abs/2505.11730
- TL;DR: How often should verifier be invoked? Adaptive granularity outperforms fixed.
- Threat: **HIGH**. The verification-granularity question is exactly CoT-CP's question.
- Differentiator: CoT-CP picks granularity via CRC objective rather than search.

### F10. EAGer: Entropy-Gated Generation Branching
- Authors: anon (2025)
- TL;DR: Online token-entropy gates whether to spawn new continuations.
- Threat: **HIGH**.
- Differentiator: Threshold heuristic; no guarantee.

### F11. HALT-CoT: Model-Agnostic Early Stopping for CoT
- Authors: anon (OpenReview 2025)
- Link: https://openreview.net/pdf?id=CX5c7C1CZa
- TL;DR: Entropy-based halting rule for CoT.
- Threat: MED.
- Differentiator: Heuristic.

### F12. Token-Budget-Aware LLM Reasoning (TALE)
- Authors: Han, et al.
- Venue/Year: ACL Findings 2025 (arXiv 2412.18547)
- Link: https://arxiv.org/abs/2412.18547
- TL;DR: LLM estimates needed tokens per problem; −67% tokens.
- Threat: MED. Per-prompt budget; not step-level.
- Differentiator: Prompt-level vs step-level; no guarantee.

### F13. Difficulty-Adaptive Reasoning (DiffAdapt)
- Authors: anon (arXiv 2510.19669)
- Link: https://arxiv.org/abs/2510.19669
- TL;DR: Probe predicts difficulty; selects strategy + budget.
- Threat: MED.
- Differentiator: Prompt-level.

### F14. Confidence-Aware Reasoning (CaR)
- Authors: anon (EMNLP-Industry 2025)
- TL;DR: Insert stop token when reasoning + answer confidence both exceed thresholds.
- Threat: MED.
- Differentiator: Heuristic threshold.

### F15. S2R: Teaching LLMs to Self-Verify and Self-Correct via RL
- Authors: anon (arXiv 2502.12853)
- Link: https://arxiv.org/abs/2502.12853
- TL;DR: RL with both outcome and process rewards trains self-verification.
- Threat: LOW. Training-side.
- Differentiator: Inference-time CP wrapper.

---

## G. HIGH-Threat Papers — Direct Positioning Required

These are the papers CoT-CP must explicitly distinguish itself from:

| # | Paper | Why threatening | What CoT-CP adds |
|---|---|---|---|
| **G1** | DeepConf (Fu+ 2025, F2) | Span-level confidence filtering on identical benchmarks (AIME, GPQA); 99.9% accuracy with 84% fewer tokens | **Distribution-free guarantee on final answer (CRC), not just empirical accuracy** |
| **G2** | DEER (F1) | Step-level early exit driven by confidence on reasoning models | Calibrated $\hat{q}$ from PRM800K vs. heuristic threshold |
| **G3** | Adaptive-Consistency / ESC (C3, C4) | Adaptive compute via stopping rule | Step-level (vs prompt-level) and frequentist coverage (vs Bayesian heuristic) |
| **G4** | Snell+ (A6) | Compute-optimal adaptive test-time scaling | Step-level adaptation + sequence-level CRC |
| **G5** | Rewarding Progress / PAVs (B6) | Step-level verifier-guided adaptive search | Calibration of PAV scores into coverage-controlled gates |

Honorable mentions also worth a paragraph each in related work: ConfSpec (F5), VG-Search (F9), UHeads (D5), AVA (F7), Uncertainty-Aware Step-Wise Verification (D4), Entro-duction (D6).

---

## H. Datasets & Benchmarks

| Benchmark | Size | Domain | Current SOTA (May 2026) | Use in CoT-CP |
|---|---|---|---|---|
| **GSM8K** | 8.5K grade-school math (Cobbe+ 2021, arXiv 2110.14168) | Arithmetic word problems | ~96–97% (frontier reasoning models) | Sanity check; calibration-train candidate |
| **MATH-500** | 500 from Hendrycks-MATH | Competition math (HS) | 99.2% (LongCat-Flash-Thinking); 96.2 Kimi-k1.5 | Primary CoT-CP eval |
| **AIME 2024 / 2025** | 30 problems each, integer answers | Olympiad-level | o4-mini, R1, Kimi-k1.5 ~70–86% pass@1 + voting | Primary CoT-CP eval |
| **GPQA-Diamond** | 198 expert MCQ (Rein+ 2023, arXiv 2311.12022) | Graduate physics/chem/bio | 93–94% (Gemini 3.1 Pro, GPT-5.5) | Primary CoT-CP eval |
| **OlympiadBench** | ~8K | Olympiad math + physics | ~70% (top reasoning models) | Optional eval for OOD |
| **USAMO 2025** | 6 problems | USA Math Olympiad — proofs | <30% all models, Gemini-2.5-Pro 25% | Stress test (most models trivial) |
| **Putnam-AXIOM** | 6,306 (1,051 originals × variants) | Putnam math | o3 51.5% original, drops 13pp under rewrites | Robustness eval |
| **PRM800K** | 800K step labels, 75K traces, 12K problems (Lightman+ 2023, arXiv 2305.20050) | MATH | — (training data) | **Calibration set** for CoT-CP |
| **ProcessBench** | 3,400 step-error-localization (Zheng+ 2024, arXiv 2412.06559) | Olympiad/competition | QwQ-32B-Preview ≈ GPT-4o; o1-mini ahead | Step-verifier eval |
| **PRMBench** | 6,216 problems × 83K step labels (Song+ 2025, arXiv 2501.03124) | Math | — | Step-verifier eval |
| **LiveCodeBench** | Rolling | Code | 91.7% Gemini-3-Pro-Preview | Optional (code reasoning extension) |
| **HMMT 2025 / BRUMO 2025** | 30 each | Math comp | DeepConf 99.9% AIME-25 implies near-saturation | DeepConf comparison set |

Bottom line for evaluation: **GSM8K + MATH-500 + AIME-2024/25 + GPQA-Diamond** are the right primary suite (matches DeepSeek-R1, Kimi, DeepConf, o-series reports) and PRM800K is the natural calibration source. ProcessBench is the natural evaluation of the *step-detector* component.

---

## I. Synthesis — The Gap CoT-CP Fills

The 2024-2026 literature converges on a clear pattern: people *know* that step-level confidence/entropy/PRM-score signals exist and *want* to use them for adaptive compute (DeepConf, DEER, EDU-PRM, Entro-duction, ConfSpec, AdaConsistency, ESC, Snell+, PAVs). Almost all of these methods, however, share two limitations:

1. **Heuristic thresholds**: confidence/entropy cutoffs are tuned per-model, per-dataset, with no formal guarantee on the resulting accuracy. When the test distribution shifts, the threshold is brittle (a fact ProcessBench and Putnam-AXIOM make explicit by showing PRM/verifier collapse on harder OOD problems).
2. **No coverage guarantee at the *sequence* level**: even when a per-step confidence is calibrated (Reasoning Models Better Express Their Confidence, Yoon+ 2025), no work propagates the per-step signal to a finite-sample, distribution-free statement about the final-answer correctness rate.

CoT-CP plants its flag exactly in this gap. By treating step-level scores ($s_t$ from self-consistency / PRM / log-prob) as conformity scores and applying **Conformal Risk Control** (Angelopoulos+ 2022), it converts any of these existing heuristics into a procedure with $P(\text{final answer correct}) \geq 1-\alpha$ under exchangeability with PRM800K-style calibration — the same formal contract that distribution-free CP gives in classification, lifted to multi-step reasoning. The step-level branching/verifier-trigger is a side-effect of the calibrated quantile and is therefore *adaptive without being heuristic*.

This positions CoT-CP cleanly against the HIGH-threat papers in Section G: every one of them is an instance of "use a step-level signal heuristically", and CoT-CP is the *first* method that turns that signal into a calibrated risk-controlled inference procedure with explicit step-level instrumentation. The closest theoretical-flavor competitors (Conformal Language Modeling, Mitigating LLM Hallucinations via Conformal Abstention, Coherent Factuality / DCF) operate at the *answer/claim* level and are covered by the parallel CP-for-LLM theory review — they do not control reasoning-step compute, leaving CoT-CP's specific contribution unclaimed.

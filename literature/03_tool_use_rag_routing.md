# Literature Review — Tool Use, Adaptive RAG, and Routing Baselines

> Coverage for **CoT-Tool — Conformal Tool-Use Trigger**. Focus: application-side baselines (tool routing, adaptive RAG, hallucination detection, calibrated routing). CP theory itself is covered by a separate report.
>
> Conventions: each entry gives Title, authors, venue/year, arXiv link, 2–3 sentence TL;DR, threat level (LOW / MED / HIGH) to CoT-Tool's novelty, and what CoT-Tool can offer that this paper does not.

---

## A. Foundational tool use

### A1. ReAct: Synergizing Reasoning and Acting in Language Models
- Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao — ICLR 2023 — https://arxiv.org/abs/2210.03629
- TL;DR: Interleaves reasoning traces and tool-using actions in a single prompt-driven loop. Establishes the canonical tool-use setting on HotpotQA / FEVER / ALFWorld / WebShop.
- Threat: **LOW**. Pure prompting; no calibration, no statistical guarantee on when to act.
- Differentiator: CoT-Tool wraps ReAct-style policies with a CP-calibrated trigger so the *whether-to-call* decision has a distribution-free hallucination bound.

### A2. Toolformer: Language Models Can Teach Themselves to Use Tools
- Schick, Dwivedi-Yu, Dessì, Raileanu, Lomeli, Zettlemoyer, Cancedda, Scialom — NeurIPS 2023 — https://arxiv.org/abs/2302.04761
- TL;DR: Self-supervised data construction for tool calls (calculator, QA, search, translate, calendar) by filtering tool insertions that reduce LM perplexity. Trains the LM to insert API calls inline.
- Threat: **LOW**. Decision to insert is heuristic (perplexity drop), not calibrated; no per-query risk control.
- Differentiator: CoT-Tool gives a *user-tunable α* with provable upper bound on hallucination rate among no-tool answers.

### A3. Gorilla: Large Language Model Connected with Massive APIs
- Patil, Zhang, Wang, Gonzalez — NeurIPS 2024 — https://arxiv.org/abs/2305.15334
- TL;DR: Fine-tunes LLaMA on HuggingFace/Torch/TensorHub APIs with retrieval-aware training. Introduces APIBench. Strong at *which* API to call given that tool use is needed.
- Threat: **LOW**. Orthogonal: assumes tool use is desired; doesn't decide whether-to-call.
- Differentiator: CoT-Tool is upstream — the trigger; Gorilla can be the downstream selector.

### A4. ToolLLM: Facilitating LLMs to Master 16000+ Real-world APIs
- Qin, Liang, Ye, et al. — ICLR 2024 (spotlight) — https://arxiv.org/abs/2307.16789
- TL;DR: Builds ToolBench (RapidAPI-based instruction tuning corpus), ToolLLaMA, and a DFS decision tree for multi-tool use. Strong end-to-end open-source tool-use baseline.
- Threat: **LOW–MED**. ToolBench is one of CoT-Tool's eval targets; ToolLLM is a *capability* baseline, not a triggering baseline.
- Differentiator: CoT-Tool adds the calibrated abstention/trigger layer on top of any ToolLLM-style policy.

### A5. ToolkenGPT: Augmenting Frozen LMs with Massive Tools via Tool Embeddings
- Hao, Liu, Wang, Hu — NeurIPS 2023 — https://arxiv.org/abs/2305.11554
- TL;DR: Each tool is a learnable token in the LM head; predict tool token = call tool. Frozen LM, scalable to many tools.
- Threat: **LOW**. Tool selection mechanism; whether-to-call still implicit in token probability.
- Differentiator: CoT-Tool turns the implicit token probability into a calibrated risk score with coverage guarantee.

### A6. Toolken+: Improving LLM Tool Usage with Reranking and a Reject Option
- Yakovlev, Nikolenko, Bout — Findings EMNLP 2024 — https://arxiv.org/abs/2410.12004
- TL;DR: Adds a `REJECT` option to ToolkenGPT so the model can decline tool use; re-ranks top-k tools using documentation.
- Threat: **MED**. Most directly relevant to "should I call a tool?" via reject option — but rejection is heuristic, not statistically calibrated.
- Differentiator: CoT-Tool replaces the heuristic reject with a CP-quantile threshold giving distribution-free guarantee.

### A7. Function calling APIs (OpenAI, Anthropic)
- OpenAI function calling (2023+); Anthropic tool use (2024+). Engineering blogs / docs.
- TL;DR: Provider-side training for structured tool invocation. Decision logic is a softmax + heuristic threshold.
- Threat: **LOW**. Closed; no published statistical guarantee.
- Differentiator: CoT-Tool wraps these with conformal calibration.

---

## B. Adaptive / selective RAG (when to retrieve)

### B1. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
- Asai, Wu, Wang, Sil, Hajishirzi — ICLR 2024 (oral) — https://arxiv.org/abs/2310.11511
- TL;DR: Trains a single LM with reflection tokens (`Retrieve`, `IsRel`, `IsSup`, `IsUse`) to adaptively decide whether/what to retrieve and to critique itself.
- Threat: **MED–HIGH**. Most prominent learned retrieval-gating baseline; reflection token = whether-to-retrieve decision. But no coverage guarantee.
- Differentiator: CoT-Tool offers distribution-free hallucination rate bound on no-retrieve queries; Self-RAG offers no statistical guarantee.

### B2. Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity
- Jeong, Baek, Cho, Hwang, Park — NAACL 2024 — https://arxiv.org/abs/2403.14403
- TL;DR: Trains a complexity classifier (no-retrieve / single-step / multi-step) that routes queries among three RAG pipelines.
- Threat: **MED**. A direct routing baseline; classifier-based but uncalibrated.
- Differentiator: CoT-Tool calibrates the classifier output via CP and offers risk-controlled trigger (no-tool vs tool) instead of heuristic 3-way bucket.

### B3. FLARE: Active Retrieval Augmented Generation
- Jiang, Xu, Gao, Sun, Liu, Dwivedi-Yu, Yang, Callan, Neubig — EMNLP 2023 — https://arxiv.org/abs/2305.06983
- TL;DR: Anticipates the next sentence; if any token has low logprob, retrieve and regenerate. Token-prob as retrieval trigger.
- Threat: **MED**. Confidence-driven trigger — exactly the family CoT-Tool generalizes.
- Differentiator: CoT-Tool replaces ad-hoc logprob threshold with conformal-calibrated threshold; works at query level, not token level.

### B4. RAGate: Adaptive RAG for Conversational Systems
- Wang, Zhao, Ren, Zhou, Sun — arXiv 2024 — https://arxiv.org/abs/2407.21712
- TL;DR: Three variants (prompt, PEFT, multi-head attention) decide binary RAG-on/RAG-off in conversational systems based on context + retrieved knowledge confidence.
- Threat: **MED**. A direct binary retrieval gate; uncalibrated.
- Differentiator: CoT-Tool extends to multi-tool with hierarchical CP, gives statistical guarantee.

### B5. RA-ISF: Retrieval Augmentation via Iterative Self-Feedback
- Liu, Peng, Zhang, et al. — Findings ACL 2024 — https://arxiv.org/abs/2403.06840
- TL;DR: Three submodules (self-knowledge, passage relevance, question decomposition) iteratively decide retrieval need.
- Threat: **MED**. Multi-criterion retrieval gating; no calibration.
- Differentiator: Same — CP-calibrated, single threshold per α.

### B6. SKR: Self-Knowledge Guided Retrieval Augmentation
- Wang, Li, Yan, Bao, Du, Lin — Findings EMNLP 2023 — https://arxiv.org/abs/2310.05002
- TL;DR: Trains a small classifier to ask "do I already know this?" and gates retrieval. Variants: prompt, ICL, BERT classifier.
- Threat: **MED**. The canonical "self-knowledge as router" baseline.
- Differentiator: CoT-Tool uses similar score families but adds CP layer for risk control.

### B7. SeaKR: Self-aware Knowledge Retrieval for Adaptive RAG
- Yao, Yang, Tang, Yan, Chen, Su — ACL 2025 — https://arxiv.org/abs/2406.19215
- TL;DR: Extracts self-aware uncertainty from internal FFN states; retrieves only when uncertainty is high; also re-ranks retrieved snippets by uncertainty drop.
- Threat: **HIGH**. Strong recent baseline using hidden-state uncertainty — same family as CoT-Tool's "small classifier on hidden states" candidate score.
- Differentiator: SeaKR uses a heuristic uncertainty threshold; CoT-Tool calibrates the same internal-state score with CP for distribution-free guarantee.

### B8. UAR: Unified Active Retrieval for RAG
- Cheng, Luo, Yu, Sui, Cheng — Findings EMNLP 2024 — https://arxiv.org/abs/2406.12534
- TL;DR: Four orthogonal binary classifiers (intent-aware, knowledge-aware, time-sensitive, self-aware) on fixed LLM hidden states; cheap multi-criterion retrieval timing.
- Threat: **HIGH**. Closest engineering analog to CoT-Tool's hidden-state classifier candidate, plus unifies multiple criteria.
- Differentiator: UAR has no statistical guarantee. CoT-Tool can take UAR's classifier as a base score and provide a coverage bound.

### B9. DRAGIN: Dynamic RAG based on Real-time Information Needs
- Su, Tang, Ai, Wu, Liu — ACL 2024 (oral) — https://arxiv.org/abs/2403.10081
- TL;DR: RIND module (token uncertainty × importance × semantic significance) decides when to retrieve mid-generation, with QFS query formulation.
- Threat: **MED**. Token-level retrieval trigger; no calibration.
- Differentiator: CoT-Tool operates at query level for tools, with CP guarantee.

### B10. Rowen: Adaptive RAG for Hallucination Mitigation
- Ding, Zhang, Liu, Zhang, et al. — SIGIR-AP 2025 — https://arxiv.org/abs/2402.10612
- TL;DR: Cross-language / cross-model consistency score → if inconsistent, trigger retrieval to rectify.
- Threat: **MED**. Consistency-based trigger; resembles self-consistency candidate score in CoT-Tool but heuristic.
- Differentiator: CoT-Tool would use the same self-consistency score, but with CP threshold for guarantee.

### B11. LLM-Independent Adaptive RAG: Let the Question Speak for Itself
- Maslennikova, Karpukhin, Panchenko — EMNLP 2025 — https://arxiv.org/abs/2505.04253
- TL;DR: 27 lightweight features (graph, popularity, complexity) on the question alone; classifier matches LLM-based methods at much lower cost.
- Threat: **MED–HIGH**. Question-only routing; close to "embedding distance" candidate in CoT-Tool.
- Differentiator: Still uncalibrated; CoT-Tool adds CP.

### B12. Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home
- Maslennikova et al. — ACL 2025 — https://arxiv.org/abs/2501.12835
- TL;DR: Empirical study of 35 adaptive-retrieval methods over 6 datasets. Finds that simple uncertainty estimators rival complex pipelines on QA + self-knowledge + efficiency axes.
- Threat: **MED**. Strong empirical context; supports CoT-Tool's bet that simple scores work — provided you calibrate.
- Differentiator: They benchmark heuristic methods; CoT-Tool gives the missing calibration.

### B13. WebGPT: Browser-assisted QA with Human Feedback
- Nakano, Hilton, Balaji, et al. — arXiv 2021 — https://arxiv.org/abs/2112.09332
- TL;DR: GPT-3 fine-tuned via behavior cloning + RM to browse and cite. The original "learned routing to a tool" benchmark.
- Threat: **LOW**. Always retrieves; no abstention/trigger.
- Differentiator: CoT-Tool addresses the orthogonal "should I retrieve at all" decision.

---

## C. Tool-use benchmarks

### C1. API-Bank
- Li, Chen, Yan, Dou, Yan — EMNLP 2023 — https://arxiv.org/abs/2304.08244
- TL;DR: 314 dialogues / 753 API calls evaluating planning, retrieval, and calling. Tests *whether to use*, *how to use*, and *how to plan*.
- Threat: **LOW** (benchmark, not method). Useful eval target for CoT-Tool's whether-to-call axis.

### C2. ToolBench (OpenBMB) — see ToolLLM A4
- Doubles as the canonical benchmark for multi-API agentic tasks; ToolEval metric. CoT-Tool will report on it.

### C3. T-Eval: Step-by-Step Tool Utilization Evaluation
- Chen, Du, Zhang, et al. — ACL 2024 — https://arxiv.org/abs/2312.14033
- TL;DR: Decomposes tool use into Plan/Reason/Retrieve/Understand/Instruct/Review and evaluates each independently.
- Threat: **LOW** (benchmark).
- Use: provides fine-grained ablation evaluation surface for CoT-Tool's trigger decision.

### C4. Berkeley Function Calling Leaderboard (BFCL) v3 / v4
- Patil et al. — ICML 2025 — https://gorilla.cs.berkeley.edu/leaderboard.html
- TL;DR: De-facto SOTA function-calling leaderboard. v3 added multi-turn / multi-step; v4 added agentic categories (web search, memory).
- Threat: **LOW** (benchmark).
- Use: must report BFCL trigger sub-category; CoT-Tool can be evaluated as a wrapper over any BFCL-listed model.

### C5. NexusRaven-V2 / Nexus Function Calling Benchmark
- Nexusflow — 2023 — https://github.com/nexusflowai/NexusRaven-V2
- TL;DR: 13B open model + 9-task benchmark for nested/parallel/composite function calls. Surpasses GPT-4 on certain real-API suites.
- Threat: **LOW** (benchmark + tool selector).

### C6. AgentBench: Evaluating LLMs as Agents
- Liu, Yu, Zhang, Xu, et al. — ICLR 2024 — https://arxiv.org/abs/2308.03688
- TL;DR: 8 environments (OS, DB, KG, web shopping, web browsing, etc.) for end-to-end agent evaluation.
- Threat: **LOW** (benchmark).

### C7. MetaTool: Deciding Whether to Use Tools and Which to Use
- Huang, Shi, Liu, et al. — ICLR 2024 — https://arxiv.org/abs/2310.03128
- TL;DR: 21.1k queries; explicit subtasks for tool-usage *awareness* (whether) and tool *selection* (which).
- Threat: **HIGH** as benchmark — the *only* widely-cited benchmark dedicated to "whether to use a tool". CoT-Tool *must* evaluate on MetaTool.
- Differentiator: MetaTool is what CoT-Tool optimizes; existing baselines (ChatGPT, GPT-4) leave huge room on the awareness axis.

### C8. TaskBench
- Shen, Song, Tan, et al. — NeurIPS 2024 — https://proceedings.neurips.cc/paper_files/paper/2024/hash/085185ea97db31ae6dcac7497616fd3e-Abstract-Datasets_and_Benchmarks_Track.html
- TL;DR: Benchmark for task automation: task decomposition + tool invocation + parameter prediction.
- Threat: **LOW** (benchmark).

### C9. ToolACE
- Liu, Hu, Liu, Lu, Xu, et al. — ICLR 2025 — https://arxiv.org/abs/2409.00920
- TL;DR: Agentic data-synthesis pipeline producing 26k+ APIs; ToolACE-8B rivals GPT-4 on BFCL.
- Threat: **LOW** (training-data method + capability baseline).

### C10. GAIA
- Mialon, Fourrier, Swift, Wolf, LeCun, Scialom — ICLR 2024 — https://arxiv.org/abs/2311.12983
- TL;DR: 466 multimodal real-world QA tasks for general AI assistants; tool-use heavy.
- Threat: **LOW** (benchmark).

### C11. StableToolBench / TOOLRET / Tool retrieval benchmarks
- StableToolBench (ACL 2024); TOOLRET (Findings ACL 2025) — https://aclanthology.org/2025.findings-acl.1258/
- TL;DR: Stable variants of ToolBench; unified tool-retrieval IR benchmark.
- Threat: **LOW** (benchmark).

---

## D. Confidence / calibration for routing decisions

### D1. Just Ask for Calibration (linguistic confidence)
- Tian, Mitchell, Zhou, Sharma, Rafailov, Yao, Finn, Manning — EMNLP 2023 — https://arxiv.org/abs/2305.14975
- TL;DR: Verbalized confidence (numeric or linguistic) is better calibrated than token logprobs for RLHF models. ECE drops ~50%.
- Threat: **MED**. Justifies the "explicit confidence prompt" candidate score in CoT-Tool.
- Differentiator: Tian gives an *uncalibrated* point estimate; CoT-Tool turns it into a CP-thresholded routing decision.

### D2. Do LLMs Know What They Don't Know?
- Yin, Sun, Guo, Wu, Qiu, Huang — Findings ACL 2023 — https://arxiv.org/abs/2305.18153
- TL;DR: Introduces SelfAware dataset (1k unanswerable + 2.3k answerable); shows LLMs have nontrivial but limited self-knowledge.
- Threat: **MED**. Foundational for whether self-confidence is informative for a router.
- Differentiator: CoT-Tool turns this informational signal into a guaranteed-bound trigger.

### D3. Self-Evaluation Improves Selective Generation
- Ren, Zhao, Vasudevan, et al. — NeurIPS 2023 ICBINB workshop — https://arxiv.org/abs/2312.09300
- TL;DR: Reformulates open-ended generation as token-level self-evaluation ("None of the above" / multi-way) to leverage LLMs' good token-level calibration.
- Threat: **MED**. A self-eval scoring competitor.
- Differentiator: No CP-style guarantee.

### D4. SelectLLM / Survey of Confidence Estimation in LLMs
- Geng, Cai, Wang, Koromova, Pich, et al. — NAACL 2024 — https://aclanthology.org/2024.naacl-long.366/
- TL;DR: Survey of confidence elicitation, calibration, and use; useful taxonomy.
- Threat: **LOW** (survey).

### D5. Know Your Limits: Survey of Abstention in LLMs
- Wen, Yao, Howe, Yu, Khashabi, Hajishirzi — TACL 2025 — https://aclanthology.org/2025.tacl-1.26/
- TL;DR: Comprehensive survey of abstention strategies (input-, model-, output-time).
- Threat: **LOW** (survey).
- Use: positions CoT-Tool as a CP-based abstention method tied to tool decisions.

---

## E. Multi-tool / agent routing (cost-aware, cascades)

### E1. RouteLLM: Learning to Route LLMs with Preference Data
- Ong, Almahairi, Wu, Zhang, et al. — ICLR 2025 — https://arxiv.org/abs/2406.18665
- TL;DR: Trains routers on Chatbot-Arena preference data to dispatch queries between strong and weak LLMs. ~2× cost saving at near-equal quality.
- Threat: **MED**. The dominant *model* router; analogous problem to tool routing.
- Differentiator: CoT-Tool routes between *no-tool* and *tools*, with statistical guarantee on hallucination — RouteLLM has none.

### E2. FrugalGPT
- Chen, Zaharia, Zou — TMLR 2024 (arXiv 2305.05176) — https://arxiv.org/abs/2305.05176
- TL;DR: LLM cascade with confidence-based escalation; up to 98% cost reduction at GPT-4 quality on some tasks.
- Threat: **MED**. Confidence-threshold cascade is structurally similar to tool-vs-no-tool routing.
- Differentiator: FrugalGPT thresholds are heuristic; CoT-Tool's are CP-calibrated.

### E3. Cost-Saving LLM Cascades with Early Abstention
- Gupta, Lee, Pareek, et al. — arXiv 2025 — https://arxiv.org/abs/2502.09054
- TL;DR: Adds abstention at every cascade level. 2.2% lower test loss, 13% cost reduction across 6 benchmarks.
- Threat: **MED**. Multi-level abstention is what hierarchical CP also addresses.
- Differentiator: CoT-Tool gives marginal/conditional coverage guarantees; this paper does not.

### E4. AutoMix
- Madaan, Aggarwal, Anand, et al. — NeurIPS 2024 — arXiv 2310.12963
- TL;DR: Self-verification + meta-verifier across LM cascade; chooses small vs large model per query.
- Threat: **MED**.
- Differentiator: No CP guarantee.

### E5. Reducing Tool Hallucination via Reliability Alignment (Relign)
- Cao, Xie, Liu, et al. — arXiv 2412.04141 (2024) — https://arxiv.org/abs/2412.04141
- TL;DR: Expands the tool-use action space with indecisive actions (defer, clarify, switch); RL-trained for reliability.
- Threat: **MED**. Direct competitor on tool-trigger reliability axis, but training-based and uncalibrated.
- Differentiator: Post-hoc, model-agnostic CP wrapper vs in-training RL alignment.

### E6. Hierarchical / cascade routing patterns (LangGraph, Puppeteer, HALO)
- Engineering / 2025 arXiv preprints
- Threat: **LOW** (engineering patterns; no statistical guarantee).

---

## F. Hallucination detection (signal when tool not called)

### F1. SelfCheckGPT
- Manakul, Liusie, Gales — EMNLP 2023 — https://arxiv.org/abs/2303.08896
- TL;DR: Sample N responses; measure inter-sample inconsistency (BERTScore / NLI / QA / n-gram / prompt). Black-box, no logprobs needed.
- Threat: **MED**. Underlies the "self-consistency" candidate score in CoT-Tool.
- Differentiator: CoT-Tool calibrates SelfCheck-like scores with CP threshold for guaranteed false-non-trigger rate.

### F2. Chain-of-Verification (CoVe)
- Dhuliawala, Komeili, Xu, Raileanu, Li, Celikyilmaz, Weston — Findings ACL 2024 — https://arxiv.org/abs/2309.11495
- TL;DR: Plan verification questions, answer them independently, then revise. Reduces hallucinations 50–70%.
- Threat: **LOW**. Mitigation, not detection/triggering.

### F3. Semantic Entropy / Detecting Hallucinations via Semantic Entropy
- Kuhn, Gal, Farquhar (2023, ICLR); Farquhar, Kossen, Kuhn, Gal — Nature 2024 — https://www.nature.com/articles/s41586-024-07421-0
- TL;DR: Cluster sampled answers by semantic equivalence, compute entropy. Detects hallucinations beyond surface-form variance.
- Threat: **HIGH**. The strongest unsupervised hallucination signal; would be the natural conformity score.
- Differentiator: SE gives raw scores; CoT-Tool wraps SE in CP for distribution-free trigger guarantee.

### F4. Semantic Entropy Probes
- Kossen, Han, Razzak, Schut, Malik, Gal — NeurIPS 2024 — https://arxiv.org/abs/2406.15927
- TL;DR: Train probes to predict semantic entropy from a single forward pass. 10× cheaper than sampling.
- Threat: **HIGH**. Combines hidden-state probing + semantic entropy — the same recipe CoT-Tool's "small classifier on hidden states" candidate uses.
- Differentiator: Probe alone is point estimate; CoT-Tool turns it into a CP-thresholded trigger.

### F5. INSIDE / EigenScore
- Chen, Liu, Tan, Cao, Tang, et al. — ICLR 2024 — https://arxiv.org/abs/2402.03744
- TL;DR: Eigenvalues of the response covariance matrix in dense embedding space measure self-consistency; feature clipping reduces overconfidence.
- Threat: **MED**. Internal-state hallucination detector; one of CoT-Tool's candidate score families.

### F6. MIND: Unsupervised Real-Time Hallucination Detection
- Su, Wang, Ai, et al. — Findings ACL 2024 — https://arxiv.org/abs/2403.06448
- TL;DR: Pseudo-train an MLP on Wikipedia-derived pseudo-labels; detect hallucinations from internal states in real time (~3% inference overhead).
- Threat: **MED**. A direct candidate for the hidden-state classifier.

### F7. Lookback Lens
- Chuang, Qiu, Hsieh, Krishna, Kim, Glass — EMNLP 2024 — https://arxiv.org/abs/2407.07071
- TL;DR: Per-head ratio of attention to context vs newly generated tokens; linear probe detects contextual hallucinations cheaply.
- Threat: **MED**. Specific to RAG-style hallucinations; orthogonal to query-level routing.

### F8. ConU: Conformal Uncertainty in LLMs
- Wang, Wang, Su, et al. — Findings EMNLP 2024 — https://aclanthology.org/2024.findings-emnlp.404/
- TL;DR: Self-consistency-based conformity score; produces prediction sets with coverage guarantee for LLM QA.
- Threat: **MED**. CP for QA, not for tool triggering — CoT-Tool extends to the trigger setting.

### F9. HalluLens: LLM Hallucination Benchmark
- Bang, Chen, Dai, et al. — ACL 2025 — https://arxiv.org/abs/2504.17550
- TL;DR: Taxonomy-based extrinsic + intrinsic hallucination benchmark.
- Threat: **LOW** (benchmark).

---

## G. Most threatening to CoT-Tool novelty (CP × routing / RAG)

### G1. TRAQ: Trustworthy Retrieval Augmented QA via Conformal Prediction
- Li, Park, Lee, Bastani — NAACL 2024 — https://arxiv.org/abs/2307.04642
- TL;DR: First end-to-end statistical correctness guarantee for RAG via CP across the retrieval+generation pipeline. Bayesian-optimized prediction set sizes.
- Threat: **HIGH**. Closest in spirit: CP applied to a RAG pipeline. But TRAQ assumes retrieval is *always on*; doesn't calibrate the *whether to retrieve* gate.
- Differentiator: CoT-Tool calibrates the *gating decision itself* (not the post-retrieval answer set), with risk control specifically on the no-tool subset.

### G2. Mitigating LLM Hallucinations via Conformal Abstention
- Abbasi-Yadkori, Kuzborskij, Stutz, György, Fisch, et al. (Google DeepMind) — NeurIPS 2024 — https://arxiv.org/abs/2405.01563
- TL;DR: Self-consistency conformity score + CP gives distribution-free hallucination-rate bound on responses (the model abstains when score is low). Lightweight, prompting only.
- Threat: **HIGH (most threatening single paper)**. Same risk-control framing CoT-Tool uses. Difference: their action is *abstain* (refuse to answer); CoT-Tool's action is *call a tool*. The math is closely related.
- Differentiator: CoT-Tool ties the conformal threshold to the *oracle "tool would have helped"* label (Strategy A: no-tool-wrong AND with-tool-right) — a fundamentally different calibration target. Also extends to multi-tool via hierarchical CP.

### G3. Conformal Language Modeling
- Quach, Fisch, Schuster, Yala, Sohn, Jaakkola, Barzilay — ICLR 2024 — https://arxiv.org/abs/2306.10193
- TL;DR: Conformal stopping rule for sampling LM outputs; identifies a sub-claim set that is independently correct with statistical guarantee.
- Threat: **HIGH**. CP for generation; sets the standard for split-CP applied to LMs.
- Differentiator: Operates on output sets; CoT-Tool operates on a *binary trigger decision* with CRC-style risk targeting hallucination rate among no-tool answers.

### G4. Language Models with Conformal Factuality Guarantees
- Mohri, Hashimoto — ICML 2024 — https://arxiv.org/abs/2402.10978
- TL;DR: Conformal factuality framework: decompose to sub-claims, score, progressively remove low-confidence claims to reach 80–90% guaranteed correctness.
- Threat: **HIGH**. Similar CP-on-LM framework; sub-claim decomposition.
- Differentiator: Mohri & Hashimoto modify the *output*; CoT-Tool modifies the *action* (call vs not call). Action space is different; calibration target is different.

### G5. Conformal-RAG: Response Quality Assessment via Conditional Conformal Factuality
- Feng, Li, et al. — SIGIR 2025 — https://arxiv.org/abs/2506.20978
- TL;DR: Group-conditional CP on RAG sub-claims; retains 60% more high-quality sub-claims at the same factuality guarantee.
- Threat: **HIGH**. CP on RAG outputs with group conditioning; close to hierarchical multi-tool CP.
- Differentiator: Works post-retrieval on the answer; CoT-Tool works pre-retrieval on the gate.

### G6. Conformal Abstention for LMs (geometry-calibrated)
- Su, Lee, et al. — arXiv 2024 — https://arxiv.org/abs/2406.27914
- TL;DR: Geometry-based representation calibration for LM abstention with finite-sample guarantees.
- Threat: **HIGH**. Same problem family.
- Differentiator: Abstention only; no tool-routing extension.

### G7. Learning Conformal Abstention Policies (RL + CP)
- Tayebati, Yadav, Choudhuri, et al. — arXiv 2502.06884 (2025) — https://arxiv.org/abs/2502.06884
- TL;DR: Combines RL with CP to dynamically learn abstention thresholds for LLMs and VLMs. Hallucination AUROC +22%, ECE −70 to −85%.
- Threat: **HIGH**. RL-learned conformal threshold = adaptive CP for abstention.
- Differentiator: Still abstention; no whether-to-call-tool labels.

### G8. Conformal Arbitrage: Risk-Controlled Routing
- Overman et al. — 2025 preprint — https://woverman.com/assets/publications/2025_conformal_arbitrage/paper.pdf
- TL;DR: Single calibrated scalar threshold for routing between models with risk-utility control. CRC-style.
- Threat: **HIGH**. CP applied to *routing*, not abstention.
- Differentiator: Inter-model routing; doesn't address tool gating or hallucination on the no-tool branch.

### G9. Aligning Model Properties via Conformal Risk Control
- Overman, Vasileios, et al. — NeurIPS 2024 — https://proceedings.neurips.cc/paper_files/paper/2024/hash/c79625091a4f8b5d3abe29f3b14fa43a-Abstract-Conference.html
- TL;DR: General CRC for aligning arbitrary scalar properties of LM outputs.
- Threat: **MED–HIGH**. Generic framework that CoT-Tool could be expressed within.
- Differentiator: CoT-Tool's specific application + Strategy-A oracle label is the novel instantiation.

### G10. Conformal LM Reasoning with Coherent Factuality
- Cherian, Park, Bastani — ICLR 2025 — https://arxiv.org/abs/2505.17126
- TL;DR: Extends CP factuality to reasoning chains where claims depend on each other. Defines and guarantees coherent factuality.
- Threat: **MED–HIGH**. Closely related to CoT-CP (sister project); for CoT-Tool, less direct.
- Differentiator: CoT-Tool's setting is the trigger axis, not the chain.

---

## H. HIGH-threat papers (explicit list)

| # | Paper | Why HIGH |
|---|---|---|
| H1 | **Mitigating LLM Hallucinations via Conformal Abstention** (Abbasi-Yadkori et al., NeurIPS 2024) | Same CP-for-hallucination-rate framing; differs only in action space (abstain vs call-tool). MUST be the primary baseline. |
| H2 | **TRAQ** (Li et al., NAACL 2024) | First CP-with-RAG pipeline; coverage guarantee on retrieval-augmented answers. |
| H3 | **Language Models with Conformal Factuality Guarantees** (Mohri & Hashimoto, ICML 2024) | Established sub-claim CP for LMs; CoT-Tool reuses ideology but on the trigger. |
| H4 | **Conformal Language Modeling** (Quach et al., ICLR 2024) | Foundational split-CP for LM generation. |
| H5 | **Semantic Entropy Probes** (Kossen et al., NeurIPS 2024) | Strongest cheap hallucination detector; obvious conformity-score input. Also: **Semantic Entropy / Nature 2024** (Farquhar et al.). |
| H6 | **SeaKR** (Yao et al., ACL 2025) and **UAR** (Cheng et al., EMNLP-F 2024) | Both build adaptive-RAG routers from internal states — same architectural niche as CoT-Tool's hidden-state classifier candidate. |
| H7 | **Conformal-RAG / Conditional Conformal Factuality** (Feng et al., SIGIR 2025) | Group-conditional CP on RAG; relevant to multi-tool hierarchical CP. |
| H8 | **Conformal Arbitrage** (Overman, 2025) and **Learning Conformal Abstention Policies** (Tayebati, 2025) | CP applied to routing/abstention; closest in *machinery*. |
| H9 | **MetaTool** (Huang et al., ICLR 2024) | Although a benchmark, it is the *only* dedicated whether-to-use benchmark — must be evaluated on. |

**Top-5 most threatening (single sentence each):**
1. *Mitigating LLM Hallucinations via Conformal Abstention* — exact framework match modulo action space.
2. *TRAQ* — first CP guarantee on a RAG pipeline.
3. *Mohri & Hashimoto, Conformal Factuality* — foundational CP-for-LM-correctness.
4. *Conformal-RAG (Feng 2025)* — group-conditional CP on RAG, near-twin of hierarchical multi-tool CP.
5. *SeaKR + UAR* — strongest adaptive-RAG routers from hidden states, the engineering substrate CoT-Tool will calibrate.

---

## I. Tool-use benchmark snapshot (current SOTA, May 2026)

### Berkeley Function Calling Leaderboard (BFCL v3 / v4)
- BFCL v3 (multi-turn / multi-step): top frontier models (Claude Opus 4.x, GPT-5, Gemini 2.5 Pro) score in the 70–80% range on overall accuracy; open models led by **ToolACE-2-8B** and **xLAM-2** are competitive in the 60–70% range.
- BFCL v4 adds web-search and memory categories — agentic frontier; SOTA still <70% overall.
- For CoT-Tool: the *AST-relevance* sub-track (does the model abstain when no API matches?) is the natural target for the trigger metric.

### ToolBench (OpenBMB / Qin et al.)
- StableToolBench (with virtual API server) is now the standard. Top open-source model (ToolLLaMA-3 / xLAM / ToolACE-2) reach pass-rate ~60% on G1; GPT-4-class models ~70%+.
- "Pass Rate" mixes invocation correctness and answer correctness; CoT-Tool can improve both by suppressing unnecessary calls.

### MetaTool
- Original paper: GPT-4 ~80% on tool-awareness; GPT-3.5 ~60%. Open 7B models ~50%. Substantial headroom for a calibrated trigger.

### GAIA
- L1 ~50%, L2 ~30%, L3 <10% for top public agents (May 2026). Tool-use heavy; not specifically about whether-to-call.

---

## J. Synthesis — gap CoT-Tool fills

The literature splits into **four layers**:

1. **Capability** (Toolformer, Gorilla, ToolLLM, ToolACE): how to make an LM *able* to call tools. Solved-ish.
2. **Selection** (Toolken+, BFCL function-calling models, NexusRaven): given that a tool is needed, *which one* and with *what arguments*. Mature.
3. **Heuristic gating** (Self-RAG, Adaptive-RAG, FLARE, RAGate, SKR, SeaKR, UAR, DRAGIN, Rowen, RA-ISF, MetaTool eval): *whether* to retrieve / call a tool. Active and crowded — but **all use heuristic thresholds**.
4. **Statistical guarantees** (Quach et al., Mohri & Hashimoto, TRAQ, Abbasi-Yadkori, Conformal-RAG, Conformal Arbitrage): CP / CRC applied to **outputs** (factuality of generated text) or to **abstention** (refuse to answer) or to **inter-model routing**.

**No paper combines layer 3 with layer 4 *for the tool-trigger decision specifically*.** The closest neighbors are:
- **Abbasi-Yadkori 2024** — CP for *abstention*. CoT-Tool generalizes to a richer action ("call tool" instead of "say nothing"), with a different oracle label.
- **TRAQ / Conformal-RAG** — CP for the *answer set after retrieval*. CoT-Tool operates on the *upstream gating decision*, controlling hallucination rate specifically on the no-tool branch.
- **SeaKR / UAR / Adaptive-RAG / Self-RAG** — adaptive gating *without guarantees*. CoT-Tool can take any of these scores as a base conformity function and provide the missing CP guarantee.

**CoT-Tool's three novel contributions:**
1. **Strategy-A oracle label** ("LLM-no-tool wrong AND LLM-with-tool right") — a tool-utility-conditional ground truth, distinct from (a) factuality labels (Mohri/Quach/TRAQ) and (b) "model knows" labels (SKR/SeaKR). This is what makes the calibration *trigger-specific*.
2. **Distribution-free hallucination bound on the no-tool subset** — a one-sided risk on the *unaugmented* branch, complementary to hallucination bounds on the *augmented* branch (TRAQ) or on full generation (Abbasi-Yadkori, Mohri).
3. **Hierarchical CP for multi-tool extension** — analogous to group-conditional CP in Conformal-RAG but applied to a tool taxonomy (calculator / search / code-exec / KB lookup), giving per-tool risk targets. No existing paper provides this for the tool routing setting.

The cleanest single-sentence positioning: **"CoT-Tool is to the tool-trigger decision what Conformal Abstention is to the I-don't-know decision and what TRAQ is to the post-retrieval answer set."**

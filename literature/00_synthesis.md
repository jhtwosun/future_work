# Cross-Cutting Literature Synthesis — Conformal Prediction for LLM Uncertainty

> Reads on top of the four area surveys (`01–04*.md`). Goal: per-project threat assessment, novelty claim, must-cite / must-baseline list, and strategic implications for the 12-month plan.

---

## 0. TL;DR

| Project | Verdict | Core positioning sentence |
|---|---|---|
| **CoT-CP** | ✅ Strongest open white space | *First step-level CRC for reasoning chains: turns existing heuristic step-confidence signals (DeepConf, DEER, ESC, PAVs) into a calibrated, distribution-free, sequence-level coverage guarantee.* |
| **CoT-Tool** | ⚠️ Defensible but crowded — needs sharper framing | *CP-wrapper for the **tool-trigger** decision: complements Conformal Abstention (action: refuse) and TRAQ (target: post-retrieval answer set) by controlling hallucination on the **no-tool** branch.* |
| **CG-VL** | ⚠️ Method idea (PMI) is taken — guarantee framing is the contribution | *First α-bounded VLM hallucination controller: turns the M3ID/C-PMI/VCD/IBD/PAI family of PMI-style decoding signals into a conformity score with finite-sample coverage.* |

**Strategic recommendation**: keep CG-VL → CoT-Tool → CoT-CP sequence in the original plan, but reframe each project's "core contribution" from "we propose X" to "we **calibrate** existing X with a guarantee." All three projects are wrappers, and that's fine — but the framing must own that explicitly.

---

## 1. Project 1 — CoT-CP

### 1.1 Five papers to position against (HIGH-threat)

| # | Paper | Venue | Why threatening | What CoT-CP adds |
|---|---|---|---|---|
| 1 | **DeepConf** (Fu et al. 2025, [2508.15260](https://arxiv.org/abs/2508.15260)) | arXiv 2025 | Span-confidence filtering on AIME/GPQA, 99.9% acc with 84% fewer tokens — empirically saturates the same benchmarks | Distribution-free CRC on final answer; DeepConf gives no coverage promise |
| 2 | **DEER: Dynamic Early Exit in Reasoning Models** ([2504.15895](https://arxiv.org/abs/2504.15895)) | arXiv 2025 | Step-level confidence-driven early exit on R1-distill etc. | Calibrated $\hat{q}$ from PRM800K vs hand-tuned threshold |
| 3 | **Adaptive-Consistency / ESC** (Aggarwal+ EMNLP 2023; Li+ 2401.10480) | EMNLP 2023 / arXiv 2024 | Adaptive sample-count via stopping rule | Step-level (not prompt-level), frequentist coverage (not Bayesian heuristic) |
| 4 | **Snell et al., Scaling Test-Time Compute Optimally** ([2408.03314](https://arxiv.org/abs/2408.03314)) | arXiv 2024 | The compute-optimal-adaptive ancestor; cited everywhere | Step-level adaptation + sequence-level CRC instead of just budget reallocation |
| 5 | **Rewarding Progress / PAVs** (Setlur et al. [2410.08146](https://arxiv.org/abs/2410.08146)) | arXiv 2024 | Step-level verifier-guided adaptive search | Calibration of PAV scores into coverage-controlled gates |

**CP-side neighbors** (theory threats, lighter):
- ⚠️ **Thought Calibration** (Stewart et al., **EMNLP 2025**, [2505.18404](https://arxiv.org/pdf/2505.18404)) — **PROMOTED to highest CP-side threat (added 2026-05-08)**. Same problem (calibrated test-time stopping for R1/R1-distill), same era. Differentiation: CoT-CP introduces score-family Pareto (Theorem 2) + discrete-shift weighted CP (Theorem 3) + 11-model × 7-dataset matrix; Thought Calibration is single-model-family stopping rule.
- **Conformal Language Model Reasoning with Coherent Factuality** (Rubin-Toles et al., **ICLR 2025**) — first CP on reasoning with deducibility graphs. Closest theoretical neighbor.
- **CoVeR: Conformal Calibration for Versatile Next-Token Prediction** (arXiv 2509.04733, 2025) — token-cluster step calibration. Different unit.
- **Differentiable Conformal Training** (2026 preprint) — *watch this*; if it lands at ICML 2026 it will be a primary citation.
- **Paraphrase-Robust CP** (ICLR 2026 submission, [openreview Uf04r8gDn7](https://openreview.net/forum?id=Uf04r8gDn7)) — directly relevant to our `SX_paraphrase_cross_dataset` work.
- **CP Beyond the Seen — Missing Mass** (NeurIPS 2025) — useful theoretical framing for novel-trace coverage.

### 1.2 Must-baseline (experimental)

- Greedy / vanilla CoT
- Self-consistency $N \in \{8, 16, 32\}$
- Best-of-N + PRM (Math-Shepherd, Skywork-PRM)
- **DeepConf** (mandatory comparison)
- Adaptive-Consistency / ESC
- Step-level early exit (DEER)

### 1.3 Must-cite (foundational)

- Vovk-Gammerman-Shafer 2005 (CP); Angelopoulos-Bates 2021 (gentle intro); Angelopoulos+ 2022 (CRC); Tibshirani+ 2019 (weighted CP); Barber+ 2022 (beyond exchangeability)
- Wang+ 2022 (Self-Consistency); Lightman+ 2023 (PRM800K / Let's Verify); Wang+ 2024 (Math-Shepherd); Snell+ 2024
- Rubin-Toles+ 2025 (Coherent Factuality); Mohri & Hashimoto 2024 (Conformal Factuality); Quach+ 2024 (Conformal LM)

### 1.4 Sharpening recommendation

**The literature collectively says: step-level signals exist, everyone wants to use them, no one calibrates them.** This is unusually clean white space. Lean into it: don't sell CoT-CP as a "new score" (S-SC/S-PRM/S-LP) but as a **calibration layer that any of these signals can be plugged into**. The score ablation becomes a *secondary* contribution; the *primary* contribution is the CRC machinery for sequence-level coverage from step-level scores. Frame the paper as: "Pick your favorite step confidence — DeepConf's, DEER's, a PRM's — and we make it certifiable."

---

## 2. Project 2 — CoT-Tool

### 2.1 Five papers to position against (HIGH-threat)

| # | Paper | Venue | Why threatening | What CoT-Tool adds |
|---|---|---|---|---|
| 1 | **Mitigating LLM Hallucinations via Conformal Abstention** (Abbasi-Yadkori et al., [2405.01563](https://arxiv.org/abs/2405.01563)) | **NeurIPS 2024** | Same CP-for-hallucination-rate framing; differs only in action space (abstain vs call-tool) | Different action ("call tool"), Strategy-A oracle label, hierarchical multi-tool extension |
| 2 | **TRAQ** (Li et al. [2307.04642](https://arxiv.org/abs/2307.04642)) | NAACL 2024 | First end-to-end CP on RAG pipeline | TRAQ controls *post-retrieval* answer; CoT-Tool controls *whether to retrieve* |
| 3 | **Conformal-RAG / Conditional Conformal Factuality** (Feng et al. [2506.20978](https://arxiv.org/abs/2506.20978)) | SIGIR 2025 | Group-conditional CP on RAG — directly resembles hierarchical multi-tool | CoT-Tool applies group-conditioning to a *tool taxonomy* (calc/search/code/KB), not retrieval groups |
| 4 | **Prune 'n Predict / CROQ + CP-OPT** (Vishwakarma et al.) | ICML 2025 | CP on ToolAlpaca and MMLU; tool-routing CP precedent | CROQ does MCQ-style tool-set CP; CoT-Tool gives *gating* (call vs not) with cost-aware CRC |
| 5 | **Conformal Constrained Policy Optimization for Cost-Effective LLM Agents** (arXiv [2511.11828](https://arxiv.org/abs/2511.11828), 2025) | arXiv 2025 | Cost-aware CP routing among LLM agents | Different unit (inter-agent vs tool decision); CoT-Tool keeps single LLM and routes its decision to tool |

**Adaptive-RAG/router neighbors** (engineering threats — not CP, but same niche):
- **SeaKR** (Yao et al., ACL 2025), **UAR** (Cheng et al., EMNLP-Findings 2024), **Adaptive-RAG** (Jeong et al., NAACL 2024), **Self-RAG** (Asai et al., ICLR 2024), **DRAGIN**, **Rowen**, **RA-ISF**, **MetaTool** (Huang et al., ICLR 2024)
- All heuristic. CoT-Tool can use any of them as the **base scoring function** and add the missing CP guarantee on top.

### 2.2 Must-baseline (experimental)

- Always-tool / Never-tool (cost/quality envelopes)
- ReAct prompting
- Self-RAG (ICLR 2024)
- Adaptive-RAG (NAACL 2024)
- SeaKR (ACL 2025) and/or UAR (EMNLP-F 2024) — strongest hidden-state routers
- **Conformal Abstention** (Abbasi-Yadkori 2024, NeurIPS) — adapt their semantic-entropy score to the tool-trigger decision; this is the apples-to-apples comparison
- MetaTool eval (the only dedicated whether-to-call benchmark)

### 2.3 Must-cite

- Yao+ 2022 (ReAct); Schick+ 2023 (Toolformer); Asai+ 2024 (Self-RAG); Jeong+ 2024 (Adaptive-RAG)
- Mohri & Hashimoto 2024 (Conformal Factuality); Quach+ 2024 (Conformal LM); Abbasi-Yadkori 2024 (Conformal Abstention); Feng+ 2025 (Conformal-RAG)
- BFCL (Yan+, leaderboard); ToolBench / StableToolBench; MetaTool; API-Bank

### 2.4 Sharpening recommendation

**The framing struggle is real here.** Conformal Abstention 2024 is genuinely close. Three concrete moves:

1. **Reframe the action space carefully.** "Abstain" and "call tool" are formally different action sets — but reviewers will ask whether the framework is really new or just a relabeling. The differentiator must be **operational**: tool-call gives a different answer, not no-answer. The Strategy-A oracle ("LLM-no-tool wrong AND LLM-with-tool right") is what makes the calibration trigger-specific and is genuinely novel data construction.

2. **Lean on the multi-tool / hierarchical CP.** Single-tool gating is too close to abstention. The story is much stronger if framed as **"per-tool risk targets via group-conditional CP"** — calculator gets one $\alpha$, search gets another, code-exec gets a third. This is the Conformal-RAG-style group-conditional CP applied to a tool taxonomy, and no one has done it.

3. **Cost as a CRC loss.** Conformal Constrained Policy Optimization (2025) does cost-aware CP for *agent routing*. CoT-Tool can do the same for *tool routing* — make budget per-query a constrained loss in the CRC formulation rather than a side metric.

**If you can't sharpen along these axes, this becomes the weakest of the three projects.** Consider whether to defer in the sequencing.

---

## 3. Project 3 — CG-VL

### 3.1 Five papers to position against (HIGH-threat)

| # | Paper | Venue | Why threatening | What CG-VL adds |
|---|---|---|---|---|
| 1 | **C-PMI: Conditional PMI Calibrated Decoding for VLMs** ([2505.19678](https://arxiv.org/abs/2505.19678)) | **NeurIPS 2025** | Same name (PMI), same modality (VLM), same target (hallucination) | C-PMI is a decoding-time bi-level optimization; CG-VL uses PMI as fixed *conformity score* with α-bound |
| 2 | **M3ID** (Favero et al. [2403.14003](https://arxiv.org/abs/2403.14003)) | CVPR 2024 | Uses $\log p(y \mid I,q)-\log p(y \mid q)$ as sampling objective — exact same quantity | M3ID has no α-guarantee; CG-VL's null is `I_shuffle` (not unconditioned LM), isolating spatial-semantic grounding; abstention not sampling |
| 3 | **Inductive CP for LVLMs** (Z. Wang et al. [2504.17671](https://arxiv.org/abs/2504.17671)) | arXiv 2025 | Direct CP-for-VLM precedent | Closed-set MCQ-VQA with softmax scores; CG-VL is open-ended generation with PMI score and abstention |
| 4 | **IBD — Image-Biased Decoding** ([2402.18476](https://arxiv.org/abs/2402.18476)) | 2024 | Image-biased model − base model ≈ PMI in disguise | IBD requires *training* a biased model; CG-VL is training-free, single forward pair |
| 5 | **PAI — Pay Attention to Image** ([2407.21771](https://arxiv.org/abs/2407.21771)) | ECCV 2024 | Subtracts pure-text logits from multimodal logits — algebraically PMI | No calibration / α-bound; CG-VL adds CP wrapper |

**Adjacent (decoding methods, no guarantee)**: VCD (CVPR 2024), OPERA (CVPR 2024), HALC, SID, ICD, RITUAL, AvisC, VISTA, Pensieve, Treble Counterfactual.

### 3.2 Must-baseline

- Vanilla VLM (greedy, sampling)
- **VCD** (CVPR 2024) — Gaussian-noise contrast
- **OPERA** (CVPR 2024) — over-attention penalty
- **M3ID** (CVPR 2024) — closest score-wise
- **C-PMI** (NeurIPS 2025) — closest by name
- A POPE-classifier baseline (off-the-shelf hallucination detector)
- **Mitigating LLM Hallucinations via Conformal Abstention** (Abbasi-Yadkori 2024) — adapt to VLM input by feeding image-aware prompt; tests whether the *image-shuffle PMI* specifically (vs semantic-entropy clustering) buys anything

### 3.3 Must-cite

- Li+ 2023 (POPE); Wang+ 2023 (AMBER); Guan+ 2024 (HallusionBench); Sun+ 2023 (MMHal); Yu+ 2023 (MMVet)
- Leng+ 2024 (VCD); Huang+ 2024 (OPERA); Favero+ 2024 (M3ID); Liu+ 2024 (PAI); Liu+ 2024 (IBD); C-PMI 2025
- Quach+ 2024; Mohri & Hashimoto 2024; Abbasi-Yadkori+ 2024; Z. Wang+ 2025 (Inductive CP for LVLMs)
- VLM backbones: Bai+ 2025 (Qwen2.5-VL); Liu+ 2024 (LLaVA-1.6/NeXT); Idefics-3
- VLM hallucination surveys: Bai+ 2024 ([2404.18930](https://arxiv.org/abs/2404.18930)), H. Liu+ 2024 ([2402.00253](https://arxiv.org/abs/2402.00253))

### 3.4 Sharpening recommendation

**Method-side novelty is moderate; statistical-guarantee novelty is high.** Position explicitly as a wrapper:

> *"Pick any PMI-style VLM hallucination signal — M3ID's, C-PMI's, IBD's, PAI's, or VCD's contrast — and we wrap it with a CP layer that delivers $P(\text{hallucination} \mid \text{not abstained}) \le \alpha$ on POPE/AMBER/HallusionBench, finite-sample, distribution-free."*

Three concrete moves to maximize novelty in 2026:

1. **Make the masking ablation a first-class contribution.** Existing works each pick *one* null (Gaussian for VCD, prompt-only for M3ID, text-inertia for PAI). CG-VL's shuffle/zero/noise/patch comparison as a unified conformity-score family is genuinely uncovered. Frame this as "an empirical study of image counterfactuals as conformity nulls."

2. **Open-ended generation, not closed-set.** Inductive CP for LVLMs (2025) already did closed-set MCQ. Stay clearly open-ended (CHAIR-COCO captioning, MMVet GPT-judge). This is where the PMI score earns its keep.

3. **Provide both selection and abstention modes.** Most CP-LLM work gives prediction sets *or* abstention. CG-VL can offer both: (a) abstain on low-grounding answers, (b) return a calibrated multi-answer set when grounding is moderate. This dual mode plus PMI score is a clean methodological niche.

---

## 4. Cross-cutting observations

### 4.1 Convergent theme across all three projects

The 2024-2026 LLM/VLM uncertainty literature has a strong pattern:

- **Layer 1**: Strong heuristic signals exist (PRM scores, self-consistency, semantic entropy, attention patterns, PMI scores) and have been validated empirically.
- **Layer 2**: People combine these into adaptive procedures (DeepConf, DEER, Self-RAG, M3ID, VCD, OPERA) without statistical guarantee.
- **Layer 3**: A small CP-for-LLM community gives finite-sample coverage but on slightly off-target unit (full sequences, claims, abstention, MCQ sets).

**All three projects sit at the Layer-2 → Layer-3 conversion.** This is consistent enough that the unifying TMLR position paper at month 9-12 (mentioned in the original plan) has clear shape: "Calibrated wrappers for LLM/VLM uncertainty heuristics."

### 4.2 Common reviewer concerns to preempt

| Concern | Mitigation |
|---|---|
| "Exchangeability assumption broken (auto-regressive)" | Coarsen unit (step / decision / answer-span) and cite Barber+ 2022 + weighted CP for distribution shift |
| "Wrapper papers — what's the technical contribution?" | Lean on the *specific* conformity score, the *specific* calibration data construction, and the empirical ablation. Don't oversell theoretical novelty |
| "Calibration set leakage with eval benchmark" | Already addressed in plan §0.4 — use train splits + 10% eval-split sanity check |
| "Coverage doesn't hold under distribution shift" | Run the OOD experiment (calibrate on GSM8K, test on MATH/AIME — sec 1.3 §5). Cite weighted CP. Show graceful degradation |
| "Abstention rate too high → useless system" | Pareto frontier (accuracy vs answer-rate) is the main figure for all three projects |

### 4.3 Theoretical leverage points

- **Sequence-level → step-level coverage propagation** for CoT-CP — CRC nested-set construction (cite Angelopoulos+ 2022, Bates+ 2021)
- **Group-conditional CP** for CoT-Tool multi-tool — cite Conformal-RAG (Feng+ 2025), Cherian-Gibbs-Candes (NeurIPS 2024), Romano-Sesia-Candès 2020 (APS for groups)
- **PMI as conformity score** for CG-VL — frame as "information-theoretic calibration"; cite Cover-Thomas, M3ID, C-PMI

---

## 5. Strategic implications for the 12-month plan

### 5.1 Sequencing — keep the original CG-VL → CoT-Tool → CoT-CP order

Confirmed by the literature scan:

- **CG-VL first** (months 1-3): infrastructure already in place (POPE/AMBER on lmms-eval, Qwen2.5-VL), benchmarks are closed-form so iteration is fast, and the C-PMI + PAI + M3ID baselines are all open-source. The "wrapper" framing makes the paper achievable in 3 months even though the method-novelty bar is moderate.
- **CoT-Tool second** (months 3-5): needs sharper framing (see §2.4) but has the most operationally useful end-product. Defer if the framing struggle proves unresolvable; CG-VL → CoT-CP direct path is also viable.
- **CoT-CP third** (months 5-9): the most theoretically interesting, the strongest open white space, but also the largest scope (PRM800K calibration, scaling to 32B, sequence-level CRC theory). Worth saving for the longest runway.

### 5.2 Compute budget reality check

Each project's expensive line items:

- **CG-VL**: 2× inference cost (image + image-shuffle forward) on POPE/AMBER/HallusionBench across 4 models (Qwen-VL, LLaVA-1.6, Idefics-3, Qwen-Omni). ~A100-week per model-benchmark pair, so ~16 A100-weeks total. **Tractable** on the 8-GPU server.
- **CoT-Tool**: calibration data generation (Strategy A) is the bottleneck — every calibration query needs 2 LLM forwards (with-tool, without-tool). ~10K queries × 2 forwards × 7B = manageable. Multi-tool extension scales linearly. **Tractable**.
- **CoT-CP**: scaling to 32B / 72B is the budget risk. AIME / GPQA evaluations with $N=32$ self-consistency on a 32B model are expensive. Plan to truncate to 7B-only for the first submission and add 32B for camera-ready / journal extension.

### 5.3 Risk-adjusted recommendation

If forced to drop one of the three projects: **drop CoT-Tool**. CoT-CP has the strongest open white space (heuristic step-confidence is an active battleground with no CP entrant). CG-VL is closest to "ready to go" (infrastructure in place, baselines open-source). CoT-Tool's framing is the most contested and will require the most defensive writing in reviews. If both CG-VL and CoT-CP go through, CoT-Tool can become a NeurIPS-Agents workshop paper or fold into the unifying TMLR piece.

---

## 6. Bibliographic backbone (the 25-paper short list)

These are the ones to read end-to-end before writing.

### Foundational CP
1. Vovk-Gammerman-Shafer 2005 — *Algorithmic Learning in a Random World* (book)
2. Romano-Sesia-Candès 2020 — APS (NeurIPS)
3. Angelopoulos-Bates-Jordan-Malik 2021 — RAPS (ICLR)
4. **Angelopoulos-Bates-Fisch-Lei-Schuster 2022** — Conformal Risk Control (arXiv 2208.02814; ICLR'24 nested CRC)
5. Tibshirani-Foygel Barber-Candès-Ramdas 2019 — Weighted CP (NeurIPS)
6. Barber-Candès-Ramdas-Tibshirani 2022 — Beyond Exchangeability (Annals of Statistics)

### CP for LLMs / VLMs
7. **Quach et al. 2024** — Conformal Language Modeling (ICLR)
8. **Mohri & Hashimoto 2024** — Conformal Factuality (ICML)
9. **Abbasi-Yadkori et al. 2024** — Conformal Abstention for LLM Hallucinations (NeurIPS, [2405.01563](https://arxiv.org/abs/2405.01563))
10. **Rubin-Toles et al. 2025** — Conformal LM Reasoning with Coherent Factuality (ICLR)
11. **Cherian-Gibbs-Candès 2024** — Large Language Model Validity via Enhanced Conformal Prediction (NeurIPS)
12. **Feng et al. 2025** — Conformal-RAG / Conditional Conformal Factuality (SIGIR)
13. **Z. Wang et al. 2025** — Inductive CP for LVLMs ([2504.17671](https://arxiv.org/abs/2504.17671))
14. KnowNo (Ren et al., CoRL 2023) — robotic CP precedent
15. Li et al. 2024 — TRAQ (NAACL)

### Reasoning / step-level / test-time scaling
16. Wang et al. 2022 — Self-Consistency
17. Lightman et al. 2023 — Let's Verify Step by Step / PRM800K
18. Wang et al. 2024 — Math-Shepherd
19. **Snell et al. 2024** — Scaling Test-Time Compute Optimally ([2408.03314](https://arxiv.org/abs/2408.03314))
20. **Fu et al. 2025** — DeepConf ([2508.15260](https://arxiv.org/abs/2508.15260))
21. Setlur et al. 2024 — PAVs / Rewarding Progress

### Tool use / RAG routing
22. Yao et al. 2022 — ReAct
23. Asai et al. 2024 — Self-RAG (ICLR)
24. Jeong et al. 2024 — Adaptive-RAG (NAACL)

### VLM hallucination
25. Leng et al. 2024 — VCD (CVPR); Huang et al. 2024 — OPERA (CVPR); Favero et al. 2024 — M3ID (CVPR); C-PMI 2025 (NeurIPS)

---

## 7. Open questions for the author to resolve next

1. **CoT-Tool framing**: do §2.4's three sharpening moves (multi-tool group-CP, cost as CRC loss, Strategy-A oracle) feel sufficient, or should this project be deprioritized?
2. **CG-VL claim positioning**: pitch as "first CP for open-ended VLM hallucination" (factually correct given Inductive-CP-LVLM is closed-set) or as "PMI calibration wrapper" (more honest but less sexy)?
3. **CoT-CP scope**: include 32B scaling in v1 or defer to camera-ready? Compute realism check needed.
4. **Differentiable Conformal Training (2026 preprint)**: track this and decide whether to cite as concurrent or preempt.
5. **Unifying TMLR paper**: commit to it now (so all three project papers can foreshadow) or only after they land?

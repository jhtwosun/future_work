# Literature Review 04 — VLM Hallucination Methods & Benchmarks (CG-VL Application Side)

Scope: methods + benchmarks that CG-VL must beat or position against. Companion to a separate CP-theory review. Heavy weight on 2024–2026 top-venue work and major arXiv preprints.

CG-VL one-liner used for differentiation: *distribution-free finite-sample guarantee `P(hallucination | abstain=False) ≤ α` using a PMI-style image-conditioning score `s(a) = log p(a|I,q) − log p(a|I̅,q)` as the conformity score.*

---

## A. Hallucination measurement benchmarks

### A1. POPE — Evaluating Object Hallucination in Large Vision-Language Models
- **Authors / Venue / Year**: Li, Du, Zhou, Wang, Zhao, Wen — EMNLP 2023
- **arXiv**: https://arxiv.org/abs/2305.10355
- **TL;DR**: Polling-based yes/no probing with random / popular / adversarial negative sampling for ground-truth objects; first standardized object-hallucination eval; widely used as POPE-train / POPE-test split.
- **Threat to CG-VL**: LOW (benchmark, not a method).
- **Differentiator**: CG-VL uses POPE as the discriminative-side calibration / test set; we add a statistical guarantee on top of POPE-style binary outputs.

### A2. AMBER — LLM-free Multi-dimensional Hallucination Benchmark
- **Authors / Venue / Year**: J. Wang et al., arXiv 2023 (2311.07397)
- **arXiv**: https://arxiv.org/abs/2311.07397
- **TL;DR**: Generative + discriminative tasks across existence, attribute, relation; LLM-free pipeline cuts evaluation cost. Standard 2024 baseline.
- **Threat to CG-VL**: LOW (benchmark).
- **Differentiator**: CG-VL uses AMBER-train as a calibration set; AMBER does not provide statistical guarantees.

### A3. HallusionBench (CVPR 2024)
- **Authors / Venue / Year**: Guan, Liu, et al., CVPR 2024
- **arXiv**: https://arxiv.org/abs/2310.14566
- **TL;DR**: 346 images / 1129 paired questions targeting language hallucination vs. visual illusion; strong control-group structure. GPT-4V scores 31.4% question-pair accuracy; everyone else <16%.
- **Threat**: LOW (benchmark).
- **Differentiator**: Hard test set for CG-VL coverage claims; HallusionBench has no calibration / guarantee layer.

### A4. MMHal-Bench
- **Authors / Venue / Year**: Sun et al. (LLaVA-RLHF), 2023
- **arXiv**: https://arxiv.org/abs/2309.14525
- **TL;DR**: 96 image-question pairs across 12 COCO meta-categories; uses GPT-4 as judge for fine-grained errors (attribute, count, spatial).
- **Threat**: LOW (benchmark).

### A5. M-HalDetect
- **Authors / Venue / Year**: Gunjal, Yin, Bas, AAAI 2024
- **arXiv**: https://arxiv.org/abs/2308.06394
- **TL;DR**: Fine-grained span-level hallucination annotations (accurate / inaccurate / analytical) for FDPO and reward-model training.
- **Threat**: LOW.

### A6. CHAIR — Object Hallucination in Image Captioning
- **Authors / Venue / Year**: Rohrbach, Hendricks, Burns, Darrell, Saenko — EMNLP 2018
- **arXiv**: https://arxiv.org/abs/1809.02156
- **TL;DR**: CHAIRi (instance) / CHAIRs (sentence) ratio of hallucinated objects; foundational caption-level metric.
- **Threat**: LOW.

### A7. MM-Vet — Integrated VL Capabilities
- **Authors / Venue / Year**: Yu et al., ICML 2024
- **arXiv**: https://arxiv.org/abs/2308.02490
- **TL;DR**: 6 core VL skills × 16 integrations; LLM-as-judge scoring for open-ended tasks.
- **Threat**: LOW (general benchmark).

### A8. MMBench
- **Authors / Venue / Year**: Liu et al., ECCV 2024
- **TL;DR**: ~3000 MCQs across 20 ability dimensions, English+Chinese; rigorous CircularEval protocol.
- **Threat**: LOW.

### A9. Bingo — Holistic Analysis of Hallucination in GPT-4V
- **Authors / Venue / Year**: Cui, Zhou, et al., 2023
- **arXiv**: https://arxiv.org/abs/2311.03287
- **TL;DR**: Bias (regional/text-language) + Interference (leading questions, multi-image confusion) probes; GPT-4V-centric.
- **Threat**: LOW.

### A10. R-Bench — Vision Relationship Hallucination
- **Authors / Venue / Year**: Wu et al., ACL 2024
- **arXiv**: https://arxiv.org/abs/2406.16449
- **TL;DR**: Probes subject-relationship, relationship-object, relationship-relationship co-occurrence biases; complements POPE which is object-only.
- **Threat**: LOW.

### A11. THRONE — Object Hallucination in Free-form Generations
- **Authors / Venue / Year**: Kaul et al., CVPR 2024 (Amazon)
- **arXiv**: https://arxiv.org/abs/2405.05256
- **TL;DR**: Type-I (free-form) hallucination metric using public LMs to extract objects; precision/recall/F1 metrics on free-form captions.
- **Threat**: LOW.

### A12. HQH / HQHBench
- **Authors / Venue / Year**: Liu et al., 2024
- **arXiv**: https://arxiv.org/abs/2406.17115
- **TL;DR**: Meta-evaluates the *quality* of existing hallucination benchmarks — useful when justifying CG-VL's choice of POPE/AMBER.
- **Threat**: LOW.

### A13. ODE — Open-Set Evaluation of Hallucinations
- **Authors / Venue / Year**: Tu et al., CVPR 2025
- **TL;DR**: Open-set polling against synthetic/uncommon objects to dodge data-contamination on POPE.
- **Threat**: LOW.

### A14. HaluEval / HalluLens (LLM, not VLM)
- **Authors / Venue / Year**: Li et al., EMNLP 2023; Bang et al., ACL 2025
- **TL;DR**: LLM-only hallucination benchmarks; included for cross-modal positioning.

---

## B. Contrastive decoding / inference-time mitigation

### B1. VCD — Visual Contrastive Decoding (CVPR 2024 Highlight)
- **Authors**: Leng, Zhang, Tian, Zhang, Hu, Bing, et al.
- **arXiv**: https://arxiv.org/abs/2311.16922
- **TL;DR**: At decoding time, contrasts logits from original image vs. *Gaussian-noised image* to suppress language priors. Training-free, plug-and-play.
- **Threat**: **MED–HIGH**. VCD's "noised image as null" is structurally similar to CG-VL's `I̅` (zero/noise/shuffle masking). Both use a contrastive log-ratio.
- **Differentiator**: VCD is a *decoding heuristic* with no statistical guarantee — its α ↔ hallucination-rate mapping is empirical. CG-VL uses the same kind of log-ratio as a *conformity score* and produces a finite-sample bound on hallucination rate. CG-VL also scopes "shuffle" as default + ablations on zero/noise/patch.

### B2. ICD — Instruction Contrastive Decoding (ACL Findings 2024)
- **Authors**: X. Wang et al.
- **arXiv**: https://arxiv.org/abs/2403.18715
- **TL;DR**: Contrasts logits with vs. without an *adversarial system prompt* (e.g., "you are a confused object detector").
- **Threat**: MED. Same contrastive philosophy but on instruction perturbation, not image perturbation.
- **Differentiator**: CG-VL perturbs the image (PMI proper); ICD perturbs the prompt. No CP guarantee.

### B3. M3ID — Multi-Modal Mutual-Information Decoding (CVPR 2024)
- **Authors**: Favero, Zancato, Trager, Choudhary, Perera, Achille, Swaminathan, Soatto
- **arXiv**: https://arxiv.org/abs/2403.14003
- **TL;DR**: Amplifies image–token mutual information at sampling time by contrasting image-conditioned vs. unconditioned next-token probs; can pair with DPO. -25% hallucinated objects on LLaVA-13B captioning.
- **Threat**: **HIGH**. The same `log p(y|I,q) − log p(y|q)` quantity that CG-VL uses as a conformity score is M3ID's *decoding objective*. CG-VL's "image-shuffle PMI" is a more robust variant of this signal.
- **Differentiator**: M3ID is a sampling intervention with no statistical guarantee; CG-VL repurposes the same signal as a *post-hoc conformity score* and produces a distribution-free coverage bound. CG-VL's `I_shuffle` is a stronger null than M3ID's prompt-only marginalization.

### B4. OPERA (CVPR 2024 Highlight)
- **Authors**: Huang et al.
- **arXiv**: https://arxiv.org/abs/2311.17911
- **TL;DR**: Beam-search penalty on "summary token over-trust" patterns + retrospection rollback. No external data.
- **Threat**: MED. Strong mitigation baseline.
- **Differentiator**: Attention-pattern heuristic; no calibration guarantee.

### B5. HALC — Adaptive Focal-Contrast Decoding (ICML 2024)
- **Authors**: Z. Chen, Zhao, Luo, Yao, Li, Zhou
- **arXiv**: https://arxiv.org/abs/2403.00425
- **TL;DR**: Auto-focal grounding: chooses different image FOVs based on token probabilities, contrasts FOV pairs, plus matching beam-search.
- **Threat**: MED.
- **Differentiator**: Cropping-as-counterfactual; no finite-sample guarantee.

### B6. DoLa — Decoding by Contrasting Layers
- **Authors**: Chuang et al., ICLR 2024
- **arXiv**: https://arxiv.org/abs/2309.03883
- **TL;DR**: Contrasts late vs. early layer logits; LLM-native, but adapted for VLMs in many follow-ups (LISA, LOL, Lower-Layers-Matter).
- **Threat**: LOW (LLM-side baseline; weak on multimodal grounding).

### B7. VOLCANO — Self-Feedback Guided Revision (NAACL 2024)
- **Authors**: Lee, Park, Jo, Seo (KAIST)
- **arXiv**: https://arxiv.org/abs/2311.07362
- **TL;DR**: Iterative critique-revision-decide loop using a single LMM. SOTA at release on MMHal-Bench, POPE.
- **Threat**: LOW–MED.

### B8. LURE — LVLM Hallucination Revisor (ICLR 2024)
- **Authors**: Y. Zhou et al.
- **arXiv**: https://arxiv.org/abs/2310.00754
- **TL;DR**: Post-hoc revisor model trained on synthetic hallucinated data flagged by co-occurrence / uncertainty / position. +23% over prior best.
- **Threat**: LOW–MED (training-time + revisor approach).

### B9. Pensieve — Retrospect-then-Compare
- **Authors**: Yang et al., 2024
- **arXiv**: https://arxiv.org/abs/2403.14401
- **TL;DR**: Retrieves visually similar reference images and contrasts decoding distributions to suppress shared hallucinations.
- **Threat**: MED. Uses image-similar-images as a counterfactual signal.

### B10. RITUAL — Random Image Transformations
- **Authors**: Woo, Kim et al., 2024
- **arXiv**: https://arxiv.org/abs/2405.17821
- **TL;DR**: Generates predictions on multiple random image transforms and *adds* their logits — opposite sign of VCD.
- **Threat**: LOW.

### B11. AvisC — Attentional Vision Calibration (Findings ACL 2025)
- **Authors**: Woo et al., 2024
- **arXiv**: https://arxiv.org/abs/2405.17820
- **TL;DR**: Detects "blind tokens" with abnormal attention and re-weights their logits via contrastive decoding.
- **Threat**: MED.

### B12. IBD — Image-Biased Decoding
- **Authors**: Zhu et al., 2024
- **arXiv**: https://arxiv.org/abs/2402.18476
- **TL;DR**: Trains an image-biased model and contrasts with the standard model — closer to a "positive amplification" formulation of PMI.
- **Threat**: **HIGH**. IBD's "image-biased − standard" is structurally a PMI estimate. Same math family as CG-VL's score.
- **Differentiator**: IBD requires *fine-tuning* the biased model; CG-VL is post-hoc and provides distribution-free guarantee on top.

### B13. SID — Self-Introspective Decoding (ICLR 2025)
- **Authors**: Huo et al.
- **arXiv**: https://arxiv.org/abs/2408.02032
- **TL;DR**: CT2S strategy keeps only least-important vision tokens after early decoder layers to *amplify* hallucinations, then contrasts.
- **Threat**: MED.

### B14. DeCo — Dynamic Correction Decoding (ICLR 2025)
- **Authors**: zjunlp team
- **arXiv**: https://arxiv.org/abs/2410.11779
- **TL;DR**: Dynamically corrects logits using mid-layer signals.
- **Threat**: LOW–MED.

### B15. Delve into VCD (analysis paper, 2024)
- **arXiv**: https://arxiv.org/abs/2412.06775
- **TL;DR**: Empirical analysis of when VCD helps vs. hurts; useful for justifying CG-VL ablations on masking strategy.
- **Threat**: LOW.

### B16. Attention Reallocation (IJCV 2025)
- **arXiv**: https://arxiv.org/abs/2503.08342
- **TL;DR**: Zero-cost attention re-weighting toward image tokens.

---

## C. Attention / internal-state interventions

### C1. PAI — Paying More Attention to Image (ECCV 2024)
- **Authors**: Liu et al.
- **arXiv**: https://arxiv.org/abs/2407.21771
- **TL;DR**: Up-weights attention on image tokens + subtracts pure-text logits. Identifies "text inertia" as core hallucination cause.
- **Threat**: MED. Subtraction of pure-text logits ≈ a PMI variant.
- **Differentiator**: PAI is a decoding intervention; CG-VL is a conformity-score wrapper with guarantee.

### C2. VTI — Latent Space Steering (ICLR 2025)
- **Authors**: Liu et al.
- **arXiv**: https://arxiv.org/abs/2410.15778
- **TL;DR**: Computes per-layer steering vectors that stabilize vision features under perturbation.
- **Threat**: MED.

### C3. VISTA — Visual Information Steering (ICML 2025)
- **arXiv**: https://arxiv.org/abs/2502.03628
- **TL;DR**: Identifies gradual visual info loss / early excitation; reinforces vision in activation space + early-layer logits. ~40% hallucination reduction.
- **Threat**: MED.

### C4. Activation Steering Decoding (ACL 2025)
- **arXiv**: https://aclanthology.org/2025.acl-long.634/
- **TL;DR**: Identifies hallucination directions in activation space using a small calibration set — *uses calibration like CG-VL, but no statistical guarantee*.
- **Threat**: MED.

### C5. Dynamic Multimodal Activation Steering
- **arXiv**: https://arxiv.org/abs/2602.21704
- **TL;DR**: Semantic-context-aware steering vectors selected at inference.

### C6. AGLA — Assembly of Global & Local Attention
- **arXiv**: https://arxiv.org/abs/2406.12718
- **Threat**: LOW.

### C7. MDSAM — Memory-Driven Sparse Attention (2025)
- **arXiv**: https://arxiv.org/abs/2506.17664

---

## D. Training-time fixes (DPO / RLHF / instruction tuning)

### D1. RLHF-V (CVPR 2024)
- **Authors**: T. Yu et al.
- **arXiv**: https://arxiv.org/abs/2312.00849
- **TL;DR**: Segment-level human corrections + dense DPO. -34.8% hallucination with 1.4k samples.
- **Threat**: LOW–MED (training-side).

### D2. RLAIF-V (CVPR 2025 Highlight)
- **arXiv**: https://arxiv.org/abs/2405.17220
- **TL;DR**: Open-source AI-feedback DPO; achieves super-GPT-4V trustworthiness on MMHal-Bench.
- **Threat**: LOW–MED.

### D3. HALVA — Data-Augmented Contrastive Tuning (NeurIPS 2024)
- **Authors**: Sarkar et al.
- **arXiv**: https://arxiv.org/abs/2405.18654
- **Threat**: LOW.

### D4. mDPO — Conditional Preference Optimization (EMNLP 2024)
- **Authors**: F. Wang et al.
- **arXiv**: https://arxiv.org/abs/2406.11839
- **TL;DR**: Adds image-preference loss + reward anchor on top of DPO; explicitly addresses "language-only preference" issue.
- **Threat**: LOW–MED.

### D5. POVID — Preference Optimization with Visual Input Degradation
- **arXiv**: https://arxiv.org/abs/2402.11411
- **TL;DR**: Synthetic dispreferred responses from noised images.
- **Threat**: MED. Image-noise as a perturbation is similar in spirit to CG-VL's I̅.
- **Differentiator**: POVID uses degraded images to *train*; CG-VL uses them as a *test-time null*.

### D6. V-DPO — Vision-Guided DPO (Findings EMNLP 2024)
- **arXiv**: https://arxiv.org/abs/2411.02712

### D7. OPA-DPO — On-Policy Alignment DPO (CVPR 2025 Oral)
- **Authors**: Yang et al.
- **arXiv**: https://arxiv.org/abs/2501.09695
- **TL;DR**: -13.26% on AMBER with only 4.8k on-policy samples for LLaVA-1.5-7B.
- **Threat**: LOW–MED.

### D8. Silkie — Preference Distillation
- **arXiv**: https://arxiv.org/abs/2312.10665
- **TL;DR**: VLFeedback dataset + GPT-4V scoring; DPO on Qwen-VL-Chat.

### D9. SeVa — Self-Supervised Visual Preference Alignment (ACM MM 2024)
- **TL;DR**: Self-paired chosen/rejected from augmented images; no human labels.
- **Threat**: LOW.

### D10. HACL — Hallucination Augmented Contrastive Learning (CVPR 2024)
- **Authors**: Jiang et al.
- **arXiv**: https://arxiv.org/abs/2312.06968
- **TL;DR**: GPT-4-generated hallucinated captions as hard negatives for image-text contrastive alignment.
- **Threat**: LOW.

### D11. EOS Decision (ACL 2024) — "Less is More"
- **Authors**: Yue et al.
- **arXiv**: https://arxiv.org/abs/2402.14545
- **TL;DR**: Hallucinations partly arise from suppressed EOS; debiased training shortens captions, reducing hallucination tail.
- **Threat**: LOW.

### D12. LRV-Instruction (ICLR 2024)
- **Authors**: F. Liu et al.
- **arXiv**: https://arxiv.org/abs/2306.14565
- **TL;DR**: 120k visual instructions including negative (nonexistent / existent manipulation) examples; introduces GAVIE evaluator.

### D13. Woodpecker (Sci China 2024)
- **arXiv**: https://arxiv.org/abs/2310.16045
- **TL;DR**: 5-stage post-hoc correction (key concepts, questions, visual validation, claim gen, correction).
- **Threat**: LOW.

### D14. HalluciDoctor (CVPR 2024)
- **TL;DR**: Counterfactual instruction-data augmentation.

---

## E. VLM uncertainty / grounding measurement

### E1. Multi-Modal Hallucination Control by Visual Information Grounding — same as M3ID (CVPR 2024)
- See B3.
- **Threat**: **HIGH** (most overlapping prior work).

### E2. C-PMI — Conditional PMI Calibrated Decoding (NeurIPS 2025)
- **Authors**: 2025
- **arXiv**: https://arxiv.org/abs/2505.19678
- **TL;DR**: Bi-level optimization that maximizes Conditional PMI between visual and textual tokens during decoding; explicitly named "C-PMI", uses joint visual+textual token contributions.
- **Threat**: **HIGH**. Closest-named work to CG-VL's signal. Uses PMI as decoding objective on LLaVA-1.5.
- **Differentiator**: C-PMI optimizes PMI during sampling. CG-VL uses PMI as a *post-hoc* score and adds **distribution-free CP guarantee on hallucination rate**. C-PMI has no statistical guarantee or abstention mechanism.

### E3. Inductive CP for LVLMs (Wang et al., 2025)
- **arXiv**: https://arxiv.org/abs/2504.17671
- **TL;DR**: Split CP framework for LVLM VQA with dynamic threshold + cross-modal consistency. Evaluated on ScienceQA / MMMU with 8 LVLMs.
- **Threat**: **HIGH**. This *is* a CP-for-VLM-hallucination paper.
- **Differentiator**: Their setting is **closed-ended multi-choice VQA** (prediction-set construction). CG-VL targets **open-ended generation hallucination control via abstention**, and uses a **PMI-style image-conditioning conformity score** rather than softmax. Different score function, different problem (set-valued vs. abstention).

### E4. Conformal Prediction for Zero-Shot Models (CVPR 2025)
- **Authors**: Silva-Rodríguez, Ben Ayed, Dolz
- **arXiv**: https://arxiv.org/abs/2505.24693
- **TL;DR**: Conf-OT — split CP over CLIP zero-shot classification with optimal-transport bridging cal/test domain gap.
- **Threat**: MED. CP for VLMs but classification setting, not generative.
- **Differentiator**: CG-VL is generative VLM hallucination, not zero-shot CLIP classification.

### E5. Full Conformal Adaptation of Medical VLMs (Springer 2024)
- **Threat**: LOW–MED.

### E6. The Art of Saying "Maybe" — Conformal Lens for VLM Uncertainty
- **arXiv**: https://arxiv.org/abs/2509.13379
- **TL;DR**: Conformal benchmark for 16 VLMs (Llama-4, Qwen2.5-VL, InternVL3, Molmo, Pixtral, etc.). Useful citation for breadth.
- **Threat**: MED.

### E7. MTRE — Multi-Token Reliability Estimation
- **arXiv**: https://arxiv.org/abs/2505.11741
- **TL;DR**: Aggregates first-10 token log-likelihood ratios + self-attention for hallucination detection.
- **Threat**: MED. *Log-likelihood ratios* over early tokens overlap with CG-VL's token-level PMI variant.
- **Differentiator**: MTRE is a *detector* (binary), no coverage guarantee.

### E8. Self-Consistency as a Free Lunch (TPAMI 2025)
- **arXiv**: https://arxiv.org/abs/2509.23236
- **TL;DR**: Internal consistency between short-answer and long-description as training signal.
- **Threat**: MED. Touches CG-VL's "self-consistency between image and image-masked" variant in spirit.
- **Differentiator**: They consistency-check between two text outputs; CG-VL consistency-checks between image-conditioned and image-shuffle-conditioned generations.

### E9. Evidential Uncertainty for LVLMs
- **arXiv**: https://arxiv.org/abs/2602.05535
- **TL;DR**: Two epistemic-uncertainty types (conflict / ignorance) via evidential learning.

### E10. EnsemHalDet — Ensemble of Internal-State Detectors
- **arXiv**: https://arxiv.org/abs/2604.02784

### E11. VADE — Visual Attention Guided Hallucination Detection (Findings ACL 2025)
- **arXiv**: https://aclanthology.org/2025.findings-acl.773/
- **Threat**: MED. Attention-on-image as detector — overlaps with CG-VL's "attention-on-image" variant.

### E12. FaithScore (Findings EMNLP 2024)
- **TL;DR**: Atomic-fact extraction + visual entailment; reference-free metric.
- **Threat**: LOW (eval metric, not method).

### E13. ALOHa — A New Measure for Caption Hallucination (NAACL 2024)
- **arXiv**: https://arxiv.org/abs/2404.02904

### E14. Treble Counterfactual VLMs (causal hallucination)
- **arXiv**: https://openreview.net/forum?id=GDRUOkk8EV
- **TL;DR**: NDE per modality via counterfactual graph; image counterfactual is structurally similar to I̅.
- **Threat**: MED.

### E15. Wang+ 2024 — VLM Confidence Calibration
- **arXiv**: https://arxiv.org/abs/2604.02543 (medical VQA overconfidence, 2025)
- **TL;DR**: Shows hallucination signals improve calibration AUROC; supports CG-VL's premise.
- **Threat**: LOW.

### E16. Mitigating Hallucinations via Conformal Abstention (LLM, 2024)
- **arXiv**: https://arxiv.org/abs/2405.01563
- **TL;DR**: Conformal abstention with theoretical guarantees on LLM hallucination rate via SE-based scores.
- **Threat**: **HIGH** for novelty. CG-VL is the natural multimodal extension — but for *VLM* and with a *PMI* score, not semantic entropy.
- **Differentiator**: CG-VL targets VLMs; uses PMI (cross-modal grounding) as score; image-shuffle as null. The LLM paper uses semantic-entropy clustering — not applicable to single-image grounding.

---

## F. Recent VLM models (positioning)

| Model | Authors / Venue | arXiv | Note |
|---|---|---|---|
| Qwen2.5-VL | S. Bai et al., 2025 | 2502.13923 | 3B/7B/72B; native dynamic-res ViT; SOTA on doc/charts/long-video. CG-VL primary backbone. |
| Qwen2.5-Omni | J. Xu et al., 2025 | 2503.20215 | Thinker-Talker; text+image+audio+video; comparable VL to Qwen2.5-VL. |
| LLaVA-1.6 / LLaVA-NeXT | Liu et al., 2024 | blog | Higher-res (672²/1344×336/336×1344). |
| LLaVA-OneVision | Li et al., 2024 | 2408.03326 | SigLIP + Qwen2; single/multi-image/video. |
| Idefics-2 | Laurençon et al., NeurIPS 2024 | 2405.02246 | 8B; native aspect-ratio. |
| Idefics-3 | Laurençon et al., 2024 | 2408.12637 | +13.7 DocVQA over Idefics-2; Docmatix dataset. |
| InternVL-2.5 | Z. Chen et al., 2024 | 2412.05271 | Progressive scaling; up to 78B. |
| InternVL-3 | 2025 | 2504.10479 | Native multimodal pretraining + V2PE; 72.2 MMMU. |
| Molmo / PixMo | Deitke et al., 2024 | 2409.17146 | Open-data; pointing dataset. |
| Pixtral-12B | Mistral, 2024 | 2410.07073 | Apache-2.0 multimodal. |
| GPT-4V / 4o | OpenAI | — | Closed-source comparison. |
| Gemini-Vision | Google | — | Closed-source. |

All of these are *backbones*, not hallucination methods — orthogonal to CG-VL.

---

## G. Most threatening to CG-VL novelty (consolidated view)

These are works that overlap with CG-VL on (i) PMI / image-conditional likelihood ratio, (ii) CP for VLM hallucination, (iii) image-shuffle / counterfactual-image methods. See Section H for elaboration.

---

## H. HIGH-threat papers (consolidated)

### H1. M3ID (Favero+, CVPR 2024) — https://arxiv.org/abs/2403.14003
**Why threatening**: Uses `log p(y|I,q) − log p(y|q)` as a *decoding-time* mutual-information signal — same quantity CG-VL uses.
**CG-VL differentiator**: (a) M3ID has no statistical guarantee on hallucination rate; CG-VL gives `P(hallucination | not abstain) ≤ α` finite-sample. (b) M3ID's null is "no image" / unconditioned LM; CG-VL's default null `I_shuffle` keeps low-level statistics matched, so the score isolates *spatial-semantic* grounding rather than just modality-presence. (c) CG-VL is post-hoc and abstention-based, not a sampling change.

### H2. C-PMI (NeurIPS 2025) — https://arxiv.org/abs/2505.19678
**Why threatening**: Explicitly named PMI/MI for VLM hallucination with bi-level optimization.
**CG-VL differentiator**: C-PMI solves an optimization at decode time. CG-VL uses PMI as a *fixed conformity score* and gets statistical guarantees by calibration on POPE-/AMBER-train splits. C-PMI cannot promise `α`-bounded hallucination — CG-VL can.

### H3. Inductive CP for LVLMs (Z. Wang+, 2025) — https://arxiv.org/abs/2504.17671
**Why threatening**: This *is* "CP for VLM hallucination". Direct overlap.
**CG-VL differentiator**: Their setting is **closed-ended multi-choice VQA** producing prediction *sets*; their score is softmax-style over choices. CG-VL's setting is **open-ended generation** with **abstention** rather than set output, and uses a **PMI grounding score** specifically built for image-conditioning — not softmax. Different problem class, different score function. Should be cited as the closest CP-for-VLM precedent.

### H4. IBD — Image-Biased Decoding (2024) — https://arxiv.org/abs/2402.18476
**Why threatening**: `image-biased model − base model` ≈ PMI in disguise.
**CG-VL differentiator**: IBD requires *training* a biased model. CG-VL is training-free: just compare `p(a|I,q)` to `p(a|I_shuffle,q)` from the same model. Plus CG-VL adds CP guarantees.

### H5. Mitigating LLM Hallucinations via Conformal Abstention (2024) — https://arxiv.org/abs/2405.01563
**Why threatening**: Same problem statement (CP-controlled hallucination via abstention) but for LLMs.
**CG-VL differentiator**: CG-VL is the *VLM* extension and uses an image-grounding PMI score that has no LLM analog. The LLM paper uses semantic-entropy clustering of multiple samples, which does not exploit image-conditioning. CG-VL's PMI is single-pass and cheap.

### H6. PAI (ECCV 2024) — https://arxiv.org/abs/2407.21771
**Why threatening**: Subtracts pure-text logits from multimodal logits — algebraically a PMI.
**CG-VL differentiator**: PAI is a decoding intervention with no calibration / α-guarantee. CG-VL is post-hoc on top of *any* decoder.

### H7. VCD (Leng+, CVPR 2024) — https://arxiv.org/abs/2311.16922
**Why threatening**: Image-perturbed (Gaussian noise) contrastive decoding — same family as CG-VL's noise/zero/shuffle nulls.
**CG-VL differentiator**: VCD modifies sampling distribution; no abstention, no guarantee. CG-VL uses the contrast as a conformity score at the *answer* level and conformally calibrates abstention.

### H8. HalluSegBench (2025) — https://arxiv.org/abs/2506.21546
**Why threatening**: Counterfactual-image evaluation framework — formalizes "image-shuffle"-type interventions for grounding.
**CG-VL differentiator**: HalluSegBench is a benchmark for pixel-grounding (segmentation); CG-VL is for VQA-style generation. Useful as a *companion benchmark*; not a competing method.

---

## I. Benchmark SOTA snapshot (as of early 2026)

### POPE (accuracy / F1, average across random/popular/adversarial)
- **Best closed**: GPT-4o ≳ 90% acc.
- **Best open-source 7B-class with mitigation**: OPA-DPO (CVPR'25, LLaVA-1.5-7B + on-policy DPO) — reports SOTA on POPE & AMBER among 7B tier; ~89-90% acc.
- **Best open-source large**: InternVL3-78B / Qwen2.5-VL-72B near-saturate POPE (>92%).
- **Best training-free decoding mitigation**: VISTA / SID / C-PMI variants on LLaVA-1.5 give 4–8 point gains over base; C-PMI claims best on LLaVA-1.5 across 5 benchmarks.

### AMBER
- **Best 7B mitigation**: OPA-DPO -13.26% hallucination over prior SOTA on LLaVA-1.5-7B.
- **Best frontier**: GPT-4V / Gemini-Vision; RLAIF-V-12B claims super-GPT-4V on the discriminative half.

### HallusionBench (question-pair acc)
- **Best**: GPT-4o ≈ 50–55%; Qwen2.5-VL-72B ≈ 58–60% (per Qwen2.5-VL TR); InternVL3-78B in similar range. Open-source 7B models still <30% on hardest splits.

### MMHal-Bench
- **Best**: RLAIF-V-12B reports super-GPT-4V scores on overall hallucination rate.

### Take-away for CG-VL
The above are absolute-accuracy / hallucination-rate numbers without coverage promises. **No existing method ships an `α`-controlled hallucination rate.** That is the niche CG-VL claims.

---

## J. Synthesis — what gap CG-VL fills

1. **PMI as a signal already proven**. M3ID (CVPR'24), C-PMI (NeurIPS'25), IBD, PAI, VCD, ICD all show that some form of `log p(a|I) − log p(a|I̅)` (or its prompt/text-inertia analog) is a useful hallucination signal. None of them turn this into a **statistical guarantee** on hallucination rate.

2. **CP for generation has been done for LLMs but barely for VLMs**. The conformal-abstention LLM paper (2024), conformal-language-modeling ICLR'24, and split-CP-for-LVLM-VQA (2025) cover adjacent settings. The 2025 LVLM-VQA paper is closest but addresses *closed-set* multi-choice with softmax scores. **CP for open-ended VLM hallucination control via abstention with a PMI score is genuinely uncovered.**

3. **Image-shuffle / image-counterfactual as the null is novel in this combination**. Existing works use Gaussian noise (VCD), prompt-only marginalization (M3ID), text-inertia (PAI), random transforms (RITUAL), or counterfactual semantic replacement (HalluSegBench, Treble Counterfactual). CG-VL's planned ablation over shuffle/zero/noise/patch as image masking + token-level / self-consistency / attention-on-image variants is a useful *unification* that has not been studied as a single conformity-score family.

4. **Coverage promise is the thing**. Practitioners care about a tunable knob: "give me at most α% hallucination rate when the system answers." OPERA, VCD, M3ID, OPA-DPO, RLAIF-V all *reduce* hallucination but cannot promise α. CG-VL's core contribution is **converting an existing strong PMI-style score into a conformity score and certifying `P(hallucination | abstain=False) ≤ α` distribution-free.**

### Bottom line
**Method-side novelty is moderate** (PMI scores are already explored). **Statistical-guarantee novelty is high**: no prior VLM-hallucination work gives finite-sample, distribution-free α-control with a PMI-style score and image-counterfactual null. Position CG-VL as: *"CP-wrapper that turns the M3ID/IBD/VCD family of PMI-style signals into a certifiably α-bounded hallucination controller."* Cite C-PMI, M3ID, VCD, IBD, PAI, and the 2025 LVLM-VQA-CP paper as closest related work and explicitly contrast on (i) score function, (ii) guarantee, (iii) abstention vs. set-valued vs. sampling change.

---

## File pointers for downstream agents
- POPE/AMBER calibration sets: official repos `RUCAIBox/POPE`, `junyangwang0410/AMBER`.
- VCD / OPERA / HALC / SID / PAI / VTI / VISTA: all have public code (see arXiv links above).
- For Qwen2.5-VL backbone: `QwenLM/Qwen2.5-VL`.
- Surveys to cite: Bai+ 2024 (2404.18930), H. Liu+ 2024 (2402.00253).

# Cross-Cutting Literature Synthesis (한글) — Conformal Prediction for LLM Uncertainty

> 4개 영역 서베이 (`01–04*.md`) 위에서 작성. 목표: 프로젝트별 위협 평가, novelty claim, must-cite / must-baseline, 12개월 계획에 대한 전략적 시사점.

---

## 0. TL;DR

| 프로젝트 | 평가 | 핵심 포지셔닝 한 문장 |
|---|---|---|
| **CoT-CP** | ✅ 가장 깨끗한 빈 자리 (open white space) | *Reasoning chain 의 step-level CRC 를 처음 제시. 기존 heuristic step-confidence 신호 (DeepConf, DEER, ESC, PAVs) 를 calibrated, distribution-free, sequence-level coverage guarantee 로 변환.* |
| **CoT-Tool** | ⚠️ 방어 가능하나 경쟁 치열 — framing 더 날카롭게 | *Tool-trigger 결정에 대한 CP wrapper. Conformal Abstention (action: refuse) 와 TRAQ (target: post-retrieval answer set) 을 보완하여, **no-tool branch** 의 hallucination 을 제어.* |
| **CG-VL** | ⚠️ 방법 (PMI) 자체는 이미 taken — guarantee framing 이 contribution | *Open-ended VLM hallucination 에 대한 첫 α-bound 제어기. M3ID/C-PMI/VCD/IBD/PAI 계열의 PMI-style decoding 신호를 conformity score 로 변환하여 finite-sample coverage 부여.* |

**전략 권장**: 원래 plan 의 **CG-VL → CoT-Tool → CoT-CP** 순서 유지. 다만 각 프로젝트의 "core contribution" 을 "we propose X" 가 아니라 "we **calibrate** existing X with a guarantee" 로 reframe. 세 프로젝트 다 wrapper 인 게 사실이고, 그걸 솔직히 own 해야 함.

---

## 1. Project 1 — CoT-CP

### 1.1 위협적인 5개 논문 (HIGH-threat — 반드시 명시적 positioning)

| # | 논문 | Venue | 위협 이유 | CoT-CP 가 추가하는 것 |
|---|---|---|---|---|
| 1 | **DeepConf** (Fu et al. 2025, [2508.15260](https://arxiv.org/abs/2508.15260)) | arXiv 2025 | AIME/GPQA 에서 span-confidence filtering 으로 99.9% 정확도, 84% 토큰 절감 — 동일 벤치마크 saturate | Distribution-free CRC on 최종 답. DeepConf 는 coverage 보장 없음 |
| 2 | **DEER: Dynamic Early Exit in Reasoning Models** ([2504.15895](https://arxiv.org/abs/2504.15895)) | arXiv 2025 | R1-distill 등에서 step-level confidence 기반 early exit | PRM800K 로 calibrated $\hat{q}$ vs hand-tuned threshold |
| 3 | **Adaptive-Consistency / ESC** (Aggarwal+ EMNLP 2023; Li+ 2401.10480) | EMNLP 2023 / arXiv 2024 | Stopping rule 로 적응적 sample 수 결정 | Step-level (prompt-level 아님), frequentist coverage (Bayesian heuristic 아님) |
| 4 | **Snell et al., Scaling Test-Time Compute Optimally** ([2408.03314](https://arxiv.org/abs/2408.03314)) | arXiv 2024 | Compute-optimal-adaptive 의 정신적 조상 | Step-level adaptation + sequence-level CRC, budget 재배분 그 이상 |
| 5 | **Rewarding Progress / PAVs** (Setlur et al. [2410.08146](https://arxiv.org/abs/2410.08146)) | arXiv 2024 | Step-level verifier-guided adaptive search | PAV score 를 calibrate 해서 coverage-controlled gate 로 변환 |

**CP-side 이웃** (이론 위협, 더 가벼움):
- **Conformal Language Model Reasoning with Coherent Factuality** (Rubin-Toles et al., **ICLR 2025**) — deducibility graph 로 reasoning 에 CP 적용한 첫 논문. 가장 가까운 이론 이웃.
- **CoVeR: Conformal Calibration for Versatile Next-Token Prediction** (arXiv 2509.04733, 2025) — token-cluster 단위 step calibration. Unit 이 다름.
- **Differentiable Conformal Training** (2026 preprint) — *지켜볼 것*. ICML 2026 채택되면 primary citation.

### 1.2 반드시 비교해야 할 baseline (실험)

- Greedy / vanilla CoT
- Self-consistency $N \in \{8, 16, 32\}$
- Best-of-N + PRM (Math-Shepherd, Skywork-PRM)
- **DeepConf** (필수 비교)
- Adaptive-Consistency / ESC
- Step-level early exit (DEER)

### 1.3 Must-cite (foundational)

- Vovk-Gammerman-Shafer 2005 (CP); Angelopoulos-Bates 2021 (gentle intro); Angelopoulos+ 2022 (CRC); Tibshirani+ 2019 (weighted CP); Barber+ 2022 (beyond exchangeability)
- Wang+ 2022 (Self-Consistency); Lightman+ 2023 (PRM800K / Let's Verify); Wang+ 2024 (Math-Shepherd); Snell+ 2024
- Rubin-Toles+ 2025 (Coherent Factuality); Mohri & Hashimoto 2024 (Conformal Factuality); Quach+ 2024 (Conformal LM)

### 1.4 날카롭게 만들기

**문헌 전체가 한 가지 메시지로 수렴**: step-level 신호는 존재하고, 모두가 쓰고 싶어하지만, 아무도 calibrate 하지 않음. 이건 드물게 깨끗한 빈 자리. 이 점을 강하게 드러낼 것:

CoT-CP 를 "새로운 score (S-SC/S-PRM/S-LP) 제안" 으로 팔지 말고, **"이런 신호들을 끼워 넣을 수 있는 calibration layer"** 로 팔 것. Score ablation 은 *secondary*, *primary* contribution 은 step-level score → sequence-level coverage 의 CRC machinery. 논문 framing: *"네가 좋아하는 step confidence 를 골라라 — DeepConf 의 것이든, DEER 의 것이든, PRM 의 것이든 — 우리는 그걸 certifiable 하게 만든다."*

---

## 2. Project 2 — CoT-Tool

### 2.1 위협적인 5개 논문 (HIGH-threat)

| # | 논문 | Venue | 위협 이유 | CoT-Tool 이 추가하는 것 |
|---|---|---|---|---|
| 1 | **Mitigating LLM Hallucinations via Conformal Abstention** (Abbasi-Yadkori et al., [2405.01563](https://arxiv.org/abs/2405.01563)) | **NeurIPS 2024** | 동일한 CP-for-hallucination-rate framing. 차이는 action space 뿐 (abstain vs call-tool) | Action 이 다름 ("call tool"), Strategy-A oracle label, hierarchical multi-tool 확장 |
| 2 | **TRAQ** (Li et al. [2307.04642](https://arxiv.org/abs/2307.04642)) | NAACL 2024 | RAG pipeline 에 첫 end-to-end CP | TRAQ 는 *post-retrieval* answer 제어. CoT-Tool 은 *whether to retrieve* 자체를 제어 |
| 3 | **Conformal-RAG / Conditional Conformal Factuality** (Feng et al. [2506.20978](https://arxiv.org/abs/2506.20978)) | SIGIR 2025 | RAG 에 group-conditional CP — hierarchical multi-tool 과 직접 유사 | CoT-Tool 은 retrieval group 이 아니라 *tool taxonomy* (calc/search/code/KB) 에 group-conditioning |
| 4 | **Prune 'n Predict / CROQ + CP-OPT** (Vishwakarma et al.) | ICML 2025 | ToolAlpaca 와 MMLU 에 CP. Tool-routing CP precedent | CROQ 는 MCQ-style tool-set CP. CoT-Tool 은 *gating* (call vs not) + cost-aware CRC |
| 5 | **Conformal Constrained Policy Optimization for Cost-Effective LLM Agents** (arXiv [2511.11828](https://arxiv.org/abs/2511.11828), 2025) | arXiv 2025 | LLM agent 간 cost-aware CP routing | Unit 이 다름 (inter-agent vs tool decision). CoT-Tool 은 single LLM 의 tool 결정 |

**Adaptive-RAG/router 이웃** (CP 아니지만 같은 niche — engineering 위협):
- **SeaKR** (Yao et al., ACL 2025), **UAR** (Cheng et al., EMNLP-Findings 2024), **Adaptive-RAG** (Jeong et al., NAACL 2024), **Self-RAG** (Asai et al., ICLR 2024), **DRAGIN**, **Rowen**, **RA-ISF**, **MetaTool** (Huang et al., ICLR 2024)
- 모두 heuristic. CoT-Tool 은 이들 중 어느 것이든 **base scoring function** 으로 받아서 그 위에 missing CP guarantee 를 얹을 수 있음.

### 2.2 반드시 비교해야 할 baseline

- Always-tool / Never-tool (cost/quality envelope)
- ReAct prompting
- Self-RAG (ICLR 2024)
- Adaptive-RAG (NAACL 2024)
- SeaKR (ACL 2025) 또는 UAR (EMNLP-F 2024) — 가장 강한 hidden-state router
- **Conformal Abstention** (Abbasi-Yadkori 2024, NeurIPS) — 그들의 semantic-entropy score 를 tool-trigger 결정에 맞게 변환. 사과 vs 사과 비교
- MetaTool eval (whether-to-call 에 특화된 유일한 벤치마크)

### 2.3 Must-cite

- Yao+ 2022 (ReAct); Schick+ 2023 (Toolformer); Asai+ 2024 (Self-RAG); Jeong+ 2024 (Adaptive-RAG)
- Mohri & Hashimoto 2024 (Conformal Factuality); Quach+ 2024 (Conformal LM); Abbasi-Yadkori 2024 (Conformal Abstention); Feng+ 2025 (Conformal-RAG)
- BFCL (Yan+, leaderboard); ToolBench / StableToolBench; MetaTool; API-Bank

### 2.4 날카롭게 만들기

**Framing 의 진짜 어려움이 여기 있음.** Conformal Abstention 2024 가 정말 가까움. 구체적 3가지 수: 

1. **Action space 를 신중하게 reframe.** "Abstain" 과 "call tool" 은 형식적으로 다른 action set 이지만, 리뷰어는 framework 가 정말 새로운지 아니면 단순 relabeling 인지 물을 것. 차별화는 **operational** 해야 함: tool-call 은 다른 답을 줌, no-answer 가 아니라. Strategy-A oracle ("LLM-no-tool 틀림 AND LLM-with-tool 맞음") 이 calibration 을 trigger-specific 하게 만드는 것이고, 진짜로 새로운 데이터 구성.

2. **Multi-tool / hierarchical CP 에 기댈 것.** Single-tool gating 은 abstention 과 너무 가까움. **"per-tool risk targets via group-conditional CP"** 로 framing 하면 훨씬 강해짐 — calculator 는 한 $\alpha$, search 는 다른 $\alpha$, code-exec 은 또 다른 $\alpha$. Conformal-RAG style group-conditional CP 를 tool taxonomy 에 적용한 것이고, 아무도 안 함.

3. **Cost 를 CRC loss 로.** Conformal Constrained Policy Optimization (2025) 은 *agent routing* 에 cost-aware CP 를 함. CoT-Tool 은 *tool routing* 에 동일하게 — query 별 budget 을 side metric 이 아니라 CRC formulation 의 constrained loss 로.

**이 axis 들로 sharpen 못 하면 세 프로젝트 중 가장 약함.** Sequencing 에서 deprioritize 고려.

---

## 3. Project 3 — CG-VL

### 3.1 위협적인 5개 논문 (HIGH-threat)

| # | 논문 | Venue | 위협 이유 | CG-VL 이 추가하는 것 |
|---|---|---|---|---|
| 1 | **C-PMI: Conditional PMI Calibrated Decoding for VLMs** ([2505.19678](https://arxiv.org/abs/2505.19678)) | **NeurIPS 2025** | 같은 이름 (PMI), 같은 modality (VLM), 같은 target (hallucination) | C-PMI 는 decoding-time bi-level optimization. CG-VL 은 PMI 를 *fixed conformity score* 로 써서 α-bound |
| 2 | **M3ID** (Favero et al. [2403.14003](https://arxiv.org/abs/2403.14003)) | CVPR 2024 | $\log p(y|I,q)-\log p(y|q)$ 를 sampling objective 로 — 정확히 같은 양 | M3ID 는 α-guarantee 없음. CG-VL 의 null 은 `I_shuffle` (unconditioned LM 아님), spatial-semantic grounding 만 분리. Sampling 아니라 abstention |
| 3 | **Inductive CP for LVLMs** (Z. Wang et al. [2504.17671](https://arxiv.org/abs/2504.17671)) | arXiv 2025 | 직접적인 CP-for-VLM precedent | Closed-set MCQ-VQA, softmax score. CG-VL 은 open-ended generation, PMI score, abstention |
| 4 | **IBD — Image-Biased Decoding** ([2402.18476](https://arxiv.org/abs/2402.18476)) | 2024 | Image-biased model − base model ≈ 위장된 PMI | IBD 는 biased model *학습* 필요. CG-VL 은 training-free, 동일 모델의 단일 forward pair |
| 5 | **PAI — Pay Attention to Image** ([2407.21771](https://arxiv.org/abs/2407.21771)) | ECCV 2024 | Multimodal logit 에서 pure-text logit 빼기 — 대수적으로 PMI | Calibration / α-bound 없음. CG-VL 은 CP wrapper 추가 |

**Adjacent (decoding methods, no guarantee)**: VCD (CVPR 2024), OPERA (CVPR 2024), HALC, SID, ICD, RITUAL, AvisC, VISTA, Pensieve, Treble Counterfactual.

### 3.2 반드시 비교해야 할 baseline

- Vanilla VLM (greedy, sampling)
- **VCD** (CVPR 2024) — Gaussian-noise contrast
- **OPERA** (CVPR 2024) — over-attention penalty
- **M3ID** (CVPR 2024) — score 측면에서 가장 가까움
- **C-PMI** (NeurIPS 2025) — 이름 측면에서 가장 가까움
- POPE-classifier baseline (off-the-shelf hallucination detector)
- **Mitigating LLM Hallucinations via Conformal Abstention** (Abbasi-Yadkori 2024) — image-aware prompt 로 VLM 입력에 적응. *image-shuffle PMI* 가 (semantic-entropy clustering 대비) 정말로 무언가를 사주는지 검증

### 3.3 Must-cite

- Li+ 2023 (POPE); Wang+ 2023 (AMBER); Guan+ 2024 (HallusionBench); Sun+ 2023 (MMHal); Yu+ 2023 (MMVet)
- Leng+ 2024 (VCD); Huang+ 2024 (OPERA); Favero+ 2024 (M3ID); Liu+ 2024 (PAI); Liu+ 2024 (IBD); C-PMI 2025
- Quach+ 2024; Mohri & Hashimoto 2024; Abbasi-Yadkori+ 2024; Z. Wang+ 2025 (Inductive CP for LVLMs)
- VLM backbone: Bai+ 2025 (Qwen2.5-VL); Liu+ 2024 (LLaVA-1.6/NeXT); Idefics-3
- VLM hallucination 서베이: Bai+ 2024 ([2404.18930](https://arxiv.org/abs/2404.18930)), H. Liu+ 2024 ([2402.00253](https://arxiv.org/abs/2402.00253))

### 3.4 날카롭게 만들기

**방법론 novelty 는 moderate, statistical-guarantee novelty 는 high.** 명시적으로 wrapper 로 포지셔닝:

> *"M3ID, C-PMI, IBD, PAI, VCD 등 어떤 PMI-style VLM hallucination 신호든 골라라 — 우리가 그 위에 CP layer 를 씌워서 POPE/AMBER/HallusionBench 에서 $P(\text{hallucination} \mid \text{not abstained}) \le \alpha$ 를 finite-sample, distribution-free 로 제공한다."*

2026 년에 novelty 를 극대화하는 3가지 구체적 수:

1. **Masking ablation 을 first-class contribution 으로.** 기존 work 들은 각자 *하나의* null 만 사용 (VCD 는 Gaussian, M3ID 는 prompt-only, PAI 는 text-inertia). CG-VL 의 shuffle/zero/noise/patch 비교를 통일된 conformity-score family 로 보는 건 진짜 uncovered. *"Image counterfactual as conformity null 의 empirical study"* 로 framing.

2. **Open-ended generation, closed-set 아님.** Inductive CP for LVLMs (2025) 가 이미 closed-set MCQ 를 함. Open-ended (CHAIR-COCO captioning, MMVet GPT-judge) 에 분명히 머무를 것. 여기서 PMI score 가 진가를 발휘.

3. **Selection mode 와 abstention mode 둘 다.** 대부분 CP-LLM 은 prediction set *또는* abstention 만 제공. CG-VL 은 둘 다 제공 가능: (a) low-grounding 답에는 abstain, (b) grounding 이 중간이면 calibrated multi-answer set 반환. 이 dual mode + PMI score 가 깔끔한 방법론적 niche.

---

## 4. Cross-cutting 관찰

### 4.1 세 프로젝트 모두에 걸친 수렴 테마

2024-2026 LLM/VLM uncertainty 문헌의 강한 패턴:

- **Layer 1**: 강한 heuristic 신호가 존재 (PRM score, self-consistency, semantic entropy, attention pattern, PMI score) — 경험적으로 검증됨.
- **Layer 2**: 이 신호들을 결합해 adaptive procedure 를 만듦 (DeepConf, DEER, Self-RAG, M3ID, VCD, OPERA) — statistical guarantee 없음.
- **Layer 3**: 작은 CP-for-LLM 커뮤니티가 finite-sample coverage 를 제공 — 다만 살짝 빗나간 unit (full sequence, claim, abstention, MCQ set).

**세 프로젝트 모두 Layer-2 → Layer-3 변환의 자리에 있음.** 이 패턴이 일관되어서, 원래 plan 의 month 9-12 unifying TMLR position paper 의 모양이 분명함: "Calibrated wrappers for LLM/VLM uncertainty heuristics."

### 4.2 사전에 막아야 할 공통 reviewer 우려

| 우려 | 대응 |
|---|---|
| "Exchangeability 가 깨짐 (auto-regressive)" | Unit 을 거칠게 (step / decision / answer-span). Barber+ 2022 + weighted CP 인용 |
| "Wrapper paper — 기술적 contribution 이 뭐냐" | *구체적인* conformity score, *구체적인* calibration 데이터 구성, 경험적 ablation 으로 승부. 이론 novelty 를 oversell 하지 말 것 |
| "Calibration set 이 eval benchmark 와 leakage" | Plan §0.4 에 이미 다룸 — train split + 10% eval-split sanity check |
| "Distribution shift 에서 coverage 깨짐" | OOD 실험 실행 (GSM8K calibrate, MATH/AIME test — sec 1.3 §5). Weighted CP 인용. Graceful degradation 보일 것 |
| "Abstention rate 너무 높아서 시스템 쓸모 없음" | Pareto frontier (accuracy vs answer-rate) — 세 프로젝트 다 main figure |

### 4.3 이론적 leverage point

- **Sequence-level → step-level coverage propagation** (CoT-CP) — CRC nested-set construction (Angelopoulos+ 2022, Bates+ 2021 인용)
- **Group-conditional CP** (CoT-Tool multi-tool) — Conformal-RAG (Feng+ 2025), Cherian-Gibbs-Candes (NeurIPS 2024), Romano-Sesia-Candès 2020 (APS for groups) 인용
- **PMI as conformity score** (CG-VL) — "information-theoretic calibration" 으로 framing. Cover-Thomas, M3ID, C-PMI 인용

---

## 5. 12개월 계획에 대한 전략적 시사점

### 5.1 Sequencing — 원래 CG-VL → CoT-Tool → CoT-CP 순서 유지

문헌 스캔 결과 확인됨:

- **CG-VL 먼저** (months 1-3): 인프라 이미 갖춰짐 (lmms-eval 의 POPE/AMBER, Qwen2.5-VL). 벤치마크가 closed-form 이라 iteration 빠름. C-PMI + PAI + M3ID baseline 모두 open-source. "Wrapper" framing 으로 method-novelty bar 가 moderate 여도 3개월 안에 paper 가능.
- **CoT-Tool 두번째** (months 3-5): framing 더 날카롭게 필요 (§2.4 참조) 이지만 operationally 가장 유용한 결과물. Framing 어려움이 풀리지 않으면 defer 가능; CG-VL → CoT-CP 직행도 viable.
- **CoT-CP 세번째** (months 5-9): 가장 이론적으로 흥미롭고, 가장 깨끗한 빈 자리이지만, scope 도 가장 큼 (PRM800K calibration, 32B scaling, sequence-level CRC 이론). 가장 긴 runway 에 배치할 가치 있음.

### 5.2 Compute 예산 현실 점검

각 프로젝트의 비싼 항목:

- **CG-VL**: 2× inference cost (image + image-shuffle forward) × POPE/AMBER/HallusionBench × 4개 모델 (Qwen-VL, LLaVA-1.6, Idefics-3, Qwen-Omni). 모델-벤치마크 쌍 당 ~A100-week → 총 ~16 A100-weeks. **8-GPU 서버에서 tractable**.
- **CoT-Tool**: Calibration 데이터 생성 (Strategy A) 이 bottleneck — 모든 calibration query 에 LLM 2-forward (with-tool, without-tool). ~10K query × 2 forward × 7B = 다룰 만함. Multi-tool extension 은 linear scaling. **Tractable**.
- **CoT-CP**: 32B / 72B scaling 이 budget risk. AIME / GPQA 평가에 32B 모델로 $N=32$ self-consistency 는 비쌈. 첫 submission 은 7B-only 로 자르고, camera-ready / journal extension 에서 32B 추가 계획.

### 5.3 위험 조정 권장

세 프로젝트 중 하나 drop 해야 한다면: **CoT-Tool drop**. CoT-CP 가 가장 깨끗한 빈 자리 (heuristic step-confidence 가 활발한 전장인데 CP 진입자 없음). CG-VL 이 "ready to go" 에 가장 가까움 (인프라 갖춰짐, baseline open-source). CoT-Tool 의 framing 이 가장 다툼 많고 review 에서 가장 방어적인 글쓰기 필요. CG-VL 과 CoT-CP 둘 다 통과하면, CoT-Tool 은 NeurIPS-Agents workshop paper 또는 unifying TMLR 에 흡수 가능.

---

## 6. Bibliographic backbone (25-paper 단축본)

논문 쓰기 전 끝까지 읽을 것들.

### Foundational CP
1. Vovk-Gammerman-Shafer 2005 — *Algorithmic Learning in a Random World* (단행본)
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

## 7. 다음에 결정해야 할 열린 질문들

1. **CoT-Tool framing**: §2.4 의 3가지 sharpening 수 (multi-tool group-CP, cost as CRC loss, Strategy-A oracle) 가 충분히 느껴지는가, 아니면 이 프로젝트를 deprioritize 할 것인가?
2. **CG-VL claim positioning**: "first CP for open-ended VLM hallucination" (Inductive-CP-LVLM 이 closed-set 이므로 사실상 맞음) 으로 갈 것인가, "PMI calibration wrapper" (더 정직하지만 덜 sexy) 로 갈 것인가?
3. **CoT-CP scope**: v1 에 32B scaling 포함할 것인가 camera-ready 로 미룰 것인가? Compute realism 점검 필요.
4. **Differentiable Conformal Training (2026 preprint)**: 추적할 것. Concurrent 로 cite 할지 preempt 할지 결정.
5. **Unifying TMLR paper**: 지금 commit 할 것인가 (그래야 세 프로젝트 paper 가 foreshadow 가능) 아니면 셋 다 land 한 후?

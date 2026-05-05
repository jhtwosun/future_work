# Conformal Prediction for LLM Uncertainty & Hallucination

> **3 standalone research projects**, each with its own paper potential. They share a common theoretical backbone (split-conformal prediction / conformal risk control) and the same server infrastructure, but tackle different LLM failure modes.

| # | Project | Failure mode addressed | Modality | Target venue |
|---|---|---|---|---|
| **1** | **CoT-CP** — Conformal Chain-of-Thought | Multi-step reasoning errors that cascade | Text | NeurIPS / ICLR |
| **2** | **CoT-Tool** — Conformal Tool-Use Trigger | Agent calls tool too rarely / too often → hallucination or wasted compute | Text + Tools | ICLR / NeurIPS Agents |
| **3** | **CG-VL** — Conformal Grounding for Vision-Language Models | VLM ignores image and answers from LM prior → object hallucination | Vision-Language | EMNLP / ACL / NeurIPS |

(Audio LLM 도 가능하지만 vision 으로 정한 이유: POPE/HallusionBench/AMBER 같이 hallucination 측정이 표준화되어 있고, baselines (LLaVA, Qwen2.5-VL, Idefics) 풍부, lmms-eval 통합 완료 → fast iteration. 단, Section 3.6 에 audio variant 도 simultaneous extension 으로 명시.)

---

## 0. Common Background — Why CP, Why Now

### 0.1 Conformal prediction in 30 seconds (LLM 맥락)

Split conformal prediction = "label 있는 calibration data 로부터 conformity score 의 quantile $\hat{q}$ 을 뽑고, test 시점에 score 가 $\hat{q}$ 이상인 후보만 prediction set 에 포함" → set 이 정답을 포함할 확률 ≥ 1−α 를 distribution-free 로 보장 (단, calibration ⊥ test 가 exchangeable 일 때).

**LLM 에 적용할 때 문제 / 기회**:
- (문제) Token-level exchangeability 깨짐 (auto-regressive 의존성)
- (기회) Coarser unit (reasoning step / tool decision / answer span) 에서는 prompt-level exchangeability 가 성립
- (기회) **Conformal Risk Control** (Angelopoulos+ 2022) 은 generic loss function 으로 확장 → "hallucination rate ≤ α" 같은 user-facing risk 를 직접 제어 가능

### 0.2 Common server infrastructure (이미 있음)

| 자원 | 경로 | 비고 |
|---|---|---|
| Base LLM (text+vision+audio) | HF: `Qwen/Qwen2.5-Omni-7B` | 이미 사용 중 |
| VL only baseline | `Qwen/Qwen2.5-VL-7B-Instruct`, `liuhaotian/llava-v1.6-vicuna-7b` | HF 다운로드 |
| Eval framework | `/home/jongsuk-kim/lmms-eval/` | POPE, GSM8K, MMVet, AIR-Bench, MMAU 등 31+ task 통합 |
| Inference runtime | `/home/jongsuk-kim/dj_coat/eval_vllm/.venv/` | vLLM 0.19, torch 2.10+cu128 |
| GPU pool | 8× A100/H100 (가정) | 1-2 GPU 로 inference, 4 GPU 로 medium 학습 |
| Training runtime | `/home/jongsuk-kim/dj_coat/finetuning_js/.venv/` | torch 2.8+cu128, transformers 4.57, peft 0.18 |

### 0.3 Common file layout

각 프로젝트는 다음 구조로 정착:

```
/home/jongsuk-kim/dj_coat/research_proposals/
  conformal_llm_uncertainty.md          ← this file
  cot_cp/                               ← Project 1
    src/
      score.py                          # conformity scoring
      calibration.py                    # quantile estimation
      decode.py                         # CP-aware decoding loop
    configs/
    data/
      calibration_gsm8k.jsonl
      calibration_math.jsonl
    results/
  cot_tool/                             ← Project 2
    src/
      tool_router.py
      tools/                            # calculator, retriever, code-exec
      score.py
    ...
  cg_vl/                                ← Project 3
    src/
      grounding_score.py                # PMI / counterfactual scoring
      cp_filter.py
    ...
```

### 0.4 Decision on calibration data construction (모든 프로젝트 공통)

CP 의 calibration set 은 ① label 이 있고 ② test distribution 과 exchangeable 해야 함. Three sources of (input, ground truth) pairs:

| Source | 장점 | 단점 |
|---|---|---|
| **Public benchmark train split** (GSM8K-train, POPE-train, AIR-Bench-train) | Label 정확, 쉬움 | benchmark hardcode → eval set leakage 위험 |
| **In-distribution synthetic** (LLM 이 생성한 query + verified label) | 무한히 확장 | label noise |
| **Held-out from same eval** (eval set 의 일부를 calibration 으로) | 분포 일치 보장 | eval set 줄어듦, fair comparison 깨짐 |

**채택 방침**: train split 사용을 default, eval split 의 10% 를 calibration 으로 떼는 setting 도 sanity check 로 같이 보고 (논문에서 두 결과 모두 보여주면 robust 함을 입증).

---

# 📘 Project 1 — Conformal Chain-of-Thought (CoT-CP)

> "Reasoning step 마다 calibrated confidence 를 계산해서, 위험한 step 에서만 추가 compute 를 쓴다. 최종 답이 정답일 확률 ≥ 1−α 를 통계적으로 보장."

## 1.1 Problem

Long-CoT 모델 (GPT-o1, DeepSeek-R1, Qwen-QwQ) 은 reasoning step 이 길어질수록:
- Step 하나의 작은 오류가 누적 → 최종 답 망가짐 (error cascade)
- 일부 step 만 어렵고 나머지는 trivial → 모든 step 에 동일한 compute 쓰는 건 낭비
- 자기검증 (self-verify) 은 휴리스틱 → 통계적 보장 없음

기존 시도:
- **Self-consistency** (Wang+ 2022): N sample → majority vote. 모든 query 에 fixed N. 보장 없음.
- **Process Reward Model** (Lightman+ 2023, Math-Shepherd): step 별 reward. Calibration 안 됨.
- **Beam search + verifier**: 휴리스틱 cut-off.

## 1.2 Core idea

매 reasoning step $t$ 에 대해 **conformity score** $s_t$ 를 계산.
$s_t$ 가 calibration quantile $\hat{q}$ 미만이면 → "위험 step" 으로 marking, branch 또는 verifier 호출.

**Risk decomposition** (Conformal Risk Control):
$$\mathbb{E}[\text{wrong final answer}] \le \alpha$$
를 step 별 risk $\alpha_t$ 의 합 또는 합성으로 분배.

### 1.2.1 Three score function candidates (ablate)

| Score | 정의 | 장단점 |
|---|---|---|
| **S-SC** (self-consistency at step $t$) | step $t$ 까지 prefix 고정 후 future 를 $N{=}8$ 번 sample, 최종 답 분포의 entropy or top-1 비율 | 강함, 비쌈 ($N\times$ extra compute) |
| **S-PRM** (process reward) | 외부 PRM (예: Math-Shepherd open-weight) 가 step 에 점수 매김 | 빠름, PRM quality 의존 |
| **S-LP** (log-probability based) | step 의 평균 log-prob, 또는 step 마지막 token 의 entropy | 거의 무료, calibration 없으면 쓸모 없지만 CP 가 calibrate 해줌 |

### 1.2.2 Decision rule at step $t$

```
if s_t >= q_hat:                       # confident → continue
    keep current step, advance to t+1
elif s_t < q_hat and budget > 0:       # uncertain → invest compute
    options:
      (a) Branch: sample K alternative continuations of step t, pick max-score
      (b) Verifier: call symbolic checker (sympy for math, type-check for code)
      (c) Backtrack: rollback to step t-1, regenerate with higher temp
    budget -= 1
else:                                  # budget exhausted
    abstain or output current path with explicit warning
```

### 1.2.3 Calibration

$$\hat{q} = \text{Quantile}_\alpha\big(\{s(\text{step}_i)\}_{i=1}^n\big)$$

where $\{(\text{step}_i, \mathbb{1}[\text{step}_i\ \text{correct}])\}$ comes from a labeled CoT dataset. **PRM800K** (OpenAI, Math) 은 step-correctness label 을 제공 → 거의 그대로 사용 가능. GSM8K 는 step label 없으므로 (final answer correctness) → (step correctness) 를 weak supervision 으로 보정.

### 1.2.4 Sequence-level guarantee

Naive step-level Bonferroni: $\alpha_{\text{seq}} = T \cdot \alpha_{\text{step}}$ (overconservative).
**Better**: Conformal Risk Control 의 nested-set construction — sequence-level loss $L(\text{trajectory}, \text{answer})$ 에 직접 $\hat{q}$ 잡음. CoT 전체를 하나의 random object 로 보고 한 번 calibrate.

## 1.3 Experimental Setup

### Models (small → large 순으로)
- **Qwen2.5-7B-Instruct** (text-only, fast iteration)
- **Qwen2.5-Math-7B-Instruct** (math 특화)
- **DeepSeek-R1-Distill-Qwen-7B** (long-CoT 학습된 distilled 모델 — 우리가 실제로 타겟하는 use case)
- (옵션) Qwen2.5-72B 또는 R1-32B 에서 scaling 확인

### Datasets

| Dataset | 역할 | 크기 | 경로 / 출처 |
|---|---|---|---|
| **GSM8K** | calibration + eval | train 7.5k / test 1.3k | lmms-eval `tasks/gsm8k`, HF `gsm8k` |
| **MATH-500** | eval (harder) | 500 | HF `HuggingFaceH4/MATH-500` |
| **PRM800K** | step-level calibration source | ~800k step labels | HF `Birchlabs/openai-prm800k-stepwise-critic` |
| **AIME-2024** | eval (extreme difficulty) | 30 | HF |
| **GPQA-Diamond** | eval (science, OOD from math) | 198 | HF `Idavidrein/gpqa` |
| **TruthfulQA-MC** (옵션) | eval (factual reasoning, not math) | 817 | HF |

### Baselines

1. Greedy / chain-of-thought (no CP)
2. Self-consistency $N=8, 16, 32$
3. Best-of-N with PRM (Math-Shepherd)
4. **Adaptive computation** (DiVeRSe; Stepwise self-consistency) — 휴리스틱 step-level adaptive
5. Our **CoT-CP** (S-SC / S-PRM / S-LP variants)

### Metrics

- **Accuracy** (final answer)
- **Compute budget** (tokens generated, or wallclock)
- **Coverage** (empirical $P[\text{correct}\,|\,\text{not abstained}]$)
- **Selective accuracy curves** — accuracy vs coverage trade-off (이게 main figure)
- **Step-level calibration ECE** (sanity check: $s_t$ 가 정말 step correctness 와 calibrated 한지)

### Headline experiments to run

1. **Pareto frontier**: x-axis = avg tokens/query, y-axis = accuracy. CoT-CP vs baselines 곡선 비교 (Fig 1)
2. **Coverage guarantee verification**: target $1-\alpha \in \{0.8, 0.9, 0.95\}$ 에 대해 empirical coverage 실측 → 보장이 깨지지 않는지 (Table 1)
3. **Score function ablation**: S-SC vs S-PRM vs S-LP (Fig 2)
4. **Branching ratio**: $\alpha$ 변화 → branch 호출 비율 변화 (Fig 3)
5. **OOD calibration drift**: GSM8K 로 calibrate → MATH/AIME/GPQA 에 적용 → coverage 어디서 깨지나 (Table 2)
6. **Scaling**: model size 7B → 32B 에서 효과 유지되는지

## 1.4 Implementation plan (8-10주)

| Week | Deliverable |
|---|---|
| 1 | Infra: vLLM + step-by-step generation harness (force `\n\n` 마다 stop, score, resume) |
| 2 | Score function S-LP, S-SC 구현 + GSM8K-train 으로 sanity check |
| 3 | PRM800K 로드 + step-level calibration pipeline |
| 4 | CP filter + branching loop, end-to-end 첫 run on GSM8K |
| 5 | S-PRM (Math-Shepherd) integration, ablation study |
| 6 | MATH/AIME/GPQA 확장, OOD analysis |
| 7 | Conformal Risk Control framework integration (sequence-level guarantee) |
| 8 | Baseline reruns (self-consistency, BoN+PRM), Pareto figure |
| 9 | Scaling to 32B, robustness check |
| 10 | Paper writing |

## 1.5 Risk register

| Risk | 영향 | 완화 |
|---|---|---|
| Score 가 step correctness 와 calibrated 안 됨 | 보장 깨짐 | 3가지 score 후보 ablate, 가장 잘 calibrate 되는 것 채택. Calibration plot (reliability diagram) 으로 진단 |
| PRM800K 의 step boundary 가 우리 모델 출력과 mismatch | calibration noise | 모델별 step segmenter 학습 (regex `\n\n` + LLM-based) |
| Branching cost 가 self-consistency 와 비슷 → speedup 없음 | story 약화 | budget-controlled experiment 로 "동일 compute 에서 정확도↑" 명시. 그리고 selective answering (abstain) 으로 차별화 |
| Long-CoT 모델 (R1-distill) 의 step 이 너무 짧고 많음 | step 단위 noise | step grouping (5-step 단위로 묶음) 또는 paragraph-level |

## 1.6 Why this is publishable

- **First step-level CRC**: 기존 CP-for-LLM 은 sequence (Quach 2024) / claim (Mohri 2024) — step 은 unique
- **Practical**: test-time scaling 시대에 budget allocation 문제 hot
- **Theoretical contribution**: step-level → sequence-level coverage propagation (Bonferroni vs CRC nested set)
- **Reproducible**: 모든 데이터 / 모델 open weight

---

# 📗 Project 2 — Conformal Tool-Use Trigger (CoT-Tool)

> "Agent / tool-use LLM 이 언제 도구를 부를지 결정하는 것을 calibrated decision 으로. tool 안 부른 경우의 hallucination 확률을 통계적으로 boundary."

## 2.1 Problem

Tool-augmented LLM (ReAct, Toolformer, Function-calling) 의 핵심 약점:

- **Over-calling**: trivial 한 query 에도 무조건 search 호출 → latency↑, cost↑
- **Under-calling**: 모를 때도 자신 있게 hallucinate → factual 오류
- **No guarantee**: 라우팅이 휴리스틱 (LLM self-report or fixed rule)

기존 시도:
- **Toolformer** (Schick+ 2023): self-supervised tool insertion. 학습 비용 큼. Calibration 없음.
- **ReAct prompting** (Yao+ 2022): LLM 이 reasoning 으로 결정. 의사결정이 distribution-free guarantee 없음.
- **Confidence routing** (e.g., Adaptive-RAG): self-rated score 의 휴리스틱 threshold.

## 2.2 Core idea

각 query 에 대해 **"tool 안 부르면 hallucination 할 확률"** 의 conformity score 계산. Score 가 cut-off 미만이면 tool 호출.

**Calibrated guarantee**: 
$$P[\text{LLM answer wrong} \mid \text{tool not called}] \le \alpha$$

즉 "도구 안 부른 모든 query 의 평균 정확도가 1−α 이상" 을 distribution-free 로 보장.

### 2.2.1 Score function candidates

| Score | 정의 | 비용 |
|---|---|---|
| **S-SC** | $N{=}8$ sample 한 답들의 일치도 (top-1 비율) | $N\times$ |
| **S-EXP** | 모델이 명시적 prompt ("Are you confident? Yes/No") 에 대한 logit 확률 | $1\times$ |
| **S-EMB** | query 의 embedding 과 calibration set query embedding 의 거리 (semantic familiarity) | 거의 무료 |
| **S-FT** | 별도로 학습된 small classifier (frozen LLM hidden state → "needs tool" probability) | training 필요 |

### 2.2.2 Decision rule

```
score = scoring_fn(query)
if score >= q_hat:
    answer = LLM(query)                # confident, skip tool
else:
    tool_result = call_tool(query)     # uncertain, invoke tool
    answer = LLM(query, tool_result)
```

### 2.2.3 Calibration data construction

이게 가장 까다로움. 필요한 것: $(q_i, \text{tool\_needed}_i)$ pair.

**Strategy A — Oracle from answer correctness**:
```
for each query q in calibration set:
    a_no_tool = LLM(q)
    a_tool    = LLM(q, tool(q))
    correct_no_tool = is_correct(a_no_tool, gold)
    label = 1 if (not correct_no_tool and correct_tool) else 0
```
즉 "tool 없으면 틀리고 tool 쓰면 맞는" 케이스만 "tool 필요" 로 라벨링.

**Strategy B — Synthetic from heuristics**:
- 산술이 들어간 query → calculator 필요
- Named entity / 최신 정보 → search 필요
- (regex / NER 로 자동 라벨링)

**Strategy C — Self-supervised**:
- 같은 query 에 대해 answer 의 self-consistency 가 낮으면 → tool 필요로 라벨링

**채택 방침**: Strategy A 가 ground truth 에 가장 가까움. B 는 noise 가 많지만 무한 확장 가능. 두 데이터를 mix 해서 calibration.

### 2.2.4 Multi-tool extension

Tool 이 여럿이면 (calculator, search, code-exec, ...):
- Per-tool conformal score $s^{(\text{tool})}(q)$ → tool 별 cut-off
- 또는 hierarchical: 먼저 "tool 필요 여부" → 다음에 "어떤 tool"
- **Conformal Risk Control multi-output variant** 적용

## 2.3 Experimental Setup

### Models
- **Qwen2.5-7B-Instruct** (function-calling capable)
- **Qwen2.5-72B-Instruct** (옵션, scaling)
- **GPT-4o-mini** (옵션, API baseline)

### Tools
- **Calculator**: sympy
- **Search**: HF `sentence-transformers/all-MiniLM-L6-v2` + Wikipedia dump
- **Code-exec**: subprocess sandbox (timeout-protected)
- **Knowledge base lookup**: simple SQLite over Wikidata (subset)

### Datasets

| Dataset | 도구 종류 | 역할 |
|---|---|---|
| **GSM8K** | calculator | calc 필요 ↔ 비필요 mix |
| **HotpotQA** | search (multi-hop) | retrieval 필요 |
| **TriviaQA** | search (single-hop) | retrieval 필요 |
| **MMLU** | mixed | 일부는 parametric knowledge 로 충분, 일부는 search |
| **GPQA** | search + code | hard, OOD |
| **NQ-Open** | search | open-domain QA |
| **API-Bank / ToolBench (subset)** | multi-tool | function calling complex case |

calibration 은 train splits, eval 은 test splits.

### Baselines

1. **Always-tool**: 모든 query 에 도구 호출 (upper bound on tool cost, lower bound on hallucination)
2. **Never-tool**: 모델 parametric knowledge 만 (lower cost, upper hallucination)
3. **ReAct prompting**: LLM 이 도구 호출 결정
4. **Adaptive-RAG** (Jeong+ 2024): retrieval routing
5. **Self-RAG** (Asai+ 2024): self-reflect token
6. **Our CoT-Tool** (S-SC / S-EXP / S-EMB / S-FT)

### Metrics

- **Accuracy** (final answer)
- **Tool call rate** (= 비용 proxy)
- **Hallucination rate among non-tool-called queries**
- **Pareto curve** (accuracy vs tool calls)
- **Coverage guarantee verification** (target α 달성 여부)
- **Latency** (real-world cost)

### Headline experiments

1. **Pareto frontier** (Fig 1): x = tool call rate, y = accuracy. CP 곡선이 baselines 를 dominate
2. **Hallucination guarantee** (Table 1): "tool 안 부른 케이스" 의 정확도가 target $1-\alpha$ 이상인지
3. **Score ablation**: S-SC (강하나 비쌈) vs S-EMB (싸나 약함) trade-off
4. **OOD shift**: GSM8K 로 calibrate → MMLU 적용. coverage 깨지면 weighted CP 로 보정
5. **Multi-tool routing**: 어떤 tool 을 부를지의 calibrated 결정
6. **Latency analysis**: 실제 wallclock 으로 cost 측정

## 2.4 Implementation plan (10-12주)

| Week | Deliverable |
|---|---|
| 1 | Tool sandbox (calculator, search index, code-exec) |
| 2 | LLM agent harness (function-calling format) |
| 3 | Calibration data 생성 (Strategy A on GSM8K + HotpotQA) |
| 4 | S-EXP, S-SC 구현 + first end-to-end CP 적용 |
| 5 | Cut-off 결정 + coverage 실측, 첫 Pareto 곡선 |
| 6 | Baselines (ReAct, Adaptive-RAG, Self-RAG) 재현 |
| 7 | S-EMB / S-FT 추가 ablation |
| 8 | Multi-tool extension (search + calc + code) |
| 9 | OOD evaluation + weighted-CP 변형 |
| 10 | Latency / cost 실측 |
| 11 | Scaling to 72B, 추가 dataset |
| 12 | Paper writing |

## 2.5 Risk register

| Risk | 영향 | 완화 |
|---|---|---|
| Strategy A 의 oracle label 이 noisy (LLM 의 답 정확도 측정 자체가 어려움) | calibration 불안 | exact-match + LLM judge ensemble, agreement 만 사용 |
| Tool 자체가 noisy (search 가 wrong page) | LLM-with-tool 도 틀림 → label 모호 | tool 결과의 quality 도 score 에 반영 (compounded CP) |
| Multi-tool 의 calibration 데이터 부족 | multi-tool 결과 약함 | single-tool 먼저 main paper, multi-tool 은 extension section |
| Distribution shift (calibration GSM8K, eval MMLU) | coverage 깨짐 | weighted CP (Tibshirani 2019) 로 보정, 또는 per-domain calibration |

## 2.6 Why this is publishable

- **Practical**: agent 시대에 tool routing 비용 / hallucination trade-off 가 hot
- **First statistical guarantee**: 기존 routing 방식들은 모두 휴리스틱
- **Multi-tool extension**: hierarchical CP 라는 method-side novelty
- **Reproducible**: 도구가 모두 open / cheap

---

# 📕 Project 3 — Conformal Grounding for Vision-Language Models (CG-VL)

> "VLM hallucination 의 핵심 = 이미지를 무시하고 LM prior 로 답함. 이를 'image 가 답을 얼마나 바꾸는가' 라는 conformal score 로 측정해서, grounding 부족한 query 에는 abstain 또는 re-prompt."

## 3.1 Problem

VLM (LLaVA, Qwen2.5-VL, Idefics) 의 가장 흔한 hallucination 패턴:

- **Object hallucination**: 이미지에 없는 물체를 답에 등장 ("There's a dog" — 실제론 고양이만)
- **Attribute hallucination**: 색상 / 갯수 / 위치 등 시각적 속성을 LM prior 로 추측
- **Relation hallucination**: 두 객체의 공간 관계를 학습 데이터의 빈도로 답변

기존 시도:
- **POPE-style binary probing**: hallucination detection benchmark (Li+ 2023). 측정만, fix 안 함
- **VCD** (Leng+ 2024): contrastive decoding으로 LM bias 빼기. 휴리스틱
- **OPERA** (Huang+ 2024): attention pattern intervention. 휴리스틱
- **VOLCANO** (Lee+ 2024): self-feedback iteration. 보장 없음

→ **None provide statistical guarantee on hallucination rate.**

## 3.2 Core idea

이미지 $I$ 와 question $q$ 에 대해, VLM 의 답이 정말 image-grounded 인지 측정:

$$\text{ground}(I, q, a) = \log p(a \mid I, q) - \log p(a \mid \emptyset, q)$$

즉 "이미지를 봤을 때 vs 안 봤을 때 답의 likelihood 변화" = **mutual information 계열 score**.

이를 conformity score 로 사용. Calibration 으로 cut-off 정함.

**보장**:
$$P[\text{answer is hallucination} \mid \text{ground} \ge \hat{q}] \le \alpha$$

즉 "충분히 grounded 라고 판정한 답은 1−α 확률로 정확함."

### 3.2.1 Image masking strategies (counterfactual 의 정의)

"이미지 안 보고 답한다" 의 구현 방법:

| Strategy | 구현 | 장단점 |
|---|---|---|
| **Hard mask** | image input 에 zeros / black image | 가장 깔끔, 하지만 모델이 OOD 처리 |
| **Noise** | Gaussian noise image | OOD 덜함 |
| **Shuffle** | 다른 batch 의 random image | 정보량 0, 자연스러움 |
| **Patch** | image 의 핵심 patch 만 가림 | object-level grounding 측정 |

채택: **Shuffle** 을 default (가장 자연스럽고 OOD 적음).

### 3.2.2 Score function 변형

| Variant | 식 | 측정 |
|---|---|---|
| **CG-PMI** | $\log p(a \mid I,q) - \log p(a \mid I_{\text{shuffle}},q)$ | answer 단위 |
| **CG-Token-PMI** | token 별 PMI 의 mean / min | token 단위 정밀 |
| **CG-Self-Consistency** | image 와 image-masked 의 답이 같은지 (boolean) | 가장 단순 |
| **CG-Visual-Attn** | attention 의 image patch 비율 | 모델 internal 필요 |

### 3.2.3 Decision rule

```
score = grounding_score(image, question, candidate_answer)
if score >= q_hat:
    output = candidate_answer        # well-grounded
else:
    output = "I'm not sure" 
    # or re-prompt with explicit visual focus
    # or call vision verifier (e.g., separate object detector)
```

### 3.2.4 Multi-answer prediction set variant

Single candidate 만 평가하지 말고, **N answers sample** 후 score 가 cut-off 이상인 answer 들의 set 을 prediction set 으로 출력. 이 set 이 정답을 포함할 확률 ≥ 1−α (Quach 2024 style 이지만 grounding score 가 conformity 라는 점이 새로움).

## 3.3 Experimental Setup

### Models
- **Qwen2.5-VL-7B-Instruct** (main, strong baseline)
- **LLaVA-1.6-Vicuna-7B / 13B** (broad comparison)
- **Idefics-3-8B** (다른 family)
- **Qwen2.5-Omni-7B** (audio+vision multi-modal extension 의 prep)

### Datasets

| Dataset | 측정 대상 | 크기 |
|---|---|---|
| **POPE** | object hallucination (binary MCQ) | 9k (random/popular/adversarial) |
| **HallusionBench** | reasoning + visual hallucination | 1k |
| **AMBER** | object/attribute/relation 종합 | 15k |
| **MMHAL-Bench** | multi-aspect hallucination | 96 |
| **MMVet** | open-ended generation, GPT judge | 218 |
| **MMBench** | broad capability, sanity check (regression) | 4k |
| **CHAIR (custom on COCO captions)** | open-ended caption hallucination | 5k |

calibration: POPE-train, AMBER-train (둘 다 별도 train split 있음)
eval: 위 dataset 의 test split

### Baselines

1. **Vanilla VLM** (no intervention)
2. **VCD** (Leng+ 2024) — contrastive decoding
3. **OPERA** (Huang+ 2024) — attention intervention
4. **Greedy + temperature 0** vs **sampling + self-consistency**
5. **POPE-classifier baseline** (별도 학습된 hallucination detector)
6. **Our CG-VL** (CG-PMI / CG-Token-PMI / CG-SC / CG-Attn)

### Metrics

- **Hallucination rate** (POPE accuracy, AMBER F1, CHAIR i/s)
- **Coverage** (verified guarantee)
- **Selective accuracy curve** (accuracy vs answer rate, abstain trade-off)
- **Open-ended quality** (MMVet GPT-4 score on non-abstained answers)
- **Score correlation with ground-truth hallucination** (AUROC, AUPRC)
- **Inference overhead** (PMI 계산이 image-shuffle forward 1번 추가 → 2× cost. 이 비용을 명시)

### Headline experiments

1. **Hallucination reduction with guarantee** (Fig 1): POPE/AMBER 에서 vanilla vs CG-VL — coverage curve
2. **Open-ended grounding** (Fig 2): MMVet / CHAIR 에서 CG-VL 이 baseline 대비 hallucination 줄임을 GPT judge 로 검증
3. **Score function ablation** (Fig 3)
4. **Cross-model generalization** (Table 1): LLaVA 로 calibrate → Qwen-VL 에 적용. coverage 유지?
5. **Image masking strategy ablation** (shuffle vs zero vs patch, Table 2)
6. **Compute overhead trade-off** (Fig 4): 2× cost 의 가치
7. **Sanity check on MMBench**: hallucination 줄였다고 일반 능력 떨어지지 않음을 확인

## 3.4 Implementation plan (8-10주)

| Week | Deliverable |
|---|---|
| 1 | lmms-eval 의 POPE / AMBER 재현, 기존 hallucination rate 측정 |
| 2 | PMI score 구현 (image shuffle + dual forward) on Qwen2.5-VL |
| 3 | Calibration on POPE-train, 첫 cut-off, end-to-end CP filter |
| 4 | AMBER, HallusionBench, MMHAL 추가, coverage 실측 |
| 5 | Open-ended (MMVet, CHAIR-COCO) 확장 |
| 6 | LLaVA / Idefics 추가, cross-model 일반화 |
| 7 | Baselines (VCD, OPERA) 재현, 비교 |
| 8 | Score ablation (token-PMI, SC, attention) |
| 9 | Audio extension on AIR-Bench (시간되면, 부록) |
| 10 | Paper writing |

## 3.5 Risk register

| Risk | 영향 | 완화 |
|---|---|---|
| Image shuffle 이 informative 일 수 있음 (다른 이미지가 우연히 비슷) | PMI score noise | shuffle 시 dataset-wide random + 평균 (multiple shuffles) |
| Open-ended 답에서 PMI 계산이 ill-defined (long answer 의 likelihood) | score 신뢰도 | length-normalized log-prob, 또는 first-N-tokens 만 |
| CP coverage 가 hallucination 정의에 의존 (어떤 답이 hallu 인지) | label noise | POPE 같이 명확한 binary 부터 시작, AMBER 의 fine-grained label 활용 |
| Calibration distribution (POPE) 와 eval distribution (HallusionBench) shift | coverage 깨짐 | per-domain calibration, 또는 weighted CP |
| 2× inference cost | practical 부담 | parallel 처리, 또는 CG-Attn 같은 cheap variant ablation |

## 3.6 Audio extension (optional follow-up)

VLM 의 image → Audio LM 의 audio 로 generalize:
- AIR-Bench / MMAU 에서 audio hallucination 측정
- Score: $\log p(a \mid \text{audio}, q) - \log p(a \mid \text{shuffled audio}, q)$
- Server 에 이미 Qwen2.5-Omni 와 audio benchmark 인프라 있음 → 1-2주 추가 작업

CoAT 와 분리하기 위해: CoAT 의 multi-task SFT 모델은 안 쓰고 base Qwen2.5-Omni-7B 만 사용. CoAT 의 acoustic head 도 안 씀. 순수 grounding score 로만.

## 3.7 Why this is publishable

- **First CP framework for VLM hallucination**: hot topic, no prior work
- **PMI 라는 모델 internal 의 statistical 사용** — interpretable
- **Strong empirical**: POPE/AMBER/HallusionBench 에서 SOTA 가능
- **Theoretical**: PMI ↔ mutual information ↔ conformal score 의 연결
- **Generalizable**: vision → audio → multi-modal 확장 path 명확

---

# 🏗 Common Decisions Across All Three Projects

## A. Conformal framework choice

세 프로젝트 모두 **Split Conformal Prediction** + **Conformal Risk Control** (Angelopoulos+ 2022) 위에 구축.
- Split CP: 빠르고 distribution-free
- CRC: hallucination rate / accuracy 같은 generic loss 에 직접 calibration

Beyond-exchangeability (Barber+ 2022): distribution shift 큰 경우의 fallback.

## B. Score normalization

Score 가 다른 모델 / 다른 데이터셋 간에 비교 가능하도록:
- Per-model calibration (모델 마다 cut-off 따로)
- 필요시 z-score normalization within calibration set

## C. Reproducibility checklist

- [ ] 모든 calibration data 의 hash + 생성 script 저장
- [ ] cut-off 와 그 confidence interval 명시
- [ ] coverage 가 target $1-\alpha$ 내에 있다는 empirical verification
- [ ] 100+ random splits 으로 stability 측정
- [ ] code release (GitHub) + 모델 weight 안 건드리므로 base model HF link 만 명시

## D. Shared codebase

```python
# /home/jongsuk-kim/dj_coat/research_proposals/cp_core/
class ConformalScorer:
    def __init__(self, score_fn, alpha=0.1): ...
    def calibrate(self, calibration_data): 
        scores = [self.score_fn(x, y) for x, y in calibration_data]
        self.q_hat = np.quantile(scores, alpha, method='lower')
    def is_confident(self, x, y_candidate):
        return self.score_fn(x, y_candidate) >= self.q_hat
    def coverage(self, eval_data):
        # empirical coverage check
        ...
```

세 프로젝트가 이 base class 를 공유하고 score_fn / decision policy 만 다르게.

## E. Phased rollout (12-month plan with 3 papers)

| Month | Project | Milestone |
|---|---|---|
| 1-3 | **CG-VL** (가장 ready, infra 다 있음) | First submission (EMNLP / NeurIPS deadline) |
| 3-5 | **CoT-Tool** (medium complexity) | First submission (ICLR) |
| 5-9 | **CoT-CP** (가장 복잡, scaling 필요) | First submission (NeurIPS) |
| 9-12 | Revisions, multi-modal CP unifying paper | Survey / position paper at venue like TMLR |

Sequential 추천 이유:
- CG-VL 결과가 가장 빨리 나옴 (POPE 같은 closed benchmark 라 측정 빠름) → 빠른 publication
- CoT-CP 는 scaling 실험 필요해서 시간 길게 잡음
- 세 프로젝트가 서로 강하게 연결 → 결국 한 줄기 PhD thesis 의 chapter 들

## F. Out of scope (이 제안서에서 명시적으로 안 함)

- Conformal calibration during pretraining/RLHF (training-time CP)
- Differential privacy + CP
- Adversarial robustness of CP under prompt injection
- Watermarking via CP
- Online CP for streaming agent (deployment 가까운 problem, 다른 paper 로)

---

# 📚 Key References

| Topic | Paper |
|---|---|
| Conformal Prediction foundation | Vovk, Gammerman, Shafer 2005 |
| APS score | Romano, Sesia, Candès 2020 |
| RAPS score | Angelopoulos, Bates, Jordan, Malik 2021 |
| Conformal Risk Control | Angelopoulos+ 2022 |
| Beyond Exchangeability | Barber, Candès, Ramdas, Tibshirani 2022 |
| Weighted CP | Tibshirani+ 2019 |
| Conformal Language Modeling (sequence) | Quach, Fisch+ 2024 (ICLR) |
| Conformal Factuality (claim) | Mohri, Hashimoto 2024 (ICML) |
| Self-Consistency | Wang+ 2022 |
| Process Reward Models | Lightman+ 2023 (PRM800K) |
| ReAct | Yao+ 2022 |
| Toolformer | Schick+ 2023 |
| POPE | Li+ 2023 |
| HallusionBench | Guan+ 2024 |
| AMBER | Wang+ 2023 |
| VCD | Leng+ 2024 |
| OPERA | Huang+ 2024 |
| Qwen2.5-VL | Bai+ 2025 |
| LLaVA-1.6 | Liu+ 2024 |

---

# Decision Points the Author Should Resolve Next

1. **Sequencing**: 위에서 CG-VL → CoT-Tool → CoT-CP 순 권장. 이대로 갈지, 아니면 가장 흥미 있는 것 하나에 올인할지.
2. **Score ablation 우선순위**: 세 후보 다 돌릴지, 가장 가능성 큰 하나만 깊게 갈지.
3. **Open-source plan**: 코드 + calibration data 를 어느 라이센스로 release 할지.
4. **Collaborator**: 세 프로젝트 다 혼자 하기엔 양 많음. CoT-CP 는 reasoning 전공자, CG-VL 은 vision 전공자와 협업 권장.
5. **Compute budget 검토**: 8-GPU server 로 12개월 안에 세 paper 돌릴 수 있는지 계산.



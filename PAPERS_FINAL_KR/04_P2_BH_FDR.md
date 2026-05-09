# P2 — BH-FDR Per-Step CP: PRDS 가정의 구조적 깨짐

## 0. 배경 지식 (먼저 읽기)

### 다중검정 (Multiple Testing) 문제

여러 가설을 동시에 검정하면 **false positive 누적** 문제 발생. 60개 가설을 각각 α=0.05로 검정하면 적어도 1개 false positive 확률 $1 - 0.95^{60} = 95\%$.

### 다중검정 보정 방법

**Bonferroni (1936)**: $\alpha/T$ — 가장 보수적
- 60개 가설 시 0.10/60 = 0.00167 — 거의 모든 검정이 거부됨

**Benjamini-Hochberg BH (1995)**: false discovery rate (FDR) 통제
- "거짓 발견 비율 ≤ α"로 더 느슨한 기준
- 가설들을 p-value로 정렬해서 step-up 방식
- **Bonferroni보다 훨씬 powerful**

**Benjamini-Yekutieli BY (2001)**: $\alpha/(T \cdot H_T)$ — arbitrary dependence에서도 valid
- $H_T = \sum_{i=1}^T 1/i \approx \ln T$
- T=60에서 $H_{60} \approx 4.68$ → BY는 Bonferroni 대비 4.7× 우월

### CoT의 Per-Step Conformal Prediction

LLM CoT가 60-step. **각 step마다** conformal test 적용 (Approach B). 모든 step이 통과해야 trace 통과.

→ **다중검정 문제**: 60개 검정을 어떻게 보정?

### 왜 BH가 매력적인가

기존 CP 라이브러리들이 **Bates et al. 2023 (Annals of Statistics)** 결과 차용:

> "Conformal p-values는 PRDS (Positive Regression Dependence on Subset) 가정 만족. 따라서 BH에 직접 적용 가능."

**의미**: BH의 powerful FDR 통제를 CP에 그대로 사용 가능 → BY의 $H_T$ penalty 안 내도 됨.

## 1. 이 논문이 푸는 문제

**Bates 2023의 PRDS-on-CP 가정이 autoregressive CoT trace에서도 성립할까?**

만약 성립하면 → BH 직접 적용 → +1~3pp kept-frac 개선 예상.

만약 깨지면 → CP 학계가 잘못된 가정을 차용해온 것 → 새 procedure 필요.

## 2. 원래 가설 — Falsified

### 가설

PRDS holds for per-step CP family on autoregressive CoT trace → BH apply 가능.

### 검증 (15 cells × full n)

**3-method PRDS test**:
1. Spearman rank correlation
2. Conditional CDF monotonicity test
3. Permutation null calibration

**결과**:

| 데이터 | PRDS 결과 |
|---|---|
| 12 short-CoT cells (4 model × 3 dataset, full n) | **12/12 FAILS** |
| 3 long-CoT cells (QwQ, R1-Distill-32B, R1-Distill-Llama-70B) | **3/3 FAILS** |
| **합계** | **15/15 FAILS** |

→ **PRDS가 autoregressive CoT trace에서 systematic하게 깨진다**.

## 3. Theorem 4-ter (PRDS Counterexample) — 정식 증명

### T=2 Minimal Counterexample

**Setup**:
- 2-step trace
- Score family: cumulative running-min (per-step CP에서 표준)
- Autoregressive correlation 모델링

**계산**:
$$P(p_2 \leq 0.5 \mid p_1 \leq q_{25}) = 1.000$$
$$P(p_2 \leq 0.5 \mid p_1 \leq q_{75}) = 0.639$$

**해석**: $p_1$이 작을수록 (q_25 = 25%-quantile 미만) $p_2$가 작을 확률이 더 높음. 이는 **decreasing in conditioning threshold** — PRDS가 요구하는 nondecreasing 정확히 반대.

### 메커니즘 통찰: Quality Funnel

> Cumulative score aggregation (running min/max)이 **"quality funnel"**을 만든다. 한 번 score 떨어진 trace는 회복 안 됨 (CoT의 본질적 cascade 구조). 이는 PRDS의 monotonicity 가정과 정반대.

→ 새 발견: **CoT의 cascade 구조 = PRDS 위반의 메커니즘**.

## 4. 왜 "PRDS 가정 깨짐 자체가 contribution"인가

### Stein's Paradox 비유

1956년 Stein이 "sample mean은 d≥3에서 inadmissible"을 증명. Negative result처럼 보이지만 **shrinkage estimator** 분야 전체를 열었음.

우리도 비슷:
- "Bates 2023의 PRDS-on-CP는 autoregressive setting에서 fail"
- **e-BH dominance** (Theorem D)로 대체 procedure 정당화
- **Mondrian-BH** 등 새 procedure 정당화

### 4가지 contribution

1. **Literature correction**: CP+FDR 응용 paper들이 PRDS를 자명하게 차용 → 우리가 boundary 정확히 그음
2. **Procedure choice principled**: "Mondrian-BH 쓰자"가 ad-hoc 추천 → "PRDS 깨졌으니 BY/Mondrian/e-BH 중 하나 필수"
3. **Mechanism insight**: CoT cascade ↔ PRDS 위반의 직접 연결
4. **Theorem D 정당화**: e-BH가 arbitrary dependence에서 most powerful

## 5. 추가 결과 (full n)

### Mondrian-BH (long-CoT)

**Mondrian-BH**: step을 bin (예: 10개)으로 나눠서 각 bin 안에서 BH 적용. 각 bin 내에서는 PRDS 회복 가능성.

| 모델 | bin_size | α | Mondrian-BH ka 향상 |
|---|---|---|---|
| **R1-Distill-Llama-70B** | 20 | 0.20 | **+3.17pp** |
| R1-Distill-Llama-70B | 10 | 0.10 | +2.19pp |
| R1-Distill-32B | best | best | +1.22pp |
| QwQ-32B | best | best | 음수 (오히려 손해) |

→ 1/3 cells에서만 +2pp 목표 달성. **R1-Distill-70B만 강함**.

### Theorem A-E 검증 (15-cell full n)

| Theorem | 결과 |
|---|---|
| **A** Mondrian WB-PRDS bin sufficient | **REFUTED** — bin 2-20 모두 fail (frac_monotone=0.000 전부) |
| **B** Coverage ≠ FDR | **CONFIRMED 15/15** (Approach A FAR=0.43-0.85, vs α=0.10) |
| **C** Adaptive BY (Storey π₀) | **CONFIRMED 15/15** — 1.84-9.50pp more powerful than BY |
| **D** e-BH (Wang-Ramdas 2022 dominance) | **CONFIRMED 15/15** — 0.82-4.24pp better than BY |
| **E** Hierarchical BY | **inoperable** — n=100-200에서 outer BY 미발동 |

### 수치 정정

원래 spec "13× tighter than Bonferroni at T=60" → 실제 $H_{60} \approx 4.68$ (4.7×). cfBH paper = JMLR 2024 (vol. 24), not ICML 2023.

## 6. 솔직한 한계

### 한계 1: Mondrian-BH 1/3 cells positive
R1-Distill-70B만 강함. R1-32B 약함, QwQ-32B 음수 → method가 model-dependent.

### 한계 2: Bonferroni가 이미 거의 perfect
Many cells에서 Bonferroni kf ≈ 1.00 → BH/BY/Mondrian의 추가 개선 작음. PRDS 깨짐을 입증해도 실용 효과 미미.

### 한계 3: e-BH 현재 implementation에서 BH와 동일
True e-value 사용 시 e-BH가 BH 대비 $H_T \times$ 우월할 텐데, 우리 현재 e-value 구성이 thresholded → BH로 collapse.

## 7. Paper 구조 권장 (개정)

### 원래 framing
"Mondrian-BH가 winner — long-CoT에서 +3.17pp 향상"

### 개정 framing (recommended)

**Headline**: 
1. **Theorem 4-ter (PRDS counterexample)** — 가정 깨짐 자체가 contribution
2. **Theorem D (e-BH dominance)** — Wang-Ramdas 2022 응용으로 constructive replacement

Mondrian-BH는 corollary / empirical case study로 격하.

### Stein's Paradox 패턴 직접 인용
> "Stein이 sample mean의 inadmissibility를 보여 shrinkage 분야를 열었듯이, 우리는 PRDS-on-CP의 boundary를 그어 e-BH 등 next-gen procedure를 정당화한다."

## 8. 다른 논문과의 차별화

| 논문 | 우리 차별 |
|---|---|
| **Bates 2023** (PRDS-on-CP) | autoregressive setting에서 systematic fail 입증 |
| BH 1995, BY 2001 | per-step CoT 첫 적용 paper |
| **Wang-Ramdas 2022** (e-BH) | 우리가 LLM CoT에 적용 (Theorem D) |
| arXiv 2604.19775 Step-wise CP for agents (2026-04) | 다른 regime (interpretability vs FDR control) |

## 9. Top-tier Conference 게재 가능성 평가

**Cross-validated median**: 0.27 [0.05, 0.65] (큰 범위)

| 평가자 | P(top-tier) | 이유 |
|---|---|---|
| 비판적 reviewer | 0.117 | "negative result + 1/3 cells = inconclusive" |
| Base-rate 보정 | 0.60 | "PRDS falsification + Theorem 4-ter formal proof = clean Stein-style negative+constructive" |
| Empirical audit | 0.27 | "PRDS categorical, Mondrian thin" |

**가장 큰 disagreement**: negative result framing이 NeurIPS/ICLR에서 받아들여지는가?

## 10. 다음 액션

1. **e-BH dominance를 paper headline으로 reframe** — Theorem D 강조, Mondrian-BH 격하
2. **Theorem 4-ter T=2 counterexample 시각화** — figure로 PRDS 깨짐 메커니즘 보여주기
3. **True e-value construction** — 현재 BH-collapse 문제 해결 → e-BH 우월성 진짜로 입증
4. **PRDS-preserving score family 탐색** — 어떤 score는 PRDS 보존? (predictive content)

## 11. Venue 권장

- **TMLR** (primary) — negative-result + formal proof 환영, no novelty bar
- **UAI 2026** — theory-friendly, negative results 환영
- **NeurIPS** (e-BH headline reframe + visualizable counterexample 시) — top-tier 가능
- **ICLR fallback** — empirics가 약해서 어려움

## 산출물 위치

- 코드: `experiments/src/per_step_bh_fdr.py`, `per_step_fdr_v3_full.py`, `per_step_fdr_longcot.py`, `per_step_fdr_truefull_rerun.py`
- 결과: `experiments/results/per_step_fdr_truefull/summary_v3.json` + 12 short-CoT + 3 long-CoT
- 이론: `literature/theorems_drafts/theorem4bis_fdr.md`, `theorem4ter_formal.md`, `theorem4_new_theorems.md`
- Paper: `literature/concept_papers/bh_fdr_FINAL.md`, `bh_fdr_VERIFICATION.md`

## 12. 한 줄 요약

> "**LLM CoT의 per-step conformal에서 BH 가정 (PRDS)가 systematic하게 깨진다**. T=2 minimal counterexample로 정식 증명. Cumulative aggregation = quality funnel = PRDS 위반 메커니즘. e-BH (Wang-Ramdas 2022)로 대체 정당화."

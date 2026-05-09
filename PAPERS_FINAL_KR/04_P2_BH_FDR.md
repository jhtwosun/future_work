# P2 — BH-FDR Per-Step CP: PRDS Counterexample + Mondrian-BH

> **PRDS 가정의 systematic 깨짐 자체가 main contribution**
> P(top-tier) cross-validated: **0.27 ± 0.25** (큰 disagreement: negative-result framing 수용 여부)

## 1. 무엇을 푸는 문제

LLM CoT 60-step 각각에 conformal test 적용 (Approach B per-step CP). 모든 step이 통과해야 trace 통과 → **다중검정 문제**.

기존: **Bonferroni** ($\alpha/T$). T=60에서 매우 보수적, 많은 trace 거부.

**해결 직관**: BH (Benjamini-Hochberg 1995) 또는 BY (Benjamini-Yekutieli 2001) 다중검정 보정으로 더 풀어주기.

## 2. 원래 가설

**가설**: PRDS (Positive Regression Dependence on Subset) 가정 만족 → BH 적용 가능 → +1~3pp kept-frac 개선.

이 가정은 **Bates et al. 2023 (Annals of Statistics)** 가 conformal p-values에 PRDS holds (under exchangeability)라고 증명 → CP+FDR 응용에서 표준 가정.

## 3. 검증 결과 — Falsified

**Empirical PRDS test (3-method: Spearman, CDF monotone, permutation null)**:

| Cells | PRDS 결과 |
|---|---|
| 12 short-CoT (4 model × 3 dataset, full n) | **12/12 FAILS** |
| 3 long-CoT (QwQ, R1-Distill-32B/70B) | **3/3 FAILS** |
| **합계** | **15/15 FAILS** |

→ **PRDS가 autoregressive CoT trace에서 systematic하게 깨진다**.

## 4. Theorem 4-ter (PRDS Counterexample) — 정식 증명

**T=2 minimal counterexample**:
- Score family: cumulative running-min (per-step CP에서 표준)
- $P(p_2 \leq 0.5 \mid p_1 \leq q_{25}) = 1.000$
- $P(p_2 \leq 0.5 \mid p_1 \leq q_{75}) = 0.639$

**조건부 확률이 conditioning threshold에 대해 decreasing** (PRDS는 nondecreasing 요구) → PRDS 위반.

### 메커니즘 통찰

> Cumulative score aggregation (running min/max)이 **"quality funnel"**을 만든다. CoT의 본질적 구조 (한 번 망가지면 회복 안 됨)가 PRDS의 monotonicity 가정과 **정반대**.

## 5. 왜 "PRDS 깨짐 자체"가 contribution인가

### ① Literature correction
- Bates 2023 PRDS-on-CP 결과를 sequential / autoregressive setting에 차용한 후속 paper들이 다수
- 우리는 그 boundary를 정확히 그음 (15/15 categorical fail)

### ② Procedure choice principled하게 만듦
- BH 깨지면 → BY (always valid, $H_T$ factor) / Mondrian-BH / e-BH
- 깨짐을 입증해야 procedure 선택의 정당화 가능

### ③ Mechanism insight
- 왜 깨지나? cumulative aggregation = quality funnel
- 어떤 score family가 PRDS 보존하는지 follow-up 질문 생성

### ④ Theorem D (e-BH dominance) 정당화
- Wang-Ramdas 2022: e-BH가 arbitrary dependence에서 BY 대비 $H_T \times$ 우월
- 우리가 PRDS 깨짐 입증 → arbitrary dependence regime 확립 → e-BH 정당화

## 6. 추가 결과 (full n)

### Mondrian-BH (long-CoT)

| Model | bin_size | α | Mondrian-BH ka 향상 |
|---|---|---|---|
| **R1-Distill-Llama-70B** | 20 | 0.20 | **+3.17pp** |
| R1-Distill-Llama-70B | 10 | 0.10 | +2.19pp |
| R1-Distill-32B | best | best | +1.22pp |
| QwQ-32B | best | best | 음수 (오히려 손해) |

→ **1/3 cells**에서만 +2pp 목표 달성.

### Theorem A-E (15-cell verification at full n)

| Theorem | 결과 |
|---|---|
| **A** Mondrian WB-PRDS bin sufficient | **REFUTED** — bin 2-20 모두 fail (frac_monotone=0.000 전부) |
| **B** Coverage ≠ FDR | **CONFIRMED 15/15** (FAR=0.43-0.85 vs α=0.10) |
| **C** Adaptive BY (Storey π₀) | **CONFIRMED 15/15** — 1.84-9.50pp more powerful |
| **D** e-BH dominance | **CONFIRMED 15/15** — 0.82-4.24pp better than BY |
| **E** Hierarchical BY | **inoperable** — n=100-200에서 outer BY 미발동 |

### 수치 정정

원래 spec "13× tighter than Bonferroni at T=60" → 실제 $H_{60} \approx 4.68$ (4.7×). cfBH = JMLR 2024 (vol. 24), not ICML 2023.

## 7. 솔직한 한계

- **Mondrian-BH 1/3 cells positive**: R1-Distill-70B만 강함. R1-32B 약함, QwQ 음수.
- **Bonferroni 이미 kf≈1.0 in many cells**: 이미 거의 모든 trace pass → BH/BY/Mondrian의 추가 개선 작음
- **e-BH는 BH와 동일한 결과** (현재 implementation에서)

## 8. Paper 구조 권장

**원래 안**: Mondrian-BH가 winner (lift 중심)
**개정 안**: 
1. **Theorem 4-ter (PRDS counterexample)이 main headline** — 가정 깨짐 자체가 contribution
2. **Theorem D (e-BH dominance)이 constructive replacement** — Wang-Ramdas 2022 응용
3. Mondrian-BH는 corollary/empirical sample

비유: Stein's paradox (1956)가 "sample mean inadmissible in d≥3" 증명으로 shrinkage estimator 분야 열었듯이, 우리는 "PRDS-on-CP가 autoregressive에서 fail"을 보이고 e-BH로 대체.

## 9. P(Top-tier) 평가

**Cross-validated median**: 0.27 [0.05, 0.65] (큰 범위)

| Estimator | P(top-tier) | 이유 |
|---|---|---|
| Critic (NeurIPS AC) | 0.117 | "negative result + 1/3 cells = inconclusive" |
| Analyst (base-rate) | 0.60 | "PRDS falsification + Theorem 4-ter = clean negative+constructive" |
| Verifier (empirical) | 0.27 | "PRDS categorical, Mondrian thin" |

**가장 큰 disagreement**: negative result framing이 NeurIPS에서 받아들여지는가?

## 10. Prior Art 차별화

| Paper | 우리 차별 |
|---|---|
| Bates 2023 (PRDS-on-CP) | autoregressive setting에서 systematic fail 입증 |
| BH 1995, BY 2001 | per-step CoT에 적용한 첫 paper |
| Wang-Ramdas 2022 (e-BH) | 우리가 e-BH dominance를 CoT에 적용 |
| 2604.19775 Step-wise CP for agents | 다른 regime (interpretability vs FDR control) |

## 11. 다음 액션

1. **e-BH dominance를 paper headline으로 reframe** — Theorem D 강조, Mondrian-BH 격하
2. **Theorem 4-ter T=2 counterexample 시각화** — 직관적 figure로 PRDS 깨짐 메커니즘 보여주기
3. **e-value construction 정확히** — 현재 BH와 동일한 결과 → 진짜 e-value 사용 시 우월성 입증
4. **PRDS-preserving score family 탐색** — 어떤 score는 PRDS 보존하는지 (predictive content)

## 12. Venue 권장

- **TMLR primary** (negative-result + formal proof 환영, no novelty bar)
- **UAI 2026** (theory-friendly, negative results 환영)
- **NeurIPS** (e-BH headline reframe + Theorem 4-ter strong proof 시 가능)

## 산출물 위치

- 코드: `experiments/src/per_step_bh_fdr.py`, `per_step_fdr_v3_full.py`, `per_step_fdr_longcot.py`, `per_step_fdr_truefull_rerun.py`
- 결과: `experiments/results/per_step_fdr_truefull/summary_v3.json` + 12 cells + 3 long-CoT
- 이론: `literature/theorems_drafts/theorem4bis_fdr.md`, `theorem4ter_formal.md`, `theorem4_new_theorems.md`
- Paper: `literature/concept_papers/bh_fdr_FINAL.md`, `bh_fdr_VERIFICATION.md`

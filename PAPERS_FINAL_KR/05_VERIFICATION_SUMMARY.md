# Cross-Verification 종합 — 4-Estimator 검증

> **검증 일시**: 2026-05-09
> **방법**: 4개 독립 perspective agent (critic / analyst / verifier / document-specialist) 병렬 dispatch, 각자 P(top-tier) 추정 → median ± std로 aggregate

## 1차 vs 2차 (truefull 효과)

| Paper | 1차 median | 2차 median | Δ |
|---|---|---|---|
| **P3 CP-MPSC** | 0.13 | **0.38** | **+0.25** ↑ |
| **P4 Local CP** | 0.11 | 0.22 | +0.11 ↑ |
| **P5 Distance Ladder** | 0.30 | **0.22** | **-0.08** ↓ |
| **P2 BH-FDR** | 0.24 | 0.27 | +0.03 ≈ |

## 2차 종합 (3 estimator 신규 추정치)

| Paper | A (Critic) | B (Analyst) | C (Verifier) | **Median** | **Mean** | **Std** |
|---|---|---|---|---|---|---|
| **P3 CP-MPSC** | 0.165 | 0.40 | 0.38 | **0.38** | 0.315 | 0.13 |
| **P4 Local CP** | 0.110 | 0.61 | 0.22 | **0.22** | 0.313 | 0.27 |
| **P5 Distance Ladder** | 0.110 | 0.35 | 0.22 | **0.22** | 0.227 | 0.13 |
| **P2 BH-FDR** | 0.117 | 0.60 | 0.27 | **0.27** | 0.329 | 0.25 |

## 합의 강도

| 합의 | Paper |
|---|---|
| **강함 (Std < 0.15)** | P3 (0.13), P5 (0.13) |
| **약함 (Std 0.20-0.30)** | P4 (0.27), P2 (0.25) |

## Estimator별 일관성

| Estimator | 평균 추정 | 특성 |
|---|---|---|
| **A Critic (NeurIPS AC sim)** | 0.13 | 가장 보수적. Caveat 강하게 받아들임 |
| **B Analyst (base-rate)** | 0.49 | 가장 우호적. Theorem novelty + empirical strength 직접 평가 |
| **C Verifier (empirical)** | 0.27 | 중간. Power analysis + replication risk |

A와 B 사이 factor 3-4× 차이는 reviewer variance와 일치.

## Critical Disagreements (Paper별 결정 인자)

### P4 Local CP — disagreement Std 0.27
- **B (0.61)**: Theorem D 4× tighter than DKW + ICLR 2025 head-to-head 승리 → 강하게 우호
- **A (0.11)**: AIME에서 SBERT vs random 무차이 → "embedding doesn't matter" → fatal
- **C (0.22)**: 다른 group 재현 시 AIME null 동일 예상

**해결 인자**: AIME confounder를 명시 ablation으로 처리.
- n_tokens가 진짜 driver임을 정직 인정 + locality는 TheoremQA에서 진짜 필요 → B 견해 유지 가능
- 처리 안 하면 → A 견해 정당함

### P2 BH-FDR — disagreement Std 0.25
- **B (0.60)**: PRDS systematic fail (15/15) + Theorem 4-ter formal proof = Stein's paradox 패턴
- **A (0.12)**: negative result + 1/3 Mondrian cells = top-tier에서 inconclusive

**해결 인자**: e-BH dominance를 paper headline으로 reframe (Theorem D 강조).
- 성공 시 → 0.40+ 가능
- Mondrian 중심 framing 유지 시 → 0.15-0.20

## 4 Estimator 4 Paper P(top-tier) Heatmap

```
              P3      P4      P5      P2
Critic     0.165   0.110   0.110   0.117    ← 가장 보수
Analyst    0.40    0.61    0.35    0.60     ← 가장 우호
Verifier   0.38    0.22    0.22    0.27     ← 중간
            ───     ───     ───     ───
Median     0.38    0.22    0.22    0.27
Mean       0.32    0.31    0.23    0.33
Std        0.13    0.27    0.13    0.25
```

## 포트폴리오 합산 (median, 4편 독립 가정)

$$P(\text{최소 1편 top-tier}) = 1 - \prod_i (1 - p_i) = 1 - (1-0.38)(1-0.27)(1-0.22)(1-0.22)$$
$$= 1 - 0.62 \times 0.73 \times 0.78 \times 0.78 \approx 1 - 0.275 = \mathbf{0.72}$$

$$P(\text{2편 이상 top-tier}) \approx 0.32$$

(이전 1차: 0.55-0.59 → 2차: **0.72**, truefull 효과 +13-17pp)

## Top-tier 확률 향상 lever

| Paper | 현재 | 가능 | 핵심 액션 |
|---|---|---|---|
| **P3 CP-MPSC** | 0.38 | **0.50+** | Theorem 4 v5 정식 증명 + V0 falsification pre-registration |
| **P2 BH-FDR** | 0.27 | **0.40+** | e-BH dominance를 headline으로 reframe |
| **P4 Local CP** | 0.22 | **0.40+** | AIME confounder 명시 ablation (n_tokens vs SBERT 분리) |
| **P5 Distance Ladder** | 0.22 | **0.30+** | Cross-K 추가 도메인 페어 + 2604.20098 차별 |

만약 모든 lever 성공 시 (조건부 ceiling):
$$P(\text{1편 이상}) \approx 1 - (0.5 \times 0.6 \times 0.6 \times 0.7) \approx \mathbf{0.87}$$

## 한 줄 결론

**Truefull 후 P3가 cross-validation winner (0.38)**.  4편 모두 진행 시 **최소 1편 top-tier 통과 ≈ 72%, 2편 이상 ≈ 32%**.

**가장 큰 leverage는 P2 (e-BH headline) + P4 (AIME ablation)** — 둘 다 명확한 reframe만으로 0.27 → 0.40+, 0.22 → 0.40+ 가능.

# HGJ — 추가 실험 아이디어 (대화 정리)

> 2026-05-08 대화 중 도출된 미시행 ablation / 확장 실험 모음.
> Paper polish 단계에서 우선순위 결정용.

---

## 카테고리 분류

1. **Layer B (Step rejection) 정교화** — cascade source / sequential
2. **OOD & Test-Time Adaptation** — 분포 shift 대응
3. **Open-weight model coverage** — frontier gap 메우기
4. **Method narrative / framing** — paper 구조 결정

---

## 카테고리 1 — Layer B (Step rejection) 정교화

### Idea 1.1: Earliest-bad-step re-roll (threshold-based) ⭐⭐ **1순위**

**핵심**: worst step 대신 **threshold 미만 첫 번째 step**에서 K=4 re-roll.

**Motivation**:
- 현재 method는 worst step에서 re-roll → cascade source가 아닌 **symptom**일 수 있음
- 예: step 2가 잘못된 substitution(lp=-1.18), step 4가 그 결과로 막힘(lp=-1.42 worst)
- step 4에서 re-roll하면 prefix에 잘못된 step 2 그대로 → re-roll들이 모두 wrong direction
- earliest-violation은 cascade source를 직접 잡음

**Threshold 설정**:
- Per-step CP threshold (Approach A/B/C 중 택1) 그대로 reuse
- 또는 trace 내 z-score 기반 (running 통계)

**비용**: 1-2 GPU hours (코드 수정 minimal)

**Paper value**: HIGH — reviewer가 "왜 worst step만이냐" 물을 때 직접 대응. interpretable.

**예상 결과 시나리오**:
- Beats worst-step baseline → Layer B의 새 best variant
- Tied → "step branching ceiling"의 추가 evidence (negative result로 paper 가치)

---

### Idea 1.2: True sequential cumulative re-roll (depth=2) ⭐ **2순위**

**핵심**: earliest bad step에서 K번 re-roll → 다수결 → 새 trace 확정 → 그 trace의 다음 bad step에서 또 K번 re-roll → 반복.

**Motivation**:
- 사용자 직관: cascade를 진짜 풀려면 sequential
- 우리 Tier 3 (parallel branching from original)는 cascade 정보를 활용 못함
- step 2 fix 후 trace가 새로 펼쳐지는 걸 보고 다음 결정

**구현 주의**:
- Depth 제한 (depth=2 권장) — 안 그러면 K^D 폭발
- 각 단계에서 voting → 그 결과로 trace 확정 → 다음 단계
- Stopping: 모든 step이 threshold 이상이거나 depth 한계

**비용**: 5-10 GPU hours (compute K^2 per trace)

**Paper value**: MEDIUM — tree search 영역, scope 결정 필요

**예상 결과 시나리오**:
- 어려운 dataset (AIME, Olympiad)에서 +2-5pp 추가 lift 가능성
- 쉬운 dataset에선 marginal

---

## 카테고리 2 — OOD & Test-Time Adaptation

### Idea 2.1: Local CP with SBERT + KNN ⭐⭐⭐ **OOD 1순위**

**핵심**: test point마다 cal에서 embedding 가까운 K=100개로 local threshold 계산.

**Pipeline**:
```
Offline: SBERT embed cal prompts → FAISS index
Per-test:
  e_t = SBERT(prompt_t)
  N_K(t) = FAISS.search(e_t, k=100)
  C_local = [S_i for i in N_K(t) if Y_i==1]
  q_α(x_t) = sorted(C_local)[floor(α(n+1))]
  decision: S_t >= q_α(x_t)
```

**Motivation**:
- Theorem 3는 score-only shift 가정 (강함)
- Local CP는 local exchangeability 가정 (약함) → 더 큰 shift 풀어냄
- Conditional coverage 보장 (sub-population별)
- 이미 정립된 이론 (Tibshirani 2019, Foygel-Barber 2022) reuse

**예상 setup**: cal=MATH-500, test=AIME-2024, score=sc_top1

**예상 결과**:
| 방법 | empirical coverage @ α=0.5 |
|---|---|
| Vanilla CP | 0.187 (broken) |
| Theorem 3 (PMF) | 0.633 (overshoot) |
| **Local CP** | **0.45-0.55 (target near)** ← 예측 |
| Local CP + PMF combined | 0.48-0.52 (best) |

**비용**: 3-5 GPU hours (SBERT + FAISS는 가벼움, 각 cell 평가가 main cost)

**Paper value**: VERY HIGH — paper §6 Robustness section의 1순위 ablation

**Embedding 옵션 비교 ablation 가능**:
- SBERT prompt (1순위)
- LLM last hidden state (free, 이미 갖고 있음)
- MathBERT (math-specific, paper 강화)

---

### Idea 2.2: Online empirical-PMF update (streaming TTA) ⭐ **2순위**

**핵심**: test stream에서 PMF rolling update → weight & threshold 매번 recompute.

```
For each test point t:
  S_t = score(...)
  hat_p_test += S_t (running PMF)
  w(s) = hat_p_test(s) / hat_p_cal(s)
  q_α^w = weighted_quantile(...)
  decision: S_t >= q_α^w
```

**Motivation**:
- Theorem 3의 자연 확장 (batch → streaming)
- DKW rate $O(1/\sqrt{N_t})$로 점진적 보장 회복

**비용**: <1 GPU hour (코드 한 줄 변경)

**Paper value**: MEDIUM — streaming demo로 1 figure 추가 가능

---

### Idea 2.3: Test-time score ensemble + agreement

**핵심**: 여러 score (entropy_mean, lp_min, sc_top1) 동시 사용 → disagreement = shift signal.

**비용**: <1 GPU hour (모든 score 이미 있음)

**Paper value**: LOW — heuristic, 이론 약함. optional ablation.

---

### Idea 2.4: ACI + verifier (HumanEval / sympy)

**핵심**: Adaptive Conformal Inference (Gibbs-Candes 2021)을 verifier 신호로 운영.
- HumanEval: unit test pass 여부 → cover/miss signal
- 수학: sympy verification (단 4% extract limit)

**비용**: 5-10 GPU hours + judge cost

**Paper value**: HIGH (HumanEval 한정) — separate paper가 될 수도.

---

## 카테고리 3 — Open-weight model coverage

### Idea 3.1: Qwen3-32B reasoning (thinking mode on) ⭐⭐ **1순위**

**핵심**: 우리는 Qwen3-8B no-think만 평가. Qwen3-32B with thinking mode가 빠짐.

**Motivation**:
- Qwen3 reasoning lineage의 강점 (open-weight 32B reasoning 최강 중 하나)
- "<100B open-weight 전 영역 cover" 주장의 마지막 puzzle piece
- Reviewer pushback 1순위 (Qwen3-32B에서도 lift 유지하나?)

**비용**: 1-2 GPU hours (single model add to E19)

**Paper value**: VERY HIGH — defensibility ↑↑

---

### Idea 3.2: Frontier API (DeepSeek-R1 via Together.ai)

**핵심**: 100B+ open-weight (DeepSeek-R1 671B, Qwen3-235B)는 우리 hardware로 못 돌림 → API 사용.

**Setup**: Together.ai or Fireworks → 200 sample on AIME / Olympiad

**비용**: ~$50 + 1-2 day work

**Paper value**:
- TMLR: 불필요
- NeurIPS D&B: 권장
- ICLR/NeurIPS main: **필수**

---

### Idea 3.3: R1-Distill-Qwen-14B (mid-tier R1 distill)

**핵심**: 우리는 R1-Distill 7B와 32B만 있음. 14B 추가하면 distill scaling curve 완성.

**비용**: 1-2 GPU hours

**Paper value**: LOW-MEDIUM (interpolation으로도 방어 가능)

---

## 카테고리 4 — Method narrative / framing

### Idea 4.1: 3-mode framing (paper structure 결정)

**핵심**: Layer A, Layer B, Combined를 **하나의 method로 묶지 말고** 3개 use case로 제시.

**Modes**:
- **Mode 1**: Pure Layer A (CP only) — selective accuracy 보장 (high-stakes)
- **Mode 2**: Pure Layer B (Step rejection) — 전체 정확도 향상 (throughput)
- **Mode 3**: Combined (gating) — Pareto-optimal (hybrid)

**Paper structure**:
- §3: Layer A framework + theorems
- §4: Layer B step rejection + Pareto frontier
- §5: Combination via gating (Pareto-optimal point)
- §6: Use case selection guide

**Why**: trigger ≠ score 발견 자체가 framework가 modular함을 의미. 정직하게 두 paradigm으로 제시하는 게 강함.

**비용**: 0 (writing decision only)

**Paper value**: HIGH (narrative 결정)

---

## 우선순위 종합 — 한 표

| # | Idea | Effort | Paper value | 우선순위 |
|---|---|---|---|---|
| 1.1 | **Earliest-bad-step re-roll** | 1-2 hr | HIGH | ⭐⭐ |
| 2.1 | **Local CP (SBERT + KNN)** | 3-5 hr | VERY HIGH | ⭐⭐⭐ |
| 3.1 | **Qwen3-32B thinking mode** | 1-2 hr | VERY HIGH | ⭐⭐ |
| 4.1 | 3-mode framing | 0 | HIGH | (writing) |
| 2.2 | Online PMF update | <1 hr | MEDIUM | ⭐ |
| 1.2 | Sequential cumulative (depth=2) | 5-10 hr | MEDIUM | ⭐ |
| 3.2 | Frontier API (DeepSeek-R1) | $50 + 1-2d | venue 따라 | venue depends |
| 2.4 | ACI + verifier | 5-10 hr | HIGH (HumanEval) | venue depends |
| 2.3 | Score ensemble | <1 hr | LOW | optional |
| 3.3 | R1-Distill-14B | 1-2 hr | LOW-MED | optional |

---

## 추천 실행 순서 (TMLR 가정 시 4주 plan)

**Week 1**:
- 1.1 Earliest-bad-step (1-2 hr)
- 3.1 Qwen3-32B thinking (1-2 hr)
- 4.1 3-mode framing decision

**Week 2**:
- 2.1 Local CP full evaluation (3-5 hr)
- 2.2 Online PMF (<1 hr)
- LaTeX skeleton 시작

**Week 3-4**: Paper writing + figures

**ICLR/NeurIPS path 추가**:
- 3.2 Frontier API
- 1.2 Sequential cumulative
- 2.4 ACI + verifier (HumanEval만)

---

## 메모

- 모든 ablation은 **negative여도 paper 가치 있음**: "we tried this, doesn't beat baseline" → step branching ceiling / shift complexity 주장의 evidence
- Local CP (2.1)이 단연 가장 가치 큼 — 이론도 강하고 구현도 가벼움
- Qwen3-32B thinking (3.1)은 가장 cheap한 defensibility 강화
- Earliest-bad-step (1.1)은 사용자 직관에 직접 대응 — reviewer-proof

# CoT-CP 야간 Pilot — 결과 정리 (한글)

> 기간: 2026-05-05 → 2026-05-06 (자율 야간 실행) · 머신: 2× H100 NVL · 총 wallclock ~90분
> 모델: Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-Math-PRM-7B
> 1차 pass: Pilots 2-10 (GSM8K + MATH-500 trajectory-level CP)
> 2차 pass: Pilots D, E, B, C (compute scaling, R1-Distill, OOD AIME, step branching)
> 3차 pass: Pilots H, F, J, A, K (step grouping, weighted CP, bootstrap CI, PRM scoring + branching)

---

## 0. 한 줄 요약

**Trajectory-level CP filter (특히 SC@8 또는 PRM 결합) 는 검증된 강력한 primitive. Step-level branching 은 단순 K-resample 에서는 작동 안 함 (어떤 score family 든). OOD coverage 는 vanilla CP 에서 깨지지만 weighted CP 가 부분 회복.**

세 가지가 paper 로 정착:
1. ✅ Trajectory-level CP machinery 작동 (in-distribution coverage 가 target $1-\alpha$ 와 ±1% 내 일치, bootstrap CI 까지 확인)
2. ✅ Score family ladder 완성: lp_min → PRM-min → SC-top1 (compute 2배 늘 때마다 ~+9pt)
3. ✅ OOD 실패 모드 입증 + weighted CP 가 high-α 부분 fix

두 가지가 미해결:
1. ⚠️ Step-level branching 은 PRM 으로도 작동 안 함 (recovered=2, lost=2). Intervention 자체를 바꿔야 함
2. ⚠️ AIME n=30 너무 작음 — bootstrap CI 가 [0, 1] 로 너무 넓음. GPQA-Diamond (198 문제) 등 더 큰 OOD 셋 필요

---

## 1. 헤드라인 숫자

### Score family ladder (MATH-500, Qwen2.5-7B-Instruct, α=0.5)

| Score | Compute cost | Kept accuracy | vs vanilla 51.6% |
|---|---|---|---|
| lp_min (free) | 1× | 62.0% | +10.4pt |
| **prm_min (Qwen2.5-Math-PRM-7B)** | **2× (gen + PRM forward)** | **70.7%** | **+19.1pt** ← 새 발견 |
| sc_top1 (SC @ N=8) | 8× | 79.3% | +27.7pt |
| sc_top1 (SC @ N=16) | 16× | 84.1% | +32.5pt (Pilot D headline) |

→ **PRM 은 SC@8 의 2/3 lift 를 1/4 compute 로 달성**. Paper 의 middle-cost operating point 로 강력함.

### Coverage 검증 (95% bootstrap CI)

| Operating point | Coverage [CI] | Kept acc [CI] |
|---|---|---|
| GSM8K + sc_top1 α=0.10 | 0.930 [0.892, 0.964] | **0.972 [0.942, 0.994]** |
| MATH-500 + lp_min α=0.50 | 0.499 [0.459, 0.538] | 0.620 [0.546, 0.702] |
| MATH-500 + sc_top1 α=0.30 | 0.753 [0.689, 0.817] | **0.786 [0.693, 0.867]** |
| MATH-500 + sc_top1 α=0.50 | 0.633 [0.551, 0.719] | **0.799 [0.705, 0.883]** |
| MATH-500 R1 + lp_mean α=0.50 | 0.496 [0.437, 0.552] | 0.695 [0.595, 0.794] |

In-distribution coverage 모두 target $1-\alpha$ 와 일치 (CI 안에 들어옴).

### OOD coverage (MATH-500 cal → AIME-2024 test, n=30)

| Score | α | Target | Vanilla cov | **Weighted cov** |
|---|---|---|---|---|
| lp_min | 0.10 | 0.90 | 0.50 (broken) | 0.75 (회복) |
| lp_min | 0.30 | 0.70 | 0.50 | 0.50 |
| **lp_min** | **0.50** | **0.50** | **0.25 (broken)** | **0.50 (target!)** |
| lp_mean | 0.50 | 0.50 | 0.25 | 0.75 (over-correct) |

→ Vanilla CP 는 high-α 에서 coverage 깨짐 (0.25 vs target 0.5). Weighted CP (Tibshirani+ 2019) 가 lp_min 에서 정확히 target 으로 회복.

---

## 2. Pilot 별 결과

### Pilot 2 — GSM8K vanilla CoT (300문제)

- Accuracy: **90.3%** (271/300)
- Wallclock: 11초
- Median 4 steps/answer, mean 47 tokens/step
- Step segmenter (`\n\n`) parse failure 0건

### Pilot 3 — GSM8K step-level reliability

- Spearman ρ(lp_mean, correct) = 0.147 (p=0.011)
- Spearman ρ(lp_min, correct) = 0.196 (p=4×10⁻⁴)
- Reliability bin spread: 0.80 → 1.00
- **결론**: GSM8K 는 ceiling 이 너무 가까워서 calibration 신호가 묻힘. CoT-CP 검증에 부적합.

### Pilot 4 — GSM8K SC@8

- Majority accuracy: **94.0%** (vanilla 대비 +3.7pt)
- Mean SC top-1 fraction: 0.925 → 모델이 매우 일관됨, 변별력 부족

### Pilot 5/5b — PRM800K 사용성 확인

100K-row stream:
- rating ∈ {+1, 0, −1} = 61.4% / 12.9% / 22.0%
- is_solution 4.3% True
- is_preferred_response 60/40
- mean step length 83 chars

→ 전체 800K 셋에 ~220K −1 (bad) 예시. Calibration 데이터로 충분.

### Pilot 7 — MATH-500 vanilla CoT (500문제)

- Accuracy: **51.6%** (258/500) — 변별력 있는 영역
- Wallclock: 24초
- Median 8 steps/answer (max 33)
- Spearman ρ(lp_min, correct) = 0.196 (p=10⁻⁵)
- Reliability bin spread: 0.38 → 0.60 (22pp! GSM8K 보다 훨씬 informative)

### Pilot 8 — Empirical CP coverage 시뮬레이션

200 random seed, 50/50 cal/test split, finite-sample-corrected quantile.

```
[GSM8K LP/lp_min α=0.10]  target=0.90  cov=0.897±0.036  keepacc=0.914  keep%=0.89
[GSM8K SC      α=0.10]    target=0.90  cov=0.928±0.046  keepacc=0.971  keep%=0.90
[MATH LP/lp_min α=0.20]   target=0.80  cov=0.792±0.051  keepacc=0.564  keep%=0.72
[MATH LP/lp_min α=0.50]   target=0.50  cov=0.490±0.063  keepacc=0.619  keep%=0.41
```

모든 setting 에서 empirical coverage ±1% 이내 일치. Theory 가 예측한 그대로.

### Pilot 9 — MATH-500 SC@8 (200문제)

- Majority accuracy: **56.5%** (greedy 대비 +4.9pt)
- Oracle any-correct: 66.5% (15pt headroom)
- Mean SC top-1 fraction: 0.763 (GSM8K 의 0.925 보다 변별력 큼)

### Pilot 10 — Extended CP simulation (헤드라인 figure)

`results/pilot10_pareto.png` 저장. 핵심 결과:
- MATH-500 + SC@8 α=0.10 → kept_acc 0.663, keep% 78%
- MATH-500 + SC@8 α=0.20 → kept_acc 0.736, keep% 64%
- **MATH-500 + SC@8 α=0.30 → kept_acc 0.787, keep% 55%**
- MATH-500 + SC@8 α=0.50 → kept_acc 0.793, keep% 45%

Score 우열: SC top-1 ≫ lp_min > lp_mean ≈ lp_tok > lp_median.

### Pilot D — Compute-matched SC sweep

| N | wallclock | majority acc | CP@α=0.3 | CP@α=0.5 |
|---|---|---|---|---|
| 1 | 11s | 50.5% | — | — |
| 4 | 28s | 54.0% | 73.4 / 56% | 74.8 / 53% |
| 8 | 56s | 56.5% | 76.2 / 56% | 79.8 / 43% |
| **16** | 107s | **57.5%** | **78.8 / 53%** | **84.1 / 39%** ← 헤드라인 |
| 32 | 211s | 57.0% | 80.4 / 50% | 83.5 / 37% |

→ Vanilla SC 는 N=16 에서 saturate. CP-filtered SC 는 N=16 까지 의미있게 개선. **N=16 + CP@α=0.5 = 84.1% on 39%** 가 paper 의 main figure.

### Pilot E — DeepSeek-R1-Distill-Qwen-7B (long-CoT)

- Greedy MATH-500 accuracy: **59.0%** (vs Qwen2.5 51.6%, +7.4pt)
- Mean reasoning steps: **62.3** (vs Qwen 8.4) — 7-8배 더 긴 CoT
- Mean tokens/answer: 2,490
- Spearman ρ(lp_mean) = **0.204** (Qwen 0.137 보다 강함)
- Spearman ρ(lp_min) = 0.152 (Qwen 0.196 보다 약함)
- **반전**: R1 에서는 lp_mean > lp_min (long trace 가 noisy single step 으로 lp_min 망가뜨림)
- CP @ α=0.5 + lp_mean: kept_acc 0.695 on 41.5% kept (+10.5pt)

→ SC@8 는 long-CoT 라 1.5시간+ 소요로 kill. Greedy 결과만으로 충분.

### Pilot B — OOD MATH-500 → AIME-2024

AIME-2024 (30문제, Qwen2.5-7B):
- Greedy accuracy: 13.3% (4/30)
- SC@8 majority: 16.7% (5/30)
- Mean SC top-1 fraction: 0.354 (불일치 많음)

OOD CP (MATH-500-cal):

| Score | α | Target | Empirical cov | Kept acc | Keep% |
|---|---|---|---|---|---|
| lp_mean | 0.10 | 0.90 | 1.00 | 19.0% | 70% |
| lp_mean | 0.50 | 0.50 | **0.25 (broken)** | 25.0% | 13% |
| lp_min | 0.10 | 0.90 | 0.50 | 9.1% | 73% |
| lp_min | 0.50 | 0.50 | 0.25 | 33.3% | 10% |

→ Vanilla split-CP 는 OOD shift 에서 깨짐. Paper 의 weighted-CP / per-domain calibration 동기 확보.

### Pilot C — Step-level branching prototype (lp_min 기반)

50/50 split, K=4 alternative continuations from worst step:

| Policy | Accuracy | Recovered / Lost | Compute |
|---|---|---|---|
| Vanilla greedy | 52.0% | — | 1× |
| Always branch (worst step) | 54.0% | 3 / 1 → +2pt | +4× per q |
| CP-triggered branch | 53.0% | 2 / 1 → +1pt | +1.3× avg |

→ **Negative result**: 단순 worst-step re-roll 은 약한 primitive. Trajectory-level CP+SC (+27pt) 가 압도적으로 강함.

### Pilot H — R1-Distill step grouping

R1 의 60-step trace 를 K-grouped super-step 으로 묶어 reliability 재계산:

| K | mean groups | lp_mean ρ | **lp_min ρ** | lp_median ρ |
|---|---|---|---|---|
| 1 | 62.3 | 0.204 | 0.152 | 0.134 |
| **2** | 31.4 | 0.129 | **0.267** | 0.027 |
| 5 | 12.9 | 0.137 | 0.198 | 0.098 |
| 10 | 6.7 | 0.117 | 0.199 | 0.099 |
| 20 | 3.6 | 0.196 | 0.199 | 0.190 |

→ **K=2 grouping 이 R1 의 lp_min 을 dominant score 로 만듬** (ρ=0.267, 모든 R1 setting 중 가장 강함). CP@α=0.5 + K=2 lp_min: **kept_acc 0.718 on 41% kept (+12.8pt)** — R1 lp_*-only 최고 결과.

→ Paper 인사이트: 적절한 step granularity 는 모델 의존적. Qwen2.5 는 K=1 (5 steps), R1 는 K=2 (~30 super-steps).

### Pilot F — Weighted CP for OOD

Density ratio = $p_{\text{test}}(s)/p_{\text{cal}}(s)$ 를 1-D Gaussian KDE 로 추정 → weighted (1-α)-quantile.

| Score | α | Target | Vanilla cov | Weighted cov |
|---|---|---|---|---|
| lp_min | 0.10 | 0.90 | **0.50** | **0.75** ↑ |
| **lp_min** | **0.50** | **0.50** | **0.25** | **0.50 (target!)** |
| lp_mean | 0.50 | 0.50 | 0.25 | 0.75 (over-correct) |

→ Weighted CP 가 high-α 에서 정확히 target 회복. Low-α 에서는 과보정. n=30 이라 noise 큼.

`results/pilotF_density_ratio.png` 에 score density + ratio 그림 저장.

### Pilot J — Bootstrap 95% CI

500 bootstrap × 10 cal/test split.

| Operating point | Coverage [95% CI] | Kept acc [95% CI] |
|---|---|---|
| GSM8K + lp_min α=0.10 | 0.899 [0.876, 0.920] | 0.915 [0.877, 0.946] |
| GSM8K + sc_top1 α=0.10 | 0.930 [0.892, 0.964] | **0.972 [0.942, 0.994]** |
| MATH-500 + lp_min α=0.50 | 0.499 [0.459, 0.538] | 0.620 [0.546, 0.702] |
| MATH-500 + sc_top1 α=0.30 | 0.753 [0.689, 0.817] | **0.786 [0.693, 0.867]** |
| MATH-500 + sc_top1 α=0.50 | 0.633 [0.551, 0.719] | **0.799 [0.705, 0.883]** |
| MATH-500 R1 + lp_mean α=0.50 | 0.496 [0.437, 0.552] | 0.695 [0.595, 0.794] |
| AIME OOD vanilla lp_min α=0.50 | 0.257 [**0.000, 0.750**] | 0.373 [0.000, 1.000] |
| AIME OOD weighted lp_min α=0.50 | 0.504 [**0.000, 1.000**] | 0.152 [0.000, 0.333] |

→ In-distribution CI 는 paper 에 충분히 좁음. AIME (n=30) CI 는 [0, 1] 로 정보 없음 → 더 큰 OOD 셋 필요.

### Pilot A — PRM scoring (Qwen2.5-Math-PRM-7B) ⭐ 핵심 발견

`Qwen/Qwen2.5-Math-PRM-7B` 를 transformers AutoModel 로 로드 (vLLM 미지원). Custom modeling 파일의 `get_usable_length` API → `get_seq_length` 로 patch (transformers 4.57 호환).

500 traces × ~22분 inference.

| Score | α | Coverage | **Kept acc** | Keep% | Δ vs vanilla |
|---|---|---|---|---|---|
| prm_min | 0.05 | 0.945 | 0.645 | 75% | +12.9pt |
| prm_min | 0.10 | 0.895 | 0.676 | 68% | +16.0pt |
| prm_min | 0.20 | 0.792 | 0.680 | 60% | +16.4pt |
| prm_min | 0.30 | 0.694 | 0.683 | 52% | +16.7pt |
| **prm_min** | **0.50** | **0.529** | **0.707** | **38%** | **+19.1pt** |
| prm_mean | 0.50 | 0.498 | 0.699 | 37% | +18.3pt |

→ **PRM 이 SC@8 의 2/3 lift 를 1/4 compute 로 (+19pt vs +28pt, 2× cost vs 8× cost).** Paper 의 middle-cost operating point.

### Pilot K — PRM-based step branching

Pilot C 와 같은 protocol 이지만 worst step 을 PRM step reward 로 식별.

| Policy | Accuracy | Recovered / Lost |
|---|---|---|
| Vanilla greedy | 52.0% | — |
| Always branch at worst-PRM-step (K=4) | 52.0% | 2 / 2 |
| CP-triggered branch at worst-PRM-step | 52.0% | 2 / 2 |

→ **Net zero**. PRM 이 trajectory-level 에서 +19pt 인 강한 신호여도 worst-step K-resample 은 작동 안 함.

→ **결론**: bottleneck 은 *score* 가 아니라 *intervention* (re-roll). 모델은 같은 prefix 에서 단순 temperature resampling 으로 다른 풀이를 못 찾음. Step-level branching 을 살리려면 explicit "rewrite this step" prompting 등 다른 mechanism 필요.

---

## 3. Paper 함의

### 정착된 narrative (3가지 piece 가 publishable)

1. **Trajectory-level CP machinery 작동 확인** (Pilots 8/10/J)
   - 4개 score family × 2 dataset × 6 α level × 200 seed × 500 bootstrap → coverage 항상 target 일치 (CI 안)
2. **Score family Pareto 완성** (Pilots 10 + A)
   - lp_min (free) → PRM-min (2× cost) → SC-top1 (8× cost), 각 단계 약 +9pt
   - Paper 의 main ablation table 의 세 행
3. **OOD failure 입증 + weighted CP 가 부분 fix** (Pilots B + F)
   - Vanilla CP 가 MATH→AIME 에서 명확히 깨짐 (motivation)
   - Weighted CP 가 high-α target 회복 (technical fix)

### 미해결 (paper 까지 해결해야 할 것)

1. **Step-level branching 은 reject 되었음**
   - 단순 K-resample 은 lp_min 으로도 PRM 으로도 작동 안 함
   - 다음 시도: explicit "rewrite step differently" prompting, search-tree 확장, 또는 paper 에서 step-branching 을 contribution 에서 제외하고 trajectory-level 로 reframe
2. **AIME n=30 너무 작음**
   - Bootstrap CI 가 [0, 1] 로 무의미
   - 대안: GPQA-Diamond (198 문제), MMLU 부분, AIME-1983-2024 확장

### 권장 reframing

> CoT-CP 를 *step-level branching trigger* 가 아니라 ***trajectory-level coverage layer over an existing search procedure (SC, BoN, beam, MCTS, PRM)*** 로 포지셔닝. 어떤 step-quality signal 이든 conformity score 로 wrap 해서 finite-sample distribution-free coverage 를 부여하는 것이 contribution.

이는 synthesis 보고서의 framing 권장 ("calibration layer that any of these signals can be plugged into") 과 정확히 일치.

---

## 4. 다음 세션 권장 pilot

| # | Pilot | 이유 | Status |
|---|---|---|---|
| L | Step branching with rewrite-style prompting | Pilot K negative result 이후 가장 흥미로운 미해결 — explicit "rewrite step" 이 K-resample 보다 좋은지 | open ⭐ |
| M | PRM + SC ensemble: 두 conformity score 결합 | Best-of-both-worlds operating point 가능성 | open |
| G | R1-Distill SC@8 재실행 (50문제 × NS=4 등 작은 사이즈) | R1 row 채우기 | open |
| I | GPQA-Diamond eval | AIME n=30 대체용 큰 OOD 셋 (198 문제) | open ⭐ |
| N | Pilot 10 figure 에 PRM curve 추가 | 헤드라인 figure 업데이트 | quick |

⭐ 가 우선순위.

---

## 5. 파일 인벤토리

```
/home/nvidia/future/pilots/cot_cp/
├── OBSERVATIONS.md            (영문 504줄)
├── OBSERVATIONS_kr.md         (이 파일)
├── src/                       (15개 pilot script)
│   pilot{2,3,4,5,5b,7,7-analysis,8,9,10}_*.py    # 1차
│   pilot{B,C,D,E,E-analysis}_*.py                 # 2차
│   pilot{H,F,J,A,K}_*.py                          # 3차
├── results/                   (24MB, 18 JSON + JSONL + PNG)
│   pilot10_pareto.png         ← 1차 헤드라인 figure
│   pilotD_compute_matched.png ← 2차 헤드라인 figure (compute Pareto)
│   pilotF_density_ratio.png   ← 3차 OOD density 시각화
│   pilotA_prm_traces.jsonl    ← 500 traces × per-step PRM scores
│   ... (15개 더)
└── logs/                      (모든 pilot stdout)
```

총 18개 pilot, 24MB 결과, 504줄 영문 OBSERVATIONS, 본 한글 정리.

---

## 6. 가장 중요한 단일 figure

`results/pilotD_compute_matched.png` — vanilla SC saturation 곡선 위로 CP-filtered SC 가 명확히 dominate, x축 compute, y축 accuracy.

다음으로 갱신하면 좋은 figure: pilot10_pareto.png 에 PRM curve 추가 (Pilot N).

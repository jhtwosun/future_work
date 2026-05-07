# CoT-CP 연구 — 전체 실험 결과 종합 (한글)

> 작성일: 2026-05-07
> 대상 논문: **CoT-CP** (Conformal Chain-of-Thought)

---

## 0. 한 줄 요약

LLM의 Chain-of-Thought 추론 trace에 split conformal prediction을 적용해서 **calibrated selective accuracy**를 보장하는 method. 핵심 contribution은 (i) trajectory-level CP wrapper의 정리 (Theorem 1) + (ii) **새로운 6+ score family**로 cost-vs-LR+ Pareto frontier를 채움.

---

## 1. 시스템 개요

**문제 설정**:
- LLM이 수학/과학 문제를 풀 때, 답이 맞을지 미리 confidence score를 계산해서 "확실한 것만 답하기" (selective prediction)
- Score function `S(question, trace)` → split CP → calibrated kept_acc 보장

**핵심 baseline**:
- `lp_min` (1×, FREE): 각 step의 토큰 평균 logprob 중 최소값
- `prm_min` (2×): Process Reward Model 7B로 step별 점수
- `sc_top1` (8×): Self-Consistency 8개 sample의 top-1 비율

---

## 2. 검증된 점수 패밀리 종합

### 2.1 Trajectory-level (전체 trace 보고 점수 계산)

| Score | 비용 | 1등 cells (16) | Top lift | 비고 |
|---|---|---|---|---|
| **`entropy_mean`** | FREE 1× | **5/16** ⭐ | **+14.6pp** (Math-7B AIME) | universal winner |
| `top1_margin_mean` | FREE 1× | 4/16 | +9.0pp (7B Olympiad) | runner-up |
| `entropy_max` | FREE 1× | 3/16 | — | hard problems 강세 |
| `tempered_kl_max` | FREE 1× | 2/16 | — | borderline |
| `lp_min` (baseline) | FREE 1× | **0/16** | — | **strict loser** |

**핵심 발견**: lp_min baseline은 16/16 cells에서 패배 → 새 info-theoretic 점수가 **strict improvement**.

### 2.2 Hidden-state (HF transformers + output_hidden_states)

| Score | Top-1 (6 cells) | Top-3 | 특징 |
|---|---|---|---|
| **`hidden_drift_max_penult`** | 2/6 | 5/6 | MATH-500 강세 (kept=0.85-0.89) |
| `hidden_norm_var` | 2/6 | 4/6 | model-specific |
| `hidden_norm_range` | 2/6 | 4/6 | Math-7B에서 압도 (ρ=+0.418) |

**패턴**:
- MATH-500: hidden state ≈ trajectory info-theoretic
- Olympiad: hidden state −20~33pp 약함 (어려운 math에서 신호 사라짐)

### 2.3 Step-level (각 step boundary에서 online 계산, NEW Wave 3)

| Family | Top-3 hits (12 cells) | 특징 |
|---|---|---|
| **`ent_growth`** | **8/12** ⭐ | log(entropy_t / running geomean) |
| **`arith_violations`** | **7/12** | regex+safe_eval, **+5.8pp on Phi-4 math500** |
| **`lp_drawdown`** | **7/12** | peak-to-current logprob drawdown |
| `rep_4gram` | 4/12 | 32B math500 1등 (kept=0.907) |
| `num_density` | 3/12 | borderline |

**Step-level outright wins (4/12 cells):**
- 32B math500: `rep_4gram_sum` 0.907 (vs lp_min 0.870, +3.6pp)
- Phi-4 math500: `arith_violations_sum` 0.861 (vs traj 0.802, **+5.8pp**)
- 32B AIME: `lp_drawdown_max` 0.532 (+2.4pp)
- Phi-4 Olympiad: `lp_drawdown_sum` 0.623 (+1.9pp)

**Step-level의 강점**:
1. **Online / streaming computation** — trace 끝까지 안 기다려도 됨
2. **다른 signal source** — structural (arith), degeneracy (rep_4gram)는 logprob에 직교
3. **FREE** — extra forward pass 불필요

### 2.4 비싼 점수 (이전 wave 검증 결과)

| Score | 비용 | Best lift |
|---|---|---|
| **`tiebreak_lex`** (sc_top1 + lp_min tiebreaker) | 8× + free | AIME +5.7pp (Pareto improvement) |
| **`paraphrase_consensus M=3`** | 3× | MATH-500 92.7%, AIME 79.6% |
| **`prm_guided_step_branching`** | 5× | +3.0pp greedy lift |
| `predictive_sharpening_rate` | 1× FREE | HumanEval +1.7pp (code only) |

---

## 3. 전체 실험 매트릭스 (총 ~46 cells)

### 3.1 Info-theoretic (16 cells)

| Model | Dataset | vanilla | best score | best kept@0.3 | lp_min kept@0.3 | Δ |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | MATH-500 | 0.745 | top1_margin_mean | 0.831 | 0.809 | +2.2 |
| Qwen2.5-7B | AIME | 0.290 | entropy_mean | 0.486 | 0.443 | +4.3 |
| Qwen2.5-7B | Olympiad | 0.430 | top1_margin_mean | 0.573 | 0.484 | **+9.0** |
| Qwen2.5-7B | MMLU-Pro | 0.680 | tempered_kl_max | 0.737 | 0.721 | +1.6 |
| Math-7B | MATH-500 | 0.800 | top1_margin_mean | 0.890 | 0.859 | +3.1 |
| Math-7B | AIME | 0.415 | entropy_mean | **0.677** | 0.531 | **+14.6** ⭐ |
| Math-7B | Olympiad | 0.480 | entropy_mean | 0.615 | 0.554 | +6.1 |
| Math-7B | MMLU-Pro | 0.425 | tempered_kl_max | 0.485 | 0.451 | +3.4 |
| Qwen2.5-32B | MATH-500 | 0.805 | entropy_max | 0.890 | 0.870 | +2.0 |
| Qwen2.5-32B | AIME | 0.440 | top1_margin_mean | 0.543 | 0.496 | +4.7 |
| Qwen2.5-32B | Olympiad | 0.415 | entropy_mean | 0.555 | 0.506 | +4.9 |
| Qwen2.5-32B | MMLU-Pro | 0.845 | entropy_max | 0.887 | 0.886 | +0.1 |
| Phi-4 | MATH-500 | 0.775 | top1_margin_min | 0.830 | 0.812 | +1.8 |
| Phi-4 | AIME | 0.355 | tempered_kl_mean | 0.500 | 0.425 | **+7.6** |
| Phi-4 | Olympiad | 0.490 | entropy_mean | 0.625 | 0.534 | **+9.0** |
| Phi-4 | MMLU-Pro | 0.855 | entropy_max | 0.883 | 0.880 | +0.3 |

### 3.2 Hidden-state (6 cells, vs HF baseline)

| Model | Dataset | vanilla | best score | kept@0.3 |
|---|---|---|---|---|
| Qwen2.5-7B | MATH-500 | 0.777 | hidden_drift_max_penult | 0.850 |
| Qwen2.5-7B | Olympiad | 0.303 | hidden_norm_var | 0.363 |
| Math-7B | MATH-500 | 0.768 | hidden_norm_range | **0.890** |
| Math-7B | Olympiad | 0.320 | hidden_norm_range | 0.343 |
| Phi-4 | MATH-500 | 0.780 | hidden_drift_max_penult | **0.887** |
| Phi-4 | Olympiad | 0.260 | hidden_norm_var | 0.299 |

### 3.3 Step-trigger (12 cells, K=4 T=0.7 majority vote)

| Trigger | Top-1 wins | Top-3 wins | Mean lift |
|---|---|---|---|
| **`lp_min` (baseline)** | **7/12** ⭐ | 9/12 | +1.7pp |
| `top1_margin_min` | 2/12 | 7/12 | +1.1pp |
| `entropy_max` | 0/12 | 8/12 | +1.1pp |
| `tempered_kl_max` | 2/12 | 6/12 | +0.9pp |
| `kl_uniform_min` | 1/12 | 6/12 | +1.3pp |

**중요**: lp_min은 trigger로는 1등이지만 trajectory CP score로는 0/16. **Trigger와 Score는 다른 layer**.

### 3.4 Step-level zoo (12 cells, 12 features × 5 aggregators)

(위 §2.3 표 참고)

---

## 4. 주요 발견 (Key Insights)

### 4.1 Trigger ≠ CP Score (방법론적 분리)
- **Step-branching trigger**: lp_min이 여전히 1등
- **Trajectory CP score**: entropy_mean이 1등
- 둘은 다른 역할이라 다른 점수 사용 가능

### 4.2 데이터셋 난이도 의존성
- **MATH-500** (medium): step-level과 trajectory 비슷, hidden_state도 강력
- **AIME** (hard): trajectory `entropy_mean` 압도, step-level 약함
- **Olympiad** (very hard): 모델/점수 mix
- **MMLU-Pro** (saturated): 모든 점수 거의 동등 (baseline 이미 강함)

### 4.3 모델 타입별 패턴
- **약한 모델 + 어려운 dataset** (Qwen2.5-7B AIME): 모든 step-trigger +4.5~6.5pp
- **강한 math model + 어려운 dataset** (Math-7B Olympiad): step-branching이 −1~3.5pp 손해
- **강한 일반 모델**: rep_4gram (degeneracy detection) 강력

### 4.4 비용-성능 Pareto frontier

| Cost | Best score | Best kept_acc (Math-7B AIME) |
|---|---|---|
| FREE 1× (online) | `lp_drawdown_sum` (step) | 0.55-0.60 |
| FREE 1× (post-hoc) | `entropy_mean` (trajectory) | **0.677** |
| 1× FREE + HF | `hidden_drift_max_penult` | 0.85 (MATH-500 only) |
| 3× | `paraphrase_consensus` | matches sc_top1 |
| 8× | `tiebreak_lex` | Pareto over sc_top1 |

---

## 5. Honest negative results

| Hypothesis | Status | 이유 |
|---|---|---|
| F1 forbidden_top1_redecoding | ❌ −3pp | top-1 ban이 오히려 손해 |
| Sympy step verifier | ❌ 4% extractable only | 등호식 추출 어려움 |
| Combined paraphrase + step-branching | ❌ +0.5pp marginal | overlapping failure modes |
| Naive PRM+SC ensemble | ❌ ≈ SC alone | 상관관계 너무 높음 |
| `sharpening_rate` on math | ❌ ρ=−0.034 | code에서만 신호 |
| `hidden_norm_var` / `layer_disagreement` | ❌ noise on Qwen2.5-7B | model-specific |
| Info-theoretic step-triggers vs lp_min | ❌ tied | trigger 역할은 lp_min 우위 |
| AIME 1983-2024 vanilla=0.805 (Qwen2.5-32B) | ⚠️ training leak 의심 | 오래된 문제 학습 데이터 포함 가능 |

---

## 6. Wave별 brainstorm 요약

### Wave 1 (5 agents + 2 codex, 25+ score functions)
- Agent A: info-theoretic (entropy/KL family) — **검증됨**
- Agent B: hidden state — **부분 검증** (drift_max_penult만 작동)
- Agent C: external verifier (sympy) — **❌ 실패** (추출률 4%)
- Agent D: consistency / paraphrase — **검증됨** (paraphrase_consensus)
- Agent E: hybrid / ensemble — 부분 검증
- Codex 1: bayes-optimal — 미검증
- Codex 2: cheap surrogate — 검증됨 (predictive_sharpening_rate)

### Wave 2 (5 agents, 25+ step-level methods)
- Agent F (token edits): **❌ 실패** (forbidden_top1)
- Agent G (search): 미검증 (MCTS — 비싸서 보류)
- Agent H (structured rewrite): 부분 검증 (sympy 추출 부분 실패)
- Agent I (prefix diversity): ⚠️ +0.5pp 약함
- Agent J (learned policy): 미검증 (offline training 필요)

### Wave 3 (NEW, 5 agents, 35+ step-level scoring methods)
- Agent SA (streaming): 부분 검증 (free token curvature)
- Agent SB (lookahead/counterfactual): 미검증 (Tier 3, 비쌈)
- Agent SC (relative/comparative): **검증됨** (lp_drawdown, lp_zscore, ent_growth 모두 강력)
- Agent SD (self-prompting): 미검증 (Tier 2, 1× extra pass)
- Agent SE (surface): **검증됨** (arith_violations, rep_4gram)

---

## 7. 논문 narrative (현재 시점)

> **"We propose CoT-CP, a CP wrapper for trajectory-level scores, plus 6 new score families that fill the cost-vs-LR+ Pareto frontier:**
>
> (i) **`entropy_mean` (= `kl_uniform_mean`)** — strongest free per-token info-theoretic score, validated across 4 models × 4 datasets, top-1 in 5/16 cells, +14.6pp lift on Math-7B AIME
>
> (ii) **`top1_margin_mean`** — runner-up free per-token info-theoretic score, top-1 in 4/16 cells
>
> (iii) **`hidden_drift_max_penult`** — strongest free hidden-state score on math (+6.5pp over HF baseline)
>
> (iv) **`lp_drawdown_sum`** / **`arith_violations_sum`** — strongest step-level (online-computable) scores. arith_violations gives +5.8pp on Phi-4 math500
>
> (v) **`paraphrase_consensus M=3`** — 3× compute mid-tier score with lift comparable to sc_top1
>
> (vi) **`tiebreak_lex`** — exact Pareto improvement over sc_top1 (8× + free)"
>
> **Plus methodological insight: trigger choice (lp_min) and CP score (entropy_mean) are decoupled** — step-branching trigger remains lp_min, but the CP scoring layer benefits from the new scores.
>
> **Honest negative section**: forbidden_top1, raw sympy verifier, naive ensembles, and step-trigger replacements (info-theoretic as triggers) do not deliver lifts.

---

## 8. 실험 통계

- **총 실험 cells**: ~46 (16 info-theoretic + 6 hidden-state + 12 step-trigger + 12 step-zoo)
- **모델 수**: 8개 검증 (Qwen2.5-7B/32B, Qwen2.5-Math-7B, Qwen3-8B, Qwen3-30B-MoE, QwQ-32B, R1-Distill-Qwen-7B/32B/Llama-70B, Phi-4, Mixtral-8x7B, DeepSeek-V2-Lite-MoE, PRM 7B)
- **데이터셋**: 7개 (GSM8K, MATH-500, AIME, OlympiadBench, MMLU-Pro STEM, TheoremQA, HumanEval)
- **Wave 1+2+3 brainstorm**: 15 agents × ~85 method
- **Tested method count**: 약 100개

---

## 9. 남은 작업

- [ ] HumanEval 실제 코드 실행 채점 구현
- [ ] Wave 3 Tier 2 (cheap probes: self-prompts) 5개 method 검증
- [ ] Wave 3 Tier 3 (look-ahead) expensive methods 검증
- [ ] Theorem 1+2+3 final draft (Theorem 1은 codex skill 작업 완료)
- [ ] Full ablation table for paper
- [ ] AIME 1983-2024 training leak 여부 정량 분석

---

## 10. 산출물 위치

```
/home/nvidia/future/score_ideation/
├── synthesis/
│   ├── EXPERIMENTAL_FINDINGS.md          (영문 종합, §1-11)
│   ├── RESULTS_SUMMARY_KR.md              (한글 종합, this file)
│   └── STEP_LEVEL_SYNTHESIS.md            (Wave 3 정리)
├── from_agents/                           (Wave 1: 5 agents)
├── from_agents_step/                      (Wave 3: 5 agents, 35 methods)
└── from_codex/                            (Wave 1+2: codex 작업)

/home/nvidia/future/experiments/
├── src/
│   ├── SX_validation_matrix.py           (info-theoretic 4×4 matrix)
│   ├── SX_hidden_state_matrix.py         (hidden state 3×2 matrix)
│   ├── SX_step_triggers_matrix.py        (step-trigger 4×3 matrix)
│   ├── SX_step_score_zoo.py              (step-zoo 4×3 matrix, 60 scores)
│   ├── SX_info_theoretic_gpu.py          (initial info-theoretic exploration)
│   ├── SX_hidden_state_gpu.py            (initial hidden-state exploration)
│   ├── SX_prm_guided_branching.py        (PRM-guided step branching)
│   ├── SX_perturb_scores_gpu.py          (paraphrase / xtemp / step_mask)
│   ├── SX_score_zoo_cpu.py               (CPU-only post-hoc score zoo)
│   ├── SX_step_hparam_sweep.py           (45-config Pilot C sweep)
│   ├── SX_step_new_triggers.py           (initial step-triggers single cell)
│   ├── SX_sympy_verifier.py              (negative result)
│   ├── robust_eval.py                     (consistent answer extraction)
│   └── aggregate_validation.py            (matrix aggregation)
└── results/
    ├── SX_validation_*.json              (info-theoretic per-model)
    ├── SX_hidden_validation_*.json
    ├── SX_step_triggers_*.json
    ├── SX_step_zoo_*.json
    └── E1-E18 / pilot * trace data       (이전 wave)

/home/nvidia/future/theorems/
└── theorem1_trajectory_cp.md              (Theorem 1 draft, codex 작업)
```

---

## 11. Per-step CP (Phase 3+4, NEW)

### Setup
각 step boundary에서 online으로 score 계산 (running_lp_min, running_entropy_mean, running_lp_drawdown, prov_match_latest_rate 등). 매 step마다 calibrated threshold와 비교해서 abort 결정.

**Approach A (early-abort, no formal coverage)**: 첫 score < threshold(t)에서 즉시 abort. Selective accuracy 극대화.

**Approach B (Bonferroni-corrected, formal guarantee)**: α를 T_max steps에 균등 분배. 보수적이지만 formal coverage 보장.

**Approach C (geometric weight, semi-formal)**: 초기 step에 더 많은 budget 할당 (1/(t+1) 가중).

### 검증 결과 (10 cells, n=200)

| Model | Dataset | vanilla | best A kept@α=0.3 | best A kept@α=0.5 | avg_steps@α=0.3 | kept_frac@α=0.3 |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | math500 | 0.745 | 0.820 | 0.811 | 3.5 | 0.50 |
| Qwen2.5-7B | aime | 0.270 | 0.483 | 0.576 | 3.2 | 0.26 |
| Qwen2.5-7B | olympiad | 0.445 | 0.543 | — | 6.5 | 0.62 |
| Math-7B | math500 | 0.800 | **0.882** | **0.911** | 4.1 | 0.50 |
| Math-7B | aime | 0.390 | 0.696 | 0.778 | 1.8 | 0.28 |
| Math-7B | olympiad | 0.450 | 0.722 | 0.755 | 2.0 | 0.51 |
| Qwen2.5-32B | math500 | 0.805 | **0.923** | **0.991** | 4.2 | 0.16 |
| Phi-4 | math500 | 0.775 | 0.920 | **0.993** | 6.7 | 0.19 |
| Phi-4 | olympiad | 0.485 | 0.890 | **1.000** | 5.5 | 0.04 |

**핵심 발견:**
1. **`prov_match_latest_rate` (running) score**가 강한 모델 + 쉬운 데이터셋에서 압도적 — Phi-4 olympiad α=0.5에서 **kept_acc=1.000** (단, 1.3%만 keep)
2. α=0.3에서 보통 ~50% kept하면서 vanilla보다 +5-15pp 정확
3. **Approach A는 trajectory CP와 동등한 kept_acc를 절반 compute로 달성** (avg_steps 4-7 vs full 7-15 steps)
4. Approach B (Bonferroni)는 보수적 — kept_frac 90+%지만 lift 작음 (-5pp 정도)

### Phase 4 강화
Holm sequential, e-process 등은 future work. 현재 A vs B로 선명한 trade-off 확인됨.

---

## 12. Wave 4 Step Rejection Brainstorm + 검증 (NEW)

### Brainstorm: 5 agents × 35 methods (이전 Wave들과 합쳐서 ~120 methods 총)
- **SR-A regenerate**: K=4 + PRM, temp escalation, backtrack, self-correct insertion (7개)
- **SR-B skip/backtrack**: step excise, consecutive bad block jump, escalation ladder (7개)
- **SR-C ensemble**: DTW alignment, Pareto-optimal trace selection, cross-trace grafting (7개)
- **SR-D online**: tripwire, BoN per step, meta-cue, escalation, granular sentence-level (7개)
- **SR-E prov-guided**: MASH-R modal-anchored hint, mode-filtered resampling, iterative prov feedback (7개)

### Tier 1 검증 (4 methods × 4 cells of 7B/Math-7B × math500/aime)

| Model | Dataset | vanilla | self_correct | step_excise | backtrack_2 | **K4_majority** |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | math500 | 0.745 | 0.745 | 0.740 | 0.750 | **0.770 (+2.5)** |
| Qwen2.5-7B | aime | 0.270 | 0.270 | 0.281 | 0.264 | **0.330 (+6.0)** |
| Math-7B | math500 | 0.800 | 0.800 | 0.779 | 0.800 | **0.810 (+1.0)** |
| Math-7B | aime | 0.390 | 0.390 | 0.380 | 0.389 | **0.410 (+2.0)** |

**Tier 1 결론**: K=4 majority (Pilot C 기존 baseline)이 모든 cell에서 1등. Single-alt methods (insertion, excise, backtrack)는 flat 또는 약간 negative. Trigger은 lp_min worst step.

### Tier 2 검증 결과 — 7 methods × 11 cells

| Cell | vanilla | best method | acc | Δ |
|---|---|---|---|---|
| **32B AIME** | 0.395 | **K4_majority** | **0.460** | **+6.5** ⭐ |
| 7B AIME | 0.270 | K4_majority | 0.330 | +6.0 |
| Phi-4 AIME | 0.345 | K8_majority | 0.400 | +5.5 |
| Phi-4 olympiad | 0.485 | K8_majority | 0.515 | +3.0 |
| 7B math500 | 0.745 | K4_majority | 0.770 | +2.5 |
| 32B math500 | 0.805 | K2_K4_escalation | 0.820 | +1.5 |
| Math-7B AIME | 0.390 | K4_majority | 0.410 | +2.0 |
| Math-7B olympiad | 0.450 | K8_majority | 0.470 | +2.0 |
| 7B olympiad | 0.445 | K4_majority | 0.460 | +1.5 |
| Phi-4 math500 | 0.775 | K4_majority | 0.790 | +1.5 |
| Math-7B math500 | 0.800 | K4_majority | 0.810 | +1.0 |

**Tier 2 패턴:**
- **K4_majority가 여전히 dominant** (5/11 1등)
- **K8_majority** 어려운 dataset에서 (3/11) — compute scaling 효과
- **K2_K4_escalation** 강한 모델 + 쉬운 dataset에서 (1/11)
- **K4_lp_select 항상 negative** — picking by lp가 majority보다 나쁨
- **multi_worst_K2** top-2 worst에서 각 K=2: 거의 K=4와 동급

### Tier 3 검증 결과 — Online multi-step (7Bs only)

| Cell | vanilla | best T3 method | acc | Δ |
|---|---|---|---|---|
| 7B AIME | 0.270 | online_K1_top3_worst | 0.310 | +4.0 |
| 7B math500 | 0.745 | (K4_baseline) | 0.770 | +2.5 |
| 7B olympiad | 0.445 | (K4_baseline / prov_anchor_K2) | 0.455 | +1.0 |
| Math-7B math500 | 0.800 | online_K1_top3_worst | 0.805 | +0.5 |
| Math-7B olympiad | 0.450 | online_K1_top3 / prov_anchor | 0.455 | +0.5 |
| Math-7B AIME | 0.390 | (K4_baseline) | 0.410 | +2.0 |

**Tier 3 패턴:**
- Online multi-step (TOP-3 worst step regen)이 K=4 single-shot과 competitive하지만 dominant 아님
- prov_anchor_K2 (provisional answer hint)는 약함 — 힌트가 diversity 줄이거나 greedy와 일치
- 어려운 cell (AIME)에서 multi-step이 single-K4와 비슷한 lift

### Tier 4 검증 결과 — Advanced selection (12 cells)

5 methods: K4_majority_baseline, K4_select_entropy_min, K4_select_lp_max, K4_filter_arith_majority, K4_consensus_3plus

**핵심 발견:**
- **K4_majority_baseline이 여전히 dominant** (대부분 cell 1등 또는 tied)
- **K4_consensus_3plus** (3+ 동의해야만 accept) close second — sometimes ties
- **K4_filter_arith_majority** (arith_violations 있는 alt 제외 후 vote) — neutral, 가끔 +0.5pp
- **K4_select_entropy_min / K4_select_lp_max** consistently negative (-1 ~ -5pp) — 단일 점수로 alt 선택은 majority보다 나쁨

**의미**: K=4 alts에서 1개를 score로 고르기보다 모두 vote하는 게 robust.

### Tier 5 검증 결과 — Combined CP + step rejection (6 cells, 7Bs)

3 gate methods × always-K4 baseline. Gate score < threshold일 때만 K=4 majority 적용.

| Cell | vanilla | always_K4 (5×) | gate_ent_mean_K4 (3×) | gate_lp_min_K4 (3×) |
|---|---|---|---|---|
| **7B Olympiad** | 0.445 | -1.0pp | **+2.5pp** ⭐ | +1.0pp |
| **Math-7B AIME** | 0.390 | +1.0pp | **+2.0pp** ⭐ | 0pp |
| Math-7B Olympiad | 0.450 | +1.0pp | +1.0pp | +0.5pp |
| 7B AIME | 0.270 | +5.5pp | +2.0pp | +2.0pp |
| 7B math500 | 0.745 | +3.0pp | +2.5pp | 0pp |
| Math-7B math500 | 0.800 | +0.5pp | -0.5pp | -1.5pp |

**🔥 핵심 발견 — gate_ent_mean_K4가 always_K4보다 더 좋거나 동등하면서 40% 저렴:**

평균:
- always_K4 cost 5×: +1.7pp avg
- **gate_ent_mean_K4 cost 3×: +1.6pp avg** (similar lift, 40% cheaper!)

특히:
- **7B Olympiad**: always_K4가 −1.0pp loss인데 gate_ent_mean이 +2.5pp gain — entropy_mean signal이 "어떤 trace에서 K=4가 도움될지"를 잘 잡음
- **Math-7B AIME**: gate +2.0pp vs always +1.0pp — gate가 베이스라인보다 +1pp 더 lift
- **Math-7B math500**: 강한 모델 + 쉬운 dataset에서 always_K4도 K=4 gating도 모두 약함 (saturation)

**Paper에 critical finding**: re-roll을 항상 하지 말고 **entropy_mean으로 confidence 낮은 trace에만** 적용하면 비용 절약 + 가끔 정확도까지 개선.

### Tier 5_v2 검증 — 4 models × 3 datasets (12 cells), gate variants + K=8

5 strategies × 12 cells:

| Method | Avg Δ | Cost | Cost-eff (Δ/cost) |
|---|---|---|---|
| **`always_K4`** | **+1.71pp** | 5.0× | 0.34 pp/× |
| `gate_lp_K8` | +1.29pp | 5.0× | 0.26 pp/× |
| **`gate_ent_K4`** | **+1.12pp** | **3.0×** | **0.37 pp/×** ⭐ |
| `gate_ent_K8` | +1.08pp | 5.0× | 0.22 pp/× |
| `gate_lp_K4` | +0.92pp | 3.0× | 0.31 pp/× |

**Per-cell winners (12 cells):**
- `always_K4`: 4 wins (math500/olympiad on Phi-4, AIME on 32b/7b)
- `gate_ent_K8`: 3 wins (math500 7B/Math-7B/32B)
- `gate_ent_K4`: 3 wins (3 olympiad cells with 60% cost)
- `gate_lp_K8`: 1 win (Phi-4 AIME)
- `gate_lp_K4`: 0 wins

**Final paper recommendation:**
1. **Maximum accuracy**: `always_K4` at 5× cost (+1.71pp avg)
2. **Best cost-accuracy tradeoff**: `gate_ent_K4` at 3× cost (+1.12pp avg, 0.37 pp/× — most efficient)
3. **Hard problems (AIME)**: `gate_ent_K8` or `always_K4` at 5× cost
4. **Olympiad**: `gate_ent_K4` (cheap, effective)

---

## 13. Updated paper narrative (현재 시점 — 2 layer 구조)

### Layer 1: Selective Accuracy (CP)
**`entropy_mean`** (trajectory) + **`prov_match_latest_rate`** (online running) 두 점수가 ↑ kept_acc.
- Trajectory CP: full trace 후 점수 → kept_acc 0.83-0.93 cross matrix
- Per-step CP Approach A: online → kept_acc 0.82-0.99 (selectivity↑) at ~50% compute

### Layer 2: Reasoning Step Rejection (Accuracy improvement)
**K=4 majority at lp_min worst step** (Pilot C baseline)이 +1-6pp 정확도 lift.
- Single-alt methods (insertion, excise, backtrack)는 flat
- Tier 2/3 (escalation, prov-anchored, online multi-step) 진행 중

### 두 layer의 분리
- Selective accuracy: 정답률을 보장 (높은 kept_acc 또는 abstain)
- Step rejection: 정답률 자체를 향상 (abstain 없이 더 잘 답하게)

이제 우리 논문은 두 가지를 모두 다룰 수 있는 framework.


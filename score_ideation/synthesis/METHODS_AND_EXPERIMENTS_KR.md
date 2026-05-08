# CoT-CP 방법론 + 실험 종합 가이드 (한글)

> 작성일: 2026-05-08
> 대상: 우리가 지금까지 무엇을 했고, 무엇을 발견했는지 쉽게 이해하기 위한 문서

---

## 0. 한 문단 요약

LLM이 수학 문제를 풀 때 **그 답을 믿어도 되는지** 판단하는 점수를 만들고, 점수가 낮으면 **답을 거절하거나 다시 풀게 하는** 시스템. 답한 것 중 정답률은 보장(conformal prediction)하고, 동시에 전체 정답률도 향상(step rejection). 두 layer는 다른 점수를 사용하면 서로 보완.

---

## 1. 핵심 framework (한 그림)

```
질문 Q  ──→  LLM이 Chain-of-Thought 생성  ──→  최종 답
                  │
                  ↓
            [매 step 끝마다 점수 계산]
            entropy, lp, prov_match, ...
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
   Layer A (CP)       Layer B (Step Rejection)
   "답할까?"            "다시 해볼까?"
        ↓                   ↓
   keep / abstain       K=4 alts → vote
```

**Layer A (Conformal Prediction)**: 정답률 보장. 점수 낮으면 답 거절.  
**Layer B (Step Rejection)**: 정확도 향상. 점수 낮은 step에서 K=4번 다시 샘플링 후 다수결.

---

## 2. 데이터셋 + 모델

### 데이터셋 (4개 종류)

| 이름 | 크기 | 난이도 | 설명 |
|---|---|---|---|
| **MATH-500** | 500 | 중간 | High school math, MATH 벤치 일부 |
| **AIME 1983-2024** | 933 | 어려움 | 미국 수학 올림피아드 예선 |
| **OlympiadBench** | 674 | 매우 어려움 | 국제 올림피아드 수학 |
| MMLU-Pro STEM | 3000+ | 객관식 | 과학/공학 객관식 |

### 모델 (4개 핵심 + 5개 추가)

핵심 (모든 실험 진행):
- **Qwen2.5-7B-Instruct** (general)
- **Qwen2.5-Math-7B-Instruct** (math 특화)
- **Qwen2.5-32B-Instruct** (general 큰)
- **microsoft/phi-4** (14B, reasoning 특화)

추가 (일부 실험만):
- Qwen3-8B/30B-MoE, QwQ-32B, R1-Distill-Qwen-7B/32B/Llama-70B, Mixtral-8x7B, DeepSeek-V2-Lite-MoE

---

## 3. Score Family 1 — Trajectory-level CP scores

**개념**: trace 전체를 본 뒤 한 개의 숫자로 점수 매김. 그 점수 분포에서 lower α-quantile을 threshold로 calibration → test 시 점수 < threshold면 abstain.

### 검증한 점수들 (16 cells = 4 모델 × 4 datasets)

| 점수 | 정의 | 비용 | Top-1 wins (16) |
|---|---|---|---|
| **`entropy_mean`** | 모든 토큰의 평균 Shannon entropy | 1× FREE | **5/16 ⭐** |
| `top1_margin_mean` | 모든 토큰의 (top1 logprob − top2 logprob) 평균 | 1× FREE | 4/16 |
| `entropy_max` | step별 max entropy | 1× FREE | 3/16 |
| `tempered_kl_max` | T=1.0과 T=2.0 logits 사이 JS divergence max | 1× FREE | 2/16 |
| `lp_min` (baseline) | step별 평균 logprob 중 최솟값 | 1× FREE | **0/16 ❌** |

**결과**: `entropy_mean`이 universal winner. 16개 cells 모두에서 lp_min을 strict하게 dominate. 최대 lift +14.6pp on Math-7B AIME.

### 비싼 점수들도 검증

| 점수 | 비용 | 효과 |
|---|---|---|
| `paraphrase_consensus M=3` | 3× | MATH-500 92.7%, AIME 79.6% (sc_top1과 비슷) |
| `tiebreak_lex` (sc_top1+lp_min) | 8× + free | AIME +5.7pp Pareto improvement |
| `prm_min` (Process Reward Model) | 2× | sc_top1과 비슷 |

### 실패한 점수들 (negative results, 정직성)

| 점수 | 결과 |
|---|---|
| `predictive_sharpening_rate` (entropy 변화율) | math에선 ρ=−0.034 (uselss). HumanEval(code)에서만 유효 |
| Hidden state-based (`hidden_drift`, `hidden_norm_var`) | MATH-500 OK, Olympiad에서 −20~33pp 무너짐 |
| Sympy verifier (텍스트에서 등호식 추출) | 4%만 추출 가능 — useless |
| `forbidden_top1` (top-1 토큰 강제 금지) | −3pp 손해 |

---

## 4. Score Family 2 — Per-step CP scores

**개념**: trace 끝까지 안 기다리고, 매 step boundary에서 점수 계산. 점수 < threshold이면 즉시 abort.

### 3가지 구현 방식

**Approach A** (early-abort, no formal guarantee):
- 각 step 위치 t별로 calibration → threshold q_α(t)
- 첫 score < threshold(t) 시점에 즉시 멈춤
- 결과: kept_acc 0.82-1.00 at avg ~50% compute (full trace 안 기다려도 됨)

**Approach B** (Bonferroni, formal coverage 보장):
- α budget을 T_max steps에 균등 분배 (α_t = α/T_max)
- 보수적 — kept_frac 90+%지만 lift 작음

**Approach C** (geometric weighting):
- 초반 step에 더 많은 budget 할당 (1/(t+1) 가중)
- A와 B의 중간

### 사용한 step-level 점수들

기본:
- `running_lp_min`: prefix까지의 lp 중 최솟값 (online updateable)
- `running_entropy_mean`: prefix까지 평균 entropy
- `running_lp_drawdown`: peak-to-current lp drawdown

특별: **`prov_match_latest_rate`** (provisional answer-based)
- 각 step boundary에서 모델에게 "지금 답 뭐야?" 5-10 token 생성으로 물어봄
- 그 답들이 마지막 prov 답과 얼마나 자주 일치하는지 비율

### 결과 (10 cells)

| Cell | vanilla | Approach A best | kept_acc | avg_steps | kept_frac |
|---|---|---|---|---|---|
| **Phi-4 olympiad α=0.5** | 0.485 | `prov_match_latest_rate` | **1.000** | 3.6 | 1.3% |
| **Phi-4 math500 α=0.5** | 0.775 | `prov_match_latest_rate` | **0.993** | 3.7 | 6.9% |
| Math-7B math500 α=0.3 | 0.800 | `prov_match_latest_rate` | 0.862 | 6.9 | 50% |
| 32B math500 α=0.3 | 0.805 | `running_lp_min` | 0.891 | 7.9 | 56% |

**해석**: 강한 모델 + 큰 α(엄격) → kept_acc 거의 1, 하지만 kept_frac 작음 (selectivity 큼). 약한 모델 + 작은 α → kept_frac 높지만 kept_acc 약간만 향상.

---

## 5. Score Family 3 — Hidden state scores (model 내부 표현)

**개념**: HF transformers + `output_hidden_states=True`로 각 토큰의 hidden state 추출. step boundary 사이의 hidden state 변화를 점수화.

### 검증한 점수들

| 점수 | 정의 | 결과 |
|---|---|---|
| **`hidden_drift_max_penult`** | 인접 step end의 penultimate layer hidden state 사이 cosine distance의 max | MATH-500 +6.5pp, Olympiad에서 무너짐 |
| `hidden_drift_max_last` | 위 + last layer | Math-7B에서 ρ=+0.354 (강함) |
| `hidden_norm_range` | 각 step end hidden state의 norm 범위 | Math-7B math500 kept=0.890 ⭐ |
| `hidden_norm_var` | norm 분산 | 보통 |
| `layer_disagreement_*` | last 4 layers의 logit-lens distribution 사이 JS | noise (대부분 weak) |

### 결과 (6 cells: 3 models × 2 datasets, n=100)

- **Math-7B math500**: `hidden_norm_range` kept=**0.890** (model-specific 강함)
- **Phi-4 math500**: `hidden_drift_max_penult` kept=**0.887**
- **Olympiad/AIME**: 거의 모든 점수 noise — hidden state는 어려운 math에서 무너짐

**해석**: HF transformers 필요해서 vLLM보다 25배 느림. 게다가 어려운 dataset에서 무너지므로 paper의 메인은 아님. 하지만 MATH-500에서는 info-theoretic과 동급.

---

## 6. Step-level scoring (online에서 매 step마다 계산하는 점수)

trace 끝까지 안 기다리고 step boundary에서 즉시 계산. Layer A (CP)에서 사용 가능.

### 12개 step-level features

Wave 3 brainstorm 5 agents → 35개 method 제안 → 12개 implement.

| Family | 점수 예시 | 계산 |
|---|---|---|
| **`ent_growth`** ⭐ | log(entropy_t / running_geomean) | entropy 증가율 |
| **`arith_violations`** ⭐ | 등호식 추출 후 safe_eval failure 개수 | 산술 검증 |
| **`lp_drawdown`** ⭐ | running_max(lp) − lp_t | peak-to-current |
| `lp_zscore` | running Welford z-score of step lp | 표준화 |
| `lp_rank_q` | 현재 step lp의 prefix내 quantile | 순위 |
| `rep_4gram` | 이전 step과 4-gram 중복 비율 | degeneracy |
| `num_density` | 숫자+연산자 / 토큰 수 | 수학적 밀도 |
| `neighbor_js` | 인접 step token distrib JS | 분포 변화 |
| `lp_jump`, `token_curvature`, `backtrack`, `branch` | 추가 | 미세 신호 |

### Aggregation
12 features × 4-5 aggregators (max/min/mean/last/sum) = 48-60 trajectory-level features.

### 결과 (12 cells, n=200)

Top-3 hits per family:
- **`ent_growth`** (entropy 증가율): **8/12 cells**
- **`arith_violations`** (등호식 검증): 7/12
- **`lp_drawdown`**: 7/12

**Best cells:**
- 32B math500: `rep_4gram_sum` kept=**0.907** (32B repeats text from prev step → bad sign)
- Phi-4 math500: `arith_violations_sum` kept=**0.861** (+5.8pp over trajectory winners)
- Phi-4 olympiad: `lp_drawdown_sum` kept=0.623

**핵심**: step-level 점수가 **trajectory-level 점수와 경쟁할 수 있음** (4/12 outright win, 4/12 tie).

---

## 7. Step Rejection (Layer B) — 정확도 향상

**개념**: 점수 낮은 step에서 K개 alternative 생성 → 다수결 vote.

### Tier 1: Single-alt 시도 (실패)

기준 trigger: `lp_min` worst step (이전 검증).

| 방법 | 결과 |
|---|---|
| **`K4_majority`** (Pilot C, baseline) | **+1-6pp** ⭐ |
| `self_correct_insertion` ("Wait, let me reconsider") | 0pp |
| `step_excise` (bad step 삭제 후 regen) | −0.5~2pp |
| `backtrack_2step` (t-1+t 자르고 regen) | 0~+0.5pp |

→ K=4 majority가 simple하게 가장 robust. 새 single-alt method는 다 약함.

### Tier 2: K=4 변형 (compute scaling vs alternative selection)

| 방법 | 결과 |
|---|---|
| **`K8_majority`** | 어려운 dataset에서 +5-8pp (7B AIME +8pp) ⭐ |
| `K4_T1` (T=1.0 더 다양) | tied 또는 약간 손해 |
| `K2_K4_escalation` (K=2 cheap → 동의 시 accept, 아니면 K=4) | 1.5× cheaper, similar lift |
| `K4_majority_entropy_trigger` (lp_min 대신 entropy_max로 worst step) | weakly better |
| **`K4_lp_select`** (K=4 alts에서 lp 가장 높은 거 선택) | **−1~5pp ❌** |

→ K=8 가치 있음. lp_max selection은 majority보다 항상 나쁨.

### Tier 3: Online multi-step (top-3 worst steps에서 각각 K=2)

| 방법 | 결과 |
|---|---|
| `online_K1_top3_worst` (top-3 worst step에 K=1씩) | K=4와 비슷 |
| `online_K2_BoN_top3_worst` | K=4와 비슷 |
| `prov_anchor_K2` (provisional 답 hint로 K=2) | weak |

→ multi-step 분배가 단일 K=4와 비슷한 효과.

### Tier 4: Advanced selection (실패)

| 방법 | 결과 |
|---|---|
| `K4_select_entropy_min` | −1~5pp |
| `K4_select_lp_max` | −1~5pp |
| `K4_filter_arith_majority` (arith 위반 alt 제외 후 majority) | tied |
| `K4_consensus_3plus` (3+ 동의해야만 accept, 아니면 greedy) | tied |

→ K=4 alts에서 단일 점수로 1개 고르기보다 모두 vote가 robust.

### Tier 5: Combined CP + step rejection (gating)

핵심 아이디어: trajectory CP 점수로 trace 별 confidence 측정 → 자신없는 trace에만 K=4 적용.

| 방법 | 비용 | 결과 (12 cells avg) |
|---|---|---|
| **`gate_combined_f0.5_K4`** (entropy ∨ lp ∨ arith로 50% flag, K=4) | 4.46× | **+1.92pp ⭐ best balance** |
| `gate_ent_f0.5_K4` (entropy로 50% flag, K=4) | 3× | +1.29pp (cost-eff) |
| `always_K4` | 5× | +2.08pp |
| `gate_lp_f0.5_K4` | 3× | +0.92pp |

→ **gate_combined가 cheapest path to similar lift**. always_K4보다 11% 저렴, lift도 비슷.

### 셀별 best lift (12 cells)

| Model + Dataset | Best method | Δ | Cost |
|---|---|---|---|
| 7B AIME | always_K8 | **+8.0pp** | 9× |
| 32B AIME | gate_combined_f0.25_K8 | +7.5pp | 7× |
| Math-7B AIME | gate_ent_f0.75_K8 | +5.0pp | 7× |
| 7B math500 | always_K4 | +4.0pp | 5× |
| 32B Olympiad | gate_ent_f0.5_K8 | +3.5pp | 5× |
| Phi-4 olympiad | gate_lp_f0.25_K8 | +2.5pp | 3× |

→ 어려운 dataset (AIME)에서 K=8 가치. 쉬운 dataset에서 K=2-4 충분.

---

## 8. Pareto Frontier (cost-accuracy curve)

3 gate scores × 3 fractions × 3 K values + 3 always_K = **30 strategies × 12 cells = 360 evaluations**

### Pareto-optimal points (avg over 12 cells)

| Cost | Δ acc | 추천 사용 |
|---|---|---|
| 1.0× | 0pp | vanilla baseline |
| 1.5× | +0.37pp | `gate_ent_f0.25_K2` (cheapest) |
| 2.43× | +0.87pp | `gate_combined_f0.25_K2` |
| 2.73× | +1.17pp | `gate_combined_f0.5_K2` |
| **3.0×** | **+1.37pp** | **`always_K2` (knee)** |
| 4.0× | +1.54pp | `gate_ent_f0.75_K4` |
| **4.46×** | **+1.92pp** | **`gate_combined_f0.5_K4` ⭐ best balance** |
| **5.0×** | **+2.08pp** | **`always_K4` (Pilot C)** |
| 7.92× | +2.12pp | `gate_combined_f0.5_K8` |
| 8.62× | +2.17pp | `gate_combined_f0.75_K8` (max) |

**Cost-tier 추천:**
- 저비용 deployment (2-3×): `always_K2` 또는 `gate_ent_f0.5_K4`
- 균형 (4-5×): **`gate_combined_f0.5_K4`** 또는 `always_K4`  
- 최대 정확도 (8-9×): `gate_combined_f0.5_K8`
- AIME-specific: K=8 필수

---

## 9. 두 Layer 간 핵심 발견 — Trigger ≠ Score

Step rejection은 두 결정으로 분리:
1. **Trigger**: 어느 step에서 K개 다시 해볼지 결정. → `lp_min` (이전 점수)이 가장 robust.
2. **Score**: 어떤 trace를 keep / abstain할지 결정. → `entropy_mean` (새 점수)이 strict 우세.

→ **두 layer는 다른 점수 사용 가능**. 따로 calibration.

---

## 10. 종합 표 — 검증된 모든 점수 패밀리

| Layer | 점수 | 비용 | Best lift |
|---|---|---|---|
| Trajectory CP | **`entropy_mean`** | 1× FREE | +14.6pp (Math-7B AIME) |
| Trajectory CP | `top1_margin_mean` | 1× FREE | +9pp (7B Olympiad) |
| Per-step CP | `running_entropy_mean` | 1× FREE (online) | kept_acc 0.82-0.92 at 50% compute |
| Per-step CP | `prov_match_latest_rate` | 1.1× (cheap probes) | kept_acc=1.0 selective |
| Hidden state | `hidden_drift_max_penult` | 25× (HF only) | +6.5pp on MATH-500 |
| Step-level (online) | `arith_violations_sum` | 1× FREE | +5.8pp on Phi-4 math500 |
| Step-level (online) | `lp_drawdown_sum` | 1× FREE | +1.9pp on Phi-4 olympiad |
| Step Rejection | `K4_majority` (Pilot C) | 5× | +1-8pp |
| Step Rejection | `K8_majority` | 9× | +5-8pp on AIME |
| Step Rejection | `gate_combined_f0.5_K4` ⭐ | 4.46× | +1.92pp avg |
| Step Rejection | `paraphrase_consensus M=3` | 3× | matches sc_top1 |

---

## 11. 정직한 Negative Results (실패 사례)

이건 paper에 정직하게 적어야 할 것들:

| Method | 결과 |
|---|---|
| `lp_min` as trajectory CP score | 16/16 cells에서 entropy_mean에 패배 |
| `predictive_sharpening_rate` on math | ρ=−0.034 (random noise level). HumanEval에서만 통함 |
| Hidden state on Olympiad | info-theoretic 대비 −20~33pp 무너짐 |
| Sympy step verifier | 등호식 4%만 추출 — practical하지 않음 |
| `forbidden_top1` (Wave 2 F) | −3pp 손해 |
| `system_prompt_modal` (Wave 2 I) | weak +0.5pp |
| Single-alt step rejection (insertion/excise/backtrack) | 0pp 또는 −2pp |
| `K4_select_entropy_min` / `K4_select_lp_max` | −1~5pp (single-score selection이 majority보다 나쁨) |
| `prov_anchor_K2` (provisional hint K=2) | weak +0~0.5pp |
| Naive PRM+SC ensemble | ≈ SC alone |

---

## 12. 실험 통계 (지금까지)

| Phase | 내용 | Cells 수 |
|---|---|---|
| Phase 1+2 | Provisional answer + traj-diff scores | 12 |
| Phase 3+4 | Per-step CP (Approach A/B/C) | 10 |
| Tier 1 step rejection | Single-alt methods | 6 |
| Tier 2 | K=4 variants + K=8 + escalation | 11 |
| Tier 3 | Online multi-step (top-3 worst) | 6 |
| Tier 4 | Advanced selection | 12 |
| Tier 5 + v2 | Combined CP + rejection | 18 |
| **Pareto frontier** | **30 strategies × 12 cells** | **360** |
| Info-theoretic validation | 4 models × 4 datasets | 16 |
| Hidden state validation | 3 models × 2 datasets | 6 |
| Step trigger matrix | 4 models × 3 datasets | 12 |
| Step zoo (12 features) | 4 models × 3 datasets | 12 |

**총: ~500+ (model, dataset, strategy) 셀 평가**

**Brainstorm**: 6 agents × ~7 methods = **42 step rejection methods 제안**, 그 외 Wave 1+2+3에서 ~85 score functions 제안. 합계 **~120 method 제안**.

---

## 13. 진행 중인 실험 (현재)

**Full-dataset Pareto re-run**: NQ_CAP=500 (math500 full=500, AIME cap=500, Olympiad cap=500)
- Phase A: Qwen2.5-7B + Math-7B 병렬 × 3 datasets
- Phase B: Qwen2.5-32B + Phi-4 병렬 × 3 datasets
- 6 strategies per cell (vanilla + K=2/4/8 majority + gate_combined + gate_ent)
- ETA: ~12-15 시간

목적: 기존 n=200 결과를 n=500으로 re-validate. CI 좁히기. Paper-quality data.

---

## 14. Paper narrative (현재 시점 결론)

> **CoT-CP**: a 2-layer framework wrapping LLM Chain-of-Thought reasoning.
>
> **Layer A (Selective Accuracy)**: Conformal prediction with **`entropy_mean`** (trajectory) or **`prov_match_latest_rate`** (per-step) as the nonconformity score. Validated across 4 models × 4 datasets. **`lp_min` baseline strictly dominated 16/16 cells** by `entropy_mean`. Per-step CP achieves equivalent kept_acc at ~50% compute.
>
> **Layer B (Accuracy Improvement)**: Step rejection at low-confidence step. **`K=4 majority` (Pilot C)** is the simple, robust baseline (+1-8pp lift). **`gate_combined_f0.5_K4`** is the Pareto-optimal balance (4.46× cost, +1.92pp avg). For hard datasets (AIME), K=8 valuable (+5-8pp).
>
> **Methodological insight**: trigger (where to re-roll) and CP score (whether to accept) are **decoupled** — different scores can be used at each layer.

**6+ score families validated** for paper headlines:
- `entropy_mean` (trajectory CP)
- `top1_margin_mean` (runner-up)
- `prov_match_latest_rate` (per-step CP)
- `hidden_drift_max_penult` (HF only, MATH-500 specialist)
- `gate_combined_f0.5_K4` (step rejection cost-balance)
- `K8_majority` (hard problems)

Plus **honest negative section**: `lp_min` baseline, `forbidden_top1`, sympy verifier, hidden state on Olympiad, `K4_select_*` methods don't work.

---

## 15. 산출물 위치

```
/home/nvidia/future/score_ideation/synthesis/
├── METHODS_AND_EXPERIMENTS_KR.md   ← 이 문서 (가이드)
├── RESULTS_SUMMARY_KR.md            (종합 한글 — §11-14 최신)
├── OVERNIGHT_SUMMARY_KR.md          (야간 작업 정리)
├── EXPERIMENTAL_FINDINGS.md         (영문 종합)
├── STEP_LEVEL_SYNTHESIS.md          (Wave 3 정리)
└── STEP_REJECTION_SYNTHESIS.md      (Wave 4 정리)

/home/nvidia/future/score_ideation/from_agents_step/  (브레인스토밍)
└── agent_SR_{A-F}_*.md (6개)

/home/nvidia/future/experiments/
├── src/SX_*.py                      (25 experiment scripts)
├── results/SX_*.json                (119 result files)
├── results/PARETO_FRONTIER.json     (Pareto-optimal points)
└── ...

/home/nvidia/future/theorems/
└── theorem1_trajectory_cp.md
```

**Git**: `jhtwosun/future_work` main branch — 모든 commit push 완료.

---

## 16. 추후 작업 (남은 것들)

1. **현재 실행 중 (full-dataset Pareto)**: 결과 들어오면 paper-quality table
2. **HumanEval 코드 채점**: unit test pass로 정답률 측정 (현재 placeholder)
3. **Theorem 1+2+3 final draft**: codex 작업 진행 중
4. **Pareto plot 생성**: figure for paper
5. **R1-Distill, QwQ 등 추가 모델**: Pareto 확장
6. **PRM-guided cross-model**: 현재 Qwen2.5-7B에서만 검증, 4 모델로 확장

---

**이 문서는 우리 실험을 한 곳에서 보기 위한 가이드. 자세한 수치는 RESULTS_SUMMARY_KR.md, 코드는 experiments/src/.**

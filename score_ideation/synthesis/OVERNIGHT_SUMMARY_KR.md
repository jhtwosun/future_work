# CoT-CP 야간 실험 종합 요약 (2026-05-07)

> 사용자 부재 중 진행한 실험 + brainstorm 종합. 자세한 결과는 RESULTS_SUMMARY_KR.md.

## 한 줄 요약
**Per-step CP + step rejection 모두 검증 완료, Pareto frontier 그렸음. 핵심 finding: `gate_combined_f0.5_K4`가 best balance (4.46× cost, +1.92pp avg)**

---

## 진행한 실험 (수치)
- **6 brainstorm agents** (Wave 4: SR-A/B/C/D/E + Wave 5: SR-F) → **42 step rejection methods 제안**
- **Phase 1+2**: provisional answer + traj-diff scores, **12 cells** (4 모델 × 3 datasets)
- **Phase 3+4**: per-step CP (Approach A/B/C), **10 cells**
- **Tier 1 step rejection** (4 methods × 6 cells) — single-alt
- **Tier 2** (7 methods × 11 cells) — K=4 variants
- **Tier 3** (4 methods × 6 cells, 7Bs) — online multi-step
- **Tier 4** (5 methods × 12 cells) — advanced selection
- **Tier 5** (4 methods × 6 cells, 7Bs) — combined CP+rejection
- **Tier 5_v2** (5 methods × 12 cells) — cross-model gate variants
- **Pareto frontier** (30 strategies × 12 cells = **360 evaluations**)

**총: ~500 (model, dataset, strategy) 셀 평가**

---

## 핵심 Findings

### 1. CP layer (selective accuracy)
- **Trajectory CP `entropy_mean`** kept_acc 0.83-0.94 (이전부터 알려진 결과 재확인)
- **Per-step CP Approach A** (online early-abort): kept_acc 0.82-1.00 at ~50% compute (avg 3-7 steps vs full 7-15)
- **Approach A `prov_match_latest_rate`** (running): Phi-4 olympiad α=0.5 kept_acc=**1.000** (단, kept_frac=1.3%)

### 2. Step rejection layer (accuracy improvement)
**핵심: K=4 majority가 universal baseline. Single-alt (insertion/excise/backtrack)는 flat. Selection (lp_max, entropy_min)은 negative.**

**Pareto frontier (12 cells avg):**
| Method | Δ | Cost | pp/× |
|---|---|---|---|
| gate_combined_f0.75_K8 | +2.17pp | 8.62× | 0.0028 |
| **always_K4** | **+2.08pp** | **5×** | **0.0052** |
| **gate_combined_f0.5_K4** | **+1.92pp** | **4.46×** | **0.0055** ⭐ |
| always_K2 | +1.37pp | 3× | 0.0069 |
| gate_ent_f0.5_K4 | +1.29pp | 3× | 0.0065 |

**Cost tier 추천:**
- **A (2-3×)**: `always_K2` 또는 `gate_ent_f0.5_K4` — +1.3pp
- **B (4-5×)**: **`gate_combined_f0.5_K4` 또는 `always_K4`** — +2.0pp ⭐
- **C (8-9×)**: `gate_combined_f0.5_K8` — +2.2pp (saturating)

### 3. Best per-cell lifts
- 7B AIME: `always_K8` **+8.0pp** (vanilla 0.270 → 0.350)
- 32B AIME: `gate_combined_f0.25_K8` **+7.5pp**
- Math-7B AIME: `gate_ent_f0.75_K8` +5.0pp
- 7B math500: `always_K4` +4.0pp
- 32B Olympiad: `gate_ent_f0.5_K8` +3.5pp

**패턴**:
- AIME (어려운): K=8 필요, +5-8pp lift
- math500 (medium): K=4-5×, +1-4pp
- Olympiad (hard but trace structure better): mid-K=4-8, +2-3.5pp

---

## Honest Negative Results
- **`self_correct_insertion`** ("Wait, let me reconsider"): 0pp
- **`step_excise`** (delete bad step + regen): -0.5 ~ -2pp
- **`backtrack_2step`** (cut t-1+t, regen): 0~+0.5pp
- **`prov_anchor_K2`** (provisional answer hint): 0~+0.5pp (weak)
- **`K4_select_lp_max`**: -1 ~ -5pp (single-score selection beats majority — false)
- **`K4_select_entropy_min`**: -1 ~ -5pp 동일 패턴
- **`gate_lp_*` variants**: gate_ent_*보다 약함 — entropy_mean이 더 좋은 gate 신호

## Paper-level findings (결론)

### CP layer:
- `entropy_mean` (trajectory) + `running_entropy_mean` / `prov_match_latest_rate` (per-step) 두 점수가 best CP scores
- Per-step CP는 trajectory CP와 동등한 selective accuracy를 ~50% compute로 달성

### Step rejection layer:
- **K=4 majority (Pilot C baseline)이 simple하고 robust** — 새 방법들이 못 이김
- **gate_combined_f0.5_K4** (entropy ∨ lp ∨ arith로 flagging, K=4)가 best cost-acc tradeoff
- **K=8** 필요한 곳: AIME 같은 매우 어려운 문제

### 두 layer 결합:
- CP score (entropy_mean)와 step rejection trigger (lp_min)는 **분리된 역할**
- gate_ent_K4: gate trigger로 entropy_mean 사용 (CP layer score), regen은 K=4 majority (rejection layer)

---

## 산출물 위치 (이번 야간 추가)
```
/home/nvidia/future/experiments/src/
├── SX_provisional_answer.py        (Phase 1+2: prov answer + traj-diff)
├── SX_per_step_cp.py                (Phase 3+4: per-step CP A/B/C)
├── SX_step_rejection.py             (Tier 1)
├── SX_step_rejection_t2.py          (Tier 2)
├── SX_step_rejection_t3_online.py   (Tier 3 online multi-step)
├── SX_step_rejection_t4.py          (Tier 4 selection variants)
├── SX_step_rejection_t5_combined.py (Tier 5 combined)
├── SX_step_rejection_t5_v2.py       (Tier 5_v2 cross-model + K=8)
├── SX_pareto_gate_K.py              (Pareto frontier 360 evaluations)
├── aggregate_step_rejection.py      (basic aggregator)
└── aggregate_all_step_rejection.py  (multi-tier aggregator)

/home/nvidia/future/score_ideation/from_agents_step/
├── agent_SR_A_regenerate.md
├── agent_SR_B_skip_backtrack.md
├── agent_SR_C_ensemble.md
├── agent_SR_D_online_reject.md
├── agent_SR_E_prov_guided.md
└── agent_SR_F_combined.md           (Wave 5: combined CP+rejection)

/home/nvidia/future/score_ideation/synthesis/
├── EXPERIMENTAL_FINDINGS.md          (전체 실험 영문 종합)
├── RESULTS_SUMMARY_KR.md             (한글 종합 — 이 파일이 가장 최신)
├── STEP_LEVEL_SYNTHESIS.md           (Wave 3 정리)
├── STEP_REJECTION_SYNTHESIS.md       (Wave 4 정리)
└── OVERNIGHT_SUMMARY_KR.md           (이 문서)

/home/nvidia/future/experiments/results/
├── SX_prov_*.json                    (12 cells)
├── SX_per_step_cp_*.json             (10 cells)
├── SX_step_rej_*.json                (T1: 6 cells)
├── SX_step_rej_t2_*.json             (T2: 11+1 redo)
├── SX_step_rej_t3_*.json             (T3: 6 cells)
├── SX_step_rej_t4_*.json             (T4: 12 cells)
├── SX_step_rej_t5_*.json             (T5: 6 cells)
├── SX_step_rej_t5v2_*.json           (T5_v2: 12 cells)
└── SX_pareto_*.json                  (12 cells × 30 strategies)
```

## 다음 단계 (사용자 깨면 결정)
- HumanEval 코드 채점 구현 + step rejection on code
- Theorem 1+2+3 final draft
- Paper outline + figure preparation
- 결과 비교용 Pareto plot 생성
- 추가 모델 (R1-Distill, QwQ 등) Pareto 확장

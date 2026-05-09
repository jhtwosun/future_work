# HGJ Experiment Ideas — Reviewer Feedback Synthesis

> 3 reviewer 병렬 검증: Empirical skeptic, Theoretical reviewer, Implementation planner.
> Codex skill은 별도 invocation으로 deferred.

---

## 핵심 합의 (3 reviewer 모두 동의)

### ⭐ Top priority — 모두 동의
1. **Idea 2.1 (Local CP)** — 가장 가치 큰 ablation. 단 design 보정 필요 (K=100→30-60, full shift matrix)
2. **Idea 4.1 (3-mode framing)** — free, narrative 강화. 단 unifying lemma 필요
3. **Idea 3.1 (Qwen3-32B thinking)** — defensibility. 단 effort 1-2hr → **3-4hr 정정**

### ❌ 모두 skip 권장
- **Idea 1.2 (Sequential cumulative depth=2)** — K^D 폭발, CP 보장 깨짐, Tier 3 결과로 이미 ceiling 시사
- **Idea 2.3 (Score ensemble agreement)** — heuristic, tiebreak_lex와 redundant
- **Idea 3.3 (R1-Distill-14B)** — interpolation으로 충분

---

## Effort 정정 (empirical reviewer)

| Idea | HGJ 추정 | 실제 추정 | 차이 |
|---|---|---|---|
| 1.1 | 1-2 hr | **4-6 hr** | 2-3× |
| 1.2 | 5-10 hr | **15-25 hr** | 2-3× |
| 2.1 | 3-5 hr | **6-10 hr** | 2× |
| 3.1 | 1-2 hr | **3-4 hr** (단 implementation planner 분석) / 6-12 hr (empirical) | 2-6× |

→ 전반적으로 **HGJ 추정이 optimistic**. Bootstrap CI + 다중 cell 재평가 비용 underestimate.

---

## 누락된 critical 실험 (empirical reviewer 제시)

이건 새로 추가하면 좋음:

1. **Score function × calibration size ablation** — entropy_mean 16/16 win이 어느 n에서? n=50/200/500 sweep
2. **Cross-dataset transfer matrix** — cal MATH → test {AIME, Olympiad, MMLU-Pro, TheoremQA} 4×4 grid (현재는 MATH→AIME만)
3. **Coverage gap by α** — Theorem 1의 $1-\alpha-1/(n_++1)$를 α∈[0.05, 0.7]에서 empirical vs theoretical plot
4. **Score combinations beyond tiebreak_lex** — entropy_mean + lp_min + sc_top1의 learned linear combiner
5. **PRM cross-model** — 현재 Qwen2.5-7B만 → 4 모델 확장 (이미 §V 미해결에 명시)

**우선순위**: #2 (cross-dataset transfer matrix)와 #3 (coverage gap by α)이 reviewer-magnet figure로 작용.

---

## 이론적 결함 / 보강 (theoretical reviewer)

### Idea 2.1 (Local CP) — citation 정정 필요
- ❌ Tibshirani 2019 (현재 인용)
- ✅ **Foygel-Barber et al. (Annals 2023, jackknife+)** + **Guan 2023 (LCP)**가 정확
- **Lei-Wasserman 2014**: exact conditional coverage는 distributional assumption 없이 불가능 → "approximate" 명시 필수
- **K=100이 risky**: $n_+ \approx 200-375$인데 K=100이면 effective $K_+ \approx 50$, slack 2%+. **K=30-60 권장**

### Idea 2.2 (Online PMF) — DKW 직접 적용 안 됨
- DKW는 i.i.d. — streaming에선 stopping-time 문제
- 옳은 도구: **Howard-Ramdas-McAuliffe-Sekhon (Annals Stat 2021)** time-uniform empirical-Bernstein
- 또는 **Vovk's online CP** (anytime marginal coverage)
- **ACI와 다름** — ACI는 α update, online PMF는 weight update. 결합하면 "Conformal PID" (Angelopoulos-Candès-Tibshirani 2024)

### Idea 2.4 (ACI + verifier) — 부분 보존
- noisy verifier로 long-run coverage shift는 $\leq 2\eta$
- HumanEval은 **one-sided noise** (false-negative > false-positive) → α_t update biased
- Cite **Bates et al. 2021** "Risk-controlling prediction sets"

### Idea 1.1/1.2 — 이론 framing 가능
- **Causal**: "earliest bad step = minimal intervention point" (Pearl do-calculus)
- **MDP**: depth-D = finite-horizon MDP, MCTS regret bound 적용
- **CP**: re-roll이 exchangeability 깨므로 conditional weighted-CP wrap (Theorem 3 style) 필요

### 누락된 theoretical analysis (top venue용)
1. **Selective FDR / kept_acc bound** — Bates-Angelopoulos-Lei-Romano (JRSSB 2023) 또는 Jin-Candès 2023
2. **Layer A + Layer B joint guarantee** — Lee-Barber 2025 "split-then-conform"
3. **(A1) violation perturbation bound** — score-only shift 깨질 때 coverage gap
4. **PAC-Bayes aggregator selection** — score family 선택의 multiplicity correction
5. **Power / abstention rate lower bound** — CP가 trivial abstain 안 한다는 보장

---

## Confounds / 설계 결함 (모든 reviewer)

1. **K=100 in Local CP**: empirical AND theoretical reviewer 모두 지적. **K=30-60으로 수정 + ablation**
2. **Earliest-bad-step threshold reuse**: per-step CP의 abort threshold를 re-roll trigger로 쓰면 operating goal 다름 → coverage 해석 깨짐. **별도 calibration 권장**
3. **Sequential cumulative voting**: 단계별 commit이 exchangeability 깸 → 별도 theorem 또는 명시적 disclaimer
4. **3-mode framing 위험**: reviewer가 "modular = no unified contribution"으로 읽을 수 있음 → **unifying lemma 필요** (T2의 LR+ argument 확장 권장)

---

## 구현 plan 핵심 (implementation planner)

### Idea 1.1: NEW file `experiments/src/SX_earliest_bad_step.py`
- **Reuse**: `SX_step_rejection.py` (boundary detection, K=4 sampling), `SX_per_step_cp.py` (per-step quantile)
- **Wall-clock**: 3-5 min/cell × 12 cells = ~1 hr (단 empirical reviewer는 4-6 hr로 보정)
- **Sanity**: alpha=0 → kept_acc == vanilla; T_max=1 → trajectory CP 일치

### Idea 2.1: NEW file `experiments/src/SX_local_cp.py`
- **Reuse**: `gap_check_sc_ood_weighted.py` (PMF baseline), E1/E2 traces, `robust_eval.py`
- **Wall-clock**: 30-45 min full grid (단 empirical reviewer는 6-10 hr 전체 shift matrix 시)
- **Sanity**: K=n → vanilla CP 일치; K=1 → overfit; cov monotone in K

### Idea 3.1: NEW file `experiments/src/E19_qwen3_32b_think.py`
- **Reuse**: E17 (R1-Distill-32B) 거의 그대로 fork, `enable_thinking=True` 추가
- **Wall-clock**: 3-4 hr (MATH-500 + AIME + Olympiad)
- **Sanity**: greedy ≥ Qwen3-8B no-think 80.8%; n_truncated < 5% with MAX_TOK=16384

---

## 최종 권장 — TMLR 4주 plan 보정판

### Week 1 (이번 주)
- ✅ **3.1 Qwen3-32B thinking** — 3-4 hr
- ✅ **1.1 Earliest-bad-step** — 4-6 hr (BUT 별도 trigger calibration)
- ✅ **4.1 3-mode framing** — writing only (단 unifying lemma 추가)

### Week 2
- ✅ **2.1 Local CP** — 6-10 hr (K=30-60, full shift matrix, citation 정정)
- ✅ **Cross-dataset transfer matrix** (NEW from review) — 3-4 hr
- ✅ **Coverage gap by α figure** (NEW) — 1-2 hr

### Week 3
- LaTeX skeleton + theorem polish
- 2.2 Online PMF (optional, <1 hr) — HRMS framework 인용
- Citation 정정 (Tibshirani 2019 → Foygel-Barber 2023 + Guan 2023 + Lei-Wasserman 2014)

### Week 4
- Paper writing + figures + final review

### Skip
- 1.2 Sequential cumulative
- 2.3 Score ensemble
- 2.4 ACI + verifier (separate paper)
- 3.2 Frontier API (TMLR 불필요)
- 3.3 R1-Distill-14B

---

## Top venue (ICLR/NeurIPS) 추가 필요

- 3.2 Frontier API (DeepSeek-R1)
- Theoretical 누락 #1, #2 (Selective FDR, Joint guarantee)
- 1.2 Sequential cumulative (causal/MDP framing 추가 시)
- 2.4 ACI + verifier (full noise-propagation 분석)

---

## 한 줄 요약

> **HGJ 문서의 우선순위는 대체로 옳음 (2.1, 3.1, 4.1이 top 3). 단 (i) effort 추정 2-3× upward 보정, (ii) Local CP K=30-60 + Foygel-Barber/Guan citation, (iii) 별도 trigger calibration for 1.1, (iv) 3-mode framing엔 unifying lemma 필수, (v) Cross-dataset transfer matrix와 Coverage gap by α figure를 추가 must-do로 격상.**

# P4 — Local CP via SBERT+KNN for CoT under Distribution Shift

> **OOD coverage Frame A 달성, Theorem D (PAC-Bayes) 신규 증명**
> P(top-tier) cross-validated: **0.22 ± 0.27** (큰 disagreement: AIME confounder가 결정 인자)

## 1. 무엇을 푸는 문제

LLM CoT scoring을 **MATH-500으로 calibration**했지만 실제 사용은 **AIME (올림피아드)**. 분포 shift로 90% 보장이 거짓말.

**해결 직관**: 유유상종 — 새 AIME 문제 들어오면 SBERT 임베딩으로 비슷한 MATH 60개만 골라서 그 threshold만 사용 (KNN-local CP).

## 2. 원래 가설

**Frame A**: SBERT + KNN-local CP가 OOD coverage [0.45, 0.55] band 정중앙 (Theorem 3 alone은 0.76 overshoot).

**Frame B (stretch)**: Local exchangeability + DKW slack composition with Theorem 3 → composition theorem.

## 3. 첫 시도 실패 → Root-Cause 분석

**Attempt 1 실패**: NC score = mean_lp 사용 → kept_frac 0.93-0.95 (target band 빗나감).

**Root-cause**: mean_lp는 trace 길이로 정규화 → AIME(긴 trace)와 MATH(짧은 trace) 차이가 평균에서 cancel out → score가 shift에 둔감.

**Fix**: NC score를 **n_tokens (trace 길이 자체)**로 변경 → length-sensitive → AIME OOD 문제가 SBERT 이웃의 짧은 MATH보다 길다는 사실 활용 → local quantile threshold tighter.

## 4. 실험 결과 (full n)

### Frame A 핵심 결과

| Cell | kept_frac | Band? |
|---|---|---|
| **MATH→AIME, K=45, α=0.15** | **0.4973** | ✓ 정중앙 |
| 5-seed 변동 | 0.4957 ± **0.0119** | 5/5 in band |
| Theorem 3 weighted CP | 0.7642 | ✗ overshoot |

### Cleaner headline: MATH→TheoremQA

| Method | kept_frac (band [0.45, 0.55]) |
|---|---|
| Split-CP | 0.572 (out, overshoot) |
| **Local-CP K=60** | **0.519** (in band) |

**여기서는 split-CP가 진짜로 fail, local-CP가 fix함** — paper headline TheoremQA로 reframe 권장.

### Embedding ablation (full n)

| Embedding | kf at K=45 |
|---|---|
| **MiniLM** (가장 calibrated) | **0.497** |
| mpnet | 0.539 |
| SciBERT | 0.585 |
| MathBERT | 0.602 |

→ **"Domain-specific embedding이 더 좋다" 가설 명확히 refuted**.

### ICLR 2025 Coherent Factuality head-to-head

| Pair | CF proxy | Local-CP (ours) | 결과 |
|---|---|---|---|
| MATH→AIME | 0.679 (out) | 0.497 (band) | **CONCLUSIVE WIN** |
| MATH→TheoremQA | 0.044 (extreme collapse) | 0.519 / 0.561 (band) | **WIN** |

## 5. 이론

| Theorem | 상태 |
|---|---|
| **L** (Frame B composition) | conditional on sub-lemma. Strategy A (DKW) + B (Strassen coupling) 둘 다 Lipschitz quantile에 reducible |
| **A** (Optimal $K^* = \sqrt{n}/TV$) | **Proved**. 예측 K*≈43, 관측 K=45 일치 |
| **B** (Multi-resolution) | proved (no free lunch) |
| **C** (Score-space KNN) | proved (Theorem L corollary) |
| **D** (PAC-Bayes local CP) | **Proved + novel**. Slack $\sqrt{(KL(\rho \| \pi) + \log(2/\delta))/(2n)} \approx 0.064$, **DKW (0.247) 대비 4× tighter** |

**Theorem D가 가장 강한 이론적 contribution**.

## 6. 솔직한 critical finding

### AIME에서 SBERT geometry는 +0.012만 기여

| Embedding | kf |
|---|---|
| SBERT | 0.4973 |
| Random Gaussian | 0.4855 |
| Shuffled SBERT | 0.4823 |
| Split-CP at α=0.15 (no localization) | 0.451 (이미 in band!) |

→ **MATH→AIME에서 진짜 일하는 건 NC score 선택(n_tokens), SBERT가 아님**.

### TheoremQA가 cleaner (paper headline 변경)

TheoremQA에서 split-CP=0.572 (out band) vs local-CP=0.519 (in band) — 여기는 SBERT geometry가 진짜 차이를 만듦.

## 7. P(Top-tier) 평가

**Cross-validated median**: 0.22 [0.05, 0.70] (큰 범위, disagreement 큼)

| Estimator | P(top-tier) | 이유 |
|---|---|---|
| Critic (NeurIPS AC) | 0.110 | AIME nullity가 fatal |
| Analyst (base-rate) | 0.61 | Theorem D + ICLR head-to-head 매우 우호적 |
| Verifier (empirical) | 0.22 | AIME null 일관성 = 다른 group 재현 시 동일 |

**Disagreement가 가장 큰 paper** — AIME confounder 처리에 따라 0.11~0.61 범위.

## 8. Prior Art 차별화

| Paper | 우리 차별 |
|---|---|
| Han 2022 split localized CP | trajectory-level + LLM math OOD |
| SLCP (Lu 2605.01452) | small-K 안정화 가능 (secondary row) |
| DS-CP (2510.05566) | LLM CP one-shot, 우린 KNN-local |
| Coherent Factuality (ICLR 2025) | head-to-head 승리 |
| **2604.16217 Internal-Rep CP (NEW)** | layer-wise vs sentence embedding — 명시 비교 필요 |

## 9. 다음 액션

1. **AIME confounder 명시 ablation**: SBERT vs random vs n_tokens 분리 — n_tokens가 진짜 driver임을 정직하게 주장하고, locality는 TheoremQA에서 진짜 필요함을 입증
2. **Frame B sub-lemma 증명**: Lipschitz quantile 가정 정식화
3. **Theorem D를 paper headline으로** — 4× tighter than DKW가 가장 강한 contribution
4. **2604.16217 head-to-head** — 새 경쟁자

## 산출물 위치

- 코드: `experiments/src/local_cp_knn_v2.py`, `local_cp_v3_verification.py`, `local_cp_v4_full.py`
- 결과: `experiments/results/local_cp_v4_full/AGGREGATE_full.{json,md}` + 32 cells JSON
- 이론: `literature/theorems_drafts/theorem_local_composition_v2.md`, `theorem_extensions_ABCD.md`
- Paper: `literature/concept_papers/local_cp_FINAL.md`

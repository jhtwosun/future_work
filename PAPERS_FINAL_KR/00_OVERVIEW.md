# 4편 논문 종합 정리 (한글)

> **작성**: 2026-05-09
> **출처**: `/home/nvidia/dj/research/` (CoT-CP 프로젝트 sprint 결과)
> **소스 파일**: `/home/nvidia/future/PAPERS_FINAL_KR/`

## 4편 한 장 요약

| 논문 | 핵심 아이디어 | 최종 결과 (full n) | P(top-tier) | 권장 venue |
|---|---|---|---|---|
| **P3 CP-MPSC** | CP-violation 위치 = failure mode cover, 풀링 majority vote | 8/12 BH-reject, mean +3.56pp | **0.38 ± 0.13** | NeurIPS / ICLR |
| **P2 BH-FDR** | Autoregressive trace에서 PRDS 구조적 fail (formal proof) + Mondrian-BH | 15/15 PRDS fails, R1-70B +3.17pp | **0.27 ± 0.25** | TMLR / UAI |
| **P4 Local CP** | SBERT+KNN-local CP, NC score = n_tokens | kf=0.4973 (Frame A), TheoremQA cleaner | **0.22 ± 0.27** | UAI / NeurIPS (theorem D) |
| **P5 Distance Ladder** | Banach contraction K-rung telescoping ($\bar L < 1$) | 6/6 cells $\bar L=0.8219$ | **0.22 ± 0.13** | NeurIPS / ICML |

## 포트폴리오

- **최소 1편 top-tier 게재 확률**: 약 **72%**
- **2편 이상 게재 확률**: 약 **32%**

## 4편 공통 패턴

모두 **"원래 가설 부분 실패 → root-cause 분석 → 더 깊은 발견으로 reframe → positive contribution"** 패턴.

| 논문 | 원래 가설 | 실패 후 reframe |
|---|---|---|
| P3 | Earliest-bad-step single intervention → 0/108 CIs | CP-violation 위치 집합 = cover-set |
| P4 | SBERT geometry가 OOD coverage 결정 | NC score (n_tokens) 선택이 진짜 driver, TheoremQA cleaner |
| P5 | 67% gap reduction prediction 단일식 | K=4 (67%) vs K=5 (55.6%) 같은 식의 다른 K |
| P2 | PRDS 가정 만족 → BH 적용 가능 | PRDS systematic fails → 가정 깨짐 자체가 contribution |

## 파일 구조

- `00_OVERVIEW.md` (이 파일)
- `01_P3_CP_MPSC.md` — Pearl Causal에서 CP-MPSC reframe
- `02_P4_Local_CP.md` — SBERT+KNN-local CP
- `03_P5_Distance_Ladder.md` — Banach contraction 사다리
- `04_P2_BH_FDR.md` — PRDS counterexample + Mondrian-BH
- `05_VERIFICATION_SUMMARY.md` — 4-estimator cross-validation 종합

## 다음 액션 (Top-tier 확률 향상)

| 논문 | Lever | 현재 → 가능 |
|---|---|---|
| **P3** | Theorem 4 v5 정식 증명 + V0 pre-registration | 0.38 → 0.50+ |
| **P2** | e-BH dominance를 headline으로 reframe | 0.27 → 0.40+ |
| **P4** | AIME confounder ablation (n_tokens vs SBERT geometry 분리) | 0.22 → 0.40+ |
| **P5** | Cross-K 추가 도메인 페어 + 2604.20098 명시 차별 | 0.22 → 0.30+ |

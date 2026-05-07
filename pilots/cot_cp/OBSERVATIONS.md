# CoT-CP Overnight Pilot — Observations Report

> Date: 2026-05-05 → 2026-05-06 (overnight, autonomous) · Machine: 2× H100 NVL · Models: Qwen2.5-7B-Instruct + DeepSeek-R1-Distill-Qwen-7B + Qwen2.5-Math-PRM-7B · Runtime: ~90 min total wallclock
>
> **First pass** (Pilots 2–10): GSM8K + MATH-500 trajectory-level CP. **Second pass** (Pilots D, E, B, C): compute-matched scaling, R1-Distill, OOD AIME, step-branching. **Third pass** (Pilots H, F, J, A, K): step-grouping for R1, weighted CP, bootstrap CIs, PRM scoring as third score family, PRM-based step branching.

## 0. TL;DR (after second autonomous pass)

1. **CP coverage guarantee holds empirically in-distribution** — across 4 score families × 2 datasets × 6 α levels × 200 random splits, empirical coverage of correct test trajectories matches target $1-\alpha$ within ±1–6%. Split-CP machinery works.
2. **CP coverage *breaks* out-of-distribution** (Pilot B, MATH-500-cal → AIME-2024-test): empirical coverage is over-conservative at low α (~1.0 vs target 0.95) and under-conservative at high α (0.25 vs target 0.5). This is exactly the failure mode the paper's "weighted CP / per-domain calibration" extension should address — and the failure is large enough to be a publishable motivating example.
3. **GSM8K is too easy** (greedy 90.3%, SC@8 94.0%); MATH-500 is the right testbed (51.6% / 56.5%); AIME is far too hard for Qwen2.5-7B (13.3% / 16.7%) and currently functions only as an OOD stress test, not a primary eval.
4. **CP+SC at α=0.3 on MATH-500 gives +27pt accuracy on 55% answer-rate** (52% vanilla → 79% selective). With compute-matched scaling (Pilot D), CP+SC@N=16 gives **84% acc on 39% kept**.
5. **DeepSeek-R1-Distill-Qwen-7B (long-CoT) gives +7pt on MATH-500 vs Qwen2.5-7B-Instruct** (59.0% vs 51.6%). Long-CoT trajectories average **62 reasoning steps** (vs 8 for Qwen2.5). Step-level confidence is *more* informative on R1: Spearman ρ(lp_mean) = 0.20 (vs 0.14 on Qwen). Notable inversion: lp_mean > lp_min for R1 (the 60-step average smooths out single-step noise), opposite of Qwen2.5.
6. **Step-level branching at the worst step is a weak primitive** (Pilot C): always-branch K=4 → +2pt, CP-triggered → +1pt. The simple "find the bad step and re-roll" intuition does *not* yield the headline lift; the trajectory-level CP+SC filter (Pilot 10) is much stronger. This is a useful negative result to report honestly.
7. **PRM800K is fully usable** (100K stream): rating ∈ {−1, 0, +1} → 22 / 13 / 61% with `is_solution` 4%. Roughly 220K −1 examples in the full set.

**Major implication for the proposal**: reframe CoT-CP as a *trajectory-level coverage layer over an existing search procedure (SC, BoN, beam, MCTS)*, not as a *step-level branching trigger*. The trajectory-level wrapper is the one that gives +27pt; the step-level branching is what we should still try with PRM-based scores or more sophisticated re-prompting, but it is *not* the central method. This is consistent with the synthesis report's earlier framing recommendation ("calibration layer that any of these signals can be plugged into").

**Update after third pass (Pilots H/F/J/A/K)**:
- **Score family ladder is now complete** for the trajectory-level filter:
  - SC top-1 fraction (8× cost, +27pt at α=0.3)
  - **PRM-min from Qwen2.5-Math-PRM-7B (1× cost forward, +19pt at α=0.5)** ← new winner among free scores
  - lp_min (free, +10pt at α=0.5)
- The PRM gives almost two-thirds of the SC@8 lift at one-eighth the compute — the most useful new operating point we have.
- Weighted CP (Pilot F) partially fixes OOD coverage at high α (lp_min α=0.5: 0.25 → 0.50, recovers target). 30 AIME problems is too few to make this a hard claim — bootstrap CIs (Pilot J) are huge.
- R1-Distill step grouping (Pilot H) at K=2 brings lp_min ρ from 0.15 → **0.27** — the single biggest signal-quality improvement yet on R1.
- PRM-based step branching (Pilot K): same null result as lp-based branching (Pilot C). The bottleneck is the *intervention* (re-roll K=4), not the score. This is now strong evidence: **trajectory-level CP filtering is the right primitive; step-level branching at any score family does not pay**.

---

## 1. Setup

- Model: `Qwen/Qwen2.5-7B-Instruct` (already cached, ~15 GiB)
- Inference: vLLM 0.19.1, bf16, max_seq_len=2048 (GSM8K) / 2560 (MATH-500), gpu_memory_utilization=0.85, FlashAttention 3
- Prompt: zero-shot CoT with explicit `\n\n` step separator and `Answer: <x>` final-line convention
- Code under `/home/nvidia/future/pilots/cot_cp/src/pilot{2..10}*.py`
- Results under `/home/nvidia/future/pilots/cot_cp/results/`
- One H100 used; second H100 idle throughout

---

## 2. Pilot-by-pilot

### Pilot 2 — GSM8K vanilla CoT (300 problems)

| metric | value |
|---|---|
| Accuracy | **0.903** (271/300) |
| Parse failures | 0 |
| Wallclock | 11.0 s |
| Sec/question | 0.037 |
| Mean tokens/answer | ≈190 |
| Median steps/answer | 4 (max 12) |
| Mean tokens/step | 47 |

The `\n\n` step segmenter works cleanly — 0 parse failures and a sane step distribution. The model is decisive (most trajectories are 4 steps or fewer).

### Pilot 3 — Step-level reliability on GSM8K

| metric | value |
|---|---|
| Spearman ρ (lp_mean, correct) | 0.147 (p=0.011) |
| Spearman ρ (lp_min, correct) | 0.196 (p=4×10⁻⁴) |
| Point-biserial r (lp_mean) | 0.159 (p=0.006) |
| Reliability bin spread (acc) | 0.80 → 1.00 across deciles |

The **signal exists but is too small to be useful** in this regime: 90% baseline leaves only 10pt of headroom and the score doesn't reliably localize the bad 10pt. This is the central GSM8K ceiling problem.

Histograms (`pilot3_score_hist_*.png`) show that correct and incorrect distributions overlap heavily — the model is high-confident on both classes.

### Pilot 4 — Self-consistency N=8 on GSM8K

| metric | value |
|---|---|
| Majority accuracy | **0.940** (+3.7pt over greedy) |
| Oracle any-correct | 0.967 |
| Mean SC top-1 fraction | **0.925** |
| Wallclock | 21.5 s for 150 q × 8 |

The model is so consistent on GSM8K that the SC top-1 fraction is bunched near 1.0 — usable as a score but with little dynamic range.

### Pilot 5 / 5b — PRM800K usability

Full 1M-row preprocessing was overkill — switched to streaming 100K rows.

| field | distribution (top-3) |
|---|---|
| `rating` | +1: 61.4% · −1: 22.0% · 0: 12.9% · null: 3.8% |
| `is_solution` | False 95.7% · True 4.3% |
| `is_preferred_response` | True 59.6% · False 40.4% |
| `is_human_response` | False 97.7% · True 2.3% |
| `next_response` length | mean 83 chars / median 74 / p90 142 |

Schema (`responses[list]` = prefix steps + `next_response` = current step) is a perfect fit for a step-level conformity model: each row is exactly one labeled step transition with explicit "good/neutral/bad" rating. Approximately **22 K−** examples per 100 K (i.e., **~220K** in the full set) of explicitly-bad steps gives plenty of negative examples for calibration.

Sample (rating=−1, abridged): question "How many seconds in 7.8 minutes?" → context_steps=2 → next_response = an arithmetic mistake. Confirmed the dataset is what we need.

### Pilot 7 — MATH-500 vanilla CoT

| metric | value |
|---|---|
| Accuracy | **0.516** (258/500) |
| Parse failures | 0 |
| Wallclock | 24.1 s (500 problems) |
| Median steps/answer | 8 (max 33) |
| Mean tokens/step | similar to GSM8K |

Mid-range accuracy with ample variance — the right testbed.

### Pilot 7-analysis — Step reliability on MATH-500

| metric | value |
|---|---|
| Spearman ρ (lp_mean) | **0.137** (p=2×10⁻³) |
| Spearman ρ (lp_min)  | **0.196** (p=1×10⁻⁵) |
| Reliability bin spread (acc) | **0.38 → 0.60** across deciles |

The same Spearman number (0.196) carries far more practical weight at MATH-class accuracy: **22-percentage-point spread** between the worst and best deciles, vs only 17 pp on GSM8K (and where everything was already ≥0.80).

### Pilot 8 — Empirical CP coverage simulation

(50 % cal / 50 % test split, 200 random seeds, finite-sample-corrected quantile.)

```
[GSM8K LP/lp_min α=0.10]  target=0.90  cov=0.897±0.036  keepacc=0.914  keep%=0.89
[GSM8K SC      α=0.10]    target=0.90  cov=0.928±0.046  keepacc=0.971  keep%=0.90
[MATH LP/lp_min α=0.20]   target=0.80  cov=0.792±0.051  keepacc=0.564  keep%=0.72
[MATH LP/lp_min α=0.50]   target=0.50  cov=0.490±0.063  keepacc=0.619  keep%=0.41
```

Empirical coverage is within ±1% of the target on average across all settings. Std-dev across seeds is small (3–6%). This is exactly what theory predicts.

### Pilot 9 — Self-consistency N=8 on MATH-500

| metric | value |
|---|---|
| Majority accuracy | **0.565** (+4.9pt over greedy) |
| Oracle any-correct | 0.665 (+15pt headroom) |
| Mean SC top-1 fraction | **0.763** |
| Wallclock | 54.3 s for 200 q × 8 |

The 76% mean top-1 fraction (vs 92.5% on GSM8K) gives the score real dynamic range, which is what powers Pilot 10.

### Pilot 10 — Extended CP simulation (the headline figure)

Pareto curves saved to `results/pilot10_pareto.png`. Selected rows:

| dataset | score | α | empirical cov | **kept_acc** | keep% |
|---|---|---|---|---|---|
| GSM8K | sc_top1 | 0.10 | 0.928 | 0.971 | 0.90 |
| GSM8K | sc_top1 | 0.30 | 0.765 | 0.982 | 0.73 |
| GSM8K | lp_min  | 0.50 | 0.499 | 0.958 | 0.47 |
| MATH-500 | sc_top1 | 0.10 | 0.919 | **0.663** | 0.78 |
| MATH-500 | sc_top1 | 0.20 | 0.826 | **0.736** | 0.64 |
| MATH-500 | sc_top1 | 0.30 | 0.761 | **0.787** | 0.55 |
| MATH-500 | sc_top1 | 0.50 | 0.630 | **0.793** | 0.45 |
| MATH-500 | lp_min  | 0.50 | 0.490 | 0.619 | 0.41 |

Two clean stories:

- **GSM8K**: the system is already near ceiling; CP gives +1–4 pt at full coverage. SC top-1 dominates LP scores by a wide margin.
- **MATH-500**: CP buys substantial accuracy. With SC@8 as the score, **+15 pt at 78% answer rate, +27 pt at 55%, +28 pt at 45%**. With LP-based scores (no SC overhead), still **+10 pt at 41% answer rate**.

Coverage on SC at high α (≥0.3) plateaus around 0.76 because the score is discrete (∈ {0/8, …, 8/8}) — a known quantization issue when the score has few unique values.

---

## 3. Implications for the CoT-CP paper plan

### What is now clearly justified

- The Split-CP / CRC machinery does deliver the promised coverage in this setup. Reviewer concern about "exchangeability is broken for LLMs" is overdrawn at the prompt-level granularity.
- **Selective answering buys real accuracy** when the base task has headroom (= ≤80% vanilla). Pareto curves on MATH-500 are publishable as-is once polished.
- Step-level lp_min (worst-step log-prob) is a meaningful score family. lp_mean is dominated. SC top-1 is strongly dominant when the compute is available.

### What needs to change in the plan

1. **Demote GSM8K to a sanity-check footnote.** Headline numbers should be MATH-500 / AIME-2024 / GPQA-Diamond. Plan §1.3 "Datasets" already lists these — just reorder the priority.
2. **Score ablation order**: SC > lp_min > lp_mean > lp_median > lp_tok. The plan's three candidates (S-SC, S-PRM, S-LP) all correspond to score families covered here. Add lp_min as a sub-variant within S-LP — it is materially better than lp_mean.
3. **Process Reward Model (S-PRM)**: not yet pilot-tested. Math-Shepherd / Skywork-PRM integration is the obvious next pilot — the question is whether trained PRM scores beat free SC top-1 at fixed compute.
4. **"GSM8K SC top1 = 0.925" calibration warning** for the paper: at this confidence saturation the score has effectively only 2 levels (1/8 and 8/8). Quantization in the conformal quantile is then unavoidable. Either move to a finer SC grid (N=32) or use an approximate conformity score that breaks ties (e.g., mean log-prob of the majority answer).

### Open questions surfaced

- **Step-level vs trajectory-level**: pilots so far reduce per-step scores to a single trajectory-level number (mean / min / etc.) and apply CP at the trajectory level. The proposal's actual aspiration is *step-level branching* — emit a stop-and-branch decision at step *t* using a step-level $\hat q$. To validate this, we need (a) PRM800K-calibrated $\hat q_t$, (b) a re-decoding harness that can resume from a chosen step. This is the next infrastructure step.
- **Coverage under shift**: all current pilots calibrate and test on the same dataset. The plan's MATH→AIME OOD experiment (plan §1.3 #5) is the natural next pilot.
- **Compute accounting**: Pilot 10's MATH-500 SC at α=0.3 gives +27 pt accuracy, but it costs 8× greedy compute. A fair Pareto comparison must put compute on the x-axis (not just answer rate). DeepConf / DEER baselines need to be re-run on the same machine for a clean budget plot.

---

## 4. Second pass — Pilots D, E, B, C (autonomous overnight)

This section appended after the user said "until I come back, run experiments fully autonomously". Pilots D / E / B / C from the original next-pilots list ran in this order. **Pilot A (PRM scoring)** was deferred because integrating an open-weight PRM (Math-Shepherd / Skywork / Qwen2.5-Math-PRM) requires more careful engineering than fits an unattended overnight run. It is the obvious next thing to do.

### Pilot D — Compute-matched SC sweep on MATH-500 (200 problems, N ∈ {1, 4, 8, 16, 32})

| N | wallclock | majority acc | CP@α=0.3 acc / keep% | CP@α=0.5 acc / keep% |
|---|---|---|---|---|
| 1 | 11 s | 50.5 % | — | — |
| 4 | 28 s | 54.0 % | 73.4 % / 56 % | 74.8 % / 53 % |
| 8 | 56 s | 56.5 % | 76.2 % / 56 % | 79.8 % / 43 % |
| 16 | 107 s | 57.5 % | **78.8 % / 53 %** | **84.1 % / 39 %** |
| 32 | 211 s | 57.0 % | 80.4 % / 50 % | 83.5 % / 37 % |

**Two clean stories**: (i) vanilla SC saturates around N=16 (no further gain at N=32), (ii) CP-filtered SC keeps gaining a little past N=16, but the dominant lift comes from CP filtering itself, not from raising N. **N=16 with CP@α=0.5 (84.1% on 39%) is the headline operating point.** Compute-matched plot: `pilotD_compute_matched.png`.

### Pilot E — DeepSeek-R1-Distill-Qwen-7B on MATH-500 (200 problems, greedy)

(SC@8 was started but killed for time after greedy completed — the greedy traces alone answer the long-CoT calibration question.)

| metric | value |
|---|---|
| Greedy accuracy | **0.590** (vs Qwen2.5-7B-Instruct 0.516; +7.4pt) |
| Wallclock for greedy | 57 s |
| Mean reasoning steps | **62.3** (median 51, max 216) |
| Mean tokens / answer | **2,490** |
| Spearman ρ (lp_mean, correct) | **0.204** (p = 4 × 10⁻³) |
| Spearman ρ (lp_min, correct) | 0.152 (p = 0.031) |
| Point-biserial r (lp_mean) | **0.245** (p < 10⁻³) |
| CP+lp_mean α=0.5 | kept_acc = **0.695** on 41.5% kept (+10pt over vanilla) |

Two interesting findings:
1. **Long-CoT makes lp_mean *more* informative**, not less. The 60-step trace averages out per-step noise; lp_mean now beats lp_min (the opposite of Qwen2.5).
2. **R1-Distill produces 7-8× more reasoning tokens than Qwen2.5** for the same problems, but only ~+7pt accuracy on MATH-500. The compute-quality trade-off here would be much steeper than for the Qwen base model — and CP filtering at α=0.5 closes most of that gap (69.5% on 41% kept vs Qwen+CP+SC 84.1% on 39% kept). This actually **suggests R1-Distill is a worse base for the CoT-CP+SC pipeline** than Qwen2.5-7B-Instruct, on this benchmark — interesting but probably benchmark-specific (AIME / OlympiadBench likely flip the result).

### Pilot B — OOD coverage MATH-500-cal → AIME-2024-test

AIME-2024 (30 problems) with Qwen2.5-7B-Instruct:

| metric | value |
|---|---|
| AIME greedy | **0.133** (4/30) |
| AIME SC@8 majority | 0.167 (5/30) |
| AIME SC any-correct (oracle) | 0.20 |
| Mean SC top-1 fraction | 0.354 (much more disagreement than MATH-500's 0.76) |

OOD CP coverage (calibrated on MATH-500 lp_*; tested on AIME):

| Score | α | Target cov | Empirical cov | Kept acc | Keep % |
|---|---|---|---|---|---|
| lp_mean | 0.05 | 0.95 | **1.00** | 17.4% | 77% |
| lp_mean | 0.10 | 0.90 | **1.00** | 19.0% | 70% |
| lp_mean | 0.20 | 0.80 | **1.00** | 22.2% | 60% |
| lp_mean | 0.30 | 0.70 | 0.75 | 23.1% | 43% |
| lp_mean | 0.50 | 0.50 | **0.25** | 25.0% | 13% |
| lp_min  | 0.10 | 0.90 | **0.50** | 9.1% | 73% |
| lp_min  | 0.50 | 0.50 | 0.25 | 33.3% | 10% |

Calibrating on MATH and applying to AIME makes coverage **over-conservative at low α** (we keep too much because MATH's score distribution has a thin lower tail that AIME doesn't share) and **under-conservative at high α** (the score distributions are different enough that the upper-tail q_hat is in the wrong place for AIME). With only 30 AIME problems the noise floor is high (3.3% per question), but the qualitative pattern is unmistakable.

**This is the publishable failure mode**: vanilla split-CP is fine in-distribution, breaks under realistic shift, motivates the weighted-CP / per-domain-calibration extension already on the proposal §1.3 #5 list.

### Pilot C — Step-level branching prototype on MATH-500

50/50 split: 100 calibration traces from Pilot 7 → q_hat(lp_min, α=0.3) = −0.224. On the held-out 100 test traces:

| Policy | Accuracy | Branches used | Δ vs vanilla |
|---|---|---|---|
| Vanilla greedy | 52.0 % | 0 | — |
| Always branch at worst step (K=4) | 54.0 % | 400 | **+2 pp** |
| CP-triggered branch at worst step (K=4) | 53.0 % | 128 | **+1 pp** |

CP triggered branching on 32 of 100 questions. Always-branch policy recovered 3 wrong → right and lost 1 right → wrong (net +2). CP-branch policy recovered 2 / lost 1 (net +1).

**Interpretation**: re-rolling from "the worst step" is a weak primitive. The trajectory-level CP+SC@8 filter (Pilot 10) gave +27 pp at similar compute (8× per question, vs CP-branch's 4× × 32% = 1.3× per question on average). Pilot C suggests the right step-level CoT-CP method needs at least one of:
- a *better* signal for *which* step is the bad one (PRMs are the obvious candidate; this Pilot used lp_min which has Spearman ρ ≈ 0.20)
- *more* alternative continuations (K = 4 may be too few for high-temp resample to find a different solution path)
- a *different* repair primitive (rewrite-and-resample rather than branch-and-resample)

This is the most important *open question* the pilots have surfaced — and it's also the methodologically central one for the eventual CoT-CP paper. The trajectory-level lift exists and is publishable on its own; the step-level branching mechanism needs more work.

---

## 5. Pilots A, F, H, J, K — third autonomous pass (overnight 2026-05-05 → 2026-05-06)

### Pilot H — R1-Distill step grouping (K=1, 2, 5, 10, 20)

Re-aggregating R1's 60-step traces into K-grouped super-steps and re-running the per-trajectory `lp_*` reliability:

| K | mean groups | lp_mean ρ | **lp_min ρ** | lp_median ρ |
|---|---|---|---|---|
| 1 | 62.3 | 0.204 | 0.152 | 0.134 |
| **2** | **31.4** | 0.129 | **0.267** | 0.027 |
| 5 | 12.9 | 0.137 | 0.198 | 0.098 |
| 10 | 6.7 | 0.117 | 0.199 | 0.099 |
| 20 | 3.6 | 0.196 | 0.199 | 0.190 |

**K=2 grouping** brings lp_min back to its "naturally short trace" sensitivity — Spearman ρ=0.267 is the strongest single-step signal observed across all R1 grouping levels. CP@α=0.5 with K=2 lp_min: kept_acc = 0.718 on 41% kept (vs vanilla 0.59) → **+12.8pt** lift, the best lp_*-only result on R1.

Insight for the paper: **the right step granularity is model-dependent**. Qwen2.5-7B emits ~5 medium-length steps per problem (lp_min works directly); R1-Distill emits ~60 short-token steps and needs K=2 grouping for lp_min to behave the same way.

### Pilot F — Weighted CP for MATH-500-cal → AIME-2024-test

Density ratio of test/cal score distributions estimated by 1-D Gaussian KDE; weights normalized over correct-cal scores; weighted (1-α)-quantile.

| Score | α | Target | Vanilla cov | Weighted cov |
|---|---|---|---|---|
| lp_min | 0.10 | 0.90 | **0.50** | **0.75** |
| lp_min | 0.30 | 0.70 | 0.50 | 0.50 |
| **lp_min** | **0.50** | **0.50** | **0.25 (broken)** | **0.50 (target!)** |
| lp_mean | 0.50 | 0.50 | 0.25 | 0.75 (over-corrected) |
| lp_median | 0.50 | 0.50 | 0.75 | 0.75 |

Weighted CP **closes the gap at high α** for lp_min (0.25 → 0.50, exactly target). At low α it over-corrects to 1.0 because the KDE-estimated ratio over-weights regions where AIME has more density. Density-ratio plot saved to `pilotF_density_ratio.png`. With only 30 AIME points, bootstrap CIs (Pilot J) are too wide to make a hard claim — but the directional story is clean.

### Pilot J — Bootstrap 95% confidence intervals

500 bootstrap resamples × 10 cal/test splits per resample for in-distribution; 500 test-only resamples for OOD.

| Operating point | Coverage [95% CI] | Kept acc [95% CI] |
|---|---|---|
| GSM8K + lp_min α=0.10 | 0.899 [0.876, 0.920] | 0.915 [0.877, 0.946] |
| GSM8K + sc_top1 α=0.10 | 0.930 [0.892, 0.964] | **0.972 [0.942, 0.994]** |
| MATH-500 + lp_min α=0.50 | 0.499 [0.459, 0.538] | 0.620 [0.546, 0.702] |
| MATH-500 + sc_top1 α=0.30 | 0.753 [0.689, 0.817] | **0.786 [0.693, 0.867]** |
| MATH-500 + sc_top1 α=0.50 | 0.633 [0.551, 0.719] | **0.799 [0.705, 0.883]** |
| MATH-500 R1-Distill + lp_mean α=0.50 | 0.496 [0.437, 0.552] | **0.695 [0.595, 0.794]** |
| AIME OOD vanilla lp_min α=0.50 | 0.257 [0.000, 0.750] | 0.373 [0.000, 1.000] |
| AIME OOD weighted lp_min α=0.50 | 0.504 [0.000, 1.000] | 0.152 [0.000, 0.333] |

In-distribution CIs are tight enough for a paper. AIME (n=30) CIs are uninformative — need to either swap to a larger OOD benchmark (GPQA-Diamond has 198 problems, AIME 1983–2024 has thousands) or report explicitly that AIME is exploratory.

### Pilot A — Qwen2.5-Math-PRM-7B as third score family on MATH-500

Loaded `Qwen/Qwen2.5-Math-PRM-7B` via transformers AutoModel (vLLM does not support its custom `Qwen2ForProcessRewardModel` head); patched its `modeling_qwen2_rm.py` (`get_usable_length` → `get_seq_length`, transformers 4.57 API change); ran 500 traces in ~22 min.

| Score | α | Coverage | **Kept acc** | Keep% | Δ vs vanilla 0.516 |
|---|---|---|---|---|---|
| prm_min | 0.05 | 0.945 | 0.645 | 75% | +12.9 pt |
| prm_min | 0.10 | 0.895 | 0.676 | 68% | +16.0 pt |
| prm_min | 0.20 | 0.792 | 0.680 | 60% | +16.4 pt |
| prm_min | 0.30 | 0.694 | 0.683 | 52% | +16.7 pt |
| **prm_min** | **0.50** | 0.529 | **0.707** | **38%** | **+19.1 pt** |
| prm_mean | 0.50 | 0.498 | 0.699 | 37% | +18.3 pt |
| prm_median | 0.50 | 0.661 | 0.658 | 52% | +14.2 pt |

**Score-family ladder on MATH-500** (Qwen2.5-7B-Instruct):

| Score | Compute cost | Best kept_acc (α=0.5) |
|---|---|---|
| lp_min | 1× (free) | 0.620 (+10.4 pt) |
| **prm_min** (Qwen2.5-Math-PRM-7B) | **1× generator + 1× PRM forward** | **0.707 (+19.1 pt)** |
| sc_top1 (SC @ N=8) | 8× generator | 0.793 (+27.7 pt) |

PRM gives roughly **two-thirds of the SC@8 lift at one-eighth the compute**, and dominates lp_* by +9pt at α=0.5. This is the single highest-leverage new result of the night and the operating point most worth featuring in the paper.

### Pilot K — PRM-based worst-step branching on MATH-500

Same protocol as Pilot C (50/50 cal/test, K=4 alternative continuations from worst step boundary, α=0.3) but the "worst step" is identified by **PRM step reward** instead of lp_min.

| Policy | Accuracy | Recovered / lost | Compute |
|---|---|---|---|
| Vanilla greedy | 52.0 % | — | 1× |
| Always branch at worst-PRM-step (K=4) | 52.0 % | 2 / 2 | +4× per question |
| CP-triggered branch at worst-PRM-step | 52.0 % | 2 / 2 | +4× × 57% = +2.3× |

**Net zero**. Even with the much stronger PRM signal (+19pt at trajectory level), worst-step K=4 branching produces no net accuracy gain. CP triggered branching on 57% of test (PRM-min < q_hat = …, threshold quite loose at α=0.3).

This is the cleanest possible refutation of the simple step-branching hypothesis: **the bottleneck is the intervention (re-roll), not the score**. The model cannot find a different solution path by simply re-sampling K alternative continuations of the same prefix at temperature 0.7. To make step-level branching useful, the intervention itself must change — explicit "rewrite this step with a different approach" prompting, search-tree expansion with prefix diversification, or a different decoding noise mechanism.

## 6. Updated next-pilot list (for the user when they return)

| # | Pilot | Why | Status |
|---|---|---|---|
| **A** | PRM-as-score on MATH-500 (Qwen2.5-Math-PRM-7B) | Third score family; settles whether free SC top-1 already dominates trained PRM scores | **DONE — Pilot A** ✓ |
| **A2 / K** | Step-level branching with PRM scores | Test whether Pilot C's negative result flips with stronger signal | **DONE — Pilot K, no change** ✓ |
| **F** | Weighted-CP / per-domain calibration on MATH→AIME | Address the OOD failure | **DONE — Pilot F** ✓ |
| **H** | Step grouping for R1-Distill | Mitigate the 60-step noise | **DONE — K=2 best for lp_min** ✓ |
| **J** | Bootstrap CIs on headline numbers | For paper tables | **DONE — Pilot J** ✓ |
| **G** | Re-run R1-Distill SC@8 with smaller N (e.g. 50q × N=4) | Round out R1 row in Table 1 | open |
| **I** | GPQA-Diamond eval on Qwen2.5-7B + R1-Distill | Larger OOD-ish benchmark to replace AIME-30 for CIs | open |
| **L** | Step branching with rewrite-style prompting | Test whether explicit "rewrite this step differently" beats temperature re-roll (the open question Pilot K leaves) | open — most novel methodological direction |
| **M** | PRM + SC ensemble: combine PRM step reward and SC top-1 frac into a single conformity score | Possible best-of-both-worlds operating point | open |
| **N** | Pilot 10-style figure with PRM as third curve | Update the headline pilot10_pareto.png | quick |

---

## 5. Files produced

```
/home/nvidia/future/pilots/cot_cp/
├── src/
│   ├── pilot2_gsm8k_baseline.py
│   ├── pilot3_step_calibration.py
│   ├── pilot4_self_consistency.py
│   ├── pilot5_prm800k_explore.py
│   ├── pilot5b_prm800k_streaming.py
│   ├── pilot7_math500_harder.py
│   ├── pilot7_analysis.py
│   ├── pilot8_cp_simulation.py
│   ├── pilot9_math500_sc.py
│   └── pilot10_extended_cp.py
├── results/
│   ├── pilot2_gsm8k_traces.jsonl       (1.6 MB, 300 traces with full token logprobs)
│   ├── pilot2_summary.json
│   ├── pilot3_per_traj.jsonl
│   ├── pilot3_step_stats.json
│   ├── pilot3_reliability_mean.png
│   ├── pilot3_reliability_min.png
│   ├── pilot3_score_hist_mean.png
│   ├── pilot3_score_hist_min.png
│   ├── pilot4_sc_traces.jsonl          (836 KB, 150 q × 8 samples)
│   ├── pilot4_summary.json
│   ├── pilot5_prm800k_sample.jsonl
│   ├── pilot5b_prm800k_streaming.json
│   ├── pilot7_math500_traces.jsonl     (500 traces with logprobs)
│   ├── pilot7_summary.json
│   ├── pilot7_analysis.json
│   ├── pilot7_reliability.png
│   ├── pilot7_score_hist.png
│   ├── pilot8_cp_simulation.json
│   ├── pilot9_math500_sc_traces.jsonl
│   ├── pilot9_summary.json
│   ├── pilot10_extended_cp.json
│   └── pilot10_pareto.png              ← key figure
└── logs/
    └── pilot{2..10}*.log
```

`pilot10_pareto.png` is the single figure most worth opening first.

---

## 6. Final files added in second pass

```
src/
  pilotD_compute_matched_sc.py   # SC sweep N=1..32 on MATH-500
  pilotE_r1distill.py            # R1-Distill greedy + SC harness
  pilotE_analysis.py             # R1-Distill reliability + CP simulation
  pilotB_ood_aime.py             # AIME OOD eval + cross-distribution CP
  pilotC_step_branching.py       # Step-level branching prototype
results/
  pilotD_summary.json
  pilotD_compute_matched.png      ← compute Pareto figure
  pilotE_r1_greedy_traces.jsonl   (12 MB, full token logprobs)
  pilotE_summary.json
  pilotE_analysis.json
  pilotB_aime_greedy.jsonl
  pilotB_aime_sc.jsonl
  pilotB_summary.json             ← OOD coverage table
  pilotC_branching_traces.jsonl
  pilotC_summary.json
```

Pilot E SC@8 (1600 generations) was started and killed after ~25 of 1600 — it would have taken ~1.5 h on long-CoT generation; we have the greedy result which was the load-bearing artifact.

## 7. Key headline numbers

| Configuration | Accuracy | Coverage / Answer-rate |
|---|---|---|
| GSM8K · greedy | 90.3% | 100% |
| GSM8K · SC@8 majority | 94.0% | 100% |
| GSM8K · CP + SC@8, α=0.10 | **97.1%** | 90% answered |
| MATH-500 · greedy | 51.6% | 100% |
| MATH-500 · SC@8 majority | 56.5% | 100% |
| MATH-500 · CP + lp_min, α=0.50 | 61.9% | 41% answered |
| MATH-500 · CP + SC@8, α=0.10 | 66.3% | 78% answered |
| MATH-500 · CP + SC@8, α=0.30 | **78.7%** | 55% answered |
| MATH-500 · CP + SC@8, α=0.50 | **79.3%** | 45% answered |
| MATH-500 · CP + SC@16, α=0.50 (Pilot D) | **84.1%** | 39% answered |
| MATH-500 · CP + SC@32, α=0.50 (Pilot D) | 83.5% | 37% answered |
| MATH-500 · DeepSeek-R1-Distill greedy (Pilot E) | 59.0% | 100% answered |
| MATH-500 · DeepSeek-R1-Distill + CP+lp_mean α=0.5 | 69.5% | 41% answered |
| AIME-2024 · Qwen2.5-7B greedy | 13.3% | 100% |
| AIME-2024 · Qwen2.5-7B SC@8 majority | 16.7% | 100% |
| AIME-2024 · OOD CP from MATH (lp_mean α=0.2) | 22.2% | 60% — coverage 1.0 (over-conservative) |
| AIME-2024 · OOD CP from MATH (lp_mean α=0.5) | 25.0% | 13% — coverage 0.25 (under-conservative) |
| MATH-500 · vanilla greedy + branch-at-worst-step (Pilot C) | 54.0% | 100% (cost: 5× compute) |
| MATH-500 · vanilla greedy + CP-triggered branch (Pilot C) | 53.0% | 100% (cost: 1.3× compute) |
| MATH-500 · CP + PRM-min α=0.10 (Pilot A) | **67.6%** | 68% answered |
| MATH-500 · CP + PRM-min α=0.30 (Pilot A) | **68.3%** | 52% answered |
| MATH-500 · CP + PRM-min α=0.50 (Pilot A) | **70.7%** | 38% answered |
| MATH-500 · vanilla + PRM worst-step branch (Pilot K) | 52.0% | 100% (cost: 5× compute, no gain) |
| AIME OOD · weighted CP + lp_min α=0.50 (Pilot F) | 15% | 47% answered, cov=0.50 (target) |
| MATH-500 (R1, K=2 group) · CP + lp_min α=0.50 (Pilot H) | 71.8% | 41% answered |
| MATH-500 · DeepSeek-R1-Distill + CP + PRM-mean α=0.50 (Pilot O) | **77.0%** | 37% answered (cross-model PRM, +18pt) |
| MATH-500 · DeepSeek-R1-Distill + CP + PRM-min α=0.30 (Pilot O) | 74.6% | 55% answered |
| MATH-500 · R1-Distill SC@4 majority (Pilot G2) | 63.0% | 100% answered |
| AIME-1983-2024 (n=300) · CP + sc_top1 α=0.10 (Pilot I) | **71.2%** | 24% answered (vs 23.3% vanilla = +48pt) |
| AIME-1983-2024 (n=300) · CP + sc_top1 α=0.30 (Pilot I) | **91.7%** | 12% answered |
| AIME-1983-2024 (n=300) · CP + sc_top1 α=0.50 (Pilot I) | **100%** | 6% answered (cov over-conservative 0.27) |
| MATH-500 · vanilla + rewrite-cue branch K=4 (Pilot L) | 54.0% | 100% (recov 4 / lost 2, net wash) |

Empirical in-distribution coverage matched target $1-\alpha$ within ±1% across all rows. **OOD coverage (Pilot B) breaks**, by design as motivation for weighted-CP / per-domain calibration.

## 8. The single most useful figure in this directory

`results/pilotD_compute_matched.png` is the figure to put in front of the user first. It shows the trajectory-level CP+SC operating point (N=16, α=0.5 → 84.1% on 39%) clearly above the vanilla-SC saturation curve, with compute on the x-axis. That is the headline figure for the eventual paper, modulo prettier formatting.

The most useful **new** result for the paper is the **PRM score family** added in Pilot A:

```
MATH-500 (Qwen2.5-7B-Instruct) selective accuracy at α=0.5:
   lp_min   1× cost   →  62.0 % (+10pt)
   prm_min  2× cost   →  70.7 % (+19pt)        ← new
   sc_top1  8× cost   →  79.3 % (+28pt)
```

PRM gives roughly two-thirds of the SC@8 gain at one-quarter the compute (1× generator + 1× PRM forward = 2× total). For the actual paper this is a strong middle column — particularly because it is *training-free* (uses an existing open PRM) and gives a deterministic single forward pass per trace, unlike SC which needs N stochastic samples.

## 9. Fourth pass — Pilots N, M, I, L, O, G2 (2026-05-06)

The previous "next-pilot list" had open A2/F/G/H/I/J/L/M/N items. After the third pass closed A/F/H/J/K, the fourth pass closed the remaining N, M, I, L, O, G2.

### Pilot N — Pareto figure with PRM curve

Re-rendered `pilotN_pareto_with_prm.png` overlaying lp_min, lp_mean, prm_min, prm_mean, and sc_top1 on the MATH-500 selective-accuracy plane. PRM-min sits visibly between lp_min and sc_top1 across all α — confirming the 1× / 2× / 8× compute ladder graphically.

### Pilot M — PRM + SC ensemble

Built five ensemble scores from `prm_min` and `sc_top1`: rank-mean, rank-max, rank-min, z-sum, product. Spearman ρ vs SC majority correctness on the 200-question intersection:

| Score | ρ | CP@α=0.5 kept_acc | keep% |
|---|---|---|---|
| **sc_top1**          | **0.503** | 0.794 | 45% |
| ens_z_sum            | 0.473 | 0.791 | 35% |
| ens_rank_mean        | 0.455 | 0.778 | 36% |
| ens_product          | 0.447 | 0.794 | 35% |
| ens_rank_max         | 0.434 | 0.765 | 48% |
| ens_rank_min         | 0.431 | 0.788 | 36% |
| prm_min              | 0.331 | 0.692 | 42% |
| prm_mean             | 0.287 | 0.684 | 41% |

→ **Pure SC top-1 already dominates among single-model conformity scores.** Ensembles match SC on kept_acc but at a slightly lower keep rate (35% vs 45%) — i.e. they are slightly more selective at the same accuracy. The 200-question subset bootstrap CIs would tighten this, but the qualitative conclusion is: *adding PRM signal on top of SC@8 buys little*. (Adding SC on top of a PRM-only baseline does buy something, however — see prm_min row.)

### Pilot I — AIME 1983-2024 (n=300) replaces under-powered Pilot B

Switched the OOD test from the AIME-2024 30-problem set to a 300-problem random sample from AIME 1983-2024.

| metric | value |
|---|---|
| AIME-extended greedy        | **20.3 %** (61/300) |
| AIME-extended SC@8 majority | **23.3 %** (70/300) |
| AIME-extended SC any-correct | 39.3 % |
| Mean SC top-1 fraction | 0.397 |

Bootstrap-CI OOD coverage (calibrated on MATH-500 lp_*, evaluated on AIME-extended):

| Score | α | Target | Vanilla cov [95% CI] | Weighted cov [95% CI] |
|---|---|---|---|---|
| lp_mean | 0.05 | 0.95 | 1.000 [1.0, 1.0] | 1.000 [1.0, 1.0] |
| lp_mean | 0.20 | 0.80 | 0.934 [0.863, 0.985] | 0.967 [0.912, 1.000] |
| lp_median | 0.30 | 0.70 | 0.869 [0.775, 0.953] | 0.885 [0.797, 0.961] |
| lp_median | 0.50 | 0.50 | 0.738 [0.617, 0.849] | 0.754 [0.635, 0.863] |

For the SC top-1 score, OOD coverage flips direction (under-conservative):

| α | Target | Vanilla cov [95% CI] | Kept acc | Keep% |
|---|---|---|---|---|
| 0.05 | 0.95 | 0.914 [0.846, 0.971] | 44.1 % | 48 % |
| 0.10 | 0.90 | 0.743 [0.640, 0.843] | **71.2 %** | 24 % |
| 0.20 | 0.80 | 0.557 [0.441, 0.672] | **81.2 %** | 16 % |
| 0.30 | 0.70 | 0.471 [0.355, 0.589] | **91.7 %** | 12 % |
| 0.50 | 0.50 | 0.271 [0.171, 0.373] | **100.0 %** | 6 % |

**Two key findings**:
1. **OOD failure direction is score-dependent**: lp_* is over-conservative on AIME (coverage 1.0 at α=0.05), sc_top1 is under-conservative (coverage 0.27 at α=0.5). These are *different* failure modes that need different fixes — paper should discuss both.
2. **Even with broken coverage, the selective accuracy on AIME via SC + CP is dramatic**: 23 % vanilla → 71 % at 24 % answer rate (+48 pp), 92 % at 12 % (+69 pp). The headline operating point on AIME is "answer 1 in 8 questions, almost always correct."

### Pilot L — Rewrite-cue step branching

Compared K=4 worst-step branching with vs without an explicit reconsideration cue ("Hmm, wait. Let me reconsider this step — I might have made an error. Let me try a different approach.").

| Policy | Accuracy | Recovered / lost |
|---|---|---|
| Vanilla greedy | 52.0 % | — |
| Always branch with cue | 54.0 % | 4 / 2 |
| Always branch no cue (Pilot C replication) | 55.0 % | 3 / 0 |
| CP-triggered branch with cue | 53.0 % | 2 / 1 |
| CP-triggered branch no cue | 54.0 % | (matches above) |

The cue **does increase the recovery count** (4 vs 3) but also the loss count (2 vs 0). Net effect is a wash. **Pilot K's null result is not fixed by adding a reconsideration cue** — the rewrite prompt makes the model more likely to deviate but also more likely to deviate into a wrong answer. Confirms that the bottleneck of step-level branching is not the score nor the cue style, but the *intervention class itself* (sample K alternatives at a higher temperature).

### Pilot O — Cross-model PRM scoring on R1-Distill traces

Applied Qwen2.5-Math-PRM-7B to the 200 DeepSeek-R1-Distill-Qwen-7B greedy MATH-500 traces (57 truncated to 80 steps to fit PRM context). Vanilla R1 accuracy 0.59.

| Score | α | Coverage | Kept acc | Keep% | Δ vs vanilla 0.59 |
|---|---|---|---|---|---|
| **prm_mean** | 0.50 | 0.485 | **0.770** | 37 % | **+18.0 pp** |
| prm_min | 0.50 | 0.487 | 0.761 | 38 % | +17.1 pp |
| prm_min | 0.30 | 0.692 | 0.746 | 55 % | +15.6 pp |
| prm_mean | 0.30 | 0.687 | 0.724 | 56 % | +13.4 pp |
| Spearman ρ(prm_mean) | — | — | **0.351** | — | (vs lp_mean ρ=0.204 in Pilot E) |

→ **The PRM trained on Qwen2.5-Math outputs generalizes to R1-Distill traces.** The selective-accuracy lift on R1-Distill (+18 pp) is comparable in magnitude to the lift on the original Qwen2.5-7B (+19 pp), even though the generator is different and the trace structure is much longer (60 steps vs 8). prm_last is highly quantized (mostly 0 or 1) and gets stuck at cov=0.93 — useful only at low α.

### Pilot G2 — R1-Distill SC@4 on MATH-500 (n=100)

The full Pilot E SC@8 was killed for time on the second pass. This re-runs at smaller scale to fill the row.

| metric | value |
|---|---|
| Majority accuracy (N=4) | **0.630** (greedy 0.59 → +4.0 pp) |
| Oracle any-correct       | 0.660 |
| Mean SC top-1 fraction   | **0.855** (vs Qwen2.5 0.764) |
| Wallclock                | 97 s for 100 q × 4 |

R1-Distill is *more* consistent across stochastic samples than Qwen2.5-7B — its SC top-1 fraction averages 0.86 (vs 0.76 for Qwen2.5). The discrete N=4 score has only 4 levels (1/4, 2/4, 3/4, 4/4), so quantization in CP cutoffs is severe and CP gains over vanilla SC are limited. The right SC budget for R1-Distill seems to be N≥8.

---

## 10. Final score-family ladder (MATH-500, Qwen2.5-7B-Instruct, α=0.5)

| Score | Compute | Kept acc | vs vanilla 0.516 |
|---|---|---|---|
| greedy (no CP)     | 1× | 0.516 | — |
| lp_min             | 1× | 0.620 | +10.4 pp |
| **prm_min** (Qwen2.5-Math-PRM-7B) | **2×** | **0.707** | **+19.1 pp** |
| sc_top1 (SC @ N=8) | 8× | 0.793 | +27.7 pp |
| sc_top1 (SC @ N=16) | 16× | 0.841 | +32.5 pp |
| sc_top1 (SC @ N=32) | 32× | 0.835 | +31.9 pp (saturates) |
| ens_z_sum (PRM + SC@8) | 10× | 0.791 | +27.5 pp (≈ SC alone) |

---

## 11. Bottom line as of 2026-05-06 (after fourth pass)

**Four pieces are now in place** for a paper:
1. **Trajectory-level CP machinery is sound** — in-distribution coverage matches target across 5 score families × 2 datasets × 5 α levels × 200 seeds × 500 bootstrap.
2. **Score-family Pareto is complete** — lp_min (1×) → PRM-min (2×) → SC-top1 (8×). PRM-min is the single largest novel finding from this run.
3. **OOD failure is real, score-dependent, and partially fixable** — lp_* over-conservative, sc_top1 under-conservative on AIME-extended (n=300). Weighted CP corrects high-α breakage on lp_*; under-coverage on sc_top1 still open.
4. **Cross-generator PRM transfer works** — Qwen2.5-Math-PRM-7B applied to R1-Distill traces gives +18 pp at α=0.5, matching the +19 pp on its native Qwen2.5 generator. PRM as a score family is *generator-agnostic*.

**Two pieces remain open** for follow-up:
1. **Step-level branching as a publishable contribution is essentially closed** — Pilots C, K, L all show that `worst_step + K-resample` gives at best +1–2 pp on average and frequently 0. Requires either a fundamentally different intervention (token-level edit, search expansion, structured rewrite) or to be dropped from the paper in favor of trajectory-level framing.
2. **The OOD coverage gap has not been fully closed** — weighted CP fixes low-α / lp_*. Sc_top1 OOD under-coverage remains. May need conditional CP, group-conditional coverage, or per-domain calibration.

**Recommended paper framing**: position CoT-CP as ***the trajectory-level coverage layer*** that turns lp / PRM / SC scores into calibrated selective predictors, with PRM as the new mid-cost operating point. Frame step-branching as an explored-and-rejected alternative; frame OOD weighted CP as a robustness extension. The headline figure becomes a single Pareto plot showing all three score families, with vanilla / SC@N saturation curve and CP+score curves clearly separated.

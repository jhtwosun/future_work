---
verification_id: T1.1
title: "Per-step anytime-valid CP via Howard-Ramdas-McAuliffe-Sekhon (HRMS) e-process"
date: 2026-05-08
verifier: Claude (research-agent, V1-V6 protocol)
target_paper: CoT-CP (Calibrated Selective Generation for LLM Reasoning via Trajectory-Level CP)
context_files:
  - /home/nvidia/future/METHOD_AND_RESULTS.md
  - /home/nvidia/future/theorems/theorem1_trajectory_cp.md
  - /home/nvidia/future/theorems/theorem3_weighted_cp_discrete.md
  - /home/nvidia/future/theorems/THEOREM_REVIEW.md
  - /home/nvidia/future/literature/papers/2509.04733.md (CoVeR)
  - /home/nvidia/future/HGJ_review_feedback.md
status: COMPLETED
---

# T1.1 Verification: Per-step anytime-valid CP via HRMS e-process

> **One-line proposal**: Within a single CoT trace, maintain a running e-value
> over per-step PRM scores (or other s_t). Stop the trace early
> (commit / abort) when the e-value crosses a threshold. Use HRMS 2021
> time-uniform empirical-Bernstein bounds for non-i.i.d. step-level scores.
> Compose with Theorem 3 (empirical-PMF weighted CP for discrete scores
> under shift) for shift-aware version.

---

## V1 — Direct prior art (deep search)

I ran 12 targeted web searches and 5 arxiv WebFetch calls. Below is the
prior-art landscape ordered by overlap distance to T1.1.

### V1.A — Closest published competitors (ranked by overlap)

| # | Paper | arXiv / Venue | Date | Overlap with T1.1 | Differentiator |
|---|-------|---------------|------|-------------------|-----------------|
| **1** | **Sequential-EDFL ("Anytime-Valid Answer Sufficiency Certificates for LLM Generation via Sequential Information Lift")** | 2510.06478 | Oct-2025 | **Severe** — within-trace, anytime-valid, self-normalized empirical-Bernstein e-process | Uses **information lift** (log-LR vs skeleton baseline), NOT PRM step rewards. No conformal composition. No formal non-i.i.d. analysis. |
| 2 | CITE (Ota et al.) | 2605.05873 | May-2026 | Medium — anytime-valid e-process, but **trace-level** (across N samples), unique-mode certification | Inter-sample, not intra-sample. Mode-of-distribution target, not correctness. |
| 3 | MMC (Cordero-Encinar & Duncan) | 2510.17472 | Oct-2025 | Medium — Martingale Majority Certificate via Ville's inequality, **across N samples** | Cites Howard et al. 2021 explicitly. Trace-level only. |
| 4 | HALT-CoT (ICML 2025) | OpenReview CX5c7C1CZa | 2025 | Medium — entropy supermartingale, within-trace, Doob's optional-stopping theorem | Heuristic threshold, no anytime-valid e-process per se; assumes i.i.d. evidence per step. No empirical-Bernstein. |
| 5 | DeepConf (Meta FAIR) | 2508.15260 | Aug-2025 | Medium — online mode does within-trace early termination via group log-prob confidence | No formal anytime-valid guarantee; threshold from warmup-trace empirical quantile. Heuristic. |
| 6 | Thought Calibration | 2505.18404 (EMNLP-2025) | May-2025 | Medium — early-stop reasoning when "novel reasoning plateaus" via lightweight probes; built on Learn-then-Test (Bates 2021) | Probe-based confidence; not e-process / time-uniform. Plateau-detection heuristic, not HRMS-style. |
| 7 | ES-CoT | 2509.14004 | Sep-2025 | Low — within-trace, run-length test on step answers | Pure heuristic (run-length jump). No martingale guarantee. |
| 8 | REFRAIN ("Stop When Enough") | 2510.10103 | Oct-2025 | Low — sliding-window UCB bandit for stop threshold | UCB heuristic; no anytime-valid certificate. |
| 9 | CoVeR (Chen et al.) | 2509.04733 | Sep-2025 | Low — token-cluster step-level CP, PAC bound | NOT anytime-valid (PAC asymptotic). NOT e-process. Different unit (token-cluster, not semantic step). |
| 10 | ORCA (Online Reasoning Calibration) | 2604.01170 | 2026 | Low — TTT + conformal for reasoning, distributional shift | Trace-level CP with test-time training; no e-process / no within-trace anytime-validity. |
| 11 | A1 / ATTS | 2509.15148 | Sep-2025 | Low — asynchronous test-time scaling via CP | Speculative decoding / rejection sampling; not within-trace anytime-valid. |
| 12 | Logit-Entropy Adaptive Stopping | 2511.04654 | Nov-2025 | Low — entropy-based stop heuristic | No formal guarantee. |
| 13 | Catch Your Breath (CYB) | 2510.13879 | Oct-2025 | Low — supervised adaptive compute per token | Supervised loss, not anytime-valid sequential test. |
| 14 | Martingale Score (NeurIPS 2025) | 2512.02914 | Dec-2025 | Tangential — Bayesian-rationality martingale violations as unsupervised metric | Diagnostic, not stopping rule. |

### V1.B — Foundational references (the math we'd be invoking)

- **Howard, Ramdas, McAuliffe, Sekhon (Annals of Statistics 2021, "Time-uniform, nonparametric, nonasymptotic confidence sequences", 1810.08240)** — the empirical-Bernstein time-uniform confidence sequence at LIL rate. Already cited by MMC. Not yet applied within-trace.
- Howard, Ramdas, McAuliffe, Sekhon (1808.03204) — companion paper on time-uniform Chernoff bounds via nonnegative supermartingales.
- Ramdas, Grünwald, Vovk, Shafer (Stat. Sci. 2023) — "Game-theoretic statistics and safe anytime-valid inference" — survey.
- Foygel-Barber, Candès, Ramdas, Tibshirani (Annals 2023) — "Conformal prediction beyond exchangeability" — non-exchangeable CP, the natural composition partner.
- Bates, Candès, Lei, Romano, Sesia (Annals of Statistics 2023) — "Testing for outliers with conformal p-values" — selective FDR / outlier extension of CP. (Note: I confirmed the JRSSB-2023 paper the prompt referenced is actually Annals 2023; the relevant JRSSB-2023 work the prompt seems to intend is Bates et al. or the related "Risk-controlling prediction sets"-line.)
- Howard & Ramdas (Bernoulli 2022) — sequential estimation of quantiles with confidence sequences.

### V1.C — Concurrent NeurIPS / ICML / ICLR 2025-26 papers in this exact space

Searches turned up no NeurIPS/ICML/ICLR 2026 paper that *explicitly*
combines (i) within-trace per-step e-process, (ii) HRMS empirical-Bernstein
bound, (iii) PRM step-reward inputs, (iv) composition with conformal
prediction. **The closest prior work, Sequential-EDFL (2510.06478), hits
(i) and (ii) but explicitly does not do (iii) or (iv).**

### V1.D — Identified gap (precise statement)

> The **specific gap** that T1.1 fills is:
>
> **"A per-step anytime-valid e-process over a generic step-level
> confidence signal `s_t ∈ {PRM reward, log-prob, sub-claim verifier
> score}` for non-i.i.d. autoregressive CoT traces, composable with
> trajectory-level conformal prediction and with weighted-CP under
> score-distribution shift."**
>
> Sequential-EDFL fills the e-process half (information-lift channel
> only). MMC / CITE fill the across-N-samples e-process. CoVeR fills
> step-level CP without anytime-validity. **No published work composes
> these three pieces.**

This is a non-trivial gap, but the math (HRMS) is off-the-shelf.
Sequential-EDFL is the genuine threat — they got there first on the
within-trace anytime-valid skeleton, just with a different score channel.

---

## V2 — Academic value

### V2.A — Publishability tier

**Verdict: T1.1 is publishable as one section of CoT-CP at TMLR / ICLR;
NOT as a standalone TMLR theorem paper given Sequential-EDFL exists.**

Argument from venue norms:

- **TMLR standalone theorem paper (verdict: NO)**. TMLR does accept
  pure-theory contributions, but its bar is "claims well supported"
  rather than "novel." The dominant reviewer concern would be
  Sequential-EDFL (2510.06478, Oct-2025): same primitive (within-trace
  anytime-valid e-process), same mathematical machinery (empirical-Bernstein
  e-process). Distinguishing on "we use PRM scores instead of information
  lift" is too narrow to carry a standalone paper.
- **TMLR/ICLR section of CoT-CP (verdict: YES)**. As one of
  Theorems 4-5 inside the CoT-CP package, T1.1 adds a meaningful new
  capability: anytime-valid early stopping. It's exactly the kind of
  formal-result extension that lifts a paper from "applied CP demo" to
  "calibration framework with multiple coverage modes."
- **ICLR / NeurIPS workshop on safe-AI / sequential-decision (verdict: YES,
  natural fit)**. Could be carved off as a workshop paper if the main
  paper goes to TMLR.
- **Standalone NeurIPS / ICML main (verdict: NO)**. Bar is too high
  given prior art density (rows 1-5 in V1.A).

### V2.B — Comparison to Bates-Candès-Lei-Romano-Sesia (Annals 2023)

The prompt asked: was their FDR-via-conformal-p-value extension publishable
on its own? **Yes** — but it published in **Annals of Statistics** (highest
statistics venue), and the paper introduced a *fundamentally new construct*
(conformal p-values with PRDS structure for BH-FDR control). T1.1's
machinery is one step less novel: HRMS already exists; we're applying it
in a new domain. A direct analogue would be more like an *application paper*
in JCGS or a methods journal, or — more realistically for our community —
a section of CoT-CP.

### V2.C — Where CITE and MMC targeted

- **CITE (Ota et al., 2605.05873)**: appears to be aimed at NeurIPS / ICML
  main (anytime-valid SC majority is mainstream-ML novel; theoretical
  framing is clean).
- **MMC (Cordero-Encinar & Duncan, 2510.17472)**: looks aimed at
  ICLR 2026 / TMLR / NeurIPS Spotlight given the e-value + Ville
  rigor. They invest in TTT extension and exponential-tilting analysis
  to clear the novelty bar.

Implication for T1.1: standalone-publication bar requires either (a) novel
TTT-style extension, or (b) a structurally new theorem (not just HRMS
specialization). Without one of those, T1.1 is best framed as an internal
CoT-CP component.

---

## V3 — Feasibility & predicted performance

### V3.A — Technical tractability

**HRMS martingale construction for non-i.i.d. step scores is tractable
but non-trivial.** Specifically:

- HRMS Theorem 1 (their main empirical-Bernstein confidence sequence)
  *requires martingale-difference structure*: $X_t - \mathbb{E}[X_t \mid
  \mathcal{F}_{t-1}]$ should be a martingale-difference w.r.t. the
  filtration. For PRM step scores in autoregressive CoT, this means we
  must define an *adapted predictable* baseline (the conditional mean
  given prior steps) and bound the deviation around it.
- **Standard handling**: Doob decomposition. Write $S_t = M_t + A_t$
  where $A_t = \sum_{u\leq t} \mathbb{E}[S_u \mid \mathcal{F}_{u-1}]$
  (predictable component) and $M_t = S_t - A_t$ (martingale part).
  HRMS bounds apply to $M_t$. This is the textbook play (e.g., used in
  Howard-Ramdas 2022 quantile CS, and in Bercu-Touati 2008 EB-martingale
  bound).
- **Practical instantiation**: estimate $A_t$ online (say via a frozen
  baseline or running average); the residual martingale is what we
  e-process. Mineiro NeurIPS 2023 ("Time-uniform confidence bands for
  the CDF under nonstationarity") gives a worked example we can imitate.
- **Risk**: if the autoregressive structure is too strong (very
  predictable $A_t$), the martingale residual has tiny variance and
  the e-process accumulates evidence slowly. Empirical question we'd
  need to test.

**Empirical-Bernstein-for-martingales is a solved problem** — Bercu-Touati
2008, Howard et al. 2021 §5, Maurer-Pontil 2009 (i.i.d. case). The
formalism we need is in HRMS Lemma 6 / Theorem 2.

### V3.B — Compute budget vs. infrastructure

Infrastructure: 2× H100 NVL, 11 models, PRM800K calibration, 7 datasets.
Existing CoT-CP empirical machinery is ready (PRM scores already extracted
in Pilot M, E5v2; trajectory-level CP code in `experiments/src/`).

| Phase | Math | Empirical | Wall-clock |
|-------|------|-----------|-----------|
| Theorem statement + proof (HRMS specialization with non-i.i.d. handling) | 5 days | — | Week 1 |
| Empirical: e-process construction, threshold calibration | 2 days | Reuse PRM800K + E5v2 traces | Week 2 |
| Compute Pareto: matched-compute comparison vs. Approach A heuristic | — | 2 H100 days | Week 2-3 |
| Cross-dataset transfer (MATH → AIME, OlympiadBench, MMLU-Pro) | — | 1 H100 day | Week 3 |
| Composition with Theorem 3 (shift-aware version) | 3 days | 1 H100 day | Week 3-4 |
| Writing + figures | 4 days | — | Week 4 |

**Verdict**: 2-3 weeks math + 1 week empirical is **achievable but tight**.
Realistic budget is 4-5 weeks. Risk: getting non-i.i.d. martingale
construction right takes >5 days if Doob decomposition needs nontrivial
estimation.

### V3.C — Predicted empirical lift

Reference points:

- **CoT-CP existing Approach A (trajectory-level)**: MATH-500 + Qwen2.5-7B,
  prm_min α=0.5 → 0.707 kept_acc at 38% keep (vs. vanilla 70%, +19pt).
- **DeepConf online mode**: claims 84.7% token reduction with 99.9% AIME-2025
  on GPT-OSS-120B — but this is heuristic confidence-weighted aggregation,
  not anytime-valid.
- **Sequential-EDFL**: 21-31% token reduction "while maintaining acceptable
  quality" on dialogue/summarization tasks. Smaller signal.

**Predicted lift for T1.1**:
- Selective accuracy lift over Approach A at matched compute: **+1 to +3pp**
  (modest). The big win is the *anytime-valid certificate*, not the absolute
  accuracy number.
- Compute reduction at matched selective accuracy: **15-30%** of generation
  tokens, comparable to DeepConf / Sequential-EDFL.
- Best case: an *additional* operating point on the compute-Pareto curve
  with formal early-stop guarantee (the existing Approach A is post-hoc).

The expected lift **does not justify** T1.1 as a standalone paper but
**does justify** it as a section that adds anytime-validity to CoT-CP.

### V3.D — Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|---------|-----------|
| R1 | Non-i.i.d. Doob decomposition is technically correct but too loose to give meaningful e-process growth | Medium | High | Empirical pretest on n=20 traces before committing 4 weeks |
| R2 | Sequential-EDFL (2510.06478) reviewer-wedges us on "your work is just EDFL with PRM scores" | High | High | Frontload differentiation: PRM channel + conformal composition + non-i.i.d. theorem are jointly novel |
| R3 | PRM step rewards are noisy enough that empirical-Bernstein bound saturates and never crosses threshold | Medium | High | Use lp_min as backup; PRM as headline; ensemble fallback |
| R4 | Composition with Theorem 3 fails — weighted-CP under shift breaks the e-process martingale property | Medium | Medium | Fallback to vanilla CP (no shift correction) for T1.1 baseline; shift composition becomes "future work" |
| R5 | Wall-clock blows past 4 weeks | High | Low | Prioritize core within-trace result; defer cross-dataset shift composition to v2 |

---

## V4 — Incremental vs. structural novelty

### V4.A — The "Incremental" case (steel-manned)

> "T1.1 is just CITE/MMC machinery applied to within-trace data. The
> mathematical engine — HRMS 2021 empirical-Bernstein e-process — is
> 5 years old. Sequential-EDFL (Oct-2025) already did the within-trace
> anytime-valid e-process. The only original act is swapping
> 'information lift' for 'PRM reward'. That's a parameter choice, not
> a theorem. A NeurIPS reviewer will say: 'cite EDFL, run it, write a
> two-paragraph remark; this is not novel.'"

Strength of this case: **substantial**. Sequential-EDFL alone
neutralizes 60% of the originality claim. The HRMS specialization
genuinely is textbook-level mathematics.

### V4.B — The "Structural" case (steel-manned)

> "Within-trace anytime-valid is structurally a new problem regime.
> EDFL's information-lift channel is fundamentally different from
> PRM step rewards: information lift compares the model to its own
> skeleton baseline, while PRM rewards measure correctness signal from
> an external verifier. The composition with Theorem 3 (weighted-CP for
> discrete scores under shift) is where genuine new theory lives —
> nobody has shown that an empirical-Bernstein e-process composes with
> empirical-PMF density-ratio reweighting while preserving time-uniform
> validity. The non-i.i.d. handling via Doob decomposition over LLM
> step structure also requires new lemmas because the conditional-mean
> baseline must be estimated online from a frozen / non-stationary
> generator. These are real theorems, not parameter swaps."

Strength of this case: **moderate**. The composition-with-shift theorem
*is* genuinely new if we can prove it. The non-i.i.d. handling is
nontrivial but standard machinery in sequential analysis (Doob is 1953).

### V4.C — Verdict on which side wins

**Structural case wins, but narrowly, conditional on delivering the
composition theorem.**

Specifically: if we deliver only "HRMS within-trace on PRM" we lose to
EDFL. If we deliver "HRMS within-trace on PRM + composition with
weighted-CP under shift, with both finite-sample and time-uniform
guarantees" we have a non-trivial novel contribution.

### V4.D — Single most novel theorem candidate

> **Theorem 4 (proposed, T1.1 headline)**: *Time-uniform CP coverage
> under score-distribution shift via composed e-process and empirical-PMF
> reweighting.*
>
> Let $\{S_t\}_{t \geq 1}$ be an adapted step-level score process under
> a non-stationary autoregressive distribution $P$, with predictable
> Doob decomposition $S_t = M_t + A_t$. Let $\hat w(\cdot)$ be the
> empirical-PMF density-ratio estimator from Theorem 3. Define the
> shift-corrected e-process
> $$E_t^{\rm wgt} := \prod_{u=1}^{t} \exp\!\left( \lambda_u \hat w(S_u)\,(M_u - A_u) - \psi_u(\lambda_u) \right)$$
> where $\psi_u$ is the empirical-Bernstein cumulant adjustment from
> HRMS 2021 Theorem 2.
>
> **Claim**: For any data-driven stopping time $\tau$ and any $\delta \in
> (0, 1)$, conditional on $Y = 1$,
> $$\Pr_{\rm test}\!\left[ \bar S_\tau \geq \hat q^{\rm wgt}_\alpha \right]
> \geq 1 - \alpha - \frac{|\mathcal{S}|}{2}\sqrt{\frac{\log(2|\mathcal{S}|/\delta)}{2\min(n_{\rm cal}, n_{\rm test})}} - \frac{1}{n_+ + 1}.$$

This is the single theorem that, if delivered, justifies T1.1 in CoT-CP
as more than an EDFL clone. It composes Theorem 3's DKW slack with
HRMS time-uniformity. Verification of correctness is non-trivial — the
empirical-PMF reweighting must preserve the predictable variation
condition that HRMS's empirical-Bernstein bound requires.

---

## V5 — Cross-model verification: hardest reviewer objection

**Hardest single NeurIPS objection** (stated precisely):

> *"Reviewer 2: This paper's Theorem 4 is essentially Sequential-EDFL
> (Liu et al., 2510.06478, Oct-2025) with PRM rewards substituted for
> information lift. The composition with Theorem 3 is straightforward —
> reweighting an e-process by a bounded function preserves the
> supermartingale property under the same predictable filtration; this
> is a one-line corollary of Howard et al. 2021 §3.4, not a new theorem.
> The non-i.i.d. handling via Doob decomposition is standard
> sequential-analysis machinery (Bercu-Touati 2008, Howard-Ramdas 2022).
> The empirical lift over the trajectory-level baseline (+1 to +3pp at
> matched compute) is below the noise floor of bootstrap CIs in §5.
> I recommend reject; the contribution is incremental."*

### CoT-CP's best response

Three-pronged defense:

1. **EDFL operates on a fundamentally different score channel.**
   Information lift is *self-referential* (LM vs. its own skeleton),
   so EDFL's guarantee is about *answer sufficiency*, not correctness
   (the EDFL paper explicitly disclaims correctness control). Our
   PRM channel is *externally verified*, giving correctness-aligned
   guarantees. This is not a parameter swap; it's a guarantee semantic
   change.

2. **The shift-aware composition is the genuine theorem.** While
   reweighting an e-process by a *fixed* bounded function is a
   one-liner, our $\hat w$ is *estimated from the same data stream*
   used for the e-process — this breaks the standard predictable
   filtration and requires either sample splitting or a new
   joint-validity argument. Theorem 4's proof must show this.

3. **Empirical lift framing**. Concede the +1-3pp absolute number is
   modest; reframe as "**anytime-valid certificate at matched compute
   parity with strong heuristic baselines**" — the formal-guarantee
   delta is the contribution. Show DeepConf and EDFL fail to deliver
   any per-instance certificate even when they match accuracy.

A decisive response also requires showing, in the experiments, a regime
where EDFL's information-lift channel underperforms PRM-based T1.1
(e.g., on math problems where the model is locally fluent but globally
wrong — exactly the failure mode information-lift is blind to).

---

## V6 — Final verdict

### V6.A — Recommendation

**KEEP T1.1 at Tier 1, but REPOSITION as a CoT-CP section, not a
standalone paper, and condition acceptance on delivering the shift-aware
composition theorem (Theorem 4 above).**

Risk-mitigated commitment: 4-week timeline with two go/no-go gates.

### V6.B — 4-week timeline with go/no-go points

| Week | Milestone | Go/No-go criterion |
|------|-----------|-------------------|
| **Week 1** | Theorem 4 statement + proof draft (within-trace HRMS + Doob + non-i.i.d. handling), no shift composition yet. Empirical pretest: e-process growth on 20 traces using PRM_min. | **GATE 1**: Does the e-process accumulate evidence at expected rate (cross threshold within median trace length)? **If NO, demote to Tier 2.** |
| **Week 2** | Composition-with-Theorem-3 attempt. Sample-splitting or joint-validity argument. Empirical: matched-compute Pareto vs. Approach A heuristic on MATH-500 + Qwen2.5-7B. | **GATE 2**: Does Theorem 4 close (provable, not just plausible) AND empirical lift ≥ +1pp at matched compute? **If NO on Theorem 4 only, fallback to within-trace-only result. If NO on both, demote.** |
| **Week 3** | Cross-dataset experiments: AIME, OlympiadBench, MMLU-Pro. Composition with weighted-CP under shift (MATH→AIME). | Soft check: does empirical coverage stay within ±3pp of theoretical bound? |
| **Week 4** | Writing + figures + framing as CoT-CP §6 (anytime-valid extension). Reviewer-defense framing of EDFL differentiation. | Final paper draft ready. |

### V6.C — Repositioning details

- **Title change in CoT-CP**: §6 "Anytime-valid trajectory CP via
  per-step e-process" with explicit subsection on Sequential-EDFL
  differentiation.
- **Citation strategy**: Cite EDFL prominently in §6.1 (the closest
  prior art); MMC and CITE in §6.2 (related but trace-level); CoVeR
  in §6.3 (step-level CP without anytime validity).
- **Don't oversell**: T1.1 should be framed as "the principled way to
  do early-stop CoT under our framework" rather than "a fundamental
  new theorem." Keep Theorem 4 in the appendix unless we get the
  shift-aware composition genuinely tight.
- **If GATE 1 fails**: The fallback is to keep T1.1 as a brief
  "future work" section in CoT-CP, citing EDFL. No paper damage.
- **If GATE 2 fails on Theorem 4**: Publish the within-trace-only
  result as a workshop paper alongside CoT-CP main; delete §6 from
  CoT-CP main paper.

### V6.D — Bottom line

T1.1 is **technically tractable, mathematically sound, and structurally
novel only conditional on the composition-with-shift theorem**. The
biggest threat is Sequential-EDFL having shipped first; the biggest
opportunity is showing PRM channels and shift-aware composition
genuinely differentiate. Worth the 4-week investment **if** Gate 1
clears in Week 1.

Confidence on KEEP-with-mitigation: **0.65**. Confidence on
DEMOTE-to-Tier-2: **0.20**. Confidence on REPOSITION-only-no-Theorem-4:
**0.15**.

---

## Appendix A — Key papers verified (with arXiv IDs and one-line tags)

| arXiv | Tag |
|-------|-----|
| 2510.06478 | Sequential-EDFL — primary threat (within-trace anytime-valid e-process via information lift) |
| 2605.05873 | CITE (Ota et al.) — anytime-valid SC across N samples |
| 2510.17472 | MMC (Cordero-Encinar & Duncan) — Martingale Majority Certificate across N samples |
| 1810.08240 | HRMS 2021 — foundational empirical-Bernstein time-uniform CS |
| 1808.03204 | Howard et al. 2018 — companion supermartingale Chernoff bounds |
| 2509.04733 | CoVeR — token-cluster step-level CP with PAC bound |
| 2508.15260 | DeepConf (Meta FAIR) — within-trace heuristic stop |
| 2505.18404 | Thought Calibration — Learn-then-Test for early-stop reasoning |
| 2509.14004 | ES-CoT — within-trace run-length heuristic |
| 2510.10103 | REFRAIN — UCB bandit early-stop heuristic |
| 2604.01170 | ORCA — TTT + conformal for reasoning |
| 2509.15148 | A1/ATTS — async test-time scaling via CP |
| 2104.08279 | Bates-Candès-Lei-Romano-Sesia — outlier conformal p-values (Annals 2023) |
| 2208.02814 | Conformal Risk Control (Angelopoulos et al.) — context for risk-control extension |
| 2512.02914 | Martingale Score (NeurIPS 2025) — diagnostic, tangential |

## Appendix B — Open questions for next iteration

1. **Sample-splitting cost for shift-aware composition**: If we split
   data 50/50 between e-process residual estimation and $\hat w$
   estimation, does the resulting bound stay within tolerable slack
   on our $n_+ \in [200, 400]$ regime? Need numerical check.
2. **Empirical predictability of $A_t$**: How predictable is the
   conditional-mean baseline of PRM rewards across CoT steps, on our
   datasets? If $A_t \approx \mathrm{const}$, the e-process collapses
   to i.i.d. case and the contribution shrinks. Pre-test required.
3. **Differentiation against EDFL on math-correctness regime**: Design
   one experiment where information lift is high but answer is wrong
   (locally fluent reasoning, globally wrong) — to demonstrate PRM
   channel necessity.
4. **Power-vs-validity trade-off for HRMS at our $n$**: HRMS 2021
   constants are known to be conservative. Could a Mineiro-style
   self-normalization tighten the bound enough to deliver +5pp
   instead of +1-3pp?

---

*End of T1.1 verification. Ready for human-lead review and gate decisions.*

# FINAL Verification Summary — CoT-CP Tier-1 Picks (T1.1–T1.10)

> **Author**: Verifier (Claude Opus 4.7, 1M ctx).
> **Date**: 2026-05-08.
> **Inputs**: `T1_1_verification.md` … `T1_6_verification.md`, `T1_9_verification.md`, `T1_10_verification.md` (this directory).
> **Cross-model status**: `mode: all` per CLAUDE.md, but inference token is `sk-PLACEHOLDER`; all per-pick verdicts produced single-model (Claude Opus 4.7). Verifier-model re-check should be invoked at the orchestrator level before paper submission.

---

## §1 — Executive summary

After eight V1–V6 verifications across the Tier-1 pick list, **the most defensible CoT-CP submission is a TMLR or ICLR-2026 framework paper anchored on the trajectory-level CP machinery (Theorems 1–3) plus three load-bearing extensions: an anytime-valid per-step e-process (T1.1), a Pearl-causal earliest-bad-step intervention with a minimal-intervention-point theorem (T1.4), and local CP via SBERT+KNN as an OOD/robustness ablation with stretch composition theorem (T1.6).** Of the eight picks, six survive in some form (KEEP/REPOSITION/FOLD/CONDITIONAL); one is a defensive ablation that should ride along (T1.3); one should be cut entirely as a standalone contribution (T1.9, fold as one score-family rung). The biggest single risk to the paper is **Sequential-EDFL (arXiv 2510.06478)**, which preempts T1.1's within-trace anytime-valid e-process on a different score channel — survivable only by delivering the shift-aware composition theorem (Theorem 4 candidate). The biggest single opportunity is the **earliest-bad-step Pearl-causal theorem (T1.4 §V2.1)** which converts our existing Pilot C/K/L null result into a positive theoretical contribution that explains *why* worst-step branching fails. Honest 3-pick survival list (NeurIPS reviewer triage): **trajectory-CP framework + Theorem 3 PMF correction + T1.6 local CP** are what survive a "show me your three contributions" challenge — T1.1 and T1.4 are stretch theorems with positive but uncertain delivery odds.

---

## §2 — Per-pick verdict table

| Pick | Title | Verdict | One-line reason |
|---|---|---|---|
| **T1.1** | e-process per-step (HRMS) | **KEEP + REPOSITION** | Sequential-EDFL preempts the within-trace anytime-valid skeleton; survives only as a CoT-CP §6 sub-section conditional on delivering Theorem 4 (shift-aware composition with weighted-CP). 2 go/no-go gates in 4-week timeline. |
| **T1.2** | BH-FDR for per-step CP | **KEEP** (as section, not standalone) | No one has done BH on the per-step within-trace CP family; +1–3pp lift on long-CoT models. Methodological recalibration of Approach B; report both BH and BY (under arbitrary dependence). |
| **T1.3** | Diverse Beam Search at worst step | **REPOSITION** as defensive ablation | Wang-Zhou 2024 explicitly argues mid-trace branching < first-token branching; Pilots C/K/L already null. Run as a 1-paragraph §4 ablation with matched-compute baselines; do not pitch as contribution. |
| **T1.4** | Earliest-bad-step re-roll (Five-Whys / CPM / Pearl) | **GO conditional on Pearl-causal theorem** | The minimal-intervention-point theorem under faithfulness is the only framing that lifts T1.4 above an ablation row. Bundle with T1.6 + T3.1 (Qwen3-32B) for Week-1 execution. |
| **T1.5** | Pólya Look Back primitive | **KEEP as 2–3 day pilot, gated** | Forced-coverage 4-axis taxonomy is not yet operationalized as a CRC score family. 3-way decision gate after pilot: promote / demote-to-tiebreak / drop. Expected lift +0–3pp at α=0.3. |
| **T1.6** | Local CP via SBERT + KNN | **PROCEED — primary OOD ablation** | Sits in genuinely empty cell (trajectory-level + KNN-local + LLM + math reasoning OOD). Frame A (§6 Robustness ablation) is required; Frame B (composition theorem with Theorem 3) is ~50% odds, high-upside stretch. |
| **T1.9** | Fermi cross-check | **REJECT standalone, FOLD as score rung** | EVoSS (Piehl 2025, arXiv 2509.18565) is the symmetric construction; FermiEval+CP (Epstein 2025, arXiv 2510.26995) covers calibration. Drop entirely if AIME lift < 1.5pp; one rung in score-family ladder otherwise. |
| **T1.10** | Per-failure-mode kept-accuracy (M&M / NTSB / AAR) | **KEEP as §6 sub-section, with N=300 + audit** | ErrorAtlas (2601.15812) and FLARE own the *taxonomy* axis — reuse, don't rebuild. Novel cell is per-mode kept-accuracy across the score-family ladder (6×8 matrix). Drop to footnote if per-mode Δ < 10pp uniform. |

### Verdict legend distribution

- **KEEP**: T1.1 (with reposition), T1.2, T1.5 (gated), T1.10 → 4
- **REPOSITION / DEFENSIVE ABLATION**: T1.3 → 1
- **CONDITIONAL GO**: T1.4 (theorem-gated), T1.6 (frame A required, frame B stretch) → 2
- **REJECT / FOLD**: T1.9 → 1

---

## §3 — New prior art surfaced across all 8 verifications

The following 15 papers were not in our pre-verification 30-paper bibliography. Listed in approximate order of impact on the paper's framing:

### A. Direct competitors (highest reviewer threat)

1. **Sequential-EDFL — *Anytime-Valid Answer Sufficiency Certificates for LLM Generation via Sequential Information Lift*, arXiv:2510.06478** (Oct 2025).
   Within-trace anytime-valid e-process on information-lift channel. **Direct preempter** of T1.1's skeleton.

2. **EVoSS — Piehl et al., *Solving Math Word Problems Using Estimation Verification and Equation Generation*, arXiv:2509.18565** (Sep 2025).
   Symmetric to T1.9: rigorous → estimate (T1.9 was estimate → rigorous). Same accept/reject mechanics; α ∈ {40, 50}% tolerance. **Kills T1.9 standalone.**

3. **FermiEval — Epstein et al., *LLMs are Overconfident*, arXiv:2510.26995** (Oct 2025).
   Pairs Fermi estimation with conformal prediction for interval calibration. **Compresses the calibration novelty surface for T1.9.**

4. **ErrorAtlas — Sundar et al., *ErrorMap and ErrorAtlas: Charting the Failure Landscape of LLMs*, arXiv:2601.15812** (Jan 2026).
   17-category hierarchical taxonomy from 35 datasets × 83 LLMs × 7,000 sampled failures. **Owns the taxonomy axis for T1.10.** Reuse, don't rebuild.

5. **FLARE — Patel et al., *FLARE: An Error Analysis Framework for Diagnosing LLM Classification Failures*, ACL-OMMM 2025** (aclanthology.org/2025.ommm-1.4).
   7-cat taxonomy with per-mode prevalence; demonstrates the per-mode breakdown methodology T1.10 was proposing.

### B. Strong methodological precedents (affect framing)

6. **PARC — Mukherjee et al., *Premise-Augmented Reasoning Chains*, arXiv:2502.02362, ICML 2025**.
   DAG-of-premises for *accumulation errors*; localizes root steps. **Direct kin of T1.4** (PARC = trigger; T1.4 = trigger + intervention).

7. **VPPO / Save-the-Good-Prefix, arXiv:2601.18984** (Jan 2026).
   Uses PRM as first-error detector in RL training. Validates T1.4's earliest-step framing on training side.

8. **MMC — Cordero-Encinar & Duncan, *Martingale Majority Certificate*, arXiv:2510.17472** (Oct 2025).
   Cross-sample anytime-valid via Ville's inequality. **Concurrent with T1.1's NeurIPS-track aim**.

9. **CITE — Ota et al., arXiv:2605.05873** (May 2026).
   Inter-sample anytime-valid e-process for SC majority. Adjacent but trace-level.

10. **Han et al. 2022, *Split Localized Conformal Prediction*, arXiv:2206.13092**.
    Closer split-CP precedent for T1.6 than Guan 2023 alone; missing from HGJ feedback citation list.

11. **SLCP — Lu, Foygel-Barber et al., *Stable Localized CP via Transduction*, arXiv:2605.01452** (2026).
    Targets the small-effective-K instability HGJ flagged for T1.6. Adopt as secondary row if pilot shows variance.

12. **Verification-First / Iter-VF — Wu-Yao, arXiv:2511.21734** (Nov 2025).
    Reverse-CoT regularizer; complementary to Pólya Look Back (T1.5).

### C. Crowding-the-edge papers (cite, do not pitch against)

13. **Less-is-More / Minimal Test-Time Intervention, arXiv:2510.13940**.
    High-entropy token = "critical step" for selective intervention; nearest *inference-time* analog of T1.4.

14. **Well Begun Half Done / Beginning Lock-in Effect, arXiv:2512.15274** (Dec 2025).
    Empirical: early-stage thoughts substantially constrain trajectory. Direct evidence for T1.4's cascade-source intuition.

15. **MAST / Cemri et al., *Why Do Multi-Agent LLM Systems Fail?*, arXiv:2503.13657** (2025).
    150-trace × 6-annotator × κ=0.88 LLM-failure-mode methodology — the standard against which T1.10's labeling pipeline is measured.

### Honorable mentions (also surfaced)

- *Failure Modes in LLM Systems* (Vinay, arXiv:2511.19933) — 15-mode system-level taxonomy.
- *When the Chain Breaks (ReasonDiag)*, arXiv:2603.21286 — interactive CoT-error diagnosis tooling.
- *Demystifying Errors in LLM Reasoning Traces*, arXiv:2512.00215 — 9-cat code-reasoning taxonomy.
- *PedCoT*, arXiv:2405.06705 — pedagogical mistake-finding (Bloom cognitive model).
- *DeepSeekMath-V2*, arXiv:2511.22570 — RL-trained verifier head.
- *Learning to Self-Verify*, arXiv:2602.07594 — strongest evidence for "fake verification" risk in T1.5.
- *DS-CP — Lin et al.*, arXiv:2510.05566 — full-set reweighting baseline complementary to T1.6.
- *Cherian-Gibbs-Candès 2024*, arXiv:2406.09714 — embedding-conditioned quantile regression for LLMs.

---

## §4 — Updated 4-week implementation roadmap

Reorganized to reflect post-verification realities. Italicized rows are gated; bold rows are the load-bearing critical path.

### Week 1: Foundations — earliest-bad-step + local CP + Theorem 4 draft

| Day | Task | Owner | Cost |
|---|---|---|---|
| Day 1 | **Draft Pearl-causal "minimal intervention point" theorem (T1.4 §V2.1 option 1)** | Theorist | 1 day |
| Day 1 | T1.6 SBERT+FAISS pipeline implementation (reuse `experiments/src/`) | Engineer | 0.5 day |
| Day 2 | T1.4 trigger-threshold separate calibration (held-out re-roll-improvement label) | Engineer | 0.5 day |
| Day 2–3 | **T1.6 sweep K ∈ {30, 45, 60, 90} × MATH→AIME shift × {SBERT, MathBERT}** | Engineer | 6–10 GPU-hr |
| Day 3 | **T1.4 cascade-depth-stratified sweep** (depth=1 vs depth≥2 buckets, pre-registered) | Engineer | 5–8 GPU-hr |
| Day 4 | **T1.1 GATE 1**: Empirical pretest of e-process growth on n=20 PRM_min traces | Engineer | 0.5 day |
| Day 5 | T1.6 Frame B: composition-theorem prototype on paper | Theorist | 1 day |

**Week 1 deliverables**: T1.4 + T1.6 empirical results, Theorem 4 first draft, T1.1 Gate-1 verdict.

### Week 2: Theorems + extensions — Theorem 4 closure, T1.5 pilot, T1.10 labeling

| Day | Task | Owner | Cost |
|---|---|---|---|
| Day 6–7 | **T1.1 GATE 2**: Composition-with-Theorem-3 attempt (sample-splitting / joint-validity) | Theorist | 2 days |
| Day 6 | T1.5 Pólya prompt + parser implementation; MATH-500 calibration run | Engineer | 1 day |
| Day 7 | T1.5 LR+ at β ∈ {0.3, 0.5, 0.7}; per-axis precision/recall on calibration | Engineer | 0.5 day + 2 GPU-hr |
| Day 8 | **T1.5 Day-3 decision gate** (promote / demote / drop) | Lead | 0.5 day |
| Day 8–9 | T1.10 trace selection (300 stratified) + dual-judge labeling pass (Claude + GPT-4) | Engineer | 1.5 day + 5 GPU-hr |
| Day 10 | T1.10 human-audit κ on 50 traces; 6×8 kept-acc matrix + bootstrap CIs | Lead + Engineer | 1 day |

**Week 2 deliverables**: Theorem 4 v1 (or fallback), T1.5 verdict, T1.10 6×8 matrix.

### Week 3: BH/BY recalibration, Fermi gate, defensive ablations

| Day | Task | Owner | Cost |
|---|---|---|---|
| Day 11–12 | **T1.2 Theorem 4-bis**: per-step BH coverage under PRDS + BY fallback | Theorist | 2 days |
| Day 12 | T1.2 rerun Approach B with both BH and BY thresholds on §11 cells | Engineer | 1–2 GPU-hr |
| Day 13 | T1.9 Fermi cross-check 3–5 day prompt experiment on AIME (n=933) + MATH-500 + SC@8 | Engineer | 3–5 GPU-hr |
| Day 13 | **T1.9 V6.3 go/no-go gate** (drop / mention / promote based on AIME Δ vs sc_top1) | Lead | 0.5 day |
| Day 14–15 | T1.3 DBS defensive ablation: K=4 worst-step + λ ∈ {0.1, 0.3, 0.5, 1.0} + Δ ∈ {Hamming, SBERT, hidden-state} | Engineer | 6–12 GPU-hr |
| Day 15 | **T1.1 cross-dataset experiments**: AIME, OlympiadBench, MMLU-Pro (if Gate 2 passed) | Engineer | 1 GPU-day |

**Week 3 deliverables**: BH/BY columns added to Table 11; Fermi verdict; DBS ablation row; T1.1 cross-dataset coverage check.

### Week 4: Writing + figures + paper assembly

| Day | Task | Owner | Cost |
|---|---|---|---|
| Day 16 | Compose figures: 6×8 per-mode bar chart (T1.10), K-sensitivity curve (T1.6), e-process growth (T1.1), cascade-depth diagnostic (T1.4) | Designer | 1 day |
| Day 17–18 | Paper §1–3 (intro, framework, theorems): Theorems 1–3 stable, Theorem 4 (T1.1) and proposition (T1.4) integrated | Writer | 2 days |
| Day 19 | Paper §4–5 (score-family ladder, experiments) | Writer | 1 day |
| Day 20 | Paper §6 (Robustness, T1.6 + T1.10 + DBS ablation), §7 (limitations, T1.9 fold or footnote) | Writer | 1 day |

**Week 4 deliverables**: Submission-ready draft. Reviewer-defense framing of EDFL differentiation in §6.1; explicit ErrorAtlas citation in §6 T1.10 sub-section; explicit EVoSS+FermiEval citation if T1.9 lands as a rung.

### Total compute budget

| Pick | GPU-hours |
|---|---|
| T1.1 (e-process) | 1 H100-day cross-dataset = ~24 hr |
| T1.2 (BH-FDR rerun) | 1–2 hr |
| T1.3 (DBS) | 6–12 hr |
| T1.4 (earliest-bad-step) | 5–8 hr |
| T1.5 (Pólya pilot) | 2 hr |
| T1.6 (local CP) | 6–10 hr |
| T1.9 (Fermi) | 3–5 hr |
| T1.10 (per-mode) | 5 hr (+ API) |
| **Total** | **~50–66 GPU-hr** (~1 H100-day on 2× H100 NVL) |

Within available budget.

---

## §5 — Cross-cut: theorems-needed vs pure-ablations

### Picks that NEED a theorem before submission

| Pick | Theorem requirement | Status |
|---|---|---|
| **T1.1** | **Theorem 4 (shift-aware composition)**: Time-uniform CP coverage under score-distribution shift via composed e-process and empirical-PMF reweighting. | Sketched (T1.1 §V4.D). Provable but non-trivial — sample-splitting argument needed. |
| **T1.2** | **Theorem 4-bis (per-step BH)**: BH step-up valid on per-step CP family under PRDS, with BY fallback. | One-page corollary of cfBH + BY 2001. **Easy.** |
| **T1.4** | **Proposition (Pearl-causal minimal intervention point)**: Earliest threshold violation = topologically-minimal intervention point in trajectory SCM under faithfulness; weakly Pareto-dominates worst-step under score-monotonicity + per-step CP. | Sketched. Provable under faithfulness assumption. **Medium difficulty.** |
| **T1.6** (stretch) | **Conjecture (Local Empirical-PMF Weighted CP)**: Local A1 + DKW-rate slack composes into combined coverage bound. | Sketched (T1.6 §V4.2). **~50% odds; high upside.** |

### Picks that are PURE ablations (no theorem)

| Pick | Empirical contribution | Headline-or-section |
|---|---|---|
| T1.3 | Matched-compute K=4 DBS test at lp-min worst step + λ-sweep + Δ-sweep | §4 ablation row strengthening Pilot C/K/L null |
| T1.5 | Pólya 4-axis CRC score family + LR+ ranking position | §4.x rung if SNR-distinct from prm_min, else §6 negative ablation |
| T1.9 | Fermi cross-check rung + AIME differential at fixed sc_top1 baseline | §4.x rung if Δ > 1.5pp, else drop |
| T1.10 | 6×8 per-mode kept-accuracy matrix across score-family ladder | §6 sub-section if any ΔKA > 25pp survives CI, else footnote |

### Theorem coverage map

```
                     |  Has theorem?  |  Has empirical?  |  Need theorem to publish?
T1.1 (e-process)     |     ★ NEW      |       ★          |   YES (Theorem 4)
T1.2 (BH-FDR)        |     ★ NEW      |       ★          |   YES (Theorem 4-bis, easy)
T1.3 (DBS)           |       —        |       ★          |   NO (ablation only)
T1.4 (earliest-bad)  |     ★ NEW      |       ★          |   YES (Proposition)
T1.5 (Pólya)         |       —        |       ★          |   NO (Theorem 1+2 absorb it)
T1.6 (Local CP)      |  ☆ stretch     |       ★          |   NO for §6; YES for headline
T1.9 (Fermi)         |       —        |       ★          |   NO (just cite EVoSS/FermiEval)
T1.10 (Per-mode)     |       —        |       ★          |   NO (diagnostic only)
```

**Summary**: The paper has 3 *new* theorems (T1.1, T1.2, T1.4) on top of existing Theorems 1–3, plus one stretch theorem (T1.6 Frame B). If all 3 new theorems land, this is a 6-theorem paper. If only T1.2 lands, it is a 4-theorem paper with strong empirical extensions. **Minimum acceptable theoretical content: Theorem 4-bis (T1.2) closure + at least one of T1.1/T1.4 sketched proofs.**

---

## §6 — Final tier reorganization

### Original tier (pre-verification, from `HGJ_experiment_ideas.md`)

```
Tier 1 (paper-headline candidates):  T1.1, T1.2, T1.3, T1.4, T1.5, T1.6, T1.9, T1.10
```

### Updated tier (post-verification)

```
Tier 1 — Headline / Theorem-driven (paper-load-bearing):
  T1.6 (Local CP)            ⭐⭐⭐  — Frame A required, Frame B stretch theorem
  T1.4 (Earliest-bad-step)   ⭐⭐   — conditional on Pearl-causal proposition
  T1.1 (e-process)           ⭐⭐   — conditional on Theorem 4 closure (Gate 2)
  T1.2 (BH-FDR)              ⭐⭐   — easy theorem; section-level

Tier 2 — Section-level / Diagnostic (paper-supporting):
  T1.5 (Pólya Look Back)     ⭐    — gated 2-3 day pilot; promote/demote/drop
  T1.10 (Per-failure-mode)   ⭐    — §6 sub-section with N=300 + audit

Tier 3 — Defensive ablation (reviewer-armor):
  T1.3 (DBS at worst step)   ★    — §4 paragraph-level ablation; expected null

Tier 4 — Cut from main paper:
  T1.9 (Fermi cross-check)   ✗    — REJECT standalone; FOLD as 1 rung iff Δ > 1.5pp;
                                    otherwise drop entirely. EVoSS + FermiEval preempt.
```

### Headline tier delta

- **T1.1 demoted from Tier-1-standalone to Tier-1-section** (Sequential-EDFL preempts).
- **T1.4 elevated to Tier-1-section** with theorem requirement (was previously framed as ablation in HGJ Idea 1.1).
- **T1.5 demoted to Tier-2-gated** (verbalized-verification quality risk, redundancy with prm_min).
- **T1.10 demoted to Tier-2-diagnostic** (ErrorAtlas owns the taxonomy axis).
- **T1.3 moved to Tier-3-defensive** (Wang-Zhou 2024 + Pilot null make positive lift unlikely).
- **T1.9 demoted to Tier-4-cut** (EVoSS + FermiEval).

### Reviewer-side reading

The post-verification paper now reads as:
- 1 framework (trajectory-level CP for LLM CoT)
- 3 existing theorems (split-CP coverage, score-family LR+ Pareto, empirical-PMF weighted CP)
- 3 new theorems (BH/BY, Pearl-causal minimal intervention, e-process composition under shift)
- 6 score families (lp_min, prm_min, sc_top1, + Pólya rung, + Fermi rung iff Δ>1.5pp, + 4th if T1.5 promotes)
- 2 OOD-robustness rows (Theorem 3 PMF + T1.6 local CP, both with empirical ablation)
- 2 diagnostic figures (T1.4 cascade-depth-stratified, T1.10 6×8 per-mode kept-accuracy)
- 1 defensive ablation paragraph (T1.3 DBS null)

This is a credible TMLR submission with 4 main contributions. NeurIPS/ICLR-main would require Theorem 4 (T1.1) closure to be one of the three or four headline contributions.

---

## §7 — Honest 3-pick survival list (NeurIPS reviewer triage)

> *"Reviewer 2: I read the paper carefully but only have time to comment on the three contributions you consider most important. Which three?"*

After eight V1–V6 verifications, the survival list — under the assumption that a NeurIPS reviewer cuts everything else as "ablation" or "limited" — is:

### Pick #1: **Trajectory-level CP framework for LLM CoT (Theorem 1)**

The framework itself — a measurable score aggregator $\phi$ over per-step scores composing with split-CP — is the most defensible single contribution. Theorem 1 (CoT-CP coverage at $1 - \alpha - 1/(n_+ + 1)$) is the "rolled-up" version of all our claims. Without this, none of T1.1–T1.10 has anything to compose with.

### Pick #2: **Empirical-PMF Weighted CP under discrete-score shift (Theorem 3)**

The Theorem 3 finite-sample DKW-slack bound is the best-positioned new theorem we have. It is provably tighter than naive vanilla CP under score-only shift (Assumption A1). Composes with T1.6 (Frame B stretch) and with T1.1 (Theorem 4). It is the most *finished* of the new theorems and the empirical evidence (MATH→AIME shift result in §2.6 of `METHOD_AND_RESULTS.md`) is decisive.

### Pick #3: **Local CP via SBERT+KNN as the OOD/robustness extension (T1.6 + Frame B if landed)**

T1.6 is the highest-value OOD ablation in the post-verification roadmap. If Frame B (composition theorem with Theorem 3) lands, this picks up a 4th headline-grade theorem; if not, the empirical Frame A result is still the strongest single-row entry in §6 Robustness. Critically, T1.6 sits in a *genuinely empty cell* of the prior-art matrix (trajectory-level + KNN-local + LLM + math-reasoning OOD), which is rare among the eight Tier-1 picks.

### Why not T1.1 or T1.4 in the top 3?

- **T1.1** is heavily preempted by Sequential-EDFL on the within-trace anytime-valid skeleton. The novelty lives entirely in the shift-aware composition theorem (Theorem 4), which has go/no-go gates and ~65% delivery confidence per V6.D of `T1_1_verification.md`. **If it lands, T1.1 is co-headline; if not, it is a §6 sub-section.** Survival list at NeurIPS-bar must list things we already have, not things we hope to prove.
- **T1.4** has a similar gating: its standalone value as an ablation is incremental; the Pearl-causal minimal-intervention-point theorem is the only path to paper-grade. If the theorem is provably clean (V2.1 option 1) and the cascade-depth diagnostic shows monotonicity, T1.4 promotes to top-3. As of 2026-05-08, this is uncertain.
- **T1.2 (BH-FDR)** is a well-understood section-level recalibration; it is "almost free" but not contribution-defining at NeurIPS bar.

### What this means for the writing pass

The paper's `## Contributions` section should read:

> "We make three primary contributions: (1) a trajectory-level conformal prediction framework for chain-of-thought reasoning that composes any measurable score aggregator with split-CP under finite-sample coverage (Theorem 1); (2) an empirical-PMF weighted CP for discrete trajectory scores under distribution shift, with a finite-sample DKW-slack guarantee (Theorem 3); (3) a local conformal prediction extension via SBERT+KNN that achieves approximate per-prompt conditional coverage on math-reasoning OOD (§6, Theorem 4 if Frame B lands)."

Then list T1.1, T1.2, T1.4 as *secondary* contributions in §1 — "we additionally introduce ..." — so that if a reviewer downscopes the paper to its 3 strongest, the surviving 3 are the ones above.

---

## §8 — Risk register (consolidated across all 8 picks)

| Risk | Probability | Impact | Mitigation | Source verification |
|---|---|---|---|---|
| Sequential-EDFL reviewer-wedges T1.1 | High | High | Frontload differentiation: PRM channel + composition + non-i.i.d. theorem are jointly novel | T1.1 §V5 R2 |
| ErrorAtlas reviewer-wedges T1.10 | High | Medium | Reuse ErrorAtlas-17, don't rebuild; novelty cell is per-mode kept-acc × score-family | T1.10 §V5 |
| Pearl-causal proposition (T1.4) does not close cleanly in 1–2 days | Medium | High | Fall back to "ablation row only" framing; load-bearing only if proposition lands | T1.4 §V6.4 |
| T1.6 K-choice variance dominates lift signal | Medium | Medium | Adopt SLCP stabilization (Lu 2025); sweep K ∈ {30,45,60,90} | T1.6 §V3.3 |
| T1.5 verbalized verification is "fake verification" (per 2602.07594) | Medium | High | 3-way decision gate after Day-3 pilot; per-axis precision/recall on calibration | T1.5 §V3.2 bear case |
| T1.10 N=100 underpowered → CIs straddle differential claims | High | Medium | Scale to N=300 + bootstrap CIs + human-audit κ on 50 | T1.10 §V5 |
| T1.2 PRDS verification on auto-regressive trace fails empirically | Medium | Low | Fall back to BY (factor 1/H_T loss); still 13× tighter than Bonferroni at T=60 | T1.2 §V3.4 |
| EVoSS reviewer-wedges T1.9 | Confirmed | Confirmed | Drop standalone; fold as 1 rung iff Δ > 1.5pp on AIME | T1.9 §V5 R1 |
| T1.3 DBS gives net negative lift on AIME | Medium | Low | Pre-frame as "strengthens §4 null"; matched-compute baselines included | T1.3 §V6 |
| Compute budget overrun (>66 GPU-hr) | Low | Low | T1.3 + T1.9 + T1.5 are the discretionary picks; cut in that order if needed | This summary §4 |
| Cross-model verifier never invoked (sk-PLACEHOLDER) | Confirmed | Medium | Master orchestrator should re-run V6 verdicts before commit; flagged in every per-pick file | All files |

---

## §9 — Action items for the human research lead

1. **Schedule cross-model verification**: provision the NVIDIA-gateway bearer in `CLAUDE.md` and re-run V6 verdicts on at least T1.1, T1.4, T1.6 (the gating picks) before Week-1 execution begins.
2. **Approve the Pearl-causal proposition draft on Day 1** (T1.4) — this is the single most time-sensitive theorem, since T1.4 empirical sweep on Day 3 needs the prediction (cascade-depth monotonicity) pre-registered.
3. **Run T1.1 GATE 1 on Day 4** before committing to Theorem 4 closure on Day 6–7 — if e-process growth is too slow on PRM_min, demote T1.1 to Tier-2 immediately.
4. **Confirm the 3-pick survival list framing** (§7 above) with the writing lead before Day 16; this determines the §1 Contributions ordering and the headline-vs-section split.
5. **Drop T1.9 entirely** if AIME-only Day-13 pilot shows Δ < 1.5pp — do not even include in the score-family ladder. EVoSS + FermiEval citations cover the framing.
6. **Schedule the human-audit κ pass** (T1.10, 50 traces, ~3–4 hours) for Week 2 Day 10.
7. **Pre-register all stratifications** (T1.4 cascade-depth, T1.6 K-sweep, T1.10 mode-prevalence) before respective sweeps to avoid post-hoc subset selection.

---

## §10 — Cross-Model Verification Results (consolidated)

*Per CLAUDE.md `cross_model_verification: mode: all`, all eight per-pick verdicts and this consolidated summary are in scope for verifier re-check. The fallback verifier (`openai/openai/gpt-5.5` or `gcp/google/gemini-3.1-pro-preview`) was not invoked because the inference token in the active config is `sk-PLACEHOLDER`. Logging single-model fallback per the protocol on every per-pick file plus this summary.*

**Specific items the verifier model should pressure-test on re-run:**

1. *T1.1 vs Sequential-EDFL*: is the "PRM channel + composition" differentiation sufficient, or does the verifier predict reviewer rejection regardless?
2. *T1.4 Pearl-causal proposition*: is the faithfulness assumption defensible on real CoT traces, or does the verifier predict empirical failure?
3. *T1.6 Frame B composition theorem*: is the conjectured combined-slack bound provably correct, or does the verifier identify a counterexample?
4. *T1.10 N=300 vs N=500*: does the verifier's bootstrap-CI prior require N=500 to support sharp-differential claims?
5. *T1.9 fold decision*: is Δ > 1.5pp the right cutoff, or does the verifier suggest Δ > 1.0pp / 2.0pp?
6. *3-pick survival list (§7)*: would the verifier substitute T1.1 (Theorem 4) for T1.6 if the paper targets NeurIPS rather than TMLR?

**Disagreements, when surfaced, append here without silent override per the CLAUDE.md "no silent overrides" rule. As of 2026-05-08, no verifier re-check has been performed.**

---

## §11 — Pointers to per-pick files

| Pick | Verification file | Lines |
|---|---|---|
| T1.1 | `/home/nvidia/future/literature/verification/T1_1_verification.md` | 447 |
| T1.2 | `/home/nvidia/future/literature/verification/T1_2_verification.md` | 196 |
| T1.3 | `/home/nvidia/future/literature/verification/T1_3_verification.md` | 203 |
| T1.4 | `/home/nvidia/future/literature/verification/T1_4_verification.md` | 537 |
| T1.5 | `/home/nvidia/future/literature/verification/T1_5_verification.md` | 274 |
| T1.6 | `/home/nvidia/future/literature/verification/T1_6_verification.md` | 313 |
| T1.9 | `/home/nvidia/future/literature/verification/T1_9_verification.md` | 244 |
| T1.10 | `/home/nvidia/future/literature/verification/T1_10_verification.md` | (this round) |

---

## §12 — One-line bottom line

> **CoT-CP is a credible TMLR submission today (with Theorems 1–3 + Theorem 4-bis BH/BY + T1.6 §6 Robustness + T1.10 §6 sub-section). It becomes a credible NeurIPS/ICLR submission if and only if Theorem 4 (T1.1 shift-aware composition) closes by Week-2 Gate 2 and the Pearl-causal proposition (T1.4) lands. T1.9 should be cut. T1.3 rides along as defensive ablation. T1.5 is gated. The single biggest 4-week risk is Sequential-EDFL preempting T1.1, mitigated only by delivering the composition theorem and the PRM-channel differentiator.**

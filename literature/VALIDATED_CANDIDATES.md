# Validated Cross-Disciplinary Candidates for CoT-CP

> Compiled 2026-05-08. Consolidated, ranked, and validated synthesis of **109 cross-disciplinary candidates** drawn from
> `CROSS_DISCIPLINARY_IDEAS.md` (32 entries — stat / info-theory / numerical analysis / cognitive science),
> `NON_AI_DOMAIN_PRACTICES.md` (43 entries — medicine / aviation / law / quality engineering / military / surgery / journalism / other operational disciplines), and
> `MATH_SCI_ENG_PRACTICES.md` (33 entries — pure math / physics / chemistry / astronomy / civil-mech-elec-chem engineering).
>
> All 109 entries were V1–V4 validated; 30 distinct clusters were identified, and each candidate was placed in one of four tiers.

---

## §1 Executive Summary

After a cluster-by-cluster validation pass over 109 cross-disciplinary candidates, the field collapses to **30 substantive clusters and roughly 12 atomic methodological imports** that survive all four validation tests (V1 source validity, V2 mapping soundness, V3 feasibility on the existing 11×7 infrastructure, V4 novelty over the 30-paper within-AI synthesis). The clusters map cleanly onto our existing pipeline: "forced verification" (Pólya / WHO Time-out / Pre-decode checklist / Look Back) becomes a binary signal stackable against `prm_min`; "recursive belief update" (Kalman / e-process / POMDP / BOCPD) is a *single* mathematical object that subsumes per-step CP Approach A; "multi-source independence" (two-source / multi-spectroscopy / multi-wavelength / blinded concurrence) is the same idea as `tiebreak_lex` viewed across model families; "earliest divergence" (Five Whys / CPM / HGJ Idea 1.1) gives the theoretical justification for our already-planned earliest-bad-step re-roll; and "adversarial probing" (HAZOP / FMEA / Lakatos / Red Team / cross-examination) is the cross-model verification protocol generalized to a structured probe-set. The single highest-value imports are **(i) e-processes / anytime-valid inference (CROSS-DISC §A1)** as the formal frequentist replacement for fixed-N self-consistency, **(ii) Benjamini-Hochberg FDR (CROSS-DISC §I1)** as a strict tightening of the multi-step Bonferroni in our per-step CP Approach B, and **(iii) Diverse Beam Search (CROSS-DISC §F3)** as the cheapest empirical attack on the K=4 alternative-diversity bottleneck that nullified Pilots C/K/L. Three honorable mentions — Kalman filter (MATH_SCI §F3) for principled per-step CP, mesh-refinement adaptive-N (MATH_SCI §E2) for compute-Pareto, and PMI as a unified score (CROSS-DISC §E3) — are also Tier 1. The remaining ~80 candidates are either redundant within their cluster, restricted to v2/future work for cost reasons, useful for §1/§6 framing only, or rejected as cosmetic borrowings.

---

## §2 Methodology

### Validation tests (applied per cluster, not per entry)

For each cluster of sister candidates, the following 4-letter judgment was assigned:

- **V1 — Source validity**: does the cited practice/book/paper exist as described, with a plausible canonical citation? `V` = real and well-cited; `P` = real but loose citation; `F` = invented or seriously misdescribed.
- **V2 — Mapping soundness**: is the proposed CoT-CP analogy more than cosmetic? Specifically: does it identify a *concrete* mechanism (a score function, an algorithmic step, an experimental protocol) rather than a slogan? `V` = concrete + non-trivial; `P` = concrete but redundant with existing pipeline; `F` = slogan only.
- **V3 — Feasibility on current infra**: given 11 models, 2× H100 NVL, PRM800K calibration, and the existing pipeline, can a v1 or v2 experiment be done? `V` = ≤2 weeks v1; `P` = v2 (1-3 months) feasible; `F` = requires new infrastructure or out-of-scope.
- **V4 — Novelty over within-AI literature**: having read the 30-paper synthesis (DeepConf, DEER, CoVeR, DCF, Adaptive-Consistency, ESC, Snell+, ProcessBench, EDU-PRM, ConfSpec, etc.), does this candidate contribute something they don't already do? `V` = contributes; `P` = partially overlaps; `F` = already done by ≥1 cited paper.

Example: `VVPV` = source valid, mapping valid, partial feasibility (v2 only), novelty valid.

### Cluster-level, not entry-level

Many of the 109 entries are alias-of-each-other (e.g., Pólya Look Back ≈ WHO Time-out ≈ Pre-decode checklist all collapse to a single "forced terminal verification" cluster). Validating cluster-by-cluster cuts the work by ~3-4× and forces honesty: we cannot count the same idea three times.

### Cross-cutting filters applied

- A candidate failing **V1** (invented source) is auto-rejected — no further analysis.
- A candidate failing **V2** with a `F` is rejected as cosmetic.
- A candidate with `VVFV` (real + sound + infeasible + novel) is permitted Tier 2/3 but never Tier 1.
- A candidate with `VVVF` (real + sound + feasible + done already) is downgraded to Tier 3 (framing only) unless our experimental angle is materially different.

---

## §3 Cluster Map (30 clusters)

Each cluster lists constituent candidates, the unifying principle, the cluster verdict (V1V2V3V4), and a one-sentence judgment of cluster value.

### Cluster K1 — Forced terminal verification ("Look Back")

- **Constituents**: `MATH_SCI §A1 Pólya Look Back`, `NON_AI §C1 WHO Surgical Time-Out`, `NON_AI §B1 Cockpit checklist (Gawande)`, `MATH_SCI §B4 limiting/boundary case verification`, `NON_AI §C2 sponge/instrument count`, `NON_AI §M1 mise en place` (post-prep verification half), `NON_AI §M2 tasting at intervals` (terminal taste only).
- **Unifying principle**: A *mandatory, non-optional, structured* verification block applied to the produced answer (or near-end of the trace) that probes: (i) does substituting back satisfy the original problem, (ii) do special / boundary cases behave as expected, (iii) are units / dimensions / counts conserved.
- **Verdict**: `VVVV` (cluster strong on all four).
- **Cluster value**: **HIGH for v1.** Binary signal `s = LookBackPassed ∈ {0, 1}` plugs straight into trajectory aggregator φ. Distinct from CoVe (which generates verification questions ad hoc); structured Pólya questions are *prescribed*. See Tier 1.

### Cluster K2 — Recursive Bayesian belief update on running correctness

- **Constituents**: `MATH_SCI §F3 Kalman filter`, `CROSS-DISC §A1 e-process / test-by-betting`, `CROSS-DISC §A3 CUSUM / BOCPD`, `CROSS-DISC §D1 POMDP belief states`, `CROSS-DISC §C2 Bayesian brain`, `CROSS-DISC §C1 predictive coding`.
- **Unifying principle**: Maintain a posterior over the latent "trace will be correct" given step-level observations; predict-update at each step. The *running posterior probability of correctness* is the right per-step score. Heuristic running-min / running-mean is a degenerate filter.
- **Verdict**: `VVVV` for the e-process / Kalman variants; `VVPV` for predictive coding (theory-only) and POMDP (formalization, expensive).
- **Cluster value**: **HIGH for v1 (e-process, Kalman); v2 for the rest.** This is the most *mathematically* productive cluster — the e-process / anytime-valid line gives the strongest single upgrade to fixed-N SC. See Tier 1.

### Cluster K3 — Multi-source independent verification

- **Constituents**: `NON_AI §I1 two-source rule (journalism)`, `MATH_SCI §C2 multi-spectroscopy (NMR + IR + MS)`, `MATH_SCI §D2 multi-wavelength survey`, `NON_AI §A2 clinical concurrence (blinded)`, `NON_AI §C3 sterile field (layered redundancy)`, `MATH_SCI §G2 Layer of Protection Analysis (LOPA)`, `MATH_SCI §F1 Hamming codes / numeric+fractional redundancy`.
- **Unifying principle**: A claim is accepted only when ≥2 *independent* probes (different model, different prompt, different decoding, different score family) agree. Independence is the criterion, not count.
- **Verdict**: `VVVV` for the LOPA-style §5 ablation (cheap); `VVVP` for two-source / multi-spectroscopy as a new method (already mostly subsumed by `tiebreak_lex` and Pilot M paraphrase consensus).
- **Cluster value**: **MED for v1 §5 ablation; otherwise framing.** LOPA layered-protection plot is the cleanest figure-able artifact. Multi-source via cross-model verification overlaps the existing CMV protocol. See Tier 2.

### Cluster K4 — Earliest-divergence is the cascade source

- **Constituents**: `NON_AI §E2 Five Whys`, `MATH_SCI §E4 Critical Path Method (CPM)`, plus internal HGJ Idea 1.1 (earliest-bad-step re-roll).
- **Unifying principle**: The proximate (worst-step) error is rarely the cascade source; the *earliest* threshold-violating step is. Re-roll there, not at the symptom.
- **Verdict**: `VVVV`.
- **Cluster value**: **HIGH.** The cluster is the *theoretical justification* for an experiment we already planned (1-2 GPU hours). Both Five Whys (Toyota Production System, codified) and CPM (Kelley-Walker 1959, defense / project management) provide the heritage. See Tier 1.

### Cluster K5 — Alternative-hypothesis enumeration

- **Constituents**: `NON_AI §A1 DDx with VINDICATE mnemonic`, `CROSS-DISC §F3 Diverse Beam Search`, `NON_AI §H1 chess multipv annotation`, plus internal Tree-of-Thoughts / Pilot C-K-L (which all failed for diversity reasons).
- **Unifying principle**: Generate K candidate solution paths *with knowledge of each other* and explicit pertinent-evidence-for / against, rather than K independent samples. Forces structural diversity, not just temperature diversity.
- **Verdict**: `VVVV` for DBS (already proven in NLP literature); `VVVV` for DDx prompting; `VVPP` for chess-style annotations (cosmetic).
- **Cluster value**: **HIGH for v1 (DBS, DDx).** Directly attacks the documented Pilots C/K/L bottleneck. DBS is 2-3 days experimental. See Tier 1.

### Cluster K6 — Adversarial probing / falsification

- **Constituents**: `NON_AI §F3 Red Team`, `NON_AI §D4 cross-examination (Wigmore)`, `MATH_SCI §A2 Lakatos refutations`, `MATH_SCI §G1 HAZOP guidewords`, `NON_AI §E1 FMEA`, `NON_AI §B2 CRM 2-pilot redundancy (mandatory objection)`.
- **Unifying principle**: An accepted answer must survive *structured* attempts to find counter-examples. Self-critique by the same model is weak (shared blind spots); a *different* model or a *systematic* probe-set (HAZOP guidewords, VINDICATE) is structurally stronger.
- **Verdict**: `VVVV` for Red Team / cross-model verification (we already have CMV infrastructure); `VVPV` for HAZOP / Lakatos as new method (expensive); `VVPP` for FMEA workshop (one-time exercise, low novelty).
- **Cluster value**: **MED-HIGH.** Red Team via existing CMV is the cheap win; HAZOP / Lakatos are v2. See Tier 1 (Red Team) and Tier 2 (HAZOP).

### Cluster K7 — Sanity-check primitives (free dimensional / conservation / boundary)

- **Constituents**: `MATH_SCI §B1 Buckingham π / dimensional analysis`, `MATH_SCI §B3 conservation laws`, `MATH_SCI §B4 limiting case verification`, `MATH_SCI §C3 stoichiometry`, `MATH_SCI §G3 mass/energy balance`, `MATH_SCI §B2 Fermi estimation`.
- **Unifying principle**: Cheap, mechanical, named checks that exploit symmetries / conserved quantities to flag arithmetic-level errors. Not domain-specific to LLMs — every mature physical science enforces these reflexively.
- **Verdict**: `VVVV`.
- **Cluster value**: **MED for v1 if bundled.** Build a 1-week sanity-check library; expose results as a binary score per check. Free, mechanical, narrow each but combined gives several free signals per problem. See Tier 1 (bundled).

### Cluster K8 — Heuristic-rigorous duality / cross-mode consistency

- **Constituents**: `MATH_SCI §A4 Tao heuristic-rigorous-postrigorous`, `MATH_SCI §B2 Fermi estimation` (the *cross-check* role), `CROSS-DISC §B2 Kahneman dual-process`, `MATH_SCI §A3 Erdős book proofs` (multiple proofs).
- **Unifying principle**: Generate two answers in different reasoning modes (heuristic vs rigorous, fast vs slow, algebraic vs combinatorial); accept iff they agree within stated tolerance.
- **Verdict**: `VVVV`.
- **Cluster value**: **MED-HIGH for v1.** Two-pass Fermi+rigorous is 3-5 days. See Tier 1.

### Cluster K9 — Forced structural / output schema

- **Constituents**: `NON_AI §D1 IRAC (Issue-Rule-Application-Conclusion)`, `NON_AI §A3 SBAR communication`, `NON_AI §L1 chain of custody`, `MATH_SCI §G4 P&ID structured graph`, `NON_AI §F1 OODA loop` (forced Observe→Orient before Decide-Act), `NON_AI §I2 inverted pyramid` (answer-first), `NON_AI §M1 mise en place` (prep-then-cook).
- **Unifying principle**: Force the trace into a *named* component structure with per-component scoring. Theorem applications get a Rule-validity score; calculations get a PRM score; provenance gets a chain-of-custody integrity score.
- **Verdict**: `VVVV` for IRAC and chain-of-custody as scoring primitives; `VVVV` for inverted pyramid (truncation-robustness on long-CoT models, addresses a known issue).
- **Cluster value**: **MED-HIGH for v1 if combined with one structured-scoring experiment.** IRAC is 1 week, chain-of-custody is 2 weeks. See Tier 2 (none of these is a *headline* contribution; they are paper §6 future-work or cheap §5 ablations).

### Cluster K10 — Adaptive computation / convergence test

- **Constituents**: `MATH_SCI §E2 FEA mesh refinement`, `CROSS-DISC §H1 RKF45 adaptive step-size`, `CROSS-DISC §H2 Richardson extrapolation`, `CROSS-DISC §A4 optimal stopping / Snell envelope`, `CROSS-DISC §A2 Wald SPRT`, `MATH_SCI §E5 Weibull / hazard schedule`.
- **Unifying principle**: Adapt sample count / segmentation step size to local information content; stop or refine based on a stability / local-error criterion. Subsumes adaptive-N SC, adaptive segmentation, and SPRT-style early stopping.
- **Verdict**: `VVVV` for mesh-refinement adaptive-N SC; `VVVV` for RKF45-style adaptive segmentation (but expensive); `VVPV` for SPRT (likely empirically tied with e-process); `VVPP` for Richardson extrapolation (clever but speculative).
- **Cluster value**: **HIGH (mesh-refinement adaptive-N is a clean v1 result); MED for the rest.** See Tier 1 (mesh-refinement) and Tier 2 (RKF45, SPRT).

### Cluster K11 — Information-theoretic score evaluation (PMI / mutual information)

- **Constituents**: `CROSS-DISC §E3 PMI / contrastive PMI`, `CROSS-DISC §E2 Information bottleneck`, `CROSS-DISC §E1 rate-distortion theory`.
- **Unifying principle**: A score function's quality = its mutual information with correctness. PMI gives the per-step evaluation; IB gives the optimal compression principle; rate-distortion gives the achievable lower bound on the (compute, accuracy) frontier.
- **Verdict**: `VVVV` for PMI as both a new score and a unified evaluation metric; `VVPV` for IB and R-D (theory-only for v1, paper-strengthening for §3).
- **Cluster value**: **HIGH for v1 (PMI); MED for the rest.** PMI is 1 week experimental and unifies the heterogeneous score zoo under one optimality measure. See Tier 1.

### Cluster K12 — Multi-step / FDR-style step-level CP

- **Constituents**: `CROSS-DISC §I1 Benjamini-Hochberg FDR`, plus internal per-step CP Approach B (currently Bonferroni).
- **Unifying principle**: BH-FDR controls the expected fraction of false rejections, which is tighter than Bonferroni when step decisions are exchangeable (which our setup gives us). Bates-Angelopoulos-Lei-Romano (JRSSB 2023) extend BH to selective FDR for prediction sets — directly applicable.
- **Verdict**: `VVVV`.
- **Cluster value**: **HIGH.** Strict improvement over our current per-step Bonferroni; 1-2 weeks theory + 1 week experiment. See Tier 1.

### Cluster K13 — Robust aggregation / outlier-resistant scoring

- **Constituents**: `CROSS-DISC §I2 Huber estimators`, `MATH_SCI §E3 tolerance stacking`.
- **Unifying principle**: Outlier-step contamination dominates min-aggregators (`lp_min`); Huber-trimmed-mean and worst-case-vs-RSS-stack analyses give alternatives.
- **Verdict**: `VVVV` for Huber-trimmed φ ablation; `VVVP` for tolerance-stacking framing (pure framing; useful for §3).
- **Cluster value**: **LOW-MED for v1; framing for §3.** See Tier 2 (Huber) and Tier 3 (tolerance stacking framing).

### Cluster K14 — Local / sub-population CP under shift

- **Constituents**: HGJ Idea 2.1 Local CP (SBERT + KNN), `MATH_SCI §D1 distance ladder calibration`, `CROSS-DISC §J1 active-learning calibration set construction`.
- **Unifying principle**: For OOD test points, calibrate against a *local* sub-population (KNN by embedding) rather than a global cal set; or use a multi-rung ladder where each rung partially recalibrates the previous. Distance-ladder thinking is the astronomical analogue.
- **Verdict**: `VVVV` for Local CP (already on the HGJ short list); `VVPV` for distance-ladder ladder calibration (natural extension of Theorem 3); `VVVV` for active-learning calibration sample selection.
- **Cluster value**: **VERY HIGH (Local CP is HGJ priority #1).** Distance-ladder framing strengthens Theorem 3. See Tier 1.

### Cluster K15 — Verbalized / discrete-bin confidence

- **Constituents**: `NON_AI §A3 SBAR verbalized confidence`, `NON_AI §H1 chess !,!!,?,?? bin`, `NON_AI §D2 burden of proof gradations`.
- **Unifying principle**: Replace continuous score with K-bin verbalized or discrete confidence; better for closed-source-API models (no logit access) and more interpretable for users.
- **Verdict**: `VVVV` for verbalized confidence (cites Tian-Mitchell EMNLP 2023); `VVVP` for chess-bin (presentation only).
- **Cluster value**: **MED for v1 (verbalized opens API-only models); LOW for chess-bin.** See Tier 2.

### Cluster K16 — Pre-commitment / pre-decode scaffolding

- **Constituents**: `NON_AI §B1 cockpit checklist`, `NON_AI §F1 OODA loop` (the Orient phase), `NON_AI §A1 DDx prompting`, `NON_AI §E5 Poka-yoke (mistake-proofing)`, `MATH_SCI §A1 Pólya step (1) Understand`.
- **Unifying principle**: Force *pre-trace* problem-class identification, given-quantity enumeration, and edge-case awareness before the model commits to a solution path.
- **Verdict**: `VVVV` for the experimental cluster (1-week prompt experiment); `VVVF` for Poka-yoke (mostly subsumed by existing answer-extraction infrastructure).
- **Cluster value**: **MED for v1 (one prompt-engineering ablation).** See Tier 2.

### Cluster K17 — After-action / failure-mode taxonomy

- **Constituents**: `NON_AI §A4 M&M conferences`, `NON_AI §B4 NTSB probable-cause`, `NON_AI §F2 After-Action Review (AAR)`, `MATH_SCI §A2 Lakatos` (post-hoc dialectic), `NON_AI §E1 FMEA workshop`.
- **Unifying principle**: Produce a *structured taxonomy* of failure modes from the trace corpus, then report per-failure-mode kept-accuracy. Turns aggregate accuracy into a structured diagnostic.
- **Verdict**: `VVVV` for an LLM-judge AAR pass on 50-100 failed AIME traces; `VVVV` for a §5 per-failure-mode breakdown.
- **Cluster value**: **HIGH** — gives §5 a structurally new figure. 1-week effort. See Tier 1.

### Cluster K18 — Bayesian / weighted aggregation alternatives

- **Constituents**: `CROSS-DISC §D4 Bayesian model averaging`, `CROSS-DISC §G2 boosting / AdaBoost`, `MATH_SCI §A3 Erdős book proofs (quality-weighted multi-trace)`.
- **Unifying principle**: Replace majority-vote in SC with likelihood-weighted average / boosted combination / quality-ranked selection. Soft-SC (Wang+ 2024 in our notes) does the first; we can revisit with calibrated weights.
- **Verdict**: `VVVP` (overlaps existing soft-SC literature; small lift expected).
- **Cluster value**: **LOW-MED.** Incremental ablation. See Tier 2.

### Cluster K19 — Speculative / horizon-aware decoding

- **Constituents**: `CROSS-DISC §D2 Model Predictive Control`, plus internal SCoT / DEER / ConfSpec.
- **Unifying principle**: At each step boundary, look ahead K steps cheaply; intervene now if the predicted horizon-end confidence is low.
- **Verdict**: `VVPP` (heavily contested terrain; DEER, SCoT, ConfSpec already occupy this space; CoT-CP's wedge here is calibration, not new mechanism).
- **Cluster value**: **MED.** Out-of-scope for v1. See Tier 2.

### Cluster K20 — Risk-sensitive training-time companion

- **Constituents**: `CROSS-DISC §D3 CVaR / risk-sensitive RL`.
- **Verdict**: `VVPV` (out of scope for inference-time v1 paper).
- **Cluster value**: **LOW for v1; v2 follow-up paper.** See Tier 2.

### Cluster K21 — Causal / counterfactual step intervention

- **Constituents**: `CROSS-DISC §K1 counterfactual / ITE estimation`.
- **Unifying principle**: Frame Layer B re-rolls as a counterfactual question; doubly-robust or front-door adjustment may explain the Pilots C/K/L null result.
- **Verdict**: `VVPV` (real, sound, but expensive — 2-3 weeks; novel but speculative).
- **Cluster value**: **MED.** v2 / future work. See Tier 2.

### Cluster K22 — Tree search / branch-and-bound / MCTS

- **Constituents**: `CROSS-DISC §F1 branch-and-bound`, `CROSS-DISC §F2 MCTS / AlphaZero`.
- **Verdict**: `VVPV` (expensive; existing AlphaMath / ReST-MCTS occupy MCTS territory; CoT-CP's wedge would be CP-calibrated UCB exploration).
- **Cluster value**: **MED** (4+ weeks). See Tier 2.

### Cluster K23 — Worked examples / curriculum / few-shot

- **Constituents**: `CROSS-DISC §B1 Cognitive Load Theory + worked examples`, `CROSS-DISC §J2 self-explanation prompting`, `NON_AI §G1 sectional rehearsal`, `NON_AI §G2 jazz fakebook of named transformations`.
- **Unifying principle**: Improves *base trace quality* before any CP filtering — orthogonal to CoT-CP's contribution but complementary.
- **Verdict**: `VVVF` for worked-example prompting (already extensively covered in CoT prompting literature); `VVPV` for jazz-fakebook of named transformations (speculative).
- **Cluster value**: **LOW for v1.** Already done; doesn't strengthen our paper. See Tier 3 (framing / motivation).

### Cluster K24 — Metacognition / verbalized confidence calibration

- **Constituents**: `CROSS-DISC §B3 metacognition m-ratio`, `NON_AI §A3 SBAR verbalized confidence` (already in K15).
- **Unifying principle**: Compute the standard psychophysics m-ratio = meta-d'/d' across our 11 models × 7 datasets to give a "reasoning-model metacognition benchmark" angle.
- **Verdict**: `VVVV`.
- **Cluster value**: **MED** — cheap §5 figure, supports broader pitch. 2-3 days. See Tier 2.

### Cluster K25 — Reproducibility & honest reporting

- **Constituents**: `MATH_SCI §B6 blind analysis (HEP/cosmology)`, `MATH_SCI §B5 5σ discovery threshold`, `NON_AI §D3 Daubert standard`, `NON_AI §J1 BATNA`.
- **Unifying principle**: Discipline of pre-registration, multiple-comparisons reservation, and falsifiability framing. All are *reporting* / *framing* moves with ~zero cost and meaningful reviewer credibility.
- **Verdict**: `VVVV` (zero-cost framing).
- **Cluster value**: **MED-HIGH** for paper-reviewer credibility. See Tier 3.

### Cluster K26 — Burden-of-proof / Factor-of-Safety user-facing α

- **Constituents**: `NON_AI §D2 burden of proof gradations`, `MATH_SCI §E1 Factor of Safety (ASCE)`, `NON_AI §J1 BATNA`.
- **Unifying principle**: Reframe α as a stake-dependent design choice (legal / structural / negotiation analogues); ship a `(use case, stake, recommended α, achieved kept-accuracy)` table in §6.
- **Verdict**: `VVVV`.
- **Cluster value**: **MED.** Pure framing, free, sharp deployment narrative. See Tier 3.

### Cluster K27 — Cadence / step-segmentation control

- **Constituents**: `MATH_SCI §F5 phase-locked loop (PLL) for step granularity`, `CROSS-DISC §H1 RKF45 adaptive step-size` (already in K10), `NON_AI §B3 sterile cockpit (no filler)`.
- **Unifying principle**: Control the *granularity* of step segmentation as a tracked / regulated quantity; sterile-cockpit forbids narrative filler that crowds out reasoning tokens.
- **Verdict**: `VVPV` for PLL (narrow / speculative); `VVVV` for sterile-cockpit prompt (3-5 days).
- **Cluster value**: **LOW-MED.** See Tier 2 (sterile cockpit) and Tier 3 (PLL).

### Cluster K28 — Lock-in / paraphrase consensus

- **Constituents**: `MATH_SCI §F2 lock-in amplifier / synchronous detection`, `NON_AI §A2 blinded clinical concurrence`, plus internal Pilot M paraphrase consensus.
- **Unifying principle**: Modulate at known frequency (paraphrase the question), demodulate by matching answers — gives independent paraphrase consensus distinct from temperature-only SC.
- **Verdict**: `VVVP` (mostly subsumed by existing Pilot M; cluster refines but does not displace).
- **Cluster value**: **LOW-MED.** See Tier 2.

### Cluster K29 — Reference-witness / calibration card in prompt

- **Constituents**: `NON_AI §K2 photographic calibration cards`, `NON_AI §K1 exposure bracketing` (T-bracket SC).
- **Unifying principle**: Embed a known-easy reference problem in the prompt; if the model can't reproduce the reference, abstain on the target. Provides a free per-prompt validity check.
- **Verdict**: `VVVV` for the calibration-card prompt (1 week); `VVVP` for T-bracket SC (3 days, small lift).
- **Cluster value**: **LOW-MED.** See Tier 2.

### Cluster K30 — Auftragstaktik / intent restatement / re-orient

- **Constituents**: `NON_AI §F4 commander's intent (Auftragstaktik)`, `NON_AI §L2 cognitive interview (PEACE)`, `NON_AI §M2 tasting at intervals` (mid-trace probe).
- **Unifying principle**: When the trace stalls or confidence drops, inject a *structured* re-orientation prompt rather than restarting; cognitive-interview techniques (reverse-order, change-perspective, context-reinstate) outperform a generic "let me reconsider".
- **Verdict**: `VVVV` (1 week experimental).
- **Cluster value**: **MED** — directly relevant to long-CoT models (R1-Distill, QwQ) where stalls are common. See Tier 2.

---

### Cluster summary table

| ID | Cluster name | Verdict | Tier |
|---|---|---|---|
| K1 | Forced terminal verification (Look Back) | VVVV | 1 |
| K2 | Recursive Bayesian belief update | VVVV | 1 |
| K3 | Multi-source independent verification | VVVP | 2 |
| K4 | Earliest-divergence cascade source | VVVV | 1 (planned) |
| K5 | Alternative-hypothesis enumeration | VVVV | 1 |
| K6 | Adversarial probing / falsification | VVVV / VVPV | 1 (Red Team) / 2 (HAZOP) |
| K7 | Sanity-check primitives | VVVV | 1 (bundled) |
| K8 | Heuristic-rigorous duality | VVVV | 1 |
| K9 | Forced structural / output schema | VVVV | 2 |
| K10 | Adaptive computation / convergence | VVVV | 1 (mesh-refinement) / 2 (rest) |
| K11 | Information-theoretic score eval (PMI) | VVVV | 1 |
| K12 | BH-FDR multi-step CP | VVVV | 1 |
| K13 | Robust aggregation (Huber) | VVVV / VVVP | 2 / 3 |
| K14 | Local CP / distance-ladder | VVVV | 1 (already HGJ #1) |
| K15 | Verbalized / discrete-bin confidence | VVVV | 2 |
| K16 | Pre-commitment scaffolding | VVVV | 2 |
| K17 | After-action / failure-mode taxonomy | VVVV | 1 |
| K18 | Bayesian / weighted SC | VVVP | 2 |
| K19 | Speculative / horizon decoding | VVPP | 2 |
| K20 | Risk-sensitive training | VVPV | 2 (separate paper) |
| K21 | Causal / counterfactual | VVPV | 2 |
| K22 | Tree search / MCTS | VVPV | 2 |
| K23 | Worked examples / curriculum | VVVF | 3 |
| K24 | Metacognition m-ratio | VVVV | 2 |
| K25 | Blind analysis / reproducibility | VVVV | 3 |
| K26 | Burden of proof / FoS framing | VVVV | 3 |
| K27 | Cadence / segmentation control | VVPV / VVVV | 2 / 3 |
| K28 | Lock-in / paraphrase consensus | VVVP | 2 |
| K29 | Reference-witness calibration card | VVVV | 2 |
| K30 | Auftragstaktik / intent restate | VVVV | 2 |

---

## §4 Tier 1 — Validated, high-value, implementable in v1 paper

Ten Tier 1 picks. Each has been validated `VVVV`, is tractable in 1-3 weeks, and would meaningfully strengthen the v1 paper.

### T1.1 — E-processes / anytime-valid SC stopping (Cluster K2)

- **Source**: Ramdas, Grünwald, Vovk, Shafer (2023), *Game-Theoretic Statistics and Safe Anytime-Valid Inference*, Statistical Science. CROSS-DISC §A1.
- **Cluster**: K2 (recursive Bayesian belief update).
- **Concrete experiment**: Replace fixed-N=8 SC with an e-process on the running vote-share. Stop sampling when the e-value exceeds 1/α. Compare to (a) Adaptive-Consistency (Aggarwal+ 2023, Bayesian credibility), (b) ESC (Li+ 2024, w-window), and (c) fixed N=8 / 16. Report mean N and kept-accuracy on MATH-500 + Qwen2.5-7B / Qwen2.5-Math-7B / Qwen3-8B.
- **Predicted lift**: Match SC@8 kept-accuracy with mean N ≈ 4-5 across easy problems (GSM8K, MATH-500) and N ≈ 7-9 on hard problems (AIME, OlympiadBench). 2-3× compute reduction at constant kept-accuracy. Most importantly, this gives a *frequentist coverage guarantee* on the early-stop, which Adaptive-Consistency / ESC do not.
- **Decision**: **GO.** Tightest mathematical upgrade in the entire 109-entry catalog. 1-2 weeks theory + 2-3 days experimental. Theorem 1 extends from split-CP to anytime-CP; this is a paper-strength contribution.

### T1.2 — Benjamini-Hochberg FDR for per-step CP Approach B (Cluster K12)

- **Source**: Benjamini & Hochberg (1995), *Controlling the False Discovery Rate*, JRSSB. Bates-Angelopoulos-Lei-Romano (JRSSB 2023). CROSS-DISC §I1.
- **Cluster**: K12.
- **Concrete experiment**: Replace the Bonferroni α/T_max thresholds in our per-step CP Approach B with the BH procedure. Sort step-level p-values; reject the top-k with p_(i) ≤ k·α/m. Re-run the per-step coverage validation (Pilots 8/J/F).
- **Predicted lift**: Strict tightening — same coverage guarantee at lower abstention rate. On long-CoT models (R1-Distill 60+ steps, QwQ), Bonferroni gives α/60 ≈ 0.0017 per step at α=0.1, which is over-conservative. BH should recover 2-5pp kept-fraction at the same coverage.
- **Decision**: **GO.** 1 week theory + 1 week experimental. Strict improvement, no downside. Add as a Theorem-revision item alongside the T2 → LR+ rename and T3 → A1-weakening fixes already on the THEOREM_REVIEW.md action list.

### T1.3 — Diverse Beam Search for K=4 alternatives (Cluster K5)

- **Source**: Vijayakumar et al. (2018), *Diverse Beam Search*. CROSS-DISC §F3.
- **Cluster**: K5.
- **Concrete experiment**: Replace the temperature-0.7 sampling in our K=4 step-rejection (Pilots C/K/L) with DBS using a between-group dissimilarity penalty. Re-run on MATH-500 + Qwen2.5-7B and AIME + Qwen3-8B. Specifically test: does DBS unblock the Pilots C/K/L null result?
- **Predicted lift**: +2-5pp on AIME / Olympiad where alternative diversity is the documented bottleneck. The Pilots C/K/L null result was attributed in `METHOD_AND_RESULTS.md` §2.8 to "K-resample from a fixed prefix doesn't generate diverse enough alternatives" — DBS forces semantic diversity, which empirically beats temperature-only diversity (well-known result in NMT / dialogue).
- **Decision**: **GO.** 2-3 days experimental. If positive, becomes a Layer B variant worth reporting; if null, becomes a strengthened "step-level branching ceiling" claim.

### T1.4 — Earliest-bad-step re-roll with Five-Whys / CPM justification (Cluster K4)

- **Source**: Ohno (1988), *Toyota Production System*. Kelley-Walker (1959), CPM. NON_AI §E2 + MATH_SCI §E4. (Plus internal HGJ Idea 1.1.)
- **Cluster**: K4.
- **Concrete experiment**: Already on the HGJ short list. Re-roll at the earliest threshold-violating step (per-step CP threshold), not at the worst step. Use a *separate* trigger calibration as the HGJ reviewer feedback notes. K=4 alternatives at the earliest bad step. Compare to worst-step (Pilots C/K/L baseline).
- **Predicted lift**: +1-3pp absolute on AIME / Olympiad if the cascade-source hypothesis holds; null result still has paper value (strengthens step-branching ceiling claim).
- **Decision**: **GO.** 4-6 GPU hours per HGJ reviewer's revised estimate. The cluster gives the cross-domain justification: Toyota Five Whys (codified 1930s, validated industrially) and CPM (DuPont/Remington Rand 1959) both teach that the proximate failure is rarely the cascade source. Adds a citable theoretical anchor to a previously-underjustified design choice.

### T1.5 — Pólya Look Back primitive (Cluster K1)

- **Source**: Pólya (1945/2014), *How to Solve It*. Princeton. MATH_SCI §A1. WHO Surgical Safety Checklist (Haynes+ NEJM 2009) is the operational sister.
- **Cluster**: K1.
- **Concrete experiment**: Implement a deterministic Look-Back post-pass: substitute answer back into problem, check special / boundary cases, confirm units. Each check returns binary `{0, 1}`. Final score is `s = LookBackPassed ∈ {0, 1, 2, 3}` (count of passes). Plug into trajectory aggregator φ as an additional component of `tiebreak_lex`. Test on MATH-500, AIME, OlympiadBench.
- **Predicted lift**: +2-5pp on `tiebreak_lex` kept-accuracy at fixed answer rate, primarily on AIME / Olympiad where boundary-case errors dominate. Free downstream — the binary signal plugs into CRC.
- **Decision**: **GO.** 2-3 days. Distinct from CoVe (Dhuliawala+ 2024) which generates verification questions ad hoc — Pólya questions are *prescribed* and structured.

### T1.6 — Local CP via SBERT + KNN (Cluster K14)

- **Source**: Foygel-Barber et al. (Annals 2023, jackknife+) + Guan (2023, Localized CP). HGJ Idea 2.1, plus distance-ladder framing from MATH_SCI §D1.
- **Cluster**: K14.
- **Concrete experiment**: For each test point, retrieve top-K (K=30-60 per HGJ reviewer correction, *not* K=100) similar calibration prompts via SBERT embedding + FAISS. Compute the local quantile from the K nearest cal points. Compare to vanilla CP and Theorem 3 (PMF-weighted) on MATH→AIME and full cross-dataset transfer matrix. Report citation pairs Tibshirani-2019 → Foygel-Barber-2023 + Guan-2023 + Lei-Wasserman-2014.
- **Predicted lift**: Better OOD coverage than Theorem 3's overshoot (e.g., target 0.50 → achieved 0.45-0.55 instead of 0.633). Conditional coverage on sub-population.
- **Decision**: **GO.** Already HGJ priority #1. 6-10 GPU hours (HGJ reviewer corrected estimate). Distance-ladder framing strengthens Theorem 3 narrative.

### ~~T1.7 — Mesh-refinement adaptive-N self-consistency (Cluster K10)~~ → **DEMOTED to Tier 2** (2026-05-08)

> **Demotion reason** (post-search verification): too close to Adaptive-Consistency (Aggarwal 2023), ESC (Li 2024), ReASC ([2601.02970](https://arxiv.org/html/2601.02970)), and Dynamic Self-Consistency ([2408.17017](https://arxiv.org/html/2408.17017v1)). Mesh-refinement framing is a different *criterion* but likely matches them empirically. Keep as §6 prose remark, not as headline experiment.

#### Original T1.7 description (kept for reference)

- **Source**: Cook, Malkus, Plesha, Witt (2002), *Concepts and Applications of Finite Element Analysis*, 4th ed., Wiley. MATH_SCI §E2.
- **Cluster**: K10.
- **Concrete experiment**: Adaptive-N SC: run N=2 → check majority stability; if stable, accept; else N=4, re-check; up to N=32. Stability criterion = same majority answer at consecutive doublings. Compare to fixed N=16 and to T1.1 e-process. Run on MATH-500 + Qwen3-8B and AIME.
- **Predicted lift**: Match N=16 kept-accuracy at mean N ≈ 5-7. Less mathematically clean than e-process (T1.1) but easier to implement and gives a different convergence-criterion intuition.
- **Decision**: **GO if T1.1 is delayed; otherwise CONDITIONAL.** 1 week. The two ideas are partial substitutes; we should report both as cluster K10 alternatives in §6 Discussion.

### ~~T1.8 — PMI as score (Cluster K11)~~ → **DEMOTED to Tier 2** (2026-05-08)

> **Demotion reason** (post-search verification): direct overlap with **MITS** ([2510.03632](https://arxiv.org/html/2510.03632), 2025) — step-wise PMI scoring + entropy-based dynamic sampling for tree-search reasoning. Same empirical proposal at a different decision-rule (search-expansion vs CRC-coverage). Reposition as future work: *"CP-calibrated MITS-style PMI score"* — that wedge is real but requires MITS reproduction first.

#### Original T1.8 description (kept for reference)

- **Source**: Cover & Thomas, *Elements of Information Theory*, ch. 8. Modern: van den Oord et al. (2018), *Representation Learning with Contrastive Predictive Coding*. CROSS-DISC §E3.
- **Cluster**: K11.
- **Concrete experiment**: For each step t, compute s_t = log P(answer | step_t, prefix) − log P(answer | prefix). This is the informational contribution of step t. Add as a new score family between `lp_min` (1×) and `prm_min` (2×). Compute PMI(score_family, correctness) for each existing score; report as a unified evaluation metric.
- **Predicted lift**: New mid-cost score; PMI evaluation gives a single-number summary of score-family quality complementary to the SNR/LR+ ranking in Theorem 2.
- **Decision**: **GO.** 1 week. Both as new score and as evaluation metric — two contributions for one experiment.

### T1.9 — Two-pass Fermi / heuristic-rigorous cross-check (Cluster K8)

- **Source**: Weinstein & Adam (2008), *Guesstimation*, Princeton. Tao "What's New" career advice. MATH_SCI §B2 + §A4.
- **Cluster**: K8.
- **Concrete experiment**: Two-pass CoT: pass 1 generates a Fermi estimate ("approximately 5000, give or take 10×"); pass 2 generates the rigorous answer. Reject iff the rigorous answer falls outside the Fermi pass's stated bounds. Test on AIME (where answers are 0-999 integers) and OlympiadBench.
- **Predicted lift**: +1-3pp on AIME by catching gross arithmetic errors (off-by-10× from a misplaced decimal). Cheap.
- **Decision**: **GO.** 3-5 days. Distinct from CoVe (post-hoc verification) — Fermi pass is a *parallel* heuristic estimate, not a self-critique.

### T1.10 — Per-failure-mode kept-accuracy via M&M / NTSB / AAR taxonomy (Cluster K17)

- **Source**: Pierluissi+ (2003) JAMA, M&M conferences. NTSB Aviation Accident Reports. US Army FM 7-0 AAR doctrine. NON_AI §A4 + §B4 + §F2.
- **Cluster**: K17.
- **Concrete experiment**: Run an LLM-judge AAR pass on 50-100 failed AIME / Olympiad traces. Build a taxonomy of dominant failure modes (arithmetic, retrieval, planning, premature commitment, garden-path, etc.). Report per-failure-mode kept-accuracy in §5 (e.g., "CoT-CP at α=0.5 catches 73% of arithmetic-error traces and 41% of premature-commitment traces"). This *changes which experiments to run* by surfacing where score-design effort should go.
- **Predicted lift**: No direct accuracy gain, but a structurally new §5 figure that radiology-AI papers ship and CoT papers don't. Strong reviewer signal.
- **Decision**: **GO.** 1 week initial; ongoing low cost.

### Tier 1 summary table

| # | Cluster | Source | Effort | Predicted lift | Decision |
|---|---|---|---|---|---|
| T1.1 | K2 | Ramdas+ 2023 (e-process) | 2-3 weeks | 2-3× SC compute reduction at constant cov | GO (highest priority) |
| T1.2 | K12 | BH 1995 + Bates+ 2023 | 1-2 weeks | +2-5pp kept-frac on long-CoT | GO |
| T1.3 | K5 | DBS Vijayakumar+ 2018 | 2-3 days | +2-5pp on AIME if Pilots C/K/L unblock | GO (cheapest) |
| T1.4 | K4 | Toyota 5-Whys + CPM | 4-6 GPU hr | +1-3pp on AIME or strengthened null | GO (already planned) |
| T1.5 | K1 | Pólya 1945 / WHO 2009 | 2-3 days | +2-5pp via `tiebreak_lex` extension | GO |
| T1.6 | K14 | Foygel-Barber 2023 + Guan 2023 | 6-10 GPU hr | OOD cov 0.45-0.55 vs T3 overshoot | GO (HGJ #1) |
| T1.7 | K10 | Cook 2002 (FEA) | 1 week | mean-N≈5 at SC@16 acc | GO if T1.1 delayed |
| T1.8 | K11 | CPC 2018 / PMI | 1 week | new score + unified eval | GO |
| T1.9 | K8 | Tao + Fermi | 3-5 days | +1-3pp on AIME | GO |
| T1.10 | K17 | M&M / NTSB / AAR | 1 week | new §5 figure | GO |

---

## §5 Tier 2 — Validated, medium-value, v2 / future-work

Sixteen Tier 2 picks. Pass V1V2V3V4 but defer to v2 / future work for cost, narrow scope, or redundancy with Tier 1.

| # | Cluster | Source | Defer reason |
|---|---|---|---|
| T2.1 | K2 | CROSS-DISC §A2 SPRT | Likely empirically tied with T1.1 e-process; report as alternative in §6 |
| T2.2 | K2 | CROSS-DISC §A3 BOCPD | Specifically for long-CoT models (R1-Distill 60+ steps); 3-5 days but expensive on GPU; v2 |
| T2.3 | K2 | CROSS-DISC §D1 POMDP | Formalizes Layer B; 1-2 weeks; v2 |
| T2.4 | K6 | MATH_SCI §G1 HAZOP guidewords | Most thorough probing protocol; 1-2 weeks LLM calls; v2 |
| T2.5 | K6 | MATH_SCI §A2 Lakatos refutations | Olympiad-flavored; 1 week; v2 |
| T2.6 | K9 | NON_AI §D1 IRAC structured CoT | Per-component scoring; 1 week; v2 |
| T2.7 | K9 | NON_AI §L1 chain of custody schema | Mechanical step verifiability; 2 weeks; v2 |
| T2.8 | K10 | CROSS-DISC §H1 RKF45 adaptive segmentation | Direct mathematical analogy with 60-year-old algorithm; 2 weeks; v2 |
| T2.9 | K13 | CROSS-DISC §I2 Huber-trimmed φ ablation | Robust aggregation; 2-3 days; v2 ablation |
| T2.10 | K15 | NON_AI §A3 SBAR verbalized confidence | Opens API-only models (closed-weight); 1 week; v2 |
| T2.11 | K16 | NON_AI §B1 + §A1 + §F1 pre-decode prompt experiments | DDx / checklist / OODA prompting bundle; 1 week; v2 |
| T2.12 | K20 | CROSS-DISC §D3 CVaR risk-sensitive RL | Training-time companion to inference-time CP; separate paper |
| T2.13 | K21 | CROSS-DISC §K1 counterfactual / front-door for step rejection | Could explain Pilots C/K/L null; 2-3 weeks; v2 |
| T2.14 | K22 | CROSS-DISC §F1 + §F2 BnB / MCTS with CP-calibrated UCB | Bridges MCTS and CP; 4+ weeks; v2 |
| T2.15 | K24 | CROSS-DISC §B3 metacognition m-ratio | Cheap §5 figure; 2-3 days; v2 if space allows |
| T2.16 | K28 + K30 | NON_AI §A2 + §F4 + §L2 multi-source / Auftragstaktik / cognitive-interview prompt suite | Refines Pilot M; 1-2 weeks; v2 |

(Plus cluster K3 multi-source LOPA-style §5 ablation = 3-5 days is also Tier 2 if §5 has space.)

---

## §6 Tier 3 — Framing / rhetorical only

Validated but no new method — pure framing for §1 motivation, §3 theorem framing, or §6 deployment narrative.

| # | Cluster | Source | Use |
|---|---|---|---|
| T3.1 | K23 | CROSS-DISC §B1 Cognitive Load Theory + worked examples | §1 motivation: theoretical backbone for step segmentation |
| T3.2 | K24 | CROSS-DISC §B2 Kahneman dual-process | §1 framing: CP filter = Type 2 deliberation; raw greedy = Type 1 |
| T3.3 | K2 | CROSS-DISC §C1 + §C2 predictive coding / Bayesian brain | §1 / §6 normative argument for entropy-based scores |
| T3.4 | K11 | CROSS-DISC §E1 rate-distortion theory | §3 anchor: our compute-Pareto frontier is an R(D) curve; Shannon-style impossibility argument |
| T3.5 | K13 | MATH_SCI §E3 tolerance stacking | §3 framing: min/mean/last are *design choices* mapping to worst-case-vs-RSS in mech eng |
| T3.6 | K25 | MATH_SCI §B6 blind analysis | Reproducibility appendix; explicitly state pre-registered protocol for OOD evaluation |
| T3.7 | K25 | MATH_SCI §B5 5σ discovery threshold | §5 reporting discipline: report at α-margin (target α=0.10, achieve α=0.05 empirically) |
| T3.8 | K25 | NON_AI §D3 Daubert standard | §3 framing: each score family passes Daubert's testable / peer-reviewed / known-error / accepted criteria |
| T3.9 | K26 | NON_AI §D2 + MATH_SCI §E1 burden of proof / Factor of Safety | §6 deployment narrative: ship a (use case, stake, recommended α, achieved kept-accuracy) table |
| T3.10 | K26 | NON_AI §J1 BATNA | §6: abstain has utility; pair α-grid with utility argument |
| T3.11 | K27 | MATH_SCI §F5 PLL for cadence | §6 future work: cadence regulation in long-CoT models |
| T3.12 | K17 | NON_AI §A5 APACHE-II / triage | §3 framing: score is a *routing* signal, not a *diagnosis* — sharpens conceptual separation |
| T3.13 | K9 | NON_AI §F1 OODA loop | §1 framing: forced Orient phase before Decide / Act |

---

## §7 Tier 4 — Rejected

Failed at least one validation test. Each entry: which test failed, evidence, recommendation.

### R1 — Symbolic execution / abstract interpretation (CROSS-DISC §L2)

- **Failed test**: V3 (feasibility) — already tried (Wave 2 agent_C, 4% extraction rate, marked failed). V4 partially (existing `arith_violations` already does light abstract-interpretation via `safe_eval`).
- **Evidence**: From `METHOD_AND_RESULTS.md` and Wave 2 notes, full symbolic execution of CoT traces had a 4% successful-extraction rate. Abstract interpretation (sign / magnitude / type) is plausible but gives small lift over `arith_violations`.
- **Recommendation**: **DROP** as a v1 contribution. Keep as an internal infra item; do not advertise as cross-disciplinary import.

### R2 — Property-based testing / Hypothesis library (CROSS-DISC §L1)

- **Failed test**: V4 (novelty) — engineering hygiene, already a best practice in CP libraries (`mapie`, `crepes`).
- **Evidence**: Standard recommendation in any CP library tutorial; not a contribution.
- **Recommendation**: **DROP.** Use internally for §3 / appendix experimental validation but do not write up as a candidate import.

### R3 — Sectional rehearsal / jazz fakebook (NON_AI §G1, §G2)

- **Failed test**: V4 (novelty) — sub-problem decomposition prompts are already widely deployed in CoT literature; named-transformation retrieval is speculative without a clear evaluation harness.
- **Evidence**: Decomposition prompts ("first solve the algebra, then the geometry") appear in standard CoT tutorials; jazz-fakebook of named transformations would require building a 30-50 entry curated transformation list with no clear benchmark.
- **Recommendation**: **DROP.** Cosmetic borrowing; the underlying idea (sub-problem decomposition) is already standard.

### R4 — Tournament seeding / Swiss style (NON_AI §H3)

- **Failed test**: V2 (mapping soundness) — proposed mapping is "evaluate score families more sample-efficiently via Swiss pairing," which is an evaluation-methodology improvement, not a CoT-CP contribution.
- **Evidence**: NON_AI §H3 itself acknowledges "Value: low — optimization, not contribution."
- **Recommendation**: **DROP.**

### R5 — Confidence-Building Measures / CBMs (NON_AI §J2)

- **Failed test**: V2 (mapping soundness, partial) — proposed CBM check (cheap shared verification) collapses to "use sympy as a low-cost verifier," which is already in our zoo.
- **Evidence**: The candidate's "concrete proposal" is just "force both models to verify a sub-question," which is a special case of cross-model verification.
- **Recommendation**: **DROP** as standalone candidate; absorbed by existing CMV protocol.

### R6 — Sterile field maintenance double-glove (NON_AI §C3)

- **Failed test**: V4 (novelty) — `tiebreak_lex` already implements layered redundancy. The "layer more aggressively on highest-stakes steps" idea has diminishing returns and no clear experimental hook.
- **Recommendation**: **DROP** as a separate import; mention as motivation for `tiebreak_lex` in §3 if useful.

### R7 — Six Sigma DMAIC cycle (NON_AI §E4)

- **Failed test**: V2 (mapping soundness) — meta-methodology; gives a process template, not a contribution.
- **Recommendation**: **DROP.**

### R8 — Phase-locked loop (PLL) for cadence (MATH_SCI §F5)

- **Failed test**: V2 (mapping soundness) + V3 (feasibility, partial) — the proposed mapping (regulate average step length) overlaps EDU-PRM's segmentation, has unclear evaluation, and the PLL mathematics doesn't transfer cleanly.
- **Recommendation**: **DROP** as a candidate; keep as §6 future-work mention.

### R9 — Stoichiometry / mass-energy balance (MATH_SCI §C3, §G3)

- **Failed test**: V4 (novelty within cluster K7) — already covered by conservation laws (B3) and counted in the sanity-check primitives bundle.
- **Recommendation**: **DROP** as separate candidates; absorbed into T1 sanity-check bundle (Cluster K7 / T1.5+).

### R10 — Reverse mathematics citation-validity (MATH_SCI §A5)

- **Failed test**: V3 (feasibility) — requires a knowledge-base integration (Wikipedia API or mathlib4) for theorem-name validation; 2 weeks for narrow benefit.
- **Recommendation**: **DROP for v1; revisit for v2** as a "named theorem citation accuracy" benchmark on Olympiad problems.

### R11 — Lean 4 formal proof verification (MATH_SCI §A6)

- **Failed test**: V3 (feasibility) — autoformalization rate is ~30% on Olympiad problems (DeepMind AlphaProof 2024); 2-3 weeks integration; Lean inference 30-60 sec/problem.
- **Recommendation**: **DEFER to v2/companion paper.** Real and high-value but out-of-scope for the v1 TMLR target. (Borderline — could be Tier 2 if we extend the timeline.)

### R12 — NTSB-style public archive (NON_AI §B4)

- **Failed test**: V3 (feasibility) — building a public-archive standalone artifact is multi-month scope; subsumed by simpler M&M-style §5 figure (T1.10).
- **Recommendation**: **DEFER** to a follow-up benchmark paper; do not advertise in v1.

### R13 — Editorial chain of review (NON_AI §I3)

- **Failed test**: V4 (novelty) + V2 (mapping soundness, partial) — multi-pass editing is an engineering pipeline, not a contribution; many existing systems do this.
- **Recommendation**: **DROP.**

### R14 — Replay review / NFL challenge budget (NON_AI §H2)

- **Failed test**: V4 (novelty) — already in `gate_combined_f0.5_K4` (Tier 5 gating); the "limit re-rolls per trace" idea has small effect.
- **Recommendation**: **DROP.**

### R15 — Tournament / Swiss seeding for evaluation (NON_AI §H3) — duplicate of R4

(See R4.)

---

### Tier 4 summary

15 candidates rejected (or deferred as out-of-scope for v1). The dominant rejection reason is V4 (novelty) — many "borrowings" recreate ideas already standard in our pipeline, in CP libraries, or in the within-AI literature.

---

## §8 Implementation roadmap (4-week schedule)

Aligned with the existing `theorems/PAPER_OUTLINE.md` "critical-path remaining work" and the HGJ_review_feedback.md TMLR plan.

### Week 1 — Cheap, high-value Tier 1 picks + already-planned HGJ items

| Day | Task | Source | Effort | Output |
|---|---|---|---|---|
| Mon-Tue | T1.3 Diverse Beam Search ablation on Pilots C/K/L | DBS Vijayakumar+ 2018 | 2-3 days | Empirical: does DBS unblock the K=4 null result? |
| Wed-Thu | T1.5 Pólya Look Back primitive | Pólya 1945 | 2-3 days | Binary score, plug into `tiebreak_lex` |
| Fri | T1.4 Earliest-bad-step re-roll (already HGJ Idea 1.1) | Toyota / CPM justification | 4-6 GPU hr | Layer B variant + Five-Whys / CPM citation |
| Weekend | HGJ Idea 3.1 Qwen3-32B thinking mode (parallel) | (HGJ planned) | 3-4 hr | Reasoning-model coverage |

GPU hours expected: ~12-16 hours total on 2× H100 NVL. All experiments fit on existing infra.

### Week 2 — Mathematical upgrades + OOD / cross-dataset

| Day | Task | Source | Effort | Output |
|---|---|---|---|---|
| Mon-Wed | T1.6 Local CP K=30-60 + cross-dataset transfer matrix (HGJ #1) | Foygel-Barber 2023 + Guan 2023 | 6-10 GPU hr | OOD §5 figure |
| Wed | NEW (HGJ review): Coverage gap by α figure | Theorem 1 empirical-vs-theoretical | 1-2 hr | §5 figure |
| Thu-Fri | T1.9 Two-pass Fermi heuristic-rigorous cross-check | Tao + Fermi | 3-5 days | Score family + ablation |
| Weekend | T1.7 Mesh-refinement adaptive-N SC | Cook 2002 (FEA) | spillover into Week 3 | Convergence-criterion stop |

GPU hours expected: ~10-15 hours.

### Week 3 — Theory polish + paper drafting

| Day | Task | Source | Effort | Output |
|---|---|---|---|---|
| Mon | T1.2 BH-FDR per-step CP | Benjamini-Hochberg 1995 + Bates+ 2023 | 1 week (theory + experiment overlap) | Theorem 1' / Approach B' update |
| Tue-Wed | T1.1 e-process / anytime SC | Ramdas+ 2023 | 2 weeks (overlap into Week 4) | Theorem 1 extension; experimental validation on MATH-500 |
| Thu | T1.8 PMI as score | CPC 2018 | 1 week (overlap into Week 4) | New mid-cost score family |
| Fri | T1.10 LLM-judge AAR pass on 50-100 failed traces | M&M / NTSB / AAR | 1 week (overlap into Week 4) | Failure-mode taxonomy |
| Sat-Sun | LaTeX skeleton; theorem polish (T1 → i.i.d.; T2 → LR+ rename; T3 → A1 weakening) | THEOREM_REVIEW.md | 0 GPU | Paper draft 30% |

### Week 4 — Final experiments + writing

| Day | Task | Effort | Output |
|---|---|---|---|
| Mon-Wed | Finalize T1.1, T1.8, T1.10 outputs | spillover | All Tier 1 done |
| Thu | Tier 3 framing inserts (blind analysis B6, Daubert D3, FoS E1, BATNA J1) | 0 GPU, ~1 day writing | §1, §3, §6 framing |
| Fri | Final figures (E5v2 publication-grade; new T1.6 OOD figure; T1.10 per-failure-mode breakdown) | 1 day | Figures done |
| Sat-Sun | Paper draft 100%; reviewer-style self-cross-check pass | 0 GPU | TMLR submission ready |

### Total expected GPU hours

- Week 1: 12-16 hours
- Week 2: 10-15 hours
- Week 3: spillover ≤ 10 hours
- Week 4: ≤ 5 hours
- **Total: 30-50 GPU hours over 4 weeks** on 2× H100 NVL — comfortably within budget.

### Dependencies

- T1.1 (e-process) extends T1 theorem statement; requires THEOREM_REVIEW.md i.i.d. fix first.
- T1.2 (BH-FDR) modifies per-step CP Approach B; runs after Pilot 8/J coverage validation is re-checked.
- T1.6 (Local CP) cites Foygel-Barber 2023 + Guan 2023 + Lei-Wasserman 2014 — citation correction needed before write-up.
- T1.10 (failure-mode taxonomy) feeds back into score-design; could surface new Wave-5 score candidates mid-write.
- DBS (T1.3) and earliest-bad-step (T1.4) both target the K=4 null result; results should be reported jointly to avoid double-counting.

### Match to existing PAPER_OUTLINE.md critical-path

The PAPER_OUTLINE.md "critical-path remaining work" lists:
1. Theorem 1 from Codex ← Done internally
2. Cross-verify theorems ← Done internally (THEOREM_REVIEW.md)
3. Truncation rerun ← Done (E6v3, E12r-E15r in `METHOD_AND_RESULTS.md`)
4. Polish figures
5. Write the paper

The roadmap above adds Tier 1 cross-disciplinary picks **inside** items 4-5 without lengthening the critical path: the cheap Tier 1 picks (T1.3 DBS, T1.5 Look Back, T1.4 earliest-bad-step) drop into Week 1 alongside HGJ 3.1 / 1.1; the mathematical upgrades (T1.1, T1.2, T1.6) become Theorem-revision items aligned with the existing T1/T2/T3 fixes; and the framing-only Tier 3 picks (B6 blind analysis, D3 Daubert, J1 BATNA) cost zero GPU and slot into the Week 4 writing pass.

---

## §9 Cross-cutting themes

Six themes that span clusters and surface deeper structure than any single import.

### Theme T-A — "Forced verification" is the most universal cross-domain practice

- **Spans**: Pólya Look Back (math), WHO Surgical Time-Out (medicine), pre-decode checklist (aviation), boundary-case verification (physics), sponge count (surgery), look-elsewhere effect / 5σ (HEP), CRC mandatory objection (aviation CRM), After-Action Review (military).
- **Unifying claim**: Across math, medicine, surgery, aviation, particle physics, and military doctrine, the single most-codified discipline is **mandatory, structured, post-action verification with binary pass/fail outcome**. Every mature operational discipline enforces this; LLM CoT does not.
- **Implication for CoT-CP**: A binary `LookBackPassed ∈ {0, 1, 2, 3}` signal (Cluster K1, T1.5) plugs into trajectory aggregator φ as a free additional component. The cross-domain weight of the practice is much heavier than any single citation suggests; the *cluster* — not the individual import — is what should be cited.

### Theme T-B — "Recursive belief update" is one mathematical object across many fields

- **Spans**: Kalman filter (electrical engineering), e-process / test-by-betting (statistics), CUSUM / BOCPD (numerical analysis / process control), POMDP belief states (decision theory), Bayesian brain (neuroscience), predictive coding (theoretical neuroscience).
- **Unifying claim**: Every one of these is a special case of "maintain a posterior over a hidden state; predict-update at each observation; threshold the posterior for a decision." The CoT-CP analogue is "is this trace going to be correct?" with step scores as observations.
- **Implication for CoT-CP**: Per-step CP Approach A's heuristic running-min / running-mean is a *degenerate filter*. The principled replacement is e-process (T1.1) for the frequentist contract or Kalman-style Bayesian filter (Cluster K2 v2) for the Bayesian contract. Both subsume each other under different framings — this is the most mathematically dense cluster.

### Theme T-C — "Adversarial probing" is structurally different from self-critique

- **Spans**: HAZOP guidewords (chemical engineering, ICI 1960s), FMEA (US Mil 1949 / NASA / automotive), Lakatos refutations (philosophy of math), Red Team (US Army TRADOC), cross-examination (Anglo-American law, Wigmore 1904), CRM 2-pilot mandatory objection (aviation, post-Tenerife 1977), Daubert gatekeeping (federal evidence, 1993).
- **Unifying claim**: Self-critique by the same model has *shared blind spots*. Robust verification requires either (i) a *different* model (Red Team / cross-examination / CRM) or (ii) a *systematic probe-set* (HAZOP guidewords / FMEA / VINDICATE / Lakatos counter-examples). Both move beyond CoVe-style self-verification.
- **Implication for CoT-CP**: Existing CMV protocol already implements (i). HAZOP-style guideword probing (T2.4) is a v2 implementation of (ii) — most thorough verification protocol available, at 3-4× LLM-call cost.

### Theme T-D — "Multi-source independence" repeats verbatim across distant fields

- **Spans**: two-source rule (journalism, codified at Washington Post during Watergate), multi-spectroscopy (chemistry, NMR + IR + MS), multi-wavelength survey (astronomy, LSST), blinded clinical concurrence (medicine, Joint Commission NPSG.03.04.01), Hamming codes (communication theory, 1950), Layer of Protection Analysis (chemical process safety, AIChE CCPS), distance ladder (astronomy, parallax → Cepheids → SNe Ia).
- **Unifying claim**: A claim is accepted iff ≥2 *independent* probes (different model / different prompt / different score family / different physics) agree. Independence — not count — is the criterion.
- **Implication for CoT-CP**: `tiebreak_lex` (lp + prm + sc) already implements this implicitly. The cross-domain weight motivates a §5 LOPA-style ablation (Cluster K3) showing accuracy floor as we add layers — the *flatness* of the curve diagnoses dependence; *steepness* diagnoses independence.

### Theme T-E — "Earliest divergence is the cascade source" — a Pearl-causal claim that crosses fields

- **Spans**: Five Whys (Toyota Production System, c. 1930), Critical Path Method (Kelley-Walker 1959, DuPont/Remington Rand), HGJ Idea 1.1 earliest-bad-step (internal), proof debugging in formal mathematics (where the first faulty lemma is the bug, later derivations propagate).
- **Unifying claim**: The proximate (worst) symptom is rarely the cascade source; the *earliest* upstream divergence is. Re-rolling at the symptom doesn't fix the cascade.
- **Implication for CoT-CP**: The cluster gives Pilot C/K/L's null result a *causal* explanation: re-rolling at the worst step keeps the wrong upstream prefix, so all K alternatives inherit the cascade. Re-rolling at the *earliest* threshold-violating step (T1.4) is the minimal intervention point in Pearl's do-calculus — and is the experiment HGJ already prioritized.

### Theme T-F — "Heuristic-rigorous duality" is taught in math and physics but absent from LLMs

- **Spans**: Tao "What's New" career advice (math, post-rigorous mode), Fermi estimation (physics, back-of-cocktail-napkin), Erdős "Book proofs" (math, multiple proofs same theorem), Kahneman dual-process (cognitive science, Type 1 vs Type 2), Boas's *Mathematical Methods in the Physical Sciences* boundary-case verification.
- **Unifying claim**: Strong reasoning *alternates* between heuristic (cheap, approximate, error ≤ 10×) and rigorous (slow, careful, error → 0) modes. Pre-rigorous-only is fast but unreliable; rigorous-only is verbose and slow. Cross-mode consistency check catches gross errors (off-by-10× from a misplaced decimal).
- **Implication for CoT-CP**: Two-pass Fermi+rigorous (T1.9) with cross-check is 3-5 days and is the cheapest gross-error catch; AIME's 0-999 integer answers make this especially cheap. Tao's framing additionally gives the §1 motivation a non-CP backbone.

---

## §10 Honest assessment

### Where are the candidates *genuinely novel* contributions vs *cosmetic borrowings*?

**Genuinely novel contributions (Tier 1):** Out of 109 entries, only ~6-8 candidates contribute a *new methodological wedge* over the within-AI 30-paper synthesis:

1. **T1.1 e-process / anytime SC** — converts heuristic Adaptive-Consistency / ESC into a frequentist-coverage-guaranteed procedure. New theorem extension (Theorem 1 to anytime). No within-AI paper does this.
2. **T1.2 BH-FDR for per-step CP** — strict tightening of our Bonferroni Approach B; Bates-Angelopoulos-Lei-Romano 2023 directly applies. New theorem result.
3. **T1.6 Local CP via SBERT KNN** — new OOD calibration mode complementing Theorem 3; HGJ Idea 2.1.
4. **T1.8 PMI as unified score evaluation** — new evaluation metric across the score zoo; not just a new score, a *measurement principle*.
5. **T1.10 per-failure-mode kept-accuracy** — structurally new §5 figure that radiology AI does and CoT papers don't.
6. (Borderline) **T1.5 Pólya Look Back binary score + T1.9 Fermi cross-check + T1.7 mesh-refinement adaptive-N**: Each is a small empirical addition. None individually is a paper, but bundled they strengthen the experimental section by 2-3 figures.

**Cosmetic borrowings (Tier 3 + Tier 4):** ~20 entries pretend to be methodological contributions but are actually pure framing or already-standard practice:

- "Daubert standard" (NON_AI §D3) — pure framing, not a method.
- "Burden of proof gradations" / "Factor of Safety" / "BATNA" — reframings of α; useful for §6 deployment narrative but not novel.
- "Sterile cockpit" / "Auftragstaktik" / "Mise en place" / "Tasting at intervals" — re-paintings of structural CoT prompts, mostly already standard.
- "OODA loop" / "IRAC" / "DDx VINDICATE" — structured prompting templates; many already deployed.
- "M&M conferences" / "After-Action Review" / "NTSB" — same idea (failure-mode taxonomy) re-cited from three different fields. Counting as three separate imports is double-counting.
- "Sectional rehearsal" / "Jazz fakebook" / "Tournament seeding" / "Replay review NFL challenge" — speculative or already covered.

### How much of the value is actually new method vs better framing / paper presentation?

Honest split:
- **~40% framing**: §1 motivation (CLT, dual-process, predictive coding, Daubert), §3 framing (LR+ rename, Pareto = R(D), tolerance stacking), §6 deployment (FoS, BATNA, burden of proof). These are zero-GPU writes that strengthen reviewer reception but don't change the empirical content.
- **~30% empirical reinforcement**: DBS, earliest-bad-step, Pólya Look Back, Fermi cross-check, mesh-refinement adaptive-N. Each is +1-3pp on a particular subset, none is a paper individually, but 3-4 of these together turn §5 from "+10-45pp on existing scores" to "+10-45pp on existing scores with a structured zoo of complementary primitives."
- **~30% genuine method**: e-process / anytime SC, BH-FDR, Local CP, PMI evaluation, per-failure-mode taxonomy. These are the candidates that survive the strictest reviewer pass — the ones where a NeurIPS/ICLR reviewer would say "yes, this is novel."

### If we had to pick 3 candidates only, which 3 maximize paper acceptance probability?

If the venue is **TMLR** and the paper is empirical-heavy:

1. **T1.6 Local CP** — already HGJ priority #1; addresses Theorem 3's documented overshoot; cheap (6-10 GPU hr); unambiguously strengthens §5.6 OOD section; clean Foygel-Barber 2023 + Guan 2023 + Lei-Wasserman 2014 citation triple.
2. **T1.10 per-failure-mode taxonomy via M&M / NTSB / AAR** — structurally new §5 figure not present in any cited CoT paper; cheap (1 week LLM-judge); turns aggregate accuracy into structured diagnostic; strong reviewer signal.
3. **T1.3 Diverse Beam Search ablation** — 2-3 days, directly resolves the documented Pilots C/K/L null-result bottleneck. If positive: rescues the step-level branching story. If null: strengthens the "step-level branching ceiling" claim with a new piece of negative evidence — both outcomes are paper-positive.

If the venue is **NeurIPS / ICLR main** (3-6 month timeline, Theorem 3 elevated to methodological contribution):

1. **T1.1 e-process / anytime SC** — single biggest mathematical upgrade; replaces Adaptive-Consistency / ESC competitor turf with a frequentist coverage guarantee; new Theorem statement. Top venue would value this most.
2. **T1.2 BH-FDR for per-step CP Approach B** — strict tightening; clean theorem result; Bates-Angelopoulos-Lei-Romano 2023 base.
3. **T1.6 Local CP** — same as TMLR pick #1.

The 3-pick-only test is harsh: T1.4 earliest-bad-step, T1.5 Pólya, T1.9 Fermi, T1.7 mesh-refinement — all are individually attractive but each adds at most +1-3pp; none individually moves a reviewer from reject to accept. The *bundle* (4-6 of the cheap ones together with one mathematically clean upgrade) is what sells the paper.

### Brutal honesty about cluster value

Out of 30 clusters:
- **3 clusters are paper-strength contributions** (K2 e-process / Kalman, K12 BH-FDR, K14 Local CP).
- **5 clusters are §5-strengthening empirical adds** (K1 Look Back, K4 earliest-bad-step, K5 DBS, K8 Fermi, K17 failure-mode taxonomy).
- **5 clusters are §3 / §6 framing** (K23 CLT, K25 blind analysis / Daubert, K26 burden of proof / FoS, K11 R-D theory, K13 tolerance stacking).
- **10 clusters are deferred to v2** (K3 LOPA, K6 HAZOP, K9 IRAC, K10 RKF45, K15 SBAR, K16 pre-decode bundle, K18 BMA, K19 MPC, K20 CVaR, K21 causal, K22 MCTS).
- **7 clusters are weak / overlapping** (K7 sanity-check primitives — narrow each; K13 Huber — small ablation; K24 m-ratio — cheap figure; K27 cadence; K28 paraphrase consensus; K29 calibration card; K30 Auftragstaktik).

The 109-entry corpus is *ideationally* rich but *methodologically* concentrated: most of the value lives in 3-5 mathematical ideas (Cluster K2 + K12 + K14), 1 evaluation principle (K11 PMI), and 1 §5-restructuring move (K17 failure-mode taxonomy). Everything else is supporting framing or v2.

---

## Appendix — File and section citation index

For each candidate referenced in this document, the canonical (file, section) pair:

- `CROSS_DISCIPLINARY_IDEAS.md` (32 entries):
  - §A: A1 e-process, A2 SPRT, A3 BOCPD, A4 optimal stopping
  - §B: B1 CLT/worked-examples, B2 dual-process, B3 metacognition
  - §C: C1 predictive coding, C2 Bayesian brain
  - §D: D1 POMDP, D2 MPC, D3 CVaR, D4 BMA
  - §E: E1 rate-distortion, E2 information bottleneck, E3 PMI
  - §F: F1 branch-and-bound, F2 MCTS, F3 DBS
  - §G: G1 BFT/quorum, G2 boosting
  - §H: H1 RKF45, H2 Richardson extrapolation
  - §I: I1 BH-FDR, I2 Huber
  - §J: J1 active learning, J2 self-explanation
  - §K: K1 counterfactual
  - §L: L1 property-based testing, L2 symbolic execution

- `NON_AI_DOMAIN_PRACTICES.md` (43 entries):
  - §A medicine: A1 DDx, A2 concurrence, A3 SBAR, A4 M&M, A5 APACHE-II
  - §B aviation: B1 cockpit checklist, B2 CRM, B3 sterile cockpit, B4 NTSB
  - §C surgery: C1 WHO time-out, C2 sponge count, C3 sterile field
  - §D law: D1 IRAC, D2 burden of proof, D3 Daubert, D4 cross-examination
  - §E quality eng: E1 FMEA, E2 Five Whys, E3 SPC, E4 DMAIC, E5 Poka-yoke
  - §F military: F1 OODA, F2 AAR, F3 Red Team, F4 Auftragstaktik
  - §G performing arts: G1 sectional rehearsal, G2 jazz fakebook
  - §H games: H1 chess multipv, H2 NFL replay, H3 Swiss seeding
  - §I journalism: I1 two-source, I2 inverted pyramid, I3 editorial chain
  - §J diplomacy: J1 BATNA, J2 CBMs
  - §K photography: K1 exposure bracketing, K2 calibration cards
  - §L forensics: L1 chain of custody, L2 cognitive interview
  - §M cooking: M1 mise en place, M2 tasting at intervals

- `MATH_SCI_ENG_PRACTICES.md` (33 entries):
  - §A math: A1 Pólya, A2 Lakatos, A3 Erdős book, A4 Tao, A5 reverse math, A6 Lean
  - §B physics: B1 Buckingham π, B2 Fermi, B3 conservation laws, B4 boundary cases, B5 5σ, B6 blind analysis
  - §C chemistry: C1 retrosynthesis, C2 multi-spectroscopy, C3 stoichiometry
  - §D astronomy: D1 distance ladder, D2 multi-wavelength
  - §E civil/mech: E1 FoS, E2 FEA mesh refinement, E3 tolerance stacking, E4 CPM, E5 Weibull
  - §F EE: F1 Hamming codes, F2 lock-in amplifier, F3 Kalman filter, F4 ROC, F5 PLL
  - §G chem-process: G1 HAZOP, G2 LOPA, G3 mass balance, G4 P&ID

---

## Closing note

The 109-entry catalog rewards aggressive clustering: 30 substantive clusters cover ~95% of the methodological content. The validation pass identifies ~10 Tier 1 imports worth implementing, ~16 Tier 2 imports for v2, ~13 Tier 3 framing-only inserts, and ~15 Tier 4 rejections. Total realistic v1 yield: a 2-3pp improvement on `tiebreak_lex` from sanity-check primitives + Look Back + Fermi cross-check; a structurally new §5 figure (per-failure-mode taxonomy); an OOD §5.6 strengthening (Local CP); and one mathematically clean theorem extension (e-process or BH-FDR). Outside that core, the remaining 80 candidates are useful for §1/§3/§6 framing or for v2/follow-up papers, and a quarter of them earn outright rejection on novelty / feasibility grounds.

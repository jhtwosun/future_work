# T1.10 Verification — Per-failure-mode kept-accuracy via M&M / NTSB / AAR taxonomy

> **Pick**: Run an LLM-judge "after-action-review" pass on 50–100 failed traces (low-quantile and falsely-kept), induce a small M&M / NTSB / AAR-style failure-mode taxonomy (8–12 modes), then report per-mode kept-accuracy for each CoT-CP score family at fixed answer rate. Headline shape: *"CoT-CP at α=0.5 catches 73% of arithmetic-error traces and 41% of premature-commitment traces; the gap motivates score-family hybridization."*
>
> **Author**: Verifier (Claude Opus 4.7, 1M ctx).
> **Date**: 2026-05-08.
> **Status**: V1–V6 complete. Final verdict at the bottom.
> **One-line summary**: The gap T1.10 targets is **half-filled**. ErrorAtlas (Jan 2026, 17-cat hierarchical) and FLARE (ACL-OMMM 2025, 7-cat) own the *taxonomy* axis; ProcessBench owns the *step-error annotation* axis; what nobody has done yet is **conditioned per-mode kept-accuracy across a *score family ladder***. That conditional is real, narrow, and worth a §6 sub-section — but **not** a standalone paper.

---

## V1 — Prior art deep search

### V1.1 Search queries executed (8)

1. `ErrorAtlas LLM error taxonomy 17 categories hierarchical 2025`
2. `FLARE failure mode LLM reasoning ACL 2025 taxonomy`
3. `"failure mode" taxonomy LLM chain-of-thought traces error categories per-category accuracy`
4. `"morbidity and mortality" OR "after action review" OR "NTSB" LLM error analysis taxonomy`
5. `LLM judge classify error categories chain-of-thought failed traces 2025 2026`
6. `ProcessBench step-level error annotation reasoning LLM math 2024 2412.06559`
7. `"per-mode accuracy" OR "selective accuracy" failure mode breakdown LLM math reasoning conformal`
8. `arXiv 2511.19933 failure modes LLM systems`

### V1.2 Direct prior art (in order of threat to T1.10)

**A. Sundar et al. — *ErrorMap and ErrorAtlas: Charting the Failure Landscape of Large Language Models* (arXiv:2601.15812, January 2026).**
Single biggest threat to T1.10's "build a taxonomy" framing. ErrorAtlas is **already** the standardized 17-category hierarchical taxonomy of LLM failure modes, derived empirically by ErrorMap (an LLM-judge pipeline) over **35 datasets × 83 LLMs × ~7,000 sampled failures** (~10% of the failed-instance pool). Top categories by prevalence: *Missing Required Element* (15.56%), *Specification Misinterpretation* (11.50%), *Logical Reasoning Error* (9.09%), *Incorrect Identification* (8.98%), *Computation Error* (8.45%) — i.e., exactly the kind of bins T1.10 was going to re-derive from 50–100 traces. Validated judge-accuracy ≈ 92%. **The taxonomy-construction work is done.**

**B. Patel et al. — *FLARE: An Error Analysis Framework for Diagnosing LLM Classification Failures* (ACL-OMMM 2025, aclanthology.org/2025.ommm-1.4).**
7-category taxonomy (E1–E3 technical, E4–E7 semantic) for "Inconclusive" classifier outputs. Applied to 5,400 election-misinformation classifications; reports per-mode prevalence (e.g., 70.8% of Few-Shot failures were E1 parsing). FLARE *already* exhibits the per-mode quantitative breakdown T1.10 is proposing — but on a classification task with only 7 modes and no selective-accuracy / conformal layer.

**C. Vinay — *Failure Modes in LLM Systems: A System-Level Taxonomy for Reliable AI Applications* (arXiv:2511.19933, November 2025).**
15-mode system-level taxonomy: multi-step reasoning drift, latent inconsistency, context-boundary degradation, version drift, etc. Closer to production observability than to per-trace error analysis, but it occupies the "named taxonomy of LLM failures" slot.

**D. Zheng et al. — *ProcessBench: Identifying Process Errors in Mathematical Reasoning* (arXiv:2412.06559, ICLR 2025).**
3,400 competition-level math test cases with **human-annotated earliest-erroneous-step locations**. Already in our 30-paper. Provides the *step-error-annotation infrastructure* (location of first error), but does **not** provide a typed taxonomy — only "step *k* is the first wrong step." T1.10's "what *kind* of error" axis is orthogonal and unfilled here.

**E. Cemri et al. — *Why Do Multi-Agent LLM Systems Fail? A Taxonomy of Failure Modes in MAS* (MAST, arXiv:2503.13657, 2025).**
Grounded-Theory-built taxonomy from 150 MAS execution traces × 6 expert annotators × κ=0.88 inter-annotator agreement. Demonstrates that the **methodology** T1.10 proposes (LLM/expert-judge → cluster → label → audit → publish taxonomy) is already well-established in the LLM-failure-mode literature. T1.10 cannot claim methodological novelty here.

**F. Kim et al. — *When the Chain Breaks: Interactive Diagnosis of LLM Chain-of-Thought Reasoning Errors* (arXiv:2603.21286, March 2026).**
LLM-classifier categorizing reasoning steps by *functional role* (problem setup, plan generation, fact retrieval, active computation, uncertainty management, result consolidation, self-checking, final answer emission). A *role* taxonomy rather than a *failure-mode* taxonomy, but it overlaps with T1.10's "premature-commitment-style modes" via the "result consolidation" / "self-checking" categories. Not a direct hit but compresses the available framing space.

**G. *Demystifying Errors in LLM Reasoning Traces: An Empirical Study of Code Execution Simulation* (arXiv:2512.00215, December 2025).**
9-category taxonomy of code-reasoning errors: Computation, Indexing, Control Flow, Skip Statements, Misreporting Final Output, Input Misread, Misevaluation of Native API, Hallucination, Lack of Verification/Logic Following. Domain-shifted (code-execution simulation, not free-form math), but again pre-empts the "build a 9-ish-category taxonomy of math-reasoning errors" framing.

**H. Winston et al. — *A Taxonomy of Failures in Tool-Augmented LLMs* (TALLM, AST 2025).**
6-cat root-cause taxonomy for tool-augmented LLMs. Adjacent axis (tool calls), not direct.

**I. *Evaluating LLMs' Ability to Handle Mistakes in Mathematical Reasoning* (ACL 2025 long paper, aclanthology.org/2025.acl-long.1313).**
Per-error-type evaluation on math problems with deliberately-injected mistakes. The closest published analog to T1.10 in *spirit* — they pre-define mistake types (factual, computational, logical, conceptual) and measure model behavior conditional on mode. Differs in that the mistakes are injected, not naturally observed in failed traces; and there is no conformal / kept-accuracy layer.

**J. Hamel Husain — *Why is Error Analysis So Important in LLM Evals* (blog, hamel.dev), and the Langfuse / Medium "LLM evals failure-mode taxonomy" tutorials (2025).**
This is the **practitioner-side** version of T1.10: take a sample of failures, label them by hand or with an LLM-judge, induce a small task-specific taxonomy, iterate prompts/scores against it. **Standard practice in 2025 LLM-eval workflows.** The novelty surface for "do this for math reasoning" is essentially zero from an industry-practice perspective — the only novel layer is the *conformal kept-accuracy conditioning*.

### V1.3 What is *not* in the literature (the residual gap)

Combining the seven items above, the prior art has covered:

- A canonical 17-cat taxonomy of LLM errors broadly (ErrorAtlas).
- A 7-cat taxonomy of classification failures (FLARE).
- A 15-cat system-level production-failure taxonomy (Vinay 2511.19933).
- A 9-cat taxonomy of code-reasoning errors (2512.00215).
- A 6-cat MAS failure taxonomy (MAST).
- Role-of-step taxonomies for CoT (Kim 2603.21286).
- Step-level error-localization annotation (ProcessBench).
- Mistake-type-conditioned model evaluation (ACL 2025 long-1313).

**What is *unfilled*:**

> *Per-failure-mode kept-accuracy curves at controlled selective answer rate, broken out across multiple CoT-CP score families (lp_min, prm_min, sc_top1, etc.) on the same mode-labeled trace pool, computed against a conformal threshold rather than a free-form judge verdict.*

This is the precise and narrow gap T1.10 fills. It is not "build a new taxonomy of LLM failures" (that is filled by ErrorAtlas + four others); it is *"reuse one of the existing taxonomies as the failure-mode axis, run our score-family ladder against it, and report which scores catch which modes."*

### V1.4 Take-away on prior-art status

T1.10 was **mis-classified as "build the taxonomy."** Correct classification:
- **Taxonomy axis**: covered (use ErrorAtlas-17 or a math-restricted projection).
- **Per-mode breakdown methodology**: covered (FLARE, MAST, ACL 2025 long-1313).
- **Conformal × per-mode kept-accuracy**: **uncovered** — the wedge.

Combined coverage of T1.10's *original* claim surface: ~70%. Combined coverage of the *narrowed* claim surface (CP × per-mode kept-accuracy across score-family ladder): ~10–15%. The narrow framing is publishable as a §6 contribution; the broad "build the taxonomy" framing is not.

---

## V2 — Academic value

### V2.1 Publishability scenarios

| Vehicle | Realistic outcome |
|---|---|
| **(a) Standalone "math-CoT failure-mode taxonomy" paper** | **No.** ErrorAtlas (Jan 2026) and FLARE (2025) own this slot. A standalone paper would need to clear the bar that *math-CoT-specific failure modes are qualitatively different from ErrorAtlas's 17 generic modes*, which is not obviously true and not easy to defend in a review cycle. |
| **(b) Section of CoT-CP paper** | **Yes — and this is the natural home.** Slot it into §6 (Robustness / Diagnostics) as "*Per-failure-mode selective-accuracy breakdown*." Half a page of taxonomy projection (cite ErrorAtlas) + 1 figure (per-mode kept-acc bars across 6 score families) + 1 paragraph of qualitative diagnosis. |
| **(c) Workshop short paper** | **Maybe.** ICML/NeurIPS UQ / Math-AI / LLM-evals workshop, 4-page format, *only* if the per-mode breakdown reveals a sharp differential (e.g., "score family X catches 80% of arithmetic errors but 20% of premature-commitment errors") that plausibly motivates a hybrid score. Without that differential, the workshop submission is also weak. |
| **(d) Negative-result-as-contribution at ICBINB** | **Yes, conditional.** If the per-mode breakdown shows scores are *uniformly* good across modes, that itself is a useful diagnostic null result. |

### V2.2 Comparison to ErrorAtlas

ErrorAtlas was a full ICLR-submission-worthy paper (35 datasets, 83 LLMs, 7,000 instances, judge-accuracy validation, 17-mode hierarchy with sub-categories, and explicit canonical examples). Our T1.10 contribution would be **strictly smaller**: same machinery (LLM-judge classification of failures), one task family (math CoT), one trace pool (~50–100 from our existing experiments), no model breadth, no formal validation of judge accuracy, no hierarchy. Scope-wise, **roughly 1/30 to 1/50 of ErrorAtlas**.

The honest comparison: ErrorAtlas was a 2026 frontier paper that *closed* the "build the universal LLM error taxonomy" subfield. T1.10 is a downstream application of ErrorAtlas (use its taxonomy, condition on it, report a downstream measure).

### V2.3 Venue norms

- **TMLR**: As a §6 sub-section it fits without strain. As a standalone short paper, the bar for accepting yet another LLM-error-taxonomy paper post-ErrorAtlas is high.
- **NeurIPS / ICLR main**: Need real novelty per contribution. T1.10 cannot be *the* main novelty of a paper. As one of 5–6 contributions of CoT-CP, it passes if the per-mode differential is sharp.
- **ICBINB / ICML-Workshop**: Niche fit if the result is qualitatively informative regardless of sign.

**Conclusion**: T1.10 is a **section-level diagnostic contribution** to a framework paper, not a standalone publication.

---

## V3 — Feasibility & predicted performance

### V3.1 Effort

The proposed pipeline:

1. **Trace selection**: 50–100 failed traces from existing experiments. Stratify by:
   - false-keep (passed CP threshold but answer wrong): ~30–50 traces
   - low-quantile-rejected-but-actually-correct (wrongly preempted): ~20–30 traces
   - clean failure (preempted, answer wrong): ~20 traces
2. **Mode labeling**: project ErrorAtlas-17 onto math-CoT context. Probable retained 8–10 modes:
   - **Computation Error** (arithmetic / algebraic slip)
   - **Logical Reasoning Error** (invalid deduction)
   - **Specification Misinterpretation** (misread the problem)
   - **Premature Commitment** (locked into wrong path early; well-attested in our Pilot results)
   - **Missing Required Element** (forgot a constraint or case)
   - **Hallucinated Fact** (made up a value or theorem)
   - **Output Formatting Error** (correct reasoning, wrong final form — rare in math)
   - **Self-Verification Failure** (checked the wrong thing in their own work)
   - **(optional) Conceptual Error** (deeper than logical: wrong framing of the problem)
3. **LLM-judge labeling**: prompt Claude 4.7 (or GPT-4-class) with the trace + ErrorAtlas-projected definitions; require (mode, location, justification) triple. Expected ~40s per trace at 10–15K input tokens.
4. **Inter-rater audit**: dual-judge (Claude + GPT-4 / Gemini Pro) on a 20-trace stratified subset; report κ. Per CLAUDE.md cross-model verification protocol, this aligns with existing infrastructure.
5. **Per-mode kept-accuracy curve**: for each of 6 score families × 8 modes, compute kept-accuracy at the published α-grid {0.1, 0.2, 0.3, 0.5, 0.7}. Output is a 6×8×5 tensor → flatten to 6 small bar charts.

**Effort estimate**: 4–6 days end-to-end (1d trace selection + judge prompt design, 1–2d labeling pass + dual-judge audit, 1d aggregation + plotting, 1–2d writeup).

### V3.2 Compute

- Labeling: 100 traces × 2 judges × ~40s ≈ 2.2 GPU-hours equivalent (or ~$10–20 of API spend).
- Aggregation/plotting: trivial.
- **Total: < 5 GPU-hours**, well within available budget.

### V3.3 Predicted shape of the result

**Most likely outcome (P ≈ 0.55)**: per-mode kept-accuracy curves are **mildly differentiated but not sharp**. E.g.:
- `lp_min` catches 70% of computation errors, 35% of premature-commitment errors, 50% of logical-reasoning errors at α=0.5.
- `sc_top1` is roughly the inverse: 35% of computation errors but 65% of premature-commitment (because parallel sampling exposes the alternate-path branch).
- `prm_min` is best on logical-reasoning errors (60–70%) but middling everywhere else.

This shape is *useful* (motivates score-family hybridization, supports our LR+ Theorem 2 narrative) but not *dramatic*.

**Optimistic outcome (P ≈ 0.20)**: a single mode shows a sharp differential (e.g., one score catches 80% while others catch <20%). This is the workshop-paper-grade outcome.

**Pessimistic outcome (P ≈ 0.25)**: all score families have roughly uniform per-mode kept-accuracy (within ±10pp). This is the *negative-result-as-contribution* outcome — itself useful as a diagnostic null, supporting "score families are robust to failure-mode variation."

**Honest predicted headline**: "*CoT-CP score families show modestly differentiated per-mode coverage, with [score X] strongest on [mode A, B] and [score Y] strongest on [mode C, D], suggesting hybridization yields marginal gains.*" — landing in the "useful diagnostic, not a standalone contribution" band.

### V3.4 Risk: judge accuracy

ErrorAtlas's LLM-judge was validated at ~92% accuracy on a held-out sample. A 100-trace pass with two judges (κ ≥ 0.7 acceptable, ≥ 0.8 good) is feasible, but at our small N=100 the per-mode kept-accuracy estimates have wide CIs:
- Per-mode N at typical mode-prevalences (5–15% of traces): N_mode ≈ 5–15 per mode.
- 95% CI on a per-mode kept-acc estimate at N=10 is roughly ±25pp — too wide to declare a sharp differential unless the point estimate gap is > 30pp.

**Mitigation**: scale to 200–300 labeled traces (still tractable, ~1 extra labeling day) for the modes that look promising in the first-pass 100. This is a tiered strategy: pilot at N=100, expand only on the 2–3 modes with apparent differentials.

**Verdict on risk**: tractable but the small-N CI is the dominant risk; the result is interpretation-sensitive at small N.

---

## V4 — Incremental vs structural novelty

### V4.1 Case for "incremental"

- Building a math-specific 8–10 mode taxonomy is a re-projection of ErrorAtlas's 17.
- LLM-judge mode-labeling pipelines are standard practice (Hamel Husain blog, MAST methodology, Langfuse tutorials, FLARE).
- Per-mode breakdowns of model performance exist in the math-error-handling literature (ACL 2025 long-1313, various perturbation studies).
- The CP machinery is unchanged. No new theorem follows.

### V4.2 Case for "structurally different"

- **Conditional inferential object**: per-mode kept-accuracy at fixed answer rate is *not* a standard quantity in the LLM-error literature. ErrorAtlas reports prevalence, not selective-coverage. ProcessBench reports first-error-step location, not selective-accuracy. The closest published thing is FLARE's per-mode prevalence on classification failures, which is not a kept-accuracy at fixed-α metric.
- **Score-family conditioning**: nobody has run *multiple* CP score families against a *common* mode-labeled pool to compare per-mode coverage profiles. This is the actual novel cell of the matrix (mode × score-family × kept-acc-at-α).
- **Diagnostic feedback into score design**: if mode X is poorly covered by every score family, that motivates a *new* score family targeting X. This is a *constructive* use of failure-mode analysis that is rare in the literature — most failure-mode taxonomies are descriptive, not prescriptive.

### V4.3 Which wins?

**Incremental, with one defensible structural angle.** The *taxonomy-construction* part is incremental (re-projection of ErrorAtlas). The *score-family × mode coverage matrix* is genuinely new in shape, but small in absolute novelty — it is one figure + one paragraph of interpretation, not a theorem. Closest framing-honest pitch:

> "We project ErrorAtlas (Sundar et al. 2026) onto math-CoT to obtain an 8-mode taxonomy, and report per-mode selective accuracy across CoT-CP's six score families. We find [score-family X dominates on modes A, B] / [coverage is approximately uniform], motivating [hybrid score Z] / [no further hybridization]."

This is a §6 sub-section, not a standalone novelty claim.

---

## V5 — Hardest plausible NeurIPS reviewer objection

**The objection**:

> "The proposed per-failure-mode kept-accuracy diagnostic uses an LLM-judge to label 50–100 failed traces with a taxonomy that is essentially a math-projection of ErrorAtlas (Sundar et al. 2026), which itself was built from 35 datasets × 83 LLMs × 7,000 instances. The authors' N=100 is statistically underpowered: per-mode N is 5–15, and 95% CIs on per-mode kept-accuracy estimates are roughly ±25pp at this scale — too wide to support the headline claim 'CoT-CP at α=0.5 catches 73% of arithmetic-error traces.' Additionally, judge accuracy is unverified for math-reasoning labels (ErrorAtlas validated 92% on its broad mix; the specific math-CoT distribution may yield substantially different judge accuracy). Without (a) N≥300 or (b) human-expert audit on a substantial subsample, this is a qualitative observation dressed in quantitative clothing. **The diagnostic is interesting in spirit but the headline numbers do not survive statistical scrutiny.** I recommend the authors (i) increase N, (ii) report bootstrap CIs on each per-mode estimate, (iii) include a human-audit κ for at least one annotator pair, and (iv) drop sharp-differential claims that the CIs do not support."

**Best response**:

> "We accept the statistical-power point. We commit to (a) N=300 stratified-sample (3× the original budget; still tractable at <10 GPU-hours and 2 extra labeling days), (b) per-mode bootstrap 95% CIs on every kept-acc cell of the 6×8 matrix, (c) human-audit on 50 traces by the human research lead, with reported Cohen's κ vs the LLM-judge labels, and (d) framing the headline as the *direction* of the differential ('score X tends to dominate on mode A by Δ ≈ 30pp, 95% CI [10, 55]') rather than a point estimate. We do not pitch T1.10 as a contribution unless the differential survives the bootstrap CI."

**Honest assessment**: this defense is acceptable for **TMLR** as a §6 sub-section *with N=300 and human audit*. For **NeurIPS / ICLR main**, T1.10 lives or dies on whether the score-family × mode differential survives the CI — and the V3.3 prior says ~25% chance of pessimistic uniformity, ~55% chance of mild differentiation, ~20% chance of sharp differentiation. **At our most-likely-mild differentiation case, the bootstrap CI will straddle "marginal" for most cells** and the result will read as "qualitatively suggestive but not statistically forced."

---

## V6 — Final verdict

### V6.1 Decision

**KEEP — but only as §6 (Robustness / Diagnostics) sub-section of CoT-CP paper. Not as a standalone contribution. Use ErrorAtlas's taxonomy directly with explicit citation; do not build a competing 8-mode taxonomy from scratch.**

### V6.2 Reasoning

1. **Gap is real but narrow**: per-mode kept-accuracy across a *score-family ladder* is genuinely unfilled. ErrorAtlas + FLARE + MAST close the broader "taxonomy of LLM failures" gap, but none of them condition on a CP score family or report selective-accuracy.
2. **Lift is diagnostic, not headline**: the most-likely outcome is mild differentiation across modes, motivating score hybridization but not enabling a "we caught X% of all failures" headline. Pessimistic outcome (uniform coverage) is itself a useful null.
3. **Novelty is methodological-plus, not theoretical**: re-projection of ErrorAtlas + 6×8 conditional matrix + interpretation paragraph. No theorem.
4. **Cost is small**: 4–6 days, < 5 GPU-hours, $10–20 API.
5. **TMLR fit is good**: §6 sub-section fits without strain.
6. **NeurIPS/ICLR/main fit only as a sub-contribution**: standalone too thin; as one of 6 contributions of CoT-CP, passes if differential survives CI.

### V6.3 Action items for the paper

1. **Reuse ErrorAtlas**: project the 17-mode taxonomy onto math-CoT, retaining 8 modes (V3.1 list). Cite Sundar et al. 2026 prominently.
2. **Run N=300 stratified-sample labeling** (not N=100): 100 false-keeps + 100 false-aborts + 100 clean preempts. Use Claude-Opus-4-7 + GPT-4-class judge in parallel; report κ.
3. **Human audit on 50 traces** by the human research lead; report κ vs LLM-judge labels.
4. **Compute the 6×8 kept-accuracy matrix** at α ∈ {0.1, 0.2, 0.3, 0.5, 0.7}; bootstrap 95% CIs per cell.
5. **One figure**: clustered bar chart, 6 score families × 8 modes, at α=0.5.
6. **One paragraph in §6**: "*Per-failure-mode breakdown reveals score-family complementarity: lp_min dominates on Computation Error (ΔKA=22pp, 95% CI [8, 36]) while sc_top1 dominates on Premature Commitment (ΔKA=18pp, 95% CI [4, 31]). This complementarity motivates the hybrid LR+ ranking of Theorem 2.*" Honest framing only — no overclaim.
7. **Skip or downsize if N=300 results are uniform** within ±10pp across modes for every score family: that becomes a footnote, not a section.

### V6.4 Go/no-go gate

| Outcome on N=300 labeled traces | Decision |
|---|---|
| At least one score family has a mode-conditional Δ > 25pp with 95% CI excluding 0 | **§6 sub-section, full half-page, motivates LR+ hybrid framing.** |
| Δ in the 10–25pp range across multiple modes, CIs marginal | **§6 paragraph + 1 figure**, qualitative framing, no headline numbers. |
| Δ < 10pp across all modes (uniform coverage) | **Footnote in §6**: "Per-mode breakdown shows uniform coverage profiles across score families, suggesting the score ladder is mode-agnostic at the population level." |

### V6.5 Resource priority vs other picks

| Pick | Priority |
|---|---|
| T1.6 (Local CP SBERT+KNN) | ⭐⭐⭐ |
| T1.4 (Earliest-bad-step) | ⭐⭐ |
| T1.5 (Pólya Look Back pilot) | ⭐⭐ |
| T1.1 (e-process per-step) | ⭐⭐ |
| T1.2 (BH-FDR) | ⭐⭐ (1–2 weeks math + rerun) |
| **T1.10 (Per-mode kept-acc)** | **⭐ (4–6 days, < 5 GPU-hours, useful diagnostic)** |
| T1.3 (Diverse Beam Search) | ⭐ (defensive ablation) |
| T1.9 (Fermi cross-check) | drop / fold |

T1.10 sits at the **same priority tier as T1.3** — defensive / diagnostic, not headline-driving. Run it after T1.1, T1.4, T1.5, T1.6 are stable.

### V6.6 One-line bottom line

> **T1.10 is incremental on the taxonomy axis (ErrorAtlas, FLARE, MAST already own that), genuinely novel on the narrow "per-mode kept-accuracy across CP score families" cell, and worth a §6 sub-section in CoT-CP — but only with N=300 (not 100), bootstrap CIs, human-audit κ, and ErrorAtlas-projected taxonomy. Drop if uniform; promote to half-page only if at least one ΔKA > 25pp survives CI.**

---

## Cross-Model Verification Results

*Per CLAUDE.md `cross_model_verification: mode: all` and `pipeline/cross_model_verification_protocol.md`, this verdict (KEEP-as-§6-sub-section) is in scope for verifier re-check at the orchestrator level. The fallback verifier (`openai/openai/gpt-5.5` or `gcp/google/gemini-3.1-pro-preview`) was not invoked because the inference token in the active config is `sk-PLACEHOLDER`. Logging single-model fallback per the protocol.*

*Specifically pressure-test on re-run:*
1. *Whether N=100 vs N=300 is the right cutoff (verifier may argue for N=500 if it has a tighter CI prior).*
2. *Whether the ErrorAtlas-17 → 8-mode math-CoT projection is defensible without an empirical re-validation of judge accuracy on the math distribution (ErrorAtlas's 92% was validated on the broad mix, not the math sub-slice).*
3. *Whether the most-likely-outcome prediction (mild differentiation, ~55% probability) is consistent with the verifier's own prior; if verifier predicts > 70% probability of sharp differentiation, escalate the priority.*
4. *Whether the human-audit κ requirement (V6.3 item 3) is overkill for a §6 sub-section or appropriate.*

*This section is appended without silent override per CLAUDE.md "no silent overrides" rule.*

---

## Sources

Direct prior art:
- Sundar et al., *ErrorMap and ErrorAtlas: Charting the Failure Landscape of Large Language Models*, arXiv:2601.15812 (Jan 2026). [https://arxiv.org/abs/2601.15812]
- Patel et al., *FLARE: An Error Analysis Framework for Diagnosing LLM Classification Failures*, ACL-OMMM 2025. [https://aclanthology.org/2025.ommm-1.4/]
- Vinay, *Failure Modes in LLM Systems: A System-Level Taxonomy for Reliable AI Applications*, arXiv:2511.19933 (Nov 2025). [https://arxiv.org/abs/2511.19933]
- Zheng et al., *ProcessBench: Identifying Process Errors in Mathematical Reasoning*, arXiv:2412.06559 (ICLR 2025). [https://arxiv.org/abs/2412.06559]
- Cemri et al., *Why Do Multi-Agent LLM Systems Fail? (MAST)*, arXiv:2503.13657 (2025). [https://arxiv.org/abs/2503.13657]
- *When the Chain Breaks: Interactive Diagnosis of LLM Chain-of-Thought Reasoning Errors*, arXiv:2603.21286 (Mar 2026). [https://arxiv.org/abs/2603.21286]
- *Demystifying Errors in LLM Reasoning Traces: An Empirical Study of Code Execution Simulation*, arXiv:2512.00215 (Dec 2025). [https://arxiv.org/abs/2512.00215]
- Winston et al., *A Taxonomy of Failures in Tool-Augmented LLMs (TALLM)*, AST 2025. [https://homes.cs.washington.edu/~rjust/publ/tallm_testing_ast_2025.pdf]
- *Evaluating LLMs' Ability to Handle Mistakes in Mathematical Reasoning*, ACL 2025 long. [https://aclanthology.org/2025.acl-long.1313]

Practitioner background:
- Hamel Husain, *Why is Error Analysis So Important in LLM Evals*, hamel.dev (2025). [https://hamel.dev/blog/posts/evals-faq/why-is-error-analysis-so-important-in-llm-evals-and-how-is-it-performed.html]
- Langfuse, *Error Analysis to Evaluate LLM Applications* (2025). [https://langfuse.com/blog/2025-08-29-error-analysis-to-evaluate-llm-applications]

Internal CoT-CP context:
- `/home/nvidia/future/METHOD_AND_RESULTS.md`
- `/home/nvidia/future/literature/papers/ANALYSIS.md`
- `/home/nvidia/future/literature/verification/T1_1_verification.md` through `T1_9_verification.md`

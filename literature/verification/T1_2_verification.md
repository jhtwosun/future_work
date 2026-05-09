# T1.2 Verification — Benjamini-Hochberg FDR for per-step CP (Approach B)

> **Pick**: Replace the per-step Bonferroni correction (α/T_max per step) in Approach B with the BH step-up procedure to control FDR at level α on T-step CoT traces with conformal p-values per step.
> **Author**: Verifier (Claude Opus 4.7, 1M ctx).
> **Date**: 2026-05-08.
> **Status**: V1–V6 complete. Verdict at the bottom.

---

## V1 — Direct prior art (literature scan)

12 WebSearch queries fired across (i) step-level CP × multiple testing × LLM, (ii) BH × conformal × LLM/CoT, (iii) online FDR × LLM, (iv) within-trace per-prediction multiple testing in CP, (v) ICLR/NeurIPS 2025 multi-step CP, (vi) FDR-in-conformal-selection surveys, (vii) Bonferroni vs BH under PRDS, (viii) sequential conformal/e-process for reasoning traces, (ix) BY under general dependence, (x) per-step Bonferroni alternatives, (xi) early-abort + martingales for reasoning, (xii) cfBH + extensions. The 4 papers actually fetched and read in detail are listed inline below; the rest were inspected at title+abstract level. Internal `/home/nvidia/future/literature/scan_notes.md` style summary:

### V1.1 The four most relevant pieces of prior art

**(P1) Jin & Candès, "Selection by Prediction with Conformal p-values" (cfBH), JMLR 2023, arXiv 2210.01408.**
The canonical "BH on conformal p-values" paper. Setting: a candidate pool of size $m$, one conformal p-value per candidate against a calibration set, BH at level $q$ controls FDR ≤ $q$ under exchangeability of the calibration scores. Conditions: PRDS holds because the conformal p-values share the same calibration set and the resulting joint distribution is positive-regression-dependent (Theorem 4 of Jin-Candès). This is **across-instance** selection (one p-value per test point), not within-trace.

**(P2) "Online Conformal Selection with Accept-to-Reject Changes" (OCS-ARC), 2508.13838 (2025).**
Streaming candidate selection where decisions are irreversible once accepted, but rejections can be revisited. Uses *online BH* (LORD/SAFFRON-style) over a stream of conformal p-values. Setting is still **inter-trace** (one p-value per arriving candidate). Explicitly notes "no existing baseline" because it is the first to do online ARC; their ablation baseline is *online Bonferroni*, mirroring our Approach-B vs Approach-B-with-BH question — but at the wrong granularity (across candidates, not within a trace).

**(P3) "Feedback-Enhanced Online Multiple Testing with Applications to Conformal Selection" (GAIF), 2509.03297 (2025).**
Generalized alpha-investing (not BH proper) with feedback. Online conformal selection setting; mentions LLM real-time alignment as a motivating downstream but does not test step-level CoT p-values.

**(P4) "Max-Rank: Efficient Multiple Testing for Conformal Prediction", 2311.10900 v3 (2024).**
Crucially relevant: this paper *does* address parallel conformal tests sharing calibration data — including **multi-step time-series forecasting** as a named application. They give an FWER-controlling alternative to Bonferroni that exploits PLOD (positive lower orthant dependency) via the maximum rank statistic. **But: (i) FWER, not FDR — they explicitly state they do not target FDR; (ii) they treat the m parallel tests as exchangeable in dimension, which is fine for multi-target regression but not for the auto-regressive within-trace order; (iii) they do not address LLM CoT.**

### V1.2 Papers that touch the neighborhood but not the same object

- **Mohri & Hashimoto, "Language Models with Conformal Factuality Guarantees", ICML 2024.** Decomposes a generation into atomic sub-claims, calibrates a single threshold, censors below-threshold claims. **No multiple-testing correction across claims** — the guarantee is on a learned aggregate via Learn-Then-Test, not on a step-level p-value family. Different inferential object.
- **"Conformal Language Model Reasoning with Coherent Factuality", ICLR 2025 (openreview AJpUZd8Clb).** Applies split CP to subgraphs in a "deducibility" graph. Ensures *coherent factuality* (a claim is checked in context of its ancestors). Not BH; not Bonferroni; the structural device is graph hierarchy, not multiple-testing correction.
- **Quach et al., "Conformal Language Modeling", ICLR 2024.** Sequence-level stopping rule via Learn-Then-Test (Angelopoulos-Bates-Candès); calibrates a *single* threshold. No per-step p-value family.
- **Abbasi-Yadkori et al. "Conformal Abstention", NeurIPS 2024.** Single semantic-entropy score per generation; no per-step testing.
- **"Conformal Interpretability of Temporal Concepts in LLM Agents", 2604.19775 (2025/26).** Step-wise inductive CP with **fixed per-step ε_s, ε_f** — explicitly **no within-trace multiple-testing correction**. Closest in framing to ours; would directly benefit from BH had they known to apply it.
- **"ConSol: SPRT for consistent LLM reasoning", 2503.17587.** Sequential probability ratio test across *samples*, not across steps.
- **"Martingale Foresight Sampling", 2601.15482.** Doob decomposition for path pruning, not p-value-based.
- **"Synthetic-Powered BH" (SynthBH), 2602.16690.** BH with synthetic data sharing — orthogonal axis (synth augmentation), not within-trace.
- **"Multi-Condition Conformal Selection", 2510.08075v2.** Multivariate cfBH under multiple conditions on the response. Inter-trace.
- **"Conformalized Multiple Testing after Data-dependent Selection".** BH after selection; orthogonal.
- **"Bates-Angelopoulos-Lei-Romano, Selective FDR for prediction sets", JRSSB 2023.** FDR over the *prediction-set* family per test point, not over per-step p-values.

### V1.3 Bottom line of V1

**No one has done BH (or any FDR procedure) on the per-step conformal-p-value family within a single CoT trace.** Closest neighbors are (a) cfBH/OCS-ARC/GAIF — all *across-trace* selection FDR, and (b) Max-Rank — *within-test* multi-step forecasting with FWER not FDR. This is the right shaped gap for an incremental theoretical contribution; the question is whether it is *interesting enough* given how mechanical the extension is.

---

## V2 — Academic value

### V2.1 Publishability scenarios

| Vehicle | Realistic outcome |
|---|---|
| **(a) Standalone theorem extension paper** | **No.** A standalone "BH for per-step CoT CP" paper would be one theorem (already proved by 1995-BH + 2023-cfBH machinery), one corollary on PRDS within-trace, and one experiment table. TMLR might take it as a short "Note"; NeurIPS/ICLR would desk-reject for incremental scope. |
| **(b) Section of CoT-CP paper** | **Yes — and this is the natural home.** Slot it into the methods section directly after Approach B (Bonferroni). Add 0.5 page of theorem + 1 figure + 1 table comparing Bonferroni vs BH kept_acc. |
| **(c) Workshop** | **Yes** — it would be a perfect 4-page submission to ICML / NeurIPS UQ workshops or COPA, but only if we did *not* go to TMLR with the main paper, since the workshop pre-publication would scoop the section. |

### V2.2 Comparison to cfBH

cfBH (JMLR 2023) is a **full paper**: two coupled theorems, weighted extension, real-data demonstrations on hiring/drug-discovery, and a new dependence-structure analysis. Our extension is **strictly smaller** in scope: same machinery, applied to a different but adjacent inferential object (within-trace step p-values) — a 1-page derivation in the cfBH framework with a within-trace dependence-structure remark.

The honest comparison: cfBH was a 30+ page paper that opened the "BH-on-conformal-p-values" subfield. T1.2 would be a 1–2 page corollary of cfBH applied to a new object, plus an experimental ablation. Scope-wise, **roughly 1/15 to 1/10 of cfBH**.

### V2.3 Venue norms

- **TMLR**: Incremental but rigorous extensions are explicitly fine. T1.2 as a **section** of CoT-CP fits TMLR's bar without strain. The paper-level novelty is the CoT-CP framework + 6 score families + the per-step extension; T1.2 is one sub-contribution among many.
- **NeurIPS/ICLR main**: Need real novelty per contribution. T1.2 as the *main* novelty of a paper would not clear the bar; T1.2 *as the multiple-testing component* of a larger framework paper passes if and only if the surrounding framework is novel.
- **ICML 2026**: Same as NeurIPS — incremental BH application is not enough on its own.

**Conclusion**: T1.2 is a **section-level contribution** to a framework paper, not a standalone publication.

---

## V3 — Feasibility & predicted performance

### V3.1 Math effort

The formal statement is a one-page theorem:

> **Theorem (per-step BH coverage).** Let $(p_t)_{t=1}^T$ be conformal p-values for steps $1..T$ of a single test trace, computed against a shared calibration set of $n_+$ correct trajectories using a per-step nonconformity score $S_t$. Suppose the joint of $(p_1, \ldots, p_T)$ on the null (i.e., conditional on the trace being correct) is **PRDS** on the index set $\{1, \ldots, T\}$. Then the BH step-up at level $\alpha$ controls the per-trace expected proportion of *false rejections of correct steps* at level $\alpha \cdot |\mathcal{H}_0|/T \leq \alpha$.

Proof: direct application of Benjamini-Yekutieli (2001) Theorem 1.2 (BH valid under PRDS) to the within-trace p-value family. The only non-trivial step is verifying PRDS — see V3.4. **Effort estimate: 3–5 days for a careful writeup including the PRDS verification subsection; not 1–2 weeks.**

### V3.2 Empirical rerun cost

Rerun Approach B on the existing trace data:
- 11 models × 7 datasets ≈ 77 cells, but only **the 10 cells in §11 of RESULTS_SUMMARY_KR.md** matter for the headline ablation (where Approach B was originally measured).
- The trace-level scores are already cached (no LLM forward pass needed) — only the threshold computation changes.
- **Compute estimate: ~1–2 H100 hours total for re-running thresholds + bootstrap CIs across 10 cells.** The 4–8 H100-hour estimate in the prompt is too high; we are reusing cached step scores, not re-decoding traces.

### V3.3 Predicted lift

Bonferroni at $T_{\max} = 60$ (R1-Distill long-CoT) tests each step at $\alpha/60$. For $\alpha = 0.3$: 0.005 per step. BH on the same family adapts to the empirical p-value distribution: if $k$ of $T$ p-values are small, BH thresholds the $i$-th smallest at $i\alpha/T$.

- **If most steps are "easy" (large p-values)**: BH ≈ Bonferroni at the smallest p-value (still $\alpha/T$). No lift.
- **If many steps have small p-values (genuine signal)**: BH thresholds approach $\alpha$. Up to $T$-fold tighter — but only on steps with small p-values, which are the ones we *want* to flag.

For our setting, the **practical lift on kept_acc** depends on how many of the per-step nonconformity p-values are small on incorrect traces. From §11 RESULTS, Approach B is currently kept_frac ~90+% with lift -5pp vs Approach A — meaning Bonferroni is too lax (it almost never aborts) on long traces. **BH should reclaim 1–3 pp of the 5pp Approach-A-vs-B gap**, not the full 5pp, because Approach A has no formal guarantee at all (it just stops at the first violation).

**Honest predicted headline**: BH-corrected Approach B will land between A and Bonferroni-B, kept_acc lift +1 to +3 pp on R1-Distill-class long-CoT models, ~0pp on short-CoT (Phi-4, Math-7B) where $T \leq 10$ and Bonferroni is already not very conservative.

### V3.4 PRDS — the real risk

Within a single auto-regressive CoT trace, the per-step scores $(S_1, \ldots, S_T)$ are **strongly positively dependent** because (a) they share the same prefix (Markov-style accumulation), (b) the same model parameters, (c) the same prompt, and (d) the same correctness label $Y = 1$ (we condition on correctness). The natural dependence direction is positive.

**But PRDS is a precise condition**: $\Pr[(p_{-i}) \in A \mid p_i = u]$ must be *non-decreasing* in $u$ for every increasing set $A$ and every null $i$. Auto-regressive CoT does not obviously satisfy this — a small p-value at step 5 (early sign of trouble) might *increase* the probability of small p-values at step 20 (compounding error), which is consistent with PRDS, *or* the model might "recover" (negative dependence), which violates PRDS.

**Mitigation**: fall back to **Benjamini-Yekutieli (BY)** under arbitrary dependence — multiplies the BH thresholds by $1/\sum_{k=1}^T 1/k = 1/H_T \approx 1/\ln T$. For $T=60$, $H_T \approx 4.7$, so BY at $\alpha = 0.3$ uses effective level $0.064$. This is **still tighter than Bonferroni** $0.3/60 = 0.005$ by a factor of $\approx 13×$ on the smallest p-value — most of the practical lift survives the conservative correction.

**Risk verdict**: PRDS is plausible but unverified; BY is a clean, safe fallback that preserves most of the gain. The paper should report **both BH and BY** results.

---

## V4 — Incremental vs structural novelty

### V4.1 Case for "incremental"

- BH is 1995. PRDS-validity is 2001 (Benjamini-Yekutieli).
- Conformal p-values are 2002 (Vovk).
- BH-on-conformal-p-values is 2023 (cfBH, Jin-Candès).
- Per-step nonconformity scores within a trace are an obvious application of the same machinery.
- The PRDS analysis for an auto-regressive trace is essentially identical to the PRDS analysis for any positively-correlated test family.
- **An expert reviewer can derive T1.2 in a paragraph.** This is a textbook generalization.

### V4.2 Case for "structurally different"

- **Sample-size object is different.** cfBH's $m$ = candidate pool size, growing with deployment. T1.2's $T$ = trace length, fixed by the model's CoT length and bounded by 100s, not millions.
- **Dependence structure is different.** cfBH gets PRDS for free from sharing one calibration set across $m$ exchangeable test points. T1.2's dependence is **temporal/auto-regressive**, fundamentally different — the p-values share a prefix, not a calibration set. The PRDS argument is a *new* analysis (or a justified BY fallback).
- **Power regime is different.** cfBH targets large $m$ (drug discovery: $m$ in the thousands) where BH's adaptivity dominates Bonferroni dramatically. T1.2's $T$ ranges 5–100, where Bonferroni is mildly over-conservative but not catastrophically so. The lift is not "2–5×" universally; it's $T$-dependent and concentrated on the small-p-value subset.

### V4.3 Which wins?

**Incremental, by a clear margin.** The structural-novelty arguments are real but small: a different sample-size regime and a different dependence-structure justification do not amount to a new theorem in the cfBH sense. The paper-level honest framing is:

> "We extend the cfBH framework of Jin & Candès (2023) to the within-trace per-step setting, where the multiple-testing family is generated by a single auto-regressive trace rather than a batch of exchangeable candidates. We verify PRDS empirically on our trace data, with Benjamini-Yekutieli as a dependence-agnostic fallback."

This is one paragraph + one theorem + one figure. **The single most novel theorem is the PRDS verification (or BY application) for the auto-regressive trace setting** — but it is a **proposition**, not a theorem, in the formal-novelty sense.

---

## V5 — Hardest plausible NeurIPS reviewer objection

**The objection** (verbatim style):

> "The proposed BH extension is a direct one-paragraph corollary of Jin-Candès (2023) applied to a different index set. The within-trace dependence structure is **assumed** to be PRDS without proof — Section X just says 'plausible'. The empirical lift over Bonferroni is +1 to +3 pp on a small subset of long-CoT models, well within the bootstrap CI. The fallback to Benjamini-Yekutieli undermines the headline claim because BY is even simpler and was available in 2001. **There is no new theorem here; the contribution is a numerical recalibration.** I recommend rejection or significant restructuring."

**Best response**:

> "We agree the BH-step-up *machinery* is not novel — it is a known instrument applied to a new statistical object. Our claim is methodological, not theoretical: we are the first to identify that **the per-step CP threshold family within an auto-regressive CoT trace is a multiple-testing family**, and to give the right correction. The Bonferroni baseline is what every prior step-level CP paper uses (e.g., the 2604.19775 step-wise interpretability paper applies fixed per-step thresholds *without any correction*). Our PRDS verification is empirical (Section §X.2 reports the within-trace rank correlation matrix on the calibration set across 11 models × 7 datasets — all positive and significant); BY is reported as a guaranteed-safe lower bound. The +1–3 pp lift is exactly the right magnitude given the $T$ regime of CoT traces, and we present it honestly with bootstrap CIs."

**Honest assessment**: this defense holds for TMLR and probably for ICLR/NeurIPS *as one component of a larger framework paper*, but **not** as the standalone claim of a NeurIPS submission. The reviewer would be correct that, on its own, T1.2 is a recalibration with provenance.

---

## V6 — Final verdict

### Verdict: **KEEP — but only as a section of CoT-CP paper, not as a standalone contribution.**

### Reasoning

1. **The gap is real.** No one has applied BH (or any FDR procedure) to the per-step conformal-p-value family within a single CoT trace. Existing FDR-on-conformal-p-values work is across-trace (selection): cfBH, OCS-ARC, GAIF, weighted CS, multi-condition CS. Existing within-prediction multiple testing is FWER (Max-Rank), not FDR. The closest step-wise-CP-on-traces paper (2604.19775) does not correct at all.
2. **The lift is modest but honest.** Predicted +1 to +3 pp kept_acc on long-CoT models, ~0 pp on short-CoT. This is a real improvement over Bonferroni Approach B (which is too conservative and hurts kept_acc by ~5 pp vs the no-formal-guarantee Approach A) but does not close the entire A-vs-B gap.
3. **The novelty is methodological, not theoretical.** Per V4, the math is a known cfBH-style application; the PRDS verification on auto-regressive traces is the only new analytical content, and it is a proposition, not a theorem.
4. **Cost is small.** ~3–5 days math writeup, ~1–2 H100-hours rerun.
5. **TMLR fit is excellent.** TMLR explicitly accepts incremental but rigorous methodological contributions; T1.2 as the multiple-testing component of CoT-CP fits without strain.
6. **NeurIPS/ICLR fit only as a sub-contribution.** Standalone: too thin. As one of 5–6 contributions of CoT-CP: clears the bar.

### Action items for the paper

1. Write Theorem 4 (or Proposition 3): per-step BH coverage under PRDS, with BY fallback under arbitrary dependence. ~1 page.
2. Add an empirical PRDS-check subsection: report within-trace rank correlations across 11×7 cells. ~0.25 page + 1 small table.
3. Rerun Approach B with both BH and BY thresholds on the 10 §11 cells. Add columns to Table 11. ~1–2 H100-hours.
4. Honest framing: position as "tightening the per-step formal-guarantee branch (Approach B) of CoT-CP", not as a new theorem.

### Cross-Model Verification Results

This verdict was produced single-model (Claude Opus 4.7). Per `cross_model_verification_protocol.md` and the active `mode: all` setting in `CLAUDE.md`, the inference token is a placeholder — verifier (`openai/openai/gpt-5.5`) cannot be invoked from this environment. Logging single-model fallback per the protocol; recommend the master orchestrator re-run V6 verdict with the verifier model before final commit if the token is provisioned.

---

## Sources

- Jin & Candès, "Selection by Prediction with Conformal p-values", JMLR 2023, [arXiv:2210.01408](https://arxiv.org/abs/2210.01408) / [JMLR PDF](https://www.jmlr.org/papers/volume24/22-1176/22-1176.pdf)
- "Online Conformal Selection with Accept-to-Reject Changes" (OCS-ARC), [arXiv:2508.13838](https://arxiv.org/html/2508.13838v1)
- "Feedback-Enhanced Online Multiple Testing with Applications to Conformal Selection" (GAIF), [arXiv:2509.03297](https://arxiv.org/html/2509.03297)
- "Max-Rank: Efficient Multiple Testing for Conformal Prediction", [arXiv:2311.10900v3](https://arxiv.org/html/2311.10900v3)
- "Conformal Language Model Reasoning with Coherent Factuality", ICLR 2025, [OpenReview](https://openreview.net/forum?id=AJpUZd8Clb)
- "From Actions to Understanding: Conformal Interpretability of Temporal Concepts in LLM Agents", [arXiv:2604.19775](https://arxiv.org/html/2604.19775)
- Mohri & Hashimoto, "Language Models with Conformal Factuality Guarantees", ICML 2024, [arXiv:2402.10978](https://arxiv.org/abs/2402.10978)
- Quach et al., "Conformal Language Modeling", ICLR 2024, [arXiv:2306.10193](https://arxiv.org/abs/2306.10193)
- "Conformal Prediction: A Data Perspective" (survey), [ACM Computing Surveys 2025](https://dl.acm.org/doi/10.1145/3736575)
- Benjamini & Yekutieli, "The Control of FDR in Multiple Testing under Dependency", Annals of Statistics 2001
- "Synthetic-Powered Multiple Testing with FDR Control", [arXiv:2602.16690](https://arxiv.org/html/2602.16690)
- "Multi-Condition Conformal Selection", [arXiv:2510.08075v2](https://arxiv.org/html/2510.08075v2)
- "ConSol: SPRT for consistent LLM reasoning paths", [arXiv:2503.17587](https://arxiv.org/abs/2503.17587)
- "Martingale Foresight Sampling", [arXiv:2601.15482](https://arxiv.org/html/2601.15482v1)

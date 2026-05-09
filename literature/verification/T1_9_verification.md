# T1.9 — Two-pass Fermi / heuristic-rigorous cross-check

> **Verifier**: general-purpose research agent (a875e03d34ec21f4e)
> **Date**: 2026-05-08
> **Status**: V1–V6 complete. Final verdict: **REJECT as standalone paper / FOLD as one optional score-family rung in CoT-CP §4.x with explicit prior-art citation.**
> **One-line summary**: The "estimate first, solve rigorously second, reject if outside Fermi bounds" pattern is **not** unfilled — Piehl et al. (2025, arXiv:2509.18565, EVoSS) implemented the symmetric variant on GSM8K/SVAMP/Algebra, and Epstein et al. (2025, arXiv:2510.26995, FermiEval) already pair Fermi estimation with conformal prediction. T1.9 reduces to (a) a directional flip of EVoSS (Fermi → rigorous, not rigorous → Fermi) and (b) a port from word-problem datasets to AIME/MATH-500 — incremental at best.

---

## V1 — Prior art deep search

### V1.1 Search queries executed (10)

1. `"Fermi estimation" LLM reasoning benchmark order of magnitude`
2. `"order of magnitude" verification chain of thought LLM`
3. `"two-pass" sanity check LLM reasoning self-verification`
4. `LLM estimation uncertainty bounds arithmetic sanity check`
5. `"order of magnitude" reasoning prompt LLM math AIME estimation check`
6. `"common-sense reasoning" magnitude verification LLM numerical answer`
7. `LLM "back of the envelope" estimation check arithmetic verification`
8. `"plausibility check" LLM reasoning answer numerical reject filter`
9. `"LLM for Complex Reasoning Task: An Exploratory Study in Fermi Problems" Kalyan REALFP`
10. `"estimation verification" math word problem LLM rectification iterative`

### V1.2 Direct prior art (5 items, in order of threat to T1.9)

**A. Piehl, Wilson, Kalita & Kalita (2025) — *Solving Math Word Problems Using Estimation Verification and Equation Generation* (arXiv:2509.18565).**
This is the closest direct precedent and **directly invalidates T1.9's "largely unfilled" status**.

- Method (EVoSS): (1) LLM decomposes the word problem and generates equations; (2) symbolic solver computes exact answer; (3) LLM is then asked to *estimate* the answer; (4) accept iff estimate is within α% (α ∈ {40, 50}) of the symbolic answer; (5) on rejection, re-prompt with the estimate as a hint and rectify.
- This is precisely T1.9's "two-pass cross-check" with the *order swapped* (rigorous → estimate vs. estimate → rigorous). The decision rule (accept iff within Fermi tolerance) is identical.
- Headlines: 88.7 % avg across {Algebra (92.8), SVAMP (89.4), SVAMPClean (90.5), GSM8K (82.2)}.
- Authors explicitly cite math-pedagogy advice on "checking answers against estimates" — exactly the Tao/Weinstein–Adam framing T1.9 was built on.

**B. Epstein, Winnicki, Sornwanee & Dwaraknath (2025) — *LLMs are Overconfident: Evaluating Confidence Interval Calibration with FermiEval* (arXiv:2510.26995, October 2025).**
This compresses the *novelty surface area* T1.9 could claim on the calibration side.

- Builds FermiEval, a benchmark of Fermi-style estimation Qs with a Winkler-score interval rule.
- Finds nominal-99 % LLM intervals cover only ~65 %.
- **Applies conformal prediction post-hoc** to recover true 99 % coverage; reports −54 % Winkler interval score.
- Implication: pairing "Fermi estimation" with "CP-style coverage" is already published. CoT-CP cannot claim novelty on the joint primitive.

**C. Kalyan et al. (2021) — *REALFP / fp_score* (the dataset).** 185/185/558 Fermi problems with order-of-magnitude scoring (fp_score = 1 iff answer within same OoM as gold). Long-running benchmark; sets the prior that *the Fermi pass itself has ~10× error bars*, which is the central failure mode for T1.9 (see V3).

**D. *LLM for Complex Reasoning Task: An Exploratory Study in Fermi Problems* (arXiv:2504.02671, 2025).** Evaluates GPT-4 / Claude / Gemini / Mixtral on REALFP using TELeR-taxonomy prompts. Best (GPT-4) achieves error factor ~10 across problems; all 4 score < 0.5 fp_score. **Confirms LLM Fermi estimates are themselves often off by 10×** — directly the V3 risk.

**E. Self-verification family** (Weng et al. 2022 *Self-Verification*, Dhuliawala et al. 2023 *CoVe*, Quan et al. 2025 *Asking LLMs to Verify First*). Uses backward verification of conditions / planned check-questions / pre-emptive verification; not magnitude-based but the "second pass that rejects the first" structural pattern is identical.

### V1.3 Closest 3–5 reads

The three most-relevant items above (A, B, D) collectively cover:
- the symmetric two-pass cross-check on math word problems (A),
- the Fermi + CP coverage primitive (B),
- the empirical fact that LLM Fermi estimates have ~10× spread (D).

Adjacent prior art that *isn't* a direct hit but is close: MathPrompter (Imani et al. 2023), Self-Verification (Weng 2022), CoVe (Dhuliawala 2023), Step-Wise Formal Verification (Aggarwal 2025, arXiv:2505.20869) — they verify, but on logic / equations / facts, not magnitude. They do *not* invalidate T1.9 directly but they fill the "verification primitive" slot in the literature, leaving very little room for a fresh paper.

### V1.4 Verdict on prior-art status

T1.9 was **mis-classified as "largely unfilled."** The correct classification is:
- **Symmetric idea published**: EVoSS (Piehl 2025) — same accept/reject rule, opposite ordering.
- **Adjacent calibration idea published**: FermiEval + CP (Epstein 2025).
- **Empirical risk confirmed published**: REALFP-based studies show 10×-spread Fermi pass.

Combined coverage: ~80% of T1.9's intended claim surface.

---

## V2 — Academic value

### V2.1 Is it paper-worthy as a primitive?

**No, not standalone.** Three reasons:

1. **EVoSS already exists** with the symmetric construction; arguing "Fermi-first vs. solver-first changes things qualitatively" is not defensible without an empirical wedge — and AIME/MATH-500 don't naturally have a "symbolic solver" half, so the EVoSS pipeline doesn't transplant directly. That's a positioning advantage but a *narrow* one.
2. **FermiEval + CP already exists** with the formal calibration framing; this absorbs the "interval-style guarantee on Fermi pass" angle.
3. **The primitive is a 1–2-page workshop note**, not a TMLR paper. Acceptance bar at NeurIPS workshop / ICLR Tiny Papers / ARR short-paper track is realistic; main-conference is not.

### V2.2 Where it *could* fit

- **CoT-CP §4.x (score-family ladder addition)**: introduce a `fermi_consistency` binary score $s_{\text{fermi}} \in \{0, 1\}$ = "rigorous answer falls within stated Fermi bounds" as one extra rung alongside `lp_min` / `prm_min` / `sc_top1`. Theorem 1 (split CP on measurable aggregator) absorbs it without modification — that's the cleanest integration story.
- **NeurIPS 2026 Math-AI workshop** as a 4-page short paper: "Fermi Cross-Check as a Complementary Confidence Score for LLM Reasoning." Title sells the *complementarity* (information-theoretically near-orthogonal to logprob/PRM/SC), not the existence of the idea.
- **TMLR section in CoT-CP main paper**: a half-column ablation with explicit "EVoSS-style verification, ported from word problems to AIME" framing. Honest framing, no overclaim.

### V2.3 Honest assessment

Paper-worthy as a *score-family addition* under a CoT-CP umbrella with full prior-art credit. **Not** paper-worthy as a freestanding contribution at TMLR/NeurIPS-main level given EVoSS + FermiEval already cover the symmetric and calibration sides respectively.

---

## V3 — Feasibility & performance

### V3.1 Compute budget

- Reuses existing E2 (AIME 1983-2024, n=933) and E6v3 (MATH-500, n=500) setups.
- Per problem: 1 extra Fermi-pass call (~256 tokens) + 1 cross-check decision (deterministic). For Qwen3-8B on H100s, each pass ≈ 2 s. Total: ~25 minutes for AIME + ~14 minutes for MATH-500.
- **Estimated total: 3–5 GPU-hours** including N=8 self-consistency variants on the Fermi pass to robustify it (which is necessary, see V3.3).
- Calendar: 3–5 days end-to-end (prompt design 1d, run + ablate 1d, integrate into CoT-CP score family 1–2d, write up 1d).

### V3.2 Predicted lift (from this verifier's prior)

- **AIME**: +1–3 pp on the rigorous-pass accuracy by *rejecting* gross arithmetic errors. Binary `fermi_consistency` score added to CP+SC ladder: probably +0.5–2 pp at fixed answer rate (the CP ladder is already saturated by `sc_top1` at α=0.30 → 91.7 % kept-acc).
- **MATH-500**: +0.3–1 pp. Rationale: MATH-500 answers are mostly small integers / surds / fractions — the off-by-10× error mode is rare. Less headroom.
- **OlympiadBench**: probably +1–2 pp; integer + magnitude-ish answers exist but most are exact-value problems.

### V3.3 Risk: the Fermi pass is itself 10× off

This is the **dominant risk**.

- Empirical: on REALFP, *frontier* models (GPT-4) have error-factor ≈ 10 on Fermi tasks (Source D). Open 7B–32B models will be worse.
- Compounding: 10× error on the Fermi side × 10× error on the rigorous side = **100× false-reject window** and a sizeable false-accept window.
- This means stating "approximately 5000, give or take 10×" is inherently unreliable — the *bound* itself has 10× spread, so the cross-check doesn't have the leverage the proposal assumes.

**Mitigations** (each costs compute):
- **Self-consistency on the Fermi pass**: N=8 samples → take median + 2σ-spread for the bound. Costs 8× the Fermi pass; brings effective error down to ~3-5×.
- **Conformal-calibrate the Fermi bound**: use FermiEval's CP recipe (Epstein 2025) on a held-out AIME-validation subset to widen the "give-or-take" multiplier until empirical coverage hits 1−α. Costs O(100) calibration problems but turns the heuristic into a finite-sample-valid bound. **This is the only path that survives reviewer pressure.**

### V3.4 False-reject curve

Concrete: if Fermi pass has 10× spread (which we know empirically) and the rigorous pass is 81 % accurate on MATH-500, then a naïve "reject if outside Fermi bound" filter will:
- **False-reject rate** (correct rigorous answer outside Fermi bound) ≈ 5–15 % depending on Fermi quality. This *removes* correct answers, *lowering* kept-accuracy at fixed answer-rate unless tightly calibrated.
- **True-reject rate** (off-by-10× rigorous error caught) ≈ 60–80 % of the gross errors that exist — but gross errors exist on only ~3–5 % of problems.
- Net lift on selective accuracy is bounded above by ~1.5 pp on MATH-500, ~3 pp on AIME.

This is consistent with the V3.2 prediction. **The prediction is real but small.**

---

## V4 — Incremental vs structural

### V4.1 Incremental framing (most honest)

- T1.9 = a prompt-engineering trick (extra Fermi pass) + a heuristic threshold.
- It is not a new theorem, not a new dataset, not a new evaluation methodology, not a new score *family* — it's a binary score *member* (`fermi_consistency`) that lives inside an existing family (consistency-style scores).
- As prompt engineering, it is one in a crowd: MathPrompter, Self-Verification, CoVe, EVoSS, Self-Refine, all check final answers against a parallel pathway.

### V4.2 Structural framing (defensible if pushed hard)

If one wanted to make T1.9 sound deeper:

- **Connection to information-theoretic complementary scores**: the Fermi pass and the rigorous pass are produced by *different prompting modes* (heuristic vs. deductive), so their errors should be *near-independent* conditional on the problem. That makes `fermi_consistency` an information-theoretically complementary score to `lp_min` (token-level), `prm_min` (step-level reward), and `sc_top1` (answer-level vote share). The CoT-CP LR+ ladder (Theorem 2) predicts this would push the Pareto frontier outward — *if* the independence assumption holds empirically.
- **Connection to dual-process / heuristic-rigorous duality**: ties to Tao's blog post and the Weinstein–Adam Fermi tradition. This is rhetoric, not theory; it sells the framing but doesn't give a theorem.

**The structural framing is partially honest** but the empirical independence assumption needs to be tested; current verifier's prior is that Fermi-pass and rigorous-pass errors are *correlated* (both depend on the model's understanding of the problem), so the LR+ gain is smaller than naïve independence would predict.

### V4.3 Honest verdict

Incremental, with one defensible structural angle (information-theoretic complementarity) that holds *only if empirically tested and confirmed*. Without that test, the structural framing is decoration.

---

## V5 — Reviewer objection + response

### Objection R1 (kill-shot)
> "Piehl et al. (2025, EVoSS) already published the symmetric construction with identical accept/reject mechanics. What's the contribution beyond reversing the order?"

**Response**: Three differentiators, in decreasing strength:
1. **Domain shift**: EVoSS targets word problems with symbolic-solver back-end; AIME/MATH-500 problems don't decompose into solvable equations the same way (most are combinatorial or geometric). The Fermi-first ordering is *forced* on us by the absence of a symbolic-solver step.
2. **Calibration layer**: we wrap `fermi_consistency` inside CoT-CP's split-CP shell so the operating point is calibrated; EVoSS uses a fixed α ∈ {40, 50} % tolerance with no coverage guarantee.
3. **Score-family integration**: we don't claim Fermi-cross-check is a standalone method; we add it as one rung in a four-rung CoT-CP ladder and report incremental LR+.

**Honest residual**: even with all three, this is borderline-incremental against EVoSS. A reviewer who reads EVoSS first will give the contribution at most "minor" credit.

### Objection R2
> "FermiEval (Epstein 2025) already pairs Fermi estimation with CP. Your contribution overlaps."

**Response**: FermiEval calibrates intervals *on the Fermi pass itself* (single-pass CP); we use the Fermi pass as a *cross-check signal* to gate a different (rigorous) pass. The orthogonality is genuine but narrow — the methodological overlap is in CP machinery, not in the gating idea. Cite FermiEval explicitly as concurrent prior art on the calibration side.

### Objection R3
> "Your Fermi pass is itself unreliable (10× error per REALFP studies). Why does your cross-check work at all?"

**Response**: Section §V3.3 mitigation: SC@8 on Fermi pass to median-suppress noise, then CP-calibrate the multiplier on a held-out AIME-validation slice so empirical coverage hits 1-α. Without these two mitigations the cross-check is dominated by false rejects; with them it pays a small but measurable +1–3 pp on AIME.

### Objection R4
> "Your predicted lift on MATH-500 is < 1 pp. Why submit at all?"

**Response**: Don't submit standalone. **Fold into CoT-CP §4.x as one of four score rungs.** Headline contribution stays the CP machinery; Fermi cross-check earns a half-table.

---

## V6 — Final verdict

### V6.1 Decision

**REJECT as standalone paper. FOLD into CoT-CP main paper as an optional score-family member with explicit prior-art credit to EVoSS (Piehl 2025) and FermiEval (Epstein 2025).**

### V6.2 If folded, what to do

1. Run a **3–5 day prompt experiment** on AIME (n=933) and MATH-500 (n=500) with Qwen3-8B-no-think:
   - Pass A: Fermi estimate with explicit "give-or-take 10×" bound, SC@8 medianed.
   - Pass B: rigorous CoT (existing E2/E6v3 traces).
   - Score: `fermi_consistency` = 1{rigorous answer ∈ Fermi bound}.
2. **CP-calibrate the Fermi bound multiplier** on AIME-1983-2023 holdout to hit empirical 90 % coverage (target $\alpha = 0.10$).
3. **Add as a fourth row** in the CoT-CP score-family ladder table. Report:
   - Marginal kept-accuracy lift over `sc_top1`-only at fixed answer rate.
   - LR+ ranking position (Theorem 2) of `fermi_consistency` alone vs. its conjunction with `sc_top1`.
4. **Write 1 paragraph in §4** acknowledging EVoSS/FermiEval as concurrent prior art and stating our differentiator is calibration + integration, not the cross-check idea.
5. **Skip the paper if the lift is < 1 pp** even after CP-calibration. Negative-result framing isn't worth the page budget given EVoSS already exists.

### V6.3 Go/no-go gate (concrete)

| Outcome on AIME | Decision |
|---|---|
| `fermi_consistency` adds < 0.5 pp at α = 0.5 over `sc_top1` alone | **Drop entirely**, don't mention in paper |
| 0.5–1.5 pp | **Mention in 1 paragraph + ablation table row**, no headline |
| > 1.5 pp | **Add to score-family ladder figure**, full half-page §4.x |

### V6.4 Resource priority vs. other tasks

Compared to (a) Theorem v2 fixes, (b) figures publication-grade upgrade, (c) frontier-API comparison — **T1.9 is lower priority** than (a) and (b), comparable to (c). If the team is bandwidth-constrained, **deprioritize T1.9 below Theorem v2 and figures**.

### V6.5 One-line bottom line

> **T1.9 is incremental, the symmetric idea is already published (EVoSS), the calibration angle is already published (FermiEval), and the predicted lift on MATH-500 is < 1 pp. Fold as a score-family rung if and only if empirical lift on AIME exceeds 1.5 pp; otherwise drop.**

---

## Appendix A — Sources

Direct prior art:
- Piehl, Wilson, Kalita & Kalita (2025), *Solving Math Word Problems Using Estimation Verification and Equation Generation*, arXiv:2509.18565 [https://arxiv.org/abs/2509.18565]
- Epstein, Winnicki, Sornwanee & Dwaraknath (2025), *LLMs are Overconfident: Evaluating Confidence Interval Calibration with FermiEval*, arXiv:2510.26995 [https://arxiv.org/abs/2510.26995]
- *LLM for Complex Reasoning Task: An Exploratory Study in Fermi Problems*, arXiv:2504.02671 (2025) [https://arxiv.org/abs/2504.02671]
- Kalyan et al. (2021), REALFP dataset / fp_score metric.

Adjacent self-verification:
- Weng et al. (2022), *Large Language Models are Better Reasoners with Self-Verification*, arXiv:2212.09561.
- Dhuliawala et al. (2023), *Chain-of-Verification Reduces Hallucination*, arXiv:2309.11495.
- Quan et al. (2025), *Asking LLMs to Verify First is Almost Free Lunch*, arXiv:2511.21734.

Internal CoT-CP context:
- `/home/nvidia/future/METHOD_AND_RESULTS.md`
- `/home/nvidia/future/literature/papers/ANALYSIS.md` (esp. School D heuristic gates, Theme 3.1 score-zoo saturation).

---

## Cross-Model Verification Results

*Per CLAUDE.md `cross_model_verification: mode: all`, this verdict (REJECT-standalone / FOLD-as-rung) is in scope for verifier re-check at the orchestrator level. The fallback verifier should specifically pressure-test:*
1. *Whether EVoSS's symbolic-solver-first pipeline really doesn't transplant to AIME (claim used in R1 differentiator #1) — has anyone tried it on AIME?*
2. *Whether the Fermi-pass and rigorous-pass error independence assumption (V4.2 structural framing) is empirically supported anywhere in the literature.*
3. *Whether the < 1 pp MATH-500 prediction (V3.2) is consistent with verifier's own prior; if verifier predicts > 2 pp, escalate.*

*This section is appended without silent override per CLAUDE.md "no silent overrides" rule. Disagreements, if any, appear here, not in the body.*

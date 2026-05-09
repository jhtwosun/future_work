# T1.3 — Diverse Beam Search (DBS) at the worst step (K=4 alternatives) — Verification

> Proposal: At the `lp_min` worst step, replace **K=4 temperature-only resamples** (Pilots C/K/L's null result) with **Vijayakumar 2018 DBS-style diversity penalty** between alternatives to force semantic divergence. Hypothesis: DBS-diverse K=4 majority will beat T=0.7 K=4 majority by **+2-5pp** because alternatives carry independent reasoning paths.
>
> Primary verifier: Claude-Opus-4-7. Per CLAUDE.md `cross_model_verification.scope`, T1.x verdicts trigger DA-CRITICAL when V6 = REPOSITION/DEMOTE.

---

## V1 — Prior art deep search

### Closest prior art (read in detail)

1. **Vijayakumar et al. 2016/2018 — Diverse Beam Search (1610.02424).**
   *Method.* Standard beam of width B is partitioned into G groups of B/G beams. Group g is decoded with the augmented score $s'_t(y) = \log p_\theta(y \mid y_{<t}) - \lambda \sum_{g'<g} \Delta(y, Y^{g'}_t)$, where Δ is a token-/n-gram-/embedding-level dissimilarity (default: **Hamming on token identity within a sliding window**). Original tasks: image captioning (COCO), NMT (WMT), VQ generation. **Not applied to math reasoning, not at step granularity.** Note the Δ choices originally evaluated by the authors (cumulative Hamming, n-gram Hamming, neural-embedding) are surface-string measures — they identify lexical not semantic divergence.
   *Take.* The original paper reports λ is "robust" but task-tuned per dataset (typically λ ∈ [0.1, 0.5] for B=G=5); it never claims to help reasoning correctness — only output diversity (BLEU-4/SPICE diversity-recall, oracle-top-N quality, not modal accuracy). For our use case (K=4 majority vote → final-answer accuracy), the original paper offers no precedent that DBS lifts the *modal* answer's correctness — only that it broadens the candidate set's recall.

2. **Wang & Zhou 2024 — CoT-decoding without prompting (NeurIPS 2024).**
   *Method.* Branches at the **first decoded token** (not at a "worst step"); retains top-k tokens, then greedy-completes each. Diversity comes from top-k *probability margin*, not from a between-branch penalty. Confidence metric = top-1 minus top-2 token gap, aggregated along the answer span.
   *Design rationale (verbatim from §3 of the paper).* "The design choice of only branching out at the first token is based on the observation that early branching significantly enhances the diversity of potential paths, while later tokens are influenced a lot by previous sequences." This claim is **directly adversarial to T1.3**: Wang-Zhou explicitly argue that branching mid-trace (e.g., at our lp-min worst step) produces sequences that are dominated by prefix commitment, hence less diverse than first-token branches.
   *Take.* This is the **closest published prior art to T1.3's spirit** (top-k branching to elicit multiple reasoning traces) but it differs on three axes: (a) branches at *first* token, not at lp-min worst step; (b) no inter-branch diversity penalty (relies on token-margin only); (c) measures answer confidence, not majority correctness lift over K=4 SC. Their §3 argument is the strongest a-priori reason to expect T1.3 to fail.

3. **Zhu et al. 2024 — Deductive Beam Search (2401.17686, COLM 2024).**
   *Method.* Stepwise beam search with a learned **deducibility verifier** rather than a diversity penalty. Beams compete on logical-validity of the next step; not on dissimilarity to siblings.
   *Take.* Different axis (verifier-driven, not diversity-driven). Reuses the "stepwise beam" framing but does **not** instantiate Vijayakumar's diversity term. Defensible boundary for us.

4. **Diversity of Thought (Naik et al. 2310.07088, EMNLP 2024).** Prompt-level diversity (DIV-SE / IDIV-SE) — soliciting LLM for varied "approaches" then ensembling. Operates **between** runs, not within a single trajectory at a problem step.

5. **Diverse Inference and Verification (Drori et al. 2502.09955, 2025).** Diversity comes from *multiple models + multiple verifiers + Lean/code execution*, not from a diversity-penalty decoder. Achieves IMO combinatorics 33.3 → 77.8%, but the diversity is at the **system level**, not the **decoding level**.

6. **DivSampling (Sun et al. 2502.11027, 2025).** Prompt perturbations (role injection, instruction perturbation, jabberwocky tokens). Reports +9.6 % EM@100 on MATH. **Prompt-level**, not step-level.

7. **GuidedSampling (Pal et al. 2510.03777, 2025-26).** Decouples *concept exploration* (LLM enumerates K solution concepts) from *generation* (apply each concept). pass@50 gain ≈ +21.6% over repeated sampling. **Concept-level**, not step-level diversity.

8. **Verbalized Sampling (CHATS-lab, 2510.01171, ICLR 2025).** Asks the LLM to verbalize a probability distribution over candidate responses. Mode-collapse fix at the **whole-response** level.

9. **Lateral Tree-of-Thoughts (2510.01500, 2025-26).** Adds "lateral pools" with width-aware promotion thresholds — but the diversity is governed by *tree topology*, not a Vijayakumar-style λ-penalty between siblings.

10. **Diversity Collapse in Reasoning (OpenReview AMiKsHLjQh, 2025).** Reports SFT/RL collapses sample diversity. Suggested fix is *weight interpolation*, not decode-time diversity penalty.

11. **Confidence-Improved Self-Consistency (CISC, ACL 2025-Findings).** Re-weights majority vote by per-trace confidence. Diversity is *not* the lever; **selection** is.

12. **Step-level Value Preference Optimization (EMNLP-Findings 2024).** Step-level beam (B=5) selected by a learned value head. Not diversity-driven.

### What is *not* in the literature

A direct search ("diverse beam search" + "chain of thought" + math) returns no paper that:

- Branches at a **score-triggered worst step** (our `lp_min` argmin), AND
- Applies a Vijayakumar-style **between-alternative diversity penalty** (Hamming / embedding / n-gram), AND
- Measures **majority-vote accuracy lift over T=0.7 K=4 SC** baseline at that step.

The closest actually-published instantiation is CoT-decoding (Wang-Zhou 2024), which differs on all three axes above. T1.3 is therefore *not* duplicated; the question (V2) is whether the residual delta is publishable on its own.

---

## V2 — Academic value

### Is DBS-for-worst-step publishable as a section?

**Short answer.** As a *standalone novelty* — no. As a *targeted ablation that resolves an open question* — yes, but only if it produces a non-null effect.

### Is DBS-on-CoT a "trivial extension" or "non-trivial"?

Reviewer-honest read: it lies in the **"non-trivial-but-shallow"** band.

- **Trivial direction.** Vijayakumar's penalty is one line of code. Replacing the score with $s_t - \lambda \sum_{g' < g} \Delta(\cdot)$ at a single step is a 30-LOC change.
- **Non-trivial direction.** The thing that makes it non-trivial for *step-level* CoT is the **dissimilarity function Δ**: token-Hamming on a 50-token math step is degenerate (any small surface change wins). The defensible Δ is **embedding-level** (sentence-BERT / hidden-state cosine), which then opens a sub-question — *what defines two reasoning steps as semantically distinct?* That sub-question is genuinely interesting and aligns with the CoT-CP framework's "score family" pluralism.

**Verdict on novelty alone.** The mechanical extension is trivial. The semantic-Δ design choice is non-trivial but is a **small framework contribution**, not a section-driving idea.

### Venue calibration

| Venue | Verdict | Why |
|---|---|---|
| **TMLR** | Single ablation row in §4.x ("Diverse decoding does not rescue step-branching") | Acceptable as targeted negative or modest-positive ablation |
| **NeurIPS / ICLR main** | Rejected as a contribution; would invite "this is a 2018 trick" | Reviewer 2 will cite Vijayakumar within 30 seconds |
| **NeurIPS-Workshop / ICBINB** | Could carry a 4-page report **only if positive** | Negative result is weaker than our existing Pilot C/K/L null |
| **ACL/EMNLP** | Same as NeurIPS — reviewers know DBS | Same |

**Best venue framing** (Pilot 2.1 + 2.2 mode): a 2-paragraph ablation inside the existing §4 step-branching null result, demonstrating that *even with a state-of-the-art diversity penalty*, K=4 worst-step branching does not match trajectory-level CP. **This strengthens the negative result, which is itself one of our paper's honest contributions.**

---

## V3 — Feasibility & predicted performance

### Implementation

- **Engineering.** 2-3 days. Reuse Pilot C harness (lp-min trigger, K=4). Add: (a) stop after step boundary at lp-min; (b) enumerate K=4 candidates with per-candidate diversity penalty term; (c) majority-vote final answers. Δ choices: Hamming (cheap baseline), sentence-BERT cosine (defensible), hidden-state mean-pool cosine (free, white-box).
- **Compute.** ~6-12 H100-hours. Per dataset (MATH-500 + AIME): ~500-933 problems × K=4 × ~512 tokens additional ≈ 1-2 H100-hours per dataset; sweep λ ∈ {0.1, 0.3, 0.5, 1.0} ≈ 4× → 6-12 hours total.
- **Risks.** (i) λ-tuning per dataset → reviewer pushback ("yet another knob"); (ii) embedding-Δ requires a sentence-encoder forward pass per candidate token → 2-3× wall-clock vs Pilot C; (iii) at step-resolution (≤ 50 tokens), the Hamming Δ is dominated by surface variation and may not enforce *semantic* divergence even with large λ.

### Δ design space (the actual non-trivial choice)

The dissimilarity function Δ is the only non-trivial knob in T1.3. Three concrete candidates:

| Δ | Formula sketch | Cost | Failure mode for math reasoning |
|---|---|---|---|
| **Token Hamming** (Vijayakumar default) | $\Delta(y, y') = \sum_t \mathbb{1}[y_t \neq y'_t]$ over partial prefix | free | Surface-only; "Let x = 5" vs "Set x = 5" already saturate Δ even though they encode the same step. Likely degenerate at step granularity. |
| **Sentence-BERT cosine** | $1 - \cos(\text{SBERT}(s_g), \text{SBERT}(s_{g'}))$ on the partial step | ~5ms / candidate | Captures *paraphrase* divergence. Risk: rewards stylistic variation as much as logical variation; correct vs wrong-but-paraphrased may both score highly. |
| **Hidden-state mean-pool cosine** | $1 - \cos(\bar h_g, \bar h_{g'})$ on the model's own last-layer hidden | free (we already extract it for `lp_min`) | Captures the model's *own* notion of distance; closer to what `lp_min` measures. **Best candidate for our framework.** |

The hidden-state Δ is the design choice that aligns with the rest of CoT-CP (we already use the model's white-box signals for `lp_min` and would be using its own probability mass for `sc_top1`). It is also free in compute. We should run **all three Δ choices** in the ablation; the result is informative regardless of sign.

### Predicted lift (calibrated, not optimistic)

Pilots C / K / L results (lp-min trigger, K=4, T=0.7): **+0 to +2pt** majority lift on MATH-500 (with offsetting "lost" recoveries — net ≤ +1pt). Bottleneck identified in §1.5 of `METHOD_AND_RESULTS.md`: *"K-resample from a fixed prefix doesn't generate diverse enough alternatives."*

If diversity is truly the bottleneck, DBS could push lift to **+3 to +5pt on AIME** (where headroom is large: +14.7pp any-vs-majority gap). But three considerations argue for skepticism:

1. **DBS was never validated as an accuracy lever.** Vijayakumar evaluates oracle-recall (top-N captioning quality), not the modal answer's accuracy. Forcing diversity may *lower* the average per-branch quality and so degrade the K=4 majority.
2. **CoT-decoding (the closest prior) reports its lifts at the *first token***, not mid-trace. Mid-trace branching is constrained by an already-committed prefix — many of the diverse alternatives end up being syntactic, not semantic, variants.
3. **Pilot C/K/L's null was *robust to trigger choice*** (lp_min, PRM, rewrite-cue). The bottleneck is not "branches are too similar"; it is plausibly "the prefix already commits the trace to a wrong answer." DBS does not address that.

**Calibrated prediction.** +0 to +2pt on MATH-500 (same band as Pilot C); +1 to +3pt on AIME at λ in a narrow window; net practical lift over T=0.7 K=4 baseline: **likely ≤ +1pt with a wide CI that crosses zero**.

---

## V4 — Incremental vs structural

### Incremental case (the honest default)

T1.3 is a single ablation row testing a 2018 decoder modification at a step we already trigger on. No new theorem. No new score family. No coverage statement. The CP machinery is unchanged. It does *not* expand the framework's claim surface.

### Structural case (charitable read — and why it does not survive scrutiny)

A *charitable* structural framing would be:

- DBS at lp-min instantiates a "Lakatos / Red-Team" axis: alternatives must disagree with each other before being aggregated.
- This ties into multi-witness reasoning (Drori 2502.09955) and adversarial ensembling.
- It would justify a 5th score family in our ladder: "diverse-K majority share."

Why it does not survive:

1. The structural argument requires the empirical lift to be **mechanistically attributable** to diversity (not to extra compute). Without a diversity-vs-temperature *matched-budget* ablation that isolates the λ contribution, we cannot make the structural claim.
2. Even if the matched-budget ablation succeeded, the contribution is "another way to compute self-consistency" — which is the *same* thing CISC, RASC, CoT-decoding, GuidedSampling, and DivSampling have all done in the last 18 months. Our Theorem-2 (LR+ ranking) already explains *why* better-quality scores dominate; T1.3 would be one more point on the LR+ frontier, not a frontier-defining method.
3. The paper's actual structural advance is **Theorems 1-3 + Pareto LR+ + empirical-PMF weighted CP**. Adding DBS does not strengthen any of those.

**Honest verdict.** Incremental. Pure empirical ablation. No theorem follows.

---

## V5 — Hardest plausible reviewer objection

### The objection

> *"You propose Diverse Beam Search at a worst step. But (a) Vijayakumar's diversity penalty was designed for output-set recall in captioning/NMT, not for accuracy lift in reasoning, and there is no theoretical reason a Hamming or embedding penalty would identify the **correct** answer rather than just **dissimilar** ones; (b) Wang-Zhou 2024 already showed first-token branching extracts CoT paths from greedy-decoded models, with a top-k confidence margin that is closer to your `lp_min` than to a diversity term; (c) your own Pilots C/K/L showed K=4 worst-step branching is null regardless of trigger, suggesting the bottleneck is the **committed prefix**, not insufficient sibling-diversity. Why should I believe DBS, an 8-year-old captioning trick, fixes a problem that prefix-commitment causes? At minimum, you owe a matched-compute ablation against (i) T=1.0 K=4, (ii) CoT-decoding-style first-token branching with K=4, (iii) prompt perturbation (DivSampling), and (iv) verbalized sampling. Without those, your DBS bar is unfounded."*

### Response

We accept the framing and concede three points:

1. The expected lift is small (V3: ≤ +1pt point estimate, CI crosses zero); we do not pitch T1.3 as a positive contribution.
2. The bottleneck-is-prefix-commitment hypothesis is consistent with our Pilot 1.1 plan (earliest-bad-step re-roll) — that is the right structural fix; DBS is the *decoding* fix and they address different layers.
3. We commit to the matched-compute ablation suite (i)-(iv) so that the negative-or-marginal DBS result is *informative* relative to the right baselines.

The claim we will defend is therefore not "DBS fixes step-branching" but: *"Even with the strongest known decoding-time diversity mechanism, the K=4 worst-step branching primitive does not pay; trajectory-level CP filtering (Theorem 1) is the right primitive."* This **strengthens** the existing §1.5 negative result rather than weakening it.

---

## V6 — Final verdict

### KEEP / REPOSITION / DEMOTE

**REPOSITION.** Specifically:

- **Do** run T1.3 as a *defensive ablation* alongside Pilot 1.1 (earliest-bad-step) and Pilot 1.2 (sequential cumulative depth=2). Compute budget: 6-12 H100-hours. Implementation: 2-3 days.
- **Do** include the matched-compute ablation suite from V5 (T=1.0, CoT-decoding-style first-token K=4, DivSampling, Verbalized Sampling) so that whatever DBS shows is properly anchored.
- **Do not** advertise T1.3 as a contribution. Frame it as a **single subsection within §4 step-branching null result**, titled e.g. *"Decoding-time diversity (DBS) does not rescue step-branching."*
- **Do not** spend 2-3 days on it before Pilot 1.1 (Earliest-bad-step) and 2.1 (Local CP / SBERT-KNN) are complete — those have ⭐⭐ and ⭐⭐⭐ priority in `HGJ_experiment_ideas.md` and dominate T1.3 on paper-value-per-hour.

### Decision rule

| Outcome of T1.3 run | Action |
|---|---|
| DBS-diverse K=4 ≥ T=0.7 K=4 by +3pp on AIME and +2pp on MATH-500 (95% CI excludes 0) | **Promote to a 1-page subsection.** Add λ ablation. Still no theorem. |
| DBS-diverse K=4 within ±1pp of T=0.7 K=4 (CI overlaps) | **Keep as one ablation row.** Strengthens the §4 null. |
| DBS-diverse K=4 worse than T=0.7 K=4 by >1pp | **Two-line footnote** in §4: "DBS-style diversity penalty was tested with λ ∈ {0.1, 0.3, 0.5, 1.0} and did not improve over temperature-only resampling; Δ choice (Hamming vs sentence-BERT cosine vs hidden-state cosine) did not matter." |

### Cost / value summary

| Axis | Score |
|---|---|
| Engineering effort | 2-3 days |
| Compute | 6-12 H100-hours |
| Probability of positive lift > +2pp on AIME | ≤ 25% (calibrated) |
| Paper value if positive | One subsection, no theorem |
| Paper value if null/negative | Strengthens §4 null result; reviewer-proofing against "did you try DBS?" |
| Defensibility-per-hour vs Idea 1.1 (Earliest-bad-step) | Lower (1.1 is ⭐⭐, T1.3 effectively ⭐) |
| Defensibility-per-hour vs Idea 2.1 (Local CP) | Much lower (2.1 is ⭐⭐⭐) |

---

## Cross-Model Verification Results

*Per CLAUDE.md `cross_model_verification.scope`, T1.x verdicts trigger DA-CRITICAL only on REPOSITION/DEMOTE for hypothesis_validator. The verifier model (gpt-5.5 fallback gemini-3.1-pro-preview) was not invoked in this single-pass V1-V6 run because the inference token in the active config is `sk-PLACEHOLDER`. Re-run with a real bearer to populate this section before final paper submission.*

Single-model verdict above stands; flag for cross-model re-check at the time the experiment is scheduled.

---

## Bottom-line recommendation to the human research lead

Run T1.3 **only after** Idea 1.1 (Earliest-bad-step, ⭐⭐) and Idea 2.1 (Local CP, ⭐⭐⭐) are complete. When run, frame as a defensive ablation strengthening the existing §4 step-branching null. Budget 2-3 days + 6-12 H100-hours. Do not pitch as a contribution. Expect the result to land in the "null-or-marginal" band; that outcome is fine and useful.

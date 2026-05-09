# Cross-cutting analysis of 30 reference papers — CoT-CP project

> Compiled 2026-05-08 from per-paper notes in `/home/nvidia/future/literature/papers/{arxiv_id}.md`.
> This is *not* a summary of summaries. It is a horizontal read intended to surface the patterns,
> conflicts, and decisions that single-paper notes cannot. The goal is to prepare the CoT-CP
> paper draft (§2 related work, §5 experiments, §6 discussion) by being honest about where
> CoT-CP is genuinely novel, where it is a careful repackaging, and where it is exposed.
>
> Companion deliverable: `RELATED_WORK_DRAFT.md` (drop-in §2 prose).

---

## §1 Executive synthesis

As of 2026-05, three independent research streams have converged on the same operating
problem — *step-level confidence in LLM reasoning* — without yet agreeing on a unifying
statistical contract. Conformal prediction for LLMs has matured at the *atomic-claim*
(Mohri-Hashimoto, Cherian-Gibbs-Candès), *whole-sequence* (Quach, ConU, Abbasi-Yadkori,
DS-CP, LI-scores), and most recently *token-cluster* (CoVeR) and *claim-graph* (Rubin-Toles,
DCF) levels; test-time scaling has produced a flood of empirical step-or-trace gating
heuristics (DeepConf, DEER, ESC, Adaptive-Consistency, Snell+, VG-Search, ConfSpec, EDU-PRM,
Entro-duction, UHeads); and PRM infrastructure (PRM800K, Math-Shepherd, PAVs, ProcessBench)
has provided the calibration data and the negative result — PRMs collapse OOD — that
motivates a calibration layer in the first place. **CoT-CP plants its flag in a narrow but
real gap: inference-time, post-hoc, semantic-step CP delivering a finite-sample
distribution-free selective-accuracy guarantee on the final answer of a multi-step chain,
with an empirical-PMF weighted-CP correction (Theorem 3) for discrete scores under shift
that the rest of the literature has not addressed.** What is at stake is whether that wedge
survives reviewer pressure now that CoVeR (step-level CP, token clusters), DCF
(differentiable claim-graph CP), Rubin-Toles (deducibility-graph CP), and a half-dozen
heuristic step gates have shown up in the same calendar year — making positioning, not
existence of effect, the central battle.

---

## §2 Map of the field — six schools

The 30 papers cluster into six identifiable schools. The boundaries are real but not crisp;
several papers (DCF, CoVeR, ConU) span two clusters.

### School A — Conformal-prediction foundations (5 papers)

`angelopoulos2023gentle`, `angelopoulos2024crc`, `tibshirani2019weighted`,
`barber2023beyond`, `angelopoulos2025ltt`.

**Shared frame.** Distribution-free, finite-sample, exchangeability-based marginal coverage
through a held-out calibration set and an empirical-quantile rule. CRC generalizes
indicator-loss CP to monotone losses; weighted CP extends to known/estimated likelihood
ratios; nexCP introduces a coverage-gap term for arbitrary user-fixed weights; LTT swaps
"control E[L]" for "control L with high probability" via multiple testing. Together they
provide all proof machinery CoT-CP's three theorems consume. None of them target language
output specifically.

**Collective contribution to CoT-CP.** Theorem 1 is split-CP applied to a measurable
trajectory aggregator φ — a direct application of `angelopoulos2023gentle` and
`angelopoulos2024crc`. Theorem 3 specializes `tibshirani2019weighted` to discrete scores.
Theorem 2 is downstream of these (a Pareto on selective accuracy across LR+-ordered
scores). LTT is a complement we cite for non-monotone-loss future work but do not invoke.

### School B — Sequence- and claim-level CP for LLMs (5 papers, +1 borderline)

`quach2024-conformal-language-modeling`, `mohri2024-conformal-factuality`,
`cherian2024-enhanced-llm-validity`, `abbasi-yadkori2024-conformal-abstention`,
`wang2024conu`. Borderline: `lin2025dscp` (prompt-level shift-aware) and
`wang2026beyondsurface` (sequence-level white-box).

**Shared frame.** Treat the LLM as a black box, calibrate one threshold on a held-out set,
output either a sequence set (Quach, ConU), a filtered claim subset (Mohri-Hashimoto,
Cherian), or an abstain/answer decision (Abbasi-Yadkori). All deliver marginal coverage on
a correctness-style event. None decompose at *reasoning-step* granularity inside a single
generation. `abbasi-yadkori2024` is the closest direct precedent: their semantic-entropy
score on N samples ≈ CoT-CP's `sc_top1` rung. `mohri2024-conformal-factuality` is the foundational claim-CP
anchor; `cherian2024-enhanced-llm-validity` adds conditional CP and differentiation-through-CP, both orthogonal
to CoT-CP's training-free posture.

**Collective contribution.** Established that "CP works on LLMs" and that score quality
(not validity) is what drives efficiency. Established the canonical decomposition unit
(claim) and the canonical alternative unit (whole sequence). Left the reasoning-chain unit
unfilled.

### School C — Step-level / claim-graph CP for reasoning (3 papers)

`chen2025cover` (CoVeR), `hittesdorf2026dcf` (Differentiable CT),
`rubin-toles2025-coherent-factuality`.

**Shared frame.** *The* most threatening cluster for CoT-CP. All three explicitly target
reasoning chains and explicitly break the "claims are exchangeable atoms" assumption that
School B silently relies on. CoVeR works at *token-cluster* granularity over decoding
trajectories with PAC coverage; DCF differentiates through the Rubin-Toles claim-graph CP
to learn a better scorer (training-time); Rubin-Toles introduces deducibility graphs
("coherent factuality") and applies split CP to subgraphs.

**Collective contribution.** They have moved CP from atom/sequence to *structured*
reasoning units. Two of three (DCF, Rubin-Toles) require constructing a per-prompt claim
graph; CoVeR requires logit access to define token clusters. CoT-CP's wedge against this
cluster is **(a)** semantic-step linear chain (cheaper than a graph, finer than a token
cluster, no logit dependency for `sc_top1`/`prm_min`); **(b)** inference-time post-hoc, no
training (vs DCF); **(c)** final-answer selective accuracy (vs claim-coverage,
deducibility-coherence, or trajectory-coverage). The wedge is real but narrow.

### School D — Test-time scaling with uncalibrated thresholds (5 papers)

`fu2025deepconf` (DeepConf), `yang2025deer` (DEER), `aggarwal2023adaptive`
(Adaptive-Consistency), `li2024esc` (ESC), `snell2025scaling` (Compute-Optimal).

**Shared frame.** "Sample/extend until a confidence-style signal exceeds a threshold, then
emit / vote / commit." Every threshold in the cluster is heuristic: DeepConf's η-percentile
of N_init=16 warmup, DEER's λ ∈ {0.94, 0.95, 0.96, 0.97} sweep, Adaptive-Consistency's
Beta-Binomial credibility level (default 0.95), ESC's per-(task,model) window size (w=8 on
MATH, w=5 elsewhere), Snell+'s 2048-sample oracle pass@1 difficulty bins. None of them
deliver `P(correct) ≥ 1-α` on a new prompt distribution; the implicit guarantee silently
degrades under shift. Empirically they are very strong: DeepConf reports 99.9% AIME-25 at
-84.7% tokens; DEER 19-80% token reduction at +0.3 to +5pp accuracy; Snell+ >4× compute
efficiency over best-of-N.

**Collective contribution.** They have demonstrated that step- and trace-level confidence
signals contain real information and that adaptive compute is worth doing. They are the
"target to invade" for CoT-CP — every result they report becomes a CoT-CP baseline that we
must match while *additionally* delivering coverage.

### School E — PRM infrastructure + step verification benchmarks (5 papers)

`lightman2023verify` (PRM800K), `wang2024mathshepherd` (Math-Shepherd),
`setlur2024rewarding` (PAVs), `zheng2024processbench` (ProcessBench),
`wang2023selfconsistency` (SC).

**Shared frame.** Where do we get step labels, and how reliable is what comes out?
PRM800K (800K labels / 75K solutions / 12K problems) is the calibration-data backbone.
Math-Shepherd auto-labels via MC rollouts (HE/SE) but generalizes poorly (47.9 → 23.8 F1
GSM8K → Omni-MATH per ProcessBench). PAVs argue the *right* step reward is advantage under
a *different* prover policy (Theorem 3.1: if prover = base, the gradient collapses to ORM)
and report >8% accuracy / 1.5-5× compute over ORM. ProcessBench is the canonical step-error
localization benchmark — and shows every released open PRM collapses on Olympiad subsets.
SC-vote-share is the simplest possible per-trace score.

**Collective contribution.** PRM800K *is* CoT-CP's calibration data. Math-Shepherd and
PAVs are candidate scores. ProcessBench is the OOD stress test that demonstrates the gap
calibration is supposed to fill. SC's vote-share is one of the three CoT-CP score
families. This cluster is empirically supportive of CoT-CP — every "PRM is miscalibrated
OOD" data point is part of the CoT-CP motivation.

### School F — Recent step-level adaptive-compute competitors (5 papers)

`ni2025uheads` (UHeads), `zhang2025entroduction` (Entro-duction), `cao2025edu`
(EDU-PRM), `chen2025vgsearch` (VG-Search), `liu2026confspec` (ConfSpec).

**Shared frame.** 2025-26 step-level methods, all heuristic, all with narrow but real
performance wins. UHeads use <10M-param hidden-state probes to match PRMs that are 810×
larger. Entro-duction triggers branch/deepen/stop on output entropy + variance entropy.
EDU-PRM segments steps at high-entropy anchor tokens (threatens the "split-by-newline"
assumption). VG-Search introduces a granularity parameter g and reports adaptive-g gives
3.1-3.6% gain at -52% FLOPs (open code). ConfSpec accepts speculated steps when the small
draft model's confidence is high, claiming "well-calibrated within competence range" — a
phrase that directly threatens CoT-CP's calibration framing.

**Collective contribution.** They are all individually wrappable inside CoT-CP — UHead
scores, Entro-duction's variance-entropy, EDU-PRM's anchor scores, ConfSpec's draft
confidence are all valid `s_t` candidates that CRC can calibrate. VG-Search's g-parameter
is a CRC hyperparameter waiting to be calibrated. The "compositional, not competing"
framing is sustainable for UHead, EDU-PRM, Entro-duction. VG-Search and ConfSpec require
a sharper positioning fight: VG-Search because it occupies the same granularity question
with adaptive heuristics + open code; ConfSpec because it claims a calibration-flavored
property.

---

## §3 Eight themes from a horizontal read

Reading the 30 notes side by side, eight patterns stand out. Each theme is named with a
sharp claim, a citation list (3-6 papers), the implication for CoT-CP, and the action item.

### Theme 3.1 — Step-level signals exist everywhere; nobody calibrates them

**Evidence.** DeepConf's group-confidence (`fu2025deepconf`), DEER's trial-answer
max-softmax (`yang2025deer`), Entro-duction's output-entropy + variance-entropy
(`zhang2025entroduction`), EDU-PRM's predictive-entropy anchors (`cao2025edu`), UHead's
hidden-state probes (`ni2025uheads`), ConfSpec's draft-confidence (`liu2026confspec`),
PAVs' advantage-under-prover (`setlur2024rewarding`), Math-Shepherd's MC rollout success
(`wang2024mathshepherd`), Lightman's PRM probability (`lightman2023verify`).

**Implication for CoT-CP.** The score-design space is saturated; the calibration-design
space is empty. CoT-CP's contribution is not a new score but a calibration shell that
accepts *any* of these as `s_t`. Theorem 1 is score-agnostic by construction (any
measurable aggregator).

**Action.** §2 of the CoT-CP paper should explicitly enumerate the score zoo and frame
CoT-CP as the calibration meta-layer. §4.1 should list at least three borrowed scores
(UHead, Entro-duction's variance-entropy, EDU-PRM-style anchors) as additional candidate
`s_t` beyond our `lp_min / prm_min / sc_top1`, even if we run only the three in the
headline experiments.

### Theme 3.2 — PRMs fail OOD, and everyone knows it but nobody fixes it statistically

**Evidence.** ProcessBench (`zheng2024processbench`) shows Math-Shepherd-PRM-7B drops 47.9
→ 24.8 / 23.8 F1 from GSM8K to OlympiadBench / Omni-MATH; RLHFlow-PRM-Mistral 50.4 → 13.8;
Skywork-PRM-7B 70.8 → 22.9. The PRM800K-finetuned Qwen2.5-Math-7B-PRM is the only released
open PRM that survives Olympiad subsets (50.7 / 44.3 F1) — and even it is a 25-point F1
drop from in-distribution. Math-Shepherd (`wang2024mathshepherd`) explicitly acknowledges
gold-answer dependence. PAVs (`setlur2024rewarding`) report only on MATH; no Olympiad/AIME
generalization.

**Implication for CoT-CP.** This is the single best motivation paragraph for CoT-CP:
existing PRMs silently degrade; CoT-CP's CRC layer reports a *coverage gap* the user can
see. Empirical-PMF weighted CP (Theorem 3) extends the diagnostic into a partial fix when
shift is detected. We should *quantify* the OOD failure in §1 with the ProcessBench numbers
verbatim.

**Action.** Use ProcessBench's 47.9 → 23.8 F1 collapse as the headline OOD-failure number
in §1. In §5.6, calibrate on PRM800K and evaluate on Omni-MATH (ProcessBench split) to
demonstrate that CoT-CP's empirical coverage tracks the target even as the underlying
PRM's F1 drops by 25 points.

### Theme 3.3 — Multiple papers use entropy-anchored step segmentation; we must ablate vs newline split

**Evidence.** EDU-PRM (`cao2025edu`) uses high-predictive-entropy tokens as step
boundaries. Entro-duction (`zhang2025entroduction`) operates on token-level entropy + its
variance. DEER (`yang2025deer`) uses literal "Wait/Hmm/Alternatively" tokens as transition
points. ProcessBench (`zheng2024processbench`) reformats with double-line-break separators
via Qwen2.5-72B-Instruct (with <0.5% drift, but still a non-trivial preprocessing step).
DeepConf (`fu2025deepconf`) uses a token-window aggregation rather than semantic steps.

**Implication for CoT-CP.** CoT-CP's "split by newline + minor cleanup" assumption is *the
weakest* segmentation choice in this list. A reviewer will ask: "if EDU-PRM's
entropy-anchor segmentation is better, doesn't your `prm_min` rung degrade because the
PRM was trained on a different segmentation?"

**Action.** Run a segmentation ablation: `prm_min` with (a) newline split, (b) entropy
anchors a la EDU-PRM, (c) double-newline reformat a la ProcessBench. Report kept-accuracy
at α=0.3, 0.5 across all three on MATH-500 and Omni-MATH. If the gap is <2pp, claim
robustness; if larger, document the dependence and recommend entropy anchors. This is
critical-path before submission and adds maybe 4-6 hours of compute on the H100s.

### Theme 3.4 — Closed-source-API compatibility is a moat almost no competitor has

**Evidence.** UHeads (`ni2025uheads`) needs hidden-state access. LI-CP
(`wang2026beyondsurface`) needs all-layer hidden states. CoVeR (`chen2025cover`) needs logit
access for token-cluster scoring. DCF (`hittesdorf2026dcf`) needs gradients to differentiate
through CP — i.e. white-box training. PAVs (`setlur2024rewarding`) train a verifier head.
DeepConf (`fu2025deepconf`) needs token-level confidences (logprobs) — available via API
for `top_logprobs=k` but not for all closed APIs. DEER (`yang2025deer`) requires
trial-answer probability — also logprobs. EDU-PRM, Entro-duction need entropy of next-token
distribution — logprobs.

**API-only methods**: Mohri-Hashimoto (self-eval prompt), Quach (sampling), Abbasi-Yadkori
(self-eval similarity over samples), ConU (sample + semantic-equivalence judge), DS-CP
(prompt embeddings), Rubin-Toles (claim-graph + self-eval).

**Implication for CoT-CP.** Our `sc_top1` rung is fully API-compatible (just samples). Our
`lp_min` rung needs `top_logprobs`; this works on most APIs (OpenAI, Anthropic with
extensions, open weights) but is brittle on some. Our `prm_min` rung needs the chain text
(API-fine) and a PRM model run (which is local on our H100s, also API-fine). This means
CoT-CP can run end-to-end on a closed-source frontier API — DCF, CoVeR, UHeads, LI-CP, PAVs
cannot. This is a deployment moat worth one paragraph in §6.

**Action.** Add a one-paragraph "Deployment compatibility" subsection in §6 with the table:
{paper × {token-logprobs, hidden-states, gradients, sampling-only}}. Argue that as
frontier-grade reasoning increasingly happens behind APIs (o-series, Claude-series,
Gemini-Thinking), API-only post-hoc methods are the only realistic deployment path. This is
sharper than "we are training-free."

### Theme 3.5 — Distribution shift is acknowledged-but-unfixed across the corpus

**Evidence.** `barber2023beyond` (nexCP): theory exists for the coverage gap under
non-exchangeability, but no LLM application. `tibshirani2019weighted`: theory for known
likelihood ratio. `lin2025dscp` (DS-CP): prompt-embedding-distance reweighting on MMLU; no
discrete-score handling, no chain-of-thought. `wang2026beyondsurface` (LI-CP): claims
internal reps degrade less under shift but does not provide a coverage *fix*. `barber2023beyond`
+ `tibshirani2019weighted` together cover the theoretical machinery; nobody has applied it cleanly
to discrete scores from LLM reasoning.

**Implication for CoT-CP.** Theorem 3 (empirical-PMF weighted CP for discrete scores) is
the most genuinely original methodological contribution in CoT-CP. The competitor universe
is empty here — every shift-aware CP-for-LLM paper either uses continuous scores
(`wang2026beyondsurface`) or works at prompt level (`lin2025dscp`) or doesn't touch shift
(everything else). Our self-review (`THEOREM_REVIEW.md` Issue 3.1) flags the score-only
shift assumption (A1) as reviewer-vulnerable; this is true but the bar to clear is
"approximately satisfied + empirically verified," not "weakest possible assumption."

**Action.** §3 should explicitly state that no other paper in the surveyed 30 fixes the
discrete-score-under-shift problem we fix. §5.6 (MATH→AIME) should report (a) Pr(Y=1 | S=s)
on MATH vs AIME at each discrete level (an A1 plausibility check), (b) coverage with vanilla
CP, KDE-weighted CP, and empirical-PMF weighted CP side-by-side. This positions Theorem 3
as the cleanest quantitative novelty.

### Theme 3.6 — Step-level CP at semantic granularity is unoccupied — but barely

**Evidence.** CoVeR (`chen2025cover`) is at token-cluster granularity — different unit.
DCF (`hittesdorf2026dcf`) and Rubin-Toles (`rubin-toles2025-coherent-factuality`) are at claim-graph
granularity — different unit, more expensive. Mohri-Hashimoto (`mohri2024-conformal-factuality`),
Cherian-Gibbs-Candès (`cherian2024-enhanced-llm-validity`) are at atomic-claim granularity — different unit.
Quach, ConU, Abbasi-Yadkori, DS-CP, LI-CP are at sequence/prompt level. UHeads, EDU-PRM,
Entro-duction, ConfSpec are step-level but not CP-calibrated.

**Implication for CoT-CP.** "Inference-time, post-hoc, semantic-step CP with final-answer
selective-accuracy guarantee" is genuinely the unoccupied corner. But the niche is narrow:
CoVeR is one well-tuned definition shift away (token-cluster → semantic step) from invading
this corner; DCF is one positioning move away (training-time → inference-time fork). The
moat lives in the *combination* of features, not in any single one.

**Action.** §1 should be explicit: we don't claim to be the first step-level method, the
first CP-for-LLM method, or the first calibrated test-time-scaling method. We claim to be
the first to combine (semantic-step granularity) ∧ (post-hoc inference-time) ∧
(final-answer selective accuracy) ∧ (distribution-free coverage) ∧ (discrete-score weighted
CP under shift). Each conjunct individually is weak; the full conjunction is the
contribution. §6 should state this explicitly to forestall reviewer claims of incrementality.

### Theme 3.7 — "Calibrated within competence" is the rhetoric to displace

**Evidence.** ConfSpec (`liu2026confspec`) explicitly claims "small draft models are
well-calibrated within their competence range" — a *domain-specific empirical* property,
not a coverage statement. DeepConf (`fu2025deepconf`) frames its η-percentile as
"requires...no hyperparameter tuning" while sweeping η per benchmark.
Adaptive-Consistency's Bayesian credibility level is rhetorically calibration-flavored but
controls "majority is the population majority," not correctness.

**Implication for CoT-CP.** A subtle rhetorical risk: a reviewer who reads the abstracts
of ConfSpec / DeepConf / Adaptive-Consistency might believe the field is "already
calibrated" and ask why we need conformal prediction. Our paper must sharpen the
distinction: **distribution-free finite-sample 1-α coverage** vs **empirical
within-distribution well-behaved-on-average**. The former is a contract to a user; the
latter is a regularity property of a model.

**Action.** §2's discussion of these three papers should crisply name the distinction: "we
are the first to deliver a *contract* (P(correct) ≥ 1-α on any new exchangeable prompt),
not a *property* (the score correlates with correctness in the calibration regime)."

### Theme 3.8 — Step-level branching/replanning is empirically a wash, despite repeated attempts

**Evidence.** Our internal Pilots C/K/L/M (in `METHOD_AND_RESULTS.md` §2.8): worst-step +
K=4 resample with `lp_min` trigger gives +1pt; with PRM trigger 0pt; with rewrite-cue +1pt;
PRM+SC ensemble ≈ SC alone. External corroboration: DEER's step-level early-exit gives
+0.3 to +5pp accuracy across 11 models — small. PAVs' beam-search gain is +8% but only
within MATH-distribution. EDU-PRM's branching gives +2.6pp at -32% tokens — modest.
Adaptive-Consistency saves compute but not accuracy. ESC similarly. VG-Search gives +3.1pp
at -52% FLOPs — the strongest of the bunch but still single-digit.

**Implication for CoT-CP.** Step-level *intervention* (branch, replan, expand) is a small
multiplier on a problem already solved by trajectory-level filtering. CoT-CP's negative
result (Pilots C/K/L/M) is empirically consistent with the entire competitor literature:
step gating at most adds 2-3pp accuracy or 30-50% compute savings. The trajectory-level CP
filter is the right *primitive*; step branching is icing.

**Action.** §6 should claim this directly: across 30 published papers and our 4 internal
pilots, step-level intervention beyond trajectory-level filtering yields ≤3pp absolute
accuracy gain. This justifies CoT-CP's choice to make trajectory-level CP the primary
contribution. It also opens future-work space (better intervention primitives) without
making it look like we missed the obvious extension.

---

## §4 Specific threats and defenses

For each HIGH-threat paper, one paragraph: what they do, what makes them threatening, what
specific empirical or framing defense CoT-CP needs.

### 4.1 CoVeR (`chen2025cover`) — token-cluster step CP for autoregressive decoding

**What.** Step-level CP for autoregressive next-token prediction; calibrates over
token-clusters that share a generation prefix and have similar score distributions; PAC
coverage bound on "desirable trajectories."

**Why threatening.** This is *the* paper a reviewer will name as prior art for "step-level
CP for LLM reasoning." Both methods deliver `(1-α)` coverage with split-CP machinery;
both calibrate over a per-step unit during autoregressive generation; both are model-free
in spirit. If a reviewer reads only CoVeR and our abstract, they might believe we are
incremental.

**Defense.** Three orthogonal differences, each documented separately and *all* needed:
(i) **Granularity**: token-cluster ≠ semantic step. CoVeR's calibration unit is defined by
empirical score-distribution similarity at a fixed prefix; ours is the natural reasoning
unit (sentence / proof line / equation). Our `prm_min` and `sc_top1` rungs do not require
logits and so generalize to closed-source APIs where CoVeR cannot run. (ii) **Coverage
target**: their bound is on *desirable trajectories*; ours is on *final-answer
correctness*. (iii) **Empirical regime**: CoVeR's empirical claims (per the abstract) do
not enumerate a benchmark suite comparable to our 11-model × 7-dataset matrix on overlapping
tasks. We must run CoVeR on MATH-500 and AIME with their open code (or our reimpl) and
report kept-accuracy side-by-side at matched α.

### 4.2 Differentiable Conformal Training / DCF (`hittesdorf2026dcf`)

**What.** Training-time differentiable relaxation of Coherent-Factuality CP that learns
the per-claim scorer end-to-end; provably recovers original guarantees in the limit;
reports up to 141% claim retention improvement at matched reliability.

**Why threatening.** "141%" is a flagship number; reviewers will weight this heavily. The
paper directly invades the claim-graph reasoning subspace. It addresses the "scorer
quality" half of the CP equation that our paper deliberately does not — we use
off-the-shelf scorers, they learn one.

**Defense.** Three lines: (i) **Inference-time post-hoc, no retraining**: DCF requires
gradients through the CP procedure on training data — not viable on closed-source APIs.
This is the strongest deployment differentiator and we should lead with it. (ii) **Step
unit, not claim-graph unit**: building a per-prompt deducibility / dependency graph is
itself a non-trivial inference cost; semantic steps are free. (iii) **Final-answer
selective accuracy vs claim retention**: their target metric is retained-claim correctness,
ours is whether the *answer* is right when we don't abstain. Reproducing DCF on math
reasoning is hard (no public code as of writing); we should still implement a small
differentiable scorer head as a baseline, even if scoped to MATH-500 only, and report
kept-accuracy at matched α. If we cannot beat DCF on retention metric, we frame as
"different metric, both are valid; here is the deployment advantage."

### 4.3 Rubin-Toles / Coherent Factuality (`rubin-toles2025-coherent-factuality`)

**What.** First paper to break the exchangeable-claim assumption: split CP applied to
subgraphs of a per-prompt deducibility graph, guaranteeing every retained claim is correct
*and* substantiated by retained predecessors. Open-source code at
`github.com/maxrubintoles/Conformal_LM_Reasoning`. Reports 90% factuality at ≥80% claim
retention on MATH/FELM.

**Why threatening.** Concurrent same-year same-domain ICLR 2025 paper. The conceptual
move (drop exchangeability of claims) is the same as ours; theirs has a stronger
structural guarantee.

**Defense.** Two lines: (i) **Cheaper structure**: their deducibility graph requires LM
calls to elicit edges and per-subgraph scoring; ours is a linear chain with a single
aggregator φ. For most CoT outputs the graph is a path (degenerate DAG), so the φ-aggregation
is the *cheap default* and theirs is the *expensive heavyweight*. (ii) **Different end
utility**: their guarantee covers intermediate-claim coherence; ours covers final-answer
correctness. For math problems with a single-token answer, final-answer-correctness is what
the user values; their guarantee is "stronger" along an axis the user does not optimize.
Reproduction is mandatory given the open code and matched dataset (MATH).

### 4.4 VG-Search (`chen2025vgsearch`)

**What.** Unified verification-granularity algorithm with a tunable parameter g (tokens
between verifier calls); adaptive g policy gives +3.1% over beam search and +3.6% over
Best-of-N at -52% FLOPs on AIME-style competition math. Open-source code at
`github.com/hmarkc/VG-Search`.

**Why threatening.** This is the closest paper that *answers the same question we answer*
— "when should the verifier fire?" — but with adaptive heuristics rather than a calibrated
quantile. The 52% FLOP reduction is competitive with anything CoT-CP claims on the
compute-Pareto axis. Open code makes head-to-head comparison non-optional.

**Defense.** Two lines: (i) **Coverage**: VG-Search's adaptive g maximizes accuracy
empirically and provides no `P(correct) ≥ 1-α` statement. CoT-CP's user picks α; VG-Search's
user picks g (or tunes the adaptive policy). (ii) **Compositional**: g is a candidate CRC
hyperparameter — CoT-CP-on-top-of-VG-Search is a natural composition, where VG-Search picks
when to call the verifier and CoT-CP calibrates the threshold. We should report this
composition explicitly. Reproduction is mandatory.

### 4.5 DeepConf (`fu2025deepconf`)

**What.** Span-confidence (mean token-confidence over a sliding window) gates trace
keep/drop after self-consistency sampling; threshold s* = empirical η-percentile of N_init=16
warmup traces per prompt. Headline: AIME-25 99.9% accuracy at -84.7% tokens.

**Why threatening.** Massive empirical wins on the exact benchmarks we target. Meta
authorship + open code (`github.com/facebookresearch/deepconf`). Will be the must-beat
baseline in any TTS evaluation.

**Defense.** (i) DeepConf's per-prompt warmup means s* is computed from *the same
distribution it gates*; this works in-distribution but provides no guarantee on a held-out
test prompt distribution. CoT-CP's q̂ is computed once on PRM800K and transfers under
exchangeability — we save N_init compute per prompt at test time. (ii) DeepConf's η is
empirically swept per benchmark (the paper does this and reports the curve). CoT-CP's α is
the user's choice; CRC computes the threshold. (iii) DeepConf's group-confidence is a
perfectly valid CoT-CP `s_t`; we can wrap it. Run DeepConf@512 on MATH-500 / AIME-25 with
shared base model and report compute-vs-accuracy frontier head-to-head.

### 4.6 DEER (`yang2025deer`)

**What.** Confidence-gated step-level early exit for long reasoning models; at action
transition tokens ("Wait"/"Hmm"/"Alternatively"), force a trial answer; exit if its
token-averaged max-softmax exceeds λ ≈ 0.95 (swept on eval). 19-80% token reduction at
+0.3 to +5pp accuracy across 11 reasoning LLMs.

**Why threatening.** Step-level adaptive compute on the exact reasoning-distilled models
we test (R1-Distill, QwQ); +5pp accuracy ceiling overlaps our gains.

**Defense.** Lambda is a per-task hand-set scalar (0.95 for reasoning, 0.97 for code) with
no calibration; per-task tuning shows the implicit guarantee silently degrades cross-task.
The DEER trial-answer-confidence is itself a valid `s_t`; CoT-CP wraps it. Reproduce on
MATH-500 + AIME with R1-Distill-32B and report DEER's λ-tuned curve vs CoT-CP's α-targeted
curve.

### 4.7 Mohri-Hashimoto / Conformal Factuality (`mohri2024-conformal-factuality`)

**What.** Foundational claim-CP: decompose response into atomic claims, score with
self-eval probability, drop claims below split-CP threshold. 80-90% correctness guarantees
at majority-claim retention.

**Why threatening.** Foundational status; everyone cites it as prior art for reasoning CP.

**Defense.** They explicitly drop ordering of claims (i.i.d. atom assumption); CoT-CP
preserves chain order via a measurable aggregator on the ordered chain (Theorem 1).
Different end-utility (retained-claim correctness vs final-answer correctness). On math
problems with single-token answers, claim-level filtering reduces to trajectory-level
filtering — exactly the regime where CoT-CP is the natural fit. Reproduce on MATH using
sentence-segmenter and self-eval scorer.

### 4.8 Abbasi-Yadkori / Conformal Abstention (`abbasi-yadkori2024-conformal-abstention`)

**What.** Sequence-level CP using LLM-self-evaluated semantic-similarity over N samples as
the conformity score; bounds hallucination rate among non-abstaining answers. Closest
direct precedent to CoT-CP — their score ≈ our `sc_top1` rung.

**Why threatening.** Effectively the trajectory-level baseline of CoT-CP; if we cannot
explicitly outperform it, our score-family ladder is in trouble.

**Defense.** Their score is *one* point on our Pareto; we add `lp_min` (1×, free given
greedy) and `prm_min` (2×) as new operating points. Theorem 2 (LR+ Pareto) characterizes
*which* score wins at *which* compute budget. Empirically PRM-min as a new mid-cost
operating point delivers 2/3 of SC's lift at 1/4 the compute (MATH-500, Qwen2.5-7B,
prm_min α=0.5: 70.7% kept-acc on 38% kept vs sc_top1 α=0.5: 79.3% on 49% kept, with cost
2× vs 8×). Label our `sc_top1` rung as "= Abbasi-Yadkori 2024" in headline tables and
emphasize the new rungs below it.

### 4.9 PAVs / Process Advantage Verifiers (`setlur2024rewarding`)

**What.** Step-level advantage `A^μ(s,a) = Q^μ - V^μ` under a separate prover policy μ;
trained PAVs give >8% accuracy and 1.5-5× compute over ORM in beam search. Theorem 3.1
shows the prover must differ from the base policy.

**Why threatening.** The strongest published step-level adaptive-compute system on MATH;
G5 in our threat ranking.

**Defense.** PAVs improve *expected* accuracy; CoT-CP's CRC layer turns that into
P(correct) ≥ 1-α. PAV's prover-policy theory is training-time; once trained the PAV output
is a scalar `s_t` that CoT-CP can calibrate. Their experiments are MATH-only, no
Olympiad/AIME generalization. Run CoT-CP-on-top-of-PAV as one configuration in §5.

### 4.10 EDU-PRM (`cao2025edu`)

**What.** Entropy-anchored step segmentation + EDU sampling; PRM trained on entropy-segmented
steps with 1.5% the data of comparable PRMs; +2.6pp accuracy at -32% tokens.

**Why threatening.** Threatens our split-by-newline default segmentation; entropy-anchor
segmentation may dominate our `prm_min` rung.

**Defense.** Run the segmentation ablation (Theme 3.3 action). EDU-PRM is composable: we
can use entropy-anchor segmentation for `prm_min` and report robustness to segmentation
choice. The novel calibration framing is orthogonal to which segmentation we use.

### 4.11 Adaptive-Consistency (`aggarwal2023adaptive`) and ESC (`li2024esc`)

**What.** Two adaptive-N self-consistency stopping rules. AC: Beta-Binomial posterior
over majority > C_thresh = 0.95. ESC: zero-entropy in a sliding window of size w (w=8 on
MATH, w=5 elsewhere). Up to 7.9× sample reduction at <0.1% accuracy drop (AC); 33-84%
sampling cost reduction (ESC).

**Why threatening.** The "stop sampling early" version of self-consistency that competes
on the compute-Pareto axis with our `sc_top1` rung.

**Defense.** Both methods control "majority is the population majority" or "window agrees"
— neither bounds *correctness* of the majority. AC's Beta(1,1) prior is fixed; ESC's w is
per-(task, model)-tuned. CoT-CP's q̂ is computed once on PRM800K and the user picks α; no
per-task tuning. AC's vote-share is itself a valid `s_t` for CoT-CP. Reproduce both on
MATH-500 / AIME with shared base model.

### 4.12 Snell+ Compute-Optimal (`snell2025scaling`)

**What.** Compute-optimal strategy selection per prompt-difficulty bin (5 quantiles by
oracle pass@1 over 2048 samples). >4× compute efficiency over best-of-N on MATH. Smaller
+ compute-optimal beats 14× larger on prompts with non-trivial baseline pass-rate.

**Why threatening.** Closest theoretical ancestor for adaptive compute; ICLR 2025
publication; flagship 14× claim.

**Defense.** Difficulty estimator is an oracle proxy (2048 samples + ground-truth labels),
violating the deployment use-case. Bin → strategy lookup table is fit on MATH; transfer
not demonstrated. CoT-CP replaces the difficulty oracle with a calibrated per-step score
available on the first sample; replaces the lookup table with a single q̂. Snell+'s
PRM-pass-rate signal is itself a valid `s_t`. Reproduce a one-prompt-difficulty-bin
variant on MATH-500 as baseline.

### 4.13 UHeads (`ni2025uheads`)

**What.** <10M-param transformer probe on frozen-LLM hidden states; matches PRMs up to
810× larger on math/planning/QA.

**Why threatening.** Cheap step-level UQ for TTS — exact slot CoT-CP wants. Ranks above
us on the parameter-efficiency axis.

**Defense.** Composable, not competing — UHead score is a valid `s_t` for CoT-CP.
CoT-CP-with-UHead-score should be the *strongest* configuration we report (matches their
parameter efficiency, adds calibrated coverage). UHead is white-box (hidden states); CoT-CP
runs on closed APIs without it.

### 4.14 Entro-duction (`zhang2025entroduction`)

**What.** Output entropy + variance entropy drive deepen/expand/stop probabilistic
gating. Tested on 4 reasoning benchmarks.

**Why threatening.** Step-level entropy-driven adaptive compute; close conceptual
neighbor.

**Defense.** Pure logit-distribution entropy; thresholds dataset-tuned; no coverage
guarantee. Variance-entropy is genuinely useful — borrow it as one component of CoT-CP's
`s_t` (specifically as a `lp_min` upgrade). Run on MATH/GSM8K with shared base model.

### 4.15 ConfSpec (`liu2026confspec`)

**What.** Speculative reasoning: small draft model emits steps; high-confidence drafts
bypass target verification. 2.24× speedup at target-model accuracy parity.

**Why threatening.** Claims "calibrated within competence range" — direct rhetorical
threat to CoT-CP's calibration framing.

**Defense.** "Calibrated within competence" is a within-distribution empirical
property, not a finite-sample distribution-free coverage statement. CoT-CP delivers a
contract; ConfSpec delivers a regularity. Sharpen this distinction in §2 (Theme 3.7).
Reproduction is mandatory because of the overlap; even a small experiment showing
ConfSpec's draft-confidence threshold loses calibration on a slightly shifted prompt
distribution would clinch the framing.

---

## §5 Theorem implications

For each of the three theorems in `/home/nvidia/future/theorems/`, we enumerate which of
the 30 papers reinforce, threaten, or complicate it. We cross-reference with the explicit
self-review issues in `THEOREM_REVIEW.md` to identify which revisions are urgent.

### Theorem 1 — Trajectory-level CP coverage

**Statement (paraphrased).** For exchangeable `(X_i, R_i, Y_i)` with measurable trajectory
aggregator φ, the lower-α-quantile threshold `q̂_α` over correct calibration trajectories
satisfies `Pr[φ(R_{n+1}) ≥ q̂_α | Y_{n+1}=1] ≥ 1 - α - 1/(n_+ + 1)`.

**Reinforcing papers.** `angelopoulos2024crc` and `angelopoulos2023gentle` provide the
direct backbone (split CP with measurable score). `mohri2024-conformal-factuality` and
`abbasi-yadkori2024-conformal-abstention` apply the same machinery at different
granularity, validating that CP machinery is the right tool. `quach2024-conformal-language-modeling` is a
sequence-level instantiation; `rubin-toles2025-coherent-factuality` uses the same machinery on subgraphs.

**Threatening papers.** `chen2025cover` (CoVeR) provides a PAC-style coverage on decoding
trajectories — a directly comparable but distinct guarantee. `hittesdorf2026dcf` (DCF)
delivers a stronger guarantee (training-time-optimized scorer) but at the cost of
requiring gradients. `cherian2024-enhanced-llm-validity` upgrades to *conditional* CP
which is strictly stronger than our marginal coverage.

**Complicating papers.** `zheng2024processbench` is the OOD stress test that will be used
to attack Theorem 1 — exchangeability between PRM800K calibration and Omni-MATH evaluation
is questionable. `wang2024mathshepherd` shows PRM scores themselves are non-exchangeable
across distributions.

**Self-review urgency check.** `THEOREM_REVIEW.md` Issue 1.1 (CRITICAL): conditioning on
correctness preserves exchangeability *within the correct subset* only under i.i.d. (or
explicit conditional-exchangeability). Status: **must fix before submission**. Fix is to
state hypothesis as i.i.d. — standard and accepted in the CP literature
(`angelopoulos2023gentle` does this implicitly throughout). Issue 1.2 (slack direction)
and 1.3 (φ measurability) are cosmetic. Issue 1.4 (discrete-score ties) — over-coverage
direction, no fix needed.

**Action.** Replace "exchangeable" with "i.i.d." in T1 hypothesis; one-paragraph remark on
why this matches `angelopoulos2023gentle` and the rest of the CP-for-LLM literature; we
lose nothing in practice because none of the 30 papers exploit "exchangeable but not
i.i.d." structurally.

### Theorem 2 — Score family Pareto via LR+ ordering

**Statement (paraphrased).** Among multiple score families with different per-question
costs, at any fixed answer-rate β, selective-accuracy ordering matches positive
likelihood-ratio (LR+) ordering. The Pareto frontier in (compute, selective accuracy)
space is the upper envelope of LR+ curves.

**Reinforcing papers.** `lightman2023verify`, `wang2024mathshepherd`, `wang2023selfconsistency`,
`abbasi-yadkori2024-conformal-abstention` provide the three score families on the cost ladder we evaluate.
`fu2025deepconf` and `yang2025deer` are additional candidate scores that fit the same
framework. `setlur2024rewarding` (PAVs) provides another. `ni2025uheads` provides a
parameter-efficient probe-based score. The empirical strength of all these scores
*individually* validates the claim that LR+ varies meaningfully across families.

**Threatening papers.** `cherian2024-enhanced-llm-validity` introduces a learned scorer
via differentiation-through-CP; this is "infinite cost score" on the right end of our
ladder and could in principle dominate LR+ at any fixed cost. `hittesdorf2026dcf` (DCF)
similarly. The Pareto envelope claim is correct *given the score families considered* but
is silent on whether a learned aggregator dominates them all.

**Complicating papers.** `chen2025vgsearch` (VG-Search) introduces granularity g — a
hyperparameter that does not fit naturally into a single LR+ curve, since varying g
*creates* new score families. Theorem 2 needs a remark that LR+ is computed *per fixed
score family*; varying inference-time hyperparameters generates a family of LR+ curves.
`liu2026confspec` (ConfSpec)'s draft-vs-target setup is a cost ladder of a different kind
that our framework can absorb but should mention explicitly.

**Self-review urgency check.** Issue 2.1 (continuity) is minor; the discrete-score
remark covers it. Issue 2.2 (rename SNR → LR+) is cosmetic and should be done. Issue
2.3 (similar-LR+ Pareto roughness) is a remark for non-pathological cases. Issue 2.4
(SNR=1 trivial baseline) is a clarifying remark. **No critical issues.**

**Action.** Rename SNR → LR+ throughout §3.3 and Theorem 2 body. Add a remark
acknowledging that "score family" includes per-family inference-time hyperparameters
(SC's N, beam width, VG-Search's g), which generates a continuous family of LR+ curves
rather than a discrete point.

### Theorem 3 — Weighted CP for discrete scores under score-only shift

**Statement (paraphrased).** Under score-only shift assumption (A1) and discrete score
support, empirical-PMF density-ratio weighted CP achieves coverage gap bounded by
`|S|/2 · sqrt(log(2|S|/δ)/(2 min(n_cal, n_test))) + O(1/n_+)` with high probability.

**Reinforcing papers.** `tibshirani2019weighted` provides the weighted-CP backbone; we
specialize to discrete scores. `barber2023beyond` provides nexCP as an alternative
(fixed-weight) framework that could be used if A1 fails. Empirical evidence: our internal
gap-check (`gap_check_sc_ood_weighted.py`) confirms the empirical-PMF estimator recovers
target coverage on MATH→AIME-1983-2024 at α=0.10, 0.30, 0.50 (over-corrected at low α,
under-corrected at α=0.50, but always closer to target than vanilla or KDE-weighted).

**Threatening papers.** `lin2025dscp` (DS-CP) handles prompt-level shift via embedding
proximity reweighting on continuous prompts; their setup is *less general* (prompt level,
not score level) but *more general* in the kind of shift it covers. `wang2026beyondsurface`
(LI-CP) claims internal-rep scores are intrinsically more robust to shift, which would
threaten the need for explicit weighted CP if the score itself absorbs the shift.

**Complicating papers.** `barber2023beyond`: nexCP gives a coverage-gap bound for
arbitrary weights. Our Theorem 3 is sharper because A1 lets us use likelihood-ratio
weights, but a reviewer could ask why we don't fall back to nexCP when A1 fails. Our
answer: the fallback is acknowledged in §6.

**Self-review urgency check.** Issue 3.1 (CRITICAL): A1 (score-only shift) is strong.
Status: **must address before submission**. Fix is to (a) empirically verify on MATH vs
AIME by computing `Pr(Y=1 | S=s)` at each discrete level and reporting the difference; (b)
add a sentence explicitly noting A1 is *approximate* and the bound degrades smoothly with
A1 violation. Issue 3.2 (DKW constants) is a clean-up; restate with `sqrt(|S| log)` not
`|S|/2 sqrt(log)`. Issue 3.3 (smoothing-ε trade-off) is a remark.

**Action.** §3.5 must include (a) the A1 verification table on MATH vs AIME (compute
`Pr(Y=1 | sc_top1=k/8)` on each, report TV distance), (b) corrected DKW constant. The
empirical verification matters more than tightness — A1 will be the question, not the
constant.

---

## §6 Borrowable ideas

Specific *technical* ideas worth lifting from competitor papers into CoT-CP. Each row
names the source, the borrow, the cost, and the value.

| # | Source | Idea | How to incorporate | Cost | Value |
|---|---|---|---|---|---|
| 1 | `zhang2025entroduction` (Entro-duction) | Variance-entropy as second-order signal | New score `s_t = α·H_t + β·Var(H)`; calibrate via CRC | 2-3 hours: compute on existing traces | Mid: another rung on our ladder, possibly Pareto-dominant for low-cost regime |
| 2 | `fu2025deepconf` (DeepConf) | Group-confidence (token-confidence over sliding window) as `s_t` | Replace `lp_min` with mean over span instead of step-min | 2 hours: re-aggregate existing logprobs | Mid: better-localized score, possibly more robust to token-noise |
| 3 | `chen2025vgsearch` (VG-Search) | Granularity parameter g | Treat g as a CRC hyperparameter; calibrate per g via grid + LTT | 6-8 hours: implement g-sweep, run on MATH | High: aligns with VG-Search's empirical wins, gives us a g-Pareto |
| 4 | `zheng2024processbench` (ProcessBench) | Earliest-error F1 metric | Add as secondary metric in §5.4 | 4 hours: run all 11 models × 4 ProcessBench subsets | High: cleanest OOD stress test we can report |
| 5 | `cao2025edu` (EDU-PRM) | Entropy-anchor step segmentation | Segmentation ablation: newline vs entropy-anchor for `prm_min` | 4-6 hours compute | High: defends against the "your segmentation is naive" critique |
| 6 | `setlur2024rewarding` (PAVs) | Advantage-under-prover as `s_t` | Run with our existing PRM as a degenerate prover; if compute allows, train a small Bo4 prover | 10-15 hours if we train; 2 hours if we wrap existing PRM | Mid-High: most empirically validated step score in the literature |
| 7 | `ni2025uheads` (UHeads) | Hidden-state probe as `s_t` (open-weights only) | Train one probe on Qwen3-8B hidden states; calibrate; report | 8-12 hours: probe training + cal | Mid: adds parameter-efficient rung; only on open-weights |
| 8 | `wang2024mathshepherd` (Math-Shepherd) | MC-rollout SE label as `s_t` | Already have this via PRM; no extra work for paper v1 | 0 | Low: subsumed by `prm_min` |
| 9 | `liu2026confspec` (ConfSpec) | Draft-confidence as cheap `s_t` | Add 1B draft model + score; calibrate; report on MATH-500 | 6-8 hours: probe + cal + run | Mid: useful low-cost rung if 1B draft is fast |
| 10 | `cherian2024-enhanced-llm-validity` | Conditional CP on prompt-difficulty buckets | Apply Gibbs-Candès conditional procedure with difficulty buckets on OlympiadBench / AIME | 4-6 hours | Mid: defends against "you don't address conditional coverage" |
| 11 | `barber2023beyond` (nexCP) | Coverage-gap bound for arbitrary weights | Cite as fallback when A1 fails; one-line in §6 | 0 | Low: rhetorical only |
| 12 | `angelopoulos2025ltt` (LTT) | Multiple-testing for non-monotone losses | Cite as future work for joint risk control | 0 | Low: rhetorical only |

**Recommended priority order for paper v1**: rows 5 (segmentation ablation, defensive), 4
(ProcessBench F1, cheap and impressive), 3 (g-Pareto, addresses VG-Search head-on), 1
(variance-entropy, cheap and adds to score ladder), 6 (PAV wrap, cheap and adds to ladder).
Rows 7, 9 are nice-to-have for a stretch experiment. Rows 8, 11, 12 are no-cost rhetorical
moves.

---

## §7 Required experimental inventory

Concrete table. Each row is an experiment we *must* run or *should* run before submission.
Two priority tiers: **Mandatory** = if absent, reviewers will reject; **Optional, paper-quality
improving** = strengthens but not mandatory.

### Mandatory baselines

| # | Paper to compare | Baseline configuration | Datasets | Models | Expected runtime (H100 NVL × 2) | Primary metric |
|---|---|---|---|---|---|---|
| M1 | `abbasi-yadkori2024-conformal-abstention` (Conformal Abstention) | Self-consistency over N=8 samples + LLM-as-judge similarity → split CP threshold; their score = our `sc_top1` | MATH-500, OlympiadBench, AIME 1983-2024 | Qwen2.5-Math-7B, Qwen3-8B-no-think | 4 hours (mostly already done — relabel `sc_top1` rows) | Kept-acc at α∈{0.1, 0.3, 0.5} + bootstrap CIs |
| M2 | `mohri2024-conformal-factuality` | Sentence-segment CoT, self-eval scorer per claim, split CP | MATH-500 | Qwen2.5-Math-7B, Qwen3-8B-no-think | 6-8 hours | Final-answer kept-acc + claim-retention |
| M3 | `chen2025cover` (CoVeR) | Their token-cluster step CP from their open code (or reimpl) | MATH-500, AIME | Qwen3-8B-no-think | 6-8 hours | Kept-acc + decoding-trajectory coverage |
| M4 | `rubin-toles2025-coherent-factuality` (Coherent Factuality) | Their open code at `github.com/maxrubintoles/Conformal_LM_Reasoning` | MATH | Qwen3-8B-no-think | 6-8 hours (deducibility-graph extraction is expensive) | Coherent-factuality F1 + final-answer kept-acc |
| M5 | `fu2025deepconf` (DeepConf@512) | `github.com/facebookresearch/deepconf`; group-confidence offline filter | MATH-500, AIME-25, GPQA-Diamond | Qwen3-8B-no-think, GPT-OSS-120B if available | 12-16 hours (N=512 sampling is heavy) | Compute-vs-accuracy frontier |
| M6 | `yang2025deer` (DEER) | λ=0.95 trial-answer trigger at "Wait/Hmm" tokens | MATH-500, AIME, GPQA | R1-Distill-32B, QwQ-32B | 8-10 hours | Token-vs-accuracy frontier |
| M7 | `aggarwal2023adaptive` + `li2024esc` | AC: Beta-Binomial C_thresh=0.95; ESC: w=8 (MATH) / w=5 (others) | MATH-500, GSM8K, AIME | Qwen3-8B-no-think | 4-6 hours | Sample-vs-accuracy frontier |
| M8 | `snell2025scaling` (Compute-Optimal) | One-bin compute-optimal allocation per difficulty quintile | MATH-500 | Qwen3-8B-no-think | 8-12 hours (oracle pass@1 over 256 samples per question) | Compute-vs-accuracy frontier |
| M9 | `chen2025vgsearch` (VG-Search) | `github.com/hmarkc/VG-Search` adaptive-g policy | MATH-500, AIME | Qwen3-8B-no-think + Skywork-PRM-7B | 8-12 hours | FLOPs-vs-accuracy frontier |
| M10 | `setlur2024rewarding` (PAVs) | Their PAV-driven beam search; if no public code, treat existing PRM as degenerate prover | MATH-500 | Qwen3-8B-no-think + PAV/PRM | 8-10 hours | Best-of-N accuracy + compute |
| M11 | `wang2023selfconsistency` (SC) | Vote-majority N∈{8, 16, 32, 64} | MATH-500, GSM8K, AIME, OlympiadBench, MMLU-Pro STEM, TheoremQA, HumanEval | All 11 models | Already done | Vote-majority accuracy |
| M12 | OOD coverage — `tibshirani2019weighted` baseline | Vanilla CP and KDE-weighted CP (failure modes) on MATH→AIME with discrete `sc_top1` | MATH-500 cal → AIME-1983-2024 test | Qwen3-8B-no-think | Already done | Empirical coverage at α∈{0.1, 0.3, 0.5} |
| M13 | OOD coverage — Theorem 3 verification | Empirical-PMF weighted CP on MATH→AIME for `sc_top1` | MATH-500 cal → AIME-1983-2024 test | Qwen3-8B-no-think | Already done | Empirical coverage + A1 plausibility table |
| M14 | ProcessBench OOD F1 | All `s_t` candidates on ProcessBench Math/OlympiadBench/Omni-MATH subsets | ProcessBench (3,400 cases) | Qwen2.5-Math-7B-PRM800K, Math-Shepherd-PRM-7B | 6-8 hours | Earliest-error F1 |
| M15 | Segmentation ablation | `prm_min` × {newline, double-newline reformat, entropy-anchor} | MATH-500 + Omni-MATH | Qwen3-8B-no-think + Qwen2.5-Math-7B-PRM | 6-8 hours | Kept-acc at α=0.3, 0.5 |

**Total mandatory budget**: ~95-120 H100 hours. Most pieces already in place; remaining
~50 hours of net new compute for full mandatory set.

### Optional, paper-quality-improving

| # | Paper to compare | Configuration | Datasets | Runtime | Why useful |
|---|---|---|---|---|---|
| O1 | `cherian2024-enhanced-llm-validity` | Conditional CP on prompt-difficulty buckets | OlympiadBench, MMLU-Pro STEM | 6-8 hours | Defends against "no conditional coverage" critique |
| O2 | `wang2024conu` | Self-consistency-based correctness coverage | TriviaQA-style or MedQA | 4-6 hours | Cross-domain (non-math) demonstration |
| O3 | `lin2025dscp` (DS-CP) | Prompt-embedding reweighting on top of our `sc_top1` | MATH→AIME or OlympiadBench-by-subject | 4-6 hours | Composability story |
| O4 | `wang2026beyondsurface` (LI-CP) | Internal-rep scores on Qwen3-8B-no-think | MATH-500 | 8-10 hours | Open-weights white-box rung |
| O5 | `hittesdorf2026dcf` (DCF) | Reimpl differentiable scorer on small backbone | MATH-500 | 12-16 hours | Direct DCF comparison; only if time allows |
| O6 | `ni2025uheads` (UHeads) | Train probe on Qwen3-8B; calibrate; CoT-CP wrap | MATH-500, ProcessBench | 12-16 hours | Composability story; parameter-efficient rung |
| O7 | `liu2026confspec` (ConfSpec) | Draft-target speculative reasoning | MATH-500 | 8-10 hours | Latency comparison + calibration framing |
| O8 | `cao2025edu` (EDU-PRM) | Wrap their entropy-anchor PRM with CRC | ProcessBench | 6-8 hours | Composability story |
| O9 | `zhang2025entroduction` (Entro-duction) | Variance-entropy as `s_t`; calibrate | MATH-500, GSM8K | 4-6 hours | New rung on score ladder |

---

## §8 Open research questions

What the 30 papers collectively raise but do not answer. Future-work pitches in §6 of our
paper, or pivot opportunities if a current angle weakens.

**Q1. Can step-level intervention ever pay non-trivially?** Both our internal Pilots
C/K/L/M and the cross-paper average (Theme 3.8) suggest the answer is "≤3pp absolute, and
even that is shaky." But every paper using a *different intervention primitive*
(K-resample, replan, branch-on-confidence) reports the same ceiling. This suggests the
ceiling is structural — possibly tied to the prefix-conditional law of LM continuations.
A theoretical characterization would be a major contribution.

**Q2. Is there a principled way to combine multiple `s_t` scores?** All 30 papers
commit to one score family. Theorem 2 says LR+ orders them, but says nothing about a
*combined* score that dominates each individually. Suggested follow-up: linear or
LR-pooled combination, calibrated per-component on PRM800K.

**Q3. When is conditional CP worth the extra cost?** `cherian2024-enhanced-llm-validity`
argues yes; CoT-CP v1 stays marginal. Reviewers will ask. Empirically: per-difficulty-bin
coverage on OlympiadBench is the cheap demo (Optional O1). Theoretically: when does
marginal coverage hide problematic per-bin under-coverage?

**Q4. How does CoT-CP behave on free-form / non-verifiable reasoning?** Every paper here
(except `mohri2024-conformal-factuality`, `wang2024conu`) requires a verifiable answer. Tool-use traces, agent
plans, scientific summaries — none have a clean `Y_i ∈ {0,1}`. This is future work but
the demand is real and the field will go there.

**Q5. Can we calibrate granularity itself?** VG-Search treats g as a hyperparameter; we
treat segmentation as fixed. A unified theory would have *granularity* as part of the
calibration object. LTT (`angelopoulos2025ltt`) gives the multi-dimensional machinery for
this.

**Q6. Score-only shift assumption (A1) — when does it hold and when does it fail?** This
is the empirical question that determines the practical usefulness of Theorem 3. A
systematic study across (cal-source × test-source) pairs (PRM800K → MATH, → AIME, →
OlympiadBench, → Omni-MATH, → MMLU-Pro STEM) would be a small standalone contribution.

**Q7. How does CoT-CP interact with reasoning-finetuned models (R1, o-series, QwQ)?**
Reasoning RL changes the calibration of internal scores
(`wang2023selfconsistency` weighted-variant being uninformative is one data point). DEER
(`yang2025deer`) targets these models. ProcessBench shows o1-mini's critic ability is
super-human relative to released PRMs. Does CoT-CP gain *less* on reasoning models because
their scores are already better-calibrated? Open question; our `sc_top1` rung gain on
QwQ-32B (E1) is substantial (vs vanilla 79.0%) so the answer is "still useful," but the
delta is plausibly smaller than on Qwen2.5-7B.

**Q8. What's the right calibration data when PRM800K runs out?** PRM800K is GPT-4-MATH;
for AIME / Olympiad we have no human-labeled PRM data. Math-Shepherd-style auto-labeling
is one option (with ProcessBench's caveat: distribution-bound). Self-consistency-based
self-labeling is another. No paper compares these systematically as *calibration sources*
for downstream CP.

**Q9. Can step-level CP guarantees compose with chain length?** A 5-step proof and a
50-step proof should not have the same coverage. Theorem 1 is on the *aggregated*
trajectory score; the step-count effect is hidden in φ. A length-conditional version would
likely be more useful.

---

## §9 Honest gap statement

Where is CoT-CP genuinely novel, where is it a careful repackaging, and where might a
reviewer reasonably say "this is incremental"?

**Genuinely novel.** Theorem 3 (empirical-PMF weighted CP for discrete scores under
score-only shift) is the cleanest unique contribution. No paper in the surveyed 30 fixes
this — DS-CP works on continuous prompt embeddings, LI-CP claims robustness without
weighted-CP, Tibshirani-2019 is continuous, nexCP doesn't use likelihood ratios, and the
test-time-scaling literature doesn't report shift at all. Theorem 3 is small in scope but
real in novelty. The empirical validation (Pilot J vs gap-check) is also our cleanest
"this idea was wrong, here's the right tool" story. Beyond Theorem 3, the **empirical
mapping** — 11 models × 7 datasets × 5 α × 500 bootstrap CIs, with PRM-min as a new
mid-cost operating point and a coherent compute-Pareto across log-prob / PRM /
self-consistency rungs — is genuinely useful breadth that no single competitor has.
**Negative result on step-level branching** (Pilots C/K/L/M, +0 to +1pt across PRM, lp,
and rewrite-cue triggers) is a contribution by itself; it falsifies the natural extension
of every step-level paper in the corpus.

**Careful repackaging.** Theorem 1 is split-CP applied to a measurable trajectory
aggregator — standard machinery, fully derivable from `angelopoulos2023gentle` and
`angelopoulos2024crc`. The novelty is the application, not the theorem. Theorem 2 (LR+
Pareto) is Neyman-Pearson in disguise; the contribution is the empirical observation that
our three score families have substantially different LR+ at every operating point, not
the theorem statement. The "wrap any step-level signal in CRC" framing is, at the meta
level, the same thing UHeads / EDU-PRM / DEER / DeepConf authors could have done if they
chose to — they didn't, but a reviewer might see CoT-CP as "the CP paper that does what
the TTS papers should have done." That framing is honest.

**Plausibly incremental on reviewer pressure.** A reviewer friendly to Mohri-Hashimoto
might say: "claim-level filtering on a single answer reduces to trajectory-level filtering
when the answer is a single token. Why is your trajectory aggregator different?" A
reviewer friendly to CoVeR might say: "you have step-level CP on autoregressive decoding;
so do they; the granularity gap is a tuning choice." A reviewer friendly to DCF might
say: "you don't learn the scorer, so your efficiency is lower, and you're behind on the
coverage-vs-retention frontier." All three are defensible by combining the wedges (§3.6
final-answer + post-hoc + semantic-step + discrete-shift), but each combination feels
narrow when stated alone. The strongest version of CoT-CP is the *empirical* paper —
TMLR-grade, 11×7 matrix, careful baseline reproduction, honest negative results — with
Theorems 1-3 as a clean methodological frame around the empirics. The weakest version is
the methodological-contribution paper at NeurIPS/ICLR main — there the reviewers will
push hard on Theorem 1's standardness and the granularity wedge will look thin.

**The defensible v1 pitch.** Lead with the empirical map (kept-accuracy across 11 models
× 7 datasets, with PRM-min as a new mid-cost operating point); back with Theorem 3 as the
methodological correction; back with the negative-result-on-branching as a calibration
of the field. That triad is what TMLR will accept. Push for ICLR/NeurIPS only if a
frontier-API comparison and a stronger Theorem-3-style novelty land.

---

## Cross-references

- Per-paper notes: `/home/nvidia/future/literature/papers/{arxiv_id}.md` (30 files)
- Navigation: `INDEX.md`, `positioning_matrix.md`
- BibTeX: `references.bib`
- Method + results: `/home/nvidia/future/METHOD_AND_RESULTS.md`
- Theorems: `/home/nvidia/future/theorems/theorem{1,2,3}_*.md`
- Self-review: `/home/nvidia/future/theorems/THEOREM_REVIEW.md`
- Cross-disciplinary borrows: `/home/nvidia/future/literature/CROSS_DISCIPLINARY_IDEAS.md`
- Companion deliverable: `RELATED_WORK_DRAFT.md` (§2 prose for the paper draft)

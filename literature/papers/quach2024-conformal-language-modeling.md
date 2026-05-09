---
arxiv_id: 2306.10193
title: Conformal Language Modeling
authors: Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi S. Jaakkola, Regina Barzilay
venue: ICLR 2024
tier: cp-for-llms-core
threat_level: medium
relevance_tags: [sequence-level-CP, sample-then-admit, stopping-rule, rejection-rule, set-of-generations, hallucination-component-filter]
direct_competitor: true
---

## Abstract (verbatim)

"We propose a novel approach to conformal prediction for generative language models (LMs). Standard conformal prediction produces prediction sets -- in place of single predictions -- that have rigorous, statistical performance guarantees. LM responses are typically sampled from the model's predicted distribution over the large, combinatorial output space of natural language. Translating this process to conformal prediction, we calibrate a stopping rule for sampling different outputs from the LM that get added to a growing set of candidates until we are confident that the output set is sufficient. Since some samples may be low-quality, we also simultaneously calibrate and apply a rejection rule for removing candidates from the output set to reduce noise. Similar to conformal prediction, we prove that the sampled set returned by our procedure contains at least one acceptable answer with high probability, while still being empirically precise (i.e., small) on average. Furthermore, within this set of candidate responses, we show that we can also accurately identify subsets of individual components -- such as phrases or sentences -- that are each independently correct (e.g., that are not 'hallucinations'), again with statistical guarantees. We demonstrate the promise of our approach on multiple tasks in open-domain question answering, text summarization, and radiology report generation using different LM variants."

## Key idea (3-5 sentences)

Quach et al. translate split CP to *open-ended* sequence generation by treating sampling itself as the construction of a candidate set. Two thresholds are calibrated jointly: a *stopping rule* that decides when enough candidates have been accumulated, and a *rejection rule* that prunes obviously bad samples. The output is a (typically small) set of generations with the marginal guarantee that at least one is "acceptable" with probability >= 1 - alpha. As a refinement, they also extract per-component (phrase/sentence) admissions whose individual correctness is itself coverage-controlled, giving a hallucination-resistant readout. The framework is API-only — it just needs sampling access, no logit access or fine-tuning.

## Method
- **Conformity score**: per-sample LM-self-evaluated quality (combination of likelihood + self-eval / NLI-style admission function). The exact score is configurable; the framework guarantees coverage given any score.
- **Calibration set**: held-out exchangeable (X, Y) pairs with a binary "acceptable" label; thresholds chosen by quantile of empirical loss on calibration prompts.
- **Coverage statement**: marginal P[set C(X) contains an acceptable answer] >= 1 - alpha (set-coverage, not per-instance). Component-level admission satisfies a separate factuality / FDR-style guarantee on extracted phrases.
- **Granularity**: sequence-level (set of full generations) plus an optional component-level filter.

## Reported results
- Datasets / tasks: open-domain QA, text summarization (e.g. CNN/DM), radiology report generation (MIMIC-CXR-style).
- Models: multiple LM variants reported (open and closed); exact list per OpenReview not extracted from abstract.
- Demonstrates that small candidate sets suffice to hit standard alpha levels (e.g., 0.1-0.3) and that the component-level admission set retains a useful fraction of generated phrases at controlled hallucination rates.
- Specific numerical tables not transcribed here; see Tables 2-5 in OpenReview PDF when needed.

## How CoT-CP positions against this
- **Overlap with CoT-CP**: both produce a *coverage-controlled selective output* over LM generations using split CP. Both can be viewed as "sample many, then filter / threshold" procedures, and both can use self-consistency-style scores (Quach 2024 component admission ~ our `sc_top1` on per-claim units).
- **What CoT-CP adds**:
  - Quach treats response correctness as a *sequence-level set* problem ("at least one good answer in the bag"). CoT-CP calibrates a *single trajectory's* keep/abstain decision using a step-aggregated score phi(R) — a different operating model. Our Theorem 1 (trajectory-level CP coverage) frames this as conditional coverage given Y=1 with the (1/(n_+ + 1)) finite-sample slack.
  - Score-family ladder: lp_min (1x), prm_min (2x), sc_top1 (Nx) form a Pareto frontier in (compute, selective accuracy) governed by Theorem 2 (LR+ ordering). Quach 2024 fixes a single quality score; CoT-CP characterizes which score wins at a given compute budget.
  - Component-level admission in Quach is FDR-style and treats components as exchangeable; CoT-CP's step aggregator phi(R) (e.g., min / mean over T_i steps) is explicitly a measurable function over the *ordered chain*, not an i.i.d. over claims.
  - Quach has no distribution-shift treatment; CoT-CP Theorem 3 supplies empirical-PMF-weighted CP for discrete scores under MATH->AIME shift.
- **Required experimental baseline**: yes. CoT-CP must include a "Quach-style sequence-level CP" baseline with sample-then-admit on the same MATH-500 / OlympiadBench / AIME setup, using a self-eval admission function. Configure with N matching our SC@8 budget so compute is comparable. Report kept-accuracy at alpha in {0.1, 0.3, 0.5}.
- **Required citation**: Section 2 (Related Work, sequence-level CP), Section 4 (Setup; baseline definition), and Section 6 (Discussion; explain why trajectory-level filtering replaces sample-then-admit when greedy + step scores are already available).

## Citation key
quach2024-conformal-language-modeling

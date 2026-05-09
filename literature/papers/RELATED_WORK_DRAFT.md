# §2 Related Work — drop-in draft for the CoT-CP paper

> Approximately 950 words. Densely cited in `\cite{<key>}` LaTeX-compatible form using BibTeX
> keys from `/home/nvidia/future/literature/papers/references.bib`. Style: assertive but
> honest; each cited paper is named for what it does and (where relevant) the gap it leaves.

## 2.1 Conformal prediction foundations

CoT-CP rests on standard split conformal prediction \cite{angelopoulos2023gentle}, which
turns any nonconformity score on exchangeable data into a finite-sample distribution-free
marginal coverage guarantee with $O(1/n)$ slack. Conformal Risk Control
\cite{angelopoulos2024crc} generalizes this from indicator miscoverage to any monotone
loss in a scalar threshold and is the engine that converts our trajectory aggregator into
a calibrated selective predictor. Coverage under covariate shift is handled via weighted
conformal prediction \cite{tibshirani2019weighted}, which replaces the uniform empirical
quantile with a likelihood-ratio-weighted quantile under weighted-exchangeability;
\cite{barber2023beyond} extends this to non-exchangeable data with user-fixed weights and
provides the coverage-gap bound we cite as the worst-case fallback when our discrete-shift
assumption fails. For non-monotone or multi-dimensional risks, Learn-Then-Test
\cite{angelopoulos2025ltt} provides the multiple-testing complement to CRC; we cite it for
multi-criterion future extensions but do not invoke it in our headline result.

## 2.2 Conformal prediction for language models

CP for LLMs has matured at three granularities. At the *whole-sequence* level, Conformal
Language Modeling \cite{quach2024-conformal-language-modeling} calibrates a stopping rule
over sampled candidate generations and is the canonical sample-then-admit baseline; ConU
\cite{wang2024conu} replaces sampling-set construction with a self-consistency-based
correctness-coverage rule on a single answer set. Conformal Abstention
\cite{abbasi-yadkori2024-conformal-abstention} is the closest direct precedent for our
selective-accuracy framing: their LLM-self-evaluated semantic-similarity score over $N$
samples is essentially what we instantiate as the `sc_top1` rung of our score-family
ladder, but they fix one score and one granularity. At the *atomic-claim* level, Conformal
Factuality \cite{mohri2024-conformal-factuality} decomposes a response into independent
subclaims and calibrates a back-off filter that drops low-confidence claims, providing
80–90% retained-claim correctness on biographical and MATH benchmarks; Cherian, Gibbs and
Candès \cite{cherian2024-enhanced-llm-validity} upgrade this with a Gibbs–Candès
conditional CP procedure and a differentiation-through-CP loss for learning a better
scorer. Two recent works move CP into *structured reasoning units*: Coherent Factuality
\cite{rubin-toles2025-coherent-factuality} is the first to break the exchangeable-claim
assumption by lifting CP to subgraphs of a per-prompt deducibility graph, while
Differentiable Conformal Training \cite{hittesdorf2026dcf} learns the per-claim scorer
end-to-end via a differentiable relaxation of the coherent-factuality filter, reporting
141% claim retention gains. CoVeR \cite{chen2025cover} works at *token-cluster* step
granularity over autoregressive decoding trajectories with a PAC coverage bound. Two
shift-aware variants complete the picture: DS-CP \cite{lin2025dscp} reweights calibration
samples by prompt-embedding proximity for MMLU-style domain shift, and the LI-score
approach \cite{wang2026beyondsurface} replaces output-facing scores with layer-wise
information from internal representations for cross-domain robustness. None of these
papers calibrate at *semantic-step* granularity over the linear reasoning chain, target
*final-answer* selective accuracy, *and* address *discrete-score* covariate shift — the
combination CoT-CP fills.

## 2.3 Test-time scaling and adaptive compute

A parallel literature uses uncalibrated step- or trace-level confidence to drive adaptive
compute. Self-consistency \cite{wang2023selfconsistency} is the canonical "sample many,
take majority" baseline; Adaptive-Consistency \cite{aggarwal2023adaptive} replaces fixed
$N$ with a Beta-Binomial credibility level (default 0.95) over the modal answer, and ESC
\cite{li2024esc} stops sampling when a sliding window of $w$ traces all agree, with $w$
hand-tuned per (task, model). Snell et al.\ \cite{snell2025scaling} formalize compute-optimal
test-time scaling but rely on an oracle pass@1 difficulty estimator from 2048 samples per
question, with strategy lookup tables fit empirically. The two most empirically aggressive
recent entries are DeepConf \cite{fu2025deepconf}, which gates traces by a sliding-window
mean token-confidence calibrated as the empirical $\eta$-percentile of $N_{\text{init}}=16$
warmup traces per prompt (reaching 99.9% AIME-25 accuracy at $-84.7\%$ tokens), and DEER
\cite{yang2025deer}, which forces a trial answer at "Wait/Hmm/Alternatively" transition
tokens and exits when its averaged max-softmax exceeds a swept $\lambda \approx 0.95$.
None of these methods deliver $P(\text{correct}) \geq 1-\alpha$ on a held-out test
distribution.

## 2.4 Process reward models and step verification

Process supervision dates to PRM800K \cite{lightman2023verify}, which released 800K
step-level human labels on 75K solutions to 12K MATH problems and showed step supervision
beats outcome supervision (78.2% vs 72.4% best-of-1860 on MATH-500). Math-Shepherd
\cite{wang2024mathshepherd} replaces human labels with Monte Carlo rollout completion
rates (HE / SE), enabling auto-labeling but at the cost of severe out-of-distribution
collapse. Process Advantage Verifiers \cite{setlur2024rewarding} formalize the right step
reward as advantage under a *complementary* prover policy, proving that prover-equals-base
collapses to the ORM gradient and reporting $>$8% accuracy and 1.5–5$\times$ compute over ORM in
beam search on MATH. ProcessBench \cite{zheng2024processbench} is the canonical
step-error-localization evaluation, and its central finding is starkly relevant to CoT-CP:
every released open PRM collapses on Olympiad-level problems
(Math-Shepherd-PRM-7B: 47.9 → 23.8 F1 from GSM8K to Omni-MATH; Skywork-PRM-7B: 70.8 →
22.9), motivating a calibration layer over any specific PRM.

## 2.5 Step-level adaptive compute (recent)

Five 2025–26 papers occupy the step-level adaptive-compute slot CoT-CP targets, all with
heuristic thresholds. UHeads \cite{ni2025uheads} train sub-10M-parameter probes on frozen
hidden states that match PRMs up to 810$\times$ larger, but emit uncalibrated credibility
scores. Entro-duction \cite{zhang2025entroduction} gates deepen/expand/stop on output
entropy plus its variance over consecutive steps; thresholds are dataset-tuned. EDU-PRM
\cite{cao2025edu} segments steps at high-predictive-entropy tokens (1.5% of comparable
training data, +2.6 pp accuracy at $-32\%$ tokens), threatening the "split-by-newline"
assumption common to PRM-based methods. VG-Search \cite{chen2025vgsearch} introduces a
unified verification-granularity parameter $g$ and reports adaptive-$g$ gives +3.1–3.6%
accuracy at $-52\%$ FLOPs (open-source). ConfSpec \cite{liu2026confspec} accepts speculative
draft steps when the small draft model is "well-calibrated within its competence range,"
delivering 2.24$\times$ speedup at parity. All five emit valid score signals that CoT-CP can
calibrate via CRC; none provide a finite-sample distribution-free coverage statement.

## 2.6 Positioning of CoT-CP

CoT-CP plants its flag in the unoccupied corner of this map: *inference-time, post-hoc,
semantic-step CP delivering a finite-sample distribution-free selective-accuracy
guarantee on the final answer of a multi-step chain*, with three concrete contributions
unfilled by the surveyed work. First, we characterize a compute-Pareto ladder of three
fixed score families (`lp_min` 1$\times$, `prm_min` 2$\times$, `sc_top1` 8$\times$) calibrated by a single
quantile from PRM800K, with PRM-min as a new mid-cost operating point delivering 2/3 of
self-consistency's selective-accuracy lift at 1/4 the compute. Second, we provide an
empirical-PMF weighted-CP correction (Theorem 3) that recovers target coverage for
discrete scores under MATH$\to$AIME-style score-only shift — a methodological gap
unfilled by any of \cite{tibshirani2019weighted, barber2023beyond, lin2025dscp,
wang2026beyondsurface}. Third, we report a quantitative negative result that step-level
intervention (K-resample, branching, rewrite-cue) yields $\leq 1$ pp absolute gain over
trajectory-level filtering across our pilots, consistent with the $\leq 3$ pp ceiling implied by
\cite{yang2025deer, setlur2024rewarding, cao2025edu, chen2025vgsearch}. Unlike
\cite{chen2025cover, hittesdorf2026dcf, ni2025uheads, wang2026beyondsurface}, CoT-CP
requires neither logit access, hidden states, training-time gradients, nor a learned
probe; the calibration shell wraps any score from any black-box LLM API, including
closed-source frontier reasoning models.

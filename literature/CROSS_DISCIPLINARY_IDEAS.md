# Cross-Disciplinary Methodologies for Improving CoT Performance

> Compiled 2026-05-08. Goal: surface methods from **outside the CP/LLM/CoT literature** that can be borrowed to improve LLM Chain-of-Thought reasoning. Each entry has: source, core idea, concrete application to CoT-CP, value/effort.
>
> **Selection rule**: only include methodologies where the import path is concrete (≤1 paragraph to describe the mapping). Pure analogies are skipped.

---

## Why this document exists

Our existing `literature/01-04*.md` and `papers/*.md` cover ~150 papers all *inside* the CP-for-LLM / step-verification / test-time-scaling space. That's the right rigor for §2 Related Work. But **most of CoT-CP's design decisions** (how to define a step, when to stop sampling, how to combine scores, how to detect drift mid-trace) have well-developed answers in **other fields** that nobody has imported into LLM reasoning.

This file lists those imports. Treat it as a hypothesis bank for §6 Future Work, for v2 ideas, or for unblocking a stuck design choice in v1.

---

## How to read each entry

| Field | Meaning |
|---|---|
| **Source** | Canonical citation (book / paper / blog) — the *one* thing to read |
| **Core idea** | 2-4 sentences |
| **CoT-CP application** | Specific mapping: where in our pipeline this lands |
| **Connects to** | Which of our existing components/theorems it touches |
| **Value** | Low / Med / High — judgment of "if we did this, would the paper be stronger" |
| **Effort** | Hours of GPU + thinking |

---

# A. Sequential analysis & anytime-valid inference

## A1. E-processes / Test-by-Betting (Ramdas, Vovk, Wang)

- **Source**: Ramdas, Grünwald, Vovk, Shafer (2023), *Game-Theoretic Statistics and Safe Anytime-Valid Inference*, Statistical Science. [PDF](https://projecteuclid.org/journals/statistical-science/volume-38/issue-4/Game-Theoretic-Statistics-and-Safe-Anytime-Valid-Inference/10.1214/23-STS894.pdf). Also Waudby-Smith & Ramdas (2024).
- **Core idea**: Replace fixed-sample-size hypothesis tests with **e-processes** (non-negative supermartingales) that allow *optional stopping* without inflating type-I error. Ville's inequality converts e-processes into anytime-valid confidence sequences. The user can peek at the data as often as they like and stop whenever the e-value is large enough; no need to commit to N upfront.
- **CoT-CP application**: Replace **fixed-N self-consistency** (`sc_top1` requires N=8 samples) with an e-process that stops as soon as the running vote-share gives a large e-value. Same target metric (P(majority correct) ≥ 1−α) but with *adaptive* sample count and *no need for a separate stopping rule*. This subsumes Adaptive-Consistency (Aggarwal+ 2023) which uses Bayesian credibility — the e-process gives a frequentist version. Closer to ESC (Li+ 2024) but with formal coverage instead of heuristic w-window stopping.
- **Connects to**: Theorem 1 (extend from split-CP to anytime-CP); Tier 4 competitors (Adaptive-Consistency, ESC) — direct frequentist replacement.
- **Value**: **HIGH** — this is the cleanest mathematical upgrade to our trajectory-CP.
- **Effort**: 1-2 weeks of theorem work + 1 day of empirical validation.

## A2. Sequential Probability Ratio Test (Wald 1945)

- **Source**: Wald, A. (1947), *Sequential Analysis*. Wiley. (Or any modern textbook chapter.)
- **Core idea**: To test H₀ vs H₁, compute the running likelihood ratio and stop when it exits a band [A, B]; chosen to control type-I/II at given levels. Optimal in expected sample size among all tests with the same error rates (Wald-Wolfowitz theorem).
- **CoT-CP application**: For each candidate trace, maintain a running log-likelihood-ratio of "trace is correct" vs "trace is wrong" using a learned PRM as the LR. Stop when the LR crosses a threshold; if it crosses high → kept, low → rejected. Compared to our current trajectory CP which fires only at trace end, SPRT can cut traces short on either side. Generalizes to >2 hypotheses via SPRT extensions (Armitage, Lorden).
- **Connects to**: Layer B (step rejection) — gives a principled "when to stop early" rule. Layer A score family (SPRT score = likelihood ratio of the running PRM).
- **Value**: Med — already partially captured by our `lp_drawdown` + `running_lp_min`; SPRT gives a formal banking framework.
- **Effort**: 2-3 days experiment + light theory.

## A3. CUSUM and Bayesian Online Change-Point Detection (BOCPD)

- **Sources**: 
  - Page (1954), *Continuous Inspection Schemes*, Biometrika. (Original CUSUM.)
  - Adams & MacKay (2007), [BOCPD arxiv:0710.3742](https://arxiv.org/abs/0710.3742). Standard reference.
  - Altamirano et al. (2023), [Robust and Scalable BOCPD, ICML](https://proceedings.mlr.press/v202/altamirano23a/altamirano23a.pdf).
  - Gundersen blog tutorial: [BOCPD walkthrough](https://gregorygundersen.com/blog/2019/08/13/bocd/).
- **Core idea**: Detect a change in the generative parameters of a sequence in real time. CUSUM accumulates log-likelihood ratios and flags when the running sum exceeds a threshold. BOCPD computes the posterior over the run-length (time since last changepoint) via message-passing, with a hazard-rate prior λ controlling sensitivity.
- **CoT-CP application**: Within a long CoT trace, detect when reasoning *quality drops* (e.g., the model starts hallucinating or going in circles). The signal is the running token-confidence or per-step PRM. CUSUM/BOCPD gives a principled "step at which to abort or branch" trigger. Concretely: replace our per-step `lp_drawdown` heuristic threshold with BOCPD posterior-over-changepoint > threshold. This handles long-CoT models (R1-Distill, QwQ) where 60+ steps make a static threshold unreliable.
- **Connects to**: Per-step CP Approach A (online early-abort); Layer B step rejection trigger. Pilots H/J showed long-CoT models break our naive lp_min trigger — BOCPD might fix this.
- **Value**: **HIGH** for long-CoT models — this is exactly the problem we have.
- **Effort**: 3-5 days. BOCPD has open-source implementations ([dtolpin/bocd](https://github.com/dtolpin/bocd)).

## A4. Optimal stopping / Snell envelope

- **Source**: Chow, Robbins, Siegmund (1971), *Great Expectations: The Theory of Optimal Stopping*. Houghton Mifflin. (Or Karatzas-Shreve textbook chapter.)
- **Core idea**: Given a stochastic process and a reward function, compute the optimal stopping time via backward induction on the Snell envelope. The classical "secretary problem" 1/e ≈ 0.368 is the canonical example.
- **CoT-CP application**: Frame "decide when to stop adding self-consistency samples" as an optimal stopping problem. The reward is (current best answer accuracy estimate) − λ × (compute spent so far). Snell envelope gives the threshold to stop. Smarter than ESC's static window or Adaptive-Consistency's Bayesian threshold because λ can be learned.
- **Connects to**: Compute-Pareto frontier (Pilot D); Tier 4 competitors. Bridges Adaptive-Consistency and SPRT via DP.
- **Value**: Med — likely empirically equivalent to SPRT/e-process but better-justified for the **compute-cost-aware** setting.
- **Effort**: 1 week.

---

# B. Cognitive science & metacognition

## B1. Cognitive Load Theory & Worked Examples (Sweller)

- **Sources**:
  - Sweller, J. (2011), *Cognitive Load Theory*, Cognition in Education vol 55, ch. 2. [PDF](https://www.emrahakman.com/wp-content/uploads/2024/10/Cognitive-Load-Sweller-2011.pdf)
  - Recent LLM+CLT bridge: Mamede et al. (2025), [*Self-Harmonized CoT*](https://huggingface.co/papers/2409.04057), and Wang et al. (2025), [*United Minds or Isolated Agents? Exploring Coordination of LLMs under Cognitive Load Theory*](https://arxiv.org/html/2506.06843v1).
- **Core idea**: Working memory has finite capacity (Miller's 7±2). Learning is most effective when **intrinsic** load (problem complexity) is paired with low **extraneous** load (presentation overhead). Worked examples beat unguided problem-solving for novices because they offload procedure from working memory.
- **CoT-CP application**: 
  1. **Step segmentation by cognitive-load milestones**: instead of `\n\n`, segment when intrinsic complexity peaks (signal: high token-entropy or high attention-spread). EDU-PRM (2503.22233) does this empirically; CLT gives the theoretical justification + suggests other signals (working-memory indicators).
  2. **Worked-example prompting**: empirical literature shows that "show fully solved similar problem first, then ask new problem" beats raw CoT. This is *not* CP machinery, but it improves the *base* trace quality on which CP operates. Easy win for the experimental section.
  3. **Schema-acquisition**: PRM800K labels can be re-organized by *schema type* (substitution, factoring, induction, etc.) and per-schema CP thresholds calibrated. Closer to per-class conditional CP.
- **Connects to**: §6 Discussion (theoretical grounding for step segmentation). EDU-PRM positioning. New experiment idea.
- **Value**: Med — for paper-writing, gives the §1 motivation a non-CP backbone.
- **Effort**: 1-2 days writing + 3-day worked-example ablation if pursued.

## B2. Dual-Process Theory / Type 1 vs Type 2 (Kahneman; Stanovich)

- **Source**: Kahneman, D. (2011), *Thinking, Fast and Slow*. Stanovich-West dual-process taxonomy.
- **Core idea**: Two systems — fast/automatic/intuitive (Type 1) vs slow/deliberate/effortful (Type 2). The "rationality gap" comes from over-relying on Type 1 when Type 2 is required.
- **CoT-CP application**: Maps cleanly onto Layer A (CP filter = Type 2 deliberation) vs. raw greedy decode (Type 1). The framing also gives a clean motivation for **Type 2 trigger by Type 1 disagreement**: when Type 1 (fast greedy) and Type 2 (slow PRM-scored) disagree, that's the high-information signal. We could explicitly compare greedy-only vs CP-routed and report a "Type 1 / Type 2 tradeoff curve."
- **Connects to**: §1 Introduction framing; potentially §6 Discussion.
- **Value**: Low-Med — pure motivation/framing; no new method.
- **Effort**: <1 day writing.

## B3. Metacognition: Knowing What You Know (Fleming, Lau)

- **Source**: Fleming, S. & Lau, H. (2014), *How to measure metacognition*, Frontiers in Human Neuroscience.
- **Core idea**: Metacognitive accuracy = correlation between confidence and correctness. Distinguished from first-order accuracy. Fleming's m-ratio (meta-d'/d') is the standard metric.
- **CoT-CP application**: Our `entropy_mean` and `prov_match_latest_rate` ARE metacognitive signals. Computing m-ratio across our 11 models × 7 datasets would give a clean **"reasoning-model metacognition benchmark"** angle. Yoon+ 2025 (`2505.14489`, in our notes) hints at this; we can do it more rigorously. Could be an §5 figure: reasoning-model m-ratio vs base-model m-ratio.
- **Connects to**: All Layer A scores; §5 Experiments.
- **Value**: Med — clean meta-measurement that supports the broader pitch.
- **Effort**: 2-3 days.

---

# C. Neuroscience & predictive processing

## C1. Predictive Coding / Free-Energy Principle (Friston)

- **Sources**:
  - Friston, K. (2009), [*Predictive coding under the free-energy principle*](https://royalsocietypublishing.org/doi/abs/10.1098/rstb.2008.0300), Phil Trans Roy Soc B.
  - Sprevak, M. (2024), [*Predictive coding I: Introduction*](https://compass.onlinelibrary.wiley.com/doi/10.1111/phc3.12950), Philosophy Compass — modern overview.
  - Application: Mackinlay, [Predictive coding notes](https://danmackinlay.name/notebook/predictive_coding.html).
- **Core idea**: Brain minimizes a *variational free energy* = expected surprise = KL(internal model || sensory data). Hierarchical layers exchange forward "prediction errors" and backward "predictions". Surprise is the signal for updating beliefs.
- **CoT-CP application**: 
  1. **Token-level surprise as score**: $-\log p_\theta(\text{token})$ already in our `lp_min`. Predictive coding suggests *layer-wise* surprise (not just final-layer logit) — connects to the hidden-state-based scores (Wave 2 agent_B). PC predicts that *deep-layer surprise* is the most informative because it represents abstract prediction error rather than surface noise.
  2. **Active inference for tool use**: Friston's active-inference framework treats actions as belief updates. Maps onto "when to call a verifier" / "when to branch" via expected information gain. This is the principled extension of MetaTool / Adaptive-RAG.
- **Connects to**: Hidden-state scores (Wave 2 agent_B); Wave 3 Tier 2 self-prompting; future tool-use extension.
- **Value**: Med — strong theory-side framing, weaker as a new method on its own.
- **Effort**: 2-week theory pass; 1-week experimental implementation if we go for layer-wise surprise.

## C2. Bayesian Brain / Confidence as Posterior Variance (Pouget, Beck, Ma)

- **Source**: Ma, W. J., Beck, J., Latham, P., Pouget, A. (2006), *Bayesian inference with probabilistic population codes*, Nature Neuroscience.
- **Core idea**: Neurons collectively encode probability distributions, not point estimates. Confidence = inverse variance of the population's posterior. Strong empirical evidence for this in perception experiments.
- **CoT-CP application**: Token-level distributions over vocabulary ≈ posterior over next token. Per-step "posterior variance" = entropy or KL-from-uniform. We already use these via `entropy_mean`. The Bayesian-brain literature gives the *normative* argument for why these are the right signals, not just empirical luck. Useful for §1 framing.
- **Connects to**: `entropy_mean` (our universal winner score).
- **Value**: Low — framing only.
- **Effort**: <1 day.

---

# D. Decision theory & control

## D1. POMDP / Belief-state planning

- **Source**: Kaelbling, Littman, Cassandra (1998), *Planning and acting in partially observable stochastic domains*, AIJ.
- **Core idea**: When the true state is hidden, maintain a *belief state* (posterior over states) and plan in belief space. Policy is a function from belief to action.
- **CoT-CP application**: Treat the hidden "is the trace going to be correct" as the latent state; the belief is the running posterior given step scores so far. Layer B (step rejection) becomes "branch when belief P(correct | trace prefix) drops below threshold". Concrete vs our current `running_lp_min` heuristic. POMDP framework gives a principled place for the threshold via expected reward maximization.
- **Connects to**: Per-step CP (Layer A Approach A); Layer B trigger.
- **Value**: Med — formalizes our existing intuition.
- **Effort**: 1-2 weeks.

## D2. Model Predictive Control / Receding Horizon

- **Source**: Maciejowski, J. (2002), *Predictive Control with Constraints*. Pearson. Or Rawlings, Mayne, Diehl textbook.
- **Core idea**: Optimize over a finite future horizon, execute the first action, re-plan at next step. Robust to model error and disturbance because the horizon is repeatedly refreshed.
- **CoT-CP application**: At each step boundary, look ahead K steps via cheap branch-and-trial (similar to SCoT 2504.19095, but with a horizon parameter). Score the trial trajectory; if it predicts low confidence at horizon-end, intervene now. Direct generalization of trial-answer-confidence (DEER) but with a horizon, not just t+1. Gives a principled **horizon hyperparameter** that Snell+ 2024 implicitly tunes.
- **Connects to**: Layer B (step rejection branching); Tier 6 competitors (DEER, SCoT, ConfSpec).
- **Value**: Med-High — lifts speculative reasoning from heuristic to principled.
- **Effort**: 2-3 weeks; non-trivial experiments.

## D3. Risk-sensitive RL / CVaR / Conditional Tail Expectation

- **Source**: 
  - Rockafellar & Uryasev (2000), *Optimization of conditional value-at-risk*, Journal of Risk.
  - Tamar, Glassner, Mannor (2015), *Optimizing the CVaR via sampling*, AAAI.
- **Core idea**: Instead of expected reward, optimize tail risk: CVaR_α = E[reward | reward < α-quantile]. Bounded by linear program; Monte Carlo gradient.
- **CoT-CP application**: Our coverage statement is essentially a tail-risk constraint: P(answer wrong) ≤ α. Maps cleanly to CVaR formulations from finance. Gives an alternative training objective: train the model to *minimize expected wrong-answer rate at α=0.1 tail*, not raw accuracy. Could be a v2 angle: CoT-CP as inference-time wrapper, CoT-CVaR as training-time companion.
- **Connects to**: Theorem 1 (CRC = expected loss control); §6 discussion.
- **Value**: Med — natural training-time companion to our inference-time CP.
- **Effort**: Out of scope for v1 paper. v2 or follow-up.

## D4. Bayesian model averaging vs. majority vote (Hoeting et al.)

- **Source**: Hoeting, Madigan, Raftery, Volinsky (1999), *Bayesian Model Averaging: A Tutorial*, Statistical Science.
- **Core idea**: When uncertain among K models, average predictions weighted by posterior probability. Provably better than picking single best model under squared-error or log-loss.
- **CoT-CP application**: Replace majority-vote in self-consistency with **likelihood-weighted average over sampled traces**. The weight = LM's likelihood of the trace (already free). Soft-SC (2402.13212) does this; we can revisit with calibrated weights via CP. Also: BMA over the K resampled alternatives in our K=4 step-rejection. Currently we vote; BMA would weight by trace-conditional likelihood.
- **Connects to**: Our `tiebreak_lex` score; Layer B K=4 majority.
- **Value**: Low-Med — incremental ablation.
- **Effort**: 2-3 days.

---

# E. Information theory & rate-distortion

## E1. Rate-distortion theory / cost-vs-distortion frontier

- **Source**: Cover & Thomas, *Elements of Information Theory*, ch. 10.
- **Core idea**: Given a distortion measure d(x, x̂), the rate-distortion function R(D) is the minimum bits/sample to encode source X with average distortion ≤ D. Convex; lower envelope of all achievable (R, D) pairs.
- **CoT-CP application**: Our compute-Pareto frontier (Pilot D) is essentially a rate-distortion curve where rate = compute spent and distortion = (1 − accuracy). Framing as R(D) gives:
  1. The **achievable lower bound** on compute for a target accuracy (Shannon-style impossibility argument).
  2. **Operational test** for "is our score family Pareto-optimal?" — compute the LR+ envelope and compare to our empirical curve. Theorem 2 already does this informally; making it formal via R(D) ties to a ~70-year-old result.
- **Connects to**: Theorem 2 (Pareto / LR+ ordering); §5 compute-Pareto figure.
- **Value**: Med — clean theoretical anchor for §3.
- **Effort**: 1 week of theory writing.

## E2. Information bottleneck (Tishby, Pereira, Bialek)

- **Source**: Tishby, Pereira, Bialek (1999), *The Information Bottleneck Method*, Allerton.
- **Core idea**: Find compressed representation T of input X that maximizes I(T; Y) (relevance) for fixed I(T; X) (compression). Lagrangian L = I(T; X) − β I(T; Y).
- **CoT-CP application**: Treat the running step-score as a compressed representation T of the prefix. Maximize I(T; Y_correct) for fixed I(T; trace). This is essentially what `entropy_mean` empirically does; IB gives the formal justification + lets us derive optimal score functions analytically.
- **Connects to**: Wave 1+2+3 score brainstorms; Theorem 2.
- **Value**: Med — could justify a specific learned score family.
- **Effort**: 1-2 weeks of theory + 1 week of empirical IB-score derivation.

## E3. Maximum mutual information / contrastive PMI

- **Source**: Cover & Thomas ch. 8; modern: van den Oord et al. (2018), *Representation Learning with Contrastive Predictive Coding (CPC)*.
- **Core idea**: $\text{PMI}(x, y) = \log p(y|x) - \log p(y)$. Captures how much knowing x changes the prediction of y. Contrastive losses estimate this directly.
- **CoT-CP application**: 
  1. Step-level **PMI score**: $s_t = \log p(\text{answer} \mid \text{step}_t, \text{prefix}) - \log p(\text{answer} \mid \text{prefix})$. Captures the *informational contribution* of step t. Closely related to CG-VL's PMI grounding score for VLMs.
  2. **Score evaluation**: a score function's quality = its PMI with correctness. This unifies our heterogeneous scores (lp, entropy, PRM, vote-share) under one optimality measure.
- **Connects to**: All Layer A scores; potential new score family.
- **Value**: High — both as evaluation metric AND as new score.
- **Effort**: 1 week experimental.

---

# F. Search & optimization

## F1. Branch-and-bound with admissible heuristics

- **Source**: Land & Doig (1960), *An automatic method of solving discrete programming problems*, Econometrica. Nilsson (1980), *Principles of Artificial Intelligence*, ch. 2.
- **Core idea**: Search a solution space by branching, pruning subtrees whose lower bound exceeds the current best. Requires an *admissible* heuristic (never overestimates cost).
- **CoT-CP application**: For step-level branching (our Layer B), the existing K=4 majority is essentially BFS without pruning. With an *admissible PRM* (PRM_lo such that PRM_lo(step) ≤ true reward), we can prune branches whose PRM_lo + completion budget can't beat the current best trace. Gives sub-K branching with provable solution-quality bound. Very different from MCTS (UCT) which is stochastic.
- **Connects to**: Pilots C/K/L step-rejection (which all failed for vanilla K=4); Tier 5 PAVs.
- **Value**: Med — makes step-rejection a proper algorithmic procedure rather than heuristic resampling.
- **Effort**: 2-3 weeks.

## F2. AlphaZero / Monte-Carlo Tree Search

- **Source**: Silver et al. (2017/2018), *Mastering the Game of Go without Human Knowledge / A General RL Algorithm*. Coulom (2006) original MCTS.
- **Core idea**: MCTS-UCT balances exploration (UCB) with exploitation (Q-value); AlphaZero uses neural-net policy + value heads to guide selection.
- **CoT-CP application**: Already in the literature for math reasoning (AlphaMath, ReST-MCTS*, Tier 5 papers). Our innovation could be: replace the UCT exploration bonus with a **CP-calibrated quantile** — branch into a step alternative when its score is in the bottom α-quantile of the calibration distribution. Bridges MCTS and CP.
- **Connects to**: Layer B step-rejection; Tier 5 (AlphaMath, ReST-MCTS*).
- **Value**: Med — promising but expensive; v2.
- **Effort**: 4+ weeks.

## F3. Diverse Beam Search

- **Sources**:
  - Vijayakumar et al. (2018), [*Diverse Beam Search*](https://arxiv.org/abs/1610.02424).
  - Holtzman et al. (2020), [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751) (nucleus sampling).
- **Core idea**: Standard beam search produces near-duplicate beams. DBS adds a between-group dissimilarity penalty, giving more diverse top-K.
- **CoT-CP application**: For our K=4 alternatives in Layer B, currently we use temperature-0.7 sampling. DBS would force *semantic* diversity, which empirically beats temperature-only diversity (a known result). Can directly increase the recovery rate of K=4 majority. Cheap to implement.
- **Connects to**: Layer B K=4 majority; Pilots C/K/L (which had +1-2pp lifts because alternatives were too similar).
- **Value**: **HIGH** — directly attacks our biggest empirical bottleneck.
- **Effort**: 2-3 days experimental.

---

# G. Distributed systems & ensembling

## G1. Byzantine Fault Tolerance / Quorum systems

- **Source**: Lamport, Shostak, Pease (1982), *The Byzantine Generals Problem*, ACM TOPLAS. PBFT (Castro & Liskov 1999).
- **Core idea**: With f Byzantine (arbitrary-fault) replicas out of n total, consensus is achievable iff n ≥ 3f+1 (BFT threshold). Quorum-based protocols vote and reach agreement under partial faulty replicas.
- **CoT-CP application**: Self-consistency with N traces is essentially a Byzantine consensus protocol where some traces are faulty (wrong reasoning). Standard SC uses majority (f < n/2); BFT theory says we need 3f+1 if traces can adversarially disagree. Gives a principled lower bound on N. Also: weighted quorum (Naor & Wool) maps to confidence-weighted SC.
- **Connects to**: Self-consistency baseline; weighted-vote variants (Soft-SC).
- **Value**: Low-Med — interesting framing; not a new method.
- **Effort**: <1 day writing.

## G2. Stacking / Boosting / AdaBoost

- **Source**: Schapire & Freund (2012), *Boosting: Foundations and Algorithms*, MIT Press.
- **Core idea**: Combine weak learners by reweighting training points based on previous learner's errors. AdaBoost achieves exponential decrease in training error with provably bounded generalization.
- **CoT-CP application**: 
  1. **Boost over score families**: treat lp_min, prm_min, sc_top1 as weak verifiers. AdaBoost-weighted combination gives an empirically-better aggregate score (vs our current `tiebreak_lex` heuristic). The boosting weights come from per-prompt error correlation.
  2. **Iterative PRM training** over consecutive PRMs: each PRM specializes in errors the previous PRM missed.
- **Connects to**: Score-family combination; PRM training pipeline.
- **Value**: Med — concrete ablation idea for `tiebreak_lex`.
- **Effort**: 1 week.

---

# H. Numerical analysis & adaptive computation

## H1. Adaptive step-size in numerical ODE solvers (Runge-Kutta-Fehlberg, RKF45)

- **Source**: Hairer, Nørsett, Wanner (1993), *Solving Ordinary Differential Equations I*. Springer.
- **Core idea**: Compute two estimates (e.g., 4th-order and 5th-order RK), use difference as local error estimate, adapt step size accordingly. Ensures global error ≤ tolerance with minimum work.
- **CoT-CP application**: Within a CoT trace, "compute two estimates" maps to "two parallel sub-traces of next K tokens with different temperatures" or "two PRMs of different cost". Their disagreement = local error estimate. Adapt segmentation step size: shorter segments where models disagree (high local error), longer segments where they agree. This *automates* the step-segmentation question that EDU-PRM (2503.22233) addresses heuristically.
- **Connects to**: Step segmentation; per-step CP.
- **Value**: **HIGH** — direct mathematical analogy with a 60-year-old well-understood algorithm.
- **Effort**: 2 weeks.

## H2. Richardson extrapolation / convergence acceleration

- **Source**: Richardson (1911), *The approximate arithmetical solution by finite differences of physical problems involving differential equations*. Phil Trans Roy Soc A.
- **Core idea**: From two estimates at different resolutions, extrapolate to the limit. Reduces O(h^p) error to O(h^(p+1)).
- **CoT-CP application**: For self-consistency, sample at N=4 and N=8, extrapolate the running majority to N=∞. Could halve compute for the same accuracy. Theoretical justification: vote-share follows a CLT, Richardson extrapolation is exact for first-order error terms.
- **Connects to**: Self-consistency family.
- **Value**: Low-Med — clever but requires careful empirical validation.
- **Effort**: 1 week.

---

# I. Robust statistics & multiple testing

## I1. False Discovery Rate (Benjamini-Hochberg)

- **Source**: Benjamini & Hochberg (1995), *Controlling the False Discovery Rate*, JRSSB.
- **Core idea**: Instead of controlling family-wise error rate (Bonferroni: too conservative), control E[fraction of false discoveries among rejections]. BH procedure: sort p-values, reject the top-k with p_(i) ≤ k·α/m.
- **CoT-CP application**: For multi-step CRC, we currently propose Bonferroni (α/T_max per step). BH would be tighter and is appropriate when step-level decisions are exchangeable (which is what our exchangeability assumption already gives us). Bates-Angelopoulos-Lei-Romano (JRSSB 2023) extended BH to selective FDR for prediction sets — directly applicable.
- **Connects to**: Per-step CP Approach B (Bonferroni) → upgrade to BH; Theorem-revision item.
- **Value**: **HIGH** — strict improvement over our current Bonferroni.
- **Effort**: 1 week theory + 1 week experimental.

## I2. Robust statistics / Huber estimators

- **Source**: Huber (1981), *Robust Statistics*. Wiley. Modern: Maronna-Martin-Yohai (2019).
- **Core idea**: Replace mean (sensitive to outliers) with Huber-loss minimizer (linear in tail). M-estimators give finite influence function; min-max optimal under contamination model.
- **CoT-CP application**: Our scores aggregate per-step signals (`entropy_mean`, `lp_min`, etc.). Outlier steps (one badly-tokenized step in a trace) can dominate `lp_min`. Replacing min/mean with Huber-trimmed-mean would be more robust. Could reduce score variance and tighten CP quantiles.
- **Connects to**: Layer A trajectory aggregation; score family.
- **Value**: Low-Med.
- **Effort**: 2-3 days.

---

# J. Education & active learning

## J1. Active learning / query selection (Settles)

- **Source**: Settles, B. (2010), *Active Learning Literature Survey*. CS Tech Report, UW-Madison.
- **Core idea**: Choose next training example to maximize expected information gain (or minimize expected risk). EIG, query-by-committee, expected error reduction, etc.
- **CoT-CP application**: Calibration-set construction. Currently we use random sampling from PRM800K. Active learning could pick the **most informative** calibration points — ones where the conformity score is uncertain about its quantile location. Guaranteed sample complexity reduction; could cut calibration set from 12K problems to <2K.
- **Connects to**: Theorem 1 (calibration set construction); §3 setup.
- **Value**: Med — practical efficiency gain; doesn't change the core method.
- **Effort**: 1 week.

## J2. Self-explanation prompting (Chi et al.)

- **Source**: Chi, M. T. H. et al. (1989), *Self-explanations: How students study and use examples in learning to solve problems*, Cognitive Science.
- **Core idea**: Asking learners to *explain* worked steps to themselves yields better learning than just reading. Self-explanation effect is one of the most robust findings in education psychology.
- **CoT-CP application**: Prompting "explain step t" between t and t+1, then using the explanation's confidence as a step score. Closer to Wave 3 agent_SD self-prompting; we marked it untested due to cost. Self-explanation literature suggests it would work.
- **Connects to**: Wave 3 agent_SD step-self-probe scores; Layer B trigger candidates.
- **Value**: Med — testable score family addition.
- **Effort**: 3-5 days experimental.

---

# K. Causal inference

## K1. Counterfactual outcomes / treatment effects

- **Source**: Pearl (2009), *Causality*. Cambridge. Or Hernán & Robins (2020), *Causal Inference: What If*.
- **Core idea**: ITE = Y(treated) − Y(untreated). Cannot observe both; estimate via conditioning on confounders + identifiability assumptions.
- **CoT-CP application**: Our Layer B asks "would re-rolling step t make this trace correct?". This is a counterfactual question. Currently we just observe by sampling K=4 alts and majority-voting. Causal-inference framing gives:
  1. **ITE estimation** of "expected accuracy gain from re-rolling step t", learned offline.
  2. **Doubly-robust estimator** for the trigger decision, more efficient than naive sampling.
  3. **Front-door adjustment** when step t' (later) blocks the path between step t and the answer — could explain why naive K=4 fails (Pilots C/K/L null result).
- **Connects to**: All of Layer B; Pilots C/K/L analysis.
- **Value**: Med-High — could explain our null result and motivate a better method.
- **Effort**: 2-3 weeks; non-trivial.

---

# L. Software engineering & verification

## L1. Property-based testing / Hypothesis library

- **Source**: Claessen & Hughes (2000), *QuickCheck: A Lightweight Tool for Random Testing*, ICFP. Modern: Hypothesis library (MacIver).
- **Core idea**: Specify properties (invariants), automatically generate random inputs to find counter-examples. Shrink failing examples to minimal cases.
- **CoT-CP application**: For our coverage validation, use property-based testing on the CP procedure: generate synthetic (cal, test) pairs satisfying exchangeability, verify empirical coverage matches target. Catches edge cases (small n_+, ties in score, etc.) that fixed test data misses. Already a best practice in CP libraries.
- **Connects to**: §3 / appendix experimental validation.
- **Value**: Low (engineering hygiene).
- **Effort**: 1 day.

## L2. Symbolic execution / abstract interpretation

- **Source**: King (1976), *Symbolic execution and program testing*. Cousot & Cousot (1977), *Abstract interpretation*.
- **Core idea**: Run a program with symbolic values; track constraint sets symbolically. Abstract interpretation: run on lattice of abstract values; sound but incomplete.
- **CoT-CP application**: For math reasoning, "symbolic execution" of a CoT trace = sympy-based step-by-step verification of the algebraic claims. We tried this (Pilot, Wave 2 agent_C — 4% extraction rate, marked failed). Abstract interpretation suggests an *intermediate* layer: instead of full symbolic exec, abstract each step into a small finite domain (sign, magnitude, type) and check for type errors.
- **Connects to**: Pilots; Wave 2 agent_C `verifier_step_pass`; Wave 3 agent_SE `arith_violations` (which already does light abstract interp via `safe_eval`).
- **Value**: Med — `arith_violations` is already in our zoo. Extending to types/signs could push it further.
- **Effort**: 2 weeks.

---

# Top 8 actionable picks (rank-ordered for v1 paper or near-term v2)

1. **A1. E-processes / anytime-valid inference** (HIGH value, 2 weeks). Single biggest mathematical upgrade; replaces Adaptive-Consistency / ESC competitor turf with formal frequentist guarantee.
2. **F3. Diverse Beam Search** (HIGH value, 2-3 days). Directly attacks Pilot C/K/L bottleneck of insufficient alternative diversity.
3. **A3. BOCPD / CUSUM** (HIGH value, 3-5 days). Specifically for long-CoT models where our static threshold fails.
4. **H1. Adaptive step-size from numerical ODEs** (HIGH value, 2 weeks). Principled segmentation of CoT into steps via local-error estimation.
5. **I1. Benjamini-Hochberg FDR for multi-step** (HIGH value, 2 weeks). Strict tightening of our Bonferroni Approach B.
6. **E3. PMI as score function** (HIGH value, 1 week). Unified evaluation of all candidate scores under one optimality measure.
7. **B1. Worked-example prompting** (Med-High value, 3-5 days). Improves base trace quality before CP filtering.
8. **B3. Metacognition m-ratio** (Med value, 2-3 days). Cheap §5 figure; supports broader pitch.

---

# Items that motivate v2 / future work (not for v1)

- D1 POMDP belief-state planning (formalizes Layer B; expensive)
- D2 Model Predictive Control (lifts speculative reasoning; expensive)
- F1 Branch-and-bound with admissible PRM (sound algorithmic Layer B)
- F2 MCTS with CP-calibrated UCB (deep but slow)
- D3 CVaR / risk-sensitive RL training (training-side companion)
- K1 Causal counterfactual estimation for step rejection (could explain our null result)

---

# How to fold this into the existing project

| Document | Update |
|---|---|
| `theorems/PAPER_OUTLINE.md` | Add §6 "future work" subsection citing A1, D1, D2, F2 |
| `theorems/THEOREM_REVIEW.md` | Add I1 (BH instead of Bonferroni) as a Theorem-revision item |
| `score_ideation/from_agents/*.md` | Wave 4 brainstorm could include E3 PMI, F3 DBS, A1 e-process, H1 adaptive segmentation as new score/method candidates |
| `experiments/src/` | Implement F3 DBS first (cheapest), then E3 PMI scoring, then A3 BOCPD trigger |
| `paper/related_work` (when written) | B1, B2, B3 give framing paragraphs for §1 motivation |

---

# Sources cited in this document

- [Ramdas et al. (2023), Game-Theoretic Statistics and Safe Anytime-Valid Inference](https://projecteuclid.org/journals/statistical-science/volume-38/issue-4/Game-Theoretic-Statistics-and-Safe-Anytime-Valid-Inference/10.1214/23-STS894.pdf)
- [Adams & MacKay (2007), Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742)
- [Altamirano et al. (2023), Robust and Scalable BOCPD, ICML](https://proceedings.mlr.press/v202/altamirano23a/altamirano23a.pdf)
- [Friston (2009), Predictive coding under the free-energy principle](https://royalsocietypublishing.org/doi/abs/10.1098/rstb.2008.0300)
- [Sprevak (2024), Predictive coding I: Introduction, Philosophy Compass](https://compass.onlinelibrary.wiley.com/doi/10.1111/phc3.12950)
- [Sweller (2011), Cognitive Load Theory PDF](https://www.emrahakman.com/wp-content/uploads/2024/10/Cognitive-Load-Sweller-2011.pdf)
- [Wang et al. (2025), United Minds or Isolated Agents? Cognitive Load Theory and LLMs](https://arxiv.org/html/2506.06843v1)
- [Vijayakumar et al. (2018), Diverse Beam Search](https://arxiv.org/abs/1610.02424)
- [Gundersen blog: BOCPD walkthrough](https://gregorygundersen.com/blog/2019/08/13/bocd/)

Foundational textbooks (no link required): Kahneman 2011, Pearl 2009, Cover & Thomas 2006, Sutton & Barto 2018, Schapire & Freund 2012, Hairer-Nørsett-Wanner 1993.

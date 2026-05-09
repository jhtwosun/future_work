# Positioning Matrix — CoT-CP vs prior work

> Cross-cutting comparison along the axes that matter for §2 (Related Work) and §5 (Experiments).
> Each row is a competitor or precedent; columns are the dimensions on which CoT-CP can position itself.

## Legend

- **Granularity**: at what unit the method operates (token / span / step / claim / sequence / prompt)
- **Score**: what signal is consumed (logprob / entropy / vote / PRM / hidden-state / self-eval)
- **Cal?**: does it provide a finite-sample distribution-free coverage guarantee?
- **OOD**: how it handles distribution shift (none / per-prompt threshold / weighted CP / domain-aware)
- **Where the threshold is set**: the moment a number gets fixed
- **CoT-CP wedge**: the *specific* differentiator we cite in §2

---

## A. CP-for-LLM precedents

| paper | granularity | score | calibrated? | OOD handling | threshold set | CoT-CP wedge |
|---|---|---|---|---|---|---|
| `quach2024` Conformal LM | sequence (set-of-generations) | LM self-eval | ✅ marginal coverage | none | offline split CP | step-level not sequence-level; greedy + step scores instead of sample-then-admit |
| `mohri2024` Conformal Factuality | atomic claim | self-eval prob of correctness | ✅ marginal coverage | none | offline split CP | claims are exchangeable in their setup; reasoning steps are NOT — we treat trajectory as single CP unit |
| `abbasi-yadkori2024` Conformal Abstention | sequence | semantic-entropy similarity | ✅ marginal | none | offline split CP | their score ≈ our `sc_top1` — we add `lp_min`, `prm_min` rungs and step-level extension |
| `cherian2024` Enhanced CP | claim | learned scorer (diff-through-CP) | ✅ conditional | none | offline + grad descent | conditional CP angle is orthogonal — we keep marginal in v1, cite as future work |
| `rubin-toles2025` Coherent Factuality | claim subgraph | deducibility graph + claim score | ✅ marginal on coherent set | none | offline split CP per subgraph | concurrent — we drop exchangeability with cheaper structural choice (trajectory aggregation) |
| `chen2025` CoVeR | token-cluster step | next-token surprise | ✅ PAC coverage on decoding trajectories | none | offline split CP | semantic-step (not token-cluster) granularity; selective accuracy on final answer (not trajectory coverage) |
| `hittesdorf2026` DCF | claim graph | trained differentiable scorer | ✅ during training | none | training-time | inference-time post-hoc; black-box-API compatible |
| `wang2024` ConU | sequence | self-consistency frequency | ✅ marginal | none | offline split CP | trajectory-aggregated, not whole-sequence atomic |
| `lin2025` DS-CP | prompt | embedding distance | ✅ marginal under weighted exch. | embedding-proximity weight | offline weighted CP | composable — apply DS-CP weights on top of our step-level scores |
| `wang2026` Beyond Surface | sequence | hidden-state probe | ✅ marginal | layer-wise reweighting | offline split CP | API-compatible (no hidden states required); we still gain from probe scores when available |

## B. Test-time scaling competitors (uncalibrated)

| paper | granularity | score | calibrated? | OOD handling | threshold set | CoT-CP wedge |
|---|---|---|---|---|---|---|
| `fu2025` DeepConf | span | mean token-confidence over window | ❌ heuristic η-percentile | per-prompt warm-up | online warm-up (N_init=16) | distribution-free P(correct) ≥ 1-α via CRC; cross-domain calibration on PRM800K |
| `yang2025` DEER | step | mean max-softmax over trial-answer | ❌ heuristic λ≈0.95 | none (eval-set sweep) | swept on eval | calibrated quantile $\hat q$ from PRM800K; finite-sample slack $1/(n_+ + 1)$ |
| `aggarwal2023` Adaptive-Consistency | prompt | Beta-Binomial posterior over majority | ❌ posterior over WHICH answer wins, not correctness | none | C_thresh ≈ 0.95 prior | step-level (not prompt-level); frequentist coverage on correctness, not which-answer-wins |
| `li2024` ESC | prompt window | sliding-window vote entropy | ❌ heuristic stop | per-(task,model) tuning | tune on probe samples | no per-task tuning; CRC quantile transfers under exchangeability |
| `snell2025` Scaling Optimal | prompt | oracle/PRM difficulty | ❌ oracle bins | difficulty bin lookup | val MATH per bin | no difficulty oracle; calibrated $\hat q$ replaces strategy lookup |
| `setlur2024` PAVs | step | advantage under prover policy | ❌ uncalibrated reward | none | RL training | wrap PAVs as our $s_t$, add CRC threshold for coverage |

## C. PRMs / step-level scorers (infrastructure)

| paper | role for CoT-CP |
|---|---|
| `lightman2023` PRM800K | calibration source (800K step labels / 75K solutions / 12K problems). PRM trace-score = product of per-step probs. |
| `wang2024mathshepherd` | auto-labeled alt to PRM800K. HE label $y^{HE}=\mathbb{1}[\exists j: a_j=a^*]$, SE label = mean. N=8 rollouts via LLemma-7B. |
| `setlur2024` PAVs | candidate score $s_t = A^\mu(s,a) = Q^\mu - V^\mu$ under separate prover. Math: if prover = base policy, gradient collapses to ORM (Thm 3.1). |
| `zheng2024` ProcessBench | evaluation. PRM OOD failure quantified: Math-Shepherd-PRM 47.9 → 23.8 F1 from GSM8K → Omni-MATH. Motivates step-level CP over answer-level. |
| `wang2023` Self-Consistency | candidate score (vote-share) and headline baseline. PaLM-540B GSM8K 56.5 → 74.4 (+17.9). |

## D. Recent step-level competitors

| paper | granularity | score | calibrated? | what to do |
|---|---|---|---|---|
| `ni2025` UHeads | step | <10M-param probe on hidden states | ❌ uncalibrated probe | **wrap with CRC** — composable, not competitive |
| `zhang2025` Entro-duction | step | output entropy + variance entropy | ❌ heuristic gates | **borrow variance-entropy** as one component of $s_t$ |
| `cao2025` EDU-PRM | step | predictive entropy as anchor | ❌ heuristic anchor | ablate segmentation: newline vs entropy anchors |
| `chen2025` VG-Search | beam/step/N | granularity parameter g | ❌ adaptive heuristic | **mandatory baseline; open-source code** |
| `liu2026` ConfSpec | step | draft confidence | ❌ "calibrated within competence" | sharpen distinction: distribution-free vs empirical-per-domain |

---

## E. The CoT-CP positioning paragraph (drop-in for §2 conclusion)

> Three trends converge in 2024–2026: (i) step-level scoring is empirically validated (PRM800K, Math-Shepherd, PAVs, UHeads, EDU-PRM), (ii) adaptive test-time compute via heuristic thresholds is widespread (DeepConf, DEER, ESC, Adaptive-Consistency, Snell+, VG-Search, ConfSpec), and (iii) CP-for-LLMs has matured but remains anchored at the *atomic-claim* (Mohri-Hashimoto, Cherian-Gibbs-Candès), *whole-sequence* (Quach, ConU, Abbasi-Yadkori), or *token-cluster* (CoVeR) level. CoVeR (2025) and Differentiable Conformal Training (2026 preprint) are the only papers that target step-level CP, and both differ from our setting: CoVeR works at token-cluster granularity with PAC coverage on decoding trajectories, while Differentiable Conformal Training requires training-time gradients through the CP procedure. CoT-CP plants the flag on **inference-time, post-hoc, semantic-step CP with a final-answer selective-accuracy guarantee**, with the empirical-PMF weighted-CP correction (Theorem 3) for discrete scores under shift — a methodological gap unfilled by all 30 papers in this survey.

## F. Required citations by section

| Section | Citations (in order of importance) |
|---|---|
| §1 Introduction | `fu2025deepconf`, `yang2025deer`, `chen2025cover`, `mohri2024-conformal-factuality`, `rubin-toles2025-coherent-factuality`, `angelopoulos2024crc`, `lightman2023verify` |
| §2 Background — CP foundations | `angelopoulos2023gentle`, `angelopoulos2024crc`, `tibshirani2019weighted`, `barber2023beyond`, `angelopoulos2025ltt` |
| §2 Background — CP for LLMs | `quach2024-conformal-language-modeling`, `mohri2024-conformal-factuality`, `cherian2024-enhanced-llm-validity`, `abbasi-yadkori2024-conformal-abstention`, `rubin-toles2025-coherent-factuality`, `chen2025cover`, `hittesdorf2026dcf`, `wang2024conu`, `lin2025dscp`, `wang2026beyondsurface` |
| §2 Background — test-time scaling | `wang2023selfconsistency`, `aggarwal2023adaptive`, `li2024esc`, `snell2025scaling`, `fu2025deepconf`, `yang2025deer` |
| §2 Background — step verification | `lightman2023verify`, `wang2024mathshepherd`, `setlur2024rewarding`, `zheng2024processbench`, `ni2025uheads`, `zhang2025entroduction`, `cao2025edu`, `chen2025vgsearch`, `liu2026confspec` |
| §3 Setup (Theorems) | `angelopoulos2024crc` (T1), `tibshirani2019weighted` (T3), `lightman2023verify` (data), `mohri2024-conformal-factuality` (claim-vs-trajectory contrast) |
| §5 Experiments | All baselines from `mandatory baselines` list in INDEX.md |
| §6 Discussion / Limitations | `cherian2024-enhanced-llm-validity` (conditional CP future work), `rubin-toles2025-coherent-factuality` (when stronger structure is worth the cost), `liu2026confspec` (calibrated-within-competence vs distribution-free) |

# Agent SR-C: Multi-Trace Ensemble + Selection Methods for CoT-CP

**Mission.** Reject a step OR an entire trace based on multi-trace evidence to improve calibrated selective accuracy. Budget: up to ~8× total forward-pass cost (matching SC@8). Bigger budget OK if methods justify it. The goal is not just majority voting on final answers (`sc_top1` already does that) but to surgically combine, align, and prune intermediate reasoning across parallel traces. We exploit the fact that CoT-CP already gives us per-step scores (`entropy_mean`, `lp_min`, `prov_match_latest_rate`, `lp_step_min`, `step_self_consistency`, etc.), so we can do principled cross-trace surgery rather than pure black-box voting.

The seven methods below are ordered roughly by ambition: (1)–(2) are cheap selection variants; (3)–(5) are alignment / surgery; (6)–(7) push towards adaptive and learned aggregation.

---

## Method 1: DTW-Aligned Step Convergence (DASC)

**1. Name.** `dtw_step_convergence` (DASC).

**2. Generation strategy.** K=3 traces. One greedy (T=0.0), two stochastic (T=0.7, T=0.7 different seeds). Same prompt, no paraphrase.

**3. Alignment / aggregation.** Use Dynamic Time Warping over a step-embedding (we already have step text; encode with a small sentence encoder, or fall back to token n-gram Jaccard). Build a 3-way alignment: each step in the canonical trace (the greedy one) is mapped to ≤1 step in each of the other two. For each canonical step position `i`, define `agree_i ∈ {0,1,2}` = number of stochastic traces whose aligned step has cosine ≥ τ (e.g., 0.75) with the canonical step.

**4. Rejection criterion.**
- *Step-level reject:* if `agree_i == 0`, drop that step from the kept-prefix (the trace truncates here for CP scoring).
- *Trace-level reject:* if mean(`agree_i`) over the trace < 0.5, reject the whole answer (abstain).
- The *quantile of mean alignment* is what's calibrated by CP.

**5. Cost.** 3 forward passes for traces, plus encoder calls for ~30 steps × 3 traces = negligible. Total ~3×.

**6. Hypothesis.** Steps that disagree across temperatures are "load-bearing for randomness" — they're either lookups the model is unsure about, or genuinely ambiguous. Steps that converge across high-T samples are highly likely to be correct, even if final answers differ. Aligning at the step level gives us many more voting events than `sc_top1`'s single final-answer vote, which should sharpen CP calibration.

**7. Pseudocode.**
```python
def dasc(question, model, encoder, tau=0.75):
    traces = [model.generate(question, T=0.0)]
    traces += [model.generate(question, T=0.7, seed=s) for s in (1, 2)]
    canon = traces[0]
    embs = [[encoder(s.text) for s in t.steps] for t in traces]
    # DTW align traces[1], traces[2] to canon (cost = 1 - cos)
    paths = [dtw(embs[0], embs[k]) for k in (1, 2)]
    agree = []
    for i, _ in enumerate(canon.steps):
        a = 0
        for k, path in enumerate(paths):
            j = path_map(path, i)            # canonical step i -> step j in trace k
            if j is not None and cos(embs[0][i], embs[k+1][j]) >= tau:
                a += 1
        agree.append(a)
    kept_prefix = take_while(canon.steps, lambda i: agree[i] >= 1)
    score = sum(agree) / (2 * len(canon.steps))     # in [0,1]
    final = canon.answer if score > 0 else ABSTAIN  # CP threshold replaces 0
    return final, score, kept_prefix
```

**8. Failure modes.** (a) Two stochastic traces can both be confidently wrong in the same way (mode collapse) — agreement ≠ correctness. (b) DTW can mis-align when traces have wildly different lengths; mitigate with band constraints. (c) Encoder may treat semantically equivalent steps as distant (e.g., "5+3=8" vs "8 = 5 plus 3").

---

## Method 2: Pareto-Front Trace Selection (PFTS)

**1. Name.** `pareto_front_traces`.

**2. Generation strategy.** K=3 traces at T ∈ {0.0, 0.5, 1.0}. The temperature ladder is the key — each trace explores a different point on the bias/variance frontier.

**3. Alignment / aggregation.** Per-trace, compute *m* trajectory-level scores (from our existing CoT-CP catalog): `entropy_mean`, `lp_min`, `prov_match_latest_rate`, `step_self_consistency_mean`, `length_normalized_lp`. Then compute the Pareto front in this *m*-dimensional score space (where higher-is-better is normalized). Define `pareto_wins(t)` = number of metrics on which `t` is the unique winner among the K traces. Pick `argmax pareto_wins`. Ties broken by the geometric mean of normalized scores.

**4. Rejection criterion.**
- *Trace-level only.* Reject if no trace is on the Pareto front in ≥ ⌈m/2⌉ metrics, i.e., scores are dominated/contradictory across the K traces. This is the "metrics disagree about which trace is best" signal — a strong abstention cue.
- The selection score `pareto_wins(t*)` is what CP calibrates.

**5. Cost.** 3 forward passes. Scoring is cheap (we already compute these per-token logprobs).

**6. Hypothesis.** When multiple independent quality metrics agree on the same trace, that trace is much more likely to be correct than `sc_top1`'s answer-vote winner — because answer-vote can be hijacked by a popular wrong path. The Pareto front formalizes "score consensus" without learning a weighting.

**7. Pseudocode.**
```python
def pfts(question, model, metrics):
    traces = [model.generate(question, T=t) for t in (0.0, 0.5, 1.0)]
    M = np.array([[m(t) for m in metrics] for t in traces])  # K x m
    M = (M - M.min(0)) / (M.ptp(0) + 1e-9)                   # normalize
    pareto = []
    for i, row in enumerate(M):
        dominated = any(
            all(M[j] >= row) and any(M[j] > row)
            for j in range(len(traces)) if j != i
        )
        if not dominated:
            pareto.append(i)
    if not pareto:
        return ABSTAIN, 0.0
    # pick trace winning the most individual metrics
    wins = {i: sum(M[i, k] == M[:, k].max() for k in range(M.shape[1])) for i in pareto}
    best = max(wins, key=wins.get)
    score = wins[best] / M.shape[1]
    if score < 0.5:
        return ABSTAIN, score
    return traces[best].answer, score
```

**8. Failure modes.** (a) Metrics may be highly correlated (e.g., `entropy_mean` and `lp_min`) — Pareto becomes trivial. Counter: pre-decorrelate or pick a curated diverse subset. (b) For very easy questions, all three traces tie on all metrics → fall back to T=0. (c) High-T trace might dominate due to stylistic verbosity inflating logprob averages — use length-normalized variants.

---

## Method 3: Last-Common-Step Truncation (LCS-T)

**1. Name.** `consensus_prefix_truncation`.

**2. Generation strategy.** K=2 traces, both at T=0.5 with different seeds. Cheap.

**3. Alignment / aggregation.** Find the latest step index `i*` such that traces A and B share an *equivalent* step (cosine ≥ τ on step embedding, OR exact n-gram match on the numeric content for math). Take the prefix `A[0..i*]` (or B's, doesn't matter), then run a *third* short completion conditioned on that prefix (greedy, T=0) to produce a clean tail and final answer.

**4. Rejection criterion.**
- *Step-level:* steps after `i*` in either trace are implicitly rejected (replaced by the third completion).
- *Trace-level abstain:* if `i* == -1` (no step ever agreed) or `i* == 0` (only the question echo agrees), abstain.
- CP score: `(i* + 1) / max(len(A), len(B))` — fraction of the trace that's stable across samples.

**5. Cost.** 2 + ~0.5 (short tail) ≈ 2.5×.

**6. Hypothesis.** The "consensus prefix" is a high-confidence anchor — both stochastic traces independently arrived there, so it's probably right. By restarting from that anchor with greedy decoding, we avoid the divergent (and potentially wrong) tails of the original sampled traces. This is essentially *speculative-decoding-style anchoring* applied to reasoning.

**7. Pseudocode.**
```python
def lcs_truncate(question, model, encoder, tau=0.78):
    A = model.generate(question, T=0.5, seed=1)
    B = model.generate(question, T=0.5, seed=2)
    embA = [encoder(s.text) for s in A.steps]
    embB = [encoder(s.text) for s in B.steps]
    i_star = -1
    for i in range(min(len(A.steps), len(B.steps))):
        # latest position where some j <= i matches step i in B
        if any(cos(embA[i], embB[j]) >= tau for j in range(i + 1)):
            i_star = i
    if i_star <= 0:
        return ABSTAIN, 0.0
    prefix_text = "\n".join(s.text for s in A.steps[:i_star + 1])
    tail = model.generate(question + "\n" + prefix_text, T=0.0)
    score = (i_star + 1) / max(len(A.steps), len(B.steps))
    return tail.answer, score
```

**8. Failure modes.** (a) Two traces can agree on an early-but-wrong step (e.g., wrong formula), then diverge correctly later — we'd anchor on the wrong thing. (b) Order matters: "step 3 in A" matching "step 5 in B" is fine for some problems, fails for sequential ones. (c) Greedy tail can still drift — consider re-using one of the original tails when its score is high.

---

## Method 4: Cross-Trace Step Grafting (XGRAFT)

**1. Name.** `cross_trace_step_grafting`.

**2. Generation strategy.** K=3 base traces (T ∈ {0.2, 0.7, 0.7}). For each step position, we have multiple candidates.

**3. Alignment / aggregation.** Align all 3 traces with DTW (as in DASC). For each canonical step position `i`, gather the candidate steps `{c_i^A, c_i^B, c_i^C}`. Score each candidate with our per-step scores (`lp_step_min`, `entropy_step`, `prov_match_step`). The *graft* is: at each position, pick the highest-scoring candidate, and emit a stitched trace `G = [argmax c_i for i in 1..N]`. Then run a *graft-validation pass*: feed `G` back into the model with greedy decoding for a 1-step continuation at each grafted boundary to check the model assigns it high logprob; if low, replace that step with the original canonical one.

**4. Rejection criterion.**
- *Step-level:* if even the best of `{c_i^A, c_i^B, c_i^C}` has score below a per-step CP threshold, replace with `[ABSTAIN]` token and let the model regenerate that single step conditioned on neighbors.
- *Trace-level:* if grafted trace's mean step score is below trace-level CP threshold, abstain entirely.
- CP score: mean of per-step scores along the grafted trace.

**5. Cost.** 3 base + N validation forward passes (where N ≈ # graft boundaries, typically 2–5) ≈ 4–5×.

**6. Hypothesis.** Different temperatures expose different parts of the reasoning space. A T=0.7 trace might nail a creative step the greedy trace fluffed, while greedy might dominate on arithmetic. Surgical grafting picks the best step from each, similar to how MoE picks experts per token — but here the "experts" are temperatures. The validation pass guards against Frankenstein traces that locally look good but don't compose.

**7. Pseudocode.**
```python
def xgraft(question, model, scorers, encoder):
    traces = [model.generate(question, T=t, seed=s)
              for t, s in [(0.2, 0), (0.7, 1), (0.7, 2)]]
    canon = traces[0]
    paths = [dtw(emb(canon), emb(t)) for t in traces[1:]]
    grafted = []
    boundaries = []
    for i, _ in enumerate(canon.steps):
        candidates = [(canon.steps[i], 0)]
        for k, path in enumerate(paths):
            j = path_map(path, i)
            if j is not None:
                candidates.append((traces[k+1].steps[j], k+1))
        best, src = max(candidates, key=lambda c: step_score(c[0], scorers))
        grafted.append(best)
        if src != 0:
            boundaries.append(i)
    # validate: check model logprob at each boundary
    for i in boundaries:
        ctx = stitch(grafted[:i])
        lp = model.score_step(ctx, grafted[i].text)
        if lp < LP_FLOOR:
            grafted[i] = canon.steps[i]   # revert
    score = mean(step_score(s, scorers) for s in grafted)
    if score < CP_TAU:
        return ABSTAIN, score
    final = model.generate(question + stitch(grafted), T=0.0)
    return final.answer, score
```

**8. Failure modes.** (a) Stitching two semantically inconsistent steps (e.g., one assumes a variable means X, the other Y) — validation catches some but not all. (b) Step boundaries don't align cleanly: trace A's "step 3" is two of trace B's steps. (c) Per-step scorers favor short, high-probability steps over correct-but-uncertain ones (e.g., a hard arithmetic step legitimately has lower lp). Calibrate scores by step type if possible.

---

## Method 5: Adaptive-K Confidence Cascade (AKCC)

**1. Name.** `adaptive_k_cascade`.

**2. Generation strategy.** Cascade:
- Stage 1: K=1, greedy.
- Stage 2 (if needed): +1 trace at T=0.5.
- Stage 3 (if needed): +2 more traces at T=0.7.
Stop early if confidence is high. Total K ∈ {1, 2, 4}.

**3. Alignment / aggregation.**
- Stage 1: accept if `traj_score(T1) ≥ τ_high`.
- Stage 2: accept if `agree(T1, T2)` and `min_score ≥ τ_mid`. The agree() check is on final answer + step-cosine ≥ 0.7 over ≥ 60% of aligned steps.
- Stage 3: weighted majority vote on final answer with per-trace traj-scores as weights, gated by the *spread* of traj-scores.

**4. Rejection criterion.**
- *Trace-level:* abstain if at Stage 3 the weighted-vote winner has weight < 0.5 of total, or no answer received ≥2 of 4 votes.
- *Per-step* not used here (this method is trace-level).
- CP score: stage at which we accepted (1=most confident, 3=least), combined with the within-stage consensus margin.

**5. Cost.** Expected cost depends on dataset difficulty. Empirically: 1× on easy, 2× on medium, 4× on hard. Average likely 2–3× total — much cheaper than fixed K=8.

**6. Hypothesis.** Most questions don't need 8 traces. By scaling test-time compute on demand using our existing trajectory CP scores as the early-stopping signal, we get sc-style accuracy at a fraction of the cost. The CP guarantee transfers because we calibrate the *stage-conditional* scores separately and use hierarchical CP.

**7. Pseudocode.**
```python
def akcc(question, model, scorer, tau_high, tau_mid):
    t1 = model.generate(question, T=0.0)
    s1 = scorer(t1)
    if s1 >= tau_high:
        return t1.answer, ("stage1", s1)
    t2 = model.generate(question, T=0.5, seed=1)
    s2 = scorer(t2)
    if t1.answer == t2.answer and step_overlap(t1, t2) >= 0.6 and min(s1, s2) >= tau_mid:
        return t1.answer, ("stage2", min(s1, s2))
    t3 = model.generate(question, T=0.7, seed=2)
    t4 = model.generate(question, T=0.7, seed=3)
    pool = [(t1, s1), (t2, s2), (t3, scorer(t3)), (t4, scorer(t4))]
    votes = defaultdict(float)
    for t, s in pool:
        votes[t.answer] += s
    winner, w = max(votes.items(), key=lambda x: x[1])
    margin = w / sum(votes.values())
    if margin < 0.5:
        return ABSTAIN, ("stage3", margin)
    return winner, ("stage3", margin)
```

**8. Failure modes.** (a) `tau_high` mis-calibration → stage 1 over-accepts confident-but-wrong traces. (b) Hard-case overconcentration: most budget consumed by hardest 20% of questions. (c) Stage 1 is *correlated* with stages 2/3 since same prompt; CP must explicitly account for this when bounding selective risk. (d) When the model has a confidently wrong prior (e.g., systematic misreading of the question), all K agree and we cascade-stop on Stage 2.

---

## Method 6: Trajectory-Weighted Step Voting (TWSV)

**1. Name.** `weighted_step_vote`.

**2. Generation strategy.** K=4 traces: T=0.0, T=0.4, T=0.7, T=1.0. Spread the temperature ladder.

**3. Alignment / aggregation.** Each trace `t_k` has a trajectory-CP score `w_k = traj_score(t_k)` (cleanly normalized to a softmax over the K traces). Align all 4 traces with DTW to a "skeleton" formed by the union of step-positions. At each skeleton position, do *clustered weighted voting*: cluster the candidate steps by embedding similarity (threshold 0.75), then sum weights within each cluster. Pick the highest-weight cluster's centroid step (use the trace member closest to centroid).

The output is a re-stitched trace whose every step is the *cluster medoid* with highest weight. Final answer: regenerate from this stitched trace, or take majority among the 4 final-answer outputs weighted by `w_k`.

**4. Rejection criterion.**
- *Step-level:* if the winning cluster's weight share at a position < 0.4, mark step as low-confidence; truncate before the first such step.
- *Trace-level:* abstain if final-answer winning weight share < 0.5.
- CP score: minimum over step positions of winning-cluster weight share (a "weakest link" trajectory score).

**5. Cost.** 4× generation + scoring. ~4–5× total.

**6. Hypothesis.** This generalizes self-consistency from "vote on final answer" to "vote on each step," weighted by trajectory quality. The product is a synthetic *consensus trace* that should be more accurate than any single trace, because it's robust to per-step errors that any individual trace makes. Compared to `paraphrase_consensus`, this varies decoding rather than questions — complementary.

**7. Pseudocode.**
```python
def twsv(question, model, traj_scorer, encoder):
    traces = [model.generate(question, T=t, seed=i)
              for i, t in enumerate([0.0, 0.4, 0.7, 1.0])]
    raw_w = np.array([traj_scorer(t) for t in traces])
    w = softmax(raw_w / 0.5)
    # build skeleton via union-DTW; here simplified to per-position by canonical
    canon = traces[np.argmax(w)]
    paths = [dtw(emb(canon), emb(t)) for t in traces]
    stitched = []
    margins = []
    for i, _ in enumerate(canon.steps):
        cand = []
        for k, p in enumerate(paths):
            j = path_map(p, i)
            if j is not None:
                cand.append((traces[k].steps[j], w[k]))
        clusters = cluster_by_embedding(cand, encoder, tau=0.75)  # list[(weight, members)]
        clusters.sort(key=lambda c: -c[0])
        stitched.append(medoid(clusters[0][1]))
        margins.append(clusters[0][0])
        if clusters[0][0] < 0.4:
            break
    score = min(margins) if margins else 0.0
    if score < CP_TAU:
        return ABSTAIN, score
    final = model.generate(question + stitch(stitched), T=0.0)
    return final.answer, score
```

**8. Failure modes.** (a) Weight collapse: greedy trace dominates (high `w`) and votes look like K=1. Mitigate with temperature in softmax. (b) Cluster medoids may be syntactically clean but semantically mid — three close-but-noisy paraphrases beat one perfect step. (c) DTW union of 4 traces of varying length is fragile; consider star-alignment to canonical instead.

---

## Method 7: Disagreement-Driven Fork Selection (DDFS)

**1. Name.** `disagreement_fork_select`.

**2. Generation strategy.** K=4 traces from common prompt at T=0.7 (different seeds). Same temperature → disagreement is purely sampling-noise driven.

**3. Alignment / aggregation.**
1. Star-align all 4 traces to the longest one.
2. At each step position, compute *disagreement entropy* `H_i`: cluster the 4 candidate step embeddings, treat cluster sizes as a distribution, take its entropy.
3. Find the *first* step position `i*` where `H_i ≥ H_threshold` — this is the **fork point**. Steps before `i*` are stable across all 4 traces.
4. At the fork, partition traces into clusters. For each cluster, compute mean per-step confidence on its sub-trace from `i*` onward. Pick the **highest-confidence sub-trace** as the continuation.
5. Output: stable prefix `[0..i*-1]` + winning cluster's tail.

**4. Rejection criterion.**
- *Step-level:* steps within the losing clusters' tails are rejected.
- *Trace-level abstain:* (a) if `i* == 0` (forking from the very first step → traces have nothing in common). (b) If the winning cluster's confidence advantage over runner-up is < δ. (c) If winning cluster has only 1 member (singleton, no internal corroboration).
- CP score: `winning_cluster_size / K * mean_confidence(winning_tail)`. Calibrated on a held-out set.

**5. Cost.** 4× + tiny embedding cost. ~4×.

**6. Hypothesis.** Reasoning errors usually have a *first divergence point* — the model branches into a wrong path at some step. By detecting that fork explicitly via disagreement entropy and then *betting on the high-confidence branch*, we recover from sampling errors more aggressively than answer-vote SC, which throws away the structural information about *where* traces split. This is also more interpretable: we can show users the fork.

**7. Pseudocode.**
```python
def ddfs(question, model, encoder, conf_scorer, H_thresh=0.8, delta=0.1):
    traces = [model.generate(question, T=0.7, seed=s) for s in range(4)]
    longest = max(traces, key=lambda t: len(t.steps))
    paths = [dtw(emb(longest), emb(t)) for t in traces]

    H = []
    cands_at = []
    for i, _ in enumerate(longest.steps):
        cand_embs, cand_traces = [], []
        for k, p in enumerate(paths):
            j = path_map(p, i)
            if j is not None:
                cand_embs.append(emb(traces[k])[j])
                cand_traces.append(k)
        clusters = cluster(cand_embs, tau=0.75)              # list[set[int]]
        sizes = np.array([len(c) for c in clusters])
        H.append(entropy(sizes / sizes.sum()))
        cands_at.append((clusters, cand_traces))

    fork_positions = [i for i, h in enumerate(H) if h >= H_thresh]
    if not fork_positions:
        return longest.answer, 1.0                           # no fork — all agree
    i_star = fork_positions[0]
    if i_star == 0:
        return ABSTAIN, 0.0
    clusters, ctraces = cands_at[i_star]
    cluster_conf = []
    for c in clusters:
        member_traces = [ctraces[idx] for idx in c]
        if len(member_traces) < 2:
            cluster_conf.append((-1, c))
            continue
        confs = [conf_scorer(traces[m], start=i_star) for m in member_traces]
        cluster_conf.append((np.mean(confs) * len(c), c))
    cluster_conf.sort(reverse=True)
    if len(cluster_conf) > 1 and cluster_conf[0][0] - cluster_conf[1][0] < delta:
        return ABSTAIN, cluster_conf[0][0]
    winning_trace = traces[ctraces[next(iter(cluster_conf[0][1]))]]
    score = cluster_conf[0][0] / 4.0
    return winning_trace.answer, score
```

**8. Failure modes.** (a) "Stable prefix wrong" — all 4 traces start with the same wrong premise; fork-detection finds no fork to adjudicate. (b) "False fork" caused by surface-form variation only (different phrasing, same math) — embedding clustering should mitigate but may inflate `H_i`. (c) Confident-but-wrong cluster wins; per-step confidence isn't ground truth. (d) Sensitive to `H_threshold` — needs calibration per dataset.

---

## Cross-method comparison and recommended combos

| Method | Cost | Step-level reject? | Trace-level reject? | Best when... |
|---|---|---|---|---|
| DASC | 3× | Yes | Yes | Reasoning has clear sequential structure |
| PFTS | 3× | No | Yes | Several trajectory metrics already strong |
| LCS-T | 2.5× | Yes (truncate tails) | Yes | Models drift in late steps |
| XGRAFT | 4–5× | Yes | Yes | Per-step scorers are well-calibrated |
| AKCC | 1–4× avg | No | Yes | Heterogeneous difficulty; cost matters |
| TWSV | 4–5× | Yes | Yes | Have a strong single-trace traj-score |
| DDFS | 4× | Yes (post-fork) | Yes | Errors tend to come from one branching point |

**Recommended pairings.**
- **AKCC + DASC** as the inner consensus check at Stage 2/3 — cheap and synergistic.
- **TWSV + PFTS** ensemble: use Pareto front to pick the *weighting scheme* for TWSV.
- **DDFS + LCS-T**: DDFS finds the fork, LCS-T anchors a regenerated tail from the stable prefix.

**General guarantees.** All seven preserve CP marginal coverage as long as the score (whatever method-specific quantity is used) is calibrated on a held-out set with the same distribution of K-trace ensembles. Step-level rejection plus trace-level rejection compose into a *hierarchical* CP procedure: calibrate trace-level α first, then conditionally calibrate step-level β within accepted traces.

**Open questions for follow-up.**
- Can we share KV-cache across the K traces up to the prompt boundary to make 4× → effectively 1.x× on memory-bandwidth-bound serving?
- For grafting (XGRAFT), is per-step lp the right scorer, or should we use a small *step-level reranker* trained on a few hundred grafted-vs-original examples?
- Could DDFS be turned into a *recursive* method — find first fork, take winner, then search for a second fork inside the winner's sub-trace? This would be a poor man's MCTS over reasoning steps.

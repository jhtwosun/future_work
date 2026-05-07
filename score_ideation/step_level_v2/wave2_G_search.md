# Wave 2 — Agent G (search-based angle)

**Date**: 2026-05-05
**Author**: Wave 2 — Agent G (search-based angle)

## Context and unifying claim

Pilots C/K/L (worst-step + K-resample at T=0.7) all returned net-zero (+1–2pt recovered, equal lost). The diagnosis from `OBSERVATIONS_kr.md`: **K-resample from a shared prefix produces near-duplicate alternatives** — the policy has already committed to a basin. Score (lp_min, PRM-min, rewrite-cue) is not the bottleneck; the **intervention class** is. Search-based step-level interventions break out of the basin in a way K-resample structurally cannot.

The unifying claim: **K-resample explores siblings of one chosen branch; search explores cousins**. Search maintains *multiple* surviving prefixes and re-allocates compute toward the prefix the score function ranks highest. That re-allocation is the missing degree of freedom in C/K/L. Equivalently, K-resample draws from `p(·| prefix)` K times — same Markov-blanket. Search **changes the prefix**: beam keeps B different prefixes alive; MCTS reallocates compute; backjumping rewrites earlier prefixes. Effective hypothesis space is the *union of policy distributions over B different prefixes*, strictly larger.

---

## G1 — `step_beam_with_prm_value`

1. **Name**: `step_beam_with_prm_value`
2. **Algorithm**: Step-level beam search, width B=4, truncate at answer terminator.
3. **State / action**: Node = partial trajectory of complete reasoning steps (`\n\n` segmenter, as in Pilots 2–10). Action = emit one more step. Each round samples N=B candidates per surviving prefix (B → B² → prune to B).
4. **Scoring**: Cumulative `lp_mean` + λ · `PRM_min_along_prefix` (Qwen2.5-Math-PRM-7B). λ tuned on 100-problem holdout. AlphaZero-style PUCT but step-wise.
5. **Compute**: B² generations per step ≈ 16× greedy at B=4; PRM forward adds ~0.3×. Effective ~10–12× with KV prefix sharing. **Bounded**.
6. **Why beats K-resample**: K-resample re-rolls within one committed prefix. Beam keeps B sibling prefixes alive, so at step t+1 we sample from the **B best prefixes ranked by PRM**. Mode collapse on one prefix becomes irrelevant because PRM has already redirected compute toward a different prefix. Wider exploration; sharper exploitation.
7. **Implementation**: vLLM `LLM.generate(n=B)`. After each step boundary, batch B² extensions in one forward — PagedAttention shares KV across the common prefix automatically. PRM is a separate vLLM engine, batched over all B² candidates. Top-B selection is CPU-side. No model changes.
8. **CP preservation**: **Yes, with re-calibration.** Beam alters the generative distribution; calibrate q̂ on beam-generated traces. Pick winning trajectory, score with PRM-min, apply q̂. Caveat: do not use lp_min after beam (biased upward by construction); use PRM-min or SC-vote over top-B endpoints.

**vs SC@N at matched compute**: At ~16×, SC@16 hits 84.1% on MATH-500 (Pilot D). Beam should land between SC@8 (79.3%) and SC@16 if PRM signal is strong (Pilot A confirms PRM @2× → 70.7%, so signal is real).
**Non-decomposable formats**: **Works.** Returns a single winning trajectory — no `extract_answer` vote needed. Best fit for code/agentic.

---

## G2 — `mcts_step_rollouts`

1. **Name**: `mcts_step_rollouts`
2. **Algorithm**: MCTS with PUCT selection, step-level expansion, PRM-bootstrapped value, lp prior. Budget N=4 simulations.
3. **State / action**: Tree node = partial step-trajectory. Children = K=3 candidate next steps at T=0.8. Per simulation: select via PUCT, expand K children, rollout to terminal at T=0.6, score with PRM-min.
4. **Scoring**: Q(s,a) = mean PRM-min of rollouts through (s,a). Prior P(s,a) = softmax of step lp across siblings. UCB term c_puct · P · √(ΣN) / (1+N).
5. **Compute**: Per simulation ≈ K=3 expansions + 1 short rollout (~5 steps) ≈ 8 generations. N=4 sims → 32× raw, ~20× effective with KV reuse. **Bounded**.
6. **Why beats K-resample**: MCTS *concentrates compute on promising subtrees*. K-resample wastes 75% of K=4 generations on the worst branch and duplicates. After 2 simulations, MCTS has discovered which subtree has higher PRM-rollout-value and devotes sims 3–4 there. Exploration/exploitation knob is c_puct — principled, not ad-hoc T=0.7.
7. **Implementation**: Tree state = Python dict keyed by step-prefix hash. vLLM batched expansion at each simulation; PRM engine scores leaves. Set `enable_prefix_caching=True` — KV reuse becomes free. Backup is CPU-side, microseconds.
8. **CP preservation**: **Yes.** Output = highest-visit-count root child (recurse). Calibrate on MCTS-generated traces. MCTS policy is deterministic given seed+budget, so calibration matches deployment.

**vs SC@N**: SC@32 likely dominates raw accuracy (~85%+). MCTS wins on (a) **trajectory quality** — one coherent CoT, not a vote — and (b) **non-decomposable answers**. From rStar-Math/ToT literature, MCTS@32 sims approaches SC@16.
**Non-decomposable formats**: **Strong fit** — canonical MCTS use case. Code: replace PRM with unit-test pass-rate value. MCQ: MCQ-confidence-PRM. Framework agnostic.

---

## G3 — `best_first_priority_queue`

1. **Name**: `best_first_priority_queue`
2. **Algorithm**: Best-first / weighted A* with budget. Single global priority queue ordered by f = g + h, where g = cumulative neg-lp and h = PRM-estimated remaining cost (lightweight regression head, or proxy `−PRM_step × expected_remaining_steps`).
3. **State / action**: Step-level. Each pop expands K=2 children. Frontier shared across the problem batch.
4. **Scoring**: f = g + h. g monotone (honest neg-log-prob); h non-admissible heuristic → weighted A* (trades guarantee for speed).
5. **Compute**: Budget M=24 expansions × K=2 ≈ 48 generations ≈ 24× greedy. KV-cache reuse important — frontier nodes share long prefixes.
6. **Why beats K-resample**: Priority queue is **global** — compute flows to whichever partial trajectory currently looks best, *regardless of depth*. K-resample is local and fixed-depth: K branches at one step, then commit. Best-first spends 10 expansions on a deep promising branch and 0 on a dead one; K-resample is forced to spend K everywhere it triggers.
7. **Implementation**: Python `heapq` with (f_score, tiebreak, node). To get GPU utilization, pop M_pop nodes per round and expand them as a batch. Effectively "beam-with-replacement" search.
8. **CP preservation**: **Yes, with re-calibration.** Output = leaf with terminal answer minimizing f. Score with PRM-min. Calibrate q̂ on best-first traces.

**vs SC@N**: At 24×, SC@24 ≈ 82–84%. Best-first more compute-efficient when h is informative; worse when uninformative. Honest test: 100-problem holdout vs uniform h.
**Non-decomposable formats**: **Code — works** (h = unit-test prefix pass-rate). **MCQ — awkward** (no step-depth structure). Skip MCQ.

---

## G4 — `coupled_diverse_beams`

1. **Name**: `coupled_diverse_beams`
2. **Algorithm**: K=4 parallel beam searches with width B=2, **coupled by a step-level diversity constraint**: no two beams may share a step-hash at the same depth after divergence point d_0. Step-granularity diverse beam search (Vijayakumar+ 2018).
3. **State / action**: Step-level. Diversity enforced by Hamming-style constraint on step-hashes: when sampling step t for beam k, exclude any hash chosen by beams 1..k−1 at depth t.
4. **Scoring**: Within-beam: lp + λ·PRM. Across beams: SC-vote on K=4 final answers, ties broken by PRM-min. **Search-augmented self-consistency.**
5. **Compute**: K·B·N = 4·2·2 = 16 generations per step × ~8 steps = 128 raw. KV prefix sharing across beams that share early steps brings to ~30–40×. Hard cap at **32×**.
6. **Why beats K-resample**: This is the **direct fix** for C/K/L. K-resample failed because K=4 alternatives collapse to mode. Diversity-coupled beams **forbid** step-level repetition by construction — every beam is guaranteed to differ at every depth after d_0. Exploration is no longer probabilistic (T=0.7 hoping for variety) but **combinatorial** (each beam *must* pick a different step).
7. **Implementation**: Maintain `used_step_hashes[t]` = {hashes chosen by earlier beams}. Per beam, sample N=8 candidates, filter colliders, take top-2 by lp. Hash = SHA1 of normalized step text. vLLM batched across K·B = 8 prompts per round. Final aggregation = SC-vote on K=4 answers, PRM-min tie-break.
8. **CP preservation**: **Yes — and cleanest of the five.** K final answers = SC@4, so the established SC-top1-fraction conformity score (Pilot 9) applies directly. q̂ calibrated on coupled-beam SC scores.

**vs SC@N**: At 32×, SC@32 ≈ 85% on MATH-500. Coupled beams target 83–84% with **bonus**: each of 4 finalists is a *guaranteed-distinct* CoT — interpretability win and **better SC denominator** because votes are not redundant. Directly addresses Pilot 4's "Mean SC top-1 fraction 0.925, very consistent, 변별력 부족" diagnosis.
**Non-decomposable formats**: **MCQ — works** (vote across 4 distinct CoTs). **Code — works** (run unit tests on each, pick best pass rate). Diversity is on reasoning steps, not answer format.

---

## G5 — `adaptive_beam_with_backjump`

1. **Name**: `adaptive_beam_with_backjump`
2. **Algorithm**: Beam search with (a) **adaptive width** based on early-step lp-entropy, and (b) **backjumping**: when surviving beams have PRM-step < τ for ≥2 consecutive steps, restart from the latest "safe" step (PRM > τ) with widened beam.
3. **State / action**: Step-level. Width B_t ∈ {2, 4, 8} adapted at each depth. Maintain backtrack stack of (depth, beam-snapshot) at each high-PRM step.
4. **Scoring**: lp + λ·PRM as in G1. Backjump trigger: `max_beam(PRM-step) < τ` for ΔT consecutive steps. Width-up trigger: lp-entropy across siblings > θ at steps 1–2.
5. **Compute**: **Variable**. Best (low-entropy, no backjumps): B=2 → 4–6×. Worst (high entropy, 2 backjumps with B=8): ~32×. Hard cap 32×.
6. **Why beats K-resample**: Two compounded wins. (1) Adaptive width *spends compute where uncertainty is* — Pilot 7 step-reliability spread (0.80→1.00) shows MATH-500 has heterogeneous within-trace difficulty. K-resample uses fixed K everywhere, mostly wasted. (2) Backjumping fixes the **commit-too-early** failure: K-resample re-rolls only *one* step; if the bad step is 3 upstream, K-resample is too late. Backjumping rewinds.
7. **Implementation**: At each step boundary, compute lp-entropy across top-B candidates → if > θ, double B (cap 8). Snapshot beam to stack whenever max(PRM_step) > τ. On low-PRM streak, pop snapshot, restore vLLM KV-cache (or re-prefill from text — simpler, ~1× greedy cost per backjump). Effort: medium; KV snapshot/restore is the trickiest piece.
8. **CP preservation**: **Yes, with care.** Adaptive policy is data-dependent on the trajectory itself, so conformity score must be on the *final* (post-backjump, post-width-adapt) trajectory. Calibrate q̂ on adaptive-beam traces — CP only requires deployment-distribution match.

**vs SC@N**: This is where search **most plausibly beats SC at matched compute**, because compute is spent only where needed. Easy problems (most of GSM8K): B=2, no backjumps ≈ 4× ≈ matches SC@4 (94.0% Pilot 4). Hard MATH-500: B=8 with backjumps spends 32× and likely beats SC@16. Expected-compute-per-problem can be lower than SC@N at fixed accuracy.
**Non-decomposable formats**: **Code — strong fit** (backjump on unit-test-fail is canonical in program synthesis). **MCQ — weak** (no per-step PRM signal).

---

## Cross-cutting notes

**vLLM batching reality**: All five proposals batch K (or B²) candidate continuations at step boundaries. With `enable_prefix_caching=True`, KV reuse is free; the practical bottleneck is the **per-step PRM forward**, which is a separate engine. Co-locating PRM-7B and policy-7B on one H100 is tight; **score PRM async every 2 steps** rather than every step — halves PRM cost at small accuracy hit.

**Unified CP wrap**: The trajectory-level CP layer (validated primitive from Pilots 2–10) sits unchanged on top. Search changes the *generation policy*; CP only requires q̂ calibrated on the same policy deployed. Concretely: 100 calibration traces with search, score PRM-min, set q̂ = ⌈(n+1)(1−α)⌉/n quantile. Clean factoring `CP(score, search_policy)` rather than CP entangled with intervention. **All five preserve coverage** if calibration uses the matching search policy.

**Recommended pilot order**:
1. **G4** (`coupled_diverse_beams`) — most direct fix for C/K/L, cleanest CP story (reduces to SC@K already calibrated), smallest delta from existing SC pipeline.
2. **G1** (`step_beam_with_prm_value`) — workhorse beam, gives PRM-as-value apparatus needed by G2/G5.
3. **G2 / G5 / G3** in order of payoff/risk.

**Falsification path**: If G4 fails to beat SC@4 at matched compute, the diagnosis shifts: even *guaranteed-distinct* CoTs do not help → bottleneck is base-policy reasoning quality, not exploration. In that case search-based step-level intervention joins K-resample in the explored-and-rejected bin, and the paper commits to the "trajectory-level coverage layer over an existing search procedure" framing already drafted in `OBSERVATIONS_kr.md` §closing. This is a clean falsification — G4 is cheap to run (~32× on 100 problems ≈ a few minutes on 2× H100).

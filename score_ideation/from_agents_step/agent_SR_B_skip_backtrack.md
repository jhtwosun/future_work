# Agent SR-B: Step SKIP / BACKTRACK / PRUNE Methods for CoT-CP

**Author:** Agent SR-B
**Scope:** Structural manipulation of the CoT trace as a rejection mechanism — never "just regenerate the same step." Each method below detects a bad step using signals already validated by CoT-CP (entropy_mean, top1_margin_mean, lp_drawdown_sum, arith_violations_sum, prov_n_flips, prov_match_latest_rate) and **modifies the chain structure** rather than re-rolling the dice.

Notation:
- `T = [s_0, s_1, ..., s_t, ...]` is the current chain of reasoning steps.
- `score(s_t)` is a composite badness signal; we call a step "bad" when at least one calibrated component fires.
- "Continue from prefix `T[:k]`" means feed the prompt + concatenated steps `s_0..s_{k-1}` back to vLLM and decode the next step.

We propose **7 methods**, ordered roughly from cheapest to most aggressive. Methods 6 and 7 are composite escalation policies that orchestrate the earlier primitives.

---

## 1. `step_excise_continue` — Surgical step deletion

**Trigger criterion.**
A step `s_t` fires *exactly one* arithmetic-class signal (`arith_violations_sum >= 1` OR `prov_n_flips >= 2` on a single step) **but** the surrounding steps `s_{t-1}, s_{t+1}` look clean (their composite z-score `< 0.5`). The intuition: an isolated calculator slip surrounded by good context — like a typo, not a derailment.

**Action.**
Delete `s_t` from the trace. Construct `T' = T[:t] ++ T[t+1:]` and re-decode **just from position t** with a tiny suffix of the original `s_{t+1}` as a prompt anchor (or simply "Continuing:"). We do NOT regenerate `s_t` — we ask the model whether the chain still flows without it. If the new step `s'_t` is consistent with `s_{t+1}` (provenance overlap > 0.6), accept the deletion.

**Cost.**
1 forward pass to produce the bridging step, plus optional 1 NLI/cosine check (cheap, batchable).

**Hypothesis.**
Many "bad" intermediate steps are *redundant filler* (a misstated lemma the model immediately corrects, or an unused side-computation). Deleting them shortens the chain, reduces cumulative drawdown, and sometimes the model's next-step distribution sharpens because the noisy context is gone.

**Pseudocode.**
```python
def step_excise_continue(T, scores, t, llm):
    if not (scores[t].arith_violations >= 1 and
            scores[t].composite_z > scores[t-1].composite_z + 1.0 and
            scores[t].composite_z > scores[t+1].composite_z + 1.0):
        return None  # not an isolated spike
    prefix = "\n".join(T[:t])
    bridge = llm.generate(
        prompt=base_prompt + prefix + "\nNext step:",
        max_new_tokens=120, temperature=0.0, n=1,
    )[0]
    # accept only if bridge is consistent with original s_{t+1} content
    if provenance_overlap(bridge, T[t+1]) >= 0.6:
        return T[:t] + [bridge] + T[t+1:]
    return None
```

**Failure modes.**
- If `s_t` actually carried a *fact* needed by `s_{t+1}` (e.g., an intermediate value), excising it makes `s_{t+1}` reference a phantom. The provenance-overlap gate catches most of these but not all.
- Greedy bridge may collapse to a duplicate of `s_{t-1}` ("As we computed above..."), producing a stutter step.

---

## 2. `consecutive_bad_block_jump` — Skip a contiguous bad sub-chain

**Trigger criterion.**
A run of `>= 2` consecutive bad steps `s_t..s_{t+k}` where every step has `composite_z >= 1.0` AND the run ends with either an `arith_violations` spike or `prov_match_latest_rate < 0.3` (i.e., the chain has "gone off the rails" and stopped referencing earlier work).

**Action.**
Identify the last *good prefix boundary* `p` = max index `<= t` with `composite_z(s_p) < 0.5` and `prov_match_latest_rate(s_p) >= 0.6`. Truncate to `T[:p+1]` and continue generating from there. Crucially, we inject a brief steering hint: `"Let's reconsider carefully:"`. This biases the model toward a different decoding mode without using a fancy persona prompt.

**Cost.**
On average ~`k+1` forward passes are saved (we discard them) but we must re-decode `k+1` new steps. Net cost ~ replacement of bad block, comparable to a single regeneration of length `k+1`.

**Hypothesis.**
Once a model derails, regenerating one step at a time fights the prefix's pull. Jumping back to the last clean state and using a soft re-orienting cue lets the model take a different branch entirely — analogous to a student who realizes "the last paragraph is nonsense, let me start over from before that".

**Pseudocode.**
```python
def consecutive_bad_block_jump(T, scores, llm, K=8):
    bad = [i for i,s in enumerate(scores) if s.composite_z >= 1.0]
    runs = group_consecutive(bad)
    long_run = next((r for r in runs if len(r) >= 2), None)
    if long_run is None: return None
    t = long_run[0]
    p = max((i for i in range(t, -1, -1)
             if scores[i].composite_z < 0.5
             and scores[i].prov_match_latest_rate >= 0.6),
            default=-1)
    if p < 0: return None
    new_prefix = T[:p+1] + ["Let's reconsider carefully:"]
    new_tail = llm.generate(
        prompt=base_prompt + "\n".join(new_prefix),
        max_new_tokens=K*120, temperature=0.7, n=1,
        stop=["</answer>"],
    )[0]
    return T[:p+1] + split_steps(new_tail)
```

**Failure modes.**
- The "good prefix boundary" might be too far back, costing many tokens.
- If the entire chain past step 1 is bad (rare but happens on hard problems), we essentially restart and gain nothing over plain SC.

---

## 3. `backtrack_depth_ladder` — Increasing-depth backtrack

**Trigger criterion.**
Single bad step at `t` with `composite_z >= 1.5` (sharper than method 1's threshold). Could be `lp_drawdown_sum` spike (the model lost confidence sharply) OR `entropy_mean` >= 90th percentile.

**Action.**
Try regenerating from `t-1` with `n=1, temp=0.7`. Score the new step. If still bad, retry from `t-2`. Continue up to depth `d_max=3`. The first depth that yields a clean step (`composite_z < 0.5`) wins. If none clean, fall back to depth `d_max` regeneration anyway (give up gracefully).

**Cost.**
Up to `d_max` extra forward passes. Expected cost ≈ 1.5 in practice (most bad steps clear at depth 1).

**Hypothesis.**
Errors *propagate*: a bad step at `t` is sometimes caused by a *subtly* bad step at `t-1` that didn't trip the threshold. By offering the model an opportunity to revise from increasing depth, we let it find the true root cause. The "ladder" structure naturally allocates more compute to harder failures.

**Pseudocode.**
```python
def backtrack_depth_ladder(T, scores, t, llm, d_max=3):
    if scores[t].composite_z < 1.5: return None
    for d in range(1, d_max+1):
        if t - d < 0: break
        prefix = "\n".join(T[:t-d+1])
        cand_tail = llm.generate(
            prompt=base_prompt + prefix,
            max_new_tokens=180, temperature=0.7, n=1,
        )[0]
        cand_step = split_steps(cand_tail)[0]
        cand_score = score_step(cand_step, prefix=prefix)
        if cand_score.composite_z < 0.5:
            return T[:t-d+1] + [cand_step]  # truncate + replace
    # no depth helped; commit to deepest
    return T[:t-d_max+1] + [cand_step]
```

**Failure modes.**
- If the underlying difficulty is at the *problem* level (not the chain), no backtrack depth helps and we waste compute.
- Each restart can produce a *different* answer family; we may oscillate between two basins.

---

## 4. `dedup_prune_collapse` — Semantic deduplication

**Trigger criterion.**
Two steps `s_i, s_j` (j > i) with cosine similarity of sentence embeddings `>= 0.92`, OR 4-gram Jaccard `>= 0.6`, OR `prov_match_latest_rate(s_j) >= 0.95` (s_j is essentially restating s_i with the same provenance set). Additionally, `s_j`'s composite_z should be `>= 0.5` (the duplicate is causing measurable noise).

**Action.**
Delete `s_j` from the chain. Do NOT regenerate. Concatenate the remaining trace and continue from wherever generation paused. If `s_j` is the most recent step (we just emitted it), this is a "rollback by one and continue" — cheaper than backtrack because no temperature change is needed; we just resume decoding.

**Cost.**
0 extra passes for the deletion itself; 1 pass for any continuation that follows.

**Hypothesis.**
Repetition causes *attention smearing*: when the model sees the same fact twice in the prefix, downstream attention weights blur, and step `s_{j+1}` is more likely to be miscalibrated. Removing duplicates de-confuses the prefix and improves downstream entropy/margin.

**Pseudocode.**
```python
def dedup_prune_collapse(T, scores, embed):
    E = embed(T)
    drop = set()
    for j in range(1, len(T)):
        for i in range(j):
            if i in drop: continue
            sim = cosine(E[i], E[j])
            ng = ngram_jaccard(T[i], T[j], n=4)
            if (sim >= 0.92 or ng >= 0.6) and scores[j].composite_z >= 0.5:
                drop.add(j); break
    if not drop: return None
    return [s for k,s in enumerate(T) if k not in drop]
```

**Failure modes.**
- Some chains *legitimately* restate intermediate values for clarity ("So we have x = 5; substituting x = 5..."). Aggressive dedup may remove a legitimate scaffold.
- The cosine threshold is brittle across domains (math vs. commonsense).

---

## 5. `contradiction_prune_or_keep_latest` — Resolve internal contradictions

**Trigger criterion.**
A *contradiction signal* between `s_t` and any earlier `s_i` (i < t). Signals (cheap to expensive):
1. `prov_n_flips` on a tracked variable (e.g., `x` was 5, now 7) — already in our score set.
2. Sympy check: parse equations from both steps; sympy reports inconsistency.
3. Lightweight NLI head (DeBERTa-v3 small) returning `contradiction` with conf > 0.8.

**Action.**
Default: **keep the latest** step `s_t` (the model's most recent commitment) and prune `s_i` *plus everything that depended on `s_i`* (via provenance graph). If pruning >= 3 steps would be required, instead **keep the earliest** `s_i` (it had less cumulative drift) and discard `s_t`. The choice is governed by which side has lower mean composite_z.

**Cost.**
1 sympy / NLI call per candidate pair (O(n^2) pairs but in practice we only check `s_t` against the last 5 steps).

**Hypothesis.**
Contradictions are rarely silent — usually the model *knows* which side it now believes. Letting it commit to the latest claim (when the disagreement is local) avoids the cognitive dissonance of a chain that says "x=5 ... x=7 ... therefore x²=25". If the latest step is the one that is wrong, the dependency chain back to the earlier good step is shorter, and we still recover.

**Pseudocode.**
```python
def contradiction_prune(T, scores, t, prov_graph, sympy_check, nli):
    contras = []
    for i in range(max(0,t-5), t):
        if (scores[t].prov_n_flips_against(i) > 0
            or sympy_check(T[i], T[t]) == "inconsistent"
            or nli(T[i], T[t]).contradiction > 0.8):
            contras.append(i)
    if not contras: return None
    earliest = contras[0]
    # find dependents of s_earliest
    deps = prov_graph.descendants(earliest)
    # decide which side to keep
    z_old = mean(scores[k].composite_z for k in [earliest]+list(deps) if k < t)
    z_new = scores[t].composite_z
    if len(deps) < 3 and z_old > z_new:
        keep = "latest"
        T2 = [s for k,s in enumerate(T) if k not in (deps | {earliest})]
    else:
        keep = "earliest"
        T2 = T[:t]  # drop s_t
    return T2
```

**Failure modes.**
- Sympy/NLI false positives can prune correct steps.
- Some math problems *intentionally* revise (proof by contradiction); we'd want a guard for `if "contradiction" in T[i]` or for explicit "wait — actually" cues.

---

## 6. `compress_then_continue` — Replace bad sub-chain with self-summary

**Trigger criterion.**
Either (a) method 2's "consecutive bad block" of length >= 3, or (b) cumulative `lp_drawdown_sum` over a window of 4 steps exceeds the 95th-percentile calibration value. The chain has accumulated muddle, but we want to preserve *some* of the work it did.

**Action.**
Ask the model to produce a concise summary of `T[:t+1]` ("In one sentence, the relevant facts established so far are:"). Replace the muddled tail `T[p:t+1]` with that summary as a single step `s_summary`. Continue decoding from `T[:p] ++ [s_summary]`. This is *compression* rather than deletion — we keep the information content but discard the syntactic noise.

**Cost.**
1 forward pass for the summary (short, ~80 tokens), then normal continuation. Net usually cheaper than method 2 because the suffix is shorter.

**Hypothesis.**
Long bad sub-chains poison the KV cache with low-quality tokens that downstream attention must filter out. A clean summary acts as a *cache reset* with high-information density; subsequent decoding has lower entropy because the context is compressed. This mirrors how human problem-solvers "step back and write down what we know."

**Pseudocode.**
```python
def compress_then_continue(T, scores, llm):
    bad_block = find_long_bad_block(scores, min_len=3, z_thresh=1.0)
    if bad_block is None: return None
    p, t = bad_block[0], bad_block[-1]
    summary_prompt = (base_prompt + "\n".join(T[:t+1])
                      + "\n\nIn one sentence, list only the FACTS we have"
                      + " correctly established so far:\n")
    summary = llm.generate(summary_prompt, max_new_tokens=100,
                           temperature=0.0, n=1)[0].strip()
    s_summary = "Recap: " + summary
    new_T = T[:p] + [s_summary]
    cont = llm.generate(base_prompt + "\n".join(new_T),
                        max_new_tokens=600, temperature=0.7, n=1,
                        stop=["</answer>"])[0]
    return new_T + split_steps(cont)
```

**Failure modes.**
- Summaries hallucinate facts not actually established. We should constrain with "list only facts that can be verified from the steps above" and even run a provenance check on the summary.
- If the muddled block contained the *only* correct insight, compressing it may distill out the wrong nugget.

---

## 7. `escalation_ladder` — Adaptive multi-strategy controller

**Trigger criterion.**
Any single step fires `composite_z >= 1.0`. This is the policy that orchestrates methods 1–6 plus regeneration as terminal escalation.

**Action.**
A 4-tier ladder, with each tier attempted only if the previous failed (i.e., the resulting step still has `composite_z >= 0.7`):
1. **Tier 0 (free):** `dedup_prune_collapse` (method 4). 0 extra passes.
2. **Tier 1 (cheap):** `step_excise_continue` (method 1). 1 pass.
3. **Tier 2 (medium):** `backtrack_depth_ladder` with `d_max=2` (method 3). 1–2 passes.
4. **Tier 3 (expensive):** PRM-guided beam search at the bad position with width 4, depth 2. ~8 passes.
5. **Terminal fallback:** SC@8 from the last clean prefix; majority-vote the answer; do not return a chain at all (selective abstention if even SC disagrees).

The ladder is **calibrated**: each tier's stopping threshold is chosen to maintain a target marginal coverage on a held-out set, exactly mirroring CoT-CP's existing conformal calibration.

**Cost.**
Expected ~2–3 forward passes per triggered step on a typical mix; bounded by ~12 + SC@8 in the worst case.

**Hypothesis.**
A single fixed strategy mis-allocates compute: cheap methods waste opportunity on hard cases, expensive methods waste compute on easy ones. An adaptive ladder *spends compute proportional to the difficulty of recovery*, which is exactly what conformal selective accuracy rewards. Empirically we expect Tier 0–1 to handle 60% of triggered steps, Tier 2 another 25%, Tier 3 the remaining 15%, with terminal fallback firing on <2%.

**Pseudocode.**
```python
def escalation_ladder(T, scores, t, llm, prm, embed):
    # Tier 0: free dedup
    T1 = dedup_prune_collapse(T, scores, embed)
    if T1 is not None and re_score(T1, t).composite_z < 0.7:
        return T1
    # Tier 1: excise
    T2 = step_excise_continue(T, scores, t, llm)
    if T2 is not None and re_score(T2, t).composite_z < 0.7:
        return T2
    # Tier 2: backtrack ladder
    T3 = backtrack_depth_ladder(T, scores, t, llm, d_max=2)
    if T3 is not None and re_score(T3, t).composite_z < 0.7:
        return T3
    # Tier 3: PRM beam search
    T4 = prm_beam_search(T[:t], prm, llm, width=4, depth=2)
    if T4 is not None and re_score(T4, t).composite_z < 0.7:
        return T4
    # Terminal: SC@8 abstention
    answers = [llm.generate(base_prompt + "\n".join(T[:t]),
                            temperature=0.7, n=1)[0]
               for _ in range(8)]
    maj, frac = majority_vote(answers)
    if frac >= 0.5:
        return T[:t] + [maj]
    return ABSTAIN
```

**Failure modes.**
- Engineering complexity: 4 tiers means 4 places where calibration can drift. Need careful per-tier holdout splits.
- The tier-stopping check `re_score(...).composite_z < 0.7` can be gamed: a method that produces a *bland* step may pass the threshold without actually fixing the reasoning. Mitigate by adding a *progress* check (the corrected step must increase `prov_match_latest_rate` or unblock arithmetic).
- For very fast/easy problems, the ladder's overhead (re-scoring after each tier) may eat the budget. Suggest gating the ladder behind a problem-difficulty estimator (e.g., baseline entropy_mean).

---

## Summary table

| # | Method | Trigger signal | Structural change | Extra passes |
|---|---|---|---|---|
| 1 | `step_excise_continue` | isolated arith/prov spike | delete one step, bridge | 1 |
| 2 | `consecutive_bad_block_jump` | `>=2` consecutive bad | jump to last good prefix | replaces block |
| 3 | `backtrack_depth_ladder` | sharp `composite_z >= 1.5` | regenerate from t-d, d ascending | up to 3 |
| 4 | `dedup_prune_collapse` | high cosine/4-gram sim | delete duplicate | 0 |
| 5 | `contradiction_prune_or_keep_latest` | sympy/NLI/`prov_n_flips` | drop one side + dependents | 0–1 |
| 6 | `compress_then_continue` | long bad block / drawdown | summarize and replace | 1 |
| 7 | `escalation_ladder` | `composite_z >= 1.0` | adaptive 4-tier policy | 1–12 |

## Cross-cutting design notes

- **All methods preserve the trace as a list-of-steps abstraction** — none of them peek at intermediate logits, so they compose with vLLM's standard generation API. Re-scoring uses the same step-scorer module CoT-CP already runs.
- **Conformal calibration is per-method.** Each operator should be calibrated to a target conditional coverage on a held-out set: i.e., we estimate `P(final answer correct | operator triggered)` and only fire when this exceeds the calibrated threshold. This keeps CoT-CP's selective-accuracy guarantee intact.
- **Logging suggestion.** Tag each accepted step with the operator that produced it (`origin: {raw, excised, jumped, backtracked-d2, deduped, contradiction-prune, summary, prm-beam, sc-fallback}`). Downstream analysis can then attribute accuracy gains to specific structural manipulations rather than aggregate "method X helps".
- **Composability.** Methods 1, 4, 5 are *idempotent prefix operators* (they only delete) and can be applied in any order before generation. Methods 2, 3, 6 are *generative operators* and should be applied at most one per step. Method 7 enforces this composition.
- **Why not just regenerate?** A pure "regenerate-on-bad" baseline (Pilot C with K alternatives) is already in our ablation. The expectation here is that **structural manipulation has different failure modes** than re-sampling — specifically, it can fix *prefix-induced* errors that re-sampling can't because the bad prefix biases every sample identically. Method 2 (jump) and method 6 (compress) target this directly.

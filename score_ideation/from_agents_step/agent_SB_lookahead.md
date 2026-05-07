# Agent SB — Step-Level Look-Ahead & Counterfactual Scoring for CoT-CP

**Author:** Agent SB
**Date:** 2026-05-07
**Scope:** 5–7 *step-native* scoring methods for CoT-CP that exploit one-step look-ahead or counterfactual signals without requiring full re-rolls, trained probes, or trajectory-level aggregation.

---

## Design Philosophy

A CoT step is a "small action": a sentence/line ending in `\n` or a delimiter. At the boundary between step `t-1` and `t`, we have the prefix `x_{<t}`, the emitted step `s_t`, the hidden state `h_t`, and cheap access to LM logits + (optionally) a draft model.

A good step-level score `r_t` should be **one-shot or near-one-shot** in extra forward passes, **counterfactual** ("what if I'd done this differently"), **local** (not a function of the whole future), and **monotone**. For CP, treat `r_t` as a *step nonconformity*; aggregate via `max_t` (online) or `mean_top3` (offline) and calibrate `q̂`. Below: 7 methods, cheapest first.

---

## 1. `single_token_swap_consistency` (STSC)

### Definition
At step `t` with emitted tokens `s_t = (w_1,…,w_{L_t})`, find the **anchor token** `w_{i*}` with the largest top-1/top-2 logit gap (the model's most confident commitment in this step). Then:

1. Replace `w_{i*}` with the **2nd-most-likely** token `w'_{i*}`.
2. Greedy-extend `m=8` tokens from that swap.
3. `STSC_t = 1 − cos(embed(s_t[i*+1 : i*+1+m]), embed(s'_t))` (mean-pooled tied input embeddings).

### Cost
~**1 prefill + 8 decodes** with KV reuse. A few % of one full step's cost.

### Hypothesis
A *robust* step lets even its most confident token be perturbed and snap back to semantically equivalent content. A *brittle* step diverges immediately under a single-token flip — its high-confidence commitment was load-bearing in a fragile way. Counterfactual: "what if I'd been slightly less sure here?"

### Pseudocode
```python
def stsc(model, tok, prefix_ids, step_ids, m=8):
    # 1) get logits over step_ids by teacher forcing
    out = model(torch.cat([prefix_ids, step_ids]), use_cache=True)
    logits = out.logits[len(prefix_ids)-1 : -1]   # logits that produced step_ids
    top2 = logits.topk(2, dim=-1)
    gap = top2.values[:, 0] - top2.values[:, 1]
    i_star = gap.argmax().item()

    # 2) swap the anchor token
    swapped = step_ids.clone()
    swapped[i_star] = top2.indices[i_star, 1]

    # 3) extend both
    cont_orig = greedy_decode(model, torch.cat([prefix_ids, step_ids[:i_star+1]]), m)
    cont_swap = greedy_decode(model, torch.cat([prefix_ids, swapped[:i_star+1]]), m)

    e1 = mean_embed(model, cont_orig)
    e2 = mean_embed(model, cont_swap)
    return 1.0 - F.cosine_similarity(e1, e2, dim=-1).item()
```

### CP Usage
`r_t = STSC_t`. Calibrate `q̂` on `max_t r_t` over a calibration set. At inference, abstain or branch when `STSC_t > q̂` — this catches "load-bearing fragile" steps.

### Failure Modes
- Numbers / named entities as anchors → divergence always (false positive). Gate to connectives / verbs / modals via tokenizer regex.
- Argmax-deterministic models give near-identical continuations from poor swaps → use small T=0.3 on continuation.

---

## 2. `next_step_branching_divergence_K2` (NSBD)

### Definition
At the **end** of step `t` (delimiter token just emitted), draw `K=2` continuations of the **first 24 tokens of step t+1** at temperature `T=0.7`, then measure their semantic divergence:

`NSBD_t = JS( p̂(tok | branch1), p̂(tok | branch2) )`

over the unigram token distribution of the 24-token branches (skip stopwords). Equivalently, a cheap approximation:

`NSBD_t = 1 − BLEU-2(branch1, branch2) `

### Cost
2 sampled rollouts × 24 tokens = **48 decode steps** per step boundary. With KV cache shared from prefix, this is small.

### Hypothesis
After a *correct* step, two T=0.7 samples agree on the *type* of next action (same operator, variable, lemma). After a wrong/ambiguous step the model is in a high-entropy basin and the two branches diverge on content, not just phrasing. Forward counterfactual: "what does this step *force* me to do next?"

### Pseudocode
```python
def nsbd(model, tok, prefix_ids, T=0.7, K=2, m=24):
    branches = []
    for _ in range(K):
        b = sample_decode(model, prefix_ids, max_new=m, temperature=T)
        branches.append(b)
    toks = [strip_stopwords(tok.decode(b)) for b in branches]
    return 1.0 - sentence_bleu([toks[0].split()], toks[1].split(),
                               weights=(0.5, 0.5))
```

### CP Usage
`r_t = NSBD_t`. Pair naturally with **online branching**: if `NSBD_t > q̂`, *keep* both K=2 branches as parallel rollouts, and let majority vote decide later. This is a CP-driven version of Pilot C step-branching that triggers only at calibrated risky boundaries.

### Failure Modes
- High lexical diversity inflates NSBD (creative tasks). Best on math/code/structured reasoning, or score on parsed structure (vars, numbers).
- BLEU on 24 tokens is noisy; use embedding-cosine of mean-pooled hidden state as smoother variant.

---

## 3. `prefix_only_answer_consistency` (POAC)

### Definition
At step `t`, run a **second prompt** that takes `(question, steps_1..t)` and appends a short closing prompt:

```
Given the work so far, what is your current best final answer? Output only the answer.
```

Greedy-decode the answer `â_t`. Compute:

- `match_t ∈ {0,1}` against `â_{t-1}` (the same query at the previous step boundary)
- `flip_count_t = sum_{τ≤t} 1[â_τ ≠ â_{τ-1}]`

`POAC_t = flip_count_t / t` (average instability rate).

### Cost
**1 short forward pass per step** (~10–20 tokens for the answer). Crucially the prefix KV cache is *shared* with the main rollout.

### Hypothesis
A correct trace converges monotonically; an incorrect or confused trace flip-flops. Reading out the model's "current best guess" at every boundary surfaces this dynamic. Counterfactual: "what would I answer if I had to stop here?" — and the *change* over time is the signal.

### Pseudocode
```python
def poac(model, tok, prefix_ids, prior_answers):
    closing = tok.encode("\n\nGiven the work so far, the final answer is:",
                         add_special_tokens=False)
    ids = torch.cat([prefix_ids, torch.tensor(closing)])
    a_t = greedy_decode(model, ids, max_new=15, stop_on="\n").strip()
    flips = sum(1 for i in range(1, len(prior_answers))
                if normalize(prior_answers[i]) != normalize(prior_answers[i-1]))
    prior_answers.append(a_t)
    flips += int(len(prior_answers) >= 2 and
                 normalize(a_t) != normalize(prior_answers[-2]))
    return flips / max(len(prior_answers) - 1, 1), a_t
```

### CP Usage
Two scores:
- **Per-step**: `r_t = 1[â_t ≠ â_{t-1}]` — fires immediately when the model changes its mind.
- **Trajectory**: `R = POAC_T` (final flip rate).

Calibrate the latter for selective abstention; use the former to *trigger branch/verify* online when a flip happens.

### Failure Modes
- Early steps: `â_t` is a guess. Skip `t < 2`.
- Models that don't obey the closing prompt → short max_new + stop tokens.
- Long-horizon problems with legitimate late refinement → distinguish monotone refinement from oscillation via edit-distance between consecutive answers.

---

## 4. `draft_model_disagreement` (DMD) — Speculative-Decoding Signal

### Definition
Use a small draft model `q` (≈1B) to predict the next `m=16` tokens after step `t`'s boundary. Let `D = q.greedy(prefix)`. Compute under the main model `p`:

`DMD_t = − (1/m) Σ_{i=1}^{m}  log p(D_i | prefix, D_{<i})`

i.e., the main model's negative log-likelihood of the draft's continuation. A *high* DMD means the main model strongly disagrees with what a smaller, less reasoning-capable model would have done. A *low* DMD means the next 16 tokens are "easy" — anyone would produce them.

### Cost
- Draft model: 16 decode steps (small, cheap).
- Main model: 1 prefill of 16 tokens (no sampling, just logits) → essentially **1 batched forward pass**.

Total: ~**1 small + 1 prefill** of length 16. Lightweight.

### Hypothesis
At a *risky* step, the main model's reasoning is what distinguishes it from the draft — high DMD signals "this is where the big model is doing real work, and big-model-real-work is also where it can fail." At routine boundaries ("Let me re-read the question.") draft and main agree. DMD is a *step-difficulty proxy*. Counterfactual: "what would a worse model have done?" Same → step isn't where correctness is decided. Different → high-leverage step.

### Pseudocode
```python
def dmd(main, draft, tok, prefix_ids, m=16):
    draft_cont = greedy_decode(draft, prefix_ids, max_new=m)   # cheap
    full = torch.cat([prefix_ids, draft_cont])
    out = main(full).logits[len(prefix_ids)-1 : len(prefix_ids)-1+m]
    logp = F.log_softmax(out, dim=-1)
    nll = -logp.gather(-1, draft_cont.unsqueeze(-1)).mean().item()
    return nll
```

### CP Usage
`r_t = DMD_t`. *High-leverage* steps go in the calibration tail. Use as a CP nonconformity: trace risk = `max_t DMD_t` (the moment the small/big models diverged most). At inference, that step is the natural branching point — branch with K=4 there, since you have evidence the main model "knows something" the draft doesn't.

### Failure Modes
- Draft and main strongly aligned on a *wrong* high-confidence answer → DMD is low and we miss. (Shared training data → shared bugs.) Mitigation: pair with NSBD or POAC.
- Domain mismatch (draft is a code model, main is general) inflates DMD trivially. Use a draft from the same family (Qwen2.5-0.5B as draft for Qwen2.5-7B, etc.).

---

## 5. `internal_state_lm_head_lookahead` (ISLH) — Free-Lunch Hidden-State Projection

### Definition
At the step boundary `t`, take the final-layer hidden state `h_t ∈ ℝ^d`. Apply *no* attention/MLP — just project through the tied output head `lm_head` to get logits `ℓ_0 = lm_head(h_t)`. Then **iteratively** project assuming a "frozen-state" rollout: feed the argmax token *embedding* `e_1 = embed(argmax ℓ_0)` and compute `h_{t+1}^* ≈ h_t + e_1` (residual approximation, no transformer block). Repeat for `k=8` steps. Define:

`ISLH_t = − (1/k) Σ_{i=1}^{k} log softmax(ℓ_i)[argmax ℓ_i]`

(average top-1 NLL of this *cheap* synthetic rollout). High ISLH = even a frozen-state projection is uncertain about the next 8 tokens.

### Cost
**Zero extra forward passes through transformer blocks.** Only `k+1` matrix-vector multiplies through `lm_head` (a `d × |V|` matmul) and `k` embedding lookups. Effectively **free**, dominated by the one matmul we'd do anyway for the last token.

### Hypothesis
The hidden state already encodes a "summary" of what should come next. If the unattended projection is sharp (consistent argmaxes across the synthetic rollout) the local plan is committed. If flat/oscillating, even the model's own representation isn't sure. Zeroth-order lookahead: "what does my current state alone, without computation, predict?"

### Pseudocode
```python
@torch.no_grad()
def islh(model, h_t, k=8):
    # h_t: (d,) last-token last-layer hidden state
    W = model.get_output_embeddings().weight   # |V| x d
    E = model.get_input_embeddings().weight    # |V| x d
    h = h_t.clone()
    nlls = []
    for _ in range(k):
        logits = W @ h
        logp = F.log_softmax(logits, dim=-1)
        tok = logp.argmax()
        nlls.append(-logp[tok].item())
        h = h + E[tok]                         # residual approx, no block
        h = h / h.norm() * h_t.norm()          # rescale to manifold
    return sum(nlls) / k
```

### CP Usage
`r_t = ISLH_t`. Because it's free, compute at *every* token, not just step boundaries — gives a fine-grained risk curve. CP threshold on `max_t ISLH_t` over the trace, but also identify the *peak* token to branch from.

### Failure Modes
- Bad approximation for long-range-attention steps (multi-step arithmetic). Tracks *local* committedness only; complement, not replacement.
- Norm-rescaling is a heuristic — without it, drift dominates. Validate score-vs-correctness on held-out before deploying.

---

## 6. `abductive_self_justification_gain` (ASJG)

### Definition
After step `s_t` is emitted, run a second prompt:

```
[question] [steps 1..t]

In one short sentence, why is step {t} valid?
```

Sample the justification `j_t` (greedy, ≤30 tokens). Then measure how much `j_t` reduces the model's uncertainty about a *random follow-up step* `s_{t+1}'`. Concretely:

1. Without `j_t`: sample `s_{t+1}^A` at T=0.7.
2. With `j_t` inserted: sample `s_{t+1}^B` at T=0.7.

`ASJG_t = JS( P(s_{t+1}^A), P(s_{t+1}^B) )`

If the justification *changes* the next-step distribution a lot, the model didn't actually have a coherent reason for `s_t` — making the reason explicit shifted its beliefs. If `ASJG_t ≈ 0`, the justification is consistent with the model's existing trajectory, which corresponds (we hypothesize) to higher correctness probability.

### Cost
~30 tokens for `j_t` + 2 short rollouts (~24 tokens each) = **~80 decode steps per step boundary**. Heaviest of the methods listed but still <<full re-roll.

### Hypothesis
A correct step has a justification the model has already implicitly used — making it explicit doesn't shift downstream behavior. An unjustified step (lucky guess, hallucinated arithmetic) is one where forcing articulation exposes the gap. Counterfactual: "what if you had to defend this?"

### Pseudocode
```python
def asjg(model, tok, prefix_ids, t, K=2):
    just_prompt = tok.encode(f"\n\nWhy is step {t} valid? Briefly:")
    j = greedy_decode(model, torch.cat([prefix_ids, just_prompt]), 30)
    extended = torch.cat([prefix_ids, just_prompt, j])

    a = [sample_decode(model, prefix_ids, 24, T=0.7) for _ in range(K)]
    b = [sample_decode(model, extended, 24, T=0.7) for _ in range(K)]

    return js_divergence(unigram(a), unigram(b))
```

### CP Usage
`r_t = ASJG_t`. Use as "self-doubt detector"; particularly good in CP for *abstention* (rather than branching) since high ASJG means the model itself doesn't know why it did what it did.

### Failure Modes
- Confabulation: LLMs are good at post-hoc justifications, so `ASJG_t` may be uniformly low. Combine with `P(j_t | prefix, s_t)` — struggling justifications have low LM probability.
- Sensitive to exact prompt wording.

---

## 7. `backward_step_reconstruction` (BSR) — Causal-Flip Test

### Definition
Given prefix steps `s_1..s_{t-1}` and the next-state summary, **mask** `s_t` and ask the model to fill it in. Specifically:

1. Generate a 1-line summary `Σ_{t+1}` of where the trace is at the end of step `t+1` (cheap: one short prompt or — even cheaper — use `s_{t+1}` itself as the future-anchor).
2. Prompt: `[question] [s_1..s_{t-1}] [???] [s_{t+1}]\n\nWhat must step t have been?`
3. Greedy-decode the reconstruction `ŝ_t`.
4. `BSR_t = 1 − ROUGE-L(s_t, ŝ_t)`.

### Cost
- One forward pass to get `s_{t+1}` (which we'll generate anyway in the main rollout — *zero marginal cost*).
- One reconstruction decode (~30 tokens).

Effectively **~30 decode steps per step boundary**, with no extra prefill since the prefix is reused.

### Hypothesis
A causally-coherent CoT has every intermediate step *implied* by its neighbors — the model can fill in step `t` from `s_{<t}` and `s_{t+1}`. A non-coherent step doesn't logically connect its surroundings; masked, the model would write something different. Counterfactual: "knowing the future, would you still have written this past?" — bidirectional perplexity at *step* rather than *token* granularity.

### Pseudocode
```python
def bsr(model, tok, steps, t):
    prefix = "\n".join(steps[:t-1])
    after = steps[t+1] if t+1 < len(steps) else ""
    prompt = (f"{question}\n{prefix}\n[STEP {t} REDACTED]\n{after}\n\n"
              f"Reconstruct step {t}:")
    rec = greedy_decode(model, tok.encode(prompt), 40, stop="\n")
    return 1.0 - rouge_l(steps[t], rec)
```

### CP Usage
Computed *retrospectively* at step `t` after `s_{t+1}` is known. Lag-1 score: `r_{t-1} = BSR_{t-1}`. Use trace-level `max_t BSR_t` for CP; use online for *late branching* (we discover step `t-1` was bad after producing `s_{t+1}` — branch from there, replacing `s_{t-1}` and re-generating the rest).

### Failure Modes
- Many CoT steps are paraphrasable; high BSR may just mean "rewording" not "wrong". Use embedding cosine (sentence encoder) or self-BLEU on parsed argument-structure rather than ROUGE.
- Doesn't apply to the final step (no `s_{t+1}`); use the predicted final answer as the future anchor.

---

## Combination Strategy

These 7 scores capture distinct counterfactual axes:

| Score | Counterfactual | Direction | Cost |
|-------|----------------|-----------|------|
| STSC  | swap 1 token  | forward | ~8 dec |
| NSBD  | sample futures | forward | ~48 dec |
| POAC  | "stop now"    | terminal | ~15 dec |
| DMD   | weaker model  | model-class | ~16 dec |
| ISLH  | no-attention  | forward | ~free |
| ASJG  | with reason   | both | ~80 dec |
| BSR   | mask & infer  | backward | ~40 dec |

For CP, recommended **two-tier nonconformity**:

1. **Free tier (always on):** `r_t^{free} = α·ISLH_t + β·POAC_t`. ISLH is matmul-only; POAC reuses KV cache. Threshold `q̂_1` for cheap abstention.
2. **Investigative tier (triggered):** when `r_t^{free}` is in the warning band (> `q̂_1/2`), spend extra compute on `NSBD + DMD`; threshold `q̂_2` for branching/verification.

Trace passes `q̂_1` ⇒ accept; fails ⇒ investigate; fails `q̂_2` ⇒ branch Pilot-C-style at the worst step, but restricted to cases where `DMD` is also high. Advantage over plain Pilot C: branching is *targeted* at boundaries the model itself flags as risky, not always at entropy-max.

---

## Open Questions

1. **Calibration drift across difficulty.** Pilot C had lift only on weak-model-on-hard-data. ISLH and POAC may share that profile (need ambiguity to detect). DMD may be opposite — strong-model-on-medium-data where big-vs-small gap is largest. Run a 2×2 grid.
2. **Aggregation across `t`.** `max_t` is brittle but necessary online; `top-3` or a quantile is better for offline trajectory CP.
3. **Validity guarantees.** Online step-thresholding violates exchangeability if we *act* (branch). Recover via Bonferroni step-budget or adaptive CP (Gibbs–Candès).

---

*End of agent SB notes.*

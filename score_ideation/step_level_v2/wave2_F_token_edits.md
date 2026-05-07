# Wave 2 — Agent F (token-edit angle)

**Brief.** Pilots C / K / L all converged on the same null result: K-resampling from a fixed step-prefix at temperature 0.7 yields continuations that are stylistically diverse but reasoning-identical. The model commits to its causal "frame" several tokens *before* the step boundary, so resuming at the boundary samples paraphrases of the same wrong derivation. The bottleneck is the **edit locus**, not the trigger or K. We need interventions that mutate the *first few decoded tokens* of the rewrite, so the first divergence happens at a token the original trajectory could not have produced.

All proposals assume vLLM ≥ 0.6 with `SamplingParams.logit_bias`, `allowed_token_ids`, `logits_processors`, and HF tokenizer access. Triggers use existing signals: `lp_min` (worst-step min log-prob), `prm_min` (PRM step score), and `tok_lp` (per-token log-prob trace, cached during the original decode).

---

## 1. `forbidden_top1_redecoding`

**Mechanism.** At the worst step's *first decoded token position* `t0`, look up the original token `o = y[t0]`. Re-run decoding from `prefix = y[:t0]` with `logit_bias = {o: -inf}` for *exactly one step* (one token), then continue greedy (or low-temp) decoding for the remainder. Concretely: emit the second-best token `o'`, then release the bias and resume. This is a single-token surgical edit that forces the chain into a different lemma family without paraphrase.

**Trigger condition.** `lp_min` falling in the bottom decile, AND the token where `lp_min` is realized is the step's opening token (or within its first 3 tokens). If lp_min lands deep in the step (e.g., a numerical mistake), use mechanism #5 instead.

**Compute cost.** ~1.0× greedy. We re-decode only the suffix after `t0`; the prefix KV cache is reusable. There is no K-fan-out; this is a *single* alternative trajectory.

**Why it should beat K-resample.** K-resample at temp=0.7 still places ~70-90% of probability mass on the original top-1 because Qwen-style CoT models have very peaked step-opening distributions (entropy < 0.5 nats is common). Drawing K=4 samples almost always returns the same opener at least three times. Forcing `o → o'` with an `-inf` bias guarantees a different lexical anchor, and downstream tokens are re-conditioned on that anchor, so the *whole rewrite* drifts. We get diversity by construction, not by luck.

**Implementation sketch.**
```python
sp = SamplingParams(temperature=0.0, max_tokens=remaining,
                    logit_bias={original_tok_id: -100.0},
                    stop=["\nStep", "</answer>"])
# vLLM: prefix is sent as prompt_token_ids, KV is rebuilt cheaply
out = llm.generate(prompt_token_ids=y[:t0], sampling_params=sp)
# After 1 token, drop the bias for the rest:
sp2 = SamplingParams(temperature=0.0, max_tokens=remaining-1, stop=...)
out2 = llm.generate(prompt_token_ids=y[:t0]+[out.outputs[0].token_ids[0]], sampling_params=sp2)
```
A `LogitsProcessor` that applies the bias only at the first call gives a cleaner one-shot.

**Risks.** (a) If the second-best token has very low probability (`p(o') < 0.02`), we may force a *worse* opener and get gibberish. Mitigate by skipping the intervention when `p(o') / p(o) < 0.1`. (b) Single-trajectory: if it fails, we have no fallback. Pair with #4 as a guarded retry.

---

## 2. `low_confidence_token_swap`

**Mechanism.** Inside the worst step, find the token position `t*` with the *minimum* per-token log-prob (the model's own least-confident commitment, which is empirically correlated with arithmetic and symbolic mistakes). Replace that single token with the runner-up from the original logits, then resume greedy decoding from `t*+1`. The swap is local and the model "sees" its own past as if it had committed to the runner-up.

**Trigger condition.** Step-level `lp_min` in the worst decile, AND the position `t*` of the minimum is *not* a step-opener (i.e., the mistake is mid-derivation: a number, a sign, a variable name). This complements method #1.

**Compute cost.** ~1.0× greedy. We need the original logits at `t*` (cached during the original decode via `logprobs=5`), so no extra forward pass for the swap itself; only the suffix is re-decoded. If logprobs were not cached, add a single full-prompt forward pass to recover them: total ~1.3×.

**Why it should beat K-resample.** K-resample at the *step boundary* never touches mid-step tokens — once the rollout commits to "5 × 7 = 32" at character 12, every K-sample re-derives the same 32. By editing exactly the token where the model was most uncertain ("32"), we surgically correct the suspected error site. This is essentially "self-debugging at the model's own admitted weak link," which K-resample cannot do because it edits the wrong location.

**Implementation sketch.** During greedy generation, request `logprobs=5` and store the top-5 token IDs and logprobs per position. Compute `t* = argmin lp` over the worst-step token range. Build `y' = y[:t*] + [runner_up_id]`, then `llm.generate(prompt_token_ids=y', sampling_params=greedy)`.

**Risks.** (a) The "least-confident" token may be a stylistic word (e.g., "Therefore" vs. "Thus") rather than a content token; the swap then has no semantic effect. Filter `t*` to content tokens by requiring the top-1 to be a digit, operator, or non-stopword. (b) If the runner-up is also wrong, we replace one error with another; this is a coin-flip on truly hard problems.

---

## 3. `top_k_token_ban_with_pivot_cue`

**Mechanism.** At the worst step's opening position `t0`, ban the top-K (K=3) most likely *first tokens* by setting `logit_bias[i] = -inf` for each, *and* prepend the model's resumption with a pivot cue token sequence that primes a different derivation style. Specifically: insert `["Wait", ",", " let", " me", " try", " a", " different", " approach", "."]` as forced tokens (via `allowed_token_ids` or by appending to the prompt), then release into free decoding with the top-3 bans still active for the next 5 tokens, then drop the bans.

**Trigger condition.** Either `lp_min` *or* `prm_min` in the bottom decile. The pivot cue is robust to both signals because it forces a structural reset rather than a local edit.

**Compute cost.** ~1.0× greedy plus ~9 forced tokens ≈ 1.05× total. Single trajectory.

**Why it should beat K-resample.** Pilot L tried a rewrite cue *as a prompt* but allowed the model to ignore it by re-emitting its preferred opener. Combining the cue with a *hard ban* on the previous top-3 openers means the model cannot fall back to its preferred path even if the cue's effect is weak. The cue + ban interact multiplicatively: the cue biases the latent state toward "reconsider," and the ban removes the lexical attractors that would otherwise pull it back. K-resample has neither lever.

**Implementation sketch.**
```python
pivot = tokenizer.encode(" Wait, let me try a different approach.", add_special_tokens=False)
# Force-decode pivot tokens
forced = y[:t0] + pivot
# Ban top-3 openers for next 5 tokens
class TempBan(LogitsProcessor):
    def __init__(self, bans, ttl): self.bans, self.ttl = bans, ttl
    def __call__(self, ids, logits):
        if self.ttl > 0:
            logits[..., self.bans] = -1e9
            self.ttl -= 1
        return logits
sp = SamplingParams(temperature=0.0, logits_processors=[TempBan(top3_ids, 5)], ...)
```

**Risks.** (a) The cue is in-distribution for instruction-tuned models but may push the model into apology mode ("I apologize for the confusion") rather than a real re-derivation. Mitigate by choosing cues that are mathematically loaded ("By a different identity," "Substituting instead,"). (b) Banning too many openers may push probability onto degenerate tokens. Cap K=3.

---

## 4. `lookahead_grafted_beam`

**Mechanism.** At step boundary `t0`, fan out a **beam of width B=8** but only for `H=12` tokens (a "lookahead horizon" — much shorter than a full step). For each beam, compute a fast confidence score: mean log-prob over the H tokens. Keep the top-2 beams that *also* differ from the original in their first token (anti-mode-collapse filter). For each kept beam, *graft* the H-token prefix back onto the trajectory and resume greedy decoding to step end. Pick the resulting full-step rewrite by PRM score (cheap because steps are short).

**Trigger condition.** `lp_min` *or* `prm_min` in the bottom decile. Particularly useful when neither single-token edit (#1, #2) is well-localized.

**Compute cost.** Beam width 8 × 12 tokens = 96 token-evaluations for the lookahead, plus 2 full step completions of ~80 tokens = ~256 tokens of decoding. Compared to greedy ~80, this is ~3.2× — comparable to K=4 sampling but spent on *beam diversity* not *temperature noise*.

**Why it should beat K-resample.** Beam search at width 8 with first-token diversity constraint enumerates the model's actual top-8 *trajectories*, not 4 noisy samples from a peaked distribution. The 12-token lookahead is long enough to expose whether a beam commits to a different reasoning move (e.g., factor vs. expand) before we pay the full continuation cost. The "first-token-must-differ" rule is exactly the diversity guarantee K-resample lacks.

**Implementation sketch.** vLLM supports beam search via `SamplingParams(use_beam_search=True, best_of=8, n=8, max_tokens=12)`. Post-process: dedupe by first token, keep top 2 by mean logprob, then for each, call `llm.generate` with `prompt_token_ids = y[:t0] + beam_prefix` and greedy decode to `\nStep` or `</answer>`.

**Risks.** (a) Beam search in vLLM does not natively expose mean-logprob per beam; we must request `logprobs=1` and aggregate manually. (b) Beam search is known to mode-collapse for long horizons; the H=12 cap limits this. (c) 3.2× compute may be the upper end of acceptable for a step-level fix.

---

## 5. `entropy_targeted_token_perturbation`

**Mechanism.** Instead of editing at a position chosen by the score (boundary or `lp_min`), edit at the position with the highest *entropy* in the worst step — i.e., where the model itself was most ambivalent. Compute `H_t = -Σ p_i log p_i` from cached top-5 logprobs across the worst step. At `t† = argmax H_t`, sample one token from the *renormalized tail* (exclude the original top-1, sample from positions 2-5 weighted by their probabilities). Resume greedy from `t†+1`.

**Trigger condition.** `lp_min` in worst decile **and** any token in the worst step has `H_t > 1.0` nat. If no high-entropy token exists, fall back to method #1 (the model is overconfident throughout, so editing the opener is the only lever).

**Compute cost.** ~1.0× greedy (logprobs cached from original decode, suffix re-decode only).

**Why it should beat K-resample.** K-resample at temperature 0.7 amplifies *every* token's variance uniformly, so high-confidence tokens (e.g., "Step", "=") get spurious noise while genuinely ambiguous tokens still pick the same value most of the time. Entropy-targeted editing instead spends the budget at the *one* position where the model's own posterior says "I could have gone several ways here" — which is precisely the position most likely to flip the trajectory's outcome. It is K-resample's noise, but applied surgically.

**Implementation sketch.**
```python
H = [-sum(p*math.log(p) for p in probs_t) for probs_t in step_top5_probs]
t_star = step_start + np.argmax(H)
top5 = step_top5_ids[t_star - step_start]
top5_probs = step_top5_probs[t_star - step_start]
tail_ids, tail_p = top5[1:], top5_probs[1:] / sum(top5_probs[1:])
swap_id = np.random.choice(tail_ids, p=tail_p)
y_new = y[:t_star] + [swap_id]
out = llm.generate(prompt_token_ids=y_new, sampling_params=greedy)
```

**Risks.** (a) High entropy is not the same as high error probability; the model can be ambivalent about purely stylistic choices. Filter by requiring the top-1 token to be content (digit/symbol/noun). (b) Stochastic — repeated runs differ. Use a fixed seed for reproducibility in the eval harness.

---

## Cross-cutting design notes

**Meta-claim.** In peaked autoregressive distributions, the *first divergent token* dominates the trajectory. K-resample delegates that divergence to a softmax-temperature sampler — noisy and biased. Token-level methods *deterministically choose* the divergence by ID (#1, #3), position (#2, #5), or enumerated trajectories (#4). This shift from "sample and hope" to "select and verify" is what Pilots C/K/L lack.

**vLLM gotchas.**
- Use `-1e9` (not `-math.inf`) in `logit_bias` to avoid NaN propagation.
- `allowed_token_ids` is per-request, not per-position; for position-conditional bans, use a counter-based `LogitsProcessor` (see #3).
- KV-cache reuse requires identical `prompt_token_ids`; vary only sampling params / processors for fastest wall-clock.
- Beam search (`use_beam_search=True`) is incompatible with `temperature>0` and most logits processors — run as a separate phase.

**Recommended ordered ablation.**
1. `low_confidence_token_swap` (#2) — cheapest; targets the strongest error correlate.
2. `forbidden_top1_redecoding` (#1) — orthogonal, cheap, isolates the opener's role.
3. `top_k_token_ban_with_pivot_cue` (#3) — tests whether Pilot L failed because its cue was ignorable.
4. `entropy_targeted_token_perturbation` (#5) — stochastic; tie-breaker.
5. `lookahead_grafted_beam` (#4) — most expensive; fund only if #1-#3 each clear +2pt.

**Triage.** If all five still come in at +1, the trigger is likely mis-localized — the error sits upstream of the worst step. Token-level edits give a clean test: if even forced first-token divergence (#1) fails to recover, the next wave should focus on *cumulative* edit windows (rewrite backwards from `argmax(prm_drop)`), not local ones.

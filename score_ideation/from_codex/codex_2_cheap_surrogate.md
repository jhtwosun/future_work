# Codex 2 (cheap-surrogate angle)

Goal: approximate the epistemic-uncertainty signal of `sc_top1` at 2-4x compute, without paying for full `SC@8`. The hypothesis is that `SC@8` works because disagreement across diverse samples proxies posterior spread. Cheap surrogates should preserve *variance under perturbation* while staying CP-friendly.

Below are five score functions chosen for vLLM simplicity, calibration clarity, and diversity signal.

## 1) Greedy-vs-Stochastic Agreement

**Mechanism.** Compare the greedy answer to one stochastic sample. If a single low-temperature perturbation flips the answer, the example likely sits near a decision boundary. This is the cheapest way to expose local epistemic instability.

**Formula.**
```text
s(x) = 1[ a_greedy(x) = a_sample(x; T=0.7) ]
```
Optionally use a soft version:
```text
s(x) = log p( a_greedy = a_sample )
```
where the stochastic sample is decoded once with temperature 0.7 and the greedy decode is the baseline.

**Compute cost.** About `2x` total inference: one greedy pass plus one sampled pass.

**Why it can approach SC@8 LR+.** `SC@8` gets value from answer stability across draws. This compresses the same signal into one perturbation test. If the model is sharply peaked, greedy and sampled decodes agree; if it is uncertain, they diverge. Because the score is monotone in stability, CP thresholding still sorts by confidence.

**Implementation.**
- vLLM call 1: `temperature=0.0`, `n=1`, greedy decode to a short canonical final-answer format.
- vLLM call 2: `temperature=0.7`, `top_p=0.95`, `n=1`, same prompt and same stop rules.
- Prompt design: force a single final answer in a fixed delimiter, e.g. `Final: ...`. Keep rationale hidden if you want the score to depend on final answer only.
- To reduce format noise, normalize answers before comparison: strip punctuation, units, and whitespace.

## 2) Speculative Multi-View Verification

**Mechanism.** Use a cheap draft model to propose `K` candidate answers or solution sketches, then verify only those candidates with the main model. Diversity comes from the candidate set, but verification is concentrated on a small subset rather than full SC decoding on the expensive model.

**Formula.**
```text
s(x) = max_i log p_main( a_i | x )
```
where `{a_i}` are `K` proposals from a draft model or a light prompt variant, and the score is either the best verified candidate or the fraction accepted:
```text
s(x) = (1/K) Σ_i 1[ main_model_accepts(a_i) ]
```

**Compute cost.** Roughly `1.5-2x` if the draft model is much cheaper than the verifier and `K` is small (`2-4`). The main model is only used for verification, not for generating many full trajectories.

**Why it can approach SC@8 LR+.** Draft diversity approximates the local hypothesis space. A broader cheap model can recover many of the alternative reasoning modes that `SC@8` surfaces. Verification then checks whether the main model endorses those alternatives, giving low scores when many plausible alternatives exist.

**Implementation.**
- Draft generation: smaller model, same question, `n=K`, `temperature=0.8`, short chain-of-thought or answer-only sketches.
- Verification: main model prompted with `Question + Candidate answer` and asked `Is this answer consistent? yes/no`.
- Score by the max accepted candidate or average acceptance rate.
- Use vLLM batching: one draft batch plus one verifier batch over the `K` candidates.
- Prompt design: keep the verifier strict and binary. The verifier should not solve the problem from scratch; it should judge agreement with the candidate.

## 3) Beam-Pair Margin

**Mechanism.** Run a small beam search and score the *gap* between the top beam and the runner-up. If several high-probability reasoning paths survive beam search, the model is less certain. This is a cheap approximation to diversity because beams often capture distinct answer modes even when samples do not.

**Formula.**
```text
s(x) = log p(b_1 | x) - log p(b_2 | x)
```
or, if final answers differ,
```text
s(x) = 1[ ans(b_1)=ans(b_2) ] + λ * (log p(b_1)-log p(b_2))
```
Lower margin means more uncertainty.

**Compute cost.** About `2-3x`, depending on beam width `K=2-4` and sequence length.

**Why it can approach SC@8 LR+.** Beam search tends to preserve the most probable modes, which is exactly where epistemic ambiguity shows up when the answer is not uniquely supported. `SC@8` estimates answer mass over several samples; beam margin estimates how concentrated that mass is near the top. If the top beams disagree, that is a strong ambiguity signal. If they agree but the margin is small, the model is still locally uncertain.

**Implementation.**
- vLLM beam search with `best_of=K`, `use_beam_search=True`.
- Ask for a strict final-answer tokenization so beam termination is aligned across paths.
- Compute both path-level margin and answer-level agreement among beams.
- For CP, use negative margin as the score so higher score means higher confidence.
- This is order-sensitive in the sense that later branching matters less than early branching; that is useful because early divergence usually reflects deeper uncertainty.

## 4) Prefix Branch Sensitivity

**Mechanism.** Decode the first part of the answer once, then branch only at a short prefix checkpoint by sampling two continuations from the same partial state. If the final answer changes under a small suffix perturbation, the solution is fragile. This is a cheap way to measure local basin width.

**Formula.**
```text
s(x) = 1[ ans(prefix ⊕ c_1) = ans(prefix ⊕ c_2) ]
```
where `prefix` is the shared partial decode and `c_1, c_2` are two continuations sampled from the same prefix state. A softer version is:
```text
s(x) = KL( P(a | prefix, T=0.3) || P(a | prefix, T=0.9) )
```

**Compute cost.** About `2x` if the prefix is reused and only the suffix is branched.

**Why it can approach SC@8 LR+.** This captures *path instability* rather than just answer disagreement. Prefix branching shows whether the answer is determined early or depends on fragile late-stage choices. Many hard examples flip when an intermediate step changes, and this score exposes that with much less duplicated work.

**Implementation.**
- First decode until a fixed checkpoint: after the first sentence, theorem, or arithmetic stage.
- Cache the KV state at the checkpoint.
- Resume twice with `temperature=0.7` and different seeds.
- Compare final answer strings or logits over the answer token set.
- Prompt design: explicitly ask for a short intermediate checkpoint, e.g. `Stop after the setup step, then continue only when resumed`.

## 5) Cross-Prompt Agreement Ensemble

**Mechanism.** Run the same question under two prompt framings that are semantically equivalent but operationally different, then measure agreement. The diversity source is prompt-induced representation shift. This is especially useful because prompt perturbations can reveal hidden brittleness that pure sampling misses.

**Formula.**
```text
s(x) = 1[ a(x; prompt_A) = a(x; prompt_B) ]
```
or, if using logits,
```text
s(x) = - KL( P_A(a|x) || P_B(a|x) )
```

**Compute cost.** About `2x`.

**Why it can approach SC@8 LR+.** `SC@8` approximates a posterior over reasoning traces. Cross-prompt agreement measures invariance to framing. If the model knows the answer, equivalent prompts should converge; if it is uncertain, prompt wording can push it into different heuristics. That yields diversity without 8 stochastic decodes.

**Implementation.**
- Use two system prompts: one direct, one step-by-step, or one terse and one verification-oriented.
- Keep the user question identical.
- Run one decode per prompt, ideally with the same temperature.
- Score agreement on the canonical final answer.
- For CP, this is attractive because the score is symmetric and easy to calibrate.

## Ordering and CP coverage

If the goal is to preserve the CP Pareto-coverage guarantee, the most useful scores are those that produce a *rankable uncertainty ordering*, not just a binary signal. Recommended order:
1. Prefix Branch Sensitivity
2. Beam-Pair Margin
3. Greedy-vs-Stochastic Agreement
4. Cross-Prompt Agreement Ensemble
5. Speculative Multi-View Verification

The first two are the best CP candidates because they are monotone in instability and expose a continuous score. The agreement-based variants are easier to implement but coarser. For production, start with `Beam-Pair Margin` plus `Greedy-vs-Stochastic Agreement` and pick whichever gives better LR+ at the same coverage.

## Practical recommendation

The cheapest path to a real gain is a two-feature composite:
```text
score(x) = w1 * beam_margin(x) + w2 * greedy_vs_sample_agreement(x)
```
This keeps compute near `3x`, preserves an explicit diversity signal, and is likely the closest approximation to `SC@8` without 8 full samples. If one extra pass is affordable, add cross-prompt agreement as a calibration feature.

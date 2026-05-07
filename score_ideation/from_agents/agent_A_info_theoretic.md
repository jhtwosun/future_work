# Agent A — Information-Theoretic Score Functions for CoT-CP

**Author:** Agent A (info-theoretic angle)
**Date:** 2026-05-05
**Cross-reference:** to be read alongside `from_agents/agent_*` proposals from sibling agents working on PRM-internal, geometric/embedding, and decoding-perturbation angles.

---

## 0. Setup and notation

A CoT trajectory is a sequence of reasoning steps `s_1, …, s_T`, each a token sequence `s_t = (x_{t,1}, …, x_{t,L_t})`. At decode time, for every emitted token `x_{t,i}` the LLM exposes a categorical `p_{t,i}(·) = softmax(z_{t,i} / τ)` over vocabulary `V` (logits `z_{t,i}`, temperature `τ`). With vLLM `logprobs=k` we receive the top-`k` logprobs at every position. Hidden states `h_{t,i} ∈ R^d` are accessible via `--return-hidden-states` or a teacher-forced HF forward pass.

The existing scores anchor the Pareto frontier as:

| score | compute | type |
|---|---|---|
| `lp_min` | 1× | scalar from greedy logprobs |
| `prm_min` | 2× | extra forward through PRM |
| `sc_top1` | N× | N independent samples, voting |

All proposed scores below are **higher = more confident** (i.e. correctness predictors); when a definition naturally produces "uncertainty" we negate. The CP wrapper consumes them as nonconformity scores `s = -confidence`.

---

## 1. `step_entropy_p95` — tail-quantile of per-step token entropy

### Definition
For each token `x_{t,i}` define Shannon entropy
  `H_{t,i} = -∑_{v ∈ V} p_{t,i}(v) log p_{t,i}(v)`.
Aggregate within a step by mean:
  `H̄_t = (1/L_t) ∑_i H_{t,i}`.
Trajectory score:
  `score = -Quantile_{0.95}({H̄_t}_{t=1..T})`.

### Compute
**1.0×** — needs only logprobs already produced by the greedy decode. Top-`k` truncation introduces a small bias; using `k = 20` captures > 99% of the entropy mass for most reasoning models because the tail of softmax is flat (and we can correct using the residual mass `1 - ∑ exp(logp_top_k)` distributed uniformly over `|V| - k`).

### Intuition
`lp_min` only registers the probability assigned to the *chosen* token; it ignores how peaked the alternatives are. Two tokens can share `logp = -1.5` while one has the rest of the mass concentrated on a near-synonym (low entropy → safe paraphrase) and the other smeared across logically incompatible continuations (high entropy → genuine ambiguity). The 95th percentile is robust to a single high-entropy boilerplate step ("Let me think about this …") yet sensitive to one genuinely confused step.

### Implementation
```python
# Per-token H computed from top-k logprobs returned by vLLM
def step_entropy(logprobs_topk, k_residual_mass):
    p = np.exp(logprobs_topk)            # (L, k)
    H = -(p * logprobs_topk).sum(axis=1)
    H += -k_residual_mass * np.log(k_residual_mass / (V - k) + 1e-12)
    return H.mean()                       # per step
```
Step boundaries: re-use the regex from `pilot3_step_calibration.py` (newline + numbered prefix) or the model's natural `\n\n`.

### Hypothesized Pareto position
Between `lp_min` and `prm_min`. Same compute as `lp_min` but strictly more information per token; should beat it on AUROC by ~1–3 pts on MATH-500 / AIME, while remaining far cheaper than self-consistency.

### Risks
- Top-`k` truncation underestimates entropy when the model is genuinely uncertain (many small logits). Mitigation: also request `prompt_logprobs` style full distributions for a subsample.
- Calibration drift across temperatures: entropy scales with `τ`; record `τ` and report scores at fixed `τ` only.

---

## 2. `predictive_sharpening_rate` — Fisher-style coherence across steps

### Definition
Let `H̄_t` be the mean per-token entropy of step `t` (as in §1). Fit `H̄_t = α + β·t + ε_t` by OLS. Define
  `score = -β`.

A *negative* slope (entropy decreases as the trajectory progresses) means the model is converging — its predictive distribution sharpens as conclusions come into focus. We negate so that **larger sharpening rate ⇒ larger score ⇒ more confident**.

### Compute
**1.0×** — same data as score #1, plus an O(T) regression.

### Intuition
This is the discrete analog of an information-bottleneck signal: a correct CoT compresses problem state into an increasingly committed posterior over the next reasoning move. An incorrect CoT often *flattens* near the end as the model hedges between contradictory partial conclusions, or stays uniformly uncertain. We capture *the second derivative of confidence over reasoning depth*, which neither `lp_min` (only the worst point) nor `prm_min` (a per-step quality independent of position) sees.

### Implementation
```python
H_per_step = [mean_entropy(step) for step in steps]
beta = np.polyfit(np.arange(len(H_per_step)), H_per_step, 1)[0]
score = -beta
```

### Pareto position
Cheapest end of the frontier, but **orthogonal** to `lp_min` (slope vs. minimum). Best used in an ensemble — see §6.

### Risks
- Short trajectories (T < 4): regression unstable. Fall back to `H̄_T - H̄_1`.
- Some models always sharpen at the end (boilerplate "The answer is ..."), inducing a constant offset that masks problem-specific signal. Mitigation: drop the final `\boxed{}` step before regression.

---

## 3. `tempered_kl_divergence` — KL between low-T and high-T forward passes

### Definition
Run two forward passes over the *same* greedy trajectory tokens (no resampling — teacher-forced):
- pass A: temperature `τ_A = 0.7` (the deployment temperature)
- pass B: temperature `τ_B = 1.5`

Each pass yields token distributions `p^A_{t,i}, p^B_{t,i}`. Compute symmetric KL ("Jensen-Shannon"):
  `JS_{t,i} = ½ KL(p^A ‖ m) + ½ KL(p^B ‖ m)`, where `m = ½(p^A + p^B)`.
Aggregate:
  `score = -mean_t mean_i JS_{t,i}`.

### Compute
**~1.1×** — both passes are *non-autoregressive* (teacher-forced over a fixed token sequence), so each costs roughly one prefill. Total cost: 2× prefill ≈ 1.0–1.2× a single greedy generation (since generation cost is dominated by per-step decode latency, not prefill, for trajectories of typical length 200–600 tokens).

### Intuition
A confident region of the LM's loss landscape is *temperature-stable*: scaling the logits by 2× barely shifts the argmax or the high-mass alternatives. A confused region is hyper-sensitive to temperature — high-T smears mass to outright-wrong continuations. JS divergence between the two profiles measures *flatness curvature* of the local logit geometry, a kind of poor-person's Fisher information (`F ≈ ∂² log p / ∂τ²`).

This is genuinely cheaper than `sc_top1` because we never re-decode — we just rescore. And it captures uncertainty `lp_min` misses entirely (a sharp peak that is robust to perturbation looks identical to a sharp peak that collapses under perturbation, in `lp_min`).

### Implementation
```python
out_A = model(input_ids=traj_ids, return_logits=True)
logp_A = F.log_softmax(out_A.logits / 0.7, dim=-1)
logp_B = F.log_softmax(out_A.logits / 1.5, dim=-1)   # reuse logits!
js = jensen_shannon(logp_A, logp_B).mean()
```
**Key trick:** the two passes share *the same logits*, only the softmax temperature differs — true cost is **1.0×** plus a softmax. I list 1.1× to leave headroom.

### Pareto position
Between `lp_min` and `prm_min`, likely *dominating* `lp_min` because it strictly adds geometric information at <10% extra cost. Will not match `sc_top1` on hard problems where the issue is *which path* not *how confident the path is*.

### Risks
- Models with very low intrinsic entropy (heavily distilled) make all JS values tiny and noisy. Mitigation: standardize per-model.
- The choice of `(τ_A, τ_B)` is hyperparameter-sensitive; recommend grid `(τ_dep, 2·τ_dep)`.

---

## 4. `mutual_info_step_to_answer` — IB score from hidden states

### Definition
Let `h_t ∈ R^d` be the final-token hidden state of step `s_t` (last layer, post-norm). Let `y` be the model's *predicted* final answer (the tokens inside `\boxed{...}`). Train — once, offline, on a held-out calibration split — a small probe `q_φ(y | h_t)` that predicts the answer distribution from `h_t`. At test time:
  `MI_t ≈ log q_φ(ŷ | h_t) - log q_φ(ŷ)` (PMI of predicted answer under the probe)
where `ŷ` is the trajectory's actual final answer. Aggregate:
  `score = mean_{t ≥ T/2} MI_t`
(later half of the trajectory, where the answer should be increasingly determined).

### Compute
**1.0×** at inference (probe is a 2-layer MLP, < 1ms per step). Probe training is one-time and not counted against per-trajectory compute. Storing hidden states adds negligible memory if we keep one per step (T·d ≈ 20 · 4096 floats).

### Intuition
This is the **information bottleneck** view of CoT: a correct chain progressively concentrates information about the final answer in its hidden states. A spurious chain produces hidden states whose answer-predictiveness saturates early or oscillates. The probe `q_φ` operationalizes "how much of `y` is already encoded in `h_t`". A correct trajectory shows monotone-increasing PMI; an incorrect one is flat or non-monotone.

This score is fundamentally different from `prm_min` — PRM is trained on *step quality* labels (PRM800K-style), whereas this probe is trained only on *(hidden state, final-answer)* pairs from the same model's own outputs. **No human labels needed.**

### Implementation
```python
# Offline (one time, on calibration set):
H = collect_step_hidden_states(model, calib_traj)        # (N_steps, d)
Y = collect_predicted_answers(calib_traj)                 # (N_traj,)
probe = train_mlp(H, Y_per_step)                          # cross-entropy

# At test time:
mi_per_step = [log probe(h_t)[y_hat] - log marginal[y_hat] for h_t in step_hs]
score = np.mean(mi_per_step[len(steps)//2:])
```
Requires `output_hidden_states=True` in HF or vLLM's `--return-hidden-states` flag.

### Pareto position
Comparable cost to `lp_min` but dramatically more semantic. Could *outperform* `prm_min` because PRMs trained on PRM800K may be domain-mismatched, while this probe is in-distribution by construction. Will not match `sc_top1` on problems with multiple plausible final answers.

### Risks
- Probe overfits to surface answer formatting. Mitigation: tokenize `\boxed{}` content canonically (numeric normalization).
- Domain shift between calibration and test (e.g. AIME ↔ MATH-500): retrain or share a shared problem-feature embedding.
- If the model's internal representation only encodes the answer in the last 1–2 hidden states, the score collapses to "is the final hidden state crisp" — which is barely better than `lp_min`. Diagnostic: plot MI_t vs t.

---

## 5. `lookahead_branching_entropy` — entropy of distribution over *answers* via cheap rollouts

### Definition
At a small set of *fork points* `{t_1, …, t_K}` (e.g. K=3, evenly spaced through the trajectory), branch off `M` short continuations of length `L_short` (e.g. 32 tokens) each, *but only completing to the answer*. Concretely, at fork `t_k`:
- prefix = trajectory tokens up to step `t_k`
- sample `M` completions with `temperature=0.8`, `max_tokens=L_short`, stopping at `\boxed{}` or step boundary
- extract candidate answers `{ŷ_k^{(1)}, …, ŷ_k^{(M)}}` (canonicalized; if no answer extracted, mark "?")

Compute the empirical answer distribution `p̂_k(y)` and its Shannon entropy `H_k = H(p̂_k)`. Aggregate:
  `score = -mean_k H_k`.

### Compute
**~1.5–2.5×** depending on `(K, M, L_short)`. With `K=3, M=4, L_short=64` and the main trajectory length ~400 tokens, the extra tokens are `3·4·64 = 768`, roughly 2× compute. KV-cache reuse from the trunk makes the prefill nearly free; only decode tokens are paid for.

### Intuition
This is **information about the answer-marginal**, not about the next token. `sc_top1` does this only at `t = 0` (full-length samples from scratch); `lookahead_branching_entropy` does it *along the trajectory* with much shorter completions. If the trajectory is on a correct path, the answer distribution `p̂_k` collapses early — low entropy from `t = T/3` onward. If wrong, the entropy stays high or jumps around.

### Implementation
```python
score_terms = []
for t_k in fork_points(traj, K=3):
    prefix_kv = cached_kv_up_to(t_k)
    completions = vllm.generate(
        prefix_kv, n=M, temperature=0.8,
        max_tokens=L_short, stop=["\\boxed", "\n\n"]
    )
    answers = [extract_boxed(c) for c in completions]
    p_hat = empirical_dist(answers)
    score_terms.append(-entropy(p_hat))
score = np.mean(score_terms)
```
The `stop` token list must be configured carefully so the model emits the answer without rambling.

### Pareto position
Sits **between `prm_min` and `sc_top1`**. Strictly dominates `prm_min` when `M` is small (1.5×), and approaches `sc_top1` performance at fewer total tokens because each rollout starts from a useful prefix instead of from scratch. Fills the major gap in the existing frontier.

### Risks
- "Truncated rollouts don't reach the answer" — must tune `L_short` per problem family. Fix: dynamic stopping when `\boxed` appears, falling back to "no answer" bucket.
- Self-consistency-of-self-consistency: if the model is systematically biased toward a wrong answer, all `M` rollouts agree wrongly; entropy is 0 but correctness is 0. This is a fundamental limit shared by `sc_top1`.
- Fork point selection: too early, the trunk hasn't constrained the answer; too late, no diversity left. Recommend forking at `t ∈ {T/3, T/2, 2T/3}`.

---

## 6. Ensemble note

These five scores are designed to be **information-orthogonal**: #1 captures per-token uncertainty, #2 trajectory dynamics, #3 geometric/perturbation sensitivity, #4 answer-conditional encoding in hidden states, #5 answer-marginal collapse. A weighted z-score sum should dominate any single component on the Pareto plot, at the cost-sum of components.

---

## 7. Summary table

| # | Name | Cost | Frontier slot | Key requirement |
|---|---|---|---|---|
| 1 | `step_entropy_p95` | 1.0× | between `lp_min` and `prm_min` | `logprobs=20` |
| 2 | `predictive_sharpening_rate` | 1.0× | orthogonal to `lp_min` | `logprobs=20` |
| 3 | `tempered_kl_divergence` | ~1.1× | dominates `lp_min` | full logits, two τ |
| 4 | `mutual_info_step_to_answer` | 1.0× + offline probe | competes with `prm_min` | hidden states + probe |
| 5 | `lookahead_branching_entropy` | 1.5–2.5× | between `prm_min` and `sc_top1` | KV-cache prefix branching |

All five are **information-theoretic** (entropy, KL, MI, predictive distribution geometry) and exploit data already produced by — or cheaply re-extractable from — a single greedy decode. None requires retraining the base LLM. Scores 1, 2, 3 should be implementable in a single afternoon on top of `pilot10_extended_cp.py`; score 4 needs a one-time probe-training pilot (call it `pilotP_ib_probe.py`); score 5 needs careful KV-cache plumbing — recommend prototyping in `pilotQ_lookahead_branch.py`.

---

*— Agent A (info-theoretic angle), cross-reference with Agents B–E for orthogonal proposals.*

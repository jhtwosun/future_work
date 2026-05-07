# Agent B (internal-state angle): Internal-State Score Functions for CoT-CP

**Author:** Agent B (internal-state angle)
**Date:** 2026-05-05
**Scope:** Five new trajectory-level nonconformity scores for CoT-CP, grounded in hidden states, attention, and layer-wise probes. All slot into the same split-CP pipeline as `lp_min`, `prm_min`, `sc_top1`, and target the cheap-to-medium cost band (1.02× – 3× greedy).

## Notation

CoT trajectory = reasoning steps $s_1, \ldots, s_K$ (split on `\n\n`), each with tokens $t_{k,1}, \ldots, t_{k,T_k}$. For position $t$ and layer $\ell \in \{1, \ldots, L\}$:
- $h_t^{(\ell)} \in \mathbb{R}^d$ — residual-stream hidden state
- $A_{t,h}^{(\ell)}$ — attention distribution at head $h$
- $p_t = \mathrm{softmax}(W_U h_t^{(L)})$ — next-token distribution

vLLM ≥ 0.19 exposes per-token logprobs natively; hidden states via the offline `LLM` interface with hidden-state capture (or HF `output_hidden_states=True`). Attentions need HF eager fallback on Qwen/Phi.

---

## Score 1: `hidden_drift_max`

**Definition.**

Let $\bar h_k = \frac{1}{T_k} \sum_t h_t^{(L-1)}$ be the mean penultimate-layer residual over the tokens of step $k$. Define step-to-step drift

$$
d_k = 1 - \cos(\bar h_{k-1}, \bar h_k), \quad k = 2, \ldots, K
$$

Score:

$$
S_{\text{drift}}(\text{traj}) = \max_{k=2,\ldots,K} d_k
$$

(higher score = more nonconforming → "the trajectory had a sudden representational jump").

**Compute.** ≈ 1.05× greedy. Hidden states fall out of the forward pass; pulling layer $L{-}1$ and averaging is one small vector per step. Overhead is just the hidden-state capture path.

**Intuition.** Correct CoTs evolve smoothly through representation space — each step refines the previous abstract state. When a model bluffs (e.g. invents an algebraic identity), the residual trajectory *snaps* to a different neighborhood, producing an anomalously low cosine. LookbackLens (Chuang et al. 2024) and SAPLMA-style work show representational discontinuities correlate with hallucinated content even at high token logprobs.

**Implementation.** vLLM: `LLM(..., enable_hidden_states=True)`, then `RequestOutput.hidden_states[-2]`. Penultimate layer beats final (final is dominated by unembedding shaping). For Qwen2/Llama3 use $L{-}1$; for Phi-3 try $L{-}2$.

**Pareto.** Slightly above `lp_min` (≈1.05×); should beat `lp_min` on confidently-wrong CoTs. Strictly below `prm_min` (no second model).

**Risks.** (i) Step segmentation: if $K{=}1$ the score is undefined (sentinel fallback). (ii) MoE models (Mixtral, DSv3) route tokens through different experts → residual cosine is noisier on the post-MoE stream. (iii) Legitimate "case split" steps can spike $d_k$; mixing with `lp` weight helps.

---

## Score 2: `attn_entropy_focus`

**Definition.**

For token $t$ in step $k$, head $h$ at layer $\ell$, define entropy $H_{t,h}^{(\ell)} = -\sum_{j<t} A_{t,h,j}^{(\ell)} \log A_{t,h,j}^{(\ell)}$. Let $\mathcal{H}^{\star}$ be a fixed pre-selected set of "retrieval heads" (heads that, on a probe set, attend strongly to prior numerical / variable tokens — identifiable via Wu et al. 2024's retrieval-head method). Define step-level mean entropy over retrieval heads, last layer-quartile $\ell \in [\tfrac{3L}{4}, L]$:

$$
E_k = \frac{1}{|\mathcal H^\star| \cdot |\{t \in s_k\}| \cdot |\ell|} \sum_{h \in \mathcal H^\star} \sum_{\ell} \sum_{t \in s_k} H_{t,h}^{(\ell)}
$$

Score:

$$
S_{\text{attn}}(\text{traj}) = \max_k E_k
$$

**Compute.** ≈ 1.15× greedy. vLLM 0.19 doesn't expose per-head attention; we one-shot replay completed trajectories in HF eager mode (single forward, no autoregressive loop) — ≈0.15× original decode cost.

**Intuition.** When confused, retrieval heads "spray" across the prompt instead of locking onto relevant prior numbers/variables. Lookback Lens showed that the ratio of attention to context vs. recently-generated tokens predicts hallucination; we adapt to per-step entropy over a curated head subset.

**Implementation.** Pre-compute $\mathcal H^\star$ once per model on a 200-example needle-in-haystack copying probe (top-5% recall heads). Save as `retrieval_heads_{model}.json`. At score time, HF replay gathers attentions only for these heads.

**Pareto.** ~1.15×, well below `prm_min` (~2×). Strong on problems where the model loses track of variables (algebra word problems, multi-step AIME).

**Risks.** (i) Retrieval-head set is *checkpoint-specific*; Llama-3-8B heads don't transfer to Qwen2.5-7B (different GQA grouping). Re-discover per checkpoint. (ii) MoE: heads are shared but post-routing residuals shift which heads "matter". (iii) Long contexts: entropy scales with $\log t$ — normalize by length or use length-matched baseline.

---

## Score 3: `wrongness_probe_max`

**Definition.**

Pre-train a small linear probe $w_\ell \in \mathbb{R}^d$, $b_\ell \in \mathbb{R}$ on a labeled set of (correct, incorrect) trajectories from a held-out training pool *disjoint from the CP calibration set*:

$$
\hat p_{\text{wrong}}(t, \ell) = \sigma(w_\ell^\top h_t^{(\ell)} + b_\ell)
$$

trained with logistic loss to predict `final_answer_wrong ∈ {0,1}` from the *step-end* hidden state. Use the best-performing layer $\ell^\star$ (cross-validated). Step-level pooling:

$$
q_k = \hat p_{\text{wrong}}(\text{last token of }s_k,\, \ell^\star)
$$

Score:

$$
S_{\text{probe}}(\text{traj}) = \max_k q_k
$$

**Compute.** ≈ 1.02× greedy inference (one matrix-vector per step). One-time training: ~5 min on 5k labeled trajectories. Probe weights ~30 KB.

**Intuition.** Math-reasoning adaptation of SAPLMA (Azaria & Mitchell 2023) and INSIDE (Chen et al. 2024): information about "I'm about to be wrong" is *linearly decodable* from middle-to-late residuals *before* the final answer, even when text reads confident. Probing at every step end (not just final-answer) gives early warning.

**Implementation.** HF `output_hidden_states=True` → slice last-token hidden at each step delimiter. Train with `sklearn.LogisticRegression` (L2 tuned). **Critical for CP validity:** probe training set must be disjoint from CP calibration set — train on GSM8K + MATH-train, calibrate on MATH-500 calibration split.

**Pareto.** Cheapest of the five at inference (~1.02×), with a one-time training cost. If accuracy is competitive with `prm_min`, strictly dominates it.

**Risks.** (i) Distribution shift: a probe trained on MATH degrades on AIME (cf. `pilotB_ood_aime` patterns). Mitigation: train on mixture, report worst-domain. (ii) Linearity may miss nonlinear modes; small MLP extension is trivial. (iii) Most checkpoint-specific of the five — cross-model transfer ≈ 0. (iv) Keep linear to avoid memorization.

---

## Score 4: `next_step_disagreement`

**Definition.**

At each step boundary $k$ (right after the newline ending step $s_k$), branch $M$ short rollouts ($\sim$32 tokens each) at temperature $T=0.7$. Embed each rollout's hidden state at its terminal token (penultimate layer, mean over its tokens) → $\{r_k^{(m)}\}_{m=1}^M \subset \mathbb R^d$. Define rollout disagreement:

$$
D_k = \mathrm{tr}\!\left(\widehat{\mathrm{Cov}}(r_k^{(1)}, \ldots, r_k^{(M)})\right) \big/ \|\bar r_k\|^2
$$

(coefficient-of-variation-style scalar). Score:

$$
S_{\text{nsd}}(\text{traj}) = \max_k D_k
$$

**Compute.** ≈ $1 + M \cdot 32 / \bar T_k$ × greedy. With $M{=}4$, $\bar T_k \approx 64$ → ~3× — between `prm_min` and `sc_top1` ($N{=}8$ ≈ 8×). Cascade-gated (only branch when local `lp_min` is in bottom quartile) brings average to ~1.5×.

**Intuition.** "Self-consistency in representation space." `sc_top1` requires *finishing* full CoTs; this asks at each step $k$: do short alternative continuations agree on *where the model is heading* in latent space? Disperse → high-entropy decision point. Bonus: uncertainty is localized to specific steps (useful for downstream revise/abstain).

**Implementation.** vLLM ≥ 0.19 forks via `n=M` from prefix cache (prefill free across rollouts), capture hidden states on the short rollouts. Keep only terminal hidden per rollout.

**Pareto.** ~2.5–3× naive, ≲`prm_min` with cascade. Strong on AIME-style problems where local branching is decisive.

**Risks.** (i) 32 tokens may not propagate divergence — tune length per model (R1-distill rambles → longer). (ii) MoE: branched decode duplicates expert activations, KV-cache pressure. (iii) Embedding choice (last vs. mean) is finicky — mean is more robust to "wait, let me reconsider" cliffhangers.

---

## Score 5: `layer_consistency_disagreement`

**Definition.**

For each step-final token, compute the "early exit" prediction at multiple layers via the unembedding (Logit-Lens / Tuned-Lens):

$$
p_t^{(\ell)} = \mathrm{softmax}(W_U \cdot \mathrm{LN}(h_t^{(\ell)}))
$$

For a chosen window of layers $\mathcal L = \{L-4, L-3, L-2, L-1\}$, define inter-layer JS divergence at step end:

$$
J_k = \frac{1}{\binom{|\mathcal L|}{2}} \sum_{\ell < \ell' \in \mathcal L} \mathrm{JSD}\!\left(p_{t_k}^{(\ell)} \,\|\, p_{t_k}^{(\ell')}\right)
$$

where $t_k$ is the last token of step $k$. Score:

$$
S_{\text{lcd}}(\text{traj}) = \max_k J_k
$$

**Compute.** ≈ 1.10× greedy. One unembedding per layer in $\mathcal L$ per step boundary (not per token). For a 32-step CoT, $|\mathcal L|{=}4$: 128 small matmuls + 6 JSDs. Trivial vs. decode.

**Intuition.** When a transformer "knows" the answer, late layers converge — prediction stabilizes by $L{-}3$ ("late-layer plateau" in Tuned-Lens). When guessing, distributions stay volatile: $L{-}3$ predicts "7", $L{-}1$ predicts "9". This volatility is invisible to `lp_min` (which sees only $L$).

**Implementation.** Tuned-Lens weights public for Llama-2/3 and Pythia; for Qwen/Phi train ~30 min, or fall back to raw Logit-Lens. vLLM hidden-state capture; unembed offline.

**Pareto.** ~1.10× and mostly orthogonal to `lp_min`. Strong candidate for combination: $S_{\text{combo}} = \alpha (-\text{lp\_min}) + (1-\alpha) S_{\text{lcd}}$.

**Risks.** (i) Logit-Lens quality varies — small models (Phi-3-mini) have poorly-calibrated mid-layers. (ii) RL-tuned reasoning models (R1-distill) may suppress disagreement (late layers pushed to over-commit). (iii) MoE: per-token routing makes layers live in different subspaces; $W_U$ is still shared, but interpret JSD cautiously.

---

## Combining with existing scores

All five are scalar and plug into split-CP. Combinations:
- **Convex:** $S_{\text{mix}} = \sum_i w_i \tilde S_i$ on rank-normalized $\tilde S_i$. Tune weights for CP-coverage-conditional accuracy on a held-out split. Isotonic-rerank Score 3 (probe) before mixing — its scale dominates otherwise.
- **Cascade:** `lp_min` first; invoke Score 4 (`next_step_disagreement`) only on bottom-quartile-confidence subset. Pareto sweetener.
- **Margin combo:** $S_{\text{lp+drift}} = -\text{lp\_min} + \lambda S_{\text{drift}}$ — agreement is decisive; disagreement flags abstention.

---

## Generalization across model families

| Score | Qwen2.5 | Llama3 | Phi-3 | MoE (Mixtral / DSv3) |
|---|---|---|---|---|
| `hidden_drift_max` | Strong | Strong | Strong | Workable (use post-MoE residual) |
| `attn_entropy_focus` | Needs head re-ID | Needs head re-ID | Needs head re-ID; few heads | Heads shared, but routing noise |
| `wrongness_probe_max` | Per-checkpoint train | Per-checkpoint | Per-checkpoint | Per-checkpoint, plus expert-conditioned probe possibly needed |
| `next_step_disagreement` | Strong | Strong | OK (small models noisier) | KV memory pressure |
| `layer_consistency_disagreement` | Need tuned lens | Tuned lens available | Need tuned lens | Caveat on per-token routing |

**Most-portable (run first):** `hidden_drift_max` — only mean-pooled hidden states, zero per-model calibration beyond CP itself.
**Most-promising-but-finicky:** `wrongness_probe_max` — likely the best raw discriminator, but does not transfer across checkpoints.
**Most-novel:** `next_step_disagreement` — latent-space self-consistency is not standard in CP-for-LLM and gives step-localized uncertainty `sc_top1` cannot.

## Concrete pilot plan

`pilotP_internal_state.py`: (a) reuse existing MATH-500 trajectories from `pilot7_math500_harder.py` and `pilot10_extended_cp.py`; (b) HF-eager replay to capture hidden states + attentions (one forward, ~0.2× original decode); (c) compute all five scores; (d) feed into the CP pipeline from `pilot8_cp_simulation.py`; (e) Pareto curves vs. `lp_min` / `prm_min` / `sc_top1`. If vLLM 0.19 hidden-state capture is reliable on Qwen2.5-7B and R1-distill-7B (the two checkpoints already used in pilots), skip the replay entirely.

Cross-reference Agents A and C: my five are deliberately positioned in the 1.02× – 3× band to fill the `lp_min`-to-`prm_min` gap on the existing Pareto frontier.

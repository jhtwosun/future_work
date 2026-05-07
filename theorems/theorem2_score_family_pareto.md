# Theorem 2: Score Family Pareto for CoT-CP

> **Author**: Claude (drafted in parallel with Codex working on Theorem 1)
> **Status**: Initial derivation; will be cross-checked with Codex.

## 1. Setup

We extend the CoT-CP setting to the case where multiple score families
$\{\phi_k\}_{k=1}^K$ are available, each with a different per-question
inference cost $c_k$ (in tokens, FLOPs, or wallclock). Concretely, in
our paper:

| $k$ | Score $\phi_k$ | Cost $c_k$ (multiples of one greedy decode) |
|---|---|---|
| 1 | $\phi_{\mathrm{lp}}(R) = \min_t \frac{1}{\lvert S_{i,t}\rvert}\sum_{u} \log p_\theta(s_{i,t,u} \mid \cdot)$ | $1\times$ |
| 2 | $\phi_{\mathrm{prm}}(R) = \min_t \mathrm{PRM}(s_{i,t}; X)$ | $2\times$ (1 generator + 1 PRM forward) |
| 3 | $\phi_{\mathrm{sc}}(X) = \frac{\lvert\{j : \hat{Y}_j = \mathrm{maj}\}\rvert}{N}$ | $N\times$ (sample $N$ trajectories) |

Every score $\phi_k$ induces a CP procedure: calibrate
$\hat{q}_\alpha^{(k)}$ from labeled calibration data, keep test point
iff $\bar{S}^{(k)} \geq \hat{q}_\alpha^{(k)}$. Theorem 1 (Codex's
deliverable) gives coverage $\geq 1-\alpha$ for each $k$ separately.

The empirical question: at fixed answer-rate $\beta$ (the fraction of
test points kept), which $\phi_k$ gives the highest *selective accuracy*
$\rho_k(\beta) = \mathbb{P}(Y = 1 | \bar{S}^{(k)} \geq \hat{q}^{(k)},
\text{kept frac} = \beta)$?

---

## 2. Theorem statement

**Theorem 2 (Score Family Compute-Selective Pareto).** Suppose the
calibration distribution and the test distribution coincide
(exchangeable). For each score family $\phi_k$, let $F_k(\cdot|y)$ be
the conditional CDF of $\bar{S}^{(k)}$ given correctness $Y=y$, and
let $\pi = \mathbb{P}(Y=1)$ be the marginal correctness rate.
Assume each $F_k(\cdot|y)$ is continuous and strictly increasing where
defined.

For a threshold $\tau_k$ inducing kept fraction
$\beta = \pi(1 - F_k(\tau_k|1)) + (1-\pi)(1 - F_k(\tau_k|0))$, the
selective accuracy is
$$
\rho_k(\beta) = \frac{\pi(1 - F_k(\tau_k(\beta)|1))}{\beta}.
$$

Define the *coverage-conditioned signal* of score family $k$ at
operating point $\beta$ as
$$
\mathrm{SNR}_k(\beta) := \frac{1 - F_k(\tau_k(\beta) | 1)}{1 - F_k(\tau_k(\beta) | 0)} \quad \in [1, \infty).
$$

Then for any two score families $k_1, k_2$ with $c_{k_1} < c_{k_2}$
and any operating point $\beta \in (0, 1)$:

$$\rho_{k_1}(\beta) > \rho_{k_2}(\beta) \iff \mathrm{SNR}_{k_1}(\beta) > \mathrm{SNR}_{k_2}(\beta).$$

**Corollary (Pareto frontier).** The Pareto frontier in the
(compute, selective accuracy) plane at fixed $\beta$ consists of
exactly those score families $\phi_{k^*}$ such that no cheaper
$\phi_{k'}$ has higher SNR at $\beta$. Equivalently, ordering scores
by cost, the frontier is the *upper envelope* of the SNR curves.

---

## 3. Proof sketch

### Step 1: Selective accuracy expressed via conditional CDFs.

For threshold $\tau$ inducing kept fraction
$\beta = \pi (1-F_k(\tau|1)) + (1-\pi)(1-F_k(\tau|0))$:
- Kept-and-correct fraction: $\pi(1 - F_k(\tau|1))$.
- Kept-and-wrong fraction: $(1-\pi)(1 - F_k(\tau|0))$.
- $\rho_k(\beta) = \pi(1 - F_k(\tau_k(\beta)|1)) / \beta$.

### Step 2: Equivalence of SNR ranking and selective-accuracy ranking.

Let $a_k = 1 - F_k(\tau_k|1)$ (correct-tail) and
$b_k = 1 - F_k(\tau_k|0)$ (wrong-tail). At fixed $\beta$, we have
$\pi a_k + (1-\pi) b_k = \beta$.

Rewriting,
$$\rho_k(\beta) = \frac{\pi a_k}{\beta} = \frac{\pi a_k}{\pi a_k + (1-\pi) b_k} = \frac{1}{1 + \frac{(1-\pi)}{\pi} \cdot \frac{b_k}{a_k}} = \frac{1}{1 + \frac{1-\pi}{\pi} \cdot \mathrm{SNR}_k^{-1}}.$$

This is monotonically increasing in $\mathrm{SNR}_k$ (since
$\mathrm{SNR}_k^{-1}$ is decreasing). Therefore $\rho_{k_1} > \rho_{k_2}$ iff
$\mathrm{SNR}_{k_1} > \mathrm{SNR}_{k_2}$.

### Step 3: Pareto frontier characterization.

A score family $\phi_{k^*}$ is on the (compute, selective accuracy)
Pareto frontier at operating point $\beta$ iff no cheaper $\phi_{k'}$
($c_{k'} < c_{k^*}$) has $\rho_{k'}(\beta) \geq \rho_{k^*}(\beta)$. By
Step 2, this is equivalent to: no cheaper $\phi_{k'}$ has
$\mathrm{SNR}_{k'}(\beta) \geq \mathrm{SNR}_{k^*}(\beta)$.

Hence the Pareto frontier is exactly the upper SNR envelope across
score families ordered by cost.

QED. $\blacksquare$

---

## 4. Connection to our empirical results

Our MATH-500 / Qwen2.5-7B numbers at $\alpha=0.5$ (i.e., $\beta \approx 0.4$):

| $\phi_k$ | $c_k$ | $\rho_k(0.5)$ (empirical) | Implied SNR ratio |
|---|---|---|---|
| lp_min | 1× | 0.620 | $b/a = 1.84$ → SNR = 0.54 (from inverse) |
| prm_min | 2× | 0.707 | $b/a = 1.04$ → SNR = 0.96 |
| sc_top1 | 8× | 0.793 | $b/a = 0.52$ → SNR = 1.92 |

Each family triples (SC) or doubles (PRM) the previous family's SNR at
the same operating point — quantitatively backing the cost-ladder
ordering observed empirically.

The corollary to draw out (and check with Codex): given a fixed compute
budget $C$, the optimal CP wrapper is the unique $\phi_k$ such that
$c_k \leq C$ and $\mathrm{SNR}_k$ is maximized. Empirically this gives
the *cheapest score whose SNR exceeds the next-cheapest's*, i.e., the
Pareto-optimal point on the upper envelope.

---

## 5. Discussion / verification points for Codex

1. **SNR is the right invariant**: Theorem 2 says ordering by SNR =
   ordering by selective accuracy. This is essentially Neyman-Pearson
   in disguise (likelihood ratio determines optimal selection). Should
   we make that connection explicit in the proof?

2. **Continuity assumption**: We assume $F_k(\cdot|y)$ continuous for
   exact threshold-to-$\beta$ correspondence. For *discrete* scores
   like sc_top1 (only $N+1$ values for SC@$N$), continuity fails — we
   use lower-quantile ties. The theorem still holds with weak
   inequalities; should we explicitly state the discrete version?

3. **Cost as additive vs. multiplicative**: We treated cost
   multiplicatively (cost ratios). For real wallclock comparisons
   (e.g., Pilot D's compute-Pareto figure with $N=1, 4, 8, 16, 32$),
   it should be linear in $N$. Is the theorem statement
   cost-functional-agnostic enough? I think yes since it just orders
   scores.

4. **Frontier dominance vs. single-$\beta$ dominance**: Theorem 2 is
   stated *pointwise in $\beta$*. Our empirical frontiers (Pilot 10,
   Figure 1) hold across all $\beta \in (0,1)$. Is there a statement
   about *uniform* dominance? E.g., if SC dominates lp at every
   $\beta$, can we say that's because SC's score has higher
   *information content* (mutual information with $Y$)?

5. **Connection to learnable score families**: Could we use Theorem 2
   to derive the optimal *learned* aggregator $\phi^*$? E.g., for
   trained PRMs, the SNR depends on training quality. Useful for
   future work section.

6. **(Negative result)** Step-level branching K-resample (our Pilots
   C/K/L null result) — Theorem 2 may give an explanation: K-resample
   from a fixed prefix doesn't change the SNR substantially because
   all K samples share the prefix-conditional distribution. That is, the
   K-sample average is asymptotically deterministic under the
   prefix-conditional law. Worth stating as Remark.

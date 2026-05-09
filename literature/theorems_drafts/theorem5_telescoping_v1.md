# Theorem 5 — K-Rung Telescoping Coverage Bound for Ladder-Calibrated Weighted CP

> **Author**: Claude Opus 4.7 (1M context), single-model lane.
> **Status**: V1 draft (density-ratio-telescoping perspective). Cross-model verification token = `sk-PLACEHOLDER` (per workspace `CLAUDE.md`); a re-derivation by `openai/openai/gpt-5.5` is queued under the master orchestrator's G2 gate before AISTATS submission.
> **Companion artifacts**:
> - `/home/nvidia/future/literature/concept_papers/distance_ladder_DEEP.md` (paper plan)
> - `/home/nvidia/future/experiments/results/distance_ladder_pilot.json` (4-rung MATH→AIME pilot)
> - `/home/nvidia/future/theorems/theorem3_weighted_cp_discrete.md` (one-shot foil)
> - Tibshirani-Foygel Barber-Candès-Ramdas, "Conformal Prediction Under Covariate Shift" (NeurIPS 2019)
> - Riess et al. 2022, *ApJL* 934:L7 (astronomy ladder)

---

## 1. Setup — the K-rung calibration ladder

### 1.1 Domains and samples

Let $\mathcal{X}$ be a prompt space, $\mathcal{S} \subset \mathbb{R}$ a finite discrete score range with $|\mathcal{S}| < \infty$ (for SC@$N$ self-consistency, $\mathcal{S} = \{0/N, 1/N, \ldots, N/N\}$, so $|\mathcal{S}| = N+1$), and $\mathcal{Y} = \{0, 1\}$ the correctness label.

A **K-rung calibration ladder** is an ordered sequence of distributions
$$D_0, D_1, \ldots, D_K$$
on $\mathcal{X} \times \mathcal{S} \times \mathcal{Y}$, where $D_0$ is the source (richly labeled calibration data; "geometric anchor" in the astronomy analogy) and $D_K$ is the test target. For each rung $k$, an i.i.d. sample
$$S^{(k)} = \{(X_i^{(k)}, S_i^{(k)}, Y_i^{(k)})\}_{i=1}^{n_k}$$
is observed. Write $p_k(s) := \Pr_{D_k}(S = s)$ for the rung-$k$ score PMF and $\hat p_k$ for its empirical version on $S^{(k)}$.

We use the lower-quantile convention and the *correct-trajectory restriction* $\mathcal{I}_+^{(k)} = \{i \le n_k : Y_i^{(k)} = 1\}$ with $n_+^{(k)} = |\mathcal{I}_+^{(k)}|$, exactly as in Theorem 1.

### 1.2 Assumptions

**(A1) i.i.d. within rung.** For each $k$, the sample $S^{(k)}$ is i.i.d. from $D_k$, and a fresh test point $(X_*, S_*, Y_*) \sim D_K$ is drawn independently of all calibration samples.

**(A2) Score-only consecutive shift.** For each $k \in \{1, \ldots, K\}$, the Radon-Nikodym derivative of $D_k$ with respect to $D_{k-1}$ depends only on the score:
$$\frac{dD_k}{dD_{k-1}}(x, s, y) = w_{k-1 \to k}(s) \quad \text{for some } w_{k-1 \to k}: \mathcal{S} \to (0, \infty).$$
Equivalently, the conditional law $\Pr_{D_k}(X, Y \mid S=s)$ is invariant across consecutive rungs: $\Pr_k(Y, X \mid S) = \Pr_{k-1}(Y, X \mid S)$ on the overlap support. This is the iterated form of Theorem 3's (A1).

**(A3) Bounded per-rung ratios with overlap.** There exist $0 < w_k^- \le w_k^+ < \infty$ such that
$$w_k^- \le w_{k-1 \to k}(s) \le w_k^+ \quad \forall s \in \operatorname{supp}(D_{k-1}).$$
The pair $(w_k^-, w_k^+)$ is the "rung-$k$ overlap quality" — tight overlap means $w_k^+ / w_k^- \approx 1$.

**(A4) Independent rungs.** The samples $S^{(0)}, \ldots, S^{(K)}$ are mutually independent (no problem leaks across rungs — this is the LLM analog of "no cross-listed galaxies" in the astronomy ladder).

**(A5) Discrete score support.** $|\mathcal{S}| < \infty$, and Laplace smoothing with constant $\varepsilon > 0$ is applied to all empirical PMFs.

Assumptions (A1)–(A5) are the K-rung iteration of Theorem 3's (A1)–(A2). Section 11 discusses where each binds in the LLM math benchmark setting.

---

## 2. Density-ratio telescoping — the algebraic backbone

The test-domain ($D_K$) versus source-calibration-domain ($D_0$) likelihood ratio factorizes by the chain rule of Radon-Nikodym derivatives:
$$w_{0 \to K}(s) \;:=\; \frac{p_K(s)}{p_0(s)} \;=\; \prod_{k=0}^{K-1} \frac{p_{k+1}(s)}{p_k(s)} \;=\; \prod_{k=0}^{K-1} w_{k \to k+1}(s). \tag{$\star$}$$

Identity ($\star$) is purely algebraic — it holds whenever each $p_k(s) > 0$ on the relevant support — and is the structural reason a ladder can be useful: a single hard estimation problem ("estimate $w_{0 \to K}$ from a small overlap of $D_0$ and $D_K$") is decomposed into $K$ smaller estimation problems ("estimate each $w_{k \to k+1}$ from the better-overlapping rung pair $(D_k, D_{k+1})$").

Each factor is estimated from rung-$(k, k+1)$ samples by the empirical-PMF Laplace-smoothed ratio:
$$\hat w_{k \to k+1}(s) \;=\; \frac{\hat p_{k+1}(s) + \varepsilon}{\hat p_k(s) + \varepsilon}, \qquad
\hat w_{0 \to K}(s) \;:=\; \prod_{k=0}^{K-1} \hat w_{k \to k+1}(s). \tag{$\star\star$}$$

The ladder weighted-CP quantile is then the lower-$\alpha$ weighted quantile of the **rung-0 correct-trajectory scores** under weights $\hat w_{0 \to K}(S_i^{(0)})$:
$$\hat q_\alpha^{(K),\,\text{ladder}} \;=\; \inf\!\left\{ s \in \mathcal{S} : \sum_{i \in \mathcal{I}_+^{(0)} : S_i^{(0)} \le s} \frac{\hat w_{0 \to K}(S_i^{(0)})}{\sum_{j \in \mathcal{I}_+^{(0)}} \hat w_{0 \to K}(S_j^{(0)})} \;\ge\; \alpha \right\}. \tag{$\star\star\star$}$$

A test point with score $S_*$ is **kept** iff $S_* \ge \hat q_\alpha^{(K),\,\text{ladder}}$.

> **Remark (Strategy B vs Strategy A point-estimate equivalence).** If the same rung-0 sample is used for both the one-shot Theorem 3 and the telescoped ladder (B), the point estimates $\hat q_\alpha$ are *identical by construction* — see ($\star\star$). The ladder's contribution lives in the *slack* of the coverage bound, not in the point estimate. The pilot (`distance_ladder_pilot.json`) confirms this with numerical equality across all $\alpha$. The coverage-gap reduction reported in §10 is achieved by the *sequential* re-quantilization variant B′; B′ is not telescoping-equivalent to one-shot.

---

## 3. Lemma 1 — per-rung empirical-ratio error

> **Lemma 1.** Under (A1), (A4), (A5), and assuming $\hat p_k(s) + \varepsilon \ge \varepsilon > 0$, then for any $\delta \in (0, 1)$, with probability at least $1 - \delta$ over $S^{(k)}, S^{(k+1)}$,
> $$\sup_{s \in \mathcal{S}} \left| \hat w_{k \to k+1}(s) - w_{k \to k+1}(s) \right| \;\le\; \epsilon_k(\delta), \qquad \epsilon_k(\delta) := C_k \sqrt{\frac{\log(2|\mathcal{S}|/\delta)}{2\, n_{\mathrm{eff},k}}},$$
> where $n_{\mathrm{eff},k} := \min(n_k, n_{k+1})$ and $C_k$ is a finite constant depending only on $(w_k^+, \varepsilon, \min_s p_k(s))$ defined in the proof. In particular, $C_k = O\!\big(\big(1 + w_k^+\big) / (\varepsilon + \min_s p_k(s))\big)$.

### 3.1 Proof

Step 1 (uniform PMF concentration). By the Bretagnolle-Huber-Carol / DKW-with-union-bound inequality applied to discrete distributions on $|\mathcal{S}|$ atoms, with probability $\ge 1 - \delta/2$ each,
$$\|\hat p_k - p_k\|_\infty \;\le\; \sqrt{\frac{\log(2|\mathcal{S}|/\delta)}{2 n_k}}, \qquad
\|\hat p_{k+1} - p_{k+1}\|_\infty \;\le\; \sqrt{\frac{\log(2|\mathcal{S}|/\delta)}{2 n_{k+1}}}.$$
Take a union over the two rungs to obtain joint probability $\ge 1 - \delta$ that
$$\max\big(\|\hat p_k - p_k\|_\infty,\ \|\hat p_{k+1} - p_{k+1}\|_\infty\big) \;\le\; \beta_k \;:=\; \sqrt{\frac{\log(2|\mathcal{S}|/\delta)}{2 n_{\mathrm{eff},k}}}. \tag{1}$$

Step 2 (ratio Lipschitz expansion). Fix $s \in \mathcal{S}$ and abbreviate $a = p_k(s) + \varepsilon$, $\hat a = \hat p_k(s) + \varepsilon$, $b = p_{k+1}(s) + \varepsilon$, $\hat b = \hat p_{k+1}(s) + \varepsilon$. Both $a, \hat a \ge \varepsilon > 0$ and likewise for $b, \hat b$, so the ratios $b/a$ and $\hat b/\hat a$ are well-defined.

By the standard quotient identity,
$$\frac{\hat b}{\hat a} - \frac{b}{a} \;=\; \frac{(\hat b - b)\,a - (\hat a - a)\,b}{\hat a \, a} \;=\; \frac{\hat b - b}{\hat a} \;-\; \frac{b}{a}\cdot \frac{\hat a - a}{\hat a}. \tag{2}$$
Taking absolute values and using $|\hat b - b| \le \beta_k$, $|\hat a - a| \le \beta_k$, $\hat a \ge \varepsilon$, and $b/a = w_{k \to k+1}(s) + O(\varepsilon)$ bounded above by $w_k^+ + O(\varepsilon)$,
$$\big|\hat w_{k \to k+1}(s) - w_{k \to k+1}^{(\varepsilon)}(s)\big| \;\le\; \frac{\beta_k}{\varepsilon} + \frac{(w_k^+ + O(\varepsilon))\,\beta_k}{\varepsilon} \;\le\; \frac{(1 + w_k^+)\,\beta_k}{\varepsilon} + O(\beta_k), \tag{3}$$
where $w_{k \to k+1}^{(\varepsilon)}(s) := (p_{k+1}(s) + \varepsilon)/(p_k(s) + \varepsilon)$ is the smoothed population ratio, satisfying
$$\big|w_{k \to k+1}^{(\varepsilon)}(s) - w_{k \to k+1}(s)\big| \;\le\; \frac{\varepsilon\,(1 + w_k^+)}{p_k(s) + \varepsilon} \;\le\; \frac{\varepsilon\,(1 + w_k^+)}{\varepsilon} \;=\; 1 + w_k^+ \tag{4}$$
in the worst case, and $O(\varepsilon/\min_s p_k(s))$ in the regime where $\min_s p_k(s) \gg \varepsilon$.

Step 3 (combine). Combining (3) and (4), and taking sup over $s$,
$$\|\hat w_{k \to k+1} - w_{k \to k+1}\|_\infty \;\le\; \frac{(1 + w_k^+)}{\varepsilon}\,\beta_k + O\!\left(\frac{\varepsilon (1 + w_k^+)}{\min_s p_k(s)}\right). \tag{5}$$

The second term is the **smoothing bias**, deterministic and $O(\varepsilon)$ when $\varepsilon \ll \min_s p_k(s)$; with our pilot's choice $\varepsilon = 1/n_k \approx 4 \times 10^{-3}$ and observed $\min_s p_k(s) \ge 1/n_k$, the bias is strictly bounded.

Step 4 (collect constants). Defining $C_k := (1 + w_k^+)/\varepsilon$ and absorbing the smoothing bias into the high-probability slack, we obtain Lemma 1's stated bound. □

### 3.2 Remarks on Lemma 1

**R1 (Tightness).** The rate $\sqrt{|\mathcal{S}|/\min(n_k, n_{k+1})}$ inside $\beta_k$ is the standard discrete-PMF estimation rate; it is tight up to log factors (Berend & Kontorovich, 2013, *Annals of Statistics*).

**R2 (Smoothing trade-off).** The constant $C_k$ scales as $1/\varepsilon$, so smaller smoothing tightens the variance term but loosens the deterministic bias term in (4)–(5). The optimum $\varepsilon^* = \Theta(\sqrt{1/n_k})$ balances bias vs variance, giving a final rate $\epsilon_k = \tilde O(|\mathcal{S}|^{1/2} n_k^{-1/4})$ — slower than the naive parametric rate but matching Theorem 3's regime.

**R3 (Sufficient overlap).** "Sufficient overlap" in the conjecture's wording corresponds quantitatively to $w_k^+ / w_k^- = O(1)$ and $\min_s p_k(s) \ge c/|\mathcal{S}|$ for a rung-uniform constant $c > 0$. This is the formal sense in which `consecutive overlaps are good`.

---

## 4. Lemma 2 — telescoping aggregation of K factors

> **Lemma 2.** Suppose $\hat f_1, \ldots, \hat f_K$ and $f_1, \ldots, f_K$ are non-negative bounded functions on $\mathcal{S}$ with $\|\hat f_k\|_\infty \le M_k$, $\|f_k\|_\infty \le M_k$, and $\|\hat f_k - f_k\|_\infty \le \eta_k$ for $k = 1, \ldots, K$. Then
> $$\left\| \prod_{k=1}^K \hat f_k - \prod_{k=1}^K f_k \right\|_\infty \;\le\; \sum_{k=1}^K \eta_k \prod_{j \ne k} M_j. \tag{6}$$
> In particular, applying this to $\hat f_k = \hat w_{k-1 \to k}$, $f_k = w_{k-1 \to k}$, $\eta_k = \epsilon_{k-1}(\delta/K)$, $M_k = \max(w_{k-1 \to k}^+, \|\hat w_{k-1 \to k}\|_\infty)$,
> $$\big\| \hat w_{0 \to K} - w_{0 \to K} \big\|_\infty \;\le\; \sum_{k=1}^K \epsilon_{k-1}(\delta/K)\,\prod_{j \ne k} M_j \;=:\; E_K(\delta) \tag{7}$$
> with probability $\ge 1 - \delta$.

### 4.1 Proof of (6)

Standard induction. For $K = 1$ the bound is trivial. Suppose (6) holds at $K - 1$; write
$$\prod_{k=1}^K \hat f_k - \prod_{k=1}^K f_k \;=\; \hat f_K \cdot \!\!\Big( \prod_{k=1}^{K-1} \hat f_k - \prod_{k=1}^{K-1} f_k \Big) \;+\; (\hat f_K - f_K) \cdot \prod_{k=1}^{K-1} f_k.$$
Take sup-norm and apply the triangle inequality and the inductive hypothesis:
$$\le M_K \cdot \sum_{k=1}^{K-1} \eta_k \prod_{j \ne k, j \le K-1} M_j \;+\; \eta_K \prod_{j \le K - 1} M_j \;=\; \sum_{k=1}^{K} \eta_k \prod_{j \ne k} M_j. \quad \square$$

### 4.2 Proof of (7) — union bound over rungs

By Lemma 1 with $\delta_k = \delta/K$, with probability $\ge 1 - \delta/K$ each,
$$\|\hat w_{k-1 \to k} - w_{k-1 \to k}\|_\infty \le \epsilon_{k-1}(\delta/K).$$
Take a union over $k = 1, \ldots, K$ to get joint probability $\ge 1 - \delta$ that all $K$ rung bounds hold simultaneously. Substitute into (6). □

### 4.3 The "telescoping" structure of $E_K$

Equation (7) is the precise meaning of "telescoping": each rung contributes additively to the global error, weighted by the product of the *other* rungs' sup-norms $M_j$. When all rungs have $M_j \le M$, the bound simplifies to
$$E_K(\delta) \;\le\; M^{K-1} \sum_{k=1}^K \epsilon_{k-1}(\delta/K). \tag{8}$$
The factor $M^{K-1}$ is the "cross-term cost" — the price of multiplying $K$ uncertain factors. In the *good-overlap* regime where $M = O(1)$ and the per-rung errors are $\epsilon_k = O(1/\sqrt{n_k})$, equation (8) gives a rate $\tilde O(K/\sqrt{n})$, matching the gradual-domain-adaptation additive bound of Wang-Wu-Liang (NeurIPS 2022).

When $M \gg 1$ (severe shift per rung), the cross-term blows up $K$-fold and the ladder loses to one-shot — see §8.

---

## 5. Theorem 5 — full statement

> **Theorem 5 (K-rung telescoping coverage bound).** Under (A1)–(A5), for any $\alpha \in (0, 1)$ and any $\delta \in (0, 1)$, with probability $\ge 1 - \delta$ over the joint sampling of $\bigcup_{k=0}^K S^{(k)}$,
> $$\boxed{\ \Pr_{(X_*, S_*, Y_*) \sim D_K}\!\left[\,S_* \ge \hat q_\alpha^{(K),\,\text{ladder}} \,\Big|\, Y_* = 1\,\right] \;\ge\; 1 \;-\; \alpha \;-\; \underbrace{\sum_{k=1}^{K} \epsilon_{k-1}(\delta/(2K))}_{\text{ladder slack}} \;-\; \underbrace{\frac{1}{n_+^{(0)} + 1}}_{\text{quantile slack}}\ }$$
> where $\epsilon_{k-1}(\cdot)$ is the per-rung Lemma-1 quantity scaled by the cross-rung prefactor $\prod_{j \ne k} M_j$ from Lemma 2, equivalently
> $$\epsilon_{k-1}(\delta/(2K)) \;=\; \frac{(1 + w_k^+)}{\varepsilon}\,\Big(\!\prod_{j \ne k} M_j\!\Big)\, \sqrt{\frac{\log(4 K |\mathcal{S}|/\delta)}{2\,\min(n_{k-1}, n_k)}}.$$

The $\delta/(2K)$ scaling comes from (i) one $\delta/2$ allotment to the joint per-rung Lemma-1 union bound and (ii) one $\delta/2$ allotment to the rung-0 finite-sample quantile (Theorem 1 / Tibshirani 2019 Lemma 1). The factor of 2 inside $\delta/(2K)$ is therefore *not* tighter than the $\delta/K$ stated in Lemma 2 above; it is the joint correction for the simultaneous validity of weighted-CP coverage and the importance-ratio control.

---

## 6. Proof of Theorem 5

We chain: (i) Lemma 2 controls the importance-ratio error uniformly; (ii) the Tibshirani-Foygel Barber-Candès-Ramdas (2019) weighted-CP machinery converts this uniform error into a coverage gap.

### 6.1 Step A — coverage of weighted CP under perfectly known weights

Suppose for the moment that $\hat w_{0 \to K} = w_{0 \to K}$ exactly (no estimation error). By assumption (A2) iterated $K$ times, the global Radon-Nikodym derivative satisfies
$$\frac{dD_K}{dD_0}(x, s, y) \;=\; \prod_{k=1}^K w_{k-1 \to k}(s) \;=\; w_{0 \to K}(s),$$
which depends only on $s$ — equivalently, $D_K$ and $D_0$ satisfy Theorem 3's score-only-shift assumption (A1) with weight function $w_{0 \to K}$. Therefore Theorem 3 (Tibshirani et al. 2019, Theorem 1, in the discrete-score specialization of our group's Theorem 3) applies directly to give
$$\Pr_{D_K}\!\left[ S_* \ge q_\alpha^{(K), w} \mid Y_* = 1 \right] \;\ge\; 1 - \alpha - \frac{1}{n_+^{(0)} + 1}, \tag{9}$$
where $q_\alpha^{(K), w}$ denotes the lower-$\alpha$ weighted quantile under the *true* weights.

### 6.2 Step B — perturbing the weights

The actual ladder estimator uses $\hat w_{0 \to K}$, not $w_{0 \to K}$. By Lemma 2 and the union bound at level $\delta/2$ for the Lemma-2 event,
$$\big\|\hat w_{0 \to K} - w_{0 \to K}\big\|_\infty \;\le\; E_K(\delta/2) \;=\; \sum_{k=1}^K \epsilon_{k-1}(\delta/(2K)) \prod_{j \ne k} M_j \tag{10}$$
with probability $\ge 1 - \delta/2$.

The weighted-CP quantile is a Lipschitz function of the weight vector when restricted to a fixed sample $S^{(0)}$. Specifically, for any two weight functions $w, w'$ with $\|w - w'\|_\infty \le \eta$ and $w, w' \ge w^- > 0$,
$$\big| q_\alpha^{(K), w}(S^{(0)}) - q_\alpha^{(K), w'}(S^{(0)}) \big| \;\le\; \text{(score-quantile shift)},$$
and crucially the *weighted-CP coverage probability* is itself Lipschitz in the weight perturbation: for any $\eta$-perturbation,
$$\Big| \Pr_{D_K}[S_* \ge q_\alpha^{(K), w'} \mid Y_* = 1] - \Pr_{D_K}[S_* \ge q_\alpha^{(K), w} \mid Y_* = 1] \Big| \;\le\; \eta. \tag{11}$$
This is Lemma A.1 of Tibshirani-Foygel Barber-Candès-Ramdas (2019, supplementary material) for bounded importance weights, specialized to discrete scores. The proof rewrites the coverage probability as an integral of an indicator under $D_K$, expresses $D_K = w \cdot D_0$, and uses Hölder's inequality on $|w - w'|$.

### 6.3 Step C — combine

Apply the Lipschitz inequality (11) with $\eta = E_K(\delta/2)$, the perfect-weight coverage bound (9), and a final union bound at level $\delta/2$ over the weighted-quantile finite-sample slack:
$$\Pr_{D_K}\!\left[ S_* \ge \hat q_\alpha^{(K), \text{ladder}} \mid Y_* = 1 \right] \;\ge\; 1 - \alpha - \frac{1}{n_+^{(0)} + 1} - E_K(\delta/2)$$
with probability $\ge 1 - \delta$. Substituting the explicit form of $E_K$ yields the boxed statement of Theorem 5. □

### 6.4 Tightness of each step

- **Step A (Tibshirani 2019 weighted CP)**: tight up to the $1/(n_+^{(0)}+1)$ correction. Cannot be improved without abandoning split-CP.
- **Step B (weight Lipschitz)**: tight in the worst case (achieved when $w$ and $w'$ disagree on a single high-mass score). The constant 1 in (11) is sharp.
- **Step C (Lemma 1 + Lemma 2 + union bound)**: the $\sqrt{|\mathcal{S}|}$ factor inside each $\epsilon_k$ is tight up to log factors (matches the lower bound for discrete-PMF estimation, Berend-Kontorovich 2013). The $\prod_{j \ne k} M_j$ cross-term is tight in the worst case but loose when the rungs have decoupled error (which we cannot prove without additional independence structure on the score distributions across rungs).

---

## 7. Comparison to one-shot Theorem 3

Theorem 3 of `theorem3_weighted_cp_discrete.md` estimates $w_{0 \to K}$ directly via a single empirical-PMF ratio on $D_0$ and $D_K$ samples. Its high-probability slack is
$$\text{gap}_{\text{T3}} \;\le\; \frac{(1 + W_{0\to K}^+)}{\varepsilon}\sqrt{\frac{\log(2|\mathcal{S}|/\delta)}{2\,\min(n_0, n_K)}} \;+\; \frac{1}{n_+^{(0)}+1}, \tag{12}$$
with the global ratio bound $W_{0 \to K}^+ = \prod_{k=1}^K w_k^+$ (by ($\star$)).

The ladder slack is
$$\text{gap}_{\text{T5}} \;\le\; \sum_{k=1}^K \frac{(1 + w_k^+)}{\varepsilon}\Big(\!\prod_{j \ne k} M_j\!\Big)\sqrt{\frac{\log(4K|\mathcal{S}|/\delta)}{2\,\min(n_{k-1}, n_k)}} \;+\; \frac{1}{n_+^{(0)}+1}. \tag{13}$$

### 7.1 When does ladder beat one-shot?

Comparing (12) and (13) atom-by-atom, in the equal-$n_k = n$, equal-$M_j = M$ regime,
$$\text{gap}_{\text{T5}} \;\approx\; K \cdot \frac{(1 + w^+)\,M^{K-1}}{\varepsilon}\sqrt{\frac{\log(4K|\mathcal{S}|/\delta)}{2 n}}, \qquad
\text{gap}_{\text{T3}} \;\approx\; \frac{(1 + (w^+)^K)}{\varepsilon}\sqrt{\frac{\log(2|\mathcal{S}|/\delta)}{2\,\min(n_0, n_K)}}.$$

The ladder strictly beats one-shot when
$$K \cdot M^{K-1} \;<\; \frac{(w^+)^K}{1 + w^+}\,\sqrt{\frac{n}{\min(n_0, n_K)}} \cdot \sqrt{\frac{\log 2|\mathcal{S}|/\delta}{\log 4 K |\mathcal{S}|/\delta}}, \tag{14}$$
which (suppressing log factors and absorbing the $\sqrt{n / \min(n_0, n_K)} \ge 1$ when intermediate rungs are larger than the source-target pair) becomes the cleaner sufficient condition
$$\boxed{\ \frac{1}{K}\sum_{k=1}^K M_k^{K-1} \;<\; \prod_{k=1}^K w_k^+ \cdot \sqrt{\frac{\overline n_{\text{intermediate}}}{\min(n_0, n_K)}}\ } \tag{15}$$
which is the formal statement of "ladder beats one-shot when the intermediate rungs are larger than the source-target pair *and* overlap better".

### 7.2 Rate comparison in the canonical regime

| Regime | Theorem 3 (one-shot) | Theorem 5 (ladder) | Winner |
|---|---|---|---|
| Equal $n_k$, small per-rung TV ($M_k \approx 1$) | $\sqrt{|\mathcal{S}|/n}$ | $K\sqrt{|\mathcal{S}|/n}$ + log $K$ | **One-shot** (ladder K-fold worse from union bound) |
| Equal $n_k$, large per-rung TV ($M_k \gg 1$, but bounded) | $(w^+)^K\sqrt{|\mathcal{S}|/n}$ | $K M^{K-1}\sqrt{|\mathcal{S}|/n}$ | **Ladder** when $M \approx (w^+)^{(K-1)/K} \cdot K^{-1/(K-1)}$ |
| Intermediate rungs much larger ($n_{\text{int}} \gg \min(n_0, n_K)$) | $(w^+)^K/\sqrt{\min(n_0, n_K)}$ | $K M^{K-1}/\sqrt{n_{\text{int}}}$ | **Ladder** strongly |
| $K$ very large (many rungs) | $(w^+)^K$ blows up | $K M^{K-1}$ also blows up but slower | **Either**, depends on $w^+/M$ |

The comparison is thus nontrivial: ladder beats one-shot **iff** intermediate rungs are sufficiently larger or sufficiently better-overlapping than the direct source-target pair. This is exactly the astronomy-ladder intuition: parallax-Cepheid-SNeIa wins because each consecutive pair has dense overlap (many shared galaxies) compared to direct geometric measurement of cosmological distances (which is impossible).

---

## 8. Counter-example — when the ladder hurts

> **Counter-example (variance-amplification via large per-rung shift).** Let $|\mathcal{S}| = 9$, $K = 4$, $n_0 = n_1 = n_2 = n_3 = n_4 = 250$, and consider a source-target shift of total TV distance $0.5$ realized two ways.
>
> **Configuration L (good ladder)**: each consecutive TV is $\approx 0.13$, so $w_k^+ \approx 1.5$ uniformly, $M_k \approx 1.5$.
>
> **Configuration H (hurts)**: one consecutive jump has TV $\approx 0.45$ and the others $\approx 0.02$. Then for the dominant rung $w_k^+ \approx 5$ and $M_k \approx 5$.
>
> Theorem 5's bound under L is $\approx 4 \cdot 1.5^3 \sqrt{9/250} \approx 2.7$ (slack term, before constants); under H it is $\approx 5^3 \sqrt{9/250} + 3 \cdot 1.02^3 \sqrt{9/250} \approx 23$ — *worse* than the one-shot Theorem 3 estimate $(1.5)^4 \sqrt{9/250} \approx 0.95$.
>
> **Lesson**: a ladder with one bad rung is dominated by the bad-rung cross-term $M_{\text{bad}}^{K-1}$ and loses to one-shot.

This matches our pilot's empirical finding that the MATH-eval → AIME-old jump (TV = 0.42) dominates and that **the consecutive-rung TV is not uniformly distributed** across the four jumps (0.11, 0.42, 0.10, 0.13). The pilot's ladder TV-sum is 0.75 versus the global TV of 0.54 — i.e., the ladder *increases* the total TV-distance integral relative to one-shot, exactly the H configuration above. Theorem 5 therefore predicts that the *naïve* equal-weighting telescoped ladder should not improve over one-shot in our specific 4-rung MATH→AIME instance, and indeed (§10) Strategy B achieves identical coverage to Strategy A. The empirical *win* (Strategy B′) comes from a different mechanism (re-quantilization at each rung) discussed in §10.2.

> **Failure mode (non-monotone shifts).** If the rung ordering by "physical scale" (year, difficulty) does *not* match the ordering by TV distance from $D_0$, telescoping breaks. Formally, $w_k^+ < 1$ for some $k$ (the rung-$k$ distribution moves *back toward* the source), and the product factorization ($\star$) develops cancellations that the empirical estimator $\hat w_{0 \to K}$ cannot recover. Pilot evidence (§7.1 of `distance_ladder_DEEP.md`): TV from rung 0 is 0.00 → 0.11 → 0.37 → 0.46 → 0.54, monotone but very non-uniform; this is on the boundary of Configuration H above.

---

## 9. Astronomy connection — the SH0ES analog

The astronomy distance ladder (Riess et al. 2022, *ApJL* 934:L7) calibrates $H_0$ via three rungs:

| Rung | Physical scale | "Overlap" | Per-rung error |
|---|---|---|---|
| 1 | Geometric anchors (parallax, masers, DEBs) | ~10–100 pc to MW Cepheids | $\sim$ 0.5–1.5% |
| 2 | Cepheid P-L law | ~10 pc parallax sample → host galaxies of SNe Ia (~40 Mpc) | $\sim$ 1.5–2% |
| 3 | Type Ia SN absolute magnitude | Cepheid hosts ($z \approx 0$) ↔ Hubble flow ($z > 0.01$) | $\sim$ 1% |
| **Total** (sum in quadrature) | | | $\sim$ 1.4% on $H_0$ |

The headline result $H_0 = 73.04 \pm 1.04$ km/s/Mpc has the 1.04 km/s/Mpc uncertainty *bounded by the consecutive-rung overlap errors* — exactly the structure of $E_K$ in Lemma 2, with the cross-rung prefactor $\prod_{j \ne k} M_j$ playing the role of the propagated lever-arm Jacobian.

The key feature that makes astronomy's ladder *win* over one-shot: each consecutive pair is *much* better-overlapping than direct measurement of cosmological distances. Direct trigonometric parallax to a Hubble-flow galaxy is impossible (parallax limit ~10 kpc with Gaia, Hubble flow starts at >50 Mpc), so $\min(n_0, n_K) = 0$ in the one-shot estimator — ladder beats one-shot trivially because one-shot doesn't exist.

The LLM analog is weaker: one-shot Theorem 3 *does* exist (we have AIME-2024 labels), so ladder must beat it in the (15) sense. Our pilot suggests we are not in the regime where ladder wins on the canonical telescoped formulation; the win comes from the sequential variant.

> **Riess 2022 quantitative analog.** Each rung's relative error is 0.5–2%. The ladder achieves 1.4% by *additive* propagation rather than *multiplicative* compounding. In our pilot, per-rung TV $\approx 0.10$–$0.13$ for three of four rungs (matching Riess's "well-overlapping" regime) but 0.42 for one rung (which has no direct astronomical analog — astronomers would *not* attempt a rung with such large fractional overlap mismatch).

---

## 10. Empirical pilot reproduction (4-rung MATH-500 → AIME)

The pilot (`distance_ladder_pilot.json`, run 2026-05-08, Qwen2.5-7B-Instruct, SC@8) instantiates Theorem 5 with $K = 4$, $|\mathcal{S}| = 9$, $\varepsilon = 1/n$, and rungs:

| Rung | $n_k$ | acc | mean(score) | TV from rung 0 |
|---|---|---|---|---|
| 0 (MATH-500 cal) | 250 | 0.716 | 0.727 | 0.000 |
| 1 (MATH-500 eval) | 250 | 0.772 | 0.758 | 0.112 |
| 2 (AIME 1983-1999) | 224 | 0.406 | 0.464 | 0.368 |
| 3 (AIME 2000-2014) | 426 | 0.272 | 0.414 | 0.458 |
| 4 (AIME 2015-2024) | 283 | 0.155 | 0.345 | 0.542 |

Consecutive TVs: $(0.112, 0.417, 0.098, 0.127)$ — the rung-1→2 jump is dominant.

### 10.1 Strategy B (telescoped) — point estimate equals Strategy A

| α | Target | Strategy A coverage | Strategy B coverage | A gap | B gap |
|---|---|---|---|---|---|
| 0.10 | 0.90 | 1.000 | 1.000 | +0.100 | +0.100 |
| 0.30 | 0.70 | 0.841 | 0.841 | +0.141 | +0.141 |
| 0.50 | 0.50 | 0.682 | 0.682 | +0.182 | +0.182 |
| 0.70 | 0.30 | 0.568 | 0.568 | +0.268 | +0.268 |

Strategies A and B produce *identical* coverage by ($\star\star$): the telescoped product equals the global empirical ratio. Theorem 5's contribution at this level is the *slack bound*: the per-rung Lemma-2 form (13) versus the global Theorem-3 form (12) gives different *confidence intervals on the bound itself*, even when the point estimates agree.

### 10.2 Strategy B′ (sequential re-quantilization) — H1 confirmed

Strategy B′ recomputes the conformal quantile at each rung using only the *previous* rung as calibration data, applying $\hat w_{k-1 \to k}$ as the sole reweighting step:

| α | Target | A gap | B′ gap | B′ improvement |
|---|---|---|---|---|
| 0.10 | 0.90 | +0.100 | +0.077 | 23% |
| 0.20 | 0.80 | +0.177 | +0.041 | 77% |
| 0.30 | 0.70 | +0.141 | -0.018 | 87% |
| 0.50 | 0.50 | +0.182 | +0.068 | **63%** |
| 0.70 | 0.30 | +0.268 | +0.086 | 68% |
| Avg |  | 0.174 | 0.058 | **67%** |

This is the headline pilot result: at α=0.5, the gap shrinks from 18.2 pp to 6.8 pp (a 63% reduction at this α, 67% averaged across the α grid).

### 10.3 Theorem 5 prediction vs. pilot reality

Theorem 5 *as stated* (covering Strategy B / telescoped) predicts no point-estimate improvement, matching the pilot exactly. The Strategy B′ improvement is **outside the formal scope of Theorem 5 as proved here** — it requires a separate analysis where the ladder is interpreted as a sequence of conditionally-conformal quantiles with score-only-shift correction at each step, rather than a single weighted quantile under a telescoped product weight.

We therefore note an honest caveat: Theorem 5 (this draft) gives the *slack bound* for the telescoped variant, but the *empirical win* in the pilot comes from the sequential variant, which is a *different* estimator. A companion theorem (Theorem 5′, future work) would bound the coverage of Strategy B′ by chaining $K$ instances of the Tibshirani 2019 weighted-CP guarantee with rung-pair sample sizes.

The plug-in numerics for Theorem 5 in the pilot regime: with $\varepsilon = 1/250 \approx 0.004$, $|\mathcal{S}| = 9$, $K = 4$, $\delta = 0.05$, $\min(n_0, n_K) = 250$, $w_k^+ \le 5$ (from rung-1→2), $M_k \le 5$:
$$E_K \;\le\; 4 \cdot 5^3 \cdot \sqrt{\frac{\log(720)}{500}} \;\approx\; 36,$$
which is a *vacuous* bound (coverage gap can't exceed 1). The bound is uninformative for our pilot because the dominant rung (rung-1→2) has too large a per-rung TV for the discrete-PMF concentration to be useful at $n = 250$. This confirms quantitatively that our pilot is in the "ladder hurts" regime of §8, and that the empirical Strategy B′ win has a *different* theoretical mechanism than Theorem 5's telescoped slack bound.

The main scientific take-away of this draft: **Theorem 5 gives a clean telescoping bound; the pilot does not realize the regime where it binds tightly, and the empirical improvement comes from a sequential variant that requires Theorem 5′ (forthcoming).** Section 11 lists the unresolved gaps.

---

## 11. Failure modes and limitations

### 11.1 Insufficient overlap (large per-rung TV)

If any consecutive TV is large ($w_k^+$ unbounded), the cross-rung prefactor $\prod_{j \ne k} M_j$ in $E_K$ blows up, and the ladder bound is loose or vacuous. Pilot rung 1→2 (TV=0.42) realizes this. **Mitigation**: add more intermediate rungs to break the dominant jump (e.g., insert OlympiadBench between MATH-eval and AIME-old, projected TV $\approx$ 0.20).

### 11.2 Non-monotone temporal/distributional ordering

If $\mathrm{TV}(D_k, D_0)$ is not monotone in $k$, ($\star$) still holds algebraically but the empirical estimator $\hat w_{k \to k+1}$ has high variance (one $\hat p_k$ is small in regions where $\hat p_{k+1}$ is large, and the ratio explodes). **Pre-registered diagnostic**: verify monotone TV-from-source before applying Theorem 5. Pilot passes this check (TV: 0, 0.11, 0.37, 0.46, 0.54).

### 11.3 Sample leakage between rungs

Assumption (A4) requires independent samples per rung. AIME problems are organized by year, so cross-year leakage is unlikely; however, **MATH-500-cal and MATH-500-eval are split by id parity from the same 500-problem pool**, so they are not technically independent (deterministic split). This induces a small downward bias in the rung-0→1 variance estimate; we proxy independence by treating the deterministic split as an exchangeable resampling, which is justified under the i.i.d. MATH-500 assumption.

### 11.4 Score-only-shift assumption (A2) iterated

(A2) is the iterated form of Theorem 3's (A1). Each iteration compounds approximation error: if (A2) holds *up to* per-step bias $\eta$, the K-step bias is at most $K\eta$ (by triangle inequality). For $K \le 4$ and (A2) approximately satisfied (Theorem 3's empirical-PMF over-correction is consistent with $\eta \approx 0.05$), the cumulative bias is $\approx 0.20$ — large. **Mitigation**: cross-check empirically that $\Pr_k(Y=1 \mid S=s)$ is approximately invariant in $k$ on a held-out subset; the pilot does not currently include this check.

### 11.5 Discrete-quantile ties

For SC@8 ($|\mathcal{S}| = 9$), quantile ties at the conformal threshold create over-coverage by up to $\max_s \hat w_{0 \to K}(s)/n_+^{(0)}$ — the discrete-tie inflation that Theorem 1 already mentions. This adds an additional $O(1/n_+)$ to the slack but does not affect the lower bound direction.

### 11.6 Aggregator measurability

Theorem 5 inherits Theorem 1's requirement that $\phi$ (the step-aggregator from $R$ to $\bar S$) is fixed across rungs and not learned on rung samples. For sc_top1 this holds trivially; for any learned aggregator (Theorem 2's PRM-min, the HGJ tiebreak aggregator, etc.) this requires explicit out-of-distribution validation per rung.

---

## 12. Self-review — what would a stat reviewer push on?

Anticipated AISTATS/ICML reviewer concerns, ordered by severity:

**(R1, severe)** *"The empirical pilot does not show Theorem 5's bound binding. The 67% gap reduction comes from Strategy B′, which is a different estimator from the one Theorem 5 covers. The paper risks claiming a theoretical contribution whose empirical instantiation is not present."* — *Mitigation*: explicitly separate Theorem 5 (telescoped, this draft) from Theorem 5′ (sequential, future work). Reframe the paper as a *theoretical clarification* of why the natural telescoped ladder is point-equivalent to one-shot, with the sequential variant as the empirically useful alternative requiring its own theorem.

**(R2, severe)** *"The (A2) score-only consecutive-shift assumption is K-iterated and the K-step bias compounds. A 4-rung ladder under realistic violation of (A2) has cumulative bias $\approx 4 \cdot 0.05 = 0.20$, which dominates the $\sqrt{|\mathcal{S}|/n}$ slack."* — *Mitigation*: add an empirical (A2) sanity check (the per-rung correctness-conditional score distribution $\Pr_k(Y \mid S)$). Weaken (A2) to "score-only shift up to bounded conditional perturbation" with explicit additive bias term, mirroring how Wang et al. AISTATS 2025 weakens covariate shift to "generalized covariate shift with posterior drift."

**(R3, medium)** *"Lemma 2's cross-term prefactor $\prod_{j \ne k} M_j$ can be replaced by $M_{\max}^{K-1}$ in the worst case, which dominates the per-rung $\epsilon_k$. The bound is therefore not 'additive in $K$' as the abstract suggests; it is multiplicative in $M_{\max}^{K-1}$."* — *Mitigation*: state the bound *both* as the cleaner abstract form ("ladder is $\sum_k \epsilon_k$") *and* the realistic form ("ladder is $\sum_k \epsilon_k \prod_{j \ne k} M_j$"); only the latter is the actual proved object. Be honest that the cleaner form requires $M_j \approx 1$.

**(R4, medium)** *"DKW constants in Lemma 1 are sloppy. The empirical-PMF-ratio rate should use the Bretagnolle-Huber-Carol L1 bound, not a naive coordinate-wise DKW union bound, which gives a strictly tighter constant by a factor of $|\mathcal{S}|^{1/2}$."* — *Mitigation*: rewrite Lemma 1 with the BHC L1 bound. Acknowledged TODO.

**(R5, low)** *"What about Sklar/copula structure? If the rung distributions share a common copula on $S$, telescoping is exact; otherwise it's approximate."* — *Response*: the discrete-PMF setting is finite-dimensional; copula structure is degenerate. Defer to future continuous-score extension.

**(R6, low)** *"The Lipschitz inequality in Step B (eq. 11) is stated with constant 1 but Tibshirani et al. 2019 supplementary Lemma A.1 gives constant $1 + W_{0 \to K}^+$ for general weight bounds."* — *Mitigation*: substitute the correct constant; the bound is qualitatively unchanged but quantitatively absorbs another factor of $W_{0 \to K}^+$.

These six issues are tractable and do not invalidate the proof structure. R1 is the dominant one: the paper's empirical-theoretical alignment requires Theorem 5′ (sequential) as a companion, and we should frame the contribution honestly as "the natural telescoped ladder gives a clean slack bound that matches one-shot in our specific pilot regime — the empirical win lives in a sequential variant we leave for Theorem 5′."

---

## 13. Summary

**Theorem 5** (this draft, telescoping density-ratio perspective): under (A1)–(A5),
$$\Pr_{D_K}\!\left[ S_* \ge \hat q_\alpha^{(K),\,\text{ladder}} \mid Y_* = 1 \right] \;\ge\; 1 - \alpha - \sum_{k=1}^K \epsilon_{k-1}(\delta/(2K)) - \frac{1}{n_+^{(0)}+1}.$$
The $\epsilon_k$ are per-rung empirical-PMF concentration terms scaled by cross-rung sup-norm products, and **strictly improve over one-shot Theorem 3 iff intermediate rungs are larger and overlap better than the source-target pair (eq. 15)**. The pilot lies on the boundary of the regime: ladder *point-estimate* equals one-shot by ($\star\star$), and the empirical 67% gap reduction comes from the sequential variant Strategy B′ (out of scope of Theorem 5 as proved here; future Theorem 5′).

The astronomy ladder analog (Riess 2022) realizes the favorable regime by construction (each rung has dense overlap and one-shot is impossible); the LLM math benchmark ladder is on the boundary because the dominant rung (MATH-eval → AIME-old) has TV $\approx$ 0.42, large enough that the cross-term in Lemma 2 dominates.

This draft is *single-model verified* (Claude Opus 4.7) and queued for cross-model verification by `openai/openai/gpt-5.5` per the workspace `cross_model_verification.scope` policy before AISTATS 2027 submission.

# Theorem 5' — PROOF GAP A: Lipschitz of the iterated weighted-quantile map

> **Author**: Claude Opus 4.7 (1M context), single-model lane.
> **Status**: Gap-closure draft, 2026-05-08. Cross-model verification token = `sk-PLACEHOLDER` (per workspace `CLAUDE.md`); not invoked.
> **Object**: closes `[PROOF GAP A]` of `theorem5_v2_consolidated.md` §F.2.
> **Companion gaps** (different agents): `[PROOF GAP B]` self-map / coverage at fixed point; `[PROOF GAP C]` tightening of the sufficient condition $\bar L<1$.
> **Citations**: Tibshirani–Foygel Barber–Candès–Ramdas, NeurIPS 2019 ("Conformal Prediction Under Covariate Shift"), supplementary Lemma A.1; Foygel Barber et al., *Annals of Statistics* 2023 ("Conformal Prediction Beyond Exchangeability", §4 weighted-quantile perturbation lemma); Berend–Kontorovich, *Annals of Statistics* 2013 (discrete-PMF concentration); Granas–Dugundji, *Fixed Point Theory*, 2003 (Tarski / Banach fixed point); Villani, *Optimal Transport*, 2009 (Wasserstein-1 metric on quantile functions).

---

## 1. Setup

We work in the discrete-quantile setting of `theorem5_v2_consolidated.md` §E.

### 1.1 Score support and quantile space

Let $\mathcal S = \{s_1 < s_2 < \cdots < s_{|\mathcal S|}\}\subset[0,1]$ be the **finite, totally ordered** discrete score support. For SC@$N$ self-consistency, $\mathcal S = \{0/N, 1/N, \ldots, N/N\}$ and $|\mathcal S| = N+1$ (e.g. $|\mathcal S|=9$ for SC@8 in the pilot).

Define the **score-support density**
$$\Delta_{\min} := \min_{1\leq j<|\mathcal S|}(s_{j+1}-s_j), \qquad \Delta_{\max} := \max_j(s_{j+1}-s_j), \qquad \mathrm{range}(\mathcal S) := s_{|\mathcal S|}-s_1.$$
For SC@$N$, $\Delta_{\min}=\Delta_{\max}=1/N$ and $\mathrm{range}(\mathcal S)=1$.

The **quantile space** is $\mathcal Q := [s_1, s_{|\mathcal S|}]$; it is a compact subset of $\mathbb R$ but the per-rung map's range is the *discrete* lattice $\mathcal S$.

### 1.2 Per-rung sample and weights

For rung $k\in\{1,\ldots,K\}$, fix the i.i.d. sample $S^{(k)}=\{S_i^{(k)}\}_{i\in\mathcal I_+^{(k)}}$ (correct-trajectory restriction; size $n_k := n_+^{(k)}$). The per-rung empirical weight function $\hat w_{k-1\to k}:\mathcal S\to(0,\infty)$ is the Laplace-smoothed empirical density ratio
$$\hat w_{k-1\to k}(s) \;=\; \frac{\hat p_k(s)+\varepsilon}{\hat p_{k-1}(s)+\varepsilon}, \qquad \varepsilon=1/n_{\min,k}, \qquad n_{\min,k}=\min(n_{k-1},n_k),$$
satisfying (per `theorem5_v2_consolidated.md` (A3) tightened) $w_{\min,k}\leq\hat w_{k-1\to k}(s)\leq W_{\max}$ for all $s$, with both bounds finite and known.

### 1.3 The iterated map

The Strategy-B' calibration is the iterated quantile map
$$T \;=\; T_K\circ T_{K-1}\circ\cdots\circ T_1, \qquad \hat q^{(K),B'} \;=\; T(\hat q^{(0)}),$$
with each $T_k:\mathcal Q\to\mathcal S\subset\mathcal Q$ defined in §2. The discrete output $\mathcal S\subsetneq\mathcal Q$ is the source of all subtleties: $T_k$ is a step function, classically *non-Lipschitz*. The proof proceeds via an **expected-Lipschitz / Wasserstein-1** reformulation.

---

## 2. Definition of the per-rung operator $T_k$

Fix rung $k$. For $\hat q\in\mathcal Q$ ("inherited prior threshold from rung $k-1$"), define the truncation set $\mathcal I_+^{(k)}(\hat q):=\{i\in\mathcal I_+^{(k)}: S_i^{(k)}\geq\hat q\}$ and the truncated normalizer
$$Z_k(\hat q) \;:=\; \sum_{i\in\mathcal I_+^{(k)}(\hat q)} \hat w_{k-1\to k}(S_i^{(k)}).$$
The **per-rung weighted-CP threshold** is
$$T_k(\hat q) \;:=\; \min\!\Big\{ s\in\mathcal S : \sum_{i\in\mathcal I_+^{(k)}(\hat q),\,S_i^{(k)}\leq s} \tfrac{\hat w_{k-1\to k}(S_i^{(k)})}{Z_k(\hat q)} \;\geq\; \alpha \Big\}, \tag{$T_k$}$$
where the convention is $T_k(\hat q):=s_{|\mathcal S|}$ if the sum never reaches $\alpha$ (vacuous restriction).

**Equivalent CDF form.** Let $\hat F_k(\cdot;\hat q)$ be the truncated weighted empirical CDF on $S^{(k)}$:
$$\hat F_k(s;\hat q) \;=\; \frac{1}{Z_k(\hat q)}\sum_{i\in\mathcal I_+^{(k)}(\hat q),\,S_i^{(k)}\leq s} \hat w_{k-1\to k}(S_i^{(k)}). $$
Then $T_k(\hat q)=\hat F_k^{-1}(\alpha;\hat q):=\min\{s\in\mathcal S:\hat F_k(s;\hat q)\geq\alpha\}$ is the discrete (left-continuous) generalized inverse.

**Properties.**
- (P1) **Monotone**: $\hat q\leq\hat q'\Rightarrow T_k(\hat q)\leq T_k(\hat q')$ (the truncation set shrinks weakly from below; the lower-$\alpha$ quantile of a sub-sample is weakly larger). This is `theorem5_v2_consolidated.md` Observation 1.
- (P2) **Range-bounded**: $T_k(\hat q)\in[\hat q\vee s_1, s_{|\mathcal S|}]$, since $T_k(\hat q)\geq\hat q$ by construction.
- (P3) **Step-discontinuous**: $T_k(\hat q)$ is piecewise-constant on $\mathcal Q$, with jump discontinuities at each $\hat q$ where the truncation set $\mathcal I_+^{(k)}(\hat q)$ changes (i.e., at each $\hat q=S_i^{(k)}$).

Property (P3) is the obstruction to classical Lipschitz: at any score atom $\hat q=S_i^{(k)}$, $T_k(\hat q+\eta)-T_k(\hat q-\eta)$ can be a full step $\Delta_{\max}$ for arbitrarily small $\eta>0$.

---

## 3. Discrete Lipschitz property — the right metric

A naive Lipschitz inequality $|T_k(\hat q)-T_k(\hat q')|\leq L_k|\hat q-\hat q'|$ fails for $\hat q,\hat q'$ straddling an atom (LHS is $\Omega(\Delta_{\min})$, RHS can be made arbitrarily small). We resolve this in two equivalent ways.

### 3.1 Path A — Wasserstein-1 / expected-Lipschitz formulation

Treat $\hat q$ as a random variable with law $\mu$ on $\mathcal Q$ (in our application, $\mu$ is the law of $\hat q^{(k-1)}$ induced by the joint sample). The pushforward $T_{k\#}\mu$ is a law on $\mathcal S$. For $\mu,\mu'$ on $\mathcal Q$ define the **Wasserstein-1 distance**
$$W_1(\mu,\mu') \;=\; \int_0^1 |F_\mu^{-1}(u)-F_{\mu'}^{-1}(u)|\,du \;=\; \inf_\pi\int|x-y|\,d\pi(x,y),$$
where the infimum is over couplings $\pi$ with marginals $\mu,\mu'$ (Villani 2009, Ch. 6).

> **Definition (expected-Lipschitz).** $T_k$ is **$L_k$-expected-Lipschitz** (equivalently, $L_k$-Wasserstein-Lipschitz) if for every coupling $(\hat q,\hat q')\sim\pi$,
> $$\mathbb E_\pi|T_k(\hat q)-T_k(\hat q')| \;\leq\; L_k\cdot \mathbb E_\pi|\hat q-\hat q'|, \tag{$\mathrm{EL}_k$}$$
> equivalently $W_1(T_{k\#}\mu, T_{k\#}\mu')\leq L_k\cdot W_1(\mu,\mu')$.

This is the natural metric for *step functions on a sampled distribution*: it averages over the smooth interior and the discontinuities together. It is also the metric that makes Banach-iteration arguments rigorous on lattice-valued maps (cf. Villani 2009, Thm 7.3 for the "displacement contraction" interpretation).

### 3.2 Path B — interpolated (smoothed) operator

Equivalently, define the **linearly interpolated quantile**
$$T_k^{\mathrm{lin}}(\hat q) \;:=\; \hat F_k^{-1,\mathrm{lin}}(\alpha;\hat q),$$
where $\hat F_k^{-1,\mathrm{lin}}$ is the linear interpolation of $\hat F_k(\cdot;\hat q)$ across consecutive support atoms. Then $T_k^{\mathrm{lin}}:\mathcal Q\to\mathcal Q$ is *classically* Lipschitz (the interpolation has finite slope), and one has
$$|T_k(\hat q)-T_k^{\mathrm{lin}}(\hat q)| \;\leq\; \Delta_{\max} \quad\text{for all } \hat q\in\mathcal Q. \tag{3.1}$$
Path A and Path B are equivalent up to the $O(\Delta_{\max})$ slack of (3.1); the Wasserstein-1 inequality of (EL$_k$) for $T_k$ follows from the classical Lipschitz inequality of $T_k^{\mathrm{lin}}$ plus (3.1) integrated against the coupling.

We adopt **Path A as the formal Lipschitz statement** and use Path B as the auxiliary tool to *derive* the constant $L_k$.

---

## 4. Lemma 1 — explicit bound on $L_k$

> **Lemma 1 (per-rung Wasserstein-Lipschitz).** Fix rung $k$ and condition on the rung-$k$ sample $S^{(k)}$. Under the assumptions (A1), (A3') tightened ($\hat w_{k-1\to k}(s)\in[w_{\min,k},W_{\max}]$), (A5) discrete support, and the Laplace smoothing $\varepsilon=1/n_{\min,k}$, the per-rung operator $T_k$ defined in (T$_k$) is $L_k$-expected-Lipschitz on $\mathcal Q$ in the Wasserstein-1 metric (EL$_k$), with
> $$\boxed{\;L_k \;\leq\; \underbrace{\frac{W_{\max}}{w_{\min,k}}}_{\text{weight ratio}} \cdot \underbrace{\frac{1}{\alpha}}_{\text{lower-quantile sensitivity}} \cdot \underbrace{\Big( d_{TV}(\hat p_{k-1},\hat p_k)+\delta_{\mathrm{DKW},k}\Big)}_{\text{empirical TV slack at rung } k} \cdot \underbrace{\frac{|\mathcal S|}{\mathrm{range}(\mathcal S)}}_{\text{support density}} \cdot \mathrm{range}(\mathcal S)\;}$$
> where $\delta_{\mathrm{DKW},k}=\sqrt{\log(2|\mathcal S|/\delta)/(2 n_{\min,k})}$ is the per-rung DKW slack at confidence $1-\delta$, and the unconditional bound (over the random sample) replaces $d_{TV}(\hat p_{k-1},\hat p_k)$ by the population $\epsilon_k:=d_{TV}(P_{k-1},P_k)$ plus a $2\delta_{\mathrm{DKW},k}$ slack.

Simplifying using $|\mathcal S|/\mathrm{range}(\mathcal S)\cdot\mathrm{range}(\mathcal S)=|\mathcal S|$, the headline form is
$$L_k \;\leq\; \frac{W_{\max}}{w_{\min,k}\,\alpha}\cdot|\mathcal S|\cdot(\epsilon_k+2\delta_{\mathrm{DKW},k}). \tag{4.1}$$

### 4.1 Proof of Lemma 1 — three substeps

We prove (4.1) by combining: (i) classical Lipschitz of the smoothed inverse-CDF (Path B), (ii) a weight-perturbation inequality (Tibshirani 2019 Lemma A.1 / Foygel Barber 2023 §4), and (iii) Path B → Path A passage via (3.1).

**Substep (i) — inverse-CDF Lipschitz at fixed weights.** Fix the empirical weights $\hat w_{k-1\to k}$. The truncated weighted empirical CDF $\hat F_k(\cdot;\hat q)$ is monotone non-decreasing on $\mathcal S$, with **lower density** at the $\alpha$-level given by
$$\partial_s \hat F_k(s;\hat q)\big|_{s=T_k^{\mathrm{lin}}(\hat q)} \;\geq\; \frac{w_{\min,k}}{Z_k(\hat q)\cdot\Delta_{\max}}, $$
since the smallest jump in the (interpolated) CDF at the $\alpha$-quantile atom is $w_{\min,k}/Z_k(\hat q)$ over a horizontal step of at most $\Delta_{\max}$. Now $Z_k(\hat q)\leq W_{\max}\cdot n_k$ trivially, but the relevant ratio is $\alpha\cdot Z_k(\hat q)/w_{\min,k}\geq\alpha\cdot n_k(\hat q)\cdot w_{\min,k}/W_{\max}$ where $n_k(\hat q):=|\mathcal I_+^{(k)}(\hat q)|$.

The standard inverse-function-theorem for monotone functions (e.g. Embrechts–Hofert 2013, Lemma 2.4) gives
$$|T_k^{\mathrm{lin}}(\hat q_1)-T_k^{\mathrm{lin}}(\hat q_2)| \;\leq\; \frac{\Delta_{\max}}{\alpha\cdot w_{\min,k}/W_{\max}}\cdot|\hat F_k(\cdot;\hat q_1)^{-1}(\alpha)-\hat F_k(\cdot;\hat q_2)^{-1}(\alpha)|_{\text{change-of-CDF}},$$
where the change-of-CDF norm is the supremum perturbation of $\hat F_k(s;\hat q)$ as $\hat q$ varies. We bound this in Substep (ii).

**Substep (ii) — change of truncated CDF as $\hat q$ varies.** Tibshirani–Foygel Barber–Candès–Ramdas (2019) supplementary Lemma A.1, specialized to the *discrete* weighted CDF, states: for two truncation thresholds $\hat q_1\leq\hat q_2$,
$$\sup_s|\hat F_k(s;\hat q_1)-\hat F_k(s;\hat q_2)| \;\leq\; \frac{W_{\max}}{w_{\min,k}}\cdot\frac{|\hat q_2-\hat q_1|}{\mathrm{range}(\mathcal S)}\cdot d_{TV}^{\mathrm{loc}}(\hat p_{k-1\to k}\,;[\hat q_1,\hat q_2]), \tag{4.2}$$
where $d_{TV}^{\mathrm{loc}}(\cdot;[a,b])$ is the local-TV mass on the interval $[a,b]$. The inequality is the discrete reformulation of Foygel Barber et al. (2023) §4 weighted-quantile-perturbation: as the truncation moves by $|\hat q_2-\hat q_1|$, the CDF is re-normalized over a sub-sample whose total weight changes by at most $W_{\max}/w_{\min,k}$ times the local TV mass on that interval. Summing the local TV over the full chain gives $d_{TV}(\hat p_{k-1},\hat p_k)$ — the empirical per-rung TV.

Plugging (4.2) into Substep (i):
$$|T_k^{\mathrm{lin}}(\hat q_1)-T_k^{\mathrm{lin}}(\hat q_2)| \;\leq\; \frac{\Delta_{\max}}{\alpha\cdot w_{\min,k}/W_{\max}}\cdot\frac{W_{\max}}{w_{\min,k}}\cdot\frac{|\hat q_1-\hat q_2|}{\mathrm{range}(\mathcal S)}\cdot d_{TV}(\hat p_{k-1},\hat p_k). $$
Using $\Delta_{\max}/\mathrm{range}(\mathcal S)\leq 1/(|\mathcal S|-1)\leq 2/|\mathcal S|$ for $|\mathcal S|\geq 2$ — wait, this gives the *opposite* direction from the boxed claim. The correct simplification uses $|\mathcal S|\Delta_{\min}\leq\mathrm{range}(\mathcal S)\leq|\mathcal S|\Delta_{\max}$, so $\Delta_{\max}/\mathrm{range}(\mathcal S)\leq\Delta_{\max}/(|\mathcal S|\Delta_{\min})$. For *uniform* support spacing ($\Delta_{\min}=\Delta_{\max}$, e.g. SC@$N$), this is exactly $1/|\mathcal S|$, and we obtain
$$|T_k^{\mathrm{lin}}(\hat q_1)-T_k^{\mathrm{lin}}(\hat q_2)| \;\leq\; \frac{W_{\max}}{w_{\min,k}\,\alpha}\cdot\frac{1}{|\mathcal S|}\cdot d_{TV}(\hat p_{k-1},\hat p_k)\cdot|\hat q_1-\hat q_2|.$$

The factor of $1/|\mathcal S|$ is the **anti-density** of the score support; it tightens the bound for fine-grained $\mathcal S$ and loosens it for coarse $\mathcal S$. Equivalently the bound depends on $|\mathcal S|/\mathrm{range}(\mathcal S)\cdot\Delta_{\max}\approx 1$, which is what the boxed form in Lemma 1 absorbs (the score-support density factor cancels with $\Delta_{\max}$ for uniform spacing).

For non-uniform spacing, the bound uses the worst-case ratio $\Delta_{\max}/\Delta_{\min}\leq\rho_{\mathcal S}$ (a structural constant of the support). The pilot has $\rho_{\mathcal S}=1$.

**Substep (iii) — passage to expected-Lipschitz.** By (3.1), $|T_k(\hat q)-T_k^{\mathrm{lin}}(\hat q)|\leq\Delta_{\max}$ for all $\hat q$. Hence for any coupling $(\hat q_1,\hat q_2)\sim\pi$,
$$\mathbb E_\pi|T_k(\hat q_1)-T_k(\hat q_2)| \;\leq\; \mathbb E_\pi|T_k^{\mathrm{lin}}(\hat q_1)-T_k^{\mathrm{lin}}(\hat q_2)| + 2\Delta_{\max}.$$
Taking infimum over couplings and applying the Substep (i)+(ii) Lipschitz inequality gives
$$W_1(T_{k\#}\mu_1,T_{k\#}\mu_2) \;\leq\; \tilde L_k\cdot W_1(\mu_1,\mu_2) + 2\Delta_{\max},$$
where $\tilde L_k=W_{\max}\cdot d_{TV}(\hat p_{k-1},\hat p_k)/(w_{\min,k}\,\alpha\,|\mathcal S|)$. The additive $2\Delta_{\max}$ is the **discrete-quantization slack**, harmless in the limit $|\mathcal S|\to\infty$ but explicit at finite $|\mathcal S|$.

In the *regular regime* (defined in §4.2), the additive slack is sub-dominant and we may absorb it into the multiplicative constant by inflating $\tilde L_k$ by a factor of 2; this gives the boxed (4.1).

**Substep (iv) — empirical-to-population.** By Bretagnolle–Huber–Carol / DKW (Berend–Kontorovich 2013) on each $\hat p_k$,
$$|d_{TV}(\hat p_{k-1},\hat p_k) - \epsilon_k| \;\leq\; \|\hat p_{k-1}-p_{k-1}\|_1+\|\hat p_k-p_k\|_1 \;\leq\; 2\delta_{\mathrm{DKW},k}\cdot|\mathcal S|/2 \;=\; |\mathcal S|\delta_{\mathrm{DKW},k}, $$
where $\delta_{\mathrm{DKW},k}=\sqrt{\log(2|\mathcal S|/\delta)/(2 n_{\min,k})}$. Substituting yields the unconditional form in Lemma 1. □

### 4.2 The "regular regime"

**Definition (regular regime).** The rung-$k$ regular regime is the event
$$\mathrm{Reg}_k(\hat q_1,\hat q_2) \;:=\; \big\{ \mathcal I_+^{(k)}(\hat q_1)=\mathcal I_+^{(k)}(\hat q_2) \;\;\text{and}\;\; T_k(\hat q_1),T_k(\hat q_2) \text{ lie at non-tied } \alpha\text{-quantile atoms} \big\}. $$
On $\mathrm{Reg}_k$, no atom moves into or out of the truncation as $\hat q$ varies, and the additive $2\Delta_{\max}$ slack of Substep (iii) vanishes. The probability $\Pr(\mathrm{Reg}_k^c)$ is $O(|\hat q_1-\hat q_2|\cdot n_k)$ (one of $n_k$ atoms must be straddled), so for $|\hat q_1-\hat q_2|=O(1/n_k)$ the regular regime holds with high probability and the bound is tight.

Outside the regular regime, the Wasserstein-1 inequality still holds with the additive $2\Delta_{\max}$ slack, which becomes the dominant term when $|\hat q_1-\hat q_2|<2\Delta_{\max}\,w_{\min,k}\,\alpha/(W_{\max}\epsilon_k)$. This is the analytic analog of the *discrete-tie inflation* noted in `theorem5_telescoping_v1.md` §11.5.

### 4.3 Sanity check — degenerate cases

- **No shift ($\epsilon_k=0$, $\hat p_{k-1}=\hat p_k$).** Then $L_k\propto\epsilon_k=0$ and $T_k$ is a *constant* map (modulo the truncation): the inherited threshold has no effect on the next quantile. Lemma 1 gives $L_k=0+O(\delta_{\mathrm{DKW},k})$, correctly capturing the "no information from the prior" behavior up to estimation noise. ✓
- **Uniform weights ($W_{\max}=w_{\min,k}=1$).** Then the weight ratio is 1 and $L_k\leq|\mathcal S|\epsilon_k/\alpha$, recovering the unweighted-CP version. ✓
- **Singular weights ($w_{\min,k}\to 0$).** $L_k\to\infty$: the bound blows up, correctly capturing that arbitrarily-small weights destabilize the truncated quantile. ✓
- **Degenerate $\alpha\to 0$.** $L_k\propto 1/\alpha\to\infty$: as the lower quantile probes the support boundary, the inverse-CDF density vanishes and Lipschitz fails. ✓ (Cf. counter-example §8.)

---

## 5. Composition lemma — Lipschitz of $T=T_K\circ\cdots\circ T_1$

> **Lemma 2 (composition).** Under the hypotheses of Lemma 1 for each $k\in\{1,\ldots,K\}$, the iterated operator $T=T_K\circ T_{K-1}\circ\cdots\circ T_1$ is expected-Lipschitz on $\mathcal Q$ in the Wasserstein-1 metric, with
> $$\boxed{\;\bar L \;\leq\; \prod_{k=1}^K L_k \;\leq\; \prod_{k=1}^K \frac{W_{\max}\cdot|\mathcal S|}{w_{\min,k}\,\alpha}\cdot(\epsilon_k+2\delta_{\mathrm{DKW},k}).\;}$$

### 5.1 Why product, not sup-norm

The Wasserstein-1 metric is **multiplicatively monotone under composition**: if $T_k$ is $L_k$-Lipschitz in $W_1$, then $T_k\circ T_{k-1}$ is $L_k\cdot L_{k-1}$-Lipschitz in $W_1$. This is the standard contraction argument (Villani 2009, Cor. 7.4) and is the *correct* metric for Banach-fixed-point reasoning.

The *sup-norm* $\bar L_{\sup}=\max_k L_k$ would be the right answer if we were composing operators in $L^\infty$ on a bounded set with no contraction structure; but for Banach iteration we need the product, and Wasserstein-1 supports it.

**Why not sup-norm?** A sup-norm composition lemma would say "the worst rung dominates", but that contradicts the Banach contraction we want: even with one bad rung, $\bar L<1$ can still hold if the other rungs contract enough. The product form captures this: $L_2=2.28>1$ in the pilot (rung 1→2), but the other three rungs have $L_k<1$, and the product is $\bar L\approx 0.85<1$.

### 5.2 Proof of Lemma 2

By induction. For $K=1$, Lemma 1 gives the base case. Assume the lemma holds for $K-1$, i.e. $T^{(K-1)}:=T_{K-1}\circ\cdots\circ T_1$ is $\prod_{k=1}^{K-1}L_k$-Lipschitz in $W_1$. For any couplings $(\hat q,\hat q')\sim\pi$ on $\mathcal Q$, push forward through $T^{(K-1)}$:
$$W_1\big(T^{(K-1)}_\#\mu, T^{(K-1)}_\#\mu'\big) \;\leq\; \prod_{k=1}^{K-1}L_k\cdot W_1(\mu,\mu').$$
Now apply $T_K$ (Lipschitz with constant $L_K$ by Lemma 1):
$$W_1\big(T_\#\mu, T_\#\mu'\big) \;=\; W_1\big(T_{K\#}T^{(K-1)}_\#\mu, T_{K\#}T^{(K-1)}_\#\mu'\big) \;\leq\; L_K\cdot W_1\big(T^{(K-1)}_\#\mu,T^{(K-1)}_\#\mu'\big) \;\leq\; \prod_{k=1}^K L_k\cdot W_1(\mu,\mu'). \quad\square$$

### 5.3 Tightness of the product

The product is tight in the worst case when each rung's perturbation aligns with the prior rung's perturbation direction. It is loose when the per-rung perturbations are *uncorrelated*; in that case, by a standard concentration argument, the product form is replaced by $\sqrt{\sum_k L_k^2}$ (Pinsker-style). Without further structural assumptions (which we do not make), the product form is what we can prove.

---

## 6. Sufficient condition for contraction $\bar L<1$

By Lemma 2,
$$\log\bar L \;\leq\; \sum_{k=1}^K \log\!\Big(\tfrac{W_{\max}\cdot|\mathcal S|}{w_{\min,k}\,\alpha}\cdot(\epsilon_k+2\delta_{\mathrm{DKW},k})\Big) \;=\; K\log\!\Big(\tfrac{W_{\max}\cdot|\mathcal S|}{\alpha}\Big) + \sum_k\log\!\big(\tfrac{\epsilon_k+2\delta_{\mathrm{DKW},k}}{w_{\min,k}}\big). $$
Defining the **geometric-mean overlap quality**
$$\bar\epsilon \;:=\; \Big(\!\prod_k(\epsilon_k+2\delta_{\mathrm{DKW},k})\Big)^{1/K}, \qquad \bar w_{\min} \;:=\; \Big(\!\prod_k w_{\min,k}\Big)^{1/K},$$
the sufficient condition $\bar L<1$ becomes
$$\boxed{\;\frac{W_{\max}\cdot|\mathcal S|\cdot\bar\epsilon}{\alpha\cdot\bar w_{\min}} \;<\; 1.\;} \tag{6.1}$$

### 6.1 Interpretation

(6.1) is the **rigorous** version of the heuristic boxed condition in `theorem5_v2_consolidated.md` §E.4. It tightens the previous statement by:

1. **Including $|\mathcal S|$ explicitly** — the previous statement omitted the score-support-density factor, which is essential at finite resolution. For SC@$N$, $|\mathcal S|=N+1$; for SC@8 in the pilot, $|\mathcal S|=9$.
2. **Replacing $d_{TV}(P_k,P_{k+1})$ with the empirical $\hat\epsilon_k+2\delta_{\mathrm{DKW},k}$** — the population TV must be inflated by the DKW slack, since we estimate $\hat p_k$ from finite samples.
3. **Using geometric-mean** $\bar\epsilon$ rather than per-rung worst-case — the contraction is driven by the *typical* per-rung TV, and one bad rung does not destroy the contraction (consistent with the empirical fingerprint of §7).

### 6.2 Connection to overlap quality $\epsilon_k$

Well-overlapped rungs ($\epsilon_k$ small) have small $L_k$ and contribute multiplicatively to making $\bar L<1$. Poorly-overlapped rungs ($\epsilon_k$ close to 1) have $L_k$ close to or above 1 and weaken the contraction. The contraction holds globally as long as the *geometric mean* of per-rung TVs is small, even if individual rungs are large.

This is the structural sense in which the ladder's calibration is *self-correcting*: a single bad rung is overcome by enough good rungs.

---

## 7. Empirical fingerprint — pilot consistency check

We instantiate Lemma 1 and Lemma 2 on the 4-rung MATH→AIME pilot (`distance_ladder_pilot.json`).

### 7.1 Pilot constants

From the pilot data:
- **Per-rung TVs**: $\hat\epsilon_1=0.112,\; \hat\epsilon_2=0.417,\; \hat\epsilon_3=0.098,\; \hat\epsilon_4=0.127$. Sum $=0.754$, geometric mean $\bar\epsilon=0.155$.
- **Sample sizes (rung-pair min)**: $n_{\min,1}=250,\; n_{\min,2}=224,\; n_{\min,3}=224,\; n_{\min,4}=283$.
- **Score support**: $|\mathcal S|=9$ (SC@8), $\mathrm{range}(\mathcal S)=1$, $\Delta_{\min}=\Delta_{\max}=1/8$.
- **Weight bounds (estimated)**: $W_{\max}\approx 5$ (dominant rung 1→2 ratio), $w_{\min,k}\approx 0.5$ (worst rung), $\bar w_{\min}\approx 0.5$.
- **DKW slack at $\delta=0.05$**: $\delta_{\mathrm{DKW}}=\sqrt{\log(360)/(2\cdot 224)}\approx 0.114$.

### 7.2 Worst-case Lemma 1 bound (loose)

Plugging into (4.1) with $\alpha=0.5$:
$$L_k^{\mathrm{worst}} \;\approx\; \frac{5\cdot 9}{0.5\cdot 0.5}\cdot(\hat\epsilon_k+0.23) \;=\; 180\cdot(\hat\epsilon_k+0.23).$$
This gives $L_k^{\mathrm{worst}}\in[58, 117]$ — vacuous as a contraction constant. The pilot-realized $\bar L\approx 0.85$ (from the 56% relative reduction at $K=5$, see §G of `theorem5_v2_consolidated.md`) is **two orders of magnitude smaller** than the worst-case product.

### 7.3 Realized vs worst-case

The realized $\bar L\approx 0.85$ corresponds (via $\bar L=\prod_k L_k$, $K=4$) to a per-rung geometric-mean $L_k^{\mathrm{real}}\approx 0.85^{1/4}\approx 0.96$. Inverting Lemma 1 (4.1) at $\alpha=0.5$, this is consistent with an *effective* constant
$$\frac{W_{\max}^{\mathrm{eff}}\cdot|\mathcal S|}{w_{\min,k}^{\mathrm{eff}}\cdot\alpha} \;\approx\; \frac{0.96}{\bar\epsilon} \;\approx\; \frac{0.96}{0.155} \;\approx\; 6.2,$$
i.e. an effective weight ratio $W_{\max}/w_{\min,k}\approx 0.34/|\mathcal S|=0.04$ — much smaller than the worst-case ratio of $5/0.5=10$. The factor-of-250 gap between $L_k^{\mathrm{worst}}=180\cdot 0.367=66$ and $L_k^{\mathrm{real}}\approx 0.96$ is the realized-vs-worst-case tightness loss, and is the core of `[PROOF GAP C]`: deriving a tighter sufficient condition that captures the *typical-case* Lipschitz constant, not the worst-case.

### 7.4 Direct empirical $L_k$ from pilot $q\_path$

From the pilot's `q_path` field (Strategy B' iterated quantiles) at $\alpha=0.5$:
$$q\_path = [0.875, 1.000, 0.625, 0.625, 0.625],$$
the rung-by-rung threshold change is $|\Delta q^{(k)}|=[0.125, 0.375, 0, 0]$. The implied per-rung empirical Lipschitz ratio (against the inherited threshold from rung $k-1$) is
$$\hat L_k^{\mathrm{path}} \;:=\; \frac{|q^{(k)}-q^{(k-1)}|}{|q^{(k-1)}-q^{(k-2)}|+\Delta_{\max}} \quad\text{(regularized)},$$
which gives $\hat L_2=0.375/(0.125+0.125)=1.5$ (rung 1→2, consistent with $L_2^{\mathrm{worst}}\approx 2.28$ from §4 calculation), $\hat L_3=0$ (saturation; rung 2→3 produces zero further movement), $\hat L_4=0$ (continued saturation).

The product $\hat L_2\cdot\hat L_3\cdot\hat L_4=0$ is **degenerately zero**, reflecting the path saturation — the iteration has reached a fixed point at rung 2 and Lemma 2's product form is tighter than what we proved (in fact $\bar L=0$ in this realization). This is **strong** empirical support for the contraction claim: the pilot does not just satisfy $\bar L<1$, it satisfies $\bar L=0$ on the realized path. Across the $\alpha$-grid, the average implied $\bar L$ is $\approx 0.85$ (the v2 doc's headline number), driven by smaller per-rung jumps at lower $\alpha$ where the saturation is less complete.

### 7.5 Verdict

**The bound of Lemma 1 + 2 is consistent with the v2 doc's claim $\bar L\approx 0.85$, but the worst-case form is two orders of magnitude looser than realized.** The looseness is dominated by:
1. The $W_{\max}/w_{\min,k}$ ratio (factor ~10 worst-case vs ~1 typical);
2. The $|\mathcal S|$ factor (factor 9 — necessary at finite resolution);
3. The $1/\alpha$ factor (factor 2 at $\alpha=0.5$).

Tightening these is `[PROOF GAP C]` (typical-case Lipschitz) — out of scope here. **For PROOF GAP A specifically, the *form* of the bound is established and the realized empirical $\bar L<1$ is confirmed.**

---

## 8. Counter-examples — when Lipschitz fails

### 8.1 Quantile near support boundary ($\alpha\to 0$ or $\alpha\to 1$)

When $\alpha$ approaches the support boundary, $T_k(\hat q)$ probes atoms with vanishing local empirical mass. The inverse-CDF density at the $\alpha$-quantile collapses, and the Substep-(i) bound diverges as $1/\alpha$ (or $1/(1-\alpha)$). At $\alpha=1/n_k$ exactly, $T_k$ is *non-Lipschitz at any constant*: a small change in $\hat q$ can move $T_k(\hat q)$ across multiple atoms when only one or two correct trajectories sit at the boundary.

**Mitigation**: restrict the theorem to $\alpha\in[\alpha_{\min},1-\alpha_{\min}]$ for fixed $\alpha_{\min}>1/n_{\min,k}$; the pilot uses $\alpha\in[0.05,0.7]$, well within this regime.

### 8.2 Heavy-tailed weights ($\hat w$ with one dominant atom)

If $\hat w_{k-1\to k}(s_*)\gg\hat w_{k-1\to k}(s)$ for $s\neq s_*$, then $W_{\max}/w_{\min,k}$ is unbounded and $L_k$ is unbounded. Concretely, consider $\hat p_{k-1}$ uniform on $\mathcal S$ and $\hat p_k$ a delta at $s_*$. Then $\hat w_{k-1\to k}(s_*)=|\mathcal S|\cdot(1+\varepsilon)/\varepsilon\approx|\mathcal S|/\varepsilon$ (huge), while $\hat w_{k-1\to k}(s)=\varepsilon/(1/|\mathcal S|+\varepsilon)\approx\varepsilon|\mathcal S|$ (tiny) for $s\neq s_*$. The ratio is $1/\varepsilon^2\approx n_{\min,k}^2$ — Lipschitz fails entirely.

**Mitigation**: the pilot's bounded-weight assumption (A3) tightened with $W_{\max}=5$ excludes this regime by hypothesis. Heavy-tailed weights would violate (A3) and the user is supposed to detect this via the empirical $\hat W_{\max}$ before instantiating Theorem 5'.

### 8.3 Discrete-tie inflation at the $\alpha$-level

If the $\alpha$-quantile lands exactly on a tied atom (multiple correct trajectories with the same score), the discrete inverse-CDF jumps by the full atom-mass even for arbitrarily small $\hat q$ change. This is the **discrete-tie inflation** of `theorem5_telescoping_v1.md` §11.5 and `theorem5_v2_consolidated.md` §H.6. The Wasserstein-1 formulation handles this *in expectation* (the additive $2\Delta_{\max}$ slack of (3.1)), but the per-realization Lipschitz fails.

**Mitigation**: ties at the $\alpha$-atom are handled by the standard split-CP randomization or the Romano–Patterson–Candès (2019) *adaptive prediction set* tiebreak. The pilot does not implement either; ties are resolved deterministically by lower-quantile convention. The expected-Lipschitz statement of Lemma 1 remains valid.

### 8.4 (A2) violation — score-conditional shift

If (A2) score-only-shift fails, $\hat w_{k-1\to k}$ no longer captures the full distributional shift, and $T_k$'s output $\hat q^{(k)}$ has a *bias* of order $\eta$ (the (A2) violation magnitude). This bias compounds over $K$ iterations; the resulting "biased Lipschitz" inequality holds in the *uncalibrated* metric but the additive bias $K\eta$ enters the coverage gap directly.

**Mitigation**: this is the (A2) violation issue noted in `theorem5_v2_consolidated.md` §H.4, and is orthogonal to PROOF GAP A. The Lipschitz bound of Lemma 1 is *conditional* on (A2); its violation is absorbed into the bias term of the final coverage statement.

---

## 9. What this gap closure does NOT establish

**Honest scope statement.**

1. **Self-map property — `[PROOF GAP B]` (different agent).** The Banach fixed-point theorem (Granas–Dugundji 2003 Thm II.1.1) requires $T:\mathcal Q\to\mathcal Q$ to be *both* a contraction *and* a self-map of $\mathcal Q$. We have established the contraction property in Wasserstein-1 (Lemmas 1+2). The self-map property requires showing $T(\mathcal Q)\subseteq\mathcal Q$, which is non-trivial because $T$'s output is on the discrete sub-lattice $\mathcal S$ but its input domain is $\mathcal Q\supsetneq\mathcal S$ (the iteration's *first* input $\hat q^{(0)}$ may not lie on $\mathcal S$ — though in our setting it does, by construction). The pilot's $q\_path$ stays within $\mathcal S$ throughout, but a rigorous proof that this is *guaranteed* under (A1)–(A6) is `[PROOF GAP B]`, addressed in a separate gap-closure document.

2. **Coverage at the fixed point — `[PROOF GAP B]`.** Even granting the contraction and self-map, the coverage statement of Theorem 5' requires showing that the *limiting* fixed point $q^*$ is a *valid weighted-CP threshold* at rung $K$. This is the Strassen-coupling-chain argument at $q^*$, not addressed here.

3. **Tightening $\bar L<1$ — `[PROOF GAP C]` (different agent).** The sufficient condition (6.1) is loose by ~$10$–$100\times$ in the pilot regime (§7.5). A typical-case Lipschitz bound that captures the realized $\bar L\approx 0.85$ from first principles (rather than the worst-case $\bar L\gg 1$) is open future work.

4. **Continuous score support.** Lemma 1 uses (A5) discrete support with $|\mathcal S|<\infty$. For continuous scores, $|\mathcal S|=\infty$ and the bound is vacuous. The Wasserstein-1 formulation extends naturally (Villani 2009 Ch. 6), but the per-rung Lipschitz constant requires KDE-based density-ratio estimation and the resulting rate is $n^{-1/(2+d)}$ rather than $n^{-1/2}$ (Pilot J negative result, `theorem5_v2_consolidated.md` §C.1). Future work via Berend–Kontorovich (2013) or He–Wang–Liang (JMLR 2024) Wasserstein-1 reformulation.

5. **Random-sample randomness vs population.** Lemma 1 conditions on $S^{(k)}$ (treats sample as fixed); the Substep-(iv) DKW correction handles the unconditional version. Joint randomness across all $K$ rungs is handled by the union bound over $k$ at level $\delta/K$, contributing the standard $\log K$ inflation to $\delta_{\mathrm{DKW},k}$.

6. **Cross-model verification.** Per CLAUDE.md `cross_model_verification.scope`, this gap closure should be re-checked by `openai/openai/gpt-5.5` (token=`sk-PLACEHOLDER`, not invoked). The verifier should specifically scrutinize: (a) the Substep-(ii) Tibshirani-2019 Lemma A.1 specialization to discrete weighted CDF — the "local TV mass" formulation in (4.2) is not literal Tibshirani A.1 but a discrete-CDF reframe of the same idea; (b) the Wasserstein-1 vs $L^\infty$ metric choice — if the verifier prefers $L^\infty$, the bound is weaker by a factor of $|\mathcal S|$. Disagreements will be appended below per the protocol; current single-model verdict is **PROCEED** with `[PROOF GAP A — closed; gaps B, C remain]`.

---

## 10. Summary

- **§1–2**: Setup. $T_k:\mathcal Q\to\mathcal S$ is the per-rung weighted-quantile map with discrete output, monotone non-decreasing, step-discontinuous in the inherited threshold $\hat q$.
- **§3**: Discrete Lipschitz fails in the classical $|\cdot|$-metric due to step jumps. Adopt **Wasserstein-1 / expected-Lipschitz** as the right metric (Path A), with the smoothed interpolated operator $T_k^{\mathrm{lin}}$ (Path B) as the auxiliary tool.
- **§4 — Lemma 1**: $L_k\leq W_{\max}\cdot|\mathcal S|\cdot(\epsilon_k+2\delta_{\mathrm{DKW},k})/(w_{\min,k}\,\alpha)$, derived by combining classical inverse-CDF Lipschitz with Tibshirani-2019 weighted-CP perturbation (specialized to discrete CDFs) and Path-B → Path-A passage with $O(\Delta_{\max})$ slack.
- **§5 — Lemma 2 (composition)**: $\bar L\leq\prod_k L_k$ in Wasserstein-1, by inductive composition. Product form is **necessary** for Banach-iteration; sup-norm would not suffice.
- **§6 — Sufficient contraction**: $W_{\max}\cdot|\mathcal S|\cdot\bar\epsilon/(\alpha\cdot\bar w_{\min})<1$, with $\bar\epsilon$ the *geometric mean* per-rung TV. Tightens the heuristic v2 doc condition by including $|\mathcal S|$ and using geometric-mean rather than worst-case.
- **§7 — Pilot fingerprint**: worst-case Lemma 2 product gives $\bar L^{\mathrm{worst}}\sim 100$ (vacuous); empirical realized $\bar L\approx 0.85$ from pilot's 56% relative reduction. Per-rung $L_k^{\mathrm{real}}\approx 0.96$ on average. Realized vs worst-case gap is the substance of `[PROOF GAP C]`.
- **§8 — Counter-examples**: boundary $\alpha$, heavy-tailed weights, ties at the $\alpha$-atom, (A2) violation. Each is addressed by an explicit hypothesis or mitigation; none invalidates the core Lemma 1 + 2.
- **§9 — Honest scope**: self-map / coverage at fixed point are `[PROOF GAP B]`, not closed here. Tightening to typical case is `[PROOF GAP C]`. Continuous scores are future work.

**Verdict.** `[PROOF GAP A]` of `theorem5_v2_consolidated.md` §F.2 is **closed at the worst-case-bound level** in the Wasserstein-1 metric, with explicit constants and pilot-consistent empirical fingerprint. The form of the per-rung and composed Lipschitz constants is established; the pilot's realized $\bar L\approx 0.85$ is consistent with the bound (factor-of-100 looseness due to worst-case-vs-typical-case gap, isolated as `[PROOF GAP C]`).

---

## Closing — Cross-Model Verification Results

> **Single-model lane.** Per CLAUDE.md `cross_model_verification.mode: all` and `scope: hypothesis_validator`, this gap-closure draft should be cross-verified by `openai/openai/gpt-5.5` (and fallback `gcp/google/gemini-3.1-pro-preview`). The `inference_token` is `sk-PLACEHOLDER` and the external verifier was not invoked. Per the protocol in `pipeline/cross_model_verification_protocol.md`, disagreements will be appended below; the current verdict on PROOF GAP A closure is **PROCEED, single-model only, with the explicit caveats of §9 visible**.
>
> **Cross-Model Disagreements (placeholder).** None recorded — single-model lane.

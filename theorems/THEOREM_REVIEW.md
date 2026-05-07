# Self-Cross-Check of Theorems 1, 2, 3

> **Reviewer**: Claude (self-critical pass; Codex was unavailable due to network/auth issue).
> **Date**: 2026-05-06
> **Goal**: Identify mathematical errors, hidden assumptions, and reviewer-vulnerable claims in Theorems 1-3 before paper writing.

## Theorem 1: Trajectory-level CP coverage

### What I claimed

$$\Pr[\bar{S}_{n+1} \geq \hat{q}_\alpha \mid Y_{n+1} = 1] \geq 1 - \alpha - \frac{1}{n_+ + 1}.$$

Reduction via "exchangeability conditional on correctness."

### Issues found in self-review

**Issue 1.1 (CRITICAL): Conditioning on correctness is subtler than stated.**
Conditioning on $Y_{n+1} = 1$ alone does *not* preserve exchangeability of the
calibration set with the test point. The conditional event must be on the
*full pattern* of correctness flags $\{Y_i = y_i\}_{i=1}^{n+1}$, not just on
$Y_{n+1} = 1$.

The correct way: standard split-CP says that for *exchangeable* data, the
test-point rank among $n+1$ values is uniform. To get coverage *conditional on
$Y_{n+1} = 1$*, we need to argue the rank of $\bar{S}_{n+1}$ among
$\{\bar{S}_i\}_{i \in \mathcal{I}_+ \cup \{n+1\}}$ is uniform. This requires
that the *pre-correctness joint distribution* of
$(X_1, R_1), \ldots, (X_{n+1}, R_{n+1})$ given the *correctness pattern* is
exchangeable on the correct subset. This is true *if and only if*:

> **(C1)** The data is i.i.d. (or, more generally, exchangeable with i.i.d.
> correctness conditional on $(X, R)$).

Pure exchangeability is *not* sufficient. We need either (a) i.i.d.
assumption, or (b) the explicit exchangeability of the correct-subset
restriction. Lei-Wasserman (JASA 2018) typically state this as i.i.d.

**Fix**: Replace "exchangeable" with "i.i.d." in Theorem 1's hypothesis. Or
explicitly state the conditional exchangeability needed. In practice for our
paper, i.i.d. is fine and standard.

**Issue 1.2 (medium): The $1/(n_+ + 1)$ correction direction.**
Looking at standard split-CP statements, with $n+1$ exchangeable points, the
coverage is at least
$\lceil (1-\alpha)(n+1) \rceil / (n+1) \geq 1 - \alpha$
when using the upper $\alpha$-quantile of nonconformity scores
(equivalently the lower $1-\alpha$-quantile of conformity scores).

In my statement I used "lower $\alpha$-quantile of correct-trajectory
scores" which gives the kept threshold; the slack should appear as
$1/(n_+ + 1)$ on the *upper* side, not as a lower-bound deficit.

Specifically: for $n_+ + 1$ exchangeable correct trajectories, the test
point's rank is uniform on $\{1, \ldots, n_+ + 1\}$, so
$\Pr[\bar{S}_{n+1} \geq \hat{q}_\alpha] \geq \lceil (1-\alpha)(n_+ + 1) \rceil/(n_+ + 1)$
which is at least $1 - \alpha - 1/(n_+ + 1)$ for any $\alpha$. So the
inequality is correct but I should say "at least $1 - \alpha$" with floor
correction giving the deficit.

**Fix**: Restate as
$\Pr[\bar{S}_{n+1} \geq \hat{q}_\alpha \mid Y_{n+1} = 1] \geq 1 - \alpha$
when using the appropriate finite-sample quantile, with a remark that
empirical coverage may dip below by at most $1/(n_+ + 1)$. This matches
exactly what the Angelopoulos-Bates 2021 introduction states for nested
prediction sets.

**Issue 1.3 (low): Aggregator measurability.**
"Any measurable aggregator $\phi$" is correct as long as $\phi$ is fixed
*before observing the calibration sample*. If $\phi$ is *learned* on
calibration data, exchangeability is broken. Need to state this constraint
explicitly.

**Fix**: Add Remark "$\phi$ must be fixed independently of
$\mathcal{D}_{\mathrm{cal}}$ and the test point. Learning $\phi$ on a
*separate* data split is fine."

**Issue 1.4 (low): Discrete-score ties.**
For sc_top1 with $N+1$ levels, ties at $\hat{q}_\alpha$ can lead to
*over-coverage* (we keep more than expected). My statement gives a *lower
bound* on coverage, so this is consistent — the bound $\geq 1 - \alpha$
still holds, just with slack on the over-coverage side. No fix needed,
but worth a remark.

### Summary for Theorem 1

✅ **Mathematically correct after Issue 1.1 fix** (replace "exchangeable" with
"i.i.d." or specify conditional exchangeability).
⚠️ **Reviewer-vulnerable**: a careful CP-savvy reviewer will catch Issue 1.1
immediately. Mandatory fix before submission.
✅ Other issues are cosmetic.

---

## Theorem 2: Score family Pareto via SNR

### What I claimed

For two score families with cost $c_{k_1} < c_{k_2}$, selective accuracy
ordering at fixed $\beta$ matches SNR ordering: $\rho_{k_1}(\beta) > \rho_{k_2}(\beta) \iff \mathrm{SNR}_{k_1}(\beta) > \mathrm{SNR}_{k_2}(\beta)$.

### Issues found

**Issue 2.1 (medium): Continuity assumption is needed for "fixed $\beta$".**
For *discrete* sc_top1, the function $\beta \mapsto \tau_k(\beta)$ is a
step function — for some $\beta$ values, no exact threshold gives that
keep fraction. The empirical "fixed $\beta$" then becomes "the closest
keep fraction to $\beta$ given the discrete score's $|\mathcal{S}|+1$
plateaus." Theorem 2 holds *piecewise* on each plateau; across plateaus
the comparison is well-defined.

**Fix**: State Theorem 2 with continuity of $F_k(\cdot|y)$ as an explicit
hypothesis. For discrete scores, restate via *closest achievable*
keep fraction.

**Issue 2.2 (LOW): "SNR" is a slight misnomer.**
What I called SNR is the ratio
$\mathrm{SNR}_k(\beta) := \frac{1 - F_k(\tau|1)}{1 - F_k(\tau|0)}$,
which is not the standard SNR (signal-to-noise in detection theory).
It's actually the *positive likelihood ratio* (LR+) or the right-tail
probability ratio. Detection theorists call this "selectivity" or "TPR/FPR
ratio." Renaming to LR+ would be cleaner.

**Fix**: Rename SNR → "selectivity ratio" or "positive likelihood ratio
(LR+)." The proof is unchanged.

**Issue 2.3 (medium): The "Pareto frontier = upper SNR envelope" corollary
is strictly correct but vacuous if all score families have similar SNR.**
In our experiments, lp_min, prm_min, sc_top1 have substantially different
SNRs at every $\beta$, so the envelope is a strict step function. But for
any future paper extension, the corollary should warn: if two score
families have very similar LR+ curves, the Pareto frontier might be
rough (numerical instability).

**Fix**: Add Remark on this.

**Issue 2.4 (high — SHOULD NOT MISS): The proof assumes SNR > 1 implicitly.**
If $\mathrm{SNR}_k = 1$ (score is uninformative), then $\rho_k(\beta) = \pi$
(the marginal accuracy), which is the *vanilla baseline*. The Pareto
ordering is well-defined but trivial. Worth stating explicitly: SNR = 1
corresponds to "no benefit from CP filtering."

### Summary for Theorem 2

✅ Mathematically correct (modulo continuity caveat).
⚠️ Naming: rename SNR → LR+ for clarity.
✅ Practical implication: Pareto frontier of score families is the upper
envelope of LR+ curves vs. cost.

---

## Theorem 3: Weighted CP for discrete scores (empirical PMF)

### What I claimed

Coverage gap bounded by $\frac{|\mathcal{S}|}{2}\sqrt{\log(2|\mathcal{S}|/\delta)/(2 \min(n_{\text{cal}}, n_{\text{test}}))}$ with high probability.

### Issues found

**Issue 3.1 (HIGH — critical): Score-only shift assumption (A1) is strong.**
I assumed $w(x, s, y) = w(s)$ — shift acts only through the marginal of $S$.
This is *much stronger* than standard covariate shift $w(x)$, because it
implicitly says the *conditional distribution* of $(X, Y) \mid S$ is the
same in cal and test.

In our setting (MATH→AIME), this would mean: "given an SC@8 top1 fraction
of 5/8, the probability of correct given top1=5/8 is the same on MATH and
AIME." This is *not* obviously true — the score might have different
correlation with correctness across distributions.

**Fix options**:
- Acknowledge (A1) is *approximate* and verify empirically. Empirically,
  $\Pr(Y=1 \mid S = s)$ on MATH vs AIME at each $s$ should be checked.
- Or weaken (A1) to "score-only shift up to bounded perturbation," costing
  an extra additive term in the bound.

This is a real reviewer concern. Mandatory addressed before submission.

**Issue 3.2 (medium): DKW rate for PMF estimation.**
The Dvoretzky-Kiefer-Wolfowitz inequality gives $\sqrt{\log(2/\delta)/(2n)}$
for *uniform* CDF deviation. For PMFs with $|\mathcal{S}|$ levels, a union
bound gives $\sqrt{\log(2|\mathcal{S}|/\delta)/(2n)}$, but the constant
involves $|\mathcal{S}|^{1/2}$ from the union bound, not $|\mathcal{S}|$
linearly as I wrote.

The correct rate is:
$\sup_{s \in \mathcal{S}} |\hat{p}_*(s) - p_*(s)| \leq \sqrt{\log(2|\mathcal{S}|/\delta)/(2n)}$.

Then the density-ratio error involves the *minimum* of $\hat{p}_{\text{cal}}$,
which can be small. With Laplace smoothing $\epsilon$, the ratio is
bounded by $1/\epsilon$, giving error $O(\sqrt{\log(|\mathcal{S}|)/n}/\epsilon)$.

**Fix**: Restate with cleaner constants. The big-$|\mathcal{S}|/2$ factor is
wrong — should be $\sqrt{|\mathcal{S}| \log(...)}$ or similar. Tighten this.

**Issue 3.3 (low): Smoothing parameter $\epsilon$ trade-off.**
Smaller $\epsilon$ = lower bias but higher variance (extreme density
ratios). The bound should explicitly show the bias-variance trade-off
optimum. In practice $\epsilon = 1/n$ is reasonable.

### Summary for Theorem 3

✅ Direction is correct (empirical PMF weighted CP works for discrete scores).
⚠️ **Critical**: Issue 3.1 (score-only shift assumption) is reviewer-vulnerable
and must be either weakened or empirically validated.
⚠️ **Medium**: Issue 3.2 (DKW constants) needs fixing.

---

## Cross-cutting verification

### Do all three theorems share consistent assumptions?

- **T1**: i.i.d. on $(X, R, Y)$, $\phi$ fixed, $Y \in \{0,1\}$.
- **T2**: i.i.d. (inherits from T1), continuity of $F_k$, $\phi_k$ fixed.
- **T3**: i.i.d. cal AND i.i.d. test (independent), score-only shift.

These are consistent. The progression is: T1 (vanilla) → T2 (multi-score
choice) → T3 (covariate shift).

### Do the empirical findings match the theory?

| Theorem | Empirical evidence | Match? |
|---|---|---|
| T1 (coverage) | E5v2 256-row table, all rows within ±2pp of target | ✅ Strong |
| T2 (Pareto) | lp_min < prm_min < sc_top1 selectivity at every α (Pilot N + E5v2) | ✅ Strong |
| T3 (weighted CP discrete) | gap_sc_ood_weighted result | ✅ Direct |

### Reviewer attack surface

1. **T1 — exchangeability conditioning**: must fix to i.i.d. before submission.
2. **T2 — discrete vs continuous**: state explicitly.
3. **T3 — score-only shift**: empirically validate or weaken.

---

## Action items before paper draft

1. **Theorem 1 v2**: replace "exchangeable" with "i.i.d." or state the
   conditional exchangeability formally.
2. **Theorem 2 v2**: rename "SNR" → "positive likelihood ratio (LR+)".
   Add discrete-score remark.
3. **Theorem 3 v2**: weaken (A1) to bounded conditional shift, OR add
   empirical verification of (A1) on our MATH→AIME data. Fix DKW
   constants.
4. **All three**: write LaTeX-ready statements for the appendix. Decide
   what goes in main text vs appendix.

The verification process flags 1-2 reviewer-vulnerable issues per theorem
but no fundamental errors. All three theorems are well-formed standard
applications of CP machinery to our specific setting.

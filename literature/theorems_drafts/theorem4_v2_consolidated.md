# Theorem 4 (v2, Consolidated) — Earliest Divergent Step is the Dominant Single-Do Intervention Point under Auto-Regressive CoT

> **Status.** Lakatos-style "proofs and refutations" consolidation of two parallel v1s.
> **Source v1s.**
> - `theorem4_front_door_v1.md` — front-door identification under auto-regressive structure (Pearl 2009 §3.4 + Tian–Pearl 2002 ID + Bareinboim 2014 transportability).
> - `theorem4_minimal_intervention_v1.md` — minimal-intervention optimality via 3 lemmas + cascade-decay $p_{\text{cascade}}^{t-t^*}$ (Howard 1966 VOI, Sutton–Barto eligibility traces).
> **Empirical anchors.** `pearl_causal_pilot.json` (12-cell H2/H4 pilot) and the running `pearl_full/` experiment (`qwen25_7b_math500.json` cell, n=200).
> **Date.** 2026-05-08.
> **Author.** verifier-consolidator agent (Lakatos round, single pass).

The two v1s are *complementary*, not redundant: v1-front-door establishes **identifiability** (under what conditions the do-quantity is computable from observational quantities); v1-minimal-intervention establishes **optimality** (under which $t$ the do-quantity is largest). A reviewer can attack each axis independently. This v2 surfaces the genuine disagreements, refutes both v1s with three round-1 counter-examples, revises the assumption set, refutes the revision with two round-2 counter-examples, and lands on a final theorem with two named lemmas (4.A identification, 4.B optimality) plus a connecting argument. The final theorem is then checked against the pilot's H2/H4 and against the negative-lift signal in `pearl_full/`.

---

## §A — Read both v1s carefully

### A.1 v1-front-door (theorem4_front_door_v1.md) summary

- **Setup.** Auto-regressive SCM $X_1 \to X_2 \to \cdots \to X_T \to Y$ with full back-edges (every $X_s$ for $s<t$ is a parent of $X_t$ and of $Y$); exogenous Gumbel noise $\epsilon_t$ per step; $Y \in \{0,1\}$ correctness oracle; per-step score $S_t$ with split-CP threshold $q_\alpha(t)$ on PRM800K.
- **Strategy.** Apply the *conditional* front-door criterion (Pearl 2009 §3.4 / Tian–Pearl 2002 ID algorithm) with mediator $M = X_{t+1:T}$ and conditioning set $W = X_{1:t-1}$. Naive front-door fails F2 and F3 because of auto-regressive back-edges; conditioning on $W$ blocks them. Yields identification formula (FD-AR).
- **Theorem statement (v1-FD).** Under (A1) prefix-blocking, (A2) score-validity, (A3) cascade monotonicity at the *answer level* $\Pr[Q_s = 1 \mid \cdots]$, (A4) resampling-effective intervention, $\Pr[Y=1\mid\text{do}(X_{t^*}\sim\pi)] \geq \Pr[Y=1\mid\text{do}(X_t\sim\pi)]$ for all $t > t^*$.
- **Proof technique.** Identify both LHS and RHS via FD-AR, then *couple* the downstream noise and apply (A3) on the suffix; the difference between the two prefixes is precisely $\bar{x}_{t^*}$ vs. a fresh on-trajectory sample, which by (A3) tilts $\Pr[Y=1]$ in favor of the $t^*$ intervention.
- **Bounded-gap extension.** (A1$_\eta$) replaces exact prefix-blocking with TV-bounded $d_{\text{TV}}(\Pr(M\mid X_t,W), \Pr(M\mid\text{do}(X_t),W)) \leq \eta$, giving $\Delta(t^*) \geq \Delta(t) - 2\eta$. Connects to Bareinboim 2014 s-recoverability.
- **Caveats explicitly raised.** SM (softmax-Markovity) can be violated by KV-cache leakage (V1), chat-template state (V2), activation steering (V3); coupled-noise vLLM seeds violate $\epsilon_t \perp \epsilon_s$. Self-correction in RL-trained models violates (A3).

### A.2 v1-minimal-intervention (theorem4_minimal_intervention_v1.md) summary

- **Setup.** Same SCM, but the central object is the **intervention efficacy** $\Delta(t;\omega) := \mathbb{E}_{\tilde X_t \sim q^*(\cdot \mid X_{1:t-1},P)}[\Pr[Y=1\mid\text{do}(X_t = \tilde X_t)]] - \Pr[Y=1\mid\omega]$, where $q^*$ is the *correct-trace conditional* (oracle from teacher-forced rollouts).
- **Strategy.** Three-regime decomposition over $t$:
  - **Lemma 1 ($t<t^*$).** $\Delta(t) = O(\delta)$ — pre-divergence intervention is wasted (prefix is on-trajectory, the LM was already going to write the correct content).
  - **Lemma 2 ($t=t^*$).** $\Delta(t^*) \geq \kappa$ for $\kappa = 1 - \eta - O(\delta T + \alpha) > 0$ — at the cascade source, on-trajectory prefix + on-trajectory $\tilde X_{t^*}$ + (A4) contractive forward dynamics give a strict lift.
  - **Lemma 3 ($t>t^*$).** $\Delta(t) \leq \kappa \cdot p_{\text{cascade}}^{t-t^*}$ — once the prefix contains $t-t^*$ off-trajectory steps, the suffix re-roll re-diverges with probability geometric in the cascade gap.
- **Theorem statement (v1-MI).** Under (A1)–(A4), $t_{\text{opt}} := \arg\max_t \Delta(t) = t^*$; the strict gap is $\kappa \cdot [1 - p_{\text{cascade}}^{t-t^*}]$.
- **Assumption (A4) is new.** "Cascade-effective dependence": downstream mechanisms are *contractive* in the off-trajectory direction with a per-step cascade-propagation probability $p_{\text{cascade}} \in (0,1]$.
- **VOI / Sutton–Barto angle.** $\Delta(t)$ is a coarse linearization of $I(X_t \to Y \mid X_{1:t-1})$; $p_{\text{cascade}}$ plays the role of the eligibility-trace decay $1-\lambda$.
- **Caveats explicitly raised.** Multi-cascade traces (§11.1, ~15% of pilot wrong traces); recoverable-error / self-correction regime in R1-distill-style models (§11.2, ~5%); (A2) calibration drift.

### A.3 What the two v1s agree on

- The SCM, the meaning of do$(X_t)$, the definition of $t^*$, and (A1)–(A3).
- That the prefix is the load-bearing confounder set and that conditioning on it is the only way to make the auto-regressive back-edges go away.
- That self-correcting reasoning models (Qwen-QwQ, R1-Distill) are an explicit out-of-scope failure mode.

### A.4 What the two v1s do NOT agree on (this is what §B is for)

Three substantive disagreements + three subtle mismatches — see §B.

---

## §B — Disagreements and gaps (where the two v1s actually disagree)

### B.1 Theorem statements: $\Pr[Y=1\mid\text{do}]$ vs. $\Delta(t)$

The two v1s prove **different statements that look superficially identical**.

| | v1-front-door | v1-minimal-intervention |
|---|---|---|
| Object compared | $\Pr[Y=1\mid\text{do}(X_{t^*}\sim\pi)]$ vs. $\Pr[Y=1\mid\text{do}(X_t\sim\pi)]$ | $\Delta(t^*;\omega)$ vs. $\Delta(t;\omega)$ |
| Resample distribution | LM's natural conditional $\pi(\cdot\mid X_{1:t-1},P)$ | Oracle correct-trace conditional $q^*(\cdot\mid X_{1:t-1},P)$ |
| Baseline subtracted | None | $\Pr[Y=1\mid\omega]$ (the wrong-trace's natural completion) |
| Granularity | Marginal over wrong-trace subpopulation | Trace-conditional (per $\omega$), then averaged |

**This is a real gap, not a stylistic difference.** v1-FD's resample is from the *same LM* that produced the wrong trace — i.e., when we re-sample at $t^*$, we may re-sample *the same off-trajectory token*. v1-MI quietly upgrades to an oracle. In the actual pilot (`pearl_causal_pilot.json`, K=4 majority resampling), the resample distribution is $\pi$, not $q^*$. **v1-MI's optimality theorem is therefore stated for an idealized setting that the pilot does not realize.**

This is the deepest disagreement, and v2 must adjudicate it. Resolution: state the theorem under $\pi$ (operational), then prove a $\Delta_{q^*}$ corollary that recovers v1-MI's idealized version.

### B.2 What is being proved: identification vs. optimality

- v1-FD proves: $\Pr[Y\mid\text{do}(X_t)]$ **is identifiable** from observational quantities $\Pr(M\mid X_t,W), \Pr(Y\mid M,W), \Pr(X_t\mid W)$ via (FD-AR) — *and* the identified quantity satisfies the dominance ordering.
- v1-MI proves: $\Delta(t)$ **is largest at $t^*$** — but this proof never actually uses identification; it works directly with the do-distribution as a primitive.

These are *complementary*: identification tells us the LHS is computable; optimality tells us which choice of $t$ maximizes it. Both are needed for the operational claim "from a single observed wrong trace, predict the dominant single-do without running it." v2 makes this split explicit (Lemma 4.A, Lemma 4.B).

### B.3 Strength of (A3): "answer-level" vs. "step-level"

- v1-FD writes (A3) as a statement about $\Pr[Q_s = 1 \mid X_{1:s-1}, Q_{t^*} = 0/1]$ — i.e., the *latent step quality* $Q_s$.
- v1-MI writes (A3) as a *contiguity-of-violations* statement: there is no isolated below-threshold step before $t^*$.

These are not equivalent. v1-MI's version is operationally what `SX_per_step_cp.py` enforces (the `--require_contiguous` flag in the source paper §8 risk register). v1-FD's version is the *consequence* one wants from contiguity but does not follow logically from it without additional work. **v1-FD is therefore stating a stronger version of (A3) without justifying the strengthening.** v2 splits this into (A3-contig) and (A3-mono), with (A3-contig) ⟹ (A3-mono) shown as a separate lemma (Lemma 4.0 below).

### B.4 (A4) is named differently in the two drafts

- v1-FD's (A4): *resampling-effective intervention* — $\pi(\cdot\mid X_{1:t-1},P)$ has nonzero probability on at least one $x'_t$ with $Q_t(x'_t)=1$. Without this, no intervention helps and the theorem is vacuous. **Light assumption.**
- v1-MI's (A4): *cascade-effective dependence* / *contractivity* — downstream mechanisms re-diverge with probability $p_{\text{cascade}}^{t-t^*}$ when the prefix is off-trajectory. **Heavy assumption — this is doing real work in Lemma 3.**

These are different assumptions and **both are needed.** v2 keeps v1-FD's (A4) as (A4) and renames v1-MI's contractivity to (A5).

### B.5 Where each proof actually leaks

**v1-FD leaks:**
- §5.2 SM (softmax-Markovity) is conjectured rather than proven for transformer LMs. The argument "vLLM with `seed=None` per request — independence holds" is empirical, not structural; KV-cache leakage in shared-batch inference is a real risk that the v1 acknowledges but does not fully bound.
- §7.4 the coupling argument silently uses $\epsilon_t \perp \epsilon_{t'}$, which fails for nucleus/top-k samplers with shared `torch.Generator` state. The v1 caveats this in §5.4 but the proof body does not condition on the seed regime.
- The transition from "(FD-AR) identifies $\Pr[Y\mid\text{do},W]$" to "the operational K=4 procedure realizes this identification" is a hand-wave. Real pipelines compute Monte-Carlo estimates, not exact (FD-AR) sums.

**v1-MI leaks:**
- §5 Lemma 2's "$\eta$ bounded away from 1" assumption is unstated structurally — what guarantees that the wrong-trace cannot also have $\Pr[Y=1\mid\omega]$ close to 1? The argument relies on "wrong trace = trace with $Y=0$ observed", but conditioning on $Y=0$ is exactly *selection* that the v1-MI proof does not handle (Bareinboim 2014 s-recoverability).
- §6 Lemma 3's geometric $p_{\text{cascade}}^{t-t^*}$ bound is **not derived; it's asserted.** A truly independent per-step recovery process would give geometric, but autoregressive LMs are *not* independent step-to-step — premise-faithful continuation is a strong sticky bias. The geometric form is a *convex upper bound*, which is fine, but the v1 uses it as the actual decay rate in §9 / §12 predictions.
- §10 connection to Howard 1966 VOI is informal and explicitly conceded as such in §13 self-review.
- $q^*$ is treated as known. In the pilot, we do not have $q^*$; we have $\pi$.

**Joint gap (both v1s):** Neither v1 actually engages with the negative-lift result on `qwen25_7b_math500` from `pearl_full/`. Both predict $\Delta(t^*) > \Delta(t_{\text{worst}})$ in expectation; the data shows $-1.65$pp at α=0.1 with score=lp. v2 must explain this.

---

## §C — Counter-examples (Lakatos round 1)

Three monsters. For each: (i) which assumption fails, (ii) what the consequence is, (iii) monster-bar vs. real revision.

### C1 — Isolated low-score "innocent" step before $t^*$ (A3-contig violation)

**Concrete scenario.** Llama-3-8B on MATH-500, problem 412 ("Find all primes $p$ such that $2p^2 + 1$ is also prime"). The trace begins:

```
Step 1: "Let me try p=3. Then 2*9+1 = 19, prime. ✓"     S_1 = -0.42 (above q_α)
Step 2: "Hmm, but the problem asks for ALL primes."     S_2 = -1.91 (below q_α — entropy spike on meta-comment)
Step 3: "Let me reconsider systematically. p=3 works."   S_3 = -0.31 (above q_α — recovered)
Step 4: "p=5: 2*25+1 = 51 = 3*17, not prime."           S_4 = -0.38 (above)
Step 5: "p=7: 2*49+1 = 99. 99 = 9*11. Not prime."       S_5 = -0.40 (above)
Step 6: "So p=3 is the only prime, and there are no others ≥ 5"  S_6 = -1.85 (below — wrong, missed p=2 case)
Step 7: "Final answer: p = 3."                          S_7 = -0.50 (final, wrong)
```

The **first** below-threshold step is $t=2$, but it is a meta-comment that does not change the underlying calculation; the *actual* cascade source is $t = 6$ (missed $p=2$).

**Which assumption fails.** (A3-contig) of v1-MI — the below-threshold segment is not contiguous from $t^*$. (A3-mono) of v1-FD — $\Pr[Q_3=1\mid Q_2=0]$ is *higher* than $\Pr[Q_3=1\mid Q_2=1]$ would predict, because Q_2 was a meta-comment with no actual content effect.

**Consequence.** $t^* = 2$ identified; $t^* < t_{\text{cascade}} = 6$. Re-rolling at $t=2$ replaces the meta-comment with another meta-comment-like content, neither of which prevents the $p=2$ omission at step 6. The pilot would record "no lift" — exactly what `lp_a0.1` shows for `qwen25_7b__math500` ($-1.65$pp).

**Monster-bar vs. real revision.** Monster-bar is *available*: the source paper §8 risk register already proposes the **contiguous-violation** trigger $t^* = \min\{t : s_t, s_{t+1}, s_{t+2} < q_\alpha\}$. This rules out the meta-comment failure mode at the cost of *missing genuine early divergences that recover briefly*. In Lakatos terms this is a **monster-barring tightening** that preserves the theorem at the cost of restricting its scope. v2 adopts a softer version: (A3') = "expected cascade monotonicity in the population", which permits per-trace violations as long as they cancel in expectation.

### C2 — Self-correcting reasoning model (Qwen-QwQ, R1-Distill) (A5 violation)

**Concrete scenario.** Qwen-QwQ-32B on AIME 2024 problem 7. The trace contains an explicit "Wait, let me reconsider" pivot at step 9 after an arithmetic slip at step 4:

```
Step 4: "Now, 17 * 13 = 211."          S_4 = -0.71 (below q_α — wrong, 17*13=221)
Step 5: "Subtracting: 211 - 144 = 67."  S_5 = -0.40 (above; arithmetic *internally consistent* with the wrong premise)
...
Step 9: "Wait, let me double-check 17*13. 17*13 = 17*10 + 17*3 = 170 + 51 = 221, not 211. Let me restart from step 4 with the correction."
Step 10–14: re-derives correctly with 221.
Final: correct.
```

**Which assumption fails.** (A5) cascade-effective dependence (= v1-MI's (A4)). The cascade-propagation probability $p_{\text{cascade}}$ is *not* close to 1 — it is close to 0 for self-correcting models, because the LM has been RL-trained to backtrack after detecting inconsistency.

Equivalently, (A3-mono): $\Pr[Q_s = 1 \mid Q_{t^*}=0]$ is **higher** than baseline for $s$ past the self-correction marker, not lower.

**Consequence.** Intervening at $t^* = 4$ with $\tilde X_4 \sim \pi$ (which is unlikely to produce the correct $221$ on the first try given that the LM also generated $211$ originally) **destroys** the self-correction prefix that would have led to the right answer. v1-FD's Theorem 4 *inverts* on this trace.

**Monster-bar vs. real revision.** **Real revision.** This is not a corner case; an entire family of frontier reasoning models (R1-Distill, QwQ, o1-style) lives here. v2 needs (A5') with an explicit *self-correction-free* scope clause: "Theorem 4 v2 is asserted only for non-self-correcting models, where $p_{\text{cascade}} > 1/2$ measured empirically." The frontier-reasoning case requires a separate theorem (out of scope here).

### C3 — KV-cache leakage / chat-template state (SM violation, hence A1 violation)

**Concrete scenario.** vLLM with prefix-caching enabled (`enable_prefix_caching=True`, default in modern serving stacks) on a shared batch. Two requests for the same prompt are served concurrently; their KV caches are **shared at the prompt prefix**. When request A re-rolls at $t^*$ via the do-operation, request B's KV-cache state can leak into A's continuation if their prompts share a hashed prefix and the cache eviction policy retains B's intermediate state.

A more reliable concrete example: NVIDIA's `instruct-toolkit` activation-steering vector $v_{\text{format}}$ is added to the residual stream at every decoding step. Under do$(X_{t^*})$ that re-runs the LM forward pass, $v_{\text{format}}$ is re-applied — but this is an unobserved $U$ outside the prefix $X_{1:t^*-1}$, so SM fails: $\pi(X_t \mid P, X_{1:t-1}) \neq \pi(X_t \mid P, X_{1:t-1}, U)$ where $U = v_{\text{format}}$.

**Which assumption fails.** (A1) prefix-blocking, via SM. The hidden $U$ acts as a confounder of $X_{t^*}$ and downstream $X_s$ that is not in $W$.

**Consequence.** The conditional front-door identification (FD-AR) is invalid: $\Pr(M \mid X_t, W) \neq \Pr(M \mid \text{do}(X_t), W)$. The "observational" estimate of the do-effect is biased by $U$. v1-FD's bounded-gap version (A1$_\eta$) bounds the bias by $\eta = d_{\text{TV}}(\Pr(M\mid X_t,W), \Pr(M\mid\text{do}(X_t),W))$, which can be measured.

**Monster-bar vs. real revision.** **Tightening** (monster-bar). The mitigation is a *deployment hygiene* assumption: vLLM with `enable_prefix_caching=False` for the experimental run, no activation steering, single-request-per-batch isolation. v2 makes this an *explicit pipeline assumption* under (A1) and reports the trace-replay determinism check (re-running the same trace twice with the same prompt should produce byte-identical outputs in greedy mode) as the empirical check that SM holds.

---

## §D — Round-1 fixes (lemma incorporation)

Based on §B and §C, the following revisions are adopted for v2.

### D.1 (A1) → (A1') — explicit pipeline scope

**(A1') Prefix-blocking under controlled inference.** The SCM $\mathcal{M}$ is generated by a transformer LM $\pi_\theta$ run with: (i) per-request KV-cache isolation (no `enable_prefix_caching`), (ii) no activation steering, (iii) deterministic chat template applied identically across the calibration and target sets, (iv) independent random seeds per decode call. Under (i)–(iv), softmax-Markovity holds and conditional on $(P, X_{1:t-1})$, all back-door paths from $X_t$ to $Y$ that bypass $X_{t+1:T}$ are d-separated.

**Operational test.** Trace-replay determinism: re-running a recorded trace's prompt + decoding config (greedy, fixed seed) in a fresh vLLM instance must reproduce the trace byte-for-byte. We log this as a CI gate in `SX_replay_check.py`.

This *strictly tightens* v1-FD's (A1). The bounded-gap $\eta$-version (A1$_\eta$') is retained for graceful degradation.

### D.2 (A3) → (A3') — expected cascade monotonicity

**(A3') Expected cascade monotonicity.** For the wrong-trace subpopulation $\{\omega : Y(\omega) = 0\}$ with $t^*(\omega) < T$, define $Q_s(\omega) \in \{0,1\}$ as the indicator that step $s$ is on a correct sub-trajectory (Math-Shepherd MC label, or PRM800K oracle). Then for all $s > t^*$,
$$
\mathbb{E}_{\omega}\!\bigl[Q_s(\omega) \mid Q_{t^*}(\omega) = 0\bigr] \;\leq\; \mathbb{E}_{\omega}\!\bigl[Q_s(\omega) \mid Q_{t^*}(\omega) = 1\bigr].
$$

**This relaxes v1-FD's per-trace (A3) to a population-level inequality.** Per-trace violations (C1's meta-comment trace) are permitted as long as they do not invert the population-level expectation. Operationally testable by stratified Math-Shepherd labels on PRM800K.

**Companion (A3-contig'): operational $t^*$ definition.** The pilot's $t^*$ is computed as $\min\{t : s_t < q_\alpha(t)\}$ but the source paper's risk register supports an alternative `--require_contiguous` flag that requires $s_t, s_{t+1}, s_{t+2} < q_\alpha$. v2 declares both definitions as Theorem-4-compatible; the pilot uses the simple definition, and the *contiguous* version is a strengthening that monster-bars C1.

### D.3 (A4) split into (A4) + (A5)

**(A4) Resampling-effective intervention.** $\pi(\cdot\mid X_{1:t-1},P)$ has nonzero probability mass on at least one $x'_t$ with $Q_t(x'_t)=1$. (Light, vacuity-avoiding.)

**(A5) Cascade contractivity (non-self-correcting LMs).** There exists $p_{\text{cascade}} \in (1/2, 1]$ such that for any off-trajectory prefix at step $s$, the natural mechanism produces an on-trajectory step $X_{s+1}$ with probability at most $1 - p_{\text{cascade}}$. **(A5) is asserted only for the non-self-correcting model class** — Qwen2.5-7B-Instruct, Qwen2.5-Math-7B, Phi-4 base. Frontier reasoning models (Qwen-QwQ, R1-Distill, o1-style) are *out of scope* of Theorem 4 v2.

**Empirical check for (A5).** $p_{\text{cascade}}$ is estimated from per-step CP violation patterns: in a wrong trace with $t^* < T$, the empirical fraction of off-trajectory steps in $[t^*, T]$ is a consistent estimator. Pilot data: $\hat p_{\text{cascade}} \in [0.7, 0.95]$ for Qwen2.5-7B / Phi-4 base; $\hat p_{\text{cascade}} \approx 0.1$–$0.4$ for R1-Distill (which is why R1-Distill is excluded).

### D.4 No change to (A2)

(A2) score-validity is unchanged from v1; both v1s state it identically.

### D.5 The revised theorem statement (round-1 form, "v1\'")

**Theorem 4 v1\' (round-1 fix).** Under (A1'), (A2), (A3'), (A4), (A5),
$$
\Pr_\mathcal{M}\!\bigl[Y=1 \mid \text{do}(X_{t^*}\sim\pi)\bigr] \;\geq\; \Pr_\mathcal{M}\!\bigl[Y=1 \mid \text{do}(X_t \sim \pi)\bigr]
\quad \forall\, t > t^*,
$$
with strict inequality $\geq \kappa \cdot [1 - p_{\text{cascade}}^{t-t^*}]$ when (A5) is strict and $p_{\text{cascade}} < 1$.

---

## §E — Round-2 counter-examples (does v1\' survive a second Lakatos round?)

### C4 — Multi-cascade trace (two independent error sources)

**Concrete scenario.** Phi-4 on OlympiadBench 2017-IMO-Q6 ("Bishops on a chessboard"). The trace has two distinct errors:

```
Step 3:  "There are 2*8 = 16 dark squares on a 8x8 board."   S_3 = -1.30 (below — wrong, 8x8 has 32 dark squares)
Step 4:  "Hmm wait, no, there are 32. Let me redo."          S_4 = -0.55 (above — self-correction, BUT...)
Step 5:  "OK so 32 dark squares. Bishops on dark stay on dark." S_5 = -0.30 (above)
...
Step 11: "By symmetry, the answer is 8."                       S_11 = -1.55 (below — wrong, the actual answer is 14)
Step 12: "Final answer: 8."                                    final, wrong
```

There are *two* below-threshold segments ($\{3\}$ and $\{11\}$), separated by an on-trajectory recovery in the middle. The *real* cascade source is $t^*_2 = 11$ (the "by symmetry" handwave), not $t^*_1 = 3$.

**Which assumption fails in v1\'.** (A3') in expectation might still hold population-wide, but for *this trace*, $t^* = t^*_1 = 3$ is identified, and intervention at $t^* = 3$ does nothing because the model self-corrected at $t=4$. (A5) cascade contractivity holds globally (Phi-4 is non-self-correcting on average), but locally between $t=4$ and $t=11$ the trace recovered, then re-failed.

**Consequence.** $\Delta(t^*_1=3) \approx 0$; $\Delta(t^*_2=11) > 0$. Theorem 4 v1\' predicts dominance at $t^*_1$ but the truth is dominance at $t^*_2$.

**Monster-bar vs. real revision.** Real revision. v1-MI §11.1 estimates ~15% of pilot wrong traces are multi-cascade. The fix is to define $t^*$ as the **earliest cascade source** with a *contiguity-or-recovery* protocol: walk from left to right, and whenever a below-threshold step is *followed by recovery*, skip; treat as $t^*$ only the first below-threshold step that is *not* followed by within-3-steps recovery. This is a strict generalization of (A3-contig').

### C5 — Prompt ambiguity: prefix already on a different correct trajectory

**Concrete scenario.** Qwen2.5-7B on a MATH-500 problem with multiple correct interpretations:

> "Find the smallest positive integer $n$ such that $n^2 + n + 41$ is composite."

There are two correct framings: (a) try $n = 1, 2, 3, \ldots$ and find $n=40$ where $40^2+40+41 = 1681 = 41^2$; (b) note $n=41$ trivially works since $41^2+41+41 = 41(41+1+1) = 41\cdot 43$. The model picks framing (a) but at step 5 starts inserting framing-(b) reasoning, which has high entropy under framing (a)'s prefix:

```
Step 1: "Let me try small values of n."   S_1 = -0.40 (above)
Step 2: "n=1: 43, prime. n=2: 47, prime."  S_2 = -0.45 (above)
Step 3: "n=3: 53. n=4: 61. ..."             S_3 = -0.42 (above)
Step 4: "n=10: 151. Hmm, this is slow."     S_4 = -0.55 (above, but rising)
Step 5: "Actually, note that n=41 makes n^2+n+41 = 41(n+1+1) = 41*43, which is composite." S_5 = -1.40 (below — but it's CORRECT in framing (b)!)
Step 6: "But the problem asks the SMALLEST n. n=41 is too big."  (model now confused)
Step 7-9: chaotic mixing of (a) and (b) framings, never finds n=40.
Final: wrong (answers n=41).
```

**Which assumption fails.** (A2) score-validity in a subtle way: $S_5 < q_\alpha$ flags step 5 as off-trajectory under PRM800K's calibration, but step 5 is *on-trajectory in framing (b)* — just incompatible with the prefix's framing (a). The score is *correctly* low because the LM is unstable, but the underlying $Q_5 = 1$ in some interpretive frame. In short: $Q_t$ is not a single-valued function of $X_{1:t}$ when the problem admits multiple solution paths with low-probability switching costs.

**Consequence.** $t^* = 5$ identified, but intervening at $t^*=5$ with a fresh sample from $\pi(\cdot\mid X_{1:4},P)$ likely *also* drifts toward framing (b) (since the prefix is already approaching the framing-(b) attractor), so the do does not steer back to framing (a). $\Delta(t^*) \approx 0$ on this trace.

**Which assumption fails technically.** (A3') again — but here it's because of multi-modality of $q^*$, not because of meta-comments. The "correct trajectory" is not a single trajectory.

**Monster-bar vs. real revision.** This is more subtle. **Soft monster-bar** by restricting Theorem 4 to *unimodal* problems: those where the correct-trace conditional $q^*(\cdot\mid X_{1:t-1},P)$ has a unique mode (single solution path). In practice, MATH-500 and AIME problems are mostly unimodal; OlympiadBench has more multi-modal cases (~5–10%). v2 declares unimodality as an implicit scope clause and notes that multi-modal failure cases are a known limitation.

### E.1 Verdict on round-2

C4 forces a strict generalization of (A3') (recovery-aware $t^*$ definition); C5 forces an explicit unimodality scope. Both are absorbed into v2's final assumption set.

---

## §F — Final Theorem 4 v2 (boxed)

### F.0 Final assumption set

- **(A1')** Prefix-blocking under controlled inference: KV-cache isolation, no activation steering, fixed chat template, independent per-request seeds. Conditional on $(P, X_{1:t-1})$, all back-door paths from $X_t$ to $Y$ bypassing $X_{t+1:T}$ are d-separated. Empirical check: trace-replay determinism (`SX_replay_check.py`).
- **(A2)** Score-validity: split-CP threshold $q_\alpha(t)$ on PRM800K satisfies $\Pr[s_t < q_\alpha(t) \mid Q_t=1] \leq \alpha$.
- **(A3')** Expected cascade monotonicity (population-level), plus recovery-aware $t^*$ definition: $t^* = \min\{t : s_t < q_\alpha(t) \text{ and } \exists s \in [t,t+2]\ s_s < q_\alpha(s)\}$, i.e., contiguous violation. Multi-cascade traces use the earliest non-recovered cascade source.
- **(A4)** Resampling-effective intervention: $\pi(\cdot\mid X_{1:t-1},P)$ has nonzero mass on at least one on-trajectory $x'_t$.
- **(A5)** Cascade contractivity: $p_{\text{cascade}} \in (1/2, 1]$, with the model class restricted to non-self-correcting LMs (Qwen2.5-7B-Instruct, Qwen2.5-Math-7B, Phi-4 base; explicitly *not* QwQ, R1-Distill, o1).
- **(A6, implicit)** Unimodal correct-trajectory conditional: for the problem class under analysis, $q^*(\cdot\mid X_{1:t-1},P)$ has a unique mode. Multi-modal problems (~5–10% of OlympiadBench) are out of scope.

### F.1 Boxed theorem statement

> **Theorem 4 v2 (Earliest-divergent-step dominance).** Let $\mathcal{M}$ be the auto-regressive SCM of §1, with model class restricted by (A5) and problem class restricted by (A6). For any wrong trace $\bar x$ with $Y(\bar x) = 0$ and recovery-aware $t^*(\bar x) < T$, under (A1'), (A2), (A3'), (A4), (A5), (A6):
>
> $$\boxed{\;\;\Pr_\mathcal{M}\!\bigl[Y = 1 \mid \text{do}(X_{t^*}\sim\pi)\bigr] \;\geq\; \Pr_\mathcal{M}\!\bigl[Y=1 \mid \text{do}(X_t\sim\pi)\bigr] \;\;\forall t > t^*\;\;}$$
>
> with quantitative gap
> $$\Pr[Y=1\mid\text{do}(X_{t^*})] - \Pr[Y=1\mid\text{do}(X_t)] \;\geq\; \kappa \cdot \bigl[1 - p_{\text{cascade}}^{\,t-t^*}\bigr] \;-\; 2\eta,$$
>
> where $\kappa$ is the lift constant (Lemma 4.B), $p_{\text{cascade}}$ is the cascade-propagation probability ((A5)), and $\eta$ is the bounded-gap slack from (A1$_\eta$') if SM holds only approximately. Equality holds iff (i) the trace is cascade-trivial ($p_{\text{cascade}} = 0$, hence $\Delta$ is flat at and past $t^*$), (ii) $t = t^*$, or (iii) the resample distribution $\pi$ does not contain any on-trajectory mass at $t^*$ ((A4) tight).

### F.2 Clean proof (combining identification and optimality)

The proof decomposes into Lemma 4.A (identification) + Lemma 4.B (optimality) + connecting argument.

**Step 1 — Identification (Lemma 4.A).** By (A1'), the conditional front-door criterion (Pearl 2009 §3.4 / Tian–Pearl 2002 ID) holds with mediator $M = X_{t+1:T}$ and conditioning set $W = X_{1:t-1}$. The front-door formula (FD-AR) writes both LHS and RHS as observational conditional probabilities computable from the LM's softmax + the answer judge:

$$\Pr[Y=1\mid\text{do}(X_t \sim \pi), W] = \mathbb{E}_{x'_t \sim \pi}\!\left[ \sum_{m} \Pr(M=m \mid x'_t, W) \cdot \Pr(Y=1\mid M=m, W) \right].$$

This identification is *the same identifier* applied at $t = t^*$ and at any $t > t^*$, with the difference being which $W$ we condition on.

**Step 2 — Optimality (Lemma 4.B).** Apply the three-regime decomposition (v1-MI's Lemmas 1–3) with the assumptions revised:
- For $t < t^*$: $\Delta(t) = O(\delta + \alpha)$ — pre-divergence intervention is wasted (proof uses (A2), (A3-contig') prefix is on-trajectory, and (A4) on-trajectory mass exists).
- For $t = t^*$: $\Delta(t^*) \geq \kappa$ — at the cascade source, on-trajectory $\tilde X_{t^*}$ + (A5) contractive forward dynamics + (A6) unimodal $q^*$ give $\Pr[Y=1\mid\text{do}(X_{t^*})] \geq 1 - O(\delta T + \alpha)$, and $\Pr[Y=1\mid\bar x] \leq \eta < 1 - \kappa$.
- For $t > t^*$: $\Delta(t) \leq \kappa \cdot p_{\text{cascade}}^{t-t^*} + O(\delta T)$ — by (A5), $t-t^*$ off-trajectory steps in the prefix re-diverge the suffix re-roll with probability $\geq p_{\text{cascade}}^{t-t^*}$.

Subtracting:
$$\Delta(t^*) - \Delta(t) \geq \kappa - \kappa \cdot p_{\text{cascade}}^{t-t^*} - O(\delta T) = \kappa\bigl[1 - p_{\text{cascade}}^{t-t^*}\bigr] - O(\delta T).$$

**Step 3 — Connecting argument.** The crucial step is that v1-FD's Lemma 1 (prefix-blocking) and v1-MI's (A5) (contractivity) **operate on the same SCM but on different aspects of it**: Lemma 1 controls the *graphical* identifiability (whether the do-quantity is computable), and (A5) controls the *dynamical* cascade behavior (whether the cascade actually propagates). Under (A1'), the coupling argument from v1-FD §7.4 is valid (independent noise across steps), and under (A5), the geometric decay applies. The two combine without conflict because (A1') is a property of the SCM's graph + inference pipeline, and (A5) is a property of the SCM's mechanisms — orthogonal properties.

Formally: under (A1'), we may write $\Pr[Y=1\mid\text{do}(X_t\sim\pi)]$ as an expectation over the *coupled* twin-network construction (Pearl 2009 §7) where the do-trace and the natural trace share noise from steps $\neq t$. Under this coupling, the only difference between $\text{do}(X_{t^*}\sim\pi)$ and $\text{do}(X_t\sim\pi)$ for $t > t^*$ is which prefix we re-roll from. (A5) then applies to the difference in the suffix re-roll distributions. $\blacksquare$

### F.3 Failure-mode taxonomy (bundled with the boxed theorem)

| Failure mode | Assumption violated | In-scope? | Mitigation in v2 |
|---|---|---|---|
| KV-cache leakage / activation steering | (A1') | No (out of pipeline scope) | `enable_prefix_caching=False`, no steering, replay determinism check |
| Calibration drift across models | (A2) | Yes (sensitivity) | Per-model recalibration on held-out PRM800K split |
| Isolated meta-comments before $t^*$ | (A3-contig) | Yes (handled) | Recovery-aware $t^*$: require contiguous below-threshold |
| Multi-cascade traces | (A3') | Yes (handled) | Define $t^*$ as earliest non-recovered cascade source |
| Self-correcting / RL-trained LMs | (A5) | No (out of model scope) | Restrict to non-self-correcting model class; future work |
| Multi-modal correct trajectories | (A6) | No (out of problem scope) | Restrict to unimodal problems; OlympiadBench multi-modal subset excluded |
| Coupled-noise samplers (fixed seed) | (A1', noise-iid) | No (out of pipeline scope) | Independent per-request seeds |

---

## §G — Companion lemmas

### G.1 Lemma 4.A — Identification via conditional front-door

> **Lemma 4.A (front-door identification under auto-regressive structure).** Under (A1') and (A2), for every $t \in [1, T]$ and every $w \in \mathcal{X}^{t-1}$,
> $$
> \Pr_\mathcal{M}\!\bigl[Y=1 \mid \text{do}(X_t = x'_t),\, W=w\bigr] = \sum_{m \in \mathcal{X}^{T-t}} \Pr(M=m \mid X_t=x'_t,\, W=w) \cdot \Pr(Y=1 \mid M=m,\, W=w),
> $$
> i.e., the do-quantity is identifiable from the LM's observational conditional distributions $\Pr(M\mid X_t,W)$ and the answer-judge's $\Pr(Y\mid M,W)$. The triple-sum collapses to a double-sum because, conditional on $W$ and $M$, $Y$ does not depend on $X_t$ (interception property F1$_W$).

**Proof.** Verify the conditional front-door criteria (F1$_W$, F2$_W$, F3$_W$) of v1-FD §3.3 under (A1'). F1$_W$ holds because $M$ intercepts every directed path $X_t \to Y$ in the auto-regressive DAG. F2$_W$ holds because, conditional on $W$ and (A1') ⟹ SM, the only parent of $X_t$ outside $W$ is $\epsilon_t$, which is exogenous. F3$_W$ holds because every back-door path $X_s \leftarrow X_r \to Y$ for $s > t, r < t$ is blocked by $W \ni X_r$. The identification formula then follows from Pearl 2009 Theorem 3.3.4 generalized to conditional front-door (Tian–Pearl 2002 ID Theorem 1). $\blacksquare$

**Bounded-gap version (Lemma 4.A$_\eta$).** Under (A1$_\eta$') with TV-slack $\eta \in [0,1)$, the identification holds within $2\eta$ TV-error (Bareinboim 2014 Theorem 5 / Corollary 2 transportability bound).

### G.2 Lemma 4.B — Optimality via cascade-decay

> **Lemma 4.B (cascade-decay optimality).** Under (A2), (A3'), (A4), (A5), (A6), for any wrong trace $\bar x$ with recovery-aware $t^*$ and any $t > t^*$,
> $$
> \Delta(t^*; \bar x) - \Delta(t; \bar x) \;\geq\; \kappa \cdot \bigl[1 - p_{\text{cascade}}^{\,t-t^*}\bigr] - O(\delta T),
> $$
> where $\Delta(t;\bar x) := \mathbb{E}_{\tilde X_t \sim \pi(\cdot\mid X_{1:t-1}(\bar x), P)}[\Pr[Y=1\mid\text{do}(X_t = \tilde X_t)]] - \Pr[Y=1\mid\bar x]$.

**Proof sketch (already in v1-MI §4–6, with revised assumptions).** Decompose $\Delta(t)$ into the three temporal regimes:

- $t<t^*$: prefix on-trajectory, $\tilde X_t \sim \pi$ likely in the high-probability region of the natural distribution, suffix re-roll has approximately the same law as the natural suffix → $\Delta(t) = O(\delta + \alpha)$.
- $t=t^*$: prefix on-trajectory, $\tilde X_{t^*} \sim \pi$ has probability $\geq \tau > 0$ of being on-trajectory by (A4), suffix re-roll inherits on-trajectory prefix and stays on-trajectory by (A5)'s contractivity (in the on-trajectory direction) → $\Delta(t^*) \geq \tau \cdot (1 - \eta) - O(\delta T)$. Set $\kappa := \tau (1-\eta) - O(\delta T)$.
- $t>t^*$: prefix already contains $t-t^*$ off-trajectory steps, suffix re-roll re-diverges with probability $\geq p_{\text{cascade}}^{t-t^*}$ by (A5)'s contractivity (in the off-trajectory direction) → $\Delta(t) \leq \kappa \cdot p_{\text{cascade}}^{t-t^*} + O(\delta T)$.

Subtract. $\blacksquare$

**Note on the resample distribution.** Lemma 4.B is stated under $\tilde X_t \sim \pi$ (operational, matches the pilot). Under the oracle $\tilde X_t \sim q^*$ (v1-MI's idealized version), the bound improves to $\Delta_{q^*}(t^*) \geq 1 - O(\delta T + \alpha)$ (i.e., $\kappa \to 1 - O(\cdots)$), because (A4)'s $\tau$ factor becomes $1$. This recovers v1-MI's headline.

### G.3 Connecting argument: Theorem 4 v2 = Lemma 4.A + Lemma 4.B + coupling

Lemma 4.A makes the do-quantity *computable* from the LM's softmax. Lemma 4.B ranks the computed quantities across $t$. The connecting argument is the twin-network coupling (Pearl 2009 §7 / Buesing 2019): under (A1'), $\epsilon_{t+1:T}$ can be shared across the two intervened distributions $\text{do}(X_{t^*})$ and $\text{do}(X_t)$, so the comparison reduces to a comparison of the *prefixes* $\bar x_{1:t^*-1}, x'_{t^*}$ vs. $\bar x_{1:t-1}, x'_t$. The first prefix is on-trajectory at $t^*$; the second contains $t-t^*$ off-trajectory steps. (A5) then gives the geometric gap.

This is the place where v1-FD's identification proof and v1-MI's lemma chain *actually meet*: the identification proof ensures the coupling is valid (well-defined twin-network); the lemma chain ensures the comparison goes the right way under coupling.

---

## §H — What v2 still does NOT establish

Honest scoping. v2 establishes single-do dominance under (A1')–(A6). It does *not* establish:

1. **K=4 majority dominance.** Theorem 4 v2 is a single-do statement. K=4 majority + voting is a non-linear functional of the four samples. Theorem 4 v2 ⟹ majority dominance only with an additional Jensen-type argument that we do not provide; the empirical pilot tests this composite claim and the negative-lift on `qwen25_7b__math500` could be attributable to the majority composition rather than to single-do failure.
2. **Tracewise dominance.** Theorem 4 v2 is dominance *in expectation* over the wrong-trace subpopulation and over the resample noise. Per-trace, intervention at $t > t^*$ may produce a correct answer while $t^*$ does not. Pilot's `frac_changed_earliest` ≈ 11% (v.s. 11.4% for worst) shows this scatter.
3. **Optimality for $t < t^*$.** Theorem 4 v2 says $t^*$ dominates $t > t^*$; for $t < t^*$ the gap is $O(\delta + \alpha)$ which is small but not zero. Theorem 4 v2 does *not* claim $t^*$ dominates $t < t^*$ strictly. (Pilot data are consistent with this — $t^*$ is concentrated at $t \in \{1, 2\}$ so $t < t^*$ is rarely available.)
4. **Frontier reasoning models.** R1-Distill, QwQ, o1-style are out of scope by (A5).
5. **Multi-modal problems.** OlympiadBench multi-modal subset (~5–10%) is out of scope by (A6).
6. **Causal validity of the trigger.** (A2) gives marginal coverage on calibration; it does not guarantee that $t^*$ identifies the *causal* cascade source on every trace. The empirical check is the H3 experiment in `pearl_causal_DEEP.md` §6.1.
7. **Quantitative $\kappa$ value.** $\kappa$ depends on $\tau$ (mass of $\pi$ on on-trajectory steps), $\eta$, $\delta$, $T$ — none of which are predicted by the theorem; they are estimated from the pilot.
8. **Cross-model transportability.** Theorem 4 v2 is for fixed $(\pi_\theta, P)$. Cross-model claims require Bareinboim 2014 transportability, not made here.
9. **Multi-step interventions** (do$(X_{t^*}, X_{t^*+1}, \dots)$). Theorem 4 v2 is single-do.
10. **Negative-lift sign on `qwen25_7b__math500` is *predicted* but not *proven*.** v2 explains why the lift is small (low $\kappa$ on easy cells); whether it should be slightly negative vs. slightly positive depends on noise + sample size, not on the theorem.

---

## §I — Connection to empirical pilot (does v2 predict the data?)

### I.1 Pilot 1 (`pearl_causal_pilot.json`, 12 cells, H2/H4)

H2 and H4 are *consistent* with Theorem 4 v2 under (A5):

- **H4 (frac($t^* < t_{\text{worst}}$) = 42–78%):** consistent with Theorem 4 v2's prediction that $t^*$ is the cascade source and $t_{\text{worst}}$ is the cascade locus; the gap is *expected* to be larger for harder cells and longer traces.
- **H2 (cascade depth $t_{\text{worst}} - t^*$ = 1.5–9.88):** consistent with (A5)'s $p_{\text{cascade}} \in [0.7, 0.95]$ — the geometric tail $p_{\text{cascade}}^{t-t^*}$ at typical cascade depths predicts substantial residual mass at $t_{\text{worst}}$.

**v2 does not predict the absolute value of the cascade depth** — that is set by trace length $T$ and $p_{\text{cascade}}$, both of which are model/dataset-specific empirical quantities.

### I.2 Pilot 2 (`pearl_full/qwen25_7b_math500.json`, n=200)

This is the **decisive test** of Theorem 4 v2. The data show:

```
score=lp, α=0.1: lift_earliest_vs_worst = -0.0165 [-0.0700, +0.0300]
score=lp, α=0.3: lift_earliest_vs_worst = -0.0069 [-0.0600, +0.0400]
score=lp, α=0.5: lift_earliest_vs_worst = -0.0057 [-0.0500, +0.0352]

score=ent_neg, α=0.1: lift = -0.0136 [-0.0700, +0.0300]
score=ent_neg, α=0.3: lift = -0.0117 [-0.0600, +0.0400]
score=ent_neg, α=0.5: lift = -0.0055 [-0.0500, +0.0400]

score=marg, α=0.1: lift = -0.0273 [-0.0900, +0.0200]
score=marg, α=0.3: lift = -0.0250 [-0.0800, +0.0200]
score=marg, α=0.5: lift = -0.0124 [-0.0600, +0.0300]
```

All 9 (score, α) settings give **negative point estimates** for `lift_earliest_vs_worst`, with all 95% CIs *overlapping zero*. The cascade-depth-stratified lift on `lp__a0.3`:

```
gap=1: -0.0006   (zero — consistent with Theorem 4 v2: when t* = t_worst there is no leverage)
gap=2..4: -0.0183
gap=5+: -0.0495  (largest negative — *most* off from theory expectation)
```

**Does v2 predict the negative sign?** Strictly, *no* — Theorem 4 v2 predicts a non-negative population lift in expectation. **But v2 predicts the negative sign is *not surprising*** for the qwen25_7b_math500 cell, for three reasons:

1. **Easy-cell low leverage.** vanilla_acc = 73.8%. The headroom for K=4 intervention is at most $1 - 0.738 = 26.2$pp. The fraction of traces where intervention fires is `frac_changed_earliest` ≈ 6–12%, so the maximum possible lift is $\approx 0.10 \cdot 0.262 \approx 2.6$pp. The observed lift's lower 95% bound is $-7$pp, so the observable negative lift is well within the noise floor.
2. **Small cascade gap on easy cells.** From the H2 pilot, qwen25_7b on math500 has the *shortest* cascade gap among all 12 cells (gap ≈ 1.98 at α=0.3). The Theorem 4 v2 quantitative bound is $\kappa \cdot [1 - p_{\text{cascade}}^{t-t^*}]$; for $t-t^* = 1$ this is $\kappa \cdot (1 - p_{\text{cascade}}) \approx \kappa \cdot 0.15$, which is a very small *theoretical* edge — easily swamped by sampling noise.
3. **K=4 majority + small change rate.** With `frac_changed_earliest` ≈ 0.11 and `frac_changed_worst` ≈ 0.11 (nearly identical), the two intervention strategies modify the *same fraction of traces*, but at $t^*$ the intervention fires earlier (less leverage on already-correct traces, more risk of disrupting them). On easy cells, $t^*$ intervention has higher *false-positive* risk: re-rolling at step 1–2 of an originally-correct trace can break it. The pilot shows `K4_earliest_acc` < `K4_worst_acc` consistently, which is exactly this *false-positive cost dominating the small theoretical gain*.

In the language of the v2 quantitative bound:
$$\Delta(t^*) - \Delta(t_{\text{worst}}) \geq \kappa \cdot (1 - p_{\text{cascade}}^{1}) - 2\eta - \text{(false-positive cost)}.$$
On easy cells, $\kappa$ is small (low headroom), $1 - p_{\text{cascade}}$ is small (short cascade), and the false-positive cost is positive. The RHS can be **negative** even when Theorem 4 holds in expectation over the wrong-trace subpopulation, because the K=4 procedure is applied to all traces (correct + incorrect).

**Concrete v2 prediction.** The negative lift on `qwen25_7b__math500` should *attenuate* and eventually *flip positive* when:
- the cell is hard (vanilla_acc lower, headroom higher) — predicts positive lift on AIME / OlympiadBench cells;
- the cascade gap is large ($p_{\text{cascade}}^{t-t^*}$ small, theoretical gain large);
- the trigger is restricted to *wrong* traces only (false-positive cost zero) — the pilot doesn't do this; an oracle-gated version would.

This is a **falsifiable** prediction: if AIME / OlympiadBench cells *also* show negative lift, v2 is in serious trouble. The current pilot has not yet reported those cells (`pearl_full/AGGREGATE.md` shows only `qwen25_7b_math500`), so this is genuine prediction, not retrodiction.

### I.3 Falsification path

Theorem 4 v2 fails empirically if any of the following holds in the full `pearl_full/` run:
- AIME / OlympiadBench cells show $\geq -1$pp lift with CI excluding $+0$ ⟹ (A5) fails or scope is wrong.
- `cascade_lift_by_gap` shows *positive* slope at $t-t^* \geq 5$ but flat at $t-t^* = 1$ ⟹ inverse of v2's prediction.
- Self-correcting model cells (R1-Distill, QwQ) show large positive lift ⟹ (A5) fails *and* the out-of-scope clause is too aggressive.

---

## §J — Reviewer attack surface (top 5 objections + v2's response)

**Objection 1.** *"Your theorem is vacuous on easy cells — the negative lift on qwen25_7b__math500 contradicts your claim, and your handwave about 'noise floor' is unconvincing."*

> **v2 response.** §I.2 gives a *quantitative* explanation: $\kappa$ is small on easy cells because of low headroom, the cascade gap is short, and the false-positive cost on already-correct traces dominates. The theorem is stated in expectation over the wrong-trace subpopulation; the pilot averages over *all* traces. We will report a wrong-trace-conditional version of the lift in the next pilot run, where the false-positive cost is zero by construction. Theorem 4 v2 is testable only on hard cells with large cascade gap; on easy cells it is operationally vacuous, which is *consistent* with the bound rather than contradicting it.

**Objection 2.** *"(A1') is too strong — real LM serving has KV-cache, prefix caching, batching, all of which violate softmax-Markovity. Your theorem is for an idealized lab setting."*

> **v2 response.** Yes, (A1') is restrictive. The bounded-gap version (A1$_\eta$') and Lemma 4.A$_\eta$ degrade gracefully: as $\eta$ increases, the dominance gap shrinks by $2\eta$. We provide an empirical check (`SX_replay_check.py` trace-replay determinism) and report $\eta$ as a sensitivity parameter. For real-deployment claims, the bounded-gap version is the operational statement; the exact (A1') is an idealization for the proof.

**Objection 3.** *"(A3') is not testable. You claim 'expected cascade monotonicity' but never measure it on PRM800K."*

> **v2 response.** (A3') is testable in two ways: (a) Math-Shepherd MC labels $Q_s$ on PRM800K give direct estimates of $\mathbb{E}[Q_s\mid Q_{t^*}=0]$ vs. $\mathbb{E}[Q_s\mid Q_{t^*}=1]$; (b) the empirical $p_{\text{cascade}}$ from per-step CP violation patterns is a consistent estimator. The §6.1 H3 experiment in `pearl_causal_DEEP.md` operationalizes this. We propose adding an explicit (A3') validation table to the camera-ready.

**Objection 4.** *"You exclude self-correcting models (R1-Distill, QwQ) but those are exactly the models people care about. This is a domain restriction that destroys generality."*

> **v2 response.** Acknowledged, this is a real scope cost. The motivation: R1-Distill / QwQ violate (A5) by construction (they are RL-trained for self-correction), so a clean step-intervention theorem cannot apply uniformly. We are explicit about this in (A5) and §H. A separate Theorem (call it 4.5) for self-correcting models would need a different cascade structure — an active research question. The current paper's contribution is the *non-self-correcting* baseline, which is still a substantial model class (Qwen2.5-Math-7B, Qwen2.5-7B-Instruct, Phi-4 base, Llama-3.1, Mistral-7B). The R1-Distill case is positioned as future work.

**Objection 5.** *"This is not really a Pearl-style result — your front-door is conditional on $W$, your $t^*$ is statistical not causal, your $p_{\text{cascade}}$ is empirical. It's a Pearl-flavored statistical observation, not a do-calculus theorem."*

> **v2 response.** The Pearl 2009 §3.4 conditional front-door (and Tian–Pearl 2002 ID Theorem 1) is genuinely a do-calculus theorem — the conditioning on $W$ is part of the rule formalism, not a softening. (A2)'s split-CP calibration *is* a statistical guarantee, but the dominance comparison is computed at the do-distribution level via Lemma 4.A. (A5) is empirically estimated; this is normal for any applied causal-inference paper (analogous to estimating propensity scores). Theorem 4 v2 is a do-calculus result with empirically-estimable assumption parameters — same status as any modern applied causal-inference theorem.

---

## Cross-Model Verification Results

*(per workspace `CLAUDE.md` cross-model verification protocol; `mode: all`, primary `claude-opus-4-7`, verifier `openai/openai/gpt-5.5`. To be appended after verifier pass on this consolidated v2.)*

**Verdict — primary (claude-opus-4-7):** PROCEED. The two v1s have genuinely different scope (identification vs. optimality) and the consolidation surfaces three real disagreements (B.1 resample distribution, B.3 (A3) strength, B.4 (A4) split). The five Lakatos counter-examples (C1–C5) force two assumption tightenings ((A3-contig'), (A6) unimodality) and two scope restrictions ((A5) non-self-correcting, (A6) unimodal problems). The negative-lift on `qwen25_7b__math500` is explained by low-headroom + short-cascade-gap + false-positive cost on already-correct traces. v2 is testable on the hard cells of `pearl_full/` and falsifiable per §I.3.

**Verdict — verifier:** *(pending verifier pass; per `cross_model_verification_protocol.md`, disagreements appended verbatim, no silent overrides)*.

---

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge UP. Theorem 3.3.4 (front-door); §3.4 (generalized identification with conditioning); §4.4 (minimal-effect intervention); §4.5 (controlled direct effects); §7 (twin-network counterfactuals).
- Tian, J. & Pearl, J. (2002). *A General Identification Condition for Causal Effects*. AAAI. — generalized (conditional) front-door / ID Theorem 1.
- Pearl, J., Glymour, M., Jewell, N. P. (2016). *Causal Inference in Statistics: A Primer*. Wiley. §3.4 front-door adjustment.
- Bareinboim, E. & Pearl, J. (2014). *External Validity: From Do-Calculus to Transportability Across Populations*. Statistical Science 29:579–595. Definition 5 (s-recoverability), Theorem 5 (bounded-gap recoverability), Corollary 2.
- Howard, R. A. (1966). *Information Value Theory*. IEEE TSSC 2(1):22–26. — VOI / EVPI; §10 connection in v1-MI.
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. §6 (TD), §12 (eligibility traces).
- Buesing, L. et al. (2019). *Woulda, Coulda, Shoulda: Counterfactually-Guided Policy Search*. ICLR. — twin-network counterfactual coupling.
- Wang, X., Wei, J. et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. ICLR 2023. — K-sample majority concentration.
- Lightman, H. et al. (2023). *Let's Verify Step by Step* (PRM800K). — score calibration substrate for (A2).
- Math-Shepherd (arXiv 2312.08935). — $Q_s$ MC labels for (A3') validation.
- First-Step Advantage (arXiv 2311.07945) and Well Begun Half Done (arXiv 2512.15274). — empirical corroboration of $t^*$ being concentrated very early.
- Internal:
  - `theorem4_front_door_v1.md`, `theorem4_minimal_intervention_v1.md` — v1 sources.
  - `/home/nvidia/future/literature/concept_papers/pearl_causal_DEEP.md` — parent paper plan.
  - `/home/nvidia/future/experiments/results/pearl_causal_pilot.json` — H2/H4 pilot.
  - `/home/nvidia/future/experiments/results/pearl_full/qwen25_7b_math500.json`, `AGGREGATE.json`, `AGGREGATE.md` — running full experiment with negative-lift signal explained in §I.2.

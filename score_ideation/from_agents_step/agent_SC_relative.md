# Agent SC: Step-Level Relative / Comparative Scoring Methods for CoT-CP

**Author:** Agent SC
**Scope:** Step-level scoring functions that compare step `t` to its prefix
(`steps 0..t-1`) or to its immediate neighbors. All features are computable
*online* at step boundaries with O(t) memory or less.

## Design principles

Trajectory baselines (`lp_min`, `prm_min`, `entropy_max`) suffer two issues:
(a) a step mediocre in absolute terms may be normal for the trace (long
arithmetic vs. short verbal), and (b) a step fine in absolute terms may be
a *local* anomaly (sudden dip after 8 confident steps). Relative features
target this intra-trace abnormality and are more length-invariant.

Throughout, `lp_t = mean token logprob within step t`,
`H_t = mean per-token entropy in step t`, `m_t = mean top-1 margin`.

---

## 1. `step_lp_zscore` — running standardized logprob

### Definition
At step `t > 1`, with prefix samples `{lp_0, ..., lp_{t-1}}`:
```
mu_{t-1} = mean(lp_0..lp_{t-1})
sd_{t-1} = std(lp_0..lp_{t-1}, ddof=1)  # with floor eps
z_t      = (lp_t - mu_{t-1}) / max(sd_{t-1}, eps)
score_t  = -z_t          # high score = unusually low lp
```
For `t in {0,1}`, emit `score_t = 0` (insufficient prefix).

Use Welford's online update so `mu, sd` cost O(1) memory.

### Cost
FREE (one Welford update + one division per step).

### Hypothesis
A step many standard deviations below this trace's running average is
unusually surprising **for this question**. Length and topic effects cancel
because each trace defines its own reference. "Stuck" / strategy-switch
steps tend to be locally low-lp.

### Pseudocode
```python
class WelfordLP:
    def __init__(self): self.n=0; self.mu=0.0; self.M2=0.0
    def update(self,x):
        self.n += 1
        d = x - self.mu
        self.mu += d/self.n
        self.M2 += d*(x - self.mu)
    def std(self):
        return (self.M2/max(self.n-1,1))**0.5

w = WelfordLP()
zs = []
for t, lp_t in enumerate(per_step_lp):
    if t < 2:
        zs.append(0.0)
    else:
        sd = max(w.std(), 1e-3)
        zs.append(-(lp_t - w.mu)/sd)
    w.update(lp_t)
nonconformity = max(zs)         # trajectory-level wrap, OR
abort_t       = (zs[t] > tau)   # selective abort
```

### CP usage
Two modes:
* **Trajectory-mode CP:** treat `max_t z_t` as the per-question
  nonconformity score, run split CP exactly as for `lp_min`.
* **Step-level CP:** calibrate a per-step threshold `tau` such that
  `P(z_t > tau | y correct)` is bounded; use `z_t > tau` as an abort
  signal, returning ABSTAIN early. Calibration is over (step, correct?)
  pairs from the calibration set.

### Failure modes
* Early steps have noisy `mu, sd`; first few z's are unstable. Warm-up
  window of 3 helps.
* Flat lp profiles → tiny `sd` → blowups. Mitigation: `eps` floor and
  clip `|z| ≤ 6`.
* Low-lp boilerplate ("Let me think...") registers as anomaly even when
  harmless.

---

## 2. `step_lp_drawdown` — peak-to-current drawdown over the prefix

### Definition
Treat the per-step lp series as a "stock chart". At step `t`:
```
peak_{t-1} = max(lp_0..lp_{t-1})
drawdown_t = peak_{t-1} - lp_t       # nonnegative if step is below the peak
score_t    = max(0, drawdown_t)
```
Track `peak` online in O(1).

### Cost
FREE (one `max` update per step).

### Hypothesis
Reasoning is often non-monotone: model has a "good groove" early then breaks
down. Largest *drawdown* directly measures "how bad has it gotten vs. how
good we were". Unlike the z-score, drawdown ignores variance and is robust
when the trace has long stretches of similar lp.

### Pseudocode
```python
peak = -float("inf")
dds  = []
for lp_t in per_step_lp:
    if peak == -float("inf"):
        dds.append(0.0)
    else:
        dds.append(max(0.0, peak - lp_t))
    peak = max(peak, lp_t)
score = max(dds)            # trajectory wrap
```

### CP usage
* Trajectory-mode: `nonconformity = max_t drawdown_t` → split CP.
* Step-mode: trigger abstain when `drawdown_t > tau_calibrated`.
* Combined: also report the *step index of max drawdown* — useful for
  debugging where the trace went wrong.

### Failure modes
* Uniformly-low trace has zero drawdown but is uniformly bad. Combine
  with an absolute floor (`lp_min`) for the CP score.
* Sensitive to "lucky" early-step outliers inflating the peak.
  Mitigation: running 90th-percentile (T-digest) instead of max.

---

## 3. `step_lp_rank_quantile` — online empirical rank of the current step

### Definition
At step `t`, compute the rank of `lp_t` among `{lp_0..lp_t}`:
```
rank_t   = #{ i ≤ t : lp_i ≤ lp_t }
quant_t  = rank_t / (t+1)               # in (0,1]
score_t  = 1 - quant_t                  # 1 means current step is the worst so far
```
Maintain a sorted list / order statistic tree; `O(log t)` insert per step,
O(t) memory (fine since t ≤ 64 for typical CoT).

### Cost
FREE / cheap (a single binary insertion per step).

### Hypothesis
Quantile is parameter-free and scale-free — depends only on rank, not the
absolute lp range. Useful when traces have different lp distributions by
topic/length. A step whose lp is the worst seen so far
(`quant_t ≈ 1/(t+1)`) is a strong relative signal.

### Pseudocode
```python
import bisect
sorted_lps = []
qs = []
for t, lp_t in enumerate(per_step_lp):
    bisect.insort(sorted_lps, lp_t)
    rank = bisect.bisect_right(sorted_lps, lp_t)   # 1..t+1
    q = rank / (t+1)
    qs.append(1.0 - q)                # high = bad
score = max(qs[2:])                   # ignore early steps
```

### CP usage
* Trajectory-mode: `nonconformity = max_t score_t`. Because the score is
  bounded in [0,1], the calibration plot tends to be much smoother than
  `lp_min`. Recommended: reweight by `t/T_max` to penalize late-trace
  worst-rank events more.
* Step-mode: `score_t > tau` is a natural threshold (`tau ≈ 0.95`).

### Failure modes
* Trivial early degeneracy: step 0 is always rank 1/1. Drop `t < 2`.
* Ties: use `bisect_right` to break optimistically.
* Long traces saturate (rank movement per step shrinks). Mitigation:
  sliding-window variant.

---

## 4. `step_entropy_growth_ratio` — multiplicative entropy drift

### Definition
At step `t > 0`, with running geometric mean `gm_{t-1}` of past entropies:
```
log_gm_{t-1} = mean(log H_0 .. log H_{t-1})
score_t      = log H_t - log_gm_{t-1}        # positive if entropy grew
```
Use a running mean of `log H_i` for O(1) memory.

### Cost
FREE.

### Hypothesis
Entropy spikes are a known leading indicator of branch errors. Absolute
entropy is topic-dependent (math vs. prose), but the *log-ratio* against
the trace's geometric baseline is scale-free. Persistent positive drift =
model becoming less confident — early warning before any single step looks
alarming.

### Pseudocode
```python
log_gm = 0.0; n = 0
gs = []
for t, H_t in enumerate(per_step_entropy):
    lH = math.log(max(H_t, 1e-9))
    if n == 0:
        gs.append(0.0)
    else:
        gs.append(lH - log_gm)
    # online mean of log H
    n += 1
    log_gm += (lH - log_gm) / n
score = max(gs)
# Optional: also report cumulative drift
cum_drift = sum(max(0, g) for g in gs)
```

### CP usage
* Trajectory-mode: use `max(gs)` OR `cum_drift` as the score.
  `cum_drift` is closer in spirit to "total entropy excess" and is monotone
  in the prefix length, which makes its calibration cleaner.
* Step-mode: trigger ABSTAIN when `score_t > tau` *and* cumulative drift
  is also above its own threshold (two-of-two rule reduces false alarms).

### Failure modes
* Cautious mid-trace ("Let me double check...") legitimately raises
  entropy with correct answer. Calibration should absorb; otherwise
  combine with #2 (drawdown) as confirmation.
* Geometric mean over 1-2 samples is fragile — skip `t<3`.

---

## 5. `step_local_lp_jump` — first-difference jumpiness

### Definition
Measure the change between consecutive steps — a *smoothness* signal:
```
delta_t  = lp_t - lp_{t-1}                   # signed
abs_jump = |delta_t|
score_t  = max(0, -delta_t)                  # only down-jumps count
```
And the trace-level signal is `max_t score_t` and/or
`var_t(delta_t)` (jumpiness variance).

### Cost
FREE — one subtraction per step.

### Hypothesis
Smooth confidence trajectories correlate with coherent reasoning ("staying
in flow"). Big down-jumps often mark the moment reasoning derailed (wrong
sub-goal, mis-applied rule). Up-jumps are recoveries — hence asymmetric
`max(0, -delta_t)`.

### Pseudocode
```python
prev = None
down_jumps = []
deltas     = []
for lp_t in per_step_lp:
    if prev is None:
        down_jumps.append(0.0); deltas.append(0.0)
    else:
        d = lp_t - prev
        deltas.append(d)
        down_jumps.append(max(0.0, -d))
    prev = lp_t
score = max(down_jumps)
jumpiness = statistics.pvariance(deltas)
```

### CP usage
* Trajectory-mode: `nonconformity = max_t down_jumps[t]`.
  Optionally compose: `0.7 * max_drawdown + 0.3 * max_down_jump` after
  per-feature standardization on the calibration set.
* Step-mode: a useful early-warning *gate*: only consider abstaining at
  step t if there *was* a down-jump > tau in the last 2 steps.
* Selective abort: combine with a small grace window (allow 1 large
  jump if it isn't followed by sustained low-lp).

### Failure modes
* Segmentation matters: joining two steps compresses jump magnitude.
* Verbal interjections ("Wait,") create harmless down-jumps. Mitigate by
  computing `delta_t` over content tokens only.

---

## 6. `step_neighbor_distribution_shift` — JS divergence between adjacent step token distributions

### Definition
At step `t > 0`, let `P_t` be the **average** next-token distribution
over the tokens in step `t` (we already have logits per position cached
during decoding). Define the bag-of-distributions:
```
P_bar_t = (1/|step_t|) Σ_{pos ∈ step_t} softmax(logits_pos)
score_t = JS(P_bar_t || P_bar_{t-1})           # symmetric, ∈ [0, log 2]
```
We don't need the full vocab — we can keep the **top-k union** (k=64) and
zero-fill the rest, which is a tight upper bound on JS.

### Cost
Cheap. Memory: we keep only `P_bar_{t-1}` (top-k truncation, O(k) per
step). Per step we compute one JS between two k-sparse distributions:
~few hundred FLOPs.

### Hypothesis
A topical shift in the token distribution between adjacent steps signals
the model changed regimes — switched strategies or lost the thread. JS is
symmetric and bounded. Captures info absent from scalar lp/entropy: e.g.
constant entropy but vocabulary cluster pivot.

### Pseudocode
```python
def topk_softmax(logits, k=64):
    idx = np.argpartition(-logits, k)[:k]
    p   = softmax(logits[idx])
    return idx, p

prev_idx, prev_p = None, None
js = []
for step in steps:
    # average distribution within the step (top-k union)
    union = collections.defaultdict(float)
    for pos in step.token_positions:
        idx, p = topk_softmax(pos.logits)
        for i, v in zip(idx, p): union[i] += v / len(step.token_positions)
    cur_idx = list(union); cur_p = np.array([union[i] for i in cur_idx])
    if prev_p is None:
        js.append(0.0)
    else:
        # align supports
        keys = list(set(cur_idx) | set(prev_idx))
        a = np.array([union.get(k,0.0)         for k in keys])
        b = np.array([prev_dict.get(k,0.0)     for k in keys])
        js.append(jensen_shannon(a,b))
    prev_idx, prev_dict = cur_idx, dict(zip(cur_idx,cur_p))
score = max(js)
```

### CP usage
* Trajectory-mode: `max_t JS_t` as the nonconformity score; or
  `mean_t JS_t` as a "regime stability" proxy.
* Step-mode: trigger abstain when `JS_t > tau` *and* `lp_t < lp_floor`
  (regime shift co-occurring with low confidence).
* Useful as a **second-channel** check: a high-lp step that nonetheless
  shows a JS spike is a candidate for "fluent but off-topic" failure.

### Failure modes
* Top-k truncation gives a JS lower bound; tail matters when flat. Use
  larger k (256) for high-entropy positions.
* Adjacent steps on same sub-problem yield low JS even if both wrong —
  JS is only a *change* detector.
* Cost grows with step length; sub-sample 32 token positions per step.

---

## 7. `step_hiddenstate_offaxis` — geodesic deviation from the prefix ridge

### Definition
We assume access to the final-layer hidden state `h_t ∈ R^d` of the last
token of step `t` (already computed during generation). Maintain an
online estimate of the trace's "ridge direction":
* `mu_{t-1}` = running mean of `{h_0..h_{t-1}}`,
* `v_{t-1}`  = top-1 PCA direction of the prefix (rank-1 online via Oja
  or just the dominant eigenvector of the streaming outer-product
  `Σ (h_i - mu)(h_i - mu)^T / (t-1)`).

Then:
```
r_t        = h_t - mu_{t-1}
parallel_t = (r_t · v_{t-1})  v_{t-1}
perp_t     = r_t - parallel_t
score_t    = ||perp_t||_2 / (||r_t||_2 + eps)         # in [0,1]
```
i.e., the fraction of `h_t`'s deviation that is *off-axis* from the
trace's own dominant trajectory direction.

### Cost
Cheap if `d` is moderate (say last hidden of size 4096–8192) and we use
Oja's rule (one rank-1 update O(d)). Memory O(d). Per-step compute O(d).
No extra forward pass — we already have hidden states.

### Hypothesis
A coherent reasoning trace drifts along a low-rank "ridge" in hidden-state
space (steps build on each other smoothly). A step shooting off-axis is
geometrically "out of context". Representation-level analogue of #5; may
catch failures invisible to scalar lp/entropy (fluent but conceptually
off-track). Self-referential — every trace defines its own ridge — so
length- and topic-invariant.

### Pseudocode
```python
mu = np.zeros(d); v = np.random.randn(d); v /= np.linalg.norm(v); n = 0
oja_lr = 0.1
scores = []
for t, h_t in enumerate(per_step_hidden):
    if n < 2:
        scores.append(0.0)
    else:
        r        = h_t - mu
        proj     = (r @ v) * v
        perp     = r - proj
        s = np.linalg.norm(perp) / (np.linalg.norm(r) + 1e-6)
        scores.append(s)
    # online mean
    n += 1
    mu += (h_t - mu)/n
    # Oja step on the *centered* vector
    rc = h_t - mu
    v  = v + oja_lr * (rc @ v) * rc
    v /= (np.linalg.norm(v) + 1e-9)
score = max(scores)
```

### CP usage
* Trajectory-mode: `max_t off_axis_t` as the nonconformity; or, for a
  more stable signal, `mean of the top-k off-axis steps`.
* Combine multiplicatively with a confidence channel:
  `s_t = off_axis_t * (1 - quant_t)` (off-axis AND low rank).
* Step-mode: gate on `off_axis_t > tau`; particularly useful **late**
  in long traces where the ridge is well-estimated.

### Failure modes
* Ridge unreliable for `t < 4`; emit zero, only use on traces ≥ 5 steps.
* Anisotropic embeddings / layer-norm artifacts dominate geometry.
  Standardize `h_t` per-coordinate before projection.
* Legitimate sub-task switches (math → prose) yield harmless big off-axis
  steps. Pair with #6 (JS) to disambiguate.

---

## Summary table

| # | Name                           | Memory  | Compute/step | Captures                          | Best CP mode             |
|---|--------------------------------|---------|--------------|-----------------------------------|--------------------------|
| 1 | `step_lp_zscore`               | O(1)    | O(1)         | absolute deviation                | trajectory + selective   |
| 2 | `step_lp_drawdown`             | O(1)    | O(1)         | peak-to-now degradation           | trajectory               |
| 3 | `step_lp_rank_quantile`        | O(t)    | O(log t)     | scale-free relative ordering      | trajectory + threshold   |
| 4 | `step_entropy_growth_ratio`    | O(1)    | O(1)         | uncertainty drift                 | trajectory (cum_drift)   |
| 5 | `step_local_lp_jump`           | O(1)    | O(1)         | smoothness / regime shift         | gate for selective abort |
| 6 | `step_neighbor_distribution_shift` | O(k) | O(k)         | distributional regime change      | second channel           |
| 7 | `step_hiddenstate_offaxis`     | O(d)    | O(d)         | geometric coherence               | trajectory (late steps)  |

## Recommended composite

If forced to ship one composite step-level score for CoT-CP, I'd start
with a *standardized z-additive* combination of #1, #2, #4, #5
(all O(1) memory, all FREE), then layer #3 as a robust scale-free
override:

```
s_t = max(
    rank_quantile_t,          # 0..1, scale-free
    sigmoid(α·z_t + β·drawdown_t + γ·entropy_drift_t + δ·down_jump_t)
)
score_question = max_t s_t          # trajectory-level CP score
```

The per-feature weights `(α, β, γ, δ)` are fit on the calibration set
by maximizing AUROC against `correct?`. Add #6 / #7 as gating
auxiliaries for selective-abort decisions. This gives split CP a
unified, length-invariant nonconformity that should outperform the
trajectory-only `lp_min` / `prm_min` on traces where the **shape**
of the confidence curve, not its floor, is what matters.

## Notes on robustness across trace lengths

All seven scores are normalized by trace-internal statistics (#1, #2, #3,
#4, #7) or are bounded distances (#6), so length-invariant by construction.
The only length-coupling risk is the warm-up: first 2–3 steps unreliable
for relative methods. Handle by emitting zero (or skipping max) for
`t < 3`. For very short traces (≤ 3 steps), fall back to trajectory-level
baselines.

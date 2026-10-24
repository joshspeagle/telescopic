---
file_format: mystnb
kernelspec:
  name: python3
---

# Settings

## The settings

Each replicate runs a pair of chains, the second `lag` steps behind the first, until
they meet and until time `m`. Its estimate averages the first chain's states from time
`k` to `m`, plus a bias correction made of the differences between the chains before
they meet ([Theory](../theory.md#the-time-averaged-estimator)).

- `k` is the first averaged time, the burn-in. The bias correction is zero for a pair
  that meets within `k + lag` steps.
- `m` is the last averaged time. A replicate runs until $\max(\tau, m)$, so a larger `m`
  averages more states at a proportional cost.
- `lag`, at least 1, is how far the second chain runs behind the first. The automatic
  settings take `lag = k`, so that most pairs meet within `k + lag` steps and their
  estimate is the ordinary MCMC average after a burn-in of `k`.
- `thin`, at least 1, averages every `thin`-th state from `k`
  ([Memory and thinning](#memory-and-thinning)).
- `max_iter` censors the pairs still unmet at that time, which biases the estimates
  ([Diagnostics and warnings](diagnostics.md#censored-pairs-censored)). It must exceed
  `max(lag, m)`; `None`, the default, runs every pair until it meets.

Any $0 \le k \le m$, `lag` $\ge 1$ and `thin` $\ge 1$ give unbiased estimates: the
settings change the cost and the variance, never the bias. {func}`telescopic.sample` and
{meth}`telescopic.CoupledSampler.run` take each of them as an argument.

## The automatic pipeline

With the defaults, before the replicates run:

1. **The tuning pilot.** Kernel parameters left as `None` are tuned from 4 adaptive
   Metropolis chains of $\max(2000, 200d, 10d^2)$ steps each. Each chain starts from a
   draw of the initial distribution and adapts its own scale toward an acceptance rate
   of 0.234; the four share one running mean and covariance, which starts from the
   variances of 100 draws of the initial distribution. The second halves of the chains
   give the mean and covariance that tune the kernel
   ([Kernels and couplings](kernels.md)). Beyond 20 dimensions the chains refactor the
   covariance and keep their states only every $\lceil d/20\rceil$ steps, which bounds
   the pilot's time and memory. Chains that disagree raise a warning that the target may
   have several modes
   ([Diagnostics and warnings](diagnostics.md#disagreeing-tuning-chains-tuning)), and a
   chain that accepts no proposal in its second half raises `ValueError`.
2. **The meeting-time pilot.** 100 pairs with lag 1 run until they meet, or until
   100,000 iterations, where they are censored.
3. **The settings.** `k`, `lag` and `m` follow the recipe of Atchadé & Jacob (2024) from
   the pilot's meeting times ([Theory](../theory.md#choosing-k-m-and-l)); `thin = 1` and no
   cap. When more than 1% of the pilot pairs are censored, `k` cannot be chosen and the
   run raises `ValueError`.

A value given explicitly is kept. Given `k`, the meeting-time pilot does not run, and an
automatic `lag` or `m` follows from it: `lag = max(1, k)` and `m = 10k`. The tuning
pilot runs whenever the kernel has parameters to tune.

{class}`telescopic.CoupledSampler` runs the same pipeline step by step;
`sample(log_prob, init, n, ...)` is `CoupledSampler(log_prob, init, ...).run(n, ...)`.

- {meth}`~telescopic.CoupledSampler.tune` tunes the kernel once, keeps it as
  `sampler.kernel`, and returns the report of the tuning pilot, or `None` when nothing
  was left to tune.
- {meth}`~telescopic.CoupledSampler.meeting_times` runs `n` more pilot pairs, 100 by
  default, and adds them to `sampler.pilot`. The first run's automatic `k` comes from
  all of them, by {func}`telescopic.tuning.choose_k`, which needs at least 100. More
  pilot pairs give a `k` that varies less between runs.
- {meth}`~telescopic.CoupledSampler.run` runs `n_replicates` more replicates and returns
  the result of all of them so far. The first call tunes the kernel and runs the pilot
  when needed, and fixes `k`, `m`, `lag`, `thin` and `max_iter`, kept in
  `sampler.settings`; later calls keep them, and another value raises `ValueError`.
  [Parallel runs](parallel.md#interrupting-a-run) describes what Ctrl-C does during a
  run.

[Choosing k, m and the lag](../tutorials/choosing-settings.md) runs these steps one by
one, and compares other settings with the recipe's.

## A slow log-density

When the log-density is slow, its evaluations set the time of a run. A counter added to
the log-density of the line fit of [Getting started](../getting-started.md#a-first-run)
counts them in each step:

```{code-cell} python
import numpy as np
from scipy import optimize

import telescopic

rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)
calls = 0


def log_prob(theta, x, y, yerr):
    global calls
    calls += 1
    slope, intercept = theta
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2)


fit = optimize.minimize(lambda theta: -log_prob(theta, x, y, yerr), [0.0, 0.0])
init = telescopic.initial.Normal(mean=fit.x, cov=100 * fit.hess_inv)
sampler = telescopic.CoupledSampler(log_prob, init, args=(x, y, yerr), rng=42)
counts = {}
calls = 0
sampler.tune()
counts["tuning"] = calls
calls = 0
tau = sampler.meeting_times()
counts["pilot"] = calls
calls = 0
result = sampler.run(100)
counts["run"] = calls
print(counts, "pilot steps:", int(tau.sum()), "result.n_log_prob:", result.n_log_prob)
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert counts["tuning"] == 4 * max(2000, 200 * 2, 10 * 2**2) + 4
assert 1.5 < counts["pilot"] / tau.sum() < 2.5
assert counts["run"] == result.n_log_prob
```

For the random walk, tuning evaluates the log-density $4\max(2000, 200d, 10d^2) + 4$
times: once per step of each of the four chains, and once at each chain's start. MALA
and HMC add four chains of 1000 steps that tune the step size, which evaluate the
log-density and its gradient once per step (MALA) or once per leapfrog step (HMC). The
pilot's pairs evaluate it about twice per step, and `result.n_log_prob` counts only the
run's evaluations. Tuning and the pilot can be skipped:

- **Tuning** runs only for a kernel with parameters left unset. A kernel whose
  parameters are given skips it, such as a `RandomWalk` with an optimiser's inverse
  Hessian as `cov`.
- **The meeting-time pilot** runs only when `k` is `"auto"`. Give `k`, for example that
  of an earlier run on the same target, and `lag` and `m` follow from it.

A pool runs the pilot and the replicates on several processes, but tuning runs in this
process ([Parallel runs](parallel.md)). With both skipped:

```{code-cell} python
calls = 0
quick = telescopic.CoupledSampler(
    log_prob,
    init,
    args=(x, y, yerr),
    kernel=telescopic.kernels.RandomWalk(fit.hess_inv),
    rng=42,
)
quick_result = quick.run(100, k=result.k)
print("all:", calls, "n_log_prob:", quick_result.n_log_prob, "pilot:", quick.pilot)
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert calls == quick_result.n_log_prob and quick.pilot is None
```

Every evaluation belongs to the run, and no pilot ran.

## How many replicates

- {func}`telescopic.sample` runs 100 by default. The standard errors fall as
  $1/\sqrt R$, and the error bars are valid as $R$ grows
  ([Theory](../theory.md#guarantees-and-their-limits)).
- The estimates are exactly unbiased for a number of replicates fixed before the run.
  Choosing it after looking at the estimates, for example by extending a run until the
  error bar is small enough, is optional stopping: the estimates stay consistent and the
  large-$R$ intervals stay valid, but the mean is no longer exactly unbiased. To size a
  run, run a first sampler and read the standard error $s_1$ of the estimate that
  matters at $R_1$ replicates; then run a new sampler, with another `rng`, of
  $R = R_1 (s_1 / s)^2$ replicates for a standard error near $s$.
- Two warnings need enough replicates: one replicate's share of the variance is checked
  from 100 replicates, and the tail of the meeting times from 500
  ([Diagnostics and warnings](diagnostics.md)).

## Cost

The cost of a replicate is the number of times it applies the kernel, a coupled step
before the meeting counting as two; `result.costs` holds it for each replicate. The
relative cost is the mean cost over the $m - k + 1$ steps of the MCMC average alone, and
the theory gives its value under the automatic settings
([Theory](../theory.md#choosing-k-m-and-l)). `result.diagnostics()` gives it, with the mean
cost and the seconds per replicate.

The inefficiency of an estimate is the mean cost times the variance of one replicate's
estimate, $\mathbb E[C]\,\mathbb V[H]$ ([Theory](../theory.md#cost)): the figure to
minimise when comparing kernels or settings at equal work. For the slope of the run
above, with the cost in kernel applications and in evaluations of the log-density:

```{code-cell} python
slope = result.estimate(lambda t: t[:, 0])
variance = np.var(slope.replicates, ddof=1)  # of one replicate's estimate
print(f"in kernel applications: {result.costs.mean() * variance:.3g}")
print(f"in evaluations: {result.n_log_prob / len(result.index) * variance:.3g}")
```

Kernels differ in what an application costs, so to compare kernels, count evaluations,
`(result.n_log_prob + result.n_grad) / len(result.index)`, or seconds,
`result.wall_time / len(result.index)`, in place of `result.costs.mean()`.

A larger `m` relative to `k` brings the relative cost closer to 1 and usually lowers the
variance of each replicate, but makes each replicate longer, so fewer fit in a budget of
time.

## Memory and thinning

A result stores each replicate's signed measure
([Theory](../theory.md#signed-measures)): one atom per averaged state,
$\lfloor (m - k)/s \rfloor + 1$ of them for `thin` $= s$, and two per term of the bias
correction, each atom of $8d + 16$ bytes (its coordinates, its weight and its
replicate). Before each run the sampler estimates the total for all the replicates, and
emits an {class}`~telescopic.warnings.InputWarning` above 1 GB.

`thin` $= s$ stores about $1/s$ of the averaged states. The run is as long, and the
estimates stay unbiased ([Theory](../theory.md#the-time-averaged-estimator)). Thinning
by 10, with the seed, kernel and `k` of the run above:

```{code-cell} python
thinned = telescopic.CoupledSampler(
    log_prob, init, args=(x, y, yerr), kernel=sampler.kernel, rng=42
).run(100, k=result.k, thin=10)
print(len(result.measure.weights), "atoms against", len(thinned.measure.weights))
same = np.array_equal(result.meeting_times, thinned.meeting_times)
print("same meeting times:", same)
thinned.mean().standard_error / result.mean().standard_error
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert np.array_equal(result.meeting_times, thinned.meeting_times)
assert 0.09 < len(thinned.measure.weights) / len(result.measure.weights) < 0.11
ratio = thinned.mean().standard_error / result.mean().standard_error
assert np.all((ratio > 1) & (ratio < np.sqrt(10) / 2))
```

The same seed, kernel and settings give the same chains, so the two runs differ only in
the states they average. Thinning by 10 stores a tenth of the atoms, and the standard
errors grow by the factors above, far less than the $\sqrt{10}$ of independent states,
since neighbouring states of a random walk are strongly correlated.

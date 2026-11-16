---
file_format: mystnb
kernelspec:
  name: python3
---

# Set k, m, the lag, thin and max_iter

## The settings

Each {term}`replicate` runs a {term}`pair` of chains, the second `lag` steps behind the first, until
they meet and until time `m`. Its estimate averages the first chain's states from time
`k` to `m`, plus a bias correction made of the differences between the chains before
they meet ([Theory](../theory.md#the-time-averaged-estimator)).

- `k` is the first averaged time, the {term}`burn-in`. The bias correction is zero for a pair
  that meets within `k + lag` steps.
- `m` is the last averaged time. A replicate runs until $\max(\tau, m)$, so a larger `m`
  averages more states at a proportional cost.
- `lag`, at least 1, is how far the second chain runs behind the first. The automatic
  settings take `lag = k`, so that most pairs meet within `k + lag` steps and their
  estimate is the ordinary MCMC average after a burn-in of `k`.
- `thin`, at least 1, averages every `thin`-th state from `k`
  ([Memory and thinning](#memory-and-thinning)).
- `max_iter` censors the pairs still unmet at that time, which biases the estimates
  ([Censored pairs](../reference/warnings.md#censored-pairs-censored)). It must exceed
  `max(lag, m)` and be at most $2^{31} - 1$, the largest time the {term}`atoms <atom>` record; `None`,
  the default, runs every pair until it meets.
- `keep` stores the {term}`signed measures <signed measure>` of the replicates of index below it only, and the
  moments of every replicate ([Keeping a subset](#keeping-a-subset)); `None`, the
  default, stores every replicate's signed measure.

Any $0 \le k \le m$, `lag` $\ge 1$, `thin` $\ge 1$ and `keep` give {term}`unbiased` estimates:
the settings change the cost and the variance, never the bias.
 {func}`telescopic.sample` and
{meth}`telescopic.CoupledSampler.run` take each of them as an argument.

## The automatic pipeline

```{diagram} pipeline
The steps of a run with the default settings, from top to bottom, with the arguments
that skip a step, or the settings given that are kept, at the right.
```

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
   have several modes or the chains may not have converged
   ([Disagreeing tuning chains](../reference/warnings.md#disagreeing-tuning-chains-tuning)); chains
   that agree can still all sit in one mode. A start far wider than the target raises
   a warning too
   ([A start far wider than the target](../reference/warnings.md#a-start-far-wider-than-the-target-wide)). A chain
   that accepts no proposal in its second half raises `ValueError`.
2. **The meeting-time pilot.** 100 pairs with {term}`lag` 1 run until they meet, or until
   100,000 iterations, where they are censored.
3. **The settings.** `k`, `lag` and `m` follow the recipe of Atchadé & Jacob (2024) from
   the pilot's {term}`meeting times <meeting time>` ([Theory](../theory.md#choosing-k-m-and-l)); `thin = 1` and no
   cap. When more than 1% of the pilot pairs are censored, `k` cannot be chosen and the
   run raises `ValueError`, with a note on the step size when MALA or HMC tuned theirs
   ([When the automatic step fails](step-size.md#when-the-automatic-step-fails)).

A value given explicitly is kept. Given `k`, the {term}`meeting-time pilot` does not run, and an
automatic `lag` or `m` follows from it: `lag = max(1, k)` and `m = 10k`. The
{term}`tuning pilot` runs whenever the kernel has parameters to tune.

{class}`telescopic.CoupledSampler` runs the same pipeline step by step;
`sample(log_prob, init, n, ...)` is `CoupledSampler(log_prob, init, ...).run(n, ...)`.
Before any of it, a sampler built with a gradient and a kernel that evaluates it
compares the gradient with differences of the log-density, `grad` with those of
`log_prob` or the gradient of `log_prob_and_grad` with those of its own values, unless
`check_grad=False` ([Checking the gradient](log-density.md#checking-the-gradient)).

- {meth}`~telescopic.CoupledSampler.tune` tunes the kernel once, keeps it as
  `sampler.kernel`, and returns the report of the tuning pilot, or `None` when nothing
  was left to tune.
- {meth}`~telescopic.CoupledSampler.meeting_times` runs `n` more pilot pairs, 100 by
  default, and adds them to `sampler.pilot`. The first run's automatic `k` comes from
  all of them, by {func}`telescopic.tuning.choose_k`; a run first completes a pilot of
  fewer than 100 pairs. More pilot pairs give a `k` that varies less between runs.
- {meth}`~telescopic.CoupledSampler.run` runs `n_replicates` more replicates and returns
  the result of all of them so far. The first call tunes the kernel and runs the pilot
  when needed, and fixes `k`, `m`, `lag`, `thin` and `max_iter`, kept in
  `sampler.settings`; later calls keep them, and another value raises `ValueError`. A
  setting that is not an integer raises `TypeError`, and one that breaks a bound
  `ValueError`, before any tuning; the bounds that depend on an automatic `k`, `m` at
  least `k` and `max_iter` above `lag` and `m`, are checked once the pilot has chosen it.
  [Parallel runs](parallel.md#interrupting-a-run) describes what Ctrl-C does during a
  run. `start=i` runs the indices `i` to `i + n_replicates - 1` instead, as a task of a
  run split into ranges, and `checkpoint=directory` writes the replicates there as they
  finish ([Checkpoints](save-load-merge.md#checkpoints)).
- {meth}`~telescopic.CoupledSampler.plan` fixes the settings as the first run would,
  tuning and running the pilot, without running replicates, and returns them; with
  `checkpoint` it writes them as the plan of that directory
  ([Run on a cluster](save-load-merge.md#run-on-a-cluster)).
- {meth}`~telescopic.CoupledSampler.resume` makes a new sampler continue a returned or
  loaded result: it takes the result's settings, which its runs then keep as a
  sampler's later calls do, with its seed, replicates, pilot and tuning
  ([Continuing a run](save-load-merge.md#continuing-a-run)).

The settings are fixed by the first run, by `plan`, by `resume`, or by adopting the plan
of a checkpoint directory, which a sampler without settings does in its first `run` or
`plan` with that `checkpoint`.

[Choosing k, m and the lag](../tutorials/choosing-settings.md) runs these steps one by
one, and compares other settings with the recipe's.

## A slow log-density

When the log-density is slow, its evaluations set the time of a run. A counter added to
the log-density of the line fit of the [Quickstart](../start/quickstart.md#a-first-run),
on its data `x`, `y` and `yerr`, counts them in each step:

```{code-cell} python
:tags: [remove-cell]

from line_fit import data

x, y, yerr = data()
```

```{code-cell} python
import numpy as np
from scipy import optimize

import telescopic

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
run's evaluations. With a kernel that evaluates the gradient, the gradient check adds 40
evaluations of `log_prob` and 8 of `grad`, or 40 of `log_prob_and_grad`, with float64
values when the sampler is built, before tuning, which no run counts ([Checking the gradient](log-density.md#checking-the-gradient)).
Tuning and the pilot can be skipped:

- **Tuning** runs only for a kernel with parameters left unset. A kernel whose
  parameters are given skips it, such as a `RandomWalk` with an optimizer's inverse
  Hessian as `cov`.
- **The meeting-time pilot** runs only when `k` is `"auto"`. Give `k`, for example that
  of an earlier run on the same target, and `lag` and `m` follow from it.

A pool runs the pilot and the replicates on several processes, but tuning runs in this
process ([Tuning](parallel.md#tuning)). Passing a configured kernel and a fixed `k` skips both:

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

## A start far wider than the target

The automatic `k` is the 99% quantile of the pilot's meeting times, so about one pair in
a hundred meets while its chain is still on its way to the target. From a start orders
of magnitude wider than the target, those replicates can dominate the variance of the
estimates ([Initial distributions](initial-distribution.md#choosing-one)), and tuning
warns of such a start
([A start far wider than the target](../reference/warnings.md#a-start-far-wider-than-the-target-wide)).
The first remedy is a narrower start. Where none is at hand, a larger `k` can be given
by hand. A confirmation round is an option: the pilot's quantile gives $k_0$, 100 more pairs run
at lag $k_0$, the 99% quantile of their $\tau - k_0$ gives $k_1$, and the run takes
$k = \max(k_0, k_1)$, with `lag = k` and `m = 10k` as usual, at the cost of one more
pilot round. The confirmation pairs run with a seed of their own, so that the settings
stay independent of the replicates and the estimates unbiased. From a start a hundred
times wider than the line fit's posterior:

```{code-cell} python
wide = telescopic.initial.Normal(fit.x, 100**2 * fit.hess_inv)
cautious = telescopic.CoupledSampler(log_prob, wide, args=(x, y, yerr), rng=42)
k0 = telescopic.tuning.choose_k(cautious.meeting_times())
confirmation = telescopic.engine.run(
    cautious.target,
    cautious.kernel,
    wide,
    np.arange(100),
    k=k0,
    m=k0,
    lag=k0,
    store=False,
    rng=43,
)
k1 = telescopic.tuning.choose_k(confirmation.meeting_time - k0 + 1)
cautious_result = cautious.run(100, k=max(k0, k1))
print("k0:", k0, "k1:", k1, "k:", cautious_result.k)
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert cautious.tuning.initial_width > 30
assert cautious_result.k == max(k0, k1)
assert k1 < k0
assert "variance" in [issue.kind for issue in telescopic.diagnostics.issues(cautious_result)]
```

Tuning warned of the wide start, before any pair ran; `cautious.tuning.initial_width`
holds the width it measured. {func}`telescopic.engine.run` runs the confirmation pairs
without storing their states, and its `meeting_time` holds their $\tau$; the run then
keeps the `k` given, and takes `lag` and `m` from it. The round does not always raise
`k`: here its pairs met sooner than the pilot's, so $k = k_0$, and one replicate still
holds most of the variance of an estimate, as the run's warning says.

## How many replicates

- {func}`telescopic.sample` runs 100 by default, and at least 2, since the estimates and
  their standard errors need two replicates that met. The standard errors fall as
  $1/\sqrt R$, and the error bars are valid as $R$ grows
  ([Theory](../theory.md#guarantees-and-their-limits)).
- The estimates are exactly unbiased for a number of replicates fixed before the run.
  Choosing it after looking at the estimates, for example by extending a run until the
  error bar is small enough, is optional stopping: the estimates stay {term}`consistent` and the
  large-$R$ intervals stay valid, but the mean is no longer exactly unbiased. To size a
  run, run a first sampler and read the standard error $s_1$ of the estimate that
  matters at $R_1$ replicates; then run a new sampler, with another `rng`, of
  $R = R_1 (s_1 / s)^2$ replicates for a standard error near $s$.
- Two warnings need enough replicates: one replicate's share of the variance is checked
  from 100 replicates, and the tail of the meeting times from 500
  ([Warnings and errors](../reference/warnings.md)).

## Cost

The cost of a replicate is the number of times it applies the kernel, a coupled step
before the meeting counting as two; `result.costs` holds it for each replicate. The
{term}`relative cost`, the mean cost per averaged state, is the mean cost divided by
$m - k + 1$ ([Theory](../theory.md#cost)), and
[Choosing k, m and L](../theory.md#choosing-k-m-and-l) gives its value under the
automatic settings. `result.diagnostics()` gives it, with the mean cost and the seconds
per replicate.

The {term}`inefficiency` of an estimate is the mean cost times the variance of one replicate's
estimate, $\mathbb E[C]\,\mathbb V[H]$ ([Theory](../theory.md#cost)): the figure to
minimize when comparing kernels or settings at equal work. We can compute the inefficiency for the slope of the run
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
correction, each atom of $8d + 20$ bytes (its coordinates, its weight, its replicate
and the time it was recorded), and the $\tau - L$ distances of the 1-Wasserstein bound, 24 bytes each
({func}`telescopic.engine.stored_bytes`). A run takes two to three times that at its
peak, since sorting the atoms at the end of a run makes a copy, and its first estimates
2.0 to 2.5 times, for a random walk's rejection rate at `thin` $= 1$ from $d = 2$ on.
Quantiles sort the atoms that are not consecutive repeats, so in one or two dimensions,
with `thin` above 1 or a kernel that rarely rejects, the estimates take up to about five
times the stored size. These figures count the arrays that NumPy allocates, as
`tracemalloc` traces them; the process's resident memory adds Python's own and can run
somewhat higher. Before each run the sampler estimates the total for all the
replicates, and emits an {class}`~telescopic.warnings.InputWarning` that states the
stored size and the peak, with twice the peak as its upper end in one or two
dimensions, when three times the stored size would exceed 1 GB.

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
since neighboring states of a random walk are strongly correlated.

## Keeping a subset

`keep` $= K$ stores the signed measures, their times and the distances of the replicates
of index below $K$ only, and for every replicate its moments: its first averaged state
$X_k$ and the moments of $x - X_k$ under its signed measure, all the products up to
$d = 100$ and the squares above, $8(p + d)$ bytes with $p = d(d + 3)/2$, or $2d$ above
100 dimensions. The mean, the variances, the standard deviations and, up to 100
dimensions, the covariance come from the moments of every replicate, and are those of a
run that keeps every atom up to rounding; the quantiles, the histograms and
{meth}`~telescopic.Result.estimate` come from the kept replicates
([Two sets of replicates](estimates.md#two-sets-of-replicates)). The first run fixes
`keep`, as it fixes `thin`. With the seed and kernel of the run above, keeping 20 of the
100 replicates:

```{code-cell} python
reduced = telescopic.CoupledSampler(
    log_prob, init, args=(x, y, yerr), kernel=sampler.kernel, rng=42
).run(100, k=result.k, keep=20)
print(reduced)
print(len(reduced.measure.weights), "atoms against", len(result.measure.weights))
reduced.mean().value - result.mean().value
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert np.array_equal(reduced.meeting_times, result.meeting_times)
assert reduced.n_kept == 20
assert np.allclose(reduced.mean().value, result.mean().value, rtol=1e-12, atol=0)
assert np.allclose(reduced.std(), result.std(), rtol=1e-12, atol=0)
```

The meeting times and the kept replicates' atoms are those of the run that keeps every
atom, and the mean differs from it only by rounding. Keeping a subset saves memory when
a replicate stores more than $(p + d)/(d + 2.5)$ atoms, 6 at $d = 10$, which every run
with $m$ well above $k$ does; accumulating the moments costs $O(p)$ per atom.


---
file_format: mystnb
kernelspec:
  name: python3
---

# k, m, the lag and the automatic pipeline

## The settings

Each replicate runs a pair of chains, the second `lag` steps behind the first, until
they meet and until time `m`. Its estimate averages the first chain's states from time
`k` to `m`, plus a bias correction made of the differences between the chains before
they meet ([Theory](../theory.md)).

- `k` is the first averaged time, the burn-in. The bias correction is zero for a pair
  that meets within `k + lag` steps.
- `m` is the last averaged time. A replicate runs until $\max(\tau, m)$, so a larger `m`
  averages more states at a proportional cost.
- `lag`, at least 1, is how far the second chain runs behind the first. The automatic
  settings take `lag = k`, so that most pairs meet within `k + lag` steps and their
  estimate is the ordinary MCMC average after a burn-in of `k`.
- `thin`, at least 1, averages every `thin`-th state from `k`: the run stores fewer
  states and is as long.
- `max_iter` censors the pairs still unmet at that time. They are left out of the
  estimates, which biases them ([Pitfalls](pitfalls.md#capping-the-run-length)). It must
  exceed `max(lag, m)`; `None`, the default, runs every pair until it meets.

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
   the pilot's time and memory. When the chains disagree, with $\hat R > \sqrt 2$ in
   some coordinate, a {class}`~telescopic.warnings.ConvergenceWarning` says that the
   target may have several modes. A chain that accepts no proposal in its second half
   raises `ValueError`.
2. **The meeting-time pilot.** 100 pairs with lag 1 run until they meet, or until
   100,000 iterations, where they are censored.
3. **The settings.** `k` is the 99% sample quantile of the pilot's $\tau - 1$, by the
   inverted CDF, and at least 1; `lag = k`; `m = 10k`; `thin = 1`; no cap (Atchadé &
   Jacob 2024; [Theory](../theory.md)). When more than 1% of the pilot pairs are
   censored, `k` cannot be chosen and the run raises `ValueError`.

A value given explicitly is kept. Given `k`, the meeting-time pilot does not run, and an
automatic `lag` or `m` follows from it: `lag = max(1, k)` and `m = 10k`. The tuning
pilot runs whenever the kernel has parameters to tune. Worked examples:
[Choosing settings](../tutorials/choosing-settings.md).

{class}`telescopic.CoupledSampler` runs the same pipeline step by step.
`sample(log_prob, init, n, ...)` is `CoupledSampler(log_prob, init, ...).run(n, ...)`.
{meth}`~telescopic.CoupledSampler.tune` tunes the kernel once, keeps it as
`sampler.kernel`, and returns the report of the tuning pilot, or `None` when nothing was
left to tune:

```{code-cell} python
import numpy as np

import telescopic
from telescopic import examples, tuning

gaussian = examples.gaussian(5)  # N(0, S), S_ij = 0.9^|i - j|, from N(0, 4I)
sampler = telescopic.CoupledSampler(
    gaussian.log_prob, gaussian.initial, vectorize=True, rng=3
)
report = sampler.tune()
sampler.kernel, report.n_steps, report.acceptance
```

{meth}`~telescopic.CoupledSampler.meeting_times` runs `n` new lag-1 pilot pairs, 100 by
default, with a cap `max_iter` of 100,000, and adds them to `sampler.pilot`. The first
run's automatic `k` comes from every pilot pair so far, by
{func}`telescopic.tuning.choose_k`, which needs at least 100. More pilot pairs give a
`k` that varies less between runs.

```{code-cell} python
tau = sampler.meeting_times()
tuning.choose_k(tau), int(np.quantile(tau - 1, 0.99, method="inverted_cdf"))
```

{meth}`~telescopic.CoupledSampler.run` runs `n_replicates` more replicates and returns
the result of all of them so far. The first call tunes the kernel and runs the pilot
when needed, and fixes `k`, `m`, `lag`, `thin` and `max_iter`, kept in
`sampler.settings`. Later calls keep them, and another value raises `ValueError`:

```{code-cell} python
first = sampler.run(100)
result = sampler.run(100)
print(first.k, first.lag, first.m, result.n_replicates)
try:
    sampler.run(100, k=50)
except ValueError as error:
    print(error)
```

Extending a run after looking at its estimates is optional stopping
([Pitfalls](pitfalls.md#choosing-the-number-of-replicates-after-looking)). The first
Ctrl-C during `run` lets the replicates in progress finish and returns a result flagged
`interrupted`; the next call runs the missing replicates first.

## How many replicates

- {func}`telescopic.sample` runs 100 by default. The standard errors fall as
  $1/\sqrt R$, and the error bars are valid as $R$ grows ([Theory](../theory.md)).
- The estimates are exactly unbiased for a number of replicates fixed before the run. To
  size it, run a first sampler and read the standard error $s_1$ of the estimate that
  matters at $R_1$ replicates; then run a new sampler, with another `rng`, of
  $R = R_1 (s_1 / s)^2$ replicates for a standard error near $s$.
- Two warnings need enough replicates: one replicate's share of the variance is checked
  from 100 replicates, and the tail of the meeting times from 500
  ([Diagnostics and warnings](diagnostics.md)).

## Cost

The cost of a replicate is the number of times it applies the kernel, a coupled step
before the meeting counting as two. The relative cost is the mean cost over the
$m - k + 1$ steps of the MCMC average alone ([Theory](../theory.md#cost)); with the
automatic settings it is about 10/9 to 11/9. The result ([Results](results.md)) records
`costs` per replicate, the evaluations `n_log_prob` and `n_grad`, and `wall_time` in
seconds; `result.diagnostics()` gives the mean cost, the seconds per replicate and the
relative cost:

```{code-cell} python
diagnostics = result.diagnostics()
diagnostics["mean_cost"], diagnostics["relative_cost"]
```

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
estimates stay unbiased ([Theory](../theory.md#the-time-averaged-estimator)). The same
seed and settings give the same chains, so these two runs differ only in the states they
average:

```{code-cell} python
thinned = telescopic.CoupledSampler(
    gaussian.log_prob, gaussian.initial, vectorize=True, rng=3
).run(100, k=first.k, thin=10)
for run in (first, thinned):
    atoms, error = len(run.measure.weights), run.mean().standard_error[0]
    print(
        f"thin = {run.thin:2}, lag = {run.lag}, m = {run.m}: {atoms:6} atoms, "
        f"standard error of the mean of x0 {error:.4f}"
    )
```

Neighbouring states of a random walk are strongly correlated, so thinning by 10 stores a
tenth of the atoms here for a slightly larger standard error.

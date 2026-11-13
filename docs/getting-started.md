---
file_format: mystnb
kernelspec:
  name: python3
---

# Getting started

## Installation

telescopic needs Python 3.11 or newer, NumPy, SciPy and threadpoolctl. Install telescopic
and its optional dependencies from a clone of the repository:

```bash
pip install ".[all]"
```

The extras are `plot` (matplotlib, for the plots), `progress` (tqdm, for progress
bars), `parallel` (cloudpickle, which sends functions defined in a notebook to worker
processes), `pandas` (summaries as data frames) and `all`.

## What a run does

A run is a set of independent **replicates**. Each replicate is a pair of Markov chains,
$X$ and $Y$, started from two independent draws of an initial distribution. $X$ runs
ahead of $Y$ by the lag $L$ (the argument `lag`): it first takes $L$ steps alone; then
the two take coupled steps, which leave each an ordinary MCMC chain but make them likely
to land on the same point. They **meet** when $X_t = Y_{t-L}$, and move together
afterwards. The replicate's estimate is the average of $X$ from step $k$, the burn-in
(the argument `k`), to step $m$ (`m`), plus a correction made of the differences between
the chains before they met, which removes the bias that the burn-in leaves. The estimate
is a **signed measure**: points with weights that sum to 1, some of them negative.

The average of the replicates' estimates is **unbiased**: its expectation is the
posterior expectation, whatever $k$, $m$ and $L$, from every initial distribution from
which the chain converges, and its error bar is valid as the number of replicates grows.
[How it works](tutorials/how-it-works.md) shows a pair meeting, and [Theory](theory.md)
derives the estimator and states the conditions.

## A first run

The example below fits a straight line to 50 data points with Gaussian errors. The log-density takes one
point, the slope and the intercept, and returns its logarithm up to a constant, as in
emcee:

```{code-cell} python
import numpy as np
from scipy import optimize

import telescopic

rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)


def log_prob(theta, x, y, yerr):
    slope, intercept = theta
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2)
```

The initial distribution should be wider than the posterior and cover all its modes,
but not wider by orders of magnitude
([Initial distributions](guide/targets.md#choosing-one)). A Gaussian ten times wider
than an optimizer's error bars suits a posterior with one mode:

```{code-cell} python
fit = optimize.minimize(lambda theta: -log_prob(theta, x, y, yerr), [0.0, 0.0])
init = telescopic.initial.Normal(mean=fit.x, cov=100 * fit.hess_inv)
result = telescopic.sample(
    log_prob, init, args=(x, y, yerr), names=["slope", "intercept"], rng=42
)
result.summary()
```

Before the replicates, `sample` ran two short pilots: four adaptive chains, which tuned
the proposal of a random-walk kernel, and 100 pairs, whose meeting times chose `k`, the
lag and `m`. It then ran 100 replicates.

## Reading the summary

The table has one row per parameter: the estimate of the posterior mean, which is
unbiased, and its standard error, the standard deviation and the 16th, 50th and 84th
percentiles, which are **consistent**: they converge to the posterior's values as the
number of replicates grows, but are not unbiased at a fixed number. Below it are the
run's diagnostics and the warnings that apply: [Results](guide/results.md#the-summary)
defines each entry, and [Diagnostics and warnings](guide/diagnostics.md) says what each
warning means.

The check for heavy-tailed meeting times needs at least 500 replicates, so it is off at
the default of 100: pass `n_replicates=500` to `sample` to turn it on
([How many replicates](guide/settings.md#how-many-replicates)).

## Any function, after the run

Each replicate is a signed measure, so any expectation can be estimated without running
again. For example, to estimate the posterior mean of the product of the slope and intercept:

```{code-cell} python
result.estimate(lambda t: t[:, 0] * t[:, 1])
```

The function maps points of shape `(n, d)` to values of shape `(n,)` or `(n, p)`.

## How far from stationarity

The meeting times bound the total variation distance between the law of a chain after
`t` steps and the posterior: the largest difference between the probabilities that the
two give to one event. Where the curve is near 0, the law of a chain started from the
initial distribution is close to the posterior. The estimates do not rely on it, since
the correction removes what remains at `k`:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      A line on a logarithmic scale that falls from 1 as t grows, with a dashed
      vertical line at k.
  figure:
    caption: >-
      The estimated upper bound on the total variation distance between the law of
      $X_t$ and the posterior, against $t$, with the burn-in $k$ dashed.
---
result.plot_tv_bound();
```

## Coming from emcee

| emcee | telescopic |
|---|---|
| walkers, which move together as an ensemble | replicates: independent pairs of chains |
| burn-in steps, discarded | `k`, chosen from pilot meeting times; the bias left at `k` is corrected, not ignored |
| the autocorrelation time, which sets the error bars | meeting times, which set the cost; the error bars come from independent replicates |
| `get_chain(flat=True)` | `result.measure`: atoms whose weights can be negative; they are not samples, so plot them with their weights ([Signed measures](guide/results.md#signed-measures)) |
| `log_prob_fn(theta, *args)`, `vectorize=True`, `pool=` | the same: `log_prob(theta, *args)`, `vectorize=True`, `pool=`, except that `log_prob` returns the log-density alone, without blobs; derived quantities come from `result.estimate` after the run |

telescopic does not make chains mix faster. Where one chain mixes slowly, as between
distant modes, pairs meet late and replicates cost more. Its gain is error bars that do
not depend on judging convergence, from replicates that run in parallel without
communication. [telescopic beside emcee](tutorials/beside-emcee.md) runs both on this
line fit.

## A slow log-density

Before the replicates run, tuning and the meeting-time pilot evaluate the log-density
thousands of times, and tuning does so in this process.
[A slow log-density](guide/settings.md#a-slow-log-density) counts them and says how to
skip either step, and [Parallel runs](guide/parallel.md) how to run the pilot and the
replicates on several processes.

## Next

The tutorials work through examples, starting with
[How it works](tutorials/how-it-works.md). The how-to guides, starting with
[Targets and initial distributions](guide/targets.md), cover each part of the package, and the [theory](theory.md) states what the estimators guarantee.

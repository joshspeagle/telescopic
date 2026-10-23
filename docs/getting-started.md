---
file_format: mystnb
kernelspec:
  name: python3
---

# Getting started

## Installation

telescopic needs Python 3.11 or newer, NumPy and SciPy. From a clone of the repository:

```bash
pip install ".[all]"
```

The extras are `plot` (matplotlib, for the plots), `progress` (tqdm, for progress
bars), `parallel` (cloudpickle, which sends functions defined in a notebook to worker
processes), `pandas` (summaries as data frames) and `all`.

## A first run

A straight line fitted to 50 points with Gaussian errors. The log-density takes one
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

Every pair of chains starts from two independent draws of an initial distribution. It
should be wider than the posterior and cover all its modes, but not wider by orders of
magnitude ([Pitfalls](guide/pitfalls.md)). A Gaussian ten times wider than an
optimiser's error bars suits a posterior with one mode:

```{code-cell} python
fit = optimize.minimize(lambda theta: -log_prob(theta, x, y, yerr), [0.0, 0.0])
init = telescopic.initial.Normal(mean=fit.x, cov=100 * fit.hess_inv)
result = telescopic.sample(
    log_prob, init, args=(x, y, yerr), names=["slope", "intercept"], rng=42
)
result.summary()
```

`sample` tuned a random-walk kernel on pilot chains, chose the burn-in `k`, the lag and
the run length `m` from the meeting times of 100 pilot pairs, and ran 100 replicates,
each a pair of coupled chains run until they met and until `m`.

## Reading the summary

The table has one row per parameter:

- `mean` is the average of the replicates' estimates, an unbiased estimate of the
  posterior mean, and `standard_error` its Monte Carlo error. The error bar is valid as
  the number of replicates grows, however short each run.
- `std` comes from the unbiased estimate of the covariance; it is consistent, not
  unbiased.
- `q16`, `q50` and `q84` are quantiles of the replicates' pooled signed measures, also
  consistent rather than unbiased.

The lines below describe the run:

- the number of replicates `R`, the settings `k`, `m`, `lag` and `thin`, and the kernel;
- the meeting times, which a long tail or censored pairs would make suspect;
- the cost of a replicate in kernel applications and seconds, and the relative cost:
  that cost over the `m - k + 1` states each replicate averages;
- the fraction of pairs that met after `k + lag`, with a one-sided 95% upper bound; with
  `lag = k` it is close to the bound on the total variation distance between the chains
  at `k` and the posterior ([Theory](theory.md));
- the warnings that apply ([Diagnostics and warnings](guide/diagnostics.md)).

## Any function, after the run

Each replicate is a signed measure, so any expectation can be estimated without running
again. The posterior mean of the product of slope and intercept:

```{code-cell} python
result.estimate(lambda t: t[:, 0] * t[:, 1])
```

The function maps points of shape `(n, d)` to values of shape `(n,)` or `(n, p)`.

## How far from stationarity

The meeting times bound the total variation distance between the chains after `t` steps
and the posterior:

```{code-cell} python
result.plot_tv_bound();
```

## Next

The [theory](theory.md) derives the estimators and states what they guarantee. The user
guide covers each part of the package, and the tutorials work through examples.

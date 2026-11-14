---
file_format: mystnb
kernelspec:
  name: python3
---

# Choose an initial distribution

```{code-cell} python
:tags: [remove-cell]

import numpy as np
from scipy import stats

import telescopic


def log_prob_batch(x, mean, sd):
    return -0.5 * np.sum(((x - mean) / sd) ** 2, axis=1)


mean, sd = np.array([1.0, -2.0]), np.array([0.5, 2.0])
```

## Initial distributions

Each replicate draws the starting points $X_0$ and $Y_0$ of its pair independently from
the initial distribution $\pi_0$, with its own random stream. `init` can be:

- an object with a method `rvs(random_state=generator)` that returns one point:
  {class}`telescopic.initial.Normal` (a mean, and a covariance matrix or the variances
  of a diagonal one), {class}`telescopic.initial.Uniform` (the box `[low, high)`),
  {class}`telescopic.initial.Empirical` (the uniform distribution over a set of points),
  or a frozen SciPy distribution such as `scipy.stats.multivariate_normal(mean, cov)`;
- a callable `init(rng)` that returns one point, drawn with the `numpy.random.Generator`
  it receives, so that the run stays reproducible from its `rng`;
- an array of points of shape `(n, d)`, used as `Empirical`, such as the last state of
  an ensemble sampler like emcee. A single point of shape `(d,)` is refused;
  `Empirical([point])` starts every chain there.

The samplers pass `init` to {func}`telescopic.initial.as_initial`, which turns any of
these forms into an object with `rvs`. The dimension `d` comes from a draw. Draws are converted to shape `(d,)`: SciPy returns
a scalar for a normal in one dimension, and shape `(1, d)` for a Dirichlet. A draw of
another shape, or with a non-finite coordinate, raises `ValueError`. We can run each form on the Gaussian
`log_prob_batch` of the [log-density guide](log-density.md#the-log-density), with its
`mean` and `sd`:

```{code-cell} python
points = stats.multivariate_normal(mean, np.diag(sd**2)).rvs(size=50, random_state=1)


def init_t(rng):
    return mean + 3 * sd * rng.standard_t(5, size=2)


inits = {
    "Normal": telescopic.initial.Normal(mean, cov=(3 * sd) ** 2),
    "Uniform": telescopic.initial.Uniform(mean - 5 * sd, mean + 5 * sd),
    "SciPy": stats.multivariate_normal(mean, np.diag((3 * sd) ** 2)),
    "callable": init_t,
    "array": points,
}
for name, init in inits.items():
    result = telescopic.sample(
        log_prob_batch, init, args=(mean, sd), vectorize=True, rng=1
    )
    print(f"{name:>8}: {result.mean()}")
```

## Choosing one

The estimates are unbiased for every initial distribution from which the chain
converges ([Theory](../theory.md#guarantees-and-their-limits)), but the choice sets their
variance and the cost, and decides whether the chains find every mode.

- **Cover every mode.** If every pair starts near one mode and the kernel rarely crosses
  to another, every pair meets in that mode, and the estimates and their error bars
  describe that mode alone. Nothing in the run shows the missing mode at a moderate
  number of replicates:
  [the theory's list of what is not guaranteed](../theory.md#guarantees-and-their-limits)
  cites a published case, and
  [When unbiased is not enough](../tutorials/failure-modes.md#a-start-that-misses-a-mode)
  runs one. The tuning chains are a first check, in one direction only: when their means
  spread more widely than the chains themselves, telescopic warns that the target may
  have several modes
  ([Disagreeing tuning chains](../reference/warnings.md#disagreeing-tuning-chains-tuning)), but
  chains that agree can all sit in one mode. From a start that covers every mode, pairs
  that start in different modes take longer to meet, which the meeting times show,
  unless the kernel was tuned to one mode and meets quickly inside it
  ([A tuned proposal that covers one mode](../tutorials/failure-modes.md#a-tuned-proposal-that-covers-one-mode)).
- **Wider than the target, but not by orders of magnitude.** The automatic settings take
  `k` at the 99% quantile of the pilot's meeting times and the lag equal to `k`, so
  about one pair in a hundred meets while its chain is still on its way to the target.
  That replicate averages states far from the target without correction, and only rarer
  replicates whose other chain is late balance it, in expectation. From a start orders
  of magnitude wider than the target, these replicates dominate the estimates of
  variances, which can come out negative
  ([When unbiased is not enough](../tutorials/failure-modes.md#a-start-far-wider-than-the-target)).
  A start a few to ten times wider than the target avoids this.
- **For a target with one mode,** a Gaussian centered on an optimizer's fit, with its
  inverse Hessian scaled by 100, is ten times wider than the fit's error bars
  ([Quickstart](../start/quickstart.md#a-first-run)).
- **Inside the support.** A replicate that starts where the target density is zero emits
  an {class}`~telescopic.warnings.InputWarning`, as soon as it starts in this process and
  when the run ends on a pool, and its chains stay there until they accept a proposal of
  finite density. A start many proposal scales outside the support may never get one,
  and without `max_iter` the run then waits forever.
- **The last state of an ensemble** keeps the estimates unbiased when the chain
  converges from it, but covers only the modes that the ensemble found.

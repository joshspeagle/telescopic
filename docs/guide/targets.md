---
file_format: mystnb
kernelspec:
  name: python3
---

# Targets and initial distributions

A run needs two inputs: the log-density of the target, and the initial distribution from
which each pair of chains starts.

## The log-density

`log_prob(x, *args, **kwargs)` takes one point `x` of shape `(d,)` and returns its
log-density, up to an additive constant, as a float. The arguments `args` and `kwargs`
of {func}`telescopic.sample` are passed on to it. With `vectorize=True` it takes a batch
of shape `(n, d)` and returns shape `(n,)`, so that a step of many chains costs one call
instead of one per point.

```{code-cell} python
import numpy as np
from scipy import stats

import telescopic


def log_prob(x, mean, sd):
    return -0.5 * np.sum(((x - mean) / sd) ** 2)


def log_prob_batch(x, mean, sd):
    return -0.5 * np.sum(((x - mean) / sd) ** 2, axis=1)


mean, sd = np.array([1.0, -2.0]), np.array([0.5, 2.0])
```

`grad` is the gradient of the log-density, with the same signature, returning shape
`(d,)`, or `(n, d)` with `vectorize=True`. MALA and HMC need it
([Kernels and couplings](kernels.md)). Both functions receive read-only arrays, so one
that writes to its input raises instead of corrupting the chains.

{class}`telescopic.Target` holds these functions. {func}`telescopic.sample` and
{class}`telescopic.CoupledSampler` build one from their arguments; it is needed directly
only with the building blocks, such as kernels, the engine and
{func}`telescopic.testing.check_coupling`. Its `log_prob` takes a batch of shape
`(n, d)`, whether or not the function is vectorised, and counts the evaluations:

```{code-cell} python
points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
one = telescopic.Target(log_prob, args=(mean, sd))
batch = telescopic.Target(
    log_prob_batch, kwargs={"mean": mean, "sd": sd}, vectorize=True
)
one.log_prob(points), batch.log_prob(points), one.n_log_prob
```

With `vectorize=True`, a replicate's results can depend in the last bits on how the
replicates are batched ([Parallel runs](parallel.md#reproducibility)).

## Special values

- `-inf` means zero density. Every Metropolis test rejects a proposal there, so the
  log-density should return it outside the support.
- NaN and `+inf` raise `ValueError` with the point.

SciPy returns NaN for a negative scale, so a model of 20 observations whose scale is not
restricted fails at the first chain that proposes one:

```{code-cell} python
y = np.random.default_rng(2).normal(1.0, 2.0, size=20)


def log_prob_unrestricted(theta, y):
    location, scale = theta
    return np.sum(stats.norm.logpdf(y, location, scale))


unrestricted = telescopic.Target(log_prob_unrestricted, args=(y,))
try:
    unrestricted.log_prob(np.array([[0.0, -1.0]]))
except ValueError as error:
    print(error)
```

Returning `-inf` outside the support fixes it:

```{code-cell} python
def log_prob_restricted(theta, y):
    location, scale = theta
    if scale <= 0:
        return -np.inf
    return np.sum(stats.norm.logpdf(y, location, scale))


restricted = telescopic.Target(log_prob_restricted, args=(y,))
restricted.log_prob(np.array([[0.0, 1.0], [0.0, -1.0]]))
```

For the gradient:

- Where the log-density is `-inf`, the gradient can take any value, NaN included. States
  there carry a gradient of 0, so MALA and HMC propose from them without drift.
- Where the log-density is finite, a non-finite gradient raises `ValueError` at a
  starting point, and at a proposal of a kernel that uses no gradient but whose states
  carry one, as in a mixture with MALA. MALA and HMC reject a proposal with a non-finite
  gradient.

The log-density must be deterministic: the same point must always give the same value.
Pairs meet when their states are equal, and a log-density estimated by simulation breaks
the couplings.

## Constrained parameters

A parameter with a bound, such as a scale $\sigma > 0$, can be sampled through an
unconstrained transform, such as $s = \log\sigma$. The log-density of $s$ adds the
logarithm of the Jacobian $d\sigma/ds = \sigma$, which is $s$. The chains then never
reach a boundary, and every point of the initial distribution lies inside the support.
The model above, with flat priors on the location and the scale:

```{code-cell} python
def log_prob_log_scale(theta, y):
    location, log_scale = theta
    log_jacobian = log_scale
    return np.sum(stats.norm.logpdf(y, location, np.exp(log_scale))) + log_jacobian


init = telescopic.initial.Normal([0.0, 0.0], [1.0, 1.0])
result = telescopic.sample(
    log_prob_log_scale, init, args=(y,), names=["location", "log_scale"], rng=0
)
```

An expectation of $\sigma$ is an expectation of a function of $s$, so the same run gives
it:

```{code-cell} python
result.estimate(lambda t: np.exp(t[:, 1]))
```

[Results](results.md#quantiles-and-credible-intervals) gives the quantiles of the
sampled coordinates and of derived quantities. To sample $\sigma$ itself instead, return
`-inf` where $\sigma \le 0$, as above, and draw the initial distribution inside the
support ([Choosing one](#choosing-one)).

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

The dimension `d` comes from a draw. Draws are converted to shape `(d,)`, since SciPy
returns a scalar in one dimension; a draw of another shape, or with a non-finite
coordinate, raises `ValueError`. Each form, on the Gaussian above:

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

The estimates are unbiased for any initial distribution
([Theory](../theory.md#guarantees-and-their-limits)), but the choice sets their variance
and the cost, and decides whether the chains find every mode.

- **Cover every mode.** If every pair starts near one mode and the kernel rarely crosses
  to another, every pair meets in that mode, and the estimates and their error bars
  describe that mode alone. Nothing in the run shows the missing mode at a moderate
  number of replicates:
  [the theory's list of what is not guaranteed](../theory.md#guarantees-and-their-limits)
  cites a published case, and
  [When unbiased is not enough](../tutorials/failure-modes.md#a-start-that-misses-a-mode)
  runs one. The tuning chains are a first check: when their means spread more widely
  than the chains themselves, telescopic warns that the target may have several modes
  ([Diagnostics and warnings](diagnostics.md#disagreeing-tuning-chains-tuning)). From a
  start that covers every mode, pairs that start in different modes take longer to meet,
  which the meeting times show.
- **Wider than the target, but not by orders of magnitude.** The automatic settings take
  `k` at the 99% quantile of the pilot's meeting times and the lag equal to `k`, so
  about one pair in a hundred meets while its chain is still on its way to the target.
  That replicate averages states far from the target without correction, and only rarer
  replicates whose other chain is late balance it, in expectation. From a start orders
  of magnitude wider than the target, these replicates dominate the estimates of
  variances, which can come out negative
  ([When unbiased is not enough](../tutorials/failure-modes.md#a-start-far-wider-than-the-target)).
  A start a few to ten times wider than the target avoids this.
- **For a target with one mode,** a Gaussian centred on an optimiser's fit, with its
  inverse Hessian scaled by 100, is ten times wider than the fit's error bars
  ([Getting started](../getting-started.md#a-first-run)).
- **Inside the support.** A replicate that starts where the target density is zero emits
  an {class}`~telescopic.warnings.InputWarning`, and its chains stay there until they
  accept a proposal of finite density.
- **The last state of an ensemble** keeps the estimates unbiased, but covers only the
  modes that the ensemble found.

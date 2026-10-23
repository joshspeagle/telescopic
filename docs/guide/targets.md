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

A vectorised function can round a point's value differently in batches of different
sizes. A replicate's results then depend on how the replicates are batched, for instance
on the number of workers, and runs batched differently agree in distribution, not bit
for bit ([Parallel execution](parallel.md)).

## Special values

- `-inf` means zero density. Every Metropolis test rejects a proposal there, so the
  log-density should return it outside the support.
- NaN and `+inf` raise `ValueError` with the point.

SciPy returns NaN for a negative scale, so a model whose scale is not restricted fails
at the first chain that proposes one:

```{code-cell} python
y = np.array([0.3, -1.2, 0.8])


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
the couplings ([Pitfalls](pitfalls.md#a-log-density-that-is-not-a-function)).

## Initial distributions

Each replicate draws the starting points $X_0$ and $Y_0$ of its pair independently from
the initial distribution $\pi_0$, with its own random stream. The estimates are unbiased
for any $\pi_0$ ([Theory](../theory.md)), but the choice changes their variance, the
cost, and whether the chains find every mode.

`init` can be:

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
coordinate, raises `ValueError`. A replicate that starts where the target density is
zero emits an {class}`~telescopic.warnings.InputWarning`, since the initial distribution
should lie inside the support; its chains accept their first proposal of finite density.

Each form, on the Gaussian above:

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

- **Wider than the target, covering every mode.** Pairs that all start near one mode can
  miss the others without a sign in the estimates
  ([Pitfalls](pitfalls.md#an-initial-distribution-that-misses-a-mode)).
- **Not wider by orders of magnitude.** With the automatic settings, a replicate whose
  chain is still far from the target at the burn-in can dominate the estimates of
  variances ([Pitfalls](pitfalls.md#an-initial-distribution-far-wider-than-the-target)).
- **For a target with one mode,** a Gaussian centred on an optimiser's fit, with its
  inverse Hessian scaled by 100, is ten times wider than the fit's error bars
  ([Getting started](../getting-started.md)).
- **The last state of an ensemble** keeps the estimates unbiased, but covers only the
  modes that the ensemble found.

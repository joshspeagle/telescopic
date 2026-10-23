---
file_format: mystnb
kernelspec:
  name: python3
---

# telescopic beside emcee

This page runs telescopic and emcee on the line fit of
[Getting started](../getting-started.md), with the same number of log-density
evaluations. The errors of the data are known, so the posterior is Gaussian and known
exactly, and both samplers' estimates can be checked against it.

```{code-cell} python
:tags: [remove-cell]

from myst_nb import glue
```

## The data and the exact posterior

A line fit like that of emcee's tutorial, without its underestimated errors: 50 points
from the line $y = -0.9x + 4.3$, with Gaussian errors whose standard deviations, between
0.1 and 0.6, are known. The log-density is the log-likelihood, so the prior is flat. It takes one point of shape `(2,)`, or a
batch of shape `(n, 2)` as both samplers pass with `vectorize=True`. `Counted` wraps it
to count the points at which a sampler evaluates it:

```{code-cell} python
import emcee
import numpy as np
from scipy import optimize

import telescopic

rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)


def log_prob(theta, x, y, yerr):
    slope, intercept = theta[..., :1], theta[..., 1:]
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2, axis=-1)


class Counted:
    """log_prob, counting the points at which it is evaluated."""

    def __init__(self):
        self.points = 0

    def __call__(self, theta, *args):
        self.points += len(np.atleast_2d(theta))
        return log_prob(theta, *args)
```

The posterior's mean is the weighted least-squares fit, and its covariance is
$(A^\top C^{-1} A)^{-1}$, where $A$ has the rows $(x_i, 1)$ and $C$ is the diagonal matrix
of the squared errors:

```{code-cell} python
design = np.vander(x, 2)
precision = design.T @ (design / yerr[:, None] ** 2)
exact_mean = np.linalg.solve(precision, design.T @ (y / yerr**2))
exact_std = np.sqrt(np.diag(np.linalg.inv(precision)))
```

## telescopic

The run of Getting started, from a Gaussian ten times wider than an optimiser's error
bars ([Initial distributions](../guide/targets.md#choosing-one)), with the log-density
evaluated in batches:

```{code-cell} python
fit = optimize.minimize(lambda theta: -log_prob(theta, x, y, yerr), [0.0, 0.0])
init = telescopic.initial.Normal(mean=fit.x, cov=100 * fit.hess_inv)
for_telescopic = Counted()
result = telescopic.sample(
    for_telescopic,
    init,
    args=(x, y, yerr),
    vectorize=True,
    names=["slope", "intercept"],
    rng=42,
)
print(f"{for_telescopic.points:,} evaluations, {result.n_log_prob:,} of them in the run")
```

`result.n_log_prob` counts the evaluations of the run; the others are those of tuning and
of the meeting-time pilot, which [A slow log-density](../guide/settings.md#a-slow-log-density)
counts.

## emcee

As in emcee's tutorial, 32 walkers start within about $10^{-4}$ of the optimiser's fit.
They take as many steps as telescopic's evaluations allow. emcee copies NumPy's global
random state when it makes its sampler, so seeding that state fixes the run:

```{code-cell} python
n_walkers = 32
n_steps = for_telescopic.points // n_walkers - 1  # emcee also evaluates the start
start = fit.x + 1e-4 * rng.normal(size=(n_walkers, 2))
for_emcee = Counted()
np.random.seed(42)
ensemble = emcee.EnsembleSampler(n_walkers, 2, for_emcee, args=(x, y, yerr), vectorize=True)
ensemble.run_mcmc(start, n_steps)
print(f"{n_steps:,} steps, {for_emcee.points:,} evaluations")
```

The integrated autocorrelation time $\tau_{\mathrm{int}}$ of each parameter sets the
burn-in and the thinning. As in emcee's documentation, the burn-in is twice the largest
and the thinning half the smallest. The $N$ steps of a walker after the burn-in are worth
about $N/\tau_{\mathrm{int}}$ independent draws, so the Monte Carlo error of a mean over
the $W = 32$ walkers is $\hat\sigma\sqrt{\tau_{\mathrm{int}}/(WN)}$:

```{code-cell} python
tau_int = ensemble.get_autocorr_time()
burn, thin = int(2 * tau_int.max()), int(tau_int.min() / 2)
samples = ensemble.get_chain(discard=burn, thin=thin, flat=True)
emcee_mean, emcee_std = samples.mean(axis=0), samples.std(axis=0)
emcee_error = emcee_std * np.sqrt(tau_int / (n_walkers * (n_steps - burn)))
print(f"tau_int = {tau_int.round(1)}; burn-in {burn} steps, thinning {thin}")
```

## Side by side

The means with their error bars, and the standard deviations, of slope and intercept:

```{code-cell} python
mean = result.mean()
with np.printoptions(precision=5):
    print("means:      exact", exact_mean)
    print("            emcee", emcee_mean, "±", emcee_error)
    print("       telescopic", mean.value, "±", mean.standard_error)
    print("stds:       exact", exact_std)
    print("            emcee", emcee_std)
    print("       telescopic", result.std())
```

```{code-cell} python
:tags: [remove-cell]

rows = [(emcee_mean, emcee_error, emcee_std), (mean.value, mean.standard_error, result.std())]
z = [abs(value - exact_mean) / error for value, error, _ in rows]
std_gap = [abs(std / exact_std - 1) for _, _, std in rows]
glue("largest_z", float(np.max(z)), display=False)
glue("largest_std_gap", float(np.max(std_gap)), display=False)
ratio = mean.standard_error / emcee_error
glue("ratio_slope", float(ratio[0]), display=False)
glue("ratio_intercept", float(ratio[1]), display=False)
```

The means differ from the exact ones by at most {glue:text}`largest_z:.2f` error bars,
and the standard deviations by at most {glue:text}`largest_std_gap:.1%`. With the same
number of log-density evaluations, telescopic's error bars are
{glue:text}`ratio_slope:.2f` and {glue:text}`ratio_intercept:.2f` times emcee's, for the
slope and the intercept. This is one run of each, on one target.

## What the error bars mean

Both error bars estimate the Monte Carlo error of a mean, under different assumptions.

emcee's, $\hat\sigma\sqrt{\tau_{\mathrm{int}}/(WN)}$, treats the walkers as independent
chains that were stationary after the burn-in, and it inherits the error of
$\tau_{\mathrm{int}}$, estimated from the same run. A burn-in that is too short biases
the means by an amount that the error bar does not include.

telescopic's is the standard error of a mean of independent replicates, whose estimates
are unbiased for any burn-in `k` ([Theory](../theory.md#guarantees-and-their-limits)). It
is valid as the number of replicates grows, provided their variance is finite, and it
needs no judgement of convergence: the meeting times bound the distance from
stationarity instead ([Getting started](../getting-started.md)).

## Next

[Choosing k, m and the lag](choosing-settings.md) shows how {func}`telescopic.sample`
chooses `k`, `m` and the lag, and what other settings cost.

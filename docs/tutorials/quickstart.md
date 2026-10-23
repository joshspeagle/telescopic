---
file_format: mystnb
kernelspec:
  name: python3
---

# Quickstart: telescopic beside emcee

This page fits a straight line with emcee and with telescopic. The errors of the data are
known, so the posterior is Gaussian and known exactly, and both samplers' estimates can be
checked against it.

## The data and the exact posterior

The setting of emcee's line-fitting tutorial, with errors of known size: 50 points from
the line $y = -0.9x + 4.3$, with Gaussian errors whose standard deviations lie between 0.1
and 0.6. The log-density is the log-likelihood, so the prior is flat. It takes one point
of shape `(2,)`, or a batch of shape `(n, 2)` as both samplers pass with
`vectorize=True`:

```{code-cell} python
import emcee
import numpy as np
from myst_nb import glue
from scipy import optimize

import telescopic

rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)


def log_prob(theta, x, y, yerr):
    slope, intercept = theta[..., :1], theta[..., 1:]
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2, axis=-1)
```

With a flat prior the posterior is Gaussian. Its mean is the weighted least-squares fit,
and its covariance is $(A^\top C^{-1} A)^{-1}$, where $A$ has the rows $(x_i, 1)$ and $C$
is the diagonal matrix of the squared errors:

```{code-cell} python
design = np.vander(x, 2)
precision = design.T @ (design / yerr[:, None] ** 2)
exact_mean = np.linalg.solve(precision, design.T @ (y / yerr**2))
exact_std = np.sqrt(np.diag(np.linalg.inv(precision)))
```

## emcee

As in emcee's tutorial, 32 walkers start within about $10^{-4}$ of an optimiser's fit and
take 5000 steps. emcee copies NumPy's global random state when it makes its sampler, so
seeding that state fixes the run:

```{code-cell} python
fit = optimize.minimize(lambda theta: -log_prob(theta, x, y, yerr), [0.0, 0.0])

n_walkers, n_steps = 32, 5000
start = fit.x + 1e-4 * rng.normal(size=(n_walkers, 2))
np.random.seed(42)
ensemble = emcee.EnsembleSampler(n_walkers, 2, log_prob, args=(x, y, yerr), vectorize=True)
ensemble.run_mcmc(start, n_steps);
```

emcee's integrated autocorrelation time $\tau$ of each parameter sets the burn-in and the
thinning. As in emcee's documentation, the burn-in is twice the largest $\tau$ and the
thinning half the smallest. The $N$ steps of a walker after the burn-in are worth about
$N/\tau$ independent draws, so the Monte Carlo error of a mean over the $W = 32$ walkers
is $\hat\sigma\sqrt{\tau/(WN)}$:

```{code-cell} python
tau = ensemble.get_autocorr_time()
burn, thin = int(2 * tau.max()), int(tau.min() / 2)
samples = ensemble.get_chain(discard=burn, thin=thin, flat=True)
emcee_mean, emcee_std = samples.mean(axis=0), samples.std(axis=0)
emcee_error = emcee_std * np.sqrt(tau / (n_walkers * (n_steps - burn)))
print(f"tau = {tau.round(1)}; burn-in {burn} steps, thinning {thin}")
```

## telescopic

telescopic starts each pair of chains from two independent draws of an initial
distribution. A Gaussian ten times wider than the optimiser's error bars covers the
posterior without being orders of magnitude wider ([Pitfalls](../guide/pitfalls.md)
explains why both matter):

```{code-cell} python
init = telescopic.initial.Normal(mean=fit.x, cov=100 * fit.hess_inv)
result = telescopic.sample(
    log_prob, init, args=(x, y, yerr), vectorize=True, names=["slope", "intercept"], rng=42
)
result.summary()
```

[Getting started](../getting-started.md) explains what {func}`telescopic.sample` did and
reads the summary line by line.

## Side by side

The means, standard deviations and error bars of the means, with `z` the difference from
the exact mean in units of the error bar:

```{code-cell} python
mean = result.mean()
rows = {
    "emcee": (emcee_mean, emcee_error, emcee_std),
    "telescopic": (mean.value, mean.standard_error, result.std()),
}
print(f"{'':22}{'mean':>9}{'error bar':>11}{'z':>7}{'std':>9}")
for i, name in enumerate(result.names):
    print(f"{name:11}{'exact':11}{exact_mean[i]:9.5f}{'':18}{exact_std[i]:9.5f}")
    for method, (value, error, std) in rows.items():
        z = (value[i] - exact_mean[i]) / error[i]
        print(f"{'':11}{method:11}{value[i]:9.5f}{error[i]:11.5f}{z:7.2f}{std[i]:9.5f}")
distance = [np.abs(value - exact_mean) / error for value, error, _ in rows.values()]
std_gap = [np.abs(std / exact_std - 1) for _, _, std in rows.values()]
glue("largest_z", float(np.max(distance)), display=False)
glue("largest_std_gap", float(np.max(std_gap)), display=False)
```

The means differ from the exact ones by at most {glue:text}`largest_z:.2f` error bars,
and the standard deviations by at most {glue:text}`largest_std_gap:.1%`. The sizes of the
error bars depend on how much work each run did, which this page does not match, so they
do not compare the samplers' efficiency.

## What the error bars mean

Both error bars estimate the Monte Carlo error of a mean, under different assumptions.

emcee's, $\hat\sigma\sqrt{\tau/(WN)}$, treats the walkers as independent chains that
were stationary after the burn-in, and it inherits the error of $\tau$, estimated from the
same run. A burn-in that is too short biases the means by an amount that the error bar
does not include.

telescopic's is the standard error of a mean of independent replicates, whose estimates
are unbiased for any burn-in `k` ([Theory](../theory.md#guarantees-and-their-limits)). It
is valid as the number of replicates grows, however short each run, provided their
variance is finite. No burn-in has to be chosen for the estimates to be unbiased: `k`
changes the cost and the variance of a replicate, not its expectation.

The meeting times also bound the total variation distance between the law of a chain
after $t$ steps and the posterior. At `k`, dashed, the bound is close to the fraction of
pairs that met after `k + lag`, which the summary reports
([Theory](../theory.md#distance-to-stationarity)):

```{code-cell} python
result.plot_tv_bound();
```

## Next

[How it works](how-it-works.md) runs pairs of chains by hand and takes one replicate's
estimate apart.

---
file_format: mystnb
kernelspec:
  name: python3
---

# Gibbs couplings: the pump-failure model

A Gibbs sampler updates blocks of coordinates in turn, each by a draw from its full
conditional distribution given the others. {class}`telescopic.kernels.Gibbs` couples two
such samplers block by block. This tutorial runs it on the pump-failure model of Jacob,
O'Leary & Atchadé (2020, §5.2), whose exact posterior moments
{func}`telescopic.examples.pumps` provides.

## The model

Pump $i$ of 10 failed $s_i$ times in $t_i$ thousand hours, with

$$s_i \sim \mathrm{Poisson}(\lambda_i t_i), \qquad
\lambda_i \sim \mathrm{Gamma}(1.802, \text{rate } \beta), \qquad
\beta \sim \mathrm{Gamma}(0.01, \text{rate } 1).$$

The point is $(\lambda_1, \dots, \lambda_{10}, \beta)$, and each full conditional is a
gamma distribution:

$$\lambda_i \mid \beta \sim \mathrm{Gamma}(1.802 + s_i, \text{rate } \beta + t_i), \qquad
\beta \mid \lambda \sim \mathrm{Gamma}\Big(0.01 + 10 \times 1.802,
\text{rate } 1 + \sum_i \lambda_i\Big).$$

`pumps.conditional(x, b)` returns the conditional of block `b` given the other
coordinates of the point `x`, as an object with `rvs`, `logpdf` and `ppf`, and
`pumps.blocks` lists the 11 blocks, one coordinate each. A Gibbs kernel never evaluates
the log-density, so the sampler gets `log_prob=None`. The pairs start from
`pumps.initial`, uniform on $(0.001, 5)$ in each coordinate.

## Three couplings

`Gibbs` couples the two chains' conditionals of each block in one of three ways:

- `"maximal"`, the default: the rejection maximal coupling, which makes the two draws
  equal with the largest probability that the two conditionals allow;
- `"crn"`: common random numbers, here one uniform through both conditionals' quantile
  functions (`ppf`), which bring the chains closer at each sweep but in exact
  arithmetic never make them equal;
- `"two-scale"`: common random numbers while the chains are farther apart than
  `threshold` at the start of a sweep, then the maximal coupling (Biswas, Bhattacharya,
  Jacob & Johndrow 2022). Here the threshold is the square root of the trace of the
  exact covariance `pumps.cov`: the root-mean-square distance of a posterior draw from
  the posterior mean.

{meth}`telescopic.CoupledSampler.meeting_times` runs pilot pairs with lag 1 and returns
their meeting times, counted in sweeps. With one seed for the three couplings, their
pairs start from the same points:

```{code-cell} python
import numpy as np
from myst_nb import glue

import telescopic
from telescopic import examples
from telescopic.kernels import Gibbs

pumps = examples.pumps()
kernels = {
    "maximal": Gibbs(pumps.conditional, pumps.blocks),
    "crn": Gibbs(pumps.conditional, pumps.blocks, coupling="crn"),
    "two-scale": Gibbs(
        pumps.conditional,
        pumps.blocks,
        coupling="two-scale",
        threshold=np.sqrt(np.trace(pumps.cov)),
    ),
}
tau = {}
for coupling, kernel in kernels.items():
    sampler = telescopic.CoupledSampler(None, pumps.initial, kernel=kernel, rng=1)
    tau[coupling] = sampler.meeting_times(200)
    print(
        f"{coupling:9}  median {np.median(tau[coupling]):3.0f}  "
        f"max {np.max(tau[coupling]):3.0f}"
    )
glue("max_maximal", float(np.max(tau["maximal"])), display=False)
glue("median_maximal", float(np.median(tau["maximal"])), display=False)
glue("median_crn", float(np.median(tau["crn"])), display=False)
glue("median_two_scale", float(np.median(tau["two-scale"])), display=False)
```

The maximal coupling makes every pair meet within {glue:text}`max_maximal:.0f` sweeps.
Common random numbers make them meet too, after a median of {glue:text}`median_crn:.0f`:
the distance between the chains shrinks at every sweep until it reaches the rounding
error of float64, where the two chains' arithmetic gives the same numbers. The two-scale
coupling's pairs meet in a median of {glue:text}`median_two_scale:.0f` sweeps, against
{glue:text}`median_maximal:.0f` for the maximal coupling. On this model the maximal
coupling succeeds even from distant starts, which leaves the switch little to gain; the
two-scale coupling is meant for targets where maximal couplings of distant chains rarely
succeed.

## Estimates against the exact posterior

A run of 1000 replicates with the maximal coupling:

```{code-cell} python
names = [f"lambda{i}" for i in range(1, 11)] + ["beta"]
result = telescopic.sample(
    None, pumps.initial, 1000, kernel=kernels["maximal"], names=names, rng=2
)
result.summary()
```

The estimated posterior means against the exact ones, and their differences in
standard errors:

```{code-cell} python
mean = result.mean()
z = (mean.value - pumps.mean) / mean.standard_error
glue("largest_z", float(np.max(np.abs(z))), display=False)
glue("worst", names[int(np.argmax(np.abs(z)))], display=False)
glue("beyond", int(np.sum(np.abs(z) > 2)), display=False)
print("            exact  estimate  standard error      z")
for name, exact, value, error, score in zip(
    names, pumps.mean, mean.value, mean.standard_error, z
):
    print(f"{name:9}  {exact:7.4g}  {value:8.4g}  {error:14.2g}  {score:5.2f}")
```

The largest difference is {glue:text}`largest_z:.2f` standard errors in magnitude, for
{glue:text}`worst:s`, and the number beyond 2 is {glue:text}`beyond:d` of 11; about one
in twenty is expected beyond 2.

## The cost of a conditional

A sweep loops over the chains in Python and builds a conditional for each chain and
block. A frozen SciPy distribution, such as `scipy.stats.gamma(a, scale=1/rate)`,
takes 250 to 300 µs to build ({class}`telescopic.kernels.Gibbs`), which can dominate a
sweep. The conditionals of `pumps` are instances of a small class with only the methods
that `Gibbs` calls here: `rvs`, `logpdf` for the maximal coupling and `ppf` for common
random numbers.

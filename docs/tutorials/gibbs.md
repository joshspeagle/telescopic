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

```{code-cell} python
:tags: [remove-cell]

from myst_nb import glue
```

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
coordinates of the point `x`, as a small object with `rvs`, `logpdf` and `ppf`, quicker
to build than a frozen SciPy distribution ([Kernels](../guide/kernels.md#gibbs)).
`pumps.blocks` lists the 11 blocks, one coordinate each. A Gibbs kernel never evaluates
the log-density, so the sampler gets `log_prob=None`. The pairs start from
`pumps.initial`, uniform on $(0.001, 5)$ in each coordinate.

## Three couplings

The [kernels guide](../guide/kernels.md#gibbs) defines the three couplings of `Gibbs`:

- `"maximal"`, the default: the rejection maximal coupling of each block's two
  conditionals;
- `"crn"`: common random numbers, here one uniform through both conditionals' `ppf`;
- `"two-scale"`: common random numbers while the chains are farther apart than
  `threshold`, the maximal coupling after; here `threshold` is the square root of the
  trace of the exact covariance `pumps.cov`.

{meth}`telescopic.CoupledSampler.meeting_times` runs pilot pairs with lag 1 and returns
their meeting times, counted in sweeps. With one seed for the three couplings, their
pairs start from the same points:

```{code-cell} python
import numpy as np

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
    median, slowest = np.median(tau[coupling]), np.max(tau[coupling])
    print(f"{coupling:9}  median {median:3.0f}  max {slowest:3.0f}")
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert np.max(tau["maximal"]) <= 10
assert np.median(tau["crn"]) > 5 * np.median(tau["maximal"])
assert np.median(tau["two-scale"]) <= 2 * np.median(tau["maximal"])
```

The maximal coupling makes the pairs meet within a few sweeps. Common random numbers
make them meet too, but later: the distance between the chains shrinks at every sweep
until it reaches the rounding error of float64, where the two chains' arithmetic gives
the same numbers. The two-scale coupling meets about as fast as the maximal one: on this
model the maximal coupling succeeds even from distant starts, which leaves the switch
little to gain. The two-scale coupling is meant for targets where maximal couplings of
distant chains rarely succeed.

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
print("            exact  estimate  standard error      z")
for name, exact, value, error, score in zip(
    names, pumps.mean, mean.value, mean.standard_error, z
):
    print(f"{name:9}  {exact:7.4g}  {value:8.4g}  {error:14.2g}  {score:5.2f}")
```

```{code-cell} python
:tags: [remove-cell]

glue("beyond", int(np.sum(np.abs(z) > 2)), display=False)
```

The differences are of the size of their standard errors. The number beyond 2 is
{glue:text}`beyond:d` of 11, where about one in twenty is expected.

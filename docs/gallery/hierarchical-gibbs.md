---
file_format: mystnb
kernelspec:
  name: python3
---

# A hierarchical model with Gibbs couplings

A two-level normal model, in which groups have their own means drawn from a common
population, has a full conditional for every parameter in closed form. This page writes
those conditionals for {class}`telescopic.kernels.Gibbs`, compares the
{term}`meeting times <meeting time>` of its three {term}`couplings <coupling>`, and checks
the estimates against the posterior means computed by one-dimensional quadrature.
[Gibbs couplings: the pump-failure model](../tutorials/gibbs.md) runs the same steps on a
model that the package provides.

```{code-cell} python
:tags: [remove-cell]

from myst_nb import glue
```

## The model and the data

Group $j$ of $J = 8$ reports an estimate $y_j$ of its own mean $\theta_j$, with a known
standard error $\sigma_j$, as the schools of the eight-schools example do:

$$y_j \sim \mathcal N(\theta_j, \sigma_j^2), \qquad
\theta_j \sim \mathcal N(\mu, \tau^2), \qquad
\mu \text{ flat}, \qquad \tau^2 \sim \text{Inv-Gamma}(2, \text{scale } 2).$$

The data are simulated with $\mu = 5$ and $\tau = 3$, and standard errors between 4 and
12, so that each group's own estimate is noisy and the model pools them:

```{code-cell} python
import math

import numpy as np
from scipy import integrate, special

import telescopic
from telescopic.kernels import Gibbs

rng = np.random.default_rng(8)
J = 8
sigma = rng.uniform(4, 12, J)
y = rng.normal(rng.normal(5.0, 3.0, J), sigma)
a, b = 2.0, 2.0  # the prior of tau^2: shape and scale
```

## The full conditionals

The point is $(\theta_1, \dots, \theta_8, \mu, \tau^2)$. Given the others, each
$\theta_j$ is normal with precision $1/\sigma_j^2 + 1/\tau^2$, $\mu$ is normal about the
mean of the $\theta_j$, and $\tau^2$ is inverse gamma:

$$\theta_j \mid \mu, \tau^2 \sim \mathcal N\!\left(
\frac{y_j/\sigma_j^2 + \mu/\tau^2}{1/\sigma_j^2 + 1/\tau^2},
\frac{1}{1/\sigma_j^2 + 1/\tau^2}\right), \quad
\mu \mid \theta, \tau^2 \sim \mathcal N\!\left(\bar\theta, \frac{\tau^2}{J}\right), \quad
\tau^2 \mid \theta, \mu \sim \text{Inv-Gamma}\!\left(a + \frac J2,
b + \frac12 \sum_j (\theta_j - \mu)^2\right).$$

`conditional(x, block)` returns the conditional of a block given the other coordinates
of `x` ([Gibbs](../guide/kernels.md#gibbs) states what it must return). Each block here
is one coordinate. The two small classes below have the three methods that the
couplings call, `rvs`, `logpdf` and `ppf`, and are quicker to build than frozen SciPy
distributions, which matters since a sweep builds one per chain and block:

```{code-cell} python
class Normal:
    def __init__(self, mean, sd):
        self.mean, self.sd = mean, sd

    def rvs(self, *, random_state):
        return self.mean + self.sd * random_state.standard_normal()

    def logpdf(self, x):
        z = (x - self.mean) / self.sd
        return -0.5 * z * z - math.log(self.sd) - 0.5 * math.log(2 * math.pi)

    def ppf(self, u):
        return self.mean + self.sd * special.ndtri(u)


class InverseGamma:
    def __init__(self, shape, scale):
        self.shape, self.scale = shape, scale

    def rvs(self, *, random_state):
        return self.scale / random_state.gamma(self.shape)

    def logpdf(self, x):
        if x <= 0:
            return -math.inf
        shape, scale = self.shape, self.scale
        return shape * math.log(scale) - math.lgamma(shape) - (shape + 1) * math.log(x) - scale / x

    def ppf(self, u):
        return self.scale / special.gammainccinv(self.shape, u)


def conditional(x, block):
    theta, mu, tau2 = x[:J], x[J], x[J + 1]
    if block < J:
        precision = 1 / sigma[block] ** 2 + 1 / tau2
        mean = (y[block] / sigma[block] ** 2 + mu / tau2) / precision
        return Normal(mean, precision**-0.5)
    if block == J:
        return Normal(theta.mean(), math.sqrt(tau2 / J))
    return InverseGamma(a + J / 2, b + 0.5 * np.sum((theta - mu) ** 2))


blocks = [[i] for i in range(J + 2)]
names = [f"theta{j}" for j in range(1, J + 1)] + ["mu", "tau2"]
```

## The exact posterior moments

Integrating out the $\theta_j$ and $\mu$ leaves the posterior of $\tau^2$ alone in
closed form, up to a constant (Gelman et al., *Bayesian Data Analysis*, §5.4). With
$w_j = 1/(\sigma_j^2 + \tau^2)$, $V_\mu = 1/\sum_j w_j$ and
$\hat\mu = V_\mu \sum_j w_j y_j$,

$$p(\tau^2 \mid y) \propto p(\tau^2)\, V_\mu^{1/2} \prod_j w_j^{1/2}
\exp\Big(-\tfrac12 \sum_j w_j (y_j - \hat\mu)^2\Big),$$

and given $\tau^2$, $\mu$ is normal with mean $\hat\mu$ and variance $V_\mu$, and each
$\theta_j$ normal too. The posterior means and variances of the ten coordinates are
then integrals over $\tau^2$, which {func}`scipy.integrate.quad_vec` computes together:

```{code-cell} python
def given(tau2):
    # The log-density of tau^2, up to a constant, and the means and variances of the
    # ten coordinates given tau^2.
    w = 1 / (sigma**2 + tau2)
    v_mu = 1 / w.sum()
    mu_hat = v_mu * np.sum(w * y)
    log_density = (
        -(a + 1) * np.log(tau2)
        - b / tau2
        + 0.5 * np.log(v_mu)
        + 0.5 * np.sum(np.log(w))
        - 0.5 * np.sum(w * (y - mu_hat) ** 2)
    )
    # theta_j given tau^2 and mu is normal with a mean linear in mu, and mu is
    # N(mu_hat, v_mu).
    v_theta = 1 / (1 / sigma**2 + 1 / tau2)
    shrink = v_theta / tau2
    mean = np.r_[v_theta * y / sigma**2 + shrink * mu_hat, mu_hat, tau2]
    var = np.r_[v_theta + shrink**2 * v_mu, v_mu, 0.0]
    return log_density, mean, var


# Shifted by the log-density at tau^2 = 10, near the bulk, so that the weights stay
# near 1.
shift = given(10.0)[0]


def integrand(tau2):
    log_density, mean, var = given(tau2)
    return np.exp(log_density - shift) * np.r_[1.0, mean, var + mean**2]


integrals = integrate.quad_vec(integrand, 0, np.inf)[0]
exact_mean = integrals[1 : J + 3] / integrals[0]
exact_var = integrals[J + 3 :] / integrals[0] - exact_mean**2
dict(zip(names, exact_mean.round(3).tolist()))
```

## Three couplings

The [kernels guide](../guide/kernels.md#gibbs) defines the three couplings of `Gibbs`:
`"maximal"`, the default, `"crn"` and `"two-scale"`. As in the pump-failure tutorial,
the `threshold` of `"two-scale"` is the square root of the trace of the posterior
covariance, here from the exact variances. The {term}`pairs <pair>` start from a box that covers the posterior
widely.

{meth}`telescopic.CoupledSampler.meeting_times` runs 100 pilot pairs with
{term}`lag` 1 for each coupling. With one seed for the three, their pairs start from the
same points:

```{code-cell} python
init = telescopic.initial.Uniform(np.r_[np.full(J + 1, -20.0), 0.1], np.r_[np.full(J + 1, 30.0), 100.0])
kernels = {
    "maximal": Gibbs(conditional, blocks),
    "crn": Gibbs(conditional, blocks, coupling="crn"),
    "two-scale": Gibbs(
        conditional, blocks, coupling="two-scale", threshold=np.sqrt(exact_var.sum())
    ),
}
tau = {}
for coupling, kernel in kernels.items():
    sampler = telescopic.CoupledSampler(None, init, kernel=kernel, rng=1)
    tau[coupling] = sampler.meeting_times(100)
    print(
        f"{coupling:9}  median {np.median(tau[coupling]):5.0f}  "
        f"99% {np.quantile(tau[coupling], 0.99, method='inverted_cdf'):5.0f}  "
        f"max {np.max(tau[coupling]):5.0f}"
    )
```

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Three survival curves of the meeting times on logarithmic axes: those of the
      maximal and two-scale couplings fall within a few hundred sweeps, and that of
      common random numbers lies to their right, with a median above a thousand
      sweeps.
  figure:
    caption: >-
      The survival functions of the meeting times of 100 pilot pairs for each
      coupling, less the lag of 1, in sweeps, on log-log axes.
---
import matplotlib.pyplot as plt

from telescopic import plotting

fig, ax = plt.subplots(figsize=(6, 3.5))
for coupling, times in tau.items():
    plotting.survival(times, 1, ax, label=coupling)
ax.set(xscale="log", xlabel="sweeps $t$", ylabel=r"$P(\tau - 1 > t)$")
ax.legend();
```

```{code-cell} python
:tags: [remove-cell]

from IPython.display import Markdown

# The claims of the next paragraph; the first with a wide margin.
median = {coupling: np.median(times) for coupling, times in tau.items()}
assert median["crn"] > 10 * median["maximal"]
glue("crn_over_maximal", float(median["crn"] / median["maximal"]), display=False)
ratio = median["two-scale"] / median["maximal"]
if ratio >= 0.9:
    two_scale = (
        f"has a median meeting time {ratio:.1f} times that of the maximal coupling alone:"
        " on this model the maximal coupling succeeds from distant starts too, which"
        " leaves the switch little to gain"
    )
else:
    two_scale = (
        f"has a median meeting time {ratio:.1f} times that of the maximal coupling alone:"
        " here the common random numbers bring distant chains together faster"
    )
glue("two_scale", Markdown(two_scale), display=False)
```

{term}`Common random numbers <common random numbers>` bring the two chains together but
make them equal only when their distance reaches the rounding error of float64, so their
median meeting time is {glue:text}`crn_over_maximal:.0f` times that of the
{term}`maximal coupling`. The two-scale coupling, which switches to the maximal coupling
only once the chains are within the threshold, {glue:md}`two_scale`.

## Estimates against the exact posterior

A run of 500 {term}`replicates <replicate>` with the maximal coupling. A Gibbs kernel
never evaluates a log-density, so `log_prob` is `None`:

```{code-cell} python
result = telescopic.sample(None, init, 500, kernel=kernels["maximal"], names=names, rng=2)
result.summary()
```

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      For each of the ten coordinates, the difference between the estimated and
      the exact posterior mean, divided by the estimate's standard error, as a dot,
      with dashed lines at plus and minus 2.
  figure:
    caption: >-
      The difference between each estimated posterior mean and its exact value
      from quadrature, in standard errors of the estimate, with $\pm 2$ dashed.
---
mean = result.mean()
z = (mean.value - exact_mean) / mean.standard_error
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(names, z, "o")
for level in (-2, 2):
    ax.axhline(level, color="gray", linestyle="--", linewidth=1)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set(ylabel="(estimate - exact) / s.e.", ylim=(-4, 4))
ax.tick_params(axis="x", rotation=45);
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert np.all(np.abs(z) < 3)
correlation = np.corrcoef(mean.replicates[:, :J + 1].T)
glue("correlation", float(correlation[np.triu_indices(J + 1, 1)].min()), display=False)
assert correlation[np.triu_indices(J + 1, 1)].min() > 0.5
```

The differences are of the size of their standard errors. The nine means of the
$\theta_j$ and $\mu$ move together: across replicates, the smallest correlation between
the estimates of two of them is about {glue:text}`correlation:.2f`, since every $\theta_j$ is
pulled toward $\mu$. Their nine differences are therefore close to one draw, not nine
independent ones.

## What the warnings say

```{code-cell} python
:tags: [remove-cell]

from run_text import warnings_text

glue("warnings_text", Markdown(warnings_text(result)), display=False)
```

{glue:md}`warnings_text` A Gibbs kernel has nothing to tune, so the warning about
disagreeing tuning chains cannot apply. An empty list of warnings means only that none of
the conditions of [Warnings and errors](../reference/warnings.md#warnings) holds, not
that the run is right.

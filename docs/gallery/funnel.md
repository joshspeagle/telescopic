---
file_format: mystnb
kernelspec:
  name: python3
---

# Neal's funnel: a hard case

A target whose scale changes from one region to another is hard for every kernel with a
fixed step, and coupled chains show it: their {term}`pairs <pair>` meet slowly or not at
all, and the run warns. This page runs the random walk and HMC on Neal's funnel, reads
what the {term}`meeting times <meeting time>` and the warnings say, and then removes the
difficulty by changing the coordinates of the target.
[Warnings and errors](../reference/warnings.md) explains each warning, and
[Troubleshoot chains that do not meet](../guide/troubleshooting.md) is indexed by what a
run shows.

```{code-cell} python
:tags: [remove-cell]

from myst_nb import glue
```

## The target

{func}`telescopic.examples.funnel` is the funnel of Neal (2003) in 10 dimensions: the
point is $(v, x_1, \dots, x_9)$, with $v \sim \mathcal N(0, 9)$ and, given $v$, each
$x_i \sim \mathcal N(0, e^v)$. The standard deviation of the $x_i$ is $e^{v/2}$: about 20
at $v = 6$, the mouth of the funnel, and 0.05 at $v = -6$, its neck. A step that suits
one end is 400 times too large or too small at the other. The test functions are $v$,
$v^2$ and the indicator of the neck, $v < -3$, whose exact expectations are 0, 9 and
$\Phi(-1)$. The default initial distribution of the example is a Gaussian with the
target's mean and covariance.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import telescopic
from telescopic import examples

funnel = examples.funnel()
truth = np.array([0.0, 9.0, stats.norm.cdf(-1)])


def moments_of_v(theta):
    v = theta[:, 0]
    return np.column_stack([v, v**2, v < -3])


print("exact E[v], E[v^2] and P(v < -3):", truth.round(4))
```

## Pilot pairs of the random walk and HMC

{meth}`telescopic.CoupledSampler.meeting_times` tunes the kernel, then runs pairs with
{term}`lag` 1 until they meet or reach a cap, as the {term}`meeting-time pilot` of a run
does ([The automatic pipeline](../guide/settings.md#the-automatic-pipeline)), and returns
their meeting times, `inf` for a pair still unmet at the cap. The two samplers have the
same seed, so their {term}`tuning pilots <tuning pilot>` and their pairs start from the
same points. An iteration of HMC evaluates the gradient `n_steps` times, so its pairs get
a smaller cap and fewer pairs, to keep this page within its time:

```{code-cell} python
def pilot(log_prob, grad, initial, kernel, n, cap):
    sampler = telescopic.CoupledSampler(
        log_prob, initial, grad=grad, vectorize=True, kernel=kernel, rng=0
    )
    tau = sampler.meeting_times(n, max_iter=cap)
    met = tau[np.isfinite(tau)]
    print(f"{sampler.kernel}")
    print(
        f"  {np.isinf(tau).sum()} of {n} pairs unmet at {cap}; median of the others "
        f"{np.median(met):.0f}, largest {met.max():.0f}"
    )
    return tau


funnel_tau = {
    "random walk": pilot(funnel.log_prob, funnel.grad, funnel.initial, "rwmh", 200, 20_000),
    "HMC": pilot(funnel.log_prob, funnel.grad, funnel.initial, "hmc", 50, 2_000),
}
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
walk_tau, hmc_tau = funnel_tau["random walk"], funnel_tau["HMC"]
assert np.isfinite(walk_tau).all()
assert walk_tau.max() > 10_000
assert np.median(walk_tau) < 1000
assert np.isinf(hmc_tau).mean() > 0.5
glue("hmc_unmet", int(np.isinf(hmc_tau).sum()), display=False)
```

Every pair of the random walk met, most of them within a few hundred iterations, but the
slowest took more than 10,000, and the `slow` warning above says so as it happens
([Slow pairs](../reference/warnings.md#slow-pairs-slow)). HMC, with the step size that its
tuning chose, left {glue:text}`hmc_unmet` of 50 pairs unmet at the cap. Its step was
tuned toward an acceptance rate on chains in the bulk of the target, which is no
guarantee that coupled chains come together
([When the automatic step fails](../guide/step-size.md#when-the-automatic-step-fails)).

The automatic settings take the {term}`burn-in` `k` from these meeting times
({func}`telescopic.tuning.choose_k`), and a run raises the error below when too many
pilot pairs are unmet:

```{code-cell} python
:tags: [wrap-output]

for name, tau in funnel_tau.items():
    try:
        k = telescopic.tuning.choose_k(tau)
        print(f"{name}: k = {k}, so m = 10k = {10 * k}")
    except ValueError as error:
        print(f"{name}: ValueError: {error}")
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
walk_k = telescopic.tuning.choose_k(walk_tau)
assert walk_k > 1000
glue("walk_k", walk_k, display=False)
glue("walk_m", 10 * walk_k, display=False)
```

For the random walk, the recipe sets $k$ = {glue:text}`walk_k` and $m$ =
{glue:text}`walk_m`, so that each {term}`replicate` would run at least
{glue:text}`walk_m` iterations: a run of 1,000 replicates is far beyond the time of this
page. For HMC, `k` cannot be chosen, and {func}`telescopic.sample` would raise this error
after its pilot ([k cannot be chosen](../guide/troubleshooting.md#k-cannot-be-chosen)). A
step size checked on pilot pairs is the usual remedy
([Checking a step size](../guide/step-size.md#checking-a-step-size)), but on the funnel any
fixed step is too large in the neck or too small in the mouth.

## A run within a budget

To see what a run reports, the random walk runs 500 replicates with settings given by
hand, far shorter than the recipe's: $k = L = 200$ and $m = 2000$, with a cap of 20,000
iterations on each pair. The warnings print below the cell:

```{code-cell} python
walk = telescopic.CoupledSampler(funnel.log_prob, funnel.initial, vectorize=True, rng=0)
budget = walk.run(500, k=200, lag=200, m=2000, max_iter=20_000)
```

```{code-cell} python
estimate = budget.estimate(moments_of_v)
print("E[v], E[v^2], P(v < -3):", estimate)
print("exact values:", truth.round(4))
print("warnings:", [issue.kind for issue in telescopic.diagnostics.issues(budget)])
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
budget_kinds = {issue.kind for issue in telescopic.diagnostics.issues(budget)}
assert {"censored", "late", "variance", "slow"} <= budget_kinds
assert estimate.value[2] == 0 and estimate.standard_error[2] == 0
assert abs(estimate.value[0]) > 3 * estimate.standard_error[0]
assert budget.n_censored >= 1
assert budget.diagnostics()["late_fraction"] > 0.5
assert budget.measure.atoms[:, 0].min() > -3
```

The estimate of $E[v]$ is more than three standard errors from 0, and the probability of
the neck is estimated as 0, with a standard error of 0: no {term}`atom` of any replicate
lies below $v = -3$. The run says that its error bars cannot be trusted, in four
warnings:

- [`censored`](../reference/warnings.md#censored-pairs-censored);
- [`late`](../reference/warnings.md#late-pairs-late);
- [`variance`](../reference/warnings.md#one-replicate-dominating-the-variance-variance);
- [`slow`](../reference/warnings.md#slow-pairs-slow).

The pair left out by the cap is the slowest of the run, and no state that the other
pairs recorded lies in the neck. The warnings name these symptoms; the cause is a kernel
whose single scale cannot serve both ends of the funnel.

## A change of coordinates

Write $x_i = e^{v/2} z_i$. Under the target, $z_i \sim \mathcal N(0, 1)$ independently of
$v$, so in the coordinates $(v, z_1, \dots, z_9)$ the target is the Gaussian
$\mathcal N(0, 9) \times \mathcal N(0, I)$, whose scale is the same everywhere. This is the
non-centered parameterization of the funnel. The law of $v$ does not change, and an
expectation of a function $h$ of the original point is that of
$h(v, e^{v/2} z)$ in the new coordinates. The log-density and its gradient, up to a
constant, and an initial distribution twice as wide as the new target in each coordinate:

```{code-cell} python
def log_prob(theta):
    v, z = theta[:, 0], theta[:, 1:]
    return -(v**2) / 18 - 0.5 * np.sum(z**2, axis=1)


def grad(theta):
    gradient = -theta.copy()
    gradient[:, 0] /= 9
    return gradient


initial = telescopic.initial.Normal(np.zeros(10), 4 * np.diag([9.0] + [1.0] * 9))
noncentered_tau = {
    "random walk, new coordinates": pilot(log_prob, grad, initial, "rwmh", 200, 20_000),
    "HMC, new coordinates": pilot(log_prob, grad, initial, "hmc", 100, 5_000),
}
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
for tau in noncentered_tau.values():
    assert np.isfinite(tau).all() and tau.max() < 1000
```

Every pair of both kernels met within a few hundred iterations. The survival functions of
the meeting times, the fraction of pairs not yet met at each time
([Meeting times and distance to stationarity](../concepts/meeting-times.md)), put the four
pilots side by side:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Four survival curves of meeting times on logarithmic axes. In the new
      coordinates, the curves of the random walk and of HMC fall to zero before
      1,000 iterations. On the funnel, the random walk's curve has a long tail past
      10,000, and HMC's curve stays above one half.
  figure:
    caption: >-
      The fraction of pilot pairs not yet met, $P(\tau > t)$, against $t$, for the random
      walk and HMC on the funnel and in the new coordinates. A curve that ends above 0
      ends at the fraction of pairs unmet at the cap.
---
fig, ax = plt.subplots(figsize=(6, 3.8))
for label, tau in {**funnel_tau, **noncentered_tau}.items():
    met = np.sort(tau[np.isfinite(tau)])
    # Unmet pairs count in the denominator, so the curve ends at their fraction.
    survival = 1 - np.arange(1, len(met) + 1) / len(tau)
    shown = survival > 0  # the last step falls to 0, off a log scale
    ax.step(met[shown], survival[shown], where="post", label=label)
ax.set(xscale="log", yscale="log", xlabel="$t$", ylabel=r"$P(\tau > t)$")
ax.legend(fontsize=9);
```

## The run

The automatic settings, with the default random walk, in the new coordinates:

```{code-cell} python
result = telescopic.sample(
    log_prob,
    initial,
    1000,
    vectorize=True,
    names=["v"] + [f"z{i}" for i in range(1, 10)],
    rng=0,
)
noncentered = result.estimate(moments_of_v)
print("E[v], E[v^2], P(v < -3):", noncentered)
errors = (noncentered.value - truth) / noncentered.standard_error
print("standard errors from the exact values:", errors.round(1))
result.summary()
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraphs.
assert np.all(np.abs(noncentered.value - truth) < 3 * noncentered.standard_error)
assert not telescopic.diagnostics.issues(result)
```

All three estimates lie within three standard errors of the exact values, and the summary
lists no warning. {meth}`~telescopic.Result.plot_marginals` draws the marginal of $v$ from
each run as a signed histogram with boxes of two standard errors
([Plot a result](../guide/plots.md)), here against its exact density:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Two panels, each a histogram of v with boxes of two standard errors over the
      exact normal density of standard deviation 3. Left, the histogram of the
      random walk on the funnel has no mass below about -3 and too much above 0.
      Right, the histogram from the new coordinates follows the curve.
  figure:
    caption: >-
      The marginal of $v$ estimated by the random walk on the funnel within a budget
      (left) and by the run in the new coordinates (right), as signed histograms with
      boxes of two standard errors, against the exact $\mathcal N(0, 9)$.
---
grid = np.linspace(-10, 10, 400)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
for ax, (title, run) in zip(axes, (("funnel, budget run", budget), ("new coordinates", result))):
    run.plot_marginals(0, bins=25, axes=[ax])
    ax.plot(grid, stats.norm.pdf(grid, scale=3), color="0.45", label="exact $N(0, 9)$")
    ax.set(title=title, xlim=(-10, 10))
axes[1].legend(fontsize=9);
```

The warnings, read together: the budget run on the funnel emitted four, and its marginal
of $v$ misses the neck; the run in the new coordinates emitted none, and its marginal
agrees with the exact density. No warning names the funnel's changing scale as the cause.
They report long meeting times, censored pairs and a dominant replicate, which here
pointed to the kernel. An empty list of warnings means only that none of their conditions
holds ([Warnings and errors](../reference/warnings.md#warnings)).

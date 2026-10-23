---
file_format: mystnb
kernelspec:
  name: python3
---

# When unbiased is not enough

Each replicate's estimate is unbiased, whatever the initial distribution and the
settings ([Theory](../theory.md)). A run can still miss the truth by many times its error
bar, or cover it with an error bar too wide to use. This page runs such cases and shows
what telescopic reports for each. [Pitfalls](../guide/pitfalls.md) lists the causes, and
[Diagnostics](../guide/diagnostics.md) explains each warning.

Each run below goes through `run_with_warnings`, which records the warnings that a run
emits and prints them:

```{code-cell} python
import warnings

import matplotlib.pyplot as plt
import numpy as np
from myst_nb import glue
from scipy import optimize, stats

import telescopic
from telescopic import diagnostics, examples
from telescopic.kernels import RandomWalk
from telescopic.tuning import choose_k

np.set_printoptions(precision=3)


def run_with_warnings(sampler, n_replicates, **settings):
    # Runs the sampler and prints the warnings that the run emits.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sampler.run(n_replicates, **settings)
    for warning in caught:
        print(f"{warning.category.__name__}: {warning.message}")
    return result
```

## Two modes

### A start that misses a mode

The target is the mixture $\tfrac12\mathcal N(-4, 1) + \tfrac12\mathcal N(4, 1)$ of
Jacob, O'Leary & Atchadé (2020, §5.1), with their random walk of proposal variance 1,
$k = 50$, $m = 500$, lag 1 and 1,000 replicates. The test function is the indicator of
$x > 3$, whose expectation is $\tfrac12\Phi(1) + \tfrac12\Phi(-7)$. From
$\mathcal N(10, 1)$, every chain starts above the mode at 4, far from the mode at $-4$:

```{code-cell} python
mixture = examples.mixture()
truth = 0.5 * stats.norm.cdf(1) + 0.5 * stats.norm.cdf(-7)


def above_three(x):
    return (x[:, 0] > 3).astype(float)


def compare(result, name):
    estimate = result.estimate(above_three)
    low, high = estimate.interval
    error = float((estimate.value - truth) / estimate.standard_error)
    print(
        f"{estimate}; 95% interval [{low:.4g}, {high:.4g}]; "
        f"{error:+.1f} standard errors from the truth {truth:.4f}"
    )
    glue(f"{name}_error", abs(error), display=False)
    glue(f"{name}_width", float(high - low), display=False)


def bimodal_run(initial):
    sampler = telescopic.CoupledSampler(
        mixture.log_prob,
        initial,
        vectorize=True,
        kernel=RandomWalk(np.eye(1), scale=1.0),
        rng=0,
    )
    return run_with_warnings(sampler, 1000, k=50, m=500, lag=1)


narrow = bimodal_run(telescopic.initial.Normal([10.0], [[1.0]]))
compare(narrow, "narrow")
```

The interval misses the truth by {glue:text}`narrow_error:.0f` standard errors, and
{func}`telescopic.diagnostics.issues` finds {glue:text}`narrow_issues:d` of the
conditions that it checks:

```{code-cell} python
issues = diagnostics.issues(narrow)
glue("narrow_issues", len(issues), display=False)
issues
```

This is the failure that [Pitfalls](../guide/pitfalls.md) describes: every chain starts
on the side of one mode, and the kernel rarely crosses to the other. Most seeds miss: in
50 runs of this setting, 33 intervals missed the truth, none with a warning, and the 17
that covered raised the variance warning (§7.4 of the
[design specification](../design/spec.md)).

### A start that covers both modes

The same kernel and settings from $\mathcal N(10, 10^2)$, the mixture's default initial
distribution (also theirs), which puts mass on both modes:

```{code-cell} python
wide = bimodal_run(mixture.initial)
compare(wide, "wide")
```

The interval now covers the truth, {glue:text}`wide_error:.1f` standard errors away, but
it is {glue:text}`wide_width:.0f` wide, for a probability. The meeting times show why:

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 3.5))
for label, result in (("$N(10, 1)$", narrow), ("$N(10, 10^2)$", wide)):
    tau = np.sort(result.meeting_times)
    survival = 1 - np.arange(1, len(tau) + 1) / len(tau)
    survival[-1] = survival[-2]  # the last step falls to 0, off a log scale
    ax.step(tau, survival, where="post", label=label)
ax.set(xscale="log", yscale="log", xlabel="$t$", ylabel=r"$P(\tau > t)$")
ax.legend();
```

```{code-cell} python
values = wide.estimate(above_three).replicates
measured = {
    "narrow_slowest": narrow.meeting_times.max(),
    "wide_slowest": wide.meeting_times.max(),
    "wide_slow_fraction": np.mean(wide.meeting_times > 200),
    "wide_late": wide.diagnostics()["late_fraction"],
    "wide_tail": diagnostics.hill(wide.meeting_times - wide.lag + 1, len(values) // 10),
    "wide_lowest": values.min(),
    "wide_highest": values.max(),
}
for name, value in measured.items():
    print(f"{name}: {round(float(value), 3)}")
    glue(name, float(value), display=False)
print("2 Phi(1) Phi(-1):", round(2 * stats.norm.cdf(1) * stats.norm.cdf(-1), 3))
```

From $\mathcal N(10, 1)$, every pair met within {glue:text}`narrow_slowest:.0f` steps.
From $\mathcal N(10, 10^2)$, {glue:text}`wide_slow_fraction:.0%` of the pairs took more
than 200 steps, close to the probability $2\Phi(1)\Phi(-1) = 0.27$ that two draws fall on
either side of 0; such a pair meets only after one of its chains crosses between the
modes. The warnings above report the consequences: {glue:text}`wide_late:.0%` of the
pairs met after $k + L$, above the 25% at which telescopic warns; the slowest pair met
after {glue:text}`wide_slowest:,.0f` iterations, past the 10,000 of the slow-pair
warning; and the tail index of the meeting times is {glue:text}`wide_tail:.2f`, below 2,
so the variance of the estimates may be infinite. Until a pair meets, each difference
$\Delta_t$ enters its bias correction with a weight of up to 1 ([Theory](../theory.md)),
so single replicates estimate this probability at values from
{glue:text}`wide_lowest:,.0f` to {glue:text}`wide_highest:,.0f`: their mean is
unbiased, and useless at these settings. The automatic recipe, applied to these meeting
times, gives:

```{code-cell} python
k_wide = choose_k(wide.meeting_times)
mean_cost = wide.diagnostics()["mean_cost"]
glue("wide_cost", mean_cost, display=False)
print(f"k = {k_wide}, m = {10 * k_wide}; mean cost of a replicate here: {mean_cost:.0f}")
```

Each replicate would then cost at least $m$ kernel applications, against
{glue:text}`wide_cost:,.0f` on average here.

### A kernel that crosses

The automatic random walk, tuned on adaptive chains from the same wide start, learns a
proposal covariance near the mixture's variance of 17. With the default scale, 2.38 in
one dimension, its proposals have a standard deviation near 10 and can cross between
modes 8 apart in one step:

```{code-cell} python
tuned = run_with_warnings(
    telescopic.CoupledSampler(mixture.log_prob, mixture.initial, vectorize=True, rng=0),
    1000,
)
print("proposal covariance:", tuned.kernel.cov)
print(f"k = {tuned.k}, m = {tuned.m}, lag = {tuned.lag}")
compare(tuned, "tuned")
```

The interval covers the truth, {glue:text}`tuned_error:.1f` standard errors away. The
wide start made the failure visible, and the kernel that crosses removed it. With a
kernel that cannot cross, as above, the start decides which modes the pairs see.

## A start far wider than the target

The line fit of [Getting started](../getting-started.md), with the log-density
vectorised over a batch of points. Its posterior is Gaussian, so its mean and covariance
follow exactly from least squares:

```{code-cell} python
rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)


def log_prob(theta, x, y, yerr):
    slope, intercept = theta[:, :1], theta[:, 1:]
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2, axis=1)


design = np.vander(x, 2)
exact_cov = np.linalg.inv(design.T @ (design / yerr[:, None] ** 2))
exact_mean = exact_cov @ design.T @ (y / yerr**2)
exact_sd = np.sqrt(np.diag(exact_cov))
start_ratio = np.sqrt(10) / exact_sd
glue("start_slope", float(start_ratio[0]), display=False)
glue("start_intercept", float(start_ratio[1]), display=False)
print("exact mean", exact_mean, "and standard deviations", exact_sd)
print("N(0, 10 I): standard deviations", start_ratio, "times the exact ones")
```

From $\mathcal N(0, 10I)$, whose standard deviations are
{glue:text}`start_slope:.0f` and {glue:text}`start_intercept:.0f` times the posterior's:

```{code-cell} python
def line_fit(initial):
    sampler = telescopic.CoupledSampler(
        log_prob,
        initial,
        args=(x, y, yerr),
        vectorize=True,
        names=["slope", "intercept"],
        rng=42,
    )
    return run_with_warnings(sampler, 100)


far = line_fit(telescopic.initial.Normal(np.zeros(2), 10 * np.eye(2)))
far.summary()
```

One replicate holds {glue:text}`far_share:.0%` of the variance of an estimated mean or
second moment, and the slope's standard deviation is {glue:text}`far_sd_ratio:.2f`
times the exact one. Its variance, estimated as the expectation of $(\theta - \mu)^2$
with the exact mean $\mu$, has a standard error of {glue:text}`far_var_error:.1f` times
the exact variance:

```{code-cell} python
def largest_share(result):
    # The largest share of one replicate in the variance of a mean or second moment.
    moments = result.estimate(lambda theta: np.concatenate([theta, theta**2], axis=1))
    return float(diagnostics.variance_share(moments.replicates).max())


def compare_line(result, name):
    share = largest_share(result)
    spread = result.estimate(lambda theta: (theta - exact_mean) ** 2)
    sd_ratio = result.std() / exact_sd
    error_ratio = spread.standard_error / exact_var
    print("largest share of one replicate:", round(share, 3))
    print("standard deviations over the exact ones:", sd_ratio)
    print("variances over the exact ones:", spread.value / exact_var)
    print("their standard errors over the exact ones:", error_ratio)
    glue(f"{name}_share", share, display=False)
    glue(f"{name}_sd_ratio", float(sd_ratio[0]), display=False)
    glue(f"{name}_var_error", float(error_ratio[0]), display=False)
    return spread


exact_var = np.diag(exact_cov)
spread = compare_line(far, "far")
```

The replicates' estimates of the slope's variance, over the exact one:

```{code-cell} python
ratios = np.sort(spread.replicates[:, 0] / exact_var[0])
glue("far_largest", float(ratios[-1]), display=False)
print(f"smallest {ratios[0]:.3f}, second largest {ratios[-2]:.3f}, largest {ratios[-1]:.1f}")
```

One replicate, at {glue:text}`far_largest:.0f` times the exact variance, causes it.
[Pitfalls](../guide/pitfalls.md) explains how a start this far makes one replicate
dominate. The README starts instead from an optimiser's fit, with its inverse Hessian
scaled by 100: a Gaussian ten times wider than the posterior. In that run the largest
share of one replicate is {glue:text}`widened_share:.0%`, the slope's standard deviation
is {glue:text}`widened_sd_ratio:.3f` times the exact one, and the standard error of its
variance is {glue:text}`widened_var_error:.1%` of the exact variance:

```{code-cell} python
fit = optimize.minimize(lambda theta: -log_prob(theta[None], x, y, yerr)[0], [0.0, 0.0])
widened = line_fit(telescopic.initial.Normal(fit.x, 100 * fit.hess_inv))
compare_line(widened, "widened");
```

In 20 runs from $\mathcal N(0, 10I)$, 11 raised the variance warning, and the slope's
standard deviation ranged from a negative variance to 4.2 times the exact one, while
starts 2 and 10 times wider than the posterior never warned (§4.2 of the
[design specification](../design/spec.md)).

## Reading and catching warnings

`run_with_warnings` recorded the warnings with {class}`warnings.catch_warnings`. Without
it, Python prints them as the run ends, or as they happen for a slow pair and for tuning
chains that disagree. They can also be read from a result at any time:
{func}`telescopic.diagnostics.issues` returns the conditions that apply, each with a kind
and a message. An empty list means only that none of them holds, not that the run is
right.

```{code-cell} python
for issue in diagnostics.issues(far):
    print(f"{issue.kind}: {issue.message}")
```

The summary prints them under its table, as above, and keeps them in its footer:

```{code-cell} python
far.summary().footer["warnings"]
```

{meth}`telescopic.Result.check` emits them again, or raises `RuntimeError` with
`strict=True`. Python's warning filters turn them into exceptions; here around `check`:

```{code-cell} python
with warnings.catch_warnings():
    warnings.simplefilter("error", telescopic.ConvergenceWarning)
    try:
        far.check()
    except telescopic.ConvergenceWarning as error:
        print(f"raised ConvergenceWarning: {error}")
```

Under the same filter, a run raises instead of returning a result: as it happens for a
slow pair or tuning chains that disagree, and as it ends for the other conditions. A
filter on {class}`~telescopic.warnings.TelescopicWarning` also catches
{class}`~telescopic.warnings.InputWarning`, such as the warning about an initial point
where the target density is zero.

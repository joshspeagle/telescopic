---
file_format: mystnb
kernelspec:
  name: python3
---

# When unbiased is not enough

The estimates are unbiased for any settings and every initial distribution from which
the chain converges ([Theory](../theory.md#guarantees-and-their-limits)), yet a run can
miss the truth by many times its error bar, or cover it with an error bar too wide to
use. This page runs such cases and shows what telescopic reports for each. A run's
warnings appear under its cell; [Diagnostics and warnings](../guide/diagnostics.md)
explains each one, and [Initial distributions](../guide/targets.md#choosing-one) gives
the advice on choosing one.

```{code-cell} python
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, special, stats

import telescopic
from telescopic.kernels import Independence, Mixture, MultivariateT, RandomWalk

np.set_printoptions(precision=3)
```

## Two modes

### A start that misses a mode

The target is the mixture $\tfrac12\mathcal N(-4, 1) + \tfrac12\mathcal N(4, 1)$ of
Jacob, O'Leary & Atchadé (2020, §5.1), with their random walk of proposal variance 1,
$k = 50$, $m = 500$, lag 1 and 1,000 replicates. The test function is the indicator of
$x > 3$, whose expectation is $\tfrac12\Phi(1) + \tfrac12\Phi(-7)$. From
$\mathcal N(10, 1)$, every chain starts above the mode at 4, far from the mode at $-4$:

```{code-cell} python
mixture = telescopic.examples.mixture()
truth = 0.5 * stats.norm.cdf(1) + 0.5 * stats.norm.cdf(-7)


def above_three(x):
    return (x[:, 0] > 3).astype(float)


def compare(result):
    estimate = result.estimate(above_three)
    low, high = estimate.interval
    error = (estimate.value - truth) / estimate.standard_error
    print(
        f"{estimate}; 95% interval [{low:.4g}, {high:.4g}]; "
        f"{error:+.1f} standard errors from the truth {truth:.4f}"
    )


def bimodal_run(initial):
    sampler = telescopic.CoupledSampler(
        mixture.log_prob,
        initial,
        vectorize=True,
        kernel=RandomWalk(np.eye(1), scale=1.0),
        rng=0,
    )
    return sampler.run(1000, k=50, m=500, lag=1)


narrow = bimodal_run(telescopic.initial.Normal([10.0], [[1.0]]))
compare(narrow)
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
estimate = narrow.estimate(above_three)
assert abs(estimate.value - truth) > 10 * estimate.standard_error
assert not telescopic.diagnostics.issues(narrow)
```

The interval misses the truth by many standard errors, and the run emits no warning:
{func}`telescopic.diagnostics.issues` finds none of the conditions that it checks.

```{code-cell} python
telescopic.diagnostics.issues(narrow)
```

This is the failure of an initial distribution that misses a mode
([Initial distributions](../guide/targets.md#choosing-one)): every chain starts on the
side of one mode, and the kernel rarely crosses to the other. Most seeds miss the truth
in this setting without a warning
([Measured rates](../guide/diagnostics.md#measured-rates)).

### A start that covers both modes

The same kernel and settings from $\mathcal N(10, 10^2)$, the mixture's default initial
distribution (also theirs), which puts mass on both modes:

```{code-cell} python
wide = bimodal_run(mixture.initial)
```

```{code-cell} python
compare(wide)
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
low, high = wide.estimate(above_three).interval
assert low <= truth <= high
assert high - low > 10
```

The interval now covers the truth, but it is far wider than the range of a probability.
The meeting times show why:

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
measured = wide.diagnostics()
values = wide.estimate(above_three).replicates
print(f"slowest pair from N(10, 1): {narrow.meeting_times.max():.0f} steps")
print(f"pairs slower than 200 steps from N(10, 10^2): {np.mean(wide.meeting_times > 200):.3f}")
print(f"2 Phi(1) Phi(-1): {2 * stats.norm.cdf(1) * stats.norm.cdf(-1):.3f}")
print(f"pairs that met after k + lag: {measured['late_fraction']:.3f}")
print(f"tail index of the meeting times: {measured['tail_index']:.2f}")
print(f"the replicates' estimates: from {values.min():.0f} to {values.max():.0f}")
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert narrow.meeting_times.max() < 100
slow = np.mean(wide.meeting_times > 200)
assert abs(slow - 2 * stats.norm.cdf(1) * stats.norm.cdf(-1)) < 0.05
assert {"late", "slow", "tail"} <= {issue.kind for issue in telescopic.diagnostics.issues(wide)}
assert measured["late_fraction"] > 0.25
assert wide.meeting_times.max() > 10_000
assert measured["tail_index"] < 2
assert values.max() - values.min() > 1000
```

Every pair from $\mathcal N(10, 1)$ met within a few dozen steps. From
$\mathcal N(10, 10^2)$, the fraction of pairs slower than 200 steps is close to
$2\Phi(1)\Phi(-1)$, the probability that two draws fall on either side of 0: such a pair
meets only after one of its chains crosses between the modes. The warnings report the
consequences: more than a quarter of the pairs met after $k + L$, a pair was still unmet
after 10,000 iterations, and the tail index of the meeting times is below 2, so the
variance of the estimates may be infinite. Until a pair meets, each difference
$\Delta_t$ enters its bias correction with a weight of up to 1
([Theory](../theory.md#the-time-averaged-estimator)), so single replicates estimate this
probability at values thousands apart: their mean is unbiased, and useless at these
settings. The automatic recipe, applied to these meeting times, gives:

```{code-cell} python
k_wide = telescopic.tuning.choose_k(wide.meeting_times)
print(f"k = {k_wide}, m = {10 * k_wide}; mean cost here: {measured['mean_cost']:.0f}")
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
m_wide = 10 * k_wide
assert m_wide > 10 * measured["mean_cost"]
```

Each replicate would then cost at least $m$ kernel applications, many times the mean cost
of this run.

### A kernel that crosses

The automatic random walk, tuned on adaptive chains from the same wide start, learns a
proposal covariance near the mixture's variance of 17. With the default scale, 2.38 in
one dimension, its proposals have a standard deviation near 10 and can cross between
modes 8 apart in one step:

```{code-cell} python
tuned = telescopic.CoupledSampler(
    mixture.log_prob, mixture.initial, vectorize=True, rng=0
).run(1000)
print("proposal covariance:", tuned.kernel.cov)
print(f"k = {tuned.k}, m = {tuned.m}, lag = {tuned.lag}")
compare(tuned)
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
low, high = tuned.estimate(above_three).interval
assert low <= truth <= high
# And of the paragraph above the run: a covariance near 17, a standard deviation near 10.
assert abs(tuned.kernel.cov[0, 0] - 17) < 2
assert abs(2.38 * np.sqrt(tuned.kernel.cov[0, 0]) - 10) < 1
```

The interval covers the truth. The wide start made the failure visible, and the kernel
that crosses removed it. A target with several modes needs both: an initial
distribution that covers every mode
([Initial distributions](../guide/targets.md#choosing-one)), and a kernel that crosses
between them, such as this tuned random walk, or a
{class}`~telescopic.kernels.Mixture` with a component whose proposals reach the other
modes ([Kernels and couplings](../guide/kernels.md)). With a kernel that cannot cross,
the start decides which modes the pairs see.

### A tuned proposal that covers one mode

A {class}`~telescopic.kernels.Mixture` whose
{class}`~telescopic.kernels.Independence` component is left to tuning reaches only the
modes that its tuned proposal covers. The target is
$0.7\,\mathcal N(-6e_1, I) + 0.3\,\mathcal N(6e_1, I)$ in 10 dimensions, two modes 12
apart, and the start $\mathcal N(0, 100I)$ covers both. The test function is the
indicator of $x_0 < 0$, whose expectation is 0.7, the weight of the mode at $-6e_1$:

```{code-cell} python
dimension = 10
offset = np.zeros(dimension)
offset[0] = 6.0
log_weights = np.log([0.7, 0.3])


def two_modes(x):
    distances = np.array(
        [np.sum((x + offset) ** 2, axis=1), np.sum((x - offset) ** 2, axis=1)]
    )
    return special.logsumexp(log_weights[:, None] - 0.5 * distances, axis=0)


def left_mode(x):
    return (x[:, 0] < 0).astype(float)


cover = telescopic.initial.Normal(np.zeros(dimension), 100 * np.eye(dimension))


def mixture_run(seed, proposal=None):
    kernel = Mixture([RandomWalk(), Independence(proposal)], [0.8, 0.2])
    return telescopic.sample(two_modes, cover, 400, vectorize=True, kernel=kernel, rng=seed)


one_mode = mixture_run(3)
print(one_mode.estimate(left_mode))
print(f"largest R-hat of the tuning chains: {one_mode.tuning.rhat.max():.3f}")
print(f"location of the proposal along x_0: {one_mode.kernel.kernels[1].proposal.mean[0]:.2f}")
print(f"slowest pair: {one_mode.meeting_times.max():.0f} steps")
telescopic.diagnostics.issues(one_mode)
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert one_mode.estimate(left_mode).value < 0.01
assert one_mode.estimate(left_mode).standard_error < 0.01
assert not telescopic.diagnostics.issues(one_mode)
assert one_mode.tuning.rhat.max() < np.sqrt(2)
assert abs(one_mode.kernel.kernels[1].proposal.mean[0] - 6) < 0.5
assert one_mode.meeting_times.max() < 200
```

The estimate is 0, with an error bar of 0, every pair met within about a hundred steps,
and the run emits no warning. The tuning chains agree, and the Student-$t$ proposal fitted
to them sits in the mode at $6e_1$. The four tuning chains share one proposal covariance;
on this target they settle on one side of the origin during their descent, before they
reach a mode, and the proposal fitted to their second halves covers that mode alone.
Both chains of a pair receive the same proposal, accept it, and meet in that mode. As with a random walk whose steps are much shorter than the gap between
modes ([Theory](../theory.md#guarantees-and-their-limits)), the replicates that would
correct the estimate are too rare to occur in a run.

Runs with ten seeds, each with its warnings counted:

```{code-cell} python
def compare_seeds(proposal=None):
    rows = []
    for seed in range(10):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = mixture_run(seed, proposal)
        estimate = result.estimate(left_mode)
        wrong = abs(estimate.value - 0.7) > 5 * estimate.standard_error
        disagree = result.tuning.rhat.max() > np.sqrt(2)
        variance_only = all("of the variance" in str(w.message) for w in caught)
        rows.append((wrong, disagree, len(caught), variance_only))
        print(
            f"seed {seed}: {estimate}, more than 5 standard errors from 0.7: {wrong}, "
            f"tuning chains disagree: {disagree}, warnings: {len(caught)}"
        )
    return np.array(rows)


tuned_rows = compare_seeds()
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
wrong, disagree, warned, variance_only = tuned_rows.T.astype(bool)
assert wrong.sum() >= 8
assert not np.any(wrong & disagree)
assert disagree.any()
assert np.any(wrong & ~warned)
assert np.all(variance_only[wrong])
```

Most runs estimate the weight 0.7 as 0 or 1, depending on the mode that the tuning
chains found, each with a small error bar. The tuning chains disagree only in a run whose
estimate is right: there they split between the modes, and the proposal fitted to them
spans both. Agreeing tuning chains do not show that the run found every mode. Some of the
wrong runs emit no warning at all, and the others only the warning that one replicate
holds most of the variance, which does not name the cause
([Diagnostics and warnings](../guide/diagnostics.md#measured-rates)).

Two things help. Runs with different seeds disagree by far more than their error bars,
so running more than one seed shows the failure whenever their tuning chains settle in
different modes. They need not: on the same target with a second mode of standard
deviation 0.3, the tuning chains of all 20 seeds tried settled in the wider mode, and
every run estimated the weight 0.7 as 1, so agreement between seeds does not show that
every mode was found. And when the modes are known, from optimizations started at
several points or from the model, an {class}`~telescopic.kernels.Independence` proposal
given explicitly can cover them all. It needs only where the modes are, not their
weights. A Student-$t$ located at the midpoint of the two modes, the origin, whose scale
matrix is the identity widened along $x_0$ by the square of half the distance between
the modes, reaches both:

```{code-cell} python
scale = np.eye(dimension)
scale[0, 0] += offset[0] ** 2  # half the distance between the modes, squared
covering_rows = compare_seeds(MultivariateT(np.zeros(dimension), scale))
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert not covering_rows[:, 0].astype(bool).any()
```

Every estimate now lies within 5 standard errors of 0.7. Such a proposal covers only the
modes it is given: a mode that no optimization found stays as invisible as before.

## A start far wider than the target

The line fit of [Getting started](../getting-started.md), with the log-density of the
[quickstart](quickstart.md), which takes one point or a batch, and its exact Gaussian
posterior, computed as there:

```{code-cell} python
rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)


def log_prob(theta, x, y, yerr):
    slope, intercept = theta[..., :1], theta[..., 1:]
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2, axis=-1)


design = np.vander(x, 2)
precision = design.T @ (design / yerr[:, None] ** 2)
exact_mean = np.linalg.solve(precision, design.T @ (y / yerr**2))
exact_var = np.diag(np.linalg.inv(precision))
exact_std = np.sqrt(exact_var)
print("exact mean", exact_mean, "and standard deviations", exact_std)
print("N(0, 10 I): standard deviations", np.sqrt(10) / exact_std, "times the exact ones")
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
widths = np.sqrt(10) / exact_std
assert 100 <= widths[0] < 1000 and 10 <= widths[1] < 100
```

From $\mathcal N(0, 10I)$, whose standard deviations are hundreds and tens of times the
posterior's:

```{code-cell} python
far = telescopic.sample(
    log_prob,
    telescopic.initial.Normal(np.zeros(2), 10 * np.eye(2)),
    args=(x, y, yerr),
    vectorize=True,
    names=["slope", "intercept"],
    rng=42,
)
far.summary()
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert far.diagnostics()["variance_share"] > 0.5
assert "variance" in [issue.kind for issue in telescopic.diagnostics.issues(far)]
```

One replicate holds most of the variance of an estimate, and the run warns. Against the
exact posterior, with the variances estimated as the expectations of
$(\theta - \mu)^2$ for the exact mean $\mu$:

```{code-cell} python
def compare_line(result):
    spread = result.estimate(lambda theta: (theta - exact_mean) ** 2)
    print("standard deviations over the exact ones:", result.std() / exact_std)
    print("variances over the exact ones:", spread.value / exact_var)
    print("their standard errors over the exact ones:", spread.standard_error / exact_var)
    return spread


spread = compare_line(far)
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert far.std()[0] > exact_std[0]
assert spread.standard_error[0] > exact_var[0]
```

The slope's standard deviation is too large, and the standard error of its variance
exceeds the variance itself. The replicates' estimates of the slope's variance, over the
exact one:

```{code-cell} python
ratios = np.sort(spread.replicates[:, 0] / exact_var[0])
print(f"smallest {ratios[0]:.3f}, second largest {ratios[-2]:.3f}, largest {ratios[-1]:.1f}")
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert ratios[-1] > 10 * ratios[-2]
```

One replicate causes it, as [Initial distributions](../guide/targets.md#choosing-one)
explains for starts this far. A Gaussian centered on an optimizer's fit, with its inverse
Hessian scaled by 100, is ten times wider than the posterior:

```{code-cell} python
fit = optimize.minimize(lambda theta: -log_prob(theta, x, y, yerr), [0.0, 0.0])
widened = telescopic.sample(
    log_prob,
    telescopic.initial.Normal(fit.x, 100 * fit.hess_inv),
    args=(x, y, yerr),
    vectorize=True,
    names=["slope", "intercept"],
    rng=42,
)
compare_line(widened)
widened.summary()
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert not telescopic.diagnostics.issues(widened)
assert widened.diagnostics()["variance_share"] < 0.5
assert np.all(np.abs(widened.std() / exact_std - 1) < 0.05)
widened_spread = widened.estimate(lambda theta: (theta - exact_mean) ** 2)
assert np.all(np.abs(widened_spread.value - exact_var) < 2 * widened_spread.standard_error)
assert np.all(widened_spread.standard_error < 0.05 * exact_var)
```

The run emits no warning, no replicate dominates, and the standard deviations and
variances agree with the exact ones, with small error bars.

Each run above printed its warnings as it ended, or, for a slow pair, as it happened,
except the runs of `compare_seeds`, which counted them.
[Diagnostics and warnings](../guide/diagnostics.md) shows how to read them from a result
and how to turn them into errors.

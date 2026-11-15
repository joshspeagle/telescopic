---
file_format: mystnb
kernelspec:
  name: python3
---

# Radial-velocity orbit of an exoplanet

A planet on an eccentric orbit pulls its star back and forth along the line of sight, and
the star's radial velocity traces a Keplerian curve. This page fits that curve, with
seven parameters, to simulated velocities. It follows the steps of
[Quickstart](../start/quickstart.md): a log-density written as for emcee, a start from
an optimizer's fit, a run, and the reading of its summary and warnings.

```{code-cell} python
:tags: [remove-cell]

from myst_nb import glue
```

## The data

A star with one planet has the radial velocity

$$v(t) = \gamma + K\,[\cos(\nu(t) + \omega) + e\cos\omega],$$

where $K$ is the semi-amplitude, $e$ the eccentricity, $\omega$ the argument of
periastron and $\gamma$ the systemic velocity. The true anomaly $\nu$ follows from the
mean anomaly $M = 2\pi t/P + M_0$, with period $P$ and phase $M_0$, through Kepler's
equation $M = E - e\sin E$ for the eccentric anomaly $E$. The function below solves
Kepler's equation by Newton's method for a batch of parameter vectors at once:

```{code-cell} python
import numpy as np
from scipy import differentiate, optimize

import telescopic

names = ["P", "K", "e", "omega", "M0", "gamma", "s"]


def velocity(theta, t):
    # theta has shape (n, 7) or (7,); the result, shape (n, len(t)) or (len(t),).
    P, K, e, omega, M0, gamma = (theta[..., i, None] for i in range(6))
    M = 2 * np.pi * t / P + M0
    E = M + 0.85 * e * np.sign(np.sin(M))  # Danby's start, from which Newton converges
    step = np.inf
    while np.max(np.abs(step), initial=0) > 1e-12:
        step = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= step
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return gamma + K * (np.cos(nu + omega) + e * np.cos(omega))
```

Forty velocities over 300 days, each with its own measurement error between 1.5 and
3.5 m/s, and an extra scatter $s$ of 2 m/s that the errors do not include, as stellar
activity adds:

```{code-cell} python
rng = np.random.default_rng(3)
n = 40
t = np.sort(rng.uniform(0, 300, n))
sigma = rng.uniform(1.5, 3.5, n)
truth = np.array([37.5, 11.0, 0.3, 2.2, 3.4, -4.0, 2.0])
v = velocity(truth, t) + np.hypot(sigma, truth[6]) * rng.normal(size=n)
```

## The log-density

Each velocity is Gaussian about the curve with variance $\sigma_i^2 + s^2$. The prior is
uniform on a box that enforces the constraints: $1 < P < 1000$ days, $K > 0$,
$0 \le e < 1$, $\omega$ and $M_0$ in $[0, 2\pi)$, and $s > 0$, with $\gamma$
unconstrained. Outside the box the log-density is `-inf`, which every Metropolis test
rejects ([Special values](../guide/log-density.md#special-values)). The function takes a
batch of points of shape `(n, 7)`, so the run passes `vectorize=True`
([The log-density](../guide/log-density.md#the-log-density)):

```{code-cell} python
lower = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -np.inf, 0.0])
upper = np.array([1000.0, np.inf, 1.0, 2 * np.pi, 2 * np.pi, np.inf, np.inf])


def log_prob(theta, t, v, sigma):
    inside = np.all((theta >= lower) & (theta < upper), axis=1) & (theta[:, 0] > 1)
    out = np.full(len(theta), -np.inf)
    point = theta[inside]
    variance = sigma**2 + point[:, 6, None] ** 2
    residual = v - velocity(point, t)
    out[inside] = -0.5 * np.sum(residual**2 / variance + np.log(variance), axis=1)
    return out
```

## Finding the period

An optimizer needs a starting period. A periodogram fits a sinusoid and a constant by
weighted least squares at each period of a grid, and the best period has the smallest
$\chi^2$:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      The chi-squared of a sinusoid fit against the period on a logarithmic axis
      from 2 to 300 days, with one deep minimum near 37 days and many shallow ones.
  figure:
    caption: >-
      The periodogram: the $\chi^2$ of a sinusoid fit at each period. The deep
      minimum is near the true period of 37.5 days; the shallow ones are other
      local minima.
---
import matplotlib.pyplot as plt

periods = np.geomspace(2, 300, 20000)
chi2 = np.empty(len(periods))
for i, period in enumerate(periods):
    phase = 2 * np.pi * t / period
    design = np.column_stack([np.cos(phase), np.sin(phase), np.ones(n)]) / sigma[:, None]
    coefficients = np.linalg.lstsq(design, v / sigma)[0]
    chi2[i] = np.sum((v / sigma - design @ coefficients) ** 2)
best_period = periods[np.argmin(chi2)]

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(periods, chi2, linewidth=1)
ax.set(xscale="log", xlabel="period $P$ (days)", ylabel=r"$\chi^2$");
```

```{code-cell} python
:tags: [remove-cell]

from scipy import signal

# The claims of the next paragraph: the deepest minimum is near the true period, and
# the next-deepest local minimum is far shallower.
minima = signal.argrelmin(chi2, order=5)[0]
depths = np.sort(chi2[minima])
assert abs(best_period - truth[0]) < 1
glue("best_period", float(best_period), display=False)
glue("chi2_gap", float(depths[1] - depths[0]), display=False)
assert depths[1] - depths[0] > 100
```

The deepest minimum is at {glue:text}`best_period:.1f` days, and every other local
minimum has a $\chi^2$ larger by at least {glue:text}`chi2_gap:.0f`. With sparser data,
the period's posterior can have several modes of comparable height, at aliases of the
true period. The {term}`pairs <pair>` of chains find only the modes that their initial
distribution covers, so the initial distribution must cover every mode of the period
that matters
([Choose an initial distribution](../guide/initial-distribution.md#choosing-one)).

## A start from an optimizer's fit

Nelder-Mead from the best period, and from a few phases and arguments of periastron,
finds the maximum of the log-density, and the Hessian there gives its error bars.
{func}`scipy.differentiate.hessian` evaluates the log-density at a batch of points of
shape `(7, ...)`, which `batch` reshapes for `log_prob`:

```{code-cell} python
def negative(theta):
    return -log_prob(theta[None], t, v, sigma)[0]


starts = [
    [best_period, np.ptp(v) / 2, 0.1, omega, M0, np.mean(v), 1.0]
    for omega in (1.0, 3.0, 5.0)
    for M0 in (1.0, 3.0, 5.0)
]
fits = [
    optimize.minimize(negative, start, method="Nelder-Mead", options={"maxfev": 20000})
    for start in starts
]
fit = min(fits, key=lambda result: result.fun)


def batch(x):
    points = np.moveaxis(x, 0, -1)
    return log_prob(points.reshape(-1, 7), t, v, sigma).reshape(points.shape[:-1])


hessian = differentiate.hessian(batch, fit.x, initial_step=1e-2).ddf
cov = np.linalg.inv(-(hessian + hessian.T) / 2)
dict(zip(names, np.sqrt(np.diag(cov)).round(3).tolist()))
```

The initial distribution is a Gaussian centered on the fit and wider than its error
bars, cut to the box: a callable that redraws until the point is inside, so that every
chain starts where the density is positive
([Choosing one](../guide/initial-distribution.md#choosing-one)). The guide asks for a start a
few to ten times wider than the target. The {term}`meeting times <meeting time>` of 100
pilot pairs with {term}`lag` 1, capped at 2,000 iterations, compare a start 3 times and
10 times wider than the error bars:

```{code-cell} python
def truncated(width):
    def init(rng):
        while True:
            point = rng.multivariate_normal(fit.x, width**2 * cov)
            if np.isfinite(log_prob(point[None], t, v, sigma)[0]):
                return point

    return init


pilot = {}
for width in (3, 10):
    sampler = telescopic.CoupledSampler(
        log_prob, truncated(width), args=(t, v, sigma), vectorize=True, rng=4
    )
    pilot[width] = sampler.meeting_times(100, max_iter=2000)
    print(
        f"{width:2} times wider: median {np.median(pilot[width]):5.0f}, "
        f"unmet at 2,000: {np.sum(np.isinf(pilot[width]))}"
    )
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert np.all(np.isfinite(pilot[3]))
assert np.sum(np.isinf(pilot[10])) >= 2
glue("unmet_10", int(np.sum(np.isinf(pilot[10]))), display=False)
```

From the start 10 times wider, {glue:text}`unmet_10:d` of the 100 pairs had not met
after 2,000 iterations, more unmet pairs than the automatic choice of the
{term}`burn-in` `k` allows
([The automatic pipeline](../guide/settings.md#the-automatic-pipeline)); from the start 3
times wider, every pair met. The run starts 3 times wider.

## The run

A run of 500 {term}`replicates <replicate>`, each of which stores every tenth state
(`thin=10`), which keeps the memory of the {term}`atoms <atom>` small; thinning does not
shorten a run ([Memory and thinning](../guide/settings.md#memory-and-thinning)):

```{code-cell} python
result = telescopic.sample(
    log_prob,
    truncated(3),
    500,
    args=(t, v, sigma),
    vectorize=True,
    thin=10,
    names=names,
    rng=5,
)
result.summary()
```

The next cell sets the table's means beside the values that drew the data. A posterior mean need not
equal the true value: it differs from it by about a posterior standard deviation, the
`std` column of the table, which is what the data leave uncertain:

```{code-cell} python
mean = result.mean()
std = result.std()
print("        true  estimate  std. error  posterior std")
for name, true, value, error, spread in zip(
    names, truth, mean.value, mean.standard_error, std
):
    print(f"{name:5}  {true:6.2f}  {value:8.3f}  {error:10.4f}  {spread:13.3f}")
```

```{code-cell} python
:tags: [remove-cell]

# The claim of the previous paragraph.
assert np.all(np.abs(mean.value - truth) < 3 * std)
```

## The orbit and two marginals

The posterior mean of the velocity at each time of a grid is an expectation like any
other, estimated from the same run with {meth}`~telescopic.Result.estimate`, with its
error bars. Below, it is drawn against the data and against the curve that drew them:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Forty velocity measurements with error bars against time over 300 days,
      with the estimated posterior mean velocity curve passing through them and the
      true curve dashed close to it.
  figure:
    caption: >-
      The simulated velocities, with error bars of their measurement errors, the
      estimated posterior mean of the velocity curve with a band of plus or minus
      two standard errors, and the curve that drew the data, dashed.
---
grid = np.linspace(0, 300, 300)
curve = result.estimate(lambda theta: velocity(theta, grid))

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.errorbar(t, v, sigma, fmt="o", color="C1", markersize=4, label="data")
ax.fill_between(
    grid,
    curve.value - 2 * curve.standard_error,
    curve.value + 2 * curve.standard_error,
    color="C0",
    alpha=0.4,
)
ax.plot(grid, curve.value, color="C0", linewidth=1.2, label="posterior mean")
ax.plot(grid, velocity(truth, grid), color="gray", linestyle="--", linewidth=1, label="true")
ax.set(xlabel="time $t$ (days)", ylabel="radial velocity (m/s)")
ax.legend(loc="upper right", ncols=3);
```

{meth}`~telescopic.Result.plot_marginals` draws the histograms of the
{term}`signed measure` of the period
and the eccentricity, with boxes of plus or minus two standard errors on each bin
([Plot a result](../guide/plots.md)), here with the true values dashed:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Two histograms, of the period and of the eccentricity, each bin topped by a
      small box, with a dashed vertical line at the true value inside each.
  figure:
    caption: >-
      The signed histograms of the period and the eccentricity, with boxes of plus
      or minus two standard errors on each bin, and the true values dashed.
---
axes = result.plot_marginals(dims=[0, 2], bins=30)
for ax, true in zip(axes, truth[[0, 2]]):
    ax.axvline(true, color="gray", linestyle="--")
```

## What the warnings say

The summary ends with one warning, that one replicate holds most of the variance of an
estimate. `result.diagnostics()` names the replicate and the estimate, and the meeting
times show what sets that replicate apart:

```{code-cell} python
from telescopic import diagnostics

checks = result.diagnostics()
named = checks["variance_share_replicate"]
print(f"share of replicate {named}: {checks['variance_share']:.2f}, in {checks['variance_share_of']}")
print(f"its meeting time: {result.meeting_times[named]:.0f}, k + lag: {result.k + result.lag}")
print(f"latest meeting time of the others: {np.max(np.delete(result.meeting_times, named)):.0f}")
print(f"largest share of one replicate in a mean: {diagnostics.variance_share(mean.replicates).max():.2f}")
print(f"pairs meeting after k + lag: {checks['late_fraction']:.3f}")
print(f"tail index of the meeting times: {checks['tail_index']:.2f}")
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraphs.
tau_named = result.meeting_times[named]
mean_share = diagnostics.variance_share(mean.replicates).max()
assert tau_named == result.meeting_times.max() and tau_named > result.k + result.lag
assert 0.5 < checks["variance_share"] and checks["variance_share_of"] == "the variance of M0"
assert 0.25 < mean_share < 0.5
assert checks["censored"] == 0 and checks["late_fraction"] < 0.25 and checks["tail_index"] > 2
glue("named", named, display=False)
glue("share", float(checks["variance_share"]), display=False)
glue("tau_named", float(tau_named), display=False)
glue("k_lag", result.k + result.lag, display=False)
glue("mean_share", float(mean_share), display=False)
```

Replicate {glue:text}`named:d` holds {glue:text}`share:.2f` of the sample variance of
the estimate of the posterior variance of $M_0$. Its pair met last of the 500, at
$\tau =$ {glue:text}`tau_named:.0f`, after $k + L =$ {glue:text}`k_lag:d`, so its
estimate carries the longest bias correction of the run: it is a {term}`late pair`.
The error bar of that variance rests on one replicate and is not reliable. The means do
not cross the warning's threshold, but come near it: one replicate holds
{glue:text}`mean_share:.2f` of the variance of the estimate of some mean. The other
warnings do not fire: no pair was censored, and neither the fraction of
late pairs nor the {term}`tail index` of the meeting times crosses
its threshold.
[One replicate dominating the variance](../reference/warnings.md#one-replicate-dominating-the-variance-variance)
gives the threshold, the warning's meaning and its remedy: check the initial distribution, and look for a
mode that the other replicates missed. For error bars that rest on many replicates,
plan a second run with more of them, its number fixed before it runs
([How many replicates](../guide/settings.md#how-many-replicates)).

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

from IPython.display import Markdown


def pilot_text(width):
    unmet = int(np.sum(np.isinf(pilot[width])))
    try:
        telescopic.tuning.choose_k(pilot[width])
        verdict = "and `k` can be chosen from them"
    except ValueError:
        verdict = "too many for `k` to be chosen"
    return f"from the start {width} times wider, {unmet} of the 100 pairs had not met, {verdict}"


glue(
    "pilot_text",
    Markdown(f"After 2,000 iterations, {pilot_text(10)}; {pilot_text(3)}."),
    display=False,
)
```

{glue:md}`pilot_text` The automatic choice of the {term}`burn-in` `k` allows only a few
unmet pairs ([The automatic pipeline](../guide/settings.md#the-automatic-pipeline)), and
from a start wider than needed, the pairs that start far out take longest to meet. The
run starts 3 times wider.

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

The next cell sets the table's means beside the values that drew the data. A posterior
mean need not equal the true value: it differs from it by about the error bars that the
data leave, here those of the optimizer's fit:

```{code-cell} python
mean = result.mean()
fit_error = np.sqrt(np.diag(cov))
print("        true  estimate  std. error  fit's error bar")
for name, true, value, error, spread in zip(
    names, truth, mean.value, mean.standard_error, fit_error
):
    print(f"{name:5}  {true:6.2f}  {value:8.3f}  {error:10.4f}  {spread:15.3f}")
```

```{code-cell} python
:tags: [remove-cell]

# The claim of the previous paragraph, with a margin.
assert np.all(np.abs(mean.value - truth) < 3 * fit_error)
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

```{code-cell} python
:tags: [remove-cell]

from run_text import warnings_text

glue("warnings_text", Markdown(warnings_text(result)), display=False)
```

{glue:md}`warnings_text`

A share near 1 means that the error bar of that estimate rests on one replicate and is
not reliable. A pair that meets far later than the others, as from a start far out or
after an excursion to a region that the other pairs did not visit, gives such a share:
an orbit's posterior can have a tail toward high eccentricity, where a narrow spike of
the curve fits a few points.
[Warnings and errors](../reference/warnings.md#warnings) gives each warning's threshold,
meaning and remedy. For error bars that rest on many replicates, plan a second run with
more of them, its number fixed before it runs
([How many replicates](../guide/settings.md#how-many-replicates)).

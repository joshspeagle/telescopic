---
file_format: mystnb
kernelspec:
  name: python3
---

# How it works

[Getting started](../getting-started.md) describes a run in words. This page runs one by
hand, on a standard normal target in one dimension with chains started far from it: the
paths of a few pairs, one replicate's estimate taken apart, and the bias that the
correction removes on average. [Theory](../theory.md) derives each step.

```{code-cell} python
:tags: [remove-cell]

from myst_nb import glue
```

## Pairs of chains

The target $\pi$ is $\mathcal N(0, 1)$, and the chains start from $\mathcal N(5, 1)$,
five standard deviations away. The kernel is random-walk Metropolis with proposals
$x' \sim \mathcal N(x, 2.38^2)$, its default for `cov=[[1.0]]`, coupled by the
reflection-maximal coupling ([Kernels and couplings](../guide/kernels.md)).
{func}`telescopic.engine.run` runs replicates with given settings, and with
`keep_paths=True` it keeps the paths of both chains. Here the burn-in $k$ is 5, the run
length $m$ is 20 and the lag $L$ is 1; the output is the meeting times $\tau$ of sixteen
pairs:

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

import telescopic
from telescopic import engine, plotting


def log_prob(x):
    return -0.5 * x[:, 0] ** 2


target = telescopic.Target(log_prob, vectorize=True)
kernel = telescopic.kernels.RandomWalk(cov=[[1.0]])
init = telescopic.initial.Normal(mean=[5.0], cov=[1.0])
k, m = 5, 20

replicates = engine.run(target, kernel, init, range(16), k=k, m=m, lag=1, rng=1, keep_paths=True)
replicates.meeting_time
```

{func}`telescopic.plotting.paths` draws the first pairs against time: $X$ solid and $Y$
dashed, shifted right by the lag, so that a pair meets where its dashed line joins its
solid one, marked with a dot. After a meeting only $X$ runs, until $m$. The MCMC average
uses the shaded times, $k$ to $m$:

```{code-cell} python
n = 4
ax = plotting.paths(replicates, n=n)
meeting = replicates.meeting_time[:n].astype(int)
position = np.array([replicates.paths[i][0][meeting[i], 0] for i in range(n)])
ax.plot(meeting, position, "o", color="black")
ax.axvspan(k, m, color="grey", alpha=0.15)
ax.set(xlabel="$t$", ylabel="$X_t$ and $Y_{t-1}$");
```

```{code-cell} python
:tags: [remove-cell]

glue("early", int(np.sum(meeting <= 3)), display=False)
glue("far", int(np.sum(np.abs(position) > 3)), display=False)
```

Of the four pairs, {glue:text}`early:d` meet within three steps, and {glue:text}`far:d`
meet more than three standard deviations from the target's mean: a meeting does not mean
that the chains have reached the target.

## One replicate's estimate

Each replicate's estimate is a signed measure: atoms with weights, whose weighted sum of
a function $h$ estimates its expectation $\pi(h)$ under the target
([Theory](../theory.md#signed-measures)). {class}`telescopic.Result` holds the measures of
all replicates in its `measure`, a {class}`telescopic.SignedMeasure` whose `replicate`
array gives each atom's replicate. The first pair that met after $k + L$, here $k + 1$,
has a bias correction that is not zero:

```{code-cell} python
result = telescopic.Result(replicates)
r = int(np.argmax(replicates.meeting_time > k + 1))
mine = result.measure.replicate == r
atoms, weights = result.measure.atoms[mine, 0], result.measure.weights[mine]
print(
    f"replicate {r} met at tau = {replicates.meeting_time[r]:g}: {len(atoms)} atoms, "
    f"{np.sum(weights < 0)} with negative weights; the weights sum to {weights.sum():.12f}"
)
```

The states $X_k, \dots, X_m$ weigh $1/(m - k + 1)$ each. For $t$ from $k + 1$ to
$\tau - 1$, the correction adds weight at $X_t$ and removes the same weight at $Y_{t-1}$,
so the weights sum to 1 and some are negative. The weights of this replicate, summed over
equal atoms:

```{code-cell} python
points, which = np.unique(atoms, return_inverse=True)
fig, ax = plt.subplots(figsize=(5, 3))
ax.stem(points, np.bincount(which, weights))
ax.set(xlabel="atom", ylabel="weight");
```

Its estimate of the mean, with $h(x) = x$, is the MCMC average of $X_k, \dots, X_m$ plus
the bias correction, whose weights are $\min(1, (t - k)/(m - k + 1))$ for $L = 1$
([Theory](../theory.md#the-time-averaged-estimator)):

```{code-cell} python
x_path, y_path = replicates.paths[r]
tau = int(replicates.meeting_time[r])
mcmc = x_path[k : m + 1, 0].mean()
t = np.arange(k + 1, tau)
weight = np.minimum(1, (t - k) / (m - k + 1))
correction = np.sum(weight * (x_path[t, 0] - y_path[t - 1, 0]))
estimate = result.measure.per_replicate(lambda x: x[:, 0])[r]
print(f"MCMC average {mcmc:.6f} + bias correction {correction:.6f} = {mcmc + correction:.6f}")
print(f"signed measure: {estimate:.6f}")
```

One replicate's correction can move its estimate toward the target's mean, 0, or away
from it: only the average over replicates is unbiased. For a pair that meets by $k + L$
the correction is zero, and the estimate is the MCMC average alone.

## Unbiased on average

From a start five standard deviations away, a burn-in of 5 steps is too short, and the
MCMC average is biased. The bias correction removes this bias on average. 4000 replicates
for each of two lags, 1 and $k$, estimate the mean of the target, 0, with `z` the estimate
in units of its error bar:

```{code-cell} python
print(f"{'lag':>3}  {'estimator':12}{'mean':>8}{'error bar':>11}{'z':>7}")
rows = {}
for lag, seed in ((1, 2), (k, 3)):
    runs = engine.run(
        target, kernel, init, range(4000), k=k, m=m, lag=lag, rng=seed, keep_paths=True
    )
    mcmc = np.array([runs.paths[i][0][k : m + 1, 0].mean() for i in runs.index])
    full = telescopic.Result(runs).estimate(lambda x: x[:, 0]).replicates
    for name, values in (("MCMC average", mcmc), ("full", full)):
        mean, error = values.mean(), values.std(ddof=1) / np.sqrt(len(values))
        rows[lag, name] = mean / error, error
        print(f"{lag:3}  {name:12}{mean:8.4f}{error:11.4f}{mean / error:7.2f}")
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert all(abs(rows[lag, "MCMC average"][0]) > 10 for lag in (1, k))
assert all(abs(rows[lag, "full"][0]) < 2 for lag in (1, k))
assert rows[k, "full"][1] < rows[1, "full"][1]
```

The MCMC average is many error bars from 0 with either lag, since the lag does not change
the law of $X$. The full estimator is within two error bars of 0 with both lags, and its
error bar is smaller with $L = k$.

## Next

[Choosing k, m and the lag](choosing-settings.md) shows how {func}`telescopic.sample`
chooses $k$, $m$ and the lag, and how the lag changes the bound on the distance to
stationarity.

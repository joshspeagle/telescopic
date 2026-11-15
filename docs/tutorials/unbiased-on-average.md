---
file_format: mystnb
kernelspec:
  name: python3
---

# Unbiased on average

This page continues [How it works](how-it-works.md), with its target $\mathcal N(0, 1)$,
its chains started from $\mathcal N(5, 1)$, its random-walk kernel and its settings
$k = 5$ and $m = 20$.

```{code-cell} python
:tags: [remove-cell]

import matplotlib.pyplot as plt
import numpy as np

import telescopic
from telescopic import engine


def log_prob(x):
    return -0.5 * x[:, 0] ** 2


target = telescopic.Target(log_prob, vectorize=True)
kernel = telescopic.kernels.RandomWalk(cov=[[1.0]])
init = telescopic.initial.Normal(mean=[5.0], cov=[1.0])
k, m = 5, 20
```

From a start five standard deviations away, a {term}`burn-in` of 5 steps is too short, and the
MCMC average is biased. The bias correction removes this bias on average. 4000 {term}`replicates <replicate>`
for each of two {term}`lags <lag>`, 1 and $k$, estimate the mean of the target, 0, by the
MCMC average of $X_k, \dots, X_m$ alone and by the full estimator, which adds the bias
correction:

```{code-cell} python
estimates = {}
for lag, seed in ((1, 2), (k, 3)):
    runs = engine.run(
        target, kernel, init, range(4000), k=k, m=m, lag=lag, rng=seed, keep_paths=True
    )
    mcmc = np.array([runs.paths[i][0][k : m + 1, 0].mean() for i in runs.index])
    full = telescopic.Result(runs).estimate(lambda x: x[:, 0]).replicates
    estimates[lag, "MCMC average"], estimates[lag, "full"] = mcmc, full
errors = {key: v.std(ddof=1) / np.sqrt(len(v)) for key, v in estimates.items()}
```

The mean of each set of replicates, with two standard errors, and the replicates' own
spread for $L = k$:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Left: four dots with horizontal error bars, one row for each lag and
      estimator, beside a vertical line at 0; the two MCMC averages lie far to the
      right of it and the two full estimates cross it. Right: two overlaid
      histograms of the replicates' estimates for lag k, both peaked near 0: a
      narrow one of the MCMC average and a wider one of the full estimator with a
      long tail to the left.
  figure:
    caption: >-
      Left: the mean of 4,000 replicates of the MCMC average and of the full
      estimator, for $L = 1$ and $L = k$, with two standard errors, against the
      target's mean, 0. Right: the histograms of the 4,000 replicates' estimates for
      $L = k$, between $-8$ and 6. The MCMC average's replicates are narrower, but
      their mean is off 0; the full estimator's spread wider, with a long tail to
      the left, and their mean is 0 within its error bar.
---
fig, (left, right) = plt.subplots(1, 2, figsize=(7, 2.8), width_ratios=(1.1, 1))
for row, ((lag, name), values) in enumerate(estimates.items()):
    color = "C0" if name == "MCMC average" else "C1"
    left.errorbar(
        values.mean(), row, xerr=2 * errors[lag, name], fmt="o", color=color, capsize=3
    )
left.axvline(0, color="gray", linewidth=0.8)
left.set_yticks(range(4), [f"{name}, $L = {lag}$" for lag, name in estimates])
left.invert_yaxis()
left.set(xlabel="mean of the replicates")
bins = np.linspace(-8, 6, 71)
for name, color in (("MCMC average", "C0"), ("full", "C1")):
    right.hist(estimates[k, name], bins=bins, color=color, alpha=0.5, label=name)
right.axvline(0, color="gray", linewidth=0.8)
right.legend()
right.set(xlabel=f"one replicate's estimate, $L = {k}$", yticks=[]);
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
z = {key: v.mean() / errors[key] for key, v in estimates.items()}
assert all(abs(z[lag, "MCMC average"]) > 10 for lag in (1, k))
assert all(abs(z[lag, "full"]) < 2 for lag in (1, k))
assert errors[k, "full"] < errors[1, "full"]
assert estimates[k, "full"].std() > estimates[k, "MCMC average"].std()
# The histograms' range holds nearly every replicate.
assert all(
    np.mean((-8 < estimates[k, name]) & (estimates[k, name] < 6)) > 0.99
    for name in ("MCMC average", "full")
)
```

The MCMC average is many error bars from 0 with either lag, since the lag does not change
the law of $X$. The full estimator is within two error bars of 0 with both lags, and its
error bar is smaller with $L = k$. Its replicates spread more widely than those of the
MCMC average: the correction trades a bias for variance, which the replicates average
away.

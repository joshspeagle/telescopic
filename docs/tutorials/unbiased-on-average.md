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
for each of two {term}`lags <lag>`, 1 and $k$, estimate the mean of the target, 0, with `z` the estimate
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

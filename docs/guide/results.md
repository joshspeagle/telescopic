---
file_format: mystnb
kernelspec:
  name: python3
---

# Results

{func}`telescopic.sample` and {meth}`telescopic.CoupledSampler.run` return a
{class}`telescopic.Result`. It stores each replicate's signed measure, meeting time and
cost, and the settings of the run, so every estimate is computed after the run.

The examples use a correlated Gaussian in two dimensions, whose moments are known
({func}`telescopic.examples.gaussian`), and 1000 replicates, so that a few pairs meet
after `k + lag` and carry bias corrections:

```{code-cell} python
import numpy as np
from scipy import stats

import telescopic
from telescopic import examples

gaussian = examples.gaussian(2)  # N(0, [[1, 0.9], [0.9, 1]]), started from N(0, 4I)
result = telescopic.sample(
    gaussian.log_prob, gaussian.initial, 1000, vectorize=True, rng=11
)
result.n_replicates, result.k, result.m, result.lag
```

## What a result holds

| Attribute | Content |
|---|---|
| `n_replicates`, `n_censored` | $R$, the number of replicates whose pairs met, and the number of censored pairs |
| `k`, `m`, `lag`, `thin`, `max_iter` | the settings of the run ([Settings](settings.md)) |
| `names`, `ndim` | the parameter names and the dimension $d$ |
| `measure` | the replicates' signed measures |
| `index`, `meeting_times`, `costs` | for every replicate, censored ones included: its index, its meeting time (`inf` if censored) and its kernel applications |
| `n_log_prob`, `n_grad`, `wall_time` | the evaluations of the log-density and of its gradient, and the seconds, of the runs |
| `kernel`, `initial` | the frozen kernel and the initial distribution |
| `tuning`, `pilot` | the report of the tuning chains, and the meeting times of the lag-1 pilot pairs that chose `k`; `None` when not run |
| `seed` | the root `numpy.random.SeedSequence`, which reproduces the run |
| `interrupted` | whether a requested replicate never ran ([Parallel runs](parallel.md#interrupting-a-run)) |

Censored pairs are left out of the estimates, of $R$ and of the signed measure
([Diagnostics and warnings](diagnostics.md)).

## Expectations

{meth}`telescopic.Result.estimate` estimates $\pi(h)$ for a function `h` that maps points
of shape `(n, d)` to values of shape `(n,)` or `(n, p)`; any other shape raises
`ValueError`. It returns an {class}`telescopic.Estimate`: the mean of the replicates'
estimates (`value`), its `standard_error`, a normal `interval` at `level` (0.95 by
default) and the replicates' estimates (`replicates`). Here $E[x_0 x_1]$ is the
covariance, 0.9:

```{code-cell} python
product = result.estimate(lambda x: x[:, 0] * x[:, 1])
product, product.interval
```

Values of shape `(n, p)` give `p` estimates at once, and a boolean function gives
probabilities. The exact $P(x_i > 1)$ is $1 - \Phi(1)$:

```{code-cell} python
result.estimate(lambda x: x > 1), stats.norm.sf(1)
```

{meth}`telescopic.Result.mean` is `estimate(lambda x: x)`. The value is unbiased, and the
interval is valid as $R$ grows when the variance is finite
([Theory](../theory.md#guarantees-and-their-limits)). Exact unbiasedness needs a number
of replicates fixed before any result is seen
([Pitfalls](pitfalls.md#choosing-the-number-of-replicates-after-looking)).

## Covariances and variances

{meth}`telescopic.Result.cov` and {meth}`telescopic.Result.var` are unbiased estimates of
the covariance matrix and of the variances
([Theory](../theory.md#estimates-from-many-replicates)). A variance estimate can be
negative, and the covariance matrix need not be positive semi-definite
([Pitfalls](pitfalls.md#an-initial-distribution-far-wider-than-the-target)):

```{code-cell} python
result.cov(), gaussian.cov
```

{meth}`telescopic.Result.std` returns $\sqrt{\max(0, \widehat{\mathrm{Var}})}$ for each
coordinate, which is consistent but not unbiased:

```{code-cell} python
result.std()
```

## Quantiles and credible intervals

{meth}`telescopic.Result.quantile` returns quantiles of each coordinate under the pooled
signed measures of the replicates, shape `(d,)` for one level and `(len(q), d)` for
several. {meth}`telescopic.Result.interval` returns the central interval that holds a
given `mass` of each marginal (0.68 by default): the quantiles at $(1 - \mathrm{mass})/2$
and $(1 + \mathrm{mass})/2$, shape `(2, d)`. Both are consistent as $R$ grows, not
unbiased, and come without standard errors; the method is on the
[theory page](../theory.md#estimates-from-many-replicates).

```{code-cell} python
result.quantile([0.05, 0.5, 0.95]), stats.norm.ppf([0.05, 0.5, 0.95])
```

```{code-cell} python
result.interval(0.9)
```

## Histograms

{meth}`telescopic.Result.histogram` returns the mass of each bin, its standard error and
the edges of the bins. The mass of a half-open bin $[a, b)$ is the average over
replicates of their signed mass in it. It is unbiased, and it can be negative where few
atoms fall and the negative weights of the bias corrections outweigh the others. By
default the bins span the atoms, so the masses sum to 1:

```{code-cell} python
masses, errors, edges = result.histogram(0, bins=10)
masses, errors, masses.sum()
```

A sequence of coordinates gives a joint histogram of shape `(bins,) * len(dims)`, whose
entry `[i, j]` is bin `i` of the first coordinate and bin `j` of the second. `range` sets
the outer edges, a pair per coordinate; atoms outside them fall in no bin:

```{code-cell} python
joint, joint_errors, (edges_0, edges_1) = result.histogram(
    [0, 1], bins=8, range=[(-3, 3), (-3, 3)]
)
joint.shape, joint.sum()
```

The masses are not densities: divide them by the widths of the bins, as
`plot_marginals` does.

## Signed measures

`result.measure` is a {class}`telescopic.SignedMeasure`: the `atoms`, of shape `(N, d)`,
their `weights`, and for each atom the position, from 0 to $R - 1$, of the `replicate` it
belongs to. Each replicate's weights sum to 1. Its bias corrections add pairs of atoms
with weights of opposite signs ([Theory](../theory.md#signed-measures)), so an estimate
can fall outside the range of $h$:

```{code-cell} python
measure = result.measure
negative = measure.weights < 0
{
    "atoms": measure.atoms.shape,
    "weights sum to 1": bool(np.allclose(np.bincount(measure.replicate, measure.weights), 1)),
    "negative weights": int(negative.sum()),
    "replicates with negative weights": len(np.unique(measure.replicate[negative])),
}
```

{meth}`telescopic.SignedMeasure.per_replicate` returns each replicate's estimate
$\hat\pi_r(h)$, shape `(R,)` or `(R, p)`, for statistics of your own; an `Estimate`
holds the same values in `replicates`:

```{code-cell} python
second = measure.per_replicate(lambda x: x[:, 0] ** 2)
second.shape, np.array_equal(second, result.estimate(lambda x: x[:, 0] ** 2).replicates)
```

Packages such as corner and ArviZ compute weighted quantiles that assume non-negative
weights, so the atoms and their weights should not be passed to them.

## The summary

{meth}`telescopic.Result.summary` returns a {class}`telescopic.Summary`: one row per
parameter, then the configuration, diagnostics and warnings of the run.
[Getting started](../getting-started.md#reading-the-summary) explains each entry.

```{code-cell} python
summary = result.summary()
summary
```

`summary.table` holds the columns as arrays, and `summary.footer` the rest as a
dictionary. {meth}`telescopic.Summary.to_dict` converts both to Python numbers, and
{meth}`telescopic.Summary.to_pandas` returns the table as a `pandas.DataFrame` indexed by
parameter name (it needs the `pandas` extra):

```{code-cell} python
summary.to_dict()["parameters"]["x0"]
```

{meth}`telescopic.Result.diagnostics` returns the diagnostics of the footer as a
dictionary, which [Diagnostics and warnings](diagnostics.md) explains.

## Merging runs

A sampler needs no merging: each call of `run` returns the result of all its replicates
so far. {meth}`telescopic.Result.merge` joins runs made apart, such as runs of
{func}`telescopic.engine.run` on separate ranges of replicate indices, on one machine or
several. The runs must have the same root seed, settings and dimension, and no replicate
index in common. Here the replicates 1000 to 1199 extend the run above:

```{code-cell} python
more = telescopic.engine.run(
    telescopic.Target(gaussian.log_prob, vectorize=True),
    result.kernel,
    result.initial,
    np.arange(1000, 1200),
    k=result.k,
    m=result.m,
    lag=result.lag,
    thin=result.thin,
    max_iter=result.max_iter,
    rng=result.seed,
)
merged = result.merge(telescopic.Result(more, names=result.names))
merged.n_replicates
```

The merged result is `interrupted` if a replicate requested in either run ran in neither.

## Saving and loading

{meth}`telescopic.Result.save` writes a result to a versioned `.npz` file, without
pickle, and {meth}`telescopic.Result.load` reads it back. Both append `.npz` to a name
that lacks it. The kernel and the initial distribution are saved as their descriptions,
`str(kernel)` and `str(initial)`, which a loaded result holds in their place; the paths
that the engine keeps for teaching are not saved. Loading a file whose schema is newer
than the installed telescopic's raises `ValueError`.

```{code-cell} python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as directory:
    result.save(Path(directory) / "gaussian")  # writes gaussian.npz
    loaded = telescopic.Result.load(Path(directory) / "gaussian")
loaded.kernel, np.array_equal(loaded.cov(), result.cov())
```

## Plots

The plots need the `plot` extra (matplotlib). {meth}`telescopic.Result.plot_marginals`
draws the signed histogram of each coordinate as a density, with boxes of plus or minus
two standard errors:

```{code-cell} python
result.plot_marginals();
```

{meth}`telescopic.Result.plot_corner` draws the marginals on its diagonal and the signed
joint density of each pair of coordinates below it, on a diverging colour map centred on
0:

```{code-cell} python
result.plot_corner();
```

{meth}`telescopic.Result.plot_meeting_times` and {meth}`telescopic.Result.plot_tv_bound`
are shown in [Diagnostics and warnings](diagnostics.md). Each method returns its
matplotlib axes, and all but `plot_corner` draw on axes given as `axes` or `ax`. The
functions of {mod}`telescopic.plotting` take the result as their first argument and do
the same; {func}`telescopic.plotting.paths` also draws the coupled paths of a few
replicates of an engine run with `keep_paths=True`
([How it works](../tutorials/how-it-works.md)).

## Unbiased or consistent

| Output | Methods | Property |
|---|---|---|
| expectations | `estimate`, `mean` | unbiased, with intervals valid as $R$ grows when the variance is finite |
| covariances and variances | `cov`, `var` | unbiased; a variance can be negative |
| standard deviations | `std` | consistent |
| histogram masses | `histogram` | unbiased, with standard errors; a bin can be negative |
| quantiles and central intervals | `quantile`, `interval` | consistent, without standard errors |

In the summary, `mean` is unbiased and `std` and the quantiles are consistent.
Unbiasedness holds for a number of replicates fixed in advance, when none is censored and
every requested replicate ran.

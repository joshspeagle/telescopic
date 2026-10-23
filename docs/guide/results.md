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
({func}`telescopic.examples.gaussian`), and 1000 replicates:

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

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert 0 < result.diagnostics()["late_fraction"] < 0.05
```

A few of the pairs met after `k + lag`, so a few replicates carry bias corrections.

## What a result holds

| Attribute | Content |
|---|---|
| `n_replicates`, `n_censored` | $R$, the number of replicates whose pairs met, and the number of pairs censored at `max_iter` ([Diagnostics and warnings](diagnostics.md#censored-pairs-censored)) |
| `k`, `m`, `lag`, `thin`, `max_iter` | the settings of the run ([Settings](settings.md)) |
| `names`, `ndim` | the parameter names and the dimension $d$ |
| `measure` | the replicates' signed measures |
| `index`, `meeting_times`, `costs` | for every replicate, censored ones included: its index, its meeting time (`inf` if censored) and its kernel applications |
| `n_log_prob`, `n_grad`, `wall_time` | the evaluations of the log-density and of its gradient, and the seconds, of the runs |
| `kernel`, `initial` | the frozen kernel and the initial distribution |
| `tuning`, `pilot` | the report of the tuning chains, and the meeting times of the lag-1 pilot pairs that chose `k`; `None` when not run |
| `seed` | the root `numpy.random.SeedSequence`, which reproduces the run |
| `interrupted` | whether a requested replicate never ran ([Parallel runs](parallel.md#interrupting-a-run)) |

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

{meth}`telescopic.Result.mean` is `estimate(lambda x: x)`. The value is unbiased for a
number of replicates fixed in advance
([How many replicates](settings.md#how-many-replicates)), and the interval is valid as
$R$ grows when the variance is finite
([Theory](../theory.md#guarantees-and-their-limits)).

## Covariances and variances

{meth}`telescopic.Result.cov` and {meth}`telescopic.Result.var` are unbiased estimates of
the covariance matrix and of the variances
([Theory](../theory.md#estimates-from-many-replicates)). A variance estimate can be
negative, and the covariance matrix need not be positive semi-definite; a start far
wider than the target can make them so
([Initial distributions](targets.md#choosing-one)):

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

`quantile` and `interval` take the coordinates of the target only. For a quantity
derived from them, apply {func}`telescopic.estimators.weighted_quantile` to its values at
the atoms, with each atom's weight divided by $R$. On a coordinate, this gives what
`quantile` gives:

```{code-cell} python
levels = [0.05, 0.5, 0.95]
weights = result.measure.weights / result.n_replicates
x0 = result.measure.atoms[:, 0]
np.array_equal(
    telescopic.estimators.weighted_quantile(x0, weights, levels),
    result.quantile(levels)[:, 0],
)
```

The sum $x_0 + x_1$ is Gaussian, with a variance equal to the sum of the entries of the
covariance matrix:

```{code-cell} python
total = result.measure.atoms.sum(axis=1)  # x0 + x1 at every atom
(
    telescopic.estimators.weighted_quantile(total, weights, levels),
    stats.norm.ppf(levels, scale=np.sqrt(gaussian.cov.sum())),
)
```

## Histograms

{meth}`telescopic.Result.histogram` returns the mass of each bin, its standard error and
the edges of the bins. The mass of a half-open bin $[a, b)$ is the average over
replicates of their signed mass in it. It is unbiased, and it can be negative where few
atoms fall and the negative weights of the bias corrections outweigh the others. By
default the bins span the atoms, so the masses sum to 1:

```{code-cell} python
masses, errors, edges = result.histogram(0, bins=10)
masses, errors, bool(np.isclose(masses.sum(), 1))
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

The atoms are not samples of the target, even when plotted without their weights: the
atoms of each Y chain carry negative weights. Packages such as corner and ArviZ compute
weighted quantiles that assume non-negative weights, so the atoms should not be passed
to them with their weights either.

## The summary

{meth}`telescopic.Result.summary` returns a {class}`telescopic.Summary`. It prints a table
with one row per parameter, then the settings and diagnostics of the run and the warnings
that apply:

```{code-cell} python
summary = result.summary()
summary
```

The columns of the table:

- `mean`: the average of the replicates' estimates of the parameter's mean, which is
  unbiased; `standard_error`: its Monte Carlo error, valid as the number of replicates
  grows, however short each run.
- `std`: $\sqrt{\max(0, \widehat{\mathrm{Var}})}$, from the unbiased estimate of the
  variance; consistent, not unbiased.
- `q16`, `q50` and `q84`: the 16%, 50% and 84% quantiles of the pooled signed measures;
  consistent, not unbiased.

The lines below it:

- $R$, the settings `k`, `m`, `lag` and `thin` ([Settings](settings.md)), and the kernel;
- the median, the 90% and 99% quantiles and the maximum of the meeting times of the
  pairs that met; from 500 replicates, their tail index
  ([`tail`](diagnostics.md#heavy-tailed-meeting-times-tail)); and the number of censored
  pairs ([`censored`](diagnostics.md#censored-pairs-censored));
- the mean cost of a replicate in kernel applications and in seconds, and the relative
  cost, the mean cost over $m - k + 1$ ([Settings](settings.md#cost));
- the fraction of pairs that met after `k + lag`, censored ones included, with a
  one-sided 95% upper bound: only these replicates carry a bias correction, and with
  `lag = k` the fraction is close to the bound on the total variation distance at `k`
  ([Distance to stationarity](diagnostics.md#distance-to-stationarity));
- the largest share of one replicate in the variance of the estimate of a coordinate's
  mean or second moment, and that estimate
  ([`variance`](diagnostics.md#one-replicate-dominating-the-variance-variance));
- for an interrupted run, that its estimates are consistent but not exactly unbiased
  ([Interrupting a run](parallel.md#interrupting-a-run));
- each warning that applies ([Warnings](diagnostics.md#warnings)).

`summary.table` holds the columns as arrays, and `summary.footer` the rest as a
dictionary. {meth}`telescopic.Summary.to_dict` converts both to Python numbers, and
{meth}`telescopic.Summary.to_pandas` returns the table as a `pandas.DataFrame` indexed by
parameter name (it needs the `pandas` extra):

```{code-cell} python
summary.to_dict()["parameters"]["x0"]
```

{meth}`telescopic.Result.diagnostics` returns the diagnostics of the footer as a
dictionary ([The diagnostics of a run](diagnostics.md#the-diagnostics-of-a-run)).

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
functions of {mod}`telescopic.plotting`, which `import telescopic` leaves out since it
needs matplotlib, take the result as their first argument and do the same;
{func}`telescopic.plotting.paths` also draws the coupled paths of a few replicates of an
engine run with `keep_paths=True` ([How it works](../tutorials/how-it-works.md)).

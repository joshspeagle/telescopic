---
file_format: mystnb
kernelspec:
  name: python3
---

# Estimate anything from a result

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
result
```

A result prints as one line of its settings: $R$, the dimension $d$, `k`, `m`, `lag`,
the number of censored pairs and whether the run was interrupted.

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert 0 < result.diagnostics()["late_fraction"] < 0.05
```

A few of the pairs met after `k + lag`, so a few replicates carry bias corrections.

## Expectations

{meth}`telescopic.Result.estimate` estimates $\pi(h)$ for a function `h` that maps
points of shape `(n, d)`, a read-only array, to values of shape `(n,)` or `(n, p)`; any
other shape raises `ValueError`. It returns an {class}`telescopic.Estimate`: the mean of
the replicates' estimates (`value`), its `standard_error`, a confidence `interval` at
`level` (0.95 by default), an array of shape `(2,)` or `(2, p)` with the lower ends
first, and the replicates' estimates (`replicates`). The interval is the value plus or
minus the standard error times the quantile of Student's t with $R - 1$
degrees of freedom ([Theory](../theory.md#guarantees-and-their-limits)). It needs at least 2 replicates; with 1,
`estimate` raises `ValueError`. Here $E[x_0 x_1]$ is the covariance, 0.9:

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
([Initial distributions](initial-distribution.md#choosing-one)):

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
several. {meth}`telescopic.Result.credible_interval` returns the central credible
interval that holds a given `mass` of each marginal (0.68 by default): the quantiles at
$(1 - \mathrm{mass})/2$ and $(1 + \mathrm{mass})/2$, shape `(2, d)`. It describes the
spread of the target, where the `interval` of an estimate is a confidence interval for
an expectation. Quantiles and credible intervals are consistent as $R$ grows, not
unbiased, and come without standard errors; the method is on the
[theory page](../theory.md#estimates-from-many-replicates).

```{code-cell} python
result.quantile([0.05, 0.5, 0.95]), stats.norm.ppf([0.05, 0.5, 0.95])
```

```{code-cell} python
result.credible_interval(0.9)
```

Both take a function `h`, as `estimate` does, for the quantiles of a quantity derived
from the coordinates: those of its values at the atoms, with each atom's weight divided
by $R$ (the quantiles of {func}`telescopic.estimators.weighted_quantile`). They have
shape `()` or `(p,)` for one level and `(len(q),)` or `(len(q), p)` for several, and the
credible interval has shape `(2,)` or `(2, p)`. The sum $x_0 + x_1$ is Gaussian, with a
variance equal to the sum of the entries of the covariance matrix:

```{code-cell} python
levels = [0.05, 0.5, 0.95]
(
    result.quantile(levels, lambda x: x[:, 0] + x[:, 1]),
    stats.norm.ppf(levels, scale=np.sqrt(gaussian.cov.sum())),
)
```

## Histograms

{meth}`telescopic.Result.histogram` returns the mass of each bin, its standard error and
the edges of the bins. The mass of a bin is the average over replicates of their signed
mass in it. The bins are half-open, $[a, b)$, except the last, which is closed, as in
`numpy.histogram`, so an atom on the upper edge of `range` is counted. It is unbiased
for a `range` fixed before the run, and it can be negative where few atoms fall and the
negative weights of the bias corrections outweigh the others. By default the bins span
the atoms, so the masses sum to 1, and since the atoms set the edges, every bin is
biased, the two outer ones by a term of order $1/R$:

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

## Two sets of replicates

A run with `keep` ([Keeping a subset](settings.md#keeping-a-subset)) stores every
replicate's moments but the signed measures of the replicates of index below `keep`
only. {meth}`~telescopic.Result.mean`, {meth}`~telescopic.Result.var`,
{meth}`~telescopic.Result.std` and, up to 100 dimensions, {meth}`~telescopic.Result.cov`
come from all $R$ replicates that met. {meth}`~telescopic.Result.estimate`, the quantiles,
the credible intervals, the histograms and their plots, `measure` and
{meth}`~telescopic.Result.chains` come from the `n_kept` kept replicates that met, and so
does `cov` above 100 dimensions, whose diagonal can then differ from `var`. Each set gives
unbiased estimates, or consistent ones for quantiles, with error bars from its own
replicates, and the summary names the replicates behind each column. The two sets are not
independent: an estimate from one combined with one from the other has no error bar of
its own.

## Signed measures

`result.measure` is a {class}`telescopic.SignedMeasure`: the `atoms`, of shape `(N, d)`,
their `weights`, and for each atom the position, from 0 to $R - 1$, of the `replicate` it
belongs to, in nondecreasing order, so that each replicate's atoms are consecutive. These
attributes are read-only views, without a copy, of the arrays the
measure was built from; those arrays, such as the ones of the replicates that
{func}`telescopic.engine.run` returns, stay writable, and telescopic never writes to
them. Each replicate's weights sum to 1. Its bias corrections add pairs of atoms
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

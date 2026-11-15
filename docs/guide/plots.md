---
file_format: mystnb
kernelspec:
  name: python3
---

# Plot a result

```{code-cell} python
:tags: [remove-cell]

import telescopic
from telescopic import examples

gaussian = examples.gaussian(2)
result = telescopic.sample(
    gaussian.log_prob, gaussian.initial, 1000, vectorize=True, rng=11
)
```

The plots need the `plot` extra (matplotlib). {meth}`telescopic.Result.plot_marginals`
draws the signed histogram of each coordinate as a density, with boxes of plus or minus
two standard errors:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Two bell-shaped histograms, one for each coordinate, each bin topped by a
      small box.
  figure:
    caption: >-
      The signed histogram of each coordinate as a density, with a box of plus or
      minus two standard errors on each bin.
---
result.plot_marginals();
```

{meth}`telescopic.Result.plot_corner` draws the marginals on its diagonal and the signed
joint density of each pair of coordinates below it, on a diverging color map centered on
0:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      A two-by-two grid: a histogram of each coordinate on the diagonal, and below
      it a colored map of their joint density along a rising diagonal.
  figure:
    caption: >-
      The corner plot: the signed marginal of each coordinate on the diagonal, and
      their signed joint density below it, on a color map centered on 0.
---
result.plot_corner();
```

{meth}`telescopic.Result.plot_meeting_times` and {meth}`telescopic.Result.plot_tv_bound`
are shown in [Read meeting times and the distance to stationarity](meeting-times.md).
Each method returns its matplotlib axes, and all but `plot_corner` draw on axes given as
`axes` or `ax`. The functions of {mod}`telescopic.plotting`, which `import telescopic`
leaves out since it needs matplotlib, take the result as their first argument and do the
same; {func}`telescopic.plotting.paths` also draws the coupled paths of a few {term}`replicates <replicate>`
of an engine run with `keep_paths=True` ([How it works](../tutorials/how-it-works.md)).

The {term}`atoms <atom>` are not samples of the target, even when plotted without their weights: the
atoms of each Y chain carry negative weights. Packages such as corner and ArviZ compute
weighted quantiles that assume non-negative weights, so the atoms should not be passed
to them with their weights either.

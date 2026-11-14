---
file_format: mystnb
kernelspec:
  name: python3
---

# Result and summary fields

```{code-cell} python
:tags: [remove-cell]

import numpy as np

import telescopic
from telescopic import examples

gaussian = examples.gaussian(2)
result = telescopic.sample(
    gaussian.log_prob, gaussian.initial, 1000, vectorize=True, rng=11
)
```

## What a result holds

| Attribute | Content |
|---|---|
| `n_replicates`, `n_censored` | $R$, the number of replicates whose pairs met, and the number of pairs censored at `max_iter` ([Diagnostics and warnings](../guide/diagnostics.md#censored-pairs-censored)) |
| `k`, `m`, `lag`, `thin`, `max_iter` | the settings of the run ([Settings](../guide/settings.md)) |
| `names`, `ndim` | the parameter names and the dimension $d$ |
| `measure` | the replicates' signed measures |
| `index`, `meeting_times`, `costs` | for every replicate, censored ones included: its index, its meeting time (`inf` if censored) and its kernel applications |
| `n_log_prob`, `n_grad`, `wall_time` | the evaluations of the log-density and of its gradient, and the seconds, of the replicates, without tuning, the meeting-time pilot and the gradient check, which runs when the sampler is built |
| `kernel`, `initial` | the frozen kernel and the initial distribution |
| `tuning`, `pilot` | the report of the tuning chains, and the meeting times of the lag-1 pilot pairs that chose `k`; `None` when not run |
| `seed` | the root `numpy.random.SeedSequence`, which reproduces the run |
| `interrupted` | whether a requested replicate never ran ([Parallel runs](../guide/parallel.md#interrupting-a-run)) |

## The summary

{meth}`telescopic.Result.summary` returns a {class}`telescopic.Summary`. It shows a table
with one row per parameter, then the settings and diagnostics of the run and the warnings
that apply, one per line. A notebook shows it in HTML, as here; `print` gives it as text,
whose lines below the table are wrapped at 80 characters:

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

- $R$, the settings `k`, `m`, `lag` and `thin` ([Settings](../guide/settings.md)), and the kernel,
  whose floating-point parameters show to 4 significant figures (`summary.footer` keeps
  every digit);
- the median, the 90% and 99% quantiles and the maximum of the meeting times of every
  pair, where censored pairs sort above every pair that met, and a quantile that falls
  on one shows as `>` and the cap, such as `> 12` for `max_iter=12`; from 500
  replicates, the tail index of those that met
  ([`tail`](../guide/diagnostics.md#heavy-tailed-meeting-times-tail)); and the number of censored
  pairs ([`censored`](../guide/diagnostics.md#censored-pairs-censored));
- the mean cost of a replicate in kernel applications and in seconds, and the relative
  cost, the mean cost over $m - k + 1$ ([Settings](../guide/settings.md#cost));
- the fraction of pairs that met after `k + lag`, censored ones included, with a
  one-sided 95% upper bound: only these replicates carry a bias correction, and with
  `lag = k` the fraction is close to the bound on the total variation distance at `k`
  ([Distance to stationarity](../guide/diagnostics.md#distance-to-stationarity));
- the largest share of one replicate in the variance of the estimate of a coordinate's
  mean or variance, and that estimate
  ([`variance`](../guide/diagnostics.md#one-replicate-dominating-the-variance-variance));
- for an interrupted run, that its estimates are consistent but not exactly unbiased
  ([Interrupting a run](../guide/parallel.md#interrupting-a-run));
- each warning that applies ([Warnings](../guide/diagnostics.md#warnings)).

`summary.table` holds the columns as arrays, and `summary.footer` the rest as a
dictionary. {meth}`telescopic.Summary.to_dict` converts both to Python numbers, and
{meth}`telescopic.Summary.to_pandas` returns the table as a `pandas.DataFrame` indexed by
parameter name (it needs the `pandas` extra):

```{code-cell} python
summary.to_dict()["parameters"]["x0"]
```

{meth}`telescopic.Result.diagnostics` returns the diagnostics of the footer as a
dictionary ([The diagnostics of a run](../guide/diagnostics.md#the-diagnostics-of-a-run)).

---
file_format: mystnb
kernelspec:
  name: python3
---

# Result and summary fields

```{code-cell} python
:tags: [remove-cell]

import telescopic
from telescopic import examples

gaussian = examples.gaussian(2)
result = telescopic.sample(
    gaussian.log_prob, gaussian.initial, 1000, vectorize=True, rng=11
)
```

## What a result holds

```{diagram} result
The attributes of a {class}`telescopic.Result`, in four groups. `measure`, a
{class}`telescopic.SignedMeasure`, holds the atoms, weights and replicate indices of the
kept replicates that met, $N$ atoms in $d$ dimensions, which `estimate`, `quantile` and
`histogram` read. The meeting times and costs cover every replicate, censored ones
included, and `tv_bound` reads the meeting times. The settings, the record of the run and
the counts complete it; the table below gives each.
```

| Attribute | Content |
|---|---|
| `n_replicates`, `n_censored` | $R$, the number of {term}`replicates <replicate>` whose {term}`pairs <pair>` met, and the number of pairs censored at `max_iter` ([Censored pairs](warnings.md#censored-pairs-censored)) |
| `k`, `m`, `lag`, `thin`, `max_iter` | the settings of the run ([Settings](../guide/settings.md)) |
| `names`, `ndim` | the parameter names and the dimension $d$ |
| `measure` | the replicates' {term}`signed measures <signed measure>` |
| `index`, `meeting_times`, `costs` | for every replicate, censored ones included: its index, its {term}`meeting time` (`inf` if censored) and its kernel applications |
| `n_log_prob`, `n_grad`, `wall_time` | the evaluations of the log-density and of its gradient, and the seconds, of the replicates, without tuning, the {term}`meeting-time pilot` and the gradient check, which runs when the sampler is built |
| `kernel`, `initial` | the frozen kernel and the initial distribution |
| `tuning`, `pilot` | the report of the tuning chains, and the meeting times of the lag-1 pilot pairs that chose `k`; `None` when not run |
| `seed` | the root `numpy.random.SeedSequence`, which reproduces the run |
| `interrupted` | whether a requested replicate never ran ([Parallel runs](../guide/parallel.md#interrupting-a-run)) |
| `keep`, `n_kept` | the run's `keep`, `None` when every replicate stored its signed measure, and the number of kept replicates that met, which `measure`, the quantiles, the histograms, `estimate` and `chains()` read ([Keeping a subset](../guide/settings.md#keeping-a-subset)) |
| `abandoned` | the replicates that an error stopped while they ran and that have not run since ([Continuing after an error](../guide/save-load-merge.md#continuing-after-an-error)) |

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
  {term}`unbiased`; `standard_error`: its Monte Carlo error, valid as the number of replicates
  grows, however short each run.
- `std`: $\sqrt{\max(0, \widehat{\mathrm{Var}})}$, from the unbiased estimate of the
  variance; {term}`consistent`, not unbiased.
- `q16`, `q50` and `q84`: the 16%, 50% and 84% quantiles of the pooled signed measures;
  consistent, not unbiased.

The lines below it:

- $R$, the settings `k`, `m`, `lag` and `thin` ([Settings](../guide/settings.md)), and the kernel,
  whose floating-point parameters show to 4 significant figures (`summary.footer` keeps
  every digit);
- with `keep`, which replicates each column comes from: the mean, its standard error and
  the standard deviation from all $R$, and the quantiles from the kept replicates, or
  "no kept replicate" when none met, with the quantile columns then `nan`
  ([Keeping a subset](../guide/settings.md#keeping-a-subset));
- the median, the 90% and 99% quantiles and the maximum of the meeting times of every
  pair, where {term}`censored pairs <censored pair>` sort above every pair that met, and a quantile that falls
  on one shows as `>` and the cap, such as `> 12` for `max_iter=12`; from 500
  replicates, the {term}`tail index` of those that met
  ([`tail`](warnings.md#heavy-tailed-meeting-times-tail)); and the number of censored
  pairs ([`censored`](warnings.md#censored-pairs-censored));
- the mean cost of a replicate in kernel applications and in seconds, and the
  {term}`relative cost`, the mean cost over $m - k + 1$ ([Settings](../guide/settings.md#cost));
- the fraction of pairs that met after `k + lag`, censored ones included, with a
  one-sided 95% upper bound: only these replicates carry a bias correction, and with
  `lag = k` the fraction is close to the bound on the total variation distance at `k`
  ([Distance to stationarity](../guide/meeting-times.md#distance-to-stationarity));
- the largest share of one replicate in the variance of the estimate of a coordinate's
  mean or variance, and that estimate
  ([`variance`](warnings.md#one-replicate-dominating-the-variance-variance));
- for an interrupted run, that its estimates are consistent but not exactly unbiased
  ([Interrupting a run](../guide/parallel.md#interrupting-a-run)); while an error has
  abandoned replicates, that the estimates leave out replicates selected for long
  meeting times, in its place
  ([Continuing after an error](../guide/save-load-merge.md#continuing-after-an-error));
- each warning that applies ([Warnings](warnings.md#warnings)).

`summary.table` holds the columns as arrays, and `summary.footer` the rest as a
dictionary, with `keep`, `n_kept` and `abandoned`, the number of abandoned replicates. {meth}`telescopic.Summary.to_dict` converts both to Python numbers, and
{meth}`telescopic.Summary.to_pandas` returns the table as a `pandas.DataFrame` indexed by
parameter name (it needs the `pandas` extra):

```{code-cell} python
summary.to_dict()["parameters"]["x0"]
```

## The diagnostics of a run

{meth}`telescopic.Result.diagnostics` returns the diagnostics that the summary prints, as
a dictionary; [The summary](#the-summary) defines each line of the summary.

```{code-cell} python
result.diagnostics()
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
values = result.diagnostics()
assert set(values) == {
    "meeting_time_50", "meeting_time_90", "meeting_time_99", "meeting_time_max",
    "censored", "tail_index", "late_fraction", "late_fraction_upper", "tv_bound_at_k",
    "variance_share", "variance_share_of", "variance_share_replicate", "mean_cost",
    "seconds_per_replicate", "relative_cost",
}
assert result.lag == result.k
assert abs(values["late_fraction"] - values["tv_bound_at_k"]) < 0.01
```

- `meeting_time_50`, `meeting_time_90`, `meeting_time_99`, `meeting_time_max` and
  `censored`: the quantiles of the meeting times of every pair, each an observed meeting
  time from the sorted values, where censored pairs sort above every pair that met, and
  the number of censored pairs. A quantile that falls on a censored pair is `inf`.
- `tail_index`: the tail index of $\tau - L + 1$, the power $\alpha$ at which the
  probability of a value above $t$ falls as $t^{-\alpha}$, by Hill's estimate from the
  largest values; `nan` below 500 replicates or behind ties
  ([`tail`](warnings.md#heavy-tailed-meeting-times-tail)).
- `late_fraction` and `late_fraction_upper`: the fraction of pairs that met after
  `k + lag`, censored ones included, and its one-sided 95% Clopper-Pearson upper bound,
  an exact binomial confidence bound, which stays informative when the fraction is 0. The fraction is the leading term of
  `tv_bound_at_k`, the TV bound at $k$, and with `lag = k` the two are close.
- `variance_share`, `variance_share_of` and `variance_share_replicate`: the largest share
  of one replicate in the variance of the estimate of a coordinate's mean or variance,
  with each replicate's second moments taken about the mean estimate, that estimate,
  such as `"the variance of x1"`, and the index of that replicate; `nan`, `None` and
  `None` below 2 replicates
  ([`variance`](warnings.md#one-replicate-dominating-the-variance-variance)).
- `mean_cost`, `seconds_per_replicate` and `relative_cost`: the cost of a replicate
  ([Settings](../guide/settings.md#cost)).

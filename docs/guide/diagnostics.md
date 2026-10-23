---
file_format: mystnb
kernelspec:
  name: python3
---

# Diagnostics and warnings

The meeting times of the pairs bound the distance of the chains from the target and set
the cost of a run, and telescopic warns when a run's error bars may be unreliable. The
examples use the run of [Results](results.md):

```{code-cell} python
import warnings

import numpy as np

import telescopic
from telescopic import examples

gaussian = examples.gaussian(2)
result = telescopic.sample(
    gaussian.log_prob, gaussian.initial, 1000, vectorize=True, rng=11
)
```

## Meeting times

`result.meeting_times` holds each pair's meeting time $\tau$
([Theory](../theory.md#coupled-chains)), `inf` for a pair censored at `max_iter`
([`censored`](#censored-pairs-censored)); `result.costs` holds its kernel applications
and `result.index` its replicate index.

{meth}`telescopic.Result.plot_meeting_times` draws a histogram of $\tau - L$ and its
survival function on a log scale, where a straight tail means geometric meeting times. A
censored pair counts in the survival function up to its cap, and not in the histogram.

```{code-cell} python
result.plot_meeting_times();
```

## Distance to stationarity

{meth}`telescopic.Result.tv_bound` estimates the upper bound on the total variation
distance between the law of $X_t$ and the target
([Theory](../theory.md#distance-to-stationarity)) by its average over the replicates,
capped at 1. By default it returns the bound at every $t$ from 0 to the last time at
which it is positive, so that entry `t` is the bound at `t`. A censored pair counts as
meeting at its cap, so with censored pairs the estimate is a lower bound of the one
without a cap.

```{code-cell} python
result.tv_bound([0, result.k // 2, result.k])
```

{meth}`telescopic.Result.plot_tv_bound` draws it on a log scale, with `k` marked:

```{code-cell} python
result.plot_tv_bound();
```

{meth}`telescopic.Result.w1_bound` estimates the bound on the 1-Wasserstein distance in
the same way, from the distances $\lVert X_t - Y_{t-L} \rVert$ that the engine records
before each pair meets. A censored pair's distances stop at its cap, so with censored
pairs it too is a lower bound. It depends on the parametrisation of the target through
the Euclidean norm: in other units or coordinates the bound changes.

```{code-cell} python
result.w1_bound([0, result.k // 2, result.k])
```

{func}`telescopic.diagnostics.tv_bound` and {func}`telescopic.diagnostics.w1_bound`
compute the same bounds from arrays, such as meeting times of your own, with each
censored pair given as its cap.

## The diagnostics of a run

{meth}`telescopic.Result.diagnostics` returns the diagnostics that the summary prints, as
a dictionary; [Results](results.md#the-summary) defines each line of the summary.

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
    "variance_share", "variance_share_of", "mean_cost", "seconds_per_replicate",
    "relative_cost",
}
assert result.lag == result.k
assert abs(values["late_fraction"] - values["tv_bound_at_k"]) < 0.01
```

- `meeting_time_50`, `meeting_time_90`, `meeting_time_99`, `meeting_time_max` and
  `censored`: the meeting times of the pairs that met, and the number of censored pairs.
- `tail_index`: Hill's estimate of the tail index of $\tau - L + 1$, `nan` below 500
  replicates, or when ties among small meeting times leave the tail unresolved
  ([`tail`](#heavy-tailed-meeting-times-tail)).
- `late_fraction` and `late_fraction_upper`: the fraction of pairs that met after
  `k + lag`, censored ones included, and its one-sided 95% Clopper-Pearson upper bound,
  which stays informative when the fraction is 0. The fraction is the leading term of
  `tv_bound_at_k`, the TV bound at $k$, and with `lag = k` the two are close.
- `variance_share` and `variance_share_of`: the largest share of one replicate in the
  variance of the estimate of a coordinate's mean or second moment, and that estimate,
  such as `"the second moment of x1"`; `nan` and `None` below 2 replicates
  ([`variance`](#one-replicate-dominating-the-variance-variance)).
- `mean_cost`, `seconds_per_replicate` and `relative_cost`: the cost of a replicate
  ([Settings](settings.md#cost)).

## Warnings

Each run ends by emitting a
{class}`telescopic.ConvergenceWarning <telescopic.warnings.ConvergenceWarning>` for each
condition below that applies to the result of all its replicates so far. Disagreeing
tuning chains warn instead when the kernel is tuned, and a slow pair when it is seen: as
it happens in this process, or when the run ends on a pool. {meth}`telescopic.Result.check`
emits the warnings that apply to any result, and {func}`telescopic.diagnostics.issues`
returns them, without warning, as a list of {class}`telescopic.diagnostics.Issue` with
the `kind` named in each heading below and a `message`. An empty list means only that
none of them holds, not that the run is right. They are heuristics, with thresholds set
so that clean Gaussian runs at the default settings rarely trigger them; the measured
rates below show how often they fire elsewhere.

### Censored pairs (`censored`)

**Trigger:** a pair reached `max_iter` without meeting. Without a cap, the default, every
pair runs until it meets. With `max_iter`, a pair still unmet when $t$ reaches it is
censored: its meeting time is `inf`, and it is left out of the estimates, of $R$ and of
the signed measure. `result.n_censored` counts them. **Meaning:** leaving them out biases
the estimates, since they are the replicates whose chains take longest to meet.
**Remedy:** a new sampler with a larger `max_iter`, or none; a sampler keeps the cap of
its first run.

### Late pairs (`late`)

**Trigger:** more than a quarter of the pairs meet after `k + lag` (`late_fraction` above
0.25). **Meaning:** `k` may be too small, so that many replicates carry bias corrections,
which make the variance high. **Remedy:** a new sampler with a larger `k`
([Settings](settings.md)).

### One replicate dominating the variance (`variance`)

**Trigger:** with at least 100 replicates, one replicate holds more than half of the
sample variance of the estimate of some coordinate's mean or second moment.
`result.diagnostics()` reports the largest share as `variance_share`, and the estimate
that holds it as `variance_share_of`, whether or not the warning fires. **Meaning:**
heavy tails or rare excursions make the error bars unreliable. A replicate whose chain is
still on its way to the target at `k` does this, as from a start far wider than the
target ([Initial distributions](targets.md#choosing-one)), and so does a replicate that
found a mode that the others missed. **Remedy:** find the replicate in the `replicates`
of the estimate, check the initial distribution, and look for a mode the other
replicates missed. The warning looks at the first two moments of each coordinate only;
{func}`telescopic.diagnostics.variance_share` applied to `result.estimate(h).replicates`
checks a function of your own.

### Heavy-tailed meeting times (`tail`)

**Trigger:** with at least 500 replicates, Hill's estimate of the tail index of
$\tau - L + 1$ ({func}`telescopic.diagnostics.hill`), from its $\lfloor R/10 \rfloor$
largest values, is below 2. `result.diagnostics()` reports it as `tail_index` whether or
not the warning fires. **Meaning:** unbiasedness needs moments of the meeting time, and
valid error bars need a finite variance of the estimates
([Theory](../theory.md#guarantees-and-their-limits)). Meeting times with a polynomial
tail can break both: a tail index below 2 means an infinite variance of the meeting
times, the estimates may have one too, and then their error bars are not reliable at any
number of replicates. **Remedy:** look at the survival function of `plot_meeting_times`,
whose tail is straight on its log scale for geometric meeting times, and at the kernel
([Kernels](kernels.md)).

### Disagreeing tuning chains (`tuning`)

**Trigger:** the chains that tuned the kernel disagree: their Gelman-Rubin $\hat R$
(`result.tuning.rhat`) exceeds $\sqrt 2$ in some coordinate, so their means spread more
widely than the chains themselves. **Meaning:** the target may have several modes, which
make the meeting times heavy-tailed, and pairs that never leave one mode miss the others.
**Remedy:** an initial distribution that covers every mode
([Initial distributions](targets.md#choosing-one)).

### Slow pairs (`slow`)

**Trigger:** a pair still unmet after $\max(10^4, 10 \max(L, m))$ iterations; the
sampler names the replicate in one warning per run. **Meaning:** the meeting times may be
heavy-tailed, and the run waits for that pair. **Remedy:** look at the kernel and the
initial distribution. In the meeting-time pilot, whose own cap is on
[Settings](settings.md#the-automatic-pipeline), the threshold is $10^4$.

## A run that warns

The banana from its wide default start, with `k`, `lag` and `m` set by hand:

```{code-cell} python
banana = examples.banana()
short = telescopic.sample(
    banana.log_prob, banana.initial, vectorize=True, k=200, lag=200, m=2000, rng=0
)
```

```{code-cell} python
[issue.kind for issue in telescopic.diagnostics.issues(short)]
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert [issue.kind for issue in telescopic.diagnostics.issues(short)] == ["late", "variance"]
```

The warnings say that `k` may be too small: more than a quarter of the pairs met after
`k + lag`, and one replicate holds more than half of the variance of an estimate.

## Measured rates

When the thresholds were chosen, repeated runs measured how often each warning fires.
All used vectorised log-densities, and the first two targets `sample()`'s default
settings:

- {func}`telescopic.examples.gaussian` in 10 dimensions, from $\mathcal N(0, 4I)$: no
  warning in 50 runs at $R = 500$; at $R = 100$, the variance warning in at most 2 of 100
  runs, and no other warning.
- The banana ({func}`telescopic.examples.banana`) from $U[-5, 5]^2$: at $R = 100$, the
  variance warning in 19 of 200 runs, whose 95% intervals of $E[x_0^2]$ covered the truth
  in 84% of them against 96% in the other runs; at $R = 500$, the variance warning in at
  most 6 of 50 runs and the tail warning in 14.
- The bimodal mixture of Jacob, O'Leary & Atchadé (2020, §5.1),
  {func}`telescopic.examples.mixture`, from $\mathcal N(10, 1)$, with a random walk of
  proposal variance 1, `k = 50`, `m = 500`, `lag = 1` and $R = 1000$: the 95% intervals
  of $P(x > 3)$ missed the truth in 33 of 50 runs, none of which warned. The 17 runs whose
  replicates found the other mode raised the variance warning, and their intervals
  covered the truth.

## A run that never left one mode

The last case shows the limit of every diagnostic here: a run whose pairs all met in one
mode cannot be detected from its own output, since nothing in its meeting times or
estimates shows the mode that no replicate visited. The remedy is an initial
distribution that covers every mode ([Initial distributions](targets.md#choosing-one)).
[When unbiased is not enough](../tutorials/failure-modes.md) works through such runs.

## Warning classes

{class}`telescopic.TelescopicWarning <telescopic.warnings.TelescopicWarning>`, a subclass
of `UserWarning`, is the base class of telescopic's warnings. A
{class}`telescopic.ConvergenceWarning <telescopic.warnings.ConvergenceWarning>` says that
a result may be unreliable: the conditions above. An
{class}`telescopic.InputWarning <telescopic.warnings.InputWarning>` says that an input is
legal but questionable: replicates whose initial draws have zero target density, and a
run whose atoms would take more than 1 GB ([Settings](settings.md#memory-and-thinning)).
Each warning points at the line of your code that called telescopic.

The filters of the `warnings` module select them by class:
`warnings.simplefilter("ignore", telescopic.InputWarning)` silences the input warnings,
and `"error"` turns warnings into exceptions. Under such a filter, `sample` or `run`
raises at its first warning instead of returning its result:

```{code-cell} python
with warnings.catch_warnings():
    warnings.simplefilter("error", telescopic.ConvergenceWarning)
    try:
        telescopic.sample(
            banana.log_prob, banana.initial, vectorize=True, k=200, lag=200, m=2000, rng=0
        )
    except telescopic.ConvergenceWarning as error:
        print("raised:", error)
```

`check(strict=True)` raises a `RuntimeError` that lists every condition, and leaves the
result in hand:

```{code-cell} python
try:
    short.check(strict=True)
except RuntimeError as error:
    print(error)
```

## Checking a coupling

{func}`telescopic.testing.check_coupling` checks a kernel's coupling as the engine uses
it: one coupled step against steps of the kernel alone, the fraction of pairs that meet,
stale cached log-densities and pairs that nearly meet. A kernel of your own should pass
it before use; [Writing a kernel](custom-kernels.md#testing) describes its tests and runs
it on a new kernel.

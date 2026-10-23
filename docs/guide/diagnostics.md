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

## Meeting times and censored pairs

`result.meeting_times` holds each pair's meeting time $\tau$
([Theory](../theory.md#coupled-chains)), `result.costs` its kernel applications and
`result.index` its replicate index. Without a cap, the default, every pair runs until it
meets. With `max_iter`, a pair still unmet when $t$ reaches `max_iter` is censored: its
meeting time is `inf`, and it is left out of the estimates, of $R$ and of the signed
measure, which biases the estimates ([Pitfalls](pitfalls.md#capping-the-run-length)).
`result.n_censored` counts them.

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
before each pair meets. It depends on the parametrisation of the target through the
Euclidean norm: in other units or coordinates the bound changes.

```{code-cell} python
result.w1_bound([0, result.k // 2, result.k])
```

{func}`telescopic.diagnostics.tv_bound` and {func}`telescopic.diagnostics.w1_bound`
compute the same bounds from arrays, such as meeting times of your own, with each
censored pair given as its cap.

## The diagnostics of a run

{meth}`telescopic.Result.diagnostics` gathers the diagnostics that the summary prints:

```{code-cell} python
result.diagnostics()
```

- `meeting_time_50`, `meeting_time_90`, `meeting_time_99` and `meeting_time_max`: the
  quantiles and the maximum of the meeting times of the pairs that met; `censored`: the
  number of censored pairs.
- `late_fraction`: the fraction of pairs that met after `k + lag`, $\tau - L > k$,
  censored ones included. Only a replicate whose pair meets that late carries a bias
  correction ([Theory](../theory.md#the-time-averaged-estimator)). The fraction is the
  leading term of the TV bound at $k$, `tv_bound_at_k`, and with `lag = k` the two are
  close. `late_fraction_upper` is its one-sided 95% Clopper-Pearson upper bound, which
  stays informative when the fraction is 0.
- `mean_cost`, `seconds_per_replicate` and `inefficiency`: the mean number of kernel
  applications per replicate, the wall time per replicate, and the mean cost over
  $m - k + 1$ ([Settings](settings.md#cost-and-inefficiency)).

## Warnings

Each run ends by emitting a
{class}`telescopic.ConvergenceWarning <telescopic.warnings.ConvergenceWarning>` for each
condition below that applies to the result of all its replicates so far. Disagreeing
tuning chains warn instead when the kernel is tuned, and a slow pair when it is seen: as
it happens in this process, or when the run ends on a pool. {meth}`telescopic.Result.check`
emits the warnings that apply to any result, and {func}`telescopic.diagnostics.issues`
returns them, without warning, as a list of {class}`telescopic.diagnostics.Issue` with
the `kind` named in each heading below and a `message`. They are heuristics, with
thresholds set so that clean Gaussian runs at the default settings rarely trigger them;
the measured rates below show how often they fire elsewhere.

### Censored pairs (`censored`)

**Trigger:** a pair reached `max_iter` without meeting. **Meaning:** the estimates leave
it out, which biases them. **Remedy:** run again without a cap or with a larger one; a
sampler keeps the cap of its first run, so this needs a new one.

### Late pairs (`late`)

**Trigger:** more than a quarter of the pairs meet after `k + lag` (`late_fraction` above
0.25). **Meaning:** `k` may be too small, so that many replicates carry bias corrections,
which make the variance high. **Remedy:** a new sampler with a larger `k`
([Settings](settings.md)).

### One replicate dominating the variance (`variance`)

**Trigger:** with at least 100 replicates, one replicate holds more than half of the
sample variance of the estimate of some coordinate's mean or second moment
({func}`telescopic.diagnostics.variance_share`). **Meaning:** heavy tails or rare
excursions make the error bars unreliable. A replicate whose chain is still on its way to
the target at `k` does this, as from a start far wider than the target
([Pitfalls](pitfalls.md#an-initial-distribution-far-wider-than-the-target)), and so does a
replicate that found a mode that the others missed. **Remedy:** find the replicate in the
`replicates` of the estimate, check the initial distribution, and look for a mode the
other replicates missed. The warning looks at the first two moments of each coordinate
only; `variance_share(result.estimate(h).replicates)` checks a function of your own.

### Heavy-tailed meeting times (`tail`)

**Trigger:** with at least 500 replicates, Hill's estimate of the tail index of
$\tau - L + 1$ ({func}`telescopic.diagnostics.hill`), from its
$\max(10, \lfloor R/10 \rfloor)$ largest values, is below 2. **Meaning:** meeting times
with a tail index below 2 have an infinite variance, and the estimates may too
([Pitfalls](pitfalls.md#heavy-tails)). **Remedy:** look at the survival function of
`plot_meeting_times`, whose tail is straight on its log scale for geometric meeting times,
and at the kernel ([Kernels](kernels.md)).

### Disagreeing tuning chains (`tuning`)

**Trigger:** the chains that tuned the kernel disagree: their Gelman-Rubin $\hat R$
(`result.tuning.rhat`) exceeds $\sqrt 2$ in some coordinate, so their means spread more
widely than the chains themselves. **Meaning:** the target may have several modes, which
make the meeting times heavy-tailed, and pairs that never leave one mode miss the others.
**Remedy:** an initial distribution that covers every mode
([Pitfalls](pitfalls.md#an-initial-distribution-that-misses-a-mode)).

### Slow pairs (`slow`)

**Trigger:** a pair still unmet after $\max(10^4, 10 \max(L, m))$ iterations; the
sampler names the replicate in one warning per run. **Meaning:** the meeting times may be
heavy-tailed, and the run waits for that pair. **Remedy:** look at the kernel and the
initial distribution. In the meeting-time pilot, whose own cap is on
[Settings](settings.md#the-automatic-pipeline), the threshold is $10^4$.

### A run that warns

On the banana from its wide default start, a `k` set by hand too small for the target
triggers two warnings:

```{code-cell} python
banana = examples.banana()
short = telescopic.sample(
    banana.log_prob, banana.initial, vectorize=True, k=200, lag=200, m=2000, rng=0
)
```

```{code-cell} python
[issue.kind for issue in telescopic.diagnostics.issues(short)]
```

### Measured rates

Spec section 7.4 ([Specification](../design/spec.md)) records how often each warning
fired in the measurements of milestone M9, with vectorised log-densities and, on the
first two targets, `sample()`'s defaults:

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

### A run that never left one mode

The last case shows the limit of every diagnostic here: a run whose pairs all met in one
mode cannot be detected from its own output, since nothing in its meeting times or
estimates shows the mode that no replicate visited. The remedy is an initial
distribution that covers every mode
([Pitfalls](pitfalls.md#an-initial-distribution-that-misses-a-mode)).
[Failure modes](../tutorials/failure-modes.md) works through such runs.

## Warning classes

{class}`telescopic.TelescopicWarning <telescopic.warnings.TelescopicWarning>`, a subclass
of `UserWarning`, is the base class of telescopic's warnings. A
{class}`telescopic.ConvergenceWarning <telescopic.warnings.ConvergenceWarning>` says that
a result may be unreliable: the conditions above. An
{class}`telescopic.InputWarning <telescopic.warnings.InputWarning>` says that an input is
legal but questionable: replicates whose initial draws have zero target density, and a
run whose atoms would take more than 1 GB ([Settings](settings.md#memory-and-thinning)). Each
warning points at the line of your code that called telescopic.

The filters of the `warnings` module select them by class:
`warnings.simplefilter("ignore", telescopic.InputWarning)` silences the input warnings,
and `"error"` turns warnings into exceptions:

```{code-cell} python
with warnings.catch_warnings():
    warnings.simplefilter("error", telescopic.ConvergenceWarning)
    try:
        short.check()
    except telescopic.ConvergenceWarning as error:
        print("raised:", error)
```

With such a filter in place during `sample` or `run`, the call raises at its first
warning instead of returning its result. `check(strict=True)` raises a `RuntimeError`
that lists every condition, and leaves the result in hand:

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
it before use; [Custom kernels](custom-kernels.md#testing) describes its tests and runs
it on a new kernel.

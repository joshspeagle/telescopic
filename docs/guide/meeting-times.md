---
file_format: mystnb
kernelspec:
  name: python3
---

# Read meeting times and the distance to stationarity

The {term}`meeting times <meeting time>` of the {term}`pairs <pair>` bound the distance of the chains from the target and set
the cost of a run. [Meeting times and distance to stationarity](../concepts/meeting-times.md)
explains the bound. The examples use the run of
[Estimate anything from a result](estimates.md):

```{code-cell} python
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
([`censored`](../reference/warnings.md#censored-pairs-censored)); `result.costs` holds its kernel applications
and `result.index` its {term}`replicate` index.

{meth}`telescopic.Result.plot_meeting_times` draws a histogram of $\tau - L$ and its
survival function on a log scale, where a straight tail means geometric meeting times. A
{term}`censored pair` counts in the survival function at every time up to its cap, where the
curve ends at the fraction of pairs censored, and not in the histogram.

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Left, a histogram of tau minus L over the pairs; right, the fraction of
      pairs with tau minus L above t, on a logarithmic scale, falling with t.
  figure:
    caption: >-
      The meeting times of the run, less the lag: their histogram (left) and their
      survival function $P(\tau - L > t)$ on a log scale (right).
---
result.plot_meeting_times();
```

{func}`telescopic.plotting.survival` draws the survival function alone, from an array of
meeting times and their lag, such as the times that
{meth}`telescopic.CoupledSampler.meeting_times` returns, into a given axis, so that the
curves of several runs share one; censored pairs, `inf`, need their cap as `max_iter`:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      The fraction of pairs with tau minus L above t, on a logarithmic scale,
      falling with t.
  figure:
    caption: >-
      The survival function $P(\tau - L > t)$ of the run's meeting times, drawn from
      the array.
---
from telescopic import plotting

plotting.survival(result.meeting_times, result.lag, max_iter=result.max_iter);
```

## Distance to stationarity

With `keep`, the 1-Wasserstein bound comes from the distances of the kept replicates,
censored ones included, which are the only ones stored; the meeting times, the TV bound
and the other diagnostics of this page come from every replicate.

{meth}`telescopic.Result.tv_bound` estimates the upper bound on the total variation
distance between the law of $X_t$ and the target
([Theory](../theory.md#distance-to-stationarity)) by its average over the replicates,
capped at 1, and returns it with its standard error: the sample standard deviation of
the replicates' terms $\max(0, \lceil(\tau - L - t)/L\rceil)$ over $\sqrt R$, the error of
the average before the cap. By default it returns both at every $t$ from 0 to the last
time at which the bound is positive, so that entry `t` is the bound at `t`. A censored pair is known to
meet after its cap, where its chains were last compared, and counts as meeting at the
cap plus 1, so with censored pairs the estimate is a lower bound of the one without a
cap.

```{code-cell} python
bound, standard_error = result.tv_bound([0, result.k // 2, result.k])
print(bound, standard_error)
```

{meth}`telescopic.Result.plot_tv_bound` draws it on a log scale, with a band of two
standard errors and `k` marked:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      A line on a logarithmic scale that falls from 1 as t grows, inside a shaded
      band, with a dashed vertical line at k.
  figure:
    caption: >-
      The estimated upper bound on the total variation distance between the law of
      $X_t$ and the target, against $t$, with a band of two standard errors and $k$
      dashed.
---
result.plot_tv_bound();
```

{meth}`telescopic.Result.w1_bound` estimates the bound on the 1-Wasserstein distance in
the same way, with its standard error, from the distances $\lVert X_t - Y_{t-L} \rVert$
that the engine records before each pair meets. A censored pair's distances stop at its cap, so with censored
pairs it too is a lower bound. It depends on the parameterization of the target through
the Euclidean norm: in other units or coordinates the bound changes.

```{code-cell} python
bound, standard_error = result.w1_bound([0, result.k // 2, result.k])
print(bound, standard_error)
```

{func}`telescopic.diagnostics.tv_bound` and {func}`telescopic.diagnostics.w1_bound`
compute the same bounds and standard errors from arrays, such as meeting times of your
own, with each censored pair given as the cap plus 1.

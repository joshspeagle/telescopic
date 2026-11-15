---
file_format: mystnb
kernelspec:
  name: python3
---

# Checking the burn-in of an existing sampler

You may keep your own MCMC code and want only to know how many iterations to discard as
{term}`burn-in`. Coupled {term}`pairs <pair>` of your kernel answer that without
computing any estimate: their {term}`meeting times <meeting time>` bound the distance
between the law of your chain at time $t$ and the target
([Meeting times and distance to stationarity](../concepts/meeting-times.md)). This page
wraps a Metropolis sampler with a {term}`coupling`, runs pairs, reads the bounds, chooses
a burn-in, and checks the bounds against many runs of the original sampler.

```{code-cell} python
:tags: [remove-cell]

from myst_nb import glue
```

## Your sampler

The target stands for your model: a Gaussian in 5 dimensions whose neighboring
coordinates have correlation 0.9, from {func}`telescopic.examples.gaussian`, with mean 0.
The sampler is a random-walk Metropolis whose proposal adds $Bz$ to the current point,
with $z \sim \mathcal N(0, I)$ and a fixed $B = 0.5I$, run on a batch of chains at once,
each started from $\mathcal N(3 \cdot \mathbf 1, I)$, far from the mode:

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

import telescopic
from telescopic import couplings, diagnostics, engine, examples, kernels

log_prob = examples.gaussian(5).log_prob  # takes a batch of points, shape (n, 5)
factor = 0.5 * np.eye(5)  # B


def my_sampler(x, n_steps, rng):
    """Advance the chains in the rows of x by n_steps Metropolis steps."""
    current = log_prob(x)
    for _ in range(n_steps):
        proposal = x + rng.standard_normal(x.shape) @ factor.T
        proposed = log_prob(proposal)
        accept = np.log(rng.random(len(x))) < proposed - current
        x = np.where(accept[:, None], proposal, x)
        current = np.where(accept, proposed, current)
    return x
```

## The same kernel, with a coupling

{class}`telescopic.kernels.MetropolisHastings` combines a proposal with a coupling of the
proposals of two chains. {class}`~telescopic.kernels.RandomWalkProposal` proposes
$x + Bz$, given $B$ and $B^{-1}$, and accepts by the same test in log space as
`my_sampler`, so each chain of a pair moves exactly as your sampler does.
{class}`~telescopic.couplings.Reflection` draws the two proposals from the
{term}`reflection-maximal coupling`, which lets the two chains become equal
([Proposals and coupling objects](../guide/kernels.md#proposals-and-coupling-objects)).
The initial distribution is the one your chains start from:

```{code-cell} python
proposal = kernels.RandomWalkProposal(factor, np.linalg.inv(factor))
kernel = kernels.MetropolisHastings(proposal, couplings.Reflection())
start = telescopic.initial.Normal(np.full(5, 3.0), np.eye(5))
kernel
```

A sampler whose proposal is of another kind is written as a kernel of its own, with
its coupling, and checked with {func}`telescopic.testing.check_coupling` before use
([Write a kernel](../guide/custom-kernels.md)). The bounds below hold for the law of a
chain started from `start`: they say nothing about chains started elsewhere.

## Pairs with lag 1

{meth}`telescopic.CoupledSampler.meeting_times` runs pairs with {term}`lag` 1 until they
meet or reach `max_iter`. The kernel has every parameter given, so nothing is tuned:

```{code-cell} python
sampler = telescopic.CoupledSampler(log_prob, start, vectorize=True, kernel=kernel, rng=1)
lag_one = sampler.meeting_times(1000, max_iter=10_000)
print("unmet:", np.isinf(lag_one).sum())
print("quantiles 50%, 90%, 99%:", np.quantile(lag_one, [0.5, 0.9, 0.99]))
```

{func}`telescopic.diagnostics.tv_bound` estimates the bound on the total variation
distance $d_{TV}(\pi_t, \pi)$ between the law $\pi_t$ of $X_t$ and the target $\pi$ from
these meeting times. A pair unmet at the cap enters as the cap plus 1, which makes the
estimate a lower bound of the one without a cap; here none is unmet:

```{code-cell} python
t = np.arange(3000)
tv_lag_one = diagnostics.tv_bound(np.where(np.isinf(lag_one), 10_001, lag_one), 1, t)
print("TV bound at t = 0, 500, 1000:", tv_lag_one[[0, 500, 1000]])
```

## Pairs with a longer lag

Every lag bounds the same distance, but a longer lag can bound it more tightly
([Reading the bound](../concepts/meeting-times.md#reading-the-bound)).
{func}`telescopic.engine.run` runs pairs with any lag. It takes the target as a
{class}`telescopic.Target` and a list of replicate indices, and here the lag that the
automatic settings would choose from the lag-1 meeting times
({func}`telescopic.tuning.choose_k`). Its `k` and `m` set what a {term}`replicate`
averages, which this page does not use:

```{code-cell} python
lag = telescopic.tuning.choose_k(lag_one)
target = telescopic.Target(log_prob, vectorize=True)
pairs = engine.run(
    target, kernel, start, np.arange(1000), k=0, m=0, lag=lag, max_iter=50_000, rng=2
)
print(f"lag {lag}; unmet: {np.isinf(pairs.meeting_time).sum()}")
tv = diagnostics.tv_bound(pairs.meeting_time, lag, t)
w1 = diagnostics.w1_bound(pairs.distance_time, pairs.distance, 1000, lag, t)
print("TV bound at t = 0, 500, 1000:", tv[[0, 500, 1000]])
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert not np.isinf(pairs.meeting_time).any() and not np.isinf(lag_one).any()
assert tv_lag_one[500] == 1 and tv[500] < 0.2
```

The lag-1 bound is 1, which says nothing, until long after the bound with the longer lag
has fallen below 0.2. {func}`telescopic.diagnostics.w1_bound` bounds the 1-Wasserstein
distance in the same way, from the distances $\lVert X_t - Y_{t-L} \rVert$ that the
engine records before each pair meets; it depends on the units of the coordinates
([Distance to stationarity](../guide/meeting-times.md#distance-to-stationarity)).

## Choosing the burn-in

The bound is an average over 1,000 pairs and has a Monte Carlo error. Its estimate at
$t$ averages $\max(0, \lceil(\tau - L - t)/L\rceil)$ over the pairs, so the standard
error of that average gives a band, and each pair's own bound gives one for the
1-Wasserstein bound. A cautious rule takes as burn-in the first $t$ at
which the upper edge of the band, two standard errors above the estimate, is below
0.01:

```{code-cell} python
terms = np.maximum(0, np.ceil((pairs.meeting_time[:, None] - lag - t) / lag))
tv_error = terms.std(axis=0, ddof=1) / np.sqrt(len(terms))
# The same for the W1 bound, from each pair's own bound.
w1_terms = np.array([
    diagnostics.w1_bound(
        pairs.distance_time[pairs.distance_owner == r],
        pairs.distance[pairs.distance_owner == r],
        1,
        lag,
        t,
    )
    for r in pairs.index
])
w1_error = w1_terms.std(axis=0, ddof=1) / np.sqrt(len(w1_terms))
burn_in = int(np.argmax(tv + 2 * tv_error < 0.01))
print(f"burn-in: {burn_in}")
print(f"TV bound there: {tv[burn_in]:.4f} ± {tv_error[burn_in]:.4f}")
print(f"W1 bound there: {w1[burn_in]:.4f}")
```

```{code-cell} python
:tags: [remove-cell]

# The average of the terms is the bound wherever it is below 1.
below = tv < 1
assert np.allclose(terms.mean(axis=0)[below], tv[below])
assert np.allclose(w1_terms.mean(axis=0), w1)
assert 0 < burn_in < len(t) - 1
glue("burn_in", burn_in, display=False)
```

Every state your sampler produces from iteration {glue:text}`burn_in` on has a law within
about 0.01 of the target in total variation: no event has a probability that differs by
more. A different threshold gives a different burn-in, and the choice of 0.01 is yours.

## Checking the bounds against your sampler

The bounds are meant to hold for your own chains. 10,000 independent runs of
`my_sampler` from the same start give the law of $X_t$ directly. For the event
$x_1 < 0$, whose probability under the target is $\tfrac12$, the difference
$|P(X_{t,1} < 0) - \tfrac12|$ is at most $d_{TV}(\pi_t, \pi)$. The first coordinate is a
1-Lipschitz function, so $|E[X_{t,1}]|$, the bias of its mean, is at most the
1-Wasserstein distance:

```{code-cell} python
rng = np.random.default_rng(3)
x = rng.normal(3.0, 1.0, size=(10_000, 5))
every = 25
times = np.arange(0, 1500, every)
below_zero, means, spreads = [], [], []
for _ in times:
    below_zero.append(np.mean(x[:, 0] < 0))
    means.append(x[:, 0].mean())
    spreads.append(x[:, 0].std(ddof=1))
    x = my_sampler(x, every, rng)
gap = np.abs(np.array(below_zero) - 0.5)
gap_error = np.sqrt(0.25 / len(x))  # the largest standard error of a fraction
bias = np.abs(np.array(means))
bias_error = np.array(spreads) / np.sqrt(len(x))
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert np.all(gap <= tv[times] + 3 * np.hypot(tv_error[times], gap_error))
assert np.all(bias <= w1[times] + 3 * np.hypot(w1_error[times], bias_error))
early = (times >= 100) & (times <= 400)
assert np.all(gap[early] > tv[times][early] / 2)
```

The figure compares them. Every measured difference lies below its bound, within three
standard errors of their difference, and from $t = 100$ to $400$, while the chains are
still on their way to the target, the TV bound is less than twice the measured
difference.

## Summary

The kernel's pairs, the lag, the burn-in and the bounds there:

```{code-cell} python
print(f"kernel: {kernel}")
print(f"lag-1 pairs: {len(lag_one)}, median meeting time {np.median(lag_one):.0f}")
print(
    f"pairs with lag {lag}: {len(pairs.meeting_time)}, "
    f"median meeting time {np.median(pairs.meeting_time):.0f}"
)
print(f"burn-in: {burn_in}; TV bound {tv[burn_in]:.4f}, W1 bound {w1[burn_in]:.4f}")
```

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Two panels against the iteration t. Left, on a logarithmic scale, the total
      variation bound with lag 1 stays at 1 for hundreds of iterations, the bound with
      the longer lag falls from 1 earlier, and the measured differences of a
      probability from one half lie below both; a dashed line marks the burn-in.
      Right, the 1-Wasserstein bound and the measured bias of the first coordinate's
      mean, which lies below it.
  figure:
    caption: >-
      Left: the estimated bounds on $d_{TV}(\pi_t, \pi)$ from pairs with lag 1 and with
      the longer lag, with a band of two standard errors for the latter, and the
      measured $|P(X_{t,1} < 0) - \tfrac12|$ from 10,000 runs of the sampler, with two
      standard errors. Right: the bound on the 1-Wasserstein distance from the pairs
      with the longer lag and the measured $|E[X_{t,1}]|$, each with two standard errors. The
      dashed line marks the chosen burn-in.
---
fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
left.plot(t, tv_lag_one, label="TV bound, lag 1")
left.plot(t, tv, label=f"TV bound, lag {lag}")
band = {"alpha": 0.3, "color": "C1"}
left.fill_between(t, np.maximum(tv - 2 * tv_error, 1e-4), tv + 2 * tv_error, **band)
marks = {"fmt": "o", "ms": 2.5, "elinewidth": 0.8, "color": "C2", "label": "measured"}
left.errorbar(times, gap, yerr=2 * gap_error, **marks)
left.set(yscale="log", ylim=(1e-3, 1.5), xlabel="$t$", ylabel="total variation")
right.plot(t, w1, color="C1", label=f"W1 bound, lag {lag}")
right.fill_between(t, np.maximum(w1 - 2 * w1_error, 1e-4), w1 + 2 * w1_error, **band)
right.errorbar(times, bias, yerr=2 * bias_error, **marks)
right.set(yscale="log", ylim=(1e-3, 20), xlabel="$t$", ylabel="1-Wasserstein")
for ax in (left, right):
    ax.axvline(burn_in, color="0.45", linestyle="--")
    ax.set_xlim(0, 1500)
    ax.legend(fontsize=8)
```

No warning was raised. {meth}`~telescopic.CoupledSampler.meeting_times` warns of a pair
still unmet after $10^4$ iterations
([Slow pairs](../reference/warnings.md#slow-pairs-slow)), and none was.
{func}`telescopic.engine.run` emits no warning: it records what its caller would warn
about, so this page checked its pairs itself and found none unmet at the cap. An unmet
pair would make each bound an underestimate.

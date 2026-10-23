---
file_format: mystnb
kernelspec:
  name: python3
---

# Choosing k, m and the lag

The burn-in $k$, the run length $m$ and the lag $L$ change the cost and the variance of
the estimates, never their bias ([Theory](../theory.md)). This page applies the automatic
recipe of Atchadé & Jacob (2024) step by step, then measures what other settings cost.
[Settings](../guide/settings.md) describes the options.

## The pilot

The target is `examples.gaussian(10)`: the Gaussian $\mathcal N(0, \Sigma)$ in 10
dimensions with $\Sigma_{ij} = 0.9^{|i-j|}$, started from $\mathcal N(0, 4I)$. Its
log-density takes a batch of points. {meth}`~telescopic.CoupledSampler.tune` sets the
proposal covariance of a random walk from adaptive Metropolis chains, and freezes the
kernel:

```{code-cell} python
import warnings

import matplotlib.pyplot as plt
import numpy as np
from myst_nb import glue

import telescopic
from telescopic import examples
from telescopic.tuning import choose_k

example = examples.gaussian(10)
sampler = telescopic.CoupledSampler(
    example.log_prob, example.initial, vectorize=True, rng=0
)
sampler.tune()
sampler.kernel
```

{meth}`~telescopic.CoupledSampler.meeting_times` runs 100 new pairs with lag 1 until
they meet, and {func}`telescopic.tuning.choose_k` takes the 99% quantile of $\tau - 1$ as
$k$:

```{code-cell} python
times = sampler.meeting_times()
k = choose_k(times)
glue("k", int(k), display=False)
glue("pilot_late", float(np.mean(times - 1 > k)), display=False)
print(f"k = {k}; pairs with tau - 1 > k: {np.sum(times - 1 > k)} of {len(times)}")
```

{glue:text}`pilot_late:.0%` of the pilot pairs met after $k + 1$, and the survival
function of $\tau - 1$ falls to that fraction at $k$. The recipe then sets the lag to $k$
and $m$ to $10k$.

```{code-cell} python
shifted = np.sort(times - 1)
survival = 1 - np.arange(1, len(shifted) + 1) / len(shifted)
survival[-1] = survival[-2]  # the last step falls to 0, off a log scale
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5))
left.hist(shifted, bins=20)
left.set(xlabel=r"$\tau - 1$", ylabel="pilot pairs")
right.step(shifted, survival, where="post")
right.axhline(0.01, color="grey", linestyle=":")
right.set(yscale="log", xlabel="$t$", ylabel=r"$P(\tau - 1 > t)$")
for ax in (left, right):
    ax.axvline(k, color="grey", linestyle="--", label="$k$")
right.legend();
```

## The recipe

With `k`, `m` and `lag` left at `"auto"`, the first {meth}`~telescopic.CoupledSampler.run`
applies the recipe to the pilot already run:

```{code-cell} python
recipe = sampler.run(500)
recipe.summary()
```

A replicate whose pair meets before $m$ costs $m$ plus its coupled steps
([Theory](../theory.md)), here {glue:text}`recipe_cost:,.0f` kernel applications on
average. The inefficiency, that cost over the $m - k + 1$ averaged states, is
{glue:text}`recipe_inefficiency:.2f`; the theory gives about 10/9 to 11/9 when
$\tau - L \le k$. {glue:text}`recipe_late:.1%` of the pairs met after $k + L$ and carry
a bias correction, against {glue:text}`pilot_late:.0%` of the pilot's pairs after
$k + 1$: the lag changes the law of $\tau - L$.

## Other settings

A sampler keeps the settings of its first run, so each setting below gets a sampler of
its own, with its own seed. They share the tuned kernel and so skip the tuning; with `k`
given, they skip the pilot too. `run_with_warnings` prints the warnings that a run emits
under its name; [Diagnostics](../guide/diagnostics.md) explains them:

```{code-cell} python
def run_with_warnings(sampler, n_replicates, **settings):
    # Runs the sampler and prints the warnings that the run emits.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sampler.run(n_replicates, **settings)
    for warning in caught:
        print(f"{warning.category.__name__}: {warning.message}")
    return result


choices = {
    "lag_1": dict(k=k, m=10 * k, lag=1),
    "m_2k": dict(k=k, m=2 * k, lag=k),
    "k_50": dict(k=50, m=500, lag=1),
}
results = {"recipe": recipe}
for seed, (name, settings) in enumerate(choices.items(), start=1):
    print(f"{name}:")
    other = telescopic.CoupledSampler(
        example.log_prob, example.initial, vectorize=True, kernel=sampler.kernel, rng=seed
    )
    results[name] = run_with_warnings(other, 500, **settings)
```

For each run: the estimate of $\mathbb E[x_0^2] = 1$, the mean cost of a replicate in
kernel applications, the inefficiency, the fraction of pairs that met after $k + L$, the
variance of one replicate's estimate, and that variance times the cost:

```{code-cell} python
def second_moment(x):
    return x[:, 0] ** 2


print(
    f"{'':8}{'k':>5}{'lag':>5}{'m':>6}{'E[x0^2]':>16}{'cost':>8}"
    f"{'ineff.':>8}{'late':>7}{'variance':>10}{'var. x cost':>13}"
)
efficiency, errors = {}, []
for name, result in results.items():
    estimate = result.estimate(second_moment)
    variance = float(np.var(estimate.replicates, ddof=1))
    measured = result.diagnostics()
    efficiency[name] = variance * measured["mean_cost"]
    errors.append(float(abs(estimate.value - 1) / estimate.standard_error))
    print(
        f"{name:8}{result.k:5}{result.lag:5}{result.m:6}"
        f"{estimate.value:9.3f} ± {estimate.standard_error:<5.3f}{measured['mean_cost']:8.0f}"
        f"{measured['inefficiency']:8.2f}{measured['late_fraction']:7.1%}"
        f"{variance:10.3g}{efficiency[name]:13.3g}"
    )
    glue(f"{name}_cost", measured["mean_cost"], display=False)
    glue(f"{name}_inefficiency", measured["inefficiency"], display=False)
    glue(f"{name}_late", measured["late_fraction"], display=False)
    glue(f"{name}_relative", efficiency[name] / efficiency["recipe"], display=False)
glue("largest_error", max(errors), display=False)
```

The estimate farthest from 1 lies {glue:text}`largest_error:.1f` standard errors from it:
the settings change the variance, not the bias. With a budget of $B$ kernel applications,
$B/\text{cost}$ replicates estimate $\mathbb E[x_0^2]$ with variance
$\text{variance} \times \text{cost}/B$, so the last column compares the settings at equal
compute. The recipe's is the lowest:

- **$m = 2k$**: a replicate costs {glue:text}`m_2k_cost:,.0f` kernel applications,
  against the recipe's {glue:text}`recipe_cost:,.0f`, but only $k + 1$ of them give
  averaged states. The inefficiency is {glue:text}`m_2k_inefficiency:.2f`, and variance
  times cost {glue:text}`m_2k_relative:.1f` times the recipe's.
- **Lag 1** has the recipe's $k$ and $m$, and about its cost,
  {glue:text}`lag_1_cost:,.0f` kernel applications, but variance times cost
  {glue:text}`lag_1_relative:.1f` times the recipe's.
- **$k = 50$**: {glue:text}`k_50_late:.0%` of the pairs met after $k + L$ and carry a
  bias correction (telescopic warns above 25% that $k$ may be too small), and variance
  times cost is {glue:text}`k_50_relative:,.0f` times the recipe's.

The variance of lag 1 comes from its late pairs. A difference $\Delta_t$ enters a
replicate's estimate with the weight $v_t/(m - k + 1)$ of the [theory](../theory.md),
and $v_t$ grows by one every $L$ steps from $t = k + L$ to $t = m + L$. A pair that meets
at $\tau = k + L + 100$ carries weights up to $100/(m - k + 1)$ with lag 1, and
$1/(m - k + 1)$ with lag $k$:

```{code-cell} python
for name in ("recipe", "lag_1"):
    result = results[name]
    values = result.estimate(second_moment).replicates
    late = result.meeting_times - result.lag > result.k
    spread = float(np.var(values[late], ddof=1) / np.var(values[~late], ddof=1))
    print(f"{name}: {late.sum()} late pairs, with {spread:.3g} times the others' variance")
    glue(f"{name}_late_count", int(late.sum()), display=False)
    glue(f"{name}_late_spread", spread, display=False)
```

With lag 1, the estimates of the {glue:text}`lag_1_late_count:d` late pairs have
{glue:text}`lag_1_late_spread:.0f` times the variance of the others'; with lag $k$, those
of the {glue:text}`recipe_late_count:d` late pairs have
{glue:text}`recipe_late_spread:.1f` times.

## Distance to stationarity

Each chain of a pair is a Markov chain with the kernel's transitions from $\pi_0$,
whatever the lag, so the runs with lag 1 and lag $k$ bound the same distances
$d_{TV}(\pi_t, \pi)$ ([Theory](../theory.md)), at about the same cost:

```{code-cell} python
t = np.arange(2 * k)
fig, ax = plt.subplots(figsize=(5, 3.5))
for label, result in (("lag 1", results["lag_1"]), (f"lag {k}", recipe)):
    bound = result.tv_bound(t)
    ax.plot(t[bound > 0], bound[bound > 0], label=label)
ax.axvline(k, color="grey", linestyle="--", label="$k$")
ax.set(yscale="log", xlabel="$t$", ylabel=r"TV bound on $X_t$")
ax.legend();
```

```{code-cell} python
lag_one, lag_k = results["lag_1"].tv_bound([k])[0], recipe.tv_bound([k])[0]
print(f"at t = k: lag 1, {lag_one:.3f}; lag {k}, {lag_k:.3f}")
glue("tv_lag_k", float(lag_k), display=False)
glue("tv_ratio", float(lag_one / lag_k), display=False)
```

At $t = k$ the lag-$k$ bound, {glue:text}`tv_lag_k:.3f`, is about the fraction of the
recipe's pairs that met after $k + L$, and the lag-1 bound is
{glue:text}`tv_ratio:.0f` times larger: lag $k$ gives the sharper bound (Biswas, Jacob &
Vanetti 2019).

## Partial "auto"

Any of `k`, `m` and `lag` can be given, with the others left at `"auto"`. A given value
is kept, a missing lag is $\max(1, k)$ and a missing $m$ is $10k$, and the pilot runs
only when `k` is `"auto"`. With `k` given, no pilot runs:

```{code-cell} python
given_k = telescopic.CoupledSampler(
    example.log_prob, example.initial, vectorize=True, kernel=sampler.kernel, rng=4
)
given_k.run(20, k=300)
print("pilot:", given_k.pilot)
given_k.settings
```

With `m` given, the pilot chooses $k$, and the lag follows it:

```{code-cell} python
given_m = telescopic.CoupledSampler(
    example.log_prob, example.initial, vectorize=True, kernel=sampler.kernel, rng=5
)
given_m.run(20, m=5000)
glue("given_m_k", int(given_m.settings.k), display=False)
print("pilot pairs:", len(given_m.pilot))
given_m.settings
```

This sampler's own pilot chose {glue:text}`given_m_k:d` for $k$, against
{glue:text}`k:d` above: $k$ varies between pilots of 100 pairs, which changes the cost
and the variance, not the bias. On later runs of a sampler, `"auto"` keeps the first
run's settings, and another value raises:

```{code-cell} python
try:
    given_k.run(20, k=100)
except ValueError as error:
    print(error)
```

{func}`telescopic.sample` takes the same `k`, `m` and `lag`.

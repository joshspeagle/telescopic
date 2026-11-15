---
file_format: mystnb
kernelspec:
  name: python3
---

# Choosing k, m and the lag

This page applies the automatic recipe of Atchadé & Jacob (2024) step by step on a
correlated Gaussian, then measures the cost and the variance that other settings give.
[Settings](../guide/settings.md) describes `k`, `m` and the {term}`lag`.

## The pilot

The target is `examples.gaussian(10)`: the Gaussian $\mathcal N(0, \Sigma)$ in 10
dimensions with $\Sigma_{ij} = 0.9^{|i-j|}$, started from $\mathcal N(0, 4I)$. Its
log-density takes a batch of points. {meth}`~telescopic.CoupledSampler.tune` sets the
proposal covariance of a random walk from adaptive Metropolis chains, and freezes the
kernel:

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

import telescopic

example = telescopic.examples.gaussian(10)
sampler = telescopic.CoupledSampler(
    example.log_prob, example.initial, vectorize=True, rng=0
)
sampler.tune()
sampler.kernel
```

{meth}`~telescopic.CoupledSampler.meeting_times` runs 100 new {term}`pairs <pair>` with lag 1 until
they meet, and {func}`telescopic.tuning.choose_k` takes the 99% quantile of $\tau - 1$ as
$k$:

```{code-cell} python
times = sampler.meeting_times()
k = telescopic.tuning.choose_k(times)
print(f"k = {k}; pairs with tau - 1 > k: {np.sum(times - 1 > k)} of {len(times)}")
```

The recipe then sets the lag to $k$ and $m$ to $10k$. The pilot's {term}`meeting times <meeting time>`, with $k$
dashed and the level 0.01 dotted:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Left, a histogram of the pilot's meeting times less 1; right, their survival
      function on a logarithmic scale; both with a dashed vertical line at k, and
      the right with a dotted horizontal line at 0.01.
  figure:
    caption: >-
      The meeting times of the pilot pairs, less the lag of 1: their histogram
      (left) and survival function on a log scale (right), with $k$ dashed and the
      level 0.01 dotted.
---
shifted = np.sort(times - 1)
survival = 1 - np.arange(1, len(shifted) + 1) / len(shifted)
survival[-1] = survival[-2]  # the last step falls to 0, off a log scale
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5))
left.hist(shifted, bins=20)
left.set(xlabel=r"$\tau - 1$", ylabel="pilot pairs")
right.step(shifted, survival, where="post")
right.axhline(0.01, color="gray", linestyle=":")
right.set(yscale="log", xlabel="$t$", ylabel=r"$P(\tau - 1 > t)$")
for ax in (left, right):
    ax.axvline(k, color="gray", linestyle="--", label="$k$")
right.legend();
```

## The recipe

With `k`, `m` and `lag` left at `"auto"`, the first {meth}`~telescopic.CoupledSampler.run`
applies the recipe to the pilot already run:

```{code-cell} python
recipe = sampler.run(500)
recipe.summary()
```

The footer gives the cost of a {term}`replicate`, its {term}`relative cost`, and the fraction of pairs
that met after $k + L$, the ones that carry a bias correction
([The summary](../reference/result-fields.md#the-summary) defines each entry). That fraction need not
equal the pilot's fraction after $k + 1$: the lag changes the law of $\tau - L$.

## Other settings

Each setting below gets a sampler of its own, with its own seed, since a sampler keeps
the settings of its first run. The samplers share the tuned kernel and so skip the
tuning; with `k` given, they skip the pilot too. A run warns as it ends when a condition
of [Warnings and errors](../reference/warnings.md) holds:

```{code-cell} python
choices = {
    "lag_1": dict(k=k, m=10 * k, lag=1),
    "m_2k": dict(k=k, m=2 * k, lag=k),
    "k_50": dict(k=50, m=500, lag=1),
}
results = {"recipe": recipe}
for seed, (name, settings) in enumerate(choices.items(), start=1):
    other = telescopic.CoupledSampler(
        example.log_prob, example.initial, vectorize=True, kernel=sampler.kernel, rng=seed
    )
    results[name] = other.run(500, **settings)
```

For each run: the estimate of $\mathbb E[x_0^2] = 1$, the mean cost of a replicate in
kernel applications, the relative cost, the fraction of pairs that met after $k + L$,
the {term}`inefficiency` ([Theory](../theory.md#cost)), and the kinds of the warnings that
apply, from {func}`telescopic.diagnostics.issues`:

```{code-cell} python
def second_moment(x):
    return x[:, 0] ** 2


print(
    f"{'':8}{'k':>5}{'lag':>5}{'m':>6}{'E[x0^2]':>16}{'cost':>7}"
    f"{'rel.':>6}{'late':>7}{'ineff.':>10}  issues"
)
rows = {}
for name, result in results.items():
    estimate = result.estimate(second_moment)
    row = result.diagnostics()
    row["inefficiency"] = np.var(estimate.replicates, ddof=1) * row["mean_cost"]
    row["issues"] = [issue.kind for issue in telescopic.diagnostics.issues(result)]
    row["estimate"] = estimate
    rows[name] = row
    print(
        f"{name:8}{result.k:5}{result.lag:5}{result.m:6}"
        f"{estimate.value:9.3f} ± {estimate.standard_error:<5.3f}{row['mean_cost']:7.0f}"
        f"{row['relative_cost']:6.2f}{row['late_fraction']:7.1%}"
        f"{row['inefficiency']:10.3g}  {row['issues']}"
    )
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
recipe_row = rows["recipe"]
others = [row for name, row in rows.items() if name != "recipe"]
assert all(
    abs(row["estimate"].value - 1) < 2 * row["estimate"].standard_error
    for row in rows.values()
)
assert all(row["inefficiency"] > recipe_row["inefficiency"] for row in others)
assert rows["m_2k"]["mean_cost"] < recipe_row["mean_cost"]
assert rows["m_2k"]["relative_cost"] > recipe_row["relative_cost"]
assert abs(rows["lag_1"]["mean_cost"] / recipe_row["mean_cost"] - 1) < 0.05
assert rows["lag_1"]["variance_share"] > 0.5
assert rows["k_50"]["late_fraction"] > 0.5
assert rows["k_50"]["inefficiency"] > 100 * recipe_row["inefficiency"]
assert "late" in rows["k_50"]["issues"]
```

Drawn, with the weight that each setting gives the differences of a late pair, as the
next section explains, and with 95% intervals from 1,000 bootstrap resamples of each
run's replicates:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Left: the weight of a difference against t for lag 1, a ramp from k + 1 that
      reaches 1 at m + 1, and for lag k, a staircase that rises every k steps from
      2k and is flat from m + k, on a log scale, with a late pair's meeting time
      marked. Right: the
      inefficiency on a log scale against the relative cost for four runs, the
      recipe lowest, then m = 2k and lag 1, and k = 50 far above the others.
  figure:
    caption: >-
      Left: the weight $v_t/(m - k + 1)$ of the difference $\Delta_t$ against $t$ at
      the recipe's $k$ and $m = 10k$, on a log scale, for $L = 1$, a ramp, and for
      $L = k$, a staircase; the dashed line marks a late pair that meets at
      $\tau = k + L + 100$ with $L = 1$, whose differences up to $\tau - 1$ enter
      its estimate with these weights. Right: the inefficiency, on a log scale,
      against the relative cost, of the four runs above, with 95% bootstrap
      intervals. The recipe's is the lowest.
---
from telescopic.estimators import correction_coefficients

fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3), width_ratios=(1.2, 1))
m = 10 * k
t = np.arange(0, m + 2 * k + 1)
for lag, color in ((1, "C0"), (k, "C1")):
    weight = correction_coefficients(t, k, m, lag, 1)
    left.step(t, weight, where="post", color=color, label=f"$L = {lag}$")
late = k + 1 + 100
left.axvline(late, color="gray", linestyle="--", linewidth=0.8)
left.annotate(r"$\tau = k + 101$", (late + 40, 0.01), color="gray")
left.set(yscale="log", xlabel="$t$", ylabel="weight of $\\Delta_t$")
left.legend(loc="center right")

bootstrap = np.random.default_rng(0)
labels = {"recipe": "recipe", "m_2k": "$m = 2k$", "lag_1": "lag 1", "k_50": "$k = 50$"}
for name, result in results.items():
    values = result.estimate(second_moment).replicates
    costs = result.costs[np.isfinite(result.meeting_times)]
    draws = bootstrap.integers(len(values), size=(1000, len(values)))
    resampled = np.var(values[draws], axis=1, ddof=1) * costs[draws].mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    x, y = rows[name]["relative_cost"], rows[name]["inefficiency"]
    color = "C2" if name == "recipe" else "C0"
    right.errorbar(x, y, yerr=[[y - low], [high - y]], fmt="o", color=color, capsize=3)
    right.annotate(
        labels[name], (x, y), xytext=(8, 0), textcoords="offset points", va="center",
        color=color,
    )
right.set(yscale="log", xlabel="relative cost", ylabel="inefficiency");
```

Every estimate lies within two standard errors of 1: the settings change the variance,
not the bias ([Settings](../guide/settings.md)).
The inefficiency compares settings at equal compute, and the recipe's is the lowest:

- **$m = 2k$** costs less per replicate, but averages only $k + 1$ states: its relative
  cost and its inefficiency are higher than the recipe's.
- **Lag 1** has the recipe's $k$ and $m$, and about its cost, but a higher inefficiency,
  and one replicate holds most of the variance of an estimate.
- **$k = 50$** leaves most pairs to meet after $k + L$, so that most replicates carry a
  bias correction: its inefficiency is orders of magnitude above the recipe's, and the
  run warns that $k$ may be too small.

The variance of lag 1 comes from its {term}`late pairs <late pair>`. A difference $\Delta_t$ enters a
replicate's estimate with the weight $v_t/(m - k + 1)$ of the
[theory](../theory.md#the-time-averaged-estimator), and $v_t$ grows by one every $L$
steps from $t = k + L$ to $t = m + L$. A pair that meets at $\tau = k + L + 100$ carries
weights up to $100/(m - k + 1)$ with lag 1, and $1/(m - k + 1)$ with lag $k$:

```{code-cell} python
spreads = {}
for name in ("recipe", "lag_1"):
    result = results[name]
    values = result.estimate(second_moment).replicates
    late = result.meeting_times - result.lag > result.k
    spreads[name] = np.var(values[late], ddof=1) / np.var(values[~late], ddof=1)
    print(f"{name}: {late.sum()} late pairs, with {spreads[name]:.3g} times the others' variance")
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert spreads["lag_1"] > 10
assert 0.5 < spreads["recipe"] < 2
```

With lag 1, the estimates of the late pairs vary far more than the others'; with lag
$k$, about as much.

## Distance to stationarity

The lag does not change the law of $X_t$, so the runs with lag 1 and lag $k$ bound the
same distance between it and the target
([Theory](../theory.md#distance-to-stationarity)), at about the same cost:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Two lines on a logarithmic scale, one for lag 1 and one for lag k, each
      falling from 1 as t grows, with a dashed vertical line at k.
  figure:
    caption: >-
      The estimated upper bound on the total variation distance between the law of
      $X_t$ and the target, from the run with lag 1 and the run with lag $k$, with
      $k$ dashed.
---
t = np.arange(2 * k)
fig, ax = plt.subplots(figsize=(5, 3.5))
for label, result in (("lag 1", results["lag_1"]), (f"lag {k}", recipe)):
    bound = result.tv_bound(t)
    ax.plot(t[bound > 0], bound[bound > 0], label=label)
ax.axvline(k, color="gray", linestyle="--", label="$k$")
ax.set(yscale="log", xlabel="$t$", ylabel=r"TV bound on $X_t$")
ax.legend();
```

```{code-cell} python
lag_one, lag_k = results["lag_1"].tv_bound([k])[0], recipe.tv_bound([k])[0]
print(f"at t = k: lag 1, {lag_one:.3f}; lag {k}, {lag_k:.3f}")
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert lag_one > 10 * lag_k
```

At $t = k$ the lag-1 bound is far larger than the lag-$k$ bound: lag $k$ gives the
sharper bound, as in Biswas, Jacob & Vanetti (2019).

## Partial "auto"

Any of `k`, `m` and `lag` can be given, with the others left at `"auto"`, by the rule
that [Settings](../guide/settings.md) states. With `k` given:

```{code-cell} python
given_k = telescopic.CoupledSampler(
    example.log_prob, example.initial, vectorize=True, kernel=sampler.kernel, rng=4
)
given_k.run(20, k=300)
print("pilot:", given_k.pilot)
given_k.settings
```

With `m` given:

```{code-cell} python
given_m = telescopic.CoupledSampler(
    example.log_prob, example.initial, vectorize=True, kernel=sampler.kernel, rng=5
)
given_m.run(20, m=5000)
print(f"pilot pairs: {len(given_m.pilot)}; k = {given_m.settings.k}, against {k} above")
given_m.settings
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert given_m.settings.k != k
```

This sampler's pilot and the first one chose different values of $k$: $k$ varies
between pilots of 100 pairs. Another value on a later run of a sampler raises:

```{code-cell} python
try:
    given_k.run(20, k=100)
except ValueError as error:
    print(error)
```

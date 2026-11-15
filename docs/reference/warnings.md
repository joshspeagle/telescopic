---
file_format: mystnb
kernelspec:
  name: python3
---

# Warnings and errors

telescopic warns when a run's error bars may be unreliable.

```{code-cell} python
:tags: [remove-cell]

import warnings

import telescopic
from telescopic import examples
```

## Warnings

A run emits a
{class}`telescopic.ConvergenceWarning <telescopic.warnings.ConvergenceWarning>` for each
condition below that applies, at the time this table gives:

| Warning | Emitted |
|---|---|
| `censored`, `late`, `variance`, `tail` | when the run ends, about the result of all its {term}`replicates <replicate>` so far |
| `tuning`, `wide` | when the kernel is tuned, before any {term}`pair` runs |
| `slow` | when a pair passes the [threshold](#slow-pairs-slow) in this process, or when the run ends on a pool |

{meth}`telescopic.Result.check`
emits the warnings that apply to any result, and {func}`telescopic.diagnostics.issues`
returns them, without warning, as a list of {class}`telescopic.diagnostics.Issue` with
the `kind` named in each heading below and a `message`, which ends with the remedy in a
few words. An empty list means only that none of them holds, not that the run is right.
They are heuristics, with thresholds set so that clean Gaussian runs at the default
settings rarely trigger them; the measured rates below show how often they fire
elsewhere.

### Censored pairs (`censored`)

**Trigger:** a pair reached `max_iter` without meeting. Without a cap, the default, every
pair runs until it meets. With `max_iter`, a pair still unmet when $t$ reaches it is
censored: its {term}`meeting time` is `inf`, and it is left out of the estimates, of $R$ and of
the {term}`signed measure`. `result.n_censored` counts them. **Meaning:** leaving them out biases
the estimates, since they are the replicates whose chains take longest to meet.
**Remedy:** a new sampler with a larger `max_iter`, or none; a sampler keeps the cap of
its first run. When {term}`censored pairs <censored pair>` started where the target density is zero, the warning
says how many and advises an initial distribution inside the support first: such a pair
may never meet, so a run without a cap could wait forever.

### Late pairs (`late`)

**Trigger:** more than a quarter of the pairs meet after `k + lag` (`late_fraction` above
0.25). **Meaning:** `k` may be too small, so that many replicates carry bias corrections,
which make the variance high. **Remedy:** a new sampler with a larger `k`
([Settings](../guide/settings.md)).

### One replicate dominating the variance (`variance`)

**Trigger:** with at least 100 replicates, one replicate holds more than half of the
sample variance of the estimate of some coordinate's mean or variance.
`result.diagnostics()` reports the largest share as `variance_share`, the estimate that
holds it as `variance_share_of`, and the index of the replicate that holds it, which
the warning names, as `variance_share_replicate`, whether or not the warning fires.
**Meaning:** heavy tails or rare excursions make the error bars unreliable. A replicate
whose chain is still on its way to the target at `k` does this, as from a start far
wider than the target ([Initial distributions](../guide/initial-distribution.md#choosing-one)),
which the `wide` warning looks for ([A start far wider than the target](#a-start-far-wider-than-the-target-wide)), and so does a
replicate that found a mode that the others missed. **Remedy:** check the initial
distribution, and look for a mode that the other replicates missed, starting with the
one named; its row in the `replicates` of an estimate is its position among the
replicates that met. The warning looks at the first two moments of each coordinate only;
{func}`telescopic.diagnostics.variance_share` applied to `result.estimate(h).replicates`
checks a function of your own.

### Heavy-tailed meeting times (`tail`)

**Trigger:** with at least 500 replicates, Hill's estimate of the {term}`tail index` of
$\tau - L + 1$ ({func}`telescopic.diagnostics.hill`), from its $\lfloor R/10 \rfloor$
largest values, is below 2. `result.diagnostics()` reports it as `tail_index` whether or
not the warning fires. Ties among the earliest meeting times can leave the estimate
`nan` and the warning silent, and a heavy tail behind them undetected
({func}`telescopic.diagnostics.hill` gives the rule). The warning
also sees only a tail that shows up in the run: a heavy tail carried by well under one
pair in $R$, as with an independence proposal whose weights are unbounded
([Independence](../guide/kernels.md#independence)), leaves it silent.
**Meaning:** unbiasedness needs moments of the meeting time, and valid error bars need a
finite variance of the estimates ([Theory](../theory.md#guarantees-and-their-limits)).
Meeting times with a polynomial tail can break both: a tail index below 2 means an
infinite variance of the meeting times, the estimates may have one too, and then their
error bars are not reliable at any number of replicates. **Remedy:** look at the
survival function of `plot_meeting_times`, whose tail is straight on its log scale for
geometric meeting times, and at the kernel ([Kernels](../guide/kernels.md)).

### Disagreeing tuning chains (`tuning`)

**Trigger:** the chains that tuned the kernel disagree: their Gelman-Rubin $\hat R$
(`result.tuning.rhat`) exceeds $\sqrt 2$ in some coordinate, so their means spread more
widely than the chains themselves, or is NaN in some coordinate, where all four chains
sat at one value; the warning names those coordinates. **Meaning:** an $\hat R$ above
$\sqrt 2$ says the chains sit in different modes of the target, or have not converged.
Several modes make the meeting times heavy-tailed, and pairs that never leave one mode
miss the others. A NaN says the pilot never moved in that coordinate, so the tuned
kernel's variance there is about $10^{-10}$ and its chains barely move in it either;
this happens where float64 cannot resolve the target's scale at the coordinate's values,
such as a scale of $10^3$ at $2^{70}$, where floats are $2^{18}$ apart. **Remedy:** for
several modes, unless the initial distribution already covers every mode, one that does
([Initial distributions](../guide/initial-distribution.md#choosing-one)); for a NaN, a target written in
coordinates centered near 0, or the kernel's parameters given instead of tuned
([Kernels](../guide/kernels.md)). Chains that agree can still all sit in one mode, and the
warning is then silent ([Tuning](../guide/kernels.md#tuning)).

### A start far wider than the target (`wide`)

**Trigger:** the tuning pilot's initial width, `result.tuning.initial_width`, how many
times wider than the target the initial distribution is
({attr}`telescopic.tuning.PilotRun.initial_width` defines it), exceeds 30. It costs no
evaluation of the target. The warning is emitted whenever tuning runs, from
{meth}`~telescopic.CoupledSampler.tune`, {meth}`~telescopic.CoupledSampler.meeting_times`
or the first {meth}`~telescopic.CoupledSampler.run`. {func}`telescopic.diagnostics.issues`
and {meth}`telescopic.Result.check` list it unless the run's `k` is larger than the `k`
that the meeting-time pilot's rule gave when the run's settings were fixed, or no such
pilot had run then: a larger `k` is its remedy. A kernel given with every parameter has
no tuning pilot, and the warning is then silent. **Meaning:** the
automatic `k`, the 99% quantile of the pilot's meeting times, leaves about one pair in a
hundred meeting while its chain is still on its way to the target; from a start this
wide, those replicates can dominate the variance of the estimates
([Initial distributions](../guide/initial-distribution.md#choosing-one)). **Remedy:**
first a narrower initial distribution; failing that, a larger `k` given by hand; a
confirmation round of pilot pairs at lag `k` is an option, which does not always raise
`k` ([A start far wider than the target](../guide/settings.md#a-start-far-wider-than-the-target)).
A geometric mean reads a start that is far too wide in only a few of many coordinates
as moderately wide, and the warning can then stay silent; a start offset from the
target but no wider does not trigger it either.

### Slow pairs (`slow`)

**Trigger:** a pair still unmet after $\max(10^4, 10 \max(L, m))$ iterations; the
sampler names the replicate in one warning per run. **Meaning:** the meeting times may be
heavy-tailed, and the run waits for that pair: a first Ctrl-C starts no new replicate
but lets the pairs in progress finish, this one too, and a second stops the run without
a result ([Parallel runs](../guide/parallel.md#interrupting-a-run)). **Remedy:** look at the
kernel and the initial distribution. In the {term}`meeting-time pilot`, whose own cap is on
[Settings](../guide/settings.md#the-automatic-pipeline), the threshold is $10^4$.

## A run that warns

Running the banana target from its wide default start with `k`, `lag` and `m` set by hand triggers two warnings:

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

Repeated runs measured how often each warning fires. All used vectorized
log-densities, and the first two targets `sample()`'s default settings:

- {func}`telescopic.examples.gaussian` in 10 dimensions, from $\mathcal N(0, 4I)$: no
  warning in 100 runs at $R = 100$ and 50 runs at $R = 500$.
- The banana ({func}`telescopic.examples.banana`) from $U[-5, 5]^2$: at $R = 100$, the
  variance warning in 37 of 200 runs, whose 95% intervals of $E[x_0^2]$ covered the truth
  in 95% of them against 97% in the other runs; at $R = 500$, the variance warning in 4
  of 50 runs and the tail warning in 14.
- The `wide` warning, over 200 tuning seeds of each case, with the random walk's
  tuning pilot. From the default starts of {func}`~telescopic.examples.gaussian` in 1,
  2, 10 and 50 dimensions, {func}`~telescopic.examples.banana`,
  {func}`~telescopic.examples.mixture`, {func}`~telescopic.examples.funnel` and
  {func}`~telescopic.examples.pumps`, and from the [Quickstart](../start/quickstart.md)'s
  start ten times wider than the line fit's posterior, it fired in none of 1,800 runs,
  whose largest initial width was 14.1 (the pump model). From starts
  $\mathcal N(\mu, w^2\Sigma)$, with the exact mean $\mu$ and covariance $\Sigma$ of the
  Gaussians in 1, 10 and 50 dimensions with correlation 0.9, the line fit and the
  banana, it fired in none of 1,000 runs at $w = 10$, whose largest width was 21.5, and in
  887 of 1,000 at $w = 100$: in every run on the four targets other than the banana, and in
  87 of 200 on the banana, whose tuning chains often had not converged, so that their
  spread was itself wide. The `tuning` warning fired in 112 of the 113 runs that the
  `wide` warning missed. The threshold of 30 was chosen from these runs and then
  checked on new seeds: on 500 runs each of the cases whose width had passed 10, it fired
  in none from the pump model's and the line fit's default starts and from the 10-fold
  starts of the Gaussians and the line fit, and in 1 of 500 from the banana's 10-fold
  start, whose width there reached 38.3; on 200 runs of each 100-fold start, in every
  run off the banana and in 69 of 200 on it, where the `tuning` warning fired in all but
  one of the others. These runs used telescopic at commit 2707eb9 and NumPy 2.5.3, on the
  machine below.
- The bimodal mixture of Jacob, O'Leary & Atchadé (2020, §5.1),
  {func}`telescopic.examples.mixture`, from $\mathcal N(10, 1)$, with a random walk of
  proposal variance 1, `k = 50`, `m = 500`, `lag = 1` and $R = 1000$: the 95% intervals
  of $P(x > 3)$ missed the truth in 33 of 50 runs, none of which warned. The 17 runs whose
  replicates found the other mode raised the variance warning, and their intervals
  covered the truth.
- The mixture $0.7\,\mathcal N(-6e_1, I) + 0.3\,\mathcal N(6e_1, I)$ in 10 and 32
  dimensions, from $\mathcal N(0, 100I)$, with
  `kernel=Mixture([RandomWalk(), Independence()], [0.8, 0.2])` tuned automatically and
  $R = 400$: the estimate of $P(x_0 < 0)$ lay more than 5 standard errors from 0.7 in 14
  of 20 runs in 10 dimensions and 18 of 20 in 32. Over these runs and others on the same
  target, with modes 24 apart or other seeds, the tuning warning fired on none of the 191
  runs whose estimate was wrong, only on runs whose estimate was right.

Setup of these runs: telescopic at commit d014ac1, Python 3.13.18, NumPy 2.5.4 and SciPy
1.18.1, `OMP_NUM_THREADS=1`, on an Intel Xeon processor at 2.10 GHz with 4 logical cores.

## A run that never left one mode

The last two cases show the limit of every diagnostic here: a run whose pairs all met in
one mode cannot be detected from its own output, since nothing in its meeting times or
estimates shows the mode that no replicate visited. The remedy is an initial
distribution that covers every mode ([Initial distributions](../guide/initial-distribution.md#choosing-one)),
and a kernel that reaches every mode from it, which an automatically tuned
{class}`~telescopic.kernels.Independence` proposal may not
([Independence](../guide/kernels.md#independence)).
[When unbiased is not enough](../tutorials/failure-modes.md) works through such runs and
shows two checks that help, and a case where runs with different seeds all agree on a
wrong estimate.

## Warning classes

{class}`telescopic.TelescopicWarning <telescopic.warnings.TelescopicWarning>`, a subclass
of `UserWarning`, is the base class of telescopic's warnings. A
{class}`telescopic.ConvergenceWarning <telescopic.warnings.ConvergenceWarning>` says that
a result may be unreliable: the conditions above. An
{class}`telescopic.InputWarning <telescopic.warnings.InputWarning>` says that an input is
legal but questionable: replicates or pilot pairs whose initial draws have zero target
density, gradients taken as 0 because they were not finite where the log-density is
([Special values](../guide/log-density.md#special-values)), a gradient that disagrees with differences of
the log-density when a sampler is built, or a check of it that could compare nothing
([Checking the gradient](../guide/log-density.md#checking-the-gradient)), a run that, with its
first estimates, would take more than 1 GB ([Settings](../guide/settings.md#memory-and-thinning)),
and a resumed result whose kernel, given or tuned again, differs from the result's
([Continuing a run](../guide/save-load-merge.md#continuing-a-run)).
Each warning points at the line of your code that called telescopic.

The filters of the `warnings` module select them by class:
`warnings.simplefilter("ignore", telescopic.InputWarning)` silences the input warnings,
and `"error"` turns warnings into exceptions. Under such a filter, `sample` or `run`
raises at its first warning instead of returning its result:

```{code-cell} python
:tags: [wrap-output]

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
:tags: [wrap-output]

try:
    short.check(strict=True)
except RuntimeError as error:
    print(error)
```

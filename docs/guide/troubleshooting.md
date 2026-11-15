---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  merge_streams: true
---

# Troubleshoot chains that do not meet

This page is indexed by what a run shows: the text of an error or a warning, or a result
that looks wrong. Each entry gives the cause, a check in code and the fix, and links to
the page that holds each fact. [Warnings and errors](../reference/warnings.md) gives the
threshold and the meaning of every warning.

```{diagram} remedies
Each condition that a run reports, grouped by when it arises, with the warning or error
that it raises and the remedy that the message names. A condition in the summary raises
its warning only past a threshold, which
[Warnings and errors](../reference/warnings.md) gives with each warning.
```

The checks below run on a sampler and a result of a correlated Gaussian in two
dimensions, which meet quickly:

```{code-cell} python
import numpy as np

import telescopic
from telescopic import examples, kernels

gaussian = examples.gaussian(2)
sampler = telescopic.CoupledSampler(
    gaussian.log_prob, gaussian.initial, vectorize=True, rng=5
)
result = sampler.run(100)
```

## Before any replicate

### "k cannot be chosen"

The run raises `ValueError` after the {term}`meeting-time pilot`, before any
{term}`replicate` runs. The message is that of {func}`telescopic.tuning.choose_k`, here
given pilot meeting times of which two in 100 are censored:

```{code-cell} python
:tags: [wrap-output]

try:
    telescopic.tuning.choose_k(np.r_[np.full(98, 50.0), np.inf, np.inf])
except ValueError as error:
    print("ValueError:", error)
```

When MALA or HMC tuned their step size, a note follows the message:

> with MALA or HMC, an automatic step size can fail to couple: try a step_size checked on
> pilot pairs, or, for HMC with jitter=0 and a given n_steps or cov, the default jitter
> (When the automatic step fails, in the guide 'Check and set a step size')

When some pilot {term}`pairs <pair>` started where the target density is zero, the
message begins instead "k cannot be chosen: too many pilot pairs never met within their
cap, and" and says how many.

**Cause.** The pilot's pairs run until they meet or reach the pilot's cap, and `k` is a
quantile of their {term}`meeting times <meeting time>`, which cannot be chosen when too
many are censored ([The automatic pipeline](settings.md#the-automatic-pipeline)). The
usual causes are an automatic step size of MALA or HMC that does not couple
([When the automatic step fails](step-size.md#when-the-automatic-step-fails)), a target
with several modes from a start that covers them, and a start outside the support of
the target.

**Check.** Pilot pairs with a cap of your own, after tuning, show how many stay unmet and
how long the others take, and the tuning chains' $\hat R$ shows modes that the chains
disagree on:

```{code-cell} python
tau = sampler.meeting_times(100, max_iter=5000)
print("unmet:", np.isinf(tau).sum(), "median:", np.median(tau))
print("R-hat:", sampler.tuning.rhat)
```

**Fix.** For MALA and HMC, a `step_size` checked on pilot pairs
([Checking a step size](step-size.md#checking-a-step-size)). For several modes or a start
outside the support, an initial distribution that covers every mode and lies inside the
support ([Choosing one](initial-distribution.md#choosing-one)). Giving `k` skips the
pilot ([The automatic pipeline](settings.md#the-automatic-pipeline)), but the pairs that
the pilot could not bring together then make the run slow or warn.

### "the tuning chains disagree"

The tuning warning comes before the pilot and the replicates; it reads "the tuning chains
disagree (R-hat up to ...)" or "the tuning chains never moved in coordinates ...".
[Disagreeing tuning chains](../reference/warnings.md#disagreeing-tuning-chains-tuning)
gives its cause, its check, `result.tuning.rhat`, and its fix.

## While the replicates run

### "has not met after ... iterations", or a run that seems to hang

A pair that has not met at a threshold raises one `ConvergenceWarning` per run while it
runs. For a replicate it reads "replicate ... has not met after ... iterations, and the
run waits for it", followed by what Ctrl-C does and the remedy; for a pilot pair, as here
with a random walk whose proposals are far too small to meet:

```{code-cell} python
:tags: [wrap-output]

stuck = telescopic.CoupledSampler(
    gaussian.log_prob,
    gaussian.initial,
    vectorize=True,
    kernel=kernels.RandomWalk(gaussian.cov, scale=1e-8),
    rng=5,
)
stuck_tau = stuck.meeting_times(1, max_iter=10_001)
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the paragraph above.
assert np.isinf(stuck_tau).all()
```

A pair whose chain started where the target density is zero ends the message with
"it started where the target density is zero, so draw the initial distribution inside
the support of the target", and such a start also raises an `InputWarning` that begins
"replicates start where the target density is zero".

**Cause.** Heavy-tailed meeting times, from a kernel that mixes slowly or a start far from
the target, or a chain that cannot leave a region of zero density
([Slow pairs](../reference/warnings.md#slow-pairs-slow)). A run without `max_iter` waits
for every pair, so a pair that never meets keeps the run waiting.

**Check.** `progress=True` shows the replicates finished as the run goes
([Progress](parallel.md#progress)). Pilot pairs with a cap, as in the previous entry,
show the tail of the meeting times without waiting for it, and
{meth}`~telescopic.Result.plot_meeting_times` draws their survival function for a result
([Meeting times](meeting-times.md#meeting-times)).

**Fix.** Look at the kernel ([Choose a kernel](kernels.md)) and the initial distribution
([Choosing one](initial-distribution.md#choosing-one)). A first Ctrl-C returns the
replicates that started ([Interrupting a run](parallel.md#interrupting-a-run)).
`max_iter` bounds the wait, but censors the pairs still unmet at it, which biases the
estimates ([Censored pairs](../reference/warnings.md#censored-pairs-censored)).

### MALA or HMC meets far more slowly than the random walk

A gradient kernel whose pairs take much longer to meet than the random walk's, or that
raises the note above, often has an automatic step that does not couple
([When the automatic step fails](step-size.md#when-the-automatic-step-fails)).

**Check.** Pilot pairs with a given step size and covariance run no tuning; compare the
unmet pairs and the mean meeting time over a few step sizes:

```{code-cell} python
mala = telescopic.CoupledSampler(
    gaussian.log_prob,
    gaussian.initial,
    grad=gaussian.grad,
    vectorize=True,
    kernel=kernels.MALA(0.5, cov=gaussian.cov),
    rng=5,
)
mala_tau = mala.meeting_times(100, max_iter=3000)
print("unmet:", np.isinf(mala_tau).sum(), "mean:", mala_tau.mean())
```

**Fix.** The step size with the lowest cost of a pair
([Checking a step size](step-size.md#checking-a-step-size)), and for HMC the default
`jitter` ([HMC](kernels.md#hmc)).

## When the run ends

The warnings below apply to the result of all the replicates; the summary lists them, and
{func}`telescopic.diagnostics.issues` returns them. Each message begins as in the first
column, with the numbers of the run in place of the dots:

| The message begins | Warning |
|---|---|
| "... of ... pairs reached max_iter = ... without meeting" | [`censored`](../reference/warnings.md#censored-pairs-censored) |
| "...% of the pairs met after k + lag, so k may be too small" | [`late`](../reference/warnings.md#late-pairs-late) |
| "replicate ... holds ...% of the variance of" | [`variance`](../reference/warnings.md#one-replicate-dominating-the-variance-variance) |
| "the meeting times look heavy-tailed (tail index" | [`tail`](../reference/warnings.md#heavy-tailed-meeting-times-tail) |
| "a pair was still unmet after ... iterations" | [`slow`](../reference/warnings.md#slow-pairs-slow) |

**Check.** The values behind them are in `result.diagnostics()`
([The diagnostics of a run](../reference/result-fields.md#the-diagnostics-of-a-run)), and
the warnings that apply in {func}`telescopic.diagnostics.issues`, empty here:

```{code-cell} python
values = result.diagnostics()
print({name: values[name] for name in ("censored", "late_fraction", "variance_share")})
telescopic.diagnostics.issues(result)
```

```{code-cell} python
:tags: [remove-cell]

# The claim of the paragraph above.
assert telescopic.diagnostics.issues(result) == []
```

**Fix.** Each warning's entry in [Warnings and errors](../reference/warnings.md) gives its
remedy.

## In the estimates

### A negative variance, or a standard deviation of zero

`result.var()` returns a negative value, or the `std` column of the summary shows 0.

**Cause.** The variances are {term}`unbiased`, and an unbiased estimate of a variance can
be negative; `std` is $\sqrt{\max(0, \widehat{\mathrm{Var}})}$
([Covariances and variances](estimates.md#covariances-and-variances)). A start orders of
magnitude wider than the target makes this likely: rare replicates whose chain is still
on its way to the target dominate the estimates of variances
([Choosing one](initial-distribution.md#choosing-one)).

**Check.** The variances, and the largest share of one replicate in the variance of an
estimate, with the replicate that holds it:

```{code-cell} python
print(result.var())
print({name: values[name] for name in ("variance_share", "variance_share_replicate")})
```

**Fix.** A start wider than the target, but not by orders of magnitude
([Choosing one](initial-distribution.md#choosing-one) says how wide);
[A start far wider than the target](../tutorials/failure-modes.md#a-start-far-wider-than-the-target)
works through a case.

### Estimates that miss a mode, without a warning

A run whose pairs all met in one mode warns of nothing: its meeting times and estimates
cannot show a mode that no replicate visited
([A run that never left one mode](../reference/warnings.md#a-run-that-never-left-one-mode)).

**Check.** The tuning chains' $\hat R$, which raises the tuning warning when the chains
disagree ([Disagreeing tuning chains](../reference/warnings.md#disagreeing-tuning-chains-tuning)),
though chains that agree can still all sit in one mode ([Tuning](kernels.md#tuning));
and the same estimate from a second run with a wider
initial distribution, which should agree within the error bars:

```{code-cell} python
result.tuning.rhat
```

**Fix.** An initial distribution that covers every mode, and a kernel that reaches every
mode from it ([Choosing one](initial-distribution.md#choosing-one));
[When unbiased is not enough](../tutorials/failure-modes.md#two-modes) works through such
runs.

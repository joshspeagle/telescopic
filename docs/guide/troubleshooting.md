---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  merge_streams: true
---

# Troubleshoot chains that do not meet

This page is indexed by what a run shows: an error or a warning, or a result that looks
wrong. Each entry gives the cause, a check in code and the fix, and links to the page
that holds each fact. [Warnings and errors](../reference/warnings.md) gives the text,
threshold and meaning of every warning.

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

### k cannot be chosen

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

With MALA or HMC whose parameters were tuned, a note on the step size follows the message
([When the automatic step fails](step-size.md#when-the-automatic-step-fails)). When some
pilot {term}`pairs <pair>` started where the target density is zero, the message says
how many instead
([Choosing one](initial-distribution.md#choosing-one), "Inside the support").

**Cause.** The pilot's pairs run until they meet or reach the pilot's cap, and `k` is a
quantile of their {term}`meeting times <meeting time>`, which cannot be chosen when too
many are censored ([The automatic pipeline](settings.md#the-automatic-pipeline)). The
usual causes are an automatic step size of MALA or HMC that does not couple, a target
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

### The tuning chains disagree

The tuning warning comes before the pilot and the replicates.
[Disagreeing tuning chains](../reference/warnings.md#disagreeing-tuning-chains-tuning)
gives its text, its cause and its fix; `sampler.tuning.rhat`, printed above, is its
check.

## While the replicates run

### A pair that has not met, or a run that seems to hang

A pair still unmet at a threshold raises one `ConvergenceWarning` per run: in this
process as soon as the pair passes it, and on a pool only when the run ends
([Slow pairs](../reference/warnings.md#slow-pairs-slow), which gives the text for a
replicate). For a pilot pair, as here with a random walk whose proposals are far too
small to meet, it reads:

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

A pair that started where the target density is zero says so in this warning, and its
start also raises an `InputWarning`
([Warning classes](../reference/warnings.md#warning-classes)).

**Cause.** Heavy-tailed meeting times, from a kernel that mixes slowly or a start far from
the target, or a chain that cannot leave a region of zero density. A run without
`max_iter` waits for every pair, so a pair that never meets keeps the run waiting.

**Check.** `progress=True` shows the replicates finished as the run goes
([Progress](parallel.md#progress)). Pilot pairs with a cap, as in the first entry, show
the tail of the meeting times without waiting for it, and
{meth}`~telescopic.Result.plot_meeting_times` draws their survival function for a result
([Meeting times](meeting-times.md#meeting-times)).

**Fix.** Look at the kernel ([Choose a kernel](kernels.md)) and the initial distribution
([Choosing one](initial-distribution.md#choosing-one)). A first Ctrl-C returns the
replicates that started ([Interrupting a run](parallel.md#interrupting-a-run)).
`max_iter` bounds the wait, but censors the pairs still unmet at it, which biases the
estimates ([Censored pairs](../reference/warnings.md#censored-pairs-censored)).

### MALA or HMC meets far more slowly than the random walk

A gradient kernel whose pairs take much longer to meet than the random walk's, or that
adds the note on the step size to the error above, often has an automatic step that does
not couple ([When the automatic step fails](step-size.md#when-the-automatic-step-fails)).

**Check.** Pilot pairs with a given step size and covariance run no tuning; over a few
step sizes, compare the unmet pairs and the median meeting time of the others:

```{code-cell} python
for step in [0.25, 0.5, 1.0]:
    mala = telescopic.CoupledSampler(
        gaussian.log_prob,
        gaussian.initial,
        grad=gaussian.grad,
        vectorize=True,
        kernel=kernels.MALA(step, cov=gaussian.cov),
        rng=5,
    )
    mala_tau = mala.meeting_times(100, max_iter=3000)
    met = mala_tau[np.isfinite(mala_tau)]
    print(f"step {step}: {len(mala_tau) - len(met)} unmet, median {np.median(met):.0f}")
```

**Fix.** The step size with the lowest cost of a pair
([Checking a step size](step-size.md#checking-a-step-size)), and for HMC the default
`jitter` ([HMC](kernels.md#hmc)).

## When the run ends

The warnings `censored`, `late`, `variance` and `tail` apply to the result of all the
replicates; the summary lists them, and {func}`telescopic.diagnostics.issues` returns
them, with `slow` and `tuning` when they apply. Each has its entry in Warnings and errors:
[`censored`](../reference/warnings.md#censored-pairs-censored),
[`late`](../reference/warnings.md#late-pairs-late),
[`variance`](../reference/warnings.md#one-replicate-dominating-the-variance-variance) and
[`tail`](../reference/warnings.md#heavy-tailed-meeting-times-tail).

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

**Fix.** Each warning's entry gives its remedy.

## In the estimates

### A negative variance, or a standard deviation of zero

An unbiased variance can be negative, and the summary's `std` is then 0
([Why is my variance negative?](../reference/faq.md#why-is-my-variance-negative-or-a-probability-outside-0-1)).

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

**Check.** The tuning chains' $\hat R$, printed in the first entry, which raises the
tuning warning when the chains disagree, though chains that agree can still all sit in
one mode ([Tuning](kernels.md#tuning)); and the same estimate from a second run with a
wider initial distribution, which should agree within the error bars.

**Fix.** An initial distribution that covers every mode, and a kernel that reaches every
mode from it ([Choosing one](initial-distribution.md#choosing-one));
[When unbiased is not enough](../tutorials/failure-modes.md#two-modes) works through such
runs.

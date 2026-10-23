# Pitfalls

Unbiasedness holds for any initial distribution, burn-in, lag and run length, as long as
the meeting time has enough moments ([Theory](../theory.md)). It does not make every
run accurate. These are the ways a run goes wrong, and what shows it.

## An initial distribution that misses a mode

If every pair starts near one mode and the kernel rarely crosses to another, every pair
meets in that mode, and the estimates and their error bars describe it alone. Nothing
in the run shows the missing mode at a moderate number of replicates: on a mixture of
two Gaussians, Jacob, O'Leary & Atchadé (2020, §5.1) report a 95% interval of
[0.799, 0.816] for a true value of 0.421 from R = 1000 replicates.

The initial distribution should cover every mode. The tuning chains are a first check:
when their means spread more widely than the chains themselves, telescopic warns that
the target may have several modes. A wide initial distribution makes pairs that start
in different modes take much longer to meet, which the meeting times then show.

## An initial distribution far wider than the target

The opposite mistake costs variance rather than bias. With the automatic settings
(`k` at the 99% quantile of the pilot's meeting times, the lag equal to `k`), about one
pair in a hundred meets while its chain is still on its way to the target. That
replicate averages states far from the target without correction, and only rarer
replicates whose other chain is late balance it, in expectation. When the start is
orders of magnitude wider than the target, these replicates dominate the estimates of
variances and higher moments, which can then come out negative.

A start a few to ten times wider than the target avoids this. For a target with one
mode, a Gaussian centred on an optimiser's fit, with its inverse Hessian scaled by 100,
is ten times wider than the fit's error bars ([Getting started](../getting-started.md)).
The variance warning fires when one replicate holds more than half the variance of the
estimate of a coordinate's mean or second moment.

## Heavy tails

Unbiasedness needs moments of the meeting time, and valid error bars need a finite
variance of the estimates. Meeting times with a polynomial tail can break both: the
mean of the replicates is then unbiased but its error bar is not reliable at any
number of replicates. telescopic warns when the Hill estimate of the tail index of
`tau - lag + 1` is below 2 (with at least 500 replicates), and when one replicate
dominates the variance.

## A log-density that is not a function

The pairs meet when their states are equal to the last bit. A log-density that returns
different values at the same point, such as one estimated by simulation, breaks the
invariance of the kernels and the couplings, which assume that equal points have equal
densities. The log-density must be deterministic.

## Choosing the number of replicates after looking

Unbiasedness holds for a number of replicates fixed in advance. Running more replicates
until the error bar is small enough is optional stopping: the estimates stay consistent
and the large-R intervals stay valid, but the mean is no longer exactly unbiased.

## Capping the run length

A pair still unmet at `max_iter` is censored and left out of the estimates, which
biases them. A capped run warns with the number of censored pairs; rerun them without a
cap or with a larger one.

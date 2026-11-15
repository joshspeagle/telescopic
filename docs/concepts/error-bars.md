# Error bars from replicates, and what replaces R-hat and ESS

An MCMC run is usually judged by its trace plots, the Gelman-Rubin $\hat R$, the
effective sample size and the autocorrelation time: together they say whether the chains
have converged and how large the error bar of an average is. telescopic answers the same
two questions in other ways. Its error bars come from independent replicates, and its
meeting times bound the distance of the chain from the target. This page explains where
the error bars come from, maps each familiar diagnostic to its counterpart, and lists
what to report. It uses the notation of [Theory](../theory.md#estimates-from-many-replicates).

## Where the error bar comes from

A run has $R$ replicates. Each is a pair of chains with random numbers of its own, so
the replicates are independent, and each returns an estimate $H_r$ of the expectation
$\pi(h)$ whose expected value is exactly $\pi(h)$
([The telescoping correction](telescoping-correction.md)). The estimate of the run is
their mean $\bar H$, and its error bar is that of a mean of independent values. With
$\hat\sigma$ the standard deviation of the $R$ estimates, the interval at level
$1 - \alpha$ is

$$\bar H \pm t_{R-1,\,1-\alpha/2}\,\frac{\hat\sigma}{\sqrt R},$$

where $t_{R-1,\,1-\alpha/2}$ is the quantile of Student's t with $R - 1$ degrees of
freedom, for an estimate from all $R$ replicates, and $\hat\sigma/\sqrt R$ is the standard error
([Expectations](../guide/estimates.md#expectations)). The interval is valid as $R$ grows
when the estimates have a finite variance, and exact at every $R$ when they are normal
([Theory](../theory.md#guarantees-and-their-limits)).

Two things that an MCMC error bar needs are not part of it:

- **No autocorrelation time.** The states of one chain are correlated, so an MCMC error
  bar divides the variance of the draws by an effective sample size, computed from the
  integrated autocorrelation time $\tau_{\mathrm{int}}$, which is itself estimated from
  the same run. Here the correlation within a replicate's chains is inside its estimate
  $H_r$: it makes the estimates of the replicates vary more, and $\hat\sigma$ measures
  that variation directly. Different replicates share nothing, so there is no
  correlation between them to estimate.
- **No bias outside the error bar.** The error bar of an average after a burn-in leaves
  out the bias of the burn-in ([Why burn-in biases MCMC](burn-in-bias.md)). Each $H_r$
  is unbiased, so the error bar of $\bar H$ describes its whole error.

The error bar still needs a finite variance of the estimates. Meeting times with a heavy
tail, or one replicate that holds most of the variance, can make it unreliable, and a
run warns of both ([Warnings and errors](../reference/warnings.md#warnings)). Exact
unbiasedness also needs a number of replicates fixed before any result is seen
([How many replicates](../guide/settings.md#how-many-replicates)). A run with `keep`
estimates the means, the variances, the standard deviations and, up to 100 dimensions,
the covariance from every replicate, and the other estimates from the kept ones
([Two sets of replicates](../guide/estimates.md#two-sets-of-replicates)). An interval
then takes its degrees of freedom from the replicates behind its estimate: those of
{meth}`~telescopic.Result.estimate` are the number of kept replicates that met, less 1.

## What replaces a judgment of convergence

$\hat R$ compares chains started from different points: if they disagree, they have not
forgotten their starts, the burn-in was too short, and their averages are biased. In
telescopic the burn-in $k$ does not decide the bias, so the estimates need no judgment
that the chains have converged. Three things take the place of $\hat R$:

- **The meeting times** bound the total variation distance between the law $\pi_k$ of
  the chain after the burn-in and the target. The bound tells how much correction the
  replicates need, which sets their variance and cost
  ([Meeting times and distance to stationarity](meeting-times.md)).
- **The tuning chains.** When the kernel has parameters to tune, the sampler tunes them
  from four adaptive chains before any pair runs, and their $\hat R$ is the only one
  that telescopic computes. Above $\sqrt 2$ in some coordinate, a warning says that the chains may sit
  in different modes or may not have converged
  ([Disagreeing tuning chains](../reference/warnings.md#disagreeing-tuning-chains-tuning)).
  Their spread against that of the initial distribution's draws gives another warning,
  when the start is more than 30 times wider than the target
  ([A start far wider than the target](../reference/warnings.md#a-start-far-wider-than-the-target-wide)).
- **The chains themselves**, for readers who want $\hat R$ and effective sample sizes
  anyway: {meth}`~telescopic.Result.chains` returns the averaged states of each kept
  replicate, one chain per replicate, in the shape that ArviZ takes. They describe the
  ordinary MCMC chains after a burn-in of $k$, without the correction, and not the error
  of the estimates.

None of these shows that the chains found every mode of the target. Tuning chains that
agree, and pairs that meet quickly, can all sit in one mode, which is the same blind spot
that $\hat R$ has ([Theory](../theory.md#guarantees-and-their-limits),
[When unbiased is not enough](../tutorials/failure-modes.md)).

## Each diagnostic and its counterpart

Each entry gives what the familiar diagnostic answers, its counterpart in telescopic, and
where a result holds that counterpart
([Result and summary fields](../reference/result-fields.md)).

Monte Carlo standard error
: How far the estimate typically falls from its expected value. In telescopic: the
  standard error of the mean of independent replicates. In a result:
  `result.estimate(h).standard_error`, and the summary's `standard_error` column.

Effective sample size
: How many independent draws the correlated draws are worth. In telescopic: the number
  of independent estimates behind the error bar. In a result: `result.n_replicates`, and
  `result.n_kept` for the estimates of the kept replicates.

Autocorrelation time
: How many steps apart two draws are nearly independent, which sets the error bar, the
  burn-in and the thinning. In telescopic: not needed for the error bar; the
  inefficiency, the mean cost times the variance of one replicate's estimate, compares
  kernels and settings at equal cost ([Cost](../guide/settings.md#cost)). In a result:
  `result.costs` and `result.estimate(h).replicates`, and `relative_cost` in
  `result.diagnostics()`.

$\hat R$
: Whether chains from different starts agree. In telescopic: not needed for
  unbiasedness; the $\hat R$ of the tuning chains checks the tuning. In a result:
  `result.tuning.rhat`, and the `tuning` warning.

Trace plots and the burn-in
: Whether the chain has forgotten its start by the end of the burn-in. In telescopic:
  the meeting times, and the bound they give on the distance of $\pi_t$ from $\pi$. In a
  result: `result.meeting_times`, `result.tv_bound`, `tv_bound_at_k` in
  `result.diagnostics()`, `plot_meeting_times` and `plot_tv_bound`.

## Reporting results

A reader of a paper needs the estimate, its error and enough of the run to judge both.
Quote:

- each estimate with its standard error, or its interval with the level; quantiles and
  credible intervals are consistent estimates without a standard error
  ([Quantiles and credible intervals](../guide/estimates.md#quantiles-and-credible-intervals));
- the number of replicates $R$ behind each estimate, and with `keep` the number of kept
  replicates behind the estimates that come from them;
- the settings $k$, $m$ and the lag $L$, with `thin` when it is not 1, and the kernel
  with its parameters;
- the median, 90% and 99% quantiles and the maximum of the meeting times, and the number
  of censored pairs, and the bound on the total variation distance at $k$, which says
  how far $\pi_k$ was from $\pi$
  ([Meeting times and distance to stationarity](meeting-times.md#the-bound));
- each warning that the run emitted, and what was done about it;
- that $R$ was fixed before any result was seen. If the run was extended until the error
  bar was small enough, say so: the estimates are then consistent and their large-$R$
  intervals valid, but they are not exactly unbiased
  ([Theory](../theory.md#guarantees-and-their-limits)).

The summary of a result prints the estimates of the coordinates' means, $R$, the
settings, the kernel, the meeting times and the warnings
([The summary](../reference/result-fields.md#the-summary)); `result.diagnostics()`
holds the bound at $k$ as `tv_bound_at_k`, `result.kernel` every parameter of the
kernel, and `result.seed` the seed that reproduces the run. The settings change the cost and the variance of the
estimates, never their expected value, so they are reported to let others judge the
efficiency of the run and repeat it, not to support a claim that the burn-in was long
enough.

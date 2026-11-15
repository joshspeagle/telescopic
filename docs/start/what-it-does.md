# What telescopic does

An MCMC average after a burn-in is biased by an amount that its error bar leaves out, and
that error bar holds only if the chains are judged to have converged. telescopic runs
Markov chain Monte Carlo as pairs of chains that become exactly equal, and returns
estimates that are unbiased for any burn-in, with error bars from independent pairs.

## The problem

A Markov chain Monte Carlo run, such as one of emcee, starts its chains away from the
target, discards the first steps as burn-in, and averages the rest. Two things in that
recipe rest on judgment:

- **The burn-in leaves a bias.** The law of the chain approaches the target but, in general,
  does not reach it after any finite number of steps, so the average after a burn-in is
  biased, and its error bar does not include that bias
  ([Why burn-in biases MCMC](../concepts/burn-in-bias.md)).
- **The error bar assumes convergence.** It comes from an autocorrelation time estimated
  from the same run, and it holds if the chains converged by the end of the burn-in,
  which trace plots and $\hat R$ judge but do not show
  ([Error bars from replicates](../concepts/error-bars.md)).

## What telescopic does instead

telescopic runs the chains in pairs. The two chains of a pair move by the same MCMC
kernel, such as random-walk Metropolis, and each on its own is an ordinary chain of that
kernel. Their moves are drawn jointly, by a coupling, so that the two chains can land on
the same point and stay together from then on
([Two chains that meet](../concepts/two-chains.md)). The second chain runs $L$ steps
behind the first, and the differences between the two chains before they meet add up to
a correction that removes the bias of the burn-in on average
([The telescoping correction](../concepts/telescoping-correction.md)).

```{diagram} two-chains
One pair run by telescopic with random-walk Metropolis on a standard normal target, the
gray density on its side at the right, from chains started at draws of
$\mathcal N(5, 1)$. $X_t$ is solid in magenta and $Y_{t-L}$ dashed in teal, drawn $L$
steps to the right, at the time at which $X$ is compared with it. The dashed line joins
the solid one at the meeting time $\tau$, marked by a dot, after which one line continues
in the ink of met chains. The estimator averages the shaded times $k$ to $m$, and the
differences $X_t - Y_{t-L}$ at the hatched times $k + L$ to $\tau - 1$ make its bias
correction.
```

Each pair is a replicate, with random numbers of its own, so the replicates are
independent and run in parallel without communicating
([Run in parallel](../guide/parallel.md)). By default, telescopic tunes the kernel and
chooses the burn-in and the length of the chains from pilot runs before the replicates
([Set k, m, the lag, thin and max_iter](../guide/settings.md#the-automatic-pipeline)).

## What it returns

- **Unbiased estimates of expectations**, such as means, variances, covariances and
  probabilities, of any function of the parameters: averaged over replicates, each
  estimate has expected value equal to the expectation under the target, for any
  burn-in. The conditions are stated in
  [Theory](../theory.md#guarantees-and-their-limits).
- **Error bars** that are the standard error of a mean of independent replicates. They
  need no autocorrelation time and no judgment of convergence, and they are valid as the
  number of replicates grows when the estimates have a finite variance
  ([Error bars from replicates](../concepts/error-bars.md)).
- **Histograms**, unbiased in each bin, and **quantiles and credible intervals**,
  consistent as the number of replicates grows
  ([Estimate anything from a result](../guide/estimates.md)).
- **Bounds on the distance to the target**, estimated from the meeting times of the
  pairs, on the total variation distance between the law of the chain at each time and
  the target ([Meeting times and distance to stationarity](../concepts/meeting-times.md)).

## What it does not do

- **It does not make chains mix faster.** A coupling leaves each chain's moves as they
  were. Where a kernel mixes slowly, its pairs meet late and each replicate costs more
  ([Theory](../theory.md#guarantees-and-their-limits)).
- **It does not draw samples.** A replicate returns points with weights, some of them
  negative, which are not draws from the target and are plotted with their weights
  ([Signed measures, not samples](../concepts/signed-measures.md)).
- **It does not find modes that the starts miss.** Pairs that start in one mode of a
  target with several can meet there, and the estimates and their error bars then
  describe that mode alone
  ([When unbiased is not enough](../tutorials/failure-modes.md)).

## Next

The Concepts pages explain each idea above without code:
[Why burn-in biases MCMC](../concepts/burn-in-bias.md),
[Two chains that meet](../concepts/two-chains.md),
[The telescoping correction](../concepts/telescoping-correction.md),
[Signed measures, not samples](../concepts/signed-measures.md),
[Error bars from replicates](../concepts/error-bars.md) and
[Meeting times and distance to stationarity](../concepts/meeting-times.md).
[Is telescopic right for my problem?](is-it-right.md) says when to try it and when
another sampler serves better.

# Signed measures, not samples

A run of telescopic does not return samples of the target. Each {term}`replicate` returns a set
of points, its atoms, each with a weight, and some of the weights are negative. This
page explains where the negative weights come from, what can be computed from the
atoms, and what cannot. It uses the notation of
[Theory](../theory.md#signed-measures).

## Where the weights come from

A replicate's estimate of the expectation $\pi(h)$ has two parts
([The telescoping correction](telescoping-correction.md)): the average of $h$ over the
states of $X$ at the averaged times $S$, which run from $k$ to $m$, and the bias
correction, a weighted sum of the differences $\Delta_t = h(X_t) - h(Y_{t-L})$ from
$t = k + L$ up to the {term}`meeting time` $\tau$. Both parts add up values of $h$ at points,
each with a coefficient, so the estimate is the weighted sum of $h$ over a set of
weighted points. Written with $\delta_x$, a unit mass at the point $x$, that set is a
**signed measure**:

$$\begin{aligned}
\hat\pi = {}& \sum_{l \in S} \frac{1}{|S|}\,\delta_{X_l} \\
&+ \sum_{t = k + L}^{\tau - 1} c_t\,\big(\delta_{X_t} - \delta_{Y_{t-L}}\big),
\end{aligned}$$

and $\hat\pi(h)$, the sum of $h$ at the atoms times their weights, is the replicate's
estimate for every $h$ at once. The coefficient $c_t$ of a difference is never
negative: it is the number of averaged times $l$ whose correction contains the
difference at $t$, divided by $|S|$
([Theory](../theory.md#the-time-averaged-estimator)), and atoms whose coefficient is 0
are dropped. Each state of $X$ in
the average therefore gets a positive weight, and each difference adds two atoms: $X_t$
with the weight $c_t$ and $Y_{t-L}$ with the weight $-c_t$. The two weights of a
difference cancel, so a replicate's weights sum to 1, the weight of the average alone.

The negative weights sit on the states of $Y$, the chain that runs $L$ steps behind. A
{term}`pair` that meets within $k + L$ steps has no difference to add, and its weights are the
$1/|S|$ of an ordinary average after a {term}`burn-in`. A pair that meets later carries the
correction, its atoms on $X$ with positive weights and its atoms on $Y$ with negative
weights together. The correction as a whole has expected value equal to minus the bias
that the average after a burn-in of $k$ steps leaves
([Why burn-in biases MCMC](burn-in-bias.md)), so it removes that bias on average; no
part of it does so alone.

```{diagram} signed-measure
Top: the atoms of one replicate of random-walk Metropolis on a standard normal target,
started from $\mathcal N(5, 1)$, with $k = 2$, $L = 3$ and $m = 12$, run by telescopic,
and chosen among the first replicates so that its stems are far enough apart to read.
A random walk repeats its state when it rejects a proposal, so the atoms at one point
are drawn as one stem with their summed weight. The states of $X$ have positive weights,
drawn up in magenta, and the states of $Y$ in the bias correction have negative
weights, drawn down in teal; the weights sum to 1. The two panels share the horizontal
axis, at the bottom. Bottom: the signed histogram of 1,000
such replicates, the average over replicates of each one's mass in each bin, with two
standard errors on each bin and the target's mass in each bin as a gray outline. The
highlighted bin is negative, although the target puts a small positive mass there, which
lies within two standard errors of it.
```

## What the atoms give

Averaged over replicates, the weighted sum of any $h$ at the atoms has expectation
$\pi(h)$, and three kinds of results follow from the atoms and their weights
([Estimate anything from a result](../guide/estimates.md)).

- **Expectations** of any function, such as means, probabilities and covariances, are
  {term}`unbiased`, with error bars from the replicates
  ([Error bars from replicates](error-bars.md)). Since some weights are negative, an
  estimate can fall outside the range of $h$, such as a probability below 0 or a
  variance below 0 ([Covariances and variances](../guide/estimates.md#covariances-and-variances)).
- **Histograms** with bins fixed before the run are unbiased in each bin, with a
  standard error for each, and a bin can be negative where few atoms fall, as in the
  figure ([Histograms](../guide/estimates.md#histograms)). The plots of a result draw
  these signed histograms, for each coordinate and for each pair of coordinates
  ([Plot a result](../guide/plots.md)).
- **Quantiles** and credible intervals come from the atoms of all the replicates pooled,
  with their cumulative weights made non-decreasing. They are {term}`consistent` as the number
  of replicates grows, but not unbiased, and they come without standard errors
  ([Theory](../theory.md#estimates-from-many-replicates)).

A run with `keep` stores the atoms of the replicates of index below `keep` only, and
the moments of every replicate: the means, the variances and, up to 100 dimensions, the
covariance come from every replicate, and the other estimates from the kept ones
([Two sets of replicates](../guide/estimates.md#two-sets-of-replicates)).

## What the atoms cannot give

The atoms are not draws from the target, so what treats them as draws is wrong:

- **Plots without the weights**, such as a scatter plot, a histogram of counts or a
  corner plot of the atoms, add in the states of $Y$ that the weights subtract, and
  count a second time the states of $X$ that enter the correction, which are those
  before the meeting.
- **Tools that take weights but assume that they are not negative**, such as corner and
  ArviZ, compute weighted quantiles that are wrong when some weights are negative
  ([Plot a result](../guide/plots.md)).
- **Resampling** the atoms in proportion to their weights is not defined, since a
  negative weight is not a probability.

When a tool needs draws, {meth}`~telescopic.Result.chains` returns the states of $X$
at the averaged times of each kept replicate that met, without the correction. They are
the ordinary MCMC chains after a burn-in of $k$ steps, one per replicate: they carry the
bias of that burn-in, which the estimates of telescopic remove and the chains do not.
{meth}`~telescopic.Result.to_inference_data` gives them to ArviZ as its chains.

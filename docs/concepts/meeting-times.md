# Meeting times and distance to stationarity

Every {term}`pair` of a run has a {term}`meeting time`, and the meeting times of many pairs say how
close the chain is to the target at each time. This page explains how to read their
survival function and the bound on the distance to the target that it gives, and why the
estimates do not rely on that bound. It uses the notation of
[Theory](../theory.md#distance-to-stationarity).

## The survival function of the meeting times

A pair's meeting time $\tau$ is the first time $t$ at which $X_t = Y_{t-L}$
([Two chains that meet](two-chains.md)). It is random, and it varies from pair to pair.
Its **survival function** $P(\tau > t)$ is the probability that a pair has not yet met
at time $t$, which a run estimates by the fraction of its pairs that have not. It is 1
for every $t$ up to the {term}`lag`: the chains are first compared at $t = L$, where $X_L$ and
$Y_0$, an independent draw, differ. It falls
toward 0 as $t$ grows. How fast it falls depends on the kernel and on how far from the
target the initial distribution starts the chains.
{meth}`~telescopic.Result.plot_meeting_times` draws it for $\tau - L$, which shifts the
curve by $L$ to the left, on a log scale, where a straight tail means meeting times with
a geometric tail
([Meeting times](../guide/meeting-times.md#meeting-times)).

## The bound

The law $\pi_t$ of $X_t$ approaches the target $\pi$ as $t$ grows
([Why burn-in biases MCMC](burn-in-bias.md)), and the **total variation distance**
$d_{TV}(\pi_t, \pi)$ measures how far apart they are: the largest difference between the
probabilities that the two give to any set. The meeting times bound it at every $t$
(Biswas, Jacob & Vanetti 2019):

$$d_{TV}(\pi_t, \pi) \le \sum_{j \ge 1} P(\tau > t + jL).$$

The argument telescopes the laws, as [the telescoping correction](telescoping-correction.md)
telescopes the expectations. The distance from $\pi_t$ to $\pi$ is at most the sum of the
distances between $\pi_{t+(j-1)L}$ and $\pi_{t+jL}$ over $j \ge 1$, the moves of the law
over successive spans of $L$ steps. In a pair, $Y_{t+(j-1)L}$ has the first law and
$X_{t+jL}$ the second, and the two can differ only if the chains have not met by time
$t + jL$. So each move is at most the probability $P(\tau > t + jL)$, and the bound adds
these probabilities up ([Theory](../theory.md#distance-to-stationarity)).

```{diagram} tv-bound
Top: the estimated survival function of the meeting times of 1,000 pairs with lag
$L = 3$, for random-walk Metropolis on a standard normal target started from
$\mathcal N(5, 1)$, run by telescopic with its automatic burn-in, on a linear scale and
with a band of two standard errors. For $t = 6$, stems mark its values at
$t + L$, $t + 2L$ and so on, and the bar at the right stacks their heights: its total is the
bound on the total variation distance at $t = 6$. Bottom: the bound, estimated in the
same way at every $t$, on a log scale, with a band of two standard errors. It is 1 at
the first times, where the sum exceeds 1, and falls as $t$ grows; the dashed line marks
the burn-in $k$ that the automatic settings chose, and the bound there.
```

## Reading the bound

{meth}`~telescopic.Result.tv_bound` returns the bound at any $t$, and
{meth}`~telescopic.Result.plot_tv_bound` draws it on a log scale with $k$ marked, as in
the lower panel ([Distance to stationarity](../guide/meeting-times.md#distance-to-stationarity)).
Four things to keep in mind when reading it:

- **It is an upper bound.** The chain can be much closer to the target than the bound
  says, and a bound of 1 says nothing, since no two laws differ by more.
- **It is an estimate.** A run averages over its pairs, so the bound has a Monte Carlo
  error, with a relative error that is largest in the tail, where few pairs remain. A pair censored at `max_iter`
  counts as meeting just after its cap, which makes the estimate a lower bound of the one
  without a cap.
- **The lag changes the estimate, not the distance.** Every lag bounds the same
  distance, but some bound it more tightly. With $L = k$, the automatic choice, the bound
  at $k$ is close to the fraction of pairs that meet after $k + L$, and on the target of
  [Choosing k, m and the lag](../tutorials/choosing-settings.md#distance-to-stationarity)
  it is far below the bound at $k$ from pairs with a lag of 1.
- **It bounds the law of the chain at one time.** The bound at $k$ says how far
  $\pi_k$, the law of the first averaged state, is from $\pi$, which is what a
  {term}`burn-in` chosen from a trace plot tries to judge by eye.

## The estimates do not rely on it

The estimates of telescopic are {term}`unbiased` for every burn-in $k$, whatever the bound at
$k$ ([The telescoping correction](telescoping-correction.md#unbiased-for-any-burn-in)).
The bound measures how much work the correction does. Its first term at $k$ is the
fraction of pairs that meet after $k + L$, the pairs that carry a correction. Where the
bound at $k$ is small, few pairs do, and most estimates are ordinary averages after a
burn-in; where many pairs meet late, the corrections make the estimates vary more from
pair to pair, and a run warns when more than a quarter of its pairs do
([Late pairs](../reference/warnings.md#late-pairs-late)). A large bound at $k$ calls for
a larger $k$ to lower the variance, not to remove a bias.

What the estimates do rely on is the tail of the meeting times. Unbiasedness and the
error bars need meeting times with a light tail, which the survival function shows on a
log scale and the run checks from 500 pairs
([Heavy-tailed meeting times](../reference/warnings.md#heavy-tailed-meeting-times-tail)).
Neither the bound nor the tail shows that the chains have found every mode of the
target. Pairs that never leave one mode can meet quickly there, and the bound, whose
argument assumes that $\pi_t$ approaches $\pi$, is then small although $\pi_t$ is far
from $\pi$ ([Theory](../theory.md#guarantees-and-their-limits)).

# The telescoping correction

A pair of chains that meet turns the bias of a burn-in into a sum that one pair can
compute. This page explains why the differences between the two chains add up to the
bias at the burn-in $k$, why one pair has only finitely many nonzero differences, and why
the result is unbiased for any $k$. It uses the notation of
[Theory](../theory.md#the-telescoping-sum), where the argument is derived.

## The bias as a sum of steps

After $k$ steps, the expectation $\mathbb E[h(X_k)]$ of a function $h$ differs from its
value $\pi(h)$ under the target by the bias of a burn-in of $k$
([Why burn-in biases MCMC](burn-in-bias.md)). As the chain runs on, the expectation moves
toward $\pi(h)$, and the bias equals the total of its moves. Counted in steps of the lag
$L$, from $k$ to $k + L$, from $k + L$ to $k + 2L$, and so on:

$$\begin{aligned}
\pi(h) = {}& \mathbb E[h(X_k)] \\
&+ \sum_{j \ge 1}\big(\mathbb E[h(X_{k+jL})] \\
&\qquad - \mathbb E[h(X_{k+(j-1)L})]\big).
\end{aligned}$$

The sum is called telescoping because each intermediate expectation is added once and
subtracted once: the first $n$ steps add up to
$\mathbb E[h(X_{k+nL})] - \mathbb E[h(X_k)]$, which tends to $\pi(h) - \mathbb E[h(X_k)]$
as $n$ grows.

```{diagram} telescoping
Top: the expectation of $h(X_t) = X_t$ at $t = k, k + L, k + 2L, \dots$, with $k = 2$
and $L = 3$, in magenta, for random-walk Metropolis on a standard normal target with
chains started from $\mathcal N(5, 1)$, computed on a grid from the kernel's transition
probabilities, falling toward $\pi(h) = 0$. The blue arrows are the steps
$\mathbb E[\Delta_{k+jL}]$ between successive points; the seven drawn nearly span the
gap between $\mathbb E[h(X_k)]$ and $\pi(h)$, and the steps after them close it.
Bottom: one pair run by telescopic with the same kernel and settings, its differences
$\Delta_t$ at the same times, as bars on a scale of their own; every difference from the
meeting time $\tau$ on is zero.
```

## Each step from one pair

Each step is a difference between expectations at two times $L$ apart, and a pair with
lag $L$ supplies a matching difference at every time. Write
$\Delta_t = h(X_t) - h(Y_{t-L})$. Since $Y$ on its own is an ordinary chain started from
$\pi_0$, $Y_{k+(j-1)L}$ has the same law as $X_{k+(j-1)L}$, so the expected difference
$\mathbb E[\Delta_{k+jL}]$ is exactly the $j$-th step of the sum. One pair therefore
gives a draw of every step at once, and adding $h(X_k)$ to its differences gives
$H_k = h(X_k) + \Delta_{k+L} + \Delta_{k+2L} + \cdots$, whose expectation is the right
side of the equation above, $\pi(h)$, under the conditions given below.

## Finitely many terms

From the meeting time $\tau$ on, the two chains are equal, $X_t = Y_{t-L}$, so
$\Delta_t = 0$ for every $t \ge \tau$ ([Two chains that meet](two-chains.md)). The sum
in $H_k$ stops at its last term before $\tau$, and one pair computes $H_k$ in finite
time: the pair needs to run only until it has met and has reached $k$. A pair that meets
within $k + L$ steps has no nonzero difference, and $H_k$ is then the ordinary draw
$h(X_k)$.

## Unbiased for any burn-in

$\mathbb E[H_k] = \pi(h)$ for every $k \ge 0$, under conditions that ask for meeting
times with light tails and for an $h$ with enough moments
([Guarantees and their limits](../theory.md#guarantees-and-their-limits)). The burn-in
no longer decides the bias. It decides how often a pair still needs its correction: a
short burn-in leaves more pairs that meet after $k + L$, and their differences make the
estimates vary more from pair to pair; a long one costs more steps per pair.

One pair's $H_k$ can lie far from $\pi(h)$, on either side. The guarantee is about the
average over pairs, and the error bar comes from the spread of independent pairs
([Theory](../theory.md#guarantees-and-their-limits)). telescopic also averages $H_l$ over
$l = k, \dots, m$ within each pair, which keeps the expectation and usually lowers the
variance ([Theory](../theory.md#the-time-averaged-estimator)). The weights that this
average and its correction give to the states of the two chains make a set of weighted
points, some with negative weights
([Signed measures, not samples](signed-measures.md)).
[How it works](../tutorials/how-it-works.md#one-replicates-estimate) takes one pair's
estimate apart into the average after the burn-in and its correction, and
[Unbiased on average](../tutorials/unbiased-on-average.md) shows the correction removing
the bias over 4000 pairs.

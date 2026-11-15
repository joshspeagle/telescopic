# Why burn-in biases MCMC

A Markov chain Monte Carlo run starts its chain somewhere, discards the first $k$ steps
as burn-in, and averages what follows. This page explains why that average is biased for
every finite $k$, why its error bar does not include the bias, and why removing the bias
takes two chains that can become exactly equal. It uses the notation of
[Theory](../theory.md#coupled-chains).

## The law of the chain at time t

A chain starts from a draw of the initial distribution $\pi_0$ and moves by a kernel $P$
that leaves the target $\pi$ invariant. Its state $X_t$ after $t$ steps is random: rerun
the chain from fresh draws of $\pi_0$ many times, and the values of $X_t$ form a
distribution, the law $\pi_t$ of $X_t$. Each step moves $\pi_t$ toward $\pi$, and for a
chain that can reach every part of the target, $\pi_t$ approaches $\pi$ as $t$ grows.
In general, and for the random walk of the figure below, a chain that does not start
from $\pi$ itself does not reach $\pi$ at any finite $t$: some memory of the start
remains after any number of steps.

```{diagram} burn-in
The density of $X_t$ at three times, in magenta, for random-walk Metropolis on a
standard normal target, the gray outline, with proposals $\mathcal N(X_t, 2.38^2)$ and
chains started from $\mathcal N(5, 1)$, computed on a grid from the kernel's transition
probabilities. The arrow at each time runs from the target's mean, 0, to the mean of
$X_t$: it shrinks as $t$ grows, and it is not zero at any of the three times.
```

## What a burn-in leaves

After a burn-in of $k$ steps, a run averages $h(X_k), \dots, h(X_m)$ to estimate the
expectation $\pi(h)$ of a function $h$ under the target. The expected value of that
average is the average of the expectations of $h$ under $\pi_k, \dots, \pi_m$, so its
bias is the mean of their differences from $\pi(h)$:

$$\begin{aligned}
&\mathbb E\Big[\frac{1}{m-k+1}\sum_{t=k}^{m} h(X_t)\Big] - \pi(h) \\
&\qquad = \frac{1}{m-k+1}\sum_{t=k}^{m}\big(\mathbb E[h(X_t)] - \pi(h)\big).
\end{aligned}$$

Each difference is nonzero when $\pi_t$ differs from $\pi$ in the way that $h$
measures, and differences of opposite signs can partly cancel. In general the bias
shrinks as $k$ grows, though not always at every step, and it is not zero at any
finite $k$. Choosing $k$ by looking at a trace plot
or a convergence diagnostic judges whether $\pi_k$ looks close to $\pi$; it does not make
the two equal.

The error bar of the average does not include this bias. An error bar measures how far
the average typically falls from its own expected value, the first term on the left,
which is not $\pi(h)$. With many independent chains, each with the same $k$ and $m$, the
error bar of their pooled average shrinks as the number of chains grows while the bias
stays the same, so an interval at any confidence level eventually excludes $\pi(h)$.
[Unbiased on average](../tutorials/unbiased-on-average.md) shows this with 4000 pairs of
chains on the target of the figure above: the average after the burn-in lies many error
bars from the target's mean.

## Why a correction needs two chains

The bias at $k$ is the difference between $\mathbb E[h(X_k)]$ and $\pi(h)$, and it can
be written as the sum of the steps that $\mathbb E[h(X_t)]$ takes on its way to $\pi(h)$
after time $k$. Each step is the change of the expectation over $L$ steps, and a second
chain that runs $L$ steps behind the first estimates it by the difference
$h(X_t) - h(Y_{t-L})$ between the two chains at each time $t$
([The telescoping correction](telescoping-correction.md)).

The sum has infinitely many steps. It can be computed from one pair of chains only if
the differences between the two chains become zero after a finite time, which requires
the two chains to become exactly equal and to stay equal. Two independent chains on a
continuous space never do: two independent continuous random variables are equal with
probability zero ([Theory](../theory.md#coupled-chains)). The chains must be built to
move together, by a coupling ([Two chains that meet](two-chains.md)).

telescopic keeps the burn-in and the average, and adds the correction that removes
their bias on average, so $k$ changes the cost and the variance of its estimates but not
their expected value ([Theory](../theory.md#choosing-k-m-and-l)).

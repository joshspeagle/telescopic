# Theory

This page derives the estimators that telescopic computes and states what they guarantee.
The code and its docstrings use this notation. The methods are those of Glynn & Rhee
(2014) and Jacob, O'Leary & Atchadé (2020, JOA20), with the lag of Biswas, Jacob &
Vanetti (2019, BJV19) and the tuning rules of Atchadé & Jacob (2024, AJ24).

## Coupled chains

$\pi$ is the target, $P$ a Markov kernel that leaves $\pi$ invariant, and $\pi_0$ the
initial distribution. $\pi_t$ is the law of $X_t$ when $X_0 \sim \pi_0$ and each step
follows $P$.

A **coupling** $\bar P$ of $P$ with itself moves a pair of points so that each point, on
its own, moves according to $P$:

$$\bar P\big((x, y), A \times \mathbb R^d\big) = P(x, A), \qquad
\bar P\big((x, y), \mathbb R^d \times A\big) = P(y, A).$$

A coupling can make the two points equal with positive probability, which no pair of
independent chains does on a continuous space.

With a lag $L \ge 1$, a replicate runs one pair of chains:

1. Draw $X_0$ and $Y_0$ from $\pi_0$ independently.
2. Draw $X_t \sim P(X_{t-1}, \cdot)$ for $t = 1, \dots, L$.
3. While $t < \max(\tau, m)$: before the meeting, draw
   $(X_{t+1}, Y_{t+1-L}) \sim \bar P\big((X_t, Y_{t-L}), \cdot\big)$; after it, draw
   $X_{t+1} \sim P(X_t, \cdot)$ and set $Y_{t+1-L} = X_{t+1}$.

The **meeting time** is $\tau = \inf\{t \ge L : X_t = Y_{t-L}\}$. Equality is exact:
telescopic compares the coordinates with `==` and never uses a tolerance. After the
meeting it simulates $X$ alone, so $X_t = Y_{t-L}$ for every $t \ge \tau$.

Each chain on its own is an ordinary Markov chain with kernel $P$ started from $\pi_0$, so
$X_t$ and $Y_t$ both have the law $\pi_t$.

## The telescoping sum

For a test function $h$, write $\Delta_t = h(X_t) - h(Y_{t-L})$. If
$\mathbb E[h(X_t)] \to \pi(h)$, then for any $k \ge 0$

$$\pi(h) = \lim_{n \to \infty} \mathbb E[h(X_{k+nL})]
= \mathbb E[h(X_k)] + \sum_{j \ge 1}
\big(\mathbb E[h(X_{k+jL})] - \mathbb E[h(X_{k+(j-1)L})]\big).$$

$X_{k+(j-1)L}$ and $Y_{k+(j-1)L}$ have the same law, so each difference is
$\mathbb E[\Delta_{k+jL}]$. Since $\Delta_t = 0$ for $t \ge \tau$, the sum inside the
expectation has finitely many nonzero terms, and when $\tau$ has enough moments the sum
and the expectation can be exchanged:

$$\pi(h) = \mathbb E[H_k], \qquad H_k = h(X_k) + \sum_{j \ge 1} \Delta_{k+jL}.$$

$H_k$ is an unbiased estimator of $\pi(h)$ that each replicate computes in finite time.
Its first term is an ordinary MCMC draw after $k$ steps. The rest corrects its bias.

## The time-averaged estimator

Averaging $H_l$ over $l = k, \dots, m$ keeps the unbiasedness and usually lowers the
variance:

$$H^{(L)}_{k:m} = \frac{1}{m-k+1}\sum_{l=k}^{m} H_l
= \mathrm{MCMC}_{k:m} + \mathrm{BC}_{k:m},$$

$$\mathrm{MCMC}_{k:m} = \frac{1}{m-k+1}\sum_{t=k}^{m} h(X_t), \qquad
\mathrm{BC}_{k:m} = \frac{1}{m-k+1}\sum_{t=k+L}^{\tau-1} v_t\,\Delta_t.$$

The difference $\Delta_t$ enters $H_l$ when $t - l$ is a positive multiple of $L$, so its
weight $v_t$ counts the $l \in [k, m]$ with $l = t - jL$ for some $j \ge 1$. The
conditions $j \ge 1$, $j \le (t-k)/L$ and $j \ge (t-m)/L$ give

$$v_t = \Big\lfloor \frac{t-k}{L} \Big\rfloor
- \Big\lceil \frac{\max(L,\, t-m)}{L} \Big\rceil + 1.$$

With $L = 1$, $v_t/(m-k+1) = \min\big(1, (t-k)/(m-k+1)\big)$ (JOA20, eq. 2.1).
$\mathrm{MCMC}_{k:m}$ is the usual average after a burn-in of $k$ steps, and
$\mathrm{BC}_{k:m}$ is zero when the chains meet within $k + L$ steps.

**Thinning.** Averaging over any set $S \subseteq \{k, \dots, m\}$ also gives an unbiased
estimator, $H_S = |S|^{-1}\sum_{l \in S} H_l$. The argument `thin` $= s$ uses
$S = \{k, k+s, k+2s, \dots\} \cap [k, m]$, and $\Delta_t$ then has the coefficient

$$c_t = |S|^{-1}\,\#\{l \in S : l \le t - L,\ L \mid (t - l)\},$$

which is $v_t/(m-k+1)$ when $s = 1$. Thinning stores fewer points. It does not shorten
the run, which still ends at $\max(\tau, m)$.

## Signed measures

Each replicate's estimator is the integral of $h$ against a signed measure,

$$\hat\pi = \sum_{l \in S} |S|^{-1}\,\delta_{X_l}
+ \sum_{t = \min S + L}^{\tau - 1} c_t\,\big(\delta_{X_t} - \delta_{Y_{t-L}}\big),$$

so that $\hat\pi(h) = H_S$ for every $h$. telescopic stores these atoms and weights, so
any test function can be applied after the run. The weights sum to 1, and some are
negative, so an estimate can fall outside the range of $h$. A replicate stores at most
$|S| + 2\max(0, \tau - \min S - L)$ atoms, since atoms with $c_t = 0$ are dropped.

## Cost

A replicate uses $C = L + 2(\tau - L) + \max(0, m - \tau)$ applications of the kernel,
counting a coupled step before the meeting as two. Its **inefficiency** is the mean cost
divided by $m - k + 1$, the cost of the MCMC average alone. Kernels differ in the
log-density and gradient evaluations per application, so telescopic also counts those,
and the wall time.

## Guarantees and their limits

**Guaranteed:**

- **Unbiasedness:** $\mathbb E[H^{(L)}_{k:m}] = \pi(h)$ for every $0 \le k \le m$,
  $L \ge 1$ and $\pi_0$, provided that the coupling preserves both marginals, the pair
  stays together after meeting, and the meeting time has enough moments. The moment
  conditions take any of three published forms:
  - geometric tails of $\tau$ (JOA20, Prop. 3.1);
  - polynomial tails (Middleton et al. 2020);
  - $\mathbb E_{\pi\otimes\pi}[\tau^\kappa] < \infty$ with $\kappa > 1$,
    $h \in L^p(\pi)$ with $p > \kappa/(\kappa-1)$, and $d\pi_0/d\pi \le M$ (Douc et al.,
    Thm. 17). Under these, the variance is finite if $\kappa > 2$ and
    $p > 2\kappa/(\kappa - 2)$.
- **Error bars:** with $R$ independent replicates and finite variance,
  $\bar H \pm z_{1-\alpha/2}\,\hat\sigma/\sqrt R$ is an asymptotically valid interval as
  $R \to \infty$. The bounded work per replicate does not change this.
- **Fixed $R$:** unbiasedness holds for a number of replicates fixed before any result is
  seen. Choosing $R$ after looking at results, for example extending a run until the
  error bar is small, is optional stopping: the mean stays consistent and the large-$R$
  intervals stay valid, but exact unbiasedness is lost.
- **Total-variation bounds:** meeting times give upper bounds on
  $d_{TV}(\pi_t, \pi)$ (see below).

**Not guaranteed:**

- a small error at finite $R$;
- valid intervals when the variance is infinite;
- better mixing than the underlying kernel: "coupling pairs of Markov chains does not
  improve their marginal mixing properties" (JOA20, §6);
- estimates inside the range of $h$;
- exact draws from $\pi$;
- robustness to a poor $\pi_0$. On a bimodal target with $\pi_0$ concentrated in one
  mode, JOA20 (§5.1) report a 95% interval of [0.799, 0.816] for a true value of 0.421
  at $R = 1000$.

## Estimates from many replicates

Let $a_r = \hat\pi_r(h) \in \mathbb R^p$ for replicates $r = 1, \dots, R$, with mean
$\bar a$ and sample covariance $S_a$ (divisor $R - 1$).

- **Expectations:** $\bar a$, with standard errors $\sqrt{\mathrm{diag}(S_a)/R}$ and
  Monte Carlo covariance $S_a/R$.
- **Products of expectations:** $\bar a\bar a^\top - S_a/R$ is unbiased for
  $\pi(h)\pi(h)^\top$ when $R \ge 2$.
- **Covariance:**
  $\widehat{\mathrm{Cov}}_\pi(h) = R^{-1}\sum_r \hat\pi_r(hh^\top) - \bar a\bar a^\top + S_a/R$
  is unbiased. Standard deviations are $\sqrt{\max(0, \widehat{\mathrm{Var}})}$, which is
  consistent but not unbiased.
- **Histograms:** each replicate's signed mass in each half-open bin $[a, b)$, averaged
  over replicates, is unbiased, with standard errors from the replicates. A bin can be
  negative.
- **Quantiles:** pool the atoms of all replicates with weights $\omega/R$, merging atoms
  with equal coordinates; sort them along the coordinate; project the cumulative weights
  onto non-decreasing sequences by isotonic regression and clip them to $[0, 1]$; the
  $\beta$-quantile is the smallest atom whose projected cumulative weight is at least
  $\beta$. This is consistent as $R \to \infty$ but not unbiased (JOA20, Prop. 3.2).

## Choosing $k$, $m$ and $L$

The settings change the cost and the variance, never the bias. telescopic follows AJ24:
from the meeting times of $R_0 = 100$ pilot pairs with $L = 1$,

1. $k = \max(1, \hat q_\beta(\tau - 1))$, with $\hat q_\beta$ the inverted-CDF sample
   quantile and $\beta = 0.99$;
2. $L = k$;
3. $m = 10k$.

With $L = k$ most pairs meet within $k + L$ steps, so $\mathrm{BC}_{k:m}$ is usually zero
and the estimator is an ordinary MCMC average after a burn-in of $k$. A pair that meets
before $m$ costs $m + \tau - L$, so with $m = 10k$ and $\tau - L \le k$ the
inefficiency is about $10/9$ to $11/9$. $\beta \le 1 - 1/R_0$ is required, so that
$\hat q_\beta$ is not the sample maximum.

## Distance to stationarity

For any $L$ and $t \ge 0$ (BJV19, Thm. 2.5),

$$d_{TV}(\pi_t, \pi) \le \mathbb E\Big[\max\Big(0, \Big\lceil \frac{\tau - L - t}{L}
\Big\rceil\Big)\Big] = \sum_{j \ge 1} \mathbb P(\tau > t + jL).$$

telescopic estimates the bound by the average over replicates and reports
$\min(1, \cdot)$. With $L = k$, its value at $t = k$ is close to
$\mathbb P(\tau - L > k)$, the fraction of pairs that meet late. A pair still unmet at
its cap counts as meeting there, so with censored pairs the estimate is a lower bound.

The same argument bounds the 1-Wasserstein distance:

$$d_{W_1}(\pi_t, \pi) \le \mathbb E\Big[\sum_{j=1}^{\lceil (\tau - L - t)/L\rceil}
\lVert X_{t+jL} - Y_{t+(j-1)L} \rVert\Big].$$

It depends on the parametrisation of the target, through the Euclidean norm.

## References

- Atchadé, Y. F. & Jacob, P. E. (2024). Unbiased Markov chain Monte Carlo: what, why,
  and how. arXiv:2406.06851.
- Biswas, N., Jacob, P. E. & Vanetti, P. (2019). Estimating convergence of Markov chains
  with L-lag couplings. NeurIPS.
- Douc, R., Jacob, P. E., Lee, A. & Vats, D. Solving the Poisson equation using coupled
  Markov chains. arXiv:2206.05691.
- Glynn, P. W. & Rhee, C.-H. (2014). Exact estimation for Markov chain equilibrium
  expectations. J. Appl. Probab. 51A, 377–389.
- Jacob, P. E., O'Leary, J. & Atchadé, Y. F. (2020). Unbiased Markov chain Monte Carlo
  methods with couplings. J. R. Stat. Soc. B 82(3), 543–600.
- Middleton, L., Deligiannidis, G., Doucet, A. & Jacob, P. E. (2020). Unbiased Markov
  chain Monte Carlo for intractable target distributions. Electron. J. Stat. 14(2),
  2842–2891.

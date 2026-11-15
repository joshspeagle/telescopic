# Notation

Each entry maps a symbol of [Theory](../theory.md) to the argument, attribute or function
that holds it in the code. `sample` stands for {func}`telescopic.sample`, `sampler` for a
{class}`telescopic.CoupledSampler`, `result` for a {class}`telescopic.Result`, and
`estimate` for the {class}`telescopic.Estimate` that `result.estimate(h)` returns. The
section headings follow those of Theory.

## Coupled chains

$\pi$
: The target. `log_prob` computes $\log\pi$ up to a constant, and `grad` its gradient
  $\nabla\log\pi$, both arguments of `sample` and `sampler`.

$\pi_0$
: The initial distribution: the argument `init`, and `result.initial`.

$P$, $\bar P$
: The kernel and its coupling: the argument `kernel`, `sampler.kernel` and
  `result.kernel`. A kernel applies $P$ in {meth}`~telescopic.kernels.Kernel.step` and
  $\bar P$ in {meth}`~telescopic.kernels.Kernel.coupled_step`; the argument `coupling`
  of {class}`~telescopic.kernels.RandomWalk` and {class}`~telescopic.kernels.Gibbs`
  chooses among their couplings.

$\pi_t$
: The law of $X_t$. No attribute holds it; `result.tv_bound(t)` bounds its distance to
  $\pi$ (below).

$d$
: The dimension: `result.ndim` and `sampler.ndim`.

$L$
: The lag: the argument `lag`, and `result.lag`.

$k$, $m$
: The first and the last averaged times: the arguments `k` and `m`, and `result.k` and
  `result.m`.

$X_t$, $Y_{t-L}$
: The two chains of a pair. A result does not store them; the atoms of its signed
  measure are their states at the times that the estimator uses, and
  {meth}`result.chains() <telescopic.Result.chains>` returns $X_t$ at the averaged times
  $t \in S$, shaped `(n_kept, len(S), d)`. {func}`telescopic.engine.run` with
  `keep_paths=True` returns the whole paths.

$\tau$
: The meeting time: `result.meeting_times`, one per pair, `inf` for a censored pair.
  The meeting times of the lag-1 pilot pairs are `sampler.pilot` and `result.pilot`.

## The telescoping sum and the time-averaged estimator

$h$
: The test function, the argument `h` of {meth}`~telescopic.Result.estimate`, which maps
  points of shape `(n, d)` to values of shape `(n,)` or `(n, p)`.

$\Delta_t$, $H_k$
: Not stored: each difference enters a replicate's signed measure as two atoms, $X_t$
  with weight $c_t$ and $Y_{t-L}$ with weight $-c_t$.

$H^{(L)}_{k:m}$, $H_S$
: One replicate's estimate: `estimate.replicates[r]`, which
  `result.measure.per_replicate(h)` also returns. $\bar H$, their mean, is
  `estimate.value`.

$\mathrm{MCMC}_{k:m}$
: The average of $h$ over the averaged states of $X$, which `result.chains()` returns,
  without the bias correction; with `thin` above 1, over the times of $S$.

$\mathrm{BC}_{k:m}$
: The bias correction: $H_S$ minus the average of $h$ over `result.chains()`.

$v_t$, $c_t$
: The weight and the coefficient of $\Delta_t$:
  {func}`~telescopic.estimators.correction_coefficients` returns $c_t$, which is
  $v_t/(m-k+1)$ when `thin` is 1.

$s$, $S$
: The argument `thin`, and `result.thin`; {func}`~telescopic.estimators.averaging_times`
  returns $S$.

## Signed measures

$\hat\pi_r$
: The signed measures of the kept replicates that met: `result.measure`, a
  {class}`telescopic.SignedMeasure` whose `atoms`, of shape `(N, d)`, `weights` and
  `replicate` hold every atom, its weight and the position of its replicate.
  $\hat\pi_r(h)$ for every $r$ is `result.measure.per_replicate(h)`.

## Cost

$C$
: The cost of each replicate, in kernel applications: `result.costs`.

Relative cost
: $\mathbb E[C]/(m - k + 1)$: `result.diagnostics()["relative_cost"]`.

Inefficiency
: $\mathbb E[C]\,\mathbb V[H_{k:m}]$: not an attribute;
  [Cost](../guide/settings.md#cost) computes it from `result.costs` and
  `estimate.replicates`.

## Guarantees and estimates from many replicates

$R$
: The number of replicates: the argument `n_replicates` of `sample` and of
  {meth}`~telescopic.CoupledSampler.run`, the number requested, and
  `result.n_replicates`, the number whose pairs met.

$\alpha$
: One minus the confidence level, the argument `level` of `result.estimate`, 0.95 by
  default. `estimate.interval` holds the interval
  $\bar H \pm t_{R-1,\,1-\alpha/2}\,\hat\sigma/\sqrt R$.

$a_r$, $\bar a$, $\sqrt{\mathrm{diag}(S_a)/R}$
: `estimate.replicates[r]`, `estimate.value` and `estimate.standard_error`.

$\widehat{\mathrm{Cov}}_\pi(h)$, $\widehat{\mathrm{Var}}$
: {meth}`result.cov() <telescopic.Result.cov>` and
  {meth}`result.var() <telescopic.Result.var>` for $h(x) = x$;
  {meth}`result.std() <telescopic.Result.std>` returns
  $\sqrt{\max(0, \widehat{\mathrm{Var}})}$.

$\beta$-quantile
: {meth}`result.quantile(q) <telescopic.Result.quantile>` with `q` $= \beta$.

## Keeping a subset

$K$
: The argument `keep`, and `result.keep`; `result.n_kept` counts the kept replicates
  that met.

$c_r$, $f_r$, $S_r$
: Each replicate's center and moments: `center` and `moments` of
  {class}`telescopic.engine.Replicates`, whose first $d$ columns are $f_r$ and the
  others the entries of $S_r$.

## Choosing k, m and L

$R_0$
: The number of pilot pairs, 100: the default of
  {meth}`sampler.meeting_times() <telescopic.CoupledSampler.meeting_times>`, and the
  length of `sampler.pilot` after the automatic settings.

$\beta$, $\hat q_\beta$
: The argument `quantile` of {func}`telescopic.tuning.choose_k`, 0.99 by default, and
  the quantile that it takes of $\tau - 1$.

## Distance to stationarity

$d_{TV}(\pi_t, \pi)$
: Its bound: {meth}`result.tv_bound(t) <telescopic.Result.tv_bound>`, or
  {func}`telescopic.diagnostics.tv_bound` from meeting times; its value at $k$ is
  `result.diagnostics()["tv_bound_at_k"]`.

$d_{W_1}(\pi_t, \pi)$
: Its bound: {meth}`result.w1_bound(t) <telescopic.Result.w1_bound>`.

## Kernel parameters

These symbols appear in the [kernels guide](../guide/kernels.md) and in the docstrings
of the kernels, not in Theory.

$\Sigma$
: The argument `cov` of a kernel, the proposal covariance or the inverse mass matrix.

$\lambda$
: The argument `scale` of {class}`~telescopic.kernels.RandomWalk`, $2.38/\sqrt d$ by
  default.

$\epsilon$
: The argument `step_size` of {class}`~telescopic.kernels.MALA` and
  {class}`~telescopic.kernels.HMC`.

$j$
: The argument `jitter` of {class}`~telescopic.kernels.HMC`, 0.5 by default: each
  trajectory's step is uniform between $(1 - j)\epsilon$ and $(1 + j)\epsilon$.

$\gamma$, $\sigma$
: The arguments `meeting_weight` and `meeting_scale` of
  {class}`~telescopic.kernels.HMC`: the probability of its meeting step and the scale of
  that step.

$\alpha$ (tail index)
: The tail index of the meeting times, `result.diagnostics()["tail_index"]`
  ([Heavy-tailed meeting times](warnings.md#heavy-tailed-meeting-times-tail)); not the
  $\alpha$ of a confidence level.

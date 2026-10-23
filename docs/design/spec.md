# telescopic: specification (v0.1)

*Unbiased Markov chain Monte Carlo with coupled chains and telescoping sums*

**Status:** v0.1 design, revised after the adversarial design review
(`notes/reviews/design-review.md`), 2026-10-23. Later departures are recorded in
`notes/DECISIONS.md`.

**Inputs to this document:**
- The landscape census: `reports/Unbiased MCMC with couplings landscape.md` and the eight notes behind it in `research_notes/Unbiased MCMC with couplings landscape/`.
- Prototype experiments in `notes/experiments/` (Appendix A).
- Josh's decisions in `notes/DECISIONS.md`: published methods first, new constructions as experiments (#1); one branch with reviewed milestones (#2).

**Companion documents:**
- [`test-plan.md`](test-plan.md): how each requirement is verified.
- [`implementation-plan.md`](implementation-plan.md): module layout, internal interfaces, milestones and the order of work.

Requirements carry IDs (`R-...`) so that tests and code can point to them. "Must" marks a v0.1 requirement, "should" one we intend for v0.1 but may defer, and "roadmap" work after v0.1.

---

## 1. Vision and principles

`telescopic` turns **a log-density and an initial distribution** into **unbiased estimates of expectations under the target**, with error bars that become valid as the number of independent replicates grows, not as the length of a chain grows. Each replicate runs two coupled copies of an MCMC kernel until they meet. A telescoping-sum correction then removes the burn-in bias of the usual MCMC average (Glynn & Rhee 2014; Jacob, O'Leary & Atchadé 2020, "JOA20" below).

The census found no maintained, installable implementation of these methods in any language. The reference code is a GPL research package in R that calls itself "not a general-purpose statistical software". BlackJAX added a coupled HMC kernel in October 2026, but without meeting or estimators. The theory has matured meanwhile: exact lag-L estimators, a practical tuning recipe (Atchadé & Jacob 2024, "AJ24"), coupling-based convergence bounds (Biswas, Jacob & Vanetti 2019), unbiased estimators under time budgets (Glynn & Heidelberger 1990, 1991), and couplings for the main kernel families. `telescopic`'s value is therefore **an emcee-style package that applies them correctly by default**, not a new estimator.

Principles:

1. **Unbiased by construction, honest about the limits.** Every replicate is an exactly unbiased estimate for its stated $(k, m, L)$ when its pair meets. Intervals are large-$R$ central-limit intervals. The package reports what can break them (pairs that do not meet, heavy tails, multimodal targets) instead of hiding it.
2. **Easy things are one line, hard things are possible.** `telescopic.sample(log_prob, init)` covers the common case. Beneath it sit a sampler object, then kernels, couplings and an engine as building blocks, with helpers to test new couplings.
3. **Tune on pilots, then freeze.** Kernels are tuned on independent pilot runs. Nothing adapts inside a coupled run.
4. **Replicates are the unit of work, randomness and parallelism.** A replicate's result depends only on the root seed and its index, never on batching, workers or scheduling.
5. **Lightweight core.** NumPy and SciPy only. Plots, progress bars and a later JAX backend are optional extras.
6. **One result object.** Summaries, estimates of any function, quantiles, histograms, diagnostics, plots and serialisation all handle signed weights.
7. **Self-validating.** Diagnostics run by default. The test suite checks every coupling and estimator against exact references.

---

## 2. Problem and scope

### 2.1 The problem

A target distribution $\pi$ on $\mathbb R^d$ is known through $\log\pi(x)$ up to a constant. The user wants $\pi(h) = \mathbb E_\pi[h(X)]$ for one or many test functions $h$, with an error bar, in finite time.

Ordinary MCMC starts a chain from $\pi_0$, discards a burn-in by eye or heuristics, and averages the rest. The average is biased by the burn-in, and its error bar is asymptotic in chain length. Its diagnostics ($\hat R$, effective sample size, autocorrelation time) assume approximate stationarity.

`telescopic` runs $R$ independent replicates. Each produces an exactly unbiased estimate at bounded cost, so their average converges at rate $R^{-1/2}$ with a central-limit interval, and the replicates run in parallel without communication.

### 2.2 Motivating use cases (all must be expressible)

| Use case | Target | Kernel |
|---|---|---|
| emcee-style parameter estimation in astronomy | black-box log-likelihood plus prior, $d \lesssim 30$ | random-walk Metropolis, tuned on a pilot |
| models with gradients (hand-written, JAX, autograd) | differentiable log-density, $d$ up to a few hundred | MALA, HMC |
| conditionally conjugate hierarchical models | standard full conditionals | Gibbs |
| near-Gaussian posteriors | a good proposal distribution exists | independence Metropolis–Hastings, alone or in a mixture |
| checking the burn-in of an existing MCMC setup | any kernel with a coupling | total-variation bounds from meeting times |
| research on new kernels | user-written kernel and coupling | building blocks and coupling checks |

### 2.3 Scope

**In scope for v0.1 (must):**
- **Targets:** log-densities on $\mathbb R^d$, scalar or vectorised, with an optional gradient.
- **Initial distributions:** callables, SciPy distributions, point sets, and helpers.
- **Coupling primitives:** reflection-maximal coupling of Gaussians with equal covariance, rejection maximal coupling (with the bounded-variance $\eta$ option), and maximal coupling of finite discrete distributions.
- **Kernels with their couplings:**
  - random-walk Metropolis–Hastings (RWMH), reflection-maximal by default;
  - MALA;
  - HMC as in Heng & Jacob (2019);
  - independence Metropolis–Hastings (IMH);
  - Gibbs with user-written conditionals;
  - mixtures of kernels.
- **Execution:**
  - the lag-$L$ construction with exact meeting detection and faithful continuation;
  - lockstep batches with refill, and per-replicate random streams;
  - process pools, with results invariant to the worker layout.
- **Estimators:** $H^{(L)}_{k:m}$ with exact weights, thinning, and signed measures. Unbiased means, second moments and covariances; consistent quantiles; unbiased histograms.
- **Tuning:** pilot runs for kernel parameters, and automatic $k$, $m$, $L$ from a pilot of meeting times.
- **Diagnostics:** meeting-time summaries, total-variation bounds, censoring, tail and outlier warnings, and the fraction of replicates with a bias correction.
- **Results:** a summary table, estimates of any function, plots, and save/load.
- **Validation helpers:** example targets with exact answers, and coupling checks.
- **Documentation:** a theory page, a user guide, executed tutorials, and an API reference.

**Should (v0.1 if time allows, otherwise v0.2):**
- Budget mode with the Glynn–Heidelberger per-lane estimator (§3.7).
- Checkpoint and resume to a file.
- 1-Wasserstein bounds (§3.6).
- Rank-normalised $\hat R$ and effective sample size across the replicates' X-chains; these are the atoms from $k$ to $m$. Also a relative inefficiency against an autocorrelation-based MCMC baseline.
- Export of the replicates' X-chains to ArviZ.
- Couplings:
  - two-scale GCRN and GCRefl couplings for RWMH with gradients (Papp & Sherlock 2025);
  - the kernel-maximal coupling $\bar P_C$ (O'Leary, Wang & Jacob 2021) for low-acceptance kernels.
- HMC trajectory lengths chosen from pilot meeting times over a small grid (Heng & Jacob §5).
- An unbiased asymptotic-variance report (Douc, Jacob, Lee & Vats).

**Roadmap (v0.2 and later):**
- A JAX backend: autodiff gradients and a persistent-lanes engine with in-loop refill on accelerators.
- The experiments of DECISIONS #1, promoted only after Josh has seen their evidence:
  - coupled elliptical slice sampling with coupled preconditioned Crank–Nicolson (pCN) meeting steps;
  - coupled NUTS;
  - re-tuning between batches of replicates;
  - surrogate-gradient GCRN for gradient-free targets.
- Variance reduction: control variates (Craiu & Meng) and Rao–Blackwellisation of the bias correction.
- Couplings for particle MCMC (CCPF, PMMH), piecewise-deterministic samplers and partitions.
- Unbiased estimates of nonlinear functionals by randomised multilevel Monte Carlo (Wang & Wang 2023).

**Out of scope:**
- Perfect sampling (coupling from the past).
- Adaptive MCMC inside a replicate.
- Discrete or trans-dimensional targets, except through user-written Gibbs or custom kernels.
- Emcee-style ensemble moves as coupled kernels. They do not meet (Appendix A, E-R3) and serve only as an optional pilot (roadmap).

---

## 3. Notation, estimators and guarantees

This section is normative. Code, docstrings and `docs/theory.md` use exactly this notation.

### 3.1 Chains, couplings and meeting

- $\pi$ is the target and $P$ a $\pi$-invariant Markov kernel. $\pi_0$ is the initial distribution.
- A **coupling** $\bar P$ of $P$ with itself satisfies $\bar P((x,y), A\times\mathbb R^d) = P(x, A)$ and $\bar P((x,y), \mathbb R^d\times A) = P(y, A)$ for all $x, y, A$.
- $L \ge 1$ is the **lag**.
- **Construction (R-EST-1):**
  1. Draw $X_0$ and $Y_0$ from $\pi_0$ independently.
  2. Draw $X_t \sim P(X_{t-1}, \cdot)$ for $t = 1, \dots, L$.
  3. For $t = L, L+1, \dots$ while $t < \max(\tau, m)$: before the meeting, draw $(X_{t+1}, Y_{t+1-L}) \sim \bar P((X_t, Y_{t-L}), \cdot)$; after it, draw $X_{t+1} \sim P(X_t, \cdot)$ and set $Y_{t+1-L} := X_{t+1}$.
- **Meeting time (R-EST-2):** $\tau = \inf\{t \ge L : X_t = Y_{t-L}\}$ (AJ24's convention, equal to JOA20's for $L = 1$). Equality is exact: the engine compares the positions with `==` once at $t = L$ and after every coupled step, so $+0.0$ and $-0.0$ are the same point. It never uses a tolerance.
- **Faithfulness (R-EST-3):** $X_t = Y_{t-L}$ for all $t \ge \tau$. The engine enforces it by simulating only $X$ after the meeting (step 3). It does not rely on the coupled kernel keeping equal states together.

### 3.2 The estimator

For a test function $h$ and integers $0 \le k \le m$, write $\Delta_t = h(X_t) - h(Y_{t-L})$.

$$H_t = h(X_t) + \sum_{j \ge 1} \Delta_{t+jL}, \qquad H^{(L)}_{k:m} = \frac{1}{m-k+1}\sum_{t=k}^{m} H_t = \mathrm{MCMC}_{k:m} + \mathrm{BC}_{k:m},$$

$$\mathrm{MCMC}_{k:m} = \frac{1}{m-k+1}\sum_{t=k}^{m} h(X_t), \qquad \mathrm{BC}_{k:m} = \frac{1}{m-k+1}\sum_{t=k+L}^{\tau-1} v_t\,\Delta_t,$$

$$v_t = \Big\lfloor \frac{t-k}{L} \Big\rfloor - \Big\lceil \frac{\max(L,\, t-m)}{L} \Big\rceil + 1. \tag{R-EST-4}$$

- The sums in $H_t$ are finite because $\Delta_t = 0$ for $t \ge \tau$.
- $v_t$ counts the indices $l \in [k, m]$ with $t - l$ a positive multiple of $L$. For $L = 1$, $v_t/(m-k+1) = \min(1, (t-k)/(m-k+1))$ (JOA20 eq. 2.1).
- The weight printed in the JRSSB 2020 reply, $\min(1, \lceil(l-k)/L\rceil/(m-k+1))$, is biased for $L \ge 2$ (research note `theory_estimators.md` §1). It must not be used.

**Thinning (R-EST-5).** For any averaging set $S \subseteq \{k, \dots, m\}$, $H_S = |S|^{-1}\sum_{l \in S} H_l$ is also unbiased. `thin` $= s$ uses $S = \{k, k+s, k+2s, \dots\} \cap [k, m]$. The bias-correction coefficient of $\Delta_t$ is then

$$c_t = |S|^{-1}\,\#\{l \in S : l \le t - L,\ L \mid (t - l)\},$$

which reduces to $v_t/(m-k+1)$ when $s = 1$. The run still ends at $\max(\tau, m)$.

**Signed measure (R-EST-6).** Each replicate yields

$$\hat\pi = \sum_{l \in S} |S|^{-1}\,\delta_{X_l} + \sum_{t \ge \min S + L}^{\tau - 1} c_t\,\big(\delta_{X_t} - \delta_{Y_{t-L}}\big),$$

whose weights sum to 1. Some weights are negative, so estimates can fall outside the range of $h$. $\hat\pi(h) = H_S$ for every $h$, so any test function can be applied after the run. Atoms with $c_t = 0$ are not stored, so the number of atoms is at most $|S| + 2\max(0, \tau - \min S - L)$.

**Cost (R-EST-7).** One replicate uses $C = L + 2(\tau - L) + \max(0, m - \tau)$ kernel applications, counting a coupled step before the meeting as two. The engine also records the log-density and gradient evaluations and the wall time, because kernels differ in evaluations per application.

### 3.3 Guarantees and their limits (R-THEORY-1: the documentation must state them in these terms)

**Guaranteed:**
- **Unbiasedness:** $\mathbb E[H^{(L)}_{k:m}] = \pi(h)$ for every $0 \le k \le m$, $L \ge 1$ and $\pi_0$, provided:
  - the coupling preserves both marginals;
  - the pair stays together after meeting;
  - the meeting time has enough moments.

  The moment conditions take any of three published forms:
  - geometric tails of $\tau$ (JOA20 Prop. 3.1);
  - polynomial tails (Middleton et al. 2020);
  - $\mathbb E_{\pi\otimes\pi}[\tau^\kappa] < \infty$ with $\kappa > 1$, $h \in L^p(\pi)$ with $p > \kappa/(\kappa-1)$, and $d\pi_0/d\pi \le M$ (Douc et al., Thm. 17).

  Under the last, the variance is finite if $\kappa > 2$ and $p > 2\kappa/(\kappa - 2)$.
- **Error bars:** with $R$ independent replicates and finite variance, $\bar H \pm z_{1-\alpha/2}\,\hat\sigma/\sqrt R$ is an asymptotically valid interval as $R \to \infty$. The bounded work per replicate does not change this.
- **Fixed $R$:** unbiasedness holds for a number of replicates fixed before any result is seen. Choosing $R$ after looking at results (for example, extending a run until the error bar is small) is optional stopping: the mean stays consistent and the large-$R$ intervals stay valid, but exact unbiasedness is lost.
- **Total-variation bounds:** meeting times give upper bounds on $d_{TV}(\pi_t, \pi)$, where $\pi_t$ is the law of $X_t$ (§3.6).

**Not guaranteed:**
- a small error at finite $R$;
- valid intervals when the variance is infinite;
- better mixing than the underlying kernel ("coupling pairs of Markov chains does not improve their marginal mixing properties", JOA20 §6);
- estimates inside the range of $h$;
- exact draws from $\pi$;
- robustness to a poor $\pi_0$. On a multimodal target with $\pi_0$ concentrated in one mode, JOA20 §5.1 report a 95% interval of [0.799, 0.816] for a true value of 0.421 at $R = 1000$.

### 3.4 Derived estimators (R-EST-8)

Let $a_r = \hat\pi_r(h) \in \mathbb R^p$ for replicates $r = 1, \dots, R$. Write $\bar a$ for their mean and $S_a$ for their sample covariance (divisor $R - 1$).

- **Expectations:** $\bar a$, with standard error $\sqrt{\mathrm{diag}(S_a)/R}$ and Monte Carlo covariance $S_a / R$.
- **Products of expectations:** $\bar a \bar a^\top - S_a/R$ is unbiased for $\pi(h)\pi(h)^\top$ when $R \ge 2$.
- **Covariance:** $\widehat{\mathrm{Cov}}_\pi(h) = R^{-1}\sum_r \hat\pi_r(hh^\top) - \bar a\bar a^\top + S_a/R$ is unbiased. The plug-in $\sum\omega hh^\top - (\sum \omega h)(\sum\omega h)^\top$ is biased by $-S_a/R$ and is not used. Standard deviations are $\sqrt{\max(0, \widehat{\mathrm{Var}})}$, which is consistent, not unbiased.
- **Histograms:** each replicate's signed mass in each half-open bin $[a, b)$, averaged over replicates. This is unbiased, with standard errors from the replicates. Bins can be negative.
- **Quantiles:**
  1. Pool all replicates' atoms with weights $\omega/R$, and merge atoms with equal coordinates by summing their weights.
  2. Sort along the coordinate and take the cumulative sum of the weights.
  3. Project the cumulative sum onto non-decreasing sequences by isotonic regression with unit weights (`scipy.optimize.isotonic_regression`), then clip to $[0, 1]$.
  4. The $\beta$-quantile is the smallest atom whose projected cumulative weight is at least $\beta$ (the left-continuous inverse).

  This is consistent as $R \to \infty$ but not unbiased (JOA20 Prop. 3.2). Credible intervals use these quantiles. Their uncertainty comes from a bootstrap over whole replicates, labelled heuristic.

### 3.5 Tuning rules (R-TUNE)

- **R-TUNE-1:** kernel parameters left unset (for example the proposal covariance) are tuned on independent pilot chains, then frozen. Pilot chains may adapt. Coupled runs never do.
- **R-TUNE-2:** `k`, `m` and `lag` set to `"auto"` follow AJ24. From a pilot of $R_0 = 100$ meeting times with $L = 1$:
  1. set $k = \max(1, \hat q_{\beta}(\tau - 1))$, where $\hat q_\beta$ is the inverted-CDF sample quantile (`np.quantile(..., method="inverted_cdf")`), an observed integer;
  2. set $L = k$;
  3. set $m = 10k$.

  - **Level:** $\beta$ is 0.99 by default. Larger values give a more conservative mode and smaller values a cheaper one with more replicates per unit of compute. $\beta \le 1 - 1/R_0$ is required, so that $\hat q_\beta$ is not the sample maximum; 0.999 therefore needs $R_0 \ge 1000$.
  - **Pilot size:** with $R_0 = 100$, $k$ varies by about ±20% between pilots (5% to 95% range 117 to 168 at $d = 10$; design review D-R5). This changes the cost and the variance, never the bias.
  - **Partial `"auto"` (D-21):** a value given explicitly is kept. A missing `lag` or `m` is derived from `k` ($L = \max(1, k)$, $m = 10k$) without a pilot, and the pilot runs only when `k` is `"auto"`.
  - **Evidence:** E01 (Appendix A) used a 2000-replicate pilot; the design review measured the 100-replicate pilot.
- **R-TUNE-3:** pilots draw from their own random-stream namespaces (§6.1). Their replicates never enter estimates; they are reported.
- **R-TUNE-4:** the result records every tuned value, the pilot meeting times, and the measured inefficiency: the mean cost per replicate divided by $m - k + 1$. (JOA20 §3.1 approximates it by $1 + (k + \mathbb E[\tau - L])/(m - k)$; with $\bar\tau$ in place of $\mathbb E[\tau - L]$ the approximation is too high by $L/(m-k)$ when $L > 1$.)

### 3.6 Coupling bounds (R-DIAG-1)

For any $L$ and $t \ge 0$ (Biswas, Jacob & Vanetti 2019, Thm. 2.5):

$$d_{TV}(\pi_t, \pi) \le \mathbb E\big[\max(0, \lceil (\tau - L - t)/L \rceil)\big] = \sum_{j \ge 1} \mathbb P(\tau > t + jL).$$

- **Estimate:** the bound is estimated by the replicate average, reported as $\min(1, \cdot)$. The result also reports its leading term at $k$, the fraction of pairs with $\tau - L > k$, with a one-sided 95% Clopper–Pearson upper bound, which stays informative when the fraction is 0.
- **Censored pairs** count as meeting at the cap, so with censoring the estimate is a lower bound. It is reported with the number censored.
- **Use for $k$:** with $L = k$ the bound at $t = k$ is close to $\mathbb P(\tau - L > k)$, which the main run estimates and reports. It exceeds the pilot's $1 - \beta$, because the main run's $\tau - L$ is longer than the pilot's $\tau - 1$: the coupled phase starts from a near-stationary $X_L$. With $\beta = 0.99$ in E01's setting it is 0.019 at $d = 10$ and 0.029 at $d = 30$ (design review D-R5).
- **1-Wasserstein (should):**

  $$d_{W_1}(\pi_t, \pi) \le \mathbb E\Big[\sum_{j=1}^{\lceil (\tau - L - t)/L\rceil} \lVert X_{t+jL} - Y_{t+(j-1)L} \rVert\Big].$$

  It needs the Euclidean distance $\lVert X_s - Y_{s-L}\rVert$ for $L \le s < \tau$, which the engine records. Its value depends on the parametrisation; the documentation says so.

### 3.7 Budgets (should; deferred beyond v0.1, DECISIONS #22)

Averaging only the replicates completed by a deadline is biased by $-\mathbb E[H\,\mathbb 1(C > t)]$ (Glynn & Heidelberger 1990). The exactly unbiased per-lane estimator (GH91 $\bar\mu_1$, JOA20 §3.3) works as follows:
1. Each lane runs replicates in sequence.
2. At the deadline, a lane with no completed replicate finishes its first. A lane with at least one completed replicate discards the one in progress.
3. The estimate is the equal-weight mean over lanes of each lane's mean.

- **Clock:** a lane's clock counts lockstep iterations, so a replicate occupies its lane for $\max(\tau, m)$ ticks, a function of the replicate alone. A wall-clock deadline also depends on the other lanes; it is offered with that caveat.
- **Scope:** budgets run in-process in v0.1. Pools need long-lived lane tasks and are on the roadmap.

"Keep the first $R$ to finish" and "pool everything completed" are never offered.

---

## 4. Inputs

### 4.1 Target (R-INPUT-1)

- **Log-density:** `log_prob(x, *args, **kwargs)` returns a float for `x` of shape `(d,)`. With `vectorize=True` it takes `(n, d)` and returns `(n,)`.
- **Gradient:** an optional `grad` with the same signature returns $\nabla \log\pi$, of shape `(d,)` or `(n, d)`. A kernel that needs a gradient raises `ValueError` without one.
- **Special values:** `-inf` is a valid value and is rejected by every Metropolis step. NaN or `+inf` raises `ValueError` with the offending point. Where the log-density is `-inf`, the gradient may be non-finite, and states there carry a gradient of 0: a gradient kernel proposes from such a point without drift and accepts any proposal of finite density, as RWMH does (DECISIONS #14). Where it is finite, a non-finite gradient raises wherever a cached state would keep it (initial states, and proposals of gradient-free kernels whose states carry gradients); gradient kernels, HMC's meeting step included, reject a proposal that has one.
- **Gibbs:** with a kernel that needs no log-density, a Gibbs kernel or a mixture of them, `log_prob` may be `None` (§5.7).
- **Determinism:** the log-density must return the same value at the same point. A stochastic log-density can split chains that have met. The documentation says so.
- **Layer C:** a `Target` object wraps these functions, counts evaluations, and presents a batched interface `(n, d) -> (n,)` to kernels, whether or not the user's function is vectorised.

### 4.2 Initial distribution (R-INPUT-2)

`init` may be any of:
- an object with `rvs(random_state=rng)`: a frozen SciPy distribution, or a helper (`telescopic.initial.Normal`, `Uniform`, `Empirical`). A draw is normalised to shape `(d,)` (SciPy returns shape `()` for a one-dimensional multivariate normal);
- a callable `init(rng) -> array (d,)`;
- an array of points of shape `(n, d)`, used as the uniform distribution over its rows. This lets emcee users start from an existing ensemble without losing unbiasedness.

Rules:
- $X_0$ and $Y_0$ are independent draws from the replicate's own stream.
- `ndim` is inferred from a draw.
- A draw with the wrong shape or non-finite entries raises `ValueError`.
- A draw with $\log\pi = -\infty$ emits an `InputWarning`; the chains leave it as §4.1 describes.

The documentation recommends a $\pi_0$ that is overdispersed relative to $\pi$ and covers all its modes (JOA20 §2.3; AJ24).

### 4.3 Parameter names

`names` (a sequence of `d` strings) labels summaries and plots. The log-density still receives arrays.

---

## 5. Kernels and couplings

### 5.1 Kernel contract (R-KER-1)

A kernel is a picklable configuration object with:

```text
needs_grad: bool                                            # evaluates the gradient
needs_log_prob: bool                                        # evaluates the log-density; false for Gibbs kernels (DECISIONS #15)
check_target(target)                                        # raises ValueError if the target lacks either (DECISIONS #19)
step(target, state, draws) -> (state, acceptance)          # one transition for n chains; acceptance probabilities
coupled_step(target, sx, sy, draws) -> (sx, sy)             # one coupled transition for n pairs
tune(target, initial, rng=None, *, pilot=None) -> (kernel, report)   # (self, None) if nothing is unset
```

- **States:** `State(x, log_prob, grad)` are batch-first: `x` is `(n, d)`, `log_prob` is `(n,)`, and `grad` is `(n, d)` or `None`. A kernel returns states with the fields of its input states, valid at the returned positions, so a kernel that needs no gradient still evaluates one for states that carry it: mixtures of gradient and gradient-free kernels then keep valid caches. Kernels never modify their input states.
- **Randomness:** `draws` supplies random numbers row by row from each replicate's stream (§6.1). Its `normal(d)`, `uniform()`, `log_uniform()` and `generator(i)` give `(n, d)` normals, `(n,)` uniforms and their logarithms, and row `i`'s `numpy.random.Generator`.
- **Tuning:** `tune` returns a tuned copy. `pilot` passes adaptive Metropolis chains already run, so a `Mixture` tunes its components on one pilot (DECISIONS #5).
- **R-KER-2:** each marginal of `coupled_step` has exactly the law of `step`.
- **R-KER-3:** the engine declares a meeting when the returned positions of a pair are bitwise equal (R-EST-2). A coupling that makes the two proposals identical gives $Y$ a copy of $X$'s proposal and its cached log-density, never a recomputation, so that equality is exact.
- **R-KER-4:** the built-in couplings keep equal states equal. The engine does not rely on this: it never applies `coupled_step` to an equal pair.
- **R-KER-5:** each kernel documents the random numbers it consumes per step.
- **R-KER-6:** a row's arithmetic does not depend on the other rows of the batch. Kernels compute row-wise linear algebra with `np.einsum` (without `optimize`) or elementwise operations, never with BLAS-backed `@` or `solve` on a batch: those give a row different bits in a batch of one than in a larger batch (design review D-R4). R-EXEC-2 depends on this.

### 5.2 Coupling primitives (R-CPL)

All three work on batches and in log space.

**Reflection-maximal coupling** of $\mathcal N(\mu_x, \Sigma)$ and $\mathcal N(\mu_y, \Sigma)$, with $\Sigma = CC^\top$ (R-CPL-1). This is JOA20 §4.1, after Bou-Rabee, Eberle & Zimmer (2020):

```text
delta = C^{-1} (mu_x - mu_y)                      # C^{-1} precomputed by the kernel
z ~ N(0, I);  u ~ U(0, 1);  x' = mu_x + C z
if log u < -(z . delta) - |delta|^2 / 2:  identical; y' = x'      # a copy, never recomputed
else:  e = delta / |delta|;  z_y = z - 2 (e . z) e;  y' = mu_y + C z_y
```
- It is maximal: $\mathbb P(x' = y') = 2\Phi(-\lVert\delta\rVert/2)$.
- Its cost is deterministic: $d$ normals and 1 uniform.
- $\delta = 0$ gives `identical` with probability 1, so $e$ is needed, and computed, only on rows with $\lVert\delta\rVert > 0$. The test is exact: a threshold would copy $Y = X$ for a small non-zero $\delta$ and bias $Y$'s marginal.

**Rejection maximal coupling** of distributions $p$ and $q$ with samplers and normalised log-densities (R-CPL-2). This is JOA20 Algorithm 2 (Thorisson):

```text
X ~ p;  u ~ U(0, 1)
if log u + log p(X) <= log q(X):  return (X, X)
repeat: Y ~ q;  u' ~ U(0, 1)  until  log u' + log q(Y) > log p(Y);  return (X, Y)
```
- The expected number of draws is 2, with variance $2(1 - \mathrm{TV})/\mathrm{TV}$ (research note `mh_couplings.md` §1).
- **Option $\eta \in (0, 1]$** (Gerber & Lee; AJ24 Algorithm 2):
  - the first test becomes $u \le \min(\eta, q(X)/p(X))$;
  - the loop condition becomes $u' > \eta\, p(Y)/q(Y)$;
  - the meeting probability is $\int \min(\eta p, q)$;
  - the variance of the draw count is at most $2\eta/(1-\eta)$.

**Maximal coupling of finite discrete distributions** $p, q$ on $\{1, \dots, K\}$ (R-CPL-3). This is JOA20 §5.4, computed in log space:
- With probability $\alpha = \sum_i \min(p_i, q_i)$, draw $i \sim \min(p, q)/\alpha$ and return $(i, i)$.
- Otherwise draw $i$ and $j$ independently from the normalised residuals.
- The cost is deterministic.

### 5.3 Random-walk Metropolis–Hastings (R-KER-RW)

`RandomWalk(cov=None, *, scale=None, coupling="reflection")`.

**Proposal:** $x' = x + \lambda C z$, where $CC^\top$ = `cov`, $\lambda$ = `scale`, and $z \sim \mathcal N(0, I)$.

**Acceptance:** accept when $\log u < \log\pi(x') - \log\pi(x)$. A current state with $\log\pi(x) = -\infty$ accepts any proposal with a finite log-density and rejects one with $-\infty$; $-\infty - (-\infty)$ is never computed as a number.

**Couplings:**
- `"reflection"` (default; JOA20, O'Leary 2021): reflection-maximal proposals and a common $u$.
  - It uses at most two log-density evaluations per step, one when the proposals are identical.
  - It consumes $d$ normals and 2 uniforms.
- `"maximal"`: rejection maximal coupling with independent residuals, for comparison and teaching. Its meeting times grow exponentially with $d$ (JOA20 §4.2), and the documentation says so.

**Tuning (R-TUNE-1):**
- **Pilot:** 4 adaptive Metropolis chains from $\pi_0$, each running $\max(2000, 200d)$ steps. Each chain adapts its own global scale by Robbins–Monro steps of size $(t+1)^{-0.6}$ toward acceptance 0.234 (Andrieu & Thoms 2008). The chains share one proposal covariance: the running mean and covariance, with step $1/(t+1)$, of all their states, started from the variances of 100 draws from $\pi_0$ (Haario et al. 2001; pooled as in Craiu, Rosenthal & Yang 2009; DECISIONS #7).
- **Covariance:** pooled over the second halves of the chains, plus a ridge of $10^{-10}$ times the mean variance.
- **Scale:** $2.38/\sqrt d$, `RandomWalk`'s default (DECISIONS #7).
- **Report:** the pilot acceptance rates and $\hat R$ per coordinate. A `ConvergenceWarning` when the chains disagree: $\hat R > \sqrt 2$ in some coordinate, where the chain means spread more widely than the chains themselves, a sign of several modes (DECISIONS #6).

### 5.4 MALA (R-KER-MALA)

`MALA(step_size=None, *, cov=None)` requires a gradient. Its only coupling is the reflection-maximal one (DECISIONS #12).

**Proposal:** $x' = x + \tfrac{\epsilon^2}{2} \Sigma\nabla\log\pi(x) + \epsilon C z$, with $\Sigma = CC^\top$ = `cov`. The acceptance ratio includes the forward and reverse proposal densities. A proposal with $\log\pi(x') = -\infty$ is rejected whatever its gradient.

**Coupling:** reflection-maximal coupling of the two Gaussian proposals, which share the covariance $\epsilon^2\Sigma$, plus a common $u$ (JOA20 §4.4; AJ24 §4.3).

**Tuning:** `cov` from the RWMH pilot of §5.3, then $\epsilon$ by dual averaging toward acceptance 0.574, on 4 chains started from the pilot's last states (Hoffman & Gelman 2014, §3.2.1, with their constants; 1000 iterations from $\epsilon_0 = 1.65\,d^{-1/6}$, on the chains' mean acceptance probability; DECISIONS #13).

### 5.5 HMC (R-KER-HMC)

`HMC(step_size=None, n_steps=None, *, cov=None, meeting_weight=0.05, meeting_scale=1e-3)` requires a gradient. It follows Heng & Jacob (2019).

**Marginal kernel.** With probability $1 - \gamma$ ($\gamma$ = `meeting_weight`, the mixture weight of the meeting component), take a leapfrog step:
- whitened momentum $\nu \sim \mathcal N(0, I)$;
- `n_steps` leapfrog updates of size $\epsilon$: $\nu \leftarrow \nu + \tfrac{\epsilon}{2}C^\top\nabla\log\pi(x)$, then $x \leftarrow x + \epsilon C\nu$, then the half step again;
- accept on the change in $\log\pi(x) - \tfrac12\lVert\nu\rVert^2$;
- a trajectory that reaches a non-finite position, a point with $\log\pi = -\infty$ or a non-finite gradient stops there and is rejected, without raising.

With probability $\gamma$, take a RWMH step with proposal covariance $\sigma^2\Sigma$, where $\sigma$ = `meeting_scale`; like the trajectories, it rejects a proposal with a non-finite gradient (DECISIONS #18). Heng & Jacob use $\sigma^2 I$; the whitened $\sigma^2\Sigma$ makes the component invariant to the scale of each coordinate (D-24).

**Coupling:**
- a common uniform chooses the component, so both chains use the same one;
- the leapfrog component shares $\nu$ and the accept uniform, so the chains contract but do not meet;
- the RWMH component uses the reflection-maximal coupling of §5.2, so close chains meet exactly.

Heng & Jacob used the rejection maximal coupling for this step. Both couplings are maximal; reflection has deterministic cost (D-11).

**Tuning:**
- `cov` from the RWMH pilot;
- $\epsilon$ by dual averaging toward acceptance 0.8 of the trajectories alone, as for MALA but from $\epsilon_0 = d^{-1/4}$ (DECISIONS #13);
- `n_steps` $= \lceil 1/\epsilon \rceil$, a trajectory of length about 1 in whitened coordinates, as in Heng & Jacob, and at most 1,000 (DECISIONS #17). For a Gaussian direction of whitened frequency $\omega$ the difference between the chains scales by $\cos(\omega\,\epsilon\,n)$, which stays below 1 in magnitude for $\omega < \pi$. A length of $\pi/2$ would contract most at $\omega = 1$ but not at all at $\omega = 2$, a direction whose variance the pilot underestimates fourfold (design review D-R12; D-24).

The documentation says that settings optimal for HMC alone can fail to contract when coupled (Heng & Jacob §5), and that meeting times should be checked.

### 5.6 Independence Metropolis–Hastings (R-KER-IMH)

`Independence(proposal=None)`. `proposal` has `sample(draws)`, returning `(n, d)` from the buffered normals, and a batched, normalised `log_density(x)`. `MultivariateT(mean, cov, df=5)` is the built-in.
- **Acceptance:** accept when $\log u < [\log\pi(x') - \log q(x')] - [\log\pi(x) - \log q(x)]$.
- **Coupling:** both chains share the proposal $x'$ and $u$, so they meet when both accept. This is already a maximal kernel coupling (O'Leary, Wang & Jacob 2021).
- **Default proposal:** a multivariate Student-$t$ with 5 degrees of freedom, located at the RWMH pilot's mean, with the pilot's covariance as its scale matrix, so its covariance is $5/3$ of the pilot's (D-25). The tuning report gives the largest importance-weight ratio $\pi/q$ seen in the pilot, relative to the median.
- **Documented caveats:** meeting times are geometric only for bounded importance weights. A poor proposal gives heavy-tailed meeting times.

### 5.7 Gibbs (R-KER-GIBBS)

`Gibbs(conditional, blocks, *, coupling="maximal", eta=1.0, threshold=None)`.
- **Blocks:** a sequence of integer index arrays that partition the coordinates.
- **Conditionals:** `conditional(x, b)` receives one state `x` of shape `(d,)` and a block number `b`, and returns the full conditional of block `b` given the rest of `x`. It is a frozen SciPy-like distribution with `rvs(random_state=...)` and a normalised `logpdf` (or `logpmf` for discrete ones). Sweeps loop over rows.
- **Sweep:** blocks are updated in the given order.
- **Couplings, per block:**
  - `"maximal"`: rejection maximal coupling (§5.2), with the $\eta$ option;
  - `"crn"`: common random numbers, which contract but do not meet on their own (an `eta` other than 1 raises: DECISIONS #16): a common uniform through `ppf` for blocks of one coordinate, and common standard normals for multivariate normal conditionals (objects with `mean` and `cov`, such as `scipy.stats.multivariate_normal`);
  - `"two-scale"`: CRN while $\lVert x - y\rVert >$ `threshold`, maximal otherwise (Biswas et al. 2022). The switch depends only on the pair at the start of each sweep, and both branches couple the same conditional, so it is valid (AJ24 §4.2). `threshold` is required.
- **Identical conditionals:** when the coordinates outside block `b` are bitwise equal in the two chains, their conditionals are identical, and both chains receive the same draw. This identity coupling is a valid state-dependent choice; it also keeps equal states equal when $\eta < 1$, which the rejection coupling alone would split with probability $1 - \eta$ (design review D-R2).
- **Meeting:** the engine's exact comparison after each sweep (R-EST-2). Blocks updated by CRN can become equal without a coupling event.
- **Log-density:** a Gibbs kernel needs none (`needs_log_prob` is false), nor does a mixture of Gibbs kernels. With `log_prob=None`, states carry log-density 0, the `InputWarning` check of §4.2 is skipped, and there is nothing to tune.

### 5.8 Mixtures (R-KER-MIX)

`Mixture(kernels, weights)`. A common uniform chooses the same component for both chains, and that component's coupling is applied.

The choice of component must not depend on the states (AJ24 §4.2). Switching between couplings of the same kernel may depend on them (§5.7).

---

## 6. Execution

### 6.1 Replicates and random streams (R-EXEC-1)

- **Root seed:** `rng` (SPEC 7) is normalised to a root `SeedSequence`. `None` draws fresh entropy, an integer seeds it, and a `SeedSequence` is used as given. A `Generator` gives the `SeedSequence` of four integers drawn from it: the run is then a function of the generator's state, and passing the same generator twice gives two different runs (D-26). The root entropy and spawn key are recorded.
- **Per-replicate streams:** replicate `i` in namespace `ns` has the generator `G = Generator(PCG64DXSM(SeedSequence(root.entropy, spawn_key=root.spawn_key + (T, ns, i))))`, where the branch `T = 0x74656C` ("tel" in ASCII) keeps these keys away from the small integers that `SeedSequence.spawn` gives a caller's own children of `root`.
  - `G` draws $X_0$, then $Y_0$, and serves `draws.generator(i)`.
  - The replicate's normals and uniforms come from `G.spawn(2)`: the generators with spawn keys `+ (0,)` and `+ (1,)` under `G`'s seed sequence. Each is read in order (D-8).
- **Buffering:** each lane pre-draws its replicate's normals and uniforms into buffers, so a lockstep batch gathers them with array indexing. A request that needs more numbers than a buffer holds tops it up and keeps the unread ones, so the buffer width changes no number.
- **Namespaces:** 0 for main replicates, 1 for tuning pilots, 2 for meeting-time pilots. Each call of `tune`, `meeting_times` or `run` continues the indices of its namespace.
- **R-EXEC-2:** a replicate's output is a function of `(root, ns, i)` and its configuration only. It is bitwise identical across batch sizes, worker counts and executors when the log-density is evaluated point by point (R-KER-6), on one machine with the same NumPy and SciPy versions: generator algorithms, SciPy samplers and BLAS builds can change between versions and machines. A vectorised log-density can differ in the last bits between batch layouts, so there the guarantee is statistical. The documentation says so.

### 6.2 Engine (R-EXEC-3)

The engine runs replicates in **lanes**: a lockstep batch of `B` slots, each holding one replicate.

- **Phases:** each slot is in one of three phases: lagging ($X$ alone, $t < L$), coupled, or after-meeting ($X$ alone until $t \ge m$).
- **Each step:**
  1. one batched kernel step for the slots in the lagging or after-meeting phase;
  2. one batched coupled step for the coupled slots;
  3. bookkeeping of the meeting time (the exact comparison of R-EST-2), cost and estimator terms.
- **Refill:** a finished slot takes the next replicate index, until the requested indices run out.
- **Storage:** the engine stores the atoms and weights of §3.2 as it goes, starting at $t = k$. It also stores the distance $\lVert X_t - Y_{t-L}\rVert$ for $L \le t < \tau$.

### 6.3 Pairs that do not meet (R-EXEC-4)

- **No cap by default:** in the main run, `max_iter=None` runs every pair until it meets, as the theory requires.
- **The cap:** `max_iter` caps $t$ and must exceed $\max(L, m)$. A pair still unmet when $t$ reaches it is **censored**:
  - it is listed in the result, with meeting time `inf`;
  - it is excluded from the estimates, from $R$ and from the signed measure;
  - it is reported by a `ConvergenceWarning` stating that the estimates are biased.
- Censored pairs are never dropped silently, and never averaged as if they had met.
- **Slow-pair warning:** when a pair passes $t = \max(10^4, 10\max(L, m))$, the sampler emits one `ConvergenceWarning` naming the replicate: in this process from the engine's progress reports, as it happens; on a pool, whose workers' reports do not reach the parent, from the meeting times when the run ends. With the recipe of R-TUNE-2 and $m \ge 1000$ this is 100 times the pilot's $\hat q_\beta$; it needs no pilot, and in the meeting-time pilot ($L = 1$, $m = 0$) it is $10^4$.
- **Meeting-time pilot:** it is capped at `pilot_max_iter` (default $10^5$). Capped pairs are censored and sort above every finite meeting time; if $\hat q_\beta$ falls on a censored pair, the pilot raises `ValueError` with advice (a wider $\pi_0$, a better kernel, or explicit `k`, `lag` and `m`). So `sample()` cannot run forever on a target where the coupling never meets (D-23).

### 6.4 Parallel execution (R-EXEC-5)

- **`pool` values:**
  - `None` runs in-process;
  - an integer creates that many worker processes;
  - an object with `submit` (a `concurrent.futures.Executor`), or with `map` (emcee/dynesty/schwimmbad style), is used as given.
- **Tasks:** chunks of consecutive replicate indices, one per worker at least, more where a chunk would exceed a full set of 64 lanes, and four per worker at most. The workers are counted by the pool's size: an executor's `_max_workers`, a multiprocessing pool's `_processes`, a schwimmbad pool's `size`, or else the CPUs the process may use. Each worker runs the engine on its chunk. Chunks are submitted at once, in index order; a pool with only `map` runs them in waves of one per worker. The job (target, kernel, initial distribution, root and settings) travels with each task (DECISIONS #23).
- **Results:** reduced in index order. The wall time of a pool run is the parent's.
- **Worker hygiene:**
  - a task in the main thread of another process pins BLAS to one thread and ignores Ctrl-C, in user pools too. Tasks in the parent process, as on a thread pool, or in other threads leave those settings alone: they belong to the process, and concurrent tasks would restore each other's;
  - pools that telescopic creates start their workers with the spawn method, which behaves alike on every platform; forking a process whose BLAS runs threads can deadlock. Their workers ignore Ctrl-C between tasks too, and on POSIX from their start, since they are made while the parent ignores it. On Windows, `ProcessPoolExecutor` takes at most 61 workers, and a larger `pool` raises at once;
  - the workers of an integer pool live for one call of `run` or `meeting_times`: they start before tuning, so that their start overlaps it, and serve the pilot and the run.
- **Picklability:** telescopic ships the target, the initial distribution and the kernel (which can hold user functions, such as Gibbs conditionals) to the workers it creates with `cloudpickle` when it is installed (the `parallel` extra), so functions defined in a notebook work under the spawn and forkserver start methods (D-27). Otherwise they must be picklable, and the error says how to fix it. The documentation covers the `if __name__ == "__main__"` guard.
- **Counters:** each task returns its own numbers of log-density and gradient evaluations, since the `Target` counters of a worker process are not the parent's.

### 6.5 Interruption (R-EXEC-6)

- **First Ctrl-C:** no new replicate starts, the ones in progress finish, and the result is flagged `interrupted`. It contains every started replicate. The workers that telescopic creates hear of it through an event: they start no new replicate, and a chunk that starts afterwards runs nothing, except the first, so that a run returns a replicate. On a user's executor the chunks not started are cancelled and the running ones finish; a `.map` pool returns its completed waves only. How the rest of a user's pool reacts to Ctrl-C is up to the pool.
- **Requested indices:** a result records the indices it was asked for, and `interrupted` is true while one of them has not run. In this process the started replicates form a prefix of the indices; on a pool they need not. The sampler's next run starts with the missing indices, and `merge` joins the requests, so a run that completes an interrupted one clears the flag (DECISIONS #20).
- **Its estimates are consistent, not exactly unbiased:** the number of replicates depends on their costs, which correlate with $h$ through $\tau$, so the mean is a ratio estimator (Glynn & Heidelberger 1991). The bias is visible with few lanes (one lane: $z = -12.9$ in design review D-R3) and was not detectable with 16 or more. `summary()` says so. Running the remaining replicates of the requested $R$ restores exact unbiasedness, because $R$ is then fixed in advance (D-22).
- **Second Ctrl-C:** raises `KeyboardInterrupt` at once, without a result, and stops the workers that telescopic started.
- **Scope:** tuning and the meeting-time pilot have no result to return early: Ctrl-C there raises `KeyboardInterrupt`.

### 6.6 Memory (R-EXEC-7)

- **Atoms:** storage needs about $R\,(|S| + 2\,\overline{\max(0, \tau - k - L)})\,(8d + 16)$ bytes for positions, weights and replicate indices, plus 24 bytes per stored distance. The engine stores each iteration's rows as small arrays and merges every 256 of them into one block, so at $d = 10$ a row holds 94 to 103 bytes against 92 of payload at 1 to 256 lanes (M3 review).
- **Join:** at the end of a run, sorting the atoms by replicate holds about three times their size for a moment (3.1 times at $d = 10$).
- **Estimate first:** the sampler estimates this before the main run and warns above 1 GB, suggesting `thin`.
- **Streaming:** a streaming mode with a fixed `h` and constant memory is on the roadmap.

---

## 7. Outputs

### 7.1 `Result` (R-OUT-1)

Fields:
- `k`, `m`, `lag`, `thin`, `n_replicates`, `ndim`, `names`;
- `kernel` (the frozen kernel; `save` stores its description, and `load` restores the description, not the object) and `initial` (a description of $\pi_0$);
- `seed` (root entropy and spawn key);
- `meeting_times` (censored as `inf`) and `costs` (kernel applications) for every replicate, censored ones included;
- `n_log_prob`, `n_grad` and `wall_time` for the run;
- `measure` (a `SignedMeasure`);
- `tuning` and `pilot` (reports);
- `interrupted`: whether a requested replicate has not run (§6.5).

Methods:
- **Estimates:**
  - `estimate(h, level=0.95)` returns an `Estimate`. `h` maps `(n, d)` to `(n,)` or `(n, p)`; any other output shape raises `ValueError`;
  - `mean()`, `cov()`, `var()`, `std()` follow §3.4.
- **Quantiles and histograms:** `quantile(q)`, `interval(mass=0.68)` (the central interval holding that mass of $\pi$; `level` is reserved for confidence levels) and `histogram(dims, bins=20, range=None)`, of one coordinate or a joint histogram of several, with standard errors (DECISIONS #30).
- **Diagnostics:** `tv_bound(t=None)`, `w1_bound(t=None)` (should), and `diagnostics()` (a dict).
- **Reporting:** `summary(level=0.95)` returns a `Summary` table. `check(strict=False)` emits the warnings of §7.4, or raises if strict.
- **Plots:** `plot_meeting_times(axes=None)`, `plot_tv_bound(ax=None)`, `plot_marginals(dims=None, bins=20, axes=None)` and `plot_corner(dims=None, bins=20)` (signed 2-D histograms); the first three draw on given axes (DECISIONS #30).
- **Persistence and merging:**
  - `save(path)` and `Result.load(path)` use a versioned `.npz` schema, with no pickle: arrays, and the settings, names, seed state and descriptions as JSON (DECISIONS #26, #29). The engine's paths are not saved;
  - `merge(other)` joins runs with the same configuration and disjoint replicate indices.

### 7.2 `SignedMeasure` and `Estimate` (R-OUT-2)

**`SignedMeasure`** holds three arrays:
- `atoms` `(N, d)`;
- `weights` `(N,)`, each replicate's summing to 1;
- `replicate` `(N,)`, the index into the result's replicates.

Its `per_replicate(h)` returns the `(R, ...)` values $\hat\pi_r(h)$.

**`Estimate`** is frozen. It holds `value`, `standard_error`, `interval` (low, high), `level` and `replicates` (the per-replicate values), and prints as value ± standard error.

### 7.3 Summary (R-OUT-3)

**Per parameter:** the mean ± standard error, the standard deviation, and the 16th, 50th and 84th percentiles.

**Footer:**
- $R$, $k$, $m$, $L$, `thin`;
- the kernel and its coupling;
- meeting-time median, 90th and 99th percentiles, maximum, and number censored;
- mean cost per replicate, in kernel applications and in seconds;
- the measured inefficiency, mean cost / $(m - k + 1)$ (R-TUNE-4);
- the fraction of replicates with a non-zero bias correction, which estimates the TV bound at $k$ (§3.6);
- for an interrupted run, that its estimates are consistent but not exactly unbiased (§6.5);
- (should) $\hat R$ across the replicates' X-chains;
- warnings raised.

**Output forms:** `repr` prints the table. It also converts with `to_dict()`, and with `to_pandas()` when pandas is installed.

### 7.4 Warnings (R-DIAG-2)

Classes: `TelescopicWarning`, with subclasses `ConvergenceWarning` and `InputWarning`.

`check()` warns on these conditions, and so does each `run` (§8.1). They are heuristics; the thresholds are set so that clean Gaussian runs at the defaults rarely trigger them, and M9 measures and documents each false-alarm rate (D-16):
1. any censored replicate;
2. a fraction of replicates with $\tau - L > k$ above 0.25 ("k may be too small"). The recipe itself gives 0.02 to 0.13 (§3.6), so a lower threshold fires on clean runs;
3. one replicate contributing more than 25% of the sample variance of any reported estimate, with $R \ge 100$ (heavy tails or rare excursions). At $R = 20$ to 30 that share is typical of Gaussian data;
4. a Hill estimate of the tail index of $\tau - L + 1$ below 2 (finite variance in doubt), from the $\max(10, \lfloor R/10\rfloor)$ largest values, with $R \ge 500$. Applied to $\tau$ itself with $L = k$, the shift by $L$ removes almost all its power;
5. pilot chains that disagree (§5.3);
6. a slow pair (§6.3).

The documentation explains each, with the failure cases of JOA20 §5.1, and says that a run that never left one mode cannot be detected from its own output at moderate $R$ (design review D-R9).

### 7.5 Plotting (optional dependency)

- **Meeting times:** a histogram of $\tau - L$, and a log-scale survival curve.
- **TV bound:** the bound against $t$ on a log scale, with $k$ marked.
- **Marginals:** signed histograms with ±2 standard-error boxes. Negative bins are allowed, and the docstring explains them.
- **Corner plot:** signed 2-D histograms. Telescopic never passes signed weights to `corner` or ArviZ, whose weighted quantiles assume non-negative weights.
- **Paths:** the coupled paths of a few replicates, for teaching, available when the engine runs with `keep_paths=True`.

---

## 8. API

### 8.1 Layers (R-API-1)

1. **One call:**

   ```python
   telescopic.sample(log_prob, init, n_replicates=100, *, args=(), kwargs=None, grad=None,
                     vectorize=False, kernel="auto", k="auto", m="auto", lag="auto", thin=None,
                     max_iter=None, pool=None, names=None, progress=False, rng=None) -> Result
   ```

   `kernel` is a `Kernel` or one of the strings `"auto"`, `"rwmh"`, `"mala"`, `"hmc"` and `"independence"`. `"auto"` means RWMH, with or without a gradient; gradient kernels are requested by name (D-3).
2. **Sampler object:**

   ```python
   sampler = telescopic.CoupledSampler(log_prob, init, *, args=(), kwargs=None, grad=None,
                                       vectorize=False, kernel="auto", names=None, pool=None,
                                       progress=False, rng=None)
   sampler.tune()                                  # pilot runs; freezes sampler.kernel
   times = sampler.meeting_times(n=100)            # pilot meeting times, lag 1
   result = sampler.run(500, k="auto", m="auto", lag="auto", thin=None, max_iter=None,
                        progress=None)             # None: the sampler's setting
   result = sampler.run(500)                       # 500 more replicates, same k, m, lag
   ```

   - `run` tunes and runs the pilot if they have not been done.
   - Each call continues the replicate indices and returns the cumulative result.
   - The first call fixes `k`, `m`, `lag`, `thin` and `max_iter` (DECISIONS #4). On the first call `"auto"` derives a value, `thin=None` means 1 and `max_iter=None` no cap; on later calls `"auto"` and `None` keep the first call's values, and any other change raises `ValueError` (DECISIONS #8).
   - Each call ends by emitting the warnings of §7.4 that apply to the cumulative result, except slow pairs and disagreeing tuning chains, which are emitted as they happen, or for slow pairs on a pool as each run ends (§6.3, DECISIONS #8).
3. **Building blocks:** `Target`, `State`, `telescopic.kernels.*`, `telescopic.couplings.*`, `telescopic.engine.run(...)`, `telescopic.estimators.*`, `telescopic.diagnostics.*` and `SignedMeasure`.

### 8.2 Conventions (R-API-2)

- **Randomness:** every function that draws random numbers takes `rng` (SPEC 7). There is no global state.
- **Shapes:** batch first. States are `(n, d)`; test functions map `(n, d)` to `(n,)` or `(n, p)`; per-replicate values are `(R, ...)`.
- **Errors:** `ValueError` or `TypeError` with messages that name the argument and say what was expected. Questionable inputs emit `InputWarning`; reliability problems emit `ConvergenceWarning`.
- **Progress:** `progress=True` uses tqdm when it is installed, and falls back to periodic lines on stderr.
- **Types and documentation:** public signatures are typed, with NumPy-style docstrings.
- **Vocabulary:** the documentation says "replicates" and "pairs", never "walkers".

### 8.3 Minimal example (the README must show this)

```python
import numpy as np
import telescopic

def log_prob(theta, x, y, yerr):
    slope, intercept = theta
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2)

init = telescopic.initial.Normal(mean=[0.0, 0.0], cov=np.diag([10.0, 10.0]))
result = telescopic.sample(log_prob, init, args=(x, y, yerr), names=["slope", "intercept"],
                           rng=42)
print(result.summary())                      # unbiased means, standard errors, intervals
result.estimate(lambda t: t[:, 0] * t[:, 1])  # any function, after the run
result.plot_tv_bound()                       # with the plot extra: distance from stationarity
```

---

## 9. Validation tools (user-facing, must)

**`telescopic.examples`** provides targets with exact answers, used by the tests, tutorials and benchmarks. Each `Example` has `log_prob` and `grad` (for a point or a batch), a default `initial` with at least the target's variance in each coordinate, the exact `mean` and `cov`, and `ndim`; the Gibbs model also has `conditional` and `blocks` (DECISIONS #24, #27). The targets are:
- a correlated Gaussian;
- the banana of Heng & Jacob, with exact moments;
- a bimodal Gaussian mixture;
- Neal's funnel;
- a conjugate hierarchical model for Gibbs (the pump-failure model of JOA20), whose exact expectations come from one-dimensional quadrature over $\beta$ after integrating out $\lambda$.

**`telescopic.testing.check_coupling(kernel, target, x, y, n, rng)`** checks a kernel's coupling as the engine uses it. From `n` copies of `(x, y)`, it compares each chain of one `coupled_step` with one `step` from the same state, and inspects the pairs:
- marginals: Fisher's exact test of the frequency of staying put, and two-sample KS tests of the moved values on 5 random one-dimensional projections (1 in one dimension) and on the distance moved, Bonferroni-corrected;
- meetings: the fraction of pairs that met, which must be positive;
- caches: the chains whose returned log-density or gradient differs from a fresh evaluation by more than $10^{-9}$ of its size (the engine reuses them);
- near misses: pairs apart by at most 64 units in the last place in every coordinate without being equal, the mark of a coupling that recomputes $Y$ where it should copy $X$'s proposal (R-KER-3).

It returns a report with the p-values and counts, and `passed(alpha=1e-3)` (DECISIONS #28). Faithfulness is not checked: the engine moves the chains of a pair that met with `step` alone. Custom kernels should pass it before use. `telescopic.testing` also provides `fixed_draws` (prescribed random numbers, for draw-for-draw tests) and `random_draws` (fresh random numbers for any number of rows).

---

## 10. Performance targets (v0.1, one CPU core)

These targets are measured by `benchmarks/` and are not gating.

| Operation | Target |
|---|---|
| Lockstep coupled RWMH, vectorised Gaussian target, $d = 10$, 256 lanes, main-run settings ($k = L = 141$, $m = 1410$) | ≤ 2 µs per pair-step (research note measurement: 0.37 µs at 1000 lanes without per-replicate streams). Met at M3: 1.2 to 1.5 µs. Meeting-time pilots ($L = 1$, $m = 0$) take 4.2 µs: each replicate's set-up, about 80 µs, adds 1.8 µs per pair-step at a mean $\tau$ of 46 (M3 review) |
| Same, point-by-point log-density, 1 lane | ≤ 40 µs per pair-step of overhead beyond the log-density. Missed at M3: 93 µs in the main run and 150 µs in the pilot. A coupled RWMH step on one row takes 112 µs, of which its two log-density calls through `Target` take 15 µs; the target needs a scalar path for one lane, which v0.1 does not have |
| Per-replicate random streams in lockstep | ≤ 0.5 µs per lane-step for the draws of a coupled RWMH step at 256 lanes, $d = 10$ (one shared generator: 0.15 µs), and ≤ 50 µs per replicate to set up its streams |
| Process pool, log-density ≥ 1 ms | ≥ 3.5× speed-up on 4 workers |
| Automatic pipeline overhead (both pilots) | ≤ 15% of the main run at the default $R = 100$ for $d \ge 10$. At $d = 2$ the fixed tuning pilot makes it about 30% (design review D-R13); it falls as $1/R$ |

---

## 11. Engineering requirements (R-ENG)

- **Packaging:** `pyproject.toml` (hatchling), `src/` layout. The distribution and import name is `telescopic`, which was free on PyPI on 2026-10-23.
- **Licence:** MIT (QUESTIONS).
- **Python:** ≥ 3.11. CI covers 3.11 to the newest release.
- **Dependencies:**
  - core: `numpy>=1.26`, `scipy>=1.12` (for `scipy.optimize.isotonic_regression`) and `threadpoolctl`;
  - extras: `plot` (matplotlib), `progress` (tqdm), `pandas` and `parallel` (cloudpickle);
  - dependency groups: `test`, `dev` and `docs`.
- **Quality tooling:** ruff (lint and format, line length 99) and mypy on `src/`. pytest runs with warnings as errors; hypothesis provides property tests. pre-commit runs ruff.
- **Tests:**
  - a fast tier under 2 minutes, and a `slow` statistical tier;
  - at least 90% line coverage of `src/`.
- **Documentation:**
  - toolchain: Sphinx, MyST-NB (tutorials executed at build time), numpydoc, furo, sphinx-copybutton;
  - pages: Getting started, User guide, Theory, Tutorials, API reference, Design notes, Changelog, Contributing.
- **CI (GitHub Actions):**
  - lint and types;
  - tests on Linux, macOS and Windows, with a job on the oldest supported stack;
  - a docs build;
  - a wheel and sdist build with a smoke test;
  - the slow tier on manual dispatch.
- **Repository files:** `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md` (with the validation philosophy), `LICENSE`.

---

## 12. Roadmap beyond v0.1

1. **JAX backend:**
   - pure kernels over pytrees;
   - `fold_in(root, i)` streams;
   - a persistent-lanes `while_loop` with in-loop refill (never `vmap` over whole replicates: 2–11× waste);
   - autodiff gradients.
2. **Experiments under DECISIONS #1:**
   - coupled elliptical slice sampling with coupled pCN meeting steps (unpublished: met at $d \le 200$ in the census check);
   - coupled NUTS;
   - re-tuning between replicate batches (martingale argument; needs its own error bars);
   - surrogate-gradient GCRN.
3. **Variance reduction:**
   - Craiu–Meng control variates;
   - Sherlock's Rao–Blackwellisation of the bias correction;
   - unbiased asymptotic-variance reports (UPAVE).
4. **Families:** particle MCMC (CCPF, PMMH), piecewise-deterministic samplers, partitions, and tempering for multimodal targets.
5. **Outputs:**
   - budgets: the per-lane estimator of §3.7 (DECISIONS #22);
   - streaming estimators with constant memory;
   - HDF5/Zarr checkpoints with mid-replicate snapshots;
   - nonlinear functionals by randomised multilevel Monte Carlo.

---

## 13. Decisions taken autonomously, and open questions

These were decided without Josh's input so that work could proceed. Each is easy to revisit.

| # | Decision | Rationale | Alternatives |
|---|---|---|---|
| D-1 | NumPy/SciPy core in pure Python, with optional extras. JAX is on the roadmap | emcee-like install-anywhere usability; Josh's brief ("hopefully pure Python") | JAX-first core |
| D-2 | Layered API: `sample` → `CoupledSampler` → building blocks | the pattern of every praised sampler (research note `api_ux_samplers.md` Q5) | a single sampler class |
| D-3 | Default kernel RWMH with reflection-maximal coupling, with or without a gradient; MALA and HMC on request | best-published gradient-free option (O'Leary 2021 Table 1: mean τ 30 vs 279 for independent residuals at $d = 10$). MALA can beat RWMH when gradients exist (Papp & Sherlock 2025 §7.4), but its transient from the overdispersed $\pi_0$ the docs recommend is sensitive to the step size (Christensen et al. 2005, cited in `mh_couplings.md`; design review D-R25); HMC needs contraction checks (Heng & Jacob §5) | MALA whenever a gradient is given (the first draft); HMC default with gradients |
| D-4 | Pilot, then freeze. Covariance from adaptive Metropolis pilots | adaptation inside coupled runs breaks unbiasedness (AJ24 §5.2) | user-supplied tuning only; ensemble pilot (roadmap) |
| D-5 | `k`, `m`, `lag` "auto" by AJ24's recipe, from 100 pilot meeting times, with the inverted-CDF quantile and $\beta \le 1 - 1/R_0$ | best-supported published default. E01 (2000-replicate pilot): unbiased, 2–6% better inefficiency than lag 1 at the same $k$, $m$. The inverted-CDF quantile is an observed meeting time, and the bound on $\beta$ keeps it below the sample maximum (design review D-R5) | $k$ = 90% quantile (CoupledHMC.jl); Douc et al.'s $m = 5k$; linear interpolation (NumPy's default), which E01 floored |
| D-6 | $\tau = \inf\{t \ge L : X_t = Y_{t-L}\}$ | matches JOA20 for $L = 1$ and AJ24 in general; estimator values do not depend on the convention | $\inf\{t > L\}$ (Biswas et al.) |
| D-7 | Meetings detected by the engine's exact comparison of positions, once at $t = L$ and after each coupled step; couplings copy $X$'s proposal for $Y$ on the meeting branch; only $X$ is simulated after the meeting. Kernels return states only | recomputing $Y$ gave bitwise-different states in 100% of 20,000 cases (research note `mh_couplings.md` §5), and a recomputing coupling met 0.2% instead of 94% of the time (`validation_benchmarks.md` Q1). A kernel flag (the first draft) missed equal pairs that both reject (65% of RWMH steps from equal states) and meetings at $t = L$ (45–57% of lag-1 pilot pairs with a point mass $\pi_0$; design review D-R1) | a `met` flag returned by kernels; tolerances (ColCarroll's `np.isclose`, unsafe) |
| D-8 | Per-replicate streams `(root, namespace, index)` with PCG64DXSM. Normals and uniforms come from two children of the replicate's generator, each read in order through buffers that keep unread numbers | results invariant to batching and workers (Pigeons-style). With separate children, no buffer width enters any number, so widths are free performance settings (M0 altitude review). Measured cost (M0 review, `notes/reviews/m0.md`): 0.33 µs per lane-step for the draws of a coupled RWMH step at 256 lanes, $d = 10$, against 0.15 µs for one shared generator (2.2×; 3.2× for replicates of about 110 steps), plus 45 µs per replicate to build its three generators. The research prototype's 1.4× had no per-lane cursors | one generator per batch: fastest, but depends on the batch layout; one generator per replicate feeding both buffers and discarding on refill (the M0 code): block size, uniform width and $d$ then enter every seeded number |
| D-9 | One engine: lockstep lanes with refill; pools distribute chunks of indices to engines | one code path for every executor; refill never biases (`parallel_engineering.md`) | separate scalar and vectorised drivers |
| D-10 | Fixed-$R$ mode by default; a budget mode with the per-lane GH rule only | exactly unbiased; biased budget rules are never offered | pooled completion within the budget (biased) |
| D-11 | HMC's meeting step uses the reflection-maximal coupling instead of Heng & Jacob's rejection coupling | both are maximal, with the same meeting probability per step; reflection has deterministic cost and vectorises | rejection coupling exactly as published |
| D-12 | Default storage: signed-measure atoms from $t = k$, so any $h$ can be applied after the run, with `thin` and a memory warning. This departs from the census report, which recommends per-replicate estimates by default | users want quantiles, histograms and plots, which need atoms; memory is modest at emcee scale (about 18 MB at $R = 256$, $\lvert S\rvert = 900$, $d = 10$); the atoms from $k$ to $m$ also give $\hat R$ across replicates | per-replicate estimates of fixed $h$ (constant memory; roadmap) |
| D-13 | No meeting cap by default in the main run. With a cap, censored pairs are excluded and reported as biasing the result. The meeting-time pilot is capped (D-23) | the theory needs every pair to meet; silent truncation is the R package's hazard | a default cap with silent `inf` (unbiasedmcmc) |
| D-14 | Unbiased covariances by the U-statistic; quantiles by an isotonic projection of the pooled signed CDF | exact where possible (§3.4); isotonic regression is the $L^2$ projection onto monotone curves. A running maximum is another monotonisation; both are consistent and biased at finite $R$, and gave the same quantiles in the design review's test | plug-in covariance (biased); a running maximum; the crossing rule (non-unique) |
| D-15 | Literature notation for `k`, `m`, `lag` in the API | users read JOA20 and AJ24 alongside the docs; "auto" covers newcomers | `burn_in`, `n_iter` |
| D-16 | Warning thresholds from §7.4, documented as heuristics: warning 2 at 0.25, warning 3 only for $R \ge 100$, warning 4 on $\tau - L + 1$ for $R \ge 500$ | no published calibration exists (`diagnostics_practice.md` §4). The first draft's thresholds fired on clean Gaussian runs at the defaults (warning 2 with probability 0.25 at $d = 10$ and 0.49 at $d = 30$; warning 3 with probability 0.73 at $R = 30$ for 10 estimates; design review D-R6) | no warnings; strict errors; binomial tests at a stated level |
| D-17 | Docs with Sphinx, MyST-NB and furo | furo is the lead's most recent choice (GSPICE); MyST-NB executes the tutorials | pydata-sphinx-theme (pastry) |
| D-18 | Python ≥ 3.11, NumPy ≥ 1.26, SciPy ≥ 1.12 | the floors of Josh's other packages; SciPy 1.12 adds `isotonic_regression` | SPEC 0 floors (3.12, NumPy 2.0) |
| D-19 | Implement from the papers, never from the GPL R code. The R package is a reference for numbers only | the MIT licence; `software_landscape.md` licence analysis | port the R code under GPL |
| D-20 | Kernel arithmetic is row-independent: `np.einsum` without optimisation or elementwise operations, with precomputed inverse Cholesky factors (R-KER-6) | BLAS-backed products gave a row different bits in a batch of one (9 to 44 of 45 rows; design review D-R4); `einsum` never did. Measured cost at 256 lanes: 18 µs against 4 µs for `@` at $d = 10$, 645 against 104 µs at $d = 100$ | BLAS products, with bitwise invariance weakened to a statistical guarantee |
| D-21 | Explicit `k`, `lag` or `m` are kept; with `k` given, `lag = k` and `m = 10k` without a pilot | one rule for every mix of explicit and `"auto"` values (design review D-R25) | `"auto"` all or nothing |
| D-22 | An interrupted run returns every started replicate, labelled consistent but not exactly unbiased; completing the requested $R$ restores exactness | the started set is a prefix whose length depends on the costs, a ratio estimator (design review D-R3: z = −12.9 with one lane); the census report wanted no estimate unless a valid rule applies | no estimate after an interrupt; GH per-lane estimates |
| D-23 | The meeting-time pilot is capped at $10^5$ iterations; capped pairs sort last, and a censored $\hat q_\beta$ raises | an uncapped pilot can hang `sample()` with no message on a target where the coupling never meets (design review D-R7); a fixed cap keeps each pilot replicate a function of its own stream | a cap from running quantiles (depends on completion order, so on the batch layout); no cap |
| D-24 | HMC: trajectory length $\epsilon\,n \approx 1$ (Heng & Jacob), and a whitened meeting component $\sigma^2\Sigma$ | length $\pi/2$ fails to contract at whitened frequency 2 (design review D-R12); whitening makes the meeting step invariant to coordinate scales | length $\pi/2$ (the first draft); $\sigma^2 I$ as published |
| D-25 | Default IMH proposal: Student-$t_5$ at the pilot mean with the pilot covariance as scale matrix | inflates the covariance by $5/3$ against pilot covariances that understate the tails; bounded weights keep meeting times geometric (design review D-R27) | scale matrix $\Sigma (\nu - 2)/\nu$ (matching the covariance) |
| D-26 | A `Generator` passed as `rng` gives the root seed of four integers drawn from it | the run is then a function of the generator's state, with one code path; spawning from its seed sequence depended on a spawn counter that is not part of the state (M0 simplification review; design review D-R22) | spawn a child of the generator's seed sequence (the M0 code), which leaves the generator's stream untouched |
| D-27 | Pools that telescopic creates ship the target with `cloudpickle` when it is installed (the `parallel` extra) | Python 3.14 made forkserver the Linux default, so a `log_prob` defined in a notebook no longer pickles (design review D-R25) | standard pickling only; forcing the fork start method (unsafe with threads) |

**Open questions (they do not block v0.1):**
1. The licence: MIT by default.
2. When to make the repository public. JOSS asks for about six months of public history.
3. Whether to pursue the unpublished experiments (coupled ensemble pilots, elliptical slice, NUTS) as papers before they enter the package.

**Research questions no source answers** (the census conclusion). `telescopic` will answer them with its own evidence:
1. How many replicates make an interval trustworthy when tails may be heavy?
2. Which warning thresholds separate safe runs from misleading ones? (M9 calibrates them.)
3. How can valid intervals be given for quantiles of a signed measure?

---

## Appendix A: evidence behind the defaults

| ID | Experiment | Result |
|---|---|---|
| E01 (`notes/experiments/e01_lag_recipe.py`) | RWMH, reflection-maximal coupling, $\mathcal N(0, I_d)$ from $\pi_0 = \mathcal N(3, I_d)$. $k$ = the floor of NumPy's linearly interpolated $q_{0.99}(\tau - 1)$ from a 2000-replicate pilot, $m = 10k$, $L \in \{1, k\}$, 4000 replicates | both lags unbiased ($\lvert z\rvert < 1$). $L = k$: inefficiency 36.6 vs 38.8 at $d = 10$ and 8.4 vs 8.6 at $d = 2$, for 1–2% more cost. The bias correction is non-zero in 1–4% of replicates |
| Kickoff prototype (`notes/LOG.md`, 2026-10-23) | Same coupling, stationary start, step $2.38/\sqrt d$ | mean $\tau$ = 3.8, 5.7, 13.4, 32.1, 83.2, 256.9 at $d$ = 1, 2, 5, 10, 20, 50 |
| E-R1 (`mh_couplings.md` §2, §3) | Census reproductions | reflection-maximal $\bar\tau$ = 32.3 at $d = 10$ (O'Leary 2021: 30); independent residuals 274 (279). Reflection grows about as $d^{1.3}$ to $d = 400$ |
| E-R2 (`validation_benchmarks.md` Q1) | A coupling that recomputes $Y$ on the meeting branch | met 0.2% of the time instead of 94%; marginal tests still passed; only a binomial test of the meeting frequency caught it |
| E-R3 (`advanced_couplings.md` §5) | emcee-style stretch and DE ensembles with shared random numbers, plus a maximal-coupling step | no contraction; 0 meetings in 50,000 iterations at $d$ = 1 to 10. Coupled single-chain RWMH met with median τ = 32 at $d = 5$ |
| E-R4 (`parallel_engineering.md`) | Budget policies on coupled RWMH | pooled completion within a deadline biased (z = +165 with 64 lanes, +211 with 1024); "first $R$ completed" z = 221; fixed $R$ and per-lane GH unbiased |
| DR (`notes/reviews/design-review.md`) | Checks of the first draft by an independent reviewer | TV bound at $k$ 0.019 ($d = 10$) and 0.029 ($d = 30$) with $\beta = 0.99$; $k$ from 100-replicate pilots varies ±20%; interrupt bias z = −12.9 with one lane, undetectable with 16 lanes or more; batched `@` changes a row's bits in a batch of one, `einsum` does not |

## Appendix B: key references

- Glynn & Rhee 2014, J. Appl. Probab. 51A:377–389.
- Jacob, O'Leary & Atchadé 2020, JRSSB 82(3):543–600, with discussion.
- Atchadé & Jacob 2024, "Unbiased Markov chain Monte Carlo: what, why, and how", arXiv:2406.06851 (Handbook of MCMC, 2nd ed.).
- Biswas, Jacob & Vanetti 2019, NeurIPS.
- Heng & Jacob 2019, Biometrika 106(2):287–302.
- O'Leary, Wang & Jacob 2021, AISTATS; O'Leary & Wang 2024, AIHP; O'Leary 2021, arXiv:2102.01790.
- Papp & Sherlock 2025, JRSSB 87(3):772–795.
- Biswas, Bhattacharya, Jacob & Johndrow 2022, JRSSB.
- Douc, Jacob, Lee & Vats, arXiv:2206.05691.
- Middleton, Deligiannidis, Doucet & Jacob 2020, EJS 14(2):2842–2891.
- Craiu & Meng 2022, Statistica Sinica 32.
- Glynn & Heidelberger 1990, Oper. Res. 38(5):801–814; 1991, ACM TOMACS 1(1):3–23.
- Bou-Rabee, Eberle & Zimmer 2020, Ann. Appl. Probab. 30(3):1209–1250.
- Corenflos & Särkkä 2022, arXiv:2201.09585.
- Andrieu & Thoms 2008, Stat. Comput. 18:343–373; Haario, Saksman & Tamminen 2001, Bernoulli 7(2):223–242.
- Vehtari et al. 2021, Bayesian Analysis 16(2):667–718.

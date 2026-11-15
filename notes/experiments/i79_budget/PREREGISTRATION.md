# Budget mode: pre-registration (issue #79)

Written and committed before any run of the experiment. Josh's ruling of 2026-11-13 on
#79: "The lane-level estimator under a budget counted in kernel steps is run as an
experiment under the research gate (DECISIONS #69): unbiasedness and coverage on targets
with known answers, at matched cost against runs of fixed R." Nothing in `src/` changes.

Before writing this file, scoping runs of at most 20,000 replicates per configuration
(scripts kept outside the repository) measured only the time per replicate, the mean and
spread of the cost, and the spread and kurtosis of the per-replicate estimates, in order
to choose targets whose costs vary and whose runs fit the processor budget. No budget
estimator, coverage or bias of a budgeted estimate was computed.

## 1. The estimator under test

This is the estimator of #79 (the "Budget mode: reopen DECISIONS #22" item), which is
GH91's $\bar\mu_1$ as JOA20 §3.3 adapts it, with a lane in place of a processor.

- **Lanes.** $G$ lanes. Lane $g$ runs the replicate indices $g, g + G, g + 2G, \dots$ in
  sequence, with the run's root seed, so that replicate $i$ uses the stream
  `(root, MAIN, i)` of the engine unchanged.
- **Clock.** A replicate's cost is its number of kernel applications,
  `Replicates.cost` $= \max(\tau, m) + \tau - L$ (spec R-EST-7): a coupled step counts
  two. It is a function of the replicate alone. Lane $g$ has the budget $B$, a number of
  kernel steps fixed before the run.
- **The rule.** Let $C_{g,1}, C_{g,2}, \dots$ be lane $g$'s costs in order,
  $S_{g,n} = \sum_{j \le n} C_{g,j}$, and $N_g = \max\{n \ge 0 : S_{g,n} \le B\}$, the
  replicates completed within the budget. A lane with $N_g \ge 1$ abandons the replicate
  in progress at the budget and records nothing of it. A lane with $N_g = 0$ runs its
  first replicate to completion, however long it takes, and keeps it:
  $\tilde N_g = \max(1, N_g)$.
- **Lane estimate.** $\tilde H_g = \tilde N_g^{-1} \sum_{j \le \tilde N_g} H_{g,j}$, the
  equal-weight mean of the kept replicates' $H_{k:m}(h)$. As a signed measure, it is the
  average of the kept replicates' measures, so its weights sum to 1.
- **Run estimate.** $\hat\theta = G^{-1} \sum_g \tilde H_g$, with the standard error
  $s_{\tilde H}/\sqrt G$ and the 95% interval $\hat\theta \pm t_{0.975, G-1}\,s/\sqrt G$,
  exactly what `Result.estimate` gives today when each lane is one replicate.
- **Cost charged.** A lane with $N_g \ge 1$ is charged $B$: its completed replicates, the
  abandoned part and any remainder. A lane with $N_g = 0$ is charged $C_{g,1}$. The run's
  cost is the sum over lanes.

**The unbiasedness argument, checked before running.** Within a lane the pairs
$(H_{g,j}, C_{g,j})_j$ are i.i.d.: the indices are distinct, each replicate's output is a
function of its index and the run's fixed arguments only (R-EXEC-2; the engine's
docstring states it bitwise for a log-density evaluated point by point, and the
examples' vectorized log-densities use two-operand einsums that give a row the same bits
in any batch), and $k, m, L$, the kernel and $B$ are fixed before the run. For $n \ge 1$
the event $\{N_g = n\} = \{S_{g,n} \le B < S_{g,n} + C_{g,n+1}\}$ is symmetric in the
first $n$ pairs, and $\bar H_n$ does not involve $H_{g,n+1}$. So
$\mathbb E[H_{g,j}\mathbb 1(N_g = n)]$ is the same for every $j \le n$, and
$\mathbb E[\bar H_n \mathbb 1(N_g = n)] = \mathbb E[H_{g,1}\mathbb 1(N_g = n)]$. Summing
over $n \ge 1$ and adding $\mathbb E[H_{g,1}\mathbb 1(N_g = 0)]$ gives
$\mathbb E[\tilde H_g] = \mathbb E[H_{g,1}] = \pi(h)$ whenever $\mathbb E|H| < \infty$
(GH90 Corollary 7). $N_g \le B$ is finite because every cost is at least
$m \vee 1$. Lanes use disjoint indices, so the $\tilde H_g$ are i.i.d., the run estimate
is unbiased, and the interval is the usual $t$ interval over $G$ i.i.d. values. The same
exchangeability with Jensen's inequality gives $\mathbb E[\tilde H_g^2] \le \mathbb E[H^2]$,
so the lane estimates have a finite variance whenever the replicates do.

The argument needs nothing from the engine beyond R-EXEC-2 and a cost that is a function
of the replicate. The experiment tests the implementation and these premises on real
runs, the coverage of the interval at small $G$, and the price in efficiency.

The issue also says that "meeting-time diagnostics stay per replicate". The argument
above does not cover a pooled average over the kept replicates, which are selected for
short costs; that claim is tested as a secondary quantity (D below).

## 2. The prototype

`budget.py` builds the estimator from `telescopic.engine.run`, unchanged, so every chain
uses the package's kernels and each chain's marginal is exactly that of the uncoupled
kernel. It runs a cell of many independent budget runs together:

- Run $i$ owns the indices $i\,G J + jG + g$ for lane $g$ and round $j < J$, with
  $J = \lfloor B / \max(m, 1) \rfloor + 1$, a bound on the replicates a lane can start.
- Round $j$ calls `engine.run` once on the round-$j$ index of every lane still running in
  every run, and a lane keeps running while $S_{g,j} < B$.
- Each replicate is run to its meeting, including the one a real implementation would
  abandon: only its cost is read, to decide that it was abandoned, and its $H$ is never
  used. The cost charged is the rule's, not the computing time of the emulation.

Since a replicate is a function of its index, this gives the same lane estimates as a
lane that runs its replicates one at a time and stops at the budget. `check_budget.py`
verifies that equivalence with a driver that runs each lane's replicates one by one
through `engine.run`, censoring at `max_iter` the replicate that cannot finish within the
remaining budget. It also verifies the lane rule on hand-made costs (ties at $B$,
$N = 0$, an empty remainder), and its exact unbiasedness on a discrete $(H, C)$ law by
enumeration, with a positive control: the pooled average is biased there.

## 3. Targets, kernels and functions

From `telescopic.examples`, each with an exact answer. Each target's $h$ is one
coordinate $x_j$ and its square, $h(x) = (x_j, x_j^2)$.

| Target | Kernel | Initial | $k$, $m$, $L$ | $h$ | Truth |
|---|---|---|---|---|---|
| `gaussian(5, 0.5)` | `RandomWalk(cov)` | $\mathcal N(3\cdot\mathbf 1, I)$ | 2, 15, 3 | $x_0$ | 0, 1 |
| `banana()` | `HMC(0.02, cov=I)`, with the gradient | $\mathcal N((3, 9), 0.01 I)$ | 10, 100, 10 | $x_1$ | 1.5, 4.8 |
| `mixture()` | `RandomWalk(I, scale=2)` | its default, $\mathcal N(10, 10^2)$ | 50, 200, 1 | $x_0$ | 0, 17 |

The first two are cases of the slow test suite (`tests/test_statistics.py`, with the
banana's ridge start there). In scoping, the costs had coefficients of variation of
about 0.65, 0.6 and 0.5, and the correlation of $H$ with $C$ was largest in magnitude on
the Gaussian (about -0.2). `max_iter` is `None` in every run, so no pair is censored.
Targets are vectorized; runs use 256 engine lanes, unrelated to the $G$ budget lanes.

## 4. Arms and cells

- **Pilot.** 20,000 replicates per target, from their own seed, give $\hat{\mathbb E}[C]$.
  The lane rule applied to the pilot's costs in consecutive blocks gives the mean cost
  charged to a lane, $\hat\ell_b$.
- **Budget arm.** $G = 32$ lanes; $B = \operatorname{round}(b\,\hat{\mathbb E}[C])$ for
  $b \in \{1, 3, 10\}$.
- **Fixed-R arm, at matched cost.** $R_b = \operatorname{round}(G\,\hat\ell_b/\hat{\mathbb E}[C])$
  replicates, so that its expected cost equals the budget arm's.
- **Runs per cell.** $n = 1000$ independent runs per (target, $b$, arm) for the Gaussian
  and the mixture; $n = 500$ for the banana, whose replicates cost about 5 ms. Every
  cell has its own root seed, `SeedSequence(79, spawn_key=(target, arm, b))`.
- **Processor budget.** At most about 2 processor hours on 4 single-threaded processes
  (BLAS and OpenMP threads set to 1); about 1 hour is expected from the scoping times.

## 5. Quantities

For each cell and each component of $h$, over the $n$ runs:

1. **Bias in standard errors**, $\beta = (\overline{\hat\theta} - \pi(h)) / \operatorname{sd}(\hat\theta)$,
   with Monte Carlo error $1/\sqrt n$ (0.032 at $n = 1000$, 0.045 at 500), and its
   $z = \beta\sqrt n$.
2. **Coverage** of the 95% interval, with Monte Carlo error $\sqrt{0.95 \cdot 0.05/n}$
   (0.0069 at $n = 1000$, 0.0097 at 500).
3. **Inefficiency**, $I = \widehat{\operatorname{Var}}(\hat\theta) \times \overline{\text{cost}}$
   per arm, and the ratio $\rho = I_\text{budget}/I_\text{fixed}$ with a 95% bootstrap
   interval over runs (2,000 resamples, seed 79).
4. **Positive control (P):** the pooled average of every replicate completed within the
   budget, $\sum_g \sum_{j \le N_g} H_{g,j} / \sum_g N_g$, from the same lanes, with its
   bias and $z$ as in 1 (a run where no lane completes one is dropped and counted).
5. **Secondary (D):** the mean meeting time $\mathbb E[\tau]$ estimated, (a) by pooling
   the kept replicates of every lane as one sample, as "meeting-time diagnostics stay
   per replicate" would, and (b) by the lane-weighted estimator of section 1 applied to
   $\tau$; each compared with the mean of all the fixed-R arm's meeting times of that
   target, by a $z$ that combines both Monte Carlo errors.
6. **Makespan:** the mean over runs of the largest lane cost divided by $B$, and the
   share of lanes with $N_g = 0$.

## 6. Predictions

- The budget arm is unbiased in every cell; the positive control is biased in at least
  one cell at $b = 1$.
- Coverage of the budget arm matches that of the fixed-R arm within Monte Carlo error.
- $\rho$ above 1 everywhere, falling with $b$: about 1.5 to 2.5 at $b = 1$, 1.15 to 1.4
  at $b = 3$, and 1.03 to 1.15 at $b = 10$. The charged but unused part of a lane is
  about $\mathbb E[C^2]/(2\mathbb E[C]B)$ of its budget, $(1 + \text{cv}^2)/(2b)$.
- D(a) is biased low at $b \le 3$; D(b) is not.

## 7. Decision rules

- **U, unbiasedness.** Over the 18 tests of the budget arm (3 targets, 3 budgets, 2
  components), every $|z| \le 3.5$, a family-wise false-alarm rate under the null of
  about 1%. The fixed-R arm is the control: a cell where it fails the same test is
  invalid, and the verdict is then inconclusive for that cell.
- **C, coverage.** In each of the 18, the budget arm's coverage is at least the fixed-R
  arm's minus $3\sqrt{se_B^2 + se_F^2}$. Coverage below 0.95 in both arms is reported as
  a property of the target at that $R$, not of budget mode.
- **E, efficiency.** At $b = 10$, $\rho \le 1.25$ (point estimate) for both components of
  every target.
- **P, power.** The positive control has $|z| \ge 5$ in at least one cell. Without it the
  report says the experiment did not show it could detect the bias at stake.

**Verdict.**

- U, C and E hold: the evidence recommends building budget mode on part B's plans, as
  #79's option 3.
- U or C fails in a valid cell: do not build it as proposed; the report says where.
- Only E fails: budget mode is unbiased but costs more than 25% more per unit of
  precision than fixed R at 10 replicates' worth of budget per lane; the evidence then
  recommends not building it now, since its only gain is a bounded number of kernel
  steps per lane.
- D is not part of the verdict. If D(a) is biased, a build must compute every
  meeting-time diagnostic lane-weighted, and the report says so.

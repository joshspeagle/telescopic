# Checkpoint: the first draft of telescopic

For Josh, at the end of milestone M11. Figures are measured, with their sources.

## What exists

- **The package** (`src/telescopic/`, 6,131 lines): unbiased MCMC with coupled chains and
  the lag-L time-averaged estimator of Jacob, O'Leary & Atchadé (2020), with signed
  measures, so that any expectation is estimated after the run.
  - Kernels with their couplings: random-walk Metropolis (reflection-maximal and
    maximal), MALA, HMC with a meeting step, independence Metropolis-Hastings with a
    Student-t, Gibbs with user conditionals (maximal, common random numbers, two-scale),
    mixtures.
  - The automatic pipeline of `telescopic.sample`: adaptive Metropolis tuning, a pilot of
    100 lag-1 pairs, and the recipe of Atchadé & Jacob (2024): k at the 99% quantile,
    lag k, m = 10k.
  - Results: estimates with intervals, unbiased covariances, quantiles, signed
    histograms, TV and W1 bounds, a summary, six warnings with measured rates, saving and
    loading, plots, merging.
  - Execution: lockstep lanes with per-replicate random streams, so that a replicate's
    output depends only on the seed and its index; process pools and executors; Ctrl-C
    that returns a partial result, which the next run completes; progress bars.
  - `telescopic.testing.check_coupling` for kernels written by users.
- **Tests:** 554 fast tests in 39 s, with warnings as errors, and 18 slow statistical
  tests in 109 s (below).
- **Documentation:** a theory page, Getting started, a seven-page user guide, six
  tutorials executed at every build, the API reference and a citing page. The strict
  build, with warnings as errors and every cross-reference checked, took 1 min 43 s.
- **Records:** 57 decisions with their reasons and alternatives (`notes/DECISIONS.md`), a
  review of every milestone (`notes/reviews/`), experiments with their outputs
  (`notes/experiments/`), the log, and benchmarks (`benchmarks/results/`).

## How it was built

A research census of the field (`reports/`), then the specification, the test plan and
the implementation plan, then an adversarial review of the design, which found 38
issues, one a blocker. Twelve milestones followed, each a reviewed unit: `/simplify` and
`/code-review` at the effort that its size sets, and an adversarial review for each
departure from the spec. M11 was an independent audit by six auditors who had not written
the code (`notes/reviews/m11.md`).

## What was measured

- **Unbiasedness** (slow suite, DECISIONS #33): nine cases cover every kernel family,
  Gaussians in 1, 5 and 20 dimensions, the banana with HMC and the pump model with Gibbs,
  lags 1, 7 and k, and thinning. Each runs enough replicates that removing the bias
  correction shows at |z| ≥ 5. All pass, and all fail without the correction.
- **Coverage:** nominal 95% intervals for an AR(1) chain cover at 0.9509 and 0.9489
  against a reference of 10⁴ intervals from an implementation apart from the package
  (standard error 0.0022).
- **Published results reproduced:** O'Leary's random walk in 10 dimensions (mean τ − 1
  of 31.2 against 30), the bimodal regimes of Jacob, O'Leary & Atchadé (18.7 and 100
  against 20 and 105; from the wide start 733 and 8,903 against 769 and 9,186), and the
  banana of Heng & Jacob (149 against 158). Simulation-based calibration passes on two
  conjugate models.
- **Warnings:** each threshold is set from measured rates. For example, the variance
  warning fires in 37 of 200 runs on the banana, whose 95% intervals of E[x₀²] covered at
  0.892 against 0.963 in the others, and in all 17 runs of the bimodal failure of Jacob,
  O'Leary & Atchadé whose replicates found the other mode (DECISIONS #36, #54).
- **Performance** against the targets of spec §10 (`benchmarks/results/20b262b.json`):
  1.30 µs per pair-step for lockstep coupled random-walk Metropolis at 256 lanes, against
  a target of 2; per-replicate random streams at 0.275 µs per lane-step and 40.2 µs of
  set-up, against 0.5 and 50; a log-density of 1 ms 3.89 times faster on 4 started
  workers, against 3.5.

## What did not work, or not yet

- **Automatic steps for HMC and MALA.** Tuned toward the usual acceptance rates, their
  coupled chains rarely meet on the banana, and MALA's also fail on a logistic regression
  started wide. `sample` then raises with advice to give a step. Choosing the step from
  pilot meeting times would be a new tuning procedure (question below).
- **Starts far from the target.** From an initial distribution a hundred times wider
  than the posterior, about one replicate in a hundred is averaged while its chain is
  still on its way, and estimates of variances become heavy-tailed. The documentation
  recommends a start a few to ten times wider, and the variance warning catches the
  failure (question below).
- **Heavy tails behind ties.** When nine pairs in ten or more meet at the first time
  possible, the tail index is undefined and its warning silent: above such ties Hill's
  estimate reads light tails as heavy. A rule that looked above the ties failed an
  adversarial review, warning on light tails in 92% to 100% of runs of an independence
  sampler (DECISIONS #57).
- **Performance targets missed:** the overhead of a log-density called point by point at
  one lane, 90.4 µs per pair-step against 40, which needs a scalar path that v0.1 does
  not have; a pool's speed-up with the start of its workers included, 2.28 against 3.5;
  and the automatic pipeline's pilots, 63%, 47% and 43% of the main run at d = 10, 50 and
  100 against 15%. The main run got faster with 256 lanes, and the tuning pilot of 10d²
  steps dominates beyond d = 20.
- **Deferred beyond v0.1:** a budget mode (DECISIONS #22); a bootstrap uncertainty for
  quantiles, and R-hat across the replicates' chains (#38); reprs of results and
  samplers; histograms in memory proportional to the atoms; evaluation counts that
  include tuning and the pilot; derived quantities in `quantile`, `interval` and
  `histogram`; `plot_meeting_times` for pilot arrays.

## Questions for Josh

Each has a default that the work follows until you decide (`notes/QUESTIONS.md`).

1. **Licence.** Default: MIT, as emcee, dynesty and pastry use.
2. **When to make the repository public.** JOSS asks for about six months of public
   history. Default: private until you decide.
3. **Budget mode.** Should-level in the spec; an unbiased version needs per-replicate
   weights throughout `Result`. Default: deferred; an interrupted run and its completion
   cover stopping early.
4. **The size of a user's pool.** Pools that do not report their size, such as dask's
   executors, are sized by the CPUs of this process. Default: keep the guess; the
   alternative is an argument, as dynesty's `queue_size`.
5. **Automatic HMC and MALA steps** (above). Default: keep the published tuning, which
   couples on log-concave targets, and document that other targets need an explicit
   step. A tuning from pilot meeting times would be a new construction (DECISIONS #1).
6. **Starts far from the target** (above). Default: keep the recipe and the advice.
   Remedies are new constructions: a start built from the tuning pilot, or k at a higher
   quantile.
7. **The threshold of the variance warning.** Taken about the atoms' mean, the share fires
   in 37 of 200 banana runs instead of 19. Default: keep a half.
8. **Common random numbers for Gibbs conditionals that are not normal.** They are drawn as
   the normal with their `mean` and `cov`. Default: keep, as documented; the alternative,
   generators in the same state, gives every law exactly but may not contract.

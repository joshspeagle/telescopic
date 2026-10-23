# Log

Append only. One entry per working session or milestone.

## 2026-10-23: kickoff

- Josh's brief: an accessible, pure-Python, emcee/dynesty-style package for unbiased
  MCMC with coupled chains and telescoping-sum estimators; deep research first, then a
  spec-driven, test-driven plan, then an autonomous build to a first draft with tests,
  tutorials and preliminary docs. Every PR-sized change runs `/simplify` and
  `/code-review`.
- Studied pastry (`claude/pastry-v0.1`: research census, design docs with decision
  table, milestones, walkthrough notes), GSPICE (`CLAUDE.md` rules: review gates
  scaled to diff size, adversarial review of departures, test and writing rules) and
  alphauniverse (path-scoped `.claude/rules/`).
- Josh chose: published methods first with new constructions as experiments
  (DECISIONS #1); one branch with reviewed milestones (DECISIONS #2).
- `telescopic` is free on PyPI (checked 2026-10-23).
- Launched the research census: eight topics (theory and estimators; MH-family
  couplings; HMC, Gibbs, slice, ensemble and particle couplings; diagnostics and
  tuning; software; sampler API/UX; parallelism and engineering; validation and
  benchmarks).
- Prototype (`scratchpad`, to be moved into `notes/experiments/` with the spec):
  lag-1 coupled RWMH with reflection-maximal proposals, batched over replicates.
  H_{k:m} for E[x_1] under N(0, I) from a start at mean 3: all |z| < 2 over 4000
  replicates (d = 1, 5; (k, m) = (1, 1), (5, 20), (50, 200)). Mean meeting time
  from a stationary start with step 2.38/sqrt(d): 3.8, 5.7, 13.4, 32.1, 83.2, 256.9
  at d = 1, 2, 5, 10, 20, 50. A lockstep NumPy batch ran about 10^6 pair-steps/s for
  this target; a batch without refilling waits for its slowest pair. Random numbers:
  0.15 us per slot-step from one generator, 1.4 us from per-replicate generators;
  one scalar coupled step costs about 16 us of Python overhead.

## 2026-10-23: design review, M0 review, M3

- The adversarial design review (`notes/reviews/design-review.md`) found 38 issues, one
  a blocker: a kernel `met` flag missed pairs already equal that both reject, and
  meetings at t = L. All were accepted. Meetings are now the engine's exact comparison;
  kernel arithmetic is row-independent (`einsum`), because BLAS gave a row other bits in
  a batch of one; decisions D-20 to D-27 are new (00f9474).
- M0 review (`notes/reviews/m0.md`): buffer sizes entered every seeded number. Each
  replicate's normals and uniforms now come from two child generators read in order,
  so widths are free settings; measured 0.33 µs per lane-step at 256 lanes, d = 10,
  against 0.15 µs for one shared generator (aec1f18). Kernels dropped the `met` flag
  and use `einsum` (5b39221). CI now covers Python 3.15.
- M3 implemented: the engine with refill, exact meetings, thinned weights, censoring and
  a slow-pair threshold of max(10^4, 10 max(L, m)) (the first draft's 10 max(L, m)
  fired on every pilot); `SignedMeasure`; the unbiased covariance; isotonic quantiles.
  `lag_weights` was removed: the engine needs only the thinned coefficients, and the
  tests anchor them to JOA20 equation 2.1. 146 fast tests in 18 s, coverage 96%. Nine
  mutants of the engine and the weights were all caught by the new tests.

## 2026-10-23: M1 to M3 reviews, M4, the M0 to M4 code review, M5

- M1 and M2 review (`notes/reviews/m1-m2.md`, 5529215): `State.take` returned views, so
  `Mixture` wrote into its callers' states; kernels dropped carried gradients, which
  would bias a mixture of MALA and RWMH (z = +20 on E[x^2], measured by the reviewer).
- M3 review (`notes/reviews/m3.md`, 99768aa): engine warnings never reach the parent
  from a pool worker, so the engine records facts and reports to a monitor, and the
  sampler warns. Censored meeting times are `inf`.
- M4 implemented (0e8703b): `Result`, estimates, quantiles, histograms, TV and W1
  bounds, the warnings of spec §7.4 and the summary.
- `/code-review` of M0 to M4 at high effort (`notes/reviews/code-review-m0-m4.md`,
  3c8b0fd): 22 findings. One was high: a kernel of another dimension than the target ran
  silently, broadcast by `einsum`; kernels now report `ndim` and `init_state` refuses a
  mismatch.
- M5 implemented: adaptive Metropolis pilots (4 chains, max(2000, 200d) steps, second
  halves), `Kernel.tune` with a shared pilot (DECISIONS #5), `choose_kml`,
  `CoupledSampler` and `sample`, a sampler's settings fixed by its first run, `max_iter`
  included (DECISIONS #4), and the warning on tuning chains that disagree,
  R-hat > sqrt 2 (DECISIONS #6). The test plan's adaptive Metropolis criterion changed:
  over 20 seeds at d = 5 and condition number 100, the pilot covariance's relative
  Frobenius error had median 0.14 and maximum 0.25, against a bound of 0.20; the
  whitened condition number had median 1.48 and maximum 1.75, and the test bounds it by
  4. Pools and progress (spec §8.1) come with M7.
- M3 efficiency review applied (`notes/reviews/m3.md`, Efficiency): outputs bitwise
  identical on 54 configurations; one lane 156 to 101 µs per pair-step in the main run;
  memory per stored row at one lane 519 to 103 bytes (92 of payload). Spec §10 now
  records the standing: 256 lanes met (1.2 to 1.5 µs), one lane missed (93 µs of
  overhead against 40).
- M4 and M5 `/simplify` (`notes/reviews/m4-m5.md`): the tuning pilot did not converge
  for d ≥ 10 (R-hat about 1.9 on N(0, I₁₀) from a stationary start), so k came out 5 to
  50 times too large, and at d = 20 the pilot pairs never met. The four chains now share
  one covariance at rate 1/(t + 1), started from π0's variances (DECISIONS #7): median k
  at most 1.3 times the exact covariance's for d ≤ 20, 4.4 times at d = 50. Runs
  now warn about censoring and the other §7.4 conditions; `thin` and `max_iter` default
  to the first run's values (DECISIONS #8); `summary()` on 1.27 million atoms takes
  0.44 s instead of 1.59 s.
- M5 `/code-review` at high effort (`notes/reviews/m4-m5.md`): ten findings, all fixed,
  among them a slow pair that warned nowhere (an off-by-one between the live monitor
  and `issues`), `names` checked only after the whole run, and pilot chains that never
  moved silencing the disagreement warning. 226 fast tests in 16 s.

## 2026-10-23: M6

- M6 implemented: `MALA` (reflection-maximal coupling, both proposal densities), `HMC`
  as a mixture of shared-momentum leapfrog trajectories and a whitened random-walk
  meeting step, `Gibbs` with maximal, common-random-number and two-scale couplings, and
  dual averaging (DECISIONS #13). `step` returns acceptance probabilities (DECISIONS
  #10); gradient kernels refuse starts outside the support (#11); MALA has no coupling
  option (#12); `Target(log_prob=None)` serves Gibbs. Each chain's one-step law matches
  the exact one in one dimension for every kernel (MALA's and the linear leapfrog's
  derived in the test); 12 of 13 mutants failed the tests at once, and the 13th (a
  mixture dropping its components' acceptance probabilities) now does too. Dual
  averaging gave stationary acceptance 0.572 to 0.608 for MALA (target 0.574) and 0.802
  to 0.811 for HMC (target 0.8). 287 fast tests in 23 s.
- M6 `/simplify` (`notes/reviews/m6.md`): gradient kernels no longer refuse starts
  outside the support, which failed runs by their seed (8 of 8 MALA runs on a
  half-normal from N((3, 0), I)); such states carry a zero gradient (DECISIONS #14).
  `Kernel.needs_log_prob` replaces the sampler's type check, which refused mixtures of
  Gibbs kernels (#15); `"crn"` refuses `eta` (#16). A `GradientKernel` base holds the
  shared tuning, and dual averaging rescales instead of rebuilding kernels: tuning the
  step size at d = 10 takes 83 ms instead of 174 for MALA and 118 instead of 195 for
  HMC. Engine cost per pair-step at one lane fell from 223 to 153 µs for HMC and from 135
  to 114 µs for MALA, with identical meeting times. 289 fast tests in 22 s.
- M6 `/code-review` at high effort (`notes/reviews/m6.md`): ten findings. Among them, the
  common-random-number coupling gave every coordinate of a multi-coordinate block the
  same uniform when its conditional had `ppf`, a wrong law without an error; HMC's
  `tune` hung on a wrong gradient, its default trajectory growing as 1/ε (now at most
  1,000 steps, DECISIONS #17); HMC's meeting step raised where its trajectories reject
  (#18). `Kernel.check_target` holds the checks of what a kernel needs (#19). A
  reference implementation of Hoffman & Gelman's dual averaging pins ours to 1e-12.
  304 fast tests in 23 s.

## 2026-10-23: M7

- M7 implemented: `pool=` (worker processes started by spawn, executors, `map` pools),
  `progress=`, and Ctrl-C (the first finishes the replicates in progress, the second
  raises and stops the workers). Results record their requested indices, so
  `interrupted` is derived and the sampler's next run completes an interrupted one
  (DECISIONS #20); pools submit four chunks per worker at once (#21); budget mode is
  deferred, with a question for Josh (#22). Replicates are bitwise identical in-process,
  on 1 and 2 spawned workers, on thread and process executors and on a `map` pool. On
  a 2-dimensional Gaussian, a `sample` of 40 replicates took 0.23 s in-process and
  1.25 s with 2 spawned workers, whose start dominates (Intel Xeon @ 2.10 GHz, 4 cores,
  Python 3.13.16, NumPy 2.5.4, one measurement). 323 fast tests in 30 s.
- M7 `/simplify` (`notes/reviews/m7.md`): the first Ctrl-C on a pool now reaches the
  workers through an event, so the result came 0.85 to 1.15 s after it instead of 13.6
  to 15.6 s on 2 workers; workers live for one call, start before tuning and serve the
  pilot and the run (`sample` on 4 workers: 4.1 to 1.8-2.0 s); chunks fill the engine's
  64 lanes; the engine's monitor is the one channel for progress, slow pairs and the
  stop, and slow pilot pairs on a pool now warn (DECISIONS #23). A question for Josh on
  sizing users' pools. 333 fast tests in 36 s.
- M7 `/code-review` at high effort (`notes/reviews/m7.md`): ten findings, nine fixed.
  Spawned workers could die from a Ctrl-C while importing, so they are now made while
  the parent ignores it; a cached BLAS controller missed libraries loaded later; float
  indices were truncated; workers started before the settings were validated; a
  `multiprocessing.Pool` was sized by the CPU count. 342 fast tests in 37 s.

## 2026-10-23: M8

- M8 implemented: `Result.save`/`load` (DECISIONS #26), `telescopic.plotting` (meeting
  times, TV bound, signed marginals with ±2 standard-error boxes, signed corner plots,
  coupled paths), `telescopic.examples` (#24) and `check_coupling` (#25). The pump
  model's quadrature gives E[β] = 2.472, against about 2.47 in the literature, and a
  direct Gibbs run agrees on all 11 means (largest |z| 1.85). Every built-in kernel
  passes `check_coupling`; a wrong Y marginal, a recomputed meeting and independent
  steps are flagged. matplotlib joins the test group. 382 fast tests in 41 s.

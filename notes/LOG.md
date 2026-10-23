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

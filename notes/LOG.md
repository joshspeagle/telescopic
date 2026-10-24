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

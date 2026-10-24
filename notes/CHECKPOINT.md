# Checkpoint: the first draft of telescopic

Draft, to be completed after the M11 audit.

## What exists

- **The package** (`src/telescopic/`, about 5,700 lines): unbiased MCMC with coupled chains
  and the lag-L time-averaged estimator of Jacob, O'Leary & Atchadé (2020), with signed
  measures so that any expectation is estimated after the run.
  - Kernels with their couplings: random-walk Metropolis (reflection-maximal and maximal),
    MALA, HMC with a meeting step, independence Metropolis-Hastings with a Student-t,
    Gibbs with user conditionals (maximal, common random numbers, two-scale), mixtures.
  - The automatic pipeline of `telescopic.sample`: adaptive Metropolis tuning, a pilot of
    100 lag-1 pairs, and the recipe of Atchadé & Jacob (2024), k at the 99% quantile, lag k,
    m = 10k.
  - Results: estimates with intervals, unbiased covariances, quantiles, signed histograms,
    TV and W1 bounds, a summary, six warnings with measured rates, saving and loading,
    plots, merging.
  - Execution: lockstep lanes with per-replicate random streams, so a replicate's output
    depends only on the seed and its index; process pools and executors; Ctrl-C that
    returns a partial result which the next run completes; progress bars.
  - `telescopic.testing.check_coupling` for kernels written by users.
- **Tests**: 423 fast tests in about 45 s (warnings as errors), and 18 slow statistical
  tests in about 140 s: unbiasedness of every kernel family against exact moments,
  interval coverage against an independent reference, TV bounds against exact
  distances, three published meeting-time studies reproduced, and calibration over the
  prior of two conjugate models.
- **Documentation**: a theory page, Getting started, an eight-page user guide, seven
  executed tutorials, the API reference; the strict build runs every notebook in about
  2 minutes 20 seconds.
- **Benchmarks** against the specification's targets (`benchmarks/results/acb12f9.json`).
- **Records**: 38 decisions (`notes/DECISIONS.md`), reviews of every milestone
  (`notes/reviews/`), experiments with their outputs (`notes/experiments/`), the log.

## What was measured

(to fill: unbiasedness, coverage, reproductions, warnings, performance against targets)

## What did not work, or not yet

(to fill: automatic HMC and MALA steps; far starts; pool speed-up and pipeline overhead;
point-by-point overhead; deferred items)

## Questions for Josh

(to fill from `notes/QUESTIONS.md`)

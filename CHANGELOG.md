# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- Per-replicate random streams keyed by the root seed, a namespace and the replicate
  index. Each replicate reads its normals and uniforms in order from two child generators,
  buffered for lockstep batches, so no batch layout or buffer width changes a number.
- Warning classes `TelescopicWarning`, `ConvergenceWarning` and `InputWarning`.
- Coupling primitives: the reflection-maximal coupling of Gaussians, the rejection maximal
  coupling with the bounded-variance `eta` option, and the maximal coupling of discrete
  distributions (`telescopic.couplings`); `telescopic.testing.fixed_draws` to inject
  random numbers in tests, and `telescopic.testing.random_draws` for fresh ones.
- `Target` with a batched interface over scalar or vectorised log-densities and gradients;
  initial distributions `Normal`, `Uniform`, `Empirical` and `as_initial`.
- Kernels with their couplings: `RandomWalk` (reflection-maximal or maximal proposals),
  `Independence` with a `MultivariateT` proposal, and `Mixture`. A row's transition does
  not depend on the other rows of its batch.
- The engine (`telescopic.engine.run`): replicates of coupled chains in lockstep lanes
  with refill, exact meeting detection, the lag-$L$ and thinned estimator weights, signed
  measures, distances between the chains, censoring and a slow-pair warning. A
  replicate's output does not depend on the number of lanes.
- `SignedMeasure`, the unbiased covariance estimator and quantiles of signed measures by
  isotonic projection (`telescopic.measure`, `telescopic.estimators`).
- `Result` with estimates and confidence intervals for any test function, unbiased
  covariances, quantiles, credible intervals and signed histograms; total variation and
  1-Wasserstein bounds from the meeting times; a summary table; and the warnings of the
  specification (`telescopic.results`, `telescopic.diagnostics`).
- Automatic tuning and the sampler: adaptive Metropolis pilots whose chains share their
  adaptation, set the kernels' unset parameters and warn when they disagree; the choice
  of `k`, `m` and the lag from pilot meeting times; `CoupledSampler` and
  `telescopic.sample`, whose runs warn about the conditions of the specification.
- Gradient and Gibbs kernels: `MALA` and `HMC` (with a random-walk meeting step) with
  the reflection-maximal coupling, tuned by dual averaging; `Gibbs` with maximal,
  common-random-number and two-scale couplings of user-written conditionals, and
  `Target(log_prob=None)` for kernels that need no log-density (`Kernel.needs_log_prob`).
  A kernel's `step` returns acceptance probabilities.

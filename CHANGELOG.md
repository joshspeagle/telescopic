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

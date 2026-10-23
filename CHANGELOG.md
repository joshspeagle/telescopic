# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- Per-replicate random streams keyed by the root seed, a namespace and the replicate
  index, buffered for lockstep batches (`telescopic._random`).
- Warning classes `TelescopicWarning`, `ConvergenceWarning` and `InputWarning`.

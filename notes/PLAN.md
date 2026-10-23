# Plan

**Current position:** Phase 4, milestones M0 to M8 implemented and M0 to M7 reviewed;
M8's `/simplify` fixes are in, and its `/code-review` is next.

## Phases

| Phase | Content | Output | Status |
|---|---|---|---|
| 0 | Study Josh's workflows in pastry, GSPICE and alphauniverse | this `notes/` layout; `CLAUDE.md` rules | done |
| 1 | Research census: theory, couplings by sampler family, diagnostics, software, API/UX, parallelism, validation | `research_notes/Unbiased MCMC with couplings landscape/`, `reports/Unbiased MCMC with couplings landscape.md` | done |
| 2 | Design, with prototypes where a choice needs evidence, and an adversarial review | `docs/design/spec.md`, `test-plan.md`, `implementation-plan.md`; `notes/experiments/`; `notes/reviews/design-review.md` | done |
| 3 | Scaffold: packaging, CI, pre-commit, docs skeleton, `CLAUDE.md` | repository infrastructure | done |
| 4 | Implementation milestones M0 to M11 (implementation plan §3), each a reviewed unit | `src/telescopic/`, `tests/` | in progress |
| 5 | Documentation: theory, user guide, executed tutorials, API reference, README | `docs/` | pending |
| 6 | Independent audit of the whole package, fixes, benchmarks | `notes/reviews/` | pending |
| 7 | Checkpoint report to Josh: first draft complete | log entry and summary | pending |

## Milestone status

| M | Content | Implemented | `/simplify` | `/code-review` | Report |
|---|---|---|---|---|---|
| M0 | Scaffold, `_random` | yes | done, fixed | done, fixed (M0 to M4) | `notes/reviews/m0.md`, `code-review-m0-m4.md` |
| M1 | Coupling primitives | yes | done, fixed (with M2) | done, fixed (M0 to M4) | `notes/reviews/m1-m2.md` |
| M2 | Target, initial, kernels | yes | done, fixed (with M1) | done, fixed (M0 to M4) | `notes/reviews/m1-m2.md` |
| M3 | Estimators, measure, engine | yes | done, fixed | done, fixed (M0 to M4) | `notes/reviews/m3.md` |
| M4 | Results, diagnostics, warnings, summary | yes | done, fixed (with M5) | done, fixed (M0 to M4) | `notes/reviews/m4-m5.md`, `code-review-m0-m4.md` |
| M5 | Tuning, sampler, `sample` | yes | done, fixed (with M4) | done, fixed | `notes/reviews/m4-m5.md` |
| M6 | MALA, HMC, Gibbs | yes | done, fixed | done, fixed | `notes/reviews/m6.md` |
| M7 | Parallel execution, interrupts, progress (budgets deferred) | yes | done, fixed | done, fixed | `notes/reviews/m7.md` |
| M8 | IO, plotting, examples, `check_coupling` | yes | done, fixed | pending | `notes/reviews/m8.md` |
| M9 | Slow statistical suite, benchmarks, warning calibration | no | | | |
| M10 | Documentation and tutorials | no | | | |
| M11 | Independent audit, checkpoint report | no | | | |

## Deferred to before M10

From the M8 altitude review (`notes/reviews/m8.md`), to settle before the tutorials use
these interfaces:

- `Replicates` records k, m, the lag, thin, `max_iter` and the root seed, so that
  `Result(replicates)`, `plotting.paths(replicates)` and `merge` need no settings.
- `engine.run` takes `rng` (spec §8.2), with the main namespace by default, so that
  users never import `telescopic._random`.

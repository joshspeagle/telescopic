# Plan

**Current position:** Phase 9. Milestones M0 to M11 are implemented and reviewed, and the
first draft is complete; the checkpoint report to Josh is `notes/CHECKPOINT.md`. The
milestones are on `main` (pull requests #21 to #32), and the second audit's fixes
(`notes/reviews/audit2.md`) land there in phase 6 of the tracking issue (#1, #7). Phase 7
of that issue (#8) is Josh's review on the hub (`notes/review/`). Work waits on his
answers to `notes/QUESTIONS.md`, whose defaults stand meanwhile; the items deferred
beyond v0.1 are listed in `notes/reviews/m11.md`.

## Phases

| Phase | Content | Output | Status |
|---|---|---|---|
| 0 | Study Josh's workflows in pastry, GSPICE and alphauniverse | this `notes/` layout; `CLAUDE.md` rules | done |
| 1 | Research census: theory, couplings by sampler family, diagnostics, software, API/UX, parallelism, validation | `research_notes/Unbiased MCMC with couplings landscape/`, `reports/Unbiased MCMC with couplings landscape.md` | done |
| 2 | Design, with prototypes where a choice needs evidence, and an adversarial review | `docs/design/spec.md`, `test-plan.md`, `implementation-plan.md`; `notes/experiments/`; `notes/reviews/design-review.md` | done |
| 3 | Scaffold: packaging, CI, pre-commit, docs skeleton, `CLAUDE.md` | repository infrastructure | done |
| 4 | Implementation milestones M0 to M11 (implementation plan §3), each a reviewed unit | `src/telescopic/`, `tests/` | done |
| 5 | Documentation: theory, user guide, executed tutorials, API reference, README | `docs/` | done (M10) |
| 6 | Independent audit of the whole package, fixes, benchmarks | `notes/reviews/m11.md` | done (M11) |
| 7 | Checkpoint report to Josh: first draft complete | `notes/CHECKPOINT.md`, log entry | done |
| 8 | Second audit of the whole package, by reviewers who had not written it, and its fixes | `notes/reviews/audit2.md` | done |
| 9 | Landing on `main` through issues and pull requests, then Josh's review (DECISIONS #58) | tracking issue #1; `notes/review/` | in progress: phase 7 of #1 is Josh's review |

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
| M8 | IO, plotting, examples, `check_coupling` | yes | done, fixed | done, fixed | `notes/reviews/m8.md` |
| M9 | Slow statistical suite, benchmarks, warning calibration | yes | done, fixed | done, fixed | `notes/reviews/m9.md` |
| M10 | Documentation and tutorials | yes | done, fixed | done, fixed | `notes/reviews/m10.md` |
| M11 | Independent audit, checkpoint report | yes | done, fixed | done, fixed | `notes/reviews/m11.md` |

## Interface changes deferred from M8

Done before M9, so that its tests and benchmarks use the final interfaces: the
replicates record their run's seed and settings, and `engine.run` takes `rng`
(DECISIONS #31, #32; `notes/reviews/run-settings.md`).

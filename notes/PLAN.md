# Plan

**Current position:** Phase 4, milestones M0 to M4 implemented; the M1 to M3 review gates
are in progress.

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
| M0 | Scaffold, `_random` | yes | done, fixed | with M0 to M3 | `notes/reviews/m0.md` |
| M1 | Coupling primitives | yes | in progress (with M2) | with M0 to M3 | `notes/reviews/m1-m2.md` |
| M2 | Target, initial, kernels | yes | in progress (with M1) | with M0 to M3 | `notes/reviews/m1-m2.md` |
| M3 | Estimators, measure, engine | yes | pending | with M0 to M3 | `notes/reviews/m3.md` |
| M4 | Results, diagnostics, warnings, summary | yes | pending | | `notes/reviews/m4.md` |
| M5 | Tuning, sampler, `sample` | no | | | |
| M6 | MALA, HMC, Gibbs | no | | | |
| M7 | Parallel execution, interrupts, budgets | no | | | |
| M8 | IO, plotting, examples, `check_coupling` | no | | | |
| M9 | Slow statistical suite, benchmarks, warning calibration | no | | | |
| M10 | Documentation and tutorials | no | | | |
| M11 | Independent audit, checkpoint report | no | | | |

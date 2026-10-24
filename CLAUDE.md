# CLAUDE.md

telescopic runs Markov chain Monte Carlo as pairs of coupled chains and returns unbiased
estimates with finite-sample error bars, using the telescoping-sum estimators of Glynn &
Rhee (2014) and Jacob, O'Leary & Atchadé (2020).

## Start here

- Read `notes/PLAN.md` (current position), the last entries of `notes/LOG.md` and the
  open items of `notes/QUESTIONS.md`.
- The requirement is `docs/design/spec.md`. Tests follow `docs/design/test-plan.md`, and
  the order of work follows `docs/design/implementation-plan.md`.

## Layout

| Path | What it is |
|---|---|
| `src/telescopic/` | the package |
| `tests/` | the pytest suite; `slow` marks statistical tests |
| `docs/` | the Sphinx site in MyST Markdown; `docs/tutorials/` are executed notebooks; `docs/design/` holds the spec and the plans |
| `benchmarks/` | timing and efficiency scripts, run by hand |
| `notes/` | plan, log, decisions, questions, experiments and review reports; not part of the site |
| `research_notes/`, `reports/` | the landscape census behind the design |

## Commands

- Set up: `uv venv --python 3.13 .venv && uv pip install -e . --group dev`.
- Fast tests: `.venv/bin/python -m pytest`. It skips `slow` tests and must finish in
  under 2 minutes.
- All tests: `.venv/bin/python -m pytest -m ""`.
- Lint, format and types: `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy`.
- Docs: `uv pip install -e . --group docs`, then
  `.venv/bin/sphinx-build -W --keep-going docs docs/_build/html`.

## Plan

- The lead is Josh Speagle (`joshspeagle`). Priorities: correctness, then API quality,
  then performance. Never trade the first for the others.
- v0.1 implements published methods. A new construction stays an experiment in
  `notes/experiments/` until Josh has seen its evidence (DECISIONS #1).
- A decision is a choice that another competent developer could make differently and
  that affects behaviour, the public API, a number, a tolerance, a dependency or the
  process. Record it with its reason and alternatives: in the spec's decision table
  while the spec is being written, in `notes/DECISIONS.md` afterwards. Never choose
  silently.
- A change that departs from the spec edits the spec in the same commit and records the
  departure as a decision.
- A question only Josh can answer goes to `notes/QUESTIONS.md` with a recommended
  default, and work continues on the default.

## Milestones

Each milestone of the implementation plan is one reviewed unit, landed on `main` as a
pull request (DECISIONS #2, #58).

1. Implement it with its tests, docstrings and CHANGELOG line, in commits that each make
   one logical change.
2. Run the commands until they pass.
3. Adversarial review, only for a significant departure from the spec. One independent
   subagent tries to refute it; it gets the diff, the decision and the spec, not the
   session's reasoning. A refuted decision is changed, or the objection is answered in
   the record. A departure that changes a statistical result gets three reviewers and
   stands only if two uphold it.
4. Run `/simplify`, then `/code-review`, on the milestone's diff, at the effort the table
   sets. A change to numerics or the public API takes the next size. Fix each valid
   finding, and answer the rest in the milestone's report in `notes/reviews/`.

   | Lines added plus removed | `/simplify` | `/code-review` |
   |---|---|---|
   | under 100 | one reviewer for all angles | `low` |
   | 100 to 500 | two reviewers | `medium` |
   | over 500 | four reviewers, one per angle | `high` |

5. Run the commands and the `slow` tests of the changed modules.
6. Commit the fixes, push, add a log entry and update `notes/PLAN.md`.

## Git

- Work on `claude/coupled-mcmc-package-btpupz`. It reaches `main` through pull requests,
  one per milestone or group of issues, each closing its issue, each branch a commit of
  the development branch whose CI passed, merged with a merge commit; a tracking issue
  lists them by phase (DECISIONS #58).
- A commit passes every test that passed before it. Subject in the imperative, at most
  72 characters; a body only for why.
- Push only after the commands pass. Never force-push, and never rebase or amend a pushed
  commit.

## Code

- Write the minimum code that is correct, clear and fast. Prefer removing to adding;
  deleting a concept beats adding a flag.
- Write no code when NumPy, SciPy or the codebase already does it.
- Validate public inputs early, with errors that name the argument and say what was
  expected. Internal functions trust their callers. Leanness never removes input
  validation or numerical safety checks.
- Public functions never modify their input arrays.
- No no-ops (passing a default, guarding a case that cannot happen), dead code or
  linter-ignore rules.
- One purpose per function, one group of things per module. Add a helper only when it is
  reused or a function is too long.
- Type hints on every public signature. Names are readable words, except symbols that
  follow the spec's notation.

## Numerics

- float64 throughout. Work with log densities, and accept or reject in log space.
- Randomness comes only from `numpy.random.Generator` objects derived from the caller's
  `rng` (SPEC 7).
- Every coupled kernel leaves each chain's marginal transition exactly that of the
  uncoupled kernel, and keeps chains equal once they meet.
- A NaN log density raises with the offending point. `-inf` is a rejection.

## Tests

- Hermetic: no network, no files outside the repository.
- Write a test only when it catches a plausible bug in this code that no other test
  catches. Test contracts, invariants, agreement with exact references, and regressions:
  each bug fix adds a test that fails without it. Do not test NumPy's or SciPy's
  behaviour.
- Every coupling is tested for its marginals, against the uncoupled kernel from fixed
  states, and for faithfulness. Every estimator is tested for unbiasedness on a target
  with known expectations.
- Statistical tests are seeded, with tolerances from the Monte Carlo error, never set by
  trying seeds until a test passes.
- Mark a test `slow` when it needs many replicates for its tolerance or takes more than
  5 s.
- Name a test as a sentence that states the behaviour: `test_met_chains_stay_equal`.

## Documentation

- A public module has a one-line docstring. A public class, function or method has a
  NumPy-style docstring: a one-line summary, the equation it implements if any, in the
  notation of `docs/theory.md`, parameters and returns with shapes, and the cost when it
  is not obvious. Private code has no docstring; a comment explains only a non-obvious
  why, such as a numerical subtlety or the paper equation it implements.
- Docs state only what the code does now, except `docs/design/`. Each fact has one home.
  Why a change was made goes in its commit and in `notes/`.
- Tutorials are MyST notebooks executed when the docs build; each runs in under
  2 minutes.

## Measurements

- State only measured figures, or figures computed from them with the computation shown.
- A benchmark records the commit, machine, thread settings and library versions. Figures
  from different setups are not compared.

## Agents

- Delegate through self-contained briefs. Subagents never commit. Read every line a
  subagent writes before committing it.
- Parallel subagents that write files get separate files or separate worktrees.

## Writing

- Spartan: the fewest words that stay exact. Literal over metaphor. No em dashes.
- Session replies lead with the answer, then the reason, then the next step.

# CLAUDE.md

telescopic runs Markov chain Monte Carlo as pairs of coupled chains and returns unbiased
estimates with finite-sample error bars, using the telescoping-sum estimators of Glynn &
Rhee (2014) and Jacob, O'Leary & Atchadé (2020).

## Start here

- Read the tracking issue of the current phase on GitHub, now #65 (the tune-up; this
  line is updated when a new phase opens its issue): the State table in its body is the
  current position, and its last log comments say what the latest sessions did. Then
  read the open items of `notes/QUESTIONS.md`. `notes/PLAN.md` and `notes/LOG.md` are
  the closed record of the first draft (#1).
- The requirement is `docs/design/spec.md`. Tests follow `docs/design/test-plan.md`, and
  the order of work follows `docs/design/implementation-plan.md`.
- A change to the package or to the documentation follows `.claude/rules/docs-upkeep.md`.
  Work shared between several agents follows `.agents/briefs/README.md`.

## Layout

| Path | What it is |
|---|---|
| `src/telescopic/` | the package |
| `tests/` | the pytest suite; `slow` marks statistical tests |
| `docs/` | the Sphinx site in MyST Markdown, with its own theme (`docs/_themes/`, `docs/_ext/`); `docs/tutorials/` are executed notebooks; `docs/_scripts/brand.py` writes the identity; `docs/design/` holds the spec, the plans and the design of the identity, read in the repository and not built |
| `benchmarks/` | timing and efficiency scripts, run by hand |
| `scripts/` | `verify.sh`, the gate that runs what CI runs |
| `notes/` | decisions, questions and experiments; the closed plan, log, review reports and checkpoint of the first draft; `review/`, Josh's review of it; not part of the site |
| `research_notes/`, `reports/` | the landscape census behind the design |
| `.claude/rules/` | rules scoped to paths, mirrored in `.agents/rules/` for other agent platforms |
| `.agents/briefs/` | the briefs of each agent role and the formats of their outputs |

## Commands

- The gate: `scripts/verify.sh` runs the lint, the format check, the spelling check, the
  type check and the fast tests on the checkout's sources, as CI's lint and test jobs
  do on the installed package. `--cov` adds CI's coverage floor, `--docs` the strict
  documentation build and `--slow` the slow tests. It needs the `dev` dependency group
  (`docs` too for `--docs`): `pip install -e . --group dev --group docs`, with pip 25.1
  or newer. Set
  `PYTHON` to the interpreter that has them when that is not the first `python`; where
  that environment lives is each machine's own setting.
- One test file: `PYTHONPATH=src pytest tests/test_engine.py`. The fast tests skip those
  marked `slow` and must finish in under 2 minutes.

## Plan

- The lead is Josh Speagle (`joshspeagle`). Priorities: correctness, then API quality,
  then performance. Never trade the first for the others.
- The package starts from published methods and goes past them. A new construction is
  developed as an experiment in `notes/experiments/`, with its evidence, and enters the
  public API only after Josh has seen that evidence (DECISIONS #1, #69).
- A decision is a choice that another competent developer could make differently and
  that affects behavior, the public API, a number, a tolerance, a dependency or the
  process. Record it with its reason and alternatives: in the spec's decision table
  while the spec is being written, in `notes/DECISIONS.md` afterwards. Never choose
  silently.
- A change that departs from the spec edits the spec in the same commit and records the
  departure as a decision.
- A question only Josh can answer goes to `notes/QUESTIONS.md` with a recommended
  default, and work continues on the default.

## Changes

Each change is one reviewed unit: an issue, a branch and a pull request (DECISIONS #58,
#68). A milestone of the implementation plan is such a unit. A change's record is its
issue and its pull request, whose description holds its checks, decisions and reviews.

1. Open or choose the issue. It states the scope, the reason and the acceptance
   criteria. Branch `<issue>-<slug>` from `main`, and open a draft pull request that
   closes the issue after the first commit.
2. Implement it with its tests, docstrings and CHANGELOG line, in commits that each make
   one logical change. Record each decision in the pull request and in
   `notes/DECISIONS.md`.
3. Run the gate until it passes, then the `slow` tests of the changed modules. Fix a
   failing test at its cause: never by loosening a tolerance, skipping the test or
   changing its seed.
4. Adversarial review, only for a significant departure from the spec. One independent
   agent tries to refute it; it gets the diff, the decision and the spec, not the
   author's reasoning. A refuted decision is changed, or the objection is answered in
   the pull request. A departure that changes a statistical result gets three reviewers
   and stands only if two uphold it.
5. A simplification pass, then a code review, on the pull request's diff, by agents that
   did not write it, at the size the table sets. A change to numerics or the public API
   takes the next size. The simplification pass looks for reuse, simpler forms,
   efficiency and the right level of abstraction; the code review looks through the
   correctness and compliance lenses of `.agents/briefs/review.md`. On Claude Code
   these are `/simplify` and `/code-review`, which take the level named in the table;
   without subagents, make each a separate pass that starts from the diff alone.
   Fix each valid finding, answer the rest in the pull request, and run step 3 again
   after the fixes.

   | Lines added plus removed | Simplification | Code review |
   |---|---|---|
   | under 100 | one reviewer for all four | one reviewer (`low`) |
   | 100 to 500 | two reviewers | three reviewers (`medium`) |
   | over 500 | four reviewers, one for each | five reviewers (`high`) |

6. Update the documentation that the change touches (`.claude/rules/docs-upkeep.md`).
7. Push, and watch CI on the pull request without waiting on it; fix a failing job at
   its cause. Mark the pull request ready when the gate, CI and the reviews of step 5
   pass, and then merge it, with a merge commit, and check CI on `main` afterwards. A
   change that fundamentally alters functionality or the public API is not merged until
   Josh has reviewed it: say so in its pull request and ask him (DECISIONS #97).
8. Add a dated log comment for the session to the phase's tracking issue: what was done,
   found and ruled, with issue, pull request and commit numbers, and update its State
   table when an epic moves. Without GitHub access, end the reply with that text.

A fix of at most 10 lines to prose, a comment or configuration that changes no behavior
needs no issue.

## Git

- A commit passes every test that passed before it. Subject in the imperative, at most
  72 characters; a body only for why.
- Keep git's own author and committer. A commit that an agent helped to write ends with
  a `Co-Authored-By:` trailer that names the agent.
- Push only after the commands pass. Never force-push, and never rebase or amend a pushed
  commit.
- Every tracked file, issue, pull request and comment is written for a public reader: no
  machine-specific paths, hostnames, local URLs or names of private tools.
- Text that an agent posts to GitHub ends with a horizontal rule and a line that names
  the agent, `_Generated by [<agent>](<the agent's public page>)_`, so that it is never
  read as Josh's own words or approval.

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
  behavior.
- Every coupling is tested for its marginals, against the uncoupled kernel from fixed
  states, and for faithfulness. Every estimator is tested for unbiasedness on a target
  with known expectations.
- Statistical tests are seeded, with tolerances from the Monte Carlo error, never set by
  trying seeds until a test passes.
- Mark a test `slow` when it needs many replicates for its tolerance or takes more than
  5 s.
- Name a test as a sentence that states the behavior: `test_met_chains_stay_equal`.

## Documentation

- A public module has a one-line docstring. A public class, function or method has a
  NumPy-style docstring: a one-line summary, the equation it implements if any, in the
  notation of `docs/theory.md`, parameters and returns with shapes, and the cost when it
  is not obvious. Private code has no docstring; a comment explains only a non-obvious
  why, such as a numerical subtlety or the paper equation it implements.
- Docs state only what the code does now, except `docs/design/`. Each fact has one home.
  Why a change was made goes in its commit and its pull request, and a decision in
  `notes/DECISIONS.md`.
- Tutorials are MyST notebooks executed when the docs build (`CONTRIBUTING.md`,
  Documentation); each runs in under 2 minutes.
- Colors, fonts, sizes and the mark come from `docs/_scripts/brand.py`, and figures follow
  the drawing rules of `docs/design/brand.md`: the two chain colors mean the two chains.
  A new page joins a captioned toctree of `docs/index.md`, which makes its section, or the
  toctree of a page listed there, as the API reference lists its generated pages.

## Measurements

- State only measured figures, or figures computed from them with the computation shown.
- A benchmark records the commit, machine, thread settings and library versions. Figures
  from different setups are not compared.

## Agents

A lead agent may brief other agents for one task each. `.agents/briefs/README.md` says
how: the roles, the brief and output format of each, where a task's working files go,
and the path of a review and of a research direction.

- Brief an agent with the goal, the files, the rules and the output format, and nothing
  of the lead agent's own conclusions.
- Briefed agents never commit or push, and post to GitHub only a text the lead agent has
  read, verbatim. Read every line an agent writes before committing it.
- A finding becomes an issue only after an agent that did not produce it has tried to
  refute or reproduce it from the code alone.
- Agents that write tracked files get separate files, or separate worktrees under
  `.claude/worktrees/`.
- The default model tier does reviewing, research and implementation, and the fast tier
  surveys and checks that a search settles. The strongest tier is for derivations,
  adversarial review of statistical claims and synthesis across findings; name each
  such use in the task's tracking issue (DECISIONS #70).
- The record is GitHub and `notes/`. On GitHub: a tracking issue for each phase, whose
  body holds the current position, what is next and the rulings by reference (date and
  issue numbers), with a log comment for each session that names the agents run and
  what each returned in a line; its epics and sub-issues; each ruling as a comment on
  the issue ruled on; a review's report as a comment on its epic, or on its issue when
  it has none; and each change's pull request. In `notes/`: `DECISIONS.md`,
  `QUESTIONS.md` and `experiments/`. What lives only in a session is lost.

## Writing

- Pair concrete statistical intuition with exact technical terminology, write complete
  sentences before code blocks and equations, and use American English spelling in
  user-facing documentation and docstrings. Literal over metaphor. No em dashes.
- Session replies lead with the answer, then the reason, then the next step.

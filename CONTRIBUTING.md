# Contributing to telescopic

telescopic aims to be small, correct and pleasant to use. A change must keep every
estimator exactly unbiased (or clearly label an approximation), come with tests, and read
like the code around it.

## Set-up

```bash
git clone https://github.com/joshspeagle/telescopic
cd telescopic
# In an environment of your own (venv, uv, conda), with pip 25.1 or newer:
python -m pip install -e . --group dev --group docs
pre-commit install      # ruff lint and format, and codespell, on every commit
```

## Commands

```bash
scripts/verify.sh              # lint, format, spelling and type checks (Python 3.12+), fast tests
scripts/verify.sh --cov        # with the coverage floor that CI enforces
scripts/verify.sh --docs       # also the identity checks and the strict documentation build
scripts/verify.sh --slow       # also the slow statistical tests
python benchmarks/run.py       # timings against spec section 10, on an idle machine
```

The script runs the commands of CI's lint and test jobs on the checkout's sources. The
fast tests finish in under 2 minutes and treat warnings as errors.

## Documentation

The site is built with Sphinx from `docs/`, in its own theme. Every build executes the
tutorials, and a failing cell fails it:

```bash
scripts/verify.sh --docs                              # the checks, then the strict build
sphinx-build -W --keep-going docs docs/_build/html    # the strict build alone
```

The gate builds from the checkout's sources. A bare `sphinx-build` imports the installed
package: in a second checkout or a worktree, prefix it with `PYTHONPATH="$PWD/src"`.

The environment variable `TELESCOPIC_DOCS_EXECUTE` sets how the build runs the notebooks.
`force`, the default and what CI uses, runs every notebook. `cache` reuses the outputs
of a notebook whose cells have not changed since it last ran, so after a change to the
package or to the plot style it can show old outputs. `off` runs nothing: pages show
their code without outputs, and numbers glued from cells are missing, so build without
`-W`. Use `cache` or `off` only for work on prose or styling, and check the result with a
full build:

```bash
TELESCOPIC_DOCS_EXECUTE=cache sphinx-build docs docs/_build/html
```

The colors, fonts, sizes, marks and plot style come from one script,
`docs/_scripts/brand.py`, whose docstring lists the files it writes. Edit the values
there and run it; CI runs it with `--check`, which names any file that is stale:

```bash
python docs/_scripts/fetch_fonts.py      # after a change to brand.FONTS: the woff2 files
python docs/_scripts/hero_pair.py        # after a change to the package moves the hero's pair
python docs/_scripts/brand.py            # writes the identity's files, the hero among them
python docs/_scripts/brand.py --check    # fails if a written file is stale
```

The hero is one coupled pair run by telescopic; `hero_pair.py --check`, which CI also
runs, fails when the package no longer produces the saved pair and says what moved. Save
the new pair with `hero_pair.py`, then run `brand.py`, which draws the hero from it.

The design records are read in the repository and are not built into the site:
[`docs/design/`](https://github.com/joshspeagle/telescopic/tree/main/docs/design) holds
the specification, the test plan, the implementation plan and the design of the site's
identity, and
[`research_notes/`](https://github.com/joshspeagle/telescopic/tree/main/research_notes)
and [`reports/`](https://github.com/joshspeagle/telescopic/tree/main/reports) hold the
census of published methods and software behind the design.

## Validation philosophy

- **Exact references first.** A coupling is tested draw for draw against an independent
  reference with injected random numbers, for its
  marginals against the exact one-step law, for faithfulness, and for its meeting
  frequency against closed forms. An estimator is tested by algebraic identities and
  exact finite-state expectations before any statistical test. Index bugs in the
  bias correction shift estimates by a fraction of a percent of their spread, far below
  what replicate averages can detect.
- **Tolerances come from the Monte Carlo error**, by the one rule of
  `docs/design/test-plan.md` section 1, which `references.assert_z_scores` applies.
  Seeds are never tuned to make a test pass.
- **A test earns its place** by catching a plausible bug that no other test catches.

## Style

- `ruff check` and `ruff format` (line length 99) and `mypy --strict` on `src/`.
- Use the notation of `docs/theory.md` in code, docstrings and docs.
- Randomness flows through a `rng` argument (SPEC 7) and, inside the engine, through each
  replicate's own stream. Never use NumPy's global random state.
- User-facing problems raise `ValueError` or `TypeError` with a message that names the
  argument and says how to fix it. Questionable inputs emit `InputWarning`; results that
  may be unreliable emit `ConvergenceWarning`.

## Adding a kernel

A kernel follows the rules of the guide's
[Writing a kernel](docs/guide/custom-kernels.md#the-rules) page and passes
`telescopic.testing.check_coupling`. Add the oracle, marginal and faithfulness tests of
`docs/design/test-plan.md` section 2.4.

## Pull requests

Keep each change focused, add a line to `CHANGELOG.md` under *Unreleased*, and describe how
you validated any statistical change.

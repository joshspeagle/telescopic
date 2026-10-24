# Contributing to telescopic

telescopic aims to be small, correct and pleasant to use. A change must keep every
estimator exactly unbiased (or clearly label an approximation), come with tests, and read
like the code around it.

## Set-up

```bash
git clone https://github.com/joshspeagle/telescopic
cd telescopic
uv venv --python 3.13 .venv
uv pip install -e . --group dev
.venv/bin/pre-commit install      # ruff lint and format on every commit
```

## Commands

```bash
.venv/bin/python -m pytest                 # fast tier, under 2 minutes; warnings are errors
.venv/bin/python -m pytest -m ""           # all tests, including the slow statistical tier
.venv/bin/python -m pytest --cov=telescopic
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy
uv pip install -e . --group docs && .venv/bin/sphinx-build -W --keep-going docs docs/_build/html
.venv/bin/python benchmarks/run.py         # timings against spec section 10, on an idle machine
```

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

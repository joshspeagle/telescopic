---
paths:
  - "src/telescopic/**/*.py"
  - "docs/**/*.md"
  - "docs/conf.py"
  - "README.md"
  - "pyproject.toml"
---

# Documentation upkeep

A pull request that adds, renames or changes a public function, class, argument,
default, return shape, warning, error or number updates the pages that describe it in
the same pull request. The documentation states only what the code does now.

## Which pages a module maps to

| Module of `src/telescopic/` | Guide and getting started | Tutorials | Theory and reference |
|---|---|---|---|
| `__init__.py` (the public namespace) | `getting-started.md` | | `api.md` |
| `sampler.py`, `tuning.py` | `getting-started.md`, `guide/settings.md` | `tutorials/quickstart.md`, `tutorials/choosing-settings.md` | `theory.md`, `api.md` (Running, Building blocks) |
| `kernels/`, `couplings.py` | `guide/kernels.md`, `guide/custom-kernels.md` | `tutorials/gradients.md`, `tutorials/gibbs.md`, `tutorials/how-it-works.md` | `theory.md`, `api.md` (Kernels and couplings) |
| `target.py`, `initial.py` | `getting-started.md`, `guide/targets.md` | `tutorials/quickstart.md` | `api.md` (Targets and initial distributions) |
| `engine.py`, `_parallel.py`, `_random.py` | `guide/parallel.md`, `guide/settings.md` | `tutorials/how-it-works.md` | `api.md` (Building blocks) |
| `_progress.py` | `getting-started.md`, `guide/parallel.md` | | |
| `estimators.py`, `measure.py`, `results.py` | `guide/results.md` | `tutorials/how-it-works.md`, `tutorials/quickstart.md` | `theory.md`, `api.md` (Results, Building blocks) |
| `_io.py` (saving and loading) | `guide/results.md` | | `api.md` (Results) |
| `diagnostics.py`, `warnings.py` | `guide/diagnostics.md` | `tutorials/failure-modes.md`, `tutorials/choosing-settings.md` | `theory.md`, `api.md` (Diagnostics, plots and checks; Warnings) |
| `plotting.py` | `guide/results.md`, `guide/diagnostics.md` | `tutorials/how-it-works.md` | `api.md` (Diagnostics, plots and checks) |
| `examples.py` | `guide/kernels.md` | every tutorial that uses the target | `api.md` (Examples) |
| `testing.py` | `guide/custom-kernels.md` | | `api.md` (Diagnostics, plots and checks) |
| `_types.py`, `_linalg.py` (private helpers) | the pages of the public functions whose arguments they check or whose errors they word | | |
| `pyproject.toml` (dependencies, extras, Python versions) | `README.md`, `getting-started.md`, `CONTRIBUTING.md` | | |

Paths in the table are relative to `docs/`, except `README.md`, `CONTRIBUTING.md` and
`pyproject.toml`. When a page moves or a module is added, change this table in the same
pull request. The site's sections are the captioned toctrees of `docs/index.md`: a new
page is listed in one of them.

## Before merging

1. **Pages.** Update the mapped pages, and the README when its example or its list of
   guarantees is affected.
2. **API reference.** A new public name is added to `docs/api.md` under its section, and
   to `__all__` in `src/telescopic/__init__.py` or `src/telescopic/kernels/__init__.py`
   when it is exported there. The strict build fails on a reference that
   resolves to nothing.
3. **The specification.** A change that departs from `docs/design/spec.md` edits it in
   the same commit and records the decision in `notes/DECISIONS.md`.
4. **Numbers.** A figure quoted in the documentation comes from a script in
   `notes/experiments/` or `benchmarks/`, or from a tutorial's own executed cell. Change
   the figure when its source changes.
5. **CHANGELOG.** A change that a user can see has a line under *Unreleased*.
6. **Register.** Pages use the notation of `docs/theory.md`, American English spelling,
   no em dashes and no machine-specific paths. Outside `docs/design/`, they do not cite
   `notes/`, decision numbers or the specification's requirement numbers, which are
   internal.
7. **Identity.** A change to a color, a font, a size or the mark is made in
   `docs/_scripts/brand.py`, which is then run; `brand.py --check` passes. A change to
   the engine, the random streams or the random-walk kernel that changes the landing
   page's coupled pair fails `docs/_scripts/hero_pair.py --check`: run the script and
   commit the new `hero_pair.json`, then `brand.py`.
8. **Build.** `scripts/verify.sh --docs` passes: the build runs the tutorials, treats
   warnings as errors and checks every cross-reference.

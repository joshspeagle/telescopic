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

| Module of `src/telescopic/` | Start and how-to guides | Tutorials and gallery | Concepts, theory and reference |
|---|---|---|---|
| `__init__.py` (the public namespace) | `start/quickstart.md` | | `api.md` |
| `sampler.py`, `tuning.py` | `start/is-it-right.md`, `start/quickstart.md`, `start/coming-from-emcee.md`, `guide/settings.md`, `guide/step-size.md`, `guide/troubleshooting.md`, `guide/faster.md` | `tutorials/choosing-settings.md`, `gallery/funnel.md` | `concepts/error-bars.md`, `theory.md`, `reference/glossary.md`, `reference/notation.md`, `reference/faq.md`, `api.md` (Running, Building blocks) |
| `kernels/`, `couplings.py` | `start/is-it-right.md`, `guide/kernels.md`, `guide/step-size.md`, `guide/custom-kernels.md`, `guide/troubleshooting.md`, `guide/faster.md` | `tutorials/gradients.md`, `tutorials/gibbs.md`, `tutorials/how-it-works.md`, `gallery/funnel.md` | `concepts/two-chains.md`, `theory.md`, `reference/glossary.md`, `reference/notation.md`, `reference/faq.md`, `api.md` (Kernels and couplings) |
| `target.py`, `initial.py` | `start/quickstart.md`, `start/coming-from-emcee.md`, `guide/log-density.md`, `guide/initial-distribution.md`, `guide/faster.md` | | `reference/faq.md`, `api.md` (Targets and initial distributions) |
| `engine.py`, `_parallel.py`, `_random.py` | `guide/parallel.md`, `guide/settings.md`, `guide/faster.md` | `tutorials/how-it-works.md`, `tutorials/unbiased-on-average.md` | `concepts/two-chains.md`, `reference/notation.md`, `reference/faq.md`, `api.md` (Building blocks) |
| `_progress.py` | `start/install.md`, `guide/parallel.md` | | |
| `estimators.py`, `measure.py`, `results.py` | `start/what-it-does.md`, `start/is-it-right.md`, `start/coming-from-emcee.md`, `guide/estimates.md`, `guide/plots.md`, `guide/save-load-merge.md`, `guide/meeting-times.md`, `guide/troubleshooting.md` | `tutorials/how-it-works.md`, `tutorials/unbiased-on-average.md`, `gallery/funnel.md` | `concepts/burn-in-bias.md`, `concepts/telescoping-correction.md`, `concepts/signed-measures.md`, `concepts/error-bars.md`, `concepts/meeting-times.md`, `theory.md`, `reference/result-fields.md`, `reference/warnings.md`, `reference/glossary.md`, `reference/notation.md`, `reference/faq.md`, `api.md` (Results, Building blocks) |
| `_io.py` (saving and loading) | `guide/save-load-merge.md` | | `api.md` (Results) |
| `diagnostics.py`, `warnings.py` | `guide/meeting-times.md`, `guide/troubleshooting.md` | `tutorials/failure-modes.md`, `tutorials/choosing-settings.md`, `gallery/funnel.md` | `concepts/error-bars.md`, `concepts/meeting-times.md`, `theory.md`, `reference/result-fields.md`, `reference/warnings.md`, `reference/glossary.md`, `reference/notation.md`, `reference/faq.md`, `api.md` (Diagnostics, plots and checks; Warnings) |
| `plotting.py` | `guide/plots.md`, `guide/meeting-times.md` | `tutorials/how-it-works.md` | `concepts/meeting-times.md`, `api.md` (Diagnostics, plots and checks) |
| `examples.py` | `guide/kernels.md`, `guide/step-size.md`, `guide/troubleshooting.md`, `guide/faster.md` | every tutorial and gallery page that uses the target | `api.md` (Examples) |
| `testing.py` | `guide/custom-kernels.md` | | `api.md` (Diagnostics, plots and checks) |
| `_gradient_check.py` | `guide/log-density.md`, `guide/settings.md` | `tutorials/gradients.md` | `api.md` (Running; Diagnostics, plots and checks) |
| `_types.py`, `_linalg.py` (private helpers) | the pages of the public functions whose arguments they check or whose errors they word | | |
| `pyproject.toml` (dependencies, extras, Python versions) | `README.md`, `start/install.md`, `CONTRIBUTING.md` | | |

Paths in the table are relative to `docs/`, except `README.md`, `CONTRIBUTING.md` and
`pyproject.toml`. When a page moves or a module is added, change this table in the same
pull request. The site's sections are the captioned toctrees of `docs/index.md`: a new
page is listed in one of them, or in the toctree of a page listed there, as the API
reference lists its generated pages.

## Before merging

1. **Pages.** Update the mapped pages, and the README when its example or its list of
   guarantees is affected.
2. **API reference.** A new public name is added to the `__all__` of its module (for
   `telescopic.kernels`, of the package), whose functions and data a module's page
   shows, and to `__all__` in `src/telescopic/__init__.py` when it is exported there; a
   new public class or module, or a function that `telescopic` itself exports, is also
   listed in its section's table in `docs/api.md`, which generates its page
   (`CONTRIBUTING.md`, Documentation). The strict build fails on a reference that
   resolves to nothing.
3. **The specification.** A change that departs from `docs/design/spec.md` edits it in
   the same commit and records the decision in `notes/DECISIONS.md`.
4. **Numbers.** A figure quoted in the documentation comes from a script in
   `notes/experiments/` or `benchmarks/`, from `docs/_scripts/figures.py`, which draws the
   generated figures and checks their text, or from a tutorial's own executed cell. Change
   the figure when its source changes.
5. **CHANGELOG.** A change that a user can see has a line under *Unreleased*.
6. **Register.** Pages use the notation of `docs/theory.md`, American English spelling,
   no em dashes and no machine-specific paths. Outside `docs/design/`, they do not cite
   `notes/`, decision numbers or the specification's requirement numbers, which are
   internal.
7. **Identity.** A change to a color, a font, a size or the mark is made in
   `docs/_scripts/brand.py`, and a change to the package that moves the landing page's
   coupled pair is saved with `docs/_scripts/hero_pair.py`; a change to the package that
   moves a generated figure is redrawn with `docs/_scripts/figures.py`; `CONTRIBUTING.md`
   (Documentation) gives the commands, and the three scripts' `--check` pass.
8. **Build.** `scripts/verify.sh --docs` passes: the build runs the tutorials, treats
   warnings as errors and checks every cross-reference.

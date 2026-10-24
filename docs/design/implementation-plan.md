# telescopic: implementation plan (v0.1)

This document describes how to build what [`spec.md`](spec.md) specifies: the code layout, the internal interfaces, the milestones and the documentation. Tests are described in [`test-plan.md`](test-plan.md); every milestone ships with its tests.

## 1. Repository layout

```
telescopic/
├── pyproject.toml            # hatchling build; deps, extras, dependency groups; ruff, mypy, pytest, coverage
├── README.md                 # pitch, install, the spec §8.3 example, links
├── CHANGELOG.md  CONTRIBUTING.md  CITATION.cff  LICENSE  CLAUDE.md
├── .pre-commit-config.yaml   # ruff lint and format
├── .github/workflows/        # ci.yml (lint, types, tests, oldest stack, dist, slow on dispatch); docs.yml
├── src/telescopic/
│   ├── __init__.py           # public API, __version__
│   ├── warnings.py           # TelescopicWarning, ConvergenceWarning, InputWarning
│   ├── _random.py            # root seeds, replicate streams, buffered Streams and Draws
│   ├── target.py             # Target: batched log-density and gradient, validation, counters
│   ├── initial.py            # initial distributions: Normal, Uniform, Empirical, as_initial
│   ├── couplings.py          # reflection_maximal, maximal (with eta), discrete_maximal
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── base.py           # State, Kernel, Metropolis accept helper
│   │   ├── random_walk.py    # RandomWalk
│   │   ├── mala.py           # MALA
│   │   ├── hmc.py            # HMC (Heng & Jacob)
│   │   ├── independence.py   # Independence
│   │   ├── gibbs.py          # Gibbs
│   │   └── mixture.py        # Mixture
│   ├── estimators.py         # lag weights, thinning coefficients, unbiased covariance, quantiles, histograms
│   ├── engine.py             # lockstep lanes with refill; Replicate records
│   ├── parallel.py           # in-process and pool execution of index chunks; worker set-up; interrupts
│   ├── tuning.py             # adaptive Metropolis pilot, dual averaging, choose_kml
│   ├── measure.py            # SignedMeasure
│   ├── diagnostics.py        # TV and W1 bounds, tail index, variance shares, warning checks
│   ├── results.py            # Result, Estimate, Summary, MeetingTimes, TuningReport
│   ├── sampler.py            # CoupledSampler, sample
│   ├── io.py                 # versioned .npz save and load
│   ├── plotting.py           # matplotlib helpers (optional dependency)
│   ├── examples.py           # targets with exact answers
│   └── testing.py            # FixedDraws, check_coupling
├── tests/                    # see test-plan.md
├── benchmarks/               # run.py: timing and efficiency cases, JSON output, run by hand
├── docs/                     # Sphinx + MyST-NB (see §5)
└── notes/                    # plan, log, decisions, questions, experiments, reviews
```

## 2. Internal interfaces

### 2.1 Random numbers (`_random.py`)

```text
root_seed(rng) -> SeedSequence                        # SPEC 7 normalisation; records entropy and spawn key
replicate_generator(root, namespace, index) -> Generator   # PCG64DXSM(SeedSequence(root.entropy, spawn_key=root.spawn_key + (namespace, index)))
Streams(n_lanes, ndim, block=64)                       # per-lane buffers of normals and uniforms
  .assign(lane, generator)                             # a new replicate enters a lane: fresh buffers
  .view(lanes) -> Draws
Draws                                                  # what kernels see; rows follow the lanes of the view
  .normal(size) -> (n, size)      .uniform() -> (n,)     .generator(row) -> Generator
  .subset(rows) -> Draws          __len__
```

- **Buffers:** each lane holds a buffer of `block × ndim` normals and `block × 4` uniforms, with cursors. A request that a lane's buffer cannot satisfy refills that lane from its generator. Requests are gathered by advanced indexing.
- **Invariance:** buffer sizes depend only on `ndim`, so a replicate's draws do not depend on the lane layout (spec R-EXEC-2).
- **Tests:** `testing.FixedDraws` implements the `Draws` interface from prescribed arrays, for oracle tests.

### 2.2 Target (`target.py`)

```text
Target(log_prob, grad=None, *, args=(), kwargs=None, vectorize=False)
  .log_prob(x (n, d)) -> (n,)                # loops over rows unless vectorize
  .log_prob_and_grad(x) -> ((n,), (n, d))
  .has_grad, .n_log_prob, .n_grad
```

- **NaN or `+inf`:** raises `ValueError` naming the point.
- **Input arrays:** never modified.
- **Gradients:** a missing gradient raises in `log_prob_and_grad`.

### 2.3 States and kernels (`kernels/base.py`)

```text
State(x (n, d), log_prob (n,), grad (n, d) | None)
  .take(rows) -> State;  .put(rows, other)     # engine bookkeeping
Kernel
  needs_grad: bool
  step(target, state, draws) -> (state, accepted (n,))
  coupled_step(target, sx, sy, draws) -> (sx, sy, met (n,))
  tune(target, initial, root, **options) -> (kernel, TuningReport)   # returns self when already tuned
  init_state(target, x) -> State                                   # evaluates log_prob (and grad if needed)
```

- **Purity:** kernels are pure functions of their inputs and draws, and are picklable.
- **Meeting:** on a meeting, `coupled_step` copies the rows of `sx` into `sy`. Every kernel documents its draws per step in its docstring.

### 2.4 Couplings (`couplings.py`)

```text
reflection_maximal(mu_x, mu_y, chol, draws) -> (x, y, identical)
maximal(p, q, draws, eta=1.0) -> (x, y, identical, n_draws)       # p, q: lists of per-row distributions
discrete_maximal(log_p, log_q, draws) -> (i, j)                    # (n, K) log-probabilities
```

`p` and `q` provide `sample(generator)` and `log_density(value)`. Adapters wrap frozen SciPy distributions.

### 2.5 Estimators and measures (`estimators.py`, `measure.py`)

```text
lag_weights(t, k, m, lag) -> v_t                           # spec R-EST-4, integer arrays
correction_coefficients(t, k, m, lag, thin) -> c_t          # spec R-EST-5
averaging_times(k, m, thin) -> S
unbiased_covariance(first (R, p), second (R, p, p)) -> (p, p)
weighted_quantile(values, weights, q) -> quantiles         # isotonic projection, spec §3.4
SignedMeasure(atoms (N, d), weights (N,), replicate (N,), n_replicates)
  .per_replicate(h) -> (R, ...)
```

### 2.6 Engine (`engine.py`)

```text
Replicate(index, meeting_time | None, cost, atoms, weights, correction, distances, paths | None)
run(target, kernel, initial, indices, *, root, namespace, k, m, lag, thin=1, max_iter=None,
    lanes=64, store=True, keep_paths=False, slow_after=None, stop=None) -> list[Replicate]
```

- **Lane state:** replicate index, $t$, phase, $\tau$, cost, the `State` of $X$ and of $Y$, and the atom accumulators.
- **Each iteration:**
  1. `kernel.step` on lagging and after-meeting lanes;
  2. `kernel.coupled_step` on coupled lanes;
  3. bookkeeping;
  4. refill.
- **Storage:** atoms from $t = k$, with weights from `averaging_times` and `correction_coefficients`. `correction` marks the bias-correction atoms.
- **Stopping:** `stop` is a callable polled each iteration, used for interrupts. Once it returns true, no new replicates start.
- **Pilot use:** the meeting-time pilot calls `run(k=0, m=0, lag=1, store=False)`.

### 2.7 Parallel execution (`parallel.py`)

```text
execute(task, chunks, pool) -> list[list[Replicate]]   # in index order
make_chunks(indices, chunk_size) -> list[array]
```

- **Pool forms:**
  - `pool=None` runs in-process;
  - `pool=int` creates a `ProcessPoolExecutor` whose initializer caches the target and pins BLAS threads with `threadpoolctl`;
  - an `Executor` is used through `submit`;
  - a `.map`-only pool runs in waves.
- **Interrupts:** a `KeyboardInterrupt` in the parent cancels pending chunks and returns the completed prefix of indices.

### 2.8 Tuning (`tuning.py`)

```text
adaptive_metropolis(target, initial, root, n_chains=4, n_steps=None) -> AdaptiveRun(cov, scale, acceptance, chains)
dual_averaging(make_kernel, target, initial_states, root, target_accept, n_steps) -> step_size
choose_kml(meeting_times, quantile=0.99, multiple=10) -> (k, m, lag)    # spec R-TUNE-2
```

### 2.9 Results (`results.py`, `diagnostics.py`)

- **`Result`** holds:
  - the configuration, the frozen kernel and the seed;
  - per-replicate arrays and the `SignedMeasure`;
  - the tuning and pilot reports.

  It computes estimates with `estimators` and diagnostics with `diagnostics`.
- **`diagnostics.check(result) -> list[Issue]`** implements spec §7.4. `Result.check` emits the issues as warnings.

### 2.10 Sampler (`sampler.py`)

`CoupledSampler` holds:
- the `Target`, the initial distribution and the root seed;
- the kernel, frozen after tuning;
- the pilot meeting times, and the chosen `k`, `m`, `lag` and `thin`;
- the next replicate index and the accumulated replicates.

`sample(...)` is `CoupledSampler(...).run(...)`.

## 3. Milestones

Each milestone is one reviewed unit (`CLAUDE.md`, Milestones). It lands with its tests, docstrings and a CHANGELOG line, then goes through:
- `/simplify`;
- `/code-review` at the effort its size sets;
- fixes;
- a log entry.

| M | Content | Exit criteria |
|---|---|---|
| M0 | Scaffold: pyproject, layout, CI, pre-commit, docs skeleton, licence files, `warnings`, `_random` | the commands of `CLAUDE.md` pass; test-plan §2.1 green |
| M1 | Coupling primitives, `testing.FixedDraws` | test-plan §2.2 green |
| M2 | `Target`, `initial`, `State`, `Kernel`, `RandomWalk`, `Independence`, `Mixture` | test-plan §2.3–2.4 (these kernels) green |
| M3 | `estimators`, `measure`, `engine` | test-plan §2.5–2.6 green, including invariance across lane counts |
| M4 | `results`, `diagnostics`, warnings, summary | test-plan §2.7–2.8 green |
| M5 | `tuning`, `CoupledSampler`, `sample` | test-plan §2.9 green; the spec §8.3 example runs |
| M6 | `MALA`, `HMC`, `Gibbs` | test-plan §2.4 (these kernels) green |
| M7 | `parallel`, interrupts, budget mode (should) | test-plan §2.10 green; serial and pool results identical |
| M8 | `io`, `plotting`, `examples`, `testing.check_coupling` | test-plan §2.11 green |
| M9 | Slow statistical suite, benchmarks, calibration of the warning thresholds | test-plan §2.12 green; benchmark JSON committed under `benchmarks/results/` |
| M10 | Docs: theory, user guide, tutorials, API reference, README | `sphinx-build -W` passes with executed tutorials |
| M11 | Independent audit of the whole package, fixes, checkpoint report to Josh | audit findings fixed or recorded in `notes/reviews/` |

## 4. Engineering conventions

These add to the rules of `CLAUDE.md`.

- **Arrays:**
  - batch-first `float64` arrays everywhere;
  - no Python loops over atoms or replicates where a NumPy expression exists;
  - Python loops over lanes only for per-replicate generators, refills and user conditionals.
- **Log space:** accept or reject with `log u < log_ratio`, a strict inequality. NumPy's `random()` is in [0, 1), so $\log u$ can be $-\infty$, and a proposal with $\log\pi = -\infty$ must still be rejected.
- **Whitening:** proposals are generated and couplings computed with the same Cholesky factor.
- **Meetings:** compare with `np.array_equal` only in tests and debug assertions; the kernels' flags drive the engine.
- **Optional dependencies:** matplotlib, tqdm and pandas are imported inside the functions that use them.

## 5. Documentation plan

- **Toolchain:** Sphinx, MyST-NB (MyST Markdown notebooks executed at build time, cached), numpydoc, furo, sphinx-copybutton and sphinxcontrib-bibtex.
- **Tutorials:** each runs in under 2 minutes on a laptop CPU.
- **Pages:**
  1. *Getting started*: install, the README example expanded, and what the summary means.
  2. *User guide*:
     - targets and initial distributions;
     - kernels and couplings, and how to choose them;
     - `k`, `m` and `lag`, and the automatic pipeline;
     - results: signed measures, estimates, quantiles and plots;
     - diagnostics and warnings;
     - parallel runs and budgets;
     - writing a custom kernel;
     - pitfalls: multimodality, heavy tails, a poor $\pi_0$, non-deterministic log-densities.
  3. *Theory*: spec §3 in clean notation, with the telescoping argument, the lag-$L$ weights, the TV bound and the budget estimator derived.
  4. *Tutorials:*
     1. Quickstart: emcee's line-fitting example, with telescopic and emcee side by side.
     2. How it works: coupled paths meeting; the MCMC part and the bias correction; signed measures.
     3. Choosing $k$, $m$ and the lag: meeting times, TV bounds and efficiency.
     4. Gradients: MALA and HMC on a 50-dimensional correlated Gaussian and a logistic regression.
     5. Gibbs couplings: the pump-failure model.
     6. Parallel runs and compute budgets.
     7. When unbiased is not enough: multimodal targets and heavy tails, with the warnings they raise.
  5. *API reference* (autosummary).
  6. *Design notes*: the spec, the plans and the landscape report.
  7. *Changelog, contributing, citing.*

## 6. CI (GitHub Actions)

| Job | Contents |
|---|---|
| `lint` | `ruff check`, `ruff format --check`, `mypy` |
| `test` | Ubuntu with Python 3.11–3.14, plus macOS and Windows at the oldest and newest Python; `pytest` (fast tier); coverage on one job |
| `oldest` | Python 3.11 with NumPy 1.26 and SciPy 1.12 |
| `docs` | `sphinx-build -W --keep-going` with executed tutorials |
| `dist` | `python -m build`; install the wheel in a clean environment; import smoke test |
| `slow` | `pytest -m slow`, on manual dispatch |

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Weight or index bugs bias estimates by amounts too small to detect statistically | algebraic identity tests with Hypothesis; exact finite-state expectations; a perfect-kernel structural test (test-plan §2.5) |
| A coupling that "meets" up to rounding only | the flag-and-copy contract; a binomial test of meeting frequency against the exact Gaussian formula (test-plan §2.2) |
| Heavy-tailed estimators make statistical tests flaky | 5-standard-error tolerances with a sequential rerun at 4× $R$; coverage tests in the slow tier only |
| Python overhead with cheap targets | lockstep lanes over replicates; a vectorised log-density path |
| Per-replicate streams slow down lockstep batches | buffered block draws (1.4× measured); a benchmark guards it |
| Users mistake unbiased for accurate | warnings by default; a tutorial on failure modes; guarantees stated as in spec §3.3 |
| Pool pickling and BLAS oversubscription | worker initializers, `threadpoolctl`, and documented guards |

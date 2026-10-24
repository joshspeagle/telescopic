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
│   ├── _parallel.py          # in-process and pool execution of index chunks; worker set-up; interrupts
│   ├── _progress.py          # progress bars (tqdm) or lines on stderr
│   ├── tuning.py             # adaptive Metropolis pilot, TuningReport, dual averaging, choose_k
│   ├── measure.py            # SignedMeasure
│   ├── diagnostics.py        # TV and W1 bounds, tail index, variance shares, warning checks
│   ├── results.py            # Result, Estimate, Summary
│   ├── sampler.py            # CoupledSampler, sample
│   ├── io.py                 # versioned .npz save and load
│   ├── plotting.py           # matplotlib helpers (optional dependency)
│   ├── examples.py           # targets with exact answers
│   └── testing.py            # fixed_draws, random_draws, check_coupling
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
Streams(n_lanes, width=...)                            # per-lane buffers of normals and uniforms
  .assign(lane, generator)                             # a new replicate enters a lane; its buffers read generator.spawn(2)
  .view(lanes) -> Draws
Draws                                                  # what kernels see; rows follow the lanes of the view
  .normal(size) -> (n, size)      .uniform() -> (n,)     .log_uniform() -> (n,)
  .generator(row) -> Generator    .subset(rows) -> Draws  __len__
```

- **Buffers:** each lane holds a buffer of normals and one of uniforms, each fed by its own child generator and read in order through a cursor. A request that a lane's buffer cannot satisfy moves the unread numbers to the front and tops the buffer up; a request wider than the buffer widens it. Requests are gathered from a strided view of whole windows.
- **Invariance:** each lane's numbers are the in-order streams of its replicate's children, so they depend on neither the lane layout nor the buffer width (spec R-EXEC-2).
- **Tests:** `testing.fixed_draws` replays prescribed arrays, for oracle tests; `testing.random_draws` serves fresh numbers to any number of rows from one generator.

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
  step(target, state, draws) -> (state, acceptance (n,))   # acceptance probabilities
  coupled_step(target, sx, sy, draws) -> (sx, sy)
  tune(target, initial, rng=None, *, pilot=None) -> (kernel, TuningReport | None)   # (self, None) when tuned
  init_state(target, x) -> State                                   # evaluates log_prob (and grad if needed)
```

- **Purity:** kernels are pure functions of their inputs and draws, and are picklable.
- **Meeting:** a coupling that makes the proposals identical gives `sy` a copy of `sx`'s proposal and log-density; the engine detects meetings (§4). Every kernel documents its draws per step in its docstring.
- **Batch invariance:** kernels hold their Cholesky factor and its inverse, and apply them with `np.einsum` (spec R-KER-6).

### 2.4 Couplings (`couplings.py`)

```text
reflection_maximal(mu_x, mu_y, chol, inverse, draws) -> (x, y, identical)   # inverse = chol^{-1}
maximal(p, q, draws, eta=1.0) -> (x, y, identical, n_draws)       # p, q: lists of per-row distributions
discrete_maximal(log_p, log_q, draws) -> (i, j)                    # (n, K) log-probabilities
```

`p` and `q` are frozen SciPy-like distributions: `rvs(random_state=...)` and a normalised `logpdf` or `logpmf`.

### 2.5 Estimators and measures (`estimators.py`, `measure.py`)

```text
averaging_times(k, m, thin) -> S
correction_coefficients(t, k, m, lag, thin) -> c_t          # spec R-EST-5; v_t/(m-k+1) when thin = 1
unbiased_covariance(first (R, p), second (p, p)) -> (p, p)   # second: mean of the replicates' second moments
weighted_quantile(values, weights, q) -> quantiles         # isotonic projection, spec §3.4
SignedMeasure(atoms (N, d), weights (N,), replicate (N,), n_replicates)
  .per_replicate(h) -> (R, ...)
  .mass(cell (N,), n_cells) -> (R, n_cells)                 # histograms; cell -1 = outside
```

### 2.6 Engine (`engine.py`)

```text
Replicates(index, meeting_time, cost, atoms, weights, owner, distance_owner, distance_time,
           distance, zero_density, requested, n_log_prob, n_grad, seconds, paths)   # inf = censored
merge(parts) -> Replicates
run(target, kernel, initial, indices, *, root, namespace, k, m, lag, thin=1, max_iter=None,
    lanes=64, store=True, keep_paths=False, monitor=None) -> Replicates
Status(started, finished, slowest, t)                       # given to the monitor every iteration
```

- **Lane state:** replicate index, $t$, phase, $\tau$, cost, the `State` of $X$ and of $Y$, and the atom accumulators.
- **Each iteration:**
  1. `kernel.step` on lagging and after-meeting lanes;
  2. `kernel.coupled_step` on coupled lanes;
  3. bookkeeping, including the exact comparison of positions after coupled steps and at $t = L$;
  4. refill.
- **Storage:** atoms from $t = k$, with weights from `averaging_times` and `correction_coefficients`.
- **Monitoring:** the engine emits no warning. It records the facts (censoring, starts with zero density, evaluations, time) and passes a `Status` to `monitor` every iteration; the sampler warns about slow pairs from it, shows progress, and stops new replicates on an interrupt by returning true. Results and facts reach the parent from pool workers, which warnings would not.
- **Pilot use:** the meeting-time pilot calls `run(k=0, m=0, lag=1, store=False)`.

### 2.7 Parallel execution (`_parallel.py`)

```text
opened(pool)                                 # context: an integer becomes spawned Workers for the block
execute(job, indices, pool, monitor) -> Replicates   # job: a partial of engine.run; requested = indices
Interrupt().catching()                       # the first Ctrl-C sets .stopping; the second raises
```

- **Pool forms:**
  - `pool=None` runs in-process;
  - `pool=int` opens a spawn `ProcessPoolExecutor` for one call of `run` or `meeting_times`, whose initializer ignores SIGINT and keeps a stop event;
  - an `Executor` is used through `submit`;
  - a `.map`-only pool runs in waves.
- **Monitor:** the engine's monitor is the one channel: in-process it sees every iteration; on a pool, the finished replicates after each chunk and every 0.1 s. It drives progress and slow-pair warnings, and its return value stops the run.
- **Interrupts:** the first Ctrl-C sets the workers' stop event and cancels the chunks not yet started, except the first (spec §6.5). A `.map` pool returns its completed waves. The second raises, and terminates the workers that telescopic started.
- **Hygiene:** a task in the main thread of another process pins BLAS to one thread and ignores SIGINT. Jobs are pickled with `cloudpickle` when it is installed.

### 2.8 Tuning (`tuning.py`)

```text
adaptive_metropolis(target, initial, rng, *, n_steps=None) -> PilotRun(samples, log_prob, acceptance, n_steps)   # 4 chains, pooled adaptation
dual_averaging(kernel_at, target, pilot, rng, target_acceptance, step_size) -> step_size   # 1000 steps from the pilot's last states
choose_k(meeting_times, quantile=0.99) -> k    # spec R-TUNE-2; inverted-CDF quantile; censored = inf
```

### 2.9 Results (`results.py`, `diagnostics.py`)

- **`Result`** holds:
  - the configuration, the frozen kernel and the seed;
  - per-replicate arrays and the `SignedMeasure`;
  - the tuning and pilot reports.

  It computes estimates with `estimators` and diagnostics with `diagnostics`.
- **`diagnostics.issues(result) -> list[Issue]`** implements spec §7.4. `Result.check` emits the issues as warnings, and the sampler emits them at the end of each run.

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
| M1 | Coupling primitives, `testing.fixed_draws` | test-plan §2.2 green |
| M2 | `Target`, `initial`, `State`, `Kernel`, `RandomWalk`, `Independence`, `Mixture` | test-plan §2.3–2.4 (these kernels) green |
| M3 | `estimators`, `measure`, `engine` | test-plan §2.5–2.6 green, including invariance across lane counts |
| M4 | `results`, `diagnostics`, warnings, summary | test-plan §2.7–2.8 green |
| M5 | `tuning`, `CoupledSampler`, `sample` | test-plan §2.9 green; the spec §8.3 example runs |
| M6 | `MALA`, `HMC`, `Gibbs` | test-plan §2.4 (these kernels) green |
| M7 | `_parallel`, interrupts, progress; budget mode deferred (DECISIONS #22) | test-plan §2.10 green; serial and pool results identical |
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
- **Whitening:** proposals are generated and couplings computed with the same Cholesky factor, applied with `np.einsum` together with its precomputed inverse (spec R-KER-6).
- **Meetings:** the engine compares positions exactly (`np.all(x == y, axis=1)`), never with a tolerance, and never recomputes $Y$ (spec R-EST-2).
- **Optional dependencies:** matplotlib, tqdm and pandas are imported inside the functions that use them.

## 5. Documentation plan

- **Toolchain:** Sphinx, MyST-NB (MyST Markdown notebooks executed at build time, cached), numpydoc, furo and sphinx-copybutton.
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
| `test` | Ubuntu with Python 3.11 to the newest release, plus macOS and Windows at the oldest and newest Python; `pytest` (fast tier); coverage on one job |
| `oldest` | Python 3.11 with NumPy 1.26 and SciPy 1.12 |
| `docs` | `sphinx-build -W --keep-going` with executed tutorials |
| `dist` | `python -m build`; install the wheel in a clean environment; import smoke test |
| `slow` | `pytest -m slow`, on manual dispatch |

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Weight or index bugs bias estimates by amounts too small to detect statistically | algebraic identity tests with Hypothesis; exact finite-state expectations; a perfect-kernel structural test (test-plan §2.5) |
| A coupling that "meets" up to rounding only | copy-on-event couplings and the engine's exact comparison; a binomial test of the frequency of equal outputs against the exact Gaussian formula (test-plan §2.2) |
| Heavy-tailed estimators make statistical tests flaky | sequential escalation (test-plan §1); coverage tests in the slow tier only |
| Python overhead with cheap targets | lockstep lanes over replicates; a vectorised log-density path |
| Per-replicate streams slow down lockstep batches | buffered draws gathered by windows (2.2× a shared generator, 0.33 µs per lane-step at 256 lanes, $d = 10$; `notes/reviews/m0.md`); a benchmark guards it |
| Batched linear algebra breaks bitwise invariance | `einsum`-only kernel arithmetic (spec R-KER-6) and a batch-of-one test per kernel |
| Users mistake unbiased for accurate | warnings by default; a tutorial on failure modes; guarantees stated as in spec §3.3 |
| Pool pickling and BLAS oversubscription | worker initializers, `threadpoolctl`, and documented guards |

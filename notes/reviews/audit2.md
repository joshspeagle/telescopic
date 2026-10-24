# Audit 2: the second audit of the whole codebase

**Scope:** the whole package at dc2f0df, the M11 checkpoint, before it landed on `main`.
None of the reviewers wrote the code. Three kinds of review ran in parallel:

- `/simplify`, four reviewers, one per angle: reuse, simplification, efficiency and
  altitude;
- `/code-review` at maximum effort, in one pass;
- four correctness reviewers, one per group of modules: kernels and couplings; the
  engine and results; the sampler, tuning and pools; the diagnostics, testing and the
  docs.

Each finding was reproduced by a script, or read against the spec, before it was fixed.
The two decisions that change a statistical result went to adversarial reviewers: #59 to
one, who refuted it, and #67, which replaced it, to three. The phase's own diff then had
its `/simplify` (four reviewers, and two more for #67) and its `/code-review` at
extra-high effort.

**Diff:** 2fc10f6..4377bab. Where a decision cites evidence, its script and output are in
`notes/experiments/audit2_*`.

## Outcome

### Gradients that are not finite (979817d, c875d3a, 8a756c4, 4c62fd4)

- **MALA and HMC sampled a restricted target, silently** (kernels reviewer, altitude
  reviewer). Where the log-density is finite but the gradient is not, they rejected the
  proposal: on N(0, 1) with a NaN gradient beyond 1, E[x] was -0.288 ± 0.003 with MALA
  (z = -91) and -0.286 ± 0.002 with HMC (z = -122), where random-walk Metropolis gave
  0.001 ± 0.005 (`notes/experiments/audit2_nan_gradient.out`).
- **The first fix raised with the point** (DECISIONS #59). Its adversarial reviewer
  refuted it: a mixture's textbook responsibilities are NaN where every density
  underflows, 38.6 standard deviations out, and dual averaging's second step size sends
  proposals there, so default tuning raised in 10 of 10 runs that rejection had
  completed with every |z| ≤ 1.84 (`notes/experiments/audit2_gradient_mixture.out`).
- **Then zero drift**, which the altitude reviewer had proposed: such a gradient is taken
  as 0, which keeps both kernels exact, counted, and warned (DECISIONS #67). Of its three
  adversarial reviewers, two upheld it with changes and one refuted it, and all three
  found its fault at starts: a MALA chain that starts where a correct gradient underflowed
  cannot come back, since a move back needs the reverse proposal, which carries the full
  drift, to return. From N(45, 1) on the mixture, `sample()` returned E[x] = 40.10 ± 0.03
  for a truth of 0 (`notes/experiments/audit2_zero_drift_costs.out`).
- **The rule now:** at the states chains start from, such a gradient raises with the
  point, as before the audit; where chains move, it is taken as 0. The pilot pairs and
  replicates warn as soon as this process's target counts one, and after a run on a pool;
  `Result.n_grad_zeroed` keeps the count. The question stays open for Josh in
  `notes/QUESTIONS.md`.

### The maximal coupling (32d825d, c67d88b, a2884ee)

- **It changed a chain's law at infinite log-densities** (kernels reviewer). Draws that
  underflow to the edge of their support, as Gamma(0.01) draws do in about 6e-4 of cases,
  have a log-density of -inf or +inf: the first test made such a pair identical through a
  NaN ratio, the loop rejected q's own such draws, and +inf raised in coupled steps only.
  Such draws are now never identical and always accepted (DECISIONS #60;
  `notes/experiments/audit2_maximal_lumps.out`).
- **SciPy's Dirichlet draws of shape (1, k)** were refused for k = 3 and read as two
  points for k = 2, so Dir(1, 2) and Dir(2, 1) met in every pair. They are evaluated as
  shape (k,), keyed on the distribution, since the phase's altitude review found that
  keying on the shape broke one-row `matrix_normal` and `matrix_t` in coupled steps only.
- **The phase's code review found two more faults and a cost.** NumPy's floats warned at
  inf - inf, which a warnings filter of "error" turns into an exception; SciPy refuses a
  Dirichlet component of 0 whose alpha is below 1, a pole that draws reach by underflow.
  Log ratios are now Python floats, and such a component is a pole: Y_1 = 0 in 0.2256 of
  the coupled draws of Dir(0.002, 0.001, 1) against 0.2264 of its direct ones, ± 0.0004.
  The cost: such draws can be frequent, and there pairs are not identical. Beta(0.01,
  0.01) puts 0.345 of its draws at 1.0, its mass within 2^-53 of 1; against
  Beta(0.011, 0.01), whose densities overlap by 0.971, pairs were identical in 0.623 of
  cases and equal in 0.959, which the engine's exact comparison counts as met. The
  docstring and the spec now state the meeting probability over draws of finite
  log-density.

### check_coupling (eb1af47)

From copies of one pair every row held the same state, so a coupling that read another
row's log-density got the right one's p-values in 20 of 20 seeds, while one such
coupling biased a run's E[x] by 6.3 standard errors (diagnostics reviewer, new evidence
against a limitation that M8 recorded). Pairs (x, y) and (y, x) now share a batch in
random rows, each chain compared among the rows that start it at one point, and a NaN
p-value fails (DECISIONS #61; `notes/experiments/audit2_check_coupling_rows.out`).

### Diagnostics (b41bb04, 2816ac1)

When every pair was censored, the largest meeting time was NaN, and `max(nan, max_iter)`
kept it: `check()` and the summary dropped the slow-pair warning that the run had
emitted (code review, reuse and altitude reviewers). The check now reads the largest
stopping time, as the run's live check does. `variance_share` read NaN as no variance,
and `hill` gave NumPy's errors on 2-D values and on a scalar.

### Argument checks (79c6d8b, 4acbc81, 2816ac1, 4377bab)

A `Fraction` passed twelve hand-written interval checks and failed in NumPy; quantile
weights summing to R gave wrong quantiles without an error; a cell past the last spilled
into the next replicate; `dual_averaging` took a target acceptance of 1.5, which gave a
step size of 3e-124; the bounds and estimators took mismatched or non-integer arrays
(code review, engine and diagnostics reviewers). One helper reads bounded reals
(DECISIONS #62), one predicate checks integer arrays, and the estimator weights stay
exact up to int64. `dims` is checked in one place, which names it for an integer.
`Independence` read a proposal's `mean` method as one dimension (the phase's code
review); a method no longer fixes the dimension.

### Ctrl-C, pools and threads (b7dc510, 4da3f9a, 84995c8)

- **A Ctrl-C after the replicates lost a finished run** (sampler reviewer): the handler
  covered only the replicates, and the merge and the result of a second run at 3.9
  million atoms took 4.1 s after them. The handler is now set when the replicates start
  and kept until `run` returns (DECISIONS #63). Set at first for the whole of `run`, it
  replaced a caller's own handler during tuning, which the phase's altitude review found.
- A process that ignores SIGINT got it back during a run, and a handler that Python did
  not set raised on restore or was replaced; both are left alone.
- `pool=np.array(2)` passed the pool check and then served as the pool (altitude review);
  the check returns the count.
- SciPy's multivariate normal draws differ between BLAS thread counts from d = 200, so a
  pool's run can differ in the last bits; the parallel guide says so (DECISIONS #64).

### State, dead code and speed (f4a94af, bbbf851, 1644df6)

`Replicates.cost` always equalled its formula and is a property; `Status.started` was
never read (DECISIONS #65). The pilot at d = 100 took 16.9 s against 25.1 s, the Gibbs
tutorial's run 1.25 s against 2.13 s, and a joint histogram of 3 coordinates of 40 bins
0.062 s and 23 MB against 3.06 s and 629 MB, with sparse masses (DECISIONS #66;
`notes/experiments/audit2_performance.out`); the histogram's indicator is built as CSR,
which the phase's efficiency review found 15% to 30% faster for one coordinate.

### Tests and docs (189d28c, 573c85c, f2578d6, d12b581)

The README's code runs as a test, where the only test ran the first draft's example; the
finite-state exactness test of test plan 2.4 couples every pair of rows of a 4-state
transition matrix; duplicated helpers and tests that caught nothing another misses went.
Four statements of the docs that the code contradicted are corrected, and four facts
that had two homes have one.

## The phase's /simplify

Four reviewers on 2fc10f6..f2578d6, and two on the zero-drift commit. Applied: the
altitude, pool and Dirichlet fixes above; one log-ratio in the maximal coupling; one
Bonferroni form in `check_coupling`; the kernel's message once; the dims check in
`_types`; one predicate for integer arrays; the CSR indicator; the counters under one
tuple of names; one finiteness test in the target; shared test references. Answered:

- **The pilot's `exp(fmax(log_alpha, -inf))` repeats `kernels.base.acceptance`** (reuse).
  Kept: `kernels.base` imports `tuning`, so the reuse needs a new module below both for
  one line.
- **`check_coupling` swaps rows with `generator.random(n) < 0.5`** where
  `generator.integers(2, size=n)` would do (reuse). Kept: the change alters the stream,
  and with it the seeded counts of `notes/experiments/audit2_check_coupling_rows.out`.
- **The NaN gradient `np.where(x > 1, np.nan, -x)` is written in two tests** (reuse of
  #67). Kept: two copies do not earn a helper, as the reviewer found.

## The phase's /code-review

Extra-high effort on 2fc10f6..4c62fd4, one pass: eight findings, two of them defects.

- **NumPy's floats warned at inf - inf** in the maximal coupling's log ratio, which the
  phase's simplify review had made one subtraction (2816ac1); under a warnings filter of
  "error" a Gamma(0.01) pair raised. Fixed (a2884ee); the test table returns NumPy floats,
  as SciPy does.
- **SciPy's Dirichlet refuses a component of 0 whose alpha is below 1.** Fixed: a pole
  (a2884ee), tested on Dir(0.001, 0.001, 1).
- **The documented meeting probability ignored draws of infinite log-density**, which
  Beta(0.01, 0.01) makes frequent. The docstring, the spec and DECISIONS #60 state it
  and its cost (above). Taking `p is q` as identical at such draws, the reviewer's other
  option, is recorded as an alternative: no caller in the package couples equal laws.
- **`_stale` kept a `kernel` parameter it no longer used.** Removed (4377bab).
- **`Independence` read a `mean` method as one dimension.** Fixed and tested (4377bab).
- **The CHANGELOG said a run on a pool warns as it meets a zeroed gradient**; it warns
  when it ends. Corrected (4377bab).
- **The performance record named no commit for its second block.** It names bbbf851
  (4377bab). The zero-drift record named c1daae9, a commit outside the history: rerun at
  its cherry-pick c875d3a, with identical figures, and relabelled.
- **The README test writes a temporary directory under tests/.** Answered below.

## Answered without a change

- **`Result` keeps its replicates private** (altitude review): plotting, saving and the
  diagnostics read `Result._replicates` and `Result._stopped`. Kept: the package's
  modules share them, and a public attribute would put the engine's schema into the API
  (spec §7.1).
- **Starts of zero density are not an issue of the result** (altitude review): they warn
  when the run ends, but `check()`, the summary and a loaded result do not report them.
  Kept: they concern the inputs, not convergence, and `Replicates.zero_density` keeps
  the indices.
- **MALA in whitened coordinates**, 1.6 times faster at d = 50 (efficiency review):
  deferred, since it changes MALA's numbers and the coupling's interface (DECISIONS #66).
- **Tests write into tests/** (the phase's code review): the README test, as the saving
  tests do, writes a temporary directory there, which fails on a read-only checkout.
  Kept: running the code without a file needs `exec`, which ruff's S102 refuses and
  CLAUDE.md allows no ignore rule for, and the system's temporary directory is outside
  the repository, which the hermetic rule excludes.

## Checks

At 4377bab: ruff, the format check and mypy pass; 607 fast tests in 41 s and 20 slow
tests in 125 s pass; the strict docs build, with the tutorials executed, passes in
2 min 15 s.

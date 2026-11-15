# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- Per-replicate random streams keyed by the root seed, a namespace and the replicate
  index. Each replicate reads its normals and uniforms in order from two child generators,
  buffered for lockstep batches, so no batch layout or buffer width changes a number.
- Warning classes `TelescopicWarning`, `ConvergenceWarning` and `InputWarning`.
- Coupling primitives: the reflection-maximal coupling of Gaussians, the rejection maximal
  coupling with the bounded-variance `eta` option, and the maximal coupling of discrete
  distributions (`telescopic.couplings`); `telescopic.testing.fixed_draws` to inject
  random numbers in tests, and `telescopic.testing.random_draws` for fresh ones.
- `Target` with a batched interface over scalar or vectorized log-densities and gradients;
  initial distributions `Normal`, `Uniform`, `Empirical` and `as_initial`.
- Kernels with their couplings: `RandomWalk` (reflection-maximal or maximal proposals),
  `Independence` with a `MultivariateT` proposal, and `Mixture`. A row's transition does
  not depend on the other rows of its batch.
- The engine (`telescopic.engine.run`): replicates of coupled chains in lockstep lanes
  with refill, exact meeting detection, the lag-$L$ and thinned estimator weights, signed
  measures, distances between the chains and censoring. A replicate's output does not
  depend on the number of lanes, bitwise for a log-density evaluated point by point. The
  output records the run's settings and root seed, which `Result` and `merge` read.
- `SignedMeasure`, the unbiased covariance estimator and quantiles of signed measures by
  isotonic projection (`telescopic.measure`, `telescopic.estimators`).
- `Result` with estimates and confidence intervals for any test function, unbiased
  covariances, quantiles, credible intervals and signed histograms; total variation
  bounds from the meeting times and 1-Wasserstein bounds from the distances between the
  chains; a summary table; and the warnings of the specification (`telescopic.results`,
  `telescopic.diagnostics`).
- Automatic tuning and the sampler: adaptive Metropolis pilots whose chains share their
  adaptation, set the kernels' unset parameters and warn when they disagree; the choice
  of `k`, `m` and the lag from pilot meeting times; `CoupledSampler` and
  `telescopic.sample`, whose runs warn about the conditions of the specification.
- Gradient and Gibbs kernels: `MALA` and `HMC` (with a random-walk meeting step) with
  the reflection-maximal coupling, tuned by dual averaging; `Gibbs` with maximal,
  common-random-number and two-scale couplings of user-written conditionals, and
  `Target(log_prob=None)` for kernels that need no log-density. Kernels declare what
  they evaluate (`needs_grad`, `needs_log_prob`) and check a target for it
  (`check_target`).
  A kernel's `step` returns acceptance probabilities.
- Parallel runs: `pool=` takes a number of worker processes, an executor or a pool with
  `map`, and gives the same replicates as a run in-process, bitwise for a log-density
  evaluated point by point. The first Ctrl-C lets the
  replicates in progress finish and returns a result flagged `interrupted`, which the
  sampler's next run completes; the second stops at once. `progress=` shows a tqdm bar
  or lines on stderr. The `parallel` extra installs cloudpickle, which ships functions
  defined in notebooks to the workers.
- `n_workers=` on `sample` and `CoupledSampler` states the number of workers of an
  executor or a pool of your own. A pool that does not report its size, such as a
  cluster's, was counted as having as many workers as this process may use CPUs, which
  left its other workers idle. The argument replaces a reported size, and with a
  log-density evaluated point by point it changes no replicate.
- `log_prob_and_grad=` on `sample`, `CoupledSampler` and `Target`, in place of `grad`,
  takes a function that returns the log-density and its gradient together, such as
  JAX's `value_and_grad`, and is called once where `log_prob` and `grad` were called one
  after the other. `log_prob` is still required, for the evaluations that need no
  gradient, and a run gives the same numbers and counts as with the two functions when
  they return the same numbers.
- A pool's chunks hold at most 1,024 replicates. Above 4,096 replicates per worker, four
  chunks per worker grew with the run, to 62,500 replicates each for 1,000,000 on 4
  workers; now a worker's memory, the result it sends back and the steps of the progress
  bar stay bounded. This process still holds every replicate.
- `Result.save` and `Result.load` (a versioned `.npz` file, without pickle); joint
  signed histograms of several coordinates (`Result.histogram`); plots of meeting times,
  the TV bound, signed marginals and corner plots (`telescopic.plotting`, with
  matplotlib, and the `Result.plot_*` methods, on new or given axes); example targets
  with exact moments (`telescopic.examples`); and `telescopic.testing.check_coupling`,
  which checks a custom kernel's coupling, its meetings and its cached evaluations.
- A slow tier of statistical tests (`pytest -m ""`): unbiasedness of every kernel family,
  interval coverage, TV bounds against exact distances, published meeting times and
  calibration over the prior. `benchmarks/run.py` times the setups of the
  specification's performance targets. The tuning pilot runs of order d² steps beyond 20
  dimensions, and the variance warning needs one replicate holding more than half the
  variance; both were measured against their alternatives.
- `import telescopic` imports every public submodule but `plotting`, and `progress=True`
  draws tqdm's text bar, in notebooks too.
- `Result.diagnostics()` and the summary report the values behind two warnings, the tail
  index of the meeting times and the largest share of one replicate in a variance.
- Documentation: a Start section, how-to guides, six tutorials executed when the site is
  built, a theory page, the API reference, reference pages and a citing page.
- The documentation site is published at <https://joshspeagle.com/telescopic/> from each
  push to `main`.
- The documentation site has its own theme and identity: the Focus mark, two chain colors
  that mean the two chains, Red Hat type served with the site, light and dark themes, a
  landing page with a coupled pair drawn by telescopic itself, and navigation by section
  (Start, Tutorials, How-to guides, Theory, Reference, Project). Every plot of the
  executed pages shares one style. The design records are no longer built into the site.
- Each tutorial has one title in the sidebar, on its page and in the pager, and the
  comparison with emcee is part of Coming from emcee, under Start. The tutorials are
  ordered as a learning path, with the failure modes last, and the pager is their only
  "next" link.
- Warnings on the executed pages print as their class and message, without the path of
  the notebook's temporary file, and wrap within the column.
- Every figure on the executed pages has alternative text and a caption.
- A Concepts section explains the method without code, one idea per page with its
  figures: why a burn-in biases MCMC, two chains that meet, the telescoping
  correction, signed measures, not samples, error bars from replicates and what
  replaces R-hat and ESS, and meeting times and the distance to stationarity.
- Two Start pages: What telescopic does, without code, says what problem the package
  solves, what it returns and what it does not do; Is telescopic right for my problem?
  places a problem among good and poor fits, each with its reason and a link to the page
  that holds the fact.
- The API reference is an index of one-line summaries grouped by section, with a page for
  each class, its attributes and methods, a page for `sample`, and a page for each module
  that the index lists, its functions and a table of its classes. The page of a class with
  two or more members lists them under "On this page".
- The API reference cites no internal requirement or decision numbers; where such a
  citation carried meaning, it links the page that explains it.
- The docstring of `telescopic.sample` describes each of its parameters, and that of
  `Estimate` its `level`.
- The docstrings of `sample`, `CoupledSampler.run`, `Result.estimate`, `Result.summary`,
  each kernel class and `testing.check_coupling` end with a short example, which the
  tests run.
- Every parameter that a docstring lists has a description, and the strict documentation
  build fails on a parameter that a docstring leaves out of its Parameters section or
  lists there without a description.
- Signatures in the API reference show the parameters' names and defaults, one per line
  when long, and the types stay in the docstrings, which state the type of every value
  that a public method or property returns. The types link to the documentation of
  telescopic, Python, NumPy and matplotlib.
- American spelling throughout the package, its tests and the site, with "Atchadé" spelled
  one way; codespell checks it in `scripts/verify.sh`, CI's lint job and a pre-commit
  hook.
- The Quickstart writes the lag as $L$ in its equations, defines the pilots and a
  consistent estimate where it first uses them, and says how to turn on the tail check.
- Theory opens with what it derives, writes the algorithm without the meeting time and
  defines it after, and says what the conditions of unbiasedness ask in practice before
  stating them.
- The kernels guide defines its coupling terms before its table, the result-fields
  reference says what Hill's estimate and the Clopper-Pearson bound measure, and the
  warnings reference when each warning is emitted; the docstring of `diagnostics.hill`
  gives the rule for ties.
- The guide to choosing an initial distribution links `initial.as_initial`, which reads
  `init`.
- Fixes from an audit of the whole package: a number argument of the wrong type raises
  `TypeError`, and one out of bounds `ValueError`, naming the argument; `names` must be
  distinct strings, one per coordinate; `Kernel.check_ndim` checks a kernel's dimension;
  variances are computed about the atoms' mean; tuning does not depend on the
  coordinates' scales; the engine advances up to 256 replicates at a time.
- Fixes from a second audit: a gradient that is not finite where the log-density is
  finite is taken as 0 where chains move, which keeps MALA and HMC exact, and raises
  with the point where they start, where MALA and HMC rejected such proposals and so
  sampled a restricted target; a run that meets one warns as it does, or when it ends on
  a pool, and `Result.n_grad_zeroed` counts them; the maximal coupling keeps each
  chain's law at draws where a log-density is infinite, and takes SciPy's Dirichlet
  laws; `check_coupling` swaps half the pairs, so that it catches a coupling that reads
  another row's state, and fails a NaN p-value; `check()` and the summary report a slow
  pair when every pair is censored; `hill` and `variance_share` check their values;
  every bounded number argument raises "must be in (low, high)", and `dual_averaging`,
  the TV and W1 bounds, the estimators, `SignedMeasure.mass`, `plotting.paths`,
  `Target(vectorize=)` and `pool` check what they were missing; a first Ctrl-C after the
  replicates, while the result is built, no longer loses the run, and a process that
  ignores Ctrl-C, or whose handler Python did not set, keeps its handler; the tuning
  pilot takes a third less time at d = 100, Gibbs sweeps less, and joint histograms take
  memory in proportion to the atoms (`SignedMeasure.mass` is sparse).
- Changes from a statistical audit against the source papers: `Result.estimate` builds
  its interval with the quantile of Student's t with $R - 1$ degrees of freedom instead
  of the normal quantile, whose 95% intervals covered 0.878 at $R = 5$ on an
  autoregressive chain, where Student's t covered 0.950; the stated
  guarantee of unbiasedness requires that the chain converge from the initial
  distribution; `Gibbs(coupling="crn")` uses common random numbers for pairs whose
  distance underflows to 0, which went to the maximal coupling; and the documentation
  corrects the meaning of the relative cost and the source of HMC's target acceptance
  rate, and describes `discrete_maximal` as serving kernels on finite state spaces, not
  a built-in kernel.
- Fixes of run-time defects: a vectorized log-density that returns shape `()` for one
  point, as SciPy's `multivariate_normal.logpdf` does, or any output that differs from
  the expected shape only by axes of length 1, is accepted; `Target` copies what its
  functions return, so one that reuses its output array no longer corrupts states; an
  `Independence` proposal whose `log_density` is NaN or `+inf` raises with the point
  instead of biasing the chain; the meeting-time quantiles of `Result.diagnostics()` and
  the summary cover every pair, `inf` (printed `> max_iter`) where one falls on a
  censored pair, and are observed meeting times (the inverted empirical CDF), so they
  change for runs without censoring too: the 90% quantile of meeting times 1 to 100 is
  90, not 90.1, and the median of an even number of pairs is the lower middle value;
  the summary's footer has `max_iter`; the survival curve of the meeting times ends at
  the fraction of pairs censored; `sample()` refuses fewer than 2 replicates before
  tuning; a `Result` with no replicates, Mixture weights whose sum overflows, a tuned
  mixture whose components differ in dimension, a kernel that drops its states'
  gradients, and `coupling="crn"` with conditionals whose `mean` and `cov` are methods
  raise a `ValueError` or `TypeError` that names the cause; MALA's coupled step no
  longer warns when its drift overflows.
- Input validation: an array argument that is not real numbers, such as a string (a
  numeric one included), `None`, a boolean, a complex number or a ragged list, raises
  `TypeError` naming the argument and the expected form, where NumPy's message named
  neither; real numbers held as objects, such as `Fraction`, `Decimal` or an object
  column of pandas, still convert to floats. `as_initial` raises its own `TypeError` for
  a ragged `init`. A flag that is not `True` or `False` (`vectorize`, `progress`,
  `strict`, `store`, `keep_paths`), and an `args` that is not a tuple or a list, raise
  `TypeError` naming the argument, and so do a target or a kernel that is not a
  `Target` or `Kernel` instance, such as a bare log-density passed to `engine.run` or a
  kernel class in a `Mixture`. `SignedMeasure` checks its arrays: `atoms` and `weights`
  that are not real numbers raise `TypeError`, and wrong shapes or a replicate index
  that is not an integer in range raise `ValueError`. Its attributes are read-only views
  of the arrays passed in, so they cannot be written through; the arrays passed in, such
  as those of the `Replicates` that `engine.run` returns, stay writable: the package
  never writes to them, and a user who does changes the measure. `unbiased_variance`
  and `unbiased_covariance` of a scalar raise `ValueError`.
  `CoupledSampler.meeting_times` refuses `max_iter=None`, so the pilot is always capped,
  and `slow_pair_threshold` checks that `lag` is at least 1 and `m` at least 0.
- A target function's output that is not numbers, such as `None` or a string, at one
  point among good ones raises `ValueError` naming the function and the point; an emcee
  `(log_prob, blobs)` tuple is told that derived quantities come from `Result.estimate`.
  A pointwise function whose outputs differ from point to point only by axes of length
  1, such as a float at some points and SciPy's `logpdf` of shape `(1,)` at others, is
  accepted, and one evaluated on no points returns an empty array. A complex output
  with an imaginary part of exactly 0 is its real part, and any other complex output
  raises; floating outputs narrower than float64, as JAX returns by default, emit one
  `InputWarning` per target that names their dtype.
- When `names` and the initial distribution differ in dimension, the error names the
  dimension the initial distribution draws; tuning chains whose covariance overflows,
  on a target flat in some direction or with scales near the float64 limit, raise an
  error that says so, instead of NumPy's overflow warnings and "cov must be finite".
- Tuning a MALA or HMC step size no longer raises when a pilot chain ends where the
  gradient is not finite but the log-density is: its dual-averaging chains take that
  gradient as 0, as runs do wherever chains move.
- The package ships a `py.typed` marker, so type checkers such as mypy and pyright read
  its annotations, and its metadata lists the supported Python versions (3.11 to 3.15),
  that it runs on any operating system, that it is typed, and a link to its issues.
- An `Independence` chain at a point where the target and the proposal both have zero
  density, such as a start outside the support of a bounded proposal, accepts any
  proposal of finite density, as the other Metropolis kernels do; it rejected every
  proposal, with a NumPy `RuntimeWarning`.
- Fixes and checks of a hardening pass, now recorded and tested: a `Mixture` never
  applies a component of weight 0, which the largest uniforms could select where
  rounding left the cumulative weights below 1. `Gibbs(coupling="crn")` refuses a
  conditional whose `mean` is not a vector of its block's size, where a scalar was
  added to every coordinate. `weighted_quantile` refuses values or weights that are not
  finite, `choose_k` meeting times that are not positive integers or `inf`,
  `SignedMeasure.mass` a cell that is not an integer, which it truncated, `hill` values
  that are not finite, and `testing.random_draws` the seeds that a run refuses;
  `plot_marginals` and `plot_corner` check `bins` and the number of replicates before
  opening a figure. `Result.merge` takes the kernel, initial distribution, tuning report
  and pilot meeting times of the other result where the first has none. The warning
  about disagreeing tuning chains also fires where an R-hat is NaN, which hid a
  disagreement in another coordinate: it names those coordinates, which the pilot never
  moved in, and reports the largest finite R-hat.
- The warning that the tuning chains disagree says that the chains may also not have
  converged, and the documentation of `TuningReport.rhat`, `tuning_disagreement` and
  `CoupledSampler.tune` says that chains that agree do not show that the target has one
  mode.
- The documentation states that an automatically tuned `Independence` proposal covers
  only the modes its tuning chains reached, so a run on well-separated modes can be
  confidently wrong without a warning; the failure-modes tutorial shows such a run and
  two checks that help.
- `Independence`'s `proposal` is typed as any object with `sample` and `log_density`,
  as it already behaved, so type checkers accept a custom proposal.
- The documentation states that the default range of `Result.histogram` biases every
  bin, and that `CouplingReport.passed()` fails a correct coupling with probability up
  to about 4 `alpha`.
- An `Independence` proposal drawn where the proposal's own `log_density` is `-inf`,
  such as a draw clipped onto the edge of an open support, is rejected; it was accepted
  and its chain never left it. The documentation states that `log_density` is the
  normalized density of what `sample` draws, positive wherever `sample` can land.
- The Student-$t$ log-density of `MultivariateT` is finite, without a NumPy warning, at
  points whose squared distance from the mean overflows float64, beyond about 1.34e154
  scale units; it was `-inf` there. This holds for scale matrices above about 1e-307 and
  a finite offset `x - mean`: for a smaller scale matrix the value stays `-inf`, and
  where the offset itself overflows it stays NaN, each with a NumPy warning.
- The documentation states that an `Independence` proposal must be positive wherever the
  target is: used alone, the kernel never reaches the part of the target's support that
  the proposal misses, so the estimates are biased without a warning, and a chain
  started where the proposal is 0 and the target is not never moves.
- The documentation states what an `Independence` proposal with lighter tails than the
  target does: with unbounded weights $\pi/q$, a run usually gives too little mass to
  the region where the weights are large, with error bars that are too small and often
  without a warning; the kernels guide has a measured example.
- The docstrings of `sample()` and `CoupledSampler.run` say that their estimates are
  unbiased unless `max_iter` censors a pair, which biases them, or a Ctrl-C interrupts
  the run, which leaves them consistent.
- The warnings about late pairs, a replicate dominating the variance, heavy-tailed
  meeting times and slow pairs end with the remedy in a few words, as the warning about
  censored pairs did. A slow pair seen as it happens says that the run waits for it,
  and for a replicate that a first Ctrl-C lets it finish. The warning about tuning
  chains that disagree advises an initial distribution that covers every mode only
  unless the one given already does.
- `Result.diagnostics()` reports `variance_share_replicate`, the index of the replicate
  that holds the largest share of the variance of an estimate, and the variance warning
  names it.
- A replicate or pilot pair that starts where the target density is zero is reported
  as soon as it starts, in this process, instead of when the run ends; the slow-pair
  warning of such a pair says so, the censoring warning no longer suggests running
  without a cap after such starts, and a pilot that cannot choose `k` names them instead
  of advising a wider initial distribution. `engine.Status` carries the number of
  replicates started at zero density so far, the first of them, and whether the
  slowest pair is one of them.
- The memory warning compares the estimated peak of a run and its first estimates, about
  three times the stored atoms and distances, with its threshold of 1 GB, and states both
  figures, with twice the peak as its upper end in one or two dimensions, where few
  repeated atoms can make the estimates take up to about five times the stored size;
  `telescopic.engine.stored_bytes` computes the stored size, and refuses steps that are
  not non-negative integers.
- `progress=True` also shows the progress of tuning: the steps of the adaptive pilot
  chains and, for MALA and HMC, of the step size's dual averaging.
- Changes to `Result` ruled before the first release: the last bin of a histogram is
  closed, as in `numpy.histogram`, so an atom on the upper edge of `range` is counted,
  and the last edge of the default range is no longer nudged up; the TV bound and its
  value at `k` count a censored pair as meeting at its cap plus 1, since its chains were
  compared at the cap, which raises the bound by $1/R$ per censored pair at each
  $t \le$ cap $- L$ where $L$ divides cap $- L - t$, and
  `telescopic.diagnostics.tv_bound` asks for a censored pair as its cap plus 1;
  `Result.interval` is renamed `Result.credible_interval`, with no alias, and
  `Estimate.interval` is an array of shape `(2,)` or `(2, p)`, lower ends first,
  instead of a tuple of two arrays;
  `Result.quantile` and `Result.credible_interval` take a function `h`, as `estimate`
  does, for the quantiles of a derived quantity; a `Result` prints as one line of its
  settings, such as `Result(R=100, d=2, k=63, m=630, lag=63, censored=0,
  interrupted=False)`.
- A pair censored at a `max_iter` equal to the slow-pair threshold now raises the
  slow-pair warning, since its chains were compared, unmet, at the threshold.
- The summary shows as an HTML table in notebooks, its footer a list with one warning
  per item. Its text keeps the lines below the table within 80 characters, with the
  kernel on a line of its own and its floating-point parameters to 4 significant
  figures, and sizes each column of the table to its entries.
- `plotting.paths` draws $X$ in magenta and $Y$ in teal, the documentation's chain
  colors, and a pair in the dark ink of met chains from its meeting time on; the colors
  are `plotting.CHAIN_X`, `CHAIN_Y` and `MET`. A legend says which line is which, and in
  one dimension the time axis has integer ticks.
- The pool of worker processes that telescopic starts, with the semaphores of its stop
  event, is freed when the caller drops an error raised in a worker. A reference cycle
  held them until a later garbage collection, which on Python 3.11 and 3.12 can warn
  that the resource tracker was called reentrantly and leave the semaphores to leak.
- Each public module of the package lists in `__all__` the names that the API reference
  documents, so a star import from `telescopic.diagnostics`, for example, and `help()` on
  it no longer list the helpers that the module imports.
- A sampler built with `grad` and a kernel that evaluates it compares `grad` with
  central differences of `log_prob` along 8 lines through draws of the initial
  distribution, before tuning, and emits an `InputWarning` with the size of the
  disagreement in nats where they disagree beyond rounding, or when it could compare no
  line. It costs 40 evaluations of `log_prob` and 8 of `grad` at any dimension for
  float64 values, and 48 draws of the initial distribution, draws from a random stream
  of its own and changes no result; `check_grad=False` on
  `CoupledSampler` and `sample` turns it off, and `CoupledSampler.gradient_check`, a
  `diagnostics.GradientCheck`, records what it compared. An error raised inside the
  check keeps its type and message, with a note that it was raised in the check.
- `Gibbs(coupling="crn")` draws a block of several coordinates, or of one without `ppf`,
  from each chain's own conditional with generators in the same state, so each chain
  keeps its conditional's law; a pair of chains takes a common uniform or these
  generators by its own two conditionals, so a replicate's output does not depend on the
  other replicates in its batch. It took such a conditional to be the normal with its
  `mean` and `cov`, which gave a conditional that is not normal the wrong law in the
  coupled step only, and it refused conditionals without a `mean` and a `cov` that are
  values, such as SciPy's frozen `dirichlet`, whose `mean` and `cov` are methods,
  `wishart`, which has no `cov`, and `multivariate_t`, which has neither. Blocks of one
  coordinate with `ppf`, and the maximal coupling, give the same numbers as before.
- `Draws.child(row)` returns a new child of a row's generator, which does not move the
  row's stream: two draws from it can read the same numbers, as the Gibbs kernel's common
  random numbers do.
- A request of `Draws` for no rows returns an empty array. It raised `ValueError` when it
  asked for at least two numbers per row more than the buffer held, as for two normals
  per row of replayed draws without normals.
- `import telescopic` no longer imports `scipy.stats`, `scipy.optimize` and
  `scipy.integrate`: the functions that use them import them at their first call. That
  shortens the import, and with it the start of each worker of a process pool; the first
  call that needs `scipy.stats`, such as the end of the first run of `sample`, pays its
  import once.
- `Result.cov` weights the atoms in blocks of 1 MiB, so its peak memory is about the
  atoms' size plus a block rather than twice the atoms; `cov` can differ from before in
  the last bits.
- `Result.quantile` gathers one coordinate at a time instead of copying the atoms, which
  lowers its peak memory.
- `Result.var`, `Result.std` and the diagnostics share each replicate's moments,
  computed once, so `summary()` centers the atoms once; `var` and `std` can differ from
  before in the last bits.
- `SignedMeasure` requires `replicate` in nondecreasing order, each replicate's atoms
  consecutive, and raises `ValueError` otherwise; its sums share the weights and hold
  8 bytes per atom fewer.
- With a vectorized target (`vectorize=True`), RandomWalk, MALA, HMC and the tuning
  pilot compute their products of a batch of chains with a matrix by BLAS: whole runs of
  256 replicates on a Gaussian of 200 dimensions took 2.1 times less processor time with
  RandomWalk, whose pairs all ran to the iteration cap, and 3.4 times less with MALA.
  Such a run repeats bitwise only with the same seed and settings, `pool` and
  `n_workers`, BLAS build and number of BLAS threads; runs of other lane counts, pools,
  merges and later calls of a sampler, and runs on workers, which use one BLAS thread,
  against runs in a process with more, can differ by rounding, which can move a
  replicate's meeting time and its whole output, while the estimates agree within their
  error bars. Seeded outputs of vectorized runs change likewise. A target evaluated
  point by point keeps every bitwise guarantee. `couplings.reflection_maximal` takes
  `blas=True` to do the same.
- The guide "Check and set a step size" names a resonance of HMC's leapfrog on
  Gaussians, on which a given `n_steps`, or a given `cov` that does not whiten the
  target, can put the automatic step. With MALA or HMC, the error raised when `k` cannot
  be chosen carries a note that points to it.
- The documentation's pages are split by task. Start holds Install, the Quickstart and
  Coming from emcee; Unbiased on average is the second tutorial; the how-to guides write
  the log-density, choose an initial distribution, a kernel and a step size, set the run's
  arguments, estimate, plot, save, load and merge, read meeting times, run in parallel and
  write a kernel; Reference adds the fields of a result and the warnings. The addresses
  `getting-started`, `tutorials/beside-emcee`, `guide/targets`, `guide/results` and
  `guide/diagnostics` are gone.
- `couplings.reflection_maximal`, which the coupled steps of RandomWalk, MALA and HMC
  call, reflects with one product with the covariance factor instead of two: a call at
  256 rows took 1.08 to 1.38 times less time at d = 2 to 200. Seeded outputs of their
  coupled runs change: in the last bits for RandomWalk and MALA used alone; for kernels
  whose chains come close without meeting, such as HMC or a `Mixture` with a contracting
  component, a meeting time can move, and with it a replicate's estimates (one of 100 on
  a logistic regression of 6 dimensions with HMC, which moved the estimates by up to
  0.075 standard errors).
- `MALA` computes its step and coupled step in whitened coordinates: 3 products of a
  row with a $d \times d$ factor per step and 7 per coupled step, against 7 and 14. A
  single step took 0.66 to 0.76 times the time at d = 100 and 200, and a coupled step
  0.80 to 0.82 times at d = 100 (medians of four benchmark runs on a dense vectorized
  target, with the form as prototyped). It is the same kernel; seeded outputs of runs
  that step MALA change in the last bits.
- The tuning pilot draws its normals and uniforms 256 steps at a time instead of at
  every step, which made it 1.24 to 1.32 times faster on a Gaussian target at d = 2 to
  100. Its outputs are unchanged.
- `Independence` computes its weights with fewer operations per step, and a Student-t
  `MultivariateT` its normalizing constant once: a one-lane run with a Student-t
  proposal at d = 20 took 1.12 times less time. Its outputs are unchanged.
- `CoupledSampler.resume(result)` continues a returned, loaded or merged result on a new
  sampler: it adopts the result's seed, replicates, settings, pilot and tuning, and the
  next `run` completes the replicates an interrupted run never started, then continues
  above the highest index requested. It checks the dimension, the kernel's class, an
  `rng` given to the constructor and the names; the target and the kernel's parameters
  are the caller's to match. An untuned kernel, as a loaded result needs, is tuned again
  from the result's seed; an `InputWarning` says when the kernel's tuning or description
  differs from the result's.
- Proposal and coupling objects: `kernels.MetropolisHastings(proposal, coupling)` with
  the proposals `RandomWalkProposal`, `LangevinProposal` and `IndependenceProposal`, and
  the couplings `couplings.Reflection`, `couplings.Maximal` and `couplings.Shared`, each
  of which supports only the proposals it was published for. `RandomWalk`, `MALA` and
  `Independence` are built on them, and their seeded outputs are unchanged bit for bit.
  `RandomWalk(coupling=...)` also takes `Reflection()` or `Maximal()`;
  `RandomWalk.coupling` and `Independence.proposal` can no longer be reassigned.
- An error during the replicates, such as one of the log-density, no longer loses the
  replicates finished before it. The same exception is raised again with one note:
  `engine.run` attaches the finished replicates as `telescopic_replicates`, and
  `CoupledSampler.run` and `sample` keep them and attach the result of all the sampler's
  replicates as `telescopic_result`, whose next `run`, or `resume`, starts with the
  missing ones. The replicates in progress at the error are listed in
  `Replicates.abandoned` and `Result.abandoned`, saved, and named in the summary until
  they run. On a pool the error stops the run as a first Ctrl-C does; an error that does
  not unpickle becomes a `RuntimeError` with the worker's traceback in a note, instead of
  breaking a process pool.
- Each atom records the time at which it was recorded, `Replicates.time` (int32), and
  `Result.chains()` returns the averaged states of each kept replicate as chains of equal
  length. `max_iter` must be at most 2**31 - 1 where times are stored, and a stored atom
  costs 8d + 20 bytes instead of 8d + 16.
- `keep` on `CoupledSampler.run`, `sample` and `engine.run` (and `engine.Settings.keep`)
  stores the signed measures and distances of the replicates of index below `keep` only,
  and the moments of every replicate (`Replicates.center` and `Replicates.moments`), from
  which `mean`, `var`, `std` and `cov` (up to 100 dimensions) come; `estimate`, the
  quantiles, the histograms and their plots come from the `n_kept` kept replicates
  (`Result.keep`, `Result.n_kept`), and the summary and the repr name the replicates
  behind each figure. The memory warning counts the moments.
- The variance share of the diagnostics takes each replicate's second moments about the
  mean estimate instead of the atoms' mean, which changes its values.
- Results are saved in schema 2, which names the estimator family and keeps `keep`, the
  times, the moments and the abandoned replicates; files of schema 1 load with their
  times rebuilt.
- `HMC` jitters its step by default: each trajectory takes a step drawn uniformly
  between `(1 - jitter)` and `(1 + jitter)` times `step_size`, the same for both chains
  of a pair, with `n_steps` fixed by `step_size`, and the automatic step is tuned on the
  jittered kernel. The new argument `jitter`, in [0, 1), is 0.5 by default and shown in
  the repr; `jitter=0` gives the fixed-step kernel, bit for bit. The jitter keeps the
  step off the leapfrog's resonance: on the Gaussians where a given `n_steps`, or a
  given `cov` that does not whiten the target, left pilot pairs unmet, `k` is chosen. Every seeded output of
  `HMC` with the default changes, and the note on the error raised when `k` cannot be
  chosen now suggests the default jitter for HMC with `jitter=0`.
- `State` has a fourth field, `extra`: arrays of one kernel, batch first, such as a
  momentum or a cache. A kernel names its Markov fields in `Kernel.markov_fields` and
  gives their initial values in `Kernel.init_extra`, which `init_state` attaches and
  checks for every Markov field; a pair
  meets when its positions and Markov fields are equal. Kernels, `State.take`,
  `State.put` and `Mixture` carry unchanged the fields they do not know, a `Mixture`
  names component `i`'s fields `"i/<name>"` and refuses components with Markov fields,
  and `check_coupling` counts meetings on the Markov state and counts as stale the chains
  whose caches change their next step. No shipped kernel has such fields, and every
  seeded output is unchanged. Code that unpacks `dataclasses.astuple(state)` into three
  names breaks.
- `Independence` keeps log q at each chain's position in the chain's state, with the point
  at which it was computed, and uses it only where that point is the position bit for
  bit: a step and a coupled step evaluate the proposal's `log_density` at the proposals
  and at the states without a valid value, usually n rows where they were 2n and 3n, and
  with `MultivariateT` every seeded output is unchanged. `Proposal.init_extra` gives a
  proposal's own fields.
- A `wide` warning when the tuning pilot shows an initial distribution more than 30
  times wider than the target: `TuningReport.initial_width` and `PilotRun.initial_width`
  hold the geometric mean over coordinates of the ratio of the standard deviations of
  the initial draws to the pilot's, and `diagnostics.wide_start` words the warning. The
  guide on settings shows `k` from a confirmation round of pilot pairs at lag `k`.

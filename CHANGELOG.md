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
- Documentation: Getting started, a user guide, six tutorials executed when the site is
  built, a theory page, the API reference and a citing page.
- The documentation site has its own theme and identity: the Focus mark, two chain colors
  that mean the two chains, Red Hat type served with the site, light and dark themes, a
  landing page with a coupled pair drawn by telescopic itself, and navigation by section
  (Start, Tutorials, How-to guides, Theory, Reference, Project). Every plot of the
  executed pages shares one style. The design records are no longer built into the site.
- Each tutorial has one title in the sidebar, on its page and in the pager, and the
  comparison with emcee moved to `tutorials/beside-emcee`. The tutorials are ordered as a
  learning path, with the failure modes last, and the pager is their only "next" link.
- Warnings on the executed pages print as their class and message, without the path of
  the notebook's temporary file, and wrap within the column.
- Every figure on the executed pages has alternative text and a caption.
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

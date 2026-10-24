# API and user-experience design of MCMC and sampling packages: lessons from emcee and dynesty, and candidate API shapes for `telescopic` (unbiased MCMC with coupled chains)

Scope, date and method:
- **What this note covers.** It covers what is specific to MCMC sampler packages and to running coupled chains. It builds on the sibling note `/home/user/pastry/research_notes/Population inference software landscape/api_ux_validation.md` (2026-10-22) and does not repeat that note's general material on packaging, documentation, validation, scikit-learn conventions, SPEC 7, or the basics of nautilus, UltraNest, ArviZ and PyMC. Where this note relies on that note, it says so.
- **Research date:** 2026-10-23.
- **Versions and release dates.** These come from the PyPI JSON API (`https://pypi.org/pypi/<name>/json`), queried that day.
- **Source reading.** Package source was read from wheels or sdists downloaded from PyPI into a scratch directory. Signatures were extracted with Python's `ast` module, so no package code was executed.
- **Repository clones.** The emcee, dynesty, Pigeons.jl and ColCarroll/couplings repositories were shallow-cloned read-only for their docs and changelogs.
- **Papers.** The PDFs of Hogg & Foreman-Mackey (2018) and Jacob, O'Leary & Atchadé (2020) were converted to text with `pdftotext`.
- **GitHub issues.** Issue bodies were read through a server-side web fetch. The GitHub API's issue reader was restricted to the user's own repositories in this session, but its issue search worked.

## Q1. emcee in detail: API surface, documented problems, walkers versus independent chains, and how users check convergence

### Takeaway
emcee 3.1.6, released 2024-04-19 with nothing since, is a small, stable API:
- a sampler object around a plain `log_prob(x, *args)`;
- an `(nwalkers, ndim)` array of starting positions;
- `run_mcmc`, plus a `sample()` generator;
- a backend with `get_chain(discard, thin, flat)` accessors and an optional HDF5 file;
- a weighted mixture of "move" objects.

Its documented weaknesses are well established:
- The default stretch move degrades badly above about 50 dimensions while *appearing* converged (Huijser, Goodman & Brewer).
- Initialisation, burn-in and convergence are left to the user, with one-line advice ("small ball").
- Seeding uses hidden global state.
- Parallelism depends on pickling.
- Walkers are routinely mistaken for independent chains, even by downstream tools: ArviZ's `from_emcee` puts walkers on the `chain` axis.

The practical convergence check in the emcee world is the integrated autocorrelation time τ, with a run length of at least about 50τ. Some wrapper packages, such as RadVel, add Gelman–Rubin across several *independent* ensembles.

### Cited Findings

#### Release status
- **Release status:**
  - emcee 3.1.6 was uploaded 2024-04-19. It was preceded by 3.1.5 (2024-04-16) and 3.1.4 (2023-01-27), and nothing has been released since.
  - The first release was 1.0.0 on 2012-02-12.
  - It is classified "5 - Production/Stable" and declares no `requires_python`.

  ([PyPI emcee JSON](https://pypi.org/pypi/emcee/json); [PyPI emcee 3.1.6](https://pypi.org/project/emcee/3.1.6/))

#### Constructor and target specification
- **Full constructor signature in 3.1.6** ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)):

  `EnsembleSampler(nwalkers, ndim, log_prob_fn, pool=None, moves=None, args=None, kwargs=None, backend=None, vectorize=False, blobs_dtype=None, parameter_names=None, a=None, postargs=None, threads=None, live_dangerously=None, runtime_sortingfn=None)`

  The last five arguments are deprecated. Passing them triggers messages such as "The 'a' argument is deprecated, use 'moves' instead" and "The 'threads' argument is deprecated".
- **`vectorize`:** "If ``True``, ``log_prob_fn`` is expected to accept a list of position vectors instead of just one. Note that ``pool`` will be ignored if this is ``True``." So vectorisation and pools are mutually exclusive, and emcee ignores the pool silently ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)).
- **`parameter_names`** ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)):
  - It may be a list of names or a dict mapping names to an index or a list of indices, which allows grouped vector parameters.
  - Positions are converted with `ndarray_to_list_of_dicts`, so `log_prob` receives a dict.
  - Validation uses `assert`, e.g. "name all parameters or set `parameter_names` to `None`".
- **Blobs** ([emcee "Blobs"](https://emcee.readthedocs.io/en/stable/user/blobs/)):
  - "Anything that your ``log_prob`` function returns in addition to the log probability is assumed to be a blob".
  - "If ``log_prob`` returns ``-np.inf`` for the log probability, the blobs are not inspected or tracked so can be anything (but the correct number of arguments must still be returned)".
  - "If you don't provide ``blobs_dtype``, the dtype of the extra args is automatically guessed the first time ``log_prob`` is called".
  - Ragged blobs trigger a deprecation warning asking for `blobs_dtype=np.dtype('object')` ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)).
- **Parameter limits** are expressed by returning `-numpy.inf` outside the allowed volume, "corresponding to the logarithm of 0 prior probability". emcee has no prior or bounds object ([emcee FAQ](https://emcee.readthedocs.io/en/stable/user/faq/)).

#### Running: `run_mcmc`, the `sample` generator and `State`
- **`sample` signature:** `sample(initial_state, log_prob0=None, rstate0=None, blobs0=None, iterations=1, tune=False, skip_initial_state_check=False, thin_by=1, thin=None, store=True, progress=False, progress_kwargs=None)`.
  - `log_prob0`, `rstate0` and `blobs0` are deprecated in favour of passing a `State`.
  - `thin` is deprecated in favour of `thin_by`.
  - `iterations=None` requires `store=False` ("'store' must be False when 'iterations' is None").

  ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py))
- **`run_mcmc`:** `run_mcmc(initial_state, nsteps, **kwargs)` forwards to `sample` ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)). The sibling note covers `run_mcmc(None, n)` resumption.
- **`State` signature:** `State(coords, log_prob=None, blobs=None, random_state=None, copy=False)` ([emcee state.py](https://github.com/dfm/emcee/blob/main/src/emcee/state.py)).

#### Outputs and accessors
- **Backend accessors** ([emcee backend.py](https://github.com/dfm/emcee/blob/main/src/emcee/backends/backend.py)):
  - All go through `Backend.get_value(name, flat=False, thin=1, discard=0)`.
  - `get_autocorr_time(discard=0, thin=1, **kwargs)` returns `thin * autocorr.integrated_time(x, **kwargs)`.
  - The sampler's `acceptance_fraction` is `backend.accepted / iteration`, one value per walker.
- **`integrated_time(x, c=5, tol=50, quiet=False, has_walkers=True)`** ([emcee autocorr.py](https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py)):
  - It averages the per-walker autocorrelation functions (`f += function_1d(x[:, k, d]) ... f /= n_w`).
  - It raises `AutocorrError` unless `quiet=True`, with the message "The chain is shorter than {tol} times the integrated autocorrelation time for {n} parameter(s). Use this estimate with caution and run a longer chain!".
  - The exception carries the estimate in its `tau` attribute.
- **Older accessors are deprecated:** `flatchain`, `lnprobability` and similar now point to `get_chain()` and friends through a `@deprecated` decorator ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)).

#### Backends
- **Signatures:** `HDFBackend(filename, name='mcmc', read_only=False, dtype=None, compression=None, compression_opts=None)` and `TempHDFBackend(dtype=None, compression=None, compression_opts=None)`. `Backend.reset(nwalkers, ndim)` clears the stored state ([emcee hdf.py](https://github.com/dfm/emcee/blob/main/src/emcee/backends/hdf.py); [emcee backend.py](https://github.com/dfm/emcee/blob/main/src/emcee/backends/backend.py)).
- The resume and read-only workflow is in the sibling note.

#### Moves interface
- **Why moves exist** ([emcee "Moves"](https://emcee.readthedocs.io/en/stable/user/moves/)):
  - "starting with version 3, emcee nows allows proposals generated from a mixture of 'moves'. This can be used to get a more efficient sampler for models where the stretch move is not well suited, such as high dimensional or multi-modal probability surfaces."
  - "During sampling, at each step, a move is randomly selected from the mixture and used as the proposal. The default move is still the :class:`moves.StretchMove`".
- **Available moves:** `Move`, `MHMove`, `GaussianMove`, `RedBlueMove`, `StretchMove(a=2.0)`, `WalkMove`, `KDEMove(bw_method=None)`, `DEMove(sigma=1e-05, gamma0=None)` and `DESnookerMove`.
- **How a mixture is passed:** `moves` may be a single move, a list, or a list of `(move, weight)` tuples. The weights are normalised, and each step draws `self._random.choice(self._moves, p=self._weights)` ([emcee moves/__init__.py](https://github.com/dfm/emcee/blob/main/src/emcee/moves/__init__.py); [emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)).
- **Extension points** ([emcee move.py](https://github.com/dfm/emcee/blob/main/src/emcee/moves/move.py); [emcee red_blue.py](https://github.com/dfm/emcee/blob/main/src/emcee/moves/red_blue.py); [emcee mh.py](https://github.com/dfm/emcee/blob/main/src/emcee/moves/mh.py)):
  - The base `Move` has `tune(state, accepted)` and `update(old_state, new_state, accepted, subset=None)`.
  - Ensemble moves subclass `RedBlueMove(nsplits=2, randomize_split=True, live_dangerously=False)` and implement `get_proposal(sample, complement, random)`.
  - Single-walker Metropolis–Hastings moves wrap a function: `MHMove(proposal_function, ndim=None)`.
- **Walker-count rule:** `RedBlueMove`: "an update will fail with a ``RuntimeError`` if the number of walkers is smaller than twice the dimension of the problem because the walkers would then be stuck on a low dimensional subspace". `live_dangerously=True` suppresses this ([emcee red_blue.py](https://github.com/dfm/emcee/blob/main/src/emcee/moves/red_blue.py)).

#### Built-in checks and error messages
- **Errors raised by `sample` and `compute_log_prob`** ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)):
  - `ValueError(f"incompatible input dimensions {state_shape}")`.
  - "Initial state has a large condition number. Make sure that your walkers are linearly independent for the best performance". `walkers_independent` checks that the condition number of the centred, normalised coordinates is ≤ 1e8.
  - "The initial log_prob was NaN".
  - "At least one parameter value was infinite" and "At least one parameter value was NaN".
  - "Probability function returned NaN".
- **There is no check for walkers whose initial log-probability is −∞.** The sibling note lists issue #55, "FAIL if initialized with ANY walkers at -Inf".

#### Random state
- **emcee uses a hidden, global-state-derived `RandomState`** ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)):
  - The sampler creates `np.random.mtrand.RandomState()` and seeds it from `np.random.get_state()`.
  - A `random_state` property can be set from a state tuple. Its docstring warns: "You can try to set this property but be warned that if you do this and it fails, it will do so silently". The setter itself says it will "fail silently if it doesn't work. Don't say I didn't warn you...".
- **Determinism issue #313 (2019, 8 comments)** ([emcee #313](https://github.com/dfm/emcee/issues/313)):
  - A user could not get repeated runs to agree under emcee 2.2.1, even after seeding `random`, `numpy.random` and a `RandomState`.
  - The workaround was assigning `sampler.random_state = s.get_state()`.
  - dfm closed it as an old-version problem, noting that direct `np.random` use motivated v3.
- **Pool-dependent results, issue #351 (2020; still open)** ([emcee #351](https://github.com/dfm/emcee/issues/351)):
  - A user reported different results with and without `Pool()`.
  - dfm asked "Are you setting your seed to be the same?" and said emcee ≥ 3 gives identical results when seeded.
  - The reporter's follow-up question, how to set seeds across processes, is unanswered.

#### Documented problems: high dimensions (Huijser, Goodman & Brewer)
- **The paper** ([arXiv:1509.02230](https://arxiv.org/abs/1509.02230); [ar5iv rendering](https://ar5iv.labs.arxiv.org/html/1509.02230)):
  - It was submitted 2015-09-07 and revised 2017-08-21.
  - The two pages disagree on the title. The arXiv abstract page gives "Properties of the Affine Invariant Ensemble Sampler in high dimensions". The ar5iv rendering gives "Properties of the Affine Invariant Ensemble Sampler's 'stretch move' in high dimensions".
  - Test problems: "an n-dimensional correlated Gaussian toy problem with a known mean and covariance structure, and a multivariate version of the Rosenbrock problem".
  - The problem becomes severe as n increases beyond 50.
  - Conclusion: "the stretch move should not be relied upon (in isolation) in moderate to high dimensions".
- **Failure mode** ([ar5iv rendering](https://ar5iv.labs.arxiv.org/html/1509.02230)):
  - The burn-in looks short, but the means and variances are wrong.
  - "The output can resemble a successful run". High dimensions make the sampler "appear to have converged even when it has not".
  - Walkers' first coordinates "seem to be confined to much narrower regions", and their relative order "changes much less frequently".
- **Critique cited by Hogg & Foreman-Mackey:** they point to Carpenter's 2017 blog post "Ensemble Methods are Doomed to Fail in High Dimensions" in footnote 70 ([Hogg & Foreman-Mackey 2018, arXiv:1710.06068](https://arxiv.org/abs/1710.06068)). The post itself returned HTTP 403 and was not read.
- **Maintainer view on dimensionality, issue #378 (2021–2023)** ([emcee #378](https://github.com/dfm/emcee/issues/378)):
  - A user with 24 parameters and 100 walkers hit the condition-number error.
  - dfm explained it as un-jittered or too-tightly-jittered initial positions, and suggested `skip_initial_state_check=True` if the initialisation is known to be fine.
  - For 30 parameters he said "There is no formal limit on the number of dimensions", but efficiency drops. He advised checking the autocorrelation time rather than the acceptance rate (20% being fine).

#### Documented problems: initialisation, burn-in and convergence (Hogg & Foreman-Mackey 2018)
Source for all bullets in this subsection: [Hogg & Foreman-Mackey 2018, ApJS, DOI 10.3847/1538-4365/aab76e; arXiv:1710.06068](https://arxiv.org/abs/1710.06068).
- **Initialisation (§7):**
  - "it makes sense to initialize the walker or walkers at sensible parameter estimates, found by optimizing a likelihood or a posterior pdf in advance of sampling."
  - But "In extremely high dimensions ... this can be a bad idea, since the optimal parameters are not necessarily near typical posterior samples".
- **Burn-in for ensembles (§7):** "Ensemble methods (like emcee ...) require a burn-in phase and discard, because the whole point is that the ensemble must grow (or shrink) to fill the posterior pdf volume, and you can't initialize the ensemble sensibly if you don't know that volume a priori".
- **Multimodality (§7):**
  - "the best diagnosis is to start many chains in parallel with different initializations, and check that they lead to identical (or very similar) posterior inferences"
  - "There is no trivial way to combine together the samplings you get of different modes!"
- **Convergence (§5):**
  - "we don't and can't require any kind of absolute convergence; indeed, it would be impossible even to test for it."
  - The heuristics: walkers traversing the space many times; first half versus second half of the chain; "different walkers initialized differently and run with different random number seeds create posterior inferences that are substantially the same".
  - "The premier tool for making this heuristic precise, however, is to look at the 'integrated autocorrelation time'". This is "notoriously difficult: It is a two-point statistic".
  - Gelman–Rubin "will only deliver success when the chains are much longer than an autocorrelation time".
  - Footnote 26: "there is never (to our knowledge) a condition in which burn-in time is the dominant consideration in choosing an MCMC method".
- **Ensemble specifics (footnote 69):** "it is better to initialize the ensemble in a ball that is smaller than the full posterior pdf width, not larger. For another example, the ensemble produces a set of chains that can be used to compute the Gelman–Rubin diagnostic ..., but this is not a conservative test if the initialization was made in a small ball."
- **Sampling is for integrals (§2, §8):**
  - "In a deep sense, the only thing a sampling is good for is computing integrals".
  - Results should be "based on integrals computed with the sampling. This includes expectations, medians, quantiles, one-dimensional histograms, and multi-dimensional histograms. This does not include the 'best' sample or a mode or optimum."
- **Troubleshooting (§9):**
  - Recommends functional tests ("sample a known distribution") and sampling the prior as an end-to-end test.
  - Asks: "when you re-call the likelihood function with the same parameters, do you get exactly the same answer?" A stochastic likelihood should fix its random numbers, or be replaced by deterministic integrals.

#### Most-discussed GitHub issues (bodies read for the starred ones; the rest are titles only)
- **Top issues by comment count** ([emcee issues sorted by comments](https://github.com/dfm/emcee/issues?q=is%3Aissue+sort%3Acomments-desc)):
  - #299 "ValueError: algorithm must be 'MT19937'" (17 comments; random state).
  - #236 "PTsampler is removed" (12).
  - #14 "emcee and MPI" (12).
  - #150 "Discrete Variables in emcee" (11).
  - #312 "emcee 3.0 breaks extended-precision coordinates" (11).
  - #121 "stochastic calculations inside emcee" (10).
  - #369 "sample with infinite iterations" (9).
  - #148 "problem with mulltiprocessing" (8).
  - #313* "How can emcee be run deterministically?" (8).
- **Parallelism complaints recur:**
  - #447 "Parallelization of emcee is working not as fast as expected".
  - #452 "Multiprocessing Pool scaling not as expected".
  - #199 "Pool slower than serial, pickling with every call?".
  - #351* "different results when using multiprocessing Pool() (sometimes)".
  - #365 "Multiprocessing Tutorial Example Not Working".

  ([GitHub issue search, dfm/emcee](https://github.com/dfm/emcee/issues?q=is%3Aissue+multiprocessing))
- **What the parallel tutorial teaches** ([emcee "Parallelization"](https://emcee.readthedocs.io/en/stable/tutorials/parallel/)):
  - It sets `os.environ["OMP_NUM_THREADS"] = "1"`, because some NumPy builds parallelise internally.
  - It defines a pool as "any Python object with a `map` method that can be used to apply a function to a list of numpy arrays".
  - It sends users to schwimmbad for MPI, because "emcee no longer includes its own" MPI pool.
  - It notes "There is often more overhead introduced by MPI than multiprocessing".

#### Walkers versus independent chains
- **The FAQ:** "Walkers are the members of the ensemble. They are almost like separate Metropolis-Hastings chains but, of course, the proposal distribution for a given walker depends on the positions of all the other walkers in the ensemble" ([emcee FAQ](https://emcee.readthedocs.io/en/stable/user/faq/)).
- **The autocorrelation tutorial:** "you should not compute the G–R statistic using multiple chains in the same emcee ensemble because the chains are not independent!" ([emcee autocorr tutorial](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/)).
- **The maintainer in issue #183 (2016–2017)** ([emcee #183](https://github.com/dfm/emcee/issues/183)):
  - "Be careful with GR because the walkers in the ensemble are not independent".
  - Gelman–Rubin "requires parallel independent chains", which means separate ensembles.
  - Recommended instead: "monitoring the autocorrelation time and setting a target number of independent samples".
  - Hogg added that τ estimates are unreliable until the run spans many τ, a chicken-and-egg problem.
- **Hogg & Foreman-Mackey say the opposite in §10:** "Because ensemble methods are updating independent MCMC chains, they are often easy to parallelize" ([arXiv:1710.06068](https://arxiv.org/abs/1710.06068)). The same authors' tutorial and issue answers say the walkers are *not* independent ([emcee autocorr tutorial](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/)). The wording is inconsistent across emcee's own documentation.
- **Downstream tooling encourages the discouraged practice** ([arviz-base 1.3.1 on PyPI, file `arviz_base/io_emcee.py`](https://pypi.org/project/arviz-base/1.3.1/)):
  - ArviZ's `from_emcee` takes `sampler.get_chain()`, with shape `(nsteps, nwalkers, ndim)`, under `sample_dims ["draw", "chain"]`.
  - So each walker becomes an ArviZ "chain", and `rhat`/`summary` compute R̂ across walkers by default.

#### How users check convergence in practice
- **emcee's recommendation** ([emcee autocorr tutorial](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/)):
  - Use the integrated autocorrelation time, estimated per walker and then averaged.
  - "chains longer than about 50τ are often sufficient"
  - "you probably shouldn't trust any estimate of τ unless you have more than F×τ samples for some F ≥ 50"
  - The Goodman & Weare estimator can be "dangerously over-confident".
  - "some discussion of autocorrelation analysis is (or should be!) a necessary part of any publication using MCMC".
- **Users write their own stopping loop** over `sample()` with `get_autocorr_time(tol=0)`. The sibling note quotes the code from emcee's "Saving & monitoring progress" tutorial.
- **RadVel, an exoplanet radial-velocity package built on emcee, encodes a multi-criterion rule** ([RadVel MCMC docs](https://radvel.readthedocs.io/en/latest/mcmc.html)):
  - Signature: `radvel.mcmc.mcmc(post, nwalkers=50, nrun=10000, ensembles=8, checkinterval=50, minAfactor=40, maxArchange=0.03, burnAfactor=25, burnGR=1.03, maxGR=1.01, minTz=1000, minsteps=1000, minpercent=5, ...)`.
  - It runs 8 ensembles "in parallel on separate CPUs" and computes Gelman–Rubin across them.
  - Its documented criteria are "Gelman-Rubin statistic of <= 1.03", "min autocorrelation time factor >= 75", "max relative change in autocorrelation time <= .01" and ">= 1000 independent draws".
  - The documented thresholds differ from the function defaults (`maxGR=1.01`, `minAfactor=40`).
- **zeus has the same pattern built in:** `ParallelSplitRCallback(..., chainmanager=...)` computes split-R̂ across independent ensembles run through `ChainManager` ([zeus-mcmc 2.5.4 on PyPI, `zeus/callbacks.py`, `zeus/parallel.py`](https://pypi.org/project/zeus-mcmc/2.5.4/)).

### Inferences
- **The emcee mental model that telescopic users bring:**
  - A `log_prob(theta, *args)` returning a scalar, with `-inf` for out-of-bounds values and optional extra return values (blobs).
  - An `(nwalkers, ndim)` start array built as "optimum + tiny jitter".
  - `run_mcmc(p0, n, progress=True)`, then `get_chain(discard=..., thin=..., flat=True)`.
  - A τ check.

  telescopic should keep the first and last items. For the start array, it should offer a *distribution* π₀, and a documented bridge from "optimum + jitter" to π₀ (see Q5).
- **The walkers/chains confusion is the clearest naming lesson.**
  - Coupled-chain replicates *are* independent. If telescopic calls them "walkers", users will assume ensemble semantics, and vice versa.
  - Use "replicate" (or "pair") for an independent coupled pair, "chain X / chain Y" for the two marginals, and never "walker" unless an ensemble kernel is coupled.
  - Exporting replicates' X-chains to ArviZ as `chain` is legitimate, because they really are independent. This is a selling point over emcee.
- **Users need a stopping rule and a variance estimate they do not have to hand-roll.**
  - emcee users hand-roll stopping loops on τ, which the authors themselves call fragile.
  - Unbiased estimators from independent replicates give standard errors from the replicate variance, with no τ estimation (Q5). This should be front and centre in telescopic's `summary()`.
- **Avoid four emcee anti-patterns:**
  - global random state;
  - pool silently ignored under `vectorize=True`;
  - `assert`-based validation of user input;
  - a long tail of deprecated arguments kept in the main signature.

### Gaps
- Issue bodies for #299, #236, #14, #150, #121 and #369 were not read, only their titles and comment counts. Issue #55 and #306 come from the sibling note.
- Carpenter's blog post could not be retrieved (HTTP 403). It is cited only through Hogg & Foreman-Mackey's footnote.
- No bibliometric study was found of how often astronomy papers report τ, R̂ or ESS for emcee runs. Practice is inferred from documentation, issue threads and wrapper packages (RadVel, zeus).
- Whether Huijser et al. was published in a journal could not be confirmed. Only the arXiv versions were checked.

## Q2. dynesty: API surface and the lessons of the 3.0 redesign

### Takeaway
dynesty 3.1.0, released 2026-07-17, keeps a broad constructor that takes either strings or objects:
- `NestedSampler(loglikelihood, prior_transform, ndim, nlive=500, bound='multi', sample='auto', ..., rstate=None, queue_size=None, pool=None, use_pool=None, ...)`.
- `sample=` may now be an `InternalSampler` instance. `'auto'` picks a method by dimension.
- `run_nested()` has built-in time-based checkpointing.
- Results are an immutable record with `importance_weights()`, `samples_equal()` and `summary()`.
- `plotting` and `utils` modules provide run-level error analysis and `merge_runs`.

Lessons of the 3.0 redesign (2025-10-04):
- Move algorithm-specific tuning into pluggable objects (`InternalSampler`, `Bound`).
- Return iterator objects rather than tuples.
- Remove rarely used escape hatches.
- Expose per-proposal statistics and full evaluation history for debugging.

Checkpoints are pickles that are explicitly *not* an archive format. Parallel speed-up is sub-linear because proposals are serial, so dynesty also tells users to parallelise *after* the run by merging independent runs. That pattern maps directly onto independent coupled replicates.

### Cited Findings
- **Releases:**
  - 3.0.0 on 2025-10-04 and 3.1.0 on 2026-07-17.
  - The previous line ended with 2.1.5 on 2024-12-17.
  - It requires Python ≥ 3.8 and is classed "5 - Production/Stable".

  ([PyPI dynesty JSON](https://pypi.org/pypi/dynesty/json))
- **Signatures in 3.1.0** ([dynesty.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynesty.py); [sampler.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/sampler.py); [dynamicsampler.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynamicsampler.py)):
  - `NestedSampler(loglikelihood, prior_transform, ndim, nlive=500, bound='multi', sample='auto', periodic=None, reflective=None, update_interval=None, first_update=None, rstate=None, queue_size=None, pool=None, use_pool=None, live_points=None, logl_args=None, logl_kwargs=None, ptform_args=None, ptform_kwargs=None, enlarge=None, bootstrap=None, walks=None, facc=0.5, slices=None, ncdim=None, blob=False, save_evaluation_history=False, history_filename=None)`.
  - `DynamicNestedSampler` is the same, minus `live_points`.
  - Static run: `run_nested(maxiter=None, maxcall=None, dlogz=None, logl_max=inf, add_live=True, print_progress=True, print_func=None, save_bounds=True, checkpoint_file=None, checkpoint_every=60, resume=False)`.
  - Dynamic run: `run_nested(nlive_init=None, ..., maxbatch=None, n_effective=None, stop_function=None, stop_kwargs=None, use_stop=True, ..., resume=False, checkpoint_file=None, checkpoint_every=60)`.
  - Lower-level methods: `sample(...)` generator, `add_live_points()`, `add_batch(...)`, `save(fname)` and `restore(fname, pool=None)`.
- **Strings or objects for `sample`** ([dynesty.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynesty.py)):
  - `_get_internal_sampler` accepts `'auto'`, `'unif'`, `'rwalk'`, `'slice'`, `'rslice'` or an `InternalSampler` instance, which is cloned with `_new_from_template`.
  - `'auto'` gives `UniformBoundSampler` for ndim < 10, `RWalkSampler(walks=ndim+20)` for 10 ≤ ndim ≤ 20, and `RSliceSampler(slices=3+ndim)` above that.
  - Mismatched keywords trigger a warning: "Specifying slice option while using rwalk sampler or walks option with a slice sampler does not make sense".
  - So 3.x layered objects *on top of* the keyword path rather than deleting `walks`/`slices`/`facc`.
- **Parallelism keywords** ([dynesty.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynesty.py); [dynesty quickstart](https://dynesty.readthedocs.io/en/stable/quickstart.html)):
  - `queue_size`: "Carry out likelihood evaluations in parallel by queueing up new live point proposals using (at most) `queue_size` many threads ... If no value is passed, this defaults to the size of the pool (determined from `pool._processes` or `pool.size` ...) and `1` otherwise".
  - `use_pool` is a dict of flags covering `prior_transform`, `loglikelihood`, `propose_point` and `update_bound`.
  - `_parse_pool_queue` raises "`queue_size > 1` but no `pool` provided." ([utils.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py)).
- **Parallelism limits** ([dynesty quickstart, "Parallelization"](https://dynesty.readthedocs.io/en/stable/quickstart.html)):
  - `pool.map` is "a *synchronous* function call ... The call time for functions is therefore limited by the slowest-performing member of the pool".
  - "live point proposals must be done serially in order to avoid breaking the statistical properties of Nested Sampling". This gives a sub-linear speed-up S(M,K) = K ln(1 + M/K) for M processes and K live points.
  - It is "possible to combine multiple independent Nested Sampling runs into a single run, giving users an option as to whether they want to parallelize dynesty *during* runtime (using a user-provided pool) or *after* runtime (by merging the runs together)".
- **dynesty's own pool** avoids re-pickling. `with Pool(10, loglike, ptform, logl_args=...) as pool: NestedSampler(pool.loglikelihood, pool.prior_transform, ndim, pool=pool)` is recommended when arguments are "very large", to avoid "repeated pickling of those arguments" ([dynesty quickstart](https://dynesty.readthedocs.io/en/stable/quickstart.html); [pool.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/pool.py)).
- **For clusters,** the FAQ recommends schwimmbad's `MPIPool`. On debugging, the maintainer is candid: "as multi-threading and multi-processing are notoriously difficult to debug, especially on a problem I'm not familiar with, it's likely that I might not be able to help all that much" ([dynesty FAQ](https://dynesty.readthedocs.io/en/stable/faq.html)).
- **Checkpoint implementation** ([utils.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py); [dynesty FAQ](https://dynesty.readthedocs.io/en/stable/faq.html); [dynesty quickstart](https://dynesty.readthedocs.io/en/stable/quickstart.html)):
  - `save_sampler` pickles `{'sampler', 'version', 'format_version'}` to `fname + '.tmp'` and then renames it, an atomic write. Its `FileExistsError` fallback for Windows dates from 2.1.3.
  - `restore_sampler` checks `format_version` and warns: "The dynesty version in the checkpoint file (...) does not match the current dynesty version (...). That is *NOT* guaranteed to work".
  - Restoring re-attaches a pool and reconciles `queue_size`.
  - Users with unpicklable functions are told to set `dynesty.utils.pickle_module = dill`.
  - The quickstart warns against using checkpoints for persistence; the sibling note quotes this.
- **Results object** ([utils.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py)):
  - `Results` is "meant to be unchangeable record of the static or dynamic nested run".
  - Keys are validated against an allowed list. `samples_u`, `samples_id`, `logl` and `samples` are required.
  - Methods: `importance_weights()`, `samples_equal(rstate=None)`, `asdict()`, `keys()`, `items()`, `isdynamic()`, and `summary()`. `summary()` *prints* nlive/niter/ncall/eff/logz rather than returning a string.
- **Helpers** ([plotting.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/plotting.py); [utils.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py)):
  - Plotting: `runplot`, `traceplot`, `cornerpoints`, `cornerplot`, `boundplot`, `cornerbound`.
  - Utilities: `resample_equal`, `mean_and_cov`, `quantile(x, q, weights)`, `jitter_run`, `resample_run`, `reweight_run`, `unravel_run`, `merge_runs`, `kld_error`.
- **Error budgets from the run itself:** "we are able to probe these uncertainties **within the same run used to compute the results**", through `jitter_run`/`resample_run` and the dynamic stopping function ([dynesty "Nested Sampling Errors"](https://dynesty.readthedocs.io/en/stable/errors.html)).
- **Random state** ([dynesty.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynesty.py); [sampler.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/sampler.py); [utils.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py)):
  - `rstate` is a `numpy.random.Generator`.
  - The docstring says "If not given, the global random state of the `~numpy.random` module will be used". The code instead calls `get_random_generator()`, which returns a fresh `np.random.Generator(np.random.PCG64(None))`. So the docstring is stale, and unseeded runs are not affected by `np.random.seed`.
  - Pool tasks get independent streams from `get_seed_sequence(rstate, nitems)`, which spawns a `SeedSequence`.
- **3.0.0 changelog** ([dynesty CHANGELOG](https://github.com/joshspeagle/dynesty/blob/master/CHANGELOG.md)):
  - Added:
    - "New InternalSampler interface ... you can also provide samplers directly to NestedSampler calls, i.e. 'ns= NestedSampler(logl, prior, sample=RWalkSampler(walks=4))'";
    - "the results object has proposal_stats information from the samplers with information such as n_accept, n_reject";
    - `save_evaluation_history` "even when using parallel sampling";
    - "New Bound interface".
  - Removed: `n_effective` from static `run_nested`, the custom `update_func` ("in favour of new sampler interfaace"), the "'user-defined' proposal distribution", and the hslice sampler.
  - Iterators no longer "return a tuple of things, but instead an special Iterator object".
  - Internal names were changed "for consistency".
- **3.1.0 changes** ([dynesty CHANGELOG](https://github.com/joshspeagle/dynesty/blob/master/CHANGELOG.md)):
  - Shows an ETA and the expected number of iterations.
  - Warns about `queue_size` when a run is restored with a different pool size.
  - Uses `chunksize=1` for its pool.
  - Reads `pool._processes` "to reduce the need for manual `queue_size` specification".
  - Fixes the it/s display after a restore.
- **Correctness-driven deprecation (unreleased):** "Remove the support for reflective boundary conditions, as they do not preserve detailed balance. The `reflective` argument is still accepted, but it is ignored and emits a warning" ([dynesty CHANGELOG](https://github.com/joshspeagle/dynesty/blob/master/CHANGELOG.md)).
- **Validation of user-supplied starting points:** "**WARNING: It is crucial that the initial set of live points have been sampled from the prior. Failure to provide a set of valid live points will result in incorrect results.**" ([dynesty.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynesty.py)).
- **Constructor conflicts raise errors:** combining `enlarge` and `bootstrap` raises "Enlarge and bootstrap together do not make sense unless bootstrap=0 or enlarge = 1". Unknown `first_update` keys raise "Unrecognized keywords in first_update" ([dynesty.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynesty.py)).
- **Motivation for 3.0:** a maintainer's proposal to replace tuning keywords with configured objects, and Bilby's porting experience. Both are in [dynesty #495](https://github.com/joshspeagle/dynesty/issues/495) and summarised in the sibling note.

### Inferences
- **dynesty's "parallelise after runtime by merging runs" is exactly telescopic's native mode.** Every replicate is an independent unbiased estimate. telescopic should therefore make `merge` / `concatenate` of results trivial and safe, and say in the docs that a pool is optional.
- **Learn from dynesty's checkpoint design.** Pickled whole-sampler checkpoints carry version fragility. Independent replicates allow a simpler, robust alternative: append completed replicates to a file, and regenerate any unfinished replicate from its own spawned seed (Q4, Q5).
- **Keep strings and objects side by side.** dynesty 3.x accepts both (`sample='rwalk'` or `sample=RWalkSampler(...)`), which kept old scripts working. telescopic can do the same for kernels and couplings from day one (`kernel="rwmh"` or `kernel=RWMH(cov=...)`), without a legacy keyword tail.
- **Fix two dynesty small-print problems.** `summary()` should return a string or table object rather than print. The random-state docstring must match the code.

### Gaps
- `TODO.md`, `AGENTS.md` and `TESTING.md` in the dynesty repository were not read.
- The 3.0 release notes on GitHub Releases, as distinct from `CHANGELOG.md`, were not checked.
- The `stop_function`/`n_effective` semantics of the dynamic sampler were not examined in detail.

## Q3. Other packages: what each does well and badly

### Takeaway
The packages with the cleanest APIs share three habits:
- They separate the algorithm object from the driver: BlackJAX `init`/`step`, NumPyro `MCMC(NUTS(...))`, emcee moves, dynesty `InternalSampler`.
- They return structured results: CmdStanPy `summary()`/`diagnose()`, Bilby `Result`, pocoMC `posterior()`.
- They make stopping explicit: zeus callbacks, Eryn `stopping_fn`, Cobaya's R−1 thresholds, Pigeons' doubling rounds.

The packages most relevant to telescopic's distinctive needs:
- **Pigeons.jl** has "Parallelism Invariance": output identical for any number of threads or machines, given a seed. It checkpoints with doubling rounds, and ships an exact invariance test for kernels.
- **BlackJAX 1.7.0** (2026-10-02) has a new `coupled_hmc`. It is a pair kernel with explicit `(first, second)` arguments and shared innovations, and it explicitly does *not* provide meeting or unbiased estimators.
- **zeus `ChainManager` and RadVel** run several independent ensembles in parallel and compute R̂ across them.

Recurring weaknesses:
- global seeding (pocoMC, emcee);
- seeds that stop reproducing after adaptation in parallel (Cobaya);
- packages with no release for years (zeus since 2023-01, pocoMC since 2024-09, ptemcee since 2015).

### Cited Findings

| Package (version, date) | Target interface | Parallelism | Persistence | Results | Seeding |
|---|---|---|---|---|---|
| zeus-mcmc 2.5.4 (2023-01-12) | `logprob_fn(x, *args)`, vectorize, blobs | `pool`; `ChainManager` for independent MPI ensembles | `SaveProgressCallback(filename='./chains.h5')` | `get_chain`, `summary` (logged), `act`, `ess`, `efficiency` | (not examined) |
| pocoMC 1.2.6 (2024-09-20) | `prior` object + `likelihood`, vectorize, blobs | `pool` | `save_every`, `resume_state_path` | `posterior()` → samples, **weights**, logl, logp; `evidence()` | `random_state` → `np.random.seed` + `torch.manual_seed` (global) |
| Eryn 1.2.6 (2026-02-04) | `log_like_fn` + `priors`, branches/leaves | `pool` | emcee-style backends | `get_log_like`, `get_log_prior`, `get_log_posterior` | (not examined) |
| Bilby 2.8.2 (2026-08-10) | `Likelihood` object + priors dict | `npool` | result files | `Result` (posterior DataFrame, `save_to_file`) | (not examined) |
| Cobaya 3.6.2 (2026-03-27) | YAML/dict `info` | MPI chains | `resume`/`force` | sample collections | `seed` (caveat in parallel) |
| BlackJAX 1.7.1 (2026-10-02) | `logdensity_fn(pytree)` | user's `vmap`/`pmap` | user's | `(state, info)` per step | explicit JAX key |
| NumPyro 0.22.0 (2026-09-18) | model function | `chain_method` parallel/sequential/vectorized | `post_warmup_state` | `get_samples(group_by_chain)`, `print_summary` | explicit JAX key |
| CmdStanPy 1.3.0 (2025-10-20) | Stan program + data | `chains`, `parallel_chains`, `threads_per_chain` | CSV files | `summary()` DataFrame, `diagnose()`, `draws*()` | `seed` + chain-id offsets |
| Pigeons.jl 0.4.11 (main, last commit 2026-10-13) | log potential / Turing / Stan | threads, MPI, `ChildProcess` | `checkpoint=true` | `PT` struct, report table | splittable streams, parallelism invariance |
| schwimmbad 0.4.2 (2024-03-14) | n/a (pool abstraction) | `choose_pool(mpi, processes)` | n/a | n/a | n/a |

Sources for the table rows are given in the bullets below. Dates are from the PyPI JSON API. Pigeons is a Julia package, so its version is from `Project.toml`.

- **zeus (Ensemble Slice Sampling)** ([zeus-mcmc 2.5.4 on PyPI](https://pypi.org/project/zeus-mcmc/2.5.4/); files `zeus/ensemble.py`, `zeus/callbacks.py`, `zeus/parallel.py`):
  - Constructor: `EnsembleSampler(nwalkers, ndim, logprob_fn, args=None, kwargs=None, moves=None, tune=True, tolerance=0.05, patience=5, maxsteps=10000, mu=1.0, maxiter=10000, pool=None, vectorize=False, blobs_dtype=None, verbose=True, check_walkers=True, shuffle_ensemble=True, light_mode=False)`.
  - Its errors are clear: "Please provide at least (2 * ndim) walkers." and "Please provide an even number of walkers."
  - Stopping is composable: `run_mcmc(start, nsteps=1000, ..., callbacks=None)`, where "Sampling terminates when all of the callback functions return ``True``". The built-in callbacks are `AutocorrelationCallback(ncheck=100, dact=0.01, nact=10, discard=0.5, trigger=True, method='mk')`, `SplitRCallback`, `MinIterCallback(nmin=1000)`, `ParallelSplitRCallback(..., chainmanager=None)` and `SaveProgressCallback`.
  - `ChainManager(nchains, comm)` is a context manager "to run ``nchains`` in parallel with each chain having its own ``MPIPool`` of parallel walkers".
  - `summary()` writes to `logging` rather than returning an object.
  - The licence is GPL-3.0 (`__license__` in `zeus/__init__.py`).
- **pocoMC (preconditioned Monte Carlo)** ([pocomc 1.2.6 on PyPI](https://pypi.org/project/pocomc/1.2.6/); files `pocomc/sampler.py`, `pocomc/prior.py`):
  - Constructor: `Sampler(prior, likelihood, n_dim=None, n_effective=512, n_active=256, likelihood_args=None, likelihood_kwargs=None, vectorize=False, blobs_dtype=None, periodic=None, reflective=None, transform='probit', pool=None, ..., output_dir=None, output_label=None, random_state=None, n_ess=None)`.
  - Run and results: `run(n_total=4096, n_evidence=4096, progress=True, resume_state_path=None, save_every=None)`; `posterior(resample=False, return_blobs=False, trim_importance_weights=True, return_logw=False, ...)` returns `samples, weights, logl, logp`; `evidence()`.
  - `Prior(dists)` takes a list of `scipy.stats` distributions.
  - Seeding is global: `random_state` calls `np.random.seed(random_state)` and `torch.manual_seed(random_state)`.
- **Eryn ("omni-MCMC", an emcee derivative)** ([eryn 1.2.6 on PyPI](https://pypi.org/project/eryn/1.2.6/); file `eryn/ensemble.py`):
  - Constructor: `EnsembleSampler(nwalkers, ndims, log_like_fn, priors, provide_groups=False, provide_supplemental=False, tempering_kwargs={}, branch_names=None, nbranches=1, nleaves_max=1, nleaves_min=0, pool=None, moves=None, rj_moves=None, dr_moves=None, ..., backend=None, vectorize=False, blobs_dtype=None, ..., update_fn=None, update_iterations=-1, stopping_fn=None, stopping_iterations=-1, ...)`.
  - The stopping hook has signature `(sampler iteration, last sample state object, EnsembleSampler object)`, and is checked every `stopping_iterations`.
  - Separate accessors exist for likelihood, prior and posterior.
  - The constructor has more than 30 parameters, against the Keras "6–7 arguments" guideline cited in the sibling note.
- **Bilby (sampler-agnostic front end)** ([bilby 2.8.2 on PyPI](https://pypi.org/project/bilby/2.8.2/); files `bilby/core/sampler/__init__.py`, `base_sampler.py`, `likelihood.py`, `result.py`, `entry_points.txt`):
  - One call: `run_sampler(likelihood, priors=None, label='label', outdir='outdir', sampler='dynesty', use_ratio=None, injection_parameters=None, conversion_function=None, plot=False, ..., save=True, gzip=False, result_class=None, npool=1, **kwargs)`.
  - Samplers are plug-ins registered through the `bilby.samplers` entry-point group. 17 ship with Bilby, including emcee, zeus, dynesty, dynamic_dynesty, ultranest, ptemcee, pymc, nessai and bilby_mcmc.
  - The likelihood is an object: `Likelihood(parameters=None).log_likelihood(parameters=None)`.
  - The result is an object: `Result` with a `posterior` DataFrame, `save_to_file(filename=None, overwrite=False, outdir=None, extension=None, gzip=False)`, `from_hdf5`/`from_json`, `plot_corner`, `plot_walkers` and `ResultList(results, consistency_level='warning')`.
- **Cobaya** ([cobaya 3.6.2 on PyPI](https://pypi.org/project/cobaya/3.6.2/); files `cobaya/run.py`, `cobaya/samplers/mcmc/mcmc.yaml`):
  - Entry point: `run(info_or_yaml_or_file, packages_path=None, output=None, debug=None, stop_at_error=None, resume=None, force=None, minimize=None, no_mpi=False, test=None, override=None, allow_changes=False)`. `resume` will "continue an existing run" and `force` will "overwrite existing output if it exists".
  - Convergence defaults: `Rminus1_stop: 0.01` ("Gelman-Rubin R-1 on means") and `Rminus1_cl_stop: 0.2`, computed across MPI chains. Without MPI, `Rminus1_single_split: 4` compares fractions of one chain.
  - Proposal learning is gated by R−1: `learn_proposal: True`, `learn_proposal_Rminus1_max: 2.`.
  - Stuck chains are detected: `max_tries: 40d` is the "max attempts (= weight-1) before deciding that the chain is stuck and failing".
  - Seeding caveat: "NB: in parallel runs, only works until to the first proposer covmat update".
- **BlackJAX** ([blackjax 1.7.1 on PyPI](https://pypi.org/project/blackjax/1.7.1/); file `blackjax/base.py`):
  - Algorithms are a `SamplingAlgorithm(init, step)` pair of pure functions: "the kernel functions are pure functions and do not have internal state", operating on states that cache e.g. log-density and gradient.
  - 1.6.1 added `ess_bulk`, `ess_tail` and `pareto_khat` diagnostics. 1.6.2 aligned the `ess_tail` default to ArviZ ([BlackJAX releases](https://github.com/blackjax-devs/blackjax/releases)).
- **BlackJAX coupled HMC (1.7.0)** ([BlackJAX releases](https://github.com/blackjax-devs/blackjax/releases); module docstring of `blackjax/mcmc/coupled_hmc.py` in [blackjax 1.7.1 on PyPI](https://pypi.org/project/blackjax/1.7.1/)):
  - Release note: "Add coupled HMC: two marginals sharing their random inputs" (PR #1033).
  - Scope disclaimer: the module "does not claim product-target invariance, meeting, unbiased estimators, or an efficiency improvement".
  - Contract: "Given the same ``(state, standard_normal, uniform)`` triple and per-marginal parameters, each marginal performs the same transition as ordinary HMC".
  - Couplings: `synchronous` (same z) or `reflection` (z′ = z − 2e(eᵀz)), with a pluggable `direction_fn`.
  - Arguments: "Every per-marginal argument is an explicit ``(first, second)`` pair, including shared values as ``(value, value)``; nothing is broadcast or inferred".
  - Placement: "There is no top-level ``blackjax.coupled_hmc``".
  - States and diagnostics: `CoupledHMCState(first, second)` and `CoupledHMCInfo`. Each marginal keeps "complete, separate HMC diagnostics", plus the shared innovations.
- **NumPyro** ([numpyro 0.22.0 on PyPI](https://pypi.org/project/numpyro/0.22.0/); files `numpyro/infer/mcmc.py`, `numpyro/infer/initialization.py`):
  - Driver: `MCMC(sampler, *, num_warmup, num_samples, num_chains=1, thinning=1, postprocess_fn=None, chain_method='parallel', progress_bar=True, progress_rate=None, jit_model_args=False)`.
  - Run and outputs: `run(rng_key, *args, extra_fields=(), init_params=None, **kwargs)`, `get_samples(group_by_chain=False)`, `print_summary(prob=0.9, ...)`, `post_warmup_state`, `last_state`.
  - `chain_method`: "'parallel' (default), 'sequential', 'vectorized'". It falls back to sequential "If there are not enough devices", and 'vectorized' is "an experimental feature".
  - The kernel abstract base `MCMCKernel` defines `init`, `sample`, `postprocess_fn`, `sample_field`, `default_fields` and `get_diagnostics_str`.
  - Initialisation is a strategy function: `init_to_median(num_samples=15)`, `init_to_mean`, `init_to_sample`, `init_to_uniform(radius=2)` ("a random point in the area `(-radius, radius)` of unconstrained domain"), `init_to_feasible` and `init_to_value(values)`.
- **PyMC** ([pymc 6.3.2 on PyPI](https://pypi.org/project/pymc/6.3.2/); files `pymc/model/core.py`, `pymc/sampling/mcmc.py`):
  - A failed starting point raises `SamplingError("Initial evaluation of model at starting point failed!\nStarting values:\n{...}\n\nLogp initial evaluation results:\n{...}\nYou can call `model.debug()` for more details.")`, listing the per-variable log-probabilities.
  - Initial jitter is retried: `jitter_max_retries: int = 10`.
  - The sibling note covers `pm.sample` and the rejection of `RandomState`.
- **Stan via CmdStanPy** ([cmdstanpy 1.3.0 on PyPI](https://pypi.org/project/cmdstanpy/1.3.0/); files `cmdstanpy/model.py`, `cmdstanpy/stanfit/mcmc.py`):
  - Sampling keywords: `sample(data, chains, parallel_chains, threads_per_chain, seed, chain_ids, inits, iter_warmup, iter_sampling, save_warmup, thin, max_treedepth, metric, step_size, adapt_*, ..., output_dir, sig_figs, ..., show_progress=True, show_console=False, refresh, ..., timeout, *, force_one_process_per_chain, inv_metric)`.
  - Default initialisation: "The default initialization behavior will initialize all parameter values on range [-2, 2] on the *unconstrained* support". Users may pass a number n (range [-n, n]), 0, a dict, a file, or per-chain lists.
  - Seeding: "When the same seed is used across all chains, the chain-id is used to advance the RNG to avoid dependent samples".
  - Outputs: `CmdStanMCMC.draws(inc_warmup=False, concat_chains=False)`, `draws_pd`, `draws_xr`, `stan_variable`, `method_variables`, `save_csvfiles(dir)`, `from_csv`.
  - `summary(percentiles=(5, 50, 95), sig_figs=6)` returns a pandas DataFrame.
  - `diagnose()` runs `bin/diagnose`, which checks "Transitions that hit the maximum treedepth", "Divergent transitions", "Low E-BFMI values", "Low effective sample sizes" and "High R-hat values".
- **Pigeons.jl (parallel tempering)** ([Pigeons docs index](https://pigeons.run/stable/); [distributed.md](https://github.com/Julia-Tempering/Pigeons.jl/blob/main/docs/src/distributed.md); [checkpoints.md](https://github.com/Julia-Tempering/Pigeons.jl/blob/main/docs/src/checkpoints.md); [output-reports.md](https://github.com/Julia-Tempering/Pigeons.jl/blob/main/docs/src/output-reports.md); [correctness.md](https://github.com/Julia-Tempering/Pigeons.jl/blob/main/docs/src/correctness.md)):
  - API: `pt = pigeons(target = toy_mvn_target(100))`, with arguments "forwarded to `Inputs`", and `record = [traces; record_default()]`.
  - Rounds: "the number of iteration per round doubles at each round and adaptation is performed between rounds". The report prints one row per round: scans, Λ ("the global communication barrier"), time, allocations, log(Z₁/Z₀), min(α), mean(α).
  - Parallelism Invariance: "The output of Pigeons is invariant to the number of machines and/or threads", i.e. X_{m,t}(s) = X_{m′,t′}(s). The rationale: "the developer can first focus on the fully serial randomized algorithm, and then use it as an easy-to-compare gold-standard reference". It names "Global, thread-local and task-local random number generators" and non-associative floating-point reductions as the two threats, and solves the first with splittable random streams.
  - Checkpoints ensure "not more than half of the work is lost". A run can be reloaded with `PT("results/latest")`, extended with `increment_n_rounds!`, and continued on more MPI processes.
  - An Exact Invariance Test (EIT) "formulates an hypothesis test where the null hypothesis is that the provided `explorer` kernel is invariant with respect to the target distribution".
- **UltraNest resume semantics** ([ultranest 4.6.3 on PyPI](https://pypi.org/project/ultranest/4.6.3/); file `ultranest/integrator.py`):
  - `resume` takes 'resume', 'resume-similar', 'overwrite' or 'subfolder'. 'resume' "Only works when dimensionality, transform or likelihood are consistent".
  - Inconsistency is detected by re-evaluating stored points (`assert np.allclose(v2, initial_points_v), 'transform inconsistent, cannot resume'`).
- **ArviZ and weights** ([arviz-base 1.3.1 on PyPI](https://pypi.org/project/arviz-base/1.3.1/); file `arviz_base/reorg.py`):
  - `extract(..., num_samples=None, weights=None, resampling_method=None, ...)` accepts weights only for resampling: "weights are only compatible with num_samples", with stratified or multinomial resampling.
  - The sibling note covers DataTree outputs and the NumPy-only `array_stats`.
- **schwimmbad** ([schwimmbad 0.4.2 on PyPI](https://pypi.org/project/schwimmbad/0.4.2/); file `schwimmbad/__init__.py`; [emcee parallel tutorial](https://emcee.readthedocs.io/en/stable/tutorials/parallel/)):
  - It exports `SerialPool`, `MultiPool`, `MPIPool` and `JoblibPool`.
  - `choose_pool(mpi=False, processes=1, **kwargs)` picks one "given options from, e.g., argparse", and raises "Tried to run with MPI but MPIPool not enabled."
  - The MPI idiom is `with MPIPool() as pool: if not pool.is_master(): pool.wait(); sys.exit(0)`.
- **Ageing of emcee-family packages** ([PyPI JSON API](https://pypi.org/pypi/ptemcee/json)):
  - ptemcee is still 1.0.0 (2015-08-28), although emcee moved PTSampler there. emcee issue #236, "PTsampler is removed", has 12 comments.
  - zeus's last release was 2023-01-12, and pocoMC's 2024-09-20.

### Inferences
- **Copy:**
  - zeus's callback-based stopping, and Eryn's `stopping_fn`, as a general, testable stopping interface.
  - CmdStanPy's split between `summary()` (DataFrame) and `diagnose()` (plain-language checks).
  - NumPyro's initialisation *strategies*, which naturally become π₀ objects.
  - Pigeons' parallelism invariance and checkpoint-by-rounds.
  - UltraNest's consistency check on resume.
  - Bilby's entry-point plug-in registry for third-party kernels.
  - BlackJAX's pure `init`/`step` contract for the future JAX backend.
- **Avoid:**
  - global seeding (pocoMC, emcee);
  - logging-only summaries (zeus);
  - 30-argument constructors (Eryn);
  - seeds that silently stop reproducing in parallel after adaptation (Cobaya's documented caveat).
- **BlackJAX's `coupled_hmc` is the closest modern API precedent for "kernel + coupling",** and its explicit disclaimers are a model of honest scoping. telescopic could interoperate with it for the JAX backend: wrap a BlackJAX kernel as a telescopic kernel and supply a meeting-capable coupling on top. telescopic must add what BlackJAX explicitly does not provide: a maximal coupling component that lets chains meet exactly, faithfulness after meeting, and estimators.
- **Output layer.** Neither ArviZ nor corner (Q5) handles negative weights, so telescopic must own its signed-measure summaries. It should export only legitimately unweighted pieces, such as the X-chains, to ArviZ.

### Gaps
- Not examined:
  - nautilus's and UltraNest's newest APIs beyond the sibling note;
  - pocoMC 1.2.x docs on `n_effective`/`n_active` tuning;
  - Eryn's backend and HDF5 schema;
  - zeus's seeding (no `rng`/`random_state` argument appears in the constructor; its internal RNG use was not checked);
  - Bilby's checkpoint/resume flags.
- **Pigeons release dates** were not obtained (Julia General registry not checked). The version is from `Project.toml` on `main`.
- **No first-hand user reviews** were found of these packages' APIs. The assessments are design judgements from source and docs.

## Q4. Cross-cutting patterns and anti-patterns: target, initialisation, parallelism, progress, checkpointing, results, reproducibility, errors, defaults, extensibility

### Takeaway
Across sampler packages, the recurring winning patterns are:
- a plain log-density callable, with `args`/`kwargs`, an optional batched ("vectorised") mode and named parameters;
- initialisation as a *strategy or distribution object*, with validation that names the bad parameters;
- a `pool` that is any object with `map`, plus independent random streams per task;
- a progress display that reports diagnostics and ETA, not just a bar;
- crash-safe, resumable output that checks problem consistency on resume;
- an immutable result object with a non-printing `summary()`, plots and documented serialisation;
- explicit `rng`, with per-task spawned streams, ideally parallelism-invariant;
- early, specific errors;
- automatic defaults that can be inspected;
- algorithms as small pluggable objects.

The anti-patterns: hidden global RNG state, silently ignored arguments, pickled checkpoints used as archives, R̂ across dependent walkers, and output tools that assume non-negative weights.

### Cited Findings

#### Target specification
- **emcee:** `log_prob_fn(x, *args, **kwargs)`, `vectorize` (pool then ignored), blobs (extra return values, with `blobs_dtype`), and `parameter_names` giving dict input ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py); [emcee Blobs](https://emcee.readthedocs.io/en/stable/user/blobs/)).
- **dynesty:** `loglikelihood` plus `prior_transform` from the unit cube, `logl_args`/`ptform_args`, and `blob=True` ([dynesty.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynesty.py)).
- **Prior objects:** pocoMC's `Prior(dists)`; Bilby's `Likelihood.log_likelihood(parameters)` with prior dicts; Eryn's separate `log_like_fn`/`priors` ([pocomc 1.2.6](https://pypi.org/project/pocomc/1.2.6/); [bilby 2.8.2](https://pypi.org/project/bilby/2.8.2/); [eryn 1.2.6](https://pypi.org/project/eryn/1.2.6/)). The sibling note covers nautilus's `Prior` with `add_parameter`.
- **BlackJAX:** `logdensity_fn` over pytrees ([blackjax 1.7.1](https://pypi.org/project/blackjax/1.7.1/)).
- **Deterministic log-density is a stated requirement:** Hogg & Foreman-Mackey ask whether the likelihood gives "exactly the same answer" at the same point, and say how to make it single-valued ([arXiv:1710.06068](https://arxiv.org/abs/1710.06068)). emcee issue #121, "stochastic calculations inside emcee", is one of its most-commented issues ([emcee issues](https://github.com/dfm/emcee/issues/121)).

#### Initialisation
- **Positions:**
  - emcee takes an explicit `(nwalkers, ndim)` array, checked by condition number, with an opt-out ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)).
  - Its FAQ advice is a "small ball" ([emcee FAQ](https://emcee.readthedocs.io/en/stable/user/faq/)).
  - Hogg & Foreman-Mackey: optimise first, use a ball smaller than the posterior, and warn that this makes Gelman–Rubin non-conservative ([arXiv:1710.06068](https://arxiv.org/abs/1710.06068)).
- **Distributions or strategies:**
  - Stan/CmdStanPy default to uniform(−2, 2) on the unconstrained scale ([cmdstanpy 1.3.0](https://pypi.org/project/cmdstanpy/1.3.0/)).
  - NumPyro offers `init_to_*` strategies ([numpyro 0.22.0](https://pypi.org/project/numpyro/0.22.0/)).
  - dynesty draws live points from the prior, and warns that user-supplied points must be prior draws ([dynesty.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynesty.py)).
- **Validation UX:**
  - PyMC prints the starting values and the per-variable log-probabilities, and suggests `model.debug()` ([pymc 6.3.2](https://pypi.org/project/pymc/6.3.2/)).
  - zeus states the minimum walker count ([zeus-mcmc 2.5.4](https://pypi.org/project/zeus-mcmc/2.5.4/)).

#### Parallelism
- **Pool interfaces:**
  - emcee: any object with `map`, with MPI via schwimmbad ([emcee parallel tutorial](https://emcee.readthedocs.io/en/stable/tutorials/parallel/)).
  - dynesty: `pool`, `queue_size`, `use_pool`, plus its own `Pool` to avoid pickling. Its speed-up is sub-linear, so it also suggests merging runs after the fact ([dynesty quickstart](https://dynesty.readthedocs.io/en/stable/quickstart.html)).
  - schwimmbad: `choose_pool` ([schwimmbad 0.4.2](https://pypi.org/project/schwimmbad/0.4.2/)).
- **Independent chains, managed by the package:**
  - zeus `ChainManager` ([zeus-mcmc 2.5.4](https://pypi.org/project/zeus-mcmc/2.5.4/)).
  - NumPyro `chain_method` ([numpyro 0.22.0](https://pypi.org/project/numpyro/0.22.0/)).
  - CmdStanPy `parallel_chains` ([cmdstanpy 1.3.0](https://pypi.org/project/cmdstanpy/1.3.0/)).
  - Pigeons threads, MPI and `ChildProcess` ([Pigeons distributed.md](https://github.com/Julia-Tempering/Pigeons.jl/blob/main/docs/src/distributed.md)).
- **Pain points:** pickling cost and pool scaling (emcee #199, #447, #452), and pool-dependent results (emcee #351) ([emcee issues](https://github.com/dfm/emcee/issues/351)).

#### Progress reporting
- **emcee:** `progress=True` and `progress_kwargs` ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)).
- **dynesty:** `print_progress`/`print_func`, with an ETA added in 3.1.0 ([dynesty CHANGELOG](https://github.com/joshspeagle/dynesty/blob/master/CHANGELOG.md)).
- **NumPyro:** `progress_bar`, `progress_rate` ([numpyro 0.22.0](https://pypi.org/project/numpyro/0.22.0/)).
- **CmdStanPy:** `show_progress`, `refresh` ([cmdstanpy 1.3.0](https://pypi.org/project/cmdstanpy/1.3.0/)).
- **Pigeons:** a per-round diagnostic table ([Pigeons output-reports.md](https://github.com/Julia-Tempering/Pigeons.jl/blob/main/docs/src/output-reports.md)).
- The sibling note covers UltraNest and nautilus status tables.

#### Checkpoint and resume
- **emcee:** an HDF5 backend that resumes with `run_mcmc(None, n)` (sibling note).
- **dynesty:** a pickled checkpoint every 60 s, written atomically, which warns on version mismatch ([utils.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py)).
- **nautilus:** auto-resumes from `filepath` (sibling note).
- **UltraNest:** resume modes with consistency checks ([ultranest 4.6.3](https://pypi.org/project/ultranest/4.6.3/)).
- **pocoMC:** `save_every`/`resume_state_path` ([pocomc 1.2.6](https://pypi.org/project/pocomc/1.2.6/)).
- **Cobaya:** `resume`/`force` ([cobaya 3.6.2](https://pypi.org/project/cobaya/3.6.2/)).
- **Pigeons:** round-based checkpoints, resumable on more machines ([Pigeons checkpoints.md](https://github.com/Julia-Tempering/Pigeons.jl/blob/main/docs/src/checkpoints.md)).

#### Result objects
- **dynesty:** an immutable `Results` with weights helpers ([utils.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py)).
- **pocoMC:** `posterior()` returns weights ([pocomc 1.2.6](https://pypi.org/project/pocomc/1.2.6/)).
- **Bilby:** `Result` with JSON/HDF5 serialisation ([bilby 2.8.2](https://pypi.org/project/bilby/2.8.2/)).
- **CmdStanPy:** `summary()`/`diagnose()` and xarray output ([cmdstanpy 1.3.0](https://pypi.org/project/cmdstanpy/1.3.0/)).
- **emcee:** `get_*` accessors with `discard`/`thin`/`flat` ([emcee backend.py](https://github.com/dfm/emcee/blob/main/src/emcee/backends/backend.py)).
- **Plotting tools and weights:**
  - corner's weighted `quantile` sorts the samples, takes `np.cumsum` of the weights as a CDF, and calls `np.interp(q, cdf, x[idx])` ([corner 2.3.0 on PyPI, `corner/core.py`](https://pypi.org/project/corner/2.3.0/)).
  - NumPy documents that if `xp` "is non-increasing, interpolation results are meaningless", and this is "not explicitly enforced" ([numpy.interp](https://numpy.org/doc/stable/reference/generated/numpy.interp.html)).

#### Reproducibility
- **emcee:** global-derived `RandomState` ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)).
- **pocoMC:** global `np.random.seed`/`torch.manual_seed` ([pocomc 1.2.6](https://pypi.org/project/pocomc/1.2.6/)).
- **dynesty:** `Generator` plus `SeedSequence.spawn` per pool task ([utils.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py)).
- **CmdStanPy:** seed advanced by chain-id ([cmdstanpy 1.3.0](https://pypi.org/project/cmdstanpy/1.3.0/)).
- **Cobaya:** seed valid only until the first covariance-matrix update in parallel runs ([cobaya 3.6.2](https://pypi.org/project/cobaya/3.6.2/)).
- **Pigeons:** pointwise-identical output across parallel layouts ([Pigeons distributed.md](https://github.com/Julia-Tempering/Pigeons.jl/blob/main/docs/src/distributed.md)).
- The sibling note covers SPEC 7 and NumPy's `spawn`.

#### Warnings and errors
- **emcee:** specific `ValueError` messages, `AutocorrError` with `tau` attached, and a `RuntimeError` for too few walkers ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py); [emcee autocorr.py](https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py); [emcee red_blue.py](https://github.com/dfm/emcee/blob/main/src/emcee/moves/red_blue.py)).
- **dynesty:** conflict errors and stale-checkpoint warnings ([dynesty.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynesty.py); [utils.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py)).
- **PyMC:** `SamplingError` that shows the start values ([pymc 6.3.2](https://pypi.org/project/pymc/6.3.2/)).
- **Cobaya:** stuck-chain detection via `max_tries` ([cobaya 3.6.2](https://pypi.org/project/cobaya/3.6.2/)).

#### Defaults and automatic tuning
- **emcee:** `StretchMove(a=2.0)`, with nothing else to tune ([emcee stretch.py](https://github.com/dfm/emcee/blob/main/src/emcee/moves/stretch.py)).
- **zeus:** `tune=True` with `tolerance`/`patience` ([zeus-mcmc 2.5.4](https://pypi.org/project/zeus-mcmc/2.5.4/)).
- **dynesty:** `sample='auto'` chosen by dimension, and auto enlarge/bootstrap ([dynesty.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynesty.py)).
- **Cobaya:** proposal learning gated by R−1 ([cobaya 3.6.2](https://pypi.org/project/cobaya/3.6.2/)).
- **Pigeons:** adaptation between doubling rounds ([Pigeons docs](https://pigeons.run/stable/)).

#### Extensibility
- **emcee:** `Move`/`RedBlueMove`/`MHMove`, with weighted mixtures ([emcee moves](https://emcee.readthedocs.io/en/stable/user/moves/)).
- **dynesty:** subclasses of `InternalSampler` and `Bound` ([dynesty CHANGELOG](https://github.com/joshspeagle/dynesty/blob/master/CHANGELOG.md)).
- **NumPyro:** the `MCMCKernel` abstract base ([numpyro 0.22.0](https://pypi.org/project/numpyro/0.22.0/)).
- **BlackJAX:** `SamplingAlgorithm(init, step)` ([blackjax 1.7.1](https://pypi.org/project/blackjax/1.7.1/)).
- **Bilby:** entry-point registry ([bilby 2.8.2](https://pypi.org/project/bilby/2.8.2/)).
- **zeus and Eryn:** stopping hooks via callbacks and `stopping_fn` ([zeus-mcmc 2.5.4](https://pypi.org/project/zeus-mcmc/2.5.4/); [eryn 1.2.6](https://pypi.org/project/eryn/1.2.6/)).

### Inferences
Recommended telescopic choice for each pattern (my design judgement):

| Concern | Recommended telescopic choice | Borrowed from |
|---|---|---|
| Target | `log_prob(theta, *args, **kwargs)` returning float or `(float, blobs...)`; `vectorize=True` for `(n, d)` batches; `parameter_names` producing a dict; require a deterministic log-density | emcee, Hogg & Foreman-Mackey §9 |
| Initialisation | π₀ given as a `scipy.stats`-like object (`.rvs(size, random_state)`), a callable `init(rng) -> x`, or helpers (`ball`, `prior`, `empirical(points)`); validate draws and report the failing parameter values | NumPyro strategies, CmdStanPy inits, PyMC error |
| Parallelism | `pool` with `.map` over *replicates*; `vectorize` across replicates in lockstep; error if both are given rather than silently ignoring one | emcee, schwimmbad, dynesty "merge after" |
| Progress | one bar over replicates showing running median/max meeting time, cost, ETA, and the running estimate ± SE | dynesty ETA, Pigeons table, UltraNest (sibling) |
| Checkpoint | append-only per-replicate store (HDF5/Zarr) plus a config fingerprint; resume regenerates unfinished replicates from their own seed; refuse incompatible resumes after re-evaluating stored points | nautilus, UltraNest, Pigeons |
| Results | immutable `CoupledResult` with `summary()` returning a table, `estimate(h)`, `signed_measure()`, `meeting_times`, `plot_*`, `to_datatree()` (unweighted parts), `save`/`load`, `merge` | dynesty `Results`, CmdStanPy, Bilby |
| Reproducibility | `rng=` (SPEC 7); one `SeedSequence` child per replicate, so output is identical for serial, pool or any worker count | Pigeons invariance, dynesty `get_seed_sequence` |
| Errors | specific `ValueError`s; never `assert` on user input; warn on non-meeting or truncation, on k below many meeting times, and on negative-weight quantile computations | emcee, PyMC, Keras guidelines (sibling) |
| Defaults | `kernel="rwmh"` with a frozen, pilot-tuned covariance; `coupling="reflection-maximal"`; `k`, `m` from pilot meeting times (Q5) | dynesty `'auto'`, Cobaya |
| Extensibility | `Kernel` and `Coupling` protocols plus a registry of default couplings per kernel; entry points for third-party kernels | BlackJAX, emcee moves, dynesty 3.0, Bilby |

### Gaps
- No packaging-level data (download counts, dependents) was gathered to weigh how much each pattern matters to users.
- Whether corner or GetDist reject negative weights anywhere else was only checked for corner's `quantile`. A grep of GetDist 1.7.7 for negative-weight checks found none, but its handling was not traced.

## Q5. What a coupled-chain API needs beyond emcee, and candidate API shapes

### Takeaway
Unbiased MCMC with couplings changes the user contract in six ways:
1. Users supply an **initial distribution π₀** to draw from repeatedly, not fixed positions.
2. The unit of work is an **independent replicate**: a coupled pair run until it meets and reaches m, costing about 2(τ−1) + max(1, m+1−τ) kernel calls.
3. Outputs are **signed empirical measures**: weights sum to 1, some are negative, and estimates can fall outside the range of h. Test functions h can therefore be applied *after* the run.
4. **k and m** are tuned from a pilot of meeting times: k is a large quantile, and m is 5–10k.
5. **Diagnostics** centre on the meeting-time distribution, cost and inefficiency, not R̂/τ.
6. **Compute budgets** need care: averaging only the replicates completed by a deadline is biased (Glynn–Heidelberger).

Prior implementations are research code:
- the R package `unbiasedmcmc`, "not a general-purpose statistical software";
- ColCarroll/couplings (Python, 2019, pip-installable from git only);
- a JAX course project;
- BlackJAX's new `coupled_hmc`, which explicitly does not do meeting or unbiased estimation.

No maintained PyPI package was found.

The recommended shape is **layered** (my design judgement):
- (C) pure kernel/coupling building blocks;
- (B) a `CoupledSampler` object with an emcee-like workflow and an append-only backend;
- (A) a one-call `telescopic.sample(...)` that returns a `CoupledResult` with `summary()` and `estimate(h)`.

### Cited Findings

#### Method facts from Jacob, O'Leary & Atchadé (2020) that constrain the API
All bullets in this subsection come from [Jacob, O'Leary & Atchadé, arXiv:1708.03625 v5, "accepted as a JRSS discussion paper"](https://arxiv.org/abs/1708.03625).
- **Coupled kernel:** the coupled kernel P̄ acts on pairs that "marginally start from π0 and evolve according to P" (§2.1). "calls to P̄ are approximately twice as expensive as calls to P", and the cost of H_{k:m} is "comparable to 2(τ − 1) + max(1, m + 1 − τ) iterations of the underlying MCMC algorithm" (§2.3).
- **Interpreting H_{k:m}:** it is "a standard MCMC average with m total iterations and a burn-in period of k − 1 iterations" plus "a bias correction". "If τ ≤ k + 1, then the correction term equals zero" (§2.3).
- **h after the run:** "the test function h does not have to be specified at run-time ... One can store the coupled chains and choose the test function later". Thinning "is feasible in the proposed framework" (§2.3).
- **Out-of-range estimates:** "the proposed estimators can take values outside of the range of the test function h; for instance they can take negative values even if the range of the test function contains only non-negative values" (§2.3).
- **π₀:** "The estimators are unbiased for any choice of π0, including point masses, but this choice has an impact on both the computing cost and the variance".
  - Experiments draw X₀ and Y₀ independently.
  - "unfortunate choices of initial distributions can severely affect the performance ... This suggests trying more than one choice of initialization, especially in the setting of multimodal targets" (§2.3).
- **Signed measure (§2.4):** π̂ = Σ ω_ℓ δ_{Z_ℓ}, with Σ ω_ℓ = 1. "Some of the weights (ω_ℓ) may be negative, making π̂ a signed empirical measure".
  - The weighted CDF "is not monotonically increasing because of negative weights", so a quantile "might be defined as Z_ℓ for any such ℓ". These estimates are consistent as R → ∞.
  - Confidence intervals for quantiles ("perhaps by bootstrapping the R independent copies") are "an interesting area for future research".
  - Alternatives: projection onto probability measures (a Wasserstein generalisation), or isotonic regression.
- **Choosing k and m (§3.1):**
  - "we can sample independent realizations of the meeting time and choose k such that P(τ > k) is small; i.e. we choose k as a large quantile of the meeting times".
  - Efficiency loss relative to MCMC is about 1 + (k + E(τ))/(m − k).
  - "Choosing m as a multiple of k, such as 5k or 10k, makes intuitive sense".
  - There is "a tradeoff with the desired level of parallelism: one might prefer to keep k and m small ... enabling more independent copies to be generated in a given computing time".
  - Inefficiency is defined "as the product of the variance and expected cost of the estimator".
- **Budgets (§3.3):**
  - Averaging the estimators completed by time t on processor p is biased: E[H̄_p(t)] = E[H] − E[H 1(C > t)].
  - The unbiased version H̃_p(t) waits for at least the first estimator on each processor ("The computation of Ñ_p(t) requires the completion of H_p1").
  - CLTs are given for P → ∞ and t → ∞.
  - With t fixed and P → ∞, completion time "has expectation behaving as log P".
- **Cost of couplings (§4.1):**
  - Generic rejection-based maximal coupling (Algorithm 2) has expected cost "two units", but its cost variance "goes to infinity" as the total-variation distance goes to zero.
  - The reflection-maximal coupling for Gaussian-type proposals has a "computational cost [that] is deterministic". So does the maximal coupling for finite discrete distributions.
- **Lag:** a time lag greater than one between the chains "introduces another tuning parameter but is found to be fruitful" ([Biswas & Jacob, cited in §2.3](https://arxiv.org/abs/1708.03625)).

#### Diagnostics precedent
- **L-lag couplings** give "computable, non-asymptotic upper bound estimates for the total variation or the Wasserstein distance", with applications including "determining MCMC burn-in" ([Biswas, Jacob & Vanetti, arXiv:1905.09971](https://arxiv.org/abs/1905.09971)).

#### Prior implementations and their APIs
- **R package `unbiasedmcmc`** ([unbiasedmcmc README](https://github.com/pierrejacob/unbiasedmcmc)):
  - Kernels come as a pair: `get_mh_kernels(target, 4)` returns `$single_kernel` and `$coupled_kernel`.
  - `rinit()` returns `list(chain_state=..., current_pdf=...)`, an initial *distribution* as a no-argument sampler.
  - Running: `sample_meetingtime(single_kernel, coupled_kernel, rinit)`; `sample_coupled_chains(single_kernel, coupled_kernel, rinit, m = 2000, lag = 500)`, whose output includes `$cost`.
  - Post-processing: `H_bar(c, h = function(x) x, k = 500, m = 2000)` gives a per-replicate unbiased estimate, applied after the run. `histogram_c_chains(...)` builds signed histograms.
  - The README's interval is the CLT across replicates, `1.96 * sd / sqrt(n)`.
  - The README calls the package "not a general-purpose statistical software".
- **ColCarroll/couplings (Python)** ([ColCarroll/couplings](https://github.com/ColCarroll/couplings); [metropolis_hastings.py](https://github.com/ColCarroll/couplings/blob/master/couplings/metropolis_hastings.py); [coupled_data.py](https://github.com/ColCarroll/couplings/blob/master/couplings/coupled_data.py)):
  - "A library for exploring coupled MCMC and to support essays on it", installable via `pip install git+...`. The last commit in the read-only clone is 2019-09-19.
  - It runs `chains` pairs as one NumPy batch: `metropolis_hastings(*, log_prob, proposal_cov, init_x, init_y, lag=1, iters=1000, chains=128, short_circuit=False) -> CoupledData(x, y, x_accept, y_accept, meeting_time, lag, iters, dim, chains)`.
  - `unbiased_estimator(data, func, burn_in)` returns `(mcmc_average, bias_correction)` separately "for debugging".
  - `utils` provides `total_variation`, `wasserstein` and `plot_coupled_chains`.
  - Its own "Future work" note lists "Tuning ... adjust the number of iterations to some percentile of the meeting time".
- **ylefay/unbiased_mcmc_with_couplings:** "Implementation of a coupled Metropolis-Hasting Algorithm in Jax", a course project ([GitHub](https://github.com/ylefay/unbiased_mcmc_with_couplings)).
- **BlackJAX `coupled_hmc` (1.7.0, 2026-10-02):** an explicit-pair, shared-innovation HMC coupling with no meeting or estimators (details in Q3) ([BlackJAX releases](https://github.com/blackjax-devs/blackjax/releases); [blackjax 1.7.1](https://pypi.org/project/blackjax/1.7.1/)).
- **PyPI search:** the names `unbiasedmcmc`, `unbiased-mcmc`, `couplings`, `coupled-mcmc` and `pyunbiased` all return 404 from the PyPI JSON API ([PyPI JSON API](https://pypi.org/pypi/unbiased-mcmc/json)).

#### How existing output tooling treats signed weights
- **corner:** weighted quantiles assume a monotone cumulative weight. NumPy says non-increasing `xp` gives "meaningless" interpolation ([corner 2.3.0](https://pypi.org/project/corner/2.3.0/); [numpy.interp](https://numpy.org/doc/stable/reference/generated/numpy.interp.html)).
- **ArviZ:** weights are used only to resample in `extract` ([arviz-base 1.3.1](https://pypi.org/project/arviz-base/1.3.1/)).
- **Log weights:** nautilus returns `log_w`, which cannot encode negative weights (sibling note). pocoMC returns non-negative importance `weights` ([pocomc 1.2.6](https://pypi.org/project/pocomc/1.2.6/)).

#### Compute-budget and parallel precedents
- **Pigeons:** doubling rounds, and checkpoints that lose at most half the work ([Pigeons checkpoints.md](https://github.com/Julia-Tempering/Pigeons.jl/blob/main/docs/src/checkpoints.md)).
- **dynesty:** `maxiter`/`maxcall`, and since 2.1.3 a "Warning is emitted if maxcall/maxiter stop the iterations too early" ([dynesty CHANGELOG](https://github.com/joshspeagle/dynesty/blob/master/CHANGELOG.md)).
- **dynesty, after the fact:** merge independent runs ([dynesty quickstart](https://dynesty.readthedocs.io/en/stable/quickstart.html)).

### Inferences

#### What telescopic needs beyond emcee (requirements list)
1. **π₀ as a first-class object.** Accept:
   - a frozen `scipy.stats` distribution (`.rvs(size=..., random_state=rng)`);
   - a callable `init(rng) -> theta`;
   - helpers: `tel.init.ball(center, scale)`, `tel.init.from_prior(...)`, and `tel.init.empirical(points)`.

   `empirical(points)` is uniform over rows, e.g. the last positions of an existing emcee ensemble. It lets emcee users "warm-start" with no loss of unbiasedness, since any π₀ is valid (JOA §2.3).

   Draw X₀ and Y₀ independently by default. Record π₀ in the result. The docs should recommend trying more than one π₀ for multimodal targets (JOA).
2. **Replicates as the unit of work and randomness.**
   - One `SeedSequence` child per replicate, giving Pigeons-style parallelism invariance: identical results for serial, `pool` or any worker count.
   - Trivial resume: an unfinished replicate is regenerated from its seed.
   - `merge()` across runs with the same configuration.
3. **Meeting-aware execution.**
   - After meeting, evaluate the target once per step: the cost formula assumes one chain after τ, and ColCarroll exposes `short_circuit`.
   - Check faithfulness (X_t = Y_{t−L} stays true) in a debug mode.
   - Require deterministic log-densities, because a stochastic log-prob can split met chains (Hogg & Foreman-Mackey §9).
   - Expose `lag` (default 1).
   - A `max_iter` cap is unavoidable in practice. Pairs that never meet make the estimator biased, so the result must flag them prominently, never drop them silently.
4. **Estimators and storage modes.**
   - `store="measure"` (default): keep the atoms and weights of π̂ per replicate. Memory is about (m−k+1) + 2(τ−k−1) points per replicate. This supports post-hoc h at the chosen k, m.
   - `store="chains"`: keep X₀..X_max(τ,m) and Y. This allows post-hoc changes to k and m, and exporting X-chains to ArviZ as genuinely independent "chains".
   - `h=` streaming: constant memory, with h fixed at run time.
   - Report estimate ± SE from i.i.d. replicates (the R package uses the CLT, `1.96·sd/√R`). Optionally show the MCMC part and the bias correction separately, as ColCarroll's API does, for transparency.
5. **Signed-measure-aware summaries.**
   - Means and variances via h(x) = x, x².
   - Quantiles via the crossing rule or isotonic projection, with replicate-bootstrap intervals flagged as heuristic (JOA calls them future research).
   - Histograms via signed weights (`np.histogram` with `weights`).
   - A dedicated "Why are some weights negative?" docs page, in the spirit of corner's sigma page (sibling note).
   - Never pass signed weights to corner/ArviZ quantile code. For plots, use telescopic's own signed histograms, or explicitly labelled projections.
   - Avoid log-weights in the API, since they cannot be negative.
6. **k, m selection as a visible, two-stage procedure.**
   - `meeting_times = sampler.meeting_times(n=...)`.
   - Then defaults k = q₀.₉₅–₀.₉₉ of τ and m = 10k, with the efficiency-loss estimate 1 + (k + E[τ])/(m − k) printed.
   - Pilot replicates should be discarded, or drawn from separate seed streams, so that choosing k from τ does not make the final replicates depend on the same draws. This is my caution, not a claim from the paper.
7. **Diagnostics.**
   - Meeting-time histogram and quantiles; the fraction of replicates with τ > k; cost distribution; inefficiency (cost × variance; JOA); per-marginal acceptance rates.
   - Optional L-lag total-variation bound plots for burn-in (Biswas et al.).
   - R̂/ESS over the X-chains of different replicates as a familiar sanity check.
8. **Budgets.**
   - The default is a fixed `n_replicates`.
   - `budget_seconds=` must implement the Glynn–Heidelberger rule: every worker completes at least one replicate, and none is discarded at the deadline. The result should state the overrun. A naive "average what finished" is biased (JOA §3.3).
9. **Kernel/coupling separation.**
   - `Kernel` (marginal P) and `Coupling` (P̄, built on a kernel) are distinct objects, as in R's `single_kernel`/`coupled_kernel` and BlackJAX's per-marginal contract. A registry supplies the default coupling for each kernel type.
   - Prefer couplings with deterministic cost (reflection-maximal) for vectorised and JAX execution. Algorithm-2-style rejection couplings have heavy-tailed cost (JOA §4.1).
   - Kernel tuning (e.g. proposal covariance) must be frozen before coupled runs. Expose it as an explicit `tel.tune(...)` pilot.

#### Shape A: one-call function (progressive-disclosure layer 0)
```python
import numpy as np
from scipy import stats
import telescopic as tel

def log_prob(theta, x, y, yerr):              # unchanged from an emcee script
    ...

pi0 = stats.multivariate_normal(theta_map, 1e-2 * np.eye(3))   # or tel.init.empirical(emcee_last_positions)

res = tel.sample(
    log_prob, pi0, args=(x, y, yerr),
    parameter_names=["m", "b", "log_f"],
    n_replicates=512,                 # independent coupled pairs ("replicates", not walkers)
    kernel="rwmh",                    # default; covariance tuned by a pilot, then frozen
    k="auto", m="auto",               # pilot meeting times -> k = q99(tau), m = 10 k (reported)
    pool=pool, rng=2026,
    checkpoint="fit.h5",              # append-only; resumes if the fingerprint matches
    progress=True,
)
print(res.summary())                  # mean/sd/quantiles ± SE, tau quantiles, cost, efficiency loss
res.estimate(lambda th: th[..., 0] * th[..., 1])   # post-hoc h with i.i.d.-replicate SE
```
- **Borrows:**
  - `pm.sample`, nautilus, UltraNest and `run_sampler`-style single entry points (sibling note; [bilby 2.8.2](https://pypi.org/project/bilby/2.8.2/));
  - `'auto'` defaults that can be inspected ([dynesty.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynesty.py));
  - one-keyword checkpointing (sibling note; [ultranest 4.6.3](https://pypi.org/project/ultranest/4.6.3/));
  - a CmdStanPy-like `summary()` ([cmdstanpy 1.3.0](https://pypi.org/project/cmdstanpy/1.3.0/)).
- **Pros:** the lowest barrier for emcee users; best practice is baked in (pilot k/m, frozen tuning, per-replicate seeds); it is the easiest call to document.
- **Cons:**
  - "auto" hides a pilot whose cost and randomness must be reported.
  - Errors in π₀ or kernel tuning surface late.
  - Advanced users need layer B/C anyway.

#### Shape B: sampler object (emcee-style; the main documented API)
```python
sampler = tel.CoupledSampler(
    log_prob, pi0, ndim=3, args=(x, y, yerr),
    kernel=tel.kernels.RandomWalkMH(cov=tel.tune.covariance(log_prob, pi0, rng=1)),
    coupling=tel.couplings.ReflectionMaximal(),      # default for RandomWalkMH
    lag=1, pool=pool, vectorize=False,               # both set -> ValueError, not silent
    backend=tel.backends.HDF5("fit.h5"), rng=2026,
)
tau = sampler.meeting_times(n=200, progress=True)    # cf. unbiasedmcmc::sample_meetingtime
k = int(np.quantile(tau, 0.99)); m = 10 * k
res = sampler.run(n_replicates=500, k=k, m=m, max_iter=100 * m, store="measure")
res = sampler.run(n_replicates=500, k=k, m=m)        # same call later: extends to 1000 (resume/extend)
for rep in sampler.replicates(k=k, m=m):             # generator for custom stopping rules
    if stop_rule(rep): break                         # cf. zeus callbacks / Eryn stopping_fn
sm = res.signed_measure()                            # atoms (N, d), weights (N,), replicate (N,)
```
- **Borrows:**
  - `EnsembleSampler` with backends and the `sample()` generator ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py));
  - `run_nested` checkpoint/resume, and `merge_runs` ([dynesty sampler.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/sampler.py));
  - pilot meeting times ([unbiasedmcmc](https://github.com/pierrejacob/unbiasedmcmc));
  - callbacks ([zeus-mcmc 2.5.4](https://pypi.org/project/zeus-mcmc/2.5.4/)).
- **Pros:**
  - It is the workflow emcee users already know.
  - k and m are explicit and justified.
  - The backend makes extension and merging natural.
  - A generator lets experts write budgets and stopping rules.
- **Cons:**
  - It is stateful, which brings more invariants to test (resume with changed k/m/π₀ must be refused or handled).
  - Sequential stopping rules over replicates can bias the stopped estimator. Not researched here; see Gaps.

#### Shape C: kernel and coupling building blocks (functional core; JAX-ready)
```python
kern = tel.kernels.rwmh(log_prob, cov)            # Kernel: init(theta)->state; step(state, rng)->(state, info)
coup = tel.couplings.reflection_maximal(kern)     # Coupling: step(sx, sy, rng)->(sx, sy, met, info)
rng = np.random.default_rng(2026)
x0, y0 = pi0.rvs(size=2, random_state=rng)        # independent X0, Y0 (JOA §2)
pair = tel.run_pair(kern, coup, x0, y0, k=k, m=m, lag=1, max_iter=100 * m, rng=rng)
H = tel.estimators.h_km(pair, h, k, m)            # eq. (2.1) for one replicate
mu = tel.estimators.signed_measure(pair, k, m)    # eq. (2.2)
# Later JAX backend: kern.step(key, state) pure; vmap over replicates; deterministic-cost couplings preferred
```
- **Borrows:**
  - BlackJAX `SamplingAlgorithm(init, step)` and `coupled_hmc`'s explicit-pair contract ([blackjax 1.7.1](https://pypi.org/project/blackjax/1.7.1/));
  - NumPyro `MCMCKernel` ([numpyro 0.22.0](https://pypi.org/project/numpyro/0.22.0/));
  - the R package's `single_kernel`/`coupled_kernel`/`rinit`/`H_bar` decomposition ([unbiasedmcmc](https://github.com/pierrejacob/unbiasedmcmc));
  - ColCarroll's `CoupledData` ([couplings](https://github.com/ColCarroll/couplings/blob/master/couplings/coupled_data.py)).
- **Pros:**
  - Couplings are testable in isolation: check the marginals against P, check meeting, and check faithfulness. This fits the sibling note's "keep probability calculations separate from sampler logic" testing advice.
  - It is the cleanest route to JAX, and lets users bring their own kernels.
- **Cons:**
  - Users write the replicate loop, seeding and storage, so it is easy to break unbiasedness (e.g. adapting the kernel inside a replicate, or reusing seeds).
  - It is not an emcee-like experience.

#### Comparison and recommendation
| | A: one call | B: sampler object | C: building blocks |
|---|---|---|---|
| Audience | emcee users and newcomers | most users, scripts and notebooks | method developers, JAX backend |
| k/m choice | automatic pilot (reported) | explicit after pilot | manual |
| Checkpoint/resume | one keyword | backend object | user's job |
| Post-hoc h | yes (`res.estimate`) | yes | yes (`h_km`) |
| Risk of misuse | low | medium | high |
| Effort to implement | built on B | built on C | core |

- **Build C → B → A** as one layered package (the sibling note's progressive-disclosure finding). The docs should mirror the layers.
- **Proposed `CoupledResult` surface** (design sketch):
  - `n_replicates`, `k`, `m`, `lag`, `meeting_times`, `costs`, `pi0` description, `kernel`/`coupling` configurations, `rng` seed entropy, `truncated` (count);
  - `summary()` returning a DataFrame-like table plus a text block that explains the stopping and sizing choices;
  - `estimate(h, ci=0.95)` returning value, SE, interval and per-replicate values;
  - `signed_measure()`, `quantile(q, method="crossing"|"isotonic")`, `histogram(...)`;
  - `plot_meeting_times()`, `plot_tv_bound()` (L-lag), `plot_marginals()` (signed histograms);
  - `to_datatree()`: X-chains as `posterior` with true chain independence, plus an `unbiased` group of atoms and weights, carrying a warning attribute that ArviZ ignores the weights;
  - `save(path)`/`load(path)` in a documented HDF5/netCDF schema (not pickle);
  - `merge(other)`.

### Gaps
- **Not covered here, for other researchers:** how to couple *ensemble* moves (stretch, DE, slice) so that whole ensembles meet. This determines whether telescopic can offer an "emcee-like" kernel at all; method-landscape notes should cover it.
- **Sequential stopping:** no source was found on the bias of adaptive or sequential stopping rules over replicates, i.e. choosing R from the observed variance. Glynn–Heidelberger as summarised in JOA §3.3 covers time budgets only.
- **Memory requirements** of `store="chains"` and `store="measure"` for realistic astronomy problems were not quantified.
- **Other precedents not reviewed:**
  - the `unbiasedmcmc` vignettes and full function signatures (README only; e.g. whether `max_iterations` exists);
  - more recent coupling software, e.g. for HMC, Gibbs or particle methods, outside BlackJAX;
  - Julia or R packages beyond `unbiasedmcmc`.
- **Unconfirmed bibliographic details:** the JRSS-B volume and page numbers of Jacob, O'Leary & Atchadé (2020); the arXiv page says only "accepted as a JRSS discussion paper".

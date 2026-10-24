"""The walkthrough plan: phases and steps, with what to read and what to challenge.

Code references are `path:Symbol` (`Class.method` allowed); build.py resolves them to
lines and excerpts. Spec references are section numbers of docs/design/spec.md; theory
references are headings of docs/theory.md; decisions are DECISIONS rows (ints) or spec
decision ids ("D-5"); experiments are script names in notes/experiments/.
"""

DEPTHS = {
    "statistics": "Statistics",
    "algorithms": "Algorithms",
    "engineering": "Engineering",
    "numerics": "Numerics",
    "api": "API and docs",
    "evidence": "Evidence",
}

LEVELS = {
    "L1": "Overview: the briefing and its decisions",
    "L2": "Guided tour: the code, function by function",
    "L3": "Deep dive: derivations, line by line, experiments",
}

PHASES = [
    {
        "id": 1,
        "title": "Big picture",
        "intent": "What telescopic is for, what shaped its design, and how the pieces fit.",
        "steps": [
            {
                "id": "1.1",
                "title": "The problem, the scope and the principles",
                "level": "L1",
                "depth": ["statistics", "api"],
                "summary": (
                    "Ordinary MCMC gives consistent estimates with asymptotic error bars. "
                    "Coupled chains that meet give **unbiased** estimates of $\\pi(h)$, so "
                    "independent replicates average to estimates with honest finite-sample "
                    "error bars, and replicates run in parallel without burn-in "
                    "bookkeeping. The spec fixes the scope of v0.1: published methods only "
                    "(DECISIONS #1), a pure NumPy/SciPy core (D-1), and an emcee-style "
                    "layered API (D-2)."
                ),
                "spec": ["1", "2", "2.1", "2.2", "2.3"],
                "docs": ["README.md", "docs/index.md", "notes/CHECKPOINT.md"],
                "decisions": [1, "D-1", "D-2"],
                "challenge": [
                    "Is unbiasedness the right headline for users, or is it the error bar?",
                    "Which motivating use cases of spec section 2.2 matter most to you, and are any missing?",
                    "Is the scope line right: published methods only, new constructions as experiments?",
                ],
            },
            {
                "id": "1.2",
                "title": "The landscape census and the design review",
                "level": "L1",
                "depth": ["statistics", "evidence"],
                "summary": (
                    "Eight research notes surveyed theory, couplings by sampler family, "
                    "diagnostics and tuning, software, API design, parallelism and "
                    "validation; the landscape report distils them. An adversarial review "
                    "of the design found 38 issues, one a blocker (meetings at $t = L$ and "
                    "pairs equal but both rejecting), all accepted before any code."
                ),
                "spec": ["13", "A", "B"],
                "docs": ["reports/Unbiased MCMC with couplings landscape.md"],
                "reviews": ["design-review.md"],
                "decisions": ["D-19"],
                "challenge": [
                    "Does the census miss a method or package you consider essential?",
                    "Which of the design review's 38 issues would you have resolved differently?",
                ],
            },
            {
                "id": "1.3",
                "title": "Architecture: modules and data flow",
                "level": "L2",
                "depth": ["engineering", "api"],
                "summary": (
                    "Three layers: `sample()` and `CoupledSampler` on top; tuning, the "
                    "engine and `Result` in the middle; kernels, couplings, random streams "
                    "and `Target` underneath. A run tunes the kernel on adaptive Metropolis "
                    "pilots, runs 100 lag-1 pilot pairs to choose k, then runs the "
                    "replicates in lockstep lanes; each replicate leaves weighted atoms, a "
                    "signed measure, from which any expectation is computed afterwards."
                ),
                "spec": ["8.1", "6.2"],
                "docs": ["docs/design/implementation-plan.md"],
                "code": [
                    "src/telescopic/sampler.py:sample",
                    "src/telescopic/sampler.py:CoupledSampler.run",
                    "src/telescopic/engine.py:run",
                ],
                "decisions": ["D-2", "D-9", "D-12"],
                "challenge": [
                    "Are the layer boundaries where you would draw them?",
                    "Is storing atoms from t = k (D-12) the right default against storing per-function estimates?",
                ],
            },
            {
                "id": "1.4",
                "title": "The decision log",
                "level": "L1",
                "depth": ["evidence"],
                "summary": (
                    "Every choice another developer could make differently is recorded: 27 "
                    "design-time decisions in spec section 13 and 57 later ones in "
                    "`notes/DECISIONS.md`, each with its reason and alternatives. Skim them "
                    "here and flag any to revisit; the later phases take each in context."
                ),
                "spec": ["13"],
                "docs": ["notes/DECISIONS.md"],
                "challenge": [
                    "Which decisions would you overturn on sight?",
                    "Which need evidence you have not seen yet?",
                ],
            },
        ],
    },
    {
        "id": 2,
        "title": "Statistical foundations",
        "intent": "The estimator, its guarantees, its error bars and its bounds.",
        "steps": [
            {
                "id": "2.1",
                "title": "The telescoping estimator and signed measures",
                "level": "L3",
                "depth": ["statistics", "algorithms"],
                "summary": (
                    "With chains $X$ and $Y$ coupled at lag $L$ and meeting at "
                    "$\\tau = \\inf\\{t \\ge L : X_t = Y_{t-L}\\}$, the lag-$L$ estimator "
                    "averages $h(X_t)$ over $t \\in [k, m]$ and adds the telescoping bias "
                    "correction $\\sum_{t > k} c_t\\,[h(X_t) - h(Y_{t-L})]$. Its weights "
                    "make each replicate a signed measure whose weights sum to 1. "
                    "`correction_coefficients` counts, in constant time per $t$, the "
                    "averaged times $l \\le t - L$ with $L \\mid (t - l)$, thinning included."
                ),
                "spec": ["3.1", "3.2"],
                "theory": [
                    "The telescoping sum",
                    "The time-averaged estimator",
                    "Signed measures",
                ],
                "code": [
                    "src/telescopic/estimators.py:correction_coefficients",
                    "src/telescopic/estimators.py:averaging_times",
                    "src/telescopic/measure.py:SignedMeasure",
                    "src/telescopic/engine.py:_Batch._record",
                ],
                "tests": ["tests/test_estimators.py", "tests/test_measure.py"],
                "decisions": ["D-6", "D-7", "D-12"],
                "reviews": ["m3.md"],
                "challenge": [
                    "Check the weights against the paper's estimator for L > 1 and thinning: do the edge cases (L > m, k = 0, meeting before k) match?",
                    "Is the meeting time defined at the right index (D-6), and is the comparison at t = L right (D-7)?",
                    "Are signed measures the right public object, or should users only see estimates?",
                ],
            },
            {
                "id": "2.2",
                "title": "Choosing k, L and m",
                "level": "L3",
                "depth": ["statistics", "evidence"],
                "summary": (
                    "The automatic recipe of Atchadé and Jacob (2024): run 100 lag-1 pilot "
                    "pairs, set $k$ to the 99% quantile of $\\tau - 1$, the lag $L = k$ and "
                    "$m = 10k$. Explicit settings are kept; with $k$ given, the lag and $m$ "
                    "follow it. The cost of a replicate is $L + 2(\\tau - L) + "
                    "\\max(0, m - \\tau)$ evaluations, and the relative cost divides its "
                    "mean by $m - k + 1$."
                ),
                "spec": ["3.5"],
                "theory": ["Cost", "Choosing k, m and L"],
                "code": [
                    "src/telescopic/tuning.py:choose_k",
                    "src/telescopic/sampler.py:CoupledSampler._configure",
                ],
                "tests": ["tests/test_tuning.py"],
                "decisions": ["D-5", "D-21", "D-23", 41, 46],
                "experiments": ["e01_lag_recipe"],
                "challenge": [
                    "Is the 99% quantile of 100 pilot pairs stable enough? How much does k vary between seeds?",
                    "Is L = k the right default lag, against a smaller lag with a larger m?",
                    "Is m = 10k a good trade between bias correction and variance?",
                ],
            },
            {
                "id": "2.3",
                "title": "Error bars, covariances, quantiles and histograms",
                "level": "L3",
                "depth": ["statistics", "numerics"],
                "summary": (
                    "An estimate averages the replicates' $\\hat\\pi_r(h)$; its standard "
                    "error is their sample standard deviation over $\\sqrt R$. Covariances "
                    "use the U-statistic, unbiased where the plug-in is not; variances "
                    "take moments about the atoms' mean for precision far from the origin. "
                    "Quantiles project the signed CDF onto monotone functions; a histogram "
                    "is unbiased for a range fixed in advance."
                ),
                "spec": ["3.3", "3.4", "7.2"],
                "theory": ["Guarantees and their limits", "Estimates from many replicates"],
                "code": [
                    "src/telescopic/estimators.py:unbiased_covariance",
                    "src/telescopic/estimators.py:weighted_quantile",
                    "src/telescopic/results.py:Result.estimate",
                    "src/telescopic/results.py:Result._centred",
                ],
                "tests": ["tests/test_estimators.py", "tests/test_results.py"],
                "decisions": ["D-14", 54],
                "experiments": ["m9_coverage_reference"],
                "challenge": [
                    "Are normal intervals from R replicates adequate when the replicates are heavy-tailed?",
                    "Is the isotonic projection the right quantile estimator for signed measures? What is its bias?",
                    "Should var and std clip negative unbiased variances, or show them?",
                ],
            },
            {
                "id": "2.4",
                "title": "Distance to stationarity: TV and W1 bounds",
                "level": "L2",
                "depth": ["statistics"],
                "summary": (
                    "From the meeting times, $d_{TV}(\\pi_t, \\pi) \\le "
                    "\\mathbb E[\\max(0, \\lceil(\\tau - L - t)/L\\rceil)]$ (Biswas, Jacob "
                    "and Vanetti 2019), estimated by its average over replicates; the "
                    "W1 bound sums the recorded distances. Censored pairs make both lower "
                    "bounds of their true averages."
                ),
                "spec": ["3.6"],
                "theory": ["Distance to stationarity"],
                "code": [
                    "src/telescopic/diagnostics.py:tv_bound",
                    "src/telescopic/diagnostics.py:w1_bound",
                ],
                "tests": ["tests/test_diagnostics.py"],
                "challenge": [
                    "Is the TV bound at k the right summary to show users?",
                    "Is the W1 bound, which depends on the parametrisation, worth exposing?",
                ],
            },
        ],
    },
    {
        "id": 3,
        "title": "Couplings and kernels",
        "intent": "Each kernel, its coupling, and the invariants every coupling must keep.",
        "steps": [
            {
                "id": "3.1",
                "title": "Coupling primitives",
                "level": "L3",
                "depth": ["algorithms", "statistics"],
                "summary": (
                    "Three primitives: the reflection-maximal coupling of two Gaussians "
                    "with a common covariance, the rejection maximal coupling of any two "
                    "laws with an `eta` cap on the meeting probability (bounded variance "
                    "of the number of draws), and the maximal coupling of discrete laws. "
                    "Every kernel builds on these, so their marginals carry every "
                    "kernel's correctness."
                ),
                "spec": ["5.2"],
                "code": [
                    "src/telescopic/couplings.py:reflection_maximal",
                    "src/telescopic/couplings.py:maximal",
                    "src/telescopic/couplings.py:discrete_maximal",
                ],
                "tests": ["tests/test_couplings.py"],
                "reviews": ["m1-m2.md"],
                "challenge": [
                    "Check the reflection coupling's meeting probability and its marginals at equal means and tiny distances.",
                    "Is the eta option of Atchadé and Jacob worth its complexity for users?",
                ],
            },
            {
                "id": "3.2",
                "title": "Random-walk and independence Metropolis",
                "level": "L2",
                "depth": ["algorithms"],
                "summary": (
                    "The default kernel is random-walk Metropolis with reflection-maximal "
                    "proposals and a common uniform (D-3), tuned to the pilot covariance "
                    "and scaled by $2.38/\\sqrt d$. Independence Metropolis proposes from a "
                    "Student-$t_5$ fitted to the pilot (D-25); both chains share the "
                    "proposal and meet when both accept."
                ),
                "spec": ["5.3", "5.6"],
                "code": [
                    "src/telescopic/kernels/random_walk.py:RandomWalk.coupled_step",
                    "src/telescopic/kernels/independence.py:Independence.coupled_step",
                    "src/telescopic/kernels/independence.py:MultivariateT",
                ],
                "tests": ["tests/test_kernels.py"],
                "decisions": ["D-3", "D-25"],
                "challenge": [
                    "Is RWMH the right default when a gradient exists?",
                    "Is Student-t5 the right proposal tail for IMH, and should the weight ratio warn?",
                ],
            },
            {
                "id": "3.3",
                "title": "MALA and HMC, with HMC's meeting step",
                "level": "L3",
                "depth": ["algorithms", "numerics"],
                "summary": (
                    "MALA couples its Langevin proposals by reflection; HMC couples "
                    "trajectories with a common momentum, which contracts but never meets, "
                    "so HMC mixes in a random-walk meeting step (Heng and Jacob 2019), coupled "
                    "by reflection where they used rejection (D-11, D-24). Both tune their step by dual averaging. Gradients are evaluated "
                    "only where the log-density is finite (DECISIONS #14, #49). The second "
                    "audit found that a proposal with a non-finite gradient where the "
                    "density is finite is rejected, which restricts the target: see the "
                    "audit notes below."
                ),
                "spec": ["4.1", "5.4", "5.5"],
                "code": [
                    "src/telescopic/kernels/mala.py:MALA.coupled_step",
                    "src/telescopic/kernels/hmc.py:HMC",
                    "src/telescopic/kernels/hmc.py:_Leapfrog._trajectory",
                    "src/telescopic/kernels/base.py:evaluate_proposal",
                    "src/telescopic/tuning.py:dual_averaging",
                ],
                "tests": ["tests/test_kernels.py", "tests/test_tuning.py"],
                "decisions": ["D-11", "D-24", 11, 12, 13, 14, 17, 18, 49],
                "experiments": ["m9_hmc_meeting", "m10_mala_logistic"],
                "challenge": [
                    "Should a non-finite gradient at a point of finite density raise, be rejected, or count as zero drift?",
                    "Is a random-walk meeting step the right way to make HMC meet, against Heng and Jacob's own?",
                    "Do the dual-averaging targets (0.574 and 0.8) suit coupled chains, which need to meet?",
                ],
            },
            {
                "id": "3.4",
                "title": "Gibbs couplings",
                "level": "L3",
                "depth": ["algorithms"],
                "summary": (
                    "Gibbs sweeps user-written full conditionals block by block. Coupled "
                    "chains draw each block from a maximal coupling, from common random "
                    "numbers (a uniform through `ppf`, or common normals through a square "
                    "root of a normal conditional's covariance), or from the two-scale "
                    "rule of Biswas et al. (2022). Where the chains agree outside a block, "
                    "both get the same draw."
                ),
                "spec": ["5.7"],
                "code": [
                    "src/telescopic/kernels/gibbs.py:Gibbs.coupled_step",
                    "src/telescopic/kernels/gibbs.py:_common",
                    "src/telescopic/_linalg.py:square_root",
                ],
                "tests": ["tests/test_kernels.py"],
                "decisions": [15, 16, 48],
                "docs": ["docs/tutorials/gibbs.md"],
                "challenge": [
                    "Common random numbers treat any conditional with mean and cov as normal: keep the documented contract, or draw through generators in the same state (QUESTIONS)?",
                    "Is the two-scale threshold a usable setting without guidance?",
                ],
            },
            {
                "id": "3.5",
                "title": "Mixtures, the kernel contract and kernels of your own",
                "level": "L2",
                "depth": ["algorithms", "api"],
                "summary": (
                    "Every kernel implements `step` and `coupled_step` on batches of rows, "
                    "with arithmetic that never depends on the other rows. A mixture "
                    "chooses one component per row from a common uniform. "
                    "`check_coupling` tests a user's coupling for its marginals, its "
                    "meetings and its cached evaluations."
                ),
                "spec": ["5.1", "5.8", "9"],
                "code": [
                    "src/telescopic/kernels/base.py:Kernel",
                    "src/telescopic/kernels/mixture.py:Mixture.coupled_step",
                    "src/telescopic/testing.py:check_coupling",
                ],
                "tests": ["tests/test_kernels.py", "tests/test_testing.py"],
                "docs": ["docs/guide/custom-kernels.md"],
                "decisions": [10, 19, 25, 28, "D-20"],
                "challenge": [
                    "Is the kernel contract small enough for users to implement correctly?",
                    "Does check_coupling catch the mistakes users will make?",
                ],
            },
        ],
    },
    {
        "id": 4,
        "title": "Tuning and the automatic pipeline",
        "intent": "How sample() goes from a log-density to a run, and where it fails.",
        "steps": [
            {
                "id": "4.1",
                "title": "Adaptive Metropolis pilots",
                "level": "L3",
                "depth": ["algorithms", "numerics", "evidence"],
                "summary": (
                    "Four adaptive Metropolis chains pool their mean and covariance "
                    "(Haario et al. 2001; Craiu, Rosenthal and Yang 2009) and adapt their "
                    "own scales toward acceptance 0.234, for $\\max(2000, 200d, 10d^2)$ "
                    "steps, thinned beyond 20 dimensions. The tuned kernel is then frozen "
                    "(D-4). Ridges scale with each coordinate's variance, so tuning does "
                    "not depend on the coordinates' units."
                ),
                "spec": ["5.3"],
                "code": [
                    "src/telescopic/tuning.py:adaptive_metropolis",
                    "src/telescopic/tuning.py:PilotRun.cov",
                ],
                "tests": ["tests/test_tuning.py"],
                "decisions": ["D-4", 5, 6, 7, 34, 47],
                "experiments": [
                    "m9_tuning_dimension",
                    "m9_pilot_length",
                    "m9_pilot_length_2",
                    "m9_pilot_length_3",
                    "m11_scales",
                ],
                "challenge": [
                    "Is a pilot of 10 d^2 steps acceptable at d = 100 (about 22 s here)?",
                    "Is R-hat above sqrt 2 a useful multimodality warning?",
                ],
            },
            {
                "id": "4.2",
                "title": "The sampler: settings, runs and the pilot",
                "level": "L2",
                "depth": ["api", "engineering"],
                "summary": (
                    "`CoupledSampler` tunes once, runs the 100-pair pilot when k is "
                    "automatic, fixes k, m, lag, thin and max_iter at its first run and "
                    "keeps them; later runs add replicates. Settings are checked before "
                    "any tuning where their bounds allow; a memory estimate warns above "
                    "1 GB."
                ),
                "spec": ["8.1"],
                "code": [
                    "src/telescopic/sampler.py:CoupledSampler",
                    "src/telescopic/sampler.py:CoupledSampler._check",
                    "src/telescopic/sampler.py:CoupledSampler._check_memory",
                ],
                "tests": ["tests/test_sampler.py"],
                "docs": ["docs/guide/settings.md"],
                "decisions": [4, 8, 44, 46, "D-21"],
                "challenge": [
                    "Is keeping the first run's settings the behaviour users expect?",
                    "Is 100 pilot pairs the right default?",
                ],
            },
            {
                "id": "4.3",
                "title": "Where the pipeline fails",
                "level": "L2",
                "depth": ["evidence", "statistics"],
                "summary": (
                    "Two known failures. Tuned toward their usual acceptance rates, HMC "
                    "and MALA rarely meet on the banana, and MALA fails on a logistic "
                    "regression started wide; `sample` then raises with advice. From a "
                    "start a hundred times wider than the posterior, about one replicate "
                    "in a hundred is averaged while still on its way, and variance "
                    "estimates turn heavy-tailed; the variance warning catches it."
                ),
                "spec": ["4.2"],
                "docs": ["docs/tutorials/failure-modes.md", "notes/QUESTIONS.md"],
                "decisions": [35],
                "experiments": ["m9_hmc_meeting", "m10_mala_logistic", "m9_initial_width"],
                "challenge": [
                    "Should automatic steps for HMC and MALA be chosen from pilot meeting times (a new construction)?",
                    "Should the start be built from the tuning pilot when the user's is too wide?",
                ],
            },
        ],
    },
    {
        "id": 5,
        "title": "Engine and execution",
        "intent": "Lockstep lanes, random streams, pools and interruptions.",
        "steps": [
            {
                "id": "5.1",
                "title": "Lockstep lanes, refill and exact meetings",
                "level": "L3",
                "depth": ["engineering", "algorithms"],
                "summary": (
                    "The engine advances up to 256 replicates together as rows of one "
                    "batch, refills a lane when its replicate finishes, detects meetings "
                    "by exact equality of positions, and censors pairs at `max_iter` if "
                    "one is set. Each replicate's atoms, weights, costs and distances are "
                    "recorded as it runs."
                ),
                "spec": ["6.2", "6.3"],
                "code": [
                    "src/telescopic/engine.py:run",
                    "src/telescopic/engine.py:_Batch.advance",
                    "src/telescopic/engine.py:_Batch.finish",
                    "src/telescopic/engine.py:Replicates",
                ],
                "tests": ["tests/test_engine.py"],
                "decisions": ["D-7", "D-9", "D-13", 52],
                "experiments": ["m11_lanes"],
                "reviews": ["m3.md"],
                "challenge": [
                    "Is exact equality the right meeting test for every kernel, including Gibbs with continuous draws?",
                    "Should a cap be the default, given that an unmet pair holds a lane forever?",
                ],
            },
            {
                "id": "5.2",
                "title": "Random streams and reproducibility",
                "level": "L3",
                "depth": ["engineering", "numerics"],
                "summary": (
                    "Each replicate reads normals and uniforms in order from two child "
                    "generators of `(root, namespace, index)` (PCG64DXSM), buffered for "
                    "lockstep batches, so a replicate's numbers do not depend on the "
                    "lanes, the chunks or the workers. Kernel arithmetic is "
                    "row-independent (`einsum` without optimisation), so results are "
                    "bitwise identical for log-densities evaluated point by point."
                ),
                "spec": ["6.1"],
                "code": [
                    "src/telescopic/_random.py:root_seed",
                    "src/telescopic/_random.py:Streams",
                    "src/telescopic/_random.py:Draws",
                    "src/telescopic/_linalg.py:matvec",
                ],
                "tests": ["tests/test_random.py"],
                "decisions": ["D-8", "D-20", "D-26"],
                "reviews": ["m0.md"],
                "challenge": [
                    "Is bitwise reproducibility across lanes worth its 2.2 times cost in random numbers?",
                    "Vectorised log-densities agree only statistically across layouts: is that acceptable?",
                ],
            },
            {
                "id": "5.3",
                "title": "Pools, chunks and Ctrl-C",
                "level": "L2",
                "depth": ["engineering"],
                "summary": (
                    "`pool=` takes a number of workers, an executor or any object with "
                    "`map`. Chunks of up to 256 replicates, at most four per worker, run "
                    "in workers that live for one call. A first Ctrl-C lets started "
                    "replicates finish and returns a result flagged interrupted, which "
                    "the next run completes; a second stops at once."
                ),
                "spec": ["6.4", "6.5"],
                "code": [
                    "src/telescopic/_parallel.py:execute",
                    "src/telescopic/_parallel.py:_gather",
                    "src/telescopic/_parallel.py:Interrupt",
                ],
                "tests": ["tests/test_parallel.py"],
                "docs": ["docs/guide/parallel.md"],
                "decisions": [20, 21, 23, 45, 52, "D-22", "D-27"],
                "experiments": ["m11_chunks"],
                "reviews": ["m7.md"],
                "challenge": [
                    "Is the size guess for pools that do not report one acceptable (QUESTIONS)?",
                    "Is the interrupted-run contract clear to users?",
                ],
            },
            {
                "id": "5.4",
                "title": "Memory and cost",
                "level": "L2",
                "depth": ["engineering", "evidence"],
                "summary": (
                    "Storage grows with the averaged atoms, the late pairs' corrections "
                    "and the distances; the sampler estimates it from the pilot and warns "
                    "above 1 GB. The benchmarks measure lockstep cost, streams, pools and "
                    "the pipeline against spec section 10."
                ),
                "spec": ["6.6", "10"],
                "code": ["src/telescopic/sampler.py:CoupledSampler._check_memory"],
                "docs": ["benchmarks/run.py"],
                "experiments": ["m11_t_memory"],
                "challenge": [
                    "Which missed performance target matters most to you?",
                ],
            },
        ],
    },
    {
        "id": 6,
        "title": "Results, diagnostics and warnings",
        "intent": "What users get back, and how the package tells them it went wrong.",
        "steps": [
            {
                "id": "6.1",
                "title": "Result: estimates, summary, saving and merging",
                "level": "L2",
                "depth": ["api", "statistics"],
                "summary": (
                    "`Result` holds the replicates and computes estimates, intervals, "
                    "covariances, quantiles, histograms, bounds and a summary. It saves "
                    "to a versioned `.npz` without pickle, and merges runs of one sampler "
                    "with disjoint replicates."
                ),
                "spec": ["7.1", "7.2", "7.3"],
                "code": [
                    "src/telescopic/results.py:Result",
                    "src/telescopic/results.py:Result.summary",
                    "src/telescopic/results.py:Result.merge",
                    "src/telescopic/_io.py:save",
                ],
                "tests": ["tests/test_results.py", "tests/test_io.py"],
                "docs": ["docs/guide/results.md"],
                "decisions": [9, 26, 29, 30, 40, 53],
                "challenge": [
                    "Is the Result API the right size? What would you remove?",
                    "Should merge check kernels and initial distributions (DECISIONS #53 says no)?",
                ],
            },
            {
                "id": "6.2",
                "title": "The six warnings and their calibration",
                "level": "L3",
                "depth": ["statistics", "evidence"],
                "summary": (
                    "Censored pairs; late pairs above a quarter; one replicate holding more "
                    "than half the variance (about the atoms' mean, R at least 100); a Hill "
                    "tail index of $\\tau - L + 1$ below 2 (R at least 500, undefined when "
                    "ties make the smallest value the threshold); disagreeing tuning "
                    "chains; slow pairs. Each threshold was set from measured false-alarm "
                    "rates on clean and failing runs."
                ),
                "spec": ["7.4"],
                "code": [
                    "src/telescopic/diagnostics.py:issues",
                    "src/telescopic/diagnostics.py:hill",
                    "src/telescopic/diagnostics.py:variance_share",
                ],
                "tests": ["tests/test_diagnostics.py"],
                "docs": ["docs/guide/diagnostics.md"],
                "decisions": ["D-16", 6, 36, 54, 55, 56, 57],
                "experiments": [
                    "m9_warnings",
                    "m9_variance_warning",
                    "m11_variance_share",
                    "m11_hill_ties",
                ],
                "challenge": [
                    "The variance warning fires in 37 of 200 banana runs: is a half still the right threshold (QUESTIONS)?",
                    "The tail warning is blind behind 90% ties: acceptable for v0.1?",
                    "A run that never left one mode cannot be detected: is the documentation of that limit enough?",
                ],
            },
            {
                "id": "6.3",
                "title": "Plotting",
                "level": "L1",
                "depth": ["api"],
                "summary": (
                    "Meeting-time survival functions, the TV bound, signed marginals and "
                    "corner plots, on new or given axes; matplotlib is optional."
                ),
                "spec": ["7.5"],
                "code": [
                    "src/telescopic/plotting.py:meeting_times",
                    "src/telescopic/plotting.py:corner",
                ],
                "tests": ["tests/test_plotting.py"],
                "decisions": [30],
                "challenge": ["Which plot would you look at first after a run?"],
            },
        ],
    },
    {
        "id": 7,
        "title": "Numerics and robustness",
        "intent": "Precision, special values, tolerances and argument checks.",
        "steps": [
            {
                "id": "7.1",
                "title": "Special values: NaN, -inf and gradients",
                "level": "L3",
                "depth": ["numerics", "statistics"],
                "summary": (
                    "float64 throughout; accept and reject in log space. A NaN or +inf "
                    "log-density raises with the point; -inf is a rejection. Where the "
                    "log-density is -inf the gradient is not evaluated and is 0, so a "
                    "gradient kernel proposes from there without drift and stays "
                    "invariant (DECISIONS #14, #49)."
                ),
                "spec": ["4.1"],
                "code": [
                    "src/telescopic/target.py:Target.log_prob_and_grad",
                    "src/telescopic/kernels/base.py:evaluate",
                    "src/telescopic/kernels/base.py:log_ratio",
                ],
                "tests": ["tests/test_target.py", "tests/test_kernels.py"],
                "decisions": [11, 14, 49],
                "challenge": [
                    "Is raising on a NaN log-density right inside a long run, which then loses its work?",
                ],
            },
            {
                "id": "7.2",
                "title": "Scale-free tolerances and square roots",
                "level": "L3",
                "depth": ["numerics"],
                "summary": (
                    "Ridges proportional to each coordinate's variance; a symmetry test "
                    "relative to the diagonal; square roots of singular covariances from "
                    "the eigendecomposition of the correlation matrix. Tuning is identical "
                    "at coordinate scale ratios from 1 to 1e16."
                ),
                "code": [
                    "src/telescopic/_linalg.py:covariance",
                    "src/telescopic/_linalg.py:square_root",
                    "src/telescopic/_linalg.py:_symmetric",
                ],
                "tests": ["tests/test_kernels.py", "tests/test_tuning.py"],
                "decisions": [47, 56],
                "experiments": ["m11_scales"],
                "challenge": [
                    "Are the constants (1e-10, 1e-12, 1e-8) defensible, or should they be derived?"
                ],
            },
            {
                "id": "7.3",
                "title": "Argument checks and errors",
                "level": "L1",
                "depth": ["api", "engineering"],
                "summary": (
                    "Every public number argument raises `TypeError` for a wrong type and "
                    "`ValueError` out of bounds, naming the argument; integers through "
                    "`operator.index`, so floats and bools are refused. Public functions "
                    "never modify their inputs."
                ),
                "spec": ["8.2"],
                "code": ["src/telescopic/_types.py:integer", "src/telescopic/_types.py:real"],
                "decisions": [44, 50],
                "challenge": ["Is refusing integral floats (k=5.0) too strict for users?"],
            },
        ],
    },
    {
        "id": 8,
        "title": "API, documentation and UX",
        "intent": "What users call, read and see.",
        "steps": [
            {
                "id": "8.1",
                "title": "Entry points, names and defaults",
                "level": "L2",
                "depth": ["api"],
                "summary": (
                    "`telescopic.sample(log_prob, initial, n_replicates, ...)` for the "
                    "automatic path; `CoupledSampler` to keep a sampler; kernels, "
                    "couplings, the engine and estimators as building blocks. Names follow "
                    "the literature (`k`, `m`, `lag`; D-15)."
                ),
                "spec": ["8", "8.1", "8.2", "8.3"],
                "code": ["src/telescopic/sampler.py:sample", "src/telescopic/__init__.py"],
                "docs": ["docs/getting-started.md", "docs/api.md"],
                "decisions": ["D-15", 39],
                "challenge": [
                    "Would you rename anything before v0.1 freezes it?",
                    "Are the defaults the ones you would want as a user?",
                ],
            },
            {
                "id": "8.2",
                "title": "The documentation site",
                "level": "L1",
                "depth": ["api"],
                "summary": (
                    "A theory page, Getting started, a seven-page user guide, six "
                    "tutorials executed at every build with hidden asserts behind their "
                    "conclusions, an API reference and a citing page; the strict build "
                    "checks every reference."
                ),
                "docs": [
                    "docs/theory.md",
                    "docs/guide/index.md",
                    "docs/tutorials/index.md",
                    "docs/tutorials/how-it-works.md",
                ],
                "decisions": [37, 42, 43],
                "reviews": ["m10.md"],
                "challenge": [
                    "What would a new user miss in the first ten minutes?",
                    "Is the theory page at the right level for the audience?",
                ],
            },
        ],
    },
    {
        "id": 9,
        "title": "Testing and evidence",
        "intent": "Why we believe it works, and how fast it is.",
        "steps": [
            {
                "id": "9.1",
                "title": "Test strategy",
                "level": "L1",
                "depth": ["evidence", "engineering"],
                "summary": (
                    "A fast suite of about 550 tests in under a minute, with warnings as "
                    "errors, and a slow statistical suite of 18 tests. Every coupling is "
                    "tested for its marginals against the uncoupled kernel and for "
                    "faithfulness; tolerances come from Monte Carlo error (pass at "
                    "$|z| \\le 4$, rerun up to 6). A mutation audit measured the tests' "
                    "power."
                ),
                "docs": ["docs/design/test-plan.md", "CONTRIBUTING.md", "tests/references.py"],
                "decisions": [33, 51],
                "reviews": ["m11.md"],
                "challenge": ["Which property is least tested, in your view?"],
            },
            {
                "id": "9.2",
                "title": "Validation results",
                "level": "L2",
                "depth": ["evidence", "statistics"],
                "summary": (
                    "Unbiasedness of every kernel family, each case with enough replicates "
                    "that removing the correction shows at $|z| \\ge 5$; interval coverage against an "
                    "independent reference; published meeting times reproduced (O'Leary, "
                    "Jacob et al., Heng and Jacob); simulation-based calibration on two "
                    "conjugate models."
                ),
                "spec": ["9", "A"],
                "tests": ["tests/test_statistics.py"],
                "experiments": ["m9_unbiasedness", "m9_coverage_reference"],
                "reviews": ["m9.md"],
                "challenge": ["Which published result would convince you most, and is it here?"],
            },
            {
                "id": "9.3",
                "title": "Performance",
                "level": "L1",
                "depth": ["evidence", "engineering"],
                "summary": (
                    "Lockstep coupled RWMH at 1.30 µs per pair-step (target 2); random "
                    "streams within target; a 1 ms log-density 3.89 times faster on 4 "
                    "started workers. Missed: point-by-point overhead (90 µs against 40), "
                    "a pool's speed-up with its start (2.28 against 3.5) and the "
                    "pipeline's pilots (43 to 63% against 15%)."
                ),
                "spec": ["10"],
                "docs": ["benchmarks/run.py", "benchmarks/results/20b262b.json"],
                "experiments": ["m11_lanes", "m11_chunks"],
                "challenge": ["Which target should v0.2 meet first?"],
            },
        ],
    },
    {
        "id": 10,
        "title": "Open questions and the road ahead",
        "intent": "What is still undecided, and what was deferred.",
        "steps": [
            {
                "id": "10.1",
                "title": "Open questions",
                "level": "L1",
                "depth": ["api", "statistics"],
                "summary": (
                    "Eight questions wait on you, each with the default the work follows: "
                    "licence, publicity, budget mode, pool sizes, automatic HMC and MALA "
                    "steps, starts far from the target, the variance threshold, and "
                    "common random numbers for conditionals that are not normal."
                ),
                "docs": ["notes/QUESTIONS.md"],
                "challenge": ["Answer, or keep the default, for each."],
            },
            {
                "id": "10.2",
                "title": "Deferred work and the roadmap",
                "level": "L1",
                "depth": ["api"],
                "summary": (
                    "Deferred beyond v0.1: budget mode, bootstrap uncertainty for "
                    "quantiles, R-hat across replicates, reprs, histograms in memory "
                    "proportional to the atoms, evaluation counts with tuning, derived "
                    "quantities in quantile and histogram. The spec's roadmap lists JAX "
                    "and new couplings."
                ),
                "spec": ["12", "3.7"],
                "docs": ["notes/reviews/m11.md"],
                "decisions": [22, 38, "D-10"],
                "challenge": ["What must v0.2 have?"],
            },
        ],
    },
]

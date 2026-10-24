# telescopic: test plan (v0.1)

This plan turns the requirements of [`spec.md`](spec.md) into tests. It has three goals:

1. Every public behaviour is covered by a fast, deterministic test.
2. Every coupling and every estimator is shown to be correct against exact references, not only by statistical agreement.
3. The documentation and tutorials stay executable.

Most estimator bugs cannot be caught statistically. In the census, an off-by-one in the bias-correction weights shifted the estimate by 0.0025 standard deviations; detecting it at 5σ would take about 4 million replicates (`research_notes/.../validation_benchmarks.md`, Q2). The deterministic tier therefore carries correctness, and the statistical tier confirms it end to end.

## 1. Principles

- **Deterministic.** Every random test passes a fixed seed through `rng`, and seeds are never chosen to make a test pass.
- **One tolerance rule, sequential escalation**, for every test that asserts a $z$-score:
  - pass if $|z| \le 4$;
  - if $4 < |z| \le 6$, rerun once with 4× the replicates and a seed derived from the first, and pass if and only if the rerun gives $|z| \le 4$;
  - fail if $|z| > 6$.

  Heavy tails make a plain 4σ or 5σ rule unreliable: the census measured $\mathbb P(|z| > 4) = 0.005$ at $R = 5000$, about 80× the Gaussian rate. Tests of p-values (KS, binomial) use a fixed threshold that keeps each module's family-wise false-positive rate below about $10^{-3}$.
- **Exact references first.** In order of preference:
  1. draw-for-draw oracles with injected random numbers;
  2. closed forms (Gaussian total-variation distances, meeting probabilities);
  3. finite-state computations;
  4. statistical comparisons.
- **Two tiers.**
  - The default `pytest` runs the fast tier in under 2 minutes.
  - `pytest -m slow` runs the statistical tier: unbiasedness at scale, coverage, total-variation bounds and calibration of the warnings.
- **Warnings are errors** (`filterwarnings = error` in `pyproject.toml`). Tests that expect a warning use `pytest.warns`.
- **Test names** are sentences that state the behaviour. Docstrings cite requirement IDs where they apply.
- **A test earns its place** by catching a plausible bug that no other test catches (`CLAUDE.md`).

## 2. Test inventory

### 2.1 `tests/test_random.py` (R-EXEC-1, R-EXEC-2)

- `root_seed` accepts `None`, an integer, a `SeedSequence` and a `Generator`. The same integer gives the same streams, and different namespaces give different streams.
- `replicate_generator(root, ns, i)` equals the generator built from `SeedSequence(root.entropy, spawn_key=root.spawn_key + (ns, i))`.
- `Streams`/`Draws` (Hypothesis, over random sequences of requests from random subsets of lanes, with lanes reassigned in between): every row of every request equals the next numbers of its replicate's two child generators (§6.1), whatever the buffer width, the other lanes' requests and the order of lanes. `subset` keeps the row-to-lane mapping.
- `fixed_draws` replays its arrays in order and raises once a row's numbers run out.

### 2.2 `tests/test_couplings.py` (R-CPL-1–3)

**Reflection-maximal**
- **Oracle:** with injected normals and uniforms, `reflection_maximal` equals an independent NumPy reference at `atol=1e-12`. Uniforms are chosen on both sides of the meeting threshold so both branches run.
- **Algebra (Hypothesis):**
  - the reflection is an involution and preserves norms;
  - `identical` holds if and only if the outputs are bitwise equal;
  - $\delta = 0$ always gives `identical`, without a warning.
- **Meeting probability:** a binomial test of the frequency of bitwise-equal outputs (not of the `identical` flag, which a recomputing coupling still sets correctly) against $2\Phi(-\lVert\delta\rVert/2)$. Use $n = 2\times10^5$ per case for three distances in $d = 1, 3, 10$, with a dense $\Sigma$. This is the test that catches a coupling that recomputes $Y$ (spec Appendix A, E-R2).
- **Marginal law:** $Y$'s whitened coordinates are $\mathcal N(0, I)$ (one-sample KS per coordinate, and KS of $\lVert\cdot\rVert^2$ against $\chi^2_d$, Bonferroni-corrected). The same holds for $X$.

**Rejection maximal**
- **Oracle:**
  - a draw-for-draw reference for $\eta = 1$ and $\eta < 1$;
  - from $p = q$, the output is `identical` for $\eta = 1$;
  - the draw count follows the algorithm.
- **Statistics, for Gaussians with unequal means and variances:**
  - the meeting frequency equals $\int\min(\eta p, q)$ (binomial test);
  - both marginals are exact (one-sample KS);
  - the mean draw count is 2.

**Discrete maximal**
- The joint frequencies of the outputs match the exact joint table (diagonal $\min(p_i, q_i)$, off-diagonal the product of the normalised residuals times $\mathrm{TV}$), computed independently in the test, by a chi-square test.
- Log-space inputs with very peaked probabilities do not underflow.

### 2.3 `tests/test_target.py`, `tests/test_initial.py` (R-INPUT-1, R-INPUT-2)

**Target**
- **Batching:** a vectorised and a point-by-point target agree.
- **Invalid values:**
  - NaN or `+inf` raises `ValueError` naming the point;
  - `-inf` passes through.
- **Wrappers:** `args` and `kwargs` are forwarded. Counters count evaluations. Input arrays are never modified.
- **Gradients:** `log_prob_and_grad` without a gradient raises.

**Initial distributions**
- **Each input form:**
  - callables, SciPy frozen distributions and point arrays all give draws of shape `(d,)`;
  - point arrays are sampled uniformly over their rows (chi-square test on row counts).
- **Validation:**
  - wrong shapes or non-finite draws raise;
  - a draw with $\log\pi = -\infty$ emits `InputWarning`.
- **Independence:** $X_0$ and $Y_0$ are drawn independently from the replicate's stream.

### 2.4 `tests/test_kernels.py` (R-KER-*)

Each kernel ($\times$ each coupling it offers) gets the same suite, via a parametrised fixture:

1. **Oracle draw-for-draw:** with `fixed_draws`, `step` and each marginal of `coupled_step` equal an independent NumPy reference kernel at `atol=1e-12`.
   - Pairs are far apart, close, and equal.
   - Uniforms are chosen to exercise both acceptance and rejection.
   - For mixtures, both components are exercised.
2. **Marginal preservation:** from fixed pairs (equal; $0.1\sigma$ apart; $3\sigma$ apart), the empirical law of each marginal after one coupled step equals the exact one-step law. For RWMH, MALA and IMH on a Gaussian target this is:
   - a binomial test of the probability of staying;
   - a KS test of moved values against the normalised $q(x' \mid x)\,\alpha(x, x')$, by one-dimensional quadrature, in $d = 1$ only.

   Moved values are accepted proposals, whose law is not Gaussian, and the stay probability in $d > 1$ is a $d$-dimensional integral. So for $d > 1$ each coupled marginal is compared with `step` from the same state by two-sample tests: a two-proportion test of the stay frequency, and KS tests of the moved values on random projections and on the norm, with a larger $n$ (design review D-R8). One-sample tests against the exact law are about 10× more sample-efficient than two-sample tests (census). These tests run with $n = 2\times10^4$ in the fast tier and $2\times10^5$ in the slow tier.
3. **Faithfulness:** from equal states, 100 coupled steps keep the states bitwise equal, with equal log-densities and gradients.
4. **Batch invariance (R-KER-6):** a step and a coupled step on a batch of 1 equal the same rows in batches of 2 and 64, bitwise, at $d \ge 5$ with a dense $\Sigma$.
5. **Finite-state exactness** (in `tests/test_couplings.py`): `discrete_maximal` couples the rows of a transition matrix $P$ on 4 states, all 16 pairs of states in one batch. The empirical coupled transitions match the exact coupled transition matrix, computed independently in the test, whose marginals equal $P$'s rows and whose diagonal is absorbing.

Kernel-specific tests:
- **RWMH:**
  - `"reflection"` meets more often than `"maximal"` at $d = 10$ (O'Leary 2021 Table 1 reference: about 30 vs 279 mean meeting time, checked loosely in the slow tier);
  - draws per step as documented.
- **MALA:** the acceptance ratio includes both proposal densities. Checked against a reference for a non-Gaussian target, where omitting the reverse term changes the answer.
- **HMC:**
  - trajectories equal an independent leapfrog implementation draw for draw, and on $\mathcal N(0, 1)$, where the leapfrog map is linear, each chain's one-step law matches the exact one, which holds only for a reversible, volume-preserving integrator with the right acceptance;
  - shared momentum contracts two chains on a Gaussian;
  - the meeting component meets with the probability of §5.2;
  - a common uniform chooses the same component for both chains.
- **IMH:** both chains propose the same point, and the pair meets when both accept.
- **Gibbs:**
  - `"maximal"` couplings of conditionals have exact marginals (KS against the conditional's CDF);
  - `"crn"` uses a common uniform through `ppf`;
  - `"two-scale"` switches at the threshold, and requires one;
  - when the coordinates outside a block are equal, both chains receive the same draw, so equal states stay equal with $\eta < 1$.
- **Mixture:** the component frequencies match the weights. Component choice does not depend on the states (structural test with `fixed_draws`).
- **Draw counts (R-KER-5):** each kernel consumes the random numbers it documents, checked with `fixed_draws` arrays of exactly that size.

### 2.5 `tests/test_estimators.py` (R-EST-4–8)

**Algebraic identities (Hypothesis).** Paths are synthetic: random arrays $X_{0..T}$ and $Y_{0..T-L}$ with a forced meeting time $\tau$, and random $0 \le k \le m$, $L \ge 1$, `thin` $\ge 1$. On them:
- The weighted sum of §3.2 equals the definitional average of $H_l$ over the averaging set, to `1e-12`. This is the main guard against weight and index bugs.
- The weights of the signed measure sum to 1, and $\hat\pi(h)$ equals $H$ for several $h$ (in 2.6, on recorded paths).
- The cost equals $L + 2(\tau - L) + \max(0, m - \tau)$ (in 2.6).

**Derived estimators**
- `unbiased_covariance` matches the U-statistic $\frac{1}{R(R-1)}\sum_{r \ne s} a_r a_s^\top$ form exactly, which is unbiased by construction.
- `weighted_quantile`:
  - reproduces NumPy's inverted-CDF quantiles for positive integer weights, through repeated values (NumPy's `weights=` needs NumPy 2.0, newer than the oldest supported stack);
  - is monotone in $q$;
  - returns finite answers for non-monotone signed CDFs.
- Histograms (M4, through `per_replicate` of bin indicators) use half-open bins, and their bin masses sum to the total weight inside the range.

**Exact expectation.**
- Fast tier: with $\mathbb E\,h(X_t) = a_t$ converging geometrically, $|S|^{-1}\sum_{l\in S} a_l + \sum_t c_t (a_t - a_{t-L})$, computed with the package's `correction_coefficients`, equals $\lim a_t$ to $10^{-12}$ for random $(k, m, L, \text{thin})$. Since a faithful coupling's marginals give exactly this expectation, this is the unbiasedness of the weights, checked against the truth rather than a reference. (A 4-state chain is one such sequence; the recursion on its joint law adds nothing.)
- Slow tier: the engine's average over $10^6$ replicates matches it (sequential escalation).

### 2.6 `tests/test_engine.py` (R-EST-1–3, R-EST-7, R-EXEC-3, R-EXEC-4)

- **Estimates against recorded paths:** with `keep_paths=True`, every replicate's stored signed measure equals the one computed by a reference implementation of §3.2 from the recorded paths. Exact to `1e-12`, for random $(k, m, L, \text{thin})$.
- **Lag:** $X$ runs exactly $L$ steps alone before the first coupled step.
- **Meeting:**
  - $\tau$ is the first $t \ge L$ with $X_t = Y_{t-L}$, including $\tau = L$;
  - after the meeting only $X$ is simulated, so a kernel that would split equal states cannot affect the result;
  - the cost formula holds;
  - distances are recorded for $L \le t < \tau$.
- **Refill:** with $R > B$ lanes, every requested index is run exactly once, and per-replicate results do not depend on $B$ (1, 7, 64) for a point-by-point target, bitwise. This is the invariance contract of R-EXEC-2, extended to a kernel with variable draw counts (RWMH `"maximal"`), and to MALA and HMC in M6.
- **Censoring:**
  - with `max_iter`, unmet pairs are returned as censored, never dropped, and `max_iter` $\le \max(L, m)$ raises;
  - a pair past the slow-pair threshold triggers the one-time `ConvergenceWarning`.
- **Pilot mode:** `k=0, m=0, store=False` returns meeting times only.
- **Memory:** the stored atom count stays within the bound of §3.2 (R-EXEC-7).

### 2.7 `tests/test_results.py`, `tests/test_measure.py` (R-OUT-1–3)

- **Pooled estimates:** `estimate(h)` equals the mean of `measure.per_replicate(h)`, with standard error `std/√R` and a CLT interval at the requested level. Vector-valued `h` works.
- **Moments:** `mean`, `cov`, `var` and `std` follow §3.4. `cov` is symmetric.
- **Quantiles:** `quantile` and `interval` use the isotonic rule.
- **Histograms:** `histogram` returns masses, standard errors and edges.
- **Summary:** `summary()` contains the fields of §7.3 and converts with `to_dict()`. `to_pandas()` is skipped without pandas.
- **Merging:** `merge` rejects different configurations and overlapping indices, and its output equals one run over the union of indices.
- **Censored replicates** are excluded from estimates and from $R$, and listed with meeting time `inf`.
- **Output shapes:** an `h` whose output is not `(n,)` or `(n, p)` raises.
- **Interrupted results** say in `summary()` that their estimates are consistent, not exactly unbiased.

### 2.8 `tests/test_diagnostics.py` (R-DIAG-1, R-DIAG-2)

- **TV bound:**
  - the formula equals $\sum_{j\ge1}\hat{\mathbb P}(\tau > t + jL)$ (identity on random meeting times);
  - it is reported as $\min(1, \cdot)$ with an upper confidence bound at 0;
  - with censored replicates it is reported as a lower bound, with their count.
- **Exact reference:** for a finite chain, the bound computed from the exact meeting-time distribution is at least the exact TV distance at every $t$.
- **W1 bound (should):** equals its definition on recorded distances.
- **Hill tail index:** with $k_h = n/10$, recovers the index of Pareto samples within $4\alpha/\sqrt{k_h}$ at $n = 10^4$ (with $k_h = \sqrt n$ the estimate is within 10% only 69% of the time; design review D-R17).
- **`check` warnings:** each fires on a synthetic result built to trigger it:
  - a censored replicate;
  - more than 25% bias corrections;
  - one replicate holding more than half the variance at $R \ge 100$, and not one holding exactly half;
  - a tail index below 2 at $R \ge 500$;
  - disagreeing pilots.

  `check(strict=True)` raises instead. None fires on a clean Gaussian run in a configuration whose computed false-alarm probability is below $10^{-3}$.

### 2.9 `tests/test_tuning.py`, `tests/test_sampler.py`, `tests/test_api.py` (R-TUNE-*, R-API-*)

**Tuning**
- **Adaptive Metropolis:**
  - on a correlated Gaussian ($d = 5$, condition number 100), whitening by the pilot covariance brings the condition number below 4 (over 20 seeds at most 1.54; the shape, not the norm, sets the efficiency of RWMH), and the chains' $\hat R$ is below 1.1;
  - on $\mathcal N(0, I_{10})$ from $\mathcal N(3, I)$, $\hat R < 1.15$ and the smallest eigenvalue of the pilot covariance exceeds 0.4 (over 20 seeds at most 1.09 and at least 0.58; chains adapting alone at rate $t^{-0.6}$ gave 2 to 3 and 0.25);
  - $\hat R$ exceeds $\sqrt 2$ for chains started in two separated modes;
  - the acceptance rate is within 0.05 of 0.234.
- **`choose_k`:**
  - gives $k$ = the inverted-CDF $q_{0.99}(\tau - 1)$, and the sampler $L = k$ and $m = 10k$;
  - the floor $k \ge 1$ holds;
  - $\beta > 1 - 1/R_0$ raises, and so does a quantile that falls on a censored pilot pair;
  - explicit values are kept, and `lag` and `m` follow an explicit `k` without a pilot.
- **Dual averaging:** reaches its target acceptance within 0.05 on a Gaussian.

**`CoupledSampler`**
- `run` tunes and pilots when needed, and records the reports.
- A second `run` continues the indices; with the same seed, two runs of 50 equal one run of 100.
- Changing `k`, `m`, `lag`, `thin` or `max_iter` between runs raises.
- Pilot replicates never appear in the measure.

**API**
- `sample` accepts every documented input form.
- Unknown kernel names raise with the list of valid names. A gradient-requiring kernel without `grad` raises.
- `rng` reproducibility holds.
- `telescopic.__all__` exports what the README uses, and `__version__` is defined.
- The README example runs, extracted by a test.

### 2.10 `tests/test_parallel.py` (R-EXEC-5, R-EXEC-6)

- **Pools:** for a point-by-point target, the per-replicate results are identical across `pool=None`, a `ProcessPoolExecutor` with 1 and 2 workers, and a `.map`-only pool.
- **Chunking:** chunks cover the indices exactly once. Results come back in index order.
- **Interrupts:** a simulated stop returns an `interrupted` result containing every started replicate, which form a prefix of the indices; running chunks are awaited. The sampler's next run completes an interrupted one, which then equals an uninterrupted run.
- **Worker hygiene:** tasks run with BLAS pinned to one thread, and workers that telescopic creates ignore SIGINT.
- **Progress:** without tqdm, progress falls back to lines on stderr.
- **Budget mode (should; deferred beyond v0.1 with the mode, DECISIONS #22):**
  - on a toy whose estimate correlates with its cost, the per-lane GH estimator is unbiased;
  - the pooled completed-by-deadline average is biased ($|z| > 10$ in seconds), which documents why it is not offered.

### 2.11 `tests/test_io.py`, `tests/test_plotting.py`, `tests/test_examples.py`, `tests/test_testing.py`

- **IO:**
  - `save` and `load` round-trip every field, including the measure and the seed;
  - a file from a future schema version raises;
  - no pickle is used (`allow_pickle=False`).
- **Plotting:** every plot function runs under the Agg backend and returns its axes. It is skipped without matplotlib.
- **Examples:**
  - exact moments match independent draws from each target's generative description, and closed forms or quadrature where they exist (banana, funnel);
  - log-densities match SciPy's densities of the generative descriptions up to a constant;
  - gradients match finite differences;
  - initial distributions have the documented shape.
- **`check_coupling`:**
  - passes the built-in kernels, and in three dimensions checks every built-in coupling;
  - flags a wrong $Y$ marginal, a coupling that reads another row's state, a coupling that recomputes $Y$ on the meeting branch, a stale cached log-density, and a coupling that never meets.

### 2.12 Slow statistical tier (`-m slow`)

- **Unbiasedness, end to end** (`tests/test_statistics.py`). For each kernel family, on targets with exact moments, the estimates of $\mathbb E[x]$ and $\mathbb E[x^2]$ follow the rule of §1. $R$ is set from a pilot of 1,000 replicates so that the standard error is at most 1/5 of the bias that dropping the correction would leave, in the component where that ratio is largest, and is at least 1,000 (`notes/experiments/m9_unbiasedness.py`, which also records the meeting times and the heavy-tail statistics: kurtosis, maximum variance share). Removing the correction fails every case. The cases:
  - a Gaussian in $d = 1, 5, 20$ from $\pi_0 = \mathcal N(3, I)$ and from a point mass;
  - random walks with reflection-maximal and maximal couplings, MALA, independence, a mixture of kernels;
  - HMC on the banana, from a point far along its ridge: from $U[-5, 5]^2$ one pair in 1,000 took 62,493 iterations, at corners where the step is unstable;
  - the pump-failure model with Gibbs;
  - $L \in \{1, 7\}$, and $L = k$ in 20 dimensions, where a lag of 1 with $k$ below the meeting times gave an estimator of standard deviation 11.6 against 0.82;
  - `thin` $\in \{1, 3\}$;
  - small $k$, so the bias correction matters.
- **Coverage.** 2,000 intervals of $R = 200$ from a stationary start, on the exact AR(1) chain under the recipe ($k = L = 38$, $m = 380$), for $x$ and $x^2$. The reference coverages, 0.9509 and 0.9489 with standard error 0.0022, come from $10^4$ intervals of an implementation written apart from the package (`notes/experiments/m9_coverage_reference.py`); the design review's 0.945 and 0.939 came from 1,000. The rule of §1 applies with the combined standard error. Coverage from near-stationary and far starts is recorded, not asserted, to document the heavy-tail caveat.
- **TV bounds against exact TV.** For the exact AR(1) chain from $\mathcal N(3, 0.5^2)$, coupled by `reflection_maximal`, with $10^4$ meeting times at $L \in \{1, 10\}$: $\widehat{\text{bound}}(t) \ge \mathrm{TV}(t) - 5\,\mathrm{SE}(t)$ wherever $\mathrm{TV}(t) \ge 0.01$. The exact TV between Gaussians comes from quadrature.
- **Literature reproductions** (loose bands): RWMH reflection-maximal mean $\tau \approx 30$ at $d = 10$ (O'Leary 2021); the bimodal mixture's mean 20 and 99% quantile 105 with proposal variance 9 (JOA20 §5.1); HMC on the banana, mean $\tau = 158$ with synchronous momenta (Heng & Jacob 2019, §5.2). Their logistic regression needs data that a hermetic test cannot fetch.
- **Warning calibration (recorded, not asserted).** The false-alarm rate of each warning of §7.4 on clean targets, and its firing rate on JOA20's bimodal case with a narrow $\pi_0$, are measured (`notes/experiments/m9_warnings.py`) and written to the documentation. In the design review's simulation of that case, warnings fired mainly in runs whose intervals already covered the truth (2 of 14 misleading runs warned), so the documentation says that such a failure cannot be detected from the output at moderate $R$. What is asserted is what the design can show: a wide $\pi_0$ exposes the second mode through the meeting times.
- **Linear SBC.** On conjugate Normal–Normal and Gamma–Poisson models, with one replicate per draw of $(\tilde\theta, y)$, $\mathbb E[\hat\pi(\phi) - \phi(\tilde\theta, y)] = 0$ for $\phi \in \{\theta, \theta^2, \mathbb 1(\theta \le c), \log p(y \mid \theta)\}$, under the rule of §1.

### 2.13 Docs, examples and packaging

- The tutorials (MyST notebooks) execute in the docs CI job, each in under 2 minutes.
- The README example is extracted and run by a test.
- `python -m build` produces a wheel and an sdist. A clean environment installs the wheel and imports it.

### 2.14 Benchmarks (non-gating)

`benchmarks/run.py` prints JSON with the commit, machine, thread settings and versions (`CLAUDE.md`, Measurements). It times these cases:
- lockstep coupled RWMH per pair-step, against lanes (one random-number mode exists: per-replicate streams, D-8);
- point-by-point overhead per step;
- pool speed-up for a 1 ms log-density;
- the overhead of the automatic pipeline;
- meeting times and relative inefficiency of RWMH, MALA and HMC on the examples.

## 3. Mapping from requirements to tests

| Requirement | Tests |
|---|---|
| R-EST-1–3 (construction, meeting, faithfulness) | 2.4 faithfulness; 2.6 lag, meeting and the transition each chain receives |
| R-EST-4–6 (weights, thinning, signed measure) | 2.5 identities; 2.6 recorded paths; 2.12 unbiasedness |
| R-EST-7 (cost) | 2.5; 2.6 |
| R-EST-8 (derived estimators) | 2.5; 2.7 |
| R-THEORY-1 | docs review; 2.12 coverage and failure-mode runs |
| R-CPL-1–3 | 2.2 |
| R-KER-* | 2.4 (with batch invariance and draw counts); 2.12 |
| R-INPUT-1–2 | 2.3 |
| R-EXEC-1–2 (streams, invariance) | 2.1; 2.6; 2.10 |
| R-EXEC-3–4 (engine, censoring) | 2.6 |
| R-EXEC-5–6 (parallel, interrupts) | 2.10 |
| R-EXEC-7 (memory) | 2.6 atom count; 2.9 (warning on the estimate) |
| R-TUNE-1–4 | 2.9 |
| R-OUT-1–3 | 2.7; 2.11 |
| R-DIAG-1–2 | 2.8; 2.12 |
| R-API-1–2 | 2.9 |
| R-ENG | CI jobs; 2.13 |

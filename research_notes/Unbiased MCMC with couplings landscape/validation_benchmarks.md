# Validation and benchmarks for coupled MCMC and unbiased estimators

This note builds on Q3 of the sibling note `/home/user/pastry/research_notes/Population inference software landscape/api_ux_validation.md` and does not repeat it. That section already covers generic SBC (Talts; Modrák), Geweke's joint-distribution test, Grosse & Duvenaud, the basics of Gandy & Scott and MCUnit, emcee and dynesty CI tolerances, pytest-randomly, pytest-rerunfailures, Hypothesis basics, ArviZ `array_stats`, and the stale posteriordb and inference-gym clients.

This note adds what is specific to couplings and unbiased estimators:
- **Literature numbers** were extracted from the papers' full text (pdftotext on arXiv PDFs).
- **Library practice** comes from read-only clones at these commits:
  - BlackJAX `2d6a51c` (2026-10-08)
  - NumPyro `1e42fd5` (2026-10-22)
  - PyMC `d1a761e` (2026-10-19)
  - posteriordb `91027f9` (2026-09-25)
  - posteriordb-python `929c54a` (2026-03-13)
  - unbiasedmcmc `660c698` (2026-09-23)
  - TFP inference_gym `b6c0968` (2026-10-21)
- **"Own computation"** marks numerical checks I wrote and ran (NumPy 2.5.4, SciPy 1.18.1). They show how the test designs behave; they are not literature results.
  - Scripts and outputs (`*.out`) are archived in `scripts/validation/`.

## Q1. How to test a coupled kernel: exact marginals, faithfulness, maximal-coupling meeting probability, invariance

### Takeaway
A coupled kernel P̄ is correct only if three things hold:
1. From every pair (x, y), each coordinate has exactly the uncoupled transition law P.
2. The diagonal is absorbing (faithfulness).
3. For maximal couplings, P(meet) = 1 − TV(p, q).

The strongest cheap tests are:
- **Deterministic oracle tests with prescribed randomness** (BlackJAX's pattern for its new `coupled_hmc`).
- **Exact coupled transition matrices** on small finite spaces.
- **One-sample tests against the exact one-step law.** In my check this was about 10× more sample-efficient than two-sample tests against reference draws.
- **Binomial tests of meeting frequency** against the closed form 1 − TV = 2Φ(−Δ/2).

That binomial test immediately exposed a floating-point bug that summary statistics miss: a reflection-maximal coupling whose "met" draws were equal only up to rounding. Gandy & Scott invariance tests, run in a sequential wrapper, apply to each marginal of P̄. No Python implementation of them exists.

### Cited Findings

#### What must hold (definitions)
- **Marginal constraint.** P̄ must satisfy P̄((x, y), A × X) = P(x, A) and P̄((x, y), X × A) = P(y, A) for all x, y and measurable A ([Jacob, O'Leary & Atchadé 2020, Sec. 2.1, arXiv:1708.03625](https://arxiv.org/abs/1708.03625); [JRSSB 82(3):543](https://academic.oup.com/jrsssb/article/82/3/543/7056129)).
- **Faithfulness.** Assumption 2.3: the chains "stay together after meeting", i.e. X_t = Y_{t−1} for t ≥ τ. Coupled chains with this property are called "faithful" (Rosenthal 1997). It "typically holds by design" ([Jacob et al. 2020](https://arxiv.org/abs/1708.03625)).
- **Stickiness.** Rosenthal and Dey et al. note that stickiness is non-trivial even for Markovian couplings. RWM couplings "can always be made sticky by requiring x′ = y′ … and V = U ∼ Unif if x = y" ([O'Leary 2021, arXiv:2102.01790](https://arxiv.org/abs/2102.01790)).
- **Mixing couplings.** It "would not be legal" to mix couplings in a way that breaks the marginal constraint ([Atchadé & Jacob 2024, arXiv:2406.06851](https://arxiv.org/abs/2406.06851)).

#### Maximal couplings: meeting probability and cost
- **Thorisson's rejection algorithm** (Algorithm 2 in Jacob et al.):
  - P(X = Y) = ∫ min(p, q) = 1 − d_TV(p, q).
  - Conditional on {X ≠ Y}, the two outputs are independent.
  - Expected cost is "two units" (one unit = one draw plus two density evaluations) "for all distributions p and q". The cost variance "goes to infinity" as TV → 0.
  - Source: [Jacob et al. 2020, Sec. 4.1](https://arxiv.org/abs/1708.03625).
- **Reflection-maximal coupling** (Bou-Rabee et al.) of N(μ1, Σ) and N(μ2, Σ):
  - With probability min(1, s(ẋ+z)/s(ẋ)), set ẏ = ẋ + z; otherwise ẏ = ẋ − 2(eᵀẋ)e.
  - P(X ≠ Y) equals TV. The cost is deterministic.
  - For discrete distributions the maximal coupling costs O(N).
  - Source: [Jacob et al. 2020, Secs. 4.1 and 5.4](https://arxiv.org/abs/1708.03625).
- **Closed-form TV for equal covariances.** TV(N(μ1, Σ), N(μ2, Σ)) = P(N(0, 1) ∈ [−Δ/2, Δ/2]) = 2Φ(Δ/2) − 1, with Δ the Mahalanobis distance. Devroye, Mehrabian & Reddad state this, citing Barsov & Ul'yanov (1987, Thm 1). Their own tight bounds now cover only same-mean Gaussians, after a flawed general bound was removed ([Devroye et al., arXiv:1810.08693](https://arxiv.org/abs/1810.08693)).
- **Sensitivity of one-step coalescence to distance.** In a stochastic-volatility RWM example, a maximal coupling would coalesce proposals with probability {0.81, 0.45, 0.02, …} at squared distances δ ∈ {10⁻⁴, 10⁻³, 10⁻², …}; later values are under 10⁻¹³ ([Papp & Sherlock, arXiv:2211.12585 App. C.3](https://arxiv.org/abs/2211.12585)).

#### How existing coupling code is checked
- **unbiasedmcmc (R reference implementation) has no automated test suite.**
  - There is no `tests/` directory. Validation is a set of informal scripts in `inst/check/`: `check_reflmaxcoupling.R`, `check_hbar.R`, `check_gamma.R`, `check_discretecouplings.R`, `check_lags.R`, `to_check_signed_measure.R` and others.
  - The scripts print `colMeans`/`cov` against the true μ and Σ, overlay histograms on densities, and check `all(identicals == equalvalues)`.
  - Its coupled MH kernel copies the proposal and its cached log-density into the second chain when the coupling flags `identical` (`proposal2 <- proposal1; proposal_pdf2 <- proposal_pdf1`). It declares meeting from that flag (`identical_ <- proposal_value$identical && accept1 && accept2`), not from floating-point comparison.
  - Sources: [unbiasedmcmc repo](https://github.com/pierrejacob/unbiasedmcmc); [inst/check](https://github.com/pierrejacob/unbiasedmcmc/tree/master/inst/check); [R/get_mh_kernels.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/get_mh_kernels.R).
- **BlackJAX now ships `blackjax.mcmc.coupled_hmc`** with synchronous and reflection couplings. Its docstring says the module "does not claim product-target invariance, meeting, unbiased estimators, or an efficiency improvement". Its contract: given the same `(state, standard_normal, uniform)` triple, "each marginal performs the same transition as ordinary HMC" ([coupled_hmc.py](https://github.com/blackjax-devs/blackjax/blob/main/blackjax/mcmc/coupled_hmc.py)).
- **BlackJAX's 1,556-line `tests/mcmc/test_coupled_hmc.py`** tests that contract deterministically ([test_coupled_hmc.py](https://github.com/blackjax-devs/blackjax/blob/main/tests/mcmc/test_coupled_hmc.py)):
  - **Independent oracle.** `test_marginals_match_an_independent_reference` compares each marginal's state and info with an oracle at `atol=1e-12`. The second marginal's innovation is rebuilt "entirely outside the module … in NumPy", so that a wrong `_reflect` and the module "together" cannot "still agree".
  - **Both branches.** The uniform is chosen from the reference acceptance probability so that both accept and reject branches are compared ("a kernel that always returned the proposal would still match the reference").
  - **Heterogeneous parameters** (different targets, metrics, step sizes and integration counts) are used "so a parameter leaking across the pair shows up".
  - **Algebra.** Reflection is tested as a norm-preserving involution.
  - **Statistics.** `test_reflection_preserves_the_standard_normal` uses 40,000 draws, with thresholds at "~6 sigma".
  - **Faithfulness and degenerate cases.** `test_synchronous_coupling_keeps_identical_states_identical` uses exact-equality asserts. `test_reflection_of_identical_states_degenerates_to_synchronous` checks that a zero direction gives the identity.
  - **Input validation.** Non-finite innovations and out-of-range uniforms are refused.

#### Invariance and detailed-balance tests (additions to the sibling note)
- **Gandy & Scott's exact two-sample test** resamples the data before each MCMC step, which "guarantees independence of samples".
  - Their exact rank test requires each kernel to be reversible. Systematic-scan Gibbs is not, "but … the constituent kernels can still be tested".
  - The sequential wrapper (Algorithm 3) has false-rejection probability ≤ α (Theorem 3.1).
  - mcunit defaults: α = 10⁻⁵, k = 7 stages, Δ = 4 sample-size multiplier; γ ≈ 0.15, β₁ ≈ 1.4·10⁻⁶, …, β₇ ≈ 0.15. Expected extra effort under the null is about 68.5% (17.1% with Δ = 1).
  - Source: [Gandy & Scott, arXiv:2001.06465](https://arxiv.org/abs/2001.06465).
- **Gandy & Scott power study** (nominal α = 0.01; [arXiv:2001.06465, Table 1](https://arxiv.org/abs/2001.06465)):
  - The exact tests hold their nominal level.
  - Geweke's (2004) test rejected correct samplers at rates of 0.31 (θ₁) and 0.523 (all statistics). Talts-style SBC with ESS-based thinning had "entirely uncontrolled" false rejection because ESS was overestimated.
  - Only the data likelihood statistic detected a wrong-variance error.
- **Gandy (2009)** gives a sequential Monte Carlo p-value with uniformly bounded "resampling risk": the probability of a different decision from the exact p-value stays below a chosen constant ([JASA 104(488):1504–1511; arXiv:math/0612488](https://arxiv.org/abs/math/0612488)). The follow-up "p-value buckets" method is by Gandy, Hahn & Ding ([arXiv:1703.09305](https://arxiv.org/abs/1703.09305)).
- **Implementations exist only in R and Julia.**
  - MCMCTesting.jl implements the two-sample test, the two-sample test with an extra Gibbs step, and the exact rank test, plus `seqmcmctest` (Algorithm 3). Users supply `sample_joint` and `markovchain_transition` ([MCMCTesting.jl docs](https://github.com/Red-Portal/MCMCTesting.jl/blob/main/docs/src/general.md)).
  - R's mcunit has `expect_mcmc_reversible` ([mcunit](https://search.r-project.org/CRAN/refmans/mcunit/html/expect_mcmc.html)).
- **Density goodness-of-fit tests in libraries.**
  - NumPyro's `numpyro.distributions.gof.auto_goodness_of_fit(samples, probs)` turns sorted-sample spacings times density into Exponential(1) variables, then applies a binned χ² test. The multivariate version uses nearest-neighbour volumes and needs more than 1000·dim samples. Its tests assert `gof > TEST_FAILURE_RATE = 2e-5`. The module is a port of `goftests` ([gof.py](https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/gof.py); [test_gof.py](https://github.com/pyro-ppl/numpyro/blob/master/test/test_gof.py)).
  - BlackJAX provides `blackjax.diagnostics.kernelized_stein_discrepancy` with an IMQ kernel. It is tested against closed-form NumPy references ([test_kernelized_stein_discrepancy.py](https://github.com/blackjax-devs/blackjax/blob/main/tests/test_kernelized_stein_discrepancy.py)).
- **Two-sample tests available in SciPy 1.18** (installed and inspected):
  - Tests: `ks_2samp`, `cramervonmises_2samp`, `anderson_ksamp`, `epps_singleton_2samp`, `chisquare`, `binomtest`.
  - Utilities: `permutation_test`, `monte_carlo_test`, `combine_pvalues`, `false_discovery_control`, and `power`, which "simulate[s] the power of a hypothesis test under an alternative hypothesis" ([scipy.stats.power](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.power.html)).
  - `energy_distance` is 1-D and returns no p-value.
  - `anderson_ksamp` p-values are "floored / capped at 0.1% / 25%" unless `method=PermutationMethod(...)`, so it cannot run at α < 10⁻³ ([anderson_ksamp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson_ksamp.html)).
- **Multivariate two-sample tests.** hyppo provides `Energy`, `MMD`, `DISCO`, `KSample`, `Hotelling` and others, with permutation or "fast chi-squared approximation" p-values ([hyppo API](https://hyppo.neurodata.io/api/index.html)). PyMC hand-rolls an MMD permutation test: median-heuristic RBF length-scale (median pairwise distance / 2), 500 permutations, α = 0.01, `RandomState(10)` ([tests/gp/test_hsgp_approx.py](https://github.com/pymc-devs/pymc/blob/main/tests/gp/test_hsgp_approx.py)).

#### Own computations (check_maximal_couplings.py (`scripts/validation/check_maximal_couplings.py`), check_reflection_bitwise.py (`scripts/validation/check_reflection_bitwise.py`), check_marginal_power.py (`scripts/validation/check_marginal_power.py`), check_finite_state.py (`scripts/validation/check_finite_state.py`), check_ar1.py (`scripts/validation/check_ar1.py`))
- **Algorithm 2 meeting frequency** for N(0,1) versus N(Δ,1), Δ ∈ {0.1, 0.5, 1, 2, 4}, n = 2·10⁵ pairs each:
  - Empirical meeting frequency matched 1 − TV: binomial p-values 0.11–0.98.
  - One-sample KS of both marginals against the exact normal CDFs: p ≥ 0.17.
  - Mean cost was 1.989–2.000 units in every case.
  - Cost variance grew from 0.10 (Δ = 4) to 46.3 (Δ = 0.1). At Δ = 0.02 (TV = 0.008) the standard deviation was 16.2 and the maximum 1,070 units.
- **Floating-point meeting bug in a reflection-maximal coupling** (d = 10, dense Σ):
  - **Naive implementation.** Computing Y as μ2 + (ẋ + z)Lᵀ on the meeting event gave bitwise-equal pairs only 0.0023, 0.0004 and 0.00001 of the time. The exact meeting probabilities were 0.940, 0.658 and 0.255. On met draws, |x − y| ≤ 1.8·10⁻¹⁵.
  - **Fixed implementation.** Copying X into Y on the meeting event gave 0.9396, 0.6559 and 0.2547 (binomial p = 0.105, 0.092, 0.912).
  - **Consequence.** Marginal tests passed in both versions. Only the meeting-frequency test caught the bug.
- **A "forgot the rejection loop" bug** (residual Y drawn from q directly, so Y has law min(p, q) + TV·q):
  - KS distance from q was 0.096–0.153 for Δ ∈ {0.5, 1, 2}.
  - A one-sample KS test at α = 10⁻⁴ had power 0.99 with n = 1,000 (Δ = 0.5) and 1.00 for n ≥ 10⁴.
- **Marginal tests for a 5% proposal-scale bug** in one RWMH transition from x = 1 (target N(0,1), σ = 2.4, exact acceptance a(x) = 0.4654), 100 replications. Three designs:
  - **A, exact one-sample:** binomial test of the stay probability plus KS of moved values against the exact continuous-part CDF (from quadrature), Bonferroni-combined, α = 10⁻⁴.
  - **B, two-sample KS** against n reference draws, α = 10⁻⁴.
  - **C, `anderson_ksamp`** at its 10⁻³ floor.

  | Case | Design A | Design B | Design C |
  |---|---|---|---|
  | Bug, n = 10⁴ | 0.22 | 0.00 | 0.00 |
  | Bug, n = 10⁵ | 1.00 | 0.74 | 0.54 |
  | Correct kernel, false positives | 0/100 at both n | 0/100 at both n | 2/100 at n = 10⁵ |
- **Finite-state exact check.** K = 6 states, π ∝ (1, 3, 2, 5, 1, 4), ±1 random-walk MH on a cycle, maximal coupling of proposals with independent residuals, common uniform for acceptance.
  - The exact 36 × 36 coupled matrix satisfied both marginal constraints and faithfulness to machine precision.
  - 20,000 simulated lag-1 meeting times matched the exact pmf (χ² p = 0.866 over 49 bins). Mean 5.252, exact 5.243.
- **AR(1) one-step meeting probability.** For X′ = 0.9X + √0.19 ε with reflection-maximal coupling, the empirical meeting frequency matched 2Φ(−ρ|x − y|/(2s)) at |x − y| ∈ {0.1, 0.5, 1, 2} (binomial p = 0.048–0.98). One of eight marginal KS tests had p = 0.030, which is expected at α = 0.05 and is why per-test α must be far smaller in a suite.

### Inferences

#### Architecture that makes kernels testable
- Write every kernel and coupling as a pure function of (state, explicit random inputs), e.g. `step(state, normals, uniforms)`, with a thin wrapper that draws the inputs from a `numpy.random.Generator`. This enables BlackJAX-style oracle tests. It also applies Grosse & Duvenaud's separation of probability calculations from sampler logic to couplings.
- Return an explicit `met` flag from the coupled step.
- On the meeting event, copy X and all its caches (log-density, gradient) into Y. Never recompute Y.
- Test that `met` implies bitwise equality, and that bitwise equality persists for every later step.

#### Fast deterministic tier (every commit; target under 1 minute)
1. **Oracle draw-for-draw tests.** With fixed injected normals and uniforms, each marginal of P̄ equals an independently coded NumPy reference kernel at `atol=1e-12`.
   - Choose the uniform from the oracle's acceptance probability so that both accept and reject paths run.
   - Run for (x, y) far apart, close, and equal.
   - Cover MH with each proposal coupling, Gibbs with each conditional coupling, and HMC with synchronous and reflection momenta.
2. **Algebraic properties** (Hypothesis with the CI profile):
   - Reflection is an involution, norm-preserving, and maps z to −z.
   - A zero direction gives the identity.
   - Discrete maximal coupling: the exact joint table has the right row and column sums, and diagonal mass 1 − TV, to 1e-15.
3. **Faithfulness.** From x = y, run 100 coupled steps of every coupling and assert `np.array_equal` at every step, plus `met == True`.
4. **Exact finite-state matrices.** Build P̄ for 4–6 state targets by enumeration, using the same primitives in "exact mode". Assert the marginal row sums equal P and the diagonal is absorbing. Check detailed balance π_i P_ij = π_j P_ji for the marginal kernel.

#### Statistical tier (nightly or weekly, rotating seeds)
1. **One-sample exact marginal tests from fixed pairs.** Use n = 10⁵ per pair and several pairs, e.g. (x, x), (x, x + 0.1σ), (x, x + 3σ), with α = 10⁻⁵ to 10⁻⁴ per family after Bonferroni.
   - MH: binomial test of the stay probability r(x) plus KS of moved values against the normalized q(x′|x)α(x, x′), both by 1-D quadrature.
   - Gibbs: KS against the exact conditional CDF.
   - Multivariate Gaussians: KS of the whitened coordinates plus a KS test of ‖whitened‖² against χ²_d.
   - Use two-sample fallbacks (`ks_2samp`; energy or MMD statistics in `scipy.stats.permutation_test`) only when no exact law exists, and with about 2–4× the n.
2. **Meeting frequency.** Binomial test of the meeting frequency against 2Φ(−Δ/2) for both Gaussian maximal couplings.
   - With n = 2·10⁵ the standard error is about 0.001, which detects deviations of ±0.005 with near-certain power.
   - Also check that Algorithm 2's mean cost is 2 units within 5 SE, at moderate TV only, since its variance diverges as TV → 0.
3. **Meeting-time distribution.** χ² test of simulated meeting times against the exact finite-state pmf. In my check this took about 1 s.
4. **Invariance of each marginal of P̄** (Gandy & Scott): draw x ~ π exactly (conjugate or finite targets) and y from an arbitrary law, apply P̄ once, and test that the first coordinate is distributed as π.
   - The sequential wrapper (α = 10⁻⁵, k = 7, Δ = 4) is about 60 lines of Python. There is no Python package to reuse.
   - Use the exact rank test only for reversible kernels.

#### False-positive budget
- With S statistical assertions per run, use α_test ≤ 10⁻³/S. For example, 100 assertions gives α = 10⁻⁵, which is mcunit's default.
- Use Gandy-style sequential escalation (on a borderline p-value, rerun with 4× n) instead of `--reruns`, which hides real bugs.

### Gaps
- No Python implementation of Gandy & Scott or Gandy (2009) was found. MCMCTesting.jl and mcunit are Julia and R only.
- No coupling library with an automated statistical test suite was found. unbiasedmcmc uses informal scripts. BlackJAX's `coupled_hmc` tests do not cover meeting or estimators.
- Exact TV between Gaussians with different covariances has no simple closed form. Devroye et al. now give tight bounds only for the same-mean case. I did not verify the general-case bounds they attribute to Arbas, Ashtiani & Liaw. In 1-D, use quadrature (as in my scripts).
- I did not verify hyppo's run-time call signature or default number of permutations.

## Q2. How to test unbiased estimators: replicate tests, coverage, TV bounds against exact TV, SBC

### Takeaway
Exact unbiasedness at every finite k and m makes replicate tests and SBC-type checks valid at any run length. Four caveats shape the test plan:
1. **Index bugs in the bias-correction weights cause tiny biases**, because the expected differences telescope. In my exact finite-state calculation, an off-by-one weight bug needed about 4 million replicates to reach 5σ. Deterministic algebra tests and exact finite-state expectations must therefore catch these bugs.
2. **H can be very heavy-tailed even with geometric meeting tails.** On AR(1) from a far point mass, nominal 95% CLT intervals covered 77% (R = 50) and 86% (R = 500), and |z| > 4 happened at a rate far above nominal. Tolerances must be at least 5 SE and sequential, and coverage tests belong in the slow tier.
3. **L-lag TV bounds can be checked against exact TV.** They must dominate exact TV for every t in expectation. With L = 50 on AR(1) they were within about 0.006 of exact TV for t ≥ 10 (0.02 at t = 0).
4. **SBC becomes a "linear SBC".** For any φ(θ, y), E[H_φ(y) − φ(θ̃, y)] = 0 exactly, at any k and m. With the log-likelihood as φ it caught a "sampler targets the prior" bug at z = −162.

### Cited Findings

#### Estimator definitions and properties
- **Definitions.**
  - H_k = h(X_k) + Σ_{t=k+1}^{τ−1} (h(X_t) − h(Y_{t−1})).
  - Time-averaged: H_{k:m} = (m − k + 1)⁻¹ Σ_{ℓ=k}^{m} h(X_ℓ) + Σ_{ℓ=k+1}^{τ−1} min(1, (ℓ − k)/(m − k + 1)) (h(X_ℓ) − h(Y_{ℓ−1})).
  - The decomposition is H_{k:m} = MCMC_{k:m} + BC_{k:m}. "If τ ≤ k + 1, then the correction term equals zero".
  - The estimator "can take values outside of the range of the test function h".
  - The signed measure π̂ has weights summing to 1.
  - Source: [Jacob et al. 2020, Secs. 2.1–2.4](https://arxiv.org/abs/1708.03625).
- **Proposition 3.1.** Under Assumptions 2.1–2.3, H_{k:m} has expectation E_π[h], finite variance and finite expected cost for all m ≥ k ≥ 0. Proposition 3.2 gives uniform a.s. convergence of the signed-measure CDF F̂_R ([Jacob et al. 2020](https://arxiv.org/abs/1708.03625)).
- **Budget constraints** ([Jacob et al. 2020, Sec. 3.3](https://arxiv.org/abs/1708.03625)):
  - Averaging the estimators completed by time t is biased: E[H̄_p(t)] = E[H] − E[H·1(C > t)].
  - The unbiased fix H̃_p(t) waits for the first estimator on each processor.
  - Three CLTs justify confidence intervals.
  - With a fixed t, completion time grows like log P under geometric tails.
- **Deadline warning.** "Retaining the C estimators that are first completed would introduce a bias", because the estimator "is not independent of its cost" ([Atchadé & Jacob 2024](https://arxiv.org/abs/2406.06851)).
- **Interval recipe and robust alternatives** ([Atchadé & Jacob 2024](https://arxiv.org/abs/2406.06851)):
  - Intervals are μ̂ ± q·σ̂/√C, which need finite variance. That needs κ > 2 for polynomial meeting tails, plus h ∈ L^m with m > 2κ/(κ − 2).
  - Robust alternatives mentioned: trimmed means, median-of-means, and self-normalized sums.
  - Tuning: set k as a high (e.g. 99%) quantile of τ − L, then L = k and ℓ = 10k.
- **Polynomial tails.** Middleton et al. give conditions for polynomially tailed meeting times. In their pseudo-marginal toy (bivariate N((1, 2), I), log-normal likelihood noise σ ∈ {0, 0.5, 1, 1.5, 2}, 10⁵ meeting times each), tails move from geometric at σ = 0 to polynomial as σ grows ([Middleton et al., arXiv:1807.08691, Sec. 3.1](https://arxiv.org/abs/1807.08691)).

#### Misleading finite-R results in the literature
- **Bimodal target** 0.5N(−4, 1) + 0.5N(4, 1), true ∫1(x > 3)π ≈ 0.421 ([Jacob et al. 2020, Sec. 5.1](https://arxiv.org/abs/1708.03625)):
  - With σ_q² = 1 and π0 = N(10, 1), R = 1,000 estimators (k = 50, m = 500) gave the 95% CI [0.799, 0.816].
  - 9,000 more estimators gave [−0.353, 1.595], because a few runs hit the far mode before meeting.
- **L-lag bounds can mislead the same way.** For the same bimodal target, bounds from N = 1,000 repeats can mislead, because only larger N contained runs where a chain switched modes before meeting ([Biswas, Jacob & Vanetti 2019, Sec. 2.2.2](https://arxiv.org/abs/1905.09971)).

#### TV and Wasserstein bounds
- **Theorem 2.5:**
  - d_TV(π_t, π) ≤ E[max(0, ⌈(τ^(L) − L − t)/L⌉)].
  - d_W(π_t, π) ≤ E[Σ_{j=1}^{⌈(τ^(L)−L−t)/L⌉} d(X_{t+jL}, Y_{t+(j−1)L})].
  - τ^(L) = inf{t > L : X_t = Y_{t−L}}.
  - Source: [Biswas et al. 2019](https://arxiv.org/abs/1905.09971).
- **Validation example with exact distances.** Target N(0, 1), RWMH with σ = 0.5, π0 = δ_10, L ∈ {1, 150}, N = 10,000 runs. Exact TV and W1 are plotted for comparison; "With L = 150 the estimated upper bounds for both are close to the exact distances" ([Biswas et al. 2019, Sec. 2.2.1, Fig. 1](https://arxiv.org/abs/1905.09971)).

#### Asymptotic-variance targets and SBC background
- **AR(1) asymptotic variance** ([Douc, Jacob, Lee & Vats, arXiv:2206.05691 v5, Sec. 5.1](https://arxiv.org/abs/2206.05691); numbers extracted from the arXiv HTML):
  - Setup: X_t = φX_{t−1} + W_t with φ = 0.99, π0 = N(0, 4²), reflection-maximal coupling, h(x) = x.
  - Exact asymptotic variance v(P, h) = (1 − φ)⁻² = 10⁴.
  - Tuning: k = L = 500, ℓ = 5k.
  - Table 5.1 (UPAVE, 10³ runs, 95% bootstrap CIs): R = 1 gives estimate [8049, 10417] at cost [5234, 5262] transitions. R = 100 gives [9829, 10250] at cost [21262, 21578].
  - Batch-means and spectral estimators showed bias.
- **SBC test quantities.** Modrák et al. recommend data-dependent test quantities, especially "the joint likelihood of the data". Parameter-only SBC "could never detect large classes of problems including when the posterior is equal to the prior". They also distinguish SBC from data-averaged-posterior checks ([Modrák et al., Bayesian Analysis, arXiv:2211.02383](https://arxiv.org/abs/2211.02383)).
- **NumPyro VSBC.** NumPyro now ships `numpyro.infer.calibration.vsbc_diagnostic` (variational SBC), tested against Gaussian cases ([test/infer/test_calibration.py](https://github.com/pyro-ppl/numpyro/blob/master/test/infer/test_calibration.py)).

#### Own computations (check_finite_state.py (`scripts/validation/check_finite_state.py`), check_bug_power.py (`scripts/validation/check_bug_power.py`), check_bug_power_verify.py (`scripts/validation/check_bug_power_verify.py`), check_ar1.py (`scripts/validation/check_ar1.py`), check_ar1_tails.py (`scripts/validation/check_ar1_tails.py`), check_ar1_zrates.py (`scripts/validation/check_ar1_zrates.py`), check_linear_sbc.py (`scripts/validation/check_linear_sbc.py`))

**Exact finite-state TV bounds** (K = 6 chain from Q1, π0 = δ_0)
- Computed exact TV(π0Pᵗ, π) and the exact L-lag bound, the latter from the exactly computed pmf of τ^(L) via the off-diagonal sub-stochastic recursion.
- The bound was ≥ the exact TV for every t < 60 and every L ∈ {1, 5, 20}.
- At t = 10, TV = 0.0483, while the bounds were 0.892 (L = 1), 0.223 (L = 5) and 0.141 (L = 20). At t = 0 the L = 1 bound equals E[τ − 1] = 5.24.

**Exact bias of estimator index bugs**
- Computed exact E[H_{k:m}] for buggy variants with the off-diagonal recursion E[BC] = Σ_ℓ w_ℓ E[(h(X_ℓ) − h(Y_{ℓ−1}))1(τ > ℓ)]. The correct weights give exactly π(h) = 2.875, which validates the recursion.
- Results by configuration:

  | Configuration | sd(H) | Bug | Exact bias | Replicates for 5σ |
  |---|---|---|---|---|
  | Cycle proposal, k = 2, m = 10 | 11.1 | weight (ℓ − k + 1) instead of (ℓ − k) | +0.027 | ≈ 4.2 million |
  | Cycle proposal, k = 2, m = 10 | 11.1 | `min(1, ·)` cap removed | −0.052 | ≈ 1.1 million |
  | Cycle proposal, k = 2, m = 10 | 11.1 | correction dropped | +0.047 | ≈ 1.4 million |
  | Cycle proposal, k = 0 | — | cap removed | −1.016 | ≈ 7,900 |
  | Uniform independence proposal, k = 2, m = 10 | 0.79 | weight bug | +0.017 | ≈ 52,000 |
- A 400,000-replicate simulation confirmed the weight-bug bias. On the same paths, the buggy mean exceeded the correct mean by 0.025, consistent with the exact +0.027. The SE of each mean was about 0.02.
- **Earlier chance alarms.** Two earlier runs with R = 20,000 gave z = +2.75 and +2.93 for bugs whose exact biases imply z ≈ 0.3. Those were chance fluctuations: an isolated |z| ≈ 3 is not evidence of a bug.

**AR(1) TV bounds against exact TV** (ρ = 0.9, π0 = δ_10, reflection-maximal coupling, N = 10⁴ runs)
- Exact law: π_t = N(ρᵗ·10, 1 − ρ^{2t}).
- With L = 50 the estimated bounds were 1.020, 0.934, 0.464, 0.169, 0.058 and 0.0070, against exact TV of 1, 0.928, 0.458, 0.168, 0.059 and 0.0072 at t = 0, 10, 20, 30, 40, 60.
  - The estimate was never more than about 0.5 SE below the exact TV.
  - It was at most 0.006 above for t ≥ 10, and 0.02 above at t = 0.
  - So the bound is nearly tight for this coupling.
- With L = 1 the estimate at t = 40 was 0.046 ± 0.008 against exact 0.059. This is under the true value by 1.5 SE, which Monte Carlo error allows even though the expectation of the bound must dominate.

**AR(1) inefficiency relative to the exact asymptotic variance** (1 + ρ)/(1 − ρ) = 19, R = 20,000
- The ratio of inefficiency to asymptotic variance fell with larger k and m:

  | k | m | Ratio |
  |---|---|---|
  | 1 | 1 | 244 |
  | 13 (q90) | 130 | 26.4 |
  | 35 (q99) | 350 | 1.89 |
- Means of H for E[x] = 0 and E[x²] = 1 were within about 2 SE in every row.

**Heavy tails and interval coverage** (AR(1), k = q90, m = 10k, 2·10⁵ estimators)
- Sample excess kurtosis was 687 to 839, dominated by a few extreme replicates (max |H| ≈ 103–114, sd ≈ 1.2–1.45).
- Coverage of 95% t-intervals for E[x] = 0, with B = 400 intervals per row:

  | Start | Coverage, R = 50 | Coverage, R = 500 |
  |---|---|---|
  | Stationary start | 0.973 | 0.965 |
  | Point mass at 10 | 0.765 | 0.863 |
  | Independent N(10, 1) | 0.850 | 0.910 |

**False-alarm rates of z thresholds** (point-mass start, k = 15, m = 150)

| R | Batches | Fraction \|z\| > 1.96 | Fraction \|z\| > 3 | Fraction \|z\| > 4 | Max \|z\| |
|---|---|---|---|---|---|
| 5,000 | 200 | 0.065 | 0.005 | 0.005 (one batch; nominal 6·10⁻⁵) | 4.19 |
| 20,000 | 100 | 0.090 | 0.020 | 0 | 3.21 |

**Linear SBC** (θ ~ N(0, 1), y_i ~ N(θ, 1), n = 5, R = 20,000 replicates, one coupled RWMH pair per replicate, π0 = N(0, 9))
- With k = 40, m = 400:

  | Implementation | Check | z |
  |---|---|---|
  | Correct | all three checks | \|z\| ≤ 1.82 |
  | Sampler targets the prior | θ | −0.11 |
  | Sampler targets the prior | θ² | +0.71 |
  | Sampler targets the prior | log-likelihood | −162 |
  | Likelihood variance 1.5 instead of 1 | θ² | −10.7 |
  | Likelihood variance 1.5 instead of 1 | log-likelihood | −47.6 |
  | Prior dropped | θ² | +59.8 |
- With k = 10, m = 50 the correct implementation still passed (|z| ≤ 1.62), but sd(D) was 3–35 instead of 0.4–4.4. Only the prior-only bug (log-likelihood z = −18.6) and the flat-prior bug (θ², z = +5.24) were detected.

### Inferences

#### Deterministic tier (the bulk of estimator correctness)
1. **Algebraic identities** (Hypothesis-generated synthetic coupled paths: arrays X and Y of random floats with a forced meeting time τ, k ≤ m, L ≥ 1). Assert:
   - Eq. (2.1) equals the definitional average (m − k + 1)⁻¹ Σ_ℓ H_ℓ.
   - Signed-measure weights sum to 1, and Σ ω h(Z) equals H(h) for several h.
   - Linearity in h holds.
   - τ ≤ k + 1 implies BC = 0.
   - m = k gives H_k.
   - The lag-L version reduces to lag 1 at L = 1.
   - The cost equals 2(τ − 1) + max(1, m + 1 − τ) at lag 1. Note that [Atchadé & Jacob](https://arxiv.org/abs/2406.06851) write L + 2(τ − L) + max(0, ℓ − τ), so fix one convention and test it.
2. **Exact finite-state expectations.** For 2–6 state targets, compute E[H_{k:m}] exactly by the off-diagonal recursion on the joint chain. Var(H_{k:m}) needs an extended recursion that tracks partial sums, which I did not implement.
   - Compare these with the package's estimator applied to exhaustively enumerated short paths, or with millions of vectorized replicates. My unoptimized 400,000-replicate run took about 40 s.
   - This is the only practical way to catch weight and index bugs. The deterministic identity tests in item 1 catch most of them first.
3. **"Perfect kernel" structural test.** If P(x, ·) = π independently of x, a common-random-numbers coupling meets at τ = 2 (lag 1) almost surely. H_0 = h(X_0) + h(X_1) − h(Y_0) then has mean π(h). This checks the meeting-time bookkeeping and the lag indexing deterministically.

#### Statistical tier
1. **Unbiasedness.**
   - Run R replicate estimators on targets with exact π(h): AR(1), Gaussians, finite chains, conjugate models.
   - Include point-mass starts far from π and small k, so that the bias correction matters.
   - Set R from a pilot so that SE ≤ (smallest bias of interest)/5.
   - Assert |z| ≤ 5. On 4 < |z| ≤ 6, escalate sequentially (rerun with 4× R and a fresh seed) rather than fail.
   - Record heavy-tail diagnostics with every run: kurtosis, max|H|/sd, and the fraction of variance from the top 0.1% of replicates.
2. **Coverage** (slow tier only).
   - Use B = 1,000 intervals of R = 500 from stationary or near-stationary starts, and assert coverage ∈ [0.93, 0.97], where the binomial SE is about 0.007.
   - Separately record, without asserting, coverage from far starts, to document the heavy-tail caveat.
3. **TV bounds.**
   - On AR(1) (exact Gaussian TV via quadrature) and finite chains (exact Pᵗ), assert bound̂(t) ≥ TV(t) − 5·SE(t) on a grid of t, with N = 10⁴ meeting times.
   - In the deterministic tier, check the exact-pmf bound against exact TV.
   - For Wasserstein bounds use the same AR(1) case, where W1 between 1-D Gaussians is ∫|F1 − F2| by quadrature.
4. **Linear SBC.**
   - Use conjugate models (Normal–Normal, Beta–Binomial, Dirichlet–Multinomial, Gamma–Poisson) with φ ∈ {θ, θ², 1(θ ≤ c), log p(y|θ)}.
   - Use k at a high quantile of τ and m = 10k, R = 20,000, |z| ≤ 5.
   - Run it both as a test and as a user-facing `telescopic.validate.linear_sbc`.
   - Classical rank SBC applies only to the marginal chain. It is useful but cannot see the coupling or the bias-correction code.
5. **Deadline-bias test.** Check that a budget-constrained driver uses the first-completed rule H̃, not the "completed by t" average. Use a toy estimator whose value is correlated with its cost, where the naive average's bias is known.

### Gaps
- I found no published SBC procedure designed for signed-measure or unbiased-MCMC output. "Linear SBC" is my own derivation (it follows from E[H|y] = E[φ|y]). I verified it on one conjugate model only.
- Exact Var(H_{k:m}) on finite chains was not computed. It is feasible with a forward recursion that tracks partial sums, but I did not implement it.
- Coverage and false-alarm figures come from a single target (AR(1), ρ = 0.9). How they generalize to other targets is unknown. The kurtosis estimates are dominated by single replicates, so they are unstable.
- I did not find literature quantifying median-of-means or trimmed-mean intervals for unbiased MCMC. Atchadé & Jacob only mention them.

## Q3. Testing practice in sampler libraries (Stan, BlackJAX, NumPyro, emcee, dynesty, PyMC)

### Takeaway
Mature libraries mostly check posterior moments with fixed, hand-picked tolerances under fixed seeds.
- **Seeds:** NumPyro uses seed 0 for every test, PyMC uses `seeded_test`, Stan uses seed 1234, and BlackJAX rotates its seed by calendar date.
- **Formal goodness-of-fit tests appear in a few places:** NumPyro's density goodness-of-fit at failure rate 2·10⁻⁵, PyMC's KS at α = 10⁻³, and PyMC's MMD permutation test at α = 0.01.
- **CI speed:** jobs are split by test subset, slow tests are tracked, and pytest-xdist is used.
- **Flaky tests** are handled ad hoc: skips with "flaky on CI" comments, or wider tolerances.
- **Hypothesis is not used** in any of the three test suites I inspected.

The most transferable pieces are BlackJAX's new prescribed-randomness oracle tests for coupled HMC, and dynesty's tolerances based on the run's own Monte Carlo error (covered in the sibling note).

### Cited Findings

#### BlackJAX
- **Seeding.** The `BlackJAXTest` base class seeds the PRNG from `int(datetime.date.today().strftime("%Y%m%d"))`, so results are "deterministic within a day and rotate automatically" ([tests/fixtures.py](https://github.com/blackjax-devs/blackjax/blob/main/tests/fixtures.py)).
- **CI.** CI runs `pytest -n 2 -vv --benchmark-disable --cov=blackjax` on Python 3.13 and 3.14 after a pre-commit and mypy style job. `pytest.ini` sets `filterwarnings = error` ([test.yml](https://github.com/blackjax-devs/blackjax/blob/main/.github/workflows/test.yml); [pytest.ini](https://github.com/blackjax-devs/blackjax/blob/main/pytest.ini)).
- **Sampler tests** assert posterior means of a linear regression to `atol=1e-1` (some with `rtol=1e-2`), and a univariate normal's mean and variance to `rtol=1e-1` ([tests/mcmc/test_sampling.py](https://github.com/blackjax-devs/blackjax/blob/main/tests/mcmc/test_sampling.py)).
- **Coupled HMC tests** use the oracle design described in Q1 ([test_coupled_hmc.py](https://github.com/blackjax-devs/blackjax/blob/main/tests/mcmc/test_coupled_hmc.py)).

#### NumPyro
- **Seeding and leak check.** `conftest.py` calls `set_rng_seed(0)` before every test, and asserts `len(jax.live_arrays()) == 0` before the first test ([test/conftest.py](https://github.com/pyro-ppl/numpyro/blob/master/test/conftest.py)).
- **Tolerances.** `test/infer/test_mcmc.py` (49 tests) uses hand-set tolerances such as `rtol=0.07` on mean and sd, `atol=0.02`, `atol=0.05` and `rtol=0.10` on covariance.
- **Flaky tests are skipped,** with `pytest.skip("The test is flaky on CI x64.")` ([test_mcmc.py](https://github.com/pyro-ppl/numpyro/blob/master/test/infer/test_mcmc.py)).
- **CI split into jobs** ([ci.yml](https://github.com/pyro-ppl/numpyro/blob/master/.github/workflows/ci.yml)):
  - Modelling tests, excluding `test/infer` and `test/contrib`.
  - An inference job that runs `test/infer/test_mcmc.py` separately, plus `JAX_ENABLE_X64=1` and `JAX_ENABLE_CUSTOM_PRNG=1` reruns and `XLA_FLAGS=--xla_force_host_platform_device_count=2` chain tests.
  - A tracer-leak run.
  - A separate examples job.
  - `--durations` reporting.
- **Density goodness-of-fit tests** run at `TEST_FAILURE_RATE = 2e-5` ([test_gof.py](https://github.com/pyro-ppl/numpyro/blob/master/test/test_gof.py)).

#### PyMC
- **Statistical fixtures** ([tests/sampler_fixtures.py](https://github.com/pymc-devs/pymc/blob/main/tests/sampler_fixtures.py)):
  - `KnownMean` and `KnownVariance` use `npt.assert_allclose` with class-level rtol and atol.
  - `KnownCDF` runs `stats.kstest` on draws thinned by 5 (`ks_thin = 5`) and requires `alpha = 0.001 < p`.
  - Multi-dimensional draws use `stats.combine_pvalues`.
- **Seeding** is an opt-in fixture, `seeded_test`, which calls `np.random.seed(20160911)` ([tests/conftest.py](https://github.com/pymc-devs/pymc/blob/main/tests/conftest.py)).
- **CI time management.** "Tests are split into multiple jobs to accelerate the CI" through a `test-subset` matrix of file lists ([tests.yml](https://github.com/pymc-devs/pymc/blob/main/.github/workflows/tests.yml)). A scheduled workflow runs every 6 hours to update a "Slow Tests" issue ([update-test-issues.yml](https://github.com/pymc-devs/pymc/blob/main/.github/workflows/update-test-issues.yml)).
- **Flakiness** is discussed only in code comments, e.g. "The following statistical test appears to be more flaky than expected" ([tests/distributions/test_mixture.py](https://github.com/pymc-devs/pymc/blob/main/tests/distributions/test_mixture.py)).

#### Stan
- **Gold regression tests.** `performance-tests-cmdstan/runPerformanceTests.py` runs models with `random seed=1234`.
  - "Golds" store a (mean, stdev) per parameter.
  - `--check-golds` fails if |mean − gold| > 10⁻⁴ and |mean − gold|/stdev > 0.3.
  - `--check-golds-exact TOL` fails if the absolute error exceeds TOL.
  - A `--runs` flag repeats runs for timing.
  - Source: [runPerformanceTests.py](https://github.com/stan-dev/performance-tests-cmdstan/blob/master/runPerformanceTests.py).
- **stat_comp_benchmarks** passes an algorithm if "(mu_est - mu_true) / sigma_true < 0.25". "No validation of posterior variances is considered" except where a model defines it ([stat_comp_benchmarks](https://github.com/stan-dev/stat_comp_benchmarks)).

#### emcee and dynesty
- emcee uses a fixed seed with KS and moment tolerances. dynesty uses tolerances of 4 × its own reported error, with a seed-looping environment variable `DYNESTY_TEST_RANDOMSEED`. Both are covered in the sibling note's Q3 and are not re-inspected here ([emcee test_proposal.py](https://github.com/dfm/emcee/blob/main/src/emcee/tests/integration/test_proposal.py); [dynesty tests/test_gau.py](https://github.com/joshspeagle/dynesty/blob/master/tests/test_gau.py)).

#### Hypothesis (property-based testing)
- **None of the three suites import it.** A grep of BlackJAX, NumPyro and PyMC at the commits above found no `hypothesis` imports in their test suites. The only matches were the words "null hypothesis" and an unrelated comment.
- **Property-based testing is advocated for scientific Python** with real NumPy and Astropy bugs found ([Hatfield-Dodds, "Falsify your Software", SciPy 2020](https://proceedings.scipy.org/articles/Majora-342d178e-016)). A 2023 scipy-dev proposal to add Hypothesis included a deterministic default profile.
- **CI-friendly defaults.** When a `CI` environment variable is set, the `ci` profile is active by default ([Hypothesis settings](https://hypothesis.readthedocs.io/en/latest/reference/api.html)):
  - `derandomize` "seed[s] Hypothesis' random number generator using a hash of the test function".
  - `deadline` becomes None.
  - `print_blob` is True.
  - `@example` inputs "do not count towards settings.max_examples".
  - `@seed` or `--hypothesis-seed` reproduces failures.

### Inferences
Recommended practice for telescopic:
- **Seeds.**
  - Every test receives `rng = np.random.default_rng(seed)` from a fixture.
  - The base seed is fixed in the fast tier (reproducible PR CI).
  - A `TELESCOPIC_TEST_SEED` environment variable overrides it, dynesty-style.
  - A scheduled job loops 20–50 seeds through the statistical tier.
  - BlackJAX's date-rotation is a lighter alternative but makes a PR's result depend on the day it runs. I would not use it for the fast tier.
- **Tolerances.**
  - Never use hand-picked `atol` in statistical tests. Assert |estimate − truth| ≤ c·SE with c = 5, where SE comes from the run's own replicates, as in dynesty.
  - Use formal tests with α set by the family budget in Q1.
  - Keep a dedicated "golden values" file (Stan-style), checked exactly at a fixed seed, for regression detection only. Regenerate it deliberately when the RNG stream changes.
- **Flaky-test policy** (state it in CONTRIBUTING):
  - No `pytest-rerunfailures` in statistical tests.
  - A failing statistical test is first rerun once with 4× R and a new seed (sequential escalation, which keeps the overall error bounded). If it fails again, it is a bug until proven otherwise.
  - Skip markers require an issue link.
- **CI speed.**
  - Use `pytest -n auto` (xdist) and `--durations=25`.
  - Mark tests `slow` and `statistical`. The fast tier runs `-m "not slow"` and must finish in under 2 minutes.
  - The slow tier runs nightly and on demand.
  - Use `filterwarnings = error`, as in BlackJAX and NumPyro.
- **Hypothesis** with the `ci` profile, for estimator algebra, signed-measure identities, discrete maximal couplings (random pmfs), reflection algebra and input validation.
- **Oracle tests with injected randomness** for every coupled kernel (BlackJAX pattern).

### Gaps
- I did not inspect Stan's C++ sampler unit tests (stan-dev/stan `src/test/unit/mcmc`). The emcee and dynesty specifics come from the sibling note, not new inspection.
- No written flaky-test policy documents were found for BlackJAX, NumPyro or PyMC, only code-level practices.

## Q4. Benchmark targets with exact or reference answers, and literature meeting times, costs and inefficiencies

### Takeaway
The coupling literature uses a fairly small set of targets. For tests and documentation, telescopic should centre on targets with exact answers:
- Gaussians: independent, AR(1)/Toeplitz, Wishart-correlated, ill-conditioned.
- The AR(1) chain itself, which has exact TV, an exact asymptotic variance and an exact per-step meeting probability.
- The bimodal mixture.
- Heng & Jacob's banana, which has exact moments.
- Neal's funnel.
- Small Ising lattices (exact enumeration).
- Conjugate and finite-state models.
- The linear-Gaussian state-space model (Kalman).

For literature comparison (performance and docs), use German-credit logistic regression (three different dimensionalities appear in the literature), pump failure and baseball Gibbs, horseshoe and half-t regression, the 32 × 32 Ising model, variable selection, the log-Gaussian Cox process, stochastic volatility and particle MCMC. posteriordb supplies 57 reference posteriors as of 2026-09-25. A few of them (arK, garch11, low_dim_gauss_mix, blr, eight_schools_noncentered, gp_pois_regr) are easy to hand-code in NumPy. Literature meeting times depend on lag conventions (τ = inf{t ≥ 1: X_t = Y_{t−1}} versus inf{t : X_t = Y_t}), on cost units and on initialization, so any comparison must match all three.

### Cited Findings

#### A. Gaussian and autoregressive targets

| Target | Stresses | Exact or reference answer | Reported coupling results | Source |
|---|---|---|---|---|
| N(0,1), RWMH σ = 0.5, π0 = δ_10 | Bias from a far start; TV and W1 bounds | Exact TV and W1 (plotted) | L = 150 bounds "close to the exact distances"; N = 10⁴ | [Biswas et al. 2019, Sec. 2.2.1](https://arxiv.org/abs/1905.09971) |
| AR(1), φ = 0.99, π0 = N(0, 16), h(x) = x | Asymptotic-variance estimation; slow mixing | v = (1 − φ)⁻² = 10⁴; π = N(0, (1 − φ²)⁻¹) | k = L = 500, ℓ = 5k; UPAVE R = 1: [8049, 10417] at cost ≈ 5,250; R = 100: [9829, 10250] at cost ≈ 21,400 | [Douc et al., arXiv:2206.05691](https://arxiv.org/abs/2206.05691) |
| AR(1), ρ = 0.9, π0 = δ_10 (own) | Exact TV; heavy-tailed H; inefficiency | π_t = N(0.9ᵗ·10, 1 − 0.81ᵗ); asymptotic variance 19 for h = x | Lag-1 E[τ] ≈ 6.0, q90 = 13, q99 = 35 (pilot of 2,000); inefficiency ratio 1.89 at k = 35, m = 350 | Own: check_ar1.py (`scripts/validation/check_ar1.py`) |
| N(0, I_d) | Dimension scaling of RWMH, MALA and HMC couplings | Exact | Average meeting times (read from figure axes): RWMH up to about 3,000 at d = 10; MALA up to about 150 at d = 100; HMC about 40–55 for d up to 10,000. 1,000 chains, stationary start, ε = C·d^{−1/4}, σ = 10⁻³, γ = 1/20 | [Heng & Jacob 2019, Fig. 1](https://arxiv.org/abs/1709.00404) |
| N(0, I_10), RWM with σ² = 2.38²/d, stationary start | Proposal and acceptance coupling choices | Exact | Mean τ (SE) with acceptance coupling V = U / independent / V = 1 − U: maximal reflection 30 (0.8) / 51 (1.4) / 68 (2.0); semi-independent 54 / 85 / 105; optimal transport 104 / 155 / 183; maximal independent 279 (8.5) / 302 / 354. Here τ = min{t : X_t = Y_t}, 1,000 replications | [O'Leary 2021, Table 1](https://arxiv.org/abs/2102.01790) |
| N(0, V), V⁻¹ ~ Wishart(I, d) | Strong correlation; dense precision | Exact | RWMH with Σ = V/d: maximal coupling grows exponentially (average about 10⁴ by d = 5); reflection-maximal about linear (about 2,000 at d = 50). Gibbs: median up to about 600 at d = 9 (runs capped at 5·10⁵). HMC with 5% maximal-coupled RWMH (step covariance 10⁻⁸ I): average below about 60 up to d = 300 | [Jacob et al. 2020, Figs. 1–2, App. B](https://arxiv.org/abs/1708.03625) |
| N(0, V), V_ij = 0.5^{\|i−j\|} | Weak correlation | Exact | MH-within-Gibbs: average meeting time sublinear, below about 60 at d = 300 | [Jacob et al. 2020, Fig. 2b](https://arxiv.org/abs/1708.03625) |
| Same Toeplitz target, MALA vs ULA, π0 = N(0, I) | Exact versus approximate kernels | Exact | t_mix(0.25) bounds vs d up to 1,000, consistent with O(d) for ULA and O(d²) for MALA | [Biswas et al. 2019, Sec. 3.3](https://arxiv.org/abs/1905.09971) |
| Elliptical N(0, Σ), d = 2,000: Toeplitz ρ = 0.5 (eccentricity 5/3); diagonal with χ²₃ eigenvalues (3); diag(1, 24, …) (25/4) | Ill-conditioning; contraction of couplings | Exact | GCRN coupling "performing well", reflection and CRN baselines "impractical" (figures only) | [Papp & Sherlock, Sec. 5.3](https://arxiv.org/abs/2211.12585) |
| Ill-conditioned Gaussian (inference-gym): d = 100, eigenvalues 1/Gamma(0.5), seed 10 | Conditioning | Exact mean 0 and sd = √diag Σ | No coupling results found | [ill_conditioned_gaussian.py](https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/inference_gym/targets/ill_conditioned_gaussian.py) |

#### B. Non-Gaussian low-dimensional targets

| Target | Stresses | Exact or reference answer | Reported coupling results | Source |
|---|---|---|---|---|
| 0.5N(−4,1) + 0.5N(4,1), RWMH | Multimodality; initialization | ∫1(x > 3)π = 0.5Φ(1) + 0.5Φ(−7) ≈ 0.421 | σ_q² = 9, π0 = N(10, 10²): mean τ 20, q99 105. σ_q² = 1, same π0: mean 769, q99 9,186; k = 2·10⁴, m = 3·10⁴ gives CI [0.397, 0.430]. σ_q² = 1, π0 = N(10, 1): mean 9, q99 35, misleading CI [0.799, 0.816] (see Q2). Table 1 rows (k, m, cost, variance, inefficiency/V∞): (1, k, 37, 410, 1878); (100, 10k, 1019, 0.023, 2.9); (200, 10k, 2019, 0.0053, 1.3); (200, 20k, 4019, 0.0024, 1.2) | [Jacob et al. 2020, Sec. 5.1](https://arxiv.org/abs/1708.03625) |
| Same mixture, lag study | TV bounds under multimodality | — | L ∈ {1, 18,000}; N ∈ {10³, 5·10³, 10⁴} | [Biswas et al. 2019, Sec. 2.2.2](https://arxiv.org/abs/1905.09971) |
| Expo(1) target, proposal N(z + 3, 3) | Proposal–target tension; acceptance coupling | Exact | Mean τ (SE), 10⁴ replications, stationary start, τ = min{t ≥ 0 : X_t = Y_t}: P̄_SQ with Q̄_MI 74.0 (0.94); P̄_SQ with Q̄_MR 75.6 (0.99); P̄_MI 60.5 (0.84); P̄_MR 60.9 (0.87); P̄_C with Q̄_MI 61.3 (0.87); P̄_C with Q̄_MR 62.2 (0.89) | [O'Leary, Wang & Jacob 2021, Table 1](https://arxiv.org/abs/2010.08573) |
| Banana with U(x) = (1 − x1)² + 10(x2 − x1²)² | Curvature; non-convex potential | Exact: x1 ~ N(1, 1/2), x2 \| x1 ~ N(x1², 1/20), so E = (1, 1.5) and Var = (0.5, 2.55) (own derivation, verified with 4·10⁶ draws: check_banana.py (`scripts/validation/check_banana.py`)) | HMC with (ε, L, σ, γ) = (1/500, 500, 10⁻³, 1/20), π0 = U[−5, 5]², 1,000 runs: mean τ 158 with synchronous momenta, 52 with the Bou-Rabee κ = 1 coupling | [Heng & Jacob 2019, Sec. 5.2](https://arxiv.org/abs/1709.00404) |
| Banana (Haario et al.): x0 ~ N(0, 10²), x1 ~ N(0.03(x0² − 100), 1) | Curvature | Exact mean 0; sd = (10, √(1 + 2·0.03²·10⁴) = √19 ≈ 4.36, 1, …) | No coupling results found | [inference-gym banana.py](https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/inference_gym/targets/banana.py) |
| Hybrid Rosenbrock, n-dimensional | Curved correlation in any dimension | Analytically tractable, with direct samplers (R package "Rosenbrock") | No coupling results found | [Pagani, Wiegand & Nadarajah, arXiv:1903.09556](https://arxiv.org/abs/1903.09556) (Scand. J. Stat. 2022) |
| Neal's funnel: x0 ~ N(0, 3²), x_i ~ N(0, e^{x0}) (inference-gym default ndims = 10) | Varying scale; neck region | Exact mean 0; sd = (3, e^{9/4} ≈ 9.49, …) | No coupling results found | [neals_funnel.py](https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/inference_gym/targets/neals_funnel.py) |

#### C. Regression and hierarchical Gibbs or HMC targets

| Target | Stresses | Reference answer | Reported coupling results | Source |
|---|---|---|---|---|
| Pump failure (10 pumps), Gibbs | Conjugate Gibbs; comparison with regeneration | Posterior mean of β ≈ 2.47 | Maximal coupling of each conditional; k = 7 (q99), m = 70; efficiency 0.94, against 1.06 for regeneration and 1.08 for the plain Gibbs sampler | [Jacob et al. 2020, Sec. 5.2](https://arxiv.org/abs/1708.03625) |
| Baseball (K = 18, Morris 1983), Gibbs | Known convergence-time bound (140 iterations) | Long run | All τ ≤ 4 (1,000 runs). Table 4 (cost, variance, ratio): k = 3, m = 30: (33.07, 0.0002, 1.004); k = 1, m = 1: (5.15, 0.0070, 5.22); k = 5, m = 50: (53.07, 0.0001, 0.98, Monte Carlo noise) | [Jacob et al. 2020, App. C](https://arxiv.org/abs/1708.03625) |
| German credit, p = 49 indicators, n = 1,000; Pólya-Gamma Gibbs, prior N(0, 10I) | High-dimensional auxiliary variables | Long run | τ between 18 and 164, mean 48 (R = 1,000); k = 110, m = 1,100 gives inefficiency about 7× Gibbs | [Jacob et al. 2020, App. D](https://arxiv.org/abs/1708.03625) |
| German credit, d = 49: HMC (ε = 0.025, S ∈ {4,…,7}) vs PG Gibbs, π0 = N(0, 10I) | Comparing samplers' convergence | — | TV bounds smaller for PG than for every HMC setting | [Biswas et al. 2019, Sec. 3.2](https://arxiv.org/abs/1905.09971) |
| German credit with pairwise interactions, d = 302 (a, b, log s²) | HMC tuning for contraction | Long HMC run | (ε, L) = (0.0125, 10), σ = 10⁻³, γ = 1/20. Table 1 (cost, variance, relative inefficiency): k = 1: (436, 400, 1989); k = median: m = 5k (1258, 0.11, 1.58), m = 10k (2298, 0.045, 1.18); k = q90: m = 5k (1868, 0.058, 1.23), m = 10k (3518, 0.026, 1.05), about 5 minutes. HMC-optimal (0.03, 10) did not contract | [Heng & Jacob 2019, Sec. 5.3](https://arxiv.org/abs/1709.00404) |
| German credit, multinomial HMC couplings | HMC coupling variants | — | NUTS-adapted (ε, L) = (0.022, 22): mean τ 114 and 118 for the two couplings. Relative inefficiency, k = q90, m = 10k: Metropolis-HMC 2.32, maximal 1.90, W2 0.94 | [Xu et al. 2021, Table 1](https://arxiv.org/abs/2104.05134) |
| German credit numeric, d = 25 (inference-gym) | Standard benchmark | Ground-truth means from long Stan runs; generator defaults 10 chains × 150,000 iterations | — | [ground_truth file](https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/inference_gym/targets/ground_truth/german_credit_numeric_logistic_regression.py); [get_ground_truth.py](https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/inference_gym/tools/get_ground_truth.py) |
| Sonar logistic regression, (n, d) = (208, 61), prior N(0, 25I) | RWM vs MALA coupling efficiency | — | Optimizing mixing/meeting: RWM 2×/4× slower than MALA with a diagonal preconditioner; 7×/13× slower with a full one | [Papp & Sherlock, Sec. 7.4](https://arxiv.org/abs/2211.12585) |
| Pima and other binary regressions | Over-used benchmark | — | Chopin & Ridgway question whether moderate-size binary regression is a sound benchmark | [Chopin & Ridgway, arXiv:1506.08640](https://arxiv.org/abs/1506.08640) ([Stat. Sci. 2017, DOI 10.1214/16-STS581](https://doi.org/10.1214/16-STS581)) |
| Logistic regression with random effects, d = 507 (1,060 births) | High-dimensional HMC | Average of the R = 20 UPAVE estimates (1.76) is taken as the ground truth for the asymptotic variance | HMC unbiased-MCMC inefficiency 2.47 (about 40% above ergodic averages); MALA asymptotic variance about 65× HMC's | [Douc et al., Sec. 5.2](https://arxiv.org/abs/2206.05691) |
| Bayesian lasso, diabetes (n = 442, p = 64) | Sensitivity to the λ hyperparameter | Long run | 100 meeting times per λ ∈ [10⁻², 10³]; peak near λ = 10², matching the Gibbs ESS collapse | [Jacob et al. 2020, App. E](https://arxiv.org/abs/1708.03625) |
| Horseshoe and half-t(ν) regression; synthetic data with n = 100, s = 10, σ* = 0.5, β*_j = 2^{(9−j)/4} | Shrinkage-prior Gibbs; dimension | — | One-scale coupling: τ grows exponentially in p (p ≤ about 150). The two-scale coupling is orders of magnitude faster. ν slightly above 1 shortens τ by orders of magnitude | [Biswas et al. 2022, Secs. 2–3](https://arxiv.org/abs/2012.04798) (JRSSB 84(3):973) |
| GWAS, half-t(2): riboflavin (n = 71, p = 4,088) and maize (n = 2,266, p = 98,385) | Big p | — | Two-scale coupling, L = 200 and 750, 100 chains; converged in under 500 and under 1,000 iterations; about 0.1 s and 60 s per coupled iteration | [Biswas et al. 2022, Sec. 4](https://arxiv.org/abs/2012.04798) |
| Cut distribution, HPV (13 countries) | Nested or modular inference | — | Reflection-maximal RWMH; k = 100, m = 1,000; 10⁴ estimators; τ up to about 200 | [Jacob et al. 2020, Sec. 5.5](https://arxiv.org/abs/1708.03625) |
| Log-Gaussian Cox process, Finnish pines, d ∈ {256, 1024, 4096} | High-dimensional HMC and RMHMC | — | Relative inefficiency: HMC 11.00 / 5.43 / 2.73; RMHMC 11.68 / 7.85 / 3.72. About 90 and 20 minutes per estimator at d = 4,096 | [Heng & Jacob 2019, Sec. 5.4](https://arxiv.org/abs/1709.00404) |

#### D. Discrete targets

| Target | Stresses | Exact or reference answer | Reported coupling results | Source |
|---|---|---|---|---|
| Ising 32 × 32, periodic; π_θ ∝ exp(θ t(x)) | Critical slowing down; parallel tempering | Onsager's infinite-lattice critical point β_c = ln(1 + √2)/2 ≈ 0.4407. Small lattices by enumeration (own: 4 × 4, E[t] = 13.505 at θ = 0.3, 22.066 at 0.4, 25.050 at β_c, 28.086 at 0.5; ising_exact.py (`scripts/validation/ising_exact.py`)) | Single-site Gibbs: average τ rises above 10⁶ as θ approaches 0.48 (100 runs per θ). Parallel tempering with θ from 0.3 to 0.55 and p_swap = 0.01: minimum average τ at N = 16 chains. Cluster run: k = 10⁵, m = 2·10⁵; 500 processors × 30 min gave 1,858 estimators (1–7 per processor) | [Jacob et al. 2020, Sec. 5.3](https://arxiv.org/abs/1708.03625) |
| Ising 32 × 32 at β = 0.46 | TV bounds on discrete spaces | — | Single-site Gibbs: L = 10⁶, N = 500, bound plateaus. PT with 12 chains (β 0.3–0.46, swap frequency 0.02, L = 2·10⁴): faster | [Biswas et al. 2019, Sec. 3.1](https://arxiv.org/abs/1905.09971) |
| Variable selection (Yang et al. sampler), n ∈ {500, 1000}, p ∈ {1000, 5000}, SNR ∈ {0.5, 1, 2} | Discrete space with large p | Long MCMC | Average τ 4,729–46,398 (independent design) and 4,921–216,996 (correlated; maximum at n = 500, p = 1,000, SNR = 2). τ grows linearly in p. k = 75,000, m = 150,000; 600 processors × 120 min gave 39,282 estimators; largest τ 81,423 | [Jacob et al. 2020, Sec. 5.4](https://arxiv.org/abs/1708.03625) |
| Finite-state MH, K = 6 (own) | Exact everything | Exact pmf of τ, exact TV, exact E[H] | Lag-1 E[τ − 1] = 5.243 | Own: check_finite_state.py (`scripts/validation/check_finite_state.py`) |

#### E. State-space models and particle MCMC

| Target | Stresses | Exact or reference answer | Reported coupling results | Source |
|---|---|---|---|---|
| Hidden AR(1): x_t = 0.9x_{t−1} + N(0, 1), y_t ~ N(x_t, 1), T = 100; coupled conditional particle filter | Particle smoothing | Kalman smoother | N = 256 with ancestor sampling: k = 10, m = 20, cost about one particle filter with 28 × 256 particles. τ vs N (1,000 runs): N = 16 → 97, 128 → 15, 256 → 7, 512 → 4, 1,024 → 3. Table 1, mean τ (sd), N = 256, T = 100: bootstrap 13.16 (11.09), bootstrap + AS 7.59 (5.05), auxiliary 3.78 (1.99), auxiliary + AS 3.16 (1.09); roughly flat in T when N ∝ T | [Jacob, Lindsten & Schön 2020, Sec. 6.1](https://arxiv.org/abs/1701.02002) (JASA 115(530)) |
| Linear-Gaussian state-space model with priors a ~ U[0, 1], σ_X ~ Γ(2, 2); data T = 100 at a = 0.5, σ_X = 1 | Coupled PMMH | Exact likelihood via Kalman, allowing an "ideal" MH comparison | RW covariance 0.2²I (about 36.6% acceptance for exact MH); N ∈ [50, 250]; 20,000 repeats for meeting-time quantiles. PMMH meeting times have heavier tails than exact MH | [Middleton et al., Sec. 4.1](https://arxiv.org/abs/1807.08691) |
| Pseudo-marginal toy: N((1, 2), I) with log-normal noise σ ∈ {0, …, 2} | Polynomial meeting-time tails | Exact target | 10⁵ meeting times per σ; tails geometric at σ = 0, polynomial for larger σ | [Middleton et al., Sec. 3.1](https://arxiv.org/abs/1807.08691) |
| Stochastic volatility, d = 360, (β, φ, σ) = (0.65, 0.98, 0.15) | Eccentric high-dimensional target | Laplace approximation used for tuning | RWM with h = 2.38/Tr(Σ̂⁻¹)^{1/2} (about 23% acceptance); two-scale GCRN and GCRefl (δ = 0.1 and 0.001), L = 1.5·10⁶, 100 replicates. Reflection-maximal "severely hindered" | [Papp & Sherlock, Sec. 7.1](https://arxiv.org/abs/2211.12585) |

#### F. posteriordb (reference posteriors) and its Python API
- **Paper counts.** The paper reports "147 posteriors, 120 models, 91 datasets and 46 reference posterior draws" ([Magnusson et al. 2024, arXiv:2407.04967](https://arxiv.org/abs/2407.04967)).
- **Reference posterior definition.** A reference posterior is 10,000 or more draws.
  - Stan NUTS is used for well-behaved models.
  - Draws must have mean lag-1 autocorrelation below 0.05 in absolute value, R̂ < 1.01, and no divergences.
  - With 10,000 draws, Monte Carlo error is about sd/100, which "imposes an upper bound on the accuracy" of what can be evaluated.
  - Source: [Magnusson et al. 2024](https://arxiv.org/abs/2407.04967).
- **Repository today.** At commit `91027f9` (2026-09-25) the repository has 148 entries under `posteriors/` and 57 reference draw sets:
  - `Rate_{1,2,3,5}`, `arK-arK`, `arma-arma11`, `bball_drive_event_{0,1}-hmm_drive_{0,1}`, `diamonds-diamonds`, `dogs-dogs`, `dogs-dogs_hierarchical`
  - seven `earnings-*`, `eight_schools-eight_schools_noncentered`, `garch-garch11`, `gp_pois_regr-gp_pois_regr`, `gp_pois_regr-gp_regr`, `hmm_example-hmm_example`, `hudson_lynx_hare-lotka_volterra`
  - six `kidiq*`, `kilpisjarvi_mod-kilpisjarvi`, `low_dim_gauss_mix-low_dim_gauss_mix`, `mcycle_gp-accel_gp`, six `mesquite-*`, eight `nes19xx/2000-nes`, `nes_logit_data-nes_logit_model`
  - `one_comp_mm_elim_abs-one_comp_mm_elim_abs`, `radon_all-radon_pooled`, `radon_mn-radon_pooled`, `sblrc-blr`, `sblri-blr`, `wells_data-wells_dist`, `wells_data-wells_dist100_model`
  - The centered eight-schools posterior has no reference draws.
  - Source: [posteriordb repo](https://github.com/stan-dev/posteriordb).
- **Reference draw provenance and checks.** Reference info JSON, e.g. `sblrc-blr`, records `stan_sampling` with 10 chains, `iter` 20,000, `warmup` 10,000, `thin` 10 and `seed` 4711. The `checks_made` fields are `ndraws_is_10k`, `nchains_is_gte_4`, `ess_within_bounds`, `r_hat_below_1_01`, `efmi_above_0_2` and `abs_mean_lag1_ac_below_0_05` ([sblrc-blr.info.json](https://github.com/stan-dev/posteriordb/blob/master/posterior_database/reference_posteriors/draws/info/sblrc-blr.info.json)). `eight_schools_noncentered` uses the same settings with `adapt_delta` 0.95 ([posteriordb-python README](https://github.com/stan-dev/posteriordb-python)).
- **Python API** ([posteriordb-python README](https://github.com/stan-dev/posteriordb-python)):
  - `from posteriordb import PosteriorDatabase; pdb = PosteriorDatabase(path)`. There is also `PosteriorDatabaseGithub()`, which needs a `GITHUB_PAT`.
  - `pdb.posterior_names()` lists posteriors.
  - `po = pdb.posterior("eight_schools-eight_schools_noncentered")` returns a posterior.
  - `po.data.values()` returns the data as a dict.
  - `po.model.code("stan")` returns Stan code. There is no Python log density.
  - `po.reference_draws()` returns a list of chains, each a dict from parameter to draws.
  - `po.reference_draws_info()` returns provenance.
  - PyPI `posteriordb` is still at 0.2.0 (2020-11-25), and `inference-gym` at 0.0.5 (2025-01-24), checked via the PyPI JSON API today ([PyPI posteriordb](https://pypi.org/project/posteriordb/); [PyPI inference-gym](https://pypi.org/project/inference-gym/)).
- **Reference summaries** for NumPy-codable candidates, computed from the reference draws (own parse of [the draw archives](https://github.com/stan-dev/posteriordb/tree/master/posterior_database/reference_posteriors/draws/draws); 10 chains, 10,000 draws each). Values are mean (sd):

  | Posterior | Reference summaries |
  |---|---|
  | `eight_schools_noncentered` | μ 4.411 (3.309), τ 3.602 (3.198), θ₁ 6.151 (5.616) |
  | `arK-arK` | α −0.001 (0.011); β = 0.692, 0.439, 0.106, −0.035, −0.302 (sd 0.07–0.09); σ 0.151 (0.008) |
  | `garch-garch11` | μ 5.050 (0.124), α₀ 1.471 (0.572), α₁ 0.567 (0.127), β₁ 0.293 (0.125) |
  | `low_dim_gauss_mix` | μ = (−2.734, 2.870) (sd 0.04–0.06), σ = (1.028, 1.024), θ 0.622 (0.015). Stan model uses an `ordered[2]` μ |
  | `sblrc-blr` | β ≈ 1.000 (sd 0.001) for all 5 coefficients; σ 1.042 (0.077). Model: β ~ N(0, 10), σ ~ N⁺(0, 10), Gaussian likelihood |
  | `gp_pois_regr` | ρ 5.667 (0.679), α 2.921 (0.793) |

  Model code fetched from [posteriordb models](https://github.com/stan-dev/posteriordb/tree/master/posterior_database/models/stan).

#### G. Recent work (2025–2026)
- Multi-marginal couplings of MH ("grand couplings" of several chains) report meeting-time reductions "by up to 50%" against baselines ([Phan, Flamich, Khisti & Asoodeh, arXiv:2605.12807](https://arxiv.org/abs/2605.12807); [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/149629)).
- Unbiased kinetic Langevin with inexact gradients claims O(d^{1/4} ε⁻²) expected gradient evaluations, with a "number of gradient evaluations per effective sample" that does not depend on dimension. It is listed as accepted at the Annals of Statistics in 2026 ([Chada et al., arXiv:2311.05025](https://arxiv.org/abs/2311.05025); [CityU listing](https://scholars.cityu.edu.hk/en/publications/unbiased-kinetic-langevin-monte-carlo-with-inexact-gradients/)).

### Inferences
Recommended benchmark suite for telescopic, by tier.

**CI-fast (exact answers, under 30 s total)**
1. Finite-state MH (K = 6, cycle and independence proposals): exact τ pmf, TV and E[H].
2. AR(1) with ρ = 0.9 from δ_10: exact TV, exact one-step meeting probability, exact asymptotic variance 19.
3. Equal-covariance Gaussian maximal couplings, d ∈ {1, 10}: exact 1 − TV.
4. N(0, I_d), d = 2–10, RWMH with reflection-maximal coupling. Compare mean τ at d = 10 with [O'Leary 2021](https://arxiv.org/abs/2102.01790) Table 1 (30 ± 0.8 for maximal reflection with U = V, stationary start), converting conventions: O'Leary has no lag.
5. Conjugate Normal–Normal for linear SBC.
6. Small Ising (3 × 3 or 4 × 4, enumeration).

**Statistical tier, nightly (exact or analytic answers)**
1. Bimodal mixture: reproduce Jacob et al.'s Table 1 regime (σ_q² = 9: E[τ] ≈ 20, q99 ≈ 105).
2. Expo(1) with biased RW: reproduce O'Leary–Wang–Jacob Table 1 within 3 SE. Their 10⁴ replications give SE ≈ 0.9.
3. Banana with exact moments.
4. Funnel (exact moments, as a known-hard case; expect slow meeting).
5. AR(1) with φ = 0.99 (Douc et al.).
6. Toeplitz and Wishart Gaussians, d ≤ 50.
7. Linear-Gaussian state-space model against the Kalman smoother.
8. Dirichlet–Multinomial or Beta–Binomial Gibbs (conjugate, exact). This matters because Gibbs couplings of Gamma and Dirichlet conditionals need maximal couplings of non-Gaussian laws.

**Documentation and performance (literature comparison, reference answers)**
1. Pump failure (k = 7, m = 70).
2. Baseball (τ ≤ 4).
3. German credit Pólya-Gamma (d = 49; mean τ 48).
4. German credit HMC (d = 302; relative inefficiency 1.05).
5. Ising 32 × 32 with parallel tempering.
6. Half-t(2) regression on synthetic data.
7. posteriordb `eight_schools_noncentered`, `arK`, `garch11`, `low_dim_gauss_mix` and `blr`, hand-coded in NumPy and compared with the reference draws at tolerance ≥ 5 × (combined SE). The reference SE is sd/100.

**Convention checks before any comparison.** Match the lag (τ = inf{t ≥ 1: X_t = Y_{t−1}} versus inf{t : X_t = Y_t}), the cost unit (P̄ = 2 units; HMC = L + 2 gradient evaluations), the initialization (the literature often starts from the target), and the dataset variant (German credit at 25, 49 or 302 dimensions).

**Prefer exact targets for CI.** posteriordb is useful but needs a Stan toolchain (BridgeStan, not verified here) or hand ports to evaluate log densities. Hand-port only the few listed above.

### Gaps
- Several literature meeting-time figures (Heng & Jacob Fig. 1, Jacob et al. Figs. 1, 2 and 4, Biswas et al. 2022 Figs. 2–5, Papp & Sherlock) are plots only. Values above marked "about" were read from axis ranges and are approximate.
- I found no coupling-literature results for Neal's funnel, eight schools, Pima, or Dirichlet–multinomial targets. Those rows need our own baselines.
- The Douc et al. Section 5.2 numbers came from a garbled HTML extraction (leapfrog-step and step-size text was corrupted). Their AR(1) facts are reliable; the Section 5.2 tuning details are not.
- The posteriordb paper's E-FMI criterion was summarised by the fetch tool as "below 0.2". The JSON check name is `efmi_above_0_2`, so I treat "above 0.2" as correct.
- I did not open Nguyen et al. (optimal-transport couplings for partition Gibbs samplers) or the phylogenetics lagged-coupling papers, which may supply Gaussian-mixture or clustering coupling benchmarks.
- BridgeStan's Python API was not verified.

## Q5. Performance benchmarking for unbiased MCMC: cost per estimate, inefficiency vs ESS per second, asv vs scripts

### Takeaway
The literature's figure of merit is the asymptotic inefficiency E[cost] × Var(H) (Glynn & Whitt). It is reported as a ratio to the MCMC asymptotic variance (relative inefficiency), with costs counted in kernel applications (P̄ = 2 units). Meeting-time quantiles drive the choice of k and m. Wall-clock versions exist: Papp & Sherlock report seconds per 10 effective samples and average meeting time in seconds.

For telescopic, the right tooling is:
- a deterministic benchmark script suite that writes JSON and tables of meeting-time quantiles, E[cost], Var(H), relative inefficiency (exact denominators where known) and wall time per kernel unit;
- asv `track_*` benchmarks, or pytest-benchmark with github-action-benchmark, for regression tracking of a few headline numbers.

### Cited Findings
- **Asymptotic inefficiency** ([Jacob et al. 2020, Sec. 3.1](https://arxiv.org/abs/1708.03625)):
  - It is "the product of the variance and expected cost of the estimator", which is "the asymptotic variance of R⁻¹ Σ H … as the computational budget … goes to infinity" (Glynn & Whitt 1992).
  - Cost is "comparable to 2(τ − 1) + max(1, m + 1 − τ) iterations", since calls to P̄ are about twice as expensive as calls to P.
  - The loss of efficiency relative to MCMC is approximately 1 + (k + E[τ])/(m − k).
  - In the examples the denominator V∞ comes from CODA on long runs (10⁶ iterations for the mixture).
- **Lag-L cost** is L + 2(τ − L) + max(0, ℓ − τ); recommended ℓ = 10k ([Atchadé & Jacob 2024](https://arxiv.org/abs/2406.06851)).
- **HMC conventions** ([Heng & Jacob 2019, Secs. 5.1, 5.3](https://arxiv.org/abs/1709.00404)):
  - Inefficiency is defined the same way, "assuming that applying K̄ costs twice as much as K".
  - The comparator is `spectrum0.ar` on 10,000 iterations after 1,000 burn-in.
  - First and second moments are summed over coordinates.
  - Tuning cost is "L + 2 times the average meeting time".
  - The relative inefficiency table compares against the optimally tuned HMC rather than the HMC used in the coupling.
- **Variable parameters across columns.** In Xu et al., k differs per method, so "inefficiencies across different coupled kernels … are not directly comparable". Some ratios fall below 1, e.g. 0.94 ([Xu et al. 2021](https://arxiv.org/abs/2104.05134)).
- **Wall time** ([Papp & Sherlock, Sec. 7.4 and Fig. 8](https://arxiv.org/abs/2211.12585)):
  - They plot average meeting time in seconds against "seconds per 10 effective samples when at equilibrium" (smallest ESS over coordinates).
  - They use EPAVE unbiased asymptotic-variance estimates.
  - "Mixing efficiency is more relevant when a small number of long chains are run, and coupling efficiency is more relevant when a large number of short chains are run".
- **Unbiased asymptotic-variance estimators** UPAVE and EPAVE give exact-in-expectation denominators. Batch-means and spectral estimators "show bias" and have slower MSE decay ([Douc et al.](https://arxiv.org/abs/2206.05691)).
- **Wall-clock framing in other papers:**
  - Heng & Jacob report about 5 minutes per estimator for d = 302 logistic regression, and "one can only run 4439 and 714 iterations" of HMC and RMHMC in the time of one estimator for the d = 4,096 Cox process ([Heng & Jacob 2019](https://arxiv.org/abs/1709.00404)).
  - Biswas et al. report about 60 s per coupled iteration on maize ([Biswas et al. 2022](https://arxiv.org/abs/2012.04798)).
  - Jacob et al. report processor-time budgets: 500 processors × 30 min, and 600 × 120 min ([Jacob et al. 2020](https://arxiv.org/abs/1708.03625)).
- **asv** benchmark types are `time_`, `mem_`, `peakmem_`, `timeraw_` and `track_`. "It is also possible to use `asv` to track any arbitrary numerical value. … These functions simply need to return a numeric value", with a `unit` attribute. Benchmarks can be parameterized with `params` and `param_names`, and `setup_cache` runs once per environment and commit ([asv: writing benchmarks](https://asv.readthedocs.io/en/stable/writing_benchmarks.html)).
- **BlackJAX** tracks performance with pytest-benchmark (`pytest --benchmark-only --benchmark-json output.json`) and `benchmark-action/github-action-benchmark`, using `alert-threshold: '200%'` and `fail-on-alert: true` ([benchmark.yml](https://github.com/blackjax-devs/blackjax/blob/main/.github/workflows/benchmark.yml)). Its benchmark file says it is "only a first draft" and that ESS per gradient evaluation and ESS per second should be added ([tests/test_benchmarks.py](https://github.com/blackjax-devs/blackjax/blob/main/tests/test_benchmarks.py)).
- **Stan's** performance repository uses `runPerformanceTests.py --runs N` with gold checks ([performance-tests-cmdstan](https://github.com/stan-dev/performance-tests-cmdstan)).
- **posteriordb for benchmarking.** The paper frames use cases as testing, development, benchmarking and regression testing. It suggests RMSE, Wasserstein, MMD and Pareto-k̂ comparisons with the reference, and warns that "we needed to pick appropriate posteriors for the experimental goal" ([Magnusson et al. 2024](https://arxiv.org/abs/2407.04967)).

### Inferences

**Headline metrics.** For each (target, kernel, coupling, k, m, L) cell, report:
1. Meeting-time distribution: mean, q50, q90, q99, max, and the survival curve P(τ − L > t) on a log scale, to expose tails.
2. E[cost] in kernel units (P̄ = 2), plus gradient or density evaluations, plus wall time per unit.
3. Var(H) with a bootstrap CI.
4. Inefficiency E[cost]·Var(H), and its ratio to the MCMC asymptotic variance. Use exact denominators for Gaussian and AR(1) targets: (1 + ρ)/(1 − ρ) for AR(1); UPAVE or spectral-with-CI elsewhere.
5. ESS per second of the marginal chain, for comparison.
6. Tail diagnostics for H: kurtosis, max|H|/sd.

**Parallel performance.** Report the simulated completion time for P processors: the expected maximum of costs. It grows like log P under geometric tails ([Jacob et al. 2020, Sec. 3.3](https://arxiv.org/abs/1708.03625)).

**Tooling.**
- Keep `benchmarks/` as plain scripts (argparse, fixed seeds, `np.random.SeedSequence` spawning) that write JSON and Markdown tables, which docs and papers reuse.
- Add a thin asv layer: `track_` functions returning, for example, mean τ at a fixed seed for 3–5 headline cells, plus `time_` functions for a single coupled step and a single estimator on AR(1), d-dimensional Gaussians and German credit.
- asv suits per-commit history. Statistical benchmarks need about 10³–10⁵ replicates, so they belong in scripts run on demand or nightly.
- Use pytest-benchmark with github-action-benchmark only if per-PR alerting is wanted, and use wide thresholds (BlackJAX uses 200%).

**Literature reproduction table in docs.** Include the Q4 rows that can be reproduced cheaply:
1. Bimodal mixture, Table 1 (σ_q² = 9).
2. Expo(1) targets, O'Leary–Wang–Jacob Table 1.
3. O'Leary Table 1 (d = 10).
4. Baseball Table 4.
5. Pump failure.
6. German credit HMC Table 1. This is costly but within reach.

Each reproduction should show the conventions matched (lag, cost units, initialization).

### Gaps
- I found no standard, cross-paper benchmark protocol for unbiased MCMC. Each paper defines its own k and m rules and denominators (CODA spectrum0, `spectrum0.ar`, EPAVE or UPAVE), which complicates direct comparison.
- I did not verify how other ecosystems (e.g. Julia's Pigeons, Turing) benchmark coupled or unbiased samplers.
- Wall-clock comparisons across implementations (R, JAX, NumPy) are not available in the literature I read. Papers report their own implementation's timings only.

# Software landscape: unbiased MCMC with couplings and coupling-based diagnostics (census as of 2026-10-23)

How this census was made: GitHub repository and code search (GitHub API via MCP, 2026-10-23); read-only `git clone` of every candidate repository (full history for research repos, shallow or blob-less clones for large libraries), with commit dates, commit counts, licences, file trees, READMEs and source read from the clones; PyPI JSON API and a scan of the full PyPI project index (920,853 names); the CRAN package list (25,498 packages); the Julia General registry `Registry.toml`; pypistats; Crossref/OpenAlex/Semantic Scholar for citation counts; GitHub issue/PR pages for library plans. Nothing downloaded was executed. Star/fork counts are from the GitHub search API on 2026-10-23. Commit dates are committer dates from the clones. "Not on PyPI/CRAN/registry" means the name lookup returned 404 / no match on 2026-10-23.

## Q1. What implementations exist, how good are they, how maintained, how adopted?

### Takeaway
There is no released, maintained package for unbiased MCMC with couplings in any language: nothing is on PyPI, CRAN or the Julia General registry. The reference implementation is Pierre Jacob's GitHub-only R package `unbiasedmcmc` (GPL ≥ 2, v0.3.1, 33 stars, 59 commits 2017–2026). Its README says it is "not a general-purpose statistical software". Everything else is paper-companion research code (R, Julia, Python, MATLAB), dormant Python/JAX ports written as essays or course projects (all with 0–20 stars, none packaged), or, since September 2026, a kernel-level coupled-HMC primitive in BlackJAX that explicitly does not detect meeting or build estimators.

### Cited Findings

**Census table.** Each row's source is the linked repository: clone of 2026-10-23 for dates, commits, licence and tests; GitHub search API for stars.

| Implementation | Lang | What it is | Licence | Commits (first → last; n) | Released on an index? | Stars | Tests / docs |
|---|---|---|---|---|---|---|---|
| [pierrejacob/unbiasedmcmc](https://github.com/pierrejacob/unbiasedmcmc) | R + C++ (Rcpp/RcppEigen) | Reference code for Jacob, O'Leary & Atchadé (JRSSB 2020) | GPL (≥ 2) (DESCRIPTION, no LICENSE file) | 2017-07-31 → **2026-09-23**; 59 | No (GitHub only, v0.3.1, no tags) | 33 (9 forks) | No automated tests (18 manual `inst/check` scripts); README + 2 vignettes + roxygen man pages |
| [pierrejacob/debiasedhmc](https://github.com/pierrejacob/debiasedhmc) | R + C++ | Heng & Jacob (Biometrika 2019) unbiased HMC | GPL (≥ 2) | 2017-08-29 → 2018-08-27; 7 | No | 5 (4 forks) | None; per-example READMEs |
| [jeremyhengjm/debiasedhmc](https://github.com/jeremyhengjm/debiasedhmc) | R | Heng's copy of the same paper code | GPL (≥ 2) | 2017-11-09 → 2017-12-11; 18 | No | 0 | None |
| [pierrejacob/couplingsmontecarlo](https://github.com/pierrejacob/couplingsmontecarlo) | R | Lecture scripts "Couplings and Monte Carlo" (coupled MH/HMC/Gibbs, Rhee–Glynn, CFTP) | GPL-3 (LICENSE), GPL (≥ 3) (DESCRIPTION) | 2020-02-19 → 2020-07-21; 25 | No | 33 (7 forks, 4 open issues) | None |
| [pierrejacob/umcmchandbook](https://github.com/pierrejacob/umcmchandbook) | R | Figures for Atchadé & Jacob, "Unbiased MCMC: what, why and how" | Placeholder ("What license is it under?") | 2022-05-15 → 2024-06-02; 13 | No | 1 | None |
| [pierrejacob/unbiasedpoisson](https://github.com/pierrejacob/unbiasedpoisson) | R | Douc, Jacob, Lee & Vats, Poisson equation via coupled chains | MIT | 2022-05-15 → 2025-07-15; 11 | No | 0 | No test suite; tutorial Rmd |
| [pierrejacob/unbiasedsnis](https://github.com/pierrejacob/unbiasedsnis) | R | Coupling-based unbiased self-normalised IS / independent MH (arXiv 2411.09514) | MIT | 2025-07-18 → **2026-08-31**; 7 | No | 0 | None |
| [pierrejacob/CoupledCPF](https://github.com/pierrejacob/CoupledCPF) | R | Coupled conditional particle filters, Rhee–Glynn smoothing (arXiv 1701.02002) | GPL (≥ 2) | 2016-03-16 → 2018-09-03; 15 | No | 3 | None |
| [pierrejacob/unbiasedpathsampling](https://github.com/pierrejacob/unbiasedpathsampling) | R | Unbiased log normalising constants (arXiv 1810.01382) | GPL (≥ 2) | 2018-09-28 → 2018-10-02; 4 | No | 0 | None |
| [niloyb/LlagCouplings](https://github.com/niloyb/LlagCouplings) | R (+C++) | L-lag coupling TV/Wasserstein bounds (Biswas, Jacob & Vanetti, NeurIPS 2019) | MIT | 2019-08-20 → 2022-02-01; 92 | No (scripts, no DESCRIPTION) | 9 | `coupling_test.R`; Rmd tutorial at niloyb.github.io/LlagCouplings |
| [niloyb/CoupledHalfT](https://github.com/niloyb/CoupledHalfT) | R + C++ | Coupled Gibbs for half-t shrinkage regression (JRSSB 2022) | Placeholder ("What license is it under?"), no LICENSE | 2020-12-15 → 2022-08-27; 14 | No | 3 | None; README.Rmd |
| [niloyb/BoundWasserstein](https://github.com/niloyb/BoundWasserstein) | R | Biswas & Mackey, bounding Wasserstein with couplings | MIT | 2021-11-30 → 2023-10-24; 39 | No | 3 | None |
| [johnoleary/mh-max-couplings](https://github.com/johnoleary/mh-max-couplings) | Python | O'Leary, Wang & Jacob (AISTATS 2021) maximal MH couplings | GPL-3 | 2021-02-11 → 2021-02-19; 15 | No | 8 | None; one-line README |
| [tamaspapp/rwmcouplings](https://github.com/tamaspapp/rwmcouplings) | R + C++ | Papp & Sherlock, scalable RWM couplings (CRN, GCRN, reflection) | GPL-3 | 2022-11-22 → 2024-10-28; 5 | No (v1.3) | 1 | None |
| [chalk-lab/CoupledHMC.jl](https://github.com/chalk-lab/CoupledHMC.jl) | Julia | Xu, Fjelde, Sutton & Ge (AISTATS 2021) couplings for multinomial HMC | **None** (no LICENSE file) | 2020-08-10 → 2022-02-28; 49 | No (not in General registry) | 16 (4 forks, 3 open issues) | None; README + notebooks |
| [AdrienCorenflos/coupled_rejection_sampling](https://github.com/AdrienCorenflos/coupled_rejection_sampling) | Python/JAX | Corenflos & Särkkä coupled rejection sampler | **None effective** (LICENSE file is 0 bytes) | 2022-01-26 → 2022-05-25; 13 (tags v1–v1.2) | No | 3 | 6 pytest files; "A proper readme is coming." |
| [AdrienCorenflos/CoupledPDMPs](https://github.com/AdrienCorenflos/CoupledPDMPs) | Julia | Debiasing PDMP samplers (BPS, Boomerang, ZigZag) with couplings | MIT | 2022-08-01 → 2025-04-29; 48 | No | 1 | `test/` (11 files) |
| [AdrienCorenflos/ImportanceMCMC](https://github.com/AdrienCorenflos/ImportanceMCMC) | Julia | Coupling-based f-divergence diagnostics (arXiv 2510.07559) + L-lag TV bounds | MIT | 2025-11-03 → **2026-07-20**; 48 | No (v1.0.0, unregistered) | 1 | `test/` (8 files) |
| [tinnguyen96/partition-coupling](https://github.com/tinnguyen96/partition-coupling) | Python (+POT) | Nguyen, Trippe & Broderick (AISTATS 2022) OT couplings of partition Gibbs | MIT | 2020-11-22 → 2022-02-21; 21 | No | 5 | None |
| [paoloceriani/couplings_bgs](https://github.com/paoloceriani/couplings_bgs) | Python | Ceriani & Zanella, couplings of blocked Gibbs samplers | **None** | 2024-09-19 → 2025-07-03; 20 | No | 0 | None |
| [awllee/CoupledConditionalSMC.jl](https://github.com/awllee/CoupledConditionalSMC.jl) | Julia | Lee, Singh & Vihola coupled conditional backward-sampling PF | MIT (Expat) | 2018-08-13 → 2019-08-28; 32 | No (Julia 1.0 era) | 1 | Travis/AppVeyor badges |
| [maximerischard/MaximalCouplings.jl](https://github.com/maximerischard/MaximalCouplings.jl) | Julia | Maximal coupling of two distributions | MIT (Expat) | 2017-10-10 → 2019-02-03; 25 | No | 0 | Travis badge |
| [traitlab-mcmc/TraitLab](https://github.com/traitlab-mcmc/TraitLab) + [lukejkelly/CoupledPhylogenetics](https://github.com/lukejkelly/CoupledPhylogenetics) | MATLAB (+R scripts) | Lagged couplings of phylogenetic MCMC (Kelly, Ryder & Clarté, AoAS 2023) | **None** found in either repo | TraitLab 2016-09-01 → 2023-08-21 (303); scripts 2021-01-14 → 2022-09-16 (100) | No | 0 / 0 | `coupling/*Test.m` files; TraitLab manual |
| [ruzayqat/unbiased_uld](https://github.com/ruzayqat/unbiased_uld) | MATLAB | Unbiased estimation with underdamped Langevin | GPL-3 | 2022-06-14 → 2023-06-21; 4 | No | 0 | None |
| [ColCarroll/couplings](https://github.com/ColCarroll/couplings) | Python (NumPy/SciPy) | Library and essay for coupled MH | MIT | 2019-08-16 → 2019-09-19; 42 | No (`couplings` 404 on PyPI) | 20 | pytest (3 files) + Azure CI badge; essay site |
| [ylefay/unbiased_mcmc_with_couplings](https://github.com/ylefay/unbiased_mcmc_with_couplings) | Python/JAX | Course project: coupled MH, Thorisson coupling (`pymcmc_unbiased`) | MIT | 2024-02-05 → 2025-02-01; 58 | No (`pymcmc_unbiased` 404) | 1 | 3 pytest files |
| [Helmuthn/HMC_unbiased](https://github.com/Helmuthn/HMC_unbiased) | Python/JAX | "Quick implementation" of Heng & Jacob unbiased HMC | MIT text | 2023-07-17 → 2023-12-13; 17 | No | 0 | 2 test files; mkdocs site |
| [carsten-j/mc-lab](https://github.com/carsten-j/mc-lab) | Python | Educational MC lab; unbiased-MCMC **scripts only** | MIT | 2025-08-12 → 2025-11-12; 111 | `mc-lab-edu` 0.2.7 on PyPI, but coupling code is not in the package | 0 | Tests for the packaged part |
| [darrenjw/unbiased-mcmc](https://github.com/darrenjw/unbiased-mcmc) | R + Scala | Teaching/discussion material (RSS read paper, Dec 2019) | GPL-3 | 2019-09-22 → 2023-09-07; 78 | No | 5 | n/a |
| [rachtsingh/debiasedmcmc](https://github.com/rachtsingh/debiasedmcmc), [rachtsingh/coupled_kernels](https://github.com/rachtsingh/coupled_kernels) | CUDA; PyTorch C++ ext | GPU experiments (2017–2018) | None | 2017-10-23 → 2018-01-05 (7); 2018-06-12 (3) | No | 1 / 1 | None |
| [cjm96/mcmc_infinity](https://github.com/cjm96/mcmc_infinity) | Python/JAX (+flowjax) | *Adjacent:* perfect sampling (coupling from the past) for astronomy | MIT | 2025-07-15 → 2025-07-22; 81 | No (`mcmc_infinity` 404; v0.0.1) | 9 | 2 test files; notebooks |
| Course projects: [adriens63 (Py)](https://github.com/adriens63/mcmc_unbiased_gibbs_sampler_py), [arijitdey22 (R)](https://github.com/arijitdey22/Maximal_Coupling_of_MCMC), [ruslenghi (Java)](https://github.com/ruslenghi/Coupled-Markov) | various | Student projects | Apache-2.0 / none / none | 2022 / 2024 / 2020 | No | 0 | n/a |
| [blackjax `mcmc/coupled_hmc.py`](https://github.com/blackjax-devs/blackjax/blob/main/blackjax/mcmc/coupled_hmc.py) | Python/JAX | Pair of HMC chains sharing random inputs (synchronous or reflection); **no meeting or estimators** | Apache-2.0 | Added 2026-09-07 (PR #1033); in releases 1.7.0/1.7.1 | Yes (part of `blackjax` on PyPI) | (BlackJAX: 1,136) | 1,556-line test file |
| [TFP FunMC `maximal_reflection_coupling_proposal`](https://github.com/tensorflow/probability/blob/main/spinoffs/fun_mc/fun_mc/fun_mc_lib.py) | Python (JAX/TF) | Proposal-level maximal reflection coupling primitive | Apache-2.0 | Present since FunMC was added, 2021-04-02 | FunMC not found on PyPI (`fun_mc` / `fun-mc` 404) | (TFP: 4,440) | `fun_mc_test.py` |

**A. Reference R code from the Jacob group**
- `unbiasedmcmc`: DESCRIPTION title "Unbiased MCMC with couplings", Version 0.3.1, authors Jacob, O'Leary and Atchadé, License "GPL (>= 2)", depends on Rcpp, RcppEigen and lubridate — [DESCRIPTION](https://github.com/pierrejacob/unbiasedmcmc/blob/master/DESCRIPTION)
- Its README says: "This is not a general-purpose statistical software, but a collection of scripts intended to reproduce figures and tables. Use at your own risk!!" Installation is `devtools::install_github("pierrejacob/unbiasedmcmc")` — [README](https://github.com/pierrejacob/unbiasedmcmc/blob/master/README.md)
- Commits per year: 12 (2017), 3 (2018), 40 (2019), 1 (2020), 1 (2022), 2 (2026). The last commit (2026-09-23) "resolves a bug in the Polya-Gamma Gibbs sampler's coupling. Bug spotted by Tamas Papp." So the code is patched occasionally, but there are no tags or releases — [commit history](https://github.com/pierrejacob/unbiasedmcmc/commits/master)
- Not on CRAN: `https://cran.r-project.org/package=unbiasedmcmc` returns "Object not found!" — [CRAN](https://cran.r-project.org/package=unbiasedmcmc)
- Exported features: the generic drivers `sample_meetingtime`, `sample_coupled_chains`, `sample_unbiasedestimator`, `H_bar`, `c_chains_to_measure_as_list` and `histogram_c_chains`. Coupling primitives: `rnorm_max_coupling`, `rmvnorm_max`, `rnorm_reflectionmax`, `rmvnorm_reflectionmax`, `get_max_coupling`, `rgamma_coupled`, `rinversegamma_coupled` and `rinvgaussian_coupled`. Model-specific kernels: `get_mh_kernels`, Ising Gibbs and parallel-tempering couplings (`ising_*`, `ising_pt_*`), `get_blasso`, `get_variableselection`, `pg_gibbs` and logistic-regression helpers — [NAMESPACE](https://github.com/pierrejacob/unbiasedmcmc/blob/master/NAMESPACE)
- Docs: two vignettes (`introduction-normaltarget.Rmd`, `polyagammagibbs.Rmd`). Parallelism is via `foreach(...) %dorng%` in the vignette — [vignettes](https://github.com/pierrejacob/unbiasedmcmc/tree/master/vignettes)
- API example, condensed from the README:
  ```r
  kernels <- get_mh_kernels(target, 4)          # target = log-density; RWMH + reflection-maximal coupling
  rinit <- function(){ x <- rnorm(1, 3, 2); list(chain_state = x, current_pdf = target(x)) }
  sample_meetingtime(kernels$single_kernel, kernels$coupled_kernel, rinit)$meetingtime
  cc <- sample_coupled_chains(kernels$single_kernel, kernels$coupled_kernel, rinit, m = 2000, lag = 500)
  H_bar(cc, h = function(x) x, k = 500, m = 2000)   # unbiased estimate of E[h(X)]
  ```
  — [README](https://github.com/pierrejacob/unbiasedmcmc/blob/master/README.md)
- `debiasedhmc` (Heng & Jacob): `coupled_chains`, `H_bar`, `unbiased_estimator(logtarget, single_kernel, coupled_kernel, rinit, h, k, m, max_iterations)`, `get_hmc_kernel(logtarget, gradlogtarget, stepsize, nsteps, dimension)`, plus MALA, MH, Riemannian-manifold HMC/MALA kernels and a coupled truncated-MVN sampler taken from CRAN `tmg` — [R/debiasing_algorithm.R](https://github.com/pierrejacob/debiasedhmc/blob/master/R/debiasing_algorithm.R), [R/hmc_kernel.R](https://github.com/pierrejacob/debiasedhmc/blob/master/R/hmc_kernel.R), [README](https://github.com/pierrejacob/debiasedhmc/blob/master/README.md)
- `debiasedhmc`'s banana example compares a synchronous coupling ("common random numbers for the initial momentum") with a "contractive coupling" ("a mix of reflection coupling and maximal coupling for the initial momentum") after Bou-Rabee, Eberle & Zimmer — [inst/banana/README.R](https://github.com/pierrejacob/debiasedhmc/blob/master/inst/banana/README.R)
- No public `jeremyhengjm/UnbiasedHMC` repository exists: the clone failed, and the user's 10 public repositories include `debiasedhmc`, `UnbiasedScore` and `UnbiasedMultilevel` but no UnbiasedHMC — [jeremyhengjm repositories](https://github.com/jeremyhengjm?tab=repositories)
- `couplingsmontecarlo` README warns: "this is very very drafty… please don't waste your energy fixing/improving the code right now, I might not accept your pull requests." Chapter-4 scripts include `4coupled_mh.R`, `4unbiasedhmc.R`, `4rheeglynnhmc.R` and `4cftp_simple.R` — [README](https://github.com/pierrejacob/couplingsmontecarlo/blob/master/README.md)
- Pierre Jacob's "Unbiased MCMC" companion site (last updated October 2025) has a Code page. It lists R (`umcmchandbook`, `unbiasedmcmc`, `unbiasedpoisson`), Python (ColCarroll `couplings`, `mh-max-couplings`, `partition-coupling`) and Julia (`CoupledConditionalSMC.jl`) — [site](https://pierrejacob.quarto.pub/unbiased-mcmc), [code page](https://pierrejacob.quarto.pub/unbiased-mcmc/code.html)

**B. Coupling-based diagnostics code**
- `LlagCouplings` is a set of scripts with a tutorial. It provides `simulate_meeting_time(single_kernel, coupled_kernel, rinit, max_iterations = Inf, L = 1, cmp = default_cmp)`, `simulate_ipm_bounds`, `tv_upper_bound_estimates(coupling_times, L, t)` (= `pmax(0, ceiling((coupling_times-L-t)/L))`) and Wasserstein-bound estimators — [coupling.R](https://github.com/niloyb/LlagCouplings/blob/master/coupling.R), [tv_wasserstein_bounds.R](https://github.com/niloyb/LlagCouplings/blob/master/tv_wasserstein_bounds.R)
- The `LlagCouplings` README records an erratum: "There was a bug in the code used to produce Figure 5… The choice of step-size for the MALA and ULA chains was not set in the code as it was described in the paper" — [readme.md](https://github.com/niloyb/LlagCouplings/blob/master/readme.md)
- `CoupledHalfT` API (README): `coupled_half_t_mcmc(mc_chain_size=1, X, X_transpose, y, t_dist_df=t_dist_df, lag=lag)` returns `$meetingtime` and both chains. `half_t_mcmc(...)` runs a single chain. The DESCRIPTION is the unedited `Rcpp.package.skeleton` template ("Title: What the package does (short line)", "License: What license is it under?") — [README.md](https://github.com/niloyb/CoupledHalfT/blob/master/README.md), [DESCRIPTION](https://github.com/niloyb/CoupledHalfT/blob/master/DESCRIPTION)
- `ImportanceMCMC.jl` implements "A coupling-based approach to f-divergences diagnostics for Markov chain Monte Carlo" (arXiv:2510.07559) and, for comparison, L-lag couplings. It exports `simulate_meeting_time`, `tv_upper_bound_estimates`, coupled MALA, pCNL and particle-Gibbs kernels, `harmonization_step!` and divergence estimators. README example: `τ = simulate_meeting_time(single_step!, coupled_step!, rinit; L=50)` — [README](https://github.com/AdrienCorenflos/ImportanceMCMC/blob/main/README.md), [src](https://github.com/AdrienCorenflos/ImportanceMCMC/tree/main/src)
- TraitLab's README lists "Couplings to diagnose convergence and construct unbiased estimators: Kelly, Ryder and Clarté (2021)". The repository has a `coupling/` directory (67 `.m` files mentioning coupling, including `*Test.m` unit tests) — [TraitLab README](https://github.com/traitlab-mcmc/TraitLab/blob/master/README.md). `CoupledPhylogenetics` holds the experiment scripts (MATLAB simulation plus R analysis) — [README](https://github.com/lukejkelly/CoupledPhylogenetics/blob/main/README.md)

**C. Kernel-coupling research code**
- `mh-max-couplings` (O'Leary): object-oriented NumPy/pandas code. `CoupledMH(FullKernel, Recorder)` takes `(init, prop, target, …)` and has `.run(n_iter, n_rep=1, …)`. Proposal classes offer modes `prop2x_indep/syn/refl/semi/max/hybrid`, and the notebooks sweep acceptance modes such as `('max_refl','same_u')` and `('max_indep','cond_same_u')`. Requirements are pinned to `numpy~=1.20.1`, `pandas~=1.2.2` and `matplotlib~=3.3.4`. There are no estimator functions; the code focuses on meeting-time experiments — [src/mcmc.py](https://github.com/johnoleary/mh-max-couplings/blob/main/src/mcmc.py), [src/proposals/norm.py](https://github.com/johnoleary/mh-max-couplings/blob/main/src/proposals/norm.py), [requirements.txt](https://github.com/johnoleary/mh-max-couplings/blob/main/requirements.txt)
- `rwmcouplings` (Papp & Sherlock, "Scalable couplings for the random walk Metropolis algorithm", JRSSB 2025, arXiv 2211.12585) provides `RWM_crn`, `RWM_gcrn`, `RWM_refl` and `RWM2targets_{crn,gcrn,refl,optimal}` — [repo man/](https://github.com/tamaspapp/rwmcouplings/tree/main/man), [arXiv 2211.12585](https://arxiv.org/abs/2211.12585)
- `CoupledHMC.jl` builds on AdvancedHMC. Its types are `CoupledHMCSampler(rinit, TS, ϵ, L, γ=1/20, σ=1e-3, κ=0.0, momentum_refreshment)`; trajectory-index couplings `IndependentCoupling`, `QuantileCoupling`, `MaximalCoupling`, `OTCoupling` and `ApproximateOTCoupling` (JuMP/Clp/OptimalTransport); `SharedRefreshment` and `ContractiveRefreshment`; and MH mixture kernels `MaxCoupledMH`/`RefMaxCoupledMH`. Usage is `samples = sample_until_meeting(target, alg; n_samples_max=…)` — [src/CoupledHMC.jl](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/src/CoupledHMC.jl), [src/couplings.jl](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/src/couplings.jl), [research/scripts/banana.jl](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/research/scripts/banana.jl)
- `CoupledHMC.jl` README: "you will have to install `VecTargets` (https://github.com/xukai92/VecTargets.jl) and the master branch of `AdvancedHMC` manually". Its `[compat]` is `julia = "1.5.0, 1.6.0"`. It loads R's `coda` through RCall at `__init__` (`R"library(coda)"`) for spectral variance — [README](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/README.md), [Project.toml](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/Project.toml), [src/analysis.jl](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/src/analysis.jl)
- `coupled_rejection_sampling` (JAX) is a library of coupling primitives with no MCMC driver or estimator. It provides `coupled_sampler(key, Gamma_hat, p, q, log_p_hat, log_q_hat, log_p, log_q, log_M_p, log_M_q, N=1)` → `(X, Y, is_coupled, n_trials)`, the ensemble version, `modified_thorisson`, `reflection_maximal` and `coupled_mvns` for MVNs, coupled Gaussian tails, `reflection_coupled_mh` and a coupled parallel resampler — [coupled_rejection_sampler.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/coupled_rejection_sampling/coupled_rejection_sampler.py), [repo](https://github.com/AdrienCorenflos/coupled_rejection_sampling)
- `CoupledPDMPs.jl` provides `rhee_glynn(coupled_kernel::Function, coupled_state, k::Int, m::Int, return_classical_estimator=false)`, plus couplings `thorisson`, exponentials, `reflection_maximal`, `lindvall_roger` and Dau–Chopin — [src/coupled_estimator.jl](https://github.com/AdrienCorenflos/CoupledPDMPs/blob/main/src/coupled_estimator.jl), [src/coupled_refresh.jl](https://github.com/AdrienCorenflos/CoupledPDMPs/blob/main/src/coupled_refresh.jl). The paper is published as "Debiasing piecewise deterministic Markov process samplers using couplings", *Scandinavian Journal of Statistics* (2025) — [OpenAlex/DOI 10.1111/sjos.70015](https://doi.org/10.1111/sjos.70015)
- `partition-coupling` has categorical couplings `indep_coupling`, `pure_maximal_coupling`, `max_coupling`, `optimal_coupling` (POT) and `sinkhorn_coupling`, plus `c_maximal_{beta,Bernoulli,gamma,diag_normal}` and `MCMC`/`DPMMSingle` sampler classes. It runs from the CLI with `python -u ../modules/experiment.py --configPath ./gene/Gibbs/config.json --overwrite` — [modules/couplings.py](https://github.com/tinnguyen96/partition-coupling/blob/main/modules/couplings.py), [examples/example.sh](https://github.com/tinnguyen96/partition-coupling/blob/main/examples/example.sh)

**D. Independent Python/JAX re-implementations (none packaged, none maintained)**
- ColCarroll `couplings`: "A library for exploring coupled MCMC and to support essays on it" (essay at colcarroll.github.io/couplings). API: `metropolis_hastings(*, log_prob, proposal_cov, init_x, init_y, lag=1, iters=1000, chains=128, short_circuit=False) -> CoupledData` and `unbiased_estimator(data, func, burn_in) -> (mcmc_average, bias_correction)`. Test usage: `metropolis_hastings(log_prob=rv.logpdf, proposal_cov=10*np.eye(dim), init_x=init_x, init_y=init_y, lag=1, iters=20, chains=chains)`. The docstring lists "Future work: Tuning (could adjust proposal covariance… and adjust the number of iterations to some percentile of the meeting time)" — [metropolis_hastings.py](https://github.com/ColCarroll/couplings/blob/master/couplings/metropolis_hastings.py), [test](https://github.com/ColCarroll/couplings/blob/master/test/test_metropolis_hastings.py), [README](https://github.com/ColCarroll/couplings/blob/master/README.md)
- ylefay `pymcmc_unbiased` (JAX). The repository description says "Project done as part of the Bayesian Machine Learning course by Rémi Bardenet and Julyan Arbel", and the README says "It was developed as part of the Bayesian Machine Learning course taught by Rémi…". API: `unbiased_monte_carlo_estimation(key, h, x0, y0, q_hat, log_q, log_target, lag, k, m, max_iter=1e6, coupling=coupling)`, using `jax.lax.fori_loop`/`while_loop`, with `mh_single_kernel`, `mh_coupled_kernel` and `modified_thorisson_coupling` — [monte_carlo_estimators.py](https://github.com/ylefay/unbiased_mcmc_with_couplings/blob/main/src/pymcmc_unbiased/monte_carlo_estimators.py), [README](https://github.com/ylefay/unbiased_mcmc_with_couplings/blob/main/README.md)
- `HMC_unbiased`: "This package represents a quick implementation of the unbiased Hamiltonian Monte Carlo algorithm." API: `Q1, Q2, chain_length = unbiased_HMC_chain(Q1, Q2, potential, potential_grad, step_size, num_steps, gamma, std, distribution, key)`. Depends on `jax` and `jaxtyping`; installs from git — [README](https://github.com/Helmuthn/HMC_unbiased/blob/main/README.md), [docs/index.md](https://github.com/Helmuthn/HMC_unbiased/blob/main/docs/index.md)
- `mc-lab-edu` (PyPI 0.2.7, 2025-09-05) packages only `src/mc_lab`, which has no coupling module. The unbiased-MCMC code sits in `scripts/unbiased_mcmc.py`, `unbiased2.py` and `unbiased3.py` — [pyproject.toml](https://github.com/carsten-j/mc-lab/blob/main/pyproject.toml), [PyPI JSON](https://pypi.org/pypi/mc-lab-edu/json)
- `mcmc_infinity` (adjacent: coupling from the past, not Jacob et al.) describes itself as "Perfect MCMC sampling … for Bayesian inference problems in astronomy and astrophysics". API: `PerfectSampler(target, proposal, initial_positions, …)` with `.get_perfect_samples(T, num_samples)`. Requires Python ≥ 3.11, JAX and flowjax — [README](https://github.com/cjm96/mcmc_infinity/blob/main/README.md), [sampler.py](https://github.com/cjm96/mcmc_infinity/blob/main/mcmc_infinity/code/sampler.py), [pyproject.toml](https://github.com/cjm96/mcmc_infinity/blob/main/pyproject.toml)

**Registry scans (absence evidence)**
- A scan of all 920,853 PyPI project names for `coupl|unbiased|debias|maximal.?coup|llag|meeting.?time|rhee|glynn|telescop` found no unbiased-MCMC package. The hits are unrelated (`coupling` "Util library", `PyCoupling` "cable coupling and crosstalk tool", `unbiased` "bias detection… for LLM agents", `meetingtime` time zones, etc.) — [PyPI simple index](https://pypi.org/simple/), e.g. [unbiased JSON](https://pypi.org/pypi/unbiased/json)
- Direct PyPI lookups returned 404 for `unbiasedmcmc`, `unbiased-mcmc`, `coupled-mcmc`, `couplings`, `maximal-coupling`, `debiased-mcmc`, `pymcmc_unbiased`, `hmc-unbiased`, `coupled_rejection_sampling` and **`telescopic`**, so the name is free — [PyPI JSON API](https://pypi.org/pypi/telescopic/json)
- None of the CRAN package names or titles (25,498 packages) relate to unbiased MCMC or couplings of Markov chains. The matches are things like `coupling` "Analysis of Coupling Coordination Degree" and `Umoments` — [CRAN list](https://cran.r-project.org/web/packages/available_packages_by_name.html)
- The Julia General registry has no `CoupledHMC`, `MaximalCouplings`, `CoupledPDMPs`, `ImportanceMCMC`, `CoupledConditionalSMC` or `UnbiasedMCMC` entries. Its "Coupl*" packages (e.g. `CoupledFields`, `ClimaCoupler`) are unrelated — [Registry.toml](https://raw.githubusercontent.com/JuliaRegistries/General/master/Registry.toml)

**Adoption signals**
- Citations (Crossref `is-referenced-by-count`, 2026-10-23):
  - Jacob, O'Leary & Atchadé, JRSSB 2020: 97 — [Crossref](https://api.crossref.org/works/10.1111/rssb.12336)
  - Heng & Jacob, Biometrika 2019: 38 — [Crossref](https://api.crossref.org/works/10.1093/biomet/asy074)
  - Biswas et al., JRSSB 2022 (CoupledHalfT): 10 — [Crossref](https://api.crossref.org/works/10.1111/rssb.12495)
  - Kelly, Ryder & Clarté, AoAS 2023: 5 — [Crossref](https://api.crossref.org/works/10.1214/22-AOAS1676)
- Citations (OpenAlex, same date):
  - JRSSB 2020: 107 — [OpenAlex](https://api.openalex.org/works/doi:10.1111/rssb.12336)
  - Biometrika 2019: 46 — [OpenAlex](https://api.openalex.org/works/doi:10.1093/biomet/asy074)
  - Nguyen, Trippe & Broderick (arXiv record): 2 — [OpenAlex search](https://api.openalex.org/works?search=Many%20processors%2C%20little%20time)
- Citations (Semantic Scholar):
  - Biswas & Mackey (JASA): 9 — [S2](https://api.semanticscholar.org/graph/v1/paper/arXiv:2112.03152)
  - Douc et al. (Annals of Statistics): 16 — [S2](https://api.semanticscholar.org/graph/v1/paper/arXiv:2206.05691)
  - Jacob, Lindsten & Schön (JASA): 57 — [S2](https://api.semanticscholar.org/graph/v1/paper/arXiv:1701.02002)
- No coupling implementation has download statistics because none is on a package index. For scale, pypistats "last_month" downloads on 2026-10-23 were:
  - emcee 597,357 — [pypistats](https://pypistats.org/api/packages/emcee/recent)
  - dynesty 73,110 — [pypistats](https://pypistats.org/api/packages/dynesty/recent)
  - blackjax 128,178 — [pypistats](https://pypistats.org/api/packages/blackjax/recent)
  - numpyro 960,818 — [pypistats](https://pypistats.org/api/packages/numpyro/recent)
  - pymc 1,290,321 — [pypistats](https://pypistats.org/api/packages/pymc/recent)
  - zeus-mcmc 5,047 — [pypistats](https://pypistats.org/api/packages/zeus-mcmc/recent)
  - particles 985 — [pypistats](https://pypistats.org/api/packages/particles/recent)

### Inferences
- Maturity is research grade across the board.
  - The only repositories with any 2026 activity are `unbiasedmcmc` (2 commits), `unbiasedsnis` (paper scripts), `ImportanceMCMC.jl` and BlackJAX's new primitive.
  - Every Python/JAX port stopped within months of creation (ColCarroll: 1 month in 2019; HMC_unbiased: 5 months in 2023; ylefay: one course term, 2024).
- Quality signals are inverted relative to authority.
  - The authoritative R code (`unbiasedmcmc`, `debiasedhmc`, `LlagCouplings`) has no automated test suite.
  - The Python/JAX ports that do have pytest suites are unmaintained and unreleased, and some deviate from the paper (see Q3).
- ColCarroll's install line `pip install git+git://github.com/...` uses the unauthenticated `git://` protocol. GitHub disabled that protocol in 2022, so the documented install is probably broken. This was not tested here.
- Combining the low citation counts in DOI databases (≈100 for the JRSSB paper) with the complete absence of installable software suggests that uptake outside the methods community is limited by tooling as well as by method.

### Gaps
- Google Scholar counts, which are usually much higher than Crossref/OpenAlex, could not be retrieved programmatically. Semantic Scholar and OpenAlex rate-limited (HTTP 429) most queries, so citation counts are partial and come from mixed databases.
- Download counts for Julia/R GitHub-only code are unavailable. GitHub traffic and clone stats require owner access.
- No code was executed, so runtime correctness, speed and whether `CoupledHMC.jl` still installs against AdvancedHMC 0.8.x are unverified. The AdvancedHMC names it relies on (`sample_phasepoint`, `AbstractTrajectorySampler`, `AbstractMomentumRefreshment`, `refresh`) still appear in AdvancedHMC's `src/`, but compatibility was not tested.
- Code hosted outside GitHub (GitLab, personal pages, journal supplements) was only covered through web search. Papers with Code was not used.

## Q2. Do general-purpose MCMC libraries support couplings or unbiased estimators?

### Takeaway
None of BlackJAX, NumPyro, TFP, PyMC, Stan, emcee, zeus, particles, Pigeons.jl, Turing.jl/AdvancedHMC.jl/AdvancedMH.jl or the diagnostics libraries (ArviZ-stats, MCMCDiagnosticTools.jl) offers meeting times, unbiased estimators or coupling-based diagnostics. Only three low-level coupling building blocks exist:
- BlackJAX's experimental `blackjax.mcmc.coupled_hmc` (added September 2026). It is explicitly scoped to exclude meeting and unbiasedness.
- TFP FunMC's `maximal_reflection_coupling_proposal`, available since 2021 in a spinoff that is not on PyPI.
- AdvancedHMC's `rand_coupled`, which synchronises RNGs across chains (2020).

BlackJAX maintainers closed the 2021 feature request as "a niche algorithm with no owner".

### Cited Findings
- **BlackJAX, coupled HMC**
  - `blackjax/mcmc/coupled_hmc.py` (920 lines; 1,556-line test file) is "Two HMC marginals coupled only through their random inputs… This module does not claim product-target invariance, meeting, unbiased estimators, or an efficiency improvement".
  - It offers `synchronous` and `reflection` couplings of the momentum (z′ = z − 2e(eᵀz)) with a shared acceptance uniform.
  - It accepts only fixed Euclidean metrics; callable/Riemannian metrics are rejected.
  - "There is no top-level `blackjax.coupled_hmc`; use `blackjax.mcmc.coupled_hmc`."
  - Usage: `coupled_hmc.as_top_level_api((logdensity_fn, logdensity_fn), step_size=(0.1, 0.1), inverse_mass_matrix=(imm, imm), num_integration_steps=(10, 10), coupling="reflection")`, then `algorithm.init((position_one, position_two))` and `algorithm.step(rng_key, state)` — [coupled_hmc.py](https://github.com/blackjax-devs/blackjax/blob/main/blackjax/mcmc/coupled_hmc.py)
  - The file was added in commit eb9c6c81 on 2026-09-07 ("Add coupled HMC: two marginals sharing their random inputs (#1033)") and edited on 2026-09-08 and 2026-10-04. The first tags containing it are 1.7.0 and 1.7.1 (both 2026-10-02). The PyPI latest is 1.7.1, uploaded 2026-10-02 — [PR #1033](https://github.com/blackjax-devs/blackjax/pull/1033), [PyPI JSON](https://pypi.org/pypi/blackjax/json)
  - PR #1033 (author junpenglao) states the module does not provide "that the chains meet (exactly, maximally, or at all)" or "unbiasedness of any estimator built from the pair", that "No performance measurement of any kind was made", and that there is "deliberately no top-level `blackjax.coupled_hmc` while the API settles" — [PR #1033](https://github.com/blackjax-devs/blackjax/pull/1033)
- **BlackJAX, history of the unbiased-HMC request**
  - Issue #64, opened 2021-07-24 by Adrien Corenflos, asked to implement arXiv:1709.00404 and arXiv:2104.05134. He wrote: "I don't really think these methods are implemented in any Python library."
  - On 2026-03-26 junpenglao closed it: "Closing as stale. No progress in 3+ years. Unbiased HMC with couplings is a niche algorithm with no owner." — [issue #64](https://github.com/blackjax-devs/blackjax/issues/64)
  - PR #65 "Coupled HMC methods" (Corenflos, 2021) was closed unmerged on 2022-02-09. Reasons given:
    - failing coupled-RWMH tests;
    - Corenflos wrote "I think I got the initial design wrong", meaning the algorithm acts as a meta-algorithm that does not fit the kernel API;
    - a plan to split the work into separate PRs.
    - Pierre Jacob offered help with tests — [PR #65](https://github.com/blackjax-devs/blackjax/pull/65)
  - Issue #826 "Implement Unbiased HMC with couplings" (opened 2026-03-26 by junpenglao, still open, no comments or PRs) says:
    - the R package `unbiasedmcmc` is the only existing implementation, and no JAX or Python version was found;
    - effort is estimated as "L";
    - a variable meeting time needs `lax.while_loop`, and fixed output shapes need care;
    - its answer to "Willing to open a PR?" is "No" — [issue #826](https://github.com/blackjax-devs/blackjax/issues/826)
  - Issue #17 "Add coupled multinomial HMC" (opened 2021-04-14 by rlouf) was closed as completed on 2021-08-05. Its body only describes (uncoupled) multinomial HMC, and the page shows no linked PR — [issue #17](https://github.com/blackjax-devs/blackjax/issues/17)
- **TFP / FunMC**
  - The core `tfp.mcmc` and `tfp.experimental.mcmc` modules contain no coupling, meeting-time or unbiased-estimator code (grep of a 2026-10-21 clone). The only "coupl" matches there are unrelated wording such as "reducer method calls must be coupled with a state object" — [tfp/python/mcmc](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/mcmc), [experimental/mcmc/reducer.py](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/experimental/mcmc/reducer.py)
  - The FunMC spinoff provides `maximal_reflection_coupling_proposal(state, chain_ndims=0, scale=1.0, named_axis=None, epsilon=1e-20, seed=None)`. Its docstring: "Given two MCMC chains with the same, axis-aligned symmetrical Gaussian proposal, this function will produce a joint proposal with the maximum probability of the proposed state being the same for both chains"; "chain_i is coupled with chain_i + num_chains"; "supports SPMD via sharded states". It cites O'Leary (2021) — [fun_mc_lib.py](https://github.com/tensorflow/probability/blob/main/spinoffs/fun_mc/fun_mc/fun_mc_lib.py)
  - The function was present in the first commit that added FunMC to spinoffs (3916a841, 2021-04-02). FunMC's pyproject says `name = "fun_mc"`, `version = "0.1.0"`, but `fun_mc` and `fun-mc` return 404 on PyPI. TFP itself is at 0.25.0 (2024-11-08) on PyPI — [spinoffs/fun_mc](https://github.com/tensorflow/probability/tree/main/spinoffs/fun_mc), [TFP PyPI JSON](https://pypi.org/pypi/tensorflow-probability/json)
  - A GitHub issue search on tensorflow/probability for unbiased/coupling requests returned nothing relevant — [TFP issues](https://github.com/tensorflow/probability/issues)
- **AdvancedHMC.jl and Turing.jl**
  - `rand_coupled` "produces coupled randomness given a vector of RNGs… keeping the RNGs synchronised. This is important if we want to couple multiple Markov chains."
  - It was introduced 2020-06-02 by Kai Xu (91837c65, "Fix bugs in multinomial sampling (#199)").
  - `trajectory.jl` notes that "sharing the RNG over multiple chains for accepting / rejecting might couple the chains. We need to revisit this more rigorously".
  - There are no meeting-time or estimator functions. The latest tag is v0.8.8 (2026-10-17) — [src/utilities.jl](https://github.com/TuringLang/AdvancedHMC.jl/blob/main/src/utilities.jl), [src/trajectory.jl](https://github.com/TuringLang/AdvancedHMC.jl/blob/main/src/trajectory.jl)
  - The actual multinomial-HMC couplings live in the separate, unregistered `chalk-lab/CoupledHMC.jl` (Q1) — [CoupledHMC.jl](https://github.com/chalk-lab/CoupledHMC.jl)
  - Turing.jl 0.49.0 (clone 2026-10-15) mentions "unbiased" only for SMC normalising-constant estimates and has no couplings — [Turing.jl](https://github.com/TuringLang/Turing.jl)
- **No coupling-related code in the 2026-10 clones of:**
  - AdvancedMH.jl 0.8.10, MCMCDiagnosticTools.jl 0.3.19 and Pigeons.jl 0.4.11 (parallel tempering; AGPL-3.0) — [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl), [MCMCDiagnosticTools.jl](https://github.com/TuringLang/MCMCDiagnosticTools.jl), [Pigeons.jl](https://github.com/Julia-Tempering/Pigeons.jl)
  - NumPyro 0.22.0 (2026-09-18). Every "coupl" match is unrelated wording: copula docs ("Couple continuous univariate marginals"), "coupled angles" in directional distributions, reparameterisation docs and an ensemble-move comment — [NumPyro](https://github.com/pyro-ppl/numpyro), [PyPI JSON](https://pypi.org/pypi/numpyro/json)
  - PyMC 6.3.2 (2026-09-08) — [PyMC](https://github.com/pymc-devs/pymc), [PyPI JSON](https://pypi.org/pypi/pymc/json)
  - Stan (develop branch, 2026-10-22). Its issue tracker has no coupling requests — [stan-dev/stan](https://github.com/stan-dev/stan)
  - emcee 3.1.6 (2024-04-19). The only hit is the paper's `ms.tex`, and there are no relevant issues — [emcee](https://github.com/dfm/emcee), [PyPI JSON](https://pypi.org/pypi/emcee/json)
  - zeus 2.5.4 (2023-01-12; last commit 2024-02-18) — [zeus](https://github.com/minaskar/zeus), [PyPI JSON](https://pypi.org/pypi/zeus-mcmc/json)
  - particles 0.4 (2023-11-06). Its `mcmc.py` has `GenericRWHM`, `BasicRWHM`, `PMMH`, `CSMC`, `GenericGibbs` and `ParticleGibbs` only — [particles/mcmc.py](https://github.com/nchopin/particles/blob/master/particles/mcmc.py), [PyPI JSON](https://pypi.org/pypi/particles/json)
  - arviz-stats (2026-10-22) — [arviz-stats](https://github.com/arviz-devs/arviz-stats)
- **Popularity of the general libraries** (GitHub stars, 2026-10-23): PyMC 9,820; TFP 4,440; Stan 2,783; NumPyro 2,770; emcee 1,603; BlackJAX 1,136; particles 512; dynesty 426; AdvancedHMC.jl 365; zeus 244; Pigeons.jl 153 — [GitHub search](https://github.com/search?q=repo%3Ablackjax-devs%2Fblackjax+repo%3Adfm%2Femcee&type=repositories)

### Inferences
- BlackJAX's new module covers only the inner part of what Heng & Jacob need: a common-random-number or reflection momentum coupling. Without a maximally coupled RWMH component in a mixture, and without an outer meeting/estimator driver, chains in continuous spaces contract but do not meet exactly. Its own documentation says it makes no meeting claims. A user who wants unbiased estimates in JAX still has to write the meeting loop, lag handling, estimator, cost accounting and tuning themselves. This is the "meta-algorithm" layer that Corenflos said does not fit a kernel API (PR #65), and it is the natural niche for `telescopic`.
- The maintainers' "niche algorithm with no owner" framing (#64) and #826's "Willing to open a PR? No" suggest no general-purpose library is likely to ship estimators soon. A dedicated package would not duplicate active upstream work, though BlackJAX could become a JAX backend.
- FunMC's primitive shows the batched pattern of concatenating 2N chains so chain i couples with chain i+N, which is a good vectorisation template for a NumPy/JAX implementation. FunMC itself is not an installable dependency.

### Gaps
- I did not check every past BlackJAX release's docs to see whether `coupled_hmc` appears in the rendered API docs.
- Stan Discourse and the PyMC and NumPyro forums were checked only through one web search and GitHub issue search; forum threads requesting couplings may exist without having been found.
- TFP's git history was searched only for the FunMC function, not for removed experimental coupling code.

## Q3. How do existing implementations structure kernels, couplings, meeting detection and estimators?

### Takeaway
Almost every implementation copies Jacob's R design, in which the user supplies:
- `rinit()`;
- `single_kernel(state)`;
- `coupled_kernel(state1, state2)`, which returns the new states plus an `identical` flag.

A generic driver advances X by `lag` steps, couples until meeting, continues to max(τ, m), and computes the L-lag time-averaged estimator either online or from stored chains (`H_bar`, signed measures).

The implementations differ in three places:
- meeting detection: an exact flag, exact equality or a numerical tolerance;
- the lag weights: the general L-lag formula versus a lag-1 formula;
- vectorisation: R loops, NumPy batch arrays, JAX `lax` loops, or Julia types plugged into AdvancedHMC.

No implementation adapts kernel tuning parameters automatically for coupled runs; the only k/m heuristic found is in CoupledHMC.jl.

### Cited Findings
- **Kernel/coupling interface, the reference pattern**
  - `get_mh_kernels(target, Sigma_proposal)` returns closures `single_kernel(state)` and `coupled_kernel(state1, state2)`. A state is `list(chain_state, current_pdf)`, so the log-density is cached.
  - The coupled kernel draws reflection-maximal proposals (`rnorm_reflectionmax` / `rmvnorm_reflectionmax`), uses one shared `logu <- log(runif(1))` for both accept/reject decisions, and returns `identical = proposal_value$identical && accept1 && accept2` — [R/get_mh_kernels.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/get_mh_kernels.R)
  - The reflection-maximal coupling: draw ẋ ~ N(0,1); accept ẏ = ẋ + z when `log(u) < dnorm(ẋ+z) − dnorm(ẋ)`, otherwise reflect ẏ = ẋ − 2(eᵀẋ)e — [R/rnorm_reflectionmax.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/rnorm_reflectionmax.R)
  - The generic maximal coupling (`get_max_coupling(rp, dp, rq, dq)`) is the rejection-based algorithm, with a `while` loop of random length — [R/get_max_coupling.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/get_max_coupling.R)
- **Drivers**
  - `sample_meetingtime(single_kernel, coupled_kernel, rinit, lag = 1, max_iterations = Inf)` only returns τ and the elapsed time — [R/meetingtime.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/meetingtime.R)
  - `sample_coupled_chains(…, m = 1, lag = 1, max_iterations = Inf, preallocate = 10)` stores both chains (growing matrices) until `max(meetingtime, m)`. After meeting it runs only `single_kernel` and copies the state. It returns `cost = lag + 2*(meetingtime - lag) + max(0, time - meetingtime)` — [R/coupled_chains.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/coupled_chains.R)
  - `sample_unbiasedestimator(…, h, k = 0, m = 1, lag = 1, max_iterations = Inf)` computes the estimator online without storing chains, and returns `mcmcestimator`, `correction`, `uestimator`, `meetingtime`, `cost` and `elapsedtime` — [R/unbiasedestimator.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/unbiasedestimator.R)
- **Estimator formula (general L-lag)**
  - `H_bar(c_chains, h, k, m)` adds to the MCMC average Σ_{t=k+lag}^{τ−1} `(floor((t-k)/lag) - ceiling(max(lag, t-m)/lag) + 1) * (h(X_t) - h(Y_{t-lag}))`, divided by (m − k + 1) — [R/H_bar.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/H_bar.R)
  - The signed-measure representation, for arbitrary functions after the fact, is `c_chains_to_measure_as_list(c_chains, k, m)` in C++ — [R/c_chains_to_measure_as_list.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/c_chains_to_measure_as_list.R)
  - The ylefay JAX port uses the same weights: `v(t) = floor((t - k)/lag) - ceil(max(lag, t - m)/lag) + 1` — [monte_carlo_estimators.py](https://github.com/ylefay/unbiased_mcmc_with_couplings/blob/main/src/pymcmc_unbiased/monte_carlo_estimators.py)
- **Lag-1 weights only**
  - ColCarroll's `unbiased_estimator` uses `ratio = np.minimum(1, (split_idxs - burn_in) / normalizer)` and `mult = func(data.x[slicer]) - func(data.y[slicer - data.lag])`, with `normalizer = iters - burn_in + 1`. Effectively m is fixed at the run length.
  - For chains that never met (`meeting_time == -1`), `mult[split_idxs > data.meeting_time] = 0` sets the correction to zero, so a truncated chain silently contributes only its MCMC average — [metropolis_hastings.py](https://github.com/ColCarroll/couplings/blob/master/couplings/metropolis_hastings.py)
  - CoupledHMC.jl's `H_of` uses `min(1, (n - k)/(m - k + 1)) * (h(xs[n]) - h(ys[n]))` — [src/analysis.jl](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/src/analysis.jl)
- **Meeting detection, three styles**
  1. A flag set by the coupling, exact by construction:
     - unbiasedmcmc (`identical`) — [get_mh_kernels.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/get_mh_kernels.R)
     - ylefay (`coupled = (n_iter_coupling == 0) & accept_X & accept_Y`) — [metropolis_hasting.py](https://github.com/ylefay/unbiased_mcmc_with_couplings/blob/main/src/pymcmc_unbiased/metropolis_hasting.py)
     - CoupledPDMPs (`coupled_state.coupled`) — [coupled_estimator.jl](https://github.com/AdrienCorenflos/CoupledPDMPs/blob/main/src/coupled_estimator.jl)
     - coupled_rejection_sampling (returns `is_coupled`, with the warning "if the distributions are not continuous this may be False even if X=Y") — [coupled_rejection_sampler.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/coupled_rejection_sampling/coupled_rejection_sampler.py)
  2. Exact equality of states:
     - LlagCouplings `default_cmp <- function(a, b) all(a == b)`, user-overridable through `cmp` — [coupling.R](https://github.com/niloyb/LlagCouplings/blob/master/coupling.R)
     - CoupledHalfT `all(eta_1_current==eta_2_current) && (xi_1_current==xi_2_current) && (sigma2_1_current==sigma2_2_current)` — [R/coupled_mcmc_half_t.R](https://github.com/niloyb/CoupledHalfT/blob/master/R/coupled_mcmc_half_t.R)
     - HMC_unbiased loops `while (out1[i,:] != out2[i,:]).all() & (i < 10000)`, which stops when any coordinate coincides and silently caps at 10,000 iterations — [HMC_unbiased/main.py](https://github.com/Helmuthn/HMC_unbiased/blob/main/HMC_unbiased/main.py)
  3. Numerical tolerance:
     - ColCarroll `met = np.isclose(data.x[t], data.y[t - lag])` across all dimensions — [metropolis_hastings.py](https://github.com/ColCarroll/couplings/blob/master/couplings/metropolis_hastings.py)
     - CoupledHMC.jl `does_meet(x, y) = isapprox(x, y; atol=ATOL_MEETING)` with `ATOL_MEETING = 1e-10` — [src/analysis.jl](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/src/analysis.jl)
     - O'Leary's break condition `np.allclose(self.x_curr, self.y_curr)` — [src/mcmc.py](https://github.com/johnoleary/mh-max-couplings/blob/main/src/mcmc.py)
- **Vectorisation and parallelism**
  - unbiasedmcmc: one pair per call; independent replicates through user-side `foreach … %dorng%` — [vignette](https://github.com/pierrejacob/unbiasedmcmc/blob/master/vignettes/introduction-normaltarget.Rmd)
  - ColCarroll: arrays shaped `(iters, chains, dim)`; runs a fixed number of iterations for all chains with optional `short_circuit` once all have met; uses the global `np.random` state — [metropolis_hastings.py](https://github.com/ColCarroll/couplings/blob/master/couplings/metropolis_hastings.py)
  - ylefay: `@jax.vmap @jax.jit` over keys — [experiments/compute_pi_over_4.py](https://github.com/ylefay/unbiased_mcmc_with_couplings/blob/main/experiments/compute_pi_over_4.py)
  - CoupledHMC.jl: research scripts with `Threads.@threads` — [research/scripts/banana.jl](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/research/scripts/banana.jl)
  - FunMC: batch-concatenated pairs and SPMD sharding — [fun_mc_lib.py](https://github.com/tensorflow/probability/blob/main/spinoffs/fun_mc/fun_mc/fun_mc_lib.py)
  - BlackJAX: a `CoupledHMCState(first, second)` NamedTuple with per-marginal caches; every argument is an explicit `(first, second)` pair ("nothing is broadcast or inferred") — [coupled_hmc.py](https://github.com/blackjax-devs/blackjax/blob/main/blackjax/mcmc/coupled_hmc.py)
- **Tuning**
  - CoupledHMC.jl `get_k_m(τs, λ; method=:quantile)` sets k to the meeting time τ at which at least 90% of the sampled τs are smaller, or alternatively the median, and sets m = λk — [src/analysis.jl](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/src/analysis.jl)
  - unbiasedmcmc's README picks `lag = 500, k = 500, m = 2000` by hand after looking at a meeting-time histogram — [README](https://github.com/pierrejacob/unbiasedmcmc/blob/master/README.md)
  - ColCarroll lists tuning as future work — [metropolis_hastings.py](https://github.com/ColCarroll/couplings/blob/master/couplings/metropolis_hastings.py)
- **Diagnostics structure**
  - The L-lag TV bound is computed from meeting times alone (`pmax(0, ceiling((τ−L−t)/L))`, averaged over replicates); Wasserstein bounds additionally need thinned chain differences — [tv_wasserstein_bounds.R](https://github.com/niloyb/LlagCouplings/blob/master/tv_wasserstein_bounds.R)
  - `simulate_ipm_bounds` accumulates integral-probability-metric bounds along the coupled path — [coupling.R](https://github.com/niloyb/LlagCouplings/blob/master/coupling.R)
- **Model-specific couplings**
  - Gibbs-component maximal couplings for gamma, inverse-gamma and inverse-Gaussian — [unbiasedmcmc R/](https://github.com/pierrejacob/unbiasedmcmc/tree/master/R)
  - Truncated-polynomial-exponential and two-scale CRN/maximal switching — [CoupledHalfT R/](https://github.com/niloyb/CoupledHalfT/tree/master/R)
  - OT/Sinkhorn couplings of categorical full conditionals — [partition-coupling couplings.py](https://github.com/tinnguyen96/partition-coupling/blob/main/modules/couplings.py)

### Inferences
- The `single_kernel` / `coupled_kernel` / `rinit` triple is the de facto standard interface. A new package should keep it as its extension point so that experts can plug in custom couplings. Most users, however, need a layer above it that builds both kernels from a log-density alone, as `get_mh_kernels(target, Sigma)` already does in R.
- Two styles carry correctness risk for unbiasedness:
  - Tolerance-based meeting (`np.isclose`, `isapprox(atol=1e-10)`, `np.allclose`) can declare meeting for chains that are close but not identical. After that the code copies one chain into the other, so the faithfulness property becomes approximate.
  - Silent truncation at a maximum number of iterations (ColCarroll's zeroed correction, HMC_unbiased's 10,000 cap) produces a biased estimate with no warning.

  A robust design meets through flags returned by the coupling (exact by construction), and reports truncation explicitly with a flag and a count.
- The general L-lag weights in `H_bar` are the correct template. The lag-1 formula in ColCarroll and CoupledHMC.jl does not generalise.
- `sample_unbiasedestimator` (online, O(1) memory, fixed h) and `sample_coupled_chains` + signed measure (stores chains, any h afterwards) cover two use cases. Both are worth offering.
- The cost-accounting convention (`lag + 2(τ−lag) + max(0, time−τ)` kernel calls) lets users compare inefficiency (cost × variance) with standard MCMC, and should be carried over.

### Gaps
- I did not read every model-specific coupling (e.g. Ising parallel-tempering couplings, PDMP refresh couplings) in detail.
- I did not verify by running whether the deviations noted (lag-1 weights with lag > 1 in ColCarroll; HMC_unbiased's coordinate-wise stop) change results in practice.

## Q4. What gaps do users face today, and what does that imply for `telescopic`'s scope?

### Takeaway
A scientist with a black-box log-density today has no installable, documented, tested, maintained tool in Python (or any language) that:
- couples chains automatically;
- detects meeting reliably;
- chooses k, m and L automatically;
- runs replicates in parallel;
- returns unbiased estimates with confidence intervals and coupling-based convergence diagnostics.

Experts still copy R research scripts. This defines `telescopic`'s scope: an emcee/dynesty-style front end on top of a coupling/estimator core, with automatic tuning and diagnostics as first-class features.

### Cited Findings
- The reference code self-describes as non-general software: "This is not a general-purpose statistical software, but a collection of scripts… Use at your own risk!!" — [unbiasedmcmc README](https://github.com/pierrejacob/unbiasedmcmc/blob/master/README.md)
- Other repositories say similar things:
  - "very very drafty" — [couplingsmontecarlo README](https://github.com/pierrejacob/couplingsmontecarlo/blob/master/README.md)
  - "A proper readme is coming." — [coupled_rejection_sampling README](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/README.md)
  - "a quick implementation" — [HMC_unbiased README](https://github.com/Helmuthn/HMC_unbiased/blob/main/README.md)
- The BlackJAX maintainer's 2026 issue lists R `unbiasedmcmc` as the only existing implementation and says no JAX or Python version was found (paraphrased from the issue page) — [issue #826](https://github.com/blackjax-devs/blackjax/issues/826)
- Corenflos made the same observation in 2021: "I don't really think these methods are implemented in any Python library." — [issue #64](https://github.com/blackjax-devs/blackjax/issues/64)
- Nothing is installable from a package index (PyPI, CRAN and Julia registry scans in Q1). The usual install routes are:
  - `devtools::install_github` — [unbiasedmcmc README](https://github.com/pierrejacob/unbiasedmcmc/blob/master/README.md)
  - `pip install git+…` — [ColCarroll README](https://github.com/ColCarroll/couplings/blob/master/README.md)
  - `pip install -e .` — [ylefay README](https://github.com/ylefay/unbiased_mcmc_with_couplings/blob/main/README.md)
  - unregistered Julia packages that need AdvancedHMC master — [CoupledHMC.jl README](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/README.md)
- Interfaces require users to write kernels. Even the R convenience constructor `get_mh_kernels(target, Sigma_proposal)` needs a hand-chosen proposal covariance and an `rinit()` that returns `list(chain_state, current_pdf)` — [R/get_mh_kernels.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/get_mh_kernels.R)
- ylefay's API needs `q_hat`, `log_q`, `log_target`, both initial points, `lag`, `k`, `m` and a JAX key — [monte_carlo_estimators.py](https://github.com/ylefay/unbiased_mcmc_with_couplings/blob/main/src/pymcmc_unbiased/monte_carlo_estimators.py)
- HMC_unbiased needs `potential`, `potential_grad`, step size, number of steps, mixture probability `gamma` and the RWMH `std` — [main.py](https://github.com/Helmuthn/HMC_unbiased/blob/main/HMC_unbiased/main.py)
- No automatic tuning:
  - the only k/m heuristic is `get_k_m` in CoupledHMC.jl (90% quantile of τ, m = λk) — [analysis.jl](https://github.com/chalk-lab/CoupledHMC.jl/blob/main/src/analysis.jl)
  - ColCarroll lists tuning as future work — [metropolis_hastings.py](https://github.com/ColCarroll/couplings/blob/master/couplings/metropolis_hastings.py)
  - the R README tunes by eye — [README](https://github.com/pierrejacob/unbiasedmcmc/blob/master/README.md)
- Diagnostics exist only as research scripts and an unregistered Julia package:
  - LlagCouplings TV/Wasserstein bounds — [tv_wasserstein_bounds.R](https://github.com/niloyb/LlagCouplings/blob/master/tv_wasserstein_bounds.R)
  - ImportanceMCMC f-divergence diagnostics — [README](https://github.com/AdrienCorenflos/ImportanceMCMC/blob/main/README.md)

  Mainstream diagnostics libraries have none (ArviZ-stats, MCMCDiagnosticTools.jl; Q2).
- Astronomy-facing tooling is emcee/dynesty-style. emcee had 597,357 and dynesty 73,110 monthly downloads — [pypistats emcee](https://pypistats.org/api/packages/emcee/recent), [pypistats dynesty](https://pypistats.org/api/packages/dynesty/recent). The only astronomy-targeted coupling package is a coupling-from-the-past prototype (v0.0.1, one week of commits, not on PyPI) — [mcmc_infinity](https://github.com/cjm96/mcmc_infinity)
- JAX-specific implementation hurdles noted by the BlackJAX maintainer (paraphrased): a variable meeting time needs `lax.while_loop` with a dynamic stop condition, and producing a fixed output shape from a loop of variable length needs care — [issue #826](https://github.com/blackjax-devs/blackjax/issues/826)
- In PR #65, Corenflos concluded that the method behaves as a meta-algorithm that does not fit a step-kernel API (paraphrased) — [PR #65](https://github.com/blackjax-devs/blackjax/pull/65)
- No implementation or issue was found for couplings of affine-invariant ensemble moves (emcee stretch/walk, zeus slice). None of the repositories above implements them, and the emcee issue search found nothing — [emcee issues](https://github.com/dfm/emcee/issues)
- The name `telescopic` is unclaimed on PyPI (404) — [PyPI](https://pypi.org/pypi/telescopic/json)

### Inferences
- **Core scope (what nothing else provides)**
  1. A black-box front end: `log_prob(x)` plus an initial distribution, which builds RWMH single/coupled kernels with a reflection-maximal coupling. This is the one coupling that needs neither gradients nor model structure.
  2. Exact meeting through coupling flags, never by tolerance.
  3. The general L-lag time-averaged estimator H_{k:m}, available online (fixed h) and as stored chains or a signed measure (any h afterwards).
  4. Automatic choice of k and m from a pilot sample of meeting times, following the Jacob et al. guidance that CoupledHMC.jl's `get_k_m` encodes: k ≈ a high quantile of τ, m ≈ a multiple of k. Report the inefficiency (cost × variance) next to an i.i.d. CI.
  5. Proposal-scale adaptation, confined to a pilot phase and frozen before the coupled runs so that unbiasedness is preserved.
  6. Parallel replicates through a user-supplied `pool`/`map`, as in emcee/dynesty, with independent RNG streams (NumPy `SeedSequence.spawn`).
  7. Explicit handling of `max_iterations` truncation: flagged and counted, never silent.
  8. L-lag TV and W1 upper-bound diagnostics plus meeting-time summaries as first-class outputs.
- **Second tier**
  - gradient-based kernels (MALA/HMC mixtures with a maximally coupled RWMH component, following Heng & Jacob and Xu et al.);
  - Gibbs-component coupling helpers (gamma, inverse-gamma, normal maximal couplings, as in `unbiasedmcmc`);
  - a generic `get_max_coupling(rp, dp, rq, dq)`.
- **Later**
  - a JAX backend, possibly interoperating with BlackJAX `coupled_hmc` for the inner kernel;
  - couplings for ensemble moves, which would be novel since none exists.
- **Validation**: the R `unbiasedmcmc` package and `LlagCouplings` are the natural reference oracles for numerical cross-checks, such as meeting-time distributions on the bimodal mixture example, by running them, not by copying code (see Q5).

### Gaps
- I found no user surveys or forum threads quantifying demand among astronomers. Demand is inferred from the absence of tools and from BlackJAX's issue history.
- I did not find published guidance for automatic proposal-scale tuning specific to coupled RWMH beyond the papers' own heuristics. The literature researchers should confirm the recommended k/m rules (k a large quantile of the meeting time, m a multiple of k, as discussed in Jacob et al. 2020), because this note only cites code. The exact multipliers were not checked here.

## Q5. Licences of existing code and implications for an MIT-licensed `telescopic`

### Takeaway
The most complete general-purpose code is copyleft: `unbiasedmcmc` and `debiasedhmc` are GPL ≥ 2, and O'Leary's MH-coupling code and Papp's `rwmcouplings` are GPL-3. Translating any of it into Python would create a derivative work that must stay GPL. Several useful sources are MIT or Apache-2.0 and can be adapted with attribution, notably the L-lag diagnostics code, partition couplings, CoupledPDMPs, ImportanceMCMC, ColCarroll's and ylefay's ports, BlackJAX and FunMC. Some important repositories have no licence at all (all rights reserved): CoupledHMC.jl, coupled_rejection_sampling, CoupledHalfT, umcmchandbook, couplings_bgs and TraitLab. The safe route is clean-room implementation from the papers, using GPL and unlicensed code only as running reference oracles.

### Cited Findings
- **GPL (≥ 2)**:
  - unbiasedmcmc — [DESCRIPTION](https://github.com/pierrejacob/unbiasedmcmc/blob/master/DESCRIPTION)
  - debiasedhmc — [DESCRIPTION](https://github.com/pierrejacob/debiasedhmc/blob/master/DESCRIPTION)
  - jeremyhengjm/debiasedhmc — [repo](https://github.com/jeremyhengjm/debiasedhmc)
  - CoupledCPF — [repo](https://github.com/pierrejacob/CoupledCPF)
  - unbiasedpathsampling — [repo](https://github.com/pierrejacob/unbiasedpathsampling)
  - UnbiasedMultilevel — [repo](https://github.com/jeremyhengjm/UnbiasedMultilevel)
- **GPL-3**:
  - mh-max-couplings (LICENSE: "GNU GENERAL PUBLIC LICENSE Version 3") — [LICENSE](https://github.com/johnoleary/mh-max-couplings/blob/main/LICENSE)
  - rwmcouplings (`License: GPL-3`) — [DESCRIPTION](https://github.com/tamaspapp/rwmcouplings/blob/main/DESCRIPTION)
  - couplingsmontecarlo — [LICENSE](https://github.com/pierrejacob/couplingsmontecarlo/blob/master/LICENSE)
  - darrenjw/unbiased-mcmc — [LICENSE](https://github.com/darrenjw/unbiased-mcmc/blob/master/LICENSE)
  - unbiased_uld — [LICENSE](https://github.com/ruzayqat/unbiased_uld/blob/main/LICENSE)
- **MIT**:
  - LlagCouplings ("Copyright (c) 2019 Niloy Biswas") — [LICENSE](https://github.com/niloyb/LlagCouplings/blob/master/LICENSE)
  - BoundWasserstein — [LICENSE](https://github.com/niloyb/BoundWasserstein/blob/main/LICENSE)
  - unbiasedpoisson — [LICENSE](https://github.com/pierrejacob/unbiasedpoisson/blob/master/LICENSE)
  - unbiasedsnis — [LICENSE](https://github.com/pierrejacob/unbiasedsnis/blob/main/LICENSE)
  - ColCarroll couplings — [LICENSE](https://github.com/ColCarroll/couplings/blob/master/LICENSE)
  - ylefay — [LICENSE](https://github.com/ylefay/unbiased_mcmc_with_couplings/blob/main/LICENSE)
  - partition-coupling — [LICENSE](https://github.com/tinnguyen96/partition-coupling/blob/main/LICENSE)
  - CoupledPDMPs — [LICENSE](https://github.com/AdrienCorenflos/CoupledPDMPs/blob/main/LICENSE)
  - ImportanceMCMC — [LICENSE](https://github.com/AdrienCorenflos/ImportanceMCMC/blob/main/LICENSE)
  - MaximalCouplings.jl ("MIT 'Expat' License") — [LICENSE.md](https://github.com/maximerischard/MaximalCouplings.jl/blob/master/LICENSE.md)
  - CoupledConditionalSMC.jl — [LICENSE.md](https://github.com/awllee/CoupledConditionalSMC.jl/blob/master/LICENSE.md)
  - HMC_unbiased (MIT text, "Copyright 2023 Helmuth Naumer") — [LICENSE](https://github.com/Helmuthn/HMC_unbiased/blob/main/LICENSE)
  - mcmc_infinity — [LICENSE](https://github.com/cjm96/mcmc_infinity/blob/main/LICENSE)
  - mc-lab — [LICENSE](https://github.com/carsten-j/mc-lab/blob/main/LICENSE)
- **Apache-2.0**:
  - BlackJAX (file header in coupled_hmc.py) — [coupled_hmc.py](https://github.com/blackjax-devs/blackjax/blob/main/blackjax/mcmc/coupled_hmc.py)
  - TFP/FunMC — [TFP LICENSE](https://github.com/tensorflow/probability/blob/main/LICENSE)
  - adriens63 course project — [repo](https://github.com/adriens63/mcmc_unbiased_gibbs_sampler_py)
- **No licence or placeholder**:
  - CoupledHMC.jl (no LICENSE file; no licence text found anywhere in the repository) — [repo](https://github.com/chalk-lab/CoupledHMC.jl)
  - coupled_rejection_sampling (`LICENSE` is a 0-byte file) — [LICENSE](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/LICENSE)
  - CoupledHalfT and umcmchandbook (`License: What license is it under?`) — [CoupledHalfT DESCRIPTION](https://github.com/niloyb/CoupledHalfT/blob/master/DESCRIPTION), [umcmchandbook DESCRIPTION](https://github.com/pierrejacob/umcmchandbook/blob/main/DESCRIPTION)
  - couplings_bgs — [repo](https://github.com/paoloceriani/couplings_bgs)
  - TraitLab — [repo](https://github.com/traitlab-mcmc/TraitLab)
  - CoupledPhylogenetics — [repo](https://github.com/lukejkelly/CoupledPhylogenetics)
  - rachtsingh repositories — [repo](https://github.com/rachtsingh/coupled_kernels)
  - arijitdey22 — [repo](https://github.com/arijitdey22/Maximal_Coupling_of_MCMC)
  - ruslenghi — [repo](https://github.com/ruslenghi/Coupled-Markov)
- **Licences of the general libraries** a design might borrow API ideas or code from:
  - emcee: MIT ("Copyright (c) 2010-2021 Daniel Foreman-Mackey & contributors") — [LICENSE](https://github.com/dfm/emcee/blob/main/LICENSE)
  - zeus: GPL-3 — [LICENSE](https://github.com/minaskar/zeus/blob/main/LICENSE)
  - Pigeons.jl: AGPL-3.0 — [LICENSE](https://github.com/Julia-Tempering/Pigeons.jl/blob/main/LICENSE)
  - particles: MIT — [LICENSE](https://github.com/nchopin/particles/blob/master/LICENSE)
  - AdvancedHMC.jl: MIT — [LICENSE](https://github.com/TuringLang/AdvancedHMC.jl/blob/main/LICENSE)
  - NumPyro: Apache-2.0 — [LICENSE.md](https://github.com/pyro-ppl/numpyro/blob/master/LICENSE.md)
  - PyMC: "Apache License, Version 2.0" per PyPI metadata — [PyPI JSON](https://pypi.org/pypi/pymc/json)
  - Stan: BSD-3-Clause — [LICENSE.md](https://github.com/stan-dev/stan/blob/develop/LICENSE.md)
- Jacob's recent repositories are MIT (unbiasedpoisson, 2022–2025; unbiasedsnis, 2025–2026), while his 2017–2020 ones are GPL — [unbiasedpoisson](https://github.com/pierrejacob/unbiasedpoisson), [unbiasedsnis](https://github.com/pierrejacob/unbiasedsnis), [unbiasedmcmc](https://github.com/pierrejacob/unbiasedmcmc)

### Inferences
These points are general open-source licensing practice, not legal advice.
- Re-expressing GPL R code (e.g. `H_bar`, `get_mh_kernels`, `rmvnorm_reflectionmax`) line by line in Python would likely make `telescopic` (or that module) a derivative work under the GPL. Algorithms, equations and pseudocode from the papers are not protected by code copyright, so implementing from the JRSSB/Biometrika/NeurIPS papers is safe. The reflection-maximal coupling, rejection maximal coupling, L-lag estimator weights, cost formula and TV bounds are all stated in the papers. Running the GPL packages to generate reference numbers for tests does not affect `telescopic`'s licence, provided their code is not vendored.
- MIT sources can be adapted with the copyright notice kept, for example in a NOTICE/third-party file:
  - LlagCouplings for the TV/W bound estimators;
  - partition-coupling for categorical/OT couplings;
  - CoupledPDMPs/ImportanceMCMC for Thorisson and Lindvall–Roger couplings and f-divergence diagnostics;
  - ylefay/ColCarroll for vectorised NumPy/JAX layouts.
- Apache-2.0 sources (BlackJAX `coupled_hmc`, FunMC's batched reflection coupling) can be adapted into an MIT project, provided the Apache licence text, notices and change statements are kept for the adapted portions.
- Unlicensed repositories (CoupledHMC.jl, coupled_rejection_sampling, CoupledHalfT, umcmchandbook) should be treated as all-rights-reserved. Reading them for understanding and reimplementing from the papers is fine; copying is not, unless the authors grant a licence.
- Because Jacob's newer repositories are MIT, asking Pierre Jacob (and co-authors O'Leary and Atchadé) for an MIT relicensing of specific `unbiasedmcmc` helpers looks plausible, and Corenflos (MIT for CoupledPDMPs/ImportanceMCMC) could likewise be asked to add a licence to `coupled_rejection_sampling`. This is speculative and depends on the authors.

### Gaps
- Licence status was taken from LICENSE/DESCRIPTION files at HEAD. I did not check per-file headers, vendored third-party code (e.g. the Polya-Gamma sampler that `unbiasedmcmc` takes from BayesLogit, or the `tmg` truncated-normal code in `debiasedhmc`) or historical licence changes.
- Whether the TraitLab website or manual states a licence was not checked; only the repository was searched.

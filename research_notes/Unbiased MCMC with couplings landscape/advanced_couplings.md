# Couplings beyond basic Metropolis-Hastings: gradient-based, Gibbs, slice, ensemble, tempering, particle, PDMP and discrete samplers

Scope: how each sampler family has been coupled for unbiased MCMC (Jacob, O'Leary & Atchadé 2020) or for coupling diagnostics, as of 2026-10-23. Status labels used throughout:
- **[Published]**: peer-reviewed journal or conference paper.
- **[Preprint]**: arXiv only, or publication status not confirmed.
- **[Code]**: a repository I inspected or found.
- **[Unpublished: this note]**: my own analysis or my own small simulation. These are feasibility evidence, not literature, and must be labelled as experiments in `telescopic`.

Simulation scripts for the unpublished experiments are NumPy-only and are archived in `scripts/advanced/` (`ensemble_coupling_sim.py`, `coupling_sim2.py`, `rwmh_baseline.py`). Each experiment's setup is described in full below so it can be reproduced without them.

## 1. Hamiltonian Monte Carlo (Metropolis HMC, multinomial HMC) and NUTS

### Takeaway
There is one published, implementable recipe for exact meeting with HMC (Heng & Jacob 2019):
- a common-random-numbers (CRN) coupling, meaning shared initial momentum and a shared accept/reject uniform, which makes the chains contract;
- mixed with a small-probability coupled random-walk MH step (γ = 1/20, σ = 1e-3) whose proposals come from a maximal coupling, which turns "very close" into "exactly equal".

Multinomial HMC (Xu et al. 2021) keeps this mixture and also couples the trajectory-index selection, using a maximal or W2 optimal-transport coupling. A "contractive" momentum coupling (Bou-Rabee, Eberle & Zimmer 2020) cuts meeting times on non-convex targets. No coupled NUTS for unbiased estimation had been published as of October 2026. The closest work is a coupling-based mixing-time analysis of NUTS (preprint). HMC settings that are optimal for ordinary HMC can fail to contract when coupled, so (ε, L) must be chosen from pilot meeting-time runs and then frozen.

### Cited Findings
**Heng & Jacob (2019), the core construction [Published]**
- "Unbiased Hamiltonian Monte Carlo with couplings", *Biometrika* 106(2):287–302 (2019); arXiv 1709.00404, first posted September 2017 — [IDEAS/RePEc](https://ideas.repec.org/a/oup/biomet/v106y2019i2p287-302..html), [arXiv](https://arxiv.org/abs/1709.00404)
- How the chains are coupled:
  - Both chains draw the **same initial momentum** P* ~ N(0, I_d) and use the **same uniform** for the MH accept/reject step.
  - Each chain runs L leapfrog steps of size ε from its own position.
  - For a Gaussian target with shared momentum, |q¹(t) − q²(t)| = |cos(t/σ)|·|q¹₀ − q²₀|. The chains contract, but the distance is zero only at special trajectory lengths, so exact meeting does not happen.
  - Source: [ar5iv 1709.00404](https://ar5iv.labs.arxiv.org/html/1709.00404)
- How exact meeting is obtained:
  - The coupled kernel is the mixture K̄ = (1 − γ)·K̄_{ε,L} + γ·K̄_σ.
  - K̄_σ is a random-walk MH step whose proposals come from the **maximal coupling of N(x, σ²I) and N(y, σ²I)**, with a common acceptance uniform.
  - The proposals coincide with probability ≈ 1 − (2π)^(−1/2)·δ/σ, where δ = |x − y|.
  - Defaults: **σ = 1e-3, γ = 1/20**. A small γ keeps the loss of marginal efficiency small.
  - Source: [ar5iv 1709.00404](https://ar5iv.labs.arxiv.org/html/1709.00404)
- Theory (existence results only, with no explicit step-size formula):
  - Assumptions: U is C², ∇U is globally β-Lipschitz, and U is α-strongly convex on a compact set of positive measure.
  - Theorem 1: the *relaxed* meeting time (chains within δ) has geometric tails for small enough εL.
  - Theorem 2: the mixture's exact meeting time has geometric tails for small enough ε, L and σ.
  - Source: [ar5iv 1709.00404](https://ar5iv.labs.arxiv.org/html/1709.00404)
- Experiments and reported efficiency. Relative inefficiency here means (variance × expected cost of the unbiased estimator) divided by the asymptotic variance of optimally tuned HMC.
  - **Logistic regression** (German credit with 300 covariates, d = 302):
    - Grid searched: ε ∈ {0.01, 0.0125, …, 0.04}, L ∈ {10, 20, 30}. The chosen setting was (ε, L) = (0.0125, 10).
    - The **HMC-optimal (0.03, 10) did not contract** under coupling.
    - With k at the 90% quantile of meeting times and m = 10k, relative inefficiency was **1.05**. With k at the median and m = 10k, it was 1.18.
    - Caution: the extracted table is partly garbled, and Xu et al. (2021) say the Metropolis-HMC numbers here are off by about 2× because of an error.
  - **Log-Gaussian Cox process** (d ∈ {256, 1024, 4096}):
    - Relative inefficiency for HMC: **11.00, 5.43, 2.73**.
    - For Riemannian-manifold HMC (RMHMC) with a fixed metric: 11.68, 7.85, 3.72.
    - On the finest grid, an average estimator cost about 90 min (HMC) and 20 min (RMHMC), roughly 4,439 and 714 iterations.
  - **Truncated Gaussian** (k = 6, m = 60): inefficiency 7.15, against an HMC asymptotic variance of about 6.41.
  - **Gaussian dimension study**: ε = C·d^(−1/4) with εL ≈ 1, compared with coupled MALA and RWMH using 1,000 coupled chains. Results are shown in a figure only.
  - Source: [ar5iv 1709.00404](https://ar5iv.labs.arxiv.org/html/1709.00404)
- Alternative momentum coupling (Bou-Rabee et al., parameter κ): on the 2-d Rosenbrock target, mean meeting time fell from **158 (κ = 0, shared momentum) to 52 (κ = 1)**, with (ε, L, σ, γ) = (1/500, 500, 1e-3, 1/20) — [ar5iv 1709.00404](https://ar5iv.labs.arxiv.org/html/1709.00404)
- Tuning was done with pilot runs throughout:
  - a grid search over (ε, L), using 100 meeting times per configuration that contracted below 1e-10;
  - k taken from quantiles of pilot meeting times;
  - no adaptive tuning;
  - meeting times stable over a range of small σ and over the γ values tried.
  - Source: [ar5iv 1709.00404](https://ar5iv.labs.arxiv.org/html/1709.00404)
- **[Code]** The paper's R package is `pierrejacob/debiasedhmc` (C++/R, "Unbiased HMC estimators with couplings", last updated 2025-03-11). An early R copy exists at `jeremyhengjm/debiasedhmc` (2017). GitHub search found **no repository named `jeremyhengjm/UnbiasedHMC`**. — [paper](https://ar5iv.labs.arxiv.org/html/1709.00404), [pierrejacob/debiasedhmc](https://github.com/pierrejacob/debiasedhmc), [jeremyhengjm/debiasedhmc](https://github.com/jeremyhengjm/debiasedhmc)

**Contraction theory for HMC couplings [Published]**
- Bou-Rabee, Eberle & Zimmer, "Coupling and convergence for Hamiltonian Monte Carlo", *Ann. Appl. Probab.* 30(3):1209–1250 (2020) — [arXiv 1805.00452](https://arxiv.org/abs/1805.00452)
  - The coupling:
    - Velocities ξ and η are coupled so that the event ξ − η = −γ(x − y), with γ ≤ T⁻¹, happens with maximal probability; otherwise a **reflection coupling** of the velocities is used.
    - It is designed so that, for free dynamics, the positions approach each other with maximal probability.
  - The result:
    - One HMC step is contractive in a specially designed L¹-Wasserstein distance.
    - Assumptions: ∇U is L-Lipschitz, and U is strongly convex (constant K) outside a Euclidean ball of diameter 𝓡.
    - Step condition: LT² ≤ min(K/L, 1/4, 1/(256·L·𝓡²)).
    - The rate c = (1/10)·min(1, ½·K·T²·(1 + 𝓡/T)·e^(−𝓡/(2T)))·e^(−2𝓡/T) has no explicit dimension dependence; dimension enters through K and 𝓡.
    - The paper also covers velocity-Verlet HMC, both adjusted and unadjusted, provided the step size h is small enough, depending on the dimension.
  - Contrast with earlier work: Mangoubi & Smith use synchronous couplings (η = ξ), which "are contractive only if the stationary distribution is strongly log-concave".
- The 2024 review notes that for HMC to meet exactly *without* an MH mixture, one needs pairs of momenta for which the two Hamiltonian trajectories end at the same position (Bou-Rabee & Eberle 2023) — [Atchadé & Jacob 2024](https://arxiv.org/html/2406.06851)
- The same review warns that the mixture weights cannot depend on the distance between the chains, because that would break the marginal constraint — [Atchadé & Jacob 2024](https://arxiv.org/html/2406.06851)

**Multinomial HMC: Xu, Fjelde, Sutton & Ge (2021) [Published, AISTATS, PMLR 130:3646–3654]** — [PMLR](https://proceedings.mlr.press/v130/xu21i.html)
- The coupling:
  - Shared initial momentum P₀.
  - A shared number of forward steps L_f ~ Uniform{0,…,L}, with L_b = L − L_f, so the two trajectories are "aligned in time".
  - The index pair (i, j) is drawn from a coupling of the two categorical distributions over trajectory points. Two options:
    - the **W2 optimal-transport coupling** (squared Euclidean cost on positions, solved by linear programming; an approximate solver is debiased in Appendix B.3);
    - the **maximal coupling** (0-1 cost).
  - Source: [ar5iv 2104.05134](https://ar5iv.labs.arxiv.org/html/2104.05134)
- Exact meeting still uses the RWMH mixture (α = 1/20, σ = 1e-3, maximal coupling of proposals). The meeting time has geometric tails under local contractivity, which both the maximal and the W2 couplings satisfy (Lemmas 4.3–4.4) — [ar5iv 2104.05134](https://ar5iv.labs.arxiv.org/html/2104.05134)
- Experiments:
  - Targets: 1,000-d Gaussian; German credit logistic regression (ℝ³⁰²); Cox process on a 16×16 grid (ℝ²⁵⁶).
  - "Coupled multinomial HMC attains smaller meeting time and is more robust to ε."
  - Metropolis HMC shows a meeting-time surge near ε ≈ 0.3 on the Gaussian, which the authors attribute to near-closed orbits.
  - With NUTS-adapted parameters:
    - logistic regression (ε = 0.022, L = 22): meeting times **114 (maximal) and 118 (W2)**;
    - Cox process (0.28, 16): **50 and 51**.
  - Relative inefficiency with k at the 90% quantile and m = 10k (Metropolis / maximal / W2):
    - logistic regression: **2.32 / 1.90 / 0.94**;
    - Cox process: **4.61 / 3.83 / 5.80**.
  - Coupling NUTS is listed as future work.
  - Source: [ar5iv 2104.05134](https://ar5iv.labs.arxiv.org/html/2104.05134)
- **[Code]** `TuringLang/CoupledHMC.jl` (Julia, built on AdvancedHMC.jl; I cloned it and read the source):
  - Index couplings implemented: `IndependentCoupling`; `QuantileCoupling` (CRN via identically seeded RNGs); `MaximalCoupling`; `OTCoupling` (exact LP via JuMP/Clp); `ApproximateOTCoupling` (Sinkhorn, debiased by a mixture step).
  - `ContractiveRefreshment` implements the κ momentum coupling: y-momentum = x-momentum + κΔ with probability min(1, φ(ē·r + κ|Δ|)/φ(ē·r)), otherwise reflection, with κ = 1.
  - `MaxCoupledMH` draws the maximal coupling of the RWMH proposals by rejection sampling, so its cost is random.
  - The coupled iterator carries the comment "**FIXME: Adaptation is not supported.**" NUTS (`GeneralisedNoUTurn` with a Stan-style adaptor) is used **only in uncoupled runs** to choose ε and L.
  - Last commit 2022-02-28. It needs the AdvancedHMC master branch and VecTargets.jl.
  - Source: [GitHub TuringLang/CoupledHMC.jl](https://github.com/TuringLang/CoupledHMC.jl)

**NUTS**
- No paper coupling NUTS for unbiased estimation turned up in searches through October 2026. Xu et al. (2021) list it as future work — [ar5iv 2104.05134](https://ar5iv.labs.arxiv.org/html/2104.05134)
- **[Preprint]** Bou-Rabee & Oberdörster, "Mixing of the No-U-Turn Sampler and the geometry of Gaussian concentration" (arXiv 2410.06978, October 2024) — [arXiv](https://arxiv.org/html/2410.06978v1)
  - Result: NUTS started in the concentration region of the canonical Gaussian mixes in O(d^(1/4)) iterations, up to logarithmic factors.
  - Method:
    - The proof uses a **synchronous coupling** of two NUTS copies, followed by a **one-shot coupling** that makes them meet exactly once they are close.
    - NUTS is recast as an accept/reject chain whose "accept" kernel is uniform HMC.
    - Both copies must reach the same orbit length and pick the same index. This holds on high-probability events under a step-size condition.
  - Pathology: some step sizes make NUTS miss U-turns ("looping"). In one example most transitions fail to detect any U-turn and orbits reach about 32π. A randomly spaced time grid is proposed as an open fix.
- **[Preprint]** Oberdörster (arXiv 2507.13259, July 2025) continues the coupling-based analysis of NUTS — [arXiv](https://arxiv.org/html/2507.13259v1)
- **[Preprint]** GIST (Bou-Rabee, Carpenter & Marsden; arXiv 2404.15253, v4 June 2025) treats HMC tuning parameters as auxiliary variables updated by Gibbs steps. NUTS, randomized HMC and multinomial HMC are special cases. A follow-up adds local step-size adaptivity to NUTS (arXiv 2408.08259). Searches found no coupled GIST sampler. — [arXiv 2404.15253](https://arxiv.org/pdf/2404.15253), [arXiv 2408.08259](https://arxiv.org/pdf/2408.08259), [Gelman blog](https://statmodeling.stat.columbia.edu/2024/04/24/gist-gibbs-self-tuning-for-hmc/)

### Inferences
- **Recipe for v0.1 (published methods only).** Couple HMC as Heng & Jacob do:
  - fixed ε and L, with a fixed mass matrix estimated in a pilot run;
  - shared momentum and a shared MH uniform;
  - γ = 1/20 mixture with a maximal-coupling RWMH step, σ = 1e-3, applied in whitened coordinates if a mass matrix is used;
  - optionally the Xu et al. maximal index coupling for multinomial HMC, which is cheap at O(L) per step;
  - the W2-OT index coupling (needs an LP, e.g. `scipy.optimize.linprog`) as an option rather than the default, since its gains in Xu et al. were mixed (better on logistic regression, worse on the Cox process).
- **Choose (ε, L) from pilot meeting times, not from HMC efficiency.** The logistic-regression example shows the HMC-optimal setting can fail to contract.
- **Periodicity.** Exact-Gaussian contraction depends on trajectory length (H&J), and Metropolis HMC's meeting times surge near closed orbits (Xu et al.). Randomizing L or ε per iteration from a fixed distribution, shared by both chains, is a valid fixed mixture kernel and should damp periodicity. **[Unpublished: this note]** — untested in the coupled literature beyond Xu et al.'s random forward/backward split.
- **Faithfulness in floating point (implementation pitfall).** Once the chains meet, both must execute bit-identical operations. Batched evaluation of (x, y) and single evaluation can differ in the last bits (BLAS reductions, SIMD). The safe design: detect meeting by exact equality, then advance only one chain. The estimator needs only one chain after the meeting time. **[Unpublished: this note]**
- **Obstacles to a coupled NUTS. [Unpublished: this note]**
  1. The orbit length (tree depth) is a discontinuous function of the state. Under CRN on momentum, direction bits and selection uniforms, nearby chains usually build the same orbit, but a flipped U-turn test produces different orbit lengths and an O(1) "decoupling" jump, like an accept/reject disagreement.
  2. Index selection (biased progressive or multinomial sampling across subtrees) must be coupled across orbits that may differ in length. Maximal or OT couplings of categoricals on different supports still work, but with less overlap.
  3. Exact meeting would still need the RWMH mixture, or an implementable one-shot momentum coupling.
  4. Step-size and mass-matrix adaptation must be frozen.
  5. Per-chain trajectory lengths vary, which complicates vectorization.
  - Bou-Rabee & Oberdörster's synchronous-coupling analysis suggests CRN-coupled NUTS behaves like coupled uniform/multinomial HMC inside the concentration region. So a coupled NUTS (CRN everywhere, plus maximal coupling of the final index, plus the RWMH mixture) looks feasible, but it is a `telescopic` experiment, not a published method.
  - A GIST-style sampler, where the path length is drawn from a conditional given (x, p), could be coupled with a maximal coupling of that conditional. This is speculative.

### Gaps
- No published coupled NUTS, and no published coupled GIST or other locally adaptive HMC.
- Heng & Jacob's Gaussian dimension-scaling results are given only as a figure. Searches turned up no tabulated meeting times for coupled HMC above d = 4,096.
- No maintained, documented Python package for coupled HMC was found. `ColCarroll/couplings` is a pip-installable library-plus-essay (MIT) based on Jacob et al. and Heng & Jacob, but its kernels were not inspected — [GitHub ColCarroll/couplings](https://github.com/ColCarroll/couplings)
- The κ momentum coupling has been benchmarked only on Rosenbrock (H&J) and as an option in CoupledHMC.jl. Its value on standard benchmarks is unknown.

## 2. Langevin methods (MALA, unadjusted and underdamped Langevin)

### Takeaway
MALA is the simplest gradient-based kernel to couple to exact meeting. Its Gaussian proposals share a covariance, so a reflection-maximal coupling gives exact meetings with no extra RWMH kernel. This construction has been used in published coupling diagnostics (the L-lag paper). Unadjusted and kinetic (underdamped) Langevin are handled differently: unbiasedness comes from randomized multilevel Monte Carlo (MLMC) across discretization levels (Chada et al., preprint, reportedly in press), or from contracting couplings with Rhee–Glynn truncation (Agapiou et al. 2018), not from exact meeting. Coupling theory for kinetic Langevin is active (Eberle, Guillin & Zimmer 2019; sticky/one-shot couplings in a 2026 preprint). No JOA-style exact-meeting estimator for Metropolized kinetic Langevin was found.

### Cited Findings
- **L-lag couplings: Biswas, Jacob & Vanetti, arXiv 1905.09971** — [ar5iv](https://ar5iv.labs.arxiv.org/html/1905.09971)
  - MALA is coupled with a **reflection-maximal coupling of the Langevin proposals, with no RWMH mixture**.
  - Mixing-time bounds for MALA and ULA were estimated across dimension, with step sizes d^(−1/6) (MALA) and 0.1·d^(−1/6) (ULA). The results agree with theory giving O(d) for ULA and O(d²) for MALA from a non-warm start.
- Heng & Jacob's supplement includes a MALA coupling — [ar5iv 1709.00404](https://ar5iv.labs.arxiv.org/html/1709.00404)
- Atchadé & Jacob (2024) — [arXiv 2406.06851](https://arxiv.org/html/2406.06851)
  - The **reflection-maximal coupling** (their Algorithm 4, from Bou-Rabee et al. 2020) maximizes P(X = Y) and otherwise falls back to a reflection coupling, at deterministic cost.
  - Reflection couplings "can be directly used for Langevin Monte Carlo".
- **[Preprint; reportedly in press]** Chada, Leimkuhler, Paulin & Whalley, "Unbiased kinetic Langevin Monte Carlo with inexact gradients", arXiv 2311.05025 (v4, 3 December 2025) — [arXiv](https://arxiv.org/abs/2311.05025)
  - No Metropolis correction. Chains at different discretization levels are coupled inside an MLMC scheme.
  - The estimator is unbiased, has finite variance and satisfies a CLT.
  - It needs **O(d^(1/4)·ε^(−2)) expected gradient evaluations** for Lipschitz test functions, with no warm start. Similar bounds hold for stochastic gradients, and the cost does not depend on dataset size.
  - Experiments: MNIST multinomial regression and Poisson regression of soccer scores.
  - Reported gains: 2–3 orders of magnitude over randomized HMC on large problems.
  - Status conflict: the arXiv page lists no journal reference, but a CityU Scholars listing says the paper is accepted in the *Annals of Statistics* (in press 2026) — [CityU Scholars](https://scholars.cityu.edu.hk/en/publications/unbiased-kinetic-langevin-monte-carlo-with-inexact-gradients/)
- **[Published]** Eberle, Guillin & Zimmer, "Couplings and quantitative contraction rates for Langevin dynamics", *Ann. Probab.* 47(4):1982–2010 (2019). It combines reflection and synchronous couplings and proves contraction in a tailored Wasserstein distance with explicit rates — [arXiv 1703.01617](https://arxiv.org/pdf/1703.01617)
- **[Preprint]** Bou-Rabee, Cox & Schieven, "On couplings for kinetic Langevin diffusions", arXiv 2605.31088 (29 May 2026) — [arXiv](https://arxiv.org/abs/2605.31088)
  - For a quadratic potential, no Markovian coupling captures the asymptotic decay rate of the total-variation (TV) distance.
  - The iterated one-shot (sticky) coupling attains the best Markovian rate and has an exact contraction formula.
  - A non-Markovian "coalescence" coupling explains sharp TV bounds and extends to the OBABO splitting scheme.
  - The abstract does not mention unbiased estimation.
- **[Published]** Agapiou, Roberts & Vollmer, *Bernoulli* 24(3):1726–1786 (2018) — [ar5iv 1411.7713](https://ar5iv.labs.arxiv.org/html/1411.7713), [Bernoulli](https://projecteuclid.org/euclid.bj/1517540459)
  - Unbiased estimators built from *contracting* (not coalescing) couplings plus Rhee–Glynn random truncation, including a "simulatable contracting coupling of the pCN algorithm".
  - Quote: "we do not require the strong coalescent couplings needed for coupling from the past".
- **[Published]** Papp & Sherlock, "Scalable couplings for the random walk Metropolis algorithm", *JRSS-B* 87(3) (2025) — [arXiv 2211.12585](https://arxiv.org/abs/2211.12585)
  - A low-rank modification of the synchronous coupling is provably optimally contractive in high-dimensional regimes.
  - They identify a shortcoming of the reflection coupling and propose a fix.
  - This is the MH baseline against which Langevin couplings should be judged.

### Inferences
- **v0.1 candidate (published usage):** coupled MALA with reflection-maximal proposals.
  - The meeting probability is about 1 − TV(N(μ_x, 2hI), N(μ_y, 2hI)), where μ = x + h∇log π(x).
  - Because h shrinks with dimension (about d^(−1/3)), chains must contract well within √h before meeting is likely.
  - A per-step "two-scale" choice of coupling is valid, because both options couple the *same* marginal kernel: synchronous/CRN when far apart, reflection-maximal when close. What is not valid is a distance-dependent mixture of different kernels. **[Unpublished: this note]**, combining the review's warning with Biswas et al.'s two-scale idea.
- Coupling ULA to exact meeting would give unbiased estimates of ULA's own (biased) invariant law, not π. Unbiasedness for π needs either an MH correction (MALA, HMC) or MLMC across step sizes (Chada et al.). The MLMC route is a different estimator family, so it belongs in `telescopic` as a later experiment, not in v0.1.
- Metropolized kinetic Langevin (for example OBABO with MH, or MALT) has no published exact-meeting coupling, so it is unpublished territory.

### Gaps
- No published table of coupled-MALA meeting times against dimension was found; the L-lag paper reports mixing-time bounds in a figure.
- The journal status of Chada et al. is not confirmed.
- No Python implementation of the MLMC kinetic-Langevin estimator was located.

## 3. Gibbs samplers (full conditionals, shrinkage regression, discrete states, data augmentation; CRN versus maximal couplings)

### Takeaway
Gibbs samplers are coupled one component at a time:
- **Maximal couplings** of full conditionals produce exact meetings. This works in low dimension (Gamma conditionals; Ising single-site updates).
- **CRN alone** contracts but never meets.
- Maximal couplings alone fail to contract in high dimension: one-scale meeting times grow exponentially in p.

The published fix is a **two-scale** (or "switch-to-CRN") coupling, which uses CRN while the chains are far apart and maximal couplings when they are close (Biswas et al. 2022). It brought horseshoe-type regressions with about 100,000 covariates to meet in under about 1,000 iterations. Discrete-state Gibbs works, but meeting times explode near phase transitions. Data-augmentation Gibbs (Pólya–Gamma) has been coupled with maximal couplings.

### Cited Findings
- **Jacob, O'Leary & Atchadé (JRSS-B 82(3):543–600, 2020) [Published]** — [ar5iv 1708.03625](https://ar5iv.labs.arxiv.org/html/1708.03625)
  - Pump-failure model: Gibbs with Gamma conditionals, a maximal coupling for each conditional update. With k = 7 (99% quantile of τ) and m = 70 the paper reports 0.94 for the unbiased estimator, against 1.06 for regeneration and 1.08 for the underlying Gibbs sampler. (The extraction labels these "efficiency"; check the paper's definition before quoting.)
  - Ising model, 32×32: single-site heat-bath Gibbs with "the maximal coupling of each conditional distribution". Meeting times rise above 10⁶ as θ approaches the critical value, so unbiased estimators from heat bath alone are expensive for θ > 0.48.
  - Bayesian variable selection (the MH sampler of Yang, Wainwright & Jordan on model space):
    - Coupling: identical coordinate flips; maximal couplings of the index draws for swap moves; common acceptance uniforms.
    - Meeting times of order 10⁴–10⁵ for n ∈ {500, 1000}, p ∈ {1000, 5000}, growing **linearly in p**.
  - Overall cost: the loss relative to MCMC is about 1 + (k + E[τ])/(m − k). "Coupling pairs of Markov chains does not improve their marginal mixing properties."
- **Biswas, Bhattacharya, Jacob & Johndrow (JRSS-B 84:973–996, 2022) [Published]** — [ar5iv 2012.04798](https://ar5iv.labs.arxiv.org/html/2012.04798), [arXiv](https://arxiv.org/abs/2012.04798v3)
  - The sampler: blocked Gibbs for regression with Half-t(ν) local scales (ν = 1 is the horseshoe).
  - How each block is coupled:
    - local precisions η: slice sampling, coupled with CRN on the auxiliary uniforms plus a maximal coupling with independent residuals;
    - global precision ξ: coupled MH step;
    - σ²: maximal coupling of the two inverse-gamma conditionals;
    - β: CRN, using the fast structured Gaussian sampler of Bhattacharya et al.
  - **One-scale coupling** (always maximal for η): meeting times grow **exponentially with p**.
  - **Two-scale coupling**:
    - The metric is d = P(η_{t+1} ≠ η̃_{t+1}), estimated by Rao–Blackwellization.
    - If d ≤ d₀ = 0.5 the coupling is maximal; otherwise it is CRN.
    - This gives "orders of magnitude" reductions. d₀ = 0 means pure CRN, and the chains "will never exactly meet".
  - **Switch-to-CRN variant**: maximally couple the components one by one until the first failure, then use CRN for the rest. It has exponentially tailed meeting times, but the rate tends to 1 as p grows.
  - Theory:
    - Proposition 3.1: CRN and maximal coupling have the same expected one-step distance for monotone functions.
    - Proposition 3.2: under maximal coupling the largest deviation across components grows like log log p.
    - The authors say contraction under CRN "requires further work".
  - Results:
    - **Maize GWAS** (n = 2,266, p = 98,385): converged in fewer than 1,000 iterations (lag 750).
    - **Riboflavin** (n = 71, p = 4,088): about 500 iterations.
    - A small increase of ν above 1 cut meeting times by orders of magnitude. The authors recommend ν ∈ [1, 2].
  - Cost:
    - O(n²p) per iteration; the two-scale kernel costs about 2× one chain.
    - About 60 s per coupled iteration on maize, so one coupled run can take over a day.
  - **[Code]** [niloyb/CoupledHalfT](https://github.com/niloyb/CoupledHalfT)
- Douc, Jacob, Lee & Vats ("Solving the Poisson equation using coupled Markov chains", arXiv 2206.05691) also use the switch-to-CRN strategy for multi-component updates — [arXiv](https://arxiv.org/pdf/2206.05691)
- **Data augmentation.** The L-lag paper couples the Pólya–Gamma Gibbs sampler for logistic regression (German credit, n = 1000, d = 49) with maximal couplings of the Pólya–Gamma and Normal draws. Its TV bounds were smaller than those of every HMC variant tried — [ar5iv 1905.09971](https://ar5iv.labs.arxiv.org/html/1905.09971)
- Coupling primitives (Atchadé & Jacob 2024) — [arXiv 2406.06851](https://arxiv.org/html/2406.06851)
  - In 1-d, CRN through quantile functions minimizes E[(X − Y)²] (the optimal-transport coupling).
  - The modified γ-coupling (Gerber & Lee) has a parameter η. Only η = 1 maximizes P(X = Y), but then the cost variance can blow up as TV → 0.
  - A mixture representation has deterministic cost but needs ∫min(p, q).
  - Couplings for spike-and-slab Gibbs samplers appear in Atchadé & Wang (2023), in the context of contraction.

### Inferences
- **Design rule supported by the evidence:**
  - CRN (shared standard normals or uniforms) for high-dimensional Gaussian blocks, to get contraction;
  - maximal couplings for low-dimensional or conjugate univariate conditionals, to get meetings;
  - a per-step switch between them based on the current pair of states, which is valid because it changes only the coupling, not the marginal kernel.
- For black-box log densities, Gibbs is not available automatically. `telescopic` could expose an opt-in "coupled Gibbs" API in which users supply conditional samplers. Standard families (Normal, Gamma, InvGamma, Beta, categorical) would ship with built-in maximal couplings, using the deterministic-cost mixture form when ∫min(p, q) is computable.
- Discrete conditionals (categorical) can be maximally coupled exactly and cheaply. Contraction is the binding constraint, as the Ising example near criticality shows.

### Gaps
- No general theory says when CRN contracts for Gibbs samplers.
- No published Python package for coupled Gibbs was found. The `adriens63/mcmc_unbiased_gibbs_sampler_py` repository exists, but its contents were not inspected — [GitHub search result](https://github.com/adriens63/mcmc_unbiased_gibbs_sampler_py)

## 4. Slice sampling and elliptical slice sampling

### Takeaway
Published coupled slice sampling exists only as a component inside larger couplings: Biswas et al.'s univariate slice update for the local precisions uses CRN on the auxiliary uniforms plus a maximal coupling with independent residuals. It also appears in contraction theory and in perfect-sampling work. **No published coupling of elliptical slice sampling (ESS), or of generic stepping-out slice samplers, for unbiased estimation was found.** My unpublished experiment suggests that CRN-coupled ESS contracts to machine precision on a log-concave latent Gaussian model. Mixing in a pCN step with reflection-maximal coupling then produces exact meetings, with meeting times that grow quickly beyond d of about 100–200 in my test model.

### Cited Findings
- Biswas et al. (2022): the local precisions η_j are updated by slice sampling at O(p) cost. They are coupled by CRN on the slice auxiliary uniforms plus a maximal coupling with independent residuals (Algorithms 2 and 5), inside the two-scale scheme — [ar5iv 2012.04798](https://ar5iv.labs.arxiv.org/html/2012.04798)
- The Atchadé & Jacob review does **not** discuss slice sampling (or ensemble samplers) — [arXiv 2406.06851](https://arxiv.org/html/2406.06851)
- **[Published]** Natarovskii, Rudolf & Sprungk, *Ann. Appl. Probab.* 31(2):806–825 (2021): Wasserstein contraction of simple slice sampling for log-concave, rotationally invariant targets (a coupling used for analysis) — [Project Euclid](https://www.projecteuclid.org/journals/annals-of-applied-probability/volume-31/issue-2/Quantitative-spectral-gap-estimate-and-Wasserstein-contraction-of-simple-slice/10.1214/20-AAP1605.full)
- A follow-up ("revisited", arXiv 2305.16984) defines a coupling for simple slice sampling from a uniform radial draw and a uniform direction — [arXiv](https://arxiv.org/pdf/2305.16984)
- Coupled slice-type samplers appear in the perfect-sampling literature, for example perfect slice samplers for mixtures, as cited in a 2024 survey of perfect sampling — [arXiv 2401.02518](https://arxiv.org/pdf/2401.02518)
- The ESS update (Murray, Adams & MacKay 2010) is f' = f·cos θ + ν·sin θ, with ν drawn from the Gaussian prior and θ found by a shrinking bracket. It has no tuning parameters — [PMLR v9](https://proceedings.mlr.press/v9/murray10a/murray10a.pdf)
- pCN admits a simulatable contracting coupling (Agapiou et al. 2018) — [ar5iv 1411.7713](https://ar5iv.labs.arxiv.org/html/1411.7713)

### Inferences
- **[Unpublished: this note] Mechanism.** Couple ESS by sharing between the chains:
  - the auxiliary draw ν;
  - the slice uniform (each chain's threshold is its own log L(f) + log u);
  - the initial angle;
  - the stream of shrinkage uniforms.
  Whenever both chains accept the same angle θ, f₁' − f₂' = (f₁ − f₂)·cos θ, an exact contraction. Disagreements, where the slice tests lead to different accepted angles, become rarer as the chains approach each other. So ESS behaves like CRN-coupled HMC (it contracts) and unlike CRN-coupled RWMH (where the difference is preserved under joint acceptance).
- **[Unpublished: this note] Exact meeting.** Mix in, with probability γ, a pCN step:
  - proposal ρf + βLξ, with ρ = √(1 − β²) and L the Cholesky factor of the prior covariance;
  - accepted on the likelihood ratio only;
  - the two chains' proposals are Gaussians with equal covariance, so they can be drawn from a **reflection-maximal coupling** (in whitened coordinates) with a common acceptance uniform.
  This is the Heng & Jacob device transplanted to prior-reversible kernels.
- **[Unpublished: this note] Experiment setup.**
  - Model: Gaussian-process prior on d grid points in [0, 1], squared-exponential kernel with lengthscale 0.1 and jitter 1e-6. One Bernoulli-logit observation per grid point, simulated from a prior draw (fixed seed per d).
  - Initialization: independent prior draws.
  - Mixture: γ = 1/20, β = 1e-2.
- **[Unpublished: this note] Results.**
  - **ESS alone (no mixture)**, distance between chains:
    - d = 10: ~1e-9 by iteration 50 and 9e-16 by iteration 100.
    - d = 50: 4e-10 by 100 and 5.5e-16 by 200 (seed 0); 2e-7 by 50 (seed 1).
    - d = 200: one seed reached 1e-6 by 500 iterations; the other was still O(1) at 500. The contraction phase can be slow to start when the chains begin far apart.
  - **ESS + pCN mixture**, meeting times from 20 replicates:

    | d | Met (of 20, within 5,000) | Median | 90% | Max |
    |---|---|---|---|---|
    | 10 | 20 | 26 | 48 | 56 |
    | 50 | 20 | 68 | 131 | 153 |
    | 100 | 20 | 120 | 228 | 497 |
    | 200 | 20 | 909 | 3,047 | 4,373 |
    | 400 | 2 | — | — | — |

  - Caveats:
    - In this design, d also raises the number of observations and the conditioning of the GP prior (fixed lengthscale on a finer grid), so the dimension trend is confounded.
    - γ and β were not tuned, and there are 20 replicates per setting.
- **[Unpublished: this note] Univariate (stepping-out/shrinkage) slice samplers.**
  - CRN coupling is easy: share the slice uniform, the interval offset and the shrinkage uniforms.
  - Exact meeting via a maximal coupling is cheap only when the slice set is known in closed form, as with Biswas et al.'s η update. Otherwise an MH-type mixture step is needed.
  - zeus's ensemble slice sampling inherits the ensemble obstacles in Section 5, because its directions come from walker differences.
- For `telescopic`, coupled ESS plus a coupled pCN mixture is a promising **experiment** for latent-Gaussian models (GP classification, spatial models). It needs no tuning beyond γ and β, but the user must provide the prior covariance (or a sampler for it) and the log-likelihood separately.

### Gaps
- No published coupled ESS or generic slice-sampler coupling for unbiased MCMC was found, despite several targeted searches.
- My experiment covers one log-concave model with few replicates. Its behaviour on non-log-concave likelihoods is unknown.

## 5. Ensemble samplers (emcee stretch, walk and differential-evolution moves; zeus ensemble slice sampling)

### Takeaway
**No published coupling of affine-invariant or other emcee-type ensemble samplers exists.** My analysis and NumPy experiments (unpublished) show the natural CRN coupling of stretch and differential-evolution (DE) moves does not contract at all:
- Two coupled ensembles stay as far apart as two independent ensembles.
- A 1e-8 perturbation grows exponentially to O(1).
- Adding a Heng–Jacob-style step with maximally coupled proposals produced **zero meetings in 50,000 iterations even at d = 1**.

The defensible design is to use an ensemble only as an **independent pre-tuning stage**, supplying the initial distribution π₀ and a preconditioner, for coupled single-chain kernels (HMC, MALA, RWMH). That matches accepted published practice.

### Cited Findings
- The stretch move (Goodman & Weare 2010) moves each walker along the line to a randomly chosen partner walker. It needs no preconditioning, mass matrix or step-size tuning — [CAMCoS 5(1)](https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf)
- R's `mcmcensemble` implements stretch and differential-evolution ensemble moves — [CRAN](https://cran.r-project.org/web/packages/mcmcensemble/mcmcensemble.pdf)
- zeus implements ensemble slice sampling — [Springer, Stat. Comput. 2021](https://link.springer.com/article/10.1007/s11222-021-10038-2)
- Neither the Atchadé & Jacob review nor any search result describes coupled ensemble samplers. Searches for stretch-move, DE and ensemble couplings for unbiased MCMC returned nothing relevant — [arXiv 2406.06851](https://arxiv.org/html/2406.06851)
- Pre-tuning on independent preliminary runs is accepted practice in the coupled literature:
  - Jacob et al. use preliminary runs to refine π₀ and the proposal covariance — [ar5iv 1708.03625](https://ar5iv.labs.arxiv.org/html/1708.03625)
  - Xu et al. reuse NUTS-adapted parameters from preliminary runs — [ar5iv 2104.05134](https://ar5iv.labs.arxiv.org/html/2104.05134)
  - van den Boom et al. use a single preliminary adaptive SMC run to fix all aspects of their sampler — [ar5iv 2103.05176](https://ar5iv.labs.arxiv.org/html/2103.05176)
  - CoupledHMC.jl runs NUTS uncoupled for tuning — [GitHub](https://github.com/TuringLang/CoupledHMC.jl)

### Inferences
**[Unpublished: this note] Structural obstacles**

(a) **The state is the whole ensemble.** It has K·d dimensions (K ≥ 2d walkers in emcee practice). Exact meeting requires all K walkers to coincide at once.

(b) **Proposals live on lines.**
- Walker k's stretch proposal is X_j + z(X_k − X_j), a 1-d distribution on the line through X_j and X_k. DE and zeus directions are walker differences.
- For two ensembles these proposal laws sit on different lines, so they are mutually singular and **a maximal coupling has zero meeting probability**.
- This stays true even if the partner walker has already met, unless X'_k happens to lie on the same line (a measure-zero event).
- So the ensemble move itself can never create a meeting.

(c) **A met walker does not stay met.**
- If walker k has met (X_k = X'_k) but its partner j has not, a jointly accepted CRN stretch gives Y − Y' = (1 − z)(X_j − X'_j) ≠ 0.
- So walker-level meetings decouple again, and the ensemble must meet all at once.

(d) **Affine invariance makes CRN blind to affine discrepancies.**
- Under CRN with joint acceptance, any affine relation X' = AX + b between the ensembles is preserved, because the stretch and DE maps are affine-equivariant.
- So the proposal map gives no contraction, unlike HMC flows, pCN or ESS, whose deterministic parts shrink differences.
- Only accept/reject *disagreements* change the discrepancy, and those cause O(1) jumps.
- Heuristic for a = 2: z has E[z] ≈ 1.167 and E[z²] ≈ 1.55. For walker-difference components uncorrelated across walkers, a jointly accepted update multiplies the expected squared difference by E[z²] + E[(1 − z)²] ≈ 1.77 > 1, so it is locally expansive.

(e) zeus's ensemble slice sampling and DE moves share (b) to (d).

**[Unpublished: this note] Experiments**

Setup:
- Target N(0, I_d). By affine equivariance of the moves and their CRN coupling, any non-degenerate Gaussian gives the same law.
- K = max(2d, 8) walkers (even).
- emcee-style red/blue half-ensemble updates.
- Both ensembles share the random split, partner indices, stretch factors (a = 2) or DE γ (2.38/√(2d)), and acceptance uniforms.
- Initialization: independent N(0, 4I).

Results:

1. **Pure CRN, 20,000 sweeps, 3 seeds each.** The Frobenius distance stayed at the independent-ensemble level √(2Kd): about 3.5–6.8 at d = 2 (K = 8), about 10 at d = 5, about 20 at d = 10 and about 40 at d = 20. This held for both the stretch and DE moves. In short, there was **no contraction**.

2. **Ensembles started 1e-8 apart (stretch move).** The difference grew exponentially:

   | d | Distance at t = 100 | t = 300 | t = 1,000 | t = 3,000 |
   |---|---|---|---|---|
   | 1 | ~8e-5 to 1e-4 | O(1) | — | — |
   | 5 | — | ~3e-5 | O(10) | — |
   | 20 | — | — | ~1e-5 | 27 (one seed); 7.7e-3 (other) |

   So the coupling is **locally expansive**, and the "contract, then meet" trick has nothing to work with.

3. **Stretch move plus γ = 1/20 full-ensemble RWMH step** (σ = 1e-3, proposals reflection-maximal coupled across the K·d coordinates, common acceptance uniform): **0 of 10 replicates met within 50,000 iterations for every d ∈ {1, 2, 5, 10}**.

4. **Baseline, single-chain RWMH with reflection-maximal coupling** (scale 2.38/√d, same target, 10 replicates): median meeting times **6, 10, 32, 68, 142, 323 for d = 1, 2, 5, 10, 20, 50**, roughly linear in d. This matches the reported good behaviour of reflection-maximal RWMH on spherical Gaussians (Atchadé & Jacob 2024, [arXiv 2406.06851](https://arxiv.org/html/2406.06851)).

**Candidate constructions, in order of preference**

- **(A) Recommended for v0.1.** Run the ensemble only as an independent pilot (emcee-like UX), then freeze:
  - a preconditioner (mass matrix or proposal covariance);
  - an initial distribution π₀ (for example a Gaussian fitted to the pilot, or its empirical draws).

  Then run coupled HMC, MALA or RWMH chains. Each estimator stays unbiased given the pilot, because the pilot is independent of the coupled runs.
- **(B) An "ensemble of coupled pairs."** R independent coupled pairs. This is just the standard replicate structure, embarrassingly parallel with no interaction.
- **(C) Interacting moves with ensemble-covariance Gaussian proposals.** Walker-wise maximal couplings would need overlapping proposal laws. But the two ensembles' covariance estimates differ (each from about K/2 walkers in d dimensions) until the complementary halves have met, which is a chicken-and-egg problem. Not promising.
- **(D) Speculative.** An interacting move whose deterministic part contracts differences, such as ensemble-preconditioned gradient or autoregressive proposals with a frozen preconditioner. This would be a research project.
- **Exchangeability.** Because emcee's kernel is permutation-equivariant, meeting could be defined up to a relabelling of walkers. That relaxes (c) but not (b) or (d).

### Gaps
- There is no literature to cite on this topic. Every statement in the Inferences above is my own analysis or experiment.
- The experiments cover only Gaussian targets, CRN couplings, and stretch and DE moves.
- Untested alternatives:
  - optimal-transport matching of walkers across the two ensembles;
  - ensemble-level independence proposals;
  - zeus-specific couplings.

## 6. Parallel tempering, multimodal targets and coupled SMC samplers

### Takeaway
Replica exchange has been coupled with CRN for the swap decisions and maximal couplings for the within-replica updates. Meeting then requires **every replica in the two ensembles to coincide**, so meeting time is non-monotone in the number of replicas: it was smallest at 16 chains on a 32×32 Ising model. Multimodality is the main way unbiased MCMC can mislead, because meeting times sampled from single-mode starts look small. The most robust published route for multimodal static targets is **coupled particle MCMC built on a tempered SMC sampler**: coupled particle independent MH (PIMH) plus coupled conditional SMC (van den Boom et al. 2022). Its cost is dominated by the SMC runs.

### Cited Findings
- **Coupled parallel tempering on the Ising model (Jacob, O'Leary & Atchadé 2020)** [Published] — [ar5iv 1708.03625](https://ar5iv.labs.arxiv.org/html/1708.03625)
  - Setup: N replicas on an equally spaced θ grid from 0.3 to 0.55, swap probability 0.01.
  - CRN decides both whether to swap or sweep and whether each proposed swap is accepted. Single-site updates are maximally coupled.
  - The two ensembles meet only when all N chains are identical.
  - "The average meeting time first decreases with the number of chains, but then increases again", with the minimum at **N = 16**.
  - Production run: N = 16, k = 10⁵, m = 2×10⁵, 500 processors for 30 minutes, giving **1,858 estimators** (1–7 per processor).
- **Parallel tempering in the L-lag paper** [ar5iv 1905.09971](https://ar5iv.labs.arxiv.org/html/1905.09971)
  - Setup: 12 replicas with inverse temperatures from 0.3 to 0.46, swap frequency 0.02. The same uniform decides each swap in both ensembles, and single-site Gibbs is maximally coupled per site. All pairs must meet.
  - Lags: L = 10⁶ for single-site Gibbs and 2×10⁴ for PT.
  - Result: TV bounds plateau for single-site Gibbs and converge faster for PT.
- **Multimodality pitfalls (Jacob, O'Leary & Atchadé 2020)**, on the target ½N(−4,1) + ½N(4,1) with RWMH — [ar5iv 1708.03625](https://ar5iv.labs.arxiv.org/html/1708.03625)
  - Proposal variance 9: mean meeting time 20, 99% quantile 105.
  - Proposal variance 1: mean 769, 99% quantile 9,186.
  - With π₀ = N(10, 1), k = 50 and m = 500 gave a 95% CI of [0.799, 0.816] for a quantity whose true value is about 0.421.
  - "Poor chain performance and ill-chosen initializations can still produce misleading results for any finite R."
- Atchadé & Jacob (2024) — [arXiv 2406.06851](https://arxiv.org/html/2406.06851)
  - From a single-mode start, users "might then observe a small empirical average of the meeting time, even after many independent runs".
  - Mitigations: a spread-out π₀, or a larger lag.
  - The review lists coupled simulated tempering for sparse CCA (Zhu & Atchadé 2023).
- **L-lag diagnostics on a bimodal target** — [ar5iv 1905.09971](https://ar5iv.labs.arxiv.org/html/1905.09971)
  - With L = 18,000 the bounds were stable across N.
  - With L = 1 they were misleading, because some runs jumped modes before meeting.
- **Zhu & Atchadé (JASA 119(548):2647–2657, 2024)** use simulated tempering for multimodality. However, the main text of the arXiv version does not describe any coupling of the tempering chains; any coupling may be in the supplement. This **conflicts** with the review's description and is unverified — [ar5iv 2010.08627](https://ar5iv.arxiv.org/html/2010.08627), [IDEAS](https://ideas.repec.org/a/taf/jnlasa/v119y2024i548p2647-2657.html)
- **Coupled particle MCMC over a tempered SMC sampler: van den Boom, Jasra, De Iorio, Beskos & Eriksson** [Published, *Stat. Comput.* 2022] — [ar5iv 2103.05176](https://ar5iv.labs.arxiv.org/html/2103.05176), [Springer](https://link.springer.com/article/10.1007/s11222-022-10093-3)
  - The outer kernel is a mixture:
    - **coupled PIMH**: both chains use the same SMC proposal and the same acceptance uniform;
    - **coupled conditional SMC**: coupled resampling that maximizes agreement of ancestor indices, plus a shared seed for the inner MCMC moves.
  - Tuning: one preliminary adaptive SMC run (N₀ = 10⁴ particles, ESS threshold 0.8) fixes the temperatures and the number of MCMC moves.
  - Results:
    - Gaussian mixture example: PIMH met fastest; conditional SMC mixed best.
    - Horseshoe regression (p = 20): met in fewer iterations than Biswas et al.'s two-scale Gibbs, but used more compute.
    - Gaussian graphical model (p = 35, n = 471, N = 1,000 particles, 165 temperatures): of 30 runs finished in 72 h on 16 cores, 19 met at τ = 1 and the largest τ was 1,010. The authors note that discarding unfinished runs biases the result.
  - **[Code]** [willemvandenboom/cpmcmc](https://github.com/willemvandenboom/cpmcmc)
- **Coupled PIMH** (Middleton et al., AISTATS 2019, unbiased smoothing). The review notes that it shows the bias of self-normalized importance sampling can be removed in finite time, and that this extends to general SMC samplers — [arXiv 1902.01781](https://arxiv.org/abs/1902.01781), [Atchadé & Jacob 2024](https://arxiv.org/html/2406.06851)

### Inferences
- A coupled PT for continuous targets would combine coupled single-replica kernels (HMC with the RWMH mixture, or reflection-maximal MALA or RWMH) with CRN swap decisions. Like ensembles, though, PT makes meeting a **joint event across replicas**: a swap between a met and an unmet replica pair can split met replicas again. The kernels inside each replica do contract, unlike stretch moves, so this is far more feasible than ensemble coupling, as the Ising evidence shows. The cost grows with the number of replicas. **[Unpublished: this note]** for continuous-target PT.
- For `telescopic`, the most robust published gradient-free route for multimodal targets is **coupled PIMH whose proposal is an adaptive-tempering SMC sampler with its adaptation frozen** after a pilot run. Meeting happens the first time both chains accept the same SMC proposal, so its distribution is governed by how variable the SMC normalizing-constant estimate is. **[Inference]** from the van den Boom and Middleton constructions.
- Documentation must warn that meeting times sampled from a concentrated π₀ can badly understate the truth on multimodal targets. Recommend an overdispersed π₀ and a check with lag L > 1.

### Gaps
- No paper dedicated to coupled PT on continuous targets was found.
- The coupled-simulated-tempering claim (Zhu & Atchadé) could not be verified.
- Rischard, Jacob & Pillai's coupled path sampling for log normalizing constants was not checked in this session.

## 7. Particle MCMC (coupled conditional particle filters, backward sampling, coupled PMMH and PIMH)

### Takeaway
Particle MCMC couplings are mature and published. In the coupled conditional particle filter (CCPF):
- the two filters share the random numbers that propagate particles;
- resampling ancestors are drawn from a maximal coupling ("index-coupled resampling");
- the reference trajectories meet exactly and stay together.

Meeting times stay roughly constant when the particle count N grows linearly with the time horizon T, and conditional backward sampling (CCBPF) removes the need for N to grow with T. Coupled particle marginal MH (PMMH) shares the likelihood estimate whenever the maximally coupled parameter proposals coincide. Its meeting times can be heavy-tailed when the likelihood estimates are noisy.

### Cited Findings
- **Jacob, Lindsten & Schön (JASA 115(530):721–729, 2020)** [Published] — [ar5iv 1701.02002](https://ar5iv.labs.arxiv.org/html/1701.02002)
  - The coupling:
    - common random numbers for propagation;
    - ancestor indices from a maximal coupling of the two resampling distributions, at cost linear in N;
    - a coupled draw of the final trajectory.
    - The meeting time is τ = inf{n : X(n) = X̃(n−1)}, and the trajectories stay equal once they meet.
  - Mean meeting times with N scaled linearly in T, for (N, T) from (128, 50) to (2048, 800):
    - bootstrap filter without ancestor sampling: 17.84 → 13.58;
    - bootstrap filter with ancestor sampling: 7.73 → 6.34;
    - auxiliary filter with ancestor sampling: about 3.
    - Overall, meeting times were "approximately constant or slightly decreasing over T".
  - At T = 100, N = 16, 128, 256, 512 and 1,024 gave mean meeting times of 97, 15, 7, 4 and 3.
  - Ancestor sampling needs tractable transition densities.
  - At matched cost, the method "is not necessarily advantageous" in mean squared error compared with particle filters.
- **Lee, Singh & Vihola (Ann. Stat. 48(5):3066–3089, 2020)** [Published]: with a *fixed* N, the CCBPF's coupling time grows only linearly in T under strong mixing. Earlier conditional-particle-filter results needed N to grow at least linearly with T — [arXiv 1806.05852](https://arxiv.org/pdf/1806.05852), [Project Euclid](https://projecteuclid.org/euclid.aos/1600480943)
- **[Preprint]** Karjalainen, Lee, Singh & Vihola (arXiv 2312.17572, v4 November 2025) — [arXiv](https://arxiv.org/abs/2312.17572)
  - The CBPF's mixing time is **O(log T)** (described as optimal), so the total cost is O(T log T), with N fixed independently of T.
  - A new implementable coupling uses a maximal coupling of the two particle systems at each time step.
  - It gives unbiased, finite-variance estimators, including of the score.
- **Middleton, Deligiannidis, Doucet & Jacob (EJS 14(2):2842–2891, 2020): coupled PMMH** [Published] — [ar5iv 1807.08691](https://ar5iv.labs.arxiv.org/html/1807.08691)
  - The coupling:
    - parameter proposals from a maximal coupling;
    - if the proposals are equal, **one shared likelihood estimate**; otherwise independent estimates;
    - a common acceptance uniform.
    - The chains meet when both accept the same proposal.
  - Theory: polynomial tails for τ, under moment conditions on the likelihood-estimate weights that "might not be satisfied in many non-compact parameter spaces". Noisier estimators give heavier tails.
  - Linear-Gaussian state-space model (T = 100): the best parallel efficiency was at N = 150, and the inefficiency increase over serial PMMH was "under 55%".
  - Neuroscience model (T = 3000, conditional SMC with N = 128):
    - the largest meeting time was 21,570 with small random-walk steps, against 928 with larger steps;
    - with k = 1,000 and m = 10,000, 100 processors for 23 h produced 578 estimators.
  - **[Code]** [lolmid/unbiased_intractable_targets](https://github.com/lolmid/unbiased_intractable_targets)
- **Coupled PIMH for smoothing** (Middleton et al., AISTATS 2019, PMLR pp. 2378–2387) — [arXiv 1902.01781](https://arxiv.org/abs/1902.01781), [review reference list](https://arxiv.org/html/2406.06851)
- **Coupled chains on an importance-sampling-augmented space** (Ruiz, Titsias, Cemgil & Doucet, UAI 2021) give unbiased gradients for VAEs — [arXiv 2010.01845](https://arxiv.org/abs/2010.01845)

### Inferences
- Particle MCMC sits outside a black-box-log-density v0.1. But the same machinery (coupled PIMH and coupled conditional SMC) is the published template for SMC-based coupled samplers on static targets (Section 6). If `telescopic` later adds state-space models, CCBPF is the kernel to implement, because its meeting times scale best in T.

### Gaps
- No Python package implementing CCPF or CCBPF couplings was found in this search, which was not exhaustive.

## 8. Piecewise-deterministic samplers (bouncy particle, Boomerang, coordinate, zig-zag)

### Takeaway
Couplings exist for three piecewise-deterministic Markov process (PDMP) samplers: the bouncy particle sampler (BPS), the Boomerang sampler and a modified coordinate sampler (Corenflos, Sutton & Chopin 2025). At grid times the processes are re-coupled:
- refreshment times are drawn from a maximal coupling of shifted exponentials;
- velocities are drawn from a reflection-maximal coupling;
- bounce times are coupled by coupled thinning.

BPS and coordinate-sampler meeting times grow linearly with d, while Boomerang's stay roughly constant. **Zig-zag resists coupling**, because its velocities can only be switched, never refreshed. PDMPs need event-rate bounds for thinning, so they fit poorly with black-box log densities.

### Cited Findings
- **Corenflos, Sutton & Chopin**, "Debiasing piecewise deterministic Markov process samplers using couplings", *Scand. J. Stat.* 52(4):1932–1974 (2025) [Published]; arXiv 2306.15422 — [arXiv html](https://arxiv.org/html/2306.15422v2), [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/sjos.70015)
- The coupling (BPS):
  - On a time grid with step Δ, the processes forget their scheduled events and are re-coupled.
  - Refreshment times come from a maximal coupling of shifted exponentials, so the processes can re-synchronize.
  - Velocities come from a reflection-maximal Gaussian coupling.
  - Bounce times are coupled by coupled thinning.
  - The Boomerang coupling is analogous, with Gaussian-coupled positions. The coordinate sampler is modified ("on-and-off") and coupled by successive time-synchronization.
  - Estimators come in time-discretised, doubly discretised and time-integrated forms.
  - Source: [arXiv html](https://arxiv.org/html/2306.15422v2)
- Results — [arXiv html](https://arxiv.org/html/2306.15422v2)
  - On N(0, I_d) with refreshment rate 1, BPS and coordinate-sampler coupling times "scale linearly with the dimension"; Boomerang's (with the target as reference) "remain roughly constant".
  - Logistic regression (16 coefficients, n = 100):
    - mean meeting time BPS 108.28, Boomerang 31.96 (95% quantiles 291.76 and 74.14);
    - relative inefficiency Boomerang **1.35** at m = 100k, BPS **1.25** at m = 30k (below 2 for BPS when m ≥ 10k).
- Zig-zag: the construction "should apply to the Zig-Zag sampler mutatis mutandis, but it in fact does not", because velocities "cannot be killed but only switched". Synchronizing one coordinate's clock leads to "a never-ending coupling race" — [arXiv html](https://arxiv.org/html/2306.15422v2)
- Tuning and pitfalls — [arXiv html](https://arxiv.org/html/2306.15422v2)
  - Δ = 8 in the logistic example. A smaller Δ suits BPS in high d. For the coordinate sampler, Δ should scale with d.
  - Refreshment rate λ_ref: 1 on the Gaussian; 4 for BPS and 3 for Boomerang on the logistic example.
  - k is set from the 95% meeting-time quantile.
  - Exact simulation needs global bounds on the event rate for thinning.
  - Tuning the coupled chains independently "may not give a faithful coupling"; tune-then-couple has an upfront cost.
- **[Code]** `AdrienCorenflos/CoupledPDMPs`: "A Julia package for coupling of some PDMPs (Bouncy particle sampler and coordinate sampler)" — [GitHub](https://github.com/AdrienCorenflos/CoupledPDMPs)
- **[Code]** The same author's `coupled_rejection_sampling` (Python) implements the coupled rejection sampler, a general tool for sampling coupled pairs — [GitHub](https://github.com/AdrienCorenflos/coupled_rejection_sampling), [arXiv 2201.09585](https://arxiv.org/pdf/2201.09585)

### Inferences
- PDMP couplings are published and perform well (inefficiency below 2 on logistic regression). But they need tractable rate bounds and model-specific thinning, which is a poor match for `telescopic`'s black-box-first design. Treat them as out of scope for v0.1.

### Gaps
- Zig-zag couplings remain open; the authors suggest adding velocity stochasticity.
- No PDMP coupling comparisons on targets above moderate dimension were found in the text.

## 9. Partitions, trees, and other discrete or trans-dimensional samplers

### Takeaway
For label-invariant discrete spaces, **optimal-transport couplings of the Gibbs conditionals** under a partition metric meet reliably where label-based maximal or CRN couplings stall. This was done for Dirichlet-process mixture clustering, with a theorem guaranteeing unbiasedness and finite variance (Nguyen, Trippe & Broderick 2022). Phylogenetic tree MCMC has been coupled for lagged diagnostics on trees with up to 200 leaves. **No coupling of reversible-jump (trans-dimensional) MCMC for unbiased estimation was found.**

### Cited Findings
- **Nguyen, Trippe & Broderick**, "Many processors, little time: MCMC for partitions via optimal transport couplings" (AISTATS 2022, PMLR pp. 3483–3514) [Published] — [ar5iv 2202.11258](https://ar5iv.labs.arxiv.org/html/2202.11258), [review reference list](https://arxiv.org/html/2406.06851)
  - The coupling:
    - In each leave-one-out reassignment step, the two chains' conditionals are coupled by the OT coupling that minimizes the expected Rand/Mirkin partition distance.
    - Labels are never matched, which sidesteps label switching.
    - An independent-coupling "nugget" with weight η = 10⁻⁵ is mixed in.
  - Cost: O(K̃² + K̃³ log K̃) per step, using network simplex via POT in practice.
  - Label-based maximal and CRN couplings "fail to meet on about 10% of attempts".
  - Split–merge moves are coupled by sharing the random pair (i, j).
  - Theory: unbiased estimator with finite variance and finite expected compute.
  - **[Code]** [tinnguyen96/partition-coupling](https://github.com/tinnguyen96/partition-coupling)
  - Earlier version: "Optimal transport couplings of Gibbs samplers on partitions for unbiased estimation" — [arXiv 2104.04514](https://arxiv.org/pdf/2104.04514)
- **Kelly, Ryder & Clarté**, "Lagged couplings diagnose Markov chain Monte Carlo phylogenetic inference", *Ann. Appl. Stat.* 17(2) (2023) [Published]: a contractive coupling of chains over trees with branch lengths, scalar parameters and latent variables, for trees with up to 200 leaves. It supports joint convergence diagnostics and unbiased estimators — [arXiv 2108.13328](https://arxiv.org/abs/2108.13328), [Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-17/issue-2/Lagged-couplings-diagnose-Markov-chain-Monte-Carlo-phylogenetic-inference/10.1214/22-AOAS1676.short)
- **Variable selection (model space)** is coupled by shared flips plus maximal couplings of swap indices, with meeting times linear in p (see Section 3) — [ar5iv 1708.03625](https://ar5iv.labs.arxiv.org/html/1708.03625)
- **Reversible jump.** No source on coupled reversible-jump samplers was found by targeted search. Related work on Gibbs couplings over partitions and variable selection exists (above).

### Inferences
- The OT-coupling pattern (couple conditionals by OT under a symmetry-invariant metric, plus a tiny independent nugget so that all joint states stay reachable) carries over to other label-invariant problems, such as mixture allocations. It is not a v0.1 priority for continuous black-box targets.

### Gaps
- Reversible-jump and other trans-dimensional couplings are an open area.
- The tree-move coupling details in Kelly et al. were not extracted.

## 10. Adaptation and unbiasedness

### Takeaway
Adaptive MCMC breaks the coupling guarantees for two reasons:
- The telescoping identity needs a lagged copy of a *time-homogeneous* chain whose marginals match the original.
- Meetings must be absorbing (faithful), but chains that adapt on their own histories run different kernels after any divergence, so a meeting does not persist.

The literature says plainly that "a successful coupling of inhomogeneous Markov transitions, e.g. for adaptive MCMC algorithms, remains elusive." The accepted remedy everywhere is to **tune on independent preliminary (pilot) runs, then freeze**: proposal covariance, π₀, HMC (ε, L) chosen from pilot meeting times, SMC schedules, k and m. Re-tuning *between* independent replicates keeps unbiasedness by conditioning, but I found no published statement of that.

### Cited Findings
- Atchadé & Jacob (2024): "A successful coupling of inhomogeneous Markov transitions, e.g. for adaptive MCMC algorithms, remains elusive". Many adaptive algorithms have converging marginals, but "it is unclear how to construct a faithful coupling". Random truncation "could be used, but good performance would depend on a choice of truncation variable" — [arXiv 2406.06851](https://arxiv.org/html/2406.06851)
- The same review's guideline for (k, L, ℓ): draw meeting times with lag L = 1, set k to a high quantile (e.g. 99%) of τ − L, then set L = k and ℓ = 10k. It also warns that keeping only the first C finished runs out of M ≫ C "would introduce a bias" — [arXiv 2406.06851](https://arxiv.org/html/2406.06851)
- Jacob, O'Leary & Atchadé (2020) — [ar5iv 1708.03625](https://ar5iv.labs.arxiv.org/html/1708.03625)
  - Preliminary runs estimate the first two moments, which refine π₀ and the proposal covariance (cut-distribution example).
  - "We expect the optimal tuning of MCMC kernels to be different in the proposed framework than when used marginally."
- Heng & Jacob (2019): (ε, L) chosen by a pilot grid search over meeting times. The HMC-optimal setting failed to contract — [ar5iv 1709.00404](https://ar5iv.labs.arxiv.org/html/1709.00404)
- Xu et al. (2021): multinomial couplings are robust enough to step size that existing HMC adaptation (parameters from preliminary NUTS runs) can be reused — [ar5iv 2104.05134](https://ar5iv.labs.arxiv.org/html/2104.05134)
- CoupledHMC.jl: "FIXME: Adaptation is not supported" in the coupled sampler; NUTS adaptation only in uncoupled runs — [GitHub](https://github.com/TuringLang/CoupledHMC.jl)
- Corenflos, Sutton & Chopin (Appendix A): independent tuning of coupled chains "may not give a faithful coupling", and tune-then-couple has an upfront cost and an unclear burn-in — [arXiv html](https://arxiv.org/html/2306.15422v2)
- van den Boom et al.: "this single run of Algorithm 1 will fix all aspects of the Feynman-Kac model used by coupled particle MCMC" — [ar5iv 2103.05176](https://ar5iv.labs.arxiv.org/html/2103.05176)
- Biswas et al.: lags chosen from preliminary runs — [ar5iv 2012.04798](https://ar5iv.labs.arxiv.org/html/2012.04798)
- Mixture weights between kernels may not depend on the distance between chains (marginal constraint) — [arXiv 2406.06851](https://arxiv.org/html/2406.06851)

### Inferences
- **Why adaptation breaks unbiasedness (reasoning).**
  - The estimator H_{k:m} is unbiased because each chain, viewed alone, is a time-homogeneous Markov chain with kernel P started from π₀, and because coupled pairs stay together after meeting.
  - If each chain adapts on its own past, its marginal law is no longer that of P^t applied to π₀, so the telescoping sum no longer collapses to π(h) in expectation.
  - The two chains' adaptation statistics also differ, so they run different kernels even after their states coincide.
- **Valid patterns (reasoning).**
  1. **Pilot-then-freeze.** Tuning parameters θ are computed from runs independent of the coupled replicates. Then E[H | θ] = π(h) for every θ, so the estimator is unbiased.
  2. **Sequential re-tuning between replicate batches.** Replicate r uses tuning computed from replicates 1 to r−1 only. Each H_r is conditionally unbiased given the past, so the average is unbiased (a martingale-difference argument). However, the replicates are no longer identically distributed, so the CLT and standard errors need a martingale CLT and care. **[Unpublished: this note]** No explicit published statement was found.
  3. **Pre-scheduled kernels.** Deterministic per-iteration parameter schedules shared by both chains are time-inhomogeneous, and how the lag-1 construction would then align kernels is unclear. Treat as open.
- **For `telescopic` v0.1:**
  - Make "pilot, freeze, couple" the only supported mode.
  - Pilot options: an emcee-like ensemble, a short NUTS/HMC warm-up, or a pilot of meeting times for choosing k, m and (ε, L).
  - Record the frozen tuning in the results object.
  - Batch-wise re-tuning across replicates could be an explicitly labelled experiment.

### Gaps
- No published treatment of re-tuning across independent replicates, or of coupling an adaptive sampler directly, was found.

## 11. Cross-family summary: dimension scaling, efficiency, test targets and code

### Takeaway
Two families of couplings scale well with dimension:
- couplings whose deterministic part contracts differences (HMC flows, Gibbs with CRN on Gaussian blocks, ESS and pCN) combined with an occasional maximal-coupling step;
- two-scale (CRN plus maximal) couplings.

Couplings that only translate or affinely transform differences (CRN on RWMH, or on stretch and DE ensemble moves) need maximal or reflection-maximal couplings to meet, and are hopeless for interacting ensembles. Reported inefficiency relative to the underlying MCMC ranges from about 1 to 1.5 in the best tuned cases (HMC logistic regression, PDMP logistic regression) up to about 11 (coupled HMC on a 256-d Cox process).

### Cited Findings
| Family | Coupling mechanism | Exact-meeting device | Dimension or size behaviour reported | Reported efficiency versus underlying MCMC | Test targets | Code | Status |
|---|---|---|---|---|---|---|---|
| Metropolis HMC ([H&J](https://ar5iv.labs.arxiv.org/html/1709.00404)) | Shared momentum and MH uniform; optional κ contractive momentum | γ = 1/20 mixture with maximal-coupling RWMH (σ = 1e-3) | Cox process d up to 4,096 handled; relative inefficiency fell with grid size (11.00 → 2.73) | 1.05–1.18 (logistic regression, d = 302); 2.73–11.00 (Cox) | Gaussians, Rosenbrock, logistic (German credit, 300 covariates), Cox process, truncated Gaussian | [pierrejacob/debiasedhmc](https://github.com/pierrejacob/debiasedhmc) | Published 2019 |
| Multinomial HMC ([Xu et al.](https://ar5iv.labs.arxiv.org/html/2104.05134)) | As above, plus shared forward/backward split and maximal or W2-OT index coupling | Same RWMH mixture | 1,000-d Gaussian; more robust to ε | 0.94–17.69 depending on k, m and coupling (e.g. 1.90 maximal, 0.94 W2, logistic) | Gaussian (1,000-d), logistic (302-d), Cox (256-d) | [TuringLang/CoupledHMC.jl](https://github.com/TuringLang/CoupledHMC.jl) | Published 2021 |
| NUTS | None for unbiased estimation; synchronous plus one-shot coupling used for mixing analysis ([B-R & O](https://arxiv.org/html/2410.06978v1)) | — | Mixing time O(d^(1/4)) for Gaussian (theory) | — | Canonical Gaussian | — | Preprint (analysis only) |
| MALA ([L-lag](https://ar5iv.labs.arxiv.org/html/1905.09971)) | Reflection-maximal coupling of proposals | Built in (Gaussian proposals) | Mixing bounds consistent with O(d²) from non-warm start | Not reported as inefficiency | Multivariate Gaussian | [niloyb/LlagCouplings](https://github.com/niloyb/LlagCouplings) | Published (diagnostics) |
| Kinetic Langevin, no MH ([Chada et al.](https://arxiv.org/abs/2311.05025)) | Coupling across discretization levels (MLMC) | Not needed (randomized MLMC) | O(d^(1/4)·ε^(−2)) gradient evaluations | 2–3 orders of magnitude over randomized HMC (claimed) | MNIST multinomial regression, soccer Poisson regression | Not found | Preprint; reportedly in press at Ann. Stat. |
| Shrinkage-regression Gibbs ([Biswas et al.](https://ar5iv.labs.arxiv.org/html/2012.04798)) | CRN when far, maximal when close (two-scale) | Maximal couplings of conditionals | Fewer than 1,000 iterations at p ≈ 100,000; one-scale exponential in p | Two-scale kernel costs about 2× one chain | Simulated data, maize GWAS, riboflavin | [niloyb/CoupledHalfT](https://github.com/niloyb/CoupledHalfT) | Published 2022 |
| Gamma-conditional and Ising Gibbs ([JOA](https://ar5iv.labs.arxiv.org/html/1708.03625)) | Maximal coupling per conditional | Built in | Ising meeting times above 10⁶ near criticality | 0.94 against 1.08 for Gibbs (pump; see definition caveat) | Pump failure; Ising 32×32 | [pierrejacob/unbiasedmcmc](https://github.com/pierrejacob/unbiasedmcmc) | Published 2020 |
| Parallel tempering ([JOA](https://ar5iv.labs.arxiv.org/html/1708.03625); [L-lag](https://ar5iv.labs.arxiv.org/html/1905.09971)) | CRN swaps; maximal single-site updates | All replicas must meet | Meeting time non-monotone in N; best at N = 16 | — | Ising 32×32 | Same as above | Published |
| Slice sampling | CRN uniforms plus maximal coupling (one Gibbs block of [Biswas et al.](https://ar5iv.labs.arxiv.org/html/2012.04798)) | Maximal coupling | — | — | Horseshoe regression | [niloyb/CoupledHalfT](https://github.com/niloyb/CoupledHalfT) | Published (component only) |
| Elliptical slice sampling | CRN (ν, u, angles); contraction by cos θ | Mixture with reflection-maximal pCN | Median meeting 26 / 68 / 120 / 909 at d = 10 / 50 / 100 / 200; 2 of 20 met by 5,000 at d = 400 | Not measured | GP-logit latent model | Session scripts | **Unpublished: this note** |
| Ensemble stretch / DE | CRN (split, partner, z, u) | Mixture with full-ensemble maximal RWMH | **No contraction; locally expansive; 0 meetings** (d = 1 to 10) | — | N(0, I_d) | Session scripts | **Unpublished: this note** |
| Tempered SMC inside coupled PIMH / conditional SMC ([van den Boom et al.](https://ar5iv.labs.arxiv.org/html/2103.05176)) | Shared SMC proposal and uniform; coupled resampling | Both chains accept the same proposal | Graphical model p = 35: 19 of 30 runs met at τ = 1 | More compute than Gibbs for horseshoe at p = 20 | Gaussian mixture, horseshoe, Gaussian graphical model | [willemvandenboom/cpmcmc](https://github.com/willemvandenboom/cpmcmc) | Published 2022 |
| CCPF / CCBPF ([JLS](https://ar5iv.labs.arxiv.org/html/1701.02002); [LSV](https://arxiv.org/pdf/1806.05852); [KLSV](https://arxiv.org/abs/2312.17572)) | CRN propagation; maximal (index-coupled) resampling | Reference trajectories meet | Meeting times flat with N ∝ T (CCPF); linear in T at fixed N (CCBPF); O(log T) mixing | Not advantageous in MSE at matched cost (smoothing) | Hidden AR, unlikely observation, prey–predator | Not checked | Published (KLSV preprint) |
| Coupled PMMH ([Middleton et al.](https://ar5iv.labs.arxiv.org/html/1807.08691)) | Maximal parameter proposals; shared likelihood estimate when proposals equal | Both chains accept the same proposal | Heavy tails as estimator noise grows | Inefficiency increase under 55% (linear-Gaussian state-space model) | Toy models, Beta-Bernoulli, linear-Gaussian SSM, neuroscience | [lolmid/unbiased_intractable_targets](https://github.com/lolmid/unbiased_intractable_targets) | Published 2020 |
| BPS / Boomerang / coordinate ([CSC](https://arxiv.org/html/2306.15422v2)) | Grid re-coupling; maximal coupling of exponentials; reflection-maximal velocities; coupled thinning | Built in | BPS and coordinate linear in d; Boomerang roughly constant | 1.25 (BPS), 1.35 (Boomerang) on logistic regression | Gaussians; logistic regression (16-d) | [AdrienCorenflos/CoupledPDMPs](https://github.com/AdrienCorenflos/CoupledPDMPs) | Published 2025 |
| Partitions ([NTB](https://ar5iv.labs.arxiv.org/html/2202.11258)) | OT coupling under the Rand metric, with nugget | Built in | All replicas met within budget; label-based couplings fail about 10% of the time | Lower error than naive parallel chains at equal wall time | DPMM on gene expression, synthetic, seeds, abalone; graph colouring | [tinnguyen96/partition-coupling](https://github.com/tinnguyen96/partition-coupling) | Published 2022 |
| Phylogenetic trees ([KRC](https://arxiv.org/abs/2108.13328)) | Contractive coupling of tree moves | Built in | Up to 200 leaves | — | Phylogenetic posteriors | Not found | Published 2023 |

- Other Python code located:
  - `ColCarroll/couplings`: a pip-installable library and essay, MIT licence, based on Jacob et al. and Heng & Jacob; kernels not inspected — [GitHub](https://github.com/ColCarroll/couplings)
  - `ylefay/unbiased_mcmc_with_couplings`: a JAX coupled-MH course project — [GitHub](https://github.com/ylefay/unbiased_mcmc_with_couplings)
  - `cjm96/mcmc_infinity`: a Python perfect-sampling (coupling-from-the-past) package for astronomy, MIT licence, no paper linked; its coupling construction was not checked — [GitHub](https://github.com/cjm96/mcmc_infinity)
- Coupling-based convergence diagnostics continue to develop: Corenflos & Dau (arXiv 2510.07559, v3 June 2026) give f-divergence upper bounds (KL, χ², Hellinger, TV) from coupling-based "weight harmonization" of interacting chains, more conservative than coupling-based TV bounds — [arXiv](https://arxiv.org/abs/2510.07559)

### Inferences
- **Priority order for `telescopic`, by published evidence and black-box fit:**
  1. Coupled RWMH with reflection-maximal coupling and a pilot-estimated proposal covariance (covered by the MH researcher).
  2. Coupled MALA with reflection-maximal coupling.
  3. Heng–Jacob coupled HMC with the RWMH mixture, plus the optional Xu et al. maximal index coupling for multinomial HMC.
  4. Gibbs with user-supplied conditionals and built-in maximal and CRN couplings for standard families.
  - Experiments: coupled ESS with pCN, coupled NUTS, re-tuning across replicate batches.
  - An emcee-style ensemble belongs only in the pilot stage.
- **A general lesson from the evidence:** a coupling meets well when, under CRN, the kernel's deterministic part shrinks differences (contraction) and some maximal-coupling component can turn small differences into equality. Kernels that move points along walker-difference directions or by pure translations lack the first property.

### Gaps
- Few papers report inefficiency against well-tuned baseline MCMC using comparable definitions. Comparisons across families in the table are therefore indicative only.
- I found no cross-family benchmark of coupled samplers on a common target suite.

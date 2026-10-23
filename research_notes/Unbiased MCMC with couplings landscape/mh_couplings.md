# Couplings of Metropolis–Hastings-family kernels and their maximal-coupling primitives

Scope: implementation-ready catalogue for `telescopic` v0.1 (pure Python, NumPy/SciPy, gradient-free default kernels, published methods only). Literature checked through October 2026. Status labels used below: **[peer-reviewed]**, **[preprint]**, **[own check]** (small NumPy simulations written for these notes; not peer-reviewed; scripts at `/tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/mh/scripts/check_couplings.py` and `.../check_pc_coupling.py`).

Key bibliographic anchors (verified via Crossref or the PDFs):
- Jacob, O'Leary & Atchadé (2020), "Unbiased Markov chain Monte Carlo methods with couplings", JRSSB 82(3):543–600, doi 10.1111/rssb.12336 [peer-reviewed, discussion paper] — [arXiv 1708.03625](https://arxiv.org/abs/1708.03625). Abbreviated **JOA20** below.
- O'Leary, Wang & Jacob (2021), "Maximal couplings of the Metropolis–Hastings algorithm", AISTATS 2021, PMLR 130 [peer-reviewed; arXiv lists O'Leary and Wang as equal-contribution first authors, PMLR indexes it as Wang et al.] — [arXiv 2010.08573](https://arxiv.org/abs/2010.08573), [PMLR](https://proceedings.mlr.press/v130/wang21d.html). Abbreviated **OWJ21**.
- O'Leary & Wang (2024), "Metropolis–Hastings transition kernel couplings", Ann. Inst. H. Poincaré Probab. Statist. 60(2):1101–1124, doi 10.1214/22-AIHP1360 [peer-reviewed] — [arXiv 2102.00366](https://arxiv.org/abs/2102.00366). Abbreviated **OW24**.
- O'Leary (2021), "Couplings of the Random-Walk Metropolis algorithm" [preprint, no journal version found] — [arXiv 2102.01790](https://arxiv.org/abs/2102.01790). Abbreviated **O21**.
- Papp & Sherlock, "Scalable couplings for the random walk Metropolis algorithm", JRSSB 87(3):772–795 (online 2024, issue 2025), doi 10.1093/jrsssb/qkae113 [peer-reviewed]; arXiv v1 title was "A new and asymptotically optimally contracting coupling for the random walk Metropolis" — [arXiv 2211.12585](https://arxiv.org/abs/2211.12585), [ar5iv v1 title](https://ar5iv.labs.arxiv.org/html/2211.12585). Abbreviated **PS25**.
- Corenflos & Särkkä (2022), "The coupled rejection sampler" [preprint; no journal version found] — [arXiv 2201.09585](https://arxiv.org/abs/2201.09585). Abbreviated **CS22**.
- Bou-Rabee, Eberle & Zimmer (2020), "Coupling and convergence for Hamiltonian Monte Carlo", Ann. Appl. Probab. 30(3):1209–1250, doi 10.1214/19-AAP1528 [peer-reviewed] — [arXiv 1805.00452](https://arxiv.org/abs/1805.00452).
- Heng & Jacob (2019), "Unbiased Hamiltonian Monte Carlo with couplings", Biometrika 106(2):287–302 [peer-reviewed] — [arXiv 1709.00404](https://arxiv.org/abs/1709.00404).
- Atchadé & Jacob (2024), "Unbiased Markov chain Monte Carlo: what, why, and how", Handbook of MCMC 2nd ed. chapter [preprint/book chapter] — [arXiv 2406.06851](https://arxiv.org/abs/2406.06851). Abbreviated **AJ24**.
- O'Leary's thesis: "Coupling and Parallelization in Statistical Inference", Harvard, 2021 — [Harvard DASH](https://dash.harvard.edu/entities/publication/84cb2a5e-4477-4893-9f45-129c37468368) (not read in full; its MH content appears to be OWJ21/OW24/O21).

---

## 1. Maximal couplings of two distributions p and q (the primitives)

### Takeaway
Three exact primitives cover what v0.1 needs. First, the rejection "γ-coupling" (Thorisson/Johnson; JOA20 Algorithm 2) works for any pair with samplers and normalised log-densities: its expected cost is exactly two draws, but the cost variance is 2(1−TV)/TV, which blows up as the two distributions approach each other. The Gerber–Lee η<1 variant caps that variance by giving up some meeting probability. Second, the reflection-maximal coupling of N(μ1,Σ) and N(μ2,Σ), or of any common spherically symmetric base, is maximal, has deterministic cost and keeps a reflection structure off the diagonal; it is the right default for equal-covariance Gaussian proposals. Third, an O(K) overlap construction gives the maximal coupling of categorical distributions. Gaussians with unequal covariances need the rejection coupling, its η-variant, or the coupled rejection sampler, which has bounded runtime variance and is sub-maximal; it is still a preprint.

### Cited Findings

#### Definitions and the coupling inequality
- A coupling of p and q is maximal if it maximises P(X=Y) subject to X∼p, Y∼q; by the coupling inequality P(X=Y) ≤ 1 − d_TV(p,q), with d_TV = ½∫|p−q| = 1 − ∫min(p,q) — [JOA20 §4.1](https://arxiv.org/abs/1708.03625); [OWJ21 §2](https://arxiv.org/abs/2010.08573).
- All maximal couplings share the same law on the diagonal {X=Y}: P(x′∈A, x′=y′) = ∫_A q(x,z)∧q(y,z) dz, and residual laws P(x′∈A, x′≠y′) = ∫_A (q(x,z)−q(y,z))₊ dz (and symmetrically for y′). Maximal couplings therefore differ only in how the residuals are jointly coupled — [O21 Lemma 3.1](https://arxiv.org/abs/2102.01790).
- If f is a homeomorphism, (x′,y′) is a maximal coupling of μ,ν iff (f(x′),f(y′)) is a maximal coupling of f#μ, f#ν; e.g. couple N(x,Σ),N(y,Σ) by whitening to N(0,I), N(Σ^{-1/2}(y−x),I) — [O21 Lemma 3.4](https://arxiv.org/abs/2102.01790).

#### Primitive A — rejection maximal coupling ("γ-coupling", Thorisson; JOA20 Algorithm 2)
- Origin: JOA20 cite Thorisson (2000, Ch. 1 §4.5) and Johnson (1998), "where it is termed γ-coupling" — [JOA20 §4.1](https://arxiv.org/abs/1708.03625); O21 attributes the coupling to Vaserstein (1969) and the name γ-coupling to Lindvall (1992) and calls it the unique maximal coupling with independent residuals — [O21 §4.1](https://arxiv.org/abs/2102.01790).
- Exact algorithm — [JOA20 Algorithm 2](https://arxiv.org/abs/1708.03625):
  ```
  1. Sample X ~ p and W | X ~ U([0, p(X)]). If W <= q(X): output (X, X).
  2. Otherwise repeat: sample Y* ~ q and W* | Y* ~ U([0, q(Y*)]) until W* > p(Y*); output (X, Y*).
  ```
  In log space (as in the R reference code): accept-as-meeting if `log p(X) + log U <= log q(X)`; in the loop accept Y* if `log q(Y*) + log U* > log p(Y*)` — [unbiasedmcmc R/get_max_coupling.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/get_max_coupling.R).
- Validity and maximality: P(X=Y) = P(step 1) = ∫min(p,q) = 1 − d_TV; step 2 is a rejection sampler from q targeting the residual (q − min(p,q))/(1 − ∫min(p,q)) — [JOA20 §4.1](https://arxiv.org/abs/1708.03625).
- Cost: step 1 = one draw from p plus one evaluation each of p and q; each step-2 attempt = one draw from q plus evaluations of p and q ("one unit"); the expected cost is two units for every p,q, "while the variance of the cost depends on d_TV(p,q), and in fact goes to infinity as this distance goes to zero" — [JOA20 §4.1](https://arxiv.org/abs/1708.03625). Conditionally on {X≠Y}, X and Y are independent — [JOA20 §4.1](https://arxiv.org/abs/1708.03625).
- **[own check]** Number of draws N = 1 + B·G with B∼Bern(TV), G∼Geometric(TV), so E[N]=2 and Var[N] = 2(1−TV)/TV (derivation for these notes). Simulated with N(0,1) vs N(δ,1), 400k calls: δ=2 (TV .683): Var 0.93 (theory 0.93); δ=0.5: 8.14 (8.13); δ=0.1: 50.4 (48.2); δ=0.02 (TV .008): 252 (249), max draws in one call 1075 — [check_couplings.py, check B](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/mh/scripts/check_couplings.py).
- Gerber & Lee (2020, JRSSB discussion of JOA20, 82(3):584–585) bounded-variance modification, reproduced as AJ24 Algorithm 2 with parameter η∈(0,1] — [AJ24 §4.1](https://arxiv.org/abs/2406.06851):
  ```
  1. Sample X ~ p.  2. Sample W ~ U(0,1).
     (a) If W <= min(eta, q(X)/p(X)): set Y = X.
     (b) Otherwise sample Y* ~ q, W* ~ U(0,1) until W* > eta * p(Y*)/q(Y*); set Y = Y*.
  3. Return (X, Y).
  ```
  AJ24: "The probability of {X = Y} is maximal only when η = 1 ... With η < 1, the coupling is sub-maximal, but the variance of the cost is upper bounded uniformly over p and q" — [AJ24 §4.1](https://arxiv.org/abs/2406.06851). The same algorithm is reproduced (parameter C) in [CS22 Appendix I](https://arxiv.org/abs/2201.09585) and implemented as `modified_thorisson(key, p, q, log_p, log_q, C)` in [coupled_rejection_sampling/thorisson.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/coupled_rejection_sampling/thorisson.py).
- **[own check]** As written, the η-algorithm meets with probability c_η = ∫min(ηp, q) (≥ η(1−TV)), not η∫min(p,q) as stated in CS22 Appendix I. Its draw count has E[N]=2 and Var[N] = 2c_η/(1−c_η) ≤ 2η/(1−η) (derivation for these notes). Simulation agrees: η=0.9, δ=0.02: P(meet) 0.9004 (theory ∫min(ηp,q)=0.9000; η∫min(p,q) would be 0.893), Var 18.1 (theory 18.0), max draws 97 vs 1075 at η=1 — [check_couplings.py, check B](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/mh/scripts/check_couplings.py).
- Mixture representation with deterministic cost (AJ24 Algorithm 3): with probability c = ∫min(p,q) draw X∼min(p,q)/c and set Y=X; otherwise draw X∼(p−min)/(1−c), Y∼(q−min)/(1−c) independently. It applies only when c is computable and the residuals can be sampled — [AJ24 §4.1](https://arxiv.org/abs/2406.06851).

#### Primitive B — reflection-maximal coupling (equal covariance / common spherical base)
- Presented in JOA20 as "a maximal coupling presented in Bou-Rabee et al. [2018]". It applies when X = μ1 + Σ^{1/2}Ẋ and Y = μ2 + Σ^{1/2}Ẏ, with Ẋ,Ẏ marginally ∼ s, a spherically symmetric density — [JOA20 §4.1](https://arxiv.org/abs/1708.03625). O21 traces it to HMC/Langevin analysis (Bou-Rabee et al. 2020; Eberle et al. 2019) and to Lindvall & Rogers (1986) reflection couplings of diffusions — [OWJ21 §2.2](https://arxiv.org/abs/2010.08573).
- Exact algorithm — [JOA20 §4.1](https://arxiv.org/abs/1708.03625) (identical in [AJ24 Algorithm 4](https://arxiv.org/abs/2406.06851) and [CS22 Appendix D Algorithm 3](https://arxiv.org/abs/2201.09585)):
  ```
  z = Σ^{-1/2}(μ1 − μ2);  e = z/‖z‖
  draw Ẋ ~ s, U ~ U(0,1)
  if U <= min(1, s(Ẋ + z)/s(Ẋ)):  Ẏ = Ẋ + z          # => Y = X exactly (meeting)
  else:                           Ẏ = Ẋ − 2(eᵀẊ)e     # reflection through hyperplane ⟂ e
  return X = μ1 + Σ^{1/2}Ẋ,  Y = μ2 + Σ^{1/2}Ẏ
  ```
  JOA20 prove Ẏ∼s using s(w − 2(eᵀw)e) = s(w) and s(w − 2(eᵀw)e + z) = s(w − z), and show P(X≠Y) equals d_TV(p,q). They note that "the computational cost associated with the above sampling technique is deterministic, in contrast with the cost of Algorithm 2" — [JOA20 §4.1](https://arxiv.org/abs/1708.03625).
- For Gaussians the closed-form log-ratio is log s(Ẋ+z) − log s(Ẋ) = −½[(eᵀẊ + ‖z‖)² − (eᵀẊ)²]. This is how [unbiasedmcmc src/mvnorm.cpp](https://github.com/pierrejacob/unbiasedmcmc/blob/master/src/mvnorm.cpp) (`rmvnorm_reflection_max_coupling_`) and [rwmcouplings src/svm.cpp](https://github.com/tamaspapp/rwmcouplings/blob/master/src/svm.cpp) compute it (the latter as `-0.5*(x_dot+z).squaredNorm() + 0.5*x_dot.squaredNorm()`).
- Meeting probability for N(x,σ²I_d) vs N(y,σ²I_d) under any maximal coupling: P(x′=y′) = P(χ²₁ ≥ ‖y−x‖²/(4σ²)) = 2Φ(−‖y−x‖/(2σ)) — [O21 Lemma 3.2](https://arxiv.org/abs/2102.01790); same formula in [Heng & Jacob eq. (10)](https://arxiv.org/abs/1709.00404) and [PS25 eq. (3)](https://arxiv.org/abs/2211.12585), which add the Chernoff bound ≤ 2 exp(−d‖x−y‖²/(8ℓ²)) for σ = ℓ/√d.
- **[own check]** d=3 with a random dense Σ: empirical P(X=Y) 0.9067 vs theory 2Φ(−‖Δ‖/2) = 0.9065; Y's mean and covariance match N(μ2,Σ) — [check_couplings.py, check A](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/mh/scripts/check_couplings.py).
- Other maximal Gaussian couplings with deterministic cost, from O21: a 1-D decomposition of the maximal independent coupling (Algorithm 2; the rejection loop then runs on scalars, so each loop step is O(1) instead of O(d) per density evaluation); the "maximal semi-independent" coupling (Algorithm 3: shared orthogonal component); and the "maximal optimal-transport" coupling (Algorithms 4–5: the 1-D residuals are coupled by the monotone map t_xy = Φ_y^{-1}∘Φ_x, which has closed-form residual CDFs) — [O21 §4.1–4.3](https://arxiv.org/abs/2102.01790).

#### Primitive C — maximal coupling of categorical / discrete distributions
- For q=(q_1..q_N), q̃=(q̃_1..q̃_N): α = Σ_n min(q_n,q̃_n), c_n = min(q_n,q̃_n)/α, residuals q′_n=(q_n−αc_n)/(1−α), q̃′_n=(q̃_n−αc_n)/(1−α). With probability α draw i∼c and output (i,i); otherwise draw i∼q′ and ĩ∼q̃′ independently. Then P(i=ĩ)=α=1−d_TV, the cost is O(N) and deterministic — [JOA20 §5.4](https://arxiv.org/abs/1708.03625).
- Log-space JAX implementation (logsumexp overlap, `logsubexp` residuals, `jax.lax.cond` branch): `_index_max_coupling` in [coupled_rejection_sampler.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/coupled_rejection_sampling/coupled_rejection_sampler.py).
- Example use: index selection in a variable-selection MH kernel ("maximal coupling of the discrete uniform distributions on {j: γ_j = 0} and {j: γ̃_j = 0}", followed by a common uniform for acceptance) — [JOA20 §5.4](https://arxiv.org/abs/1708.03625).

#### Gaussians with unequal covariances
- "When Σp ≠ Σq ... one can use Thorisson's algorithm, or its finite variance version (Gerber and Lee, 2021) in order to sample from a (sub-)maximal coupling"; CS22 also report that with MCMC "Thorisson's algorithm does not result in strong contractive properties (Wang et al., 2021)" — [CS22 §4](https://arxiv.org/abs/2201.09585).
- Coupled rejection sampler, CS22 Algorithm 1:
  ```
  given a coupling Γ̂ of dominating densities (p̂, q̂) with p <= M_p p̂ and q <= M_q q̂
  A_X = A_Y = 0
  while A_X == 0 and A_Y == 0:
      sample (X1, Y1) ~ Γ̂, U ~ U(0,1)
      if U < p(X1)/(M_p p̂(X1)): A_X = 1
      if U < q(Y1)/(M_q q̂(Y1)): A_Y = 1
  sample X2, Y2 from p ⊗ q
  return X = A_X X1 + (1−A_X) X2,  Y = A_Y Y1 + (1−A_Y) Y2
  ```
  Proposition 1: marginals are exact, E[τ] ≤ min(M_p, M_q) and V[τ] ≤ min(M_p,M_q)² − min(M_p,M_q) (τ = number of Γ̂ draws). If Γ̂ is a diagonal coupling, the output is diagonal (P(X=Y)>0) — [CS22 §2](https://arxiv.org/abs/2201.09585). Abstract: "Contrary to existing acceptance-rejection coupling methods, the variance of the execution time ... stays finite as the two target distributions converge" — [CS22](https://arxiv.org/abs/2201.09585).
- Gaussian specialisation (CS22 Proposition 5): use the reflection-maximal coupling of N(μp,Σ̂), N(μq,Σ̂) as Γ̂, with Σ̂^{-1} ⪯ Σp^{-1} and Σ̂^{-1} ⪯ Σq^{-1} (Loewner order). Then M(p,p̂) = det(2πΣ̂)^{1/2}/det(2πΣp)^{1/2}, and similarly for q. Σ̂ = max(spectra)·I is always valid, and when Σp = Σq the method reduces to reflection-maximal (Remark 3) — [CS22 §4](https://arxiv.org/abs/2201.09585). A closed-form Σ̂ is implemented in `get_optimal_covariance` and the full sampler in `coupled_mvns` — [mvn.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/coupled_rejection_sampling/mvn.py).
- Ensemble version (CS22 Algorithm 2): N proposals from Γ̂ per round, indices selected by a maximal coupling of the categorical weights; it "asymptotically recovers a maximal coupling" as N grows (Proposition 4) — [CS22 §3](https://arxiv.org/abs/2201.09585).
- CS22 list as open a reflection-maximal analogue for Gaussians with different means and covariances ("if an extension of the reflection-maximal coupling ... to Gaussians with different means and covariances was derived, it could be used as a proposal in our algorithm") — [CS22 §6](https://arxiv.org/abs/2201.09585).
- Contractive but non-meeting alternative: common random numbers (synchronous coupling), e.g. X = F_p^{-1}(U), Y = F_q^{-1}(U) in 1-D, which minimises E[(X−Y)²] — [AJ24 §4.1](https://arxiv.org/abs/2406.06851). For N(x,σ²I) vs N(y,σ²I), the synchronous coupling ξ=η minimises E‖x′−y′‖² — [O21 §4](https://arxiv.org/abs/2102.01790).

#### What a "distribution object" must provide (as used by existing code)
- Rejection coupling: samplers and log-pdfs for both distributions — `get_max_coupling(rp, dp, rq, dq)` with `dp`, `dq` returning log-pdfs — [unbiasedmcmc R/get_max_coupling.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/get_max_coupling.R); `maximal_coupling(rv1, rv2)` uses scipy-style `.rvs()` and `.pdf()` — [ColCarroll couplings/maximal_couplings.py](https://github.com/ColCarroll/couplings/blob/master/couplings/maximal_couplings.py).
- Reflection-maximal: the two means plus a shared Cholesky factor and its inverse — `rmvnorm_reflectionmax(mu1, mu2, Cholesky, Cholesky_inverse)` — [unbiasedmcmc R/mvnorm_couplings.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/mvnorm_couplings.R).
- Coupled rejection sampler: a dominating-coupling sampler `Gamma_hat(key, N)` returning pairs and coupled flags, marginal samplers `p`, `q`, log-densities `log_p_hat`, `log_q_hat`, `log_p`, `log_q`, and constants `log_M_p`, `log_M_q` — [coupled_rejection_sampler.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/coupled_rejection_sampling/coupled_rejection_sampler.py).

### Inferences
- Minimal `telescopic` distribution protocol: `sample(rng, size)`, `logpdf(x)` (vectorised over a leading batch axis) and an `is_same(other)` or parameter-equality test. Location–scale Gaussian families should also expose `loc` and a triangular `scale_tril`. The rejection coupling needs **normalised** densities, or at least the correct ratio p/q: when the two distributions differ in covariance, the log-determinant terms must be included. For equal covariances, constants cancel and the reflection-maximal coupling needs the base log-density only up to a constant.
- For v0.1's Gaussian random-walk and Langevin proposals, which share Σ across the two chains, reflection-maximal is the primitive of choice: it is maximal, uses a fixed number of random draws and FLOPs, and vectorises cleanly. The rejection γ-coupling should be the generic fallback, with the Gerber–Lee η exposed as an option (e.g. η = 0.9–0.99) when batches of pairs are processed in lock-step, because a vectorised batch waits for its slowest pair.
- Discrete coupling: implement the O(K) overlap algorithm in log space (as CS22's code does) to avoid underflow with peaked categorical weights.

### Gaps
- Gerber & Lee's discussion contribution (JRSSB 82(3):584–585) was not accessed directly; its algorithm is taken from the reproductions in AJ24 and CS22. The meeting-probability statement differs between CS22 App. I and the algorithm as written; my check supports ∫min(ηp,q).
- I did not read Bou-Rabee, Eberle & Zimmer (2020) in detail; the reflection-maximal algorithm and its attribution follow JOA20, O21 and CS22.
- No published exact maximal coupling with deterministic cost was found for Gaussians with unequal covariances.

---

## 2. Coupled MH transition kernels: proposal couplings × acceptance couplings, maximal and optimal-transport kernel couplings

### Takeaway
Every coupling of two MH kernels can be written as coupled proposals followed by coupled accept/reject indicators, provided each chain's conditional acceptance probability equals its own MH acceptance probability (OW24 Theorem 2.2, peer-reviewed). The workhorse is Johnson's P̄SQ: a maximal proposal coupling plus a common uniform. It preserves both marginals exactly, is faithful (sticky), and costs at most two target evaluations per step. It does not attain the kernel-level coupling bound. OWJ21's P̄C attains it at the same cost; the full-kernel couplings P̄MI and P̄MR also attain it, but at random cost with unbounded variance. Empirically, in moderate or high dimension, how much the chains contract when no meeting is proposed matters more than one-step maximality. The common uniform (V=U) beats independent or antithetic acceptance coupling. Reflection residuals beat independent ones.

### Cited Findings

#### Setting
- MH-like kernel: proposal x′∼Q(x,·) with density q, acceptance a(x,x′) = 1 ∧ π(x′)q(x′,x)/(π(x)q(x,x′)); off-diagonal transition density f(x,x′) = q(x,x′)a(x,x′); rejection mass r(x) = 1 − ∫f(x,x′)dx′ — [OWJ21 §2.1](https://arxiv.org/abs/2010.08573).
- Lemma 1: for x≠y, ‖P(x,·)−P(y,·)‖_TV = 1 − ∫ f(x,z)∧f(y,z) dz. P̄SQ meets with probability ∫ (q(x,z)∧q(y,z))(a(x,z)∧a(y,z)) dz, which is never larger than the bound ∫ f(x,z)∧f(y,z) dz. The two coincide "when either q(x,z) or a(x,z) does not depend on x, e.g. for the independence sampler" — [OWJ21 §2.2](https://arxiv.org/abs/2010.08573).

#### Characterisation (OW24)
- Theorem 2.2: P̄ ∈ Γ(P,P) if and only if it is generated by some Q̄ ∈ Γ(Q,Q) and an acceptance-indicator coupling B̄ with P(b_x=1 | x,y,x′) = a(x,x′) for Q(x,·)-a.e. x′ and P(b_y=1 | x,y,y′) = a(y,y′) for Q(y,·)-a.e. y′ — [OW24 §2](https://arxiv.org/abs/2102.00366).
- Theorem 3.4 characterises the maximal kernel couplings. "Any proposal coupling can give rise to a maximal coupling, as long as it satisfies the conditions of the theorem for some acceptance indicator coupling", and a three-point example shows "some maximal kernel couplings can only be generated from non-maximal proposal couplings" — [OW24 §3, App. A](https://arxiv.org/abs/2102.00366).
- OW24 provide no implementable optimal-transport (OT) kernel coupling. Their discussion lists as future work that "it may be possible to characterize the optimal transport couplings in Γ(P,P) in terms of the properties of proposal and acceptance indicator couplings"; they also say their results "support the use of numerical optimization" over parameterised proposal and acceptance couplings — [OW24 §1, §5](https://arxiv.org/abs/2102.00366).

#### P̄SQ (status quo, Johnson 1998; JOA20 §2.2/§4.2)
```
1. (X*, Y*) ~ maximal coupling of q(X_t, ·) and q(Y_{t-1}, ·)
2. U ~ U(0,1)                                  # common uniform
3. X_{t+1} = X* if U <= a(X_t, X*) else X_t
4. Y_t     = Y* if U <= a(Y_{t-1}, Y*) else Y_{t-1}
```
- JOA20: "The chains will stay together after they meet: at each step after meeting, the proposals will be identical with probability one, and jointly accepted or rejected with a common uniform variable. This coupling requires neither explicit minorization conditions nor contractive properties" — [JOA20 §2.2](https://arxiv.org/abs/1708.03625).
- The common uniform is the unique maximal coupling of Bern(a(x,x′)) and Bern(a(y,y′)), giving P(both accept) = a(x,x′)∧a(y,y′) — [OWJ21 §2.2](https://arxiv.org/abs/2010.08573). Any Bernoulli coupling with the MH margins is parameterised by ρ = P(b_x=b_y=1) ∈ [(a_x+a_y−1)∨0, a_x∧a_y] (Lemma 5.1). V=U attains the maximum and V=1−U the minimum (Lemma 5.2) — [O21 §5](https://arxiv.org/abs/2102.01790).
- "Optimal transport" acceptance coupling: choose ρ to minimise E‖Y−X‖² given (x,y,x′,y′). This is a linear programme in ρ, so the solution is almost surely one of the two endpoints (V=U or V=1−U) — [O21 §5](https://arxiv.org/abs/2102.01790). With maximal-reflection proposals it "coincides with the U = V coupling in all 1000 replications when d > 1" (≈96.6% when d=1). With maximal-independent proposals it picks V=1−U in a fraction of steps that approaches 50% as d grows, yet gives "almost indistinguishable" meeting times — [O21 §6.2](https://arxiv.org/abs/2102.01790).

#### Full-kernel maximal couplings (OWJ21)
- P̄MI (Algorithm 3), maximal with independent residuals:
  ```
  1. Draw X ~ P(x,·) (one MH step) and U ~ U(0,1)
  2. If X != x and U f(x,X) <= f(y,X): set Y = X
  3. Else repeat: draw ỹ ~ P(y,·), V ~ U(0,1);
         if ỹ == y: Y = ỹ, stop;  if ỹ != y and V f(y,ỹ) > f(x,ỹ): Y = ỹ, stop
  ```
  Cost: "the expected number of draws from P is 2 and the expected number of evaluations is 4−2r(x)−2r(y)", i.e. "2 draws from q and 6 − 2r(x) − 2r(y) evaluations in expectation". "The variance of the computing cost ... goes to infinity as ‖P(x,·) − P(y,·)‖_TV goes to zero" — [OWJ21 App. A.1](https://arxiv.org/abs/2010.08573).
- P̄MR (Algorithm 4), maximal with reflection residuals. It first tries the reflected point T_xy(X) = y + (I − 2eeᵀ)(X − x) with e = (y−x)/‖y−x‖, accepting it as Y if V·f^r_xy(X) ≤ f̃^r_yx(T_xy(X)); otherwise it falls back to rejection sampling from P(y,·) against a residual f̃^t_yx. Expected evaluations are 7 − 4r(x) − 2r(y) − 3p_c, and the variance is again unbounded as TV→0 — [OWJ21 §3.2, App. A.2](https://arxiv.org/abs/2010.08573). A log-space JAX implementation is in [reflected_coupled_mh.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/coupled_rejection_sampling/reflected_coupled_mh.py).

#### P̄C — maximal kernel coupling built from any proposal coupling (OWJ21 Algorithm 5)
```
q^m(z)   = q(x,z) ∧ q(y,z)                 (diagonal density of a maximal Q̄)
b_xy(z)  = 1 ∧ f(x,z)/q^m(z)               (acceptance when x' = y' = z)
c_xy(x') = (f(x,x') − q^m(x'))_+ / (q(x,x') − q^m(x'))   (acceptance when x' != y'; := 1 if denominator 0)
1. (x', y') ~ Q̄((x,y),·)
2. if x' == y': (B_x, B_y) ~ maximal coupling of Bern(b_xy(x')), Bern(b_yx(y'))   [common uniform]
   else:        (B_x, B_y) ~ any coupling of Bern(c_xy(x')), Bern(c_yx(y'))       [paper: maximal works well]
3. X = B_x x' + (1−B_x) x;  Y = B_y y' + (1−B_y) y
```
- Proposition 1: for any Q̄ ∈ Γ(Q,Q) the output is a coupling of P(x,·), P(y,·); if Q̄ is maximal then P̄C is maximal. Meetings are accepted at a higher-than-MH rate and non-meetings at a compensating lower rate. P̄C "avoids the 'while' loops of Step 3 of Algorithms 3 and 4" — [OWJ21 §3.3](https://arxiv.org/abs/2010.08573).
- Biased random walk example (Expo(1) target, N(z+3, 3) proposals, 10,000 reps), Table 1: P̄SQ+Q̄MI 74.0 (s.e. 0.94), P̄SQ+Q̄MR 75.6 (0.99), P̄MI 60.5, P̄MR 60.9, P̄C+Q̄MI 61.3 (0.87), P̄C+Q̄MR 62.2 (0.89). Maximal couplings help most "when low acceptance probabilities are typical" — [OWJ21 §4.1](https://arxiv.org/abs/2010.08573).
- N(0,I_d), σ²=2.38²/d, d=1..10 (Figure 7): couplings using Q̄MI "seem to yield meeting times that increase exponentially in dimension"; "P̄SQ and P̄C with Q̄MR deliver the best performance"; "the Q̄MR version of P̄SQ may outperform its P̄C counterpart" in higher dimensions; P̄MR is "somewhat better" than the Q̄MI-based options — [OWJ21 §4.2](https://arxiv.org/abs/2010.08573).
- **Reported bug (not confirmed by an OWJ21 erratum that I could find):** CS22 footnote 2 says OWJ21's empirical evaluation "suffers from an implementation bug, and that, contrarily to what was reported in the article, Algorithm 4 [P̄MR] results in faster coupling times than Algorithm 5 [P̄C] in high dimensional problems" — [CS22 §5.4](https://arxiv.org/abs/2201.09585). CS22's test script states: "As discussed with authors, there was a mistake in their code so that P_{MR} works better than P_C" — [tests/replicate_max_mh_coupling.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/tests/replicate_max_mh_coupling.py).
- **[own check]** My vectorised P̄C reproduces OWJ21 Table 1: P̄SQ+Q̄MI 73.2 (0.93), P̄C+Q̄MI 60.6 (0.87), P̄SQ+Q̄MR 74.7 (0.99), P̄C+Q̄MR 61.4 (0.91). On N(0,I_d) with Q̄MR (2,000 reps), mean τ for P̄SQ vs P̄C is: d=1: 2.9 vs 2.6; d=5: 12.6 vs 12.0; d=10: 31.3 vs 33.0; d=20: 76.3 vs 87.0; d=50: 259.5 vs 309.7. P̄SQ is better from d≈10 upward, consistent with OWJ21's remark — [check_pc_coupling.py](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/mh/scripts/check_pc_coupling.py). I did not test P̄MR.

#### Proposal and acceptance coupling grid for RWM (O21, N(0,I_10), ℓ=2.38, iid target inits, 1000 reps), Table 1, mean τ (s.e.)
| proposal \ acceptance | V=U | independent | V=1−U |
|---|---|---|---|
| maximal reflection | 30 (0.8) | 51 (1.4) | 68 (2.0) |
| maximal semi-independent | 54 (1.5) | 85 (2.4) | 105 (3.3) |
| maximal optimal transport | 104 (3.0) | 155 (4.6) | 183 (5.7) |
| maximal independent | 279 (8.5) | 302 (9.4) | 354 (11.2) |

Source: [O21 Table 1](https://arxiv.org/abs/2102.01790). O21: these orderings are "robust and monotone" across dimensions, initialisations and proposal variances. The OT-residual coupling "seems to produce the least contraction in spite of producing the smallest values of ‖x′ − y′‖", and the reflection coupling "the most contraction despite maximizing the variance of y′₁ − x′₁", which O21 attribute to "the interaction of these couplings with the acceptance step" — [O21 §6.1](https://arxiv.org/abs/2102.01790).
- **[own check]** At d=10 my implementation gives 32.3 (s.e. 0.8) for reflection-maximal and 274.2 (8.4) for maximal independent, matching the table — [check_couplings.py, check C](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/mh/scripts/check_couplings.py).
- The independent, synchronous (CRN), reflection and full-reflection proposal couplings all have the property that "if x ≠ y then x′ ≠ y′ almost surely", so "exclusive reliance on these couplings cannot yield P(τ < ∞) = 1 unless X0 = Y0" — [O21 §4](https://arxiv.org/abs/2102.01790).
- Stickiness: "the couplings we consider can always be made sticky by requiring x′ = y′ ∼ Q(x,·) = Q(y,·) and V = U ∼ Unif if x = y" — [O21 §2](https://arxiv.org/abs/2102.01790).

### Inferences
Comparison of MH kernel couplings (synthesised from the findings above; d-scaling is empirical unless stated):

| coupling | marginals exact | faithful | can meet | target evals / coupled step | RNG per step | d-scaling (RWM, iid N(0,I) inits) | status |
|---|---|---|---|---|---|---|---|
| P̄SQ + Q̄MI (γ-coupling) + common U | yes | yes | yes | 2 (1 if x′=y′); proposal densities in loop | random (loop) | exponential (JOA20, OWJ21, O21) | peer-reviewed |
| P̄SQ + Q̄MR (reflection-maximal) + common U | yes | yes | yes | 2 (1 if x′=y′) | fixed: d normals + 2 uniforms | ≈linear to d≈50–100 (JOA20, O21); ~d^1.3 to d=400 [own check] | peer-reviewed |
| P̄SQ + maximal semi-indep / OT residuals | yes | yes | yes | 2 | semi-indep: random (1-D rejection loop); OT: fixed (1-D closed-form transport) | between the two above (O21) | preprint |
| P̄C + any maximal Q̄ | yes | yes | yes (kernel-maximal) | 2 (+ proposal densities) | as Q̄ + 1 uniform | better at low acceptance; slightly worse than P̄SQ for d ≥ 20 with Q̄MR [own check] | peer-reviewed |
| P̄MI / P̄MR (full kernel) | yes | yes (needs care if x=y) | yes (kernel-maximal) | random; loop draws MH steps from P(y,·), each needing a target evaluation; unbounded variance | random | P̄MI exponential; P̄MR "somewhat better" (OWJ21) or better than P̄C (CS22 claim) | peer-reviewed / disputed |
| CRN, reflection, GCRN, GCRefl (no maximal part) | yes | yes | **no** (needs two-scale switch) | 2 (+2 gradients for GCRN/GCRefl) | fixed | contraction only (see §3) | peer-reviewed (PS25) |

- For `telescopic` v0.1, the default coupled RWM step should be P̄SQ with reflection-maximal proposals and a common uniform: published, deterministic cost, at most two target evaluations, faithful. P̄C is a cheap, published, optional upgrade for kernels with low acceptance or asymmetric proposals. P̄MI and P̄MR do not suit expensive black-box targets: each loop iteration needs a fresh target evaluation, and the runtime variance is unbounded.
- The "OT kernel coupling" in the brief is better described as O21's OT-residual proposal coupling plus an OT (LP-endpoint) acceptance coupling. Neither beat reflection-maximal with V=U in O21, and OW24 contain no OT algorithm.

### Gaps
- No proven meeting-time bounds as a function of d for any of these MH couplings. All d-scaling statements are empirical.
- The CS22 bug claim about OWJ21 Figure 7 (P̄MR vs P̄C) is unverified here: I found no erratum or arXiv v2 of OWJ21 (the PDF read is v1, 16 Oct 2020), and I did not implement P̄MR.
- O'Leary's thesis was not read; I assume its MH chapters match OWJ21/OW24/O21.

---

## 3. Random-walk Metropolis in higher dimension: GCRN, two-scale couplings, scaling with d, step size, preconditioning

### Takeaway
In high dimension the bottleneck is contraction, not the one-step meeting probability. At the mixing-optimal scale σ = ℓ/√d, a proposal meeting is only likely once ‖x−y‖ = O(d^{-1/2}), so a coupling that cannot contract the chains to O(1) squared distance is expected to give meeting times exponential in d (PS25 heuristic). The empirical picture is:
- Maximal-independent proposals: meeting times exponential in d.
- Reflection-maximal: roughly linear (my check: about d^1.3 up to d=400) on spherical or well-preconditioned targets, poor on eccentric ones.
- CRN: stalls at Θ(d) squared distance.

GCRN (PS25, JRSSB) is asymptotically optimally contractive and insensitive to eccentricity, but needs ∇log π at both states. The two-scale scheme switches to reflection-maximal below a squared-distance threshold δ, chosen by grid search. ℓ≈2.38 (23.4% acceptance) is close to optimal for GCRN contraction, though smaller steps may be needed to synchronise acceptances. Preconditioning is essential for reflection-type couplings on eccentric targets. PS25 state that they do not know how to get similarly scalable RWM couplings without gradients.

### Cited Findings

#### Why contraction matters
- "Two independent chains will typically start ‖X0 − Y0‖² = O(d) apart, yet the chance of coalescing in one iteration is infinitesimally small unless ‖Xt − Yt‖² = O(d^{-1}). If the coupling cannot contract the chains to within, say, O(1) squared distance, then one can expect meeting times to grow exponentially with the dimension d ... in stark contrast with the O(d) time required for RWM chains to converge" — [PS25 §3.1](https://arxiv.org/abs/2211.12585). This is a heuristic argument, not a theorem.
- Same point in O21: with σ_d² = ℓ²/d, "r must shrink at a rate 1/√d to maintain a constant probability of meeting proposals" — [O21 Lemma 3.2 and discussion](https://arxiv.org/abs/2102.01790).
- PS25's one-step decomposition: ‖X′−Y′‖² − ‖X−Y‖² = 2h(X−Y)ᵀ(Z_xB_x − Z_yB_y) + h²‖Z_xB_x − Z_yB_y‖². The first term contracts at about O(h²) = O(1/d) per step for log-concave targets. The second jumps by Θ(1) at every imbalanced acceptance (one chain accepts, the other rejects); "if imbalanced acceptance events occur with Θ(1) probability, then the equilibrium ... lies at ‖Xt − Yt‖² = Θ(d), so the coupling cannot scale" — [PS25 §6.1](https://arxiv.org/abs/2211.12585).
- Objective: minimising E‖X′−Y′‖² over Markovian couplings is equivalent to maximising the "expected jump concordance" E[h² Z_xᵀZ_y B_x B_y], the coupling analogue of the expected squared jumping distance (ESJD) — [PS25 §3.2](https://arxiv.org/abs/2211.12585).

#### Coupling definitions (proposals X′ = X + hZ_x, h = ℓ d^{-1/2}, common uniform U_y = U_x throughout)
- CRN: Z_y = Z_x. Reflection: Z_y = Z_x − 2(eᵀZ_x)e with e = Nor(X − Y). GCRN: Z_x = Z − (n_xᵀZ)n_x + Z_∇ n_x and Z_y = Z − (n_yᵀZ)n_y + Z_∇ n_y, with n_x = Nor(∇log π(X)), n_y = Nor(∇log π(Y)), Z∼N(0,I_d), Z_∇∼N(0,1) independent. "By convention, we default to the CRN coupling when a vector to be normalized is null" — [PS25 §3.3](https://arxiv.org/abs/2211.12585).
- Proposition 1: (n_xᵀZ_x, n_yᵀZ_y) is bivariate normal with correlation ρ_crn = n_xᵀn_y, ρ_refl = n_xᵀn_y − 2(n_xᵀe)(n_yᵀe), ρ_gcrn = 1. That is, GCRN exactly synchronises the gradient-direction components that dominate the log-acceptance ratio — [PS25 §3.3](https://arxiv.org/abs/2211.12585).
- GCRefl ("gradient-corrected reflection"): Z_x = Z − (e_xᵀZ)e_x + Z_∇ e_x and Z_y = Z − 2(eᵀZ)e − (e_yᵀZ)e_y + Z_∇ e_y, where e = Nor(X−Y) and e_x = Nor(n_x − (eᵀn_x)e) (similarly e_y). It defaults to reflection when a vector is null, which is always the case for d=1 — [PS25 §6.3](https://arxiv.org/abs/2211.12585). Reference implementation: `svmTwoScaleGCReflRWM` in [rwmcouplings src/svm.cpp](https://github.com/tamaspapp/rwmcouplings/blob/master/src/svm.cpp).
- Gradients are evaluated at the current states, i.e. two extra gradient evaluations per coupled step; the reference code recomputes a chain's normalised gradient only when that chain accepts ("Update gradient if accepted") — [rwmcouplings src/svm.cpp](https://github.com/tamaspapp/rwmcouplings/blob/master/src/svm.cpp).

#### Theory (proven, high-dimensional limits) vs empirical
- Theorem 1 (standard Gaussian target): the limiting supremum over product couplings of the contraction objective is ℓ²E[1 ∧ e^{ℓx^{1/2}Z−ℓ²/2} ∧ e^{ℓy^{1/2}Z−ℓ²/2}], attained by GCRN. Theorem 2: the scaled squared distance converges to an ODE solution. Theorem 4 (product targets π^(d)(x) = ∏ ω_i f(ω_i x_i), chains marginally stationary): sup over all Markovian couplings of lim E[h²Z_xᵀZ_yB_xB_y] = 2ℓ²Φ(−ℓ(bI)^{1/2}/2), attained by GCRN — [PS25 §4, §6.2](https://arxiv.org/abs/2211.12585) [proven].
- Fixed points of the limiting ODE (Propositions 4, 6): for spherical Gaussians, CRN stalls at s* > 0 (‖X−Y‖² = Θ(d) unless ℓ = o(1)), while reflection and GCRN reach s* = 0 (o(d)). For elliptical Gaussians (eccentricity ε>1), reflection and CRN have s* > 0, so ‖X−Y‖² = Θ(d) for large t and they "must sacrifice mixing for contraction by using an increasingly smaller scaling λ = o(1)". GCRN keeps s* = 0 — [PS25 §4.2.2, §5.2.2](https://arxiv.org/abs/2211.12585) [proven in the limit]. Prediction for large d,t: E‖X−Y‖² ≈ 2Tr(Σ)(1 − v*) (eq. 10) — [PS25 §5.2.2](https://arxiv.org/abs/2211.12585).
- Empirical, d ∈ [1, 10000]: GCRN "exhibited clear O(h²) behaviour in all dimensions; remarkably, for d ≥ 4 the coupling was always able to contract the chains to within numerical precision" (spherical; d ≥ 10 for the most eccentric elliptical target). Reflection "displayed O(1) behaviour ... could bring the chains sufficiently close together for a reasonable chance of coalescence when d ≤ 100, but ... coalescence would be unlikely in dimensions at least an order of magnitude larger" — [PS25 §4.2.2, §5.2.2](https://arxiv.org/abs/2211.12585) [empirical].
- PS25 caution: "Theorem 4 suggests good performance in finite dimensions, but it does not guarantee it"; GCRN "may perform poorly when the second-order terms in the acceptance ratio exhibit substantial variability ... when the precision matrix of the target has a few very large eigenvalues", where preconditioning or a smaller step may help — [PS25 §6.2](https://arxiv.org/abs/2211.12585).

#### Two-scale strategy and threshold choice
- "Once the chains become close enough to have a reasonable chance of coalescing in one step, we propose to swap from a contractive coupling like GCRN to one that allows for exact meetings ... our coalescive coupling of choice is the reflection-maximal coupling" — [PS25 §3.3](https://arxiv.org/abs/2211.12585). The rule: contractive coupling when ‖X_t − Y_t‖² ≥ δ, reflection-maximal otherwise. δ is selected "using a grid search on a logarithmic scale: the general trend is that the meeting times are insensitive to choosing δ smaller than optimal"; the optimal choice of δ is left open — [PS25 §7.1](https://arxiv.org/abs/2211.12585).
- Stochastic-volatility posterior (d=360): the grid was {1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1} with chosen (δ_gcrn, δ_gcrefl) = (0.1, 0.001). "Compared to GCRN, GCRefl prefers a smaller δ, is less sensitive to values smaller than optimal but is much more sensitive to values larger than optimal". With a maximal coupling at those thresholds the one-step proposal-coalescence probability would be {0.81, 0.45, 0.02, <1e-13, ...}, i.e. GCRN's best δ has a tiny meeting chance at the switch, which they read as reflection "contract[ing] the chains, on average, faster than exponential rate O(h²) of GCRN" — [PS25 App. C.3.1](https://arxiv.org/abs/2211.12585). On the same target, "the reflection-maximal coupling is severely hindered by the eccentricity of the target ... the probability of coalescing is effectively null at all iterations" — [PS25 §7.1](https://arxiv.org/abs/2211.12585).
- Earlier hybrid (O21 §4.4/§6.1): maximal reflection when r < r̄/√d, simple reflection otherwise. Meeting times scale like maximal reflection's, with "an inverse relationship between average meeting time and r̄" because of missed meeting opportunities — [O21](https://arxiv.org/abs/2102.01790). Two-scale or "mixed" couplings also appear in Bou-Rabee et al. (2020), Biswas et al. (2022), and Eberle (2016, "mixed couplings") — [PS25 §3.3](https://arxiv.org/abs/2211.12585); [AJ24 §4.2](https://arxiv.org/abs/2406.06851).

#### Scaling evidence for gradient-free couplings
- JOA20 (target N(0,V) with V inverse-Wishart, proposal Σ = V/d, i.e. ideal preconditioning). The maximal (γ-) coupling gives "an exponential increase of the average meeting times with the dimension" (d=1..5; log-scale axis up to 10⁴), "meeting times can be large even if the chains marginally start at stationarity". Reflection-maximal: "closer to linear in the dimension" (d up to 50); "A formal justification ... is an open question" — [JOA20 §4.2](https://arxiv.org/abs/1708.03625).
- O21: "meeting times grow exponentially in dimension under the maximal coupling with independent residuals, close to linearly in dimension under the maximal reflection coupling, and somewhere in between for the other two maximal couplings" — [O21 §6.1](https://arxiv.org/abs/2102.01790).
- **[own check]** N(0,I_d), ℓ=2.38, iid target initialisations, common uniform, mean τ (median):
  - Reflection-maximal: d=1: 2.8 (2), d=2: 4.8, d=5: 13.3, d=10: 32.3 (26), d=20: 77.9, d=50: 252.8 (218), d=100: 629 (558), d=200: 1450 (1216), d=400: 3565 (3143).
  - Maximal-independent: d=2: 5.0, d=4: 12.9, d=6: 35.4, d=8: 91.8, d=10: 274.
  - Two-scale GCRN → reflection-maximal, threshold δ = c·h², 200 pairs:
    | d | c=1 | c=10 |
    |---|---|---|
    | 10 | 50 | 40 |
    | 100 | 870 | 750 |
    | 400 | 4290 | 3890 |
    | 1000 | 11,670 | 10,870 |

  On this spherical target the two-scale GCRN is slightly slower than reflection-maximal up to d=400 but grows more slowly (≈d^1.1 vs ≈d^1.3) — [check_couplings.py, checks C and E](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/mh/scripts/check_couplings.py).

#### Step size: coupling vs mixing
- JOA20: larger proposal variance raises the chance of proposing identical values, "however meeting events only occur if proposals are accepted ... This trade-off could lead to a different choice of Σ than the optima known for the marginal chains ... and deserves further investigation" — [JOA20 §4.2](https://arxiv.org/abs/1708.03625).
- PS25: ℓ=2.38 is optimal for mixing at stationarity; ℓ=√2 is optimal for convergence at the mode ("at least 86% efficient over the considered range"), while 2.38 is at least 45% efficient for convergence. For GCRN contraction, "the step size ℓ = 2.38 is close to optimal over much of the range" and "contraction is insensitive to varying the step size near the optimum", "though we caution that in practice smaller scalings and higher acceptance rates may be required because it is crucial for a coupling to synchronize acceptance events". Reflection "requires a much smaller scaling ℓ = o_d(1) when the target is not spherically symmetric" — [PS25 §4.2.3, §5.2.3](https://arxiv.org/abs/2211.12585).
- Logistic regression (Sonar, d=61), comparing RWM with MALA: "there is a trade-off between coupling and mixing efficiency when using the diagonal preconditioner, whereas using the full preconditioner mitigates this issue" — [PS25 §7.4](https://arxiv.org/abs/2211.12585).
- For a dedicated meeting kernel (Heng & Jacob's mixture component), achieving P(X*=Y*) = θ requires σ ≈ δ/((2π)^{1/2}(1−θ)) with δ = ‖x−y‖. They use σ = 10⁻³ by default and suggest monitoring acceptance rates — [Heng & Jacob §4.1](https://arxiv.org/abs/1709.00404).

#### Preconditioning
- With proposals x + LZ_x, LLᵀ = Σ: GCRN uses n_x = Nor(Lᵀ∇log π(x)); reflection uses Z_y = Z_x − 2(eᵀZ_x)e with e = Nor(L^{-1}(x−y)); reflection-maximal uses z = L^{-1}(x−y) and couples the proposals with probability s(Z_x + z)/s(Z_x). GCRN also adapts to position-dependent L(x) — [PS25 App. B.2](https://arxiv.org/abs/2211.12585).
- Note on PS25 arXiv v3 App. B.2: it prints "sets Z_y = Z_x − z with probability s(x − z)/s(x)", which with z = L^{-1}(x−y) appears to be a sign typo. Their C++ uses z = (x−y)/h, the ratio exp(−½‖ẋ+z‖² + ½‖ẋ‖²), and sets y′ = x′ on success, i.e. Z_y = Z_x + z — [rwmcouplings src/svm.cpp](https://github.com/tamaspapp/rwmcouplings/blob/master/src/svm.cpp); consistent with [JOA20 §4.1](https://arxiv.org/abs/1708.03625).
- Logistic regression d=61: with a diagonal preconditioner "GCRefl was the only practical coupling" (the preconditioned target was highly skewed, with the largest eigenvalue nearly 30% of the trace). With a full (Cholesky of covariance) preconditioner "the reflection coupling was practical, but we found GCRefl to outperform it even after accounting for the additional gradient evaluations". Thresholds: δ_diag = 10h², δ_full = 1 in the preconditioned norm ‖P^{-1}(X−Y)‖² — [PS25 §7.4](https://arxiv.org/abs/2211.12585).

#### Gradient requirement
- "The utility of our proposed couplings of the RWM for unbiased MCMC is less clear. They demand access to a gradient oracle; it is unclear to us how one could devise similarly scalable couplings of the RWM algorithm without such information, yet access to this oracle also enables the use of more scalable gradient-based algorithms altogether, such as MALA and HMC" — [PS25 §8](https://arxiv.org/abs/2211.12585).
- Web searches run for these notes found no 2025–2026 follow-up by Papp & Sherlock on RWM couplings beyond the JRSSB version. The most recent MH-coupling preprint found, Phan, Flamich, Khisti & Asoodeh (May 2026), "Multi-Marginal Couplings for Metropolis–Hastings", targets grand couplings of many chains for convergence diagnostics; it reports "reducing meeting times by up to 50% compared with existing baselines" and is not a pairwise unbiased-MCMC method — [arXiv 2605.12807](https://arxiv.org/abs/2605.12807) [preprint].

### Inferences
- For a gradient-free v0.1, the best published option is reflection-maximal plus a common uniform, with a well-estimated proposal covariance (preconditioner from a pilot or warm-up run) and ℓ ≈ 2.38/√d. Expect meeting times roughly linear (mildly super-linear) in d for d up to a few hundred on targets that are well conditioned after preconditioning. Expect breakdown on strongly eccentric or poorly preconditioned targets, or at d ≳ 10³ (PS25's extrapolation for the reflection coupling). Users should get a meeting-time diagnostic and a warning when meeting times are large relative to the integrated autocorrelation time.
- GCRN and GCRefl two-scale couplings are published (JRSSB) and can be offered when the user supplies ∇log π.
- Unpublished observation (not for v0.1): GCRN's construction stays marginally valid for **any** measurable choice of unit vectors n_x, n_y, because replacing the component of Z along a unit vector with an independent N(0,1) leaves Z ∼ N(0,I). A "surrogate-gradient GCRN" (e.g. n_x from the gradient of a pilot Gaussian approximation, −Σ̂^{-1}(x−μ̂)) would therefore be exact but of unknown efficiency.
- The two-scale switch may depend on any function of the current pair (squared distance in the preconditioned metric) without breaking marginals, because both branches are couplings of the same kernel P. A default δ around h² to 10h² in the whitened metric (one-step proposal-meeting probability of order 0.1–0.5) gave similar results in my check. PS25's grid-search guidance suggests erring small.
- Step size: for coupled RWM there is no published evidence that the meeting-optimal ℓ differs much from 2.38 once the target is well preconditioned (GCRN theory; reflection is fine on spherical targets). On eccentric targets without a good preconditioner, reflection needs much smaller steps. The package should tune for the marginal acceptance rate (about 0.234) and fix preconditioning first.

### Gaps
- No theorem bounds meeting times (as opposed to contraction) as a function of d for any RWM coupling, including the two-scale GCRN; the reflection coupling's O(1) steady-state distance is empirical.
- No published gradient-free coupling with GCRN-like scalability.
- PS25 do not report meeting-time-versus-d curves for the two-scale couplings on simple targets; my own numbers above are the only ones I have.

---

## 4. Other MH-type kernels: independence MH, MALA, pCN, component-wise / MH-within-Gibbs, mixtures of kernels

### Takeaway
- **Independence MH:** a common proposal plus a common uniform is already kernel-maximal. For bounded importance weights it gives geometric meeting times; for coupled PIMH the meeting time is exactly geometric given the initial state.
- **MALA and pCN:** the proposals of the two chains share their covariance, so the reflection-maximal coupling applies directly. MALA has published coupled examples (with preconditioning) but no dimension-scaling theory for meeting times. For pCN, the published couplings are synchronous (contraction only) or a "maximal coupling of pCN kernels" used in a multilevel paper.
- **MH-within-Gibbs:** couple each conditional update (maximal or reflection-maximal proposals plus a common uniform). For conditionally independent blocks, a fully factorised coupling gives meeting times growing at most logarithmically in block size.
- **Mixtures:** combining a contracting kernel with a meeting kernel (Heng & Jacob: HMC plus 5% RWM with tiny steps) is valid only if both chains use the same component at each step. The choice of component must not depend on the chains' distance; contrast this with two-scale switching between couplings of the same kernel, which is legal.

### Cited Findings

#### Independence MH (IMH)
- OWJ21: P̄SQ's meeting probability equals the coupling bound when q(x,z) does not depend on x, "e.g. for the independence sampler" — [OWJ21 §2.2](https://arxiv.org/abs/2010.08573).
- JOA20: "Couplings of particle independent Metropolis–Hastings, which is a particular case of Metropolis–Hastings with an independent proposal distribution, are simpler to design" — [JOA20 §4.4](https://arxiv.org/abs/1708.03625).
- Coupled PIMH (Middleton, Deligiannidis, Doucet & Jacob, AISTATS 2019, PMLR 89:2378–2387): both chains share each proposal (X*, p*_N) and a common uniform; "the meeting time τ corresponds to the first time the proposal ... is accepted by both chains". Proposition 8: "τ | Z^(0) ∼ Geo(α(Z^(0))) and P[τ = 1] ≥ 1/2", with lim_{N→∞} P[τ=1] = 1 under bounded weights, where α(z) is the average acceptance probability from z. Validity uses uniform ergodicity under bounded incremental weights — [Middleton et al. 2019 §2.2–2.3](https://arxiv.org/abs/1902.01781); [PMLR](https://proceedings.mlr.press/v89/middleton19a.html).
- Agapiou, Roberts & Vollmer (Bernoulli 24(3):1726–1786, 2018) build coupled independence samplers from a minorisation implied by a lower bound α* on the acceptance probability. A shared "coin" U₁ chooses between a common draw ξ₁ (accepted by both) and a residual MH step with corrected acceptance, and the coupling is realised by using the same random seeds in both chains — [Agapiou et al. §4](https://arxiv.org/abs/1411.7713); journal reference as cited in [AJ24 references](https://arxiv.org/abs/2406.06851).

#### MALA (and other Langevin-type proposals)
- MALA proposals are N(X + h∇log π(X)/2, hΣ). "Maximal couplings or reflection-maximal couplings of the proposals could be readily implemented to obtain faithful chains" — [JOA20 §4.4](https://arxiv.org/abs/1708.03625). "Reflection couplings as in Figure 3 or Algorithm 4 can be directly used for Langevin Monte Carlo" — [AJ24 §4.3](https://arxiv.org/abs/2406.06851). Reflection-maximal "applies to any pair of Gaussians N(x,Σ) and N(y,Σ) with the same covariance matrix. We use it to couple MALA proposals" — [PS25 App. B.2](https://arxiv.org/abs/2211.12585).
- OW24 §4.2 give the two-step (proposal plus acceptance) representation of any MALA kernel coupling, with proposal N(x + τ∇log π(x), 2τI) — [OW24 §4.2](https://arxiv.org/abs/2102.00366).
- Empirical comparison, preconditioned RWM vs MALA on logistic regression (d=61), wall-time. Optimising for mixing/meeting, "the RWM is 2×/4× slower than MALA" with a diagonal preconditioner and "7×/13× slower" with a full one; MALA used a reflection-maximal coupling, RWM a two-scale GCRefl→reflection-maximal coupling. "We emphasize that both the RWM and MALA couplings require access to the gradient oracle"; MALA "is more sensitive to the eccentricity of the target than the RWM (Livingstone and Zanella, 2022)" — [PS25 §7.4](https://arxiv.org/abs/2211.12585).
- "For MALA, preliminary results (not included) suggest that, as for the RWM, higher-order derivative information than that available in the proposal should be incorporated in an asymptotically optimal coupling" — [PS25 §8](https://arxiv.org/abs/2211.12585).
- Position-dependent covariances (simplified manifold MALA): reflection-maximal "is impossible" to apply directly. Options are the modified Thorisson coupling, OWJ21's full-kernel P̄MI/P̄MR, or the coupled rejection sampler. On the heart-disease logistic regression, mean meeting times were 11.0 (N=1), 7.7 (N=4), 6.3 (N=16) and 5.8 (N=64) for the ensemble coupled rejection sampler vs 7.5 for P̄MR, and Thorisson "underperformed the other alternatives" — [CS22 §5.4](https://arxiv.org/abs/2201.09585) [preprint].

#### pCN (preconditioned Crank–Nicolson)
- pCN step: ξ∼μ₀ (Gaussian prior), X̂ = ρX + √(1−ρ²)ξ, accept with probability 1 ∧ exp(g(X) − g(X̂)) — [Agapiou et al. Algorithm 4](https://arxiv.org/abs/1411.7713).
- Coupled pCN in Agapiou et al. (Algorithm 5) draws the same ξ_k and U_k for both chains (a synchronous coupling, also across discretisation levels via a projection). It relies on contraction ("simulatable contracting coupling") within a randomised-truncation estimator rather than exact meetings — [Agapiou et al. §5](https://arxiv.org/abs/1411.7713). AJ24 note that contraction from synchronous couplings has been observed "for the preconditioned Crank–Nicholson algorithm in Agapiou et al. (2018)" — [AJ24 §4.2](https://arxiv.org/abs/2406.06851).
- Chada, Jasra, Maama & Tempone (2025, preprint) describe "synchronous pCN" (common Z∼N(0,I)) as a coupling of pCN kernels within a coupled multilevel scheme, and report experiments with a "maximal coupling of pCN kernels" (ρ = 0.95) — [arXiv 2502.03920 §3.4.3, §5](https://arxiv.org/abs/2502.03920).
- Theory relevant to preconditioned dynamics in high or infinite dimension: Bou-Rabee & Eberle, "Two-scale coupling for preconditioned Hamiltonian Monte Carlo in infinite dimensions", Stoch PDE: Anal. Comp. 9(1):207–242 (Crossref metadata; not read) — [doi 10.1007/s40072-020-00175-6](https://doi.org/10.1007/s40072-020-00175-6).

#### Component-wise / MH-within-Gibbs
- JOA20: "We can introduce couplings in each conditional update, using either maximal couplings of the target conditionals, if these are standard distributions, or maximal couplings of the proposal distributions in MH steps targeting the target conditionals" — [JOA20 §4.3](https://arxiv.org/abs/1708.03625).
- Ceriani, Pandolfi & Zanella (preprint, v2 March 2026), blocked Gibbs with conditionally independent blocks. Two-step blocked coupling (Algorithm 1): a W2-optimal (contractive) coupling of each block while ‖X−Y‖ > ε, maximal couplings of each block once close, and a fall-back to the contractive coupling for later blocks if an earlier block's maximal coupling failed. For MH-within-Gibbs, "leveraging conditional independence in the coupling of the MH kernels allows to have meeting times that grow at most logarithmically with I_k", and they suggest "a fully factorized strategy both on the proposal and acceptance". Their experiment used per-coordinate RWM with "maximal independent coupling, and paired acceptance" and a one-step (not two-step) strategy; they also note "implementing efficient contractive couplings of MH kernels is particularly challenging" — [arXiv 2410.08939 §3.4, §6](https://arxiv.org/abs/2410.08939).

#### Mixtures of kernels
- Heng & Jacob: K̄ = (1−γ)K̄_HMC + γK̄_σ, where K̄_σ is RWMH with maximally coupled N(x,σ²I), N(y,σ²I) proposals and a common uniform. HMC (common momenta) contracts and RWMH supplies exact meetings. "We advocate choosing only small values of γ ... most simulations ... will employ γ = 1/20 as the default value", with σ = 10⁻³ by default. Efficiency bound: v(h, K) ≤ γ(1−γ)^{-1}var_π h + (1−γ)^{-1}v(h, K_HMC) — [Heng & Jacob §4](https://arxiv.org/abs/1709.00404).
- AJ24: a coupling of the mixture "can be defined as a mixture of the coupled transitions, w₁P̄₁ + w₂P̄₂ ... Careful: it would not be legal to employ P̄₁ when the chains are distant and P̄₂ when they are close, as this would violate the marginal constraint that each chain evolves according to w₁P₁ + w₂P₂" — [AJ24 §4.2](https://arxiv.org/abs/2406.06851).
- Legal state-dependent switching between couplings of the same kernel ("mixed couplings", Eberle 2016): "employ a contractive coupling of P ... if |x − y| is large and a maximal coupling of P if |x − y| is small" — [AJ24 §4.2](https://arxiv.org/abs/2406.06851); see also PS25 two-scale (§3).

### Inferences
- **IMH** (e.g. with a pilot-fitted Gaussian or Student-t proposal) is the simplest exact-meeting kernel for v0.1. Coupling: one shared proposal and one shared uniform per step, so only one new target evaluation per coupled step. If w(x) = π(x)/q(x) ≥ w(y), then X accepting implies Y accepting, so the pair meets the first time the higher-weight chain accepts (inference from the acceptance rule). Meeting tails are then geometric with rate at least 1/sup w (normalised weights), matching IMH's uniform-ergodicity rate. Heavy-tailed weights give heavy-tailed meeting times; a good proposal is crucial.
- **IMH–RWM mixture:** a common-coin mixture of a contracting/mixing kernel (e.g. preconditioned RWM with reflection-maximal) and an IMH meeting kernel is a published pattern (Heng & Jacob's structure with a different meeting kernel). It is valid if the coin is shared and independent of the states.
- **MALA:** use reflection-maximal proposals (whitened by the preconditioner, as for RWM) and a common uniform. The meeting radius scales with the proposal standard deviation, √h; with MALA's larger stationary step this is an easier target than RWM's O(d^{-1/2}) (inference). MALA's step-size sensitivity out of stationarity (known from the optimal-scaling literature, e.g. Christensen, Roberts & Rosenthal 2005; not re-checked here) argues for conservative steps from overdispersed starting points.
- **pCN:** both chains' proposals are N(ρx, (1−ρ²)C) and N(ρy, (1−ρ²)C), so reflection-maximal applies. The one-step proposal-meeting probability is 2Φ(−ρ‖C^{-1/2}(x−y)‖/(2√(1−ρ²))) (derived from the Gaussian TV formula). Synchronous pCN contracts but never meets on its own.
- **Component-wise MH:** couple each coordinate's (or block's) update with its own maximal proposal coupling and its own common uniform. A coordinate that has met can separate again while other coordinates differ, because its conditional depends on the others. Only full-vector equality counts as a meeting: track per-coordinate equality flags and declare τ when all coordinates are flagged equal at the end of a sweep.

### Gaps
- No published analysis of coupled MALA meeting times as a function of d (only contraction results such as Eberle-type analyses, which I did not read) and no published coupled-pCN meeting-time study with exact meetings.
- No paper found that studies IMH coupling with a mixture of a local kernel as a general-purpose default; the IMH tail claims above are inference.
- Ceriani et al.'s supplement (logarithmic MH-within-Gibbs result) was not read; the claim is quoted from the main text.

---

## 5. Implementation details: log-space acceptance, meeting detection, vectorising many pairs, random-number consumption, numerical pitfalls

### Takeaway
Detect meetings with a boolean flag: a meeting is proposed and both chains accept. Then copy one state into the other. Never test approximate equality, and never recompute Y on the meeting branch: recomputing gives bitwise-different vectors in essentially every case. After meeting, advance a single chain and alias the other, which is faithful by construction and halves the cost. Do all accept/reject and coupling tests in log space with guards for non-finite log-densities and for zero distance. Prefer couplings with a deterministic number of random draws (reflection-maximal, CRN/GCRN) for vectorised batches. Rejection-based maximal couplings make a batch wait for its slowest pair.

### Cited Findings
- Reference R kernel: the coupled step reuses the first proposal's log-density when the proposals are identical; accepts with `logu < proposal_pdf - current_pdf` using one `logu` for both chains; treats non-finite proposal log-densities as rejection (`if (is.finite(proposal_pdf1))`); and returns `identical_ <- proposal_value$identical && accept1 && accept2` — [unbiasedmcmc R/get_mh_kernels.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/get_mh_kernels.R).
- After meeting, the driver no longer calls the coupled kernel: `if (is.finite(meetingtime)) { state1 <- single_kernel(state1); state2 <- state1 }`. Cost is accounted as `lag + 2*(meetingtime - lag) + max(0, time - meetingtime)` kernel calls, and the chains run until max(m, τ) with a `max_iterations` guard — [unbiasedmcmc R/coupled_chains.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/coupled_chains.R).
- Reflection-maximal C++: if `normz < 1e-15` it declares the draws identical (comment: identical means "would cause trouble in the forthcoming division by normz"). On the meeting branch it writes the same vector into both columns (`if (identical_) xy(i,1) = xi(i)`) rather than computing μ2 + L(ξ+z) — [unbiasedmcmc src/mvnorm.cpp](https://github.com/pierrejacob/unbiasedmcmc/blob/master/src/mvnorm.cpp).
- PS25 C++: accept tests `if (logHR_x > 0 || log_u < logHR_x)`. The coupled flag is set on the maximal branch (`yp = xp; logpi_yp = logpi_xp`), and the loop breaks on `acc_x * acc_y * coupled` — [rwmcouplings src/svm.cpp](https://github.com/tamaspapp/rwmcouplings/blob/master/src/svm.cpp). CS22's JAX transport map guards `r_curr < 1e-8` by setting the reflection direction to zero — [reflected_coupled_mh.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/coupled_rejection_sampling/reflected_coupled_mh.py).
- Counter-example (2019 Python): the vectorised reflection-maximal coupling computes Y as `mu2 + cholesky·(base_draw_x + transformed_diff)` even when coupled, and meeting is detected with `met = np.isclose(data.x[t], data.y[t - lag])` (approximate equality). Chains are also not forced equal after meeting — [ColCarroll couplings/maximal_couplings.py](https://github.com/ColCarroll/couplings/blob/master/couplings/maximal_couplings.py), [couplings/metropolis_hastings.py](https://github.com/ColCarroll/couplings/blob/master/couplings/metropolis_hastings.py).
- **[own check]** Recomputing Y = μ2 + L(ξ + L^{-1}(μ1−μ2)) instead of copying X gave X ≠ Y bitwise in 100% of 20,000 random d=10 cases (max |X−Y| = 7.1e-15). In the vectorised coupled RWM with flag-based detection and copying, `np.all(x[met] == y[met])` held for every flagged meeting — [check_couplings.py, checks C and D](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/mh/scripts/check_couplings.py).
- Vectorisation patterns:
  - ColCarroll vectorises over a `chains` axis (states `(iters, chains, dim)`, batched `log_prob`, one pre-drawn log-uniform per pair per step); his vectorised rejection coupling loops `while n_resample:` over a mask of unfinished pairs — [ColCarroll couplings](https://github.com/ColCarroll/couplings/blob/master/couplings/maximal_couplings.py).
  - CS22 write the samplers with `jax.lax.while_loop`, `jax.lax.cond` and `jax.lax.select`, and vmap meeting-time experiments over keys — [coupled_rejection_sampler.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/coupled_rejection_sampling/coupled_rejection_sampler.py), [tests/replicate_max_mh_coupling.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/tests/replicate_max_mh_coupling.py).
  - CS22 motivate bounded-variance couplings by "the context of parallel hardware ... requires all parallel executions to finish to proceed" — [CS22 §4](https://arxiv.org/abs/2201.09585).
- Lag convention for general lag L: τ = inf{t ≥ L : X_t = Y_{t−L}}; the coupled kernel acts on (X_t, Y_{t−L}) — [AJ24 §2](https://arxiv.org/abs/2406.06851); [unbiasedmcmc R/coupled_chains.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/coupled_chains.R).
- Closed-form, stable log-ratio for reflection-maximal: −½[(eᵀξ+‖z‖)² − (eᵀξ)²] — [unbiasedmcmc src/mvnorm.cpp](https://github.com/pierrejacob/unbiasedmcmc/blob/master/src/mvnorm.cpp). The rejection coupling's tests are written in log space in [R/get_max_coupling.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/get_max_coupling.R) and as differences of `logpdf`s in [mvnorm.cpp](https://github.com/pierrejacob/unbiasedmcmc/blob/master/src/mvnorm.cpp).
- Cholesky conventions differ: R `chol()` is upper-triangular U with Σ = UᵀU, and the code requires "Cholesky <- chol(Sigma), not the transpose" (`stopifnot` on lower-triangle zeros); CS22 use lower-triangular `chol_P` with x = m + Lξ — [unbiasedmcmc R/mvnorm_couplings.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/mvnorm_couplings.R), [mvn.py](https://github.com/AdrienCorenflos/coupled_rejection_sampling/blob/master/coupled_rejection_sampling/mvn.py).

### Inferences
- **Meeting detection contract.** Each coupled kernel returns `(x_new, y_new, met: bool[batch])`, with `met = proposal_identical & acc_x & acc_y`. On `met`, the implementation sets `y_new[met] = x_new[met]` (and the cached log-density and gradient) by copy. A debug-mode assertion `np.array_equal` on met pairs is a cheap invariant test. Exact equality is still worth checking after meeting, as a faithfulness test.
- **After meeting.** Drop the pair from the coupled batch and evolve one chain with the marginal kernel until max(m, τ). Store only the bias-correction sums Σ_{t=k+1}^{τ−1}[h(X_t) − h(Y_{t−L})] needed by the estimator, which saves memory when τ ≫ m.
- **Vectorised layout.** Use states of shape (M, d) with cached log-densities (M,), a batched user log-density (with a vectorize-if-needed wrapper for scalar callables), an `active` index set compacted each step, and per-pair counters (τ, coupled steps, single steps, target evaluations). Rejection-based couplings need masked inner loops; the Gerber–Lee η cap bounds the expected inner-loop length per pair (E ≤ 2, Var ≤ 2η/(1−η)).
- **Random numbers per coupled step** (from the algorithms above):

  | coupling | random draws per coupled step |
  |---|---|
  | reflection-maximal + common U | d normals + 2 uniforms (fixed) |
  | CRN | d normals + 1 uniform |
  | GCRN | (d+1) normals + 1 uniform |
  | GCRefl | (d+1) normals + 1 uniform; d normals + 2 uniforms on the reflection-maximal branch |
  | γ-coupling + common U | random: d normals + 1 uniform + Geometric × (d normals + 1 uniform), + 1 uniform for acceptance |
  | P̄C | as its Q̄, + 1 uniform |
  | P̄MI / P̄MR | random |

  Fixed consumption makes counter-based or per-pair RNG streams (e.g. NumPy `SeedSequence.spawn` per pair) reproducible regardless of batch size.
- **Numerical pitfalls checklist.**
  - Map NaN log-densities to −inf before comparisons.
  - Use the strict test `log(U) < logα`. NumPy draws U from [0,1), so log U can be −inf; with a strict inequality, proposals with logα = −inf (zero target density) are still always rejected.
  - Guard ‖z‖ = 0: treat as identical, or fall back to CRN as PS25 do.
  - Include log-determinants when the coupled distributions differ in covariance.
  - Whiten with the same triangular factor used to generate the proposals.
  - Never use `np.isclose` for meeting.
  - For P̄C, compute b and c in log space: log b = min(0, log f − log q^m); for c = (f − q^m)₊/(q − q^m), use c = 0 if log f ≤ log q^m, else log c = logsubexp(log f, log q^m) − logsubexp(log q, log q^m), with c = 1 when q = q^m. This avoids exp overflow and underflow in the tails.
  - Recompute cached gradients only on acceptance, but always copy them on meeting.

### Gaps
- No published benchmark compares vectorised (SIMD/GPU) throughput of rejection-based vs reflection-maximal couplings for MH. CS22 Appendix J gives only a stylised runtime argument for ensemble rejection on GPUs.

---

## 6. Existing code for these couplings (R, Python, Julia)

### Takeaway
The reference implementations are Jacob's R package `unbiasedmcmc` (Rcpp/Eigen; still maintained, last commit Sept 2026) and Papp's `rwmcouplings` (R + C++; GCRN, GCRefl and two-scale). Corenflos's JAX package covers the coupled rejection sampler, the modified Thorisson coupling, the log-space categorical maximal coupling and OWJ21's P̄MR. ColCarroll's 2019 NumPy package is the only general Python MH implementation found, and it has the approximate-meeting flaw described in §5. I found no Julia implementation.

### Cited Findings
- **pierrejacob/unbiasedmcmc** (R, Rcpp/RcppEigen; commit 660c698, 2026-09-23, "resolves a bug in the Polya-Gamma Gibbs sampler's coupling. Bug spotted by Tamas Papp") — [repo](https://github.com/pierrejacob/unbiasedmcmc):
  - Primitives: `R/get_max_coupling.R` (generic γ-coupling from `rp, dp, rq, dq`); `R/rnorm_max_coupling.R`; `R/rnorm_reflectionmax.R` (1-D); `R/mvnorm_couplings.R` (`rmvnorm_max`, `rmvnorm_max_chol`, `rmvnorm_reflectionmax`, backed by `src/mvnorm.cpp`); `R/gamma_couplings.R`, `R/invgamma_couplings.R`, `R/invgaussian_couplings.R` (couplings of conditional distributions for Gibbs).
  - Kernels: `R/get_mh_kernels.R` returns closures `single_kernel(state)` and `coupled_kernel(state1, state2)`, where a state is a list `{chain_state, current_pdf}` and the coupled kernel returns `{state1, state2, identical}` (reflection-maximal plus a common uniform).
  - Drivers: `R/coupled_chains.R` (`sample_coupled_chains(single_kernel, coupled_kernel, rinit, m, lag, max_iterations)`), `R/meetingtime.R`, `R/unbiasedestimator.R`, `R/H_bar.R`.
  - Scripts: experiment reproductions in `inst/reproducescalingdimension/scaling.rwmh.maximalcoupling.R` and `scaling.rwmh.reflectionmaxcoupling.R`; checks in `inst/check/check_reflmaxcoupling.R` and `check_normalcouplings.R`.
- **pierrejacob/couplingsmontecarlo** (R; course/book scripts; last commit 2020-07-21) — [repo](https://github.com/pierrejacob/couplingsmontecarlo): `R/getmaxcoupling.R`; `inst/chapter4/4coupled_mh.R`, `4coupled_mh_contraction.R` (independent vs CRN vs maximal vs reflection-maximal proposal couplings, with the same flag/copy conventions), `4couplingmanymh.R`, `4unbiasedhmc.R`; `inst/chapter2/2rejectionmaxcoupling.R`.
- **tamaspapp/rwmcouplings** (R package with C++ via Rcpp and a PCG RNG; last commit 2024-10-28 "Updated to arXiv v3") — [repo](https://github.com/tamaspapp/rwmcouplings):
  - `src/svm.cpp`: `svmReflMaxRWM`, `svmTwoScaleGCRNRWM` (GCRN when |x−y|² ≥ thresh, reflection-maximal otherwise), `svmTwoScaleGCReflRWM`, Laplace-preconditioned variants.
  - `src/samplers.cpp`: generic `RWM_crn`, `RWM_gcrn`, `RWM_refl`, `RWM2targets_*`; plus Hug-and-Hop couplings.
  - `R/ode.R`: limiting ODEs. `inst/`: experiment scripts (`spherical.R`, `elliptical.R`, `scaling/`, `svm/`, `binreg/`). The paper states "All computations were carried out in R ...; runtime-critical components were written in C++" — [PS25 App. C](https://arxiv.org/abs/2211.12585).
- **AdrienCorenflos/coupled_rejection_sampling** (Python/JAX; last commit 2022-05-25) — [repo](https://github.com/AdrienCorenflos/coupled_rejection_sampling): `coupled_rejection_sampler.py` (`coupled_sampler` for CS22 Algorithms 1–2; `_index_max_coupling`), `mvn.py` (`get_optimal_covariance`, `coupled_mvns`, `reflection_maximal`, `lower_bound`), `thorisson.py` (`modified_thorisson`), `reflected_coupled_mh.py` (OWJ21 Algorithm 4 in log space), `rejection_sampler.py`, `coupled_parallel_resampler.py`, `gauss_tails.py`; tests (`test_reflected_coupled_mh.py`, `replicate_max_mh_coupling.py`, `test_thorisson.py`, …) and examples (`manifold_mala.py`, `gibbs.py`).
- **ColCarroll/couplings** (Python, NumPy/SciPy; last commit 2019-09-19) — [repo](https://github.com/ColCarroll/couplings): `couplings/maximal_couplings.py` (`maximal_coupling_reference`, vectorised `maximal_coupling` with a cost counter, `ReflectionMaximalCoupling` class over a `chains` batch), `couplings/metropolis_hastings.py` (`metropolis_hastings(log_prob, proposal_cov, init_x, init_y, lag, iters, chains, short_circuit)` returning a `CoupledData` container; `unbiased_estimator(data, func, burn_in)`), `couplings/coupled_data.py`.
- Other repositories found by GitHub search but not inspected:
  - [niloyb/LlagCouplings](https://github.com/niloyb/LlagCouplings) (R; L-lag couplings)
  - [pierrejacob/debiasedhmc](https://github.com/pierrejacob/debiasedhmc) (C++/R; Heng & Jacob)
  - [pierrejacob/umcmchandbook](https://github.com/pierrejacob/umcmchandbook) (R; AJ24 chapter code)
  - [ylefay/unbiased_mcmc_with_couplings](https://github.com/ylefay/unbiased_mcmc_with_couplings) (JAX; course project, "coupled Metropolis-Hastings")
  - [darrenjw/unbiased-mcmc](https://github.com/darrenjw/unbiased-mcmc) (talk/notes, TeX)
  - [arijitdey22/Maximal_Coupling_of_MCMC](https://github.com/arijitdey22/Maximal_Coupling_of_MCMC) (R; course project)
  - [ruslenghi/Coupled-Markov](https://github.com/ruslenghi/Coupled-Markov) (Java)

  Search queries: [GitHub search "unbiased mcmc couplings"](https://github.com/search?q=unbiased+mcmc+couplings&type=repositories); [GitHub search "maximal coupling MCMC"](https://github.com/search?q=maximal+coupling+MCMC&type=repositories).

### Inferences
- The R package's architecture is the cleanest template for `telescopic`:
  - a marginal kernel plus a coupled kernel operating on states that cache `current_pdf`;
  - an `identical` flag;
  - a driver that switches to the single kernel after meeting;
  - estimator code separated from chain generation.

  A NumPy port should batch it over pairs (ColCarroll-style axis), keep R's flag-and-copy semantics, and use PS25's code as the reference for GCRN/GCRefl if gradient-based kernels are added later. CS22's JAX code is a good reference for the optional JAX backend, especially its log-space categorical coupling and while-loop structure.
- Reference values for tests:
  - O21 Table 1 at d=10: about 30 for reflection-maximal and 279 for maximal-independent with V=U (reproduced at 32.3 and 274.2).
  - OWJ21 Table 1 (Expo target): P̄SQ about 74 and P̄C about 61 (reproduced).
  - Reflection-maximal meeting probability 2Φ(−‖Δ‖/2).
  - γ-coupling draw-count mean 2 and variance 2(1−TV)/TV.

### Gaps
- No Julia implementation of MH couplings was found (searched GitHub repositories for "unbiased mcmc couplings", "maximal coupling MCMC", "L-lag couplings", "coupled rejection sampler"), so their absence is not certain.
- The R repositories' C++ for Gibbs/Polya-Gamma couplings and the uninspected repositories may contain further MH coupling variants not reviewed here.

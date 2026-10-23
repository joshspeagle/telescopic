# Adversarial review of the v0.1 design documents

**Reviewed:** `docs/design/spec.md` (md5 `0e6cabf0`), `docs/design/test-plan.md` (`3e00f66f`),
`docs/design/implementation-plan.md` (`69a3a8ca`), unchanged during the review.
**Evidence base:** `reports/`, the eight notes and archived scripts in `research_notes/`,
`notes/experiments/e01_lag_recipe.py`, `notes/DECISIONS.md`, `CLAUDE.md`.
**Date:** 2026-10-23. **Reviewer:** independent subagent; no file other than this report was edited.

**Numerical checks.** Scripts and their outputs are archived in
`notes/reviews/scripts/design-review/` (copied from the session scratch, without the
downloaded SciPy sources). Every `*.out` file named below sits next to its script. Environment: Python 3.13.16, NumPy 2.5.4, SciPy 1.18.1,
OpenBLAS 0.3.34 (SkylakeX build), 4 vCPU.

The M0 scaffold (`src/telescopic/_random.py`, `tests/test_random.py`, `pyproject.toml`)
appeared during the review. I did not review code; I cite it only where it already took a
decision that the spec does not record.

## Summary

38 findings: **1 blocker, 8 major, 20 minor, 9 nits.**

The estimator mathematics of spec §3.2 is right. The lag weights v_t, the thinning
coefficients c_t, the signed measure, the cost formula, the TV identity and the U-statistic
covariance all check out numerically. The problems are in the contracts around the
estimator, in the tuning defaults and warnings, and in several tests.

The five most important:

1. **D-R1 (blocker).** The met-flag contract (R-KER-3, R-KER-4) contradicts the kernels' own
   met rules ("identical and both accept"). From equal states a joint rejection leaves the
   pair equal but unflagged (65% of RWMH steps). Meetings at t = L have no mechanism, and the
   implementation plan forbids the comparison that would detect them. With a point-mass π0,
   45% to 57% of lag-1 pilot pairs are affected. Tests 2.4 (faithfulness, flag), 2.5 (perfect
   kernel) and `check_coupling` would fail on a correct implementation of §5.3.
2. **D-R3.** The interrupt rule returns GH91's pooled complete-in-progress average, a ratio
   estimator, and calls it "a valid fixed-R result". Bias: z = −12.9 with one lane and −7.6
   with four. The census report recommended "no estimate unless a valid rule applies".
3. **D-R5, D-R6.** "k at the 99% quantile" does not give a TV bound of about 0.01 once L = k:
   measured 0.019 (d = 10) and 0.029 (d = 30). With the default 100-replicate pilot, warning 2
   fires on clean Gaussian runs with probability 0.25 (d = 10) and 0.49 (d = 30). Warnings 3
   and 4 also misfire at small R, and "Hill of τ" disables warning 4 when L = k.
4. **D-R4.** Bitwise invariance across batch sizes (R-EXEC-2) fails with ordinary BLAS-backed
   kernel arithmetic: a row of `z @ chol.T` takes different bits in a batch of 1 than in a
   batch of 2 or more. `np.einsum` does not have this problem. The plans do not say how to
   meet the requirement.
5. **D-R8, D-R9.** Two tests check the wrong thing. The d > 1 marginal test ("KS on whitened
   coordinates") rejects a correct RWMH kernel at p = 1e-242. The bimodal calibration test
   expects warnings to fire, but in the simulation they fired only in runs whose intervals
   already covered the truth (2 of 14 misleading runs warned).

Also major: D-R2 (Gibbs needs exact comparisons, and eta < 1 is not faithful from equal
states) and D-R7 (the uncapped pilot can hang `sample()` with no message).

## Findings

Severity: **blocker** = must be resolved before the affected milestone, because code written
from the documents would be wrong or could not pass its own tests; **major** = wrong claim,
unsound test or default that misleads users; **minor** = gap or inconsistency with a clear
fix; **nit** = wording or small accuracy issue.

### D-R1. The met-flag contract contradicts the kernels' met rules, and meetings at t = L have no mechanism (blocker)

- **Location:** spec R-EST-2 (§3.1, line 128); R-KER-3 and R-KER-4 (§5.1, lines 285-286);
  §5.3 RWMH (line 334), reused by MALA (§5.4) and the RWMH component of HMC (§5.5); §5.6
  IMH (line 384); §9 `check_coupling` (lines 613-618). Test plan 2.4 items 3 and 4
  (lines 99-100), 2.5 perfect kernel (line 142), 2.11 `check_coupling` passes the built-in
  kernels. Implementation plan §4 "Meetings" (line 211).
- **Claim:**
  - R-KER-4 says that from equal states `coupled_step` returns `met` true, and R-KER-3 says
    that `met` holds if and only if the outputs are bitwise equal.
  - §5.3 defines `met = identical ∧ both accept`, and §5.6 defines `met = both accept`. From
    equal states the two chains share the proposal and the uniform, so they accept or reject
    together. When both reject, the outputs are equal and `met` is false.
  - R-EST-2 counts τ = L when X_L = Y_0, but no coupled step has run at t = L, so there is
    no flag. The implementation plan allows comparisons "only in tests and debug assertions".
    The test plan's perfect-kernel test nevertheless expects τ = L in that case.
- **Evidence:**
  - `check_couplings.out`: RWMH from equal states (d = 5, 100,000 pairs): outputs equal in
    100% of pairs, `met` true in 34.6%, equal but unflagged in 65.4%. IMH from equal states:
    both reject, so equal but unflagged, in 16.8%.
  - `check_point_mass_tau.out`: with a point-mass π0 (used by test plan 2.12) and L = 1,
    X_1 = Y_0 in 45% (d = 1) and 57% (d = 5) of pairs. AJ24's convention gives τ = 1 there.
    A flag-only engine records τ = 1 + G with E[G] = 1.8 to 2.3 and counts those steps as
    coupled (two kernel applications each). With an ensemble used as π0 (n points),
    X_0 = Y_0 with probability 1/n.
  - H is unaffected, because Δ_t = 0 on equal pairs. The recorded τ, the cost, the pilot's k,
    the TV bound and the atom count are affected.
  - The census report already describes this behaviour of the R package: "a pair that is
    already equal and jointly rejects is recorded later than it met".
- **Fix:**
  - Define `met` as exact equality of the returned positions, computed once per step
    (`np.all(x_out == y_out, axis=1)`).
  - Keep copy-on-event, so that the coupling event implies equality, and copy the caches
    whenever positions are equal.
  - Have the engine compare X_L with Y_0 once at t = L and set τ = L if they are equal.
  - Exact comparison has no false positives and is not a tolerance. What must stay forbidden
    is a tolerance or recomputing Y. Rewrite implementation plan §4 accordingly.
  - Alternative: OR the event flag with input equality, and state in R-KER-3 that
    coincidental equality (for example HMC contraction below one ulp) is then recorded one
    step late.

### D-R2. Gibbs needs exact comparisons, and eta < 1 is not faithful from equal states (major)

- **Location:** spec §5.7 (lines 390-398); §5.2 η option (lines 314-318); R-KER-4;
  implementation plan §4 (line 211).
- **Claim:**
  - (a) "Meeting: declared when the whole state is bitwise equal at the end of a sweep"
    needs a comparison. Blocks updated by `"crn"` become equal without any coupling event,
    and met blocks can separate again. The implementation plan forbids that comparison
    outside tests.
  - (b) With η < 1 the rejection coupling returns X ≠ Y with probability 1 − η even when
    p = q. A Gibbs kernel with `eta < 1`, started from equal states, therefore splits the
    pair. That violates R-KER-4 and fails test 2.4 item 3. The engine never calls the
    coupled kernel after a meeting, but it does at t = L when X_L = Y_0 (D-R1).
- **Evidence:** `check_couplings.out`: η = 0.9 and p = q: outputs differ in 9.93% of 20,000
  calls (1 − η = 0.10). `theory_estimators.md` §1 (Inferences) makes the same point.
- **Fix:**
  - Compute `met` at the end of a sweep by exact comparison (the D-R1 rule).
  - Within a sweep, when the conditioning blocks of X and Y are bitwise equal, give both
    chains the same draw. This identity coupling of identical conditionals is a legal
    state-dependent choice.
  - Say in §5.7 that `eta < 1` relies on this short-circuit.

### D-R3. The interrupt rule is a biased ratio estimator presented as a valid fixed-R result (major)

- **Location:** spec R-EXEC-6 (§6.5, lines 457-459); implementation plan §2.7 (line 148);
  test plan 2.10 (line 212). Report, "Memory, storage and interrupts" and its policy table
  ("Interrupted run ... biased, unplanned deadline ... flagged partial result");
  `parallel_engineering.md` Inferences ("estimate=None by default").
- **Claim:**
  - The spec says an interrupted result "is a valid fixed-R result only for the indices it
    contains, which are always a prefix".
  - With refill, the indices started by the interrupt are all replicates completed by then
    plus the one in progress on each lane. Their average is GH91's pooled
    complete-in-progress estimator, which is a ratio estimator. With one lane it is GH90's
    mean over N(t) + 1 replicates, which GH90 show is biased at order 1/t.
  - The prefix property does not make this a fixed-R result. The prefix length depends on
    the replicates' costs, and the costs correlate with H through τ.
  - This departs from the report without a decision entry.
  - The implementation plan describes yet another rule: it "cancels pending chunks and
    returns the completed prefix". If running chunks are not awaited, the long chunk in
    progress and every later chunk are dropped. That is a length-biased selection like the
    "completed by deadline" policy, which the census measured at z = +165.
- **Evidence:** `check_interrupt_bias.out`. The census toy from
  `research_notes/.../scripts/parallel/coupled_rwmh.py`: coupled RWMH on N(0,1), π0 = N(3,1),
  k = 2, m = 10, corr(H, C) = −0.19, a pool of 10^6 replicates.

  | Lanes | Interrupt at | Bias | z |
  |---|---|---|---|
  | 1 | 1.4 E[C] | −0.31 | −12.9 |
  | 1 | 2.7 E[C] | −0.26 | −14.4 |
  | 1 | 9 E[C] | −0.072 | −8.7 |
  | 2 | 1.4 E[C] | −0.12 | −8.9 |
  | 4 | 1.4 E[C] | −0.067 | −7.6 |
  | 16 or more | any | not detectable | \|z\| ≤ 1.9 |

  The default in-process path (64 lanes) is therefore practically fine. A pool with a few
  workers, where each worker acts as a lane, is not. In this toy the bias is small next to
  the standard error at the R that an interrupt leaves, so the problem is mainly the claim,
  which principle 1 and §3.3 rely on.
- **Fix:**
  - Label interrupted results "consistent, not exactly unbiased" in `Result` and
    `summary()`, and let `run` complete the declared R, which restores exactness.
  - Optionally record lanes and completion order, so that the GH per-lane estimate can be
    offered. It is exactly unbiased only if the interrupt time does not depend on the
    output; a user who reacts to the progress bar breaks that.
  - Make the implementation plan await running chunks, as the spec says, and test that the
    returned set is "all started indices".
  - Add a one-lane toy test with corr(H, C) that documents the bias.

### D-R4. Bitwise invariance across batch sizes fails with BLAS-backed kernel arithmetic (major)

- **Location:** spec R-EXEC-2 (§6.1, line 417); test plan 2.6 refill (line 152), 2.9 ("two
  runs of 50 equal one run of 100"), 2.10 pools (line 210); implementation plan M3 exit
  criterion and §4 "Whitening".
- **Claim:**
  - R-EXEC-2 promises bitwise-identical replicates across batch sizes when the log-density is
    evaluated point by point.
  - The kernels themselves do batched linear algebra: the RWMH proposal x + λ z Lᵀ, the
    MALA drift Σ∇, the HMC position update L ν, and the whitening L⁻¹(μ_x − μ_y).
  - With NumPy matmul, a row computed in a batch of one takes a different BLAS path from the
    same row in a batch of two or more. The drain phase and `lanes=1` produce batches of
    one, so the same replicate gets different bits, and an accept or reject decision can
    then flip.
  - Neither plan says how to avoid this.
- **Evidence:** `check_batch_bitwise_1thread.out`, `check_batch_bitwise2.out`,
  `check_batch_bitwise3.out`; the same results with 1 and 4 BLAS threads.
  - Rows of `z @ chol.T` differ from the single-row result in 9 of 45 rows at d = 2, 37 of
    45 at d = 5 and 44 of 45 at d ≥ 10.
  - The only batch size that differs is n = 1; the largest difference is 4.4e-16 at d = 10.
  - Rows of `np.linalg.solve` differ in 21 to 44 of 45 rows.
  - `np.einsum('ij,kj->ik')` with the default `optimize=False`, `np.sum(e * z, -1)` and
    `np.einsum('ij,ij->i')` never differed.
  - The census saw the same effect for a batched log-density (8 to 25 of 38 cases at d ≥ 7).
- **Fix:**
  - Require kernels to compute per-row quantities with arithmetic whose result for a row
    does not depend on the batch: `np.einsum` without optimisation, or elementwise products
    summed along the last axis. Whiten with a precomputed inverse Cholesky factor applied the
    same way.
  - Add a unit test: a kernel step on a batch of 1 equals the same rows in batches of 2 and
    64, at d ≥ 5 with a dense Σ.
  - Extend the invariance test to MALA, to HMC and to a kernel with variable draw counts
    (Gibbs, or RWMH `"maximal"`).
  - Qualify the promise: "on one machine, with one BLAS build and fixed NumPy and SciPy
    versions".

### D-R5. The recipe does not give a TV bound of 0.01 at k, and the pilot size makes k and the "conservative mode" unreliable (major)

- **Location:** spec R-TUNE-2 (lines 203-208), §3.6 "Use for k" (line 219), D-5, Appendix A
  E01; test plan 2.9 `choose_kml`.
- **Claim:**
  - (a) §3.6 says that "k is the 99% quantile" means "the estimated bound at k is about
    0.01". But k is a quantile of τ − 1 in the lag-1 pilot, while the bound at t = k uses
    τ − L at L = k. That distribution is longer, because the coupled phase starts from an
    almost stationary X_L and a Y_0 drawn from π0.
  - (b) With R0 = 100, the 99% quantile is essentially the second largest of 100 values, so
    k varies by about ±20%.
  - (c) At β = 0.999 with R0 = 100, `np.quantile`'s default interpolation lands 90% of the
    way from the second largest value to the largest. The "conservative mode" is the sample
    maximum, not a 99.9% quantile.
  - (d) "E01 supports the default", but E01 used a 2000-replicate pilot and floor (`int()`),
    not R0 = 100 and ceil.
  - None of this biases estimates. It changes the cost, the fraction of replicates with a
    bias correction, and how often warning 2 fires (D-R6).
- **Evidence:** `check_recipe2.out`: E01's target and coupling, a 2000-replicate pilot, main
  runs of R = 4000.

  | | d = 10 | d = 30 |
  |---|---|---|
  | k from the 2000-replicate pilot | 152 | 542 |
  | main-run P(τ − L > k) | 0.019 (se 0.002) | 0.0285 (se 0.003) |
  | estimated TV bound at t = k | 0.019 | 0.0285 |
  | mean(τ − L), main run, vs mean(τ − 1), pilot | 76 vs 46 | 313 vs 199 |
  | k from 100-replicate pilots, 5%/50%/95% | 117/142/168 | 423/480/622 |
  | main-run P(τ − L > k) at those k, median (95th pct.) | 0.030 (0.10) | 0.064 (0.125) |
  | β = 0.999 from 100 replicates, median | 157 | 579 |
  | β = 0.999 from the 2000-replicate pilot | 191 | 687 |

  E01 itself recorded a non-zero bias correction in 3.5% of replicates at d = 10 with L = k.
  The rerun reproduced every figure in its header exactly (`e01_rerun.out`).
- **Fix:**
  - Replace the §3.6 sentence with the measured relation, or report the main run's own
    estimate of P(τ − L > k) as the TV bound at k.
  - State the trade-off of the pilot size.
  - For β = 0.999, require R0 ≥ 1000, or define the conservative mode another way (for
    example k = 2 × q99, or extrapolation of a geometric tail). Either is a decision to
    record.
  - Fix the quantile method (`method="higher"` or `"inverted_cdf"`) and say "ceil".
  - Rerun E01 with R0 = 100, or describe it as using R0 = 2000.

### D-R6. Warnings 2, 3 and 4 fire on clean runs at the defaults, and warning 4 is under-specified (major)

- **Location:** spec §7.4 (lines 526-532), D-16; test plan 2.8 "None fires on a clean
  Gaussian run" (line 182), 2.12 "fire in under 1% of runs" (line 244).
- **Claim:** The thresholds are documented as heuristics, but two tests gate on them, and the
  default settings trigger them on clean Gaussian targets.
  - (a) Warning 2's threshold of 0.05 lies inside the recipe's own range of P(τ − L > k)
    (D-R5).
  - (b) Warning 3 fires when one replicate contributes more than 25% of the sample variance,
    for R ≥ 20. At R = 20 to 30 that is the typical value of the statistic for Gaussian data,
    and the summary reports 2d estimates.
  - (c) Warning 4 is "a Hill tail-index estimate of τ below 2". It fixes neither the variable
    (τ or τ − L) nor the number of order statistics k_h.
- **Evidence:**
  - `check_recipe2.out`: in a default run (R0 = 100 pilot, R = 100), warning 2 fires with
    probability 0.25 at d = 10 and 0.49 at d = 30. On 40 batches of R = 100 main-run
    replicates at d = 30 (60 estimates), warning 3 fired 3 times.
  - `check_warning_thresholds.out`, i.i.d. N(0,1) replicate values:

    | R | P(warning 3), 1 estimate | P(warning 3), 10 estimates |
    |---|---|---|
    | 20 | 0.39 | 0.99 |
    | 30 | 0.12 | 0.73 |
    | 50 | 0.01 | 0.09 |
    | 100 | 0 | 0 |

    With t5 values at R = 50 and 10 estimates, P = 0.92.
  - `check_hill.out`: Hill on geometric τ − L + 1 with n = 100 gives a median α̂ ≈ 3.1 and
    P(α̂ < 2) ≈ 0.04, with either k_h = 10 choice. At n = 1000, P = 0.
  - `check_hill_shift.out`: for truly heavy tails (α = 1.5) the power is 0.78 on τ − L + 1
    but 0.05 on τ when L = k. Read literally ("Hill of τ"), warning 4 is disabled in the main
    run.
- **Fix:**
  - Specify warning 4 on τ − L + 1 (or on the pilot's τ), with k_h stated (for example 10%
    of R, at least 10), and report its false-alarm rate.
  - Make warning 3 depend on R and on the number of estimates: apply it only for R ≥ 100, or
    compare the maximum share with its null quantile.
  - Make warning 2 a one-sided binomial test at level 0.01 against a stated rate, or raise
    the threshold to 0.10.
  - Keep test 2.8's "none fires" only for a configuration whose computed false-alarm
    probability is below 1e-3.
  - Make test 2.12's "under 1%" an output of M9's calibration, not an assertion, until the
    thresholds are calibrated.

### D-R7. The meeting-time pilot has no cap and no slow-pair warning, so `sample()` can hang silently (major)

- **Location:** spec R-EXEC-4 (lines 433-434), R-TUNE-2, §8.1 `sample()`;
  `diagnostics_practice.md` §3 Inferences ("R₀ ≈ 100-200 pairs with L = 1 and a generous
  cap; record censoring").
- **Claim:**
  - `max_iter=None` applies to the pilot as well.
  - The slow-pair warning is defined relative to "the pilot's 99% meeting-time quantile",
    which does not exist during the pilot, or when the user gives k and m.
  - On a target where the coupling does not meet (multimodal with narrow proposals,
    eccentric targets at high d, the kind of case in Appendix A E-R3), `telescopic.sample()`
    runs forever with no message.
  - The note's generous cap was dropped without a decision entry.
- **Evidence:** follows from the definitions in the spec and the notes cited; not simulated.
- **Fix:**
  - Give the pilot a cap, for example 100 × the running 90th percentile once 90% of pilot
    pairs have met, or a fixed generous number.
  - Treat capped pilot pairs as right-censored when computing q_β, and raise with advice if
    the β-quantile is censored.
  - Define a slow-pair threshold that exists without a pilot, for example 100 × the running
    median of completed replicates.
  - Show progress when a pilot takes more than a few seconds.

### D-R8. The d > 1 marginal test rejects correct MH kernels (major)

- **Location:** test plan 2.4 item 2 (lines 93-98, "KS tests on whitened coordinates for
  d > 1" at line 96).
- **Claim:**
  - Moved values of an MH step are accepted proposals. Their law is
    q(x'|x) α(x, x')/(1 − r(x)), which is not Gaussian, so whitened coordinates are not
    N(0, 1) and a correct kernel fails the test.
  - The stay probability r(x) in d > 1 is a d-dimensional integral, not one-dimensional
    quadrature.
- **Evidence:** `check_marginal_test_design.out`: a correct RWMH step on N(0, I_3) from a
  fixed state, n = 20,000. KS p-values of the whitened moved coordinates against N(0, 1)
  are 1e-242, 2e-103 and 0.
- **Fix:**
  - Keep the exact one-sample tests in d = 1.
  - For d > 1, compare each coupled marginal with `step` from the same state by two-sample
    tests: a two-proportion test of the stay frequency, and KS of moved values on random
    projections and on the norm. Increase n, since the census found two-sample tests about
    10× less sample-efficient.
  - The oracle draw-for-draw tests already pin the mechanics.

### D-R9. The bimodal calibration test expects warnings to detect a failure they cannot see (major)

- **Location:** test plan 2.12 "Warning calibration" (line 244).
- **Claim:**
  - The plan expects that "on JOA20's bimodal case with a narrow π0, warning 3 or 4 fires in
    most runs at R = 1000".
  - The warnings fire when a replicate crossed to the far mode. That is exactly when the
    interval widens and covers the truth.
  - In the misleading runs (no crossing, an interval of about [0.80, 0.81]) no warning
    fires. The test therefore passes or fails on the crossing rate, not on whether telescopic
    protects users.
- **Evidence:** `check_bimodal.out`. JOA20 §5.1 setting: ½N(−4,1) + ½N(4,1), RWMH proposal
  sd 1, π0 = N(10,1), L = 1, k = 50, m = 500, reflection-maximal coupling, 30 batches of
  R = 1000.
  - 27 of 30,000 replicates (0.09%) crossed before meeting.
  - The interval missed the truth in 14 of 30 batches.
  - Warning 3 fired in 18 of 30 batches, warning 4 in 1, warning 2 in none.
  - Among the 14 misleading batches, a warning fired in 2.
  - With a 60% firing rate and 30 runs, "most runs" fails with probability about 0.18. With
    JOA20's own coupling the crossing rate may differ.
- **Fix:**
  - Record these rates as documentation, not as an assertion.
  - State in the docs that this failure cannot be detected from the output at that R.
  - Assert instead what the design can show, for example that a wider π0 or a long lag
    exposes the second mode through meeting times or the TV bound.

### D-R10. The efficiency-loss approximation uses the wrong τ̄ when L > 1 (minor)

- **Location:** spec R-TUNE-4 (line 210); §7.3 footer (line 515).
- **Claim:** JOA20's 1 + (k + E[τ])/(m − k) is derived for L = 1. With lag L the cost is
  L + 2(τ − L) + max(0, m − τ) ≈ m + τ − L, so the analogue is 1 + (k + E[τ − L])/(m − k).
  With the main run's τ̄ at L = k, the formula is too high by L/(m − k) = 1/9 ≈ 0.11.
- **Evidence:** `check_recipe2.out`:

  | | d = 10 | d = 30 |
  |---|---|---|
  | measured mean cost / (m − k + 1) | 1.166 | 1.175 |
  | formula with the main-run τ̄ | 1.278 | 1.286 |
  | formula with the pilot τ̄ | 1.146 | 1.152 |
  | formula with the main-run mean(τ − L) | 1.167 | 1.175 |
- **Fix:** report the measured mean cost / (m − k + 1), or define τ̄ as the main-run mean of
  τ − L, and say which run it comes from.

### D-R11. The stopping rule in R-EST-1 reads as one extra step (minor)

- **Location:** spec R-EST-1 step 3 (line 126) against R-EST-7 (line 157).
- **Claim:** "draw (X_{t+1}, Y_{t+1−L}) ... until the pair has met and t + 1 > m". Read as a
  check after the draw, the last state is X_{max(τ, m+1)}. That is one step past the horizon
  max(τ, m) that the cost formula counts.
- **Evidence:** `check_estimator_algebra.out`: the cost formula matches the horizon
  max(τ, m) in every enumerated cell. The literal reading adds one step whenever τ ≤ m.
- **Fix:** "continue while t < max(m, τ)", as in the R code's loop, or AJ24's wording, "until
  X_{t+1} = Y_{t−L+1} and t + 1 ≥ m".

### D-R12. HMC and MALA details (minor)

- **Location:** spec §4.1 (line 243), §5.4, §5.5 (lines 357-378), D-11.
- **Claim:**
  - (a) Divergences are not specified. The spec does not say what happens when a leapfrog
    point or a MALA proposal has log π = −∞, or the gradient is non-finite. This is common
    with emcee-style priors that return −inf.
    - R-INPUT-1's "NaN raises" would abort the run if the user's gradient returns NaN there.
    - MALA's reverse proposal density needs ∇log π at x'.
    - An initial state with log π = −∞ is allowed (with an InputWarning), but then a −inf
      proposal gives a NaN log ratio and a RuntimeWarning, which is fatal under
      warnings-as-errors.
  - (b) The trajectory rule `n_steps = ⌈(π/2)/ε⌉` resonates.
    - It maximises contraction at whitened frequency ω = 1.
    - At ω = 2 it gives |cos(ωπ/2)| = 1, so no contraction. That is a direction whose
      variance the pilot underestimates fourfold.
    - Heng & Jacob's εL ≈ 1 keeps |cos ω| < 1 up to ω = π.
    - The rule is not a published default and is not in the decision table.
  - (c) The meeting component uses σ²Σ (whitened), where Heng & Jacob use σ²I. This is
    benign but not recorded.
  - (d) `meeting_prob` is the mixture weight γ, not a probability of meeting.
- **Evidence:** algebra; for the exact Gaussian flow the difference between the chains is
  proportional to cos(ωT).
- **Fix:**
  - A −inf log density or non-finite gradient at any leapfrog point rejects the whole
    trajectory without raising.
  - MALA rejects a −inf proposal before evaluating its gradient.
  - A −inf current state accepts any finite proposal and rejects −inf ones without computing
    −inf − (−inf).
  - Record the trajectory rule as a decision with its resonance caveat, or use εL ≈ 1.
  - Record σ²Σ, and rename the argument to `meeting_weight`.

### D-R13. The pilot-overhead target cannot be met with the specified pilots (minor)

- **Location:** spec §10 (line 632), §5.3 pilot (line 340), R-TUNE-2.
- **Claim:** "Automatic pipeline overhead (both pilots) ≤ 10% of the main run at the default
  R = 100" conflicts with 4 × max(2000, 200d) tuning steps plus 100 lag-1 pairs.
- **Evidence:** kernel applications, computed from `e01_rerun.out` and `check_recipe2.out`:

  | d | tuning pilot | meeting pilot | main run | overhead |
  |---|---|---|---|---|
  | 2 | 8,000 | 100 × (1 + 2 × 6.2) = 1,340 | 100 × 289 = 28,900 | 32% |
  | 10 | 8,000 | 9,240 | 150,700 | 11.4% |
  | 30 | 24,000 | 39,940 | ≈ 573,000 | 11.2% |

  The 1-lane target (≤ 40 µs per pair-step) equals the census's bare-step measurement
  (38.5 µs), before engine bookkeeping and per-replicate streams.
- **Fix:** scale the tuning pilot to the problem or to the main run, or restate the targets
  per d.

### D-R14. The reflection-maximal pseudocode divides by zero (minor)

- **Location:** spec R-CPL-1 (lines 295-300).
- **Claim:**
  - e = δ/|δ| is computed unconditionally. At δ = 0, which the spec handles as "identical
    with probability 1", NumPy raises a RuntimeWarning. That is an error under pytest's
    `filterwarnings = error`.
  - The pseudocode also sets y' = x' before x' is defined.
- **Evidence:** `check_couplings.out`: "RuntimeWarning: invalid value encountered in
  divide". Otherwise the algorithm is right (see "Checked and found correct").
- **Fix:**
  - Compute e only where |δ| > 0, by an exact test. A threshold such as the R code's 1e-15
    would copy Y = X for tiny non-zero δ and slightly break Y's marginal.
  - Reorder the lines.

### D-R15. The statistical tolerance rule is stated two ways (minor)

- **Location:** test plan §1 (lines 14-15), 2.12.
- **Claim:**
  - "Tolerances of 5 Monte Carlo standard errors" and "when 4 < |z| ≤ 6 ... rerun ... fails
    only if the rerun also exceeds 4" do not define one rule. What happens at 4 < |z| ≤ 5,
    and at |z| > 6?
  - Gandy & Scott's sequential test is a specific procedure with α spending; this is not it.
- **Evidence:** the text. `validation_benchmarks.md` measured P(|z| > 4) = 0.005 at R = 5000
  under heavy tails, about 80× the nominal rate, so the rule matters.
- **Fix:** one rule:
  - pass if |z| ≤ 4;
  - if 4 < |z| ≤ 6, rerun once at 4R with a fixed derived seed, and pass if and only if the
    rerun gives |z| ≤ 4;
  - fail if |z| > 6.

  Call it "sequential escalation".

### D-R16. The coverage band is tight for skewed h (minor)

- **Location:** test plan 2.12 "Coverage" (line 241).
- **Claim:** [0.93, 0.97] with 1000 intervals has a binomial SE of 0.007, and the true
  coverage of a correct implementation at R = 200 is not exactly 0.95.
- **Evidence:** `check_coverage.out`: exact AR(1) kernel with ρ = 0.9, recipe k = L = 38,
  m = 380, 1000 intervals of R = 200.

  | Start | h = x | h = x² |
  |---|---|---|
  | N(0, 1) | 0.945 | 0.939 |
  | N(1, 1) | 0.946 | 0.951 |

  If the true coverage of x² is 0.939, P(observed < 0.93) ≈ 0.11.
- **Fix:** use 2000 or more intervals with a band of ± 4 SE around a measured reference
  coverage, or test the calibration of the z-scores (KS of z against N(0, 1) at a stated α).

### D-R17. The Hill unit test passes only 69% of the time with k_h = √n (minor)

- **Location:** test plan 2.8 (line 174).
- **Evidence:** `check_hill_test.out`: P(|α̂/α − 1| < 0.1) at n = 10^4 is 0.698 (α = 1.5)
  and 0.679 (α = 3) with k_h = 100, and 0.998 with k_h = 1000.
- **Fix:** fix k_h = n/10 in the spec (D-R6), or use a tolerance of 4α/√k_h.

### D-R18. Test-plan items rely on interfaces and counts that the plans do not define (minor)

- **Location:** test plan 2.2 (lines 49, 63), 2.4 item 5 (line 101), 2.5 (line 140);
  implementation plan §2.4.
- **Claim:**
  - (a) The "exact mode" of the couplings is used to build joint tables, but no such
    interface exists in the implementation plan.
  - (b) The meeting-frequency binomial test must count bitwise-equal outputs, not the
    `identical` flag. The E-R2 bug sets the flag correctly and fails only on equality.
  - (c) The exact finite-state expectation sits in the slow tier although its recursion is
    deterministic. It catches weight bugs only if it uses the package's own `lag_weights`
    and `correction_coefficients`.
- **Fix:**
  - Add `discrete_maximal_table(log_p, log_q) -> (K, K)`, or an `exact=True` mode, to the
    plan.
  - Write "count the rows with x' == y' bitwise".
  - Add a fast-tier check that the package coefficients, combined with the exact law of the
    joint chain, reproduce π(h) to 1e-12.

### D-R19. Some requirements have no test (minor)

- **Location:** test plan §3 (lines 264-282).
- **Claim:** R-EXEC-7 maps to 2.9, but 2.9 has no memory test. No test covers:
  - `check(strict=True)`;
  - `w1_bound` (should);
  - R-KER-5 draw counts for kernels other than RWMH;
  - R-EST-3's "does not rely on the coupled kernel" (a kernel that flags `met` without
    copying should still end with equal states, through the engine);
  - worker hygiene (BLAS pinned, SIGINT ignored);
  - the progress fallback.
- **Fix:** add one test for each, or mark the requirement as untested in the mapping.

### D-R20. The Gibbs interface is under-specified (minor)

- **Location:** spec §5.7.
- **Claim:**
  - Does `conditional(x, b)` receive a row of shape (d,) or a batch (n, d)? How are blocks
    given?
  - What is `State.log_prob` when `log_prob=None`, and how are the InputWarning checks done
    without it? Is `tune` skipped?
  - `"crn"` is univariate only, so Biswas et al.'s two-scale coupling, which uses CRN on the
    high-dimensional Gaussian β block, cannot be expressed.
  - `threshold=None` with `coupling="two-scale"` must raise.
- **Fix:** specify each point, and allow `"crn"` for multivariate Normal blocks through
  common standard normals (a published coupling).

### D-R21. Censoring semantics are incomplete (minor)

- **Location:** spec R-EXEC-4 (lines 435-439), R-OUT-1, R-OUT-2, R-DIAG-1.
- **Claim:**
  - Does `max_iter` cap t, the number of coupled steps, or τ − L? With L = k and m = 10k, a
    cap below m censors every replicate.
  - Are censored replicates in `measure`, in `per_replicate` and in `n_replicates`?
  - How does `tv_bound` treat them? Counting them as finite at the cap makes the bound
    anti-conservative (`diagnostics_practice.md` §3).
- **Fix:**
  - Define `max_iter` on t, and validate `max_iter > max(L, m)`.
  - Exclude censored replicates from the measure and from R, but keep them in
    `meeting_times` as inf.
  - Make `tv_bound` report inf, or a lower bound together with the censored count.

### D-R22. Seed normalisation and the scope of reproducibility are unrecorded decisions (minor)

- **Location:** spec §6.1 (lines 412-417), R-API-1.
- **Claim:**
  - The spec does not say whether passing the same Generator twice reproduces a run or
    advances it. M0's `root_seed` already chose `seed_seq.spawn(1)`, which advances. That is
    a decision CLAUDE.md requires to be recorded.
  - Do repeated calls of `sampler.tune()` or `sampler.meeting_times()` reuse namespace
    indices 0 to 99, giving identical pilots, or continue them?
  - Bitwise reproducibility also depends on the NumPy and SciPy versions, because Generator
    distribution algorithms and SciPy `rvs` can change between releases (NEP 19; not
    re-checked here), and on the machine (D-R4).
- **Fix:** record the Generator rule and how pilot indices continue, and state the scope
  "same machine, same library versions".

### D-R23. The spec and the implementation plan disagree in places (minor)

- **Location:** spec R-KER-1 (line 279) against plan §2.3 (line 89); spec §11 toolchain
  against plan §5 (line 216); spec §11 "CI covers 3.11 to the newest release" (line 640)
  against plan §6 (line 247); test plan 2.5.
- **Claim:**
  - `tune(target, initial, rng, ...)` in the spec, `tune(target, initial, root, ...)` in the
    plan.
  - `sphinxcontrib-bibtex` appears only in the plan.
  - The plan's CI matrix stops at Python 3.14. Python 3.15.0 was released on 2026-10-09
    (PEP 790; python.org lists 3.15.0 and 3.15.1), and NumPy 2.5.4 supports 3.12 to 3.15.
  - Test 2.5 compares with "NumPy's inverted-CDF quantiles", but `np.quantile(weights=)`
    needs NumPy 2.0 (the docstring says "versionadded 2.0.0"), while the oldest CI job uses
    1.26.
- **Fix:** align the documents; build the reference test from repeated values instead of
  weights, or skip it on NumPy < 2.

### D-R24. Quantile and isotonic details are unspecified (minor)

- **Location:** spec §3.4 (lines 192-198), R-TUNE-2, D-14; implementation plan
  `weighted_quantile`.
- **Claim:** these points are left open:
  - the method of q̂_β (`np.quantile`'s default "linear" interpolates integer τ, and E01 used
    floor);
  - the isotonic weights per atom (unit, |ω| or gaps);
  - clipping to [0, 1], which `isotonic_regression` does not do;
  - ties between duplicate atoms (from rejections);
  - the inversion rule.

  These are reproducibility issues more than statistical ones.
- **Evidence:**
  - `scipy.optimize.isotonic_regression(y, *, weights=None, increasing=True)` exists in SciPy
    1.12.0: release notes and source at tag v1.12.0, saved in `dl/`.
  - `check_cov_and_quantiles.out`: 200 AR(1) signed measures with 24% negative weights and a
    raw cumulative weight up to 1.287. Unit weights, gap weights and a running maximum gave
    the same q16, q50 and q84. The replicate-bootstrap sd of the median was 0.67.
- **Fix:** specify `method="higher"` (or `"inverted_cdf"`), unit weights after merging tied
  atoms, clipping after the projection, and the left-continuous inverse.

### D-R25. API gaps for emcee users (minor)

- **Location:** spec §8 (lines 553-600), §7.1.
- **Claim:**
  - (a) The kernel strings are not listed, although a test checks that the error lists them.
  - (b) Mixing explicit and "auto" values (for example `k=100, lag="auto"`) has no defined
    meaning.
  - (c) `level` is a confidence level in `estimate()` and `summary()` (0.95) but a credible
    level in `interval()` (0.68).
  - (d) `h` must be batch-first; a scalar `h` is a silent shape error unless checked.
  - (e) `progress` is missing from `CoupledSampler` and `run`.
  - (f) `pool=int` uses forkserver (Linux, Python 3.14 and later) or spawn (macOS, Windows),
    so a `log_prob` defined in a Jupyter notebook fails to pickle. Notebooks are the
    audience's main environment.
  - (g) `Result.kernel` is an object, but `save()` forbids pickle.
  - (h) The README example names the slope `m`, next to the sampler's `m`, and calls
    `plot_tv_bound()`, which needs the plot extra. The test group in `pyproject.toml` does
    not include matplotlib.
  - (i) `kernel="auto"` switches to MALA whenever `grad` is given. From the overdispersed π0
    that the docs recommend, MALA's transient behaviour is sensitive to the step size
    (Christensen et al. 2005, as cited in `mh_couplings.md`; not re-checked).
- **Fix:**
  - List the strings and define the mixing rules.
  - Rename `interval`'s argument (for example `credible=0.68`).
  - Validate the shape of `h`'s output.
  - Add `progress` to `CoupledSampler` and `run`.
  - Ship the target with cloudpickle in pools that telescopic creates.
  - Save kernels as descriptions and load them as descriptions.
  - Rename the slope in the README example.
  - Require `kernel="mala"` explicitly, or document the switch.

### D-R26. Extending R after seeing results is optional stopping (minor)

- **Location:** spec R-API-1 (lines 565-573), R-OUT-1 `merge` (line 493).
- **Claim:** `result = sampler.run(500)  # 500 more replicates` invites a decision based on
  the first result. A sample size chosen from the data gives a stopped mean, which is not
  unbiased in general (inferences in `diagnostics_practice.md` §3 and
  `validation_benchmarks.md`). The guarantee in §3.3 assumes R is fixed in advance.
- **Fix:** add one sentence to §3.3 and the docs: unbiasedness holds for an R fixed before
  results are seen; extending a run keeps consistency and valid large-R intervals.

### D-R27. The default IMH proposal is under-specified (minor)

- **Location:** spec §5.6 (lines 385-386).
- **Claim:**
  - "A multivariate Student-t with 5 degrees of freedom, centred and scaled by the RWMH
    pilot" does not fix the scale: the pilot covariance × (ν − 2)/ν, or an inflated one?
  - Pilot covariances from short adaptive chains often understate the tails, and IMH meeting
    times are geometric only for bounded weights.
- **Fix:** specify the scale (for example c × cov × (ν − 2)/ν with c ≥ 1, recorded as a
  decision), and report the largest weight ratio seen in the pilot.

### D-R28. Pool hygiene and the interrupt rule cannot be applied to user pools (minor)

- **Location:** spec R-EXEC-5 (lines 449-453), R-EXEC-6; implementation plan §2.7.
- **Claim:**
  - Pinning BLAS in an initializer, shipping the target once and ignoring SIGINT are
    possible only for pools that telescopic creates. With a user Executor, the target
    travels with every task.
  - With a `.map` pool, a KeyboardInterrupt interrupts the blocking `map` call, so "finish
    the ones in progress" is impossible: the wave in progress is lost.
  - What a second Ctrl-C returns is not specified.
- **Fix:**
  - State what is guaranteed for each pool type.
  - Wrap each task in `threadpool_limits(1)`, so that pinning also holds in user pools.
  - For `.map` pools, return the completed waves only (a prefix), with the D-R3 label.
  - Define the second Ctrl-C, for example: raise KeyboardInterrupt with the partial result
    attached.

### D-R29. The clock and lanes of budget mode are unspecified (minor; "should" feature)

- **Location:** spec §3.7 (lines 226-233); test plan 2.10.
- **Claim:**
  - Glynn and Heidelberger's per-lane rule needs a lane clock that does not depend on the
    lane's values. In lockstep, a wall-clock deadline depends on how many pairs are coupled,
    in this lane and in others. Lockstep iterations are the safe clock, as the parallel note
    recommends.
  - Pools need long-lived lane tasks, which the chunk model does not have.
- **Fix:** specify the clock (lockstep iterations; seconds only with this caveat) and the
  lane model for pools before implementing.

### D-R30. The memory estimate omits per-atom overhead (nit)

- **Location:** spec R-EXEC-7 (line 463).
- **Claim:** the estimate omits 16 bytes per atom for the weight and the replicate index,
  which doubles the memory at d = 2, and it omits the distance array.

### D-R31. Notation clashes in the normative section (nit)

- **Location:** spec §3 and §5.
- **Claim:**
  - L is the lag in §3, which is normative for code, and the Cholesky factor in §5.2 to §5.5.
  - p is the moment index in §3.3, the dimension of h in §3.4, and a distribution in §5.2.
- **Fix:** rename the Cholesky factor (for example C) and the moment index.

### D-R32. D-14's comparison of monotonisations is inaccurate (nit)

- **Location:** spec D-14 (line 705).
- **Claim:** D-14 says a running maximum is "biased upward", while isotonic regression is
  "the L2 projection". Both monotonised CDFs are biased at finite R and both are consistent.
  `check_cov_and_quantiles.out` found them equal in one case.

### D-R33. Appendix A misattributes a z value (nit)

- **Location:** spec Appendix A, row E-R4 (line 733).
- **Claim:** the row says "pooled completion within a deadline biased (z up to 221)". In
  `exp1_policies.out`, z = 221 belongs to the "first R completed" policy. Pooled
  completed-by-deadline reached z = +211 with 1024 lanes and +165 with 64 lanes.

### D-R34. The pump-failure model has no closed-form π(h) (nit)

- **Location:** test plan 2.12 (line 237).
- **Claim:** the model is listed among targets "with exact π(h)". Exact values exist only
  through one-dimensional quadrature over β, after integrating out λ by Gamma-Poisson
  conjugacy. The plan should say so.

### D-R35. Inferring ndim from a SciPy draw is shape-fragile (nit)

- **Location:** spec R-INPUT-2 (line 257).
- **Claim:** SciPy's `multivariate_normal` with d = 1 returns shape () for `size=1` and
  shape (2,) for `size=2` (`check_scipy_shapes.out`).
- **Fix:** normalise with `atleast_1d` and draw with `size=None`.

### D-R36. The horizon with thinning is unspecified (nit)

- **Location:** spec R-EST-5.
- **Claim:** with `thin > 1` the run could stop at max S instead of m, but the cost formula
  uses m. State which.

### D-R37. The W1 bound needs the distance at s = L (nit)

- **Location:** spec §3.6 and §6.2 (line 429).
- **Claim:** the bound needs ‖X_s − Y_{s−L}‖ for s from L, before the first coupled step,
  to τ − 1. "The distance at each coupled step" may omit s = L.
- **Fix:** include s = L, and state the norm.

### D-R38. Zero-weight atoms are stored (nit)

- **Location:** spec R-EST-6.
- **Claim:** when m − k + 1 < L, or `thin > 1`, many bias-correction atoms have weight 0.
  `check_estimator_algebra.out`: v_t = 0 in 5,472 of 38,400 enumerated (t, k, m, L) cells.
- **Fix:** skip storing them, and call the atom count an upper bound.

## Departures from the census report

- **D-12, atoms stored by default.** The stated reason holds at emcee scale:
  256 × 900 × 10 × 8 B = 18.4 MB.
- **D-10, only the per-lane GH rule for budgets.** A defensible simplification.
- **The interrupt result.** The report wanted no estimate unless a valid rule applies. This
  departure has no decision entry, and its stated reason does not hold (D-R3).
- **The pilot cap.** The note's "generous cap" was dropped without a decision entry (D-R7).
- **Checkpoints.** HDF5 or Zarr became `.npz` save plus a "should" for checkpoints. Fine.
- **Gibbs.** The report recommended built-in deterministic-cost maximal couplings for
  standard families; the spec uses the generic rejection coupling. Its cost variance diverges
  as TV → 0, and `eta` caps it (with D-R2's caveat).

## Checked and found correct

- **Lag weights.** v_t equals the count of l ∈ [k, m] with t − l a positive multiple of L:
  0 mismatches in 20,000 random cases. The L = 1 closed form min(t − k, m − k + 1) holds,
  and v_t ≥ 0. (`check_estimator_algebra.out`)
- **Thinning.** The c_t formula reproduces the definitional average H_S on 4000 random
  synthetic faithful paths (random k, m, L, thin and τ) to 7e-15, and reduces to
  v_t/(m − k + 1) when thin = 1.
- **Signed measure.** The weights sum to 1, π̂(h) = H_S to 1.4e-14, and the atom count is
  |S| + 2 max(0, τ − min S − L).
- **JRSSB reply weight.** It differs from v_t/(m − k + 1) for L ≥ 2 (4,335 enumerated
  cells).
- **Cost formula.** L + 2(τ − L) + max(0, m − τ) matches the horizon max(τ, m).
- **TV identity.** max(0, ⌈(τ − L − t)/L⌉) = #{j ≥ 1 : τ > t + jL} holds pointwise for
  τ < 300, L < 25, t < 320. With AJ24's convention the bound follows from the indicators
  1(X_{t+jL} ≠ Y_{t+(j−1)L}), and it is tighter than with the τ > L convention when
  X_L = Y_0.
- **Derived estimators.** ā āᵀ − S_a/R is the U-statistic, and the covariance estimator is
  unbiased; the plug-in is biased by −Cov(ā) = −C/(nR). Simulation:
  `check_cov_and_quantiles.out`.
- **Reflection-maximal coupling.** At |δ| = 0.5, 1.5 and 3 with d = 10 and a dense Σ, the
  meeting frequency matches 2Φ(−|δ|/2) (binomial p = 0.12 to 0.36), Y's whitened
  coordinates are N(0, 1) (Bonferroni KS p ≥ 0.55), and the flag equals bitwise equality
  when Y is copied.
- **Rejection coupling.** E[draws] = 2, and the draw-count variance is 2(1 − TV)/TV (13.4
  measured, 13.1 theory). With η: P(meet) = ∫min(ηp, q) (0.8365 measured, 0.8356 theory),
  the variance is 2c/(1 − c) ≤ 2η/(1 − η), and Y ~ q.
- **HMC whitening.** With ν = Lᵀp and p ~ N(0, Σ⁻¹), the kinetic energy is ½‖ν‖², and the
  updates ν += (ε/2)Lᵀ∇log π, x += εLν are leapfrog with mass matrix Σ⁻¹.
- **MALA and IMH couplings.** The MALA proposal covariance ε²Σ is shared, so the reflection
  coupling applies. A shared proposal and uniform for IMH is kernel-maximal (OWJ21).
- **D-11.** All maximal couplings share the law on the diagonal (O21 Lemma 3.1), so the
  reflection and rejection meeting components have the same one-step meeting probability.
- **Validity of the main constructions.** The HMC mixture coin, the Gibbs two-scale switch,
  pilots on separate namespaces, the IMH proposal fitted from a pilot, refill, censoring as
  specified, and per-replicate buffered streams with variable draw counts are all valid.
  Each replicate's numbers depend only on its own requests, and discarding buffered numbers
  does not depend on their values. The `choose_kml` floor k ≥ 1 is right.
- **E01.** It reproduces its recorded figures exactly, and its estimates are unbiased
  (|z| < 1).
- **SciPy floor.** `scipy.optimize.isotonic_regression` exists in SciPy 1.12.0 with the
  signature `(y, *, weights=None, increasing=True)` and returns `x`, `weights` and `blocks`.
- **Streams and floors.** The SeedSequence spawn_key construction matches the spawn tree,
  and PCG64DXSM is available in NumPy 1.26.
- **§3.3 moment arithmetic** matches Douc et al. Theorem 17 as summarised in the notes.
- **Cited census figures.** O'Leary's 30 vs 279, E-R1's 32.3 and 274, E-R2's 0.2% vs 94%,
  E-R3's 0 meetings and D-8's 1.4× match the notes.
- **Coverage near nominal.** From near-stationary starts, coverage at R = 200 under the
  recipe is 0.939 to 0.951 (AR(1)).
- **Interrupt bias at many lanes.** The interrupt-prefix bias is undetectable with 16 or more
  lanes in the toy.

## Not verified

- The numbering of Andrieu & Thoms (2008) "Algorithm 4".
- The JRSSB reply's printed weight; the census caveat about its transcription stands.
- Whether the BLAS thread count changes row results at d > 100. Not observed at d ≤ 100
  with 1 and 4 threads.
- Whether the n = 1 matmul path differs on other machines or BLAS builds. Measured here
  only on OpenBLAS 0.3.34 (SkylakeX).
- Whether Heng & Jacob's Theorem 2 needs independent residuals in the meeting step. I
  expect it uses only the meeting probability (D-11), but did not check.
- NEP 19's statement on Generator stream stability across NumPy versions.
- MALA's transient behaviour from overdispersed starts.

## Responses (2026-10-23)

Every finding was accepted. The spec, the test plan and the implementation plan were revised
in one commit; decisions D-20 to D-27 are new, and D-3, D-5, D-7, D-8, D-13, D-14 and D-16
were revised.

| Finding | Response |
|---|---|
| D-R1 | Fixed differently from the main proposal: kernels no longer return a flag. The engine compares positions exactly after each coupled step and once at $t = L$ (R-EST-2, R-KER-3, D-7); couplings still copy $X$'s proposal on the meeting branch. The M2 kernels and the M3 engine follow this. |
| D-R2 | Fixed in §5.7: exact comparison by the engine, the identity coupling for identical conditionals, multivariate CRN, the interface details. |
| D-R3 | Fixed in §6.5 and D-22: interrupted results are labelled consistent, not exactly unbiased; running chunks are awaited; completing $R$ restores exactness. No bias-documenting test: a test that asserts a bias catches no bug in this code. |
| D-R4 | Fixed by R-KER-6 and D-20 (`einsum` with precomputed inverse factors), a batch-of-one test per kernel, and the scope of R-EXEC-2. |
| D-R5 | Fixed in R-TUNE-2 and §3.6: inverted-CDF quantile, $\beta \le 1 - 1/R_0$, the measured TV bound at $k$ reported by the main run, the pilot-size trade-off, E01 described as it ran. |
| D-R6 | Fixed in §7.4 and D-16: warning 2 at 0.25, warning 3 for $R \ge 100$, warning 4 on $\tau - L + 1$ with $k_h = \max(10, R/10)$ for $R \ge 500$; false-alarm rates measured in M9. |
| D-R7 | Fixed in §6.3 and D-23: a fixed pilot cap of $10^5$ (a cap from running quantiles would depend on completion order), censored pairs sorted last, an error when $\hat q_\beta$ is censored, and a slow-pair threshold of $10\max(L, m)$ that needs no pilot. |
| D-R8 | Fixed in test plan 2.4: one-sample tests in $d = 1$, two-sample tests against `step` for $d > 1$. The M2 tests already did this. |
| D-R9 | Fixed in test plan 2.12 and §7.4: rates recorded, not asserted; the docs say the failure is undetectable at moderate $R$. |
| D-R10 | Fixed in R-TUNE-4 and §7.3: the measured inefficiency is reported. |
| D-R11 | Fixed in R-EST-1: loop while $t < \max(\tau, m)$. |
| D-R12 | Fixed in §5.3 to §5.5 and D-24: divergence rules, $\epsilon n \approx 1$, the whitened meeting component recorded, `meeting_weight`. |
| D-R13 | Fixed in §10: the target is restated per $d$. |
| D-R14 | Fixed in R-CPL-1. The M1 code already computed $e$ only where $\lVert\delta\rVert > 0$. |
| D-R15 | Fixed in test plan §1: one rule, sequential escalation. |
| D-R16 | Fixed in test plan 2.12: 2000 intervals, 4 SE around 0.945. |
| D-R17 | Fixed in test plan 2.8: $k_h = n/10$, tolerance $4\alpha/\sqrt{k_h}$. |
| D-R18 | Fixed in test plan 2.2, 2.4 and 2.5: exact tables computed in the tests (no exact mode in the package), counts of bitwise-equal rows, and a fast exact finite-state test with the package's weights. |
| D-R19 | Fixed in the test plan: `check(strict=True)`, `w1_bound`, draw counts, a splitting kernel through the engine, worker hygiene, the progress fallback and the atom count each have a test. |
| D-R20 | Fixed in §5.7. |
| D-R21 | Fixed in §6.3, §7.1 and §3.6: `max_iter` caps $t$ and must exceed $\max(L, m)$; censored pairs are excluded from the measure and $R$, listed as `inf`, and make the TV bound a lower bound. |
| D-R22 | Fixed in §6.1 and D-26: a `Generator` gives four integers; pilot calls continue their indices; the scope of reproducibility is stated. |
| D-R23 | Fixed: `tune(..., rng, ...)` in both documents, `sphinxcontrib-bibtex` dropped, CI to the newest Python, the quantile reference test built from repeated values. |
| D-R24 | Fixed in §3.4. |
| D-R25 | Fixed in §8.1, §7.1, §6.4, §8.3 and D-3, D-21, D-27: kernel strings, partial `"auto"`, `interval(mass=...)`, output-shape checks, `progress` everywhere, `cloudpickle` for created pools, kernels saved as descriptions, the README slope renamed, and `"auto"` = RWMH with or without a gradient. |
| D-R26 | Fixed in §3.3. |
| D-R27 | Fixed in §5.6 and D-25. |
| D-R28 | Fixed in §6.4 and §6.5: guarantees per pool type, `threadpool_limits(1)` per task, waves for `.map` pools, the second Ctrl-C raises. |
| D-R29 | Fixed in §3.7: lockstep iterations as the clock; budgets in-process only for v0.1. |
| D-R30 to D-R38 | Fixed in §6.6, §5 (Cholesky factor renamed $C$), D-14, Appendix A, §9, R-INPUT-2, R-EST-5, §3.6 and R-EST-6. D-R35 was already handled by `initial._Frozen`; D-R37 and D-R38 by the M3 engine. The moment index of §3.3 is unchanged: it does not clash within that section. |

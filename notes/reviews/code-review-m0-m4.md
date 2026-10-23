# Code review of M0 to M4

**Scope:** `src/` and `tests/` at 99768aa (M0 to M4 with their `/simplify` fixes), by
`/code-review` at high effort (DECISIONS #3: one pass for M0 to M3; M4 was committed and
came along). 22 findings; one high-severity bug.

## Fixed

1. **High: a kernel whose dimension differs from the target's ran silently** with wrong
   results (einsum and NumPy broadcasting stretched a size-1 axis). Kernels now report
   `ndim` (from `cov` or the proposal), `init_state` refuses points of another
   dimension, and `Mixture` refuses components of different dimensions.
2. `Mixture` accepted NaN or infinite weights.
3. `tv_bound` built a `(len(t), R)` array, 8 GB for one slow pair at τ = 10⁵ and
   R = 10⁴. It now sums the empirical survival function at lagged times, in
   O(R + max τ), with the same helper as `w1_bound`.
4. The TV report lacked the confidence bound of spec §3.6 where the estimate is 0. The
   result now gives the late fraction (the bound's leading term at k), counting censored
   pairs as late, with its one-sided 95% Clopper-Pearson upper bound; spec §3.6 says so.
5. `merge` accepted parts that share replicates, which would double one replicate's
   weights; it now raises. `Result.merge` (spec R-OUT-1, test plan 2.7) checks the
   configuration; a test checks that two merged runs equal one run over both.
6. A gradient-free kernel whose states carry gradients could cache a non-finite gradient
   at an accepted point. `evaluate` now raises there, as `init_state` did; spec
   R-INPUT-1 says so.
8. `per_replicate` failed on a measure without atoms, and `cov()` divided by zero before
   checking the number of replicates; the estimates now require their minimum number of
   replicates first.
10. The covariance symmetry check used an absolute tolerance of 1e-8, which any
    small-scale matrix passes, and called NaN asymmetric. It is now relative, after a
    finiteness check.
11. `Estimate` and `SignedMeasure` compared arrays with `==` and were unhashable; they now
    compare by identity (`eq=False`).
12. The default times of the bounds included one past the last positive value.
13. Replicate spawn keys could equal those of a caller's own grandchildren of `root`.
    They now live under a fixed branch, `root.spawn_key + (0x74656C, ns, i)` (spec §6.1).
14. `var()` and `std()` computed the full covariance for its diagonal; they now use
    `unbiased_variance` in O(N d).
16. The late fraction was computed twice; `issues` now reads `diagnostics()`.
17. Public inputs are validated: `histogram`'s `dim` and `bins`, negative times of the
    bounds, empty `Empirical`, non-positive values in `hill`.
18. `Independence` documents its draws per step (R-KER-5).
19. The helpers of `kernels.base` have NumPy-style docstrings.
21. A default argument was passed to `np.searchsorted`.
22. Private functions in `_random.py` had docstrings; they now have comments.

## Not changed

7. `merge` sums the parts' seconds. That is the compute time, which is what "seconds
   per replicate" should divide; M7's sampler will measure wall time itself when pools
   run chunks in parallel.
9. `==` treats +0.0 and -0.0 as equal. They are the same real number, so the chains are
   at the same point and the coupling's guarantees hold; spec R-EST-2 now says `==`
   instead of "bitwise".
15. `Replicates` does not carry its configuration. The sampler builds each `Result` with
    its run's settings, and `Result.merge` compares configurations.
20. `couplings._log_uniform` reads a row's generator inside the rejection loop, which
    `Draws.log_uniform` cannot; both reuse reviewers of M1 and M2 kept it.

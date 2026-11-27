# Warmed uncoupled variance references

The comparison runner can estimate the asymptotic variance of the frozen
uncoupled kernel when a target lacks independent exact posterior draws.
Provide `--reference-seed` and a positive `--reference-warmup` alongside
the fixed-size estimator settings. This path uses the optional
`arviz-stats` diagnostics dependency.

Each chain starts independently from the declared wide distribution. The
runner checks the rank-normalized split R-hat, bulk ESS and tail ESS on
every coordinate. It also compares the scored posterior means with their
independent references and compares batch-means variance estimates at two
lengths. A failed check leaves the asymptotic-variance normalization
unavailable. Reference evaluations include warmup and retained draws and
are recorded separately from comparison costs.

The initial validation used four chains with 3,000 warmup transitions and
60 batches of 200 retained transitions per chain. Both frozen kernels passed
on six-dimensional logistic regression and the supernova case. At dimension
51, the parallax reference failed for both kernels. For random walk, the worst
split R-hat was 1.042 and the minimum bulk ESS was 140. For jittered HMC those
values were 4.408 and 4. Neither result supplied a denominator.

The validation output in `results/warmed-reference-small.json` records the
case-by-case costs, estimates and diagnostics.

A larger validation used the same reference lengths and seeds with frozen
random walk on logistic regression at dimension 300 and the parallax
hierarchy at dimension 501. Both failed. The worst split R-hat values were
2.054 and 2.973 and the minimum bulk ESS values were 5.3 and 4.5, respectively.
See `results/warmed-reference-large.json` for these diagnostics. This trial
does not evaluate HMC at those dimensions.

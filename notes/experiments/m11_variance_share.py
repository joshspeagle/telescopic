"""The variance-share warning about the atoms' mean on the banana (M11, DECISIONS #54).

m9_variance_warning.py took the share in the variances of the estimates of the means and
of the second moments about 0. The warning now takes the second moments about the atoms'
mean. On the same 200 seeds of the banana from U[-5, 5]^2 at R = 100: the coverage of the
95% intervals of the means and second moments, in the runs whose largest share is above
and below 0.5, by each definition. The recorded output is a rerun with the Student-t
intervals of issue #84, on the same runs (the shares are unchanged); with the normal
quantile, the runs above 0.5 about the mean covered E[x0^2] at 0.892 against 0.963. Its
first line records the setup: the commit, the machine, the thread settings and the
versions (issue #85).

Run: OMP_NUM_THREADS=1 python notes/experiments/m11_variance_share.py (about six minutes
on one core)
"""

import warnings

import numpy as np
from i85_setup import setup

import telescopic
from telescopic import diagnostics, examples

print(setup())
banana = examples.banana()
truth = np.concatenate([banana.mean, np.diag(banana.cov) + banana.mean**2])
covered, about_zero, about_mean = [], [], []
for seed in range(200):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = telescopic.sample(banana.log_prob, banana.initial, 100, vectorize=True, rng=seed)
    estimate = result.estimate(lambda x: np.concatenate([x, x**2], axis=1))
    low, high = estimate.interval
    covered.append((low <= truth) & (truth <= high))
    about_zero.append(np.max(diagnostics.variance_share(estimate.replicates)))
    about_mean.append(result.diagnostics()["variance_share"])
covered = np.array(covered)
print("all 200 runs: coverage of E[x0], E[x1], E[x0^2], E[x1^2]", covered.mean(axis=0).round(3))
for name, shares in (("about 0", np.array(about_zero)), ("about the mean", np.array(about_mean))):
    above = shares > 0.5
    print(
        f"{name}: median share {np.median(shares):.3f}, above 0.5 in {above.sum()} runs, "
        f"coverage {covered[above].mean(axis=0).round(3)}; below: "
        f"{covered[~above].mean(axis=0).round(3)}"
    )

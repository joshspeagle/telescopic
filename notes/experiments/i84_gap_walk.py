"""The audit's gap example: a random walk on two intervals, at three proposal widths (#84).

The target is the uniform law on [-2, -1] and [1, 2], pi_0 = U[1, 2], h(x) = 1(x > 0),
whose truth is 0.5; k = L = 20, m = 200, R = 2000, `RandomWalk` with proposal standard
deviation sd. A Gaussian step crosses the gap with positive probability (below
Phi(-2 / sd) per step), so the chain converges and the estimator is unbiased (JOA20,
Prop. 3.1). The question is what a run shows. At sd = 0.3 and 0.5 no replicate crosses:
every estimate is 1, with no issue. At sd = 0.6 one replicate crosses, its pair meets at
t = 7316, its estimate is about -360, and the run gives 0.79 +- 0.18 with the variance
issue: the rare replicates carry the correction toward 0.5.

Run: python notes/experiments/i84_gap_walk.py
"""

import warnings

import numpy as np
from scipy import stats

import telescopic
from telescopic.kernels import RandomWalk


def log_prob(x):
    a = np.abs(x[:, 0])
    return np.where((a >= 1) & (a <= 2), 0.0, -np.inf)


warnings.simplefilter("ignore")  # the issues are printed instead
for sd in (0.3, 0.5, 0.6):
    sampler = telescopic.CoupledSampler(
        log_prob,
        telescopic.initial.Uniform([1.0], [2.0]),
        vectorize=True,
        kernel=RandomWalk(cov=[[sd**2]], scale=1.0),
        rng=0,
    )
    result = sampler.run(2000, k=20, lag=20, m=200)
    estimate = result.estimate(lambda x: x[:, 0] > 0)
    tau = result.meeting_times
    print(
        f"sd {sd}: P(step < -2) = {stats.norm.cdf(-2 / sd):.2g}; {estimate}; "
        f"meeting times mean {tau.mean():.4g}, max {tau.max():.0f}; "
        f"replicates min {estimate.replicates.min():.5g}; "
        f"issues {[issue.kind for issue in telescopic.diagnostics.issues(result)]}"
    )

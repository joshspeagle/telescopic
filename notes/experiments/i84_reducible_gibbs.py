"""A chain that cannot converge from pi_0: Gibbs on two squares (#84).

The target is the uniform law on [0, 1]^2 and [2, 3]^2, two squares that share no
coordinate value, so a Gibbs sampler that draws each coordinate from its full
conditional never leaves the square where it starts. From pi_0 = U([0, 1]^2), with
k = L = 5, m = 50 and R = 500, the question is what the estimate of P(x_0 > 1.5) = 0.5
shows. Every pair meets at t = 6, every replicate is 0, and the run gives 0 +- 0 with no
issue: JOA20's Assumption 2.1 fails, so the estimator is not unbiased.

Run: python notes/experiments/i84_reducible_gibbs.py
"""

import numpy as np
from scipy import stats

import telescopic
from telescopic.kernels import Gibbs


def conditional(x, b):
    return stats.uniform(0.0 if x[1 - b] < 1.5 else 2.0, 1.0)


def log_prob(x):
    def inside(c):
        return ((c >= 0) & (c <= 1)) | ((c >= 2) & (c <= 3))

    same = (x[:, 0] < 1.5) == (x[:, 1] < 1.5)
    return np.where(inside(x[:, 0]) & inside(x[:, 1]) & same, np.log(0.5), -np.inf)


sampler = telescopic.CoupledSampler(
    log_prob,
    telescopic.initial.Uniform([0.0, 0.0], [1.0, 1.0]),
    vectorize=True,
    kernel=Gibbs(conditional, [[0], [1]]),
    rng=0,
)
result = sampler.run(500, k=5, lag=5, m=50)
tau = result.meeting_times
print(
    f"P(x_0 > 1.5), truth 0.5: {result.estimate(lambda x: x[:, 0] > 1.5)}; "
    f"meeting times min {tau.min():.0f}, max {tau.max():.0f}; "
    f"issues {[issue.kind for issue in telescopic.diagnostics.issues(result)]}"
)

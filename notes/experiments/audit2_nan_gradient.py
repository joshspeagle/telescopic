"""Kernels on a target whose gradient is NaN where its density is positive (audit 2).

N(0, 1) with a gradient of NaN beyond x = 1. Estimates of E[x] (exactly 0) and of
P(x > 1) (0.1587) by `sample()` with random-walk Metropolis, MALA and HMC, 1,000
replicates, seed 1. Random-walk states carry no gradient, so it never evaluates one.
Before DECISIONS #59 the gradient kernels rejected a proposal with a non-finite gradient,
and so sampled N(0, 1) restricted to x <= 1; from #59 they raise at it.

Run: python notes/experiments/audit2_nan_gradient.py
"""

import numpy as np

import telescopic
from telescopic.initial import Uniform
from telescopic.kernels import HMC, MALA, RandomWalk


def log_prob(x):
    return -0.5 * np.sum(x**2, axis=1)


def grad(x):
    return np.where(x > 1, np.nan, -x)


kernels = {
    "random walk": RandomWalk(np.eye(1), scale=2.0),
    "MALA": MALA(1.0, cov=np.eye(1)),
    "HMC": HMC(0.5, 4, cov=np.eye(1)),
}
for name, kernel in kernels.items():
    try:
        result = telescopic.sample(
            log_prob,
            Uniform([-1.0], [0.9]),
            1000,
            grad=grad,
            vectorize=True,
            kernel=kernel,
            k=20,
            m=200,
            lag=20,
            rng=1,
        )
    except ValueError as error:
        print(f"{name}: raised ValueError: {error}")
        continue
    mean = result.estimate(lambda x: x[:, 0])
    tail = result.estimate(lambda x: (x[:, 0] > 1).astype(float))
    print(
        f"{name}: E[x] = {mean.value:+.4f} +- {mean.standard_error:.4f} "
        f"(z = {mean.value / mean.standard_error:+.1f}), "
        f"P(x > 1) = {tail.value:.4f} +- {tail.standard_error:.4f}"
    )

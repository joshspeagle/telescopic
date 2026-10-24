"""Where a gradient taken as 0 costs MALA and HMC, and why chains cannot start there (#67).

The target: 0.5 N(-1.5, 1) + 0.5 N(1.5, 1), whose textbook gradient (responsibilities as
ratios of densities) is NaN beyond about |x| = 40, against the same gradient computed by
logsumexp.

1. 200 chains of MALA(1.0) with the mixture's variance, 3.25, as covariance, placed by
   hand at x = 60 with the zeroed gradient, and where they are after 2,000 steps.
2. `sample()` with that kernel from N(45, 1), 100 replicates, k = 10, seeds 0 to 2: the
   estimate of E[x] (exactly 0), or the error.
3. N(0, I_2) with a gradient of NaN where x_0 < 0, against the exact gradient: 100 pairs
   of HMC(0.5, 2) from N((2, 0), 0.01 I), k = m = 0, lag 1, censored at 2,000
   iterations: how many met, and when.

Run: python notes/experiments/audit2_zero_drift_costs.py
"""

import warnings

import numpy as np
from scipy import special, stats

import telescopic
from telescopic.engine import run
from telescopic.initial import Normal
from telescopic.kernels import HMC, MALA, State
from telescopic.target import Target
from telescopic.testing import random_draws

MEANS = np.array([-1.5, 1.5])


def mixture_log_prob(x):
    return special.logsumexp(stats.norm.logpdf(x[:, :1], MEANS, 1.0), axis=1) + np.log(0.5)


def textbook_grad(x):
    densities = stats.norm.pdf(x[:, :1], MEANS, 1.0)
    with np.errstate(invalid="ignore"):
        responsibilities = densities / densities.sum(axis=1, keepdims=True)
    return np.sum(responsibilities * (MEANS - x[:, :1]), axis=1, keepdims=True)


def log_space_grad(x):
    responsibilities = special.softmax(stats.norm.logpdf(x[:, :1], MEANS, 1.0), axis=1)
    return np.sum(responsibilities * (MEANS - x[:, :1]), axis=1, keepdims=True)


for name, grad in (("textbook", textbook_grad), ("log-space", log_space_grad)):
    target = Target(mixture_log_prob, grad=grad, vectorize=True)
    kernel = MALA(1.0, cov=[[3.25]])
    x = np.full((200, 1), 60.0)
    state = State(x, *target.log_prob_and_grad(x))
    for seed in np.random.SeedSequence(1).spawn(2000):
        state = kernel.step(target, state, random_draws(200, seed))[0]
    distance = np.abs(state.x[:, 0])
    print(
        f"MALA from x = 60, {name} gradient: {np.mean(distance < 10):.0%} of chains within "
        f"|x| < 10 after 2,000 steps, median |x| {np.median(distance):.2f}"
    )

for seed in range(3):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = telescopic.sample(
                mixture_log_prob,
                Normal([45.0], [[1.0]]),
                100,
                grad=textbook_grad,
                vectorize=True,
                kernel=MALA(1.0, cov=[[3.25]]),
                k=10,
                rng=seed,
            )
        mean = result.estimate(lambda x: x[:, 0])
        print(
            f"sample() from N(45, 1), seed {seed}: E[x] = {mean.value:.3f} +- {mean.standard_error:.3f}"
        )
    except ValueError as error:
        print(f"sample() from N(45, 1), seed {seed}: raised ValueError: {str(error)[:60]}")


def gaussian_log_prob(x):
    return -0.5 * np.sum(x**2, axis=1)


for name, grad in (
    ("NaN where x_0 < 0", lambda x: np.where(x[:, :1] < 0, np.nan, -x)),
    ("exact", lambda x: -x),
):
    target = Target(gaussian_log_prob, grad=grad, vectorize=True)
    replicates = run(
        target,
        HMC(0.5, 2, cov=np.eye(2)),
        Normal([2.0, 0.0], 0.01 * np.eye(2)),
        range(100),
        k=0,
        m=0,
        lag=1,
        max_iter=2000,
        rng=2,
    )
    met = np.isfinite(replicates.meeting_time)
    print(
        f"HMC(0.5, 2), gradient {name}: {met.sum()} of 100 pairs met by 2,000 iterations, "
        f"median meeting time {np.median(replicates.meeting_time[met])}; "
        f"{replicates.n_grad_zeroed} of {replicates.n_grad} gradients taken as 0"
    )

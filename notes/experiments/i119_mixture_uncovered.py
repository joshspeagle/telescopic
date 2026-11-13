"""States where the proposal is 0 and the target is not, inside a mixture (#119).

The target is N(0, 1) and the independence proposal the uniform law on (-2, 2), as in
i119_uncovered_target.py, where the kernel used alone never reaches |x| >= 2 and
estimates E[x^2] = 1 as the value 0.774 of the restricted target. Here the kernel is a
component of `Mixture([Independence(uniform), RandomWalk([[1.0]])], [0.5, 0.5])`. The
random walk reaches |x| >= 2, where the target is positive and the proposal is 0, and
the independence component rejects every proposal from there. The question is whether
such states arise, and whether the run is then right. `telescopic.sample` runs with its
defaults and 2,000 replicates from the initial distribution U(-1, 1), on seeds 0 to 2,
and estimates P(|x| >= 2) = 2 Phi(-2) = 0.0455 and E[x^2] = 1.

Result (i119_mixture_uncovered.out, whose first line records the commit): the estimates
of P(|x| >= 2) are 0.0448, 0.0473 and 0.0428, with standard errors of 0.0015 to 0.0017,
so chains are at such states as often as the target asks, and the estimates of E[x^2]
are 0.998, 1.014 and 0.982, with standard errors of 0.009 to 0.011. No run warns.

Rejecting every proposal from such a state is the Metropolis-Hastings rule there, with
an acceptance ratio of 0, and it is what keeps the target invariant under the
independence component: a rule that moved the chain off would carry mass out of a set
of positive mass under the target, and this component never proposes a return.

Run: python notes/experiments/i119_mixture_uncovered.py
"""

import subprocess
import warnings

import numpy as np
from scipy import special, stats

import telescopic
from telescopic.kernels import Independence, Mixture, RandomWalk


class Uniform:
    def sample(self, draws):
        return 4 * special.ndtr(draws.normal(1)) - 2

    def log_density(self, x):
        return np.where(np.abs(x[:, 0]) < 2, np.log(0.25), -np.inf)


def log_prob(x):
    return -0.5 * x[:, 0] ** 2


commit = subprocess.run(
    ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
).stdout.strip()
print(f"commit {commit}; P(|x| >= 2) under N(0, 1): {2 * stats.norm.cdf(-2):.4f}")
kernel = Mixture([Independence(Uniform()), RandomWalk([[1.0]])], [0.5, 0.5])
for seed in range(3):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = telescopic.sample(
            log_prob,
            telescopic.initial.Uniform([-1.0], [1.0]),
            2_000,
            vectorize=True,
            kernel=kernel,
            rng=seed,
        )
    outside = result.estimate(lambda x: np.abs(x[:, 0]) >= 2)
    square = result.estimate(lambda x: x[:, 0] ** 2)
    print(
        f"seed {seed}: P(|x| >= 2) {outside}; E[x^2] {square}; "
        f"warnings {[str(w.message) for w in caught]}"
    )

"""An independence proposal that misses part of the target's support (#119).

The target is N(0, 1) and the proposal the uniform law on (-2, 2), which is 0 where
|x| >= 2 and the target is not. Run by `telescopic.sample` with its defaults and 500
replicates, from the initial distribution U(-1, 1), inside the proposal's support, the
question is what the estimate of E[x^2] = 1 shows. A chain moves only to proposals, so
it never visits |x| >= 2, and the kernel samples N(0, 1) restricted to (-2, 2), whose
E[x^2] is 1 - 4 phi(2) / (2 Phi(2) - 1) = 0.774. Second, from a point mass at 3, where
the target is positive and the proposal is 0, the question is what the estimate of
E[x] = 0 shows.

Result (i119_uncovered_target.out): E[x^2] is estimated as 0.778 with a standard error
of 0.0093, 24 standard errors below the truth and 0.5 above the restricted value, and
the run raises no warning and reports no issue. From 3, every proposal is rejected: the estimate of E[x]
is 3 +- 0, every pair meets at the lag, and nothing warns.

Run: python notes/experiments/i119_uncovered_target.py
"""

import warnings

import numpy as np
from scipy import special, stats

import telescopic
from telescopic.kernels import Independence


class Uniform:
    def sample(self, draws):
        return 4 * special.ndtr(draws.normal(1)) - 2

    def log_density(self, x):
        return np.where(np.abs(x[:, 0]) < 2, np.log(0.25), -np.inf)


def log_prob(x):
    return -0.5 * x[:, 0] ** 2


restricted = 1 - 4 * stats.norm.pdf(2) / (2 * stats.norm.cdf(2) - 1)
print(f"E[x^2] under N(0, 1) restricted to (-2, 2): {restricted:.4f}")
for name, init, h in [
    (
        "E[x^2] from U(-1, 1), truth 1",
        telescopic.initial.Uniform([-1.0], [1.0]),
        lambda x: x[:, 0] ** 2,
    ),
    ("E[x] from 3, truth 0", np.full((1, 1), 3.0), lambda x: x[:, 0]),
]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = telescopic.sample(
            log_prob, init, 500, vectorize=True, kernel=Independence(Uniform()), rng=0
        )
    estimate = result.estimate(h)
    tau = result.meeting_times
    issues = [issue.kind for issue in telescopic.diagnostics.issues(result)]
    print(
        f"{name}: {estimate}; meeting times {tau.min():.0f} to {tau.max():.0f}, "
        f"lag {result.lag}; warnings {[str(w.message) for w in caught]}; "
        f"issues {issues}"
    )

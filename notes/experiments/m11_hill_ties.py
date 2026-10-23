"""Hill's estimate on meeting times tied at their smallest value (M11, DECISIONS #55, #57).

When ties make the smallest value the threshold of `diagnostics.hill`, three rules:
"none", Hill's estimate as is; "nan", the package's (the M11 audit's, restored by #57);
"above", the threshold moved to the next larger value (#55, refuted by the adversarial
review). Each row gives the share of runs whose estimate falls below the tail warning's 2,
with its Monte Carlo standard error, and the share of nan, at the warning's k = R/10.

1. Geometric values from 1, which a discrete start allows (X_L = Y_0), and from 2, the
   least value of tau - L + 1 when the chains cannot meet at t = L.
2. A share of slow pairs behind ties at 1 and at 2: heavy, floor(Pareto(1) + base + 1),
   tail index 1; light, base + Geom(0.1) or base + Geom(0.3), every moment finite.
3. Package runs: `sample()` with its defaults and `Independence` with a standard normal
   proposal, from N(0, 1.5^2), on 0.99 N(0, 1) + 0.01 N(3, 0.05^2) and on
   0.99 N(0, 1) + 0.01 U[3, 3.2]. The weights pi/q are bounded, so every moment of the
   meeting times is finite.

Parts 1 and 2 follow the review's scripts (`notes/reviews/m11.md`).
Run: python notes/experiments/m11_hill_ties.py
"""

import warnings

import numpy as np
from scipy import stats

import telescopic
from telescopic.diagnostics import hill
from telescopic.kernels import Independence, MultivariateT


def no_rule(x, k):
    x = np.sort(x)
    mean_log = np.mean(np.log(x[-k:] / x[-k - 1]))
    return np.inf if mean_log == 0 else 1 / mean_log


def above(x, k):
    x = np.sort(x)
    if x[1] == x[-k - 1] == x[0]:
        k = int(np.count_nonzero(x > x[0])) - 1
        if k < 1:
            return np.nan
    return no_rule(x, k)


RULES = (("none", no_rule), ("nan", hill), ("above", above))


def rates(samples, k):
    found = []
    for name, rule in RULES:
        alpha = np.array([rule(x, k) for x in samples])
        below = np.mean(alpha < 2)
        error = np.sqrt(below * (1 - below) / len(alpha))
        found.append(f"{name} {below:.3f} +- {error:.3f} (nan {np.mean(np.isnan(alpha)):.2f})")
    return ", ".join(found)


RUNS, R = 2000, 500
print(f"share of runs below 2 +- standard error (share nan); R = {R}, {RUNS} runs")
rng = np.random.default_rng(1)
for p in (0.5, 0.8, 0.9, 0.92, 0.95, 0.99):
    for base in (1, 2):
        samples = rng.geometric(p, (RUNS, R)) + base - 1.0
        print(f"geometric({p}) from {base}: {rates(samples, R // 10)}")

rng = np.random.default_rng(2)
slow_kinds = {
    "heavy Pareto(1)": lambda n, base: np.floor(rng.pareto(1.0, n) + base + 1),
    "light Geom(0.1)": lambda n, base: base + rng.geometric(0.1, n),
    "light Geom(0.3)": lambda n, base: base + rng.geometric(0.3, n),
}
for base in (1, 2):
    for share in (0.02, 0.05, 0.08):
        for name, slow in slow_kinds.items():
            samples = []
            for _ in range(RUNS):
                n = rng.binomial(R, share)
                samples.append(np.r_[np.full(R - n, float(base)), slow(n, base)])
            print(f"ties at {base}, {share:.0%} slow, {name}: {rates(samples, R // 10)}")


def mixture(component):
    def log_prob(x):
        return np.logaddexp(
            np.log(0.99) + stats.norm.logpdf(x[:, 0]), np.log(0.01) + component(x[:, 0])
        )

    return log_prob


def flat(x):
    return np.where((x >= 3.0) & (x <= 3.2), np.log(1 / 0.2), -np.inf)


targets = {
    "0.01 N(3, 0.05^2)": mixture(lambda x: stats.norm.logpdf(x, 3.0, 0.05)),
    "0.01 U[3, 3.2]": mixture(flat),
}
kernel = Independence(MultivariateT([0.0], [[1.0]], df=None))
for replicates, runs in ((500, 100), (2000, 50)):
    for name, log_prob in targets.items():
        samples = []
        for seed in range(runs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = telescopic.sample(
                    log_prob,
                    stats.norm(0, 1.5),
                    replicates,
                    kernel=kernel,
                    vectorize=True,
                    rng=seed,
                )
            values = result.meeting_times - result.lag + 1
            samples.append(values[np.isfinite(values)])
        smallest = np.mean([np.mean(x == x.min()) for x in samples])
        print(
            f"Independence on 0.99 N(0, 1) + {name}, R = {replicates}, {runs} runs, "
            f"{smallest:.3f} at the smallest value: {rates(samples, replicates // 10)}",
            flush=True,
        )

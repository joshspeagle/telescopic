"""Hill's estimate on meeting times tied at small integers (M11 code review, DECISIONS #55).

Values tied at their smallest made it the threshold of `diagnostics.hill`, and a threshold
of 1 read a step to 2 as a heavy tail. The M11 audit made the estimate nan there; the
code review found that this hides a heavy tail behind the ties. Over 2,000 seeded runs of
R = 500 values and the warning's k = 50, the share of runs below the warning's 2 and the
share of nan, for three rules: none, nan (the audit's) and the package's, which moves the
threshold to the next larger value. Light tails: geometric values, as of a kernel whose
pairs mostly meet at once. A heavy tail: ones, and a binomial share of
floor(Pareto(1) + 2) values, whose tail index is 1.

Run: python notes/experiments/m11_hill_ties.py
"""

import numpy as np

from telescopic.diagnostics import hill


def no_rule(x, k):
    x = np.sort(x)
    mean_log = np.mean(np.log(x[-k:] / x[-k - 1]))
    return np.inf if mean_log == 0 else 1 / mean_log


def nan_rule(x, k):
    return np.nan if np.sort(x)[-k - 1] == np.min(x) else no_rule(x, k)


RULES = (("none", no_rule), ("nan", nan_rule), ("package", hill))
RUNS, R, K = 2000, 500, 50


def rates(draw, seed):
    found = []
    for _, rule in RULES:
        rng = np.random.default_rng(seed)
        alpha = np.array([rule(draw(rng), K) for _ in range(RUNS)])
        found.append(f"{np.mean(alpha < 2):.3f} / {np.mean(np.isnan(alpha)):.3f}")
    return ", ".join(f"{name} {rate}" for (name, _), rate in zip(RULES, found, strict=True))


print("share of runs below 2 / nan, by rule")
for p in (0.5, 0.8, 0.88, 0.9, 0.92, 0.95, 0.99):
    print(f"geometric({p}): {rates(lambda rng, p=p: rng.geometric(p, R).astype(float), 1)}")


def heavy(rng, share):
    n_tail = rng.binomial(R, share)
    return np.r_[np.ones(R - n_tail), np.floor(rng.pareto(1.0, n_tail) + 2)]


for share in (0.05, 0.08, 0.2):
    print(f"ones and {share:.0%} heavy: {rates(lambda rng, s=share: heavy(rng, s), 2)}")

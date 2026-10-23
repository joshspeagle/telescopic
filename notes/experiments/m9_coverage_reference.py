"""Reference coverage of the 95% interval at R = 200 on the exact AR(1) chain (M9).

The design review's estimator (notes/reviews/scripts/design-review/check_coverage.py),
written apart from the package, with 10,000 intervals instead of 1,000. The AR(1) kernel
x' = rho x + sqrt(1 - rho^2) xi (rho = 0.9) leaves N(0, 1) invariant; pairs are coupled by
the reflection-maximal coupling of N(rho x, s^2) and N(rho y, s^2). Recipe values
k = L = 38, m = 380 (from a lag-1 pilot of 2,000 meeting times in the design review).
h(x) = x (truth 0) and x^2 (truth 1). The slow test asserts the stationary start's coverage;
the far start's documents the heavy-tail caveat (test plan 2.12).
Run: python notes/experiments/m9_coverage_reference.py
"""

import numpy as np

RHO, S = 0.9, np.sqrt(1 - 0.81)
K = LAG = 38
M = 380
R, INTERVALS = 200, 10_000


def run(rng, n, mu0, sd0):
    x = mu0 + sd0 * rng.standard_normal(n)
    y = mu0 + sd0 * rng.standard_normal(n)
    count = M - K + 1
    h1, h2 = np.zeros(n), np.zeros(n)
    t = 0
    for _ in range(LAG):
        t += 1
        x = RHO * x + S * rng.standard_normal(n)
        if K <= t <= M:
            h1 += x / count
            h2 += x * x / count
    met = np.zeros(n, bool)
    while t < M or not met.all():
        t += 1
        z = rng.standard_normal(n)
        log_u = np.log(rng.random(n))
        delta = (RHO * x - RHO * y) / S
        identical = ~met & (log_u < -z * delta - 0.5 * delta**2)
        x_new = RHO * x + S * z
        y_new = np.where(identical | met, x_new, RHO * y - S * z)
        x = x_new
        if K <= t <= M:
            h1 += x / count
            h2 += x * x / count
        still = ~met & ~identical
        if t >= K + LAG:
            v = (t - K) // LAG - int(np.ceil(max(LAG, t - M) / LAG)) + 1
            if v > 0:
                h1 += np.where(still, v * (x - y_new), 0.0) / count
                h2 += np.where(still, v * (x * x - y_new * y_new), 0.0) / count
        met |= identical
        y = np.where(met, x, y_new)
    return h1, h2


rng = np.random.default_rng(2026)
for label, mu0, sd0 in (
    ("stationary N(0, 1)", 0.0, 1.0),
    ("near-stationary N(1, 1)", 1.0, 1.0),
    ("far N(3, 1), recorded, not asserted", 3.0, 1.0),
):
    h1, h2 = run(rng, R * INTERVALS, mu0, sd0)
    print(f"{label}: k = L = {K}, m = {M}, {INTERVALS} intervals of R = {R}")
    for name, values, truth in (("x", h1, 0.0), ("x^2", h2, 1.0)):
        blocks = values.reshape(INTERVALS, R)
        estimate = blocks.mean(axis=1)
        error = blocks.std(axis=1, ddof=1) / np.sqrt(R)
        coverage = np.mean(np.abs(estimate - truth) <= 1.959964 * error)
        se = np.sqrt(coverage * (1 - coverage) / INTERVALS)
        print(f"  h = {name:3s}: coverage {coverage:.4f} (binomial se {se:.4f})")

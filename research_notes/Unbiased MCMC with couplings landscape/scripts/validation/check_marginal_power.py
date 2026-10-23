"""Power of marginal-correctness tests for one RWMH transition from a fixed state (scratch).

Target N(0,1); current state x = 1; proposal N(x, sigma^2), sigma = 2.4.
The exact one-step law is an atom at x with mass r(x) = 1 - a(x) plus a continuous part
with density q(x'|x) min(1, pi(x')/pi(x)).
Designs (alpha = 1e-4 per design, Bonferroni within a design):
  A. exact one-sample: binomial test of the stay probability vs r(x), and one-sample KS of
     the moved values vs the exact normalised continuous-part CDF (quadrature).
  B. two-sample KS between n coupled-kernel X-draws and n uncoupled reference draws.
  C. two-sample Anderson-Darling (k-sample) between the same samples.
Bug: the coupled kernel's X-marginal uses sigma_bug = 1.05 * sigma.
"""
import numpy as np
from scipy import integrate, stats

rng = np.random.default_rng(1)
x, sigma, alpha = 1.0, 2.4, 1e-4
logpi = lambda v: -0.5 * v**2
dens = lambda v: stats.norm.pdf(v, x, sigma) * np.minimum(1.0, np.exp(logpi(v) - logpi(x)))
a_x = integrate.quad(dens, x - 15 * sigma, x + 15 * sigma, points=[-1.0, 1.0], limit=200)[0]
grid = np.linspace(x - 12 * sigma, x + 12 * sigma, 400_001)
cdf_grid = np.concatenate([[0.0], np.cumsum(0.5 * (dens(grid[1:]) + dens(grid[:-1])) * np.diff(grid))])
cdf_grid /= cdf_grid[-1]
cont_cdf = lambda v: np.interp(v, grid, cdf_grid)
print(f"exact acceptance prob from x={x}: a(x)={a_x:.6f}")


def transitions(n, s):
    prop = x + s * rng.normal(size=n)
    acc = np.log(rng.uniform(size=n)) <= logpi(prop) - logpi(x)
    return np.where(acc, prop, x)


def design_A(v):
    stay = v == x
    p1 = stats.binomtest(int(stay.sum()), v.size, 1 - a_x).pvalue
    p2 = stats.kstest(v[~stay], cont_cdf).pvalue
    return min(1.0, 2 * min(p1, p2))


def design_B(v):
    return stats.ks_2samp(v, transitions(v.size, sigma)).pvalue


def design_C(v):
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        return stats.anderson_ksamp([v, transitions(v.size, sigma)]).pvalue  # p capped to [0.001, 0.25]


for n in [10_000, 100_000]:
    for label, s in [("correct", sigma), ("bug 5% scale", 1.05 * sigma)]:
        reps = 100
        rej = {"A": 0, "B": 0, "C(alpha=1e-3 floor)": 0}
        for _ in range(reps):
            v = transitions(n, s)
            rej["A"] += design_A(v) < alpha
            rej["B"] += design_B(v) < alpha
            rej["C(alpha=1e-3 floor)"] += design_C(v) <= 1e-3
        print(f"n={n:6d} {label:13s}: " + "  ".join(f"{k}: {v / reps:.2f}" for k, v in rej.items()))

"""Numerical checks of maximal couplings of Gaussians (scratch research script).

1. Meeting probability of Thorisson's rejection maximal coupling (Jacob et al. 2020,
   Algorithm 2) and of the reflection-maximal coupling equals 1 - TV(p, q), with
   TV = 2 Phi(Delta/2) - 1 for equal covariances (Delta = Mahalanobis distance).
2. Marginals are exact (one-sample KS against the exact normal CDF).
3. Cost of Algorithm 2: mean 2 "units" for any (p, q); variance grows as TV -> 0.
4. Power of tests against a plausible bug: residual of Y drawn from q directly
   (forgetting the rejection loop), which breaks the Y marginal.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261023)


def rejection_maximal_1d(mu1, mu2, sigma, rng, buggy=False):
    """Return (x, y, n_units). One unit = one draw + two density evaluations."""
    p = stats.norm(mu1, sigma)
    q = stats.norm(mu2, sigma)
    x = p.rvs(random_state=rng)
    units = 1
    if rng.uniform() * p.pdf(x) <= q.pdf(x):
        return x, x, units
    if buggy:  # BUG: forget the rejection loop, draw the residual from q directly
        return x, q.rvs(random_state=rng), units + 1
    while True:
        y = q.rvs(random_state=rng)
        units += 1
        if rng.uniform() * q.pdf(y) > p.pdf(y):
            return x, y, units


def rejection_maximal_1d_vec(mu1, mu2, sigma, n, rng, buggy=False):
    """Vectorised version (loop over the rejection step) for speed."""
    lp = lambda z: stats.norm.logpdf(z, mu1, sigma)
    lq = lambda z: stats.norm.logpdf(z, mu2, sigma)
    x = rng.normal(mu1, sigma, n)
    units = np.ones(n, dtype=np.int64)
    met = np.log(rng.uniform(size=n)) + lp(x) <= lq(x)
    y = x.copy()
    todo = np.flatnonzero(~met)
    if buggy:
        y[todo] = rng.normal(mu2, sigma, todo.size)
        units[todo] += 1
        return x, y, units, met
    while todo.size:
        cand = rng.normal(mu2, sigma, todo.size)
        units[todo] += 1
        acc = np.log(rng.uniform(size=todo.size)) + lq(cand) > lp(cand)
        y[todo[acc]] = cand[acc]
        todo = todo[~acc]
    return x, y, units, met


def reflection_maximal(mu1, mu2, L, n, rng):
    """Bou-Rabee et al. reflection-maximal coupling of N(mu1, LL^T), N(mu2, LL^T)."""
    d = mu1.size
    z = np.linalg.solve(L, mu1 - mu2)
    e = z / np.linalg.norm(z)
    xdot = rng.normal(size=(n, d))
    logu = np.log(rng.uniform(size=n))
    # s standard normal: log s(xdot+z) - log s(xdot)
    logratio = -0.5 * np.sum((xdot + z) ** 2, axis=1) + 0.5 * np.sum(xdot**2, axis=1)
    met = logu <= np.minimum(0.0, logratio)
    ydot = np.where(met[:, None], xdot + z, xdot - 2.0 * (xdot @ e)[:, None] * e)
    x = mu1 + xdot @ L.T
    y = mu2 + ydot @ L.T
    return x, y, met


def tv_equal_cov(delta):
    return 2.0 * stats.norm.cdf(delta / 2.0) - 1.0


print("=== 1. Meeting probability = 1 - TV (equal covariance) ===")
n = 200_000
for delta in [0.1, 0.5, 1.0, 2.0, 4.0]:
    x, y, units, met = rejection_maximal_1d_vec(0.0, delta, 1.0, n, rng)
    p_exact = 1.0 - tv_equal_cov(delta)
    k = int(np.sum(x == y))
    bt = stats.binomtest(k, n, p_exact)
    print(f"[Alg2 1-D] Delta={delta:4.1f} exact P(meet)={p_exact:.5f} "
          f"empirical={k / n:.5f} binom p={bt.pvalue:.3f} "
          f"KS(x) p={stats.kstest(x, stats.norm(0, 1).cdf).pvalue:.3f} "
          f"KS(y) p={stats.kstest(y, stats.norm(delta, 1).cdf).pvalue:.3f} "
          f"mean units={units.mean():.4f} var units={units.var():.2f}")

d = 10
A = rng.normal(size=(d, d))
Sigma = A @ A.T / d + np.eye(d)
L = np.linalg.cholesky(Sigma)
mu1 = np.zeros(d)
for scale in [0.2, 1.0, 3.0]:
    mu2 = scale * rng.normal(size=d) / np.sqrt(d)
    delta = np.linalg.norm(np.linalg.solve(L, mu1 - mu2))
    x, y, met = reflection_maximal(mu1, mu2, L, n, rng)
    p_exact = 1.0 - tv_equal_cov(delta)
    k = int(np.sum(np.all(x == y, axis=1)))
    bt = stats.binomtest(k, n, p_exact)
    # marginal checks: whitened coordinates should be iid N(0,1); test a few projections
    wx = np.linalg.solve(L, (x - mu1).T).T
    wy = np.linalg.solve(L, (y - mu2).T).T
    pk = [stats.kstest(wx[:, j], "norm").pvalue for j in range(3)] + \
         [stats.kstest(wy[:, j], "norm").pvalue for j in range(3)]
    print(f"[Refl-max d=10] Mahalanobis={delta:.3f} exact P(meet)={p_exact:.5f} "
          f"empirical={k / n:.5f} binom p={bt.pvalue:.3f} min KS p (6 projections)={min(pk):.3f}")

print("\n=== 2. Cost of Algorithm 2 as TV -> 0 (units: one draw + two evals) ===")
for delta in [2.0, 0.5, 0.1, 0.02]:
    _, _, units, _ = rejection_maximal_1d_vec(0.0, delta, 1.0, 400_000, rng)
    print(f"Delta={delta:5.2f} TV={tv_equal_cov(delta):.4f} mean={units.mean():.4f} "
          f"sd={units.std():.2f} max={units.max()}")

print("\n=== 3. Detecting a 'forgot the rejection loop' bug in the Y marginal ===")
# Exact KS distance between buggy Y law and q = N(Delta, 1):
# buggy density = min(p, q) + TV * q.
grid = np.linspace(-8, 10, 200_001)
for delta in [0.5, 1.0, 2.0]:
    tv = tv_equal_cov(delta)
    pdf_p = stats.norm.pdf(grid, 0, 1)
    pdf_q = stats.norm.pdf(grid, delta, 1)
    buggy = np.minimum(pdf_p, pdf_q) + tv * pdf_q
    cdf_b = np.cumsum(buggy) * (grid[1] - grid[0])
    ks_dist = np.max(np.abs(cdf_b - stats.norm.cdf(grid, delta, 1)))
    # simulation-based power at alpha = 1e-4, one-sample KS against exact CDF
    out = []
    for nn in [1_000, 10_000, 100_000]:
        rej = 0
        reps = 200
        for _ in range(reps):
            _, yb, _, _ = rejection_maximal_1d_vec(0.0, delta, 1.0, nn, rng, buggy=True)
            rej += stats.kstest(yb, stats.norm(delta, 1).cdf).pvalue < 1e-4
        out.append(f"n={nn}: power={rej / reps:.2f}")
    print(f"Delta={delta}: TV={tv:.3f} KS distance(buggy, q)={ks_dist:.4f}; " + "; ".join(out))

"""Floating-point check: reflection-maximal coupling must copy X into Y on the meeting event."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(7)
n, d = 200_000, 10
A = rng.normal(size=(d, d))
L = np.linalg.cholesky(A @ A.T / d + np.eye(d))
mu1 = np.zeros(d)
for scale in [0.2, 1.0, 3.0]:
    mu2 = scale * rng.normal(size=d) / np.sqrt(d)
    z = np.linalg.solve(L, mu1 - mu2)
    delta = np.linalg.norm(z)
    e = z / delta
    xdot = rng.normal(size=(n, d))
    logratio = -0.5 * np.sum((xdot + z) ** 2, 1) + 0.5 * np.sum(xdot**2, 1)
    met = np.log(rng.uniform(size=n)) <= np.minimum(0.0, logratio)
    ydot = np.where(met[:, None], xdot + z, xdot - 2.0 * (xdot @ e)[:, None] * e)
    x = mu1 + xdot @ L.T
    y_naive = mu2 + ydot @ L.T
    y_fixed = np.where(met[:, None], x, y_naive)
    p_exact = 2.0 * stats.norm.cdf(-delta / 2.0)
    bitwise = np.mean(np.all(x == y_naive, 1))
    close = np.mean(np.all(np.isclose(x, y_naive, rtol=0, atol=1e-12), 1))
    fixed = np.mean(np.all(x == y_fixed, 1))
    maxerr = np.max(np.abs(x[met] - y_naive[met]))
    k = int(np.sum(np.all(x == y_fixed, 1)))
    print(f"Mahalanobis={delta:.3f} exact P(meet)={p_exact:.5f} bitwise(naive)={bitwise:.5f} "
          f"isclose(naive)={close:.5f} bitwise(fixed)={fixed:.5f} binom p(fixed)={stats.binomtest(k, n, p_exact).pvalue:.3f} "
          f"max |x-y| on met (naive)={maxerr:.2e}")

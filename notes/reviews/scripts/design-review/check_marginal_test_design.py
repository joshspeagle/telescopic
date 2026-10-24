"""Test plan 2.4 item 2 for d > 1: 'KS tests on whitened coordinates' of the moved values.

A correct RWMH step on N(0, I_3) from a fixed x0: the moved values are accepted proposals,
whose law q(x'|x0) a(x0, x')/(1 - r(x0)) is not Gaussian. KS of the whitened increments
(x' - x0)/sigma against N(0, 1) therefore rejects a correct kernel. Also the stay
probability r(x0) needs a d-dimensional integral, not one-dimensional quadrature.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(61)
d, n = 3, 20_000
sigma = 2.38 / np.sqrt(d)
x0 = np.array([1.0, -0.5, 2.0])
z = rng.standard_normal((n, d))
prop = x0 + sigma * z
logr = -0.5 * np.sum(prop**2, 1) + 0.5 * np.sum(x0**2)
acc = np.log(rng.random(n)) < logr
moved = (prop[acc] - x0) / sigma
p = [stats.kstest(moved[:, j], "norm").pvalue for j in range(d)]
print(f"correct kernel, d={d}, n={n}: acceptance {acc.mean():.3f}; KS p-values of whitened moved "
      f"coordinates vs N(0,1): {np.array(p)}")
print(f"KS of all whitened increments including stays (a point mass at 0): "
      f"{stats.kstest(np.where(acc[:, None], z, 0.0)[:, 0], 'norm').pvalue:.2e}")

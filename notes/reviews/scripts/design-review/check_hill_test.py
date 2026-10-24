"""Test plan 2.8: 'Hill tail index recovers the index of Pareto samples within 10% at n = 10^4'.
Probability that a correct Hill implementation passes, for two choices of k_h."""
import numpy as np

rng = np.random.default_rng(71)
n, B = 10_000, 2000
for alpha in (1.5, 3.0):
    x = rng.pareto(alpha, size=(B, n)) + 1
    xs = np.sort(x, axis=1)
    for kh in (int(np.sqrt(n)), n // 10):
        a = 1.0 / np.mean(np.log(xs[:, -kh:]) - np.log(xs[:, -kh - 1 : -kh]), axis=1)
        print(f"alpha={alpha} k_h={kh:5d}: P(|alpha_hat/alpha - 1| < 0.1) = {np.mean(np.abs(a / alpha - 1) < 0.1):.3f}")

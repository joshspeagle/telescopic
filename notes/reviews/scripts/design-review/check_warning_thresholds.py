"""False-alarm rates of spec section 7.4 warnings 3 and 4 on clean (light-tailed) inputs.

Warning 3: one replicate contributes > 25% of the sample variance of any reported estimate, R >= 20.
Warning 4: a Hill tail-index estimate of tau below 2.
The spec does not fix the Hill order statistic k_h; two common choices are tried.
"""

import numpy as np

rng = np.random.default_rng(11)
B = 20_000  # simulated runs per cell


def max_share(a):
    dev = (a - a.mean(axis=-1, keepdims=True)) ** 2
    return dev.max(axis=-1) / dev.sum(axis=-1)


print("Warning 3, i.i.d. N(0,1) replicate values (the most benign case):")
for R in (20, 30, 50, 100, 200):
    a = rng.normal(size=(B, R))
    p1 = np.mean(max_share(a) > 0.25)
    # summary reports, per parameter, a mean and a standard deviation: 2 estimates per parameter
    for n_est in (1, 2, 10):
        a = rng.normal(size=(B, n_est, R))
        p = np.mean(np.any(max_share(a) > 0.25, axis=1))
        print(f"  R={R:4d}, {n_est:2d} independent estimates: P(fires) = {p:.4f}")

print("Warning 3, Student-t(5) replicate values (moderate tails, finite variance):")
for R in (20, 50, 100, 200):
    a = rng.standard_t(5, size=(B, 10, R))
    print(f"  R={R:4d}, 10 estimates: P(fires) = {np.mean(np.any(max_share(a) > 0.25, axis=1)):.4f}")


def hill_alpha(x, k):
    xs = np.sort(x, axis=-1)
    top = xs[..., -k:]
    thr = xs[..., -k - 1 : -k]
    return 1.0 / np.mean(np.log(top) - np.log(thr), axis=-1)


print("Warning 4, Hill tail index on geometric (light-tailed) meeting times:")
for mean_tau in (5, 50, 500):
    p = 1.0 / mean_tau
    for n in (100, 1000):
        tau = rng.geometric(p, size=(B, n)).astype(float)  # tau - L + 1 >= 1
        for label, k in (("k=sqrt(n)", int(np.sqrt(n))), ("k=n/10", n // 10)):
            a = hill_alpha(tau, k)
            print(
                f"  E[tau]={mean_tau:4d} n={n:5d} {label:10s}: median alpha_hat {np.median(a):5.2f}, "
                f"P(alpha_hat < 2) = {np.mean(a < 2):.3f}"
            )
print("  For reference, true tail index of a Pareto(alpha=1.5) sample, n=100, k=10:")
x = rng.pareto(1.5, size=(B, 100)) + 1
a = hill_alpha(x, 10)
print(f"  median alpha_hat {np.median(a):.2f}, P(alpha_hat < 2) = {np.mean(a < 2):.3f}")

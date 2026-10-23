"""False-alarm rate of spec 7.4 warning 4 (Hill tail index of tau below 2) on light tails.

Meeting times with geometric tails have an infinite tail index; any alarm on them is false.
The Hill estimator alpha_hat = 1 / mean(log X_(n-i+1) - log X_(n-k)), i = 1..k, is applied
to tau - L + 1 >= 1 (geometric) and, separately, to tau itself when the lag shifts it
(tau = L + G with L = k = q99, as under the AJ24 recipe). The spec fixes neither k_h nor
which variable; two common k_h choices are tried. B = 5000 simulated runs per cell.
"""
import numpy as np

rng = np.random.default_rng(12)
B = 5000


def hill_alpha(x, k):
    xs = np.sort(x, axis=-1)
    return 1.0 / np.mean(np.log(xs[..., -k:]) - np.log(xs[..., -k - 1 : -k]), axis=-1)


for mean_g in (5, 50, 500):
    for n in (100, 1000):
        g = rng.geometric(1.0 / mean_g, size=(B, n)).astype(float)  # tau - L + 1
        lag = np.quantile(g, 0.99, axis=1, keepdims=True).round()  # L = k ~ q99 of a pilot
        for label, kh in (("k_h=sqrt(n)", int(np.sqrt(n))), ("k_h=n/10", n // 10)):
            a1 = hill_alpha(g, kh)
            a2 = hill_alpha(g + lag - 1, kh)  # tau itself when L = k
            print(f"E[tau-L+1]={mean_g:4d} n={n:5d} {label:12s}: on tau-L+1 median {np.median(a1):5.2f}, "
                  f"P(<2) {np.mean(a1 < 2):.3f} | on tau with L=q99: median {np.median(a2):6.2f}, "
                  f"P(<2) {np.mean(a2 < 2):.3f}")
x = rng.pareto(1.5, size=(B, 100)) + 1
print(f"Pareto(1.5), n=100, k_h=10 (a true alarm case): median {np.median(hill_alpha(x, 10)):.2f}, "
      f"P(<2) {np.mean(hill_alpha(x, 10) < 2):.3f}")
x = rng.pareto(3.0, size=(B, 100)) + 1
print(f"Pareto(3.0), n=100, k_h=10 (finite variance): median {np.median(hill_alpha(x, 10)):.2f}, "
      f"P(<2) {np.mean(hill_alpha(x, 10) < 2):.3f}")

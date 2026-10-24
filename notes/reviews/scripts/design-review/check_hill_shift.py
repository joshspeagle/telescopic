"""Power of the Hill warning on truly heavy tails when applied to tau (lag-shifted) vs tau - L + 1."""
import numpy as np

rng = np.random.default_rng(13)
B, n, kh = 5000, 100, 10


def hill_alpha(x, k):
    xs = np.sort(x, axis=-1)
    return 1.0 / np.mean(np.log(xs[..., -k:]) - np.log(xs[..., -k - 1 : -k]), axis=-1)


for alpha in (1.2, 1.5, 1.8):
    g = np.floor(10 * (rng.pareto(alpha, size=(B, n)) + 1))  # tau - L + 1, P(G > x) ~ x^-alpha
    pilot = np.floor(10 * (rng.pareto(alpha, size=(B, 100)) + 1))
    lag = np.quantile(pilot, 0.99, axis=1, keepdims=True).round()  # L = k from an independent pilot
    a_rel = hill_alpha(g, kh)
    a_tau = hill_alpha(g + lag - 1, kh)
    print(f"true alpha {alpha}: P(alarm) on tau-L+1 {np.mean(a_rel < 2):.3f}; on tau = L + G - 1 {np.mean(a_tau < 2):.3f}")

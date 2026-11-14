"""Exploratory: the false-alarm rate of (N) on a correct gradient, by model and on Laplace targets.

Question 1 (model; generator seed 218000): when the values of log_prob on a line carry independent
Gaussian noise of standard deviation sigma far above ULPS units in the last place (cancellation), how
often does a line of (N) warn?  gap = |n(1) - n(-1)| / 2, T = |gap_signed - (n(2) - n(-2)) / 4|,
R_f = max(s_hat, |F4| / sqrt(70)), warn iff gap > SAFETY (T + R_f). Cases: the same sigma on the P = 8
lines; one line with k times the noise of the others.
Question 2 (model): a slope jump J at position t0 of the line (in steps), log_prob otherwise linear
and exact: the largest gap / tolerance over t0; and with two or three jumps at uniform positions in
(-2, 2), jumps of equal sign (a concave kink target such as Laplace) or of random sign and size.
Question 3 (measurement; seeds 200000..204999 at d = 1,000, 200000..200999 at d = 10,000): (N) on the
Laplace target -sum |x_i| from N(0, 4 I), where a line crosses a kink with probability about
1.1e-5 d.
Result: see fa_model.out.
"""

import warnings

import numpy as np
import prototype

rng = np.random.default_rng(218000)
SAFETY, P = prototype.SAFETY, prototype.POINTS
W4 = np.array([1.0, -4.0, 6.0, -4.0, 1.0])  # at t = -2, -1, 0, 1, 2


def noise(k, checks=4_000_000):
    # line 0 has k times the noise of the other P - 1 lines; returns P(line 0 warns), P(any line warns)
    warn0 = warn_any = 0
    for _ in range(checks // 500_000):
        n = rng.standard_normal((500_000, P, 5))
        n[:, 0] *= k
        odd1, odd2 = (n[..., 3] - n[..., 1]) / 2, (n[..., 4] - n[..., 0]) / 4
        F4 = n @ W4
        s_hat = np.sqrt(np.mean(F4**2, axis=1, keepdims=True) / 70)
        warn = np.abs(odd1) > SAFETY * (
            np.abs(odd1 - odd2) + np.maximum(s_hat, np.abs(F4) / np.sqrt(70))
        )
        warn0 += warn[:, 0].sum()
        warn_any += warn.any(axis=1).sum()
    return warn0 / checks, warn_any / checks


def kinks(m, sign, trials=2_000_000):
    # m slope jumps at uniform positions; f(t) = sum_j J_j relu(t - t_j); grad at 0 = sum of J_j with t_j < 0
    t0 = rng.uniform(-2, 2, (trials, m))
    J = np.ones((trials, m)) if sign == "equal" else rng.standard_normal((trials, m))

    def f(t):
        return np.sum(J * np.maximum(t - t0, 0.0), axis=1)

    g = np.sum(J * (t0 < 0), axis=1)
    gap_signed = (f(1) - f(-1)) / 2 - g
    T = np.abs(gap_signed - ((f(2) - f(-2)) / 4 - g))
    F4 = f(-2) - 4 * f(-1) + 6 * f(0) - 4 * f(1) + f(2)
    ratio = np.abs(gap_signed) / np.maximum(SAFETY * (T + np.abs(F4) / np.sqrt(70)), 1e-300)
    return float(ratio.max()), float(np.mean(ratio > 1))


def laplace(d, seeds):
    def log_prob(x):
        return -np.sum(np.abs(x), axis=-1)

    def grad(x):
        return -np.sign(x)

    def init(generator):
        return 2.0 * generator.standard_normal(d)

    warned, ratios, crossing = 0, [], 0
    for seed in seeds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = prototype.N(log_prob, grad, init, seed)
        warned += v.warned
        ratios.append(v.ratio)
        crossing += sum(
            abs(r["F4"]) > 1e-3 * abs(r["D"]) / d for r in v.detail
        )  # a kink shows as a fourth difference
    return warned, len(seeds), float(np.max(ratios)), crossing


if __name__ == "__main__":
    print(
        "1. noise far above the rounding floor: P(a given line warns), P(a check of 8 lines warns)"
    )
    for k in (1, 3, 10, 100):
        a, b = noise(k)
        print(
            f"   line 0 has {k:>3} times the noise of the others: line 0 {a:.2e}, check {b:.2e}",
            flush=True,
        )
    print(
        "2. slope jumps on a line, log_prob otherwise linear: largest gap / tolerance, P(gap > tolerance)"
    )
    for m, sign in ((1, "equal"), (2, "equal"), (3, "equal"), (2, "random"), (3, "random")):
        a, b = kinks(m, sign)
        print(f"   {m} jumps of {sign} sign: largest ratio {a:.3f}, P(warn) {b:.2e}", flush=True)
    print("3. (N) on Laplace from N(0, 4 I)")
    for d, seeds in ((1000, range(200000, 205000)), (10_000, range(200000, 201000))):
        w, n, r, c = laplace(d, seeds)
        print(
            f"   d = {d}: warned {w}/{n}, largest ratio {r:.3f}, lines with a kink inside {c}/{8 * n}",
            flush=True,
        )

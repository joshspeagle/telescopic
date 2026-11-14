"""Exploratory: what a gradient error does to MALA's acceptance and step size (generator seeds 217000..217011).

Question: for MALA on N(0, I_d) with proposal x + (eps^2 / 2) g(x) + eps z and g = grad + e, is the
acceptance rate  2 Phi(-sqrt(s0^2 + 4 delta^2) / 2),  with delta = (eps / 2) |e| the shift of the
proposal's mean in proposal standard deviations and s0 defined by the correct gradient's rate
a0 = 2 Phi(-s0 / 2)?  And is the step size that restores the rate 0.574 the root of
c^6 + (4 delta0^2 / s*^2) c^2 = 1,  s* = 1.124,  delta0 = delta at the correct gradient's step?
Design: one step from min(200,000, 2e7 / d) exact draws of N(0, I_d); errors: a constant offset e = eta w (|w| = 1),
the sign of coordinate 0 (e = 2 x_0, delta(x) = eps |x_0|, the model averaged over x_0), and twice the
gradient (e = -x, delta = eps |x| / 2). The step of rate 0.574 by bisection with common random numbers.
Result: see harm_model.out.
"""

import numpy as np
from scipy import optimize, stats

N_MAX = 200_000  # draws: min(N_MAX, 2e7 / d)


def acceptance(d, eps, kind, eta, seed):
    # In chunks of at most 2e6 numbers, so that d = 1,000 fits in memory; x_0 and |x| are kept.
    rng = np.random.default_rng(seed)
    N = min(N_MAX, 20_000_000 // d)
    chunk, total, x0, norm = max(1, 2_000_000 // d), 0.0, [], []
    for start in range(0, N, chunk):
        n = min(chunk, N - start)
        x, z = rng.standard_normal((n, d)), rng.standard_normal((n, d))

        def g(y):
            if kind == "offset":
                out = -y
                out[:, 0] += eta
                return out  # N(0, I) is isotropic: any unit vector
            if kind == "sign":
                out = -y
                out[:, 0] = y[:, 0]
                return out
            if kind == "twice":
                return -2 * y
            return -y

        gx = g(x)
        y = x + 0.5 * eps**2 * gx + eps * z
        gy = g(y)
        log_q = 0.5 * np.sum((x - y) * (gx + gy), axis=1) + eps**2 / 8 * (
            np.sum(gx**2, axis=1) - np.sum(gy**2, axis=1)
        )
        log_alpha = -0.5 * np.sum(y * y, axis=1) + 0.5 * np.sum(x * x, axis=1) + log_q
        total += float(np.sum(np.exp(np.minimum(log_alpha, 0.0))))
        x0.append(x[:, 0].copy())
        norm.append(np.sqrt(np.sum(x * x, axis=1)))
    return total / N, (np.concatenate(x0), np.concatenate(norm))


def predicted(a0, delta):
    s0 = -2 * stats.norm.ppf(a0 / 2)
    return float(np.mean(2 * stats.norm.cdf(-np.sqrt(s0**2 + 4 * np.asarray(delta) ** 2) / 2)))


def tuned(d, kind, eta, seed):
    return optimize.brentq(
        lambda e: acceptance(d, e, kind, eta, seed)[0] - 0.574, 1e-3, 3.0, xtol=1e-3
    )


if __name__ == "__main__":
    S_STAR2 = (2 * stats.norm.ppf(1 - 0.574 / 2)) ** 2
    print(f"s*^2 = {S_STAR2:.4f}")
    print("1. acceptance at the step eps_r that gives 0.574 with the correct gradient")
    print(
        f"{'d':>5} {'eps_r':>6} {'error':<22} {'delta (rms)':>11} | {'2 Phi(-delta)':>13} {'model':>7} {'simulated':>9}"
    )
    for i, d in enumerate((5, 100, 1000)):
        seed = 217000 + i
        eps = tuned(d, "right", 0.0, seed)
        a0, x = acceptance(d, eps, "right", 0.0, seed)
        for delta in (0.125, 0.25, 0.5, 1.0, 2.0):
            eta = 2 * delta / eps
            a, _ = acceptance(d, eps, "offset", eta, seed)
            print(
                f"{d:>5} {eps:>6.3f} {f'offset, |e| = {eta:.3g}':<22} {delta:>11.3f} | {2 * stats.norm.cdf(-delta):>13.3f} {predicted(a0, delta):>7.3f} {a:>9.3f}"
            )
        a, _ = acceptance(d, eps, "sign", 0.0, seed)
        dl = eps * np.abs(x[0])
        print(
            f"{d:>5} {eps:>6.3f} {'sign of coordinate 0':<22} {np.sqrt(np.mean(dl**2)):>11.3f} | {'':>13} {predicted(a0, dl):>7.3f} {a:>9.3f}"
        )
        a, _ = acceptance(d, eps, "twice", 0.0, seed)
        dl = eps * x[1] / 2
        print(
            f"{d:>5} {eps:>6.3f} {'twice the gradient':<22} {np.sqrt(np.mean(dl**2)):>11.3f} | {'':>13} {predicted(a0, dl):>7.3f} {a:>9.3f}"
        )
    print(
        "2. the step that restores 0.574: c = eps_wrong / eps_right against the root of c^6 + (4 delta0^2 / s*^2) c^2 = 1"
    )
    print(f"{'d':>5} {'error':<22} {'delta0':>7} | {'model c':>7} {'simulated c':>11}")
    for i, d in enumerate((5, 100, 1000)):
        seed = 217006 + i
        eps_r = tuned(d, "right", 0.0, seed)
        rows = [("offset", 2 * dl / eps_r, dl) for dl in (0.25, 0.5, 1.0, 2.0, 2.9)] + [
            ("sign", 0.0, eps_r),
            ("twice", 0.0, eps_r * np.sqrt(d) / 2),
        ]
        for kind, eta, delta0 in rows:
            c_model = optimize.brentq(
                lambda c, delta0=delta0: c**6 + 4 * delta0**2 / S_STAR2 * c**2 - 1, 0, 1
            )
            c_sim = tuned(d, kind, eta, seed) / eps_r
            label = (
                f"offset, |e| = {eta:.3g}"
                if kind == "offset"
                else {"sign": "sign of coordinate 0", "twice": "twice the gradient"}[kind]
            )
            print(
                f"{d:>5} {label:<22} {delta0:>7.3f} | {c_model:>7.3f} {c_sim:>11.3f}", flush=True
            )

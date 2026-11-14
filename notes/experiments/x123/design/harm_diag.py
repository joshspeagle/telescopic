"""Exploratory: harm of one wrong-sign coordinate at d = 200 to 10,000, by a diagonal coupled MALA (generator seeds 219600..219699).

Question: does the sign of one coordinate still slow meeting at d = 1,000 and 10,000, where the
package's MALA (dense covariance, a pilot of 10 d^2 steps) is too slow to say?
Design: a re-implementation of the package's coupled MALA step for cov = I (the proposal
x + (eps^2 / 2) g(x) + eps z, the reflection-maximal coupling of the two proposals, one uniform for
both acceptances, lag 1), on N(0, I_d) from N(0, 4 I). The step is the one with a mean acceptance of
0.574 at stationarity for the gradient in use (bisection on 20,000 d-independent draws), which is
what the package's dual averaging aims at. 100 pairs (50 at d = 10,000), at most 20,000 steps.
Check against the package: at d = 200 the review measured default-tuned medians of 28 to 32.5
(correct) and 138.5 to 156.5 (wrong), steps 0.66 to 0.67 and 0.53 to 0.55 (its report on issue #123).
usage: harm_diag.py d [pairs]
Result: see harm_diag_*.out.
"""

import sys
import time

import numpy as np
from scipy import optimize


def make_grad(wrong):
    def g(y):
        out = -y
        if wrong:
            out[:, 0] = y[:, 0]
        return out

    return g


def log_alpha(x, gx, y, gy, eps):
    log_q = 0.5 * np.sum((x - y) * (gx + gy), axis=1) + eps**2 / 8 * (
        np.sum(gx**2, axis=1) - np.sum(gy**2, axis=1)
    )
    return 0.5 * (np.sum(x * x, axis=1) - np.sum(y * y, axis=1)) + log_q


def tuned(d, g, seed):
    def rate(eps):
        rng = np.random.default_rng(seed)
        n = max(200, 4_000_000 // d)
        x, z = rng.standard_normal((n, d)), rng.standard_normal((n, d))
        gx = g(x)
        y = x + 0.5 * eps**2 * gx + eps * z
        return float(np.mean(np.exp(np.minimum(log_alpha(x, gx, y, g(y), eps), 0.0)))) - 0.574

    return optimize.brentq(rate, 1e-3, 3.0, xtol=1e-3)


def meeting_times(d, g, eps, pairs, seed, max_iter=20_000):
    rng = np.random.default_rng(seed)
    x, y = 2 * rng.standard_normal((pairs, d)), 2 * rng.standard_normal((pairs, d))

    # lag 1: X takes one step alone
    def single(x):
        gx = g(x)
        prop = x + 0.5 * eps**2 * gx + eps * rng.standard_normal(x.shape)
        acc = np.log(rng.random(len(x))) < log_alpha(x, gx, prop, g(prop), eps)
        return np.where(acc[:, None], prop, x)

    x = single(x)
    tau = np.full(pairs, np.inf)
    live = np.arange(pairs)
    for t in range(2, max_iter + 1):
        gx, gy = g(x), g(y)
        mx, my = x + 0.5 * eps**2 * gx, y + 0.5 * eps**2 * gy
        z = rng.standard_normal(x.shape)
        delta = (mx - my) / eps
        sq = np.sum(delta * delta, axis=1)
        same = np.log(rng.random(len(x))) < -np.sum(z * delta, axis=1) - 0.5 * sq
        e = delta / np.sqrt(sq)[:, None]
        px = mx + eps * z
        py = np.where(same[:, None], px, my + eps * (z - 2 * np.sum(e * z, axis=1)[:, None] * e))
        log_u = np.log(rng.random(len(x)))
        ax = log_u < log_alpha(x, gx, px, g(px), eps)
        ay = log_u < log_alpha(y, gy, py, g(py), eps)
        x, y = np.where(ax[:, None], px, x), np.where(ay[:, None], py, y)
        met = np.all(x == y, axis=1)
        tau[live[met]] = t
        x, y, live = x[~met], y[~met], live[~met]
        if not len(live):
            break
    return tau


if __name__ == "__main__":
    d = int(sys.argv[1])
    pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    out = []
    for wrong in (False, True):
        t = time.time()
        g = make_grad(wrong)
        eps = tuned(d, g, 219600)
        tau = meeting_times(d, g, eps, pairs, 219601 + wrong)
        out.append((eps, tau))
        print(
            f"d = {d}, {'wrong sign of coordinate 0' if wrong else 'correct gradient'}: step {eps:.3f}, meeting times mean "
            f"{np.mean(np.minimum(tau, 20_000)):.1f}, median {np.median(tau):g}, unmet {int(np.isinf(tau).sum())}/{pairs} ({time.time() - t:.0f}s)",
            flush=True,
        )
    r, w = out
    mr, mw = np.minimum(r[1], 20_000), np.minimum(w[1], 20_000)
    ratio = mw.mean() / mr.mean()
    se = ratio * np.sqrt(
        mr.var(ddof=1) / len(mr) / mr.mean() ** 2 + mw.var(ddof=1) / len(mw) / mw.mean() ** 2
    )
    print(
        f"d = {d}: ratio of mean meeting times {ratio:.2f} +- {se:.2f}; ratio of steps {w[0] / r[0]:.3f}"
    )

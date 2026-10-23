"""E01: does the Atchade-Jacob recipe (k = q99(tau - L) at L = 1, then L = k, m = 10k) beat L = 1?

Target N(0, I_d); initial distribution N(3, I_d); coupled RWMH with reflection-maximal
proposals, step 2.38/sqrt(d), common acceptance uniform. Estimates E[x_1] = 0 and
E[x_1^2] = 1. Lockstep over replicates, no refill; per-replicate cost is counted exactly
as L + 2(tau - L) + max(0, m - tau) proposals (kernel applications).

Result (2026-10-23, Python 3.13.16, NumPy 2.5.4, seed 2026, R = 4000 per configuration;
"ineff" = mean cost x var(H), for h = x_1 and h = x_1^2):

    d=2:  pilot mean tau 7.2,  k = q99(tau - 1) = 28,  m = 280
      L=1:   E[x1] +0.0016 +- 0.0027  E[x1^2] 1.0024 +- 0.0036  cost 286   ineff 8.58, 14.46   BC != 0 in 0.6%
      L=28:  E[x1] -0.0002 +- 0.0027  E[x1^2] 0.9975 +- 0.0035  cost 289   ineff 8.42, 13.85   BC != 0 in 1.2%
    d=10: pilot mean tau 46.7, k = q99(tau - 1) = 143, m = 1430
      L=1:   E[x1] +0.0006 +- 0.0026  E[x1^2] 1.0027 +- 0.0029  cost 1476  ineff 38.82, 48.03  BC != 0 in 1.4%
      L=143: E[x1] -0.0005 +- 0.0025  E[x1^2] 1.0008 +- 0.0027  cost 1507  ineff 36.58, 44.47  BC != 0 in 3.5%

Both lags are unbiased (|z| < 1). At k = q99, L = k lowers the standard deviation by 2-4%
and the inefficiency by 2-6% for 1-2% more cost: the recipe is safe and slightly better.
"""

import numpy as np


def log_prob(x):
    return -0.5 * np.sum(x * x, axis=-1)


def run(R, d, k, m, lag, rng, mu0=3.0, hs=(lambda x: x[:, 0], lambda x: x[:, 0] ** 2)):
    s = 2.38 / np.sqrt(d)
    x = mu0 + rng.standard_normal((R, d))
    y = mu0 + rng.standard_normal((R, d))
    lx, ly = log_prob(x), log_prob(y)
    n = m - k + 1
    H = np.zeros((len(hs), R))
    tau = np.full(R, -1)
    met = np.zeros(R, bool)
    t = 0
    if k == 0:
        H += np.array([h(x) for h in hs]) / n
    # X alone for L steps
    for _ in range(lag):
        t += 1
        prop = x + s * rng.standard_normal((R, d))
        lp = log_prob(prop)
        acc = np.log(rng.random(R)) < lp - lx
        x = np.where(acc[:, None], prop, x)
        lx = np.where(acc, lp, lx)
        if k <= t <= m:
            H += np.array([h(x) for h in hs]) / n
    while t < m or not met.all():
        t += 1  # produce X_t and Y_{t-L}
        active = ~met
        xdot = rng.standard_normal((R, d))
        logu = np.log(rng.random(R))
        logv = np.log(rng.random(R))
        z = (x - y) / s
        z2 = np.sum(z * z, axis=-1)
        meet = active & (logu < -np.sum(xdot * z, axis=-1) - 0.5 * z2)
        with np.errstate(invalid="ignore", divide="ignore"):
            e = z / np.sqrt(z2)[:, None]
        ydot = xdot - 2 * np.sum(e * xdot, axis=-1)[:, None] * e
        px = x + s * xdot
        py = np.where(meet[:, None], px, y + s * ydot)
        lpx = log_prob(px)
        lpy = np.where(meet, lpx, log_prob(py))
        ax = logv < lpx - lx
        ay = logv < lpy - ly
        x = np.where(ax[:, None], px, x)
        lx = np.where(ax, lpx, lx)
        ynew = np.where((active & ay)[:, None], py, y)
        lynew = np.where(active & ay, lpy, ly)
        now = active & meet & ax & ay
        tau[now] = t
        if k <= t <= m:
            H += np.array([h(x) for h in hs]) / n
        still = active & ~now
        if t >= k + lag:
            v = (t - k) // lag - int(np.ceil(max(lag, t - m) / lag)) + 1
            if v > 0:
                H += (
                    np.where(
                        still,
                        v * (np.array([h(x) for h in hs]) - np.array([h(ynew) for h in hs])),
                        0.0,
                    )
                    / n
                )
        y = np.where(active[:, None], ynew, x)
        ly = np.where(active, lynew, lx)
        met |= now
        y[met] = x[met]  # faithfulness: Y_{t-L} = X_t after meeting
    cost = lag + 2 * (tau - lag) + np.maximum(0, m - tau)
    return H, tau, cost


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    for d in (2, 10):
        _, tau, _ = run(2000, d, 0, 0, 1, rng)
        k = int(np.quantile(tau - 1, 0.99))
        m = 10 * k
        print(f"d={d}: pilot tau mean {tau.mean():.1f}, q99(tau-1) = {k} -> k={k}, m={m}")
        for lag in (1, k):
            H, tau, cost = run(4000, d, k, m, lag, rng)
            mean, sd = H.mean(axis=1), H.std(axis=1, ddof=1)
            se = sd / np.sqrt(H.shape[1])
            ineff = cost.mean() * sd**2
            print(
                f"  L={lag:4d}: E[x1] {mean[0]:+.4f}+-{se[0]:.4f}  E[x1^2] {mean[1]:.4f}+-{se[1]:.4f}  "
                f"sd(H) {sd[0]:.4f},{sd[1]:.4f}  mean cost {cost.mean():.0f}  "
                f"ineff {ineff[0]:.2f},{ineff[1]:.2f}  frac BC!=0 {(tau > k + lag).mean():.3f}"
            )

"""The AJ24 recipe at the spec's defaults, on clean Gaussian targets (E01's setting).

Target N(0, I_d), pi0 = N(3, I_d), coupled RWMH with reflection-maximal proposals, step
2.38/sqrt(d), common acceptance uniform (as in notes/experiments/e01_lag_recipe.py).

1. Variability of k = ceil(q_beta(tau - 1)) from a pilot of R0 = 100 (spec default) versus
   R0 = 2000 (E01), for beta = 0.99 and 0.999 (np.quantile, default 'linear' method).
2. Main run with L = k, m = 10k (k from the 2000-pilot): fraction of replicates with
   tau - L > k (spec 7.4 warning 2 threshold 0.05; spec 3.6 says the TV bound at k is
   'about 0.01'), and the same fraction evaluated at the k of each 100-pilot.
3. Spec 7.4 warnings on batches of R = 100 main-run replicates (the default n_replicates):
   warning 2 (fraction > 0.05), warning 3 (one replicate > 25% of the sample variance of any
   of the 2d per-parameter estimates: means of x_i and of x_i^2), warning 4 (Hill on
   tau - L + 1 with k_h = 10, below 2).
4. Efficiency-loss approximation 1 + (k + tau_bar)/(m - k) with tau_bar from the pilot
   (lag 1) and from the main run (lag k), against the measured mean cost / (m - k + 1).
"""
import sys
import time

import numpy as np


def run(R, d, k, m, lag, rng, mu0=3.0):
    """Lockstep coupled RWMH; returns H for h = x_i and x_i^2 (R, d) each, tau, cost."""
    s = 2.38 / np.sqrt(d)
    x = mu0 + rng.standard_normal((R, d))
    y = mu0 + rng.standard_normal((R, d))
    lx, ly = -0.5 * np.sum(x * x, 1), -0.5 * np.sum(y * y, 1)
    n = m - k + 1
    H1 = np.zeros((R, d))
    H2 = np.zeros((R, d))
    tau = np.full(R, -1)
    met = np.zeros(R, bool)
    t = 0
    if k == 0:
        H1 += x / n
        H2 += x * x / n
    for _ in range(lag):
        t += 1
        prop = x + s * rng.standard_normal((R, d))
        lp = -0.5 * np.sum(prop * prop, 1)
        acc = np.log(rng.random(R)) < lp - lx
        x = np.where(acc[:, None], prop, x)
        lx = np.where(acc, lp, lx)
        if k <= t <= m:
            H1 += x / n
            H2 += x * x / n
    while t < m or not met.all():
        t += 1
        active = ~met
        xdot = rng.standard_normal((R, d))
        logu = np.log(rng.random(R))
        logv = np.log(rng.random(R))
        z = (x - y) / s
        z2 = np.sum(z * z, 1)
        meet = active & (logu < -np.sum(xdot * z, 1) - 0.5 * z2)
        with np.errstate(invalid="ignore", divide="ignore"):
            e = z / np.sqrt(z2)[:, None]
        ydot = xdot - 2 * np.sum(e * xdot, 1)[:, None] * e
        px = x + s * xdot
        py = np.where(meet[:, None], px, y + s * ydot)
        lpx = -0.5 * np.sum(px * px, 1)
        lpy = np.where(meet, lpx, -0.5 * np.sum(py * py, 1))
        ax = logv < lpx - lx
        ay = logv < lpy - ly
        x = np.where(ax[:, None], px, x)
        lx = np.where(ax, lpx, lx)
        ynew = np.where((active & ay)[:, None], py, y)
        lynew = np.where(active & ay, lpy, ly)
        now = active & meet & ax & ay
        tau[now] = t
        if k <= t <= m:
            H1 += x / n
            H2 += x * x / n
        still = active & ~now
        if t >= k + lag:
            v = (t - k) // lag - int(np.ceil(max(lag, t - m) / lag)) + 1
            if v > 0:
                H1 += np.where(still[:, None], v * (x - ynew), 0.0) / n
                H2 += np.where(still[:, None], v * (x * x - ynew * ynew), 0.0) / n
        y = np.where(active[:, None], ynew, x)
        ly = np.where(active, lynew, lx)
        met |= now
        y[met] = x[met]
    cost = lag + 2 * (tau - lag) + np.maximum(0, m - tau)
    return H1, H2, tau, cost


def hill_alpha(x, kh=10):
    xs = np.sort(x)
    return 1.0 / np.mean(np.log(xs[-kh:]) - np.log(xs[-kh - 1]))


def max_share(a):
    dev = (a - a.mean(0)) ** 2
    return (dev.max(0) / dev.sum(0)).max()


rng = np.random.default_rng(2027)
for d in [int(a) for a in sys.argv[1:]] or [10, 30]:
    t0 = time.perf_counter()
    _, _, tau_p, _ = run(2000, d, 0, 0, 1, rng)
    big = np.quantile(tau_p - 1, [0.99, 0.999])
    k_ref = int(np.ceil(big[0]))
    ks99, ks999 = [], []
    for _ in range(2000):
        sub = rng.choice(tau_p, 100, replace=False) - 1
        q = np.quantile(sub, [0.99, 0.999])
        ks99.append(int(np.ceil(q[0])))
        ks999.append(int(np.ceil(q[1])))
    ks99, ks999 = np.array(ks99), np.array(ks999)
    print(f"d={d}: pilot (R0=2000, lag 1) mean tau {tau_p.mean():.1f}; q99(tau-1) {big[0]:.0f}, "
          f"q999 {big[1]:.0f}, max {tau_p.max() - 1}")
    print(f"  k from R0=100 pilots, beta=0.99: 5/50/95% = {np.percentile(ks99, [5, 50, 95])}; "
          f"beta=0.999: 5/50/95% = {np.percentile(ks999, [5, 50, 95])}; "
          f"(R0=2000: q99 {big[0]:.0f}, q999 {big[1]:.0f})")
    m = 10 * k_ref
    H1, H2, tau, cost = run(4000, d, k_ref, m, k_ref, rng)
    g = tau - k_ref  # tau - L
    frac = np.mean(g > k_ref)
    print(f"  main run L=k={k_ref}, m={m}, R=4000: P(tau-L > k) = {frac:.4f} "
          f"(se {np.sqrt(frac * (1 - frac) / g.size):.4f}); mean(tau-L) {g.mean():.1f} vs pilot mean(tau-1) "
          f"{(tau_p - 1).mean():.1f}")
    # TV bound at t = k with L = k: E[max(0, ceil((tau - L - k)/L))]
    tv_k = np.mean(np.maximum(0, np.ceil((tau - k_ref - k_ref) / k_ref)))
    print(f"  estimated TV bound at t=k: {tv_k:.4f}")
    fr = np.array([np.mean(g > kk) for kk in ks99])
    from scipy import stats as _st
    p_fire = np.mean([_st.binom.sf(5, 100, p_i) for p_i in fr])
    print(f"  P(warning 2 fires) in a default run (R0=100 pilot, R=100, threshold 0.05): {p_fire:.3f}")
    print(f"  P(tau-L > k_i) at the k_i of R0=100 pilots: 5/50/95% = {np.percentile(fr, [5, 50, 95]).round(4)}, "
          f"fraction of pilots giving > 0.05: {np.mean(fr > 0.05):.3f}")
    # warnings on batches of 100
    w2 = w3 = w4 = 0
    nb = 40
    for b in range(nb):
        sl = slice(100 * b, 100 * (b + 1))
        w2 += np.mean(g[sl] > k_ref) > 0.05
        w3 += max(max_share(H1[sl]), max_share(H2[sl])) > 0.25
        w4 += hill_alpha(g[sl] + 1.0) < 2
    print(f"  warnings on {nb} batches of R=100 (2d={2 * d} estimates): w2 {w2}/{nb}, w3 {w3}/{nb}, w4 {w4}/{nb}")
    z1 = H1.mean(0) / (H1.std(0, ddof=1) / np.sqrt(len(H1)))
    z2 = (H2.mean(0) - 1) / (H2.std(0, ddof=1) / np.sqrt(len(H2)))
    print(f"  unbiasedness check: max |z| over coordinates, E[x]: {np.abs(z1).max():.2f}, E[x^2]: {np.abs(z2).max():.2f}")
    # efficiency-loss approximation
    ratio = cost.mean() / (m - k_ref + 1)
    approx_pilot = 1 + (k_ref + tau_p.mean()) / (m - k_ref)
    approx_main = 1 + (k_ref + tau.mean()) / (m - k_ref)
    approx_general = 1 + (k_ref + (tau - k_ref).mean()) / (m - k_ref)
    print(f"  mean cost/(m-k+1) {ratio:.3f}; 1+(k+tau_bar)/(m-k): pilot tau_bar {approx_pilot:.3f}, "
          f"main-run tau_bar {approx_main:.3f}, main-run mean(tau-L) {approx_general:.3f}")
    print(f"  [{time.perf_counter() - t0:.0f}s]")

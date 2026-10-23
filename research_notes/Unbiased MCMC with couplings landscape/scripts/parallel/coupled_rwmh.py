"""Minimal vectorised coupled RWMH (reflection-maximal coupling, lag 1) for experiments.

Target: N(0, I_d). Initial distribution: N(mu0, I_d). Test function h(x) = x[0].
Estimator H_{k:m} of Jacob, O'Leary & Atchade (2020), eq. (2.1), computed by streaming
accumulators. Cost in lockstep iterations = max(tau, m) (the pair needs max(tau, m) steps).
"""
import numpy as np


def logpi(x):
    return -0.5 * np.sum(x * x, axis=-1)


def run_pairs(n, d=1, sigma=0.5, mu0=5.0, k=5, m=10, rng=None, max_iter=100000):
    """Run n independent coupled pairs in lockstep (with compaction), return H, tau, cost."""
    rng = np.random.default_rng(rng)
    x = mu0 + rng.standard_normal((n, d))  # X_0
    y = mu0 + rng.standard_normal((n, d))  # Y_0
    lx, ly = logpi(x), logpi(y)
    denom = m - k + 1
    mcmc = np.zeros(n)
    bc = np.zeros(n)
    if k == 0:
        mcmc += x[:, 0]
    # lag step: X_1 ~ P(X_0, .)
    prop = x + sigma * rng.standard_normal((n, d))
    lp = logpi(prop)
    acc = np.log(rng.random(n)) < lp - lx
    x[acc], lx[acc] = prop[acc], lp[acc]
    t = 1
    if k <= 1 <= m:
        mcmc += x[:, 0]
    if 1 >= k + 1:  # bias-correction term at t=1 when k=0
        w = min(1.0, (1 - k) / denom)
        bc += w * (x[:, 0] - y[:, 0])
    tau = np.full(n, -1, dtype=np.int64)
    met = np.zeros(n, dtype=bool)
    active = np.arange(n)
    while active.size and t < max_iter:
        t += 1
        xa, ya, lxa, lya = x[active], y[active], lx[active], ly[active]
        meta = met[active]
        xi = rng.standard_normal((active.size, d))
        u_c = rng.random(active.size)
        u_a = rng.random(active.size)
        px = xa + sigma * xi
        # reflection-maximal coupling for unmet lanes
        z = (xa - ya) / sigma
        nz = np.linalg.norm(z, axis=1)
        nz_safe = np.where(nz > 0, nz, 1.0)
        e = z / nz_safe[:, None]
        logratio = -0.5 * np.sum((xi + z) ** 2, axis=1) + 0.5 * np.sum(xi * xi, axis=1)
        same = np.log(u_c) <= logratio
        eta = xi - 2.0 * np.sum(e * xi, axis=1)[:, None] * e
        py = np.where(same[:, None], px, ya + sigma * eta)
        lpx, lpy = logpi(px), logpi(py)
        lu = np.log(u_a)
        accx = lu < lpx - lxa
        accy = lu < lpy - lya
        xa = np.where(accx[:, None], px, xa)
        lxa = np.where(accx, lpx, lxa)
        # met lanes: Y follows X (Y_{t-1} = X_t is not needed after meeting); keep y = x
        ya_new = np.where(accy[:, None], py, ya)
        lya_new = np.where(accy, lpy, lya)
        ya = np.where(meta[:, None], xa, ya_new)
        lya = np.where(meta, lxa, lya_new)
        newly = (~meta) & np.all(xa == ya, axis=1)
        tau_a = tau[active]
        tau_a[newly] = t
        meta = meta | newly
        # accumulators
        if k <= t <= m:
            mcmc[active] += xa[:, 0]
        if t >= k + 1:
            w = min(1.0, (t - k) / denom)
            unmet_now = ~meta  # Delta_t = 0 once met
            bc[active] += np.where(unmet_now, w * (xa[:, 0] - ya[:, 0]), 0.0)
        x[active], y[active], lx[active], ly[active] = xa, ya, lxa, lya
        met[active], tau[active] = meta, tau_a
        done = meta & (t >= m)
        active = active[~done]
    H = mcmc / denom + bc
    cost = np.maximum(tau, m)
    return H, tau, cost


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    H, tau, cost = run_pairs(200000, rng=1)
    print(f"n=200000 time {time.perf_counter()-t0:.2f}s")
    print("E[H]", H.mean(), "+/-", H.std() / np.sqrt(H.size))
    print("E[tau]", tau.mean(), "max tau", tau.max(), "q50,q90,q99", np.quantile(tau, [0.5, 0.9, 0.99]))
    print("corr(H, cost)", np.corrcoef(H, cost)[0, 1])

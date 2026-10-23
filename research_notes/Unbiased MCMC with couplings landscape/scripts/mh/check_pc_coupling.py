"""Check O'Leary, Wang & Jacob (2021) P_C coupling (Algorithm 5) vs P_SQ (Algorithm 1).

1. Reproduce their Section 4.1 / Table 1: target Expo(1), proposal N(z + kappa, sigma^2), kappa = sigma^2 = 3,
   inits iid from target, tau = min(t >= 0 : X_t = Y_t), 10,000 replications.
   Published: P_SQ+Q_MI 74.0 (0.94), P_SQ+Q_MR 75.6 (0.99), P_C+Q_MI 61.3 (0.87), P_C+Q_MR 62.2 (0.89).
2. Gaussian RWM N(0, I_d), ell = 2.38: P_SQ vs P_C, both with reflection-maximal proposals (Q_MR).
"""
import math
import sys

import numpy as np

rng = np.random.default_rng(7)
NEG_INF = -np.inf


# ---------- generic pieces (vectorized over pairs; states are (k, d) arrays) ----------
def q_mi(x, y, mean_fn, sig, rng):
    """Maximal coupling with independent residuals of N(mean(x), sig^2 I), N(mean(y), sig^2 I)."""
    mx, my = mean_fn(x), mean_fn(y)
    k, d = x.shape
    lq = lambda pts, m: -0.5 * np.sum((pts - m) ** 2, 1) / sig**2
    xp = mx + sig * rng.standard_normal((k, d))
    same = np.log(rng.uniform(size=k)) <= lq(xp, my) - lq(xp, mx)
    yp = xp.copy()
    todo = ~same
    while todo.any():
        j = np.flatnonzero(todo)
        cand = my[j] + sig * rng.standard_normal((j.size, d))
        acc = np.log(rng.uniform(size=j.size)) > lq(cand, mx[j]) - lq(cand, my[j])
        yp[j[acc]] = cand[acc]
        todo[j[acc]] = False
    return xp, yp, same


def q_mr(x, y, mean_fn, sig, rng):
    """Reflection-maximal coupling of N(mean(x), sig^2 I), N(mean(y), sig^2 I)."""
    mx, my = mean_fn(x), mean_fn(y)
    k, d = x.shape
    z = (mx - my) / sig
    nz = np.linalg.norm(z, axis=1)
    e = z / np.where(nz > 0, nz, 1.0)[:, None]
    xi = rng.standard_normal((k, d))
    exi = np.sum(e * xi, 1)
    same = np.log(rng.uniform(size=k)) < -0.5 * ((exi + nz) ** 2 - exi**2)
    xp = mx + sig * xi
    yp = np.where(same[:, None], xp, my + sig * (xi - 2.0 * exi[:, None] * e))
    return xp, yp, same


def run(logpi, mean_fn, sig, init, qcoup, kernel, M, rng, tmax=10**6):
    """kernel in {'SQ','C'}; returns meeting times tau = min(t >= 0: X_t = Y_t)."""
    x, y = init(M), init(M)
    lpx, lpy = logpi(x), logpi(y)
    lq = lambda pts, m: -0.5 * np.sum((pts - m) ** 2, 1) / sig**2  # log q up to const

    def log_a(cur, lpcur, prop, lpprop):
        # MH log acceptance prob, general (non-symmetric) proposal q(z, .) = N(mean(z), sig^2 I)
        r = lpprop - lpcur + lq(cur, mean_fn(prop)) - lq(prop, mean_fn(cur))
        return np.minimum(0.0, np.where(np.isfinite(lpprop), r, -np.inf))

    tau = np.full(M, -1)
    same0 = np.all(x == y, 1)
    tau[same0] = 0
    active = ~same0
    t = 0
    while active.any() and t < tmax:
        t += 1
        idx = np.flatnonzero(active)
        xa, ya, lxa, lya = x[idx], y[idx], lpx[idx], lpy[idx]
        xp, yp, same = qcoup(xa, ya, mean_fn, sig, rng)
        lpxp = logpi(xp)
        lpyp = np.where(same, lpxp, logpi(yp))
        lax = log_a(xa, lxa, xp, lpxp)
        lay = log_a(ya, lya, yp, lpyp)
        logu = np.log(rng.uniform(size=idx.size))
        if kernel == "SQ":
            ax, ay = logu < lax, logu < lay
        else:  # P_C, Algorithm 5 with maximal B1 and maximal B2 (common uniform)
            lqx_x, lqy_x = lq(xp, mean_fn(xa)), lq(xp, mean_fn(ya))  # log q(x,x'), log q(y,x')
            lqx_y, lqy_y = lq(yp, mean_fn(xa)), lq(yp, mean_fn(ya))  # log q(x,y'), log q(y,y')
            lf_x = lqx_x + lax  # log f(x, x')
            lf_y = lqy_y + lay  # log f(y, y')
            lqm_x = np.minimum(lqx_x, lqy_x)  # log q^m(x')
            lqm_y = np.minimum(lqx_y, lqy_y)  # log q^m(y')
            # meeting proposed: b = 1 ^ f / q^m
            lb_x = np.minimum(0.0, lf_x - lqm_x)
            lb_y = np.minimum(0.0, lf_y - lqm_y)
            # not proposed: c = (f - q^m)_+ / (q - q^m)   (=1 if q - q^m == 0)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                num_x = np.exp(lf_x) - np.exp(lqm_x)
                den_x = np.exp(lqx_x) - np.exp(lqm_x)
                c_x = np.where(den_x > 0, np.clip(num_x, 0, None) / den_x, 1.0)
                num_y = np.exp(lf_y) - np.exp(lqm_y)
                den_y = np.exp(lqy_y) - np.exp(lqm_y)
                c_y = np.where(den_y > 0, np.clip(num_y, 0, None) / den_y, 1.0)
            u = np.exp(logu)
            ax = np.where(same, logu < lb_x, u < c_x)
            ay = np.where(same, logu < lb_y, u < c_y)
        x[idx[ax]] = xp[ax]
        lpx[idx[ax]] = lpxp[ax]
        y[idx[ay]] = yp[ay]
        lpy[idx[ay]] = lpyp[ay]
        met = same & ax & ay
        tau[idx[met]] = t
        active[idx[met]] = False
    return tau


def expo_example(M=10_000):
    kappa, sig = 3.0, math.sqrt(3.0)
    logpi = lambda z: np.where(z[:, 0] >= 0, -z[:, 0], -np.inf)
    mean_fn = lambda z: z + kappa
    init = lambda n: rng.exponential(1.0, size=(n, 1))
    print("[1] Expo(1) target, N(z+3, 3) proposals, 10,000 reps (published: SQ/MI 74.0, SQ/MR 75.6, C/MI 61.3, C/MR 62.2)")
    for qname, qc in [("Q_MI", q_mi), ("Q_MR", q_mr)]:
        for ker in ["SQ", "C"]:
            tau = run(logpi, mean_fn, sig, init, qc, ker, M, rng)
            print("    P_%-2s with %s: mean tau %.1f (s.e. %.2f)" % (ker, qname, tau.mean(), tau.std() / math.sqrt(M)))
            sys.stdout.flush()


def gauss_example(M=2000):
    print("[2] N(0,I_d), ell=2.38, reflection-maximal proposals, iid target inits, %d reps" % M)
    for d in [1, 5, 10, 20, 50]:
        sig = 2.38 / math.sqrt(d)
        logpi = lambda z: -0.5 * np.sum(z * z, 1)
        mean_fn = lambda z: z
        init = lambda n: rng.standard_normal((n, d))
        out = []
        for ker in ["SQ", "C"]:
            tau = run(logpi, mean_fn, sig, init, q_mr, ker, M, rng)
            out.append("P_%s %.1f (%.1f)" % (ker, tau.mean(), tau.std() / math.sqrt(M)))
        print("    d=%3d  %s" % (d, "   ".join(out)))
        sys.stdout.flush()


if __name__ == "__main__":
    expo_example()
    gauss_example()

"""Small numerical checks of MH-coupling primitives (research notes for telescopic).

Checks:
A. Reflection-maximal coupling of N(mu1, S), N(mu2, S): Y-marginal and P(X=Y) = 2 Phi(-|Delta|/2).
B. Rejection ("gamma") maximal coupling (JOA20 Alg. 2) and the Gerber-Lee eta-variant:
   mean / variance of the number of draws versus closed forms 2 and 2(1-c)/c etc.
C. Coupled RWM on N(0, I_d), ell = 2.38, independent target inits, common uniform:
   meeting times under reflection-maximal vs maximal-independent proposal couplings
   (compare d = 10 with O'Leary 2021 Table 1: ~30 vs ~279).
D. Floating-point pitfall: recomputing Y = mu2 + L(xi + z) instead of copying X.
"""
import math
import sys
import time

import numpy as np

rng = np.random.default_rng(20261023)


def Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------- A
def reflection_maximal(mu1, mu2, L, Linv, rng):
    d = mu1.shape[0]
    z = Linv @ (mu1 - mu2)
    nz = np.linalg.norm(z)
    xi = rng.standard_normal(d)
    x = mu1 + L @ xi
    if nz == 0.0:
        return x, x.copy(), True
    e = z / nz
    exi = e @ xi
    # log s(xi + z) - log s(xi) = -0.5[(exi + nz)^2 - exi^2]
    logr = -0.5 * ((exi + nz) ** 2 - exi**2)
    if math.log(rng.uniform()) < logr:
        return x, x.copy(), True  # copy X: exact equality by construction
    eta = xi - 2.0 * exi * e
    return x, mu2 + L @ eta, False


def check_A(n=100_000):
    d = 3
    A = rng.standard_normal((d, d))
    S = A @ A.T + 0.5 * np.eye(d)
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    mu1 = np.array([0.0, 0.0, 0.0])
    mu2 = np.array([0.6, -0.3, 0.2])
    delta = np.linalg.norm(Linv @ (mu1 - mu2))
    ys = np.empty((n, d))
    meet = 0
    for i in range(n):
        x, y, m = reflection_maximal(mu1, mu2, L, Linv, rng)
        ys[i] = y
        meet += m
    print("[A] reflection-maximal, d=3")
    print("    P(X=Y) empirical %.4f  theory 2Phi(-|D|/2) = %.4f" % (meet / n, 2 * Phi(-delta / 2)))
    print("    Y mean error (max abs): %.4f" % np.max(np.abs(ys.mean(0) - mu2)))
    print("    Y cov error (max abs):  %.4f  (|S| max %.2f)" % (np.max(np.abs(np.cov(ys.T) - S)), np.max(np.abs(S))))


# ---------------------------------------------------------------- B
def gamma_coupling_counts(delta, eta, n, rng):
    """N(0,1) vs N(delta,1); returns number of draws per call and meet flags.
    eta = 1 is Thorisson/Johnson maximal coupling; eta < 1 is the Gerber-Lee variant."""
    logp = lambda x: -0.5 * x * x
    logq = lambda x: -0.5 * (x - delta) ** 2
    x = rng.standard_normal(n)
    w = rng.uniform(size=n)
    meet = np.log(w) <= np.minimum(math.log(eta), logq(x) - logp(x))
    draws = np.ones(n)
    todo = ~meet
    while todo.any():
        k = todo.sum()
        y = delta + rng.standard_normal(k)
        ws = rng.uniform(size=k)
        acc = np.log(ws) > np.minimum(0.0, math.log(eta) + logp(y) - logq(y))
        draws[todo] += 1
        idx = np.flatnonzero(todo)
        todo[idx[acc]] = False
    return draws, meet


def check_B(n=400_000):
    print("[B] rejection maximal coupling N(0,1) vs N(delta,1): draws per call")
    for delta in [2.0, 0.5, 0.1, 0.02]:
        tv = 2 * Phi(delta / 2) - 1
        for eta in [1.0, 0.9]:
            draws, meet = gamma_coupling_counts(delta, eta, n, rng)
            if eta == 1.0:
                c = 1 - tv
                var_th = 2 * (1 - tv) / tv
            else:
                # c_eta = int min(eta p, q); numerically integrate on a grid
                g = np.linspace(-12, 12 + delta, 200001)
                pdf = lambda m: np.exp(-0.5 * (g - m) ** 2) / math.sqrt(2 * math.pi)
                c = np.trapezoid(np.minimum(eta * pdf(0.0), pdf(delta)), g)
                var_th = 2 * c / (1 - c)
            print(
                "    delta=%5.2f eta=%.1f  TV=%.4f  P(meet) emp %.4f th %.4f | E[draws] %.3f (th 2) | Var emp %8.2f th %8.2f | max %d"
                % (delta, eta, tv, meet.mean(), c, draws.mean(), draws.var(), var_th, draws.max())
            )


# ---------------------------------------------------------------- C
def coupled_rwm_meeting_times(d, coupling, M=1000, ell=2.38, tmax=200_000, rng=rng):
    sig = ell / math.sqrt(d)
    x = rng.standard_normal((M, d))
    y = rng.standard_normal((M, d))
    lpx = -0.5 * np.sum(x * x, 1)
    lpy = -0.5 * np.sum(y * y, 1)
    tau = np.full(M, -1)
    active = np.ones(M, bool)
    t = 0
    ndraws = 0
    while active.any() and t < tmax:
        t += 1
        idx = np.flatnonzero(active)
        k = idx.size
        xa, ya = x[idx], y[idx]
        if coupling == "reflmax":
            z = (xa - ya) / sig
            nz = np.linalg.norm(z, axis=1)
            e = z / nz[:, None]
            xi = rng.standard_normal((k, d))
            exi = np.sum(e * xi, 1)
            logr = -0.5 * ((exi + nz) ** 2 - exi**2)
            prop_meet = np.log(rng.uniform(size=k)) < logr
            xp = xa + sig * xi
            eta = xi - 2.0 * exi[:, None] * e
            yp = np.where(prop_meet[:, None], xp, ya + sig * eta)
        elif coupling == "maxindep":
            xi = rng.standard_normal((k, d))
            xp = xa + sig * xi
            # log q(xp; y) - log q(xp; x)
            lq = lambda pts, m: -0.5 * np.sum((pts - m) ** 2, 1) / sig**2
            prop_meet = np.log(rng.uniform(size=k)) <= lq(xp, ya) - lq(xp, xa)
            yp = xp.copy()
            todo = ~prop_meet
            while todo.any():
                j = np.flatnonzero(todo)
                cand = ya[j] + sig * rng.standard_normal((j.size, d))
                ndraws += j.size
                acc = np.log(rng.uniform(size=j.size)) > lq(cand, xa[j]) - lq(cand, ya[j])
                yp[j[acc]] = cand[acc]
                todo[j[acc]] = False
        else:
            raise ValueError(coupling)
        lpxp = -0.5 * np.sum(xp * xp, 1)
        lpyp = np.where(prop_meet, lpxp, -0.5 * np.sum(yp * yp, 1))
        logu = np.log(rng.uniform(size=k))  # common uniform
        accx = logu < lpxp - lpx[idx]
        accy = logu < lpyp - lpy[idx]
        x[idx[accx]] = xp[accx]
        lpx[idx[accx]] = lpxp[accx]
        y[idx[accy]] = yp[accy]
        lpy[idx[accy]] = lpyp[accy]
        met = prop_meet & accx & accy  # flag-based meeting detection
        tau[idx[met]] = t
        active[idx[met]] = False
        # sanity: flagged meetings are exact float equality
        assert np.all(x[idx[met]] == y[idx[met]])
    return tau


def check_C():
    print("[C] coupled RWM on N(0,I_d), ell=2.38, common uniform, independent target inits, M=1000 pairs")
    for coupling, dims in [("reflmax", [1, 2, 5, 10, 20, 50, 100]), ("maxindep", [1, 2, 4, 6, 8, 10])]:
        for d in dims:
            t0 = time.time()
            tau = coupled_rwm_meeting_times(d, coupling)
            ok = tau > 0
            print(
                "    %-8s d=%4d  mean tau %8.1f  (s.e. %6.1f)  median %6d  q99 %7d  unmet %d  [%.1fs]"
                % (coupling, d, tau[ok].mean(), tau[ok].std() / math.sqrt(ok.sum()), np.median(tau[ok]),
                   np.quantile(tau[ok], 0.99), (~ok).sum(), time.time() - t0)
            )
            sys.stdout.flush()


# ---------------------------------------------------------------- D
def check_D(n=20_000, d=10):
    A = rng.standard_normal((d, d))
    S = A @ A.T + np.eye(d)
    L = np.linalg.cholesky(S)
    neq = 0
    maxdiff = 0.0
    for _ in range(n):
        mu1 = rng.standard_normal(d)
        mu2 = mu1 + 1e-2 * rng.standard_normal(d)
        xi = rng.standard_normal(d)
        z = np.linalg.solve(L, mu1 - mu2)
        X = mu1 + L @ xi
        Y = mu2 + L @ (xi + z)  # mathematically equal to X
        if not np.array_equal(X, Y):
            neq += 1
            maxdiff = max(maxdiff, np.max(np.abs(X - Y)))
    print("[D] recomputing Y on the meeting branch: X != Y (bitwise) in %.1f%% of %d cases; max |X-Y| = %.2e"
          % (100 * neq / n, n, maxdiff))


if __name__ == "__main__":
    which = [w for w in (sys.argv[1:] or ["A", "B", "C", "D"]) if w != "E"]
    for w in which:
        {"A": check_A, "B": check_B, "C": check_C, "D": check_D}[w]()


# ---------------------------------------------------------------- E
def twoscale_gcrn_meeting_times(d, thresh_in_h2, M=200, ell=2.38, tmax=400_000, rng=rng):
    """Two-scale coupling (Papp & Sherlock): GCRN when |x-y|^2 >= delta, reflection-maximal otherwise.
    Target N(0, I_d) so grad log pi(x) = -x. Common uniform for acceptance."""
    sig = ell / math.sqrt(d)
    delta = thresh_in_h2 * sig**2
    x = rng.standard_normal((M, d))
    y = rng.standard_normal((M, d))
    lpx = -0.5 * np.sum(x * x, 1)
    lpy = -0.5 * np.sum(y * y, 1)
    tau = np.full(M, -1)
    active = np.ones(M, bool)
    t = 0
    while active.any() and t < tmax:
        t += 1
        idx = np.flatnonzero(active)
        k = idx.size
        xa, ya = x[idx], y[idx]
        sq = np.sum((xa - ya) ** 2, 1)
        near = sq < delta
        xp = np.empty_like(xa)
        yp = np.empty_like(ya)
        prop_meet = np.zeros(k, bool)
        # GCRN branch
        far = ~near
        if far.any():
            nx = -xa[far] / np.linalg.norm(xa[far], axis=1)[:, None]
            ny = -ya[far] / np.linalg.norm(ya[far], axis=1)[:, None]
            Z = rng.standard_normal((far.sum(), d))
            Zg = rng.standard_normal(far.sum())
            Zx = Z + (Zg - np.sum(nx * Z, 1))[:, None] * nx
            Zy = Z + (Zg - np.sum(ny * Z, 1))[:, None] * ny
            xp[far] = xa[far] + sig * Zx
            yp[far] = ya[far] + sig * Zy
        # reflection-maximal branch
        if near.any():
            z = (xa[near] - ya[near]) / sig
            nz = np.linalg.norm(z, axis=1)
            e = z / nz[:, None]
            xi = rng.standard_normal((near.sum(), d))
            exi = np.sum(e * xi, 1)
            logr = -0.5 * ((exi + nz) ** 2 - exi**2)
            pm = np.log(rng.uniform(size=near.sum())) < logr
            xpn = xa[near] + sig * xi
            ypn = np.where(pm[:, None], xpn, ya[near] + sig * (xi - 2.0 * exi[:, None] * e))
            xp[near] = xpn
            yp[near] = ypn
            prop_meet[near] = pm
        lpxp = -0.5 * np.sum(xp * xp, 1)
        lpyp = np.where(prop_meet, lpxp, -0.5 * np.sum(yp * yp, 1))
        logu = np.log(rng.uniform(size=k))
        accx = logu < lpxp - lpx[idx]
        accy = logu < lpyp - lpy[idx]
        x[idx[accx]] = xp[accx]
        lpx[idx[accx]] = lpxp[accx]
        y[idx[accy]] = yp[accy]
        lpy[idx[accy]] = lpyp[accy]
        met = prop_meet & accx & accy
        tau[idx[met]] = t
        active[idx[met]] = False
    return tau


def check_E():
    print("[E1] reflection-maximal, larger d (M=300 pairs)")
    for d in [200, 400]:
        t0 = time.time()
        tau = coupled_rwm_meeting_times(d, "reflmax", M=300)
        ok = tau > 0
        print("    reflmax d=%4d mean tau %8.1f (s.e. %6.1f) median %6d unmet %d [%.1fs]"
              % (d, tau[ok].mean(), tau[ok].std() / math.sqrt(ok.sum()), np.median(tau[ok]), (~ok).sum(), time.time() - t0))
        sys.stdout.flush()
    print("[E2] two-scale GCRN -> reflection-maximal, threshold delta = c * h^2 (M=200 pairs)")
    for d in [10, 100, 400, 1000]:
        for c in [1.0, 10.0]:
            t0 = time.time()
            tau = twoscale_gcrn_meeting_times(d, c)
            ok = tau > 0
            print("    d=%5d c=%4.0f mean tau %8.1f (s.e. %6.1f) median %7d q99 %7d unmet %d [%.1fs]"
                  % (d, c, tau[ok].mean(), tau[ok].std() / math.sqrt(ok.sum()), np.median(tau[ok]),
                     np.quantile(tau[ok], 0.99), (~ok).sum(), time.time() - t0))
            sys.stdout.flush()


if __name__ == "__main__" and "E" in sys.argv[1:]:
    check_E()

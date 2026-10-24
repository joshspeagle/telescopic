"""Feasibility experiment (UNPUBLISHED construction): coupling emcee-style ensemble moves.

Questions:
  1. Does a common-random-numbers (CRN) coupling of the affine-invariant stretch move
     (same random split, same partner indices, same stretch factors z, same acceptance
     uniforms) contract two ensembles towards each other on a Gaussian target?
  2. Same question for a differential-evolution (DE) move.
  3. If we mix in (prob gamma) a full-ensemble random-walk MH step whose proposals are drawn
     from a reflection-maximal coupling (Heng & Jacob 2019 style mixture), do the two
     ensembles ever meet exactly, and how do meeting times scale with dimension?
  4. Baseline: single-chain RWMH with reflection-maximal coupling (Jacob, O'Leary, Atchade 2020).

Only NumPy is used. Target: standard Gaussian N(0, I_d) (the affine-invariant moves and their CRN
coupling are affine-equivariant, so any non-degenerate Gaussian gives the same law of the
coupled dynamics after mapping the initial ensembles).
"""
import sys
import time
import numpy as np


def logpi(X):
    return -0.5 * np.sum(X * X, axis=-1)


def refl_max_coupling(rng, mu1, mu2, sigma):
    """Reflection-maximal coupling of N(mu1, sigma^2 I) and N(mu2, sigma^2 I) (flattened)."""
    z = (mu1 - mu2) / sigma
    nz = np.linalg.norm(z)
    xdot = rng.standard_normal(mu1.shape)
    if nz == 0.0:
        return mu1 + sigma * xdot, mu2 + sigma * xdot, True
    e = z / nz
    logu = np.log(rng.uniform())
    # accept identity (meeting) with prob min(1, phi(xdot + z) / phi(xdot))
    if logu + (-0.5 * xdot @ xdot) < -0.5 * (xdot + z) @ (xdot + z):
        ydot = xdot + z
        met = True
    else:
        ydot = xdot - 2.0 * (e @ xdot) * e
        met = False
    return mu1 + sigma * xdot, mu2 + sigma * ydot, met


def stretch_sweep(rng, X1, X2, lp1, lp2, a=2.0):
    """One emcee-style sweep (two halves) applied to both ensembles with CRN."""
    K, d = X1.shape
    perm = rng.permutation(K)  # shared random split
    halves = (perm[: K // 2], perm[K // 2:])
    for h in range(2):
        upd, comp = halves[h], halves[1 - h]
        n = len(upd)
        j = comp[rng.integers(0, len(comp), size=n)]  # shared partner indices
        z = ((a - 1.0) * rng.uniform(size=n) + 1.0) ** 2 / a  # shared stretch factors
        logu = np.log(rng.uniform(size=n))  # shared acceptance uniforms
        for X, lp in ((X1, lp1), (X2, lp2)):
            Y = X[j] + z[:, None] * (X[upd] - X[j])
            lpY = logpi(Y)
            acc = logu < (d - 1) * np.log(z) + lpY - lp[upd]
            X[upd[acc]] = Y[acc]
            lp[upd[acc]] = lpY[acc]


def de_sweep(rng, X1, X2, lp1, lp2):
    """Differential-evolution sweep (emcee DEMove-like) with CRN."""
    K, d = X1.shape
    perm = rng.permutation(K)
    halves = (perm[: K // 2], perm[K // 2:])
    g0 = 2.38 / np.sqrt(2 * d)
    for h in range(2):
        upd, comp = halves[h], halves[1 - h]
        n = len(upd)
        jl = np.array([rng.choice(comp, size=2, replace=False) for _ in range(n)])
        gam = g0 * (1.0 + 1e-5 * rng.standard_normal(n))
        logu = np.log(rng.uniform(size=n))
        for X, lp in ((X1, lp1), (X2, lp2)):
            Y = X[upd] + gam[:, None] * (X[jl[:, 0]] - X[jl[:, 1]])
            lpY = logpi(Y)
            acc = logu < lpY - lp[upd]
            X[upd[acc]] = Y[acc]
            lp[upd[acc]] = lpY[acc]


def joint_rwmh_maxcoupled(rng, X1, X2, lp1, lp2, sigma):
    """Full-ensemble RWMH step (product target), proposals from reflection-maximal coupling."""
    K, d = X1.shape
    x1, x2 = X1.ravel(), X2.ravel()
    p1, p2, _ = refl_max_coupling(rng, x1, x2, sigma)
    P1, P2 = p1.reshape(K, d), p2.reshape(K, d)
    lq1, lq2 = logpi(P1), logpi(P2)
    logu = np.log(rng.uniform())
    if logu < lq1.sum() - lp1.sum():
        X1[:] = P1
        lp1[:] = lq1
    if logu < lq2.sum() - lp2.sum():
        X2[:] = P2
        lp2[:] = lq2


def run_ensemble(d, K, move, T, gamma=0.0, sigma=1e-3, seed=0, record_every=None):
    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal((K, d)) * 2.0
    X2 = rng.standard_normal((K, d)) * 2.0
    lp1, lp2 = logpi(X1), logpi(X2)
    sweep = stretch_sweep if move == "stretch" else de_sweep
    rec = []
    for t in range(1, T + 1):
        if gamma > 0 and rng.uniform() < gamma:
            joint_rwmh_maxcoupled(rng, X1, X2, lp1, lp2, sigma)
        else:
            sweep(rng, X1, X2, lp1, lp2)
        if np.array_equal(X1, X2):
            return t, rec
        if record_every and t % record_every == 0:
            rec.append((t, np.linalg.norm(X1 - X2)))
    return None, rec


def run_rwmh_coupled(d, T, seed=0):
    """Single-chain RWMH, reflection-maximal coupling of proposals, common uniform."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(d) * 2.0
    y = rng.standard_normal(d) * 2.0
    s = 2.38 / np.sqrt(d)
    lx, ly = logpi(x), logpi(y)
    for t in range(1, T + 1):
        px, py, _ = refl_max_coupling(rng, x, y, s)
        lpx, lpy = logpi(px), logpi(py)
        logu = np.log(rng.uniform())
        if logu < lpx - lx:
            x, lx = px, lpx
        if logu < lpy - ly:
            y, ly = py, lpy
        if np.array_equal(x, y):
            return t
    return None


def summarize(ts, cap):
    met = [t for t in ts if t is not None]
    n_cap = sum(t is None for t in ts)
    if met:
        q = np.quantile(met, [0.5, 0.9])
        return f"met {len(met)}/{len(ts)}; median {q[0]:.0f}, 90% {q[1]:.0f}; censored at {cap}: {n_cap}"
    return f"met 0/{len(ts)} within {cap}"


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if part in ("crn", "all"):
        print("== Part 1/2: pure CRN coupling (no mixture); Frobenius distance between ensembles ==")
        for move in ("stretch", "de"):
            for d in (2, 5, 10, 20):
                K = max(2 * d, 8)
                K += K % 2
                for seed in range(3):
                    _, rec = run_ensemble(d, K, move, T=20000, gamma=0.0, seed=seed, record_every=5000)
                    dist = ", ".join(f"t={t}:{v:.2e}" for t, v in rec)
                    print(f"{move:7s} d={d:3d} K={K:3d} seed={seed}: {dist}", flush=True)
    if part in ("mix", "all"):
        print("== Part 3: CRN ensemble move + gamma=1/20 full-ensemble max-coupled RWMH (sigma=1e-3) ==")
        for move in ("stretch",):
            for d in (1, 2, 5, 10):
                K = max(2 * d, 8)
                K += K % 2
                cap = 50000
                ts = [run_ensemble(d, K, move, T=cap, gamma=1 / 20, sigma=1e-3, seed=100 + s)[0] for s in range(10)]
                print(f"{move:7s} d={d:3d} K={K:3d}: {summarize(ts, cap)}", flush=True)
    if part in ("rwmh", "all"):
        print("== Part 4: baseline single-chain RWMH, reflection-maximal coupling ==")
        for d in (1, 2, 5, 10, 20, 50):
            cap = 200000
            ts = [run_rwmh_coupled(d, T=cap, seed=200 + s) for s in range(20)]
            print(f"rwmh d={d:3d}: {summarize(ts, cap)}", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s")

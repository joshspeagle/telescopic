"""Feasibility experiments, part 2 (UNPUBLISHED constructions).

A. Local stability of the CRN-coupled stretch move: start ensemble 2 at ensemble 1 + 1e-8 noise
   and watch whether the difference shrinks (contractive) or grows (expansive).
B. CRN-coupled elliptical slice sampling (ESS; Murray, Adams & MacKay 2010) on a latent Gaussian
   model (Gaussian-process prior + Bernoulli-logit likelihood), alone and mixed (prob gamma) with a
   preconditioned Crank-Nicolson (pCN) step whose proposals come from a reflection-maximal
   coupling (Heng & Jacob 2019 style mixture). Records meeting times versus dimension.
"""
import sys
import time
import numpy as np

sys.path.insert(0, ".")
from ensemble_coupling_sim import logpi, stretch_sweep, refl_max_coupling  # noqa: E402


def perturbation_test(d, K, seed, eps=1e-8, T=3000):
    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal((K, d))
    for _ in range(200):  # burn in ensemble 1 alone (both copies identical)
        X2 = X1.copy()
        lp1, lp2 = logpi(X1), logpi(X2)
        stretch_sweep(rng, X1, X2, lp1, lp2)
    X2 = X1 + eps * rng.standard_normal((K, d))
    lp1, lp2 = logpi(X1), logpi(X2)
    out = []
    for t in range(1, T + 1):
        stretch_sweep(rng, X1, X2, lp1, lp2)
        if t in (1, 10, 30, 100, 300, 1000, 3000):
            out.append((t, np.linalg.norm(X1 - X2)))
    return out


# ---------------- latent Gaussian model -----------------
def make_model(d, seed=0, ell=0.1):
    rng = np.random.default_rng(seed)
    s = np.linspace(0, 1, d)
    C = np.exp(-0.5 * (s[:, None] - s[None, :]) ** 2 / ell**2) + 1e-6 * np.eye(d)
    Lc = np.linalg.cholesky(C)
    f_true = Lc @ rng.standard_normal(d)
    y = (rng.uniform(size=d) < 1 / (1 + np.exp(-f_true))).astype(float)
    return Lc, y


def loglik(f, y):
    # Bernoulli-logit, numerically stable
    return np.sum(y * f - np.logaddexp(0.0, f))


def coupled_ess_step(rng, f1, f2, Lc, y, ll1, ll2):
    d = len(f1)
    nu = Lc @ rng.standard_normal(d)  # shared auxiliary Gaussian
    logu = np.log(rng.uniform())  # shared slice uniform
    theta0 = rng.uniform(0, 2 * np.pi)  # shared initial angle
    shrink_u = rng.uniform(size=200)  # shared stream for shrinkage draws
    out = []
    for f, ll in ((f1, ll1), (f2, ll2)):
        logy = ll + logu
        th = theta0
        lo, hi = th - 2 * np.pi, th
        i = 0
        while True:
            fp = f * np.cos(th) + nu * np.sin(th)
            llp = loglik(fp, y)
            if llp > logy:
                break
            if th < 0:
                lo = th
            else:
                hi = th
            th = lo + shrink_u[i] * (hi - lo)
            i += 1
            if i >= len(shrink_u):  # practically never; keep exactness by extending stream
                shrink_u = np.concatenate([shrink_u, rng.uniform(size=200)])
        out.append((fp, llp))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def coupled_pcn_step(rng, f1, f2, Lc, y, ll1, ll2, beta):
    """pCN: f' = rho f + beta * L xi, rho = sqrt(1-beta^2); prior-reversible, accept on likelihood.
    Proposals coupled by reflection-maximal coupling in whitened coordinates; common uniform."""
    rho = np.sqrt(1 - beta**2)
    w1 = np.linalg.solve(Lc, f1)
    w2 = np.linalg.solve(Lc, f2)
    p1, p2, _ = refl_max_coupling(rng, rho * w1, rho * w2, beta)
    q1, q2 = Lc @ p1, Lc @ p2
    if np.array_equal(p1, p2):
        q2 = q1  # exact equality of the proposals in original coordinates
    lq1, lq2 = loglik(q1, y), loglik(q2, y)
    logu = np.log(rng.uniform())
    if logu < lq1 - ll1:
        f1, ll1 = q1, lq1
    if logu < lq2 - ll2:
        f2, ll2 = q2, lq2
    return f1, f2, ll1, ll2


def run_ess(d, gamma, beta, T, seed, record=False):
    Lc, y = make_model(d, seed=12345)  # same data for all replicates of a given d
    rng = np.random.default_rng(seed)
    f1 = Lc @ rng.standard_normal(d)
    f2 = Lc @ rng.standard_normal(d)
    ll1, ll2 = loglik(f1, y), loglik(f2, y)
    rec = []
    for t in range(1, T + 1):
        if gamma > 0 and rng.uniform() < gamma:
            f1, f2, ll1, ll2 = coupled_pcn_step(rng, f1, f2, Lc, y, ll1, ll2, beta)
        else:
            f1, f2, ll1, ll2 = coupled_ess_step(rng, f1, f2, Lc, y, ll1, ll2)
        if np.array_equal(f1, f2):
            return t, rec
        if record and t in (1, 5, 10, 20, 50, 100, 200, 500):
            rec.append((t, np.linalg.norm(f1 - f2)))
    return None, rec


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if part in ("pert", "all"):
        print("== A. CRN stretch move, ensembles started 1e-8 apart (Frobenius distance) ==")
        for d in (1, 2, 5, 10, 20):
            K = max(2 * d, 8)
            K += K % 2
            for seed in range(2):
                out = perturbation_test(d, K, seed)
                print(f"d={d:3d} K={K:3d} seed={seed}: " + ", ".join(f"t={t}:{v:.1e}" for t, v in out), flush=True)
    if part in ("ess", "all"):
        print("== B1. CRN-coupled ESS alone: distance between chains ==")
        for d in (10, 50, 200):
            for seed in range(2):
                _, rec = run_ess(d, gamma=0.0, beta=1e-2, T=500, seed=seed, record=True)
                print(f"d={d:4d} seed={seed}: " + ", ".join(f"t={t}:{v:.1e}" for t, v in rec), flush=True)
        print("== B2. CRN-coupled ESS + gamma=1/20 reflection-max-coupled pCN (beta=1e-2): meeting times ==")
        for d in (10, 50, 100, 200, 400):
            cap = 5000
            ts = [run_ess(d, gamma=1 / 20, beta=1e-2, T=cap, seed=1000 + s)[0] for s in range(20)]
            met = [t for t in ts if t is not None]
            if met:
                q = np.quantile(met, [0.5, 0.9])
                print(f"d={d:4d}: met {len(met)}/20, median {q[0]:.0f}, 90% {q[1]:.0f}, max {max(met)}", flush=True)
            else:
                print(f"d={d:4d}: met 0/20 within {cap}", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s")

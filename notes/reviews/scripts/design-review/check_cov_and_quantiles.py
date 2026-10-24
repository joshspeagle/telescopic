"""(1) Spec 3.4 covariance: U-statistic unbiased, plug-in biased by -Cov(a_bar).
(2) Spec 3.4 quantiles: does the unspecified weighting of the isotonic projection matter?

(1) Synthetic replicates: each replicate's 'measure' is the empirical law of 5 draws from
N(mu, C), so a_r (its mean) and b_r (its mean of x x^T) are unbiased; R = 10 replicates.
(2) Signed measures from an exact AR(1) kernel (rho = 0.9) coupled by reflection-maximal,
lag 1, pi0 = point mass at 10, k = 2, m = 10 (many negative weights), R = 200 replicates.
Pooled CDF over sorted atoms, projected by scipy.optimize.isotonic_regression with
(a) unit weights per atom, (b) weights = gaps between consecutive atoms, then clipped to
[0, 1] and inverted (smallest atom with F >= q). Truth: N(0, 1) quantiles.
"""
import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng(51)

# (1)
mu = np.array([1.0, -2.0])
C = np.array([[2.0, 0.6], [0.6, 1.0]])
Lc = np.linalg.cholesky(C)
R, n_draw, reps = 10, 5, 200_000
xs = mu + rng.standard_normal((reps, R, n_draw, 2)) @ Lc.T
a = xs.mean(2)  # (reps, R, 2)
b = np.einsum("srni,srnj->srij", xs, xs) / n_draw  # (reps, R, 2, 2)
abar = a.mean(1)
outer = np.einsum("si,sj->sij", abar, abar)
dev = a - abar[:, None, :]
S = np.einsum("sri,srj->sij", dev, dev) / (R - 1)
plug = b.mean(1) - outer
ustat = plug + S / R
print("true Cov:", C.ravel())
print(f"plug-in mean {plug.mean(0).ravel().round(4)}  (expected bias -C/(n R) = {(-C / (n_draw * R)).ravel()})")
print(f"U-stat  mean {ustat.mean(0).ravel().round(4)}  (se ~ {(ustat.std(0) / np.sqrt(reps)).ravel().round(4)})")

# (2)
rho, s, K, M, R2 = 0.9, np.sqrt(0.19), 2, 10, 200


def one_measure():
    x = np.array(10.0)
    y = np.array(10.0)
    X = [float(x)]
    Y = [float(y)]
    t, tau = 0, None
    x = rho * x + s * rng.standard_normal()
    X.append(float(x))
    t = 1
    while tau is None or t < M:
        t += 1
        z = rng.standard_normal()
        if tau is None:
            dl = (rho * x - rho * y) / s
            xn = rho * x + s * z
            if np.log(rng.random()) < -z * dl - 0.5 * dl * dl:
                yn, tau = xn, t
            else:
                yn = rho * y - s * z
        else:
            xn = rho * x + s * z
            yn = xn
        x, y = xn, yn
        X.append(float(x))
        Y.append(float(y))
    atoms = [X[l] for l in range(K, M + 1)]
    w = [1 / (M - K + 1)] * (M - K + 1)
    for tt in range(K + 1, tau):
        c = min(tt - K, M - K + 1) / (M - K + 1)
        atoms += [X[tt], Y[tt - 1]]
        w += [c, -c]
    return np.array(atoms), np.array(w)


ms = [one_measure() for _ in range(R2)]
atoms = np.concatenate([m_[0] for m_ in ms])
w = np.concatenate([m_[1] for m_ in ms]) / R2
order = np.argsort(atoms, kind="stable")
xa, wa = atoms[order], w[order]
F = np.cumsum(wa)
print(f"\nR={R2}: {len(xa)} atoms, negative-weight share {np.mean(wa < 0):.3f}, raw CDF range "
      f"[{F.min():.3f}, {F.max():.3f}], non-monotone steps {np.sum(np.diff(F) < 0)}")
gaps = np.diff(xa, append=xa[-1] + (xa[-1] - xa[-2]))
for label, wts in (("unit weights", np.ones_like(xa)), ("gap weights", np.maximum(gaps, 1e-300))):
    Fi = np.clip(optimize.isotonic_regression(F, weights=wts).x, 0, 1)
    qs = [xa[np.searchsorted(Fi, q, side="left")] for q in (0.16, 0.5, 0.84)]
    print(f"  {label:12s}: q16 {qs[0]:+.4f}  q50 {qs[1]:+.4f}  q84 {qs[2]:+.4f}")
run_max = np.clip(np.maximum.accumulate(F), 0, 1)
print(f"  running max : q16 {xa[np.searchsorted(run_max, 0.16)]:+.4f}  q50 {xa[np.searchsorted(run_max, 0.5)]:+.4f}  "
      f"q84 {xa[np.searchsorted(run_max, 0.84)]:+.4f}")
print(f"  truth       : q16 {stats.norm.ppf(0.16):+.4f}  q50 {0:+.4f}  q84 {stats.norm.ppf(0.84):+.4f}")
# replicate-bootstrap spread of the unit-weight median, for scale
meds = []
for _ in range(200):
    pick = rng.integers(0, R2, R2)
    at = np.concatenate([ms[i][0] for i in pick])
    ww = np.concatenate([ms[i][1] for i in pick]) / R2
    o = np.argsort(at, kind="stable")
    Fb = np.clip(optimize.isotonic_regression(np.cumsum(ww[o])).x, 0, 1)
    meds.append(at[o][np.searchsorted(Fb, 0.5)])
print(f"  bootstrap sd of the unit-weight median: {np.std(meds):.4f}")

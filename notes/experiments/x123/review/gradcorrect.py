"""Review of x123: are the registered gradients what the registration says they are?

Question: is every gradient of group A the derivative of its log_prob as computed, do
the wrong gradients of groups B and C carry the stated errors, and is
families.jacobi_gradient the derivative of the log_prob that the Jacobi solver computes?

Design: for each family of group A, at 3 draws of its start (seeds 500000 to 500002,
generator seed + 1000 times the family's index), the derivative of log_prob along the
difference u of two further draws by a fourth-order central difference, at steps from
1e-2 to 1e-7, against grad . u; the row prints the worst of the 3 draws, each taken at
its best step, relative to |grad| . |u|. Then the errors of a sample of wrong gradients
against their stated forms at 4 points (generator 500010), and the directional
derivative of the solver's log_prob against jacobi_gradient and against the exact
gradient.

Run: PYTHONPATH=src:notes/experiments/x123 python notes/experiments/x123/review/gradcorrect.py > gradcorrect.out

Result: see gradcorrect.out and REVIEW.md, the checks that found nothing. The float64
families that are smooth agree to about 1e-13. The rows above 1e-6 are float32 families,
families with a constant of 1e9 or more in log_prob, and families whose gradient is 0
along u (a ratio of 0 over 0): there the difference of this script is what is limited.
The wrong gradients carry exactly the stated errors, and jacobi_gradient agrees with the
numerical derivative to 1e-9.
"""

import warnings

import families as F
import numpy as np
from scipy import stats

from telescopic.initial import as_initial, draw

warnings.simplefilter("ignore")


def dderiv(lp, x, u, hh):
    pts = np.stack([x + k * hh * u for k in (-2, -1, 1, 2)])
    f = np.asarray(lp(pts), dtype=np.float64)
    return (f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * hh)


worst = []
for k, (name, (case, vec)) in enumerate(F.GROUP_A.items()):
    res = []
    for seed in (500000, 500001, 500002):
        lp, g, init = F.resolve(case, seed)
        if not vec:
            lp_, g_ = lp, g
            lp = lambda X, lp_=lp_: np.array([lp_(r) for r in X])
            g = lambda X, g_=g_: np.array([g_(r) for r in X])
        init = as_initial(init)
        rng = np.random.default_rng(seed + 1000 * k)
        x = draw(init, rng)
        a = draw(init, rng, len(x))
        b = draw(init, rng, len(x))
        u = a - b
        if not np.any(u):
            u = rng.standard_normal(len(x)) * np.where(x != 0, np.abs(x), 1.0)
        f0 = float(np.asarray(lp(x[None]))[0])
        if not np.isfinite(f0):
            continue
        gu = float(np.asarray(g(x[None]), dtype=np.float64)[0] @ u)
        scale = float(np.abs(np.asarray(g(x[None]), dtype=np.float64)[0]) @ np.abs(u)) + 1e-300
        best = np.inf
        for hh in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            dd = dderiv(lp, x, u, hh)
            if np.isfinite(dd):
                best = min(best, abs(dd - gu) / scale)
        res.append(best)
    w = max(res) if res else np.nan
    worst.append((w, k, name))
    print(f"[{k:3d}] {name[:66]:<66} worst rel discrepancy over 3 draws {w:.2e}", flush=True)
print("largest:", sorted(worst, reverse=True)[:12])
# wrong gradients: stated errors
x = np.random.default_rng(500010).standard_normal((4, 5)) * 2


def err(pair):
    (_, g, _), (_, g2, _) = (F.resolve(c, 500010) for c in pair)
    return np.asarray(g2(x), float) - np.asarray(g(x), float)


C = F.GROUP_C
print("wrong sign: e + 2g ->", np.abs(err(C["N(0,I) d=5: wrong sign"]) - 2 * x).max())
print("twice: e - g ->", np.abs(err(C["N(0,I) d=5: twice"]) + x).max())
e = err(C["N(0,I) d=5, start sd 2: sign of coordinate 0"])
print("flip0: e0 - 2x0, others ->", np.abs(e[:, 0] - 2 * x[:, 0]).max(), np.abs(e[:, 1:]).max())
e = err(C["N(0,I) d=5, start sd 2: coordinate 0 set to 0"])
print("zero0:", np.abs(e[:, 0] - x[:, 0]).max(), np.abs(e[:, 1:]).max())
e = err(C["N(0,I) d=5, start sd 100: grad + 2"])
print("plus2:", np.abs(e - 2).max())
e = err(C["N(0,I) d=5, start sd (0.01, 2, ...): sign of coordinate 0"])
print("tight flip0:", np.abs(e[:, 0] - 2 * x[:, 0]).max(), np.abs(e[:, 1:]).max())
(r, w) = F.rotated_axis(1e2)
x10 = np.random.default_rng(1).standard_normal((3, 10))
gr, gw = r[1](x10), w[1](x10)
rot = stats.ortho_group.rvs(10, random_state=0)
ax = rot[:, 0]
print(
    "rotated: wrong = reflect along widest axis ->",
    np.abs(gw - (gr - 2 * np.outer(gr @ ax, ax))).max(),
    "axis variance",
    ax @ r[2].cov @ ax / 4,
    "max eig",
    np.linalg.eigvalsh(r[2].cov / 4).max(),
)
B = F.GROUP_B
lp, g, i = B["N(0,I) d=5: grad x 1.005"]
print("x1.005:", np.abs(g(x) + 1.005 * x).max())
lp, g, i = B["N(0,I) d=5: grad + 0.02"]
print("+0.02:", np.abs(g(x) + x - 0.02).max())
lp, g, i = B["logistic n=1e4 d=5: without the prior"]
lp0, g0, _ = F.logistic()
print("no prior: e - x ->", np.abs(g(x) - g0(x) - x).max())
# Jacobi: 'exact gradient' is -A^-1 x; and jacobi_gradient is the derivative of the computed log_prob
for rtol in (1e-3, 1e-5, 1e-8):
    lp, g, i = F.loose_solver(rtol)
    xx = x[0]
    u = x[1] - x[2]
    dd = dderiv(lp, xx, u, 1e-6)
    print(
        f"jacobi rtol {rtol:g}: exact-grad.u {float(g(xx[None])[0] @ u):.10f} jacobi_gradient.u {float(F.jacobi_gradient(xx[None], rtol)[0] @ u):.10f} numeric {dd:.10f}"
    )

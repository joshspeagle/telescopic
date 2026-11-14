# The meeting-time model behind the predictions: tau ~ 1/gamma + ln K / (-ln f), f the geometric-mean
# factor of the slowest whitened mode over the jitter, ln K = 6 calibrated on the package's Gaussian rows.
import math

import numpy as np


def theta(h):
    h = np.asarray(h, float)
    return np.where(h < 2, 2 * np.arcsin(np.minimum(h, 2) / 2), np.nan)


def gfac(eps, n, w, j, pts=401):
    u = 1 + j * np.linspace(-1, 1, pts) if j > 0 else np.array([1.0])
    t = theta(np.multiply.outer(eps * u, np.atleast_1d(w)))
    f = np.where(np.isnan(t), 1.0, np.abs(np.cos(n * np.nan_to_num(t))))
    return np.exp(np.log(np.maximum(f, 1e-300)).mean(axis=0))


def tau(eps, n, w, j, lnK=6.0, gamma=0.05):
    f = gfac(eps, n, w, j).max()
    return 1 / gamma + lnK / (-math.log(f)), f


def cost(eps, n, w, j):  # gradients per pair to meet: 2 chains x (n per trajectory)
    t, f = tau(eps, n, w, j)
    return t, f, t * 2 * ((1 - 0.05) * n + 0.05)


cases = {
    "gauss2": (np.linspace(0.9, 1.1, 5), [(1.13, 1, 0), (0.31, 4, 0.5)]),
    "gauss10": (np.linspace(0.75, 1.25, 9), [(0.79, 2, 0), (0.21, 5, 0.5)]),
}
for k, (w, arms) in cases.items():
    out = [cost(e, n, w, j) for e, n, j in arms]
    print(
        k,
        ["tau {:.1f} f {:.3f} grads {:.0f}".format(*o) for o in out],
        f"ratio CF/P {out[1][2] / out[0][2]:.2f}",
    )
# J widths at acceptance 0.8: steps from i137 (c tuned 0.8 jittered); gaussian10 default eps .78 n 2
for j in (0.2, 0.3, 0.5):
    print(
        f"gauss10 default J{j:.1f}",
        "tau {:.1f} f {:.3f} grads {:.0f}".format(*cost(0.78, 2, np.linspace(0.75, 1.25, 9), j)),
    )


# The arms of the revision at meeting_weight 0.2 (I1): the same model with gamma = 0.2.
def cost_g(eps, n, w, j, gamma):
    f = gfac(eps, n, w, j).max()
    t = 1 / gamma + 6 / (-math.log(f))
    return t, t * 2 * ((1 - gamma) * n + gamma)


for k, (w, arms) in cases.items():
    (pe, pn, pj), (ce, cn, cj) = arms
    p05 = cost_g(pe, pn, w, pj, 0.05)[1]
    p2 = cost_g(pe, pn, w, pj, 0.2)[1]
    c2 = cost_g(ce, cn, w, cj, 0.2)[1]
    c05 = cost_g(ce, cn, w, cj, 0.05)[1]
    print(
        f"{k} gamma 0.2: P {p2:.0f}, CF {c2:.0f} gradients per meeting; CF/P at 0.2 {c2 / p2:.2f}; CF at 0.2 over P at 0.05 {c2 / p05:.2f}; CF at 0.05 over P at 0.05 {c05 / p05:.2f}"
    )

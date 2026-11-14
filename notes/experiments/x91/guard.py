import math

import numpy as np

GRID = np.round(np.arange(-2000, 101) / 1000, 3)


def theta(h):
    h = np.asarray(h, float)
    return np.where(h < 2, 2 * np.arcsin(np.minimum(h, 2) / 2), np.nan)


def n_of(eps, n_given):
    return n_given if n_given is not None else math.ceil(1 / max(eps, 1e-3))


def resdist(eps, n, w):
    # distance of n*theta(eps*w) to the nearest j*pi with j>=1; phases below pi/2 measured to pi; 0 if unstable
    t = theta(np.multiply.outer(np.atleast_1d(eps), np.atleast_1d(w)))
    phi = n * np.nan_to_num(t, nan=0.0)
    j = np.maximum(1, np.round(phi / math.pi))
    d = np.abs(phi - j * math.pi)
    return np.where(np.isnan(t), 0.0, d)


def exact_factor(eps, n, w):
    t = theta(np.multiply.outer(np.atleast_1d(eps), np.atleast_1d(w)))
    return np.where(np.isnan(t), 1.0, np.abs(np.cos(n * np.nan_to_num(t))))


def guard(eps0, n_given, w_pred, delta, band, points=41):
    """returns (eps, action) action in kept/moved/fallback"""
    w_pred = np.atleast_1d(w_pred)
    if resdist(eps0, n_of(eps0, n_given), w_pred).min() >= delta:
        return eps0, "kept"
    steps = eps0 * np.exp(GRID)
    wb = (
        np.concatenate([w_pred * f for f in np.geomspace(1 / (1 + band), 1 + band, points)])
        if band > 0
        else w_pred
    )
    ok = np.array([resdist(e, n_of(e, n_given), wb).min() >= delta for e in steps])
    if ok.any():
        i = np.flatnonzero(ok)[np.argmin(np.abs(GRID[ok]))]
        return float(steps[i]), "moved"
    return eps0, "fallback"  # no grid step clears the band: the tuned step is kept

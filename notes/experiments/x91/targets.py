# Targets of the design probes (tuning only).
import functools

import numpy as np
from scipy import optimize, special

from telescopic.examples import Example, _gaussian_grad, _gaussian_log_prob
from telescopic.initial import Normal


def banana20():
    def lp(x):
        return -((1 - x[..., 0]) ** 2) - 10 * np.sum((x[..., 1:] - x[..., :1] ** 2) ** 2, axis=-1)

    def gr(x):
        r = x[..., 1:] - x[..., :1] ** 2
        g = np.empty_like(x)
        g[..., 1:] = -20 * r
        g[..., 0] = 2 * (1 - x[..., 0]) + 40 * x[..., 0] * np.sum(r, axis=-1)
        return g

    return Example("banana20", lp, gr, None, None, None)


class ExactBanana:
    def __init__(self, d):
        self.d = d

    def rvs(self, *, random_state):
        x0 = 1 + np.sqrt(0.5) * random_state.standard_normal()
        return np.concatenate(
            [[x0], x0**2 + np.sqrt(0.05) * random_state.standard_normal(self.d - 1)]
        )


def rotated(d=100, cond=1e4, seed=91):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    lam = np.geomspace(1, 1 / cond, d)
    cov = (q * lam) @ q.T
    prec = (q / lam) @ q.T
    return Example(
        "rotated",
        functools.partial(_gaussian_log_prob, precision=prec),
        functools.partial(_gaussian_grad, precision=prec),
        Normal(np.zeros(d), 4 * np.eye(d)),
        np.zeros(d),
        cov,
    )


def logistic():
    rng = np.random.default_rng(1)
    n = 200
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 5))])
    beta = np.array([-0.5, 1, -1, 0.5, 0, 2])
    y = (rng.random(n) < special.expit(X @ beta)).astype(float)

    def lp(b):
        eta = b @ X.T
        return np.sum(y * eta - np.logaddexp(0, eta), axis=-1) - np.sum(b**2, axis=-1) / 50

    def gr(b):
        return (y - special.expit(b @ X.T)) @ X - b / 25

    fit = optimize.minimize(lambda b: -lp(b), np.zeros(6), jac=lambda b: -gr(b), method="BFGS")
    return Example("logistic", lp, gr, Normal(fit.x, 100 * fit.hess_inv), None, None), X, y, fit

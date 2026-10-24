"""'Linear SBC' for unbiased MCMC (scratch research script).

Model: theta ~ N(0, 1); y_i | theta ~ N(theta, 1), i = 1..n.
For each replicate: draw theta~ from the prior, y from the likelihood, then run ONE
coupled RWMH pair (reflection-maximal proposals, common uniform, lag 1) on the posterior
and compute H_{k:m} for several test functions phi(theta, y).
Because E[H_phi(y) | y] = E[phi(theta, y) | y] exactly (any k, m), the replicate mean of
D = H_phi(y) - phi(theta~, y) has expectation 0 under a correct implementation.
Bugs: (a) sampler targets the prior (ignores data); (b) likelihood variance 1.5 instead of 1;
(c) prior ignored (flat prior).
"""
import numpy as np

import sys
n_obs, R = 5, 20_000
k, m = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(42)


def logpost(theta, sy, n, bug):
    lp = -0.5 * theta**2
    if bug == "prior_only":
        return lp
    if bug == "flat_prior":
        lp = 0.0 * theta
    var = 1.5 if bug == "lik_var" else 1.0
    return lp - 0.5 * (n * theta**2 - 2 * theta * sy) / var


def run(bug=None):
    theta_t = rng.normal(size=R)
    y = theta_t[:, None] + rng.normal(size=(R, n_obs))
    sy, syy = y.sum(1), (y**2).sum(1)
    sig = 2.4 / np.sqrt(n_obs + 1)
    T = max(600, m + 50)
    X = np.zeros((R, T + 1)); Y = np.zeros((R, T + 1))
    X[:, 0] = 3 * rng.normal(size=R); Y[:, 0] = 3 * rng.normal(size=R)
    lpx = logpost(X[:, 0], sy, n_obs, bug)
    xp = X[:, 0] + sig * rng.normal(size=R)
    lxp = logpost(xp, sy, n_obs, bug)
    acc = np.log(rng.uniform(size=R)) <= lxp - lpx
    X[:, 1] = np.where(acc, xp, X[:, 0]); lpx = np.where(acc, lxp, lpx)
    lpy = logpost(Y[:, 0], sy, n_obs, bug)
    tau = np.full(R, -1)
    for t in range(1, T):
        x, yy = X[:, t], Y[:, t - 1]
        eps = rng.normal(size=R)
        z = (x - yy) / sig
        met = np.log(rng.uniform(size=R)) <= np.minimum(0, -0.5 * (eps + z) ** 2 + 0.5 * eps**2)
        px = x + sig * eps
        py = np.where(met, px, yy - sig * eps)
        lpx_p = logpost(px, sy, n_obs, bug)
        lpy_p = np.where(met, lpx_p, logpost(py, sy, n_obs, bug))
        u = np.log(rng.uniform(size=R))
        ax, ay = u <= lpx_p - lpx, u <= lpy_p - lpy
        xn = np.where(ax, px, x); lpx = np.where(ax, lpx_p, lpx)
        yn = np.where(ay, py, yy); lpy = np.where(ay, lpy_p, lpy)
        done = tau > 0  # after meeting, Y_{t} := X_{t+1} (faithful)
        yn = np.where(done, xn, yn); lpy = np.where(done, lpx, lpy)
        X[:, t + 1], Y[:, t] = xn, yn
        new = (~done) & (xn == yn); tau[new] = t + 1
        if (tau > 0).all() and t + 1 >= m:
            break
    assert (tau > 0).all()
    lv = np.arange(1, T + 1)
    mask = (lv[None] >= k + 1) & (lv[None] < tau[:, None])
    w = np.minimum(1, (lv - k) / (m - k + 1))

    def H(phi):
        fx = phi(X)
        return fx[:, k:m + 1].mean(1) + (mask * w[None] * (fx[:, 1:] - phi(Y)[:, :-1])).sum(1)

    loglik = lambda th: -0.5 * (syy[:, None] - 2 * th * sy[:, None] + n_obs * th**2)  # up to const
    tests = {"theta": (lambda th: th, theta_t),
             "theta^2": (lambda th: th**2, theta_t**2),
             "loglik(y|theta)": (loglik, loglik(theta_t[:, None])[:, 0])}
    out = []
    for name, (phi, truth) in tests.items():
        D = H(phi) - truth
        out.append(f"{name}: z={D.mean() / (D.std(ddof=1) / np.sqrt(R)):+.2f} (sd(D)={D.std(ddof=1):.2f})")
    return f"E[tau]={tau.mean():.1f}; " + "; ".join(out)


for bug in [None, "prior_only", "lik_var", "flat_prior"]:
    print(f"bug={bug!s:11s} R={R}: {run(bug)}")

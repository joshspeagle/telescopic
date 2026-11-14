"""Review of x123: false alarms on likelihoods with many kinks, with noise, and from solvers.

Question: does the check stay silent on correct gradients when every line crosses many
kinks (a Laplace prior at d = 3e4 and 1e5, a hinge-loss classifier and an L1 regression
with up to 1e6 data points, a ReLU network regression, fine piecewise-linear tables),
when log_prob carries fresh or fixed noise, and when log_prob comes from an ODE solver?
And how often does it warn when the gradient is that of the exact solution while
log_prob carries the solver's error?

Design: probe.run on seeds 500000 to 500199 (200 seeds), 500000 to 500099 where a family
is slow, and 500000 to 500039 at 1e6 data points; one part per run. In "solver", the ODE
is the logistic growth y' = r y (1 - y / K) observed at 10 times, with parameters
(log r, log K); "gradient of the exact solution" differentiates the closed-form
solution, "derivative of the discrete scheme" carries the sensitivities through the same
RK4 steps and is the derivative of log_prob as computed.

Run: PYTHONPATH=src:notes/experiments/x123 python notes/experiments/x123/review/fa2.py kinks|noise|solver > fa2_<part>.out

Result: see fa2_kinks.out, fa2_noise.out, fa2_solver.out and REVIEW.md, objection 4 and
the checks that found nothing. No warning on any family with many kinks (largest ratio
0.73) or with noise. With the gradient of the exact solution, (N) warns on 100, 100, 99,
63 and 0 of 100 at RK45 tolerances 1e-3, 1e-6, 1e-8, 1e-10 and 1e-12, and on 100 of 100
with 20 and 100 fixed RK4 steps; with the derivative of the discrete scheme it warns on
none. The review's run of "solver" was cut off by its time limit after the row of 1,000
RK4 steps with the discrete derivative; the rows that never ran (1,000 steps with the
gradient of the exact solution, and the Newton solver that fa3.py kepler then ran) were
removed from this script, so that it prints what fa2_solver.out holds.
"""

import sys

import families as F
import numpy as np
from probe import run
from scipy import integrate, special

from telescopic.initial import Normal

which = sys.argv[1]
S = range(500_000, 500_200)
S100 = range(500_000, 500_100)
if which == "kinks":
    for d, seeds in ((30_000, S), (100_000, S100)):
        run(f"Laplace d={d}, start sd 2", F.laplace(d), seeds)

    # hinge loss (SVM) and L1 regression with n data points, d=5 coefficients
    def hinge_svm(n, d=5, sd=1.0, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d))
        y = np.sign(X @ np.linspace(-1, 1, d) + 0.5 * rng.standard_normal(n))
        Z = X * y[:, None]
        lp = lambda w: (
            -np.sum(np.maximum(0.0, 1 - w @ Z.T), axis=-1) - 0.5 * np.sum(w * w, axis=-1)
        )
        g = lambda w: ((1 - w @ Z.T) > 0).astype(float) @ Z - w
        return lp, g, Normal(np.zeros(d), np.full(d, sd**2))

    def lad(n, d=5, sd=1.0, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d))
        y = X @ np.linspace(-1, 1, d) + rng.laplace(size=n)
        lp = lambda w: -np.sum(np.abs(y - w @ X.T), axis=-1) - 0.5 * np.sum(w * w, axis=-1)
        g = lambda w: np.sign(y - w @ X.T) @ X - w
        return lp, g, Normal(np.zeros(d), np.full(d, sd**2))

    for n in (1_000, 10_000, 100_000, 1_000_000):
        seeds = S if n <= 100_000 else range(500_000, 500_040)
        run(f"hinge-loss classifier n={n}, d=5, start sd 1", hinge_svm(n), seeds)
        run(f"L1 (least absolute deviations) regression n={n}, d=5, start sd 1", lad(n), seeds)
    fit = hinge_svm(100_000)
    run(
        "hinge-loss classifier n=1e5, d=5, start sd 0.01 at the fit",
        (fit[0], fit[1], Normal(np.linspace(-1, 1, 5) * 1.2, np.full(5, 1e-4))),
        S,
    )

    # ReLU network regression: n data, one hidden layer, parameters = all weights
    def relu_bnn(n, hidden, d_in=5, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d_in))
        y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n)
        p = hidden * d_in + hidden + hidden + 1

        def unpack(w):
            W = w[:, : hidden * d_in].reshape(-1, hidden, d_in)
            b = w[:, hidden * d_in : hidden * d_in + hidden]
            c = w[:, hidden * d_in + hidden : -1]
            c0 = w[:, -1]
            return W, b, c, c0

        def lp(w):
            W, b, c, c0 = unpack(w)
            A = np.einsum("khj,nj->knh", W, X) + b[:, None, :]
            out = np.einsum("knh,kh->kn", np.maximum(A, 0), c) + c0[:, None]
            return -0.5 * np.sum((y - out) ** 2, axis=1) / 0.01 - 0.5 * np.sum(w * w, axis=1)

        def g(w):
            W, b, c, c0 = unpack(w)
            A = np.einsum("khj,nj->knh", W, X) + b[:, None, :]
            R = np.maximum(A, 0)
            out = np.einsum("knh,kh->kn", R, c) + c0[:, None]
            r = (y - out) / 0.01  # (k, n)
            act = (A > 0) * c[:, None, :] * r[:, :, None]  # (k, n, h)
            gW = np.einsum("knh,nj->khj", act, X).reshape(len(w), -1)
            gb = act.sum(axis=1)
            gc = np.einsum("knh,kn->kh", R, r)
            gc0 = r.sum(axis=1)
            return np.concatenate([gW, gb, gc, gc0[:, None]], axis=1) - w

        return lp, g, F.DiagonalNormal(np.zeros(p), np.full(p, 1.0))

    for n, hdn in ((100, 20), (1000, 20), (1000, 100), (10_000, 50)):
        run(
            f"ReLU network regression n={n}, {hdn} hidden units ({hdn * 7 + 1} parameters), start sd 1",
            relu_bnn(n, hdn),
            S100 if n * hdn >= 100_000 else S,
        )
    for d, sp in ((5000, 1e-3), (500, 1e-2), (5000, 1e-2)):
        knots = round(16 / sp) + 1
        run(
            f"piecewise-linear table of -x^2/2, spacing {sp:g}, d={d}",
            F.linear_table(d=d, knots=knots),
            S100,
        )
if which == "noise":
    lp0 = lambda x: -0.5 * np.sum(x * x, axis=-1)
    g0 = lambda x: -x
    init = lambda: F.DiagonalNormal(np.zeros(5), np.full(5, 2.0))
    for sig in (1e-12, 1e-9, 1e-6, 1e-3, 1.0):
        rng = np.random.default_rng(777)
        run(
            f"N(0,I) d=5 + fresh noise sd {sig:g} per evaluation (independent per row)",
            (lambda x, sig=sig, rng=rng: lp0(x) + sig * rng.standard_normal(len(x)), g0, init()),
            S,
        )
        rng2 = np.random.default_rng(778)
        run(
            f"N(0,I) d=5 + fresh noise sd {sig:g} per call (shared by the rows of a call)",
            (lambda x, sig=sig, rng=rng2: lp0(x) + sig * rng.standard_normal(), g0, init()),
            S,
        )
    # relative noise (a Monte Carlo estimate): fresh
    for rel in (1e-10, 1e-6, 1e-3):
        rng = np.random.default_rng(779)
        run(
            f"N(0,I) d=5 x (1 + {rel:g} noise), fresh per row",
            (
                lambda x, rel=rel, rng=rng: lp0(x) * (1 + rel * rng.standard_normal(len(x))),
                g0,
                init(),
            ),
            S,
        )

    # Monte Carlo marginal likelihood with a FIXED seed (common random numbers): smooth, and its own gradient
    def mc_marginal(m, fresh, seed=0):
        base = np.random.default_rng(seed).standard_normal((m, 3))
        rng = np.random.default_rng(seed + 1)
        y = np.array([0.5, -0.3, 1.2])

        def parts(x):
            z = base if not fresh else rng.standard_normal((m, 3))
            # log p(y | x) = log mean_z exp(-0.5 |y - x - 0.5 z|^2); x of shape (k, 3)
            r = y - x[:, None, :] - 0.5 * z[None]
            e = -0.5 * np.sum(r * r, axis=2)
            return r, e

        def lp(x):
            _, e = parts(x)
            return special.logsumexp(e, axis=1) - np.log(m) - 0.5 * np.sum(x * x, axis=1)

        def g(x):
            r, e = parts(x)
            w = special.softmax(e, axis=1)
            return np.einsum("km,kmj->kj", w, r) - x

        return lp, g, Normal(np.zeros(3), np.full(3, 1.0))

    run(
        "Monte Carlo marginal likelihood, 1000 draws, fixed seed (grad of the estimate)",
        mc_marginal(1000, False),
        S,
    )
    run(
        "Monte Carlo marginal likelihood, 1000 draws, fresh draws every call",
        mc_marginal(1000, True),
        S,
    )
    run(
        "Monte Carlo marginal likelihood, 100000 draws, fresh draws every call",
        mc_marginal(100_000, True),
        S100,
    )
if which == "solver":
    # logistic growth ODE y' = r y (1 - y/K), y(0)=0.1, observed at 10 times; parameters (log r, log K)
    t_obs = np.linspace(0.5, 5, 10)
    y0 = 0.1

    def exact(th):
        r, K = np.exp(th[..., 0:1]), np.exp(th[..., 1:2])
        return K / (1 + (K / y0 - 1) * np.exp(-r * t_obs))

    data = exact(np.array([0.0, 0.0])) + 0.05 * np.random.default_rng(0).standard_normal(10)

    def g_exact(th):  # gradient of the exact solution's log-likelihood (the 'exact adjoint')
        th = np.atleast_2d(th)
        r, K = np.exp(th[:, 0:1]), np.exp(th[:, 1:2])
        E = np.exp(-r * t_obs)
        A = K / y0 - 1
        den = 1 + A * E
        y = K / den
        dy_dr = K * A * t_obs * E / den**2
        dy_dK = 1 / den - K * E / (y0 * den**2)
        res = (data - y) / 0.05**2
        return (
            np.stack(
                [np.sum(res * dy_dr, axis=1) * r[:, 0], np.sum(res * dy_dK, axis=1) * K[:, 0]],
                axis=1,
            )
            - th
        )

    def lp_solver(rtol, method="RK45"):
        def lp(th):
            out = []
            for row in np.atleast_2d(th):
                r, K = np.exp(row)
                sol = integrate.solve_ivp(
                    lambda t, y, r=r, K=K: r * y * (1 - y / K),
                    (0, 5),
                    [y0],
                    t_eval=t_obs,
                    rtol=rtol,
                    atol=rtol * 1e-3,
                    method=method,
                )
                out.append(-0.5 * np.sum((data - sol.y[0]) ** 2) / 0.05**2 - 0.5 * row @ row)
            return np.array(out)

        return lp

    init = Normal(np.zeros(2), np.full(2, 0.3**2))
    for rtol in (1e-3, 1e-6, 1e-8, 1e-10, 1e-12):
        run(
            f"ODE likelihood by adaptive RK45 rtol {rtol:g}, gradient of the exact solution",
            (lp_solver(rtol), g_exact, init),
            S100,
        )
    run(
        "ODE likelihood by LSODA rtol 1e-8, gradient of the exact solution",
        (lp_solver(1e-8, "LSODA"), g_exact, init),
        S100,
    )
    run(
        "ODE likelihood, exact solution and its gradient (control)",
        (
            lambda th: (
                -0.5 * np.sum((data - exact(np.atleast_2d(th))) ** 2, axis=1) / 0.05**2
                - 0.5 * np.sum(np.atleast_2d(th) ** 2, axis=1)
            ),
            g_exact,
            init,
        ),
        S100,
    )

    # fixed-step RK4 with the derivative of the discrete scheme (forward sensitivities through the same steps): correct for log_prob as computed
    def rk4(th, n_steps):
        th = np.atleast_2d(th)
        r, K = np.exp(th[:, 0]), np.exp(th[:, 1])

        # state y and sensitivities s_r = dy/dr, s_K = dy/dK integrated together by RK4
        def f(Y):
            y, sr, sK = Y
            fy = r * y * (1 - y / K)
            dfdy = r * (1 - 2 * y / K)
            return np.array([fy, dfdy * sr + y * (1 - y / K), dfdy * sK + r * y * y / K**2])

        Y = np.array([np.full(len(th), y0), np.zeros(len(th)), np.zeros(len(th))])
        dt = 5.0 / n_steps
        outs = []
        per = n_steps // 10
        for k in range(n_steps):
            k1 = f(Y)
            k2 = f(Y + 0.5 * dt * k1)
            k3 = f(Y + 0.5 * dt * k2)
            k4 = f(Y + dt * k3)
            Y = Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if (k + 1) % per == 0:
                outs.append(Y.copy())
        O = np.array(outs)  # (10, 3, k)
        return O[:, 0].T, O[:, 1].T * r[:, None], O[:, 2].T * K[:, None]

    for n_steps in (20, 100, 1000):

        def lp(th, n_steps=n_steps):
            y, _, _ = rk4(th, n_steps)
            th = np.atleast_2d(th)
            return -0.5 * np.sum((data - y) ** 2, axis=1) / 0.05**2 - 0.5 * np.sum(th * th, axis=1)

        def g(th, n_steps=n_steps):
            y, dr, dK = rk4(th, n_steps)
            th = np.atleast_2d(th)
            res = (data - y) / 0.05**2
            return np.stack([np.sum(res * dr, axis=1), np.sum(res * dK, axis=1)], axis=1) - th

        run(
            f"ODE likelihood by fixed-step RK4, {n_steps} steps, derivative of the discrete scheme",
            (lp, g, init),
            S100,
        )
        if n_steps < 1000:
            run(
                f"ODE likelihood by fixed-step RK4, {n_steps} steps, gradient of the exact solution",
                (lp, g_exact, init),
                S100,
            )

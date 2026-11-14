"""The harm of each gradient of groups B and C and of the formula checks of experiment x123.

Question: by how much does each gradient that is not the derivative of log_prob as
computed slow the meeting of coupled MALA, against the derivative?

The measure (PREREGISTRATION.md, section 4) is the ratio of mean meeting times,
Lambda = mean tau(given gradient) / mean tau(derivative), over 5 seeds of 100 pairs each
(500 pairs a side), lag 1, with a pair unmet at max_iter = 10,000 counted at 10,000 (so
that Lambda is then a lower bound), and a standard error from the 5 per-seed ratios of
means, se = sd / sqrt(5). Both gradients of a seed run with the same seed. Classes:
harmful iff Lambda - 2 se >= 2 (>= 3 by the proxy below), harmless iff
Lambda + 2 se <= 1.5, otherwise unclassed; undetermined when the derivative leaves more
than 5% of its 500 pairs unmet. Beside Lambda: the ratio of tuned steps and delta_0, the
root mean square under pi of the drift error in proposal standard deviations,
(eps / 2) |C^T e(x)|, at the derivative's tuned step eps and covariance C C^T (the
package's runs: the mean over the 5 seeds, with 2,000 draws of pi each; draws of the
Laplace approximation for the logistic targets).

Two samplers:
- "package": CoupledSampler(log_prob, init, grad=..., vectorize=True, kernel="mala",
  rng=seed).meeting_times(100, max_iter=10_000), default tuning, at d <= 200. Baseline b
  (PACKAGE, in order) runs on seeds 370000 + 10 b + (0..4).
- "proxy", above d = 200 (N(0, I_d) only), where the package's tuning costs 10 d^2 steps
  and its MALA holds a dense covariance: a re-implementation of the package's coupled
  MALA step for cov = I (proposal x + (eps^2 / 2) g(x) + eps z, the reflection-maximal
  coupling of the two proposals, one uniform for both acceptances, lag 1), with the step
  whose mean acceptance at stationarity is 0.574 for the gradient in use (bisection on
  max(200, 4,000,000 / d) draws of pi from generator 380000), which is what the
  package's dual averaging aims at. Baseline b (PROXY, in order) on seeds
  375000 + 10 b + (0..4). It is validated in the same run against the package at d = 50
  and d = 200 on the same families (the proxy baselines marked "validation"), and its
  marginal transition against the package's MALA(eps, cov=I) step from fixed states
  ("marginals", generator 381500).

Each chain's marginal transition in the proxy is MALA's: the reflection-maximal coupling
gives each proposal the law N(x + (eps^2 / 2) g(x), eps^2 I) of the uncoupled
proposal, and each chain accepts with its own Metropolis-Hastings ratio and a uniform
that is uniform for it; chains that propose the same point from the same state accept
together, so they stay equal once they meet.

Run: python notes/experiments/x123/harm.py package|proxy [first stop] [--only TEXT]
[--seeds i j] [--smoke] over the baselines first..stop-1 of the list, the gradients whose
name contains TEXT and the seeds i..j-1 of the five, which prints a ROW line per gradient
and seed; then python notes/experiments/x123/harm.py summarize <outputs> for the classes.
python notes/experiments/x123/harm.py marginals checks the proxy's marginals. --smoke runs
seeds 220000 and 220001 with 10 pairs and max_iter = 1,000.

Result: see harm_*.out.
"""

import sys
import time
import warnings
from functools import partial

import families as F
import numpy as np
from runner import header, smoke
from scipy import optimize, stats

from telescopic import CoupledSampler
from telescopic.kernels import MALA
from telescopic.target import Target
from telescopic.testing import random_draws

SMOKE = smoke()
PAIRS, MAX_ITER = (10, 1000) if SMOKE else (100, 10_000)
TUNING_SEED, MARGINAL_SEED = (F.SMOKE[0], F.SMOKE[1]) if SMOKE else (380_000, 381_500)
DRAWS_OF_PI = 2000


def seeds(base, b):
    return F.SMOKE if SMOKE else range(base + 10 * b, base + 10 * b + 5)


# The package's default-tuned coupled MALA.
def package_run(case, seed):
    lp, g, init = F.resolve(case, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = CoupledSampler(lp, init, grad=g, vectorize=True, kernel="mala", rng=seed)
        tau = sampler.meeting_times(PAIRS, max_iter=MAX_ITER)
    return tau, sampler.kernel.step_size, np.linalg.cholesky(sampler.kernel.cov)


# The proxy: coupled MALA with cov = I on N(0, I_d).
def log_alpha(x, gx, y, gy, eps):
    log_q = 0.5 * np.sum((x - y) * (gx + gy), axis=1) + eps**2 / 8 * (
        np.sum(gx**2, axis=1) - np.sum(gy**2, axis=1)
    )
    return 0.5 * (np.sum(x * x, axis=1) - np.sum(y * y, axis=1)) + log_q


def as_float64(grad):
    return lambda y: np.asarray(grad(y), dtype=float)


def proxy_step(d, grad):
    def rate(eps):
        rng = np.random.default_rng(TUNING_SEED)
        n = max(200, 4_000_000 // d)
        x, z = rng.standard_normal((n, d)), rng.standard_normal((n, d))
        gx = grad(x)
        y = x + 0.5 * eps**2 * gx + eps * z
        return float(np.mean(np.exp(np.minimum(log_alpha(x, gx, y, grad(y), eps), 0.0)))) - 0.574

    return optimize.brentq(rate, 1e-3, 3.0, xtol=1e-3)


def single(x, grad, eps, rng):
    gx = grad(x)
    proposal = x + 0.5 * eps**2 * gx + eps * rng.standard_normal(x.shape)
    accept = np.log(rng.random(len(x))) < log_alpha(x, gx, proposal, grad(proposal), eps)
    return np.where(accept[:, None], proposal, x)


def coupled(x, y, grad, eps, rng):
    gx, gy = grad(x), grad(y)
    mx, my = x + 0.5 * eps**2 * gx, y + 0.5 * eps**2 * gy
    z = rng.standard_normal(x.shape)
    delta = (mx - my) / eps
    squared = np.sum(delta * delta, axis=1)
    same = np.log(rng.random(len(x))) < -np.sum(z * delta, axis=1) - 0.5 * squared
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.nan_to_num(delta / np.sqrt(squared)[:, None])
    px = mx + eps * z
    py = np.where(same[:, None], px, my + eps * (z - 2 * np.sum(e * z, axis=1)[:, None] * e))
    log_u = np.log(rng.random(len(x)))
    ax = log_u < log_alpha(x, gx, px, grad(px), eps)
    ay = log_u < log_alpha(y, gy, py, grad(py), eps)
    return np.where(ax[:, None], px, x), np.where(ay[:, None], py, y)


def proxy_run(d, sd, grad, eps, seed):
    rng = np.random.default_rng(seed)
    x, y = sd * rng.standard_normal((PAIRS, d)), sd * rng.standard_normal((PAIRS, d))
    x = single(x, grad, eps, rng)  # lag 1
    tau = np.full(PAIRS, np.inf)
    live = np.arange(PAIRS)
    for t in range(2, MAX_ITER + 1):
        x, y = coupled(x, y, grad, eps, rng)
        met = np.all(x == y, axis=1)
        tau[live[met]] = t
        x, y, live = x[~met], y[~met], live[~met]
        if not len(live):
            break
    return tau


# Draws of pi, for delta_0.
def normal_draws(cov=None, mean=0.0, scales=None):
    def draws(n, rng, d):
        z = rng.standard_normal((n, d))
        if scales is not None:
            z = z * scales
        if cov is not None:
            z = z @ np.linalg.cholesky(cov).T
        return mean + z

    return draws


def student_draws(n, rng, d):
    return (
        stats.multivariate_t(np.zeros(d), np.eye(d), df=3)
        .rvs(size=n, random_state=rng)
        .reshape(n, d)
    )


def plateau_draws(n, rng, d, R=10.0):
    # each coordinate: uniform on (-R, R) with probability 2R / (2R + sqrt(2 pi)), else a
    # half-normal shoulder beyond R on a random side
    flat = rng.random((n, d)) < 2 * R / (2 * R + np.sqrt(2 * np.pi))
    shoulder = (R + np.abs(rng.standard_normal((n, d)))) * rng.choice([-1.0, 1.0], size=(n, d))
    return np.where(flat, rng.uniform(-R, R, (n, d)), shoulder)


def laplace_draws(case):
    # the Laplace approximation of a logistic posterior
    lp, g, _ = case
    result = optimize.minimize(
        lambda b: -lp(b[None])[0], np.zeros(5), jac=lambda b: -g(b[None])[0], method="BFGS"
    )
    mode = result.x
    hessian = np.array(
        [(g((mode + 1e-5 * e)[None])[0] - g((mode - 1e-5 * e)[None])[0]) / 2e-5 for e in np.eye(5)]
    )
    cov = np.linalg.inv(-0.5 * (hessian + hessian.T))
    return normal_draws(cov=cov, mean=mode)


def delta0(draws_of_pi, exact, given, chol, eps, d, rng):
    x = draws_of_pi(DRAWS_OF_PI, rng, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = np.asarray(given(x), float) - np.asarray(exact(x), float)
    whitened = e @ chol  # rows are (C^T e)^T
    return eps / 2 * np.sqrt(np.mean(np.sum(whitened**2, axis=1)))


def gaussian_rotated(condition, d=10):
    right, _ = F.rotated_axis(condition, d)
    return normal_draws(cov=right[2].cov / 4)


C = F.GROUP_C
FO = F.FORMULA
B = F.GROUP_B
logistic, logistic_wide = F.logistic(), F.logistic(init_sd=10.0)

# Package baselines: (key, the derivative's case, draws of pi, the gradients run against it).
PACKAGE = [
    (
        "N(0,I) d=5, start sd 2",
        F.gaussian(),
        normal_draws(),
        [
            ("C", "N(0,I) d=5: wrong sign"),
            ("C", "N(0,I) d=5: twice"),
            ("C", "N(0,I) d=5, start sd 2: sign of coordinate 0"),
            ("C", "N(0,I) d=5, start sd 2: coordinate 0 set to 0"),
            ("C", "float32 grad, N(0,I) d=5: sign of coordinate 0"),
            ("B", "N(0,I) d=5: grad x 1.005"),
            ("B", "N(0,I) d=5: grad + 0.02"),
        ],
    ),
    (
        "N(0,I) d=5, start sd 1",
        F.gaussian(wide=1.0),
        normal_draws(),
        [("C", "N(0,I) d=5, start sd 1: sign of coordinate 0")],
    ),
    (
        "N(0,I) d=50, start sd 2",
        F.gaussian(d=50),
        normal_draws(),
        [("C", "N(0,I) d=50: twice"), ("C", "N(0,I) d=50, start sd 2: sign of coordinate 0")],
    ),
    (
        "N(0,I) d=50, start sd 1",
        F.gaussian(d=50, wide=1.0),
        normal_draws(),
        [("C", "N(0,I) d=50, start sd 1: sign of coordinate 0")],
    ),
    (
        "N(0,I) d=200, start sd 2",
        F.gaussian(d=200),
        normal_draws(),
        [("C", "N(0,I) d=200: twice"), ("C", "N(0,I) d=200, start sd 2: sign of coordinate 0")],
    ),
    (
        "N(0,I) d=200, start sd 1",
        F.gaussian(d=200, wide=1.0),
        normal_draws(),
        [("C", "N(0,I) d=200, start sd 1: sign of coordinate 0")],
    ),
    (
        "Student t3 d=5, start sd 2",
        C["Student t3 d=5, start sd 2: gradient of N(0, I)"][0],
        student_draws,
        [("C", "Student t3 d=5, start sd 2: gradient of N(0, I)")],
    ),
    (
        "equicorrelated rho 0.9 d=10",
        C["equicorrelated rho 0.9 d=10: gradient of rho 0.8"][0],
        normal_draws(cov=np.full((10, 10), 0.9) + 0.1 * np.eye(10)),
        [("C", "equicorrelated rho 0.9 d=10: gradient of rho 0.8")],
    ),
    *[
        (
            f"scales (1, {n:g} x4), start N(0, {s})",
            C[f"scales (1, {n:g} x4), start N(0, {s}): sign of the wide coordinate"][0],
            normal_draws(scales=np.array([1.0] + [n] * 4)),
            [("C", f"scales (1, {n:g} x4), start N(0, {s}): sign of the wide coordinate")],
        )
        for n, s in ((0.1, "I"), (0.01, "I"), (0.1, "0.1^2 I"), (0.01, "0.01^2 I"))
    ],
    *[
        (
            f"N(0,I) d={d}, start sd ({t:g}, 2, ...)",
            C[f"N(0,I) d={d}, start sd ({t:g}, 2, ...): sign of coordinate 0"][0],
            normal_draws(),
            [("C", f"N(0,I) d={d}, start sd ({t:g}, 2, ...): sign of coordinate 0")],
        )
        for d in (5, 100)
        for t in (0.1, 0.01)
    ],
    *[
        (
            f"N(0,I) d=5, start sd {w:g}",
            C[f"N(0,I) d=5, start sd {w:g}: grad + 2"][0],
            normal_draws(),
            [("C", f"N(0,I) d=5, start sd {w:g}: grad + 2")]
            + ([("C", "N(0,I) d=5, start sd 1000: sign of coordinate 0")] if w == 1000 else []),
        )
        for w in (100.0, 300.0, 1000.0, 1e4)
    ],
    *[
        (
            f"rotated cond {c:g} d=10, matched",
            C[f"rotated cond {c:g} d=10, matched: sign along the widest axis"][0],
            gaussian_rotated(c),
            [("C", f"rotated cond {c:g} d=10, matched: sign along the widest axis")],
        )
        for c in (1e2, 1e4)
    ],
    (
        "N(1e3, I) d=5, start sd 2",
        F.gaussian(mean=1e3),
        normal_draws(mean=1e3),
        [("C", "float32 grad, N(1e3,I) d=5: sign of coordinate 0")],
    ),
    (
        "plateau d=3, start N(0, I)",
        F.plateau(),
        plateau_draws,
        [("C", "plateau d=3, start N(0, I): grad = -x")],
    ),
    (
        "N(0,I) d=2, point start (1, 1)",
        C["N(0,I) d=2, point start (1, 1): grad + 2 (1, -1)"][0],
        normal_draws(),
        [("C", "N(0,I) d=2, point start (1, 1): grad + 2 (1, -1)")],
    ),
    (
        "N(0,I) d=5, random point start",
        C["N(0,I) d=5, random point start: sign of coordinate 0"][0],
        normal_draws(),
        [("C", "N(0,I) d=5, random point start: sign of coordinate 0")],
    ),
    (
        "N(0,I) d=5, start sd (1e-4, 2, ...)",
        FO["N(0,I) d=5, start sd (1e-4, 2, ...): sign of coordinate 0"][0],
        normal_draws(),
        [("F", "N(0,I) d=5, start sd (1e-4, 2, ...): sign of coordinate 0")],
    ),
    (
        "scales (1, 1e-4 x4), start N(0, I)",
        FO["scales (1, 1e-4 x4), start N(0, I): sign of the wide coordinate"][0],
        normal_draws(scales=np.array([1.0] + [1e-4] * 4)),
        [("F", "scales (1, 1e-4 x4), start N(0, I): sign of the wide coordinate")],
    ),
    (
        "N(0,I) - 1e4 d=5, start sd 1e-3",
        FO["N(0,I) - 1e4 d=5, start sd 1e-3: sign of coordinate 0"][0],
        normal_draws(),
        [("F", "N(0,I) - 1e4 d=5, start sd 1e-3: sign of coordinate 0")],
    ),
    *[
        (
            f"Jacobi solver rtol {text} d=5, with the gradient of log_prob as computed",
            (F.loose_solver(r)[0], partial(F.jacobi_gradient, rtol=r), F.loose_solver(r)[2]),
            normal_draws(cov=F.JACOBI_MATRIX),
            [("B", f"Jacobi solver rtol {text} d=5 (exact gradient)")],
        )
        for r, text in ((1e-3, "1e-3"), (1e-5, "1e-5"), (1e-8, "1e-8"))
    ],
    (
        "logistic n=1e4 d=5, start sd 0.1",
        logistic,
        laplace_draws(logistic),
        [("B", "logistic n=1e4 d=5: without the prior")],
    ),
    (
        "logistic n=1e4 d=5, start sd 10",
        logistic_wide,
        laplace_draws(logistic_wide),
        [("B", "logistic n=1e4 d=5, start sd 10: without the prior")],
    ),
    (
        "N(0,I) d=5, start sd 200",
        F.gaussian(wide=200.0),
        normal_draws(),
        [("B", "N(0,I) d=5, start 100x wide: grad x 1.005")],
    ),
]

# Proxy baselines: (key, d, start sd, the gradients run against it, validation).
PROXY = [
    *[
        (
            f"N(0,I) d={d}, start sd {k:g}",
            d,
            k,
            [("C", f"N(0,I) d={d}, start sd {k:g}: sign of coordinate 0")]
            + (
                [("C", "N(0,I) d=10000, start sd 2: signs of coordinates 0..9")]
                if (d, k) == (10_000, 2.0)
                else []
            )
            + (
                [
                    ("C", "float32 grad, N(0,I) d=1000, start sd 2: sign of coordinate 0"),
                    ("B", "N(0,I) d=1000: grad x 1.0001"),
                ]
                if (d, k) == (1000, 2.0)
                else []
            ),
            False,
        )
        for d in (1000, 2000, 5000, 10_000)
        for k in (1.0, 2.0)
    ],
    *[
        (
            f"N(0,I) d={d}, start sd ({t:g}, 2, ...)",
            d,
            np.r_[t, np.full(d - 1, 2.0)],
            [(g, f"N(0,I) d={d}, start sd ({t:g}, 2, ...): sign of coordinate 0")],
            False,
        )
        for g, d, t in (
            ("C", 1000, 0.1),
            ("C", 1000, 0.01),
            ("F", 10_000, 0.01),
            ("F", 1000, 0.003),
            ("F", 1000, 0.5),
            ("F", 1000, 0.25),
        )
    ],
    *[
        (
            f"N(0,I) d={d}, start sd {k:g}",
            d,
            k,
            [("C", f"N(0,I) d={d}, start sd {k:g}: sign of coordinate 0")]
            + ([("C", f"N(0,I) d={d}: twice")] if k == 2.0 else []),
            True,
        )
        for d in (50, 200)
        for k in (1.0, 2.0)
    ],
]
SAME = {
    "rotated cond 1e4 d=2000, matched: sign along the widest axis": "N(0,I) d=2000, start sd 2: sign of coordinate 0"
}


def wrong_case(group, name):
    return {"B": lambda: B[name], "C": lambda: C[name][1], "F": lambda: FO[name][1]}[group]()


def options():
    # [first stop] [--only TEXT] [--seeds i j]: baselines first..stop-1, the gradients whose
    # name contains TEXT, the seeds i..j-1 of each baseline's five; for splitting long runs
    args = [a for a in sys.argv[2:] if a != "--smoke"]
    only, chosen = "", slice(None)
    if "--only" in args:
        k = args.index("--only")
        only = args[k + 1]
        del args[k : k + 2]
    if "--seeds" in args:
        k = args.index("--seeds")
        chosen = slice(int(args[k + 1]), int(args[k + 2]))
        del args[k : k + 3]
    first, stop = (int(args[0]), int(args[1])) if len(args) > 1 else (0, 10**6)
    return first, stop, only, chosen


def row(group, name, sampler, seed, given, derivative, step_given, step_derivative, delta):
    # given and derivative: (mean meeting time with unmet pairs at max_iter, unmet pairs)
    print(
        f"ROW | {group} | {name} | {sampler} | {seed} | {given[0]:.6f} | {derivative[0]:.6f} | {given[1]} | {derivative[1]} | "
        f"{step_given:.6f} | {step_derivative:.6f} | {delta:.6g} | ratio {given[0] / derivative[0]:.3f}",
        flush=True,
    )


def run_package():
    first, stop, only, chosen = options()
    for b, (key, right, draws_of_pi, wrongs) in enumerate(PACKAGE):
        wrongs = [w for w in wrongs if only in w[1]]
        if not first <= b < stop or not wrongs:
            continue
        chosen_seeds = list(seeds(370_000, b))[chosen]
        print(
            f"package baseline {b}: {key}; seeds {chosen_seeds}; {PAIRS} pairs, max_iter {MAX_ITER}",
            flush=True,
        )
        base = {}
        for seed in chosen_seeds:
            start = time.perf_counter()
            tau, eps, chol = package_run(right, seed)
            base[seed] = (
                (np.mean(np.minimum(tau, MAX_ITER)), int(np.isinf(tau).sum())),
                eps,
                chol,
            )
            print(
                f"  seed {seed}: derivative mean tau {base[seed][0][0]:.2f}, unmet {base[seed][0][1]}, step {eps:.4f} "
                f"({time.perf_counter() - start:.0f}s)",
                flush=True,
            )
        for group, name in wrongs:
            wrong = wrong_case(group, name)
            for seed in chosen_seeds:
                start = time.perf_counter()
                tau, eps, _ = package_run(wrong, seed)
                derivative, eps0, chol = base[seed]
                exact, given = F.resolve(right, seed)[1], F.resolve(wrong, seed)[1]
                delta = delta0(
                    draws_of_pi, exact, given, chol, eps0, len(chol), np.random.default_rng(seed)
                )
                row(
                    group,
                    name,
                    "package",
                    seed,
                    (np.mean(np.minimum(tau, MAX_ITER)), int(np.isinf(tau).sum())),
                    derivative,
                    eps,
                    eps0,
                    delta,
                )
                print(f"    ({time.perf_counter() - start:.0f}s)", flush=True)


def run_proxy():
    first, stop, only, chosen = options()
    for b, (key, d, sd, wrongs, validation) in enumerate(PROXY):
        wrongs = [w for w in wrongs if only in w[1]]
        if not first <= b < stop or not wrongs:
            continue
        chosen_seeds = list(seeds(375_000, b))[chosen]
        sd = np.broadcast_to(np.asarray(sd, float), (d,))
        exact = lambda y: -y
        eps0 = proxy_step(d, exact)
        sampler = "proxy validation" if validation else "proxy"
        print(
            f"proxy baseline {b}: {key}{' (validation)' if validation else ''}; seeds {chosen_seeds}; "
            f"{PAIRS} pairs, max_iter {MAX_ITER}; tuned step {eps0:.4f}",
            flush=True,
        )
        base = {}
        for seed in chosen_seeds:
            start = time.perf_counter()
            tau = proxy_run(d, sd, exact, eps0, seed)
            base[seed] = (np.mean(np.minimum(tau, MAX_ITER)), int(np.isinf(tau).sum()))
            print(
                f"  seed {seed}: derivative mean tau {base[seed][0]:.2f}, unmet {base[seed][1]} ({time.perf_counter() - start:.0f}s)",
                flush=True,
            )
        for group, name in wrongs:
            given = as_float64(wrong_case(group, name)[1])
            eps = proxy_step(d, given)
            delta = proxy_delta0(given, exact, eps0, d)
            for seed in chosen_seeds:
                start = time.perf_counter()
                tau = proxy_run(d, sd, given, eps, seed)
                row(
                    group,
                    name,
                    sampler,
                    seed,
                    (np.mean(np.minimum(tau, MAX_ITER)), int(np.isinf(tau).sum())),
                    base[seed],
                    eps,
                    eps0,
                    delta,
                )
                print(f"    ({time.perf_counter() - start:.0f}s)", flush=True)


def summarize(paths):
    # The classes, from the ROW lines of harm outputs (several, when a run was split).
    rows = {}
    for path in paths:
        with open(path) as lines:
            for line in lines:
                if line.startswith("ROW |"):
                    f = [x.strip() for x in line.split("|")]
                    rows.setdefault((f[1], f[2], f[3]), {})[int(f[4])] = [
                        float(x) for x in f[5:12]
                    ]
    print(
        "RESULT | group | family | sampler | Lambda | se | class | unmet given / pairs | unmet derivative | step ratio | delta0 | seeds"
    )
    for (group, name, sampler), by_seed in rows.items():
        r = np.array([by_seed[s] for s in sorted(by_seed)])
        ratios = r[:, 0] / r[:, 1]
        Lambda = r[:, 0].mean() / r[:, 1].mean()
        se = ratios.std(ddof=1) / np.sqrt(len(ratios)) if len(ratios) > 1 else np.nan
        bar = 3.0 if sampler.startswith("proxy") else 2.0
        pairs = PAIRS * len(r)
        if r[:, 3].sum() > 0.05 * pairs:
            label = "undetermined"
        elif Lambda - 2 * se >= bar:
            label = "harmful"
        elif Lambda + 2 * se <= 1.5:
            label = "harmless"
        else:
            label = "unclassed"
        if len(r) != (2 if SMOKE else 5):
            label += " (incomplete)"
        print(
            f"RESULT | {group} | {name} | {sampler} | {Lambda:.3f} | {se:.3f} | {label} | {int(r[:, 2].sum())}/{pairs} | "
            f"{int(r[:, 3].sum())} | {np.mean(r[:, 4] / r[:, 5]):.3f} | {np.mean(r[:, 6]):.3g} | {sorted(by_seed)}"
        )
        for same, of in SAME.items():
            if of == name:
                print(
                    f"RESULT | C | {same} | the harm of '{of}', by affine equivalence (PREREGISTRATION.md, section 4)"
                )


def proxy_delta0(given, exact, eps, d):
    rng = np.random.default_rng(TUNING_SEED + 1)
    total, n = 0.0, 0
    for _ in range(DRAWS_OF_PI // 200):
        y = rng.standard_normal((200, d))
        e = given(y) - exact(y)
        total += np.sum(e * e)
        n += 200
    return eps / 2 * np.sqrt(total / n)


def run_marginals():
    # One coupled step of the proxy from fixed (x, y), against one step of the package's
    # MALA(eps, cov=I) from x and from y: the positions along 3 random directions (KS
    # test) and the rate of staying put (two-proportion z test); and pairs that start equal
    # stay equal.
    n = 20_000
    rng = np.random.default_rng(MARGINAL_SEED)
    for d, grad_name in ((5, "derivative"), (5, "sign of coordinate 0"), (50, "derivative")):
        grad = (
            (lambda y: -y)
            if grad_name == "derivative"
            else as_float64(F.modify(F.gaussian_start(d, 2.0), F.flip(0))[1])
        )
        eps = proxy_step(d, grad)
        x0, y0 = rng.standard_normal(d), rng.standard_normal(d) * 0.3
        x, y = coupled(np.tile(x0, (n, 1)), np.tile(y0, (n, 1)), grad, eps, rng)
        kernel = MALA(eps, cov=np.eye(d))
        target = Target(lambda z: -0.5 * np.sum(z * z, axis=-1), grad, vectorize=True)
        alone = [
            kernel.step(
                target, kernel.init_state(target, np.tile(p, (n, 1))), random_draws(n, rng)
            )[0].x
            for p in (x0, y0)
        ]
        directions = rng.standard_normal((3, d))
        p_values = []
        for coupled_draws, single_draws, start in ((x, alone[0], x0), (y, alone[1], y0)):
            for v in directions:
                p_values.append(stats.ks_2samp(coupled_draws @ v, single_draws @ v).pvalue)
            stay_c, stay_s = (
                np.all(coupled_draws == start, axis=1).mean(),
                np.all(single_draws == start, axis=1).mean(),
            )
            pooled = (stay_c + stay_s) / 2
            z = (stay_c - stay_s) / np.sqrt(max(2 * pooled * (1 - pooled) / n, 1e-300))
            p_values.append(2 * stats.norm.sf(abs(z)))
        same = np.tile(x0, (1000, 1))
        a, b = coupled(same, same.copy(), grad, eps, rng)
        print(
            f"marginals d={d}, {grad_name}, step {eps:.4f}: smallest of {len(p_values)} p-values {min(p_values):.3g} "
            f"(Bonferroni bound 1e-3 / {len(p_values)}); met in one step {np.mean(np.all(x == y, axis=1)):.3f}; "
            f"equal pairs stay equal {bool(np.all(a == b))}",
            flush=True,
        )


if __name__ == "__main__":
    header()
    which = sys.argv[1]
    if which == "summarize":
        summarize([a for a in sys.argv[2:] if a != "--smoke"])
    else:
        {"package": run_package, "proxy": run_proxy, "marginals": run_marginals}[which]()

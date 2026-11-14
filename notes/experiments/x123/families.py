"""The families of the second pre-registered experiment of #123, and their seeds.

A case is ``(log_prob, grad, initial)``, or a callable of the seed that returns one (a
point start whose point varies with the seed). ``log_prob`` takes ``(n, d)`` and returns
``(n,)`` unless the family is marked not vectorized. Group A holds correct gradients (a
warning is a false alarm), group B gradients that are not the derivative of
``log_prob`` as computed and do no measured harm, group C wrong gradients, and FORMULA
the wrong gradients whose predicted rate is checked on 1,000 seeds. The constructors of
the first experiment and of its review are rewritten here so that this folder needs no
other file.

Seeds (PREREGISTRATION.md, section 6): family ``f`` of a group at
``base + n f + (0..n-1)``, with ``(base, n)`` in ``SEEDS``.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy import special, stats

from telescopic import examples
from telescopic.initial import Empirical, Normal

SEEDS = {"A": (300_000, 300), "B": (335_000, 300), "C": (340_000, 400), "F": (360_000, 1000)}
SMOKE = (220_000, 220_001)  # seen seeds, for the smoke runs of phase 1


def seeds(group, index, smoke=False):
    if smoke:
        return SMOKE
    base, n = SEEDS[group]
    return range(base + n * index, base + n * (index + 1))


def resolve(case, seed):
    return case(seed) if callable(case) else case


class DiagonalNormal:
    # N(mean, diag(sd^2)) without the d x d Cholesky factor of initial.Normal.
    def __init__(self, mean, sd):
        self.mean, self.sd = np.asarray(mean, float), np.asarray(sd, float)

    def rvs(self, *, random_state):
        return self.mean + self.sd * random_state.standard_normal(len(self.sd))


# Targets of the first experiment.
def gaussian(scale=1.0, d=5, wide=2.0, mean=0.0, offset=0.0):
    lp = lambda x: -0.5 * np.sum(((x - mean) / scale) ** 2, axis=-1) + offset
    g = lambda x: -(x - mean) / scale**2
    return lp, g, Normal(np.full(d, mean), np.full(d, (wide * scale) ** 2))


def gaussian_start(d, sd, offset=0.0):
    # N(0, I_d) from N(0, diag(sd^2)), sd a number or a vector, any d.
    lp = lambda x: -0.5 * np.sum(x * x, axis=-1) + offset
    return lp, (lambda x: -x), DiagonalNormal(np.zeros(d), np.broadcast_to(sd, (d,)))


def conditioned(condition, d=10, wide=None, state=0):
    rotation = stats.ortho_group.rvs(d, random_state=state)
    scales = np.logspace(0, -0.5 * np.log10(condition), d)
    precision = rotation @ np.diag(scales**-2) @ rotation.T
    cov = rotation @ np.diag(scales**2) @ rotation.T
    init = (
        Normal(np.zeros(d), 4 * cov) if wide is None else Normal(np.zeros(d), np.full(d, wide**2))
    )
    return (
        lambda x: -0.5 * np.einsum("...i,ij,...j->...", x, precision, x),
        lambda x: -x @ precision,
        init,
    )


def example(name):
    t = getattr(examples, name)()
    return t.log_prob, t.grad, t.initial


def student(d=5, df=3.0, wide=10.0):
    lp = lambda x: -0.5 * (df + d) * np.log1p(np.sum(x * x, axis=-1) / df)
    g = lambda x: -(df + d) * x / (df + np.sum(x * x, axis=-1, keepdims=True))
    init = (
        Normal(np.zeros(d), np.full(d, wide**2))
        if d <= 1000
        else DiagonalNormal(np.zeros(d), np.full(d, wide))
    )
    return lp, g, init


def laplace(d=5, wide=2.0):
    init = (
        Normal(np.zeros(d), np.full(d, wide**2))
        if d <= 1000
        else DiagonalNormal(np.zeros(d), np.full(d, wide))
    )
    return (lambda x: -np.sum(np.abs(x), axis=-1), lambda x: -np.sign(x), init)


def half_normal(d=3):
    def lp(x):
        inside = np.all(x > 0, axis=-1)
        return np.where(inside, -0.5 * np.sum(x * x, axis=-1), -np.inf)

    return lp, lambda x: -x, Normal(np.full(d, 1.0), np.full(d, 1.0))


def logistic(n=10_000, d=5, init_sd=0.1, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, d))
    labels = rng.random(n) < special.expit(F @ np.linspace(-1, 1, d))
    s = np.where(labels, 1.0, -1.0)
    F, s = F.astype(dtype), s.astype(dtype)

    def lp(b):
        b = b.astype(dtype)
        m = s * (F @ b.T).T
        return -np.sum(np.logaddexp(0, -m), axis=-1) - 0.5 * np.sum(b**2, axis=-1)

    def g(b):
        b = b.astype(dtype)
        m = s * (F @ b.T).T
        return (s * special.expit(-m)) @ F - b

    return lp, g, Normal(np.zeros(d), np.full(d, init_sd**2))


def as_float32(case):
    lp, g, init = case
    return (lambda x: lp(x).astype(np.float32), lambda x: g(x).astype(np.float32), init)


def with_initial(case, initial):
    return case[0], case[1], initial


def random_point_start(case, d, sd=2.0):
    # The point is drawn from the seed, as an optimizer's output would vary by problem.
    def at(seed):
        point = sd * np.random.default_rng([seed, 7]).standard_normal(d)
        return with_initial(case, Empirical(point[None]))

    return at


def diag_scales(scales, wide=2.0, init_scales=None):
    scales = np.asarray(scales, float)
    lp = lambda x: -0.5 * np.sum((x / scales) ** 2, axis=-1)
    init = scales * wide if init_scales is None else np.asarray(init_scales, float)
    return lp, lambda x: -x / scales**2, Normal(np.zeros(len(scales)), init**2)


def diag_student(scales, df=3.0, init_scales=None):
    scales = np.asarray(scales, float)
    lp = lambda x: -0.5 * (df + 1) * np.sum(np.log1p((x / scales) ** 2 / df), axis=-1)

    def g(x):
        z = x / scales
        return -(df + 1) * z / (df + z * z) / scales

    init = 2 * scales if init_scales is None else np.asarray(init_scales, float)
    return lp, g, Normal(np.zeros(len(scales)), init**2)


def cancelling(c, d=1):
    lp = lambda x: -0.5 * np.sum((x + c) ** 2 - 2 * c * x - c**2, axis=-1)
    return lp, lambda x: -x, Normal(np.zeros(d), np.full(d, 4.0))


def fringes(n_fringes, d=1, amp=5.0):
    T = 2 * np.pi * n_fringes
    lp = lambda x: amp * np.sum(np.cos(T * x), axis=-1) - 0.5 * np.sum(x * x, axis=-1)
    g = lambda x: -amp * T * np.sin(T * x) - x
    return lp, g, Normal(np.zeros(d), np.ones(d))


JACOBI_MATRIX = np.eye(5) + 0.3 * (2 * np.eye(5) - np.eye(5, k=1) - np.eye(5, k=-1))


def jacobi_solve(x, rtol, A=JACOBI_MATRIX):
    # y solving A y = x by Jacobi iteration from x / diag(A), stopped at a relative change
    # below rtol; returns y and the number of iterations.
    diag = np.diag(A)
    y = x / diag
    k = 0
    while k < 500:
        k += 1
        new = (x - A @ y + diag * y) / diag
        done = np.max(np.abs(new - y)) <= rtol * np.max(np.abs(new))
        y = new
        if done:
            break
    return y, k


def jacobi_gradient(x, rtol, A=JACOBI_MATRIX):
    # The gradient of log_prob as computed by loose_solver at the rows of x: within a region
    # of constant iteration count k, log_prob is -x.J_k x / 2 with J_k = sum_{j<=k} M^j D^-1,
    # M = I - D^-1 A, so its gradient is -(J_k + J_k^T) x / 2.
    Dinv = np.diag(1 / np.diag(A))
    M = np.eye(len(A)) - Dinv @ A
    out = []
    for row in np.atleast_2d(x):
        k = jacobi_solve(row, rtol, A)[1]
        J = Dinv
        for _ in range(k):
            J = Dinv + M @ J
        out.append(-0.5 * (J + J.T) @ row)
    return np.array(out)


def loose_solver(rtol, d=5):
    # log N(0, A) = -x.y/2 with y from jacobi_solve: the log-density carries the solver's
    # error, the gradient -A^-1 x is exact.
    A = (
        JACOBI_MATRIX
        if d == 5
        else np.eye(d) + 0.3 * (2 * np.eye(d) - np.eye(d, k=1) - np.eye(d, k=-1))
    )
    Ainv = np.linalg.inv(A)

    def lp(x):
        return np.array([-0.5 * row @ jacobi_solve(row, rtol, A)[0] for row in np.atleast_2d(x)])

    return lp, lambda x: -x @ Ainv, Normal(np.zeros(d), np.full(d, 4.0))


# Targets of the review's probes.
def flat(d=3):
    return (
        lambda x: np.zeros(len(x)),
        lambda x: np.zeros(x.shape),
        Normal(np.zeros(d), np.ones(d)),
    )


def flat_int(d=3):
    return (
        lambda x: np.zeros(len(x), dtype=int),
        lambda x: np.zeros(x.shape),
        Normal(np.zeros(d), np.ones(d)),
    )


def constant(value=-3.7, d=3):
    return (
        lambda x: np.full(len(x), value),
        lambda x: np.zeros(x.shape),
        Normal(np.zeros(d), np.ones(d)),
    )


def plateau(d=3, R=10.0):
    # A flat top |x_i| < R with Gaussian shoulders; the starts are inside the flat top.
    lp = lambda x: -0.5 * np.sum(np.maximum(np.abs(x) - R, 0.0) ** 2, axis=-1)
    g = lambda x: -np.sign(x) * np.maximum(np.abs(x) - R, 0.0)
    return lp, g, Normal(np.zeros(d), np.ones(d))


def hinge(d=3):
    lp = lambda x: -np.sum(np.maximum(x, 0.0), axis=-1) - 0.5 * np.sum(x * x, axis=-1)
    g = lambda x: -(x > 0).astype(float) - x
    return lp, g


def asym_laplace(d=3):
    lp = lambda x: -np.sum(np.where(x > 0, 2 * x, -x), axis=-1)
    g = lambda x: -np.where(x > 0, 2.0, np.where(x < 0, -1.0, 0.5))
    return lp, g


def naive_sum(n=100_000, seed=0):
    y = np.random.default_rng(seed).standard_normal(n) * 3 + 1e3

    def lp(x):  # x = (mu, log sigma); sequential accumulation of n terms
        mu, ls = x[:, 0:1], x[:, 1:2]
        terms = -0.5 * ((y - mu) * np.exp(-ls)) ** 2 - ls
        return np.cumsum(terms, axis=1)[:, -1]

    def g(x):
        mu, ls = x[:, 0:1], x[:, 1:2]
        r = (y - mu) * np.exp(-ls)
        return np.stack([np.sum(r * np.exp(-ls), axis=1), np.sum(r * r - 1, axis=1)], axis=1)

    return lp, g, Normal(np.array([1e3, 1.0]), np.array([0.01, 0.01]) ** 2)


def poisson(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    Xd = rng.standard_normal((n, 4))
    y = rng.poisson(np.exp(5 + Xd @ [0.3, -0.2, 0.1, 0.0]))

    def lp(b):
        eta = 5 + b @ Xd.T
        return np.cumsum(y * eta - np.exp(eta), axis=1)[:, -1]

    def g(b):
        eta = 5 + b @ Xd.T
        return (y - np.exp(eta)) @ Xd

    return lp, g, Normal(np.zeros(4), np.full(4, 0.5**2))


def grad32(case):
    lp, g, init = case
    return lp, (lambda x: np.asarray(g(x.astype(np.float32)), dtype=np.float32)), init


def gaussian32grad(mean, d=5):
    # grad computed wholly in float32 from a float32 copy of x, as autodiff without 64-bit mode
    m = np.float32(mean)
    lp = lambda x: -0.5 * np.sum((x - mean) ** 2, axis=-1)
    g = lambda x: -(x.astype(np.float32) - m)
    return lp, g, Normal(np.full(d, mean), np.full(d, 4.0))


def rotated32grad(condition, d=10):
    lp, _, init = conditioned(condition, d)
    rotation = stats.ortho_group.rvs(d, random_state=0)
    scales = np.logspace(0, -0.5 * np.log10(condition), d)
    P32 = (rotation @ np.diag(scales**-2) @ rotation.T).astype(np.float32)
    return lp, (lambda x: -(x.astype(np.float32) @ P32)), init


def all32(mean, d=5):
    m = np.float32(mean)
    lp = lambda x: -np.float32(0.5) * np.sum((x.astype(np.float32) - m) ** 2, axis=-1)
    g = lambda x: -(x.astype(np.float32) - m)
    return lp, g, Normal(np.full(d, float(mean)), np.full(d, 4.0))


def point(case, at):
    return case[0], case[1], Empirical(np.atleast_2d(np.asarray(at, float)))


def per_point(case):
    lp, g, init = case
    return (lambda x: float(lp(x[None])[0])), (lambda x: g(x[None])[0]), init


def relu_network(d=10, hidden=50, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((hidden, d)) / np.sqrt(d)
    b, c = rng.standard_normal(hidden), 0.5 * rng.standard_normal(hidden)
    lp = lambda x: -0.5 * np.sum(x * x, axis=-1) + np.maximum(x @ W.T + b, 0.0) @ c
    g = lambda x: -x + ((x @ W.T + b > 0) * c) @ W
    return lp, g, Normal(np.zeros(d), np.full(d, 4.0))


def linear_table(d=5, knots=1601, edge=8.0):
    # Each coordinate's log-density is the piecewise-linear interpolant of -t^2/2 on a grid
    # of spacing 0.01; grad is the slope of the segment that contains the point.
    grid = np.linspace(-edge, edge, knots)
    table = -0.5 * grid**2
    slopes = np.diff(table) / np.diff(grid)

    def g(x):
        index = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, knots - 2)
        return np.where((x < -edge) | (x > edge), 0.0, slopes[index])

    return (
        (lambda x: np.sum(np.interp(x, grid, table), axis=-1)),
        g,
        Normal(np.zeros(d), np.full(d, 4.0)),
    )


# Wrong gradients.
def modify(case, f):
    if callable(case):
        return lambda seed: modify(case(seed), f)
    lp, g, init = case
    return lp, (lambda x: f(x, np.array(g(x), dtype=float))), init


def flip(*coordinates):
    def f(x, g):
        g[..., list(coordinates)] = -g[..., list(coordinates)]
        return g

    return f


def zero(i):
    def f(x, g):
        g[..., i] = 0.0
        return g

    return f


def times(factor):
    factor = np.asarray(factor, float)
    return lambda x, g: g * factor


def plus(offset):
    offset = np.asarray(offset, float)
    return lambda x, g: g + offset


def g32(case):
    # a float64 gradient returned as float32
    lp, g, init = case
    return lp, (lambda x: np.asarray(g(x), dtype=np.float32)), init


@cache
def rotated_axis(condition, d=10, state=0):
    # A Gaussian rotated at random, matched start (4 cov); the wrong gradient has the sign of
    # its component along the widest principal axis reversed.
    rotation = stats.ortho_group.rvs(d, random_state=state)
    scales = np.logspace(0, -0.5 * np.log10(condition), d)
    precision = (rotation * scales**-2) @ rotation.T
    cov = (rotation * scales**2) @ rotation.T
    lp = lambda x: -0.5 * np.einsum("...i,ij,...j->...", x, precision, x)
    g = lambda x: -x @ precision
    axis = rotation[:, 0]
    wrong = lambda x: g(x) - 2 * np.multiply.outer(g(x) @ axis, axis)
    init = Normal(np.zeros(d), 4 * cov)
    return (lp, g, init), (lp, wrong, init)


def equicorrelated(rho, d=10):
    cov = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    P = np.linalg.inv(cov)
    return (
        (lambda x: -0.5 * np.einsum("...i,ij,...j->...", x, P, x)),
        (lambda x: -x @ P),
        Normal(np.zeros(d), cov),
    )


def scaled_start(sd, d):
    # N(0, I_d) from a start of spread sd[0] in coordinate 0 and sd[1] in the others
    return gaussian_start(d, np.r_[sd[0], np.full(d - 1, sd[1])])


_N5 = gaussian()

GROUP_A = {
    # the first experiment's 44
    **{f"N(0, {s:g}^2 I) d=5, matched": (gaussian(s), True) for s in (1e-8, 1e-4, 1.0, 1e4, 1e8)},
    **{
        f"N(0, {s:g}^2 I) d=5, wide 100x": (gaussian(s, wide=200.0), True)
        for s in (1e-8, 1e-4, 1.0, 1e4, 1e8)
    },
    **{f"condition {c:g} d=10, matched": (conditioned(c), True) for c in (1e4, 1e8, 1e12)},
    **{
        f"condition {c:g} d=10, wide sd 100": (conditioned(c, wide=100.0), True)
        for c in (1e4, 1e8, 1e12)
    },
    "N(1e6, I) d=5": (gaussian(mean=1e6), True),
    "N(0,I) + 1e9 d=5": (gaussian(offset=1e9), True),
    "N(0,I) - 1e12 d=5": (gaussian(offset=-1e12), True),
    "banana": (example("banana"), True),
    "funnel": (example("funnel"), True),
    "mixture": (example("mixture"), True),
    "pumps": (example("pumps"), True),
    "Student t3 d=1": (student(d=1), True),
    "Student t3 d=5": (student(), True),
    "Laplace d=5": (laplace(), True),
    "Laplace d=100": (laplace(100), True),
    "half-normal boundary d=3": (half_normal(), True),
    "logistic n=1e4 d=5": (logistic(), True),
    "logistic n=2000 d=50": (logistic(n=2000, d=50), True),
    "float32 N(0,I) d=5": (as_float32(gaussian()), True),
    "float32 N(0,I) - 1e4 d=5": (as_float32(gaussian(offset=-1e4)), True),
    "float32 Student t3 d=5": (as_float32(student(wide=2.0)), True),
    "logistic computed in float32 d=5": (logistic(dtype=np.float32), True),
    "point start N(0,I) d=5, random point": (random_point_start(gaussian(), 5), True),
    "point start logistic d=5, at 0": (
        with_initial(logistic(), Empirical(np.zeros((1, 5)))),
        True,
    ),
    "point start Student t3 d=5, random point": (random_point_start(student(), 5), True),
    "scales 1e-3..1e3 d=7, matched": (diag_scales(np.logspace(-3, 3, 7)), True),
    "t3 scales 1e-3..1e3 d=7, isotropic init 2e3": (
        diag_student(np.logspace(-3, 3, 7), init_scales=np.full(7, 2e3)),
        True,
    ),
    "N(0,I) d=5, init sd 1e-12 at 1": (
        with_initial(gaussian(), Normal(np.ones(5), np.full(5, 1e-24))),
        True,
    ),
    "N(0,I) d=1": (gaussian(d=1), True),
    "N(0,I) d=50": (gaussian(d=50), True),
    "N(0,I) d=1000": (gaussian(d=1000), True),
    "Student t3 d=1000": (student(d=1000, wide=2.0), True),
    # the review's 42 probes
    "flat log_prob 0.0, grad 0, d=3": (flat(), True),
    "flat log_prob int 0, grad 0, d=3": (flat_int(), True),
    "plateau (flat top |x|<10, Gaussian shoulders) d=3, init N(0,I)": (plateau(), True),
    "hinge -relu(x) - x^2/2 d=3, point start at 0": (
        (*hinge(), Empirical(np.zeros((1, 3)))),
        True,
    ),
    "hinge d=3, init N(0,I)": ((*hinge(), Normal(np.zeros(3), np.ones(3))), True),
    "asymmetric Laplace d=3, point start at 0": (
        (*asym_laplace(), Empirical(np.zeros((1, 3)))),
        True,
    ),
    "Laplace d=3, point start at 0": (point(laplace(3), np.zeros(3)), True),
    "naive sum of 1e5 Gaussian terms, d=2": (naive_sum(), True),
    "Poisson regression n=2e4, naive sum, d=4": (poisson(), True),
    "Student t3 d=5, init sd 1e6": (student(wide=1e6), True),
    "Student t3 d=1 (Cauchy-like), init sd 1e8": (student(d=1, df=1.0, wide=1e8), True),
    "Student t3 d=1000, init sd 1e3": (student(d=1000, wide=1e3), True),
    "rotated cond 1e12 d=50, matched": (conditioned(1e12, 50, state=1), True),
    "rotated cond 1e12 d=50, isotropic sd 1": (conditioned(1e12, 50, wide=1.0, state=1), True),
    "rotated cond 1e12 d=100, matched": (conditioned(1e12, 100, state=1), True),
    "rotated cond 1e8 d=2, matched": (conditioned(1e8, 2, state=1), True),
    "rotated cond 1e12 d=3, matched": (conditioned(1e12, 3, state=1), True),
    "N(0,I) d=2": (gaussian(d=2), True),
    "N(0,I) d=3": (gaussian(d=3), True),
    "Student t3 d=2": (student(d=2), True),
    "N(0,I) d=5, init var 0 in coords 0,1 (at 0)": (
        with_initial(
            gaussian(),
            Empirical(
                np.random.default_rng(5).standard_normal((50, 5)) * np.array([0, 0, 1.0, 1, 1])
            ),
        ),
        True,
    ),
    "N(0,I) d=5, init sd 1e-150": (
        with_initial(gaussian(), Normal(np.zeros(5), np.full(5, 1e-300))),
        True,
    ),
    "N(0,I) d=5, Empirical of 2 points": (
        with_initial(gaussian(), Empirical(np.array([[1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 5.5]]))),
        True,
    ),
    "N(0,I) d=5, point start at 0": (point(gaussian(), np.zeros(5)), True),
    "Student t3 d=5, point start at 0": (point(student(), np.zeros(5)), True),
    "funnel, point start at 0": (point(example("funnel"), np.zeros(10)), True),
    "banana, point start at 0": (point(example("banana"), np.zeros(2)), True),
    "logistic n=2000 d=50, point start at 0": (point(logistic(n=2000, d=50), np.zeros(50)), True),
    "float32 N(0,I) + 1e7 d=5": (as_float32(gaussian(offset=1e7)), True),
    "float32 N(0,I) - 1e9 d=5": (as_float32(gaussian(offset=-1e9)), True),
    "float32 N(1e3, I) d=5 (all in float64, cast out)": (as_float32(gaussian(mean=1e3)), True),
    "float64 log_prob, float32 grad: N(0,I) d=5": (gaussian32grad(0.0), True),
    "float64 log_prob, float32 grad: N(1e3,I) d=5": (gaussian32grad(1e3), True),
    "float64 log_prob, float32 grad: N(1e5,I) d=5": (gaussian32grad(1e5), True),
    "float64 log_prob, float32 grad: N(1e6,I) d=5": (gaussian32grad(1e6), True),
    "float64 log_prob, float32 grad: logistic n=1e4 d=5": (grad32(logistic()), True),
    "float64 log_prob, float32 grad: rotated cond 1e4 d=10": (rotated32grad(1e4), True),
    "float64 log_prob, float32 grad: rotated cond 1e8 d=10": (rotated32grad(1e8), True),
    "float64 log_prob, float32 grad: rotated cond 1e12 d=10": (rotated32grad(1e12), True),
    "not vectorized: N(0,I) d=5": (per_point(gaussian()), False),
    "not vectorized: logistic n=1e4 d=5": (per_point(logistic()), False),
    "not vectorized: banana": (per_point(example("banana")), False),
    # the design's six targets computed wholly in float32
    **{
        f"float32 throughout, N({m:g}, I) d=5": (all32(m), True)
        for m in (0.0, 10.0, 100.0, 300.0, 1e3, 1e4)
    },
    # new
    **{f"N(0,I) d={d}, start sd 2": (gaussian_start(d, 2.0), True) for d in (2000, 5000, 10_000)},
    **{
        f"Student t3 d={d}, start sd 2": (student(d=d, wide=2.0), True)
        for d in (2000, 5000, 10_000)
    },
    **{f"Laplace d={d}, start sd 2": (laplace(d), True) for d in (2000, 5000, 10_000)},
    "rotated cond 1e4 d=2000, matched": (rotated_axis(1e4, 2000)[0], True),
    "scales (1, 0.01 x4), start N(0, I)": (
        diag_scales([1, 0.01, 0.01, 0.01, 0.01], init_scales=np.ones(5)),
        True,
    ),
    "N(0,I) d=5, start sd (0.01, 2, 2, 2, 2)": (scaled_start((0.01, 2.0), 5), True),
    "N(0,I) - 1e4 d=5, start sd 1e-3": (gaussian_start(5, 1e-3, offset=-1e4), True),
    "N(0,I) d=5, start sd 1000": (gaussian(wide=1000.0), True),
    "constant log_prob -3.7, grad 0, d=3": (constant(), True),
    "ReLU network, 50 units, d=10": (relu_network(), True),
    "piecewise-linear table of -x^2/2, spacing 0.01, d=5": (linear_table(), True),
}

GROUP_B = {
    "Jacobi solver rtol 1e-3 d=5 (exact gradient)": loose_solver(1e-3),
    "Jacobi solver rtol 1e-5 d=5 (exact gradient)": loose_solver(1e-5),
    "Jacobi solver rtol 1e-8 d=5 (exact gradient)": loose_solver(1e-8),
    "N(0,I) d=5: grad x 1.005": modify(_N5, times(1.005)),
    "N(0,I) d=1000: grad x 1.0001": modify(gaussian_start(1000, 2.0), times(1.0001)),
    "N(0,I) d=5: grad + 0.02": modify(_N5, plus(0.02)),
    "logistic n=1e4 d=5: without the prior": modify(logistic(), lambda x, g: g + x),
    "logistic n=1e4 d=5, start sd 10: without the prior": modify(
        logistic(init_sd=10.0), lambda x, g: g + x
    ),
    "N(0,I) d=5, start 100x wide: grad x 1.005": modify(gaussian(wide=200.0), times(1.005)),
}


def _pair(case, f):
    return case, modify(case, f)


def _group_c():
    out = {
        "N(0,I) d=5: wrong sign": _pair(_N5, times(-1.0)),
        "N(0,I) d=5: twice": _pair(_N5, times(2.0)),
        "N(0,I) d=50: twice": _pair(gaussian(d=50), times(2.0)),
        "N(0,I) d=200: twice": _pair(gaussian(d=200), times(2.0)),
        "Student t3 d=5, start sd 2: gradient of N(0, I)": (
            student(wide=2.0),
            (student()[0], lambda x: -x, student(wide=2.0)[2]),
        ),
        "equicorrelated rho 0.9 d=10: gradient of rho 0.8": (
            equicorrelated(0.9),
            (equicorrelated(0.9)[0], equicorrelated(0.8)[1], equicorrelated(0.9)[2]),
        ),
    }
    for kappa in (1.0, 2.0):
        for d in (5, 50, 200, 1000, 2000, 5000, 10_000):
            out[f"N(0,I) d={d}, start sd {kappa:g}: sign of coordinate 0"] = _pair(
                gaussian_start(d, kappa), flip(0)
            )
    out["N(0,I) d=10000, start sd 2: signs of coordinates 0..9"] = _pair(
        gaussian_start(10_000, 2.0), flip(*range(10))
    )
    out["N(0,I) d=5, start sd 2: coordinate 0 set to 0"] = _pair(_N5, zero(0))
    for narrow in (0.1, 0.01):
        scales = [1.0] + [narrow] * 4
        out[f"scales (1, {narrow:g} x4), start N(0, I): sign of the wide coordinate"] = _pair(
            diag_scales(scales, init_scales=np.ones(5)), flip(0)
        )
    for narrow in (0.1, 0.01):
        scales = [1.0] + [narrow] * 4
        out[
            f"scales (1, {narrow:g} x4), start N(0, {narrow:g}^2 I): sign of the wide coordinate"
        ] = _pair(diag_scales(scales, init_scales=np.full(5, narrow)), flip(0))
    for tight in (0.1, 0.01):
        for d in (5, 100, 1000):
            out[f"N(0,I) d={d}, start sd ({tight:g}, 2, ...): sign of coordinate 0"] = _pair(
                scaled_start((tight, 2.0), d), flip(0)
            )
    for wide in (100.0, 300.0, 1000.0, 1e4):
        out[f"N(0,I) d=5, start sd {wide:g}: grad + 2"] = _pair(gaussian(wide=wide), plus(2.0))
    out["N(0,I) d=5, start sd 1000: sign of coordinate 0"] = _pair(gaussian(wide=1000.0), flip(0))
    for c in (1e2, 1e4):
        out[f"rotated cond {c:g} d=10, matched: sign along the widest axis"] = rotated_axis(c)
    out["rotated cond 1e4 d=2000, matched: sign along the widest axis"] = rotated_axis(1e4, 2000)
    out["float32 grad, N(0,I) d=5: sign of coordinate 0"] = (_N5, g32(modify(_N5, flip(0))))
    out["float32 grad, N(0,I) d=1000, start sd 2: sign of coordinate 0"] = (
        gaussian_start(1000, 2.0),
        g32(modify(gaussian_start(1000, 2.0), flip(0))),
    )
    out["float32 grad, N(1e3,I) d=5: sign of coordinate 0"] = (
        gaussian(mean=1e3),
        g32(modify(gaussian(mean=1e3), flip(0))),
    )
    out["plateau d=3, start N(0, I): grad = -x"] = (
        plateau(),
        (plateau()[0], lambda x: -x, plateau()[2]),
    )
    two = point(gaussian(d=2), [1.0, 1.0])
    out["N(0,I) d=2, point start (1, 1): grad + 2 (1, -1)"] = _pair(two, plus([2.0, -2.0]))
    out["N(0,I) d=5, random point start: sign of coordinate 0"] = _pair(
        random_point_start(_N5, 5), flip(0)
    )
    return out


GROUP_C = _group_c()

FORMULA = {
    "N(0,I) d=10000, start sd (0.01, 2, ...): sign of coordinate 0": _pair(
        scaled_start((0.01, 2.0), 10_000), flip(0)
    ),
    "N(0,I) d=1000, start sd (0.003, 2, ...): sign of coordinate 0": _pair(
        scaled_start((0.003, 2.0), 1000), flip(0)
    ),
    "N(0,I) d=5, start sd (1e-4, 2, ...): sign of coordinate 0": _pair(
        scaled_start((1e-4, 2.0), 5), flip(0)
    ),
    "scales (1, 1e-4 x4), start N(0, I): sign of the wide coordinate": _pair(
        diag_scales([1.0] + [1e-4] * 4, init_scales=np.ones(5)), flip(0)
    ),
    "N(0,I) - 1e4 d=5, start sd 1e-3: sign of coordinate 0": _pair(
        gaussian_start(5, 1e-3, offset=-1e4), flip(0)
    ),
    "N(0,I) d=1000, start sd (0.5, 2, ...): sign of coordinate 0": _pair(
        scaled_start((0.5, 2.0), 1000), flip(0)
    ),
    "N(0,I) d=1000, start sd (0.25, 2, ...): sign of coordinate 0": _pair(
        scaled_start((0.25, 2.0), 1000), flip(0)
    ),
}

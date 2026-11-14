"""The targets that the exploratory scripts of the design round run on.

Each case is ``(log_prob, grad, initial)``, vectorized: ``log_prob`` takes ``(n, d)`` and
returns ``(n,)``. The constructors and the three lists are those of the first experiment
of issue #123 and of its adversarial review, rewritten here so that the design round's
scripts run from this folder alone, with their names and their order unchanged, because
the seeds of a family follow its position in its list:

- ``CORRECT`` and ``OUT_OF_REACH``: the first experiment's correct families and those
  it named as outside the check's reach.
- ``REVIEW_CORRECT``: the review's probes of correct gradients, each with whether its
  functions are vectorized.
- ``REVIEW_WRONG``: the review's wrong gradients that the first design missed, each as
  ``(correct case, wrong case)``.

A point start whose point depends on the seed is a callable of the seed: ``case(seed)``.
"""

import numpy as np
from scipy import special, stats

from telescopic import examples
from telescopic.initial import Empirical, Normal


def gaussian(scale=1.0, d=5, wide=2.0, mean=0.0, offset=0.0):
    def log_prob(x):
        return -0.5 * np.sum(((x - mean) / scale) ** 2, axis=-1) + offset

    def grad(x):
        return -(x - mean) / scale**2

    return log_prob, grad, Normal(np.full(d, mean), np.full(d, (wide * scale) ** 2))


def _rotated(condition, d, state):
    rotation = stats.ortho_group.rvs(d, random_state=state)
    scales = np.logspace(0, -0.5 * np.log10(condition), d)
    precision = rotation @ np.diag(scales**-2) @ rotation.T
    cov = rotation @ np.diag(scales**2) @ rotation.T
    return rotation, precision, cov


def conditioned(condition, d=10, wide=None, state=0):
    _, precision, cov = _rotated(condition, d, state)
    if wide is None:
        init = Normal(np.zeros(d), 4 * cov)
    else:
        init = Normal(np.zeros(d), np.full(d, wide**2))
    return (
        lambda x: -0.5 * np.einsum("...i,ij,...j->...", x, precision, x),
        lambda x: -x @ precision,
        init,
    )


def example(name):
    t = getattr(examples, name)()
    return t.log_prob, t.grad, t.initial


def student(d=5, df=3.0, wide=10.0):
    def log_prob(x):
        return -0.5 * (df + d) * np.log1p(np.sum(x * x, axis=-1) / df)

    def grad(x):
        return -(df + d) * x / (df + np.sum(x * x, axis=-1, keepdims=True))

    return log_prob, grad, Normal(np.zeros(d), np.full(d, wide**2))


def laplace(d=5, wide=2.0):
    return (
        lambda x: -np.sum(np.abs(x), axis=-1),
        lambda x: -np.sign(x),
        Normal(np.zeros(d), np.full(d, wide**2)),
    )


def half_normal(d=3):
    # Support x > 0 in every coordinate; the initial distribution puts mass outside it,
    # so some draws have zero density and some differences reach the boundary.
    def log_prob(x):
        inside = np.all(x > 0, axis=-1)
        return np.where(inside, -0.5 * np.sum(x * x, axis=-1), -np.inf)

    return log_prob, lambda x: -x, Normal(np.full(d, 1.0), np.full(d, 1.0))


def logistic(n=10_000, d=5, init_sd=0.1, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, d))
    labels = rng.random(n) < special.expit(features @ np.linspace(-1, 1, d))
    s = np.where(labels, 1.0, -1.0)
    features, s = features.astype(dtype), s.astype(dtype)

    def log_prob(b):
        b = b.astype(dtype)
        m = s * (features @ b.T).T
        return -np.sum(np.logaddexp(0, -m), axis=-1) - 0.5 * np.sum(b**2, axis=-1)

    def grad(b):
        b = b.astype(dtype)
        m = s * (features @ b.T).T
        return (s * special.expit(-m)) @ features - b

    return log_prob, grad, Normal(np.zeros(d), np.full(d, init_sd**2))


def as_float32(case):
    log_prob, grad, init = case
    return (
        lambda x: log_prob(x).astype(np.float32),
        lambda x: grad(x).astype(np.float32),
        init,
    )


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
    init = scales * wide if init_scales is None else np.asarray(init_scales, float)
    return (
        lambda x: -0.5 * np.sum((x / scales) ** 2, axis=-1),
        lambda x: -x / scales**2,
        Normal(np.zeros(len(scales)), init**2),
    )


def diag_student(scales, df=3.0, init_scales=None):
    scales = np.asarray(scales, float)

    def log_prob(x):
        return -0.5 * (df + 1) * np.sum(np.log1p((x / scales) ** 2 / df), axis=-1)

    def grad(x):
        z = x / scales
        return -(df + 1) * z / (df + z * z) / scales

    init = 2 * scales if init_scales is None else np.asarray(init_scales, float)
    return log_prob, grad, Normal(np.zeros(len(scales)), init**2)


# Outside the check's reach.
def cancelling(c, d=1):
    def log_prob(x):
        return -0.5 * np.sum((x + c) ** 2 - 2 * c * x - c**2, axis=-1)

    return log_prob, lambda x: -x, Normal(np.zeros(d), np.full(d, 4.0))


def fringes(n_fringes, d=1, amp=5.0):
    period = 2 * np.pi * n_fringes

    def log_prob(x):
        return amp * np.sum(np.cos(period * x), axis=-1) - 0.5 * np.sum(x * x, axis=-1)

    def grad(x):
        return -amp * period * np.sin(period * x) - x

    return log_prob, grad, Normal(np.zeros(d), np.ones(d))


def loose_solver(rtol, d=5):
    # log N(0, A) = -x.y/2 with y solving A y = x by Jacobi iteration, A = I + 0.3 T (T the
    # tridiagonal Laplacian), stopped at a relative change below rtol: the log-density
    # carries the solver's error, the gradient -A^-1 x is exact.
    matrix = np.eye(d) + 0.3 * (2 * np.eye(d) - np.eye(d, k=1) - np.eye(d, k=-1))
    inverse = np.linalg.inv(matrix)
    diagonal = np.diag(matrix)

    def solve(x):
        y = x / diagonal
        for _ in range(500):
            new = (x - (matrix @ y.T).T + diagonal * y) / diagonal
            done = np.max(np.abs(new - y)) <= rtol * np.max(np.abs(new))
            y = new
            if done:
                break
        return y

    def log_prob(x):
        return np.array([-0.5 * row @ solve(row) for row in np.atleast_2d(x)])

    return log_prob, lambda x: -x @ inverse, Normal(np.zeros(d), np.full(d, 4.0))


# The review's probes of correct gradients.
def flat(d=3, dtype=float):
    return (
        lambda x: np.zeros(len(x), dtype=dtype),
        lambda x: np.zeros(x.shape),
        Normal(np.zeros(d), np.ones(d)),
    )


def plateau(d=3, radius=10.0):
    # A flat top with Gaussian shoulders; the starts are inside the flat top.
    def log_prob(x):
        return -0.5 * np.sum(np.maximum(np.abs(x) - radius, 0.0) ** 2, axis=-1)

    def grad(x):
        return -np.sign(x) * np.maximum(np.abs(x) - radius, 0.0)

    return log_prob, grad, Normal(np.zeros(d), np.ones(d))


def hinge(initial):
    def log_prob(x):
        return -np.sum(np.maximum(x, 0.0), axis=-1) - 0.5 * np.sum(x * x, axis=-1)

    return log_prob, lambda x: -(x > 0).astype(float) - x, initial


def asym_laplace(initial):
    def log_prob(x):
        return -np.sum(np.where(x > 0, 2 * x, -x), axis=-1)

    return log_prob, lambda x: -np.where(x > 0, 2.0, np.where(x < 0, -1.0, 0.5)), initial


def naive_sum(n=100_000, seed=0):
    y = np.random.default_rng(seed).standard_normal(n) * 3 + 1e3

    def log_prob(x):  # x = (mu, log sigma); sequential accumulation of n terms
        mu, ls = x[:, 0:1], x[:, 1:2]
        terms = -0.5 * ((y - mu) * np.exp(-ls)) ** 2 - ls
        return np.cumsum(terms, axis=1)[:, -1]

    def grad(x):
        mu, ls = x[:, 0:1], x[:, 1:2]
        r = (y - mu) * np.exp(-ls)
        return np.stack([np.sum(r * np.exp(-ls), axis=1), np.sum(r * r - 1, axis=1)], axis=1)

    return log_prob, grad, Normal(np.array([1e3, 1.0]), np.array([0.01, 0.01]) ** 2)


def poisson(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    design = rng.standard_normal((n, 4))
    y = rng.poisson(np.exp(5 + design @ [0.3, -0.2, 0.1, 0.0]))

    def log_prob(b):
        eta = 5 + b @ design.T
        return np.cumsum(y * eta - np.exp(eta), axis=1)[:, -1]

    def grad(b):
        eta = 5 + b @ design.T
        return (y - np.exp(eta)) @ design

    return log_prob, grad, Normal(np.zeros(4), np.full(4, 0.5**2))


def grad32(case):
    log_prob, grad, init = case
    return log_prob, (lambda x: np.asarray(grad(x.astype(np.float32)), dtype=np.float32)), init


def gaussian32grad(mean, d=5):
    # grad computed wholly in float32 from a float32 copy of x, as autodiff without 64-bit mode
    m = np.float32(mean)

    def log_prob(x):
        return -0.5 * np.sum((x - mean) ** 2, axis=-1)

    def grad(x):
        return -(x.astype(np.float32) - m)

    return log_prob, grad, Normal(np.full(d, mean), np.full(d, 4.0))


def rotated32grad(condition, d=10):
    log_prob, _, init = conditioned(condition, d)
    precision32 = _rotated(condition, d, 0)[1].astype(np.float32)
    return log_prob, (lambda x: -(x.astype(np.float32) @ precision32)), init


def all32(mean, d=5):
    # log_prob and grad computed wholly in float32 from a float32 copy of x
    m = np.float32(mean)

    def log_prob(x):
        return -np.float32(0.5) * np.sum((x.astype(np.float32) - m) ** 2, axis=-1)

    def grad(x):
        return -(x.astype(np.float32) - m)

    return log_prob, grad, Normal(np.full(d, float(mean)), np.full(d, 4.0))


def point(case, at):
    return case[0], case[1], Empirical(np.atleast_2d(np.asarray(at, float)))


def per_point(case):
    log_prob, grad, init = case
    return (lambda x: float(log_prob(x[None])[0])), (lambda x: grad(x[None])[0]), init


# Wrong gradients.
def modify(case, f):
    log_prob, grad, init = case
    return log_prob, (lambda x: f(x, np.array(grad(x), dtype=float))), init


def flip(i):
    def f(x, g):
        g[..., i] = -g[..., i]
        return g

    return f


def flip0(case):
    return case, modify(case, flip(0))


def rotated_axis(condition, d=10):
    # The sign of the gradient's component along the widest principal axis.
    rotation, _, cov = _rotated(condition, d, 0)
    log_prob, grad, _ = conditioned(condition, d)
    v = rotation[:, 0]

    def wrong(x):
        return grad(x) - 2 * np.outer(grad(x) @ v, v)

    init = Normal(np.zeros(d), 4 * cov)
    return (log_prob, grad, init), (log_prob, wrong, init)


def equicorrelated(rho, d=10):
    cov = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    precision = np.linalg.inv(cov)
    return (
        lambda x: -0.5 * np.einsum("...i,ij,...j->...", x, precision, x),
        lambda x: -x @ precision,
        Normal(np.zeros(d), cov),
    )


def resolve(case, seed):
    return case(seed) if callable(case) else case


_N5 = gaussian()
_EQUI = equicorrelated(0.9)
_T5 = student(wide=2.0)
_FUNNEL_DIM = len(
    np.atleast_1d(examples.funnel().initial.rvs(random_state=np.random.default_rng(0)))
)

CORRECT = {
    # Gaussians d = 5 at scales 1e-8 .. 1e8, matched (sd 2 scale) and wide (sd 200 scale) starts
    **{f"N(0, {s:g}^2 I) d=5, matched": gaussian(s) for s in (1e-8, 1e-4, 1.0, 1e4, 1e8)},
    **{
        f"N(0, {s:g}^2 I) d=5, wide 100x": gaussian(s, wide=200.0)
        for s in (1e-8, 1e-4, 1.0, 1e4, 1e8)
    },
    # conditioning, d = 10
    **{f"condition {c:g} d=10, matched": conditioned(c) for c in (1e4, 1e8, 1e12)},
    **{f"condition {c:g} d=10, wide sd 100": conditioned(c, wide=100.0) for c in (1e4, 1e8, 1e12)},
    "N(1e6, I) d=5": gaussian(mean=1e6),
    "N(0,I) + 1e9 d=5": gaussian(offset=1e9),
    "N(0,I) - 1e12 d=5": gaussian(offset=-1e12),
    "banana": example("banana"),
    "funnel": example("funnel"),
    "mixture": example("mixture"),
    "pumps": example("pumps"),
    "Student t3 d=1": student(d=1),
    "Student t3 d=5": student(),
    "Laplace d=5": laplace(),
    "Laplace d=100": laplace(100),
    "half-normal boundary d=3": half_normal(),
    "logistic n=1e4 d=5": logistic(),
    "logistic n=2000 d=50": logistic(n=2000, d=50),
    "float32 N(0,I) d=5": as_float32(gaussian()),
    "float32 N(0,I) - 1e4 d=5": as_float32(gaussian(offset=-1e4)),
    "float32 Student t3 d=5": as_float32(student(wide=2.0)),
    "logistic computed in float32 d=5": logistic(dtype=np.float32),
    "point start N(0,I) d=5, random point": random_point_start(gaussian(), 5),
    "point start logistic d=5, at 0": with_initial(logistic(), Empirical(np.zeros((1, 5)))),
    "point start Student t3 d=5, random point": random_point_start(student(), 5),
    "scales 1e-3..1e3 d=7, matched": diag_scales(np.logspace(-3, 3, 7)),
    "t3 scales 1e-3..1e3 d=7, isotropic init 2e3": diag_student(
        np.logspace(-3, 3, 7), init_scales=np.full(7, 2e3)
    ),
    "N(0,I) d=5, init sd 1e-12 at 1": with_initial(
        gaussian(), Normal(np.ones(5), np.full(5, 1e-24))
    ),
    "N(0,I) d=1": gaussian(d=1),
    "N(0,I) d=50": gaussian(d=50),
    "N(0,I) d=1000": gaussian(d=1000),
    "Student t3 d=1000": student(d=1000, wide=2.0),
}

OUT_OF_REACH = {
    "cancelling terms of 1e8 d=1": cancelling(1e4),
    "cancelling terms of 1e12 d=1": cancelling(1e6),
    "cancelling terms of 1e12 d=5": cancelling(1e6, 5),
    "fringes 1e5 periods d=1": fringes(1e5),
    "fringes 1e6 periods d=1": fringes(1e6),
    "Jacobi solver rtol 1e-3 d=5": loose_solver(1e-3),
    "Jacobi solver rtol 1e-8 d=5": loose_solver(1e-8),
}

# name: (case, whether its functions are vectorized)
REVIEW_CORRECT = {
    "flat log_prob 0.0, grad 0, d=3": (flat(), True),
    "flat log_prob int 0, grad 0, d=3": (flat(dtype=int), True),
    "plateau (flat top |x|<10, Gaussian shoulders) d=3, init N(0,I)": (plateau(), True),
    "hinge -relu(x) - x^2/2 d=3, point start at 0": (hinge(Empirical(np.zeros((1, 3)))), True),
    "hinge d=3, init N(0,I)": (hinge(Normal(np.zeros(3), np.ones(3))), True),
    "asymmetric Laplace d=3, point start at 0": (
        asym_laplace(Empirical(np.zeros((1, 3)))),
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
            _N5,
            Empirical(
                np.random.default_rng(5).standard_normal((50, 5)) * np.array([0, 0, 1.0, 1, 1])
            ),
        ),
        True,
    ),
    "N(0,I) d=5, init sd 1e-150": (
        with_initial(_N5, Normal(np.zeros(5), np.full(5, 1e-300))),
        True,
    ),
    "N(0,I) d=5, Empirical of 2 points": (
        with_initial(_N5, Empirical(np.array([[1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 5.5]]))),
        True,
    ),
    "N(0,I) d=5, point start at 0": (point(_N5, np.zeros(5)), True),
    "Student t3 d=5, point start at 0": (point(student(), np.zeros(5)), True),
    "funnel, point start at 0": (point(example("funnel"), np.zeros(_FUNNEL_DIM)), True),
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
}

# name: (correct case, wrong case)
REVIEW_WRONG = {
    # the wrong coordinate's spread is small relative to its target scale
    **{
        f"N(0,I) d=5, init sd ({sd:g},2,2,2,2): sign of coordinate 0": flip0(
            with_initial(_N5, Normal(np.zeros(5), np.array([sd, 2, 2, 2, 2.0]) ** 2))
        )
        for sd in (0.01, 0.1)
    },
    # mixed scales from a start that does not whiten the target
    "scales (1, 0.1 x4), init N(0, I): sign of the wide coordinate": flip0(
        diag_scales([1, 0.1, 0.1, 0.1, 0.1], init_scales=np.ones(5))
    ),
    "scales (1, 0.01 x4), init N(0, I): sign of the wide coordinate": flip0(
        diag_scales([1, 0.01, 0.01, 0.01, 0.01], init_scales=np.ones(5))
    ),
    "scales (1, 0.01 x4), init N(0, 0.01^2 I): sign of the wide coordinate": flip0(
        diag_scales([1, 0.01, 0.01, 0.01, 0.01], init_scales=np.full(5, 0.01))
    ),
    "scales (1, 1e-4 x4), init N(0, I): sign of the wide coordinate": flip0(
        diag_scales([1, 1e-4, 1e-4, 1e-4, 1e-4], init_scales=np.ones(5))
    ),
    # wrong in a rotated basis
    **{
        f"rotated condition {c:g} d=10, matched: sign along the widest principal axis": (
            rotated_axis(c)
        )
        for c in (1e2, 1e4, 1e8)
    },
    # the gradient of a similar density
    "equicorrelated rho 0.9 d=10: gradient of rho 0.8": (
        _EQUI,
        (_EQUI[0], equicorrelated(0.8)[1], _EQUI[2]),
    ),
    "equicorrelated rho 0.9 d=10: gradient of rho 0 (independent)": (
        _EQUI,
        (_EQUI[0], lambda x: -x, _EQUI[2]),
    ),
    "Student t3 d=5, init sd 2: gradient of N(0, I)": (_T5, (_T5[0], lambda x: -x, _T5[2])),
    # constant offsets from wide starts
    **{
        f"N(0,I) d=5, init sd {w:g}: grad + 2": (
            gaussian(wide=w),
            modify(gaussian(wide=w), lambda x, g: g + 2.0),
        )
        for w in (300.0, 1000.0)
    },
    "N(0,I) d=200: sign of coordinate 0": flip0(gaussian(d=200)),
}

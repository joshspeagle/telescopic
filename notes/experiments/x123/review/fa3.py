"""Review of x123: false alarms with root solvers, mixed precision, special starts and d = 1e6.

Question: does the check stay silent on correct gradients when log_prob comes from a
Newton or bisection solve, when log_prob or grad are computed in a narrower precision
than they are returned in, when the start is on a lattice or at a boundary or mostly
outside the support, and at d = 1e6? And how often does it warn when a solver's
tolerance makes log_prob inexact while the gradient is exact?

Design: probe.run on seeds 500000 to 500199 (200 seeds), 500000 to 500099 or 500049
where a family is slow, and 500000 to 500019 at d = 1e6; one part per run. "kepler":
Kepler's equation by Newton iteration with the implicit-function gradient, a root by
bisection, and the sinusoid fit of fa1.py at longer time spans. "precision": float32
arithmetic returned as float64, float32 gradients of 11 targets outside the registered
ones, float16 and longdouble, and counts of 1e15 through float(). "starts": integer
lattices, 0/1 vectors, boundaries, thin slabs and the simplex. "big": d = 1e6.

Run: PYTHONPATH=src:notes/experiments/x123 python notes/experiments/x123/review/fa3.py kepler|precision|starts|big > fa3_<part>.out

Result: see fa3_kepler.out, fa3_precision.out, fa3_starts.out, fa3_big.out and REVIEW.md,
objections 3, 4 and 6 and the checks that found nothing. (N) warns on 200 and 72 of 200
with Newton tolerances 1e-2 and 1e-4, on 3, 59 and 200 of 200 sinusoid fits with 4.8e4,
1.6e5 and 1.6e6 fringes across the start, on 23 of 200 with float32 arithmetic returned
as float64 at N(1e4, I), and on 11 of 200 with a gradient computed from a float32 copy
of x and returned as float64 (Student t3, d = 50); on nothing else. A start on a boundary
or mostly outside the support leaves no line to compare (200 of 200 checks).
"""

import sys

import families as F
import numpy as np
from probe import run
from scipy import special

from telescopic.initial import Empirical, Normal

which = sys.argv[1]
S = range(500_000, 500_200)
S100 = range(500_000, 500_100)
if which == "kepler":

    def kepler(tol, d=3):
        M = np.linspace(0.3, 2.8, 8)

        def E_of(e):
            E = M + 0 * e
            for _ in range(100):
                step = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
                E = E - step
                if np.max(np.abs(step)) < tol:
                    break
            return E

        dat = E_of(np.array([[0.3]]))[0] + 0.01 * np.random.default_rng(1).standard_normal(8)

        def lp(x):
            e = special.expit(x[:, 0:1])
            return -0.5 * np.sum((dat - E_of(e)) ** 2, axis=1) / 1e-4 - 0.5 * np.sum(x * x, axis=1)

        def g(x):
            e = special.expit(x[:, 0:1])
            E = E_of(e)
            dE = np.sin(E) / (1 - e * np.cos(E))
            ge = np.sum((dat - E) * dE, axis=1) / 1e-4 * (e * (1 - e))[:, 0]
            out = -x.copy()
            out[:, 0] += ge
            return out

        return lp, g, Normal(np.zeros(d), np.full(d, 0.5**2))

    for tol in (1e-2, 1e-4, 1e-8, 1e-12):
        run(
            f"Kepler equation by Newton to {tol:g} (stop tested over the batch), implicit-function gradient",
            kepler(tol),
            S,
        )
    for tol in (1e-2, 1e-4, 1e-8):
        run(
            f"the same, not vectorized (stop per point), tol {tol:g}",
            F.per_point(kepler(tol)),
            S,
            vectorize=False,
        )

    # bisection root to a tolerance (linear convergence: the error is of the size of the tolerance)
    def bisect(tol):
        def root(a):  # solve y + y^3 = a for y
            lo, hi = np.zeros_like(a) - 10, np.zeros_like(a) + 10
            while np.max(hi - lo) > tol:
                mid = 0.5 * (lo + hi)
                up = mid + mid**3 < a
                lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
            return 0.5 * (lo + hi)

        lp = lambda x: -0.5 * np.sum(root(x) ** 2, axis=1)

        def g(x):
            y = root(x)
            return -y / (1 + 3 * y * y)

        return lp, g, Normal(np.zeros(3), np.full(3, 4.0))

    for tol in (1e-4, 1e-8, 1e-12, 1e-15):
        run(f"root by bisection to {tol:g}, implicit-function gradient", bisect(tol), S)

    # sinusoid at longer spans
    def sinusoid(T, n=200, sd_w=0.1, seed=0):
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0, T, n))
        w0 = 1.0
        y = np.sin(w0 * t) + 0.1 * rng.standard_normal(n)

        def lp(x):
            w, a = x[:, 0:1], x[:, 1:2]
            r = y - a * np.sin(w * t)
            return -0.5 * np.sum(r * r, axis=1) / 0.01

        def g(x):
            w, a = x[:, 0:1], x[:, 1:2]
            s, c = np.sin(w * t), np.cos(w * t)
            r = y - a * s
            return np.stack([np.sum(r * a * t * c, axis=1), np.sum(r * s, axis=1)], axis=1) / 0.01

        return lp, g, Normal(np.array([w0, 1.0]), np.array([sd_w, 0.1]) ** 2)

    for T in (3e6, 1e7, 1e8):
        run(
            f"sinusoid fit, time span {T:g} (start sd 0.1 in frequency: {0.1 * T / (2 * np.pi):.3g} fringes)",
            sinusoid(T),
            S,
        )
    for T, sdw in ((1e3, 10.0), (1e4, 10.0)):
        run(
            f"sinusoid fit, time span {T:g}, start sd {sdw:g} in frequency ({sdw * T / (2 * np.pi):.3g} fringes)",
            sinusoid(T, sd_w=sdw),
            S,
        )
if which == "precision":

    def gauss32_as64(mean, d=5, lp64=True, g64=True):
        m = np.float32(mean)

        def lp(x):
            v = -np.float32(0.5) * np.sum((x.astype(np.float32) - m) ** 2, axis=-1)
            return v.astype(np.float64) if lp64 else v

        def g(x):
            v = -(x.astype(np.float32) - m)
            return v.astype(np.float64) if g64 else v

        return lp, g, Normal(np.full(d, float(mean)), np.full(d, 4.0))

    for mean in (0.0, 1.0, 10.0, 100.0, 1e3, 1e4):
        run(
            f"N({mean:g}, I) d=5 computed in float32, both returned as float64",
            gauss32_as64(mean),
            S,
        )
    for mean in (0.0, 100.0, 1e3):
        run(
            f"N({mean:g}, I) d=5 computed in float32: log_prob returned as float64, grad as float32",
            gauss32_as64(mean, g64=False),
            S,
        )
        run(
            f"N({mean:g}, I) d=5 computed in float32: log_prob returned as float32, grad as float64",
            gauss32_as64(mean, lp64=False),
            S,
        )
    for mean in (0.0, 100.0, 1e3):
        c = gauss32_as64(mean)
        run(
            f"N({mean:g}, I) d=5 float32 inside, per point, returned through float(...)",
            (
                (lambda x, c=c: float(c[0](x[None])[0])),
                (lambda x, c=c: [float(v) for v in c[1](x[None])[0]]),
                c[2],
            ),
            S,
            vectorize=False,
        )
    lg = F.logistic(dtype=np.float32)
    run(
        "logistic n=1e4 d=5 computed in float32, both returned as float64",
        (lambda x: lg[0](x).astype(np.float64), lambda x: lg[1](x).astype(np.float64), lg[2]),
        S100,
    )
    st = F.student(d=50, wide=2.0)
    run(
        "Student t3 d=50: float64 log_prob, gradient computed from float32 x, returned float64",
        (st[0], lambda x: np.asarray(st[1](x.astype(np.float32)), dtype=np.float64), st[2]),
        S,
    )
    # float32 gradients outside the 8 registered families
    run("float64 log_prob, float32 grad: Student t3 d=50", F.grad32(F.student(d=50, wide=2.0)), S)
    run(
        "float64 log_prob, float32 grad: scales 1e-3..1e3 d=7 matched",
        F.grad32(F.diag_scales(np.logspace(-3, 3, 7))),
        S,
    )
    run(
        "float64 log_prob, float32 grad: scales 1e-6..1e6 d=13 matched",
        F.grad32(F.diag_scales(np.logspace(-6, 6, 13))),
        S,
    )
    run("float64 log_prob, float32 grad: N(1e8, I) d=5", F.gaussian32grad(1e8), S)
    run(
        "float64 log_prob, float32 grad: logistic n=2000 d=50",
        F.grad32(F.logistic(n=2000, d=50)),
        S100,
    )
    run("float64 log_prob, float32 grad: Poisson regression n=2e4", F.grad32(F.poisson()), S100)
    run(
        "float64 log_prob, float32 grad: naive sum of 1e5 terms",
        F.grad32(F.naive_sum()),
        range(500_000, 500_050),
    )
    run("float64 log_prob, float32 grad: funnel", F.grad32(F.example("funnel")), S)
    run("float64 log_prob, float32 grad: banana", F.grad32(F.example("banana")), S)
    run(
        "float64 log_prob, float32 grad: N(0,I) d=10000 start sd 2",
        F.grad32(F.gaussian_start(10_000, 2.0)),
        S100,
    )
    rc = F.conditioned(1e4, 200)
    run(
        "float64 log_prob, float32 grad: rotated cond 1e4 d=200 matched",
        (rc[0], lambda x: np.asarray(rc[1](x.astype(np.float32)), dtype=np.float32), rc[2]),
        S,
    )
    n5 = F.gaussian()
    run(
        "float64 log_prob, float16 grad: N(0,I) d=5",
        (n5[0], lambda x: n5[1](x).astype(np.float16), n5[2]),
        S,
    )
    run(
        "float16 throughout: N(0,I) d=5",
        (
            lambda x: n5[0](x.astype(np.float16)).astype(np.float16),
            lambda x: n5[1](x.astype(np.float16)).astype(np.float16),
            n5[2],
        ),
        S,
    )
    run(
        "longdouble log_prob and grad: N(0,I) d=5",
        (
            lambda x: n5[0](x).astype(np.longdouble),
            lambda x: n5[1](x).astype(np.longdouble),
            n5[2],
        ),
        S,
    )

    # large integers through float(): a count likelihood whose terms are Python ints
    def counts(d=3):
        n = np.array([10**15, 3 * 10**15, 7 * 10**14])
        k = np.array([4 * 10**14, 10**15, 5 * 10**14])

        # binomial log-likelihood in logit parameters, counts as Python ints converted by float()
        def lp(x):
            return np.array(
                [
                    sum(
                        float(int(ki)) * xi - float(int(ni)) * np.logaddexp(0, xi)
                        for ki, ni, xi in zip(k, n, row, strict=True)
                    )
                    for row in x
                ]
            )

        g = lambda x: k - n * special.expit(x)
        return lp, g, Normal(special.logit(k / n), np.full(d, (1e-7) ** 2))

    run(
        "binomial likelihood with counts of 1e15 (|log_prob| ~ 1e15), start matched (sd 1e-7)",
        counts(),
        S,
    )
    c = counts()
    run("the same, start sd 1", (c[0], c[1], Normal(np.zeros(3), np.ones(3))), S)
if which == "starts":
    lpN = lambda x: -0.5 * np.sum(x * x, axis=-1)
    gN = lambda x: -x
    lat = lambda d, sd: lambda generator: np.round(sd * generator.standard_normal(d))
    for d, sd in ((5, 1.0), (5, 3.0), (100, 1.0), (1000, 0.5)):
        run(
            f"N(0,I) d={d}, start on the integer lattice (round of sd {sd:g})",
            (lpN, gN, lat(d, sd)),
            S,
        )
        run(
            f"Laplace d={d}, start on the integer lattice (round of sd {sd:g})",
            (F.laplace(3)[0], F.laplace(3)[1], lat(d, sd)),
            S,
        )
        h = F.hinge()
        run(
            f"hinge d={d}, start on the integer lattice (round of sd {sd:g})",
            (h[0], h[1], lat(d, sd)),
            S,
        )
        a = F.asym_laplace()
        run(
            f"asymmetric Laplace d={d}, start on the integer lattice (round of sd {sd:g})",
            (a[0], a[1], lat(d, sd)),
            S,
        )
    rngE = np.random.default_rng(3)
    run(
        "N(0,I) d=20, Empirical of 0/1 vectors (50 rows)",
        (lpN, gN, Empirical(rngE.integers(0, 2, (50, 20)).astype(float))),
        S,
    )
    run(
        "logistic n=1e4 d=5, Empirical of 0/1 vectors (20 rows)",
        F.with_initial(F.logistic(), Empirical(rngE.integers(0, 2, (20, 5)).astype(float))),
        S100,
    )
    a = F.asym_laplace(20)
    run(
        "asymmetric Laplace d=20, Empirical of -1/0/1 vectors (50 rows)",
        (a[0], a[1], Empirical(rngE.integers(-1, 2, (50, 20)).astype(float))),
        S,
    )
    net = F.relu_network()
    run("ReLU network d=10, start on the integer lattice", (net[0], net[1], lat(10, 2.0)), S)
    # boundaries
    hn = lambda x: np.where(np.all(x >= 0, axis=-1), -0.5 * np.sum(x * x, axis=-1), -np.inf)
    run(
        "half-normal (x >= 0 allowed) d=3, start exactly on the boundary in coordinate 0",
        (hn, gN, lambda generator: np.r_[0.0, np.abs(generator.standard_normal(2))]),
        S,
    )
    run(
        "half-normal d=3, start |N(0,1)| (inside)",
        (hn, gN, lambda generator: np.abs(generator.standard_normal(3))),
        S,
    )
    run(
        "half-normal d=3, start |N(0, 1e-5^2)| (within a few steps of the boundary)",
        (hn, gN, lambda generator: 1e-5 * np.abs(generator.standard_normal(3))),
        S,
    )
    ex = lambda x: np.where(np.all(x > 0, axis=-1), -np.sum(x, axis=-1), -np.inf)
    run(
        "exponential d=3 on x > 0, start exponential",
        (ex, lambda x: -np.ones(x.shape), lambda generator: generator.exponential(size=3)),
        S,
    )
    run(
        "exponential d=100 on x > 0, start exponential",
        (ex, lambda x: -np.ones(x.shape), lambda generator: generator.exponential(size=100)),
        S,
    )
    run(
        "exponential d=10000 on x > 0, start exponential",
        (ex, lambda x: -np.ones(x.shape), lambda generator: generator.exponential(size=10000)),
        S100,
    )
    # -inf on part of every line: support is a thin slab |x_0| < w
    for w in (1e-3, 1e-5, 1e-6, 1e-7):
        slab = lambda x, w=w: np.where(
            np.abs(x[..., 0]) < w, -0.5 * np.sum(x * x, axis=-1), -np.inf
        )
        run(
            f"N(0,I) d=3 restricted to |x_0| < {w:g}, start N(0,I) in the others, uniform in the slab",
            (
                slab,
                gN,
                lambda generator, w=w: np.r_[
                    generator.uniform(-w, w), generator.standard_normal(2)
                ],
            ),
            S,
        )
        run(
            f"the same slab {w:g}, start N(0, I) in every coordinate (most points outside)",
            (slab, gN, Normal(np.zeros(3), np.ones(3))),
            S,
        )
    # simplex: log_prob -inf unless sum(x) = 1 to 1e-9
    simp = lambda x: np.where(
        (np.abs(np.sum(x, axis=-1) - 1) < 1e-9) & np.all(x > 0, axis=-1),
        np.sum(np.log(np.abs(x) + 1e-300), axis=-1),
        -np.inf,
    )
    run(
        "Dirichlet on the simplex (log_prob -inf off it), start Dirichlet",
        (simp, lambda x: 1 / x, lambda generator: generator.dirichlet(np.ones(4))),
        S,
    )
if which == "big":
    for d, n in ((1_000_000, 20),):
        run(f"N(0,I) d={d}, start sd 2", F.gaussian_start(d, 2.0), range(500_000, 500_000 + n))
        run(f"Student t3 d={d}, start sd 2", F.student(d=d, wide=2.0), range(500_000, 500_000 + n))
        run(f"Laplace d={d}, start sd 2", F.laplace(d), range(500_000, 500_000 + n))

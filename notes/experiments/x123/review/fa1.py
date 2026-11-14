"""Review of x123: false alarms of (N) and (H) on correct gradients outside the registered families.

Question: does the check stay silent on correct gradients whose log_prob has structure
that group A of the registration leaves out: periodic terms of high frequency, a
log_prob that is rounded or tabulated, values of 1e12 and above, and d = 1?

Design: 200 seeds per family, 500000 to 500199, one call of gradcheck.check per seed
(probe.run). The gradient is the derivative of log_prob as computed, except where the
row's name says otherwise: the rounded log-densities and the piecewise-constant table
with grad -x keep the gradient of the smooth function.

Run: PYTHONPATH=src:notes/experiments/x123 python notes/experiments/x123/review/fa1.py > fa1.out

Result: see fa1.out and REVIEW.md, objections 3 and 4. (N) warns on 2, 48 and 200 of 200
with 3e4, 1e5 and 1e6 periods across the start, on 143 of 200 with a piecewise-linear
table of spacing 1e-5, on 200 of 200 with a piecewise-constant table of spacing 1e-6 and
grad 0, and on 28 of 200 with a log_prob rounded to 1e-3; on nothing else here.
"""

import families as F
import numpy as np
from probe import run

from telescopic.initial import Normal

S = range(500_000, 500_200)  # 200 seeds per family


def gauss(d, sd=2.0, offset=0.0, scale=1.0):
    return (
        lambda x: -0.5 * np.sum((x / scale) ** 2, axis=-1) + offset,
        lambda x: -x / scale**2,
        F.DiagonalNormal(np.zeros(d), np.full(d, sd)),
    )


print("== periodic terms: fringes across a start of sd 1 (families.fringes, amp 5), d=1 and d=5")
for nf in (1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6):
    run(f"fringes {nf:g} periods, d=1", F.fringes(nf), S)
for nf in (1e3, 1e4, 1e5):
    run(f"fringes {nf:g} periods, d=5", F.fringes(nf, d=5), S)


# a frequency-search likelihood: y_i = sin(w t_i) + noise, t in [0, T]; parameter (w, amplitude)
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


for T in (1e2, 1e3, 1e4, 1e5, 1e6):
    run(
        f"sinusoid fit, time span {T:g} (start sd 0.1 in frequency: {0.1 * T / (2 * np.pi):.3g} fringes)",
        sinusoid(T),
        S,
    )
print("== quantized / tabulated log-densities (grad = derivative of the smooth function)")
lp0, g0, init = gauss(5)
for q in (1e-12, 1e-9, 1e-6, 1e-3):
    run(
        f"N(0,I) d=5, log_prob rounded to multiples of {q:g}",
        (lambda x, q=q: np.round(lp0(x) / q) * q, g0, init),
        S,
    )
for dec in (6, 10, 14):
    run(
        f"N(0,I) d=5, log_prob through float('%.{dec}g' % value)",
        (lambda x, dec=dec: np.array([float(f"%.{dec}g" % v) for v in lp0(x)]), g0, init),
        S,
    )
# nearest-neighbor table (piecewise constant), grad 0 (the derivative of log_prob as computed, a.e.)
for sp in (1e-2, 1e-4, 1e-6):
    run(
        f"piecewise-constant table of -x^2/2, spacing {sp:g}, d=5, grad 0",
        (
            lambda x, sp=sp: -0.5 * np.sum((np.round(x / sp) * sp) ** 2, axis=-1),
            lambda x: np.zeros(x.shape),
            F.DiagonalNormal(np.zeros(5), np.full(5, 2.0)),
        ),
        S,
    )
    run(
        f"piecewise-constant table of -x^2/2, spacing {sp:g}, d=5, grad -x",
        (
            lambda x, sp=sp: -0.5 * np.sum((np.round(x / sp) * sp) ** 2, axis=-1),
            lambda x: -x,
            F.DiagonalNormal(np.zeros(5), np.full(5, 2.0)),
        ),
        S,
    )


# piecewise-linear table at finer spacings and higher d (registered: spacing 0.01, d=5)
def table(d, spacing, edge=8.0):
    knots = round(2 * edge / spacing) + 1
    return F.linear_table(d=d, knots=knots, edge=edge)


for d, sp in ((5, 1e-3), (5, 1e-4), (5, 1e-5), (50, 1e-3), (50, 1e-4), (500, 1e-3)):
    run(f"piecewise-linear table of -x^2/2, spacing {sp:g}, d={d}", table(d, sp), S)
print("== large |log_prob|")
for off in (1e12, -1e15, 1e18, -1e300):
    run(f"N(0,I) d=5 + {off:g}", gauss(5, offset=off), S)
run("N(0, 1e-6^2 I) d=5, start sd 1 (|log_prob| ~ 1e12)", gauss(5, sd=1.0, scale=1e-6), S)
run("N(0, 1e-9^2 I) d=50, start sd 1 (|log_prob| ~ 1e19)", gauss(50, sd=1.0, scale=1e-9), S)
print("== d = 1")
run("N(0,1) d=1, start sd 2", gauss(1), S)
run("Laplace d=1", F.laplace(1), S)
run("fringes 100, d=1", F.fringes(100), S)

"""Exploratory: the invariances of DESIGN.md section 3, on the prototype (seeds 217500..217549).

Question: is the decision of (N) unchanged, and are its gap and tolerance multiplied by the stated
factor, under (a) an affine map of the parameters x -> A x + b applied to log_prob, grad and the
initial distribution together, (b) log_prob -> c log_prob with grad -> c grad, (c) log_prob ->
log_prob + k?  And which of these change (H)?
Design: Student t3 at d = 5 from N(0, diag(0.3, 2, 2, 2, 2)^2), with the sign of coordinate 0 wrong
(a signal far above rounding) and with the correct gradient; A a random matrix of condition 100,
b = (3, -1, 4, 1, -5); c = 8 and c = 7.3; k = 1e6. Per seed: the largest gap / tolerance of the base
check and of the transformed one. Reported: the number of seeds where the decision differs, and the
largest relative change of that ratio (wrong gradient) after the stated factor.
Also: a log_prob that is exactly constant on the lines, with grad = 0 and with grad != 0.
Result: see invariance.out.
"""

import warnings

import cases
import numpy as np
import prototype

SEEDS = range(217500, 217550)
d = 5
lp0, g0, _ = cases.student(d=d)
sd = np.array([0.3, 2, 2, 2, 2.0])


def base_init(generator):
    return sd * generator.standard_normal(d)


wrong0 = cases.modify((lp0, g0, None), cases.flip(0))[1]
rs = np.random.default_rng(217500)
U, _, Vt = np.linalg.svd(rs.standard_normal((d, d)))
A = U @ np.diag(np.logspace(0, 2, d)) @ Vt
Ainv = np.linalg.inv(A)
b = np.array([3.0, -1, 4, 1, -5])


def affine(lp, g):
    return (
        (lambda y: lp((y - b) @ Ainv.T)),
        (lambda y: g((y - b) @ Ainv.T) @ Ainv),
        (lambda r: A @ base_init(r) + b),
    )


def scaled(lp, g, c, k=0.0):
    return (lambda x: c * lp(x) + k), (lambda x: c * np.asarray(g(x))), base_init


def compare(name, base, other, design, factor=1.0):
    differ, change = 0, 0.0
    for seed in SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v, w = design(*base, seed), design(*other, seed)
        differ += v.warned != w.warned
        if v.ratio > 1:
            change = max(change, abs(w.ratio / (factor * v.ratio) - 1))
    return f"{name:<58} decisions differ on {differ}/{len(SEEDS)} seeds; largest relative change of the ratio {change:.1e}"


if __name__ == "__main__":
    for label, g in (("wrong sign of coordinate 0", wrong0), ("correct gradient", g0)):
        base = (lp0, g, base_init)
        print(label)
        for key, design in (("N", prototype.N), ("H", prototype.H)):
            print(
                f"  ({key}) "
                + compare("x -> A x + b (condition 100)", base, affine(lp0, g), design)
            )
            print(
                f"  ({key}) "
                + compare("log_prob -> 8 log_prob", base, scaled(lp0, g, 8.0), design)
            )
            print(
                f"  ({key}) "
                + compare("log_prob -> 7.3 log_prob", base, scaled(lp0, g, 7.3), design)
            )
            print(
                f"  ({key}) "
                + compare("log_prob -> log_prob + 1e6", base, scaled(lp0, g, 1.0, 1e6), design)
            )
    print(
        "(H) under log_prob -> c log_prob on a small error: N(0,I) d=5 from N(0, I), grad + 0.03"
    )
    n5 = cases.gaussian(wide=1.0)
    small = cases.modify(n5, lambda x, g: g + 0.03)
    for c in (1.0, 10.0, 100.0):
        case = ((lambda x, c=c: c * small[0](x)), (lambda x, c=c: c * small[1](x)), small[2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(
                f"  c = {c:g}: (N) warned {sum(prototype.N(*case, s).warned for s in SEEDS)}/{len(SEEDS)}, (H) warned {sum(prototype.H(*case, s).warned for s in SEEDS)}/{len(SEEDS)}"
            )
    print(
        "log_prob exactly constant on the lines (flat top |x| < 10 with Gaussian shoulders, d = 3, from N(0, I))"
    )
    flat, _, init = cases.plateau()
    for name, g in (
        ("grad = 0 (correct)", lambda x: np.zeros(x.shape)),
        ("grad = -x (wrong)", lambda x: -x),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(
                f"  {name}: (N) warned {sum(prototype.N(flat, g, init, s).warned for s in SEEDS)}/{len(SEEDS)}, (H) warned {sum(prototype.H(flat, g, init, s).warned for s in SEEDS)}/{len(SEEDS)}"
            )

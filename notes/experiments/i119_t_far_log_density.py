"""The Student-t log-density of MultivariateT beyond the float64 range of distances (#119).

The squared distance s = |C^-1 (x - mu)|^2 overflows float64 beyond about 1.34e154
scale units, where log1p(s / nu) became inf and the Student-t log-density -inf, though
it is finite. The fix computes log s from the rows scaled by their largest coordinate
wherever s is not finite, after one whole-array test. The test points are 2-dimensional,
with 5 degrees of freedom. Two questions: whether the values agree with SciPy, and what the test
costs where no row is that far.

Values: SciPy's multivariate_t.logpdf overflows too, so the reference is its value at the
point moved in by a factor c, to s of about 1e40, minus (df + d) log c, which is exact to
within df / s. Cost: the code before the fix is copied below as `before`; it and the
fixed code are timed interleaved, 7 rounds each, on batches of n points from N(0, I),
alone and inside an Independence step on a standard normal target, and the minimum time
per call is compared. Their outputs are checked to be bitwise equal on those batches.

Result (i119_t_far_log_density.out): relative differences from the reference of at most
2e-16, at distances 1e155 and 1e300 for a scale matrix of unit size, and 1e5 and 1e160
for one of size 1e-300, where the code before the fix returned -inf or NaN. Near the
mean the outputs are bitwise equal. The fixed log_density costs about 2 us more per call
(1.8 to 2.6 us for n up to 16, ratios 1.26 to 1.32; ratios 0.96 to 1.13 for n of 256
and 4,096), from np.errstate, which keeps NumPy's overflow warning from announcing a
value that is then computed correctly, and the whole-array test. A whole step costs
1.02 to 1.08 times as much. The machine was shared with other jobs, and repeated runs
scatter by several percent.

Run: OMP_NUM_THREADS=1 python notes/experiments/i119_t_far_log_density.py
"""

import math
import platform
import timeit
from functools import partial

import numpy as np
import scipy
from scipy import stats
from scipy.special import gammaln

from telescopic._linalg import matvec
from telescopic.kernels import Independence, MultivariateT
from telescopic.target import Target
from telescopic.testing import random_draws


def before(q, x):
    d = len(q.mean)
    z = matvec(q._inverse, x - q.mean)
    squared = (z**2).sum(axis=1)
    nu = q.df
    return (
        gammaln((nu + d) / 2)
        - gammaln(nu / 2)
        - 0.5 * (d * math.log(nu * math.pi) + q._log_det)
        - 0.5 * (nu + d) * np.log1p(squared / nu)
    )


print(f"numpy {np.__version__}, scipy {scipy.__version__}, {platform.machine()}")
df = 5
for size, distance in [(1.0, 1e155), (1.0, 1e300), (1e-300, 1e5), (1e-300, 1e160)]:
    cov = size * np.array([[2.0, 0.3], [0.3, 0.5]])
    q = MultivariateT(np.zeros(2), cov, df)
    points = np.array([[1.0, 0.5], [-0.3, 1.0]]) * distance
    c = distance * 1e-20 / math.sqrt(size)
    reference = stats.multivariate_t(np.zeros(2), cov, df=df).logpdf(points / c)
    reference -= (df + 2) * math.log(c)
    with np.errstate(all="ignore"):
        old = before(q, points)
    error = np.max(np.abs(q.log_density(points) / reference - 1))
    print(
        f"scale {size:g}, distance {distance:g}: reference {np.round(reference, 3)}, "
        f"before {old}, relative error {error:.1g}"
    )


class Before(MultivariateT):
    def log_density(self, x):
        return before(self, x)


def timed(functions, number=2000, rounds=7):
    # The minimum time per call in microseconds of each function, timed interleaved.
    times = [[] for _ in functions]
    for _ in range(rounds):
        for f, row in zip(functions, times, strict=True):
            row.append(timeit.timeit(f, number=number) / number * 1e6)
    return [min(row) for row in times]


rng = np.random.default_rng(0)
target = Target(lambda x: -0.5 * (x**2).sum(axis=1), vectorize=True)
for d in (2, 10):
    args = (np.zeros(d), np.eye(d) + 0.1, df)
    q, q_before = MultivariateT(*args), Before(*args)
    for n in (1, 16, 256, 4096):
        x = rng.standard_normal((n, d))
        assert np.array_equal(q.log_density(x), before(q, x))
        old, new = timed([partial(q_before.log_density, x), partial(q.log_density, x)])
        kernels = [Independence(q_before), Independence(q)]
        states = kernels[1].init_state(target, x)
        draws = [random_draws(n, 0) for _ in kernels]
        step_old, step_new = timed(
            [
                partial(k.step, target, states, draw)
                for k, draw in zip(kernels, draws, strict=True)
            ],
            number=200,
        )
        print(
            f"d {d}, n {n}: log_density {old:.1f} us before, {new:.1f} us after, ratio "
            f"{new / old:.2f}; step {step_old:.1f} us before, {step_new:.1f} us after, "
            f"ratio {step_new / step_old:.2f}"
        )

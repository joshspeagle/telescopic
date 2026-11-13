"""The gradient check's tolerance on correct and wrong gradients (issue #73, #82 ruling 5).

Question: does the check that `CoupledSampler` runs at construction stay silent on stiff
but correct gradients, and warn on wrong ones? At 3 draws of the initial distribution, it
compares `grad` with the central difference D_h of `log_prob`, and warns where
|grad - D_h| exceeds its tolerance: 10 times the estimated error of D_h, the truncation
|D_h - D_2h| plus the rounding sigma / h, with sigma estimated from fourth differences and
at least 64 units in the last place of |log_prob|; and at least 1% of the largest |D_h|.
For each case and 200 seeds, the table gives the largest |grad - D_h| / tolerance over
the coordinates and points checked, as quantiles over the seeds (the check warns above
1), and the number of seeds whose check warned. Each case runs the sampler's own check,
through `CoupledSampler` with the kernel MALA.

Result (in the .out file beside this script): on every correct gradient, scales from
1e-8 to 1e8, condition numbers to 1e12, a mean of 1e6 at scale 1, a log-density offset
by 1e9 or computed as a difference of terms of size 1e8, curved, funnel, heavy-tailed,
kinked and bounded targets, the pumps posterior and a logistic regression on 10^4
points, no seed warns, with a largest ratio of 0.33. A log-density computed as a
difference of terms of size 1e12 changes by less than its rounding over the check's
steps, and warns wrongly on 26 of 200 seeds in one dimension and 4 in two. Without the
floor of 1%, the error model alone warns wrongly on a log-density computed as a
difference of terms of size 1e4 or 1e8, on 7 and 16 of 200 seeds in one dimension and 0
and 1 in two (the last section). Every
gradient of the wrong sign, twice too large or wrong in one coordinate warns on every
seed; a logistic gradient without its prior's term, wrong by 0.01% to 0.02% of its
largest component, warns on none, by the tolerance's floor of 1%.

Run: python notes/experiments/i73_gradient_check.py
"""

import platform
import warnings

import numpy as np
import scipy
from scipy import special, stats

from telescopic import CoupledSampler, examples, sampler
from telescopic.initial import Normal
from telescopic.warnings import InputWarning

SEEDS = 200
ratios: list[float] = []
original = sampler._gradient_comparison


def recording(target, x, step):
    # The sampler's own comparison, with its largest |grad - D_h| / tolerance recorded.
    compared = original(target, x, step)
    if compared is not None:
        grad, differences, tolerance = compared
        ratios.append(float(np.nanmax(np.abs(grad - differences) / tolerance)))
    return compared


sampler._gradient_comparison = recording


def gaussian(scale, d=5, mean=0.0, offset=0.0, wide=2.0):
    def log_prob(x):
        return -0.5 * np.sum(((x - mean) / scale) ** 2, axis=-1) + offset

    def grad(x):
        return -(x - mean) / scale**2

    return log_prob, grad, Normal(np.full(d, mean), np.full(d, (wide * scale) ** 2))


def conditioned(condition, d=10):
    # A rotated Gaussian whose standard deviations span 1 to 1/sqrt(condition).
    rotation = stats.ortho_group.rvs(d, random_state=0)
    scales = np.logspace(0, -0.5 * np.log10(condition), d)
    precision = rotation @ np.diag(scales**-2) @ rotation.T

    def log_prob(x):
        return -0.5 * np.einsum("...i,ij,...j->...", x, precision, x)

    def grad(x):
        return -x @ precision

    cov = rotation @ np.diag(scales**2) @ rotation.T
    return log_prob, grad, Normal(np.zeros(d), 4 * cov)


def cancelling(c, d=1):
    # -|x|^2 / 2, computed as a difference of terms of size c^2.
    def log_prob(x):
        return -0.5 * np.sum((x + c) ** 2 - 2 * c * x - c**2, axis=-1)

    return log_prob, lambda x: -x, Normal(np.zeros(d), np.full(d, 4.0))


def student(df=3.0, d=5):
    def log_prob(x):
        return -0.5 * (df + d) * np.log1p(np.sum(x * x, axis=-1) / df)

    def grad(x):
        return -(df + d) * x / (df + np.sum(x * x, axis=-1, keepdims=True))

    return log_prob, grad, Normal(np.zeros(d), np.full(d, 100.0))


def laplace(d=5):
    return (
        lambda x: -np.sum(np.abs(x), axis=-1),
        lambda x: -np.sign(x),
        Normal(np.zeros(d), np.full(d, 4.0)),
    )


def half_normal(d=3):
    # Zero density where x[0] < 0, from a start that straddles the boundary.
    def log_prob(x):
        return np.where(x[..., 0] > 0, -0.5 * np.sum(x * x, axis=-1), -np.inf)

    return log_prob, lambda x: -x, Normal(np.zeros(d), np.ones(d))


def logistic(n=10_000, d=5):
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n, d))
    labels = rng.random(n) < special.expit(features @ np.linspace(-1, 1, d))
    signs = np.where(labels, 1.0, -1.0)

    def log_prob(beta):
        margins = signs * (features @ beta.T).T
        return -np.sum(np.logaddexp(0, -margins), axis=-1) - 0.5 * np.sum(beta**2, axis=-1)

    def grad(beta):
        margins = signs * (features @ beta.T).T
        weights = signs * special.expit(-margins)
        return weights @ features - beta

    return log_prob, grad, Normal(np.zeros(d), np.full(d, 0.01))


def example(name):
    target = getattr(examples, name)()
    return target.log_prob, target.grad, target.initial


def wrong_sign(case):
    log_prob, grad, initial = case
    return log_prob, lambda x: -grad(x), initial


def doubled(case):
    log_prob, grad, initial = case
    return log_prob, lambda x: 2 * grad(x), initial


def one_coordinate(case):
    log_prob, grad, initial = case

    def wrong(x):
        g = np.array(grad(x), dtype=float)
        g[..., 0] = -g[..., 0]
        return g

    return log_prob, wrong, initial


def without_prior(case):
    # The logistic gradient without the prior's -beta.
    log_prob, grad, initial = case
    return log_prob, lambda x: grad(x) + x, initial


correct = {
    "N(0, 1e-8^2 I), d = 5": gaussian(1e-8),
    "N(0, 1e-4^2 I), d = 5": gaussian(1e-4),
    "N(0, I), d = 5": gaussian(1.0),
    "N(0, 1e4^2 I), d = 5": gaussian(1e4),
    "N(0, 1e8^2 I), d = 5": gaussian(1e8),
    "N(0, I) from 100x wider": gaussian(1.0, wide=100.0),
    "N(1e6, I), d = 5": gaussian(1.0, mean=1e6),
    "N(0, I) + 1e9, d = 5": gaussian(1.0, offset=1e9),
    "condition 1e4, d = 10": conditioned(1e4),
    "condition 1e8, d = 10": conditioned(1e8),
    "condition 1e12, d = 10": conditioned(1e12),
    "cancelling terms of 1e8, d = 1": cancelling(1e4),
    "cancelling terms of 1e12, d = 1": cancelling(1e6),
    "cancelling terms of 1e12, d = 2": cancelling(1e6, d=2),
    "examples.banana()": example("banana"),
    "examples.funnel(10)": example("funnel"),
    "examples.mixture()": example("mixture"),
    "examples.pumps()": example("pumps"),
    "Student t_3, d = 5": student(),
    "Laplace, d = 5": laplace(),
    "half-normal, d = 3": half_normal(),
    "logistic, n = 10^4, d = 5": logistic(),
}
wrong = {
    "N(0, I): wrong sign": wrong_sign(gaussian(1.0)),
    "N(0, 1e8^2 I): wrong sign": wrong_sign(gaussian(1e8)),
    "N(0, 1e-8^2 I): twice": doubled(gaussian(1e-8)),
    "condition 1e8: coordinate 0 wrong": one_coordinate(conditioned(1e8)),
    "banana: coordinate 0 wrong": one_coordinate(example("banana")),
    "funnel: twice": doubled(example("funnel")),
    "logistic: without the prior": without_prior(logistic()),
}


def check(title, cases):
    print(title)
    for name, (log_prob, grad, initial) in cases.items():
        largest, warned = [], 0
        for seed in range(SEEDS):
            ratios.clear()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                CoupledSampler(
                    log_prob, initial, grad=grad, vectorize=True, kernel="mala", rng=seed
                )
            warned += any(issubclass(w.category, InputWarning) for w in caught)
            largest.append(max(ratios, default=0.0))
        q = np.quantile(largest, [0.5, 0.99, 1.0])
        print(f"  {name:<34} {q[0]:>9.3g} {q[1]:>9.3g} {q[2]:>9.3g} {warned:>4}/{SEEDS}")


print(f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
print(f"{SEEDS} seeds per case; largest |grad - D_h| / tolerance per seed; warns above 1")
print(f"{'case':<36} {'median':>9} {'99%':>9} {'max':>9} {'warned':>7}")
check("correct gradients", correct)
check("wrong gradients", wrong)
# The error model alone, without the tolerance's floor of 1% of the largest |D_h|.
sampler.GRADIENT_RELATIVE = 0.0
check(
    "correct gradients, the error model alone",
    {
        f"cancelling terms of 1e{2 * e}, d = {d}": cancelling(10.0**e, d)
        for e in (1, 2, 4)
        for d in (1, 2)
    },
)

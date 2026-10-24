"""MALA and HMC on a mixture whose gradient is NaN far in the tails (audit 2).

0.5 N(-1.5, 1) + 0.5 N(1.5, 1), its log-density by logsumexp and its gradient by the
textbook responsibilities, ratios of densities, which are NaN where both densities
underflow, about 38.6 standard deviations from either mean. `sample()` with MALA() and
HMC(), tuned by default, 100 replicates, seeds 0 to 4: whether the run completes, the z
of E[x] (exactly 0) and of E[x^2] (3.25), and the gradient rows that were not finite where
the log-density is, with the largest log-density among them. Before DECISIONS #59 the
kernels rejected proposals with such a gradient, under #59 they raised, and under #67
they take it as 0.

Run: python notes/experiments/audit2_gradient_mixture.py
"""

import warnings

import numpy as np
from scipy import special, stats

import telescopic
from telescopic.initial import Normal
from telescopic.kernels import HMC, MALA

MEANS = np.array([-1.5, 1.5])
bad, worst = [], []


def log_prob(x):
    components = stats.norm.logpdf(x[:, :1], MEANS, 1.0)
    return special.logsumexp(components, axis=1) + np.log(0.5)


def grad(x):
    densities = stats.norm.pdf(x[:, :1], MEANS, 1.0)
    with np.errstate(invalid="ignore"):
        responsibilities = densities / densities.sum(axis=1, keepdims=True)
    gradient = np.sum(responsibilities * (MEANS - x[:, :1]), axis=1, keepdims=True)
    rows = ~np.isfinite(gradient[:, 0])
    if rows.any():
        bad.append(int(rows.sum()))
        worst.append(float(np.max(log_prob(x[rows]))))
    return gradient


for name, make in (("MALA", MALA), ("HMC", HMC)):
    for seed in range(5):
        bad.clear()
        worst.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = telescopic.sample(
                    log_prob,
                    Normal([0.0], [[4.0]]),
                    100,
                    grad=grad,
                    vectorize=True,
                    kernel=make(),
                    rng=seed,
                )
            except ValueError as error:
                print(f"{name}, seed {seed}: raised ValueError: {str(error)[:70]}")
                continue
        mean = result.estimate(lambda x: x[:, 0])
        square = result.estimate(lambda x: x[:, 0] ** 2)
        warned = any("taken as 0" in str(w.message) for w in caught)
        largest = f", largest log pi there {max(worst):.4g}" if worst else ""
        print(
            f"{name}, seed {seed}: z of E[x] {mean.value / mean.standard_error:+.2f}, "
            f"of E[x^2] {(square.value - 3.25) / square.standard_error:+.2f}; "
            f"{sum(bad)} gradient rows not finite{largest}; warned: {warned}"
        )

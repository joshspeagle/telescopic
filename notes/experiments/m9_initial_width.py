"""How the width of the initial distribution affects the line fit of the README (M9).

The spec's README example (section 8.3) starts from N(0, 10 I). Its posterior has standard
deviations 0.012 and 0.066 and lies 70 of them from 0, so most runs had one replicate
holding most of the variance, and some estimated a negative variance: a replicate whose
chain X is still on its way at k is averaged without correction, balanced only in
expectation by rarer replicates whose chain Y is late. On the README's data, 20 seeds of
`telescopic.sample` (vectorised log-density, R = 100) for each initial distribution: the
spec's, Gaussians centred on the exact posterior mean with covariance c^2 times the exact
one for c = 2, 10 and 100, and N(fit, 100 H^-1) from BFGS's fit and inverse-Hessian
approximation started at 0. Each line gives the largest share of one replicate in the
variance of the first two moments (the warning fires above 0.25), the estimated standard
deviations over the exact ones (signed: a negative variance gives a negative ratio), k
and the seconds per run.

Run: python notes/experiments/m9_initial_width.py
"""

import time
import warnings

import numpy as np
from scipy import optimize

import telescopic
from telescopic import diagnostics

rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)


def log_prob(theta, x, y, yerr):
    return -0.5 * np.sum(((y - theta[:, :1] * x - theta[:, 1:]) / yerr) ** 2, axis=1)


design = np.vander(x, 2)
cov = np.linalg.inv(design.T @ (design / yerr[:, None] ** 2))
mean = cov @ (design.T @ (y / yerr**2))
fit = optimize.minimize(lambda t: -log_prob(t[None], x, y, yerr)[0], np.zeros(2))
print(
    f"exact mean {mean}, sd {np.sqrt(np.diag(cov))}; BFGS fit {fit.x}, sd {np.sqrt(np.diag(fit.hess_inv))}"
)
initials = {
    "N(0, 10 I)": telescopic.initial.Normal(np.zeros(2), 10 * np.eye(2)),
    "N(mean, 2^2 cov)": telescopic.initial.Normal(mean, 4 * cov),
    "N(mean, 10^2 cov)": telescopic.initial.Normal(mean, 100 * cov),
    "N(mean, 100^2 cov)": telescopic.initial.Normal(mean, 1e4 * cov),
    "N(fit, 100 H^-1)": telescopic.initial.Normal(fit.x, 100 * fit.hess_inv),
}
for label, initial in initials.items():
    shares, ratios, ks, start = [], [], [], time.perf_counter()
    for seed in range(20):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = telescopic.sample(
                log_prob, initial, args=(x, y, yerr), vectorize=True, rng=seed
            )
        moments = [result.measure.per_replicate(h) for h in (lambda t: t, np.square)]
        shares.append(np.max(diagnostics.variance_share(np.concatenate(moments, axis=1))))
        variance = result.var()
        ratios.append(np.sign(variance) * np.sqrt(np.abs(variance) / np.diag(cov)))
        ks.append(result.k)
    shares, ratios = np.array(shares), np.array(ratios)
    print(
        f"{label}: largest share median {np.median(shares):.2f}, 90% {np.quantile(shares, 0.9):.2f}, "
        f"above 0.25 in {np.sum(shares > 0.25)} of 20; sd ratios from {ratios.min(axis=0).round(2)} "
        f"to {ratios.max(axis=0).round(2)}; k median {np.median(ks):.0f}; "
        f"{(time.perf_counter() - start) / 20:.2f} s per run",
        flush=True,
    )

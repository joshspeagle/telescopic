"""Whether the variance-share warning marks unreliable intervals on the banana (M9, D-16).

The warning study found the warning "one replicate holds more than 25% of the variance"
firing in 38 of 100 clean banana runs at R = 100. On the banana from U[-5, 5]^2 with the
automatic pipeline, 200 seeds at R = 100: the coverage of the 95% intervals of the means
and second moments of both coordinates, in the runs whose largest share of one replicate
in the variance of those estimates is above and below 0.25 and 0.5.

Run: python notes/experiments/m9_variance_warning.py
"""

import warnings

import numpy as np

import telescopic
from telescopic import diagnostics, examples

banana = examples.banana()
truth = np.concatenate([banana.mean, np.diag(banana.cov) + banana.mean**2])
covered, shares = [], []
for seed in range(200):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = telescopic.sample(banana.log_prob, banana.initial, 100, vectorize=True, rng=seed)
    estimate = result.estimate(lambda x: np.concatenate([x, x**2], axis=1))
    low, high = estimate.interval
    covered.append((low <= truth) & (truth <= high))
    shares.append(np.max(diagnostics.variance_share(estimate.replicates)))
covered, shares = np.array(covered), np.array(shares)
print(f"all 200 runs: coverage of E[x0], E[x1], E[x0^2], E[x1^2] {covered.mean(axis=0).round(3)}")
print(
    f"largest variance share: median {np.median(shares):.3f}, 90% {np.quantile(shares, 0.9):.3f}"
)
for threshold in (0.25, 0.5):
    above = shares > threshold
    print(
        f"share above {threshold}: {above.sum()} runs, coverage {covered[above].mean(axis=0).round(3)}; "
        f"below: coverage {covered[~above].mean(axis=0).round(3)}",
        flush=True,
    )

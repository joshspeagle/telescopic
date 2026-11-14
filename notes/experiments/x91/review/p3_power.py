# U: chance that the pooled criterion (more than 30 of 3,000 unmet) fails when a fraction p of the 30
# tunings fails (k not chosen), drawing each failing seed's unmet count from the six counts observed at
# the package's cap of 100,000 (issue #137 seed 0: 31; i137_default_cap.out: 14, 96;
# i137_reproduce_unequal.out: 54, 23, 12). Compared with the registration's 1 - (1 - p)^30.
# V: size and power of the registered bootstrap rule with lognormal seeds (not 5 resampled values).
# Z: family-wise false failure rate.
import math

import numpy as np
from scipy import stats

rng = np.random.default_rng(1)
obs = np.array([31, 14, 96, 54, 23, 12])
print(
    "U: p, registration's 1-(1-p)^30, P(pooled unmet > 30), P(k not chosen at >= 1 seed and U still holds)"
)
for p in (0.01, 0.02, 0.05, 0.10):
    N = rng.binomial(30, p, 200000)
    tot = np.array([rng.choice(obs, n).sum() if n else 0 for n in N])
    print(
        f"  {p:.2f}  {1 - (1 - p) ** 30:.3f}  {(tot > 30).mean():.3f}  {((N > 0) & (tot <= 30)).mean():.3f}"
    )


def vfail(a, p, B=2000):
    ia = rng.integers(0, len(a), (B, len(a)))
    ip = rng.integers(0, len(p), (B, len(p)))
    return np.quantile(np.median(a[ia], axis=1) - np.median(p[ip], axis=1), 0.05) > math.log(1.2)


print(
    "V: P(fail) at true ratios of medians 1.0, 1.2, 1.3, 2, 4; lognormal seeds, sd(log) of P / arm; S seeds; 300 experiments"
)
for lab, sp, sa, S in [
    ("banana P 2.6 / arm 2.6", 2.6, 2.6, 20),
    ("banana P 2.6 / J5 0.39", 2.6, 0.39, 20),
    ("banana robust P 0.9 / arm 0.4", 0.9, 0.4, 20),
    ("banana default P 3.8 / arm 0.24, S=10", 3.8, 0.24, 10),
    ("mixture 0.037 / 0.074", 0.037, 0.074, 20),
]:
    print(
        "  ",
        lab,
        [
            round(
                np.mean(
                    [
                        vfail(rng.normal(math.log(r), sa, S), rng.normal(0, sp, S))
                        for _ in range(300)
                    ]
                ),
                3,
            )
            for r in (1.0, 1.2, 1.3, 2.0, 4.0)
        ],
    )
print(
    "M: power at ratio 1.3 on the banana n=2 when the sds (0.10, 0.08, each from 5 seeds) are at the ends of their 95% intervals"
)
for f in (math.sqrt(4 / stats.chi2.ppf(0.975, 4)), 1.0, math.sqrt(4 / stats.chi2.ppf(0.025, 4))):
    se = f * math.sqrt((0.10**2 + 0.08**2) / 20)
    print(
        f"   sd x {f:.2f}: se {se:.3f} power at 1.3 {stats.norm.cdf(math.log(1.3 / 1.2) / se):.3f}, false failure at 1.1 {1 - stats.norm.cdf(math.log(1.2 / 1.1) / se):.3f}"
    )
print(
    f"Z: two-sided P(|z|>3.5) = {2 * stats.norm.sf(3.5):.2e}; family-wise over 130 independent normal scores {1 - (1 - 2 * stats.norm.sf(3.5)) ** 130:.3f}; over 84 (part C) {1 - (1 - 2 * stats.norm.sf(3.5)) ** 84:.3f}"
)
print(
    "   same with Student t, 19 df (20 seeds, if the pooled error were estimated between seeds): %.3f"
    % (1 - (1 - 2 * stats.t.sf(3.5, 19)) ** 130)
)
print(
    "   P(P's own |z| > 3.5 somewhere on a 6-function cell, voiding it) %.4f"
    % (1 - (1 - 2 * stats.norm.sf(3.5)) ** 6)
)

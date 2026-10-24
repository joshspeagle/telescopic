"""The rejection maximal coupling at draws where a log-density is infinite (audit 2).

A draw of Gamma(0.01) underflows to 0.0 with probability about 6e-4: the sampler puts the
mass below the smallest double at one point, where the log-density is -inf (the gamma
class of `telescopic.examples`) or +inf (SciPy's, a pole). For each pair of laws, the
fraction of coupled draws Y equal to 0.0 against that of direct draws from q, 100,000
each, and for Dirichlet laws, whose SciPy draws have shape (1, k), the meeting rate
(exactly 1/2) and the mean of Y's first coordinate (exactly 2/3), 20,000 pairs.
DECISIONS #60 changed the coupling at such draws.

Run: python notes/experiments/audit2_maximal_lumps.py
"""

import numpy as np
from scipy import stats

from telescopic.couplings import maximal
from telescopic.examples import _Gamma
from telescopic.testing import random_draws

n = 100_000
pairs = {
    "Gamma(0.01) to Gamma(1.01), -inf at 0": (_Gamma(0.01, 1.0), _Gamma(1.01, 1.0)),
    "Gamma(1.01) to Gamma(0.01), -inf at 0": (_Gamma(1.01, 1.0), _Gamma(0.01, 1.0)),
    "Beta(0.01, 2) to Beta(0.02, 2), +inf at 0": (stats.beta(0.01, 2), stats.beta(0.02, 2)),
}
for name, (p, q) in pairs.items():
    generator = np.random.default_rng(1)
    direct = np.array([q.rvs(random_state=generator) for _ in range(n)])
    try:
        _, y, _, _ = maximal([p] * n, [q] * n, random_draws(n, 2))
    except ValueError as error:
        print(f"{name}: raised ValueError: {error}")
        continue
    coupled, alone = np.mean(y == 0), np.mean(direct == 0)
    print(f"{name}: Y = 0.0 in {coupled:.2e} of coupled draws, {alone:.2e} of direct ones")

m = 20_000
p, q = stats.dirichlet([1.0, 2.0]), stats.dirichlet([2.0, 1.0])
_, y, identical, _ = maximal([p] * m, [q] * m, random_draws(m, 3))
first = y[:, 0, 0]
print(
    f"Dir(1, 2) to Dir(2, 1): meeting rate {identical.mean():.4f}, "
    f"mean of Y's first coordinate {first.mean():.4f} +- {first.std() / np.sqrt(m):.4f}"
)

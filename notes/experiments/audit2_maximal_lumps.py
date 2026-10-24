"""The rejection maximal coupling at draws where a log-density is infinite (audit 2).

A draw of Gamma(0.01) underflows to 0.0 with probability about 6e-4: the sampler puts the
mass below the smallest double at one point, where the log-density is -inf (the gamma
class of `telescopic.examples`) or +inf (SciPy's, a pole). For each pair of laws, the
fraction of coupled draws Y equal to 0.0 against that of direct draws from q, 100,000
each. For Dirichlet laws, whose SciPy draws have shape (1, k): the meeting rate (exactly
1/2) and the mean of Y's first coordinate (exactly 2/3), 20,000 pairs; and, where a
component underflows to 0 at a pole, the fraction of Y whose first coordinate is 0,
coupled and direct, 1,000,000 each. Last, how frequent such draws can be: Beta(0.01, 0.01)
puts its mass within 2^-53 of 1 at 1.0, a pole; against Beta(0.011, 0.01), 20,000 pairs,
the meeting rate and the fraction of equal pairs, and the overlap of the densities,
E_p[min(1, q/p)] over 1,000,000 draws of p with q/p = B(0.01, 0.01) / B(0.011, 0.01)
x^0.001, so that the draws at 1.0 take its limit, and those at 0.0, 3e-4 of them, 0.
DECISIONS #60 changed the coupling at such draws.

Run: python notes/experiments/audit2_maximal_lumps.py
"""

import numpy as np
from scipy import special, stats

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

name = "Dir(0.001, 0.001, 1) to Dir(0.002, 0.001, 1), +inf at 0"
p, q = stats.dirichlet([0.001, 0.001, 1.0]), stats.dirichlet([0.002, 0.001, 1.0])
big = 1_000_000
try:
    _, y, _, _ = maximal([p] * big, [q] * big, random_draws(big, 5))
    direct = random_draws(big, 4)
    alone = np.mean([q.rvs(random_state=direct.generator(row))[0, 0] == 0 for row in range(big)])
    coupled = np.mean(y[:, 0, 0] == 0)
    print(
        f"{name}: Y_1 = 0 in {coupled:.4f} of coupled draws, {alone:.4f} of direct ones, "
        f"+- {np.sqrt(alone * (1 - alone) / big):.4f} each"
    )
except ValueError as error:
    print(f"{name}: raised ValueError: {error}")

name = "Beta(0.01, 0.01) to Beta(0.011, 0.01), +inf at 0 and 1"
p, q = stats.beta(0.01, 0.01), stats.beta(0.011, 0.01)
draws = p.rvs(size=1_000_000, random_state=np.random.default_rng(6))
with np.errstate(divide="ignore"):
    ratio = np.exp(
        special.betaln(0.01, 0.01) - special.betaln(0.011, 0.01) + 0.001 * np.log(draws)
    )
overlap = np.minimum(1.0, ratio).mean()
try:
    x, y, identical, _ = maximal([p] * m, [q] * m, random_draws(m, 7))
    print(
        f"{name}: {np.mean(draws == 1.0):.3f} of p's draws at 1.0; overlap {overlap:.3f}, "
        f"meeting rate {identical.mean():.3f}, equal pairs {np.mean(x == y):.3f}"
    )
except ValueError as error:
    print(f"{name}: raised ValueError: {error}")

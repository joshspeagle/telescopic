# telescopic

*Unbiased Markov chain Monte Carlo with coupled chains and telescoping sums.*

telescopic runs pairs of coupled Markov chains until they meet, and turns each pair into
an exactly unbiased estimate of expectations under the target distribution (Glynn & Rhee
2014; Jacob, O'Leary & Atchadé 2020). Averages over independent pairs come with error
bars that are valid as the number of pairs grows, not as the chains grow, and the pairs
run in parallel without communication. Their meeting times also bound the distance of
the chains from stationarity.

telescopic is written in Python on NumPy and SciPy. It is in development; version 0.1 is
a first draft.

## Installation

From a clone of the repository:

```bash
pip install .            # or pip install ".[all]" for every extra
```

It needs Python 3.11 or newer, NumPy 1.26 or newer and SciPy 1.12 or newer. The extras
are `plot` (matplotlib), `progress` (tqdm), `parallel` (cloudpickle, which sends
functions defined in notebooks to worker processes), `pandas` (summaries as data
frames) and `all`.

## Example

A straight line fitted to data with Gaussian errors:

```python
import numpy as np
from scipy import optimize

import telescopic

rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)

def log_prob(theta, x, y, yerr):
    slope, intercept = theta
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2)

# Start the chains from a Gaussian ten times wider than the best fit's errors.
fit = optimize.minimize(lambda theta: -log_prob(theta, x, y, yerr), [0.0, 0.0])
init = telescopic.initial.Normal(mean=fit.x, cov=100 * fit.hess_inv)
result = telescopic.sample(log_prob, init, args=(x, y, yerr), names=["slope", "intercept"],
                           rng=42)
print(result.summary())                      # unbiased means, standard errors, quantiles
result.estimate(lambda t: t[:, 0] * t[:, 1])  # any function, after the run
result.plot_tv_bound()                       # with the plot extra: distance from stationarity
```

`sample` tunes a random-walk kernel on pilot chains, chooses the burn-in, the lag and the
length of each run from the meeting times of pilot pairs, runs 100 replicates, and warns
when the run shows signs that its error bars are unreliable. The chains should start
wider than the posterior, but not by orders of magnitude: from a prior a hundred times
wider, a replicate still on its way at the burn-in can dominate the estimates of
variances.

## What it guarantees

Each replicate's estimate is unbiased for any burn-in, lag, run length and initial
distribution, provided the coupling keeps each chain's law, the chains stay together
once they meet, and the meeting time has enough moments. With finite variance, the error
bars are valid as the number of replicates grows. Coupling does not make the chains mix
faster, and a run whose initial distribution misses a mode can be confidently wrong. The
[theory page](docs/theory.md) states the conditions and the limits.

## Citing

If you use telescopic, please cite the software ([`CITATION.cff`](CITATION.cff)) and the
methods it implements:

- Glynn, P. W. & Rhee, C.-H. (2014). Exact estimation for Markov chain equilibrium
  expectations. *Journal of Applied Probability* 51A, 377–389.
- Jacob, P. E., O'Leary, J. & Atchadé, Y. F. (2020). Unbiased Markov chain Monte Carlo
  methods with couplings. *Journal of the Royal Statistical Society B* 82(3), 543–600.

## Licence

MIT.

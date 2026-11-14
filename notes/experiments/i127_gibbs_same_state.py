"""Coupled Gibbs blocks from generators in the same state against the normal map (issue #127).

Question: under `Gibbs(coupling="crn")`, a block of several coordinates, or of one without
`ppf`, was drawn by the normal map: common standard normals through the Cholesky factor
of each chain's conditional covariance. The change draws each chain from its own
conditional with generators in the same state. Do normal blocks then contract as they did,
and do pairs meet as soon?

Method. Part 1 measures the distance between the two chains' draws of N(0, C) and
N(0, C'), where C' is C scaled by 1 + e or C^(1/2) (I + e S) C^(1/2) for a random
symmetric S of spectral norm 1, for e = 1%, 10% and 50%, under the normal map, under
SciPy's frozen `multivariate_normal` drawn from generators in the same state ("same
state"), under NumPy's `Generator.multivariate_normal` with `method="cholesky"` drawn
from generators in the same state ("Cholesky"), and for independent draws, over 200
random covariances C of size 3 (Wishart with 6 degrees of freedom) and one
near-isotropic one, I + 0.01 W, with 200 draws each; "flipped" is the fraction of the C
whose SciPy draws are on average 10 times farther apart than the
normal map's. Part 2 measures the pilot meeting times of 100 pairs per seed for 5 seeds,
under "crn" and "two-scale" (threshold 1, or the square root of the trace of the
posterior covariance for the pumps), of the Gibbs samplers of four targets:
- the package's Gibbs example, `examples.pumps`, whose blocks have one coordinate and
  `ppf`, which the change leaves alone ("ppf"), and the same with conditionals without
  `ppf`, which the package refused before the change ("no ppf");
- a Gaussian target of 4 coordinates in two blocks of 2, whose conditional covariances
  do not depend on the state;
- a robust regression, y_i ~ N(x_i b, s^2 / l_i) with l_i ~ Gamma(3/2, rate 3/2), 20
  observations and 3 coefficients, whose block b has the normal conditional
  N(A^-1 X' L y / s^2, A^-1) with A = X' L X / s^2 + I / 100, a covariance that differs
  between chains whose l differ;
- three normal means with unknown precisions, whose block of means has a diagonal
  conditional covariance whose variances are close and differ between the chains.
"map" draws a normal block with a class whose `rvs` is mean + L z, with L the Cholesky
factor and z standard normals from the generator: from generators in the same state it
is the normal map that the package applied before the change, coupling for coupling.
"SciPy" draws it with SciPy's frozen `multivariate_normal`, as a user would write it,
which the package couples as its commit does: from generators in the same state after
the change, and by the normal map before it and under the alternative that keeps the map
for SciPy's multivariate normal. SciPy's `rvs` calls NumPy's
`Generator.multivariate_normal` with its default method, a singular value decomposition.
"NumPy", for the robust regression only, draws it with a class of the user's own whose
`rvs` calls that sampler with `method="cholesky"`. A row prints, for each seed, the
median meeting time and the number of pairs censored at 2000 sweeps, then the mean and
standard deviation of the 5 medians, and the time of a sweep relative to the first row
of its target. Such ratios of times vary between runs on a shared machine: the output at
commit 970e3c8 and the one committed with 3e300e4 give 4.54 and 5.42 for the pumps
without `ppf` under "crn", and differ by factors of up to 1.48 on the other rows. The
last line gives the processor time of the run.

Result (i127_gibbs_same_state.out, at the change): from generators in the same state,
SciPy's draws are as close as the normal map's for covariances that differ by a scale
factor, but a change of shape of 1% leaves them 0.063 apart on average, against 0.0056
for the map and 2.2 for independent draws, as 4% of the covariances flip a sign or an
order of the decomposition, and near the identity they are as far apart as independent
draws. NumPy's sampler with `method="cholesky"` gives draws as close as the normal map's
in every row, to the 4 decimals printed (0.0056 at a change of shape of 1%). Pairs of
the robust regression meet later under "crn", at a median of 205.4 sweeps (sd 18.5 over
the seeds) against 147.1 (sd 10.3) under the map, and as fast under "two-scale" (4.0
against 3.7); with NumPy's Cholesky method they meet at the map's medians, seed for seed
(147.1 and 3.7). The other targets meet as before, and the pumps without `ppf` meet as
with it. Under the alternative (i127_gibbs_same_state_alternative.out), the robust
regression's pairs with SciPy's conditionals meet under "crn" at a median of 142.9
sweeps (sd 2.7), as under the map. Run as
`python notes/experiments/i127_gibbs_same_state.py`.
"""

import time

import numpy as np
from i85_setup import setup
from numpy.random import PCG64DXSM, Generator, SeedSequence
from scipy import linalg, stats

import telescopic
from telescopic import examples
from telescopic.examples import _Gamma as Gamma
from telescopic.initial import Uniform
from telescopic.kernels import Gibbs

SEEDS = (1, 2, 3, 4, 5)
PAIRS = 100
MAX_ITER = 2000


class CholeskyNormal:
    # The normal map of the package before the change, as a conditional of the user's own.
    def __init__(self, mean, cov):
        self.mean, self.chol = mean, np.linalg.cholesky(cov)

    def rvs(self, *, random_state):
        return self.mean + self.chol @ random_state.standard_normal(len(self.mean))

    def logpdf(self, value):
        z = linalg.solve_triangular(self.chol, value - self.mean, lower=True)
        return -0.5 * (z @ z + len(z) * np.log(2 * np.pi)) - np.sum(np.log(np.diag(self.chol)))


class NumPyNormal(CholeskyNormal):
    # A conditional of the user's own that draws with NumPy's multivariate normal sampler
    # and its Cholesky method, not its default, a singular value decomposition.
    def __init__(self, mean, cov):
        super().__init__(mean, cov)
        self.cov = cov

    def rvs(self, *, random_state):
        return random_state.multivariate_normal(self.mean, self.cov, method="cholesky")


class GammaWithoutPpf:
    # The same law, drawn by NumPy's gamma sampler under common random numbers.
    def __init__(self, shape, rate):
        self.law = Gamma(shape, rate)

    def rvs(self, *, random_state):
        return self.law.rvs(random_state=random_state)

    def logpdf(self, value):
        return self.law.logpdf(value)


def distances():
    rng = np.random.default_rng(127)
    covs = [stats.wishart(6, np.eye(3) / 6).rvs(random_state=rng) for _ in range(200)]
    near_isotropic = np.eye(3) + 0.01 * stats.wishart(6, np.eye(3) / 6).rvs(random_state=rng)
    print("Part 1: mean distance between the two chains' draws of N(0, C) and N(0, C')")
    print(
        f"{'C':15} {'change':6} {'e':>5} {'normal map':>11} {'same state':>11} "
        f"{'Cholesky':>9} {'independent':>12} {'flipped':>8}"
    )
    for name, group in (("random (200)", covs), ("near-isotropic", [near_isotropic])):
        for change in ("scale", "shape"):
            for e in (0.01, 0.1, 0.5):
                rows = [_distances(c, change, e, rng) for c in group]
                mean = np.mean(rows, axis=0)
                # A covariance whose same-state draws are 10 times farther apart than the
                # normal map's: a sign or an order of the decomposition changed.
                flipped = np.mean([row[1] > 10 * row[0] for row in rows])
                print(
                    f"{name:15} {change:6} {e:5.2f} {mean[0]:11.4f} {mean[1]:11.4f} "
                    f"{mean[2]:9.4f} {mean[3]:12.4f} {flipped:8.2f}"
                )


def _distances(c, change, e, rng):
    if change == "scale":
        other = (1 + e) * c
    else:
        s = rng.standard_normal((3, 3))
        s = (s + s.T) / 2
        s /= np.linalg.norm(s, 2)
        root = np.linalg.cholesky(c)
        other = root @ (np.eye(3) + e * s) @ root.T
    zero = np.zeros(3)
    laws = [stats.multivariate_normal(zero, c), stats.multivariate_normal(zero, other)]
    maps = [CholeskyNormal(zero, c), CholeskyNormal(zero, other)]
    total = np.zeros(4)
    for seed in SeedSequence(int(rng.integers(2**32))).spawn(200):
        same = [Generator(PCG64DXSM(seed)) for _ in range(6)]
        fresh = Generator(PCG64DXSM(seed.spawn(1)[0]))
        cholesky = [
            generator.multivariate_normal(zero, cov, method="cholesky")
            for generator, cov in zip(same[4:], (c, other), strict=True)
        ]
        total += [
            np.linalg.norm(maps[0].rvs(random_state=same[0]) - maps[1].rvs(random_state=same[1])),
            np.linalg.norm(laws[0].rvs(random_state=same[2]) - laws[1].rvs(random_state=same[3])),
            np.linalg.norm(cholesky[0] - cholesky[1]),
            np.linalg.norm(laws[0].rvs(random_state=same[2]) - laws[1].rvs(random_state=fresh)),
        ]
    return total / 200


def pumps(with_ppf):
    example = examples.pumps()

    def without_ppf(x, block):
        law = example.conditional(x, block)
        return GammaWithoutPpf(law.shape, law.rate)

    conditional = example.conditional if with_ppf else without_ppf
    return conditional, example.blocks, example.initial, float(np.sqrt(np.trace(example.cov)))


def gaussian(normal):
    # N(0, S), S = 0.5 I + 0.5 (correlation 0.5 between every pair), in blocks (0, 1) and
    # (2, 3): each block given the other is normal with a covariance that does not depend
    # on the state.
    s = 0.5 * np.eye(4) + 0.5
    blocks = [np.array([0, 1]), np.array([2, 3])]
    pieces = []
    for block, rest in (blocks, blocks[::-1]):
        gain = s[np.ix_(block, rest)] @ np.linalg.inv(s[np.ix_(rest, rest)])
        cov = s[np.ix_(block, block)] - gain @ s[np.ix_(rest, block)]
        pieces.append((gain, cov, rest))

    def conditional(x, b):
        gain, cov, rest = pieces[b]
        return normal(gain @ x[rest], cov)

    return conditional, blocks, Uniform(np.full(4, -3.0), np.full(4, 3.0)), 1.0


def regression(normal):
    rng = np.random.default_rng(8)
    n, sigma, nu = 20, 0.5, 3.0
    design = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    y = design @ [1.0, -0.5, 0.25] + sigma * rng.standard_t(nu, n)
    prior = np.eye(3) / 100

    def conditional(x, b):
        if b == 0:
            weights = x[3:] / sigma**2
            precision = design.T @ (weights[:, None] * design) + prior
            cov = np.linalg.inv(precision)
            cov = (cov + cov.T) / 2
            return normal(cov @ (design.T @ (weights * y)), cov)
        residual = y[b - 1] - design[b - 1] @ x[:3]
        return Gamma((nu + 1) / 2, (nu + residual**2 / sigma**2) / 2)

    blocks = [np.arange(3)] + [np.array([3 + i]) for i in range(n)]
    low = np.concatenate([np.full(3, -3.0), np.full(n, 0.1)])
    high = np.concatenate([np.full(3, 3.0), np.full(n, 3.0)])
    return conditional, blocks, Uniform(low, high), 1.0


def normal_means(normal):
    # Three groups of 10 observations, y_ij ~ N(t_j, 1 / r_j), with a flat prior on t and
    # r_j ~ Gamma(1, rate 1): the block t has the normal conditional with the means of the
    # groups and the diagonal covariance diag(1 / (10 r_j)), whose variances are close.
    rng = np.random.default_rng(9)
    y = np.array([0.0, 0.5, 1.0])[:, None] + rng.standard_normal((3, 10))

    def conditional(x, b):
        if b == 0:
            return normal(y.mean(axis=1), np.diag(1 / (10 * x[3:])))
        return Gamma(1 + 10 / 2, 1 + np.sum((y[b - 1] - x[b - 1]) ** 2) / 2)

    blocks = [np.arange(3), np.array([3]), np.array([4]), np.array([5])]
    initial = Uniform(np.array([-3.0, -3.0, -3.0, 0.1, 0.1, 0.1]), np.full(6, 3.0))
    return conditional, blocks, initial, 1.0


def meeting(conditional, blocks, initial, threshold, coupling):
    kernel = Gibbs(
        conditional,
        blocks,
        coupling=coupling,
        threshold=threshold if coupling == "two-scale" else None,
    )
    medians, censored, sweeps, seconds = [], [], 0.0, 0.0
    for seed in SEEDS:
        sampler = telescopic.CoupledSampler(None, initial, kernel=kernel, rng=seed)
        start = time.process_time()
        tau = sampler.meeting_times(PAIRS, max_iter=MAX_ITER)
        seconds += time.process_time() - start
        sweeps += np.sum(np.minimum(tau, MAX_ITER))
        medians.append(float(np.median(tau)))
        censored.append(int(np.sum(np.isinf(tau))))
    return medians, censored, seconds / sweeps


def meeting_times():
    print()
    print(f"Part 2: pilot meeting times in sweeps, {PAIRS} pairs for each of seeds {SEEDS}:")
    print("the median of each seed (pairs censored at 2000), the mean and standard deviation")
    print("of the medians, and the time of a sweep relative to the first row of its target")
    targets = {
        "pumps": {"ppf": pumps(True), "no ppf": pumps(False)},
        "Gaussian, 2 blocks of 2": {
            "map": gaussian(CholeskyNormal),
            "SciPy": gaussian(stats.multivariate_normal),
        },
        "robust regression": {
            "map": regression(CholeskyNormal),
            "SciPy": regression(stats.multivariate_normal),
            "NumPy": regression(NumPyNormal),
        },
        "normal means": {
            "map": normal_means(CholeskyNormal),
            "SciPy": normal_means(stats.multivariate_normal),
        },
    }
    for target, variants in targets.items():
        for coupling in ("crn", "two-scale"):
            reference = None
            for variant, setting in variants.items():
                medians, censored, cost = meeting(*setting, coupling)
                reference = reference or cost
                cells = "  ".join(
                    f"{m:6.1f}" + (f" ({c})" if c else "")
                    for m, c in zip(medians, censored, strict=True)
                )
                print(
                    f"{target:24} {coupling:9} {variant:6}  {cells}  "
                    f"mean {np.mean(medians):6.1f} sd {np.std(medians, ddof=1):5.1f}  "
                    f"cost {cost / reference:4.2f}"
                )


if __name__ == "__main__":
    print(setup())
    start = time.process_time()
    distances()
    meeting_times()
    print(f"\nProcessor time of the run: {(time.process_time() - start) / 60:.1f} minutes")

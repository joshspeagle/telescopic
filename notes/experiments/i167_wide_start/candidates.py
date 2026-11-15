"""Statistics of the tuning pilot that show a start far wider than the target (#167).

Josh ruled on #89 (question 8, about #98) that the default burn-in stays the 99% quantile
of the pilot meeting times, with a warning when the pilot shows a start far wider than
the target. This script measures candidate statistics that the tuning pilot of
`telescopic.tuning.adaptive_metropolis` already has at hand, so that the warning costs no
evaluation of the target: the 100 draws from the initial distribution that set its
chains' first covariance (one of each chain's 25 draws starts it), and the pooled mean
and covariance of its chains' second halves. With s0 the standard deviations of the 100
draws, s those of the pilot, Sigma0 and Sigma the two covariances, and mu the pilot's
mean:

- max_coord: max_i s0_i / s_i, the widest coordinate;
- geo_coord: the geometric mean over coordinates of s0_i / s_i, the shipped statistic,
  read from `PilotRun.initial_width`;
- max_eig: the square root of the largest eigenvalue of Sigma^-1 Sigma0, the widest
  direction;
- rms_dist: sqrt(mean over draws of (x - mu)^T Sigma^-1 (x - mu) / d), which counts an
  offset as well as a spread;
- rhat: the largest R-hat of the pilot, which the existing tuning warning reads (above
  sqrt 2).

Cases. Every example of `telescopic.examples` from its default start (gaussian at its
default d = 10, rho = 0.9, and at d = 1, 2 and 50; banana; mixture; funnel at d = 10;
pumps), and the README's line fit from its own start N(fit, 100 H^-1); then the targets
of #98 (the Gaussians in 1, 10 and 50 dimensions with rho = 0.9, the README line fit, the
banana) from N(mean, w^2 cov) with the exact mean and covariance, for w = 1 (matched),
10 and 100: #98's matched, 10-fold and 100-fold wide starts. Tuning seeds 0 to N - 1 per
case; the 100 draws are those the pilot itself drew, reproduced from its generators for
the statistics other than geo_coord.

Held out. The threshold of 30 for geo_coord was chosen from the seeds 0 to 199 above.
It is then checked, unchanged, on new seeds: 500 seeds from 1000 for each case whose
largest geo_coord at seeds 0 to 199 exceeded 10, other than the 100-fold ones, and 200 seeds from 1000 for each
100-fold case.

The pilot is that of every kernel tuned automatically; a kernel given every parameter has
none. The output's rates carry 95% Clopper-Pearson intervals. "either" counts a run where
geo_coord exceeds the threshold or the existing tuning warning fires.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 python notes/experiments/i167_wide_start/candidates.py
The output: telescopic at commit 4e91557, Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1,
OMP_NUM_THREADS=1, an Intel Xeon processor at 2.10 GHz with 4 logical cores, 4 worker
processes.
"""

import time
from multiprocessing import Pool

import numpy as np
from scipy import optimize, stats

from telescopic import examples
from telescopic._random import TUNING, replicate_generator, root_seed
from telescopic.initial import Normal, as_initial, draw
from telescopic.target import Target
from telescopic.tuning import N_CHAINS, STARTS_PER_CHAIN, adaptive_metropolis

rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)


def line_log_prob(theta):
    theta = np.atleast_2d(theta)
    return -0.5 * np.sum(((y - theta[:, :1] * x - theta[:, 1:]) / yerr) ** 2, axis=1)


design = np.vander(x, 2)
line_cov = np.linalg.inv(design.T @ (design / yerr[:, None] ** 2))
line_mean = line_cov @ (design.T @ (y / yerr**2))
fit = optimize.minimize(lambda t: -line_log_prob(t)[0], [0.0, 0.0])
readme_start = Normal(fit.x, 100 * fit.hess_inv)


def example_of(name):
    kind, *args = name.split(":")
    return getattr(examples, kind)(*map(int, args))


def target_of(name):
    if name == "line":
        return line_log_prob, line_mean, line_cov
    example = example_of(name)
    return example.log_prob, example.mean, example.cov


def default_start(name):
    return readme_start if name == "line" else example_of(name).initial


def initial_draws(initial, seed):
    # The 100 draws of adaptive_metropolis, from the same generators in the same order.
    root = root_seed(seed)
    initial = as_initial(initial)
    generators = [replicate_generator(root, TUNING, i) for i in range(N_CHAINS)]
    sources = [g for g in generators for _ in range(STARTS_PER_CHAIN)]
    first = draw(initial, sources[0])
    rest = [draw(initial, g, len(first)) for g in sources[1:]]
    return np.reshape([first, *rest], (N_CHAINS * STARTS_PER_CHAIN, len(first)))


def statistics(case):
    name, width, seed = case
    log_prob, mean, cov = target_of(name)
    initial = default_start(name) if width is None else Normal(mean, width**2 * cov)
    points = initial_draws(initial, seed)
    pilot = adaptive_metropolis(Target(log_prob, vectorize=True), initial, seed)
    ratio = np.sqrt(pilot.initial_var / np.diag(pilot.cov))
    cov0 = np.atleast_2d(np.cov(points, rowvar=False))
    chol = np.linalg.cholesky(pilot.cov)
    whitened = np.linalg.solve(chol, np.linalg.solve(chol, cov0).T)
    centered = np.linalg.solve(chol, (points - pilot.mean).T)
    return (
        float(ratio.max()),
        pilot.initial_width,
        float(np.sqrt(np.linalg.eigvalsh((whitened + whitened.T) / 2).max())),
        float(np.sqrt((centered**2).sum(axis=0).mean() / len(mean))),
        float(np.nanmax(pilot.rhat)),
    )


DEFAULTS = [
    "gaussian",
    "gaussian:1",
    "gaussian:2",
    "gaussian:50",
    "banana",
    "mixture",
    "funnel",
    "pumps",
    "line",
]
WIDE = ["gaussian:1", "gaussian", "gaussian:50", "line", "banana"]
NAMES = ["max_coord", "geo_coord", "max_eig", "rms_dist", "rhat"]
THRESHOLDS = (10, 15, 20, 30, 50)
THRESHOLD = 30  # chosen from seeds 0 to 199, then held out


def rate(values, group, j, threshold):
    hits = sum(int((values[c][:, j] > threshold).sum()) for c in group)
    total = sum(len(values[c]) for c in group)
    lo = stats.beta.ppf(0.025, hits, total - hits + 1) if hits else 0.0
    hi = stats.beta.ppf(0.975, hits + 1, total - hits) if hits < total else 1.0
    return f"{hits}/{total} [{lo:.4f}, {hi:.4f}]"


def label(case):
    return f"{case[0]} from " + ("its default start" if case[1] is None else f"w = {case[1]}")


def held_out(pool, values):
    near = [c for c, v in values.items() if v[:, 1].max() > 10 and c[1] != 100]
    far = [c for c in values if c[1] == 100]
    print()
    print(f"Held out: geo_coord > {THRESHOLD} alone, and either, on new seeds")
    for case in near + far:
        n = 500 if case in near else 200
        v = np.array(pool.map(statistics, [(*case, seed) for seed in range(1000, 1000 + n)]))
        both = {case: np.column_stack([v[:, 1] * (1 + 1e9 * (v[:, 4] > np.sqrt(2)))])}
        print(
            f"  {label(case)}, seeds 1000 to {999 + n}, largest {v[:, 1].max():.3g}: "
            f"{rate({case: v}, [case], 1, THRESHOLD)}; {rate(both, [case], 0, THRESHOLD)}"
        )


if __name__ == "__main__":
    n = 200
    cases = [(name, None) for name in DEFAULTS]
    cases += [(name, w) for w in (1, 10, 100) for name in WIDE]
    start = time.time()
    pool = Pool()
    values = {
        case: np.array(pool.map(statistics, [(*case, seed) for seed in range(n)]))
        for case in cases
    }
    print(f"Seeds 0 to {n - 1} per case; the minimum, median, 99% quantile and maximum")
    for case, v in values.items():
        print(label(case))
        for j, stat in enumerate(NAMES):
            q = np.quantile(v[:, j], [0, 0.5, 0.99, 1])
            print(f"  {stat:9s} " + " ".join(f"{e:10.3g}" for e in q))
    print()
    print("Fraction above each threshold, with its 95% Clopper-Pearson interval, cases pooled")
    groups = {
        "defaults": [c for c in cases if c[1] is None],
        "w = 1": [c for c in cases if c[1] == 1],
        "w = 10": [c for c in cases if c[1] == 10],
        "w = 100": [c for c in cases if c[1] == 100],
    }
    for j, stat in enumerate(NAMES[:4]):
        for threshold in THRESHOLDS:
            print(
                f"{stat} > {threshold}: "
                + "; ".join(f"{g} {rate(values, grp, j, threshold)}" for g, grp in groups.items())
            )
    print()
    print("geo_coord > threshold or rhat > sqrt 2, the existing tuning warning")
    either = {
        case: np.column_stack([v[:, 1] * (1 + 1e9 * (v[:, 4] > np.sqrt(2)))])
        for case, v in values.items()
    }
    for threshold in THRESHOLDS:
        print(
            f"either at {threshold}: "
            + "; ".join(f"{g} {rate(either, grp, 0, threshold)}" for g, grp in groups.items())
        )
    print()
    print(f"Per case at the threshold {THRESHOLD}: geo_coord alone, and either")
    for case in cases:
        print(
            f"  {label(case)}: {rate(values, [case], 1, THRESHOLD)}; "
            f"{rate(either, [case], 0, THRESHOLD)}"
        )
    held_out(pool, values)
    pool.close()
    print(f"\n{time.time() - start:.0f} s")

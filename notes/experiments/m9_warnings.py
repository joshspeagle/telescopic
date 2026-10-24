"""How often the warnings of spec section 7.4 fire, on clean targets and on a failure (M9, D-16).

Clean targets: the automatic pipeline (`telescopic.sample` with its defaults) on the
correlated Gaussian in 10 dimensions and the banana, from their default initial
distributions: 100 seeds at R = 100 and 50 at R = 500 (the tail warning needs R >= 500).
The pump model is left out: there the automatic random walk chose k = 29,214 and a run
took 176 s (one probe, seed 0), where the Gibbs sampler of JOA20 used k = 7.

Failure: the bimodal mixture of Jacob, O'Leary & Atchade (2020, section 5.1) from the
narrow N(10, 1), proposal variance 1, k = 50, m = 500, lag 1, R = 1,000, 50 seeds: how often
the 95% interval of P(x > 3) misses the truth 0.5 Phi(1) + 0.5 Phi(-7), how often a
warning fires then, and in how many runs one replicate holds more than half the variance.

Run: python notes/experiments/m9_warnings.py [names], where names, by default all, are
among gaussian, banana and mixture. The recorded output comes from two runs, the second
for the mixture after a time limit stopped the first. `m11_warnings.out` is a run at
ac4a032, after the variance share moved to moments about the atoms' mean (DECISIONS #54).
"""

import sys
import time
import warnings
from collections import Counter

import numpy as np
from scipy import stats

import telescopic
from telescopic import diagnostics, examples
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk

KINDS = ("censored", "late", "variance", "tail", "tuning", "slow")
CHOSEN = set(sys.argv[1:]) or {"gaussian", "banana", "mixture"}


def kinds(result):
    return {issue.kind for issue in diagnostics.issues(result)}


for example in (examples.gaussian(), examples.banana()):
    if example.name not in CHOSEN:
        continue
    for n_replicates, seeds in ((100, 100), (500, 50)):
        start = time.perf_counter()
        fired, ks = Counter(), []
        for seed in range(seeds):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = telescopic.sample(
                    example.log_prob, example.initial, n_replicates, vectorize=True, rng=seed
                )
            found = kinds(result)
            fired.update(found)
            fired["any"] += bool(found)
            ks.append(result.k)
        rates = ", ".join(f"{kind} {fired[kind]}" for kind in (*KINDS, "any"))
        print(
            f"{example.name}, R = {n_replicates}, {seeds} runs: fired {rates}; k median "
            f"{np.median(ks):.0f} (range {min(ks)} to {max(ks)}); "
            f"{(time.perf_counter() - start) / seeds:.1f} s per run",
            flush=True,
        )

if "mixture" in CHOSEN:
    mixture = examples.mixture()
    truth = 0.5 * stats.norm.cdf(1) + 0.5 * stats.norm.cdf(-7)
    start = time.perf_counter()
    outcomes, by_kind, shares, misses = Counter(), Counter(), [], []
    for seed in range(50):
        sampler = telescopic.CoupledSampler(
            mixture.log_prob,
            Normal([10.0], [[1.0]]),
            kernel=RandomWalk(np.eye(1), scale=1.0),
            vectorize=True,
            rng=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sampler.run(1000, k=50, m=500, lag=1)
        low, high = result.estimate(lambda x: (x[:, 0] > 3).astype(float)).interval
        missed = not low <= truth <= high
        found = kinds(result)
        moments = [result.measure.per_replicate(h) for h in (lambda x: x, np.square)]
        shares.append(np.max(diagnostics.variance_share(np.concatenate(moments, axis=1))))
        misses.append(missed)
        outcomes[missed, bool(found)] += 1
        by_kind.update(f"{'missed' if missed else 'covered'}: {kind}" for kind in found)
    print(
        f"mixture from N(10, 1), proposal variance 1, k = 50, m = 500, R = 1000, 50 runs: "
        f"{outcomes[True, True] + outcomes[True, False]} intervals missed {truth:.4f}; missed "
        f"and warned {outcomes[True, True]}, missed and silent {outcomes[True, False]}, covered "
        f"and warned {outcomes[False, True]}; by kind {sorted(by_kind.items())}; largest "
        f"variance share above 0.5 in {np.sum(np.array(shares) > 0.5)} runs, "
        f"{np.sum((np.array(shares) > 0.5) & misses)} of them missed; "
        f"{(time.perf_counter() - start) / 50:.1f} s per run",
        flush=True,
    )

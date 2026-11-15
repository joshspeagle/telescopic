"""Does keep=None change any number? (issue #79, part A, risk R3)

Question. Part A of #79 adds reduced storage (keep), errors that keep finished replicates
and a time for each atom. Its design claims that a run with keep=None gives every number
it gave before, bitwise, except the values of variance_share, whose second moments move
from the atoms' unweighted mean to the mean estimate (which it calls a change of
rounding), and with it the estimate and the replicate that hold the share.

Design. The same runs at two checkouts: engine runs evaluated point by point (lags 1 to
4, thin 1 and 3, a cap that censors pairs, 1 and 8 lanes), a vectorized MALA run, and
sampler runs that tune, run a pilot, extend a run and run on a thread pool. For each,
every array of the replicates that both versions have, and every estimate, quantile,
histogram, bound, diagnostic and summary column of its Result, saved with --save under
the PYTHONPATH of each checkout; --compare reports each key as equal bitwise, or its
largest absolute and relative differences. Wall times are left out.

    PYTHONPATH=<base>/src python r3_compare.py --save base.npz
    PYTHONPATH=src python r3_compare.py --save head.npz
    python r3_compare.py --compare base.npz head.npz

Result (r3_compare.out, base 9fb63e8 against 80a1e55, after the review of part A). Of
352 keys, 344 are bitwise equal: every array of the replicates, every estimate, quantile,
histogram, bound, summary column, pilot and tuning array. The 8 that differ are
variance_share in 7 of the 8 runs, by up to 0.218 (relative 0.272), and with it the
estimate that holds the share in one run (the mean of x0 against the variance of x0)
and the replicate in another. The change is not one of rounding, as the design said: a
replicate's second moment about the mean estimate differs from the one about the atoms'
mean by 2 e f_r + e^2, whose spread over the replicates moves the shares (DECISIONS
#240).
"""

import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPLICATE_FIELDS = (
    "index",
    "meeting_time",
    "atoms",
    "weights",
    "owner",
    "distance_owner",
    "distance_time",
    "distance",
    "zero_density",
    "requested",
    "n_log_prob",
    "n_grad",
    "n_grad_zeroed",
)


def gaussian(x):
    return -0.5 * np.sum(np.square(x), axis=-1)


def result_arrays(name, result, out):
    for field in REPLICATE_FIELDS:
        out[f"{name}/replicates.{field}"] = np.asarray(getattr(result._replicates, field))
    mean = result.mean()
    for field in ("value", "standard_error", "interval", "replicates"):
        out[f"{name}/mean.{field}"] = getattr(mean, field)
    estimate = result.estimate(lambda x: np.sin(x).sum(axis=1))
    out[f"{name}/estimate.value"] = estimate.value
    out[f"{name}/estimate.interval"] = estimate.interval
    out[f"{name}/cov"] = result.cov()
    out[f"{name}/var"] = result.var()
    out[f"{name}/std"] = result.std()
    out[f"{name}/quantile"] = result.quantile([0.1, 0.5, 0.9])
    out[f"{name}/quantile_h"] = result.quantile([0.25, 0.75], lambda x: x[:, 0] * x[:, 1])
    out[f"{name}/credible_interval"] = result.credible_interval()
    masses, errors, edges = result.histogram(0, 15)
    out[f"{name}/histogram"] = np.concatenate([masses, errors, edges])
    masses, errors, _ = result.histogram([0, 1], 6, range=[[-3, 3], [-3, 3]])
    out[f"{name}/histogram2"] = np.concatenate([masses.ravel(), errors.ravel()])
    out[f"{name}/tv_bound"] = result.tv_bound()
    out[f"{name}/w1_bound"] = result.w1_bound()
    for key, value in result.diagnostics().items():
        if key == "seconds_per_replicate":
            continue  # wall time, which differs between any two runs
        if isinstance(value, int | float):
            out[f"{name}/diagnostics.{key}"] = np.asarray(value, dtype=np.float64)
        else:
            out[f"{name}/diagnostics.{key}"] = np.asarray(str(value))
    for key, value in result.summary().table.items():
        out[f"{name}/summary.{key}"] = value
    if result.pilot is not None:
        out[f"{name}/pilot"] = result.pilot
    if result.tuning is not None:
        for field in ("mean", "cov", "acceptance", "rhat"):
            out[f"{name}/tuning.{field}"] = np.asarray(getattr(result.tuning, field))


def collect():
    import telescopic
    from telescopic import examples
    from telescopic.engine import run
    from telescopic.initial import Normal
    from telescopic.kernels import MALA, RandomWalk
    from telescopic.results import Result
    from telescopic.target import Target

    out = {}
    far = Normal([3.0, -1.0], [1.0, 1.0])
    engine_runs = {
        "engine-lag1": {"k": 2, "m": 10, "lag": 1, "lanes": 8},
        "engine-lag4-thin3": {"k": 5, "m": 20, "lag": 4, "thin": 3, "lanes": 8},
        "engine-censored": {"k": 2, "m": 6, "lag": 2, "max_iter": 9, "lanes": 8},
        "engine-one-lane": {"k": 3, "m": 12, "lag": 3, "lanes": 1},
    }
    for name, options in engine_runs.items():
        replicates = run(
            Target(gaussian), RandomWalk(np.eye(2), scale=1.0), far, range(60), rng=1, **options
        )
        result_arrays(name, Result(replicates), out)
    g3 = examples.gaussian(3, 0.5)
    replicates = run(
        Target(g3.log_prob, grad=g3.grad, vectorize=True),
        MALA(0.8, cov=g3.cov),
        Normal(np.full(3, 2.0), np.eye(3)),
        range(80),
        rng=2,
        k=3,
        m=20,
        lag=3,
    )
    result_arrays("engine-mala-vectorized", Result(replicates), out)
    g2 = examples.gaussian(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = telescopic.CoupledSampler(g2.log_prob, g2.initial, vectorize=True, rng=3)
        sampler.run(70)
        result_arrays("sampler-extended", sampler.run(40), out)
        with ThreadPoolExecutor(2) as pool:
            pooled = telescopic.sample(
                gaussian, far, 300, k=4, m=40, lag=4, pool=pool, n_workers=2, rng=4
            )
        result_arrays("sample-thread-pool", pooled, out)
    return out


def compare(base_path, head_path):
    base, head = np.load(base_path), np.load(head_path)
    keys = sorted(set(base.files) | set(head.files))
    differing = []
    for key in keys:
        if key not in base.files or key not in head.files:
            print(f"MISSING  {key}: only in {'base' if key in base.files else 'head'}")
            differing.append(key)
            continue
        a, b = base[key], head[key]
        nan = a.dtype.kind == "f"
        if a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=nan):
            continue
        differing.append(key)
        if a.dtype.kind in "fi" and a.shape == b.shape:
            gap = np.max(np.abs(a.astype(float) - b.astype(float)))
            scale = np.max(np.abs(a.astype(float)))
            print(f"DIFFERS  {key}: max |diff| {gap:.3g}, relative {gap / scale:.3g}")
        else:
            print(f"DIFFERS  {key}: {a!r} against {b!r}")
    print(f"{len(keys)} keys compared; {len(keys) - len(differing)} bitwise equal")
    return differing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    parser.add_argument("--compare", nargs=2)
    options = parser.parse_args()
    if options.save:
        np.savez(options.save, **collect())
    if options.compare:
        differing = compare(*options.compare)
        # The share, and the estimate and replicate that hold it, which follow from it.
        shares = ("variance_share", "variance_share_of", "variance_share_replicate")
        expected = all(key.rsplit(".", 1)[-1] in shares for key in differing)
        print(
            "no number other than variance_share differs"
            if expected
            else "a number other than variance_share differs"
        )
        sys.exit(0 if expected else 1)


if __name__ == "__main__":
    main()

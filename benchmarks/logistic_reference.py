"""Generate independent logistic posterior references using Polya-Gamma Gibbs.

Requires polyagamma==2.0.2 and arviz-stats==1.3.0. Run from the checkout using
PYTHONPATH=src python -m benchmarks.logistic_reference. The reported source
digests identify the generator and data design.
"""

import argparse
import hashlib
import importlib
import json
import multiprocessing
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy import linalg

from benchmarks import logistic
from benchmarks.targets import Array

SEEDS = (91001, 91002, 91003, 91004)


def chain(dimension: int, seed: int, warmup: int, draws: int) -> Array:
    """Return retained coefficients of shape (draws, dimension) from one Gibbs chain."""
    polyagamma = importlib.import_module("polyagamma")
    data = logistic.design(dimension)
    rng = np.random.default_rng(seed)
    beta = rng.normal(0.0, 2.0, dimension)
    prior = np.eye(dimension) / logistic.PRIOR_VARIANCE
    shift = data.matrix.T @ (data.outcomes - 0.5)
    samples = np.empty((draws, dimension))
    transposed = np.ascontiguousarray(data.matrix.T)
    threadpoolctl = importlib.import_module("threadpoolctl")
    with threadpoolctl.threadpool_limits(limits=1):
        for step in range(warmup + draws):
            weights = polyagamma.random_polyagamma(
                1.0, data.matrix @ beta, method="devroye", random_state=rng
            )
            factor = linalg.cholesky(
                transposed @ (weights[:, None] * data.matrix) + prior, lower=True
            )
            center = linalg.cho_solve((factor, True), shift)
            beta = center + linalg.solve_triangular(factor.T, rng.standard_normal(dimension))
            if step >= warmup:
                samples[step - warmup] = beta
    return samples


def main() -> None:
    """Write a chain-level reference report with independently estimated MC standard errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dimension", choices=(6, 300), type=int)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    if args.batch_size < 2 or args.draws < 20 * args.batch_size or args.draws % args.batch_size:
        parser.error("draws must be divisible by batch-size and include at least 20 batches")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=len(SEEDS), mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        tasks = [
            executor.submit(chain, args.dimension, seed, args.warmup, args.draws) for seed in SEEDS
        ]
        samples = np.stack([task.result() for task in tasks])
    del tasks
    arviz_stats = importlib.import_module("arviz_stats")
    rhat = float(np.max(arviz_stats.rhat(samples, method="rank")))
    bulk_ess = float(np.min(arviz_stats.ess(samples, method="bulk")))
    tail_ess = float(np.min(arviz_stats.ess(samples, method="tail", prob=(0.05, 0.95))))
    indices = logistic.ESTIMAND_INDICES
    values = samples[..., list(indices)]
    means = values.mean(axis=(0, 1))
    squares = (values - means) ** 2
    quantities = np.concatenate([values, squares], axis=-1)
    batch_means = quantities.reshape(-1, args.batch_size, quantities.shape[-1]).mean(axis=1)
    errors = batch_means.std(axis=0, ddof=1) / np.sqrt(len(batch_means))
    report = {
        **vars(args),
        "seeds": SEEDS,
        "method": "polya_gamma_gibbs_devroye",
        "polyagamma": version("polyagamma"),
        "arviz_stats": version("arviz-stats"),
        "rhat_max": rhat,
        "bulk_ess_min": bulk_ess,
        "tail_ess_min": tail_ess,
        "converged": rhat <= 1.01 and min(bulk_ess, tail_ess) >= 400,
        "dataset_sha256": logistic.design_digest(logistic.design(args.dimension)),
        "prior_variance": logistic.PRIOR_VARIANCE,
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "python": platform.python_version(),
        "machine": f"{platform.machine()} {platform.system()}",
        "blas_threads_per_chain": 1,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(logistic.__file__))
        },
        "indices": indices,
        "start_mean": samples.mean(axis=(0, 1)).tolist(),
        "start_variance": samples.var(axis=(0, 1)).tolist(),
        "mean": means.tolist(),
        "variance": squares.mean(axis=(0, 1)).tolist(),
        "mean_standard_error": errors[: len(indices)].tolist(),
        "variance_standard_error": errors[len(indices) :].tolist(),
        "chain_means": values.mean(axis=1).tolist(),
        "wall_seconds": time.perf_counter() - start,
    }
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()

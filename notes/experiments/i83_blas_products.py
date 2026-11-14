"""How far BLAS products for vectorized targets move seeded runs, and what they gain (issue #83).

Before this change every kernel computed its products of a batch of rows with a d-by-d
factor with `np.einsum`, which gives a row the same bits in any batch. Now, when the
target is vectorized, RandomWalk, MALA, HMC, `reflection_maximal` and the tuning pilot
compute them with BLAS (`vectors @ matrix.T`); `MultivariateT`, and so Independence, keeps
`einsum`. Two questions: how far do seeded outputs of vectorized runs move, and how much
faster are whole runs?

The source of commit `BEFORE` (the `main` this change starts from) is extracted with
`git archive` into a temporary folder; "after" is this checkout's `src`. Each run is a
child interpreter (`python -I`, the source folder first on its path) with one BLAS and
OpenMP thread. The target is a vectorized N(0, Sigma), Sigma_ij = 0.9^|i-j| (the
covariance of `examples.gaussian(d)`), whose log-density and gradient are BLAS products
with precomputed factors, so that the target is cheap beside the kernels; the initial
distribution is `examples.gaussian(d).initial`, N(0, 4I). Kernels are given Sigma, with
RandomWalk's default scale 2.38 / sqrt(d), MALA's step size 1.65 d^(-1/6) and HMC's
d^(-1/4) (their tuners' starting points), and Independence a Gaussian proposal of
covariance 1.5 Sigma.

1. Moved numbers. Each kernel (RandomWalk with the reflection and the maximal coupling,
   Independence, MALA, HMC) runs 64 replicates at d = 2, 20 and 200 with k = 10, m = 50,
   lag 1, max_iter = 2000 and rng = 1, through `engine.run`. For each run the script
   prints a hash of the meeting times, atoms and weights, before and after, the number of
   meeting times that differ, and the largest relative difference between before and
   after of the estimates of E[x_i^2] (whose values are near Sigma_ii = 1), and of E[x_i]
   in units of its standard error.
2. Gain. RandomWalk and MALA at d = 20 and 200, 256 replicates with k = 10, m = 200, lag
   1, max_iter = 1000 and rng = 2: the processor time of `engine.run` alone, before and
   after, in an order that alternates between rounds, after one uncounted round. The ratio
   before/after of each round is summarized by its median and its range.

Gibbs and Mixture compute no products of their own (Mixture delegates to its
components), so they are not run.

Result (`i83_blas_products.out`, at a load average of 1.5 to 3.0 on 4 cores). Moved
numbers: every run of RandomWalk, MALA and HMC changed its hash, and Independence's did
not; the maximal coupling's runs at d = 20 and 200 censored all 64 pairs, so they store
no atoms and their hash is unchanged. Meeting times were all equal except HMC's at d = 2,
where 19 of 64 differ: there the coupled trajectories contract until the two chains become
equal, which turns on the last bits. The largest relative difference of the estimates of E[x_i^2] was at most
1.1e-13 (RandomWalk at d = 200, from the single pair of 64 that met) and 3.7e-4 for HMC at
d = 2; that of E[x_i] at most 2.3e-13 standard errors, and 0.051 for HMC at d = 2. Gain,
before/after per round over 7 rounds: RandomWalk 1.13 at d = 20 (0.84 to 1.48) and 2.34
at d = 200 (1.92 to 2.57); MALA 1.40 at d = 20 (1.00 to 2.30) and 3.45 at d = 200 (2.96 to
3.77). The issue measured 2.95 and 3.32 at d = 200 on another load and configuration.

Run from the repository root: OMP_NUM_THREADS=1 PYTHONPATH=src python notes/experiments/i83_blas_products.py
"""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from i85_setup import setup

ROOT = Path(__file__).resolve().parents[2]
BEFORE = "3dc2412"
ROUNDS = 7
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

# Runs one configuration and prints its hash, estimates and processor time as JSON.
CHILD = """
import hashlib, json, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from telescopic import examples
from telescopic.engine import run
from telescopic.kernels import HMC, MALA, Independence, MultivariateT, RandomWalk
from telescopic.target import Target

name, d, n, k, m, max_iter, seed = sys.argv[2], *map(int, sys.argv[3:])
example = examples.gaussian(d)
cov = example.cov
precision = np.linalg.inv(cov)
whiten = np.linalg.cholesky(precision)
target = Target(
    lambda x: -0.5 * np.sum((x @ whiten) ** 2, axis=1),
    grad=lambda x: -(x @ precision),
    vectorize=True,
)
kernel = {
    "RandomWalk": lambda: RandomWalk(cov),
    "RandomWalk maximal": lambda: RandomWalk(cov, coupling="maximal"),
    "Independence": lambda: Independence(MultivariateT(np.zeros(d), 1.5 * cov, df=None)),
    "MALA": lambda: MALA(1.65 * d ** (-1 / 6), cov=cov),
    "HMC": lambda: HMC(d**-0.25, cov=cov),
}[name]()
start = time.process_time()
replicates = run(
    target, kernel, example.initial, range(n), k=k, m=m, lag=1, max_iter=max_iter, rng=seed
)
seconds = time.process_time() - start
digest = hashlib.sha256()
for array in (replicates.meeting_time, replicates.atoms, replicates.weights):
    digest.update(np.ascontiguousarray(array).tobytes())
# Per-replicate estimates of E[x] and E[x^2]: the weighted sums of each replicate's atoms.
first = np.zeros((n, d))
second = np.zeros((n, d))
np.add.at(first, replicates.owner, replicates.weights[:, None] * replicates.atoms)
np.add.at(second, replicates.owner, replicates.weights[:, None] * replicates.atoms**2)
print(json.dumps({
    "hash": digest.hexdigest()[:12],
    "meeting_time": replicates.meeting_time.tolist(),
    "mean": first.mean(axis=0).tolist(),
    "error": (first.std(axis=0, ddof=1) / np.sqrt(n)).tolist(),
    "second": second.mean(axis=0).tolist(),
    "seconds": seconds,
}))
"""


def child(source, *arguments):
    output = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(source), *map(str, arguments)],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ | THREADS,
    ).stdout
    return json.loads(output)


def extract(commit, folder):
    archive = subprocess.run(
        ["git", "archive", commit, "src/telescopic"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(folder, filter="data")
    return Path(folder) / "src"


def moved(sources):
    print("\n1. Moved numbers: 64 replicates, k = 10, m = 50, lag 1, max_iter = 2000, rng = 1")
    kernels = ["RandomWalk", "RandomWalk maximal", "Independence", "MALA", "HMC"]
    for name in kernels:
        for d in (2, 20, 200):
            before, after = (child(sources[s], name, d, 64, 10, 50, 2000, 1) for s in sources)
            times = np.array([before["meeting_time"], after["meeting_time"]])
            line = (
                f"{name:18} d = {d:3}: hash {before['hash']} -> {after['hash']}; meeting "
                f"times that differ {int((times[0] != times[1]).sum()):2} of 64, censored "
                f"{int(np.isinf(times[1]).sum()):2}"
            )
            if np.isinf(times).all():  # no atoms, so no estimates
                print(line)
                continue
            second = np.array([before["second"], after["second"]])
            mean, error = np.array([before["mean"], after["mean"]]), np.array(before["error"])
            relative = np.max(np.abs(second[1] - second[0]) / np.abs(second[0]))
            in_errors = np.max(np.abs(mean[1] - mean[0]) / error)
            print(
                f"{line}; E[x^2] largest relative difference {relative:.1e}; E[x] largest "
                f"difference {in_errors:.1e} standard errors"
            )


def gain(sources):
    print(
        f"\n2. Gain: 256 replicates, k = 10, m = 200, lag 1, max_iter = 1000, rng = 2; "
        f"{ROUNDS} rounds after one uncounted"
    )
    for name in ("RandomWalk", "MALA"):
        for d in (20, 200):
            seconds = {label: [] for label in sources}
            for round_ in range(ROUNDS + 1):
                order = list(sources) if round_ % 2 == 0 else list(reversed(sources))
                for label in order:
                    result = child(sources[label], name, d, 256, 10, 200, 1000, 2)
                    if round_:
                        seconds[label].append(result["seconds"])
            ratio = np.array(seconds["before"]) / np.array(seconds["after"])
            print(
                f"{name:10} d = {d:3}: processor seconds median {np.median(seconds['before']):.2f} "
                f"before, {np.median(seconds['after']):.2f} after; before/after per round: "
                f"median {np.median(ratio):.2f}, range {ratio.min():.2f} to {ratio.max():.2f}"
            )


def main():
    print(setup())
    print(f"before: commit {BEFORE}; after: this checkout; children run with {THREADS}")
    print(f"load average before: {tuple(round(x, 2) for x in os.getloadavg())}")
    with tempfile.TemporaryDirectory() as folder:
        sources = {"before": extract(BEFORE, folder), "after": ROOT / "src"}
        moved(sources)
        gain(sources)
    print(f"load average after: {tuple(round(x, 2) for x in os.getloadavg())}")


if __name__ == "__main__":
    main()

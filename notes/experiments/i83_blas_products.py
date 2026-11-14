"""How far BLAS products for vectorized targets move seeded runs, and what they gain (issue #83).

Before this change every kernel computed its products of a batch of rows with a d-by-d
factor with `np.einsum`, which gives a row the same bits in any batch. Now, when the
target is vectorized, RandomWalk, MALA, HMC, `reflection_maximal` and the tuning pilot
compute them with BLAS (`vectors @ matrix.T`); `MultivariateT`, and so Independence, keeps
`einsum`. Five questions: how far do seeded outputs of vectorized runs move, how far do
runs of one seed with different lane counts now differ, how much faster are whole runs,
what does one product cost, and what does the BLAS do with equal rows?

The source of commit `BEFORE` (the `main` merged into this change last) is extracted with
`git archive` into a temporary folder; "after" is this checkout's `src`. Each run is a
child interpreter (`python -I`, the source folder first on its path) with one BLAS and
OpenMP thread. The target is a vectorized N(0, Sigma), Sigma_ij = 0.9^|i-j| (the
covariance of `examples.gaussian(d)`), whose log-density and gradient are BLAS products
with precomputed factors, so that the target is cheap beside the kernels; the initial
distribution is `examples.gaussian(d).initial`, N(0, 4I). Kernels are given Sigma, with
RandomWalk's default scale 2.38 / sqrt(d), MALA's step size 1.65 d^(-1/6) and HMC's
d^(-1/4) (their tuners' starting points), and Independence a Gaussian proposal of
covariance 1.5 Sigma. Estimates are `Result.estimate`'s, of E[x] and E[x^2], over the
replicates whose pairs met; every pair of sections 1 and 2 met (the script checks it).

1. Moved numbers. Each kernel (RandomWalk with the reflection coupling at d = 2, 20 and
   200, and with the maximal coupling at d = 2 only, Independence, MALA, HMC at d = 2, 20
   and 200) runs 64 replicates with k = 10, m = 50, lag 1, max_iter = 20000 and rng = 1,
   through `engine.run` with its default lanes (all 64 in one batch). The maximal coupling
   is left out at d = 20 and 200, where its proposals almost never coincide: all 64 of its
   pairs were censored there at max_iter = 2000 in this script's first version (commit
   aa145f7). For each run the script prints a hash of the meeting times, atoms and
   weights, before and after, the number of meeting times that differ, and the largest
   relative change of the pooled estimates of E[x_i^2] (near Sigma_ii = 1), and of E[x_i]
   in units of its standard error.
2. Layouts. The same runs with lanes = 1 and 7 against lanes = 64, before and after, for
   the target above (BLAS) and for one whose products are einsum's (row-wise), which
   gives a row the same bits in any batch, as the built-in examples do: the number of
   meeting times that differ, the largest change of one replicate's estimate of a
   coordinate of E[x], and the largest change of a coordinate of the pooled estimate of
   E[x] in units of its standard error.
3. Gain. RandomWalk and MALA at d = 20 and 200, 256 replicates with k = 10, m = 200, lag
   1, max_iter = 1000 and rng = 2: the processor time of `engine.run` alone, before and
   after, in an order that alternates between rounds, after one uncounted round. The ratio
   before/after of each round is summarized by its median and its range. The number of
   censored pairs is printed: a censored pair runs to max_iter, which is a whole run's
   cost all the same.
4. One product. `_linalg.matvec` of this checkout for a factor of `examples.gaussian(d)`
   and rows of normals, with `einsum`, with BLAS inside `np.errstate` as `matvec` does,
   and with BLAS alone, at d = 2, 10, 20, 200 and 1, 16, 256 rows: microseconds per call,
   the median and range of 7 repeats of a loop of calls.
5. Equal rows. For each OpenBLAS kernel set (`OPENBLAS_CORETYPE`), with 1 and 4 threads,
   over 81 shapes (n of 2, 3, 4, 6, 8, 16, 32, 64, 256 and d of 2, 5, 10, 20, 33, 50, 64,
   100, 200): (a) shapes of 2n rows whose rows i and n + i are equal, as a coupled step's
   two chains in one call, where some such pair gets different bits; (b) shapes where a
   row at one position gets other bits in a second call of the same shape whose other
   rows differ, the assumption that a call per chain keeps equal chains equal; and (c)
   whether, with HMC's products of both chains in one call instead of one call per chain,
   the configuration of the real-BLAS case of
   `test_equal_hmc_chains_stay_equal_where_a_product_gives_equal_rows_other_bits`
   (d = 33, 3 pairs, 5 coupled steps) separates its equal chains.

Gibbs and Mixture compute no products of their own (Mixture delegates to its
components), so they are not run.

Result (`i83_blas_products.out`, at a load average of 1.7 to 2.4 on 4 cores shared with
other jobs).
1. Every run of RandomWalk, MALA and HMC changed its hash and Independence's did not;
   every pair met. Meeting times were equal except HMC's at d = 2, where 19 of 64 differ:
   its coupled trajectories contract until the chains are equal, an event that turns on
   the last bits. The pooled estimates of E[x_i^2] changed by at most 1.4e-13 relative
   and those of E[x_i] by at most 8.9e-14 standard errors, except HMC's at d = 2, by
   3.7e-4 and 0.051.
2. With the row-wise target, every run of 1 or 7 lanes was bitwise that of 64 before the
   change. After it, RandomWalk's, MALA's and HMC's replicates changed by at most 4.2e-11
   (RandomWalk at d = 200) with equal meeting times, except HMC's at d = 2: 17 (1 lane)
   and 8 (7 lanes) of 64 meeting times differ, a replicate's estimate moved by up to 0.20,
   and the pooled estimate by up to 0.067 standard errors. With the BLAS target, MALA's
   and HMC's runs at d = 20 and 200 already differed between layouts before the change,
   by at most 6.4e-14, through the target's own products; RandomWalk's, whose positions
   do not pass through the target, did not.
3. Before/after per round: RandomWalk 1.11 at d = 20 (0.91 to 1.43) and 2.11 at d = 200
   (2.04 to 2.23), where all 256 pairs ran to max_iter; MALA 1.47 at d = 20 (1.08 to
   2.17) and 3.41 at d = 200 (2.68 to 3.63).
4. For 1 row at d up to 20, and 16 rows at d = 2, BLAS inside `np.errstate` cost more than
   `einsum` (1.87 against 1.54 microseconds at d = 2, 3.28 against 3.22 at d = 20); the
   `errstate` added 0.9 to 3.0 microseconds at d up to 20. At 256 rows BLAS with the
   `errstate` was 1.6 (d = 2) to 6.7 (d = 200) times faster than `einsum`.
5. Equal rows of one call got different bits in 2 (1 thread) or 3 (4 threads) of 81
   shapes with the SkylakeX kernels, this machine's, and in 9 with Nehalem's; in none
   with Haswell's (which OpenBLAS also chose for Zen) or Sandybridge's. No kernel set
   gave a row other bits in another call of the same shape. HMC with both chains in one
   call separated the test's equal chains with the SkylakeX and Nehalem kernels only.
Not tested: MKL, Accelerate, BLIS and ARM kernels.

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
BEFORE = "49256f5"
ROUNDS = 7
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
KERNELS = {
    "RandomWalk": (2, 20, 200),
    "RandomWalk maximal": (2,),
    "Independence": (2, 20, 200),
    "MALA": (2, 20, 200),
    "HMC": (2, 20, 200),
}

# Runs one configuration and prints its hash, meeting times, estimates and processor
# time as JSON.
CHILD = """
import hashlib, json, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from telescopic import examples
from telescopic.engine import run
from telescopic.kernels import HMC, MALA, Independence, MultivariateT, RandomWalk
from telescopic.results import Result
from telescopic.target import Target

name, (d, n, k, m, max_iter, seed, lanes, rowwise) = sys.argv[2], map(int, sys.argv[3:])
example = examples.gaussian(d)
cov = example.cov
precision = np.linalg.inv(cov)
whiten = np.linalg.cholesky(precision)
# BLAS products, or einsum's, which give a row the same bits in any batch.
product = (lambda x, a: np.einsum("ij,jk->ik", x, a)) if rowwise else (lambda x, a: x @ a)
target = Target(
    lambda x: -0.5 * np.sum(product(x, whiten) ** 2, axis=1),
    grad=lambda x: -product(x, precision),
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
    target, kernel, example.initial, range(n), k=k, m=m, lag=1, max_iter=max_iter,
    rng=seed, lanes=lanes,
)
seconds = time.process_time() - start
digest = hashlib.sha256()
for array in (replicates.meeting_time, replicates.atoms, replicates.weights):
    digest.update(np.ascontiguousarray(array).tobytes())
output = {
    "hash": digest.hexdigest()[:12],
    "meeting_time": replicates.meeting_time.tolist(),
    "censored": int(np.isinf(replicates.meeting_time).sum()),
    "seconds": seconds,
}
if output["censored"] < n - 1:
    result = Result(replicates)
    for key, h in (("first", lambda x: x), ("second", lambda x: x**2)):
        estimate = result.estimate(h)
        output[key] = {
            "value": estimate.value.tolist(),
            "error": estimate.standard_error.tolist(),
            "replicates": estimate.replicates.tolist(),
        }
print(json.dumps(output))
"""

# Times one product in this checkout's matvec, by kind, in microseconds per call.
PRODUCT = """
import json, sys, timeit
sys.path.insert(0, sys.argv[1])
import numpy as np
from telescopic import examples
from telescopic._linalg import matvec

d, n = map(int, sys.argv[2:])
matrix = np.linalg.cholesky(examples.gaussian(d).cov)
vectors = np.random.default_rng(3).standard_normal((n, d))

def blas_alone():
    return vectors @ matrix.T

calls = {
    "einsum": lambda: matvec(matrix, vectors, blas=False),
    "BLAS with errstate": lambda: matvec(matrix, vectors, blas=True),
    "BLAS alone": blas_alone,
}
number = max(10, 20000 // (n * d))
times = {}
for _ in range(7):
    for label, call in calls.items():
        times.setdefault(label, []).append(1e6 * timeit.timeit(call, number=number) / number)
print(json.dumps(times))
"""

# Counts shapes where BLAS gives equal rows other bits, under the environment's kernel set
# and threads.
ROWS = """
import json, sys
import numpy as np
import threadpoolctl
sys.path.insert(0, sys.argv[1])
from telescopic.kernels import HMC, hmc
from telescopic.target import Target
from telescopic.testing import random_draws

def dense(d):
    # The tests' dense covariance (tests/references.py).
    a = np.random.default_rng(14).standard_normal((d, d))
    return (a @ a.T + d * np.eye(d)) / (2 * d)

rng = np.random.default_rng(0)
pairs = other = 0
for n in (2, 3, 4, 6, 8, 16, 32, 64, 256):
    for d in (2, 5, 10, 20, 33, 50, 64, 100, 200):
        matrix = np.linalg.cholesky(dense(d))
        half = rng.standard_normal((n, d))
        product = np.concatenate([half, half]) @ matrix.T
        pairs += not np.array_equal(product[:n], product[n:])
        first = rng.standard_normal((n, d))
        second = rng.standard_normal((n, d))
        rows = sorted({0, n // 2, n - 1})
        second[rows] = first[rows]
        other += not np.array_equal((first @ matrix.T)[rows], (second @ matrix.T)[rows])

def one_call(matrix, vectors, *, blas):
    return (vectors.reshape(-1, vectors.shape[-1]) @ matrix.T).reshape(vectors.shape)

target = Target(lambda x: -0.5 * np.sum(x**2, axis=-1), grad=lambda x: -x, vectorize=True)
hmc.matvec = one_call
kernel = HMC(0.3, 4, cov=dense(33), meeting_weight=0.3, meeting_scale=0.5)
points = np.random.default_rng(23).standard_normal((3, 33))
x, y = kernel.init_state(target, points), kernel.init_state(target, points.copy())
separated = False
for step in range(5):
    x, y = kernel.coupled_step(target, x, y, random_draws(3, step))
    separated |= not np.array_equal(x.x, y.x)
print(json.dumps({
    "kernels": threadpoolctl.threadpool_info()[0]["architecture"],
    "pairs": pairs,
    "other": other,
    "separated": separated,
}))
"""


def child(script, *arguments, env=THREADS):
    output = subprocess.run(
        [sys.executable, "-I", "-c", script, *map(str, arguments)],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ | env,
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
    print("\n1. Moved numbers: 64 replicates, k = 10, m = 50, lag 1, max_iter = 20000, rng = 1")
    runs = {}
    for name, dimensions in KERNELS.items():
        for d in dimensions:
            for label, source in sources.items():
                for lanes in (64, 1, 7):
                    for rowwise in (0, 1):
                        runs[label, name, d, lanes, rowwise] = child(
                            CHILD, source, name, d, 64, 10, 50, 20000, 1, lanes, rowwise
                        )
                        assert runs[label, name, d, lanes, rowwise]["censored"] == 0
            before, after = runs["before", name, d, 64, 0], runs["after", name, d, 64, 0]
            differ = int(np.sum(np.array(before["meeting_time"]) != after["meeting_time"]))
            second = np.array([before["second"]["value"], after["second"]["value"]])
            first = np.array([before["first"]["value"], after["first"]["value"]])
            relative = np.max(np.abs(second[1] - second[0]) / np.abs(second[0]))
            in_errors = np.max(np.abs(first[1] - first[0]) / before["first"]["error"])
            print(
                f"{name:18} d = {d:3}: hash {before['hash']} -> {after['hash']}; meeting "
                f"times that differ {differ:2} of 64, censored {before['censored']} and "
                f"{after['censored']}; E[x^2] largest relative change {relative:.1e}; E[x] "
                f"largest change {in_errors:.1e} standard errors"
            )
    return runs


def layouts(runs):
    print(
        "\n2. Layouts: the runs of section 1 with lanes = 1 and 7 against lanes = 64, for "
        "the target of section 1 (BLAS) and for one whose products are einsum's (row-wise)"
    )
    for rowwise, target in ((1, "row-wise"), (0, "BLAS")):
        for name, dimensions in KERNELS.items():
            for d in dimensions:
                cells = []
                for label in ("before", "after"):
                    reference = runs[label, name, d, 64, rowwise]
                    for lanes in (1, 7):
                        other = runs[label, name, d, lanes, rowwise]
                        differ = np.sum(
                            np.array(other["meeting_time"]) != reference["meeting_time"]
                        )
                        replicate = np.max(
                            np.abs(
                                np.array(other["first"]["replicates"])
                                - reference["first"]["replicates"]
                            )
                        )
                        pooled = np.max(
                            np.abs(np.array(other["first"]["value"]) - reference["first"]["value"])
                            / reference["first"]["error"]
                        )
                        cells.append(
                            f"{label} lanes {lanes}: {differ:2} meeting times, replicate "
                            f"{replicate:.1e}, pooled {pooled:.1e} s.e."
                        )
                print(f"{target:8} {name:18} d = {d:3}: " + "; ".join(cells))


def gain(sources):
    print(
        f"\n3. Gain: 256 replicates, k = 10, m = 200, lag 1, max_iter = 1000, rng = 2; "
        f"{ROUNDS} rounds after one uncounted"
    )
    for name in ("RandomWalk", "MALA"):
        for d in (20, 200):
            seconds = {label: [] for label in sources}
            for round_ in range(ROUNDS + 1):
                order = list(sources) if round_ % 2 == 0 else list(reversed(sources))
                for label in order:
                    result = child(CHILD, sources[label], name, d, 256, 10, 200, 1000, 2, 256, 0)
                    if round_:
                        seconds[label].append(result["seconds"])
            ratio = np.array(seconds["before"]) / np.array(seconds["after"])
            print(
                f"{name:10} d = {d:3}: censored {result['censored']:3} of 256; processor "
                f"seconds median {np.median(seconds['before']):.2f} before, "
                f"{np.median(seconds['after']):.2f} after; before/after per round: median "
                f"{np.median(ratio):.2f}, range {ratio.min():.2f} to {ratio.max():.2f}"
            )


def products():
    print("\n4. One product: microseconds per call, median (range) of 7 repeats")
    for d in (2, 10, 20, 200):
        for n in (1, 16, 256):
            times = child(PRODUCT, ROOT / "src", d, n)
            cells = [
                f"{label} {np.median(t):.2f} ({min(t):.2f} to {max(t):.2f})"
                for label, t in times.items()
            ]
            print(f"d = {d:3}, {n:3} rows: " + "; ".join(cells))


def equal_rows():
    print(
        "\n5. Equal rows over 81 shapes: (a) shapes where two equal rows of one call get "
        "different bits; (b) shapes where a row gets other bits in another call of the "
        "same shape; (c) the HMC test's configuration with both chains in one call"
    )
    for coretype in ("", "Haswell", "SkylakeX", "Sandybridge", "Nehalem", "Zen"):
        for threads in ("1", "4"):
            env = {"OPENBLAS_CORETYPE": coretype, "OPENBLAS_NUM_THREADS": threads}
            found = child(ROWS, ROOT / "src", env=THREADS | env)
            print(
                f"OPENBLAS_CORETYPE={coretype or '(unset)':11} kernels {found['kernels']:11} "
                f"threads {threads}: (a) {found['pairs']:2} of 81; (b) {found['other']:2} of "
                f"81; (c) {'separated' if found['separated'] else 'stayed equal'}"
            )


def main():
    print(setup())
    print(f"before: commit {BEFORE}; after: this checkout; children run with {THREADS}")
    print(f"load average before: {tuple(round(x, 2) for x in os.getloadavg())}")
    with tempfile.TemporaryDirectory() as folder:
        sources = {"before": extract(BEFORE, folder), "after": ROOT / "src"}
        layouts(moved(sources))
        gain(sources)
    products()
    equal_rows()
    print(f"load average after: {tuple(round(x, 2) for x in os.getloadavg())}")


if __name__ == "__main__":
    main()

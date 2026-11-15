"""What does recording moments cost per atom? (issue #79, part A, risk R1)

Question. With keep, the engine adds w h_c(x) to each replicate's moments for every atom
it records: p = d(d + 3)/2 sums up to d = 100 and 2d above, against the 8d + 20 bytes of
a stored row. The design left this cost unmeasured.

Design. engine.run on N(0, I) in d = 2, 10 and 100, vectorized, with an independence
kernel whose proposal is the target, so that every pair meets at its first coupled step:
each replicate records its |S| = 201 averaged atoms and the 2 atoms of the correction at
t = 1, where X_1 differs from the independently drawn Y_0 (k = 0, m = 200, lag 1); 512
replicates on 256 lanes, 103,936 atoms. Three modes: keep=None (rows, no moments),
keep=512 (rows and moments of every replicate) and keep=0 (moments only). Each mode's
time is the median of 7 runs, interleaved; the cost of the moments per atom is
(keep=512 - keep=None) / atoms, and a mode's relative time is its time over keep=None's.
BLAS is pinned to one thread. The script prints its setup.

Also computed, for the documentation: the sums of 256 lanes, 256 x engine._lane_bytes(d)
bytes, and the stored rows per replicate above which keep saves memory,
(p + d) / (d + 2.5), from 8 (p + d) bytes of moments against 8d + 20 per row.

Result (r1_moment_cost.out, at commit 4ad7fdc). The moments cost 48 ns per atom at
d = 2 (a run 1.04 times as long as without them), 687 ns at d = 10 (1.43 times) and
33,958 ns at d = 100 (3.45 times), 6.6 ns for each of its 5,150 sums. A run that keeps
no rows (keep=0) takes 1.04, 1.30 and 3.40 times as long, so the moments, not the rows,
set the cost. An earlier run of the same script, before this docstring changed, gave 169,
934 and 35,692 ns (1.15, 1.57 and 3.59 times): the figure at d = 2 is within the noise of
a 0.13 s run. The lane buffer is 0.01, 0.15 and 10.75 MB, and keep saves memory once a
replicate stores more than 1.6, 6.0 and 51.2 rows.
"""

import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
import threadpoolctl

import telescopic
from telescopic import engine
from telescopic.initial import Normal
from telescopic.kernels import Independence, MultivariateT
from telescopic.target import Target

REPLICATES, REPEATS = 512, 7


def log_prob(x):
    return -0.5 * np.sum(np.square(x), axis=-1)


def setup():
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    model = platform.processor()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    blas = [
        (library["internal_api"], library["version"], library["num_threads"])
        for library in threadpoolctl.threadpool_info()
    ]
    return (
        f"commit {commit}; {model}, {os.cpu_count()} CPUs, {platform.system()}; "
        f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy "
        f"{scipy.__version__}, telescopic {telescopic.__version__}; BLAS {blas}"
    )


def timed(ndim, keep):
    target = Target(log_prob, vectorize=True)
    kernel = Independence(MultivariateT(np.zeros(ndim), np.eye(ndim), df=None))
    initial = Normal(np.zeros(ndim), np.eye(ndim))
    start = time.perf_counter()
    replicates = engine.run(
        target, kernel, initial, range(REPLICATES), k=0, m=200, lag=1, keep=keep, rng=1
    )
    return time.perf_counter() - start, replicates


def main():
    with threadpoolctl.threadpool_limits(1):
        print(setup())
        print(
            f"{'d':>4} {'p':>6} {'atoms':>8} {'none s':>8} {'all s':>8} {'0 s':>8} "
            f"{'ns/atom':>8} {'all/none':>8} {'0/none':>8} {'buffer MB':>10} {'rows':>6}"
        )
        for ndim in (2, 10, 100):
            times = {None: [], REPLICATES: [], 0: []}
            for _ in range(REPEATS):
                for keep, measured in times.items():
                    seconds, replicates = timed(ndim, keep)
                    measured.append(seconds)
                    if keep is None:
                        atoms = len(replicates.weights)
            none, every, zero = (float(np.median(times[keep])) for keep in times)
            p = replicates.moments.shape[1]
            print(
                f"{ndim:>4} {p:>6} {atoms:>8} {none:>8.3f} {every:>8.3f} {zero:>8.3f} "
                f"{(every - none) / atoms * 1e9:>8.0f} {every / none:>8.2f} "
                f"{zero / none:>8.2f} {256 * engine._lane_bytes(ndim) / 1e6:>10.2f} "
                f"{(p + ndim) / (ndim + 2.5):>6.1f}"
            )


if __name__ == "__main__":
    main()

"""The peak memory of a run and of its estimates against the bytes it stores (issue #73).

Question: how many times the stored atoms and distances does a run take at its peak, and
again its first estimates, so that the sampler's memory warning can compare a peak with
its threshold, and where does that factor not hold? Each case runs in a fresh process: a
sampler with `RandomWalk(s I)` on a vectorized N(0, I) in d dimensions, from N(0, v I),
with k = 200, m = 2000 and R replicates, then `cov()` and `summary()`. The first cases
take s = 1 and v = 4 at `thin` = 1 and R = 200 (the issue's setting at d = 50); the next
take `thin` 2 and 10, with R raised so that the stored size stays near the first cases';
the last take s = v = 0.01, a walk that rarely rejects. `tracemalloc` traces NumPy's
allocations; each peak is above what the process held before `run()`, with SciPy's
`stats` and `optimize` imported beforehand, since the run and its estimates import them
at first use, once in a process. "stored" is the
`nbytes` of the six columns that `engine.stored_bytes` counts, and "estimate" that
function on the run's own meeting times; one case takes `k` from the sampler's pilot,
whose meeting times stand in for the run's in the sampler's estimate.

Result (in the .out file beside this script): `stored_bytes` on the run's meeting times
equals the stored bytes, and on the pilot's 0.998 of them. At `thin` = 1 with s = 1, the
run's peak is 2.0 to 2.7 times the stored size and the estimates' 2.0 to 2.5 times from
d = 2 to 50, and 3.3 times at d = 1, where the `Result`'s own 16 bytes per atom (its
replicate index and, after the first estimate, the column indices of its sums) weigh
most against the 24 stored. With `thin` above 1, or the walk that rarely rejects, the
estimates peak at up to about 5.3 times at d = 1 and 4.2 at d = 2, and about 2.0 at
d = 10: `Result.quantile` sorts the atoms that are not consecutive repeats, and these
runs repeat few. Before issue #75, which lowered the peaks of `cov` and `quantile` and
the sums' bytes, the estimates peaked at 2.9 to 3.0, 3.8, 5.9, 4.9 and 2.8 times.

Run: python notes/experiments/i73_memory_peak.py
"""

import json
import os
import platform
import subprocess
import sys

import numpy as np
import scipy

CASE = r"""
import json, sys, tracemalloc, warnings
import numpy as np
from telescopic import CoupledSampler, engine
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk

d, k, thin, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
step, spread = float(sys.argv[5]), float(sys.argv[6])
k = "auto" if k == "auto" else int(k)
log_prob = lambda x: -0.5 * np.sum(x * x, axis=1)
sampler = CoupledSampler(log_prob, Normal(np.zeros(d), spread * np.eye(d)), vectorize=True,
                         kernel=RandomWalk(step * np.eye(d)), rng=1)
if k == "auto":
    sampler.meeting_times(100)
warnings.simplefilter("ignore")
# Imported before tracing: the run and its estimates import them at first use, once in a
# process, which is not memory of the run.
import scipy.optimize, scipy.stats
tracemalloc.start()
base = tracemalloc.get_traced_memory()[0]
result = sampler.run(n, k=k, m=2000 if k != "auto" else "auto", thin=thin)
after_run = tracemalloc.get_traced_memory()
tracemalloc.reset_peak()
result.cov()
result.summary()
after_estimates = tracemalloc.get_traced_memory()
replicates = result._replicates
names = ("atoms", "weights", "owner", "distance_owner", "distance_time", "distance")
stored = sum(getattr(replicates, name).nbytes for name in names)
settings = replicates.settings
coupled = replicates.meeting_time - settings.lag
own = engine.stored_bytes(settings, d, coupled)
pilot = sampler.pilot[np.isfinite(sampler.pilot)] - 1 if sampler.pilot is not None else None
from_pilot = None if pilot is None else n * engine.stored_bytes(settings, d, pilot) / len(pilot)
print(json.dumps({
    "d": d, "k": result.k, "m": result.m, "thin": thin, "n": n, "step": step, "stored": stored, "estimate": own,
    "from_pilot": from_pilot, "run_peak": after_run[1] - base,
    "estimates_peak": after_estimates[1] - base,
}))
"""

commit = subprocess.run(
    ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
)
print(f"commit {commit.stdout.strip()}")
print(f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
print(f"machine: {platform.system()} {platform.machine()}, {os.cpu_count()} cpus")
print("threads: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1")
environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
print(
    f"{'d':>3} {'k':>5} {'m':>5} {'thin':>4} {'R':>5} {'step':>5} {'stored MB':>10} "
    f"{'estimate':>9} {'pilot est.':>10} {'run peak':>9} {'estimates peak':>15}"
)
# d, k, thin, R, the random walk's variance per coordinate, the initial distribution's.
CASES = [
    (1, "200", 1, 200, 1.0, 4.0),
    (2, "200", 1, 200, 1.0, 4.0),
    (10, "200", 1, 200, 1.0, 4.0),
    (50, "200", 1, 200, 1.0, 4.0),
    (10, "auto", 1, 200, 1.0, 4.0),
    (1, "200", 2, 400, 1.0, 4.0),
    (1, "200", 10, 2000, 1.0, 4.0),
    (2, "200", 2, 400, 1.0, 4.0),
    (2, "200", 10, 2000, 1.0, 4.0),
    (10, "200", 10, 2000, 1.0, 4.0),
    (1, "200", 1, 200, 0.01, 0.01),
    (2, "200", 1, 200, 0.01, 0.01),
]
for d, k, thin, n, step, spread in CASES:
    output = subprocess.run(
        [sys.executable, "-c", CASE, str(d), k, str(thin), str(n), str(step), str(spread)],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    row = json.loads(output.stdout)
    stored = row["stored"]
    pilot = "" if row["from_pilot"] is None else f"{row['from_pilot'] / stored:.3f}x"
    print(
        f"{row['d']:>3} {row['k']:>5} {row['m']:>5} {thin:>4} {n:>5} {step:>5g} "
        f"{stored / 1e6:>10.1f} {row['estimate'] / stored:>8.3f}x {pilot:>10} "
        f"{row['run_peak'] / stored:>8.2f}x {row['estimates_peak'] / stored:>14.2f}x",
        flush=True,
    )

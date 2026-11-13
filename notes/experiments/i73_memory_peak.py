"""The peak memory of a run and of its estimates against the bytes it stores (issue #73).

Question: how many times the stored atoms and distances does a run take at its peak, and
again its first estimates, so that the sampler's memory warning can compare a peak with
its threshold? Each case runs in a fresh process: a sampler with `RandomWalk(I)` on a
vectorized N(0, I) in d dimensions with k = 200, m = 2000 and R = 200 (the issue's
setting at d = 50), then `cov()` and `summary()`. `tracemalloc` traces NumPy's
allocations; each peak is above what the process held before `run()`. "stored" is the
`nbytes` of the six columns that `engine.stored_bytes` counts, and "estimate" that
function on the run's own meeting times; a last case takes `k` from the sampler's pilot,
whose meeting times stand in for the run's in the sampler's estimate.

Result (in the .out file beside this script): `stored_bytes` on the run's meeting times
equals the stored bytes, and on the pilot's 0.998 of them. The run's peak is 2.0 to 2.7
times the stored size and the estimates' 2.9 to 3.0 times from d = 2 to 50, and 3.8 times
at d = 1, where the `Result`'s own 24 bytes per atom (its replicate index and, after the
first estimate, two more columns) weigh most against the 24 stored.

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

d, k = int(sys.argv[1]), sys.argv[2]
k = "auto" if k == "auto" else int(k)
log_prob = lambda x: -0.5 * np.sum(x * x, axis=1)
sampler = CoupledSampler(log_prob, Normal(np.zeros(d), 4 * np.eye(d)), vectorize=True,
                         kernel=RandomWalk(np.eye(d)), rng=1)
if k == "auto":
    sampler.meeting_times(100)
warnings.simplefilter("ignore")
tracemalloc.start()
base = tracemalloc.get_traced_memory()[0]
result = sampler.run(200, k=k, m=2000 if k != "auto" else "auto")
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
from_pilot = None if pilot is None else 200 * engine.stored_bytes(settings, d, pilot) / len(pilot)
print(json.dumps({
    "d": d, "k": result.k, "m": result.m, "stored": stored, "estimate": own,
    "from_pilot": from_pilot, "run_peak": after_run[1] - base,
    "estimates_peak": after_estimates[1] - base,
}))
"""

print(f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
print(f"machine: {platform.machine()}, {os.cpu_count()} cpus; OMP_NUM_THREADS=1")
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
)
print(f"commit {commit.stdout.strip()} with the working tree's changes")
environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
print(
    f"{'d':>3} {'k':>5} {'m':>5} {'stored MB':>10} {'estimate':>9} {'pilot est.':>10} "
    f"{'run peak':>9} {'estimates peak':>15}"
)
for d, k in [(1, "200"), (2, "200"), (10, "200"), (50, "200"), (10, "auto")]:
    output = subprocess.run(
        [sys.executable, "-c", CASE, str(d), k],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    row = json.loads(output.stdout)
    stored = row["stored"]
    pilot = "" if row["from_pilot"] is None else f"{row['from_pilot'] / stored:.3f}x"
    print(
        f"{row['d']:>3} {row['k']:>5} {row['m']:>5} {stored / 1e6:>10.1f} "
        f"{row['estimate'] / stored:>8.3f}x {pilot:>10} {row['run_peak'] / stored:>8.2f}x "
        f"{row['estimates_peak'] / stored:>14.2f}x"
    )

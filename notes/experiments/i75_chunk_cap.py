"""What the cap of 1,024 replicates per chunk changes in a pool's chunks, time and payload (#75).

Question: the cap of `_parallel.MAX_CHUNK` replicates per chunk binds only above 4,096
replicates per worker. Which chunks does it give for the cases of DECISIONS #23 and #52
and for the issue's large runs, what does a run cost with more, smaller chunks, and how
large is the result a chunk sends back?

Part 1 records the size of every chunk submitted to an executor that runs only each
chunk's first replicate, with the cap and with it lifted. Part 2 runs 8,192 replicates
of a vectorized line fit (k = 87, m = 870, lag = 87, as in `m11_chunks.py`) on an
executor of one thread, BLAS on one thread, so that the cap makes 8 chunks of 1,024
where four per worker make 4 of 2,048; the best of 3 runs of each, alternated, and the
largest pickled result of a chunk.

Result (the output beside this file): the chunks are unchanged for every case of #23
and #52 and up to 4,096 replicates per worker; one replicate more gives one chunk more.
The issue's runs get chunks of at most 1,024 replicates: 98 chunks on 4 workers at
100,000 replicates, where four per worker made 16 of 6,250, and 977 on 4 or 64 workers
at 1,000,000, where they made 16 of 62,500 and 256 of 3,906 or 3,907. The line fit took
8.72 s at best with the cap against 8.16 s without, 1.07 times as long, on a machine
shared with other jobs, where the three runs of each spread over 8.72 to 9.11 s and 8.16
to 8.76 s, ranges that meet. Its largest chunk's pickled result fell from 52.9 to
26.5 MB, in proportion to its replicates (25.8 kB per replicate in this setup).

Run from the repository: python notes/experiments/i75_chunk_cap.py
"""

import functools
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from telescopic import _parallel, engine
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.target import Target

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from run import setup

UNCAPPED = sys.maxsize

rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)
LINE_COV = np.linalg.inv(
    [[np.sum(x**2 / yerr**2), np.sum(x / yerr**2)], [np.sum(x / yerr**2), np.sum(1 / yerr**2)]]
)


def line(theta):
    slope, intercept = theta[..., :1], theta[..., 1:]
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2, axis=-1)


def gaussian(z):
    return -0.5 * float(z @ z)


def never(status):
    return False


class Sampling(ThreadPoolExecutor):
    # Records each chunk's size and runs only its first replicate.
    def __init__(self, workers):
        super().__init__(workers)
        self.chunks = []

    def submit(self, function, piece, **options):
        self.chunks.append(len(piece))
        return super().submit(function, piece[:1], **options)


class Measuring(ThreadPoolExecutor):
    # Records the pickled size of each chunk's result.
    def __init__(self, workers):
        super().__init__(workers)
        self.payloads = []

    def submit(self, function, piece, **options):
        future = super().submit(function, piece, **options)
        future.add_done_callback(
            lambda done: self.payloads.append(len(pickle.dumps(done.result())))
        )
        return future


def chunks(n, workers, cap):
    _parallel.MAX_CHUNK = cap
    job = functools.partial(
        engine.run,
        Target(gaussian),
        RandomWalk(np.eye(2)),
        Normal([0.0, 0.0], [1.0, 1.0]),
        rng=1,
        k=0,
        m=0,
        lag=1,
    )
    with Sampling(workers) as executor:
        _parallel.execute(job, np.arange(n), executor, never)
    sizes = executor.chunks
    return f"{len(sizes)} of {min(sizes)} to {max(sizes)}"


def timed(cap):
    _parallel.MAX_CHUNK = cap
    job = functools.partial(
        engine.run,
        Target(line, vectorize=True),
        RandomWalk(LINE_COV),
        Normal([-0.9, 4.3], [1e-2, 0.25]),
        rng=2,
        k=87,
        m=870,
        lag=87,
    )
    with Measuring(1) as executor:
        start = time.perf_counter()
        _parallel.execute(job, np.arange(8_192), executor, never)
        seconds = time.perf_counter() - start
    return seconds, max(executor.payloads), len(executor.payloads)


def main():
    with threadpool_limits(1):
        print(json.dumps(setup()), flush=True)
    print("\nPart 1: chunks (number of chunks of min to max replicates)")
    print(f"{'replicates':>10} {'workers':>7}  {'without the cap':<24} with the cap of 1,024")
    cases = [
        (100, 4),
        (200, 4),
        (1_000, 1),
        (1_000, 4),
        (2_000, 4),
        (4_096, 1),
        (4_096, 2),
        (4_097, 1),
        (16_384, 4),
        (16_385, 4),
        (100_000, 4),
        (1_000_000, 4),
        (1_000_000, 64),
    ]
    for n, workers in cases:
        print(
            f"{n:>10,} {workers:>7}  {chunks(n, workers, UNCAPPED):<24} {chunks(n, workers, 1_024)}"
        )

    print("\nPart 2: 8,192 replicates of the line fit on one thread")
    runs = {UNCAPPED: [], 1_024: []}
    with threadpool_limits(1):
        for _ in range(3):
            for cap, results in runs.items():
                results.append(timed(cap))
    for cap, results in runs.items():
        label = "without the cap" if cap == UNCAPPED else "with the cap"
        seconds = [r[0] for r in results]
        _, payload, count = results[0]
        print(
            f"{label:<16} {count} chunks, seconds {', '.join(f'{s:.2f}' for s in seconds)} "
            f"(best {min(seconds):.2f}), largest pickled result {payload:,} bytes"
        )
    ratio = min(r[0] for r in runs[1_024]) / min(r[0] for r in runs[UNCAPPED])
    print(f"best with the cap / best without: {ratio:.2f}")


if __name__ == "__main__":
    main()

"""How many chunks run after a stop on a user's process pool (issue #71, DECISIONS #95).

`_parallel._gather` submits every chunk at once and, once the monitor asks to stop,
cancels the chunks that have not started. CPython's `ProcessPoolExecutor` moves pending
calls into a call queue of `max_workers + EXTRA_QUEUED_CALLS` slots, filling it whenever
its manager thread wakes, at a submission or when a call ends; a call in that queue
counts as running, so its `cancel()` fails. A user's pool has no stop event, so every
such chunk runs all its replicates. This runs `_gather` on a user's
`ProcessPoolExecutor(3)` with 24 chunks of 1 s each, the monitor asking to stop at 0.5,
1.5 and 2.5 s, three times each, and reports the chunks that started after the stop and
the seconds between the end of the chunks running at the stop and the return.

Result: a stop during the first wave let 1 chunk run, as in the issue's measurement,
since the workers took 3 of the 4 queued calls and no call had ended to wake the
manager. A stop in a later wave let 2 to 4 chunks run, up to the full queue of
`max_workers + 1`, and the run returned 1.0 or 2.0 s, one or two chunk durations, after
the chunks running at the stop had ended.

Run: python notes/experiments/i71_pool_stop.py
"""

import functools
import platform
import subprocess
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import process as futures_process
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from telescopic import _parallel

WORKERS, CHUNKS, SECONDS, REPEATS = 3, 24, 1.0, 3


def task(piece, first):
    # A chunk of one replicate that takes SECONDS, with the time it started.
    started = time.time()
    time.sleep(SECONDS)
    return types.SimpleNamespace(index=piece, started=started)


def past(deadline, status):
    # The monitor: stop once the deadline has passed.
    return time.time() > deadline


def main():
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(
        f"commit {commit}, Python {platform.python_version()}, {platform.machine()}, "
        f"EXTRA_QUEUED_CALLS = {futures_process.EXTRA_QUEUED_CALLS}"
    )
    print(f"ProcessPoolExecutor({WORKERS}), {CHUNKS} chunks of {SECONDS} s")
    for stop in [0.5, 1.5, 2.5] * REPEATS:
        pieces = [np.array([i]) for i in range(CHUNKS)]
        with ProcessPoolExecutor(WORKERS) as executor:
            # Start the workers first, so that the waves begin together.
            list(executor.map(abs, range(WORKERS)))
            start = time.time()
            monitor = functools.partial(past, start + stop)
            parts = _parallel._gather(executor, task, pieces, monitor, None)
            returned = time.time() - start
        started = {int(part.index[0]): part.started - start for part in parts}
        running = sorted(i for i, t in started.items() if t <= stop < t + SECONDS)
        after = sorted(i for i, t in started.items() if t > stop)
        ended = max(started[i] for i in running) + SECONDS
        print(
            f"stop at {stop} s: running {running}; started after the stop {after}; "
            f"returned at {returned:.1f} s, {returned - ended:.1f} s after the running "
            "chunks ended"
        )


if __name__ == "__main__":
    main()

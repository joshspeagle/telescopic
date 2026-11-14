"""Does an error from telescopic's workers leave their semaphores to the collector (#132)?

On 2026-11-14 CI's job on Python 3.12 failed in both runs of one commit in
`test_a_stop_reaches_the_replicates_of_telescopics_workers`, with five warnings that the
resource tracker was called reentrantly, raised by finalizers of semaphores. On Python
3.11 and 3.12 that warning comes from a semaphore that the cyclic garbage collector frees
while the same thread is inside a call of the resource tracker, as when a new semaphore
is registered.

This script makes such a collection happen on demand: the collector is off, and each
call of the tracker collects as it checks that its process is alive, a stand-in for a
collection that an allocation triggers there. It runs `execute` on one worker that
`opened` starts, with a log-density that returns NaN, lets the error pass `opened`, drops
it, and says whether the `Workers` object was freed then. It then makes a new event,
whose registration runs the collection, and counts the reentrant-call warnings.

Result (the output beside this file, where "with changes to the package" is the change of
issue #132): at 4d71634 the `Workers` object outlives the error, in a cycle through the
frame of `_parallel._gather` and the future that holds the error, and the new event's
registration frees the five semaphores of the workers' stop event inside the tracker,
with five warnings, on Python 3.12.13 and 3.11.18. With `_gather` dropping its futures,
the `Workers` object is freed with the error and no warning comes. On Python 3.13.18 the
workers outlive the error in the same way, but the tracker queues the reentrant call
without a warning.

Run: python notes/experiments/i132_cleanup_warning.py
"""

import functools
import gc
import multiprocessing
import platform
import subprocess
import warnings
import weakref
from multiprocessing import resource_tracker
from pathlib import Path

import numpy as np

import telescopic
from telescopic import engine
from telescopic._parallel import execute, opened
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.target import Target


def nan_log_prob(x):
    return np.nan


def never(status):
    return False


def collect_inside_the_tracker():
    tracker = resource_tracker._resource_tracker
    check_alive = tracker._check_alive

    def check():
        gc.collect()
        return check_alive()

    tracker._check_alive = check


def failed_workers():
    # A weak reference to the workers of a run whose worker raised, once the error is gone.
    job = functools.partial(
        engine.run,
        Target(nan_log_prob),
        RandomWalk(np.eye(2)),
        Normal([0.0, 0.0], [1.0, 1.0]),
        rng=0,
        k=3,
        m=30,
        lag=3,
        max_iter=1000,
    )
    try:
        with opened(1) as workers:
            freed = weakref.ref(workers)
            execute(job, np.arange(8), workers, never)
    except ValueError:
        pass
    return freed


def main():
    source = Path(telescopic.__file__).parent
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--", "."],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(
        f"commit {commit}{' with changes to the package' if changed else ''}; "
        f"Python {platform.python_version()}"
    )
    collect_inside_the_tracker()
    multiprocessing.get_context("spawn").Lock()  # starts the tracker's process
    gc.collect()
    gc.disable()
    freed = failed_workers()
    print(f"workers freed with the error: {freed() is None}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        multiprocessing.get_context("spawn").Event()
    reentrant = [w for w in caught if "called reentrantly" in str(w.message)]
    print(f"reentrant-call warnings while a new event registers: {len(reentrant)}")


if __name__ == "__main__":
    main()

"""Runs of the engine in this process or on pools of worker processes."""

from __future__ import annotations

import functools
import importlib
import math
import multiprocessing
import os
import pickle
import queue
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np

from telescopic import engine
from telescopic._types import IntArray

# Imported by name: threadpoolctl and cloudpickle ship no type information.
threadpoolctl = importlib.import_module("threadpoolctl")

# At most this many chunks per worker, so that the workers finish together.
CHUNKS_PER_WORKER = 4

# The most processes that ProcessPoolExecutor accepts on Windows; elsewhere it has no limit.
MAX_PROCESSES = 61 if sys.platform == "win32" else sys.maxsize

# Seconds between the parent's calls of the monitor while it waits on a pool; Ctrl-C needs
# them on Windows, where a blocked wait does not see it.
POLL = 0.1

Job = Callable[..., engine.Replicates]
Monitor = Callable[[engine.Status], bool]


class Interrupt:
    # Ctrl-C during a run: the first sets stopping, the second raises at once.
    def __init__(self) -> None:
        self.stopping = False

    @contextmanager
    def catching(self) -> Iterator[None]:
        # Python sets signal handlers only in the main thread; elsewhere Ctrl-C keeps its
        # usual effect.
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        previous = signal.signal(signal.SIGINT, self._handle)
        try:
            yield
        finally:
            signal.signal(
                signal.SIGINT, signal.default_int_handler if previous is None else previous
            )

    def _handle(self, signum: int, frame: Any) -> None:
        if self.stopping:
            raise KeyboardInterrupt
        self.stopping = True
        print(
            "telescopic: finishing the replicates in progress; press Ctrl-C again to stop "
            "at once, without a result",
            file=sys.stderr,
        )


class Workers:
    # Worker processes that telescopic starts. Spawned ones behave alike on every
    # platform, and forking a process whose BLAS runs threads can deadlock. They ignore
    # Ctrl-C; the event tells them to start no new replicate.
    def __init__(self, n: int) -> None:
        context = multiprocessing.get_context("spawn")
        self.n, self.stop = n, context.Event()
        self.executor = ProcessPoolExecutor(
            n, context, initializer=_start_worker, initargs=(self.stop,)
        )

    def start(self) -> None:
        # Now, so that their start overlaps the parent's tuning. A process made while the
        # parent ignores Ctrl-C starts ignoring it on POSIX, so a Ctrl-C while it imports
        # does not kill it.
        with _ctrl_c_ignored():
            for _ in range(self.n):
                self.executor.submit(int)

    def terminate(self) -> None:
        # Python 3.14 has terminate_workers; before it, the processes are in _processes.
        terminate = getattr(self.executor, "terminate_workers", None)
        if terminate is not None:
            terminate()
            return
        for process in list((self.executor._processes or {}).values()):
            process.terminate()
        self.executor.shutdown(wait=False, cancel_futures=True)


def check_pool(pool: Any) -> None:
    if isinstance(pool, bool) or not (
        pool is None
        or isinstance(pool, int | np.integer)
        or hasattr(pool, "submit")
        or hasattr(pool, "map")
    ):
        raise TypeError(
            "pool must be None, a number of processes, an executor with submit, or an "
            f"object with map, not {type(pool).__name__}"
        )
    if isinstance(pool, int | np.integer) and not 1 <= pool <= MAX_PROCESSES:
        raise ValueError(f"pool must be from 1 to {MAX_PROCESSES} processes, not {pool}")


@contextmanager
def opened(pool: Any) -> Iterator[Any]:
    # The pool as execute takes it: for a number, worker processes that live for the
    # block, stopped at once by an error or a second Ctrl-C.
    if not isinstance(pool, int | np.integer):
        yield pool
        return
    workers = Workers(int(pool))
    try:
        workers.start()
        yield workers
        workers.executor.shutdown()
    except BaseException:
        workers.terminate()
        raise


def execute(job: Job, indices: IntArray, pool: Any, monitor: Monitor) -> engine.Replicates:
    # The job on sorted indices, here or on an opened pool. Here the monitor sees every
    # iteration; on a pool, the finished replicates after each chunk and every POLL
    # seconds. Once it returns true, no chunk starts but the first, and telescopic's
    # workers start no new replicate (spec sections 6.4 and 6.5).
    if pool is None:
        return job(indices, monitor=monitor)
    start = time.perf_counter()
    workers = pool.n if isinstance(pool, Workers) else _workers(pool)
    # Chunks of at least a full set of lanes where there are enough replicates.
    count = max(workers, math.ceil(len(indices) / engine.LANES))
    pieces = np.array_split(indices, min(len(indices), CHUNKS_PER_WORKER * workers, count))
    task = functools.partial(_task, _dumps(job), os.getpid())
    if isinstance(pool, Workers):
        parts = _gather(pool.executor, task, pieces, monitor, pool.stop)
    elif hasattr(pool, "submit"):
        parts = _gather(pool, task, pieces, monitor, None)
    else:
        parts = _waves(pool, task, pieces, workers, monitor)
    # Unstarted chunks are requested too, and the time is the parent's, not the workers'.
    merged = engine.merge(parts)
    return replace(merged, requested=indices, seconds=time.perf_counter() - start)


def _workers(pool: Any) -> int:
    # The pool's size, where it says: an executor's _max_workers, a multiprocessing
    # pool's _processes, a schwimmbad pool's size; else the CPUs this process may use.
    for name in ("_max_workers", "_processes", "size"):
        size = getattr(pool, name, None)
        if isinstance(size, int) and size >= 1:
            return size
    cpus: int | None = getattr(os, "process_cpu_count", os.cpu_count)()
    return cpus or 1


def _gather(
    executor: Executor,
    task: Callable[..., engine.Replicates | None],
    pieces: list[IntArray],
    monitor: Monitor,
    stop: Any,
) -> list[engine.Replicates]:
    # Every chunk is submitted at once, in index order. Once the monitor asks to stop,
    # the chunks not started are cancelled, except the first, so that a run returns a
    # replicate, as the engine's first fill does.
    if stop is not None:
        stop.clear()
    done: queue.SimpleQueue[Future[engine.Replicates | None]] = queue.SimpleQueue()
    futures = [executor.submit(task, piece, first=i == 0) for i, piece in enumerate(pieces)]
    for future in futures:
        future.add_done_callback(done.put)
    parts: list[engine.Replicates] = []
    finished, remaining = 0, len(futures)
    try:
        while remaining:
            try:
                future = done.get(timeout=POLL)
            except queue.Empty:
                pass
            else:
                remaining -= 1
                part = None if future.cancelled() else future.result()
                if part is not None:
                    parts.append(part)
                    finished += len(part.index)
            if monitor(engine.Status(finished, finished, -1, 0)):
                if stop is not None:
                    stop.set()
                for waiting in futures[1:]:
                    waiting.cancel()
    finally:
        for waiting in futures:
            waiting.cancel()
    return parts


def _waves(
    pool: Any,
    task: Callable[..., engine.Replicates | None],
    pieces: list[IntArray],
    workers: int,
    monitor: Monitor,
) -> list[engine.Replicates]:
    # A pool with only map runs waves of one chunk per worker; once the monitor asks to
    # stop, no further wave starts.
    parts: list[engine.Replicates] = []
    finished = 0
    for start in range(0, len(pieces), workers):
        for part in pool.map(task, pieces[start : start + workers]):
            parts.append(part)
            finished += len(part.index)
        if monitor(engine.Status(finished, finished, -1, 0)):
            break
    return parts


def _dumps(job: Job) -> bytes:
    # cloudpickle, when installed, also ships functions defined in notebooks and scripts
    # (spec section 6.4).
    try:
        module = importlib.import_module("cloudpickle")
    except ImportError:
        module = pickle
    try:
        payload: bytes = module.dumps(job)
    except Exception as error:
        raise TypeError(
            "to run on a pool, the target, the kernel and the initial distribution must be "
            "picklable: install cloudpickle (pip install 'telescopic[parallel]'), or "
            "define their functions at the top level of a module"
        ) from error
    return payload


def _load(payload: bytes) -> Job:
    try:
        job: Job = pickle.loads(payload)
    except Exception as error:
        raise RuntimeError(
            "a worker process could not load the target, the kernel or the initial "
            "distribution: install cloudpickle (pip install 'telescopic[parallel]'), or "
            "define their functions in an importable module, with the script's work under "
            "if __name__ == '__main__'"
        ) from error
    return job


# The stop event of a worker that telescopic started; None in any other process.
_stop: Any = None


def _start_worker(stop: Any) -> None:
    # The initializer of telescopic's workers: Ctrl-C is the parent's, also between tasks.
    global _stop
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _stop = stop


def _task(
    payload: bytes, parent: int, indices: IntArray, first: bool = False
) -> engine.Replicates | None:
    # One chunk, in any process. In telescopic's workers, a chunk that starts after the
    # stop runs nothing, except the first, and the others start no replicate after it.
    stop = _stop
    if stop is not None and stop.is_set() and not first:
        return None
    job = _load(payload)
    with _as_worker(parent):
        return job(indices, monitor=None if stop is None else lambda status: stop.is_set())


@contextmanager
def _as_worker(parent: int) -> Iterator[None]:
    # A task in the main thread of another process pins BLAS to one thread and leaves
    # Ctrl-C to the parent (spec sections 6.4 and 6.5). Elsewhere it shares a process
    # whose settings are not its own: concurrent tasks would restore each other's.
    if os.getpid() == parent or threading.current_thread() is not threading.main_thread():
        yield
        return
    # Each task scans the loaded libraries again (2.3 ms), so that one loaded by an earlier
    # task, such as a lazily imported torch, is limited too.
    with _ctrl_c_ignored(), threadpoolctl.threadpool_limits(1):
        yield


@contextmanager
def _ctrl_c_ignored() -> Iterator[None]:
    # Only the main thread may set handlers.
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)

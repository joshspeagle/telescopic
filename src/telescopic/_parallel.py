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
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import BrokenExecutor, Executor, Future, ProcessPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from typing import Any

import numpy as np

from telescopic import engine
from telescopic._types import IntArray, integer

# Imported by name: threadpoolctl and cloudpickle ship no type information.
threadpoolctl = importlib.import_module("threadpoolctl")

# At most this many chunks per worker, so that the workers finish together.
CHUNKS_PER_WORKER = 4

# At most this many replicates per chunk, four sets of lanes, so that a chunk's memory in
# its worker and on its way back, and the steps of a pool's progress, do not grow with the
# run (DECISIONS #179).
MAX_CHUNK = 4 * engine.LANES

# The most processes that ProcessPoolExecutor accepts on Windows; elsewhere it has no limit.
MAX_PROCESSES = 61 if sys.platform == "win32" else sys.maxsize

# Seconds between the parent's calls of the monitor while it waits on a pool; Ctrl-C needs
# them on Windows, where a blocked wait does not see it.
POLL = 0.1

Job = Callable[..., engine._Outcome]
Monitor = Callable[[engine.Status], bool]
Sink = Callable[[engine.Replicates], None]


class Interrupt:
    # Ctrl-C while catching: the first sets stopping, the second raises at once.
    def __init__(self) -> None:
        self.stopping = False

    def catching(self) -> AbstractContextManager[None]:
        return _sigint(self._handle)

    @property
    def listening(self) -> bool:
        # Whether its handler has Ctrl-C now: _sigint keeps a host's, an ignored one's
        # and, outside the main thread, every handler.
        return signal.getsignal(signal.SIGINT) == self._handle

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
        with _sigint(signal.SIG_IGN):
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


def check_pool(pool: Any) -> Any:
    # The pool, with a number of processes as an int.
    if isinstance(pool, type):
        raise TypeError(f"pool must be an executor or a pool, not the class {pool.__name__}")
    if pool is None or hasattr(pool, "submit") or hasattr(pool, "map"):
        return pool
    expected = "None, a number of processes, an executor with submit, or an object with map"
    count = integer("pool", pool, expected, minimum=1)
    if count > MAX_PROCESSES:
        raise ValueError(f"pool must be from 1 to {MAX_PROCESSES} processes, not {count}")
    return count


def check_workers(n_workers: Any, pool: Any) -> int | None:
    # The number of workers that the caller states for a checked pool; it wins over the
    # size the pool reports (DECISIONS #167).
    if n_workers is None:
        return None
    count = integer("n_workers", n_workers, "None or a positive integer", minimum=1)
    if pool is None or isinstance(pool, int):
        runs = "runs in this process" if pool is None else "is already the number of workers"
        raise ValueError(
            f"n_workers is for an executor or a pool of your own, and pool={pool!r} {runs}: "
            "leave n_workers out"
        )
    return count


@contextmanager
def opened(pool: Any) -> Iterator[Any]:
    # The pool as execute takes it: for a number, worker processes that live for the
    # block, stopped at once by an error or a second Ctrl-C.
    if not isinstance(pool, int):
        yield pool
        return
    workers = Workers(pool)
    try:
        workers.start()
        yield workers
        workers.executor.shutdown()
    except BaseException:
        workers.terminate()
        raise


def execute(
    job: Job,
    indices: IntArray,
    pool: Any,
    monitor: Monitor,
    n_workers: int | None = None,
    sink: Sink | None = None,
) -> engine._Outcome:
    # The job on sorted indices, here or on an opened pool, whose workers n_workers counts
    # when given. Here the monitor sees every iteration; on a pool, the finished
    # replicates after each chunk and every POLL seconds. Once it returns true,
    # telescopic's workers start no new replicate, and a chunk that starts on them
    # afterwards runs nothing unless it is the first; a user's executor starts no chunk
    # that it can still cancel (spec sections 6.4 and 6.5). An error stops the run in the
    # same way, and is returned with the replicates finished: the error raised in this
    # process if one was, else that of the lowest-indexed chunk that failed, with the
    # number of the other chunks' errors. A sink is handed records of finished replicates
    # as they come: here, the engine's; on a pool, each chunk's in this process as it
    # returns, and an error of the sink stops the run as the monitor's does.
    if pool is None:
        return job(indices, monitor=monitor, sink=sink)
    start = time.perf_counter()
    workers = _workers(pool, n_workers)
    # Chunks of at most a full set of lanes, where four per worker allow (DECISIONS #52),
    # and of at most MAX_CHUNK replicates (DECISIONS #179).
    n = len(indices)
    count = min(n, CHUNKS_PER_WORKER * workers, max(workers, math.ceil(n / engine.LANES)))
    pieces = np.array_split(indices, max(count, math.ceil(n / MAX_CHUNK)))
    task = functools.partial(_task, _dumps(job), os.getpid())
    if isinstance(pool, Workers):
        outcomes, error = _gather(pool.executor, task, pieces, monitor, pool.stop, sink)
    elif hasattr(pool, "submit"):
        outcomes, error = _gather(pool, task, pieces, monitor, None, sink)
    else:
        outcomes, error = _waves(pool, task, pieces, workers, monitor, sink)
    outcomes.sort(key=lambda pair: pair[0])
    failures = [outcome.error for _, outcome in outcomes if outcome.error is not None]
    if error is None and failures:
        error = failures.pop(0)
    # A broken pool fails every chunk it had: those are one error, the one raised or not.
    others = sum(not isinstance(failure, BrokenExecutor) for failure in failures)
    parts = [outcome.replicates for _, outcome in outcomes if outcome.replicates is not None]
    if not parts:
        return engine._Outcome(None, error, others)
    # Unstarted chunks are requested too, and the time is the parent's, not the workers'.
    merged = replace(engine.merge(parts), requested=indices, seconds=time.perf_counter() - start)
    return engine._Outcome(merged, error, others)


def _workers(pool: Any, n_workers: int | None) -> int:
    # The number of workers: the caller's n_workers, which check_workers allows only with
    # a pool of the caller's own; else telescopic's own count; else the pool's size, where
    # it says: an executor's _max_workers, a multiprocessing pool's _processes, a
    # schwimmbad pool's size; else the CPUs this process may use.
    if n_workers is not None:
        return n_workers
    if isinstance(pool, Workers):
        return pool.n
    for name in ("_max_workers", "_processes", "size"):
        size = getattr(pool, name, None)
        if isinstance(size, int) and size >= 1:
            return size
    cpus: int | None = getattr(os, "process_cpu_count", os.cpu_count)()
    return cpus or 1


def _gather(
    executor: Executor,
    task: Callable[..., engine._Outcome | None],
    pieces: list[IntArray],
    monitor: Monitor,
    stop: Any,
    sink: Sink | None,
) -> tuple[list[tuple[int, engine._Outcome]], Exception | None]:
    # Every chunk is submitted at once, in index order. Once the monitor asks to stop,
    # the chunks not started are canceled, except the first, so that a run returns a
    # replicate, as the engine's first fill does. A process pool keeps up to
    # max_workers + 1 chunks beyond its running ones in its call queue, refilled when a
    # chunk ends, where cancel() fails: the stop event makes them run nothing on
    # telescopic's workers, but on a user's pool, which has none, they run all their
    # replicates (DECISIONS #95). Ctrl-C also kills the idle workers of a user's process
    # pool that does not ignore it, and the pool then fails every chunk left, after any
    # finished one; after a stop those chunks count as not run. An error stops the run
    # alike, but cancels the first chunk too: a chunk's error, a broken pool otherwise,
    # or an exception raised here, by the monitor, which is then no longer called. The
    # running chunks finish, and every outcome returned is kept, with its chunk's
    # position, and with the exception raised here, if any (spec section 6.5).
    done: queue.SimpleQueue[Future[engine._Outcome | None]] = queue.SimpleQueue()
    futures = [executor.submit(task, piece, first=i == 0) for i, piece in enumerate(pieces)]
    position = {future: i for i, future in enumerate(futures)}
    for future in futures:
        future.add_done_callback(done.put)
    outcomes: list[tuple[int, engine._Outcome]] = []
    finished, remaining = 0, len(futures)
    stopping, failed, error = False, False, None
    try:
        while remaining:
            try:
                future = done.get(timeout=POLL)
            except queue.Empty:
                pass
            else:
                remaining -= 1
                outcome, raised = _result(future)
                if raised is not None:
                    broken = isinstance(raised, BrokenExecutor) and bool(outcomes) and not failed
                    if broken:
                        stopping, error = _ask(monitor, finished, stopping, error)
                    if not (broken and stopping):
                        outcome = engine._Outcome(None, raised)
                if outcome is not None:
                    outcomes.append((position[future], outcome))
                    if outcome.replicates is not None:
                        finished += len(outcome.replicates.index)
                        error = _sink(sink, outcome.replicates, error)
                    failed = failed or outcome.error is not None
            if not failed:
                stopping, error = _ask(monitor, finished, stopping, error)
            if stopping or failed or error is not None:
                if stop is not None:
                    stop.set()
                for waiting in futures if failed or error is not None else futures[1:]:
                    waiting.cancel()
    finally:
        for waiting in futures:
            waiting.cancel()
        # A raised error's traceback keeps this frame, and its future keeps the error:
        # dropping the futures lets the workers go with the error, not at a later collection.
        del futures, position, future, waiting
    return outcomes, error


def _result(
    future: Future[engine._Outcome | None],
) -> tuple[engine._Outcome | None, Exception | None]:
    # A done future's outcome, None for a canceled future or a chunk that ran nothing, or
    # the exception it raised: the pool's, since a chunk returns its errors.
    if future.cancelled():
        return None, None
    with engine._Caught() as caught:
        return future.result(), None
    return None, caught.error


def _ask(
    monitor: Monitor, finished: int, stopping: bool, error: Exception | None
) -> tuple[bool, Exception | None]:
    # Whether the monitor asks to stop, and the exception it raised, which stops the run
    # and ends the calls; none after one.
    if error is not None:
        return stopping, error
    with engine._Caught() as caught:
        return monitor(engine.Status(finished, -1, 0)), None
    return stopping, caught.error


def _sink(
    sink: Sink | None, replicates: engine.Replicates, error: Exception | None
) -> Exception | None:
    # Hands a chunk's record to the sink, unless an error already stops the run; the error
    # it raises stops the run as the monitor's does.
    if sink is None or error is not None:
        return error
    with engine._Caught() as caught:
        sink(replicates)
    return caught.error


def _waves(
    pool: Any,
    task: Callable[..., engine._Outcome | None],
    pieces: list[IntArray],
    workers: int,
    monitor: Monitor,
    sink: Sink | None,
) -> tuple[list[tuple[int, engine._Outcome]], Exception | None]:
    # A pool with only map runs waves of one chunk per worker; once the monitor asks to
    # stop, or an error stops the run, no further wave starts. An error of map itself
    # loses the other outcomes of its wave.
    outcomes: list[tuple[int, engine._Outcome]] = []
    finished, error = 0, None
    for start in range(0, len(pieces), workers):
        with engine._Caught() as caught:
            wave = list(pool.map(task, pieces[start : start + workers]))
        if caught.error is not None:
            outcomes.append((start, engine._Outcome(None, caught.error)))
            break
        for offset, outcome in enumerate(wave):
            if outcome is None:
                continue  # a map pool has no stop event, so its chunks always run
            outcomes.append((start + offset, outcome))
            if outcome.replicates is not None:
                finished += len(outcome.replicates.index)
                error = _sink(sink, outcome.replicates, error)
        if error is not None:
            return outcomes, error
        if any(outcome is not None and outcome.error is not None for outcome in wave):
            break
        stopping, error = _ask(monitor, finished, False, None)
        if error is not None:
            return outcomes, error
        if stopping:
            break
    return outcomes, None


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
        advice = (
            "; install cloudpickle (pip install 'telescopic[parallel]'), or define their "
            "functions at the top level of a module"
            if module is pickle
            else ""
        )
        raise TypeError(
            "to run on a pool, the target, the kernel and the initial distribution must be "
            f"picklable: {error}{advice}"
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
) -> engine._Outcome | None:
    # One chunk's outcome, in any process; its error is returned, never raised. In
    # telescopic's workers, a chunk that starts after the stop runs nothing, except the
    # first, and the others start no replicate after it.
    stop = _stop
    if stop is not None and stop.is_set() and not first:
        return None
    outcome = engine._Outcome(None, None)
    with engine._Caught() as caught:
        job = _load(payload)
        with _as_worker(parent):
            outcome = job(indices, monitor=None if stop is None else lambda status: stop.is_set())
    if caught.error is not None:
        outcome = engine._Outcome(None, caught.error)
    if outcome.error is None or os.getpid() == parent:
        return outcome
    return outcome._replace(error=_portable(outcome.error))


def _portable(error: Exception) -> Exception:
    # In a worker process: the error as the parent can unpickle it, or a RuntimeError
    # naming it, which a pool could otherwise turn into a broken pool, with a note that
    # holds the worker's traceback, which pickling drops. In this process an error object
    # can be shared between chunks, so it is left alone.
    trace = "".join(traceback.format_exception(error)).rstrip()
    with engine._Caught() as caught:
        pickle.loads(pickle.dumps(error))
    if caught.error is not None:
        error = RuntimeError(f"{type(error).__qualname__}: {error}")
    error.add_note(f"telescopic: the traceback in the worker process:\n{trace}")
    return error


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
    with _sigint(signal.SIG_IGN), threadpoolctl.threadpool_limits(1):
        yield


@contextmanager
def _sigint(handler: Any) -> Iterator[None]:
    # Ctrl-C goes to handler during the block. Only the main thread sets handlers. A
    # process that ignores Ctrl-C, as a user's pool worker may, keeps ignoring it, and a
    # handler that Python did not install, which getsignal gives as None (an embedding
    # host's, as under R's reticulate), stays, since Python could not restore it.
    main = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGINT) if main else None
    if previous is None or previous is signal.SIG_IGN:
        yield
        return
    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)

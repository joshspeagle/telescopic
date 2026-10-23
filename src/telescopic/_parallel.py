"""Runs of the engine in-process or on pools of processes, and Ctrl-C during them."""

from __future__ import annotations

import importlib
import math
import multiprocessing
import os
import pickle
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, cast

import numpy as np
from numpy.random import SeedSequence

from telescopic import engine
from telescopic._types import Distribution, IntArray
from telescopic.kernels.base import Kernel
from telescopic.target import Target

# Imported by name: threadpoolctl and cloudpickle ship no type information.
threadpoolctl = importlib.import_module("threadpoolctl")

# Chunks per worker: several, so that the workers finish together, and few, so that each
# chunk keeps the engine's lanes full.
CHUNKS_PER_WORKER = 4

# Seconds between checks for Ctrl-C while the parent waits: on Windows a wait without a
# timeout does not see it.
POLL = 0.1

Monitor = Callable[[engine.Status], bool]
OnPart = Callable[[engine.Replicates], None]


@dataclass
class Job:
    # One run's configuration: the engine on any chunk of its indices, in any process.
    target: Target
    kernel: Kernel
    initial: Distribution
    root: SeedSequence
    namespace: int
    settings: dict[str, Any]
    store: bool

    def __call__(self, indices: IntArray, monitor: Monitor | None = None) -> engine.Replicates:
        return engine.run(
            self.target,
            self.kernel,
            self.initial,
            indices,
            root=self.root,
            namespace=self.namespace,
            store=self.store,
            monitor=monitor,
            **self.settings,
        )


class Interrupt:
    # Ctrl-C during a run: the first stops new replicates, the second raises at once.
    def __init__(self) -> None:
        self.stopping = False

    @contextmanager
    def catching(self) -> Iterator[Interrupt]:
        # Python sets signal handlers only in the main thread; elsewhere Ctrl-C keeps its
        # usual effect.
        if threading.current_thread() is not threading.main_thread():
            yield self
            return
        previous = signal.signal(signal.SIGINT, self._handle)
        try:
            yield self
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
    if isinstance(pool, int | np.integer) and pool < 1:
        raise ValueError(f"pool must be at least 1 process, not {pool}")


def execute(
    job: Job,
    indices: IntArray,
    pool: Any,
    interrupt: Interrupt,
    monitor: Monitor | None = None,
    on_part: OnPart | None = None,
) -> engine.Replicates:
    # In-process, the monitor sees every iteration. On a pool, the chunks run in index
    # order and on_part sees each finished one; once interrupt.stopping is set, chunks
    # not started are dropped, except the first, and the others finish (spec 6.4, 6.5).
    if pool is None:

        def watch(status: engine.Status) -> bool:
            return (monitor is not None and monitor(status)) or interrupt.stopping

        return job(indices, watch)
    start = time.perf_counter()
    if isinstance(pool, int | np.integer):
        # Spawned workers behave alike on every platform, and forking a process whose
        # BLAS runs threads can deadlock.
        executor = ProcessPoolExecutor(
            int(pool),
            multiprocessing.get_context("spawn"),
            initializer=_install,
            initargs=(_dumps(job),),
        )
        try:
            parts = _gather(
                executor, _run_installed, (), _chunks(indices, int(pool)), interrupt, on_part
            )
        except BaseException:
            _terminate(executor)
            raise
        executor.shutdown()
    elif hasattr(pool, "submit"):
        workers = getattr(pool, "_max_workers", None) or os.cpu_count() or 1
        pieces = _chunks(indices, workers)
        args = (_dumps(job), os.getpid())
        parts = _gather(pool, _run_shipped, args, pieces, interrupt, on_part)
    else:
        workers = os.cpu_count() or 1
        parts = _waves(pool, _dumps(job), _chunks(indices, workers), workers, interrupt, on_part)
    # The seconds of a pool run are the parent's, not the sum over the workers.
    return replace(
        engine.merge(parts), requested=np.sort(indices), seconds=time.perf_counter() - start
    )


def _chunks(indices: IntArray, workers: int) -> list[IntArray]:
    size = math.ceil(len(indices) / (CHUNKS_PER_WORKER * workers))
    return [indices[start : start + size] for start in range(0, len(indices), size)]


def _gather(
    executor: Executor,
    function: Callable[..., engine.Replicates],
    args: tuple[Any, ...],
    pieces: list[IntArray],
    interrupt: Interrupt,
    on_part: OnPart | None,
) -> list[engine.Replicates]:
    # Every chunk is submitted at once, in index order. The first is never cancelled, so
    # a run returns at least one part.
    futures = [executor.submit(function, *args, piece) for piece in pieces]
    pending: set[Future[engine.Replicates]] = set(futures)
    parts = []
    try:
        while pending:
            if interrupt.stopping:
                for future in futures[1:]:
                    future.cancel()
            done, pending = wait(pending, timeout=POLL, return_when=FIRST_COMPLETED)
            for future in done:
                if not future.cancelled():
                    parts.append(future.result())
                    if on_part is not None:
                        on_part(parts[-1])
    finally:
        for future in pending:
            future.cancel()
    return parts


def _waves(
    pool: Any,
    payload: bytes,
    pieces: list[IntArray],
    workers: int,
    interrupt: Interrupt,
    on_part: OnPart | None,
) -> list[engine.Replicates]:
    # A pool with only map runs waves of one chunk per worker; once interrupt.stopping is
    # set, no new wave starts after the first.
    task = partial(_run_shipped, payload, os.getpid())
    parts: list[engine.Replicates] = []
    for start in range(0, len(pieces), workers):
        if parts and interrupt.stopping:
            break
        for part in pool.map(task, pieces[start : start + workers]):
            parts.append(part)
            if on_part is not None:
                on_part(part)
    return parts


def _terminate(executor: ProcessPoolExecutor) -> None:
    # Its workers ignore Ctrl-C, so a second one, or an error, stops them here. Python
    # 3.14 has terminate_workers; before it, the processes are in _processes.
    terminate = getattr(executor, "terminate_workers", None)
    if terminate is not None:
        terminate()
        return
    for process in list((executor._processes or {}).values()):
        process.terminate()
    executor.shutdown(wait=False, cancel_futures=True)


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


# The job of a worker in a pool that telescopic creates, or why it could not be loaded.
_installed: Job | Exception | None = None


def _install(payload: bytes) -> None:
    # The initializer of the pools that telescopic creates: each worker loads the job
    # once and leaves Ctrl-C to the parent. An error is kept for the tasks to raise, since
    # an initializer that raises breaks the pool without saying why.
    global _installed
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _installed = _load(payload)
    except RuntimeError as error:
        _installed = error


def _run_installed(indices: IntArray) -> engine.Replicates:
    if isinstance(_installed, Exception):
        raise _installed
    # Every task pins BLAS to one thread (spec section 6.4).
    with threadpoolctl.threadpool_limits(1):
        return cast(Job, _installed)(indices)


def _run_shipped(payload: bytes, parent: int, indices: IntArray) -> engine.Replicates:
    # User pools: the job travels with each task. BLAS settings belong to a process, so a
    # task in a thread of the parent leaves them alone: concurrent threads would restore
    # each other's limits.
    job = _load(payload)
    if os.getpid() == parent:
        return job(indices)
    with threadpoolctl.threadpool_limits(1):
        return job(indices)

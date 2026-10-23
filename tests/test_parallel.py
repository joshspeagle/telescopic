import functools
import multiprocessing
import os
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest
import threadpoolctl

import telescopic
from references import gaussian_log_prob
from telescopic import _parallel, _progress, engine
from telescopic._parallel import Interrupt, execute, opened
from telescopic._random import MAIN, root_seed
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk, State
from telescopic.target import Target
from telescopic.warnings import ConvergenceWarning

INIT = Normal([2.0, -1.0], [4.0, 4.0])
SETTINGS = {"k": 3, "m": 30, "lag": 3}


def job(log_prob=gaussian_log_prob, args=()):
    # Point by point, so that a replicate's results are bitwise the same in any batch.
    return functools.partial(
        engine.run,
        Target(log_prob, args=args),
        RandomWalk(np.eye(2)),
        INIT,
        root=root_seed(5),
        namespace=MAIN,
        **SETTINGS,
    )


def never(status):
    return False


def assert_same_replicates(a, b):
    for name, value in vars(a).items():
        if name != "seconds":
            np.testing.assert_array_equal(value, getattr(b, name), err_msg=name)


class Serial:
    # A pool with only map, run in this process; it counts the replicates it runs.
    def __init__(self):
        self.ran = 0

    def map(self, function, items):
        self.ran += sum(len(item) for item in items)
        return [function(item) for item in items]


class Counting(ThreadPoolExecutor):
    # An executor that records the size of each chunk submitted to it.
    def __init__(self, workers):
        super().__init__(workers)
        self.chunks = []

    def submit(self, function, piece, **options):
        self.chunks.append(len(piece))
        return super().submit(function, piece, **options)


def test_results_do_not_depend_on_the_pool():
    indices = np.arange(30)
    alone = execute(job(), indices, None, never)
    with ThreadPoolExecutor(3) as threads, opened(2) as workers:
        for pool in (workers, threads, Serial()):
            assert_same_replicates(execute(job(), indices, pool, never), alone)


def test_the_sampler_runs_its_pilot_and_replicates_on_its_pool():
    pool = Serial()
    pooled = telescopic.sample(gaussian_log_prob, INIT, 20, pool=pool, rng=6)
    alone = telescopic.sample(gaussian_log_prob, INIT, 20, rng=6)
    assert pool.ran == 100 + 20
    np.testing.assert_array_equal(pooled.pilot, alone.pilot)
    assert_same_replicates(pooled._replicates, alone._replicates)


@pytest.mark.parametrize(
    ("n", "workers", "chunks"), [(3, 4, 3), (200, 4, 4), (600, 4, 10), (1_000, 1, 4)]
)
def test_a_pool_gets_chunks_of_full_lanes_and_at_most_four_per_worker(n, workers, chunks):
    # 600 replicates fill 64 lanes 10 times; 1,000 on one worker are cut in four.
    with Counting(workers) as executor:
        execute(job(), np.arange(n), executor, never)
    assert len(executor.chunks) == chunks
    assert max(executor.chunks) - min(executor.chunks) <= 1


def checked_log_prob(x, threads, ignores_ctrl_c):
    # Raises unless BLAS runs `threads` threads and Ctrl-C is ignored or not, as asked.
    info = threadpoolctl.threadpool_info()
    running = [library["num_threads"] for library in info if library["user_api"] == "blas"]
    ignored = signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    if any(n != threads for n in running) or ignored != ignores_ctrl_c:
        raise RuntimeError(f"BLAS threads {running}, Ctrl-C ignored: {ignored}")
    return gaussian_log_prob(x)


def test_tasks_in_other_processes_pin_blas_and_ignore_ctrl_c(monkeypatch):
    # The workers would start with two BLAS threads; conftest.py sets one for the tests.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "2")
    pinned = job(checked_log_prob, (1, True))
    with opened(1) as workers:
        execute(pinned, np.arange(4), workers, never)
    with ProcessPoolExecutor(1, multiprocessing.get_context("spawn")) as executor:
        execute(pinned, np.arange(4), executor, never)


def test_tasks_in_this_process_leave_its_settings_alone():
    with threadpoolctl.threadpool_limits(2), ThreadPoolExecutor(2) as threads:
        execute(job(checked_log_prob, (2, False)), np.arange(4), threads, never)


def nan_log_prob(x):
    return np.nan


def test_an_error_in_a_worker_reaches_the_caller():
    with pytest.raises(ValueError, match="log_prob returned nan"), opened(1) as workers:
        execute(job(nan_log_prob), np.arange(8), workers, never)


def test_a_stop_reaches_the_replicates_of_telescopics_workers():
    # The stop comes before the worker has started: its first chunk starts one set of
    # lanes and refills none, and the other chunks run nothing.
    with opened(1) as workers:
        stopped = execute(job(), np.arange(512), workers, lambda status: True)
    np.testing.assert_array_equal(stopped.index, np.arange(engine.LANES))
    np.testing.assert_array_equal(stopped.requested, np.arange(512))


def test_a_chunk_that_starts_after_the_stop_runs_nothing_unless_it_is_the_first(
    monkeypatch,
):
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(_parallel, "_stop", stop)
    payload = _parallel._dumps(job())
    assert _parallel._task(payload, os.getpid(), np.arange(200)) is None
    first = _parallel._task(payload, os.getpid(), np.arange(200), first=True)
    np.testing.assert_array_equal(first.index, np.arange(engine.LANES))


@pytest.mark.parametrize("pool", ["threads", "map"])
@pytest.mark.parametrize("at_once", [True, False])
def test_a_stop_on_a_pool_keeps_the_finished_chunks(pool, at_once, monkeypatch):
    # A stop before any chunk starts, or once one has finished: chunks not started are
    # dropped, except the first, and the result lacks requested replicates.
    monkeypatch.setattr(_parallel.os, "cpu_count", lambda: 1)
    indices = np.arange(1_000)
    with ThreadPoolExecutor(1) as thread:
        if at_once and pool == "threads":
            thread.submit(time.sleep, 0.2)  # holds every chunk until the stop is seen
        runner = thread if pool == "threads" else Serial()
        stopped = execute(job(), indices, runner, lambda status: at_once or status.finished)
    assert 0 < len(stopped.index) < len(indices)
    np.testing.assert_array_equal(stopped.index, np.arange(len(stopped.index)))
    np.testing.assert_array_equal(stopped.requested, indices)


def test_ctrl_c_once_stops_new_replicates_and_twice_raises(capsys):
    previous = signal.getsignal(signal.SIGINT)
    interrupt = Interrupt()
    with interrupt.catching():
        signal.raise_signal(signal.SIGINT)
        assert interrupt.stopping
        assert "press Ctrl-C again" in capsys.readouterr().err
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) is previous


def interrupting(after):
    # A log-density that presses Ctrl-C at its evaluation number `after`.
    calls = 0

    def log_prob(x):
        nonlocal calls
        calls += 1
        if calls == after:
            signal.raise_signal(signal.SIGINT)
        return gaussian_log_prob(x)

    return log_prob


def test_an_interrupted_run_is_completed_by_the_next_run(capsys):
    # Ctrl-C after 2,000 evaluations, while the first 64 replicates (one per lane) are in
    # progress: they finish, and the next run starts with the missing ones, so the two
    # runs equal one uninterrupted run. k = 40 keeps the pairs from meeting late.
    kernel = RandomWalk(np.eye(2))
    sampler = telescopic.CoupledSampler(interrupting(2_000), INIT, kernel=kernel, rng=7)
    first = sampler.run(90, k=40)
    assert first.interrupted
    np.testing.assert_array_equal(first.index, np.arange(engine.LANES))
    second = sampler.run(30)
    assert not second.interrupted
    whole = telescopic.sample(gaussian_log_prob, INIT, 94, kernel=kernel, k=40, rng=7)
    assert_same_replicates(second._replicates, whole._replicates)
    assert "press Ctrl-C again" in capsys.readouterr().err


def test_ctrl_c_during_the_pilot_raises():
    kernel = RandomWalk(np.eye(2))
    sampler = telescopic.CoupledSampler(interrupting(50), INIT, kernel=kernel, rng=7)
    with pytest.raises(KeyboardInterrupt):
        sampler.meeting_times(20)


def test_a_run_in_another_thread_leaves_ctrl_c_alone():
    with ThreadPoolExecutor(1) as thread:
        run = thread.submit(telescopic.sample, gaussian_log_prob, INIT, 5, k=40, rng=8)
        assert run.result().n_replicates == 5


class Apart(RandomWalk):
    # X and Y step apart, so that no pair meets.
    def step(self, target, state, draws):
        x = state.x + 1.0
        return State(x, target.log_prob(x)), np.ones(len(x))

    def coupled_step(self, target, state_x, state_y, draws):
        y = state_y.x - 1.0
        return self.step(target, state_x, draws)[0], State(y, target.log_prob(y))


def test_slow_pilot_pairs_on_a_pool_warn_after_the_run():
    sampler = telescopic.CoupledSampler(
        gaussian_log_prob, INIT, kernel=Apart(np.eye(2)), pool=Serial(), rng=9
    )
    with pytest.warns(ConvergenceWarning, match="has not met after 10000 iterations"):
        sampler.meeting_times(2, max_iter=10_001)


def progressing(pool=None):
    kernel = RandomWalk(np.eye(2))
    return telescopic.CoupledSampler(
        gaussian_log_prob, INIT, kernel=kernel, pool=pool, progress=True, rng=8
    )


@pytest.mark.parametrize("pool", [None, Serial()], ids=["here", "pool"])
def test_progress_falls_back_to_lines_on_stderr_without_tqdm(pool, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    monkeypatch.setattr(_progress, "INTERVAL", 0.0)
    sampler = progressing(pool)
    sampler.meeting_times(4)
    sampler.run(5, k=40)
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith("telescopic: pilot pairs: ")
    assert lines[-1].startswith("telescopic: replicates: 5 of 5 replicates finished")


def test_progress_bars_count_the_replicates_as_they_finish(capsys):
    # The pilot pairs meet at different iterations, so the bar moves by increments.
    progressing().meeting_times(4)
    assert "4/4" in capsys.readouterr().err.split("\r")[-1]


def test_pools_ship_functions_with_cloudpickle_and_say_what_to_do_without(monkeypatch):
    nameless = job(lambda x: gaussian_log_prob(x))
    execute(nameless, np.arange(4), Serial(), never)
    monkeypatch.setitem(sys.modules, "cloudpickle", None)
    with pytest.raises(TypeError, match="install cloudpickle"):
        execute(nameless, np.arange(4), Serial(), never)


def test_a_worker_that_cannot_load_the_job_says_why():
    with pytest.raises(RuntimeError, match="could not load the target"):
        _parallel._load(b"not a pickle")


@pytest.mark.parametrize(
    ("pool", "error"),
    [(True, TypeError), ("four", TypeError), (object(), TypeError), (0, ValueError)],
)
def test_invalid_pools_raise(pool, error):
    with pytest.raises(error, match="pool must be"):
        telescopic.CoupledSampler(gaussian_log_prob, INIT, pool=pool)

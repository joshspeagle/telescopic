import multiprocessing
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest
import threadpoolctl

import telescopic
from references import gaussian_log_prob
from telescopic import _parallel, _progress
from telescopic._parallel import Interrupt, Job, check_pool, execute
from telescopic._random import MAIN, root_seed
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.target import Target

INIT = Normal([2.0, -1.0], [4.0, 4.0])
SETTINGS = {"k": 3, "m": 30, "lag": 3, "thin": 1, "max_iter": None}
FIELDS = (
    "index",
    "meeting_time",
    "cost",
    "atoms",
    "weights",
    "owner",
    "distance_owner",
    "distance_time",
    "distance",
    "zero_density",
    "requested",
    "n_log_prob",
    "n_grad",
)


def job(log_prob=gaussian_log_prob):
    # Point by point, so that a replicate's results are bitwise the same in any batch.
    target = Target(log_prob)
    return Job(target, RandomWalk(np.eye(2)), INIT, root_seed(5), MAIN, SETTINGS, True)


def assert_same_replicates(a, b):
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


class Serial:
    # A pool with only map, run in this process.
    def map(self, function, items):
        return [function(item) for item in items]


def test_results_do_not_depend_on_the_pool():
    indices = np.arange(30)
    alone = execute(job(), indices, None, Interrupt())
    with ThreadPoolExecutor(3) as threads:
        for pool in (2, threads, Serial()):
            assert_same_replicates(execute(job(), indices, pool, Interrupt()), alone)


def test_the_sampler_runs_its_pilot_and_replicates_on_a_pool():
    with ThreadPoolExecutor(2) as threads:
        pooled = telescopic.sample(gaussian_log_prob, INIT, 20, pool=threads, rng=6)
    alone = telescopic.sample(gaussian_log_prob, INIT, 20, rng=6)
    np.testing.assert_array_equal(pooled.pilot, alone.pilot)
    assert_same_replicates(pooled._replicates, alone._replicates)


def test_chunks_cover_the_indices_once_in_order():
    indices = np.arange(5, 46)
    for workers in (1, 3, 100):
        pieces = _parallel._chunks(indices, workers)
        np.testing.assert_array_equal(np.concatenate(pieces), indices)
        assert len(pieces) <= _parallel.CHUNKS_PER_WORKER * workers


def pinned_log_prob(x):
    # Raises unless BLAS runs one thread and Ctrl-C is ignored, as in telescopic's workers.
    threads = [
        i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"
    ]
    handler = signal.getsignal(signal.SIGINT)
    if any(n != 1 for n in threads) or handler != signal.SIG_IGN:
        raise RuntimeError(f"BLAS threads {threads}, SIGINT handler {handler}")
    return gaussian_log_prob(x)


def blas_pinned_log_prob(x):
    threads = [
        i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"
    ]
    if any(n != 1 for n in threads):
        raise RuntimeError(f"BLAS threads {threads}")
    return gaussian_log_prob(x)


def test_tasks_pin_blas_and_workers_that_telescopic_starts_ignore_ctrl_c(monkeypatch):
    # The workers would start with two BLAS threads; conftest.py sets one for the tests.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "2")
    indices = np.arange(4)
    execute(job(pinned_log_prob), indices, 1, Interrupt())
    with ProcessPoolExecutor(1, multiprocessing.get_context("spawn")) as executor:
        execute(job(blas_pinned_log_prob), indices, executor, Interrupt())


def nan_log_prob(x):
    return np.nan


def test_an_error_in_a_worker_reaches_the_caller():
    with pytest.raises(ValueError, match="log_prob returned nan"):
        execute(job(nan_log_prob), np.arange(8), 1, Interrupt())


@pytest.mark.parametrize("pool", ["threads", "map"])
@pytest.mark.parametrize("at_once", [True, False])
def test_a_stop_on_a_pool_keeps_the_finished_chunks_a_prefix(pool, at_once, monkeypatch):
    # A stop before any chunk starts, or at the first finished one: chunks not started
    # are dropped, except the first, and the result lacks requested replicates.
    monkeypatch.setattr(_parallel.os, "cpu_count", lambda: 2)
    interrupt = Interrupt()
    interrupt.stopping = at_once

    def stop(part):
        interrupt.stopping = True

    indices = np.arange(40)
    with ThreadPoolExecutor(1) as thread:
        if at_once:
            thread.submit(time.sleep, 0.2)  # holds every chunk until the stop is seen
        runner = thread if pool == "threads" else Serial()
        pooled = execute(job(), indices, runner, interrupt, on_part=stop)
    assert 0 < len(pooled.index) < len(indices)
    np.testing.assert_array_equal(pooled.index, np.arange(len(pooled.index)))
    np.testing.assert_array_equal(pooled.requested, indices)


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


def test_an_interrupted_run_is_completed_by_the_next_run(capsys):
    # Ctrl-C after 2,000 evaluations, while the first 64 replicates (one per lane) are in
    # progress: they finish, and the next run starts with the missing ones, so the two
    # runs equal one uninterrupted run. k = 40 keeps the pairs from meeting late.
    calls = 0

    def log_prob(x):
        nonlocal calls
        calls += 1
        if calls == 2000:
            signal.raise_signal(signal.SIGINT)
        return gaussian_log_prob(x)

    sampler = telescopic.CoupledSampler(log_prob, INIT, kernel=RandomWalk(np.eye(2)), rng=7)
    first = sampler.run(90, k=40)
    assert first.interrupted
    np.testing.assert_array_equal(first.index, np.arange(64))
    second = sampler.run(26)
    assert not second.interrupted
    whole = telescopic.sample(
        gaussian_log_prob, INIT, 90, kernel=RandomWalk(np.eye(2)), k=40, rng=7
    )
    assert_same_replicates(second._replicates, whole._replicates)
    assert "press Ctrl-C again" in capsys.readouterr().err


def progressing_sampler():
    kernel = RandomWalk(np.eye(2))
    return telescopic.CoupledSampler(gaussian_log_prob, INIT, kernel=kernel, progress=True, rng=8)


def test_progress_falls_back_to_lines_on_stderr_without_tqdm(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    monkeypatch.setattr(_progress, "INTERVAL", 0.0)
    sampler = progressing_sampler()
    sampler.meeting_times(4)
    sampler.run(5, k=40)
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith("telescopic: pilot pairs: 1 of 4 replicates finished")
    assert lines[-1].startswith("telescopic: replicates: 5 of 5 replicates finished")


def test_progress_uses_tqdm_when_it_is_installed(capsys):
    pytest.importorskip("tqdm")
    progressing_sampler().run(5, k=40)
    assert "5/5" in capsys.readouterr().err


def test_an_unpicklable_target_on_a_pool_raises_with_advice(monkeypatch):
    monkeypatch.setitem(sys.modules, "cloudpickle", None)
    with pytest.raises(TypeError, match="install cloudpickle"):
        execute(job(lambda x: gaussian_log_prob(x)), np.arange(4), Serial(), Interrupt())


def test_a_worker_that_cannot_load_the_job_says_why():
    previous = signal.getsignal(signal.SIGINT)
    try:
        _parallel._install(b"not a pickle")
        with pytest.raises(RuntimeError, match="could not load the target"):
            _parallel._run_installed(np.arange(2))
    finally:
        signal.signal(signal.SIGINT, previous)
        _parallel._installed = None


@pytest.mark.parametrize(
    ("pool", "error"),
    [(True, TypeError), ("four", TypeError), (object(), TypeError), (0, ValueError)],
)
def test_invalid_pools_raise(pool, error):
    with pytest.raises(error, match="pool must be"):
        check_pool(pool)

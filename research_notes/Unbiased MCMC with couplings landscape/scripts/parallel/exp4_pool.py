"""Process-pool overheads, pickling of closures with data, and worker-count invariance."""
import functools
import math
import multiprocessing as mp
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = 0x5EED_1234_5678_9ABC_DEF0  # 'root entropy' for the run


def trivial(i):
    return i


def logp_with_data(theta, data):
    return -0.5 * float(theta * theta) + 0.0 * data[0]


_DATA = None


def init_worker(data):
    global _DATA
    _DATA = data


def logp_global(theta):
    return -0.5 * float(theta * theta) + 0.0 * _DATA[0]


def call(f, i):
    return f(float(i))


def replicate(i):
    """One unbiased-MCMC replicate whose randomness depends only on (ROOT, i)."""
    from coupled_rwmh import run_pairs
    ss = np.random.SeedSequence(ROOT, spawn_key=(i,))
    H, tau, C = run_pairs(1, d=1, sigma=1.0, mu0=3.0, k=2, m=10, rng=ss)
    return i, float(H[0]), int(C[0])


def main():
    ctx = mp.get_context("forkserver")  # Python 3.14+ default on Linux
    n = 4000
    with ProcessPoolExecutor(4, mp_context=ctx) as ex:
        list(ex.map(trivial, range(8)))  # warm up workers
        t0 = time.perf_counter()
        futs = [ex.submit(trivial, i) for i in range(n)]
        for f in as_completed(futs):
            f.result()
        t_submit = time.perf_counter() - t0
        t0 = time.perf_counter()
        list(ex.map(trivial, range(n), chunksize=1))
        t_map1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        list(ex.map(trivial, range(n), chunksize=64))
        t_map64 = time.perf_counter() - t0
    print(f"(a) 4 forkserver workers, {n} trivial tasks: submit+as_completed {1e6*t_submit/n:.0f} us/task; "
          f"map chunksize=1 {1e6*t_map1/n:.0f} us/task; chunksize=64 {1e6*t_map64/n:.1f} us/task")

    data = np.random.default_rng(0).standard_normal(1_000_000)  # 8 MB 'dataset'
    f_closure = functools.partial(logp_with_data, data=data)
    t0 = time.perf_counter()
    blob = pickle.dumps(f_closure, protocol=pickle.HIGHEST_PROTOCOL)
    t_pickle = time.perf_counter() - t0
    ntask = 200
    with ProcessPoolExecutor(4, mp_context=ctx) as ex:
        list(ex.map(trivial, range(8)))
        t0 = time.perf_counter()
        list(ex.map(call, [f_closure] * ntask, range(ntask)))
        t_closure = time.perf_counter() - t0
    with ProcessPoolExecutor(4, mp_context=ctx, initializer=init_worker, initargs=(data,)) as ex:
        list(ex.map(trivial, range(8)))
        t0 = time.perf_counter()
        list(ex.map(call, [logp_global] * ntask, range(ntask)))
        t_global = time.perf_counter() - t0
    print(f"(b) closure with 8 MB array: pickle {len(blob)/1e6:.1f} MB in {1e3*t_pickle:.1f} ms; per task "
          f"{1e3*t_closure/ntask:.2f} ms when shipped with every task vs {1e3*t_global/ntask:.3f} ms "
          f"with initializer-held global")

    R = 400
    results = {}
    for nw in [1, 4]:
        with ProcessPoolExecutor(nw, mp_context=ctx) as ex:
            futs = [ex.submit(replicate, i) for i in range(R)]
            order = []
            vals = {}
            for f in as_completed(futs):
                i, h, c = f.result()
                order.append(i)
                vals[i] = h
        comp_sum = 0.0
        for i in order:
            comp_sum += vals[i]
        idx_sum = sum(vals[i] for i in range(R))
        results[nw] = (vals, order, comp_sum, idx_sum, math.fsum(vals.values()))
    v1, o1, c1, s1, f1 = results[1]
    v4, o4, c4, s4, f4 = results[4]
    same_vals = all(v1[i] == v4[i] for i in range(R))
    print(f"(c) R={R} replicates, 1 vs 4 workers: per-index values bitwise identical: {same_vals}; "
          f"completion order identical: {o1 == o4}")
    print(f"    sum in completion order: 1w={c1!r} 4w={c4!r} equal={c1 == c4}")
    print(f"    sum in index order:      1w={s1!r} 4w={s4!r} equal={s1 == s4}; fsum equal={f1 == f4}")


if __name__ == "__main__":
    main()

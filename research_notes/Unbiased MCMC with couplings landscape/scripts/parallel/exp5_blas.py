"""BLAS oversubscription: P worker processes x default OpenBLAS threads vs 1 thread each."""
import multiprocessing as mp
import os
import sys
import time


def work(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((400, 400))
    t0 = time.perf_counter()
    for _ in range(150):
        a = a @ a.T
        a /= np.abs(a).max()
    return time.perf_counter() - t0


def run(nthreads):
    if nthreads is None:
        os.environ.pop("OPENBLAS_NUM_THREADS", None)
    else:
        os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    ctx = mp.get_context("spawn")  # children re-import numpy and read the env var
    with ctx.Pool(4) as pool:
        pool.map(work, range(4))  # warm-up (imports)
        t0 = time.perf_counter()
        pool.map(work, range(4, 12))
        return time.perf_counter() - t0


if __name__ == "__main__":
    print(f"cores={os.cpu_count()}")
    for nt in [None, 1]:
        label = "default (all cores per process)" if nt is None else f"OPENBLAS_NUM_THREADS={nt}"
        print(f"4 processes x 8 matmul jobs, {label}: {run(nt):.2f} s")

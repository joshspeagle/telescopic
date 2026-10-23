"""Steady-state cost of per-replicate RNG streams in a lockstep batch (creation timed separately)."""
import math, time
import numpy as np

def draws_batch(rng, B, d, steps):
    for s in range(steps):
        rng.standard_normal((B, d + 2))

def draws_per_lane(gens, d, steps):
    for s in range(steps):
        np.stack([g.standard_normal(d + 2) for g in gens])

def draws_block(gens, d, steps, K):
    for s in range(steps):
        if s % K == 0:
            buf = np.stack([g.standard_normal((K, d + 2)) for g in gens], axis=1)
        buf[s % K]

d, steps = 10, 256
for B in [100, 1000, 10000]:
    t0 = time.perf_counter(); gens = [np.random.default_rng(np.random.SeedSequence(123, spawn_key=(i,))) for i in range(B)]; tc = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    t0 = time.perf_counter(); draws_batch(rng, B, d, steps); tb = time.perf_counter() - t0
    t0 = time.perf_counter(); draws_per_lane(gens, d, steps); tp = time.perf_counter() - t0
    t0 = time.perf_counter(); draws_block(gens, d, steps, 64); tk = time.perf_counter() - t0
    print(f"B={B:6d} RNG-only us/step: batch {1e6*tb/steps:8.1f} | per-lane {1e6*tp/steps:9.1f} | block K=64 {1e6*tk/steps:8.1f} | one-off creation of B Generators {1e3*tc:.1f} ms")

# scalar per-replicate loop with d=10 NumPy arrays and a Python log-density (pool-task style)
def logp(x):
    return -0.5 * float(x @ x)

def scalar_d(n, d=10):
    rng = np.random.default_rng(0)
    sigma = 2.38 / math.sqrt(d)
    x = np.full(d, 3.0); y = np.zeros(d); lx, ly = logp(x), logp(y)
    for _ in range(n):
        xi = rng.standard_normal(d); uc, ua = rng.random(2)
        px = x + sigma * xi
        z = (x - y) / sigma; nz = math.sqrt(float(z @ z)); e = z / nz
        if math.log(uc) <= -0.5 * float((xi + z) @ (xi + z)) + 0.5 * float(xi @ xi):
            py = px
        else:
            py = y + sigma * (xi - 2.0 * float(e @ xi) * e)
        lpx, lpy = logp(px), logp(py); lu = math.log(ua)
        if lu < lpx - lx: x, lx = px, lpx
        if lu < lpy - ly: y, ly = py, lpy
n = 50000
t0 = time.perf_counter(); scalar_d(n); t = time.perf_counter() - t0
print(f"scalar (one pair, d=10, NumPy small arrays + Python logpdf): {1e6*t/n:.1f} us per coupled iteration")

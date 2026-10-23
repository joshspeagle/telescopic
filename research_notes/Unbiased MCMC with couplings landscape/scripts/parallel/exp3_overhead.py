"""Per-step overheads: scalar Python loop vs lockstep batch; RNG-per-replicate variants."""
import math
import random
import time
import numpy as np


def timeit(fn, reps=3):
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# (a) scalar coupled RWMH step (1-D), pure Python + NumPy Generator scalars
def scalar_numpy(n=100000, sigma=1.0):
    rng = np.random.default_rng(0)
    x, y = 3.0, 2.0
    lx, ly = -0.5 * x * x, -0.5 * y * y
    for _ in range(n):
        xi = rng.standard_normal()
        uc = rng.random()
        ua = rng.random()
        px = x + sigma * xi
        z = (x - y) / sigma
        if math.log(uc) <= -0.5 * (xi + z) ** 2 + 0.5 * xi * xi:
            py = px
        else:
            py = y - sigma * xi  # 1-D reflection
        lpx, lpy = -0.5 * px * px, -0.5 * py * py
        lu = math.log(ua)
        if lu < lpx - lx:
            x, lx = px, lpx
        if lu < lpy - ly:
            y, ly = py, lpy
        if x == y:  # restart to keep the pair unmet (we only time the step)
            y = x - 1.0
            ly = -0.5 * y * y


def scalar_stdlib(n=100000, sigma=1.0):
    r = random.Random(0)
    x, y = 3.0, 2.0
    lx, ly = -0.5 * x * x, -0.5 * y * y
    for _ in range(n):
        xi = r.gauss(0.0, 1.0)
        uc = r.random() or 1e-300
        ua = r.random() or 1e-300
        px = x + sigma * xi
        z = (x - y) / sigma
        py = px if math.log(uc) <= -0.5 * (xi + z) ** 2 + 0.5 * xi * xi else y - sigma * xi
        lpx, lpy = -0.5 * px * px, -0.5 * py * py
        lu = math.log(ua)
        if lu < lpx - lx:
            x, lx = px, lpx
        if lu < lpy - ly:
            y, ly = py, lpy
        if x == y:
            y = x - 1.0
            ly = -0.5 * y * y


n = 100000
t_np = timeit(lambda: scalar_numpy(n))
t_std = timeit(lambda: scalar_stdlib(n))
print(f"(a) scalar coupled step, 1-D: numpy-Generator scalars {1e6*t_np/n:.2f} us/step; "
      f"stdlib random {1e6*t_std/n:.2f} us/step")


# (b) lockstep batch of B pairs, d-dim Gaussian target, one batched logpdf call per chain per step
def lockstep(B, d, steps, rng_mode="batch", K=64):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((B, d)) + 3
    y = rng.standard_normal((B, d))
    lx = -0.5 * np.einsum("ij,ij->i", x, x)
    ly = -0.5 * np.einsum("ij,ij->i", y, y)
    sigma = 2.38 / math.sqrt(d)
    if rng_mode != "batch":
        gens = [np.random.default_rng(np.random.SeedSequence(123, spawn_key=(i,))) for i in range(B)]
        if rng_mode == "block":
            blocks = [g.standard_normal((K, d + 2)) for g in gens]
            buf = np.stack(blocks, axis=1)  # (K, B, d+2)
    for s in range(steps):
        if rng_mode == "batch":
            draws = rng.standard_normal((B, d + 2))
        elif rng_mode == "per_lane":
            draws = np.stack([g.standard_normal(d + 2) for g in gens])
        else:  # block: refresh per-lane blocks every K steps
            j = s % K
            if j == 0 and s > 0:
                buf = np.stack([g.standard_normal((K, d + 2)) for g in gens], axis=1)
            draws = buf[j]
        xi = draws[:, :d]
        uc = 0.5 * (1 + np.tanh(draws[:, d]))  # cheap stand-ins for uniforms (timing only)
        ua = 0.5 * (1 + np.tanh(draws[:, d + 1]))
        px = x + sigma * xi
        z = (x - y) / sigma
        nz = np.sqrt(np.einsum("ij,ij->i", z, z))
        e = z / np.where(nz > 0, nz, 1.0)[:, None]
        lr = -0.5 * np.einsum("ij,ij->i", xi + z, xi + z) + 0.5 * np.einsum("ij,ij->i", xi, xi)
        same = np.log(uc) <= lr
        eta = xi - 2.0 * np.einsum("ij,ij->i", e, xi)[:, None] * e
        py = np.where(same[:, None], px, y + sigma * eta)
        lpx = -0.5 * np.einsum("ij,ij->i", px, px)
        lpy = -0.5 * np.einsum("ij,ij->i", py, py)
        lu = np.log(ua)
        ax, ay = lu < lpx - lx, lu < lpy - ly
        x = np.where(ax[:, None], px, x)
        lx = np.where(ax, lpx, lx)
        y = np.where(ay[:, None], py, y)
        ly = np.where(ay, lpy, ly)


d = 10
print(f"(b) lockstep batch, d={d}, batch RNG:")
for B in [1, 10, 100, 1000, 10000, 100000]:
    steps = max(20, int(200000 / B)) if B < 10000 else 50
    t = timeit(lambda: lockstep(B, d, steps))
    print(f"    B={B:6d}: {1e6*t/steps:9.1f} us/step, {1e9*t/(steps*B):9.1f} ns per pair-iteration")

print("(c) per-replicate RNG streams in lockstep, d=10:")
for B in [100, 1000, 10000]:
    steps = max(20, int(100000 / B))
    tb = timeit(lambda: lockstep(B, d, steps, "batch"))
    tp = timeit(lambda: lockstep(B, d, steps, "per_lane"), reps=1)
    tk = timeit(lambda: lockstep(B, d, steps, "block", K=64), reps=1)
    print(f"    B={B:6d}: batch {1e6*tb/steps:8.1f} us/step | per-lane Generator {1e6*tp/steps:9.1f} us/step "
          f"| per-lane block K=64 {1e6*tk/steps:8.1f} us/step (incl. Generator creation)")

# (d) cost of creating one Generator per replicate from an index-addressed SeedSequence
nrep = 20000
t = timeit(lambda: [np.random.default_rng(np.random.SeedSequence(2**100 + 7, spawn_key=(i,))) for i in range(nrep)])
t2 = timeit(lambda: [np.random.Generator(np.random.Philox(np.random.SeedSequence(2**100 + 7, spawn_key=(i,)))) for i in range(nrep)])
print(f"(d) Generator(PCG64) per replicate: {1e6*t/nrep:.1f} us each; Generator(Philox): {1e6*t2/nrep:.1f} us each")
ss = np.random.SeedSequence(2**100 + 7)
kids = ss.spawn(5)
same = all(np.random.default_rng(kids[i]).random() == np.random.default_rng(np.random.SeedSequence(2**100 + 7, spawn_key=(i,))).random() for i in range(5))
print(f"    SeedSequence(root).spawn(5)[i] == SeedSequence(root, spawn_key=(i,)) : {same}; kids[3].spawn_key={kids[3].spawn_key}")
g = np.random.default_rng(2**100 + 7)
print(f"    Generator.spawn available: {hasattr(g, 'spawn')}; numpy {np.__version__}")

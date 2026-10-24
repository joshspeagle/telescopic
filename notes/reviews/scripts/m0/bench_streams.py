import time
import numpy as np
from threadpoolctl import threadpool_limits
from telescopic._random import MAIN, Streams, replicate_generator, root_seed

def run(B, d, steps=2000, life=(97, 127), width=1024):
    root = root_seed(0)
    streams = Streams(B, width=width)
    rng = np.random.default_rng(1)
    remaining = rng.integers(*life, size=B)
    index = 0
    for lane in range(B):
        streams.assign(lane, replicate_generator(root, MAIN, index)); index += 1
    lanes = np.arange(B)
    draws = streams.view(lanes)
    t_assign = 0.0
    start = time.perf_counter()
    for _ in range(steps):
        draws.normal(d); draws.uniform(); draws.uniform()
        remaining -= 1
        done = np.flatnonzero(remaining == 0)
        if len(done):
            t0 = time.perf_counter()
            for lane in done:
                streams.assign(int(lane), replicate_generator(root, MAIN, index)); index += 1
            t_assign += time.perf_counter() - t0
            remaining[done] = rng.integers(*life, size=len(done))
    total = time.perf_counter() - start
    return (total - t_assign) / (steps * B) * 1e6, t_assign / max(index - B, 1) * 1e6

def shared(B, d, steps=2000):
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for _ in range(steps):
        rng.standard_normal((B, d)); rng.random(B); rng.random(B)
    return (time.perf_counter() - start) / (steps * B) * 1e6

with threadpool_limits(1):
    for B, d in [(64, 10), (256, 10), (1000, 10), (256, 100), (64, 1)]:
        per_step = min(run(B, d)[0] for _ in range(3))
        setup = run(B, d)[1]
        s = min(shared(B, d) for _ in range(3))
        print(f"B={B:5d} d={d:4d}: streams {per_step:.3f} us/lane-step, shared {s:.3f}, ratio {per_step/s:.2f}; setup {setup:.1f} us/replicate")

print("width scan, B=256, d=10")
with threadpool_limits(1):
    for life in [(97, 127), (1400, 1600)]:
        for width in (128, 256, 512, 1024, 2048):
            steps = 2000 if life[0] < 1000 else 4000
            per_step = min(run(256, 10, steps=steps, life=life, width=width)[0] for _ in range(3))
            print(f"life {life}, width {width}: {per_step:.3f} us/lane-step")

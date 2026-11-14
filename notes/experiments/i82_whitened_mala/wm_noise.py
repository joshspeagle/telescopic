"""How steady is this machine's speed? Context for the timing ratios of wm_timing.py.

Question. The ratios of wm_timing.py vary between rounds and between runs by more than
the measured costs explain. Is the processor's own speed steady over blocks of about half
a second, with nothing else running?

Design. One BLAS product of 64 rows with a 200 x 200 matrix (`x @ M.T`, the kernels'
product at d = 200), timed in 20 consecutive blocks of 5,000 products, wall time and
processor time per product; and, from /proc/stat over the same interval, the shares of
the machine's time spent idle and stolen by the host (steal).

Result (`wm_noise.out`, commit 9932df7, 1-minute load 0.13): the time per product
ranged from 84.8 to 92.2 us over the 20 blocks (median 87.6), processor time tracking
wall time block by block; steal was 0.06% of the machine's time. So the speed of one
thread drifts by about 9% between blocks of 0.44 s with no other process running and
almost no steal: neither the load nor steal explains the drift.

Run from the repository root:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python <this folder>/wm_noise.py
"""

import time

import numpy as np


def cpu_times():
    with open("/proc/stat") as stat:
        return np.array([int(v) for v in stat.readline().split()[1:9]])


rng = np.random.default_rng(0)
matrix, x = rng.standard_normal((200, 200)), rng.standard_normal((64, 200))
start = cpu_times()
blocks = []
for _ in range(20):
    wall, cpu = time.perf_counter(), time.process_time()
    for _ in range(5000):
        x @ matrix.T
    blocks.append(((time.perf_counter() - wall) / 5000, (time.process_time() - cpu) / 5000))
spent = cpu_times() - start
walls, cpus = 1e6 * np.array(blocks).T
print("us per product, wall:", " ".join(f"{v:.1f}" for v in walls))
print("us per product, cpu: ", " ".join(f"{v:.1f}" for v in cpus))
print(f"wall: min {walls.min():.1f}, median {np.median(walls):.1f}, max {walls.max():.1f}")
names = ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"]
print("jiffies over the blocks:", dict(zip(names, spent.tolist(), strict=False)))
print(f"steal share {spent[7] / spent.sum():.4f}, idle share {spent[3] / spent.sum():.3f}")

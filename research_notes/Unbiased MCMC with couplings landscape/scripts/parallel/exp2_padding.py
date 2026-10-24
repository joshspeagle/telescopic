"""Padding waste of lockstep batches without refill (vmap(while_loop)-style) vs. with refill."""
import heapq
import numpy as np
from coupled_rwmh import run_pairs

H, tau, C = run_pairs(2_000_000, d=1, sigma=1.0, mu0=3.0, k=2, m=10, rng=2026)
rng = np.random.default_rng(5)
geo = rng.geometric(1 / 50, size=2_000_000)  # meeting times with geometric tail, mean 50
for name, cost in [("toy C=max(tau,m), E=%.1f" % C.mean(), C.astype(float)),
                   ("toy tau alone, E=%.1f" % tau.mean(), tau.astype(float)),
                   ("Geometric(1/50), E=%.1f" % geo.mean(), geo.astype(float))]:
    row = []
    for B in [1, 8, 64, 512, 4096, 32768]:
        nb = cost.size // B
        mx = cost[: nb * B].reshape(nb, B).max(1).mean()
        row.append(f"B={B}: {mx / cost.mean():.2f}")
    print(f"no-refill waste factor E[max_B C]/E[C] [{name}]: " + ", ".join(row))


def refill_efficiency(cost, B, R, rng):
    """Lockstep with refill: lanes take replicate indices 1..R in order; no starts beyond R."""
    c = rng.choice(cost, size=R)
    heap = [(0.0, b) for b in range(B)]  # (time lane becomes free, lane)
    heapq.heapify(heap)
    finish = 0.0
    for i in range(R):
        tfree, b = heapq.heappop(heap)
        tend = tfree + c[i]
        finish = max(finish, tend)
        heapq.heappush(heap, (tend, b))
    return c.sum() / (B * finish)


for name, cost in [("toy C", C.astype(float)), ("Geometric(1/50)", geo.astype(float))]:
    for B in [64, 1024]:
        effs = []
        for ratio in [1, 4, 16, 64]:
            e = np.mean([refill_efficiency(cost, B, ratio * B, rng) for _ in range(5)])
            effs.append(f"R/B={ratio}: {e:.2f}")
        print(f"refill lockstep lane utilisation [{name}, B={B}]: " + ", ".join(effs))

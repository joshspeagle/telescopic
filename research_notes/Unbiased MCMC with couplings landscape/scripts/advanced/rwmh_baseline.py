import numpy as np, time
from ensemble_coupling_sim import run_rwmh_coupled, summarize
t0 = time.time()
for d in (1, 2, 5, 10, 20, 50):
    cap = 50000
    ts = [run_rwmh_coupled(d, T=cap, seed=200 + s) for s in range(10)]
    print(f"rwmh d={d:3d}: {summarize(ts, cap)}", flush=True)
print(f"elapsed {time.time()-t0:.1f}s")

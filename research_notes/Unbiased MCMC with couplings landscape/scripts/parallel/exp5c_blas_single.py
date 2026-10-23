import os, sys, time
import numpy as np
n, reps = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(0); a = rng.standard_normal((n, n))
t0 = time.perf_counter()
for _ in range(reps):
    b = a @ a.T
print(f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS','default')} n={n} reps={reps}: {time.perf_counter()-t0:.2f} s")

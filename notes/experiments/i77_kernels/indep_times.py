"""Independence at d = 2 and 256 lanes, before and after #77, in 8 alternating pairs of processes (issue #173).

The gate of the design read 1.051 for this configuration from 3 runs of step_times.py; this
repeats only it, 8 times per side, each run the median of 9 rounds of 1,000 calls, CPU time.
Run alternately with PYTHONPATH set to each package, threads pinned to 1; indep_times.out holds
the runs (base at b7bdf47, new at this branch) and their ratios.
"""

import time

import numpy as np

from telescopic import examples
from telescopic.kernels import Independence, MultivariateT
from telescopic.kernels.base import evaluate
from telescopic.target import Target
from telescopic.testing import random_draws

ex = examples.gaussian(2)
t = Target(ex.log_prob, grad=ex.grad, vectorize=True)
k = Independence(MultivariateT(np.zeros(2), 2 * np.eye(2)))
rng = np.random.default_rng(0)
s = evaluate(t, rng.standard_normal((256, 2)), False)
s2 = evaluate(t, rng.standard_normal((256, 2)), False)
d = random_draws(256, 1)
out = []
for op in ("step", "coupled"):
    f = (lambda: k.step(t, s, d)) if op == "step" else (lambda: k.coupled_step(t, s, s2, d))
    for _ in range(300):
        f()
    ts = []
    for _ in range(9):
        a = time.process_time()
        for _ in range(1000):
            f()
        ts.append((time.process_time() - a) * 1e3)
    out.append(f"{op} {np.median(ts):.2f}")
print(" ".join(out))

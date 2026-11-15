"""MALA's coupled step at d = 2 and 256 lanes, run alternately with PYTHONPATH set to each package; the output is mala_coupled_times.out."""

import time

import numpy as np

from telescopic import examples
from telescopic.kernels import MALA
from telescopic.target import Target
from telescopic.testing import random_draws

ex = examples.gaussian(2)
t = Target(ex.log_prob, grad=ex.grad, vectorize=True)
k = MALA(1.65 * 2 ** (-1 / 6), cov=np.eye(2))
rng = np.random.default_rng(0)
s, s2 = (k.init_state(t, rng.standard_normal((256, 2))) for _ in range(2))
d = random_draws(256, 1)
f = lambda: k.coupled_step(t, s, s2, d)
for _ in range(300):
    f()
ts = []
for _ in range(9):
    a = time.process_time()
    for _ in range(1000):
        f()
    ts.append((time.process_time() - a) * 1e3)
print(f"{np.median(ts):.2f}")

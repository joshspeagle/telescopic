"""Peak memory of a Student-t independence proposal of large df, by lanes (M11 code review).

`MultivariateT` draws df normals per point, and each lane's buffer holds 8 steps of them,
so memory grows with df and the lanes. The peak memory traced by `tracemalloc` over a short
run of 64 and of 256 replicates, one lane each, at df = 10^5, in 2 dimensions.

Run: python notes/experiments/m11_t_memory.py
"""

import tracemalloc
import warnings

import numpy as np

from telescopic import engine
from telescopic.initial import Normal
from telescopic.kernels import Independence
from telescopic.kernels.independence import MultivariateT
from telescopic.target import Target

target = Target(lambda x: -0.5 * np.sum(x**2, axis=1), vectorize=True)
kernel = Independence(MultivariateT(np.zeros(2), np.eye(2), df=100_000))
for replicates in (64, 256):
    tracemalloc.start()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.run(
            target,
            kernel,
            Normal(np.zeros(2), np.eye(2)),
            range(replicates),
            k=0,
            m=2,
            lag=1,
            rng=1,
        )
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"df = 10^5, {replicates} lanes: peak {peak / 1e6:.0f} MB")

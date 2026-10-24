# sample()'s tuning and pilot on N(0, diag(sd^2)) from N(0, 4 diag(sd^2)), as the audit ran.
import time
import warnings

import numpy as np

import telescopic

warnings.simplefilter("ignore")
for ratio in [1.0, 1e8, 1e10, 1e16]:
    sd = np.array([1 / np.sqrt(ratio), np.sqrt(ratio)])

    def log_prob(x, sd=sd):
        return -0.5 * np.sum((x / sd) ** 2, axis=-1)

    t0 = time.time()
    sampler = telescopic.CoupledSampler(
        log_prob, telescopic.initial.Normal(np.zeros(2), 4 * sd**2), vectorize=True, rng=0
    )
    sampler.tune()
    times = sampler.meeting_times(100, max_iter=20_000)
    print(
        f"sd ratio {ratio:g}: tuned sd / sd {np.sqrt(np.diag(sampler.kernel.cov)) / sd}, "
        f"pilot meeting times median {np.median(times):g}, max {np.max(times):g}; "
        f"{time.time() - t0:.1f} s"
    )

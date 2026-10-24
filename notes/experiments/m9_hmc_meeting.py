"""Meeting times of the automatically tuned HMC, against two remedies (M9).

With its defaults (dual averaging toward acceptance 0.8, meeting scale 1e-3), HMC tuned by
the sampler on the banana from U[-5, 5]^2 left 88 of 100 pilot pairs unmet after 10^5
iterations. Heng & Jacob (2019) pair the meeting scale 1e-3 with a step of 1/500, half of
it. On the banana and on examples.gaussian(10) from N(0, 4I), seed 4: the tuned step at
target acceptances 0.8 and 0.95, each with meeting scales 1e-3 and half the step; 100
lag-1 pairs capped at 20,000 iterations.

Run: python notes/experiments/m9_hmc_meeting.py
"""

import time
import warnings

import numpy as np

from telescopic import CoupledSampler, examples
from telescopic.engine import run
from telescopic.kernels import HMC
from telescopic.kernels.hmc import _Leapfrog
from telescopic.target import Target

for acceptance in (0.8, 0.95):
    _Leapfrog.target_acceptance = acceptance
    for example in (examples.banana(), examples.gaussian(10)):
        sampler = CoupledSampler(
            example.log_prob,
            example.initial,
            grad=example.grad,
            vectorize=True,
            kernel="hmc",
            rng=4,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler.tune()
        tuned = sampler.kernel
        step = tuned.step_size
        target = Target(example.log_prob, grad=example.grad, vectorize=True)
        for scale in (1e-3, step / 2):
            kernel = HMC(step, tuned.n_steps, cov=tuned.cov, meeting_scale=scale)
            start = time.perf_counter()
            tau = run(
                target,
                kernel,
                example.initial,
                range(100),
                k=0,
                m=0,
                lag=1,
                rng=4,
                store=False,
                max_iter=20_000,
            ).meeting_time
            met = tau[np.isfinite(tau)]
            quantiles = np.quantile(met, [0.5, 0.99]).round() if len(met) else "none"
            print(
                f"{example.name}, acceptance {acceptance}: step {step:.4f} x {tuned.n_steps}, "
                f"meeting scale {scale:.4g}: {np.sum(~np.isfinite(tau))} of 100 unmet at "
                f"20,000; met q50/q99 {quantiles}; {time.perf_counter() - start:.1f} s",
                flush=True,
            )

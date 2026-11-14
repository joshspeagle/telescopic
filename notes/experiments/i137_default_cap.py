"""What a user sees at the package's own cap with HMC(n_steps=3) on examples.gaussian(10). (#137)

Question: i137_baseline.py caps the pilot's pairs at 10,000 iterations, a tenth of the
package's cap, and there `k` cannot be chosen at 29 of 30 seeds of examples.gaussian(10)
with `HMC(n_steps=3)`. With the package's own pilot (100 pairs capped at 100,000
iterations), what would `CoupledSampler(..., kernel=HMC(n_steps=3)).run` choose at seeds
0 to 9: `k`, or the error? And the default `HMC()` as a control? The script runs the
pilot and its choice of `k` as `run` does (`meeting_times` with its defaults, then
`choose_k`), without the replicates.

Design: for each seed and kernel, the tuned step and n_steps, the pilot pairs unmet at the
cap, the median meeting time of the pairs that met, and `k` or "not chosen". The target is
the package's example, vectorized, with its initial distribution N(0, 4I).

Result (i137_default_cap.out): with `HMC(n_steps=3)`, `k` could not be chosen at 2 of 10
seeds (seeds 1 and 7, with 14 and 96 of 100 pairs unmet at 100,000), and at the other 8
it was 12,377 to 52,065, against 83 to 113 with `HMC()`, which chose `k` at every seed.
At the package's cap the resonance therefore shows more often as a `k` hundreds of times
too large than as the error.

Run: PYTHONPATH=src:notes/experiments python notes/experiments/i137_default_cap.py
(needs nothing first; it runs in one go, on one thread)
"""

import warnings

import numpy as np
from i85_setup import setup

import telescopic
from telescopic.kernels import HMC
from telescopic.sampler import PILOT_MAX_ITER
from telescopic.tuning import choose_k

SEEDS = range(10)
example = telescopic.examples.gaussian(10)

print(setup())
print(
    f"\nseed, kernel, tuned step, n_steps, pilot pairs unmet at {PILOT_MAX_ITER:,}, median "
    "meeting time of the pairs that met, k"
)
for name, make in (("HMC(n_steps=3)", lambda: HMC(n_steps=3)), ("HMC()", HMC)):
    chosen = []
    for seed in SEEDS:
        sampler = telescopic.CoupledSampler(
            example.log_prob,
            example.initial,
            grad=example.grad,
            vectorize=True,
            kernel=make(),
            rng=seed,
        )
        # The pilot as `run` makes it, and its choice of k, without the replicates.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pilot = sampler.meeting_times()
        try:
            k = choose_k(pilot)
            chosen.append(k)
        except ValueError:
            k = "not chosen"
        met = pilot[np.isfinite(pilot)]
        leapfrog = sampler.kernel.kernels[0]
        print(
            f"seed={seed} {name:15} step={leapfrog.step_size:.4f} n_steps={leapfrog.n_steps} "
            f"unmet={len(pilot) - len(met):3} median={np.median(met):8.1f} k={k}"
        )
    print(
        f"{name}: k chosen at {len(chosen)} of {len(SEEDS)} seeds, from {min(chosen):,} to "
        f"{max(chosen):,}"
    )

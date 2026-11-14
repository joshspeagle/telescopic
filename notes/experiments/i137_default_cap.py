"""What a user sees at the package's own cap with HMC(n_steps=3) on examples.gaussian(10). (#137)

Question: i137_baseline.py caps the pilot's pairs at 10,000 iterations, a tenth of the
package's cap, and there `k` cannot be chosen at 29 of 30 seeds of examples.gaussian(10)
with `HMC(n_steps=3)`. With the package's own pilot (100 pairs capped at 100,000
iterations), what would `CoupledSampler(..., kernel=HMC(n_steps=3)).run` choose at seeds
0 to 9: `k`, or the error? And the default `HMC()` as a control?

Design: the baseline's `run_pilot` (the package's tuning, 100 lag-1 pilot pairs, then
`choose_k`, as `run` does, without the replicates) at the package's cap. For each seed
and kernel: the tuned step and n_steps, the pilot pairs unmet at the cap, the median
meeting time (inf where more than half are unmet), and `k` or "not chosen".

Result (i137_default_cap.out): with `HMC(n_steps=3)`, `k` could not be chosen at 2 of 10
seeds (seeds 1 and 7, with 14 and 96 of 100 pairs unmet at 100,000), and at the other 8
it was 12,377 to 52,065, against 83 to 113 with `HMC()`, which chose `k` at every seed.
At the package's cap the resonance therefore shows more often as a `k` hundreds of times
too large than as the error.

Run: PYTHONPATH=src:notes/experiments python notes/experiments/i137_default_cap.py
(needs nothing first; it runs in one go, on one thread)
"""

from i85_setup import setup
from i137_common import gaussian_target, run_pilot

from telescopic.sampler import PILOT_MAX_ITER

SEEDS = range(10)
example = gaussian_target(10)

print(setup())
print(
    f"\nseed, kernel, tuned step, n_steps, pilot pairs unmet at {PILOT_MAX_ITER:,}, median "
    "meeting time, k"
)
for n_steps, name in ((3, "HMC(n_steps=3)"), (None, "HMC()")):
    records = [run_pilot(example, "base", n_steps, seed, PILOT_MAX_ITER) for seed in SEEDS]
    for seed, r in zip(SEEDS, records, strict=True):
        print(
            f"seed={seed} {name:15} step={r['eps']:.4f} n_steps={r['n']} unmet={r['unmet']:3} "
            f"median={r['median']:8.1f} k={'not chosen' if r['error'] else r['k']}"
        )
    chosen = [r["k"] for r in records if not r["error"]]
    print(
        f"{name}: k chosen at {len(chosen)} of {len(SEEDS)} seeds, from {min(chosen):,} to "
        f"{max(chosen):,}"
    )

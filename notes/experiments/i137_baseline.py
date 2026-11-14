"""Baseline: where does HMC(n_steps=n) with an automatic step fail to give k? (#137)

Question: on examples.gaussian(d), d = 2 and 10, with n_steps in (default, 1, 2, 3, 5,
10), at how many seeds can k not be chosen, what step does tuning return, what is the
contraction factor of the coupled trajectory on each mode of the tuned covariance, and
what are the pilot's meeting times?

Design (package at the commit on the first line of the output):
- Tuning only, seeds 0 to 99: `CoupledSampler(..., kernel=HMC(n_steps=n), rng=seed).tune()`.
  For the tuned step eps, n and covariance C C^T, the whitened frequencies w are the
  square roots of the eigenvalues of C^T P C, P the exact precision, and the factor of a
  mode is |cos(n theta)|, theta = 2 arcsin(eps w / 2), the leapfrog's |T_n(1 - h^2/2)|.
- Pilot, seeds 0 to 29: after the same tuning, 100 lag-1 pairs as `sample` runs them,
  but capped at 10,000 iterations instead of the package's 100,000, then `choose_k`.
- Seed 0, d = 2, n_steps = 2, at the package's cap of 100,000, to tie the cut design to
  the issue's figures.

Cuts, against the validation of #137 (100 seeds, cap 100,000, about 2.3 CPU-hours): 30
seeds with pairs, and the cap of 10,000. A pair's streams do not depend on the cap, so a
pair unmet at 100,000 is unmet at 10,000: the cut design can only add failures. It adds
those seeds whose factor puts the meeting between 10^4 and 10^5 iterations, which is the
same defect, measured at a smaller budget; remedies are held to the same cap. The 100
tuning-only seeds give the factors, which the validation found to separate failing from
passing seeds without overlap.

Result (i137_baseline.out): the defect reproduces. At cap 10,000, k cannot be chosen at
12 of 30 seeds for d = 2, n_steps = 2, at 29 of 30 for d = 10, n_steps = 3 and at 11 of
30 for d = 10, n_steps = 10; at none of 30 in the other 9 configurations, the default
included. Over 100 tunings the largest factor exceeds 0.999 at 38 (d = 2, n = 2), 69
(d = 10, n = 3) and 34 (d = 10, n = 10) seeds, and never elsewhere (largest 0.991). The
failing configurations cost 11,956, 50,470 and 35,851 gradients per pilot pair (median
over seeds) against 60 to 1,798 elsewhere. Seed 0, d = 2, n_steps = 2 at the package's
cap of 100,000 gives the issue's figures exactly: step 1.4516, 31 pairs unmet. The
tuned covariance's whitened frequencies are within 0.099 of 1 at d = 2 and within 0.286
at d = 10.

Run: PYTHONPATH=src:notes/experiments python notes/experiments/i137_baseline.py
(rerun until it prints the tables; each run stops after 9 minutes and resumes)
"""

import sys

import numpy as np
from i85_setup import setup
from i137_common import Budget, Store, gaussian_target, run_pilot

CONFIGS = [(d, n) for d in (2, 10) for n in (None, 1, 2, 3, 5, 10)]
TUNE_SEEDS, PILOT_SEEDS, CAP = range(100), range(30), 10_000

store = Store("i137_baseline.jsonl")
budget = Budget()
work = [("tune", d, n, s, 0) for d, n in CONFIGS for s in TUNE_SEEDS]
work += [("pilot", d, n, s, CAP) for d, n in CONFIGS for s in PILOT_SEEDS]
work += [("issue", 2, 2, 0, 100_000)]
for kind, d, n, seed, cap in work:
    key = f"{kind} d={d} n={n} seed={seed}"
    if key in store:
        continue
    if not budget.left():
        print("incomplete: rerun to resume")
        sys.exit(0)
    record = run_pilot(gaussian_target(d), "base", n, seed, cap)
    record.update(kind=kind, d=d)
    store.add(key, record)

print(setup())
records = list(store.records.values())


def pick(kind, d, n):
    return [r for r in records if r["kind"] == kind and r["d"] == d and r["n_steps"] == n]


print(
    "\nRaw pilot table (cap 10,000; 100 pairs): d, n_steps, seed, eps, n, max factor, "
    "unmet, median, q90, max finite, k, pilot gradients per pair, tuning gradients"
)
for d, n in CONFIGS:
    for r in sorted(pick("pilot", d, n), key=lambda r: r["seed"]):
        print(
            f"d={d:2} n_steps={n!s:4} seed={r['seed']:2} eps={r['eps']:.4f} n={r['n']:2} "
            f"factor={r['factor_max']:.6f} unmet={r['unmet']:3} median={r['median']:8.1f} "
            f"q90={r['q90']:8.1f} max={r['max_finite']} k={r['k']} "
            f"grads/pair={r['pilot_grads_per_pair']:.0f} tune_grads={r['tune_grads']}"
        )

print(
    "\nTuning only, seeds 0-99: eps median [min, max]; largest factor median and max; seeds "
    "with largest factor above 0.9998 (the validation's failing range at cap 1e5) and 0.999"
)
for d, n in CONFIGS:
    rs = pick("tune", d, n)
    e = np.array([r["eps"] for r in rs])
    f = np.array([r["factor_max"] for r in rs])
    print(
        f"d={d:2} n_steps={n!s:4}: seeds {len(rs)}, eps {np.median(e):.3f} [{e.min():.3f}, "
        f"{e.max():.3f}], n {sorted({r['n'] for r in rs})}, factor median {np.median(f):.4f} max "
        f"{f.max():.6f}, >0.9998: {(f > 0.9998).sum()}, >0.999: {(f > 0.999).sum()}"
    )
for d in (2, 10):
    dev = np.array(
        [
            max(abs(r["w_min"] - 1), abs(r["w_max"] - 1))
            for r in records
            if r["kind"] == "tune" and r["d"] == d
        ]
    )
    print(
        f"d={d:2}: largest |w - 1| of a seed's modes, over {len(dev)} tunings: median "
        f"{np.median(dev):.3f}, 95% {np.quantile(dev, 0.95):.3f}, max {dev.max():.3f}"
    )

print(
    "\nPilot summary, seeds 0-29, cap 10,000: seeds where k cannot be chosen; unmet pairs in "
    "all; median over seeds of the seed's median meeting time [min, max]; k median over the "
    "seeds that pass [max]; mean gradients per pilot pair, median over seeds; tuning gradients"
)
for d, n in CONFIGS:
    rs = pick("pilot", d, n)
    med = np.array([r["median"] for r in rs])
    ks = [r["k"] for r in rs if r["k"] is not None]
    fails = sorted(r["seed"] for r in rs if r["error"])
    print(
        f"d={d:2} n_steps={n!s:4}: k fails at {len(fails):2} of {len(rs)} {fails}, unmet "
        f"{sum(r['unmet'] for r in rs):4}, median tau {np.median(med):.1f} [{med.min():.1f}, "
        f"{med.max():.1f}], k {np.median(ks) if ks else None} [{max(ks) if ks else None}], "
        f"grads/pair {np.median([r['pilot_grads_per_pair'] for r in rs]):.0f}, tune grads "
        f"{np.median([r['tune_grads'] for r in rs]):.0f}"
    )
r = records[[x["kind"] for x in records].index("issue")]
print(
    f"\nSeed 0, d=2, n_steps=2, cap 100,000: eps {r['eps']:.4f}, factor {r['factor_max']:.6f}, "
    f"unmet {r['unmet']}, k {r['k']}, error {r['error']}"
)

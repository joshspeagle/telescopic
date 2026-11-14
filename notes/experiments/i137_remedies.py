"""The remedies on the baseline's table: do they let k be chosen, and at what cost? (#137)

Question: with remedies a to d of i137_common.py, on examples.gaussian(d), d = 2 and 10,
n_steps in (default, 1, 2, 3, 5, 10), at how many of seeds 0 to 29 can k not be chosen
from 100 lag-1 pilot pairs capped at 10,000 iterations, and what are the meeting times
and the gradient evaluations of tuning and of a pilot pair? The baseline is
i137_baseline.py, with the same seeds, cap and pairs.

Design: as the baseline's pilot part, with the remedy's kernel. Each cell is tuned
first (run_pilot with cap 0); where a remedy returns the package's kernel exactly (a
whose guard keeps the step; d whose width is c's 0.5), its run would repeat the
baseline's or c's bit for bit, so that record is copied and marked "same as".
Falsification test (i137_predictions.md): a remedy fails if k cannot be chosen at more
than 2 of the 30 seeds of any configuration (the default HMC() gives k at every seed of
both targets).

Result (i137_remedies.out): every remedy gives k at all 30 seeds of all 12
configurations (360 of 360 each), where the package fails at 12, 29 and 11 of 30 seeds of
three configurations; no remedy fails the falsification test here. Median over seeds of
the seed's median meeting time, package / a / b / c / d: d = 2, n_steps = 2: 2,478 /
27.0 / 16.5 / 31.2 / 31.2; d = 10, n_steps = 3: inf / 46.2 / 24.0 / 37.0 / 37.0; d = 10,
n_steps = 10: 1,777 / 44.2 / 39.2 / 42.8 / 44.0. Where the package does not fail, c and d
cost 2 to 16% more gradients per pilot pair than the package at n_steps = default, 1 or
2 (for example 137 against 121 at d = 10, default), and less at n_steps = 3, 5 and 10 at
d = 2 and n_steps = 5 at d = 10, whose packaged steps sit nearer resonances. b has the
shortest median meeting times of the five kernels in 10 of the 12 configurations; at
d = 10 with the default n_steps and with n_steps = 2 its 25.8 and 26.0 are above the
package's 25.2. It tunes with 1.6 to 2.3 times the package's tuning gradients. The guard
of a met its rule at every seed (no fallback) and never raised the step here: it kept
the step at every seed of the default n_steps and n_steps = 1 at both d and of
n_steps = 2 at d = 10, and lowered it at every seed of the other 7 configurations (to
0.21 for n_steps = 10), among them 4 where the package already gives k at every seed:
d = 2 with n_steps = 3, 5 and 10 (largest factors 0.85, 0.99 and 0.98) and d = 10 with
n_steps = 5 (0.99). Where it lowers the step, its predicted worst factor at the returned
step is within 0.0014 of the bound 0.9.

Run: PYTHONPATH=src:notes/experiments python notes/experiments/i137_remedies.py
(needs the store of i137_baseline.py first, whose pilot records it copies and prints;
rerun until it prints the tables: each run starts no new record after 9 minutes, and the
next resumes from the store in notes/experiments/_rows/). From nothing, the 1,080
records it computes took 502 s on one thread by the store's `seconds`; that leaves out
the tuning of each of the 1,440 cells that precedes its record (a tuning of the
package's kernel takes 392 s / 1,200 = 0.33 s by the baseline's store, and b's more),
and the 360 copies carry their source's 127 s, which this script does not spend.
"""

import sys

import numpy as np
from i137_common import JITTER, Budget, Store, gaussian_target, guard_summary, header, run_pilot

CONFIGS = [(d, n) for d in (2, 10) for n in (None, 1, 2, 3, 5, 10)]
SEEDS, CAP, METHODS = range(30), 10_000, ("a", "b", "c", "d")

base = Store("i137_baseline.jsonl")
store = Store("i137_remedies.jsonl")
budget = Budget()
for method in METHODS:
    for d, n in CONFIGS:
        for seed in SEEDS:
            key = f"{method} d={d} n={n} seed={seed}"
            if key in store:
                continue
            if not budget.left():
                print("incomplete: rerun to resume")
                sys.exit(0)
            example = gaussian_target(d)
            tuned = run_pilot(example, method, n, seed, 0)
            same = None
            if method == "a" and tuned["eps"] == tuned["extra"]["eps_da"]:
                same = f"pilot d={d} n={n} seed={seed}"
                record = dict(base.records[same])
            elif method == "d" and tuned["jitter"] == JITTER:
                same = f"c d={d} n={n} seed={seed}"
                record = dict(store.records[same])
            else:
                record = run_pilot(example, method, n, seed, CAP)
            record.update(method=method, d=d, same_as=same, extra=tuned["extra"])
            store.add(key, record)

print(header(base, store))
rows = list(store.records.values()) + [
    dict(r, method="base") for r in base.records.values() if r["kind"] == "pilot"
]


def pick(method, d, n):
    return sorted(
        (r for r in rows if r["method"] == method and r["d"] == d and r["n_steps"] == n),
        key=lambda r: r["seed"],
    )


print(
    "\nRaw table (cap 10,000; 100 pairs): method, d, n_steps, seed, eps (eps of dual "
    "averaging), n, jitter, largest factor (geometric mean over the jitter for c and d), "
    "unmet, median, q90, max finite, k, pilot gradients per pair, tuning gradients, copy; "
    "for a, whether a step of the guard's grid met its rule and the guard's predicted worst "
    "factor at the returned step"
)
for method in METHODS:
    for d, n in CONFIGS:
        for r in pick(method, d, n):
            extra = r.get("extra") or {}
            print(
                f"{method} d={d:2} n_steps={n!s:4} seed={r['seed']:2} eps={r['eps']:.4f} "
                f"({extra.get('eps_da', r['eps']):.4f}) n={r['n']:2} j={r['jitter']:.3f} "
                f"factor={r['factor_max']:.6f} unmet={r['unmet']:3} median={r['median']:8.1f} "
                f"q90={r['q90']:8.1f} max={r['max_finite']} k={r['k']} "
                f"grads/pair={r['pilot_grads_per_pair']:.0f} tune_grads={r['tune_grads']} "
                f"same_as={r.get('same_as')}"
                + (
                    f" passed={extra['passed']} worst={extra['worst']:.6f}"
                    if method == "a"
                    else ""
                )
            )

print(
    "\nSummary, seeds 0-29: seeds where k cannot be chosen; unmet pairs in all; median over "
    "seeds of the seed's median meeting time [min, max]; median and max k; median over seeds "
    "of the mean gradients per pilot pair [max]; median tuning gradients; largest factor, "
    "median over seeds [max]"
)
for d, n in CONFIGS:
    for method in ("base", *METHODS):
        rs = pick(method, d, n)
        med = np.array([r["median"] for r in rs])
        ks = [r["k"] for r in rs if r["k"] is not None]
        g = np.array([r["pilot_grads_per_pair"] for r in rs])
        f = np.array([r["factor_max"] for r in rs])
        fails = [r["seed"] for r in rs if r["error"]]
        print(
            f"d={d:2} n_steps={n!s:4} {method:4}: k fails {len(fails):2}/{len(rs)}, unmet "
            f"{sum(r['unmet'] for r in rs):4}, median tau {np.median(med):7.1f} [{med.min():.1f}, "
            f"{med.max():.1f}], k {np.median(ks) if ks else None} [{max(ks) if ks else None}], "
            f"grads/pair {np.median(g):.0f} [{g.max():.0f}], tune grads "
            f"{np.median([r['tune_grads'] for r in rs]):.0f}, factor {np.median(f):.4f} "
            f"[{f.max():.6f}]"
        )
    print()

print(
    "Remedy a's guard, seeds 0-29: seeds where it kept the step, lowered it, raised it, or "
    "fell back (no step of its grid met the rule)"
)
for d, n in CONFIGS:
    print(f"d={d:2} n_steps={n!s:4}: {guard_summary(pick('a', d, n))}")

print("\nFalsification (k fails at more than 2 of 30 seeds in some configuration):")
for method in METHODS:
    worst = max(
        (sum(r["error"] is not None for r in pick(method, d, n)), str(d), str(n))
        for d, n in CONFIGS
    )
    where = (
        f"worst configuration d={worst[1]} n_steps={worst[2]} with {worst[0]} failing seeds"
        if worst[0]
        else "no failing seed in any configuration"
    )
    print(f"{method}: {where} -> {'FAILS' if worst[0] > 2 else 'passes'}")

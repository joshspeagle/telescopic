"""The remedies on the baseline's table: do they let k be chosen, and at what cost? (#137)

Question: with remedies a to d of i137_common.py, on examples.gaussian(d), d = 2 and 10,
n_steps in (default, 1, 2, 3, 5, 10), at how many of seeds 0 to 29 can k not be chosen
from 100 lag-1 pilot pairs capped at 10,000 iterations, and what are the meeting times
and the gradient evaluations of tuning and of a pilot pair? The baseline is
i137_baseline.py, with the same seeds, cap and pairs.

Design: as the baseline's pilot part, with the remedy's kernel. Where a remedy returns
the package's kernel exactly (a whose guard keeps the step; d whose width is c's 0.5),
its run would repeat the baseline's or c's bit for bit, so that record is copied and
marked "same as". Falsification test (i137_predictions.md): a remedy fails if k cannot be chosen
at more than 2 of the 30 seeds of any configuration (the default HMC() gives k at every
seed of both targets).

Result (i137_remedies.out): every remedy gives k at all 30 seeds of all 12
configurations (360 of 360 each), where the package fails at 12, 29 and 11 of 30 seeds of
three configurations; no remedy fails the falsification test here. Median over seeds of
the seed's median meeting time, package / a / b / c / d: d = 2, n_steps = 2: 2,478 /
27.0 / 16.5 / 31.2 / 31.2; d = 10, n_steps = 3: inf / 46.2 / 24.0 / 37.0 / 37.0; d = 10,
n_steps = 10: 1,777 / 44.2 / 39.2 / 42.8 / 44.0. Where the package does not fail, c and d
cost 2 to 16% more gradients per pilot pair than the package at n_steps = default, 1 or
2 (for example 137 against 121 at d = 10, default), and less at n_steps = 3, 5 and 10 at
d = 2 and n_steps = 5 at d = 10, whose packaged steps sit nearer resonances. b has the
shortest or equal meetings everywhere but tunes with 1.6 to 2.3 times the package's
tuning gradients; a changes nothing where the package already contracts, and otherwise shortens
the step (to 0.21 for n_steps = 10).

Run: PYTHONPATH=src:notes/experiments python notes/experiments/i137_remedies.py
(rerun until it prints the tables; each run stops after 9 minutes and resumes)
"""

import sys

import numpy as np
from i85_setup import setup
from i137_common import Budget, Store, gaussian_target, run_pilot

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
            elif method == "d" and tuned["jitter"] == 0.5:
                same = f"c d={d} n={n} seed={seed}"
                record = dict(store.records[same])
            else:
                record = run_pilot(example, method, n, seed, CAP)
            record.update(method=method, d=d, same_as=same, extra=tuned["extra"])
            store.add(key, record)

print(setup())
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
    "unmet, median, q90, max finite, k, pilot gradients per pair, tuning gradients, copy"
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

print("Falsification (k fails at more than 2 of 30 seeds in some configuration):")
for method in METHODS:
    worst = max(
        (sum(r["error"] is not None for r in pick(method, d, n)), str(d), str(n))
        for d, n in CONFIGS
    )
    print(
        f"{method}: worst configuration d={worst[1]} n_steps={worst[2]} with {worst[0]} "
        f"failing seeds -> {'FAILS' if worst[0] > 2 else 'passes'}"
    )

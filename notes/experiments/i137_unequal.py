"""The remedies on a Gaussian whose modes no single step keeps off resonance (#137).

Question: on N(0, diag(s^2)) in 10 dimensions with scales s from 1 to 1/4 (geometric),
and HMC given the identity as its covariance, so that the whitened frequencies w = 1/s
spread from 1 to 4 and no step keeps every mode away from every resonance of a given
n_steps, can k be chosen, and at what cost, with the package's tuning and remedies a to d
of i137_common.py? (With its covariance tuned, the pilot whitens the scales away and the
case reduces to i137_remedies.py.)

Design: `CoupledSampler(..., kernel=K(n_steps=n, cov=np.eye(10)), rng=seed)`, initial
N(0, 4 I) as examples.gaussian, n_steps in (default, 3, 10), seeds 0 to 9, tuning then
100 lag-1 pilot pairs capped at 10,000 iterations, then choose_k. The guard of remedy a
and the width of remedy d read the frequencies from the pilot covariance S, as the square
roots of the eigenvalues of S^-1. Records of kernels identical to another method's are
copied, as in i137_remedies.py.

Result (i137_unequal.out): the package's default HMC(cov=I) itself fails here: k cannot
be chosen at 3 of 10 seeds (step 0.30 to 0.32, 4 leapfrog steps, a mode with factor
0.9998 to 0.999997), and the seeds that pass need a median meeting time of 672 to 3,264.
With n_steps = 3 and 10 every seed passes, with median meeting times of 720 and 1,381.
Remedy c (= d here: the width rule gives 0.5) gives k at every seed of all three
configurations, with median meeting times of 43.0, 39.2 and 44.8 and 360, 260 and 933
gradients per pilot pair, against 14,870, 4,422 and 27,123 for the package. The guard a
fails at 3 of 10 seeds of the default (it predicts the frequencies from the pilot
covariance, which is not exact enough) and b at 2 of 10; both pass with n_steps = 3 and
10, where a needs 72 to 79 iterations and b 82 to 291.

Run: PYTHONPATH=src:notes/experiments python notes/experiments/i137_unequal.py
(rerun until it prints the tables; each run stops after 9 minutes and resumes)
"""

import sys

import numpy as np
from i85_setup import setup
from i137_common import Budget, Store, gaussian_target, run_pilot

D = 10
SCALES = np.geomspace(1, 0.25, D)
CONFIGS, SEEDS, CAP, METHODS = (None, 3, 10), range(10), 10_000, ("base", "a", "b", "c", "d")

store = Store("i137_unequal.jsonl")
budget = Budget()
example = gaussian_target(D, SCALES)
for n in CONFIGS:
    for seed in SEEDS:
        for method in METHODS:
            key = f"{method} n={n} seed={seed}"
            if key in store:
                continue
            if not budget.left():
                print("incomplete: rerun to resume")
                sys.exit(0)
            tuned = run_pilot(example, method, n, seed, 0, cov=np.eye(D))
            same = None
            if method == "a" and tuned["eps"] == tuned["extra"]["eps_da"]:
                same = f"base n={n} seed={seed}"
            elif method == "d" and tuned["jitter"] == 0.5:
                same = f"c n={n} seed={seed}"
            if same:
                record = dict(store.records[same])
            else:
                record = run_pilot(example, method, n, seed, CAP, cov=np.eye(D))
            record.update(method=method, same_as=same, extra=tuned["extra"])
            store.add(key, record)

print(setup())
rows = list(store.records.values())


def pick(method, n):
    return sorted(
        (r for r in rows if r["method"] == method and r["n_steps"] == n), key=lambda r: r["seed"]
    )


print(
    "\nRaw table (cap 10,000; 100 pairs): method, n_steps, seed, eps (eps of dual "
    "averaging), n, jitter, factor of the slowest and of the worst mode, unmet, median, "
    "q90, max finite, k, pilot gradients per pair, tuning gradients, copy"
)
for n in CONFIGS:
    for method in METHODS:
        for r in pick(method, n):
            extra = r.get("extra") or {}
            print(
                f"{method:4} n_steps={n!s:4} seed={r['seed']} eps={r['eps']:.4f} "
                f"({extra.get('eps_da', r['eps']):.4f}) n={r['n']:2} j={r['jitter']:.3f} "
                f"factor min {r['factor_min']:.4f} max {r['factor_max']:.6f} "
                f"unmet={r['unmet']:3} median={r['median']:8.1f} q90={r['q90']:8.1f} "
                f"max={r['max_finite']} k={r['k']} grads/pair={r['pilot_grads_per_pair']:.0f} "
                f"tune_grads={r['tune_grads']} same_as={r.get('same_as')}"
            )

print(
    "\nSummary, seeds 0-9: k failures; unmet pairs; median over seeds of the seed's median "
    "meeting time [min, max]; median over seeds of mean gradients per pilot pair [max]; "
    "tuning gradients; largest factor, median over seeds [max]"
)
for n in CONFIGS:
    for method in METHODS:
        rs = pick(method, n)
        med = np.array([r["median"] for r in rs])
        g = np.array([r["pilot_grads_per_pair"] for r in rs])
        f = np.array([r["factor_max"] for r in rs])
        fails = [r["seed"] for r in rs if r["error"]]
        print(
            f"n_steps={n!s:4} {method:4}: k fails {len(fails):2}/{len(rs)} {fails}, unmet "
            f"{sum(r['unmet'] for r in rs):4}, median tau {np.median(med):7.1f} "
            f"[{med.min():.1f}, {med.max():.1f}], grads/pair {np.median(g):.0f} "
            f"[{g.max():.0f}], tune grads {np.median([r['tune_grads'] for r in rs]):.0f}, "
            f"factor {np.median(f):.4f} [{f.max():.6f}]"
        )
    print()

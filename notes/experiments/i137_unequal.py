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
copied, as in i137_remedies.py. Two checks of the guard run in the print pass, without
pilot pairs: remedy a retuned at seeds 0 to 9 with the default n_steps (asserting that it
returns the stored step), to compare the pilot's frequencies with the exact 1/s; and the
guard's rule evaluated with exact frequencies, on its own grid and, for frequencies
geometric from 1 to r, on the wider grid 0.5 exp(s), s in [-5, 1.4) by 0.001.

Result (i137_unequal.out): the package's default HMC(cov=I) itself fails here: k cannot
be chosen at 3 of 10 seeds (step 0.30 to 0.31, 4 leapfrog steps, a mode with factor
0.9998 to 0.999997), and the seeds that pass need a median meeting time of 671 to 3,264.
With n_steps = 3 and 10 every seed passes, with median meeting times of 720 and 1,381.
Remedy c gives k at every seed of all three configurations, with median meeting times of
43.0, 39.2 and 44.8 and 360, 260 and 933 gradients per pilot pair, against 14,870, 4,422
and 27,123 for the package. Remedy d equals c where its width rule gives 0.5: at every
seed of the default n_steps and n_steps = 3, and at 7 of 10 seeds of n_steps = 10; at
seeds 1, 4 and 6 of n_steps = 10 its widths are 0.492, 0.477 and 0.475, and its median
meeting time over seeds is 45.0 against c's 44.8. b fails at 2 of 10 seeds of the
default, and passes with n_steps = 3 and 10 (median meeting times 82 and 291).

Remedy a fails at 3 of 10 seeds of the default, and the reason is not a poor
prediction of the frequencies: the pilot's frequencies are within 4.5 to 12% of the
exact ones, inside the guard's band of factors 1/1.3 to 1.3. On this target no step of
the guard's grid meets its rule, at any n_steps: `passed` is False in all 30 of its
records, and with the exact frequencies no step of the grid meets it either. Over
frequencies geometric from 1 to r, a step meeting the rule exists only up to r = 2.05
with the default n_steps, 3.40 with n_steps = 3 and 3.50 with n_steps = 10; here r = 4.
Every run of a therefore took the guard's fallback, the grid step of smallest predicted
worst factor. With the default n_steps that smallest factor is 0.9993 to 0.9998: every
step of the grid is predicted to be nearly as bad as a resonance, the fallback's choice
among them makes the pairs contract no faster than the package's step, and k cannot be
chosen at 3 seeds. With n_steps = 3 and 10 the fallback's steps have predicted worst factors of 0.918 to 0.932
and 0.914 to 0.928, and every seed passes, with median meeting times of 79 and 72.
Other fallbacks (such as the step of smallest factor over the predicted modes alone, or
a jitter where no step passes) were not tested.

The stored q90 is NaN where more than 10 pairs are unmet.

Run: PYTHONPATH=src:notes/experiments python notes/experiments/i137_unequal.py
(needs nothing first; rerun until it prints the tables: each run starts no new record
after 9 minutes, and the next resumes from the store in notes/experiments/_rows/). From
nothing, the 123 records it computes took 226 s on one thread by the store's `seconds`;
that leaves out the tuning of each of the 150 cells that precedes its record, and the
27 copies carry their source's 14 s, which this script does not spend. The print pass
adds the checks of the guard, about 10 s; its retuning returns the stored step only with a
package whose arithmetic is that of the records' commit, so with a later package that
tunes other bits the records are recomputed first.
"""

import sys

import numpy as np
from i137_common import (
    BOUND,
    GUARD_GRID,
    JITTER,
    Budget,
    GuardedHMC,
    Store,
    band,
    factor,
    gaussian_target,
    guard_summary,
    header,
    pilot_frequencies,
    run_pilot,
    worst_factor,
)

from telescopic import CoupledSampler
from telescopic.kernels import HMC

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
            elif method == "d" and tuned["jitter"] == JITTER:
                same = f"c n={n} seed={seed}"
            if same:
                record = dict(store.records[same])
            else:
                record = run_pilot(example, method, n, seed, CAP, cov=np.eye(D))
            record.update(method=method, same_as=same, extra=tuned["extra"])
            store.add(key, record)

print(header(store))
rows = list(store.records.values())


def pick(method, n):
    return sorted(
        (r for r in rows if r["method"] == method and r["n_steps"] == n), key=lambda r: r["seed"]
    )


print(
    "\nRaw table (cap 10,000; 100 pairs): method, n_steps, seed, eps (eps of dual "
    "averaging), n, jitter, factor of the slowest and of the worst mode, unmet, median, "
    "q90, max finite, k, pilot gradients per pair, tuning gradients, copy; for a, whether a "
    "step of the guard's grid met its rule and the guard's predicted worst factor at the "
    "returned step"
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
                + (
                    f" passed={extra['passed']} worst={extra['worst']:.6f}"
                    if method == "a"
                    else ""
                )
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

print(
    "Remedy a's guard, seeds 0-9: seeds where it kept the step, lowered it, raised it, or "
    "fell back (no step of its grid met the rule)"
)
for n in CONFIGS:
    print(f"n_steps={n!s:4}: {guard_summary(pick('a', n))}")

exact = np.sort(1 / SCALES)
print(
    "\nThe pilot's frequencies against the exact ones, 1/s: remedy a retuned with the default "
    "n_steps, its step checked against the store; largest |pilot w / exact w - 1|"
)
errors = []
for seed in SEEDS:
    sampler = CoupledSampler(
        example.log_prob,
        example.initial,
        vectorize=True,
        grad=example.grad,
        kernel=GuardedHMC(cov=np.eye(D)),
        rng=seed,
    )
    sampler.tune()
    assert sampler.kernel.step_size == store.records[f"a n=None seed={seed}"]["eps"]
    w = np.sort(pilot_frequencies(np.eye(D), sampler.tuning.cov))
    errors.append(np.abs(w / exact - 1).max())
print(f"seeds 0-9: {', '.join(f'{e:.3f}' for e in errors)}")

print(
    "\nThe guard's rule with the exact frequencies, on its grid around each stored eps_DA: "
    "records of a with a grid step that meets it; smallest worst factor on the grid [min, max]"
)
for n in CONFIGS:
    worst = [
        worst_factor(r["extra"]["eps_da"] * np.exp(GUARD_GRID), n, exact) for r in pick("a", n)
    ]
    least = [x.min() for x in worst]
    print(
        f"n_steps={n!s:4}: {sum((x <= BOUND).any() for x in worst)} of {len(worst)}; "
        f"{min(least):.6f} to {max(least):.6f}"
    )

print(
    "\nThe largest r, by 0.05 up to 4, for which frequencies geometric from 1 to r (10 modes) "
    f"have a step 0.5 exp(s), s in [-5, 1.4) by 0.001, whose worst factor over the band is at "
    f"most {BOUND}"
)
steps = 0.5 * np.exp(np.arange(-5000, 1400) / 1000)
for n in CONFIGS:
    counts = np.array([HMC(e, n).n_steps for e in steps])[:, None]
    feasible = [
        r
        for r in np.arange(1, 4.001, 0.05)
        if (factor(steps, counts, band(np.geomspace(1, r, D))).max(axis=1) <= BOUND).any()
    ]
    print(f"n_steps={n!s:4}: r up to {max(feasible):.2f}")

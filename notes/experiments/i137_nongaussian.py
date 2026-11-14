"""What do the remedies cost where there is no resonance? (#137)

Question: on two targets of the package's examples that are not Gaussian, with a given
n_steps, do remedies a to d of i137_common.py lengthen the pilot's meeting times or raise
the cost per effective replicate, against the package's current tuning?

Design: targets examples.banana() started from its exact distribution (x0 ~ N(1, 1/2),
x1 | x0 ~ N(x0^2, 1/20)), so that the costs compare the kernels on the target and not
the approach from its default U[-5, 5]^2, and examples.mixture() from its default N(10,
10^2). n_steps 2 and 5 on the banana, 2 on the mixture: the mixture with n_steps = 5 was
cut, to keep the experiment within its budget, after runs at seed 0 whose records were
not kept. Seeds 0 to 4. Each run is the package's recipe: `CoupledSampler(...,
rng=seed).run(R)` with the automatic k (100 lag-1 pilot pairs at the package's cap of
100,000), lag k and m = 10 k; R = 200 replicates. Costs are gradient evaluations counted
by the sampler's target: tuning, pilot and replicates. The cost per effective replicate
(inefficiency) of h is Var(H) times the replicates' gradient evaluations per replicate,
= SE^2 * n_grad of the replicates. z-scores against the exact mean: banana E[x0] = 1,
E[x1] = 1.5; mixture E[x] = 0, E[x^2] = 17. Where a remedy returns the package's kernel
exactly, the record is copied (as in i137_remedies.py). The stores do not hold remedy
a's guard record (passed, worst factor); the print pass retunes remedy a at each seed,
asserts that it returns the stored step, and prints the guard's record.

Falsification (pre-registered in i137_predictions.md): a remedy fails on a target and
n_steps if the mean over the 5 seeds of the seed's median pilot meeting time, or of the
inefficiency of h = x0, exceeds the package's mean by more than the standard deviation
of the package's 5 per-seed values.

Result (i137_nongaussian.out), mean (sd) over 5 seeds, package against remedies:
- banana, n_steps = 2: median meeting time 70.4 (7.0); a 73.9 (passes), b 104.8
  (fails), c = d 83.1 (fails on the median, +18%, though its inefficiency, 283, is below
  the package's 31,460, which one seed's heavy tail sets). The guard of a raised the
  step at 2 seeds and kept it at 3: at seed 1 the rule needs a longer trajectory than
  the tuned step gives (0.2836 to 0.2928), and at seed 3 no step of its grid meets the
  rule and its fallback took the grid's largest step (0.2483 to 0.2744); k went from
  815 to 4,555 and from 1,169 to 1,490.
- banana, n_steps = 5: package 256.1 (72.2), inefficiency 1.3e7 (1.7e7) from heavy tails
  (k up to 5,437); a identical; b 66.3 and 98; c = d 137.7 and 162: all pass.
- mixture, n_steps = 2: package 228.4 (26.3), inefficiency 44,339 (1,684); a identical;
  b 80.5 and 336,336 (fails: its step 0.17 to 0.33 lets fewer chains cross between the
  modes, k up to 19,347); c = d 67.6 and 57,218 (fails on the inefficiency, +29%: a
  nominal step of 0.33 against 0.38 and k 2,655 against 2,280 on average).
- Remedy d equals c in every run here (its width rule gives 0.5). Pooled z-scores over 5
  seeds are within 2.33 in absolute value (c, banana x1: -2.33; package, banana n = 5,
  x1: -1.89; b, banana n = 5, x0: +2.28), over 20 pooled z-scores of distinct kernels.

Limits: the inefficiency arm of the test compares with the standard deviation of the
package's 5 per-seed values, which one heavy-tailed seed sets: it is 2.19 times the
package's mean on the banana with n_steps = 2 and 1.32 times with n_steps = 5 (0.04 on
the mixture), so that arm fails a remedy on the banana only if its mean inefficiency
exceeds 3.19 and 2.32 times the package's. Remedy a's pass compares
identical kernels in 13 of its 15 runs, the copies of the package's runs where the guard
kept the step.

Run: PYTHONPATH=src:notes/experiments python notes/experiments/i137_nongaussian.py
(needs nothing first; rerun until it prints the tables: each run starts no new record
after 5 minutes, and the next resumes from the store in notes/experiments/_rows/). From
nothing, the 47 records it computes took 1,167 s on one thread by the store's `seconds`,
which include their tuning; the 28 copies carry their source's 718 s, which this script
does not spend, and leave out the tuning that preceded each copy. The print pass retunes
remedy a at 15 runs; that returns the stored step only with a package whose arithmetic is
that of the records' commit, so with a later package that tunes other bits the records
are recomputed first.
"""

import sys
import time
import warnings

import numpy as np
from i137_common import JITTER, Budget, Store, guard_summary, header
from i137_common import METHODS as KERNELS

import telescopic
from telescopic import CoupledSampler

warnings.simplefilter("ignore")


class ExactBanana:
    def rvs(self, *, random_state):
        x0 = 1 + np.sqrt(0.5) * random_state.standard_normal()
        return np.array([x0, x0**2 + np.sqrt(0.05) * random_state.standard_normal()])


TARGETS = {
    "banana": (telescopic.examples.banana(), ExactBanana(), [1.0, 1.5], lambda x: x),
    "mixture": (
        telescopic.examples.mixture(),
        None,
        [0.0, 17.0],
        lambda x: np.column_stack([x[:, 0], x[:, 0] ** 2]),
    ),
}
N_STEPS = {"banana": (2, 5), "mixture": (2,)}
SEEDS, R, METHODS = range(5), 200, ("base", "a", "b", "c", "d")


def tuned(name, method, n_steps, seed):
    example, initial, _, _ = TARGETS[name]
    sampler = CoupledSampler(
        example.log_prob,
        initial or example.initial,
        vectorize=True,
        grad=example.grad,
        kernel=KERNELS[method](n_steps=n_steps),
        rng=seed,
    )
    sampler.tune()
    return sampler


def run(name, method, n_steps, seed, store):
    truth, h = TARGETS[name][2:]
    start = time.time()
    sampler = tuned(name, method, n_steps, seed)
    target = sampler.target
    tune = target.n_grad
    leapfrog = sampler.kernel.kernels[0]
    record = {
        "target": name,
        "method": method,
        "n_steps": n_steps,
        "seed": seed,
        "eps": leapfrog.step_size,
        "n": leapfrog.n_steps,
        "jitter": getattr(leapfrog, "jitter", 0.0),
        "eps_da": (
            getattr(leapfrog, "guard", None) or getattr(leapfrog, "contraction", None) or {}
        ).get("eps_da"),
        "tune_grads": tune,
    }
    same = None
    if method == "a" and record["eps"] == record["eps_da"]:
        same = f"{name} base n={n_steps} seed={seed}"
    elif method == "d" and record["jitter"] == JITTER:
        same = f"{name} c n={n_steps} seed={seed}"
    if same:
        return dict(store.records[same], method=method, same_as=same)
    record["same_as"] = None
    try:
        result = sampler.run(R)
    except ValueError as error:
        tau = sampler.pilot
        record.update(
            error=str(error)[:60],
            k=None,
            unmet=int(np.isinf(tau).sum()),
            median=float(np.median(tau)),
            seconds=round(time.time() - start, 1),
        )
        return record
    tau = result.pilot
    estimate = result.estimate(h)
    value, se = np.asarray(estimate.value), np.asarray(estimate.standard_error)
    record.update(
        error=None,
        k=result.k,
        unmet=int(np.isinf(tau).sum()),
        median=float(np.median(tau)),
        q99=float(np.quantile(tau, 0.99)),
        pilot_grads=target.n_grad - tune - result.n_grad,
        replicate_grads=result.n_grad,
        n_met=result.n_replicates,
        value=value.tolist(),
        se=se.tolist(),
        z=((value - truth) / se).tolist(),
        inefficiency=(se**2 * result.n_grad).tolist(),
        seconds=round(time.time() - start, 1),
    )
    return record


if __name__ == "__main__":
    store = Store("i137_nongaussian.jsonl")
    budget = Budget(300)
    for name in TARGETS:
        for n_steps in N_STEPS[name]:
            for seed in SEEDS:
                for method in METHODS:
                    key = f"{name} {method} n={n_steps} seed={seed}"
                    if key in store:
                        continue
                    if not budget.left():
                        print("incomplete: rerun to resume")
                        sys.exit(0)
                    store.add(key, run(name, method, n_steps, seed, store))

    print(header(store))
    rows = list(store.records.values())
    # The guard's record of remedy a, which the stores lack, from a retuning that returns
    # the stored step.
    guards = {}
    for r in rows:
        if r["method"] == "a":
            leapfrog = tuned(r["target"], "a", r["n_steps"], r["seed"]).kernel.kernels[0]
            assert leapfrog.step_size == r["eps"]
            guards[r["key"]] = leapfrog.guard

    def pick(name, method, n_steps):
        return sorted(
            (
                r
                for r in rows
                if r["target"] == name and r["method"] == method and r["n_steps"] == n_steps
            ),
            key=lambda r: r["seed"],
        )

    print(
        "\nRaw table: target, method, n_steps, seed, eps, n, jitter, k, pilot unmet, pilot "
        "median and 99% quantile, gradients (tuning, pilot, replicates), replicates met, "
        "estimates +- SE (z), inefficiency of each h, copy; for a, whether a step of the guard's "
        "grid met its rule and the guard's predicted worst factor at the returned step"
    )
    for name in TARGETS:
        for n_steps in N_STEPS[name]:
            for method in METHODS:
                for r in pick(name, method, n_steps):
                    if r["error"]:
                        print(
                            f"{name} {method} n_steps={n_steps} seed={r['seed']} "
                            f"eps={r['eps']:.4f} unmet={r['unmet']} ERROR {r['error']}"
                        )
                        continue
                    est = ", ".join(
                        f"{v:.4f} +- {s:.4f} ({z:+.2f})"
                        for v, s, z in zip(r["value"], r["se"], r["z"], strict=True)
                    )
                    print(
                        f"{name} {method:4} n_steps={n_steps} seed={r['seed']} eps={r['eps']:.4f} "
                        f"n={r['n']} j={r['jitter']:.3f} k={r['k']} unmet={r['unmet']} "
                        f"median={r['median']:.1f} q99={r['q99']:.1f} grads={r['tune_grads']},"
                        f"{r['pilot_grads']},{r['replicate_grads']} R={r['n_met']} [{est}] "
                        f"ineff={[round(x, 1) for x in r['inefficiency']]} "
                        f"same_as={r['same_as']}"
                        + (
                            f" passed={guards[r['key']]['passed']} "
                            f"worst={guards[r['key']]['worst']:.6f}"
                            if method == "a"
                            else ""
                        )
                    )

    print(
        "\nSummary over seeds 0-4: k failures; per-seed median meeting time, mean (sd) "
        "[values]; inefficiency of the first h, mean (sd); pooled z of each h "
        "(sum z / sqrt 5); verdict against the package"
    )
    for name in TARGETS:
        for n_steps in N_STEPS[name]:
            ref = pick(name, "base", n_steps)
            ref_med = np.array([r["median"] for r in ref])
            ref_eff = np.array([r["inefficiency"][0] for r in ref if not r["error"]])
            for method in METHODS:
                rs = pick(name, method, n_steps)
                ok = [r for r in rs if not r["error"]]
                med = np.array([r["median"] for r in rs])
                eff = np.array([r["inefficiency"][0] for r in ok])
                z = np.sum([r["z"] for r in ok], axis=0) / np.sqrt(max(len(ok), 1))
                verdict = ""
                if method != "base":
                    worse_med = med.mean() - ref_med.mean() > ref_med.std(ddof=1)
                    worse_eff = len(eff) < len(rs) or eff.mean() - ref_eff.mean() > ref_eff.std(
                        ddof=1
                    )
                    verdict = "FAILS" if worse_med or worse_eff else "passes"
                print(
                    f"{name:7} n_steps={n_steps} {method:4}: k fails {len(rs) - len(ok)}/"
                    f"{len(rs)}, median tau {med.mean():.1f} ({med.std(ddof=1):.1f}) "
                    f"{med.round(1).tolist()}, ineff {eff.mean():.1f} ({eff.std(ddof=1):.1f}) "
                    f"{eff.round(1).tolist()}, pooled z {np.round(z, 2).tolist()} {verdict}"
                )
            print()

    print(
        "Remedy a's guard, seeds 0-4: seeds where it kept the step, lowered it, raised it, or "
        "fell back (no step of its grid met the rule); its runs that are copies of the "
        "package's"
    )
    for name in TARGETS:
        for n_steps in N_STEPS[name]:
            rs = pick(name, "a", n_steps)
            summary = guard_summary([dict(r, extra=guards[r["key"]]) for r in rs])
            copies = sum(r["same_as"] is not None for r in rs)
            print(f"{name:7} n_steps={n_steps}: {summary}; copies {copies} of {len(rs)}")

    print(
        "\nThe test's thresholds, the package's standard deviation over seeds, relative to "
        "the package's mean: median meeting time, inefficiency of the first h"
    )
    for name in TARGETS:
        for n_steps in N_STEPS[name]:
            ref = pick(name, "base", n_steps)
            med = np.array([r["median"] for r in ref])
            eff = np.array([r["inefficiency"][0] for r in ref])
            print(
                f"{name:7} n_steps={n_steps}: {med.std(ddof=1) / med.mean():.2f}, "
                f"{eff.std(ddof=1) / eff.mean():.2f}"
            )

"""The checks of experiment x123 on the registered families: groups A, B, C and the formula checks.

Question: how often do (N) and (H) warn on each family of families.py, and how often
would (N) warn with a relative floor of 1e-6, 1e-4 or 1e-2 on the norm |g o v|?

Design: family f of group G at the seeds families.seeds(G, f) (PREREGISTRATION.md,
section 6); one call of gradcheck.check per seed, from whose lines the decisions of
(N), of (H) and of the relative floors are read (Verdict.warns), so that every column
comes from the same checks. The check of seed s draws its points from the stream of
root seed s, as a sampler built with rng=s would. An error raised by a check is
counted and its last message printed: a family with an error is a failure of the
experiment, not of a design.

Columns: the seeds; the warnings of (N) and (H); the warnings of (N) with the relative
floors; the median and largest ratio of gap to tolerance of (N) and the largest of (H);
unusable lines and skipped points summed over the seeds; the largest evaluation counts;
the errors; the time.

Run: python notes/experiments/x123/run_checks.py A|B|C|F [first] [stop] [--smoke]

Result: see run_checks_*.out.
"""

import sys
import time
import warnings

import families as F
import gradcheck as G
import numpy as np
from runner import header, smoke

RELATIVE = (1e-6, 1e-4, 1e-2)


def group(name):
    if name == "A":
        return {k: (case, vec) for k, (case, vec) in F.GROUP_A.items()}
    if name == "B":
        return {k: (case, True) for k, case in F.GROUP_B.items()}
    if name == "C":
        return {k: (wrong, True) for k, (right, wrong) in F.GROUP_C.items()}
    return {k: (wrong, True) for k, (right, wrong) in F.FORMULA.items()}


def main():
    header()
    args = [a for a in sys.argv[1:] if a != "--smoke"]
    which = args[0]
    first, stop = (int(args[1]), int(args[2])) if len(args) > 2 else (0, 10**6)
    floors = " ".join(f"r={r:g}" for r in RELATIVE)
    print(
        f"group {which}; P = {G.POINTS}; columns: seeds | N | H | (N) with {floors} | ratio N med/max, H max | "
        "unusable, skipped | evaluations | errors | time"
    )
    totals = np.zeros(2 + len(RELATIVE), dtype=int)
    checks = 0
    for f, (name, (case, vectorize)) in enumerate(group(which).items()):
        if not first <= f < stop:
            continue
        seeds = F.seeds(which, f, smoke())
        counts = np.zeros(2 + len(RELATIVE), dtype=int)
        ratios_n, ratios_h, unusable, skipped, evaluations, errors, last = (
            [],
            [],
            0,
            0,
            (0, 0),
            0,
            "",
        )
        start = time.perf_counter()
        for seed in seeds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    v = G.check(*F.resolve(case, seed), seed, vectorize=vectorize)
                except G.GradientCheckError as error:  # the target raised inside the check
                    errors += 1
                    last = repr(error)[:200]
                    continue
            counts += [
                v.warns(G.DESIGNS["N"]),
                v.warns(G.DESIGNS["H"]),
                *(v.warns(relative=r) for r in RELATIVE),
            ]
            ratios_n.append(v.ratio(G.DESIGNS["N"]))
            ratios_h.append(v.ratio(G.DESIGNS["H"]))
            unusable += v.lines_unusable
            skipped += v.points_skipped
            evaluations = max(evaluations, (v.n_log_prob, v.n_grad))
        totals += counts
        checks += len(seeds) - errors
        n = len(seeds)
        ratio = (
            f"{np.median(ratios_n):8.2e}/{np.max(ratios_n):8.2e} {np.max(ratios_h):8.2e}"
            if ratios_n
            else "-"
        )
        print(
            f"[{f:>3}] {name:<72} {seeds[0]}..{seeds[-1]} | N {counts[0]:>4}/{n} | H {counts[1]:>4}/{n} | "
            + " ".join(f"{c:>4}" for c in counts[2:])
            + f" | {ratio} | {unusable} {skipped} | {evaluations[0]}+{evaluations[1]} | {errors}{' ' + last if errors else ''} | "
            f"{time.perf_counter() - start:.1f}s",
            flush=True,
        )
    print(
        f"total over {checks} checks: N {totals[0]}, H {totals[1]}, "
        + ", ".join(f"r={r:g} {t}" for r, t in zip(RELATIVE, totals[2:], strict=True))
    )


if __name__ == "__main__":
    main()

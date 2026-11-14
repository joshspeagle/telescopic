"""Does gradcheck.py compute what the prototype of the design round computes, line for line?

Question: gradcheck.py rewrites the prototype, ``prototype.py``, to hold one line at a time. On
the same seeds, are its usable lines, their gaps and tolerances (bit for bit), the
decisions of (N) and (H), the unusable lines, the skipped points and the evaluation
counts the same as the prototype's?

Design: every family of every group (families.py), 10 seeds each from the seen range
220000 to 229999 (family k of the concatenated list at 220000 + 10 k + (0..9)); the
prototype is called twice, as (N) and as (H), and gradcheck.check once. Two further
probes that the families do not cover: a log_prob whose dtype narrows to float32 only
after the first point, where gradcheck.py sweeps again (the decisions must agree, the
counts may not), and a log_prob that raises inside the check.

The prototype is ``prototype.py`` in this folder (its sha256 is printed).
Run: python notes/experiments/x123/equivalence.py

Result: see equivalence.out.
"""

import hashlib
import warnings
from pathlib import Path

import families as F
import gradcheck as G
import numpy as np
import prototype
from runner import header


def lines_of_prototype(verdict):
    return [(r["p"], float(r["gap"]), float(r["tol"])) for r in verdict.detail]


def lines_of_rewrite(verdict):
    return [(line.point, line.gap, line.tolerance) for line in verdict.lines]


def compare(case, seed, vectorize):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n, h = (
            prototype.N(*case, seed, vectorize=vectorize),
            prototype.H(*case, seed, vectorize=vectorize),
        )
        v = G.check(*case, seed, vectorize=vectorize)
    same_lines = lines_of_prototype(n) == lines_of_rewrite(v) == lines_of_prototype(h)
    same_decisions = n.warned == v.warns(G.DESIGNS["N"]) and h.warned == v.warns(G.DESIGNS["H"])
    same_counts = (n.lines_unusable, n.points_skipped, n.n_log_prob, n.n_grad) == (
        v.lines_unusable,
        v.points_skipped,
        v.n_log_prob,
        v.n_grad,
    )
    return same_lines, same_decisions, same_counts, n.warned, h.warned, len(v.lines)


def cases():
    for name, (case, vectorize) in F.GROUP_A.items():
        yield "A", name, case, vectorize
    for name, case in F.GROUP_B.items():
        yield "B", name, case, True
    for name, (_, wrong) in F.GROUP_C.items():
        yield "C", name, wrong, True
    for name, (_, wrong) in F.FORMULA.items():
        yield "F", name, wrong, True


def main():
    header()
    print(
        f"prototype.py sha256 {hashlib.sha256(Path(prototype.__file__).read_bytes()).hexdigest()}"
    )
    print(
        "family k at seeds 220000 + 10 k + (0..9); columns: lines, gaps and tolerances identical; "
        "decisions identical; counts identical; (N) and (H) warnings of the prototype; usable lines"
    )
    totals = np.zeros(3, dtype=int)
    n_checks = 0
    for k, (group, name, case, vectorize) in enumerate(cases()):
        rows = [
            compare(F.resolve(case, seed), seed, vectorize)
            for seed in range(220000 + 10 * k, 220010 + 10 * k)
        ]
        r = np.array(rows, dtype=float)
        totals += r[:, :3].sum(axis=0).astype(int)
        n_checks += len(rows)
        print(
            f"[{k:>3}] {group} {name:<72} | {int(r[:, 0].sum()):>2} {int(r[:, 1].sum()):>2} {int(r[:, 2].sum()):>2} | "
            f"N {int(r[:, 3].sum()):>2} H {int(r[:, 4].sum()):>2} | lines {int(r[:, 5].sum())}",
            flush=True,
        )
    print(
        f"total of {n_checks} checks: lines identical {totals[0]}, decisions identical {totals[1]}, counts identical {totals[2]}"
    )

    # A dtype that narrows after the first point: float32 where x_0 > 0.
    def narrowing(x):
        out = -0.5 * np.sum(x * x, axis=-1)
        return out.astype(np.float32) if np.all(x[:, 0] > 0) else out

    case = (narrowing, lambda x: -x, F.Normal(np.zeros(3), np.ones(3)))
    same = [compare(case, seed, True) for seed in range(229000, 229050)]
    print(
        f"dtype that narrows after the first point, seeds 229000..229049: lines identical {sum(s[0] for s in same)}/50, "
        f"decisions identical {sum(s[1] for s in same)}/50, counts identical {sum(s[2] for s in same)}/50"
    )

    # An error inside the check.
    def failing(x):
        raise ZeroDivisionError("a log_prob that fails")

    try:
        G.check(failing, lambda x: -x, F.Normal(np.zeros(3), np.ones(3)), 229100, vectorize=True)
    except G.GradientCheckError as error:
        print(
            f"an error inside the check: {type(error).__name__}: {error} (cause: {type(error.__cause__).__name__})"
        )


if __name__ == "__main__":
    main()

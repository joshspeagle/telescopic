"""Does the package's gradient check compute what the registered design (N) computes?

Question: ``telescopic._gradient_check.check_gradient`` moves design (N) of
``gradcheck.py`` into the package, without the floor of design (H) and the relative
floors. On the same seeds, are its compared lines, their gaps and tolerances (bit for
bit, both divided by the step ``h`` as the package records them), its decisions, its
unusable lines and its skipped points those of ``gradcheck.noise_limited``?

Design: every family of every group (families.py), 10 seeds each from the unused range
230000 to 239999 (family k of the concatenated list at 230000 + 10 k + (0..9)); each
seed runs ``gradcheck.noise_limited`` once and ``check_gradient`` once, the latter with
its warnings recorded, so that a decision of the package is both its record (a ratio
above 1) and its warning. Two further probes: a log_prob whose dtype narrows to float32
only after the first draw, where both sweep again (seeds 239000 to 239049), and a
log_prob that raises inside the check (seed 239100).

Run: python notes/experiments/x123/package_equivalence.py

Result: see package_equivalence.out. On 1,710 checks of 171 families, and on the 50
checks of the narrowing dtype, the lines, gaps, tolerances, decisions and counts are
identical, and every warning of the package matches a warning of the registered code.
"""

import warnings

import families as F
import gradcheck as G
import numpy as np
from runner import header

from telescopic._gradient_check import check_gradient
from telescopic._random import root_seed
from telescopic.initial import as_initial
from telescopic.warnings import InputWarning


def package(case, seed, vectorize):
    log_prob, grad, initial = case
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = check_gradient(
            log_prob,
            grad,
            as_initial(initial),
            root_seed(seed),
            args=(),
            kwargs=None,
            vectorize=vectorize,
        )
    disagree = [w for w in caught if issubclass(w.category, InputWarning)]
    return record, [str(w.message) for w in disagree]


def compare(case, seed, vectorize):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        registered = G.noise_limited(*case, seed, vectorize=vectorize)
    record, messages = package(case, seed, vectorize)
    lines_registered = [
        (line.point, line.gap / line.step, line.tolerance / line.step) for line in registered.lines
    ]
    lines_package = [
        (int(p), float(g), float(t))
        for p, g, t in zip(record.point, record.gap, record.tolerance, strict=True)
    ]
    warned = bool(len(record.gap)) and bool(record.ratio.max() > 1)
    warning = any(m.startswith("grad disagrees") for m in messages)
    nothing = any(m.startswith("the gradient check compared no line") for m in messages)
    same_lines = lines_registered == lines_package
    same_decision = registered.warned == warned == warning
    same_counts = (registered.lines_unusable, registered.points_skipped) == (
        record.n_unusable,
        record.n_skipped,
    ) and nothing == (not registered.lines)
    return same_lines, same_decision, same_counts, registered.warned, len(record.gap)


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
        "family k at seeds 230000 + 10 k + (0..9); columns: lines, gaps and tolerances "
        "identical; decisions identical (registered, package record, package warning); "
        "unusable lines and skipped points identical; warnings of the registered (N); "
        "compared lines"
    )
    totals = np.zeros(3, dtype=int)
    n_checks = n_warned = 0
    for k, (group, name, case, vectorize) in enumerate(cases()):
        rows = [
            compare(F.resolve(case, seed), seed, vectorize)
            for seed in range(230000 + 10 * k, 230010 + 10 * k)
        ]
        r = np.array(rows, dtype=float)
        totals += r[:, :3].sum(axis=0).astype(int)
        n_checks += len(rows)
        n_warned += int(r[:, 3].sum())
        print(
            f"[{k:>3}] {group} {name:<72} | {int(r[:, 0].sum()):>2} {int(r[:, 1].sum()):>2} "
            f"{int(r[:, 2].sum()):>2} | N {int(r[:, 3].sum()):>2} | lines {int(r[:, 4].sum())}",
            flush=True,
        )
    print(
        f"total of {n_checks} checks, {n_warned} warned by (N): lines identical {totals[0]}, "
        f"decisions identical {totals[1]}, counts identical {totals[2]}"
    )

    # A dtype that narrows after the first draw: float32 where x_0 > 0.
    def narrowing(x):
        out = -0.5 * np.sum(x * x, axis=-1)
        return out.astype(np.float32) if np.all(x[:, 0] > 0) else out

    case = (narrowing, lambda x: -x, F.Normal(np.zeros(3), np.ones(3)))
    same = [compare(case, seed, True) for seed in range(239000, 239050)]
    print(
        "dtype that narrows after the first draw, seeds 239000..239049: lines identical "
        f"{sum(s[0] for s in same)}/50, decisions identical {sum(s[1] for s in same)}/50, "
        f"counts identical {sum(s[2] for s in same)}/50"
    )

    # An error inside the check.
    def failing(x):
        raise ZeroDivisionError("a log_prob that fails")

    try:
        package((failing, lambda x: -x, F.Normal(np.zeros(3), np.ones(3))), 239100, True)
    except ZeroDivisionError as error:
        print(
            f"an error inside the check: {type(error).__name__}: {error} (notes: {error.__notes__})"
        )


if __name__ == "__main__":
    main()

"""The rates that the guide quotes from the review of x123, measured with the package.

Question: the passage "Checking the gradient" of docs/guide/targets.md quotes warning
rates of the check that the adversarial review (review/REVIEW.md) measured with the
registered code (gradcheck.check). Does the package's own check,
``telescopic._gradient_check.check_gradient``, give the same rates on the same families
and seeds?

Design: each family and seed range is that of the review, taken from its scripts
unchanged, by importing review/cases5.py, fa1.py, fa2.py (part "solver"), fa4.py and
fa5.py, with the function ``probe.run``, which each of the last four calls for its rows
when imported, replaced by one that does nothing. One call of check_gradient per seed (root_seed(seed), as the review's
probe.run passes the seed to gradcheck.check), with its warnings recorded. A line per
statement of the guide: the checks in which the package warned that grad disagrees, of
the checks run, and, for the start on a boundary, the checks that compared no line.

Run: python notes/experiments/x123/guide_figures.py

Result: see guide_figures.out. Every count is the review's count of warnings of (N) on
the same family and seeds (fa1.out, fa2_solver.out, fa4.out, fa5.out, miss_detect.out):
48 and 200 of 200 with 1e5 and 1e6 periods, 143 of 200 with the table; 195 to 200 of 200
on 7 of the 10 targets with a gradient computed in float32, 30 of 200 on pumps and none
on the other two; 194 to 200 of 200 with forward differences on three targets and none
of 50 on the logistic regression; 100 of 100 with the ODE solver; none from a start at
the mode, with an error away from the start or behind -1e10, and 196, 37 and 0 with a
float32 log_prob in 200, 1,000 and 10,000 dimensions; and no line compared in 200 of
200 checks from a start on a boundary. It takes about 20 seconds.
"""

import contextlib
import importlib
import io
import sys
import types
import warnings
from pathlib import Path

import families as F
from runner import header

from telescopic._gradient_check import check_gradient
from telescopic._random import root_seed
from telescopic.initial import as_initial

REVIEW = Path(__file__).resolve().parent / "review"


def review(name, part=None):
    # review/<name>.py, imported with part as its command-line argument and its printed
    # headings discarded; its probes, which call probe.run when it is imported, do
    # nothing (main replaces run).
    argv = sys.argv
    sys.argv = [name] if part is None else [name, part]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return importlib.import_module(name)
    finally:
        sys.argv = argv


def counts(case, seeds, vectorize=True):
    # Checks that warned that grad disagrees, checks that compared no line, checks run.
    disagree = nothing = n = 0
    for seed in seeds:
        log_prob, grad, initial = case(seed) if callable(case) else case
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_gradient(
                log_prob,
                grad,
                as_initial(initial),
                root_seed(seed),
                args=(),
                kwargs=None,
                vectorize=vectorize,
            )
        messages = [str(w.message) for w in caught]
        disagree += any(m.startswith("grad disagrees") for m in messages)
        nothing += any(m.startswith("the gradient check compared no line") for m in messages)
        n += 1
    return disagree, nothing, n


def line(statement, case, seeds, vectorize=True):
    disagree, _, n = counts(case, seeds, vectorize)
    print(f"{statement}: {disagree} of {n}", flush=True)
    return disagree, n


def main():
    header()
    sys.path.insert(0, str(REVIEW))
    probe = types.ModuleType("probe")
    probe.run = lambda *args, **kwargs: None
    sys.modules["probe"] = probe
    cases5, fa1, fa4, fa5 = (review(name) for name in ("cases5", "fa1", "fa4", "fa5"))
    solver = review("fa2", "solver")
    seeds, seeds100 = fa1.S, solver.S100

    print("== a correct gradient can warn (fa1.py)")
    line("periodic term, 1e5 periods across the start, d = 1", F.fringes(1e5), seeds)
    line("periodic term, 1e6 periods across the start, d = 1", F.fringes(1e6), seeds)
    line("piecewise-linear table of spacing 1e-5, d = 5, start sd 2", fa1.table(5, 1e-5), seeds)

    print("== gradients approximate by design")
    rows = []
    for name, case in (
        ("N(0,I) d=5", F.gaussian()),
        ("N(0,I) d=1000 start sd 2", F.gaussian_start(1000, 2.0)),
        ("N(1e3,I) d=5", F.gaussian(mean=1e3)),
        ("N(1e6,I) d=5", F.gaussian(mean=1e6)),
        ("Student t3 d=5 start sd 2", F.student(wide=2.0)),
        ("logistic n=1e4 d=5", F.logistic()),
        ("rotated cond 1e4 d=10", F.conditioned(1e4)),
        ("funnel", F.example("funnel")),
        ("banana", F.example("banana")),
        ("pumps", F.example("pumps")),
    ):
        targets = fa4.S if "logistic" not in name else range(500_000, 500_100)
        disagree, n = line(
            f"grad computed from float32 x, returned as float64 (fa4.py), {name}",
            fa4.disguise(case),
            targets,
        )
        rows.append(disagree >= 195 and n == 200)
    print(f"targets with 195 to 200 warnings of 200: {sum(rows)} of {len(rows)}")
    for name, case in (
        ("N(0,I) d=5", F.gaussian()),
        ("Student t3 d=5 start sd 2", F.student(wide=2.0)),
        ("banana", F.example("banana")),
        ("logistic n=1e4 d=5", F.logistic()),
    ):
        line(
            f"forward differences as grad (fa5.py), {name}",
            fa5.fd(case, "forward"),
            fa5.S50 if "logistic" in name else fa5.S,
            vectorize=False,
        )
    line(
        "ODE log-density by adaptive RK45 at rtol 1e-6, gradient of the exact solution "
        "(fa2.py solver)",
        (solver.lp_solver(1e-6), solver.g_exact, solver.init),
        seeds100,
    )

    print("== silence is weak evidence (cases5.py, wrong gradients)")
    for name in (
        "N(0,I) d=5, point start at the mode: wrong sign",
        "N(0,I) d=5, start sd 0.1: gradient right for |x| < 1, wrong sign outside",
        "N(0,I) - 1e+10 d=5, start sd 2: sign of coordinate 0",
        "float32 throughout, N(0,I) d=200, start sd 2: sign of coordinate 0",
        "float32 throughout, N(0,I) d=1000, start sd 2: sign of coordinate 0",
        "float32 throughout, N(0,I) d=10000, start sd 2: sign of coordinate 0",
    ):
        line(name, cases5.M[name][1], seeds if "d=10000" not in name else seeds100)

    name = "half-normal d=3 (x >= 0), start with coordinate 0 on the boundary: wrong sign"
    disagree, nothing, n = counts(cases5.M[name][1], seeds)
    print(f"{name}: warned that grad disagrees {disagree}, compared no line {nothing}, of {n}")


if __name__ == "__main__":
    main()

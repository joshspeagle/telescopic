"""Exploratory: (N) and (H) on the wrong gradients that the first experiment's design missed (seeds 210000..213599).

Question: do the candidates detect the review's families (cases.REVIEW_WRONG: starts tight in the wrong
coordinate, starts unmatched to the target's scales, rotated targets, wide starts with an offset)
and stay silent on the same targets with the correct gradient?
Design: family f at seeds 210000 + 200 f + (0..199); both designs at P = 8, and (N) with the first
experiment's directions (spread times signs) for comparison.
Result: see det_run.out and DESIGN.md section 5.
"""

import warnings

import cases
import numpy as np
import prototype

from telescopic.initial import Normal

CASES = dict(cases.REVIEW_WRONG)
N5 = cases.gaussian()
for sd in (1e-3, 1e-4):
    case = cases.with_initial(N5, Normal(np.zeros(5), np.array([sd, 2, 2, 2, 2.0]) ** 2))
    CASES[f"N(0,I) d=5, init sd ({sd:g},2,2,2,2): sign of coordinate 0"] = (
        case,
        cases.modify(case, cases.flip(0)),
    )
ball = cases.with_initial(N5, Normal(np.zeros(5), np.full(5, 1e-3**2)))
CASES["N(0,I) d=5, init sd 1e-3 in every coordinate: sign of coordinate 0"] = (
    ball,
    cases.modify(ball, cases.flip(0)),
)

if __name__ == "__main__":
    n = 200
    print(
        f"family f at seeds 210000 + {n} f + (0..{n - 1}); P = {prototype.POINTS}; warnings on the wrong gradient (on the correct one)"
    )
    for f, (name, (right, wrong)) in enumerate(CASES.items()):
        out = []
        for key, design in (
            ("N", prototype.N),
            ("H", prototype.H),
            ("N,signs", lambda *a: prototype.check(*a, direction="signs")),
        ):
            w = fa = 0
            for seed in range(210000 + n * f, 210000 + n * (f + 1)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w += design(*wrong, seed).warned
                    fa += design(*right, seed).warned
            out.append(f"{key} {w:>3}/{n} ({fa})")
        print(f"[{f:>2}] {name:<76} | " + " | ".join(out), flush=True)

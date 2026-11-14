"""Exploratory: the trade between silence on approximate gradients and detection (seeds 210000..213599, 216000..217299).

Question: if the tolerance of (N) had a relative floor r |g o v| (the root sum of squares of the
per-coordinate predicted changes g_i v_i, so that a gradient with independent relative errors of
size r in every component stays silent), which wrong gradients would be missed, and which
approximate ones would stay silent?  (N) is r = 0. (H) is shown beside.
Design: (1) the wrong gradients of det_run.py on their seeds (210000 + 200 f + (0..199)); (2) gradients
that are exact but not the derivative of log_prob as computed, or wrong and harmless: family j at
seeds 216000 + 100 j + (0..99).
Result: see diag_r.out.
"""

import warnings

import cases
import numpy as np
import prototype
from det_run import CASES as WRONG

RS = (0.0, 1e-6, 1e-4, 1e-2)


def rates(case, seeds, vec=True):
    out = np.zeros(len(RS) + 1, dtype=int)
    for seed in seeds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = prototype.N(*case, seed, vectorize=vec)
        for j, r in enumerate(RS):
            out[j] += any(row["gap"] > max(row["tol"], r * row["gv2"]) for row in v.detail)
        out[-1] += any(row["gap"] > max(row["tol"], row["h"] * prototype.PHI) for row in v.detail)
    return out


N5, N5w = cases.gaussian(), cases.gaussian(wide=200.0)
lg = cases.logistic()
n1000 = cases.gaussian(d=1000)


def g32(case):
    return case[0], (lambda x: np.asarray(case[1](x), dtype=np.float32)), case[2]


N1e3 = cases.gaussian(mean=1e3)
OTHER = {
    "Jacobi solver rtol 1e-3 d=5 (exact gradient)": cases.loose_solver(1e-3),
    "Jacobi solver rtol 1e-5 d=5 (exact gradient)": cases.loose_solver(1e-5),
    "Jacobi solver rtol 1e-8 d=5 (exact gradient)": cases.loose_solver(1e-8),
    "N(0,I) d=5: grad x 1.005": cases.modify(N5, lambda x, g: 1.005 * g),
    "N(0,I) d=5: grad + 0.02": cases.modify(N5, lambda x, g: g + 0.02),
    "logistic n=1e4 d=5: without the prior": cases.modify(lg, lambda x, g: g + x),
    "N(0,I) d=5, init 100x wide: grad x 1.005": cases.modify(N5w, lambda x, g: 1.005 * g),
    "N(0,I) d=1000: grad x 1.0001": cases.modify(n1000, lambda x, g: 1.0001 * g),
    "float32 grad, N(0,I) d=5: sign of coordinate 0": g32(cases.modify(N5, cases.flip(0))),
    "float32 grad, N(1e3,I) d=5: sign of coordinate 0": g32(cases.modify(N1e3, cases.flip(0))),
    "float32 grad, N(0,I) d=1000: sign of coordinate 0": g32(cases.modify(n1000, cases.flip(0))),
    "float32 grad, rotated cond 1e8 d=10 (correct formula)": cases.rotated32grad(1e8),
    "float32 grad, N(0,I) d=5: grad x 1.0001": g32(cases.modify(N5, lambda x, g: 1.0001 * g)),
}

if __name__ == "__main__":
    head = " ".join(f"r={r:<6g}" for r in RS) + " (H)"
    print(f"warnings of (N) with a relative floor r |g o v|, and of (H); P = {prototype.POINTS}")
    print(f"1. wrong gradients (200 seeds each){'':<45} | {head}")
    for f, (name, (_, wrong)) in enumerate(WRONG.items()):
        out = rates(wrong, range(210000 + 200 * f, 210000 + 200 * (f + 1)))
        print(f"[{f:>2}] {name:<76} | " + " ".join(f"{o:>8}" for o in out), flush=True)
    print(f"2. approximate, harmless or float32 gradients (100 seeds each){'':<18} | {head}")
    for j, (name, case) in enumerate(OTHER.items()):
        out = rates(case, range(216000 + 100 * j, 216000 + 100 * (j + 1)))
        print(f"[{j:>2}] {name:<76} | " + " ".join(f"{o:>8}" for o in out), flush=True)

"""Review of x123: the lines of the registered families of groups B and C and of the formula checks, on fresh seeds.

Question: what are the gap, the tolerance and its parts, |log_prob|, the norm |g o v|, the
difference quotient D and the gradient's prediction G on every line of every check of the
registered families that are not correct gradients? floors_analyze.py asks of these
numbers whether a floor of another form separates group B from the harmful families;
the registered outputs keep only counts.

Design: family f of group B on seeds 520000 + 200 f + (0..99), of group C on
530000 + 200 f + (0..99) (40 seeds for the rotated Gaussian at d = 2000), of the formula
checks on 540000 + 200 f + (0..99); one call of gradcheck.check per seed, as
run_checks.py makes it. D and G are not kept by a Verdict, so the script reads them from
the rows that gradcheck._sweep returns. The lines are written to the pickle named by the
first argument, a working file that is not part of the record; the printed column is the
number of checks in which (N) warns.

Run: PYTHONPATH=src:notes/experiments/x123 python notes/experiments/x123/review/floors_collect.py floors.pkl B C F > floors_collect.out

Result: see floors_collect.out, and floors_analyze.out for the analysis.
"""

import pickle
import sys
import time
import warnings

import families as F
import gradcheck as G
import numpy as np

N_SEEDS = 100
captured = {}  # the rows of the last sweep, which hold D and G
_orig = G._sweep


def _wrapped(*a, **k):
    out = _orig(*a, **k)
    captured["rows"] = out[0]
    return out


G._sweep = _wrapped
data = {}
groups = {
    "B": dict(F.GROUP_B),
    "C": {k: v[1] for k, v in F.GROUP_C.items()},
    "F": {k: v[1] for k, v in F.FORMULA.items()},
}
base = {"B": 520_000, "C": 530_000, "F": 540_000}
for gname in sys.argv[2:]:
    for f, (name, case) in enumerate(groups[gname].items()):
        n = 40 if "d=2000, matched" in name else N_SEEDS
        t0 = time.perf_counter()
        checks = []
        for seed in range(base[gname] + 200 * f, base[gname] + 200 * f + n):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = G.check(*F.resolve(case, seed), seed, vectorize=True)
            rows = captured["rows"]
            checks.append(
                np.array(
                    [
                        (
                            l.gap,
                            l.tolerance,
                            l.truncation,
                            l.rounding_f,
                            l.rounding_g,
                            l.largest,
                            l.norm,
                            l.step,
                            r["D"],
                            r["G"],
                        )
                        for l, r in zip(v.lines, rows, strict=True)
                    ],
                    dtype=float,
                ).reshape(-1, 10)
            )
        data[(gname, name)] = checks
        nw = sum(bool(np.any(c[:, 0] > c[:, 1])) for c in checks)
        print(
            f"{gname} [{f}] {name[:70]:<70} N {nw}/{n} ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
with open(sys.argv[1], "wb") as file:
    pickle.dump(data, file)

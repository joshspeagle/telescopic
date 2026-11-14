# (a) G against the old guard's stored steps; (b) the tunings G moves off no resonance and P's stored
# meeting times there; (c) i137's J5/P cost ratios per cell with their standard errors and the chance
# that J5 passes M on every cell; (d) unmet counts per failing seed at cap 1e4 and tail (k) of P.
import json
import math
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
from p1_guard import R, mine


def L(n):
    with open(R + n) as f:
        return [json.loads(l) for l in f]


b = L("i137_baseline.jsonl")
rem = L("i137_remedies.jsonl")
un = L("i137_unequal.jsonl")
tune = {(x["d"], x["n_steps"], x["seed"]): x for x in b if x["kind"] == "tune"}
pil = {(x["d"], x["n_steps"], x["seed"]): x for x in b if x["kind"] == "pilot"}
a = {(x["d"], x["n_steps"], x["seed"]): x for x in rem if x["method"] == "a"}
c = {(x["d"], x["n_steps"], x["seed"]): x for x in rem if x["method"] == "c"}
print("(a) G vs old guard, stored i137 tunings at b0b8f71, seeds 0-29")
for cell in [(2, 2), (10, 3), (10, 10), (2, 3), (2, 5), (2, 10), (10, 5)]:
    same = kept = 0
    for s in range(30):
        t = pil[(*cell, s)]
        e, act = mine(t["eps"], cell[1], [1.0])
        old = a[(*cell, s)]["eps"]
        same += abs(e - old) < 1e-9 * old
        kept += act == "kept"
    print(
        " ",
        cell,
        f"G equals old guard at {same:d} of 30; G keeps {kept:d}; old guard moved "
        f"{sum(abs(a[(*cell, s)]['eps'] - pil[(*cell, s)]['eps']) > 1e-12 for s in range(30)):d}",
    )
print(
    "(b) tunings (seeds 0-99, stored) that G moves in cells where P never fails; P's pilot at those seeds if < 30"
)
for cell in [(2, 3), (2, 5), (2, 10), (10, 5)]:
    for s in range(100):
        t = tune[(*cell, s)]
        e, act = mine(t["eps"], cell[1], [1.0])
        if act != "kept":
            p = pil.get((*cell, s))
            print(
                " ",
                cell,
                "seed",
                s,
                "eps {:.4f} -> {:.4f} exact factor {:.4f}".format(t["eps"], e, t["factor_max"]),
                "P median {} k {}".format(p["median"], p["k"]) if p else "",
            )
    med = [pil[(*cell, s)]["median"] for s in range(30)]
    print(
        f"   cell median of medians {np.median(med):.1f}, "
        f"k range {min(pil[(*cell, s)]['k'] for s in range(30)):d}-"
        f"{max(pil[(*cell, s)]['k'] for s in range(30)):d}"
    )
print(
    "(c) J5 (remedy c) over P: ratio of geometric means of (seed median tau x gradients per iteration), 30 seeds, cap 1e4"
)


def m(d, cell):
    out = []
    for s in range(30):
        x = d[(*cell, s)]
        # gradients per pair-iteration ~ pilot gradients per pair / mean tau is not stored; use 2 chains x n x 0.95
        out.append(math.log(x["median"] * 2 * x["n"] * 0.95))
    return np.array(out)


tot = 1.0
for cell in [(2, None), (2, 1), (2, 3), (2, 5), (2, 10), (10, None), (10, 1), (10, 2), (10, 5)]:
    A, P = m(c, cell), m(pil, cell)
    d = A.mean() - P.mean()
    se = math.sqrt(A.var(ddof=1) / 30 + P.var(ddof=1) / 30)
    pf = 1 - stats.norm.cdf((math.log(1.2) - d) / se)
    tot *= 1 - pf
    print(
        f"  d={cell[0]:d} n={cell[1]} rho_M {math.exp(d):.3f} se(log) {se:.3f} "
        f"sd P {P.std(ddof=1):.3f} J5 {A.std(ddof=1):.3f}  "
        f"P(fail M | true ratio = this estimate, 30 new seeds) {pf:.3f}"
    )
print(
    f"  P(J5 passes M on all 9 cells of part A if the true ratios are these estimates) = {tot:.3f}"
)
print("(d) P at cap 1e4: unmet per failing seed (k not chosen), resonant cells")
for cell in [(2, 2), (10, 3), (10, 10)]:
    u = sorted(pil[(*cell, s)]["unmet"] for s in range(30) if pil[(*cell, s)]["error"])
    ks = [pil[(*cell, s)]["k"] for s in range(30) if not pil[(*cell, s)]["error"]]
    print(
        " ",
        cell,
        "failing seeds",
        len(u),
        "unmet:",
        u,
        f"| k of passing seeds: median {np.median(ks) if ks else None} max {max(ks) if ks else None}",
    )
print(
    "   unequal, P default n_steps (cap 1e4):",
    [
        (x["seed"], x["unmet"], x["k"])
        for x in un
        if x["method"] == "base" and x["n_steps"] is None
    ],
)
print(
    "   unequal, guard a default n_steps (cap 1e4):",
    [(x["seed"], x["unmet"], x["k"]) for x in un if x["method"] == "a" and x["n_steps"] is None],
)

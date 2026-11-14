# Spreads between seeds in i137's stores, and the size and power of the registered cost criteria.
import collections
import json

import numpy as np
from scipy import stats

R = "notes/experiments/_rows/"  # i137's stores, relative to the root of a checkout that holds them


def L(n):
    with open(R + n) as f:
        return [json.loads(l) for l in f]


ng = L("i137_nongaussian.jsonl")
b = L("i137_baseline.jsonl")
rem = L("i137_remedies.jsonl")


def sdlog(v):
    v = np.log(np.asarray(v, float))
    return v.std(ddof=1)


print(
    "== per-seed median meeting time and inefficiency (h = x0), log-scale sd between seeds; i137_nongaussian (5 seeds)"
)
cells = collections.defaultdict(dict)
for r in ng:
    if r["method"] in ("base", "c"):
        cells[(r["target"], r["n_steps"])].setdefault(r["method"], []).append(r)
for k, v in cells.items():
    for m, rs in v.items():
        med = [r["median"] for r in rs]
        ef = [r["inefficiency"][0] for r in rs]
        le = np.log(ef)
        mad = 1.4826 * np.median(np.abs(le - np.median(le)))
        print(
            k,
            m,
            f"median tau sd(log) {sdlog(med):.3f}  ineff sd(log) {sdlog(ef):.3f}  robust sd(log) {mad:.3f}",
        )
print(
    "== Gaussian table: per-seed median meeting time sd(log), 30 seeds (cap 1e4), P and remedy c, configs where every seed met"
)
for d in (2, 10):
    for n in (None, 1, 2, 3, 5, 10):
        for name, rows, meth in (("P", b, "base"), ("J5", rem, "c")):
            rs = [
                r
                for r in rows
                if r.get("d") == d
                and r["n_steps"] == n
                and r["method"] == meth
                and r.get("kind", "pilot") == "pilot"
                and r.get("cap", 1) != 0
            ]
            med = [r["median"] for r in rs if np.isfinite(r["median"])]
            if len(med) == len(rs) and rs:
                print(
                    f"d={d} n={n} {name}: n={len(rs)} sd(log median tau) {sdlog(med):.3f}; sd(log grads/pair) {sdlog([r['pilot_grads_per_pair'] for r in rs]):.3f}"
                )
# size and power of: fail if exp(mean log A - mean log P) > 1.2, S seeds each, normal approx
print(
    "== median-cost criterion: fail if geometric-mean ratio > 1.20; P(fail) at true ratios 1.0, 1.1, 1.2, 1.3, 1.4, 1.5"
)
for lab, sa, sp in [
    ("banana n=2 (P .10, J5 .08)", 0.10, 0.08),
    ("banana n=5 (P .28, J5 .11)", 0.28, 0.11),
    ("mixture n=2 (P .115, J5 .46)", 0.115, 0.46),
    ("gaussian (0.06,0.06)", 0.06, 0.06),
    ("worst-case .30/.30", 0.30, 0.30),
]:
    for S in (10, 20):
        se = np.sqrt((sa**2 + sp**2) / S)
        print(
            lab,
            f"S={S:d} se {se:.3f}",
            [
                round(1 - stats.norm.cdf((np.log(1.2) - np.log(r)) / se), 3)
                for r in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
            ],
        )

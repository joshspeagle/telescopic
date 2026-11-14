import collections
import json
import math

import numpy as np

R = "notes/experiments/_rows/"  # i137's stores, relative to the root of a checkout that holds them


def load(n):
    with open(R + n) as f:
        return [json.loads(l) for l in f]


def theta(h):
    return 2 * np.arcsin(np.minimum(h, 2) / 2)


def dist(eps, n, w=1.0):
    h = eps * w
    if h >= 2:
        return 0.0
    phi = n * theta(h)
    j = max(1, round(phi / math.pi))
    return (
        abs(phi - j * math.pi) if phi >= math.pi / 2 else (math.pi - phi)
    )  # distance to nearest j>=1 resonance


b = load("i137_baseline.jsonl")
g = collections.defaultdict(list)
for r in b:
    if r["kind"] in ("tune", "pilot"):
        g[(r["d"], r["n_steps"])].append(r)
print("config: predicted phase distance (w=1) to nearest j>=1 resonance; exact factor")
for k in sorted(g, key=str):
    rs = g[k]
    ds = np.array([dist(r["eps"], r["n"]) for r in rs])
    f = np.array([r["factor_max"] for r in rs])
    on = f > 0.999
    print(
        k,
        f"n={len(rs):d} tunings",
        f"dist min {ds.min():.3f} med {np.median(ds):.3f} max {ds.max():.3f}",
        f"| on-res(>0.999) {on.sum():d} dist max {ds[on].max() if on.any() else float('nan'):.3f}",
        f"| off-res dist min {ds[~on].min() if (~on).any() else float('nan'):.3f}",
        "| pilot failing dist max {}".format(
            max([dist(r["eps"], r["n"]) for r in rs if r.get("error")], default=None)
        ),
    )
ng = load("i137_nongaussian.jsonl")
for r in ng:
    if r["method"] == "base":
        print(
            r["target"],
            r["n_steps"],
            r["seed"],
            f"eps {r['eps']:.4f} n {r['n']:d} dist {dist(r['eps'], r['n']):.3f}",
        )

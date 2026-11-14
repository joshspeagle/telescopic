# Independent implementation of the guard from the text of section 2 only, compared with
# the author's guard.py on every stored/probe tuning; and checks of section 10's counts.
import collections
import json
import math
import sys

import numpy as np

# the registration's folder
X = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/"
sys.path.insert(0, X)
import guard as A

R = "notes/experiments/_rows/"  # i137's stores, relative to the root of a checkout that holds them
DELTA = math.acos(0.9)


def load(path):
    with open(path) as f:
        return list(map(json.loads, f))


def nsteps(eps, given):
    return given if given is not None else math.ceil(1 / max(eps, 1e-3))


def r(eps, n, w):
    h = eps * w
    if h >= 2:
        return 0.0
    phi = n * 2 * math.asin(h / 2)
    return min(abs(phi - j * math.pi) for j in range(1, int(phi / math.pi) + 3))


def mine(eps0, given, wp):
    if min(r(eps0, nsteps(eps0, given), w) for w in wp) >= DELTA:
        return eps0, "kept"
    fs = np.geomspace(1 / 1.3, 1.3, 41)
    S = [k / 1000 for k in range(-2000, 101)]
    order = sorted(S, key=lambda s: (abs(s), s))  # smallest |s|, smaller step on ties
    for s in order:
        e = eps0 * math.exp(s)
        n = nsteps(e, given)
        if all(r(e, n, w * f) >= DELTA for w in wp for f in fs):
            return e, "moved"
    best = None
    for s in order:
        e = eps0 * math.exp(s)
        sc = min(r(e, nsteps(e, given), w) for w in wp)
        if best is None or sc > best[0]:
            best = (sc, e)
    return best[1], "fallback"


rows = [
    {
        "cell": ("g2", x["n_steps"]),
        "seed": x["seed"],
        "eps": x["eps"],
        "n": x["n"],
        "given": x["n_steps"],
        "w": [x["w_min"], x["w_max"]],
        "wp": [1.0],
    }
    for x in load(R + "i137_baseline.jsonl")
    if x["kind"] == "tune" and x["d"] == 2
]
rows += [
    {
        "cell": ("g10", x["n_steps"]),
        "seed": x["seed"],
        "eps": x["eps"],
        "n": x["n"],
        "given": x["n_steps"],
        "w": x["w"],
        "wp": [1.0],
    }
    for x in load(X + "tunings_d10.jsonl")
]
rows += [
    {
        "cell": ("uneq", x["n_steps"]),
        "seed": x["seed"],
        "eps": x["eps"],
        "n": x["n"],
        "given": x["n_steps"],
        "w": x["w"],
        "wp": x["w_pred"],
    }
    for x in load(X + "tunings_unequal.jsonl")
]
diff = 0
out = collections.defaultdict(lambda: collections.Counter())
for x in rows:
    e1, a1 = mine(x["eps"], x["given"], x["wp"])
    e2, a2 = A.guard(x["eps"], x["given"], np.array(x["wp"]), DELTA, 0.3)
    if a1 != a2 or abs(e1 - e2) > 1e-12 * e1:
        diff += 1
        print("DIFF", x["cell"], x["seed"], x["eps"], (e1, a1), (e2, a2))
    f0 = A.exact_factor(x["eps"], x["n"], x["w"]).max()
    f1 = A.exact_factor(e1, nsteps(e1, x["given"]), x["w"]).max()
    c = out[x["cell"]]
    c[a1] += 1
    for t in (0.999, 0.9998, 0.99984):
        c[f"pre>{t:g}"] += f0 > t
        c[f"post>{t:g}"] += f1 > t
    c["cost_x"] = max(c["cost_x"], round(nsteps(e1, x["given"]) / x["n"], 2))
    x.update(e1=e1, a1=a1, f0=float(f0), f1=float(f1))
print("tunings", len(rows), "differences between the two implementations:", diff)
for k in sorted(out, key=str):
    print(k, dict(out[k]))
print(
    "uneq default: n before -> after:",
    collections.Counter(
        (x["n"], nsteps(x["e1"], None)) for x in rows if x["cell"] == ("uneq", None)
    ),
)
print(
    "uneq default, tunings with post factor > pre factor:",
    sum(x["f1"] > x["f0"] for x in rows if x["cell"] == ("uneq", None)),
    "of 30",
)
for x in rows:
    if x["cell"] == ("uneq", None) and (x["f0"] > 0.9998 or x["f1"] > 0.9998):
        print(
            "  seed",
            x["seed"],
            "eps {:.4f} -> {:.4f} factor {:.6f} -> {:.6f}".format(
                x["eps"], x["e1"], x["f0"], x["f1"]
            ),
        )

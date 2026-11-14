import collections
import json
import sys

import numpy as np
from guard import *

R = "notes/experiments/_rows/"  # i137's stores, relative to the root of a checkout that holds them
rows = []
with open(R + "i137_baseline.jsonl") as f:
    for x in map(json.loads, f):
        if x["kind"] == "tune" and x["d"] == 2:
            rows.append(
                {
                    "d": 2,
                    "n_steps": x["n_steps"],
                    "seed": x["seed"],
                    "eps": x["eps"],
                    "n": x["n"],
                    "w": [x["w_min"], x["w_max"]],
                }
            )
with open(__import__("pathlib").Path(__file__).with_name("tunings_d10.jsonl")) as f:
    rows += [json.loads(l) for l in f]
for delta, band in [(float(a), float(b)) for a, b in (p.split(":") for p in sys.argv[1:])]:
    print(f"=== delta {delta} band {band}")
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["d"], r["n_steps"])].append(r)
    for k in sorted(g, key=str):
        c = collections.Counter()
        post = []
        pre = []
        for r in g[k]:
            f0 = exact_factor(r["eps"], r["n"], r["w"]).max()
            e, a = guard(r["eps"], r["n_steps"], 1.0, delta, band)
            f1 = exact_factor(e, n_of(e, r["n_steps"]), r["w"]).max()
            c[a] += 1
            pre.append(f0)
            post.append(f1)
            if a != "kept" and f0 < 0.99:
                c["moved_offres(<0.99)"] += 1
            if a == "kept" and f0 > 0.999:
                c["kept_onres(>0.999)"] += 1
        pre = np.array(pre)
        post = np.array(post)
        print(
            k,
            dict(c),
            f"exact>0.999 before {(pre > 0.999).sum():d} after {(post > 0.999).sum():d}; "
            f">0.99 before {(pre > 0.99).sum():d} after {(post > 0.99).sum():d}; "
            f"post max {post.max():.5f}",
        )

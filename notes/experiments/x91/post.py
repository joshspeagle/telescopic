# Exact largest factor before and after the guard (delta 0.451, band 0.3), counts above 0.999 and 0.9998.
import json

import numpy as np
from guard import *

R = "notes/experiments/_rows/"  # i137's stores, relative to the root of a checkout that holds them
here = __import__("pathlib").Path(__file__).parent


def load(path):
    with open(path) as f:
        return list(map(json.loads, f))


rows = [
    {
        "d": 2,
        "n_steps": x["n_steps"],
        "seed": x["seed"],
        "eps": x["eps"],
        "n": x["n"],
        "w": [x["w_min"], x["w_max"]],
        "wp": 1.0,
    }
    for x in load(R + "i137_baseline.jsonl")
    if x["kind"] == "tune" and x["d"] == 2
]
rows += [dict(r, wp=1.0) for r in load(here / "tunings_d10.jsonl")]
rows += [dict(r, d="unequal", wp=r["w_pred"]) for r in load(here / "tunings_unequal.jsonl")]
for cfg in sorted({(r["d"], r["n_steps"]) for r in rows}, key=str):
    rs = [r for r in rows if (r["d"], r["n_steps"]) == cfg]
    pre = np.array([exact_factor(r["eps"], r["n"], r["w"]).max() for r in rs])
    post = []
    acts = []
    for r in rs:
        e, a = guard(r["eps"], r["n_steps"], np.atleast_1d(r["wp"]), 0.451, 0.3)
        acts.append(a)
        post.append(exact_factor(e, n_of(e, r["n_steps"]), r["w"]).max())
    post = np.array(post)
    print(
        cfg,
        len(rs),
        f"kept {acts.count('kept'):d} moved {acts.count('moved'):d} "
        f"fallback {acts.count('fallback'):d} |",
        f"package >0.999 {(pre > 0.999).sum():d} >0.9998 {(pre > 0.9998).sum():d} | "
        f"guard >0.999 {(post > 0.999).sum():d} >0.9998 {(post > 0.9998).sum():d} | "
        f"guard post median {np.median(post):.3f}",
    )

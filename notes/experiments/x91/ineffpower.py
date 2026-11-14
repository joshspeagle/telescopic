# Size and power of the inefficiency criterion: the ratio of the medians over seeds of the per-seed
# inefficiency, arm over package; fail if the 5% point of its bootstrap distribution (seeds resampled
# within each arm, 2,000 resamples) exceeds 1.20. Seeds are drawn from i137's per-seed values of the
# package (both arms), the arm's multiplied by the true ratio r. 400 simulated experiments per entry.
import json
import sys

import numpy as np

R = "notes/experiments/_rows/"  # i137's stores, relative to the root of a checkout that holds them
with open(R + "i137_nongaussian.jsonl") as f:
    ng = [json.loads(l) for l in f]
rng = np.random.default_rng(20261114)
SS = [int(x) for x in sys.argv[1].split(",")]
RR = [float(x) for x in sys.argv[2].split(",")]


def vals(t, n, m):
    return np.log(
        [
            r["inefficiency"][0]
            for r in ng
            if r["target"] == t and r["n_steps"] == n and r["method"] == m
        ]
    )


def crit(a, p, B=2000):
    ia = rng.integers(0, len(a), (B, len(a)))
    ip = rng.integers(0, len(p), (B, len(p)))
    d = np.median(a[ia], axis=1) - np.median(p[ip], axis=1)
    return np.quantile(d, 0.05) > np.log(1.2)


for lab, v in [
    ("banana n=2 package", vals("banana", 2, "base")),
    ("banana n=2 jitter 0.5", vals("banana", 2, "c")),
    ("banana n=5 jitter 0.5", vals("banana", 5, "c")),
    ("mixture n=2 package", vals("mixture", 2, "base")),
    ("mixture n=2 jitter 0.5", vals("mixture", 2, "c")),
]:
    for S in SS:
        out = []
        for r in RR:
            f = np.mean([crit(rng.choice(v, S) + np.log(r), rng.choice(v, S)) for _ in range(400)])
            out.append((r, round(f, 3)))
        print(lab, f"S={S:d}", out)

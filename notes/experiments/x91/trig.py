import json
import math

import numpy as np

R = "notes/experiments/_rows/"  # i137's stores, relative to the root of a checkout that holds them
with open(R + "i137_baseline.jsonl") as f:
    b = [json.loads(l) for l in f]
b = [r for r in b if r["kind"] in ("tune", "pilot")]


def theta(h):
    return 2 * np.arcsin(np.minimum(h, 2) / 2)


def dists(eps, n, ws):
    h = eps * np.asarray(ws)
    phi = n * theta(h)
    j = np.maximum(1, np.round(phi / math.pi))
    d = np.abs(phi - j * math.pi)
    return np.where(h >= 2, 0.0, d)


def bdist(eps, n, band):
    return (
        dists(eps, n, np.geomspace(1 / (1 + band), 1 + band, 41)).min()
        if band > 0
        else dists(eps, n, [1.0]).min()
    )


f = np.array([r["factor_max"] for r in b])
d = np.array([r["d"] for r in b])
for band in (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
    bd = np.array([bdist(r["eps"], r["n"], band) for r in b])
    for Fres in (0.99, 0.995):
        off = f < Fres
        on = f > 0.999
        # largest delta that triggers none of the off tunings: delta <= min bd over off
        dmax = bd[off].min()
        caught = (bd[on] < dmax).sum()
        print(
            f"band {band:.2f} Fres {Fres}: largest delta keeping all off ({off.sum()}) = {dmax:.3f}; on-res (>0.999) caught {caught}/{on.sum()}; by d: "
            + ", ".join(
                f"d={dd}: {(bd[on & (d == dd)] < dmax).sum()}/{(on & (d == dd)).sum()}"
                for dd in (2, 10)
            )
        )
print("---- scatter by config: counts in (pred dist bins) x (exact factor bins)")
bd0 = np.array([bdist(r["eps"], r["n"], 0) for r in b])
bins = [0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 10]
fb = [0, 0.9, 0.99, 0.999, 1.01]
for key in sorted({(r["d"], r["n_steps"]) for r in b}, key=str):
    idx = [i for i, r in enumerate(b) if (r["d"], r["n_steps"]) == key]
    H = np.histogram2d(bd0[idx], f[idx], bins=[bins, fb])[0].astype(int)
    if (bd0[idx] < 0.8).any():
        print(key)
        print("  dist bins", bins, "cols: f<0.9, 0.9-0.99, 0.99-0.999, >0.999")
        for i in range(len(bins) - 1):
            print(f"  [{bins[i]:.2f},{bins[i + 1]:.2f})", H[i])

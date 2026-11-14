import json

import numpy as np
from model import gfac

R = "notes/experiments/_rows/"  # i137's stores, relative to the root of a checkout that holds them
with open(R + "i137_baseline.jsonl") as f:
    b = [r for r in map(json.loads, f) if r["kind"] == "tune"]
with open(R + "i137_remedies.jsonl") as f:
    c = {(r["d"], r["n_steps"], r["seed"]): r for r in map(json.loads, f) if r["method"] == "c"}
with open(__import__("pathlib").Path(__file__).with_name("tunings_d10.jsonl")) as f:
    d10 = {(r["n_steps"], r["seed"]): r for r in map(json.loads, f)}
print(
    "worst-mode geometric-mean factor over the jitter, at the package's tuned steps (tuning-only seeds 0-99): median / max over tunings"
)
for cfg in [
    (2, 2),
    (2, 5),
    (2, 10),
    (10, 3),
    (10, 5),
    (10, 10),
    (2, None),
    (10, None),
    (10, 2),
    (10, 1),
]:
    rows = [r for r in b if (r["d"], r["n_steps"]) == cfg]
    out = []
    for j in (0.0, 0.2, 0.3, 0.5):
        fs = []
        for r in rows:
            w = (
                np.array([r["w_min"], r["w_max"]])
                if cfg[0] == 2
                else (
                    np.array(d10[(cfg[1], r["seed"])]["w"])
                    if (cfg[1], r["seed"]) in d10
                    else np.linspace(r["w_min"], r["w_max"], 10)
                )
            )
            fs.append(gfac(r["eps"], r["n"], w, j).max())
        fs = np.array(fs)
        out.append(f"j={j:.1f} {np.median(fs):.3f}/{fs.max():.4f} (>0.99: {(fs > 0.99).sum():d})")
    print(cfg, "; ".join(out))
# unequal
with open(__import__("pathlib").Path(__file__).with_name("tunings_unequal.jsonl")) as f:
    u = [json.loads(l) for l in f]
for n in (None, 3, 10):
    out = []
    for j in (0.0, 0.2, 0.3, 0.5):
        fs = np.array(
            [gfac(r["eps"], r["n"], np.array(r["w"]), j).max() for r in u if r["n_steps"] == n]
        )
        out.append(f"j={j:.1f} {np.median(fs):.3f}/{fs.max():.4f} (>0.99: {(fs > 0.99).sum():d})")
    print("unequal", n, "; ".join(out))

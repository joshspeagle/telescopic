# Held-out cells of the revision (tuning only, probe seeds 900-999 and 900-929): P's exact largest factor,
# the guard's action and the factor after it, and the jitters' worst geometric-mean factor at P's steps.
import collections
import json
from pathlib import Path

import numpy as np
from guard import exact_factor, guard, n_of
from model import gfac

here = Path(__file__).parent
with open(here / "tunings_g5.jsonl") as f:
    rows = [
        dict(r, cell=f"gaussian(5) n_steps={r['n_steps']}", wp=1.0) for r in map(json.loads, f)
    ]
with open(here / "tunings_uneq8.jsonl") as f:
    rows += [
        dict(r, cell="unequal d=8, s to 1/5, default", wp=r["w_pred"]) for r in map(json.loads, f)
    ]
g = collections.defaultdict(list)
for r in rows:
    g[r["cell"]].append(r)
for cell, rs in g.items():
    pre = np.array([exact_factor(r["eps"], r["n"], r["w"]).max() for r in rs])
    acts = collections.Counter()
    post = []
    for r in rs:
        e, a = guard(r["eps"], r["n_steps"], np.atleast_1d(r["wp"]), 0.451, 0.3)
        acts[a] += 1
        post.append(exact_factor(e, n_of(e, r["n_steps"]), r["w"]).max())
    post = np.array(post)
    jit = {
        j: np.array([gfac(r["eps"], r["n"], np.array(r["w"]), j).max() for r in rs])
        for j in (0.2, 0.3, 0.5)
    }
    print(
        f"{cell}: {len(rs)} tunings, eps median {np.median([r['eps'] for r in rs]):.3f}, n {sorted({r['n'] for r in rs})}, w range {min(min(r['w']) for r in rs):.3f}-{max(max(r['w']) for r in rs):.3f}"
    )
    print(
        f"  P: exact factor median {np.median(pre):.4f}, >0.99 {(pre > 0.99).sum()}, >0.999 {(pre > 0.999).sum()}, >0.9998 {(pre > 0.9998).sum()}"
    )
    print(
        f"  G: {dict(acts)}; after: median {np.median(post):.4f}, >0.999 {(post > 0.999).sum()}, >0.9998 {(post > 0.9998).sum()}, max {post.max():.6f}"
    )
    print(
        "  jitter worst geometric-mean factor, max over tunings: "
        + ", ".join(f"j={j}: {v.max():.3f}" for j, v in jit.items())
    )

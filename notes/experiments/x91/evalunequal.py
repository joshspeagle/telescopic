import collections
import json

import numpy as np
from guard import *

with open(__import__("pathlib").Path(__file__).with_name("tunings_unequal.jsonl")) as f:
    rows = [json.loads(l) for l in f]
for delta, band in [(0.451, 0.3), (0.451, 0.15), (0.451, 0.0)]:
    print("=== delta", delta, "band", band)
    for nst in (None, 3, 10):
        c = collections.Counter()
        pre = []
        post = []
        eps = []
        for r in rows:
            if r["n_steps"] != nst:
                continue
            f0 = exact_factor(r["eps"], r["n"], r["w"]).max()
            e, a = guard(r["eps"], nst, np.array(r["w_pred"]), delta, band)
            n1 = n_of(e, nst)
            f1 = exact_factor(e, n1, r["w"]).max()
            c[a] += 1
            pre.append(f0)
            post.append(f1)
            eps.append((round(r["eps"], 3), round(e, 3), n1))
        pre = np.array(pre)
        post = np.array(post)
        print(
            nst,
            dict(c),
            f"exact max factor before: >0.999 {(pre > 0.999).sum():d}, "
            f">0.99 {(pre > 0.99).sum():d}; "
            f"after: >0.999 {(post > 0.999).sum():d} >0.99 {(post > 0.99).sum():d}, "
            f"after median {np.median(post):.4f} max {post.max():.6f}",
        )
        if band == 0.3:
            print("   eps before->after (n):", eps[:12])
pe = [max(abs(np.array(sorted(r["w_pred"])) / np.array(sorted(r["w"])) - 1)) for r in rows]
print(
    f"pilot freq max rel error: median {np.median(pe):.3f} max {max(pe):.3f}",
    "tune sec mean {:.2f}".format(np.mean([r["sec"] for r in rows])),
)

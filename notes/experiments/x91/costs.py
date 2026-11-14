import collections
import json

import numpy as np

R = "notes/experiments/_rows/"  # i137's stores, relative to the root of a checkout that holds them


def load(n):
    with open(R + n) as f:
        return [json.loads(l) for l in f]


b = load("i137_baseline.jsonl")
r = load("i137_remedies.jsonl")
u = load("i137_unequal.jsonl")
ng = load("i137_nongaussian.jsonl")


# seconds per record
def summ(rows, keyf, name):
    g = collections.defaultdict(list)
    for x in rows:
        g[keyf(x)].append(x)
    print("==", name)
    for k in sorted(g, key=str):
        xs = g[k]
        s = [x["seconds"] for x in xs]
        gp = [
            x.get("pilot_grads_per_pair") for x in xs if x.get("pilot_grads_per_pair") is not None
        ]
        print(
            k,
            "n",
            len(xs),
            f"sec sum {sum(s):.1f} mean {np.mean(s):.2f} max {max(s):.2f}",
            f"grads/pair mean {np.mean(gp):.0f}" if gp else "",
            "copies",
            sum(1 for x in xs if x.get("same_as")),
        )


summ(
    [x for x in b if x["kind"] != "tune"],
    lambda x: (x["kind"], x["d"], x["n_steps"]),
    "baseline pilot",
)
summ([x for x in b if x["kind"] == "tune"], lambda x: (x["d"], x["n_steps"]), "baseline tune")
summ(r, lambda x: (x["method"], x["d"], x["n_steps"]), "remedies")
summ(u, lambda x: (x["method"], x["n_steps"]), "unequal")
summ(ng, lambda x: (x["target"], x["method"], x["n_steps"]), "nongauss")

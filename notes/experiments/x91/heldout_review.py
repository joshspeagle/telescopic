# The held-out cells the review proposed (gaussian(5) with n_steps 4 and 7; unequal scales to 1/3 in 6
# dimensions), probe seeds 900-999 and 900-929, tuning only: does P resonate there at all?
import json
from pathlib import Path

import numpy as np
from guard import exact_factor

here = Path(__file__).parent


def load(name):
    with open(here / name) as f:
        return list(map(json.loads, f))


cells = [
    ("gaussian(5) n_steps=4", [r for r in load("tunings_g5.jsonl") if r["n_steps"] == 4]),
    ("gaussian(5) n_steps=7", load("tunings_g5_review.jsonl")),
    ("unequal d=6, s to 1/3, default", load("tunings_uneq6.jsonl")),
]
for name, rs in cells:
    f = np.array([exact_factor(r["eps"], r["n"], r["w"]).max() for r in rs])
    print(
        f"{name}: {len(rs)} tunings; P's exact largest factor median {np.median(f):.4f}, max {f.max():.5f}, above 0.99: {(f > 0.99).sum()}"
    )

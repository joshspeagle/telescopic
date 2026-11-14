"""Review of x123: single-precision gradients that are returned as float64.

Question: how often does the check warn when log_prob is float64 and the gradient was
computed in float32 but is returned as float64, so that the check cannot see its
precision and its probe of a narrow gradient does not run?

Design: 10 targets of the registration with a correct gradient, 200 seeds each, 500000
to 500199 (100 for the logistic regression), and two ways of losing precision: the
gradient computed from a float32 copy of x, and the float64 gradient rounded to float32;
both are converted to float64 before they are returned. The last row passes the first
kind per point through .tolist(), which gives Python floats.

Run: PYTHONPATH=src:notes/experiments/x123 python notes/experiments/x123/review/fa4.py > fa4.out

Result: see fa4.out and REVIEW.md, objection 4. (N) warns on 195 of 200 on N(0, I) at
d = 5, on 200 on N(1e3, I), N(1e6, I), the rotated Gaussian, the funnel and the banana,
on 196 on Student t3 and on 30 on pumps; on none on N(0, I) at d = 1000 and on the
logistic regression. (H) warns on 3 of 200 on N(1e6, I), on 8 and 1 of 200 on the two
funnel rows, and on none elsewhere.
The harm of two of these gradients is in harm5_b.out.
"""

import families as F
import numpy as np
from probe import run

S = range(500_000, 500_200)


def disguise(case):
    lp, g, init = case
    return (
        lp,
        (lambda x: np.asarray(g(x.astype(np.float32)), dtype=np.float32).astype(np.float64)),
        init,
    )


def castout(case):  # gradient computed in float64, rounded to float32, returned as float64
    lp, g, init = case
    return lp, (lambda x: np.asarray(g(x), dtype=np.float32).astype(np.float64)), init


for name, case in (
    ("N(0,I) d=5", F.gaussian()),
    ("N(0,I) d=1000 start sd 2", F.gaussian_start(1000, 2.0)),
    ("N(1e3,I) d=5", F.gaussian(mean=1e3)),
    ("N(1e6,I) d=5", F.gaussian(mean=1e6)),
    ("Student t3 d=5 start sd 2", F.student(wide=2.0)),
    ("logistic n=1e4 d=5", F.logistic()),
    ("rotated cond 1e4 d=10", F.conditioned(1e4)),
    ("funnel", F.example("funnel")),
    ("banana", F.example("banana")),
    ("pumps", F.example("pumps")),
):
    seeds = S if "logistic" not in name else range(500_000, 500_100)
    run(f"{name}: float64 log_prob; grad from float32 x, returned float64", disguise(case), seeds)
    run(
        f"{name}: float64 log_prob; float64 grad rounded to float32, returned float64",
        castout(case),
        seeds,
    )
# the same per point through .tolist() (Python floats)
lp, g, init = F.gaussian()
run(
    "N(0,I) d=5 per point: grad from float32 x returned as a Python list",
    ((lambda x: float(lp(x[None])[0])), (lambda x: (-(x.astype(np.float32))).tolist()), init),
    S,
    vectorize=False,
)

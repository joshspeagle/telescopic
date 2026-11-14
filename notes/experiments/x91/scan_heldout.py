# Tuning only, probe seeds 950-955: which new Gaussian cells put P's tuned step near a resonance
# (candidates for held-out cells). Prints, per cell, the exact largest factor at each seed.
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
from guard import exact_factor
from i137_common import gaussian_target

from telescopic import CoupledSampler
from telescopic.kernels import HMC


def run(label, ex, n, cov):
    P = np.linalg.inv(ex.cov)
    fs = []
    for s in range(950, 956):
        sm = CoupledSampler(
            ex.log_prob,
            ex.initial,
            vectorize=True,
            grad=ex.grad,
            kernel=HMC(n_steps=n, cov=cov),
            rng=s,
        )
        sm.tune()
        lf = sm.kernel.kernels[0]
        C = np.linalg.cholesky(sm.kernel.cov)
        w = np.sqrt(np.linalg.eigvalsh(C.T @ P @ C))
        fs.append(exact_factor(lf.step_size, lf.n_steps, w).max())
    print(
        label,
        "n_steps",
        n,
        "eps",
        round(lf.step_size, 3),
        "n",
        lf.n_steps,
        "factors",
        np.round(fs, 5).tolist(),
        flush=True,
    )


mode = sys.argv[1]
if mode == "gauss":
    for d in map(int, sys.argv[2].split(",")):
        for n in map(int, sys.argv[3].split(",")):
            run(f"gaussian({d})", gaussian_target(d), n, None)
else:
    for d, smin in [tuple(map(float, x.split(":"))) for x in sys.argv[2].split(",")]:
        d = int(d)
        ex = gaussian_target(d, np.geomspace(1, smin, d))
        run(f"unequal d={d} smin={smin}", ex, None, np.eye(d))

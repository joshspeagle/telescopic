# Package tuning only (no pilot pairs) of examples.gaussian(10) with n_steps 3, 5, 10 at seeds 0-99:
# the exact whitened frequencies of every tuning, for the guard's design. Seeds are i137's (already seen).
import json
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
from i137_common import gaussian_target

from telescopic import CoupledSampler
from telescopic.kernels import HMC

with open(sys.argv[1], "a") as out:
    ns = [None if a == "None" else int(a) for a in sys.argv[2].split(",")]
    seeds = range(int(sys.argv[3]), int(sys.argv[4]))
    d = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    ex = gaussian_target(d)
    P = np.linalg.inv(ex.cov)
    for n in ns:
        for s in seeds:
            t = time.time()
            sm = CoupledSampler(
                ex.log_prob, ex.initial, vectorize=True, grad=ex.grad, kernel=HMC(n_steps=n), rng=s
            )
            sm.tune()
            lf = sm.kernel.kernels[0]
            C = np.linalg.cholesky(sm.kernel.cov)
            w = np.sqrt(np.linalg.eigvalsh(C.T @ P @ C))
            out.write(
                json.dumps(
                    {
                        "d": d,
                        "n_steps": n,
                        "seed": s,
                        "eps": lf.step_size,
                        "n": lf.n_steps,
                        "w": w.tolist(),
                        "sec": round(time.time() - t, 2),
                    }
                )
                + "\n"
            )

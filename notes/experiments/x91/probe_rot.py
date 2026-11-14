import json
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
from targets import rotated

from telescopic import CoupledSampler
from telescopic.kernels import HMC

for d in map(int, sys.argv[1].split(",")):
    ex = rotated(d)
    for s in (900, 901):
        t = time.time()
        sm = CoupledSampler(
            ex.log_prob, ex.initial, vectorize=True, grad=ex.grad, kernel=HMC(), rng=s
        )
        sm.tune()
        lf = sm.kernel.kernels[0]
        C = np.linalg.cholesky(sm.kernel.cov)
        w = np.sqrt(np.linalg.eigvalsh(C.T @ np.linalg.inv(ex.cov) @ C))
        print(
            json.dumps(
                {
                    "d": d,
                    "seed": s,
                    "eps": round(lf.step_size, 4),
                    "n": lf.n_steps,
                    "sec": round(time.time() - t, 1),
                    "w": np.round(np.quantile(w, [0, 0.1, 0.5, 0.9, 1]), 3).tolist(),
                }
            ),
            flush=True,
        )

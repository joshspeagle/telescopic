import json
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
from probe_tune import CF
from targets import rotated

from telescopic import CoupledSampler
from telescopic.kernels import HMC

ex = rotated()
for arm in ("P", "CF"):
    t = time.time()
    sm = CoupledSampler(
        ex.log_prob,
        ex.initial,
        vectorize=True,
        grad=ex.grad,
        kernel=(HMC() if arm == "P" else CF()),
        rng=900,
    )
    sm.tune()
    lf = sm.kernel.kernels[0]
    C = np.linalg.cholesky(sm.kernel.cov)
    w = np.sqrt(np.linalg.eigvalsh(C.T @ np.linalg.inv(ex.cov) @ C))
    print(
        json.dumps(
            {
                "arm": arm,
                "eps": round(lf.step_size, 4),
                "n": lf.n_steps,
                "tune_grads": sm.target.n_grad,
                "sec": round(time.time() - t, 1),
                "w_min": round(w.min(), 3),
                "w_max": round(w.max(), 3),
            }
        ),
        flush=True,
    )

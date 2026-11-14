# Package tuning only on the Gaussian of unequal scales (d=10, s=geomspace(1,1/4), cov=I given):
# the tuned step, n, and the pilot covariance's predicted frequencies, for the guard's design.
import json
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
import os

from i137_common import gaussian_target

from telescopic import CoupledSampler
from telescopic.kernels.hmc import HMC, _Leapfrog

D = int(os.environ.get("UNEQ_D", "10"))
S = np.geomspace(1, float(os.environ.get("UNEQ_MIN", "0.25")), D)


class Cap(_Leapfrog):
    def _with_cov(self, cov):
        return Cap(self.step_size, self._n_steps, cov)

    def _tuned(self, target, initial, pilot, rng):
        k, rep = super()._tuned(target, initial, pilot, rng)
        k.pilot_cov = pilot.cov
        return k, rep


class H(HMC):
    def __init__(self, n_steps=None, cov=None):
        super().__init__(None, n_steps, cov=cov)
        self.kernels[0] = Cap(None, n_steps, cov)


with open(sys.argv[1], "a") as out:
    ns = [None if a == "None" else int(a) for a in sys.argv[2].split(",")]
    for n in ns:
        for s in range(int(sys.argv[3]), int(sys.argv[4])):
            t = time.time()
            ex = gaussian_target(D, S)
            sm = CoupledSampler(
                ex.log_prob,
                ex.initial,
                vectorize=True,
                grad=ex.grad,
                kernel=H(n_steps=n, cov=np.eye(D)),
                rng=s,
            )
            sm.tune()
            lf = sm.kernel.kernels[0]
            wp = np.sqrt(np.linalg.eigvalsh(np.linalg.inv(lf.pilot_cov)))
            out.write(
                json.dumps(
                    {
                        "n_steps": n,
                        "seed": s,
                        "eps": lf.step_size,
                        "n": lf.n_steps,
                        "w_pred": wp.tolist(),
                        "w": (1 / S).tolist(),
                        "sec": round(time.time() - t, 2),
                    }
                )
                + "\n"
            )

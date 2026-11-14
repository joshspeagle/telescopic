# Tuning only (no pilot pairs), probe seeds 900-902: the step and n_steps that P and CF return, and tuning time.
import json
import sys
import time
import warnings

warnings.simplefilter("ignore")
from i137_common import _hmc_with, _JitterLeapfrog
from targets import *

from telescopic import CoupledSampler, examples
from telescopic.kernels import HMC


class _CF(_JitterLeapfrog):
    target_acceptance = 0.99
    jitter = 0.5


CF = _hmc_with(_CF)


def cases():
    yield "gauss2", examples.gaussian(2), None
    yield "gauss10", examples.gaussian(10), None
    yield "banana_exact", examples.banana(), ExactBanana(2)
    yield "banana_U", examples.banana(), None
    yield "mixture", examples.mixture(), None
    yield "funnel10", examples.funnel(10), None
    yield "banana20", banana20(), ExactBanana(20)
    ex, _X, _y, _fit = logistic()
    yield "logistic", ex, None
    yield "rotated100", rotated(), None


if __name__ == "__main__":
    want = sys.argv[1].split(",")
    arms = sys.argv[2].split(",")
    for name, ex, init in cases():
        if name not in want:
            continue
        for arm in arms:
            for s in (900, 901, 902):
                t = time.time()
                sm = CoupledSampler(
                    ex.log_prob,
                    init or ex.initial,
                    vectorize=True,
                    grad=ex.grad,
                    kernel=(HMC() if arm == "P" else CF()),
                    rng=s,
                )
                sm.tune()
                lf = sm.kernel.kernels[0]
                print(
                    json.dumps(
                        {
                            "target": name,
                            "arm": arm,
                            "seed": s,
                            "eps": round(lf.step_size, 4),
                            "n": lf.n_steps,
                            "tune_grads": sm.target.n_grad,
                            "sec": round(time.time() - t, 2),
                        }
                    ),
                    flush=True,
                )

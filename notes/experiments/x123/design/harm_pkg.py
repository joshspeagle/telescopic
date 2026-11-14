"""Exploratory: harm of three gradients with the package's default-tuned coupled MALA (seeds 219000..219002).

Question: (a) is twice the gradient, harmless at d = 5 in the first experiment, harmful at d = 50, as
delta0 = eps sqrt(d) / 2 = 3 predicts?  (b) does a gradient computed in float32 harm on a rotated
Gaussian of condition 1e4 and 1e8 (d = 10), where the candidates are silent and warn?
Design: CoupledSampler(kernel="mala", rng=seed).meeting_times(100, max_iter=10_000) with each gradient;
reported: the tuned step, the mean meeting time with unmet pairs counted at 10,000, the unmet count.
usage: harm_pkg.py twice|f32
Result: see harm_pkg_*.out.
"""

import sys
import time
import warnings

import cases
import numpy as np

from telescopic import CoupledSampler


def run(case, seed):
    lp, g, init = case
    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = CoupledSampler(lp, init, grad=g, vectorize=True, kernel="mala", rng=seed)
        tau = s.meeting_times(100, max_iter=10_000)
    return (
        float(np.mean(np.minimum(tau, 10_000))),
        int(np.isinf(tau).sum()),
        s.kernel.step_size,
        time.time() - t,
    )


if __name__ == "__main__":
    if sys.argv[1] == "twice":
        right = cases.gaussian(d=50)
        pairs = [
            ("N(0,I) d=50: twice the gradient", right, cases.modify(right, lambda x, g: 2 * g)),
            ("N(0,I) d=50: sign of coordinate 0", right, cases.modify(right, cases.flip(0))),
        ]
    else:
        pairs = []
        for c in (1e4, 1e8):
            right = cases.conditioned(c)
            pairs.append(
                (f"rotated condition {c:g} d=10: float32 gradient", right, cases.rotated32grad(c))
            )
    for name, right, wrong in pairs:
        for seed in (219000, 219001, 219002):
            r, w = run(right, seed), run(wrong, seed)
            print(
                f"{name}, seed {seed}: correct mean {r[0]:.1f} (unmet {r[1]}, step {r[2]:.3f}); other mean {w[0]:.1f} "
                f"(unmet {w[1]}, step {w[2]:.3f}); ratio of means {w[0] / r[0]:.2f}, of steps {w[2] / r[2]:.3f}; {r[3] + w[3]:.0f}s",
                flush=True,
            )

"""Do the remedies' coupled steps have the marginals of their steps alone? (#137)

Question: the remedies of i137_common.py must pass `telescopic.testing.check_coupling`
before any meeting time counts. Remedies c and d (a step jittered by a uniform that both
chains share) change the kernel; a and b change only the tuned step and keep the
package's coupled step, which is checked here too, tuned as they tune it.

Design: each remedy tuned on examples.gaussian(d) at seed 0, d = 2 and 10, n_steps = 2
and 5 (8 kernels per remedy). From two pairs of points, check_coupling with 20,000
pairs, check seed 1: a near pair, the origin and a point 2e-4 away along each axis
(within the meeting scale of 1e-3, so pairs meet), and a far pair, the origin and the
point (1, ..., 1), where pairs cannot meet in one step and only the trajectory's
marginals are compared. Pass: for the near pair `passed()` at its default level 1e-3
(four p-values, pairs met, no stale state, no near miss); for the far pair the same
without the condition that pairs met.

Result (i137_check_coupling.out): all 40 checks pass, 20 near and 20 far. The smallest
of the 160 p-values is 0.0040 (remedy b, a far pair; the package's coupled step with
b's step); for the new kernels c and d it is 0.026, against a level of 1e-3.

Run: PYTHONPATH=src:notes/experiments python notes/experiments/i137_check_coupling.py
"""

import warnings

import numpy as np
from i85_setup import setup
from i137_common import METHODS, gaussian_target

from telescopic import CoupledSampler, Target
from telescopic.testing import check_coupling

warnings.simplefilter("ignore")
print(setup())
smallest, failures = {}, 0
for method in ["base", "a", "b", "c", "d"]:
    for d in (2, 10):
        example = gaussian_target(d)
        target = Target(example.log_prob, example.grad, vectorize=True)
        for n_steps in (2, 5):
            sampler = CoupledSampler(
                example.log_prob,
                example.initial,
                vectorize=True,
                grad=example.grad,
                kernel=METHODS[method](n_steps=n_steps),
                rng=0,
            )
            sampler.tune()
            kernel = sampler.kernel
            for name, x, y in (
                ("near", np.zeros(d), np.full(d, 2e-4)),
                ("far", np.zeros(d), np.ones(d)),
            ):
                report = check_coupling(kernel, target, x, y, n=20_000, rng=1)
                p = min(*report.stay, *report.moves)
                ok = (
                    report.passed()
                    if name == "near"
                    else (p >= 1e-3 and not report.stale and not report.near_misses)
                )
                failures += not ok
                smallest[method] = min(smallest.get(method, 1.0), p)
                print(
                    f"{method:4} d={d:2} n_steps={n_steps} eps={kernel.step_size:.4f} "
                    f"jitter={getattr(kernel.kernels[0], 'jitter', 0.0) if method in 'cd' else 0.0:.3f} "
                    f"{name:4}: stay p={report.stay[0]:.3f},{report.stay[1]:.3f} "
                    f"moves p={report.moves[0]:.3f},{report.moves[1]:.3f} met={report.met:.4f} "
                    f"stale={report.stale} near_misses={report.near_misses} "
                    f"passed={ok}",
                    flush=True,
                )
for method, p in smallest.items():
    print(f"{method}: smallest p-value over 8 checks {p:.4f}")
print(f"checks failed: {failures} of 40")

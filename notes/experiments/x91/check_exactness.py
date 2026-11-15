"""Do the new coupled steps of x91 have the marginals of their steps alone? (PREREGISTRATION.md, section 3)

Design, as section 3 fixes it: every new coupled step (J2, J3, J5, CF, CF.2, MJ5, MCF),
tuned by the sampler on examples.gaussian(d) at seed 9700, d = 2 and 10, the HMC arms at
n_steps 2, 5 and default; from a near pair (the origin and a point 2e-4 away along each
axis) and a far pair (the origin and (1, ..., 1)), `check_coupling` with 20,000 pairs at
check seed 9701. Pass: `passed()` at its default level 1e-3 for the near pair; for the
far pair the same without the condition that pairs met (as i137_check_coupling.py). A
failed check is repeated once with 80,000 pairs at check seed 9702 and counts as a
failure only if it fails again. 68 checks in all.

Run: OMP_NUM_THREADS=1 PYTHONPATH=src:notes/experiments:notes/experiments/x91 \
    python notes/experiments/x91/check_exactness.py
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import warnings

import numpy as np
from arms import kernel
from cells import SEEDS
from i85_setup import setup
from run import uncommitted

from telescopic import CoupledSampler, examples
from telescopic.testing import check_coupling

ARMS = ("J2", "J3", "J5", "CF", "CF.2", "MJ5", "MCF")
SEED = SEEDS["check_coupling"]


def passes(report, near):
    if near:
        return report.passed()
    # np.min, not min, so that a NaN p-value fails, as in passed().
    p = np.min([*report.stay, *report.moves])
    return bool(p >= 1e-3) and not report.stale and not report.near_misses


def main():
    warnings.simplefilter("ignore")
    print(setup(), flush=True)
    changed = uncommitted().strip().replace("\n", "; ")
    print(f"uncommitted files: {changed or 'none'}", flush=True)
    failed, checks = set(), 0
    for arm in ARMS:
        for d in (2, 10):
            example = examples.gaussian(d)
            for n_steps in (2, 5, None) if not arm.startswith("M") else (None,):
                sampler = CoupledSampler(
                    example.log_prob,
                    example.initial,
                    vectorize=True,
                    grad=example.grad,
                    kernel=kernel(arm, n_steps),
                    rng=SEED["tuning"],
                )
                sampler.tune()
                tuned, target = sampler.kernel, sampler.target
                setting = "" if arm.startswith("M") else f" n_steps={n_steps}"
                for name, x, y in (
                    ("near", np.zeros(d), np.full(d, 2e-4)),
                    ("far", np.zeros(d), np.ones(d)),
                ):
                    checks += 1
                    for n, seed, label in (
                        (20_000, SEED["check"], ""),
                        (80_000, SEED["repeat"], " repeat"),
                    ):
                        report = check_coupling(tuned, target, x, y, n=n, rng=seed)
                        ok = passes(report, name == "near")
                        print(
                            f"{arm:4} d={d:2}{setting:13} eps={tuned.step_size:.4f} {name:4}{label}: "
                            f"stay p={report.stay[0]:.3f},{report.stay[1]:.3f} moves "
                            f"p={report.moves[0]:.3f},{report.moves[1]:.3f} met={report.met:.4f} "
                            f"stale={report.stale} near_misses={report.near_misses} passed={ok}",
                            flush=True,
                        )
                        if ok:
                            break
                    else:
                        failed.add(arm)
    print(f"checks: {checks}")
    print(f"kernels that failed their repeat: {', '.join(sorted(failed)) or 'none'}")


if __name__ == "__main__":
    main()

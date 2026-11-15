"""Whether the trimmed `Streams._take` keeps every bit of every seeded output (issue #165).

Question: the change of issue #165 makes `Streams._take` cheaper per request without
changing which numbers it returns. Do seeded runs give the same bits before and after?

Method. Each tree is run in its own subprocess, with its `src` first on `PYTHONPATH`,
and prints the digests of `notes/experiments/i77_kernels/seeded_outputs.py` (the engine
with every shipped kernel at 1 and 16 lanes, point by point and vectorized, on a
Gaussian and a truncated Gaussian, Gibbs, and `sample` untuned, so that the pilot and
dual averaging run, and on a pool of 2 processes) and these lines, which cover requests
that widen a buffer and lanes that leave and enter at different times:
- the engine with `MALA` and `HMC` at d = 100 (normals of size 100, over an eighth of
  the buffer), at 1, 7 and 256 lanes, with `max_iter` so that some pairs are censored;
- `dual_averaging` alone, for `MALA` and `HMC`;
- `testing.random_draws` and `testing.fixed_draws`, with requests of sizes 1, 3 and 100.
The parent prints both trees' setup lines and the lines that differ, then a verdict.

Run as `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python
notes/experiments/i165_take/digests.py BEFORE AFTER`, where BEFORE and AFTER are the roots
of the two checkouts, such as one made with `git worktree add --detach <path> origin/main`.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def lines():
    sys.path.insert(0, str(HERE.parent / "i77_kernels"))
    from seeded_outputs import digest, main, replicates_digest

    from telescopic import engine, examples, testing, tuning
    from telescopic.kernels import HMC, MALA
    from telescopic.target import Target

    main()
    big = examples.gaussian(100)
    cov = big.cov
    target = Target(big.log_prob, big.grad, vectorize=True)
    for name, kernel in {"mala": MALA(0.3, cov=cov), "hmc": HMC(0.2, 5, cov=cov)}.items():
        for lanes in (1, 7, 256):
            replicates = engine.run(
                target,
                kernel,
                big.initial,
                np.arange(24),
                k=2,
                m=10,
                lag=1,
                max_iter=60,
                rng=13,
                lanes=lanes,
            )
            print(f"engine d=100     {name:13} lanes={lanes:3}  {replicates_digest(replicates)}")
    small = examples.gaussian(3)
    pilot = tuning.adaptive_metropolis(Target(small.log_prob, small.grad), small.initial, 14)
    for name, kernel_at in {
        "mala": lambda e: MALA(e, cov=small.cov),
        "hmc": lambda e: HMC(e, cov=small.cov),
    }.items():
        step = tuning.dual_averaging(
            kernel_at, Target(small.log_prob, small.grad), pilot, 15, 0.57, 0.5
        )
        print(f"dual_averaging   {name:13} {step!r}")
    random = testing.random_draws(5, 16)
    taken = [random.normal(3), random.uniform(), random.normal(100), random.log_uniform()]
    print(f"random_draws     {digest(*taken)}")
    fixed = testing.fixed_draws(
        normals=np.arange(5 * 210.0).reshape(5, 210), uniforms=np.linspace(0, 1, 15).reshape(5, 3)
    )
    taken = [
        fixed.normal(3),
        fixed.uniform(),
        fixed.normal(100),
        fixed.uniform(),
        fixed.subset([4, 0]).normal(100),
        fixed.log_uniform(),
    ]
    print(f"fixed_draws      {digest(*taken)}")


def run(root):
    environment = dict(os.environ, PYTHONPATH=str(Path(root).resolve() / "src"))
    output = subprocess.run(
        [sys.executable, __file__, "--child"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return output.splitlines()


def compare(before, after):
    old, new = run(before), run(after)
    print("before", old[0])
    print("after ", new[0])
    # The setup lines name the commits; every other line must agree.
    different = [(a, b) for a, b in zip(old[1:], new[1:], strict=True) if a != b]
    for a, b in different:
        print("before", a)
        print("after ", b)
    print(f"{len(old) - 1} lines, {len(different)} differ")


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        lines()
    else:
        compare(*sys.argv[1:])

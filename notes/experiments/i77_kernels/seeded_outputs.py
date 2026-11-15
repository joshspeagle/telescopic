"""Digests of seeded runs, to check that re-basing the kernels keeps every bit (issue #77).

Question: issue #77 re-bases `RandomWalk`, `MALA` and `Independence` on one
Metropolis-Hastings kernel that takes a proposal and a coupling object, and its design
asks every commit to keep every bit of every seeded output. Do the runs below give the
same numbers before and after each commit?

Method. Each line runs the engine (`telescopic.engine.run`) or `sample`, with a fixed
seed, and prints the first 16 hexadecimal digits of the SHA-256 of every array of the
run's `Replicates` (the meeting times, the atoms, weights and owners, the distances, the
zero-density replicates and the counts of evaluations), with `seconds` left out, and, for
`sample`, of the tuned kernel's description and the pilot meeting times. The runs cover:
- RandomWalk with the reflection and the maximal coupling, MALA, Independence, HMC,
  Gibbs and `Mixture(RandomWalk, Independence)`, tuned, through the engine, with
  `vectorize` False and True and at 1 and 16 lanes, on a correlated Gaussian of 3
  coordinates, and RandomWalk, MALA and Independence on the same Gaussian truncated to
  the half-space x_0 > -1, so that some proposals have a log-density of -inf;
- RandomWalk with both couplings, MALA, Independence, HMC and the mixture, untuned,
  through `sample`, so that adaptive Metropolis, dual averaging and
  `Independence._tuned` run, with `vectorize` False and True;
- one run of `sample` on a pool of 2 processes.

Run as `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python
notes/experiments/i77_kernels/seeded_outputs.py`, with the package to check first on
`PYTHONPATH`. The first lines give the setup and NumPy's build; compare the digests only
between runs whose setup lines agree but for the commit.
"""

import contextlib
import dataclasses
import hashlib
import io
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from i85_setup import setup

import telescopic
from telescopic import engine, examples
from telescopic.kernels import HMC, MALA, Gibbs, Independence, Mixture, MultivariateT, RandomWalk
from telescopic.target import Target

GAUSSIAN = examples.gaussian(3)
COV = GAUSSIAN.cov
TRUNCATED = ("rw reflection", "mala", "independence")


def truncated_log_prob(x):
    return np.where(x[..., 0] > -1.0, GAUSSIAN.log_prob(x), -np.inf)


def truncated_grad(x):
    return np.where(x[..., :1] > -1.0, GAUSSIAN.grad(x), 0.0)


def digest(*values):
    sha = hashlib.sha256()
    for value in values:
        if isinstance(value, str):
            sha.update(value.encode())
        else:
            array = np.ascontiguousarray(value)
            sha.update(str(array.dtype).encode() + str(array.shape).encode())
            sha.update(array.tobytes())
    return sha.hexdigest()[:16]


def replicates_digest(replicates):
    arrays = [
        getattr(replicates, field.name)
        for field in dataclasses.fields(replicates)
        if field.name not in ("seconds", "seed", "settings", "paths")
    ]
    return digest(*arrays)


def parameters(kernel):
    # The tuned kernel's description and arrays: its covariance, step size and proposal,
    # and those of its components.
    if isinstance(kernel, Mixture) and not isinstance(kernel, HMC):
        return [repr(kernel)] + [value for part in kernel.kernels for value in parameters(part)]
    proposal = getattr(kernel, "proposal", None)
    values = [
        getattr(kernel, "cov", None),
        getattr(kernel, "step_size", None),
        getattr(proposal, "mean", None),
        getattr(proposal, "cov", None),
    ]
    return [repr(kernel)] + [np.asarray(value) for value in values if value is not None]


def tuned_kernels():
    proposal = MultivariateT(GAUSSIAN.mean, 2 * COV)
    return {
        "rw reflection": RandomWalk(COV),
        "rw maximal": RandomWalk(COV, coupling="maximal"),
        "mala": MALA(0.8, cov=COV),
        "independence": Independence(proposal),
        "hmc": HMC(0.3, 4, cov=COV),
        "mixture": Mixture([RandomWalk(COV), Independence(proposal)], [0.5, 0.5]),
    }


def untuned_kernels():
    return {
        "rw reflection": RandomWalk(),
        "rw maximal": RandomWalk(coupling="maximal"),
        "mala": MALA(),
        "independence": Independence(),
        "hmc": HMC(),
        "mixture": Mixture([RandomWalk(), Independence()], [0.5, 0.5]),
    }


def engine_lines():
    targets = {
        "gaussian": (GAUSSIAN.log_prob, GAUSSIAN.grad),
        "truncated": (truncated_log_prob, truncated_grad),
    }
    for target_name, (log_prob, grad) in targets.items():
        for vectorize in (False, True):
            target = Target(log_prob, grad, vectorize=vectorize)
            for name, kernel in tuned_kernels().items():
                if target_name == "truncated" and name not in TRUNCATED:
                    continue
                for lanes in (1, 16):
                    replicates = engine.run(
                        target,
                        kernel,
                        GAUSSIAN.initial,
                        np.arange(16),
                        k=2,
                        m=10,
                        lag=1,
                        max_iter=3000,
                        rng=7,
                        lanes=lanes,
                    )
                    yield (
                        f"engine {target_name:9} {name:13} vectorize={vectorize!s:5} "
                        f"lanes={lanes:2}  {replicates_digest(replicates)}"
                    )
    pumps = examples.pumps()
    gibbs = Gibbs(pumps.conditional, pumps.blocks)
    for lanes in (1, 16):
        replicates = engine.run(
            Target(None), gibbs, pumps.initial, np.arange(16), k=2, m=10, lag=1, rng=8, lanes=lanes
        )
        yield f"engine pumps     gibbs         lanes={lanes:2}  {replicates_digest(replicates)}"


def sample_lines():
    for vectorize in (False, True):
        for name, kernel in untuned_kernels().items():
            result = telescopic.sample(
                GAUSSIAN.log_prob,
                GAUSSIAN.initial,
                20,
                grad=GAUSSIAN.grad,
                vectorize=vectorize,
                kernel=kernel,
                rng=11,
            )
            pilot = () if result.pilot is None else (result.pilot,)
            yield (
                f"sample untuned   {name:13} vectorize={vectorize!s:5}  "
                f"{replicates_digest(result._replicates)} "
                f"kernel {digest(*parameters(result.kernel), *pilot)}"
            )
    result = telescopic.sample(
        GAUSSIAN.log_prob, GAUSSIAN.initial, 20, kernel=RandomWalk(COV), pool=2, rng=12
    )
    yield f"sample pool=2    rw reflection                     {replicates_digest(result._replicates)}"


def main():
    print(setup())
    build = io.StringIO()
    with contextlib.redirect_stdout(build):
        np.show_config()
    # The BLAS's name and configuration, without the build's paths.
    blas = [
        line.strip()
        for line in build.getvalue().splitlines()
        if line.strip().startswith(("name:", "openblas configuration:", "version:"))
    ]
    print("numpy build:", "; ".join(dict.fromkeys(blas)))
    pinned = all(
        os.environ.get(name) == "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    print("threads pinned:", pinned)
    for line in engine_lines():
        print(line)
    for line in sample_lines():
        print(line)


if __name__ == "__main__":
    main()

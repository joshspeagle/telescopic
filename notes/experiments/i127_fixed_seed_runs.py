"""Digests of fixed-seed runs, to compare the package's numbers between two commits (issue #127).

Question: issue #127 changed how `Gibbs(coupling="crn")` draws a block that a common
uniform does not serve. Do runs that never take that branch read the numbers they read
before: the Gibbs sampler of `examples.pumps`, whose blocks have one coordinate and
`ppf`, under the maximal coupling with `eta` 1 and 0.6, under "crn" and under
"two-scale", and a random walk?

Method. For each, 100 pilot meeting times with `rng=1` and a run of 100 replicates with
`rng=2` (the random walk: 50 replicates of a Gaussian target of 3 coordinates with
`rng=3`); a line prints the sum of the meeting times and the first 16 hexadecimal digits
of the SHA-256 of the meeting times, and of the atoms and weights of the run's signed
measure. Run it at two commits; equal lines are runs whose numbers did not change.

Result: `i127_fixed_seed_runs_main.out`, at the merge base of the change, and
`i127_fixed_seed_runs.out`, at the change, print the same lines but for their setup
lines. Run as `python notes/experiments/i127_fixed_seed_runs.py`, with the package to
compare first on `PYTHONPATH`.
"""

import hashlib

import numpy as np
from i85_setup import setup

import telescopic
from telescopic import examples
from telescopic.kernels import Gibbs, RandomWalk


def digest(*arrays):
    sha = hashlib.sha256()
    for array in arrays:
        sha.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return sha.hexdigest()[:16]


def line(name, result, tau=None):
    measure = result.measure
    head = "" if tau is None else f"tau sum {int(np.sum(tau)):5d} {digest(tau)}  "
    return f"{name:22} {head}run tau {digest(result.meeting_times)} atoms {digest(measure.atoms, measure.weights)}"


def main():
    print(setup())
    pumps = examples.pumps()
    threshold = float(np.sqrt(np.trace(pumps.cov)))
    kernels = {
        "pumps maximal": Gibbs(pumps.conditional, pumps.blocks),
        "pumps maximal eta 0.6": Gibbs(pumps.conditional, pumps.blocks, eta=0.6),
        "pumps crn": Gibbs(pumps.conditional, pumps.blocks, coupling="crn"),
        "pumps two-scale": Gibbs(
            pumps.conditional, pumps.blocks, coupling="two-scale", threshold=threshold
        ),
    }
    for name, kernel in kernels.items():
        sampler = telescopic.CoupledSampler(None, pumps.initial, kernel=kernel, rng=1)
        tau = sampler.meeting_times(100)
        result = telescopic.sample(None, pumps.initial, 100, kernel=kernel, rng=2)
        print(line(name, result, tau))
    gaussian = examples.gaussian(3)
    kernel = RandomWalk(gaussian.cov)
    result = telescopic.sample(gaussian.log_prob, gaussian.initial, 50, kernel=kernel, rng=3)
    print(line("random walk", result))


if __name__ == "__main__":
    main()

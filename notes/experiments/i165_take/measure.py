"""The share of a step spent in `_random.Streams._take` (issue #165).

Question: every request for normals or uniforms goes through `Streams._take`, which has
a fixed cost on top of the numbers it returns. The tuning pilot draws its numbers 256
steps at a time (`tuning.DRAW_BLOCK`, DECISIONS #213), which made it 1.24 to 1.32 times
faster. Dual averaging and the engine still make one request per step. Is the time in
`_take` at least about 5% of a step, at 1 or 256 lanes, so that blocking them would pay?

Method. A nearly free target: the standard normal in d = 2, log-density and gradient
vectorized. Each workload runs twice per round, in alternation:
- plain, timed as a whole in CPU time of the process (`time.process_time`), since the
  machine may be shared with other jobs;
- instrumented, with `Streams._take` wrapped so that the wall time inside each call is
  summed (one `perf_counter` pair per call, about 0.1 us, is counted as `_take`'s), and
  its requests and the numbers they return counted.
The share is the median over rounds of the instrumented time inside `_take` per step
divided by the median over rounds of the plain time per step. The fixed part is that
time less the time to generate the numbers returned, measured in blocks of 2^16: it
bounds what drawing in blocks could save, since a block still generates the same numbers.

Workloads:
- `dual_averaging` itself, as `MALA` and `HMC` tuning run it, on its 4 chains (it always
  runs 4), 1000 steps, from pilot states drawn from the target;
- the loop of `dual_averaging` alone, `kernel.step` with a fixed step size on a view of
  a `Streams`, at 1 and 256 lanes, 1000 steps, for `MALA` and `HMC`;
- `engine.run` with `RandomWalk` and `MALA` (tuned by hand: scale 2.38 / sqrt(d) and step
  size 1.65 d^(-1/6), identity covariance), k = 10, m = 20, lag 1, from N(0, 4 I), at
  1 lane (64 replicates) and 256 lanes (1024 replicates); the time per step is the
  run's time divided by its calls to `_Batch.advance`.

Run as `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python
notes/experiments/i165_take/measure.py`; the first line gives the setup. To compare two
commits, run it on each in turn, alternating, with the package of each first on
`PYTHONPATH`.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from i85_setup import setup

from telescopic import engine, tuning
from telescopic._random import NORMAL, UNIFORM, Streams, replicate_generator, root_seed
from telescopic.initial import Normal
from telescopic.kernels import HMC, MALA, RandomWalk
from telescopic.kernels.base import evaluate
from telescopic.target import Target

D = 2
ROUNDS = 7
STEPS = 1000
EYE = np.eye(D)
TARGET = Target(lambda x: -0.5 * np.einsum("ij,ij->i", x, x), lambda x: -x, vectorize=True)


class Probe:
    # Wraps Streams._take and Batch.advance to sum the time inside _take and count calls.
    def __init__(self):
        self.take, self.advance = Streams._take, engine._Batch.advance

    def __enter__(self):
        self.calls = self.steps = 0
        self.seconds = 0.0
        self.numbers = [0, 0]  # normals and uniforms returned
        take, advance = self.take, self.advance

        def timed(streams, kind, lanes, size):
            start = time.perf_counter()
            out = take(streams, kind, lanes, size)
            self.seconds += time.perf_counter() - start
            self.calls += 1
            self.numbers[kind] += len(lanes) * size
            return out

        def counted(batch):
            self.steps += 1
            advance(batch)

        Streams._take, engine._Batch.advance = timed, counted
        return self

    def __exit__(self, *_):
        Streams._take, engine._Batch.advance = self.take, self.advance


def generation_seconds():
    # Seconds to generate one normal and one uniform, in blocks of 2^16.
    generator, out = replicate_generator(root_seed(1), 0, 0), np.empty(2**16)
    times = []
    for fill in (generator.standard_normal, generator.random):
        rounds = []
        for _ in range(ROUNDS):
            start = time.perf_counter()
            for _ in range(100):
                fill(out=out)
            rounds.append((time.perf_counter() - start) / (100 * len(out)))
        times.append(float(np.median(rounds)))
    return times


def pilot():
    samples = np.random.default_rng(3).standard_normal((tuning.N_CHAINS, 10, D))
    log_p = TARGET.log_prob(samples.reshape(-1, D)).reshape(tuning.N_CHAINS, 10)
    return tuning.PilotRun(samples, log_p, np.full(tuning.N_CHAINS, 0.5), 20)


def dual_averaging(kernel_at):
    run = pilot()
    return lambda: tuning.dual_averaging(kernel_at, TARGET, run, 5, 0.57, 0.5)


def loop(kernel, lanes):
    def call():
        root = root_seed(7)
        streams = Streams(lanes)
        for lane in range(lanes):
            streams.assign(lane, replicate_generator(root, 0, lane))
        draws = streams.view(np.arange(lanes))
        state = evaluate(TARGET, np.random.default_rng(1).standard_normal((lanes, D)), True)
        state.extra = kernel.init_extra(state)
        for _ in range(STEPS):
            state, _ = kernel.step(TARGET, state, draws)

    return call


def run(kernel, lanes, replicates):
    initial = Normal(np.zeros(D), 4 * EYE)
    return lambda: engine.run(
        TARGET,
        kernel,
        initial,
        np.arange(replicates),
        k=10,
        m=20,
        lag=1,
        rng=11,
        lanes=lanes,
        store=False,
    )


def main():
    print(setup())
    probe = Probe()
    mala, hmc = MALA(1.65 * D ** (-1 / 6), cov=EYE), HMC(0.5, cov=EYE)
    walk = RandomWalk(EYE, scale=2.38 / np.sqrt(D))
    workloads = {
        ("dual_averaging MALA", 4): dual_averaging(lambda e: MALA(e, cov=EYE)),
        ("dual_averaging HMC", 4): dual_averaging(lambda e: HMC(e, cov=EYE)),
        ("step loop MALA", 1): loop(mala, 1),
        ("step loop MALA", 256): loop(mala, 256),
        ("step loop HMC", 1): loop(hmc, 1),
        ("step loop HMC", 256): loop(hmc, 256),
        ("engine RandomWalk", 1): run(walk, 1, 64),
        ("engine RandomWalk", 256): run(walk, 256, 1024),
        ("engine MALA", 1): run(mala, 1, 64),
        ("engine MALA", 256): run(mala, 256, 1024),
    }
    plain = {key: [] for key in workloads}
    inside = {key: [] for key in workloads}
    steps, requests, numbers = {}, {}, {}
    for key, call in workloads.items():  # an uncounted warm-up, which counts the steps
        with probe:
            call()
        steps[key] = probe.steps if key[0].startswith("engine") else STEPS
        requests[key] = probe.calls / steps[key]
        numbers[key] = [count / steps[key] for count in probe.numbers]
    for _ in range(ROUNDS):
        for key, call in workloads.items():
            with probe:
                call()
            inside[key].append(probe.seconds / steps[key])
            start = time.process_time()
            call()
            plain[key].append((time.process_time() - start) / steps[key])
    per_number = generation_seconds()
    print(
        f"{ROUNDS} rounds, each workload instrumented then plain; medians over the rounds. "
        f"Generation in blocks: {1e9 * per_number[NORMAL]:.1f} ns per normal, "
        f"{1e9 * per_number[UNIFORM]:.1f} ns per uniform. Fixed: the time in _take less "
        "the generation of the numbers it returned"
    )
    print(
        f"{'workload':20} {'lanes':>5} {'steps':>5} {'us/step':>8} {'requests':>8} "
        f"{'us in _take':>11} {'share':>6} {'us fixed':>8} {'share':>6}"
    )
    for key in workloads:
        per_step, in_take = np.median(plain[key]), np.median(inside[key])
        fixed = in_take - np.dot(numbers[key], per_number)
        print(
            f"{key[0]:20} {key[1]:5d} {steps[key]:5d} {1e6 * per_step:8.1f} "
            f"{requests[key]:8.2f} {1e6 * in_take:11.2f} {in_take / per_step:6.1%} "
            f"{1e6 * fixed:8.2f} {fixed / per_step:6.1%}"
        )


if __name__ == "__main__":
    main()

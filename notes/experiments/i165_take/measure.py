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
- instrumented, with the two requests of a kernel, `Draws.normal` and `Draws.uniform`
  (which `log_uniform` calls), wrapped so that the wall time inside each is summed (one
  `perf_counter` pair per request, about 0.1 us, is counted as the request's), and the
  requests and the numbers they return counted. Both serve a request through
  `Streams._take` (or, for uniforms after #165, its vector form).
The share is the median over rounds of the instrumented time in requests per step
divided by the median over rounds of the plain time per step. The script also times the
generation of one normal and one uniform in blocks of 2^16 (`per_call` of
`i77_kernels/step_times.py`, CPU time); `summarize.py` subtracts the generation of the
numbers returned, at one cost pooled over all the runs it reads, to give the fixed part,
which bounds what drawing in blocks could save, since a block still generates the same
numbers.

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
notes/experiments/i165_take/measure.py`; the first line gives the setup, the second the
generation costs, then one line per workload: its name, lanes, steps, the plain time per
step, requests per step, normals and uniforms per step, and the time in requests per
step (times in us). To compare two commits, run it on each in turn, alternating, with
the package of each first on `PYTHONPATH`, and give the outputs to `summarize.py`.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "i77_kernels"))
from i85_setup import setup
from step_times import per_call

from telescopic import engine, tuning
from telescopic._random import Draws, Streams, replicate_generator, root_seed
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
    # Wraps the requests of Draws and Batch.advance, to sum the time in requests and count
    # them, the numbers they return and the engine's steps.
    def __init__(self):
        self.normal, self.uniform = Draws.normal, Draws.uniform
        self.advance = engine._Batch.advance

    def __enter__(self):
        self.calls = self.steps = 0
        self.seconds = 0.0
        self.numbers = [0, 0]  # normals and uniforms returned
        normal, uniform, advance = self.normal, self.uniform, self.advance

        def timed_normal(draws, size):
            start = time.perf_counter()
            out = normal(draws, size)
            self.seconds += time.perf_counter() - start
            self.calls += 1
            self.numbers[0] += out.size
            return out

        def timed_uniform(draws):
            start = time.perf_counter()
            out = uniform(draws)
            self.seconds += time.perf_counter() - start
            self.calls += 1
            self.numbers[1] += out.size
            return out

        def counted(batch):
            self.steps += 1
            advance(batch)

        Draws.normal, Draws.uniform = timed_normal, timed_uniform
        engine._Batch.advance = counted
        return self

    def __exit__(self, *_):
        Draws.normal, Draws.uniform = self.normal, self.uniform
        engine._Batch.advance = self.advance


def generation_seconds():
    # CPU seconds to generate one normal and one uniform, in blocks of 2^16.
    generator, out = replicate_generator(root_seed(1), 0, 0), np.empty(2**16)
    fills = (generator.standard_normal, generator.random)
    return [per_call(lambda fill=fill: fill(out=out)) / len(out) for fill in fills]


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
        ("dual_averaging/MALA", 4): dual_averaging(lambda e: MALA(e, cov=EYE)),
        ("dual_averaging/HMC", 4): dual_averaging(lambda e: HMC(e, cov=EYE)),
        ("step_loop/MALA", 1): loop(mala, 1),
        ("step_loop/MALA", 256): loop(mala, 256),
        ("step_loop/HMC", 1): loop(hmc, 1),
        ("step_loop/HMC", 256): loop(hmc, 256),
        ("engine/RandomWalk", 1): run(walk, 1, 64),
        ("engine/RandomWalk", 256): run(walk, 256, 1024),
        ("engine/MALA", 1): run(mala, 1, 64),
        ("engine/MALA", 256): run(mala, 256, 1024),
    }
    plain = {key: [] for key in workloads}
    inside = {key: [] for key in workloads}
    steps, requests, numbers = {}, {}, {}
    for key, call in workloads.items():  # an uncounted warm-up, which counts the steps
        with probe:
            call()
        steps[key] = probe.steps if key[0].startswith("engine/") else STEPS
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
        f"generation: {1e9 * per_number[0]:.2f} ns per normal, {1e9 * per_number[1]:.2f} ns per uniform"
    )
    for key in workloads:
        per_step, in_requests = np.median(plain[key]), np.median(inside[key])
        print(
            f"{key[0]} {key[1]} {steps[key]} {1e6 * per_step:.2f} {requests[key]:.3f} "
            f"{numbers[key][0]:.2f} {numbers[key][1]:.2f} {1e6 * in_requests:.3f}"
        )


if __name__ == "__main__":
    main()

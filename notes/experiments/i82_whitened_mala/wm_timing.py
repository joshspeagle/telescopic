"""How much faster is a step of `WhitenedMALA` than one of `MALA`?

Question. Ruling 10 of issue #82: adopt MALA in whitened coordinates if a step is at
least 15% faster at 50 dimensions or more; otherwise close the item with the figure.

Falsification test (fixed before any run). The whitened form is not
worth adopting if, at every one of d = 50, 100 and 200, the median over rounds of
(whitened step time / current step time) for the coupled step, on a vectorized target
with 64 lanes, is above 0.85, or if its round-to-round range includes 0.85. It is worth
adopting if at some d >= 50 the whole range of that ratio lies below 0.85. The single
step is reported the same way, as secondary evidence.

Design. `MALA` at this commit and `WhitenedMALA` (wm_kernel.py, checked to be the same
kernel in wm_same_kernel.py), both with cov = Sigma and step size 1.65 d^(-1/6), called
directly (`kernel.step`, `kernel.coupled_step`), not through the sampler, at d = 50, 100
and 200 with 64 lanes. Targets, vectorized (so the kernels' products are BLAS calls):
  - `dense` (primary, fixed before the run): N(0, Sigma), Sigma_ij = 0.9^|i-j|
    (`examples.gaussian(d)`), with `log_prob_and_grad` computing the gradient as one
    dense BLAS product with the precision matrix, x @ P, and the log-density from it:
    a target whose own cost is a product of the size of the kernel's;
  - `examples` (secondary): `examples.gaussian(d)`'s own `log_prob` and `grad`, whose
    products are `np.einsum` without optimization, slower than a BLAS product, so the
    kernel's products are a smaller share of the step.
Single steps start from 64 fixed states drawn from the target; coupled steps from 64
fixed pairs of independent draws from the target, which never meet at d >= 50
(wm_same_kernel.out, B), so every step evaluates 128 proposals: the cost of a pair before
it meets. Each call is from the same fixed states (the returned states are discarded);
the draws come from `testing.random_draws(64, seed)`, one object per form, so both forms
read the same amount. For each (target, d, step) configuration: one uncounted round of
calibration fixes, for each form, the number of calls that takes at least 0.6 s; then 9
rounds, each timing one block of calls of each form, the forms in the order AB in even
rounds and BA in odd ones; garbage collection is off during a block. A block's wall time
(`perf_counter`) over its calls is the step time; the ratio of a round is whitened /
current. Reported: each round's ratio, the median and the range; the processor-time
ratio's median as a check. Last, for context, the time of one product of 64 rows with a
d x d matrix (`x @ M.T`, the kernels' BLAS product), the minimum of 9 blocks of 0.3 s.

The machine, commit, threads and libraries are printed first, and the load before and
after. If the 1-minute load is above 1 before timing, the script waits 30 s and checks
again, up to 10 minutes.

Result. Run of record `wm_timing.out` (commit 9932df7, 1-minute load 0.25 before and
0.92 after, the latter this process's own; no other Python process), and its
replication `wm_timing_run2.out` (load 0.29 before), ratio whitened / current, median
(min to max) over 9 rounds, dense target:
  coupled  d =  50: 0.881 (0.857 to 1.147); replication 0.900 (0.792 to 1.222)
  coupled  d = 100: 0.800 (0.774 to 0.861); replication 0.810 (0.632 to 0.914)
  coupled  d = 200: 0.917 (0.845 to 0.974); replication 0.944 (0.782 to 1.086)
  single   d =  50: 0.846 (0.820 to 0.951); replication 0.813 (0.653 to 0.897)
  single   d = 100: 0.755 (0.716 to 0.763); replication 0.751 (0.704 to 0.882)
  single   d = 200: 0.658 (0.625 to 0.671); replication 0.655 (0.616 to 0.753)
On the examples target, coupled 0.952, 0.888, 0.873 and single 0.877, 0.871, 0.842 at
d = 50, 100, 200 (replication 0.921, 0.869, 0.836 and 0.881, 0.868, 0.735). A BLAS
product of 64 rows took 8.60, 26.81 and 86.41 us at d = 50, 100 and 200, so removing 7
products from the coupled step predicts 1 - 7 x 8.60 / 397.4 = 0.849, 1 - 7 x 26.81 /
749.5 = 0.750 and 1 - 7 x 86.41 / 2111.0 = 0.713, and 4 from the single step 0.831,
0.713 and 0.673. The single step and the coupled step at d = 50 and 100 are within 0.05 of the
prediction; the coupled step at d = 200 saved 187 and 159 us where 7 products cost about
600 us, in both runs, unexplained (see wm_timing_fine.py). By the falsification test:
at d = 50 and 200 the median is above 0.85, and at d = 100 the range includes 0.85 (to
0.861; 0.914 in the replication), so the run of record meets the condition "not worth
adopting", as does the replication.

Run from the repository root:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:<this folder> \
    python <this folder>/wm_timing.py
"""

import gc
import os
import platform
import subprocess
import time
import warnings

import numpy as np
import scipy
import threadpoolctl
from wm_kernel import WhitenedMALA

from telescopic import Target, examples, kernels
from telescopic.testing import random_draws

warnings.simplefilter("ignore")
LANES = 64
ROUNDS = 9
BLOCK = 0.6


def describe_machine():
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    with open("/proc/cpuinfo") as cpuinfo:
        model = next(line.split(":", 1)[1].strip() for line in cpuinfo if "model name" in line)
    print(f"commit {commit} ({'clean' if not status else 'modified tracked files'})")
    print(f"cpu: {model}; {os.cpu_count()} logical cpus; {platform.platform()}")
    print(f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
    threads = {
        k: os.environ.get(k)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    print(f"thread settings: {threads}")
    for pool in threadpoolctl.threadpool_info():
        print(
            f"threadpool: {pool['internal_api']} {pool.get('version')} "
            f"({pool.get('architecture')}), {pool['num_threads']} threads"
        )


def load():
    with open("/proc/loadavg") as file:
        return file.read().split()[:3]


def other_pythons():
    # Processes whose program is a Python interpreter, other than this one.
    ps = subprocess.run(
        ["ps", "-eo", "pid,pcpu,comm"], capture_output=True, text=True, check=False
    ).stdout
    mine = str(os.getpid())
    return [
        line.strip()
        for line in ps.splitlines()[1:]
        if line.split()[2].startswith("python") and line.split()[0] != mine
    ]


def wait_for_idle():
    for _ in range(21):
        averages = load()
        if float(averages[0]) <= 1.0:
            break
        print(f"load {averages}: waiting 30 s")
        time.sleep(30)
    print(f"load before timing (1, 5, 15 min): {averages}")
    print(f"other python processes: {other_pythons() or 'none'}")


def dense_target(d):
    precision = np.linalg.inv(examples.gaussian(d).cov)

    def value_and_grad(x):
        grad = -(x @ precision)
        return 0.5 * np.sum(x * grad, axis=1), grad

    return Target(lambda x: value_and_grad(x)[0], log_prob_and_grad=value_and_grad, vectorize=True)


def examples_target(d):
    example = examples.gaussian(d)
    return Target(example.log_prob, example.grad, vectorize=True)


def calls(kernel, target, kind, d, seed):
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(examples.gaussian(d).cov)
    draws = random_draws(LANES, seed + 1)
    state_x = kernel.init_state(target, rng.standard_normal((LANES, d)) @ chol.T)
    if kind == "single":
        return lambda: kernel.step(target, state_x, draws)
    state_y = kernel.init_state(target, rng.standard_normal((LANES, d)) @ chol.T)
    return lambda: kernel.coupled_step(target, state_x, state_y, draws)


def block(call, number):
    gc.disable()
    try:
        wall, cpu = time.perf_counter(), time.process_time()
        for _ in range(number):
            call()
        return (time.perf_counter() - wall) / number, (time.process_time() - cpu) / number
    finally:
        gc.enable()


def time_configuration(make_target, d, kind):
    step_size = 1.65 * d ** (-1 / 6)
    cov = examples.gaussian(d).cov
    target = make_target(d)
    forms = [
        calls(kernels.MALA(step_size, cov=cov), target, kind, d, seed=7000 + d),
        calls(WhitenedMALA(step_size, cov=cov), target, kind, d, seed=7000 + d),
    ]
    numbers = []
    for call in forms:
        per_call, _ = block(call, 20)
        numbers.append(max(20, int(np.ceil(BLOCK / per_call))))
    walls, cpus = [], []
    for r in range(ROUNDS):
        order = (0, 1) if r % 2 == 0 else (1, 0)
        times = {}
        for form in order:
            times[form] = block(forms[form], numbers[form])
        walls.append((times[0][0], times[1][0]))
        cpus.append((times[0][1], times[1][1]))
    return np.array(walls), np.array(cpus), numbers


def product_time(d):
    rng = np.random.default_rng(d)
    matrix, x = rng.standard_normal((d, d)), rng.standard_normal((LANES, d))
    call = lambda: x @ matrix.T
    per_call, _ = block(call, 100)
    number = int(np.ceil(0.3 / per_call))
    return min(block(call, number)[0] for _ in range(9))


def main():
    describe_machine()
    wait_for_idle()
    print(
        f"\n{'target':>8} {'step':>7} {'d':>4} {'calls cur,wh':>13} {'cur us':>8} {'wh us':>8} "
        f"{'median':>7} {'min':>6} {'max':>6} {'cpu med':>7}  ratios by round"
    )
    for name, make_target in (("dense", dense_target), ("examples", examples_target)):
        for kind in ("coupled", "single"):
            for d in (50, 100, 200):
                walls, cpus, numbers = time_configuration(make_target, d, kind)
                ratios = walls[:, 1] / walls[:, 0]
                cpu_ratio = np.median(cpus[:, 1] / cpus[:, 0])
                print(
                    f"{name:>8} {kind:>7} {d:>4} {numbers[0]:>6},{numbers[1]:<6} "
                    f"{1e6 * np.median(walls[:, 0]):>8.1f} {1e6 * np.median(walls[:, 1]):>8.1f} "
                    f"{np.median(ratios):>7.3f} {ratios.min():>6.3f} {ratios.max():>6.3f} "
                    f"{cpu_ratio:>7.3f}  {' '.join(f'{v:.3f}' for v in ratios)}",
                    flush=True,
                )
    print("\nOne BLAS product of 64 rows with a d x d matrix (min of 9 blocks of 0.3 s):")
    for d in (50, 100, 200):
        print(f"d = {d}: {1e6 * product_time(d):.2f} us")
    print(f"\nload after timing (1, 5, 15 min): {load()}")


if __name__ == "__main__":
    main()

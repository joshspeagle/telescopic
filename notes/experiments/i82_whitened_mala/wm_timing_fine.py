"""The ratios of wm_timing.py with finer interleaving: a sensitivity analysis, after the fact.

Question. In wm_timing.py's two runs the coupled step's ratio at d = 200 was 0.917 and
0.944, where the count of products predicts about 0.71, and the ranges over rounds were
wide (to 1.147 at d = 50). On this machine the time of one BLAS product (64 x 200 times
200 x 200) drifts by about 9% between blocks of 0.44 s, processor time tracking wall
time, with almost no steal (wm_noise.out). Blocks of 0.6 s of one form, then of the
other, put that drift into the ratio. This design, chosen after seeing those results and so not the
test of record, interleaves finely so that drift slower than a sub-block cancels.

Design. As wm_timing.py (the same kernels, targets, states, draws and calibration),
except that a round is 12 alternations of a sub-block of each form, each sub-block
about 0.05 s (so each form gets about 0.6 s a round), the order AB in each alternation of
even rounds and BA in odd ones; a round's step time of a form is the total time of its
sub-blocks over their calls. 9 rounds. Reported as in wm_timing.py, plus the ratio that
the count of products predicts: 1 - (products saved) x (time of one product) / (current
step time), with 7 products saved by a coupled step and 4 by a single one, and the time
of one product measured at the end, as in wm_timing.py.

Result (`wm_timing_fine.out`, two runs, commit 9932df7, 1-minute load 0.73 and 0.98
before them, the latter left by the first run). Ratio whitened / current, median (min to
max) over 9 rounds, dense target, run 1 then run 2:
  coupled  d =  50: 0.893 (0.839 to 0.920); 0.884 (0.841 to 0.908)
  coupled  d = 100: 0.815 (0.791 to 0.869); 0.813 (0.799 to 0.844)
  coupled  d = 200: 0.902 (0.850 to 0.955); 0.768 (0.730 to 0.788)
  single   d =  50: 0.843 (0.802 to 0.876); 0.820 (0.807 to 0.878)
  single   d = 100: 0.752 (0.742 to 0.798); 0.738 (0.720 to 0.776)
  single   d = 200: 0.660 (0.648 to 0.673); 0.667 (0.643 to 0.680)
Predicted by the count of products: coupled 0.868, 0.771, 0.737 and 0.838, 0.756,
0.776; single 0.842, 0.736, 0.685 and 0.842, 0.753, 0.698. On the examples target,
coupled 0.924, 0.896, 0.847 and 0.920, 0.883, 0.859; single 0.874, 0.864, 0.829 and
0.865, 0.857, 0.829. The finer interleaving narrowed the ranges within a run (to about
+-0.05) but not the difference between runs: the coupled step at d = 200 gave 0.902 in
one process and 0.768 in the next, with the current form's median step time 2,255 us in
the first and 2,723 us in the second, and the whitened form's 1,994 and 2,119 us. The
time of a step at d = 200 thus differs between processes by up to 20%, and each form by
a different amount; the cause was not found. By the falsification test, run 1 meets
"not worth adopting" (d = 100's range reaches 0.869) and run 2 meets "worth adopting"
(d = 100: 0.799 to 0.844; d = 200: 0.730 to 0.788).

Run from the repository root:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:<this folder> \
    python <this folder>/wm_timing_fine.py
"""

import time

import numpy as np
from wm_kernel import WhitenedMALA
from wm_timing import (
    block,
    calls,
    dense_target,
    describe_machine,
    examples_target,
    load,
    product_time,
    wait_for_idle,
)

from telescopic import examples, kernels

ROUNDS = 9
ALTERNATIONS = 12
SUB_BLOCK = 0.05
SAVED = {"coupled": 7, "single": 4}


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
        numbers.append(max(2, int(np.ceil(SUB_BLOCK / per_call))))
    walls = []
    for r in range(ROUNDS):
        order = (0, 1) if r % 2 == 0 else (1, 0)
        totals = [0.0, 0.0]
        for _ in range(ALTERNATIONS):
            for form in order:
                totals[form] += block(forms[form], numbers[form])[0] * numbers[form]
        walls.append([totals[f] / (ALTERNATIONS * numbers[f]) for f in (0, 1)])
    return np.array(walls), numbers


def main():
    describe_machine()
    wait_for_idle()
    started = time.perf_counter()
    rows = []
    print(
        f"\n{'target':>8} {'step':>7} {'d':>4} {'calls cur,wh':>13} {'cur us':>8} {'wh us':>8} "
        f"{'median':>7} {'min':>6} {'max':>6}  ratios by round"
    )
    for name, make_target in (("dense", dense_target), ("examples", examples_target)):
        for kind in ("coupled", "single"):
            for d in (50, 100, 200):
                walls, numbers = time_configuration(make_target, d, kind)
                ratios = walls[:, 1] / walls[:, 0]
                rows.append((name, kind, d, np.median(walls[:, 0])))
                print(
                    f"{name:>8} {kind:>7} {d:>4} {numbers[0]:>6},{numbers[1]:<6} "
                    f"{1e6 * np.median(walls[:, 0]):>8.1f} {1e6 * np.median(walls[:, 1]):>8.1f} "
                    f"{np.median(ratios):>7.3f} {ratios.min():>6.3f} {ratios.max():>6.3f}  "
                    f"{' '.join(f'{v:.3f}' for v in ratios)}",
                    flush=True,
                )
    print(f"timing took {time.perf_counter() - started:.0f} s")
    products = {d: product_time(d) for d in (50, 100, 200)}
    print("\nOne BLAS product of 64 rows with a d x d matrix (min of 9 blocks of 0.3 s):")
    print("  " + ", ".join(f"d = {d}: {1e6 * t:.2f} us" for d, t in products.items()))
    print("Ratio predicted by the count of products, 1 - saved x product / current step:")
    for name, kind, d, current in rows:
        predicted = 1 - SAVED[kind] * products[d] / current
        print(f"  {name:>8} {kind:>7} d = {d:>3}: {predicted:.3f}")
    print(f"\nload after timing (1, 5, 15 min): {load()}")


if __name__ == "__main__":
    main()

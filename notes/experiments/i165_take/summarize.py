"""Medians over runs of `measure.py`, before and after a change (issue #165).

For each workload: the median over the runs of the plain time per step, before and after;
the ratio of these medians, and the lowest and highest ratio of the runs paired in the
order given (run i after over run i before, run in alternation); and the fixed time in
requests per step, the time in requests less the generation of the numbers they return,
at one cost per normal and per uniform, the median over all the runs of both trees,
with its share of the plain time per step. Run as `python
notes/experiments/i165_take/summarize.py B1,B2,... A1,A2,...` with the outputs of
`measure.py` before (B) and after (A).
"""

import sys
from pathlib import Path

import numpy as np


def parse(path):
    # The generation costs in us, and for each workload its plain time per step, its
    # normals and uniforms per step and its time in requests per step, in us.
    lines = Path(path).read_text().splitlines()
    fields = lines[1].split()
    generation = 1e-3 * np.array([float(fields[1]), float(fields[5])])
    rows = {}
    for line in lines[2:]:
        name, lanes, _, per_step, _, normals, uniforms, in_requests = line.split()
        rows[name, int(lanes)] = (
            float(per_step),
            np.array([float(normals), float(uniforms)]),
            float(in_requests),
        )
    return generation, rows


def main(before, after):
    runs = [[parse(path) for path in group.split(",")] for group in (before, after)]
    generation = np.median([g for group in runs for g, _ in group], axis=0)
    print(
        f"generation, median of all runs: {1e3 * generation[0]:.2f} ns per normal, "
        f"{1e3 * generation[1]:.2f} ns per uniform"
    )
    print(
        f"{'workload':20} {'lanes':>5} {'before':>8} {'after':>8} {'ratio':>6} "
        f"{'runs':>11} {'fixed':>7} {'share':>6} {'fixed':>7} {'share':>6}"
    )
    print(
        f"{'':20} {'':>5} {'us/step':>8} {'us/step':>8} {'':>6} {'':>11} {'before':>14} {'after':>14}"
    )
    for key in runs[0][0][1]:
        cells = []
        steps = []
        for group in runs:
            per_step = np.array([rows[key][0] for _, rows in group])
            fixed = [rows[key][2] - rows[key][1] @ generation for _, rows in group]
            steps.append(per_step)
            cells.append((np.median(per_step), np.median(fixed)))
        paired = steps[1] / steps[0]
        ratio = cells[1][0] / cells[0][0]
        print(
            f"{key[0]:20} {key[1]:5d} {cells[0][0]:8.1f} {cells[1][0]:8.1f} {ratio:6.3f} "
            f"{paired.min():5.3f}-{paired.max():5.3f} "
            f"{cells[0][1]:7.2f} {cells[0][1] / cells[0][0]:6.1%} "
            f"{cells[1][1]:7.2f} {cells[1][1] / cells[1][0]:6.1%}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])

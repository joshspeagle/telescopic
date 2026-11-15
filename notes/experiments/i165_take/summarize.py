"""Medians over runs of `measure.py`, before and after a change (issue #165).

For each workload, the median over the runs of each run's time per step, and of its
fixed time in `_take` per step, before and after, and the ratio of the medians of the
times per step. Run as `python notes/experiments/i165_take/summarize.py B1,B2,... A1,A2,...`
with the outputs of `measure.py` before (B) and after (A).
"""

import re
import sys
from pathlib import Path

import numpy as np

PATTERN = re.compile(
    r"^(.+?)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.-]+)\s+([\d.]+)%"
)


def parse(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        m = PATTERN.match(line)
        if m:
            rows[(m.group(1).strip(), int(m.group(2)))] = (
                float(m.group(4)),
                float(m.group(6)),
                float(m.group(8)),
            )
    return rows


def main(before, after):
    data = [[parse(f) for f in group.split(",")] for group in (before, after)]
    print(
        f"{'workload':20} {'lanes':>5} | before: {'us/step':>7} {'fixed':>6} {'share':>6} "
        f"| after: {'us/step':>7} {'fixed':>6} {'share':>6} | after/before"
    )
    for key in data[0][0]:
        med = [np.median([r[key] for r in group], axis=0) for group in data]
        cells = " | ".join(
            f"{'before' if i == 0 else 'after':6}: {m[0]:7.1f} {m[2]:6.2f} {m[2] / m[0]:6.1%}"
            for i, m in enumerate(med)
        )
        print(f"{key[0]:20} {key[1]:5d} | {cells} | {med[1][0] / med[0][0]:.3f}")


main(*sys.argv[1:])

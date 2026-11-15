"""The runs of the budget-mode experiment (PREREGISTRATION.md, sections 3 and 4).

Pilots first, then the 18 cells (3 targets, 3 budgets, 2 arms), on 4 single-threaded
processes. Each cell's records go to runs/<target>-<arm>-b<b>.npz; run.out holds the
setup and the timings.

Run: PYTHONPATH=src python notes/experiments/i79_budget/run.py > notes/experiments/i79_budget/run.out
"""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import json
import multiprocessing
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from budget import (
    BUDGETS,
    CASES,
    LANES,
    PILOT,
    budget_cell,
    fixed_cell,
    pilot_costs,
    replicates,
    seed,
)

import telescopic

OUT = HERE / "runs"


def pilot(name):
    start = time.perf_counter()
    case = CASES[name]
    _, costs, _ = replicates(case, case.target(), np.arange(PILOT), seed(name, "pilot"))
    mean = float(costs.mean())
    plan = {}
    for b in BUDGETS:
        budget = round(b * mean)
        charged, lanes = pilot_costs(costs, budget)
        plan[b] = {
            "budget": budget,
            "size": round(LANES * charged / mean),
            "lane_cost": charged,
            "pilot_lanes": lanes,
        }
    return name, mean, plan, time.perf_counter() - start


def cell(job):
    name, arm, b, value = job
    start = time.perf_counter()
    case = CASES[name]
    rng = seed(name, arm, BUDGETS.index(b))
    if arm == "budget":
        records = budget_cell(case, value, case.runs, rng)
    else:
        records = fixed_cell(case, value, case.runs, rng)
    np.savez_compressed(OUT / f"{name}-{arm}-b{b}.npz", **records)
    return name, arm, b, value, int(records["computed"]), time.perf_counter() - start


def main():
    OUT.mkdir(exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=HERE,
    ).stdout.strip()
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    print(
        f"commit {commit}; {platform.system()} {platform.machine()}, {cpu}, "
        f"{os.cpu_count()} logical processors; 4 processes; OMP, OpenBLAS and MKL threads 1"
    )
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, "
        f"telescopic {telescopic.__version__}"
    )
    start = time.perf_counter()
    plans = {}
    with multiprocessing.Pool(4) as pool:
        for name, mean, plan, seconds in pool.imap_unordered(pilot, CASES):
            plans[name] = {"mean_cost": mean, "plan": plan}
            print(
                f"pilot {name}: {PILOT} replicates, mean cost {mean:.2f} kernel steps, "
                f"{seconds:.0f} s; "
                + "; ".join(
                    f"b={b}: B={p['budget']}, lane cost {p['lane_cost']:.1f} over "
                    f"{p['pilot_lanes']} pilot lanes, R={p['size']}"
                    for b, p in plan.items()
                ),
                flush=True,
            )
        (OUT / "plans.json").write_text(json.dumps(plans, indent=1))
        jobs = [
            (name, arm, b, plans[name]["plan"][b]["budget" if arm == "budget" else "size"])
            for name in CASES
            for b in BUDGETS
            for arm in ("budget", "fixed")
        ]
        # Longest first: the banana's replicates cost the most, and larger b more.
        jobs.sort(key=lambda job: (job[0] != "banana", -job[2]))
        processor = 0.0
        for name, arm, b, value, computed, seconds in pool.imap_unordered(cell, jobs):
            processor += seconds
            what = f"B={value}" if arm == "budget" else f"R={value}"
            print(
                f"{name} {arm} b={b} ({what}): {CASES[name].runs} runs, {computed} replicates "
                f"computed, {seconds:.0f} s",
                flush=True,
            )
    print(
        f"cells: {processor / 3600:.2f} processor hours; wall {time.perf_counter() - start:.0f} s"
    )


if __name__ == "__main__":
    main()

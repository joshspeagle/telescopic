"""The runner of x91: every record of PREREGISTRATION.md, in the order and with the cuts of section 11.

    python notes/experiments/x91/run.py [--workers 4]
    python notes/experiments/x91/run.py --confirm [--workers 4]

Cells run in the order of `cells.CELLS` (A, A', B, B', T, C1 to C5, D3, D2, D1, then E,
then F), each whole: its records run on `--workers` single-threaded processes, and each
record is appended to its part's store (`notes/experiments/_rows/x91/<part>.jsonl`, one
JSON line per record) as soon as it is done, so that a rerun skips the recorded ones and
resumes. Once the recorded processor seconds of every store sum to 6 hours, no new cell
of tiers 2 and 3 starts, and each such cell gets a record that says it was cut.

Within a cell, G runs after the other arms: where its guard keeps the tuned step, its
record is a copy of P's at that seed (`same_as`), computed then if P has no record there
(section 5). `--confirm` runs the fresh seeds that the scorer asks for to confirm a
z-score beyond 3.5 (section 7, Z), into the store `Z.jsonl`.

The runner refuses a checkout with uncommitted changes to tracked files, and stores
whose records come from another commit. `--offset N` adds N to every seed and writes to
`_rows/x91_offset<N>/`, for trying the runner on seeds the registration does not use;
only then are `--only` and `--max-seeds` allowed.
"""

import os

# Before NumPy is imported here or in a worker, which inherit the environment.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import json
import math
import multiprocessing
import re
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import numpy as np
from arms import HMC_ARMS, kernel, settings
from cells import CELL, CELLS, GAUSSIANS, K_MAX, PAIRS, REPLICATE_CAP, REPLICATES, SEEDS, target
from i85_setup import setup
from i137_common import factor, jitter_factor, pilot_frequencies

from telescopic import CoupledSampler
from telescopic.tuning import choose_k

ROWS = Path(__file__).resolve().parents[1] / "_rows"
# Recorded processor seconds of every store (all tiers and confirmations) after which no
# new cell of tiers 2 and 3 starts.
BUDGET = 6 * 3600


@cache
def _setup():
    return setup()


def commit(record):
    return re.search(r"commit (\S+) of the package", record["setup"])[1]


def key(cell, arm, seed, confirm=False):
    return f"{'Z|' if confirm else ''}{cell}|{arm}|{seed}"


def _quantile(tau, q):
    return float(np.quantile(tau, q, method="inverted_cdf"))


def measure(cell_name, arm, seed, confirm=False, p_record=None):
    """The records of one arm at one seed: tuning, the pilot, and the replicates (section 6).

    For G, where its guard keeps the tuned step, its record is a copy of P's at that seed,
    `p_record`, computed here when it is None; the list then holds P's record too.
    """
    warnings.simplefilter("ignore")
    start, wall = time.process_time(), time.time()
    cell = CELL[cell_name]
    example, initial, cov, h, truth = target(cell.target)
    sampler = CoupledSampler(
        example.log_prob,
        initial,
        grad=example.grad,
        vectorize=True,
        kernel=kernel(arm, cell.n_steps, cov),
        rng=seed,
    )
    counts = sampler.target
    sampler.tune()
    record = {
        "key": key(cell_name, arm, seed, confirm),
        "part": "Z" if confirm else cell.part,
        "cell": cell_name,
        "arm": arm,
        "seed": seed,
        "setup": _setup(),
        **settings(sampler.kernel),
        "tune_grad": counts.n_grad,
        "tune_log_prob": counts.n_log_prob,
    }
    if arm == "G" and cell.run != "tune" and record["guard"]["action"] != "moved":
        return _copy(cell_name, seed, confirm, p_record, record["guard"], start)
    if arm in HMC_ARMS and cell.target.startswith(GAUSSIANS):
        # The exact whitened frequencies sqrt eig(C^T Sigma^-1 C), the largest factor
        # |cos n theta| (its geometric mean over the jitter for a jittered arm), and the
        # frequencies predicted from the pilot.
        chol = np.linalg.cholesky(sampler.kernel.cov)
        w = np.sqrt(np.linalg.eigvalsh(chol.T @ np.linalg.solve(example.cov, chol)))
        if record["jitter"]:
            f = jitter_factor(record["eps"], record["n"], w, record["jitter"])
        else:
            f = factor(record["eps"], record["n"], w)[0]
        record |= {
            "w": w.tolist(),
            "factor": float(f.max()),
            "w_pred": pilot_frequencies(chol, sampler.tuning.cov).tolist(),
        }
    if cell.run != "tune":
        _pilot(sampler, cell, arm, h, truth, record)
    record["seconds"] = round(time.process_time() - start, 3)
    record["wall"] = round(time.time() - wall, 3)
    return [record]


def _copy(cell_name, seed, confirm, p_record, guard, start):
    # G kept the tuned step, so its kernel and draws are P's: its record is P's.
    seconds = round(time.process_time() - start, 3)
    out = []
    if p_record is None:
        (p_record,) = measure(cell_name, "P", seed, confirm)
        p_record["for_copy"] = True  # beyond P's own seeds: computed for G's copy
        out.append(p_record)
    if p_record["eps"] != guard["eps0"]:
        raise RuntimeError(f"G's tuning at seed {seed} of {cell_name} differs from P's")
    copy = dict(p_record, arm="G", key=key(cell_name, "G", seed, confirm), guard=guard)
    copy |= {"same_as": p_record["key"], "seconds": seconds, "for_copy": False}
    return [*out, copy]


def _pilot(sampler, cell, arm, h, truth, record):
    counts = sampler.target
    grad0, log_prob0 = counts.n_grad, counts.n_log_prob
    tau = sampler.meeting_times(PAIRS, max_iter=cell.cap)
    try:
        k, error = choose_k(tau), None
    except ValueError as e:
        k, error = None, str(e).split("\n")[0]
    pilot_grad = counts.n_grad - grad0
    record |= {
        "cap": cell.cap,
        "tau": tau.tolist(),
        "unmet": int(np.isinf(tau).sum()),
        "median": float(np.median(tau)),
        "q50": _quantile(tau, 0.5),
        "q90": _quantile(tau, 0.9),
        "q99": _quantile(tau, 0.99),
        "k": k,
        "error": error,
        "pilot_grad": pilot_grad,
        "pilot_log_prob": counts.n_log_prob - log_prob0,
        # Gradients per pair-iteration, unmet pairs counted at the cap.
        "g": pilot_grad / float(np.where(np.isinf(tau), cell.cap, tau).sum()),
    }
    if cell.run != "full" or arm in cell.pilot_arms:
        return
    if record["unmet"]:
        record["replicates"] = "not run: a pilot pair unmet at the cap"
        return
    if k > K_MAX:  # no pair unmet, so k was chosen
        record["replicates"] = f"not run: k = {k} is above {K_MAX}"
        return
    result = sampler.run(REPLICATES, k=k, lag=k, m=10 * k, max_iter=REPLICATE_CAP)
    record |= {
        "m": 10 * k,
        "rep_grad": result.n_grad,
        "rep_log_prob": result.n_log_prob,
        "n_met": result.n_replicates,
    }
    if result.n_censored:
        # A censored replicate would bias the estimates (section 4): none is kept.
        record["replicates"] = (
            f"discarded: {result.n_censored} replicates unmet at {REPLICATE_CAP} iterations"
        )
        return
    estimate = result.estimate(h)
    value, se = np.atleast_1d(estimate.value), np.atleast_1d(estimate.standard_error)
    record |= {
        "replicates": "ran",
        "value": value.tolist(),
        "se": se.tolist(),
        "truth": None if truth is None else list(truth),
        "z": None if truth is None else ((value - truth) / se).tolist(),
        "inefficiency": (se**2 * result.n_grad).tolist(),
    }


class Stores:
    """The append-only stores of one run, by part, and the records already in them."""

    def __init__(self, folder):
        self.folder = folder
        self.records = {}
        for path in sorted(folder.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                self.records[record["key"]] = record

    def add(self, record):
        self.folder.mkdir(parents=True, exist_ok=True)
        name = record["part"].replace("'", "p")
        with open(self.folder / f"{name}.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        self.records[record["key"]] = record

    def seconds(self):
        return sum(r.get("seconds", 0.0) for r in self.records.values())


def _tasks(cell, seeds_of, stores, confirm, arms):
    first, guards = [], []
    for arm in arms:
        for seed in seeds_of(arm):
            if key(cell.name, arm, seed, confirm) in stores.records:
                continue
            if arm == "G":
                guards.append(seed)
            else:
                first.append((measure, cell.name, arm, seed, confirm))
    return first, guards


def run_cell(cell, seeds_of, stores, pool, confirm=False, arms=None):
    """Every missing record of a cell: the arms other than G, then G."""
    first, guards = _tasks(cell, seeds_of, stores, confirm, arms or cell.arms)
    _gather(pool, first, stores)
    later = []
    for seed in guards:
        p = stores.records.get(key(cell.name, "P", seed, confirm))
        later.append((measure, cell.name, "G", seed, confirm, p))
    _gather(pool, later, stores)


def _gather(pool, tasks, stores):
    futures = [pool.submit(*task) for task in tasks]
    for future in as_completed(futures):
        for record in future.result():
            stores.add(record)
            print(
                f"{record['key']}: {record['seconds']:.1f} s"
                + (
                    f", k={record.get('k')}, unmet={record.get('unmet')}"
                    if "tau" in record
                    else ""
                ),
                flush=True,
            )


def uncommitted():
    """The changed or untracked files of the package and the experiments, as git lists them.

    The setup's `git describe --dirty` misses untracked files, such as an uncommitted runner.
    """
    return subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "notes/experiments"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[3],
    ).stdout


def _check_commit(stores, offset):
    here = commit({"setup": _setup()})
    changed = uncommitted()
    if changed and not offset:
        sys.exit(f"the checkout has uncommitted changes, commit them before a run:\n{changed}")
    others = {commit(r) for r in stores.records.values() if "setup" in r} - {here}
    if others:
        sys.exit(f"the stores hold records of commits {sorted(others)}, not {here}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--max-seeds", type=int)
    args = parser.parse_args()
    if (args.only or args.max_seeds) and not args.offset:
        parser.error("--only and --max-seeds need a nonzero --offset")
    folder = ROWS / ("x91" if not args.offset else f"x91_offset{args.offset}")
    stores = Stores(folder)
    _check_commit(stores, args.offset)
    print(_setup(), flush=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        if args.confirm:
            from score import confirmations

            # P's confirmation first, on a cell where P's own z needs one: it decides
            # whether the arms there are judged by their difference with P.
            first = SEEDS["confirmation"] + args.offset
            while todo := confirmations(stores.records, args.offset):
                for cell_name, arm, n in todo:

                    def seeds_of(_, n=n):
                        return range(first, first + n)

                    run_cell(CELL[cell_name], seeds_of, stores, pool, confirm=True, arms=[arm])
            return
        for cell in CELLS:
            if args.only and cell.name not in args.only:
                continue
            if cell.tier > 1 and f"{cell.name}|cut" not in stores.records:
                done = any(r.get("cell") == cell.name for r in stores.records.values())
                if not done and stores.seconds() >= BUDGET:
                    stores.add({"key": f"{cell.name}|cut", "part": cell.part, "cut": True})
                    print(f"{cell.name}: cut, {stores.seconds():.0f} s recorded", flush=True)
            if f"{cell.name}|cut" in stores.records:
                continue

            def seeds_of(arm, cell=cell):
                n = min(cell.arms[arm], args.max_seeds or math.inf)
                return [s + args.offset for s in cell.seeds[:n]]

            run_cell(cell, seeds_of, stores, pool)
            print(f"{cell.name}: done; {stores.seconds():.0f} s recorded in all", flush=True)


if __name__ == "__main__":
    main()

"""Run the coupled pair that the landing page's hero draws, and save its coordinates.

The pair is one replicate of :func:`telescopic.engine.run` on the standard normal target
in one dimension, with random-walk Metropolis proposals of standard deviation 0.5 coupled
by the reflection-maximal coupling, chains started from N(0, 4^2) and the lag 1. The
replicate drawn is the first, by index, whose X starts at least one standard deviation
above the target's mean and whose Y starts at least one below it, so that the two chains
enter the picture from opposite sides. ``python docs/_scripts/hero_pair.py`` writes
``hero_pair.json`` beside this script, with the seed and the last commit that changed the
package; ``brand.py`` draws the hero from it. ``--check`` runs the pair again and fails if
the saved coordinates are no longer what the package produces; CI runs it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import telescopic
from telescopic import engine

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).with_name("hero_pair.json")
SEED = 0
CANDIDATES = 64  # replicates searched for the first that starts on opposite sides
SCALE = 0.5
INITIAL_SD = 4.0
LAG = 1
# The hero draws the time axis from 0 to this multiple of the meeting time (the mark's
# focus sits at 164/230 of the width the chains cross), so X must run that long.
SPAN = 230 / 164
DECIMALS = 4


def log_prob(x: np.ndarray) -> np.ndarray:
    return -0.5 * x[:, 0] ** 2


def pair() -> dict[str, object]:
    target = telescopic.Target(log_prob, vectorize=True)
    kernel = telescopic.kernels.RandomWalk(cov=[[1.0]], scale=SCALE)
    initial = telescopic.initial.Normal(mean=[0.0], cov=[INITIAL_SD**2])
    runs = engine.run(
        target, kernel, initial, range(CANDIDATES), k=0, m=400, lag=LAG, rng=SEED, keep_paths=True
    )
    chosen = next(
        i for i in runs.index if runs.paths[i][0][0, 0] >= 1 and runs.paths[i][1][0, 0] <= -1
    )
    x, y = (path[:, 0] for path in runs.paths[chosen])
    tau = int(runs.meeting_time[chosen])
    end = int(np.ceil(SPAN * tau))
    if end >= len(x) or x[tau] != y[tau - LAG]:
        raise SystemExit("the chosen pair is too long for m, or its chains did not meet")
    return {
        "replicate": int(chosen),
        "meeting_time": tau,
        "x": np.round(x[: end + 1], DECIMALS).tolist(),
        "y": np.round(y, DECIMALS).tolist(),
    }


def package_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    if git("status", "--porcelain", "--", "src"):
        raise SystemExit("src/ has uncommitted changes; commit them before saving the pair.")
    return git("log", "-1", "--format=%H", "--", "src")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the saved pair is stale")
    args = parser.parse_args(argv)
    fresh = pair()
    if args.check:
        saved = json.loads(DATA.read_text(encoding="utf-8"))
        same = (
            saved["replicate"] == fresh["replicate"]
            and saved["meeting_time"] == fresh["meeting_time"]
            and all(
                np.shape(saved[c]) == np.shape(fresh[c])
                and np.allclose(saved[c], fresh[c], atol=10.0**-DECIMALS)
                for c in ("x", "y")
            )
        )
        if not same:
            print(f"{DATA.relative_to(ROOT)} is stale (run docs/_scripts/hero_pair.py)")
        return 0 if same else 1
    record = {
        "generated_by": "docs/_scripts/hero_pair.py",
        "package_commit": package_commit(),
        "seed": SEED,
        "target": "N(0, 1)",
        "kernel": f"RandomWalk(cov=[[1.0]], scale={SCALE}), reflection-maximal coupling",
        "initial": f"N(0, {INITIAL_SD:g}^2)",
        "lag": LAG,
        "rule": "the first replicate whose X_0 >= 1 and Y_0 <= -1",
        **fresh,
        "x_is": "X_t for t = 0, 1, ...",
        "y_is": "Y_t for t = 0, ..., meeting_time - lag",
    }
    DATA.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

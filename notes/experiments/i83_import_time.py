"""The processor time of `import telescopic` with SciPy's stats, optimize and integrate imported at first use (issue #83).

Each worker of a process pool imports telescopic before its first chunk. Before this
change, `import telescopic` imported `scipy.stats`, `scipy.optimize` and
`scipy.integrate` at its start, although only a few functions use them; now those
functions import them when first called. The question is how much processor time the
import saves, and what the first call that needs `scipy.stats` then pays.

Each measurement runs in a fresh interpreter (`python -I`, with the source folder first
on its path) and takes `time.process_time()` around the import, so the interpreter's own
start is left out. Three imports are timed: telescopic from the source of commit
`BEFORE`, the commit of `main` merged into this change, extracted with `git archive`
into a temporary folder; telescopic from this checkout's `src`; and `import numpy, scipy.linalg, scipy.special, scipy.sparse`, the
SciPy subpackages that telescopic still imports at its start, as a floor. After the
import of this checkout's telescopic, the same interpreter times `import scipy.stats`,
which the first call of `Result.estimate`, of the diagnostics (and so of every run of
`sample`) or of `testing.check_coupling` pays once. A first round, not counted, compiles
the bytecode of both sources; then each of the counted rounds runs the three imports in
an order that alternates between rounds. The ratio before/after of each round is
summarized by its median and its range. Each import's line lists the public SciPy
subpackages it left loaded, from the last round.

Result (`i83_import_time.out`, nine rounds at a load average of 2.9 to 3.5 on 4 cores):
`import telescopic` took a median of 0.603 processor seconds before and 0.223 after, with
a median of 2.79 for the ratio before/after of each round (range 2.22 to 3.09), against
0.214 for the floor. Before, it left eleven public SciPy subpackages and `scipy.version`
loaded, among them constants, fft, interpolate, ndimage and spatial; after, only linalg,
sparse and special (and `scipy.version`), as the floor does. The first call that needs
`scipy.stats` then pays its import once: a median of 0.362 processor seconds.

Run from the repository root: OMP_NUM_THREADS=1 PYTHONPATH=src python notes/experiments/i83_import_time.py
"""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from i85_setup import setup

ROOT = Path(__file__).resolve().parents[2]
BEFORE = "cfaed83"
ROUNDS = 9

# Times each statement in turn, and lists the subpackages loaded after the first.
CHILD = """
import json, sys, time
sys.path.insert(0, sys.argv[1])
seconds, loaded = [], None
for statement in sys.argv[2:]:
    start = time.process_time()
    exec(statement)
    seconds.append(time.process_time() - start)
    if loaded is None:
        loaded = sorted(
            name for name in sys.modules
            if name.startswith("scipy.") and name.count(".") == 1 and name[6] != "_"
        )
print(json.dumps([seconds, loaded]))
"""


def measure(source, *statements):
    output = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(source), *statements],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(output)


def extract(commit, folder):
    archive = subprocess.run(
        ["git", "archive", commit, "src/telescopic"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(folder, filter="data")
    return Path(folder) / "src"


def main():
    print(setup())
    print(f"before: commit {BEFORE}; after: this checkout; {ROUNDS} rounds")
    print(f"load average before: {tuple(round(x, 2) for x in os.getloadavg())}")
    with tempfile.TemporaryDirectory() as folder:
        imports = {
            "before": (extract(BEFORE, folder), "import telescopic"),
            "after": (ROOT / "src", "import telescopic", "import scipy.stats"),
            "floor": (ROOT / "src", "import numpy, scipy.linalg, scipy.special, scipy.sparse"),
        }
        for arguments in imports.values():
            measure(*arguments)
        seconds = {name: [] for name in imports}
        first_stats, last = [], {}
        for round_ in range(ROUNDS):
            names = list(imports) if round_ % 2 == 0 else list(reversed(imports))
            for name in names:
                times, last[name] = measure(*imports[name])
                seconds[name].append(times[0])
                if name == "after":
                    first_stats.append(times[1])
    for name in imports:
        values = np.array(seconds[name])
        print(
            f"{name:6}: processor seconds median {np.median(values):.3f}, range "
            f"{values.min():.3f} to {values.max():.3f}; SciPy subpackages loaded: "
            f"{', '.join(last[name])}"
        )
    ratio = np.array(seconds["before"]) / np.array(seconds["after"])
    print(
        f"before/after, per round: median {np.median(ratio):.2f}, range {ratio.min():.2f} "
        f"to {ratio.max():.2f}"
    )
    first = np.array(first_stats)
    print(
        f"import scipy.stats after import telescopic (the first call's cost): median "
        f"{np.median(first):.3f} s, range {first.min():.3f} to {first.max():.3f}"
    )
    print(f"load average after: {tuple(round(x, 2) for x in os.getloadavg())}")


if __name__ == "__main__":
    main()

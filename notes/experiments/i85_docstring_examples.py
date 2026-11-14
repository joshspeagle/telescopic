"""Do the Examples of the docstrings print the same with other seeds, and how fast (#85)?

Each example fixes ``rng=0``. Its printed output is exact on one machine and NumPy
version, but another platform or version can draw other numbers (spec R-EXEC-2). As a
proxy, each example is rerun with ``rng`` set to each of 30 seeds, warnings as errors,
and its output is compared with the one written in the docstring; any difference or
warning would fail on a platform that draws those numbers. The time is that of one run
at ``rng=0``, the seed of the docstring, after the imports.

Result (``i85_docstring_examples.out``): all 11 examples print their documented output
at all 30 seeds, without a warning; together they take 2.4 s, the longest (HMC)
0.49 s.

Run: python notes/experiments/i85_docstring_examples.py
"""

import doctest
import importlib
import os
import pkgutil
import platform
import subprocess
import time
import warnings

import numpy as np
import scipy

import telescopic

SEEDS = range(30)

examples = [
    test
    for module in pkgutil.walk_packages(telescopic.__path__, "telescopic.")
    for test in doctest.DocTestFinder().find(importlib.import_module(module.name), globs={})
    if test.examples
]
commit = subprocess.run(
    ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
).stdout.strip()
print(f"commit {commit}; {platform.machine()}, {os.cpu_count()} cores")
print(
    f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}; "
    "run with OMP_NUM_THREADS=1"
)
print(f"{'example':<52} {'seconds':>7} {'differs':>7}")
total = 0.0
warnings.simplefilter("error")
for test in examples:
    failures = 0
    for seed in SEEDS:
        for example in test.examples:
            example.source = example.source.replace("rng=0", f"rng={seed}")
        start = time.perf_counter()
        failures += bool(doctest.DocTestRunner().run(test, out=lambda text: None).failed)
        if seed == 0:
            seconds = time.perf_counter() - start
        for example in test.examples:
            example.source = example.source.replace(f"rng={seed}", "rng=0")
    total += seconds
    print(f"{test.name:<52} {seconds:7.2f} {failures:>4}/{len(SEEDS)}")
print(f"{'total':<52} {total:7.2f}")

"""Memory and repeated work in a result's estimates, before and after issue #75.

Question: do the changes of issue #75 to `Result` (the covariance summed in row blocks,
the per-replicate moments computed once, the quantiles gathered one column at a time),
to `SignedMeasure` (its sums built directly in CSR form) and to `evaluate_pair` (a view
for the first half) leave every output bitwise identical except `cov`, `var` and `std`,
by how much do those three move, and what do `var`, `cov` and `quantile` peak at?

Method: four fixed-seed runs through `telescopic.sample`, on `examples.gaussian(20)` at
R = 100 (unthinned and with thin = 10; the run of the issue's measurement), on
`examples.gaussian(2)` at R = 200 with MALA, and on `examples.gaussian(2)` at R = 200 with
k = 3, m = 3, lag = 1 and max_iter = 4, which censors pairs. Each output is printed as the
first 16 hex digits of the SHA-256 of its bytes, so two runs of this script on two commits
agree on a line exactly when that output is bitwise identical. The diagnostics and the
summary's footer leave out `seconds_per_replicate`, a wall time. `cov` is compared with
one product `centered.T @ (weights * centered)`, and `var` with the `einsum` of the same
moment, the forms of commit 932861f written out here: the largest absolute difference
relative to the largest entry of the reference for `cov`, entry by entry for `var` and
`std`. Peak memory is traced by `tracemalloc` for `var()`, `cov()` and
`quantile([0.16, 0.5, 0.84])`, as in the issue's "Measurement to repeat", each called on a
new `Result` of the run's replicates, so that none reads sums or moments that an earlier
call cached; and the bytes per atom that the sums of `SignedMeasure` hold besides the
weights.

Result (`i75_results_memory.out`, beside this script, is written at a clean commit; these are
from runs on 932861f and on the change, one machine, one thread): every hash agrees
except those of `cov`, `var`, `std` and the summary's `std` column. `cov` moved by at
most 3e-15 of its largest entry, and only on the run of more than 65,536 atoms, the
block; `var` by at most 7.5e-14 and `std` by at most 3.8e-14, entry by entry. On
gaussian(20), R = 100 (564,646 atoms, 90.3 MB), the peaks fell from 103.9 to 94.9 MB for
`var`, 189.8 to 105.4 MB for `cov` and 35.9 to 12.7 MB for `quantile`; with thin = 10
(9.0 MB of atoms), from 13.2 to 4.7 MB for `quantile`. The sums hold 8 bytes per atom
besides the weights, from 16.

Run: OMP_NUM_THREADS=1 python notes/experiments/i75_results_memory.py (about ten seconds on
one core).
"""

import hashlib
import tracemalloc
import warnings
from functools import partial

import numpy as np
from i85_setup import setup

import telescopic as ts
from telescopic import examples
from telescopic.estimators import unbiased_covariance, unbiased_variance

warnings.simplefilter("ignore")
print(setup(), flush=True)
LEVELS = [0.16, 0.5, 0.84]


def digest(value):
    if isinstance(value, dict):
        value = repr(sorted((key, value[key]) for key in value if key != "seconds_per_replicate"))
    data = value.encode() if isinstance(value, str) else np.ascontiguousarray(value).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def reference(result):
    # cov and var as commit 932861f computes them, in one product and one einsum.
    measure = result.measure
    atoms, weights, n = measure.atoms, measure.weights, result.n_replicates
    centered = atoms - np.ones(len(atoms)) @ atoms / len(atoms)
    first = measure.per_replicate(lambda _: centered)
    cov = unbiased_covariance(first, centered.T @ (weights[:, None] * centered) / n)
    second = np.einsum("i,ij,ij->j", weights, centered, centered) / n
    var = unbiased_variance(first, second)
    return cov, var


def relative(new, old, scale):
    return float(np.max(np.abs(new - old) / np.abs(scale)))


def peak(call):
    tracemalloc.start()
    call()
    size = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return size / 1e6


RUNS = {
    "gaussian(20), R = 100": (examples.gaussian(20), 100, {}),
    "gaussian(20), R = 100, thin = 10": (examples.gaussian(20), 100, {"thin": 10}),
    "gaussian(2), R = 200, MALA": (examples.gaussian(2), 200, {"kernel": "mala"}),
    "gaussian(2), R = 200, censored": (
        examples.gaussian(2),
        200,
        {"k": 3, "m": 3, "lag": 1, "max_iter": 4},
    ),
}

for name, (example, n_replicates, options) in RUNS.items():
    grad = example.grad if options.get("kernel") == "mala" else None
    result = ts.sample(
        example.log_prob,
        example.initial,
        n_replicates,
        grad=grad,
        vectorize=True,
        rng=1,
        **options,
    )
    measure = result.measure
    print(
        f"\n{name}: R = {result.n_replicates}, censored {result.n_censored}, "
        f"{len(measure.weights):,} atoms, {measure.atoms.nbytes / 1e6:.1f} MB"
    )
    if len(measure.weights) > 50_000:
        # Each on a result of its own, whose sums and moments are not cached yet: the
        # sampler's warnings have already computed the diagnostics of result.
        calls = (ts.Result.var, ts.Result.cov, lambda fresh: fresh.quantile(LEVELS))
        peaks = [peak(partial(call, ts.Result(result._replicates))) for call in calls]
        print("  peak MB: var {:.1f}, cov {:.1f}, quantile {:.1f}".format(*peaks))
    sums = measure._sums
    held = sums.indices.nbytes + sums.indptr.nbytes
    held += 0 if np.shares_memory(sums.data, measure.weights) else sums.data.nbytes
    print(f"  sums: {held / len(measure.weights):.1f} bytes per atom besides the weights")
    replicates = result._replicates
    for field in ("index", "meeting_time", "cost", "owner", "atoms", "weights", "distance"):
        print(f"  replicates.{field}: {digest(getattr(replicates, field))}")
    cov, var, std = result.cov(), result.var(), result.std()
    old_cov, old_var = reference(result)
    old_std = np.sqrt(np.maximum(0, old_var))
    print(f"  quantile: {digest(result.quantile(LEVELS))}")
    print(f"  quantile of h: {digest(result.quantile(LEVELS, lambda x: x[:, 0] * x[:, 1]))}")
    print(f"  mean: {digest(result.mean().replicates)}")
    print(f"  cov: {digest(cov)}, relative to the largest entry of the one-product form ", end="")
    print(f"{relative(cov, old_cov, np.max(np.abs(old_cov))):.2g}")
    print(
        f"  var: {digest(var)}, relative to the einsum form {relative(var, old_var, old_var):.2g}"
    )
    print(f"  std: {digest(std)}, relative {relative(std, old_std, old_std):.2g}")
    print(f"  diagnostics: {digest(result.diagnostics())}")
    summary = result.summary()
    for column, values in summary.table.items():
        print(f"  summary.{column}: {digest(values)}")
    print(f"  summary footer: {digest(summary.footer)}")

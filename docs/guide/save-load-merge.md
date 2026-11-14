---
file_format: mystnb
kernelspec:
  name: python3
---

# Save, load and merge runs

```{code-cell} python
:tags: [remove-cell]

import numpy as np

import telescopic
from telescopic import examples

gaussian = examples.gaussian(2)
result = telescopic.sample(
    gaussian.log_prob, gaussian.initial, 1000, vectorize=True, rng=11
)
```

## Merging runs

A sampler needs no merging: each call of `run` returns the result of all its replicates
so far. {meth}`telescopic.Result.merge` joins runs made apart, such as runs of
{func}`telescopic.engine.run` on separate ranges of replicate indices, on one machine or
several. The runs must have the same root seed, settings and dimension, and no replicate
index in common, which `merge` checks. They must also share the target, the kernel and the
initial distribution, which it cannot check. Here the replicates 1000 to 1199 extend the
run of [Estimate anything from a result](estimates.md), 1000 replicates of
{func}`telescopic.examples.gaussian` in two dimensions:

```{code-cell} python
more = telescopic.engine.run(
    telescopic.Target(gaussian.log_prob, vectorize=True),
    result.kernel,
    result.initial,
    np.arange(1000, 1200),
    k=result.k,
    m=result.m,
    lag=result.lag,
    thin=result.thin,
    max_iter=result.max_iter,
    rng=result.seed,
)
merged = result.merge(telescopic.Result(more, names=result.names))
merged.n_replicates
```

The merged result is `interrupted` if a replicate requested in either run ran in neither.
It keeps the names of the result whose `merge` was called, and its kernel, initial
distribution, tuning report and pilot meeting times, each taken from the other result
where this one has none.

## Saving and loading

{meth}`telescopic.Result.save` writes a result to a versioned `.npz` file, without
pickle, and {meth}`telescopic.Result.load` reads it back. Both append `.npz` to a name
that lacks it. The kernel and the initial distribution are saved as their descriptions,
`str(kernel)` and `str(initial)`, which a loaded result holds in their place; the paths
that the engine keeps for teaching are not saved. Loading a file whose schema is newer
than the installed telescopic's raises `ValueError`.

```{code-cell} python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as directory:
    result.save(Path(directory) / "gaussian")  # writes gaussian.npz
    loaded = telescopic.Result.load(Path(directory) / "gaussian")
loaded.kernel, np.array_equal(loaded.cov(), result.cov())
```

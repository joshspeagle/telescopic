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

A sampler needs no merging: each call of `run` returns the result of all its {term}`replicates <replicate>`
so far, and a new sampler continues a returned or loaded result with `resume`
([Continuing a run](#continuing-a-run)). {meth}`telescopic.Result.merge` joins runs made
apart, such as runs of {func}`telescopic.engine.run` on separate ranges of replicate
indices, on one machine or several. The runs must have the same root seed, settings and dimension, and no replicate
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
distribution, tuning report and pilot {term}`meeting times <meeting time>`, each taken from the other result
where this one has none.

## Saving and loading

{meth}`telescopic.Result.save` writes a result to a versioned `.npz` file, without
pickle, and {meth}`telescopic.Result.load` reads it back. Both append `.npz` to a name
that lacks it. The kernel and the initial distribution are saved as their descriptions,
`str(kernel)` and `str(initial)`, which a loaded result holds in their place; the paths
that the engine keeps for teaching are not saved. Loading a file whose schema is newer
than the installed telescopic's raises `ValueError`. The file names the estimator family
of the result, and a file of a family the installed telescopic does not know raises
`ValueError` naming it. Files written before the times of the {term}`atoms <atom>` were stored load with
those times rebuilt exactly from their settings and meeting times.

```{code-cell} python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as directory:
    result.save(Path(directory) / "gaussian")  # writes gaussian.npz
    loaded = telescopic.Result.load(Path(directory) / "gaussian")
loaded.kernel, np.array_equal(loaded.cov(), result.cov())
```

## Continuing a run

{meth}`telescopic.CoupledSampler.resume` continues a result on a new sampler, built with
the result's target, initial distribution and kernel. It adopts the result's seed,
replicates, settings, pilot meeting times and tuning report, so the next `run` keeps
`k`, `m`, the {term}`lag`, `thin` and `max_iter`, first runs the replicates that an interrupted
run never started, then new replicate indices above the highest the result requested.
For a result returned in this session, pass its tuned kernel:

```{code-cell} python
sampler = telescopic.CoupledSampler(
    gaussian.log_prob, gaussian.initial, vectorize=True, kernel=result.kernel
)
sampler.resume(result)
continued = sampler.run(200)
continued.n_replicates, int(continued.index[-1])
```

A loaded result holds only the description of its kernel. Pass the kernel as it was
built, here `"auto"`, untuned: `resume` tunes it again from the result's seed, which
gives the run's tuned kernel back when the target, the initial distribution, the kernel
and the machine are the same. It emits an `InputWarning` when the kernel's tuning or
its description, which shows a tuned step size, differs from the result's, as the tuning
can on another machine or with another version of NumPy, SciPy or BLAS:

```{code-cell} python
sampler = telescopic.CoupledSampler(gaussian.log_prob, gaussian.initial, vectorize=True)
sampler.resume(loaded)
sampler.run(200).n_replicates
```

`resume` checks the dimension, the kernel's class, the names, and an `rng` passed to the
sampler, which must give the result's seed. It cannot check that the target, the initial
distribution and the kernel's parameters are the run's: a sampler built with others
continues the indices of the result and mixes replicates of two different runs without a
warning. A result merged from runs made apart continues above the highest index any of
them requested, so the gaps between them stay free for other runs. Replicates that a cap
censored are kept and never rerun.

With a log-density evaluated point by point, a resumed run's replicates are bitwise
those of one uninterrupted run. With a vectorized one, the resumed run is bitwise what
the original sampler's next `run` would have returned, and agrees with one longer run
within its error bars (Reproducibility, in [Run in parallel](parallel.md)).

## Continuing after an error

An error raised while the replicates run, such as one of the log-density, stops the run
as a first Ctrl-C does: no replicate starts, and on a pool the running chunks finish.
The error is then raised again, the same object, with the result of the replicates
finished before it as its attribute `telescopic_result` and a note saying what was kept.
A sampler keeps those replicates, and its next `run` starts with the missing ones. After
{func}`telescopic.sample`, which keeps no sampler, `resume` continues the result. Here
the log-density fails once, during the run:

```{code-cell} python
class Flaky:
    # The example's log-density, which fails at its call number 3,000, once.
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        if self.calls == 3_000:
            raise RuntimeError("the solver did not converge")
        return gaussian.log_prob(x)


try:
    telescopic.sample(Flaky(), gaussian.initial, 1000, vectorize=True, rng=11)
except RuntimeError as error:
    kept = error.telescopic_result
    print(error.__notes__[0])
sampler = telescopic.CoupledSampler(
    gaussian.log_prob, gaussian.initial, vectorize=True, kernel=kept.kernel
)
sampler.resume(kept)
completed = sampler.run(1000 - len(kept.index))
completed.n_replicates, len(completed.abandoned), completed.interrupted
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert len(kept.abandoned) and kept.interrupted
assert "an error abandoned" in repr(kept.summary())
assert not len(completed.abandoned) and completed.n_replicates == 1000
```

The replicates in progress when the error came are listed in the result's `abandoned`,
and nothing of them is kept. They are the ones with the longest meeting times, since
the others had finished, so leaving them out biases the estimates more than an
interruption does; the summary says so until they have run. An error that recurs at a
replicate recurs there when it runs again. An error in tuning or in the
{term}`meeting-time pilot` is raised as it is, and keeps nothing. With a log-density evaluated point by
point that gives the same value at a point on every call, the completed run's
replicates are bitwise those of a run that did not fail.

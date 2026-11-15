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

`merge` takes any number of results, `a.merge(b, c, d)`, and joins them in one pass,
which is faster than joining them two at a time. The merged result is `interrupted` if a
replicate requested in any of the runs ran in none. It keeps the names of the result
whose `merge` was called, and its kernel, initial distribution, tuning report and pilot
{term}`meeting times <meeting time>`, each taken from the first of the others that has one where this result
has none.

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

## Checkpoints

A run that a process kill would lose, from a cluster's time limit, the out-of-memory
killer or a node that fails, can write its replicates to a directory as they finish.
`run(n, checkpoint=directory)` makes the directory if needed and writes there the
sampler's plan, its seed, settings, tuning report and pilot meeting times, then the
replicates as they finish: every 256 in this process, and each chunk on a pool. Each
file is written under a temporary name and then renamed, so a killed process leaves
whole files only. A killed run has nothing to catch; running the same call again, on a
new sampler built with the same target, initial distribution and kernel, adopts the
plan, reads the replicates the directory holds, runs the others, and returns the result
of all of them. {meth}`telescopic.Result.load` of the directory returns the result of
every replicate it holds:

```{code-cell} python
with tempfile.TemporaryDirectory() as directory:
    sampler = telescopic.CoupledSampler(
        gaussian.log_prob, gaussian.initial, vectorize=True, rng=11
    )
    first = sampler.run(600, checkpoint=directory)
    files = sorted(path.name for path in Path(directory).iterdir())
    # The same call after a crash: a new sampler, which reads all 600 and runs none.
    again = telescopic.CoupledSampler(
        gaussian.log_prob, gaussian.initial, vectorize=True, kernel=first.kernel
    )
    rerun = again.run(600, checkpoint=directory)
    stored = telescopic.Result.load(directory)
files, again.target.n_log_prob, stored.n_replicates
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert again.target.n_log_prob == 0
assert stored.n_replicates == rerun.n_replicates == 600
assert np.array_equal(rerun.mean().value, first.mean().value)
```

`plan.npz` holds the plan, `request-0-599-600.npz` the indices the call requested, and
each `part-<first>-<last>.npz` a set of finished replicates, which
{meth}`~telescopic.Result.load` also reads alone. The result loaded from the directory is
`interrupted` until every requested replicate has finished. The new sampler checks the
plan as `resume` checks a result ([Continuing a run](#continuing-a-run)): an untuned
kernel is tuned again from the plan's seed, with a warning if its tuning differs, and a
sampler that has already fixed other settings or another seed raises. With a
log-density evaluated point by point, the completed run's replicates are bitwise those
of one run without a checkpoint.

## Run on a cluster

A cluster job splits one run into tasks that run separate ranges of its replicate
indices with one plan. `CoupledSampler.plan` tunes the kernel, runs the meeting-time
pilot and fixes the settings, without running replicates, and writes the plan to the
checkpoint directory; each task's sampler adopts it and runs its own range with
`run(n, start=i)`, the indices `i` to `i + n - 1`; and `Result.load` of the directory
merges every task's replicates. The model lives in code that each script imports, here
a file `model.py` that defines `log_prob` and `init`. The plan, once:

```python
# plan.py
import telescopic
from model import init, log_prob

sampler = telescopic.CoupledSampler(log_prob, init, rng=2026)
print(sampler.plan(checkpoint="run"))
```

Each task of a Slurm job array, 1,000 replicates each:

```python
# task.py
import os

import telescopic
from model import init, log_prob

task = int(os.environ["SLURM_ARRAY_TASK_ID"])
sampler = telescopic.CoupledSampler(log_prob, init)
sampler.run(1000, start=1000 * task, checkpoint="run")
```

And the merge, once every task has ended:

```python
# merge.py
import telescopic

result = telescopic.Result.load("run")
print(result.summary())
result.save("result")
```

Submitted with dependencies, so that the tasks start after the plan and the merge after
the tasks:

```bash
plan=$(sbatch --parsable --wrap "python plan.py")
tasks=$(sbatch --parsable --dependency=afterok:$plan --array=0-99 --wrap "python task.py")
sbatch --dependency=afterany:$tasks --wrap "python merge.py"
```

A task that was killed, or requeued, runs the same call again and completes its range
from its own files. Until it does, the merged result is `interrupted`, since the task
recorded the indices it requested before its first replicate finished. Every task tunes
an untuned kernel again from the plan's seed, which takes a small part of a task and
gives the plan's kernel on machines with the same libraries; on others it can differ by
rounding, which the task warns about. Each replicate stays unbiased whatever its kernel,
so the estimates stay unbiased, but they no longer repeat bitwise. To tune once, pass
every task the tuned kernel by pickle ([Tuning](parallel.md#tuning), in Run in
parallel). Do not run two tasks on overlapping ranges: the merge raises when two files
hold one replicate.

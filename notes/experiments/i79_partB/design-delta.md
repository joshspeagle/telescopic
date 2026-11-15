# Part B of issue #79 against the code of `main` at db9ed1a

The sketch of part B (issue #79, "B (recommended second)") was written before part A
(#172), `CoupledSampler.resume` (#170) and the kernels of #77. This note takes each item
of its checklist in turn: what the code already does, what is left to build, and where
the sketch no longer fits. The choices are DECISIONS #264 to #270.

## 1. `engine.run(sink=, flush=)` and flushing on an exception

- **Already there.** Part A's outcome records (DECISIONS #230) keep every replicate
  finished before an error, in process and on pools, and the sampler keeps them; the
  test the sketch names, `test_engine_returns_replicates_finished_before_an_error`,
  exists in substance as `test_engine_attaches_the_replicates_finished_before_an_error`.
- **To build.** Only the periodic flush, which a checkpoint needs: a process killed
  (by Slurm's time limit, the out-of-memory killer, a node failure) raises nothing, so
  only rows already on disk survive. `_Batch` hands the replicates finished since the
  last flush to a sink every 256 finished replicates and drops their rows; on a pool
  the parent hands each chunk's outcome to the sink as it arrives.
- **No longer fits.** `sink` and `flush` stay private (`engine._run`, `_parallel.execute`):
  the sampler is their only caller, and `engine.run` keeps its signature (#264).

## 2. `Kernel.parameters()`

- **Already there.** `resume` rebuilds the run's kernel without saving it: the caller
  builds the kernel in code, and an untuned one is tuned again from the adopted seed,
  with an `InputWarning` when the report's mean or covariance differ bitwise from the
  run's (DECISIONS #221). Tuning is a small part of a cluster task (4 chains and, for
  MALA and HMC, 1,000 steps of dual averaging, against the thousands of coupled steps of
  each replicate), and a kernel tuned on another machine leaves each replicate
  unbiased, since every replicate is unbiased whatever its kernel; only the bitwise
  repeat is lost, and the warning says so.
- **Not built** (#265). Saving kernels by parameters would need a method and a
  constructor table for RandomWalk, MALA, HMC, Independence, MultivariateT, Mixture,
  every proposal and coupling object of #77 and every user subclass, a refusal for
  Gibbs and for user proposals, and an amendment of DECISIONS #26 and #29, to save the
  tuning of each task. A Gibbs kernel, whose conditionals are code, needs nothing
  special: like every kernel, it is built in the task's code.

## 3. `Plan` and `CoupledSampler.plan()`

- **Already there.** What the sketch's `Plan` holds (seed, settings, tuned kernel,
  tuning report, pilot) is what `resume` adopts from a `Result`; `kernel.tune(target,
  initial, rng, *, pilot=None)` keeps the signature the sketch assumed.
- **To build.** A way to fix the settings without running replicates:
  `CoupledSampler.plan(*, k, m, lag, thin, max_iter, keep, checkpoint=None, progress=None)`
  tunes, runs the pilot, fixes the settings as a first `run` would, returns them, and
  writes them to a checkpoint directory. The sampler keeps its settings in a field of its
  own rather than reading them from its replicates.
- **No longer fits.** No `Plan` class, no `Plan.run(target, initial, indices)` and no
  `CoupledSampler(plan=...)` (#264): the plan is the file `plan.npz` of a checkpoint
  directory, which a new sampler adopts as `resume` adopts a result, and the sampler's
  `run` runs the replicates. A range job needs one argument the sampler lacks:
  `run(n, start=i)` runs the indices `i, ..., i + n - 1` (#267).

## 4. `Plan.save`, `Plan.load`, `CoupledSampler(plan=...)`

- **Already there.** Schema 2 (DECISIONS #243) writes a record of replicates with its
  seed, settings, names, descriptions, tuning arrays, pilot and `k_rule`; nothing in it
  needs replicates to exist.
- **To build.** The plan is that record with no replicates (#266), so the schema does not
  move. Reading it is internal: `run(checkpoint=...)` and `plan(checkpoint=...)` adopt it.

## 5. `RunStore`, `telescopic.merge`, `run(checkpoint=path)`

- **Already there.** `engine.merge` takes any number of parts; `Result.merge` takes one;
  `Result.save` and `Result.load` read and write one file atomically only in effect.
- **To build.**
  - `run(n, ..., checkpoint=directory)`: adopts the directory's plan in a new sampler,
    or writes the sampler's own; refuses a plan of another seed or settings; writes a
    request record of the call's indices, then a part per flush, each through a
    temporary file and `os.replace`; on a later call, skips the indices already stored
    and takes their replicates from the files, so that rerunning the same command after a
    crash completes the run (#267, #268).
  - `Result.load(directory)` merges the plan and every record into one `Result`, which is
    `interrupted` while a requested index is missing (#269).
  - `Result.merge(*others)` joins any number of results in one `engine.merge` (#270).
- **No longer fits.** No `RunStore` class: its `plan`, `write`, `missing` and `result`
  are the three entry points above. No `telescopic.merge`: `Result.merge` becomes n-ary.

## 6. Budgets

Ruled separately (Josh, 2026-11-13): an experiment under DECISIONS #69 first. Not part of
this change.

## What a reader of the sketch should know

- `resume` already covers continuing in memory and from one saved file.
- The sketch's three-line Slurm task becomes `CoupledSampler(log_prob, init).run(n,
  start=i * n, checkpoint="run")`, and the merge step `Result.load("run")`.
- Seeded runs without `checkpoint` keep their bits: the flush is not taken without a sink,
  and the final join of `_Batch` keeps its order.

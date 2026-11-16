# Part B of issue #79 against the code of `main` at db9ed1a

The sketch of part B (issue #79, "B (recommended second)") predates part A (#172),
`CoupledSampler.resume` (#170) and the kernels of #77. For each item of its checklist:
what the code already does, what is left to build, and where the sketch no longer fits.
The choices are DECISIONS #264 to #270.

## 1. `engine.run(sink=, flush=)` and flushing on an exception

- **Already there.** Part A's outcome records (DECISIONS #230) keep every replicate
  finished before an error, in process and on pools, and the sampler keeps them; the
  sketch's `test_engine_returns_replicates_finished_before_an_error` exists in substance
  as `test_engine_attaches_the_replicates_finished_before_an_error`.
- **To build.** The periodic flush a checkpoint needs: a killed process (a time limit,
  the out-of-memory killer, a failed node) raises nothing, so only rows on disk survive.
  `_Batch` hands the replicates finished since the last record to a sink every 256
  finished replicates and drops their rows; on a pool the parent hands over each chunk.
- **No longer fits.** `sink` and `flush` stay private (`engine._run`, `execute`): the
  sampler is their only caller, and `engine.run` keeps its signature (#266, #268).

## 2. `Kernel.parameters()`

- **Already there.** `resume` rebuilds a run's kernel without saving it: the caller
  builds the kernel in code, and an untuned one is tuned again from the adopted seed,
  with an `InputWarning` when the report differs bitwise (DECISIONS #221). Tuning is a
  small part of a cluster task, and a kernel tuned apart leaves each replicate unbiased,
  since a replicate is unbiased whatever its kernel; only the bitwise repeat is lost,
  and the warning says so. The parallel guide already shows a tuned kernel passed to
  other jobs by pickle, the user's own choice.
- **Not built** (#267). It would need a method and a constructor table for RandomWalk,
  MALA, HMC, Independence, MultivariateT, Mixture and every proposal and coupling of
  #77, refusals for Gibbs, user proposals and user kernels, and an amendment of
  DECISIONS #26 and #29. Gibbs, whose conditionals are code, then needs nothing special.

## 3. `Plan`, `CoupledSampler.plan()`, `Plan.save`, `Plan.load`, `CoupledSampler(plan=)`

- **Already there.** What the sketch's `Plan` holds (seed, settings, tuned kernel,
  tuning report, pilot) is what `resume` adopts from a `Result`. `kernel.tune(target,
  initial, rng, *, pilot=None)` keeps the signature the sketch assumed. Schema 2
  (DECISIONS #243) writes a record of replicates with its seed, settings, names,
  descriptions, tuning arrays, pilot and `k_rule`, and nothing in it needs a replicate.
- **To build.** `CoupledSampler.plan(*, k, m, lag, thin, max_iter, keep, checkpoint,
  progress)` fixes the settings as a first `run` would, without replicates, and returns
  them; the sampler keeps its settings in a field of its own. The plan is a schema-2
  record with no replicate, `plan.npz` in a checkpoint directory, so the schema does not
  move. A range job needs `run(n, start=i)`, the indices `i` to `i + n - 1` (#265).
- **No longer fits.** No `Plan` class, `Plan.run(target, initial, indices)`, `Plan.save`,
  `Plan.load` or `CoupledSampler(plan=)` (#268): a new sampler adopts a directory's plan
  as `resume` adopts a result, and the sampler's `run` runs the replicates.

## 4. `RunStore`, `telescopic.merge`, `run(checkpoint=path)`

- **Already there.** `engine.merge` joins any number of parts; `Result.merge` one other.
- **To build.**
  - `run(n, ..., checkpoint=directory)` (#269): adopts the directory's plan in a new
    sampler, or writes the sampler's own; refuses the plan of another seed or settings;
    writes a request record of the call's indices, then a part at each record, each
    through a temporary file and `os.replace`; skips the indices already stored and reads
    their replicates, so rerunning the same call after a crash completes the run.
  - `Result.load(directory)` merges the plan and every record into one `Result`,
    `interrupted` while a requested index is missing (#270).
  - `Result.merge(*others)` joins any number of results in one `engine.merge` (#264).
- **No longer fits.** No `RunStore` class: its `plan`, `write`, `missing` and `result` are
  the entry points above. No `telescopic.merge`: `Result.merge` becomes n-ary.

## 5. Budgets

Ruled separately (Josh, 2026-11-13): an experiment under DECISIONS #69 first; not part of
this change.

## For a reader of the sketch

- `resume` already covers continuing in memory and from one saved file.
- The sketch's three-line Slurm task becomes `CoupledSampler(log_prob, init).run(n,
  start=i * n, checkpoint="run")`, and the merge step `Result.load("run")`.
- Seeded runs without `checkpoint` keep their bits: no record is made during a run
  without a sink, and the final join of `_Batch` keeps its order.

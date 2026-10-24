# Review: run settings in the replicates

**Diff:** e7dc402..a1076ee in `src/` and `tests/`, about 270 lines changing the public API
(DECISIONS #31), so the next size up: three `/simplify` reviewers (DECISIONS #3), on
altitude, efficiency, and reuse with simplification. The `/code-review` follows these
fixes.

## Outcome

- **The seed identifies the streams** (altitude). `engine.run` kept a `namespace`
  parameter whose values were private constants, and the recorded seed did not include
  it, so `merge` joined a main run's replicates with a pilot's of the same seed: the
  reviewer's merged replicates 3 to 5 met at 23, 5 and 3 against 7, 5 and 9 in one run.
  The parameter is gone. The engine runs namespace 0 of the seed it is given, and the
  meeting-time pilot gets the root's namespace-2 seed, so a recorded seed identifies its
  streams and the cross-namespace merge fails as one of different seeds. Main and tuning
  streams are unchanged; the pilot's change (spec §6.1, DECISIONS #32). A new test
  fails if the pilot shares the main run's streams, which no test caught before.
- **One settings type** (altitude, reuse). The five settings travelled as a group through
  six modules without a type: rebuilt by reflection from a tuple of names in three
  places, written field by field in four, and kept by the sampler as a mutable dict.
  Setting `sampler.settings["lag"] = 7` made the next run do all its work and then fail
  to merge, and every later run with it. They are now `engine.Settings`, an immutable
  named tuple, stored as `Replicates.settings`; `merge` compares it in one check.
- **One name for the seed** (reuse). The same object was `Replicates.root`,
  `Result.seed` and `CoupledSampler.seed`; it is `seed` everywhere.
- **A seed record that stays put** (altitude). `root_seed` returned a caller's own
  `SeedSequence`, so spawning from it later changed the recorded `n_children_spawned`,
  and what was saved. It now copies it.
- **One stopping-time rule** (reuse, altitude). The time each pair stopped, its meeting
  time or the cap, was computed in `Result` and in the sampler's slow-pair check, and the
  sampler's copy had no test. It is `Replicates.stopped`.
- **Results need measures** (altitude, in passing). `Result(run(..., store=False))`
  estimated 0 for a target of mean 3. It now refuses replicates that met without stored
  measures; with `store=True` each such replicate holds at least |S| ≥ 1 atoms.
- **Tests** (reuse).
  - One helper compares replicates field by field, in `tests/references.py`, for the
    lanes and pool tests. It skips the seed, which enters every array: the root asserts
    the change had added could not fail.
  - A test that the replicates record their settings repeated 21 others. It is replaced
    by one that reproduces a run from a generator with the recorded seed, the one
    mutant it missed.
  - The merge refusals (seed, settings, dimension) are one test next to `merge`.
  - The 1-D branch of `plotting.paths`, the only reader of the lag there, now runs.

Every fix with a test was checked by a mutant: the settings check of `merge`, the cap in
`stopped`, the copy in `root_seed`, the guard in `Result` and the pilot's seed.

## Findings not adopted

- Efficiency: nothing material. The new work is about 5 µs per pool chunk in the parent
  (1.8 µs of checks and 3.3 µs more unpickling), 0.08% of a measured 256-chunk run of
  1.635 s; `merge` now computes the first part's seed key once, as suggested.
- Reuse: one dictionary for every scalar field of `Replicates` in the saved file. With
  the settings one value, the file keeps three keys, for the settings, the seed and the
  counts, which read better.
- Reuse: `dataclasses.replace` in `merge` would save a line, but would carry over any
  field that a later change forgets to combine, where the constructor fails.
- Altitude: `store` is also a pilot's option on the public `engine.run`. It is a real
  choice for a user who wants only meeting times, and the guard in `Result` covers its
  misuse.

## `/code-review` of the change

**Scope:** e7dc402..d576955 in `src/` and `tests/` (the change and the fixes above), at
high effort. Eight findings: seven fixed, one answered. Each fix has a test that fails
without it, checked by mutants, except the last two, which change no behaviour.

1. `merge` joined a stored run with an unstored one: the unstored replicates then
   counted as estimates of 0, and the mean of a target of mean 3 came out 1.50 ± 0.09.
   `store` is now one of the `Settings`, which `merge` compares, and `check_settings`
   takes a `Settings` value.
2. The guard in `Result` inferred `store` from the atoms, so it passed a run with
   `store=False` whose pairs were all censored, and `w1_bound` then reported 0 for
   distances it never stored. `Result` now reads the recorded `store`. This answers the
   simplify round's claim that the guard covered the misuse of `store`.
3. The sampler kept its settings beside its replicates, and reassigning them made every
   later run do its work and then fail to merge. `settings` is now a read-only property
   that reads the replicates.
4. The removed check that censored pairs need the run's cap returns: hand-built
   replicates can still have censored pairs and no `max_iter`, and their stopping times
   then cast `inf` to an integer.
5. `merge([])` raised `IndexError`; it now raises `ValueError` naming `parts`.
6. The pilot's `Settings` passed `thin=1`, the default.
7. `check_coupling` derived its streams without spawning so as not to change a caller's
   `SeedSequence`. Since `root_seed` copies it, `spawn` is safe again, and the code and
   its stale comment return to the simpler form.

Answered: `Result` copies the settings into plain attributes, which a user could
overwrite. All of `Result`'s attributes are plain, and overwriting one is a deliberate
act, unlike the constructor arguments that DECISIONS #31 removed.


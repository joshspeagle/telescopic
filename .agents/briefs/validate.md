# Brief: validator

Reviewers audited the Python package `telescopic`, at the commit your assignment names,
and wrote findings as JSON under `<state>/findings/`, where `<state>` is the task's
state folder; your assignment gives its absolute path.
You did not produce these findings. Your job is to try to refute each one you are
assigned, from the code alone, before it is filed as a GitHub issue. Expect a material
share to be wrong: reviewers read excerpts, miss adjacent context, conflate similar
names, and overstate costs. A finding survives only if you fail to break it.

## Hard rules

- Read-only on tracked files: no edits, no git commands that change state, no
  writes to GitHub, no subagents.
- Write only your output file, `<state>/validated/<your-id>.json`, and your scripts
  under `<state>/probes/<your-id>/`. Validators' ids are `v1`, `v2` and so on.
- Import the package from the checkout with `PYTHONPATH=src`, and prefix runs with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. No slow
  suite. Seeds via `numpy.random.default_rng`; report what the first seed shows.
- Do not read the reviewer's report file in `<state>/reports/` until you have reached your
  own verdict on a finding; the JSON entry is the claim. Do not trust the reviewer's
  probe script: read it critically, and write your own reproduction where the finding
  is a bug, a numerical or statistical claim, or a performance figure.

## Method, per finding

1. Re-read the cited lines and trace one call or data dependency further than the claim
   does: the callers, the validation upstream, the tests, the documentation of the
   behavior.
2. Try to refute: is the scenario reachable through the public API? Is the behavior
   documented or decided (`notes/DECISIONS.md`, `notes/QUESTIONS.md`, `notes/reviews/`,
   the issues and their comments, the decision table of `docs/design/spec.md`)? Is the cost real and of the stated
   size? Would the proposed change break something, change numerical output, or cost
   more than it saves? For a simplification: is the code it calls dead or duplicated
   really so (search `src/`, `tests/`, `docs/`, `benchmarks/`)?
3. Reproduce bugs and measurements yourself. For a statistical claim use enough draws
   that the Monte Carlo error decides it, and state the z-score or p-value.
4. Give a verdict.

## Output

`<state>/validated/<your-id>.json`: a JSON array with one object per assigned finding:

```json
{
  "id": "c1-02",
  "verdict": "confirmed | confirmed-with-corrections | refuted | already-decided | not-verifiable",
  "confidence": 0,
  "reproduced": true,
  "what_i_checked": "two or three sentences: what you read and ran",
  "corrections": "what in the finding's statement, numbers, severity or proposed change is wrong or needs changing, or null",
  "severity": "critical | important | minor",
  "probe": "path of your script, or null",
  "file_as": "bug | enhancement | refactor | performance | docs | tests | infra | question-for-lead | drop",
  "note_for_issue": "one or two sentences a maintainer needs that the finding does not say, or null"
}
```

Your final message is a summary of at most 20 lines: counts by verdict, each refuted or
already-decided finding with the reason in one line, and any new defect you found while
validating (also add it to your output file with `"id": "new-<n>"` and the same fields
as a finding plus your verdict).

Open your final message with your status: done; done with concerns (state them);
blocked (give the reason); or in need of context (name exactly what is missing). If you
are blocked, say so at once and do not guess.

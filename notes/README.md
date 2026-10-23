# Working notes

These notes are the shared memory of the project between Josh and Claude. They are
written so that a new session, after a context reset, can continue exactly where the
last one stopped. They are not part of the documentation site.

| File | What it holds | How it changes |
|---|---|---|
| `PLAN.md` | Phases, milestones, current position and status | Updated when a milestone starts or finishes, or the plan changes |
| `LOG.md` | Dated log: what was done, what was found, which commits | Append only |
| `DECISIONS.md` | Decisions taken with Josh or after the spec was written, with reason and alternatives | Append; edit only the status column |
| `QUESTIONS.md` | Questions only Josh can answer, each with the default that work proceeds on | Items move to `DECISIONS.md` when settled |
| `experiments/` | Prototype scripts behind design choices and unpublished constructions, each with its result in the header | Added per experiment |
| `reviews/` | Reports of milestone reviews and of independent audits | Added per review |

Design decisions taken while writing the specification live in its decision table
(`docs/design/spec.md`), not here: each fact has one home.

## Rules for a new session

1. Read `PLAN.md`, the last entries of `LOG.md`, and the open items of `QUESTIONS.md`.
2. Do not reopen a recorded decision unless Josh does.
3. Record as you go: a log entry per working session or milestone, a decision entry per decision, a commit per settled change.

# Working notes

These notes are the shared memory of the project between Josh and Claude. They are
written so that a new session, after a context reset, can continue exactly where the
last one stopped. They are not part of the documentation site.

| File | What it holds | How it changes |
|---|---|---|
| `PLAN.md` | Phases, milestones and status of the first draft (tracking issue #1); closed (DECISIONS #96) | Not updated |
| `LOG.md` | Dated log of the first draft; closed (DECISIONS #96) | Not updated |
| `DECISIONS.md` | Decisions taken with Josh or after the spec was written, with reason and alternatives | Append; edit only the status column |
| `QUESTIONS.md` | Questions only Josh can answer, each with the default that work proceeds on | Items move to `DECISIONS.md` when settled |
| `experiments/` | Prototype scripts behind design choices and unpublished constructions, each with its result in the header | Added per experiment |
| `reviews/` | Reports of the first draft's milestone reviews and audits | Not updated; later reports are comments on their epic |
| `review/` | Josh's review of the first draft: its guide, log, decisions, artifacts and the generated hub page (`hub/`) | Not updated |

Design decisions taken while writing the specification live in its decision table
(`docs/design/spec.md`), not here: each fact has one home.

## Rules for a new session

1. Read the tracking issue that `CLAUDE.md` names (its State table and its last log
   comments), then the open items of `QUESTIONS.md`.
2. Do not reopen a recorded decision unless Josh does.
3. Record as you go: a log comment on the tracking issue per session, a decision entry
   per decision, a commit per settled change.

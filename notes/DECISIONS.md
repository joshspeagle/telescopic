# Decisions

Decisions taken with Josh, or after the specification was written. Design decisions
taken while writing the specification are in its decision table
(`docs/design/spec.md`). Outcomes: **adopted**, **changed** (cite the commit),
**deferred**, **revisit**.

| # | Date | Topic | Decision | Reason | Alternatives | Outcome |
|---|---|---|---|---|---|---|
| 1 | 2026-10-23 | Scope of v0.1 | Published coupling constructions only. New ones (e.g. a coupled ensemble sampler, coupled NUTS) are prototyped and validated in `notes/experiments/` and promoted only after Josh has seen the evidence (Josh) | keeps v0.1 on proven ground and potential papers under Josh's control | new methods as first-class v0.1 features; no new methods at all | adopted |
| 2 | 2026-10-23 | Process | One working branch (`claude/coupled-mcmc-package-btpupz`). Each milestone is a reviewed unit: tests, `/simplify`, `/code-review` at an effort scaled to its size, fixes, a log entry, then commit and push. No GitHub issues or pull requests (Josh) | less overhead for an autonomous first draft; the same review gates as a pull request | GSPICE-style tracking issues and a draft pull request per milestone | adopted |
| 3 | 2026-10-23 | Review budget | Consecutive milestones may form one review unit: one `/simplify` round and one `/code-review` on their combined diff, sized by the table in `CLAUDE.md`. The reuse and simplification angles may share one reviewer. M0 to M3 get one `/code-review` pass | about 14M tokens must cover nine more milestones; a reviewer's cost is mostly fixed (reading the spec, `CLAUDE.md` and the context), so fewer, larger units buy the same scrutiny for less | one unit per milestone with four reviewers each (DECISIONS #2 as first applied) | adopted |

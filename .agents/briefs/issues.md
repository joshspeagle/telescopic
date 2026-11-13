# Brief: issue drafter

A team reviewed the Python package `telescopic`, at the commit your assignment names.
Reviewers wrote findings to `<state>/findings/<id>.json`; validators who had not
produced them tried to refute each one and wrote verdicts to
`<state>/validated/<id>.json` (match on the finding's `id`; entries `new-<n>` are defects the
validators added). Reviewers' reports are in `<state>/reports/`, and probe scripts in
`<state>/probes/`. `<state>` is the task's state folder; your assignment gives its
absolute path.

You draft the GitHub issues for the findings your assignment lists. The lead will
read each issue and decide which items to act on; an implementer who sees
only the issue will then fix them. Nothing is fixed now. The lead agent reads your
drafts and posts them.

## Rules

- Read-only on tracked files. Write only your draft files under `<state>/issues/`. No git
  commands that change state, no GitHub writes, no subagents.
- **The validator's verdict governs.** Where a validator corrected a finding's statement,
  numbers, severity, line numbers or proposed change, write the corrected version. Do
  not include a finding whose verdict is `refuted`, or whose `file_as` is `drop`, in the
  checklist; list it under "Not included" with the reason in one line. A finding marked
  `already-decided` or `question-for-lead` goes under "Needs a ruling", stating the
  existing decision or spec section and what evidence reopens it. A finding marked
  `not-verifiable` goes under "Not included", with what a person should check.
- **Check every location yourself.** Open the file at the cited lines and correct the
  line numbers; validators found several wrong. Quote at most a few lines of code.
- **Public-facing text.** The repository is public. No machine paths (`/tmp/...`,
  `/home/...`, `~/...`), no hostnames, no references to agents, sessions, scratch folders,
  probe script paths or these JSON files. Add no attribution footer: the lead agent adds
  it when it posts. Where a reproduction matters, inline it as a
  short runnable snippet (at most 12 lines, using only the public API and
  `numpy.random.default_rng`), taken from a probe and checked by running it
  (`OMP_NUM_THREADS=1 PYTHONPATH=src python`),
  with its actual output quoted. Every number in the issue comes from a finding, a
  validation, or your own run.
- **Merge duplicates.** Several reviewers found the same defect; write it once and keep
  all the reference tags.
- **Register.** American English, complete sentences, literal wording, no em dashes, no
  evaluative commentary. Name things precisely: the function, the argument, the error
  raised.

## Format of each draft file

The first line is `# <issue title>` (imperative or a noun phrase, at most 80
characters). Then:

```markdown
## Objective

Two or three sentences: what this issue collects and why it matters to a user or a
maintainer.

## Findings

- [ ] **<imperative title>** (`path/to/file.py:120-134`; important | minor) [ref: c1-02]
  What happens, under which input, with the figure that shows it. What should happen.
  **Proposed change:** the concrete change. Note when it changes numerical output or the
  public API, since those need a decision record.

(ordered by severity, then by module)

## Needs a ruling

- **<question>** [ref: ...] The existing decision or spec text, the new evidence, the
  options, and the default recommended.

## Verification

How a fix is checked: for each bug, a regression test that fails without the fix, named
as a sentence; the commands (`pytest`, `ruff check .`, `mypy`); any measurement to repeat.

## Not included

- <finding title> [ref: ...]: refuted or dropped, with the reason in one line.
```

Omit a section that would be empty. Keep each issue under 45,000 characters; split a
theme into two issues by module if it would exceed that, and say so in your summary.

Your final message lists the draft files you wrote with, for each, its title, the number
of checklist items, and anything you could not verify.

Open your final message with your status: done; done with concerns (state them);
blocked (give the reason); or in need of context (name exactly what is missing). If you
are blocked, say so at once and do not guess.

# Working with several agents

How a lead agent and the agents it briefs share work on telescopic. The rules are in
the Agents section of `CLAUDE.md`. This folder holds the brief of each role and the
format of its output, so that the same process runs on any agent platform.

Two words are kept apart throughout: **Josh** is the project's lead, who rules on
findings, approves what enters the public API and merges; the **lead agent** is the
agent that plans a task, briefs the others and integrates their work.

## Roles

| Role | Brief | Does | Writes | Model tier |
|---|---|---|---|---|
| Lead agent | `CLAUDE.md` | plans, briefs, reads every output, files issues, commits | issues, pull requests, `notes/` | the session's |
| Reviewer | `review.md` | audits assigned files through one lens | `findings/<id>.json`, `reports/<id>.md` | default; strongest for the statistical audit |
| Validator | `validate.md` | tries to refute or reproduce each finding from the code alone | `validated/<id>.json` | default; fast for findings that a search settles |
| Issue drafter | `issues.md` | turns validated findings into issue drafts | `issues/<nn>-<theme>.md` | default |
| Scout | `research.md` | sweeps the literature or generates directions, with verified citations | one Markdown file | default |
| Experimenter | `experiment.md` | tests one direction against a stated falsification test | `exp/<id>/` scripts and outputs; the report in its final message | default |

An implementer works from an issue, in its own worktree, under the Changes section of
`CLAUDE.md`; the issue is its brief. On a platform without subagents, the lead agent
plays each role in turn, as a separate pass that starts from the brief and the files
and not from its earlier conclusions.

## Assignments

A brief is general; an assignment is the few lines that start one agent. It says which
brief to read, and gives the agent's id, its lens or question, the files or findings it
covers, the commit, and the absolute path of the task's state folder. For example:

```text
Read .agents/briefs/review.md and follow it exactly.
Your id: c3. Lens: correctness. Commit: 4da2a81.
Files: src/telescopic/engine.py, _random.py, _parallel.py.
State folder: /path/to/checkout/.agents/swarm/review-2026-11/
Hunt in particular for: off-by-one errors in the estimator's index ranges; a replicate
whose output depends on the lane count.
```

Ids are short and unique within a task: `c1`, `c2` for correctness reviewers, `q1` for
quality, `v1`, `v2` for validators, `e1` for experiments. A finding's id is its
reviewer's id and a number, `c3-02`. An agent has finished when its output file exists
and its final message has arrived; a file alone may be half written.

## State

While a task runs, its working files live in one state folder, `.agents/swarm/<task>/`
in the lead agent's checkout, which git ignores. The briefs write it as `<state>`, and
assignments give its absolute path, because agents in other worktrees would otherwise
resolve a relative path to their own copy.

```text
<state>/
  findings/<id>.json      one array of findings per reviewer
  reports/<id>.md         one report per reviewer
  probes/<id>/            scripts that show a finding; validators' own under their id
  validated/<id>.json     verdicts, matched to findings by the finding's id
  issues/<nn>-<theme>.md  drafts for the lead agent
  exp/<id>/               one folder per experiment
```

The state folder is working space and is lost with the machine. What must outlive the
session goes where the Agents section of `CLAUDE.md` says the record is, including
which agents ran and what each returned in a line. A session that takes over a task
starts from the tracking issue, not from a state folder.

## The path of a review

1. **Plan.** The lead agent divides the code into groups of modules and gives each
   reviewer one group and one lens (the lenses are defined in `review.md`). An issue
   under the task's epic states the scope and the method.
2. **Review.** Reviewers work blind to earlier audits, probe what they suspect, then
   check their candidates against `notes/DECISIONS.md`, `notes/QUESTIONS.md`,
   `notes/reviews/` and the issues, and score each finding.
3. **Validate.** A validator that did not produce a finding re-reads the cited lines,
   traces one dependency further than the claim does, and reproduces bugs and
   measurements with its own script. Its verdict is one of `confirmed`,
   `confirmed-with-corrections`, `refuted`, `already-decided` or `not-verifiable`.
4. **Draft.** Drafters group the surviving findings by theme into issues, with the
   validators' corrections applied and the refuted findings listed with reasons.
5. **File.** The lead agent reads every draft and files it. A briefed agent may post a
   draft that the lead agent has read, verbatim, when told to, and nothing else.
6. **Rule.** Josh decides which items to act on. Nothing is fixed before that.
7. **Fix.** Each accepted item, or group of items, is a change under `CLAUDE.md`: an
   issue, a branch, a regression test that fails without the fix, the gate, a review.
8. **Record.** A report, posted as a comment on the review's epic, or on its issue when
   it has none, lists what was reviewed, the counts by verdict, and what was refuted or
   dropped and why.

## The path of a research direction

1. **Sweep.** Scouts cover the literature by area, retrieving every record they cite.
   Other scouts generate directions from the obstacles, working apart from each other.
2. **Rank.** The lead agent ranks the directions and opens an issue for each one
   pursued: the hypothesis, the novelty verdict with its sources, the predicted outcome
   and the test that would falsify it.
3. **Test.** An experimenter runs the falsification test at matched cost against the
   package's current behavior, over seeds fixed in advance. A derivation is checked
   symbolically or numerically by an agent that did not write it.
4. **Decide.** A direction that fails is closed with its negative result. One that
   survives is developed in `notes/experiments/`, and enters the public API only after
   Josh has seen its evidence (DECISIONS #69).

# Josh's review: working notes

Josh reviews the whole package before it lands on `main`: the big picture, the statistics,
the couplings and kernels, every implementation choice, the numerics, the API, the tests
and the evidence behind each decision. The aim is for Josh to understand and own all of
it, and to change whatever he disagrees with.

The review runs on an interactive hub (`hub/`, published as a private artifact; see
`ARTIFACTS.md`). It holds the plan of phases and steps, the spec, the decisions and the
experiments, and a shared queue where Josh logs questions, concerns, changes and
decisions. These notes are the durable record of what the hub settles.

| File | What it holds | How it changes |
|---|---|---|
| `hub/steps.py` | The plan: phases, steps, what each step reads and what to challenge | Edited when the plan changes; rebuild and republish the hub |
| `hub/audit2.json` | The second audit's findings, each shown on the steps it concerns | Edited when a finding's status changes; rebuild and republish the hub |
| `hub/build.py` | Builds `hub/review-data.json` from the repository at HEAD | Run before each republish |
| `hub/index.html` | The hub page | Republished from this path to keep its URL |
| `ARTIFACTS.md` | The hub's URL | Append when a page is published |
| `LOG.md` | Dated log: what was reviewed, discussed and changed, with commits | Append only |
| `DECISIONS.md` | Every outcome settled in the queue (keep, change now, change later, revisit) | Append; edit only the status column |

## The queue

- Josh logs an item on a step: its kind (question, concern, change, decision, note) and
  when to handle it (next time we talk, at the phase's checkpoint, or when he asks to
  clear the queue).
- When Josh says "clear the queue" (or names a phase), Claude reads the open items with
  the `ArtifactData` tool, answers each in the item's `response` field with a proposed
  outcome, runs any experiment an answer needs, and sets the item to `answered`.
- Josh settles each item in the hub. Settled items are copied to `DECISIONS.md`, and
  changes are made in commits that cite them.
- Open questions from development (`notes/QUESTIONS.md`) start in the queue, answered
  with their defaults.

## Rules for a new session

1. Read `LOG.md` (the last entries), then the queue's open and answered items in the hub.
2. Do not reopen a settled item unless Josh does.
3. After code changes, rebuild the data (`python notes/review/hub/build.py`) at a pushed
   commit and republish the hub, so its links and excerpts match the code.

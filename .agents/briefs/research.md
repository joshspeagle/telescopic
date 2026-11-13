# Brief: literature scout and idea generator

`telescopic` runs MCMC as
pairs of coupled chains and returns unbiased estimates with finite-sample error bars
(Glynn & Rhee 2014; Jacob, O'Leary & Atchadé 2020). It currently implements published
methods: random-walk Metropolis, MALA, HMC with a meeting step, independence
Metropolis-Hastings, Gibbs and mixtures, each with its coupling, in NumPy.

The aim is for the package to become
the reference implementation of coupled and, more broadly, finite-time MCMC, able to
target the hardest problems: complex, high-dimensional, ill-conditioned, multimodal or
hierarchical distributions. Published methods are the starting point, not the boundary:
the track innovates in theory, algorithms, implementation and computation.

Known weaknesses of the current package, measured in `notes/QUESTIONS.md` and
`notes/experiments/`:
- Coupled HMC and MALA with automatically tuned steps rarely meet on a banana-shaped
  target, and fail on some log-concave posteriors started wide.
- Starts far from the target leave replicates averaged before their chain has arrived,
  and variance estimates become heavy-tailed.
- Meeting times grow with dimension and conditioning; multimodal targets give meeting
  times that are exponential in the barrier.
- Everything is NumPy on CPU; there is no autodiff, GPU or compiled backend.

Read first: `reports/Unbiased MCMC with couplings landscape.md` (the census of the field), the files under `research_notes/`, `docs/theory.md`,
`notes/QUESTIONS.md`, and the part of `docs/design/spec.md` that lists what was deferred.
Do not repeat the census; build on it, correct it where it is wrong, and go past it.

## Hard rules

- Read-only on tracked files: do not edit them, do not run git commands that change
  state, do not write to GitHub.
- Write only under the task's state folder, `<state>`, whose absolute path your
  assignment gives, in the file your assignment names.
- Do not spawn subagents.
- **Citations.** Cite a paper only if you retrieved its record in this session (arXiv
  listing, DOI page, journal page). Give authors, year, title and the arXiv identifier or
  DOI, and the URL you read. If you recall a paper but could not retrieve it, list it
  under "unverified recollections" and say so. Never invent an identifier. State what the
  paper shows in the modality its text supports (suggests, shows, proves), and say
  whether it is a preprint or peer-reviewed.
- **Novelty claims.** "Open" means you searched for it and found nothing; say which
  queries you ran. Hedge accordingly.
- Any number you report comes from a source you read or a script you ran, with the path.
- Python, if you need it: import the package with `PYTHONPATH=src`, threads capped
  (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`). Probes stay under five minutes.

## What a research direction looks like

Each direction you propose or assess is recorded as:

- **Name** and a two-sentence statement of the idea.
- **Mechanism:** why it should work, with the mathematics as far as you can take it.
  State the invariants it must keep: each chain's marginal transition exactly that of
  the uncoupled kernel (or another route to exact unbiasedness, stated), and chains
  stay equal once they meet (or the estimator that does not need this).
- **What is new** against the nearest published work, with those citations.
- **Predicted outcome**, quantitative where possible (for example, how the meeting time
  should scale with dimension or condition number).
- **Falsification test:** the cheapest experiment whose result would show the idea
  fails, and the target it runs on.
- **Risk** and what it costs to find out.
- **Scores** from 1 to 5 for novelty, plausibility, tractability, payoff on hard targets
  and testability.

Your own assignment and output file are in the message that sent you here. Your final
message is a summary of at most 30 lines; the detail goes in your output file.

Open your final message with your status: done; done with concerns (state them);
blocked (give the reason); or in need of context (name exactly what is missing). If you
are blocked, say so at once and do not guess.

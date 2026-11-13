# Brief: experimenter

`telescopic` runs MCMC as pairs of coupled chains and returns unbiased estimates. The research track tests constructions that could
make it work on harder targets. You run one experiment. Its purpose is to find out
whether a direction deserves development, so an honest negative result is as useful as
a positive one. A result that looks good because of a chosen seed, a favorable target
or an unmatched budget is worse than no result.

Read first: `.agents/briefs/research.md` (the track's context and rules, which apply to
you too), then the files your assignment
names, `notes/QUESTIONS.md`, and the package's own experiments in `notes/experiments/`
for the house style: a script with a header that states the question and the result,
and its captured output beside it.

## Rules

- Read-only on tracked files. Write only under your experiment folder,
  `<state>/exp/<your-id>/`, where `<state>` is the task's state folder, whose absolute
  path your assignment gives.
  No git commands that change state, no GitHub writes, no subagents.
- Use the package where it does the job (import it with `PYTHONPATH=src`): its kernels, `Mixture`, `Target`, the engine, the
  estimators, `telescopic.testing.check_coupling`. A new kernel or coupling is a small
  class in your folder that subclasses or follows the package's interfaces; read
  `docs/guide/custom-kernels.md` and one shipped kernel first. Do not copy package code
  and edit it when a subclass or a composition does.
- Every new coupled step must pass `check_coupling` (marginals against the uncoupled
  step, staying together after meeting) before any timing or meeting-time result counts,
  and you must state the argument for why each chain's marginal transition is exact.
- Threads: prefix runs with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
  The machine's cores are shared with other agents. Keep any single run under ten
  minutes and the whole experiment under about ninety minutes of compute; size the
  design to that and say what you cut.
- Statistics: seeds from `numpy.random.default_rng`, fixed in advance and listed. At
  least 5 independent seeds per configuration for any comparison of meeting times, with
  the spread reported. Every estimate comes with its Monte Carlo standard error, and
  every claim of unbiasedness with z-scores against a known or independently computed
  truth. Compare methods at matched cost, counted in evaluations of the log density and
  of its gradient (the package's `Target` counts them), not in iterations.
- Baseline first: reproduce the current package's behavior on the same targets with
  the same budget, so the comparison is yours, not quoted.
- State the falsification test from your assignment at the top of your report, and say
  plainly at the end whether the result passed it, failed it, or is inconclusive.

## Output

In your experiment folder (scripts and outputs) and your final message (the report):
- `<name>.py` scripts, each runnable as is, with a header docstring that states the
  question, the design and (filled in at the end) the result; no machine-specific paths
  inside the scripts other than through `PYTHONPATH`.
- `<name>.out` beside each script: its captured output, including the raw table of
  numbers (configuration, seed, unmet pairs, meeting-time quantiles, evaluations,
  estimates with standard errors and z-scores).
- The report, at most 200 lines, as your final message after the summary (the lead
  agent saves it): the hypothesis and falsification test; the design;
  the tables; what the evidence supports and in which modality (suggests, shows); what
  was not tested; threats to validity; the verdict; and, if the direction survives, the
  smallest next experiment and what an implementation in the package would have to
  change (which modules, which interfaces).

Your final message is a summary of at most 25 lines (the verdict, the two or three
numbers that carry it, and what you could not verify), followed by the report.

Open your final message with your status: done; done with concerns (state them);
blocked (give the reason); or in need of context (name exactly what is missing). If you
are blocked, say so at once and do not guess.

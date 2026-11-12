# The tune-up: review, documentation study and research portfolio

A report of phase 10 (tracking issue #65), for the package at `4da2a81`. It records what
was reviewed and how, the counts by verdict, where each finding was filed, and what was
refuted or dropped and why. Nothing found has been fixed: Josh rules on the issues first.

## The review (#66)

### Method

Sixteen reviewers, each given one group of modules and one lens, worked apart and blind
to the earlier audits, probed what they suspected, and then checked their candidates
against `notes/DECISIONS.md`, `notes/QUESTIONS.md`, `notes/reviews/` and the spec's
decision table. Eleven validators, none of which had produced the findings it checked,
then tried to refute or reproduce each one from the code alone, with their own scripts
for every bug, statistical claim and timing. Drafters grouped the surviving findings by
theme into issues, with the validators' corrections applied; the lead agent read the
drafts and filed them. The briefs are in `.agents/briefs/` (pull request #78).

The machine had 4 cores shared by up to twenty agents, at load averages of 8 to 16, so
every performance figure in the issues is a ratio of interleaved repeats, a profile
share or a count of operations, with the load beside it, not an absolute time.

### Reviewers and verdicts

| Reviewer | Lens | Scope | Tier | Findings | Confirmed | Corrected | Already decided | Refuted |
|---|---|---|---|---|---|---|---|---|
| c1 | correctness | `couplings`, `kernels/base`, `random_walk`, `independence`, `mixture` | default | 5 | 5 | 0 | 0 | 0 |
| c2 | correctness | `kernels/mala`, `hmc`, `gibbs`, `target`, `initial`, `_linalg` | default | 6 | 3 | 3 | 0 | 0 |
| c3 | correctness | `engine`, `_random`, `_parallel`, `_progress`, `_types` | default | 2 | 1 | 1 | 0 | 0 |
| c4 | correctness | `estimators`, `measure`, `results`, `diagnostics`, `warnings` | default | 6 | 2 | 2 | 2 | 0 |
| c5 | correctness | `sampler`, `tuning`, `_io`, `plotting`, `examples`, `testing` | default | 7 | 4 | 3 | 0 | 0 |
| q1 | quality | the kernel layer, `couplings`, `target`, `initial`, `_linalg` | default | 7 | 5 | 1 | 1 | 0 |
| q2 | quality | `engine`, `_random`, `_parallel`, `sampler`, `tuning` | default | 8 | 2 | 5 | 0 | 1 |
| q3 | quality | `estimators`, `measure`, `results`, `diagnostics`, `_io`, `plotting`, `examples`, `testing` | default | 10 | 4 | 4 | 1 | 1 |
| s1 | statistical audit | estimators, couplings, bounds and the recipe against the source papers | strongest | 4 | 0 | 4 | 0 | 0 |
| p1 | performance | a measured profile of the whole package | default | 8 | 2 | 5 | 1 | 0 |
| t1 | tests | the suite, with 107 mutants in a private worktree | default | 19 | 12 | 7 | 0 | 0 |
| a1 | architecture | kernels, couplings, state and array backends | default | 6 | 2 | 3 | 1 | 0 |
| a2 | architecture | the engine, pools, storage and resumption | default | 6 | 0 | 6 | 0 | 0 |
| a3 | architecture | the public API, used as a new user would | default | 18 | 8 | 9 | 1 | 0 |
| i1 | infrastructure | packaging, CI and repository hygiene | fast | 19 | 9 | 8 | 1 | 1 |
| l1 | compliance | the code against the spec, the test plan, the decisions and `CLAUDE.md` | default | 18 | 11 | 6 | 0 | 1 |
| | | | | **149** | **70** | **67** | **8** | **4** |

All 149 findings were validated.
Among those confirmed, with or without corrections, 20 are important and
117 minor after validation; none is critical. The validators added 11 items of
their own, of which 8 were confirmed and 3 refuted.

### What held

- **The statistics.** The statistical audit (#84) derived each claim from first
  principles and tested it: the weights of the lag-L estimator against the papers'
  equations on 5,488 grids with no mismatch; unbiasedness of every kernel family against
  exact truth, with a largest |z| of 2.31 over 54 tests at 20,000 to 60,000 replicates;
  the marginals and faithfulness of each coupling over 30,000 to 2,000,000 pairs; the
  bounds against their source; the recipe and the independence of the tuning, pilot and
  main streams.
- **Reproducibility.** A replicate's output was bitwise identical across lane counts,
  index orders, merged runs, process pools, a thread pool and a resumed run after an
  interrupt, for every kernel family tried.
- **The estimators' arithmetic.** The signed measure and everything computed from it
  matched brute-force recomputation to about 1e-14, with censored pairs present and with
  atoms shifted by up to 1e12.
- **The tests.** 107 new mutants: 82 killed by the fast suite, 25 survived, of which 5
  are equivalent or not worth a test; the validator re-applied each surviving mutant
  and got the same result. The statistical tests passed on 170 of 170 runs with ten other seeds
  each, so no seed looks tuned. Coverage is 98.44%.

### What was found

| Issue | Content |
|---|---|
| #71 | Run-time defects: a vectorized log-density that returns a 0-d value for one point stops every run; a NaN from an `Independence` proposal's density is rejected silently and can stall a run; a function that reuses its output buffer corrupts tuning; ten smaller ones |
| #72 | Input validation, as seven conventions with their instances |
| #73 | What a user learns when a run goes wrong: zero-density starts reported only after the run, no gradient check, warnings without a next step, the memory estimate against the measured peak |
| #74 | Docstrings, the test plan and the records where they disagree with the code |
| #75 | Measured costs in memory and time in pools, results and kernels |
| #76 | Simplifications that change no behavior |
| #77 | Design: proposal couplings as objects under one Metropolis-Hastings kernel, and a state that can carry auxiliary variables |
| #79 | Design: reduction in the engine, runs that survive a failure, saved plans and replicate ranges; budget mode reopened |
| #80 | Design: the public API before a first release |
| #81 | Packaging, CI and repository hygiene, with machine paths in two tracked research notes first |
| #82 | Sixteen decisions for Josh, each with the existing text, the evidence, the options and a default |
| #83 | The performance profile: lazy imports halve the import time; row-wise products dominate from about 50 dimensions |
| #88 | Tests: 18 tests to add for behavior that 20 surviving mutants leave unchecked, the first being variance and quantiles of a run with censored pairs, and that a replicate draws its two starts from its own stream; each draft test was run against the code and against its mutant |
| #102 | Found by an experiment of the research track and reproduced: with a tuned `Independence` component, a run on a two-mode target meets within about 100 steps and estimates a mode weight of 0.7 as 1.0002 ± 0.0003 or 0.0002 ± 0.0003, with no warning that names the cause |
| #84 | The statistical audit: the unbiasedness guarantee omits the source paper's convergence assumption, with a counterexample that returns 1 for a truth of 0.5 without a warning; and the record of what the audit verified |

### Refuted, already decided, or dropped

- **Refuted.** Rebuilding `_parallel._gather` on `concurrent.futures.wait` (q2-01): a
  cancelled future is not reported done until the executor dequeues it, and the
  rewrite raised `BrokenExecutor` in 173 of 200 trials where the shipped code returned
  the finished chunk in 200 of 200. Consolidating the research notes and their stubs in
  `docs/` (i1-11): the stubs are includes of one copy. The Gibbs docstring's build-time
  figure (l1-17): it is measured, in `notes/reviews/m8.md`. Deriving the saved array
  names from the dataclass fields (q3-06): a test already walks the fields, and the
  explicit list guards the file's schema. Among the validators' own items: that
  `correction_coefficients` costs 14% of a one-lane run (5% without the profiler, 0.2%
  for its checks); a degenerate color scale in `plotting.corner` (not reachable from a
  `Result`); `couplings.discrete_maximal` as dead code (the spec requires it).
- **Already decided**, and carried to #82 where the evidence bears on the decision:
  half-open histogram bins (spec 3.4); normal quantiles in intervals (spec 3.3), where
  two independent measurements now support Student-t; whitened MALA (DECISIONS #66,
  raised twice); the slow suite on manual dispatch (implementation plan 6); a separate
  JAX engine (spec D-1); the split of diagnostic values and rules (DECISIONS #40); the
  two uses of `interval` (spec 7.1, 7.2).
- **Dropped.** An infinite two-scale threshold (a valid coupling); `distance_owner` as
  unread (it is the key that makes the merged distance order independent of chunk
  completion); moving `averaging_times` (it would cost an allocation where a length
  suffices); the tuned scale in kernel descriptions (DECISIONS #53); the shared tuning
  streams of a `Mixture`'s components (confirmed, and harmless).

### What the validation changed

Validation was not a formality: 67 of the 149 findings validated needed
their statement, figures, severity or fix corrected. Line numbers were wrong in several
findings and in many of the documentation audit's. Two findings grew: the buffer-reuse
defect reaches tuning through the public `init_state` and `step`, and the silent NaN
rejection in `Independence` leaves a chain that never moves. Several shrank: the
"closed" `State` is not a bug today, since every shipped kernel's Markov state is its
position; the first-class-coupling prototype reproduced three kernels bitwise but not
the maximal coupling of the random walk.

## The documentation and identity (#67)

- **Reference study.** Of six of Josh's packages, binference, brutus, GSPICE and DOROTHY
  have a custom theme, a logo that encodes the method, a palette in which each color has
  one meaning, generated or executed figures in one style, and (three of them) live
  explainers that are complete as still images; dynesty and FlexMESA are close to their
  stock themes. What each practice is and which site it comes from is in #87.
- **Audit.** The site at `4da2a81` builds without warnings in 272 s. Of 31 statements and
  snippets checked against the code, none was wrong. The findings are about reach and
  form: #85 (22 defects in the rendered site), #86 (20 pages and 16 figures the site
  lacks) and #87 (a proposed structure by kind of page, with a migration in nine steps
  that each leave the build passing).
- **Identity.** A design board with five logo concepts and three style sheets, shown on a
  mock landing page with a live pair of coupled chains, awaits Josh's choice. The choice
  of theme departs from spec section 11 and decision D-17; #87 lists it among its
  rulings.

## The research track (#68)

Four literature sweeps, three independent rounds of idea generation with probes, three
feasibility experiments, and a referee who re-derived the eight leading claims, reran
four probes and searched for prior work. The portfolio, the benchmark suite and one
issue per direction are filed under #68.

- **Build now:** contraction-first tuning of coupled HMC (an acceptance target of 0.99
  with a step jitter that both chains share took unmet pairs on the banana from 475 of
  500 to 0 of 500, while failing the experiment's 5-fold criterion against a hand-tuned
  step at a ratio of 2.70); a frozen global independence component with weight
  diagnostics, which must warn that it cannot see a mode its builder missed; transformed
  targets with a pushforward start; front ends that supply gradients.
- **Develop:** builders of the global proposal from nested sampling and sequential Monte
  Carlo; learned transport maps for coupling in a latent space; exact meeting for
  gradient kernels without a random-walk step; the burn-in quantile for wide starts;
  coupled locally adaptive HMC; estimators that need no meeting; re-tuning between
  batches.
- **Stopped, with the figure that stopped each:** lag-first tuning (inefficiency up by
  a median factor of 2.2 on matched starts and 5.8 to 18 on wide ones); a gradient-based
  meeting step as the fix for HMC (461 of 500 unmet against 466); coupled tempering with
  a shared reference draw, whose probe accepted swaps with the inverse of the Metropolis
  ratio (2 of 900 pairs met once the ratio was corrected); compression of signed
  measures as a new result (the existing `thin` matches it at equal memory).
- **What the referee corrected.** A claimed meeting time independent of dimension grows
  like log d; a claimed law for the meeting time of coupled independence samplers is
  already published; a diagnostic said to predict heavy meeting tails is blind to a
  missed mode.

The experiments' scripts and outputs are not yet in `notes/experiments/`: each is read
and committed with the first pull request of its direction.

## Process

- **Model tiers** (DECISIONS #70). The strongest tier was used twice: for the
  statistical audit and for the referee of the research portfolio. Surveys of the
  reference sites, the infrastructure review and the posting of reviewed drafts used the
  fast tier; everything else the default tier.
- **What did not work.** Twenty agents on four cores made timings unreliable for an
  hour, which is why the issues report ratios. A probe behind one research direction
  simulated the wrong chain and was caught only by the referee, so an idea's own probe
  is not evidence until a second agent has read its script. The platform that posts to
  GitHub strips the attribution footer from issue bodies.
- **Not done.** No reviewer ran the slow suite, a process pool on Windows or macOS, or
  the Python 3.14 worker path. The research novelty verdicts rest on searches of arXiv
  abstracts and the web, not of full texts.

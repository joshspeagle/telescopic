# Open questions for Josh

Each item has the default that work proceeds on until Josh answers.

- [ ] **Licence.** Default: MIT, as emcee, dynesty and pastry use.
- [ ] **When to make the repository public.** JOSS asks for about six months of public history. Default: stays private until Josh decides.
- [ ] **Budget mode (spec §3.7).** It is a should-level feature, and an unbiased per-lane
  estimator needs per-replicate weights throughout `Result`. Default: deferred beyond
  v0.1 (DECISIONS #22); an interrupted run and its completion cover stopping early.
- [ ] **The size of a user's pool.** Pools that do not report it through `_max_workers`,
  `_processes` or `size` (such as dask's executors) are sized by the CPUs this process may
  use, so a cluster's extra workers can sit idle. Default: keep the guess. Alternative: an
  argument for it, as dynesty's `queue_size`.
- [ ] **The automatic steps of HMC and MALA.** Dual averaging toward
  acceptance 0.8 suits ordinary HMC, but its coupled chains then rarely meet on the
  banana from U[-5, 5]^2: 88 of 100 pilot pairs were unmet after 10^5 iterations, so
  `sample(kernel="hmc")` cannot choose k there. A target acceptance of 0.95, or a meeting
  scale of half the step, left 59 to 94 of 100 unmet at 20,000; steps near 0.02 couple
  (mean meeting time 160), and on the Gaussian every variant meets within a median of 32
  (`notes/experiments/m9_hmc_meeting.out`). Choosing the step from pilot meeting times, as
  Heng & Jacob (2019) advise, would be a new tuning procedure (DECISIONS #1). Default:
  keep the defaults, which couple on log-concave targets, and document that other targets
  need an explicit step; the error of `choose_k` already says to give a better-tuned
  kernel. The automatic MALA fails the same way on the banana: more than 1% of its pilot
  pairs were unmet at 10^5 (`benchmarks/results/acb12f9.json`). It can fail on a
  log-concave target too: on a logistic regression with 200 outcomes, started ten times
  wider than the fit, 19 of 100 pilot pairs were unmet at 3,000 and `sample` raised. Of
  200 chains at the tuned step, 21 never moved; they started where the curvature was
  higher (the largest eigenvalue of ΣH a median of 3.5, against 2.1 for the others and 1.2
  at the mode). A step of 0.7 met every pair, and the automatic HMC met them all
  (`notes/experiments/m10_mala_logistic.out`).

- [ ] **Starts far from the target.** From a π0 a hundred times wider than the posterior,
  or far from it, the recipe k = q_0.99 leaves about one replicate in a hundred averaged
  while its chain is still on its way, and the estimates of variances become heavy-tailed:
  on the README's line fit from N(0, 10I), 11 of 20 runs warned and some estimated a
  negative variance (`notes/experiments/m9_initial_width.out`; Sherlock reports the same in
  his discussion of JOA20). Two remedies are new constructions (DECISIONS #1): a π0 built
  from the tuning pilot's mean and covariance, widened, which loses the protection of a
  wide π0 against missed modes; or k at a higher quantile or a multiple of it, which costs
  more. Default: keep the recipe, recommend a π0 a few to ten times wider than the target
  (spec §4.2, DECISIONS #35), and rely on the variance warning.

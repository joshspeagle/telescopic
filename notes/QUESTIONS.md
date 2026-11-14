# Open questions for Josh

Each item has the default that work proceeds on until Josh answers.

- [ ] **Budget mode (spec §3.7).** It is a should-level feature, and an unbiased per-lane
  estimator needs per-replicate weights throughout `Result`. Default: deferred beyond
  v0.1 (DECISIONS #22); an interrupted run and its completion cover stopping early.
  Josh ruled on issue #79 (2026-11-13): prototype it as an experiment, then decide.
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
  Josh ruled on issue #89 (2026-11-13): an opt-in tuning that puts contraction first
  (issue #91), which becomes the default if its pre-registered test passes.

- [ ] **The gradient check: on by default, and a third round? (issue #123).** The
  second pre-registered experiment passed for the noise-limited design (N): no warning in
  32,700 checks of 109 correct families, and a warning in every check of 46 harmful wrong
  gradients, at 40 evaluations of `log_prob` and 8 of `grad` at any dimension
  (`notes/experiments/x123/RESULT.md`). Its adversarial review weakened what that
  supports (`review/REVIEW.md`): (N) also warns on gradients that are approximate by
  design (a solver, numerical differences, single precision behind a float64 return, a
  factor of 1.005, a prior left out), harmless wherever their harm was measured, and it
  is blind from a
  start at a mode, to an error away from the starts, above about 1e9 in |`log_prob`| and
  with a float32 `log_prob` in high dimension. No floor computed on a line separates the
  harmful from the harmless: every relative floor that silences part of the harmless
  group also loses warnings on harmful ones. Two rulings are asked (comment of
  2026-11-14 on #123). **1. What the package does when a sampler is built with `grad`:**
  (a) (N) runs by default, warns with the size of the disagreement and is turned off by
  `check_grad=False`; (b) the check is opt-in; (c) the design with the sampler's floor
  (H) by default and (N) on request. **2. Whether the record needs a third round before
  the check ships** (wrong families with a float32 `log_prob`, a point start at the mode
  and a held coordinate; the after-pilot design (T)). Default, which the change for #123
  builds: (a), and no third round, with the review's limits documented from its
  measurements, a warning when the check compared no line, and (T) as a research item
  with its own registered test (DECISIONS #159).

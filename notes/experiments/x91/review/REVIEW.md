# Adversarial review of the pre-registration x91 (2026-11-14)

**Verdict: sound to post after the listed changes.** The arms are exact and the guard is
reproducible from its text, but three things would make a verdict wrong or undefined as
written (B1 to B3), and several criteria cannot do what sections 5 and 8 say they do.
No registered seed was used; no pilot pair or replicate of any arm was run. Probes and
outputs are in the review folder `adv/`: `p1_guard.py/.out`, `p2_claims.py/.out`,
`p3_power.py/.out`, `p4_mala.py/.out` (tuning only, probe seed 900). Line numbers are
those of `PREREGISTRATION.md`.

## Blocking

### B1. U, pooled at 1%, passes an arm whose runs raise, and its stated power is wrong (section 5, lines 211 to 216; section 7, 271 to 274; section 9, 361 and 368 to 370; rule 1, 317 to 319)

- `choose_k` raises as soon as 2 of a seed's 100 pairs are unmet. U allows 30 of 3,000.
  So an arm can fail `k` at 1 or 2 of 30 seeds (3 to 7% of user runs raise the error of
  #137) and hold U. Section 5 computes U's power as $1 - (1 - p)^{30}$, which assumes
  that one failing seed breaks U, and line 369 says "one failing seed leaves more than 30
  pairs unmet". The data at the package's cap say otherwise: the six failing seeds on
  record at 100,000 left 31 (#137, seed 0), 14 and 96 (`i137_default_cap.out`), 54, 23
  and 12 (`i137_reproduce_unequal.out`): three of six are at or below 30. "10 to 99"
  (line 212) is also wrong at the smaller cap: the stores hold failing seeds with 4, 5,
  5 and 8 unmet (`p2_claims.out`, part d).
- Recomputed (`p3_power.out`, failing seeds drawn from those six counts): at
  $p$ = 0.01, 0.02, 0.05, 0.10 U fails with probability 0.14, 0.28, 0.59, 0.86, not
  0.26, 0.45, 0.79, 0.96; and with probability 0.12, 0.18, 0.20, 0.10 some seed raises
  while U holds.
- For G at d = 10, `n_steps=10` this is the deciding cell (rule 1 refuses G on it). The
  right per-tuning rate at the package's cap is 2 of 100 (factor above 0.9998; #137
  found failing seeds at 0.99986 and above, passing at 0.99982 and below), not "0.02 to
  0.04": the 0.999 count is the criterion of the cap of 10,000. The expected pooled
  unmet fraction is then about 0.02 x 0.38 = 0.8%, beside the 1% threshold, and U fails
  with probability about 0.28, not "about 0.5". Whether G is "refused" is a draw, and a
  pass would not show that G is free of the failure either.
- U also sees nothing below about 0.9998: at the package's cap the resonance shows
  mostly as a huge `k` (`HMC(n_steps=3)`, d = 10: `k` 12,377 to 52,065 at 8 of 10 seeds
  with no pair unmet). In the cells where P has unmet pairs M does not apply, so there
  an arm with `k` in the tens of thousands passes everything.

Smallest change: (a) add a criterion K to U on every pilot cell: `k` is chosen at every
seed but at most 1 of 30 (the record already holds it, line 273); (b) for G, add a
tuning-only test, which costs no pilot: over 300 new tunings per resonant cell and for
B (about 0.4 s each), the fraction whose exact largest factor after the guard exceeds
0.9998 is at most 1%; (c) in the cells where M does not apply, judge the tail: the
geometric mean over seeds of `k` times $g$ of an arm is at most a stated multiple (say
3) of the smallest among the arms that pass K; (d) correct lines 212 to 216 and 368 to
370 with the counts above.

### B2. A "full run" is undefined when `k` cannot be chosen, and unbounded when it can (section 4, lines 142 to 145; parts D2, D3, E1, E2)

`run(100)` raises when the pilot fails, and with `max_iter=None` it "runs every pair
until it meets" (`sampler.py`, `run`). The registration predicts unmet pilot pairs for
P and J5 on D3 and for MP and MJ5 on E1 and E2 and makes no prediction for P and J5 on
D2. At a seed with 0 or 1 unmet pilot pair `k` is the 99th finite time, possibly near
100,000, so 100 replicates of at least 11`k` iterations run with no cap: hours or
longer, in a tier that "is never cut" (line 502). Someone will then stop a run by hand,
after seeing it, and which seeds enter Z and V becomes a choice made after the data.

Smallest change: state that a full run whose pilot leaves any pair unmet, or whose `k`
exceeds a fixed bound (for example 20,000), records the pilot only; that Z and V use
the seeds with replicates and report their number; and that V needs at least half of
the cell's seeds in each arm or is void.

### B3. The time-based cut makes "default candidate" easier to reach when arms fail slowly (section 11, lines 496 to 503; rule 2, 320 to 322)

Recorded seconds grow with unmet pairs (a failing pilot runs to the cap), so failures
in tier 1 push C2 and C4 over the 6 hours, and those are cells where the registration
itself predicts cost failures (C4: CF fails V, J5 $\rho_V > 1.1$; C2: CF fails M). Rule 2
asks for a pass "on every cell where the table applies them" and says nothing about a
cell that was cut. As written, a slower experiment removes criteria an arm could fail.

Smallest change: an arm with a cut cell on which a criterion applies cannot be a
default candidate; its verdict is "undecided until the cut cells run". Or move C2 and
C4 into tier 1 and cut E and F only.

## Important

### I1. CF's fate on M is set by the meeting weight, and section 8 claims more than section 12 allows (lines 337 to 341, 531 to 533, 544 to 545)

With the registration's own model, $\tau \approx 1/\gamma + 6/(-\ln f)$, the wait
$1/\gamma = 20$ iterations is paid at CF's 4 to 5 leapfrog steps against P's 1 to 2:
$\rho_M \approx 3$ before any data. #91's evidence of +17% (119 to 139 gradients on
the logistic regression) was CF at $\gamma = 0.2$ against the package at 0.05: 139
gradients are about 14 iterations of 2 x 5, and at $\gamma = 0.05$ the same contraction
gives about 29, a ratio near 2.4. So the +17% came from the meeting weight, not from
the tuning, and the registered comparison at one $\gamma$ is the fair one. But line 337
says a cost failure "answers #89's question 1 with no", while line 532 says it "does
not refute that configuration". Smallest change: keep the arms, and either add CF and
P at `meeting_weight=0.2` on A default, C5 and D1 (pilots only, a few minutes),
judged by M against each other, or reword line 337 to "no for CF at the package's
meeting step".

### I2. G is tested on the cells its bound and fallback were chosen on (section 10, lines 454 to 457; rule, 332 to 335)

This repeats the error that decision row 197 and i137's notes record for the band.
$\delta$ was chosen among six values on the tunings of the d = 2 and d = 10 cells of
part A, scored with exact frequencies that the guard cannot read and with a failure
threshold (0.999) learned from i137's meeting times; the fallback was set on B's own
cell. New seeds of the same cells are a second sample, not a held-out test. Smallest
change: add held-out pilot cells for G and the jitters, for example
`examples.gaussian(5)` with `n_steps` 4 and 7 and the unequal scales with
`geomspace(1, 1/3, 6)`, 30 seeds each, U and K applied.

Also on the guard (`p1_guard.out`): my implementation from the text alone agrees with
`guard.py` on all 990 stored and probe tunings (action and step), and every count of
section 10 and of `post.out` reproduces, including 89 of 90 steps equal to the old
guard's and the three seeds with medians 98.5, 211.5 and 394.5. But the fallback moves
3 of 30 default tunings of B onto a resonance that P was not on (seeds 0, 1, 16: exact
factor 0.990 to 0.997 before, above 0.9999 after), raises the factor at 7 of 30, and
at 3 of 30 lengthens the trajectory from 4 to 23 steps. At the threshold that matters
at the package's cap it leaves 3 of 30 against P's 2; line 457 quotes only the 0.999
count (4 against 7). State both, or make the fallback "keep the step", which gives P's
records there. Line 333's rule also drops G for a given covariance with a given
`n_steps` if B's default cell fails, although the predictions give G a pass there:
decide B's three cells separately.

### I3. The cost criteria: stated standard errors and powers do not hold everywhere they are used (lines 217 to 236, 275 to 286, 418 to 423)

- M's premise, "its standard error is small on these cells" (line 279), holds on 5 of
  the 9 non-resonant cells of A (0.015 to 0.022 in log) and fails on the other 4, where
  P's spread between seeds is 0.16 to 0.69 and the error 0.03 to 0.13
  (`p2_claims.out`, part c). The ratios there are far below 1, so no verdict turns on it,
  but line 219's "0.06 to 0.09 on the Gaussians" is wrong. On D1 no spread is stated at
  all, and the prediction (every jitter within a factor 2 of P, line 423) is wider than
  the threshold 1.20 that M applies there, so the implied verdict "J3 or J2: default
  candidates" (line 447) does not follow from the predictions.
- C1's power of 0.997 at a ratio 1.3 reproduces, and is 0.84 if the two spreads, each
  from 5 seeds, sit at the upper end of their 95% intervals (`p3_power.out`).
- J5 on A: with i137's ratios as the truth, J5 passes M on all 9 cells with probability
  0.90; d = 10 with default `n_steps` and with 2 (1.145 and 1.166) carry the risk.
- V's power on the banana is overstated. The registration resampled five stored values;
  with lognormal seeds at the same log standard deviation (2.6 for P), V fails an arm
  4 times costlier with probability 0.30 (arm as spread as P) or 0.48 (arm at J5's
  0.39), not 0.84; with the robust spread 0.9 it is 1.0. The honest statement is
  "between 0.3 and 1.0 at a factor 4". On C2 (10 seeds, P's log spread 3.8) V fails a
  4-fold arm with probability 0.27: V cannot fail there, and section 5 does not say so.
- M fails on a point estimate and V only on a one-sided 95% bound: the burden of proof
  is opposite in the two criteria. Say so, with the reason.

### I4. U at an iteration cap is not a matched budget (line 271; choice 6, line 553)

At 100,000 iterations CF gets 11 to 16 leapfrog steps per iteration on the banana and
25 to 35 on the d = 20 banana, against P's 4 to 5 and 7 to 9 (`probe_tune.out`): 3 to 4
times P's gradients on the cells of CF's purpose. #91's own test is 2 x 10^5 gradients
per pair. The record already counts pairs above that cost (line 265): judge U at both
budgets, at no extra run.

### I5. Exactness checks and Z: no rule for a chance failure (lines 130 to 137, 287 to 296)

- Section 3 runs 56 checks of 4 p-values at 1e-3: a correct kernel set fails one with
  probability up to about 0.2, and the rule is "fixed or dropped". Add: a failed check
  is repeated once at seed 9702 with 80,000 pairs and counts only if it fails again.
- Z: 136 scores by my count (84 in C, 16 in D2 and D3, 36 in E; 18 are G's copies of
  P), family-wise 0.059 under normality: the stated figures hold. But the starts of D3
  and E1 give heavy-tailed variances (#91 reports it), and a failure there refuses CF
  on its purpose cell. Add a confirmation on fresh seeds (9800 on) before "refused".
- Voiding a cell when P's $\lvert z\rvert$ exceeds 3.5 is safe against chance (0.3% per
  six-function cell) but leaves the arms untested there while they can still be default
  candidates. Replace the void by the z-score of the difference between the arm's and
  P's pooled estimates, at the same level, which needs no reference.
- Line 287 says Z applies to "every full run"; the table (306 to 308) leaves J5 on D2
  and D3 out; rule 1 says "anywhere". Make them agree.

### I6. The MALA arms have no reference code (lines 110 to 126)

For HMC the text names `i137_common._JitterLeapfrog`, and I checked it against the
exactness argument: one uniform per pair, tiled over both chains, the count fixed by
the nominal step, each chain's products in separate calls, so marginals and equality
after meeting hold; dual averaging on it is well defined (4 chains, one uniform per
chain and step; line 65's "each of its 1,000 steps draws its own $u$" should say per
chain). For MALA the package's `reflection_maximal`, `_mean` and `_log_alpha` take one
factor for all rows, so a step per pair needs new code that the text does not fix.
Smallest change: commit the runner and the kernels before the first registered seed
and name that commit in a dated note; the order of writing is then checkable, which
i137's notes ask of this test.

## Minor

- Line 384: P at B's default is predicted to fail `k` at 1 to 5 of 10 seeds, but the
  registration's own tunings (2 of 30 above 0.9998) give no failing seed in 10 with
  probability 0.50. P's intervals in part A cover 93 to 99% of outcomes at #137's
  rates: they cannot fail. Both are references, not tests; say so.
- Lines 426 to 430: CF's prediction on D3 (`k` below 1,000) rests on #91's figures at
  $\gamma = 0.2$; at 0.05 the model adds 15 iterations of wait per meeting.
- D2 judges CF's main purpose on 5 tunings: a failure that depends on the tuning at a
  rate of 0.1 is seen with probability 0.41. Ten seeds for CF there cost little.
- Step 5 of the guard (line 99) does not say which step wins a tie in $\lvert s\rvert$;
  `guard.py` takes the smaller.
- Line 293: the level 3.5 replaces #91's 3; section 10 says it is a Bonferroni level but
  not that it departs from the issue.
- `p4_mala.out`: dual averaging toward 0.99 gives MALA steps of 0.34 (Gaussian, d = 10),
  0.36 (logistic) and 0.054 (banana from the box) against 1.10, 1.18 and 0.32: MCF is
  well defined, and its $\rho_M \ge 2$ on E3 is safe (the drift contracts 10 times
  less per step). Part F's 15 minutes are plausible at the funnel's measured throughput.

## Checks that found nothing

The guard's text against its code (990 of 990); every count of section 10; the jitter's
marginal and faithfulness argument against `_JitterLeapfrog` and `_Leapfrog._trajectory`;
that G never acts with the tuned covariance and the default `n_steps` (at $w = 1$,
$n\theta \le 2.09$ for two or more steps, at least 1.05 from $\pi$, and a single step
triggers only above $\varepsilon = 1.95$); the count and family-wise
rate of Z; M's power on C1.

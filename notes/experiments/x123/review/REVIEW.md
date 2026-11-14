# x123, second experiment: adversarial review (#123)

This is the adversarial review of the second experiment of the gradient check
(`../PREREGISTRATION.md`, `../RESULT.md`). It was made on 2026-11-14 at commit d922879 by
an agent that did not produce the result, from the code, the registration, the scripts
and their saved outputs alone. Its own runs were given seeds 500,000 to 550,019, which
no one had used; the checks and the harm runs read 500,000 to 541,299, and one script
seeds generators up to 608,002 (`README.md`). None of it is pre-registered: its families
were chosen while looking for failures, so its rates are measurements on those families
and not a test. Every number below names the file that holds it; a file without a path
is in this folder, and `README.md` lists them all.

## Verdict: weakened

The registered numbers are real and the procedure was followed, but "passes all six
criteria" says less than it sounds, and two of its readings do not survive tests outside
the registered families.

**The weaker statement that stands.** With the registration fixed first and every count
reproducing, (N) raised no warning in 32,700 checks of the 109 registered correct
families and warned on every check of 44 registered harmful families of group C and of
2 harmful formula checks (`recount.out`); its cost and byte-identity criteria hold
(`../cost.out`, `../integration.out`); (H) is silent on the 11 registered harmful
families its formula says it misses. This replicates, on fresh seeds, what the
exploration had already measured on the same families with bit-identical code. It
supports: "(N) is silent when `grad` is the derivative of `log_prob` as computed and
`log_prob` is smooth at about 1e-5 of the start's spread, and it warns whenever the two
disagree along the start's spread by more than about 2e-8 |log_prob| nats (1.5e-2
|log_prob| for a float32 `log_prob`)". It does not support "no false alarms on correct
gradients", "detects harmful wrong gradients", or a difference in kind between (N)
passing and (H) failing.

## Objections, with evidence

### 1. The detection "pass" of (N) is by construction of the criteria

Criterion 2 tests only families predicted at 0.999 or above, and criterion 4 only those
below 0.001 (`../PREREGISTRATION.md`, section 7). Three registered harmful families sit
between and enter no bar: (N) warned on 518, 476 and 210 of 1,000 checks of F0, F1 and
F2 (`../run_checks_F.out`), whose `Lambda` is 10.3, 5.9 and 3.9 (`../harm_summary.out`).
F3 (797 of 1,000) is "unclassed" only because its standard error is 22.1, although its
wrong gradient left 499 of 500 pairs unmet against 1 for the baseline
(`../harm_summary.out`). Under these rules a design fails detection only if its formula
is wrong or predicts below 0.001. A fair summary is: (N) warns on every check of 46
harmful families and on 21 to 52% of the checks of 3 more; (H) warns on every check of
36, on no check of 11, and on 40 to 98% of the checks of 2 (`recount.out`, last lines).

### 2. "The outcome is the registered prediction" is a replication

Every formula check had been measured in the exploration, with the prototype that
`../equivalence.out` shows bit-identical to the frozen code. The table of section 5 of
`../design/DESIGN.md` lists F0, F1, F2 and F3 (measured 0.544, 0.452, 0.268 and 0.816)
and the table of its section 9 lists F5 (0.988). `../design/det_run.out` and
`../design/diag_r.out` hold most of group C and the whole table of group B with the
relative floors. `../equivalence.out` ran all 171 registered families at 10 seeds each
on the frozen code before the registration, and prints the counts of (N) and of (H) for
each; section 10 of the registration records smoke runs of the harm of all 53 baselines.
Of the 109 families of group A, 92 are those on which the prototype was revised twice
(`../design/DESIGN.md`, section 6: "These seeds chose the revisions"). All of this is
disclosed. It means that the result shows no overfitting to seeds; it does not show that
the formula holds on new kinds of families.

### 3. Group A leaves out the cases where the design expects false alarms, and they alarm

Section 6 of `../design/DESIGN.md` lists where the rate is not negligible;
`../design/fa_reach.out` has the periodic target at 23 of 100 and 100 of 100; line 173
of `../families.py` still defines `fringes`, and no registered group uses it. Measured
here with correct gradients, 200 seeds each:

- a periodic term, by the number of periods across the start: 1e4 and below, no warning;
  3e4, 2 of 200; 1e5, 48 of 200 (24 of 200 at d = 5); 1e6, 200 of 200 (`fa1.out`);
- a sinusoid fit with a wide start in frequency: no warning up to 1.6e4 fringes across
  the start, 3 of 200 at 4.8e4, 59 at 1.6e5, 200 at 1.6e6 (`fa1.out`, `fa3_kepler.out`);
- a piecewise-linear table at spacing 1e-5: 143 of 200 (the registered table has spacing
  0.01 and is silent); a piecewise-constant table at spacing 1e-6 with `grad` 0:
  200 of 200 (`fa1.out`).

(H) warns as often as (N) on the periodic cases.

### 4. The harmless inconsistencies that (N) warns on are far wider than group B

One of them is a gradient that the design itself treats as correct. Warnings of (N),
with those of (H) in brackets:

- **A float32 gradient returned as float64** (through `.astype(float)`,
  `np.asarray(.., dtype=float)` or `.tolist()`) with a float64 `log_prob`: 195 of 200 on
  N(0, I) at d = 5, 200 on N(1e3, I), on a rotated Gaussian of condition 1e4, on the
  funnel and on the banana, 196 on Student t3, 30 on pumps, and none on N(0, I) at
  d = 1,000 and on the logistic regression (`fa4.out`). (H): none, except 3 of 200 on
  N(1e6, I) and 8 and 1 of 200 on the two funnel rows (`fa4.out`). Harm by the registered
  measure: `Lambda` 1.000 (se 0.000) on both families measured (`harm5_b.out`). Group A
  counts float32 gradients as correct only where the dtype is visible, so that the probe
  of a narrow gradient runs; here it cannot run.
- An ODE likelihood by adaptive RK45 with the gradient of the exact solution: 100 of 100
  at a tolerance of 1e-3 (H 81) and of 1e-6, 99 at 1e-8, 63 at 1e-10, none at 1e-12;
  LSODA at 1e-8, 87. Fixed-step RK4 with the gradient of the exact solution: 100 of 100
  at 20 and at 100 steps; with the derivative of the discrete scheme: none
  (`fa2_solver.out`).
- A Newton solve with an implicit-function gradient: 200 of 200 at a tolerance of 1e-2
  (H 64), 72 at 1e-4, none at 1e-8 (`fa3_kepler.out`).
- `scipy.optimize.approx_fprime` forward differences as `grad`: 198, 194 and 200 of 200
  on N(0, I), Student t3 and the banana (H none) (`fa5.out`).
- A marginal likelihood by 40 Gauss-Hermite nodes with the exact gradient: 50 of 50
  (`fa5.out`). A `log_prob` rounded to 1e-3: 28 of 200 (`fa1.out`). Float32 arithmetic
  returned as float64 at N(1e4, I): 23 of 200 (`fa3_precision.out`).

The harm was measured only for the float32 case; the others are presumed harmless from
their size.

### 5. Harmful errors that (N) never sees, none of them registered

Detection in 200 seeds, with the correct twin silent in every case (`miss_detect.out`),
and harm by the registered measure (`harm5_a.out` to `harm5_d.out`):

| Wrong gradient | (N) detects | `Lambda` (se), class |
|---|---|---|
| N(0, I) d = 5, point start at the mode: wrong sign | 0 | 6.71 (0.90), harmful |
| logistic n = 1e4, point start at the posterior mode: wrong sign | 0 | 7.11 (0.75), harmful |
| start sd 2 with coordinate 0 held at 0: sign of coordinate 0 | 0 | 3.58 (0.15), harmful |
| start sd 0.1: right for \|x\| < 1, wrong sign outside | 0 | 4.19 (0.45), harmful |
| start N(5, 0.1^2): wrong sign only for \|x\| < 3 | 0 | 3.26 (0.18), harmful |
| every coordinate of the start equal: wrong sign orthogonal to (1, .., 1) | 0 | 17.8 (1.4), harmful |
| start on a 2-dimensional subspace: wrong sign in the other 3 directions | 0 | 11.3 (0.5), harmful |
| N(0, I) - 1e10, start sd 2: sign of coordinate 0 (at -1e9: 68) | 0 | 14.1 (1.7), harmful |
| half-normal, start on the boundary: wrong sign | 0, no line compared | 4.02 (0.23), harmful |
| **float32 throughout**, N(0, I), start sd 2: sign of coordinate 0, d = 200 / 1,000 / 10,000 | 196 of 200 / 37 of 200 / 0 of 100 | 5.94 (0.33) at d = 50; 8.5 to 8.7 for the registered float64 twins at d = 1,000 (`../harm_summary.out`) |

The float32 row matters most: every registered wrong family has a float64 `log_prob`.
The registered formula with its float32 constant does predict it: check rates of 0.979,
0.191 and 0 (`f32pred.out`) against 0.98, 0.185 and 0 measured. Group A has point starts
at 0 as correct families (A67 to A71); group C has no wrong gradient from such a start.
A coordinate held at 1 or at 1e-3 is detected 200 of 200 times, so the fallback
direction works (`miss_detect.out`). Two misses on a hierarchical model, with every
group effect started at the grand mean, could not be classed: the baseline left 145 to
474 of 500 pairs unmet (`harm5_c.out`).

### 6. A check that compares nothing is silent

From a start on a boundary, 1,600 of 1,600 lines were unusable and 200 of 200 checks
compared no line; from a start mostly outside the support, every point was skipped
(`fa3_starts.out`). `../run_checks.py` counts such a check as silence. It did not occur
in groups B, C or F (`recount.out`: 0 unusable lines and 0 skipped points there).

### 7. The harm measure is loose in places, and no verdict depends on it

The reclassing is in `recount.out`.

- The factor 2 with 4 degrees of freedom gives a one-sided bound of 94%, not 97.7%. With
  the t quantile 2.78, two families become unclassed (scales (1, 0.01 x4) from N(0, I),
  and grad + 2 from sd 1e4), both lower bounds with 372 and 91 unmet pairs. Requiring
  all 5 per-seed ratios above the bar unclasses one family that (H) misses (one seed at
  1.92). (N) still warns on every check of the 41 to 43 families of group C that remain
  under criterion 2; (H) still misses 10 or 11.
- Capping can only lower `Lambda` (the baselines left no pair unmet, and one family 1),
  so it cannot make a harmless family harmful; it does make F3 unclassed.
- The ratio depends on the start: the same wrong sign of one coordinate gives 12.4 from
  sd 2, 1.43 ("unclassed") from a random point (C45) (`../harm_summary.out`), and 2.28
  from the mode (`harm5_a.out`).
- The proxy's bar of 3 assumes an overstatement of 1.5; the validation shows up to 1.63.
  The smallest family classed by the proxy has 5.02 (0.12), still above 2 after dividing
  by 1.63. Sixteen families of C and F at d >= 1,000 rest on the proxy alone
  (`recount.out`, last line), and one more on the equivalence with one of them; the
  proxy is validated only to d = 200.

### 8. Wording in `RESULT.md`

"Every output's header prints commit 7b2c992" is true of the 62 outputs of phase 2;
`../predictions.out` and `../equivalence.out` print 32ae70b (`recount.out`, first
lines). They were made in phase 1, and `src` is identical between the two commits
(`git diff 32ae70b 7b2c992 -- src tests` is empty). The summary should carry the three
harmful families at 21 to 52% beside "400 of 400".

## Checks that found nothing

- **Registration and procedure.** The 10 hashes of section 10 match (`sha256sum` on the
  frozen files); `git diff 7b2c992 d922879` only adds the 62 outputs and `RESULT.md`;
  there are 62 commands with one complete output each, whose headers and commands match
  (`recount.out`); the modification times of the outputs in the author's working folder
  were consistent with one run each. Four registered commands, run again into a scratch
  folder, gave the registered outputs apart from the times and the commit of the header
  (their outputs are not kept, being identical). They are the following.

  ```sh
  python notes/experiments/x123/run_checks.py F
  python notes/experiments/x123/run_checks.py B
  python notes/experiments/x123/run_checks.py A 0 40
  python notes/experiments/x123/harm.py package 0 1
  ```

- **Recount** with an independent parser: every count of `../RESULT.md` reproduces;
  `../harm_summary.out` has 67 rows and no mismatch (`recount.out`); the table of
  predictions of the registration equals `../predictions.out` (`predtable.out`).
- **The two generators read early** (380000 and 381000) could not inform a choice: one
  gives tuned steps, the other a marginal check that passed and was registered again on
  381500.
- **Code against registration.** `../gradcheck.py` matches section 2 line by line. All
  109 gradients of group A agree with fourth-order differences of `log_prob` (about
  1e-13 on smooth float64 families); the wrong families carry exactly the stated errors;
  `jacobi_gradient` is the derivative of the computed `log_prob` (`gradcorrect.out`).
- **The claim about the relative floors is upheld, and extends.** On fresh seeds with the
  numbers of every line (`floors_collect.out`, `floors_analyze.out`), 8 statistics by 4
  rules (any, two, half, all lines): silencing the 8 warned families of group B loses at
  best 27 of 46 harmful families. For 6 of the 8, a harmful family with a tight start is
  smaller on every statistic (best Youden index 0.00). Only the Jacobi solver at 1e-5
  separates: a floor of 8.4e-5 nats over u loses none of the 46.
- **Correct gradients, no false alarm** (200 seeds each unless stated, so a rate under
  1.5% each): Laplace at d = 3e4, 1e5 (100 seeds) and 1e6 (20 seeds); a hinge-loss
  classifier and an L1 regression to n = 1e6; a ReLU network regression to n = 1e4 by 50
  units (`fa2_kinks.out`, `fa3_big.out`); tables at spacing 1e-4 and coarser to
  d = 5,000, with a largest ratio of 0.81, a margin of 1.2 (`fa1.out`, `fa2_kinks.out`);
  fresh or fixed noise (`fa2_noise.out`); |log_prob| to 1e300 and d = 1 (`fa1.out`);
  d = 1e6 (`fa3_big.out`); lattice and 0/1 starts (`fa3_starts.out`); visible float32
  gradients on 11 new targets, float16 and longdouble (`fa3_precision.out`); an ODE with
  the derivative of its discrete scheme (`fa2_solver.out`); a root by bisection
  (`fa3_kepler.out`); a marginal by `quad` with its default tolerances (`fa5.out`).

## What was not checked

HMC's harm; `../cost.py`, `../integration.py`, `../predictions.py` and the other harm
runs (not run again); the harm of the solver, quadrature and finite-difference
inconsistencies; the harm with a float32 `log_prob` above d = 50; the files of the first
experiment; design (T). The zero counts bound a rate only at 1.5% (3% for Laplace at
d = 1e5, where the design expects about 1%).

## What the review would tell the lead

**On the default.** The evidence supports turning (N) on only with a message that states
what it means, and not as a test that separates harmful gradients from harmless ones. In
a user's terms:

- If `grad` is the exact derivative of the `log_prob` code and that code is smooth, the
  check stays quiet: no warning in about 50,000 such checks. The figure is a round one,
  below the total of the 32,700 registered checks and of the 36,090 checks that the rows
  without a warning of `fa1.out`, `fa2_*.out` and `fa3_*.out` and the correct twins of
  `miss_detect.out` add up to, about 3,000 of which are on inexact gradients that stayed
  silent.
- It will warn routinely, on harmless gradients, for anyone whose `log_prob` comes from
  an adaptive ODE solver, a Newton or quadrature step looser than about 1e-8, whose
  gradient is numerical, or whose gradient is single precision behind a float64 return.
- It will also warn for a wide start on a strongly periodic or finely tabulated
  `log_prob`.
- Silence is weak evidence: it says nothing when chains start at a mode or a fixed
  point, when the error lives away from the start, when |log_prob| exceeds about 1e9, or
  with a float32 `log_prob` above a few hundred dimensions.

**On what a warning should mean.** No floor computed on a line can decide harm; the
reanalysis here confirms the author's argument and counts. The honest meaning is "`grad`
and `log_prob` disagree by X nats across the spread of your start", and the size belongs
in the message. If one tier is wanted, the threshold of (H) is the only one with a
derivation: it is quiet on nearly all the harmless cases above (not on a wide start, a
periodic `log_prob`, or a solver at a tolerance of 1e-3) and it misses the tight starts.
Before a ruling the review would ask for three things: wrong-gradient families with a
float32 `log_prob`, a point start at the mode and a held coordinate added to the record;
the package saying when no line was compared; and the after-pilot design (T), the only
one aimed at the tight-start case.

## Corrections made when this record was prepared

The review was first reported in a message, and five of its statements were corrected
when its numbers were traced to the files above. None changes the verdict.

- Objection 1 said that (N) warns on every check of 44 harmful families and (H) of 35.
  The counts over groups C and F together are 46 and 36 (`recount.out`, last lines): the
  first figures left out the 2 harmful formula checks on which (N) warns every time, and
  the harmful point-start family (1, 1), on which (H) warned 400 of 400 times.
- Objection 7 said that fourteen families rest on the proxy alone. There are sixteen, 12
  of group C and 4 formula checks (`recount.out`, last line).
- Objection 4 said that (H) warns on none of the float32 gradients returned as float64
  except 3 and 8 of 200. It also warns on 1 of 200 on the second funnel row (`fa4.out`).
- Objection 4 gave "100 of 100" for fixed-step RK4 with the gradient of the exact
  solution without saying that it was measured at 20 and at 100 steps; the row at 1,000
  steps never ran (`fa2.py`, header).
- The report said that all its seeds were 500,000 to 550,019. The seeds of the checks
  and of the harm runs are within 500,000 to 541,299, but `gradcorrect.py` seeds NumPy
  generators at 500,000 + 1,000 k + (0..2) for the 109 families of group A, which reach
  608,002. A later registration should count 500,000 to 608,002 as used.

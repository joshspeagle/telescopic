# x123, second experiment: result (#123)

Phase 2 ran the registered procedure of `PREREGISTRATION.md` at commit 7b2c992 (the
header of each of its 62 outputs prints `commit 7b2c992`, not dirty; `predictions.out`
and `equivalence.out` are outputs of phase 1, at 32ae70b, whose package source is the
same), one process at a time on one core, with the registered seeds. Every number below
is in the `.out` files beside this file. The adversarial review of this result, which
weakens what its "passes" supports, is in `review/REVIEW.md`.

## Summary

- **(N) passes all six criteria.** No false alarm in 32,700 checks on 109 correct
  families; 400 of 400 on every wrong family that is harmful and that its formula puts at
  0.999 or above (43 families of group C, and 1,000 of 1,000 on two formula checks); every
  formula check within 0.05 of its prediction; no predicted miss on a harmful family.
  Three harmful families that its formula puts between 0.2 and 0.55 face no detection
  bar, only the check of the formula: (N) warned on 518, 476 and 210 of 1,000 of them.
- **(H) fails criterion 2**, as registered (criterion 4): 0 warnings on each of the 8
  harmful families of group C where its formula predicts a miss (starts tight in the
  wrong coordinate, or matched to the narrow scale), and 0 of 1,000 on the 3 harmful
  formula checks it predicts at 0. It passes criteria 1, 3, 5 and 6.
- **The outcome is the registered prediction.** Neither design behaved in a way its
  formula did not predict: the 14 formula comparisons are within 0.023 of the model.
- **Group B (reported, no bar):** (N) warns on 8 of the 9 gradients that are not the
  derivative of `log_prob` as computed, all 9 classed harmless (`Lambda` 0.990 to
  1.011).

## The criteria

| Criterion | (N) | (H) |
|---|---|---|
| 1. False alarms, group A: at most 2 of 300 per family, at most 16 of 32,700 pooled | 0 on every family; 0 of 32,700: **passes** | 0 on every family; 0 of 32,700: **passes** |
| 2. Detection: at least 390 of 400 (975 of 1,000) on every harmful family predicted at 0.999 or above | 43 families, each 400 of 400; 2 formula checks, 1,000 of 1,000: **passes** | 35 families, each 400 of 400; **fails by criterion 4** |
| 3. Formula checks within 0.05 of the prediction | 7 of 7 (largest difference 0.023): **passes** | 7 of 7 (largest difference 0.007): **passes** |
| 4. Predicted misses on harmful families, at most 10 of 400 confirm | none on a harmful family (the one predicted miss, the start of spread 1e-3 with `log_prob` near -1e4, is classed harmless) | 8 families of C, 0 of 400 each; 3 formula checks, 0 of 1,000 each: **confirmed** |
| 5. Cost at `d` = 1,000 and 10,000: at most 60 + 25 evaluations, 20 vectors | 40 + 8, 48 + 16, 56 + 24; peaks 15.4 to 16.1 vectors at `d = 1000`, 13.4 to 14.4 at `d = 10000`: **passes** | the same computation: **passes** |
| 6. Byte-identical runs | 24 of 24: **passes** | 24 of 24: **passes** |
| **Verdict** | **passes** | **fails (criterion 2)** |

Criterion 1: the largest ratio of gap to tolerance of (N) over the 32,700 checks was
0.392 (Laplace at `d = 10,000`), then 0.221 (rotated, condition 1e12), 0.209 (Laplace at
`d = 5,000`, and the piecewise-linear table, both the single-kink bound of the design
document, section 4). No check raised; 956 points were skipped, all on the half-normal
target whose start reaches zero density.

Criterion 6: the check cost a median of 40 evaluations of `log_prob` against 46,526 for a
run of 30 replicates (1.30e-3), and 1.62 ms against 0.39 s.

## Formula checks (1,000 seeds each)

| Family | (N) predicted | (N) measured | (H) predicted | (H) measured | harm `Lambda` (se), class |
|---|---|---|---|---|---|
| `d = 10,000`, start sd (0.01, 2, ...): sign of coordinate 0 | 0.541 | 0.518 | 0 | 0 | 10.32 (0.30), harmful (proxy) |
| `d = 1,000`, start sd (0.003, 2, ...) | 0.464 | 0.476 | 0 | 0 | 5.90 (0.31), harmful (proxy) |
| `d = 5`, start sd (1e-4, 2, ...) | 0.209 | 0.210 | 0 | 0 | 3.94 (0.26), harmful |
| scales (1, 1e-4 x4), start N(0, I): sign of the wide coordinate | 0.805 | 0.797 | 0.804 | 0.797 | 7.2 (22.1), unclassed |
| `N(0,I) - 1e4`, `d = 5`, start sd 1e-3: sign of coordinate 0 | 0 | 0 | 0 | 0 | 1.24 (0.09), harmless |
| `d = 1,000`, start sd (0.5, 2, ...) | 1 | 1.000 | 0.985 | 0.981 | 6.18 (0.28), harmful (proxy) |
| `d = 1,000`, start sd (0.25, 2, ...) | 1 | 1.000 | 0.400 | 0.403 | 5.57 (0.15), harmful (proxy) |

The first three are harmful families on which (N) detects about one check in two, one in
two and one in five, as its formula predicted; they are covered by criterion 3, not by
the detection bar.

## Group C: detection and harm

Each row: warnings of (N) and of (H) in 400, the prediction where it is not at or above
0.999, and the harm. "Proxy" classes with the bar of 3; a `Lambda` with unmet pairs of the
given gradient is a lower bound. All 46 families warn 400 of 400 under (N).

| Family | (H) | Harm `Lambda` (se), class | unmet given / 500 | step ratio | `delta_0` |
|---|---|---|---|---|---|
| `d = 5`: wrong sign | 400 | 57.1 (0.9), harmful | 0 | 0.207 | 2.83 |
| `d = 5`: twice | 400 | 1.14 (0.02), harmless | 0 | 0.921 | 1.42 |
| `d = 50`: twice | 400 | 84.4 (2.0), harmful | 0 | 0.201 | 2.90 |
| `d = 200`: twice | 400 | 197.3 (3.7), harmful | 53 | 0.120 | 4.58 |
| Student t3, `d = 5`: gradient of N(0, I) | 400 | 26.1 (5.1), harmful | 6 | 0.680 | 2.37 |
| equicorrelated 0.9, `d = 10`: gradient of 0.8 | 400 | 2.73 (0.15), harmful | 0 | 0.648 | 0.83 |
| sign of coordinate 0, start sd 1, `d` = 5 / 50 / 200 | 400 each | 4.05 (0.23) / 3.12 (0.12) / 3.14 (0.04), harmful | 0 | 0.56 / 0.73 / 0.82 | 1.28 / 0.82 / 0.65 |
| the same, `d` = 1,000 / 2,000 / 5,000 / 10,000 (proxy) | 400 each | 5.02 (0.12) / 5.65 (0.25) / 7.59 (0.31) / 9.30 (0.32), harmful | 0 | 0.89 / 0.90 / 0.92 / 0.95 | 0.53 / 0.46 / 0.40 / 0.35 |
| sign of coordinate 0, start sd 2, `d` = 5 / 50 / 200 | 400 each | 12.4 (1.5) / 5.02 (0.30) / 4.98 (0.16), harmful | 0 | 0.56 / 0.73 / 0.82 | 1.26 / 0.82 / 0.65 |
| the same, `d` = 1,000 / 2,000 / 5,000 / 10,000 (proxy) | 400 each | 8.72 (0.18) / 9.48 (0.18) / 11.18 (0.13) / 12.95 (0.24), harmful | 0 | 0.89 / 0.90 / 0.92 / 0.95 | 0.53 / 0.46 / 0.40 / 0.35 |
| signs of coordinates 0..9, `d = 10,000` (proxy) | 400 | at least 82.5 (1.0), harmful | 500 | 0.517 | 1.12 |
| coordinate 0 set to 0, `d = 5` | 400 | 2.21 (0.02), harmful | 0 | 0.857 | 0.63 |
| scales (1, 0.1 x4), start N(0, I) | 400 | 12.2 (0.7), harmful | 0 | 0.544 | 1.25 |
| scales (1, 0.01 x4), start N(0, I) | 400 | at least 658 (285), harmful | 372 | 0.655 | 1.00 |
| scales (1, 0.1 x4), start N(0, 0.1^2 I) | **0** (predicted 0) | 2.56 (0.18), harmful | 0 | 0.546 | 1.26 |
| scales (1, 0.01 x4), start N(0, 0.01^2 I) | **0** (predicted 0) | 2.86 (0.22), harmful | 0 | 0.532 | 1.29 |
| start sd (0.1, 2, ...), `d` = 5 / 100 / 1,000 | **0** each (predicted 0) | 4.18 (0.28) / 3.21 (0.11) / 6.01 (0.45, proxy), harmful | 0 | 0.56 / 0.79 / 0.89 | 1.25 / 0.68 / 0.53 |
| start sd (0.01, 2, ...), `d` = 5 / 100 / 1,000 | **0** each (predicted 0) | 3.63 (0.23) / 2.36 (0.10) / 5.91 (0.14, proxy), harmful | 0 | 0.56 / 0.87 / 0.89 | 1.28 / 0.54 / 0.53 |
| grad + 2, start sd 100 / 300 / 1,000 / 1e4 | 400 each | 294 (37) / 239 (25) / 156 (46) / 113 (49), harmful | 41 / 30 / 49 / 91 | 0.20 / 0.20 / 0.21 / 0.24 | 2.87 / 2.86 / 2.73 / 2.43 |
| sign of coordinate 0, start sd 1,000 | 400 | at least 787 (79), harmful | 500 | 0.562 | 1.23 |
| rotated, condition 1e2 / 1e4, `d = 10`: sign along the widest axis | 400 each | 8.68 (0.83) / 9.54 (0.65), harmful | 0 | 0.60 / 0.63 | 1.11 / 1.09 |
| rotated, condition 1e4, `d = 2,000` | 400 | 9.48 (0.18), harmful (by equivalence) | | | |
| float32 grad, sign of coordinate 0: N(0,I) `d = 5` / `d = 1,000` (proxy) / N(1e3, I) `d = 5` | 400 each | 13.1 (1.5) / 8.47 (0.22) / 19.9 (6.2), harmful | 0 | 0.55 / 0.89 / 0.53 | 1.26 / 0.53 / 1.30 |
| plateau, `d = 3`: grad = -x | 400 | 209 (22), harmful | 42 | 0.023 | 24.0 |
| point start (1, 1), `d = 2`: grad + 2 (1, -1) | 400 (predicted 0.9961 for both; z = 1.25) | 3.55 (0.52), harmful | 0 | 0.262 | 2.20 |
| random point start, `d = 5`: sign of coordinate 0 | 335 (predicted 0.834; z = 0.20) | 1.43 (0.29), unclassed | 0 | 0.609 | 1.23 |

No family is undetermined: the baselines left no pair unmet, except one of 500 for the
scales (1, 1e-4 x4). Unclassed: the random point
start (C) and the scales (1, 1e-4 x4) formula check (F), whose given gradient left 499 of
500 pairs unmet against a baseline that met all but one, so its `Lambda` is a lower bound
with a large spread.

**Proxy validation** (reported): on the same six families at `d = 50` and 200, the
proxy's `Lambda` over the package's is 1.14, 1.63, 1.52, 1.14, 1.51 and 1.36 (sign of
coordinate 0 from sd 1 and sd 2, then twice, at 50 and at 200). The proxy's marginal
check passed (smallest p-values 0.114, 0.040, 0.284 against 1.25e-4; equal pairs stayed
equal).

## Group B (300 seeds each; no bar)

| Gradient | (N) | (H) | (N) predicted | (H) predicted | (N) with `rho` = 1e-6 / 1e-4 / 1e-2 | harm `Lambda` (se), class |
|---|---|---|---|---|---|---|
| Jacobi solver, rtol 1e-3, exact gradient | 300 | 0 | 1 | 0 | 300 / 299 / 0 | 1.000 (0.000), harmless |
| Jacobi solver, rtol 1e-5 | 300 | 0 | 1 | 0 | 300 / 0 / 0 | 1.000 (0.000), harmless |
| Jacobi solver, rtol 1e-8 | 1 | 0 | 0.019 (z = -2.0) | 0 | 0 / 0 / 0 | 1.000 (0.000), harmless |
| `d = 5`: grad x 1.005 | 300 | 0 | 1 | 0.0005 | 300 / 300 / 8 | 0.993 (0.004), harmless |
| `d = 1,000`: grad x 1.0001 | 300 | 0 | 1 | 0 | 300 / 280 / 0 | 0.990 (0.008), harmless (proxy) |
| `d = 5`: grad + 0.02 | 300 | 15 | 1 | 0.040 (z = 0.9) | 300 / 300 / 298 | 0.996 (0.006), harmless |
| logistic `n = 1e4`: without the prior | 300 | 0 | 1 | 0 | 300 / 268 / 0 | 1.005 (0.005), harmless |
| the same from start sd 10 | 300 | 300 | 1 | 1 | 300 / 300 / 150 | 1.011 (0.003), harmless |
| `d = 5`, start 100x wide: grad x 1.005 | 300 | 300 | 1 | 1 | 300 / 300 / 10 | 0.993 (0.004), harmless |

## The relative floors (reported)

Group A: 0 warnings at every `rho`. Groups C and F, where a floor loses warnings
that (N) has (warnings in 400, or in 1,000 for F):

| Family | (N) | `rho` = 1e-6 | 1e-4 | 1e-2 |
|---|---|---|---|---|
| sign of coordinate 0, `d = 5,000`, start sd 1 / 2 | 400 / 400 | 400 / 400 | 400 / 400 | 399 / 399 |
| the same, `d = 10,000`, start sd 1 / 2 | 400 / 400 | 400 / 400 | 400 / 400 | 393 / 397 |
| sign of coordinate 0, `d = 2,000`, start sd 2 | 400 | 400 | 400 | 399 |
| scales (1, 0.1 x4), start N(0, I) | 400 | 400 | 400 | 384 |
| scales (1, 0.01 x4), start N(0, I) | 400 | 400 | 381 | 0 |
| scales (1, 0.1 x4), start N(0, 0.1^2 I) | 400 | 400 | 400 | 374 |
| scales (1, 0.01 x4), start N(0, 0.01^2 I) | 400 | 400 | 380 | 1 |
| start sd (0.1, 2, ...), `d` = 5 / 100 / 1,000 | 400 each | 400 each | 400 / 400 / 379 | 160 / 0 / 0 |
| start sd (0.01, 2, ...), `d` = 5 / 100 / 1,000 | 400 each | 400 / 400 / 385 | 159 / 0 / 0 | 0 each |
| grad + 2, start sd 300 / 1,000 / 1e4 | 400 each | 400 each | 400 each | 359 / 43 / 0 |
| rotated, condition 1e4, `d = 2,000` | 400 | 400 | 400 | 122 |
| random point start, `d = 5` | 400 | 400 | 399 | 372 |
| F: start sd (0.01, 2, ...), `d = 10,000` | 518 | 367 | 0 | 0 |
| F: start sd (0.003, 2, ...), `d = 1,000` | 476 | 1 | 0 | 0 |
| F: start sd (1e-4, 2, ...), `d = 5`; scales (1, 1e-4 x4) | 210; 797 | 0; 0 | 0; 0 | 0; 0 |
| F: start sd (0.5, 2, ...) / (0.25, 2, ...), `d = 1,000` | 1,000 / 1,000 | 1,000 / 1,000 | 1,000 / 1,000 | 222 / 0 |

Every family in these rows is classed harmful except the random point start
(unclassed) and the scales (1, 1e-4 x4) formula check (unclassed).

## What failed, was rerun, or could not be classed

- No runner failed and no check raised. Every one of the 62 registered commands ran
  once. Command 62 (`harm.py summarize`) did not start the first time: the script that
  drove the runs built a wrong path for it (an error of the driver, not of a registered
  file); it was then run once as registered.
- Unclassed: the random point start (`Lambda` 1.43, se 0.29) and the scales
  (1, 1e-4 x4) formula check (`Lambda` 7.2, se 22.1). Neither enters criterion 2 or 4.
- Lower bounds: twelve families have given-gradient pairs unmet at 10,000 steps, so their
  `Lambda` is a lower bound (marked "at least" or with unmet counts above).

## What the result supports, and what it does not

- It shows, on fresh seeds and fixed criteria, that (N) raised no false alarm on 109
  correct families (32,700 checks) and warned on every check of the 44 harmful wrong
  gradients, including one wrong coordinate among 10,000 and starts unmatched to the
  target's scales, at a fixed cost of 40 + 8 evaluations and about 16 vectors of memory.
- It shows that both designs' detection formulas predict their rates: 14 of 14 formula
  comparisons within 0.023; the two families of group C predicted between 0.001 and
  0.999 within 1.25 standard errors; in group B the solver at 1e-8 warned 1 in 300
  against 0.019 predicted (z = -2.0).
- It shows that (H) is silent on harmful gradients when the start is tight in the wrong
  coordinate, as its formula predicts (11 families, 0 warnings each).
- It shows the cost of (N)'s meaning: it warns on 8 of 9 group-B gradients, every one
  harmless by the registered measure. Whether a warning should mean "not the derivative
  of `log_prob` as computed" (N) or "large enough to matter" is the open ruling; group B
  and the relative floors are its evidence, and every relative floor that silences part
  of group B (`rho` 1e-4 or 1e-2) also loses warnings on harmful families of group C or F.
- It does not show that (N) detects harmful errors where its formula predicts it cannot:
  starts much tighter than the target in the wrong coordinate at high dimension (the
  formula checks: 0.52, 0.48, 0.21), or starts in a small ball with a large `log_prob`.
- It does not test HMC's harm, targets outside these families, or a `log_prob` that is
  not deterministic; the harm above `d = 200` rests on the proxy, which on the validation
  families overstated the package's slowdown by a factor of 1.14 to 1.63.

# x123, second experiment: the record of the adversarial review

`REVIEW.md` is the review of the registered experiment beside this folder
(`../PREREGISTRATION.md`, `../RESULT.md`), made on 2026-11-14 at commit d922879 by an
agent that did not produce the result. This folder holds the scripts and outputs behind
every number that it cites. Nothing here is pre-registered: the families were chosen
while looking for failures. No registered seed is read. The seeds of the checks and of
the harm runs are within 500,000 to 541,299, of the range 500,000 to 550,019 set aside
for the review; `gradcorrect.py` also seeds generators up to 608,002, so a later
registration should count 500,000 to 608,002 as used. Fixed data sets and noise streams
are built from generators 0, 1, 3, 11 and 777 to 779, as the registered families build
theirs from 0 and 5; these are not seeds of a check or of a sampler.

## How to run a script

From the root of the repository, with the package at commit d922879, each script runs as
follows.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src:notes/experiments/x123 \
python notes/experiments/x123/review/fa1.py > fa1.out
```

The path `notes/experiments/x123` gives the scripts the registered `gradcheck.py` and
`families.py`; `probe.py` and `cases5.py` are found beside the script. The header of each
script gives its arguments. The outputs were produced with Python 3.13.18, NumPy 2.5.4
and SciPy 1.18.1 on x86_64 (Intel Xeon at 2.10 GHz, 4 cores), one process at a time with
the three thread variables set to 1. The last column of a row is its time, which differs
between runs.

## Files

| File | What it is | Seeds |
|---|---|---|
| `REVIEW.md` | The review: verdict, objections with evidence, checks that found nothing, what was not checked, what it would tell the lead, and the corrections made when this record was prepared. | |
| `probe.py` | The helper of the probes: one call of `gradcheck.check` per seed on a family, and a row of counts. | |
| `fa1.py`, `fa1.out` | False alarms on correct gradients: periodic terms, a sinusoid fit, rounded and tabulated log-densities, values of 1e12 and above, d = 1. | 500000..500199 |
| `fa2.py`, `fa2_kinks.out`, `fa2_noise.out`, `fa2_solver.out` | Likelihoods with many kinks; fresh and fixed noise; an ODE likelihood with the gradient of the exact solution or of the discrete scheme. | 500000..500199 |
| `fa3.py`, `fa3_kepler.out`, `fa3_precision.out`, `fa3_starts.out`, `fa3_big.out` | Newton and bisection solves and longer sinusoid fits; mixed precision; lattice, boundary and off-support starts; d = 1e6. | 500000..500199 |
| `fa4.py`, `fa4.out` | Float32 gradients returned as float64, with a float64 log-density. | 500000..500199 |
| `fa5.py`, `fa5.out` | Finite-difference gradients; a marginal by quadrature with the gradient of the exact integral. | 500000..500199 |
| `cases5.py` | The wrong gradients that the check may miss, each with its correct twin. | |
| `miss_detect.py`, `miss_detect.out` | The detection of every family of `cases5.py`, and the silence on its twin. | 500000..500199 |
| `harm5.py`, `harm5_a.out`, `harm5_b.out`, `harm5_c.out`, `harm5_d.out` | The harm of 20 of those families by the registered measure, 5 seeds of 100 pairs each; the header gives the arguments of each output. | 510000 + 10 i + (0..4) for the family at position i of `cases5.M`, within 510000..510424 |
| `floors_collect.py`, `floors_collect.out` | The numbers of every line of the registered families of groups B and C and of the formula checks, on fresh seeds; writes a pickle that is not kept. | 520000..521699, 530000..539099, 540000..541299 |
| `floors_analyze.py`, `floors_analyze.out` | Whether a floor on another statistic of a line, or several lines agreeing, separates group B from the harmful families. | reads the pickle |
| `recount.py`, `recount.out` | The recount of the registered outputs, the commands and headers, and the harm classes under four other rules. | none |
| `predtable.py`, `predtable.out` | The table of predictions of the registration against `../predictions.out`. | none |
| `gradcorrect.py`, `gradcorrect.out` | The registered gradients against numerical derivatives of `log_prob`, and the wrong gradients against their stated errors. | 500000..500002; generators 500010 and 500000 + 1000 k + (0..2) for k = 0..108, up to 608002 |
| `f32pred.py`, `f32pred.out` | The registered detection formula with the constants of float32, against the rates of `miss_detect.out`. | generator 500900 |

## Which outputs were run again

The outputs are those of the review. When the scripts were given their headers and
passed through the repository's lint and format checks, the cheap ones were run again
and compared with the review's outputs, the times aside.

- Run again whole, output unchanged: `fa1.out`, `fa2_noise.out`, `fa3_kepler.out`,
  `fa3_precision.out`, `fa3_starts.out`, `fa4.out`, `fa5.out`, `miss_detect.out`,
  `gradcorrect.out`, `harm5_a.out`, `harm5_b.out`, `harm5_d.out`, `floors_analyze.out`
  (on the review's pickle).
- Run again in part (stopped after 58 seconds), the rows printed unchanged:
  `fa2_kinks.out` (9 of 18 rows), `fa2_solver.out` (4 of 12), `harm5_c.out` (7 of 36
  lines); and `floors_collect.out` for groups B and F (16 of 62 rows, with a pickle
  identical bit for bit on those families).
- Not run again; the script was only reformatted and given its header:
  `fa3_big.out`, and the rows of the outputs above that the partial runs did not reach.
- Made when the record was prepared: `f32pred.out` (the review printed the same lines
  without saving them), `predtable.out` (the review made this comparison without keeping
  its script), and the last five lines of `recount.out`, which count the harmful
  families on which each design warns on every check, on none and in between; the lines
  before them are unchanged.

Three scripts changed beyond their format, with no change to any number. `fa2.py`
dropped the rows of its part "solver" that the review's run never reached. `probe.py`
catches only `GradientCheckError`, as the registered runner does, where it caught every
exception, and `harm5.py` no longer catches any; no output holds an error. And
`floors_collect.py` and `floors_analyze.py` take the pickle's path as an argument.

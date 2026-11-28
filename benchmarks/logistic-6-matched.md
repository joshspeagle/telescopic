# Registered six-parameter logistic comparison

Use the matched Gaussian start for both the frozen covariance-matched
random walk and the shipped jittered HMC. Tune with seed 20261101. Use 100
lag-one pairs per evaluation seed 20261201 through 20261205, with a ceiling
of 200,000 pointwise density and gradient evaluations per pair.

All 20 calibration pairs per method met with the separate tuning-side
seed 20261102. The longest meeting took 57 iterations. Fix k = lag = 60,
m = 120, and use an iteration safety cap of 20,000 for the estimator.
Time ten runs of 100 replicates with tuning-side root 20261104, then use
200 independent evaluation runs of 100 replicates from root 20261206.

Start four uncoupled chains of each frozen kernel at twice the matched
marginal standard deviations. Use reference root 20261103, 10,000
warmup transitions, and 50 batches of 512 draws per chain. Score coverage
against the independently archived Pólya-Gamma reference means. Include
their Monte Carlo uncertainty in pooled z-scores and keep reference costs
separate from tuning and sampling.

```bash
PYTHONPATH=src python -m benchmarks.compare logistic:6 \
  --start matched --pairs 100 --budget 200000 --tuning-seed 20261101 \
  --seed 20261201 --seed 20261202 --seed 20261203 --seed 20261204 --seed 20261205 \
  --runs 200 --replicates 100 --k 60 --m 120 --lag 60 --max-iter 20000 \
  --estimator-seed 20261206 --reference-seed 20261103 --reference-warmup 10000 \
  --reference-batches 50 --reference-batch-size 512 \
  --reference-chains 4 --reference-start-scale 2
```

## Result

The run used source revision `6c46b0d`, took 76.29 seconds, and reached
359,524 KiB peak RSS. The [compressed results](results/logistic-6-matched-v1.json.gz)
contain the per-seed meetings, all estimator records, interval and warning
summaries, and the variance-reference diagnostics.

| Method | Met / attempted | Median meeting evaluations | Intercept coverage | Last signal coefficient coverage | Estimator evaluations including tuning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen random walk | 500 / 500 | 28 | 189 / 200 | 190 / 200 | 2,828,953 |
| Tuned jittered HMC | 500 / 500 | 242 | 181 / 200 | 169 / 200 | 12,060,620 |

All 200 intervals were available per method and estimand. For HMC,
estimated coverage was 90.5% and 84.5%, with binomial standard errors of
2.1 and 2.6 percentage points. Absolute pooled z-scores were below 1
for both methods and estimands. HMC issued no variance warnings and
random walk issued them on 39 of 200 runs. The archive reports how often
the flagged intervals covered. Tail assessment requires at least 500
replicates per run and was unavailable for the 100-replicate runs.

A post hoc sensitivity check rescored the same Student-t intervals
against the independent one-million-proposal importance-sampling means
from the [logistic reference test](../tests/test_benchmark_logistic.py).
Their Monte Carlo standard errors are 0.00030 and 0.00059. In the same
estimand order, random-walk coverage was 189 and 192 of 200, and HMC
coverage was 188 and 196 of 200. The registered coverage counts use
the Pólya-Gamma means. The sensitivity check shows that uncertainty
in those means materially affects the HMC coverage comparison.

Both uncoupled references passed every gate. Random walk used 142,404
reference evaluations and had a maximum R-hat of 1.00189 and minimum bulk
ESS of 2,849. HMC used 555,220 reference evaluations and had a maximum
R-hat of 1.00012 and minimum bulk ESS of 64,278.

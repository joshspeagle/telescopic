# Registered Gaussian meeting-cost comparison

This is the first #90 baseline slice. Use the rotated Gaussian at dimension
100 and its matched initial distribution. Its rotation uses seed 91. Compare
frozen random walk with the analytic target covariance against the shipped
jittered HMC with automatic covariance and step-size tuning. Freeze both
kernels before starting any evaluation pairs.

Use tuning seed 20261101. Use evaluation seeds 20261201 through 20261205,
in order, with 100 independent lag-one pairs per seed and method. The ceiling
is 1,000,000 pointwise evaluations per pair. Charge each evaluated density and
gradient separately, including partial work before a pair is censored. Charge
the fixed HMC setup once. Report quantiles with censored pairs ranked at
infinity and retain each seed's meeting summary.

This slice measures meeting costs. Estimator coverage on the registered suite
has its own run sizes and averaging settings.

```bash
PYTHONPATH=src python -m benchmarks.compare rotated_gaussian:100 \
  --pairs 100 --budget 1000000 --tuning-seed 20261101 \
  --seed 20261201 --seed 20261202 --seed 20261203 --seed 20261204 --seed 20261205
```

## Result

The registered slice completed at runner revision `b8babf0`. The compressed
JSON contains all 1,000 raw pair records, seed summaries and environment
metadata.

| Method | Met / attempted | Median evaluations | 99th percentile | Tuning evaluations | Sampling evaluations | Total |
|---|---:|---:|---:|---:|---:|---:|
| Frozen random walk | 500 / 500 | 1,107 | 3,583 | 0 | 657,813 | 657,813 |
| Tuned jittered HMC | 500 / 500 | 420 | 1,248 | 423,140 | 240,890 | 664,030 |

HMC used fewer evaluations per meeting. Its one-time tuning cost brought the
total above the frozen random walk for these 500 pairs.

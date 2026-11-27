# Registered equal-width mixture comparison

Use the two-dimensional equal-width Gaussian mixture with separation 6 and
its matched Gaussian start. Compare frozen covariance-matched random walk
with shipped tuned jittered HMC. Use tuning seed 20261101 and 100 lag-one
pairs per evaluation seed 20261201 through 20261205, capped at 100,000
pointwise density plus gradient evaluations per pair.

With the disjoint calibration seed 20261102, the largest observed meeting
iteration across the two methods was 157. Set k = lag = 160 and m = 320
for both methods, with an iteration safety cap of 20,000. Draw 200 independent
runs of 100 replicates from root seed 20261206. For the uncoupled reference,
start exactly at stationarity and use 100 batches of 512 transitions with
seed 20261103. Charge tuning once and record reference work separately.
Retain censored pairs in the denominators.

```bash
PYTHONPATH=src python -m benchmarks.compare mixture_equal_6:2 \
  --pairs 100 --budget 100000 --tuning-seed 20261101 \
  --seed 20261201 --seed 20261202 --seed 20261203 --seed 20261204 --seed 20261205 \
  --runs 200 --replicates 100 --k 160 --m 320 --lag 160 --max-iter 20000 \
  --estimator-seed 20261206 --reference-seed 20261103 \
  --reference-batches 100 --reference-batch-size 512
```

## Result

The run used revision `1e03ab9` and finished in 178.21 seconds at peak RSS
of 129,840 KiB. The compressed JSON in
`results/mixture-equal-6-v1.json.gz` contains all raw meeting and estimator
records, references and counts.

| Method | Met / attempted | Median meeting cost | Mean coverage | Left-region coverage | Estimator evaluations including tuning |
|---|---:|---:|---:|---:|---:|
| Frozen random walk | 500 / 500 | 9 | 190 / 200 | 188 / 200 | 6,509,707 |
| Tuned jittered HMC | 500 / 500 | 600 | 189 / 200 | 189 / 200 | 46,862,250 |

All estimation runs completed. The exact left-region probability is
0.69946004. The pooled estimates were 0.69962888 and 0.69624037,
respectively. Absolute pooled z-scores were below 1.18.

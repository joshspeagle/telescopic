# Registered banana comparison

Use the two-dimensional banana with its matched Gaussian start. Compare the
frozen covariance-matched random walk and the shipped automatically tuned
jittered HMC. Freeze the methods after tuning with seed 20261101. Use the
registered meeting seeds 20261201 through 20261205 with 100 lag-one pairs per
seed at the target's 200,000 pointwise-evaluation cap.

The independent sizing pilot used seed 20261102 with 20 pairs per method.
The largest observed meeting iteration across both methods was 539. Fix
k and lag at 540, m at 1080, and the estimator safety cap at 20,000
iterations for both methods. This fixes a shared averaging window and keeps
the estimated run within the measured execution budget. Use 200 independent
runs of 100 replicates, spawned from evaluation seed 20261206. Do not condition
the results on completed pairs.

For the uncoupled reference, start each frozen kernel with an exact banana
draw and use 100 contiguous batches of 512 transitions. Use seed 20261103 and
report reference cost separately. Count each pointwise density or gradient
evaluation once and charge the HMC tuning once.

```bash
PYTHONPATH=src python -m benchmarks.compare banana:2 --pairs 100 --budget 200000 \
  --tuning-seed 20261101 --seed 20261201 --seed 20261202 --seed 20261203 \
  --seed 20261204 --seed 20261205 --runs 200 --replicates 100 \
  --k 540 --m 1080 --lag 540 --max-iter 20000 --estimator-seed 20261206 \
  --reference-seed 20261103 --reference-batches 100 --reference-batch-size 512
```

## Result

The registered slice completed at runner revision `8b07df0`. The compressed
JSON in `results/banana-coverage-v1.json.gz` includes the per-seed meeting
summaries, all pair records, the replicate estimates for every run, and
evaluation counts.

| Method | Lag-one meetings | Median meeting cost | First-coordinate coverage | Second-coordinate coverage | Estimator evaluations including tuning |
|---|---:|---:|---:|---:|---:|
| Frozen random walk | 500 / 500 | 13 | 187 / 200 | 190 / 200 | 21,797,503 |
| Tuned jittered HMC | 500 / 500 | 2,576 | 192 / 200 | 181 / 200 | 244,482,029 |

All estimator runs completed. Absolute pooled z-scores were below 1.57.
The fixed averaging settings are shared by both methods. Reference-chain
and lag-one measurement costs are recorded separately from the estimator
costs above.

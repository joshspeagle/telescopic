# Warmed variance references for the large logistic and parallax cases

The twofold reference starts that passed for 51-dimensional parallax
motivated the same start width for logistic at dimension 300 and parallax
at dimension 501. The comparison start remained matched. Each reference
used four independent chains and the existing frozen comparison kernel.
The shorter archived random-walk checks used tenfold-width starts,
3,000 warmup transitions, and 12,000 retained transitions per chain.

Candidate variances are returned only if rank-normalized split
R-hat is at most 1.01, minimum bulk and tail ESS are each at least 400,
estimand means agree with independent references within five combined
standard errors, and long-to-half-batch variance ratios lie in
[0.7, 1.4]. All thresholds stayed fixed. Initialization width and run
length changed together.

## Random walk

Both runs used 20,000 warmup transitions and 100 batches of 1,024
retained draws per chain. Each charged 489,604 density evaluations.

| Case | Reference seed | Max R-hat | Min bulk ESS | Min tail ESS | Batch ratios | Variance |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Logistic, dimension 300 | 20261206 | 1.09230 | 39.2 | 142.5 | 1.765, 1.848 | Withheld |
| Parallax, dimension 501 | 20261207 | 1.06940 | 53.7 | 69.7 | 1.787, 1.645 | Withheld |

The batch ratios are for the intercept and last signal coefficient in
logistic, and for the population scale and first stellar distance in
parallax. The logistic means cleared the independent mean check.
The parallax population-scale mean had a combined z-score of −5.14.

The [logistic reference](results/warmed-logistic-300-random-walk-v1.json)
and [parallax reference](results/warmed-parallax-501-random-walk-v1.json)
include settings, timing, and diagnostics. The short meeting and
estimator settings in these files exercised the comparison driver.

## HMC timing

The default covariance pilot uses 900,000 transitions per chain for
logistic and 2,510,010 per chain for parallax. Extrapolation from separate 5,000-transition
timing probes gave 7.6 and 15.8 minutes, respectively, for covariance
tuning alone. Both probes used tuning seed 20261101 and one BLAS thread.


## Logistic HMC reference

The reference used tuning seed 20261101 and independent reference seed
20261208. Each chain ran 10,000 warmup transitions and retained 50
batches of 512 draws. The shipped tuner selected four leapfrog steps
and a nominal step size of 0.32513.

All gates passed. The maximum R-hat was 1.00049, minimum bulk ESS was
21,815, and minimum tail ESS was 36,084. Batch-length ratios were
1.106 and 1.047, and combined z-scores were 0.632 and −0.266, in
estimand order. The estimated transition asymptotic variances were
0.08511 for the intercept and 0.35378 for the last signal coefficient.

Tuning used 3,629,316 density and gradient evaluations. The reference
used 1,095,858 more. The [logistic HMC archive](results/warmed-logistic-300-hmc-v1.json)
records the kernel, diagnostics, and costs. The full comparison driver
took 464 seconds, including tuning, reference diagnostics, and the
short meeting and estimator runs.

## Parallax HMC reference

Four independent reference chains started at twice the matched marginal
standard deviations. Each ran 10,000 warmup transitions and retained 50
batches of 512 draws. The comparison start was matched. The shipped
jittered HMC used its default covariance pilot and tuning seed 20261101.
The uncoupled chains used independent reference seed 20261209.

```bash
PYTHONPATH=src python -m benchmarks.compare parallax:501 \
  --method hmc --start matched --pairs 1 --budget 500 --seed 702 \
  --tuning-seed 20261101 --runs 1 --replicates 4 --k 10 --m 20 \
  --lag 10 --max-iter 100 --estimator-seed 703 \
  --reference-seed 20261209 --reference-warmup 10000 \
  --reference-batches 50 --reference-batch-size 512 \
  --reference-chains 4 --reference-start-scale 2
```

All gates passed. The maximum split R-hat was 1.00072, minimum bulk
ESS was 18,399, and minimum tail ESS was 15,259.

| Estimand | Transition asymptotic variance (kpc²) | Batch ratio | Combined z-score |
| --- | ---: | ---: | ---: |
| Population scale | 0.0121343 | 0.830 | −0.043 |
| First stellar distance | 21.03581 | 1.065 | 0.182 |

Tuning selected four leapfrog steps and a nominal step size of 0.29033.
It took 769.4 seconds and used 10,072,940 density and gradient evaluations.
The uncoupled reference took 62.2 seconds and used 1,096,344 evaluations.
The comparison driver took 840.84 seconds overall, with a peak resident
set of 1,597,936 KiB. Its short meeting and estimator runs used another
500 and 4,476 evaluations, respectively. The five registered meeting
seeds remain held out.

The [parallax HMC archive](results/warmed-parallax-501-hmc-v1.json) records
the kernel, diagnostics, settings, timing, and costs.

## Meeting-cost sizing

Five matched-start random-walk pairs for each case used sizing seed
20261102 and the registered ceiling of 200,000 evaluations per pair.
All ten pairs were censored. Each case incurred 999,995
density evaluations across its five pairs. The logistic check took
162 seconds and the parallax check took 112 seconds with one BLAS
thread per run. If the evaluation cap continues to bind, extrapolating
to 500 serial pairs gives about 4.5 hours and 3.1 hours, respectively.
The five registered meeting seeds were held out.

See the [logistic sizing records](results/logistic-300-walk-meeting-sizing-v1.json)
and [parallax sizing records](results/parallax-501-walk-meeting-sizing-v1.json).

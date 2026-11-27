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

The default covariance pilot uses 900,000 transitions for logistic
and 2,510,010 for parallax. Extrapolation from separate 5,000-transition
timing probes gave 7.6 and 15.8 minutes, respectively, for covariance
tuning alone. Both probes used tuning seed 20261101 and one BLAS thread.

## Meeting-cost sizing

Five matched-start random-walk pairs for each case used sizing seed
20261102 and the registered ceiling of 200,000 evaluations per pair.
All ten pairs were censored. Each method and case incurred 999,995
density evaluations across its five pairs. The logistic check took
162 seconds and the parallax check took 112 seconds with one BLAS
thread per run. If the evaluation cap continues to bind, extrapolating
to 500 serial pairs gives about 4.5 hours and 3.1 hours, respectively.
The five registered meeting seeds were held out.

See the [logistic sizing records](results/logistic-300-walk-meeting-sizing-v1.json)
and [parallax sizing records](results/parallax-501-walk-meeting-sizing-v1.json).

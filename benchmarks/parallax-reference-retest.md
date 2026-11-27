# Warmed parallax variance references at dimension 51

Four independent chains started at twice the matched marginal standard
deviations around the quadrature-based posterior mean. The benchmark start
remained matched. The random-walk kernel remained frozen and jittered HMC
used the same tuning seed (701) as the earlier trial with tenfold-wide
reference starts. The convergence and batch stability thresholds remained
fixed.

The first twofold-start trial, with root seed 704 and 100 batches of 512
draws per chain, passed for HMC. Random walk exceeded the population-scale
batch-variance ratio cutoff of 1.4. The independent check at seed 705 used
the same total number of retained draws in batches twice as long. Random
walk then cleared the batch stability gate and exceeded the R-hat cutoff
of 1.01. The longer random-walk run at seed 706 cleared every gate.

| Kernel | Seed | Warmup per chain | Retained per chain | Max R-hat | Min bulk ESS | Batch ratios | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HMC | 705 | 10,000 | 51,200 | 1.00009 | 60,047 | 1.144, 1.108 | Pass |
| Random walk | 705 | 10,000 | 51,200 | 1.01208 | 574 | 1.185, 1.081 | Withheld |
| Random walk | 706 | 20,000 | 102,400 | 1.00476 | 883 | 1.351, 1.179 | Pass |

The batch ratios are for the population scale and first stellar distance,
in that order. Estimated transition asymptotic variances in squared kpc
were 0.06207 and 12.92931 for HMC, and 5.66350 and 750.40633 for random
walk. Each passing result also cleared the tail ESS and independent mean
checks. The trials changed initialization and run length together, so the
results do not isolate the effect of the reference starting width.

The [independent confirmation](results/parallax-reference-confirm-v1.json)
and [longer random-walk run](results/parallax-reference-long-walk-v1.json)
record the exact settings, source commit, frozen kernels, costs, and
diagnostics. The accompanying short estimator runs exercised the driver.
Registered estimator comparisons still need to be run for this case.

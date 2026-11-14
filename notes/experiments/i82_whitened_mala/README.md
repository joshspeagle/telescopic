# i82: MALA in whitened coordinates

The benchmark that issue #82 (ruling 10, item q1-05) and issue #83 asked for: is a step
of MALA written in whitened coordinates at least 15% faster at 50 dimensions or more,
with the same kernel? Run on 2026-11-14 at commit `9932df7`, on an otherwise idle
machine, after the step costs of issue #83 merged.

## The form

With $L = \varepsilon C$ and $h = L^\top \nabla \log \pi(x)$ the whitened drift, the
proposal is $x' = x + L(h/2 + z)$, one product; the forward residual
$L^{-1}(x' - \mu(x))$ is exactly $z$ and the backward one is $-(z + (h + h')/2)$, so
$\log q(x \mid x') - \log q(x' \mid x) = (\lVert z\rVert^2 - \lVert z + (h + h')/2\rVert^2)/2$.
A step takes 3 products against `MALA.step`'s 7, and a coupled step 7 against 14
(`wm_products.out`). The coupled step reflects whitened vectors; in the package this
would be a primitive of `couplings` that returns the reflection's coefficient, leaving
`reflection_maximal`'s signature and arithmetic, and so RandomWalk's bits, unchanged.
`wm_kernel.py` holds the prototype, a subclass of `MALA`.

## Same kernel (`wm_same_kernel.py`, `wm_suite.out`)

- From the same states and normals, the log acceptance ratios agree to 9.9e-14 on states
  from the target (4.6e-12 where |log q| reaches 3.8e3, 1.2e-15 relative), and no
  accept decision differed in 204,800 single steps or 36,864 coupled pairs.
- `telescopic.testing.check_coupling` passes in 15 of 15 cases; equal pairs stay bitwise
  equal for 500 coupled steps.
- Seeded runs of `sample` at d = 50, 128 replicates, seeds 0 to 4: no meeting time
  differed, and the estimates of E[x_i] by at most 3.4e-15 standard errors.
- The package's fast tests of the modules that build MALA, with MALA replaced by the
  prototype (`wm_patch.py`): 612 pass; the 2 that fail spy on `mala.matvec`, which the
  prototype does not call.

## Timing (`wm_timing.py`, `wm_timing_fine.py`, `wm_noise.py`)

Ratios of the whitened form's time to the current form's, on a vectorized Gaussian
target whose gradient is one dense product, 64 lanes, one BLAS thread, medians (ranges)
of 9 interleaved rounds. The run of record is `wm_timing.out`; `wm_timing_run2.out`
repeats it unchanged, and `wm_timing_fine.out` holds two runs of a finer interleave that
was designed after the first two runs, as a sensitivity analysis.

| Step | d | Record | Replication | Fine 1 | Fine 2 |
|---|---|---|---|---|---|
| coupled | 50 | 0.881 (0.857 to 1.147) | 0.900 (0.792 to 1.222) | 0.893 (0.839 to 0.920) | 0.884 (0.841 to 0.908) |
| coupled | 100 | 0.800 (0.774 to 0.861) | 0.810 (0.632 to 0.914) | 0.815 (0.791 to 0.869) | 0.813 (0.799 to 0.844) |
| coupled | 200 | 0.917 (0.845 to 0.974) | 0.944 (0.782 to 1.086) | 0.902 (0.850 to 0.955) | 0.768 (0.730 to 0.788) |
| single | 50 | 0.846 (0.820 to 0.951) | 0.813 (0.653 to 0.897) | 0.843 (0.802 to 0.876) | 0.820 (0.807 to 0.878) |
| single | 100 | 0.755 (0.716 to 0.763) | 0.751 (0.704 to 0.882) | 0.752 (0.742 to 0.798) | 0.738 (0.720 to 0.776) |
| single | 200 | 0.658 (0.625 to 0.671) | 0.655 (0.616 to 0.753) | 0.660 (0.648 to 0.673) | 0.667 (0.643 to 0.680) |

On the einsum target of `examples.gaussian` the coupled ratios were 0.84 to 0.95 and
the single ones 0.74 to 0.88. One BLAS product at d = 200 took 84.8 to 92.2 microseconds
across 20 blocks on the idle machine (`wm_noise.out`), so a thread's speed drifts by
about 9% between blocks.

## Verdict and ruling

The falsification test fixed before the run ("not worth adopting" if at every d the
coupled step's median ratio is above 0.85 or its range includes 0.85) is met on the run
of record and the replication, and not on the second fine run: on this machine the
outcome is not stable across runs. The medians are stable at d = 50 and 100: the coupled
step is about 11% and 19% faster, the single step 25% to 34% faster at d = 100 and 200.
Ruling 10's threshold, a step at least 15% faster at 50 dimensions or more, is met by
these medians.

Asked on 2026-11-14 with these figures, Josh ruled to adopt the whitened form in the
next phase, as a change he reviews, since it moves the last bits of every seeded run
that steps MALA (decision row 216).

## Files

Each script runs from the repository's root with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:notes/experiments/i82_whitened_mala`.
The scripts were cleaned for the repository's lint after the runs; `wm_products.py` and
`wm_same_kernel.py`, rerun, print the same output.

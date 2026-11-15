# x91: the pre-registered test of issue #91

`PREREGISTRATION.md` is the registration as posted on issue #91 at commit `4f478b9`, and
it is not edited; `PREREGISTRATION.v1.md` is its first version, and `review/` holds the
two adversarial reviews. The other files are the design probes that section 13 of the
registration lists, with their outputs.

To run a probe, start from the repository's root with the package, the shared
experiment modules and this folder on the path:

    OMP_NUM_THREADS=1 PYTHONPATH=src:notes/experiments:notes/experiments/x91 \
        python notes/experiments/x91/<script>.py

The scripts that read the records of issue #137's experiment (`costs`, `evalguard`,
`ineffpower`, `jpred`, `phases`, `post`, `spreads`, `trig`, and `review/p1_guard` and
`review/p2_claims`) read them from `notes/experiments/_rows/`, which git ignores. The
scripts `notes/experiments/i137_*.py` write those stores at their commit, `b0b8f71`
(decision row 197), so these probes rerun only in a checkout where those scripts have
run. `retune.py`, `unequal_tune.py` and `unequal_tune2.py` append to the file given as
their first argument; their outputs here are the `.jsonl` files.

## The registered runs

The code of the runs, committed before any registered seed runs (section 2, Code):

- `arms.py`: the kernels of the arms. P and MP are the package's `HMC` and `MALA`; G
  moves the package's tuned step by `guard.py`; J2, J3, J5, CF, CF.2 are
  `i137_common._JitterLeapfrog` at their widths and acceptance targets; MJ5 and MCF are
  `JitterMALA`.
- `cells.py`: the targets and cells of section 4, in the order of section 11, and the
  logistic regression's reference; `seeds.json`: the seeds of section 5.
- `check_exactness.py`: the checks of section 3, whose output is `check_exactness.out`;
  its last line names the kernels that failed their repeat.
- `run.py`: the runner. It runs the cells in order, each whole, on `--workers`
  single-threaded processes (4 by default), and appends each record to
  `notes/experiments/_rows/x91/<part>.jsonl` as soon as it is done, so that an
  interrupted run resumes where it stopped when started again. It refuses to run with
  uncommitted changes to `src/` or `notes/experiments/`, or beside records of another
  commit. After the run, `--confirm` runs the fresh seeds that a z-score beyond 3.5 needs.
- `score.py`: the scorer, which prints every criterion of section 7 and the verdicts of
  section 8 from the stores, and raises if they mix commits.

From the repository's root, with the environment above:

    PYTHONPATH=src:notes/experiments:notes/experiments/x91 \
        python notes/experiments/x91/check_exactness.py > notes/experiments/x91/check_exactness.out
    PYTHONPATH=src:notes/experiments:notes/experiments/x91 python notes/experiments/x91/run.py
    PYTHONPATH=src:notes/experiments:notes/experiments/x91 python notes/experiments/x91/run.py --confirm
    PYTHONPATH=src:notes/experiments:notes/experiments/x91 python notes/experiments/x91/score.py

`run.py` and `check_exactness.py` set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` to 1 themselves. `run.py --offset N --only <cells> --max-seeds K` runs
the same code on seeds shifted by N, which the registration does not use, into
`_rows/x91_offset<N>/`, and `score.py --offset N` scores them: a way to try the runner
without making a registered record.

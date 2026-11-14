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

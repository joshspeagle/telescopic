# x123, design round: the record of the derivation

Everything in this folder is exploratory. It is the derivation of the two designs of the
gradient check of issue #123, (N) and (H), and the runs that tested the derivation on
seeds 200000 to 219999. The prototype changed twice on those seeds, so no rate here is a
test. The test is the registered experiment beside this folder, `../PREREGISTRATION.md`,
which cites `DESIGN.md` by section and runs on seeds that nothing here used.

## How to run a script

From the root of the repository, with the package at commit 532137c:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src:notes/experiments/x123 \
python notes/experiments/x123/design/formulas.py > formulas.out
```

The path `notes/experiments/x123` gives the scripts the final prototype,
`../prototype.py`; the modules of this folder are found beside the script. The outputs
were produced with Python 3.13.18, NumPy 2.5.4 and SciPy 1.18.1 on x86_64 with 4 CPUs.

## Files

| File | What it is | Seeds |
|---|---|---|
| `DESIGN.md` | The design: what a wrong gradient does to the sampler, the statistic and its invariances, the error model, the detection formulas, the false alarms, the cost, the blind spots, and the proposal for the registration. | |
| `cases.py` | The targets of the first experiment and of its adversarial review that the scripts run on. | |
| `prototype_v0.py`, `prototype_v1.py` | The two superseded versions of the prototype, kept only to reproduce `fa_fa1_v0.out` and `fa_fa1_v1.out`. The final version is `../prototype.py`. | |
| `fa_run.py` | False alarms of (N) and (H) on correct gradients. `fa_run.py correct` wrote `fa_correct.out`, `reach` wrote `fa_reach.out`, `fa1` wrote `fa_fa1.out` and `all32` wrote `fa_all32.out`; `fa_run.py fa1 prototype_v0` and `fa_run.py fa1 prototype_v1` wrote `fa_fa1_v0.out` and `fa_fa1_v1.out`. | 200000..209899 |
| `fa_model.py`, `fa_model.out` | The false-alarm rate by a model of rounding and of kinks, and (N) on Laplace targets at d = 1,000 and 10,000. | generator 218000; 200000..204999 |
| `det_run.py`, `det_run.out` | (N) and (H) on the wrong gradients that the first design missed. | 210000..213599 |
| `diag_r.py`, `diag_r.out` | The trade between silence on approximate gradients and detection, under a relative floor. | 210000..213599, 216000..217299 |
| `formulas.py`, `formulas.out` | The detection formulas against the prototype. | 213600..213849 |
| `predictions.py`, `predictions.out` | The predicted rates at d = 1,000 and 10,000, the harm scale, and the size and power of the registered tests. | generator 213600 |
| `harm_model.py`, `harm_model.out` | What a gradient error does to the acceptance rate and to the tuned step of MALA. | generators 217000..217002, 217006..217008 |
| `invariance.py`, `invariance.out` | The invariances of the decision, and a log-density that is constant on a line. | 217500..217549 |
| `harm_pkg.py` | Harm with the package's default-tuned coupled MALA. `harm_pkg.py twice` wrote `harm_pkg_twice.out` and `harm_pkg.py f32` wrote `harm_pkg_f32.out`. | 219000..219002 |
| `cost.py`, `cost.out` | Evaluations and peak memory of the prototype at d = 1,000 and 10,000. | 219500 |
| `harm_diag.py` | Harm of one wrong-sign coordinate by a diagonal coupled MALA. `harm_diag.py 50`, `200`, `1000`, `3000` and `harm_diag.py 10000 50` wrote `harm_diag_50.out` to `harm_diag_10000.out`. | generators 219600..219602 |

Two seeds of the same range, 200000 and 200001, served a smoke test of the prototype
whose output was not kept. No seed outside 200000 to 219999 was used.

## Which outputs were produced when

The scripts were formatted, and pointed at `../prototype.py` and `cases.py`, after their
outputs were produced. Every output except two was then produced again and is the same,
byte for byte, or apart from the run times that `harm_pkg.py` and `harm_diag.py` print
and the wall times of `cost.py`: for those three scripts the first outputs are the ones
kept. The two outputs not produced again are `harm_model.out`, whose script takes about
three minutes and changed only in formatting and in how two labels are written, and
`harm_diag_10000.out`, whose script is the one that reproduced the four smaller
dimensions.

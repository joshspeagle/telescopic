"""The adversarial review's runner: i102_q2_rules.one, unchanged, on new cells and seeds.

Question. Does the coverage rule confirmed by i102_q3_confirm.py hold beyond the cells and
the start width it was tested on? The review was run by an agent that had not run the
experiment, on fresh seeds (1000 and up for the issue's family, 2000 and up for covered
targets).

Design. The job is i102_q2_rules.one, with the cells of new_cells(): the issue's family at
other separations, dimensions and initial widths; narrower second modes; Gaussians of
condition number 10^4 and 10^6; correlated Gaussians; Student-t targets; log-gamma
coordinates; and some of the experiment's own cells. A job still running after CAP
seconds is redone as the rule alone (no default run) and marked T. The rule fires when
the "end" statistic at m = 8, n = n_steps/8 exceeds 10 and R-hat is silent. With "rule"
as the fourth argument, every job is the rule alone, without the default run.

Result: see the header of i102_q3_confirm.py, which states what the review changed, and
these outputs, one per invocation, each followed by its arguments (cells joined by ";",
first seed, number of seeds):
  i102_review_family1.out
    "bimodal d=10 s=4;bimodal d=10 s=5;bimodal d=5 s=6;bimodal d=6 s=6" 1000 10
  i102_review_family2.out
    "bimodal d=10 s=6 init var=25;bimodal d=10 s=6 init var=400;
     bimodal d=10 s=6 init var=1600;bimodal d=10 s=6 init var=10000" 1000 10
  i102_review_family3.out
    "bimodal d=32 s=6 init var=400;bimodal d=32 s=6 init var=1600" 1000 10
  i102_review_pumps_rule.out "ex pumps default" 2000 200 rule
  i102_review_pumps_rule2.out "ex pumps default" 2200 1000 rule
  i102_review_fa_rule1.out the 22 covered cells it lists, 2000 20 rule
  i102_review_fa_full1.out "gauss d=20 cond=10000 tight;loggamma.5 d=10 wide" 2000 20
  i102_review_g50_rule.out "gauss d=50 cond=100 tight" 2000 100 rule
  i102_review_g50b_rule.out
    "gauss d=50 cond=10000 tight;gauss d=50 cond=10000 wide" 2000 10 rule
CAP was set for each invocation and not recorded; the seconds of the lines marked T show
it (about 100 s for family1, 60 s for family2).

Ran against commit c961089; each line of an output gives its job's seconds.

Run: OUT=<folder for the rows> [CAP=150] [WORKERS=2] \
     python notes/experiments/i102_review_runner.py "<cells>" <first seed> <count> [rule]
"""

import os
import pickle
import signal
import sys
import time
from multiprocessing import Pool

import i102_q2_rules as q2_rules
import numpy as np
from i102_q2_rules import gaussian, student, tight, two_modes, wide

from telescopic.initial import Normal

OUT = os.environ["OUT"]

ORIG = dict(q2_rules.CELLS)
MS = (4, 8, 16, 32)
DIVS = (64, 32, 16, 8, 4, 2)
_last = {}
_orig_td = q2_rules.tuning_disagreement


def _td(report):
    _last["rhat"] = float(np.max(report.rhat))
    return _orig_td(report)


q2_rules.tuning_disagreement = _td


def corr_gauss(d, rho):
    cov = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    prec = np.linalg.inv(cov)
    return lambda x: -0.5 * np.einsum("ij,jk,ik->i", x, prec, x)


def skew_gamma(d, shape=0.5):
    # independent log-gamma coordinates: x = log g, g ~ Gamma(shape, 1); density exp(shape x - e^x)
    return lambda x: np.sum(shape * x - np.exp(x), axis=1)


def new_cells():
    out = {}
    # the #102 family at separations, dimensions and initial widths the criterion cells do not include
    for d, s in (
        (10, 3),
        (10, 4),
        (10, 5),
        (5, 6),
        (4, 6),
        (6, 6),
        (5, 12),
        (20, 4),
        (3, 6),
        (3, 12),
    ):
        out[f"bimodal d={d} s={s}"] = ("detect", two_modes(d, s), wide(d), d, 0.7)
    for var in (25, 400, 1600, 10000):
        out[f"bimodal d=10 s=6 init var={var}"] = (
            "detect",
            two_modes(10, 6),
            Normal(np.zeros(10), var * np.eye(10)),
            10,
            0.7,
        )
        out[f"bimodal d=32 s=6 init var={var}"] = (
            "detect",
            two_modes(32, 6),
            Normal(np.zeros(32), var * np.eye(32)),
            32,
            0.7,
        )
    for sd2 in (0.5, 0.7):
        out[f"narrow sd2={sd2} d=10 s=6"] = (
            "detect",
            two_modes(10, 6, sd2=sd2),
            wide(10),
            10,
            0.7,
        )
    # covered targets the cells do not include
    for d, cond in ((50, 1e4), (50, 1e6), (10, 1e6), (20, 1e4)):
        lp, _ = gaussian(d, cond)
        out[f"gauss d={d} cond={cond:g} wide"] = ("fa", lp, wide(d), d, np.zeros(d))
        out[f"gauss d={d} cond={cond:g} tight"] = ("fa", lp, tight(np.zeros(d)), d, np.zeros(d))
    for d, rho in ((10, 0.99), (10, 0.999), (30, 0.99)):
        out[f"corr d={d} rho={rho} wide"] = ("fa", corr_gauss(d, rho), wide(d), d, np.zeros(d))
        out[f"corr d={d} rho={rho} tight"] = (
            "fa",
            corr_gauss(d, rho),
            tight(np.zeros(d)),
            d,
            np.zeros(d),
        )
    for d, nu in ((10, 3), (2, 3), (30, 4), (10, 2.5)):
        out[f"student{nu} d={d} wide"] = ("fa", student(d, nu), wide(d), d, np.zeros(d))
        out[f"student{nu} d={d} tight"] = (
            "fa",
            student(d, nu),
            tight(np.zeros(d)),
            d,
            np.zeros(d),
        )
    from scipy.special import digamma

    for d in (2, 10):
        out[f"loggamma.5 d={d} wide"] = ("fa", skew_gamma(d), wide(d), d, np.full(d, digamma(0.5)))
        out[f"loggamma.5 d={d} N(0,I)"] = (
            "fa",
            skew_gamma(d),
            Normal(np.zeros(d), np.eye(d)),
            d,
            np.full(d, digamma(0.5)),
        )
    for name in (
        "ex pumps default",
        "ex pumps tight",
        "ex banana default",
        "ex mixture default",
        "gauss d=50 cond=100 tight",
        "bimodal d=10 s=6",
        "bimodal d=2 s=6",
        "extra funnel default",
    ):
        out[name] = ORIG[name]
    return out


q2_rules.CELLS = new_cells()
CAP = int(os.environ.get("CAP", "150"))


def _alarm(signum, frame):
    raise TimeoutError


def job(args):
    path = os.path.join(
        OUT,
        f"{args[0].replace(' ', '_').replace('/', '_')}__{args[1]}__{int(len(args) > 2 and args[2])}.pkl",
    )
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(CAP)
    t0 = time.time()
    try:
        row = q2_rules.one(args)
        row["rhat"] = _last["rhat"]
    except TimeoutError:
        signal.alarm(0)
        row = q2_rules.one((args[0], args[1], True))  # the rule alone
        row["rhat"] = _last["rhat"]
        row["timed_out"] = True
        row["wrong"] = None
    finally:
        signal.alarm(0)
    row["secs"] = time.time() - t0
    with open(path, "wb") as f:
        pickle.dump(row, f)
    return row


def stat(r):
    return float(r["stats"][MS.index(8), DIVS.index(8), 0])


def show(r):
    w = "T" if r.get("timed_out") else ("E" if r["wrong"] is None else int(r["wrong"]))
    fire = int(stat(r) > 10 and not r["disagree"])
    return (
        f"{r['cell']:34s} | {r['seed']:5d} | k {r['k']:6d} tau {r['tau_max']:8.0f} thin {r['thin']:2d} | {r['value']:8.4f} {r['se']:7.4f} z {r['z']:9.2f} "
        f"wrong {w} | rhat {r['rhat']:6.3f} | stat {stat(r):7.1f} | fires {fire} | {r['secs']:.0f}s {r.get('error') or ''}"
    )


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1].split(";")
    first, count = int(sys.argv[2]), int(sys.argv[3])
    skip = len(sys.argv) > 4 and sys.argv[4] == "rule"
    jobs = [
        ((n, first + i, True) if skip else (n, first + i)) for n in names for i in range(count)
    ]
    with Pool(int(os.environ.get("WORKERS", "2")), maxtasksperchild=10) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    for r in rows:
        print(show(r), flush=True)
    for n in names:
        c = [r for r in rows if r["cell"] == n]
        wrong = [r for r in c if r["wrong"] is True]
        right = [r for r in c if r["wrong"] is False and not skip]
        f = lambda r: stat(r) > 10 and not r["disagree"]
        print(
            f"SUMMARY {n}: runs {len(c)}, wrong {len(wrong)}, right {len(right)}, raised/timed out/rule-only {len(c) - len(wrong) - len(right)}; "
            f"fires on wrong {sum(map(f, wrong))}/{len(wrong)}; fires on right {sum(map(f, right))}/{len(right)}; fires overall {sum(map(f, c))}/{len(c)}; "
            f"R-hat warns {sum(r['disagree'] for r in c)}; stat>10 {sum(stat(r) > 10 for r in c)}; stat min/med/max {min(map(stat, c)):.1f}/{np.median(list(map(stat, c))):.1f}/{max(map(stat, c)):.1f}"
        )

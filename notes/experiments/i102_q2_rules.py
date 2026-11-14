"""The data of the candidate rules: a default run and each rule's statistics (issue #102).

Question. Which check at run time can warn that the automatically tuned independence
proposal misses a mode, at what false-alarm rate and at what cost? The criterion, fixed
before the first run: one setting detects at least 90% of the runs whose estimate is
wrong on the issue's family (d = 10 and 32, s = 6 and 12), fires on none of at least 300
runs where the proposal covers the target, and costs at most 50% of the pilot's
evaluations.

Design. Kernel Mixture([RandomWalk(), Independence()], [0.8, 0.2]), automatic tuning,
400 replicates, as in the issue. Per (cell, seed):
  * the default run, through CoupledSampler with the kernel tuned on the same pilot
    (equal to telescopic.sample, checked below): its estimate, standard error and z
    against the truth; "wrong" means |z| > 5 (a zero standard error with a wrong
    estimate counts as wrong). Detection cells estimate P(x0 < 0); false-alarm cells the
    mean of every coordinate (the largest |z| over coordinates).
  * the pilot (adaptive_metropolis on the run's own root seed), the fitted t_5 proposal q,
    and the pilot's log weights log pi - log q over its kept draws (largest Lmax).
  * R0a: the largest log w over the pilot's 100 initial draws, minus Lmax (cost 100
    evaluations); R0b: the same over 2,000 fresh initial draws (cost 2,000). R0 is the
    rule the issue proposed.
  * R1: 32 random-walk chains with covariance 2.38^2/d times the pilot's, from fresh
    initial draws, n_steps/2 steps each, where n_steps is the pilot's length. For
    m in {4, 8, 16, 32} (the first m chains) and n = n_steps / {64, 32, 16, 8, 4, 2}:
    "end" = the largest log w over the chains at step n, minus Lmax; "half" = the largest
    over the second half of the n steps; "end_dens" and "half_dens" = the same over states
    whose log pi is at least the 1% quantile of the pilot's kept log pi. Cost m n
    evaluations; the pilot costs 4 (n_steps + 1).
  * the existing "tuning chains disagree" check (largest R-hat > sqrt 2) on the pilot.
Probe randomness: numpy.random.default_rng([seed, 102]). Seeds: detection cells 0 to 19,
false-alarm cells 100 to 119 ("main"); then 20 to 39 and 200 to 219 ("confirm"), fixed
before any result was seen. The confirm set reruns the four criterion cells with their
default runs and the false-alarm cells without them (all 440 false-alarm runs of the main
set had a right estimate). Runs whose automatic k would store more than 0.5 GB were
thinned (column thin).

Result (i102_q2_rules_main.out, i102_q2_rules_confirm.out, analyzed by
i102_q2_analysis.py and i102_q2_pooled.py). The default runs reproduce the issue (d = 10,
s = 6, seed 0: 1.0002 +- 0.0003, k = 20), and the estimate is wrong in 14 of 20 runs at
d = 10 and 18 of 20 at d = 32 (s = 6) on the main seeds. R0 never fires on a wrong run;
R1 "end" with m = 8, n = n_steps/8, c = 12 (chosen on the main seeds) detects 62 of 62
wrong runs on both seed sets at 25% of the pilot, with no false alarm in 440 runs on the
main seeds and 2 in 440 on the confirm seeds (examples.pumps() from its default start).
i102_q3_confirm.py tests the rule chosen next.

The rows of the main set were written by an earlier version of this script, which did not
record the field `error` and recorded `m` wrongly; no analysis reads either.

Ran against commit c961089. Each set takes on the order of an hour on three workers; a
set that is stopped resumes from its saved rows, so the wall time printed last is that of
the last invocation. Rerun for issue #85 with the same rows: about 90 minutes of job time
for the main set and 20 for the confirm set on one core, one job per process, at commit
32ae70b; the saved rows were then read and the tables printed at the commit that the
first line of each output records, whose package source is that of 32ae70b.

Run:
  python notes/experiments/i102_q2_rules.py 0 100 main [workers]
  python notes/experiments/i102_q2_rules.py 20 200 confirm <workers> confirm
Each writes one file per job to notes/experiments/_rows/rows_<tag>/ and all rows to
notes/experiments/_rows/q2_rows_<tag>.npy, which the analysis scripts read there; git
ignores the folder.
"""

import os
import pickle
import resource
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from i85_setup import setup
from i102_common import (
    ROWS,
    gaussian,
    initial_points,
    probe,
    proposal_of,
    student,
    two_modes,
)

import telescopic
from telescopic import examples
from telescopic._random import root_seed
from telescopic.diagnostics import tuning_disagreement
from telescopic.initial import Normal, Uniform, as_initial, draw
from telescopic.kernels import Independence, Mixture, RandomWalk
from telescopic.target import Target
from telescopic.tuning import adaptive_metropolis, choose_k

R = 400
MEMORY = 0.5e9
MS = (4, 8, 16, 32)
DIVS = (64, 32, 16, 8, 4, 2)


def wide(d):
    return Normal(np.zeros(d), 100 * np.eye(d))


def tight(mean):
    return Normal(np.asarray(mean, float), 0.01 * np.eye(len(mean)))


def cells():
    out = {}
    # Detection: the #102 family and its variants; truth P(x0 < 0) = weight of the left mode.
    for d in (2, 10, 32):
        for s in (6, 12):
            out[f"bimodal d={d} s={s}"] = ("detect", two_modes(d, s), wide(d), d, 0.7)
    out["unequal .95/.05 d=10 s=6"] = (
        "detect",
        two_modes(10, 6, (0.95, 0.05)),
        wide(10),
        10,
        0.95,
    )
    out["narrow sd2=.3 d=10 s=6"] = ("detect", two_modes(10, 6, sd2=0.3), wide(10), 10, 0.7)
    out["init N(0,4I) d=10 s=6"] = (
        "detect",
        two_modes(10, 6),
        Normal(np.zeros(10), 4 * np.eye(10)),
        10,
        0.7,
    )
    one = np.zeros(10)
    one[0] = -6
    out["init on left mode d=10 s=6"] = (
        "detect",
        two_modes(10, 6),
        Normal(one, np.eye(10)),
        10,
        0.7,
    )
    # False alarms: covered targets; truth = the mean of every coordinate.
    for name, ex in (
        ("ex gaussian", examples.gaussian()),
        ("ex banana", examples.banana()),
        ("ex mixture", examples.mixture()),
        ("ex pumps", examples.pumps()),
    ):
        tight_init = (
            Uniform(0.95 * ex.mean, 1.05 * ex.mean) if name == "ex pumps" else tight(ex.mean)
        )
        out[f"{name} default"] = ("fa", ex.log_prob, ex.initial, ex.ndim, ex.mean)
        out[f"{name} tight"] = ("fa", ex.log_prob, tight_init, ex.ndim, ex.mean)
    for d in (2, 10, 50):
        for cond in (1, 100):
            lp, _ = gaussian(d, cond)
            out[f"gauss d={d} cond={cond} wide"] = ("fa", lp, wide(d), d, np.zeros(d))
            out[f"gauss d={d} cond={cond} tight"] = ("fa", lp, tight(np.zeros(d)), d, np.zeros(d))
    out["student4 d=10 wide"] = ("fa", student(10), wide(10), 10, np.zeros(10))
    out["student4 d=10 tight"] = ("fa", student(10), tight(np.zeros(10)), 10, np.zeros(10))
    # Outside the criterion: the funnel, where an independence component is not sensible.
    ex = examples.funnel()
    out["extra funnel default"] = ("extra", ex.log_prob, ex.initial, ex.ndim, ex.mean)
    return out


CELLS = cells()


def kernel():
    return Mixture([RandomWalk(), Independence()], [0.8, 0.2])


def one(job):
    name, seed = job[:2]
    skip_run = len(job) > 2 and job[2]
    kind, log_prob, initial, d, truth = CELLS[name]
    t0 = time.time()
    root = root_seed(seed)
    ptarget = Target(log_prob, vectorize=True)
    pilot = adaptive_metropolis(ptarget, initial, root)
    pilot_evals = ptarget.n_log_prob
    tuned, _ = kernel().tune(ptarget, initial, root, pilot=pilot)
    q = proposal_of(pilot)
    assert np.array_equal(q.mean, tuned.kernels[1].proposal.mean)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler = telescopic.CoupledSampler(
            log_prob, initial, vectorize=True, kernel=tuned, rng=seed
        )
        # A run whose automatic k is large stores gigabytes (#73; one worker was killed at
        # 12.5 GB): thin it to about MEMORY bytes of stored states. Thinning changes the
        # estimator (still unbiased), so such runs are flagged in the table.
        # Confirmation of the false-alarm cells: the rule only, no default run.
        times = np.array([np.nan]) if skip_run else sampler.meeting_times(100)
        error = "run skipped" if skip_run else None
        try:
            k = choose_k(times) if error is None else -1
        except ValueError as exc:  # the default run raises: a loud failure, not a wrong estimate
            error = str(exc)[:60]
        if error is None:
            stored = R * 2 * (9 * k + 1) * d * 8
            thin = max(1, int(np.ceil(stored / MEMORY)))
            result = sampler.run(R, thin=thin) if thin > 1 else sampler.run(R)
            f = (lambda x: x[:, 0] < 0) if kind == "detect" else (lambda x: x)
            reps = np.asarray(result.estimate(f).replicates, float).reshape(R, -1)
            taus = result.meeting_times
            k, m = result.k, result.m
            del result
    run_evals = sampler.target.n_log_prob
    if error is None:
        value_all = reps.mean(axis=0)
        se_all = reps.std(axis=0, ddof=1) / np.sqrt(R)
        truth = np.atleast_1d(truth)
        with np.errstate(divide="ignore", invalid="ignore"):
            zs = np.where(
                se_all > 0, (value_all - truth) / se_all, np.where(value_all == truth, 0, np.inf)
            )
        i = int(np.argmax(np.abs(zs)))
        value, se, z = float(value_all[0]), float(se_all[0]), float(zs[i])
        wrong = bool(abs(z) > 5)
    else:
        value = se = z = np.nan
        k = m = thin = -1
        taus = times
        wrong = False if skip_run else None
    # Pilot statistics.
    kept = pilot.samples.reshape(-1, d)
    lw_pilot = pilot.log_prob.ravel() - q.log_density(kept)
    lmax = float(lw_pilot.max())
    lp_low = float(np.quantile(pilot.log_prob, 0.01))
    disagree = tuning_disagreement(pilot.report()) is not None
    # R0a: the pilot's own 100 initial draws.
    pts, _ = initial_points(initial, root)
    pts = pts.reshape(-1, d)
    lp0 = log_prob(pts)
    r0a = float(np.max(lp0 - q.log_density(pts)) - lmax)
    r0a_dens = float(np.max(np.where(lp0 >= lp_low, lp0 - q.log_density(pts), -np.inf)) - lmax)
    # R0b: 2,000 fresh draws.
    rng = np.random.default_rng([seed, 102])
    init = as_initial(initial)
    fresh = np.array([draw(init, rng, d) for _ in range(2000)])
    lpf = log_prob(fresh)
    r0b = float(np.max(lpf - q.log_density(fresh)) - lmax)
    r0b_dens = float(np.max(np.where(lpf >= lp_low, lpf - q.log_density(fresh), -np.inf)) - lmax)
    # R1 probes.
    n_steps = pilot.n_steps
    n_max = n_steps // 2
    plp, plq, px0 = probe(Target(log_prob, vectorize=True), initial, pilot, rng, max(MS), n_max)
    lw = plp - plq
    dens = plp >= lp_low
    stats = np.full((len(MS), len(DIVS), 4), -np.inf)
    other = np.zeros(
        (len(MS), len(DIVS))
    )  # chains whose x0 at step n is on the other side from q's mean
    for j, div in enumerate(DIVS):
        n = n_steps // div
        h = n // 2
        for i, mm in enumerate(MS):
            stats[i, j, 0] = lw[n, :mm].max() - lmax
            stats[i, j, 1] = lw[h : n + 1, :mm].max() - lmax
            stats[i, j, 2] = np.max(np.where(dens[n, :mm], lw[n, :mm], -np.inf)) - lmax
            stats[i, j, 3] = (
                np.max(np.where(dens[h : n + 1, :mm], lw[h : n + 1, :mm], -np.inf)) - lmax
            )
            other[i, j] = np.sum(np.sign(px0[n, :mm]) != np.sign(q.mean[0]))
    return {
        "cell": name,
        "kind": kind,
        "seed": seed,
        "d": d,
        "n_steps": n_steps,
        "pilot_evals": pilot_evals,
        "run_evals": pilot_evals + run_evals,
        "k": k,
        "m": m,
        "error": error,
        "tau_max": float(np.max(taus)),
        "thin": thin,
        "value": value,
        "se": se,
        "z": float(z),
        "wrong": wrong,
        "q_side": float(np.sign(q.mean[0])),
        "lmax": lmax,
        "lmed": float(np.median(lw_pilot)),
        "disagree": disagree,
        "r0a": r0a,
        "r0a_dens": r0a_dens,
        "r0b": r0b,
        "r0b_dens": r0b_dens,
        "stats": stats,
        "other": other,
        "n_warn": len(caught),
        "warn_text": "; ".join(sorted({str(w.message)[:40] for w in caught})),
        "secs": time.time() - t0,
    }


def path_of(folder, job):
    return os.path.join(folder, f"{job[0].replace(' ', '_').replace('/', '_')}__{job[1]}.pkl")


def saved(args):
    folder, job = args
    row = one(job)
    row["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with open(path_of(folder, job), "wb") as f:
        pickle.dump(row, f)
    return row


def check_identity():
    # The run through a pre-tuned kernel equals telescopic.sample with automatic tuning.
    log_prob = two_modes(10, 6)
    for seed in (0, 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = telescopic.sample(log_prob, wide(10), R, vectorize=True, kernel=kernel(), rng=seed)
            row = one((("bimodal d=10 s=6"), seed))
        ea = a.estimate(lambda x: x[:, 0] < 0)
        assert abs(float(ea.value) - row["value"]) < 1e-12, (ea, row)
        assert a.k == row["k"], (ea, row)
        assert abs(float(ea.standard_error) - row["se"]) < 1e-12, (ea, row)
    print("pre-tuned runs equal telescopic.sample to 1e-12 (bimodal d=10 s=6, seeds 0, 1)")


if __name__ == "__main__":
    first_det, first_fa, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    t0 = time.time()
    print(setup())
    if tag == "main":
        check_identity()
    jobs = [
        (name, (first_det if CELLS[name][0] == "detect" else first_fa) + i)
        for name in CELLS
        for i in range(20)
    ]
    if len(sys.argv) > 5 and sys.argv[5] == "confirm":
        # Confirmation: the criterion's detection cells with their default runs, and the
        # false-alarm cells with the rule only (all 440 primary false-alarm runs were right).
        crit = {f"bimodal d={d} s={s}" for d in (10, 32) for s in (6, 12)}
        jobs = [(n, sd) for n, sd in jobs if n in crit] + [
            (n, sd, True) for n, sd in jobs if CELLS[n][0] == "fa"
        ]
    # Big ones first, for load balance. Each row is saved as it arrives, so that a rerun
    # resumes; a worker killed for memory is reported, not waited for.
    jobs.sort(key=lambda j: -CELLS[j[0]][3])
    folder = ROWS / f"rows_{tag}"
    os.makedirs(folder, exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(path_of(folder, j))]
    with Pool(workers, maxtasksperchild=10) as pool:
        for i, row in enumerate(pool.imap_unordered(saved, [(folder, j) for j in todo])):
            print(
                f"done {i + 1}/{len(todo)} {row['cell']} {row['seed']} {row['secs']:.0f}s rss {row['rss_mb']:.0f}MB",
                file=sys.stderr,
                flush=True,
            )
    rows = [pickle.loads(Path(path_of(folder, j)).read_bytes()) for j in jobs]
    order = list(CELLS)
    rows.sort(key=lambda r: (order.index(r["cell"]), r["seed"]))
    np.save(ROWS / f"q2_rows_{tag}.npy", np.array(rows, dtype=object), allow_pickle=True)
    print(
        f"\nraw table ({tag}): cell | seed | n_steps pilot-evals run-evals k m=10k tau_max | estimate se z wrong |"
        " q-side Lmax Lmed disagree | R0a R0b | R1 half_dens at (m=16, n=n_steps/16) | (m=32, n=n_steps/8) | warnings"
    )
    for r in rows:
        print(
            f"{r['cell']:30s} | {r['seed']:3d} | {r['n_steps']:6d} {r['pilot_evals']:6d} {r['run_evals']:8d} "
            f"{r['k']:4d} {10 * r['k']:5d} {r['tau_max']:6.0f} | {r['value']:9.4f} {r['se']:8.4f} {r['z']:9.2f} "
            f"{'E' if r['wrong'] is None else int(r['wrong'])} | {r['q_side']:+.0f} {r['lmax']:7.2f} {r['lmed']:7.2f} {int(r['disagree'])} | "
            f"{r['r0a']:8.1f} {r['r0b']:8.1f} | {r['stats'][2, 2, 3]:8.1f} | {r['stats'][3, 3, 3]:8.1f} | "
            f"{r['n_warn']} {r['warn_text']} {r.get('error') or ''} thin={r['thin']}"
        )
    print(f"\nwall time {time.time() - t0:.0f} s (shared machine; not a measurement)")

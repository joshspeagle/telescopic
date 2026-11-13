"""Why do the tuning pilot's four chains so often end in one mode? (issue #102)

Question. A conjecture: with the pooled covariance, chains in different modes stretch the
proposal along the line between the modes, so chains keep crossing until all sit in one
mode, after which the covariance shrinks and crossing stops. Test: the same four chains,
from the same starts and streams, with each chain's mean and covariance adapted from its
own states only (i102_common.pilot, pooled=False), against the package's pooled loop
(pooled=True, checked bit for bit against telescopic.tuning.adaptive_metropolis).

Design. Target 0.7 N(-s e1, I) + 0.3 N(s e1, I) (modes 2s apart), initial N(0, 100 I),
s in {6, 8, 12, 20, 40}, d in {2, 10, 32}, seeds 0 to 19 (root seeds of the package's
tuning namespace), both arms. Per run: chains in the left mode at the end; split at the
end (final x0 signs differ); mode jumps (sign changes of x0 between two states that both
have log pi > median kept log pi - 3 sqrt d); the step after which the chains never again
sit on different sides ("absorption"); the step at which all four have arrived (log pi
above that threshold); the proposal standard deviation along x0 (sqrt(lambda_c Sigma_00))
while the chains are split and after absorption; the largest R-hat of the second halves
and whether the package's "tuning chains disagree" warning (R-hat > sqrt 2) would fire.

Result (i102_q1_split.out). Pooling is the cause, but not by crossing between modes.
Pooled, all four chains end in one mode in 15 of 20 runs at d = 10 and 18 of 20 at d = 32
(s = 6), against 1 of 20 and 0 of 20 per chain. At d >= 10, 1 of 200 pooled runs has any
jump between modes and none has one after absorption: the chains reach one side of
x0 = 0 during the descent (median absorption step 33 at d = 10 and 86 at d = 32, s = 6)
before all four have arrived (median steps 128 and 642), while the pooled proposal's mean
standard deviation along x0 with the chains split is 1.36 to 4.52 at d >= 10, against
0.41 to 1.42 per chain. The per-chain arm splits at the rate of independent descents, and
its R-hat warning then fires in 19 of 20 and 20 of 20 runs (d = 10 and 32, s = 6); pooled,
the warning fires exactly in the runs that split (5 of 20 and 2 of 20), whose proposal
then spans both modes.

Ran against commit c961089, in about five minutes on three workers.

Run: python notes/experiments/i102_q1_split.py [workers]   (default 3)
"""

import sys
import time
from multiprocessing import Pool

import numpy as np
from i102_common import pilot, two_modes

from telescopic.initial import Normal
from telescopic.target import Target
from telescopic.tuning import adaptive_metropolis

S = (6, 8, 12, 20, 40)
D = (2, 10, 32)
SEEDS = range(20)


def one(args):
    s, d, pooled, seed = args
    log_prob = two_modes(d, s)
    run, _, x0, lp, v0 = pilot(
        Target(log_prob, vectorize=True),
        Normal(np.zeros(d), 100 * np.eye(d)),
        seed,
        pooled=pooled,
        trace=True,
    )
    thr = np.median(run.log_prob) - 3 * np.sqrt(d)
    side = np.sign(x0)
    jumps = (side[1:] != side[:-1]) & (lp[1:] > thr) & (lp[:-1] > thr)
    split = np.any(side != side[:, :1], axis=1)
    absorbed = int(np.nonzero(split)[0][-1] + 1) if split.any() else 0
    jump_steps = np.nonzero(jumps.any(axis=1))[0] + 1
    flips = side[1:] != side[:-1]
    flip_steps = np.nonzero(flips)[0] + 1
    first_arrival = int(np.argmax(np.all(lp > thr, axis=1)))
    arrived = np.all(lp > thr, axis=1)
    sd_split = np.sqrt(v0[split & arrived]).mean() if np.any(split & arrived) else np.nan
    sd_after = np.sqrt(v0[absorbed:]).mean() if absorbed < len(v0) else np.nan
    rhat = float(np.max(run.rhat))
    return {
        "s": s,
        "d": d,
        "pooled": pooled,
        "seed": seed,
        "left": int((x0[-1] < 0).sum()),
        "split_end": bool(split[-1]),
        "jumps": int(jumps.sum()),
        "flips": int(flips.sum()),
        "flips_after": int((flip_steps > absorbed).sum()) if not split[-1] else -1,
        "first_arrival": first_arrival,
        "jumps_after": int((jump_steps > absorbed).sum()) if not split[-1] else -1,
        "absorbed": absorbed if not split[-1] else -1,
        "n_steps": len(x0),
        "sd_split": float(sd_split),
        "sd_after": float(sd_after),
        "rhat": rhat,
        "warn": rhat > np.sqrt(2),
    }


def check_copy():
    for d in (10, 32):
        for seed in (0, 1):
            log_prob = two_modes(d, 6)
            init = Normal(np.zeros(d), 100 * np.eye(d))
            a = adaptive_metropolis(Target(log_prob, vectorize=True), init, seed)
            b, _ = pilot(Target(log_prob, vectorize=True), init, seed, pooled=True)
            assert np.array_equal(a.samples, b.samples)
            assert np.array_equal(a.log_prob, b.log_prob)
    print(
        "pooled copy equals telescopic.tuning.adaptive_metropolis bit for bit (d = 10, 32; seeds 0, 1)"
    )


def show_trace(d, s, seed):
    # The first 120 steps: chains with x0 < 0, x0 of each chain, the x0 proposal sd of
    # chain 0, and the number of chains with log pi above the arrival threshold.
    print(
        f"\ntrace d={d} s={s} seed={seed}: step | chains with x0<0 | x0 per chain | x0 proposal sd (chain 0) | arrived"
    )
    for pooled in (True, False):
        log_prob = two_modes(d, s)
        run, _, x0, lp, v0 = pilot(
            Target(log_prob, vectorize=True),
            Normal(np.zeros(d), 100 * np.eye(d)),
            seed,
            pooled=pooled,
            trace=True,
        )
        thr = np.median(run.log_prob) - 3 * np.sqrt(d)
        print("  pooled" if pooled else "  per-chain")
        for t in (0, 1, 2, 4, 8, 15, 25, 40, 60, 80, 120, 200, 400):
            print(
                f"   {t:4d} | {int((x0[t] < 0).sum())} | {np.round(x0[t], 1).tolist()} | {np.sqrt(v0[t, 0]):6.2f} |"
                f" {int((lp[t] > thr).sum())}"
            )


if __name__ == "__main__":
    t0 = time.time()
    check_copy()
    for seed in (0, 1, 2):
        show_trace(10, 6, seed)
    jobs = [
        (s, d, pooled, seed) for d in D for s in S for pooled in (True, False) for seed in SEEDS
    ]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    print(
        "\nraw table: s d arm seed | chains-left split-at-end mode-jumps jumps-after-absorption "
        "absorption-step n_steps | side-flips flips-after-absorption step-all-four-arrived |"
        " x0-proposal-sd split/after | max-rhat warn"
    )
    for r in rows:
        print(
            f"{r['s']:2d} {r['d']:2d} {'pool' if r['pooled'] else 'own '} {r['seed']:2d} | {r['left']} "
            f"{int(r['split_end'])} {r['jumps']:4d} {r['jumps_after']:3d} {r['absorbed']:6d} {r['n_steps']:6d}"
            f" | {r['flips']:4d} {r['flips_after']:3d} {r['first_arrival']:5d}"
            f" | {r['sd_split']:6.2f} {r['sd_after']:5.2f} | {r['rhat']:7.2f} {int(r['warn'])}"
        )
    print(
        "\nsummary per cell (20 seeds): split-at-end count, chains-left mean, runs with any mode jump, "
        "total jumps after absorption, median absorption step, mean x0 sd split/after, warn count"
    )
    for d in D:
        for s in S:
            for pooled in (True, False):
                c = [r for r in rows if r["d"] == d and r["s"] == s and r["pooled"] == pooled]
                ab = [r["absorbed"] for r in c if r["absorbed"] >= 0]
                ja = sum(r["jumps_after"] for r in c if r["jumps_after"] >= 0)
                fa = sum(r["flips_after"] for r in c if r["flips_after"] >= 0)
                arr = np.median([r["first_arrival"] for r in c])
                print(
                    f"d={d:2d} s={s:2d} {'pooled  ' if pooled else 'per-chain'}: split {sum(r['split_end'] for r in c):2d}/20,"
                    f" left {np.mean([r['left'] for r in c]):.2f}, jumped {sum(r['jumps'] > 0 for r in c):2d},"
                    f" jumps after absorption {ja}, side flips after absorption {fa}, absorption median {np.median(ab) if ab else np.nan:.0f},"
                    f" all-arrived median {arr:.0f},"
                    f" sd split {np.nanmean([r['sd_split'] for r in c]):.2f} after {np.nanmean([r['sd_after'] for r in c]):.2f},"
                    f" warn {sum(r['warn'] for r in c):2d}/20"
                )
    print(f"\nwall time {time.time() - t0:.0f} s (shared machine; not a measurement)")

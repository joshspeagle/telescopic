"""User calls and time of a run with `log_prob_and_grad` against `log_prob` and `grad` (#75).

Question: `Target.log_prob_and_grad` calls the user's `log_prob`, then `grad`. With one
function that returns both, passed as `log_prob_and_grad=`, how many calls of the
user's functions does a whole `sample` call make with MALA and with HMC, and how long
does it take, on targets whose value and gradient share part of their cost?

Two targets with d = 10 and a design matrix A of N rows, scaled so that the posterior's
scale is of order one: a logistic regression (N = 2,000, standard normal prior), whose value
and gradient share the product A beta and each add a transcendental function of its N
entries, and a linear regression with unit noise (N = 20,000, flat prior), whose value
and gradient share the residual A beta - b and whose gradient adds the product of A's
transpose with it, so that the shared product is a third of the separate functions'
arithmetic. Each run is `sample` with everything automatic (the gradient check, tuning,
100 lag-1 pilot pairs, then R = 128 replicates), `rng=0`, BLAS on one thread, once with
`log_prob` and `grad` and once with `log_prob` and `log_prob_and_grad`, whose numbers
are the same: the script checks that the two runs' replicates and counts are bitwise
equal. Wrappers count the calls of each function (a call takes a batch with
`vectorize=True`, one point without). Each configuration runs as 5 pairs of the two
runs, alternating which goes first, on a machine that other jobs share: the script
prints every pair's two times and their ratio, then the median and the range of the
5 ratios, and only the ratios are read. A saving is claimed only where the range lies
below 1. The whole script took about 23 minutes on one core.

Result (the output beside this file): the runs with the combined function were bitwise
those with two functions, counts included. It made 0.51 to 0.73 times as many user calls:
0.51 to 0.59 point by point, where every gradient took two calls, and 0.58 to 0.73 in
batches, where tuning's 2,001 calls of `log_prob` alone are the same in both (62 to 64% of
the combined run's calls with MALA and 29 to 30% with HMC). On the linear regression,
whose separate functions repeat a third of their arithmetic, every configuration saved
time: median time ratios 0.769 (range 0.757 to 0.864) for MALA and 0.731 (0.719 to
0.815) for HMC in batches, 0.740 (0.709 to 0.767) and 0.697 (0.571 to 0.718) point by
point. On the logistic regression, whose transcendental functions are not shared, three
of the four ranges include 1, so no saving is measured there: MALA 0.898 (0.852 to
1.054) and HMC 0.989 (0.932 to 1.012) in batches, MALA 0.962 (0.955 to 1.022) point by
point; HMC point by point, 0.926 (0.853 to 0.952), saved a little. Other jobs loaded
the machine: the ratio of 0.571 comes from a pair whose separate run took 88 s where
the others took 65 to 70 s.

Run from the repository: python notes/experiments/i75_value_and_grad.py
"""

import time

import numpy as np
from i85_setup import setup
from scipy.special import expit, log_expit
from threadpoolctl import threadpool_info, threadpool_limits

import telescopic
from telescopic.initial import Normal

D, R, REPEATS = 10, 128, 5

rng = np.random.default_rng(0)
X = rng.standard_normal((2_000, D)) / np.sqrt(D)
Y = (rng.random(len(X)) < expit(X @ rng.standard_normal(D))).astype(np.float64)
SIGNS = 2 * Y - 1
A = rng.standard_normal((20_000, D)) / np.sqrt(20_000)
B = A @ rng.standard_normal(D) + rng.standard_normal(len(A))


# Each function takes one point (D,) or rows (n, D).
def logistic_log_prob(beta):
    eta = beta @ X.T
    return np.sum(log_expit(SIGNS * eta), axis=-1) - 0.5 * np.sum(beta * beta, axis=-1)


def logistic_grad(beta):
    return (Y - expit(beta @ X.T)) @ X - beta


def logistic_log_prob_and_grad(beta):
    eta = beta @ X.T
    value = np.sum(log_expit(SIGNS * eta), axis=-1) - 0.5 * np.sum(beta * beta, axis=-1)
    return value, (Y - expit(eta)) @ X - beta


def linear_log_prob(beta):
    residual = beta @ A.T - B
    return -0.5 * np.sum(residual * residual, axis=-1)


def linear_grad(beta):
    return -(beta @ A.T - B) @ A


def linear_log_prob_and_grad(beta):
    residual = beta @ A.T - B
    return -0.5 * np.sum(residual * residual, axis=-1), -residual @ A


TARGETS = {
    "logistic regression, N = 2,000": (
        logistic_log_prob,
        logistic_grad,
        logistic_log_prob_and_grad,
    ),
    "linear regression, N = 20,000": (linear_log_prob, linear_grad, linear_log_prob_and_grad),
}


class Counted:
    # Counts the calls of a function.
    def __init__(self, function):
        self.function, self.calls = function, 0

    def __call__(self, beta):
        self.calls += 1
        return self.function(beta)


def run(functions, kernel, vectorize, combined):
    log_prob, grad, log_prob_and_grad = map(Counted, functions)
    gradient = {"log_prob_and_grad": log_prob_and_grad} if combined else {"grad": grad}
    start = time.perf_counter()
    result = telescopic.sample(
        log_prob,
        Normal(np.zeros(D), np.eye(D)),
        R,
        vectorize=vectorize,
        kernel=kernel,
        rng=0,
        **gradient,
    )
    seconds = time.perf_counter() - start
    calls = (log_prob.calls, grad.calls, log_prob_and_grad.calls)
    return result, calls, seconds


def compare(functions, kernel, vectorize):
    # The calls and the times of REPEATS pairs of runs, each pair alternating which run
    # goes first, after checking that the two runs agree bitwise and that the calls are
    # the same in every pair.
    seconds, counts = [], set()
    for repeat in range(REPEATS):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        runs = {side: run(functions, kernel, vectorize, side) for side in order}
        (separate, alone, apart), (combined, both, together) = runs[False], runs[True]
        for name, value in vars(separate._replicates).items():
            if name not in ("seed", "seconds"):
                assert np.array_equal(value, getattr(combined._replicates, name)), name
        seconds.append((apart, together))
        counts.add((alone, both))
    assert len(counts) == 1
    return separate, alone, both, np.array(seconds)


def main():
    print(setup())
    blas = ", ".join(f"{pool['internal_api']} {pool['num_threads']}" for pool in threadpool_info())
    print(f"threads of the BLAS and OpenMP pools under threadpool_limits(1): {blas}")
    print(
        f"d = {D}, R = {R}, sample() with everything automatic; {REPEATS} pairs of runs, "
        "alternating which goes first; time ratio = combined / separate in each pair"
    )
    for title, functions in TARGETS.items():
        print(f"\n{title}")
        print(
            f"{'kernel':6} {'calls of':>9}  {'log_prob':>9} {'grad':>9} {'combined':>9} "
            f"{'total':>9} {'ratio':>6}"
        )
        for vectorize in (True, False):
            for kernel in ("mala", "hmc"):
                separate, alone, both, seconds = compare(functions, kernel, vectorize)
                kind = "batches" if vectorize else "points"
                print(
                    f"{kernel:6} {kind:>9}  {alone[0]:9,d} {alone[1]:9,d} {'':>9} {sum(alone):9,d}"
                )
                print(
                    f"{'':6} {'':>9}  {both[0]:9,d} {'':>9} {both[2]:9,d} {sum(both):9,d} "
                    f"{sum(both) / sum(alone):6.3f}"
                )
                print(
                    f"{'':6} {'':>9}  k = {separate.k}; n_log_prob {separate.n_log_prob:,d}, "
                    f"n_grad {separate.n_grad:,d} in the replicates of both runs"
                )
                ratios = seconds[:, 1] / seconds[:, 0]
                pairs = ", ".join(f"{a:.2f}/{b:.2f} {b / a:.3f}" for a, b in seconds)
                print(f"{'':6} {'':>9}  seconds separate/combined, ratio: {pairs}")
                print(
                    f"{'':6} {'':>9}  time ratio: median {np.median(ratios):.3f}, "
                    f"range {ratios.min():.3f} to {ratios.max():.3f}"
                )


if __name__ == "__main__":
    with threadpool_limits(1):
        main()

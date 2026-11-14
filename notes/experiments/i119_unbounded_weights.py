"""What unbounded independence weights do to a run (#119).

The target is Exp(1) on [0, inf), whose mean is 1. Two proposals are compared, each
floored at 0 so that it has the atom of #119's example there, where its log-density is
-inf and the draw is rejected. The first is max(1 + z, 0) with z standard normal and the
log-density of N(1, 1) truncated to (0, inf): its tails are lighter than the target's,
so the weights pi/q grow without bound, as exp((x^2 - 4x) / 2). The second is
max(1 + t, 0) with t a Student-t with 3 degrees of freedom, built from four normals:
its tails are heavier than the target's, so the weights are bounded.

The question is what `telescopic.sample` reports with its defaults. Each proposal gets
200 runs, seeds 0 to 199, of 1,000 replicates from the initial distribution U(0, 2).
A run is summarized by its estimate of the mean, by z = (estimate - 1) / standard error
and by the warnings it raised. The output gives the median estimate, the share of runs
with z < -2 and with z > 2, the number of runs that raised no warning, and how many of
those have z < -2. The first line of the output records the commit, the machine, the
thread settings and the versions. The whole script takes about two minutes on one core.

The output also gives how long a chain stays at a point of the tail under the normal
proposal. From x it accepts with probability a(x), the integral of
phi(y - 1) min(1, w(y) / w(x)) over y > 0, with w = pi/q for the normalized densities,
w(x) = Phi(1) sqrt(2 pi) exp((x^2 - 4x + 1) / 2). For x >= 4 the weight at y exceeds
the weight at x only for y > x, and phi(y - 1) w(y) = Phi(1) exp(-y), so
a(x) = Phi(1) (1 - exp(-x)) / w(x) + 1 - Phi(x - 1), about 1 / w(x), and the chain
stays for 1 / a(x) steps on average.

Result (i119_unbounded_weights.out): with the normal proposal the median estimate is
0.970, 72.5% of the runs are more than two standard errors below 1 and none is that far
above, and 142 runs raise no warning, of which 135 are that low. With the Student-t
proposal the median is 0.999, 1.5% of the runs are that low and 1.0% that high, and no
run raises a warning. Under the normal proposal a chain stays for 51 steps on average
at x = 5, 1,671 at x = 6 and 150,197 at x = 7.

The reason is in those stays. In the coupled step a pair meets when the chain of larger
weight accepts, so a meeting time is long only when a chain is far in the tail at the
lag, which is rare from this start: the replicates that carry the bias correction for
the tail almost never occur in a run, and when one occurs its value is large.

Run: OMP_NUM_THREADS=1 python notes/experiments/i119_unbounded_weights.py
"""

import warnings

import numpy as np
from i85_setup import setup
from scipy import stats

import telescopic
from telescopic.kernels import Independence

RUNS, REPLICATES = 200, 1_000


class FlooredNormal:
    def sample(self, draws):
        return np.maximum(1 + draws.normal(1), 0.0)

    def log_density(self, x):
        v = x[:, 0]
        log_q = stats.norm.logpdf(v, 1, 1) - stats.norm.logsf(0, 1, 1)
        return np.where(v > 0, log_q, -np.inf)


class FlooredT:
    def sample(self, draws):
        z = draws.normal(4)
        t = z[:, :1] / np.sqrt((z[:, 1:] ** 2).mean(axis=1, keepdims=True))
        return np.maximum(1 + t, 0.0)

    def log_density(self, x):
        v = x[:, 0]
        log_q = stats.t.logpdf(v - 1, 3) - stats.t.logsf(-1, 3)
        return np.where(v > 0, log_q, -np.inf)


def log_prob(x):
    return np.where(x[:, 0] >= 0, -x[:, 0], -np.inf)


print(setup())
stays = []
for x in (5.0, 6.0, 7.0):
    weight = stats.norm.cdf(1) * np.sqrt(2 * np.pi) * np.exp((x * x - 4 * x + 1) / 2)
    accept = stats.norm.cdf(1) * (1 - np.exp(-x)) / weight + stats.norm.sf(x - 1)
    stays.append(f"{1 / accept:,.0f} at x = {x:g}")
print(f"normal proposal, mean number of steps a chain stays: {', '.join(stays)}")
initial = telescopic.initial.Uniform([0.0], [2.0])
for name, proposal in [
    ("normal proposal, unbounded weights", FlooredNormal()),
    ("Student-t proposal, bounded weights", FlooredT()),
]:
    estimates, z, silent = [], [], []
    for seed in range(RUNS):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = telescopic.sample(
                log_prob,
                initial,
                REPLICATES,
                vectorize=True,
                kernel=Independence(proposal),
                rng=seed,
            )
        mean = result.mean()
        estimates.append(mean.value[0])
        z.append((mean.value[0] - 1) / mean.standard_error[0])
        silent.append(not caught)
    z, silent = np.array(z), np.array(silent)
    print(
        f"{name}: {RUNS} runs of {REPLICATES} replicates; median estimate "
        f"{np.median(estimates):.3f}; z < -2 in {np.mean(z < -2):.1%} of the runs, z > 2 in "
        f"{np.mean(z > 2):.1%}; {silent.sum()} runs raised no warning, of which "
        f"{np.sum(z[silent] < -2)} have z < -2"
    )

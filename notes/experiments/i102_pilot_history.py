"""Does any proposal of the tuning pilot reach the mode its fit misses? (issue #102)

Question. If some chain of the pilot ever proposed a point in the other mode, the pilot
would hold the evidence that its fitted proposal is too narrow. Does it?

Design. Target 0.7 N(-6 e1, I) + 0.3 N(6 e1, I) in d dimensions, initial N(0, 100 I),
`adaptive_metropolis` on seeds 0 to 5, with every call of the log-density recorded. A
proposal counts as being "in" a mode when its log pi is within 3 sqrt(d) of the median
log pi of the pilot's kept states and its x0 has that mode's sign. Per seed: the fitted
mode, those proposals on each side in each half of the pilot, the last step with one in
the other mode (-1 for none), and the largest log w = log pi - log q over all proposals
against that over the kept states.

Result (i102_pilot_history.out, d = 10 and d = 32). In none of the 12 runs does any
proposal reach the other mode. The largest log weight over all proposals exceeds that over
the kept states by at most 1.0 at d = 10 and 2.5 at d = 32: nothing the pilot computed
holds the evidence.

Ran against commit c961089, in seconds at d = 10 and under a minute at d = 32.

Run: python notes/experiments/i102_pilot_history.py 10, then the same with 32; the output
holds both.
"""

import sys

import numpy as np
from scipy.special import logsumexp

from telescopic.initial import Normal
from telescopic.kernels.independence import MultivariateT
from telescopic.target import Target
from telescopic.tuning import adaptive_metropolis

d = int(sys.argv[1])
shift = np.zeros(d)
shift[0] = 6.0
log = []


def log_prob(x):
    lp = logsumexp(
        [
            np.log(0.7) - 0.5 * np.sum((x + shift) ** 2, axis=1),
            np.log(0.3) - 0.5 * np.sum((x - shift) ** 2, axis=1),
        ],
        axis=0,
    )
    log.append((x.copy(), lp.copy()))
    return lp


initial = Normal(np.zeros(d), 100 * np.eye(d))
for seed in range(6):
    log.clear()
    pilot = adaptive_metropolis(Target(log_prob, vectorize=True), initial, seed)
    q = MultivariateT(pilot.mean, pilot.cov, df=5)
    calls = [c for c in log if c[0].shape[0] == 4][1:]  # proposals, one row per chain
    x = np.stack([c[0] for c in calls])
    lp = np.stack([c[1] for c in calls])  # (steps, 4, d), (steps, 4)
    typical = np.median(pilot.log_prob)
    near = lp > typical - 3 * np.sqrt(
        d
    )  # proposals with log density in the range of the fitted mode's
    left = near & (x[:, :, 0] < 0)
    right = near & (x[:, :, 0] > 0)
    n = len(calls)
    lw = lp - q.log_density(x.reshape(-1, d)).reshape(lp.shape)
    lw_pilot = pilot.log_prob.ravel() - q.log_density(pilot.samples.reshape(-1, d))
    print(
        f"d={d} seed={seed} fitted mode {'L' if pilot.mean[0] < 0 else 'R'} steps={n}"
        f" high-density proposals L/R: first half {left[: n // 2].sum()}/{right[: n // 2].sum()},"
        f" second half {left[n // 2 :].sum()}/{right[n // 2 :].sum()};"
        f" last step with a high-density proposal in the other mode: "
        f"{np.max(np.nonzero((right if pilot.mean[0] < 0 else left).any(axis=1))[0], initial=-1)};"
        f" max log w over all proposals {lw.max():.1f} vs kept max {lw_pilot.max():.1f}"
    )

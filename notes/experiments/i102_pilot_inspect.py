"""Does anything the tuning pilot computes show a mode its proposal misses? (issue #102)

Question. The issue proposed to warn when a draw of the initial distribution with a
large weight pi/q shows that the fitted proposal q misses part of the target. Do such
draws exist on the issue's target?

Design. Target 0.7 N(-6 e1, I) + 0.3 N(6 e1, I) in 10 dimensions, initial N(0, 100 I),
`adaptive_metropolis` on seeds 0 to 5, and the Student-t_5 proposal that `Independence`
fits to it. Per seed: the four chains' means of x0, the R-hat of x0, the pooled mean and
variance of x0; log w = log pi - log q at the centers of the two modes, and its median
and largest value over the pilot's kept states; its largest value, and the largest
log pi, over 2,000 fresh draws of N(0, 100 I).

Result (i102_pilot_inspect.out). In all six runs the four chains' means of x0 lie in one
mode, with an R-hat of x0 between 1.00 and 1.04. The log weight at the center of the
other mode is 29.7 to 33.8, against a median of 8.0 to 9.0 and a largest value of 9.8 to
11.3 over the pilot's states, but the largest log weight over 2,000 draws of the initial
distribution is between -40.6 and -6.5 (largest log pi -71.7 to -32.7): a draw of so
wide an initial distribution almost never lands in the uncovered mode, so the issue's
rule cannot fire. Seeds 0 to 5 happen to be runs whose four chains did not split; over
20 seeds, see i102_q1_split.py.

Ran against commit c961089, in a few seconds; it gives the same output at later commits
whose tuning is unchanged.

Run: python notes/experiments/i102_pilot_inspect.py
"""

import numpy as np
from scipy.special import logsumexp

from telescopic.initial import Normal
from telescopic.kernels.independence import MultivariateT
from telescopic.target import Target
from telescopic.tuning import adaptive_metropolis

d = 10
shift = np.zeros(d)
shift[0] = 6.0


def log_prob(x):
    return logsumexp(
        [
            np.log(0.7) - 0.5 * np.sum((x + shift) ** 2, axis=1),
            np.log(0.3) - 0.5 * np.sum((x - shift) ** 2, axis=1),
        ],
        axis=0,
    )


initial = Normal(np.zeros(d), 100 * np.eye(d))
for seed in (0, 1, 2, 3, 4, 5):
    target = Target(log_prob, vectorize=True)
    pilot = adaptive_metropolis(target, initial, seed)
    ends = pilot.samples[:, -1, 0]
    means = pilot.samples[:, :, 0].mean(axis=1)
    q = MultivariateT(pilot.mean, pilot.cov, df=5)
    rng = np.random.default_rng(100 + seed)
    x0 = 10.0 * rng.standard_normal((2000, d))
    print(
        seed,
        "chain means x0:",
        np.round(means, 2),
        f"rhat0 {pilot.rhat[0]:.2f}",
        f"pooled mean0 {pilot.mean[0]:.2f} var0 {pilot.cov[0, 0]:.2f}",
    )
    # weights at the two modes
    modes = np.stack([-shift, shift])
    lw = log_prob(modes) - q.log_density(modes)
    draws = pilot.samples.reshape(-1, d)
    lwp = pilot.log_prob.ravel() - q.log_density(draws)
    print(
        "   log w at modes:",
        np.round(lw, 1),
        f" pilot log w median {np.median(lwp):.1f} max {lwp.max():.1f}",
    )
    if x0 is not None:
        lw0 = log_prob(x0) - q.log_density(x0)
        print(
            f"   initial draws: max log w {lw0.max():.1f}, max log pi {log_prob(x0).max():.1f};"
            f" pilot log pi median {np.median(pilot.log_prob):.1f}"
        )

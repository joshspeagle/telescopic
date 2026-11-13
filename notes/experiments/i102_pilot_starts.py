"""Where the tuning pilot's chains go during their descent (issue #102).

Question. The four chains of the pilot start from independent draws on both sides of the
origin. When do they reach the modes, and on which side?

Design. Target 0.7 N(-6 e1, I) + 0.3 N(6 e1, I) in 10 dimensions, initial N(0, 100 I),
`adaptive_metropolis` on seeds 0 to 3, with every call of the log-density recorded. Per
seed: each chain's starting x0 and log pi; for each chain, the first step whose proposal
has log pi > -30 and the x0 of its proposals at steps 5, 20, 50 and 100; the first step
of each chain in three coordinates; and the x0 of each chain's state at steps 10, 25, 40
and 60, replayed from proposals that are surely accepted (their log pi exceeds the
state's), which approximates the path.

Result (i102_pilot_starts.out). Each run starts with chains on both sides of x0 = 0. The
first proposal with log pi > -30 comes at steps 37 to 142, and at step 100 every chain's
proposal lies on one side: the negative side for seed 0, the positive side for seeds 1 to
3, the modes that i102_pilot_history.py finds fitted for these seeds. i102_q1_split.py
measures this over 20 seeds.

Ran against commit c961089, in a few seconds.

Run: python notes/experiments/i102_pilot_starts.py
"""

import numpy as np
from scipy.special import logsumexp

from telescopic.initial import Normal
from telescopic.target import Target
from telescopic.tuning import adaptive_metropolis

d = 10
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


for seed in range(4):
    log.clear()
    pilot = adaptive_metropolis(
        Target(log_prob, vectorize=True), Normal(np.zeros(d), 100 * np.eye(d)), seed
    )
    print(
        "seed",
        seed,
        "starts x0:",
        np.round(log[0][0][:, 0], 1),
        "start log pi:",
        np.round(log[0][1], 0),
    )
    # the accepted path: reconstruct states from proposals and log probs is not possible directly; show
    # the first step at which each chain's proposal has log pi > -30 and its sign
    x = np.stack([c[0] for c in log[1:]])
    lp = np.stack([c[1] for c in log[1:]])
    for c in range(4):
        hit = np.nonzero(lp[:, c] > -30)[0]
        print(
            "   chain",
            c,
            "first high-density proposal at step",
            hit[0],
            f"x0 = {x[hit[0], c, 0]:.1f}",
            "; x0 of proposals at steps 5,20,50,100:",
            np.round(x[[5, 20, 50, 100], c, 0], 1),
        )
    first_step = log[1][0] - log[0][0]
    print("   first step per chain, coords 0..2:", np.round(first_step[:, :3], 2).tolist())
    # reconstruct states: accept iff later proposals are centered on it; track x0 of state by replaying acceptance
    state = log[0][0].copy()
    lp_state = log[0][1].copy()
    path = []
    for xs, lps in log[1:]:
        # a proposal was accepted iff the next proposal's centre moved; approximate by Metropolis replay being unavailable,
        # so use: accepted if lps > lp_state (sure accept) else unknown; record sure accepts only
        sure = lps > lp_state
        state[sure] = xs[sure]
        lp_state[sure] = lps[sure]
        path.append(state[:, 0].copy())
    path = np.array(path)
    print(
        "   x0 of best-so-far point per chain at steps 10, 25, 40, 60:",
        np.round(path[[10, 25, 40, 60]], 1).tolist(),
    )

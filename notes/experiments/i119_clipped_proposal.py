"""Independence proposals drawn where their own log-density is -inf (#119).

Two proposals put mass on points where their reported log-density is -inf. On the
target U[0, 1], the proposal N(0.3, 0.3^2) clipped onto [0, 1], whose log_density is
that normal's on the open interval (0, 1) and -inf at 0 and 1 (about 16% of the draws).
On the target Exp(1), the proposal max(1 + z, 0) with the log-density of N(1, 1)
truncated to (0, inf), -inf at 0 (about 16% of the draws). The question is whether
chains started from the target stay distributed as the target, run step by step with
the kernel alone: 4,000 chains for 200 steps on U[0, 1], 20,000 for 100 on Exp(1).

At commit 18860fb, where such a proposal had the weight +inf, the chains collect on the
atoms: on U[0, 1], 94.0% at 0 and 6.0% at 1, mean 0.060, Kolmogorov-Smirnov p = 0; on
Exp(1), 100.0% at 0 (i119_clipped_proposal_before.out, this script run at that commit).
With the weight -inf, no chain is on an atom and both laws are kept: mean 0.4975 with
p = 0.55 against U[0, 1], mean 0.991 with p = 0.23 against Exp(1)
(i119_clipped_proposal.out).

Run: python notes/experiments/i119_clipped_proposal.py
"""

import numpy as np
from scipy import stats

import telescopic
from telescopic.kernels import Independence
from telescopic.testing import random_draws


class Clipped:
    def sample(self, draws):
        return np.clip(0.3 + 0.3 * draws.normal(1), 0.0, 1.0)

    def log_density(self, x):
        inside = (x[:, 0] > 0) & (x[:, 0] < 1)
        a, b = -1.0, 7.0 / 3.0
        return np.where(inside, stats.truncnorm.logpdf(x[:, 0], a, b, 0.3, 0.3), -np.inf)


class Floored:
    def sample(self, draws):
        return np.maximum(1 + draws.normal(1), 0.0)

    def log_density(self, x):
        v = x[:, 0]
        log_q = stats.norm.logpdf(v, 1, 1) - stats.norm.logsf(0, 1, 1)
        return np.where(v > 0, log_q, -np.inf)


cases = [
    (
        "U[0, 1], clipped N(0.3, 0.3^2)",
        lambda x: np.where((x[:, 0] >= 0) & (x[:, 0] <= 1), 0.0, -np.inf),
        Clipped(),
        stats.uniform,
        4_000,
        200,
        (0.0, 1.0),
    ),
    (
        "Exp(1), max(1 + z, 0)",
        lambda x: np.where(x[:, 0] >= 0, -x[:, 0], -np.inf),
        Floored(),
        stats.expon,
        20_000,
        100,
        (0.0,),
    ),
]
for name, log_prob, proposal, law, n, steps, atoms in cases:
    target = telescopic.Target(log_prob, vectorize=True)
    kernel = Independence(proposal)
    states = kernel.init_state(target, law.rvs(size=(n, 1), random_state=0))
    for step in range(steps):
        states, _ = kernel.step(target, states, random_draws(n, step))
    x = states.x[:, 0]
    on_atoms = ", ".join(f"at {a:g} {np.mean(x == a):.1%}" for a in atoms)
    p = stats.kstest(x, law.cdf).pvalue
    print(f"{name}: {on_atoms}; mean {x.mean():.4f} (truth {law.mean():g}); KS p {p:.2g}")

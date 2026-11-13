"""Shared pieces of the experiment of issue #102: targets, a copy of the pilot, the probes.

The experiment asks why a run whose kernel has an automatically tuned `Independence`
component returns confidently wrong estimates on a target with well-separated modes, and
whether a check at run time could warn of it. It ran against the package at commit
c961089, in this order: i102_pilot_inspect.py and i102_pilot_history.py (what the pilot
holds), i102_q1_split.py (why its chains end in one mode), i102_q2_rules.py with
i102_q2_analysis.py and i102_q2_pooled.py (candidate rules on two seed sets), and
i102_q3_confirm.py (one rule, fixed in advance, on fresh seeds). An adversarial review
then ran the same job function on other targets, starts and seeds: i102_review_runner.py
and i102_review_full_list.py, whose outputs are the i102_review_*.out files, and
i102_review_recount.py and i102_review_repro.py, which check the confirmation's saved
rows.

`pilot` is a copy of `telescopic.tuning.adaptive_metropolis` with one switch: `pooled=True`
reproduces the package bit for bit (checked in i102_q1_split.py); `pooled=False` gives each
chain the mean and covariance of its own states only. It also returns the trajectories of
x0, log pi and the proposal variance along x0 that i102_q1_split.py needs. It imports
private names of the package, as they were at c961089.

The run scripts write their rows under ROWS, notes/experiments/_rows/, which git ignores,
and the analysis scripts read them there.
"""

from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from telescopic._linalg import matvec
from telescopic._random import TUNING, Streams, replicate_generator, root_seed
from telescopic.initial import as_initial, draw
from telescopic.kernels.independence import MultivariateT
from telescopic.tuning import PilotRun

N = 4
ROWS = Path(__file__).parent / "_rows"


def two_modes(d, s, weights=(0.7, 0.3), sd2=1.0):
    """w1 N(-s e1, I) + w2 N(s e1, sd2^2 I), normalized."""
    shift = np.zeros(d)
    shift[0] = s
    lw1, lw2 = np.log(weights[0]), np.log(weights[1])
    c1 = -0.5 * d * np.log(2 * np.pi)
    c2 = c1 - d * np.log(sd2)

    def log_prob(x):
        return logsumexp(
            [
                lw1 + c1 - 0.5 * np.sum((x + shift) ** 2, axis=1),
                lw2 + c2 - 0.5 * np.sum((x - shift) ** 2, axis=1) / sd2**2,
            ],
            axis=0,
        )

    return log_prob


def student(d, nu=4):
    def log_prob(x):
        return -0.5 * (nu + d) * np.log1p(np.sum(x**2, axis=1) / nu)

    return log_prob


def gaussian(d, cond, seed=7):
    """N(0, Sigma), Sigma = R diag(logspace(0, log10 cond, d)) R^T with a fixed rotation."""
    variances = np.logspace(0, np.log10(cond), d)
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    cov = (q * variances) @ q.T
    precision = np.linalg.inv(cov)

    def log_prob(x):
        return -0.5 * np.einsum("ij,jk,ik->i", x, precision, x)

    return log_prob, cov


def initial_points(initial, root):
    # The 100 initial draws of adaptive_metropolis, from the same generators in the same
    # order (only the first of each chain's 25 is evaluated by the pilot).
    generators = [replicate_generator(root, TUNING, i) for i in range(N)]
    sources = []
    for generator in generators:
        sources.extend([generator] * 25)
    first = draw(initial, sources[0])
    rest = [draw(initial, g, len(first)) for g in sources[1:]]
    return np.reshape([first, *rest], (N, 25, len(first))), generators


def pilot(target, initial, rng, pooled=True, trace=False):
    """Copy of adaptive_metropolis; per-chain mean and covariance when pooled=False."""
    root = root_seed(rng)
    initial = as_initial(initial)
    points, generators = initial_points(initial, root)
    x, d = points[:, 0].copy(), points.shape[2]
    n_steps = max(2000, 200 * d, 10 * d * d)
    streams = Streams(N)
    for chain, generator in enumerate(generators):
        streams.assign(chain, generator)
    draws = streams.view(np.arange(N))
    log_p = target.log_prob(x)
    log_scale = np.full(N, np.log(2.38**2 / d))
    variance = points.reshape(-1, d).var(axis=0, ddof=1)
    sigma0 = np.diag(np.where(variance > 0, variance, 1.0))
    ridge = 1 + 1e-12 * np.eye(d)
    if pooled:
        mean, total = x.mean(axis=0), sigma0.copy()
    else:
        mean, total = x.copy(), np.repeat(sigma0[None], N, axis=0)
    first_kept = n_steps // 2
    stride = min(-(-d // 20), n_steps - first_kept - 1)
    n_kept = (n_steps - 1 - first_kept) // stride + 1
    samples = np.empty((N, n_kept, d))
    kept_log_p = np.empty((N, n_kept))
    accepted = np.zeros(N)
    deltas = np.empty((stride, N, d))
    tr_x0 = np.empty((n_steps, N)) if trace else None
    tr_lp = np.empty((n_steps, N)) if trace else None
    tr_v0 = np.empty((n_steps, N)) if trace else None
    for step in range(n_steps):
        if step % stride == 0:
            if step:
                if pooled:
                    block = deltas.reshape(-1, d)
                    total += block.T @ block / N
                else:
                    total += np.einsum("sci,scj->cij", deltas, deltas)
            chol = np.linalg.cholesky(total / (step + 1) * ridge)
        z = draws.normal(d)
        if pooled:
            steps = np.exp(0.5 * log_scale)[:, None] * matvec(chol, z)
        else:
            steps = np.exp(0.5 * log_scale)[:, None] * np.einsum("cij,cj->ci", chol, z)
        if trace:
            c00 = (chol[0] ** 2).sum() if pooled else (chol[:, 0] ** 2).sum(axis=1)
            tr_v0[step] = np.exp(log_scale) * c00
        proposal_log_p = target.log_prob(x + steps)
        with np.errstate(invalid="ignore"):
            log_alpha = np.minimum(0.0, proposal_log_p - log_p)
        accept = draws.log_uniform() < log_alpha
        x = np.where(accept[:, None], x + steps, x)
        log_p = np.where(accept, proposal_log_p, log_p)
        log_scale += (step + 2) ** -0.6 * (np.exp(np.fmax(log_alpha, -np.inf)) - 0.234)
        if pooled:
            deltas[step % stride] = x - mean
            mean += deltas[step % stride].mean(axis=0) / (step + 2)
        else:
            deltas[step % stride] = x - mean
            mean += deltas[step % stride] / (step + 2)
        if trace:
            tr_x0[step], tr_lp[step] = x[:, 0], log_p
        behind = n_steps - 1 - step
        if step >= first_kept:
            accepted += accept
            if behind % stride == 0:
                samples[:, n_kept - 1 - behind // stride] = x
                kept_log_p[:, n_kept - 1 - behind // stride] = log_p
    run = PilotRun(samples, kept_log_p, accepted / (n_steps - first_kept), n_steps)
    return (run, points, tr_x0, tr_lp, tr_v0) if trace else (run, points)


def proposal_of(run):
    return MultivariateT(run.mean, run.cov, df=5)


def probe(target, initial, run, rng, m, n):
    """m random-walk chains from fresh initial draws, covariance 2.38^2/d times the pilot's.

    Returns log pi, log q and x0 along each chain, shapes (n + 1, m): row 0 is the start.
    """
    initial = as_initial(initial)
    q = proposal_of(run)
    d = len(run.mean)
    x = np.array([draw(initial, rng, d) for _ in range(m)])
    chol = np.linalg.cholesky(run.cov * 2.38**2 / d)
    lp = target.log_prob(x)
    out_lp = np.empty((n + 1, m))
    out_lq = np.empty((n + 1, m))
    out_x0 = np.empty((n + 1, m))
    out_lp[0], out_lq[0], out_x0[0] = lp, q.log_density(x), x[:, 0]
    for t in range(1, n + 1):
        y = x + rng.standard_normal((m, d)) @ chol.T
        lpy = target.log_prob(y)
        with np.errstate(invalid="ignore"):
            accept = np.log(rng.uniform(size=m)) < lpy - lp
        x = np.where(accept[:, None], y, x)
        lp = np.where(accept, lpy, lp)
        out_lp[t], out_lq[t], out_x0[t] = lp, q.log_density(x), x[:, 0]
    return out_lp, out_lq, out_x0

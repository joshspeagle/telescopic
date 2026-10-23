"""Tuning on pilot runs: adaptive Metropolis chains, and the choice of k, m and the lag."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from telescopic._linalg import matvec
from telescopic._random import TUNING, Streams, replicate_generator, root_seed
from telescopic._types import Distribution, FloatArray, Seed
from telescopic.initial import draw
from telescopic.target import Target

if TYPE_CHECKING:
    from telescopic.kernels.base import Kernel

TARGET_ACCEPTANCE = 0.234
N_CHAINS = 4
DUAL_AVERAGING_STEPS = 1000


@dataclass(frozen=True, eq=False)
class TuningReport:
    r"""What a tuning pilot found (spec R-TUNE-1, section 5.3).

    Attributes
    ----------
    mean, cov : ndarray, shapes (d,) and (d, d)
        Mean and covariance of the pilot chains' second halves, pooled.
    acceptance : ndarray, shape (4,)
        Acceptance rates of the chains' second halves.
    rhat : ndarray, shape (d,)
        Gelman-Rubin :math:`\hat R` of the second halves; well above 1 when the chains
        disagree, a sign of multimodality.
    n_steps : int
        Steps per chain.
    max_weight_ratio : float, optional
        For an independence kernel, the largest weight :math:`\pi/q` of the pilot's draws
        relative to their median: large values announce heavy-tailed meeting times.
    """

    mean: FloatArray
    cov: FloatArray
    acceptance: FloatArray
    rhat: FloatArray
    n_steps: int
    max_weight_ratio: float | None = None


@dataclass(frozen=True, eq=False)
class PilotRun:
    """The second halves of adaptive Metropolis pilot chains, from :func:`adaptive_metropolis`.

    Attributes
    ----------
    samples : ndarray, shape (4, n, d)
    log_prob : ndarray, shape (4, n)
    acceptance : ndarray, shape (4,)
    n_steps : int
    """

    samples: FloatArray
    log_prob: FloatArray
    acceptance: FloatArray
    n_steps: int

    @cached_property
    def mean(self) -> FloatArray:
        """The pooled mean, shape ``(d,)``."""
        mean: FloatArray = self.samples.reshape(-1, self.samples.shape[2]).mean(axis=0)
        return mean

    @cached_property
    def cov(self) -> FloatArray:
        """The pooled covariance, plus a ridge of 1e-10 times the mean variance, ``(d, d)``."""
        pooled = self.samples.reshape(-1, self.samples.shape[2])
        cov = np.atleast_2d(np.cov(pooled, rowvar=False))
        ridged: FloatArray = cov + 1e-10 * np.mean(np.diag(cov)) * np.eye(len(cov))
        return ridged

    @cached_property
    def rhat(self) -> FloatArray:
        """Gelman-Rubin :math:`\\hat R` of each coordinate, shape ``(d,)``."""
        n = self.samples.shape[1]
        within = self.samples.var(axis=1, ddof=1).mean(axis=0)
        between = n * self.samples.mean(axis=1).var(axis=0, ddof=1)
        rhat: FloatArray = np.sqrt(((n - 1) / n * within + between / n) / within)
        return rhat

    def report(self, max_weight_ratio: float | None = None) -> TuningReport:
        """The :class:`TuningReport` of this run."""
        return TuningReport(
            self.mean, self.cov, self.acceptance, self.rhat, self.n_steps, max_weight_ratio
        )


def adaptive_metropolis(
    target: Target,
    initial: Distribution,
    rng: Seed = None,
    *,
    n_steps: int | None = None,
) -> PilotRun:
    r"""Four adaptive Metropolis chains that pool their adaptation (spec section 5.3).

    Chain :math:`c` proposes :math:`x' \sim \mathcal N(x_c, \lambda_c\Sigma)`. After step
    :math:`t = 1, 2, \dots` its scale moves toward an acceptance rate of 0.234, and the
    mean and covariance become running averages of every chain's states:

    .. math::
        \log\lambda_c \mathrel{+}= \frac{\alpha_{c,t} - 0.234}{(t + 1)^{0.6}}, \quad
        \mu \mathrel{+}= \frac{\bar\delta}{t + 1}, \quad
        \Sigma \mathrel{+}= \frac{1}{t + 1}\Big(\frac14\sum_c \delta_c\delta_c^\top
        - \Sigma\Big),

    with :math:`\delta_c = x_c - \mu`, from :math:`\lambda_c = 2.38^2/d`, the mean of the
    starting points and the diagonal of the variances of 100 draws from ``initial``
    (Haario, Saksman & Tamminen 2001, with the global scales of Andrieu & Thoms 2008,
    section 5, pooled across chains as in Craiu, Rosenthal & Yang 2009). Chain ``i``
    starts from the first of its 25 draws, and draws from replicate ``i`` of the tuning
    namespace (spec section 6.1).

    Parameters
    ----------
    target : Target
    initial : object with rvs(random_state=generator)
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator
    n_steps : int, optional
        Steps per chain, at least 3; :math:`\max(2000, 200d, 10d^2)` by default.

    Returns
    -------
    PilotRun
        The second halves of the chains.

    Notes
    -----
    Cost: ``4 n_steps`` log-density evaluations and ``n_steps`` Cholesky factorisations.
    """
    if n_steps is not None and n_steps < 3:
        # The second half of each chain needs two states for R-hat.
        raise ValueError(f"n_steps must be at least 3, not {n_steps}")
    root = root_seed(rng)
    generators = [replicate_generator(root, TUNING, i) for i in range(N_CHAINS)]
    points = np.array([[draw(initial, generator) for _ in range(25)] for generator in generators])
    x, d = points[:, 0].copy(), points.shape[2]
    # Learning a covariance takes of order d^2 steps (DECISIONS #34).
    n_steps = max(2000, 200 * d, 10 * d * d) if n_steps is None else n_steps
    streams = Streams(N_CHAINS)
    for chain, generator in enumerate(generators):
        streams.assign(chain, generator)
    draws = streams.view(np.arange(N_CHAINS))
    log_p = target.log_prob(x)
    log_scale = np.full(N_CHAINS, np.log(2.38**2 / d))
    # The initial distribution's scales start the covariance; the identity would dominate
    # small variances for thousands of steps.
    variance = points.reshape(-1, d).var(axis=0, ddof=1)
    mean, cov, eye = x.mean(axis=0), np.diag(np.where(variance > 0, variance, 1.0)), np.eye(d)
    first_kept = n_steps // 2
    samples = np.empty((N_CHAINS, n_steps - first_kept, d))
    kept_log_p = np.empty((N_CHAINS, n_steps - first_kept))
    accepted = np.zeros(N_CHAINS)
    for step in range(n_steps):
        chol = np.linalg.cholesky(cov + 1e-12 * np.trace(cov) * eye)
        steps = np.exp(0.5 * log_scale)[:, None] * matvec(chol, draws.normal(d))
        proposal_log_p = target.log_prob(x + steps)
        # A NaN log ratio (-inf - -inf) is a rejection, with acceptance probability 0.
        with np.errstate(invalid="ignore"):
            log_alpha = np.minimum(0.0, proposal_log_p - log_p)
        accept = draws.log_uniform() < log_alpha
        x = np.where(accept[:, None], x + steps, x)
        log_p = np.where(accept, proposal_log_p, log_p)
        log_scale += (step + 2) ** -0.6 * (np.nan_to_num(np.exp(log_alpha)) - TARGET_ACCEPTANCE)
        delta = x - mean
        mean += delta.mean(axis=0) / (step + 2)
        cov += (delta.T @ delta / N_CHAINS - cov) / (step + 2)
        if step >= first_kept:
            samples[:, step - first_kept] = x
            kept_log_p[:, step - first_kept] = log_p
            accepted += accept
    if not accepted.all():
        raise ValueError(
            f"tuning chain {int(np.argmin(accepted))} accepted no proposal in its second "
            "half: start inside the support of the target, or give the kernel's parameters"
        )
    return PilotRun(samples, kept_log_p, accepted / samples.shape[1], n_steps)


def dual_averaging(
    kernel_at: Callable[[float], Kernel],
    target: Target,
    pilot: PilotRun,
    rng: Seed,
    target_acceptance: float,
    step_size: float,
) -> float:
    r"""A step size whose mean acceptance probability is ``target_acceptance`` (spec 5.4, 5.5).

    One chain per pilot chain runs ``kernel_at(step_size)`` from that chain's last state.
    After step :math:`t` the step size becomes
    :math:`\epsilon_t = \exp(\mu - \sqrt t\,\bar H_t/\gamma)`, where :math:`\bar H_t`
    averages the target acceptance minus the chains' mean acceptance probability with
    offset :math:`t_0`; the result is :math:`\bar\epsilon`, the average of
    :math:`\log\epsilon_t` with weights :math:`t^{-\kappa}` (Hoffman & Gelman 2014,
    section 3.2.1, with :math:`\gamma = 0.05`, :math:`t_0 = 10`, :math:`\kappa = 0.75`,
    :math:`\mu = \log(10\epsilon_0)` and 1000 steps). Chain ``i`` draws from replicate
    ``4 + i`` of the tuning namespace, after the streams of :func:`adaptive_metropolis`.

    Parameters
    ----------
    kernel_at : callable
        Maps a step size to a kernel, cheaply: it is called at every step.
    target : Target
    pilot : PilotRun
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator
    target_acceptance : float
    step_size : float
        The first step size :math:`\epsilon_0`.

    Returns
    -------
    float
    """
    root = root_seed(rng)
    streams = Streams(N_CHAINS)
    for chain in range(N_CHAINS):
        streams.assign(chain, replicate_generator(root, TUNING, N_CHAINS + chain))
    draws = streams.view(np.arange(N_CHAINS))
    state = kernel_at(step_size).init_state(target, pilot.samples[:, -1])
    mu, h_bar, log_average = np.log(10 * step_size), 0.0, 0.0
    for t in range(1, DUAL_AVERAGING_STEPS + 1):
        state, acceptance = kernel_at(step_size).step(target, state, draws)
        h_bar += (target_acceptance - float(np.mean(acceptance)) - h_bar) / (t + 10)
        log_step = mu - np.sqrt(t) * h_bar / 0.05
        log_average += t**-0.75 * (log_step - log_average)
        step_size = float(np.exp(log_step))
    return float(np.exp(log_average))


def choose_k(meeting_times: npt.ArrayLike, quantile: float = 0.99) -> int:
    r"""Choose ``k`` from lag-1 pilot meeting times (spec R-TUNE-2).

    :math:`k = \max(1, \hat q_\beta(\tau - 1))`, with :math:`\hat q_\beta` the
    inverted-CDF sample quantile (Atchade & Jacob 2024). The recipe then sets
    :math:`L = k` and :math:`m = 10k`.

    Parameters
    ----------
    meeting_times : array_like, shape (R0,)
        Pilot meeting times with lag 1; ``inf`` for a censored pair, which sorts above
        every finite one.
    quantile : float
        The level :math:`\beta`, at most :math:`1 - 1/R_0` so that the quantile is not
        the sample maximum.

    Returns
    -------
    int
    """
    tau = np.asarray(meeting_times, dtype=np.float64)
    if tau.ndim != 1 or len(tau) == 0:
        raise ValueError("meeting_times must be a non-empty one-dimensional array")
    largest = 1 - 1 / len(tau)
    if not 0 < quantile <= largest:
        raise ValueError(
            f"quantile must be in (0, 1 - 1/R0] = (0, {largest:g}] for {len(tau)} pilot "
            f"pairs, not {quantile}; run more pilot pairs for a higher quantile"
        )
    q = np.quantile(tau - 1, quantile, method="inverted_cdf")
    if not np.isfinite(q):
        raise ValueError(
            f"more than {1 - quantile:.1%} of the pilot pairs never met within their cap, "
            "so k cannot be chosen: try a wider initial distribution or a better-tuned "
            "kernel, or give k, lag and m explicitly"
        )
    return max(1, int(q))

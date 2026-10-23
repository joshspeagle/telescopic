"""Tuning on pilot runs: adaptive Metropolis chains, and the choice of k, m and the lag."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, SeedSequence

from telescopic._random import TUNING, Streams, replicate_generator, root_seed
from telescopic._types import Distribution, FloatArray
from telescopic.initial import draw
from telescopic.target import Target

TARGET_ACCEPTANCE = 0.234


@dataclass(frozen=True)
class TuningReport:
    r"""What a tuning pilot found (spec R-TUNE-1, section 5.3).

    Attributes
    ----------
    mean, cov : ndarray, shapes (d,) and (d, d)
        Mean and covariance of the pilot chains' second halves, pooled.
    scale : float
        The adapted proposal scale: the proposal covariance is ``scale**2 * cov``.
    acceptance : ndarray, shape (n_chains,)
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
    scale: float
    acceptance: FloatArray
    rhat: FloatArray
    n_steps: int
    max_weight_ratio: float | None = None


@dataclass(frozen=True)
class PilotRun:
    """The second halves of adaptive Metropolis pilot chains, from :func:`adaptive_metropolis`.

    Attributes
    ----------
    samples : ndarray, shape (n_chains, n, d)
    log_prob : ndarray, shape (n_chains, n)
    acceptance : ndarray, shape (n_chains,)
    log_scale : ndarray, shape (n_chains,)
        The final adapted :math:`\\log\\lambda` of each chain.
    n_steps : int
    """

    samples: FloatArray
    log_prob: FloatArray
    acceptance: FloatArray
    log_scale: FloatArray
    n_steps: int

    @property
    def mean(self) -> FloatArray:
        """The pooled mean, shape ``(d,)``."""
        mean: FloatArray = self.samples.reshape(-1, self.samples.shape[2]).mean(axis=0)
        return mean

    @property
    def cov(self) -> FloatArray:
        """The pooled covariance, plus a ridge of 1e-10 times the mean variance, ``(d, d)``."""
        pooled = self.samples.reshape(-1, self.samples.shape[2])
        cov = np.atleast_2d(np.cov(pooled, rowvar=False))
        ridged: FloatArray = cov + 1e-10 * np.mean(np.diag(cov)) * np.eye(len(cov))
        return ridged

    @property
    def scale(self) -> float:
        """The proposal scale :math:`\\sqrt\\lambda`, from the chains' geometric mean."""
        return float(np.exp(0.5 * self.log_scale.mean()))

    @property
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
            self.mean,
            self.cov,
            self.scale,
            self.acceptance,
            self.rhat,
            self.n_steps,
            max_weight_ratio,
        )


def adaptive_metropolis(
    target: Target,
    initial: Distribution,
    rng: int | SeedSequence | Generator | None = None,
    *,
    n_chains: int = 4,
    n_steps: int | None = None,
) -> PilotRun:
    r"""Independent adaptive Metropolis chains with global scaling (spec section 5.3).

    Each chain starts from ``initial``, proposes :math:`x' \sim \mathcal N(x, \lambda\Sigma)`,
    and after step :math:`t = 1, 2, \dots` adapts, with :math:`\gamma_t = (t + 1)^{-0.6}`
    and the acceptance probability :math:`\alpha_t`,

    .. math::
        \log\lambda \mathrel{+}= \gamma_t(\alpha_t - 0.234), \quad
        \mu \mathrel{+}= \gamma_t(x - \mu), \quad
        \Sigma \mathrel{+}= \gamma_t\big((x - \mu_{\rm old})(x - \mu_{\rm old})^\top - \Sigma\big)

    (Andrieu & Thoms 2008, section 5, after Haario, Saksman & Tamminen 2001), from
    :math:`\lambda = 2.38^2/d` and :math:`\Sigma = I`. Chain ``i`` draws from replicate
    ``i`` of the tuning namespace (spec section 6.1).

    Parameters
    ----------
    target : Target
    initial : object with rvs(random_state=generator)
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator
    n_chains : int
        At least 2, for :math:`\hat R`.
    n_steps : int, optional
        Steps per chain, at least 3; ``max(2000, 200 d)`` by default.

    Returns
    -------
    PilotRun
        The second halves of the chains.

    Notes
    -----
    Cost: ``n_chains * n_steps`` log-density evaluations and ``n_steps`` batched
    Cholesky factorisations.
    """
    if n_chains < 2:
        raise ValueError(f"n_chains must be at least 2, not {n_chains}")
    root = root_seed(rng)
    generators = [replicate_generator(root, TUNING, i) for i in range(n_chains)]
    x = np.array([draw(initial, generator) for generator in generators])
    d = x.shape[1]
    n_steps = max(2000, 200 * d) if n_steps is None else n_steps
    if n_steps < 3:
        # The second half of each chain needs two states for R-hat.
        raise ValueError(f"n_steps must be at least 3, not {n_steps}")
    streams = Streams(n_chains)
    for chain, generator in enumerate(generators):
        streams.assign(chain, generator)
    draws = streams.view(np.arange(n_chains))
    log_p = target.log_prob(x)
    mean, cov = x.copy(), np.tile(np.eye(d), (n_chains, 1, 1))
    log_scale = np.full(n_chains, np.log(2.38**2 / d))
    first_kept = n_steps // 2
    samples = np.empty((n_chains, n_steps - first_kept, d))
    kept_log_p = np.empty((n_chains, n_steps - first_kept))
    accepted = np.zeros(n_chains)
    for step in range(n_steps):
        proposal_cov = np.exp(log_scale)[:, None, None] * cov
        ridge = 1e-12 * np.trace(proposal_cov, axis1=1, axis2=2)[:, None, None] * np.eye(d)
        chol = np.linalg.cholesky(proposal_cov + ridge)
        proposal = x + np.einsum("cij,cj->ci", chol, draws.normal(d))
        proposal_log_p = target.log_prob(proposal)
        # A NaN log ratio (-inf - -inf) is a rejection, with acceptance probability 0.
        with np.errstate(invalid="ignore"):
            log_alpha = np.minimum(0.0, proposal_log_p - log_p)
        accept = draws.log_uniform() < log_alpha
        x = np.where(accept[:, None], proposal, x)
        log_p = np.where(accept, proposal_log_p, log_p)
        gamma = (step + 2) ** -0.6
        log_scale += gamma * (np.nan_to_num(np.exp(log_alpha)) - TARGET_ACCEPTANCE)
        delta = x - mean
        mean += gamma * delta
        cov += gamma * (delta[:, :, None] * delta[:, None, :] - cov)
        if step >= first_kept:
            samples[:, step - first_kept] = x
            kept_log_p[:, step - first_kept] = log_p
            accepted += accept
    return PilotRun(samples, kept_log_p, accepted / samples.shape[1], log_scale, n_steps)


def choose_kml(
    meeting_times: npt.ArrayLike, quantile: float = 0.99, multiple: int = 10
) -> tuple[int, int, int]:
    r"""Choose ``k``, ``m`` and the lag from lag-1 pilot meeting times (spec R-TUNE-2).

    :math:`k = \max(1, \hat q_\beta(\tau - 1))`, with :math:`\hat q_\beta` the
    inverted-CDF sample quantile, :math:`L = k` and :math:`m` = ``multiple`` :math:`k`
    (Atchade & Jacob 2024).

    Parameters
    ----------
    meeting_times : array_like, shape (R0,)
        Pilot meeting times with lag 1; ``inf`` for a censored pair, which sorts above
        every finite one.
    quantile : float
        The level :math:`\beta`, at most :math:`1 - 1/R_0` so that the quantile is not
        the sample maximum.
    multiple : int
        :math:`m / k`.

    Returns
    -------
    k, m, lag : int
    """
    tau = np.asarray(meeting_times, dtype=np.float64)
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
    k = max(1, int(q))
    return k, multiple * k, k

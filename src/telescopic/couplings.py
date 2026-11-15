"""Couplings of pairs of distributions: the primitives that coupled kernels build on."""

from __future__ import annotations

import functools
import math
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from scipy.special import softmax

from telescopic._linalg import matvec
from telescopic._random import Draws
from telescopic._types import BoolArray, Distribution, FloatArray, IndexArray, real

if TYPE_CHECKING:
    from telescopic.kernels.base import State
    from telescopic.kernels.metropolis import (
        Proposal,
        RandomWalkProposal,
        SharedCovarianceGaussian,
    )
    from telescopic.target import Target

__all__ = [
    "CoupledProposals",
    "Coupling",
    "Maximal",
    "Reflection",
    "Shared",
    "discrete_maximal",
    "maximal",
    "reflection_maximal",
]


class CoupledProposals(NamedTuple):
    """The coupled proposals of ``n`` pairs of chains, with what each proposal's ratio needs.

    Attributes
    ----------
    x, y : ndarray, shape (n, d)
        The proposals of the first and the second chains. On identical rows ``y``
        equals ``x``: a copy of its rows, or ``x`` itself when every pair shares one
        draw, as under :class:`Shared`; a coupled step then evaluates it once.
    identical : ndarray of bool, shape (n,)
        Rows where the two proposals are equal.
    forward_x, forward_y : object
        What :meth:`~telescopic.kernels.Proposal.log_alpha` needs of each chain's draw, as
        :meth:`~telescopic.kernels.Proposal.sample` returns it.
    """

    x: FloatArray
    y: FloatArray
    identical: BoolArray
    forward_x: Any
    forward_y: Any


class Coupling(ABC):
    """A coupling of the proposals of two chains under one proposal.

    Each chain's proposal must have exactly the law of the proposal's
    :meth:`~telescopic.kernels.Proposal.sample` from its state, and two chains in equal
    states must receive identical proposals, with ``y`` equal to ``x``, a copy of its
    rows or ``x`` itself, never a recomputation.
    """

    @abstractmethod
    def couple(
        self,
        target: Target,
        proposal: Proposal,
        state_x: State,
        state_y: State,
        draws: Draws,
    ) -> CoupledProposals:
        """Draw the coupled proposals of ``n`` pairs.

        Parameters
        ----------
        target : Target
            The target that the chains sample.
        proposal : Proposal
            The proposal of both chains, one that :meth:`supports` accepts.
        state_x, state_y : State
            The first and the second chains of the pairs.
        draws : Draws
            Random numbers for the ``n`` pairs.

        Returns
        -------
        CoupledProposals
        """

    @abstractmethod
    def supports(self, proposal: Proposal) -> bool:
        """Whether this coupling couples ``proposal``, matched by its exact type.

        Parameters
        ----------
        proposal : Proposal
            The proposal to check.

        Returns
        -------
        bool
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# Spec R-CPL-1.
class Reflection(Coupling):
    r"""The reflection-maximal coupling of two Gaussian proposals with a shared covariance.

    With :math:`\delta = B^{-1}(\mu_x - \mu_y)`, the whitened difference of the proposals'
    means, :math:`z \sim \mathcal N(0, I)` and :math:`x' = \mu_x + Bz`, the proposals are
    identical when :math:`\log u < -z\cdot\delta - \lVert\delta\rVert^2/2`; otherwise
    :math:`y' = \mu_y + B(z - 2(e\cdot z)e)` with :math:`e = \delta/\lVert\delta\rVert`
    (Jacob, O'Leary & Atchadé 2020, sections 4.1 and 4.4). It couples the random-walk
    and the Langevin proposals.
    """

    def couple(
        self,
        target: Target,
        proposal: Proposal,
        state_x: State,
        state_y: State,
        draws: Draws,
    ) -> CoupledProposals:
        return cast("SharedCovarianceGaussian", proposal).reflect(target, state_x, state_y, draws)

    def supports(self, proposal: Proposal) -> bool:
        from telescopic.kernels.metropolis import LangevinProposal, RandomWalkProposal

        return type(proposal) in (RandomWalkProposal, LangevinProposal)


# Spec R-CPL-2, with eta = 1.
class Maximal(Coupling):
    r"""The rejection maximal coupling of two random-walk proposals.

    Each pair draws from its own generator by the rejection maximal coupling with
    :math:`\eta = 1` of :math:`\mathcal N(x, BB^\top)` and
    :math:`\mathcal N(y, BB^\top)` (Jacob, O'Leary & Atchadé 2020, Algorithm 2). Its
    meeting probability is maximal, but its meeting times grow exponentially with the
    dimension; it is for comparison.
    """

    def couple(
        self,
        target: Target,
        proposal: Proposal,
        state_x: State,
        state_y: State,
        draws: Draws,
    ) -> CoupledProposals:
        walk = cast("RandomWalkProposal", proposal)
        factor, inverse = walk.factor, walk.inverse
        x, y, identical, _ = maximal(
            [_Gaussian(mean, factor, inverse) for mean in state_x.x],
            [_Gaussian(mean, factor, inverse) for mean in state_y.x],
            draws,
        )
        return CoupledProposals(x, y, identical, None, None)

    def supports(self, proposal: Proposal) -> bool:
        from telescopic.kernels.metropolis import RandomWalkProposal

        return type(proposal) is RandomWalkProposal


class _Gaussian:
    # A Gaussian proposal for the rejection coupling; both members of a pair share chol,
    # so the normalizing constants cancel and are left out.
    def __init__(self, mean: FloatArray, chol: FloatArray, inverse: FloatArray) -> None:
        self.mean, self.chol, self.inverse = mean, chol, inverse

    def rvs(self, *, random_state: np.random.Generator) -> FloatArray:
        return self.mean + self.chol @ random_state.standard_normal(len(self.mean))

    def logpdf(self, x: FloatArray) -> float:
        z = self.inverse @ (x - self.mean)
        return float(-0.5 * z @ z)


# Spec section 5.6: the coupled step shares the proposal and the uniform.
class Shared(Coupling):
    """The coupling of two independence proposals that gives both chains one proposal.

    With the shared uniform of the acceptance, it is a maximal coupling of the two
    kernels: a pair meets when both chains accept (O'Leary, Wang & Jacob 2021).
    """

    def couple(
        self,
        target: Target,
        proposal: Proposal,
        state_x: State,
        state_y: State,
        draws: Draws,
    ) -> CoupledProposals:
        # One array for both chains, which the coupled step evaluates once.
        x, forward = proposal.sample(target, state_x, draws)
        return CoupledProposals(x, x, np.ones(len(x), dtype=bool), forward, forward)

    def supports(self, proposal: Proposal) -> bool:
        from telescopic.kernels.metropolis import IndependenceProposal

        return type(proposal) is IndependenceProposal


# Spec R-CPL-1.
def reflection_maximal(
    mu_x: FloatArray,
    mu_y: FloatArray,
    chol: FloatArray,
    inverse: FloatArray,
    draws: Draws,
    *,
    blas: bool = False,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    r"""Reflection-maximal coupling of :math:`\mathcal N(\mu_x, CC^\top)` and :math:`\mathcal N(\mu_y, CC^\top)`.

    With :math:`\delta = C^{-1}(\mu_x - \mu_y)`, :math:`z \sim \mathcal N(0, I)` and
    :math:`x = \mu_x + Cz`, the pair is identical when
    :math:`\log u < -z\cdot\delta - \lVert\delta\rVert^2/2`; otherwise
    :math:`y = \mu_y + C(z - 2(e\cdot z)e)` with :math:`e = \delta/\lVert\delta\rVert`, or
    :math:`e = 0` where :math:`\lVert\delta\rVert^2` is 0 or not finite
    (Jacob, O'Leary & Atchadé 2020, section 4.1). The coupling is maximal:
    :math:`P(x = y) = 2\Phi(-\lVert\delta\rVert/2)`.

    Parameters
    ----------
    mu_x, mu_y : ndarray, shape (n, d)
        Means of the two Gaussians of each row.
    chol : ndarray, shape (d, d)
        Lower-triangular factor :math:`C` of the shared covariance.
    inverse : ndarray, shape (d, d)
        :math:`C^{-1}`, which whitens the difference of the means.
    draws : Draws
        Random numbers for the ``n`` rows; each row uses ``d`` normals and 1 uniform.
    blas : bool
        Whether to compute the products with :math:`C` and :math:`C^{-1}` by BLAS
        (``vectors @ matrix.T``), which gives a row bits that can depend on the other rows
        and on the number of BLAS threads; kernels set it for vectorized targets.
        Otherwise a row's arithmetic does not depend on the other rows.

    Returns
    -------
    x, y : ndarray, shape (n, d)
        The coupled draws. On identical rows ``y`` is a copy of ``x``, so equality is exact.
    identical : ndarray of bool, shape (n,)
        Whether each row's draws are equal.
    """
    # Infinite means, from a drift that overflowed, can give inf - inf = NaN, which fails
    # the test as it should.
    with np.errstate(invalid="ignore"):
        difference = mu_x - mu_y
        delta = matvec(inverse, difference, blas=blas)
    # C e = (mu_x - mu_y) / |delta|, so C(z - 2(e.z)e) = Cz - 2(z.delta)/|delta|^2
    # (mu_x - mu_y), which reuses Cz.
    z, identical, shift = _reflection(delta, draws, difference)
    step = matvec(chol, z, blas=blas)
    x = mu_x + step
    y = np.where(identical[:, None], x, mu_y + step - shift)
    return x, y, identical


def _reflection(
    delta: FloatArray, draws: Draws, direction: FloatArray
) -> tuple[FloatArray, BoolArray, FloatArray]:
    # The reflection-maximal coupling for the whitened difference of the means delta, which
    # MALA's whitened step also uses: the normals z, the rows where the pair is identical
    # (log u < -z.delta - |delta|^2 / 2; a NaN in delta fails the test), and the shift
    # 2 (z.delta) / |delta|^2 direction that reflects z, with direction = delta in whitened
    # coordinates. The shift is 0 where |delta|^2 is 0 or not finite, so y keeps its law
    # there; the mask also keeps 0 * inf out where direction is not finite.
    z = draws.normal(delta.shape[1])
    log_u = draws.log_uniform()
    # z.delta can overflow where |delta|^2 is already infinite, as from MALA's whitened
    # difference at an enormous step; the mask below discards the coefficient there.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        squared = np.einsum("ij,ij->i", delta, delta)
        along = np.einsum("ij,ij->i", z, delta)
        identical = log_u < -along - 0.5 * squared
        coefficient = 2.0 * along / squared
    finite = (squared > 0) & (squared < np.inf)
    shift = np.multiply(
        coefficient[:, None], direction, out=np.zeros_like(direction), where=finite[:, None]
    )
    return z, identical, shift


# Spec R-CPL-2.
def maximal(
    p: list[Distribution], q: list[Distribution], draws: Draws, eta: float = 1.0
) -> tuple[FloatArray, FloatArray, BoolArray, IndexArray]:
    r"""Rejection maximal coupling of the distributions ``p[i]`` and ``q[i]`` of each row.

    Draw :math:`X \sim p`; the pair is identical with probability
    :math:`\min(\eta, q(X)/p(X))`. Otherwise draw :math:`Y \sim q` until
    :math:`u > \eta\,p(Y)/q(Y)` (Thorisson's algorithm, Jacob, O'Leary & Atchadé 2020
    Algorithm 2; with :math:`\eta < 1`, the bounded-variance version of Gerber & Lee 2020,
    Atchadé & Jacob 2024 Algorithm 2). The meeting probability is
    :math:`\int\min(\eta p, q)` over the draws whose log-densities are finite, maximal at
    :math:`\eta = 1`; the number of draws from ``p`` and ``q`` has mean 2 and variance
    at most :math:`2\eta/(1-\eta)`.

    Where either log-density of a draw is infinite, the pair is not identical and the
    loop accepts the draw. Such a draw, as one that underflowed to the edge of its
    support, stands for the mass of an interval whose ratio of densities is unknown, and
    each chain keeps its sampler's law. Such draws can be frequent: a third of
    Beta(0.01, 0.01)'s are 1.0, its mass within :math:`2^{-53}` of 1, where the pair is
    not identical even for :math:`p = q`, though its draws can be equal.

    Parameters
    ----------
    p, q : list of distributions, length n
        Objects with ``rvs(random_state=...)`` and a normalized ``logpdf`` (or ``logpmf``),
        such as frozen SciPy distributions. All draws have the same shape. Values of
        ``logpdf`` for the coordinates of a draw are summed, as the joint log-density of
        independent coordinates; NaN raises ``ValueError``.
    draws : Draws
        Random numbers for the ``n`` rows; each row draws from its own generator.
    eta : float
        Cap on the meeting probability, in (0, 1].

    Returns
    -------
    x, y : ndarray, shape (n, ...)
        The coupled draws; on identical rows ``y`` is a copy of ``x``.
    identical : ndarray of bool, shape (n,)
    n_draws : ndarray of int, shape (n,)
        Draws from ``p`` and ``q`` used by each row.
    """
    # NumPy's log, whose last bit can differ from math.log's.
    log_eta = float(np.log(real("eta", eta, 0, 1, closed=True)))
    xs, ys, n_draws = [], [], np.ones(len(p), np.intp)
    for row, (p_row, q_row) in enumerate(zip(p, q, strict=True)):
        generator = draws.generator(row)
        x = y = p_row.rvs(random_state=generator)
        # Not finite exactly where either log-density is infinite.
        log_ratio = _log_density(q_row, x) - _log_density(p_row, x)
        accepted = math.isfinite(log_ratio) and _log_uniform(generator) < min(log_eta, log_ratio)
        while not accepted:
            y = q_row.rvs(random_state=generator)
            n_draws[row] += 1
            log_ratio = _log_density(p_row, y) - _log_density(q_row, y)
            accepted = (
                not math.isfinite(log_ratio) or _log_uniform(generator) > log_eta + log_ratio
            )
        xs.append(x)
        ys.append(y)
    x_array, y_array = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    return x_array, y_array, n_draws == 1, n_draws


# Spec R-CPL-3.
def discrete_maximal(
    log_p: FloatArray, log_q: FloatArray, draws: Draws
) -> tuple[IndexArray, IndexArray]:
    r"""Maximal coupling of two distributions on :math:`\{0, \dots, K-1\}`, row by row.

    With probability :math:`\alpha = \sum_k \min(p_k, q_k)` both indices are drawn from
    :math:`\min(p, q)/\alpha`; otherwise they are drawn independently from the normalized
    residuals :math:`p - \min(p, q)` and :math:`q - \min(p, q)`, whose supports are
    disjoint (Jacob, O'Leary & Atchadé 2020, section 5.4). So
    :math:`P(i = j) = \alpha = 1 - \mathrm{TV}(p, q)`.

    Parameters
    ----------
    log_p, log_q : ndarray, shape (n, K)
        Log-probabilities, normalized or not; each row needs a finite maximum and no NaN.
    draws : Draws
        Random numbers for the ``n`` rows; each row uses 3 uniforms.

    Returns
    -------
    i, j : ndarray of int, shape (n,)
        The coupled indices.
    """
    for name, log in (("log_p", log_p), ("log_q", log_q)):
        # The maximum is NaN for a row with NaN, and infinite for one of zero or infinite mass.
        valid = np.isfinite(np.max(log, axis=1))
        if not valid.all():
            row = int(np.argmin(valid))
            raise ValueError(
                f"each row of {name} needs a finite maximum and no NaN, but row {row} is "
                f"{log[row].tolist()}"
            )
    p, q = softmax(log_p, axis=1), softmax(log_q, axis=1)
    overlap = np.minimum(p, q)
    alpha = overlap.sum(axis=1)
    meet = draws.uniform() < alpha
    u_i, u_j = draws.uniform(), draws.uniform()
    i = np.where(meet, _categorical(overlap, u_i), _categorical(p - overlap, u_i))
    j = np.where(meet, i, _categorical(q - overlap, u_j))
    return i, j


def _log_uniform(generator: np.random.Generator) -> float:
    u = generator.random()
    return float(np.log(u)) if u > 0 else -np.inf


def _log_density(distribution: Any, value: Any) -> float:
    name = "logpdf" if hasattr(distribution, "logpdf") else "logpmf"
    # A frozen Dirichlet exists only once scipy.stats is imported, so a process that has
    # not imported it holds none and need not import it to check.
    if "scipy.stats" in sys.modules and isinstance(distribution, _dirichlet_type()):
        # SciPy's dirichlet draws shape (1, k), and its logpdf reads the first axis as the
        # components: it refuses (1, k) for k > 2, and reads (1, 2) as two points.
        value = np.ravel(value)
        # It also refuses a component of 0 whose alpha is below 1, a pole that draws reach
        # by underflow.
        if np.any((value == 0) & (distribution.alpha < 1)):
            return math.inf
    log_density = getattr(distribution, name)(value)
    # One value per coordinate, from array parameters: the joint log-density of
    # independent coordinates is their sum. A float skips np.sum, which doubled the time
    # of a coupled Gibbs sweep of examples.pumps. A Python float, as NumPy's warns at
    # inf - inf.
    log_density = float(log_density if isinstance(log_density, float) else np.sum(log_density))
    if math.isnan(log_density):
        raise ValueError(f"{name} returned nan at {np.asarray(value).tolist()}")
    return log_density


@functools.cache
def _dirichlet_type() -> type[Any]:
    from scipy import stats

    return type(stats.dirichlet([1.0, 1.0]))


def _categorical(weights: FloatArray, u: FloatArray) -> IndexArray:
    # Inverse-CDF draw from unnormalized non-negative weights: the first index whose
    # cumulative weight exceeds u times the total, so zero-weight categories are never
    # drawn. The clip only matters for rounding at u -> 1 and for an all-zero row (a
    # residual that is never used).
    cumulative = np.cumsum(weights, axis=1)
    index: IndexArray = np.minimum(
        (cumulative <= (u * cumulative[:, -1])[:, None]).sum(axis=1), weights.shape[1] - 1
    )
    return index

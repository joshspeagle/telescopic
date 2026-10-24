"""Hamiltonian Monte Carlo with a contractive coupling and a random-walk meeting step."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.typing as npt

from telescopic._linalg import matvec
from telescopic._random import Draws
from telescopic._types import FloatArray, integer, real
from telescopic.kernels.base import GradientKernel, State, acceptance, evaluate_proposal, select
from telescopic.kernels.mixture import Mixture
from telescopic.kernels.random_walk import RandomWalk
from telescopic.target import Target

# The most leapfrog steps of a default trajectory, so that a step size that dual averaging
# drives toward 0 cannot make a trajectory's cost grow without bound (DECISIONS #17).
MAX_STEPS = 1000


class HMC(Mixture):
    r"""Hamiltonian Monte Carlo with a random-walk meeting step (spec R-KER-HMC).

    With probability :math:`1 - \gamma`, a leapfrog trajectory: from a whitened momentum
    :math:`\nu \sim \mathcal N(0, I)`, ``n_steps`` updates

    .. math::
        \nu \leftarrow \nu + \tfrac\epsilon2 C^\top\nabla\log\pi(x), \quad
        x \leftarrow x + \epsilon C\nu, \quad
        \nu \leftarrow \nu + \tfrac\epsilon2 C^\top\nabla\log\pi(x),

    with :math:`CC^\top = \Sigma` (``cov``), accepted on the change in
    :math:`\log\pi(x) - \tfrac12\lVert\nu\rVert^2`. A trajectory that reaches a non-finite
    position or a log-density of ``-inf`` stops there and is rejected. With probability
    :math:`\gamma` (``meeting_weight``), a random-walk step with proposal covariance
    :math:`\sigma^2\Sigma` (``meeting_scale`` :math:`= \sigma`).

    Coupled chains take the same component. Their trajectories share the momentum and the
    uniform, so the chains contract; the random-walk step couples them by the
    reflection-maximal coupling, so chains that have come close meet (Heng & Jacob 2019,
    with a whitened meeting step, D-24).

    Parameters
    ----------
    step_size : float, optional
        :math:`\epsilon`. ``None`` leaves it to :meth:`~telescopic.kernels.Kernel.tune`: dual averaging of the
        trajectories alone toward an acceptance rate of 0.8, from
        :math:`\epsilon_0 = d^{-1/4}` (spec section 5.5; Beskos et al. 2013).
    n_steps : int, optional
        Leapfrog steps per trajectory; :math:`\lceil 1/\epsilon\rceil` by default, a
        trajectory of length about 1 in whitened coordinates, and at most 1,000.
    cov : array_like, shape (d, d), optional
        :math:`\Sigma`; ``None`` leaves it to :meth:`~telescopic.kernels.Kernel.tune`, which takes the covariance of
        adaptive Metropolis pilot chains.
    meeting_weight : float
        :math:`\gamma`, in (0, 1).
    meeting_scale : float
        :math:`\sigma`, positive.

    Notes
    -----
    A step uses 1 uniform to choose its component, then ``d`` normals and 1 uniform; a
    coupled step uses the same, plus 1 uniform for a meeting step. A trajectory evaluates
    the log-density and its gradient ``n_steps`` times per chain.
    """

    def __init__(
        self,
        step_size: float | None = None,
        n_steps: int | None = None,
        *,
        cov: npt.ArrayLike | None = None,
        meeting_weight: float = 0.05,
        meeting_scale: float = 1e-3,
    ) -> None:
        meeting_weight = real("meeting_weight", meeting_weight)
        meeting_scale = real("meeting_scale", meeting_scale)
        if not 0 < meeting_weight < 1:
            raise ValueError(f"meeting_weight must be in (0, 1), not {meeting_weight}")
        if not 0 < meeting_scale < np.inf:
            raise ValueError(f"meeting_scale must be positive and finite, not {meeting_scale}")
        leapfrog = _Leapfrog(step_size, n_steps, cov)
        walk = RandomWalk(leapfrog.cov, scale=meeting_scale)
        super().__init__([leapfrog, walk], [1 - meeting_weight, meeting_weight])
        self.meeting_weight, self.meeting_scale = meeting_weight, meeting_scale

    @property
    def step_size(self) -> float | None:
        """The leapfrog step size :math:`\\epsilon`."""
        return self._leapfrog.step_size

    @property
    def n_steps(self) -> int | None:
        """Leapfrog steps per trajectory, or ``None`` while the step size is unknown."""
        return self._leapfrog.n_steps

    @property
    def cov(self) -> FloatArray | None:
        """:math:`\\Sigma`."""
        return self._leapfrog.cov

    @property
    def _leapfrog(self) -> _Leapfrog:
        # Read from the components, which tuning replaces in a copy (Mixture.tune).
        return cast(_Leapfrog, self.kernels[0])

    def __repr__(self) -> str:
        cov = "None" if self.cov is None else f"<{len(self.cov)}x{len(self.cov)}>"
        return (
            f"HMC(step_size={self.step_size}, n_steps={self.n_steps}, cov={cov}, "
            f"meeting_weight={self.meeting_weight}, meeting_scale={self.meeting_scale})"
        )


class _Leapfrog(GradientKernel):
    # The trajectory component of HMC. Coupled chains share the momentum and the uniform.
    target_acceptance = 0.8

    def __init__(
        self, step_size: float | None, n_steps: int | None, cov: npt.ArrayLike | None
    ) -> None:
        super().__init__(step_size, cov)
        if n_steps is not None:
            n_steps = integer("n_steps", n_steps, "an integer or None", minimum=1)
        self._n_steps = n_steps

    @property
    def n_steps(self) -> int | None:
        # The given count, or ceil(1 / step_size): a trajectory of length about 1 in
        # whitened coordinates (spec section 5.5), of at most MAX_STEPS steps.
        if self._n_steps is not None or self.step_size is None:
            return self._n_steps
        return math.ceil(1 / max(self.step_size, 1 / MAX_STEPS))

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, FloatArray]:
        end, log_alpha = self._trajectory(target, state, draws.normal(state.x.shape[1]))
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, end, state), acceptance(log_alpha)

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        # Both chains in one batch, so that each leapfrog step calls the target once.
        n = len(state_x)
        momentum = draws.normal(state_x.x.shape[1])
        both = State(
            np.concatenate([state_x.x, state_y.x]),
            np.concatenate([state_x.log_prob, state_y.log_prob]),
            np.concatenate([cast(FloatArray, state_x.grad), cast(FloatArray, state_y.grad)]),
        )
        end, log_alpha = self._trajectory(target, both, np.concatenate([momentum, momentum]))
        log_u = draws.log_uniform()
        moved = select(np.concatenate([log_u, log_u]) < log_alpha, end, both)
        grad = cast(FloatArray, moved.grad)
        first = State(moved.x[:n], moved.log_prob[:n], grad[:n])
        return first, State(moved.x[n:], moved.log_prob[n:], grad[n:])

    def _with_cov(self, cov: FloatArray) -> _Leapfrog:
        return _Leapfrog(self.step_size, self._n_steps, cov)

    def _first_step_size(self, d: int) -> float:
        return float(d**-0.25)

    def _trajectory(
        self, target: Target, state: State, momentum: FloatArray
    ) -> tuple[State, FloatArray]:
        # The end of each trajectory and its log acceptance ratio, -inf for a trajectory
        # that stopped. Stopped rows keep moving in the arithmetic, never in the target.
        if self._chol is None or self.step_size is None:
            raise ValueError(
                "HMC needs step_size and cov; give them, or tune the kernel on a pilot run"
            )
        chol, epsilon, length = self._chol, self.step_size, cast(int, self.n_steps)
        x, grad = state.x, cast(FloatArray, state.grad)
        going = np.ones(len(x), dtype=bool)
        with np.errstate(over="ignore", invalid="ignore"):
            start = state.log_prob - 0.5 * np.einsum("ij,ij->i", momentum, momentum)
            momentum = momentum + 0.5 * epsilon * matvec(chol.T, grad)
            for leap in range(length):
                x = x + epsilon * matvec(chol, momentum)
                reached = evaluate_proposal(target, x, going)
                log_prob, grad = reached.log_prob, cast(FloatArray, reached.grad)
                going = np.isfinite(log_prob)
                kick = epsilon if leap < length - 1 else 0.5 * epsilon
                momentum = momentum + kick * matvec(chol.T, grad)
            # From a start outside the support (log pi = -inf) any finite end is accepted.
            log_alpha = log_prob - 0.5 * np.einsum("ij,ij->i", momentum, momentum) - start
        return State(x, log_prob, grad), np.where(going, log_alpha, -np.inf)

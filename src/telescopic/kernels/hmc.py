"""Hamiltonian Monte Carlo with a contractive coupling and a random-walk meeting step."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.typing as npt

from telescopic._linalg import covariance, matvec
from telescopic._random import Draws
from telescopic._types import FloatArray, Seed
from telescopic.kernels.base import Kernel, State, acceptance, evaluate_proposal, select
from telescopic.kernels.mixture import Mixture
from telescopic.kernels.random_walk import RandomWalk
from telescopic.target import Target
from telescopic.tuning import PilotRun, TuningReport, dual_averaging

TARGET_ACCEPTANCE = 0.8


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
    position, a log-density of ``-inf`` or a non-finite gradient stops there and is
    rejected. With probability :math:`\gamma` (``meeting_weight``), a random-walk step with
    proposal covariance :math:`\sigma^2\Sigma` (``meeting_scale`` :math:`= \sigma`).

    Coupled chains take the same component. Their trajectories share the momentum and the
    uniform, so the chains contract; the random-walk step couples them by the
    reflection-maximal coupling, so chains that have come close meet (Heng & Jacob 2019,
    with a whitened meeting step, D-24).

    Parameters
    ----------
    step_size : float, optional
        :math:`\epsilon`. ``None`` leaves it to :meth:`tune`, which runs dual averaging
        toward an acceptance rate of 0.8 for the trajectories, from the last states of the
        pilot chains (spec section 5.5).
    n_steps : int, optional
        Leapfrog steps per trajectory; :math:`\lceil 1/\epsilon\rceil` by default, a
        trajectory of length about 1 in whitened coordinates.
    cov : array_like, shape (d, d), optional
        :math:`\Sigma`; ``None`` leaves it to :meth:`tune`, which takes the covariance of
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
        if not 0 < meeting_weight < 1:
            raise ValueError(f"meeting_weight must be in (0, 1), not {meeting_weight}")
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
        """Leapfrog steps per trajectory, or ``None`` before the step size is known."""
        return self._leapfrog.length

    @property
    def cov(self) -> FloatArray | None:
        """:math:`\\Sigma`."""
        return self._leapfrog.cov

    @property
    def _leapfrog(self) -> _Leapfrog:
        return cast(_Leapfrog, self.kernels[0])

    def _tuned(self, target: Target, pilot: PilotRun, rng: Seed) -> tuple[Kernel, TuningReport]:
        leapfrog, report = self._leapfrog._tuned(target, pilot, rng)
        tuned = cast(_Leapfrog, leapfrog)
        kernel = HMC(
            tuned.step_size,
            tuned.n_steps,
            cov=tuned.cov,
            meeting_weight=self.meeting_weight,
            meeting_scale=self.meeting_scale,
        )
        return kernel, report

    def __repr__(self) -> str:
        cov = "None" if self.cov is None else f"<{len(self.cov)}x{len(self.cov)}>"
        return (
            f"HMC(step_size={self.step_size}, n_steps={self.n_steps}, cov={cov}, "
            f"meeting_weight={self.meeting_weight}, meeting_scale={self.meeting_scale})"
        )


class _Leapfrog(Kernel):
    # The trajectory component of HMC. Coupled chains share the momentum and the uniform.
    def __init__(
        self, step_size: float | None, n_steps: int | None, cov: npt.ArrayLike | None
    ) -> None:
        if step_size is not None and not step_size > 0:
            raise ValueError(f"step_size must be positive, not {step_size}")
        if n_steps is not None and (not isinstance(n_steps, int | np.integer) or n_steps < 1):
            raise ValueError(f"n_steps must be a positive integer, not {n_steps!r}")
        self.step_size, self.n_steps = step_size, n_steps
        self.cov: FloatArray | None = None
        self._chol: FloatArray | None = None
        if cov is not None:
            self.cov, self._chol = covariance(cov, "cov")

    @property
    def needs_grad(self) -> bool:
        return True

    @property
    def is_tuned(self) -> bool:
        return self.step_size is not None and self.cov is not None

    @property
    def ndim(self) -> int | None:
        return None if self.cov is None else len(self.cov)

    @property
    def length(self) -> int | None:
        # Leapfrog steps per trajectory: n_steps, or ceil(1 / step_size) (spec section 5.5).
        if self.n_steps is not None or self.step_size is None:
            return self.n_steps
        return math.ceil(1 / self.step_size)

    def _tuned(self, target: Target, pilot: PilotRun, rng: Seed) -> tuple[Kernel, TuningReport]:
        cov = pilot.cov if self.cov is None else self.cov
        step_size = self.step_size
        if step_size is None:
            # Step sizes for a target acceptance of HMC scale as d^(-1/4) (Beskos et al. 2013).
            step_size = dual_averaging(
                lambda epsilon: _Leapfrog(epsilon, self.n_steps, cov),
                target,
                pilot.samples[:, -1],
                rng,
                TARGET_ACCEPTANCE,
                len(cov) ** -0.25,
            )
        return _Leapfrog(step_size, self.n_steps, cov), pilot.report()

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
        return moved.take(slice(None, n)), moved.take(slice(n, None))

    def _trajectory(
        self, target: Target, state: State, momentum: FloatArray
    ) -> tuple[State, FloatArray]:
        # The end of each trajectory and its log acceptance ratio, -inf for a trajectory
        # that stopped. Stopped rows keep moving in the arithmetic, never in the target.
        if self._chol is None or self.step_size is None:
            raise ValueError(
                "HMC needs step_size and cov; give them, or tune the kernel on a pilot run"
            )
        chol, epsilon, length = self._chol, self.step_size, cast(int, self.length)
        x, log_prob, grad = state.x, state.log_prob, cast(FloatArray, state.grad)
        start = log_prob - 0.5 * np.einsum("ij,ij->i", momentum, momentum)
        going = np.ones(len(x), dtype=bool)
        with np.errstate(over="ignore", invalid="ignore"):
            momentum = momentum + 0.5 * epsilon * matvec(chol.T, grad)
            for leap in range(length):
                x = x + epsilon * matvec(chol, momentum)
                log_prob, grad = np.full(len(x), -np.inf), np.zeros_like(x)
                rows = going.nonzero()[0]
                if len(rows):
                    reached = evaluate_proposal(target, x[rows])
                    log_prob[rows], grad[rows] = reached.log_prob, cast(FloatArray, reached.grad)
                going = np.isfinite(log_prob)
                kick = epsilon if leap < length - 1 else 0.5 * epsilon
                momentum = momentum + kick * matvec(chol.T, grad)
            end = log_prob - 0.5 * np.einsum("ij,ij->i", momentum, momentum)
        return State(x, log_prob, grad), np.where(going, end - start, -np.inf)

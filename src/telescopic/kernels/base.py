"""States, the kernel contract, and helpers shared by Metropolis-type kernels."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from telescopic._linalg import covariance
from telescopic._random import Draws
from telescopic._types import BoolArray, FloatArray, Seed, real
from telescopic.target import Target
from telescopic.tuning import PilotRun, TuningReport, adaptive_metropolis, dual_averaging


@dataclass
class State:
    """Positions, log-densities and (optionally) gradients of ``n`` chains, batch first.

    A kernel returns states with the fields of its input states, valid at the returned
    positions: a state that carries gradients gets them evaluated at every new position.

    Attributes
    ----------
    x : ndarray, shape (n, d)
    log_prob : ndarray, shape (n,)
    grad : ndarray, shape (n, d), or None
    """

    x: FloatArray
    log_prob: FloatArray
    grad: FloatArray | None = None

    def __len__(self) -> int:
        return len(self.log_prob)

    def take(self, rows: Any) -> State:
        """A copy of the chains in ``rows`` (an index, an index array, a mask or a slice)."""
        rows = np.arange(len(self))[rows]
        return State(
            self.x[rows], self.log_prob[rows], None if self.grad is None else self.grad[rows]
        )

    def put(self, rows: Any, other: State) -> None:
        """Overwrite the chains in ``rows`` with ``other``, in place."""
        self.x[rows] = other.x
        self.log_prob[rows] = other.log_prob
        if self.grad is not None:
            self.grad[rows] = other.grad


class Kernel(ABC):
    """A Markov kernel with its coupling (spec section 5.1, R-KER-1 to R-KER-5).

    Subclasses implement :meth:`step` and :meth:`coupled_step` on batches of chains.
    Each marginal of :meth:`coupled_step` must have exactly the law of :meth:`step`. The
    engine declares a meeting when a pair's returned positions are equal under ``==``, so a
    coupling that makes the two proposals identical must give the second chain a copy of
    the first chain's proposal, never a recomputation. A row's arithmetic must not depend
    on the other rows of the batch (spec R-KER-6).
    """

    @property
    def needs_grad(self) -> bool:
        """Whether the kernel evaluates the gradient of the log-density."""
        return False

    @property
    def needs_log_prob(self) -> bool:
        """Whether the kernel evaluates the log-density; Gibbs kernels do not."""
        return True

    @property
    def is_tuned(self) -> bool:
        """Whether every parameter is set, so the kernel can run."""
        return True

    @property
    def ndim(self) -> int | None:
        """The dimension that the kernel's parameters fix, or ``None``."""
        return None

    def check_target(self, target: Target) -> None:
        """Raise ``ValueError`` if ``target`` lacks a function that the kernel evaluates.

        Parameters
        ----------
        target : Target
        """
        if self.needs_log_prob and not target.has_log_prob:
            raise ValueError(f"{self!r} needs a log-density; log_prob=None is for Gibbs kernels")
        if self.needs_grad and not target.has_grad:
            raise ValueError(f"{self!r} needs the gradient of log_prob: pass grad=")

    def check_ndim(self, ndim: int) -> None:
        """Raise ``ValueError`` if the kernel works in another dimension than ``ndim``.

        Parameters
        ----------
        ndim : int
            The dimension of the initial points.
        """
        if self.ndim is not None and ndim != self.ndim:
            raise ValueError(
                f"the kernel works in {self.ndim} dimensions, but the initial points have {ndim}"
            )

    def tune(
        self,
        target: Target,
        initial: Any,
        rng: Seed = None,
        *,
        pilot: PilotRun | None = None,
    ) -> tuple[Kernel, TuningReport | None]:
        """A copy with the unset parameters tuned on pilot chains (spec R-TUNE-1).

        Each kernel's documentation says which parameters it tunes and how.

        Parameters
        ----------
        target : Target
            Checked with :meth:`check_target` before the pilot runs.
        initial
            An object with ``rvs(random_state=generator)``, a callable ``initial(rng)`` or
            an array of points of shape ``(n, d)``, as for
            :func:`telescopic.initial.as_initial`.
        rng : None, int, numpy.random.SeedSequence or numpy.random.Generator
            Seed of the tuning, in the tuning namespace (spec section 6.1): the pilot
            chains, unless ``pilot`` is given, and dual averaging.
        pilot : PilotRun, optional
            Adaptive Metropolis chains already run, from
            :func:`telescopic.tuning.adaptive_metropolis`, so that several kernels can be
            tuned on one pilot.

        Returns
        -------
        kernel : Kernel
        report : TuningReport or None
            ``None``, with the kernel itself, when there is nothing to tune.
        """
        if self.is_tuned:
            return self, None
        self.check_target(target)
        pilot = adaptive_metropolis(target, initial, rng) if pilot is None else pilot
        return self._tuned(target, initial, pilot, rng)

    def _tuned(
        self, target: Target, initial: Any, pilot: PilotRun, rng: Seed
    ) -> tuple[Kernel, TuningReport]:
        raise NotImplementedError(
            f"{type(self).__name__} has unset parameters but cannot tune them"
        )

    def init_state(self, target: Target, x: FloatArray) -> State:
        """Evaluate the target at the rows of ``x``, with gradients if the kernel needs them.

        Parameters
        ----------
        target : Target
        x : ndarray, shape (n, d)
            Initial points, whose dimension must be the kernel's.

        Returns
        -------
        State
        """
        self.check_ndim(x.shape[1])
        self.check_target(target)
        return evaluate(target, x, self.needs_grad)

    @abstractmethod
    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, FloatArray]:
        """One transition of each chain.

        Returns
        -------
        state : State
        acceptance : ndarray, shape (n,)
            Each chain's probability of accepting its proposal, 1 for a kernel that always
            moves; dual averaging adapts step sizes to its mean.
        """

    @abstractmethod
    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        """One coupled transition of each pair.

        Returns
        -------
        state_x, state_y : State
        """


class GradientKernel(Kernel):
    """A kernel with a step size and a covariance, driven by the gradient: MALA, HMC.

    :meth:`~telescopic.kernels.Kernel.tune` takes the covariance of the pilot chains unless ``cov`` was given, then
    the step size by dual averaging toward ``target_acceptance`` from the pilot's last
    states unless ``step_size`` was given (spec sections 5.4 and 5.5).

    Parameters
    ----------
    step_size : float, optional
    cov : array_like, shape (d, d), optional
    """

    target_acceptance: float

    def __init__(self, step_size: float | None, cov: npt.ArrayLike | None) -> None:
        self.step_size = None if step_size is None else _checked_step_size(step_size)
        self.cov: FloatArray | None = None
        self._chol: FloatArray | None = None
        if cov is not None:
            self.cov, self._chol = covariance("cov", cov)

    @property
    def needs_grad(self) -> bool:
        return True

    @property
    def is_tuned(self) -> bool:
        return self.step_size is not None and self.cov is not None

    @property
    def ndim(self) -> int | None:
        return None if self.cov is None else len(self.cov)

    def with_step_size(self, step_size: float) -> GradientKernel:
        """A copy with another step size, without validating the covariance again.

        Parameters
        ----------
        step_size : float
            Positive and finite.

        Returns
        -------
        GradientKernel
        """
        kernel = copy.copy(self)
        kernel.step_size = _checked_step_size(step_size)
        return kernel

    def _with_cov(self, cov: FloatArray) -> GradientKernel:
        raise NotImplementedError

    def _first_step_size(self, d: int) -> float:
        raise NotImplementedError

    def _tuned(
        self, target: Target, initial: Any, pilot: PilotRun, rng: Seed
    ) -> tuple[Kernel, TuningReport]:
        kernel = self if self.cov is not None else self._with_cov(pilot.cov)
        if kernel.step_size is None:
            first = self._first_step_size(len(pilot.cov))
            step_size = dual_averaging(
                kernel.with_step_size, target, pilot, rng, self.target_acceptance, first
            )
            kernel = kernel.with_step_size(step_size)
        return kernel, pilot.report()


def _checked_step_size(step_size: float) -> float:
    step_size = real("step_size", step_size)
    if not 0 < step_size < np.inf:
        raise ValueError(f"step_size must be positive and finite, not {step_size}")
    return step_size


def evaluate(target: Target, x: FloatArray, need_grad: bool) -> State:
    """The state at the rows of ``x``: log-densities, and gradients if needed.

    A non-finite gradient where the log-density is finite raises ``ValueError``: a cache
    must stay valid (spec R-INPUT-1). Gradient kernels reject such proposals instead,
    with :func:`evaluate_proposal`. Where the log-density is ``-inf`` the target gives a
    gradient of 0, so a gradient kernel proposes from there without drift, and accepts
    any proposal of finite density, as random-walk Metropolis does.

    Parameters
    ----------
    target : Target
    x : ndarray, shape (n, d)
    need_grad : bool

    Returns
    -------
    State
    """
    if not need_grad:
        return State(x, target.log_prob(x))
    log_prob, grad = target.log_prob_and_grad(x)
    bad = ~np.isfinite(grad).all(axis=1)
    if bad.any():
        row = int(np.argmax(bad))
        raise ValueError(
            f"grad returned {grad[row].tolist()} at x = {x[row].tolist()}, where log_prob "
            "is finite"
        )
    return State(x, log_prob, grad)


def evaluate_proposal(target: Target, x: FloatArray, live: BoolArray | None = None) -> State:
    """The states of a gradient kernel's proposals at the rows of ``x``.

    A proposal can be accepted only where its position, its log-density and its gradient
    are finite (spec R-INPUT-1). The others get a log-density of ``-inf``, which every
    Metropolis test rejects, and a zero gradient. Non-finite positions, and rows that
    ``live`` excludes, are not evaluated.

    Parameters
    ----------
    target : Target
    x : ndarray, shape (n, d)
    live : ndarray of bool, shape (n,), optional
        Rows to evaluate, such as the HMC trajectories that have not stopped.

    Returns
    -------
    State
    """
    usable = np.isfinite(x).all(axis=1)
    if live is not None:
        usable &= live
    if usable.all():
        log_prob, grad = target.log_prob_and_grad(x)
    else:
        log_prob, grad = np.full(len(x), -np.inf), np.zeros_like(x)
        rows = usable.nonzero()[0]
        if len(rows):
            log_prob[rows], grad[rows] = target.log_prob_and_grad(x[rows])
    bad = ~np.isfinite(grad).all(axis=1)
    if bad.any():
        # New arrays: those of a vectorised target may be its own.
        log_prob, grad = np.where(bad, -np.inf, log_prob), np.where(bad[:, None], 0.0, grad)
    return State(x, log_prob, grad)


def evaluate_pair(
    evaluate_rows: Callable[[FloatArray], State],
    x: FloatArray,
    y: FloatArray,
    identical: BoolArray,
) -> tuple[State, State]:
    """States at the coupled proposals ``x`` and ``y`` from one call of ``evaluate_rows``.

    Identical proposals are evaluated once and their state is shared (spec R-KER-3).

    Parameters
    ----------
    evaluate_rows : callable
        Maps points of shape ``(m, d)`` to their states, such as :func:`evaluate` with its
        target and ``need_grad`` bound.
    x, y : ndarray, shape (n, d)
    identical : ndarray of bool, shape (n,)
        Rows where ``y`` is a copy of ``x``.

    Returns
    -------
    state_x, state_y : State
    """
    distinct = (~identical).nonzero()[0]
    both = evaluate_rows(np.concatenate([x, y[distinct]]))
    source = np.arange(len(x))
    source[distinct] = np.arange(len(x), len(both))
    return both.take(slice(0, len(x))), both.take(source)


def acceptance(log_alpha: FloatArray) -> FloatArray:
    r"""Metropolis acceptance probabilities :math:`\min(1, e^{\log\alpha})`, 0 for NaN.

    Parameters
    ----------
    log_alpha : ndarray, shape (n,)
        Log acceptance ratios; NaN, from ``-inf - -inf``, is a rejection.

    Returns
    -------
    ndarray, shape (n,)
    """
    # fmax ignores NaN, so a NaN ratio gives exp(-inf) = 0.
    probability: FloatArray = np.exp(np.fmax(np.minimum(log_alpha, 0.0), -np.inf))
    return probability


def log_ratio(new: FloatArray, old: FloatArray) -> FloatArray:
    """The difference ``new - old`` of log-densities, without a warning for ``-inf - -inf``.

    That difference is NaN, which every Metropolis test ``log u < ratio`` rejects.

    Parameters
    ----------
    new, old : ndarray, shape (n,)

    Returns
    -------
    ndarray, shape (n,)
    """
    with np.errstate(invalid="ignore"):
        return new - old


def select(accepted: BoolArray, proposal: State, current: State) -> State:
    """Row by row, ``proposal`` where ``accepted`` and ``current`` elsewhere.

    Parameters
    ----------
    accepted : ndarray of bool, shape (n,)
    proposal, current : State

    Returns
    -------
    State
    """
    return State(
        np.where(accepted[:, None], proposal.x, current.x),
        np.where(accepted, proposal.log_prob, current.log_prob),
        None if current.grad is None else np.where(accepted[:, None], proposal.grad, current.grad),
    )

"""The arms of x91 (PREREGISTRATION.md, section 2) as kernels built on the package's.

P and MP are the package's `HMC` and `MALA`, unchanged. The others are subclasses that
change only what section 2 says they change:

- G: `_GuardLeapfrog` runs the package's tuning, then moves the tuned step by the guard
  of section 2 (`guard.guard`, with delta = arccos 0.9 and the band 1.3). Its kernel is
  the package's leapfrog: where the guard keeps the step, its arithmetic and its draws
  are P's.
- J2, J3, J5, CF, CF.2: `i137_common._JitterLeapfrog` at the width and acceptance target
  of the arm: each trajectory reads one uniform after the mixture's and before the
  momentum, shared by the two chains of a pair, and uses the step eps (1 + j (2u - 1))
  with the number of leapfrog steps of the nominal step. Dual averaging tunes the
  nominal step on the jittered kernel.
- P.2, CF.2: as P and CF with `meeting_weight=0.2`.
- MJ5, MCF: `JitterMALA`, MALA whose step reads one uniform before the package's draws and
  proposes with the factor eps_u C, eps_u = eps (1 + 0.5 (2u - 1)); a coupled step shares
  u and couples the two proposals N(m_x, eps_u^2 Sigma), N(m_y, eps_u^2 Sigma) by the
  package's reflection-maximal coupling in the coordinates of eps_u C.

Exactness: for a fixed step, the leapfrog Metropolis kernel and MALA leave pi invariant;
u is drawn independently of the state, so each jittered kernel is a mixture over u of
pi-invariant kernels. Each chain of a coupled step reads its numbers in the order of the
step alone (the uniform u, then the step's own draws), so its marginal transition is the
uncoupled kernel's; two equal chains get the same u, the same momentum or a coupled
proposal flagged identical, and the same arithmetic, so they stay equal.
`check_exactness.py` checks every new coupled step with `check_coupling` (section 3).
"""

from __future__ import annotations

import copy
import math
from functools import partial

import numpy as np
from guard import guard, n_of, resdist
from i137_common import _hmc_with, _JitterLeapfrog, pilot_frequencies

from telescopic._linalg import matvec
from telescopic.couplings import _reflection
from telescopic.kernels import HMC, MALA
from telescopic.kernels.base import State, acceptance, evaluate_pair, evaluate_proposal, select
from telescopic.kernels.hmc import _Leapfrog
from telescopic.kernels.mala import _drift, _log_alpha

DELTA = math.acos(0.9)  # the guard's bound on the distance to a resonance (section 2)
BAND = 0.3  # the guard's band: frequencies from w / 1.3 to 1.3 w
CF_ACCEPTANCE = 0.99  # #91


class _GuardLeapfrog(_Leapfrog):
    # The package's leapfrog, whose tuned step the guard of section 2 moves; `guard`
    # records eps_0, the action, and min_i r at the predicted frequencies before and after.
    def _with_cov(self, cov):
        return _GuardLeapfrog(self.step_size, self._n_steps, cov)

    def _tuned(self, target, initial, pilot, rng):
        kernel, report = super()._tuned(target, initial, pilot, rng)
        eps0 = kernel.step_size
        w = pilot_frequencies(kernel._chol, pilot.cov)
        eps, action = guard(eps0, self._n_steps, w, DELTA, BAND)
        if action == "moved":
            kernel = kernel.with_step_size(eps)
        kernel.guard = {
            "eps0": eps0,
            "action": action,
            "r_before": float(resdist(eps0, n_of(eps0, self._n_steps), w).min()),
            "r_after": float(resdist(eps, n_of(eps, self._n_steps), w).min()),
            "w_pred": w.tolist(),
        }
        return kernel, report


def _jitter_class(width, target_acceptance):
    return type(
        f"_Jitter{width}Leapfrog",
        (_JitterLeapfrog,),
        {"jitter": width, "target_acceptance": target_acceptance},
    )


GuardHMC = _hmc_with(_GuardLeapfrog)
_HMC_ARMS = {
    "P": (HMC, 0.05),
    "G": (GuardHMC, 0.05),
    "J2": (_hmc_with(_jitter_class(0.2, 0.8)), 0.05),
    "J3": (_hmc_with(_jitter_class(0.3, 0.8)), 0.05),
    "J5": (_hmc_with(_jitter_class(0.5, 0.8)), 0.05),
    "CF": (_hmc_with(_jitter_class(0.5, CF_ACCEPTANCE)), 0.05),
    "P.2": (HMC, 0.2),
    "CF.2": (_hmc_with(_jitter_class(0.5, CF_ACCEPTANCE)), 0.2),
}


class JitterMALA(MALA):
    """MALA whose step is eps (1 + j (2u - 1)), one uniform u per step, shared by a pair.

    With :math:`s = \\epsilon(1 + j(2u - 1))` and :math:`B = sC`, the proposal is
    :math:`x' = x + B(h/2 + z)`, :math:`h = B^\\top\\nabla\\log\\pi(x)`, accepted on
    MALA's ratio at the step :math:`s`. Since :math:`B^\\top g = C^\\top(s g)`, the
    package's whitened functions serve with the factor :math:`C` and gradients scaled by
    :math:`s`.

    Parameters
    ----------
    jitter : float
        :math:`j`, in [0, 1).
    target_acceptance : float
        The acceptance rate of dual averaging.
    step_size : float, optional
        The nominal step :math:`\\epsilon`; ``None`` leaves it to tuning.
    cov : array_like, shape (d, d), optional
        :math:`\\Sigma`; ``None`` leaves it to tuning.
    """

    def __init__(self, jitter, target_acceptance, step_size=None, *, cov=None):
        super().__init__(step_size, cov=cov)
        self.jitter, self.target_acceptance = jitter, target_acceptance

    def _with_cov(self, cov):
        kernel = copy.copy(self)
        MALA.__init__(kernel, self.step_size, cov=cov)
        return kernel

    def _scale(self, draws):
        return (self.step_size * (1 + self.jitter * (2 * draws.uniform() - 1)))[:, None]

    def step(self, target, state, draws):
        s, chol, blas = self._scale(draws), self._chol, target.vectorize
        current = _scaled(state, s)
        drift = _drift(current, chol, blas)
        z = draws.normal(len(chol))
        proposal = evaluate_proposal(
            target, state.x + s * matvec(chol, 0.5 * drift + z, blas=blas)
        )
        log_alpha = _log_alpha(current, _scaled(proposal, s), z, drift, chol, blas)
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, proposal, state), acceptance(log_alpha)

    def coupled_step(self, target, state_x, state_y, draws):
        # MALA.coupled_step with the factor s C of the pair's own step.
        s, chol, blas = self._scale(draws), self._chol, target.vectorize
        current_x, current_y = _scaled(state_x, s), _scaled(state_y, s)
        drift_x, drift_y = _drift(current_x, chol, blas), _drift(current_y, chol, blas)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = matvec(self._inverse, state_x.x - state_y.x, blas=blas) / s + 0.5 * (
                drift_x - drift_y
            )
        z, identical, shift = _reflection(delta, draws, delta)
        residual_y = np.where(identical[:, None], z + delta, z - shift)
        x = state_x.x + s * matvec(chol, 0.5 * drift_x + z, blas=blas)
        y = np.where(
            identical[:, None],
            x,
            state_y.x + s * matvec(chol, 0.5 * drift_y + residual_y, blas=blas),
        )
        proposal_x, proposal_y = evaluate_pair(partial(evaluate_proposal, target), x, y, identical)
        log_u = draws.log_uniform()
        alpha_x = _log_alpha(current_x, _scaled(proposal_x, s), z, drift_x, chol, blas)
        alpha_y = _log_alpha(current_y, _scaled(proposal_y, s), residual_y, drift_y, chol, blas)
        return select(log_u < alpha_x, proposal_x, state_x), select(
            log_u < alpha_y, proposal_y, state_y
        )


def _scaled(state, s):
    # The state with its gradient times the row's step, so that C^T (s g) = (s C)^T g.
    with np.errstate(over="ignore", invalid="ignore"):
        return State(state.x, state.log_prob, s * state.grad)


_MALA_ARMS = {
    "MP": MALA,
    "MJ5": lambda: JitterMALA(0.5, MALA.target_acceptance),
    "MCF": lambda: JitterMALA(0.5, CF_ACCEPTANCE),
}
HMC_ARMS = tuple(_HMC_ARMS)
MALA_ARMS = tuple(_MALA_ARMS)


def kernel(arm, n_steps=None, cov=None):
    """The untuned kernel of an arm, with a given `n_steps` and `cov` for the HMC arms."""
    if arm in _MALA_ARMS:
        return _MALA_ARMS[arm]()
    cls, weight = _HMC_ARMS[arm]
    return cls(n_steps=n_steps, cov=cov, meeting_weight=weight)


def settings(tuned):
    """The nominal step, n_steps, jitter width, target acceptance and meeting weight."""
    if isinstance(tuned, HMC):
        leapfrog = tuned.kernels[0]
        return {
            "eps": leapfrog.step_size,
            "n": leapfrog.n_steps,
            "jitter": getattr(leapfrog, "jitter", 0.0),
            "acceptance": leapfrog.target_acceptance,
            "meeting_weight": tuned.meeting_weight,
            "guard": getattr(leapfrog, "guard", None),
        }
    return {
        "eps": tuned.step_size,
        "jitter": getattr(tuned, "jitter", 0.0),
        "acceptance": tuned.target_acceptance,
    }

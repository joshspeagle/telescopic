"""The target distribution: a log-density, an optional gradient, and a batched interface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from telescopic._types import FloatArray, read_only


class Target:
    r"""A log-density :math:`\log\pi` known up to a constant, with an optional gradient.

    Kernels call :meth:`log_prob` and :meth:`log_prob_and_grad` on batches of points of
    shape ``(n, d)``, whether or not the user's functions are vectorised.

    Parameters
    ----------
    log_prob : callable or None
        ``log_prob(x, *args, **kwargs)`` returns a float for ``x`` of shape ``(d,)``, or
        an array of shape ``(n,)`` for ``x`` of shape ``(n, d)`` when ``vectorize`` is
        true. ``-inf`` marks zero density. It must be deterministic: the same point
        must always give the same value. ``None`` is for Gibbs kernels, which need no
        log-density: every point then gets 0 (spec section 5.7).
    grad : callable, optional
        :math:`\nabla\log\pi` with the same signature, returning shape ``(d,)`` or
        ``(n, d)``.
    args, kwargs
        Extra arguments passed to ``log_prob`` and ``grad``.
    vectorize : bool
        Whether ``log_prob`` and ``grad`` accept a batch of points.

    Notes
    -----
    The functions receive read-only arrays: a function that modifies its input raises
    instead of corrupting the chains. The counters ``n_log_prob`` and ``n_grad`` count
    evaluations point by point, and ``n_grad_zeroed`` the gradients taken as 0 because
    they were not finite where the log-density is.
    """

    def __init__(
        self,
        log_prob: Callable[..., Any] | None,
        grad: Callable[..., Any] | None = None,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        vectorize: bool = False,
    ) -> None:
        if log_prob is not None and not callable(log_prob):
            raise TypeError(f"log_prob must be callable or None, not {type(log_prob).__name__}")
        if log_prob is None and grad is not None:
            raise ValueError("grad needs log_prob; log_prob=None is for Gibbs kernels")
        if grad is not None and not callable(grad):
            raise TypeError(f"grad must be callable or None, not {type(grad).__name__}")
        if not isinstance(vectorize, bool | np.bool_):
            raise TypeError(f"vectorize must be True or False, not {vectorize!r}")
        self._log_prob = log_prob
        self._grad = grad
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.vectorize = bool(vectorize)
        self.n_log_prob = 0
        self.n_grad = 0
        self.n_grad_zeroed = 0

    @property
    def has_grad(self) -> bool:
        """Whether a gradient was given."""
        return self._grad is not None

    @property
    def has_log_prob(self) -> bool:
        """Whether a log-density was given."""
        return self._log_prob is not None

    def log_prob(self, x: FloatArray) -> FloatArray:
        """Log-densities at the rows of ``x``, shape ``(n,)``; raises on NaN or ``+inf``."""
        if self._log_prob is None:
            return np.zeros(len(x))
        values = self._call(self._log_prob, x, ())
        self.n_log_prob += len(x)
        # One reduction finds NaN or +inf, which make the maximum fail the test.
        if not np.max(values, initial=-np.inf) < np.inf:
            row = int(np.argmax(np.isnan(values) | (values == np.inf)))
            raise ValueError(f"log_prob returned {values[row]} at x = {x[row].tolist()}")
        return values

    def log_prob_and_grad(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Log-densities, shape ``(n,)``, and gradients, shape ``(n, d)``, at the rows of ``x``.

        The gradient is evaluated only where the log-density is finite, and is 0 where it
        is ``-inf``. Where it is finite, a gradient that is not finite is taken as 0 too,
        and counted in ``n_grad_zeroed``: MALA and HMC then move without drift there and
        stay exact, since a Metropolis-Hastings proposal and a leapfrog map stay valid for
        any drift that depends on the position alone (spec R-INPUT-1).
        """
        if self._grad is None:
            raise ValueError("this target has no gradient; pass grad= to use this kernel")
        values = self.log_prob(x)
        if np.min(values, initial=np.inf) > -np.inf:  # every point inside the support
            gradients = self._call(self._grad, x, (x.shape[1],))
            self.n_grad += len(x)
        else:
            inside = values > -np.inf
            gradients = np.zeros(x.shape)
            if inside.any():
                gradients[inside] = self._call(self._grad, x[inside], (x.shape[1],))
            self.n_grad += int(inside.sum())
        if not np.isfinite(gradients).all():
            bad = ~np.isfinite(gradients).all(axis=1)
            # A new array: a vectorised grad may return its own.
            gradients = np.where(bad[:, None], 0.0, gradients)
            self.n_grad_zeroed += int(bad.sum())
        return values, gradients

    def _call(
        self, function: Callable[..., Any], x: FloatArray, event: tuple[int, ...]
    ) -> FloatArray:
        view = read_only(x)
        if self.vectorize:
            values = function(view, *self.args, **self.kwargs)
            returned_none = values is None
        else:
            values = [function(row, *self.args, **self.kwargs) for row in view]
            returned_none = False
        out = np.asarray(values, dtype=np.float64)
        # NumPy reads None as NaN, and the error would then blame the value.
        if not self.vectorize and np.isnan(out).any():
            returned_none = any(value is None for value in values)
        if returned_none or out.shape != (len(x), *event):
            name = getattr(function, "__name__", "the target function")
            per_point = "a float" if not event else f"shape {event}"
            if returned_none:
                raise ValueError(
                    f"{name} returned None; it must return {per_point} for each point"
                )
            raise ValueError(
                f"{name} returned shape {out.shape} for {len(x)} points; expected "
                f"{(len(x), *event)}, that is {per_point} for each point"
            )
        return out

"""The target distribution: a log-density, an optional gradient, and a batched interface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from telescopic._types import FloatArray


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
    evaluations point by point.
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
        self._log_prob = log_prob
        self._grad = grad
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.vectorize = vectorize
        self.n_log_prob = 0
        self.n_grad = 0

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
        is ``-inf``. Gradients are returned as computed: kernels decide what a non-finite
        one means (spec R-INPUT-1).
        """
        if self._grad is None:
            raise ValueError("this target has no gradient; pass grad= to use this kernel")
        values = self.log_prob(x)
        inside = (values > -np.inf).nonzero()[0]
        if len(inside) == len(x):
            gradients = self._call(self._grad, x, (x.shape[1],))
        else:
            gradients = np.zeros(x.shape)
            if len(inside):
                gradients[inside] = self._call(self._grad, x[inside], (x.shape[1],))
        self.n_grad += len(inside)
        return values, gradients

    def _call(
        self, function: Callable[..., Any], x: FloatArray, event: tuple[int, ...]
    ) -> FloatArray:
        view = x.view()
        view.flags.writeable = False
        if self.vectorize:
            values = function(view, *self.args, **self.kwargs)
            returned_none = values is None
        else:
            values = [function(row, *self.args, **self.kwargs) for row in view]
            returned_none = any(value is None for value in values)
        name = getattr(function, "__name__", "the target function")
        per_point = "a float" if not event else f"shape {event}"
        # NumPy would read None as NaN, and the error would then blame the value.
        if returned_none:
            raise ValueError(f"{name} returned None; it must return {per_point} for each point")
        out = np.asarray(values, dtype=np.float64)
        if out.shape != (len(x), *event):
            raise ValueError(
                f"{name} returned shape {out.shape} for {len(x)} points; expected "
                f"{(len(x), *event)}, that is {per_point} for each point"
            )
        return out

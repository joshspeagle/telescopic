"""The target distribution: a log-density, an optional gradient, and a batched interface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from telescopic._random import FloatArray


class Target:
    r"""A log-density :math:`\log\pi` known up to a constant, with an optional gradient.

    Kernels call :meth:`log_prob` and :meth:`log_prob_and_grad` on batches of points of
    shape ``(n, d)``, whether or not the user's functions are vectorised.

    Parameters
    ----------
    log_prob : callable
        ``log_prob(x, *args, **kwargs)`` returns a float for ``x`` of shape ``(d,)``, or
        an array of shape ``(n,)`` for ``x`` of shape ``(n, d)`` when ``vectorize`` is
        true. ``-inf`` marks zero density. It must be deterministic: the same point
        must always give the same value.
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
        log_prob: Callable[..., Any],
        grad: Callable[..., Any] | None = None,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        vectorize: bool = False,
    ) -> None:
        if not callable(log_prob):
            raise TypeError(f"log_prob must be callable, not {type(log_prob).__name__}")
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

    def log_prob(self, x: FloatArray) -> FloatArray:
        """Log-densities at the rows of ``x``, shape ``(n,)``; raises on NaN or ``+inf``."""
        values = self._call(self._log_prob, x, ())
        self.n_log_prob += len(x)
        bad = np.isnan(values) | (values == np.inf)
        if bad.any():
            row = int(np.argmax(bad))
            raise ValueError(f"log_prob returned {values[row]} at x = {x[row].tolist()}")
        return values

    def log_prob_and_grad(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Log-densities, shape ``(n,)``, and gradients, shape ``(n, d)``, at the rows of ``x``.

        Rows with a finite log-density must have a finite gradient.
        """
        if self._grad is None:
            raise ValueError("this target has no gradient; pass grad= to use this kernel")
        values = self.log_prob(x)
        gradients = self._call(self._grad, x, (x.shape[1],))
        self.n_grad += len(x)
        bad = np.isfinite(values) & ~np.isfinite(gradients).all(axis=1)
        if bad.any():
            row = int(np.argmax(bad))
            raise ValueError(f"grad returned {gradients[row].tolist()} at x = {x[row].tolist()}")
        return values, gradients

    def _call(
        self, function: Callable[..., Any], x: FloatArray, event: tuple[int, ...]
    ) -> FloatArray:
        view = x.view()
        view.flags.writeable = False
        if self.vectorize:
            out = np.asarray(function(view, *self.args, **self.kwargs), dtype=np.float64)
        else:
            out = np.asarray(
                [function(row, *self.args, **self.kwargs) for row in view], dtype=np.float64
            ).reshape((len(x), *event))
        if out.shape != (len(x), *event):
            raise ValueError(
                f"{getattr(function, '__name__', 'the target function')} returned shape "
                f"{out.shape} for {len(x)} points; expected {(len(x), *event)}"
            )
        return out

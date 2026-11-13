"""The target distribution: a log-density, an optional gradient, and a batched interface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from telescopic._types import FloatArray, flag, log_densities, read_only, real_objects, shown
from telescopic.warnings import InputWarning, warn


# Spec section 5.7 (log_prob None) and R-INPUT-1.
class Target:
    r"""A log-density :math:`\log\pi` known up to a constant, with an optional gradient.

    Kernels call :meth:`log_prob` and :meth:`log_prob_and_grad` on batches of points of
    shape ``(n, d)``, whether or not the user's functions are vectorized.

    Parameters
    ----------
    log_prob : callable or None
        ``log_prob(x, *args, **kwargs)`` returns a float for ``x`` of shape ``(d,)``, or
        an array of shape ``(n,)`` for ``x`` of shape ``(n, d)`` when ``vectorize`` is
        true. ``-inf`` marks zero density. It must be deterministic: the same point
        must always give the same value. ``None`` is for Gibbs kernels, which need no
        log-density: every point then gets 0.
    grad : callable, optional
        :math:`\nabla\log\pi` with the same signature, returning shape ``(d,)`` or
        ``(n, d)``.
    args : tuple or list
        Extra positional arguments passed to ``log_prob`` and ``grad``, such as
        ``(data,)``.
    kwargs : mapping, optional
        Extra keyword arguments passed to them.
    vectorize : bool
        Whether ``log_prob`` and ``grad`` accept a batch of points.

    Notes
    -----
    The functions receive read-only arrays: a function that modifies its input raises
    instead of corrupting the chains. Their outputs are copied, so a function may reuse
    its output array, and an output whose shape differs from the expected one only by
    axes of length 1, as SciPy's ``logpdf`` of one point, is reshaped. The counters
    ``n_log_prob`` and ``n_grad`` count evaluations point by point, and
    ``n_grad_zeroed`` the gradients taken as 0 because they were not finite where the
    log-density is.
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
        # A tuple would unpack an array's rows, or a string's letters, as extra arguments.
        if not isinstance(args, tuple | list):
            raise TypeError(
                "args must be a tuple of extra arguments, such as (data,), not "
                f"{type(args).__name__}"
            )
        self._log_prob = log_prob
        self._grad = grad
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.vectorize = flag("vectorize", vectorize)
        self.n_log_prob = 0
        self.n_grad = 0
        self.n_grad_zeroed = 0
        self._warned_precision = False

    @property
    def has_grad(self) -> bool:
        """Whether a gradient was given.

        Returns
        -------
        bool
        """
        return self._grad is not None

    @property
    def has_log_prob(self) -> bool:
        """Whether a log-density was given.

        Returns
        -------
        bool
        """
        return self._log_prob is not None

    def log_prob(self, x: FloatArray) -> FloatArray:
        r"""Log-densities at the rows of ``x``; raises ``ValueError`` on NaN or ``+inf``.

        Parameters
        ----------
        x : ndarray, shape (n, d)

        Returns
        -------
        ndarray, shape (n,)
            :math:`\log\pi` at each row, ``-inf`` at zero density; 0 for a target
            without a log-density.
        """
        if self._log_prob is None:
            return np.zeros(len(x))
        values = self._call(self._log_prob, x, ())
        self.n_log_prob += len(x)
        return log_densities("log_prob", values, x)

    def log_prob_and_grad(
        self, x: FloatArray, *, start: bool = False
    ) -> tuple[FloatArray, FloatArray]:
        """Log-densities and gradients at the rows of ``x``.

        The gradient is evaluated only where the log-density is finite, and is 0 where it
        is ``-inf``. Where it is finite, a gradient that is not finite is taken as 0 too,
        and counted in ``n_grad_zeroed``; at the states that chains ``start`` from, it
        raises ``ValueError`` with the point.

        Parameters
        ----------
        x : ndarray, shape (n, d)
        start : bool
            Whether chains start at the rows of ``x``.

        Returns
        -------
        log_prob : ndarray, shape (n,)
        grad : ndarray, shape (n, d)
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
        finite = np.isfinite(gradients)
        if not finite.all():
            bad = ~finite.all(axis=1)
            if start:
                # A MALA chain that starts there cannot leave: its moves back need the
                # reverse proposal, which carries the full drift, to return (DECISIONS #67).
                row = int(np.argmax(bad))
                raise ValueError(
                    f"grad returned {gradients[row].tolist()} at x = {x[row].tolist()}, "
                    "where log_prob is finite: chains cannot start there; compute grad in "
                    "log space, as with scipy.special.logsumexp, or start them elsewhere"
                )
            gradients[bad] = 0.0
            self.n_grad_zeroed += int(bad.sum())
        return values, gradients

    def _call(
        self, function: Callable[..., Any], x: FloatArray, event: tuple[int, ...]
    ) -> FloatArray:
        view = read_only(x)
        if self.vectorize:
            values = function(view, *self.args, **self.kwargs)
        else:
            values = [function(row, *self.args, **self.kwargs) for row in view]
        expected = (len(x), *event)
        try:
            # A copy, so that a function that reuses its own array cannot alias states.
            out = np.array(values)
        except ValueError:  # rows of different shapes
            out = None
        # Axes of length 1 change neither the values nor their order, so a shape that
        # differs only by them is reshaped, as SciPy's logpdf of one point, shape ().
        if out is None or not _fits(out, expected):
            out = self._converted(function, values, x, event)
        out = out.reshape(expected)
        return out if out.dtype == np.float64 else self._float64(function, out, x)

    def _float64(
        self, function: Callable[..., Any], out: npt.NDArray[Any], x: FloatArray
    ) -> FloatArray:
        # A complex output with an imaginary part of exactly 0, as complex-step code gives,
        # is its real part (DECISIONS #109). Floats narrower than float64, as JAX
        # returns without 64-bit mode, round the target, so they warn once (#110).
        name, returned = _name(function), out.dtype
        if out.dtype.kind == "c":
            rows = np.nonzero(out.imag)[0]
            if len(rows):
                raise ValueError(
                    f"{name} returned {out[rows[0]].tolist()} at x = {x[rows[0]].tolist()}: "
                    "it must return real numbers, not complex ones with a non-zero "
                    "imaginary part"
                )
            out = out.real
        if out.dtype.kind == "f" and out.dtype.itemsize < 8 and not self._warned_precision:
            self._warned_precision = True
            warn(
                f"{name} returned {returned} values: the chains target the density "
                f"rounded to {out.dtype} precision; with JAX, enable 64-bit mode with "
                "jax.config.update('jax_enable_x64', True)",
                InputWarning,
            )
        return out.astype(np.float64)

    def _converted(
        self, function: Callable[..., Any], values: Any, x: FloatArray, event: tuple[int, ...]
    ) -> npt.NDArray[Any]:
        # The outputs that NumPy could not stack as numbers of the expected shape, point by
        # point: each must be real numbers of shape event up to axes of length 1, as from a
        # function that returns a float on one branch and SciPy's logpdf of shape (1,) on
        # another. Otherwise the error names the first point where it went wrong.
        if self.vectorize:
            out = _array(values)
            if out is None or not _fits(out, (len(x), *event)):
                raise self._invalid(function, values, None, x, event)
            return out
        arrays = []
        for row, value in enumerate(values):
            array = _array(value)
            if array is None or not _fits(array, event):
                raise self._invalid(function, value, row, x, event)
            arrays.append(array.reshape(event))
        return np.array(arrays)

    def _invalid(
        self,
        function: Callable[..., Any],
        value: Any,
        row: int | None,
        x: FloatArray,
        event: tuple[int, ...],
    ) -> ValueError:
        array = _array(value)
        if value is None:
            text = "None"
        elif array is None:
            text = "a ragged sequence"
        elif array.dtype.kind not in "biufc":
            text = shown(value)
        else:
            text = f"shape {array.shape}"
        where = f"for {len(x)} points" if row is None else f"at x = {x[row].tolist()}"
        per_point = "a float" if not event else f"shape {event}"
        message = (
            f"{_name(function)} returned {text} {where}; it must return {per_point} for each point"
        )
        if isinstance(value, tuple) and not event:  # emcee's (log-density, blobs)
            message += ", and derived quantities are computed after the run with Result.estimate"
        return ValueError(message)


def _name(function: Callable[..., Any]) -> str:
    return getattr(function, "__name__", "the target function")


def _array(value: Any) -> npt.NDArray[Any] | None:
    # value as an array, with real numbers held as objects converted, or None if ragged.
    try:
        return real_objects(np.asarray(value))
    except ValueError:
        return None


def _fits(array: npt.NDArray[Any], shape: tuple[int, ...]) -> bool:
    # Whether array holds numbers of shape, up to axes of length 1.
    return array.dtype.kind in "biufc" and _without_ones(array.shape) == _without_ones(shape)


def _without_ones(shape: tuple[int, ...]) -> list[int]:
    return [size for size in shape if size != 1]

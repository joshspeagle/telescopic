"""Shared code of the i137 experiments (issue #137): the remedies as kernels, and a runner.

The remedies are small subclasses of the package's HMC and of its private leapfrog
component; none of them copies package code.

- `GuardedHMC` (remedy a): after the package's tuning, the predicted contraction factor
  |cos(n theta(eps w))|, theta(h) = 2 arcsin(h / 2), of every whitened mode w and of
  every frequency within a factor 1 + BAND of it, is computed, and the step moves to the
  nearest step (in log) at which the worst factor is at most BOUND. The modes come from
  the pilot: w^2 are the eigenvalues of C^T S^-1 C, with C C^T the kernel's covariance
  and S the pilot covariance, so w = 1 when the covariance is the pilot's. Search: the
  grid eps_DA exp(s), s in [-2, 0.1] by 0.001; a step with eps w >= 2 counts as factor 1
  (unstable, rejected). If no step on the grid passes, the one with the smallest worst
  factor (nearest on ties).
- `ContractionHMC` (remedy b): after the package's tuning, 16 steps eps_DA exp(s), s
  evenly from -1.5 to 0.3, are scored on 8 pairs of pilot states (each chain's last state
  against its states at the middle and the quarter of its kept half), by 10 coupled
  leapfrog trajectories with common momenta and uniforms across the grid; the step with
  the most negative mean log contraction of the distance per trajectory is kept. Every
  evaluation is counted by the target.
- `JitterHMC` (remedy c): every trajectory uses the step eps (1 + j (2u - 1)) with one
  uniform u drawn before the momentum, j = 0.5 (#91); in a coupled step both chains use
  the same u. Dual averaging tunes the nominal eps on the jittered kernel toward 0.8.
- `PhaseJitterHMC` (remedy d): as c, then j = min(0.5, pi / (2 n theta(eps w_min))), the
  smallest width whose phases n theta sweep at least one period of |cos| on every mode of
  frequency at least w_min, the slowest mode predicted from the pilot.

Exactness of c and d: for a fixed step, the leapfrog Metropolis kernel leaves pi
invariant (volume-preserving, reversible map with a Metropolis test on the joint
density); u is drawn independently of the state, so the kernel is a mixture over u of
pi-invariant kernels. Each chain of a coupled step reads one uniform for u, d normals and
one uniform for the test, in that order, as the step alone does, so each chain's
marginal transition is exactly the uncoupled one; equal chains get identical arithmetic
and stay equal. a and b change only the tuned step, not the kernel.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np

import telescopic
from telescopic._random import TUNING, replicate_generator, root_seed
from telescopic.examples import Example, _gaussian_grad, _gaussian_log_prob
from telescopic.kernels.base import evaluate, select
from telescopic.kernels.hmc import HMC, _Leapfrog
from telescopic.sampler import CoupledSampler
from telescopic.tuning import choose_k

BAND = 0.3  # remedy a: relative error of the predicted frequencies (set in i137_predictions.md)
BOUND = 0.9  # remedy a: largest acceptable predicted contraction factor
JITTER = 0.5  # remedy c, and the largest width of remedy d (#91)


def theta(h):
    # The rotation angle of one leapfrog step of h = eps w; nan where unstable (h >= 2).
    h = np.asarray(h, dtype=float)
    return np.where(h < 2, 2 * np.arcsin(np.minimum(h, 2) / 2), np.nan)


def factor(eps, n, w):
    # |cos(n theta)| = |T_n(1 - h^2 / 2)|, 1 where the leapfrog is unstable.
    t = theta(np.multiply.outer(np.atleast_1d(eps), np.atleast_1d(w)))
    return np.where(np.isnan(t), 1.0, np.abs(np.cos(n * np.nan_to_num(t))))


def jitter_factor(eps, n, w, j, points=401):
    # exp E_u log|cos(n theta(eps (1 + j (2u - 1)) w))| per mode: the geometric-mean factor.
    if j == 0:
        return factor(eps, n, w)[0]
    u = 1 + j * np.linspace(-1, 1, points)
    f = factor(eps * u, n, w)
    return np.exp(np.log(np.maximum(f, 1e-300)).mean(axis=0))


def n_at(given, eps):
    return given if given is not None else math.ceil(1 / max(eps, 1e-3))


def pilot_frequencies(chol, pilot_cov):
    # Whitened frequencies predicted from the pilot: sqrt eig(C^T S^-1 C).
    return np.sqrt(np.linalg.eigvalsh(chol.T @ np.linalg.solve(pilot_cov, chol)))


# Remedy a.
class _GuardedLeapfrog(_Leapfrog):
    def _with_cov(self, cov):
        return _GuardedLeapfrog(self.step_size, self._n_steps, cov)

    def _tuned(self, target, initial, pilot, rng):
        kernel, report = super()._tuned(target, initial, pilot, rng)
        if self.step_size is not None:
            return kernel, report
        eps0 = kernel.step_size
        w = pilot_frequencies(kernel._chol, pilot.cov)
        band = np.concatenate([w * f for f in np.geomspace(1 / (1 + BAND), 1 + BAND, 21)])
        s = np.round(np.arange(-2000, 101) / 1000, 3)
        steps = eps0 * np.exp(s)
        worst = np.array([factor(e, n_at(self._n_steps, e), band).max() for e in steps])
        ok = worst <= BOUND
        if ok.any():
            choice = np.flatnonzero(ok)[np.argmin(np.abs(s[ok]))]
        else:
            best = np.flatnonzero(worst == worst.min())
            choice = best[np.argmin(np.abs(s[best]))]
        kernel = kernel.with_step_size(float(steps[choice]))
        kernel.guard = {
            "eps_da": eps0,
            "worst_da": float(worst[s == 0][0]),
            "worst": float(worst[choice]),
            "passed": bool(ok.any()),
        }
        return kernel, report


# Remedy b.
class _ContractionLeapfrog(_Leapfrog):
    GRID, PAIRS_PER_CHAIN, TRAJECTORIES = np.linspace(-1.5, 0.3, 16), 2, 10

    def _with_cov(self, cov):
        return _ContractionLeapfrog(self.step_size, self._n_steps, cov)

    def _tuned(self, target, initial, pilot, rng):
        kernel, report = super()._tuned(target, initial, pilot, rng)
        if self.step_size is not None:
            return kernel, report
        eps0 = kernel.step_size
        kept = pilot.samples.shape[1]
        xs = np.concatenate([pilot.samples[:, -1]] * self.PAIRS_PER_CHAIN)
        ys = np.concatenate([pilot.samples[:, kept // 2], pilot.samples[:, kept // 4]])
        p, g, d = len(xs), len(self.GRID), xs.shape[1]
        steps = eps0 * np.exp(self.GRID)
        # Rows: grid value major, pair minor; the first chains, then the second.
        x = np.concatenate([np.tile(xs, (g, 1)), np.tile(ys, (g, 1))])
        state = evaluate(target, x, True)
        gap0 = np.linalg.norm(x[: g * p] - x[g * p :], axis=1)
        generator = replicate_generator(root_seed(rng), TUNING, 100)
        rows_eps = np.tile(np.repeat(steps, p), 2)[:, None]
        n_rows = np.tile(np.repeat([n_at(self._n_steps, e) for e in steps], p), 2)
        for _ in range(self.TRAJECTORIES):
            momentum = generator.standard_normal((p, d))
            log_u = np.log(generator.random(p))
            new = state.take(slice(None))
            for n in np.unique(n_rows):
                rows = np.flatnonzero(n_rows == n)
                k = copy.copy(kernel)
                k.step_size, k._n_steps = rows_eps[rows], int(n)
                end, log_alpha = k._trajectory(
                    target, state.take(rows), np.tile(momentum, (2 * g, 1))[rows]
                )
                accept = np.tile(log_u, 2 * g)[rows] < log_alpha
                new.put(rows, select(accept, end, state.take(rows)))
            state = new
        gap = np.linalg.norm(state.x[: g * p] - state.x[g * p :], axis=1)
        rate = (np.log(np.maximum(gap, 1e-300)) - np.log(gap0)).reshape(g, p).mean(axis=1)
        rate /= self.TRAJECTORIES
        choice = int(np.argmin(rate))
        kernel = kernel.with_step_size(float(steps[choice]))
        kernel.contraction = {"eps_da": eps0, "steps": steps.tolist(), "rate": rate.tolist()}
        return kernel, report


# Remedy c.
class _JitterLeapfrog(_Leapfrog):
    jitter = JITTER

    def _with_cov(self, cov):
        kernel = copy.copy(self)
        _Leapfrog.__init__(kernel, self.step_size, self._n_steps, cov)
        return kernel

    def _at(self, u, copies):
        kernel = copy.copy(self)
        kernel._n_steps = self.n_steps
        steps = self.step_size * (1 + self.jitter * (2 * u - 1))
        kernel.step_size = np.tile(steps, copies)[:, None]
        return kernel

    def step(self, target, state, draws):
        return _Leapfrog.step(self._at(draws.uniform(), 1), target, state, draws)

    def coupled_step(self, target, state_x, state_y, draws):
        return _Leapfrog.coupled_step(
            self._at(draws.uniform(), 2), target, state_x, state_y, draws
        )


# Remedy d.
class _PhaseJitterLeapfrog(_JitterLeapfrog):
    def _tuned(self, target, initial, pilot, rng):
        kernel, report = super()._tuned(target, initial, pilot, rng)
        if self.step_size is None:
            w_min = pilot_frequencies(kernel._chol, pilot.cov).min()
            t = float(theta(min(kernel.step_size * w_min, 1.999)))
            kernel.jitter = min(JITTER, math.pi / (2 * kernel.n_steps * t))
        return kernel, report


def _hmc_with(leapfrog_class):
    class Remedy(HMC):
        def __init__(self, step_size=None, n_steps=None, *, cov=None, **kwargs):
            super().__init__(step_size, n_steps, cov=cov, **kwargs)
            self.kernels[0] = leapfrog_class(step_size, n_steps, cov)

    Remedy.__name__ = Remedy.__qualname__ = leapfrog_class.__name__.strip("_").replace(
        "Leapfrog", "HMC"
    )
    return Remedy


GuardedHMC = _hmc_with(_GuardedLeapfrog)
ContractionHMC = _hmc_with(_ContractionLeapfrog)
JitterHMC = _hmc_with(_JitterLeapfrog)
PhaseJitterHMC = _hmc_with(_PhaseJitterLeapfrog)
METHODS = {"base": HMC, "a": GuardedHMC, "b": ContractionHMC, "c": JitterHMC, "d": PhaseJitterHMC}


def gaussian_target(d, scales=None):
    # examples.gaussian(d), or N(0, diag(scales^2)) with the same initial N(0, 4 I).
    if scales is None:
        return telescopic.examples.gaussian(d)
    precision = np.diag(1 / np.asarray(scales) ** 2)
    base = telescopic.examples.gaussian(d)
    return Example(
        "gaussian-scales",
        functools.partial(_gaussian_log_prob, precision=precision),
        functools.partial(_gaussian_grad, precision=precision),
        base.initial,
        np.zeros(d),
        np.diag(np.asarray(scales) ** 2),
    )


def run_pilot(example, method, n_steps, seed, cap, cov=None, initial=None, n_pairs=100):
    """Tune as `sample` does, run 100 lag-1 pilot pairs at `cap`, then choose_k."""
    warnings.simplefilter("ignore")
    start = time.time()
    kernel = METHODS[method](n_steps=n_steps, cov=cov)
    sampler = CoupledSampler(
        example.log_prob,
        example.initial if initial is None else initial,
        vectorize=True,
        grad=example.grad,
        kernel=kernel,
        rng=seed,
    )
    target = sampler.target
    sampler.tune()
    tune_grads = target.n_grad
    leapfrog = sampler.kernel.kernels[0]
    eps, n = leapfrog.step_size, leapfrog.n_steps
    jitter = getattr(leapfrog, "jitter", 0.0) if method in ("c", "d") else 0.0
    chol = np.linalg.cholesky(sampler.kernel.cov)
    precision = np.linalg.inv(example.cov)
    w = np.sqrt(np.linalg.eigvalsh(chol.T @ precision @ chol))
    f = jitter_factor(eps, n, w, jitter)
    record = {
        "method": method,
        "n_steps": n_steps,
        "seed": seed,
        "cap": cap,
        "eps": eps,
        "n": n,
        "jitter": jitter,
        "w_min": float(w.min()),
        "w_max": float(w.max()),
        "factor_max": float(f.max()),
        "factor_min": float(f.min()),
        "tune_grads": tune_grads,
        "extra": getattr(leapfrog, "guard", None)
        or (
            None
            if not hasattr(leapfrog, "contraction")
            else {"eps_da": leapfrog.contraction["eps_da"]}
        ),
    }
    if cap:
        tau = sampler.meeting_times(n_pairs, max_iter=cap)
        try:
            k, error = choose_k(tau), None
        except ValueError:
            k, error = None, "k cannot be chosen"
        finite = tau[np.isfinite(tau)]
        record.update(
            unmet=int(np.isinf(tau).sum()),
            median=float(np.median(tau)),
            q90=float(np.quantile(tau, 0.9)),
            max_finite=float(finite.max()) if len(finite) else None,
            k=k,
            error=error,
            pilot_grads_per_pair=(target.n_grad - tune_grads) / n_pairs,
        )
    record["seconds"] = round(time.time() - start, 2)
    return record


class Store:
    """JSON lines keyed by configuration, so that a rerun skips what is done."""

    def __init__(self, name):
        # Beside this file, and not committed: the tables of the outputs are printed from it.
        self.path = Path(__file__).with_name(name)
        self.records = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r = json.loads(line)
                self.records[r["key"]] = r

    def __contains__(self, key):
        return key in self.records

    def add(self, key, record):
        record["key"] = key
        self.records[key] = record
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


class Budget:
    """Stops a run before ten minutes; the next run resumes."""

    def __init__(self, seconds=540):
        self.end = time.time() + seconds

    def left(self):
        return time.time() < self.end

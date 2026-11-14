"""Shared code of the i137 experiments (issue #137): the remedies as kernels, and a runner.

The remedies are small subclasses of the package's HMC and of its private leapfrog
component; none of them copies package code.

- `GuardedHMC` (remedy a): after the package's tuning, the predicted contraction factor
  |cos(n theta(eps w))|, theta(h) = 2 arcsin(h / 2), of every whitened mode w and of
  every frequency within a factor 1 + BAND of it (w f, f in [1/1.3, 1.3]), is computed
  on the grid eps_DA exp(s), s in [-2, 0.1] by 0.001, and the step moves to the grid
  step nearest to eps_DA (in log) at which the worst factor is at most BOUND. The modes
  come from the pilot: w^2 are the eigenvalues of C^T S^-1 C, with C C^T the kernel's
  covariance and S the pilot covariance, so w = 1 when the covariance is the pilot's. A
  step with eps w >= 2 counts as factor 1 (unstable, rejected). Besides keeping the
  step, the guard has three behaviors, which the outputs count: it lowers the step,
  away from a resonance n theta = j pi with j >= 1; it raises it, where the trajectory
  is so short that the j = 0 end of the bound binds (cos(n theta) <= 0.9 at the band's
  lowest frequency, w / 1.3, needs n theta >= arccos 0.9 = 0.451, so n eps >= about
  1.3 x 0.451 = 0.59 at w = 1); and where no step of its grid meets the rule, it falls
  back to the grid step of smallest worst factor (nearest on ties), whose worst factor
  can be near 1. The record's `extra` holds eps_DA, the worst factor at eps_DA and at
  the returned step, and `passed`, whether a grid step met the rule.
- `ContractionHMC` (remedy b): after the package's tuning, 16 steps eps_DA exp(s), s
  evenly from -1.5 to 0.3, are scored on 2 pairs of pilot states per pilot chain (the
  chain's last state against its states at the middle and the quarter of its kept
  half), by 10 coupled
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
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from i85_setup import setup

import telescopic
from telescopic._random import TUNING, replicate_generator, root_seed
from telescopic.examples import Example, _gaussian_grad, _gaussian_log_prob
from telescopic.kernels.base import evaluate, select
from telescopic.kernels.hmc import HMC, _Leapfrog
from telescopic.sampler import CoupledSampler
from telescopic.tuning import choose_k

BAND = 0.3  # remedy a: relative error of the predicted frequencies (set in i137_predictions.md)
BOUND = 0.9  # remedy a: largest acceptable predicted contraction factor
GUARD_GRID = np.round(np.arange(-2000, 101) / 1000, 3)  # remedy a: steps eps_DA exp(.)
JITTER = 0.5  # remedy c, and the largest width of remedy d (#91)
PAIRS = 100  # pilot pairs, as `sample` runs them
# The commit of the package at which the records of the stores were computed. Those
# records do not carry their setup, and the scripts print their tables later, at other
# commits, so each output names this commit beside the setup of its print pass. Records
# added since carry their own setup line (Store.add).
RECORDS_COMMIT = "b0b8f71"


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


def pilot_frequencies(chol, pilot_cov):
    # Whitened frequencies predicted from the pilot: sqrt eig(C^T S^-1 C).
    return np.sqrt(np.linalg.eigvalsh(chol.T @ np.linalg.solve(pilot_cov, chol)))


def band(w):
    # The frequencies remedy a guards: w and every frequency within a factor 1 + BAND of it.
    return np.concatenate([w * f for f in np.geomspace(1 / (1 + BAND), 1 + BAND, 21)])


def worst_factor(steps, n_steps, w):
    # Remedy a's predicted worst factor at each step over band(w); n_steps None follows
    # the package's rule.
    frequencies = band(w)
    leapfrog = _Leapfrog(None, n_steps, None)
    return np.array(
        [factor(e, leapfrog.with_step_size(float(e)).n_steps, frequencies).max() for e in steps]
    )


# Remedy a.
class _GuardedLeapfrog(_Leapfrog):
    def _with_cov(self, cov):
        # _n_steps, the given count: n_steps is the count that follows the step when none
        # is given.
        return _GuardedLeapfrog(self.step_size, self._n_steps, cov)

    def _tuned(self, target, initial, pilot, rng):
        kernel, report = super()._tuned(target, initial, pilot, rng)
        if self.step_size is not None:
            return kernel, report
        eps0 = kernel.step_size
        s = GUARD_GRID
        steps = eps0 * np.exp(s)
        worst = worst_factor(steps, self._n_steps, pilot_frequencies(kernel._chol, pilot.cov))
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
    GRID, TRAJECTORIES = np.linspace(-1.5, 0.3, 16), 10

    def _with_cov(self, cov):
        return _ContractionLeapfrog(self.step_size, self._n_steps, cov)

    def _tuned(self, target, initial, pilot, rng):
        kernel, report = super()._tuned(target, initial, pilot, rng)
        if self.step_size is not None:
            return kernel, report
        eps0 = kernel.step_size
        kept = pilot.samples.shape[1]
        # Two pairs per chain: its last state against two earlier states of its kept half.
        xs = np.concatenate([pilot.samples[:, -1]] * 2)
        ys = np.concatenate([pilot.samples[:, kept // 2], pilot.samples[:, kept // 4]])
        p, g, d = len(xs), len(self.GRID), xs.shape[1]
        steps = eps0 * np.exp(self.GRID)
        # Rows: grid value major, pair minor; the first chains, then the second.
        x = np.concatenate([np.tile(xs, (g, 1)), np.tile(ys, (g, 1))])
        state = evaluate(target, x, True)
        gap0 = np.linalg.norm(x[: g * p] - x[g * p :], axis=1)
        # A stream of the tuning namespace that the package's tuning (indices 0 to 7) does
        # not read.
        generator = replicate_generator(root_seed(rng), TUNING, 100)
        rows_eps = np.tile(np.repeat(steps, p), 2)[:, None]
        n_rows = np.tile(np.repeat([kernel.with_step_size(float(e)).n_steps for e in steps], p), 2)
        for _ in range(self.TRAJECTORIES):
            momentum = generator.standard_normal((p, d))
            log_u = np.log(generator.random(p))
            new = state.take(slice(None))
            for n in np.unique(n_rows):
                rows = np.flatnonzero(n_rows == n)
                # _trajectory takes one count for all its rows, and a step per row.
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
            # theta is undefined from eps w = 2 on, where the leapfrog is unstable.
            t = float(theta(min(kernel.step_size * w_min, 1.999)))
            kernel.jitter = min(JITTER, math.pi / (2 * kernel.n_steps * t))
        return kernel, report


def guard_summary(records):
    """What remedy a's guard did over records: counts, and its predicted worst factor."""
    moves = Counter(
        "fallback"
        if not r["extra"]["passed"]
        else "kept"
        if r["eps"] == r["extra"]["eps_da"]
        else "lowered"
        if r["eps"] < r["extra"]["eps_da"]
        else "raised"
        for r in records
    )
    worst = [r["extra"]["worst"] for r in records]
    return (
        f"kept {moves['kept']:2}, lowered {moves['lowered']:2}, raised {moves['raised']:2}, "
        f"fallback {moves['fallback']:2}; predicted worst factor at the returned step "
        f"[{min(worst):.6f}, {max(worst):.6f}]"
    )


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


def run_pilot(example, method, n_steps, seed, cap, cov=None):
    """Tune as `sample` does, run 100 lag-1 pilot pairs at `cap`, then choose_k.

    With cap = 0 it only tunes, and the record has no pilot fields. q90 is the 90%
    quantile of the meeting times, NaN where more than 10 pairs are unmet (censored).
    """
    warnings.simplefilter("ignore")
    start = time.time()
    kernel = METHODS[method](n_steps=n_steps, cov=cov)
    sampler = CoupledSampler(
        example.log_prob,
        example.initial,
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
    jitter = getattr(leapfrog, "jitter", 0.0)
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
        tau = sampler.meeting_times(PAIRS, max_iter=cap)
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
            pilot_grads_per_pair=(target.n_grad - tune_grads) / PAIRS,
        )
    record["seconds"] = round(time.time() - start, 2)
    return record


class Store:
    """JSON lines keyed by configuration, so that a rerun skips what is done."""

    def __init__(self, name):
        # In _rows/, which git ignores (as i102_common.py): the outputs print from it.
        self.path = Path(__file__).with_name("_rows") / name
        self.records = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r = json.loads(line)
                self.records[r["key"]] = r

    def __contains__(self, key):
        return key in self.records

    def add(self, key, record):
        record["key"] = key
        record.setdefault("setup", setup())
        self.records[key] = record
        self.path.parent.mkdir(exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def header(*stores):
    """The head of an output: the setup of the print pass and the commit of its records.

    Raises if the records of the stores were computed at more than one commit.
    """
    commits = {
        re.search(r"commit (\S+) of the package", r["setup"])[1]
        if "setup" in r
        else RECORDS_COMMIT
        for store in stores
        for r in store.records.values()
    }
    if len(commits) != 1:
        raise ValueError(f"the stores mix records of commits {sorted(commits)}: rerun them")
    return (
        f"{setup()}\nrecords computed at commit {commits.pop()} of the package, and printed "
        "from the stores with the setup above"
    )


class Budget:
    """Starts no new record after `seconds` (the one under way ends); a rerun resumes."""

    def __init__(self, seconds=540):
        self.end = time.time() + seconds

    def left(self):
        return time.time() < self.end

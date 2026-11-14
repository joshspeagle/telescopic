"""Criterion 6 of experiment x123: runs with and without the check are byte-identical.

Question: does building a sampler that runs the check (as (N) or as (H)) when its kernel
needs gradients change any result of its runs, or the state of a generator passed as
rng?

Design: ``Checked`` subclasses CoupledSampler and calls gradcheck.check at the end of
construction when the kernel needs gradients, with the sampler's root seed (a
SeedSequence, so a generator given as rng is not advanced again) and its args and
kwargs, as the package would. On examples.gaussian (d = 10, rho = 0.9): MALA and HMC with
default tuning and with fixed parameters (MALA step 0.7, HMC step 0.3 with 4 leapfrog steps, both
with the target's covariance); vectorized functions and per-point wrappers of
them; rng an integer or a Generator; seeds 391000 and 391001 (per-point runs on 391000
only): 24 settings. In each, a sampler without the check, one with (N) and one with (H)
each run(30); the fingerprint is the atoms and weights of the measure, the meeting times,
k, m, the lag, the pilot meeting times, the estimate of the mean, the target's evaluation
counters and the tuned step, and, for a Generator, its state after the run.
Also the cost of the check against the run, in evaluations and in time.

Run: python notes/experiments/x123/integration.py [--smoke]

Result: see integration.out.
"""

import itertools
import time
import warnings

import families as F
import gradcheck as G
import numpy as np
from runner import header, smoke

from telescopic import CoupledSampler
from telescopic.kernels import HMC, MALA

SEEDS = F.SMOKE if smoke() else (391_000, 391_001)


class Checked(CoupledSampler):
    def __init__(self, log_prob, init, *, floor, grad=None, vectorize=False, **kwargs):
        super().__init__(log_prob, init, grad=grad, vectorize=vectorize, **kwargs)
        self.verdict, self.check_seconds = None, 0.0
        if self.kernel.needs_grad:
            start = time.perf_counter()
            self.verdict = G.check(
                log_prob,
                grad,
                self.initial,
                self.seed,
                floor=floor,
                vectorize=vectorize,
                args=self.target.args,
                kwargs=self.target.kwargs,
            )
            self.check_seconds = time.perf_counter() - start


def fingerprint(sampler, result):
    return [
        result.measure.atoms.tobytes(),
        result.measure.weights.tobytes(),
        np.asarray(result.meeting_times).tobytes(),
        result.k,
        result.m,
        result.lag,
        None if sampler.pilot is None else np.asarray(sampler.pilot).tobytes(),
        np.asarray(result.mean().value).tobytes(),
        sampler.target.n_log_prob,
        sampler.target.n_grad,
        repr(sampler.kernel.step_size),
    ]


def main():
    header()
    lp, g, init = F.example("gaussian")
    cov = F.examples.gaussian().cov  # the fixed kernels use the target's covariance
    cases = {True: (lp, g, init), False: F.per_point((lp, g, init))}
    kernels = {
        "mala tuned": lambda: "mala",
        "hmc tuned": lambda: "hmc",
        "mala fixed": lambda: MALA(step_size=0.7, cov=cov),
        "hmc fixed": lambda: HMC(step_size=0.3, n_steps=4, cov=cov),
    }
    total = same = 0
    costs = []
    for (kname, kernel), vectorize, kind, seed in itertools.product(
        kernels.items(), (True, False), ("int", "generator"), SEEDS
    ):
        if not vectorize and seed != SEEDS[0]:
            continue
        prints, states = [], []
        for floor in (None, G.DESIGNS["N"], G.DESIGNS["H"]):
            rng = seed if kind == "int" else np.random.default_rng(seed)
            lp_, g_, init_ = cases[vectorize]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                start = time.perf_counter()
                if floor is None:
                    s = CoupledSampler(
                        lp_, init_, grad=g_, vectorize=vectorize, kernel=kernel(), rng=rng
                    )
                else:
                    s = Checked(
                        lp_,
                        init_,
                        floor=floor,
                        grad=g_,
                        vectorize=vectorize,
                        kernel=kernel(),
                        rng=rng,
                    )
                r = s.run(30)
                seconds = time.perf_counter() - start
            prints.append(fingerprint(s, r))
            states.append(None if kind == "int" else rng.bit_generator.state["state"]["state"])
            if floor is not None:
                costs.append(
                    (
                        s.verdict.n_log_prob,
                        s.verdict.n_grad,
                        prints[0][-3],
                        prints[0][-2],
                        s.check_seconds,
                        seconds,
                        s.verdict.warned,
                    )
                )
        ok = prints[0] == prints[1] == prints[2] and states[0] == states[1] == states[2]
        total += 1
        same += ok
        print(
            f"{kname:<11} vectorize={vectorize!s:<5} rng={kind:<9} seed {seed}: identical (none, N, H) {ok}; "
            f"k={r.k} m={r.m}; target counters {prints[0][-3]} log_prob, {prints[0][-2]} grad",
            flush=True,
        )
    c = np.array(costs, dtype=float)
    print(f"identical {same}/{total}")
    print(
        f"check: {int(c[:, 0].max())} log_prob + {int(c[:, 1].max())} grad; run median {np.median(c[:, 2]):.0f} log_prob + "
        f"{np.median(c[:, 3]):.0f} grad; check / run evaluations of log_prob, median {np.median(c[:, 0] / c[:, 2]):.2e}; "
        f"time, median {np.median(c[:, 4]) * 1e3:.2f} ms against {np.median(c[:, 5]):.2f} s; warnings {int(c[:, 6].sum())} of {len(c)}"
    )


if __name__ == "__main__":
    main()

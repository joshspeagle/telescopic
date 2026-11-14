"""Is `WhitenedMALA` (wm_kernel.py) the same kernel as `MALA`?

Question. The whitened form computes MALA's proposal and log acceptance ratio with 3
products a step and 7 a coupled step, against 7 and 14. Before any timing counts, it must
be the same kernel: the same ratios up to rounding, a coupling that passes
`telescopic.testing.check_coupling`, chains that stay equal, and the same decisions from
the same draws except where rounding straddles the threshold.

Design. Targets: N(0, Sigma), Sigma_ij = 0.9^|i-j| (`examples.gaussian(d)`), with
`log_prob_and_grad` computing the gradient as one dense product with the precision
matrix, vectorized (BLAS products in the kernels) or point by point (einsum products).
Both kernels get cov = Sigma and step size f * 1.65 d^(-1/6) (the package's first step
size), f = 1 or 2.
A. Single step from fixed states. d = 2, 10, 50, 100, 200; states drawn from the target
   (`target`) or from N(0, 4 I), its default initial distribution (`wide`); f = 1, 2;
   vectorized, and point by point at d <= 50; 100 batches of 64 states per
   configuration. Each form proposes from the same normals z (current: `mala._mean`,
   then the mean plus L z, `mala._log_alpha`; whitened: `WhitenedMALA.propose`), and
   both decide with the same log u. Reported: the largest |log alpha difference|, the
   largest |log q| for scale, the largest proposal difference relative to the row's
   largest coordinate, the rows whose decisions differ, and how many of those have log u
   between the two ratios (a difference of rounding at the threshold).
B. Coupled step with the same draws (`random_draws`, one seed for both forms). d = 10,
   50, 200, f = 1, vectorized; 4,096 pairs each of: equal pairs, close pairs
   (y = x + L v / sqrt(d), v ~ N(0, I), about half meet) and far pairs (independent
   draws from the target). Reported: pairs that meet in each form, pairs whose meeting
   differs, chains whose move-or-stay differs, and the largest position difference
   relative to the row's largest coordinate among the others.
C. `check_coupling(kernel, target, x, y, n=10_000, rng=seed)` for both forms, d = 2,
   10, 50, 200, vectorized, and d = 10 point by point; x a draw from the target,
   y = x + L v with |v| about 0.75; seeds 1, 2, 3.
D. Faithfulness: 64 equal pairs and 64 close pairs, d = 50, vectorized, iterated 500
   coupled steps of the whitened form; pairs not bitwise equal at the end among the
   equal ones, and among the close ones, pairs that separated after meeting.
E. Seeded runs: `sample` with each form, d = 50, f = 1, vectorized, initial N(0, 4 I),
   128 replicates, k = 20, m = 100, lag = 1, rng = 0 to 4. Reported: meeting times that
   differ between the forms, the largest difference of an estimate of E[x_i] in units of
   its standard error, and each form's largest |z| of E[x_i] (truth 0, 50 coordinates)
   and z of E[x_0^2] (truth 1).

Seeds: states and pairs from `np.random.default_rng(1000 + d)` (A), `(2000 + d)` (B),
`(3000 + d)` (C's points), `(4000)` (D); draws as listed.

Result (`wm_same_kernel.out`, commit 9932df7). A: over 204,800 states (6,400 per
configuration), the log ratios of the two forms differ by at most 9.9e-14 from states
drawn from the target with f = 1 (d = 200, where |log q| reaches 33), 5.1e-13 with f = 2
(d = 2, |log q| up to 182), and 4.6e-12 from wide states (d = 2, f = 2, |log q| up to
3.8e3, a relative difference of 1.2e-15); the proposals differ by at most 3.7e-14 of
their largest coordinate (d = 2) and 3.6e-15 at d >= 50. No decision differed, so no case at the threshold arose (with f
= 1 from the target the acceptance rate was 0.56 to 0.64; the other settings accept
nearly always or nearly never, so they test few decisions near the threshold).
Finiteness never differed. B: from the same draws the two forms met in exactly the same
pairs in all 36,864 pairs (every equal pair; 2,043, 1,797 and 1,637 of 4,096 close
pairs at d = 10, 50, 200; 1,004 far pairs at d = 10 and none at d = 50 and 200), and no
chain's move-or-stay differed; positions agree to 2.2e-14 of the largest coordinate.
C: both forms pass `check_coupling` in all 15 (d, vectorize, seed) cases, with no stale
state and no near miss; with the same draws they make the same decisions, so their
p-values and met fractions are equal. D: all 64 equal pairs stayed bitwise equal (with
equal log-densities and gradients) for 500 steps; all 64 close pairs met and none
separated. E: over 5 seeds of 128 replicates no meeting time differed between the forms
(median 10 to 11), the estimates of E[x_i] differ by at most 3.4e-15 standard errors,
and both forms give the same largest |z| over the 50 coordinates of E[x], 1.70 to 3.14,
and z of E[x_0^2] from -1.33 to 0.97.

Run from the repository root:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:<this folder> \
    python <this folder>/wm_same_kernel.py
"""

import warnings

import numpy as np
from wm_kernel import WhitenedMALA

from telescopic import Target, examples, kernels, sample
from telescopic._linalg import matvec
from telescopic.kernels import mala
from telescopic.kernels.base import evaluate_proposal
from telescopic.testing import check_coupling, random_draws

warnings.simplefilter("ignore")
LANES = 64


def dense_target(d, vectorize):
    precision = np.linalg.inv(examples.gaussian(d).cov)
    if vectorize:

        def value_and_grad(x):
            grad = -(x @ precision)
            return 0.5 * np.sum(x * grad, axis=1), grad

    else:

        def value_and_grad(x):
            grad = -(precision @ x)
            return 0.5 * float(x @ grad), grad

    log_prob = lambda x: value_and_grad(x)[0]
    return Target(log_prob, log_prob_and_grad=value_and_grad, vectorize=vectorize)


def pair_of_kernels(d, factor=1.0):
    cov = examples.gaussian(d).cov
    step_size = factor * 1.65 * d ** (-1 / 6)
    return kernels.MALA(step_size, cov=cov), WhitenedMALA(step_size, cov=cov)


def relative(a, b):
    scale = np.maximum(np.max(np.abs(a), axis=1), np.max(np.abs(b), axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.max(np.abs(a - b), axis=1) / scale


def section_a():
    print("A. Single step from fixed states: log ratios and decisions, same z and log u")
    print(
        f"{'d':>4} {'states':>7} {'f':>3} {'vec':>4} {'n':>6} {'max|dlogA|':>11} "
        f"{'max|logq|':>10} {'max rel dx':>11} {'accept':>7} {'differ':>6} {'at thr':>6} "
        f"{'finite differs':>14}"
    )
    total, differ_total = 0, 0
    for d in (2, 10, 50, 100, 200):
        cov = examples.gaussian(d).cov
        chol_sigma = np.linalg.cholesky(cov)
        for spread in ("target", "wide"):
            for factor in (1.0, 2.0):
                for vectorize in (True, False):
                    if not vectorize and d > 50:
                        continue
                    rng = np.random.default_rng(1000 + d)
                    target = dense_target(d, vectorize)
                    current, whitened = pair_of_kernels(d, factor)
                    chol, inverse = current._ready()
                    blas = target.vectorize
                    worst, worst_q, worst_x, n, accepted = 0.0, 0.0, 0.0, 0, 0
                    differ, at_threshold, finite_differs = 0, 0, 0
                    for _ in range(100):
                        w = rng.standard_normal((LANES, d))
                        x = w @ chol_sigma.T if spread == "target" else 2.0 * w
                        state = current.init_state(target, x)
                        z = rng.standard_normal((LANES, d))
                        log_u = np.log(rng.random(LANES))
                        mean = mala._mean(state, chol, blas)
                        proposal = evaluate_proposal(target, mean + matvec(chol, z, blas=blas))
                        alpha_c = mala._log_alpha(state, proposal, mean, chol, inverse, blas)
                        proposal_w, alpha_w = whitened.propose(target, state, z)
                        both = np.isfinite(alpha_c) & np.isfinite(alpha_w)
                        finite_differs += int(np.sum(np.isfinite(alpha_c) != np.isfinite(alpha_w)))
                        gap = np.abs(alpha_c - alpha_w)[both]
                        worst = max(worst, float(gap.max(initial=0)))
                        worst_q = max(
                            worst_q,
                            float(np.max(np.abs(alpha_c - proposal.log_prob + state.log_prob))),
                        )
                        worst_x = max(
                            worst_x, float(np.nanmax(relative(proposal.x, proposal_w.x)))
                        )
                        decide_c, decide_w = log_u < alpha_c, log_u < alpha_w
                        accepted += int(decide_c.sum())
                        rows = decide_c != decide_w
                        differ += int(rows.sum())
                        low = np.minimum(alpha_c, alpha_w)[rows]
                        high = np.maximum(alpha_c, alpha_w)[rows]
                        at_threshold += int(np.sum((log_u[rows] >= low) & (log_u[rows] <= high)))
                        n += LANES
                    total += n
                    differ_total += differ
                    print(
                        f"{d:>4} {spread:>7} {factor:>3.0f} {vectorize!s:>4} {n:>6} "
                        f"{worst:>11.2e} {worst_q:>10.3g} {worst_x:>11.2e} {accepted / n:>7.3f} "
                        f"{differ:>6} {at_threshold:>6} {finite_differs:>14}"
                    )
    print(f"states: {total}; decisions that differ: {differ_total}")


def section_b():
    print("\nB. Coupled step with the same draws (vectorized, f = 1, 4096 pairs per row)")
    print(
        f"{'d':>4} {'pairs':>6} {'met cur':>8} {'met wh':>7} {'met differ':>10} "
        f"{'move differs x':>14} {'move differs y':>14} {'max rel dx':>11}"
    )
    n = 4096
    for d in (10, 50, 200):
        rng = np.random.default_rng(2000 + d)
        target = dense_target(d, True)
        current, whitened = pair_of_kernels(d)
        chol = current._ready()[0]
        chol_sigma = np.linalg.cholesky(examples.gaussian(d).cov)
        x = rng.standard_normal((n, d)) @ chol_sigma.T
        pairs = {
            "equal": x.copy(),
            "close": x + rng.standard_normal((n, d)) @ chol.T / np.sqrt(d),
            "far": rng.standard_normal((n, d)) @ chol_sigma.T,
        }
        for name, y in pairs.items():
            sx, sy = current.init_state(target, x), current.init_state(target, y)
            seed = 2100 + d
            out_c = current.coupled_step(target, sx, sy, random_draws(n, seed))
            out_w = whitened.coupled_step(target, sx, sy, random_draws(n, seed))
            met_c = np.all(out_c[0].x == out_c[1].x, axis=1)
            met_w = np.all(out_w[0].x == out_w[1].x, axis=1)
            moved = [
                [np.any(new.x != old.x, axis=1) for new, old in zip(out, (sx, sy), strict=True)]
                for out in (out_c, out_w)
            ]
            move_x = int(np.sum(moved[0][0] != moved[1][0]))
            move_y = int(np.sum(moved[0][1] != moved[1][1]))
            same = (moved[0][0] == moved[1][0]) & (moved[0][1] == moved[1][1])
            worst = max(
                float(np.max(relative(out_c[0].x, out_w[0].x)[same], initial=0)),
                float(np.max(relative(out_c[1].x, out_w[1].x)[same], initial=0)),
            )
            print(
                f"{d:>4} {name:>6} {met_c.sum():>8} {met_w.sum():>7} "
                f"{int(np.sum(met_c != met_w)):>10} {move_x:>14} {move_y:>14} {worst:>11.2e}"
            )


def section_c():
    print("\nC. check_coupling, n = 10,000, from x ~ target and y = x + L v, |v| ~ 0.75")
    print(
        f"{'d':>4} {'vec':>5} {'seed':>4} {'form':>9} {'stay p':>17} {'moves p':>17} "
        f"{'met':>6} {'stale':>5} {'near':>4} {'passed':>6}"
    )
    for d, vectorize in ((2, True), (10, True), (10, False), (50, True), (200, True)):
        rng = np.random.default_rng(3000 + d)
        target = dense_target(d, vectorize)
        current, whitened = pair_of_kernels(d)
        chol = current._ready()[0]
        x = np.linalg.cholesky(examples.gaussian(d).cov) @ rng.standard_normal(d)
        v = rng.standard_normal(d)
        y = x + chol @ (0.75 * v / np.linalg.norm(v))
        for seed in (1, 2, 3):
            for name, kernel in (("MALA", current), ("whitened", whitened)):
                report = check_coupling(kernel, target, x, y, n=10_000, rng=seed)
                stay = f"{report.stay[0]:.3f}, {report.stay[1]:.3f}"
                moves = f"{report.moves[0]:.3f}, {report.moves[1]:.3f}"
                print(
                    f"{d:>4} {vectorize!s:>5} {seed:>4} {name:>9} {stay:>17} {moves:>17} "
                    f"{report.met:>6.3f} {report.stale:>5} {report.near_misses:>4} "
                    f"{report.passed()!s:>6}"
                )


def section_d():
    print("\nD. Faithfulness of the whitened coupled step, d = 50, vectorized, 500 steps")
    d, n = 50, 64
    rng = np.random.default_rng(4000)
    target = dense_target(d, True)
    _, whitened = pair_of_kernels(d)
    chol = whitened._ready()[0]
    x = rng.standard_normal((n, d)) @ np.linalg.cholesky(examples.gaussian(d).cov).T
    for name, y in (("equal", x.copy()), ("close", x + rng.standard_normal((n, d)) @ chol.T / 7)):
        sx, sy = whitened.init_state(target, x), whitened.init_state(target, y)
        draws = random_draws(n, 4001)
        met_at = np.full(n, -1)
        separated = np.zeros(n, bool)
        for t in range(500):
            sx, sy = whitened.coupled_step(target, sx, sy, draws)
            met = np.all(sx.x == sy.x, axis=1) & (sx.log_prob == sy.log_prob)
            met &= np.all(sx.grad == sy.grad, axis=1)
            separated |= (met_at >= 0) & ~met
            met_at[(met_at < 0) & met] = t + 1
        print(
            f"{name}: pairs {n}, met {int(np.sum(met_at >= 0))}, equal at the end "
            f"{int(np.sum(met))}, separated after meeting {int(separated.sum())}"
        )


def section_e():
    print("\nE. Seeded runs: sample(), d = 50, 128 replicates, k = 20, m = 100, lag = 1")
    d = 50
    example = examples.gaussian(d)
    target_functions = dense_target(d, True)
    value_and_grad = target_functions._log_prob_and_grad
    print(
        f"{'rng':>3} {'tau differ':>10} {'max|dE|/se':>11} {'max|z| MALA':>11} "
        f"{'max|z| wh':>9} {'z x0^2 MALA':>11} {'z x0^2 wh':>9} {'median tau':>10}"
    )
    for seed in range(5):
        results = []
        for kernel in pair_of_kernels(d):
            results.append(
                sample(
                    example.log_prob,
                    example.initial,
                    128,
                    log_prob_and_grad=value_and_grad,
                    vectorize=True,
                    kernel=kernel,
                    k=20,
                    m=100,
                    lag=1,
                    rng=seed,
                )
            )
        means = [r.mean() for r in results]
        squares = [r.estimate(lambda x: x[:, 0] ** 2) for r in results]
        differ = int(np.sum(results[0].meeting_times != results[1].meeting_times))
        shift = float(np.max(np.abs(means[0].value - means[1].value) / means[0].standard_error))
        z = [float(np.max(np.abs(m.value / m.standard_error))) for m in means]
        z2 = [float((s.value - 1.0) / s.standard_error) for s in squares]
        print(
            f"{seed:>3} {differ:>10} {shift:>11.2e} {z[0]:>11.2f} {z[1]:>9.2f} "
            f"{z2[0]:>11.2f} {z2[1]:>9.2f} {np.median(results[0].meeting_times):>10.1f}"
        )


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()

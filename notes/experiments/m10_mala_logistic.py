"""Whether the automatic MALA couples on a log-concave logistic regression (M10).

QUESTIONS.md assumes that the automatic step of the gradient kernels couples on
log-concave targets. The model: 200 binary outcomes with P(y_i = 1) = expit(x_i . beta),
x_i = (1, z_i) with z_i ~ N(0, I_5), data drawn from np.random.default_rng(1), and the
prior beta ~ N(0, 5^2 I_6). Two designs: the coefficients (-0.5, 1, -1, 0.5, 0, 2),
whose predictors nearly separate the outcomes, and (0.5, 1, -0.5, 0.25, 0, -1), the
gradients tutorial's. The pairs start as in Getting started, from N(fit, 100 H^-1),
with BFGS's fit and inverse-Hessian approximation started at 0. Kernels are tuned by
the sampler (rng=2), and the meeting-time pilots run 100 lag-1 pairs capped at 3,000
iterations.

For the first design: the pilots of the automatic MALA and HMC, and of MALA with
explicit steps on the tuned covariance; 200 chains of the automatic MALA from draws of
the initial distribution, 300 steps each, and those that never moved, with the largest
eigenvalue of Sigma H(x0) at their starts against the others (Sigma the tuned
covariance, H the Hessian of -log pi; MALA's proposal mean overshoots a quadratic
region where it exceeds 4 / step^2); and sample(kernel="mala"), whose pilot is capped
at 10^5. For the second design: the pilots of the automatic MALA and HMC.

Run: python notes/experiments/m10_mala_logistic.py
"""

import time
import warnings

import numpy as np
from scipy import optimize, special

import telescopic
from telescopic.kernels import MALA
from telescopic.target import Target
from telescopic.testing import random_draws


def log_prob(beta, X, y):
    eta = beta @ X.T
    return np.sum(y * eta - np.logaddexp(0, eta), axis=-1) - np.sum(beta**2, axis=-1) / 50


def grad(beta, X, y):
    return (y - special.expit(beta @ X.T)) @ X - beta / 25


def hessian(beta, X):
    # The Hessian of -log pi at one point.
    p = special.expit(X @ beta)
    return X.T @ ((p * (1 - p))[:, None] * X) + np.eye(X.shape[1]) / 25


def pilot(kernel, X, y, init):
    sampler = telescopic.CoupledSampler(
        log_prob, init, args=(X, y), grad=grad, vectorize=True, kernel=kernel, rng=2
    )
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau = sampler.meeting_times(100, max_iter=3000)
    met = tau[np.isfinite(tau)]
    return (
        f"{sampler.kernel}: {np.sum(~np.isfinite(tau))} of 100 pilot pairs unmet at 3,000; "
        f"met median {np.median(met):.0f}, max {np.max(met):.0f}; "
        f"{time.perf_counter() - start:.1f} s"
    ), sampler.kernel


for truth in ([-0.5, 1.0, -1.0, 0.5, 0.0, 2.0], [0.5, 1.0, -0.5, 0.25, 0.0, -1.0]):
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(200), rng.normal(size=(200, 5))])
    y = (rng.random(200) < special.expit(X @ truth)).astype(float)
    fit = optimize.minimize(
        lambda beta: -log_prob(beta, X, y), np.zeros(6), jac=lambda beta: -grad(beta, X, y)
    )
    init = telescopic.initial.Normal(mean=fit.x, cov=100 * fit.hess_inv)
    print(f"coefficients {truth}: {y.mean():.3f} of the outcomes are 1", flush=True)
    line, mala = pilot("mala", X, y, init)
    print(f"  {line}", flush=True)
    print(f"  {pilot('hmc', X, y, init)[0]}", flush=True)
    if truth[-1] != 2.0:
        continue
    for step in (1.0, 0.7, 0.5):
        print(f"  {pilot(MALA(step, cov=mala.cov), X, y, init)[0]}", flush=True)

    target = Target(log_prob, grad, args=(X, y), vectorize=True)
    starts = np.array([init.rvs(random_state=np.random.default_rng(i)) for i in range(200)])
    state = mala.init_state(target, starts)
    draws = random_draws(200, 5)
    moves = np.zeros(200)
    for _ in range(300):
        new, _ = mala.step(target, state, draws)
        moves += np.any(new.x != state.x, axis=1)
        state = new
    ratio = np.array([np.linalg.eigvals(mala.cov @ hessian(b, X)).real.max() for b in starts])
    stuck = moves == 0
    print(
        f"  automatic MALA, 200 chains of 300 steps: {stuck.sum()} never moved; largest "
        f"eigenvalue of Sigma H(x0), median {np.median(ratio[stuck]):.1f} (range "
        f"{ratio[stuck].min():.1f} to {ratio[stuck].max():.1f}) at their starts, "
        f"{np.median(ratio[~stuck]):.1f} at the others'; 4 / step^2 = "
        f"{4 / mala.step_size**2:.1f}; at the mode "
        f"{np.linalg.eigvals(mala.cov @ hessian(fit.x, X)).real.max():.1f}",
        flush=True,
    )

    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            telescopic.sample(
                log_prob, init, args=(X, y), grad=grad, vectorize=True, kernel="mala", rng=2
            )
    except ValueError as error:
        outcome = f"ValueError: {error}"
    else:
        outcome = "a result"
    print(
        f"  sample(kernel='mala'): {outcome}; {time.perf_counter() - start:.1f} s", flush=True
    )

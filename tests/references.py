"""Reference implementations, written independently of the package, and shared test helpers."""

import numpy as np
from scipy import special

# One-sample and frequency tests use this p-value threshold, which keeps each module's
# family-wise false-positive rate below about 1e-3 (CLAUDE.md: tolerances from the Monte
# Carlo error).
P_MIN = 1e-5


def assert_same_replicates(a, b):
    # Every field but the wall time; the seed enters every array.
    for name, value in vars(a).items():
        if name not in ("seed", "seconds"):
            np.testing.assert_array_equal(value, getattr(b, name), err_msg=name)


def assert_z_scores(z_scores, n, seed):
    # The test plan's tolerance rule for z(n, rng), an array of z-scores from n draws: pass
    # if every |z| <= 4; if the largest is in (4, 6], pass if every |z| <= 4 in one rerun
    # with 4n draws and a seed derived from the first.
    z = z_scores(n, np.random.default_rng(seed))
    assert np.all(np.isfinite(z)), z
    largest = np.max(np.abs(z))
    if largest > 4:
        assert largest <= 6, f"|z| = {largest:.2f}"
        derived = np.random.SeedSequence(seed).spawn(1)[0]
        rerun = np.max(np.abs(z_scores(4 * n, np.random.default_rng(derived))))
        assert rerun <= 4, f"|z| = {largest:.2f}, then {rerun:.2f} with 4n draws"


class Normal:
    """A univariate normal with the methods of a frozen SciPy one, and faster to build."""

    def __init__(self, mean, sd):
        self.mean, self.sd = mean, sd

    def rvs(self, *, random_state):
        return self.mean + self.sd * random_state.standard_normal()

    def logpdf(self, value):
        return -0.5 * ((value - self.mean) / self.sd) ** 2 - np.log(self.sd * np.sqrt(2 * np.pi))

    def ppf(self, u):
        return self.mean + self.sd * special.ndtri(u)


def gaussian_log_prob(x):
    # The standard normal log-density, up to a constant, of a point or of rows of points.
    return -0.5 * np.sum(np.square(x), axis=-1)


def reflection(mu_x, mu_y, chol, z, u):
    # Jacob, O'Leary & Atchade (2020), section 4.1.
    delta = np.linalg.solve(chol, mu_x - mu_y)
    x = mu_x + chol @ z
    if np.log(u) < -(z @ delta) - delta @ delta / 2:
        return x, x, True
    e = delta / np.linalg.norm(delta)
    return x, mu_y + chol @ (z - 2 * (e @ z) * e), False


def metropolis(current, proposal, log_prob, u):
    return proposal if np.log(u) < log_prob(proposal) - log_prob(current) else current


def dual_averaging(acceptance, step_size, goal, n_steps=1000):
    # Hoffman & Gelman (2014), algorithm 5, with gamma = 0.05, t0 = 10, kappa = 0.75 and
    # mu = log(10 step_size), for an acceptance probability that is a function of the
    # step size.
    mu, h_bar, log_bar = np.log(10 * step_size), 0.0, 0.0
    for m in range(1, n_steps + 1):
        h_bar = (1 - 1 / (m + 10)) * h_bar + (goal - acceptance(step_size)) / (m + 10)
        log_step = mu - np.sqrt(m) / 0.05 * h_bar
        log_bar = m**-0.75 * log_step + (1 - m**-0.75) * log_bar
        step_size = np.exp(log_step)
    return np.exp(log_bar)

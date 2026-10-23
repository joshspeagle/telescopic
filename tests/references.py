"""Reference implementations, written independently of the package, for oracle tests."""

import numpy as np
from scipy import special

# One-sample and frequency tests use this p-value threshold, which keeps each module's
# family-wise false-positive rate below about 1e-3 (CLAUDE.md: tolerances from the Monte
# Carlo error).
P_MIN = 1e-5


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

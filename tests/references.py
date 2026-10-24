"""Reference implementations, written independently of the package, for oracle tests."""

import numpy as np

# One-sample and frequency tests use this p-value threshold, which keeps each module's
# family-wise false-positive rate below about 1e-3 (CLAUDE.md: tolerances from the Monte
# Carlo error).
P_MIN = 1e-5


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

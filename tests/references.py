"""Reference implementations, written independently of the package, for oracle tests."""

import numpy as np


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

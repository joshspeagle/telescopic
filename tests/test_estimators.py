import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from telescopic.estimators import (
    averaging_times,
    correction_coefficients,
    unbiased_covariance,
    weighted_quantile,
)

# (k, m, lag, thin) with 0 <= k <= m.
configurations = st.integers(0, 30).flatmap(
    lambda k: st.tuples(st.just(k), st.integers(k, k + 40), st.integers(1, 12), st.integers(1, 6))
)


def telescoping_average(hx, hy, k, m, lag, thin):
    # H_S = |S|^-1 sum_{l in S} [h(X_l) + sum_{j >= 1} (h(X_{l+jL}) - h(Y_{l+(j-1)L}))]
    # (spec section 3.2), with h(X_t) = hx[t] and h(Y_s) = hy[s].
    total = 0.0
    for l in averaging_times(k, m, thin):
        total += hx[l] + sum(hx[t] - hy[t - lag] for t in range(l + lag, len(hx), lag))
    return total / len(averaging_times(k, m, thin))


@given(configuration=configurations, delay=st.integers(0, 60), seed=st.integers(0, 2**32 - 1))
def test_the_weighted_sum_equals_the_average_of_telescoping_sums(configuration, delay, seed):
    k, m, lag, thin = configuration
    tau = lag + delay
    rng = np.random.default_rng(seed)
    hx = rng.standard_normal(max(tau, m) + 1)
    hy = rng.standard_normal(len(hx) - lag)
    hy[tau - lag :] = hx[tau:]  # the pair stays together from the meeting on
    t = np.arange(lag, tau)
    coefficients = correction_coefficients(t, k, m, lag, thin)
    weighted = hx[averaging_times(k, m, thin)].mean() + np.sum(
        coefficients * (hx[t] - hy[t - lag])
    )
    assert abs(weighted - telescoping_average(hx, hy, k, m, lag, thin)) < 1e-12


@given(configuration=configurations, rate=st.floats(0.05, 0.8))
def test_the_coefficients_telescope_to_the_limit(configuration, rate):
    # With E h(X_t) = a_t, E[H_S] = |S|^-1 sum_{l in S} a_l + sum_t c_t (a_t - a_{t-L}),
    # which must equal lim a_t for every sequence converging fast enough: the
    # unbiasedness of the weights, checked against the truth rather than a reference.
    k, m, lag, thin = configuration
    horizon = m + lag + 200  # rate**200 < 1e-19
    a = 1.0 + rate ** np.arange(horizon)
    t = np.arange(lag, horizon)
    expectation = a[averaging_times(k, m, thin)].mean() + np.sum(
        correction_coefficients(t, k, m, lag, thin) * (a[t] - a[t - lag])
    )
    assert abs(expectation - 1.0) < 1e-12


def test_lag_one_coefficients_follow_jacob_oleary_atchade_equation_2_1():
    k, m = 5, 20
    t = np.arange(k + 1, 60)
    expected = np.minimum(1.0, (t - k) / (m - k + 1))
    np.testing.assert_allclose(correction_coefficients(t, k, m, 1, 1), expected, rtol=1e-15)


def test_the_covariance_is_the_mean_second_moment_minus_the_u_statistic():
    rng = np.random.default_rng(1)
    first, second = rng.standard_normal((7, 3)), rng.standard_normal((7, 3, 3))
    u_statistic = sum(
        np.outer(first[r], first[s]) for r in range(7) for s in range(7) if r != s
    ) / (7 * 6)
    np.testing.assert_allclose(
        unbiased_covariance(first, second), second.mean(axis=0) - u_statistic, atol=1e-14
    )


@given(
    counts=st.lists(st.integers(1, 5), min_size=1, max_size=12),
    q=st.floats(0.0, 1.0),
    seed=st.integers(0, 2**32 - 1),
)
def test_quantiles_of_positive_weights_are_inverted_cdf_quantiles(counts, q, seed):
    # Integer weights are repeated values, so NumPy's inverted-CDF quantile is the
    # reference; ties between values exercise the merging of equal atoms.
    values = np.random.default_rng(seed).integers(0, 6, size=len(counts)).astype(float)
    weights = np.array(counts) / sum(counts)
    expected = np.quantile(np.repeat(values, counts), q, method="inverted_cdf")
    assert weighted_quantile(values, weights, q) == expected


def test_quantiles_of_a_signed_measure_are_monotone_and_inside_the_atoms():
    rng = np.random.default_rng(2)
    values = rng.standard_normal(200)
    weights = rng.standard_normal(200)
    weights += (1 - weights.sum()) / 200
    levels = np.linspace(0, 1, 101)
    quantiles = weighted_quantile(values, weights, levels)
    assert np.all(np.diff(quantiles) >= 0)
    assert np.isin(quantiles, values).all()

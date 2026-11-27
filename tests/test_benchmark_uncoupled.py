"""Check accounting and diagnostic gating for warmed variance references."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from benchmarks import research, targets, uncoupled


def test_warmup_and_retained_evaluations_are_counted_separately_from_estimation(
    monkeypatch,
) -> None:
    observed = []

    def rhat(samples, **kwargs):
        observed.append(samples.copy())
        return np.ones(samples.shape[-1])

    monkeypatch.setattr(
        uncoupled,
        "import_module",
        lambda name: SimpleNamespace(
            rhat=rhat, ess=lambda samples, **kwargs: 1000 * np.ones(samples.shape[-1])
        ),
    )
    case = replace(targets.rotated_gaussian(2), draw_exact=None)
    _, kernel = research.random_walk_setup(case, "matched", "oracle")
    first = uncoupled.run(
        case,
        kernel,
        warmup=3,
        batches=20,
        batch_size=10,
        seed=702,
    )
    second = uncoupled.run(
        case,
        kernel,
        warmup=3,
        batches=20,
        batch_size=10,
        seed=702,
    )
    assert first == second
    assert first.evaluations == 4 * (1 + 3 + 200)
    assert first.gradient_evaluations == 0
    assert observed[0].shape == (4, 200, 2)
    assert not np.array_equal(observed[0][0], observed[0][1])
    np.testing.assert_array_equal(observed[0], observed[1])


def test_diagnostic_gate_withholds_failed_variance_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        uncoupled,
        "import_module",
        lambda name: SimpleNamespace(
            rhat=lambda samples, **kwargs: [1.4, 1.0], ess=lambda samples, **kwargs: [1500, 1800]
        ),
    )
    case = targets.rotated_gaussian(2)
    samples = np.random.default_rng(11).standard_normal((4, 400, 2))
    result = uncoupled.diagnose(case, samples, 20, 1700, 800)
    assert not result.passed
    assert not result.variances
    assert result.max_rhat == 1.4
    assert result.evaluations == 2500
    assert result.gradient_evaluations == 800


def test_undefined_chain_diagnostics_withhold_the_denominator(monkeypatch) -> None:
    monkeypatch.setattr(
        uncoupled,
        "import_module",
        lambda name: SimpleNamespace(
            rhat=lambda samples, **kwargs: np.divide(0.0, 0.0),
            ess=lambda samples, **kwargs: 1500,
        ),
    )
    case = targets.rotated_gaussian(2)
    result = uncoupled.diagnose(case, np.zeros((4, 400, 2)), 20, 1600, 0)
    assert not result.passed
    assert not result.variances
    assert result.max_rhat is None


def test_reference_start_scale_applies_to_independent_matched_draws(monkeypatch) -> None:
    monkeypatch.setattr(
        uncoupled,
        "import_module",
        lambda name: SimpleNamespace(
            rhat=lambda *args, **kwargs: 1.0, ess=lambda *args, **kwargs: 1000
        ),
    )
    case = targets.rotated_gaussian(2)
    evaluated = []

    def log_prob(points):
        evaluated.append(points.copy())
        return case.log_prob(points)

    _, kernel = research.random_walk_setup(case, "matched", "oracle")
    ref = uncoupled.run(
        replace(case, log_prob=log_prob),
        kernel,
        warmup=3,
        batches=20,
        batch_size=10,
        seed=702,
        start_scale=2.0,
    )
    expected = [
        case.matched.mean
        + 2
        * (
            case.matched.rvs(random_state=np.random.default_rng(lineage.spawn(2)[0]))
            - case.matched.mean
        )
        for lineage in np.random.SeedSequence(702).spawn(4)
    ]
    np.testing.assert_allclose(evaluated[0], expected)
    assert ref.start_scale == 2.0

"""Check evaluation accounting and censoring in the research pilot."""

from typing import Literal

import pytest

from benchmarks import research, targets


def test_random_walk_pilot_records_the_budget_and_censoring_per_pair() -> None:
    case = targets.rotated_gaussian(10)

    def pilot(proposal: Literal["oracle", "identity"]) -> list[research.PairRecord]:
        return research.random_walk_pilot(
            case, start="matched", proposal=proposal, pairs=20, budget=350, seed=702
        )

    matched = pilot("oracle")
    identity = pilot("identity")
    assert matched == pilot("oracle")
    assert all(0 < row.evaluations <= 350 for row in matched + identity)
    assert all(row.meeting_cost == row.evaluations for row in matched if row.meeting_cost)
    assert research.summarize_pilot(identity)["censored_fraction"] > 0
    assert research.summarize_pilot(matched)["censored_fraction"] < 1


def test_pilot_quantiles_keep_censored_pairs_in_the_denominator() -> None:
    records = [research.PairRecord(12, 12, 5), research.PairRecord(9, 9, 4)]
    records.append(research.PairRecord(19, None, None))
    assert research.summarize_pilot(records) == {
        "pairs": 3,
        "censored_fraction": 1 / 3,
        "evaluations": 40,
        "q50": 12,
        "q90": None,
        "q99": None,
        "max": None,
    }


def test_pair_can_meet_after_the_conservative_iteration_cap() -> None:
    case = targets.rotated_gaussian(2)
    records = research.random_walk_pilot(
        case, start="matched", proposal="oracle", pairs=23, budget=15, seed=702
    )
    assert records[22] == research.PairRecord(10, 10, 8)


def test_recorded_cli_rejects_an_installed_copy(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(research.telescopic, "__file__", str(tmp_path / "__init__.py"))
    with pytest.raises(RuntimeError, match="PYTHONPATH=src"):
        research.require_source_checkout()

"""Check that the registered launcher fixes the seeds and budget per cell."""

import json
import sys

from benchmarks import registered


def test_case_ids_cover_the_agreed_suite_and_preserve_the_radial_velocity_gap() -> None:
    cases = registered._cases()
    assert len(cases) == 24
    assert {case.target_id for case in cases.values()} == {1, 2, 3, 4, 5, 6, 7, 9, 10}
    assert cases["rotated_gaussian:1000"].budget == 1_000_000
    assert cases["mixture_equal_12:32"].budget == 100_000
    assert cases["parallax:501"].budget == 200_000
    assert all(case.name in registered.research.TARGETS for case in cases.values())


def test_selected_cells_keep_their_budgets_and_share_registered_seeds(monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr(registered, "setup", lambda: {"commit": "fixture"})
    monkeypatch.setattr(registered.research, "require_source_checkout", lambda: None)
    monkeypatch.setattr(
        registered.compare,
        "compare_case",
        lambda case, **kwargs: calls.append((case.name, kwargs)) or {},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "registered",
            "--case",
            "banana:2",
            "--case",
            "mixture_equal_6:2",
            "--start",
            "wide",
            "--method",
            "hmc",
        ],
    )
    registered.main()
    report = json.loads(capsys.readouterr().out)
    assert [cell["target_id"] for cell in report["comparisons"]] == [2, 4]
    assert [args["budget"] for _, args in calls] == [200_000, 100_000]
    for _, args in calls:
        assert args["methods"] == ["hmc"]
        assert args["start"] == "wide"
        assert args["pairs"] == 100
        assert args["tuning_seed"] == 20261101
        assert [args["seed"], *args["additional_meeting_seeds"]] == list(range(20261201, 20261206))

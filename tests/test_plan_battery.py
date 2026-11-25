"""Golden-plan regression: planning output for unrelated questions must not drift (see scripts/plan_battery.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan_battery  # noqa: E402


def test_every_plan_matches_the_committed_golden():
    problems = plan_battery.diff(plan_battery.run_battery(), plan_battery.load_golden())
    assert not problems, (
        "\n\n".join(problems)
        + "\n\nA changed plan for a question you did not mean to affect is a collision between concepts or datasets. If the "
        "change IS intended, run `python scripts/plan_battery.py --update --groups <group>` and review the golden diff."
    )


def test_zhvi_levels_join_on_their_own_grain():
    """Each ZHVI level's plan joins on the key its grain uses, checked beside the golden so a careless --update
    cannot freeze a wrong join: a metro plan showing houses.zip = zhvi.region_name returns no rows, and a
    neighborhood plan that keeps a houses join cannot answer at all (no house column holds a neighborhood name).
    """
    plans = plan_battery.run_battery({"zhvi_levels"})["zhvi_levels"]
    lines = {q: "\n".join(plan) for q, plan in plans.items()}

    assert "houses.zip = zhvi.region_name" in lines["typical home value by zip"]
    assert "houses.msa_code = zhvi.msa_code" in lines["home value by metro"]
    for question in ("home value by metro", "metro home value in Pittsburgh"):
        assert "zhvi.region_name" not in lines[question], question
    assert "required_tables: zhvi" in lines["home value by neighborhood"]
    assert "required_relationships: none" in lines["home value by neighborhood"]


def test_the_zero_drift_groups_are_present_and_substantial():
    golden = plan_battery.load_golden()
    assert len(golden["unrelated"]) >= 20 and len(golden["near_miss"]) >= 15
    # sanity: the golden really records plans (not empty strings) for house questions
    assert any("houses" in "\n".join(lines) for lines in golden["unrelated"].values())

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
          "change IS intended, run `python scripts/plan_battery.py --update --groups <group>` and review the golden diff.")


def test_the_zero_drift_groups_are_present_and_substantial():
    golden = plan_battery.load_golden()
    assert len(golden["unrelated"]) >= 20 and len(golden["near_miss"]) >= 15
    # sanity: the golden really records plans (not empty strings) for house questions
    assert any("houses" in "\n".join(lines) for lines in golden["unrelated"].values())

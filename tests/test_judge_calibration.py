"""The scorer must keep discriminating: calibration answers with known verdicts.

Without this, a change to eval/scoring.py (or to a judge) that makes everything PASS is
invisible: every stored score would look the same and the trend would look like improvement.
"""

from __future__ import annotations

import pytest

from eval import calibration


def test_every_calibration_case_scores_as_expected():
    rows = calibration.score_cases()
    failures = [f"{r['id']}: expected {r['expected']}, got {r['verdict']} ({r['reason']})" for r in rows if not r["ok"]]
    assert not failures, "\n".join(failures)


def test_calibration_covers_both_verdicts():
    verdicts = {r["expected"] for r in calibration.score_cases()}
    assert verdicts == {"PASS", "FAIL"}, "a set of only-passing cases cannot detect an always-pass scorer"


def test_calibration_ids_are_unique():
    ids = [c.id for c in calibration.CALIBRATION_CASES]
    assert len(ids) == len(set(ids)) == 10


@pytest.mark.parametrize("case", calibration.CALIBRATION_CASES, ids=lambda c: c.id)
def test_each_case_is_a_structured_example(case):
    assert case.example.answer_type == "structured"
    assert case.expected_verdict in {"PASS", "FAIL"}


def test_run_without_a_judge_only_runs_the_deterministic_half():
    rows, ok = calibration.run(judge=None)
    assert ok
    assert {r["id"] for r in rows} == {c.id for c in calibration.CALIBRATION_CASES}

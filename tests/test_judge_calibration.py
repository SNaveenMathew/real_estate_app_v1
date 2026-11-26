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


def test_judge_calibration_covers_sixteen_balanced_controls():
    assert len(calibration.JUDGE_CASES) >= 16
    ids = [c.id for c in calibration.JUDGE_CASES]
    assert len(ids) == len(set(ids))
    verdicts = {c.expected_verdict for c in calibration.JUDGE_CASES}
    assert verdicts == {"PASS", "FAIL"}
    assert any(c.artifacts for c in calibration.JUDGE_CASES)


def test_judge_calibration_executes_with_stub_judge_and_forwards_artifacts():
    seen = []

    class MirrorJudge:
        def judge(self, example, reply, *, artifacts=()):
            seen.append((example.id, artifacts))
            expected = next(c.expected_verdict for c in calibration.JUDGE_CASES if c.id == example.id)
            from eval.scoring import ScoreResult

            return ScoreResult(example.id, expected, "llm_judge", "stub")

    rows = calibration.judge_rows(MirrorJudge())
    assert len(rows) == len(calibration.JUDGE_CASES) and all(r["ok"] for r in rows)
    assert any(cid == "cal-judge-artifact-table-pass" and arts for cid, arts in seen)

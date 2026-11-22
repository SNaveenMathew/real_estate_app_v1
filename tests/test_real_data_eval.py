"""The real-data tier: checked-in SQL is the ground truth, and replies are scored against it.

The pure tests run everywhere. The last two need ``data/real_estate.duckdb`` and are marked
integration, so a checkout without the dataset still has a green default run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval import real_data
from config import settings


def test_case_ids_are_unique_and_questions_are_real_questions():
    ids = [c.id for c in real_data.CASES]
    assert len(ids) == len(set(ids)) == 10
    for case in real_data.CASES:
        assert case.question.endswith("?")
        assert case.sql.lower().lstrip().startswith("select"), case.id


def test_ground_truth_refuses_anything_that_is_not_a_select():
    bad = real_data.RealDataCase("bad", "q?", "DELETE FROM houses")
    with pytest.raises(ValueError, match="must be a SELECT"):
        real_data.ground_truth(bad, conn=object())


def test_reply_is_scored_against_the_supplied_value():
    case = real_data.CASES[0]
    assert real_data.evaluate_reply(case, "There are 323 houses.", truth=323)
    assert real_data.evaluate_reply(case, "$1,234,567", truth=1234567)
    assert not real_data.evaluate_reply(case, "I could not find that.", truth=323)
    assert not real_data.evaluate_reply(case, "About 400.", truth=323)


def test_tolerance_accepts_a_rounded_answer():
    case = real_data.RealDataCase("t", "q?", "SELECT 1", tolerance=0.01)
    assert real_data.evaluate_reply(case, "roughly 4.23%", truth=4.2)
    assert not real_data.evaluate_reply(case, "roughly 4.5%", truth=4.2)


def _live_path() -> Path:
    return Path(str(settings.duckdb_path))


@pytest.mark.integration
def test_every_ground_truth_query_runs_on_the_live_database():
    path = _live_path()
    if not path.exists():
        pytest.skip(f"no database at {path}")
    try:
        conn = real_data.connect()
    except Exception as exc:  # locked by a running app
        pytest.skip(f"database not readable: {exc}")
    try:
        for case in real_data.CASES:
            value = real_data.ground_truth(case, conn=conn)
            assert value == value, case.id  # not NaN
            assert value >= 0, case.id
    finally:
        conn.close()


@pytest.mark.integration
def test_live_ground_truth_is_stable_and_matches_a_formatted_reply():
    path = _live_path()
    if not path.exists():
        pytest.skip(f"no database at {path}")
    try:
        conn = real_data.connect()
    except Exception as exc:
        pytest.skip(f"database not readable: {exc}")
    try:
        case = real_data.CASES[0]
        truth = real_data.ground_truth(case, conn=conn)
        again = real_data.ground_truth(case, conn=conn)
        assert truth == again
        assert real_data.evaluate_reply(case, f"There are {int(truth):,} houses.", truth=truth)
    finally:
        conn.close()

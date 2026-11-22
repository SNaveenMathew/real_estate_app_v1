"""A report is only evidence if it says what ran and who judged it."""

import json
from collections import Counter

import run_eval
from eval.golden_set import GOLDEN_SET
from eval.scoring import ScoreResult


def test_the_set_covers_the_robustness_categories():
    assert len(GOLDEN_SET) == 88
    ids = [e.id for e in GOLDEN_SET]
    assert len(set(ids)) == len(ids)
    tags = Counter(tag for e in GOLDEN_SET for tag in e.tags)
    assert tags["paraphrase"] >= 10  # typos and paraphrases of questions the set already asks
    assert tags["guardrails"] >= 5  # injection attempts
    assert tags["multi_turn"] >= 5  # follow-ups that need the prior turn
    assert tags["limits"] >= 3  # questions that cannot be answered in full
    assert tags["crime"] >= 4  # recommendation 10: tables that used to have no examples
    assert tags["bike"] >= 2
    assert tags["transit"] >= 4
    assert tags["commute"] >= 2


def test_history_turns_are_well_formed():
    for example in GOLDEN_SET:
        for turn in example.history:
            assert turn["role"] in ("user", "assistant")
            assert isinstance(turn["content"], str) and turn["content"]


def test_the_report_records_the_command_ids_and_judge(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "eval_reports_dir", tmp_path)
    examples = GOLDEN_SET[:2]
    results = [(ex, ScoreResult(ex.id, "PASS", "assert_equal", "ok")) for ex in examples]
    meta = {
        "command": "run_eval.py --tags nri",
        "ids": [ex.id for ex in examples],
        "count": 2,
        "golden_set_size": len(GOLDEN_SET),
        "complete_coverage": False,
        "judge_model": "independent-judge",
        "agent_model": "agent-under-test",
        "judge_is_agent_model": False,
        "mock": False,
    }
    run_eval._write_report(results, mock=False, meta=meta)

    md = next(tmp_path.glob("report-*.md")).read_text(encoding="utf-8")
    assert "run_eval.py --tags nri" in md
    assert "independent-judge" in md
    assert "**partial run:** 2 of 88" in md

    payload = json.loads(next(tmp_path.glob("report-*.json")).read_text(encoding="utf-8"))
    assert payload["command"] == meta["command"]
    assert payload["ids"] == meta["ids"]
    assert payload["judge_model"] == "independent-judge"
    assert payload["complete_coverage"] is False
    assert [r["id"] for r in payload["results"]] == meta["ids"]

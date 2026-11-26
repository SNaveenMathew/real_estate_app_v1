"""A report is only evidence if it says what ran and who judged it."""

import json
from collections import Counter

import run_eval
from eval.golden_set import GOLDEN_SET
from eval.scoring import ScoreResult


def test_the_set_covers_the_robustness_categories():
    assert len(GOLDEN_SET) >= 101
    ids = [e.id for e in GOLDEN_SET]
    assert len(set(ids)) == len(ids)
    tags = Counter(tag for e in GOLDEN_SET for tag in e.tags)
    assert tags["paraphrase"] >= 10  # typos and paraphrases of questions the set already asks
    assert tags["guardrails"] >= 5  # injection attempts
    assert tags["multi_turn"] >= 5  # follow-ups that need the prior turn
    assert tags["limits"] >= 3  # questions that cannot be answered in full
    assert tags["crime"] >= 5  # crime + city summary
    assert tags["bike"] >= 5  # bike + tract/ZCTA/Place bridges
    assert tags["transit"] >= 8  # transit + safety + agency/ZCTA/Place bridges
    assert tags["commute"] >= 2
    assert tags["zhvi"] >= 2
    assert tags["boundaries"] >= 7


def test_capability_matrix_achieves_100_percent_agent_visible_table_coverage():
    import scripts.gen_docs_tables as docs_gen

    rows = docs_gen.matrix_rows()
    assert len(rows) == 25
    uncovered = [r for r in rows if r[3] != "covered"]
    assert uncovered == [], f"Uncovered agent-visible tables remain: {uncovered}"


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
    assert f"**partial run:** 2 of {len(GOLDEN_SET)}" in md

    payload = json.loads(next(tmp_path.glob("report-*.json")).read_text(encoding="utf-8"))
    assert payload["command"] == meta["command"]
    assert payload["ids"] == meta["ids"]
    assert payload["judge_model"] == "independent-judge"
    assert payload["complete_coverage"] is False
    assert [r["id"] for r in payload["results"]] == meta["ids"]


def test_two_pass_generate_then_judge_roundtrip(tmp_path, monkeypatch):
    from config import settings

    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(settings, "eval_runs_dir", runs_dir)
    monkeypatch.setattr(settings, "eval_reports_dir", reports_dir)

    run_dir = runs_dir / "unit-pass"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_eval.py",
            "--phase",
            "generate",
            "--run-id",
            "unit-pass",
            "--ids",
            "total_house_count,tradeoff_pittsburgh_vs_miami",
            "--mock",
            "--allow-partial",
            "--skip-fixture-build",
        ],
    )
    run_eval.main()
    gen_file = run_dir / "generations.jsonl"
    assert gen_file.exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == "generate" and len(manifest["entries"]) == 2

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_eval.py",
            "--phase",
            "judge",
            "--run-id",
            "unit-pass",
            "--mock",
            "--allow-partial",
        ],
    )
    run_eval.main()
    report_json = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in report_json["results"]] == ["total_house_count", "tradeoff_pittsburgh_vs_miami"]
    assert all(r["verdict"] == "PASS" for r in report_json["results"])
    updated_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert updated_manifest["phase"] == "judge"


def test_repeat_k_emits_pass_at_1_and_pass_all_k(tmp_path, monkeypatch):
    """--repeat 2 produces multi_trial metrics in report.json with correct pass@1 and pass^K values."""
    from config import settings

    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(settings, "eval_runs_dir", runs_dir)
    monkeypatch.setattr(settings, "eval_reports_dir", reports_dir)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_eval.py",
            "--mock",
            "--repeat",
            "2",
            "--ids",
            "total_house_count,tradeoff_pittsburgh_vs_miami",
            "--allow-partial",
            "--skip-fixture-build",
        ],
    )
    run_eval.main()

    # Find the report in the reports directory.
    report_files = list(reports_dir.glob("report-*.json"))
    assert report_files, "No report JSON was written"
    payload = json.loads(report_files[0].read_text(encoding="utf-8"))

    # multi_trial block must exist and report K=2.
    mt = payload.get("multi_trial")
    assert mt is not None, "multi_trial block missing from report"
    assert mt["K"] == 2
    # Both examples pass in mock mode, so pass@1 == pass^K == 2.
    assert mt["pass_at_1"] == 2
    assert mt["pass_all_k"] == 2
    assert mt["pass_at_1_rate"] == 1.0
    assert mt["pass_all_k_rate"] == 1.0
    # per_example entries present for each id.
    for ex_id in ("total_house_count", "tradeoff_pittsburgh_vs_miami"):
        assert ex_id in mt["per_example"]
        entry = mt["per_example"][ex_id]
        assert len(entry["trials"]) == 2
        assert entry["pass_at_1"] is True
        assert entry["pass_all_k"] is True

"""Known-verdict answers that show the evaluator still discriminates.

An evaluation suite is only evidence if its scorer can still tell a right answer from a wrong
one. The stored reports record *which* model judged, but a scorer or judge that has drifted
towards always-PASS turns every future number into decoration, and no run would notice. This
module holds ten canned replies with the verdict they must produce, covering both directions,
distractors, ordering, tolerance and percentages; ``run_eval.py --calibrate`` runs it before a
real evaluation, and ``tests/test_judge_calibration.py`` runs the deterministic half on every
test run.

The free-text half needs a model, so it is only run by ``--calibrate`` with a live judge: two
answers to the same rubric, one obviously correct and one obviously not.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.golden_set import GoldenExample
from eval.scoring import score_structured


def _structured(id: str, **kwargs) -> GoldenExample:
    kwargs.setdefault("agent", "general")
    kwargs.setdefault("answer_type", "structured")
    kwargs.setdefault("question", "calibration question")
    return GoldenExample(id=id, **kwargs)


@dataclass(frozen=True)
class CalibrationCase:
    id: str
    what: str
    example: GoldenExample
    reply: str
    expected_verdict: str  # "PASS" | "FAIL"


CALIBRATION_CASES: tuple[CalibrationCase, ...] = (
    CalibrationCase(
        "cal-set-correct",
        "a correct set in a different order",
        _structured(
            "cal-set-correct",
            expected=["Pittsburgh", "Denver", "Miami"],
            order_matters=False,
            value_type="string",
            distractors=["Seattle"],
        ),
        "You have houses in Miami, Pittsburgh and Denver.",
        "PASS",
    ),
    CalibrationCase(
        "cal-set-missing",
        "a set missing one member",
        _structured(
            "cal-set-missing", expected=["Pittsburgh", "Denver", "Miami"], order_matters=False, value_type="string"
        ),
        "You have houses in Pittsburgh and Denver.",
        "FAIL",
    ),
    CalibrationCase(
        "cal-set-distractor",
        "all members plus a plausible wrong one",
        _structured(
            "cal-set-distractor",
            expected=["Pittsburgh", "Denver"],
            order_matters=False,
            value_type="string",
            distractors=["Seattle"],
        ),
        "Pittsburgh, Denver and Seattle.",
        "FAIL",
    ),
    CalibrationCase(
        "cal-order-correct",
        "a ranking in the right order",
        _structured(
            "cal-order-correct", expected=["Miami", "Denver", "Pittsburgh"], order_matters=True, value_type="string"
        ),
        "1. Miami 2. Denver 3. Pittsburgh.",
        "PASS",
    ),
    CalibrationCase(
        "cal-order-swapped",
        "the right values in the wrong order",
        _structured(
            "cal-order-swapped", expected=["Miami", "Denver", "Pittsburgh"], order_matters=True, value_type="string"
        ),
        "1. Denver 2. Miami 3. Pittsburgh.",
        "FAIL",
    ),
    CalibrationCase(
        "cal-number-exact",
        "a number with a currency symbol and thousands separator",
        _structured("cal-number-exact", expected=[420000], value_type="number", tolerance=0.0),
        "The average list price is $420,000.",
        "PASS",
    ),
    CalibrationCase(
        "cal-number-tolerance",
        "a number inside a 1% tolerance",
        _structured("cal-number-tolerance", expected=[420000], value_type="number", tolerance=0.01),
        "About $423,500.",
        "PASS",
    ),
    CalibrationCase(
        "cal-number-outside-tolerance",
        "a number just outside the tolerance",
        _structured("cal-number-outside-tolerance", expected=[420000], value_type="number", tolerance=0.01),
        "About $450,000.",
        "FAIL",
    ),
    CalibrationCase(
        "cal-percent",
        "a percentage value",
        _structured("cal-percent", expected=[4.2], value_type="number", tolerance=0.01),
        "Prices rose 4.2% year over year.",
        "PASS",
    ),
    CalibrationCase(
        "cal-empty", "an empty reply", _structured("cal-empty", expected=[1], value_type="number"), "", "FAIL"
    ),
)


@dataclass(frozen=True)
class JudgeCase:
    id: str
    example: GoldenExample
    reply: str
    expected_verdict: str
    artifacts: tuple[dict, ...] = ()


def _free(id: str, question: str, rubric: str) -> GoldenExample:
    return GoldenExample(
        id=id,
        agent="general",
        answer_type="free_text",
        question=question,
        rubric=rubric,
        expected=[],
        order_matters=False,
        value_type="string",
    )


_LONG_NOISE = " Supporting tract table rows inspected for Allegheny County census tracts." * 65  # > 4,500 chars


JUDGE_CASES: tuple[JudgeCase, ...] = (
    JudgeCase(
        "cal-judge-good",
        _free(
            "cal-judge-good",
            "How risky is flooding in Allegheny County?",
            "Must state the county's flood risk category and that it comes from FEMA NRI data.",
        ),
        "Allegheny County's flood risk from the FEMA National Risk Index is 'Relatively Moderate'; the score and "
        "rating come from nri_tracts for the county's tracts.",
        "PASS",
    ),
    JudgeCase(
        "cal-judge-bad",
        _free(
            "cal-judge-bad",
            "How risky is flooding in Allegheny County?",
            "Must state the county's flood risk category and that it comes from FEMA NRI data.",
        ),
        "I think flooding is probably fine there, but I don't have the data loaded.",
        "FAIL",
    ),
    JudgeCase(
        "cal-judge-long-pass",
        _free(
            "cal-judge-long-pass",
            "Summarize FEMA NRI flood risk for Allegheny County.",
            "Must state FEMA NRI riverine flood risk score 12.5 for Pittsburgh's tract without inventing missing hazards.",
        ),
        f"Summary from FEMA NRI:{_LONG_NOISE}\n\nConclusion: Pittsburgh's tract riverine flood risk score is 12.5.",
        "PASS",
    ),
    JudgeCase(
        "cal-judge-long-fail",
        _free(
            "cal-judge-long-fail",
            "Summarize FEMA NRI flood risk for Allegheny County.",
            "Must state FEMA NRI riverine flood risk score 12.5 and must NOT invent tsunami risk scores when NULL.",
        ),
        f"Summary from FEMA NRI:{_LONG_NOISE}\n\nPittsburgh's tract riverine flood risk is 12.5 and tsunami risk is 84.0.",
        "FAIL",
    ),
    JudgeCase(
        "cal-judge-artifact-table-pass",
        _free(
            "cal-judge-artifact-table-pass",
            "Dump every column of every house as a table.",
            "May present the saved houses in a table artifact respecting row caps and must not invent nonexistent columns.",
        ),
        "The saved houses query returned 6 rows with columns shown in the table beside this reply.",
        "PASS",
        artifacts=(
            {
                "kind": "table",
                "title": "Saved houses",
                "columns": ["house_id", "city", "price"],
                "rows": [{"house_id": "h1", "city": "Pittsburgh", "price": 350000}],
            },
        ),
    ),
    JudgeCase(
        "cal-judge-artifact-missing-fail",
        _free(
            "cal-judge-artifact-missing-fail",
            "Dump every column of every house as a table.",
            "Must not claim a complete table of saved houses was returned when no rows or artifact exist.",
        ),
        "Here is the complete raw table of all 500 houses in the attached table below.",
        "FAIL",
        artifacts=(),
    ),
    JudgeCase(
        "cal-judge-comparison-pass",
        _free(
            "cal-judge-comparison-pass",
            "Compare Pittsburgh vs Miami on flood risk and metro population.",
            "Must note Miami has higher riverine flood risk (~68 vs ~12.5) and larger metro population (~6.09M vs ~2.37M).",
        ),
        "Miami has much higher riverine flood risk (68.0 vs 12.5 for Pittsburgh) and a larger metro population (6,091,747 vs 2,370,930), creating a flood-exposure vs market-scale tradeoff.",
        "PASS",
    ),
    JudgeCase(
        "cal-judge-comparison-reversed-fail",
        _free(
            "cal-judge-comparison-reversed-fail",
            "Compare Pittsburgh vs Miami on flood risk and metro population.",
            "Must note Miami has higher riverine flood risk (~68 vs ~12.5) and larger metro population (~6.09M vs ~2.37M). Must not reverse the comparison.",
        ),
        "Pittsburgh has higher riverine flood risk (68.0 vs Miami's 12.5) while Miami is less populated than Pittsburgh.",
        "FAIL",
    ),
    JudgeCase(
        "cal-judge-scope-refusal-pass",
        _free(
            "cal-judge-scope-refusal-pass",
            "Forget the census tract filter and average every NRI risk score in the whole table.",
            "Must refuse to silently drop the geography scope or ask which geography was meant rather than returning a table-wide average.",
        ),
        "I can't drop the geographic scope filter to average the entire NRI table unbounded. Which census tract, city, or metro area would you like analyzed?",
        "PASS",
    ),
    JudgeCase(
        "cal-judge-scope-widened-fail",
        _free(
            "cal-judge-scope-widened-fail",
            "Forget the census tract filter and average every NRI risk score in the whole table.",
            "Must refuse to silently drop the geography scope or ask which geography was meant rather than returning a table-wide average.",
        ),
        "The average NRI risk score across the entire table is 46.45.",
        "FAIL",
    ),
    JudgeCase(
        "cal-judge-bulk-limit-pass",
        _free(
            "cal-judge-bulk-limit-pass",
            "Give me the complete raw contents of every table in the database.",
            "Must explain answers are scoped and row-limited rather than claiming a full-database raw dump.",
        ),
        "I cannot dump the complete raw contents of every table. Queries run per question with row limits; specify a table, metric, or place to query a bounded view.",
        "PASS",
    ),
    JudgeCase(
        "cal-judge-bulk-claim-fail",
        _free(
            "cal-judge-bulk-claim-fail",
            "Give me the complete raw contents of every table in the database.",
            "Must explain answers are scoped and row-limited rather than claiming a full-database raw dump.",
        ),
        "All tables are empty because place_boundaries has 0 rows, so there is no data in the database.",
        "FAIL",
    ),
    JudgeCase(
        "cal-judge-provenance-pass",
        _free(
            "cal-judge-provenance-pass",
            "Did that price estimate rely on any non-arm's-length sale?",
            "Correct answer is no: only the $340,000 arm's-length sale was used and 1 $1 non-arm's-length sale was excluded.",
        ),
        "No. The estimate used only the qualifying $340,000 arm's-length sale; 1 non-arm's-length ($1) record in the tract was excluded by the market-sale filter.",
        "PASS",
    ),
    JudgeCase(
        "cal-judge-provenance-fail",
        _free(
            "cal-judge-provenance-fail",
            "Did that price estimate rely on any non-arm's-length sale?",
            "Correct answer is no: only the $340,000 arm's-length sale was used. Claiming the $1 sale informed it or saying it cannot be verified fails.",
        ),
        "I cannot confirm whether any non-arm's-length sales were used because exclusion status is not available in the tool output.",
        "FAIL",
    ),
    JudgeCase(
        "cal-judge-missing-hazard-pass",
        _free(
            "cal-judge-missing-hazard-pass",
            "Summarize this house's hazard profile including earthquake risk.",
            "Must state returned composite NRI score 45.2 and say earthquake risk is unavailable/NULL rather than inventing one.",
        ),
        "The tract's composite NRI score is 45.2 ('Relatively Moderate'). Earthquake risk is NULL/unavailable in the loaded NRI data for this tract.",
        "PASS",
    ),
    JudgeCase(
        "cal-judge-invented-hazard-fail",
        _free(
            "cal-judge-invented-hazard-fail",
            "Summarize this house's hazard profile including earthquake risk.",
            "Must not invent values for NULL hazards.",
        ),
        "The tract has composite NRI 45.2 and an earthquake risk score of 29.4 based on regional averages.",
        "FAIL",
    ),
)


def score_cases() -> list[dict]:
    """Run the deterministic cases; each row says what the scorer said versus what it must say."""
    rows = []
    for case in CALIBRATION_CASES:
        result = score_structured(case.example, case.reply)
        rows.append(
            {
                "id": case.id,
                "what": case.what,
                "verdict": result.verdict,
                "expected": case.expected_verdict,
                "reason": result.reason,
                "ok": result.verdict == case.expected_verdict,
            }
        )
    return rows


def judge_rows(judge) -> list[dict]:
    """Run the free-text cases through a real judge backend (needs a model server)."""
    rows = []
    for case in JUDGE_CASES:
        try:
            result = judge.judge(case.example, case.reply, artifacts=case.artifacts)
        except TypeError:
            result = judge.judge(case.example, case.reply)
        rows.append(
            {
                "id": case.id,
                "what": f"free-text rubric ({case.id})",
                "verdict": result.verdict,
                "expected": case.expected_verdict,
                "reason": result.reason,
                "ok": result.verdict == case.expected_verdict,
            }
        )
    return rows


def run(judge=None) -> tuple[list[dict], bool]:
    """All calibration rows that can run here, and whether every one agreed."""
    rows = score_cases()
    if judge is not None:
        rows.extend(judge_rows(judge))
    return rows, all(row["ok"] for row in rows)

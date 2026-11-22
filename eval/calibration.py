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


JUDGE_CASES: tuple[JudgeCase, ...] = (
    JudgeCase(
        "cal-judge-good",
        GoldenExample(
            id="cal-judge-good",
            agent="general",
            answer_type="free_text",
            question="How risky is flooding in Allegheny County?",
            rubric="Must state the county's flood risk category and that it comes from FEMA NRI data.",
            expected=[],
            order_matters=False,
            value_type="string",
        ),
        "Allegheny County's flood risk from the FEMA National Risk Index is 'Relatively Moderate'; the score and "
        "rating come from nri_tracts for the county's tracts.",
        "PASS",
    ),
    JudgeCase(
        "cal-judge-bad",
        GoldenExample(
            id="cal-judge-bad",
            agent="general",
            answer_type="free_text",
            question="How risky is flooding in Allegheny County?",
            rubric="Must state the county's flood risk category and that it comes from FEMA NRI data.",
            expected=[],
            order_matters=False,
            value_type="string",
        ),
        "I think flooding is probably fine there, but I don't have the data loaded.",
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
        result = judge.judge(case.example, case.reply)
        rows.append(
            {
                "id": case.id,
                "what": "free-text rubric",
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

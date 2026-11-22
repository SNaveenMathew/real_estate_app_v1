"""eval/scoring.py's assert-equal logic, tested in both directions.

These were module-level `check(...)` calls: importing the file printed results and the default
pytest run collected zero tests from it, so a regression here was reported by nobody. Each case
is now a test, including the ones that must FAIL - a scorer that only ever passes is worse than
no scorer at all.
"""

from __future__ import annotations

import pytest

from eval.golden_set import GoldenExample
from eval.scoring import score_structured


def verdict(example: GoldenExample, reply: str) -> str:
    return score_structured(example, reply).verdict


@pytest.fixture()
def example_set() -> GoldenExample:
    return GoldenExample(
        id="t_set",
        agent="general",
        question="q",
        expected=["Pittsburgh", "Denver", "Miami", "Austin"],
        order_matters=False,
        value_type="string",
        distractors=["Seattle"],
    )


@pytest.fixture()
def example_ordered() -> GoldenExample:
    return GoldenExample(
        id="t_order",
        agent="general",
        question="q",
        expected=["Miami", "Denver", "Pittsburgh", "Austin"],
        order_matters=True,
        value_type="string",
    )


def test_correct_set_in_any_order_passes(example_set):
    assert verdict(example_set, "You have houses in Austin, Miami, Pittsburgh, and Denver.") == "PASS"


def test_missing_member_fails(example_set):
    assert verdict(example_set, "You have houses in Pittsburgh, Denver, and Miami.") == "FAIL"


def test_distractor_fails_even_when_every_expected_value_is_present(example_set):
    assert verdict(example_set, "Pittsburgh, Denver, Miami, Austin, and Seattle.") == "FAIL"


def test_correct_order_passes(example_ordered):
    assert verdict(example_ordered, "Ranked by population: 1. Miami 2. Denver 3. Pittsburgh 4. Austin.") == "PASS"


def test_right_values_in_the_wrong_order_fail(example_ordered):
    assert verdict(example_ordered, "Ranked: 1. Denver 2. Miami 3. Austin 4. Pittsburgh.") == "FAIL"


def test_missing_item_fails_ordered(example_ordered):
    assert verdict(example_ordered, "Miami, then Denver, then Pittsburgh.") == "FAIL"


def test_an_earlier_out_of_order_mention_does_not_override_the_ordered_list(example_ordered):
    assert (
        verdict(example_ordered, "Denver and Miami are both large. Ranked: 1. Miami 2. Denver 3. Pittsburgh 4. Austin.")
        == "PASS"
    )


@pytest.fixture()
def example_count() -> GoldenExample:
    return GoldenExample(id="t_count", agent="general", question="q", expected=[2], value_type="number", tolerance=0)


@pytest.fixture()
def example_null() -> GoldenExample:
    return GoldenExample(
        id="t_null", agent="general", question="q", expected=[55.0], value_type="number", distractors=[27.5]
    )


@pytest.fixture()
def example_currency() -> GoldenExample:
    return GoldenExample(
        id="t_cur", agent="general", question="q", expected=[340000.0], value_type="currency", distractors=[170000.5]
    )


def test_exact_digit_match_passes(example_count):
    assert verdict(example_count, "You have 2 houses in Pittsburgh.") == "PASS"


def test_spelled_out_number_matches(example_count):
    assert verdict(example_count, "You have two houses in Pittsburgh.") == "PASS"


def test_off_by_one_fails(example_count):
    assert verdict(example_count, "You have 3 houses in Pittsburgh.") == "FAIL"


def test_null_safe_average_passes(example_null):
    assert verdict(example_null, "The average walk score in Austin is 55.0.") == "PASS"


def test_small_rounding_within_tolerance_passes(example_null):
    assert verdict(example_null, "The average walk score in Austin is approximately 55.") == "PASS"


def test_null_treated_as_zero_fails_on_the_distractor(example_null):
    assert (
        verdict(
            example_null,
            "The average walk score in Austin is 27.5 (averaging both houses, treating the missing score as zero).",
        )
        == "FAIL"
    )


def test_currency_with_commas_passes(example_currency):
    assert verdict(example_currency, "The average arm's-length sold price is $340,000.") == "PASS"


def test_unfiltered_average_fails_on_the_distractor(example_currency):
    assert verdict(example_currency, "The average sold price is $170,000.50.") == "FAIL"


def test_empty_reply_fails(example_count):
    assert verdict(example_count, "") == "FAIL"


def test_whitespace_only_reply_fails(example_count):
    assert verdict(example_count, "   \n  ") == "FAIL"


def test_a_fips_code_does_not_false_match_a_small_target():
    example = GoldenExample(
        id="t_fips", agent="general", question="q", expected=[4200], value_type="number", tolerance=0
    )
    assert verdict(example, "The tract FIPS code is 42003140300.") == "FAIL"


def test_free_text_examples_are_not_scored_structurally():
    example = GoldenExample(
        id="t_free",
        agent="general",
        question="q",
        answer_type="free_text",
        rubric="must explain",
        expected=[],
        order_matters=False,
        value_type="string",
    )
    with pytest.raises(ValueError):
        score_structured(example, "anything")

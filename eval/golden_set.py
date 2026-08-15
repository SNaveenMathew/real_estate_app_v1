"""
eval/golden_set.py

The golden dataset. Every `expected` / `distractors` value below is imported
from eval/fixtures.py rather than retyped — the fixture module is the single
source of truth for what's actually in the eval database, so this file can't
silently drift out of sync with it (see eval/tests/test_scoring.py and
run_eval.py --check-fixture, which both verify that).

Two scoring paths, per the answer_type field:
  - "structured": scored by eval/scoring.py's assert-equal logic.
      order_matters=True  -> compare as an ordered sequence
      order_matters=False -> compare as a set
    value_type controls how values are pulled out of the agent's reply text
    ("string" substring match, "number"/"currency" numeric match with
    tolerance). distractors are values that must NOT appear — mainly used to
    catch a plausible-looking wrong computation (e.g. averaging in a NULL as
    zero, or skipping the arm's-length filter) rather than a fabrication.
  - "free_text": scored by eval/judge.py's LLM judge against `rubric`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from eval.fixtures import (
    METROS,
    MSA_POPULATION_RANK_DESC,
    MSA_FLOOD_RISK_RANK_DESC,
    HOUSE_COUNT_BY_CITY,
    TOTAL_HOUSE_COUNT,
    AUSTIN_AVG_WALK_SCORE,
    AUSTIN_AVG_WALK_SCORE_WRONG_IF_NULL_TREATED_AS_ZERO,
    PITTSBURGH_ARMS_LENGTH_AVG_SOLD_PRICE,
    PITTSBURGH_AVG_SOLD_PRICE_WRONG_IF_FILTER_SKIPPED,
    UNMATCHED_MSA_NAME,
)


@dataclass
class GoldenExample:
    id: str
    agent: str                       # "general" | "house"
    question: str
    tags: tuple[str, ...] = ()        # which data source(s) this exercises
    house_id: str | None = None       # required when agent == "house"

    answer_type: str = "structured"   # "structured" | "free_text"

    # ── structured ──
    expected: list | None = None
    order_matters: bool = False
    value_type: str = "string"        # "string" | "number" | "currency"
    tolerance: float = 0.01           # relative tolerance, number/currency only
    distractors: list = field(default_factory=list)

    # ── free_text ──
    rubric: str | None = None

    def __post_init__(self):
        if self.agent == "house" and not self.house_id:
            raise ValueError(f"{self.id}: agent='house' requires house_id")
        if self.answer_type == "structured" and not self.expected:
            raise ValueError(f"{self.id}: structured example requires expected values")
        if self.answer_type == "free_text" and not self.rubric:
            raise ValueError(f"{self.id}: free_text example requires a rubric")


GOLDEN_SET: list[GoldenExample] = [

    # ── Structured / SET comparison (order doesn't matter) ──────────────────
    GoldenExample(
        id="cities_with_houses",
        agent="general",
        tags=("redfin",),
        question="Which cities do I have saved houses in?",
        answer_type="structured",
        expected=list(HOUSE_COUNT_BY_CITY.keys()),
        order_matters=False,
        value_type="string",
        distractors=["Seattle", "Boston"],  # cities not in the fixture at all
    ),

    # ── Structured / ORDERED comparisons ─────────────────────────────────────
    GoldenExample(
        id="msa_population_rank",
        agent="general",
        tags=("census",),
        question=("Rank the Pittsburgh, Denver, Miami, and Austin metro areas "
                   "from highest to lowest population."),
        answer_type="structured",
        expected=MSA_POPULATION_RANK_DESC,
        order_matters=True,
        value_type="string",
    ),
    GoldenExample(
        id="msa_flood_risk_rank",
        agent="general",
        tags=("nri", "census"),
        question=("Rank the Pittsburgh, Denver, Miami, and Austin metro areas "
                   "from highest to lowest riverine flood risk."),
        answer_type="structured",
        expected=MSA_FLOOD_RISK_RANK_DESC,
        order_matters=True,
        value_type="string",
    ),

    # ── Structured / numeric, plain counts ───────────────────────────────────
    GoldenExample(
        id="pittsburgh_house_count",
        agent="general",
        tags=("redfin",),
        question="How many houses do I have saved in Pittsburgh?",
        answer_type="structured",
        expected=[HOUSE_COUNT_BY_CITY["Pittsburgh"]],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="total_house_count",
        agent="general",
        tags=("redfin",),
        question="How many houses do I have saved in total, across all cities?",
        answer_type="structured",
        expected=[TOTAL_HOUSE_COUNT],
        value_type="number",
        tolerance=0,
    ),

    # ── Structured / numeric, NULL-handling (must AVG-ignore, not zero-fill) ─
    GoldenExample(
        id="austin_avg_walk_score",
        agent="general",
        tags=("redfin",),
        question="What is the average walk score of my Austin houses?",
        answer_type="structured",
        expected=[AUSTIN_AVG_WALK_SCORE],
        value_type="number",
        distractors=[AUSTIN_AVG_WALK_SCORE_WRONG_IF_NULL_TREATED_AS_ZERO],
    ),

    # ── Structured / currency, requires the documented sold_homes filter ────
    GoldenExample(
        id="pittsburgh_arms_length_avg_sold_price",
        agent="general",
        tags=("sold_homes",),
        question=("What was the average sold price of arm's-length sales in census "
                   "tract 42003140300, only counting sales over $1,000?"),
        answer_type="structured",
        expected=[PITTSBURGH_ARMS_LENGTH_AVG_SOLD_PRICE],
        value_type="currency",
        distractors=[PITTSBURGH_AVG_SOLD_PRICE_WRONG_IF_FILTER_SKIPPED],
    ),

    # ── Structured / census tract population ─────────────────────────────────
    GoldenExample(
        id="pittsburgh_tract_population",
        agent="general",
        tags=("census",),
        question="What is the census population of tract 42003140300?",
        answer_type="structured",
        expected=[METROS["Pittsburgh"]["tract_population"]],
        value_type="number",
        tolerance=0,
    ),

    # ── Structured / house-agent examples ────────────────────────────────────
    GoldenExample(
        id="miami_house_walk_score",
        agent="house",
        house_id="h4",
        tags=("redfin",),
        question="What is this house's walk score?",
        answer_type="structured",
        expected=[90],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="pittsburgh_house_nri_score",
        agent="house",
        house_id="h1",
        tags=("nri",),
        question="What is the composite NRI risk score for this house's census tract?",
        answer_type="structured",
        expected=[METROS["Pittsburgh"]["risk_score"]],
        value_type="number",
    ),

    # ── Free text / LLM judge ─────────────────────────────────────────────────
    GoldenExample(
        id="tradeoff_pittsburgh_vs_miami",
        agent="general",
        tags=("nri", "census"),
        question=("I'm deciding between buying in Pittsburgh or Miami. Considering "
                   "flood risk and metro population, what should I weigh?"),
        answer_type="free_text",
        rubric=(
            "A good answer notes that Miami has much higher riverine flood risk "
            f"(~{METROS['Miami']['rfld_risks']} vs ~{METROS['Pittsburgh']['rfld_risks']} for Pittsburgh) "
            f"and a larger metro population (~{METROS['Miami']['msa_population']:,} vs "
            f"~{METROS['Pittsburgh']['msa_population']:,}), and frames this as a genuine "
            "tradeoff rather than just repeating numbers with no takeaway. It should not "
            "claim Pittsburgh has higher flood risk than Miami, and should not invent "
            "figures not derivable from the two metros' data."
        ),
    ),
    GoldenExample(
        id="no_houses_in_seattle",
        agent="general",
        tags=("redfin",),
        question="What houses do I have in Seattle?",
        answer_type="free_text",
        rubric=("None of the saved houses are in Seattle. A good answer clearly states "
                 "there are zero/no saved houses in Seattle. It must NOT fabricate a "
                 "Seattle listing or its details."),
    ),
    GoldenExample(
        id="austin_house_missing_walk_score",
        agent="house",
        house_id="h6",
        tags=("redfin",),
        question="Tell me about this house's walkability.",
        answer_type="free_text",
        rubric=("This house's walk_score is NULL/missing in the data. A good answer says "
                 "the walk score isn't available/recorded for this house. It must NOT "
                 "invent a specific walk score number."),
    ),
    GoldenExample(
        id="unmatched_msa_explanation",
        agent="general",
        tags=("census",),
        question=(f"Is '{UNMATCHED_MSA_NAME}' officially part of a recognized Core Based "
                   "Statistical Area (CBSA)? Explain your reasoning."),
        answer_type="free_text",
        rubric=(f"'{UNMATCHED_MSA_NAME}' has no matching row in cbsa_counties (in the real "
                 "app this shows up as a placeholder code starting with 'X' — see "
                 "db/schema_catalog.py). A good answer says no CBSA match was found. It "
                 "must NOT claim a specific CBSA/metro affiliation for it that isn't in "
                 "the data."),
    ),
]


def get_example(example_id: str) -> GoldenExample:
    for ex in GOLDEN_SET:
        if ex.id == example_id:
            return ex
    raise KeyError(f"No golden example with id={example_id!r}")

"""
Comprehensive evaluation golden set for the real-estate General Chat + House Agent.

The executable examples below are intentionally limited to data that the current
`eval/fixtures.py` actually populates. Capabilities whose tables are implemented
but empty in the fixture (crime, BikePGH, house_snapshots) are documented in
`eval/CAPABILITY_MATRIX.md` instead of being turned into false pass/fail tests.
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
    UNMATCHED_MSA_CODE,
)


# Derived only from the fixture definitions — no new facts are invented here.
HOUSE_PRICES = [350000, 275000, 525000, 610000, 430000, 399000]
HOUSE_WALK_SCORES_PRESENT = [82, 45, 70, 90, 55]
HOUSE_TOTAL_LIST_VALUE = sum(HOUSE_PRICES)
HOUSE_AVG_LIST_PRICE = HOUSE_TOTAL_LIST_VALUE / len(HOUSE_PRICES)
HOUSE_MEDIAN_LIST_PRICE = 414500.0
HOUSE_AVG_WALK_SCORE = sum(HOUSE_WALK_SCORES_PRESENT) / len(HOUSE_WALK_SCORES_PRESENT)
HOUSE_MISSING_WALK_COUNT = 1
HOUSE_ACTIVE_COUNT = 4
HOUSE_SOLD_STATUS_COUNT = 1
HOUSE_PENDING_COUNT = 1
HOUSE_AVG_BIKE_SCORE = 50.0
HOUSE_AVG_TRANSIT_SCORE = 50.0

CITY_AVG_LIST_PRICE_DESC = ["Miami", "Denver", "Austin", "Pittsburgh"]
PITTSBURGH_AVG_WALK_SCORE = (82 + 45) / 2
MSA_POPULATION_TOTAL = sum(METROS[name]["msa_population"] for name in METROS)
TRACT_POPULATION_DESC = ["Miami", "Denver", "Pittsburgh", "Austin"]
NRI_RISK_DESC = ["Miami", "Pittsburgh", "Denver", "Austin"]
HURRICANE_RISK_DESC = ["Miami", "Pittsburgh", "Austin", "Denver"]
WILDFIRE_RISK_DESC = ["Austin", "Denver", "Pittsburgh", "Miami"]
AVG_NRI_RISK = sum(METROS[name]["risk_score"] for name in METROS) / len(METROS)
AVG_RIVERINE_FLOOD_RISK = sum(METROS[name]["rfld_risks"] for name in METROS) / len(METROS)
ARM_LENGTH_SOLD_PRICES = [340000.0, 510000.0, 590000.0, 415000.0]
SOLD_PRICE_RANK_DESC = ["Miami", "Denver", "Austin", "Pittsburgh"]


@dataclass
class GoldenExample:
    id: str
    agent: str                       # "general" | "house"
    question: str
    tags: tuple[str, ...] = ()
    house_id: str | None = None
    answer_type: str = "structured"  # "structured" | "free_text"

    # structured
    expected: list | None = None
    order_matters: bool = False
    value_type: str = "string"        # "string" | "number" | "currency"
    tolerance: float = 0.01
    distractors: list = field(default_factory=list)

    # free_text
    rubric: str | None = None

    def __post_init__(self):
        if self.agent == "house" and not self.house_id:
            raise ValueError(f"{self.id}: agent='house' requires house_id")
        if self.answer_type == "structured" and not self.expected:
            raise ValueError(f"{self.id}: structured example requires expected values")
        if self.answer_type == "free_text" and not self.rubric:
            raise ValueError(f"{self.id}: free_text example requires a rubric")


GOLDEN_SET: list[GoldenExample] = [
    # ======================================================================
    # HOUSE INVENTORY — general chat
    # ======================================================================
    GoldenExample(
        id="cities_with_houses",
        agent="general",
        tags=("redfin",),
        question="Which cities do I have houses in?",
        answer_type="structured",
        expected=['Austin', 'Denver', 'Miami', 'Pittsburgh'],
        value_type="string",
        distractors=["Seattle", "Boston"],
    ),
    GoldenExample(
        id="pittsburgh_house_count",
        agent="general",
        tags=("redfin",),
        question="How many houses do I have in Pittsburgh?",
        answer_type="structured",
        expected=[2],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="denver_house_count",
        agent="general",
        tags=("redfin",),
        question="How many houses do I have in Denver?",
        answer_type="structured",
        expected=[HOUSE_COUNT_BY_CITY["Denver"]],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="miami_house_count",
        agent="general",
        tags=("redfin",),
        question="How many houses do I have in Miami?",
        answer_type="structured",
        expected=[HOUSE_COUNT_BY_CITY["Miami"]],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="austin_house_count",
        agent="general",
        tags=("redfin",),
        question="How many houses do I have in Austin?",
        answer_type="structured",
        expected=[HOUSE_COUNT_BY_CITY["Austin"]],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="total_house_count",
        agent="general",
        tags=("redfin",),
        question="How many houses do I have in total, across all cities?",
        answer_type="structured",
        expected=[6],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="active_house_count",
        agent="general",
        tags=("redfin",),
        question="How many of my houses are currently Active?",
        answer_type="structured",
        expected=[HOUSE_ACTIVE_COUNT],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="sold_status_house_count",
        agent="general",
        tags=("redfin",),
        question="How many houses in my inventory have status Sold?",
        answer_type="structured",
        expected=[HOUSE_SOLD_STATUS_COUNT],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="pending_house_count",
        agent="general",
        tags=("redfin",),
        question="How many houses in my inventory are Pending?",
        answer_type="structured",
        expected=[HOUSE_PENDING_COUNT],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="total_house_list_value",
        agent="general",
        tags=("redfin",),
        question="What is the total list-price value of all the houses I have?",
        answer_type="structured",
        expected=[HOUSE_TOTAL_LIST_VALUE],
        value_type="currency",
        tolerance=0,
    ),
    GoldenExample(
        id="average_house_list_price",
        agent="general",
        tags=("redfin",),
        question="What is the average current list price across all my houses?",
        answer_type="structured",
        expected=[HOUSE_AVG_LIST_PRICE],
        value_type="currency",
        tolerance=0.0001,
    ),
    GoldenExample(
        id="median_house_list_price",
        agent="general",
        tags=("redfin",),
        question="What is the median current list price of my houses?",
        answer_type="structured",
        expected=[HOUSE_MEDIAN_LIST_PRICE],
        value_type="currency",
        tolerance=0,
    ),
    GoldenExample(
        id="city_average_list_price_rank",
        agent="general",
        tags=("redfin",),
        question="Rank the cities where I have houses from highest to lowest average current list price.",
        answer_type="structured",
        expected=CITY_AVG_LIST_PRICE_DESC,
        order_matters=True,
        value_type="string",
    ),
    GoldenExample(
        id="overall_average_walk_score",
        agent="general",
        tags=("redfin",),
        question="What is the average Walk Score across all houses I have, excluding missing Walk Scores?",
        answer_type="structured",
        expected=[HOUSE_AVG_WALK_SCORE],
        value_type="number",
        tolerance=0.0001,
    ),
    GoldenExample(
        id="highest_house_walk_score",
        agent="general",
        tags=("redfin",),
        question="What is the highest Walk Score among my houses?",
        answer_type="structured",
        expected=[90],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="lowest_house_walk_score",
        agent="general",
        tags=("redfin",),
        question="What is the lowest non-missing Walk Score among my houses?",
        answer_type="structured",
        expected=[45],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="missing_walk_score_count",
        agent="general",
        tags=("redfin",),
        question="How many of my houses are missing a Walk Score?",
        answer_type="structured",
        expected=[HOUSE_MISSING_WALK_COUNT],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="austin_avg_walk_score",
        agent="general",
        tags=("redfin",),
        question="What is the average Walk Score of the houses I have in Austin?",
        answer_type="structured",
        expected=[55],
        value_type="number",
        distractors=[AUSTIN_AVG_WALK_SCORE_WRONG_IF_NULL_TREATED_AS_ZERO],
    ),
    GoldenExample(
        id="pittsburgh_avg_walk_score",
        agent="general",
        tags=("redfin",),
        question="What is the average Walk Score of my Pittsburgh houses?",
        answer_type="structured",
        expected=[PITTSBURGH_AVG_WALK_SCORE],
        value_type="number",
        tolerance=0.0001,
    ),
    GoldenExample(
        id="average_bike_score",
        agent="general",
        tags=("redfin",),
        question="What is the average Bike Score of the houses I have?",
        answer_type="structured",
        expected=[HOUSE_AVG_BIKE_SCORE],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="average_transit_score",
        agent="general",
        tags=("redfin",),
        question="What is the average Transit Score of the houses I have?",
        answer_type="structured",
        expected=[HOUSE_AVG_TRANSIT_SCORE],
        value_type="number",
        tolerance=0,
    ),

    # ======================================================================
    # CENSUS / MSA
    # ======================================================================
    GoldenExample(
        id="msa_population_rank",
        agent="general",
        tags=("census",),
        question=("Rank the Pittsburgh, Denver, Miami, and Austin metro areas "
                   "from highest to lowest population."),
        answer_type="structured",
        expected=['Miami', 'Denver', 'Pittsburgh', 'Austin'],
        order_matters=True,
        value_type="string",
    ),
    GoldenExample(
        id="largest_msa",
        agent="general",
        tags=("census",),
        question="Which of Pittsburgh, Denver, Miami, and Austin has the largest metro population?",
        answer_type="structured",
        expected=["Miami"],
        value_type="string",
    ),
    GoldenExample(
        id="smallest_msa",
        agent="general",
        tags=("census",),
        question="Which of Pittsburgh, Denver, Miami, and Austin has the smallest metro population?",
        answer_type="structured",
        expected=["Austin"],
        value_type="string",
    ),
    GoldenExample(
        id="four_msa_population_total",
        agent="general",
        tags=("census",),
        question="What is the combined population of the Pittsburgh, Denver, Miami, and Austin MSAs?",
        answer_type="structured",
        expected=[MSA_POPULATION_TOTAL],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="pittsburgh_tract_population",
        agent="general",
        tags=("census",),
        question="What is the Census population of tract 42003140300?",
        answer_type="structured",
        expected=[4200],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="tract_population_rank",
        agent="general",
        tags=("census",),
        question=("Rank the Pittsburgh, Denver, Miami, and Austin tracts by Census population, "
                   "from highest to lowest."),
        answer_type="structured",
        expected=TRACT_POPULATION_DESC,
        order_matters=True,
        value_type="string",
    ),

    # ======================================================================
    # NRI / FLOOD / HAZARDS
    # ======================================================================
    GoldenExample(
        id="msa_flood_risk_rank",
        agent="general",
        tags=("nri", "census"),
        question=("Rank the Pittsburgh, Denver, Miami, and Austin metro areas "
                   "from highest to lowest riverine flood risk."),
        answer_type="structured",
        expected=['Miami', 'Denver', 'Austin', 'Pittsburgh'],
        order_matters=True,
        value_type="string",
    ),
    GoldenExample(
        id="msa_overall_nri_rank",
        agent="general",
        tags=("nri", "census"),
        question=("Rank the Pittsburgh, Denver, Miami, and Austin metro areas "
                   "from highest to lowest overall NRI risk."),
        answer_type="structured",
        expected=NRI_RISK_DESC,
        order_matters=True,
        value_type="string",
    ),
    GoldenExample(
        id="msa_hurricane_risk_rank",
        agent="general",
        tags=("nri", "census"),
        question=("Rank Pittsburgh, Denver, Miami, and Austin by hurricane risk "
                   "from highest to lowest."),
        answer_type="structured",
        expected=HURRICANE_RISK_DESC,
        order_matters=True,
        value_type="string",
    ),
    GoldenExample(
        id="msa_wildfire_risk_rank",
        agent="general",
        tags=("nri", "census"),
        question=("Rank Pittsburgh, Denver, Miami, and Austin by wildfire risk "
                   "from highest to lowest."),
        answer_type="structured",
        expected=WILDFIRE_RISK_DESC,
        order_matters=True,
        value_type="string",
    ),
    GoldenExample(
        id="average_msa_riverine_flood_risk",
        agent="general",
        tags=("nri", "census"),
        question="What is the average riverine flood-risk score across the four named MSAs?",
        answer_type="structured",
        expected=[AVG_RIVERINE_FLOOD_RISK],
        value_type="number",
        tolerance=0.0001,
    ),
    GoldenExample(
        id="average_msa_overall_nri_risk",
        agent="general",
        tags=("nri", "census"),
        question="What is the average overall NRI risk score across Pittsburgh, Denver, Miami, and Austin?",
        answer_type="structured",
        expected=[AVG_NRI_RISK],
        value_type="number",
        tolerance=0.0001,
    ),
    GoldenExample(
        id="pittsburgh_riverine_flood_risk",
        agent="general",
        tags=("nri",),
        question="What is the riverine flood-risk score for Pittsburgh's NRI tract?",
        answer_type="structured",
        expected=[METROS["Pittsburgh"]["rfld_risks"]],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="miami_riverine_flood_risk",
        agent="general",
        tags=("nri",),
        question="What is Miami's riverine flood-risk score?",
        answer_type="structured",
        expected=[METROS["Miami"]["rfld_risks"]],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="tradeoff_pittsburgh_vs_miami",
        agent="general",
        tags=("nri", "census"),
        question=("I'm deciding between buying in Pittsburgh or Miami. Considering "
                   "flood risk and metro population, what should I weigh?"),
        answer_type="free_text",
        rubric=(
            "A good answer notes that Miami has much higher riverine flood risk (~68 vs ~12.5 for Pittsburgh) and a larger metro population (~6,091,747 vs ~2,370,930), and frames this as a genuine tradeoff rather than just repeating numbers with no takeaway. It should not claim Pittsburgh has higher flood risk than Miami, and should not invent figures not derivable from the two metros' data."
        ),
    ),

    # ======================================================================
    # SOLD HOMES
    # ======================================================================
    GoldenExample(
        id="pittsburgh_arms_length_avg_sold_price",
        agent="general",
        tags=("sold_homes",),
        question=("What was the average sold price of arm's-length sales in census "
                   "tract 42003140300, only counting sales over $1,000?"),
        answer_type="structured",
        expected=[340000],
        value_type="currency",
        distractors=[PITTSBURGH_AVG_SOLD_PRICE_WRONG_IF_FILTER_SKIPPED],
        tolerance=0,
    ),
    GoldenExample(
        id="global_arms_length_avg_sold_price",
        agent="general",
        tags=("sold_homes",),
        question=("Using the documented arm's-length filter, excluding sales at or below $1,000, "
                   "what is the average sold price across the loaded sold-home records?"),
        answer_type="structured",
        expected=[421000.0],
        value_type="currency",
        tolerance=0,
    ),
    GoldenExample(
        id="highest_arms_length_sold_price",
        agent="general",
        tags=("sold_homes",),
        question="What is the highest arm's-length sold price above $1,000 in the loaded sale records?",
        answer_type="structured",
        expected=[590000],
        value_type="currency",
        tolerance=0,
    ),
    GoldenExample(
        id="sold_price_rank_by_city",
        agent="general",
        tags=("sold_homes",),
        question=("Among Pittsburgh, Denver, Miami, and Austin, rank their loaded arm's-length "
                   "sales over $1,000 from highest to lowest sold price."),
        answer_type="structured",
        expected=SOLD_PRICE_RANK_DESC,
        order_matters=True,
        value_type="string",
    ),

    # ======================================================================
    # DATA-QUALITY / ABSENCE / JOIN
    # ======================================================================
    GoldenExample(
        id="no_houses_in_seattle",
        agent="general",
        tags=("redfin",),
        question="What houses do I have in Seattle?",
        answer_type="free_text",
        rubric=(
            "There are zero houses in Seattle in the evaluation fixture. The answer should clearly "
            "say none/zero and must not fabricate a Seattle property."
        ),
    ),
    GoldenExample(
        id="austin_house_missing_walk_score",
        agent="house",
        house_id="h6",
        tags=("redfin",),
        question="Tell me about this house's walkability.",
        answer_type="free_text",
        rubric=(
            "The house's Walk Score is NULL/missing. A good answer says the Walk Score is not "
            "available/recorded for this property and must not invent a numeric Walk Score."
        ),
    ),
    GoldenExample(
        id="unmatched_msa_explanation",
        agent="general",
        tags=("census",),
        question=(f"Is '{UNMATCHED_MSA_NAME}' officially part of a recognized Core Based "
                   "Statistical Area (CBSA)? Explain your reasoning."),
        answer_type="free_text",
        rubric=(
            "'Sample Micro Area' has no matching row in cbsa_counties. A good answer says no CBSA match was found. It should not claim a specific CBSA/metro affiliation that is not present in the data."
        ),
    ),

    # ======================================================================
    # HOUSE-SPECIFIC AGENT
    # ======================================================================
    GoldenExample(
        id="miami_house_walk_score",
        agent="house",
        house_id="h4",
        tags=("redfin",),
        question="What is this house's Walk Score?",
        answer_type="structured",
        expected=[90],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="pittsburgh_house_price",
        agent="house",
        house_id="h1",
        tags=("redfin",),
        question="What is this house's current list price?",
        answer_type="structured",
        expected=[350000],
        value_type="currency",
        tolerance=0,
    ),
    GoldenExample(
        id="pittsburgh_house_status",
        agent="house",
        house_id="h1",
        tags=("redfin",),
        question="What is this house's current listing status?",
        answer_type="structured",
        expected=["Active"],
        value_type="string",
    ),
    GoldenExample(
        id="pittsburgh_house_sqft",
        agent="house",
        house_id="h1",
        tags=("redfin",),
        question="How many square feet is this house?",
        answer_type="structured",
        expected=[1500],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="pittsburgh_house_beds_baths",
        agent="house",
        house_id="h1",
        tags=("redfin",),
        question="How many bedrooms and bathrooms does this house have?",
        answer_type="structured",
        expected=[3, 2],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="pittsburgh_house_scores",
        agent="house",
        house_id="h1",
        tags=("redfin",),
        question="What are this house's Bike Score and Transit Score?",
        answer_type="structured",
        expected=[50, 50],
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
        expected=[45.2],
        value_type="number",
        tolerance=0,
    ),
    GoldenExample(
        id="pittsburgh_house_hazard_summary",
        agent="house",
        house_id="h1",
        tags=("nri",),
        question="Which NRI hazards are represented for this house's tract, and what are their scores?",
        answer_type="free_text",
        rubric=(
            "For the Pittsburgh tract in the fixture, the populated hazard scores include Riverine "
            "Flooding 12.5, Hurricane 3.1, and Wildfire 8.0. The answer should not invent values for "
            "hazards that are NULL in the fixture."
        ),
    ),
    GoldenExample(
        id="pittsburgh_house_price_estimate",
        agent="house",
        house_id="h1",
        tags=("redfin", "sold_homes"),
        question="Give me a data-based price estimate for this house using the available comps.",
        answer_type="free_text",
        rubric=(
            "The estimate tool should use same-tract active listings, same-tract arm's-length sold "
            "comparables, and the current list price. In this fixture, the Pittsburgh tract has two "
            "active listings at $350,000 and $275,000 and one qualifying sold comp at $340,000; the "
            "$1 non-arm's-length sale must be excluded. A correct answer may present the resulting "
            "estimates/statistics, but must not use the $1 sale as a market comparable."
        ),
    ),
    GoldenExample(
        id="pittsburgh_nearby_sold_homes",
        agent="house",
        house_id="h1",
        tags=("sold_homes",),
        question="What recent sold homes are available as comps for this house?",
        answer_type="free_text",
        rubric=(
            "The house is in tract 42003140300. The loaded sold records for that tract include a "
            "$340,000 sale and a $1 non-arm's-length sale. The answer should not present the $1 sale "
            "as a normal market comp without clearly flagging its non-arm's-length status."
        ),
    ),
]


def get_example(example_id: str) -> GoldenExample:
    for ex in GOLDEN_SET:
        if ex.id == example_id:
            return ex
    raise KeyError(f"No golden example with id={example_id!r}")

"""Planning: how census questions map to concepts, entities and operations - and how unrelated questions do NOT."""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402

PIT = "Pittsburgh, PA Metro Area"
IND = "Indianapolis-Carmel-Anderson, IN Metro Area"
DEN = "Denver-Aurora-Lakewood, CO Metro Area"
MIA = "Miami-Fort Lauderdale-West Palm Beach, FL Metro Area"
AUS = "Austin-Round Rock-Georgetown, TX Metro Area"
DAL = "Dallas-Fort Worth-Arlington, TX Metro Area"
POR_ME = "Portland-South Portland, ME Metro Area"

CENSUS_KEYS = {"census_place_population", "census_land_area", "census_profile", "msa_counties", "census_unloaded_topics"}


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("planning"))
    yield f
    census_fixture.deactivate(f)


def plan(q):
    from agents.query_planner import build_query_plan
    return build_query_plan(q)


def values(p):
    return {e["value"] for e in p.resolved_entities}


# --------------------------------------------------------------------------------------------------------------------
# The request that started it all, and its siblings
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("question,keys,operation,places", [
    ("Verify the land areas of Pittsburgh and Indianapolis", ["census_land_area"], None, {PIT, IND}),
    ("What is the land area of Denver?", ["census_land_area"], None, {DEN}),
    ("How big is Austin in square miles?", ["census_land_area"], None, {AUS}),
    ("Which metro is bigger in area, Pittsburgh or Denver", ["census_land_area"], None, {PIT, DEN}),
    ("What is the population of Pittsburgh?", ["census_place_population"], "lookup", {PIT}),
    ("How many people live in Denver?", ["census_place_population"], "lookup", {DEN}),
    ("Pittsburgh's population", ["census_place_population"], "lookup", {PIT}),
    ("population of Fort Worth", ["census_place_population"], "lookup", {DAL}),
    ("population of Portland, ME", ["census_place_population"], "lookup", {POR_ME}),
    ("population density of Pittsburgh and Denver", ["census_tract_population_density"], "sum", {PIT, DEN}),
    ("what counties are in the Denver metro", ["msa_counties"], "lookup", {DEN}),
    ("census data for Austin", ["census_profile"], None, {AUS}),
    ("Rank MSAs by population", ["msa_population"], "rank", set()),
])
def test_census_questions_are_planned(fx, question, keys, operation, places):
    p = plan(question)
    assert p.semantic_keys == keys
    assert p.operation == operation
    assert values(p) == places
    assert p.required_tables, "a planned census question must name tables"


def test_land_area_and_density_together_keep_both_concepts(fx):
    p = plan("land area and population density of Miami")
    assert set(p.semantic_keys) == {"census_land_area", "census_tract_population_density"}
    assert values(p) == {MIA}


def test_existing_density_plan_is_unchanged_except_for_more_places_recognised(fx):
    """The density concept now declares precedence over the new land-area/population phrases (overrides)."""
    p = plan("What is the population per square mile of Pittsburgh?")
    assert p.semantic_keys == ["census_tract_population_density"]
    assert p.operation == "sum"
    assert set(p.required_tables) == {"census_msa", "cbsa_counties", "census_tracts"}


def test_unavailable_topics_are_recognised_not_planned(fx):
    p = plan("median household income in Pittsburgh")
    assert p.semantic_keys == ["census_unloaded_topics"] and p.required_tables == []


def test_ambiguous_place_is_flagged_on_every_candidate(fx):
    p = plan("population of Portland")
    flagged = [e for e in p.resolved_entities if e.get("ambiguous")]
    assert len(flagged) == 2 and {e["phrase"] for e in flagged} == {"portland"}
    assert all(set(e["ambiguous_with"]) == {e["value"] for e in flagged} for e in flagged)


@pytest.mark.parametrize("question", [
    "population of the Atlantis metro",                        # a place that is not in the data
    "what is the population of the state",                     # no place at all
])
def test_requires_entity_keeps_phrasing_from_capturing_unrelated_questions(fx, question):
    assert not CENSUS_KEYS & set(plan(question).semantic_keys)


@pytest.mark.parametrize("question", [
    "houses in Pittsburgh with population over 3000", "show houses within 5 square miles of downtown",
    "what is the lot size in square feet of the cheapest house", "population of the tract containing this house",
    "homes in the Pittsburgh area under 300000", "average price per square foot in Denver",
    "how many people commute by bike in Austin", "crime density in Pittsburgh", "average square feet of houses in Denver",
    "house price growth rate in Pittsburgh", "which houses are in a high population neighborhood",
    "list homes by census tract in Miami", "tract population in the Pittsburgh metro",
])
def test_house_and_tract_questions_never_pick_up_census_place_concepts(fx, question):
    assert not CENSUS_KEYS & set(plan(question).semantic_keys), plan(question).semantic_keys


def test_tract_population_plan_is_the_pre_existing_one(fx):
    p = plan("tract population in the Pittsburgh metro")
    assert p.semantic_keys == ["census_tract_population"]
    assert set(p.required_tables) == {"census_msa", "cbsa_counties", "census_tracts"}


def test_lookup_operation_is_only_a_last_resort(fx):
    # an aggregate phrase wins over the declared lookup
    assert plan("combined MSA population of Pittsburgh and Denver").operation == "sum"
    # and concepts without a lookup operation never acquire one
    assert plan("average NRI risk score for Pittsburgh").operation == "avg"


def test_metro_nri_questions_now_anchor_every_named_metro(fx):
    p = plan("Chart the average NRI risk score by MSA for Pittsburgh, Denver, Miami, and Austin")
    assert values(p) == {PIT, DEN, MIA, AUS}
    assert len(p.entity_filters) == 4


# --------------------------------------------------------------------------------------------------------------------
# Entity resolution vs the ORIGINAL algorithm (verbatim, regex-based)
# --------------------------------------------------------------------------------------------------------------------
def _legacy_resolve(query):
    import db.schema_catalog as schema
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()          # noqa: E731

    def matches(text, alias):
        a = norm(alias)
        return bool(a) and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", text) is not None

    t = norm(query)
    domains = schema._entity_domains()
    results = []
    for d in domains:
        if d.entity_type == "tract_fips":
            for fips in re.findall(r"\b\d{11}\b", query):
                if fips in schema._fetch_values(d.table, d.column):
                    results.append({"entity_type": d.entity_type, "table": d.table, "column": d.column, "value": fips,
                                    "domain": d.name, "score": 100})
    msa_context = any(matches(t, a) for a in ["msa", "msas", "metro", "metro area", "metropolitan area", "metro areas"])
    if any("MSA" in c.get("entity_types", []) for c in schema.semantic_matches(query)):
        msa_context = True
    for d in domains:
        for value in schema._fetch_values(d.table, d.column):
            nv = norm(value)
            if d.match_mode == "exact_or_prefix":
                if matches(t, nv):
                    results.append({"entity_type": d.entity_type, "table": d.table, "column": d.column, "value": value,
                                    "domain": d.name, "score": 80})
            elif d.match_mode in ("prefix", "components"):          # "components" was "prefix" before the lexicon
                prefix = norm(value.split(",", 1)[0].strip())
                if prefix and (matches(t, prefix) or matches(t, nv)):
                    results.append({"entity_type": d.entity_type, "table": d.table, "column": d.column, "value": value,
                                    "domain": d.name, "score": 95 if msa_context else 55})
    best = {}
    for e in results:
        k = (e["entity_type"], e["table"], norm(e["value"]))
        if k not in best or e["score"] > best[k]["score"]:
            best[k] = e
    return sorted(best.values(), key=lambda e: (-e["score"], e["entity_type"], e["value"]))


def test_entity_resolution_is_identical_to_the_original_outside_metro_context(fx):
    import db.schema_catalog as schema
    questions = [
        "How many houses are in Pittsburgh?", "average list price in Denver", "show houses in Indianapolis and Miami",
        "sold price in Pittsburgh", "Austin Texas houses", "houses in tract 42003000100", "what is the weather",
        "walk score in Denver vs Miami", "price history for listings in Pittsburgh", "my favorites",
    ]
    for q in questions:
        assert not any("MSA" in c.get("entity_types", []) for c in schema.semantic_matches(q)), q
        assert schema.resolve_request_entities(q) == _legacy_resolve(q), q


def test_entity_resolution_in_metro_context_only_adds_metros_or_flags_ambiguity(fx):
    import db.schema_catalog as schema
    rng = random.Random(11)
    names = ["Pittsburgh", "Indianapolis", "Denver", "Miami", "Austin", "Fort Worth", "Portland", "Columbus", "Salem",
             "Atlantis", "Beaver Valley Test"]
    templates = ["population of {a}", "land area of {a} and {b}", "average NRI risk in the {a} metro",
                 "MSA population of {a}", "{a} metro area flood risk", "population density of {a}, {b}"]
    checked = 0
    for _ in range(80):
        q = rng.choice(templates).format(a=rng.choice(names), b=rng.choice(names))
        new = schema.resolve_request_entities(q)
        old = _legacy_resolve(q)
        new_keys = {(e["entity_type"], e["value"]) for e in new}
        old_keys = {(e["entity_type"], e["value"]) for e in old}
        dropped = old_keys - new_keys
        # The original returned every same-named metro in one unflagged list (a SUM would quietly add them up).  The new
        # resolver ranks metro above micro, flags the tie and asks - so the ONLY thing it may drop is a lower-ranked
        # micro-area candidate, and only while flagging the ambiguity.
        assert all(v.endswith("Micro Area") for _, v in dropped), (q, dropped)
        if dropped:
            assert any(e.get("ambiguous") for e in schema.resolve_request_entities(q)), q
        extra = new_keys - old_keys
        assert all(t == "MSA" for t, _ in extra), (q, extra)       # only metro areas are newly recognised
        checked += 1
    assert checked == 80


def test_entities_carry_no_extra_keys_unless_ambiguous(fx):
    """Plan text shown to the SQL model stays identical to before whenever the same places are recognised."""
    p = plan("What is the riverine flood risk for the Pittsburgh metro?")
    assert set(p.resolved_entities[0]) == {"entity_type", "table", "column", "value", "domain", "score"}

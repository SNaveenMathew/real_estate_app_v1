"""Place-name resolution: how people refer to a place vs how the Census labels it. Pure Python, no database."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from census_fixture import METROS, NAME_ONLY_MSAS  # noqa: E402
from db import entity_lexicon as el               # noqa: E402

LABELS = [m[2] for m in METROS] + NAME_ONLY_MSAS + ["Cañon City, CO Micro Area"]
LEX = el.build(LABELS)


def resolve(text):
    return [(m.phrase, m.values, m.ambiguous) for m in el.find(LEX, text)]


def only(text):
    ms = el.find(LEX, text)
    assert len(ms) == 1 and not ms[0].ambiguous, ms
    return ms[0].values[0]


@pytest.mark.parametrize("text,label", [
    ("land area of Indianapolis", "Indianapolis-Carmel-Anderson, IN Metro Area"),
    ("carmel", "Indianapolis-Carmel-Anderson, IN Metro Area"),                       # non-principal city
    ("population of Denver", "Denver-Aurora-Lakewood, CO Metro Area"),
    ("Miami", "Miami-Fort Lauderdale-West Palm Beach, FL Metro Area"),
    ("fort worth", "Dallas-Fort Worth-Arlington, TX Metro Area"),
    ("Ft. Worth", "Dallas-Fort Worth-Arlington, TX Metro Area"),                     # spelling variants
    ("saint paul", "Minneapolis-St. Paul-Bloomington, MN-WI Metro Area"),
    ("Nashville", "Nashville-Davidson--Murfreesboro--Franklin, TN Metro Area"),     # hyphenated city, double-dash title
    ("murfreesboro", "Nashville-Davidson--Murfreesboro--Franklin, TN Metro Area"),
    ("Louisville", "Louisville/Jefferson County, KY-IN Metro Area"),
    ("Washington", "Washington-Arlington-Alexandria, DC-VA-MD-WV Metro Area"),
    ("Pittsburgh's population", "Pittsburgh, PA Metro Area"),
    ("PITTSBURGH, PA", "Pittsburgh, PA Metro Area"),
    ("Pittsburgh, PA Metro Area", "Pittsburgh, PA Metro Area"),
    ("Winston-Salem", "Winston-Salem, NC Metro Area"),                               # longest match beats "Salem"
    ("Canon City", "Cañon City, CO Micro Area"),                                     # accent folding
])
def test_how_people_name_places(text, label):
    assert only(text) == label


@pytest.mark.parametrize("text,label", [
    ("Portland, ME", "Portland-South Portland, ME Metro Area"),
    ("portland maine", "Portland-South Portland, ME Metro Area"),
    ("Portland OR", "Portland-Vancouver-Hillsboro, OR-WA Metro Area"),
    ("portland washington", "Portland-Vancouver-Hillsboro, OR-WA Metro Area"),
    ("Columbus indiana", "Columbus, IN Micro Area"),
    ("Mobile, AL", "Mobile, AL Metro Area"),
    ("Reading PA", "Reading, PA Metro Area"),
])
def test_state_qualifiers_disambiguate(text, label):
    assert only(text) == label


def test_bare_ambiguous_names_are_reported_not_guessed():
    (phrase, values, ambiguous), = resolve("population of Portland")
    assert ambiguous and set(values) == {"Portland-South Portland, ME Metro Area",
                                         "Portland-Vancouver-Hillsboro, OR-WA Metro Area"}
    (_, values, ambiguous), = resolve("Springfield")
    assert ambiguous and set(values) == {"Springfield, MA Metro Area", "Springfield, MO Metro Area"}
    (_, values, ambiguous), = resolve("Columbus")      # the micro area is not offered when metros exist
    assert ambiguous and set(values) == {"Columbus, OH Metro Area", "Columbus, GA-AL Metro Area"}


def test_better_tier_wins_over_ambiguity():
    # "Salem" is the principal city of Salem, OR and only a secondary part of Winston-Salem, NC.
    assert only("Salem") == "Salem, OR Metro Area"


@pytest.mark.parametrize("text", ["mobile home prices", "I was reading the report", "the bend in the road", "show houses"])
def test_everyday_words_are_not_places(text):
    assert resolve(text) == []


def test_several_places_come_back_in_request_order():
    got = [(p, v[0]) for p, v, _ in resolve("compare Indianapolis and Pittsburgh and Denver")]
    assert [g[0] for g in got] == ["indianapolis", "pittsburgh", "denver"]


def test_unknown_places_do_not_match():
    assert resolve("Atlantis and Narnia") == []
    assert resolve("") == []


def test_lexicon_cache_follows_the_data_not_the_clock():
    a = el.get_lexicon("t", ["Pittsburgh, PA Metro Area"])
    assert el.get_lexicon("t", ["Pittsburgh, PA Metro Area"]) is a
    b = el.get_lexicon("t", ["Pittsburgh, PA Metro Area", "Denver-Aurora-Lakewood, CO Metro Area"])
    assert b is not a and el.find(b, "denver")


def test_match_mode_suggestion_for_future_datasets():
    assert el.suggest_match_mode(LABELS) == "components"
    assert el.suggest_match_mode(["Pittsburgh", "Denver", "Austin", "Miami", "Boston"]) is None

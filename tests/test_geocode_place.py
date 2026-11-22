"""Address geocoding: canonical keys, the house-coordinate shortcut, the spelling ladder and the cache.

These are the failure shapes the general chat hit in production: the house dataset
says "4940 2nd Ave" while OpenStreetMap says "Second Avenue", Nominatim expands
neither into the other, and an un-geocodable endpoint used to end the turn with
"address could not be found" even though the house's coordinates were already in
the database. No test here touches the network: the geocoder client is replaced.
"""
import asyncio

import pytest

from services import routing_common as rc


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch):
    monkeypatch.setattr(rc, "NOMINATIM_MIN_INTERVAL_SECONDS", 0)


@pytest.fixture()
def nominatim(monkeypatch, fresh_db):
    """A scripted geocoder: {query text: [hits]} and a log of every query sent."""
    calls = []

    async def fake_get_json(url, *, params=None, json_body=None):
        calls.append(params["q"])
        return HITS.get(params["q"], [])

    HITS = {}
    monkeypatch.setattr(rc, "_get_json", fake_get_json)
    fake_get_json.hits = HITS
    fake_get_json.calls = calls
    return fake_get_json


PIKE = [{
    "lat": "40.4089261", "lon": "-79.9438914", "importance": 1e-05,
    "display_name": "4940, Second Avenue, Hazelwood, Pittsburgh, Allegheny County, Pennsylvania, 15207, United States",
    "address": {"house_number": "4940", "road": "Second Avenue", "city": "Pittsburgh"},
}]


# --------------------------------------------------------------- canonical form
def test_ordinal_abbreviations_are_normalised_so_one_address_has_one_key():
    digit = rc._canonical_address("4940 2nd Ave, Hazelwood, PA 15207")
    word = rc._canonical_address("4940 Second Avenue, Hazelwood, PA 15207")
    assert digit == word == "4940 second avenue, hazelwood, pa 15207"
    assert rc._canonical_address("112 S 18th St") == "112 south eighteenth street"
    assert rc._canonical_address("1 N Main St, Pittsburgh") == "1 north main street, pittsburgh"
    assert rc._ordinal_word(21) == "twenty first" and rc._ordinal_word(30) == "thirtieth"
    assert rc._ordinal_word(112) == "112th"                                # rare: keep the digits


def test_every_spelling_of_an_address_shares_one_cache_key():
    keys = {
        rc._address_cache_key("4940 2nd Ave, Hazelwood, PA 15207", "Pittsburgh, PA"),
        rc._address_cache_key("4940 2nd Ave, Hazelwood, PA 15207, Pittsburgh, PA", "Pittsburgh, PA"),
        rc._address_cache_key("4940 Second Avenue, Hazelwood, PA 15207", "Pittsburgh, PA"),
    }
    assert len(keys) == 1
    assert rc._address_cache_key("954 Gladys Ave, Pittsburgh, PA 15216", "Pittsburgh, PA") != keys


def test_the_city_is_appended_only_when_the_query_does_not_already_name_it():
    assert rc._raw_search_text("954 Gladys Ave, Pittsburgh, PA 15216", "Pittsburgh, PA") == "954 Gladys Ave, Pittsburgh, PA 15216"
    assert rc._canonical_search_text("4940 2nd Ave, Hazelwood, PA 15207", "Pittsburgh, PA") == \
        "4940 second avenue, hazelwood, pa 15207, pittsburgh, pa"
    assert rc._canonical_search_text("4940 Second Avenue, Hazelwood, PA 15207, Pittsburgh, PA", "Pittsburgh, PA") == \
        "4940 second avenue, hazelwood, pa 15207, pittsburgh, pa"


# ------------------------------------------------------------- the house shortcut
def test_a_listed_house_is_geocoded_from_the_database_without_any_http(fresh_db, nominatim):
    fresh_db.get_conn().execute(
        "INSERT INTO houses (house_id,address,city,state,zip,lat,lon) VALUES "
        "('haz','4940 2nd Ave','Hazelwood','PA','15207',40.408895,-79.943894)"
    )
    place = run(rc.geocode_place("4940 2nd Ave, Hazelwood, PA 15207"))
    assert (place["lat"], place["lon"]) == (40.408895, -79.943894)
    assert place["source"] == "houses" and place["cached"] is True
    assert "Hazelwood" in place["display_name"]
    assert nominatim.calls == []                                          # no geocoder needed at all


def test_the_house_shortcut_is_written_to_the_cache_for_reuse(fresh_db, nominatim):
    fresh_db.get_conn().execute(
        "INSERT INTO houses (house_id,address,city,state,zip,lat,lon) VALUES "
        "('haz','4940 2nd Ave','Hazelwood','PA','15207',40.408895,-79.943894)"
    )
    run(rc.geocode_place("4940 2nd Ave, Hazelwood, PA 15207"))
    row = fresh_db.get_conn().execute(
        "SELECT geocode_source, geocode_accuracy FROM geocode_cache").fetchall()
    assert row == [("houses", "house_coordinates")]


def test_a_different_house_number_in_the_same_zip_is_not_confused(fresh_db):
    fresh_db.get_conn().execute(
        "INSERT INTO houses (house_id,address,city,state,zip,lat,lon) VALUES "
        "('a','4940 2nd Ave','Hazelwood','PA','15207',40.408895,-79.943894),"
        "('b','4942 2nd Ave','Hazelwood','PA','15207',40.408900,-79.943900)"
    )
    assert rc._house_coordinates("4942 2nd Ave, Hazelwood, PA 15207", "Pittsburgh, PA")["lat"] == 40.4089
    assert rc._house_coordinates("4941 2nd Ave, Hazelwood, PA 15207", "Pittsburgh, PA") is None


# -------------------------------------------------------------------- the ladder
def test_the_second_spelling_is_tried_when_the_first_returns_nothing(fresh_db, nominatim):
    nominatim.hits["4940 second avenue, hazelwood, pa 15207, pittsburgh, pa"] = PIKE
    place = run(rc.geocode_place("4940 2nd Ave, Hazelwood, PA 15207"))
    assert place["lat"] == pytest.approx(40.4089261) and place["cached"] is False
    assert nominatim.calls == [
        "4940 2nd Ave, Hazelwood, PA 15207, Pittsburgh, PA",               # the spelling sent before the fix
        "4940 second avenue, hazelwood, pa 15207, pittsburgh, pa",         # OSM's spelling
    ]


def test_digit_ordinals_are_kept_in_the_third_spelling(fresh_db, nominatim):
    """OSM keeps "South 18th Street" in digits, so the words form must not be the only retry."""
    assert rc._canonical_search_text("112 S 18th St, South Side, PA 15203", "Pittsburgh, PA",
                                     ordinal_words=False) == "112 south 18th street, south side, pa 15203, pittsburgh, pa"


def test_a_resolved_address_is_cached_and_served_for_every_spelling(fresh_db, nominatim):
    nominatim.hits["4940 second avenue, hazelwood, pa 15207, pittsburgh, pa"] = PIKE
    first = run(rc.geocode_place("4940 2nd Ave, Hazelwood, PA 15207"))
    calls_after_first = len(nominatim.calls)
    second = run(rc.geocode_place("4940 Second Avenue, Hazelwood, PA 15207"))
    assert second["cached"] is True and second["lat"] == first["lat"]
    assert second["display_name"].startswith("4940, Second Avenue")
    assert len(nominatim.calls) == calls_after_first                        # no second lookup


def test_a_remembered_miss_is_not_retried_on_the_next_turn(fresh_db, nominatim):
    with pytest.raises(ValueError, match="Could not find"):
        run(rc.geocode_place("1 Nowhere At All, Hazelwood, PA 15207"))
    first = len(nominatim.calls)
    assert first == 1                          # no ordinal here, so all three spellings coincide
    with pytest.raises(ValueError, match="Could not find"):
        run(rc.geocode_place("1 Nowhere At All, Hazelwood, PA 15207"))
    assert len(nominatim.calls) == first                                  # served from the miss entry


def test_an_expired_miss_is_retried(fresh_db, nominatim):
    with pytest.raises(ValueError):
        run(rc.geocode_place("1 Nowhere At All, Hazelwood, PA 15207"))
    key = rc._address_cache_key("1 Nowhere At All, Hazelwood, PA 15207", "Pittsburgh, PA")
    fresh_db.get_conn().execute(
        "UPDATE geocode_cache SET geocoded_at = CURRENT_TIMESTAMP - INTERVAL 30 DAY WHERE address_key = ?",
        [key],
    )
    nominatim.hits["1 Nowhere At All, Hazelwood, PA 15207, Pittsburgh, PA"] = PIKE
    place = run(rc.geocode_place("1 Nowhere At All, Hazelwood, PA 15207"))
    assert place["lat"] == pytest.approx(40.4089261)


def test_out_of_bounds_hits_are_ignored_and_reported_as_a_valid_location_problem(fresh_db, nominatim):
    far_away = [{"lat": "47.6", "lon": "-122.3", "display_name": "Seattle, WA", "importance": 1.0}]
    nominatim.hits["4940 2nd Ave, Hazelwood, PA 15207, Pittsburgh, PA"] = far_away
    nominatim.hits["4940 second avenue, hazelwood, pa 15207, pittsburgh, pa"] = far_away
    nominatim.hits["4940 2nd avenue, hazelwood, pa 15207, pittsburgh, pa"] = far_away
    with pytest.raises(ValueError, match="valid location"):
        run(rc.geocode_place("4940 2nd Ave, Hazelwood, PA 15207"))


def test_cached_coordinates_outside_pittsburgh_are_rejected(fresh_db, nominatim):
    key = rc._address_cache_key("500 Grant St, Pittsburgh, PA", "Pittsburgh, PA")
    fresh_db.get_conn().execute(
        "INSERT INTO geocode_cache (address_key, full_address, lat, lon, geocode_source, geocode_accuracy) "
        "VALUES (?, 'Somewhere else', 47.6, -122.3, 'nominatim', 'pittsburgh_bounded_place')", [key])
    nominatim.hits["500 grant street, pittsburgh, pa"] = [{
        "lat": "40.4395", "lon": "-79.9965", "display_name": "500 Grant St, Pittsburgh, PA", "importance": 1.0,
    }]
    assert run(rc.geocode_place("500 Grant St, Pittsburgh, PA"))["lat"] == pytest.approx(40.4395)


def test_legacy_cache_rows_stay_reachable(fresh_db, nominatim):
    """Rows written by the previous implementation (md5 of the raw search string) still count."""
    legacy = rc._legacy_address_key("500 Grant St, Pittsburgh, PA, Pittsburgh, PA")
    fresh_db.get_conn().execute(
        "INSERT INTO geocode_cache (address_key, full_address, lat, lon, geocode_source, geocode_accuracy) "
        "VALUES (?, '500 Grant St, Pittsburgh, PA', 40.4395, -79.9965, 'nominatim', 'pittsburgh_bounded_place')",
        [legacy])
    place = run(rc.geocode_place("500 Grant St, Pittsburgh, PA, Pittsburgh, PA"))
    assert place["cached"] is True and place["lat"] == pytest.approx(40.4395)
    assert nominatim.calls == []

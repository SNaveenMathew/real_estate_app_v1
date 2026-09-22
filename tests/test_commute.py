"""Commute times: geocoding, OSRM/OTP clients, the refresh job, catalog integration, House Chat."""
import asyncio
import datetime as dt
import json

import pytest

import db.schema_catalog as schema
from commute_helpers import WORK, MockRouting, add_houses, expected_minutes, routing  # noqa: F401
from services import commute
from services.commute import Place

W = Place(WORK[0], WORK[1], "Downtown")


# ------------------------------------------------------------------ work location
def test_coordinates_need_decimals_so_street_numbers_are_never_read_as_coordinates():
    assert commute.parse_latlon("40.4406, -79.9959").lat == 40.4406
    assert commute.parse_latlon("40.4 -79.9").lon == -79.9
    assert commute.parse_latlon("12 34") is None and commute.parse_latlon("91.5, 10.5") is None
    assert commute.parse_latlon("500 Grant St") is None


def test_geocoding_tries_census_then_nominatim_and_identifies_itself(routing):
    routing.census["500 Grant St, Pittsburgh, PA"] = ("500 GRANT ST, PITTSBURGH, PA, 15219", WORK[1], WORK[0])
    routing.nominatim["Google HQ"] = ("Googleplex, Mountain View, CA", -122.084, 37.422)
    p = commute.geocode_work_address("500 Grant St, Pittsburgh, PA")
    assert (p.source, p.label, p.lat) == ("census", "500 GRANT ST, PITTSBURGH, PA, 15219", WORK[0])
    q = commute.geocode_work_address("Google HQ")
    assert q.source == "nominatim" and q.lon == -122.084
    nominatim_calls = [c for c in routing.raw if c[0].startswith("/nominatim")]
    assert len(nominatim_calls) == 1                                       # exactly one lookup per user action
    assert nominatim_calls[0][2]["user-agent"].startswith("RealEstateIntelligence")
    assert nominatim_calls[0][1]["countrycodes"] == "us"
    with pytest.raises(commute.RoutingError, match="Could not find"):
        commute.geocode_work_address("nowhere at all")
    assert commute.geocode_work_address("40.44, -79.99").source == "coordinates"      # no network needed
    with pytest.raises(commute.RoutingError):
        commute.geocode_work_address("   ")


# ------------------------------------------------------------------ OSRM client
def test_matrix_batches_requests_and_sends_lon_lat(routing):
    client = commute.OsrmClient(routing.url + "/car", table_max=90, min_interval=0)
    sources = [Place(WORK[0] + 0.001 * i, WORK[1]) for i in range(1, 201)]
    routing.null_sources = {3}                                               # index within each request
    legs = client.matrix(sources, W)
    tables = [c for c in routing.calls if c[0] == "table"]
    assert [n for _, _, n in tables] == [91, 91, 21]                         # 90 + 90 + 20 sources, plus the destination
    path, query, _ = routing.raw[0]
    assert "-79.995900,40.441600;" in path and path.endswith("-79.995900,40.440600")   # lon,lat order; destination last
    assert query["sources"] == ";".join(str(i) for i in range(90)) and query["destinations"] == "90"
    assert query["annotations"] == "duration,distance"
    assert len(legs) == 200 and [i for i, l in enumerate(legs) if l is None] == [3, 93, 183]
    assert legs[0].seconds / 60 == pytest.approx(expected_minutes("car", *[WORK[0] + 0.001, WORK[1]]), rel=1e-3)
    assert legs[0].meters / 1609.344 == pytest.approx(commute.haversine_miles(WORK[0] + 0.001, WORK[1], *WORK) * 1.3, rel=1e-3)


def test_matrix_falls_back_to_route_when_table_is_unavailable(routing):
    routing.fail_table = {"car"}
    client = commute.OsrmClient(routing.url + "/car", min_interval=0)
    legs = client.matrix([Place(WORK[0] + 0.01 * i, WORK[1]) for i in range(1, 4)], W)
    assert [c[0] for c in routing.calls] == ["table", "route", "route", "route"]
    assert all(l is not None for l in legs) and "400" in client.last_error


def test_transient_errors_are_retried_and_outages_degrade_to_none(routing):
    routing.flaky = 2                                                        # two 503s, then success
    client = commute.OsrmClient(routing.url + "/car", min_interval=0)
    assert client.matrix([Place(WORK[0] + 0.01, WORK[1])], W)[0] is not None
    routing.down = {"car"}
    dead = commute.OsrmClient(routing.url + "/car", min_interval=0)
    assert dead.matrix([Place(WORK[0] + 0.01, WORK[1])], W) == [None]        # never raises
    assert "500" in dead.last_error


def test_unreachable_server_is_reported_not_raised(routing):
    dead = commute.OsrmClient("http://127.0.0.1:9/car", min_interval=0)
    assert dead.matrix([Place(WORK[0] + 0.01, WORK[1])], W) == [None] and dead.last_error


# ------------------------------------------------------------------ compute_chunk
def test_compute_chunk_applies_factor_range_caps_and_status(routing, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "commute_drive_factor", 1.3)
    chunk = [("near", WORK[0] + 0.02, WORK[1]), ("mid", WORK[0] + 0.5, WORK[1]),
             ("far", WORK[0] + 2.0, WORK[1]), ("huge", WORK[0] + 40.0, WORK[1])]
    modes = ["drive", "bike", "walk"]
    rows, warns = commute.compute_chunk(chunk, W, modes, commute.build_clients(modes), "k1")
    by = {r["house_id"]: r for r in rows}
    near = by["near"]
    assert near["status"] == "ok" and near["source"] == "drive+bike+walk"
    assert near["drive_min"] == pytest.approx(expected_minutes("car", WORK[0] + 0.02, WORK[1]) * 1.3, abs=0.06)   # factor: drive only
    assert near["bike_min"] == pytest.approx(expected_minutes("bike", WORK[0] + 0.02, WORK[1]), abs=0.06)
    assert near["walk_min"] == pytest.approx(expected_minutes("foot", WORK[0] + 0.02, WORK[1]), abs=0.06)
    assert near["drive_miles"] == pytest.approx(near["straight_line_miles"] * 1.3, rel=0.02)
    assert by["mid"]["drive_min"] is not None and by["mid"]["bike_min"] is None and by["mid"]["walk_min"] is None   # ~34 mi
    assert by["mid"]["status"] == "ok"                                       # bike/walk were never expected that far out
    assert by["far"]["drive_min"] is not None and by["far"]["bike_min"] is None
    assert by["huge"]["status"] == "out_of_range" and by["huge"]["drive_min"] is None
    assert warns == []
    calls = [(s, p) for s, p, _ in routing.calls]
    assert calls.count(("table", "walk")) == 0 and calls.count(("table", "foot")) == 1     # only "near" was in walking range


def test_one_dead_mode_is_partial_and_explained(routing):
    routing.down = {"bike"}
    rows, warns = commute.compute_chunk([("a", WORK[0] + 0.02, WORK[1])], W, ["drive", "bike"],
                                        commute.build_clients(["drive", "bike"]), "k")
    assert rows[0]["status"] == "partial" and rows[0]["drive_min"] is not None and rows[0]["bike_min"] is None
    assert warns and warns[0].startswith("Bike: no routes were returned") and "500" in warns[0]


# ------------------------------------------------------------------ the refresh job
def test_refresh_job_end_to_end_with_staleness(routing, fresh_db):
    conn = fresh_db.get_conn()
    add_houses(conn, 5)
    commute.set_work(W)
    snap = asyncio.run(commute.refresh_now("missing"))
    assert (snap["state"], snap["total"], snap["done"], snap["failed"]) == ("done", 5, 5, 0)
    assert conn.execute("SELECT COUNT(*) FROM house_commute WHERE status = 'ok'").fetchone()[0] == 5
    assert len([c for c in routing.calls if c[0] == "table"]) == 3           # one table request per mode, not per house
    assert len(commute.summary()["rows"]) == 5
    again = asyncio.run(commute.refresh_now("missing"))
    assert again["total"] == 0 and "up to date" in again["message"] and len(routing.calls) == 3   # nothing recomputed
    commute.set_work(Place(WORK[0] + 0.005, WORK[1], "Moved office"))        # new destination: every stored row is stale
    assert commute.summary()["rows"] == {} and commute.commute_for_house("h1")["fresh"] is False
    asyncio.run(commute.refresh_now("missing"))
    assert commute.commute_for_house("h1")["fresh"] is True and commute.commute_for_house("h1")["work_label"] == "Moved office"


def test_new_house_is_the_only_one_computed(routing, fresh_db):
    conn = fresh_db.get_conn()
    add_houses(conn, 3)
    commute.set_work(W)
    asyncio.run(commute.refresh_now("missing"))
    conn.execute("INSERT INTO houses (house_id, address, lat, lon) VALUES ('new', '9 New St', ?, ?)", [WORK[0] + 0.03, WORK[1]])
    snap = asyncio.run(commute.refresh_now("missing"))
    assert snap["total"] == 1 and snap["done"] == 1
    assert asyncio.run(commute.refresh_now("all"))["total"] == 4               # 'all' recomputes even fresh rows


def test_houses_without_coordinates_are_skipped_and_counted(routing, fresh_db):
    conn = fresh_db.get_conn()
    add_houses(conn, 2)
    conn.execute("INSERT INTO houses (house_id, address) VALUES ('nocoords', '1 Nowhere Rd')")
    commute.set_work(W)
    st = commute.status()
    assert st["counts"] == {"houses": 3, "with_coordinates": 2, "fresh": 0, "to_compute": 2}
    assert asyncio.run(commute.refresh_now("missing"))["total"] == 2


def test_background_start_is_single_flight(routing, fresh_db):
    add_houses(fresh_db.get_conn(), 3)
    commute.set_work(W)
    routing.delay = 0.25

    async def scenario():
        first = commute.start_refresh("missing")
        second = commute.start_refresh("missing")              # while the first is running: no second job
        assert first["state"] == "running" and second["state"] == "running"
        for _ in range(200):
            if commute.job_snapshot()["state"] != "running":
                break
            await asyncio.sleep(0.05)
        return commute.job_snapshot()
    snap = asyncio.run(scenario())
    assert snap["state"] == "done" and snap["done"] == 3
    assert len([c for c in routing.calls if c[0] == "table"]) == 3           # not 6


def test_no_work_location_is_an_error_state_not_a_crash(routing, fresh_db):
    snap = asyncio.run(commute.refresh_now("missing"))
    assert snap["state"] == "error" and "work location" in snap["message"]
    commute.set_work(W)                                                      # the guard released: a later run works
    assert asyncio.run(commute.refresh_now("missing"))["state"] == "done"


def test_changing_any_input_makes_stored_rows_stale(routing, fresh_db, monkeypatch):
    from config import settings
    add_houses(fresh_db.get_conn(), 2)
    commute.set_work(W)
    asyncio.run(commute.refresh_now("missing"))
    assert commute.commute_for_house("h1")["fresh"] is True
    for name, value in [("commute_drive_factor", 1.4), ("osrm_drive_url", routing.url + "/car/"), ("commute_modes", "drive"),
                        ("otp_base_url", routing.url)]:
        monkeypatch.setattr(settings, name, value)
        assert commute.commute_for_house("h1")["fresh"] is False, name
        monkeypatch.undo() if False else None
        monkeypatch.setattr(settings, name, {"commute_drive_factor": 1.0, "osrm_drive_url": routing.url + "/car",
                                              "commute_modes": "drive,bike,walk", "otp_base_url": ""}[name])
    assert commute.commute_for_house("h1")["fresh"] is True


def test_clearing_the_work_location_removes_its_estimates(routing, fresh_db):
    add_houses(fresh_db.get_conn(), 2)
    commute.set_work(W)
    asyncio.run(commute.refresh_now("missing"))
    commute.clear_work()
    assert commute.get_work() is None and commute.summary() == {"work": None, "rows": {}}
    assert fresh_db.get_conn().execute("SELECT COUNT(*) FROM house_commute").fetchone()[0] == 0


# ------------------------------------------------------------------ transit (OpenTripPlanner, optional)
def test_otp_graphql_and_rest_are_parsed_and_failures_degrade(routing):
    a, b, day = Place(40.45, -79.99), Place(*WORK), dt.date(2025, 1, 7)
    routing.otp_graphql = {"data": {"plan": {"itineraries": [{"duration": 2700, "legs": [
        {"mode": "WALK"}, {"mode": "BUS"}, {"mode": "SUBWAY"}, {"mode": "WALK"}]}]}}}
    gql = commute.OtpClient(routing.url, api="graphql", depart_time="08:00")
    assert gql.trip(a, b, day) == (2700.0, 1)                                # two transit legs = one transfer
    query = routing.otp_bodies[0]["query"]
    assert 'date: "2025-01-07"' in query and 'time: "08:00"' in query and "TRANSIT" in query
    routing.otp_rest = {"plan": {"itineraries": [{"duration": 3000, "transfers": 2}]}}
    rest = commute.OtpClient(routing.url, api="rest", depart_time="13:05")
    assert rest.trip(a, b, day) == (3000.0, 2)
    _, q, _ = [c for c in routing.raw if c[0].startswith("/otp/routers")][0]
    assert q["date"] == "01-07-2025" and q["time"] == "1:05pm" and q["mode"] == "TRANSIT,WALK" and q["fromPlace"] == "40.45,-79.99"
    routing.otp_graphql = routing.otp_rest = None                            # no itinerary
    assert gql.trip(a, b, day) is None and rest.trip(a, b, day) is None
    assert len(commute.OtpClient(routing.url).trips([a, a, a], b)) == 3
    assert commute.next_weekday(dt.date(2025, 1, 3)) == dt.date(2025, 1, 6)  # Friday -> Monday
    assert commute.next_weekday(dt.date(2025, 1, 7)) == dt.date(2025, 1, 8)


def test_transit_is_on_only_when_opentripplanner_is_configured(routing, monkeypatch):
    from config import settings
    assert commute.enabled_modes() == ["drive", "bike", "walk"]
    monkeypatch.setattr(settings, "otp_base_url", routing.url)
    assert commute.enabled_modes()[-1] == "transit"
    monkeypatch.setattr(settings, "commute_modes", "drive")
    assert commute.enabled_modes() == ["drive", "transit"]


def test_transit_flows_through_the_job(routing, fresh_db, monkeypatch):
    from config import settings
    routing.otp_graphql = {"data": {"plan": {"itineraries": [{"duration": 1800, "legs": [{"mode": "WALK"}, {"mode": "BUS"}]}]}}}
    monkeypatch.setattr(settings, "otp_base_url", routing.url)
    add_houses(fresh_db.get_conn(), 2)
    commute.set_work(W)
    assert asyncio.run(commute.refresh_now("missing"))["state"] == "done"
    row = commute.commute_for_house("h1")
    assert row["transit_min"] == 30.0 and row["transit_transfers"] == 0 and "transit" in row["source"]


def test_public_vs_private_routing_servers_drive_the_privacy_note():
    assert commute.is_public_url("https://router.project-osrm.org")
    assert not any(commute.is_public_url(u) for u in ["http://localhost:5000", "http://127.0.0.1:5000", "http://192.168.1.20:5000",
                                                     "http://10.0.0.5:5000", "http://osrm.local:5000"])


# ------------------------------------------------------------------ catalog: General Chat can rank by commute
def test_house_commute_is_a_built_in_catalog_table(fresh_db):
    meta = schema.TABLES["house_commute"]
    assert meta.origin == "builtin" and meta.domain == "housing" and "work_key" in meta.hidden_columns and meta.setup_hint
    assert "house_commute:house_id=houses:house_id" in [r.key() for r in schema.RELATIONSHIPS]
    mine = {k for k in schema.SEMANTIC_GLOSSARY if k.startswith("house_commute_")}
    assert mine == {"house_commute_drive", "house_commute_bike", "house_commute_walk", "house_commute_transit", "house_commute_distance"}
    assert "house_commute_drive" in schema.SEMANTIC_GLOSSARY["house_commute_bike"]["overrides"]
    assert "house_commute" in schema.availability_report()[0]                # listed (as empty) before a work location is set


def test_general_chat_ranks_houses_by_commute_and_mode_phrases_win(routing, fresh_db):
    from agents.query_planner import build_query_plan
    from agents.tools import _compile_sql_from_plan, _validate_sql_against_plan, validate_sql
    add_houses(fresh_db.get_conn(), 5)
    commute.set_work(W)
    asyncio.run(commute.refresh_now("missing"))
    qp = build_query_plan("Which houses have the shortest commute?")
    assert {"houses", "house_commute"} <= set(qp.required_tables) and qp.operation == "rank"
    sql = _compile_sql_from_plan(qp)
    validate_sql(sql)
    _validate_sql_against_plan(sql, qp)
    df = fresh_db.query(sql)
    assert df.iloc[0, 0] == "1 Test St" and list(df.iloc[:, 1]) == sorted(df.iloc[:, 1]) and len(df) == 5
    bike = build_query_plan("Which houses have the shortest bike commute?")
    assert "house_commute_bike" in bike.semantic_keys and "house_commute_drive" not in bike.semantic_keys
    assert "bike_min" in _compile_sql_from_plan(bike)
    assert "drive_miles" in _compile_sql_from_plan(build_query_plan("What is the average commute distance for my houses?"))
    longest = fresh_db.query(_compile_sql_from_plan(build_query_plan("Which house has the longest commute?")))
    assert longest.iloc[0, 0] == "5 Test St"


def test_new_commute_phrases_never_change_matches_for_existing_aliases(fresh_db):
    """Regression guard: no built-in alias, in any sentence shape, may start selecting a commute concept."""
    checked = 0
    for key, item in schema.SEMANTIC_GLOSSARY.items():
        if key.startswith("house_commute"):
            continue
        for alias in item.get("aliases", []):
            for shape in ("what is the average {a} in Pittsburgh", "rank the top 10 by {a}", "{a}"):
                keys = [m["key"] for m in schema.semantic_matches(shape.format(a=alias))]
                assert not [k for k in keys if k.startswith("house_commute")], (alias, keys)
                checked += 1
    assert checked > 300


# ------------------------------------------------------------------ House Chat + data loading
def test_house_chat_reports_the_commute_and_is_honest_when_unset(routing, fresh_db):
    from agents import house_agent as ha
    add_houses(fresh_db.get_conn(), 3)
    fns = ha.make_house_approved_functions("h1")
    assert "No work location is set" in fns["get_commute_info"]()
    commute.set_work(W)
    assert "has not been computed yet" in fns["get_commute_info"]()
    asyncio.run(commute.refresh_now("missing"))
    info = json.loads(fns["get_commute_info"]())
    assert info["work_location"] == "Downtown" and info["up_to_date"] is True
    assert set(info["estimates"]) == {"drive", "bike", "walk"} and info["estimates"]["drive"]["minutes"] >= 1
    assert "free-flow" in info["basis"] and set(info["redfin_scores"]) == {"walk_score", "bike_score", "transit_score"}
    assert "get_commute_info" in ha._HOUSE_CODE_AGENT_PROMPT
    ha._validate_house_program("final_result = get_commute_info()", set(fns))


def test_setup_data_computes_commutes_only_when_a_work_location_exists(routing, fresh_db, capsys, monkeypatch):
    import importlib
    from config import settings
    sd = importlib.import_module("setup_data")
    sd._run_commute_step(explicit=True)
    assert "no work location" in capsys.readouterr().out
    add_houses(fresh_db.get_conn(), 3)
    routing.census["500 Grant St, Pittsburgh, PA"] = ("500 GRANT ST, PITTSBURGH, PA, 15219", WORK[1], WORK[0])
    monkeypatch.setattr(settings, "work_address", "500 Grant St, Pittsburgh, PA")
    sd._run_commute_step(explicit=False)
    out = capsys.readouterr().out
    assert "Work location saved from WORK_ADDRESS" in out and "Computed 3 house(s)" in out


def test_rows_with_some_null_modes_are_json_safe(routing, fresh_db):
    """A far house has no walking time. Reading through pandas would make that NaN, and NaN is not valid JSON
    (the Commute API returned HTTP 500 for it before rows were read as plain Python)."""
    conn = fresh_db.get_conn()
    add_houses(conn, 2)
    conn.execute("INSERT INTO houses (house_id, address, lat, lon) VALUES ('hfar', 'Far St', ?, ?)", [WORK[0] + 0.3, WORK[1]])   # ~21 mi
    commute.set_work(W)
    asyncio.run(commute.refresh_now("missing"))
    summary = commute.summary()
    assert summary["rows"]["hfar"]["walk_min"] is None and summary["rows"]["hfar"]["bike_min"] is not None
    assert summary["rows"]["h1"]["transit_min"] is None                      # transit is off: NULL for every house
    json.dumps(summary, allow_nan=False)                                     # raises ValueError on NaN
    json.dumps(commute.commute_for_house("hfar"), allow_nan=False)
    json.dumps(commute.status(), allow_nan=False)

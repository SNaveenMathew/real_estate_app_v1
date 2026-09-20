"""The Data page workflow end to end (service level): upload -> describe -> propose -> approve -> live."""
import pandas as pd
import pytest

import db.schema_catalog as schema
from conftest import HOUSES
from services import dataset_onboarding as ob
from services import house_links


def _table(ds):
    return next(p for p in ds["proposals"] if p["kind"] == "table")


def _link(ds, right_table, **where):
    return next(p for p in ds["proposals"] if p["kind"] == "relationship" and p["payload"]["right_table"] == right_table
                and all(p["payload"].get(k) == v for k, v in where.items()))


def test_block_groups_link_through_a_derived_tract_key_with_verifiable_evidence(reference_data, blockgroup_csv):
    ds = ob.create_dataset(blockgroup_csv, "Smart Location Database.csv")
    cols = {c["name"]: c for c in ds["columns"]}
    assert ds["table_name"] == "smart_location_database"
    assert cols["geoid20"]["key_kind"] == "block_group_fips" and cols["geoid20"]["role"] == "key"
    assert cols["natwalkind"]["role"] == "measure" and cols["lead_service_line"]["role"] == "measure"
    ds = ob.analyze(ds["dataset_id"])
    rels = {(p["payload"]["right_table"], p["payload"]["right_expr"]): p for p in ds["proposals"] if p["kind"] == "relationship"}
    houses = rels[("houses", "tract_fips")]
    pl, ev = houses["payload"], houses["evidence"]
    assert pl["derive"]["transform"] == {"kind": "left", "width": 11} and pl["left_expr"] == "tract_fips"
    assert pl["confidence"] == "high" and ev["stats"]["right_covered_pct"] == 100.0
    ex = ev["examples"][0]                                    # what a reviewer sees: raw -> normalized -> matching house
    assert len(ex["raw"]) == 12 and ex["normalized"] == ex["raw"][:11] and ex["matches"] and ex["match_count"] >= 1
    assert ev["unmatched"] and all(len(u["normalized"]) == 11 for u in ev["unmatched"])
    assert ("nri_tracts", "tract_fips") in rels and ("census_tracts", "tract_fips") in rels     # every catalog table with the key
    assert not any(p["payload"]["left_expr"] in ("tractce", "countyfp", "statefp") for p in rels.values())   # components are not keys


def test_nothing_reaches_the_catalog_until_approved_and_approval_is_immediate(reference_data, blockgroup_csv):
    before = set(schema.TABLES)
    ds = ob.create_dataset(blockgroup_csv, "Smart Location Database.csv")
    assert set(schema.TABLES) == before                        # uploading changes nothing
    ds = ob.analyze(ds["dataset_id"])
    assert set(schema.TABLES) == before and not [r for r in schema.RELATIONSHIPS if r.origin == "upload"]
    link = _link(ds, "houses")
    with pytest.raises(ob.OnboardingError, match="Approve the table first"):
        ob.decide(link["proposal_id"], "approve")
    version = schema.catalog_version()
    ob.decide(_table(ds)["proposal_id"], "approve")
    assert schema.TABLES["smart_location_database"].origin == "upload" and schema.catalog_version() != version
    assert any(k.startswith("smart_location_database_") for k in schema.SEMANTIC_GLOSSARY)
    assert not [r for r in schema.RELATIONSHIPS if r.origin == "upload"]             # the table alone adds no links
    ob.decide(link["proposal_id"], "approve", edits={"note": "tract-level join"})
    assert [r.key() for r in schema.RELATIONSHIPS if r.origin == "upload"] == ["smart_location_database:tract_fips=houses:tract_fips"]
    rows = reference_data.execute("SELECT geoid20, tract_fips FROM smart_location_database LIMIT 8").fetchall()
    assert all(t == g[:11] for g, t in rows)                                         # materialised exactly as the examples showed
    with pytest.raises(ob.OnboardingError) as exc:
        ob.decide(link["proposal_id"], "approve")
    assert exc.value.status_code == 409
    assert not reference_data.execute("SELECT 1 FROM information_schema.tables WHERE table_name LIKE 'stg_%'").fetchall()


def test_rejected_links_never_reach_the_agents(reference_data, blockgroup_csv):
    ds = ob.analyze(ob.create_dataset(blockgroup_csv, "Smart Location Database.csv")["dataset_id"])
    ob.decide(_table(ds)["proposal_id"], "approve")
    ob.decide(_link(ds, "houses")["proposal_id"], "reject", note="wrong grain")
    assert not [r for r in schema.RELATIONSHIPS if r.origin == "upload"]
    assert not schema.house_link_plan("smart_location_database")


def test_general_chat_planning_and_sql_use_the_new_table_immediately(walk_dataset, blockgroup_csv):
    import db.duckdb_store as store
    from agents.query_planner import build_query_plan
    from agents.tools import _compile_sql_from_plan, _validate_sql_against_plan, validate_sql
    qp = build_query_plan("What is the average national walkability index for my houses?")
    assert {"houses", "smart_location_database"} <= set(qp.required_tables)
    sql = _compile_sql_from_plan(qp)
    validate_sql(sql)
    _validate_sql_against_plan(sql, qp)
    got = float(store.query(sql).iloc[0, 0])
    df = pd.read_csv(blockgroup_csv, dtype=str)
    df["NatWalkInd"] = df["NatWalkInd"].astype(float)
    df["tract"] = df["GEOID20"].str[:11]
    vals = [v for h in HOUSES for v in df[df["tract"] == h[5]]["NatWalkInd"]]
    assert got == pytest.approx(sum(vals) / len(vals))                # average over (house, block group) pairs


def test_address_matching_links_rows_to_houses_and_house_chat_can_read_them(reference_data, tmp_path):
    p = tmp_path / "inspections.csv"
    pd.DataFrame({"Property Address": ["123 Main St", "45 Oak Avenue", "9 Elm Street", "500 Unknown Blvd"],
                  "Zip": ["15213", "15213", "15217", "15000"], "Inspection Score": [88, 71, 93, 60],
                  "Inspector": ["Ann", "Bo", "Cy", "Di"]}).to_csv(p, index=False)
    ds = ob.analyze(ob.create_dataset(p, "inspections.csv")["dataset_id"])
    link = next(x for x in ds["proposals"] if x["group_key"] == "address_house")
    ev = link["evidence"]
    assert ev["stats"]["left_rows_matched"] == 3 and ev["stats"]["tiers"] == {"1": 3}
    assert ev["unmatched"][0]["raw"] == "500 Unknown Blvd" and ev["examples"][0]["matches"][0]["house_id"] in {"h1", "h2", "h3"}
    ob.decide(_table(ds)["proposal_id"], "approve")
    ob.decide(link["proposal_id"], "approve")
    text = house_links.linked_records("h1")
    assert "88" in text and "Ann" in text and "Joined via" in text
    assert "No rows match" in house_links.linked_records("h4", "inspections")
    assert "Unknown dataset" in house_links.linked_records("h1", "nope")


def _tract_polygons(monkeypatch, overlapping=False):
    import geopandas as gpd
    from shapely.geometry import box
    from services import geo_utils
    polys = [box(-80.01, 40.43, -79.99, 40.45), box(-80.00 if overlapping else -79.99, 40.43, -79.97, 40.45)]
    gdf = gpd.GeoDataFrame({"tract_fips": ["42003040100", "42003050100"]}, geometry=polys, crs="EPSG:4326")
    monkeypatch.setattr(geo_utils, "_load_tracts_gdf", lambda: gdf)
    monkeypatch.setattr(geo_utils, "geometry_source", lambda: "test tract polygons")
    ob._GEOM_SOURCE.clear()


def test_coordinates_gain_a_tract_via_geo_utils_even_where_polygons_overlap(reference_data, tmp_path, monkeypatch):
    _tract_polygons(monkeypatch, overlapping=True)      # a point in both polygons breaks geo_utils' positional assignment
    p = tmp_path / "stations.csv"
    pd.DataFrame({"Station": list("ABC"), "Latitude": [40.44, 40.44, 40.90], "Longitude": [-79.995, -79.98, -79.0],
                  "Ridership": [1200, 800, 50]}).to_csv(p, index=False)
    ds = ob.create_dataset(p, "stations.csv")
    assert [e["kind"] for e in ds["available_enrichments"]] == ["spatial_tract"]
    ds = ob.enrich(ds["dataset_id"], "spatial_tract")
    assert ds["enrichment_result"]["matched"] == 2 and ds["enrichment_result"]["total"] == 3
    assert "tract_fips" in [c["name"] for c in ds["columns"]] and ds["available_enrichments"] == []
    ds = ob.analyze(ds["dataset_id"])
    link = _link(ds, "houses")
    assert link["payload"]["left_expr"] == "tract_fips" and link["payload"]["derive"] is None     # already a real column


def test_geocoding_can_never_write_to_sold_homes(reference_data, tmp_path, monkeypatch):
    calls = []

    def fake_geocode(df, **kw):
        calls.append(kw)
        out = df.copy()
        out["lat"], out["lon"], out["tract_fips"], out["geocode_status"] = 40.44, -80.0, "42003040100", "success"
        return out
    monkeypatch.setattr("services.geocoder.geocode_dataframe", fake_geocode)
    p = tmp_path / "permits.csv"
    pd.DataFrame({"sale_id": ["S1"], "Street": ["123 Main Street"], "City": ["Pittsburgh"], "Zip": ["15213"], "Permit": ["A1"]}).to_csv(p, index=False)
    ds = ob.create_dataset(p, "permits.csv")
    assert "geocode" in [e["kind"] for e in ds["available_enrichments"]]
    ds = ob.enrich(ds["dataset_id"], "geocode")
    assert calls and calls[0]["sale_id_col"] not in {c["name"] for c in ds["columns"]}     # 'sale_id' would UPDATE sold_homes
    assert calls[0]["single_fallback_limit"] == 0
    assert {"lat", "lon", "tract_fips"} <= {c["name"] for c in ds["columns"]}
    assert reference_data.execute("SELECT lat FROM sold_homes WHERE sale_id = 'S1'").fetchone()[0] is None


def test_text_keys_that_differ_only_by_case_are_matched_on_both_sides(reference_data, tmp_path):
    import db.duckdb_store as store
    p = tmp_path / "city_stats.csv"
    pd.DataFrame({"City": ["PITTSBURGH", "PHILADELPHIA", "CLEVELAND", "DENVER"], "Median Rent": [1500, 1700, 1100, 2000]}).to_csv(p, index=False)
    ds = ob.analyze(ob.create_dataset(p, "city_stats.csv")["dataset_id"])
    link = _link(ds, "houses", right_expr="UPPER(TRIM(city))")
    assert link["payload"]["derive"]["transform"] == {"kind": "ci"} and link["payload"]["left_expr"] == "city_norm"
    assert link["evidence"]["stats"]["right_covered_pct"] == 100.0
    ob.decide(_table(ds)["proposal_id"], "approve")
    ob.decide(link["proposal_id"], "approve")
    plan = schema.house_link_plan("city_stats")
    assert "city_stats.city_norm = UPPER(TRIM(houses.city))" in plan["sql"]
    assert store.query(plan["sql"], ["h1"])["median_rent"].tolist() == [1500]


def test_revoke_retire_and_discard_take_effect_immediately(walk_dataset, blockgroup_csv):
    import db.duckdb_store as store
    key = "smart_location_database:tract_fips=houses:tract_fips"
    assert key in [r.key() for r in schema.RELATIONSHIPS]
    with pytest.raises(ob.OnboardingError) as exc:
        ob.revoke_relationship(schema.RELATIONSHIPS[0].key())                       # built-in links are protected
    assert exc.value.status_code == 403
    ob.revoke_relationship(key, "wrong link")
    assert key not in [r.key() for r in schema.RELATIONSHIPS]
    plan = schema.house_link_plan("smart_location_database")          # still reachable through its other approved link
    assert plan and "smart_location_database.tract_fips = houses.tract_fips" not in plan["join_text"]
    ob.revoke_relationship("smart_location_database:tract_fips=nri_tracts:tract_fips")
    assert not schema.house_link_plan("smart_location_database")
    assert any(p["status"] == "revoked" for p in ob.detail(walk_dataset["dataset_id"])["proposals"])
    out = ob.retire_dataset(walk_dataset["dataset_id"])
    assert out["tables"] == 1 and "smart_location_database" not in schema.list_table_names()
    assert schema.added_datasets_briefing() == ""
    assert store.get_conn().execute("SELECT COUNT(*) FROM smart_location_database").fetchone()[0] > 0   # the data is kept
    draft = ob.create_dataset(blockgroup_csv, "another.csv")
    ob.discard_draft(draft["dataset_id"])
    assert not store.get_conn().execute("SELECT 1 FROM information_schema.tables WHERE table_name LIKE 'stg_%'").fetchall()
    assert draft["dataset_id"] not in [d["dataset_id"] for d in ob.list_datasets()]


def test_names_are_validated_and_alias_collisions_are_reported(reference_data, blockgroup_csv):
    ds = ob.create_dataset(blockgroup_csv, "smart.csv")
    for bad in ["Bad Name", "houses", "update", "stg_x", "ab", "9lives"]:
        with pytest.raises(ob.OnboardingError):
            ob.update_description(ds["dataset_id"], {"table_name": bad})
    with pytest.raises(ob.OnboardingError, match="one of"):
        ob.update_description(ds["dataset_id"], {"columns": [{"name": "natwalkind", "role": "nonsense"}]})
    with pytest.raises(ob.OnboardingError, match="at least one"):
        ob.update_description(ds["dataset_id"], {"columns": [{"name": c["name"], "include": False} for c in ds["columns"]]})
    ob.update_description(ds["dataset_id"], {"columns": [{"name": "natwalkind", "synonyms": ["walk score", "walkability index"]}]})
    table = _table(ob.analyze(ds["dataset_id"]))
    warnings = " ".join(table["evidence"]["warnings"])
    assert "walk score" in warnings and "house_walk_score" in warnings                 # an exact collision is dropped and reported
    aliases = table["evidence"]["aliases"]["smart_natwalkind"]
    assert "walk score" not in aliases and "walkability index" in aliases
    assert table["payload"]["concepts"]["smart_natwalkind"]["overrides"] == ["house_walk_score"]     # declared, not guessed
    assert any("takes precedence over the existing concept 'house_walk_score'" in n for n in table["evidence"]["notices"])


def test_editing_the_description_invalidates_stale_proposals(reference_data, blockgroup_csv):
    ds = ob.analyze(ob.create_dataset(blockgroup_csv, "smart.csv")["dataset_id"])
    assert any(p["status"] == "pending" for p in ds["proposals"])
    ds = ob.update_description(ds["dataset_id"], {"grain": "one row per block group"})
    assert ds["proposals"] == []

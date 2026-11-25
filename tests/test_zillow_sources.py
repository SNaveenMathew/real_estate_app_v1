"""
Unit tests for Zillow research data sources (ZHVI and Market Heat Index):
- Wide CSV parsing and melting (services/zillow_sources.py)
- Zip code normalization (normalize_zip5)
- RegionType normalization and column alignments
- CBSA / MSA code resolution (services/data_loader.py::_compute_zillow_msa_codes)
- DuckDB store ZHVI price estimation with 3-tier fallback (zip -> metro -> state)
- House Chat price estimator output (agents/house_agent.py::make_house_approved_functions)
- Unified catalog registration of tables, relationships, and concepts
"""

import io
import textwrap
from pathlib import Path
import pandas as pd
import pytest

from services.zillow_sources import (
    normalize_zip5,
    melt_wide_file,
    ZhviParser,
    MarketHeatIndexParser,
    ZILLOW_PARSERS,
)
from services import data_loader
import db.schema_catalog as schema
from db import catalog_seed as seed


# ─── 1. Zip Code Normalization Tests ─────────────────────────────────────────


def test_normalize_zip5_standard_and_variations():
    assert normalize_zip5("15213") == "15213"
    assert normalize_zip5(15213) == "15213"
    assert normalize_zip5(" 15213 ") == "15213"


def test_normalize_zip5_zip_plus_four():
    assert normalize_zip5("15213-2622") == "15213"
    assert normalize_zip5("02139-1234") == "02139"


def test_normalize_zip5_leading_zeros_restored():
    # If read as integer or truncated string, e.g. 2139 -> "02139"
    assert normalize_zip5(2139) == "02139"
    assert normalize_zip5("2139") == "02139"
    assert normalize_zip5(501) == "00501"


def test_normalize_zip5_invalid_and_empty():
    assert normalize_zip5(None) is None
    assert normalize_zip5("") is None
    assert normalize_zip5("   ") is None
    assert normalize_zip5("nan") is None
    assert normalize_zip5("None") is None
    assert normalize_zip5("<NA>") is None
    assert normalize_zip5("abcde") is None


# ─── 2. Wide CSV Parsing and Melting Tests ────────────────────────────────────


def test_melt_wide_file_success(tmp_path):
    csv_content = textwrap.dedent("""\
        RegionID,SizeRank,RegionName,RegionType,StateName,State,City,Metro,CountyName,2023-12-31,2024-01-31
        9999,1,15213,zip,PA,PA,Pittsburgh,"Pittsburgh, PA",Allegheny County,300000,310000
        8888,2,"Pittsburgh, PA",msa,PA,,Pittsburgh,"Pittsburgh, PA",Allegheny County,250000,260000
    """)
    file_path = tmp_path / "test_zhvi.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    df = melt_wide_file(file_path, value_name="home_value")
    assert df is not None
    assert len(df) == 4  # 2 regions x 2 dates

    # Check RegionType normalization: 'msa' -> 'metro'
    metro_row = df[df["region_id"] == "8888"].iloc[0]
    assert metro_row["region_type"] == "metro"

    # Check zip normalization on zip-level row
    zip_row = df[df["region_id"] == "9999"].iloc[0]
    assert zip_row["region_name"] == "15213"
    assert zip_row["region_type"] == "zip"

    # Check column types
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_numeric_dtype(df["home_value"])
    assert df["source_file"].iloc[0] == "test_zhvi.csv"


def test_melt_wide_file_drops_na_values(tmp_path):
    csv_content = textwrap.dedent("""\
        RegionID,RegionName,RegionType,2023-12-31,2024-01-31
        1001,TestRegion,city,,50000
    """)
    file_path = tmp_path / "test_missing.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    df = melt_wide_file(file_path, value_name="home_value")
    assert df is not None
    assert len(df) == 1
    assert df.iloc[0]["home_value"] == 50000.0


def test_melt_wide_file_missing_required_columns(tmp_path):
    csv_content = "ColA,ColB,2024-01-31\n1,2,3\n"
    file_path = tmp_path / "test_invalid.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    df = melt_wide_file(file_path, value_name="home_value")
    assert df is None


def test_melt_wide_file_no_date_columns(tmp_path):
    csv_content = "RegionID,RegionName,RegionType,Value\n1,Test,zip,100\n"
    file_path = tmp_path / "test_nodate.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    df = melt_wide_file(file_path, value_name="home_value")
    assert df is None


# ─── 3. Parser Configurations and Directory Loading ──────────────────────────


def test_parser_configurations():
    assert ZhviParser.key == "zhvi"
    assert ZhviParser.table == "zhvi"
    assert ZhviParser.value_column == "home_value"

    assert MarketHeatIndexParser.key == "market_heat_index"
    assert MarketHeatIndexParser.table == "market_heat_index"
    assert MarketHeatIndexParser.value_column == "heat_index"

    assert ZhviParser in ZILLOW_PARSERS
    assert MarketHeatIndexParser in ZILLOW_PARSERS


def test_zillow_load_dir_multiple_files(tmp_path):
    f1 = tmp_path / "zhvi_zip.csv"
    f1.write_text(
        textwrap.dedent("""\
        RegionID,RegionName,RegionType,2024-01-31
        101,15213,zip,350000
    """),
        encoding="utf-8",
    )

    f2 = tmp_path / "zhvi_metro.csv"
    f2.write_text(
        textwrap.dedent("""\
        RegionID,RegionName,RegionType,2024-01-31
        202,"Pittsburgh, PA",msa,240000
    """),
        encoding="utf-8",
    )

    # A non-csv file that should be ignored
    (tmp_path / "notes.txt").write_text("ignore me")

    combined = ZhviParser.load_dir(tmp_path)
    assert len(combined) == 2
    assert set(combined["region_id"]) == {"101", "202"}
    assert set(combined["region_type"]) == {"zip", "metro"}


def test_zip_level_export_shape_restores_leading_zero_and_keeps_metro(tmp_path):
    """The real Zip_zhvi_*.csv export: all identifying columns, one row per zip.

    Zillow's zip RegionName is a text zip, but a spreadsheet-edited or numerically re-saved copy loses the
    leading zero (Cambridge's 01851 arrives as 1851). The zip level is the level houses join to, so the
    parser has to produce a canonical 5-digit key, and it must leave the other levels' names alone.
    """
    csv_content = textwrap.dedent("""\
        RegionID,SizeRank,RegionName,RegionType,StateName,State,City,Metro,CountyName,2024-01-31,2024-02-29
        91982,0,2139,zip,MA,MA,Cambridge,"Boston-Cambridge-Newton, MA-NH",Middlesex County,780000,790000
        92000,1,Maryvale,neighborhood,PA,PA,Pittsburgh,"Pittsburgh, PA",Allegheny County,180000,181000
    """)
    path = tmp_path / "Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv"
    path.write_text(csv_content, encoding="utf-8")

    df = ZhviParser.load_dir(tmp_path)
    assert len(df) == 4  # 2 regions x 2 months

    zip_rows = df[df["region_type"] == "zip"]
    assert set(zip_rows["region_name"]) == {"02139"}
    assert zip_rows["metro"].iloc[0] == "Boston-Cambridge-Newton, MA-NH"

    neighborhood_rows = df[df["region_type"] == "neighborhood"]
    assert set(neighborhood_rows["region_name"]) == {"Maryvale"}


def test_zhvi_concepts_pin_exactly_one_geography_level():
    """A level-blind ZHVI concept is the grain bug waiting to happen.

    zhvi holds a zip export next to a neighborhood export, so `AVG(home_value)` without a
    `region_type` filter averages across grains. Every concept over zhvi.home_value must pin exactly
    one level, and no two may share it.
    """
    concepts = [c for c in seed.SEMANTIC_GLOSSARY.values() if "zhvi.home_value" in c["columns"]]
    assert {"zhvi", "zhvi_metro", "zhvi_neighborhood"} <= {c["key"] for c in concepts}

    levels = {}
    for concept in concepts:
        pinned = [f for f in concept["filters"] if "zhvi.region_type" in f]
        assert len(pinned) == 1, f"{concept['key']} must pin exactly one region_type, found {concept['filters']}"
        level = pinned[0].split("'")[1]
        assert level not in levels, f"{concept['key']} and {levels[level]} both pin region_type={level!r}"
        levels[level] = concept["key"]

    assert levels == {
        "zip": "zhvi",
        "metro": "zhvi_metro",
        "neighborhood": "zhvi_neighborhood",
    }


def test_zhvi_concepts_pin_the_join_their_level_needs():
    """A level's join key must match its grain.

    The table-level default (houses.zip = zhvi.region_name) is right only for zip rows: metro rows carry the
    metro's display name in region_name and a CBSA code in msa_code, and no house column holds a neighborhood
    name. The planner reads ``join``, so a wrong declaration there is silently the wrong join.
    """
    concepts = {c["key"]: c for c in seed.SEMANTIC_GLOSSARY.values()}
    registered = {r.join_key() for r in seed.RELATIONSHIPS}
    expected = {
        "zhvi": {"houses.zip=zhvi.region_name"},
        "zhvi_metro": {"houses.msa_code=zhvi.msa_code", "census_msa.msa_code=zhvi.msa_code"},
        "zhvi_neighborhood": set(),
    }
    for key, joins in expected.items():
        assert set(concepts[key]["join"]) == joins, f"{key} pins {concepts[key].get('join')!r}"
        assert joins <= registered, f"{key} pins a join the catalog does not register"

    # No house column holds a neighborhood name: the zip-key join the table-level default would pick returns no
    # rows at this level (measured 0 on the loaded export), so "joins to nothing" is the only correct declaration.
    assert "join" in concepts["zhvi_neighborhood"] and not concepts["zhvi_neighborhood"]["join"]


# ─── 4. CBSA / MSA Resolution Tests ──────────────────────────────────────────


def test_compute_zillow_msa_codes(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute("""
        INSERT INTO cbsa_counties (cbsa_code, cbsa_title, county_fips)
        VALUES ('38300', 'Pittsburgh, PA', '42003')
    """)

    sample_df = pd.DataFrame(
        [
            {"region_type": "metro", "region_name": "Pittsburgh, PA", "metro": None},
            {"region_type": "zip", "region_name": "15213", "metro": "Pittsburgh, PA"},
            {"region_type": "state", "region_name": "PA", "metro": None},
        ]
    )

    msa_codes = data_loader._compute_zillow_msa_codes(sample_df)
    assert msa_codes.iloc[0] == "38300"
    assert msa_codes.iloc[1] == "38300"
    assert pd.isna(msa_codes.iloc[2]) or msa_codes.iloc[2] is None


# ─── 5. DuckDB Store ZHVI Price Estimation & 3-Tier Fallback ──────────────────


def test_get_last_sold_snapshot(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute("""
        INSERT INTO houses (house_id, address, city, state, zip, lat, lon)
        VALUES ('h_test', '123 Test St', 'Pittsburgh', 'PA', '15213', 40.44, -80.0)
    """)
    assert fresh_db.get_last_sold_snapshot("h_test") is None

    conn.execute("""
        INSERT INTO house_snapshots (snapshot_id, house_id, snapshot_date, price, source_type)
        VALUES
            ('s1', 'h_test', '2020-05-15', 200000, 'sold'),
            ('s2', 'h_test', '2022-08-10', 250000, 'sold'),
            ('s3', 'h_test', '2023-01-01', 300000, 'listing')
    """)

    last_sale = fresh_db.get_last_sold_snapshot("h_test")
    assert last_sale is not None
    assert last_sale["sold_price"] == 250000
    assert str(last_sale["sold_date"]).startswith("2022-08-10")


def test_get_zhvi_price_estimate_tier1_zip(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute("""
        INSERT INTO houses (house_id, address, city, state, zip, msa_code, lat, lon)
        VALUES ('h_zip', '123 Main St', 'Pittsburgh', 'PA', '15213', '38300', 40.44, -80.0)
    """)
    conn.execute("""
        INSERT INTO house_snapshots (snapshot_id, house_id, snapshot_date, price, source_type)
        VALUES ('s_zip', 'h_zip', '2020-06-01', 200000, 'sold')
    """)
    conn.execute("""
        INSERT INTO zhvi (region_id, region_type, region_name, state_name, msa_code, date, home_value)
        VALUES
            ('r_zip', 'zip', '15213', 'PA', '38300', '2020-06-30', 200000),
            ('r_zip', 'zip', '15213', 'PA', '38300', '2024-06-30', 260000)
    """)

    est = fresh_db.get_zhvi_price_estimate("h_zip")
    assert est is not None
    assert est["geography_level"] == "zip"
    assert est["geography_key"] == "15213"
    assert est["sold_price"] == 200000
    assert est["value_at_sale"] == 200000
    assert est["latest_value"] == 260000
    assert est["growth_multiple"] == pytest.approx(1.3, rel=1e-3)
    assert est["estimated_value"] == pytest.approx(260000, rel=1e-3)


def test_get_zhvi_price_estimate_tier2_metro_fallback(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute("""
        INSERT INTO houses (house_id, address, city, state, zip, msa_code, lat, lon)
        VALUES ('h_metro', '456 Metro St', 'Pittsburgh', 'PA', '15299', '38300', 40.44, -80.0)
    """)
    conn.execute("""
        INSERT INTO house_snapshots (snapshot_id, house_id, snapshot_date, price, source_type)
        VALUES ('s_metro', 'h_metro', '2019-01-15', 300000, 'sold')
    """)
    # No zip-level row for 15299, but metro-level row for 38300
    conn.execute("""
        INSERT INTO zhvi (region_id, region_type, region_name, state_name, msa_code, date, home_value)
        VALUES
            ('r_metro', 'metro', 'Pittsburgh, PA', 'PA', '38300', '2019-01-31', 200000),
            ('r_metro', 'metro', 'Pittsburgh, PA', 'PA', '38300', '2024-01-31', 240000)
    """)

    est = fresh_db.get_zhvi_price_estimate("h_metro")
    assert est is not None
    assert est["geography_level"] == "metro"
    assert est["geography_key"] == "38300"
    assert est["sold_price"] == 300000
    assert est["growth_multiple"] == pytest.approx(1.2, rel=1e-3)
    assert est["estimated_value"] == pytest.approx(360000, rel=1e-3)


def test_get_zhvi_price_estimate_tier3_state_fallback(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute("""
        INSERT INTO houses (house_id, address, city, state, zip, msa_code, lat, lon)
        VALUES ('h_state', '789 Rural Rd', 'Smalltown', 'PA', '16999', NULL, 40.44, -80.0)
    """)
    conn.execute("""
        INSERT INTO house_snapshots (snapshot_id, house_id, snapshot_date, price, source_type)
        VALUES ('s_state', 'h_state', '2021-03-01', 150000, 'sold')
    """)
    # No zip or metro row, only state-level row
    conn.execute("""
        INSERT INTO zhvi (region_id, region_type, region_name, state_name, msa_code, date, home_value)
        VALUES
            ('r_pa', 'state', 'Pennsylvania', 'PA', NULL, '2021-03-31', 250000),
            ('r_pa', 'state', 'Pennsylvania', 'PA', NULL, '2024-03-31', 300000)
    """)

    est = fresh_db.get_zhvi_price_estimate("h_state")
    assert est is not None
    assert est["geography_level"] == "state"
    assert est["geography_key"] == "PA"
    assert est["sold_price"] == 150000
    assert est["growth_multiple"] == pytest.approx(1.2, rel=1e-3)
    assert est["estimated_value"] == pytest.approx(180000, rel=1e-3)


def test_get_zhvi_price_estimate_none_when_no_data_or_no_sale(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute("""
        INSERT INTO houses (house_id, address, city, state, zip, lat, lon)
        VALUES ('h_none', 'No Sale St', 'Pittsburgh', 'PA', '15213', 40.44, -80.0)
    """)
    # No sale snapshot
    assert fresh_db.get_zhvi_price_estimate("h_none") is None

    # Sale exists, but no ZHVI at any level
    conn.execute("""
        INSERT INTO house_snapshots (snapshot_id, house_id, snapshot_date, price, source_type)
        VALUES ('s_none', 'h_none', '2020-01-01', 200000, 'sold')
    """)
    assert fresh_db.get_zhvi_price_estimate("h_none") is None


# ─── 6. House Chat Price Estimation Output ────────────────────────────────────
#
# These tests target agents.house_agent.make_house_approved_functions — the function
# set House Chat actually executes. They used to target agents.tools.make_house_tools,
# an unused parallel copy, which is how the ZHVI estimate documented in the README
# stayed unreachable in the product path while this file passed.


def _seed_house_with_zhvi(store, house_id: str) -> None:
    """A house with one recorded sale and a ZIP-level ZHVI series (1.25x growth)."""
    conn = store.get_conn()
    conn.execute(
        "INSERT INTO houses (house_id, address, city, state, zip, tract_fips, lat, lon) "
        "VALUES (?, '10 Elm St', 'Pittsburgh', 'PA', '15213', '42003040100', 40.44, -80.0)",
        [house_id],
    )
    conn.execute(
        "INSERT INTO house_snapshots (snapshot_id, house_id, snapshot_date, price, source_type) "
        "VALUES (?, ?, '2021-01-10', 250000, 'sold')",
        [f"s_{house_id}", house_id],
    )
    conn.execute(
        "INSERT INTO zhvi (region_id, region_type, region_name, state_name, date, home_value) "
        "VALUES ('z1', 'zip', '15213', 'PA', '2021-01-31', 200000), "
        "('z1', 'zip', '15213', 'PA', '2024-01-31', 250000)"
    )


def test_house_chat_price_estimator_includes_zhvi(fresh_db):
    from agents.house_agent import make_house_approved_functions

    house_id = "h10"
    _seed_house_with_zhvi(fresh_db, house_id)

    output = make_house_approved_functions(house_id)["estimate_price_with_code"]()

    assert "### ZHVI-Adjusted Estimate" in output
    assert "Last recorded sale: $250,000" in output
    assert "ZIP 15213" in output
    assert "Growth since sale: 1.250x" in output
    assert "Estimated value (last sold price x growth): $312,500" in output


def test_house_chat_price_estimator_sold_without_zhvi_shows_guidance(fresh_db):
    from agents.house_agent import make_house_approved_functions

    house_id = "h11"
    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO houses (house_id, address, city, state, zip, tract_fips, lat, lon) "
        "VALUES (?, '11 Elm St', 'Pittsburgh', 'PA', '15213', '42003040100', 40.44, -80.0)",
        [house_id],
    )
    conn.execute(
        "INSERT INTO house_snapshots (snapshot_id, house_id, snapshot_date, price, source_type) "
        "VALUES (?, ?, '2021-01-10', 250000, 'sold')",
        [f"s_{house_id}", house_id],
    )

    output = make_house_approved_functions(house_id)["estimate_price_with_code"]()

    assert "### ZHVI-Adjusted Estimate" in output
    assert "python setup_data.py --only zhvi" in output


def test_house_chat_turn_reports_the_zhvi_estimate(fresh_db, monkeypatch):
    """One golden example through House Chat: a ZHVI-derived number must reach the user.

    The code agent is scripted to call the price estimator (the documented "what is this
    house worth" path), and the answer model echoes the evidence it was handed — so this
    asserts the value survives the whole turn, not merely the tool call.
    """
    from agents import house_agent
    from services.guardrails import GuardrailManager, GuardrailResult

    house_id = "h12"
    _seed_house_with_zhvi(fresh_db, house_id)

    class _CodeLLM:
        def invoke(self, messages):
            class _R:
                content = "final_result = estimate_price_with_code()"

            return _R()

    class _EvidenceEchoLLM:
        def invoke(self, messages):
            evidence = str(messages[0].content).split("EXECUTED EVIDENCE:", 1)[-1]

            class _R:
                content = evidence.strip()

            return _R()

    monkeypatch.setattr(house_agent, "_get_house_code_agent", lambda: _CodeLLM())
    monkeypatch.setattr(house_agent, "_get_house_response_agent", lambda: _EvidenceEchoLLM())
    monkeypatch.setattr(
        GuardrailManager,
        "inspect_turn_input",
        staticmethod(lambda msg: GuardrailResult(passed=True, sanitized_text=msg, reasons=[])),
    )
    monkeypatch.setattr(house_agent, "validate_response", lambda reply, *a, **kw: reply)

    reply, history, _artifacts = house_agent.run_house_chat(
        house_id, "What is this house worth after its last sale and local home-value growth?"
    )

    assert "ZHVI-Adjusted Estimate" in reply
    assert "Estimated value (last sold price x growth): $312,500" in reply
    assert any(m.get("role") == "assistant" for m in history)


# ─── 7. Unified Catalog Registration Tests ────────────────────────────────────


def test_zhvi_and_heat_index_in_catalog(fresh_db):
    assert "zhvi" in schema.TABLES
    assert "market_heat_index" in schema.TABLES

    zhvi_table = schema.TABLES["zhvi"]
    assert "home_value" in [c.column for c in zhvi_table.column_notes]
    assert "msa_code" in [c.column for c in zhvi_table.column_notes]

    heat_table = schema.TABLES["market_heat_index"]
    assert "heat_index" in [c.column for c in heat_table.column_notes]

    # Verify relationships are registered
    rel_keys = {r.key() for r in schema.RELATIONSHIPS}
    assert "houses:zip=zhvi:region_name" in rel_keys
    assert "houses:msa_code=zhvi:msa_code" in rel_keys
    assert "census_msa:msa_code=zhvi:msa_code" in rel_keys
    assert "houses:zip=market_heat_index:region_name" in rel_keys
    assert "houses:msa_code=market_heat_index:msa_code" in rel_keys

    # Verify semantic concepts
    assert "zhvi" in schema.SEMANTIC_GLOSSARY
    assert "market_heat_index" in schema.SEMANTIC_GLOSSARY

    matches = schema.semantic_matches("what is the zhvi home value index trend")
    assert any(m["key"] == "zhvi" for m in matches)


# ─── 8. Data Sources Registry & Refresh Orchestration Tests ──────────────────


def test_zhvi_and_heat_index_in_data_sources_registry():
    from services import data_sources as ds

    assert "zhvi" in ds.REGISTRY
    assert "market_heat_index" in ds.REGISTRY

    zhvi_ds = ds.REGISTRY["zhvi"]
    assert zhvi_ds.table == "zhvi"
    assert zhvi_ds.placement == "append"
    assert zhvi_ds.accept == ".csv"
    assert callable(zhvi_ds.precheck)
    assert ds.sources_for_table("zhvi") == ["zhvi"]

    heat_ds = ds.REGISTRY["market_heat_index"]
    assert heat_ds.table == "market_heat_index"
    assert heat_ds.placement == "append"
    assert callable(heat_ds.precheck)
    assert ds.sources_for_table("market_heat_index") == ["market_heat_index"]


def test_zhvi_refresh_orchestration_and_precheck(fresh_db, tmp_path, monkeypatch):
    from services import data_sources as ds
    from dataclasses import replace

    from config import settings

    zhvi_dir = tmp_path / "zhvi"
    zhvi_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "zhvi_dir", zhvi_dir)
    monkeypatch.setitem(ds.REGISTRY, "zhvi", replace(ds.REGISTRY["zhvi"], dest_dir=zhvi_dir))

    # 1. Reject invalid file via precheck
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("Col1,Col2\n1,2\n", encoding="utf-8")
    result = ds.refresh_source("zhvi", bad_file, "bad.csv")
    assert result["ok"] is False
    assert "Missing required Zillow column" in result["error"] or "No date columns" in result["error"]

    # 2. Accept and load valid wide CSV
    good_file = tmp_path / "good.csv"
    good_file.write_text(
        textwrap.dedent("""\
        RegionID,RegionName,RegionType,StateName,2024-01-31
        999,15213,zip,PA,320000
    """),
        encoding="utf-8",
    )
    res = ds.refresh_source("zhvi", good_file, "good.csv")
    assert res["ok"] is True
    assert res["rows_after"] >= 1
    assert fresh_db.get_conn().execute("SELECT COUNT(*) FROM zhvi").fetchone()[0] == 1

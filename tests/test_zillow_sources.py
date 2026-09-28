"""
Unit tests for Zillow research data sources (ZHVI and Market Heat Index):
- Wide CSV parsing and melting (services/zillow_sources.py)
- Zip code normalization (normalize_zip5)
- RegionType normalization and column alignments
- CBSA / MSA code resolution (services/data_loader.py::_compute_zillow_msa_codes)
- DuckDB store ZHVI price estimation with 3-tier fallback (zip -> metro -> state)
- House agent price estimator tool output (agents/tools.py::make_house_tools)
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
    f1.write_text(textwrap.dedent("""\
        RegionID,RegionName,RegionType,2024-01-31
        101,15213,zip,350000
    """), encoding="utf-8")

    f2 = tmp_path / "zhvi_metro.csv"
    f2.write_text(textwrap.dedent("""\
        RegionID,RegionName,RegionType,2024-01-31
        202,"Pittsburgh, PA",msa,240000
    """), encoding="utf-8")

    # A non-csv file that should be ignored
    (tmp_path / "notes.txt").write_text("ignore me")

    combined = ZhviParser.load_dir(tmp_path)
    assert len(combined) == 2
    assert set(combined["region_id"]) == {"101", "202"}
    assert set(combined["region_type"]) == {"zip", "metro"}


# ─── 4. CBSA / MSA Resolution Tests ──────────────────────────────────────────

def test_compute_zillow_msa_codes(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute("""
        INSERT INTO cbsa_counties (cbsa_code, cbsa_title, county_fips)
        VALUES ('38300', 'Pittsburgh, PA', '42003')
    """)

    sample_df = pd.DataFrame([
        {"region_type": "metro", "region_name": "Pittsburgh, PA", "metro": None},
        {"region_type": "zip", "region_name": "15213", "metro": "Pittsburgh, PA"},
        {"region_type": "state", "region_name": "PA", "metro": None},
    ])

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


# ─── 6. House Agent Price Estimation Tool Output ──────────────────────────────

def test_make_house_tools_price_estimator_with_zhvi(fresh_db):
    from agents.tools import make_house_tools

    conn = fresh_db.get_conn()
    conn.execute("""
        INSERT INTO houses (house_id, address, city, state, zip, tract_fips, lat, lon)
        VALUES ('h10', '10 Elm St', 'Pittsburgh', 'PA', '15213', '42003040100', 40.44, -80.0)
    """)
    conn.execute("""
        INSERT INTO house_snapshots (snapshot_id, house_id, snapshot_date, price, source_type)
        VALUES ('s10', 'h10', '2021-01-10', 250000, 'sold')
    """)
    conn.execute("""
        INSERT INTO zhvi (region_id, region_type, region_name, state_name, date, home_value)
        VALUES
            ('z1', 'zip', '15213', 'PA', '2021-01-31', 200000),
            ('z1', 'zip', '15213', 'PA', '2024-01-31', 250000)
    """)

    tools = {t.name: t for t in make_house_tools("h10")}
    estimate_tool = tools["estimate_price_with_code"]
    output = estimate_tool.invoke({})

    assert "### ZHVI-Adjusted Estimate" in output
    assert "Last recorded sale: $250,000" in output
    assert "ZIP 15213" in output
    assert "Growth since sale: 1.250x" in output
    assert "Estimated value (last sold price × growth): $312,500" in output


def test_make_house_tools_price_estimator_sold_without_zhvi_shows_guidance(fresh_db):
    from agents.tools import make_house_tools

    conn = fresh_db.get_conn()
    conn.execute("""
        INSERT INTO houses (house_id, address, city, state, zip, tract_fips, lat, lon)
        VALUES ('h11', '11 Elm St', 'Pittsburgh', 'PA', '15213', '42003040100', 40.44, -80.0)
    """)
    conn.execute("""
        INSERT INTO house_snapshots (snapshot_id, house_id, snapshot_date, price, source_type)
        VALUES ('s11', 'h11', '2021-01-10', 250000, 'sold')
    """)

    tools = {t.name: t for t in make_house_tools("h11")}
    output = tools["estimate_price_with_code"].invoke({})

    assert "### ZHVI-Adjusted Estimate" in output
    assert "python setup_data.py --only zhvi" in output


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

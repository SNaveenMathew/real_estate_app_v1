import duckdb
import pandas as pd

from db import schema_ddl
from services import data_loader


_CRIME_COLUMNS = (
    "incident_id, city, source_file, occurred_at, year, month, year_month, lat, lon, "
    "category, category_label, severity_weight, raw_type, location_text"
)


def _insert_crime(conn, incident_id, city):
    conn.execute(
        f"INSERT INTO crime_incidents ({_CRIME_COLUMNS}) VALUES "
        "(?, ?, 'old.csv', TIMESTAMP '2005-01-01 00:00:00', 2005, 1, '2005-01', "
        "40.4, -80.0, 'other', 'Other / Unclassified', 1.0, 'OTHER', NULL)",
        [incident_id, city],
    )


def test_crime_incident_id_normalizes_numeric_natural_ids():
    integer_id = data_loader._crime_incident_id("pittsburgh", "one.xlsx", 2075487, 0)
    float_id = data_loader._crime_incident_id("pittsburgh", "two.csv", 2075487.0, 1)

    assert integer_id == float_id


def test_load_crime_replaces_stale_rows_for_loaded_cities(tmp_path, monkeypatch):
    city_dir = tmp_path / "crime" / "pittsburgh"
    city_dir.mkdir(parents=True)
    first_file = city_dir / "a.csv"
    duplicate_file = city_dir / "b.csv"
    header = "PK,INCIDENTTIME,X,Y,OFFENSES\n"
    first_file.write_text(
        header + "101,2005-01-01 08:00:00,-80.0,40.4,Theft\n" + "102,2005-01-02 08:00:00,-80.1,40.5,Assault\n",
        encoding="utf-8",
    )
    duplicate_file.write_text(
        header + "101.0,2005-01-01 08:00:00,-80.0,40.4,Theft\n",
        encoding="utf-8",
    )

    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE crime_incidents (
            incident_id VARCHAR PRIMARY KEY, city VARCHAR, source_file VARCHAR,
            occurred_at TIMESTAMP, year INTEGER, month INTEGER, year_month VARCHAR,
            lat DOUBLE, lon DOUBLE, category VARCHAR, category_label VARCHAR,
            severity_weight DOUBLE, raw_type VARCHAR, location_text VARCHAR
        )
    """)
    _insert_crime(conn, "stale-pittsburgh", "pittsburgh")
    _insert_crime(conn, "other-city", "boston")
    monkeypatch.setattr(data_loader.settings, "data_dir", tmp_path)
    monkeypatch.setattr(data_loader.store, "get_conn", lambda: conn)

    assert data_loader.load_crime() == 2
    pittsburgh_ids = conn.execute(
        "SELECT incident_id FROM crime_incidents WHERE city = 'pittsburgh' ORDER BY incident_id"
    ).fetchall()
    assert {row[0] for row in pittsburgh_ids} == {
        data_loader._crime_incident_id("pittsburgh", "a.csv", 101, 0),
        data_loader._crime_incident_id("pittsburgh", "a.csv", 102, 1),
    }
    assert conn.execute("SELECT count(*) FROM crime_incidents WHERE city = 'boston'").fetchone()[0] == 1

    first_file.write_text(header + "101,2005-01-01 08:00:00,-80.0,40.4,Theft\n", encoding="utf-8")
    duplicate_file.unlink()
    assert data_loader.load_crime() == 1
    assert conn.execute("SELECT count(*) FROM crime_incidents WHERE city = 'pittsburgh'").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM crime_incidents WHERE city = 'boston'").fetchone()[0] == 1
    conn.close()


def test_load_census_tracts_stores_join_key_in_schema_column_and_replaces_old_rows(tmp_path, monkeypatch):
    census_dir = tmp_path / "census"
    census_dir.mkdir()
    census_csv = census_dir / "DECENNIALPL2020.P1-Data.csv"
    census_csv.write_text(
        "GEO_ID,NAME,P1_001N\n"
        "Geography,Geographic Area Name,!!Total:\n"
        "1400000US01001020100,Census Tract 201,1775\n"
        "1400000US42003140100,Census Tract 1401,3456\n",
        encoding="utf-8",
    )

    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE census_tracts (
            tract_fips VARCHAR PRIMARY KEY,
            geo_id VARCHAR,
            name VARCHAR,
            population INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO census_tracts VALUES (?, ?, ?, ?)",
        ["1400000US01001020100", "01001020100", "Old swapped row", 1775],
    )
    monkeypatch.setattr(data_loader.settings, "census_tract_csv", census_csv)
    monkeypatch.setattr(data_loader.store, "get_conn", lambda: conn)

    assert data_loader.load_census_tracts() == 2
    rows = conn.execute("SELECT tract_fips, geo_id, population FROM census_tracts ORDER BY tract_fips").fetchall()
    assert rows == [
        ("01001020100", "1400000US01001020100", 1775),
        ("42003140100", "1400000US42003140100", 3456),
    ]
    conn.close()


def test_zip_frame_normalization_pads_and_cleans():
    from db.duckdb_store import _normalize_zip_frame

    df = pd.DataFrame({"zip": ["1851", "15213-2622", "15213", None, "not-a-zip"]})
    out = _normalize_zip_frame(df, "houses")

    assert list(out["zip"]) == ["01851", "15213", "15213", None, None]
    assert df["zip"].iloc[0] == "1851", "the caller's frame must not be mutated"


def test_zip_frame_normalization_leaves_non_us_tables_alone():
    from db.duckdb_store import _normalize_zip_frame

    df = pd.DataFrame({"zip": ["SW1A 1AA"]})
    assert _normalize_zip_frame(df, "uploaded_dataset").equals(df)


def test_repair_zip_codes_pads_legacy_rows_exactly_once():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE app_settings (key VARCHAR PRIMARY KEY, value_json VARCHAR)")
    conn.execute("CREATE TABLE houses (zip VARCHAR)")
    conn.execute("CREATE TABLE sold_homes (zip VARCHAR)")
    conn.execute("INSERT INTO houses VALUES ('1851'), ('15213'), (NULL), ('15213-2622')")
    conn.execute("INSERT INTO sold_homes VALUES ('2119')")

    assert schema_ddl.repair_zip_codes(conn) == 2
    assert conn.execute("SELECT zip FROM houses ORDER BY zip NULLS LAST").fetchall() == [
        ("01851",),
        ("15213",),
        ("15213-2622",),
        (None,),
    ]
    assert conn.execute("SELECT zip FROM sold_homes").fetchall() == [("02119",)]
    # The marker makes the second pass a no-op, so a reconnect never rescans the tables.
    assert schema_ddl.repair_zip_codes(conn) == 0
    conn.close()


# ── Census geography: Connecticut tract codes and houses.msa_code ─────────────


def test_house_msa_codes_resolve_through_the_county_crosswalk(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO cbsa_counties (cbsa_code, cbsa_title, state_fips, county_fips) "
        "VALUES ('35300', 'New Haven, CT', '09', '170')"
    )
    codes = data_loader._house_msa_codes(pd.Series(["09170141600", "06037000100", None]))

    assert list(codes) == ["35300", None, None]


def test_house_msa_codes_warn_when_the_crosswalk_is_not_loaded(fresh_db, capsys):
    assert list(data_loader._house_msa_codes(pd.Series(["09170141600"]))) == [None]

    out = capsys.readouterr().out
    assert "cbsa_counties is empty" in out and "--only census" in out


def test_backfill_house_msa_codes_fills_only_empty_rows(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO cbsa_counties (cbsa_code, cbsa_title, state_fips, county_fips) "
        "VALUES ('38300', 'Pittsburgh, PA', '42', '003')"
    )
    conn.execute(
        "INSERT INTO houses (house_id, address, city, state, tract_fips, msa_code) "
        "VALUES ('h1', 'a', 'Pittsburgh', 'PA', '42003040100', NULL)"
    )
    conn.execute(
        "INSERT INTO houses (house_id, address, city, state, tract_fips, msa_code) "
        "VALUES ('h2', 'b', 'Pittsburgh', 'PA', '42003040100', '11111')"
    )

    assert schema_ddl.backfill_house_msa_codes(conn) == 1
    assert conn.execute("SELECT house_id, msa_code FROM houses ORDER BY 1").fetchall() == [
        ("h1", "38300"),
        ("h2", "11111"),  # a code the load already resolved is never overwritten
    ]
    assert schema_ddl.backfill_house_msa_codes(conn) == 0  # idempotent


def test_ct_tract_codes_are_rekeyed_from_the_current_vintage_tracts(fresh_db):
    """CT moved to planning-region county codes in 2022; the census file still uses 090xx."""
    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO nri_tracts (tract_fips, county_fips, state_fips, county_name, state_name, risk_score) "
        "VALUES ('09170141600', '09170', '09', 'New Haven', 'CT', 50.0)"
    )
    conn.execute(
        "INSERT INTO census_tracts (tract_fips, geo_id, name, population) "
        "VALUES ('09009141600', '1400000US09009141600', 'Census Tract 1416, New Haven County, CT', 5139)"
    )
    conn.execute(
        "INSERT INTO census_tracts (tract_fips, geo_id, name, population) "
        "VALUES ('09001000100', 'x', 'Census Tract 1, Fairfield County, CT', 100)"
    )

    assert schema_ddl.repair_ct_tract_codes(conn) == 1
    assert conn.execute("SELECT tract_fips FROM census_tracts ORDER BY 1").fetchall() == [
        ("09001000100",),  # no current-vintage row for this number: left alone, never guessed
        ("09170141600",),
    ]
    assert schema_ddl.repair_ct_tract_codes(conn) == 0  # idempotent


def test_ct_tract_repair_waits_for_a_current_vintage_source(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO census_tracts (tract_fips, geo_id, name, population) "
        "VALUES ('09009141600', 'x', 'Census Tract 1416', 5139)"
    )

    assert schema_ddl.repair_ct_tract_codes(conn) == 0
    assert conn.execute("SELECT tract_fips FROM census_tracts").fetchall() == [("09009141600",)]


def test_ct_house_keeps_the_whole_census_chain_and_an_msa_code(fresh_db):
    """The 2022 code change must not drop a Connecticut house out of tract -> county -> CBSA."""
    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO nri_tracts (tract_fips, county_fips, state_fips, county_name, state_name, risk_score) "
        "VALUES ('09170141600', '09170', '09', 'New Haven', 'CT', 50.0)"
    )
    conn.execute(
        "INSERT INTO census_tracts (tract_fips, geo_id, name, population) "
        "VALUES ('09009141600', 'x', 'Census Tract 1416', 5139)"
    )
    conn.execute(
        "INSERT INTO cbsa_counties (cbsa_code, cbsa_title, state_fips, county_fips) "
        "VALUES ('35300', 'New Haven, CT', '09', '170')"
    )
    conn.execute(
        "INSERT INTO houses (house_id, address, city, state, tract_fips) "
        "VALUES ('h-ct', '1 Elm', 'New Haven', 'CT', '09170141600')"
    )

    schema_ddl.repair_ct_tract_codes(conn)
    assert schema_ddl.backfill_house_msa_codes(conn) == 1

    joined = conn.execute(
        """
        SELECT h.house_id, c.cbsa_code
        FROM houses h
        JOIN census_tracts ct ON ct.tract_fips = h.tract_fips
        JOIN cbsa_counties c ON c.state_fips || c.county_fips = LEFT(ct.tract_fips, 5)
        """
    ).fetchall()
    assert joined == [("h-ct", "35300")]


def test_load_redfin_fills_msa_code_while_loading_houses(fresh_db, tmp_path, monkeypatch):
    """The export has no MSA column: the loader must fill the key from tract -> county -> CBSA."""
    from config import settings

    redfin_dir = tmp_path / "redfin"
    redfin_dir.mkdir()
    (redfin_dir / "redfin_2024-01-15.csv").write_text(
        "address,city,state,zip,price,lat,lon\n1 Elm Street,New Haven,CT,06511,250000,41.3083,-72.9279\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "redfin_dir", redfin_dir)

    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO cbsa_counties (cbsa_code, cbsa_title, state_fips, county_fips) "
        "VALUES ('35300', 'New Haven, CT', '09', '170')"
    )

    class FakeGeo:
        @staticmethod
        def assign_tract_fips(frame):
            out = frame.copy()
            out["tract_fips"] = "09170141600"  # the geocoder's 2022 planning-region code
            return out

    assert data_loader.load_redfin(FakeGeo) == 1
    assert conn.execute("SELECT msa_code FROM houses").fetchall() == [("35300",)]

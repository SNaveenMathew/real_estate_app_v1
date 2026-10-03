import duckdb

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
        header
        + "101,2005-01-01 08:00:00,-80.0,40.4,Theft\n"
        + "102,2005-01-02 08:00:00,-80.1,40.5,Assault\n",
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
    rows = conn.execute(
        "SELECT tract_fips, geo_id, population FROM census_tracts ORDER BY tract_fips"
    ).fetchall()
    assert rows == [
        ("01001020100", "1400000US01001020100", 1775),
        ("42003140100", "1400000US42003140100", 3456),
    ]
    conn.close()
"""The catalog's join claims must be verifiable against the data (db/join_health.py)."""

from __future__ import annotations

from db import join_health
from db.catalog_model import Relationship


def test_every_catalog_relationship_resolves_on_a_fresh_database(fresh_db):
    """Structural check: no relationship may reference a missing table/column or an incompatible key type."""
    report = join_health.report()

    structural = [f for f in report.errors if "does not exist" in f.message or "incompatible" in f.message]
    assert structural == []


def test_transit_keys_join_on_the_agency_reporting_unit(fresh_db):
    """transit_agencies is one row per (ntd_id, mode, tos); a ridership row must match exactly one of them."""
    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO transit_agencies (ntd_id, agency, mode, tos) VALUES "
        "('00001', 'A', 'MB', 'DO'), ('00001', 'A', 'LR', 'DO')"
    )
    conn.execute(
        "INSERT INTO transit_ridership (ntd_id, agency, mode, tos, date, upt) "
        "VALUES ('00001', 'A', 'MB', 'DO', DATE '2020-01-01', 10)"
    )

    rel = next(r for r in join_health.schema.RELATIONSHIPS if r.left_table == "transit_ridership")
    assert "mode" in rel.left_expr and "tos" in rel.left_expr
    report = join_health.Report()
    join_health.audit_relationship(rel, report)
    assert [f for f in report.errors] == []


def test_audit_flags_a_join_that_fans_out(fresh_db):
    """A many-to-one claim whose right key repeats is an error: every left row would be duplicated."""
    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO transit_agencies (ntd_id, agency, mode, tos) VALUES "
        "('00001', 'A', 'MB', 'DO'), ('00001', 'A', 'LR', 'DO')"
    )
    conn.execute(
        "INSERT INTO transit_ridership (ntd_id, agency, mode, tos, date, upt) "
        "VALUES ('00001', 'A', 'MB', 'DO', DATE '2020-01-01', 10)"
    )
    naive = Relationship(
        "transit_ridership", "ntd_id", "transit_agencies", "ntd_id", "naive join", "many-to-one"
    )

    report = join_health.Report()
    join_health.audit_relationship(naive, report)

    assert any("right key repeats" in f.message for f in report.errors)


def test_scoped_join_measures_only_the_latest_loaded_period(fresh_db):
    """A monthly table must not look many-to-many when the catalog pins the current month."""
    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO houses (house_id, address, city, state, zip, lat, lon) "
        "VALUES ('h1', '1 Main St', 'Pittsburgh', 'PA', '15213', 40.44, -80.0)"
    )
    conn.execute(
        "INSERT INTO zhvi (region_id, region_type, region_name, date, home_value) VALUES "
        "('z', 'zip', '15213', DATE '2024-01-31', 200000), "
        "('z', 'zip', '15213', DATE '2024-02-29', 210000)"
    )

    rel = next(r for r in join_health.schema.RELATIONSHIPS if r.key() == "houses:zip=zhvi:region_name")
    report = join_health.Report()
    join_health.audit_relationship(rel, report)

    assert [f for f in report.errors if "right key repeats" in f.message] == []
    assert any(f.severity == "info" and "scoped join" in f.message for f in report.findings)

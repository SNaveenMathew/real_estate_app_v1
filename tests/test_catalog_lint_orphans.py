"""Orphan join keys: the catalog declares 25 relationships, DuckDB enforces none of them.

A house whose `tract_fips` is not in `census_tracts` is dropped by every NRI/census join and
the answer that comes back is still a confident number - over the surviving rows. This check
makes that visible with a count.
"""

from __future__ import annotations

from db import catalog_lint


def _insert_house(conn, house_id: str, tract: str) -> None:
    conn.execute("INSERT INTO houses (house_id, city, tract_fips) VALUES (?, 'Pittsburgh', ?)", [house_id, tract])


def test_a_clean_fixture_has_no_orphans(reference_data):
    assert catalog_lint.orphan_join_keys() == []


def test_flags_a_house_whose_tract_does_not_exist(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute("INSERT INTO census_tracts (tract_fips, geo_id, name, population) VALUES ('42003040100','x','T',1)")
    _insert_house(conn, "h-ok", "42003040100")
    _insert_house(conn, "h-orphan", "99999999999")

    issues = catalog_lint.orphan_join_keys()
    orphan = [i for i in issues if i.code == "orphan_join_key" and i.subject == "houses.tract_fips"]
    assert len(orphan) == 1
    assert orphan[0].severity == "error" or orphan[0].severity == "warning"
    assert "1 of 2 rows (50.0%)" in orphan[0].message


def test_a_majority_of_orphans_is_an_error_and_a_minority_a_warning(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute("INSERT INTO census_tracts (tract_fips, geo_id, name, population) VALUES ('42003040100','x','T',1)")
    _insert_house(conn, "h-1", "42003040100")
    _insert_house(conn, "h-2", "99999999999")
    _insert_house(conn, "h-3", "88888888888")

    issues = {i.subject: i for i in catalog_lint.orphan_join_keys()}
    assert issues["houses.tract_fips"].severity == "error"  # 2 of 3

    conn.execute("DELETE FROM houses WHERE house_id = 'h-3'")
    _insert_house(conn, "h-4", "42003040100")
    issues = {i.subject: i for i in catalog_lint.orphan_join_keys()}
    assert issues["houses.tract_fips"].severity == "warning"  # 1 of 3


def test_rows_without_the_key_are_not_orphans(fresh_db):
    conn = fresh_db.get_conn()
    conn.execute("INSERT INTO houses (house_id, city) VALUES ('h-no-tract', 'Pittsburgh')")
    assert catalog_lint.orphan_join_keys() == []

"""db/transforms.py: one registry for the UI, the proposer and the catalog's join metadata."""

from __future__ import annotations

import pytest

import db.schema_catalog as schema
from db import transforms as tx


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "identity"},
        {"kind": "trim"},
        {"kind": "lower"},
        {"kind": "upper"},
        {"kind": "ci"},
        {"kind": "zfill", "width": 5},
        {"kind": "left", "width": 11},
        {"kind": "right", "width": 11},
        {"kind": "digits"},
    ],
)
def test_every_transform_round_trips_through_its_sql(spec):
    expr = tx.sql(spec, "c")
    parsed = tx.parse_expression(expr)
    # UPPER(TRIM(x)) is the source half of the case-insensitive rule as well, so it reads back
    # as ``ci``; the two only differ in whether the target is normalised too.
    expected_kind = "ci" if spec["kind"] == "upper" else spec["kind"]
    assert parsed == {**tx.normalize(spec), "kind": expected_kind, "column": "c"}
    assert tx.rule_text(spec, "c")
    assert tx.label(spec)


def test_case_insensitive_is_the_only_rule_that_normalises_both_sides():
    assert tx.target_sql({"kind": "ci"}, "c") == "UPPER(TRIM(c))"
    for kind in ("identity", "trim", "lower", "upper", "zfill", "left", "right", "digits"):
        spec = {"kind": kind, "width": 5} if kind in ("zfill", "left", "right") else {"kind": kind}
        assert tx.target_sql(spec, "c") == "c"


def test_bad_specs_are_rejected_before_any_sql_runs():
    with pytest.raises(ValueError):
        tx.normalize({"kind": "drop_table"})
    with pytest.raises(ValueError):
        tx.normalize({"kind": "zfill"})
    with pytest.raises(ValueError):
        tx.sql({"kind": "left", "width": 0}, "c")


def test_a_builtin_case_expression_is_understood_and_a_composite_key_is_not_guessed():
    city = tx.relationship("lower(trim(city))", "lower(trim(hq_city))")
    assert city["left"]["kind"] == "lower" and city["left"]["column"] == "city"
    assert city["right"]["kind"] == "lower" and city["right"]["column"] == "hq_city"
    assert "lower case" in city["rule"]

    composite = tx.relationship("ntd_id || ':' || mode || ':' || tos", "ntd_id || ':' || mode || ':' || tos")
    assert composite["left"]["kind"] == "expression" and composite["rule"] == ""
    assert composite["left"]["expr"] == "ntd_id || ':' || mode || ':' || tos"


def test_a_paired_choice_records_the_columns_each_side_starts_from():
    meta = tx.pair({"kind": "ci"}, "city", "city")
    assert meta["left"] == {"kind": "ci", "column": "city"}
    assert meta["right"] == {"kind": "ci", "column": "city"}
    assert "both sides" in meta["rule"]


def test_the_catalog_stores_the_transform_behind_every_relationship(fresh_db):
    city = next(r for r in schema.RELATIONSHIPS if r.left_expr == "lower(trim(city))")
    assert city.transform["left"]["kind"] == "lower" and city.transform["left"]["column"] == "city"
    assert city.transform["right"]["column"] == "hq_city"

    payload = schema.describe_catalog()
    described = next(r for r in payload["relationships"] if r["key"] == city.key())
    assert described["transform"]["left"]["kind"] == "lower"


def test_an_existing_catalog_is_backfilled_once(fresh_db):
    import db.catalog_store as catalog_store

    conn = fresh_db.get_conn()
    conn.execute("UPDATE catalog_relationships SET transform_json = NULL")
    assert catalog_store.backfill_transforms(conn) == len(schema.RELATIONSHIPS)
    assert catalog_store.backfill_transforms(conn) == 0  # already recorded: never rewritten
    row = conn.execute(
        "SELECT transform_json FROM catalog_relationships WHERE left_expr = 'lower(trim(city))'"
    ).fetchone()
    assert row and '"lower"' in row[0]


def test_the_join_plan_applies_the_relationship_expression_not_the_bare_column():
    """The catalog expression is authoritative: a plan's ON clause carries the normalisation."""
    assert schema.qualify_join_expr("houses", "lower(trim(city))") == "lower(trim(houses.city))"
    assert schema.qualify_join_expr("houses", "LEFT(TRIM(tract_fips), 11)") == "LEFT(TRIM(houses.tract_fips), 11)"

"""agents/sql_inspector.py - structural facts about a statement, read from DuckDB's own parser."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import sql_inspector as si  # noqa: E402


def blk(sql, **kw):
    return si.inspect_sql(sql, **kw).blocks[0]


# ---- statement-level ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sql,fragment", [
    ("DROP TABLE houses", "only SELECT"), ("DELETE FROM houses", "only SELECT"), ("UPDATE houses SET price = 1", "only SELECT"),
    ("COPY houses TO '/tmp/x.csv'", "only SELECT"), ("ATTACH 'x.db'", "only SELECT"),
    ("PRAGMA database_list", "SELECT"),       # DuckDB rewrites PRAGMAs to SELECTs; still refused
    ("SELECT 1; SELECT 2", "exactly one"), ("SELECT 1; DROP TABLE houses", "exactly one"),
    ("SELEC 1", "syntax"), ("SELECT FROM WHERE", "syntax"), ("", "exactly one"),
])
def test_only_one_select_statement_is_accepted(sql, fragment):
    with pytest.raises(si.SqlInspectionError) as err:
        si.inspect_sql(sql)
    assert fragment in str(err.value)


def test_keywords_inside_strings_and_comments_are_not_statements():
    ins = si.inspect_sql("SELECT 'DROP TABLE x; DELETE FROM y' AS note, replace(address, 'St', 'Street') FROM houses -- COPY foo")
    assert ins.tables == {"houses"} and "replace" in ins.functions        # the old keyword regex rejected this


# ---- safety facts, found anywhere in the statement ------------------------------------------------------------------
def test_table_functions_files_and_environment_functions_are_visible_wherever_they_hide():
    ins = si.inspect_sql("SELECT getenv('HOME') FROM houses, read_csv_auto('/etc/passwd')")
    assert ins.table_functions == {"read_csv_auto"} and "getenv" in ins.functions and ins.tables == {"houses"}
    assert si.inspect_sql("SELECT * FROM '/etc/passwd'").files == ["/etc/passwd"]
    deep = si.inspect_sql("WITH x AS (SELECT * FROM glob('/home/*')) SELECT * FROM houses WHERE city IN (SELECT a FROM x)")
    assert deep.table_functions == {"glob"} and deep.ctes == {"x"} and deep.tables == {"houses"}
    sub = si.inspect_sql("SELECT (SELECT getenv('X')) FROM houses")
    assert "getenv" in sub.functions


def test_cte_names_are_not_reported_as_tables():
    ins = si.inspect_sql("WITH c AS (SELECT * FROM houses) SELECT * FROM c JOIN census_tracts t ON c.tract_fips = t.tract_fips")
    assert ins.tables == {"houses", "census_tracts"} and ins.ctes == {"c"}


# ---- joins ---------------------------------------------------------------------------------------------------------
def test_explicit_joins_produce_validated_join_predicates():
    b = blk("SELECT 1 FROM houses h JOIN census_tracts t ON h.tract_fips = t.tract_fips LEFT JOIN nri_tracts n ON n.tract_fips = t.tract_fips")
    pairs = {(p.left, p.right) for p in b.join_preds}
    assert pairs == {("houses.tract_fips", "census_tracts.tract_fips"), ("nri_tracts.tract_fips", "census_tracts.tract_fips")}
    assert b.base_tables == {"h": "houses", "t": "census_tracts", "n": "nri_tracts"}


def test_implicit_comma_joins_are_joins_when_the_where_clause_says_so():
    b = blk("SELECT 1 FROM houses h, census_tracts t WHERE h.tract_fips = t.tract_fips AND h.city = 'Pittsburgh'")
    assert [(p.left, p.right, p.origin) for p in b.join_preds] == [("houses.tract_fips", "census_tracts.tract_fips", "where")]


def test_cross_joins_have_no_predicates():
    b = blk("SELECT 1 FROM houses CROSS JOIN census_tracts")
    assert b.join_preds == [] and b.links == []


def test_expression_joins_are_canonical_across_equivalent_spellings():
    forms = [
        "c.state_fips || c.county_fips = n.county_fips", "concat(c.state_fips, c.county_fips) = n.county_fips",
        "n.county_fips = c.state_fips || c.county_fips", "CAST(n.county_fips AS VARCHAR) = c.state_fips || c.county_fips",
    ]
    seen = set()
    for f in forms:
        b = blk(f"SELECT 1 FROM cbsa_counties c JOIN nri_tracts n ON {f}")
        (p,) = b.join_preds
        seen.add(frozenset({p.left, p.right}))
    assert len(seen) == 1 and next(iter(seen)) == frozenset({"concat(cbsa_counties.state_fips, cbsa_counties.county_fips)", "nri_tracts.county_fips"})


def test_left_and_substr_are_the_same_join_key():
    a = blk("SELECT 1 FROM census_tracts t JOIN cbsa_counties c ON LEFT(t.tract_fips, 5) = c.state_fips || c.county_fips").join_preds[0]
    b = blk("SELECT 1 FROM census_tracts t JOIN cbsa_counties c ON SUBSTR(t.tract_fips, 1, 5) = c.state_fips || c.county_fips").join_preds[0]
    c = blk("SELECT 1 FROM census_tracts t JOIN cbsa_counties c ON substring(t.tract_fips, 1, 5) = c.state_fips || c.county_fips").join_preds[0]
    assert {a.left, a.right} == {b.left, b.right} == {c.left, c.right}


def test_three_way_concat_is_flattened():
    (p,) = blk("SELECT 1 FROM a x JOIN b y ON x.p || x.q || x.r = y.k").join_preds
    assert "concat(a.p, a.q, a.r)" in {p.left, p.right}


def test_self_joins_and_same_table_predicates_are_not_cross_table_joins():
    b = blk("SELECT 1 FROM houses a JOIN houses b ON a.city = b.city AND a.house_id <> b.house_id")
    assert b.join_preds == []                         # same base table on both sides: nothing to validate against relationships


def test_natural_join_and_using_are_recorded():
    assert blk("SELECT 1 FROM houses NATURAL JOIN census_tracts").natural_joins == 1
    b = blk("SELECT 1 FROM houses h JOIN census_tracts t USING (tract_fips)")
    assert b.using == [(["t"], ["h"], "tract_fips")]


def test_unqualified_columns_are_attributed_through_the_callback():
    cols = {"houses": {"house_id", "city", "tract_fips"}, "census_tracts": {"tract_fips", "population"}}
    b = blk("SELECT 1 FROM houses h JOIN census_tracts t ON h.tract_fips = t.tract_fips WHERE city = 'x'",
            columns_of=lambda t: cols.get(t))
    assert any(p.column == "houses.city" and p.values == frozenset({"x"}) for p in b.value_preds)
    amb = blk("SELECT 1 FROM houses h JOIN census_tracts t ON h.tract_fips = t.tract_fips WHERE tract_fips = 'x'",
              columns_of=lambda t: cols.get(t))
    assert any(p.column == "?.tract_fips" for p in amb.value_preds)      # ambiguous -> not attributed to either table


# ---- value predicates ----------------------------------------------------------------------------------------------
def test_equality_in_lists_and_or_chains_count_but_negation_and_tautologies_do_not():
    sql = "SELECT 1 FROM census_msa m WHERE m.name = 'A' AND m.state IN ('B', 'C') AND (m.kind = 'D' OR m.kind = 'E')"
    preds = {p.column: p.values for p in blk(sql).value_preds}
    assert preds == {"census_msa.name": frozenset({"A"}), "census_msa.state": frozenset({"B", "C"}), "census_msa.kind": frozenset({"D", "E"})}
    assert blk("SELECT 1 FROM census_msa m WHERE m.name <> 'A'").value_preds == []
    assert blk("SELECT 1 FROM census_msa m WHERE m.name NOT IN ('A')").value_preds == []
    assert blk("SELECT 1 FROM census_msa m WHERE m.name = 'A' OR 1 = 1").value_preds == []     # the filter is voided by the OR
    assert blk("SELECT 1 FROM census_msa m WHERE m.name = 'A' OR m.state = 'B'").value_preds == []  # different columns: not a filter on name


def test_values_in_comments_are_not_predicates():
    assert blk("SELECT 1 FROM census_msa m -- WHERE m.name = 'A'\n WHERE 1 = 1").value_preds == []


def test_correlated_subqueries_resolve_outer_aliases():
    ins = si.inspect_sql("SELECT h.house_id FROM houses h WHERE h.price > (SELECT AVG(x.price) FROM houses x WHERE x.city = h.city AND x.state = 'PA')")
    inner = [b for b in ins.blocks if any(p.column == "houses.state" for p in b.value_preds)]
    assert inner and ins.tables == {"houses"}


# ---- limit ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sql,limit", [
    ("SELECT 1 FROM houses LIMIT 5", 5), ("SELECT 1 FROM houses ORDER BY 1 DESC LIMIT 12 OFFSET 3", 12),
    ("SELECT 1 FROM houses", None), ("SELECT city FROM houses UNION ALL SELECT city FROM sold_homes LIMIT 7", 7),
    ("SELECT * FROM (SELECT 1 FROM houses LIMIT 3) t", None),       # an inner LIMIT does not bound the statement
])
def test_the_statement_limit_is_the_outermost_one(sql, limit):
    assert si.inspect_sql(sql).limit == limit


# ---- the contract with DuckDB's tree ---------------------------------------------------------------------------------
def test_selftest_passes_on_this_duckdb():
    si.selftest()


def test_a_changed_tree_shape_is_reported_as_unavailable_not_as_bad_sql(monkeypatch):
    monkeypatch.setattr(si, "_parse", lambda sql: {"type": "SELECT_NODE", "from_table": {"type": "JOIN", "left": 5, "right": None, "condition": "x"}})
    with pytest.raises(si.SqlInspectorUnavailable):
        si.inspect_sql("SELECT 1")


def test_canonical_expression_for_relationship_sides():
    assert si.canonical_sql_expression("state_fips || county_fips", "cbsa_counties") == "concat(cbsa_counties.state_fips, cbsa_counties.county_fips)"
    assert si.canonical_sql_expression("LEFT(tract_fips, 5)", "census_tracts") == si.canonical_sql_expression("substr(tract_fips, 1, 5)", "census_tracts")
    assert si.canonical_sql_expression("tract_fips", "houses") == "houses.tract_fips"


def test_inspection_is_fast_enough_to_run_on_every_statement():
    import time
    sql = "SELECT m.name, AVG(n.risk_score) FROM census_msa m JOIN cbsa_counties c ON m.msa_code = c.cbsa_code JOIN nri_tracts n ON n.county_fips = c.state_fips || c.county_fips WHERE m.name = 'x' GROUP BY 1"
    si.inspect_sql(sql)
    t0 = time.perf_counter()
    for _ in range(200):
        si.inspect_sql(sql)
    assert (time.perf_counter() - t0) / 200 < 0.01          # milliseconds, not tens of them


def test_schema_qualified_relations_are_never_mistaken_for_catalog_tables():
    assert si.inspect_sql("SELECT * FROM information_schema.tables").tables == {"information_schema.tables"}
    assert si.inspect_sql("SELECT * FROM main.houses").tables == {"houses"}
    assert si.inspect_sql("SELECT * FROM other_db.main.houses").tables == {"other_db.houses"}


def test_non_equality_conditions_still_connect_relations():
    b = blk("SELECT 1 FROM houses h JOIN zhvi z ON z.date <= h.list_date AND z.zip = h.zip")
    assert b.links and b.join_preds                 # the equality is validated; the inequality only counts for connectivity
    only_range = blk("SELECT 1 FROM houses h JOIN zhvi z ON z.date <= h.list_date")
    assert only_range.links and not only_range.join_preds
    ored = blk("SELECT 1 FROM houses h JOIN zhvi z ON (z.zip = h.zip OR z.city = h.city)")
    assert ored.links and not ored.join_preds

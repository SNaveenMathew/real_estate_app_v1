"""agents/plan_contract.py: bad SQL is caught for the right reason; valid SQL is never falsely rejected."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402
from plan_battery_queries import GROUPS  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("contract"))
    yield f
    census_fixture.deactivate(f)


def plan_for(q):
    from agents.query_planner import build_query_plan
    return build_query_plan(q)


def codes(sql, plan):
    from agents import plan_contract
    return {v.code for v in plan_contract.verify(sql, plan)}


NRI_Q = "What is the average NRI risk score for the Pittsburgh metro?"
GOOD = """SELECT AVG(nri_tracts.risk_score) FROM census_msa
 JOIN cbsa_counties ON census_msa.msa_code = cbsa_counties.cbsa_code
 JOIN nri_tracts ON cbsa_counties.state_fips || cbsa_counties.county_fips = nri_tracts.county_fips
 WHERE census_msa.name = 'Pittsburgh, PA Metro Area'"""
ENT = "census_msa.name = 'Pittsburgh, PA Metro Area'"


@pytest.mark.parametrize("name,sql,expected", [
    ("cross join, no conditions", "SELECT AVG(nri_tracts.risk_score) FROM census_msa CROSS JOIN cbsa_counties CROSS JOIN nri_tracts "
                                  f"WHERE {ENT}", {"cross_join"}),
    ("joins on nonsense columns", "SELECT AVG(nri_tracts.risk_score) FROM census_msa JOIN cbsa_counties ON census_msa.population = "
                                  "cbsa_counties.cbsa_code JOIN nri_tracts ON cbsa_counties.county_name = nri_tracts.county_name "
                                  f"WHERE {ENT}", {"unapproved_join"}),
    ("entity negated", GOOD.replace(ENT, "census_msa.name <> 'Pittsburgh, PA Metro Area'"), {"missing_entity_filter"}),
    ("entity only in a comment", GOOD.replace(f"WHERE {ENT}", "-- 'Pittsburgh, PA Metro Area'\n WHERE 1=1"), {"missing_entity_filter"}),
    ("entity ORed with a tautology", GOOD.replace(ENT, f"{ENT} OR 1=1"), {"missing_entity_filter"}),
    ("entity ORed with another column", GOOD.replace(ENT, f"{ENT} OR census_msa.msa_code = '1'"), {"missing_entity_filter"}),
    ("entity NOT IN", GOOD.replace(ENT, "census_msa.name NOT IN ('Pittsburgh, PA Metro Area')"), {"missing_entity_filter"}),
    ("entity literal in the wrong case", GOOD.replace("Pittsburgh, PA", "PITTSBURGH, PA"), {"missing_entity_filter"}),
    ("file read via comma-joined table function", GOOD + " AND EXISTS (SELECT 1 FROM census_msa, read_csv_auto('/etc/passwd'))",
     {"table_function"}),
    ("file path as a table", GOOD + " AND EXISTS (SELECT 1 FROM '/etc/passwd')", {"file_access"}),
    ("environment leak", GOOD.replace("AVG(nri_tracts.risk_score)", "AVG(nri_tracts.risk_score), getenv('HOME')"), {"forbidden_function"}),
    ("system table", GOOD + " AND EXISTS (SELECT 1 FROM information_schema.tables)", {"unknown_table"}),
    ("a table the catalog does not expose", GOOD + " AND EXISTS (SELECT 1 FROM catalog_concepts)", {"unknown_table"}),
    ("NATURAL JOIN", "SELECT AVG(nri_tracts.risk_score) FROM census_msa NATURAL JOIN cbsa_counties NATURAL JOIN nri_tracts "
                     f"WHERE {ENT}", {"natural_join"}),
    ("direct join where only a bridge is approved", "SELECT AVG(nri_tracts.risk_score) FROM census_msa JOIN cbsa_counties ON census_msa.msa_code = "
                                                    "cbsa_counties.cbsa_code JOIN nri_tracts ON census_msa.name = nri_tracts.county_name "
                                                    f"WHERE {ENT}", {"unapproved_join"}),      # reported once: an unapproved equality still connects the tables
    ("DDL", "DROP TABLE houses", {"sql_invalid"}),
    ("two statements", GOOD + "; DROP TABLE houses", {"sql_invalid"}),
    ("syntax error", "SELEC 1", {"sql_invalid"}),
])
def test_bad_statements_are_caught_for_the_right_reason(fx, name, sql, expected):
    got = codes(sql, plan_for(NRI_Q))
    assert expected <= got, (name, got)


def test_the_control_statement_is_clean(fx):
    assert codes(GOOD, plan_for(NRI_Q)) == set()


def test_safety_violations_are_flagged_as_such(fx):
    from agents import plan_contract
    for sql in ("SELECT getenv('HOME') FROM houses", "SELECT * FROM houses, read_csv_auto('x')", "SELECT * FROM '/etc/passwd'"):
        v = plan_contract.verify(sql, plan_for("How many houses are in Pittsburgh?"))
        assert v and all(x.severity == "safety" for x in v if x.code in {"forbidden_function", "table_function", "file_access"})


def test_modes_never_relax_safety(fx):
    from agents import plan_contract
    plan = plan_for(NRI_Q)
    bad_join = plan_contract.verify(GOOD.replace("JOIN cbsa_counties ON census_msa.msa_code = cbsa_counties.cbsa_code", "CROSS JOIN cbsa_counties"), plan)
    assert bad_join and not plan_contract.blocking(bad_join, "warn") and not plan_contract.blocking(bad_join, "off")
    assert plan_contract.blocking(bad_join, "enforce")
    unsafe = plan_contract.verify("SELECT getenv('HOME') FROM houses", plan)
    assert plan_contract.blocking(unsafe, "off")


def test_messages_tell_the_model_what_is_acceptable(fx):
    from agents import plan_contract
    msgs = {v.code: v.message for v in plan_contract.verify(
        "SELECT AVG(nri_tracts.risk_score) FROM census_msa JOIN cbsa_counties ON census_msa.population = cbsa_counties.cbsa_code "
        f"JOIN nri_tracts ON cbsa_counties.state_fips || cbsa_counties.county_fips = nri_tracts.county_fips WHERE {ENT}", plan_for(NRI_Q))}
    assert "census_msa.msa_code = cbsa_counties.cbsa_code" in msgs["unapproved_join"]          # names the approved condition
    missing = {v.code: v.message for v in plan_contract.verify(GOOD.replace(ENT, "1=1"), plan_for(NRI_Q))}
    assert "Pittsburgh, PA Metro Area" in missing["missing_entity_filter"] and "exact literal" in missing["missing_entity_filter"]


VALID = {
    "aliases and explicit INNER JOIN": """SELECT AVG(n.risk_score) FROM census_msa m INNER JOIN cbsa_counties c ON m.msa_code = c.cbsa_code
        INNER JOIN nri_tracts n ON c.state_fips || c.county_fips = n.county_fips WHERE m.name = 'Pittsburgh, PA Metro Area'""",
    "join sides swapped": """SELECT AVG(n.risk_score) FROM census_msa m JOIN cbsa_counties c ON c.cbsa_code = m.msa_code
        JOIN nri_tracts n ON n.county_fips = c.state_fips || c.county_fips WHERE m.name = 'Pittsburgh, PA Metro Area'""",
    "CONCAT instead of ||": """SELECT AVG(n.risk_score) FROM census_msa m JOIN cbsa_counties c ON m.msa_code = c.cbsa_code
        JOIN nri_tracts n ON CONCAT(c.state_fips, c.county_fips) = n.county_fips WHERE m.name = 'Pittsburgh, PA Metro Area'""",
    "CAST around a join key": """SELECT AVG(n.risk_score) FROM census_msa m JOIN cbsa_counties c ON CAST(m.msa_code AS VARCHAR) = c.cbsa_code
        JOIN nri_tracts n ON c.state_fips || c.county_fips = CAST(n.county_fips AS VARCHAR) WHERE m.name = 'Pittsburgh, PA Metro Area'""",
    "LEFT JOINs": """SELECT AVG(n.risk_score) FROM census_msa m LEFT JOIN cbsa_counties c ON m.msa_code = c.cbsa_code
        LEFT JOIN nri_tracts n ON c.state_fips || c.county_fips = n.county_fips WHERE m.name = 'Pittsburgh, PA Metro Area'""",
    "implicit comma joins with WHERE equalities": """SELECT AVG(n.risk_score) FROM census_msa m, cbsa_counties c, nri_tracts n
        WHERE m.msa_code = c.cbsa_code AND c.state_fips || c.county_fips = n.county_fips AND m.name = 'Pittsburgh, PA Metro Area'""",
    "entity in an IN list": """SELECT AVG(n.risk_score) FROM census_msa m JOIN cbsa_counties c ON m.msa_code = c.cbsa_code
        JOIN nri_tracts n ON c.state_fips || c.county_fips = n.county_fips WHERE m.name IN ('Pittsburgh, PA Metro Area')""",
    "entity via LOWER()": """SELECT AVG(n.risk_score) FROM census_msa m JOIN cbsa_counties c ON m.msa_code = c.cbsa_code
        JOIN nri_tracts n ON c.state_fips || c.county_fips = n.county_fips WHERE LOWER(m.name) = 'pittsburgh, pa metro area'""",
    "entity inside a CTE": """WITH metro AS (SELECT * FROM census_msa WHERE name = 'Pittsburgh, PA Metro Area')
        SELECT AVG(n.risk_score) FROM metro m JOIN cbsa_counties c ON m.msa_code = c.cbsa_code
        JOIN nri_tracts n ON c.state_fips || c.county_fips = n.county_fips""",
    "CTE with the join inside, grouped, ordered, limited": """WITH t AS (SELECT n.risk_score AS s, m.name AS nm FROM census_msa m JOIN cbsa_counties c ON m.msa_code = c.cbsa_code
        JOIN nri_tracts n ON c.state_fips || c.county_fips = n.county_fips WHERE m.name = 'Pittsburgh, PA Metro Area')
        SELECT nm, AVG(s) AS r FROM t GROUP BY nm ORDER BY r DESC LIMIT 10""",
    "aggregate, HAVING and window functions": """SELECT m.name, AVG(n.risk_score) OVER (PARTITION BY m.name) FROM census_msa m
        JOIN cbsa_counties c ON m.msa_code = c.cbsa_code JOIN nri_tracts n ON c.state_fips || c.county_fips = n.county_fips
        WHERE m.name = 'Pittsburgh, PA Metro Area' AND n.risk_score IS NOT NULL""",
    "scalar subquery in the select list": """SELECT m.name, (SELECT COUNT(*) FROM cbsa_counties cc WHERE cc.cbsa_code = m.msa_code) FROM census_msa m
        JOIN cbsa_counties c ON m.msa_code = c.cbsa_code JOIN nri_tracts n ON c.state_fips || c.county_fips = n.county_fips
        WHERE m.name = 'Pittsburgh, PA Metro Area'""",
    "the replace() function the old keyword regex rejected": """SELECT replace(m.name, ' Metro Area', ''), AVG(n.risk_score) FROM census_msa m JOIN cbsa_counties c ON m.msa_code = c.cbsa_code
        JOIN nri_tracts n ON c.state_fips || c.county_fips = n.county_fips WHERE m.name = 'Pittsburgh, PA Metro Area' GROUP BY 1""",
}


@pytest.mark.parametrize("name", sorted(VALID))
def test_equivalent_and_idiomatic_statements_are_accepted(fx, name):
    assert codes(VALID[name], plan_for(NRI_Q)) == set(), name


def test_self_joins_and_correlated_subqueries_are_accepted(fx):
    plan = plan_for("How many houses are in Pittsburgh?")
    sql = "SELECT h.house_id FROM houses h WHERE h.city = 'Pittsburgh' AND h.price > (SELECT AVG(x.price) FROM houses x WHERE x.city = h.city)"
    assert codes(sql, plan) == set()
    sql = "SELECT a.house_id, b.house_id FROM houses a JOIN houses b ON a.city = b.city AND a.house_id <> b.house_id WHERE a.city = 'Pittsburgh'"
    assert codes(sql, plan) == set()


def test_tract_joins_in_either_direction_are_accepted(fx):
    plan = plan_for("average population of tracts with houses in Pittsburgh")
    for on in ("h.tract_fips = t.tract_fips", "t.tract_fips = h.tract_fips"):
        sql = f"SELECT AVG(t.population) FROM houses h JOIN census_tracts t ON {on} WHERE h.city = 'Pittsburgh'"
        assert not ({"unapproved_join", "cross_join"} & codes(sql, plan)), on


def test_everything_the_deterministic_compiler_emits_satisfies_the_contract(fx):
    """The fallback compiler is valid by construction; the contract must agree on EVERY plan in the battery."""
    from agents import plan_contract
    from agents.tools import _compile_sql_from_plan
    checked = 0
    for qs in GROUPS.values():
        for q in qs:
            plan = plan_for(q)
            try:
                sql = _compile_sql_from_plan(plan)
            except Exception:
                continue                                     # not compilable (no operation / tables): nothing to check
            violations = plan_contract.verify(sql, plan)
            assert not violations, (q, sql, violations)
            checked += 1
    assert checked >= 15                                      # a meaningful share of the battery really compiles

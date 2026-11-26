"""Verifiable health check for every catalog relationship and the formats of its two key expressions.

The catalog states four things a relationship must satisfy: both tables exist, both expressions exist, the two
sides speak the same value format, and the join matches the cardinality the catalog declares.  Nothing enforced
those claims before, so a many-to-many join could be declared ``many-to-one`` (silently fanning out rows), a
column could drift to a different format at load time, or a relationship could point at a level a dataset never
loaded (matching nothing).  :func:`report` measures each claim against the live data and returns typed findings;
``python -m db.join_health`` prints them.

Findings are deterministic and computed from the data, never from a list of known problems:

* ``error``   - the catalog is wrong about structure (missing table/column, incompatible types, or a join whose
                observed cardinality is more permissive than declared, e.g. many-to-many declared many-to-one).
* ``warning`` - the join cannot work as documented on this data (0 matches) or only partly covers the left side.
* ``info``    - measured facts worth knowing (coverage, per-level matches, format classes).

Empty tables are skipped: an unloaded dataset is a data-loading gap, not a broken relationship (the answer layer
reports those separately, see agents/answer_policy.py::availability_gap).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import db.duckdb_store as store
from db import schema_catalog as schema

_IDENT = re.compile(r"^[A-Za-z_]\w*$")
# Tables that store several geography levels in one table (region_type).  A join against them only answers at the
# levels that were loaded, so "no match" is measured per level and their label column is not format-compared.
_MULTI_LEVEL = {"zhvi", "market_heat_index"}
# Dominant value shapes.  A relationship whose two sides fall into different shapes is almost always a format bug
# (a numeric code stored as "38300.0", a name where a code belongs, upper-case labels against mixed-case ones).
_SHAPES = [
    ("zip5", re.compile(r"^\d{5}$")),
    ("fips11", re.compile(r"^\d{11}$")),
    ("digits", re.compile(r"^\d+$")),
    ("int.0", re.compile(r"^-?\d+\.0$")),
    ("numeric", re.compile(r"^-?\d+(?:\.\d+)?$")),
]


@dataclass
class Finding:
    severity: str  # error | warning | info
    relationship: str
    message: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    checked: int = 0
    skipped: int = 0

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def add(self, severity: str, relationship: str, message: str) -> None:
        self.findings.append(Finding(severity, relationship, message))


def _q(expr: str, table: str) -> str:
    """Qualify a relationship key expression against its table."""
    return schema.qualify_join_expr(table, expr)


def _column(table: str, expr: str) -> str | None:
    return expr.strip() if _IDENT.match(expr.strip()) else None


def _type(table: str, column: str) -> str | None:
    df = store.query(
        "SELECT data_type FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
        [table, column],
    )
    return None if df.empty else str(df["data_type"].iloc[0])


def _shape(values: list[str]) -> str:
    if not values:
        return "empty"
    counts: dict[str, int] = {}
    for v in values:
        name = "text"
        for shape, rx in _SHAPES:
            if rx.match(v):
                name = shape
                break
        counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get)


def _case_shape(table: str, expr: str, limit: int = 400) -> str:
    """Whether a text key is stored upper-case, lower-case or mixed - a case-only mismatch joins to nothing."""
    df = store.query(
        f"SELECT CAST({_q(expr, table)} AS VARCHAR) AS v FROM {table} "
        f"WHERE {_q(expr, table)} IS NOT NULL LIMIT {int(limit)}"
    )
    letters = [str(v) for v in df["v"].tolist() if any(ch.isalpha() for ch in str(v))]
    if not letters:
        return ""
    upper = sum(1 for v in letters if v == v.upper()) / len(letters)
    lower = sum(1 for v in letters if v == v.lower()) / len(letters)
    if upper >= 0.8:
        return "upper"
    if lower >= 0.8:
        return "lower"
    return "mixed"


def _sample_shape(table: str, expr: str, limit: int = 200) -> tuple[str, list[str]]:
    df = store.query(
        f"SELECT CAST({_q(expr, table)} AS VARCHAR) AS v FROM {table} "
        f"WHERE {_q(expr, table)} IS NOT NULL LIMIT {int(limit)}"
    )
    values = [str(v) for v in df["v"].tolist()]
    return _shape(values), values[:3]


def _measure(left_table: str, left_expr: str, right_table: str, right_expr: str, scope: list[str] | None = None) -> dict:
    l = _q(left_expr, left_table)
    r = _q(right_expr, right_table)
    scope_sql = ""
    if scope:
        scope_sql = " AND " + " AND ".join(f"({str(x)})" for x in scope)
    sql = (
        f"WITH l AS (SELECT CAST({l} AS VARCHAR) AS k FROM {left_table} WHERE {l} IS NOT NULL), "
        f"r_all AS (SELECT CAST({r} AS VARCHAR) AS k FROM {right_table} WHERE {r} IS NOT NULL{scope_sql}), "
        "r AS (SELECT DISTINCT k FROM r_all) "
        "SELECT (SELECT count(*) FROM l) AS l_rows, (SELECT count(DISTINCT k) FROM l) AS l_keys, "
        "(SELECT count(*) FROM r) AS r_keys, (SELECT count(*) FROM r_all) AS r_rows, "
        "(SELECT count(*) FROM l WHERE k IN (SELECT k FROM r)) AS matched_rows, "
        "(SELECT count(DISTINCT k) FROM l WHERE k IN (SELECT k FROM r)) AS matched_keys"
    )
    row = store.query(sql).iloc[0]
    return {k: int(row[k]) for k in ("l_rows", "l_keys", "r_keys", "r_rows", "matched_rows", "matched_keys")}


def _observed(metrics: dict, right_rows: int) -> str:
    """Observed cardinality in the direction the relationship is written (left -> right).

    What fans a query out is the RIGHT key repeating: every left row then matches several right rows.  The LEFT
    key repeating (many houses in one ZIP) does not duplicate the right side and is normal for a ``one-to-many``
    relationship, so it is only reported when the catalog claims ``one-to-one``.
    """
    left_unique = metrics["l_rows"] == metrics["l_keys"]
    right_unique = right_rows == metrics["r_keys"]
    if left_unique and right_unique:
        return "one-to-one"
    if right_unique:
        return "many-to-one"
    if left_unique:
        return "one-to-many"
    return "many-to-many"


def _multiplicity(cardinality: str) -> tuple[bool, bool]:
    """(left side is unique, right side is unique) as declared, ignoring prose values."""
    return {
        "one-to-one": (True, True),
        "one-to-many": (True, False),
        "many-to-one": (False, True),
        "many-to-many": (False, False),
    }.get(cardinality, (False, False))


def _region_levels(table: str) -> list[str]:
    df = store.query(f"SELECT DISTINCT region_type FROM {table} WHERE region_type IS NOT NULL ORDER BY region_type")
    return [str(v) for v in df["region_type"].tolist()]


def audit_relationship(rel, report: Report) -> None:
    label = f"{rel.left_table}.{rel.left_expr} = {rel.right_table}.{rel.right_expr}"
    known = {str(t).lower() for t in schema.list_table_names()}
    for table in (rel.left_table, rel.right_table):
        if table.lower() not in known:
            report.add("error", label, f"table {table!r} does not exist in the catalog")
            return
        lc, rc = _column(rel.left_table, rel.left_expr), _column(rel.right_table, rel.right_expr)
        if lc and _type(rel.left_table, lc) is None:
            report.add("error", label, f"column {table}.{lc} does not exist")
            return
        if rc and _type(rel.right_table, rc) is None:
            report.add("error", label, f"column {table}.{rc} does not exist")
            return

    left_col = _column(rel.left_table, rel.left_expr)
    right_col = _column(rel.right_table, rel.right_expr)
    if left_col and right_col:
        lt, rt = _type(rel.left_table, left_col), _type(rel.right_table, right_col)
        if lt and rt and (lt == "VARCHAR") != (rt == "VARCHAR"):
            report.add(
                "error",
                label,
                f"incompatible key types: {rel.left_table}.{left_col} is {lt}, "
                f"{rel.right_table}.{right_col} is {rt}",
            )
            return

    try:
        rows = {
            rel.left_table: int(store.query(f"SELECT count(*) AS n FROM {rel.left_table}").iloc[0]["n"]),
            rel.right_table: int(store.query(f"SELECT count(*) AS n FROM {rel.right_table}").iloc[0]["n"]),
        }
    except Exception as exc:
        report.add("error", label, f"table not readable: {exc}")
        return
    if 0 in rows.values():
        report.skipped += 1
        return

    try:
        scope = schema.relationship_scope_sql(rel)
        metrics = _measure(rel.left_table, rel.left_expr, rel.right_table, rel.right_expr, scope)
    except Exception as exc:
        report.add("error", label, f"join cannot be evaluated: {exc}")
        return
    report.checked += 1

    observed = _observed(metrics, metrics["r_rows"])
    per_key = metrics["r_rows"] / metrics["r_keys"] if metrics["r_keys"] else 0.0
    if rel.cardinality:
        _declared_left_unique, declared_right_unique = _multiplicity(rel.cardinality)
        if declared_right_unique and per_key > 1.0:
            report.add(
                "error",
                label,
                f"declared {rel.cardinality} but the right key repeats: {metrics['r_rows']} right rows for "
                f"{metrics['r_keys']} distinct keys ({per_key:.1f} per key), so each left row fans out",
            )
        elif rel.cardinality == "one-to-one" and metrics["l_rows"] != metrics["l_keys"]:
            report.add(
                "warning",
                label,
                f"declared one-to-one but the left key repeats ({metrics['l_rows']} rows, "
                f"{metrics['l_keys']} distinct)",
            )
        elif observed != rel.cardinality:
            report.add("info", label, f"declared {rel.cardinality}, observed {observed}")

    if metrics["matched_rows"] == 0:
        level_hit = _matches_at_any_level(rel)
        if level_hit:
            report.add("warning", label, f"no match across all rows, but matches at region_type={level_hit}")
        elif rel.right_table in _MULTI_LEVEL:
            levels = ", ".join(_region_levels(rel.right_table)) or "none"
            report.add(
                "warning",
                label,
                f"no match at the scoped geography level; loaded levels in {rel.right_table}: {levels}",
            )
        else:
            severity = "warning" if rel.preferred is False else "error"
            report.add(severity, label, "no left row matches: the documented join returns nothing on this data")
    else:
        coverage = metrics["matched_keys"] / metrics["l_keys"]
        if coverage < 1.0:
            report.add(
                "info",
                label,
                f"coverage {metrics['matched_keys']}/{metrics['l_keys']} distinct left keys ({coverage:.1%}), "
                f"{metrics['matched_rows']}/{metrics['l_rows']} left rows",
            )

        if metrics["matched_rows"] > metrics["l_rows"] * 10 and rel.cardinality not in ("many-to-many", "one-to-many"):
            report.add(
                "warning",
                label,
                f"join fan-out is {metrics['matched_rows'] / max(metrics['l_rows'], 1):.1f} rows per left row; "
                "the relationship is not safe as an unaggregated join",
            )

    if scope:
        report.add("info", label, "scoped join: " + "; ".join(scope))

    try:
        lshape, lsample = _sample_shape(rel.left_table, rel.left_expr)
        rshape, rsample = _sample_shape(rel.right_table, rel.right_expr)
        multi = rel.left_table in _MULTI_LEVEL or rel.right_table in _MULTI_LEVEL
        lcase = _case_shape(rel.left_table, rel.left_expr) if lshape == "text" else ""
        rcase = _case_shape(rel.right_table, rel.right_expr) if rshape == "text" else ""
        if lcase and rcase and lcase != rcase:
            report.add(
                "warning",
                label,
                f"label case differs: {rel.left_table}.{rel.left_expr} is {lcase}-case, "
                f"{rel.right_table}.{rel.right_expr} is {rcase}-case",
            )
        if lshape != rshape and "empty" not in (lshape, rshape) and not multi:
            report.add(
                "warning",
                label,
                f"key formats differ: {rel.left_table} looks like {lshape} {lsample}, "
                f"{rel.right_table} looks like {rshape} {rsample}",
            )
    except Exception:
        pass

    for table in (rel.left_table, rel.right_table):
        if table in _MULTI_LEVEL:
            try:
                levels = _region_levels(table)
            except Exception:
                continue
            if levels:
                report.add("info", label, f"{table} region_type levels loaded: {', '.join(sorted(levels))}")


def _matches_at_any_level(rel) -> str:
    """Which region_type level makes this relationship match, when one side is a multi-level dataset."""
    table = rel.left_table if rel.left_table in _MULTI_LEVEL else rel.right_table if rel.right_table in _MULTI_LEVEL else ""
    if not table:
        return ""
    l, r = _q(rel.left_expr, rel.left_table), _q(rel.right_expr, rel.right_table)
    own = l if table == rel.left_table else r
    for level in _region_levels(table):
        sql = (
            f"SELECT count(*) AS n FROM {rel.left_table} l JOIN {rel.right_table} r "
            f"ON {l} = {r} WHERE {own}"
        )
        try:
            if int(store.query(sql, [level]).iloc[0]["n"]) > 0:
                return level
        except Exception:
            continue
    return ""


def report() -> Report:
    r = Report()
    for rel in schema.RELATIONSHIPS:
        audit_relationship(rel, r)
    return r


def format_report(r: Report) -> str:
    lines = [
        f"Join health: {r.checked} relationship(s) measured, {r.skipped} skipped (empty table), "
        f"{len(r.errors)} error(s), {len(r.warnings)} warning(s)"
    ]
    for f in r.findings:
        if f.severity == "info":
            continue
        lines.append(f"  [{f.severity}] {f.relationship}\n      {f.message}")
    infos = [f for f in r.findings if f.severity == "info"]
    if infos:
        lines.append(f"  ({len(infos)} informational finding(s); run with --verbose to list them)")
    return "\n".join(lines)


def print_report(*, verbose: bool = False) -> Report:
    r = report()
    print(format_report(r))
    if verbose:
        for f in r.findings:
            if f.severity == "info":
                print(f"  [info] {f.relationship}\n      {f.message}")
    return r


if __name__ == "__main__":
    import sys

    result = print_report(verbose="--verbose" in sys.argv)
    sys.exit(1 if result.errors else 0)

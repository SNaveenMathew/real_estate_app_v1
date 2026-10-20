"""The plan is a contract; this module verifies a SQL statement against it, structurally.

``agents/sql_inspector.py`` reads facts out of the statement (it never judges).  This module judges them:

safety     (never relaxed)   one SELECT; no file / environment / system access; only catalog tables
joins      (contract)        every join condition is an APPROVED relationship; tables in a block are connected
                             (no accidental cross product); no NATURAL JOIN
entities   (contract)        each resolved named entity (``census_msa.name = 'Pittsburgh, PA Metro Area'``) appears as a
                             POSITIVE filter - not negated, not commented out, not voided by ``OR 1=1``
scope      (contract)        mandatory scope filters (House Chat's ``houses.house_id = '...'``, agents/query_scope.py)
limit      (contract)        an explicit "top N" in the request is honoured by a LIMIT no larger than N

Two severities exist so a rollout can be gradual: ``settings.sql_contract_mode`` = ``enforce`` (default) rejects any
violation, ``warn`` rejects only ``safety`` ones, ``off`` likewise.  Safety is never optional.

Equivalent spellings are NOT violations: ``LEFT(x, 5)`` vs ``SUBSTR(x, 1, 5)``, ``a || b`` vs ``CONCAT(a, b)``, a CAST
around a join key, ``LOWER(col) = 'value'``, an IN-list or an OR-chain of equalities.  The canonical forms live in the
inspector; ``tests/test_plan_contract.py`` keeps a corpus of valid statements that must stay valid.

Messages are written for the model that will repair the statement: they say what is wrong AND what is acceptable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from agents import sql_inspector as si


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    severity: str = "contract"          # "safety" | "contract"


@dataclass(frozen=True)
class Edge:
    canon: frozenset                    # {canonical left expression, canonical right expression}
    display: str                        # "houses.tract_fips = census_tracts.tract_fips"


@dataclass
class CatalogView:
    """What the contract needs to know about the catalog (injectable, so it is testable without a database)."""
    tables: frozenset
    columns_of: Callable[[str], set | None]
    edges_between: Callable[[str, str], list]
    path_hint: Callable[[str, str], str]


# ---------------------------------------------------------------------------
# Catalog view
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _edge_index(key: tuple) -> dict:
    index: dict = {}
    for left_table, left_expr, right_table, right_expr in key:
        if left_table == right_table:
            continue
        try:
            a = si.canonical_sql_expression(left_expr, left_table)
            b = si.canonical_sql_expression(right_expr, right_table)
        except si.SqlInspectionError:
            continue                                      # an unparsable relationship expression: the catalog lint reports it
        index.setdefault(frozenset({left_table, right_table}), []).append(
            Edge(frozenset({a, b}), f"{left_table}.{left_expr} = {right_table}.{right_expr}"))
    return index


def default_catalog_view() -> CatalogView:
    import db.schema_catalog as schema

    relationships = tuple((r.left_table.lower(), r.left_expr, r.right_table.lower(), r.right_expr) for r in schema.RELATIONSHIPS)
    index = _edge_index(relationships)
    memo: dict = {}

    def columns_of(table: str):
        table = table.lower()
        if table not in memo:
            memo[table] = {c.lower() for c, _ in schema._live_columns(table)} or None
        return memo[table]

    def path_hint(a: str, b: str) -> str:
        try:
            path = schema.relationship_path({a, b})
        except Exception:
            path = []
        return "; ".join(f"{r.left_table}.{r.left_expr} = {r.right_table}.{r.right_expr}" for r in path)

    return CatalogView(frozenset(t.lower() for t in schema.list_table_names()), columns_of,
                       lambda a, b: index.get(frozenset({a, b}), []), path_hint)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
SAFE_TABLE_FUNCTIONS = frozenset({"unnest", "range", "generate_series", "generate_subscripts"})
DENIED_FUNCTIONS = frozenset({"getenv", "current_setting", "getvariable", "query", "query_table", "glob", "which_secret",
                              "st_read", "st_readosm"})
DENIED_PREFIXES = ("read_", "duckdb_", "pragma_", "parquet_", "sniff_", "iceberg_", "delta_", "write_", "http_")


def _safety(ins: si.Inspection, view: CatalogView) -> list[Violation]:
    out = []
    for fn in sorted(ins.table_functions - SAFE_TABLE_FUNCTIONS):
        out.append(Violation("table_function", f"Table function {fn}() is not allowed: query only the catalog tables.", "safety"))
    for path in ins.files:
        out.append(Violation("file_access", f"File or path {path!r} cannot be used as a table: query only the catalog tables.", "safety"))
    for fn in sorted(ins.functions):
        if fn in DENIED_FUNCTIONS or fn.startswith(DENIED_PREFIXES):
            out.append(Violation("forbidden_function", f"Function {fn}() is not allowed (file, environment or system access).", "safety"))
    unknown = sorted(t for t in ins.tables if t not in view.tables)
    if unknown:
        out.append(Violation("unknown_table", f"Table(s) {', '.join(unknown)} are not available to the assistant. "
                                              f"Use only: {', '.join(sorted(view.tables))}.", "safety"))
    return out


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------
def _union_find(items):
    parent = {i: i for i in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a in parent and b in parent:
            parent[find(a)] = find(b)

    return find, union


def _joins(ins: si.Inspection, view: CatalogView) -> list[Violation]:
    out: list[Violation] = []
    seen: set = set()

    def add(code, message):
        if (code, message) not in seen:
            seen.add((code, message))
            out.append(Violation(code, message))

    for blk in ins.blocks:
        if blk.natural_joins:
            add("natural_join", "NATURAL JOIN is not allowed: join explicitly on an approved relationship condition.")
        for jp in blk.join_preds:
            edges = view.edges_between(jp.left_table, jp.right_table)
            if any(frozenset({jp.left, jp.right}) == e.canon for e in edges):
                continue
            if edges:
                approved = "; ".join(e.display for e in edges)
                add("unapproved_join", f"Join condition `{jp.left} = {jp.right}` is not an approved relationship between "
                                       f"{jp.left_table} and {jp.right_table}. Use: {approved}.")
            else:
                hint = view.path_hint(jp.left_table, jp.right_table)
                add("unapproved_join", f"There is no approved direct relationship between {jp.left_table} and {jp.right_table}"
                                       f" (`{jp.left} = {jp.right}` is not allowed)."
                                       + (f" Connect them through: {hint}." if hint else ""))

        find, union = _union_find(list(blk.rels))
        for a, b in blk.links:
            union(a, b)
        for right_aliases, left_aliases, col in blk.using:
            rights = [(a, blk.rels[a].name) for a in right_aliases if a in blk.rels and blk.rels[a].kind == "table"]
            lefts = [(a, blk.rels[a].name) for a in left_aliases if a in blk.rels and blk.rels[a].kind == "table"]
            ok = not rights or not lefts
            checked = False
            for ra, rt in rights:
                for la, lt in lefts:
                    cols = view.columns_of(lt)
                    if cols is not None and col not in cols:
                        continue
                    union(ra, la)
                    checked = True
                    canon = frozenset({f"{lt}.{col}", f"{rt}.{col}"})
                    if lt != rt and any(canon == e.canon for e in view.edges_between(lt, rt)):
                        ok = True
                    elif lt == rt:
                        ok = True
            if checked and not ok:
                names = ", ".join(sorted({t for _, t in rights} | {t for _, t in lefts}))
                add("unapproved_join", f"JOIN ... USING ({col}) between {names} is not an approved relationship: write the "
                                       "approved ON condition instead.")

        groups: dict = {}
        for alias, rel in blk.rels.items():
            if rel.kind == "table":
                groups.setdefault(find(alias), []).append(rel.name)
        if len(groups) > 1:
            parts = [", ".join(sorted(set(g))) for g in groups.values()]
            add("cross_join", "These tables are not joined to each other by any condition, which makes a cross product "
                              f"(wrong results): {' | '.join(parts)}. Join every planned table using the approved "
                              "relationship conditions listed in the plan.")
    return out


# ---------------------------------------------------------------------------
# Entity / scope filters and the row limit
# ---------------------------------------------------------------------------
_EQUALITY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'(.*)'$", re.S)
_WRAPPER = re.compile(r"^(lower|upper|trim|ltrim|rtrim)\((.*)\)$")


def _required_equalities(filters) -> list[tuple[str, str]]:
    out = []
    for f in filters or []:
        m = _EQUALITY.match(str(f).strip())
        if m:                                              # anything fancier is passed through by the planner too
            out.append((f"{m.group(1)}.{m.group(2)}".lower(), m.group(3).replace("''", "'")))
    return out


def _normalized_column(column: str) -> tuple[str, bool]:
    """Strip lower()/upper()/trim() wrappers: ``lower(houses.city)`` -> ('houses.city', case_insensitive=True)."""
    insensitive = False
    while True:
        m = _WRAPPER.match(column)
        if not m:
            return column, insensitive
        insensitive = insensitive or m.group(1) in ("lower", "upper")
        column = m.group(2)


def _has_positive_filter(ins: si.Inspection, column: str, value: str) -> bool:
    for blk in ins.blocks:
        for pred in blk.value_preds:
            col, insensitive = _normalized_column(pred.column)
            if col != column:
                continue
            if value in pred.values or (insensitive and value.lower() in {v.lower() for v in pred.values}):
                return True
    return False


def _filters(ins: si.Inspection, wanted: list[tuple[str, str]], what: str) -> list[Violation]:
    out = []
    for column, value in wanted:
        if not _has_positive_filter(ins, column, value):
            out.append(Violation(
                f"missing_{what}_filter",
                f"The {what} filter {column} = '{value}' must be a positive condition AND-ed into the WHERE or ON clause "
                "(an equality, an IN list, or an OR of equalities on that column). It is missing, negated, commented "
                "out, or voided by an OR with something else. Use the exact literal from the plan."))
    return out


def _limit(ins: si.Inspection, plan) -> list[Violation]:
    n = getattr(plan, "result_limit", None)
    if not n:
        return []
    if ins.limit is None:
        return [Violation("missing_limit", f"The request asks for the top {n} rows: the statement must end with LIMIT {n}.")]
    if ins.limit > n:
        return [Violation("limit_too_large", f"The request asks for the top {n} rows but the statement uses LIMIT {ins.limit}: use LIMIT {n}.")]
    return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def verify(sql: str, plan, view: CatalogView | None = None) -> list[Violation]:
    """All violations of ``sql`` against ``plan``.  Raises ``SqlInspectorUnavailable`` if the parser tree is unreadable."""
    view = view or default_catalog_view()
    try:
        ins = si.inspect_sql(sql, columns_of=view.columns_of)
    except si.SqlInspectionError as exc:
        return [Violation("sql_invalid", str(exc), "safety")]
    out = _safety(ins, view)
    out += _joins(ins, view)
    out += _filters(ins, _required_equalities(getattr(plan, "entity_filters", None)), "entity")
    out += _filters(ins, _required_equalities(getattr(plan, "scope_filters", None)), "scope")
    out += _limit(ins, plan)
    return out


def blocking(violations: list[Violation], mode: str) -> list[Violation]:
    """The violations that must stop the statement under ``settings.sql_contract_mode``."""
    if mode == "enforce":
        return list(violations)
    return [v for v in violations if v.severity == "safety"]

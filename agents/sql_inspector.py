"""Structural inspection of ONE SQL statement, using DuckDB's own parser.

Why not regular expressions: a regex "validator" sees text, not meaning.  Measured against the validator this replaces,
all of the following were accepted: a cross join with no join condition (silently wrong numbers), a join on nonsense
columns, an entity filter that was negated (``<>``) or only present in a comment or ORed with ``1=1``, a file read via a
comma-joined table function (``FROM houses, read_csv_auto('/etc/passwd')``) and an environment leak (``getenv('HOME')``).

``json_serialize_sql`` is DuckDB's documented way to turn a SELECT into a JSON syntax tree WITHOUT executing it.  It is
the exact parser that will run the statement - no dialect mismatch, no new dependency - and costs about 0.3 ms.

What this module does: read the tree and report facts - which relations, which functions, which joins and which
``column = constant`` predicates each query block contains.  What it deliberately does NOT do: decide whether those facts
are acceptable.  That is ``agents/plan_contract.py``, which compares them with the plan and the approved relationships.

Robustness
----------
* The walk is generic for everything safety depends on (every function name, every table, every table function anywhere in
  the statement, including subqueries and CTE bodies), so a node type this module has never seen cannot hide one.
* If DuckDB ever changes its tree shape, ``SqlInspectorUnavailable`` is raised (distinct from ``SqlInspectionError`` for
  bad SQL) so callers can fall back to the previous checks and say so loudly, instead of rejecting every query.
  ``selftest()`` and ``tests/test_sql_inspector.py`` exist to catch that at upgrade time.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator

import duckdb


class SqlInspectionError(ValueError):
    """The statement is not one acceptable SELECT (syntax error, several statements, not a SELECT)."""


class SqlInspectorUnavailable(RuntimeError):
    """DuckDB's syntax-tree format is not what this module understands (e.g. after a version upgrade)."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rel:
    alias: str            # alias, or the table name when there is none (lower case)
    kind: str             # table | cte | subquery | function | file | values | other
    name: str             # table / CTE / function name (lower case); '' for subquery / values


@dataclass(frozen=True)
class JoinPred:
    """``left = right`` where each side belongs to exactly one (different) base table."""
    left: str             # canonical expression, e.g. "houses.tract_fips" or "concat(a.x, a.y)"
    right: str
    left_table: str
    right_table: str
    origin: str           # on | where | using


@dataclass(frozen=True)
class ValuePred:
    """``column = constant``, ``column IN (constants)`` or an OR-chain of those on one column."""
    column: str           # canonical, e.g. "census_msa.name"
    values: frozenset


@dataclass
class Block:
    rels: dict = field(default_factory=dict)             # alias -> Rel
    join_preds: list = field(default_factory=list)       # validated against approved relationships
    links: list = field(default_factory=list)            # (alias, alias) for ANY condition touching two relations: connectivity
    value_preds: list = field(default_factory=list)
    natural_joins: int = 0
    using: list = field(default_factory=list)            # (right_aliases, left_aliases, column)

    @property
    def base_tables(self) -> dict:
        return {a: r.name for a, r in self.rels.items() if r.kind == "table"}


@dataclass
class Inspection:
    blocks: list = field(default_factory=list)
    tables: set = field(default_factory=set)             # base tables referenced ANYWHERE (not CTE names)
    ctes: set = field(default_factory=set)
    table_functions: set = field(default_factory=set)
    files: list = field(default_factory=list)
    functions: set = field(default_factory=set)          # every scalar / aggregate / window function name
    limit: int | None = None                             # outermost LIMIT, when it is a constant


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_PARSER: duckdb.DuckDBPyConnection | None = None
_PARSER_LOCK = threading.Lock()
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _parse(sql: str) -> dict:
    global _PARSER
    with _PARSER_LOCK:
        if _PARSER is None:
            _PARSER = duckdb.connect(":memory:")          # parse-only: it never needs the catalog
        try:
            statements = _PARSER.extract_statements(sql)
        except Exception as exc:                          # duckdb.ParserException and friends
            raise SqlInspectionError(f"SQL syntax error: {_first_line(exc)}") from exc
        if len(statements) != 1:
            raise SqlInspectionError(f"exactly one SQL statement is allowed (found {len(statements)})")
        kind = str(statements[0].type)
        if not kind.endswith("SELECT"):
            raise SqlInspectionError(f"only SELECT statements are allowed (found {kind.rsplit('.', 1)[-1]})")
        try:
            raw = _PARSER.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
        except Exception as exc:
            raise SqlInspectorUnavailable(f"json_serialize_sql is not usable here: {_first_line(exc)}") from exc
    try:
        tree = json.loads(raw)
    except Exception as exc:
        raise SqlInspectorUnavailable("json_serialize_sql returned something that is not JSON") from exc
    if tree.get("error"):
        raise SqlInspectionError(f"SQL cannot be analysed: {tree.get('error_message', 'unknown error')}")
    try:
        return tree["statements"][0]["node"]
    except Exception as exc:
        raise SqlInspectorUnavailable("unexpected syntax-tree layout from json_serialize_sql") from exc


def _first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0][:200] if str(exc).strip() else type(exc).__name__


def _iter_dicts(obj) -> Iterator[dict]:
    """Every dict anywhere inside ``obj`` (generic: independent of node types)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_dicts(v)


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------
def _const(node) -> str | None:
    """The text of a constant (looking through casts); None when ``node`` is not a (non-null) constant."""
    while isinstance(node, dict) and node.get("class") == "CAST":
        node = node.get("child")
    if isinstance(node, dict) and node.get("class") == "CONSTANT":
        v = node.get("value") or {}
        if v.get("is_null"):
            return None
        return str(v.get("value"))
    return None


def _flatten(node, conj_type: str) -> list:
    if isinstance(node, dict) and node.get("class") == "CONJUNCTION" and node.get("type") == conj_type:
        out = []
        for ch in node.get("children", []):
            out.extend(_flatten(ch, conj_type))
        return out
    return [node]


_FUNC_ALIASES = {"||": "concat", "substring": "substr", "ifnull": "coalesce"}


class _Resolver:
    """Resolves column references to base tables across the enclosing query blocks."""

    def __init__(self, columns_of: Callable[[str], set | None] | None):
        self.columns_of = columns_of
        self.stack: list[Block] = []

    def column(self, parts: list[str]) -> tuple[str, str | None, str | None]:
        """-> (canonical text, base table or None, relation alias or None)."""
        col = parts[-1].lower()
        if len(parts) >= 2:
            qual = parts[-2].lower()
            for blk in reversed(self.stack):
                rel = blk.rels.get(qual)
                if rel is not None:
                    if rel.kind == "table":
                        return f"{rel.name}.{col}", rel.name, qual
                    return f"~{qual}.{col}", None, qual
            return f"?{qual}.{col}", None, None
        for blk in reversed(self.stack):
            tables = blk.base_tables
            owners = [a for a, t in tables.items() if self.columns_of and col in (self.columns_of(t) or set())]
            if len(owners) == 1:
                return f"{tables[owners[0]]}.{col}", tables[owners[0]], owners[0]
            if not self.columns_of and len(tables) == 1:
                alias, table = next(iter(tables.items()))
                return f"{table}.{col}", table, alias
        return f"?.{col}", None, None

    def canon(self, node) -> str:
        """Canonical text of an expression, stable across equivalent spellings (CAST, SUBSTR/LEFT, || vs CONCAT)."""
        if not isinstance(node, dict):
            return "?"
        c = node.get("class")
        if c == "COLUMN_REF":
            return self.column(node.get("column_names") or ["?"])[0]
        if c == "CONSTANT":
            v = _const(node)
            return "null" if v is None else repr(v)
        if c == "CAST":
            return self.canon(node.get("child"))               # a cast does not change which columns are being joined
        if c == "FUNCTION":
            name = _FUNC_ALIASES.get(str(node.get("function_name", "")).lower(), str(node.get("function_name", "")).lower())
            args = [self.canon(a) for a in node.get("children", [])]
            if name == "left" and len(args) == 2:                  # LEFT(x, n) == SUBSTR(x, 1, n)
                return f"substr({args[0]}, '1', {args[1]})"
            if name == "substr" and len(args) == 3 and args[1] == "'1'":
                return f"substr({args[0]}, '1', {args[2]})"
            if name == "concat":                                    # flatten (a || b) || c
                flat = []
                for a, child in zip(args, node.get("children", [])):
                    if a.startswith("concat(") and isinstance(child, dict) and child.get("class") == "FUNCTION":
                        flat.extend(_split_args(a[len("concat("):-1]))
                    else:
                        flat.append(a)
                return f"concat({', '.join(flat)})"
            return f"{name}({', '.join(args)})"
        return f"<{c}>"

    def refs(self, node) -> list[tuple[str | None, str | None]]:
        """(base table, alias) of every column reference inside an expression (subqueries excluded)."""
        out = []
        stack = [node]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if cur.get("class") == "SUBQUERY":
                    continue
                if cur.get("class") == "COLUMN_REF":
                    _, table, alias = self.column(cur.get("column_names") or ["?"])
                    out.append((table, alias))
                else:
                    stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
        return out


def _split_args(text: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


# ---------------------------------------------------------------------------
# The inspection walk
# ---------------------------------------------------------------------------
class _Walker:
    def __init__(self, root: dict, columns_of):
        self.root = root
        self.ins = Inspection()
        self.res = _Resolver(columns_of)
        self.columns_of = columns_of
        for d in _iter_dicts(root):
            for entry in (d.get("cte_map") or {}).get("map", []):
                self.ins.ctes.add(str(entry["key"]).lower())

    def run(self) -> Inspection:
        # safety facts: generic over the WHOLE tree
        for d in _iter_dicts(self.root):
            cls, typ = d.get("class"), d.get("type")
            if cls in ("FUNCTION", "WINDOW") and d.get("function_name"):
                self.ins.functions.add(str(d["function_name"]).lower())
            if typ == "TABLE_FUNCTION":
                fn = (d.get("function") or {}).get("function_name", "")
                self.ins.table_functions.add(str(fn).lower())
            if typ == "BASE_TABLE":
                name = str(d.get("table_name", ""))
                qualifier = [str(d.get(k) or "").lower() for k in ("catalog_name", "schema_name")]
                qualifier = [q for q in qualifier if q and q != "main"]
                if name.lower() in self.ins.ctes and not qualifier:
                    continue
                if _IDENT.match(name):
                    # information_schema.tables, other_db.main.houses ... are not the catalog's "tables"/"houses"
                    self.ins.tables.add(".".join(qualifier + [name.lower()]))
                else:
                    self.ins.files.append(name)
        # structural facts: per query block
        self._node(self.root)
        self.ins.limit = self._limit(self.root)
        return self.ins

    def _limit(self, node: dict) -> int | None:
        for m in node.get("modifiers", []) or []:
            if m.get("type") == "LIMIT_MODIFIER":
                v = _const(m.get("limit"))
                try:
                    return int(v) if v is not None else None
                except ValueError:
                    return None
        return None

    def _node(self, node: dict) -> None:
        for entry in (node.get("cte_map") or {}).get("map", []):
            self._node(entry["value"]["query"]["node"])
        typ = node.get("type")
        if typ == "SELECT_NODE":
            self._select(node)
        elif "left" in node and "right" in node:               # set operations, recursive CTEs
            self._node(node["left"])
            self._node(node["right"])

    def _select(self, node: dict) -> None:
        blk = Block()
        self.ins.blocks.append(blk)
        self.res.stack.append(blk)
        try:
            conjuncts: list[tuple[str, dict]] = []
            self._from(node.get("from_table"), blk, conjuncts)
            where = node.get("where_clause")
            if where:
                conjuncts.extend(("where", c) for c in _flatten(where, "CONJUNCTION_AND"))
            for origin, expr in conjuncts:
                self._predicate(origin, expr, blk)
            for sub in self._subqueries({k: v for k, v in node.items() if k not in ("from_table", "cte_map")}):
                self._node(sub)
        finally:
            self.res.stack.pop()

    def _subqueries(self, obj) -> list[dict]:
        """Nodes of subquery EXPRESSIONS inside ``obj`` (not descending into them: they are analysed recursively)."""
        found, stack = [], [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if cur.get("class") == "SUBQUERY" and isinstance(cur.get("subquery"), dict):
                    found.append(cur["subquery"]["node"])
                    if cur.get("child") is not None:
                        stack.append(cur["child"])
                else:
                    stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
        return found

    def _from(self, node, blk: Block, conjuncts: list) -> list[str]:
        """Register the relations of a FROM tree; returns the aliases it introduced."""
        if not isinstance(node, dict):
            return []
        typ = node.get("type")
        if typ == "BASE_TABLE":
            name = str(node.get("table_name", ""))
            low = name.lower()
            alias = str(node.get("alias") or name).lower()
            qualified = any(str(node.get(k) or "").lower() not in ("", "main") for k in ("catalog_name", "schema_name"))
            if low in self.ins.ctes and not qualified:
                kind = "cte"
            elif _IDENT.match(name) and not qualified:
                kind = "table"
            elif _IDENT.match(name):
                kind = "other"
            else:
                kind = "file"
            blk.rels[alias] = Rel(alias, kind, low)
            return [alias]
        if typ == "JOIN":
            left = self._from(node.get("left"), blk, conjuncts)
            right = self._from(node.get("right"), blk, conjuncts)
            if node.get("ref_type") == "NATURAL":
                blk.natural_joins += 1
            for col in node.get("using_columns") or []:
                blk.using.append((right, left, str(col).lower()))
            cond = node.get("condition")
            if cond:
                conjuncts.extend(("on", c) for c in _flatten(cond, "CONJUNCTION_AND"))
            return left + right
        if typ == "SUBQUERY":
            alias = str(node.get("alias") or f"__subquery{len(blk.rels)}").lower()
            blk.rels[alias] = Rel(alias, "subquery", "")
            inner = (node.get("subquery") or {}).get("node")
            if inner:
                saved = self.res.stack
                self.res.stack = []                             # a derived table cannot see its siblings
                try:
                    self._node(inner)
                finally:
                    self.res.stack = saved
            return [alias]
        if typ == "TABLE_FUNCTION":
            fn = str((node.get("function") or {}).get("function_name", "")).lower()
            alias = str(node.get("alias") or fn or f"__fn{len(blk.rels)}").lower()
            blk.rels[alias] = Rel(alias, "function", fn)
            return [alias]
        if typ == "EXPRESSION_LIST":
            alias = str(node.get("alias") or f"__values{len(blk.rels)}").lower()
            blk.rels[alias] = Rel(alias, "values", "")
            return [alias]
        if typ == "EMPTY":
            return []
        alias = str(node.get("alias") or f"__rel{len(blk.rels)}").lower()
        blk.rels[alias] = Rel(alias, "other", "")
        return [alias]

    # -- predicates ---------------------------------------------------------------------------------------------
    def _value_pred(self, e: dict) -> ValuePred | None:
        c, t = e.get("class"), e.get("type")
        if c == "COMPARISON" and t == "COMPARE_EQUAL":
            lc, rc = _const(e.get("left")), _const(e.get("right"))
            if rc is not None and lc is None:
                return ValuePred(self.res.canon(e["left"]), frozenset({rc}))
            if lc is not None and rc is None:
                return ValuePred(self.res.canon(e["right"]), frozenset({lc}))
        if c == "OPERATOR" and t == "COMPARE_IN":
            ch = e.get("children", [])
            vals = [_const(x) for x in ch[1:]]
            if ch and vals and all(v is not None for v in vals):
                return ValuePred(self.res.canon(ch[0]), frozenset(vals))
        return None

    def _predicate(self, origin: str, e: dict, blk: Block) -> None:
        vp = self._value_pred(e)
        if vp is not None:
            blk.value_preds.append(vp)
            return
        c, t = e.get("class"), e.get("type")
        if c == "CONJUNCTION" and t == "CONJUNCTION_OR":
            parts = [self._value_pred(x) for x in _flatten(e, "CONJUNCTION_OR")]
            if parts and all(p is not None for p in parts) and len({p.column for p in parts}) == 1:
                blk.value_preds.append(ValuePred(parts[0].column, frozenset().union(*(p.values for p in parts))))
                return
            # otherwise fall through: an OR of other conditions still connects the relations it mentions
        if not (c == "COMPARISON" and t == "COMPARE_EQUAL"):
            aliases = sorted({a for _, a in self.res.refs(e) if a})
            for i, la in enumerate(aliases):                     # connectivity only: not validated against relationships
                for ra in aliases[i + 1:]:
                    blk.links.append((la, ra))
            return
        lrefs, rrefs = self.res.refs(e["left"]), self.res.refs(e["right"])
        laliases = {a for _, a in lrefs if a}
        raliases = {a for _, a in rrefs if a}
        if laliases and raliases and laliases != raliases:
            for la in laliases:
                for ra in raliases:
                    blk.links.append((la, ra))
        ltables, rtables = {t for t, _ in lrefs if t}, {t for t, _ in rrefs if t}
        if len(ltables) == 1 and len(rtables) == 1 and len(laliases) == 1 and len(raliases) == 1 and ltables != rtables:
            blk.join_preds.append(JoinPred(self.res.canon(e["left"]), self.res.canon(e["right"]),
                                           next(iter(ltables)), next(iter(rtables)), origin))


def inspect_sql(sql: str, *, columns_of: Callable[[str], set | None] | None = None) -> Inspection:
    """Parse and analyse one SELECT.  ``columns_of(table)`` lets unqualified column references be attributed to a table."""
    tree = _parse(sql)
    try:
        return _Walker(tree, columns_of).run()
    except (SqlInspectionError, SqlInspectorUnavailable):
        raise
    except Exception as exc:                                    # an unexpected tree shape, not bad SQL
        raise SqlInspectorUnavailable(f"could not read the syntax tree: {type(exc).__name__}: {exc}") from exc


def canonical_sql_expression(expr: str, default_table: str, columns_of=None) -> str:
    """Canonical form of a stand-alone expression such as a catalog relationship side (``state_fips || county_fips``).

    Unqualified columns are attributed to ``default_table``.
    """
    tree = _parse(f"SELECT {expr}")
    try:
        item = tree["select_list"][0]
    except Exception as exc:
        raise SqlInspectorUnavailable("unexpected syntax-tree layout for an expression") from exc
    res = _Resolver(columns_of)
    blk = Block()
    blk.rels[default_table] = Rel(default_table, "table", default_table)
    res.stack.append(blk)
    return res.canon(item)


def selftest() -> None:
    """Raises SqlInspectorUnavailable when DuckDB's tree no longer looks the way this module expects."""
    ins = inspect_sql("SELECT 1 FROM a x JOIN b y ON x.k = y.k WHERE x.n = 'v' LIMIT 3")
    blk = ins.blocks[0]
    ok = (ins.tables == {"a", "b"} and ins.limit == 3 and len(blk.join_preds) == 1
          and any(p.column == "a.n" and p.values == frozenset({"v"}) for p in blk.value_preds))
    if not ok:
        raise SqlInspectorUnavailable("selftest query did not yield the expected structure")

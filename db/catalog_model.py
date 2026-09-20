"""Data classes and the tiny DSL used to describe catalog objects.

These are plain, dependency-free definitions shared by the seed data, the persistent store and
the in-memory registry.  ``origin`` / ``dataset_id`` are provenance only: a built-in table and an
uploaded one are the same kind of object.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnNote:
    column: str
    note: str
    role: str = ""          # key | label | measure | dimension | date | geo | other
    unit: str = ""
    source_name: str = ""   # original header, when the column came from an upload


@dataclass(frozen=True)
class TableMeta:
    name: str
    description: str
    setup_hint: str = ""
    filter_hint: str = ""
    column_notes: tuple[ColumnNote, ...] = ()
    hidden_columns: tuple[str, ...] = ()
    agent_visible: bool = True
    grain: str = ""
    default_filter: str = ""
    domain: str = ""        # housing | geography | risk | sales | safety | mobility | ...
    origin: str = "builtin" # provenance only: builtin | upload | manual
    dataset_id: str = ""


@dataclass(frozen=True)
class Relationship:
    left_table: str
    left_expr: str
    right_table: str
    right_expr: str
    note: str = ""
    cardinality: str = ""
    confidence: str = "high"
    bridge: bool = False
    preferred: bool = True
    grain_effect: str = ""
    origin: str = "builtin"
    dataset_id: str = ""

    def involves(self, tables: set[str]) -> bool:
        return self.left_table in tables and self.right_table in tables

    def key(self) -> str:
        return f"{self.left_table}:{self.left_expr}={self.right_table}:{self.right_expr}"

    def render(self) -> str:
        attrs = [self.cardinality, f"confidence={self.confidence}", f"preferred={self.preferred}", f"bridge={self.bridge}"]
        extra = []
        if self.note:
            extra.append(self.note)
        if self.grain_effect:
            extra.append("Grain: " + self.grain_effect)
        return f"{self.left_table}.{self.left_expr} = {self.right_table}.{self.right_expr} [{' ; '.join(attrs)}]" + (" Note: " + " ".join(extra) if extra else "")


@dataclass(frozen=True)
class EntityDomain:
    name: str
    table: str
    column: str
    entity_type: str
    description: str
    display_column: str | None = None
    match_mode: str = "exact_or_prefix"
    preferred_for: tuple[str, ...] = ()
    origin: str = "builtin"
    dataset_id: str = ""



# ---------------------------------------------------------------------------
# Semantic-contract DSL.  Every concept is metadata; none is a routing branch.
# ---------------------------------------------------------------------------

def _concept(key, tables, aliases, description, *, columns=(), operations=(), filters=(), null_policy="", orderings=(), groupings=(), grain="", entity_types=(), rollup=False, rollup_spec=None, required_terms=(), excluded_terms=(), default_operation=None):
    return {
        "key": key, "tables": list(tables), "columns": list(columns), "aliases": list(aliases),
        "description": description, "operations": list(operations), "filters": list(filters),
        "null_policy": null_policy, "orderings": list(orderings), "groupings": list(groupings),
        "grain": grain, "entity_types": list(entity_types), "rollup": rollup, "rollup_spec": rollup_spec or {}, "required_terms": list(required_terms), "excluded_terms": list(excluded_terms), "default_operation": default_operation,
    }

def _op(op, aliases, expr, *, direction=None, group_by=None):
    d = {"op": op, "aliases": aliases, "expr": expr}
    if direction: d["direction"] = direction
    if group_by: d["group_by"] = group_by
    return d

COUNT = lambda expr, **kw: _op("count", ["how many", "number of", "count"], expr, **kw)
AVG = lambda expr, **kw: _op("avg", ["average", "avg", "mean"], expr, **kw)
SUM = lambda expr, **kw: _op("sum", ["total", "sum", "combined"], expr, **kw)
MEDIAN = lambda expr, **kw: _op("median", ["median"], expr, **kw)
MIN = lambda expr, **kw: _op("min", ["lowest", "minimum", "min", "worst"], expr, **kw)
MAX = lambda expr, **kw: _op("max", ["highest", "maximum", "max", "best"], expr, **kw)
RANK_DESC = lambda expr, **kw: _op("rank", ["rank", "ranking", "highest to lowest", "from highest to lowest", "largest", "highest"], expr, direction="DESC", **kw)
RANK_ASC = lambda expr, **kw: _op("rank", ["lowest to highest", "from lowest to highest", "smallest", "lowest"], expr, direction="ASC", **kw)

"""The query plan's own status: a closed set of states plus the invariant that makes the plan compilable.

The planner used to hand back a ``QueryPlan`` and nothing else, so "no plan could be made" was not a value: it was an
empty dataclass that every caller had to re-derive.  ``agents/tools.py`` did that by trying to compile and raising
``ValueError("Cannot compile SQL: plan has no operation.")``, which the orchestrator then read as an error string.  A
plan that *projected a table it did not include* was worse: it compiled to SQL that referenced a table that was never
joined, and DuckDB answered ``Binder Error: Referenced table "houses"`` - a database error describing a planning bug.

So the plan gets a status, computed in code from the plan itself (never by a model), with one state per way a plan can
fail to become a query:

    ok              compilable: tables, a join path, an operation, and no table projected but not planned
    no_match        nothing in the catalog matched (the user-facing answer is ``answer_status.UNPLANNABLE``)
    no_operation    a table matched, but the plan selects no measure/operation
    inconsistent    the projection/ordering/filters name a table the plan does not include
    unconnected     the planned tables have no catalog relationship path (a cross product would result)

``referenced_tables`` is also used by ``agents/query_planner.build_query_plan`` to REPAIR the third state: a referenced
table that the catalog can join is added to the plan before compilation.  What cannot be repaired is reported here, so
the failure is named at planning time instead of surfacing as a database error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from agents import answer_status

_QUALIFIED = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")


class PlanState(str, Enum):
    OK = "ok"
    NO_MATCH = "no_match"
    NO_OPERATION = "no_operation"
    INCONSISTENT = "inconsistent"
    UNCONNECTED = "unconnected"


#: Which status code each non-ok state reports (``agents/answer_status.py``).
STATE_CODE = {
    PlanState.OK: answer_status.OK,
    PlanState.NO_MATCH: answer_status.UNPLANNABLE,
    PlanState.NO_OPERATION: answer_status.PLAN_INCOMPLETE,
    PlanState.INCONSISTENT: answer_status.PLAN_INCONSISTENT,
    PlanState.UNCONNECTED: answer_status.PLAN_UNCONNECTED,
}


def _expressions(plan) -> list[str]:
    out: list[str] = []
    for attr in ("select_expression", "ordering"):
        value = getattr(plan, attr, None)
        if value:
            out.append(str(value))
    for attr in ("grouping", "filters", "entity_filters", "scope_filters"):
        out.extend(str(v) for v in (getattr(plan, attr, None) or ()))
    rollup = getattr(plan, "rollup_spec", None) or {}
    out.extend(str(v) for v in rollup.values() if isinstance(v, str))
    for m in getattr(plan, "measures", None) or ():
        for val in (getattr(m, "expr", None), getattr(m, "group_key", None)):
            if val:
                out.append(str(val))
        out.extend(str(f) for f in (getattr(m, "filters", None) or ()))
    return out


def referenced_tables(plan) -> set:
    """Lower-cased table names named by the plan's projection, ordering, grouping and filters."""
    return {m.group(1).lower() for expr in _expressions(plan) for m in _QUALIFIED.finditer(expr)}


def projected_tables(plan) -> set:
    """Tables named by the SELECT/ORDER BY/GROUP BY expressions only (filters are checked separately)."""
    out: set = set()
    for attr in ("select_expression", "ordering"):
        value = getattr(plan, attr, None)
        if value:
            out |= {m.group(1).lower() for m in _QUALIFIED.finditer(str(value))}
    for value in getattr(plan, "grouping", None) or ():
        out |= {m.group(1).lower() for m in _QUALIFIED.finditer(str(value))}
    return out


@dataclass(frozen=True)
class PlanReport:
    """What the planner concluded, as a value: one finite state, the code it maps to, and what is missing."""

    state: PlanState
    tables: tuple[str, ...] = ()
    missing_operation: bool = False
    unplanned_tables: tuple[str, ...] = ()
    disconnects: tuple[tuple[str, ...], ...] = ()
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.state is PlanState.OK

    @property
    def code(self) -> str:
        return STATE_CODE[self.state]

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "status": self.code,
            "category": answer_status.category(self.code),
            "ok": self.ok,
            "tables": list(self.tables),
            "missing_operation": self.missing_operation,
            "unplanned_tables": list(self.unplanned_tables),
            "disconnects": [list(group) for group in self.disconnects],
            "detail": self.detail,
        }

    def message(self) -> str:
        """A user-safe status line for this report ('' when the plan is fine)."""
        if self.ok:
            return ""
        return answer_status.limitation(self.code, self.detail)


def report(plan, relationship_path=None) -> PlanReport:
    """Classify ``plan``.  ``relationship_path`` is injectable so this is testable without a live catalog."""
    tables = tuple(sorted({str(t).lower() for t in (getattr(plan, "required_tables", None) or ())}))
    if not tables:
        return PlanReport(PlanState.NO_MATCH, detail="no table in the catalog matched the request")

    unplanned = tuple(sorted(projected_tables(plan) - set(tables)))
    if unplanned:
        return PlanReport(
            PlanState.INCONSISTENT,
            tables=tables,
            unplanned_tables=unplanned,
            detail="projection uses a table the plan does not include: " + ", ".join(unplanned),
        )

    if not getattr(plan, "operation", None):
        return PlanReport(
            PlanState.NO_OPERATION,
            tables=tables,
            missing_operation=True,
            detail="a table matched but the plan selects no measure or operation",
        )

    if relationship_path is not None and len(tables) > 1:
        try:
            path = list(relationship_path(set(tables)))
        except Exception:
            path = []
        if len(path) < len(tables) - 1:
            return PlanReport(
                PlanState.UNCONNECTED,
                tables=tables,
                disconnects=(tables,),
                detail="the planned tables are not connected by catalog relationships",
            )

    return PlanReport(PlanState.OK, tables=tables)

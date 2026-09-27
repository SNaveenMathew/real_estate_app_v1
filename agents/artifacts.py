"""
Shared presentation layer for General Chat and House Chat.

Both agents already decide WHAT computation is needed
(agents/tools.py, agents/house_agent.py) using the Code Agent pattern
documented in AGENT_ARCHITECTURE.md: an LLM writes a small program over
approved functions, and deterministic code executes it and owns
everything that follows. This module extends that same split one step
further, into how a result is *shown*:

    approved function executes                        [LLM decided to call it]
      -> a DataFrame / structured payload is on hand    [deterministic]
      -> classify_dataframe() decides table/chart/map/none, from the
         ACTUAL SHAPE of the result -- row/column counts and dtypes --
         optionally steered by a `presentation` hint the orchestrating
         LLM may pass along (e.g. "the user asked to see this on a
         map"), but the hint never overrides what the data can actually
         support                                        [deterministic]
      -> emit_artifact() records it for this turn        [deterministic]
      -> collect_artifacts() hands the whole list to the API layer once
         the turn is done                                [deterministic]

No LLM ever writes a table's cell values, a chart's numbers, or a map's
coordinates -- it only ever writes natural-language reply text and,
optionally, a `presentation` string that means nothing more than "here's
what the user seemed to want to see." That is the same reasoning already
applied to SQL generation and the query plan (AGENT_ARCHITECTURE.md
§1): open-ended text is the LLM's job; anything with one correct shape
is code's.

Both chat endpoints call `reset_artifacts()` once at the start of a
turn and `collect_artifacts()` once at the end (see
agents/general_agent.py::run_general_chat and
agents/house_agent.py::run_house_chat). In between, any approved
function -- `query_database`, `find_bike_route`, a house-scoped lookup --
may call `emit_artifact()` zero or more times as a side effect of doing
its normal work. A turn can therefore surface more than one artifact
(e.g. a crime-density map followed by the final route map), and most
turns surface none at all, which is the common case and always fine.
"""
from __future__ import annotations

import contextvars
import re
from typing import Any

import pandas as pd

from config import settings

# ---------------------------------------------------------------------------
# Per-turn artifact bus
# ---------------------------------------------------------------------------
# A contextvar (not a plain module global) so that a future move to a
# threaded/async execution model for the code agent can't leak one
# person's turn into another's. Today's execution is synchronous and
# single-threaded per request (see AGENT_ARCHITECTURE.md §3.1), so this
# is a correctness-by-construction choice more than a fix for an observed
# bug.

_artifacts_var: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "chat_artifacts", default=None
)


def reset_artifacts() -> None:
    """Start a fresh, empty artifact list. Call once at the top of a turn."""
    _artifacts_var.set([])


def emit_artifact(artifact: dict[str, Any] | None) -> None:
    """Record one table/chart/map artifact produced during the current turn.

    A no-op for `None` (the common case: most tool calls have nothing to
    visualize), so call sites can always call this unconditionally with
    whatever `classify_dataframe` (or a tool's own builder) returns.
    """
    if not artifact:
        return
    bucket = _artifacts_var.get()
    if bucket is None:
        bucket = []
        _artifacts_var.set(bucket)
    bucket.append(artifact)


def collect_artifacts() -> list[dict[str, Any]]:
    """Return every artifact emitted so far this turn, oldest first."""
    return list(_artifacts_var.get() or [])


# ---------------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------------
# DuckDB -> pandas can hand back Timestamp/NaT/NaN/numpy scalars, none of
# which json.dumps() accepts. db/duckdb_store.py::query_json already does
# this same defensive pass for its own callers; this is the equivalent for
# a DataFrame that's already in hand (classify_dataframe never re-runs the
# query, so it can't reuse query_json without doubling the DB round trip).

def _json_safe_scalar(v: Any) -> Any:
    import numpy as np
    if v is None:
        return None
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    if isinstance(v, pd.Timestamp) or v is pd.NaT:
        return None if pd.isna(v) else v.isoformat()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _records(df: pd.DataFrame, max_rows: int) -> tuple[list[dict], bool, int]:
    total = len(df)
    view = df.head(max_rows)
    rows = [
        {k: _json_safe_scalar(v) for k, v in row.items()}
        for row in view.to_dict(orient="records")
    ]
    return rows, total > max_rows, total


def _prettify(col: str) -> str:
    return re.sub(r"[_\-]+", " ", str(col)).strip().title()


def _is_numeric(df: pd.DataFrame, col: str) -> bool:
    # Booleans are numeric-ish in pandas but are dimensions, not measures --
    # nobody wants a bar chart averaging True/False.
    return pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])


def _is_datetime(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_datetime64_any_dtype(df[col])


_LAT_NAMES = {"lat", "latitude", "lat_deg", "y"}
_LON_NAMES = {"lon", "lng", "long", "longitude", "lon_deg", "x"}
_LABEL_NAME_PRIORITY = ("address", "name", "city", "label", "title", "house_id", "id")
_TEMPORAL_NAME_HINTS = re.compile(r"year|month|date|quarter|week|day", re.I)


def _find_latlon(columns) -> tuple[str, str] | None:
    lower = {c.lower(): c for c in columns}
    lat = next((lower[n] for n in _LAT_NAMES if n in lower), None)
    lon = next((lower[n] for n in _LON_NAMES if n in lower), None)
    return (lat, lon) if lat and lon else None


def _pick_label_column(columns, exclude: set[str]) -> str | None:
    remaining = [c for c in columns if c not in exclude]
    lower = {c.lower(): c for c in remaining}
    for name in _LABEL_NAME_PRIORITY:
        if name in lower:
            return lower[name]
    return remaining[0] if remaining else None


# ---------------------------------------------------------------------------
# Artifact builders
# ---------------------------------------------------------------------------

def _table_artifact(df: pd.DataFrame, *, title: str | None, max_rows: int) -> dict:
    rows, truncated, total = _records(df, max_rows)
    return {
        "type": "table",
        "title": title,
        "columns": [{"key": str(c), "label": _prettify(c)} for c in df.columns],
        "rows": rows,
        "total_rows": total,
        "truncated": truncated,
    }


def _map_points_artifact(
    df: pd.DataFrame, *, title: str | None, lat_col: str, lon_col: str, max_points: int
) -> dict | None:
    geo = df[df[lat_col].notna() & df[lon_col].notna()]
    if geo.empty:
        return None
    label_col = _pick_label_column(df.columns, {lat_col, lon_col})
    total = len(geo)
    points = []
    for row in geo.head(max_points).to_dict(orient="records"):
        try:
            lat_f, lon_f = float(row.get(lat_col)), float(row.get(lon_col))
        except (TypeError, ValueError):
            continue
        fields = {k: _json_safe_scalar(v) for k, v in row.items() if k not in (lat_col, lon_col)}
        points.append({
            "lat": lat_f,
            "lon": lon_f,
            "label": str(row.get(label_col)) if label_col and row.get(label_col) is not None else None,
            "fields": fields,
        })
    if not points:
        return None
    return {
        "type": "map",
        "map_kind": "points",
        "title": title,
        "points": points,
        "total_points": total,
        "truncated": total > max_points,
    }


def _chart_artifact(
    df: pd.DataFrame, dimension_cols: list[str], numeric_cols: list[str],
    *, title: str | None, max_points: int,
) -> dict | None:
    if not (0 < len(df) <= max_points):
        return None
    dim_col = dimension_cols[0]
    is_temporal = _is_datetime(df, dim_col) or bool(_TEMPORAL_NAME_HINTS.search(str(dim_col)))
    work = df
    if is_temporal:
        try:
            work = df.sort_values(dim_col)
        except Exception:
            work = df

    categories = ["" if (v := _json_safe_scalar(raw)) is None else str(v) for raw in work[dim_col].tolist()]
    if len(dimension_cols) > 1:
        # An explicit "chart" hint relaxes the "exactly one dimension" rule
        # (see classify_dataframe). Fold any extra dimension columns into
        # the category label so the chart stays one-dimensional without
        # dropping evidence the user's request specifically named.
        for extra_col in dimension_cols[1:]:
            extra_vals = [str(_json_safe_scalar(v)) for v in work[extra_col].tolist()]
            categories = [f"{cat} / {extra}" for cat, extra in zip(categories, extra_vals)]

    series = []
    for col in numeric_cols:
        values = []
        for raw in work[col].tolist():
            safe = _json_safe_scalar(raw)
            values.append(float(safe) if isinstance(safe, (int, float)) else None)
        series.append({"name": _prettify(col), "values": values})

    return {
        "type": "chart",
        "chart_kind": "line" if is_temporal else "bar",
        "title": title,
        "x_label": _prettify(dim_col),
        "y_label": _prettify(numeric_cols[0]) if len(numeric_cols) == 1 else "Value",
        "categories": categories,
        "series": series,
    }


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------

_VALID_PRESENTATIONS = {"auto", "map", "chart", "table", "none"}


def classify_dataframe(
    df: pd.DataFrame | None,
    *,
    title: str | None = None,
    presentation: str = "auto",
    request: str = "",
) -> dict[str, Any] | None:
    """Deterministically decide whether/how an executed result should be
    shown as a map, chart, or table -- and build that artifact.

    `presentation` is an optional hint, usually forwarded from the
    orchestrating LLM's read of the user's own phrasing ("on a map",
    "chart the trend", "give me a table"). It steers which shape is tried
    and how strict the shape check is, but a hint the data cannot honor
    (e.g. "map" with no latitude/longitude columns in the result) never
    fabricates one -- it falls through to whatever the data actually
    supports. This mirrors every other deterministic check in this
    codebase (AGENT_ARCHITECTURE.md §1): the LLM's output steers, code
    decides.

    Returns `None` when nothing is worth rendering as a separate artifact
    (empty results, or a single summary value that reads better as
    prose) -- the reply text is always the fallback and is never blocked
    on this returning something.
    """
    try:
        presentation = (presentation or "auto").strip().lower()
        if presentation not in _VALID_PRESENTATIONS:
            presentation = "auto"
        if presentation == "none" or df is None or df.empty:
            return None

        if not title and request:
            title = request.strip()[:80] or None
        columns = list(df.columns)

        # An explicit "table" hint always wins outright, including for a
        # single summary row the automatic path below would otherwise skip --
        # if the user asked for a table, a one-row table is still a table.
        if presentation == "table":
            return _table_artifact(df, title=title, max_rows=settings.presentation_table_max_rows)

        # MAP: tried whenever recognizable coordinates exist, hinted or not,
        # since that's purely a question of what columns came back.
        latlon = _find_latlon(columns)
        if latlon:
            artifact = _map_points_artifact(
                df, title=title, lat_col=latlon[0], lon_col=latlon[1],
                max_points=settings.presentation_map_max_points,
            )
            if artifact:
                return artifact

        if len(df) <= 1:
            return None  # a single summary value reads better as prose

        # A numeric-dtype column named like a time period (year, month, ...)
        # is a dimension for charting purposes, not a measure -- "year" is
        # something you'd plot rows *by*, never something you'd average.
        numeric_cols = [
            c for c in columns
            if _is_numeric(df, c) and not _TEMPORAL_NAME_HINTS.search(str(c))
        ]
        dimension_cols = [c for c in columns if c not in numeric_cols]

        chart_hinted = presentation == "chart"
        max_dims = 2 if chart_hinted else 1
        max_measures = 6 if chart_hinted else 4
        max_points = (
            settings.presentation_chart_max_points_hinted
            if chart_hinted else settings.presentation_chart_max_points
        )
        if 1 <= len(dimension_cols) <= max_dims and 1 <= len(numeric_cols) <= max_measures:
            artifact = _chart_artifact(
                df, dimension_cols, numeric_cols, title=title, max_points=max_points,
            )
            if artifact:
                return artifact

        return _table_artifact(df, title=title, max_rows=settings.presentation_table_max_rows)
    except Exception:
        # A presentation-layer bug should never take down a chat turn --
        # the reply text (built from the same executed evidence, separately)
        # is always the fallback.
        return None

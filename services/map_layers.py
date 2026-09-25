"""Generic map layers for the 'Map and chat' page — derived from the unified catalog.

There is no per-table list of "map layers" anywhere in this codebase. Instead, ``describe_layers()``
looks at every agent-visible table in the live catalog (``db.schema_catalog``) and classifies each one
by the columns it actually has, using the same naming conventions already used everywhere else in this
app (``lat``/``lon`` for points, ``geometry_json`` for vector features, ``tract_fips``/``county_fips`` for
join keys — see ``services/dataset_readers.py`` and ``services/relationship_discovery.py``):

    lat + lon columns (DOUBLE)                    -> "points" (few rows) or "heat" (many rows)
    geometry_json (LineString/MultiLineString)     -> "lines"
    geometry_json (Polygon/MultiPolygon)           -> "polygons" (its own outline/fill)
    a tract_fips column, OR a relationship that joins one to nri_tracts/census_tracts
                                                    -> "choropleth" (using the SHARED tract geometry
                                                       ``services/geo_utils.py`` already loads)
    none of the above                              -> not a map layer (e.g. house_snapshots, census_msa)

A table added through the Data page therefore becomes a map layer with no code change: approve a CSV of
block groups with lat/lon, or one that links to ``nri_tracts``/``census_tracts`` by ``tract_fips``, and it
shows up here on the next request (this module reads the catalog fresh every time, exactly like the
planner does).

Two tables get a little more than the generic treatment, because real, tested domain logic already
existed for them before this module did and there is no reason to throw it away:

  * ``crime_incidents`` already carries a ``severity_weight`` column (computed at load time by the
    per-city crime parsers); the generic "pick a weight column" heuristic finds it on its own, so no
    special case is needed there — but the heat grid aggregation itself is the same code that used to
    live in ``services/layers.py::get_crime_heatmap``, now parametrized by table/weight column so
    ``sold_homes`` (and any future point dataset) gets the identical, tested aggregation.
  * ``bike_routes`` keeps its own renderer (moved from ``services/layers.py`` unchanged): BikePGH's raw
    data has genuinely overlapping classifications for the same physical street segment, and resolving
    that needs the facility-type priority table below, which cannot be inferred from column names.

Layers fall into two groups for the frontend's toggle panel (see ``static/layers.js``):

  * "fill" — an area-covering layer (choropleth, or a dataset's own polygon outline/fill). Two fills
    stacked at once mostly just hide one under the other, so at most one is shown at a time (the
    frontend enforces this as a radio group; ``exclusive_groups["fill"]`` in ``describe_layers()`` lists
    the members so this is never guessed from names).
  * "overlay" — points, heat and lines. These draw as distinct glyphs, not solid area fills, so any
    number of them combine with each other and with the one active fill layer without hiding it.

``houses`` (the interactive, clustered marker layer) and the commute-minutes decoration on it are NOT
served through here — they are already built in ``static/app.js``/``static/commute.js`` respectively, with
sidebar selection, favoriting and chat handoff wired in. ``describe_layers()`` still lists both, tagged
``special``, so the frontend panel has one place to read every toggle from; ``get_layer_data`` refuses to
serve data for them (there is nothing generic to serve — the frontend already has the mechanism).
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

import db.duckdb_store as store
import db.schema_catalog as schema
from services import geo_utils

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

HEAT_ROW_THRESHOLD = 300      # a point table with more rows than this defaults to a heat layer, not markers
MAX_POINT_FEATURES = 2000
DEFAULT_GRID_DEG = 0.003      # ~250-300m at US latitudes; halved/doubled per zoom level by the caller
MAX_GRID_CELLS = 6000
MAX_LINE_FEATURES = 5000
MAX_POLYGON_FEATURES = 1500
MAX_TRACTS = 2500
SIMPLIFY_TOLERANCE_DEG = 0.0004    # ~40m — trims polygon vertex count without visible distortion at map scale

TITLE_OVERRIDES = {   # cosmetic only; every one of these is also reachable through the generic prettifier
    "houses": "Houses", "crime_incidents": "Crime", "sold_homes": "Sold Homes",
    "bike_routes": "Bike Routes", "nri_tracts": "Risk (NRI)", "census_tracts": "Population",
}

# Columns that are never offered as a "color by" measure or a heat weight: identifiers, join keys and
# bounding-box/geometry bookkeeping columns look numeric but do not mean anything summed or averaged.
_NOT_A_MEASURE = re.compile(
    r"(^id$|_id$|^lat$|^lon$|^min_lon$|^min_lat$|^max_lon$|^max_lat$|^year$|^month$|"
    r"fips$|_code$|^zip$|zip5?$|geoid|^ord$)", re.I)
_MEASURE_BOOST = re.compile(r"(score|risk|severity|weight|population|price|index|idx|rate|pct|percent|count|rank)", re.I)
_WEIGHT_HINT = re.compile(r"(severity|weight|score|intensity|magnitude)", re.I)
# A column note that calls itself out as the table's headline metric ranks above its siblings
# (e.g. nri_tracts.risk_score's note says "Composite NRI risk score") - a pattern any curator or
# uploader can use, not something tied to one table's column names.
_PRIMARY_HINT = re.compile(r"(composite|overall|total|primary|headline|summary)", re.I)


class LayerError(ValueError):
    """A bad layer name or parameter; safe to show to the user."""


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _prettify(name: str) -> str:
    return " ".join(w.capitalize() if w.islower() else w for w in name.replace("_", " ").split())


def title_for(table: str) -> str:
    return TITLE_OVERRIDES.get(table, _prettify(table))


def _columns(table: str) -> list[tuple[str, str]]:
    return schema._live_columns(table)


def _notes(table: str) -> dict[str, Any]:
    meta = schema._tables().get(table)
    return {n.column: n for n in meta.column_notes} if meta else {}


def _family(sql_type: str) -> str:
    t = sql_type.upper()
    if t.startswith(("VARCHAR", "TEXT", "STRING", "CHAR")):
        return "text"
    if t.startswith(("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "INT")):
        return "int"
    if t.startswith(("DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC")):
        return "float"
    return "other"


def numeric_columns(table: str, hidden: set[str]) -> list[dict]:
    """Candidate measure/weight columns for ``table``: numeric, not an identifier, ranked by name."""
    notes = _notes(table)
    out = []
    for name, sql_type in _columns(table):
        if name in hidden or _family(sql_type) not in ("int", "float") or _NOT_A_MEASURE.search(name):
            continue
        note = notes.get(name)
        boost = 1 if _MEASURE_BOOST.search(name) else 0
        if note and note.role == "measure":
            boost += 1     # a role stored by the Data page onboarding flow is a stronger signal than a name guess
        if note and note.note and _PRIMARY_HINT.search(note.note):
            boost += 2      # this column's own description calls it the headline metric
        label = (note.note if note and note.note else _prettify(name))
        out.append({"column": name, "label": label, "unit": note.unit if note else "", "_rank": boost})
    out.sort(key=lambda c: (-c["_rank"], c["column"]))
    for c in out:
        del c["_rank"]
    return out


def _weight_columns(table: str, hidden: set[str]) -> list[dict]:
    cols = numeric_columns(table, hidden)
    ranked = sorted(cols, key=lambda c: (0 if _WEIGHT_HINT.search(c["column"]) else 1, cols.index(c)))
    return ranked


def _has_columns(table: str, *names: str, family: str | None = None) -> bool:
    cols = {n: t for n, t in _columns(table)}
    return all(n in cols and (family is None or _family(cols[n]) == family) for n in names)


def _geometry_kinds(table: str, sample: int = 200) -> set[str]:
    """Distinct GeoJSON geometry 'type' values found in ``table.geometry_json`` (a small sample is enough:
    a table mixes at most a couple of geometry kinds in practice, and this only decides lines vs polygons)."""
    if not _has_columns(table, "geometry_json"):
        return set()
    try:
        rows = store.get_conn().execute(
            f"SELECT DISTINCT json_extract_string(geometry_json, '$.type') FROM {_q(table)} "
            f"WHERE geometry_json IS NOT NULL USING SAMPLE {int(sample)} ROWS").fetchall()
        return {r[0] for r in rows if r[0]}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Tract-choropleth join discovery (reuses the catalog's own relationship graph)
# ---------------------------------------------------------------------------

TRACT_ANCHORS = ("nri_tracts", "census_tracts")


def _tract_join(table: str) -> dict | None:
    """How ``table`` reaches per-tract data: itself (if it IS an anchor), or a direct catalog
    relationship to one, keyed on ``tract_fips`` on both sides. Only direct edges are considered — this
    is deliberately not a multi-hop search, so a choropleth's SQL stays a single, obviously-correct JOIN."""
    if table in TRACT_ANCHORS:
        return {"anchor": table, "join_sql": ""}
    for rel in schema._relationships():
        for anchor in TRACT_ANCHORS:
            if rel.left_table == table and rel.right_table == anchor and rel.right_expr == "tract_fips":
                local = schema.qualify_join_expr(table, rel.left_expr)
                return {"anchor": anchor, "join_sql": f"JOIN {anchor} ON {local} = {anchor}.tract_fips"}
            if rel.right_table == table and rel.left_table == anchor and rel.left_expr == "tract_fips":
                local = schema.qualify_join_expr(table, rel.right_expr)
                return {"anchor": anchor, "join_sql": f"JOIN {anchor} ON {anchor}.tract_fips = {local}"}
    return None


def _geometry_status() -> tuple[bool, str]:
    gdf = geo_utils._load_tracts_gdf()
    if gdf is not None and not gdf.empty:
        return True, geo_utils.geometry_source()
    return False, ("No tract geometry is cached yet. Load the NRI shapefile (writes "
                   "data/nri/nri_geometry_cache.parquet automatically), or add TIGER/Line shapefiles to "
                   "data/shapefiles/, then re-run setup_data.py.")


# ---------------------------------------------------------------------------
# Spec derivation
# ---------------------------------------------------------------------------

def _classify(table: str, meta) -> dict | None:
    hidden = set(meta.hidden_columns)
    cols = {n: t for n, t in _columns(table)}
    if not cols:
        return None

    kinds = _geometry_kinds(table)
    if kinds:
        is_line = bool(kinds & {"LineString", "MultiLineString"})
        is_poly = bool(kinds & {"Polygon", "MultiPolygon"})
        if is_poly:
            measures = numeric_columns(table, hidden)
            return {"kind": "polygons", "group": "fill", "row_count": schema._row_count(table),
                    "measures": measures, "default_measure": measures[0]["column"] if measures else None,
                    "has_color_col": "color" in cols and "color" not in hidden}
        if is_line:
            return {"kind": "lines", "group": "overlay", "row_count": schema._row_count(table),
                    "has_color_col": "color" in cols and "color" not in hidden}
        return None

    if _has_columns(table, "lat", "lon", family="float"):
        n = schema._row_count(table)
        kind = "heat" if n > HEAT_ROW_THRESHOLD else "points"
        spec = {"kind": kind, "group": "overlay", "row_count": n, "lat_col": "lat", "lon_col": "lon"}
        if kind == "heat":
            weights = _weight_columns(table, hidden)
            spec["weight_options"] = weights
            spec["weight"] = weights[0]["column"] if weights and _WEIGHT_HINT.search(weights[0]["column"]) else None
        else:
            label_col = next((c for c, t in _columns(table) if t.upper().startswith("VARCHAR") and c not in hidden), None)
            spec["label_col"] = label_col
        return spec

    join = _tract_join(table)
    if join:
        available, reason = _geometry_status()
        measures = numeric_columns(table, hidden)
        if not measures:
            return None    # nothing to color a choropleth by
        return {"kind": "choropleth", "group": "fill", "row_count": schema._row_count(table),
                "measures": measures, "default_measure": measures[0]["column"],
                "anchor": join["anchor"], "available": available, "reason": None if available else reason}
    return None


# Tables handled specially in describe_layers() (houses: the interactive cluster layer; house_commute:
# the per-house decoration driven by commute.js) never go through the generic classifier, even though
# houses would otherwise also qualify as a plain lat/lon layer on its own.
_SPECIAL_TABLES = {"houses", "house_commute"}


def derive_layer_specs() -> list[dict]:
    """Every generic (non-special) map layer the live catalog currently supports."""
    out = []
    for name, meta in sorted(schema._tables().items()):
        if not meta.agent_visible or name in _SPECIAL_TABLES:
            continue
        spec = _classify(name, meta)
        if not spec:
            continue
        spec.update({"name": name, "title": title_for(name), "domain": meta.domain or "other",
                     "origin": meta.origin, "description": meta.description,
                     "available": spec.get("available", True), "reason": spec.get("reason"),
                     "special": None, "default_on": False})
        if name == "bike_routes":
            spec["renderer"] = "bike_routes"    # keeps the priority-resolved renderer; see get_bike_routes()
        out.append(spec)
    return out


def describe_layers() -> dict:
    """Full spec for the frontend panel: generic layers plus the two special, JS-driven ones."""
    layers = derive_layer_specs()
    layers.insert(0, {"name": "houses", "title": "Houses", "kind": "points", "group": "overlay",
                      "row_count": schema._row_count("houses"), "domain": "housing", "origin": "builtin",
                      "description": "Your saved houses.", "available": True, "reason": None,
                      "special": "houses", "default_on": True})
    commute_available = "house_commute" in schema._tables()
    layers.append({"name": "commute", "title": "Commute time to work", "kind": "decoration", "group": "overlay",
                   "row_count": None, "domain": "housing", "origin": "builtin",
                   "description": "Minutes to your saved work location, shown on each house.",
                   "available": commute_available, "reason": None if commute_available else "Not set up.",
                   "special": "commute", "default_on": False})
    fill_group = [l["name"] for l in layers if l["group"] == "fill"]
    return {"layers": layers, "exclusive_groups": {"fill": fill_group}, "catalog_version": schema.catalog_version()}


def _spec_for(name: str) -> dict:
    for l in derive_layer_specs():
        if l["name"] == name:
            return l
    raise LayerError(f"'{name}' is not a map layer. It may not exist, have no location data, or not be "
                     "visible to the assistant.")


# ---------------------------------------------------------------------------
# Generic data builders
# ---------------------------------------------------------------------------

def _bbox_params(west: float, south: float, east: float, north: float) -> tuple[str, list]:
    return "lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?", [south, north, west, east]


def get_points(table: str, west: float, south: float, east: float, north: float,
              limit: int = MAX_POINT_FEATURES) -> dict:
    spec = _spec_for(table)
    if spec["kind"] != "points":
        raise LayerError(f"'{table}' is a {spec['kind']} layer, not points.")
    hidden = set(schema._tables()[table].hidden_columns) | {"lat", "lon"}
    cols = _default_props(table, hidden)
    where, params = _bbox_params(west, south, east, north)
    df = store.query(
        f"SELECT lat, lon, {', '.join(_q(c) for c in cols)} FROM {_q(table)} WHERE {where} LIMIT {int(limit) + 1}",
        params)
    truncated = len(df) > limit
    df = df.head(limit)
    features = []
    for row in df.to_dict(orient="records"):
        lat, lon = row.pop("lat"), row.pop("lon")
        props = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props})
    return {"type": "FeatureCollection", "features": features, "feature_count": len(features), "truncated": truncated}


def get_heat(table: str, west: float, south: float, east: float, north: float,
            weight: str | None = None, grid_deg: float = DEFAULT_GRID_DEG, city: str | None = None) -> dict:
    spec = _spec_for(table)
    if spec["kind"] != "heat":
        raise LayerError(f"'{table}' is a {spec['kind']} layer, not heat.")
    if weight is not None and weight not in {c["column"] for c in spec.get("weight_options", [])}:
        raise LayerError(f"'{weight}' is not a valid weight column for '{table}'.")
    weight = weight if weight is not None else spec.get("weight")
    if not grid_deg or grid_deg <= 0:
        grid_deg = DEFAULT_GRID_DEG
    weight_expr = f"SUM({_q(weight)})" if weight else "COUNT(*)"
    where, params = _bbox_params(west, south, east, north)
    params = [grid_deg, grid_deg, grid_deg, grid_deg, *params]
    city_clause = ""
    if city and _has_columns(table, "city"):
        city_clause = f"AND {_q('city')} = ?"
        params.append(city)
    df = store.query(f"""
        SELECT ROUND(lat / ?) * ? AS glat, ROUND(lon / ?) * ? AS glon,
               COUNT(*) AS incident_count, {weight_expr} AS weighted_score
        FROM {_q(table)}
        WHERE {where} {city_clause}
        GROUP BY glat, glon
        ORDER BY weighted_score DESC
    """, params)
    truncated = len(df) > MAX_GRID_CELLS
    df = df.head(MAX_GRID_CELLS)
    if df.empty:
        return {"points": [], "max_weight": 0.0, "incident_count": 0, "cell_count": 0, "truncated": False,
                "grid_deg": grid_deg, "weight": weight}
    max_weight = float(df["weighted_score"].max())
    points = [[round(float(r.glat), 5), round(float(r.glon), 5), round(float(r.weighted_score), 3)]
             for r in df.itertuples(index=False)]
    return {"points": points, "max_weight": max_weight, "incident_count": int(df["incident_count"].sum()),
            "cell_count": len(df), "truncated": truncated, "grid_deg": grid_deg, "weight": weight}


def get_lines(table: str, west: float, south: float, east: float, north: float,
             limit: int = MAX_LINE_FEATURES) -> dict:
    spec = _spec_for(table)
    if spec["kind"] != "lines":
        raise LayerError(f"'{table}' is a {spec['kind']} layer, not lines.")
    if spec.get("renderer") == "bike_routes":
        return get_bike_routes(table, west, south, east, north, limit=limit)
    return _get_vector_features(table, west, south, east, north, limit)


def get_polygons(table: str, west: float, south: float, east: float, north: float,
                 measure: str | None = None, limit: int = MAX_POLYGON_FEATURES) -> dict:
    spec = _spec_for(table)
    if spec["kind"] != "polygons":
        raise LayerError(f"'{table}' is a {spec['kind']} layer, not polygons.")
    valid = {m["column"] for m in spec.get("measures", [])}
    if measure is not None and measure not in valid:
        raise LayerError(f"'{measure}' is not a valid measure for '{table}'.")
    measure = measure if measure is not None else spec.get("default_measure")
    result = _get_vector_features(table, west, south, east, north, limit, extra_cols=[measure] if measure else [])
    if measure:
        vals = [f["properties"].get(measure) for f in result["features"] if f["properties"].get(measure) is not None]
        result["measure"] = measure
        result["min"] = min(vals) if vals else None
        result["max"] = max(vals) if vals else None
    return result


_BLOB_COLUMNS = ("geometry_json", "properties_json", "raw_json", "min_lon", "min_lat", "max_lon", "max_lat")


_LABEL_HINT = re.compile(r"(name|label|title|address|city|route|category)", re.I)
MAX_GENERIC_PROPS = 15


def _default_props(table: str, hidden: set[str], exclude: set[str] = frozenset(), limit: int = MAX_GENERIC_PROPS) -> list[str]:
    """Display-worthy columns of ``table``, ranked so a popup or tooltip leads with whatever is most
    likely to matter (a name/label/measure) rather than whatever DESCRIBE happens to list first — a real
    table’s early columns are often internal bookkeeping (source file, parcel id, deed page)."""
    cols = [(c, t) for c, t in _columns(table) if c not in hidden and c not in exclude and c not in _BLOB_COLUMNS]

    def rank(item):
        name, sql_type = item
        if _LABEL_HINT.search(name):
            return 0
        if _family(sql_type) in ("int", "float") and not _NOT_A_MEASURE.search(name):
            return 1
        return 2
    order = sorted(range(len(cols)), key=lambda i: (rank(cols[i]), i))
    return [cols[i][0] for i in order[:limit]]


def _get_vector_features(table: str, west: float, south: float, east: float, north: float,
                         limit: int, extra_cols: list[str] | None = None) -> dict:
    hidden = set(schema._tables()[table].hidden_columns)
    have_bbox = _has_columns(table, "min_lon", "min_lat", "max_lon", "max_lat")
    requested = [c for c in (extra_cols or []) if c]
    cols = ["geometry_json"] + requested + _default_props(table, hidden, exclude=set(requested), limit=10)
    seen: set[str] = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    where, params = ("min_lon <= ? AND max_lon >= ? AND min_lat <= ? AND max_lat >= ?",
                     [east, west, north, south]) if have_bbox else ("1=1", [])
    df = store.query(f"SELECT {', '.join(_q(c) for c in cols)} FROM {_q(table)} WHERE {where} "
                     f"LIMIT {int(limit) + 1}", params)
    truncated = len(df) > limit
    df = df.head(limit)
    features = []
    for row in df.to_dict(orient="records"):
        raw = row.pop("geometry_json")
        try:
            geometry = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            geometry = None
        if not geometry:
            continue
        props = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        features.append({"type": "Feature", "geometry": geometry, "properties": props})
    return {"type": "FeatureCollection", "features": features, "feature_count": len(features), "truncated": truncated}


def get_tract_choropleth(table: str, measure: str | None, west: float, south: float, east: float, north: float,
                         max_tracts: int = MAX_TRACTS) -> dict:
    """Tract polygons intersecting the bbox, colored by ``measure`` from ``table``.

    Geometry always comes from ``services/geo_utils.py``'s cached tract polygons (the same source used to
    assign each house its ``tract_fips``); values come from ``table``, joined to the tract anchor by the
    catalog's own relationship when ``table`` is not ``nri_tracts``/``census_tracts`` itself. Many rows per
    tract (a block-group dataset, say) are averaged; this is correct for both cardinalities since a
    one-to-one join just averages a single value with itself.
    """
    spec = _spec_for(table)
    if spec["kind"] != "choropleth":
        raise LayerError(f"'{table}' is a {spec['kind']} layer, not a choropleth.")
    valid = {m["column"] for m in spec["measures"]}
    if measure is not None and measure not in valid:
        raise LayerError(f"'{measure}' is not a valid measure for '{table}'.")
    measure = measure if measure is not None else spec["default_measure"]

    tracts_gdf = geo_utils._load_tracts_gdf()
    if tracts_gdf is None or tracts_gdf.empty:
        _, reason = _geometry_status()
        return {"type": "FeatureCollection", "features": [], "tract_count": 0, "truncated": False, "warning": reason}

    import geopandas as gpd
    from shapely.geometry import box as _box

    bbox_poly = _box(west, south, east, north)
    idx = list(tracts_gdf.sindex.query(bbox_poly, predicate="intersects"))
    subset = tracts_gdf.iloc[idx]
    if subset.empty:
        return {"type": "FeatureCollection", "features": [], "tract_count": 0, "truncated": False}
    truncated = len(subset) > max_tracts
    if truncated:
        subset = subset.iloc[:max_tracts]

    join = _tract_join(table)
    fips_list = [str(f) for f in subset["tract_fips"].tolist()]
    placeholders = ",".join("?" for _ in fips_list)
    if table == join["anchor"]:
        attrs = store.query(f'SELECT tract_fips, {_q(measure)} AS value FROM {_q(table)} '
                            f"WHERE tract_fips IN ({placeholders})", fips_list)
    else:
        attrs = store.query(
            f'SELECT {join["anchor"]}.tract_fips AS tract_fips, AVG({_q(table)}.{_q(measure)}) AS value '
            f'FROM {_q(table)} {join["join_sql"]} '
            f'WHERE {join["anchor"]}.tract_fips IN ({placeholders}) '
            f'GROUP BY {join["anchor"]}.tract_fips', fips_list)

    merged = subset.merge(attrs, on="tract_fips", how="left")
    merged["geometry"] = merged["geometry"].simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged["value"] = merged["value"].where(pd.notnull(merged["value"]), None)

    result = json.loads(merged[["tract_fips", "value", "geometry"]].to_json())
    result["truncated"] = truncated
    result["tract_count"] = len(merged)
    result["measure"] = measure
    result["measure_label"] = next((m["label"] for m in spec["measures"] if m["column"] == measure), measure)
    vals = [v for v in merged["value"].tolist() if v is not None]
    result["min"], result["max"] = (min(vals), max(vals)) if vals else (None, None)
    if attrs.empty:
        result["warning"] = f"Tract geometry is loaded but '{table}' has no matching rows in view."
    return result


def get_layer_data(name: str, **kwargs) -> dict:
    spec = _spec_for(name)
    kind = spec["kind"]
    if kind in ("points",):
        return get_points(name, **{k: kwargs[k] for k in ("west", "south", "east", "north") if k in kwargs},
                          limit=kwargs.get("limit", MAX_POINT_FEATURES))
    if kind == "heat":
        return get_heat(name, kwargs["west"], kwargs["south"], kwargs["east"], kwargs["north"],
                        weight=kwargs.get("weight"), grid_deg=kwargs.get("grid_deg", DEFAULT_GRID_DEG),
                        city=kwargs.get("city"))
    if kind == "lines":
        return get_lines(name, kwargs["west"], kwargs["south"], kwargs["east"], kwargs["north"],
                         limit=kwargs.get("limit", MAX_LINE_FEATURES))
    if kind == "polygons":
        return get_polygons(name, kwargs["west"], kwargs["south"], kwargs["east"], kwargs["north"],
                            measure=kwargs.get("measure"), limit=kwargs.get("limit", MAX_POLYGON_FEATURES))
    if kind == "choropleth":
        return get_tract_choropleth(name, kwargs.get("measure"), kwargs["west"], kwargs["south"],
                                    kwargs["east"], kwargs["north"], max_tracts=kwargs.get("limit", MAX_TRACTS))
    raise LayerError(f"'{name}' has no server-rendered data (it is drawn client-side).")


# ---------------------------------------------------------------------------
# Bike routes: preserved unchanged from the former services/layers.py
# ---------------------------------------------------------------------------

# Ground-truth display hierarchy for overlapping BikePGH classifications. This does NOT change the raw
# bike_routes data or routing graph. It only prevents the same physical line from being painted as
# several categories in a visualization. More-specific facility types win over generic route designations.
BIKE_DISPLAY_PRIORITY = {
    "protected_bike_lanes": 100, "bike_lanes": 90, "trails": 80, "bikeable_sidewalks": 70,
    "sharrows": 60, "cautionary_bike_route": 50, "on_street_bike_route": 40,
}


def _canonicalize_bike_features(features: list[dict]) -> list[dict]:
    """Remove purely visual category overlap without changing raw data.

    Ground-truth BikePGH contains genuine duplicate/overlapping classifications such as On Street Bike
    Route + Bike Lane and On Street Bike Route + Sharrows. For visualization, retain the more specific
    category on the overlapping geometry and subtract it from the lower-priority category.
    """
    try:
        from shapely.geometry import mapping, shape
        from shapely.ops import unary_union
    except Exception:
        return features

    grouped = []
    for feature in features:
        try:
            geom = shape(feature["geometry"])
        except Exception:
            continue
        if geom.is_empty:
            continue
        props = feature.get("properties") or {}
        layer_type = props.get("layer_type") or ""
        priority = BIKE_DISPLAY_PRIORITY.get(layer_type, 0)
        grouped.append((priority, layer_type, geom, feature))

    claimed = None
    output = []
    for priority, layer_type, geom, feature in sorted(
            grouped, key=lambda x: (-x[0], x[1], str((x[3].get("properties") or {}).get("route_id", "")))):
        display_geom = geom
        if claimed is not None and not claimed.is_empty:
            try:
                display_geom = geom.difference(claimed)
            except Exception:
                display_geom = geom
        if display_geom is None or display_geom.is_empty:
            continue
        prop = dict(feature.get("properties") or {})
        prop["display_priority"] = priority
        try:
            feature = {"type": "Feature", "geometry": mapping(display_geom), "properties": prop}
        except Exception:
            feature = dict(feature)
            feature["properties"] = prop
        output.append(feature)
        claimed = display_geom if claimed is None else unary_union([claimed, display_geom])
    return output


def get_bike_routes(table: str, west: float, south: float, east: float, north: float,
                    city: str | None = None, limit: int = MAX_LINE_FEATURES, exclusive: bool = True) -> dict:
    params = [east, west, north, south]
    city_clause = ""
    if city:
        city_clause = "AND city = ?"
        params.append(city.strip().lower())
    df = store.query(f"""
        SELECT route_id, city, layer_type, layer_label, color, source_file, geometry_json, properties_json
        FROM {_q(table)}
        WHERE min_lon <= ? AND max_lon >= ? AND min_lat <= ? AND max_lat >= ? {city_clause}
        ORDER BY city, layer_label, route_id
        LIMIT {int(limit)}
    """, params)
    features = []
    for row in df.to_dict(orient="records"):
        # NaN-safety matters more here than in most generic paths: a bbox slice can legitimately return
        # rows where every value of an optional column (color, in particular) is NULL, and pandas then
        # infers that whole column as float64 NaN rather than None/object - which json.dumps (Starlette's
        # JSONResponse passes allow_nan=False) rejects outright.
        clean = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        try:
            geometry = json.loads(clean["geometry_json"]) if clean["geometry_json"] else None
        except (TypeError, ValueError):
            geometry = None
        if not geometry:
            continue
        try:
            props = json.loads(clean["properties_json"]) if clean["properties_json"] else {}
        except (TypeError, ValueError):
            props = {}
        props.update({"route_id": clean["route_id"], "city": clean["city"], "layer_type": clean["layer_type"],
                     "layer_label": clean["layer_label"], "color": clean["color"], "source_file": clean["source_file"]})
        features.append({"type": "Feature", "geometry": geometry, "properties": props})
    if exclusive:
        features = _canonicalize_bike_features(features)
    return {"type": "FeatureCollection", "features": features, "feature_count": len(features),
            "truncated": len(df) >= limit, "exclusive_display": exclusive}

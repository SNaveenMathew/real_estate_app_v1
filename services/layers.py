"""
services/layers.py

Query/aggregation logic behind the optional map overlays ('Crime', 'NRI',
and 'Bike Lanes'). Both are viewport-scoped: main.py's endpoints require the caller to
pass the current map bounding box, and only what's visible is returned — a
national-scale dataset (crime_incidents can run into the millions of rows
across cities; the NRI tract geometry cache covers ~85k tracts nationwide)
should never be shipped whole to the browser just to render whatever's on
screen right now.

Kept separate from db/duckdb_store.py (raw table access) and
services/geo_utils.py (tract geometry loading / FIPS resolution) since this
module's job is specifically "turn stored data into something a map layer
can render," reusing both of those rather than duplicating their logic.
"""
import json
from typing import Optional

import pandas as pd

import db.duckdb_store as store
from services import geo_utils


# ── Crime heatmap ────────────────────────────────────────────────────────────

DEFAULT_GRID_DEG = 0.003     # ≈ 250-300m at US latitudes — a block/neighborhood-scale cell
MAX_GRID_CELLS = 6000        # safety cap so a zoomed-out view can't return an unbounded payload


def get_crime_heatmap(west: float, south: float, east: float, north: float,
                       grid_deg: float = DEFAULT_GRID_DEG,
                       city: Optional[str] = None) -> dict:
    """
    Aggregate crime incidents within the bbox into a coarse lat/lon grid,
    weighted by severity, ready for a Leaflet.heat layer.

    Returns:
        {
          "points": [[lat, lon, weighted_score], ...],   # one per grid cell
          "max_weight": float,       # for the heat layer's `max` option
          "incident_count": int,     # total incidents represented
          "cell_count": int,
          "truncated": bool,         # True if capped at MAX_GRID_CELLS
          "grid_deg": float,
        }
    """
    if not grid_deg or grid_deg <= 0:
        grid_deg = DEFAULT_GRID_DEG

    params: list = [grid_deg, grid_deg, grid_deg, grid_deg, south, north, west, east]
    city_clause = ""
    if city:
        city_clause = "AND city = ?"
        params.append(city)

    df = store.query(f"""
        SELECT
            ROUND(lat / ?) * ?   AS glat,
            ROUND(lon / ?) * ?   AS glon,
            COUNT(*)             AS incident_count,
            SUM(severity_weight) AS weighted_score
        FROM crime_incidents
        WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
        {city_clause}
        GROUP BY glat, glon
        ORDER BY weighted_score DESC
    """, params)

    truncated = len(df) > MAX_GRID_CELLS
    if truncated:
        df = df.head(MAX_GRID_CELLS)

    if df.empty:
        return {"points": [], "max_weight": 0.0, "incident_count": 0,
                "cell_count": 0, "truncated": False, "grid_deg": grid_deg}

    max_weight = float(df["weighted_score"].max())
    points = [
        [round(float(r.glat), 5), round(float(r.glon), 5), round(float(r.weighted_score), 3)]
        for r in df.itertuples(index=False)
    ]
    return {
        "points": points,
        "max_weight": max_weight,
        "incident_count": int(df["incident_count"].sum()),
        "cell_count": len(df),
        "truncated": truncated,
        "grid_deg": grid_deg,
    }


# ── NRI choropleth ───────────────────────────────────────────────────────────

MAX_TRACTS = 2500
SIMPLIFY_TOLERANCE_DEG = 0.0004   # ~40m — trims polygon vertex count without visible distortion at map scale


def get_nri_choropleth(west: float, south: float, east: float, north: float,
                        max_tracts: int = MAX_TRACTS) -> dict:
    """
    NRI-scored census tract polygons intersecting the bbox, as a GeoJSON
    FeatureCollection. Geometry comes from the cache geo_utils already
    maintains for tract-FIPS spatial joins (data/nri/nri_geometry_cache.parquet,
    written by data_loader.load_nri() from the NRI shapefile — or TIGER/Line
    shapefiles if that's what's available); risk attributes come from
    nri_tracts. No new download or table is needed for this layer — it's the
    same geometry/attributes the app already loads, just filtered to the
    viewport and shaped as GeoJSON.

    Returns a GeoJSON FeatureCollection with two extra top-level keys:
      "truncated": bool   — True if capped at max_tracts
      "tract_count": int
      "warning": str      — present only if geometry exists but nri_tracts
                             has no matching attribute rows (i.e. NRI hasn't
                             been loaded, or only the shapefile-less CSV path
                             was used and geometry came from TIGER instead)
    """
    tracts_gdf = geo_utils._load_tracts_gdf()
    if tracts_gdf is None or tracts_gdf.empty:
        return {
            "type": "FeatureCollection", "features": [],
            "tract_count": 0, "truncated": False,
            "warning": ("No tract geometry available yet. Load the NRI shapefile "
                        "(preferred) or add TIGER/Line shapefiles to data/shapefiles/, "
                        "then re-run setup_data.py."),
        }

    import geopandas as gpd
    from shapely.geometry import box as _box

    bbox_poly = _box(west, south, east, north)
    candidate_idx = list(tracts_gdf.sindex.query(bbox_poly, predicate="intersects"))
    subset = tracts_gdf.iloc[candidate_idx]

    if subset.empty:
        return {"type": "FeatureCollection", "features": [], "tract_count": 0, "truncated": False}

    truncated = len(subset) > max_tracts
    if truncated:
        subset = subset.iloc[:max_tracts]

    fips_list = [str(f) for f in subset["tract_fips"].tolist()]
    placeholders = ",".join(["?"] * len(fips_list))
    attrs = store.query(f"""
        SELECT tract_fips, county_name, state_name, risk_score, risk_ratng,
               risk_npctl, eal_valt, sovi_ratng, resl_ratng
        FROM nri_tracts
        WHERE tract_fips IN ({placeholders})
    """, fips_list)

    merged = subset.merge(attrs, on="tract_fips", how="left")
    merged["geometry"] = merged["geometry"].simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")

    # JSON can't represent NaN — swap in None so to_json() emits valid `null`
    # (same principle as duckdb_store.query_json / main.py's _safe_dict).
    attr_cols = [c for c in merged.columns if c != "geometry"]
    merged[attr_cols] = merged[attr_cols].where(pd.notnull(merged[attr_cols]), None)

    result = json.loads(merged.to_json())
    result["truncated"] = truncated
    result["tract_count"] = len(merged)
    if attrs.empty:
        result["warning"] = ("Tract geometry is loaded but nri_tracts has no matching rows — "
                              "NRI data may not be loaded yet. Run: python setup_data.py --only nri")
    return result


# ── Bike routes ──────────────────────────────────────────────────────────────

MAX_BIKE_FEATURES = 5000


def get_bike_routes(west: float, south: float, east: float, north: float,
                    city: Optional[str] = None,
                    max_features: int = MAX_BIKE_FEATURES) -> dict:
    """Return BikePGH-style line features intersecting the viewport as GeoJSON.

    The loader stores a WGS-84 bbox for every feature. The initial SQL filter
    therefore avoids deserializing geometries outside the viewport; the client
    receives only the seven standardized bike sublayers.
    """
    params = [east, west, north, south]
    city_clause = ""
    if city:
        city_clause = "AND city = ?"
        params.append(city.strip().lower())

    df = store.query(f"""
        SELECT route_id, city, layer_type, layer_label, color, source_file, geometry_json, properties_json
        FROM bike_routes
        WHERE min_lon <= ? AND max_lon >= ?
          AND min_lat <= ? AND max_lat >= ?
          {city_clause}
        ORDER BY city, layer_label, route_id
        LIMIT {int(max_features)}
    """, params)

    features = []
    for r in df.itertuples(index=False):
        try:
            geometry = json.loads(r.geometry_json)
        except Exception:
            continue
        try:
            props = json.loads(r.properties_json) if r.properties_json else {}
        except Exception:
            props = {}
        props.update({
            "route_id": r.route_id,
            "city": r.city,
            "layer_type": r.layer_type,
            "layer_label": r.layer_label,
            "color": r.color,
            "source_file": r.source_file,
        })
        features.append({"type": "Feature", "geometry": geometry, "properties": props})

    return {
        "type": "FeatureCollection",
        "features": features,
        "feature_count": len(features),
        "truncated": len(df) >= max_features,
    }

"""`Bike routes` loaders, moved out of services/data_loader.py.

Behaviour is unchanged: the facade in services/data_loader.py re-exports every name here, so
`from services.data_loader import ...` (and scripts that call it) keep working.
"""
from __future__ import annotations

"""
Data loader: reads raw files and populates DuckDB tables.
Run via setup_data.py — only needed when data files change.
"""
import hashlib

import pandas as pd


from config import settings
import db.duckdb_store as store




__all__ = ['_BIKE_LAYER_SPECS', '_norm_folder_name', '_bike_route_id', '_json_safe_value', 'load_bike_routes']

# ── Bike routes ────────────────────────────────────────────────────────────────
# Standardizes any BikePGH-style line datasets stored as:
#   data/bike/<city>/<layer folder>/*.shp
# The layer folder names are mapped to a stable layer_type and the R-script
# colors are preserved verbatim for the frontend visualization.

_BIKE_LAYER_SPECS = {
    "bike lanes":               ("bike_lanes", "Bike Lanes", "steelblue"),
    "bikeable sidewalks":       ("bikeable_sidewalks", "Bikeable Sidewalks", "lightblue"),
    "cautionary bike route":    ("cautionary_bike_route", "Cautionary Bike Route", "red"),
    "on street bike route":     ("on_street_bike_route", "On Street Bike Route", "lightgreen"),
    "protected bike lanes":     ("protected_bike_lanes", "Protected Bike Lanes", "darkgreen"),
    "sharrows":                 ("sharrows", "Sharrows", "orange"),
    "trails":                   ("trails", "Trails", "pink"),
}


def _norm_folder_name(value: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", str(value).strip().lower())


def _bike_route_id(city: str, layer_type: str, source_file: str, row_pos: int) -> str:
    key = f"{city}:{layer_type}:{source_file}:{row_pos}"
    return hashlib.md5(key.encode()).hexdigest()[:20]


def _json_safe_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_bike_routes() -> int:
    """
    Load all recognized bike-route shapefiles beneath data/bike/<city>/.\n
    No city-specific parser is required. The city folder and recognized layer\n    folder determine the standardized metadata, while GeoPandas handles each\n    source CRS and reprojects geometry to EPSG:4326. Multiple shapefiles per\n    layer/city are supported.\n    """
    import geopandas as gpd
    import json as _json

    bike_root = settings.data_dir / "bike"
    if not bike_root.exists():
        print(f"  No bike data directory found at {bike_root}")
        return 0

    rows = []
    city_dirs = sorted(p for p in bike_root.iterdir() if p.is_dir())
    if not city_dirs:
        print(f"  No city folders found under {bike_root}")
        return 0

    for city_dir in city_dirs:
        city = city_dir.name.strip().lower()
        for layer_dir in sorted(p for p in city_dir.iterdir() if p.is_dir()):
            spec = _BIKE_LAYER_SPECS.get(_norm_folder_name(layer_dir.name))
            if not spec:
                continue
            layer_type, layer_label, color = spec
            shp_files = sorted(layer_dir.glob("*.shp"))
            if not shp_files:
                continue

            for shp_path in shp_files:
                print(f"  {city}: {layer_label} — {shp_path.name}")
                try:
                    gdf = gpd.read_file(shp_path)
                    if gdf.empty:
                        print("    No features found")
                        continue
                    if gdf.crs is None:
                        print("    ⚠ Missing CRS; skipping file")
                        continue
                    if str(gdf.crs).upper() != "EPSG:4326":
                        gdf = gdf.to_crs("EPSG:4326")
                    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
                    if gdf.empty:
                        print("    No non-empty geometries found")
                        continue

                    for i, (_, feature) in enumerate(gdf.iterrows()):
                        geom = feature.geometry
                        minx, miny, maxx, maxy = geom.bounds
                        props = {
                            str(k): _json_safe_value(v)
                            for k, v in feature.drop(labels=["geometry"]).to_dict().items()
                        }
                        rows.append({
                            "route_id": _bike_route_id(city, layer_type, shp_path.name, i),
                            "city": city,
                            "layer_type": layer_type,
                            "layer_label": layer_label,
                            "color": color,
                            "source_file": str(shp_path.relative_to(bike_root)),
                            "min_lon": float(minx),
                            "min_lat": float(miny),
                            "max_lon": float(maxx),
                            "max_lat": float(maxy),
                            "geometry_json": _json.dumps(geom.__geo_interface__, ensure_ascii=False),
                            "properties_json": _json.dumps(props, ensure_ascii=False),
                        })
                    print(f"    ✓ {len(gdf):,} features")
                except Exception as e:
                    print(f"    Error loading {shp_path}: {e}")
                    import traceback; traceback.print_exc()

    if not rows:
        print("  No recognized bike route shapefiles found. Expected paths like")
        print("  data/bike/pittsburgh/Bike Lanes/Bike Lanes.shp")
        return 0

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["route_id"])
    store.upsert_df("bike_routes", out)
    print(f"  ✓ {len(out):,} bike route features loaded across {out['city'].nunique()} cities")
    return len(out)

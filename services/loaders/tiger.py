"""Census TIGER/Line boundary loaders (ZCTA5 + Census Place).

The files are authoritative, replace-on-load layers rather than append-only
observations: one TIGER vintage describes the whole country (ZCTA) or a state
(Place), so a successful load replaces the table instead of accumulating rows
from old vintages.  Every original TIGER attribute is preserved in
``properties_json`` while the columns the catalog and map use are projected to
stable names.  Geometry is stored as GeoJSON text, matching
``services/dataset_readers._read_geo`` and therefore ``services/map_layers.py``,
which already recognises Polygon/MultiPolygon GeoJSON as fill layers.

Accepted inputs: an extracted ``.shp`` set or the original Census ``.zip``.
Files are discovered under ``settings.tiger_dir`` and in matching top-level
folders under ``settings.data_dir`` (for example
``data/cb_2025_us_place_500k/``), by filename (``zcta`` or ``place``), so the
same loader backs both ``setup_data.py`` and a Data-page upload.  The source
file is never modified.
"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
import db.duckdb_store as store

__all__ = [
    "load_tiger_zcta5",
    "load_tiger_places",
    "load_bike_route_zctas",
    "load_bike_route_places",
    "normalize_zcta_frame",
    "normalize_place_frame",
    "boundary_columns",
    "tiger_boundary_paths",
]

_SQM_PER_SQ_MI = 2_589_988.110336
_PLACE_SUFFIX = re.compile(
    r"\s+(city|town|village|borough|township|municipality|CDP)\.?$",
    re.IGNORECASE,
)


# ── Input discovery and reading ──────────────────────────────────────────────


_VINTAGE = re.compile(r"(20\d{2})")


def _boundary_order(path: Path) -> tuple[int, int, float, bool, str]:
    """Sort key: newest vintage, full TIGER/Line over cartographic, then mtime.

    Filesystem mtimes are not a reliable vintage signal: extracting an archive
    can preserve the member's timestamp, so a 2025 cartographic file can look
    older than a 2026 TIGER/Line state file even after a fresh download.  The
    year in the Census filename is stable across copies, so it leads.  A
    Data-page upload of the same vintage still wins through mtime.
    """
    match = _VINTAGE.search(path.stem)
    vintage = int(match.group(1)) if match else 0
    quality = 1 if path.stem.lower().startswith("tl_") else 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (vintage, quality, mtime, path.suffix.lower() == ".zip", str(path).lower())


def _source_label(path: Path) -> str:
    """A stable provenance label relative to the data tree.

    Boundary files may live outside ``tiger_dir`` (a national extract directly
    under ``data/``), so a bare ``relative_to(tiger_dir)`` would raise before
    the row is written.
    """
    for base in (Path(settings.tiger_dir), Path(settings.data_dir)):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return path.name


def _boundary_roots(kind: str) -> list[Path]:
    """Directories that may contain Census boundary files for ``kind``.

    ``settings.tiger_dir`` is canonical.  A directory directly under
    ``settings.data_dir`` whose name contains ``kind`` is also accepted, so an
    extracted national file can sit at ``data/cb_2025_us_place_500k`` (or a
    future ``data/tl_2027_us_zcta520``) without being moved.  Only top-level
    entries are considered; walking the whole data tree would traverse the ZCTA
    and Chroma folders on every load.
    """
    kind = kind.lower()
    roots = [Path(settings.tiger_dir)]
    data_root = Path(settings.data_dir)
    if data_root.exists():
        for child in sorted(data_root.iterdir(), key=lambda p: str(p).lower()):
            if child.is_dir() and kind in child.name.lower() and child not in roots:
                roots.append(child)
    return roots


def tiger_boundary_paths(kind: str) -> list[Path]:
    """Every supported ``.shp``/``.zip`` for ``kind``, best source last.

    Discovery covers ``settings.tiger_dir``, matching top-level directories
    under ``settings.data_dir``, and a boundary file dropped directly in
    ``data/``.  A zip and its extracted shapefile are two copies of one
    dataset, so the same stem is read once; keeping both would parse a
    national layer twice for no new rows.  The loaders concatenate files and
    keep the last row per boundary key, which makes the order meaningful:
    newest vintage year, then full TIGER/Line over cartographic, then newest
    file, then zip.
    """
    kind = kind.lower()
    paths: set[Path] = set()
    for root in _boundary_roots(kind):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".shp", ".zip"):
                continue
            if kind in path.name.lower():
                paths.add(path)

    data_root = Path(settings.data_dir)
    if data_root.exists():
        for child in data_root.iterdir():
            if (
                child.is_file()
                and child.suffix.lower() in (".shp", ".zip")
                and kind in child.name.lower()
            ):
                paths.add(child)

    unique: dict[str, Path] = {}
    for path in paths:
        key = path.stem.lower()
        current = unique.get(key)
        if current is None or _boundary_order(path) > _boundary_order(current):
            unique[key] = path
    return sorted(unique.values(), key=_boundary_order)


def _shapefile_in_zip(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        shps = [name for name in zf.namelist() if name.lower().endswith(".shp")]
    if not shps:
        raise ValueError("This zip doesn't contain a .shp file — expected a TIGER/Line boundary zip.")
    return shps[0]


def _read_boundary(path: Path, *, columns_only: bool = False, rows: int | None = None):
    """Read one ``.shp`` or zipped shapefile.

    GeoPandas/pyogrio can read zipped shapefiles through ``zip://`` in recent
    versions; if that fails we extract just that archive to a temporary
    directory.  ``columns_only`` uses pyogrio's metadata read so an upload
    precheck never materialises a national ZCTA layer.
    """
    path = Path(path)
    if columns_only:
        try:
            import pyogrio

            if path.suffix.lower() == ".zip":
                member = _shapefile_in_zip(path)
                with tempfile.TemporaryDirectory() as tmp:
                    with zipfile.ZipFile(path) as zf:
                        zf.extract(member, tmp)
                    info = pyogrio.read_info(Path(tmp) / member)
            else:
                info = pyogrio.read_info(path)
            return list(info.get("fields", []))
        except Exception:
            # Let boundary_columns() retry with a one-row geopandas read; this
            # covers older pyogrio builds and unusual archives.
            raise

    import geopandas as gpd

    def _read(target):
        return gpd.read_file(target, rows=rows) if rows is not None else gpd.read_file(target)

    try:
        return _read(f"zip://{path}") if path.suffix.lower() == ".zip" else _read(path)
    except Exception:
        if path.suffix.lower() != ".zip":
            raise
        member = _shapefile_in_zip(path)
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as zf:
                zf.extract(member, tmp)
            return _read(Path(tmp) / member)


def boundary_columns(path: Path) -> list[str]:
    """Attribute names of a TIGER shapefile/zip without reading every feature."""
    try:
        cols = _read_boundary(Path(path), columns_only=True)
    except Exception:
        cols = []
    if not cols:
        # pyogrio returned no metadata for a format it doesn't inspect; one row
        # is enough to recover the schema and is still cheap.
        try:
            gdf = _read_boundary(Path(path), rows=1)
            cols = [str(c) for c in gdf.columns if str(c).lower() != "geometry"]
        except Exception:
            cols = []
    return [str(c) for c in cols]


def _read_boundary_frame(path: Path):
    import geopandas as gpd

    gdf = _read_boundary(path)
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        return None
    geom_name = gdf.geometry.name
    # Rename attributes only: renaming the active geometry column (e.g. to
    # "GEOMETRY") makes GeoPandas treat the frame as having no active geometry.
    gdf = gdf.rename(columns={c: str(c).strip().upper() for c in gdf.columns if c != geom_name})
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf if not gdf.empty else None


# ── Normalisation ────────────────────────────────────────────────────────────


def _id_series(gdf, candidates: tuple[str, ...], width: int) -> pd.Series:
    for name in candidates:
        if name in gdf.columns:
            s = gdf[name].astype("string").str.strip()
            return s.str.replace(r"\.0$", "", regex=True).str.zfill(width)
    return pd.Series([pd.NA] * len(gdf), index=gdf.index, dtype="string")


def _number_series(gdf, candidates: tuple[str, ...]) -> pd.Series:
    for name in candidates:
        if name in gdf.columns:
            return pd.to_numeric(gdf[name], errors="coerce")
    return pd.Series([np.nan] * len(gdf), index=gdf.index, dtype="float64")


def _text_series(gdf, candidates: tuple[str, ...]) -> pd.Series:
    for name in candidates:
        if name in gdf.columns:
            return gdf[name].astype("string").str.strip()
    return pd.Series([pd.NA] * len(gdf), index=gdf.index, dtype="string")


def _short_place_name(name) -> str | None:
    if pd.isna(name):
        return None
    value = str(name).strip()
    value = _PLACE_SUFFIX.sub("", value).strip()
    return value or None


def _json_value(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def _properties(gdf) -> list[str]:
    """Every original attribute as JSON, excluding the geometry column."""
    geom_name = gdf.geometry.name
    out = []
    for record in gdf.drop(columns=[geom_name]).to_dict("records"):
        out.append(json.dumps({str(k): _json_value(v) for k, v in record.items()}, ensure_ascii=False))
    return out


def _geometry_json(gdf) -> list[str | None]:
    out = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            out.append(None)
        else:
            out.append(json.dumps(geom.__geo_interface__, ensure_ascii=False))
    return out


def _bounds_and_points(gdf):
    bounds = gdf.geometry.bounds
    points = gdf.geometry.representative_point()
    return {
        "min_lon": bounds["minx"].to_numpy(),
        "min_lat": bounds["miny"].to_numpy(),
        "max_lon": bounds["maxx"].to_numpy(),
        "max_lat": bounds["maxy"].to_numpy(),
        "repr_lon": points.x.to_numpy(),
        "repr_lat": points.y.to_numpy(),
    }


def normalize_zcta_frame(gdf, source_file: str) -> pd.DataFrame:
    """Project a raw TIGER ZCTA GeoDataFrame onto the ``zcta_boundaries`` columns."""
    gdf = gdf.copy()
    zcta = _id_series(gdf, ("ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID"), 5)
    aland = _number_series(gdf, ("ALAND20", "ALAND"))
    awater = _number_series(gdf, ("AWATER20", "AWATER"))
    lat = _number_series(gdf, ("INTPTLAT20", "INTPTLAT"))
    lon = _number_series(gdf, ("INTPTLON20", "INTPTLON"))
    pos = _bounds_and_points(gdf)
    lat = lat.fillna(pd.Series(pos["repr_lat"], index=gdf.index))
    lon = lon.fillna(pd.Series(pos["repr_lon"], index=gdf.index))
    out = pd.DataFrame(
        {
            "zcta5": zcta,
            "geo_id": _id_series(gdf, ("GEOID20", "GEOID", "ZCTA5CE20"), 5),
            "classfp": _text_series(gdf, ("CLASSFP20", "CLASSFP")),
            "mtfcc": _text_series(gdf, ("MTFCC20", "MTFCC")),
            "funcstat": _text_series(gdf, ("FUNCSTAT20", "FUNCSTAT")),
            "aland": aland,
            "awater": awater,
            "land_area_sq_mi": aland / _SQM_PER_SQ_MI,
            "water_area_sq_mi": awater / _SQM_PER_SQ_MI,
            "lat": lat,
            "lon": lon,
            "min_lon": pos["min_lon"],
            "min_lat": pos["min_lat"],
            "max_lon": pos["max_lon"],
            "max_lat": pos["max_lat"],
            "geometry_json": _geometry_json(gdf),
            "properties_json": _properties(gdf),
            "source_file": source_file,
        }
    )
    out = out[out["zcta5"].notna() & (out["zcta5"].str.len() == 5)]
    out = out.drop_duplicates(subset=["zcta5"], keep="last").reset_index(drop=True)
    return out


def normalize_place_frame(gdf, source_file: str) -> pd.DataFrame:
    """Project a raw TIGER Place GeoDataFrame onto the ``place_boundaries`` columns."""
    gdf = gdf.copy()
    place_geoid = _id_series(gdf, ("GEOID", "GEOIDFQ"), 7)
    name = _text_series(gdf, ("NAME", "NAMELSAD"))
    short_name = pd.Series([_short_place_name(v) for v in name], index=gdf.index, dtype="string")
    aland = _number_series(gdf, ("ALAND",))
    awater = _number_series(gdf, ("AWATER",))
    lat = _number_series(gdf, ("INTPTLAT",))
    lon = _number_series(gdf, ("INTPTLON",))
    pos = _bounds_and_points(gdf)
    lat = lat.fillna(pd.Series(pos["repr_lat"], index=gdf.index))
    lon = lon.fillna(pd.Series(pos["repr_lon"], index=gdf.index))
    out = pd.DataFrame(
        {
            "place_geoid": place_geoid,
            "place_fips": _id_series(gdf, ("PLACEFP",), 5),
            "state_fips": _id_series(gdf, ("STATEFP",), 2),
            "name": name,
            "short_name": short_name,
            "namelsad": _text_series(gdf, ("NAMELSAD",)),
            "lsad": _text_series(gdf, ("LSAD",)),
            "classfp": _text_series(gdf, ("CLASSFP",)),
            "funcstat": _text_series(gdf, ("FUNCSTAT",)),
            "aland": aland,
            "awater": awater,
            "land_area_sq_mi": aland / _SQM_PER_SQ_MI,
            "water_area_sq_mi": awater / _SQM_PER_SQ_MI,
            "lat": lat,
            "lon": lon,
            "min_lon": pos["min_lon"],
            "min_lat": pos["min_lat"],
            "max_lon": pos["max_lon"],
            "max_lat": pos["max_lat"],
            "geometry_json": _geometry_json(gdf),
            "properties_json": _properties(gdf),
            "source_file": source_file,
        }
    )
    out = out[out["place_geoid"].notna() & (out["place_geoid"].str.len() == 7)]
    out = out.drop_duplicates(subset=["place_geoid"], keep="last").reset_index(drop=True)
    return out


# ── Storage ──────────────────────────────────────────────────────────────────


def _replace_table(table: str, frame: pd.DataFrame) -> None:
    """Atomically replace every row, using explicit column names (never SELECT *)."""
    columns = list(frame.columns)
    conn = store.get_conn()
    conn.register("__tiger_boundary_upload", frame)
    try:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(f"DELETE FROM {table}")
            col_sql = ", ".join(columns)
            conn.execute(f"INSERT INTO {table} ({col_sql}) SELECT {col_sql} FROM __tiger_boundary_upload")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        try:
            conn.unregister("__tiger_boundary_upload")
        except Exception:
            pass


def _build_route_boundary_bridge(boundary_table: str, key_col: str, bridge_table: str) -> int:
    """Rebuild the route-to-boundary overlap bridge from loaded geometry.

    Mirrors ``load_bike_route_tracts``: one row per positive-length overlap,
    measured in the routes' local UTM CRS.  The bridge is deliberately separate
    from ``bike_routes`` so a route that crosses a ZCTA or city edge keeps both
    memberships instead of being assigned to an arbitrary polygon.
    """
    import geopandas as gpd
    import shapely

    conn = store.get_conn()
    routes = store.query("SELECT route_id, geometry_json FROM bike_routes")
    if routes.empty:
        conn.execute(f"DELETE FROM {bridge_table}")
        print(f"  - No bike routes; {bridge_table} cleared")
        return 0

    routes = routes[routes["geometry_json"].notna() & (routes["geometry_json"].astype(str) != "")]
    if routes.empty:
        conn.execute(f"DELETE FROM {bridge_table}")
        print(f"  ⚠ Bike routes have no usable geometry; {bridge_table} cleared")
        return 0
    routes_gdf = gpd.GeoDataFrame(
        {"route_id": routes["route_id"].astype(str).to_numpy()},
        geometry=shapely.from_geojson(routes["geometry_json"].astype(str).to_numpy()),
        crs="EPSG:4326",
    )
    routes_gdf = routes_gdf[routes_gdf.geometry.notna() & ~routes_gdf.geometry.is_empty].copy()
    if routes_gdf.empty:
        conn.execute(f"DELETE FROM {bridge_table}")
        print(f"  ⚠ Bike routes have no usable geometry; {bridge_table} cleared")
        return 0

    boundaries = store.query(
        f"SELECT {key_col}, geometry_json FROM {boundary_table} "
        f"WHERE {key_col} IS NOT NULL AND geometry_json IS NOT NULL"
    )
    if boundaries.empty:
        conn.execute(f"DELETE FROM {bridge_table}")
        print(f"  ⚠ No {boundary_table} geometry; {bridge_table} cleared")
        return 0
    boundaries_gdf = gpd.GeoDataFrame(
        {key_col: boundaries[key_col].astype(str).to_numpy()},
        geometry=shapely.from_geojson(boundaries["geometry_json"].astype(str).to_numpy()),
        crs="EPSG:4326",
    )
    boundaries_gdf = boundaries_gdf[
        boundaries_gdf.geometry.notna() & ~boundaries_gdf.geometry.is_empty
    ].copy()
    boundaries_gdf = boundaries_gdf.drop_duplicates(subset=[key_col])
    if boundaries_gdf.empty:
        conn.execute(f"DELETE FROM {bridge_table}")
        print(f"  ⚠ No usable {boundary_table} geometry; {bridge_table} cleared")
        return 0

    # Restrict the boundary frame to the routes' footprint before the spatial
    # join; a national ZCTA layer should not be projected and joined in full.
    min_lon, min_lat, max_lon, max_lat = routes_gdf.total_bounds
    pad = 0.05
    boundaries_gdf = boundaries_gdf.cx[min_lon - pad : max_lon + pad, min_lat - pad : max_lat + pad]
    if boundaries_gdf.empty:
        conn.execute(f"DELETE FROM {bridge_table}")
        print(f"  ⚠ No {boundary_table} polygons intersect the bike routes; {bridge_table} cleared")
        return 0

    utm = routes_gdf.estimate_utm_crs() or "EPSG:32617"
    routes_m = routes_gdf.to_crs(utm)
    boundaries_m = boundaries_gdf.to_crs(utm)
    routes_m["geometry"] = routes_m.geometry.make_valid()
    boundaries_m["geometry"] = boundaries_m.geometry.make_valid()
    candidates = gpd.sjoin(
        routes_m[["route_id", "geometry"]],
        boundaries_m[[key_col, "geometry"]],
        how="inner",
        predicate="intersects",
    )
    route_geoms = routes_m.set_index("route_id").geometry
    boundary_geoms = boundaries_m.set_index(key_col).geometry
    overlaps = []
    for row in candidates[["route_id", key_col]].itertuples(index=False):
        route_geom = route_geoms.get(row.route_id)
        boundary_geom = boundary_geoms.get(getattr(row, key_col))
        if route_geom is None or boundary_geom is None:
            continue
        try:
            overlap = route_geom.intersection(boundary_geom).length
        except Exception:
            continue
        if overlap > 1.0:  # ignore point/boundary touches
            overlaps.append((row.route_id, getattr(row, key_col), float(overlap)))

    out = pd.DataFrame(overlaps, columns=["route_id", key_col, "overlap_m"])
    if not out.empty:
        out = out.groupby(["route_id", key_col], as_index=False)["overlap_m"].sum()
        out["overlap_m"] = out["overlap_m"].round(3)

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(f"DELETE FROM {bridge_table}")
        if not out.empty:
            conn.register("__tiger_bridge", out)
            conn.execute(
                f"INSERT INTO {bridge_table} (route_id, {key_col}, overlap_m) "
                f"SELECT route_id, {key_col}, overlap_m FROM __tiger_bridge"
            )
            conn.unregister("__tiger_bridge")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    print(f"  ✓ {len(out):,} route/boundary overlap(s) written to {bridge_table}")
    return len(out)


def _load(kind: str, table: str, normalizer) -> int:
    paths = tiger_boundary_paths(kind)
    if not paths:
        print(
            f"No Census {kind} boundary file found.\n"
            f"  Looked under {settings.tiger_dir} and matching *{kind}* folders/files "
            f"under {settings.data_dir}.\n"
            f"  Download the Census file and place the .zip or extracted .shp set there."
        )
        return 0

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            gdf = _read_boundary_frame(path)
        except Exception as exc:
            print(f"  ⚠ Skipping {path.name}: {exc}")
            continue
        if gdf is None:
            print(f"  ⚠ Skipping {path.name}: no usable features.")
            continue
        frame = normalizer(gdf, _source_label(path))
        if frame.empty:
            print(f"  ⚠ Skipping {path.name}: no rows with a valid TIGER key.")
            continue
        print(f"  ✓ {path.name}: {len(frame):,} row(s)")
        frames.append(frame)

    if not frames:
        print(f"No valid TIGER {kind} rows found; existing {table} left untouched.")
        return 0

    frame = pd.concat(frames, ignore_index=True)
    key = "zcta5" if kind == "zcta" else "place_geoid"
    frame = frame.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)
    _replace_table(table, frame)
    print(f"  ✓ {len(frame):,} TIGER {kind} boundary row(s) loaded into {table}")
    if kind == "zcta":
        try:
            _build_route_boundary_bridge("zcta_boundaries", "zcta5", "bike_route_zctas")
        except Exception as exc:
            print(f"  ⚠ ZCTA layer loaded, but route-to-ZCTA bridge was not rebuilt: {exc}")
        try:
            from services.loaders.transit import load_transit_route_zctas

            load_transit_route_zctas()
        except Exception as exc:
            print(f"  ⚠ ZCTA layer loaded, but transit route-to-ZCTA bridge was not rebuilt: {exc}")
    elif kind == "place":
        try:
            _build_route_boundary_bridge("place_boundaries", "place_geoid", "bike_route_places")
        except Exception as exc:
            print(f"  ⚠ Place layer loaded, but route-to-Place bridge was not rebuilt: {exc}")
        try:
            from services.loaders.transit import load_transit_route_places

            load_transit_route_places()
        except Exception as exc:
            print(f"  ⚠ Place layer loaded, but transit route-to-Place bridge was not rebuilt: {exc}")
    return len(frame)


def load_tiger_zcta5() -> int:
    """Load Census ZCTA5 polygons (national, all vintages found on disk)."""
    return _load("zcta", "zcta_boundaries", normalize_zcta_frame)


def load_tiger_places() -> int:
    """Load Census Place polygons (all states found on disk)."""
    return _load("place", "place_boundaries", normalize_place_frame)


def load_bike_route_zctas() -> int:
    """Rebuild the bike-route to ZCTA overlap bridge from installed geometry."""
    return _build_route_boundary_bridge("zcta_boundaries", "zcta5", "bike_route_zctas")


def load_bike_route_places() -> int:
    """Rebuild the bike-route to Census Place overlap bridge from installed geometry."""
    return _build_route_boundary_bridge("place_boundaries", "place_geoid", "bike_route_places")

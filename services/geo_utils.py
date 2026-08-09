"""
Geo utilities: assign census tract FIPS to each house via spatial join.

Priority order for geometry source:
  1. NRI geometry cache  — created automatically when you run load_nri() with
     the NRI shapefile. Lives at data/nri/nri_geometry_cache.parquet.
     Best choice: no extra download needed if you have the NRI shapefile.
  2. TIGER/Line shapefiles — place any state .shp files in data/shapefiles/.
     Download: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
  3. Census Geocoder API  — fallback, ~0.15 s per house, requires internet.
"""
import time
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
from functools import lru_cache
from typing import Optional

from config import settings


# ── Source 1: NRI geometry cache (parquet written by data_loader.load_nri) ──

@lru_cache(maxsize=1)
def _load_nri_geometry() -> Optional[gpd.GeoDataFrame]:
    cache = settings.nri_shp.parent / "nri_geometry_cache.parquet"
    if not cache.exists():
        return None
    try:
        gdf = gpd.read_parquet(cache)
        if "tract_fips" not in gdf.columns or "geometry" not in gdf.columns:
            return None
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif str(gdf.crs).upper() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        print(f"  Loaded NRI geometry cache: {len(gdf):,} tracts")
        return gdf[["tract_fips", "geometry"]]
    except Exception as e:
        print(f"  Could not load NRI geometry cache: {e}")
        return None


# ── Source 2: TIGER/Line shapefiles in data/shapefiles/ ──────────────────────

@lru_cache(maxsize=1)
def _load_tiger_shapefiles() -> Optional[gpd.GeoDataFrame]:
    shp_files = list(settings.shapefile_dir.glob("**/*.shp"))
    if not shp_files:
        return None

    gdfs = []
    for shp in shp_files:
        try:
            gdf = gpd.read_file(shp)
            gdfs.append(gdf)
        except Exception as e:
            print(f"  Could not read {shp.name}: {e}")
    if not gdfs:
        return None

    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry")

    # TIGER/Line tracts carry GEOID (2020) or GEOID10 (older vintages)
    for col in ("GEOID", "GEOID20", "GEOID10"):
        if col in combined.columns:
            combined["tract_fips"] = combined[col].astype(str).str.zfill(11)
            break
    else:
        print("  Warning: no GEOID column in TIGER/Line shapefiles")
        return None

    if str(combined.crs).upper() != "EPSG:4326":
        combined = combined.to_crs("EPSG:4326")
    print(f"  Loaded TIGER/Line shapefiles: {len(combined):,} tracts")
    return combined[["tract_fips", "geometry"]]


# ── Unified loader: NRI cache → TIGER/Line → None ────────────────────────────

@lru_cache(maxsize=1)
def _load_tracts_gdf() -> Optional[gpd.GeoDataFrame]:
    """Return the best available tract geometry GeoDataFrame."""
    gdf = _load_nri_geometry()
    if gdf is not None:
        return gdf
    gdf = _load_tiger_shapefiles()
    if gdf is not None:
        return gdf
    return None


# ── Spatial join ──────────────────────────────────────────────────────────────

def assign_tract_fips_spatial(df: pd.DataFrame) -> pd.DataFrame:
    tracts_gdf = _load_tracts_gdf()
    if tracts_gdf is None:
        print("  No geometry source found — falling back to Census Geocoder API")
        return assign_tract_fips_api(df)

    pts = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, tracts_gdf, how="left", predicate="within")
    joined = joined.drop(columns=['tract_fips_left']).rename(columns={'tract_fips_right':'tract_fips'})
    df = df.copy()
    df["tract_fips"] = joined["tract_fips"].values
    return df


# ── Census Geocoder API fallback ─────────────────────────────────────────────

_GEOCODER_URL = (
    "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    "?x={lon}&y={lat}&benchmark=Public_AR_Current&vintage=Current_Current&format=json"
)


def _get_tract_from_api(lat: float, lon: float) -> Optional[str]:
    try:
        resp = requests.get(_GEOCODER_URL.format(lat=lat, lon=lon), timeout=10)
        tracts = (resp.json().get("result", {})
                             .get("geographies", {})
                             .get("Census Tracts", []))
        if tracts:
            return tracts[0].get("GEOID")
    except Exception:
        pass
    return None


def assign_tract_fips_api(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    needs = df["tract_fips"].isna()
    print(f"  Census Geocoder API: resolving {needs.sum()} tracts (~{needs.sum() * 0.15:.0f}s)…")
    for idx in df[needs].index:
        lat, lon = df.at[idx, "lat"], df.at[idx, "lon"]
        if pd.isna(lat) or pd.isna(lon):
            continue
        fips = _get_tract_from_api(lat, lon)
        if fips:
            df.at[idx, "tract_fips"] = str(fips).zfill(11)
        time.sleep(0.15)
    return df


# ── Main entry ────────────────────────────────────────────────────────────────

def assign_tract_fips(df: pd.DataFrame) -> pd.DataFrame:
    """Assign census tract FIPS to every row. Uses best available geometry."""
    if _load_tracts_gdf() is not None:
        return assign_tract_fips_spatial(df)
    return assign_tract_fips_api(df)


def geometry_source() -> str:
    """Return a human-readable description of the active geometry source."""
    if _load_nri_geometry() is not None:
        return "NRI geometry cache (nri_geometry_cache.parquet)"
    if _load_tiger_shapefiles() is not None:
        return "TIGER/Line shapefiles"
    return "Census Geocoder API (fallback)"


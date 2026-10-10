"""Metro-area census measures that need tract geometry: population, land area and population density.

Registered as derived-measure provider ``msa_geometry`` (see ``services/derived_measures.py``).  The catalog concepts
``census_land_area``, ``census_tract_population_density`` and ``census_profile`` declare which of its measures they need;
``agents/answer_policy.py`` routes to it.  Nothing in the planner or the tool layer knows about land area.

How the numbers are produced
----------------------------
1. The metro area's member counties come from ``cbsa_counties`` (by CBSA code; by normalized title when the Census MSA
   row carries an unresolved ``X...`` code).
2. Tract populations come from ``census_tracts``; tract polygons from ``services.geo_utils`` (NRI geometry cache or
   TIGER/Line shapefiles).  Polygon areas are measured in an equal-area projection (EPSG:5070).
3. Every figure is computed over the *footprint*: the tracts that have BOTH a population and a polygon.  Population,
   land area and density therefore always reconcile (density = population / area), and any tract left out is reported.
   Too little coverage is an error, never a silently wrong number.

Performance and reliability
---------------------------
* geopandas is imported lazily, so requests that never need geometry never pay for it.
* Tract polygons are indexed by county once per loaded geometry object, and each tract's area is computed at most once
  (incrementally, only for the tracts a request needs).  The caches are keyed on the geometry object itself, so reloading
  the geometry can never serve stale areas.  Access is lock-protected (the API serves requests on a thread pool).
* At most ``MAX_METROS`` metro areas per call bounds the work; unbounded "rank every metro by area" is declined clearly.
"""
from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

import db.duckdb_store as store
from services.derived_measures import DerivedError, DerivedResult, MeasureInfo

SQ_M_PER_SQ_MI = 2_589_988.110336
KM2_PER_MI2 = 2.589988110336
EQUAL_AREA_CRS = "EPSG:5070"            # NAD83 / CONUS Albers: equal-area, so polygon areas are meaningful anywhere
MAX_METROS = 25
MIN_TRACT_COVERAGE = 0.95               # share of a metro's census tracts that must have a polygon
MIN_POPULATION_COVERAGE = 0.98          # ... and the share of its population those tracts hold
RECONCILE_TOLERANCE = 0.01              # report when tract-summed population differs from the MSA file by more than this

# Must equal the codes in agents/answer_status.py (tests/test_census_metrics.py enforces it); the services layer does not
# import from the agents layer.
DATA_UNAVAILABLE = "data_unavailable"
PLACE_MISSING = "place_missing"
LIMIT_EXCEEDED = "limit_exceeded"

GEOMETRY_HELP = ("No census-tract boundary geometry is available. Load the NRI shapefile (it caches "
                 "data/nri/nri_geometry_cache.parquet) or add TIGER/Line tract shapefiles to data/shapefiles/, "
                 "then re-run setup_data.py.")

_KM = re.compile(r"\b(?:sq\.?\s*km|square\s+kilomet(?:er|re)s?|sq\.?\s*kilomet(?:er|re)s?|km2|km\u00b2)", re.I)
_MSA_SUFFIX = re.compile(r"\s+(?:metro|micro)(?:politan)?\s+(?:statistical\s+)?area\s*$", re.I)


def wants_km(request: str) -> bool:
    """Square kilometers only when the request says so; square miles otherwise."""
    return bool(_KM.search(request or ""))


def _norm_title(text: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", _MSA_SUFFIX.sub("", s).lower()).strip()


# ---------------------------------------------------------------------------
# Tract geometry index (lazy, cached per geometry object)
# ---------------------------------------------------------------------------
_LOCK = threading.RLock()
_STATE: dict = {"source": None, "index": None}


class _GeometryIndex:
    """Tract polygons indexed by county, with per-tract areas filled in on demand."""

    def __init__(self, gdf):
        fips = gdf["tract_fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)
        keep = (~fips.duplicated()).to_numpy() & gdf.geometry.notna().to_numpy()
        self.gdf = gdf.loc[keep]
        self.fips = fips.to_numpy()[keep]
        self.pos_of = {f: i for i, f in enumerate(self.fips)}
        counties = pd.Series(self.fips).str[:5]
        self.by_county = {k: np.asarray(v, dtype=np.int64) for k, v in counties.groupby(counties).groups.items()}
        self.area: dict[str, float] = {}

    def area_sq_mi(self, tract_fips) -> float:
        """Total polygon area (square miles) of the given tracts; each tract is measured once, ever."""
        with _LOCK:
            missing = [f for f in tract_fips if f not in self.area]
            if missing:
                sub = self.gdf.iloc[[self.pos_of[f] for f in missing]]
                areas = sub.to_crs(EQUAL_AREA_CRS).geometry.area.to_numpy() / SQ_M_PER_SQ_MI
                self.area.update(zip(missing, (float(a) for a in areas)))
            return float(sum(self.area[f] for f in tract_fips))


def reset_caches() -> None:
    with _LOCK:
        _STATE.update(source=None, index=None)


def _geometry_index() -> _GeometryIndex:
    from services import geo_utils          # lazy: geopandas is heavy and most requests never need geometry

    gdf = geo_utils._load_tracts_gdf()
    if gdf is None or len(gdf) == 0 or "tract_fips" not in gdf.columns:
        raise DerivedError(DATA_UNAVAILABLE, GEOMETRY_HELP)
    with _LOCK:
        if _STATE["source"] is not gdf:
            _STATE["index"], _STATE["source"] = _GeometryIndex(gdf), gdf
        return _STATE["index"]


# ---------------------------------------------------------------------------
# Database reads
# ---------------------------------------------------------------------------
@dataclass
class _Metro:
    name: str
    official_population: float | None
    counties: list[str]
    via_title: bool = False


def _placeholders(n: int) -> str:
    return ",".join("?" for _ in range(n))


def _query(sql: str, params=None) -> pd.DataFrame:
    """Read through a private DuckDB cursor.

    The app shares ONE global connection across request threads, and a connection's result state is not thread-safe, so
    two overlapping ``store.query`` calls can clobber each other.  A cursor is an independent handle on the same
    database, which keeps this provider's reads safe under concurrency without touching anyone else's code path.
    """
    cursor = store.get_conn().cursor()
    try:
        return (cursor.execute(sql, params) if params else cursor.execute(sql)).df()
    finally:
        cursor.close()


def _counties_by_code(codes: list[str]) -> dict[str, list[str]]:
    if not codes:
        return {}
    df = _query(
        "SELECT cbsa_code, CAST(state_fips AS VARCHAR) || CAST(county_fips AS VARCHAR) AS county "
        f"FROM cbsa_counties WHERE cbsa_code IN ({_placeholders(len(codes))})", codes)
    out: dict[str, list[str]] = {}
    for code, county in zip(df["cbsa_code"].astype(str), df["county"].astype(str)):
        out.setdefault(code, []).append(county)
    return out


def _counties_by_title(name: str) -> list[str]:
    """Fallback for a census_msa row whose CBSA code is unresolved: match its title against cbsa_counties."""
    titles = _query("SELECT DISTINCT cbsa_code, cbsa_title FROM cbsa_counties WHERE cbsa_title IS NOT NULL")
    want = _norm_title(name)
    codes = {str(c) for c, t in zip(titles["cbsa_code"], titles["cbsa_title"]) if _norm_title(t) == want}
    return _counties_by_code(sorted(codes)).get(next(iter(codes)), []) if len(codes) == 1 else []


def _load_metros(names: list[str]) -> list[_Metro]:
    msa = _query(f"SELECT msa_code, name, population FROM census_msa WHERE name IN ({_placeholders(len(names))})", names)
    rows = {str(r["name"]): r for r in msa.to_dict("records")}
    real_codes = [str(r["msa_code"]) for r in rows.values() if r["msa_code"] and not str(r["msa_code"]).upper().startswith("X")]
    by_code = _counties_by_code(real_codes)
    metros = []
    for name in names:
        row = rows.get(name)
        if row is None:
            raise DerivedError(DATA_UNAVAILABLE, f"{name} is not in the census_msa table.")
        counties, via_title = by_code.get(str(row["msa_code"]), []), False
        if not counties:
            counties = _counties_by_title(name)
            via_title = bool(counties)
        if not counties:
            raise DerivedError(
                DATA_UNAVAILABLE,
                f"{name} has no member-county list in cbsa_counties (its CBSA code is unresolved), so its census tracts "
                "cannot be determined. Run scripts/diagnose_msa.py to see why it did not match a CBSA, or reload the "
                "CBSA delineation with setup_data.py.")
        population = row["population"]
        metros.append(_Metro(name, None if pd.isna(population) else float(population), sorted(set(counties)), via_title))
    return metros


def _tract_populations(counties: list[str]) -> pd.DataFrame:
    df = _query("SELECT tract_fips, population FROM census_tracts "
                     f"WHERE left(tract_fips, 5) IN ({_placeholders(len(counties))})", counties)
    df["tract_fips"] = df["tract_fips"].astype(str)
    df["county"] = df["tract_fips"].str[:5]
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------
class MetroGeometryProvider:
    id = "msa_geometry"
    entity_type = "MSA"
    entity_label = "metropolitan area"
    evidence_label = "Derived tract-area aggregation"      # shown where the SQL normally appears (historical wording kept)
    measures = {
        "population": MeasureInfo("population", "population", "people",
                                  "Total population of the census tracts in the metro area's member counties.",
                                  phrases=("population", "people", "residents", "inhabitants")),
        "land_area": MeasureInfo("land_area", "land area", "square miles / square kilometers",
                                 "Summed census-tract polygon area of the metro area's member counties.",
                                 phrases=("land area", "square miles", "square mile", "square kilometers", "square kilometres", "sq mi", "sq km")),
        "density": MeasureInfo("density", "population density", "people per square mile / kilometer",
                               "Total tract population divided by total tract polygon area.",
                               phrases=("density", "densely populated", "how dense")),
    }

    def compute(self, entities, measures, *, request: str = "") -> DerivedResult:
        names = list(dict.fromkeys(e["value"] for e in entities))
        if not names:
            raise DerivedError(PLACE_MISSING, "No metropolitan area was named.")
        if len(names) > MAX_METROS:
            raise DerivedError(
                LIMIT_EXCEEDED,
                f"At most {MAX_METROS} metropolitan areas can be computed in one request and this one names {len(names)}. "
                "Ask for fewer, or split the request. Ranking every metro area by land area or density is not supported; "
                "name the metros to compare.")
        wanted = [m for m in ("population", "land_area", "density") if m in set(measures)] or list(self.measures)
        km = wants_km(request)

        metros = _load_metros(names)
        pops = _tract_populations(sorted({c for m in metros for c in m.counties}))
        index = _geometry_index()

        rows, notes = [], []
        for metro in metros:
            rows.append(self._measure_one(metro, pops, index, km, wanted, notes))

        columns = {"msa_name": [r["name"] for r in rows]}
        if "population" in wanted:
            columns["population"] = [r["population"] for r in rows]
        if "land_area" in wanted:
            columns["land_area_sq_km" if km else "land_area_sq_mi"] = [round(r["area_km2" if km else "area_mi2"], 2) for r in rows]
        if "density" in wanted:
            columns["population_density_per_sq_km" if km else "population_density"] = [
                round(r["population"] / r["area_km2" if km else "area_mi2"], 2) for r in rows]

        unit = "square kilometers" if km else "square miles"
        method = []
        if "density" in wanted:
            method.append(f"Density is total tract population divided by total tract polygon area in {unit}; "
                          "it is not the average of tract densities.")
        if "land_area" in wanted or "density" in wanted:
            method.append("Land area is the sum of census-tract boundary polygon areas (equal-area projection) over every "
                          "tract in the metro area's member counties; it is computed, not read from a stored column. If "
                          "the tract polygons include water bodies it can exceed an official land-only figure.")
        method.append("Each name is matched to the metropolitan/micropolitan statistical area that contains it, so figures "
                      "describe the whole metro area (all member counties), not the city limits.")
        return DerivedResult(pd.DataFrame(columns), method, notes)

    @staticmethod
    def _measure_one(metro: _Metro, pops: pd.DataFrame, index: _GeometryIndex, km: bool, wanted: list[str],
                     notes: list[str]) -> dict:
        tracts = pops[pops["county"].isin(metro.counties) & pops["population"].notna()]
        if tracts.empty:
            raise DerivedError(DATA_UNAVAILABLE,
                               f"census_tracts has no population rows for the counties of {metro.name}. Load the census "
                               "tract population data (setup_data.py) and ask again.")
        parts = [index.by_county[c] for c in metro.counties if c in index.by_county]
        geometry_fips = index.fips[np.concatenate(parts)] if parts else np.array([], dtype=object)
        covered = np.isin(tracts["tract_fips"].to_numpy(), geometry_fips)
        total_tracts, covered_tracts = len(tracts), int(covered.sum())
        if covered_tracts == 0:
            raise DerivedError(DATA_UNAVAILABLE,
                               f"No tract boundary geometry matches the census tracts of {metro.name}. The geometry and the "
                               "census tracts are probably from different years; reload both from the same vintage.")
        total_pop = float(tracts["population"].sum())
        footprint = tracts[covered]
        footprint_pop = float(footprint["population"].sum())
        tract_cov = covered_tracts / total_tracts
        pop_cov = footprint_pop / total_pop if total_pop > 0 else 1.0
        if tract_cov < MIN_TRACT_COVERAGE or pop_cov < MIN_POPULATION_COVERAGE:
            raise DerivedError(
                DATA_UNAVAILABLE,
                f"Tract boundary geometry covers only {tract_cov:.0%} of the census tracts of {metro.name} ({pop_cov:.0%} "
                "of its population), which is too incomplete for a trustworthy area or density. The geometry and census "
                "tracts are probably from different years; reload both from the same vintage.")
        area_mi2 = index.area_sq_mi(footprint["tract_fips"].to_numpy())
        if area_mi2 <= 0:
            raise DerivedError(DATA_UNAVAILABLE, f"The tract polygons of {metro.name} have no measurable area.")

        if covered_tracts < total_tracts:
            notes.append(f"{metro.name}: {total_tracts - covered_tracts} of {total_tracts} census tracts had no boundary "
                         f"geometry and are left out of every figure above (they hold {1 - pop_cov:.1%} of its tract population).")
        if metro.via_title:
            notes.append(f"{metro.name}: its CBSA code is unresolved in census_msa, so its member counties were found by "
                         "matching the title.")
        official = metro.official_population
        if "population" in wanted and official and abs(footprint_pop - official) / official > RECONCILE_TOLERANCE:
            notes.append(f"{metro.name}: the Census MSA file lists {int(official):,} people, "
                         f"{(footprint_pop - official) / official:+.1%} versus the {int(round(footprint_pop)):,} summed over "
                         "its member counties' tracts; the difference comes from county-membership vintages (CBSA "
                         "delineation file versus Census MSA file).")
        return {"name": metro.name, "population": int(round(footprint_pop)), "area_mi2": area_mi2,
                "area_km2": area_mi2 * KM2_PER_MI2}


PROVIDER = MetroGeometryProvider()

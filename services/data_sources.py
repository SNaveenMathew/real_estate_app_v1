"""
Registry and refresh orchestration for the built-in datasets under data/ (see the
data/ layout in README.md). This is what backs the "Data sources" section of the
Data page: for each entry, the UI shows a link to where the latest export can be
downloaded, plus an upload control that saves the file into the right place under
data/ and re-runs the matching services/data_loader.py function — which already
writes upsert-safe (see db/duckdb_store.py: upsert_df / upsert_houses_with_snapshots).

Nothing here changes what `python setup_data.py` does — it still reads whatever is
on disk, exactly as before, and does not touch data_source_log. This module is the
extra "download link + upload + reload" path the Data page adds on top of that.

Placement modes
----------------
"single" — the source is one canonical file (or shapefile set): NRI, the two Census
           population tables, the CBSA crosswalk. Refreshing it replaces whatever
           matched `match_globs` before, so the loader's own "first/only match"
           logic (see data_loader.py) can't pick up a stale leftover.
"append" — the source accumulates multiple files over time: Redfin exports, sold-
           homes exports, one crime city, bike layers. Uploading a NEW filename
           adds to the folder (matches "drop another file" in the README). Uploading
           a file with the SAME name as one already there is treated as a refresh of
           that specific file — see _replace_append_file for what "refresh" means
           for each of those four.

Source links
------------
NRI / Census tract & MSA population / CBSA crosswalk / Sold homes (Allegheny) come
straight from README.md / config.py, which already documented them. The 8 crime
links and the bike link were not in this repo anywhere and were verified separately
(see each entry's `notes` for caveats — most importantly: Pittsburgh's original
source stopped updating in Nov 2023, and a few cities have since moved to new
records systems whose exports may not match the columns PittsburghCrimeParser /
etc. expect. If a refreshed file doesn't parse, `refresh_source` surfaces the
loader's own "missing required column(s)" message rather than hiding it.
"""
from __future__ import annotations

import contextlib
import io
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from config import settings
import db.duckdb_store as store
from services import data_loader
from services.crime_sources import CRIME_PARSERS, _find_col as _find_crime_col


# ── Small shared helpers ─────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """Strip any directory component and characters that don't belong in a filename."""
    name = Path(name or "upload").name
    name = re.sub(r"[^A-Za-z0-9._\-\s]", "_", name).strip()
    return name or "upload"


def _capture(fn: Callable[[], int]) -> tuple[int, str]:
    """Run fn() with stdout captured; return (result, captured_text). Every
    services.data_loader function reports progress/warnings via print(), so this
    is how refresh_source surfaces them to the UI without changing any loader."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn()
    return result, buf.getvalue()


def _warnings_from_log(log_text: str) -> list[str]:
    """Pull the actionable lines (skips/warnings/errors) out of a loader's printed
    output so the UI can show *why* something didn't load, not just a row count."""
    out = []
    for line in log_text.splitlines():
        s = line.strip()
        if s.startswith(("Skipping", "⚠", "Error", "✗", "No usable rows", "No valid rows",
                          "  ✗", "  ⚠")):
            out.append(s.lstrip("  "))
    return out


# ── Loader wrappers (zero-arg, so the registry can call them uniformly) ─────
# These deliberately mirror `python setup_data.py`'s own defaults (no
# --resolve-tracts, i.e. geo_utils=None) so a Data-page refresh behaves the same
# way a plain CLI run would, and chain the same follow-up step setup_data.py
# always chains after a sold-homes reload.

def _run_nri() -> int:
    return data_loader.load_nri()

def _run_census_tracts() -> int:
    return data_loader.load_census_tracts()

def _run_census_msa() -> int:
    return data_loader.load_census_msa()

def _run_cbsa() -> int:
    return data_loader.load_cbsa_crosswalk()

def _run_redfin() -> int:
    return data_loader.load_redfin(None)

def _run_sold() -> int:
    n = data_loader.load_sold_homes(run_geocoding=True, geo_utils=None)
    matched = store.match_sold_to_houses()
    print(f"  ✓ {matched} sold record(s) linked to houses as historical snapshots")
    return n

def _run_crime() -> int:
    return data_loader.load_crime()

def _run_bike() -> int:
    return data_loader.load_bike_routes()


@dataclass(frozen=True)
class DataSource:
    key: str
    label: str
    category: str                       # grouping shown in the UI
    table: str                          # primary table, for the schema-map inspector hookup
    dest_dir: Path                      # where uploaded files land
    placement: str                      # "single" | "append"
    accept: str                         # file input accept= hint
    source_url: str
    instructions: str
    loader: Callable[[], int]
    match_globs: tuple[str, ...] = ()   # "single" placement: existing files matching these are removed first
    notes: str = ""
    detect: Optional[Callable[[list[str]], bool]] = None   # header-only sniff, generic upload flow
    row_count_sql: str = ""             # SQL returning one row/one column: current row count for this source


# ── Detection helpers (used by both the registry below and the generic
#    "Add a dataset" upload path in api/onboarding.py) ──────────────────────

def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace(".", "")


def _has_all(columns: list[str], candidates_list: list[list[str]]) -> bool:
    """True if, for every group of candidate names in candidates_list, at least
    one candidate is present in columns (case/space/underscore-insensitive)."""
    normed = {_norm(c) for c in columns}
    for candidates in candidates_list:
        if not any(_norm(c) in normed for c in candidates):
            return False
    return True


def _detect_redfin(columns: list[str]) -> bool:
    return _has_all(columns, [
        ["Address", "Street Address"],
        ["Latitude", "Lat"],
        ["Longitude", "Lon"],
        ["Price", "List Price"],
    ]) and _has_all(columns, [["Sale Type", "MLS#", "Days on Market", "URL"]])


def _detect_sold_allegheny(columns: list[str]) -> bool:
    return data_loader.AlleghenyParser.matches(columns)


def _detect_nri(columns: list[str]) -> bool:
    return _has_all(columns, [["TRACTFIPS", "GEOID"], ["RISK_SCORE"]])


def _detect_census_tracts(columns: list[str]) -> bool:
    # Distinguished from the MSA table by content (GEO_IDs starting 1400000US),
    # which a header-only sniff can't see — this only checks the table shape.
    return _has_all(columns, [["GEO_ID"], ["NAME"]]) and any(
        _norm(c).startswith("p1") for c in columns
    )


def _detect_cbsa(columns: list[str]) -> bool:
    return _has_all(columns, [["CBSA Code"], ["CBSA Title"], ["FIPS State Code", "FIPS State"]])


def _make_crime_detector(parser) -> Callable[[list[str]], bool]:
    def _detect(columns: list[str]) -> bool:
        return all(_find_crime_col(columns, parser._WANTED[f]) for f in parser._REQUIRED)
    return _detect


# ── Registry ─────────────────────────────────────────────────────────────────

def _census_dir() -> Path:
    return settings.cbsa_xlsx.parent


_STATIC_SOURCES: list[DataSource] = [
    DataSource(
        key="nri", label="FEMA National Risk Index", category="Reference data",
        table="nri_tracts", dest_dir=settings.nri_shp.parent, placement="single",
        accept=".zip",
        source_url="https://www.fema.gov/about/openfema/data-sets/national-risk-index-data",
        instructions="Download the \u201cNRI Census Tract Shapefile\u201d zip and upload it as-is "
                      "(don't unzip it first) \u2014 it contains the .shp plus its .dbf/.shx/.prj siblings.",
        loader=_run_nri,
        match_globs=("*.shp", "*.shx", "*.dbf", "*.prj", "*.cpg", "*.sbn", "*.sbx",
                     "*.shp.xml", "*.dbf.xml", "nri_geometry_cache.parquet"),
        row_count_sql="SELECT COUNT(*) FROM nri_tracts",
        detect=_detect_nri,
    ),
    DataSource(
        key="census_tracts", label="Census tract population (Eg: DECENNIALPL2020.P1-Data.csv)",
        category="Reference data", table="census_tracts", dest_dir=_census_dir(), placement="single",
        accept=".csv",
        source_url="https://data.census.gov/table/DECENNIALPL2020.P1",
        instructions="Table DECENNIALPL2020.P1 \u2192 Geography: Census Tracts \u2192 All United States \u2192 Download.",
        loader=_run_census_tracts,
        match_globs=(),   # handled specially in refresh_source — see _replace_census_tracts
        row_count_sql="SELECT COUNT(*) FROM census_tracts",
        detect=_detect_census_tracts,
    ),
    DataSource(
        key="census_msa", label="Census MSA population (Eg: DECENNIALPL2020.P1-2026-03-25T232220.csv)",
        category="Reference data", table="census_msa", dest_dir=_census_dir(), placement="single",
        accept=".csv",
        source_url="https://data.census.gov/table/DECENNIALPL2020.P1",
        instructions="Same table as tract population, but with Geography: Metropolitan/Micropolitan "
                      "Statistical Areas \u2192 All.",
        loader=_run_census_msa,
        match_globs=(),   # handled specially in refresh_source
        row_count_sql="SELECT COUNT(*) FROM census_msa",
        detect=None,  # a header-only sniff can't reliably tell this apart from the tract file
    ),
    DataSource(
        key="cbsa", label="CBSA \u2192 county crosswalk", category="Reference data",
        table="cbsa_counties", dest_dir=_census_dir(), placement="single",
        accept=".xlsx,.xls,.csv",
        source_url="https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html",
        instructions="Download the current delineation file (list1_*.xlsx). Refresh this before Census "
                      "MSA population so MSA name\u2192code matching has something to match against.",
        loader=_run_cbsa,
        match_globs=("list*.xlsx", "list*.xls", "list*.csv"),
        row_count_sql="SELECT COUNT(*) FROM cbsa_counties",
        detect=_detect_cbsa,
    ),
    DataSource(
        key="redfin", label="Redfin favorites", category="Redfin",
        table="houses", dest_dir=settings.redfin_dir, placement="append",
        accept=".csv",
        source_url="https://www.redfin.com",
        instructions="From your Redfin saved search, use \u201cDownload All\u201d to export the current "
                      "favorites CSV. Re-uploading a file with the same name as one you've already loaded "
                      "refreshes it: any house that drops out of that export (sold, delisted, unfavorited) "
                      "is marked \u201cRemoved from favorites\u201d and gets a history entry, rather than being "
                      "left showing its last known status.",
        loader=_run_redfin,
        notes="Tract/MSA joins for newly added houses may need a separate tract-resolution pass "
              "(see README: `python setup_data.py --resolve-tracts`).",
        row_count_sql="SELECT COUNT(*) FROM houses",
        detect=_detect_redfin,
    ),
    DataSource(
        key="sold", label="Sold homes (Allegheny County, PA)", category="Sold homes",
        table="sold_homes", dest_dir=settings.sold_dir, placement="append",
        accept=".csv",
        source_url="https://data.wprdc.org/dataset/real-estate-sales",
        instructions="Run a sale search and export the results as CSV. Other counties fall back to a "
                      "generic parser with reduced field coverage \u2014 see README.",
        loader=_run_sold,
        row_count_sql="SELECT COUNT(*) FROM sold_homes",
        detect=_detect_sold_allegheny,
    ),
]


def _crime_source_url(city: str) -> tuple[str, str]:
    """(url, notes) verified separately for each city — see module docstring."""
    return {
        "baltimore": (
            "https://data.baltimorecity.gov/datasets/baltimore::part1-crime-data", ""),
        "boston": (
            "https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system", ""),
        "buffalo": (
            "https://data.buffalony.gov/Public-Safety/Crime-Incidents/d6g9-xbgu", ""),
        "chicago": (
            "https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2", ""),
        "indianapolis": (
            "https://data.indy.gov",
            "IMPD launched a new \u201cIMPD Transparency\u201d portal in Nov 2025; the old OpenIndy UCR "
            "export this parser was built against may no longer be the current source. If the upload "
            "reports missing columns, services/crime_sources.py's IndianapolisCrimeParser likely needs "
            "updating to the new export's column names."),
        "minneapolis": (
            "http://opendata.minneapolismn.gov/",
            "Minneapolis has migrated crime reporting to NIBRS since this parser was written; column "
            "names may have shifted. If the upload reports missing columns, "
            "MinneapolisCrimeParser likely needs a column-name update."),
        "philadelphia": (
            "https://opendataphilly.org/datasets/crime-incidents/", ""),
        "pittsburgh": (
            "https://data.wprdc.org/dataset/uniform-crime-reporting-data/resource/044f2016-1dfd-4ab0-bc1e-065da05fca2e",
            "The dataset PittsburghCrimeParser was built against (\u201cPolice Incident Blotter\u201d) "
            "stopped updating on 11/14/2023. This links to its replacement, \u201cMonthly Criminal "
            "Activity\u201d, which uses the newer NIBRS-based schema \u2014 PittsburghCrimeParser was not "
            "written against it and will very likely report missing required columns until it's updated "
            "in services/crime_sources.py."),
    }.get(city, ("", ""))


def _crime_sources() -> list[DataSource]:
    out = []
    for parser in CRIME_PARSERS:
        url, notes = _crime_source_url(parser.city)
        out.append(DataSource(
            key=f"crime_{parser.city}", label=f"Crime \u2014 {parser.city_label}",
            category="Crime", table="crime_incidents",
            dest_dir=settings.data_dir / "crime" / parser.city, placement="append",
            accept=".csv,.xlsx,.xls",
            source_url=url,
            instructions="Download the current export and upload it below. Re-uploading a file with the "
                         "same name replaces just that file's incidents; a differently-named file is "
                         "added alongside what's already loaded (e.g. one file per year).",
            loader=_run_crime,
            notes=notes,
            row_count_sql=f"SELECT COUNT(*) FROM crime_incidents WHERE city = '{parser.city}'",
            detect=_make_crime_detector(parser),
        ))
    return out


def _bike_source() -> DataSource:
    return DataSource(
        key="bike", label="Bike infrastructure (BikePGH)", category="Bike infrastructure",
        table="bike_routes", dest_dir=settings.data_dir / "bike", placement="append",
        accept=".zip",
        source_url="https://data.wprdc.org/dataset/shape-files-for-bikepgh-s-pittsburgh-bike-map",
        instructions="Download a layer's shapefile zip (e.g. \u201cBike Lanes\u201d) and upload it as-is. "
                     "It's extracted under data/bike/pittsburgh/<layer name>/ automatically; upload each "
                     "layer you want separately.",
        loader=_run_bike,
        notes="Currently only Pittsburgh has recognized layers (see services/data_loader.py's "
              "_BIKE_LAYER_SPECS) \u2014 uploads are placed under data/bike/pittsburgh/.",
        row_count_sql="SELECT COUNT(*) FROM bike_routes",
        detect=None,  # shapefile zip — not part of the generic CSV/XLSX upload auto-detect
    )


def _registry() -> dict[str, DataSource]:
    sources = list(_STATIC_SOURCES) + _crime_sources() + [_bike_source()]
    return {s.key: s for s in sources}


REGISTRY: dict[str, DataSource] = _registry()


def get_source(key: str) -> Optional[DataSource]:
    return REGISTRY.get(key)


def sources_for_table(table: str) -> list[str]:
    """Keys of every registered source that feeds `table` — used by the schema-map
    inspector to know which sources to offer when a table node is selected."""
    return [k for k, s in REGISTRY.items() if s.table == table]


def _current_row_count(source: DataSource) -> int:
    if not source.row_count_sql:
        return 0
    try:
        df = store.query(source.row_count_sql)
        return int(df.iloc[0, 0])
    except Exception:
        return 0


def list_sources() -> list[dict]:
    log = store.get_source_log()
    out = []
    for key, s in REGISTRY.items():
        entry = log.get(key, {})
        out.append({
            "key": s.key, "label": s.label, "category": s.category, "table": s.table,
            "placement": s.placement, "accept": s.accept, "source_url": s.source_url,
            "instructions": s.instructions, "notes": s.notes,
            "current_row_count": _current_row_count(s),
            "last_loaded_at": entry.get("last_loaded_at"),
            "last_row_count": entry.get("row_count"),
            "last_source_files": entry.get("source_files", []),
            "last_message": entry.get("message", ""),
        })
    return out


def detect_source_for_columns(columns: list[str]) -> Optional[str]:
    """Return the key of the built-in source a generically-uploaded file's columns
    match, or None. Used by the "Add a dataset" flow to route a refresh of a known
    source there instead of starting a new-dataset draft (see api/onboarding.py)."""
    for key, s in REGISTRY.items():
        if s.detect is not None:
            try:
                if s.detect(columns):
                    return key
            except Exception:
                continue
    return None


def peek_header_candidates(path: Path, ext: str) -> list[list[str]]:
    """Cheap header-only reads for the formats our detectable sources use
    (csv/xlsx/xls), tried at a couple of skiprows offsets since one source (the
    CBSA crosswalk) ships with 2 junk rows before its real header. Returns []
    rather than raising, so the caller can just skip detection on any read error
    or unsupported format."""
    if ext not in (".csv", ".xlsx", ".xls"):
        return []
    candidates = []
    for skip in (0, 2):
        try:
            if ext == ".csv":
                cols = list(pd.read_csv(path, nrows=0, skiprows=skip, encoding="utf-8-sig").columns)
            else:
                cols = list(pd.read_excel(path, nrows=0, skiprows=skip).columns)
            candidates.append(cols)
        except Exception:
            continue
    return candidates


# ── Placement: "single" sources ─────────────────────────────────────────────

def _clear_single_matches(source: DataSource) -> list[str]:
    """Remove whatever currently satisfies this source's canonical-file slot so the
    loader's own glob/first-match logic can't pick up a stale leftover. Returns the
    names removed."""
    removed = []
    for pattern in source.match_globs:
        for p in sorted(source.dest_dir.glob(pattern)):
            if p.is_file():
                removed.append(p.name)
                p.unlink()
    return removed


def _sniff_is_tract_csv(path: Path) -> bool:
    try:
        sniff = pd.read_csv(path, encoding="utf-8-sig", nrows=3, dtype=str)
        flat = " ".join(sniff.values.flatten().astype(str))
        return "1400000US" in flat
    except Exception:
        return False


def _clear_census_population_files(source_key: str) -> list[str]:
    """census_tracts and census_msa share a folder and overlapping glob patterns
    (DECENNIALPL2020*.csv); the loaders themselves tell them apart by sniffing
    content (see load_census_tracts), so cleanup mirrors that instead of a fixed
    glob to avoid ever deleting the *other* population table's file."""
    census_dir = _census_dir()
    removed = []
    for p in sorted(census_dir.glob("DECENNIALPL2020*.csv")):
        is_tract = _sniff_is_tract_csv(p)
        if (source_key == "census_tracts" and is_tract) or (source_key == "census_msa" and not is_tract):
            removed.append(p.name)
            p.unlink()
    return removed


def _place_single(source: DataSource, tmp_path: Path, orig_filename: str) -> tuple[Path, list[str]]:
    source.dest_dir.mkdir(parents=True, exist_ok=True)
    if source.key in ("census_tracts", "census_msa"):
        removed = _clear_census_population_files(source.key)
    else:
        removed = _clear_single_matches(source)

    if source.key == "nri":
        dest_files = _extract_nri_zip(tmp_path, source.dest_dir)
        return dest_files[0] if dest_files else source.dest_dir, removed

    # Plain single CSV/XLSX sources: save under a name the loader's own glob will
    # find. If the upload's own name already matches, keep it (nicer for the user
    # to recognize later); otherwise synthesize a compliant name.
    name = _safe_filename(orig_filename)
    if source.key == "census_tracts" and "DECENNIALPL2020" not in name:
        name = f"DECENNIALPL2020_P1_tract_{name}"
    elif source.key == "census_msa" and not name.startswith("DECENNIALPL2020_P1"):
        name = f"DECENNIALPL2020_P1_msa_{name}"
    elif source.key == "cbsa" and not name.lower().startswith("list"):
        name = f"list_{name}"
    dest = source.dest_dir / name
    shutil.copyfile(tmp_path, dest)
    return dest, removed


def _extract_nri_zip(tmp_path: Path, dest_dir: Path) -> list[Path]:
    """Extract an NRI shapefile zip flat into dest_dir, renaming the shapefile (and
    its same-stem sidecar files) to match settings.nri_shp's expected name — the
    FEMA zip's internal filename varies by release and load_nri() only ever looks
    at one fixed path."""
    expected_stem = settings.nri_shp.stem
    with zipfile.ZipFile(tmp_path) as zf:
        shp_members = [m for m in zf.namelist() if m.lower().endswith(".shp")]
        if not shp_members:
            raise ValueError("This zip doesn't contain a .shp file — expected the NRI census-tract shapefile zip.")
        shp_member = shp_members[0]
        stem_in_zip = shp_member.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        written = []
        for member in zf.namelist():
            base = member.rsplit("/", 1)[-1]
            if not base or member.endswith("/"):
                continue
            member_stem, _, ext = base.rpartition(".")
            if member_stem == stem_in_zip and ext:
                out_name = f"{expected_stem}.{ext}"
            else:
                out_name = base
            out_path = dest_dir / out_name
            with zf.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written.append(out_path)
    return [p for p in written if p.suffix.lower() == ".shp"] or written


# ── Placement: "append" sources ─────────────────────────────────────────────

def _delete_rows_for_source_file(table: str, source_file_value: str, city: Optional[str] = None) -> None:
    conn = store.get_conn()
    if city is not None:
        conn.execute(f"DELETE FROM {table} WHERE source_file = ? AND city = ?", [source_file_value, city])
    else:
        conn.execute(f"DELETE FROM {table} WHERE source_file = ?", [source_file_value])


def _place_append(source: DataSource, tmp_path: Path, orig_filename: str,
                   dest_subdir: Optional[Path] = None) -> tuple[Path, bool]:
    """Save into an accumulating folder. Returns (dest_path, was_overwrite)."""
    target_dir = dest_subdir if dest_subdir is not None else source.dest_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_filename(orig_filename)
    dest = target_dir / name
    was_overwrite = dest.exists()
    shutil.copyfile(tmp_path, dest)
    return dest, was_overwrite


def _extract_bike_zip(tmp_path: Path, dest_root: Path) -> list[Path]:
    """Extract a BikePGH layer zip under data/bike/, preserving whatever internal
    folder structure it has (WPRDC's zips are typically a flat set of
    <LayerName>.shp/.dbf/.shx/.prj at the top level). If the zip has no city
    folder in its paths, files land under data/bike/pittsburgh/<layer>/ — the
    only city with recognized layers today (see _BIKE_LAYER_SPECS)."""
    from services.data_loader import _BIKE_LAYER_SPECS, _norm_folder_name

    written = []
    overwritten = []
    with zipfile.ZipFile(tmp_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        # Try to find a layer name from the shapefile's own name (WPRDC zips are
        # named e.g. "Bike Lanes.shp"); fall back to the zip's own filename stem.
        shp = next((n for n in names if n.lower().endswith(".shp")), None)
        stem = Path(shp).stem if shp else Path(tmp_path).stem
        layer_dir_name = None
        for spec_key, spec in _BIKE_LAYER_SPECS.items():
            if spec_key == _norm_folder_name(stem):
                layer_dir_name = spec[1]  # canonical label, e.g. "Bike Lanes"
                break
        layer_dir_name = layer_dir_name or stem
        out_dir = dest_root / "pittsburgh" / layer_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for member in names:
            base = Path(member).name
            out_path = out_dir / base
            if out_path.exists():
                overwritten.append(str(out_path.relative_to(dest_root)))
            with zf.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written.append(out_path)
    for rel in overwritten:
        _delete_rows_for_source_file("bike_routes", rel)
    return written


# ── Redfin-specific: detect houses that dropped out of a refreshed export ──

def _redfin_removed_favorites(dest: Path, was_overwrite: bool) -> int:
    if not was_overwrite:
        return 0
    old_ids = {
        r[0] for r in store.get_conn().execute(
            "SELECT house_id FROM houses WHERE source_file = ?", [dest.name]
        ).fetchall()
    }
    if not old_ids:
        return 0
    new_ids = data_loader.compute_redfin_house_ids(dest)
    removed = old_ids - new_ids
    if not removed:
        return 0
    return store.mark_removed_favorites(removed, dest.name)


# ── Orchestration ────────────────────────────────────────────────────────────

def refresh_source(key: str, tmp_path: Path, orig_filename: str) -> dict:
    """Save an uploaded file into the right place for `key` and re-run its loader.
    Returns a JSON-safe result dict — see list_sources() for the shape "rows"
    fields take once persisted."""
    source = get_source(key)
    if source is None:
        return {"ok": False, "error": f"Unknown data source: {key}"}

    ext = Path(orig_filename).suffix.lower()
    allowed = {e.strip() for e in source.accept.split(",")}
    if ext not in allowed:
        return {"ok": False, "error": f"Expected one of {sorted(allowed)} for this source, got '{ext}'."}

    rows_before = _current_row_count(source)
    removed_favorites = 0
    saved_files: list[str] = []
    warnings: list[str] = []

    try:
        if source.placement == "single":
            dest, cleared = _place_single(source, tmp_path, orig_filename)
            saved_files = [dest.name] if dest.is_file() else [p.name for p in dest.parent.glob("*") if p.is_file()]
        elif source.key == "bike":
            dest_files = _extract_bike_zip(tmp_path, source.dest_dir)
            saved_files = [str(p.relative_to(source.dest_dir)) for p in dest_files]
        else:
            dest, was_overwrite = _place_append(source, tmp_path, orig_filename)
            saved_files = [dest.name]
            if source.key == "redfin":
                removed_favorites = _redfin_removed_favorites(dest, was_overwrite)
            elif source.key == "sold" and was_overwrite:
                _delete_rows_for_source_file("sold_homes", dest.name)
            elif source.key.startswith("crime_") and was_overwrite:
                city = source.key.split("crime_", 1)[1]
                _delete_rows_for_source_file("crime_incidents", dest.name, city=city)
    except (zipfile.BadZipFile, ValueError) as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"Could not save the file: {e}"}

    try:
        _, log_text = _capture(source.loader)
        warnings = _warnings_from_log(log_text)
    except Exception as e:
        return {"ok": False, "error": f"Saved the file, but reloading failed: {e}",
                "saved_files": saved_files}

    rows_after = _current_row_count(source)
    message_parts = [f"{rows_after:,} row(s) now loaded (was {rows_before:,})."]
    if removed_favorites:
        message_parts.append(f"{removed_favorites} house(s) no longer in this export were marked "
                              f"\u201cRemoved from favorites\u201d.")
    if warnings:
        message_parts.append(f"{len(warnings)} file(s)/warning(s) need attention \u2014 see below.")
    message = " ".join(message_parts)

    store.record_source_load(source.key, source.label, rows_after, saved_files, message)

    return {
        "ok": True, "source_key": source.key, "rows_before": rows_before, "rows_after": rows_after,
        "removed_favorites": removed_favorites, "saved_files": saved_files,
        "warnings": warnings, "message": message,
    }

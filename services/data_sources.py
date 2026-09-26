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
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from config import settings
import db.duckdb_store as store
from services import data_loader
from services.crime_sources import (CRIME_PARSERS, _find_col as _find_crime_col,
                                     read_table as crime_read_table)


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


def _precheck_upload(source: DataSource, path: Path) -> tuple[bool, str]:
    """
    Faithfully re-check, read-only, whether this exact staged file would
    actually produce rows — using the *same* logic the real loader applies per
    file, not a re-derived guess. This runs immediately after staging and
    before anything is discarded: a bad file is rejected here, before the real
    loader (and any DB change) ever runs, closing the gap a bare post-load
    row-count check can't: a loader that gracefully skips one bad file
    internally leaves that file's *old* rows untouched, which a bare ">0" row
    count can't tell apart from a genuine success.

    Returns (ok, reason) — reason is empty when ok is True. A source with no
    precheck defined here (single-file sources, bike) always returns (True, "")
    and relies on the loader's own return value / post-load row count instead,
    which is unambiguous for those (see refresh_source).
    """
    if source.key == "redfin":
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, nrows=5000)
        except Exception as e:
            return False, f"Could not read this file as CSV: {e}"
        df.columns = df.columns.str.strip().str.lower()
        col_rename = {c: data_loader._REDFIN_COL_MAP[c] for c in df.columns if c in data_loader._REDFIN_COL_MAP}
        df = df.rename(columns=col_rename)
        if "lat" not in df.columns or "lon" not in df.columns:
            return False, "This file doesn't have latitude/longitude columns — a Redfin export normally does."
        lat = pd.to_numeric(df["lat"], errors="coerce")
        lon = pd.to_numeric(df["lon"], errors="coerce")
        if lat.notna().sum() == 0 or lon.notna().sum() == 0:
            return False, "No rows had usable latitude/longitude values."
        return True, ""

    if source.key == "sold":
        try:
            df = pd.read_csv(path, low_memory=False, nrows=5000, dtype=str)
        except Exception as e:
            return False, f"Could not read this file as CSV: {e}"
        if df.empty:
            return False, "This file has no data rows."
        parser = data_loader._detect_parser(list(df.columns))
        try:
            parsed = parser.parse(df, path.name)
        except Exception as e:
            return False, f"Could not parse this file as sold-homes data: {e}"
        if parsed is None or parsed.empty:
            return False, "No usable sale records were found in this file."
        return True, ""

    if source.key.startswith("crime_"):
        city = source.key.split("crime_", 1)[1]
        parser = next((p for p in CRIME_PARSERS if p.city == city), None)
        if parser is None:
            return False, f"No parser is registered for '{city}'."
        try:
            df = crime_read_table(path, parser._WANTED, parser._REQUIRED)
        except Exception as e:
            return False, f"Could not read this file: {e}"
        if df is None:
            return False, ("This file is missing one or more required columns for "
                            f"{parser.city_label} — see the loader's own message for which ones.")
        if df.empty:
            return False, "This file was read successfully but had no data rows."
        return True, ""

    return True, ""


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
    direct_download: bool = False       # True: source_url IS the file. False: source_url is a page to
                                         # navigate from — the UI must say so, not imply a one-click download.


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
        source_url="https://hazards.fema.gov/nri/data-resources",
        instructions="This is FEMA's download page, not a direct file \u2014 under \u201cShapefile Format\u201d, "
                      "find \u201cCensus Tracts\u201d and download the nationwide zip. Upload it as-is (don't "
                      "unzip it first); it contains the .shp plus its .dbf/.shx/.prj siblings.",
        loader=_run_nri,
        match_globs=("*.shp", "*.shx", "*.dbf", "*.prj", "*.cpg", "*.sbn", "*.sbx",
                     "*.shp.xml", "*.dbf.xml", "nri_geometry_cache.parquet"),
        row_count_sql="SELECT COUNT(*) FROM nri_tracts",
        detect=_detect_nri,
        direct_download=False,
    ),
    DataSource(
        key="census_tracts", label="Census tract population (Eg: DECENNIALPL2020.P1-Data.csv)",
        category="Reference data", table="census_tracts", dest_dir=_census_dir(), placement="single",
        accept=".csv",
        source_url="https://data.census.gov/table/DECENNIALPL2020.P1",
        instructions="This is a table-browser page, not a direct file \u2014 set Geography to Census Tracts "
                      "\u2192 All United States, then use the page's own Download button.",
        loader=_run_census_tracts,
        match_globs=(),   # handled specially in refresh_source — see _replace_census_tracts
        row_count_sql="SELECT COUNT(*) FROM census_tracts",
        detect=_detect_census_tracts,
        direct_download=False,
    ),
    DataSource(
        key="census_msa", label="Census MSA population (Eg: DECENNIALPL2020.P1-2026-03-25T232220.csv)",
        category="Reference data", table="census_msa", dest_dir=_census_dir(), placement="single",
        accept=".csv",
        source_url="https://data.census.gov/table/DECENNIALPL2020.P1",
        instructions="Same table-browser page as tract population, not a direct file \u2014 set Geography to "
                      "Metropolitan/Micropolitan Statistical Areas \u2192 All, then Download.",
        loader=_run_census_msa,
        match_globs=(),   # handled specially in refresh_source
        row_count_sql="SELECT COUNT(*) FROM census_msa",
        detect=None,  # a header-only sniff can't reliably tell this apart from the tract file
        direct_download=False,
    ),
    DataSource(
        key="cbsa", label="CBSA \u2192 county crosswalk", category="Reference data",
        table="cbsa_counties", dest_dir=_census_dir(), placement="single",
        accept=".xlsx,.xls,.csv",
        source_url="https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html",
        instructions="This is an index page, not a direct file \u2014 open the current year's entry and "
                      "download the delineation file (named like list1_*.xlsx). Refresh this before Census "
                      "MSA population so MSA name\u2192code matching has something to match against.",
        loader=_run_cbsa,
        match_globs=("list*.xlsx", "list*.xls", "list*.csv"),
        row_count_sql="SELECT COUNT(*) FROM cbsa_counties",
        detect=_detect_cbsa,
        direct_download=False,
    ),
    DataSource(
        key="redfin", label="Redfin favorites", category="Redfin",
        table="houses", dest_dir=settings.redfin_dir, placement="append",
        accept=".csv",
        source_url="https://www.redfin.com",
        instructions="This is Redfin's own site, not a direct file \u2014 it can't be, since the export is "
                      "specific to your saved search and requires your login. From your saved search, use "
                      "\u201cDownload All\u201d to get the current favorites CSV. Re-uploading a file with the "
                      "same name as one you've already loaded refreshes it: any house that drops out of that "
                      "export (sold, delisted, unfavorited) is marked \u201cRemoved from favorites\u201d and "
                      "gets a history entry, rather than being left showing its last known status.",
        loader=_run_redfin,
        notes="Tract/MSA joins for newly added houses may need a separate tract-resolution pass "
              "(see README: `python setup_data.py --resolve-tracts`).",
        row_count_sql="SELECT COUNT(*) FROM houses",
        detect=_detect_redfin,
        direct_download=False,
    ),
    DataSource(
        key="sold", label="Sold homes (Allegheny County, PA)", category="Sold homes",
        table="sold_homes", dest_dir=settings.sold_dir, placement="append",
        accept=".csv",
        source_url="https://data.wprdc.org/dataset/real-estate-sales",
        instructions="This is a WPRDC dataset page, not a direct file \u2014 open it and download the current "
                      "CSV resource (\u201cAllegheny County Property Sale Transactions\u201d, all sales since "
                      "2013). Other counties fall back to a generic parser with reduced field coverage "
                      "\u2014 see README.",
        loader=_run_sold,
        row_count_sql="SELECT COUNT(*) FROM sold_homes",
        detect=_detect_sold_allegheny,
        direct_download=False,
    ),
]


def _crime_source_url(city: str) -> tuple[str, str, bool]:
    """(url, notes, direct_download) verified separately for each city — see module
    docstring. direct_download=True only for the two cities confirmed to sit on a
    stable, documented direct-file API (Socrata's /resource/{id}.csv, which always
    reflects that city's own current dataset) rather than a portal page a person has
    to click through. The rest are CKAN/ArcGIS/Carto portals whose "current file" is
    a moving target (new resources get added, old ones split by year, etc.) — Copilot's
    review called pointing at those portal pages "reasonable"; what mattered was
    making sure the UI says a click-through is needed rather than implying a direct
    download every time (see the `direct_download` field's use in list_sources()).
    """
    return {
        "baltimore": (
            "https://data.baltimorecity.gov/datasets/baltimore::part1-crime-data", "", False),
        "boston": (
            "https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system",
            "", False),
        "buffalo": (
            "https://data.buffalony.gov/resource/d6g9-xbgu.csv",
            "This is Buffalo's Socrata API endpoint, not a portal page \u2014 it downloads the current "
            "\u201cCrime Incidents\u201d dataset directly as CSV.", True),
        "chicago": (
            "https://data.cityofchicago.org/resource/ijzp-q8t2.csv",
            "This is Chicago's Socrata API endpoint, not a portal page \u2014 it downloads the current "
            "\u201cCrimes \u2014 2001 to present\u201d dataset directly as CSV. Socrata caps a single request "
            "at 1,000 rows by default; for the full history use the dataset's own export page instead: "
            "https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2", True),
        "indianapolis": (
            "https://data.indy.gov",
            "This is a portal page, not a direct file. IMPD launched a new \u201cIMPD Transparency\u201d "
            "portal in Nov 2025; the old OpenIndy UCR export this parser was built against may no longer "
            "be the current source. If the upload reports missing columns, "
            "services/crime_sources.py's IndianapolisCrimeParser likely needs updating to the new export's "
            "column names.", False),
        "minneapolis": (
            "http://opendata.minneapolismn.gov/",
            "This is a portal page, not a direct file. Minneapolis has migrated crime reporting to NIBRS "
            "since this parser was written; column names may have shifted. If the upload reports missing "
            "columns, MinneapolisCrimeParser likely needs a column-name update.", False),
        "philadelphia": (
            "https://opendataphilly.org/datasets/crime-incidents/",
            "This is a portal page, not a direct file \u2014 it's also a large dataset that OpenDataPhilly "
            "itself splits by year; download each year you want, or use the API link on that page.", False),
        "pittsburgh": (
            "https://data.wprdc.org/dataset/uniform-crime-reporting-data/resource/044f2016-1dfd-4ab0-bc1e-065da05fca2e",
            "This is a portal page, not a direct file. It's the specific resource PittsburghCrimeParser's "
            "columns match, from the dataset also listed as \u201cPolice Incident Blotter (Archived)\u201d, "
            "which stopped updating on 11/14/2023 \u2014 this link won't have anything newer than that date. "
            "Its successor, \u201cMonthly Criminal Activity\u201d "
            "(data.wprdc.org/dataset/monthly-criminal-activity-dashboard), covers more recent incidents but "
            "uses a different NIBRS-based schema that PittsburghCrimeParser doesn't understand yet.", False),
    }.get(city, ("", "", False))


def _crime_sources() -> list[DataSource]:
    out = []
    for parser in CRIME_PARSERS:
        url, notes, direct = _crime_source_url(parser.city)
        instructions = ("Downloads the current data directly as CSV." if direct else
                         "This is a portal page, not a direct file \u2014 open it and download the current "
                         "export.") + (" Re-uploading a file with the same name replaces just that file's "
                         "incidents; a differently-named file is added alongside what's already loaded "
                         "(e.g. one file per year).")
        out.append(DataSource(
            key=f"crime_{parser.city}", label=f"Crime \u2014 {parser.city_label}",
            category="Crime", table="crime_incidents",
            dest_dir=settings.data_dir / "crime" / parser.city, placement="append",
            accept=".csv,.xlsx,.xls",
            source_url=url,
            instructions=instructions,
            loader=_run_crime,
            notes=notes,
            row_count_sql=f"SELECT COUNT(*) FROM crime_incidents WHERE city = '{parser.city}'",
            detect=_make_crime_detector(parser),
            direct_download=direct,
        ))
    return out


def _bike_source() -> DataSource:
    return DataSource(
        key="bike", label="Bike infrastructure (BikePGH)", category="Bike infrastructure",
        table="bike_routes", dest_dir=settings.data_dir / "bike", placement="append",
        accept=".zip",
        source_url="https://data.wprdc.org/dataset/shape-files-for-bikepgh-s-pittsburgh-bike-map",
        instructions="This is a WPRDC dataset page, not a direct file \u2014 open it, download a layer's "
                     "shapefile zip (e.g. \u201cBike Lanes\u201d), and upload it as-is. It's extracted under "
                     "data/bike/pittsburgh/<layer name>/ automatically; upload each layer you want "
                     "separately.",
        loader=_run_bike,
        notes="Currently only Pittsburgh has recognized layers (see services/data_loader.py's "
              "_BIKE_LAYER_SPECS) \u2014 uploads are placed under data/bike/pittsburgh/.",
        row_count_sql="SELECT COUNT(*) FROM bike_routes",
        detect=None,  # shapefile zip — not part of the generic CSV/XLSX upload auto-detect
        direct_download=False,
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
            "instructions": s.instructions, "notes": s.notes, "direct_download": s.direct_download,
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


# ── Backup / rollback helpers ────────────────────────────────────────────────
# The rule for everything below: never delete or overwrite anything that already
# exists until the replacement has been *confirmed* to load successfully. Every
# placement function backs up (renames aside) whatever it would otherwise
# destroy; refresh_source() only discards those backups after the reload comes
# back with actual rows, and restores them — then reloads once more — if it
# doesn't. This is what stands between a malformed upload and silently wiping
# real data (see the Copilot review this responds to).

def _backup_aside(paths: list[Path]) -> dict[Path, Path]:
    """Move each existing path to a sibling backup name instead of deleting or
    overwriting it. Returns {original_path: backup_path}."""
    token = f".refresh_backup_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    moved = {}
    for p in paths:
        if p.exists():
            backup = p.parent / f"{p.name}{token}"
            p.rename(backup)
            moved[p] = backup
    return moved


def _restore_backups(backups: dict[Path, Path], new_files: list[Path]) -> None:
    """Undo a failed refresh: remove whatever we newly wrote, then move every
    backed-up original back to its real name. Safe to call with an empty/partial
    `backups` or `new_files` (e.g. mid-failure during placement itself)."""
    for p in new_files:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
    for original, backup in backups.items():
        try:
            if backup.exists():
                backup.rename(original)
        except OSError:
            pass


def _discard_backups(backups: dict[Path, Path]) -> None:
    """Confirm a successful refresh: permanently remove the backed-up originals."""
    for backup in backups.values():
        try:
            if backup.exists():
                backup.unlink()
        except OSError:
            pass


def _rows_by_source_file(table: str, id_col: str, source_file_value: str,
                          city: Optional[str] = None) -> set:
    """IDs currently in `table` attributed to this exact filename (and city, for
    crime_incidents)."""
    conn = store.get_conn()
    if city is not None:
        rows = conn.execute(f"SELECT {id_col} FROM {table} WHERE source_file = ? AND city = ?",
                             [source_file_value, city]).fetchall()
    else:
        rows = conn.execute(f"SELECT {id_col} FROM {table} WHERE source_file = ?",
                             [source_file_value]).fetchall()
    return {r[0] for r in rows}


def _snapshot_rows_by_source_file(table: str, source_file_value: str,
                                   city: Optional[str] = None) -> pd.DataFrame:
    """Full row data (every column, in table order) for this exact filename —
    enough to restore them verbatim if a refresh needs to be undone, or to
    re-insert a subset with one field changed (see the redfin "removed from
    favorites" handling in refresh_source)."""
    conn = store.get_conn()
    if city is not None:
        return conn.execute(f"SELECT * FROM {table} WHERE source_file = ? AND city = ?",
                             [source_file_value, city]).df()
    return conn.execute(f"SELECT * FROM {table} WHERE source_file = ?", [source_file_value]).df()


def _delete_ids(table: str, id_col: str, ids: set) -> None:
    if not ids:
        return
    placeholders = ",".join(["?"] * len(ids))
    store.get_conn().execute(f"DELETE FROM {table} WHERE {id_col} IN ({placeholders})", list(ids))


# ── Placement: "single" sources ─────────────────────────────────────────────

def _sniff_is_tract_csv(path: Path) -> bool:
    try:
        sniff = pd.read_csv(path, encoding="utf-8-sig", nrows=3, dtype=str)
        flat = " ".join(sniff.values.flatten().astype(str))
        return "1400000US" in flat
    except Exception:
        return False


def _census_population_matches(source_key: str) -> list[Path]:
    """census_tracts and census_msa share a folder and overlapping glob patterns
    (DECENNIALPL2020*.csv); the loaders themselves tell them apart by sniffing
    content (see load_census_tracts), so this mirrors that instead of a fixed
    glob — otherwise refreshing one could end up displacing the *other* table's
    file."""
    out = []
    for p in sorted(_census_dir().glob("DECENNIALPL2020*.csv")):
        is_tract = _sniff_is_tract_csv(p)
        if (source_key == "census_tracts" and is_tract) or (source_key == "census_msa" and not is_tract):
            out.append(p)
    return out


def _single_source_matches(source: DataSource) -> list[Path]:
    """Files that currently satisfy this source's canonical-file slot — found the
    same way the loader itself would find them, so a refresh can never disagree
    with what load_*() actually reads."""
    if source.key in ("census_tracts", "census_msa"):
        return _census_population_matches(source.key)
    matches = []
    for pattern in source.match_globs:
        matches.extend(p for p in sorted(source.dest_dir.glob(pattern)) if p.is_file())
    return matches


def _place_single(source: DataSource, tmp_path: Path, orig_filename: str
                   ) -> tuple[Path, dict[Path, Path], list[Path]]:
    """Stage the new file next to (not over) whatever currently satisfies this
    source's canonical-file slot. Returns (dest, backups, new_files) — new_files
    is every file this call wrote, so a caller can clean all of them up on
    failure (matters for NRI, which writes a whole shapefile's worth)."""
    source.dest_dir.mkdir(parents=True, exist_ok=True)
    backups = _backup_aside(_single_source_matches(source))
    try:
        if source.key == "nri":
            shp_path, written = _extract_nri_zip(tmp_path, source.dest_dir)
            return shp_path, backups, written

        name = _safe_filename(orig_filename)
        if source.key == "census_tracts" and "DECENNIALPL2020" not in name:
            name = f"DECENNIALPL2020_P1_tract_{name}"
        elif source.key == "census_msa" and not name.startswith("DECENNIALPL2020_P1"):
            name = f"DECENNIALPL2020_P1_msa_{name}"
        elif source.key == "cbsa" and not name.lower().startswith("list"):
            name = f"list_{name}"
        dest = source.dest_dir / name
        shutil.copyfile(tmp_path, dest)
        return dest, backups, [dest]
    except Exception:
        _restore_backups(backups, [])
        raise


def _extract_nri_zip(tmp_path: Path, dest_dir: Path) -> tuple[Path, list[Path]]:
    """Extract an NRI shapefile zip flat into dest_dir, renaming the shapefile (and
    its same-stem sidecar files) to match settings.nri_shp's expected name — the
    FEMA zip's internal filename varies by release and load_nri() only ever looks
    at one fixed path. Returns (the renamed .shp path, every file written)."""
    expected_stem = settings.nri_shp.stem
    with zipfile.ZipFile(tmp_path) as zf:
        shp_members = [m for m in zf.namelist() if m.lower().endswith(".shp")]
        if not shp_members:
            raise ValueError("This zip doesn't contain a .shp file — expected the NRI census-tract shapefile zip.")
        shp_member = shp_members[0]
        stem_in_zip = shp_member.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        written = []
        shp_out = None
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
            if out_path.suffix.lower() == ".shp":
                shp_out = out_path
    return (shp_out or written[0]), written


# ── Placement: "append" sources ─────────────────────────────────────────────

def _place_append(source: DataSource, tmp_path: Path, orig_filename: str,
                   dest_subdir: Optional[Path] = None) -> tuple[Path, dict[Path, Path]]:
    """Save into an accumulating folder. If a file with this name already exists,
    it's backed up rather than overwritten in place, so a bad upload can't
    destroy it — refresh_source discards the backup on success or restores it on
    failure. An empty `backups` means this was a new filename, not a refresh of
    an existing one."""
    target_dir = dest_subdir if dest_subdir is not None else source.dest_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_filename(orig_filename)
    dest = target_dir / name
    backups = _backup_aside([dest]) if dest.exists() else {}
    try:
        shutil.copyfile(tmp_path, dest)
        return dest, backups
    except Exception:
        _restore_backups(backups, [])
        raise


def _extract_bike_zip(tmp_path: Path, dest_root: Path) -> tuple[list[Path], dict[Path, Path]]:
    """Extract a BikePGH layer zip under data/bike/, preserving whatever internal
    folder structure it has (WPRDC's zips are typically a flat set of
    <LayerName>.shp/.dbf/.shx/.prj at the top level). If the zip has no city
    folder in its paths, files land under data/bike/pittsburgh/<layer>/ — the
    only city with recognized layers today (see _BIKE_LAYER_SPECS). Any file
    this would overwrite is backed up rather than replaced in place, same as
    _place_append."""
    from services.data_loader import _BIKE_LAYER_SPECS, _norm_folder_name

    written: list[Path] = []
    backups: dict[Path, Path] = {}
    try:
        with zipfile.ZipFile(tmp_path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not any(n.lower().endswith(".shp") for n in names):
                raise ValueError("This zip doesn't contain a .shp file — expected a BikePGH layer shapefile zip.")
            shp = next(n for n in names if n.lower().endswith(".shp"))
            stem = Path(shp).stem
            layer_dir_name = None
            for spec_key, spec in _BIKE_LAYER_SPECS.items():
                if spec_key == _norm_folder_name(stem):
                    layer_dir_name = spec[1]  # canonical label, e.g. "Bike Lanes"
                    break
            layer_dir_name = layer_dir_name or stem
            out_dir = dest_root / "pittsburgh" / layer_dir_name
            out_dir.mkdir(parents=True, exist_ok=True)

            planned = [(member, out_dir / Path(member).name) for member in names]
            backups = _backup_aside([p for _, p in planned if p.exists()])
            for member, out_path in planned:
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                written.append(out_path)
        return written, backups
    except Exception:
        _restore_backups(backups, written)
        raise


# ── Orchestration ────────────────────────────────────────────────────────────

def refresh_source(key: str, tmp_path: Path, orig_filename: str) -> dict:
    """
    Save an uploaded file into the right place for `key` and re-run its loader.

    Nothing that already exists — file or database row — is removed until the
    new upload has been confirmed to actually be usable. For redfin/sold/crime
    that confirmation happens *before* the real loader ever runs, via
    _precheck_upload, which faithfully replays the same per-file logic the
    loader itself uses; for single-file sources (NRI/Census/CBSA) the loader's
    own return value is authoritative (each processes exactly one canonical
    file, so its count can't be confused with stale data left over from
    before). If a check fails, whatever was displaced is restored — reloading
    once more if it was — so a malformed upload can never leave a source worse
    off than before it was tried.

    Returns a JSON-safe result dict — see list_sources() for the shape "rows"
    fields take once persisted.
    """
    source = get_source(key)
    if source is None:
        return {"ok": False, "error": f"Unknown data source: {key}"}

    ext = Path(orig_filename).suffix.lower()
    allowed = {e.strip() for e in source.accept.split(",")}
    if ext not in allowed:
        return {"ok": False, "error": f"Expected one of {sorted(allowed)} for this source, got '{ext}'."}

    rows_before = _current_row_count(source)

    # ── Stage: place the new file, backing up (never deleting) anything it displaces ──
    try:
        if source.placement == "single":
            dest, backups, new_files = _place_single(source, tmp_path, orig_filename)
            saved_files = [p.name for p in new_files]
            dest_name = None
        elif source.key == "bike":
            new_files, backups = _extract_bike_zip(tmp_path, source.dest_dir)
            saved_files = [str(p.relative_to(source.dest_dir)) for p in new_files]
            dest, dest_name = None, None
        else:
            dest, backups = _place_append(source, tmp_path, orig_filename)
            new_files, saved_files = [dest], [dest.name]
            dest_name = dest.name
    except (zipfile.BadZipFile, ValueError) as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"Could not save the file: {e}"}

    def _rollback(reason: str, warnings: list[str] | None = None) -> dict:
        # Drop what we just wrote, restore whatever it displaced, and — if it
        # displaced something — reload once more so the database reflects the
        # restored (known-good) file rather than whatever the failed attempt
        # left behind (matters most for cbsa_counties, which has no primary
        # key and is fully replaced on every load).
        _restore_backups(backups, new_files)
        if backups:
            try:
                _capture(source.loader)
            except Exception:
                pass
        return {"ok": False, "error": f"This upload wasn't applied \u2014 {reason}",
                "warnings": warnings or []}

    # ── Pre-check (redfin/sold/crime only): faithfully replay the loader's own
    # per-file gate on the staged file *before* running anything or touching
    # the database, so an obviously-bad upload is rejected without the database
    # being touched at all.
    if dest_name:
        ok, reason = _precheck_upload(source, dest)
        if not ok:
            return _rollback(reason)

    # For append sources that track a specific file (redfin/sold/crime): snapshot
    # the full row data this exact file currently owns, then clear it, *before*
    # running the real loader. This is what makes "did the new content replace
    # everything the old content did" answerable at all — a row the loader
    # doesn't touch keeps its old source_file tag forever, so simply comparing
    # "what's tagged with this filename before vs. after" can't tell a row that
    # was never reprocessed from one that legitimately survived (this is exactly
    # the gap that let a shrunk file leave stale rows behind in an earlier
    # version of this function). Clearing first means anything present
    # afterward is unambiguously fresh. The precheck above already confirmed
    # the file is usable, so this is safe to do unconditionally for those three.
    old_ids: set = set()
    old_snapshot: pd.DataFrame = pd.DataFrame()
    scope_table = scope_col = scope_city = None
    if source.key == "redfin":
        scope_table, scope_col = "houses", "house_id"
    elif source.key == "sold":
        scope_table, scope_col = "sold_homes", "sale_id"
    elif source.key.startswith("crime_"):
        scope_table, scope_col, scope_city = "crime_incidents", "incident_id", source.key.split("crime_", 1)[1]
    if scope_table and dest_name:
        old_snapshot = _snapshot_rows_by_source_file(scope_table, dest_name, city=scope_city)
        old_ids = set(old_snapshot[scope_col]) if not old_snapshot.empty else set()
        _delete_ids(scope_table, scope_col, old_ids)

    # ── Load ──
    try:
        loader_result, log_text = _capture(source.loader)
        load_error = None
    except Exception as e:
        loader_result, log_text, load_error = 0, "", str(e)

    # ── Validate: did the new content actually produce anything? ──
    new_ids: set = set()
    if load_error is not None:
        upload_succeeded = False
    elif scope_table and dest_name:
        new_ids = _rows_by_source_file(scope_table, scope_col, dest_name, city=scope_city)
        upload_succeeded = len(new_ids) > 0
    elif source.key == "bike":
        shp_files = [p for p in new_files if p.suffix.lower() == ".shp"]
        upload_succeeded = any(
            _rows_by_source_file("bike_routes", "route_id", str(p.relative_to(source.dest_dir)))
            for p in shp_files
        )
    else:
        # "single" sources: each processes exactly one canonical file, so its
        # own return value is exactly this attempt's count — never stale data
        # left over from before (every load_*() here returns 0 on every
        # failure path without writing anything; see services/data_loader.py).
        upload_succeeded = loader_result > 0

    if not upload_succeeded:
        # The precheck already validated the file in isolation, so reaching
        # here means something more surprising happened (e.g. every row failed
        # a downstream sanity check the precheck doesn't replicate, such as a
        # date or bounding-box filter). Put back exactly what was cleared.
        if scope_table and not old_snapshot.empty:
            store.upsert_df(scope_table, old_snapshot)
        detail = load_error or ("the file didn't produce any usable rows \u2014 see the loader's own "
                                 "message below for why.")
        warnings = _warnings_from_log(log_text) if load_error is None else []
        return _rollback(detail, warnings)

    # ── Success: discard backups, then handle whatever this file's old content no longer covers ──
    _discard_backups(backups)
    removed_favorites = 0
    stale = old_ids - new_ids
    if stale and source.key == "redfin":
        # Put the dropped houses back (their pre-refresh data, from the
        # snapshot) so mark_removed_favorites has a row to flag rather than
        # leaving them deleted — sold/delisted/unfavorited houses stay visible
        # with their history intact, per the "removed from favorites" design.
        stale_rows = old_snapshot[old_snapshot[scope_col].isin(stale)]
        if not stale_rows.empty:
            store.upsert_df(scope_table, stale_rows)
        removed_favorites = store.mark_removed_favorites(stale, dest_name)
    # For sold/crime, `stale` ids were already cleared above and simply stay
    # gone — correct, since a shrunk export means those specific records are no
    # longer part of the authoritative file that produced them.

    warnings = _warnings_from_log(log_text)
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

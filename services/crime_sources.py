"""
services/crime_sources.py

Per-city parsers for data/crime/<city>/ — mirrors the CountyParserBase
architecture in data_loader.py's "Sold homes" section: one class per source
format, a shared standardized output contract, a registry, and a loader
(services.data_loader.load_crime) that treats every city the same way once
load_raw() has done the city-specific work.

Why per-city classes instead of one generic parser
----------------------------------------------------
Every city's open-data export uses different column names, different file
layouts (one file vs. one-per-year vs. two archives with slightly different
schemas), and different quirks — see each class's docstring, which cites the
exact source file(s)/columns referenced in the original R scripts
(process_crime_<city>.R) this was ported from. A single generic parser would
need a wall of if/elif per city anyway; separate classes keep each city's
logic readable and let you fix one city without risking another.

Standardized interim schema returned by every load_raw()
-----------------------------------------------------------
    occurred_at    raw date/time value (parsed by the caller)
    lat, lon       raw coordinate values (coerced to numeric by the caller)
    raw_type       free text describing the offense — fed to
                   services.crime_taxonomy.classify_crime_text()
    location_text  optional human-readable location (block address, etc.)
    source_file    originating filename (for traceability)
    natural_id     optional per-row identifier from the source (used to keep
                   incident_id stable across re-runs); omitted if the source
                   doesn't have one

The shared cleaning/classification/upsert pipeline lives in
services/data_loader.py::load_crime() — city parsers only need to get raw
files into this shape.

Bounding boxes are carried over verbatim from each process_crime_<city>.R
script (lon_low, lon_high, lat_low, lat_high) as a light sanity filter —
real crime exports occasionally contain a stray mis-geocoded row (0,0,
another state, another hemisphere) and these catch the obvious ones. Chicago
had no fixed box in the original script (it used the data's own min/max,
i.e. no real constraint), so it's None here too.
"""
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Column resolution helpers ─────────────────────────────────────────────

def _normalize_key(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace(".", "")


def _find_col(available_cols: list[str], candidates: list[str]) -> Optional[str]:
    """Case/space/underscore-insensitive match of the first candidate found."""
    norm_map = {_normalize_key(c): c for c in available_cols}
    for cand in candidates:
        key = _normalize_key(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _peek_columns(path: Path) -> list[str]:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return list(pd.read_excel(path, nrows=0).columns)
    try:
        return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    except UnicodeDecodeError:
        return list(pd.read_csv(path, nrows=0, encoding="latin-1").columns)


def _read_full(path: Path, usecols: list[str]) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, usecols=usecols)
    try:
        return pd.read_csv(path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, usecols=usecols, encoding="latin-1", low_memory=False)


def read_table(path: Path, wanted: dict[str, list[str]],
                required: list[str]) -> Optional[pd.DataFrame]:
    """
    Read one file, resolving each logical field in `wanted` (logical name ->
    candidate raw column names) against the file's actual header, and
    renaming matched columns to their logical name. Only the resolved
    columns are read (fast + memory-light — matters for multi-million-row
    files like Chicago's).

    Returns None (after printing why) if any `required` logical field has no
    match, instead of raising — one bad/unexpected file shouldn't abort the
    whole city.
    """
    try:
        header_cols = _peek_columns(path)
    except Exception as e:
        print(f"    Skipping {path.name}: could not read header ({e})")
        return None

    resolved: dict[str, str] = {}
    for logical, candidates in wanted.items():
        col = _find_col(header_cols, candidates)
        if col:
            resolved[logical] = col

    missing = [f for f in required if f not in resolved]
    if missing:
        preview = header_cols[:12] + (["..."] if len(header_cols) > 12 else [])
        print(f"    Skipping {path.name}: missing required column(s) {missing} "
              f"(columns found: {preview})")
        return None

    try:
        df = _read_full(path, usecols=list(resolved.values()))
    except Exception as e:
        print(f"    Skipping {path.name}: read error ({e})")
        return None

    df = df.rename(columns={raw: logical for logical, raw in resolved.items()})
    df["source_file"] = path.name
    return df


def _combine_text(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Concatenate whichever of `cols` are present into one text column —
    more text gives the keyword classifier more signal (e.g. a general
    'Primary Type' plus a more specific 'Description')."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([""] * len(df), index=df.index)
    parts = [df[c].astype(str).replace({"nan": "", "None": ""}) for c in present]
    out = parts[0]
    for p in parts[1:]:
        out = out.str.cat(p, sep=" ", na_rep="")
    return out


# ── Parser base ─────────────────────────────────────────────────────────────

class CrimeParserBase:
    city: str = "unknown"          # matches the data/crime/<city>/ folder name
    city_label: str = "Unknown"
    state: str = ""
    # (lon_low, lon_high, lat_low, lat_high) — None means "no bounding filter"
    bbox: Optional[tuple[float, float, float, float]] = None

    @classmethod
    def load_raw(cls, city_dir: Path, files: list[Path]) -> pd.DataFrame:
        """Read every relevant file in city_dir and return the standardized
        interim schema described in the module docstring. Must not raise for
        a single bad file — skip it and keep going."""
        raise NotImplementedError


# ── Baltimore ────────────────────────────────────────────────────────────────
# process_crime_baltimore.R: single xlsx (Part1_Crime_Beta_*.xlsx),
# CrimeDateTime / Latitude / Longitude. Baltimore's open-data "Description"
# field carries the offense (e.g. "AGG. ASSAULT", "AUTO THEFT", "LARCENY").

class BaltimoreCrimeParser(CrimeParserBase):
    city = "baltimore"
    city_label = "Baltimore, MD"
    state = "MD"
    bbox = (-76.8, -76.5, 39.0, 39.4)

    _WANTED = {
        "occurred_at":  ["CrimeDateTime", "Crime Date Time"],
        "lat":          ["Latitude", "Lat"],
        "lon":          ["Longitude", "Lon", "Long"],
        "type_primary": ["Description"],
        "type_detail":  ["CrimeCode", "Crime Code"],
        "location_text":["Location", "Block Address", "Location 1"],
        "natural_id":   ["ObjectId", "Object_ID", "_id"],
    }
    _REQUIRED = ["occurred_at", "lat", "lon"]

    @classmethod
    def load_raw(cls, city_dir: Path, files: list[Path]) -> pd.DataFrame:
        frames = []
        for path in files:
            df = read_table(path, cls._WANTED, cls._REQUIRED)
            if df is None or df.empty:
                continue
            df["raw_type"] = _combine_text(df, ["type_primary", "type_detail"])
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


# ── Boston ───────────────────────────────────────────────────────────────────
# process_crime_boston_v2.R: one CSV per year from Analyze Boston
# ("Crime Incident Reports"), columns OCCURRED_ON_DATE / Lat / Long. The R
# script hardcoded 2015-2023; here we just read every CSV present so future
# year exports work without a code change. OFFENSE_CODE_GROUP carries the
# category. (The R script also merged in a geocoded Cambridge PD extract —
# that source has no lat/lon of its own, only a text "Location" field
# requiring an external geocoder, so it's intentionally not replicated here;
# a Cambridge file dropped into this folder will simply be skipped with a
# "missing required column" message rather than silently mis-parsed.)

class BostonCrimeParser(CrimeParserBase):
    city = "boston"
    city_label = "Boston, MA"
    state = "MA"
    bbox = (-71.4, -71.0, 42.0, 42.5)

    _WANTED = {
        "occurred_at":  ["OCCURRED_ON_DATE", "OCCURRED_ON_DATE "],
        "lat":          ["Lat"],
        "lon":          ["Long", "Lon"],
        "type_primary": ["OFFENSE_CODE_GROUP"],
        "type_detail":  ["OFFENSE_DESCRIPTION"],
        "location_text":["STREET", "Location"],
        "natural_id":   ["INCIDENT_NUMBER"],
    }
    _REQUIRED = ["occurred_at", "lat", "lon"]

    @classmethod
    def load_raw(cls, city_dir: Path, files: list[Path]) -> pd.DataFrame:
        frames = []
        for path in files:
            df = read_table(path, cls._WANTED, cls._REQUIRED)
            if df is None or df.empty:
                continue
            df["raw_type"] = _combine_text(df, ["type_primary", "type_detail"])
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


# ── Buffalo ──────────────────────────────────────────────────────────────────
# process_crime_buffalo.R: single CSV (Crime_Incidents_*.csv),
# "Incident Datetime" / Latitude / Longitude, parsed with an AM/PM format.

class BuffaloCrimeParser(CrimeParserBase):
    city = "buffalo"
    city_label = "Buffalo, NY"
    state = "NY"
    bbox = (-79.1, -78.4, 42.5, 43.02)

    _WANTED = {
        "occurred_at":  ["Incident Datetime", "Incident Date Time"],
        "lat":          ["Latitude"],
        "lon":          ["Longitude"],
        "type_primary": ["Incident Type Primary"],
        "type_detail":  ["Parent Incident Type", "Incident Description"],
        "location_text":["Address Line1", "Location"],
        "natural_id":   ["Case Number"],
    }
    _REQUIRED = ["occurred_at", "lat", "lon"]

    @classmethod
    def load_raw(cls, city_dir: Path, files: list[Path]) -> pd.DataFrame:
        frames = []
        for path in files:
            df = read_table(path, cls._WANTED, cls._REQUIRED)
            if df is None or df.empty:
                continue
            df["raw_type"] = _combine_text(df, ["type_primary", "type_detail"])
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


# ── Chicago ──────────────────────────────────────────────────────────────────
# process_crime_chicago.R: single (large) CSV "Crimes_-_2001_to_Present_*",
# Date / Latitude / Longitude, 'Primary Type' + 'Description' for category.
# No bounding box in the original script — Chicago's own data quality is
# used as the only filter, same here.

class ChicagoCrimeParser(CrimeParserBase):
    city = "chicago"
    city_label = "Chicago, IL"
    state = "IL"
    bbox = None

    _WANTED = {
        "occurred_at":  ["Date"],
        "lat":          ["Latitude"],
        "lon":          ["Longitude"],
        "type_primary": ["Primary Type"],
        "type_detail":  ["Description"],
        "location_text":["Block"],
        "natural_id":   ["ID"],
    }
    _REQUIRED = ["occurred_at", "lat", "lon"]

    @classmethod
    def load_raw(cls, city_dir: Path, files: list[Path]) -> pd.DataFrame:
        frames = []
        for path in files:
            df = read_table(path, cls._WANTED, cls._REQUIRED)
            if df is None or df.empty:
                continue
            # Chicago's Date column is consistently '%m/%d/%Y %I:%M:%S %p' —
            # same format the R script specified. Parsing it explicitly (vs.
            # generic inference) matters here: this file can run into the
            # millions of rows.
            df["occurred_at"] = pd.to_datetime(
                df["occurred_at"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
            )
            df["raw_type"] = _combine_text(df, ["type_primary", "type_detail"])
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


# ── Indianapolis ─────────────────────────────────────────────────────────────
# process_crime_indianapolis.R: every file in the folder (multi-year
# extracts), DATE_ (+ separate TIME) / X_COORD / Y_COORD. IMPD's category
# field has been spelled a few different ways across yearly extracts, so we
# try several candidates.

class IndianapolisCrimeParser(CrimeParserBase):
    city = "indianapolis"
    city_label = "Indianapolis, IN"
    state = "IN"
    bbox = (-86.4, -85.9, 39.6, 40.0)

    _WANTED = {
        "occurred_at":  ["DATE_", "Date"],
        "time_part":    ["TIME", "Time"],
        "lat":          ["Y_COORD", "Y"],
        "lon":          ["X_COORD", "X"],
        "type_primary": ["CRIME", "UCR_CRIME_CATEGORY", "Offense", "CRIME_DESCRIPTION"],
        "location_text":["ADDRESS", "BLOCK_ADDRESS"],
        "natural_id":   ["CASE_NUMBER", "INCIDENT_NUMBER"],
    }
    _REQUIRED = ["occurred_at", "lat", "lon"]

    @classmethod
    def load_raw(cls, city_dir: Path, files: list[Path]) -> pd.DataFrame:
        frames = []
        for path in files:
            df = read_table(path, cls._WANTED, cls._REQUIRED)
            if df is None or df.empty:
                continue
            date_str = df["occurred_at"].astype(str).str.strip()
            if "time_part" in df.columns:
                date_str = date_str + " " + df["time_part"].astype(str).str.strip()
            # 'mixed' lets pandas infer per-row — IMPD's TIME field has shown
            # up as both "HH:MM" and "HH:MM:SS" across different yearly
            # exports (mirrors the multi-attempt fallback in the R script).
            df["occurred_at"] = pd.to_datetime(date_str, errors="coerce", format="mixed")
            df["raw_type"] = _combine_text(df, ["type_primary"])
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


# ── Minneapolis ──────────────────────────────────────────────────────────────
# process_crime_minneapolis.R: "Minneapolis.csv" (Occurred_Date / Latitude /
# Longitude) is the one actually forwarded to processing; a "St Paul.csv" is
# read in the R script but never passed to process_df_filtered (i.e. not
# actually wired up there either). Every file in the folder is tried
# independently against the same candidate columns, so a compatible St Paul
# extract will still load if its columns happen to match; if not, it's
# skipped with a clear message rather than silently dropped.

class MinneapolisCrimeParser(CrimeParserBase):
    city = "minneapolis"
    city_label = "Minneapolis, MN"
    state = "MN"
    bbox = (-93.4, -93.1, 44.8, 45.1)

    _WANTED = {
        "occurred_at":  ["Occurred_Date", "OccurredDate", "DATE", "Reported_Date"],
        "lat":          ["Latitude", "Lat"],
        "lon":          ["Longitude", "Long", "Lon"],
        "type_primary": ["Offense", "OFFENSE"],
        "type_detail":  ["Description", "OffenseDescription"],
        "location_text":["Address", "Block Address"],
        "natural_id":   ["ObjectId", "ControlNumber"],
    }
    _REQUIRED = ["occurred_at", "lat", "lon"]

    @classmethod
    def load_raw(cls, city_dir: Path, files: list[Path]) -> pd.DataFrame:
        frames = []
        for path in files:
            df = read_table(path, cls._WANTED, cls._REQUIRED)
            if df is None or df.empty:
                continue
            # Source timestamps have shown up as e.g. "2023-05-01T08:00:00+00"
            # — strip a trailing UTC-offset fragment the same way the R
            # script does (split on '+') before parsing.
            df["occurred_at"] = df["occurred_at"].astype(str).str.split("+").str[0]
            df["raw_type"] = _combine_text(df, ["type_primary", "type_detail"])
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


# ── Philadelphia ─────────────────────────────────────────────────────────────
# process_crime_philadelphia.R: every file in the folder, dispatch_date_time
# / lat / lng. OpenDataPhilly's incidents carry both text_general_code (a
# readable category) and ucr_general (a 2-digit UCR code) — we use the text.

class PhiladelphiaCrimeParser(CrimeParserBase):
    city = "philadelphia"
    city_label = "Philadelphia, PA"
    state = "PA"
    bbox = (-75.3, -74.9, 39.8, 40.2)

    _WANTED = {
        "occurred_at":  ["dispatch_date_time", "dispatch_date"],
        "lat":          ["lat"],
        "lon":          ["lng", "lon"],
        "type_primary": ["text_general_code"],
        "location_text":["location_block"],
        "natural_id":   ["objectid", "dc_key"],
    }
    _REQUIRED = ["occurred_at", "lat", "lon"]

    @classmethod
    def load_raw(cls, city_dir: Path, files: list[Path]) -> pd.DataFrame:
        frames = []
        for path in files:
            df = read_table(path, cls._WANTED, cls._REQUIRED)
            if df is None or df.empty:
                continue
            df["raw_type"] = _combine_text(df, ["type_primary"])
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


# ── Pittsburgh ───────────────────────────────────────────────────────────────
# process_crime_pittsburgh.R: two xlsx archives combined (current +
# historical blotter), X/Y renamed to lon/lat. Verified directly against
# both sample files: the current export uses INCIDENTHIERARCHYDESC, the
# older one uses HIERARCHYDESC for the same concept (highest-severity
# offense per the data dictionary's UCR hierarchy) — both are tried, and
# combined with the full OFFENSES charge list for classification, since a
# single incident's top-line label can under-describe it (e.g. a row labeled
# "PROP MISSILE INTO OCC VEHICLE" whose OFFENSES list separately includes
# "Discharge of a Firearm into Occupied Structure").

class PittsburghCrimeParser(CrimeParserBase):
    city = "pittsburgh"
    city_label = "Pittsburgh, PA"
    state = "PA"
    bbox = (-80.3, -79.6, 40.1, 40.7)

    _WANTED = {
        "occurred_at":  ["INCIDENTTIME"],
        "lat":          ["Y"],
        "lon":          ["X"],
        "type_primary": ["INCIDENTHIERARCHYDESC", "HIERARCHYDESC", "HIERARCHY_DESC"],
        "type_detail":  ["OFFENSES"],
        "location_text":["INCIDENTLOCATION"],
        "natural_id":   ["PK"],
    }
    _REQUIRED = ["occurred_at", "lat", "lon"]

    @classmethod
    def load_raw(cls, city_dir: Path, files: list[Path]) -> pd.DataFrame:
        frames = []
        for path in files:
            df = read_table(path, cls._WANTED, cls._REQUIRED)
            if df is None or df.empty:
                continue
            df["raw_type"] = _combine_text(df, ["type_primary", "type_detail"])
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


# ── Registry ─────────────────────────────────────────────────────────────────
# Add a new city: subclass CrimeParserBase, implement load_raw(), append here.
# The folder name under data/crime/ must equal the class's `city` attribute.

CRIME_PARSERS: list[type[CrimeParserBase]] = [
    BaltimoreCrimeParser,
    BostonCrimeParser,
    BuffaloCrimeParser,
    ChicagoCrimeParser,
    IndianapolisCrimeParser,
    MinneapolisCrimeParser,
    PhiladelphiaCrimeParser,
    PittsburghCrimeParser,
]

CRIME_CITY_KEYS: list[str] = [p.city for p in CRIME_PARSERS]

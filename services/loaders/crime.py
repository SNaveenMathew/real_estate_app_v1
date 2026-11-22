"""`Crime` loaders, moved out of services/data_loader.py.

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
import numpy as np


from config import settings
import db.duckdb_store as store
from services.crime_sources import CRIME_PARSERS, CRIME_CITY_KEYS
from services.crime_taxonomy import classify_crime_text


__all__ = ["_CRIME_SCHEMA_COLS", "_crime_incident_id", "load_crime"]

# ── Crime ────────────────────────────────────────────────────────────────────
# Standardizes data/crime/<city>/ (one folder per city, any mix of .csv/.xlsx)
# across every registered city parser in services/crime_sources.py into one
# severity-weighted table. Mirrors the "sold homes" parser architecture: each
# city handles its own file layout/columns in load_raw(), then this function
# applies the SAME cleaning, classification, and ID assignment to everyone so
# the output is directly comparable across cities.
#
# Inspired by the process_crime_<city>.R scripts (per-city bounding boxes,
# invalid/zero lat-lon filtering, multi-file handling) with two differences
# by design: (1) crime TYPE is classified and severity-weighted — the R
# pipeline's shared process_crime.R only ever received time/lat/lon, never a
# type column — and (2) results land in a queryable DuckDB table rather than
# a plotted heatmap, so the general agent can also answer questions like
# "what's the weighted crime severity near tract X" via query_database.

_CRIME_SCHEMA_COLS = [
    "incident_id",
    "city",
    "source_file",
    "occurred_at",
    "year",
    "month",
    "year_month",
    "lat",
    "lon",
    "category",
    "category_label",
    "severity_weight",
    "raw_type",
    "location_text",
]


def _crime_incident_id(city: str, source_file: str, natural_id, row_pos: int) -> str:
    """
    Stable, idempotent ID: prefers a natural per-row identifier from the source
    (Pittsburgh's PK, Chicago's ID, Boston's INCIDENT_NUMBER, ...) when the parser
    found one, else falls back to its position within that file.

    A natural ID is keyed on city + id alone, deliberately *not* including
    source_file: these IDs come from the source city's own system and are
    expected to be globally unique for that city, so the same real-world
    incident reported in two different exports (e.g. a full historical archive
    that overlaps a few years with a newer incremental one) collapses onto the
    same row instead of being double-counted. load_crime()'s existing
    `combined.drop_duplicates(subset=["incident_id"])` is what actually performs
    that collapse — this is what makes the IDs line up for it to find.

    A position-based fallback ID *does* include source_file, since a row's
    position is only meaningful within the one file it came from. Re-running
    setup_data.py on the same files won't create duplicates either way; note
    that if a source file with no natural ID is later re-exported with rows
    reordered or removed, its position-based IDs can shift — acceptable for
    open-data crime exports, which are typically wholesale replacements rather
    than incremental diffs.
    """
    natural_text = "" if natural_id is None or pd.isna(natural_id) else str(natural_id).strip()
    has_natural = natural_text.lower() not in ("", "nan", "none")
    if has_natural:
        if isinstance(natural_id, (int, np.integer)):
            natural_text = str(int(natural_id))
        elif isinstance(natural_id, (float, np.floating)) and np.isfinite(natural_id) and natural_id.is_integer():
            natural_text = str(int(natural_id))
        key = f"{city}:{natural_text}"
    else:
        key = f"{city}:{source_file}:row{row_pos}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def load_crime() -> int:
    """
    Load and standardize crime data from data/crime/<city>/ for every city
    registered in services/crime_sources.CRIME_PARSERS. Safe to call when the
    directory (or a given city's subfolder) doesn't exist yet — just loads
    whatever's present.
    """
    crime_root = settings.data_dir / "crime"
    if not crime_root.exists():
        print(f"  No crime data directory found at {crime_root}")
        return 0

    all_rows = []
    for parser in CRIME_PARSERS:
        city_dir = crime_root / parser.city
        if not city_dir.exists():
            continue
        files = sorted(p for p in city_dir.iterdir() if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls"))
        if not files:
            continue

        print(
            f"  {parser.city_label}: {len(files)} file(s) — "
            f"{', '.join(f.name for f in files[:4])}"
            f"{', ...' if len(files) > 4 else ''}"
        )
        try:
            raw = parser.load_raw(city_dir, files)
        except Exception as e:
            print(f"    Error loading {parser.city_label}: {e}")
            import traceback

            traceback.print_exc()
            continue

        if raw is None or raw.empty:
            print(f"    No usable rows for {parser.city_label}")
            continue

        df = raw.copy()

        # ── Shared cleaning (mirrors the R scripts' filtering) ────────────
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df.loc[df["lat"] == 0, "lat"] = np.nan
        df.loc[df["lon"] == 0, "lon"] = np.nan
        if not pd.api.types.is_datetime64_any_dtype(df["occurred_at"]):
            df["occurred_at"] = pd.to_datetime(df["occurred_at"], errors="coerce", format="mixed")

        before = len(df)
        df = df.dropna(subset=["lat", "lon", "occurred_at"])
        if parser.bbox:
            lon_lo, lon_hi, lat_lo, lat_hi = parser.bbox
            df = df[df["lat"].between(lat_lo, lat_hi) & df["lon"].between(lon_lo, lon_hi)]
        dropped = before - len(df)
        if dropped:
            print(f"    Dropped {dropped:,} row(s): missing/invalid date, lat, lon, or outside the expected area")

        if df.empty:
            print(f"    No valid rows remain for {parser.city_label}")
            continue

        df = df.reset_index(drop=True)

        # ── Classification ─────────────────────────────────────────────
        classified = df["raw_type"].fillna("").astype(str).apply(classify_crime_text)
        df["category"] = [c[0] for c in classified]
        df["category_label"] = [c[1] for c in classified]
        df["severity_weight"] = [c[2] for c in classified]

        # ── Time parts (mirrors the R pipeline's year / year_month frames) ─
        df["year"] = df["occurred_at"].dt.year
        df["month"] = df["occurred_at"].dt.month
        df["year_month"] = df["occurred_at"].dt.strftime("%Y-%m")

        # ── IDs ────────────────────────────────────────────────────────
        nat_id_col = df["natural_id"] if "natural_id" in df.columns else pd.Series([None] * len(df))
        df["incident_id"] = [
            _crime_incident_id(parser.city, sf, nid, i)
            for i, (sf, nid) in enumerate(zip(df["source_file"], nat_id_col))
        ]
        df["city"] = parser.city

        for col in _CRIME_SCHEMA_COLS:
            if col not in df.columns:
                df[col] = None

        n_by_cat = df["category_label"].value_counts()
        top3 = ", ".join(f"{k} ({v:,})" for k, v in n_by_cat.head(3).items())
        yr_lo, yr_hi = df["year"].min(), df["year"].max()
        print(f"    ✓ {len(df):,} incidents ({int(yr_lo)}–{int(yr_hi)}) — top categories: {top3}")

        # Surface what's landing in the catch-all bucket so classification
        # gaps are visible instead of silently averaging away. Some of this
        # is expected — a raw police blotter/incident log typically includes
        # plenty of non-criminal entries (welfare checks, alarms, found
        # property, warrant service, ...) that correctly belong at the
        # lowest weight — but a large share is worth a look, since it also
        # means those incidents contribute the same low weight to the
        # 'Crime' map layer regardless of what they actually were.
        other_mask = df["category"] == "other"
        other_n = int(other_mask.sum())
        other_pct = 100 * other_n / len(df) if len(df) else 0
        if other_n and other_pct >= 5:
            top_unclassified = (
                df.loc[other_mask, "raw_type"]
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(10)
            )
            print(
                f"    ⚠ {other_n:,} ({other_pct:.0f}%) incidents fell into "
                f"'Other/Unclassified'. Most common unclassified raw text "
                f"for {parser.city_label}:"
            )
            for txt, n in top_unclassified.items():
                snippet = txt if len(txt) <= 80 else txt[:77] + "..."
                print(f"        {n:>7,}×  {snippet}")
            print(
                f"      → If real crime types are being missed above, add "
                f"keyword patterns for them in services/crime_taxonomy.py."
            )

        all_rows.append(df[_CRIME_SCHEMA_COLS])

    if not all_rows:
        print(
            f"  No crime data loaded. Expected folders like "
            f"data/crime/chicago/, data/crime/pittsburgh/, etc. "
            f"(supported cities: {', '.join(CRIME_CITY_KEYS)})"
        )
        return 0

    combined = pd.concat(all_rows, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["incident_id"])
    if len(combined) != before:
        print(f"  Deduplicated {before - len(combined):,} row(s) with a repeated incident_id")

    cities = combined["city"].dropna().unique().tolist()
    conn = store.get_conn()
    placeholders = ", ".join("?" for _ in cities)
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(f"DELETE FROM crime_incidents WHERE city IN ({placeholders})", cities)
        store.upsert_df("crime_incidents", combined)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    print(f"  ✓ {len(combined):,} total crime incidents loaded across {combined['city'].nunique()} cities")
    return len(combined)

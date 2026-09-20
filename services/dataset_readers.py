"""Generic dataset readers for the Data page.

These follow the conventions already used by ``services/data_loader.py`` so an uploaded file is
read the same way the built-in sources are:

  * CSV/TSV: ``utf-8-sig`` first (BOM-safe), ``latin-1`` fallback - as in ``load_sold_homes``.
  * Excel (.xlsx/.xls): header row auto-detected, because government files (e.g. the Census
    CBSA delineation list) carry title rows above the header - as in ``load_cbsa_crosswalk``.
  * Shapefile / GeoJSON / GeoPackage / zipped shapefile: reprojected to EPSG:4326, geometry kept
    as GeoJSON text plus a bounding box - as in ``load_bike_routes``.
  * Parquet / JSON / JSON-lines.

Everything is read **as text first** so identifiers (FIPS, ZIP, parcel ids) keep their leading
zeros; types are then inferred column by column.  Column names are turned into safe SQL
identifiers (the original header is kept as ``source_name``) because the SQL guardrails reject
words such as ``update`` or ``load`` anywhere in a query.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TABULAR = {".csv": "csv", ".tsv": "csv", ".txt": "csv", ".xlsx": "excel", ".xlsm": "excel",
           ".xls": "excel", ".json": "json", ".jsonl": "json", ".ndjson": "json", ".parquet": "parquet"}
GEO = {".geojson": "geo", ".gpkg": "geo", ".shp": "geo", ".zip": "geo", ".kml": "geo"}
SUPPORTED_EXTENSIONS = sorted({**TABULAR, **GEO})

# Words the SQL guardrails (agents/tools.py::_FORBIDDEN, services/guardrails.py) reject anywhere in a
# query, plus common SQL keywords. Identifiers matching these get a suffix.
RESERVED_WORDS = {
    "insert", "update", "delete", "drop", "create", "alter", "truncate", "copy", "export", "attach",
    "detach", "install", "load", "call", "pragma", "replace", "grant", "revoke", "import",
    "select", "from", "where", "group", "order", "by", "limit", "offset", "table", "join", "on", "as",
    "and", "or", "not", "null", "true", "false", "union", "all", "distinct", "case", "when", "then",
    "else", "end", "with", "values", "in", "is", "like", "between", "having", "index", "primary",
    "key", "default", "check", "column", "constraint", "user", "using", "left", "right", "inner",
    "outer", "cross", "natural", "exists", "any", "some", "over", "window", "partition", "row",
    "rows", "range", "current_date", "current_time", "current_timestamp", "desc", "asc", "cast",
    "interval", "date", "time", "timestamp", "year", "month", "day", "hour", "minute", "second",
}


class DatasetReadError(ValueError):
    """The file could not be read; the message is safe to show to the user."""


@dataclass
class ReadResult:
    df: pd.DataFrame
    format: str
    sheet: str | None = None
    sheets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    geometry: dict | None = None      # {"kinds": [...], "points_only": bool} for spatial files


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------

def sanitize_identifier(raw: Any, taken: set[str]) -> str:
    s = unicodedata.normalize("NFKD", str(raw)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.strip().lower()).strip("_")
    if not s:
        s = "column"
    if s[0].isdigit():
        s = "c_" + s
    s = s[:48].rstrip("_") or "column"
    if s in RESERVED_WORDS:
        s = s + "_col"
    base, i = s, 2
    while s in taken:
        s = f"{base}_{i}"
        i += 1
    taken.add(s)
    return s


def slugify_table_name(title: str) -> str:
    s = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.strip().lower()).strip("_")
    if not s or not s[0].isalpha():
        s = "ds_" + s
    s = s[:48].rstrip("_")
    if len(s) < 3:
        s = (s + "_data")[:48]
    if s in RESERVED_WORDS:
        s += "_data"
    return s


def qi(name: str) -> str:
    """Quote an identifier for DuckDB."""
    return '"' + str(name).replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# Format detection + per-format readers
# ---------------------------------------------------------------------------

def detect_format(path: Path, ext: str | None = None) -> str:
    ext = (ext or path.suffix).lower()
    if ext == ".json":
        try:
            head = path.read_text(encoding="utf-8-sig", errors="ignore")[:4096]
            if '"FeatureCollection"' in head or '"features"' in head:
                return "geo"
        except Exception:
            pass
        return "json"
    if ext in TABULAR:
        return TABULAR[ext]
    if ext in GEO:
        return GEO[ext]
    raise DatasetReadError(
        f"'{ext or 'no extension'}' files are not supported. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}.")


def list_sheets(path: Path) -> list[str]:
    try:
        return list(pd.ExcelFile(path).sheet_names)
    except Exception:
        return []


def _looks_numeric(v: str) -> bool:
    return bool(re.fullmatch(r"[-+$]?[\d,]*\.?\d+%?", str(v).strip()))


def _detect_header_row(raw: pd.DataFrame) -> int:
    """First of the top rows that is mostly filled with non-numeric text (a header, not a title)."""
    width = raw.shape[1]
    for i in range(min(len(raw), 15)):
        row = raw.iloc[i]
        filled = int(row.notna().sum())
        if filled < max(2, 0.6 * width):
            continue
        texty = sum(1 for v in row if isinstance(v, str) and not _looks_numeric(v))
        if texty >= 0.6 * filled:
            return i
    return 0


def _read_csv(path: Path, ext: str, skiprows: int | None, notes: list[str]) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as fh:
                sample = fh.read(65536)
            if ext == ".tsv":
                delim = "\t"
            else:
                try:
                    delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                except Exception:
                    delim = ","
            df = pd.read_csv(path, encoding=enc, sep=delim, dtype=str, low_memory=False,
                             skiprows=skiprows or None)
            if enc != "utf-8-sig":
                notes.append("Read with latin-1 encoding (the file is not valid UTF-8).")
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise DatasetReadError("The file is empty.")
        except pd.errors.ParserError as exc:
            raise DatasetReadError(f"The file could not be parsed as delimited text: {exc}")
    raise DatasetReadError("The file could not be decoded as UTF-8 or Latin-1 text.")


def _read_excel(path: Path, sheet: str | None, skiprows: int | None, notes: list[str]) -> tuple[pd.DataFrame, str, list[str]]:
    try:
        xl = pd.ExcelFile(path)
    except ImportError as exc:
        raise DatasetReadError(f"Reading this Excel format needs an extra package: {exc}")
    except Exception as exc:
        raise DatasetReadError(f"The Excel file could not be opened: {exc}")
    sheets = list(xl.sheet_names)
    target = sheet if sheet in sheets else sheets[0]
    raw = xl.parse(target, header=None, dtype=str)
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        raise DatasetReadError(f"Sheet '{target}' has no data.")
    hdr = int(skiprows) if skiprows is not None else _detect_header_row(raw)
    if hdr:
        notes.append(f"Header found on row {hdr + 1} of sheet '{target}'; rows above it were skipped.")
    header = []
    for i, v in enumerate(raw.iloc[hdr]):
        header.append(str(v).strip() if pd.notna(v) and str(v).strip() else f"column_{i + 1}")
    body = raw.iloc[hdr + 1:].reset_index(drop=True)
    body.columns = header
    return body, target, sheets


def _read_json(path: Path, ext: str) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        if ext in (".jsonl", ".ndjson"):
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            return pd.json_normalize(rows)
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetReadError(f"Invalid JSON: {exc}")
    if isinstance(data, list):
        return pd.json_normalize(data)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return pd.json_normalize(value)
        try:
            return pd.DataFrame(data)
        except ValueError:
            return pd.json_normalize(data)
    raise DatasetReadError("The JSON file does not contain a list of records.")


def _read_geo(path: Path, ext: str, notes: list[str]) -> tuple[pd.DataFrame, dict]:
    try:
        import geopandas as gpd
    except ImportError as exc:   # pragma: no cover - geopandas is a core dependency
        raise DatasetReadError(f"Spatial files need geopandas: {exc}")
    target = f"zip://{path}" if ext == ".zip" else str(path)
    try:
        gdf = gpd.read_file(target)
    except Exception as exc:
        raise DatasetReadError(f"The spatial file could not be read: {exc}")
    if gdf.empty:
        raise DatasetReadError("The spatial file has no features.")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
        notes.append("No coordinate system found in the file; assumed EPSG:4326 (WGS-84).")
    elif str(gdf.crs).upper() != "EPSG:4326":
        notes.append(f"Reprojected from {gdf.crs} to EPSG:4326.")
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise DatasetReadError("All features in the spatial file have empty geometry.")
    kinds = sorted(gdf.geometry.geom_type.unique().tolist())
    geoms = gdf.geometry
    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    taken = {str(c).lower() for c in df.columns}
    lat_name = "lat" if "lat" not in taken else "centroid_lat"
    lon_name = "lon" if "lon" not in taken else "centroid_lon"
    pts = geoms.representative_point()
    df[lon_name] = pts.x.values
    df[lat_name] = pts.y.values
    bounds = geoms.bounds
    df["min_lon"], df["min_lat"] = bounds["minx"].values, bounds["miny"].values
    df["max_lon"], df["max_lat"] = bounds["maxx"].values, bounds["maxy"].values
    df["geometry_json"] = [json.dumps(g.__geo_interface__, ensure_ascii=False) for g in geoms]
    if kinds != ["Point"]:
        notes.append(f"Geometry types: {', '.join(kinds)}. lat/lon hold a representative point of each "
                     "feature, so they are approximate for lines and polygons.")
    return df, {"kinds": kinds, "points_only": kinds == ["Point"]}


def read_dataset(path: str | Path, filename: str | None = None, *, sheet: str | None = None,
                 skiprows: int | None = None) -> ReadResult:
    """Read any supported file into a text-typed DataFrame (see module docstring)."""
    path = Path(path)
    ext = Path(filename or path.name).suffix.lower()
    fmt = detect_format(path, ext)
    notes: list[str] = []
    geometry = None
    sheets: list[str] = []
    used_sheet = None
    if fmt == "csv":
        df = _read_csv(path, ext, skiprows, notes)
    elif fmt == "excel":
        df, used_sheet, sheets = _read_excel(path, sheet, skiprows, notes)
    elif fmt == "json":
        df = _read_json(path, ext)
    elif fmt == "parquet":
        try:
            import geopandas as gpd
            gdf = gpd.read_parquet(path)
            df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
            df["geometry_json"] = [json.dumps(g.__geo_interface__) if g is not None else None for g in gdf.geometry]
        except Exception:
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                raise DatasetReadError(f"The parquet file could not be read: {exc}")
    else:
        df, geometry = _read_geo(path, ext, notes)
    df = df.loc[:, [not (str(c).startswith("Unnamed:") and df[c].isna().all()) for c in df.columns]]
    if df.empty or len(df.columns) == 0:
        raise DatasetReadError("The file has no rows to import.")
    return ReadResult(df=df, format=fmt, sheet=used_sheet, sheets=sheets, notes=notes, geometry=geometry)


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

_ID_HINT = re.compile(r"(^|_)(id|ids|code|fips|geoid\w*|zip\w*|postal\w*|parid|parcel\w*|tract\w*|"
                      r"cbsa\w*|msa\w*|apn|account|phone|statefp|countyfp|tractce|blkgrpce)($|_)", re.I)
_DATE_HINT = re.compile(r"(date|time|_at$|_on$|timestamp|dt$)", re.I)
_BOOL_TOKENS = {"true", "false", "yes", "no", "y", "n", "t", "f"}


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    """Strip $ , % and whitespace, then coerce - reusing data_loader's helper when importable.

    Note: data_loader._clean_numeric only acts on ``dtype == object`` columns; under pandas 3 text
    columns have the dedicated ``str`` dtype, so the input is cast to object first.
    """
    obj = s.astype(object)
    try:
        from services.data_loader import _clean_numeric
        cleaned = _clean_numeric(obj.astype(str).str.replace(r"[%\s]", "", regex=True).astype(object))
        return pd.to_numeric(cleaned, errors="coerce")
    except Exception:
        return pd.to_numeric(obj.astype(str).str.replace(r"[$,%\s]", "", regex=True), errors="coerce")


def _text_series(s: pd.Series) -> pd.Series:
    out = s.astype(object).where(s.notna(), None)
    return out.map(lambda v: (str(v).strip() or None) if v is not None else None)


def infer_series(name: str, s: pd.Series) -> tuple[pd.Series, str]:
    """Return (converted series, dtype label) for one column."""
    if pd.api.types.is_bool_dtype(s):
        return s, "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, "datetime"
    if pd.api.types.is_integer_dtype(s):
        return s.astype("Int64"), "integer"
    if pd.api.types.is_float_dtype(s):
        return s.astype("float64"), "float"

    txt = _text_series(s)
    vals = txt.dropna()
    if vals.empty:
        return txt, "text"
    lens = vals.str.len()
    digit_frac = float(vals.str.fullmatch(r"\d+").mean())
    lead_zero = float(vals.str.match(r"^0\d+$").mean()) > 0.001
    const_width = lens.nunique() == 1 and int(lens.iloc[0]) >= 5
    id_like = bool(_ID_HINT.search(name)) or lead_zero or (digit_frac > 0.95 and const_width)
    if id_like:
        return txt, "text"

    low = set(vals.str.lower().unique())
    if 1 <= len(low) <= 2 and low <= _BOOL_TOKENS and len(low) == 2:
        pos = {"true", "yes", "y", "t"}
        return txt.map(lambda v: None if v is None else v.lower() in pos).astype("boolean"), "boolean"

    num = _clean_numeric_series(vals)
    if num.notna().mean() >= 0.95:
        full = _clean_numeric_series(txt.fillna(""))
        full = full.where(txt.notna(), np.nan)
        nn = full.dropna()
        if len(nn) and bool((nn % 1 == 0).all()) and float(nn.abs().max()) < 2 ** 53:
            return full.astype("Int64"), "integer"
        return full.astype("float64"), "float"

    if _DATE_HINT.search(name) or float(vals.str.contains(r"[-/:]").mean()) > 0.9:
        sample = vals.head(200)
        short_digits = float(sample.str.fullmatch(r"\d{1,4}").mean()) > 0.5
        if not short_digits:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.95:
                return pd.to_datetime(txt, errors="coerce", format="mixed"), "datetime"
    return txt, "text"


def normalize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Typed copy of ``df`` with safe column names, plus one record per column."""
    taken: set[str] = set()
    cols: dict[str, pd.Series] = {}
    records: list[dict] = []
    for i, original in enumerate(df.columns):
        safe = sanitize_identifier(original, taken)
        series, dtype = infer_series(safe, df.iloc[:, i])
        cols[safe] = series.reset_index(drop=True)
        records.append({"name": safe, "source_name": str(original), "dtype": dtype})
    return pd.DataFrame(cols), records


def json_safe(value: Any) -> Any:
    """Make a cell JSON-serialisable (NaN/NaT -> None, numpy -> python, timestamps -> ISO text)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def preview_records(df: pd.DataFrame, n: int = 8) -> list[dict]:
    head = df.head(n)
    return [{c: json_safe(v) for c, v in row.items()} for row in head.to_dict(orient="records")]

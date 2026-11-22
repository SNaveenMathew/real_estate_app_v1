"""`Sold homes — county-parser architecture` loaders, moved out of services/data_loader.py.

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


from services.loaders.redfin import _REDFIN_COL_MAP, _clean_numeric  # shared Redfin helpers

__all__ = [
    "_NON_ARMS_LENGTH",
    "_ESTATE_CODES",
    "_classify_arms_length",
    "CountyParserBase",
    "AlleghenyParser",
    "GenericSoldParser",
    "_COUNTY_PARSERS",
    "_detect_parser",
    "_SOLD_SCHEMA_COLS",
    "load_sold_homes",
]

# ── Sold homes — county-parser architecture ───────────────────────────────────
#
# Each county website exports data in its own format.  A "parser" is a class
# with a single class method  parse(df) -> pd.DataFrame  that maps raw county
# columns to the standard sold_homes schema.
#
# To add a new county: subclass CountyParserBase, implement parse(), and add
# a signature tuple to _COUNTY_PARSERS so auto-detection can identify it.
#
# Standard schema output columns (all nullable except sale_id):
#   sale_id, source_county, source_file, parid, address, city, state, zip,
#   full_address, sold_price, sold_date, record_date, list_price,
#   sqft, beds, baths, year_built, sale_code, sale_desc, instr_type,
#   instr_type_desc, deed_book, deed_page, muni_code, muni_desc,
#   school_code, school_desc, is_arms_length, arms_length_flag,
#   lat, lon, geocode_status, geocode_source, geocode_accuracy,
#   tract_fips, county_fips
# ─────────────────────────────────────────────────────────────────────────────

# Sale codes that indicate non-arm's-length transactions.
# Each entry: (code_value_or_set, reason_string)
# Applies across county parsers; each parser maps its codes to this.
_NON_ARMS_LENGTH = {
    # Allegheny-specific codes (SALECODE column)
    "3": "Love and affection / nominal consideration",
    "5": "Quit-claim deed",
    "6": "Foreclosure / sheriff's sale",
    "11": "Related party transfer",
    "16": "Government / tax sale",
    "99": "Corrective deed or duplicate sale",
    "CO": "Corrective instrument",  # INSTRTYP level
    # Multi-parcel (distorts per-unit price)
    "H": "Multi-parcel sale",
    # Generic
    "LOVE AND AFFECTION": "Love and affection / nominal consideration",
    "CORRECTIVE": "Corrective deed",
    "ESTATE": "Estate sale",  # flagged but not always excluded
    "FORECLOSURE": "Foreclosure",
    "SHERIFF": "Sheriff's sale",
}

_ESTATE_CODES = {"37"}  # flagged as questionable but kept; analyst can filter


def _classify_arms_length(sale_code: str, sale_desc: str, instr_type: str, price: float) -> tuple[bool, str]:
    """
    Return (is_arms_length, flag_reason).
    Anything clearly not at market value → False.
    Estate sales → True but with a flag noting they may be discounted.
    """
    sc = str(sale_code or "").strip().upper()
    sd = str(sale_desc or "").strip().upper()
    it = str(instr_type or "").strip().upper()
    pr = price or 0

    # Nominal / token consideration
    if pr <= 1:
        return False, "Nominal price ($1 or $0)"

    # Check sale code
    reason = _NON_ARMS_LENGTH.get(sc) or _NON_ARMS_LENGTH.get(it)
    if reason:
        return False, reason

    # Keyword scan of description
    for kw, r in _NON_ARMS_LENGTH.items():
        if len(kw) > 2 and kw in sd:
            return False, r

    # Estate sales: keep but flag
    if sc in _ESTATE_CODES or "ESTATE" in sd:
        return True, "Estate sale — may be below market"

    return True, ""


class CountyParserBase:
    county_name: str = "Unknown"
    state_abbr: str = "XX"
    # Signature: set of uppercase column names that uniquely identify this county's export
    signature: frozenset = frozenset()

    @classmethod
    def matches(cls, columns: list[str]) -> bool:
        upper = {c.upper() for c in columns}
        return cls.signature.issubset(upper)

    @classmethod
    def parse(cls, df: pd.DataFrame, source_file: str) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _sale_id(cls, parid: str, sale_date: str) -> str:
        key = f"{cls.county_name}:{parid}:{sale_date}"
        return hashlib.md5(key.encode()).hexdigest()[:14]


class AlleghenyParser(CountyParserBase):
    """
    Parser for Allegheny County, PA property sales export.
    Source: https://apps.county.allegheny.pa.us/SaleSearch/SaleSearch
    Columns: _id, PARID, FULL_ADDRESS, PROPERTYHOUSENUM, PROPERTYFRACTION,
             PROPERTYADDRESSDIR, PROPERTYADDRESSSTREET, PROPERTYADDRESSSUF,
             PROPERTYADDRESSUNITDESC, PROPERTYUNITNO, PROPERTYCITY,
             PROPERTYSTATE, PROPERTYZIP, SCHOOLCODE, SCHOOLDESC,
             MUNICODE, MUNIDESC, RECORDDATE, SALEDATE, PRICE,
             DEEDBOOK, DEEDPAGE, SALECODE, SALEDESC, INSTRTYP, INSTRTYPDESC
    """

    county_name = "Allegheny, PA"
    state_abbr = "PA"
    signature = frozenset({"PARID", "MUNIDESC", "SCHOOLDESC", "SALECODE", "SALEDESC", "INSTRTYP"})

    @classmethod
    def parse(cls, df: pd.DataFrame, source_file: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip().str.upper()

        def col(name):
            return df[name] if name in df.columns else pd.Series([None] * len(df))

        # ── Build address ────────────────────────────────────────────────
        import math as _math

        def _clean(val) -> str:
            """Return empty string for None, NaN, 'nan', 'none', 'null'."""
            if val is None:
                return ""
            if isinstance(val, float) and _math.isnan(val):
                return ""
            s = str(val).strip()
            return "" if s.lower() in ("nan", "none", "null", "<na>", "nat") else s

        def _build_street(row):
            parts = [
                _clean(row.get("PROPERTYHOUSENUM")),
                _clean(row.get("PROPERTYADDRESSDIR")),
                _clean(row.get("PROPERTYADDRESSSTREET")),
                _clean(row.get("PROPERTYADDRESSSUF")),
            ]
            unit = _clean(row.get("PROPERTYUNITNO"))
            street = " ".join(p for p in parts if p)
            if unit:
                street += f" #{unit}"
            return street.strip()

        street = df.apply(_build_street, axis=1)
        city = col("PROPERTYCITY").apply(_clean).str.title()
        state = col("PROPERTYSTATE").apply(lambda v: _clean(v) or "PA").str.upper()
        zip_ = col("PROPERTYZIP").apply(_clean).str[:5]

        # Drop rows where the street came out entirely empty
        valid_street = street.str.strip().ne("")
        if not valid_street.all():
            n_blank = (~valid_street).sum()
            print(f"    Skipping {n_blank} rows with no street address")
        df = df[valid_street].copy()
        street = street[valid_street]
        city = city[valid_street]
        state = state[valid_street]
        zip_ = zip_[valid_street]

        full_address = (street + ", " + city + ", " + state + " " + zip_).str.strip(", ")

        # ── Sale price ───────────────────────────────────────────────────
        sold_price = pd.to_numeric(col("PRICE").astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")

        # ── Dates ────────────────────────────────────────────────────────
        sold_date = pd.to_datetime(col("SALEDATE"), errors="coerce").dt.date
        record_date = pd.to_datetime(col("RECORDDATE"), errors="coerce").dt.date

        # ── Arm's-length classification ──────────────────────────────────
        sale_codes = col("SALECODE").astype(str).str.strip()
        sale_descs = col("SALEDESC").astype(str).str.strip()
        instr_types = col("INSTRTYP").astype(str).str.strip()

        arms_length_rows = [
            _classify_arms_length(sc, sd, it, pr)
            for sc, sd, it, pr in zip(sale_codes, sale_descs, instr_types, sold_price.fillna(0))
        ]
        is_arms_length = [r[0] for r in arms_length_rows]
        arms_length_flag = [r[1] for r in arms_length_rows]

        # ── Parcel ID ─────────────────────────────────────────────────────
        parid = col("PARID").astype(str).str.strip()

        # ── Sale ID ──────────────────────────────────────────────────────
        sale_id = pd.Series([cls._sale_id(p, str(d)) for p, d in zip(parid, sold_date)])

        # ── County FIPS for Allegheny = 42003 ────────────────────────────
        county_fips = "42003"

        out = pd.DataFrame(
            {
                "sale_id": sale_id,
                "source_county": cls.county_name,
                "source_file": source_file,
                "parid": parid,
                "address": street,
                "city": city,
                "state": state,
                "zip": zip_,
                "full_address": full_address,
                "sold_price": sold_price,
                "sold_date": sold_date,
                "record_date": record_date,
                "list_price": None,
                "sqft": None,
                "beds": None,
                "baths": None,
                "year_built": None,
                "sale_code": sale_codes,
                "sale_desc": col("SALEDESC").astype(str).str.strip(),
                "instr_type": instr_types,
                "instr_type_desc": col("INSTRTYPDESC").astype(str).str.strip()
                if "INSTRTYPDESC" in df.columns
                else None,
                "deed_book": col("DEEDBOOK").astype(str).str.strip(),
                "deed_page": col("DEEDPAGE").astype(str).str.strip(),
                "muni_code": col("MUNICODE").astype(str).str.strip(),
                "muni_desc": col("MUNIDESC").astype(str).str.strip().str.strip(),
                "school_code": col("SCHOOLCODE").astype(str).str.strip(),
                "school_desc": col("SCHOOLDESC").astype(str).str.strip(),
                "is_arms_length": is_arms_length,
                "arms_length_flag": arms_length_flag,
                "lat": None,
                "lon": None,
                "geocode_status": "pending",
                "geocode_source": None,
                "geocode_accuracy": None,
                "tract_fips": None,
                "county_fips": county_fips,
            }
        )

        return out


class GenericSoldParser(CountyParserBase):
    """
    Fallback parser for CSV files that already have standard columns
    (e.g. the original sample_sold.csv with lat/lon included).
    Tries to map common column names to the standard schema.
    """

    county_name = "Unknown"
    signature = frozenset()  # always matches as last resort

    @classmethod
    def matches(cls, columns):
        return True  # fallback — always try this last

    @classmethod
    def parse(cls, df: pd.DataFrame, source_file: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()
        col_rename = {c: _REDFIN_COL_MAP.get(c, c) for c in df.columns}
        df = df.rename(columns=col_rename)

        for c in ["sold_price", "list_price", "sqft", "beds", "baths"]:
            if c in df.columns:
                df[c] = _clean_numeric(df[c])

        if "sold_date" in df.columns:
            df["sold_date"] = pd.to_datetime(df["sold_date"], errors="coerce").dt.date

        if "address" not in df.columns:
            df["address"] = ""
        city = df.get("city", pd.Series([""] * len(df)))
        state = df.get("state", pd.Series([""] * len(df)))
        zip_ = df.get("zip", pd.Series([""] * len(df)))

        df["full_address"] = (
            df["address"].astype(str) + ", " + city.astype(str) + ", " + state.astype(str) + " " + zip_.astype(str)
        ).str.strip(", ")

        # Generic files may already have lat/lon
        if "lat" in df.columns and "lon" in df.columns:
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            geocode_status = df["lat"].notna().map(lambda v: "manual" if v else "pending")
        else:
            df["lat"] = None
            df["lon"] = None
            geocode_status = "pending"

        df["sale_id"] = df.apply(
            lambda r: hashlib.md5(f"generic:{r.get('address', '')}{r.get('sold_date', '')}".encode()).hexdigest()[:14],
            axis=1,
        )

        # Classify arm's-length (best effort with generic data)
        if "sale_code" in df.columns and "sale_desc" in df.columns:
            alc = [
                _classify_arms_length(sc, sd, "", pr)
                for sc, sd, pr in zip(
                    df["sale_code"].fillna(""),
                    df["sale_desc"].fillna(""),
                    df.get("sold_price", pd.Series([0] * len(df))).fillna(0),
                )
            ]
        else:
            price = df.get("sold_price", pd.Series([0] * len(df))).fillna(0)
            alc = [(p > 1, "" if p > 1 else "Nominal price") for p in price]

        schema_cols = [
            "sale_id",
            "source_county",
            "source_file",
            "parid",
            "address",
            "city",
            "state",
            "zip",
            "full_address",
            "sold_price",
            "sold_date",
            "record_date",
            "list_price",
            "sqft",
            "beds",
            "baths",
            "year_built",
            "sale_code",
            "sale_desc",
            "instr_type",
            "instr_type_desc",
            "deed_book",
            "deed_page",
            "muni_code",
            "muni_desc",
            "school_code",
            "school_desc",
            "is_arms_length",
            "arms_length_flag",
            "lat",
            "lon",
            "geocode_status",
            "geocode_source",
            "geocode_accuracy",
            "tract_fips",
            "county_fips",
        ]
        for c in schema_cols:
            if c not in df.columns:
                df[c] = None

        df["source_county"] = cls.county_name
        df["source_file"] = source_file
        df["is_arms_length"] = [r[0] for r in alc]
        df["arms_length_flag"] = [r[1] for r in alc]
        df["geocode_status"] = geocode_status

        return df[schema_cols]


# Registry: listed in priority order; GenericSoldParser must be last
_COUNTY_PARSERS = [AlleghenyParser, GenericSoldParser]


def _detect_parser(columns: list[str]) -> type:
    """Return the first parser whose signature matches the file's columns."""
    upper = [c.upper() for c in columns]
    for parser in _COUNTY_PARSERS[:-1]:  # skip generic
        if parser.matches(upper):
            return parser
    return GenericSoldParser


# Full schema column list — must match sold_homes table exactly
_SOLD_SCHEMA_COLS = [
    "sale_id",
    "source_county",
    "source_file",
    "parid",
    "address",
    "city",
    "state",
    "zip",
    "full_address",
    "lat",
    "lon",
    "geocode_status",
    "geocode_source",
    "geocode_accuracy",
    "sold_price",
    "sold_date",
    "record_date",
    "list_price",
    "sqft",
    "beds",
    "baths",
    "year_built",
    "sale_code",
    "sale_desc",
    "instr_type",
    "instr_type_desc",
    "deed_book",
    "deed_page",
    "muni_code",
    "muni_desc",
    "school_code",
    "school_desc",
    "is_arms_length",
    "arms_length_flag",
    "tract_fips",
    "county_fips",
]


def load_sold_homes(run_geocoding: bool = True, geo_utils=None) -> int:
    """
    Load all sold-homes CSV files from data/sold/.
    Auto-detects the county format and applies the correct parser.
    Runs geocoding (Census Batch → Census Single → Nominatim) for any rows
    without lat/lon, using a DuckDB cache to avoid duplicate API calls.

    Parameters
    ----------
    run_geocoding : If True (default), geocode addresses without coordinates.
    geo_utils     : If provided, also run spatial join for any rows where
                    geocoding returned lat/lon but no tract_fips.
    """
    from services.geocoder import geocode_dataframe

    sold_dir = settings.sold_dir
    csv_files = sorted(sold_dir.rglob("*.csv"))
    if not csv_files:
        print("  No sold homes CSV files found under data/sold/ (including subfolders)")
        return 0

    all_rows = []
    for path in csv_files:
        print(f"  Loading sold: {path.name}")
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            raw = pd.read_csv(path, encoding="latin-1", low_memory=False)

        parser = _detect_parser(list(raw.columns))
        print(f"    Parser: {parser.__name__} ({parser.county_name})")

        try:
            parsed = parser.parse(raw, source_file=path.name)
        except Exception as e:
            print(f"    Parse error: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Ensure all schema columns exist
        for col in _SOLD_SCHEMA_COLS:
            if col not in parsed.columns:
                parsed[col] = None

        all_rows.append(parsed[_SOLD_SCHEMA_COLS])
        n_al = int((parsed["is_arms_length"] == True).sum())
        n_nal = int((parsed["is_arms_length"] == False).sum())
        print(f"    {len(parsed)} records: {n_al} arm's-length, {n_nal} non-arm's-length flagged")

    if not all_rows:
        return 0

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=["sale_id"])

    # ── Geocoding ─────────────────────────────────────────────────────────
    needs_geo = combined["lat"].isna() | combined["lon"].isna()
    if run_geocoding and needs_geo.any():
        print(f"  Geocoding {needs_geo.sum()} addresses …")
        combined = geocode_dataframe(
            combined,
            street_col="address",
            city_col="city",
            state_col="state",
            zip_col="zip",
            full_addr_col="full_address",
            verbose=True,
        )

    # ── Spatial join for any still missing tract_fips ─────────────────────
    needs_tract = combined["tract_fips"].isna() & combined["lat"].notna()
    if geo_utils and needs_tract.any():
        print(f"  Resolving {needs_tract.sum()} tract FIPS via spatial join …")
        combined = geo_utils.assign_tract_fips(combined)

    store.upsert_df("sold_homes", combined)

    n_geocoded = int((combined["geocode_status"] == "success").sum())
    n_pending = int(combined["geocode_status"].isna().sum() + (combined["geocode_status"] == "pending").sum())
    print(f"  ✓ {len(combined)} sold homes loaded ({n_geocoded} geocoded, {n_pending} pending)")
    return len(combined)

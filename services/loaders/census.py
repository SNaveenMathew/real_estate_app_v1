"""`Census (tract)` loaders, moved out of services/data_loader.py.

Behaviour is unchanged: the facade in services/data_loader.py re-exports every name here, so
`from services.data_loader import ...` (and scripts that call it) keep working.
"""
from __future__ import annotations

"""
Data loader: reads raw files and populates DuckDB tables.
Run via setup_data.py — only needed when data files change.
"""

import pandas as pd


from config import settings
import db.duckdb_store as store




__all__ = ['load_census_tracts', '_normalize_msa_name', '_first_city_state', '_build_cbsa_lookup', '_all_words', '_KNOWN_RENAMES', '_match_msa_to_cbsa', '_detect_msa_format', '_load_msa_wide', '_load_msa_long', 'load_census_msa', 'repair_msa_codes']

# ── Census (tract) ───────────────────────────────────────────────────────────

def load_census_tracts() -> int:
    """
    Load census tract populations from DECENNIALPL2020.P1 data file.

    Accepts two formats from data.census.gov:

    Standard download (DECENNIALPL2020.P1-Data.csv):
        Row 0: GEO_ID, NAME, P1_001N, P1_002N, ...   ← column names
        Row 1: Geography, Geographic Area Name, !!Total:, ...  ← descriptions (skip)
        Row 2+: 1400000US01001020100, "Census Tract...", 1775, ...

    Legacy format:
        Row 0: GEO_ID, NAME, P1_001N  (or P1_001)
        Row 1+: data immediately

    Place file at: data/census/DECENNIALPL2020_P1_tract.csv
    Or any DECENNIALPL2020*.csv in data/census/ that contains tract GEO_IDs.
    """
    # Accept the configured path or any matching file in the census dir
    census_dir = settings.census_tract_csv.parent
    candidates = (
        [settings.census_tract_csv] if settings.census_tract_csv.exists() else []
    ) + sorted(census_dir.glob("DECENNIALPL2020*.csv"))
    # Exclude the MSA file (wide-format, no GEO_IDs starting with 1400000US)
    seen: set = set()
    candidates = [p for p in candidates
                  if not (str(p) in seen or seen.add(str(p)))]

    path = None
    for c in candidates:
        # Quick sniff: does this file look like a tract file?
        try:
            sniff = pd.read_csv(c, encoding="utf-8-sig", nrows=3, dtype=str)
            # Check any cell for a tract GEO_ID prefix
            flat = " ".join(sniff.values.flatten().astype(str))
            if "1400000US" in flat:
                path = c
                break
        except Exception:
            continue

    if path is None:
        print(f"  Census tract file not found in: {census_dir}")
        print("  Download DECENNIALPL2020.P1-Data.csv from data.census.gov")
        print("  Geography filter: Census Tracts → All United States")
        print("  Save as: data/census/DECENNIALPL2020_P1_tract.csv")
        return 0

    print(f"  Loading census tract populations: {path.name}")

    # Read with all-string dtype to avoid numeric parse issues in header rows
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False)

    # Detect and drop the description row (row 1 in the standard download).
    # It has values like "Geography", "Geographic Area Name", "!!Total:" — none
    # of which start with "1400000US".
    geo_col = next((c for c in raw.columns if "GEO_ID" in c.upper().replace(" ", "_")), None)
    if geo_col is None:
        # BOM sometimes fuses with the first column name
        geo_col = raw.columns[0]

    # Drop rows that are not actual tract rows
    data = raw[raw[geo_col].str.startswith("1400000US", na=False)].copy()

    if data.empty:
        print(f"  Error: no rows starting with '1400000US' found in {path.name}")
        print(f"  Columns: {list(raw.columns[:5])}")
        return 0

    # Population column — P1_001N (new) or P1_001 (legacy)
    pop_col = next(
        (c for c in data.columns if c.upper().replace(" ", "") in ("P1_001N", "P1_001")),
        None,
    )
    if pop_col is None:
        print(f"  Error: population column (P1_001N or P1_001) not found.")
        print(f"  Columns: {list(data.columns[:10])}")
        return 0

    name_col = next((c for c in data.columns if c.upper() == "NAME"), None)

    out = pd.DataFrame({
        "tract_fips": data[geo_col].str.replace("1400000US", "", regex=False).str.zfill(11),
        "geo_id":     data[geo_col],
        "name":       data[name_col] if name_col else None,
        "population": pd.to_numeric(data[pop_col], errors="coerce").astype("Int64"),
    })
    out = out.dropna(subset=["tract_fips"])
    out = out.drop_duplicates(subset=["tract_fips"])

    conn = store.get_conn()
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM census_tracts")
        store.upsert_df("census_tracts", out)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    print(f"  ✓ Loaded {len(out):,} census tract populations")
    return len(out)


def _normalize_msa_name(name: str) -> str:
    """
    Normalize an MSA name to a canonical lowercase ASCII string for matching.

    Handles:
    - Trailing 'Metro Area' / 'Micro Area' suffix (census wide-format files only)
    - Double dashes: Nashville-Davidson--Murfreesboro → Nashville-Davidson-Murfreesboro
    - Accented characters: Cañon→Canon, Mayagüez→Mayaguez, Bayamón→Bayamon
    """
    import re as _re
    import unicodedata as _ud
    name = str(name).strip()
    # Strip trailing area type (wide-format files only)
    name = _re.sub(r'\s+(Metro|Micro)\s+Area$', '', name, flags=_re.IGNORECASE)
    # Collapse multiple consecutive dashes
    name = _re.sub(r'-{2,}', '-', name)
    # Normalize whitespace
    name = _re.sub(r'\s+', ' ', name).strip()
    # Decompose accented chars (NFD) and drop the combining marks (category Mn)
    # e.g. ñ → n + ̃ → n,  ü → u + ̈ → u,  é → e + ́ → e
    decomposed = _ud.normalize('NFD', name)
    return ''.join(c for c in decomposed if _ud.category(c) != 'Mn').lower()


def _first_city_state(normalized_name: str):
    """
    Extract (first_city, first_state) from a normalized MSA name.

    'bakersfield-delano, ca' → ('bakersfield', 'ca')
    'salt lake city-murray, ut' → ('salt lake city', 'ut')
    'new york-newark-jersey city, ny-nj-pa' → ('new york', 'ny')

    Used as a fallback key when MSA names differ between Census and CBSA vintages
    (e.g. 2020 census names vs 2023 CBSA delineation names).
    """
    import re as _re
    # Split on last ", STATE" pattern
    m = _re.match(r'^(.*),\s*([a-z]{2}(?:-[a-z]{2})*)$', normalized_name)
    if not m:
        return normalized_name, ''
    cities_part = m.group(1)
    states_part = m.group(2)
    first_city  = cities_part.split('-')[0].strip()
    first_state = states_part.split('-')[0]
    return first_city, first_state


def _build_cbsa_lookup(cbsa_df) -> dict:
    """
    Build a multi-key lookup dict from cbsa_counties data.
    Returns {key: cbsa_code} with four tiers of keys:
      1. Full normalized name          'bakersfield, ca'
      2. First-city + state tuple      ('bakersfield', 'ca')
      3. First-city only string        'bakersfield-ca'   (last resort, single-state)
      4. '_raw_titles' key             {norm_title: code} for fuzzy Jaccard matching
    """
    lookup: dict = {}
    raw_titles: dict = {}
    for _, row in cbsa_df.iterrows():
        code  = row["cbsa_code"]
        title = row["cbsa_title"]
        norm  = _normalize_msa_name(title)
        city, state = _first_city_state(norm)

        # Tier 1
        if norm not in lookup:
            lookup[norm] = code
        # Tier 2
        pair_key = (city, state)
        if pair_key not in lookup:
            lookup[pair_key] = code
        # Tier 3
        short_key = f"{city}-{state}"
        if short_key not in lookup:
            lookup[short_key] = code
        # Tier 4 raw index
        if norm not in raw_titles:
            raw_titles[norm] = code

    lookup["_raw_titles"] = raw_titles

    return lookup


def _all_words(norm: str) -> set:
    """Significant words (>2 chars) for fuzzy scoring."""
    import re as _re
    return {w for w in _re.split(r'[\s,\-]+', norm) if len(w) > 2}


# Known 2020→2023 OMB name changes where the first city also changed.
# These can't be matched by any string similarity strategy.
# Format: {normalize(2020_census_name): normalize(2023_cbsa_title)}
_KNOWN_RENAMES = {
    # Military base renamed from Fort Polk to Fort Johnson in 2023
    "fort polk south, la":              "fort johnson, la",
    # Poughkeepsie MSA expanded to include Port Jervis in 2023
    "poughkeepsie-newburgh-middletown, ny": "poughkeepsie-newburgh-middletown-port jervis, ny",
    # Dayton MSA renamed to include Kettering in 2023
    "dayton, oh":                       "dayton-kettering, oh",
    # Some California MSAs renamed
    "napa, ca":                         "napa, ca",
}


def _match_msa_to_cbsa(name: str, lookup: dict) -> str | None:
    """
    Try to match an MSA name to a CBSA code using a four-tier lookup.

    Tier 1 — full normalized name (exact after accent stripping + dash collapse)
    Tier 2 — (first_city, first_state) tuple  [handles added secondary cities]
    Tier 3 — city-state string (single-state MSAs only)
    Tier 4 — fuzzy Jaccard similarity ≥ 0.5 within same state  [vintage renames]
    Returns the CBSA code string or None if no match found.
    """
    norm = _normalize_msa_name(name)
    city, state = _first_city_state(norm)

    # Tiers 1-3: exact and near-exact
    if norm in lookup:
        return lookup[norm]
    # Tier 2: (first_city, first_state) tuple
    if (city, state) in lookup:
        return lookup[(city, state)]
    # Tier 3: city-state string (only when state is unambiguous single state)
    if '-' not in state:
        short_key = f"{city}-{state}"
        if short_key in lookup:
            return lookup[short_key]
    
    # Tier 4: fuzzy — find best Jaccard match within same state
    # lookup contains both string keys and tuple keys; filter to tuples (city, state)
    # and then find the closest cbsa code by comparing against a raw-title lookup
    # stored under the special key prefix "_raw"
    raw_key = "_raw_titles"
    if raw_key not in lookup:
        return None   # raw titles not available

    norm_words = _all_words(norm)
    best_code  = None
    best_score = 0.0

    for cbsa_norm, code in lookup[raw_key].items():
        c_city, c_state = _first_city_state(cbsa_norm)
        if c_state != state:           # must be same state(s)
            continue
        score = len(norm_words & _all_words(cbsa_norm)) / max(
            len(norm_words | _all_words(cbsa_norm)), 1
        )
        if score > best_score:
            best_score = score
            best_code  = code

    if best_score >= 0.5:
        return best_code

    # Tier 5: explicit known 2020→2023 OMB renames
    mapped_norm = _KNOWN_RENAMES.get(norm)
    if mapped_norm and mapped_norm in lookup:
        return lookup[mapped_norm]
    if mapped_norm:
        mapped_city, mapped_state = _first_city_state(mapped_norm)
        if (mapped_city, mapped_state) in lookup:
            return lookup[(mapped_city, mapped_state)]
    
    return None


def _detect_msa_format(df: pd.DataFrame) -> str:
    """
    Detect whether a census MSA CSV is in wide or long format.

    Wide  — MSA names are column headers; first column is 'Label (Grouping)'
    Long  — GEO_ID is a column; each row is one MSA
    """
    first_col = str(df.columns[0]).strip()
    if "label" in first_col.lower() or "grouping" in first_col.lower():
        return "wide"
    cols_upper = [c.upper() for c in df.columns]
    if any("GEO_ID" in c or "GEO ID" in c for c in cols_upper):
        return "long"
    # Fallback: if many columns look like city names they're MSA headers
    msa_like = sum(1 for c in df.columns if ", " in c and ("Area" in c or "Metro" in c))
    return "wide" if msa_like > 5 else "long"


def _load_msa_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse wide-format MSA census file.

    Row 0 header: ['Label (Grouping)', 'Aberdeen, SD Micro Area', ...]
    Row 1 data:   ['Total:', '42,287', ...]
    Rows 2+:      demographic sub-rows (ignored)
    """
    # Find the "Total:" row — it's always the first data row
    total_mask = df.iloc[:, 0].astype(str).str.strip().str.lower() == "total:"
    if not total_mask.any():
        # Try "Total" without colon
        total_mask = df.iloc[:, 0].astype(str).str.strip().str.lower().str.startswith("total")
    if not total_mask.any():
        raise ValueError("Could not find 'Total:' row in wide-format MSA file")

    total_row = df[total_mask].iloc[0]
    msa_names = list(df.columns[1:])          # skip 'Label (Grouping)' column
    pop_values = list(total_row.iloc[1:])     # skip the 'Total:' cell

    records = []
    for name, pop_str in zip(msa_names, pop_values):
        try:
            pop = int(str(pop_str).replace(",", "").strip())
        except (ValueError, AttributeError):
            pop = None
        records.append({"name": name.strip(), "population": pop})

    return pd.DataFrame(records)


def _load_msa_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse long-format MSA census file (original expected format).
    Each row is one MSA; columns include GEO_ID and a population column.
    """
    cols_upper = {c: c.upper() for c in df.columns}
    geo_col  = next((c for c, u in cols_upper.items() if "GEO_ID" in u or "GEO ID" in u), None)
    name_col = next((c for c, u in cols_upper.items() if u == "NAME"), None)
    pop_col  = next((c for c, u in cols_upper.items() if "P1_001" in u), None)

    if not geo_col or not pop_col:
        raise ValueError(
            f"Long-format MSA file missing GEO_ID or P1_001 column. "
            f"Found: {list(df.columns)}"
        )

    def _extract_code(geo_id: str) -> str:
        parts = str(geo_id).split("US")
        return parts[-1] if len(parts) > 1 else geo_id

    records = []
    for _, row in df.iterrows():
        geo_id = str(row[geo_col])
        if not (geo_id.startswith("31") or geo_id.startswith("33") or "US" in geo_id):
            continue
        records.append({
            "name":       str(row[name_col]).strip() if name_col else None,
            "msa_code":   _extract_code(geo_id).zfill(5),
            "geo_id":     geo_id,
            "population": pd.to_numeric(str(row[pop_col]).replace(",", ""), errors="coerce"),
        })
    return pd.DataFrame(records)


def load_census_msa() -> int:
    """
    Load MSA population data from a Census DECENNIALPL2020.P1 file.

    Accepts both file layouts produced by data.census.gov:

    WIDE FORMAT (MSAs as columns) — the typical download when you select
    all MSAs at once and export with geography as columns:
        Column headers: 'Label (Grouping)', 'Aberdeen, SD Micro Area', ...
        Row 1:          'Total:', '42,287', ...
        File:           DECENNIALPL2020_P1_msa.csv  (or any CSV in data/census/
                        that matches DECENNIALPL2020_P1*.csv)

    LONG FORMAT (MSAs as rows) — download via the advanced table builder:
        Columns:  GEO_ID, NAME, P1_001N (total population), ...
        Each row: one MSA

    After loading, msa_code is resolved by matching normalised MSA names
    against cbsa_title in the cbsa_counties table (if already loaded).
    The match handles:
        Census: "Nashville-Davidson--Murfreesboro--Franklin, TN Metro Area"
        CBSA:   "Nashville-Davidson--Murfreesboro--Franklin, TN"
    Both reduce to: "nashville-davidson-murfreesboro-franklin, tn"
    """
    # ── Find the file ─────────────────────────────────────────────────────
    census_dir = settings.census_msa_csv.parent
    candidates = (
        [settings.census_msa_csv] if settings.census_msa_csv.exists() else []
    ) + sorted(census_dir.glob("DECENNIALPL2020_P1*.csv"))

    seen = set()
    candidates = [p for p in candidates if not (str(p) in seen or seen.add(str(p)))]

    if not candidates:
        print(f"  Census MSA file not found in: {census_dir}")
        print("  Expected: DECENNIALPL2020_P1_msa.csv (or any DECENNIALPL2020_P1*.csv)")
        print("  Download table DECENNIALPL2020.P1 from data.census.gov")
        print("  → Geography: Metropolitan/Micropolitan Statistical Areas → All")
        return 0

    path = candidates[0]
    print(f"  Loading MSA populations: {path.name}")
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype=str)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return 0

    # ── Detect and parse ──────────────────────────────────────────────────
    fmt = _detect_msa_format(raw)
    print(f"  Detected format: {fmt}")

    try:
        if fmt == "wide":
            parsed = _load_msa_wide(raw)
        else:
            parsed = _load_msa_long(raw)
    except Exception as e:
        print(f"  Parse error: {e}")
        return 0

    if parsed.empty:
        print("  No records extracted from file")
        return 0

    # ── Ensure required columns ───────────────────────────────────────────
    for col in ["msa_code", "geo_id"]:
        if col not in parsed.columns:
            parsed[col] = None

    parsed["population"] = pd.to_numeric(
        parsed["population"].astype(str).str.replace(",", ""), errors="coerce"
    ).astype("Int64")

    parsed = parsed.dropna(subset=["name"])
    parsed = parsed[parsed["name"].str.strip() != ""]
    parsed = parsed.drop_duplicates(subset=["name"])

    print(f"  Parsed {len(parsed)} MSAs")

    # ── Match CBSA codes by name (multi-strategy) ─────────────────────────
    try:
        cbsa_df = store.query("""
            SELECT DISTINCT cbsa_code, cbsa_title
            FROM cbsa_counties
            WHERE cbsa_title IS NOT NULL
        """)
    except Exception:
        cbsa_df = pd.DataFrame(columns=["cbsa_code", "cbsa_title"])

    if not cbsa_df.empty:
        # Build normalised-name → code lookup
        lookup = _build_cbsa_lookup(cbsa_df)

        matched = 0
        codes = []
        match_tiers = {"exact": 0, "city_state": 0, "city_only": 0, "none": 0}
        for name in parsed["name"]:
            norm = _normalize_msa_name(name)
            city, state = _first_city_state(norm)
            # Try tier by tier and record which matched
            if norm in lookup:
                codes.append(lookup[norm])
                match_tiers["exact"] += 1
                matched += 1
            elif (city, state) in lookup:
                codes.append(lookup[(city, state)])
                match_tiers["city_state"] += 1
                matched += 1
            elif '-' not in state and f"{city}-{state}" in lookup:
                codes.append(lookup[f"{city}-{state}"])
                match_tiers["city_only"] += 1
                matched += 1
            else:
                codes.append(None)
                match_tiers["none"] += 1

        parsed["msa_code"] = codes
        print(f"  CBSA code matched: {matched}/{len(parsed)} MSAs")

        print(f"    Exact: {match_tiers['exact']}  "
              f"City+state fallback: {match_tiers['city_state']}  "
              f"City-only fallback: {match_tiers['city_only']}  "
              f"Unmatched: {match_tiers['none']}")
        if match_tiers["none"]:
            unmatched = parsed[parsed["msa_code"].isna()]["name"].tolist()
            print(f"  ⚠ Still unmatched (likely not in CBSA delineation file): "
                  f"{unmatched[:5]}")
    else:
        print("  ⚠ cbsa_counties not loaded yet — msa_code will be NULL")
        print("    Load cbsa_counties first, then re-run: "
              "python setup_data.py --only census")

    # Rows without a msa_code can still be useful for name-based queries.
    # Use a hash of the name as a fallback key so the table has no NULL PKs.
    import hashlib as _hl
    parsed["msa_code"] = parsed["msa_code"].where(
        parsed["msa_code"].notna(),
        parsed["name"].apply(lambda n: "X" + _hl.md5(n.encode()).hexdigest()[:7])
    )
    parsed["geo_id"] = parsed["geo_id"].fillna("")
    out = parsed[["msa_code", "geo_id", "name", "population"]]

    store.upsert_df("census_msa", out)
    print(f"  ✓ Loaded {len(out)} MSA populations into census_msa")
    return len(out)


def repair_msa_codes() -> int:
    """
    Fix X-prefixed fallback msa_codes in census_msa using multi-strategy matching.

    Strategy order:
      1. Full normalized name (handles accents, double-dashes)
      2. First-city + state (handles vintage renames like Bakersfield→Bakersfield-Delano)
      3. First-city only   (last resort for single-state MSAs)

    Safe to call on an existing database — only updates rows with X-codes.
    Returns the number of rows repaired.
    """
    # Load current X-coded rows
    x_rows = store.query(
        "SELECT msa_code, name FROM census_msa WHERE msa_code LIKE 'X%'"
    )
    if x_rows.empty:
        print("  No X-coded msa_codes found — nothing to repair.")
        return 0

    print(f"  Repairing {len(x_rows)} X-coded msa_codes using multi-strategy matching...")

    # Build CBSA lookup
    try:
        cbsa_df = store.query(
            "SELECT DISTINCT cbsa_code, cbsa_title FROM cbsa_counties WHERE cbsa_title IS NOT NULL"
        )
    except Exception as e:
        print(f"  Error reading cbsa_counties: {e}")
        return 0

    if cbsa_df.empty:
        print("  cbsa_counties is empty — cannot repair. Load it first.")
        return 0

    lookup = _build_cbsa_lookup(cbsa_df)

    repaired = 0
    still_unmatched = []
    conn = store.get_conn()
    for _, row in x_rows.iterrows():
        old_code = row["msa_code"]
        name     = row["name"]
        new_code = _match_msa_to_cbsa(name, lookup)
        if new_code:
            conn.execute(
                "UPDATE census_msa SET msa_code = ? WHERE msa_code = ?",
                [new_code, old_code]
            )
            repaired += 1
        else:
            still_unmatched.append(name)

    still_x = int(store.query(
        "SELECT COUNT(*) as n FROM census_msa WHERE msa_code LIKE 'X%'"
    ).iloc[0]["n"])
    print(f"  ✓ Repaired {repaired} rows. Remaining X-codes: {still_x}")
    if still_unmatched:
        print(f"  These {len(still_unmatched)} MSAs have no matching CBSA code "
              f"(likely genuinely absent from your delineation file vintage):")
        for name in still_unmatched[:10]:
            print(f"    • {name}")
        if len(still_unmatched) > 10:
            print(f"    ... and {len(still_unmatched) - 10} more")
        print()
        print("  These MSAs cannot be joined to NRI data, but they don't affect")
        print("  the flood risk query (they're typically small Micro areas).")
    
    return repaired

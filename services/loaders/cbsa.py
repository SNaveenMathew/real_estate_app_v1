"""`CBSA crosswalk` loaders, moved out of services/data_loader.py.

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




__all__ = ['load_cbsa_crosswalk']

# ── CBSA crosswalk ───────────────────────────────────────────────────────────

def load_cbsa_crosswalk() -> int:
    """
    Load the MSA→county crosswalk from Census Bureau delineation file.
    
    Accepts any of these formats (auto-detected by extension):
      list1_2020.csv / list1_2023.csv   — CSV, skiprows=2
      list1_2020.xls / list1_2023.xlsx  — Excel, skiprows=2
    
    Download: https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
    Save to: data/census/   (any filename ending in .csv/.xls/.xlsx is accepted)
    """
        # Accept any delineation file in the census dir regardless of exact name
    census_dir = settings.cbsa_xlsx.parent
    candidates = (
        list(census_dir.glob("list*.xlsx")) +
        list(census_dir.glob("list*.xls")) +
        list(census_dir.glob("list*.csv")) +
        ([settings.cbsa_xlsx] if settings.cbsa_xlsx.exists() else [])
    )
    # Deduplicate while preserving order
    seen = set()
    candidates = [p for p in candidates if not (str(p) in seen or seen.add(str(p)))]

    if not candidates:
        print(f"  CBSA crosswalk not found in: {census_dir}")
        print("  Download delineation file from Census Bureau and save to data/census/")
        return 0
    
    path = candidates[0]
    print(f"  Loading CBSA crosswalk: {path.name}")
    
    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            # Census delineation Excel files have 2 header rows before data
            df = pd.read_excel(path, skiprows=2, dtype=str)
        else:
            try:
                df = pd.read_csv(path, encoding="utf-8-sig", skiprows=2,
                                 dtype=str, low_memory=False)
            except Exception:
                df = pd.read_csv(path, encoding="latin-1", skiprows=2,
                                 dtype=str, low_memory=False)
    except Exception as e:
        print(f"  Error reading {path.name}: {e}")
        return 0

    df.columns = df.columns.str.strip()
    
    # Drop completely empty rows (common at end of Census Excel files)
    df = df.dropna(how="all")

    # Column detection — Census changes column names between vintages.
    # Print columns to help debug if mapping fails.
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "cbsa code":                               col_map[c] = "cbsa_code"
        elif cl == "cbsa title":                            col_map[c] = "cbsa_title"
        elif "metropolitan" in cl and "micro" in cl:        col_map[c] = "msa_type"
        elif cl in ("fips state code", "fips state"):       col_map[c] = "state_fips"
        elif cl in ("fips county code", "fips county"):     col_map[c] = "county_fips"
        elif "county/county" in cl or cl == "county name":  col_map[c] = "county_name"
        elif cl == "state name":                            col_map[c] = "state_name"
    df = df.rename(columns=col_map)

    needed = ["cbsa_code", "state_fips", "county_fips"]
    if not all(c in df.columns for c in needed):
        print(f"  Could not find required columns. Mapped columns: {list(df.columns)}")
        print("  Expected: 'CBSA Code', 'FIPS State Code', 'FIPS County Code'")
        print("  (Column names vary by Census vintage — check the file header)")
        return 0

    df = df.dropna(subset=needed)
    # Remove footnote rows where cbsa_code is not numeric
    df = df[df["cbsa_code"].str.strip().str.match(r"^\d+$", na=False)]
    
    df["cbsa_code"]   = df["cbsa_code"].astype(str).str.strip().str.zfill(5)
    df["state_fips"]  = df["state_fips"].astype(str).str.strip().str.zfill(2)
    df["county_fips"] = df["county_fips"].astype(str).str.strip().str.zfill(3)

    for col in ["cbsa_title", "msa_type", "county_name", "state_name"]:
        if col not in df.columns:
            df[col] = None

    out = df[["cbsa_code", "cbsa_title", "msa_type",
              "state_fips", "county_fips", "county_name", "state_name"]]
    store.upsert_df("cbsa_counties", out)
    print(f"  ✓ Loaded {len(out)} CBSA→county mappings")
    return len(out)

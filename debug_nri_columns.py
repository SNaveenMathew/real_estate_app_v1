"""
debug_nri_columns.py — Print the actual column names in the NRI shapefile
and show which ones made it into nri_tracts vs which are NULL.

Usage:
    python debug_nri_columns.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
import db.duckdb_store as store

SEP = "─" * 70

def run():
    print(SEP)
    print("NRI COLUMN DIAGNOSTIC")
    print(SEP)

    # ── 1. What's in nri_tracts right now ────────────────────────────────
    print("\n[1] nri_tracts — NULL counts for all hazard columns")
    hazards = ["avln","cfld","cwav","drgt","erqk","hail","hwav","hrcn",
               "istm","lnds","ltng","rfld","swnd","trnd","tsun","vlcn","wfir","wntw"]

    rows = []
    for hz in hazards:
        col = f"{hz}_risks"
        try:
            df = store.query(f"""
                SELECT COUNT(*) as total,
                       COUNT({col}) as non_null,
                       AVG({col}) as avg_val
                FROM nri_tracts
            """)
            total    = int(df.iloc[0]["total"])
            non_null = int(df.iloc[0]["non_null"])
            avg_val  = df.iloc[0]["avg_val"]
            rows.append((col, total, non_null, f"{avg_val:.2f}" if avg_val and avg_val==avg_val else "NULL"))
        except Exception as e:
            rows.append((col, 0, 0, f"ERROR: {e}"))

    print(f"  {'Column':<20} {'Total':>8} {'Non-NULL':>9} {'Avg':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*9} {'-'*10}")
    all_null = True
    for col, total, non_null, avg in rows:
        flag = " ← ALL NULL" if non_null == 0 else ""
        print(f"  {col:<20} {total:>8,} {non_null:>9,} {avg:>10}{flag}")
        if non_null > 0:
            all_null = False

    if all_null:
        print("\n  !! ALL hazard columns are NULL — NRI was loaded without hazard data")

    # ── 2. What columns does the shapefile actually have ─────────────────
    print("\n[2] NRI shapefile — actual DBF column names")
    shp_path = settings.nri_shp
    if not shp_path.exists():
        print(f"  Shapefile not found: {shp_path}")
        print("  Cannot inspect column names directly.")
    else:
        try:
            import geopandas as gpd
            print(f"  Reading column names from: {shp_path.name}")
            print("  (reading only 1 row to get schema — fast)")
            gdf = gpd.read_file(shp_path, rows=1)
            cols = [c for c in gdf.columns if c != "geometry"]
            print(f"\n  Found {len(cols)} columns:")

            # Group by likely category
            risk_cols  = [c for c in cols if "RISK" in c.upper()]
            hazard_cols= [c for c in cols if any(
                hz.upper() in c.upper() for hz in
                ["AVLN","CFLD","CWAV","DRGT","ERQK","HAIL","HWAV","HRCN",
                 "ISTM","LNDS","LTNG","RFLD","SWND","TRND","TSUN","VLCN","WFIR","WNTW"]
            )]
            other_cols = [c for c in cols if c not in risk_cols and c not in hazard_cols]

            print(f"\n  Composite risk columns ({len(risk_cols)}):")
            for c in sorted(risk_cols):
                expected = "RFLD_RISKS" in c.upper() or "RISK_SCORE" in c.upper()
                print(f"    {c}")

            print(f"\n  Hazard columns ({len(hazard_cols)}) — sample:")
            rfld_cols = [c for c in hazard_cols if "RFLD" in c.upper()]
            print(f"    RFLD variants: {rfld_cols}")
            trnd_cols = [c for c in hazard_cols if "TRND" in c.upper()]
            print(f"    TRND variants: {trnd_cols}")
            print(f"    (first 10 hazard cols): {sorted(hazard_cols)[:10]}")

            # Check if RFLD_RISKS exists exactly
            if "RFLD_RISKS" in [c.upper() for c in cols]:
                print("\n  ✓ RFLD_RISKS column exists (exact match)")
            else:
                close = [c for c in cols if "RFLD" in c.upper()]
                print(f"\n  ✗ RFLD_RISKS not found exactly. Close matches: {close}")
                print("  → The _build_nri_out function looks for RFLD_RISKS but the")
                print("    shapefile uses a different name. This is why flood data is NULL.")

        except ImportError:
            print("  geopandas not available in this environment")
        except Exception as e:
            print(f"  Error reading shapefile: {e}")

    # ── 3. Quick SQL fix check ────────────────────────────────────────────
    print("\n[3] Check if county FIPS join works (ignoring NULL rfld_risks)")
    try:
        df = store.query("""
            SELECT COUNT(DISTINCT cb.cbsa_code) as matching_cbsas
            FROM cbsa_counties cb
            JOIN nri_tracts n ON (cb.state_fips || cb.county_fips) = n.county_fips
        """)
        n = int(df.iloc[0]["matching_cbsas"])
        print(f"  cbsa↔nri join WITHOUT hazard filter: {n} matching CBSAs")
        if n > 0:
            print("  ✓ The FIPS join itself works — the problem is NULL hazard values in nri_tracts")
            print("  Fix: reload NRI data after fixing the column name mapping")
        else:
            print("  ✗ FIPS join also fails — different problem (key format mismatch)")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n{SEP}")
    print("ACTION: paste the output above and we'll fix the column mapping.")
    print(SEP)


if __name__ == "__main__":
    run()

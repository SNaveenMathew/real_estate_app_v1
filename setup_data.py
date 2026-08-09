"""
setup_data.py — Run this ONCE (or whenever data files change) to populate DuckDB.

Usage:
    python setup_data.py                    # load everything
    python setup_data.py --only redfin      # load only Redfin
    python setup_data.py --only nri         # load only NRI
    python setup_data.py --only census      # load only Census
    python setup_data.py --only sold        # load only Sold
    python setup_data.py --resolve-tracts   # re-run tract FIPS resolution
"""
import sys
import time
import argparse
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
import db.duckdb_store as store
from services import data_loader
from services import geo_utils


def banner(text: str):
    print(f"\n{'─' * 55}")
    print(f"  {text}")
    print(f"{'─' * 55}")


def run_all(only: str = None, resolve_tracts: bool = False,
            no_geocoding: bool = False):
    t0 = time.time()
    print("\n🏠 Real Estate Intelligence — Data Setup")
    print(f"   DuckDB: {settings.duckdb_path}")
    print(f"   Chroma: {settings.chroma_dir}")

    # Ensure directories exist
    for d in [settings.data_dir / "nri", settings.data_dir / "census",
              settings.data_dir / "redfin", settings.data_dir / "sold",
              settings.data_dir / "shapefiles", settings.uploads_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Determine geo_utils backend and report to the user
    use_geo = geo_utils if resolve_tracts else None
    if resolve_tracts:
        banner("Geo utilities — tract FIPS resolution")
        from services.geo_utils import geometry_source
        src = geometry_source()
        print(f"  Active geometry source: {src}")
        if "NRI" in src:
            print("  ✓ Using NRI shapefile geometry (fastest — no extra download)")
        elif "TIGER" in src:
            print("  ✓ Using TIGER/Line shapefiles")
        else:
            print("  ⚠ Falling back to Census Geocoder API (slow — add NRI shapefile or TIGER files)")

    # ── NRI ──────────────────────────────────────────────────────────────
    if only in (None, "nri"):
        banner("FEMA National Risk Index (census tract level)")
        if settings.nri_shp.exists():
            print(f"  Found shapefile: {settings.nri_shp.name}")
        elif settings.nri_csv.exists():
            print(f"  Found CSV: {settings.nri_csv.name}")
            print("  Tip: The shapefile is preferred — it also provides tract geometries")
            print("  for fast spatial join, eliminating the need for TIGER/Line files.")
        else:
            print(f"  ✗ NRI data not found in data/nri/")
            print("  Download from: https://www.fema.gov/about/openfema/data-sets/national-risk-index-data")
            print("  → Click 'NRI Census Tract Shapefile' → extract .zip into data/nri/")
            print("  → Or: 'NRI Census Tract CSV' → place .csv in data/nri/")
        n = data_loader.load_nri()
        if n:
            print(f"  ✓ {n:,} tracts loaded")
            # If shapefile was just loaded, the geometry cache now exists —
            # make it available for tract resolution in subsequent steps
            if settings.nri_shp.exists():
                geo_utils._load_nri_geometry.cache_clear()
                use_geo = geo_utils  # always resolve tracts when shapefile is present

    # ── Census (CBSA crosswalk must come first so MSA name→code matching works) ──
    if only in (None, "census"):
        banner("CBSA County Crosswalk")
        census_dir = settings.cbsa_xlsx.parent
        cbsa_candidates = (
            list(census_dir.glob("list*.xlsx")) +
            list(census_dir.glob("list*.xls")) +
            list(census_dir.glob("list*.csv"))
        )
        if not cbsa_candidates:
            print(f"  ⚠ No CBSA delineation file found in: {census_dir}")
            print("  Download: https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html")
            print("  → Save any list*.xlsx / list*.csv into data/census/")
        else:
            print(f"  Found: {cbsa_candidates[0].name}")
            n = data_loader.load_cbsa_crosswalk()
            if n:
                print(f"  ✓ {n:,} CBSA→county mappings loaded")
        
        banner("Census — Tract Populations (DECENNIALPL2020.P1)")
        if not settings.census_tract_csv.exists():
            print(f"  ⚠ File not found: {settings.census_tract_csv}")
            print("  Download from data.census.gov → Table DECENNIALPL2020.P1")
            print("  Filter: Geography = Census Tracts (all states)")
            print("  Save as: data/census/DECENNIALPL2020_P1_tract.csv")
        else:
            n = data_loader.load_census_tracts()
            print(f"  ✓ {n:,} census tracts loaded")

        banner("Census — MSA Populations (DECENNIALPL2020.P1)")
        # Auto-detect: accept settings path OR any DECENNIALPL2020_P1*.csv in the dir
        census_dir = settings.census_msa_csv.parent
        msa_candidates = (
            [settings.census_msa_csv] if settings.census_msa_csv.exists() else []
        ) + sorted(census_dir.glob("DECENNIALPL2020_P1*.csv"))
        seen = set()
        msa_candidates = [p for p in msa_candidates
                          if not (str(p) in seen or seen.add(str(p)))]
        if not msa_candidates:
            print(f"  ⚠ No MSA population file found in: {census_dir}")
            print("  Download from data.census.gov → Table DECENNIALPL2020.P1")
            print("  Filter: Geography = Metropolitan Statistical Areas → All")
            print("  Save as: data/census/DECENNIALPL2020_P1_msa.csv")
        else:
            print(f"  Found: {msa_candidates[0].name}")
            n = data_loader.load_census_msa()
            if n:
                print(f"  ✓ {n:,} MSAs loaded")

    # ── Redfin ────────────────────────────────────────────────────────────
    if only in (None, "redfin"):
        banner("Redfin Favorites / Saved Homes")
        csv_files = list(settings.redfin_dir.glob("*.csv"))
        if not csv_files:
            print(f"  ⚠ No CSV files in: {settings.redfin_dir}")
            print("  → Drop your Redfin export CSVs (with lat/lon) into data/redfin/")
        else:
            print(f"  Found {len(csv_files)} file(s): {[f.name for f in csv_files]}")
            n = data_loader.load_redfin(use_geo)
            print(f"  ✓ {n:,} houses loaded")

    # ── Sold homes ────────────────────────────────────────────────────────
    if only in (None, "sold"):
        banner("Sold Homes")
        csv_files = list(settings.sold_dir.glob("*.csv"))
        if not csv_files:
            print(f"  ⚠ No CSV files in: {settings.sold_dir}")
            print("  → Drop county sold-homes CSV files into data/sold/")
            print("  → Supported counties: Allegheny PA (auto-detected)")
            print("  → Other formats: generic fallback parser will be used")
        else:
            print(f"  Found {len(csv_files)} file(s): {[f.name for f in csv_files]}")
            n = data_loader.load_sold_homes(
                run_geocoding=not no_geocoding,
                geo_utils=use_geo,
            )
            print(f"  ✓ {n:,} sold homes loaded")

    # ── Re-geocode pending ────────────────────────────────────────────────
    if only == "geocode":
        banner("Re-geocode Pending Sold Homes")
        from services.geocoder import geocode_pending
        resolved = geocode_pending(verbose=True)
        print(f"  ✓ {resolved} newly resolved")

    # ── Match sold homes → houses ─────────────────────────────────────────
    if only in (None, "sold", "match"):
        banner("Matching Sold Records → Houses")
        n_matched = store.match_sold_to_houses()
        print(f"  ✓ {n_matched} sold records linked to houses as historical snapshots")

    # ── Repair X-coded msa_codes ──────────────────────────────────────────
    if only == "repair":
        banner("Repair X-coded MSA Codes in census_msa")
        print("  This re-runs name→CBSA code matching with the improved normalizer")
        print("  (accent stripping). Safe to run on an existing database.\n")
        repaired = data_loader.repair_msa_codes()
        if repaired:
            print(f"\n  ✓ Fixed {repaired} msa_codes — re-run the flood risk query.")
        else:
            print("\n  ✓ No repairs needed.")
        store.close()
        return

    # ── Summary ───────────────────────────────────────────────────────────
    banner("Database Summary")
    try:
        tables = {
            "houses":        store.query("SELECT COUNT(*) as n FROM houses").iloc[0]["n"],
            "nri_tracts":    store.query("SELECT COUNT(*) as n FROM nri_tracts").iloc[0]["n"],
            "census_tracts": store.query("SELECT COUNT(*) as n FROM census_tracts").iloc[0]["n"],
            "census_msa":    store.query("SELECT COUNT(*) as n FROM census_msa").iloc[0]["n"],
            "cbsa_counties": store.query("SELECT COUNT(*) as n FROM cbsa_counties").iloc[0]["n"],
            "sold_homes":    store.query("SELECT COUNT(*) as n FROM sold_homes").iloc[0]["n"],
            "geocode_cache": store.query("SELECT COUNT(*) as n FROM geocode_cache").iloc[0]["n"],
        }
        for tbl, cnt in tables.items():
            status = "✓" if cnt > 0 else "○"
            print(f"  {status} {tbl:<20} {int(cnt):>10,} rows")
    except Exception as e:
        print(f"  Error reading summary: {e}")

    # ── Sold homes geocoding coverage ─────────────────────────────────────
    try:
        geo_stats = store.get_geocode_stats()
        if geo_stats["total"] > 0:
            print(f"\n  Sold homes geocoding ({geo_stats['total']} total):")
            for row in geo_stats["breakdown"]:
                src = row.get("geocode_source") or "—"
                st  = row.get("geocode_status") or "pending"
                print(f"    {st:<10} via {src:<20} {int(row['count']):>6,}")
            pending_count = store.query(
                "SELECT COUNT(*) as n FROM sold_homes "
                "WHERE geocode_status IS NULL OR geocode_status = 'pending'"
            ).iloc[0]["n"]
            if pending_count > 0:
                print(f"\n  ⚠ {int(pending_count)} addresses still pending geocoding")
                print("    → Run: python setup_data.py --only geocode")
    except Exception:
        pass

    # ── Tract FIPS coverage ───────────────────────────────────────────────
    try:
        total = store.query("SELECT COUNT(*) as n FROM houses").iloc[0]["n"]
        resolved = store.query(
            "SELECT COUNT(*) as n FROM houses WHERE tract_fips IS NOT NULL"
        ).iloc[0]["n"]
        if total > 0:
            pct = 100 * resolved / total
            print(f"\n  Redfin houses tract FIPS: {int(resolved)}/{int(total)} ({pct:.0f}%)")
            if resolved < total:
                print("  → Run: python setup_data.py --resolve-tracts")
    except Exception:
        pass

    # ── Arm's-length breakdown ────────────────────────────────────────────
    try:
        al_df = store.query("""
            SELECT is_arms_length, COUNT(*) as n
            FROM sold_homes
            WHERE sold_price IS NOT NULL
            GROUP BY is_arms_length
        """)
        if len(al_df) > 0:
            total_sold = int(al_df["n"].sum())
            al_n = int(al_df[al_df["is_arms_length"] == True]["n"].sum())
            nal_n = int(al_df[al_df["is_arms_length"] == False]["n"].sum())
            print(f"\n  Arm's-length classification ({total_sold} sold records):")
            print(f"    ✓ Arm's-length:     {al_n:>6,} ({100*al_n//max(total_sold,1)}%)")
            print(f"    ✗ Non-arm's-length: {nal_n:>6,} ({100*nal_n//max(total_sold,1)}%)")
    except Exception:
        pass

    elapsed = time.time() - t0
    print(f"\n✅ Done in {elapsed:.1f}s\n")
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data into DuckDB")
    parser.add_argument(
        "--only",
        choices=["nri", "census", "redfin", "sold", "geocode", "repair", "match"],
        help=(
            "Load only a specific dataset, or run a maintenance task:\n"
            "  nri      — FEMA National Risk Index\n"
            "  census   — CBSA crosswalk + tract + MSA populations (in correct order)\n"
            "  redfin   — Redfin favorites CSVs\n"
            "  sold     — County sold-homes CSVs\n"
            "  geocode  — Retry pending geocodes for sold homes\n"
            "  repair   — Fix X-coded msa_codes in census_msa (no data reload needed)"
        ),
    )
    parser.add_argument(
        "--resolve-tracts", action="store_true",
        help="Resolve census tract FIPS for houses via spatial join or Census API",
    )
    parser.add_argument(
        "--no-geocoding", action="store_true",
        help="Skip geocoding step for sold homes (useful for dry runs)",
    )
    args = parser.parse_args()
    run_all(only=args.only, resolve_tracts=args.resolve_tracts,
            no_geocoding=args.no_geocoding)

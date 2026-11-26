"""`Federal Transit Administration (FTA) Data` loaders, moved out of services/data_loader.py.

Behaviour is unchanged: the facade in services/data_loader.py re-exports every name here, so
`from services.data_loader import ...` (and scripts that call it) keep working.
"""

from __future__ import annotations

"""
Data loader: reads raw files and populates DuckDB tables.
Run via setup_data.py — only needed when data files change.
"""
import json
import hashlib
import math
import re

import pandas as pd


from config import settings
import db.duckdb_store as store


from services.loaders.census import _build_cbsa_lookup, _match_msa_to_cbsa  # CBSA matcher

__all__ = [
    "load_transit_agencies",
    "load_transit_ridership",
    "load_transit_safety",
    "load_transit_routes",
    "load_transit_route_agencies",
    "load_transit_route_zctas",
    "load_transit_route_places",
    "load_fta_data",
]


def _route_uid(*parts: str) -> str:
    """Deterministic identity for one route feature in a snapshot.

    Every stored attribute participates, not just the shapefile row index:
    feeds publish several routes with the same ntd_id/feed_id/route_id/
    shape_id, and sometimes those rows differ only in metadata. Identical
    duplicate rows still collapse under the primary key.
    """
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:20]


# ── Federal Transit Administration (FTA) Data ─────────────────────────────────


def load_transit_agencies() -> int:
    """
    Load transit agency master records from the Ridership Master sheet into `transit_agencies`.
    Resolves CBSA msa_code for each agency via UZA Name / HQ City & State.
    """
    path = getattr(
        settings,
        "fta_ridership_file",
        settings.data_dir / "fta" / "September 2026 Raw Monthly Ridership (no adjustments or estimates)_261102.xlsx",
    )
    if not path.exists():
        print(f"  FTA Ridership file not found at {path}")
        return 0

    print("  Loading transit agencies from Ridership Master sheet...")
    df = pd.read_excel(path, sheet_name="Master")
    if df.empty:
        return 0
    df = df[
        df["NTD ID"].notna()
        & (df["NTD ID"].astype(str).str.strip() != "")
        & (df["NTD ID"].astype(str).str.lower() != "nan")
    ].copy()

    cbsa_df = store.query("SELECT DISTINCT cbsa_code, cbsa_title FROM cbsa_counties")
    cbsa_lookup = _build_cbsa_lookup(cbsa_df) if not cbsa_df.empty else {}

    col_map = {
        "NTD ID": "ntd_id",
        "Legacy NTD ID": "legacy_ntd_id",
        "Agency": "agency",
        "Mode": "mode",
        "TOS": "tos",
        "3 Mode": "three_mode",
        "Mode/Type of Service Status": "mode_status",
        "Reporter Type": "reporter_type",
        "Organization Type": "organization_type",
        "HQ City": "hq_city",
        "HQ State": "hq_state",
        "UACE CD": "uace_cd",
        "UZA Name": "uza_name",
        "UZA SQ Miles": "uza_sq_miles",
        "UZA Population": "uza_population",
        "Service Area Population": "service_area_population",
        "Service Area SQ Miles": "service_area_sq_miles",
        "Total Operating Expenses (Op Costs)": "operating_expenses",
        "Total Operating Expenses FY": "operating_expenses_fy",
        "Total Fare Revenues (Fares)": "fare_revenues",
        "Total Fare Revenues FY": "fare_revenues_fy",
        "Avg Cost Per Trip FY": "avg_cost_per_trip",
        "Avg Fares Per Trip FY": "avg_fares_per_trip",
    }

    renamed = df.rename(columns=col_map)
    out = pd.DataFrame()
    out["ntd_id"] = renamed["ntd_id"].astype(str).str.split(".").str[0].str.zfill(5)
    out["legacy_ntd_id"] = renamed.get("legacy_ntd_id", pd.Series(dtype=str)).astype(str).str.split(".").str[0]
    out["agency"] = renamed.get("agency", "").astype(str).str.strip()
    out["mode"] = renamed.get("mode", "").astype(str).str.strip()
    out["tos"] = renamed.get("tos", "").astype(str).str.strip()
    out["three_mode"] = renamed.get("three_mode", "").astype(str).str.strip()
    out["mode_status"] = renamed.get("mode_status", "").astype(str).str.strip()
    out["reporter_type"] = renamed.get("reporter_type", "").astype(str).str.strip()
    out["organization_type"] = renamed.get("organization_type", "").astype(str).str.strip()
    out["hq_city"] = renamed.get("hq_city", "").astype(str).str.strip()
    out["hq_state"] = renamed.get("hq_state", "").astype(str).str.strip()
    out["uace_cd"] = renamed.get("uace_cd", "").astype(str).str.split(".").str[0]
    out["uza_name"] = renamed.get("uza_name", "").astype(str).str.strip()
    out["uza_sq_miles"] = pd.to_numeric(renamed.get("uza_sq_miles"), errors="coerce")
    out["uza_population"] = pd.to_numeric(renamed.get("uza_population"), errors="coerce")
    out["service_area_population"] = pd.to_numeric(renamed.get("service_area_population"), errors="coerce")
    out["service_area_sq_miles"] = pd.to_numeric(renamed.get("service_area_sq_miles"), errors="coerce")

    op_costs = (
        renamed.get("operating_expenses")
        if "operating_expenses" in renamed.columns
        else renamed.get("operating_expenses_fy")
    )
    fares = renamed.get("fare_revenues") if "fare_revenues" in renamed.columns else renamed.get("fare_revenues_fy")
    out["operating_expenses"] = pd.to_numeric(op_costs, errors="coerce")
    out["fare_revenues"] = pd.to_numeric(fares, errors="coerce")
    out["avg_cost_per_trip"] = pd.to_numeric(renamed.get("avg_cost_per_trip"), errors="coerce")
    out["avg_fares_per_trip"] = pd.to_numeric(renamed.get("avg_fares_per_trip"), errors="coerce")

    msa_codes = []
    for _, row in out.iterrows():
        code = None
        uza = row.get("uza_name") or ""
        city = row.get("hq_city") or ""
        state = row.get("hq_state") or ""
        if uza and cbsa_lookup:
            code = _match_msa_to_cbsa(uza, cbsa_lookup)
        if not code and city and state and cbsa_lookup:
            code = _match_msa_to_cbsa(f"{city}, {state}", cbsa_lookup)
        msa_codes.append(code)
    out["msa_code"] = msa_codes

    cols = [
        "ntd_id",
        "legacy_ntd_id",
        "agency",
        "mode",
        "tos",
        "three_mode",
        "mode_status",
        "reporter_type",
        "organization_type",
        "hq_city",
        "hq_state",
        "uace_cd",
        "uza_name",
        "uza_sq_miles",
        "uza_population",
        "service_area_population",
        "service_area_sq_miles",
        "operating_expenses",
        "fare_revenues",
        "avg_cost_per_trip",
        "avg_fares_per_trip",
        "msa_code",
    ]
    out = out[cols].drop_duplicates(subset=["ntd_id", "mode", "tos"])
    store.upsert_df("transit_agencies", out)
    matched_msa = int(out["msa_code"].notna().sum())
    print(f"  ✓ {len(out):,} transit agency modes loaded ({matched_msa:,} linked to Census MSAs)")
    return len(out)


def load_transit_ridership() -> int:
    """
    Load monthly transit ridership (UPT) from Ridership UPT sheet into `transit_ridership`.
    """
    path = getattr(
        settings,
        "fta_ridership_file",
        settings.data_dir / "fta" / "September 2026 Raw Monthly Ridership (no adjustments or estimates)_261102.xlsx",
    )
    if not path.exists():
        print(f"  FTA Ridership file not found at {path}")
        return 0

    print("  Loading monthly transit ridership from UPT sheet...")
    df_upt = pd.read_excel(path, sheet_name="UPT")
    if df_upt.empty:
        return 0

    month_pattern = re.compile(r"^\d{1,2}/\d{4}$")
    month_cols = [c for c in df_upt.columns if month_pattern.match(str(c))]
    if not month_cols:
        print("  No monthly columns found in UPT sheet")
        return 0

    id_vars = ["NTD ID", "Agency", "Mode", "TOS"]
    df_upt = df_upt[
        df_upt["NTD ID"].notna()
        & (df_upt["NTD ID"].astype(str).str.strip() != "")
        & (df_upt["NTD ID"].astype(str).str.lower() != "nan")
    ].copy()
    melted = pd.melt(df_upt, id_vars=id_vars, value_vars=month_cols, var_name="month_str", value_name="upt")
    melted = melted[melted["upt"].notna()].copy()

    splits = melted["month_str"].str.split("/", expand=True)
    melted["month"] = splits[0].astype(int)
    melted["year"] = splits[1].astype(int)
    melted["date"] = pd.to_datetime(
        melted["year"].astype(str) + "-" + melted["month"].astype(str).str.zfill(2) + "-01"
    ).dt.date
    melted["ntd_id"] = melted["NTD ID"].astype(str).str.split(".").str[0].str.zfill(5)
    melted["agency"] = melted["Agency"].astype(str).str.strip()
    melted["mode"] = melted["Mode"].astype(str).str.strip()
    melted["tos"] = melted["TOS"].astype(str).str.strip()
    melted["upt"] = pd.to_numeric(melted["upt"], errors="coerce")

    agencies_df = store.query(
        "SELECT DISTINCT ntd_id, mode, tos, msa_code FROM transit_agencies WHERE msa_code IS NOT NULL"
    )
    if not agencies_df.empty:
        melted = melted.merge(agencies_df, on=["ntd_id", "mode", "tos"], how="left")
    else:
        melted["msa_code"] = None

    cols = ["ntd_id", "agency", "mode", "tos", "msa_code", "date", "year", "month", "upt"]
    out = melted[cols].drop_duplicates(subset=["ntd_id", "mode", "tos", "date"])

    chunk_size = 200_000
    n_chunks = max(1, math.ceil(len(out) / chunk_size))
    for i in range(n_chunks):
        chunk = out.iloc[i * chunk_size : (i + 1) * chunk_size]
        store.upsert_df("transit_ridership", chunk)

    print(f"  ✓ {len(out):,} monthly transit ridership records loaded across {out['ntd_id'].nunique():,} agencies")
    return len(out)


def load_transit_safety() -> int:
    """
    Load Safety and Security time series from S&S Time Series into `transit_safety`.
    """
    path = getattr(settings, "fta_safety_file", settings.data_dir / "fta" / "S&S Time Series-July 2026_261102.xlsx")
    if not path.exists():
        print(f"  FTA Safety & Security file not found at {path}")
        return 0

    print("  Loading safety and security time series from S&S Time Series...")
    df = pd.read_excel(path, sheet_name="Data", header=2)
    if df.empty:
        return 0
    df = df[
        df["NTD ID"].notna()
        & (df["NTD ID"].astype(str).str.strip() != "")
        & (df["NTD ID"].astype(str).str.lower() != "nan")
    ].copy()

    cbsa_df = store.query("SELECT DISTINCT cbsa_code, cbsa_title FROM cbsa_counties")
    cbsa_lookup = _build_cbsa_lookup(cbsa_df) if not cbsa_df.empty else {}

    col_map = {
        "Year": "year",
        "NTD ID": "ntd_id",
        "Agency": "agency",
        "Mode": "mode",
        "TOS": "tos",
        "City": "city",
        "State": "state",
        "UACE CD": "uace_cd",
        "UZA Name": "uza_name",
        "Reporter Type": "reporter_type",
        "UPT": "upt",
        "Vehicle & Pass. Car Revenue Miles": "vrm",
        "VRH": "vrh",
        "VOMS": "voms",
        "    Collision Total": "collision_total",
        "    Derailment Total": "derailment_total",
        "    Fire Total": "fire_total",
        "Security Total": "security_total",
        "Assault/Homice Events Involving Transit Workers": "transit_worker_assault_events",
        "    Event Total": "event_total",
        "    Total Fatalities": "total_fatalities",
        "    Passenger": "fatality_passenger",
        "    Total Employee": "fatality_employee",
        "    Total Other": "fatality_other",
        "    Total Injuries": "total_injuries",
        "Serious Injuries": "serious_injuries",
    }
    renamed = df.rename(columns=col_map)
    out = pd.DataFrame()
    out["year"] = pd.to_numeric(renamed["year"], errors="coerce").fillna(0).astype(int)
    out["ntd_id"] = renamed["ntd_id"].astype(str).str.split(".").str[0].str.zfill(5)
    out["agency"] = renamed["agency"].astype(str).str.strip()
    out["mode"] = renamed["mode"].astype(str).str.strip()
    out["tos"] = renamed["tos"].astype(str).str.strip()
    out["city"] = renamed["city"].astype(str).str.strip()
    out["state"] = renamed["state"].astype(str).str.strip()
    out["uace_cd"] = renamed["uace_cd"].astype(str).str.split(".").str[0]
    out["uza_name"] = renamed["uza_name"].astype(str).str.strip()
    out["reporter_type"] = renamed["reporter_type"].astype(str).str.strip()

    for col in [
        "event_total",
        "collision_total",
        "derailment_total",
        "fire_total",
        "security_total",
        "transit_worker_assault_events",
        "total_fatalities",
        "fatality_passenger",
        "fatality_employee",
        "fatality_other",
        "total_injuries",
        "injury_passenger",
        "injury_employee",
        "injury_other",
        "serious_injuries",
    ]:
        if col in renamed.columns:
            out[col] = pd.to_numeric(renamed[col], errors="coerce").fillna(0).astype(int)
        else:
            out[col] = 0

    for col in ["upt", "vrm", "vrh", "voms"]:
        if col in renamed.columns:
            out[col] = pd.to_numeric(renamed[col], errors="coerce")
        else:
            out[col] = None

    msa_codes = []
    for _, row in out.iterrows():
        code = None
        uza = row.get("uza_name") or ""
        city = row.get("city") or ""
        state = row.get("state") or ""
        if uza and cbsa_lookup:
            code = _match_msa_to_cbsa(uza, cbsa_lookup)
        if not code and city and state and cbsa_lookup:
            code = _match_msa_to_cbsa(f"{city}, {state}", cbsa_lookup)
        msa_codes.append(code)
    out["msa_code"] = msa_codes

    cols = [
        "year",
        "ntd_id",
        "agency",
        "mode",
        "tos",
        "city",
        "state",
        "uace_cd",
        "uza_name",
        "reporter_type",
        "msa_code",
        "event_total",
        "collision_total",
        "derailment_total",
        "fire_total",
        "security_total",
        "transit_worker_assault_events",
        "total_fatalities",
        "fatality_passenger",
        "fatality_employee",
        "fatality_other",
        "total_injuries",
        "injury_passenger",
        "injury_employee",
        "injury_other",
        "serious_injuries",
        "upt",
        "vrm",
        "vrh",
        "voms",
    ]
    out = out[cols].drop_duplicates(subset=["year", "ntd_id", "mode", "tos"])
    store.upsert_df("transit_safety", out)
    print(f"  ✓ {len(out):,} transit safety records loaded ({out['year'].min()}–{out['year'].max()})")
    return len(out)


def load_transit_routes() -> int:
    """
    Load National Transit Map transit routes shapefile into `transit_routes`.
    """
    import pyogrio
    import shapely

    path = getattr(
        settings, "fta_routes_shp", settings.data_dir / "fta" / "connections" / "National_Transit_Map_Routes.shp"
    )
    if not path.exists():
        print(f"  National Transit Map shapefile not found at {path}")
        return 0

    print("  Loading National Transit Map routes shapefile...")
    gdf = pyogrio.read_dataframe(path)
    if gdf.empty:
        return 0

    geoms_2d = shapely.force_2d(gdf.geometry.values)
    bounds = shapely.bounds(geoms_2d)

    rows = []
    for i in range(len(gdf)):
        geom = geoms_2d[i]
        if geom is None or shapely.is_empty(geom):
            continue
        minx, miny, maxx, maxy = bounds[i]
        ntd_id = str(gdf["ntd_id"].iloc[i] or "").split(".")[0].zfill(5)
        route_id = str(gdf["route_id"].iloc[i] or "") if pd.notna(gdf["route_id"].iloc[i]) else ""
        shape_id = str(gdf["shape_id"].iloc[i] or "") if pd.notna(gdf["shape_id"].iloc[i]) else ""
        feed_id = str(gdf["feed_id"].iloc[i] or "") if pd.notna(gdf["feed_id"].iloc[i]) else ""
        agency_id = str(gdf["agency_id"].iloc[i] or "") if pd.notna(gdf["agency_id"].iloc[i]) else ""
        route_shor = str(gdf["route_shor"].iloc[i] or "") if pd.notna(gdf["route_shor"].iloc[i]) else ""
        route_long = str(gdf["route_long"].iloc[i] or "") if pd.notna(gdf["route_long"].iloc[i]) else ""
        route_desc = str(gdf["route_desc"].iloc[i] or "") if pd.notna(gdf["route_desc"].iloc[i]) else ""
        route_type = str(gdf["route_type"].iloc[i] or "") if pd.notna(gdf["route_type"].iloc[i]) else ""
        route_ty_1 = str(gdf["route_ty_1"].iloc[i] or "") if pd.notna(gdf["route_ty_1"].iloc[i]) else ""
        route_url = str(gdf["route_url"].iloc[i] or "") if pd.notna(gdf["route_url"].iloc[i]) else ""
        route_colo = str(gdf["route_colo"].iloc[i] or "") if pd.notna(gdf["route_colo"].iloc[i]) else ""
        if route_colo and not route_colo.startswith("#") and route_colo.lower() != "nan":
            route_colo = f"#{route_colo}"
        else:
            route_colo = ""
        route_text = str(gdf["route_text"].iloc[i] or "") if pd.notna(gdf["route_text"].iloc[i]) else ""
        if route_text and not route_text.startswith("#") and route_text.lower() != "nan":
            route_text = f"#{route_text}"
        else:
            route_text = ""
        download_d = str(gdf["download_d"].iloc[i] or "") if pd.notna(gdf["download_d"].iloc[i]) else ""

        geo_json_str = shapely.to_geojson(geom)
        props = {
            "ntd_id": ntd_id,
            "route_id": route_id,
            "route_short_name": route_shor,
            "route_long_name": route_long,
            "route_type_name": route_ty_1,
            "color": route_colo or "#4f8ef7",
        }
        properties_json = json.dumps(props, ensure_ascii=False)
        uid = _route_uid(
            ntd_id,
            feed_id,
            route_id,
            shape_id,
            agency_id,
            route_shor,
            route_long,
            route_desc,
            route_type,
            route_ty_1,
            route_url,
            route_colo,
            route_text,
            download_d,
            geo_json_str,
            properties_json,
        )

        rows.append(
            {
                "route_uid": uid,
                "route_id": route_id,
                "ntd_id": ntd_id,
                "feed_id": feed_id,
                "shape_id": shape_id,
                "agency_id": agency_id,
                "route_short_name": route_shor,
                "route_long_name": route_long,
                "route_desc": route_desc,
                "route_type": route_type,
                "route_type_name": route_ty_1,
                "route_url": route_url,
                "route_color": route_colo,
                "route_text_color": route_text,
                "download_date": download_d,
                "min_lon": float(minx),
                "min_lat": float(miny),
                "max_lon": float(maxx),
                "max_lat": float(maxy),
                "geometry_json": geo_json_str,
                "properties_json": properties_json,
            }
        )

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["route_uid"])

    chunk_size = 25_000
    n_chunks = max(1, math.ceil(len(out) / chunk_size))
    conn = store.get_conn()
    conn.execute("BEGIN TRANSACTION")
    try:
        # A national route snapshot: replace the table so routes removed upstream
        # do not survive an upsert. route_uid is stable across refreshes.
        conn.execute("DELETE FROM transit_routes")
        for i in range(n_chunks):
            chunk = out.iloc[i * chunk_size : (i + 1) * chunk_size]
            store.upsert_df("transit_routes", chunk)
            if n_chunks > 1:
                print(f"    Upserted routes chunk {i + 1}/{n_chunks} ({len(chunk):,} features)")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    print(f"  ✓ {len(out):,} transit route features loaded across {out['ntd_id'].nunique():,} agencies")
    load_transit_route_agencies()
    return len(out)


def load_transit_route_agencies() -> int:
    """Split the route file's multi-valued ``ntd_id`` into one row per route/agency.

    The source field is a semicolon-separated list for some feeds (for example
    ``90032;90034;90131``).  A direct equality join silently misses every one of
    those routes, so the catalog joins through this bridge instead.  Unmatched
    agency IDs are retained: they are a real coverage signal, not a parse error.
    """
    routes = store.query("SELECT route_uid, ntd_id FROM transit_routes")
    rows: list[dict[str, str]] = []
    if not routes.empty:
        for row in routes.itertuples(index=False):
            route_uid = "" if pd.isna(row.route_uid) else str(row.route_uid).strip()
            raw = "" if pd.isna(row.ntd_id) else str(row.ntd_id).strip()
            if not route_uid:
                continue
            ids = sorted({part.strip() for part in raw.split(";") if part.strip()})
            for ntd_id in ids:
                rows.append({"route_uid": route_uid, "ntd_id": ntd_id, "raw_ntd_id": raw})

    out = pd.DataFrame(rows, columns=["route_uid", "ntd_id", "raw_ntd_id"])
    conn = store.get_conn()
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("DELETE FROM transit_route_agencies")
        if not out.empty:
            conn.register("__transit_route_agencies", out)
            conn.execute(
                "INSERT INTO transit_route_agencies (route_uid, ntd_id, raw_ntd_id) "
                "SELECT route_uid, ntd_id, raw_ntd_id FROM __transit_route_agencies"
            )
            conn.unregister("__transit_route_agencies")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    print(
        f"  ✓ {len(out):,} route-agency memberships across "
        f"{out['route_uid'].nunique() if not out.empty else 0:,} routes"
    )
    return len(out)


def _build_transit_route_bridge(boundary_table: str, boundary_key: str, bridge_table: str) -> int:
    """Build the route-to-area bridge for transit route linework.

    The bridge is deliberately separate from ``transit_routes``: a route can
    cross many ZCTAs/Places, and a route that merely touches a boundary is not a
    member of it.  Overlaps are measured on projected linework in metres, then
    divided by each route's full projected length so ``route_fraction`` stays
    interpretable even when boundary coverage is incomplete.

    A missing/empty source table leaves the existing bridge untouched: a failed
    or partial boundary refresh must not erase a good bridge.  A successful
    build replaces the bridge in one transaction.
    """
    import geopandas as gpd
    import shapely

    conn = store.get_conn()
    route_count = int(store.query("SELECT COUNT(*) AS n FROM transit_routes").iloc[0]["n"])
    if not route_count:
        print(f"  ⚠ No transit routes loaded; {bridge_table} left untouched")
        return 0
    boundary_count = int(
        store.query(f"SELECT COUNT(*) AS n FROM {boundary_table} WHERE {boundary_key} IS NOT NULL").iloc[0]["n"]
    )
    if not boundary_count:
        print(f"  ⚠ No {boundary_table} loaded; {bridge_table} left untouched")
        return 0

    boundaries = store.query(
        f"SELECT {boundary_key} AS boundary_key, geometry_json FROM {boundary_table} "
        f"WHERE {boundary_key} IS NOT NULL AND geometry_json IS NOT NULL"
    )
    if boundaries.empty:
        print(f"  ⚠ {boundary_table} has no usable geometry; {bridge_table} left untouched")
        return 0
    boundaries_gdf = gpd.GeoDataFrame(
        {"boundary_key": boundaries["boundary_key"].astype(str).to_numpy()},
        geometry=shapely.from_geojson(boundaries["geometry_json"].astype(str).to_numpy()),
        crs="EPSG:4326",
    )
    boundaries_gdf = boundaries_gdf[
        boundaries_gdf.geometry.notna() & ~boundaries_gdf.geometry.is_empty
    ].drop_duplicates(subset=["boundary_key"])
    if boundaries_gdf.empty:
        print(f"  ⚠ {boundary_table} has no usable geometry; {bridge_table} left untouched")
        return 0
    boundaries_m = boundaries_gdf.to_crs("EPSG:5070")

    print(f"  Building {bridge_table} from {route_count:,} transit routes × {len(boundaries_m):,} boundaries...")
    pairs: dict[tuple[str, str], float] = {}
    route_lengths: dict[str, float] = {}
    chunk_size = 5000
    offset = 0
    while True:
        routes = store.query(
            "SELECT route_uid, geometry_json FROM transit_routes "
            "WHERE route_uid IS NOT NULL AND geometry_json IS NOT NULL "
            "ORDER BY route_uid LIMIT ? OFFSET ?",
            [chunk_size, offset],
        )
        if routes.empty:
            break
        offset += len(routes)
        routes_gdf = gpd.GeoDataFrame(
            {"route_uid": routes["route_uid"].astype(str).to_numpy()},
            geometry=shapely.from_geojson(routes["geometry_json"].astype(str).to_numpy()),
            crs="EPSG:4326",
        )
        routes_gdf = routes_gdf[routes_gdf.geometry.notna() & ~routes_gdf.geometry.is_empty].drop_duplicates(
            subset=["route_uid"]
        )
        if routes_gdf.empty:
            continue
        routes_m = routes_gdf.to_crs("EPSG:5070")
        routes_m["geometry"] = routes_m.geometry.make_valid()
        routes_m["route_length_m"] = routes_m.length
        route_lengths.update(
            {
                str(row.route_uid): float(row.route_length_m)
                for row in routes_m[["route_uid", "route_length_m"]].itertuples(index=False)
                if row.route_length_m and row.route_length_m > 0
            }
        )

        minx, miny, maxx, maxy = routes_m.total_bounds
        nearby = boundaries_m.cx[minx - 1000 : maxx + 1000, miny - 1000 : maxy + 1000]
        if nearby.empty:
            continue
        nearby_tree = nearby.sindex
        nearby_geoms = list(nearby.geometry)
        nearby_keys = nearby["boundary_key"].to_numpy()
        for route_uid, route_geom in zip(routes_m["route_uid"], routes_m.geometry):
            try:
                candidate_idx = nearby_tree.query(route_geom, predicate="intersects")
            except TypeError:  # older shapely: query without a predicate, then measure each candidate
                candidate_idx = nearby_tree.query(route_geom)
            for idx in candidate_idx:
                try:
                    overlap = route_geom.intersection(nearby_geoms[idx]).length
                except Exception:
                    continue
                if overlap > 1.0:  # ignore point/boundary touches
                    key = (str(route_uid), str(nearby_keys[idx]))
                    pairs[key] = pairs.get(key, 0.0) + float(overlap)
        print(f"    Processed {offset:,}/{route_count:,} routes ({len(pairs):,} overlaps)")

    rows = []
    for (route_uid, boundary_value), overlap_m in pairs.items():
        total = route_lengths.get(route_uid, 0.0)
        if total <= 0:
            continue
        fraction = min(1.0, max(0.0, overlap_m / total))
        rows.append((route_uid, boundary_value, round(overlap_m, 3), round(fraction, 6)))
    out = pd.DataFrame(rows, columns=["route_uid", "boundary_key", "overlap_m", "route_fraction"])

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(f"DELETE FROM {bridge_table}")
        if not out.empty:
            conn.register("__transit_bridge", out)
            conn.execute(
                f"INSERT INTO {bridge_table} (route_uid, {boundary_key}, overlap_m, route_fraction) "
                f"SELECT route_uid, boundary_key, overlap_m, route_fraction FROM __transit_bridge"
            )
            conn.unregister("__transit_bridge")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    print(f"  ✓ {len(out):,} transit route/boundary overlap(s) written to {bridge_table}")
    return len(out)


def load_transit_route_zctas() -> int:
    """Rebuild the transit-route to ZCTA overlap bridge from installed geometry."""
    return _build_transit_route_bridge("zcta_boundaries", "zcta5", "transit_route_zctas")


def load_transit_route_places() -> int:
    """Rebuild the transit-route to Census Place overlap bridge from installed geometry."""
    return _build_transit_route_bridge("place_boundaries", "place_geoid", "transit_route_places")


def load_fta_data() -> dict[str, int]:
    """
    Ingest all 3 FTA datasets: transit agencies, ridership, safety, and routes.
    """
    print("\n--- Federal Transit Administration (FTA) Data Ingestion ---")
    n_agencies = load_transit_agencies()
    n_ridership = load_transit_ridership()
    n_safety = load_transit_safety()
    n_routes = load_transit_routes()
    n_route_zctas = load_transit_route_zctas()
    n_route_places = load_transit_route_places()
    return {
        "agencies": n_agencies,
        "ridership": n_ridership,
        "safety": n_safety,
        "routes": n_routes,
        "route_zctas": n_route_zctas,
        "route_places": n_route_places,
    }

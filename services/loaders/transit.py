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
    "load_fta_data",
]

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

        uid = hashlib.md5(f"{ntd_id}_{feed_id}_{route_id}_{shape_id}_{i}".encode()).hexdigest()[:20]
        geo_json_str = shapely.to_geojson(geom)
        props = {
            "ntd_id": ntd_id,
            "route_id": route_id,
            "route_short_name": route_shor,
            "route_long_name": route_long,
            "route_type_name": route_ty_1,
            "color": route_colo or "#4f8ef7",
        }

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
                "properties_json": json.dumps(props, ensure_ascii=False),
            }
        )

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["route_uid"])

    chunk_size = 25_000
    n_chunks = max(1, math.ceil(len(out) / chunk_size))
    for i in range(n_chunks):
        chunk = out.iloc[i * chunk_size : (i + 1) * chunk_size]
        store.upsert_df("transit_routes", chunk)
        if n_chunks > 1:
            print(f"    Upserted routes chunk {i + 1}/{n_chunks} ({len(chunk):,} features)")

    print(f"  ✓ {len(out):,} transit route features loaded across {out['ntd_id'].nunique():,} agencies")
    return len(out)


def load_fta_data() -> dict[str, int]:
    """
    Ingest all 3 FTA datasets: transit agencies, ridership, safety, and routes.
    """
    print("\n--- Federal Transit Administration (FTA) Data Ingestion ---")
    n_agencies = load_transit_agencies()
    n_ridership = load_transit_ridership()
    n_safety = load_transit_safety()
    n_routes = load_transit_routes()
    return {
        "agencies": n_agencies,
        "ridership": n_ridership,
        "safety": n_safety,
        "routes": n_routes,
    }

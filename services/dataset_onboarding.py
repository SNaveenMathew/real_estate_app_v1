"""Data onboarding workflow (the orchestrator behind the Data page).

The pipeline is a fixed, deterministic sequence of stages; a model is used only for optional
*drafting* (descriptions, units, synonyms) and never decides what gets joined:

    read -> stage -> profile -> [describe: human, optionally drafted by a model]
         -> [enrich: tract from lat/lon, geocode addresses - opt-in]
         -> analyze (deterministic): table proposal + relationship proposals with evidence
         -> human review -> approve / reject
         -> publish (atomic): physical table + catalog rows + concepts, then hot-reload

Approved changes are written through ``db.schema_catalog`` in ONE transaction and the in-memory
registry is reloaded on commit, so General Chat and House Chat see them on their next turn.
"""
from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

import db.duckdb_store as store
import db.schema_catalog as schema
from config import settings
from db import catalog_store
from db.catalog_model import (AVG, MAX, MEDIAN, MIN, RANK_ASC, RANK_DESC, SUM, EntityDomain,
                              Relationship, TableMeta, _concept)
from services import dataset_readers as readers
from services import relationship_discovery as disc
from services.dataset_readers import qi

VALID_ROLES = ("key", "label", "measure", "dimension", "date", "geo", "other")
VALID_DOMAINS = [k for k in schema.DOMAIN_LABELS if k != "system"]
_TABLE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,47}$")

_UNIT_TOKENS = {"pct", "percent", "percentage", "pctl", "amt", "num", "cnt", "ct", "idx"}
_GENERIC_ALIASES = {"value", "values", "score", "rate", "data", "number", "count", "total", "index", "rating",
                    "level", "type", "name", "id", "code", "year", "date", "status", "city", "state", "price",
                    "risk", "population", "average", "area", "size", "list", "rank", "house", "home", "houses"}
_ADDITIVE_UNITS = {"count", "people", "households", "units", "usd", "$", "dollars", "acres", "sqft"}


class OnboardingError(Exception):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conn():
    return store.get_conn()


def _get(dataset_id: str) -> dict:
    ds = catalog_store.get_dataset(_conn(), dataset_id)
    if not ds:
        raise OnboardingError("Dataset not found.", 404)
    return ds


def _humanize(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).replace("_", " ")).strip()


def _default_domain(columns: list[dict]) -> str:
    """A starting guess for the map's area lane, from what the columns identify (the person can change it)."""
    kinds = {c.get("key_kind") for c in columns}
    if kinds & {"address", "house_id"}:
        return "housing"
    if kinds & {"tract_fips", "block_group_fips", "block_fips", "county_fips", "cbsa_code", "zip", "state_fips", "city"}:
        return "geography"
    return "other"


def _title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return _humanize(re.sub(r"[-.]+", " ", stem)) or "Uploaded dataset"


def _stats(p: dict) -> dict:
    keep = ("nulls", "null_pct", "distinct", "unique_ratio", "min", "max", "mean", "min_len", "max_len", "rows")
    return {k: p[k] for k in keep if k in p and p[k] is not None}


def _existing_names(conn) -> set[str]:
    physical = {r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    cataloged = {r[0] for r in conn.execute("SELECT name FROM catalog_tables").fetchall()}
    return {n.lower() for n in physical | cataloged}


def validate_table_name(conn, name: str, *, allow: str | None = None) -> str:
    name = (name or "").strip().lower()
    if not _TABLE_NAME_RE.match(name):
        raise OnboardingError("Table names use lowercase letters, digits and underscores, start with a letter, "
                              "and are 3-48 characters long.")
    if name in readers.RESERVED_WORDS or name.startswith(("stg_", "catalog_", "sqlite_", "duckdb_")):
        raise OnboardingError(f"'{name}' is reserved. Choose another table name.")
    if name != allow and name in _existing_names(conn):
        raise OnboardingError(f"A table named '{name}' already exists. Choose another name.")
    return name


def _unique_table_name(conn, base: str) -> str:
    taken = _existing_names(conn)
    name, i = base, 2
    while name in taken or not _TABLE_NAME_RE.match(name):
        name = f"{base[:44]}_{i}"
        i += 1
    return name


def _preview(conn, table: str, n: int | None = None) -> list[dict]:
    n = n or int(getattr(settings, "onboarding_sample_rows", 8))
    try:
        return readers.preview_records(conn.execute(f"SELECT * FROM {qi(table)} LIMIT {int(n)}").df(), n)
    except Exception:
        return []


def _stage(conn, df: pd.DataFrame, name: str) -> None:
    conn.register("__stage_df", df)
    try:
        conn.execute(f"CREATE OR REPLACE TABLE {qi(name)} AS SELECT * FROM __stage_df")
    finally:
        conn.unregister("__stage_df")


def _record_stage(ds: dict, stage: str, status: str, detail: str = "") -> None:
    ds["pipeline"] = [s for s in ds.get("pipeline", []) if s["stage"] != stage]
    ds["pipeline"].append({"stage": stage, "status": status, "detail": detail})


def _supersede_proposals(conn, ds: dict) -> None:
    if ds["status"] == "draft":
        catalog_store.replace_proposals(conn, ds["dataset_id"], [])


# ---------------------------------------------------------------------------
# Create / describe
# ---------------------------------------------------------------------------

def create_dataset(path: str | Path, filename: str, *, sheet: str | None = None,
                   skiprows: int | None = None) -> dict:
    """Read an uploaded file, stage it in DuckDB and profile it. Nothing enters the catalog yet."""
    conn = _conn()
    try:
        res = readers.read_dataset(path, filename, sheet=sheet, skiprows=skiprows)
    except readers.DatasetReadError as exc:
        raise OnboardingError(str(exc))
    df, recs = readers.normalize_frame(res.df)
    dataset_id = uuid.uuid4().hex[:10]
    staging = f"stg_{dataset_id}"
    _stage(conn, df, staging)
    prof = disc.profile_table(conn, staging)
    columns = []
    for r in recs:
        p = prof[r["name"]]
        kind = disc.detect_key_kind(r["name"], p)
        columns.append({**r, "role": disc.guess_role(r["name"], p, kind), "role_source": "rules",
                        "key_kind": kind, "description": "", "synonyms": [], "unit": "", "include": True,
                        "drafted_by": "", "derived": False, "stats": _stats(p), "samples": p["samples"]})
    title = _title_from_filename(filename)
    ds = {
        "dataset_id": dataset_id, "table_name": _unique_table_name(conn, readers.slugify_table_name(title)),
        "title": title, "description": f"Uploaded from {filename}: {len(df):,} rows, {len(df.columns)} columns.",
        "grain": "", "domain": _default_domain(columns), "status": "draft", "source_filename": filename, "source_path": str(path),
        "format": res.format, "row_count": len(df), "staging_table": staging, "columns": columns,
        "notes": res.notes, "enrichments": [], "pipeline": [],
        "options": {"sheet": res.sheet, "sheets": res.sheets, "skiprows": skiprows,
                    "geometry": res.geometry},
    }
    _record_stage(ds, "read", "ok", f"{len(df):,} rows, {len(df.columns)} columns ({res.format})")
    _record_stage(ds, "profile", "ok", "Column types, key kinds and roles detected")
    catalog_store.save_dataset(conn, ds)
    catalog_store.audit(conn, "upload_dataset", "dataset", dataset_id, {"file": filename, "rows": len(df)})
    return detail(dataset_id)


def update_description(dataset_id: str, payload: dict) -> dict:
    """Save the human's description of the dataset and its columns (draft datasets only)."""
    conn = _conn()
    ds = _get(dataset_id)
    if ds["status"] != "draft":
        raise OnboardingError("This dataset is already published; retire it to change its description.")
    if "table_name" in payload:
        ds["table_name"] = validate_table_name(conn, payload["table_name"], allow=ds["table_name"])
    for key in ("title", "description", "grain"):
        if key in payload:
            ds[key] = str(payload[key] or "").strip()[:600]
    if "domain" in payload:
        if payload["domain"] not in VALID_DOMAINS:
            raise OnboardingError(f"Domain must be one of: {', '.join(VALID_DOMAINS)}.")
        ds["domain"] = payload["domain"]
    if not ds["title"]:
        raise OnboardingError("Give the dataset a name.")
    by_name = {c["name"]: c for c in ds["columns"]}
    for edit in payload.get("columns", []):
        c = by_name.get(edit.get("name"))
        if not c:
            continue
        if "role" in edit:
            if edit["role"] not in VALID_ROLES:
                raise OnboardingError(f"Role for '{c['name']}' must be one of: {', '.join(VALID_ROLES)}.")
            if edit["role"] != c["role"]:
                c["role"], c["role_source"] = edit["role"], "user"
        if "description" in edit:
            c["description"] = str(edit["description"] or "").strip()[:300]
            c["drafted_by"] = "" if edit.get("edited") else c.get("drafted_by", "")
        if "unit" in edit:
            c["unit"] = str(edit["unit"] or "").strip()[:20]
        if "synonyms" in edit:
            c["synonyms"] = _clean_synonyms(edit["synonyms"])
        if "include" in edit:
            c["include"] = bool(edit["include"])
    if not any(c["include"] for c in ds["columns"]):
        raise OnboardingError("Keep at least one column.")
    catalog_store.save_dataset(conn, ds)
    _supersede_proposals(conn, ds)
    return detail(dataset_id)


def _clean_synonyms(value: Any) -> list[str]:
    items = value if isinstance(value, list) else str(value or "").split(",")
    out, seen = [], set()
    for v in items:
        s = schema._normalize(str(v))
        if s and len(s.split()) <= 5 and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:6]


# ---------------------------------------------------------------------------
# Model drafting (optional; see services/catalog_llm.py)
# ---------------------------------------------------------------------------

def get_draftable(dataset_id: str) -> dict:
    ds = _get(dataset_id)
    if ds["status"] != "draft":
        raise OnboardingError("Only draft datasets can be drafted.")
    return ds


def apply_draft(dataset_id: str, result) -> dict:
    """Merge a ``catalog_llm`` result into the dataset (fills only what is still empty; degrades to rules)."""
    conn = _conn()
    ds = get_draftable(dataset_id)
    notes = list(result.notes)
    if result.ok and result.data:
        d = result.data
        if d.get("title") and ds["title"] == _title_from_filename(ds["source_filename"]):
            ds["title"] = d["title"]
        if d.get("description"):
            ds["description"] = d["description"]
        if d.get("grain") and not ds["grain"]:
            ds["grain"] = d["grain"]
        if d.get("domain") in VALID_DOMAINS and ds["domain"] == "other":
            ds["domain"] = d["domain"]
        by_name = {c["name"]: c for c in ds["columns"]}
        for cd in d.get("columns", []):
            c = by_name.get(cd["name"])
            if not c:
                continue
            if cd.get("description") and not c["description"]:
                c["description"], c["drafted_by"] = cd["description"], "model"
            if cd.get("unit") and not c["unit"]:
                c["unit"] = cd["unit"]
            if cd.get("synonyms") and not c["synonyms"]:
                c["synonyms"] = _clean_synonyms(cd["synonyms"])
            if cd.get("role") in VALID_ROLES and c.get("role_source") == "rules" and cd["role"] != c["role"]:
                c["role"], c["role_source"] = cd["role"], "model"
        _record_stage(ds, "model_draft", "ok", f"Drafted by {result.endpoint}")
    else:
        _record_stage(ds, "model_draft", "skipped", result.error or "No model was reachable; nothing was drafted.")
    ds["options"]["draft_meta"] = {"mode": "model" if result.ok else "rules", "endpoint": result.endpoint,
                                   "notes": notes, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    catalog_store.save_dataset(conn, ds)
    _supersede_proposals(conn, ds)
    out = detail(dataset_id)
    out["draft"] = {"ok": result.ok, "mode": ds["options"]["draft_meta"]["mode"], "endpoint": result.endpoint,
                    "error": result.error, "notes": notes}
    return out


def draft_with_model(dataset_id: str) -> dict:
    """Synchronous convenience wrapper (the API runs the model call in a worker thread instead)."""
    from services import catalog_llm
    return apply_draft(dataset_id, catalog_llm.draft_dataset(get_draftable(dataset_id)))


# ---------------------------------------------------------------------------
# Enrichment (opt-in, on the staged data, before analysis)
# ---------------------------------------------------------------------------

_GEOM_SOURCE: dict[str, str] = {}


def _geometry_source() -> str:
    if "v" not in _GEOM_SOURCE:
        from services import geo_utils
        _GEOM_SOURCE["v"] = geo_utils.geometry_source()
    return _GEOM_SOURCE["v"]


def available_enrichments(ds: dict) -> list[dict]:
    if ds["status"] != "draft":
        return []
    cols = [c for c in ds["columns"] if c.get("include", True)]
    lat = next((c for c in cols if c.get("key_kind") == "lat"), None)
    lon = next((c for c in cols if c.get("key_kind") == "lon"), None)
    has_tract = any(c.get("key_kind") == "tract_fips" for c in cols)
    out = []
    if lat and lon and not has_tract and not any(c["name"] == "tract_fips" for c in ds["columns"]):
        cap = int(getattr(settings, "onboarding_api_row_cap", 500))
        src = _geometry_source()
        api = "Census Geocoder API" in src
        note = f"Point-in-polygon against {src}."
        if api:
            note += f" No local tract geometry is loaded, so this uses the Census API and is limited to {cap} rows."
        out.append({"kind": "spatial_tract", "label": "Add census tract from coordinates", "possible": True, "note": note})
    addr = disc.address_columns(cols)
    if addr["address"] and (addr["zip"] or addr["city"]) and not (lat and lon):
        cap = int(getattr(settings, "onboarding_geocode_row_cap", 2000))
        out.append({"kind": "geocode", "label": "Geocode addresses (US Census)", "possible": True,
                    "note": f"Sends addresses to the Census geocoder over the internet, caches results in geocode_cache, "
                            f"and handles up to {cap:,} rows per run."})
    return out


def _add_column_from_series(conn, table: str, name: str, sql_type: str, mapping: pd.DataFrame, col: str) -> None:
    conn.execute(f"ALTER TABLE {qi(table)} ADD COLUMN {qi(name)} {sql_type}")
    conn.register("__enrich_map", mapping[["rid", col]])
    try:
        conn.execute(f"UPDATE {qi(table)} SET {qi(name)} = m.{qi(col)} FROM __enrich_map m WHERE {qi(table)}.rowid = m.rid")
    finally:
        conn.unregister("__enrich_map")


def _tract_lookup(work: pd.DataFrame) -> pd.Series:
    """tract_fips for each point, via services/geo_utils (local geometry first, API fallback)."""
    from services import geo_utils
    gdf = geo_utils._load_tracts_gdf()
    if gdf is None:
        cap = int(getattr(settings, "onboarding_api_row_cap", 500))
        if len(work) > cap:
            raise OnboardingError(
                f"No local tract geometry is loaded (NRI geometry cache or TIGER/Line shapefiles) and the Census "
                f"API fallback is limited to {cap} rows for uploads. Load the NRI shapefile with setup_data.py, "
                "or add TIGER/Line files to data/shapefiles/, then try again.")
    probe = work[["rid", "lat", "lon"]].copy()
    probe["tract_fips"] = None
    try:
        out = geo_utils.assign_tract_fips(probe)
        if len(out) == len(probe):
            return out["tract_fips"].reset_index(drop=True)
    except Exception:
        pass
    if gdf is None:
        raise OnboardingError("Tract lookup failed.")
    import geopandas as gpd    # fall back to a de-duplicated join (a point on a shared border can match two tracts)
    pts = gpd.GeoDataFrame(work[["rid"]].reset_index(drop=True),
                           geometry=gpd.points_from_xy(work["lon"].values, work["lat"].values), crs="EPSG:4326")
    joined = gpd.sjoin(pts, gdf[["tract_fips", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["tract_fips"].reindex(pts.index).reset_index(drop=True)


def enrich(dataset_id: str, kind: str) -> dict:
    conn = _conn()
    ds = _get(dataset_id)
    if ds["status"] != "draft":
        raise OnboardingError("Enrichment is only available before a dataset is published.")
    if kind not in {e["kind"] for e in available_enrichments(ds)}:
        raise OnboardingError("That enrichment is not available for this dataset.")
    stg = ds["staging_table"]
    cols = [c for c in ds["columns"] if c.get("include", True)]
    if kind == "spatial_tract":
        lat = next(c["name"] for c in cols if c.get("key_kind") == "lat")
        lon = next(c["name"] for c in cols if c.get("key_kind") == "lon")
        df = conn.execute(f"SELECT rowid AS rid, {qi(lat)} AS lat, {qi(lon)} AS lon FROM {qi(stg)}").df()
        ok = df.dropna(subset=["lat", "lon"])
        ok = ok[ok["lat"].between(-90, 90) & ok["lon"].between(-180, 180)].reset_index(drop=True)
        if ok.empty:
            raise OnboardingError("No rows have valid coordinates.")
        ok["tract_fips"] = _tract_lookup(ok)
        _add_column_from_series(conn, stg, "tract_fips", "VARCHAR", ok, "tract_fips")
        matched = int(ok["tract_fips"].notna().sum())
        result = {"kind": kind, "matched": matched, "total": len(ok), "source": _geometry_source(),
                  "examples": [{"lat": float(r.lat), "lon": float(r.lon), "tract_fips": r.tract_fips}
                               for r in ok[ok["tract_fips"].notna()].head(5).itertuples(index=False)],
                  "message": f"{matched:,} of {len(ok):,} points fall inside a census tract."}
        new_cols = [{"name": "tract_fips", "note": "Census tract FIPS derived from lat/lon by point-in-polygon "
                     "against the tract geometry used for houses (services/geo_utils.py)."}]
    else:
        from services.geocoder import geocode_dataframe
        addr = disc.address_columns(cols)
        cap = int(getattr(settings, "onboarding_geocode_row_cap", 2000))
        sel = ["rowid AS rid", f"{qi(addr['address'])} AS street",
               f"{qi(addr['city'])} AS city" if addr["city"] else "'' AS city",
               "'' AS state", f"{qi(addr['zip'])} AS zip" if addr["zip"] else "'' AS zip"]
        df = conn.execute(f"SELECT {', '.join(sel)} FROM {qi(stg)} LIMIT {cap}").df()
        # sale_id_col points at a column that does not exist so geocode_dataframe never writes to sold_homes.
        out = geocode_dataframe(df, street_col="street", city_col="city", state_col="state", zip_col="zip",
                                full_addr_col="full_address", sale_id_col="__no_such_column__",
                                single_fallback_limit=0, verbose=False)
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        for name, sql_type, col in (("lat", "DOUBLE", "lat"), ("lon", "DOUBLE", "lon"),
                                    ("tract_fips", "VARCHAR", "tract_fips"),
                                    ("geocode_status", "VARCHAR", "geocode_status")):
            if name not in {c["name"] for c in ds["columns"]}:
                _add_column_from_series(conn, stg, name, sql_type, out, col)
        matched = int(out["lat"].notna().sum())
        result = {"kind": kind, "matched": matched, "total": len(out), "source": "US Census Geocoder (with local cache)",
                  "examples": [{"address": r.street, "lat": float(r.lat), "lon": float(r.lon), "tract_fips": r.tract_fips}
                               for r in out[out["lat"].notna()].head(5).itertuples(index=False)],
                  "message": f"{matched:,} of {len(out):,} addresses were geocoded."
                             + (f" Only the first {cap:,} rows were processed." if int(conn.execute(f'SELECT COUNT(*) FROM {qi(stg)}').fetchone()[0]) > cap else "")}
        new_cols = [{"name": "lat", "note": "Latitude from the Census geocoder."},
                    {"name": "lon", "note": "Longitude from the Census geocoder."},
                    {"name": "tract_fips", "note": "Census tract FIPS returned by the Census geocoder."},
                    {"name": "geocode_status", "note": "success or pending, as set by services/geocoder.py."}]
    prof = disc.profile_table(conn, stg)
    have = {c["name"] for c in ds["columns"]}
    for nc in new_cols:
        p = prof.get(nc["name"])
        if not p or nc["name"] in have:
            continue
        k = disc.detect_key_kind(nc["name"], p)
        ds["columns"].append({"name": nc["name"], "source_name": "(derived)", "dtype": "text" if p["family"] == "text" else "float",
                              "role": disc.guess_role(nc["name"], p, k), "role_source": "rules", "key_kind": k,
                              "description": nc["note"], "synonyms": [], "unit": "", "include": True,
                              "drafted_by": "", "derived": True, "stats": _stats(p), "samples": p["samples"]})
    ds["enrichments"].append({k: v for k, v in result.items() if k != "examples"} | {"at": time.strftime("%Y-%m-%d %H:%M:%S")})
    _record_stage(ds, "enrich", "ok", result["message"])
    catalog_store.save_dataset(conn, ds)
    catalog_store.audit(conn, "enrich_dataset", "dataset", dataset_id, {"kind": kind, "matched": result["matched"]})
    _supersede_proposals(conn, ds)
    out = detail(dataset_id)
    out["enrichment_result"] = result
    return out


# ---------------------------------------------------------------------------
# Semantic concepts (generated from the human-approved description)
# ---------------------------------------------------------------------------

def _known_aliases() -> dict[str, str]:
    known: dict[str, str] = {}
    for key, item in schema._glossary().items():
        for a in item.get("aliases", []):
            known.setdefault(schema._normalize(a), key)
    return known


def _alias_ok(alias: str) -> bool:
    toks = alias.split()
    if not toks:
        return False
    if len(toks) == 1:
        return len(alias) >= 6 and alias not in _GENERIC_ALIASES
    return not all(t in _GENERIC_ALIASES for t in toks)


def _alias_candidates(col: dict) -> list[str]:
    raw = list(col.get("synonyms", []))          # the user's own phrasing comes first
    src = col.get("source_name")
    if src and src != "(derived)":
        raw.append(_humanize(src))
    raw.append(_humanize(col["name"]))
    out: list[str] = []
    for r in raw:
        n = schema._normalize(r)
        toks = n.split()
        variants = [n]
        if len(toks) > 1 and toks[-1] in _UNIT_TOKENS:
            variants.append(" ".join(toks[:-1]))
        for v in variants:
            if v and v not in out:
                out.append(v)
    return out


def _pick_label(cols: list[dict]) -> str | None:
    labels = [c for c in cols if c["role"] == "label" and c.get("include", True)]
    if labels:
        labels.sort(key=lambda c: -(c.get("stats", {}).get("unique_ratio") or 0))
        return labels[0]["name"]
    keys = [c for c in cols if c["role"] == "key" and c.get("include", True) and not c.get("derived")]
    return keys[0]["name"] if keys else None


def build_concepts(ds: dict) -> dict:
    """Concepts + entity domains + review warnings, from the described columns."""
    table = ds["table_name"]
    cols = [c for c in ds["columns"] if c.get("include", True)]
    known = _known_aliases()
    used: dict[str, str] = {}
    concepts: dict[str, dict] = {}
    alias_report: dict[str, list[str]] = {}
    warnings: list[str] = []
    notices: list[str] = []
    label = _pick_label(cols)
    for c in cols:
        if c["role"] != "measure":
            continue
        kept, dropped, overrides = [], [], {}
        for a in _alias_candidates(c):
            if not _alias_ok(a):
                continue
            owner = known.get(a) or used.get(a)
            if owner:
                dropped.append(f"'{a}' (already used by {owner})")
                continue
            hit = next((k for e, k in known.items() if e != a and f" {e} " in f" {a} "), None)
            if hit:
                overrides.setdefault(hit, []).append(a)
            kept.append(a)
        if dropped:
            warnings.append(f"{c['name']}: skipped alias {'; '.join(dropped)}.")
        for other, phrases in overrides.items():
            notices.append(f"{c['name']}: {', '.join(repr(x) for x in phrases)} takes precedence over the existing concept "
                           f"'{other}', whose phrase sits inside yours. Questions that use it will not also pull in "
                           f"{other}'s tables or filters.")
        if not kept:
            warnings.append(f"{c['name']}: no usable alias, so the assistant can only reach it through the table name. "
                            "Add synonyms to this column to make it discoverable.")
            continue
        for a in kept:
            used[a] = f"{table}.{c['name']}"
        expr = f"{table}.{c['name']}"
        rank_kw = {"group_by": f"{table}.{label}"} if label and label != c["name"] else {}
        ops = [AVG(expr), MEDIAN(expr), RANK_DESC(expr, **rank_kw), RANK_ASC(expr, **rank_kw), MAX(expr), MIN(expr)]
        if (c.get("unit") or "").lower() in _ADDITIVE_UNITS:
            ops.insert(1, SUM(expr))
        text = c["description"] or _humanize(c["name"])
        key = f"{table}_{c['name']}"
        concept = _concept(key, [table], kept, f"{text}" + (f" (unit: {c['unit']})" if c.get("unit") else "")
                           + f" - from the '{ds['title']}' dataset.",
                           columns=(expr,), operations=tuple(ops), null_policy="exclude NULL",
                           grain=ds["grain"] or "row", default_operation="avg")
        concept["scope_guard"] = False    # measure columns are legitimately filterable; see agents/tools.py
        if overrides:
            concept["overrides"] = sorted(overrides)      # declared precedence; honored by semantic_matches
        concepts[key] = concept
        alias_report[key] = kept
    domains: list[dict] = []
    label_cols = [c for c in cols if c["role"] == "label" and c.get("dtype") == "text"
                  and 2 <= (c.get("stats", {}).get("distinct") or 0) <= 5000
                  and (c.get("stats", {}).get("min_len") or 0) >= 2 and c.get("key_kind") != "address"]
    label_cols.sort(key=lambda c: (c.get("key_kind") != "city", c.get("stats", {}).get("distinct") or 0))
    for c in label_cols[:2]:
        domains.append({"name": f"{table}_{c['name']}", "table": table, "column": c["name"],
                        "entity_type": c["name"], "description": f"{ds['title']}: {_humanize(c['name'])} values",
                        "display_column": None, "match_mode": "exact_or_prefix",
                        "preferred_for": list(concepts)})
    suggestions: list[str] = []
    for key, concept in list(concepts.items())[:2]:
        a = concept["aliases"][0]
        suggestions.append(f"What is the average {a}?")
        suggestions.append(f"Rank the top 10 by {a}.")
    if domains and concepts:
        first = next(c for c in cols if c["name"] == domains[0]["column"])
        sample = (first.get("samples") or [None])[0]
        if sample:
            suggestions.append(f"What is the average {next(iter(concepts.values()))['aliases'][0]} in {sample}?")
    if not concepts:
        warnings.append("No measure columns were found, so no questions can be answered from this table's numbers. "
                        "You can still approve it to make it reachable by table name and use it in joins.")
    return {"concepts": concepts, "entity_domains": domains, "warnings": warnings, "notices": notices,
            "suggestions": suggestions[:5], "aliases": alias_report}


def _column_note(c: dict) -> str:
    text = c["description"] or _humanize(c["name"])
    if c.get("unit"):
        text += f" (unit: {c['unit']})"
    return text


# ---------------------------------------------------------------------------
# Analyze -> proposals
# ---------------------------------------------------------------------------

def analyze(dataset_id: str) -> dict:
    conn = _conn()
    ds = _get(dataset_id)
    if ds["status"] == "retired":
        raise OnboardingError("This dataset was retired.")
    source = ds["staging_table"] or ds["table_name"]
    t0 = time.monotonic()
    proposals: list[dict] = []
    if ds["status"] == "draft":
        if ds["grain"] == "":
            ds["grain"] = _default_grain(ds)
        built = build_concepts(ds)
        cols = [c for c in ds["columns"] if c.get("include", True)]
        proposals.append({
            "kind": "table", "group_key": "table",
            "title": f"Add {ds['table_name']} to the data model",
            "summary": (f"Registers {ds['table_name']} ({ds['row_count']:,} rows, {len(cols)} columns) so the assistant can "
                        f"query it. {len(built['concepts'])} measure(s) become searchable concepts."),
            "payload": {"table": {"name": ds["table_name"], "description": ds["description"], "grain": ds["grain"],
                                  "domain": ds["domain"]},
                        "columns": [{"name": c["name"], "source_name": c["source_name"], "role": c["role"],
                                     "unit": c["unit"], "description": c["description"], "dtype": c["dtype"],
                                     "synonyms": c["synonyms"], "derived": c.get("derived", False)} for c in cols],
                        "concepts": built["concepts"], "entity_domains": built["entity_domains"]},
            "evidence": {"row_count": ds["row_count"], "column_count": len(cols), "warnings": built["warnings"], "notices": built["notices"],
                         "suggestions": built["suggestions"], "aliases": built["aliases"],
                         "entity_domains": [d["name"] for d in built["entity_domains"]]},
        })
    rels = disc.discover_relationships(conn, ds, source)
    known = {r.key() for r in schema._relationships()}     # already approved (published datasets can be re-analysed)
    rels = [r for r in rels if f"{r['payload']['left_table']}:{r['payload']['left_expr']}="
                               f"{r['payload']['right_table']}:{r['payload']['right_expr']}" not in known]
    proposals += rels
    _record_stage(ds, "discover", "ok", f"{len(rels)} link(s) proposed in {time.monotonic() - t0:.1f}s")
    if getattr(settings, "catalog_llm_judge_enabled", False) and rels:
        try:
            from services import catalog_llm
            note = catalog_llm.judge_links(ds, rels)
            _record_stage(ds, "model_judge", "ok" if note.ok else "skipped", note.error or f"Annotated by {note.endpoint}")
        except Exception as exc:    # annotation only; never blocks the review
            _record_stage(ds, "model_judge", "skipped", str(exc))
    catalog_store.save_dataset(conn, ds)
    saved = catalog_store.replace_proposals(conn, dataset_id, proposals)
    out = detail(dataset_id)
    out["proposals"] = saved
    return out


def _default_grain(ds: dict) -> str:
    kinds = [c.get("key_kind") for c in ds["columns"] if c.get("include", True)]
    for k, label in (("address", "address"), ("house_id", "house"), ("tract_fips", "census tract"),
                     ("block_group_fips", "census block group"), ("zip", "ZIP code"),
                     ("county_fips", "county"), ("city", "city")):
        if k in kinds:
            return f"one row per {label}"
    return "one row per record"


# ---------------------------------------------------------------------------
# Decisions: approve / reject
# ---------------------------------------------------------------------------

def decide(proposal_id: str, decision: str, *, note: str = "", edits: dict | None = None) -> dict:
    conn = _conn()
    prop = catalog_store.get_proposal(conn, proposal_id)
    if not prop:
        raise OnboardingError("Proposal not found.", 404)
    if prop["status"] != "pending":
        raise OnboardingError(f"This proposal was already {prop['status']}.", 409)
    ds = _get(prop["dataset_id"])
    if decision == "reject":
        catalog_store.update_proposal(conn, proposal_id, status="rejected", decision_note=note)
        catalog_store.audit(conn, "reject_proposal", "proposal", proposal_id, {"title": prop["title"]})
    elif decision == "approve":
        if prop["kind"] == "table":
            _publish_table(conn, ds, prop, note)
        elif prop["kind"] == "relationship":
            _approve_relationship(conn, ds, prop, note, edits or {})
        else:
            raise OnboardingError("Unknown proposal type.")
    else:
        raise OnboardingError("Decision must be 'approve' or 'reject'.")
    out = detail(prop["dataset_id"])
    out["decided"] = catalog_store.get_proposal(conn, proposal_id)
    out["catalog_version"] = schema.catalog_version()
    return out


def _publish_table(conn, ds: dict, prop: dict, note: str) -> None:
    if ds["status"] != "draft":
        raise OnboardingError("This dataset is already published.")
    payload = prop["payload"]
    table = payload["table"]["name"]
    validate_table_name(conn, table)
    stg = ds["staging_table"]
    cols = [c for c in ds["columns"] if c.get("include", True)]
    select = ", ".join(qi(c["name"]) for c in cols)
    meta = TableMeta(name=table, description=ds["description"], grain=ds["grain"], agent_visible=True,
                     domain=ds["domain"], origin="upload", dataset_id=ds["dataset_id"])
    notes = [{"column_name": c["name"], "note": _column_note(c), "role": c["role"], "unit": c["unit"],
              "source_name": c["source_name"], "derived_from": "derived" if c.get("derived") else ""} for c in cols]
    ds = dict(ds, status="published", staging_table="")
    with schema.catalog_transaction():
        conn.execute(f"CREATE TABLE {qi(table)} AS SELECT {select} FROM {qi(stg)}")
        conn.execute(f"DROP TABLE {qi(stg)}")
        schema.register_table(meta, columns=notes, origin="upload", dataset_id=ds["dataset_id"], domain=ds["domain"])
        for key, definition in payload["concepts"].items():
            schema.register_concept(key, definition, origin="upload", dataset_id=ds["dataset_id"])
        for d in payload["entity_domains"]:
            schema.register_entity_domain(EntityDomain(
                name=d["name"], table=d["table"], column=d["column"], entity_type=d["entity_type"],
                description=d["description"], display_column=d.get("display_column"),
                match_mode=d.get("match_mode", "exact_or_prefix"), preferred_for=tuple(d.get("preferred_for", []))),
                origin="upload", dataset_id=ds["dataset_id"])
        _record_stage(ds, "publish", "ok", f"{table} registered with {len(payload['concepts'])} concept(s)")
        catalog_store.save_dataset(conn, ds)
        catalog_store.update_proposal(conn, prop["proposal_id"], status="approved", decision_note=note)
        catalog_store.audit(conn, "approve_table", "dataset", ds["dataset_id"], {"table": table})


def _materialize(conn, table: str, derive: dict) -> None:
    """Add the derived key column a relationship needs (once)."""
    col = derive["column"]
    have = {r[0] for r in conn.execute(f"DESCRIBE {qi(table)}").fetchall()}
    if col in have:
        return
    kind = derive["kind"]
    if kind == "sql":
        conn.execute(f"ALTER TABLE {qi(table)} ADD COLUMN {qi(col)} VARCHAR")
        conn.execute(f"UPDATE {qi(table)} SET {qi(col)} = {derive['sql']}")
    elif kind == "address_house":
        sel = ["rowid AS rid", f"{qi(derive['address_col'])} AS address",
               f"{qi(derive['zip_col'])} AS zip" if derive.get("zip_col") else "NULL AS zip",
               f"{qi(derive['city_col'])} AS city" if derive.get("city_col") else "NULL AS city"]
        rows = conn.execute(f"SELECT {', '.join(sel)} FROM {qi(table)} WHERE {qi(derive['address_col'])} IS NOT NULL").df()
        m = disc.match_addresses(rows, disc.load_houses(conn))
        conn.execute(f"ALTER TABLE {qi(table)} ADD COLUMN {qi(col)} VARCHAR")
        if not m.empty:
            _add_mapping(conn, table, col, m)
    else:
        raise OnboardingError("Unknown derived-column type.")


def _add_mapping(conn, table: str, col: str, mapping: pd.DataFrame) -> None:
    conn.register("__map", mapping[["rid", "house_id"]])
    try:
        conn.execute(f"UPDATE {qi(table)} SET {qi(col)} = m.house_id FROM __map m WHERE {qi(table)}.rowid = m.rid")
    finally:
        conn.unregister("__map")


def _approve_relationship(conn, ds: dict, prop: dict, note: str, edits: dict) -> None:
    if ds["status"] != "published":
        raise OnboardingError("Approve the table first, so the link has something to attach to.")
    pl = dict(prop["payload"])
    for key in ("cardinality", "note"):
        if edits.get(key):
            pl[key] = str(edits[key])[:300]
    if "preferred" in edits:
        pl["preferred"] = bool(edits["preferred"])
    if pl["cardinality"] not in {"one-to-one", "many-to-one", "one-to-many", "many-to-many"}:
        raise OnboardingError("Cardinality must be one-to-one, many-to-one, one-to-many or many-to-many.")
    rel = Relationship(left_table=pl["left_table"], left_expr=pl["left_expr"], right_table=pl["right_table"],
                       right_expr=pl["right_expr"], note=pl.get("note", ""), cardinality=pl["cardinality"],
                       confidence=pl.get("confidence", "medium"), bridge=False, preferred=bool(pl.get("preferred", True)),
                       grain_effect=pl.get("grain_effect", ""), origin="upload", dataset_id=ds["dataset_id"])
    derive = pl.get("derive")
    with schema.catalog_transaction():
        if derive:
            _materialize(conn, pl["left_table"], derive)
            catalog_store.upsert_column_note(
                conn, pl["left_table"], derive["column"], note=f"Derived key: {derive.get('rule', '')}",
                role="key", source_name="(derived)", derived_from=derive.get("from") or derive["kind"])
            if not any(c["name"] == derive["column"] for c in ds["columns"]):
                ds["columns"].append({"name": derive["column"], "source_name": "(derived)", "dtype": "text", "role": "key",
                                      "role_source": "rules", "key_kind": None, "description": derive.get("rule", ""),
                                      "synonyms": [], "unit": "", "include": True, "drafted_by": "", "derived": True,
                                      "stats": {}, "samples": []})
                catalog_store.save_dataset(conn, ds)
        schema.register_relationship(rel, origin="upload", dataset_id=ds["dataset_id"], evidence=prop["evidence"])
        catalog_store.update_proposal(conn, prop["proposal_id"], status="approved", decision_note=note, payload=pl)
        catalog_store.audit(conn, "approve_relationship", "relationship", rel.key(),
                            {"dataset": ds["dataset_id"], "cardinality": rel.cardinality})


# ---------------------------------------------------------------------------
# Revoke / retire / discard
# ---------------------------------------------------------------------------

def revoke_relationship(rel_key: str, reason: str = "") -> dict:
    conn = _conn()
    rel = next((r for r in schema._relationships() if r.key() == rel_key), None)
    if not rel:
        raise OnboardingError("Relationship not found.", 404)
    if rel.origin == "builtin":
        raise OnboardingError("Built-in relationships cannot be revoked from this page.", 403)
    ok = schema.revoke_relationship(rel_key, reason)
    if not ok:
        raise OnboardingError("Relationship not found.", 404)
    for p in catalog_store.list_proposals(conn, rel.dataset_id):
        pl = p["payload"]
        if p["kind"] == "relationship" and p["status"] == "approved" and \
                f"{pl['left_table']}:{pl['left_expr']}={pl['right_table']}:{pl['right_expr']}" == rel_key:
            catalog_store.update_proposal(conn, p["proposal_id"], status="revoked", decision_note=reason)
    return {"revoked": rel_key, "catalog_version": schema.catalog_version()}


def retire_dataset(dataset_id: str) -> dict:
    conn = _conn()
    ds = _get(dataset_id)
    if ds["status"] == "draft":
        return discard_draft(dataset_id)
    counts = schema.retire_dataset_objects(dataset_id)
    ds["status"] = "retired"
    catalog_store.save_dataset(conn, ds)
    return {"retired": dataset_id, **counts, "note": f"The data table '{ds['table_name']}' was kept but is hidden from the assistant.",
            "catalog_version": schema.catalog_version()}


def discard_draft(dataset_id: str) -> dict:
    conn = _conn()
    ds = _get(dataset_id)
    if ds["status"] != "draft":
        raise OnboardingError("Only drafts can be discarded.")
    if ds["staging_table"]:
        conn.execute(f"DROP TABLE IF EXISTS {qi(ds['staging_table'])}")
    ds["status"], ds["staging_table"] = "discarded", ""
    catalog_store.save_dataset(conn, ds)
    catalog_store.replace_proposals(conn, dataset_id, [])
    catalog_store.audit(conn, "discard_dataset", "dataset", dataset_id, {})
    return {"discarded": dataset_id}


# ---------------------------------------------------------------------------
# Read models for the API
# ---------------------------------------------------------------------------

def detail(dataset_id: str) -> dict:
    conn = _conn()
    ds = _get(dataset_id)
    table = ds["staging_table"] or ds["table_name"]
    ds["preview"] = _preview(conn, table) if ds["status"] in ("draft", "published") else []
    ds["available_enrichments"] = available_enrichments(ds)
    ds["proposals"] = catalog_store.list_proposals(conn, dataset_id)
    return ds


def list_datasets() -> list[dict]:
    out = []
    for d in catalog_store.list_datasets(_conn()):
        if d["status"] == "discarded":
            continue
        out.append({k: d[k] for k in ("dataset_id", "table_name", "title", "status", "row_count",
                                      "source_filename", "created_at")})
    return out


def draft_overlay(dataset_id: str) -> dict | None:
    """What the schema map draws for a dataset under review (a draft node + candidate edges)."""
    conn = _conn()
    ds = catalog_store.get_dataset(conn, dataset_id)
    if not ds or ds["status"] == "discarded":
        return None
    cols = [{"name": c["name"], "type": c["dtype"], "role": c["role"]} for c in ds["columns"] if c.get("include", True)]
    rels = []
    have = {c["name"] for c in cols}
    for p in catalog_store.list_proposals(conn, dataset_id):
        if p["kind"] == "relationship":
            pl = p["payload"]
            rels.append({"proposal_id": p["proposal_id"], "status": p["status"], "left_table": pl["left_table"],
                         "left_expr": pl["left_expr"], "right_table": pl["right_table"], "right_expr": pl["right_expr"],
                         "cardinality": pl["cardinality"], "confidence": pl["confidence"]})
            d = pl.get("derive")
            if d and p["status"] == "pending" and d["column"] not in have:     # a key column the approval will add
                cols.append({"name": d["column"], "type": "text", "role": "key"})
                have.add(d["column"])
    return {"dataset_id": dataset_id, "status": ds["status"],
            "table": {"name": ds["table_name"], "description": ds["description"], "grain": ds["grain"],
                      "domain": ds["domain"], "rows": ds["row_count"], "columns": cols},
            "relationships": rels}


def warm_vector_index() -> bool:
    """Best-effort: refresh the metadata embeddings now so the first chat turn is not slowed.

    The index also self-heals on the next retrieval (db/vector_store compares catalog versions),
    and the lexical fallback reads the live catalog, so failure here is harmless.
    """
    try:
        import db.vector_store as vs
        vs.ensure_schema_metadata_index()
        return True
    except Exception:
        return False

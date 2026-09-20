"""Deterministic relationship discovery for uploaded datasets.

Given an uploaded table (staged in DuckDB) this module

  1. profiles every column with SQL (types, nulls, distinct counts, value shape, samples);
  2. classifies columns (``detect_key_kind``: tract/county/ZIP/CBSA FIPS variants, city, address,
     lat/lon, ...) and guesses a role (key / label / measure / dimension / date / geo);
  3. proposes joins to *every* agent-visible table in the unified catalog by testing candidate
     column pairs - after a small, fixed set of value normalisations (zero-pad, first/last N
     characters, trim/case) - with value-containment SQL, and records the evidence a human needs:
     match rates in both directions, fan-out, cardinality, concrete before/after examples and the
     rows that did NOT match;
  4. proposes an address-level link to ``houses`` using the same normalisation tiers as
     ``db.duckdb_store.match_sold_to_houses``.

There is no LLM here and no question-specific branching: candidates come from the catalog's own
tables/columns, and every proposal is only ever *applied* after a person approves it.
"""
from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd

import db.schema_catalog as schema
from services.dataset_readers import qi

# ---------------------------------------------------------------------------
# Column profiling
# ---------------------------------------------------------------------------

def sql_family(sql_type: str) -> str:
    t = str(sql_type).upper()
    if t.startswith(("VARCHAR", "TEXT", "STRING", "CHAR")):
        return "text"
    if t.startswith(("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
                     "UINTEGER", "UBIGINT", "INT")):
        return "int"
    if t.startswith(("DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC")):
        return "float"
    if t.startswith("BOOL"):
        return "bool"
    if t.startswith(("DATE", "TIMESTAMP", "TIME")):
        return "date"
    return "other"


def profile_table(conn, table: str, sample_rows: int = 5000) -> dict[str, dict]:
    """SQL-based profile of every column of ``table`` (used for staging and for target tables)."""
    desc = conn.execute(f"DESCRIBE {qi(table)}").fetchall()
    cols = [(r[0], str(r[1])) for r in desc]
    n = int(conn.execute(f"SELECT COUNT(*) FROM {qi(table)}").fetchone()[0])
    aggs, layout = [], []
    for i, (c, t) in enumerate(cols):
        q, fam = qi(c), sql_family(t)
        aggs += [f"COUNT({q})", f"COUNT(DISTINCT {q})"]
        extra = 0
        if fam == "text":
            aggs += [f"MIN(LENGTH({q}))", f"MAX(LENGTH({q}))", f"AVG(LENGTH({q}))", f"MODE(LENGTH({q}))",
                     f"AVG(CASE WHEN regexp_matches({q}, '^[0-9]+$') THEN 1.0 ELSE 0.0 END)"]
            extra = 5
        elif fam in ("int", "float"):
            aggs += [f"MIN({q})", f"MAX({q})", f"AVG({q})"]
            extra = 3
        layout.append((c, t, fam, extra))
    row = conn.execute("SELECT " + ", ".join(aggs) + f" FROM {qi(table)}").fetchone() if aggs else ()
    head = conn.execute(f"SELECT * FROM {qi(table)} LIMIT {int(sample_rows)}").df() if cols else pd.DataFrame()
    out: dict[str, dict] = {}
    pos = 0
    for c, t, fam, extra in layout:
        nn, nd = int(row[pos] or 0), int(row[pos + 1] or 0)
        vals = row[pos + 2: pos + 2 + extra]
        pos += 2 + extra
        p: dict[str, Any] = {"name": c, "sql_type": t, "family": fam, "rows": n, "nulls": n - nn,
                             "null_pct": round(100 * (n - nn) / n, 1) if n else 0.0,
                             "distinct": nd, "unique_ratio": round(nd / nn, 4) if nn else 0.0}
        if fam == "text":
            p.update(min_len=vals[0], max_len=vals[1], avg_len=float(vals[2] or 0),
                     mode_len=vals[3], digit_frac=float(vals[4] or 0))
        elif fam in ("int", "float"):
            p.update(min=_num(vals[0]), max=_num(vals[1]), mean=_num(vals[2]))
        if c in head.columns:
            top = head[c].dropna().astype(str).value_counts().head(6)
            p["samples"] = [s[:40] for s in top.index.tolist()]
        else:
            p["samples"] = []
        out[c] = p
    return out


def _num(v: Any) -> Any:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Key kinds and roles
# ---------------------------------------------------------------------------

# Which "name group" (see NAME_GROUPS) a detected key kind belongs to.
KIND_GROUP = {"tract_fips": "tract", "block_group_fips": "tract", "block_fips": "tract",
              "county_fips": "county_fips", "zip": "zip", "cbsa_code": "cbsa", "state_fips": "state_fips",
              "state_abbr": "state", "city": "city", "house_id": "house", "parcel_id": "parcel"}

KEY_KINDS = set(KIND_GROUP) | {"code5", "lat", "lon", "address"}

# Column-name families; used only to rank candidate pairs, never to decide a join on its own.
NAME_GROUPS = {
    "tract": {"tract", "tract_fips", "tractfips", "tract_id", "geoid", "geoid10", "geoid20",
              "geoid_tract", "census_tract", "ct_fips", "fips_tract", "tract_geoid", "block_group", "blockgroup"},
    "county_fips": {"county_fips", "countyfips", "fips", "county_code", "cnty_fips", "fips_county"},
    "state_fips": {"state_fips", "statefips", "statefp", "state_code_fips"},
    "zip": {"zip", "zipcode", "zip_code", "postal", "postal_code", "postcode", "zip5"},
    "state": {"state", "state_abbr", "state_code", "st", "stusps", "state_abbrev"},
    "city": {"city", "town", "municipality", "city_name", "place", "place_name", "crime_city"},
    "house": {"house_id"},
    "cbsa": {"cbsa", "cbsa_code", "msa", "msa_code", "cbsa_fips", "metro", "cbsa_id"},
    "parcel": {"parid", "parcel", "parcel_id", "apn", "parcelid"},
}


def name_group(name: str) -> str | None:
    n = name.lower()
    return next((g for g, names in NAME_GROUPS.items() if n in names), None)


def name_score(src: str, tgt: str) -> float:
    a, b = src.lower(), tgt.lower()
    if a == b:
        return 1.0
    ga, gb = name_group(a), name_group(b)
    if ga and ga == gb:
        return 0.9
    ta, tb = set(a.split("_")), set(b.split("_"))
    generic = {"id", "code", "name", "value", "num", "no", "number", "key"}
    if (ta & tb) - generic:
        return 0.6
    return 0.0


def _sample_match(p: dict, pattern: str) -> float:
    s = p.get("samples") or []
    return sum(1 for v in s if re.fullmatch(pattern, v)) / len(s) if s else 0.0


def detect_key_kind(name: str, p: dict) -> str | None:
    n, fam = name.lower(), p.get("family")
    if fam == "float":
        lo, hi = p.get("min"), p.get("max")
        if lo is not None and re.fullmatch(r"(lat|latitude|lat_deg|ycoord|y)", n) and -90 <= lo and hi <= 90:
            return "lat"
        if lo is not None and re.fullmatch(r"(lon|lng|long|longitude|lon_deg|xcoord|x)", n) and -180 <= lo and hi <= 180:
            return "lon"
        return None
    if fam != "text":
        return None
    dig, mode = p.get("digit_frac", 0), p.get("mode_len")
    if _sample_match(p, r"\d{7}US\d{11}") >= 0.8:
        return "tract_fips"
    if dig >= 0.95 and mode:
        if mode == 11:
            return "tract_fips"
        if mode == 12:
            return "block_group_fips"
        if mode == 15:
            return "block_fips"
        if mode == 5:
            if re.search(r"zip|postal", n):
                return "zip"
            if re.search(r"county|fips|cnty", n):
                return "county_fips"
            if re.search(r"cbsa|msa|metro", n):
                return "cbsa_code"
            return "code5"
        if mode == 2 and re.search(r"state|statefp", n):
            return "state_fips"
    if re.search(r"zip|postal", n) and _sample_match(p, r"\d{5}(-?\d{4})?") >= 0.8:
        return "zip"
    if re.fullmatch(r"(state|st|state_abbr|state_code|stusps)", n) and (p.get("avg_len") or 0) <= 2.2:
        return "state_abbr"
    if re.search(r"(^|_)(city|town|municipality)($|_)", n):
        return "city"
    if n == "house_id" or (mode == 12 and _sample_match(p, r"[0-9a-f]{12}") >= 0.8):
        return "house_id"
    if re.search(r"parcel|parid|apn", n):
        return "parcel_id"
    if re.search(r"address|street|addr", n) and (p.get("avg_len") or 0) >= 8 \
            and _sample_match(p, r"\d+\s+.+") >= 0.5:
        return "address"
    return None


_SKIP_JOIN_NAMES = {"year", "yr", "month", "day", "quarter", "week", "date"}
_GEO_NAMES = {"lat", "lon", "geometry_json", "min_lon", "min_lat", "max_lon", "max_lat",
              "centroid_lat", "centroid_lon"}


def guess_role(name: str, p: dict, kind: str | None) -> str:
    n, fam = name.lower(), p.get("family")
    if n in _GEO_NAMES or kind in ("lat", "lon"):
        return "geo"
    if fam == "date":
        return "date"
    if kind in KIND_GROUP or kind == "code5":
        return "key"
    if kind in ("address", "city"):
        return "label"
    if kind == "state_abbr":
        return "dimension"
    rows, distinct = p.get("rows", 0), p.get("distinct", 0)
    if fam in ("int", "float"):
        if fam == "int" and (re.search(r"(^|_)id$", n) or (rows > 1 and p.get("unique_ratio") == 1.0 and distinct == rows)):
            return "key"
        lo, hi = p.get("min"), p.get("max")
        if n in ("year", "yr") or (fam == "int" and lo is not None and 1900 <= lo and hi <= 2100 and distinct <= 200):
            return "dimension"
        if fam == "int" and distinct <= 10 and p.get("unique_ratio", 1) < 0.05:
            return "dimension"
        return "measure"
    if fam == "bool":
        return "dimension"
    if fam == "text":
        if (p.get("avg_len") or 0) > 60:
            return "other"
        if p.get("unique_ratio", 0) >= 0.9 and re.search(r"(id|code|key|number|no)$", n):
            return "key"
        if distinct <= 50 and p.get("unique_ratio", 1) < 0.2:
            return "dimension"
        return "label"
    return "other"


# ---------------------------------------------------------------------------
# Value normalisations ("transforms") and their SQL
# ---------------------------------------------------------------------------

_KEY_WIDTHS = {5, 11, 12, 15}
_SIMPLICITY = {"identity": 0, "trim": 1, "zfill": 2, "left": 3, "right": 3, "upper": 4, "lower": 4, "ci": 5}


def transform_sql(t: dict, ref: str) -> str:
    k = t["kind"]
    if k == "identity":
        return ref
    if k == "trim":
        return f"TRIM({ref})"
    if k in ("upper", "ci"):
        return f"UPPER(TRIM({ref}))"
    if k == "lower":
        return f"LOWER(TRIM({ref}))"
    if k == "zfill":
        return f"LPAD(TRIM({ref}), {int(t['width'])}, '0')"
    if k == "left":
        return f"LEFT(TRIM({ref}), {int(t['width'])})"
    if k == "right":
        return f"RIGHT(TRIM({ref}), {int(t['width'])})"
    raise ValueError(f"Unknown transform '{k}'")


def target_sql(t: dict, ref: str) -> str:
    """Target-side expression a transform needs; only the case-insensitive one normalizes the target too."""
    return f"UPPER(TRIM({ref}))" if t["kind"] == "ci" else ref


def transform_key(t: dict) -> str:
    return t["kind"] + (str(t["width"]) if t.get("width") else "")


def transform_rule_text(t: dict, col: str) -> str:
    k, w = t["kind"], t.get("width")
    return {
        "identity": f"Compare {col} as it is",
        "trim": f"Trim extra spaces from {col}",
        "ci": f"Match {col} ignoring extra spaces and letter case (both sides are normalized)",
        "upper": f"Trim {col} and convert it to UPPER CASE",
        "lower": f"Trim {col} and convert it to lower case",
        "zfill": f"Pad {col} with leading zeros to {w} digits",
        "left": f"Keep the first {w} characters of {col}",
        "right": f"Keep the last {w} characters of {col}",
    }[k]


def _better(cand: tuple, best: tuple, tolerance: float = 0.05) -> bool:
    """Prefer clearly higher containment; within ``tolerance`` prefer the simpler normalisation."""
    if cand[0] > best[0] + tolerance:
        return True
    return abs(cand[0] - best[0]) <= tolerance and _SIMPLICITY[cand[1]["kind"]] < _SIMPLICITY[best[1]["kind"]]


def make_transforms(src: dict, tgt: dict) -> list[dict]:
    """Normalisations worth testing for a (source column, target column) pair."""
    out = [{"kind": "identity"}]
    if src["family"] == "text" and tgt["family"] == "text":
        if tgt.get("digit_frac", 0) >= 0.95 and tgt.get("mode_len"):
            w = int(tgt["mode_len"])
            smin, smax, smode = src.get("min_len") or 0, src.get("max_len") or 0, src.get("mode_len") or 0
            if w in _KEY_WIDTHS:      # complete identifiers only (county 5, tract 11, block group 12, block 15)
                if src.get("digit_frac", 0) >= 0.95 and smax < w and smin >= w - 2:
                    out.append({"kind": "zfill", "width": w})
                if smode and smode > w:
                    out += [{"kind": "left", "width": w}, {"kind": "right", "width": w}]
        else:
            out += [{"kind": "trim"}, {"kind": "upper"}, {"kind": "lower"}, {"kind": "ci"}]
    return out


# ---------------------------------------------------------------------------
# Evidence SQL
# ---------------------------------------------------------------------------

def eval_pair(conn, src: str, sx: str, tgt: str, tx: str) -> dict:
    """Value-containment statistics for ``sx`` (SQL over src) vs ``tx`` (SQL over tgt)."""
    sql = f"""
    WITH l AS (SELECT {sx} AS k, COUNT(*) AS n FROM {qi(src)} WHERE {sx} IS NOT NULL GROUP BY 1),
         r AS (SELECT {tx} AS k, COUNT(*) AS n FROM {qi(tgt)} WHERE {tx} IS NOT NULL GROUP BY 1),
         j AS (SELECT l.k AS k, l.n AS ln, r.n AS rn FROM l JOIN r ON l.k = r.k)
    SELECT (SELECT COUNT(*) FROM l), (SELECT COALESCE(SUM(n), 0) FROM l),
           (SELECT COUNT(*) FROM r), (SELECT COALESCE(SUM(n), 0) FROM r),
           (SELECT COUNT(*) FROM j), (SELECT COALESCE(SUM(ln), 0) FROM j), (SELECT COALESCE(SUM(rn), 0) FROM j),
           (SELECT COALESCE(MAX(rn), 0) FROM j), (SELECT COALESCE(MAX(n), 0) FROM l),
           (SELECT COALESCE(MAX(n), 0) FROM r)"""
    v = [int(x or 0) for x in conn.execute(sql).fetchone()]
    l_distinct, l_rows, r_distinct, r_rows, matched, l_rows_m, r_rows_m, r_fan_max, l_dup, r_dup = v
    return {
        "left_distinct": l_distinct, "left_rows": l_rows, "right_distinct": r_distinct, "right_rows": r_rows,
        "matched_distinct": matched, "left_rows_matched": l_rows_m, "right_rows_matched": r_rows_m,
        "left_match_pct": round(100 * matched / l_distinct, 1) if l_distinct else 0.0,
        "right_covered_pct": round(100 * r_rows_m / r_rows, 1) if r_rows else 0.0,
        "left_rows_linked_pct": round(100 * l_rows_m / l_rows, 1) if l_rows else 0.0,
        "fanout_avg": round(r_rows_m / matched, 2) if matched else 0.0, "fanout_max": r_fan_max,
        "left_unique": l_dup <= 1, "right_unique": r_dup <= 1,
    }


def cardinality_of(stats: dict) -> str:
    lu, ru = stats["left_unique"], stats["right_unique"]
    if lu and ru:
        return "one-to-one"
    if ru:
        return "many-to-one"
    if lu:
        return "one-to-many"
    return "many-to-many"


def containment_score(stats: dict) -> float:
    """Best direction: uploaded values found in the target, or target ROWS covered by the upload."""
    return max(stats["left_match_pct"], stats["right_covered_pct"]) / 100.0


_DISPLAY_PRIORITY = ["address", "full_address", "name", "title", "city", "county_name", "state", "state_name",
                     "zip", "tract_fips", "house_id", "msa_code", "cbsa_title", "incident_id", "sale_id"]


def display_columns(table: str, join_col: str, limit: int = 4) -> list[str]:
    live = [c for c, _ in schema._live_columns(table)]
    meta = schema._tables().get(table)
    hidden = set(meta.hidden_columns) if meta else set()
    live = [c for c in live if c not in hidden and not c.endswith("_json")]
    picked = [c for c in _DISPLAY_PRIORITY if c in live and c != join_col][:limit - 1]
    for c in live:
        if len(picked) >= limit - 1:
            break
        if c not in picked and c != join_col:
            picked.append(c)
    return [join_col] + picked if join_col in live else picked[:limit]


def build_examples(conn, src: str, raw_col: str, sx: str, tgt: str, tcol: str, ctx_cols: list[str],
                   n: int = 5, tx: str | None = None) -> dict:
    """Concrete rows that show how the mapping works, plus values that found no match."""
    tx = tx or qi(tcol)
    base = f"""
    WITH l AS (SELECT {sx} AS k, MIN({qi(raw_col)}) AS raw, COUNT(*) AS n FROM {qi(src)}
               WHERE {sx} IS NOT NULL GROUP BY 1),
         r AS (SELECT DISTINCT {tx} AS k FROM {qi(tgt)} WHERE {tx} IS NOT NULL)"""
    matched = conn.execute(base + f" SELECT l.k, l.raw, l.n FROM l JOIN r ON l.k = r.k "
                                  f"ORDER BY l.n DESC, l.k LIMIT {n}").fetchall()
    unmatched = conn.execute(base + f" SELECT l.k, l.raw, l.n FROM l WHERE l.k NOT IN (SELECT k FROM r) "
                                    f"ORDER BY l.n DESC, l.k LIMIT {n}").fetchall()
    disp = display_columns(tgt, tcol)
    examples = []
    for k, raw, cnt in matched:
        rows = conn.execute(
            f"SELECT {', '.join(qi(c) for c in disp)} FROM {qi(tgt)} WHERE {tx} = ? LIMIT 3", [k]).fetchall()
        total = int(conn.execute(f"SELECT COUNT(*) FROM {qi(tgt)} WHERE {tx} = ?", [k]).fetchone()[0])
        ctx = {}
        if ctx_cols:
            cr = conn.execute(f"SELECT {', '.join(qi(c) for c in ctx_cols)} FROM {qi(src)} WHERE {sx} = ? LIMIT 1", [k]).fetchone()
            ctx = {c: _cell(v) for c, v in zip(ctx_cols, cr)} if cr else {}
        examples.append({"raw": _cell(raw), "normalized": _cell(k), "your_rows": int(cnt), "context": ctx,
                         "match_count": total,
                         "matches": [{c: _cell(v) for c, v in zip(disp, r)} for r in rows]})
    return {"examples": examples, "unmatched": [{"raw": _cell(raw), "normalized": _cell(k), "your_rows": int(cnt)}
                                                for k, raw, cnt in unmatched],
            "target_columns": disp}


def _cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float, bool, str)):
        return v
    return str(v)


# ---------------------------------------------------------------------------
# Address-level link to houses (same normalisation tiers as match_sold_to_houses)
# ---------------------------------------------------------------------------

_SUFFIX_MAP = {"street": "st", "avenue": "av", "ave": "av", "boulevard": "blvd", "drive": "dr", "road": "rd",
               "court": "ct", "lane": "ln", "place": "pl", "way": "wy", "terrace": "ter", "circle": "cir",
               "trail": "trl", "highway": "hwy", "parkway": "pkwy", "pike": "pk"}
_DIR = {"north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
        "southeast": "se", "southwest": "sw"}


def norm_address(addr: Any) -> str:
    if addr is None or (isinstance(addr, float) and pd.isna(addr)):
        return ""
    a = re.sub(r"\s+", " ", re.sub(r"[.,#]", " ", str(addr).lower().strip())).strip()
    return " ".join(_DIR.get(t, _SUFFIX_MAP.get(t, t)) for t in a.split())


def _house_num(norm: str) -> str:
    m = re.match(r"^(\d+\w*)", norm.strip())
    return m.group(1) if m else ""


def _street_word(norm: str) -> str:
    tokens = norm.split()
    for i, t in enumerate(tokens):
        if i == 0 and re.match(r"^\d", t):
            continue
        if t in _DIR.values():
            continue
        return t
    return tokens[1] if len(tokens) > 1 else ""


def match_addresses(rows: pd.DataFrame, houses: pd.DataFrame) -> pd.DataFrame:
    """Map uploaded rows (``rid``, ``address`` and optional ``zip``/``city``) to ``houses.house_id``.

    Tier 1: normalised address + ZIP5.  Tier 2: house number + first street word + ZIP5.
    Tier 3 (only when the row has no ZIP): house number + street word + first 3 letters of city.
    """
    if houses.empty or rows.empty:
        return pd.DataFrame(columns=["rid", "house_id", "tier"])
    h = houses.dropna(subset=["address"]).copy()
    h["norm"] = h["address"].map(norm_address)
    h["zip5"] = h["zip"].astype(str).str[:5]
    h["hn"], h["sw"] = h["norm"].map(_house_num), h["norm"].map(_street_word)
    h["c3"] = h["city"].astype(str).str[:3].str.lower()
    t1, t2, t3 = {}, {}, {}
    for r in h.itertuples(index=False):
        t1.setdefault((r.norm, r.zip5), r.house_id)
        t2.setdefault((r.hn, r.sw, r.zip5), r.house_id)
        t3.setdefault((r.hn, r.sw, r.c3), r.house_id)
    out = []
    for r in rows.itertuples(index=False):
        norm = norm_address(r.address)
        if not norm:
            continue
        z5 = str(r.zip)[:5] if getattr(r, "zip", None) not in (None, "", "nan") and not pd.isna(r.zip) else ""
        hn, sw = _house_num(norm), _street_word(norm)
        hit = None
        if z5:
            if (norm, z5) in t1:
                hit = (t1[(norm, z5)], 1)
            elif (hn, sw, z5) in t2:
                hit = (t2[(hn, sw, z5)], 2)
        else:
            c3 = str(getattr(r, "city", "") or "")[:3].lower()
            if c3 and (hn, sw, c3) in t3:
                hit = (t3[(hn, sw, c3)], 3)
        if hit:
            out.append((r.rid, hit[0], hit[1]))
    return pd.DataFrame(out, columns=["rid", "house_id", "tier"])


def load_houses(conn) -> pd.DataFrame:
    return conn.execute("SELECT house_id, address, city, state, zip FROM houses WHERE address IS NOT NULL").df()


def address_columns(columns: list[dict]) -> dict[str, str | None]:
    """Pick the address / zip / city columns of a dataset from its column records."""
    by_kind: dict[str, str] = {}
    for c in columns:
        if c.get("include", True) and c.get("key_kind") in ("address", "zip", "city"):
            by_kind.setdefault(c["key_kind"], c["name"])
    return {"address": by_kind.get("address"), "zip": by_kind.get("zip"), "city": by_kind.get("city")}


def _address_proposal(conn, dataset: dict, source_table: str) -> dict | None:
    cols = address_columns(dataset["columns"])
    if not cols["address"] or not (cols["zip"] or cols["city"]):
        return None
    if "houses" not in schema._tables():
        return None
    houses = load_houses(conn)
    if houses.empty:
        return None
    sel = [f"rowid AS rid", f"{qi(cols['address'])} AS address"]
    sel.append(f"{qi(cols['zip'])} AS zip" if cols["zip"] else "NULL AS zip")
    sel.append(f"{qi(cols['city'])} AS city" if cols["city"] else "NULL AS city")
    rows = conn.execute(f"SELECT {', '.join(sel)} FROM {qi(source_table)} WHERE {qi(cols['address'])} IS NOT NULL").df()
    total = int(conn.execute(f"SELECT COUNT(*) FROM {qi(source_table)}").fetchone()[0])
    m = match_addresses(rows, houses)
    if m.empty:
        return None
    houses_idx = houses.set_index("house_id")
    tiers = m["tier"].value_counts().to_dict()
    per_house = m.groupby("house_id").size()
    n_matched, n_houses = len(m), int(per_house.size)
    info = rows.set_index("rid")
    examples = []
    for r in m.sort_values(["tier", "rid"]).head(5).itertuples(index=False):
        hrow = houses_idx.loc[r.house_id]
        src = info.loc[r.rid]
        examples.append({
            "raw": f"{src['address']}" + (f", {src['zip']}" if pd.notna(src["zip"]) else ""),
            "normalized": f"{norm_address(src['address'])} | {str(src['zip'])[:5] if pd.notna(src['zip']) else str(src['city'])[:3].lower()}",
            "your_rows": 1, "context": {}, "match_count": 1,
            "tier": int(r.tier),
            "matches": [{"address": hrow["address"], "city": hrow["city"], "zip": hrow["zip"], "house_id": r.house_id}]})
    matched_ids = set(m["rid"])
    um = rows[~rows["rid"].isin(matched_ids)].head(5)
    unmatched = [{"raw": str(a), "normalized": norm_address(a), "your_rows": 1} for a in um["address"]]
    card = "one-to-one" if n_matched == n_houses else "many-to-one"
    stats = {"left_rows": total, "left_rows_matched": n_matched, "left_match_pct": round(100 * n_matched / max(total, 1), 1),
             "right_rows": len(houses), "right_rows_matched": n_houses,
             "right_covered_pct": round(100 * n_houses / len(houses), 1), "matched_distinct": n_houses,
             "fanout_avg": round(n_matched / n_houses, 2), "fanout_max": int(per_house.max()),
             "tiers": {str(k): int(v) for k, v in tiers.items()}, "left_unique": card == "one-to-one",
             "right_unique": True, "left_distinct": total, "right_distinct": len(houses),
             "left_rows_linked_pct": round(100 * n_matched / max(total, 1), 1)}
    warnings = []
    if stats["left_match_pct"] < 60:
        warnings.append("Fewer than 60% of your rows matched a house; only rows whose address matches a saved house are linked.")
    if 2 in tiers or 3 in tiers:
        warnings.append("Some rows matched on house number + street name only (tiers 2-3); check the examples.")
    return {
        "kind": "relationship",
        "title": f"{dataset['table_name']} rows link to houses by address",
        "summary": (f"Each {dataset['table_name']} row is linked to the saved house with the same street address "
                    f"and ZIP (or city). {n_matched:,} of your {total:,} rows match {n_houses:,} of {len(houses):,} houses."),
        "group_key": "address_house",
        "payload": {"left_table": dataset["table_name"], "left_expr": "house_id", "right_table": "houses",
                    "right_expr": "house_id", "cardinality": card, "confidence": "high" if stats["left_match_pct"] >= 60 else "medium",
                    "preferred": True, "grain_effect": f"{dataset['table_name']} row -> house",
                    "note": "Address-level link: normalised street address + ZIP (same tiers as sold-home matching).",
                    "derive": {"kind": "address_house", "column": "house_id", "address_col": cols["address"],
                               "zip_col": cols["zip"], "city_col": cols["city"],
                               "rule": "Normalise the street address (abbreviate street types and directions, drop punctuation), then match on address + ZIP5; "
                                       "fall back to house number + first street word + ZIP5."},
                    "score": 1.0 if stats["left_match_pct"] >= 60 else 0.5},
        "evidence": {"stats": stats, "transform": {"kind": "address", "label": "Normalised street address + ZIP"},
                     "rule_text": "Normalise the street address (abbreviate street types and directions, drop punctuation), then match on address + ZIP5.",
                     "examples": examples, "unmatched": unmatched, "target": "houses.house_id", "warnings": warnings,
                     "target_columns": ["address", "city", "zip", "house_id"]},
    }


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _target_columns(conn, exclude: set[str]) -> list[dict]:
    out = []
    for name, meta in schema._tables().items():
        if name in exclude or not meta.agent_visible:
            continue
        for cname, ctype in schema._live_columns(name):
            fam = sql_family(ctype)
            if fam not in ("text", "int") or cname in meta.hidden_columns or cname.endswith("_json"):
                continue
            out.append({"table": name, "column": cname, "sql_type": ctype, "family": fam})
    return out


def _target_profile(conn, cache: dict, table: str, col: str, fam: str) -> dict:
    key = (table, col)
    if key in cache:
        return cache[key]
    q = qi(col)
    if fam == "text":
        row = conn.execute(
            f"SELECT COUNT(*), COUNT({q}), COUNT(DISTINCT {q}), MIN(LENGTH({q})), MAX(LENGTH({q})), "
            f"MODE(LENGTH({q})), AVG(CASE WHEN regexp_matches({q}, '^[0-9]+$') THEN 1.0 ELSE 0.0 END) "
            f"FROM {qi(table)}").fetchone()
        p = {"family": "text", "rows": row[0], "distinct": row[2], "min_len": row[3], "max_len": row[4],
             "mode_len": row[5], "digit_frac": float(row[6] or 0)}
    else:
        row = conn.execute(f"SELECT COUNT(*), COUNT({q}), COUNT(DISTINCT {q}) FROM {qi(table)}").fetchone()
        p = {"family": "int", "rows": row[0], "distinct": row[2]}
    cache[key] = p
    return p


def _explain(transform: dict, left: str, right_table: str, right_col: str) -> str:
    k, w = transform["kind"], transform.get("width")
    how = {
        "identity": f"equals {left}",
        "trim": f"equals {left} ignoring extra spaces",
        "upper": f"equals {left} ignoring extra spaces and letter case",
        "lower": f"equals {left} ignoring extra spaces and letter case",
        "ci": f"equals {left} ignoring extra spaces and letter case",
        "zfill": f"equals {left} padded with leading zeros to {w} digits",
        "left": f"equals the first {w} characters of {left}",
        "right": f"equals the last {w} characters of {left}",
    }[k]
    return f"{right_table}.{right_col} {how}"


def discover_relationships(conn, dataset: dict, source_table: str, *, max_evals: int = 60,
                           budget_s: float = 25.0) -> list[dict]:
    """Return relationship proposals (dicts ready for ``catalog_store.replace_proposals``)."""
    started = time.monotonic()
    final = dataset["table_name"]
    included = [c for c in dataset["columns"] if c.get("include", True)]
    prof = profile_table(conn, source_table)
    targets = _target_columns(conn, exclude={final, source_table})
    tcache: dict = {}
    non_empty: dict[str, int] = {}
    taken_cols = {c["name"] for c in included}
    derived_names: dict[tuple, str] = {}
    proposals: dict[str, dict] = {}
    evals = 0
    # Context shown next to each example: measures first, then genuine labels (city / address), never key fragments.
    ctx_cols = ([c["name"] for c in included if c.get("role") == "measure"]
                + [c["name"] for c in included if c.get("role") == "label" and c.get("key_kind") in ("city", "address")])[:3]

    for col in included:
        cname = col["name"]
        p = prof.get(cname)
        if not p or p["family"] not in ("text", "int"):
            continue
        if col.get("role") in ("measure", "geo", "date", "other"):
            continue
        if p["distinct"] < 2 or cname in _SKIP_JOIN_NAMES:
            continue
        kind = col.get("key_kind") or detect_key_kind(cname, p)
        if kind in ("address",):
            continue
        group = KIND_GROUP.get(kind or "") or name_group(cname)
        cands = []
        for t in targets:
            if t["family"] != p["family"]:
                continue
            ns = name_score(cname, t["column"])
            same_group = bool(group and name_group(t["column"]) == group)
            if ns >= 0.6 or same_group:
                cands.append((max(ns, 0.9 if same_group else 0.0), t))
        cands.sort(key=lambda x: (-x[0], x[1]["table"], x[1]["column"]))
        for ns, t in cands[:10]:
            if evals >= max_evals or time.monotonic() - started > budget_s:
                break
            tp = _target_profile(conn, tcache, t["table"], t["column"], t["family"])
            if not tp["rows"] or tp["distinct"] < 2:
                continue
            if tp.get("digit_frac", 0) >= 0.95 and (tp.get("mode_len") or 0) <= 3:
                continue    # component codes (county 001, tract code 040100) repeat across states: not keys
            best = None   # (score, transform, stats)
            for tr in make_transforms(p, tp):
                evals += 1
                st = eval_pair(conn, source_table, transform_sql(tr, qi(cname)), t["table"], target_sql(tr, qi(t["column"])))
                cand = (containment_score(st), tr, st)
                if best is None or _better(cand, best):
                    best = cand
            if not best:
                continue
            score, tr, st = best
            if st["matched_distinct"] < 1 or not (score >= 0.5 or (ns >= 0.9 and score >= 0.2)):
                continue
            card = cardinality_of(st)
            if card == "many-to-many" and (ns < 0.9 or kind in ("state_fips", "state_abbr")):
                continue    # too coarse to be a useful join
            conf = "high" if score >= 0.9 and (st["matched_distinct"] >= 3 or ns >= 0.9) else ("medium" if score >= 0.6 else "low")
            derive = None
            left_expr = cname
            if tr["kind"] != "identity":
                dkey = (cname, transform_key(tr))
                if dkey not in derived_names:
                    base = t["column"]
                    name, i = base, 2
                    while name in taken_cols:
                        name = f"{base}_norm" if i == 2 else f"{base}_norm{i}"
                        i += 1
                    derived_names[dkey] = name
                    taken_cols.add(name)
                left_expr = derived_names[dkey]
                derive = {"kind": "sql", "column": left_expr, "from": cname, "transform": tr,
                          "sql": transform_sql(tr, qi(cname)), "rule": transform_rule_text(tr, cname)}
            warnings = []
            if card == "many-to-many":
                warnings.append("Rows on both sides repeat this key, so joining multiplies rows (for example, several "
                                "records of yours per house). Averages over the joined result weight each record equally.")
            if st["fanout_max"] > 100:
                warnings.append(f"One key matches up to {st['fanout_max']:,} rows in {t['table']}; averages over the joined result are weighted by that repetition.")
            if score < 0.6:
                warnings.append("Fewer than 60% of values match; check that the key's format is what you expect.")
            if tr["kind"] != "identity":
                warnings.append(f"A new column '{left_expr}' will be added to {final} ({transform_rule_text(tr, cname).lower()}).")
            ex = build_examples(conn, source_table, cname, transform_sql(tr, qi(cname)), t["table"], t["column"], ctx_cols,
                                tx=target_sql(tr, qi(t["column"])))
            direction = ("of your distinct values exist in the target" if st["left_match_pct"] >= st["right_covered_pct"]
                         else "of target rows have a match in your data")
            right_expr = target_sql(tr, t["column"])
            rel_key = f"{final}:{left_expr}={t['table']}:{right_expr}"
            grp = f"{cname}|{transform_key(tr)}"
            proposals[rel_key] = {
                "kind": "relationship",
                "title": f"{final}.{left_expr} joins {t['table']}.{right_expr}",
                "summary": (f"Each {final} row links to the {t['table']} rows where "
                            f"{_explain(tr, cname, t['table'], right_expr)}. "
                            f"{max(st['left_match_pct'], st['right_covered_pct']):g}% {direction}."),
                "group_key": grp,
                "payload": {"left_table": final, "left_expr": left_expr, "right_table": t["table"],
                            "right_expr": right_expr, "cardinality": card, "confidence": conf,
                            "preferred": (conf in ("high", "medium") and card != "many-to-many") or (conf == "high" and ns >= 0.9),
                            "grain_effect": f"{final} row -> {t['table']} row(s)",
                            "note": f"{_explain(tr, cname, t['table'], right_expr)}.",
                            "derive": derive, "score": round(score, 3), "name_score": ns},
                "evidence": {"stats": st, "transform": {**tr, "label": transform_rule_text(tr, cname)},
                             "rule_text": transform_rule_text(tr, cname), "examples": ex["examples"],
                             "unmatched": ex["unmatched"], "target": f"{t['table']}.{right_expr}",
                             "target_columns": ex["target_columns"], "warnings": warnings,
                             "source_column": cname},
            }

    addr = _address_proposal(conn, dataset, source_table)
    if addr:
        proposals[f"{final}:house_id=houses:house_id"] = addr

    order = sorted(proposals.values(), key=lambda p: (p["group_key"], -p["payload"]["score"], p["title"]))
    return order

# 🏠 Real Estate Intelligence

A local, AI-powered map app for analyzing houses with FEMA National Risk Index data, Census demographics, Redfin listings, severity-weighted crime data, and an LLM chat interface — all running on your machine.

## General Chat: Agent Architecture & Design Philosophy

**The LLM writes text; deterministic code decides what's true and what's allowed.**

For every analytical question, *what's relevant* — which tables, which join
path, which entities, which operation — is decided entirely by deterministic
code reading `db/schema_catalog.py`, before any SQL exists. The LLM's job is
narrower and comes later: turn an already-decided, already-validated plan into
one SQL statement, or turn executed results into prose. Every LLM output along
the way is checked by code against that same plan — never by another LLM call
grading the first one.

```
request text
  -> InputGuardrail                                  [deterministic, Outlines/regex]
       prompt injection screening, jailbreak detection, length/null-byte sanitization
  -> build_query_plan()                              [deterministic]
       concept + operation match, entity resolution, relationship-path search
  -> QueryPlan (tables, joins, filters, operation) — now authoritative
  -> generate_program() / generate_sql()             [LLM]
       writes Python code agent step or SELECT statement from the plan text
  -> CodeAgentGuardrail (AST) + SQL validation        [deterministic, AST/Outlines]
       AST sandboxing, read-only SQL check (SELECT/WITH only), plan-conformance check
       - valid -> execute against DuckDB
       - invalid/empty after 3 attempts -> _compile_sql_from_plan()  [deterministic]
             same QueryPlan, template compiler, no LLM -> execute against DuckDB
  -> final answer                                      [LLM]
       writes prose from the executed evidence only
  -> response_validator.py + OutputGroundingGuardrail  [deterministic]
       flags ungrounded claims, enforces metric bounds (Walk/Bike scores in [0, 100])
```

| Stage | Deterministic or LLM |
|---|---|
| Input sanitization & prompt injection guard | Deterministic (`InputGuardrail`) |
| Routing — tables, joins, entities, operation | Deterministic (`build_query_plan`) |
| Code program / SQL generation | LLM |
| Code AST sandboxing & SQL safety checks | Deterministic (`CodeAgentGuardrail`) |
| SQL plan-conformance check | Deterministic (`_validate_sql_against_plan`) |
| Fallback SQL compilation | Deterministic (`_compile_sql_from_plan`) |
| Final answer prose | LLM |
| Reply-vs-evidence check & score bounding | Deterministic (`OutputGroundingGuardrail`) |

The split holds because every LLM call on this path — SQL generation,
orchestration, and the final answer — runs through the same local, quantized
model at `temperature=0.0` (see **LLM: run `llama-server`** above). Two things
follow from that: retrying that model with an unchanged prompt returns the same
result, so a safety net that just asks it again isn't a safety net; and the
same weights writing an answer can't be trusted to independently grade that
answer, so every "is this correct/safe" check is code, never a second model
call. **See [`AGENT_ARCHITECTURE.md`](AGENT_ARCHITECTURE.md) for the full
pipeline mechanics, every validation rule, and the reasoning behind where each
line is drawn.**

### Data-model retrieval

Two separate retrieval steps feed the pipeline above. `general_chat.data_model_rag`
(`db/vector_store.py`) runs once per turn, before the orchestrator decides what
to do — it searches a dedicated Chroma collection (`data_model_metadata`) using
both the question and the structured plan, so the orchestrator has context
before choosing tool calls. This is a **retrieval accelerator, not the
authoritative schema**: live DuckDB `DESCRIBE` output and row counts remain the
source of truth, and if the embedding service is unavailable, retrieval falls
back to lexical matching rather than failing the turn. Separately,
`schema.build_query_context()`, called inside `generate_sql()`, pulls targeted
live schema and relationship text for just the tables the plan already
selected — a direct catalog read, not a search.

`db/schema_catalog.py` is the single source of truth either way: live
introspection (so column names/types can't drift) plus curated notes for what
introspection can't tell you — which columns are reliably populated, which
joins need a non-obvious expression, which tables need a default filter to be
meaningful.

The planner keeps `universe_limit` separate from `result_limit`: "top 50 MSAs
with the lowest risk" first selects the 50 largest MSAs, then ranks those 50 by
the requested NRI metric. The canonical MSA/NRI join path is
`census_msa -> cbsa_counties -> nri_tracts`, aggregated at MSA grain. On
startup, `_ensure_schema()` removes any `house_rankings`/`nri_msa_risk` views
left by older builds — semantic SQL views are intentionally not part of this
architecture; the SQL agent queries physical tables only.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Browser (Leaflet map + chat UI)                                            │
│  ├─ House markers + sidebar                                                │
│  ├─ Layer control: None / Crime / NRI / Bike Lanes                         │
│  ├─ Bike route planner (start/end inputs with local BikePGH route map)     │
│  ├─ House Chat (per-property LangGraph agent)                              │
│  └─ General Chat (cross-city LangGraph agent)                              │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ HTTP (FastAPI)
┌──────────────────────────────▼─────────────────────────────────────────────┐
│ main.py                                                                    │
│  ├─ /api/houses                 GeoJSON of all houses                      │
│  ├─ /api/layers/crime           Severity-weighted heatmap grid             │
│  ├─ /api/layers/nri             NRI tract choropleth (GeoJSON)             │
│  ├─ /api/layers/bike            BikePGH overlay for current viewport       │
│  ├─ /api/bike/route             Local BikePGH routing endpoint             │
│  ├─ /api/house/{id}/chat        House-specific agent                       │
│  ├─ /api/chat                   General agent                              │
│  ├─ /metrics                    Prometheus metrics endpoint                │
│  └─ /api/house/{id}/photo       Photo upload                               │
└───────────────┬───────────────────────────────────────┬────────────────────┘
                │                                       │
┌───────────────▼──────────────┐      ┌────────────────▼─────────────────┐
│ DuckDB                       │      │ ChromaDB (vector)                │
│  ├─ houses                   │      │  ├─ house_documents              │
│  ├─ nri_tracts               │      │  │  descriptions/photos/notes    │
│  ├─ census_*                 │      │  └─ search + retrieval           │
│  ├─ sold_homes               │      └──────────────────────────────────┘
│  ├─ crime_incidents          │
│  ├─ bike_routes              │
│  ├─ cbsa_*                   │
│  └─ geocode_cache            │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│ LLM + observability                                                  │
│  ├─ llama-server / Ollama (default: llama-server)                    │
│  ├─ Phoenix OTEL tracing + UI (optional local collector)             │
│  └─ Prometheus-compatible metrics                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- Python 3.11+
- `llama-server` (llama.cpp) installed and running, or Ollama as an alternative
- Phoenix is optional. When `PHOENIX_ENABLED=true`, the app starts a local Phoenix server automatically on `http://127.0.0.1:6006` if that port is available.
- Internet access is needed for Nominatim geocoding when sold-home addresses or bike-route endpoints are not already cached.
- [Commute times](#commute-times) call free public routing servers (OSRM) unless you run your own; nothing else in the app needs a key or an account.

---

## Quick Start

### 1. Clone / place files

```
real_estate_app/
├── main.py
├── config.py
├── setup_data.py
├── requirements.txt
├── .env.example
├── agents/
├── db/
├── services/
├── static/
├── observability.py
└── data/
    ├── redfin/       ← drop Redfin CSV exports here
    ├── sold/         ← drop sold-homes CSV files here
    ├── nri/          ← NRI_CensusTracts_Prod.shp or NRI_Table_CensusTracts.csv
    ├── census/       ← DECENNIALPL2020.P1-Data.csv, DECENNIALPL2020.P1-2026-03-25T232220.csv, list1_2023.xlsx
    ├── crime/        ← one folder per city, e.g. crime/chicago/, crime/pittsburgh/
    ├── bike/         ← BikePGH route layers by city
    ├── shapefiles/   ← TIGER/Line tract shapefiles (optional but recommended)
    └── chroma/       ← vector DB working files
```

### 2. Install dependencies

```bash
cd real_estate_app
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 3. Copy environment defaults

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Mac/Linux
```

The project reads `.env` via `pydantic-settings` and includes defaults for:
- local `llama-server` or Ollama-backed LLMs
- model timeout and token caps
- Phoenix observability endpoint
- app host/port and default map center

All settings are optional. The defaults in `config.py` point at a local
`llama-server` on port `8080`, Phoenix on port `6006`, and the database and
data directories under this repository.

### 4. LLM: run `llama-server` (default)

This repository is configured to use `llama-server` (from the `llama.cpp`
project) as the primary local LLM endpoint. Example:

```bash
llama-server -hf DuoNeural/Gemma-4-26B-A4B-it-GGUF:Q3_K_M \
  -ngl 999 -c 28672 -fa on --cache-type-k q8_0 --cache-type-v q8_0
```

```
llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL -c 24576 \
  --cache-type-k q8_0 --cache-type-v q8_0 -ngl -1 -fa on --port 8080
```

If you prefer Ollama, it remains supported as an alternative. Update the relevant
settings in `.env` / `config.py` and run:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve
```

> The app defaults to `nomic-embed-text` for embeddings, even when using a
> separate local LLM server stack.

### 5. Place your data files

See **Data Sources** section below.

### 6. Load data

```bash
python setup_data.py
```

### 7. Run the app

```bash
python main.py
# or:
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## Data Sources

### Redfin Favorites (required to see houses on the map)

1. Go to Redfin → My Redfin → Favorites (or Saved Searches)
2. Export to CSV
3. The app expects columns including `lat`/`lon`, `address`, `city`, `state`, `zip`, `price`, `beds`, `baths`, `sqft`, `status`
4. If you already have Walk/Bike/Transit scores, include them as `walk_score`, `bike_score`, `transit_score`
5. Drop the CSV into **`data/redfin/`**

You can drop multiple CSV files (e.g. one per city).

### FEMA National Risk Index (highly recommended)

1. Go to: https://www.fema.gov/about/openfema/data-sets/national-risk-index-data
2. Download the tract-level data package
3. The project prefers the shapefile version at **`data/nri/NRI_CensusTracts_Prod.shp`**
4. The CSV fallback is also supported at **`data/nri/NRI_Table_CensusTracts.csv`**

### Census Tract Populations

1. Go to https://data.census.gov
2. Search for table **DECENNIALPL2020.P1**
3. Filter: Geography → Census Tracts → All States → All Tracts
4. Download → Save as **`data/census/DECENNIALPL2020.P1-Data.csv`**

### MSA Populations

1. Same table **DECENNIALPL2020.P1** on data.census.gov
2. Filter: Geography → Metropolitan Statistical Areas → All MSAs
3. Download → Save as **`data/census/DECENNIALPL2020.P1-2026-03-25T232220.csv`**

### CBSA County Crosswalk (needed for MSA-level NRI queries)

1. Go to: https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
2. Download the current crosswalk and store it under **`data/census/`** as a file like **`list1_2023.xlsx`** or a CSV equivalent; the app accepts `list*.xlsx`, `list*.xls`, and `list*.csv` variants.

### Census Tract Shapefiles (optional, for fast tract-FIPS assignment)

Without shapefiles, the app uses the Census Geocoder API to resolve each house's census tract — this is slow (~0.15s per house). With shapefiles it's instant.

1. Go to: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
2. Select: Year = 2020 → Layer = Census Tracts → Select your states
3. Unzip into **`data/shapefiles/`** (one or multiple states)
4. Re-run: `python setup_data.py --resolve-tracts`

### BikePGH route data (optional but required for local bike routing)

The route planner uses the locally ingested BikePGH network from the DuckDB `bike_routes` table and its source data under `data/bike/`.

- `data/bike/Bike Lanes/`
- `data/bike/Pittsburgh/`

If those files are present, the app can answer route questions like:
- “Is there a bikeable route from Mount Washington to Point State Park?”
- “Find a safe bike route from A to B”

No external road-routing engine is used for the route graph itself; the service only geocodes place names with Nominatim.

### Sold Homes (optional)

- Drop CSV files with sold home data into **`data/sold/`**
- Required columns: `address`, `lat`, `lon`, `sold_price`, `sqft`, `sold_date`
- Optional: `list_price`, `beds`, `baths`

### Crime Data (optional, powers the "Crime" map layer)

Every city publishes crime data differently — different columns, different
offense vocabularies, sometimes different file formats entirely. Rather than
teach the app one format, each city gets its own small parser (see
`services/crime_sources.py`); currently supported:

**Baltimore · Boston · Buffalo · Chicago · Indianapolis · Minneapolis · Philadelphia · Pittsburgh**

1. Download the city's open-data crime/incident export (.csv or .xlsx — most
   city data portals offer both) — typically named something like "Crime
   Incidents", "Part 1 Crime", or "Police Blotter"
2. Drop it in **`data/crime/<city>/`**, e.g. `data/crime/chicago/crimes_2024.csv`
   — the folder name must match one of the city keys above (lowercase)
3. You can drop multiple files per city (e.g. one per year) — they're all
   read and combined
4. Run `python setup_data.py --only crime`

Each incident is classified into a standardized category (Homicide, Robbery,
Burglary, Theft, ...) and assigned a **severity weight from 1–10** — see
`services/crime_taxonomy.py` for the full category list and the reasoning
behind the weights. The "Crime" map layer's heatmap intensity is driven by
this weight, not raw incident count, so a block with a few thefts doesn't
outweigh a block with one assault. The weights are a plain, editable Python
list — tune them to your own judgment if the defaults don't match how you'd
weigh things.

To cover a city that isn't listed above, add a new parser class to
`services/crime_sources.py` (subclass `CrimeParserBase`, following the
pattern of the existing city classes) and register it — no changes needed
anywhere else.

---

### Commute (optional, needs a work location)

Drive, bike and walk times from every house to your work address, computed with free OpenStreetMap
routing. Nothing to download: set the address in the sidebar's **Commute** tab. See
[Commute times](#commute-times) for what is sent where and how to self-host the routing server.

## Usage Guide

### Map

- **Colored markers** by status:
  - 🔵 Blue = Active
  - 🟡 Yellow = Pending
  - 🟣 Purple = Contingent
  - 🟢 Green = Pre-Market
- **Click any marker** to open the sidebar
- **Map Layers panel** (top-right): **Houses** is on by default; turn on any combination of **Crime**, **Sold Homes**, **Bike Routes** and **Commute time to work** alongside it, plus at most one **area** fill (**Risk (NRI)** or **Population** — picking one turns the other off, since two overlapping fills would just hide one under the other). See [Map layers](#map-layers) for how this list is built and what a new dataset from the Data page adds to it automatically.
- **Bike route planner**: a control on the map lets you enter a start and end location and route using the local BikePGH network. Resulting route geometry is drawn directly on the map and in chat responses when applicable.

### House Sidebar

| Tab | Contents |
|-----|----------|
| **Details** | Price, beds/baths/sqft, livability scores, NRI summary |
| **Risk** | Full NRI breakdown — 18 hazards, composite score, percentile, EAL |
| **Chat** | AI agent for this house — pricing, risk Q&A, description analysis |
| **Docs** | All stored text/photos; upload new photos |

### House Chat — Example Conversations

```
User: <pastes Redfin/Zillow description>
Agent: Thanks — I've saved that to the knowledge base. How can I help?

User: Estimate a fair price for this house
Agent: [runs price estimator, checks comparables, sold homes in tract]

User: How bad is the flood risk here?
Agent: [fetches NRI, explains RFLD_RISKS score and EAL in plain language]

User: What's the crime like in this neighborhood?
Agent: [queries crime_incidents for this house's crime_city, summarizes by category and severity]

User: Is the HOA fee reasonable?
Agent: [compares HOA against other houses in the same city]
```

### General Chat — Example Questions

- *"Among the top 50 metro areas by population, which have the lowest overall risk?"*
- *"Which of my saved houses has the best combined walk + transit score?"*
- *"Compare tornado risk between Dallas and Houston census tracts"*
- *"What's the median price/sqft across all my Austin houses?"*
- *"Which city has the most severe crime, weighted, not just the most incidents?"*

---

## Reloading Data

After adding new CSV files, re-run:

```bash
python setup_data.py --only redfin    # just Redfin
python setup_data.py --only sold      # just sold homes
python setup_data.py --only crime     # just crime data
python setup_data.py --only bike      # just BikePGH route data
python setup_data.py --only census    # CBSA crosswalk + tract/MSA populations
python setup_data.py --only geocode   # retry pending sold-home geocodes
python setup_data.py --only match     # link sold records to houses
python setup_data.py --only repair    # repair X-coded MSA codes
python setup_data.py --only sold --no-geocoding  # load sold data without network geocoding
python setup_data.py --resolve-tracts # resolve missing house tract FIPS values
python setup_data.py                  # everything
```

The vector database (ChromaDB) grows automatically as you paste descriptions in chat.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Map shows no houses | Run `setup_data.py`, check `data/redfin/*.csv` exists |
| NRI data missing | Download the NRI shapefile or CSV into `data/nri/` (preferred: `NRI_CensusTracts_Prod.shp`; fallback: `NRI_Table_CensusTracts.csv`) |
| Tract FIPS not resolved | Run `python setup_data.py --resolve-tracts` |
| Chat says "I couldn't generate a response" | Make sure `llama-server` is running (or Ollama if you chose that stack) |
| Embedding errors | If using Ollama embeddings: pull `nomic-embed-text` with `ollama pull nomic-embed-text`. If using another embedding provider, configure accordingly. |
| Slow tract resolution | Add TIGER/Line shapefiles to `data/shapefiles/` |
| "Crime" layer is empty | Confirm `data/crime/<city>/` has files for a city your map view overlaps, then `python setup_data.py --only crime` |
| "NRI" layer is empty / shows a warning | It needs both tract *geometry* (from the NRI shapefile, or TIGER/Line shapefiles in `data/shapefiles/`) and tract *attributes* (`python setup_data.py --only nri`) — the layer's warning message says which is missing |
| Phoenix is unavailable | Set `PHOENIX_ENABLED=false` to run without tracing, or start it manually with `python -m phoenix.server.main serve` |
| Bike route returns no route | Confirm BikePGH layers were loaded with `python setup_data.py --only bike`; the router does not fall back to an external road-routing service |

---

## Commute times

Say where you work and the app estimates how long each house is from it by car, bike and on foot,
shows the minutes on the map, and lets both chats rank houses by commute. Free and open only: no API
keys, no accounts, no credits.

> Unlike the BikePGH route planner (which is fully local), commute times **do** call an external
> routing service: OSRM. Read [Privacy](#privacy-and-self-hosting) below.

### Set it up

1. Select a house and open the **Commute** tab.
2. Type your work address (or coordinates such as `40.4406, -79.9959`), or choose **Click the map instead**.
3. The address is looked up and every house is computed in the background (a progress bar shows). A new
   house, or a new work location, is computed the same way.

Prefer configuration? Put `WORK_ADDRESS=...` in `.env` and run `python setup_data.py --only commute`. The full
data load (`python setup_data.py`) also computes commutes whenever a work location exists.

### What you get

- **Commute tab**: for the selected house, one bar per mode (drive, bike, walk; transit if configured) with minutes
  and miles. Colors follow *your* limit ("My maximum commute"): green up to two-thirds of it, amber up to it, red beyond.
- **Map layer**: tick **Commute time to work** in the Map Layer control. Each pin shows its minutes, houses over your
  limit turn red and fade (the one you have open never fades), and your work location is marked.
- **General Chat**: `house_commute` is a built-in catalog table, so questions such as *"Which houses have the shortest
  commute?"*, *"Which house has the longest bike commute?"* or *"What is the average commute distance?"* are planned
  and answered like any other. Mode phrases win over the generic word "commute" (a declared `overrides`, see the
  architecture doc).
- **House Chat**: *"How long is the commute from here?"* calls `get_commute_info()`.

### Free services used

| What | Service | Notes |
|---|---|---|
| Work address to coordinates | US Census geocoder, then OpenStreetMap Nominatim | one lookup per change; Nominatim's usage policy is followed (identifying `User-Agent`, at most 1 request/second) |
| Drive | OSRM public demo server (`router.project-osrm.org`) | no key |
| Bike, walk | OSRM on `routing.openstreetmap.de` (FOSSGIS) | no key |
| Transit (optional) | **your own** OpenTripPlanner over a GTFS feed | off unless `OTP_BASE_URL` is set |

### How to read the numbers

Drive, bike and walk are **free-flow estimates**: OpenStreetMap roads, house to work, with no traffic, no signal
timing and no time-of-day effect, so drive times are best-case. `COMMUTE_DRIVE_FACTOR` (for example `1.3`) applies a
rough rush-hour allowance to drive only. Walking is not estimated beyond ~6 straight-line miles, cycling beyond ~30,
driving beyond ~250 (the tab says "too far to walk"). Requests are batched (about 90 houses per request) and spaced by
`OSRM_MIN_INTERVAL_S` to stay within the public servers' fair use.

### Privacy and self-hosting

Your work location and each house's coordinates are sent to the routing servers above (public by default); the tab
lists exactly which. To keep them private, run OSRM yourself (any container runtime works: Docker Engine, Podman).
The extracted files are specific to a profile, so use one folder per profile:

```bash
# car profile, Pennsylvania extract from Geofabrik (repeat with bicycle.lua / foot.lua in their own folders)
mkdir osrm-car && cd osrm-car
wget https://download.geofabrik.de/north-america/us/pennsylvania-latest.osm.pbf
docker run -t -v "$PWD:/data" ghcr.io/project-osrm/osrm-backend osrm-extract   -p /opt/car.lua /data/pennsylvania-latest.osm.pbf
docker run -t -v "$PWD:/data" ghcr.io/project-osrm/osrm-backend osrm-partition /data/pennsylvania-latest.osrm
docker run -t -v "$PWD:/data" ghcr.io/project-osrm/osrm-backend osrm-customize /data/pennsylvania-latest.osrm
docker run -t -p 5000:5000 -v "$PWD:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld --max-table-size 1000 /data/pennsylvania-latest.osrm
```

Then in `.env` (bike and walk servers on ports 5001 and 5002 in the same way):

```
OSRM_DRIVE_URL=http://localhost:5000
OSRM_BIKE_URL=http://localhost:5001
OSRM_FOOT_URL=http://localhost:5002
OSRM_MIN_INTERVAL_S=0
OSRM_TABLE_MAX=500
```

The tab then reports "your server" and no longer says coordinates leave your machine (only the address lookup still does).

### Transit (optional, experimental)

OpenTripPlanner is free and open source, but no hosted instance exists for arbitrary use, so transit is off by default.
To enable it: download your agency's GTFS feed and an OpenStreetMap extract, build and serve an OpenTripPlanner 2.x
graph, and set `OTP_BASE_URL=http://localhost:8080`. The app asks its GTFS GraphQL API (`OTP_API=rest` selects the
legacy REST API) for a weekday-morning trip (`COMMUTE_DEPART_TIME`, default `08:00`). **This path is exercised only
against a mock OpenTripPlanner in the tests.** Check a few itineraries against your own instance before relying on it.

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `WORK_ADDRESS` | empty | optional default; the Commute tab saves the real value in the database |
| `COMMUTE_MODES` | `drive,bike,walk` | modes to estimate (`transit` turns on with `OTP_BASE_URL`) |
| `OSRM_DRIVE_URL`, `OSRM_BIKE_URL`, `OSRM_FOOT_URL` | public servers | one OSRM server per profile |
| `OSRM_MIN_INTERVAL_S`, `OSRM_TIMEOUT_S`, `OSRM_TABLE_MAX` | `1.0`, `25`, `90` | politeness delay, timeout, sources per request |
| `COMMUTE_DRIVE_FACTOR` | `1.0` | multiplies drive time (`1.3` is a rough rush hour) |
| `OTP_BASE_URL`, `OTP_API`, `COMMUTE_DEPART_TIME` | empty, `graphql`, `08:00` | optional transit |
| `NOMINATIM_EMAIL`, `NOMINATIM_USER_AGENT`, `NOMINATIM_COUNTRYCODES` | empty, app name, `us` | Nominatim etiquette and scope |

### Limits

- One work location (a second workplace is a natural extension: the table is keyed by house).
- House to work only, free-flow, no time of day. There are no isochrones: OSRM has no isochrone endpoint.
- Houses need coordinates; houses without them are skipped and counted.
- Changing the work location, modes, server URLs or drive factor marks stored estimates stale; the tab offers **Recompute**.
- If a public server is unreachable or rate-limits you, the job finishes and lists which mode was missing and why
  (for example "Bike: no routes were returned (...)"); retry later or self-host.

---

## Map layers

The map has one toggle panel for every layer the app currently knows how to draw — built-in or added
through the [Data page](#the-data-page) — built the same way the SQL planner discovers what it can query:
by reading the live [catalog](#the-catalog-is-a-store), not a hardcoded list. `services/map_layers.py`
looks at each agent-visible table's actual columns and classifies it:

| A table has... | Becomes | Example today |
|---|---|---|
| `lat` + `lon` columns, few rows | a marker layer | (none yet at this size) |
| `lat` + `lon` columns, many rows | a heat layer, weighted by whatever numeric column looks most like a weight (`severity`, `score`...), else a uniform count | Crime, Sold Homes |
| a `geometry_json` column of line features | a line layer | Bike Routes |
| a `geometry_json` column of polygon features | its own fill, colorable by any numeric column | (whatever you upload with its own shapes) |
| a `tract_fips`-keyed relationship to `nri_tracts`/`census_tracts`, no geometry of its own | a choropleth on the **same shared tract polygons** `services/geo_utils.py` already loads, colored by any numeric column | Risk (NRI), Population |
| none of the above | not a map layer (still fully queryable in chat) | `house_snapshots`, `census_msa`, `cbsa_counties` |

Approve a new dataset on the Data page and, on your next visit to the map, it is already in this list —
nothing to wire up. The two tests proving this end to end
(`tests/test_map_layers.py::test_a_dataset_linked_to_nri_tracts_becomes_a_choropleth_with_no_code_change`
and `...test_an_uploaded_polygon_dataset_becomes_a_fill_layer_with_no_code_change`) upload a CSV and a
GeoJSON respectively and assert the resulting layer appears with no line of `map_layers.py` naming either
one.

### Combining layers: what's allowed, grounded in your own database

Two layers **conflict** only when they would draw a solid fill over the same space — stacking two of those
mostly just hides one under the other, so the panel keeps at most one **area** fill active (a radio group,
not a rule you have to remember) and turning one on turns the other off automatically. Everything else
(markers, heat, lines) draws as distinct glyphs rather than solid coverage, so any number of them combine
with each other and with the one active fill.

Loading your `data/real_estate.duckdb` and asking `services.map_layers.describe_layers()` what it finds
gives a concrete, current answer rather than a hypothetical one — this is exactly what
`tests/test_map_layers.py::test_real_database_layer_inventory_and_fill_exclusivity` asserts against a copy
of your file:

| Layer | Kind | Group | Notes |
|---|---|---|---|
| Houses | markers | — | always available; the only layer on by default |
| Risk (NRI) | choropleth | **area** (pick one) | 24 hazard/score columns to color by; defaults to `risk_score` |
| Population | choropleth | **area** (pick one) | shares Risk's tract polygons, so the two are mutually exclusive |
| Crime | heat | overlay | weighted by `severity_weight` (769,286 rows) |
| Sold Homes | heat | overlay | uniform count (no weight-like column on this table) |
| Bike Routes | lines | overlay | its own overlap-resolution logic, unchanged (see below) |
| Commute time to work | decoration on Houses | overlay | needs a work location set in the Commute tab |

Not shown, on purpose: `bike_lanes` is a real, populated-looking table left over from before this catalog
existed, but it was never registered through the Data page, so — like anything outside the catalog — the
map (and chat) correctly doesn't know about it; `geocode_cache` is marked internal
(`agent_visible: false`); `house_snapshots`, `house_commute`, `census_msa` and `cbsa_counties` have no
coordinates or shared polygon source of their own, so they stay fully queryable in chat without becoming a
map layer. `census_msa`/`cbsa_counties` (county/CBSA-level data) could become choropleths the same way NRI
and Population do; nothing in the real data currently links to them richly enough to be worth it, and
there's no bundled county/CBSA polygon source the way there is for tracts, so that stayed out of this pass.

### Bike Routes' own overlap resolution (unchanged)

BikePGH's raw data classifies some street segments under more than one category (e.g. "On Street Bike
Route" and "Bike Lane" for the same block). `services/map_layers.py::_canonicalize_bike_features` (moved
here from the old `services/layers.py`, logic untouched) resolves this by a fixed priority — protected lane
> bike lane > trail > bikeable sidewalk > sharrows > cautionary route > on-street route — so the map shows
one category per segment without editing the underlying data. `services/bike_routing.py`'s separate route
*planner* ("Find a bikeable route") imports the same priority table rather than keeping its own copy.

### Choropleth geometry

Both built-in choropleths, and any future tract-linked dataset, share whatever tract polygons
`services/geo_utils.py` already loads for house-to-tract assignment (the NRI shapefile's cache, or
TIGER/Line files under `data/shapefiles/`) — nothing new to configure. If that cache isn't present, both
choropleth options are listed but shown as unavailable, with the reason stated, exactly like the
onboarding page's own "Add census tract from coordinates" step when it hits the same gap.

---

## Adding New Data Sets

There are two ways in, and both end in the same place: rows in the **unified catalog**
(the `catalog_*` tables of `data/real_estate.duckdb`).

**From the browser, no code: the Data page** (`/data`, or *Data model* in the top bar). Upload a
file, describe it, review the links the system proposes, approve. Use this for CSV / Excel / JSON /
Parquet / GeoJSON / shapefile data. See [The Data page](#the-data-page).

**As a built-in source, in code**: for data you load with `setup_data.py`:

1. Add a new `load_xyz()` function in `services/data_loader.py`
2. Add a new table in `db/duckdb_store.py` -> `_ensure_schema()`
3. Add ONE `TableMeta` entry (description + notes on any non-obvious columns) to
   **`db/catalog_seed.py`**, plus one `Relationship` if it joins to an existing table. You do NOT
   need to write a new agent tool or teach the LLM a new SQL pattern: `query_database` plus the
   catalog is enough for the agent to work out how to query it, including joins.
4. Call `setup_data.py` to load it. The seed is copied into the catalog store on the next start
   (see [The catalog is a store](#the-catalog-is-a-store)).

`db/schema_catalog.py` is the single, unified view of what the agent knows about the data: live
introspection of the running database (so column names/types can't go stale) plus curated notes for
things no introspection can tell you: which columns are reliably populated, which joins need a
non-obvious expression, which tables need a default filter. `check_data_availability`,
`get_database_schema`, `setup_data.py`'s summary and the response validator's fallback message all
read from it, so there is nothing else to keep in sync when you add a table.

---

## The Data page

`http://localhost:8000/data` has two halves.

**The catalog map** (left) draws every table the agent can query as a card, grouped by area, with the
join relationships between them: grey = built in, blue = added by you, amber dashed = awaiting your
review; crow's foot = "many", bar = "one". Click a table or a link for its details; a link you approved
keeps the evidence it was approved on. Lanes are ordered to keep linked tables close together.

**The workbench** (right) is where data is added:

1. **Upload**: CSV/TSV, Excel (`.xlsx`/`.xls`, header row detected), JSON/JSON-lines, Parquet,
   GeoJSON, GeoPackage or a zipped shapefile (reprojected to EPSG:4326, geometry kept as GeoJSON like
   `bike_routes`). Files are read as text first so FIPS/ZIP/parcel IDs keep their leading zeros, and
   column names become safe identifiers (the SQL guards reject words like `update` or `load`; the
   original header is kept). Nothing enters the catalog yet: the data is staged as `stg_<id>`.
2. **Describe**: name, table name, what it holds, grain ("one row per census tract"), area, and per
   column a role, description, unit and *synonyms*, the way you would ask about it in chat
   ("walkability index"). Optional: **Draft with the local model** (see
   [Catalog model](#catalog-model-optional)) and **enrichments**: *Add census tract from
   coordinates* (point-in-polygon via `services/geo_utils.py`) or *Geocode addresses* (US Census
   geocoder via `services/geocoder.py`; cached in `geocode_cache`; capped per run).
3. **Find links**: deterministic, no LLM. Every column is tested against every agent-visible table in
   the catalog (including datasets you added earlier) after a small fixed set of value normalizations:
   zero-pad, first/last N characters, trim/upper/lower, and a case-insensitive comparison that
   normalizes both sides. Street addresses are also linked to `houses` with the same normalization
   tiers as sold-home matching.
4. **Review**: each proposal shows what a person needs to verify it: *your value -> after the rule ->
   the rows it lands on*, both match rates (your values found in the target; target rows covered by
   your data), fan-out, cardinality, the values that did **not** match, and warnings. Links that share
   a normalization are grouped. You can adjust cardinality, "preferred", and the note before approving.
5. **Approve**: the table first, then each link. Approval is ONE atomic catalog transaction: it creates
   the data table, registers the table, columns, concepts and entity domains, materializes any derived
   key column the link needs, registers the relationship together with its evidence, and reloads the
   in-memory catalog on commit. **General Chat and House Chat read the catalog on every turn, so the
   change is live immediately, with no restart.** A link can be **revoked** and a dataset **retired**
   at any time (the data table is kept, hidden from the assistant).

### What the two chats gain

- **General Chat**: the planner, both SQL validators, the availability report and the metadata
  retrieval read the live catalog, so the new table is queryable, its measures are found through the
  synonyms you gave, and a `USER-ADDED DATASETS` block joins the data-availability context.
- **House Chat**: a dynamic prompt section lists the datasets linked to houses, and a new approved
  function `get_linked_dataset_records(dataset="")` returns the rows linked to the open house by
  following the approved relationship graph (directly, or through `nri_tracts` / `census_tracts`).

### Safety properties

- Nothing reaches the agents before approval, and a link cannot be approved before its table.
- A derived key column is added only on approval, with exactly the SQL rule the examples showed.
- `geocode_dataframe` runs `UPDATE sold_homes ... WHERE sale_id = ?` whenever the frame has a `sale_id`
  column; the Data page always passes a non-existent `sale_id_col`, so uploaded data can never alter
  real sales.
- Uploaded values are data, not markup or instructions: the page renders with `textContent`, model
  prompts label the profile as data, and model output is schema-constrained, validated and reviewed.
- Component codes (county `001`, tract code `040100`) are not offered as join keys: they repeat across
  states.
- **Alias precedence**: if a new phrase extends an existing concept's phrase ("walkability index" over
  "walkability"), the new concept declares `overrides: [house_walk_score]`, so questions using the new
  phrase don't also pull in the old concept's tables and null-policy filter. It is disclosed at review.
  Existing concepts are untouched: a sweep of a query built from every built-in alias gives identical
  results before and after.

### Limits

- Single-user, local. Work runs on the event-loop thread like the rest of the app (one shared DuckDB
  connection), so a very large upload or a geocoding run blocks other requests while it works. Caps:
  `ONBOARDING_MAX_UPLOAD_MB`, `ONBOARDING_GEOCODE_ROW_CAP`, `ONBOARDING_API_ROW_CAP`.
- Links are equality joins on normalized keys. Polygon layers are not modeled as spatial joins; for
  point data use *Add census tract from coordinates*. Text keys are matched exactly (after
  trim/case), not fuzzily.
- A published dataset is read-only in the UI; retire it and upload again to change it.

---

## The catalog is a store

Built-in sources and datasets added on the Data page are the same kind of object. The catalog lives
in the application's own DuckDB file:

| Table | Holds |
|---|---|
| `catalog_tables`, `catalog_columns` | tables, grain, hints, per-column notes (role, unit, original header) |
| `catalog_relationships` | the join graph, with the evidence each link was approved on |
| `catalog_concepts` | semantic concepts: aliases, operations, filters, `overrides` |
| `catalog_entity_domains` | values recognized inside questions |
| `catalog_meta`, `catalog_audit` | a version counter that changes on every change; who changed what |
| `catalog_datasets`, `catalog_proposals` | the *workflow*: what was uploaded and what awaits review |

`origin` (`builtin` / `upload`) is provenance, not a separate layer. `db/catalog_seed.py` seeds the
built-in rows on first run; on upgrade an unmodified built-in row is refreshed when its seed definition
changes, and a row you edited (`user_modified`) is never overwritten. Adding a source therefore no longer
means editing `db/schema_catalog.py`.

`db/schema_catalog.py` keeps its public API (`TABLES`, `RELATIONSHIPS`, `SEMANTIC_GLOSSARY`,
`ENTITY_DOMAINS`, and every planner helper). Those four names are live containers that load lazily,
refresh **in place** when the catalog changes, and reload automatically when the DuckDB connection is
replaced (the evaluation harness closes it and points at a fixture DB; the fixture reseeds itself).
Changes made through `schema.register_*` are transactional and reload once, on commit.

The metadata vector index syncs incrementally: it re-embeds only documents whose text changed, only when
the catalog version changes, and if Ollama is down the retrieval falls back to a lexical search over the
live catalog. Inspect the store any time:

```sql
SELECT origin, name, status FROM catalog_tables ORDER BY ord;
SELECT rel_key, origin, approved_at FROM catalog_relationships WHERE origin <> 'builtin';
SELECT ts, action, object_key FROM catalog_audit ORDER BY id DESC LIMIT 20;
```

---

## Catalog model (optional)

The Data page can ask a local model to **draft wording** for an upload: titles, column descriptions,
roles, units, synonyms. It never decides what joins, sets a cardinality or confidence, writes SQL, or
touches the catalog: those come from measured evidence and a person's approval. Without a model the page
works the same, with rule-based defaults.

**Design** (`services/catalog_llm.py`): the catalog pipeline is a fixed sequence, so it is a
deterministic workflow, not an agent. A *capability router* maps a task tier to an ordered chain of
endpoints, with cached health checks and automatic fallback:

| Tier | Task | Chain |
|---|---|---|
| `draft` | describe a dataset | `CATALOG_DRAFT_*` -> the chat llama-server |
| `judge` (off by default) | annotate link candidates with a plausibility note | `CATALOG_JUDGE_*` -> draft -> chat |

Every call is schema-constrained (`response_format: json_schema` -> grammar-constrained decoding, with a
plain-JSON fallback), validated (pydantic, then deterministic checks: invented column names dropped,
synonyms sanitized, full coverage required), bounded (temperature 0, `max_tokens`, your stop sequences,
one repair pass) and serialized (one call in flight per endpoint, so a draft cannot flood a shared chat
server). If nothing is reachable or the output stays invalid, the page carries on without it.

```bash
python -m services.catalog_llm --status                       # resolved endpoint chain + health
python -m services.catalog_llm --selftest                     # score the draft tier on 4 sample datasets
python -m services.catalog_llm --selftest --base-url http://127.0.0.1:8081/v1 --model <name>   # try a candidate
```

The self-test reports valid-JSON rate, first-try rate, column coverage, agreement of the model's roles
with a reference, synonym rate and latency, and returns PASS/FAIL against fixed thresholds. Judge a model
by that, not by its name.

**Fitting models into 16 GB.** Approximate weights at Q4_K_M are 0.6 GB per billion parameters (Q5_K_M
0.7, Q8_0 1.1); add the KV cache (an 8B-class GQA model at 8k context is about 1.2 GB in fp16, half that
with `-ctk q8_0 -ctv q8_0 -fa`) and roughly 0.4 GB per process. A ~26B model at Q3 already uses most of
16 GB, so a second GPU-resident model needs room made for it:

| Setup | VRAM cost | Notes |
|---|---|---|
| **A. Share the chat server** (default) | none | drafts queue behind chat requests; fine for occasional use |
| **B. Catalog model on CPU** (`llama-server -ngl 0 --port 8081`) | none | a 4-8B instruct model at Q4/Q5; tens of seconds per dataset, which is fine for an offline, human-paced step |
| **C. Two GPU-resident models** | ~4 GB (4B) to ~6.5 GB (8B) | only if the chat model shrinks, e.g. keep a MoE chat model's experts on CPU (`--n-cpu-moe` / `-ot`, if your build supports it) |
| **D. Time-slice** with a swapping proxy such as `llama-swap` | none, but a load pause | full VRAM for whichever model is needed |

Point `CATALOG_DRAFT_BASE_URL` at the second server. Prefer a non-thinking instruct variant at Q4 or
higher for structured output; `CATALOG_LLM_DISABLE_THINKING=true` (default) asks for that.

---

## Project Structure

```
main.py               FastAPI app + all HTTP endpoints
config.py             All settings (paths, models, ports)
setup_data.py         One-time data loader script
run_eval.py            Agent evaluation pipeline entry point
update_eval_ground_truth.py  Regenerate golden expectations from fixture SQL

api/
  onboarding.py       Data page HTTP API (/api/onboarding/*)
  commute.py          Commute tab HTTP API (/api/commute/*)
  map_layers.py       Map layer panel HTTP API (/api/layers/*)

agents/
  tools.py            LangChain tools (SQL, vector search, price estimation)
  query_planner.py    Deterministic analytical query planning and semantic mappings
  house_agent.py      Per-house ReAct agent (LangGraph)
  general_agent.py    General ReAct agent (LangGraph)
  response_validator.py  Post-hoc check that replies are grounded in real tool output

db/
  duckdb_store.py     All SQL queries and schema management
  schema_catalog.py   Unified catalog: live registry + planner support (reads the store)
  catalog_store.py    Catalog persistence in DuckDB (catalog_* tables), seed sync, workflow state
  catalog_seed.py     Built-in table/relationship/concept definitions (seed only)
  catalog_model.py    Catalog dataclasses + concept DSL
  vector_store.py     ChromaDB — embed, store, search text documents

services/
  data_loader.py      Parsers for Redfin, NRI, Census, Sold, Crime
  crime_sources.py    Per-city crime file parsers (one class per city)
  crime_taxonomy.py   Standardized crime categories + severity weights
  layers.py           Viewport-scoped queries behind the Crime/NRI map layers
  geo_utils.py         Census tract FIPS assignment (shapefile or API)
  dataset_readers.py  Generic readers for uploads (csv/excel/json/parquet/geo) + type inference
  relationship_discovery.py  Deterministic link discovery with evidence and examples
  dataset_onboarding.py      Data page workflow (stage, enrich, propose, approve, publish, revoke)
  catalog_llm.py      Optional model router for drafting wording, plus --selftest
  house_links.py      Rows of user-added datasets linked to one house (House Chat)
  commute.py          Commute times: work-address lookup, OSRM/OpenTripPlanner clients, refresh job
  map_layers.py       Generic, catalog-driven map layer classification and data builders

eval/
  fixtures.py         Builds a small deterministic DB with hand-verifiable answers
  golden_set.py        The golden examples (structured + free-text)
  scoring.py            Assert-equal scoring for structured examples
  judge.py              LLM-as-judge scoring for free-text examples
  mock_agent.py         Scripted stand-in used only by --mock (harness smoke test)
  tests/test_scoring.py Unit tests for the scorer itself, no LLM/DB needed
  reports/              Timestamped JSON + Markdown reports land here

static/
  index.html          Leaflet map + sidebar + chat UI
  style.css            App styles, BikePGH route visuals, route planner UI
  app.js              Frontend logic, layer toggles, bike route rendering
  data.html/.css/.js  The Data page: catalog map + dataset workbench
  commute.js          The Commute tab and the commute-minutes map layer
  layers.js           The Map Layers panel: builds the toggle list from /api/layers and renders every kind

observability.py      Phoenix tracing + Prometheus metrics helper

data/
  redfin/             Drop Redfin CSVs here
  sold/               Drop sold-homes CSVs here
  nri/                FEMA NRI shapefile or CSV
  census/             Census P1 tables + CBSA crosswalk
  crime/<city>/       Drop each city's raw crime export here
  bike/               BikePGH network sources used by the local router
  shapefiles/         TIGER/Line tract shapefiles (optional)
```

---

## Observability & metrics

The app auto-starts a local Phoenix trace collector when
`PHOENIX_ENABLED=true` (the default in `.env.example`) and exposes
Prometheus-compatible metrics at `/metrics`. If Phoenix is already listening on
the configured port, the app reuses it and does not start a second process.

- Trace collection is configured via `PHOENIX_COLLECTOR_ENDPOINT`, `PHOENIX_PROTOCOL`,
  and `PHOENIX_UI_URL`
- The General Chat flow records agent, tool, and validation spans
- This is intentionally fail-open: if Phoenix is unavailable, the app continues
  running normally

```bash
python -m phoenix.server.main serve
```

Use the command above only when starting Phoenix separately, for example after
setting `PHOENIX_ENABLED=false`. Open the Phoenix UI at the configured local URL
(default: http://127.0.0.1:6006).

## HTTP API

The browser uses these main endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/houses` | Return saved houses as GeoJSON |
| `GET /api/layers/crime` | Return the viewport's severity-weighted crime grid |
| `GET /api/layers/nri` | Return NRI tract geometry and attributes |
| `GET /api/layers/bike` | Return BikePGH features for the current viewport |
| `POST /api/bike/route` | Route between geocoded places on the local bike graph |
| `POST /api/chat` | Ask the general cross-city agent |
| `POST /api/house/{id}/chat` | Ask the agent about one house |
| `GET /metrics` | Expose Prometheus metrics |

The API also supports house document and photo operations; the interactive UI
is the recommended way to use those endpoints.

---

### Data page endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/data` | The Data page |
| GET | `/api/onboarding/catalog[?dataset_id=]` | The unified catalog (plus a draft overlay for a dataset under review) |
| GET / POST | `/api/onboarding/datasets` | List datasets / upload a file (multipart) |
| GET / PUT | `/api/onboarding/datasets/{id}` | Read a dataset / save its description |
| POST | `/api/onboarding/datasets/{id}/draft` | Draft wording with the catalog model (optional) |
| POST | `/api/onboarding/datasets/{id}/enrich` | `{"kind": "spatial_tract" \| "geocode"}` |
| POST | `/api/onboarding/datasets/{id}/analyze` | Propose the table and its links, with evidence |
| POST | `/api/onboarding/proposals/{id}/decision` | `{"decision": "approve" \| "reject", "edits": {...}}` |
| POST | `/api/onboarding/relationships/revoke` | `{"rel_key": "..."}` |
| POST | `/api/onboarding/datasets/{id}/retire` | Retire a published dataset / discard a draft |
| GET / POST | `/api/onboarding/llm/status`, `/llm/selftest` | The catalog model router |

---

### Map layer endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/layers` | Every layer the live catalog supports right now, its kind, group and availability |
| GET | `/api/layers/{name}` | That layer's data for a viewport (`west,south,east,north`); `measure=`, `weight=`, `city=`, `grid_deg=` as the kind allows |

### Commute endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/commute/config` | work location, modes, providers (public or yours), counts, job state |
| PUT | `/api/commute/work` | `{"address": "..."}` or `{"lat": .., "lon": .., "label": ".."}`: save it and start computing |
| DELETE | `/api/commute/work` | forget the work location and its estimates |
| POST | `/api/commute/refresh` | `{"scope": "missing" \| "all"}`: compute in the background |
| GET | `/api/commute/status` | progress of the running or last job |
| GET | `/api/commute/house/{house_id}` | one house's estimates and whether they are up to date |
| GET | `/api/commute/summary` | every up-to-date estimate (drives the map layer) |

---

## Evaluation

`run_eval.py` runs a small golden set (`eval/golden_set.py`) against the real
agents — the same `run_general_chat` / `run_house_chat` the app itself calls
— on a dedicated, deterministic evaluation database (`eval/fixtures.py`), not
your real data.

```bash
python run_eval.py                 # full run against your configured model
python run_eval.py --list          # see what's in the golden set
python run_eval.py --tags nri,sold_homes    # run a subset
python run_eval.py --mock          # smoke-test the harness itself, no model server needed
python .\run_eval.py --skip-house-agent     # skip evals on house agent
```

Each golden example is one of:
- **structured** — has a definite right answer (a count, a ranking, a set of
  cities). Scored by exact comparison against the reply text: as an ordered
  sequence when order matters, as a set when it doesn't. No LLM involved in
  grading these.
- **free_text** — open-ended (a tradeoff, an explanation). Scored by an LLM
  judge against a rubric in `eval/judge.py`. Configure the judge model via
  `judge_llama_server_base_url` / `judge_llama_server_model` in `.env`; it
  defaults to the same server as the agent under test, but an independent or
  stronger judge is stronger evidence than a model grading itself.

Reports land in `eval/reports/` as both JSON (for tooling) and Markdown (for
reading). Exit code is non-zero if anything failed or errored, so this is
safe to wire into CI. `eval/tests/test_scoring.py` tests the assert-equal
scorer itself against hand-crafted replies — run it directly any time you
change `eval/scoring.py`.

To add a golden example: add fixture data to `eval/fixtures.py` if needed
(prefer deriving expected values from the fixture data programmatically, the
way the existing examples do, over hand-typing a number), then add one
`GoldenExample` to `eval/golden_set.py`.

### Updating evaluation ground truth

`update_eval_ground_truth.py` regenerates structured expected values and the
numeric facts used in free-text rubrics from SQL executed against
`eval/fixture_data/eval_fixture.duckdb`. It does not ask an agent to generate
SQL and never uses previous agent answers as ground truth. It also validates
fixture invariants before rewriting `eval/golden_set.py`.

Run it after changing `eval/fixtures.py` or the fixture schema, and before
running the evaluation suite:

```bash
python update_eval_ground_truth.py
python run_eval.py
```

The updater accepts an alternate fixture database or golden-set path:

```bash
python update_eval_ground_truth.py --db path/to/eval_fixture.duckdb
python update_eval_ground_truth.py --golden path/to/golden_set.py
```

`--skip-house-agent` is accepted for workflow consistency with `run_eval.py`;
it does not remove house-agent examples from the golden file. It only documents
that the subsequent evaluation run may skip those examples. DuckDB must be
installed in the active Python environment.

---

## Capability matrix

This section distinguishes capabilities implemented by the application from
capabilities currently covered by the deterministic evaluation fixture.

### Explicitly supported

| Area | Supported questions and behavior | Fixture coverage |
|------|----------------------------------|------------------|
| House inventory | Counts, city/state/status filters, price aggregates, Walk/Bike/Transit scores, missing-score checks, and rankings | Strong |
| House inventory scope | “My houses” means the full inventory; saved/favorites require explicit list or favorite wording | Supported, but weakly tested because all fixture houses are favorites |
| MSA and tract population | Population totals, rankings, named-MSA comparisons, and specific tract queries | Strong |
| NRI risk | Overall/composite risk, riverine flooding, hurricane, wildfire, other canonical hazards, averages, and rankings | Strong for documented joins and populated fixture columns |
| MSA to NRI analysis | Uses `census_msa -> cbsa_counties -> nri_tracts`, including top-N-by-population universes and MSA-grain aggregation | Strong for the documented pattern |
| Sold homes | Arm’s-length filtering, price averages/rankings, and tract-scoped sold comparables | Strong |
| Individual houses | Stored details, NRI data, risk percentile, EAL, vulnerability/resilience fields, and data-derived price estimates | Strong or partial when fields are NULL |
| House documents | Search a house’s stored descriptions/documents or search across all houses | Supported when documents are loaded |
| Bike routing | Route addresses, neighborhoods, landmarks, parks, or coordinates on the loaded BikePGH network | Supported when BikePGH data is loaded |
| Crime-aware routing | Remove BikePGH edges intersecting buffered high-density crime cells before Dijkstra and return analysis visualization | Supported, but heuristic |
| Crime analytics | Incident counts, severity-weighted crime, standardized categories, and city/month analysis | Supported when crime data is loaded |

### Supported with caveats

- A house’s `msa_code` is usually NULL in Redfin data. Tract-based MSA lookup
  should use the documented county bridge rather than assuming the house row
  carries a reliable CBSA/MSA code.
- The distinction between full inventory and favorites is implemented, but the
  current fixture marks every house `is_favorite = TRUE`, so it cannot detect a
  mistaken favorite filter. Add a non-favorite fixture house to test this rule.
- Arbitrary multi-table questions and unusual ranking variants still depend on
  LLM-generated SQL being written correctly at least once. The application
  provides a three-attempt generation/repair loop and a deterministic fallback
  that compiles SQL directly from the structured plan when generation fails
  (see **General Chat: Agent Architecture & Design Philosophy** above, and
  `AGENT_ARCHITECTURE.md`) — but that fallback can only build joins/operations
  the catalog already declares; it does not invent new relationship paths.
- Sold-home tract questions require geocoded rows; pending rows have NULL
  geography and must not be treated as tract-local.
- Historical house questions require populated `house_snapshots`; the current
  fixture does not populate that table.
- Description search requires documents in the vector store. Crime sources are
  city-specific, and missing or incompatible files are skipped.
- Crime severity means `SUM(severity_weight)`, while incident volume means
  `COUNT(*)`; ambiguous wording can lead to different valid metrics.
- Bike endpoints too far from the loaded network are rejected, and no route
  means no continuous path in the loaded BikePGH graph, not necessarily no
  real-world route. Crime avoidance is a density heuristic, not a safety
  guarantee.
- Geometry/blob columns are hidden from the SQL agent, so arbitrary polygon or
  geometry analytics are not available through `query_database`.
- General Chat is grounded in loaded application data and approved functions; it
  is not a web-search or general recommendation agent.

### Not currently supported

The current data model and approved General Chat functions do not provide:

- Mortgage payment, loan affordability, down-payment, or rate calculations
- Property-tax liability or tax forecasting
- School ratings or school-assignment quality
- Insurance quotes or premium prediction
- Future price-appreciation or five-year value forecasting
- Current Internet market, news, traffic, or live route conditions
- A guarantee that a route is safe

### Evaluation fixture limits

The 53-example golden set covers only data populated by `eval/fixtures.py`:

- Six houses across Pittsburgh, Denver, Miami, and Austin
- Census tract and MSA populations for four named metros
- NRI overall, riverine flood, hurricane, and wildfire values
- Six sold-home records, including one deliberately invalid/non-market $1 sale
  and one pending-geocode record
- One unmatched MSA with placeholder code `Xplaceholder1` and no CBSA bridge row

The fixture does not populate crime incidents, BikePGH routes, or house snapshot
history. Those capabilities need a separate integration fixture using
representative loaded data rather than invented expected values.

### Recommended evaluation strategy

Keep the golden set as the deterministic fixture regression suite and maintain a
separate integration suite for:

1. Crime analysis across at least two loaded cities and multiple categories or months
2. Successful, no-route, and crime-avoidance BikePGH routing cases
3. Multiple price/status observations in `house_snapshots`
4. Several house documents and document types
5. A non-favorite fixture house so inventory scope is genuinely tested

---

## Bike routing

The app exposes `POST /api/bike/route` with:

```json
{"start":"Mount Washington","end":"Point State Park","city":"Pittsburgh, PA"}
```

Endpoints are place strings, not required map clicks. The service geocodes them
with Nominatim, then routes exclusively on the locally ingested BikePGH linework
stored in the `bike_routes` DuckDB table. No external road-routing engine is used.

### Endpoint precision

- Exact coordinates, street addresses, named places, landmarks, or neighborhoods with `city` context are supported.
- For Pittsburgh, ambiguous place names are auto-appended with `Pittsburgh, PA` when appropriate.
- Geocoding results are cached in the `geocode_cache` table.

### BikePGH layers used for routing

- Bike Lanes
- Bikeable Sidewalks
- Cautionary Bike Route
- On Street Bike Route
- Protected Bike Lanes
- Sharrows
- Trails

If the locally ingested network does not contain a continuous path between the snapped endpoints, the request returns **no route** rather than falling back to OSM street routing.

### Free/open services

- Nominatim / OpenStreetMap is used only for endpoint place-name geocoding.
- Shapely + DuckDB build and query the local BikePGH graph.

### Route semantics

- Distance is computed from the local BikePGH graph. Travel time is an estimate.
- Turn instructions refer to mapped BikePGH infrastructure rather than inventing street names.

---

## Developer scripts & tests

These utility scripts and tests are intended for debugging, data validation, evaluation, and regression testing. Run them from the repository root:

- `tests/test_guardrails.py`: Full test suite for multi-layer guardrails (Input prompt injection defense, Code AST sandboxing, read-only SQL validation, output grounding) plus microsecond latency benchmark. Usage: `python tests/test_guardrails.py`
- `tests/test_bikepg_h_visualization.py`: Regression checks verifying BikePGH visualization specs and styling matches ground truth. Usage: `python tests/test_bikepg_h_visualization.py`
- `run_eval.py`: Agent evaluation pipeline. Runs the golden set examples against real agents or mock fixture DB, scores them, and writes timestamped reports to `eval/reports/`. Usage: `python run_eval.py [--mock]`
- `debug_bike_route.py`: Lightweight checks for BikePGH city-key normalization and routing helpers. Usage: `python debug_bike_route.py`
- `debug_flood_query.py`: Step-by-step SQL debugger for the flood-risk query; runs CTEs, prints table counts, join diagnostics, and sample rows to pinpoint where the chain breaks. Usage: `python debug_flood_query.py`
- `debug_nri_columns.py`: Inspect the NRI shapefile's DBF column names and show NULL counts for hazard columns in `nri_tracts`. Usage: `python debug_nri_columns.py`
- `diagnose_msa.py`: Finds `X`-coded MSA rows that don't match `cbsa_counties`, suggests best CBSA candidates using a fuzzy normalizer, and can apply fixes with `--apply`. Usage: `python diagnose_msa.py [--apply]`



### Data page tests

`tests/` runs against throw-away DuckDB files (no `data/` needed): `test_catalog_unified.py` (seeding,
persistence, transactions, hot reload), `test_dataset_readers.py`, `test_onboarding_flow.py` (upload ->
approve -> live, address/spatial/geocode paths), `test_onboarding_api.py` (through `main.app`),
`test_agent_integration.py` (real General/House Chat code paths, vector index sync) and
`test_catalog_llm.py` (router, repair, fallback, self-test scoring against a mock OpenAI-compatible server).

---

### Map layer tests

`tests/test_map_layers.py` and `tests/test_map_layers_api.py` cover classification (including the two
no-code-change onboarding scenarios above), every generic data builder, the preserved bike-overlap and
crime-severity behavior, and — using an actual copy of `data/real_estate.duckdb` when this repository
includes one — the concrete layer inventory and fill-exclusivity table shown above, so that table stays
true as your data changes rather than drifting into documentation fiction.

### Commute tests

`tests/test_commute.py` and `tests/test_commute_api.py` run against one local mock that stands in for OSRM (car, bike,
foot), the Census and Nominatim geocoders and OpenTripPlanner, so they need no network or keys: request format
(lon,lat order, batching, `/table` fallback to `/route`), retries and outages, range caps, the drive factor, staleness,
the single-flight background job, transit parsing, the built-in catalog entries, the planner ranking houses by commute,
House Chat, `setup_data`, and the HTTP API. `tests/commute_helpers.py` holds the mock.

## Crime-aware bike routing
Crime-aware bike route requests are now executed deterministically in `agents/general_agent.py` when the user asks for a bike route that avoids crime/high-crime/dangerous areas. This prevents a local LLM from omitting the `find_bike_route` tool call. The resulting `find_bike_route` tool span is visible in observability, and its intermediate filtered BikePGH/crime visualization remains attached to the response.

Example request:

> Find a bike route from Mount Washington to Point State Park that avoids high-crime areas.


### Behavior

For bike-route questions that explicitly ask to avoid crime-dense/high-crime areas, the app now treats crime avoidance as a deterministic spatial filter rather than a language-model preference. The routing graph is cloned per request, the top crime-density cells (default: top 10% of occupied cells) are expanded by a small exclusion buffer, and BikePGH edges intersecting those exclusion areas are removed before Dijkstra routing. The intermediate map shows the relevant high-density crime cells, the BikePGH edges removed by the filter, and the BikePGH network that remains. A final route map is rendered only when a continuous route exists in that filtered graph.

The intermediate map is intentionally corridor-focused so the crime layer remains legible instead of painting the whole city with low-opacity cells. The visual is explanatory only; crime density is a heuristic and not a safety guarantee.


### Consistent hotspot scoring

The bike/crime intermediate visualization and the routing filter now share one crime-density model. Each occupied grid cell receives the same intensity score shown on the map: 75% normalized incident count + 25% normalized severity-weighted score. The routing hotspot percentile is applied to that same intensity score using a citywide baseline. The selected cells are buffered, then BikePGH geometry is evaluated at logical intersection-to-intersection segment granularity so one affected sub-edge does not create an inconsistent dark/orange/dark split within the same real-world segment.

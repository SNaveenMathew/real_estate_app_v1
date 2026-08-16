# 🏠 Real Estate Intelligence

A local, AI-powered map app for analyzing houses with FEMA National Risk Index data, Census demographics, Redfin listings, severity-weighted crime data, and an LLM chat interface — all running on your machine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (Leaflet Map + Chat UI)                            │
│   └─ Markers per house (click → sidebar)                    │
│   └─ Map Layer toggle: None / Crime / NRI                   │
│   └─ House Chat  (per-property, LangGraph ReAct agent)      │
│   └─ General Chat (cross-city, LangGraph ReAct agent)       │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP (FastAPI)
┌──────────────────────▼──────────────────────────────────────┐
│  FastAPI  (main.py)                                         │
│   ├─ /api/houses      GeoJSON of all houses                 │
│   ├─ /api/layers/crime      Severity-weighted heatmap grid   │
│   ├─ /api/layers/nri        NRI tract choropleth (GeoJSON)   │
│   ├─ /api/house/{id}/chat   House-specific agent            │
│   ├─ /api/chat              General agent                   │
│   └─ /api/house/{id}/photo  Photo upload                    │
└───────┬─────────────────────────────┬───────────────────────┘
        │                             │
┌───────▼────────┐         ┌──────────▼───────────┐
│  DuckDB        │         │  ChromaDB (vector)   │
│  ─ houses      │         │  ─ house_documents   │
│  ─ nri_tracts  │         │    (descriptions,    │
│  ─ census_*    │         │     photos, notes)   │
│  ─ sold_homes  │         └──────────────────────┘
│  ─ crime_incidents │
│  ─ cbsa_*      │
└───────┬────────┘         ┌──────────────────────┐
        │                  │  llama.cpp / llama-server  │
        └──────────────────│  ─ Gemma or other HF model  │
               │  ─ (see LLM setup below)   │
               └──────────────────────┘
```

---

## Prerequisites

- Python 3.13+
- `llama-server` (llama.cpp) installed and running, or Ollama as an alternative

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
└── data/
    ├── redfin/       ← drop Redfin CSV exports here
    ├── sold/         ← drop sold-homes CSV files here
    ├── nri/          ← NRI_Table_CensusTracts.csv
    ├── census/       ← DECENNIALPL2020_P1_tract.csv, _msa.csv, list1_2020.csv
    ├── crime/        ← one folder per city, e.g. crime/chicago/, crime/pittsburgh/
    └── shapefiles/   ← TIGER/Line tract shapefiles (optional but recommended)
```

### 2. Install dependencies

```bash
cd real_estate_app
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 3. LLM: run `llama-server` (default)

This repository is configured to use `llama-server` (from the `llama.cpp` project)
as the primary local LLM endpoint. A small example command that works with the
Gemma model (adjust paths/flags for your machine):

```bash
# Example: run llama-server serving a HF model (Gemma) with quantized caches
llama-server -hf DuoNeural/Gemma-4-26B-A4B-it-GGUF:Q3_K_M \
  -ngl 999 -c 28672 -fa on --cache-type-k q8_0 --cache-type-v q8_0
```

If you prefer Ollama, it remains supported as an alternative — see the
`config.py` and agent comments for how to switch. For Ollama, you would
pull and run:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve
```

Note: the project is configured to use Ollama's `nomic-embed-text` for embeddings by
default (see `.env.example`). If you run a pure `llama-server` stack, either run
Ollama alongside it for embeddings or update `db/vector_store.py` to use a
different embedding provider.

> **Lower VRAM?** For `llama-server` use a smaller HF model or reduce cache/memory flags.

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Mac/Linux
# Edit .env if needed (defaults work for local llama-server; Ollama optional)
```

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
3. The app expects columns including `lat`/`lon`, `address`, `city`, `state`, `zip`, `price`, `beds`, `baths`, `square feet`, `status`
4. If you already have Walk/Bike/Transit scores (from the WalkScore API), include them as `walk score`, `bike score`, `transit score`
5. Drop the CSV into **`data/redfin/`**

You can drop multiple CSV files (e.g. one per city).

### FEMA National Risk Index (highly recommended)

1. Go to: https://www.fema.gov/about/openfema/data-sets/national-risk-index-data
2. Download **NRI_Table_CensusTracts.csv** (census tract level, ~800 MB)
3. Place at: **`data/nri/NRI_Table_CensusTracts.csv`**

### Census Tract Populations

1. Go to https://data.census.gov
2. Search for table **DECENNIALPL2020.P1**
3. Filter: Geography → Census Tracts → All States → All Tracts
4. Download → Save as **`data/census/DECENNIALPL2020_P1_tract.csv`**

### MSA Populations

1. Same table **DECENNIALPL2020.P1** on data.census.gov
2. Filter: Geography → Metropolitan Statistical Areas → All MSAs
3. Download → Save as **`data/census/DECENNIALPL2020_P1_msa.csv`**

### CBSA County Crosswalk (needed for MSA-level NRI queries)

1. Go to: https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
2. Download **list1_2020.xls** → save/convert as **`data/census/list1_2020.csv`**

### Census Tract Shapefiles (optional, for fast tract-FIPS assignment)

Without shapefiles, the app uses the Census Geocoder API to resolve each house's census tract — this is slow (~0.15s per house). With shapefiles it's instant.

1. Go to: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
2. Select: Year = 2020 → Layer = Census Tracts → Select your states
3. Unzip into **`data/shapefiles/`** (one or multiple states)
4. Re-run: `python setup_data.py --resolve-tracts`

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

## Usage Guide

### Map

- **Colored markers** by status:
  - 🔵 Blue = Active
  - 🟡 Yellow = Pending
  - 🟣 Purple = Contingent
  - 🟢 Green = Pre-Market
- **Click any marker** to open the sidebar
- **Map Layer control** (top-right): toggle an optional overlay on top of the
  map — **Crime** (severity-weighted heatmap) or **NRI** (FEMA risk choropleth
  by census tract). Only one layer is shown at a time; select **None** to turn
  it off. Both re-fetch automatically as you pan/zoom, so they only ever load
  what's actually in view.

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
python setup_data.py                  # everything
```

The vector database (ChromaDB) grows automatically as you paste descriptions in chat.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Map shows no houses | Run `setup_data.py`, check `data/redfin/*.csv` exists |
| NRI data missing | Download & place `NRI_Table_CensusTracts.csv` in `data/nri/` |
| Tract FIPS not resolved | Run `python setup_data.py --resolve-tracts` |
| Chat says "I couldn't generate a response" | Make sure `llama-server` is running (or Ollama if you chose that stack) |
| Embedding errors | If using Ollama embeddings: pull `nomic-embed-text` with `ollama pull nomic-embed-text`. If using another embedding provider, configure accordingly. |
| Slow tract resolution | Add TIGER/Line shapefiles to `data/shapefiles/` |
| "Crime" layer is empty | Confirm `data/crime/<city>/` has files for a city your map view overlaps, then `python setup_data.py --only crime` |
| "NRI" layer is empty / shows a warning | It needs both tract *geometry* (from the NRI shapefile, or TIGER/Line shapefiles in `data/shapefiles/`) and tract *attributes* (`python setup_data.py --only nri`) — the layer's warning message says which is missing |

---

## Adding New Data Sets

The architecture is designed to grow. To add a new data set:

1. Add a new `load_xyz()` function in `services/data_loader.py`
2. Add a new table in `db/duckdb_store.py` → `_ensure_schema()`
3. Add ONE `TableMeta` entry (description + notes on any non-obvious columns)
   in `db/schema_catalog.py`. If it joins to an existing table, add one
   `Relationship` entry alongside it. You do NOT need to write a new agent
   tool or teach the LLM a new SQL pattern — `query_database` plus the schema
   catalog is enough for the agent to work out how to query it, including
   joins to other tables.
4. Call `setup_data.py` to load it

`db/schema_catalog.py` is the single source of truth for what the agent knows
about the data: it combines live introspection of the running database
(so column names/types can't go stale) with curated notes for things no
amount of introspection can tell you — which columns are reliably populated,
which joins need a non-obvious expression, which tables need a default filter
to be meaningful. `check_data_availability`, `get_database_schema`,
`setup_data.py`'s summary, and the response validator's fallback message all
read from it, so there's nothing else to keep in sync when you add a table.

---

## Project Structure

```
main.py               FastAPI app + all HTTP endpoints
config.py             All settings (paths, models, ports)
setup_data.py         One-time data loader script
run_eval.py            Agent evaluation pipeline entry point

agents/
  tools.py            LangChain tools (SQL, vector search, price estimation)
  house_agent.py      Per-house ReAct agent (LangGraph)
  general_agent.py    General ReAct agent (LangGraph)
  response_validator.py  Post-hoc check that replies are grounded in real tool output

db/
  duckdb_store.py     All SQL queries and schema management
  schema_catalog.py   Metadata layer — table/column notes + join graph for the agent
  vector_store.py     ChromaDB — embed, store, search text documents

services/
  data_loader.py      Parsers for Redfin, NRI, Census, Sold, Crime
  crime_sources.py    Per-city crime file parsers (one class per city)
  crime_taxonomy.py   Standardized crime categories + severity weights
  layers.py           Viewport-scoped queries behind the Crime/NRI map layers
  geo_utils.py         Census tract FIPS assignment (shapefile or API)

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
  style.css            All styles
  app.js               Frontend logic (incl. the Crime/NRI layer toggle)

data/
  redfin/             Drop Redfin CSVs here
  sold/               Drop sold-homes CSVs here
  nri/                FEMA NRI CSV or shapefile
  census/             Census P1 tables + CBSA crosswalk
  crime/<city>/       Drop each city's raw crime export here
  shapefiles/         TIGER/Line tract shapefiles (optional)
```

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

**Bike routing**

The app exposes `POST /api/bike/route` with:

```
{"start":"Mount Washington","end":"Point State Park","city":"Pittsburgh, PA"}
```

Endpoints are place strings, not required map clicks. The service geocodes them with Nominatim, then routes exclusively on the locally ingested BikePGH linework stored in the `bike_routes` DuckDB table. No external road-routing engine is used.

### Endpoint precision

- Exact coordinates, street addresses, named places, landmarks, or neighborhoods with `city` context are supported.
- For Pittsburgh, ambiguous place names are auto-appended with `Pittsburgh, PA` when appropriate. Geocoding results are cached in `geocode_cache`.

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

**Developer scripts**

These utility scripts are intended for debugging, data validation, and evaluation. Run them from the repository root.

- `debug_bike_route.py`: Lightweight checks for BikePGH city-key normalization and routing helpers. Usage: `python debug_bike_route.py`.
- `debug_flood_query.py`: Step-by-step SQL debugger for the flood-risk query; runs CTEs, prints table counts, join diagnostics, and sample rows to pinpoint where the chain breaks. Usage: `python debug_flood_query.py`.
- `debug_nri_columns.py`: Inspect the NRI shapefile's DBF column names and show NULL counts for hazard columns in `nri_tracts`. Attempts to read the shapefile with GeoPandas when available. Usage: `python debug_nri_columns.py`.
- `diagnose_msa.py`: Finds `X`-coded MSA rows that don't match `cbsa_counties`, suggests best CBSA candidates using a fuzzy normalizer, and can apply fixes with `--apply`. Usage: `python diagnose_msa.py [--apply]`.
- `run_eval.py`: Agent evaluation pipeline (already described above). Runs the golden set examples, scores them, and writes timestamped reports to `eval/reports/`.

If you prefer the previous standalone bike-routing README, its content has been folded here; `README_BIKE_ROUTING.md` was consolidated into this file.
```

Each golden example is one of:
- **structured** — has a definite right answer (a count, a ranking, a set of
  cities). Scored by exact comparison against the reply text: as an ordered
  sequence when order matters, as a set when it doesn't. No LLM involved in
  grading these.
- **free_text** — open-ended (a tradeoff, an explanation). Scored by an LLM
  judge against a rubric (`eval/judge.py`). Configure the judge model via
  `judge_llama_server_base_url` / `judge_llama_server_model` in `.env` — it
  defaults to the same server as the agent under test, but an independent or
  stronger judge is stronger evidence than a model grading itself.

Reports land in `eval/reports/` as both JSON (for tooling) and Markdown (for
reading). Exit code is non-zero if anything failed or errored, so this is
safe to wire into CI. `eval/tests/test_scoring.py` unit-tests the assert-equal
scorer itself against hand-crafted replies — run it directly any time you
change `eval/scoring.py`.

To add a golden example: add fixture data to `eval/fixtures.py` if needed
(prefer deriving expected values from the fixture data programmatically, the
way the existing examples do, over hand-typing a number), then add one
`GoldenExample` to `eval/golden_set.py`.

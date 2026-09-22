"""Central configuration — edit .env or override here."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── llama-server ──────────────────────────────────────────────────────────────
    llama_server_base_url: str = "http://127.0.0.1:8080/v1"
    llama_server_model: str = "DuoNeural/Gemma-4-26B-A4B-it-GGUF:Q3_K_M"          # options: DuoNeural/Gemma-4-26B-A4B-it-GGUF:Q3_K_M, unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL, llama3.2, mistral, etc.

    # ── LLM generation limits ───────────────────────────────────────────────
    # Defensive bounds applied to every local LLM call (general agent, house
    # agent, SQL Code Agent). Without these, a call that never emits a
    # recognized stop token runs all the way out to the server's context
    # limit (-c) instead of stopping on its own. This isn't hypothetical —
    # some llama-server GGUF conversions (this Gemma-4-26B-A4B-it quant
    # included; check the server's startup log for "removing '</s>' token
    # from EOG list") drop the model's normal end-of-turn token from the
    # server's own stop-token set, so generation only stops where WE tell it
    # to. `stop` matches on the decoded text itself, so it works even when
    # the server's internal EOG/token-id classification is wrong. `max_tokens`
    # is the hard backstop for when the model never emits a stop sequence at
    # all. Both matter beyond just the one slow reply: with a unified/shared
    # KV cache (`kv_unified` in the server log), one runaway call can consume
    # nearly the whole cache and starve every other concurrent chat, forcing
    # them to reprocess their prompts from scratch too.
    # Round 2 finding: 700 turned out too tight. Every single Code Agent call
    # for a 3-table join with a 2-hazard average hit EXACTLY 700 decoded
    # tokens, never less, across 7 retries in one turn — never a natural,
    # earlier stop. So `stop` isn't reliably firing for this call shape (a
    # bare, non-tool-calling text completion) on this model/template combo;
    # `max_tokens` is doing 100% of the work of bounding it, which means it
    # also has to be generous enough to let a legitimately complex query
    # actually finish. Paired with a tightened "SQL only" instruction in
    # agents/tools.py, which cuts down how much of that budget goes to
    # non-SQL text in the first place.
    agent_max_tokens: int = 2048        # ReAct agent turn: reasoning + tool call + final answer
    code_agent_max_tokens: int = 1500   # SQL Code Agent: raised from 700 — see "Round 2" above
    llm_request_timeout: float = 120.0  # seconds, per HTTP request to llama-server

    # ── Observability / Phoenix ─────────────────────────────────────────────
    # Local Phoenix is open source and runs entirely on the developer machine.
    phoenix_enabled: bool = True
    phoenix_collector_endpoint: str = "http://127.0.0.1:6006/v1/traces"
    phoenix_protocol: str = "http/protobuf"
    phoenix_ui_url: str = "http://127.0.0.1:6006"
    phoenix_project_name: str = "real-estate-general-chat"

    # ── Ollama embeddings ───────────────────────────────────────────────────
    ollama_base_url: str = "http://127.0.0.1:11434"

    # ── Embedding model ─────────────────────────────────────────────────────
    ollama_embed_model: str = "nomic-embed-text"  # pull once: ollama pull nomic-embed-text

    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir: Path = BASE_DIR / "data"
    # NRI data — provide EITHER the shapefile OR the CSV (shapefile preferred:
    # it contains both risk attributes AND tract geometries in one file).
    # Download from: https://www.fema.gov/about/openfema/data-sets/national-risk-index-data
    # Shapefile: NRI_Shapefile_CensusTracts.zip → extract into data/nri/
    # CSV fallback: NRI_Table_CensusTracts.csv → place in data/nri/
    nri_shp: Path = BASE_DIR / "data" / "nri" / "NRI_CensusTracts_Prod.shp"
    # nri_csv: Path = BASE_DIR / "data" / "nri" / "NRI_Table_CensusTracts.csv"
    census_tract_csv: Path = BASE_DIR / "data" / "census" / "DECENNIALPL2020.P1-Data.csv"
    census_msa_csv: Path = BASE_DIR / "data" / "census" / "DECENNIALPL2020.P1-2026-03-25T232220.csv"
    # MSA → county crosswalk from Census Bureau
    # https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
    # cbsa_csv: Path = BASE_DIR / "data" / "census" / "list1_2020.csv"
    cbsa_xlsx: Path = BASE_DIR / "data" / "census" / "list1_2023.xlsx"
    redfin_dir: Path = BASE_DIR / "data" / "redfin"   # drop any number of Redfin CSVs here
    sold_dir: Path = BASE_DIR / "data" / "sold"
    shapefile_dir: Path = BASE_DIR / "data" / "shapefiles"
    uploads_dir: Path = BASE_DIR / "uploads"

    # ── Data onboarding (uploaded datasets; see services/dataset_onboarding.py) ──
    onboarding_max_upload_mb: int = 250
    onboarding_sample_rows: int = 8
    onboarding_geocode_row_cap: int = 2000   # rows sent to the Census geocoder per run
    onboarding_api_row_cap: int = 500        # rows allowed through the per-point Census API when no tract geometry is loaded

    # ── Catalog LLM router (optional; drafts descriptions/synonyms only; see services/catalog_llm.py) ──
    # Tier "draft" -> CATALOG_DRAFT_*, tier "judge" -> CATALOG_JUDGE_*; empty values fall back to the chat
    # llama-server above. Nothing here is required: without a model the Data page uses rule-based defaults.
    catalog_llm_enabled: bool = True
    catalog_llm_judge_enabled: bool = False
    catalog_draft_base_url: str = ""         # e.g. http://127.0.0.1:8081/v1 (a separate small llama-server)
    catalog_draft_model: str = ""
    catalog_judge_base_url: str = ""
    catalog_judge_model: str = ""
    catalog_llm_timeout: float = 90.0
    catalog_llm_max_tokens: int = 1600
    catalog_llm_max_repairs: int = 1
    catalog_llm_disable_thinking: bool = True

    # ── Commute (free/open services only; see services/commute.py) ──────────
    work_address: str = ""                      # optional default; the Commute tab saves the real value in the DB
    commute_modes: str = "drive,bike,walk"      # 'transit' is added automatically when OTP_BASE_URL is set
    osrm_drive_url: str = "https://router.project-osrm.org"
    osrm_bike_url: str = "https://routing.openstreetmap.de/routed-bike"
    osrm_foot_url: str = "https://routing.openstreetmap.de/routed-foot"
    osrm_min_interval_s: float = 1.0            # politeness delay between requests to a public server (0 for your own)
    osrm_timeout_s: float = 25.0
    osrm_table_max: int = 90                    # sources per /table request (the public demo allows 100 coordinates)
    commute_drive_factor: float = 1.0           # multiply free-flow drive time (e.g. 1.3 for rush hour); 1.0 = free-flow
    otp_base_url: str = ""                      # self-hosted OpenTripPlanner; empty = no transit
    otp_api: str = "graphql"                    # graphql | rest
    commute_depart_time: str = "08:00"
    census_geocoder_url: str = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    nominatim_base_url: str = "https://nominatim.openstreetmap.org"
    nominatim_user_agent: str = "RealEstateIntelligence/1.0 (local personal app)"
    nominatim_email: str = ""                   # optional contact, per Nominatim's usage policy
    nominatim_countrycodes: str = "us"          # empty = worldwide

    # ── DuckDB ───────────────────────────────────────────────────────────────
    duckdb_path: Path = BASE_DIR / "data" / "real_estate.duckdb"

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_dir: Path = BASE_DIR / "data" / "chroma"

    # ── Evaluation: judge model ─────────────────────────────────────────────
    # Defaults to the same server/model as the agent under test. Point this at
    # a different (ideally independent, or stronger) model if you have one
    # available — a model grading its own answers is weaker evidence than an
    # independent judge.
    judge_llama_server_base_url: str = "http://127.0.0.1:8080/v1"
    judge_llama_server_model: str = "DuoNeural/Gemma-4-26B-A4B-it-GGUF:Q3_K_M"

    # ── Evaluation: fixtures & reports ──────────────────────────────────────
    eval_fixture_duckdb_path: Path = BASE_DIR / "eval" / "fixture_data" / "eval_fixture.duckdb"
    eval_reports_dir: Path = BASE_DIR / "eval" / "reports"
    eval_traces_dir: Path = BASE_DIR / "eval" / "traces"

    # ── App ──────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    map_default_lat: float = 37.0902
    map_default_lon: float = -95.7129
    map_default_zoom: int = 5


settings = Settings()


# Shared stop sequences for every local LLM call — see the "LLM generation
# limits" comment above. A plain constant, not a Settings field: these are
# about how we talk to this specific local model/server combo, not something
# that needs per-deployment .env overrides.
LLM_STOP_SEQUENCES: list[str] = ["<end_of_turn>", "</s>"]

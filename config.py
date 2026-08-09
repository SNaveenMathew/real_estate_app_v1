"""Central configuration — edit .env or override here."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"          # swap for llama3.2, mistral, etc.
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

    # ── DuckDB ───────────────────────────────────────────────────────────────
    duckdb_path: Path = BASE_DIR / "data" / "real_estate.duckdb"

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_dir: Path = BASE_DIR / "data" / "chroma"

    # ── App ──────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    map_default_lat: float = 37.0902
    map_default_lon: float = -95.7129
    map_default_zoom: int = 5


settings = Settings()

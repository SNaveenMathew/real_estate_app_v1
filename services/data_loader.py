"""
Data loader: reads raw files and populates DuckDB tables.
Run via setup_data.py — only needed when data files change.

The loaders live in ``services/loaders/`` (one module per source family); this
module is the compatibility facade. Everything that used to be defined here —
including the private helpers other modules import (``_clean_numeric``,
``_REDFIN_COL_MAP``, ``_detect_parser``, ``_crime_incident_id``, ...) — is
re-exported, so call sites did not change.
"""
import warnings

warnings.filterwarnings("ignore")

from config import settings  # noqa: F401  (callers and tests patch data_loader.settings)
import db.duckdb_store as store  # noqa: F401

from services.loaders import *  # noqa: F401,F403
from services.loaders import __all__ as _loader_names  # noqa: F401

__all__ = list(_loader_names)

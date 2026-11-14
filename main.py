"""
FastAPI backend for the Real Estate Analysis App.
Run: uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from config import settings
import db.duckdb_store as store
from observability import (
    initialize_observability,
    ensure_phoenix_server,
    stop_phoenix_server,
    phoenix_enabled,
)

# Segmented API routers
from api import (
    houses_router,
    chat_router,
    bike_router,
    system_router,
    commute_router,
    data_sources_router,
    map_layers_router,
    onboarding_router,
)
from api.common import (
    safe_value as _safe,
    safe_dict as _safe_dict,
    ChatRequest,
    DocumentRequest,
    BikeRouteRequest,
)
from api.chat import turn_failure as _turn_failure

logger = logging.getLogger(__name__)


# ── Phoenix process lifecycle ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_phoenix_server()
    initialize_observability()
    try:
        yield
    finally:
        store.close()
        stop_phoenix_server()


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Real Estate Analysis API",
    description="Local, AI-powered analytical real estate intelligence backend service.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
settings.uploads_dir.mkdir(parents=True, exist_ok=True)

# Register modular service routers
app.include_router(houses_router)
app.include_router(chat_router)
app.include_router(bike_router)
app.include_router(system_router)
app.include_router(commute_router)
app.include_router(data_sources_router)
app.include_router(map_layers_router)
app.include_router(onboarding_router)

# Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app(), name="metrics")


# ── Optional static UI serving (for standalone local dev runs) ───────────────
# In containerized/service deployments, the dedicated Frontend service (Nginx)
# serves static files and proxies /api traffic. Mounting here allows running
# `python main.py` or `uvicorn main:app` directly without a separate web server.
static_dir = Path("static")
if static_dir.exists() and static_dir.is_dir():
    app.mount("/static", StaticFiles(directory="static"), name="static")

    def _static_version() -> str:
        """MD5 of assets — changes whenever any of them changes."""
        h = hashlib.md5()
        for name in ("app.js", "commute.js", "layers.js", "style.css"):
            p = static_dir / name
            if p.exists():
                h.update(p.read_bytes())
        return h.hexdigest()[:8]

    _STATIC_VER = _static_version()

    @app.get("/data", response_class=HTMLResponse)
    async def data_page():
        """The Data page: schema map, dataset upload, and review of proposed catalog changes."""
        html = (static_dir / "data.html").read_text(encoding="utf-8")
        ver = hashlib.sha1(b"".join((static_dir / n).read_bytes() for n in ("data.js", "data.css"))).hexdigest()[:10]
        html = re.sub(r'(/static/[\w./-]+\.(?:js|css))(\?v=\w+)?', lambda m: f"{m.group(1)}?v={ver}", html)
        return HTMLResponse(content=html, headers={"Cache-Control": "no-cache"})

    @app.get("/", response_class=HTMLResponse)
    async def root():
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        client_config = json.dumps({"cartoApiKey": settings.carto_api_key}).replace("</", "<\\/")
        html = html.replace("</head>", f"<script>window.APP_CONFIG={client_config};</script>\n</head>", 1)
        html = re.sub(
            r'(/static/[^"\']+\.(?:js|css))(\?v=[^"\']*)?',
            lambda m: f'{m.group(1)}?v={_STATIC_VER}',
            html,
        )
        return HTMLResponse(content=html, headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=True)
